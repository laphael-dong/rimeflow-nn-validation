#!/usr/bin/env python3
"""用真实临时文件系统验证 OpenVINO 双文件 record 发布事务。"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_openvino_replay.py")
SPEC = importlib.util.spec_from_file_location("run_openvino_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def snapshot(paths: tuple[Path, Path]) -> tuple[bytes | None, bytes | None]:
    return tuple(path.read_bytes() if path.exists() else None for path in paths)


def leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if ".recording-" in path.name or ".backup-" in path.name or ".recovery-" in path.name)


def expect_failure(name: str, operation, paths: tuple[Path, Path], expected: tuple[bytes | None, bytes | None]) -> str:
    try:
        operation()
    except BaseException:
        if snapshot(paths) != expected:
            raise AssertionError(f"{name}: target bytes/existence were not rolled back")
        if leftovers(paths[0].parent):
            raise AssertionError(f"{name}: staging/backup residue remained")
        return name
    raise AssertionError(f"{name}: publication unexpectedly succeeded")


def main() -> None:
    cases = []
    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-publish-") as temporary:
        directory = Path(temporary)
        manifest = directory / "openvino-ep-manifest.json"
        report = directory / "openvino-ep-report.json"
        paths = (manifest, report)
        payloads = {manifest: b"new manifest\n", report: b"new report\n"}

        def reset(manifest_bytes=b"old manifest\n", report_bytes=b"old report\n"):
            for path, value in zip(paths, (manifest_bytes, report_bytes)):
                path.unlink(missing_ok=True)
                if value is not None:
                    path.write_bytes(value)
            if leftovers(directory):
                raise AssertionError("test setup residue")

        def fail_replace(number: int):
            calls = 0

            def replace(source, target):
                nonlocal calls
                calls += 1
                if calls == number:
                    raise OSError(f"injected target replace failure {number}")
                os.replace(source, target)

            return replace

        reset()
        expected = snapshot(paths)
        cases.append(expect_failure("first target replace failure rolls back both old files", lambda: MODULE.publish(payloads, replace_target=fail_replace(1)), paths, expected))

        reset()
        expected = snapshot(paths)
        cases.append(expect_failure("second target replace failure rolls back both old files", lambda: MODULE.publish(payloads, replace_target=fail_replace(2)), paths, expected))

        reset(None, b"old report only\n")
        expected = snapshot(paths)
        cases.append(expect_failure("missing manifest and existing report recover existence and bytes", lambda: MODULE.publish(payloads, replace_target=fail_replace(2)), paths, expected))

        reset(b"old manifest only\n", None)
        expected = snapshot(paths)
        cases.append(expect_failure("existing manifest and missing report recover existence and bytes", lambda: MODULE.publish(payloads, replace_target=fail_replace(2)), paths, expected))

        reset()
        expected = snapshot(paths)

        def corrupt_first_stage(target: Path, staged: Path) -> None:
            if target == manifest:
                staged.write_bytes(b"corrupt staging\n")

        cases.append(expect_failure("staging digest mismatch preserves both targets", lambda: MODULE.publish(payloads, stage_mutator=corrupt_first_stage), paths, expected))

        reset()
        expected = snapshot(paths)
        sync_calls = 0

        def fail_first_directory_sync(path: Path) -> None:
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 1:
                raise OSError("injected directory fsync failure")
            MODULE.fsync_directory(path)

        cases.append(expect_failure("directory fsync failure rolls back both targets", lambda: MODULE.publish(payloads, sync_directory=fail_first_directory_sync), paths, expected))

        reset()
        expected = snapshot(paths)
        sync_calls = 0

        def fail_cleanup_directory_sync(path: Path) -> None:
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 5:
                raise OSError("injected cleanup directory fsync failure")
            MODULE.fsync_directory(path)

        cases.append(expect_failure("cleanup directory fsync failure restores both targets", lambda: MODULE.publish(payloads, sync_directory=fail_cleanup_directory_sync), paths, expected))

        reset()
        MODULE.publish(payloads)
        if snapshot(paths) != (payloads[manifest], payloads[report]):
            raise AssertionError("successful publication did not update both targets")
        if leftovers(directory):
            raise AssertionError("successful publication left staging/backup residue")
        cases.append("successful publication updates both targets without residue")

    print(json.dumps({"ok": True, "filesystemCases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
