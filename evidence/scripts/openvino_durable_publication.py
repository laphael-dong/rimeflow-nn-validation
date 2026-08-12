#!/usr/bin/env python3
"""OpenVINO tracked evidence 的双文件 durable publication 与崩溃恢复。"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = 1
JOURNAL_NAME = "journal.json"
FaultHook = Callable[[str, dict[str, object]], None]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _fault(hook: FaultHook | None, event: str, **context: object) -> None:
    if hook is not None:
        hook(event, context)


def fsync_directory(path: Path, *, fault: FaultHook | None = None, event: str = "directory.fsync") -> None:
    _fault(fault, event, path=str(path))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_durable(path: Path, payload: bytes, *, fault: FaultHook | None, event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _fault(fault, f"{event}.write", path=str(path))
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        _fault(fault, f"{event}.fsync", path=str(path))
        os.fsync(stream.fileno())


def _replace(source: Path, target: Path, *, fault: FaultHook | None, event: str) -> None:
    _fault(fault, event, source=str(source), target=str(target))
    os.replace(source, target)


def _unlink(path: Path, *, fault: FaultHook | None, event: str) -> None:
    if path.exists():
        _fault(fault, event, path=str(path))
        path.unlink()


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"exists": False, "bytes": None, "sha256": None}
    payload = path.read_bytes()
    return {"exists": True, "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _matches(path: Path, identity: dict[str, object]) -> bool:
    return _identity(path) == {
        "exists": identity["exists"],
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
    }


def _journal_path(transaction_dir: Path) -> Path:
    return transaction_dir / JOURNAL_NAME


def _write_journal(transaction_dir: Path, journal: dict[str, object], *, fault: FaultHook | None) -> None:
    transaction_dir.mkdir(parents=True, exist_ok=True)
    journal_path = _journal_path(transaction_dir)
    descriptor, name = tempfile.mkstemp(prefix=".journal-", dir=transaction_dir)
    temporary = Path(name)
    try:
        _fault(fault, "journal.write", path=str(temporary), phase=journal["phase"])
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_stable_bytes(journal))
            stream.flush()
            _fault(fault, "journal.fsync", path=str(temporary), phase=journal["phase"])
            os.fsync(stream.fileno())
        _replace(temporary, journal_path, fault=fault, event="journal.replace")
        fsync_directory(transaction_dir, fault=fault, event="journal.directory_fsync")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_journal(transaction_dir: Path) -> dict[str, object] | None:
    path = _journal_path(transaction_dir)
    if not path.is_file():
        return None
    journal = json.loads(path.read_text())
    if journal.get("schemaVersion") != SCHEMA_VERSION or journal.get("phase") not in {"prepared", "publishing", "committed"}:
        raise RuntimeError("unsupported or corrupt OpenVINO publication journal")
    if not isinstance(journal.get("targets"), list) or len(journal["targets"]) != 2:
        raise RuntimeError("OpenVINO publication journal must contain two targets")
    return journal


def _validate_journal_paths(
    journal: dict[str, object], expected_targets: set[Path] | None = None
) -> list[dict[str, object]]:
    targets = journal["targets"]
    seen: set[Path] = set()
    for entry in targets:
        target = Path(entry["target"])
        staging = Path(entry["staging"])
        backup = Path(entry["backup"]) if entry["backup"] is not None else None
        recovery = Path(entry["recovery"])
        if target != target.resolve() or staging != staging.resolve() or recovery != recovery.resolve() or (backup is not None and backup != backup.resolve()):
            raise RuntimeError("OpenVINO publication journal contains non-canonical path")
        if target in seen:
            raise RuntimeError("OpenVINO publication journal repeats a target")
        seen.add(target)
    if expected_targets is not None and seen != {path.resolve() for path in expected_targets}:
        raise RuntimeError("OpenVINO publication journal target allowlist mismatch")
    return targets


def _verify_generation(targets: list[dict[str, object]], generation: str) -> None:
    for entry in targets:
        expected = entry[generation]
        if not _matches(Path(entry["target"]), expected):
            raise RuntimeError(f"{generation} generation identity mismatch: {entry['target']}")


def _restore_old(entry: dict[str, object], *, fault: FaultHook | None) -> None:
    target = Path(entry["target"])
    old = entry["old"]
    if _matches(target, old):
        return
    if not old["exists"]:
        _unlink(target, fault=fault, event="recovery.target_unlink")
        return
    backup = Path(entry["backup"])
    if not _matches(backup, old):
        raise RuntimeError(f"old generation backup missing or drifted: {backup}")
    recovery = Path(entry["recovery"])
    if recovery.exists() and not _matches(recovery, old):
        raise RuntimeError(f"old generation recovery copy drifted: {recovery}")
    if not recovery.exists():
        _write_durable(recovery, backup.read_bytes(), fault=fault, event="recovery.file")
        fsync_directory(recovery.parent, fault=fault, event="recovery.material_directory_fsync")
    _replace(recovery, target, fault=fault, event="recovery.replace")


def _finish_new(entry: dict[str, object], *, fault: FaultHook | None) -> None:
    target = Path(entry["target"])
    new = entry["new"]
    if _matches(target, new):
        return
    staging = Path(entry["staging"])
    if not _matches(staging, new):
        raise RuntimeError(f"new generation staging missing or drifted: {staging}")
    _replace(staging, target, fault=fault, event="recovery.replace")


def _cleanup(transaction_dir: Path, journal: dict[str, object], *, fault: FaultHook | None) -> None:
    targets = _validate_journal_paths(journal)
    _fault(fault, "cleanup.begin", phase=journal["phase"])
    cleanup_errors: list[BaseException] = []
    for entry in targets:
        for key in ("staging", "backup", "recovery"):
            value = entry[key]
            if value is None:
                continue
            try:
                _unlink(Path(value), fault=fault, event="cleanup.unlink")
            except BaseException as error:
                cleanup_errors.append(error)
    if cleanup_errors:
        raise ExceptionGroup("OpenVINO publication material cleanup failed", cleanup_errors)
    directories = {Path(entry["target"]).parent for entry in targets}
    directories.add(transaction_dir)
    for directory in directories:
        fsync_directory(directory, fault=fault, event="cleanup.directory_fsync")
    _unlink(_journal_path(transaction_dir), fault=fault, event="cleanup.journal_unlink")
    fsync_directory(transaction_dir, fault=fault, event="cleanup.final_directory_fsync")


def recover(
    transaction_dir: Path,
    *,
    expected_targets: set[Path] | None = None,
    fault: FaultHook | None = None,
) -> dict[str, object]:
    """将未完成事务恢复为完整 old 或完整 new generation。"""
    transaction_dir = transaction_dir.resolve()
    journal = _load_journal(transaction_dir)
    if journal is None:
        return {"generation": None, "recovered": False}
    targets = _validate_journal_paths(journal, expected_targets)
    generation = "new" if journal["phase"] == "committed" else "old"
    errors: list[BaseException] = []
    for entry in targets:
        try:
            if generation == "new":
                _finish_new(entry, fault=fault)
            else:
                _restore_old(entry, fault=fault)
            fsync_directory(Path(entry["target"]).parent, fault=fault, event="recovery.directory_fsync")
        except BaseException as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup(f"OpenVINO {generation} generation recovery failed", errors)
    _verify_generation(targets, generation)
    for directory in {Path(entry["target"]).parent for entry in targets}:
        fsync_directory(directory, fault=fault, event="recovery.confirm_directory_fsync")
    _cleanup(transaction_dir, journal, fault=fault)
    return {"generation": generation, "recovered": True, "transactionId": journal["transactionId"]}


def publish(payloads: dict[Path, bytes], transaction_dir: Path, *, fault: FaultHook | None = None) -> dict[str, object]:
    """发布两个文件；跨目录一致性由 durable journal 与 recover() 保证。"""
    if len(payloads) != 2:
        raise ValueError("OpenVINO record publication requires exactly two targets")
    if len({path.resolve().parent for path in payloads}) != 2:
        raise ValueError("OpenVINO record targets must use two distinct parent directories")
    transaction_dir = transaction_dir.resolve()
    expected_targets = {path.resolve() for path in payloads}
    recover(transaction_dir, expected_targets=expected_targets, fault=fault)
    transaction_id = uuid.uuid4().hex
    targets: list[dict[str, object]] = []
    try:
        for sequence, (raw_target, payload) in enumerate(payloads.items(), 1):
            target = raw_target.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            old = _identity(target)
            staging = target.parent / f".{target.name}.openvino-stage-{transaction_id}"
            backup = transaction_dir / f"{sequence}-old-{transaction_id}.backup" if old["exists"] else None
            recovery_copy = transaction_dir / f"{sequence}-old-{transaction_id}.recovery"
            entry = {
                "backup": str(backup) if backup is not None else None,
                "new": {"exists": True, "bytes": len(payload), "sha256": _sha256_bytes(payload)},
                "old": old,
                "recovery": str(recovery_copy),
                "staging": str(staging),
                "target": str(target),
            }
            targets.append(entry)
            _write_durable(staging, payload, fault=fault, event="staging.file")
            _fault(fault, "staging.ready", path=str(staging), target=str(target))
            if _identity(staging) != {"exists": True, "bytes": len(payload), "sha256": _sha256_bytes(payload)}:
                raise RuntimeError(f"staging bytes/SHA-256 mismatch: {target}")
            if backup is not None:
                _write_durable(backup, target.read_bytes(), fault=fault, event="backup.file")
        for directory in {Path(entry["staging"]).parent for entry in targets} | {transaction_dir}:
            fsync_directory(directory, fault=fault, event="prepare.directory_fsync")
        generation_payload = "\n".join(f"{entry['target']}:{entry['new']['sha256']}" for entry in targets).encode()
        journal = {
            "intendedGeneration": _sha256_bytes(generation_payload),
            "phase": "prepared",
            "schemaVersion": SCHEMA_VERSION,
            "targets": targets,
            "transactionId": transaction_id,
        }
        _write_journal(transaction_dir, journal, fault=fault)
        journal["phase"] = "publishing"
        _write_journal(transaction_dir, journal, fault=fault)
        for entry in targets:
            _replace(Path(entry["staging"]), Path(entry["target"]), fault=fault, event="publish.replace")
            _fault(fault, "publish.replaced", target=entry["target"])
            fsync_directory(Path(entry["target"]).parent, fault=fault, event="publish.directory_fsync")
        _verify_generation(targets, "new")
        journal["phase"] = "committed"
        _write_journal(transaction_dir, journal, fault=fault)
        _fault(fault, "journal.committed", transactionId=transaction_id)
        _cleanup(transaction_dir, journal, fault=fault)
        return {"generation": journal["intendedGeneration"], "transactionId": transaction_id}
    except BaseException as original:
        journal_exists = _journal_path(transaction_dir).is_file()
        if not journal_exists:
            cleanup_errors: list[BaseException] = []
            for entry in targets:
                for key in ("staging", "backup", "recovery"):
                    value = entry[key]
                    if value is None:
                        continue
                    try:
                        _unlink(Path(value), fault=fault, event="prejournal.cleanup_unlink")
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            for directory in {Path(entry["target"]).parent for entry in targets} | {transaction_dir}:
                try:
                    fsync_directory(directory, fault=fault, event="prejournal.cleanup_directory_fsync")
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise ExceptionGroup("OpenVINO publication failed before journaling and cleanup was incomplete", [original, *cleanup_errors])
            raise
        try:
            recover(transaction_dir, expected_targets=expected_targets, fault=fault)
        except BaseException as recovery_error:
            raise ExceptionGroup("OpenVINO publication failed and recovery was incomplete", [original, recovery_error])
        raise


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[], type=Path)
    args = parser.parse_args()
    expected_targets = {path.resolve() for path in args.target} if args.target else None
    print(json.dumps(recover(args.recover, expected_targets=expected_targets), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
