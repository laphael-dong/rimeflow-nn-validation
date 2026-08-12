#!/usr/bin/env python3
"""双目录 OpenVINO durable publication 的真实文件系统故障注入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from openvino_durable_publication import JOURNAL_NAME, publish, recover


OLD_MANIFEST = b'{"generation":"old","kind":"manifest"}\n'
OLD_REPORT = b'{"generation":"old","kind":"report"}\n'
NEW_MANIFEST = b'{"generation":"new","kind":"manifest"}\n'
NEW_REPORT = b'{"generation":"new","kind":"report"}\n'


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"exists": False, "bytes": None, "sha256": None}
    payload = path.read_bytes()
    return {"exists": True, "bytes": len(payload), "sha256": digest(payload)}


def expected(payload: bytes | None) -> dict[str, object]:
    return {"exists": payload is not None, "bytes": None if payload is None else len(payload), "sha256": None if payload is None else digest(payload)}


def flatten(error: BaseException) -> list[str]:
    messages = [f"{type(error).__name__}: {error}"]
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            messages.extend(flatten(child))
    return messages


class Fail:
    def __init__(self, event: str, occurrence: int = 1, *, crash: bool = False):
        self.event = event
        self.occurrence = occurrence
        self.count = 0
        self.crash = crash

    def __call__(self, event: str, _context: dict[str, object]) -> None:
        if event != self.event:
            return
        self.count += 1
        if self.count == self.occurrence:
            if self.crash:
                os._exit(97)
            raise OSError(f"injected {event} occurrence {self.occurrence}")


def topology(base: Path, manifest_old: bytes | None = OLD_MANIFEST, report_old: bytes | None = OLD_REPORT):
    manifest = base / "conversions" / "openvino-ep-manifest.json"
    report = base / "reports" / "openvino-ep-report.json"
    transaction = base / "ignored" / "transaction"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    transaction.mkdir(parents=True, exist_ok=True)
    if manifest_old is not None:
        manifest.write_bytes(manifest_old)
    if report_old is not None:
        report.write_bytes(report_old)
    return manifest, report, transaction


def payloads(manifest: Path, report: Path) -> dict[Path, bytes]:
    return {manifest: NEW_MANIFEST, report: NEW_REPORT}


def journal(transaction: Path) -> dict[str, object] | None:
    path = transaction / JOURNAL_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def residue(base: Path) -> list[str]:
    return sorted(str(path.relative_to(base)) for path in base.rglob("*") if path.is_file() and ("openvino-stage" in path.name or "backup" in path.name or "journal" in path.name or "recovery" in path.name))


def assert_generation(manifest: Path, report: Path, manifest_bytes: bytes | None, report_bytes: bytes | None) -> None:
    for path, payload in ((manifest, manifest_bytes), (report, report_bytes)):
        actual = identity(path)
        wanted = expected(payload)
        if actual != wanted:
            raise AssertionError(f"identity mismatch for {path}: {actual} != {wanted}")
        if payload is not None and path.read_bytes() != payload:
            raise AssertionError(f"byte mismatch for {path}")
    generations = []
    for path in (manifest, report):
        if path.is_file():
            generations.append(json.loads(path.read_text())["generation"])
    if len(set(generations)) > 1:
        raise AssertionError(f"mixed generation: {generations}")


def failure_case(name: str, event: str, occurrence: int = 1, *, old=(OLD_MANIFEST, OLD_REPORT), recovery_fault=None, retain=False) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-transaction-") as raw:
        base = Path(raw)
        manifest, report, transaction = topology(base, *old)
        error = None
        primary_fault = Fail(event, occurrence)
        def injected(event_name: str, context: dict[str, object]) -> None:
            primary_fault(event_name, context)
            if recovery_fault is not None:
                recovery_fault(event_name, context)
        try:
            publish(payloads(manifest, report), transaction, fault=injected)
        except BaseException as caught:
            error = caught
        if error is None:
            raise AssertionError(f"{name}: expected failure")
        if retain:
            if journal(transaction) is None or not residue(base):
                raise AssertionError(f"{name}: unrecoverable state lost journal/material")
        else:
            recover(transaction)
            current = (identity(manifest), identity(report))
            old_ids = (expected(old[0]), expected(old[1]))
            new_ids = (expected(NEW_MANIFEST), expected(NEW_REPORT))
            if current == old_ids:
                assert_generation(manifest, report, *old)
            elif current == new_ids:
                assert_generation(manifest, report, NEW_MANIFEST, NEW_REPORT)
            else:
                raise AssertionError(f"{name}: recovery left neither complete generation: {current}")
            if journal(transaction) is not None or residue(base):
                raise AssertionError(f"{name}: successful recovery left residue {residue(base)}")
        messages = flatten(error)
        if not any("injected" in item for item in messages):
            raise AssertionError(f"{name}: injected error was masked: {messages}")
        return {"name": name, "errors": messages, "journal": journal(transaction), "residue": residue(base)}


def crash_child(base: Path, event: str, occurrence: int) -> int:
    manifest, report, transaction = topology(base)
    publish(payloads(manifest, report), transaction, fault=Fail(event, occurrence, crash=True))
    return 0


def crash_case(name: str, event: str, occurrence: int, expected_generation: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-crash-") as raw:
        base = Path(raw)
        result = subprocess.run([sys.executable, __file__, "--crash-child", str(base), event, str(occurrence)])
        if result.returncode != 97:
            raise AssertionError(f"{name}: child did not terminate at injected crash ({result.returncode})")
        manifest = base / "conversions" / "openvino-ep-manifest.json"
        report = base / "reports" / "openvino-ep-report.json"
        transaction = base / "ignored" / "transaction"
        if journal(transaction) is None:
            raise AssertionError(f"{name}: crash did not leave durable journal")
        recovered = recover(transaction)
        wanted = (NEW_MANIFEST, NEW_REPORT) if expected_generation == "new" else (OLD_MANIFEST, OLD_REPORT)
        assert_generation(manifest, report, *wanted)
        if journal(transaction) is not None or residue(base):
            raise AssertionError(f"{name}: recovery residue {residue(base)}")
        return {"name": name, "recovery": recovered}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-child", nargs=3, metavar=("BASE", "EVENT", "OCCURRENCE"))
    args = parser.parse_args()
    if args.crash_child:
        return crash_child(Path(args.crash_child[0]), args.crash_child[1], int(args.crash_child[2]))

    cases = []
    cases.append(failure_case("first publish replace failure", "publish.replace", 1))
    cases.append(failure_case("second publish replace failure", "publish.replace", 2))
    cases.append(failure_case("manifest absent report present restores existence", "publish.replace", 2, old=(None, OLD_REPORT)))
    cases.append(failure_case("manifest present report absent restores existence", "publish.replace", 2, old=(OLD_MANIFEST, None)))
    cases.append(failure_case("both targets absent restore existence", "publish.replace", 2, old=(None, None)))

    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-staging-") as raw:
        base = Path(raw)
        manifest, report, transaction = topology(base)
        def corrupt(event: str, context: dict[str, object]) -> None:
            if event == "staging.ready" and Path(context["target"]) == manifest:
                Path(context["path"]).write_bytes(b"corrupt")
        try:
            publish(payloads(manifest, report), transaction, fault=corrupt)
            raise AssertionError("staging digest mismatch unexpectedly passed")
        except RuntimeError as error:
            if "staging bytes/SHA-256 mismatch" not in str(error):
                raise
        assert_generation(manifest, report, OLD_MANIFEST, OLD_REPORT)
        if journal(transaction) is not None or residue(base):
            raise AssertionError("staging digest failure left residue")
        cases.append({"name": "staging bytes/SHA mismatch preserves targets"})

    cases.append(failure_case("journal write failure", "journal.write", 1))
    cases.append(failure_case("journal fsync failure", "journal.fsync", 1))
    cases.append(failure_case("first target directory fsync failure", "publish.directory_fsync", 1))
    cases.append(failure_case("second target directory fsync failure", "publish.directory_fsync", 2))
    cases.append(failure_case("rollback replace failure retains recovery state", "publish.replace", 2, recovery_fault=Fail("recovery.replace", 1), retain=True))
    cases.append(failure_case("rollback directory fsync failure retains recovery state", "publish.replace", 2, recovery_fault=Fail("recovery.directory_fsync", 1), retain=True))

    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-recovery-fault-") as raw:
        base = Path(raw)
        manifest, report, transaction = topology(base)
        child = subprocess.run([sys.executable, __file__, "--crash-child", str(base), "publish.replaced", "1"])
        if child.returncode != 97:
            raise AssertionError("recovery setup crash failed")
        for name, fault in (
            ("recovery replace failure retains journal", Fail("recovery.replace", 1)),
            ("recovery directory fsync failure retains journal", Fail("recovery.directory_fsync", 1)),
        ):
            try:
                recover(transaction, fault=fault)
                raise AssertionError(f"{name}: expected failure")
            except BaseException as error:
                if "injected" not in "\n".join(flatten(error)):
                    raise
            if journal(transaction) is None or not residue(base):
                raise AssertionError(f"{name}: durable recovery material missing")
            cases.append({"name": name})
        recover(transaction)
        assert_generation(manifest, report, OLD_MANIFEST, OLD_REPORT)

    cases.append(failure_case("backup cleanup unlink failure", "cleanup.unlink", 1))
    cases.append(failure_case("cleanup final directory fsync failure reports error", "cleanup.final_directory_fsync", 1))
    cases.append(crash_case("crash after first replace recovers old generation", "publish.replaced", 1, "old"))
    cases.append(crash_case("crash after second replace before commit recovers old generation", "publish.replaced", 2, "old"))
    cases.append(crash_case("crash after commit marker before cleanup finishes new generation", "journal.committed", 1, "new"))

    with tempfile.TemporaryDirectory(prefix="rimeflow-openvino-success-") as raw:
        base = Path(raw)
        manifest, report, transaction = topology(base)
        result = publish(payloads(manifest, report), transaction)
        assert_generation(manifest, report, NEW_MANIFEST, NEW_REPORT)
        if journal(transaction) is not None or residue(base):
            raise AssertionError(f"successful publication residue: {residue(base)}")
        cases.append({"name": "successful publication updates both targets", "result": result})
        cases.append({"name": "success leaves both parent directories and transaction directory clean"})

    cases.append(failure_case("unrecoverable recovery retains journal and backup", "publish.replace", 2, recovery_fault=Fail("recovery.replace", 1), retain=True))
    print(json.dumps({"filesystemCases": cases, "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
