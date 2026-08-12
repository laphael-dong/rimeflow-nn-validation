#!/usr/bin/env python3
"""TensorRT tracked report 的可恢复单文件事务发布。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload or digest_bytes(path.read_bytes()) != digest_bytes(payload):
        raise RuntimeError(f"staging verification failed: {path}")


def update_journal(path: Path, value: dict, sync_directory: Callable[[Path], None]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        replacement = path.with_suffix(path.suffix + ".next")
        durable_write(replacement, payload)
        os.replace(replacement, path)
    else:
        durable_write(path, payload)
    sync_directory(path.parent)


def publish_report(
    candidate: Path,
    target: Path,
    *,
    replace: Callable[[Path, Path], None] = os.replace,
    sync_directory: Callable[[Path], None] = fsync_directory,
    verify: Callable[[Path, bytes], None] | None = None,
    verify_digest: Callable[[bytes, str], None] | None = None,
) -> None:
    candidate = candidate.absolute()
    target = target.absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("record candidate must be a regular non-symlink file")
    target.parent.mkdir(parents=True, exist_ok=True)
    target_parent = target.parent.resolve(strict=True)
    target = target_parent / target.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RuntimeError("record target must be a regular non-symlink file")
    payload = candidate.read_bytes()
    if not payload:
        raise RuntimeError("record candidate is empty")
    transaction = uuid.uuid4().hex
    staging = target_parent / f".{target.name}.{transaction}.staging"
    backup = target_parent / f".{target.name}.{transaction}.backup"
    journal = target_parent / f".{target.name}.{transaction}.journal"
    rollback_staging = target_parent / f".{target.name}.{transaction}.rollback"
    if candidate == target:
        raise RuntimeError("record candidate and target must be different files")
    old_exists = target.is_file()
    old_payload = target.read_bytes() if old_exists else None
    published = False
    rollback_errors: list[BaseException] = []
    journal_value = {
        "transaction": transaction,
        "candidate": str(candidate),
        "target": str(target),
        "staging": str(staging),
        "backup": str(backup),
        "targetExisted": old_exists,
        "candidateSha256": digest_bytes(payload),
        "previousSha256": digest_bytes(old_payload) if old_payload is not None else None,
        "state": "initializing",
    }
    try:
        durable_write(staging, payload)
        if verify_digest is not None:
            verify_digest(staging.read_bytes(), journal_value["candidateSha256"])
        sync_directory(target.parent)
        if old_exists:
            durable_write(backup, old_payload or b"")
            sync_directory(target.parent)
        update_journal(journal, {**journal_value, "state": "prepared"}, sync_directory)
        replace(staging, target)
        published = True
        update_journal(journal, {**journal_value, "state": "replaced"}, sync_directory)
        sync_directory(target.parent)
        actual = target.read_bytes()
        if actual != payload or digest_bytes(actual) != digest_bytes(payload):
            raise RuntimeError("published report bytes/SHA mismatch")
        if verify is not None:
            verify(target, payload)
        update_journal(journal, {**journal_value, "state": "verified"}, sync_directory)
        if backup.exists():
            backup.unlink()
            sync_directory(target.parent)
        journal.unlink()
        sync_directory(target.parent)
    except BaseException as original:
        if published:
            try:
                if old_exists:
                    if not backup.exists():
                        durable_write(backup, old_payload or b"")
                        sync_directory(target.parent)
                    durable_write(rollback_staging, backup.read_bytes())
                    sync_directory(target.parent)
                    replace(rollback_staging, target)
                elif target.exists():
                    target.unlink()
                sync_directory(target.parent)
                if old_exists and target.read_bytes() != old_payload:
                    raise RuntimeError("rollback bytes mismatch")
                if not old_exists and target.exists():
                    raise RuntimeError("rollback existence mismatch")
                if journal.exists():
                    update_journal(journal, {**journal_value, "state": "rolled-back"}, sync_directory)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise ExceptionGroup("TensorRT report publication and rollback failed", [original, *rollback_errors])
        raise
    finally:
        if staging.exists():
            staging.unlink()
        if not rollback_errors and rollback_staging.exists():
            rollback_staging.unlink()
        if not rollback_errors and backup.exists():
            backup.unlink()
        if not rollback_errors and journal.exists():
            journal.unlink()
        if not rollback_errors:
            sync_directory(target.parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--validator", type=Path)
    args = parser.parse_args()
    verify = None
    if args.validator is not None:
        validator = args.validator.resolve(strict=True)
        def verify(path: Path, _payload: bytes) -> None:
            result = subprocess.run(["node", str(validator), str(path)], text=True, capture_output=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"published TensorRT report validator failed: {result.stdout}{result.stderr}")
    publish_report(args.candidate, args.target, verify=verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
