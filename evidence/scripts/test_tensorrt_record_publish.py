#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tensorrt_record_publish as publish


class PublishTest(unittest.TestCase):
    def scenario(self, old: bytes | None = b"old"):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        candidate = root / "candidate.json"
        target = root / "tracked" / "report.json"
        target.parent.mkdir()
        candidate.write_bytes(b"new")
        if old is not None:
            target.write_bytes(old)
        return temporary, candidate, target

    def assert_restored(self, target: Path, old: bytes | None):
        self.assertEqual(target.exists(), old is not None)
        if old is not None:
            self.assertEqual(target.read_bytes(), old)

    def test_success_existing_and_missing(self):
        for old in (b"old", None):
            with self.subTest(old=old):
                temporary, candidate, target = self.scenario(old)
                with temporary:
                    publish.publish_report(candidate, target)
                    self.assertEqual(target.read_bytes(), b"new")
                    self.assertEqual(list(target.parent.glob(".*.staging")), [])
                    self.assertEqual(list(target.parent.glob(".*.backup")), [])

    def test_staging_write_failure_preserves_old(self):
        temporary, candidate, target = self.scenario()
        with temporary, mock.patch.object(publish, "durable_write", side_effect=OSError("write")):
            with self.assertRaises(OSError):
                publish.publish_report(candidate, target)
            self.assert_restored(target, b"old")

    def test_candidate_digest_failure_preserves_existing_and_missing(self):
        for old in (b"old", None):
            with self.subTest(old=old):
                temporary, candidate, target = self.scenario(old)
                with temporary:
                    with self.assertRaises(RuntimeError):
                        publish.publish_report(candidate, target, verify_digest=lambda _payload, _digest: (_ for _ in ()).throw(RuntimeError("digest")))
                    self.assert_restored(target, old)

    def test_replace_failure_preserves_old(self):
        temporary, candidate, target = self.scenario()
        with temporary:
            with self.assertRaises(OSError):
                publish.publish_report(candidate, target, replace=mock.Mock(side_effect=OSError("replace")))
            self.assert_restored(target, b"old")

    def test_candidate_and_target_symlinks_are_rejected(self):
        temporary, candidate, target = self.scenario()
        with temporary:
            candidate_link = candidate.parent / "candidate-link.json"
            candidate_link.symlink_to(candidate)
            with self.assertRaisesRegex(RuntimeError, "candidate.*non-symlink"):
                publish.publish_report(candidate_link, target)
            real_target = target.parent / "real-report.json"
            real_target.write_bytes(b"old")
            target.unlink()
            target.symlink_to(real_target)
            with self.assertRaisesRegex(RuntimeError, "target.*non-symlink"):
                publish.publish_report(candidate, target)
            self.assertEqual(real_target.read_bytes(), b"old")

    def test_post_publish_verification_failure_rolls_back(self):
        for old in (b"old", None):
            with self.subTest(old=old):
                temporary, candidate, target = self.scenario(old)
                with temporary:
                    with self.assertRaises(RuntimeError):
                        publish.publish_report(candidate, target, verify=lambda _path, _payload: (_ for _ in ()).throw(RuntimeError("verify")))
                    self.assert_restored(target, old)

    def test_directory_fsync_failure_rolls_back(self):
        temporary, candidate, target = self.scenario()
        calls = 0
        def fail_after_replace(_path):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise OSError("fsync")
        with temporary:
            with self.assertRaises(OSError):
                publish.publish_report(candidate, target, sync_directory=fail_after_replace)
            self.assert_restored(target, b"old")

    def test_rollback_failure_preserves_backup(self):
        temporary, candidate, target = self.scenario()
        replacements = 0
        def replace(source, destination):
            nonlocal replacements
            replacements += 1
            if replacements == 2:
                raise OSError("rollback")
            publish.os.replace(source, destination)
        with temporary:
            with self.assertRaises(ExceptionGroup):
                publish.publish_report(candidate, target, replace=replace, verify=lambda _path, _payload: (_ for _ in ()).throw(RuntimeError("verify")))
            self.assertTrue(list(target.parent.glob(".*.backup")))
            journals = list(target.parent.glob(".*.journal"))
            self.assertEqual(len(journals), 1)
            self.assertIn('"state": "replaced"', journals[0].read_text())

    def test_missing_target_rollback_failure_preserves_journal(self):
        temporary, candidate, target = self.scenario(None)
        with temporary, mock.patch.object(Path, "unlink", side_effect=OSError("rollback unlink")):
            with self.assertRaises(ExceptionGroup):
                publish.publish_report(candidate, target, verify=lambda _path, _payload: (_ for _ in ()).throw(RuntimeError("verify")))
            self.assertTrue(list(target.parent.glob(".*.journal")))


if __name__ == "__main__":
    unittest.main()
