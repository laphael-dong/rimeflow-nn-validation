#!/usr/bin/env python3
"""MindSpore record/replay 文件系统权限边界负向测试。"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import run_mindspore_replay as replay


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class MindsporeReplayGuardTests(unittest.TestCase):
    def test_workspace_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifacts = root / ".evidence/mindspore/artifacts"
            candidate = artifacts / "yolov8n-fp32.ms"
            for workspace in [artifacts, candidate, artifacts / "child", root / ".evidence/mindspore"]:
                with self.subTest(workspace=workspace):
                    with self.assertRaisesRegex(SystemExit, "must not overlap"):
                        replay.validate_workspace(root, workspace, artifacts)
            replay.validate_workspace(root, root / ".evidence/mindspore/replay", artifacts)

    def test_recorded_candidate_drift_fails_preflight(self) -> None:
        expected = b"recorded"
        changed = b"tampered"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.ms"
            candidate.write_bytes(changed)
            manifest = {
                "artifact": {"bytes": len(expected)},
                "recordedArtifactSha256": digest(expected),
            }
            with self.assertRaisesRegex(SystemExit, "drifted before"):
                replay.verify_non_record_preflight(replay.file_snapshot(candidate), manifest)
            self.assertEqual(candidate.read_bytes(), changed)

    def test_candidate_change_during_replay_fails_without_overwrite(self) -> None:
        original = b"recorded"
        external_change = b"changed-during-replay"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.ms"
            candidate.write_bytes(original)
            before = replay.file_snapshot(candidate)
            candidate.write_bytes(external_change)
            after = replay.file_snapshot(candidate)
            with self.assertRaisesRegex(SystemExit, "changed or created"):
                replay.verify_non_record_artifact(
                    before, after, len(original), digest(original)
                )
            self.assertEqual(candidate.read_bytes(), external_change)

    def test_missing_candidate_stays_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.ms"
            before = replay.file_snapshot(candidate)
            after = replay.file_snapshot(candidate)
            verification = replay.verify_non_record_artifact(
                before, after, 8, digest(b"recorded")
            )
            self.assertEqual(verification["status"], "recorded-artifact-unavailable")
            self.assertFalse(candidate.exists())

    def test_non_record_candidate_creation_is_rejected_without_deletion(self) -> None:
        created = b"unexpected"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.ms"
            before = replay.file_snapshot(candidate)
            candidate.write_bytes(created)
            after = replay.file_snapshot(candidate)
            with self.assertRaisesRegex(SystemExit, "created"):
                replay.verify_non_record_artifact(before, after, len(created), digest(created))
            self.assertEqual(candidate.read_bytes(), created)

    def test_tracked_evidence_change_is_rejected(self) -> None:
        before = {
            key: {
                "available": True,
                "bytes": 10,
                "mtimeNs": 1,
                "path": path,
                "sha256": digest(key.encode()),
            }
            for key, path in replay.TRACKED_EVIDENCE_PATHS.items()
        }
        after = {key: dict(value) for key, value in before.items()}
        after["goldenReport"]["sha256"] = digest(b"changed")
        with self.assertRaisesRegex(SystemExit, "changed tracked goldenReport"):
            replay.verify_tracked_evidence_preserved(before, after)

    def test_failed_staging_preserves_candidate_and_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "workspace.ms"
            candidate = root / "artifacts/candidate.ms"
            source.write_bytes(b"new-candidate")
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"old-candidate")
            tracked = {
                root / "manifest.json": b"old-manifest",
                root / "golden.json": b"old-golden",
                root / "conversion.json": b"old-conversion",
            }
            for path, payload in tracked.items():
                path.write_bytes(payload)
            with self.assertRaisesRegex(SystemExit, "staged MindSpore artifact differs"):
                replay.publish_recording(
                    source,
                    candidate,
                    {path: b"new" for path in tracked},
                    source.stat().st_size,
                    "0" * 64,
                )
            self.assertEqual(candidate.read_bytes(), b"old-candidate")
            for path, payload in tracked.items():
                self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
