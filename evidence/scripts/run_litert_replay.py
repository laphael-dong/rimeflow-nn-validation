#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from android_litert_candidate import (
    EXPECTED_ARTIFACT_MANIFEST_SHA256,
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_SOURCE_SHA256,
    build_candidate_manifest,
    build_candidate_report,
    validate_candidate_report,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_status(root: Path) -> dict[str, str]:
    tracked = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    full = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return {"full": full, "tracked": tracked}


def sanitize(value: str, root: Path, source: Path) -> str:
    return value.replace(str(source), "$HANDOFF_ASSETS/yolov8n.pt").replace(str(root), "$REPO")


def execute(command: list[str], root: Path, source: Path) -> dict[str, object]:
    started_at = utc_now()
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    ended_at = utc_now()
    return {
        "command": [sanitize(item, root, source) for item in command],
        "endedAt": ended_at,
        "exitCode": result.returncode,
        "startedAt": started_at,
        "stderr": sanitize(result.stderr, root, source),
        "stdout": sanitize(result.stdout, root, source),
    }


def parse_last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("command stdout did not contain a JSON object")


def stable_artifact_manifest(
    root: Path,
    conversion: dict[str, object],
    golden: dict[str, object],
) -> dict[str, object]:
    lock_path = root / "evidence/tooling/litert-requirements.lock"
    input_lock = root / "evidence/tooling/litert-requirements.in"
    artifact = golden["artifact"]
    return {
        "artifact": artifact,
        "conversion": {
            "command": [
                "$LITERT_PYTHON",
                "evidence/scripts/export_litert_model.py",
                "--pt",
                "$HANDOFF_ASSETS/yolov8n.pt",
                "--output-dir",
                ".evidence/litert/replay/round-N/conversion",
            ],
            "intermediates": [
                {
                    "format": "torch.export ExportedProgram",
                    "persistence": "in-memory",
                    "purpose": "capture the fused PyTorch graph with static batch=1/imgsz=640",
                },
                {
                    "format": "FX graph and MLIR/LiteRT compiler IR",
                    "persistence": "in-memory",
                    "purpose": "decompose/lower the ExportedProgram and emit the TFLite FlatBuffer",
                },
            ],
            "metadataNormalization": conversion["converter"]["metadataNormalization"],
            "parameters": conversion["commandParameters"],
            "path": "official Ultralytics YOLO.export(format='litert') -> litert-torch -> litert-converter -> TFLite",
        },
        "dependencies": {
            "lock": {
                "bytes": lock_path.stat().st_size,
                "path": "evidence/tooling/litert-requirements.lock",
                "sha256": sha256(lock_path),
            },
            "requested": {
                "bytes": input_lock.stat().st_size,
                "path": "evidence/tooling/litert-requirements.in",
                "sha256": sha256(input_lock),
            },
            "sources": [
                {
                    "packages": ["torch", "torchvision"],
                    "url": "https://download.pytorch.org/whl/cpu",
                },
                {
                    "packages": "all other locked Python distributions",
                    "url": "https://pypi.org/simple",
                },
                {
                    "component": "Ultralytics LiteRT exporter documentation/source",
                    "url": "https://docs.ultralytics.com/integrations/tflite/",
                },
                {
                    "component": "LiteRT PyTorch converter",
                    "url": "https://github.com/google-ai-edge/litert-torch",
                },
            ],
        },
        "ioContract": golden["contract"],
        "mapping": golden["mapping"],
        "ownership": golden["ownership"],
        "runtime": golden["runtime"],
        "schemaVersion": 1,
        "source": conversion["source"],
        "status": {
            "androidRunnerVerified": False,
            "artifactVerified": True,
            "hostInferenceVerified": True,
            "supported": False,
            "value": "host-inference-verified",
        },
        "toolchain": {
            "ai-edge-litert": version("ai-edge-litert"),
            "ai-edge-quantizer": version("ai-edge-quantizer"),
            "litert-converter": version("litert-converter"),
            "litert-torch": version("litert-torch"),
            "numpy": version("numpy"),
            "onnx2tf": {"installed": False, "reason": "the selected official format=litert path does not use onnx2tf"},
            "pip": version("pip"),
            "python": platform.python_version(),
            "tensorflow": {"installed": False, "reason": "the selected official format=litert path converts PyTorch directly"},
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "ultralytics": version("ultralytics"),
        },
        "usageScope": {
            "artifactRedistributionAllowed": False,
            "modelLicenseMetadata": "AGPL-3.0 License (https://ultralytics.com/license)",
            "productPackaging": "excluded",
            "purpose": "internal-onnx-base-framework-validation-only",
            "trackedByGit": False,
        },
        "validation": {
            "fixtureCount": golden["summary"]["fixtureCount"],
            "goldenPassed": golden["passed"],
            "summary": golden["summary"],
            "tolerances": golden["tolerances"],
            "webReference": golden["webReference"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--workspace", default=".evidence/litert/replay", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-candidate", action="store_true")
    parser.add_argument("--bundle-dir", default=".evidence/litert/candidate-bundle", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = args.pt.resolve(strict=True)
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {sha256(source)}"
        )
    workspace = (root / args.workspace).resolve()
    bundle_root = (root / args.bundle_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    rounds = []
    stable_manifests = []
    golden_reports = []
    for round_number in (1, 2):
        round_root = workspace / f"round-{round_number}"
        conversion_dir = round_root / "conversion"
        web_dir = round_root / "web-reference"
        golden_path = round_root / "golden-report.json"
        conversion_dir.mkdir(parents=True, exist_ok=True)
        before = git_status(root)
        web = execute(
            ["node", "evidence/scripts/export_web_reference_tensors.mjs", str(web_dir)], root, source
        )
        if web["exitCode"] != 0:
            raise SystemExit(f"round {round_number} Web reference export failed: {web['stderr']}")
        conversion = execute(
            [
                sys.executable,
                "evidence/scripts/export_litert_model.py",
                "--pt",
                str(source),
                "--output-dir",
                str(conversion_dir),
            ],
            root,
            source,
        )
        if conversion["exitCode"] != 0:
            raise SystemExit(f"round {round_number} conversion failed: {conversion['stderr']}")
        conversion_result = parse_last_json(conversion["stdout"])
        artifact_path = conversion_dir / "yolov8n.tflite"
        artifact_sha256 = sha256(artifact_path)
        if artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise SystemExit(
                f"round {round_number} produced a new failed candidate identity: "
                f"expected {EXPECTED_ARTIFACT_SHA256}, got {artifact_sha256}"
            )
        validation = execute(
            [
                sys.executable,
                "evidence/scripts/validate_litert_golden.py",
                "--model",
                str(artifact_path),
                "--web-reference-dir",
                str(web_dir),
                "--output",
                str(golden_path),
                "--expected-artifact-sha256",
                EXPECTED_ARTIFACT_SHA256,
            ],
            root,
            source,
        )
        if validation["exitCode"] != 0:
            raise SystemExit(f"round {round_number} LiteRT golden validation failed: {validation['stderr']}")
        golden = json.loads(golden_path.read_text())
        stable_manifest = stable_artifact_manifest(root, conversion_result, golden)
        stable_manifest_path = round_root / "artifact-manifest.json"
        stable_manifest_path.write_bytes(json_bytes(stable_manifest))
        stable_manifest_sha256 = sha256(stable_manifest_path)
        if stable_manifest_sha256 != EXPECTED_ARTIFACT_MANIFEST_SHA256:
            raise SystemExit(
                f"round {round_number} produced a new failed artifact manifest identity: "
                f"expected {EXPECTED_ARTIFACT_MANIFEST_SHA256}, got {stable_manifest_sha256}"
            )
        after = git_status(root)
        rounds.append(
            {
                "artifact": {
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256(artifact_path),
                },
                "artifactManifest": {
                    "bytes": stable_manifest_path.stat().st_size,
                    "sha256": sha256(stable_manifest_path),
                },
                "conversion": conversion,
                "goldenReport": {
                    "bytes": golden_path.stat().st_size,
                    "sha256": sha256(golden_path),
                    "summary": golden["summary"],
                },
                "round": round_number,
                "validation": validation,
                "webReference": web,
                "worktreeAfter": after,
                "worktreeBefore": before,
            }
        )
        stable_manifests.append(stable_manifest)
        golden_reports.append(golden)

    artifact_equal = rounds[0]["artifact"] == rounds[1]["artifact"]
    manifest_equal = rounds[0]["artifactManifest"] == rounds[1]["artifactManifest"]
    golden_equal = rounds[0]["goldenReport"] == rounds[1]["goldenReport"]
    metadata_equal = stable_manifests[0]["ioContract"] == stable_manifests[1]["ioContract"]
    fixtures_equal = golden_reports[0]["fixtures"] == golden_reports[1]["fixtures"]
    tracked_unchanged = all(
        item["worktreeBefore"]["tracked"] == item["worktreeAfter"]["tracked"] for item in rounds
    )
    comparison = {
        "artifactSha256Equal": artifact_equal,
        "fixtureRawAndDecodedEqual": fixtures_equal,
        "goldenReportDigestEqual": golden_equal,
        "ioMetadataEqual": metadata_equal,
        "manifestDigestEqual": manifest_equal,
        "trackedWorktreeUnchanged": tracked_unchanged,
    }
    if not all(comparison.values()):
        raise SystemExit(f"LiteRT replay determinism failed: {comparison}")

    final_artifact = root / ".evidence/litert/artifacts/yolov8n-fp32.tflite"
    final_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workspace / "round-2/conversion/yolov8n.tflite", final_artifact)
    replay = {
        "artifactLocation": ".evidence/litert/artifacts/yolov8n-fp32.tflite",
        "comparison": comparison,
        "rounds": rounds,
        "schemaVersion": 1,
    }
    replay_path = workspace / "litert-replay.json"
    replay_path.write_bytes(json_bytes(replay))

    candidate_manifest = build_candidate_manifest(
        root,
        final_artifact,
        workspace,
        bundle_root,
        rounds,
        stable_manifests[1],
        golden_reports[1],
        comparison,
    )
    candidate_manifest_bytes = json_bytes(candidate_manifest)
    candidate_workspace_path = workspace / "android-litert-candidate-manifest.json"
    candidate_workspace_path.write_bytes(candidate_manifest_bytes)
    (bundle_root / "manifest.json").write_bytes(candidate_manifest_bytes)
    candidate_report = build_candidate_report(root, candidate_manifest, candidate_manifest_bytes)
    candidate_report_bytes = json_bytes(candidate_report)
    candidate_report_workspace_path = workspace / "android-litert-candidate-report.json"
    candidate_report_workspace_path.write_bytes(candidate_report_bytes)

    tracked_manifest = root / "evidence/conversions/litert-artifact-manifest.json"
    tracked_golden = root / "evidence/reports/litert-golden-report.json"
    tracked_report = root / "evidence/reports/litert-conversion-report.json"
    tracked_candidate_manifest = root / "evidence/conversions/android-litert-candidate-manifest.json"
    tracked_candidate_report = root / "evidence/reports/android-litert-candidate-report.json"
    if args.record:
        tracked_manifest.write_bytes(json_bytes(stable_manifests[1]))
        tracked_golden.write_bytes(json_bytes(golden_reports[1]))
        tracked_report.write_bytes(
            json_bytes(
                {
                    "comparison": comparison,
                    "commands": {
                        "conversion": rounds[0]["conversion"]["command"],
                        "goldenValidation": rounds[0]["validation"]["command"],
                        "webReference": rounds[0]["webReference"]["command"],
                    },
                    "rounds": rounds,
                    "schemaVersion": 1,
                }
            )
        )
    else:
        expected = [
            (tracked_manifest, json_bytes(stable_manifests[1])),
            (tracked_golden, json_bytes(golden_reports[1])),
        ]
        for path, actual in expected:
            if not path.exists() or path.read_bytes() != actual:
                raise SystemExit(f"tracked LiteRT evidence drift: {path.relative_to(root)}")

    if args.record_candidate:
        tracked_candidate_manifest.write_bytes(candidate_manifest_bytes)
        tracked_candidate_report.write_bytes(candidate_report_bytes)
    else:
        for path, actual in (
            (tracked_candidate_manifest, candidate_manifest_bytes),
            (tracked_candidate_report, candidate_report_bytes),
        ):
            if not path.exists() or path.read_bytes() != actual:
                raise SystemExit(f"tracked Android LiteRT candidate evidence drift: {path.relative_to(root)}")
    validate_candidate_report(root, candidate_report, candidate_manifest)

    print(
        json.dumps(
            {
                "artifact": {"bytes": final_artifact.stat().st_size, "sha256": sha256(final_artifact)},
                "candidateManifest": {
                    "bytes": len(candidate_manifest_bytes),
                    "sha256": hashlib.sha256(candidate_manifest_bytes).hexdigest(),
                },
                "comparison": comparison,
                "recordedCandidate": args.record_candidate,
                "recorded": args.record,
                "replay": str(replay_path.relative_to(root)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
