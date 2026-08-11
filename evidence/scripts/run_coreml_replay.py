#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


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


def file_map(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {item["path"]: item for item in result["artifact"]["tree"]["files"]}


def compare_rounds(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    first_files = file_map(first)
    second_files = file_map(second)
    paths = sorted(set(first_files) | set(second_files))
    changed = []
    for path in paths:
        left = first_files.get(path)
        right = second_files.get(path)
        if left != right:
            changed.append({"path": path, "round1": left, "round2": right})
    first_metadata = first["spec"]["volatileMetadata"]
    second_metadata = second["spec"]["volatileMetadata"]
    return {
        "changedFiles": changed,
        "expectedVolatileFields": [
            "Manifest.json itemInfoEntries UUID keys/rootModelIdentifier",
            "Data/com.apple.CoreML/model.mlmodel description.metadata.userDefined.date",
        ],
        "ioMetadataEqual": {
            key: first["spec"][key] == second["spec"][key]
            for key in ["input", "output", "modelType", "specificationVersion"]
        },
        "normalizedPackageManifestDigestEqual": first["packageManifest"]["normalizedSha256"]
        == second["packageManifest"]["normalizedSha256"],
        "normalizedSpecDigestEqual": first["spec"]["normalizedSpecSha256"]
        == second["spec"]["normalizedSpecSha256"],
        "packageTreeDigestEqual": first["artifact"]["tree"]["digest"]
        == second["artifact"]["tree"]["digest"],
        "precisionContractEqual": first["spec"]["computePrecision"]
        == second["spec"]["computePrecision"],
        "sourceStateEqualAndUnchanged": first["source"]["before"] == first["source"]["after"]
        and second["source"]["before"] == second["source"]["after"]
        and first["source"]["before"] == second["source"]["before"],
        "volatileMetadata": {"round1": first_metadata, "round2": second_metadata},
        "weightBlobDigestEqual": first_files["Data/com.apple.CoreML/weights/weight.bin"]
        == second_files["Data/com.apple.CoreML/weights/weight.bin"],
    }


def tracked_manifest(root: Path, result: dict[str, object], comparison: dict[str, object]) -> dict[str, object]:
    input_path = root / "evidence/tooling/coreml-requirements.in"
    lock_path = root / "evidence/tooling/coreml-requirements.lock"
    return {
        "artifact": {
            "format": result["artifact"]["format"],
            "location": ".evidence/coreml/artifacts/yolov8n-fp32.mlpackage",
            "trackedByGit": False,
            "tree": result["artifact"]["tree"],
        },
        "conversion": result["conversion"]
        | {
            "command": [
                "$COREML_PYTHON",
                "evidence/scripts/export_coreml_model.py",
                "--pt",
                "$HANDOFF_ASSETS/yolov8n.pt",
                "--output-dir",
                ".evidence/coreml/replay/round-N/conversion",
            ],
            "parameters": result["commandParameters"],
        },
        "dependencies": {
            "lock": {
                "bytes": lock_path.stat().st_size,
                "path": "evidence/tooling/coreml-requirements.lock",
                "sha256": sha256(lock_path),
            },
            "requested": {
                "bytes": input_path.stat().st_size,
                "path": "evidence/tooling/coreml-requirements.in",
                "sha256": sha256(input_path),
            },
            "sources": [
                {"packages": ["torch", "torchvision"], "url": "https://download.pytorch.org/whl/cpu"},
                {"packages": "all other locked Python distributions", "url": "https://pypi.org/simple"},
                {"component": "Ultralytics Core ML integration", "url": "https://docs.ultralytics.com/integrations/coreml/"},
                {"component": "coremltools", "url": "https://github.com/apple/coremltools"},
            ],
        },
        "determinism": comparison,
        "packageManifest": result["packageManifest"],
        "schemaVersion": 1,
        "source": result["source"],
        "spec": result["spec"],
        "status": {
            "artifactVerified": True,
            "hostInferenceVerified": False,
            "iosRuntimeVerified": False,
            "macosRuntimeVerified": False,
            "supported": False,
            "value": "artifact-spec-verified",
        },
        "toolchain": {
            "coremltools": version("coremltools"),
            "numpy": version("numpy"),
            "pip": version("pip"),
            "python": platform.python_version(),
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--workspace", default=".evidence/coreml/replay", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = args.pt.resolve(strict=True)
    workspace = (root / args.workspace).resolve()
    allowed_root = (root / ".evidence/coreml").resolve()
    if workspace != allowed_root and allowed_root not in workspace.parents:
        raise SystemExit("Core ML replay workspace must stay under .evidence/coreml")
    workspace.mkdir(parents=True, exist_ok=True)

    rounds = []
    results = []
    for number in (1, 2):
        round_root = workspace / f"round-{number}"
        if round_root.exists():
            shutil.rmtree(round_root)
        conversion_dir = round_root / "conversion"
        conversion_dir.mkdir(parents=True)
        before = git_status(root)
        execution = execute(
            [
                sys.executable,
                "evidence/scripts/export_coreml_model.py",
                "--pt",
                str(source),
                "--output-dir",
                str(conversion_dir),
            ],
            root,
            source,
        )
        if execution["exitCode"] != 0:
            raise SystemExit(f"round {number} Core ML conversion failed: {execution['stderr']}")
        result = parse_last_json(execution["stdout"])
        after = git_status(root)
        rounds.append(
            {
                "artifactTree": result["artifact"]["tree"],
                "conversion": execution,
                "normalizedPackageManifestSha256": result["packageManifest"]["normalizedSha256"],
                "normalizedSpecSha256": result["spec"]["normalizedSpecSha256"],
                "round": number,
                "source": result["source"],
                "worktreeAfter": after,
                "worktreeBefore": before,
            }
        )
        results.append(result)

    comparison = compare_rounds(results[0], results[1])
    required_true = [
        comparison["ioMetadataEqual"]["input"],
        comparison["ioMetadataEqual"]["output"],
        comparison["ioMetadataEqual"]["modelType"],
        comparison["ioMetadataEqual"]["specificationVersion"],
        comparison["normalizedPackageManifestDigestEqual"],
        comparison["normalizedSpecDigestEqual"],
        comparison["precisionContractEqual"],
        comparison["sourceStateEqualAndUnchanged"],
        comparison["weightBlobDigestEqual"],
    ]
    if not all(required_true):
        raise SystemExit(f"Core ML semantic replay determinism failed: {comparison}")
    changed_paths = {item["path"] for item in comparison["changedFiles"]}
    expected_changed = {"Manifest.json", "Data/com.apple.CoreML/model.mlmodel"}
    if not comparison["packageTreeDigestEqual"] and changed_paths != expected_changed:
        raise SystemExit(f"unexpected Core ML package nondeterminism: {changed_paths}")

    final_artifact = root / ".evidence/coreml/artifacts/yolov8n-fp32.mlpackage"
    if final_artifact.exists():
        shutil.rmtree(final_artifact)
    final_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace / "round-2/conversion/yolov8n.mlpackage", final_artifact)
    manifest = tracked_manifest(root, results[1], comparison)
    replay = {
        "artifactLocation": ".evidence/coreml/artifacts/yolov8n-fp32.mlpackage",
        "comparison": comparison,
        "rounds": rounds,
        "schemaVersion": 1,
    }
    replay_path = workspace / "coreml-replay.json"
    replay_path.write_bytes(json_bytes(replay))

    if args.record:
        (root / "evidence/conversions/coreml-artifact-manifest.json").write_bytes(json_bytes(manifest))
        (root / "evidence/reports/coreml-conversion-report.json").write_bytes(
            json_bytes(
                {
                    "commands": {"conversion": rounds[0]["conversion"]["command"]},
                    "comparison": comparison,
                    "rounds": rounds,
                    "schemaVersion": 1,
                }
            )
        )

    print(
        json.dumps(
            {
                "comparison": comparison,
                "recorded": args.record,
                "replay": str(replay_path.relative_to(root)),
                "round2Tree": results[1]["artifact"]["tree"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
