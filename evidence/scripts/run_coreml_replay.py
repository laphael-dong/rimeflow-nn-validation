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


def stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def package_tree(package: Path) -> dict[str, object]:
    files = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(package).as_posix(),
                "sha256": sha256(path),
            }
        )
    return {
        "canonicalization": "sorted POSIX relative path + bytes + SHA-256; compact JSON with sorted keys; directory timestamps excluded",
        "digest": stable_digest(files),
        "fileCount": len(files),
        "files": files,
        "totalFileBytes": sum(item["bytes"] for item in files),
    }


def artifact_snapshot(package: Path) -> dict[str, object]:
    if not package.exists():
        return {
            "available": False,
            "treeDigest": None,
            "totalFileBytes": None,
            "weightBlobSha256": None,
        }
    if not package.is_dir():
        raise SystemExit(f"recorded Core ML artifact is not a directory: {package}")
    tree = package_tree(package)
    files = {item["path"]: item for item in tree["files"]}
    weight = files.get("Data/com.apple.CoreML/weights/weight.bin")
    if tree["fileCount"] != 3 or weight is None:
        raise SystemExit(f"recorded Core ML artifact has an invalid package tree: {tree}")
    return {
        "available": True,
        "treeDigest": tree["digest"],
        "totalFileBytes": tree["totalFileBytes"],
        "weightBlobSha256": weight["sha256"],
    }


def weight_blob_sha256(result: dict[str, object]) -> str:
    return file_map(result)["Data/com.apple.CoreML/weights/weight.bin"]["sha256"]


def semantic_digests(result: dict[str, object]) -> dict[str, str]:
    return {
        "normalizedPackageManifestSha256": result["packageManifest"]["normalizedSha256"],
        "normalizedSpecSha256": result["spec"]["normalizedSpecSha256"],
        "weightBlobSha256": weight_blob_sha256(result),
    }


def portable_spec_evidence(package: Path) -> dict[str, object]:
    from coremltools.proto import Model_pb2

    spec_path = package / "Data/com.apple.CoreML/model.mlmodel"
    spec = Model_pb2.Model()
    spec.ParseFromString(spec_path.read_bytes())
    volatile_keys = ["date", "com.github.apple.coremltools.conversion_date"]
    removed = {}
    for key in volatile_keys:
        if key in spec.description.metadata.userDefined:
            removed[key] = spec.description.metadata.userDefined[key]
            del spec.description.metadata.userDefined[key]
    return {
        "normalization": "coreml-spec-portable-v2: remove only description.metadata.userDefined date and com.github.apple.coremltools.conversion_date",
        "removedMetadata": removed,
        "sha256": hashlib.sha256(spec.SerializeToString(deterministic=True)).hexdigest(),
    }


def validate_manifest_tree(manifest: dict[str, object]) -> None:
    tree = manifest["artifact"]["tree"]
    if stable_digest(tree["files"]) != tree["digest"]:
        raise SystemExit("tracked Core ML manifest canonical tree digest is invalid")
    if tree["fileCount"] != len(tree["files"]):
        raise SystemExit("tracked Core ML manifest file count is invalid")
    if tree["totalFileBytes"] != sum(item["bytes"] for item in tree["files"]):
        raise SystemExit("tracked Core ML manifest byte count is invalid")


def load_recorded_evidence(root: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    manifest_path = root / "evidence/conversions/coreml-artifact-manifest.json"
    report_path = root / "evidence/reports/coreml-conversion-report.json"
    manifest_bytes = manifest_path.read_bytes()
    report_bytes = report_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = json.loads(report_bytes)
    validate_manifest_tree(manifest)
    recorded_digest = manifest["artifact"]["tree"]["digest"]
    if manifest.get("recordedArtifactTreeDigest") != recorded_digest:
        raise SystemExit("tracked Core ML manifest does not distinguish recorded artifact identity")
    if report.get("mode") != "record" or report.get("recordedArtifactTreeDigest") != recorded_digest:
        raise SystemExit("tracked Core ML conversion report is not synchronized with the recorded artifact")
    if report.get("semanticReplayDigests", {}).get("recorded") != manifest.get("semanticReplayDigests"):
        raise SystemExit("tracked Core ML semantic digests are not synchronized")
    if report["rounds"][-1]["artifactTree"] != manifest["artifact"]["tree"]:
        raise SystemExit("tracked Core ML report does not identify the recorded package tree")
    return manifest, report, {
        "manifest": {
            "bytes": len(manifest_bytes),
            "path": "evidence/conversions/coreml-artifact-manifest.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "report": {
            "bytes": len(report_bytes),
            "path": "evidence/reports/coreml-conversion-report.json",
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
    }


def validate_semantic_replay(
    root: Path,
    manifest: dict[str, object],
    report: dict[str, object],
    results: list[dict[str, object]],
    portable_recorded: dict[str, object],
) -> dict[str, object]:
    recorded = manifest["semanticReplayDigests"]
    lock_path = root / manifest["dependencies"]["lock"]["path"]
    expected_versions = {
        key: manifest["toolchain"][key]
        for key in ["coremltools", "numpy", "torch", "torchvision", "ultralytics"]
    }
    checks = {
        "coordinates": all(item["spec"]["coordinates"] == manifest["spec"]["coordinates"] for item in results),
        "float32Precision": all(item["spec"]["computePrecision"] == manifest["spec"]["computePrecision"] for item in results),
        "inputContract": all(item["spec"]["input"] == manifest["spec"]["input"] for item in results),
        "nmsResponsibility": all(item["spec"]["nms"] == manifest["spec"]["nms"] for item in results),
        "normalizedPackageManifestDigest": all(semantic_digests(item)["normalizedPackageManifestSha256"] == recorded["normalizedPackageManifestSha256"] for item in results),
        "normalizedSpecDigest": all(item["portableSpec"]["sha256"] == portable_recorded["sha256"] for item in results),
        "outputContract": all(item["spec"]["output"] == manifest["spec"]["output"] for item in results),
        "preprocessingResponsibility": all(item["spec"]["preprocessing"] == manifest["spec"]["preprocessing"] for item in results),
        "recordedReport": report["semanticReplayDigests"]["recorded"] == recorded,
        "source": all(item["source"]["before"] == manifest["source"]["before"] and item["source"]["after"] == manifest["source"]["after"] for item in results),
        "toolchain": all(item["versions"] == expected_versions and item["host"]["python"] == manifest["toolchain"]["python"] for item in results),
        "toolchainLock": lock_path.stat().st_size == manifest["dependencies"]["lock"]["bytes"] and sha256(lock_path) == manifest["dependencies"]["lock"]["sha256"],
        "weightBlob": all(semantic_digests(item)["weightBlobSha256"] == recorded["weightBlobSha256"] for item in results),
    }
    checks["allMatched"] = all(checks.values())
    checks["portableSpec"] = {
        "recordedArtifact": portable_recorded,
        "rounds": [item["portableSpec"] for item in results],
    }
    if not checks["allMatched"]:
        raise SystemExit(f"Core ML replay differs from recorded semantic evidence: {checks}")
    return checks


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
            "Data/com.apple.CoreML/model.mlmodel description.metadata.userDefined.com.github.apple.coremltools.conversion_date",
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
    recorded_tree_digest = result["artifact"]["tree"]["digest"]
    recorded_semantic_digests = semantic_digests(result)
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
        "recordedArtifactTreeDigest": recorded_tree_digest,
        "schemaVersion": 1,
        "semanticReplayDigests": recorded_semantic_digests,
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


def recorded_artifact_verification(
    before: dict[str, object],
    after: dict[str, object],
    expected_digest: str,
    record_mode: bool,
) -> dict[str, object]:
    unchanged = (
        before["available"] == after["available"]
        and before["treeDigest"] == after["treeDigest"]
    )
    exact_before = before["available"] and before["treeDigest"] == expected_digest
    exact_after = after["available"] and after["treeDigest"] == expected_digest
    if record_mode:
        status = "recorded" if exact_after else "recording-failed"
    elif not before["available"] and not after["available"]:
        status = "recorded-artifact-unavailable"
    elif unchanged and exact_before and exact_after:
        status = "verified-preserved"
    else:
        status = "recorded-artifact-changed"
    return {
        "afterTreeDigest": after["treeDigest"],
        "availableAfter": after["available"],
        "availableBefore": before["available"],
        "beforeTreeDigest": before["treeDigest"],
        "exactIdentityVerifiedAfter": exact_after,
        "exactIdentityVerifiedBefore": exact_before,
        "expectedTreeDigest": expected_digest,
        "status": status,
        "unchanged": unchanged,
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
    final_artifact = root / ".evidence/coreml/artifacts/yolov8n-fp32.mlpackage"
    if workspace != allowed_root and allowed_root not in workspace.parents:
        raise SystemExit("Core ML replay workspace must stay under .evidence/coreml")
    if (
        workspace == final_artifact
        or workspace in final_artifact.parents
        or final_artifact in workspace.parents
    ):
        raise SystemExit("Core ML replay workspace must not overlap the recorded artifact")

    recorded_manifest = None
    recorded_report = None
    tracked_evidence_before = None
    if not args.record:
        recorded_manifest, recorded_report, tracked_evidence_before = load_recorded_evidence(root)
    recorded_before = artifact_snapshot(final_artifact)
    if (
        not args.record
        and recorded_before["available"]
        and recorded_before["treeDigest"] != recorded_manifest["recordedArtifactTreeDigest"]
    ):
        raise SystemExit("recorded Core ML artifact tree drifted before non-record replay")
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
        result["portableSpec"] = portable_spec_evidence(
            conversion_dir / "yolov8n.mlpackage"
        )
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

    if args.record:
        manifest = tracked_manifest(root, results[1], comparison)
        expected_recorded_digest = manifest["recordedArtifactTreeDigest"]
        staging_artifact = final_artifact.parent / ".yolov8n-fp32.mlpackage.recording"
        final_artifact.parent.mkdir(parents=True, exist_ok=True)
        if staging_artifact.exists():
            shutil.rmtree(staging_artifact)
        shutil.copytree(workspace / "round-2/conversion/yolov8n.mlpackage", staging_artifact)
        if package_tree(staging_artifact)["digest"] != expected_recorded_digest:
            raise SystemExit("staged Core ML artifact differs from the recorded manifest")
        if final_artifact.exists():
            shutil.rmtree(final_artifact)
        staging_artifact.rename(final_artifact)
        recorded_after = artifact_snapshot(final_artifact)
        recorded_verification = recorded_artifact_verification(
            recorded_before, recorded_after, expected_recorded_digest, True
        )
        if not recorded_verification["exactIdentityVerifiedAfter"]:
            raise SystemExit("recorded Core ML artifact identity verification failed")
        manifest["recordedArtifactVerification"] = recorded_verification
        semantic_replay_digests = {
            "recorded": manifest["semanticReplayDigests"],
            "rounds": [semantic_digests(item) for item in results],
        }
        semantic_validation = {
            "allMatched": all(
                item == manifest["semanticReplayDigests"]
                for item in semantic_replay_digests["rounds"]
            ),
            "scope": "normalized spec/package manifest and weight blob; never an artifact identity",
        }
        if not semantic_validation["allMatched"]:
            raise SystemExit("recorded Core ML conversion rounds are not semantically deterministic")
        tracked_evidence = None
    else:
        manifest = recorded_manifest
        expected_recorded_digest = manifest["recordedArtifactTreeDigest"]
        if not recorded_before["available"]:
            raise SystemExit("portable Core ML replay requires the fixed recorded artifact")
        portable_recorded = portable_spec_evidence(final_artifact)
        semantic_validation = validate_semantic_replay(
            root, manifest, recorded_report, results, portable_recorded
        )
        semantic_validation["scope"] = (
            "normalized spec/package manifest, weight, contracts, source and toolchain; "
            "never an artifact identity"
        )
        semantic_replay_digests = {
            "recorded": manifest["semanticReplayDigests"],
            "rounds": [semantic_digests(item) for item in results],
        }
        recorded_after = artifact_snapshot(final_artifact)
        recorded_verification = recorded_artifact_verification(
            recorded_before, recorded_after, expected_recorded_digest, False
        )
        if not recorded_verification["unchanged"]:
            raise SystemExit("non-record Core ML replay changed the recorded artifact")
        if recorded_before["available"] and not recorded_verification["exactIdentityVerifiedAfter"]:
            raise SystemExit("non-record Core ML replay lost the recorded artifact identity")
        _, _, tracked_evidence_after = load_recorded_evidence(root)
        tracked_evidence = {
            key: {
                **tracked_evidence_before[key],
                "sha256After": tracked_evidence_after[key]["sha256"],
                "unchanged": tracked_evidence_before[key]["sha256"]
                == tracked_evidence_after[key]["sha256"],
            }
            for key in ["manifest", "report"]
        }
        if not all(item["unchanged"] for item in tracked_evidence.values()):
            raise SystemExit("non-record Core ML replay changed tracked evidence")

    replay = {
        "artifactLocation": ".evidence/coreml/artifacts/yolov8n-fp32.mlpackage",
        "comparison": comparison,
        "mode": "record" if args.record else "replay",
        "recordedArtifactTreeDigest": expected_recorded_digest,
        "recordedArtifactVerification": recorded_verification,
        "rounds": rounds,
        "schemaVersion": 1,
        "semanticReplayDigests": semantic_replay_digests,
        "semanticReplayValidation": semantic_validation,
        "trackedEvidence": tracked_evidence,
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
                    "mode": "record",
                    "recordedArtifactTreeDigest": expected_recorded_digest,
                    "recordedArtifactVerification": recorded_verification,
                    "rounds": rounds,
                    "schemaVersion": 1,
                    "semanticReplayDigests": semantic_replay_digests,
                    "semanticReplayValidation": semantic_validation,
                }
            )
        )

    print(
        json.dumps(
            {
                "comparison": comparison,
                "recorded": args.record,
                "recordedArtifactTreeDigest": expected_recorded_digest,
                "recordedArtifactVerification": recorded_verification,
                "replay": str(replay_path.relative_to(root)),
                "semanticReplayDigests": semantic_replay_digests,
                "workspaceRound2Tree": results[1]["artifact"]["tree"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
