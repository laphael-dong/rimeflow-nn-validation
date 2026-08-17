#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath


EXPECTED_ARTIFACT_SHA256 = "794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1"
EXPECTED_ARTIFACT_BYTES = 12841227
EXPECTED_ARTIFACT_MANIFEST_SHA256 = "f1f95bac2006cd02e364123ab9e7556cc331e6d20f49006c50fcebd15d0c4881"
EXPECTED_SOURCE_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
EXPECTED_VALIDATION_BASE = "1c42b81c6066126b303c4a2ec606eafc7328517f"
EXPECTED_BASE_RUNTIME = "aaff04fa416b0366d6e69a61b4887b6d2fdd4215"
EXPECTED_MODEL_COMMIT = "8eb3a3f33ca945694af5a08b81b28a013af8695a"
EXPECTED_FIXTURE_IDS = [
    "no-detection",
    "single-target",
    "multi-class",
    "boundary-box",
    "extreme-aspect",
]
EXPECTED_COMPARISON = {
    "artifactSha256Equal": True,
    "fixtureRawAndDecodedEqual": True,
    "goldenReportDigestEqual": True,
    "ioMetadataEqual": True,
    "manifestDigestEqual": True,
    "trackedWorktreeUnchanged": True,
}
TRACKED_INPUTS = [
    "LICENSE",
    "models/yolov8n.onnx",
    "src/postprocess.rs",
    "evidence/conversions/litert-artifact-manifest.json",
    "evidence/fixtures/manifest.json",
    "evidence/golden/web-reference.json",
    "evidence/reports/litert-golden-report.json",
    "evidence/reports/model-provenance.json",
    "evidence/requirements/validation-requirement-test-matrix.json",
    "evidence/schemas/android-litert-candidate-manifest.schema.json",
    "evidence/schemas/android-litert-candidate-report.schema.json",
    "evidence/scripts/android_litert_candidate.py",
    "evidence/scripts/export_litert_model.py",
    "evidence/scripts/export_web_reference_tensors.mjs",
    "evidence/scripts/preprocess_contract.mjs",
    "evidence/scripts/run_litert_replay.py",
    "evidence/scripts/validate_litert_golden.py",
    "evidence/tooling/litert-requirements.in",
    "evidence/tooling/litert-requirements.lock",
    "evidence/tooling/web/bun.lock",
    "evidence/tooling/web/package.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def file_identity(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    return {"bytes": path.stat().st_size, "path": relative_path, "sha256": sha256(path)}


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_candidate_manifest(
    root: Path,
    artifact_path: Path,
    workspace: Path,
    bundle_root: Path,
    rounds: list[dict[str, object]],
    stable_manifest: dict[str, object],
    golden_report: dict[str, object],
    comparison: dict[str, bool],
) -> dict[str, object]:
    frozen = json.loads((root / "evidence/golden/web-reference.json").read_text())
    fixtures_manifest = json.loads((root / "evidence/fixtures/manifest.json").read_text())
    round_web_root = workspace / "round-2/web-reference"

    model_destination = bundle_root / "model/yolov8n-fp32.tflite"
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact_path, model_destination)

    fixtures = []
    for frozen_fixture in frozen["fixtures"]:
        fixture_id = frozen_fixture["id"]
        input_source = round_web_root / fixture_id / "input.f32le"
        input_relative = f"inputs/{fixture_id}.f32le"
        input_destination = bundle_root / input_relative
        input_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_source, input_destination)
        host_fixture = next(item for item in golden_report["fixtures"] if item["id"] == fixture_id)
        fixture_license = next(item["license"] for item in fixtures_manifest["images"] if item["id"] == fixture_id)
        fixtures.append(
            {
                "golden": {
                    "decodedCanonicalSha256": canonical_json_sha256(frozen_fixture["runs"][0]["decoded"]),
                    "decodedCount": len(frozen_fixture["runs"][0]["decoded"]),
                    "hostMappedRawSha256": host_fixture["mappedOutput"]["sha256Float32Le"],
                    "webRawSha256": frozen_fixture["runs"][0]["rawTensor"]["sha256Float32Le"],
                },
                "id": fixture_id,
                "input": {
                    "byteOrder": "little-endian",
                    "bytes": input_destination.stat().st_size,
                    "dtype": "float32",
                    "path": input_relative,
                    "sha256": sha256(input_destination),
                    "shape": [1, 3, 640, 640],
                },
                "license": fixture_license,
                "sourceImageSha256": frozen_fixture["imageSha256"],
            }
        )

    validation_tree = git_output(root, "rev-parse", f"{EXPECTED_VALIDATION_BASE}^{{tree}}")
    tracked_inputs = [file_identity(root, path) for path in TRACKED_INPUTS]
    manifest = {
        "candidateId": f"rimeflow-yolov8n-android-litert-fp32-{EXPECTED_ARTIFACT_SHA256[:12]}",
        "contract": {
            "io": stable_manifest["ioContract"],
            "mapping": stable_manifest["mapping"],
            "ownership": stable_manifest["ownership"],
        },
        "fixtures": fixtures,
        "hostProof": {
            "comparison": comparison,
            "goldenReport": {
                "path": "evidence/reports/litert-golden-report.json",
                "sha256": sha256(root / "evidence/reports/litert-golden-report.json"),
                "summary": golden_report["summary"],
            },
            "replayArtifactManifestSha256": [item["artifactManifest"]["sha256"] for item in rounds],
            "replayArtifactSha256": [item["artifact"]["sha256"] for item in rounds],
            "rounds": 2,
        },
        "licenses": {
            "checkpoint": {
                "declared": "AGPL-3.0 License (https://ultralytics.com/license)",
                "redistributionAllowed": False,
            },
            "converterAndRuntime": {
                "declared": "Apache-2.0",
                "packages": ["ai-edge-litert", "litert-converter", "litert-torch"],
            },
            "fixtures": {
                "manifestPath": "evidence/fixtures/manifest.json",
                "manifestSha256": sha256(root / "evidence/fixtures/manifest.json"),
                "policy": "per-file; test-and-evidence-only",
            },
            "repository": {
                "path": "LICENSE",
                "sha256": sha256(root / "LICENSE"),
                "spdx": "MIT",
            },
        },
        "model": {
            "artifact": {
                "bytes": model_destination.stat().st_size,
                "format": "TFLite FlatBuffer with deterministic Ultralytics metadata ZIP trailer",
                "path": "model/yolov8n-fp32.tflite",
                "sha256": sha256(model_destination),
            },
            "canonicalOnnx": {
                "firstRepositoryCommit": EXPECTED_MODEL_COMMIT,
                "path": "models/yolov8n.onnx",
                "sha256": sha256(root / "models/yolov8n.onnx"),
            },
            "id": "rimeflow-yolov8n",
            "sourceCheckpoint": {
                "bytes": 6549796,
                "releaseAssetId": 195719301,
                "releaseTag": "v8.3.0",
                "sha256": EXPECTED_SOURCE_SHA256,
                "url": EXPECTED_SOURCE_URL,
            },
            "version": "yolov8n-onnx-20260707",
        },
        "provenance": {
            "baseRuntime": {
                "commit": EXPECTED_BASE_RUNTIME,
                "repository": "https://github.com/laphael-dong/rimeflow-nn-base.git",
            },
            "trackedInputs": tracked_inputs,
            "validation": {
                "baseCommit": EXPECTED_VALIDATION_BASE,
                "baseTree": validation_tree,
                "repository": "https://github.com/laphael-dong/rimeflow-nn-validation.git",
            },
        },
        "schema": file_identity(root, "evidence/schemas/android-litert-candidate-manifest.schema.json"),
        "schemaVersion": 1,
        "status": {
            "androidTargetAccepted": False,
            "hostGoldenPassed": True,
            "supported": False,
            "value": "host-candidate-verified-pending-real-device",
        },
        "target": {
            "architecture": "arm64-v8a",
            "backend": "litert-v2",
            "os": "android",
            "runtimeVersion": "2.1.6",
        },
        "tolerances": frozen["tolerances"],
        "usageScope": {
            "artifactRedistributionAllowed": False,
            "productPackaging": "excluded",
            "purpose": "internal-rimeflow-framework-validation-only",
            "transfer": "approved-private-runner-only",
        },
    }
    validate_candidate_manifest(root, manifest, bundle_root=bundle_root, require_bundle=True)
    return manifest


def build_candidate_report(root: Path, manifest: dict[str, object], manifest_bytes: bytes) -> dict[str, object]:
    return {
        "candidateId": manifest["candidateId"],
        "candidateManifest": {
            "bytes": len(manifest_bytes),
            "path": "evidence/conversions/android-litert-candidate-manifest.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "hostGolden": copy.deepcopy(manifest["hostProof"]),
        "publication": {
            "artifactPublished": False,
            "immutableLocation": None,
            "reason": "no tracked approved private artifact destination; public model upload is prohibited",
            "remoteParityVerified": False,
            "testedLocalArtifactRetained": True,
        },
        "schema": file_identity(root, "evidence/schemas/android-litert-candidate-report.schema.json"),
        "schemaVersion": 1,
        "sourceCheckpoint": copy.deepcopy(manifest["model"]["sourceCheckpoint"]),
        "status": copy.deepcopy(manifest["status"]),
        "trackedInputState": {
            "deterministic": True,
            "files": copy.deepcopy(manifest["provenance"]["trackedInputs"]),
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_bundle_path(bundle_root: Path, relative_path: str) -> Path:
    parsed = PurePosixPath(relative_path)
    _require(not parsed.is_absolute() and ".." not in parsed.parts, f"unsafe bundle path: {relative_path}")
    resolved_root = bundle_root.resolve()
    resolved_path = (bundle_root / Path(*parsed.parts)).resolve()
    _require(resolved_path.is_relative_to(resolved_root), f"bundle path escapes root: {relative_path}")
    return resolved_path


def validate_candidate_manifest(
    root: Path,
    manifest: dict[str, object],
    bundle_root: Path | None = None,
    require_bundle: bool = False,
) -> None:
    _require(manifest.get("schemaVersion") == 1, "candidate schemaVersion mismatch")
    expected_target = {
        "architecture": "arm64-v8a",
        "backend": "litert-v2",
        "os": "android",
        "runtimeVersion": "2.1.6",
    }
    _require(manifest.get("target") == expected_target, "candidate target contract mismatch")
    status = manifest.get("status", {})
    _require(status.get("hostGoldenPassed") is True, "candidate host golden is not passed")
    _require(status.get("androidTargetAccepted") is False, "candidate overclaims Android target acceptance")
    _require(status.get("supported") is False, "candidate overclaims supported status")

    artifact = manifest.get("model", {}).get("artifact", {})
    _require(artifact.get("sha256") == EXPECTED_ARTIFACT_SHA256, "historical TFLite SHA-256 mismatch")
    _require(artifact.get("bytes") == EXPECTED_ARTIFACT_BYTES, "historical TFLite byte count mismatch")
    _require(artifact.get("path") == "model/yolov8n-fp32.tflite", "candidate artifact path mismatch")
    source = manifest.get("model", {}).get("sourceCheckpoint", {})
    _require(source.get("sha256") == EXPECTED_SOURCE_SHA256, "source checkpoint SHA-256 mismatch")
    _require(source.get("url") == EXPECTED_SOURCE_URL, "source checkpoint URL mismatch")
    onnx = manifest.get("model", {}).get("canonicalOnnx", {})
    _require(onnx.get("firstRepositoryCommit") == EXPECTED_MODEL_COMMIT, "model commit mismatch")
    _require(onnx.get("sha256") == sha256(root / "models/yolov8n.onnx"), "canonical ONNX mismatch")

    proof = manifest.get("hostProof", {})
    _require(proof.get("rounds") == 2, "candidate requires two replay rounds")
    _require(proof.get("comparison") == EXPECTED_COMPARISON, "candidate replay comparison mismatch")
    _require(
        proof.get("replayArtifactSha256") == [EXPECTED_ARTIFACT_SHA256, EXPECTED_ARTIFACT_SHA256],
        "independent replay artifact identities mismatch",
    )
    _require(
        proof.get("replayArtifactManifestSha256")
        == [EXPECTED_ARTIFACT_MANIFEST_SHA256, EXPECTED_ARTIFACT_MANIFEST_SHA256],
        "independent replay manifest identities mismatch",
    )
    summary = proof.get("goldenReport", {}).get("summary", {})
    _require(summary.get("fixtureCount") == 5, "host golden fixture count mismatch")
    _require(summary.get("allFinite") is True, "host golden contains non-finite values")
    _require(summary.get("allShapesMatched") is True, "host golden shape mismatch")
    _require(summary.get("deterministic") is True, "host golden is nondeterministic")
    _require(summary.get("classMismatchCount") == 0, "host golden class mismatch")
    _require(summary.get("rawToleranceMismatchCount") == 0, "host golden raw tolerance mismatch")

    frozen = json.loads((root / "evidence/golden/web-reference.json").read_text())
    _require(manifest.get("tolerances") == frozen["tolerances"], "frozen tolerances changed")
    fixtures = manifest.get("fixtures", [])
    _require([item.get("id") for item in fixtures] == EXPECTED_FIXTURE_IDS, "candidate fixture set mismatch")
    for item, frozen_fixture in zip(fixtures, frozen["fixtures"]):
        input_record = item.get("input", {})
        expected_input_sha = frozen_fixture["canonicalInput"]["sha256Float32Le"]
        _require(input_record.get("sha256") == expected_input_sha, f"{item.get('id')}: input SHA-256 mismatch")
        _require(input_record.get("bytes") == 4915200, f"{item.get('id')}: input byte count mismatch")
        _require(input_record.get("shape") == [1, 3, 640, 640], f"{item.get('id')}: input shape mismatch")
        golden = item.get("golden", {})
        expected_raw = frozen_fixture["runs"][0]["rawTensor"]["sha256Float32Le"]
        _require(golden.get("webRawSha256") == expected_raw, f"{item.get('id')}: Web golden mismatch")
        _require(
            golden.get("decodedCanonicalSha256")
            == canonical_json_sha256(frozen_fixture["runs"][0]["decoded"]),
            f"{item.get('id')}: decoded golden mismatch",
        )
        if require_bundle:
            _require(bundle_root is not None, "bundle root is required")
            path = _safe_bundle_path(bundle_root, input_record["path"])
            _require(path.is_file() and not path.is_symlink(), f"{item.get('id')}: bundle input missing or symlinked")
            _require(path.stat().st_size == input_record["bytes"], f"{item.get('id')}: bundle input bytes mismatch")
            _require(sha256(path) == input_record["sha256"], f"{item.get('id')}: bundle input digest mismatch")

    provenance = manifest.get("provenance", {})
    _require(provenance.get("validation", {}).get("baseCommit") == EXPECTED_VALIDATION_BASE, "validation base commit mismatch")
    _require(provenance.get("baseRuntime", {}).get("commit") == EXPECTED_BASE_RUNTIME, "Base runtime commit mismatch")
    tracked = provenance.get("trackedInputs", [])
    _require([item.get("path") for item in tracked] == TRACKED_INPUTS, "tracked input set mismatch")
    for item in tracked:
        actual = file_identity(root, item["path"])
        _require(item == actual, f"tracked input identity mismatch: {item['path']}")

    usage = manifest.get("usageScope", {})
    _require(usage.get("artifactRedistributionAllowed") is False, "candidate redistribution must remain prohibited")
    _require(usage.get("productPackaging") == "excluded", "candidate product packaging must remain excluded")
    _require(manifest.get("licenses", {}).get("checkpoint", {}).get("redistributionAllowed") is False, "checkpoint license boundary mismatch")

    schema = manifest.get("schema", {})
    _require(schema == file_identity(root, "evidence/schemas/android-litert-candidate-manifest.schema.json"), "candidate schema identity mismatch")
    matrix = json.loads((root / "evidence/requirements/validation-requirement-test-matrix.json").read_text())
    trace = matrix.get("androidLiteRtCandidateEvidence", {})
    _require(trace.get("taskId") == "RFB-ANDROID-VAL-CANDIDATE-01", "candidate trace task mismatch")
    _require(trace.get("manifestPath") == "evidence/conversions/android-litert-candidate-manifest.json", "candidate trace manifest mismatch")
    _require(trace.get("reportPath") == "evidence/reports/android-litert-candidate-report.json", "candidate trace report mismatch")
    tests = trace.get("tests", [])
    _require(
        [item.get("testId") for item in tests]
        == ["RFB-ANDROID-VAL-CANDIDATE-001", "RFB-ANDROID-VAL-CANDIDATE-002"],
        "candidate trace test coverage mismatch",
    )
    _require(len(set(trace.get("requirementScenarios", []))) == 3, "candidate trace requirement coverage mismatch")
    if require_bundle:
        model_path = _safe_bundle_path(bundle_root, artifact["path"])
        _require(model_path.is_file() and not model_path.is_symlink(), "bundle model missing or symlinked")
        _require(model_path.stat().st_size == artifact["bytes"], "bundle model bytes mismatch")
        _require(sha256(model_path) == artifact["sha256"], "bundle model digest mismatch")


def validate_candidate_report(root: Path, report: dict[str, object], manifest: dict[str, object]) -> None:
    _require(report.get("schemaVersion") == 1, "candidate report schemaVersion mismatch")
    _require(report.get("candidateId") == manifest.get("candidateId"), "candidate report ID mismatch")
    manifest_identity = report.get("candidateManifest", {})
    tracked_manifest_path = root / manifest_identity.get("path", "")
    _require(tracked_manifest_path.is_file(), "candidate report manifest path missing")
    _require(manifest_identity == file_identity(root, manifest_identity["path"]), "candidate report manifest identity mismatch")
    _require(report.get("hostGolden") == manifest.get("hostProof"), "candidate report host proof drift")
    _require(report.get("sourceCheckpoint") == manifest.get("model", {}).get("sourceCheckpoint"), "candidate report source drift")
    _require(report.get("status") == manifest.get("status"), "candidate report status drift")
    _require(report.get("trackedInputState", {}).get("deterministic") is True, "candidate tracked input state is not deterministic")
    _require(report.get("trackedInputState", {}).get("files") == manifest.get("provenance", {}).get("trackedInputs"), "candidate report tracked inputs drift")
    publication = report.get("publication", {})
    _require(publication.get("artifactPublished") is False, "candidate report overclaims artifact publication")
    _require(publication.get("immutableLocation") is None, "candidate report invents immutable artifact location")
    _require(publication.get("remoteParityVerified") is False, "candidate report overclaims remote parity")
    _require(report.get("schema") == file_identity(root, "evidence/schemas/android-litert-candidate-report.schema.json"), "candidate report schema identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="evidence/conversions/android-litert-candidate-manifest.json",
        type=Path,
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--require-bundle", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    bundle_root = None
    if args.bundle_root is not None:
        bundle_root = args.bundle_root if args.bundle_root.is_absolute() else root / args.bundle_root
    validate_candidate_manifest(root, manifest, bundle_root=bundle_root, require_bundle=args.require_bundle)
    if args.report is not None:
        report_path = args.report if args.report.is_absolute() else root / args.report
        validate_candidate_report(root, json.loads(report_path.read_text()), manifest)
    print(
        json.dumps(
            {
                "artifactSha256": manifest["model"]["artifact"]["sha256"],
                "candidateId": manifest["candidateId"],
                "fixtureCount": len(manifest["fixtures"]),
                "passed": True,
                "targetAccepted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
