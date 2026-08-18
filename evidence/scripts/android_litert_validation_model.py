#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


CONTRACT_ID = "DC-RFB-ANDROID-VAL-X64-LITERT-PRODUCER-01"
CONTRACT_VERSION = 1
CONTRACT_HASH = "sha256:7d5f2f46a7e2acdcb97bea3adf2ae5c9f3a2d0ae711dd91d7c3f06ade6298c49"
EXPECTED_BASE_COMMIT = "ff31c6523fea98e929ac736feec06735a03565ac"
EXPECTED_MODEL_BYTES = 12841227
EXPECTED_MODEL_SHA256 = "794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1"
EXPECTED_ARTIFACT_MANIFEST_SHA256 = "f1f95bac2006cd02e364123ab9e7556cc331e6d20f49006c50fcebd15d0c4881"
EXPECTED_ONNX_SHA256 = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
EXPECTED_SOURCE_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_RUNTIME_VERSION = "2.1.6"
MODEL_PATH = "models/yolov8n-fp32.tflite"
MANIFEST_PATH = "evidence/conversions/android-litert-validation-model-manifest.json"
REPORT_PATH = "evidence/reports/android-litert-validation-model-report.json"
MANIFEST_SCHEMA_PATH = "evidence/schemas/android-litert-validation-model-manifest.schema.json"
REPORT_SCHEMA_PATH = "evidence/schemas/android-litert-validation-model-report.schema.json"
TRACE_PATH = "evidence/requirements/android-litert-validation-model-trace.json"
GATE_KEYS = [
    "AC-VAL-X64-1",
    "AC-VAL-X64-2",
    "AC-VAL-X64-3",
    "AC-VAL-X64-4",
    "INV-VAL-X64-1",
    "INV-VAL-X64-2",
    "INV-VAL-X64-3",
    "EX-VAL-X64-1",
    "CON-VAL-X64-1",
    "AUTH-VAL-X64-1",
]
EXPECTED_COMPARISON = {
    "artifactSha256Equal": True,
    "fixtureRawAndDecodedEqual": True,
    "goldenReportDigestEqual": True,
    "ioMetadataEqual": True,
    "manifestDigestEqual": True,
    "trackedWorktreeUnchanged": True,
}
PRESERVED_EVIDENCE = {
    "evidence/conversions/android-litert-candidate-manifest.json": "365567e382a6edfb590ee9caae4dc290d4d320e2c2151b064761e7a7770b8ed5",
    "evidence/reports/android-litert-candidate-report.json": "f18dca9363ae30fc76888ee290d8da6d27677776834998dffecb94d523daeab7",
    "evidence/reports/android-litert-device-validation-report.json": "b9aa52ce8e046f67cca266b0ad90763f83cec3a692b81d0318ca0a94b460098c",
    "evidence/requirements/android-litert-device-validation-trace.json": "9b79186904a78033baf1a5cda71895950ea7772c6199b22c77cb9f609b6cf6c2",
    "evidence/reports/litert-golden-report.json": "071f418b79c0612e00bcea97aa7785cc4e90e4aeb2356884703e666c4bfb9e28",
    "evidence/conversions/litert-artifact-manifest.json": EXPECTED_ARTIFACT_MANIFEST_SHA256,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def file_identity(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    return {"bytes": path.stat().st_size, "path": relative_path, "sha256": sha256(path)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def git_tracked(root: Path, relative_path: str) -> bool:
    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=root,
        capture_output=True,
        text=True,
    ).returncode == 0


def validate_schema_document(root: Path, relative_path: str, expected_title: str) -> None:
    schema = load_json(root / relative_path)
    require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", f"schema draft mismatch: {relative_path}")
    require(schema.get("title") == expected_title, f"schema title mismatch: {relative_path}")
    require(schema.get("type") == "object", f"schema root type mismatch: {relative_path}")
    require(schema.get("additionalProperties") is False, f"schema must reject unknown root fields: {relative_path}")


def validate_replay(root: Path, replay_root: Path) -> dict[str, object]:
    replay = load_json(replay_root / "litert-replay.json")
    require(replay.get("schemaVersion") == 1, "replay schemaVersion mismatch")
    require(replay.get("comparison") == EXPECTED_COMPARISON, "replay comparison mismatch")
    rounds = replay.get("rounds", [])
    require([item.get("round") for item in rounds] == [1, 2], "exactly two ordered replay rounds are required")
    for item in rounds:
        require(item.get("artifact") == {"bytes": EXPECTED_MODEL_BYTES, "sha256": EXPECTED_MODEL_SHA256}, "replay model identity mismatch")
        require(item.get("artifactManifest", {}).get("sha256") == EXPECTED_ARTIFACT_MANIFEST_SHA256, "replay artifact manifest mismatch")
        require(item.get("conversion", {}).get("exitCode") == 0, "replay conversion failed")
        require(item.get("validation", {}).get("exitCode") == 0, "replay golden validation failed")
        require(item.get("webReference", {}).get("exitCode") == 0, "replay Web reference failed")
        summary = item.get("goldenReport", {}).get("summary", {})
        require(summary.get("fixtureCount") == 5, "replay fixture count mismatch")
        require(summary.get("allFinite") is True, "replay produced non-finite values")
        require(summary.get("allShapesMatched") is True, "replay shape mismatch")
        require(summary.get("deterministic") is True, "replay runtime output is nondeterministic")
        require(summary.get("classMismatchCount") == 0, "replay class mismatch")
        require(summary.get("rawToleranceMismatchCount") == 0, "replay raw tolerance mismatch")
    round_two_model = replay_root / "round-2/conversion/yolov8n.tflite"
    require(round_two_model.stat().st_size == EXPECTED_MODEL_BYTES, "round-2 model byte count mismatch")
    require(sha256(round_two_model) == EXPECTED_MODEL_SHA256, "round-2 model SHA-256 mismatch")
    return replay


def build_manifest(root: Path, replay_root: Path, replay: dict[str, object]) -> dict[str, object]:
    artifact_manifest = load_json(root / "evidence/conversions/litert-artifact-manifest.json")
    golden = load_json(root / "evidence/reports/litert-golden-report.json")
    rounds = replay["rounds"]
    consumers = [
        {
            "abi": abi,
            "bytes": EXPECTED_MODEL_BYTES,
            "modelPath": MODEL_PATH,
            "os": "android",
            "sha256": EXPECTED_MODEL_SHA256,
            "usage": "validation-package-only",
        }
        for abi in ("x86_64", "arm64-v8a")
    ]
    return {
        "artifact": {
            "architectureNeutral": True,
            "abiSpecificBytes": False,
            "bytes": EXPECTED_MODEL_BYTES,
            "format": "TFLite FlatBuffer with deterministic Ultralytics metadata ZIP trailer",
            "path": MODEL_PATH,
            "sha256": EXPECTED_MODEL_SHA256,
            "trackedByGit": True,
        },
        "consumption": {
            "consumers": consumers,
            "sameArtifactRequired": True,
            "scope": "Android x86_64 CI and arm64-v8a device validation",
        },
        "deliveryContract": {
            "hash": CONTRACT_HASH,
            "id": CONTRACT_ID,
            "mode": "formal-acceptance",
            "version": CONTRACT_VERSION,
        },
        "ioContract": copy.deepcopy(artifact_manifest["ioContract"]),
        "provenance": {
            "base": {"commit": EXPECTED_BASE_COMMIT, "ref": "origin/main"},
            "canonicalOnnx": {
                "bytes": (root / "models/yolov8n.onnx").stat().st_size,
                "path": "models/yolov8n.onnx",
                "sha256": EXPECTED_ONNX_SHA256,
            },
            "conversion": {
                "command": "$REPO/.evidence/litert/verify-venv/bin/python evidence/scripts/run_litert_replay.py --pt $REPO/.evidence/litert/inputs/yolov8n.pt --workspace $REPO/.evidence/litert/x64-model/replay --bundle-dir $REPO/.evidence/litert/x64-model/bundle",
                "converterScript": file_identity(root, "evidence/scripts/export_litert_model.py"),
                "requirementsLock": file_identity(root, "evidence/tooling/litert-requirements.lock"),
            },
            "requirementTrace": file_identity(root, TRACE_PATH),
            "sourceCheckpoint": {
                "bytes": 6549796,
                "releaseTag": "v8.3.0",
                "sha256": EXPECTED_SOURCE_SHA256,
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            },
            "validationCommit": {
                "command": "git rev-parse HEAD",
                "kind": "commit-containing-this-manifest",
            },
        },
        "preservedEvidence": [file_identity(root, path) for path in PRESERVED_EVIDENCE],
        "releaseBoundary": {
            "apkOrAabBuilt": False,
            "artifactRedistributionAllowed": False,
            "productPackaging": "excluded",
            "releaseCandidate": False,
            "releasePublicationInput": False,
            "supported": False,
            "targetAccepted": False,
            "validationOnly": True,
            "validationPackageConsumption": "allowed",
        },
        "replay": {
            "artifactManifestSha256": [item["artifactManifest"]["sha256"] for item in rounds],
            "artifactSha256": [item["artifact"]["sha256"] for item in rounds],
            "comparison": copy.deepcopy(replay["comparison"]),
            "hostGolden": {
                "path": "evidence/reports/litert-golden-report.json",
                "sha256": sha256(root / "evidence/reports/litert-golden-report.json"),
                "summary": copy.deepcopy(golden["summary"]),
            },
            "producerCommand": "$REPO/.evidence/litert/verify-venv/bin/python evidence/scripts/android_litert_validation_model.py --replay-root $REPO/.evidence/litert/x64-model/replay --record",
            "rounds": 2,
        },
        "runtime": {
            "name": "ai-edge-litert Interpreter",
            "threads": 1,
            "version": EXPECTED_RUNTIME_VERSION,
        },
        "schemaVersion": 1,
        "schemas": {
            "manifest": file_identity(root, MANIFEST_SCHEMA_PATH),
            "report": file_identity(root, REPORT_SCHEMA_PATH),
        },
    }


def contract_evidence() -> dict[str, object]:
    common = [MODEL_PATH, MANIFEST_PATH, REPORT_PATH]
    return {
        "AC-VAL-X64-1": {"evidence": common + ["evidence/scripts/export_litert_model.py"], "outcome": "satisfied"},
        "AC-VAL-X64-2": {"evidence": common + [TRACE_PATH, MANIFEST_SCHEMA_PATH, REPORT_SCHEMA_PATH], "outcome": "satisfied"},
        "AC-VAL-X64-3": {"evidence": [REPORT_PATH, "evidence/reports/litert-golden-report.json", "evidence/scripts/test_android_litert_validation_model_guards.py"], "outcome": "satisfied"},
        "AC-VAL-X64-4": {"evidence": [MANIFEST_PATH, REPORT_PATH], "outcome": "satisfied-by-tracked-paths-and-post-push-parity"},
        "INV-VAL-X64-1": {"evidence": list(PRESERVED_EVIDENCE), "outcome": "preserved"},
        "INV-VAL-X64-2": {"evidence": [MANIFEST_PATH, REPORT_PATH], "outcome": "preserved"},
        "INV-VAL-X64-3": {"evidence": [REPORT_PATH], "outcome": "preserved-by-diff-guard"},
        "EX-VAL-X64-1": {"evidence": [REPORT_PATH], "outcome": "observed"},
        "CON-VAL-X64-1": {"evidence": [MANIFEST_PATH, "evidence/tooling/litert-requirements.lock"], "outcome": "enforced"},
        "AUTH-VAL-X64-1": {"evidence": [REPORT_PATH], "outcome": "enforced-by-publication-procedure"},
    }


def build_report(root: Path, manifest: dict[str, object], manifest_bytes: bytes) -> dict[str, object]:
    return {
        "contractToEvidence": contract_evidence(),
        "deliveryContract": copy.deepcopy(manifest["deliveryContract"]),
        "deviationLog": "none",
        "identities": {
            "manifest": {
                "bytes": len(manifest_bytes),
                "path": MANIFEST_PATH,
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            "model": copy.deepcopy(manifest["artifact"]),
            "schemas": copy.deepcopy(manifest["schemas"]),
            "validationCommit": copy.deepcopy(manifest["provenance"]["validationCommit"]),
        },
        "releaseBoundary": copy.deepcopy(manifest["releaseBoundary"]),
        "remoteParity": {
            "branch": "rfb-android-val-x64-model-01",
            "headCommand": "git ls-remote origin refs/heads/rfb-android-val-x64-model-01",
            "prHeadCommand": "gh pr view --json headRefName,headRefOid,url",
            "requiredAfterSingleAuthorizedPush": True,
            "statusAtRecord": "verify-after-single-authorized-push",
        },
        "replayCommands": [
            manifest["provenance"]["conversion"]["command"],
            manifest["replay"]["producerCommand"],
            "$REPO/.evidence/litert/verify-venv/bin/python evidence/scripts/android_litert_validation_model.py --replay-root $REPO/.evidence/litert/x64-model/replay",
            "python3 evidence/scripts/test_android_litert_validation_model_guards.py",
        ],
        "schemaVersion": 1,
        "validation": {
            "comparison": copy.deepcopy(manifest["replay"]["comparison"]),
            "fixtureCount": manifest["replay"]["hostGolden"]["summary"]["fixtureCount"],
            "negativeGuardCommand": "python3 evidence/scripts/test_android_litert_validation_model_guards.py",
            "rounds": manifest["replay"]["rounds"],
        },
    }


def validate_manifest(root: Path, manifest: dict[str, object], require_tracked: bool = True) -> None:
    require(manifest.get("schemaVersion") == 1, "manifest schemaVersion mismatch")
    require(manifest.get("deliveryContract") == {"hash": CONTRACT_HASH, "id": CONTRACT_ID, "mode": "formal-acceptance", "version": CONTRACT_VERSION}, "delivery contract mismatch")
    artifact = manifest.get("artifact", {})
    require(artifact == {
        "architectureNeutral": True,
        "abiSpecificBytes": False,
        "bytes": EXPECTED_MODEL_BYTES,
        "format": "TFLite FlatBuffer with deterministic Ultralytics metadata ZIP trailer",
        "path": MODEL_PATH,
        "sha256": EXPECTED_MODEL_SHA256,
        "trackedByGit": True,
    }, "tracked artifact contract mismatch")
    model = root / MODEL_PATH
    require(model.is_file() and not model.is_symlink(), "tracked TFLite model missing or symlinked")
    require(model.stat().st_size == EXPECTED_MODEL_BYTES, "tracked TFLite byte count mismatch")
    require(sha256(model) == EXPECTED_MODEL_SHA256, "tracked TFLite SHA-256 mismatch")
    require(model.read_bytes()[4:8] == b"TFL3", "tracked model is not a TFLite FlatBuffer")
    with zipfile.ZipFile(model) as archive:
        require(archive.namelist() == ["metadata.json"], "unexpected TFLite metadata trailer")
    if require_tracked:
        require(git_tracked(root, MODEL_PATH), "TFLite model is not tracked by Git")

    consumers = manifest.get("consumption", {}).get("consumers", [])
    require([item.get("abi") for item in consumers] == ["x86_64", "arm64-v8a"], "consumer ABI set mismatch")
    require(manifest.get("consumption", {}).get("sameArtifactRequired") is True, "same-artifact rule missing")
    for item in consumers:
        require(item.get("modelPath") == MODEL_PATH, "consumer uses an ABI-specific model path")
        require(item.get("sha256") == EXPECTED_MODEL_SHA256, "consumer model digest mismatch")
        require(item.get("bytes") == EXPECTED_MODEL_BYTES, "consumer model byte count mismatch")

    io_contract = manifest.get("ioContract", {})
    require(io_contract.get("input", {}).get("shape") == [1, 3, 640, 640], "LiteRT input shape mismatch")
    require(io_contract.get("input", {}).get("dtype") == "float32", "LiteRT input dtype mismatch")
    require(io_contract.get("output", {}).get("shape") == [1, 84, 8400], "LiteRT output shape mismatch")
    require(io_contract.get("output", {}).get("dtype") == "float32", "LiteRT output dtype mismatch")
    require(manifest.get("runtime", {}).get("version") == EXPECTED_RUNTIME_VERSION, "LiteRT runtime version mismatch")

    replay = manifest.get("replay", {})
    require(replay.get("rounds") == 2, "manifest requires two replay rounds")
    require(replay.get("artifactSha256") == [EXPECTED_MODEL_SHA256, EXPECTED_MODEL_SHA256], "manifest replay model identities mismatch")
    require(replay.get("artifactManifestSha256") == [EXPECTED_ARTIFACT_MANIFEST_SHA256, EXPECTED_ARTIFACT_MANIFEST_SHA256], "manifest replay contract identities mismatch")
    require(replay.get("comparison") == EXPECTED_COMPARISON, "manifest replay comparison mismatch")
    summary = replay.get("hostGolden", {}).get("summary", {})
    require(summary.get("fixtureCount") == 5 and summary.get("allFinite") is True, "host golden summary mismatch")
    require(summary.get("allShapesMatched") is True and summary.get("deterministic") is True, "host golden determinism mismatch")
    require(summary.get("classMismatchCount") == 0 and summary.get("rawToleranceMismatchCount") == 0, "host golden tolerance mismatch")

    provenance = manifest.get("provenance", {})
    require(provenance.get("base") == {"commit": EXPECTED_BASE_COMMIT, "ref": "origin/main"}, "origin/main base identity mismatch")
    require(provenance.get("canonicalOnnx", {}).get("sha256") == EXPECTED_ONNX_SHA256, "canonical ONNX digest mismatch")
    require(sha256(root / "models/yolov8n.onnx") == EXPECTED_ONNX_SHA256, "canonical ONNX file drift")
    require(provenance.get("sourceCheckpoint", {}).get("sha256") == EXPECTED_SOURCE_SHA256, "source checkpoint digest mismatch")
    require(provenance.get("conversion", {}).get("requirementsLock") == file_identity(root, "evidence/tooling/litert-requirements.lock"), "LiteRT lock identity mismatch")
    require(provenance.get("conversion", {}).get("converterScript") == file_identity(root, "evidence/scripts/export_litert_model.py"), "converter script identity mismatch")
    require(provenance.get("requirementTrace") == file_identity(root, TRACE_PATH), "requirement trace identity mismatch")

    preserved = manifest.get("preservedEvidence", [])
    require([item.get("path") for item in preserved] == list(PRESERVED_EVIDENCE), "preserved evidence set mismatch")
    for item in preserved:
        require(item == file_identity(root, item["path"]), f"preserved evidence drift: {item['path']}")
        require(item["sha256"] == PRESERVED_EVIDENCE[item["path"]], f"accepted evidence identity changed: {item['path']}")

    boundary = manifest.get("releaseBoundary", {})
    require(boundary == {
        "apkOrAabBuilt": False,
        "artifactRedistributionAllowed": False,
        "productPackaging": "excluded",
        "releaseCandidate": False,
        "releasePublicationInput": False,
        "supported": False,
        "targetAccepted": False,
        "validationOnly": True,
        "validationPackageConsumption": "allowed",
    }, "validation-only release boundary mismatch")
    require(manifest.get("schemas", {}).get("manifest") == file_identity(root, MANIFEST_SCHEMA_PATH), "manifest schema identity mismatch")
    require(manifest.get("schemas", {}).get("report") == file_identity(root, REPORT_SCHEMA_PATH), "report schema identity mismatch")

    trace = load_json(root / TRACE_PATH)
    require(trace.get("contractId") == CONTRACT_ID, "requirement trace contract ID mismatch")
    require(trace.get("contractVersion") == CONTRACT_VERSION, "requirement trace contract version mismatch")
    require(trace.get("contractHash") == CONTRACT_HASH, "requirement trace contract hash mismatch")
    require(trace.get("modelPath") == MODEL_PATH, "requirement trace model path mismatch")
    require(trace.get("manifestPath") == MANIFEST_PATH, "requirement trace manifest path mismatch")
    require(trace.get("reportPath") == REPORT_PATH, "requirement trace report path mismatch")
    require(trace.get("contractToEvidenceKeys") == GATE_KEYS, "requirement trace gate coverage mismatch")


def validate_report(root: Path, report: dict[str, object], manifest: dict[str, object]) -> None:
    require(report.get("schemaVersion") == 1, "report schemaVersion mismatch")
    require(report.get("deliveryContract") == manifest.get("deliveryContract"), "report delivery contract drift")
    require(report.get("deviationLog") == "none", "deviation log must be none")
    evidence_keys = list(report.get("contractToEvidence", {}))
    require(len(evidence_keys) == len(GATE_KEYS) and set(evidence_keys) == set(GATE_KEYS), "Contract-to-Evidence Matrix keys mismatch")
    require(report.get("contractToEvidence") == contract_evidence(), "Contract-to-Evidence Matrix content drift")
    require(report.get("identities", {}).get("model") == manifest.get("artifact"), "report model identity drift")
    require(report.get("identities", {}).get("schemas") == manifest.get("schemas"), "report schema identities drift")
    require(report.get("identities", {}).get("manifest") == file_identity(root, MANIFEST_PATH), "report manifest identity mismatch")
    require(report.get("releaseBoundary") == manifest.get("releaseBoundary"), "report release boundary drift")
    require(report.get("validation", {}).get("rounds") == 2, "report replay count mismatch")
    require(report.get("validation", {}).get("comparison") == EXPECTED_COMPARISON, "report replay comparison mismatch")
    remote = report.get("remoteParity", {})
    require(remote.get("branch") == "rfb-android-val-x64-model-01", "remote parity branch mismatch")
    require(remote.get("requiredAfterSingleAuthorizedPush") is True, "remote parity requirement missing")
    require(remote.get("statusAtRecord") == "verify-after-single-authorized-push", "report must not pre-claim remote parity")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=".evidence/litert/x64-model/replay", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    replay_root = args.replay_root if args.replay_root.is_absolute() else root / args.replay_root
    replay = validate_replay(root, replay_root)
    validate_schema_document(root, MANIFEST_SCHEMA_PATH, "Android LiteRT architecture-neutral validation model manifest")
    validate_schema_document(root, REPORT_SCHEMA_PATH, "Android LiteRT architecture-neutral validation model report")

    if args.record:
        source_model = replay_root / "round-2/conversion/yolov8n.tflite"
        tracked_model = root / MODEL_PATH
        tracked_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_model, tracked_model)
        manifest = build_manifest(root, replay_root, replay)
        manifest_bytes = json_bytes(manifest)
        (root / MANIFEST_PATH).write_bytes(manifest_bytes)
        report = build_report(root, manifest, manifest_bytes)
        (root / REPORT_PATH).write_bytes(json_bytes(report))
    else:
        manifest = load_json(root / MANIFEST_PATH)
        report = load_json(root / REPORT_PATH)

    validate_manifest(root, manifest, require_tracked=not args.record)
    validate_report(root, report, manifest)
    print(json.dumps({
        "consumers": [item["abi"] for item in manifest["consumption"]["consumers"]],
        "manifestPath": MANIFEST_PATH,
        "model": {"bytes": EXPECTED_MODEL_BYTES, "path": MODEL_PATH, "sha256": EXPECTED_MODEL_SHA256},
        "passed": True,
        "recorded": args.record,
        "reportPath": REPORT_PATH,
        "rounds": 2,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
