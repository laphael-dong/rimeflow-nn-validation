#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED = {
    "candidateManifest": "365567e382a6edfb590ee9caae4dc290d4d320e2c2151b064761e7a7770b8ed5",
    "candidateReport": "f18dca9363ae30fc76888ee290d8da6d27677776834998dffecb94d523daeab7",
    "deviceEvidenceManifest": "bc7ab86c8afc89b019ed3831fdad0a23ec9b2d15d8ec2c9adba6d765a5b322ff",
    "deviceExecutionSummary": "0f45ab0fbe30d0017b6c70c5e43e04a17838d89dd947ccecbf7a0c0f8f2337db",
    "deviceWorkerReport": "5cb262c23a5b6ee03b278de1aed7d1a28de3b25392f6f680e38936845fe92081",
    "faultSummary": "3c23a7fc9747a05f776bd304ae979227e9d7243ee5ec6a66de6c210bcc8e307e",
    "frozenReference": "114d7ab29afa5bf9b6246dfd6f1cff37ee3d4fc8212b355d2fb3ac9122f9ce68",
    "ioContract": "a1b655d7aebf88d953de8f1cb05845673176a31ebe979e2b9020aa1a443f633b",
    "outputAnalysis": "16020a6bcc013c1c0f5611021ac87097befcdea68b5e269fc619580ff5bf8b63",
    "packageLoad": "08a2477b20c94c154a8c2a205ebf83a4f04da4c718614e84f6479954f9052a30",
    "performance": "d73553ac1ca4051c80b4522330e0f221a0cb5cdad668fec7b8e5e35824b44899",
    "requirementTrace": "9b79186904a78033baf1a5cda71895950ea7772c6199b22c77cb9f609b6cf6c2",
    "runnerReport": "1320cfb3e2ea321f4df3d487dc2f96b9e6d81aceb1a33dbdd26e51332367d64a",
}
EXPECTED_ARTIFACT_SHA256 = "794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1"
EXPECTED_BUNDLE_ID = "sha256-0c309e3f63aa69920a1ac55a42b0a5211771671bf7df0fa180205550ce535866"
EXPECTED_DEVICE_DISPATCH = "ctx_67518c7c38fb"
EXPECTED_DEVICE_TASK = "task_e723335e3bb8"
EXPECTED_FIXTURE_IDS = [
    "no-detection",
    "single-target",
    "multi-class",
    "boundary-box",
    "extreme-aspect",
]
EXPECTED_RUN_ID = "run_88ee582c93cd"
EXPECTED_RUNNER_COMMIT = "52c85ec737dc4b4ab482f6c2654f914150c4dfae"
EXPECTED_TOLERANCE_SHA256 = "8d2ac80770ce336df0e1cded48bb0c225c808f5f8e0ee428b2821952edb6ba80"
EXPECTED_VALIDATION_COMMIT = "c7bca1cdc1235bbb294c0e40cdfcfacfdc95a660"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _checked_json(path: Path, expected_sha256: str) -> dict[str, object]:
    actual = sha256(path)
    _require(actual == expected_sha256, f"SHA-256 mismatch for {path}: {actual}")
    return _load_json(path)


def _golden_module(root: Path):
    source = root / "evidence/scripts/validate_litert_golden.py"
    spec = importlib.util.spec_from_file_location("rimeflow_litert_golden", source)
    _require(spec is not None and spec.loader is not None, "cannot load frozen LiteRT comparison code")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_evidence_manifest(device_root: Path, manifest: dict[str, object]) -> dict[str, object]:
    files = manifest.get("files")
    _require(isinstance(files, list), "device evidence manifest files must be an array")
    _require(manifest.get("fileCount") == 164 and len(files) == 164, "device evidence file count mismatch")
    listed: set[str] = set()
    for entry in files:
        _require(isinstance(entry, dict), "device evidence manifest entry must be an object")
        relative = entry.get("path")
        _require(isinstance(relative, str), "device evidence manifest path must be a string")
        parsed = PurePosixPath(relative)
        _require(not parsed.is_absolute() and ".." not in parsed.parts, f"unsafe evidence path: {relative}")
        _require(relative not in listed, f"duplicate evidence path: {relative}")
        _require(relative != "evidence-manifest.json", "evidence manifest must exclude itself")
        listed.add(relative)
        evidence_path = device_root / Path(*parsed.parts)
        _require(evidence_path.is_file() and not evidence_path.is_symlink(), f"invalid evidence file: {relative}")
        _require(evidence_path.stat().st_size == entry.get("bytes"), f"evidence byte count mismatch: {relative}")
        _require(sha256(evidence_path) == entry.get("sha256"), f"evidence SHA-256 mismatch: {relative}")

    actual = {
        path.relative_to(device_root).as_posix()
        for path in device_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "evidence-manifest.json"
    }
    _require(actual == listed, "device evidence manifest coverage mismatch")
    return {
        "actualFileCount": len(actual),
        "declaredFileCount": manifest["fileCount"],
        "duplicateEntryCount": len(files) - len(listed),
        "everyByteCountMatched": True,
        "everySha256Matched": True,
        "extraFileCount": len(actual - listed),
        "missingFileCount": len(listed - actual),
        "selfEntryCount": int("evidence-manifest.json" in listed),
        "status": "passed",
    }


def compare_fixtures(
    root: Path,
    device_root: Path,
    web_reference_dir: Path,
    frozen: dict[str, object],
    output_analysis: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    golden = _golden_module(root)
    web_manifest = _load_json(web_reference_dir / "manifest.json")
    fixtures = []
    failures = []
    for frozen_fixture in frozen["fixtures"]:
        fixture_id = frozen_fixture["id"]
        _require(fixture_id in EXPECTED_FIXTURE_IDS, f"unexpected frozen fixture: {fixture_id}")
        web_fixture = next(item for item in web_manifest["fixtures"] if item["id"] == fixture_id)
        reference_path = web_reference_dir / fixture_id / "raw.f32le"
        expected_reference_sha256 = frozen_fixture["runs"][0]["rawTensor"]["sha256Float32Le"]
        _require(sha256(reference_path) == expected_reference_sha256, f"{fixture_id}: frozen raw digest mismatch")
        reference = np.fromfile(reference_path, dtype="<f4").reshape(golden.EXPECTED_OUTPUT_SHAPE)
        _require(bool(np.isfinite(reference).all()), f"{fixture_id}: frozen raw contains non-finite value")

        mapped_runs = []
        observed_digests = []
        for run in (1, 2):
            observation = next(
                item
                for item in output_analysis["frozenRawDigestObservations"]
                if item["fixtureId"] == fixture_id and item["run"] == run
            )
            output_path = device_root / f"pulled/outputs/{fixture_id}/run-{run}.bin"
            observed_sha256 = sha256(output_path)
            _require(observed_sha256 == observation["sha256"], f"{fixture_id} run {run}: digest mismatch")
            runtime = np.fromfile(output_path, dtype="<f4").reshape(golden.EXPECTED_OUTPUT_SHAPE)
            _require(bool(np.isfinite(runtime).all()), f"{fixture_id} run {run}: non-finite output")
            mapped = runtime.astype(np.float32, copy=True)
            mapped[:, :4, :] *= np.float32(640.0)
            observed_digests.append(observed_sha256)
            mapped_runs.append(mapped)

        deterministic = bool(np.array_equal(mapped_runs[0], mapped_runs[1]))
        raw = golden.raw_comparison(mapped_runs[0], reference, frozen["tolerances"])
        before_nms, decoded = golden.decode(
            mapped_runs[0],
            int(web_fixture["image"]["width"]),
            int(web_fixture["image"]["height"]),
            frozen_fixture["preprocessing"],
        )
        decoded_comparison = golden.decoded_comparison(decoded, frozen_fixture["runs"][0]["decoded"])
        mismatches = []
        if not deterministic:
            mismatches.append("device-repetition")
        if raw["mismatchCount"]:
            mismatches.append("raw-tolerance")
        if (
            decoded_comparison["classMismatchCount"]
            or decoded_comparison["anchorMismatchCount"]
            or decoded_comparison["countMismatch"]
        ):
            mismatches.append("decoded-identity")
        if decoded_comparison["maximumConfidenceAbsolute"] > frozen["tolerances"]["confidenceAbsolute"]:
            mismatches.append("confidence")
        minimum_iou = decoded_comparison["minimumBboxIou"]
        if minimum_iou is not None and minimum_iou < frozen["tolerances"]["boxIouMinimum"]:
            mismatches.append("bbox-iou")
        if decoded_comparison["maximumDecodedBboxAbsolute"] > frozen["tolerances"]["decodedBoxAbsolute"]:
            mismatches.append("decoded-bbox")
        failures.extend(f"{fixture_id}:{reason}" for reason in mismatches)
        fixtures.append(
            {
                "decodedComparison": decoded_comparison,
                "decodedCount": len(decoded),
                "determinism": {
                    "exact": deterministic,
                    "rawDeviceSha256": observed_digests,
                    "runs": 2,
                },
                "fixtureId": fixture_id,
                "mappedRawSha256": golden.tensor_sha256(mapped_runs[0]),
                "mismatches": mismatches,
                "rawComparison": raw,
                "webDecodedCount": len(frozen_fixture["runs"][0]["decoded"]),
                "webRawSha256": expected_reference_sha256,
                "beforeNmsCount": len(before_nms),
            }
        )

    _require([item["fixtureId"] for item in fixtures] == EXPECTED_FIXTURE_IDS, "fixture order or coverage mismatch")
    ious = [item["decodedComparison"]["minimumBboxIou"] for item in fixtures]
    ious = [value for value in ious if value is not None]
    summary = {
        "anchorMismatchCount": sum(item["decodedComparison"]["anchorMismatchCount"] for item in fixtures),
        "classMismatchCount": sum(item["decodedComparison"]["classMismatchCount"] for item in fixtures),
        "countMismatchCount": sum(int(item["decodedComparison"]["countMismatch"]) for item in fixtures),
        "deterministic": all(item["determinism"]["exact"] for item in fixtures),
        "fixtureCount": len(fixtures),
        "maximumConfidenceAbsolute": max(item["decodedComparison"]["maximumConfidenceAbsolute"] for item in fixtures),
        "maximumDecodedBboxAbsolute": max(
            item["decodedComparison"]["maximumDecodedBboxAbsolute"] for item in fixtures
        ),
        "maximumRawAbsolute": max(item["rawComparison"]["maxAbsolute"] for item in fixtures),
        "minimumBboxIou": min(ious),
        "passed": not failures,
        "rawMismatchCount": sum(item["rawComparison"]["mismatchCount"] for item in fixtures),
    }
    _require(not failures, "frozen golden comparison failed: " + ", ".join(failures))
    return fixtures, summary


def build_report(root: Path, device_root: Path, web_reference_dir: Path) -> dict[str, object]:
    candidate_manifest_path = root / "evidence/conversions/android-litert-candidate-manifest.json"
    candidate_report_path = root / "evidence/reports/android-litert-candidate-report.json"
    frozen_path = root / "evidence/golden/web-reference.json"
    candidate_manifest = _checked_json(candidate_manifest_path, EXPECTED["candidateManifest"])
    candidate_report = _checked_json(candidate_report_path, EXPECTED["candidateReport"])
    frozen = _checked_json(frozen_path, EXPECTED["frozenReference"])
    evidence_manifest = _checked_json(device_root / "evidence-manifest.json", EXPECTED["deviceEvidenceManifest"])
    execution = _checked_json(device_root / "execution-summary.json", EXPECTED["deviceExecutionSummary"])
    io_contract = _checked_json(device_root / "io-contract.json", EXPECTED["ioContract"])
    output_analysis = _checked_json(device_root / "output-analysis.json", EXPECTED["outputAnalysis"])
    faults = _checked_json(device_root / "fault-fallback-summary.json", EXPECTED["faultSummary"])
    performance = _checked_json(device_root / "performance-summary.json", EXPECTED["performance"])
    package_load = _checked_json(device_root / "package-load-proof.json", EXPECTED["packageLoad"])
    requirement_trace_path = root / "evidence/requirements/android-litert-device-validation-trace.json"
    requirement_trace = _checked_json(requirement_trace_path, EXPECTED["requirementTrace"])
    _require(sha256(device_root / "worker-report.md") == EXPECTED["deviceWorkerReport"], "device worker report mismatch")
    _require(sha256(device_root / "pulled/runner-report.json") == EXPECTED["runnerReport"], "runner report mismatch")
    manifest_parity = verify_evidence_manifest(device_root, evidence_manifest)

    _require(execution["taskId"] == EXPECTED_DEVICE_TASK, "device task identity mismatch")
    _require(execution["dispatchId"] == EXPECTED_DEVICE_DISPATCH, "device dispatch identity mismatch")
    _require(execution["runId"] == EXPECTED_RUN_ID, "Orca Run identity mismatch")
    accepted = execution["acceptedBundleIdentity"]
    _require(accepted["bundleId"] == EXPECTED_BUNDLE_ID, "accepted bundle identity mismatch")
    _require(accepted["tfliteSha256"] == EXPECTED_ARTIFACT_SHA256, "accepted TFLite identity mismatch")
    _require(accepted["candidateManifestSha256"] == EXPECTED["candidateManifest"], "candidate manifest continuity mismatch")
    _require(accepted["validationCommit"] == EXPECTED_VALIDATION_COMMIT, "validation commit mismatch")
    _require(accepted["runnerCommit"] == EXPECTED_RUNNER_COMMIT, "runner commit mismatch")
    _require(candidate_manifest["model"]["artifact"]["sha256"] == EXPECTED_ARTIFACT_SHA256, "candidate artifact mismatch")
    _require(candidate_report["status"]["hostGoldenPassed"] is True, "candidate host golden did not pass")
    _require(candidate_report["status"]["androidTargetAccepted"] is False, "candidate report overclaims Android")
    _require(canonical_json_sha256(frozen["tolerances"]) == EXPECTED_TOLERANCE_SHA256, "frozen tolerance mismatch")

    expected_input = candidate_manifest["contract"]["io"]["input"]
    expected_output = candidate_manifest["contract"]["io"]["output"]
    _require(io_contract["input"]["role"] == "image", "device input role mismatch")
    _require(io_contract["output"]["role"] == candidate_manifest["contract"]["io"]["outputRole"], "device output role mismatch")
    _require(io_contract["input"]["shape"] == expected_input["shape"], "device input shape mismatch")
    _require(io_contract["output"]["shape"] == expected_output["shape"], "device output shape mismatch")
    _require(io_contract["input"]["dtype"] == "f32" and expected_input["dtype"] == "float32", "device input dtype mismatch")
    _require(io_contract["output"]["dtype"] == "f32" and expected_output["dtype"] == "float32", "device output dtype mismatch")
    _require(io_contract["input"]["layout"] == expected_input["layout"], "device input layout mismatch")
    _require(expected_output["layout"] == "N_ATTRIBUTES_ANCHORS", "frozen output semantic layout mismatch")
    _require(io_contract["output"]["layout"] == "NCHW", "device output rank layout mismatch")
    _require(io_contract["input"]["namesAreDistinct"] is True, "input binding/runtime names collapsed")
    _require(io_contract["output"]["namesAreDistinct"] is True, "output binding/runtime names collapsed")
    _require(io_contract["output"]["allFinite"] is True, "device output is not finite")
    _require(io_contract["executionPath"] == ["AndroidLiteRtV2Factory", "CompiledModel", "RuntimeBackend::infer"], "execution path mismatch")

    fixtures, comparison_summary = compare_fixtures(
        root, device_root, web_reference_dir, frozen, output_analysis
    )
    _require(faults["initializationFault"]["fallbackContract"]["eligible"] is True, "initialization fault not fallback-eligible")
    _require(faults["postReadyFault"]["errorContract"]["expected"] == "InferenceError", "post-ready error contract mismatch")
    _require(faults["postReadyFault"]["backendSwitchObserved"] is False, "post-ready backend switch observed")
    _require(faults["postReadyFault"]["backendRecreationObserved"] is False, "post-ready backend recreation observed")
    exception = execution["comparatorException"]
    _require(exception["approved"] is True and exception["notPerformanceApproval"] is True, "performance exception boundary mismatch")
    _require(exception["expiry"] == "independent Validation report completion", "performance exception expiry mismatch")
    _require(package_load["status"] == "passed" and package_load["runtimeShaMatches"] is True, "package-load proof failed")
    _require(requirement_trace["taskId"] == "RFB-ANDROID-VAL-REPORT-01", "requirement trace identity mismatch")

    schema_path = root / "evidence/schemas/android-litert-device-validation-report.schema.json"
    return {
        "candidate": {
            "candidateId": candidate_manifest["candidateId"],
            "manifestSha256": EXPECTED["candidateManifest"],
            "reportSha256": EXPECTED["candidateReport"],
            "runnerBundleId": EXPECTED_BUNDLE_ID,
            "runnerCommit": EXPECTED_RUNNER_COMMIT,
            "sourceCheckpointSha256": candidate_manifest["model"]["sourceCheckpoint"]["sha256"],
            "tfliteSha256": EXPECTED_ARTIFACT_SHA256,
            "validationCommit": EXPECTED_VALIDATION_COMMIT,
        },
        "deviceEvidence": {
            "dispatchId": EXPECTED_DEVICE_DISPATCH,
            "evidenceManifest": {
                "path": "evidence-manifest.json",
                "sha256": EXPECTED["deviceEvidenceManifest"],
            },
            "executionSummary": {
                "path": "execution-summary.json",
                "sha256": EXPECTED["deviceExecutionSummary"],
            },
            "immutableRootId": ".device-evidence/rfb-android-device-accept-02",
            "manifestParity": manifest_parity,
            "runId": EXPECTED_RUN_ID,
            "taskId": EXPECTED_DEVICE_TASK,
            "workerReport": {
                "path": "worker-report.md",
                "sha256": EXPECTED["deviceWorkerReport"],
            },
        },
        "faultContract": {
            "initialization": {
                "contractOutcome": "UseWebFallback-eligible",
                "fallbackExecutionClaimed": False,
                "failureStage": faults["initializationFault"]["failureStage"],
                "interpretation": faults["initializationFault"]["fallbackContract"]["rawObservation"],
                "resolvedBackend": False,
                "selectionCode": faults["initializationFault"]["selectionCode"],
            },
            "postReady": {
                "backendRecreationObserved": faults["postReadyFault"]["backendRecreationObserved"],
                "backendSwitchObserved": faults["postReadyFault"]["backendSwitchObserved"],
                "contractOutcome": "InferenceError-style",
                "failureStage": faults["postReadyFault"]["failureStage"],
                "resolvedBackendRetained": True,
                "selectionCode": faults["postReadyFault"]["selectionCode"],
            },
            "source": {"path": "fault-fallback-summary.json", "sha256": EXPECTED["faultSummary"]},
            "status": "observations-match-contract-boundaries",
        },
        "goldenComparison": {
            "coordinateMapping": candidate_manifest["contract"]["mapping"]["outputCoordinates"],
            "fixtures": fixtures,
            "frozenReference": {"path": "evidence/golden/web-reference.json", "sha256": EXPECTED["frozenReference"]},
            "summary": comparison_summary,
            "toleranceCanonicalSha256": EXPECTED_TOLERANCE_SHA256,
            "tolerances": frozen["tolerances"],
        },
        "ioContract": {
            "accelerator": io_contract["accelerator"],
            "backendKind": io_contract["backendKind"],
            "executionPath": io_contract["executionPath"],
            "input": io_contract["input"],
            "output": io_contract["output"],
            "outputLayoutInterpretation": {
                "frozenSemanticLayout": expected_output["layout"],
                "identityMapping": candidate_manifest["contract"]["mapping"]["outputLayout"],
                "runnerRankLayout": io_contract["output"]["layout"],
                "shape": expected_output["shape"],
            },
            "provider": io_contract["provider"],
            "runtimeVersion": io_contract["runtimeVersion"],
            "source": {"path": "io-contract.json", "sha256": EXPECTED["ioContract"]},
            "status": "matched",
        },
        "ownership": {
            "finalCloseOwner": "RFB-ANDROID-BASE-CLOSE-01",
            "platformMatrixChanged": False,
            "supportedDecisionMade": False,
        },
        "performance": {
            "absoluteMetrics": {
                "bundleBytes": performance["bundleBytes"],
                "coldInferenceMs": performance["coldInferenceMs"],
                "initializationMs": performance["initializationMs"],
                "peakProcessRssBytes": performance["peakProcessRssBytes"],
                "resolvedInitializationMs": execution["performance"]["resolvedInitializationMs"],
                "warmInferenceMs": performance["summary"],
                "warmupRuns": performance["warmupRuns"],
                "measuredSampleCount": performance["measuredSampleCount"],
            },
            "exception": {
                "approved": exception["approved"],
                "deviceSerial": exception["device"]["serial"],
                "expiry": exception["expiry"],
                "kind": exception["kind"],
                "notGoldenApproval": exception["notGoldenApproval"],
                "notPerformanceApproval": exception["notPerformanceApproval"],
                "notSupportedStatus": exception["notSupportedStatus"],
                "reason": exception["reason"],
                "scope": exception["scope"],
            },
            "outcome": "exception-recorded-no-comparative-pass",
            "packageLoad": {
                "accelerator": package_load["accelerator"],
                "loadedLibraryCount": package_load["loadedLibraryCount"],
                "loadedRuntimeSha256": package_load["loadedRuntime"][0]["sha256"],
                "runtimeShaMatches": package_load["runtimeShaMatches"],
                "status": package_load["status"],
            },
            "performancePassed": False,
            "samePhoneBaselineAvailable": False,
            "source": {"path": "performance-summary.json", "sha256": EXPECTED["performance"]},
        },
        "reportId": "RFB-ANDROID-VAL-REPORT-01",
        "requirementTrace": {
            "path": "evidence/requirements/android-litert-device-validation-trace.json",
            "sha256": EXPECTED["requirementTrace"],
        },
        "schema": {
            "path": "evidence/schemas/android-litert-device-validation-report.schema.json",
            "sha256": sha256(schema_path),
        },
        "schemaVersion": 1,
        "status": {
            "finalPlatformClose": False,
            "performancePassed": False,
            "supported": False,
            "validationComparisonPassed": True,
            "value": "device-golden-verified-pending-base-close",
        },
    }


def validate_report(report: dict[str, object]) -> None:
    _require(report.get("schemaVersion") == 1, "report schemaVersion mismatch")
    _require(report.get("reportId") == "RFB-ANDROID-VAL-REPORT-01", "report identity mismatch")
    _require(report.get("requirementTrace", {}).get("sha256") == EXPECTED["requirementTrace"], "report requirement trace drift")
    status = report.get("status", {})
    _require(status.get("validationComparisonPassed") is True, "validation comparison is not passed")
    _require(status.get("performancePassed") is False, "report overclaims performance pass")
    _require(status.get("supported") is False, "report overclaims supported status")
    _require(status.get("finalPlatformClose") is False, "report overclaims platform close")
    candidate = report.get("candidate", {})
    _require(candidate.get("manifestSha256") == EXPECTED["candidateManifest"], "report candidate manifest drift")
    _require(candidate.get("tfliteSha256") == EXPECTED_ARTIFACT_SHA256, "report TFLite identity drift")
    golden = report.get("goldenComparison", {})
    _require(golden.get("toleranceCanonicalSha256") == EXPECTED_TOLERANCE_SHA256, "report tolerance drift")
    _require(canonical_json_sha256(golden.get("tolerances")) == EXPECTED_TOLERANCE_SHA256, "report tolerance bytes drift")
    _require(golden.get("summary", {}).get("passed") is True, "report golden outcome is not passed")
    _require(golden.get("summary", {}).get("rawMismatchCount") == 0, "report hides raw mismatches")
    _require(len(golden.get("fixtures", [])) == 5, "report fixture count mismatch")
    _require(all(not item.get("mismatches") for item in golden["fixtures"]), "report contains fixture mismatch")
    _require(report.get("ioContract", {}).get("status") == "matched", "report I/O contract is not matched")
    _require(report.get("faultContract", {}).get("initialization", {}).get("fallbackExecutionClaimed") is False, "report overclaims fallback execution")
    performance = report.get("performance", {})
    _require(performance.get("outcome") == "exception-recorded-no-comparative-pass", "performance exception outcome drift")
    _require(performance.get("performancePassed") is False, "performance section overclaims pass")
    _require(performance.get("samePhoneBaselineAvailable") is False, "report invents comparator baseline")
    ownership = report.get("ownership", {})
    _require(ownership.get("finalCloseOwner") == "RFB-ANDROID-BASE-CLOSE-01", "final close ownership drift")
    _require(ownership.get("supportedDecisionMade") is False, "report makes supported decision")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-evidence-root", required=True, type=Path)
    parser.add_argument("--web-reference-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        default=Path("evidence/reports/android-litert-device-validation-report.json"),
        type=Path,
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    generated = build_report(root, args.device_evidence_root, args.web_reference_dir)
    validate_report(generated)
    payload = json_bytes(generated)
    if args.write:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(payload)
    else:
        _require(args.report.read_bytes() == payload, "tracked Android device validation report is stale")
    print(
        json.dumps(
            {
                "fixtureCount": generated["goldenComparison"]["summary"]["fixtureCount"],
                "reportSha256": hashlib.sha256(payload).hexdigest(),
                "status": generated["status"]["value"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
