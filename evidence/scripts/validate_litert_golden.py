#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


EXPECTED_RUNTIME_VERSION = "2.1.6"
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPE = [1, 84, 8400]
NEAR_ZERO_THRESHOLD = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def tensor_summary(values: np.ndarray) -> dict[str, object]:
    return {
        "elementCount": int(values.size),
        "finiteCount": int(np.count_nonzero(np.isfinite(values))),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "min": float(np.min(values)),
        "sha256Float32Le": tensor_sha256(values),
    }


def tensor_detail(detail: dict[str, object], layout: str) -> dict[str, object]:
    quantization = detail["quantization"]
    parameters = detail["quantization_parameters"]
    return {
        "dtype": detail["dtype"].__name__,
        "index": int(detail["index"]),
        "layout": layout,
        "name": detail["name"],
        "quantization": {
            "scale": float(quantization[0]),
            "zeroPoint": int(quantization[1]),
        },
        "quantizationParameters": {
            "quantizedDimension": int(parameters["quantized_dimension"]),
            "scales": parameters["scales"].tolist(),
            "zeroPoints": parameters["zero_points"].tolist(),
        },
        "shape": detail["shape"].tolist(),
        "shapeSignature": detail["shape_signature"].tolist(),
    }


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return 0.0 if union <= 0.0 else intersection / union


def decode(raw: np.ndarray, width: int, height: int, prep: dict[str, object]):
    anchors = 8400
    found = []
    for anchor in range(anchors):
        scores = raw[0, 4:, anchor]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < 0.25:
            continue
        cx = float(raw[0, 0, anchor])
        cy = float(raw[0, 1, anchor])
        box_width = float(raw[0, 2, anchor])
        box_height = float(raw[0, 3, anchor])
        scale = float(prep["scale"])
        pad_x = float(prep["padXPixels"])
        pad_y = float(prep["padYPixels"])
        clamp = lambda value: min(1.0, max(0.0, value))
        found.append(
            {
                "anchor": anchor,
                "classId": class_id,
                "score": score,
                "bbox": [
                    clamp(((cx - box_width / 2.0) - pad_x) / (width * scale)),
                    clamp(((cy - box_height / 2.0) - pad_y) / (height * scale)),
                    clamp(((cx + box_width / 2.0) - pad_x) / (width * scale)),
                    clamp(((cy + box_height / 2.0) - pad_y) / (height * scale)),
                ],
            }
        )
    found.sort(key=lambda item: (-item["score"], item["anchor"]))
    kept = []
    for candidate in found:
        if not any(iou(current["bbox"], candidate["bbox"]) > 0.45 for current in kept):
            kept.append(candidate)
    return found, kept


def raw_comparison(candidate: np.ndarray, reference: np.ndarray, tolerances: dict[str, object]):
    difference = np.abs(candidate - reference)
    reference_absolute = np.abs(reference)
    near_zero = reference_absolute <= NEAR_ZERO_THRESHOLD
    relative = np.full(reference.shape, np.nan, dtype=np.float64)
    np.divide(difference, reference_absolute, out=relative, where=~near_zero)
    mismatch = (difference > tolerances["rawTensorAbsolute"]) & (
        near_zero | (relative > tolerances["rawTensorRelative"])
    )
    max_absolute_flat = int(np.argmax(difference))
    relative_for_max = np.where(np.isfinite(relative), relative, -1.0)
    max_relative_flat = int(np.argmax(relative_for_max))
    return {
        "maxAbsolute": float(difference.flat[max_absolute_flat]),
        "maxAbsoluteLocation": [int(value) for value in np.unravel_index(max_absolute_flat, difference.shape)],
        "maxRelative": float(relative_for_max.flat[max_relative_flat]),
        "maxRelativeLocation": [int(value) for value in np.unravel_index(max_relative_flat, difference.shape)],
        "mismatchCount": int(np.count_nonzero(mismatch)),
        "nearZeroPolicy": {
            "absoluteToleranceOnly": True,
            "count": int(np.count_nonzero(near_zero)),
            "threshold": NEAR_ZERO_THRESHOLD,
        },
    }


def decoded_comparison(candidate: list[dict[str, object]], reference: list[dict[str, object]]):
    class_mismatches = abs(len(candidate) - len(reference))
    anchor_mismatches = abs(len(candidate) - len(reference))
    maximum_confidence = 0.0
    maximum_bbox_absolute = 0.0
    minimum_iou = 1.0 if candidate or reference else None
    for actual, expected in zip(candidate, reference):
        class_mismatches += int(actual["classId"] != expected["classId"])
        anchor_mismatches += int(actual["anchor"] != expected["anchor"])
        maximum_confidence = max(maximum_confidence, abs(actual["score"] - expected["score"]))
        maximum_bbox_absolute = max(
            maximum_bbox_absolute,
            *(abs(a - b) for a, b in zip(actual["bbox"], expected["bbox"])),
        )
        minimum_iou = min(minimum_iou, iou(actual["bbox"], expected["bbox"]))
    return {
        "anchorMismatchCount": anchor_mismatches,
        "classMismatchCount": class_mismatches,
        "countMismatch": len(candidate) != len(reference),
        "maximumConfidenceAbsolute": maximum_confidence,
        "maximumDecodedBboxAbsolute": maximum_bbox_absolute,
        "minimumBboxIou": minimum_iou,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--web-reference-dir", required=True, type=Path)
    parser.add_argument("--frozen-reference", default="evidence/golden/web-reference.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-artifact-sha256")
    args = parser.parse_args()

    runtime_version = version("ai-edge-litert")
    if runtime_version != EXPECTED_RUNTIME_VERSION:
        raise SystemExit(f"LiteRT runtime version mismatch: {runtime_version}")
    artifact_sha256 = sha256(args.model)
    if args.expected_artifact_sha256 and artifact_sha256 != args.expected_artifact_sha256:
        raise SystemExit(
            f"artifact SHA-256 mismatch: expected {args.expected_artifact_sha256}, got {artifact_sha256}"
        )

    frozen = json.loads(args.frozen_reference.read_text())
    web_manifest = json.loads((args.web_reference_dir / "manifest.json").read_text())
    tolerances = frozen["tolerances"]
    interpreter = Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit("LiteRT model must expose exactly one input and one output")
    input_contract = tensor_detail(inputs[0], "NCHW")
    output_contract = tensor_detail(outputs[0], "N_ATTRIBUTES_ANCHORS")
    if input_contract["shape"] != EXPECTED_INPUT_SHAPE or input_contract["dtype"] != "float32":
        raise SystemExit(f"unsupported LiteRT input contract: {input_contract}")
    if output_contract["shape"] != EXPECTED_OUTPUT_SHAPE or output_contract["dtype"] != "float32":
        raise SystemExit(f"unsupported LiteRT output contract: {output_contract}")
    for contract in (input_contract, output_contract):
        if contract["quantization"] != {"scale": 0.0, "zeroPoint": 0}:
            raise SystemExit(f"unexpected FP32 quantization metadata: {contract}")

    op_details = interpreter._get_ops_details()
    op_names = [item["op_name"] for item in op_details]
    nms_ops = [name for name in op_names if "NON_MAX_SUPPRESSION" in name.upper() or name.upper() == "NMS"]
    fixtures = []
    failures = []
    for frozen_fixture in frozen["fixtures"]:
        fixture_id = frozen_fixture["id"]
        web_fixture = next((item for item in web_manifest["fixtures"] if item["id"] == fixture_id), None)
        if web_fixture is None:
            raise SystemExit(f"{fixture_id}: ignored Web tensor manifest missing fixture")
        fixture_dir = args.web_reference_dir / fixture_id
        input_path = fixture_dir / "input.f32le"
        raw_path = fixture_dir / "raw.f32le"
        if sha256(input_path) != web_fixture["input"]["sha256"]:
            raise SystemExit(f"{fixture_id}: Web input tensor digest mismatch")
        if sha256(raw_path) != frozen_fixture["runs"][0]["rawTensor"]["sha256Float32Le"]:
            raise SystemExit(f"{fixture_id}: Web raw tensor digest mismatch")
        input_tensor = np.fromfile(input_path, dtype="<f4").reshape(EXPECTED_INPUT_SHAPE)
        reference_raw = np.fromfile(raw_path, dtype="<f4").reshape(EXPECTED_OUTPUT_SHAPE)
        if not np.isfinite(input_tensor).all() or not np.isfinite(reference_raw).all():
            raise SystemExit(f"{fixture_id}: Web reference contains NaN or Infinity")

        mapped_runs = []
        runtime_runs = []
        for _ in range(2):
            interpreter.set_tensor(input_contract["index"], input_tensor)
            interpreter.invoke()
            runtime_raw = interpreter.get_tensor(output_contract["index"])
            if list(runtime_raw.shape) != EXPECTED_OUTPUT_SHAPE:
                raise SystemExit(f"{fixture_id}: runtime output Shape changed")
            if not np.isfinite(runtime_raw).all():
                raise SystemExit(f"{fixture_id}: runtime output contains NaN or Infinity")
            mapped = runtime_raw.astype(np.float32, copy=True)
            mapped[:, :4, :] *= np.float32(640.0)
            runtime_runs.append(tensor_summary(runtime_raw))
            mapped_runs.append(mapped)
        deterministic = np.array_equal(mapped_runs[0], mapped_runs[1])
        if not deterministic:
            failures.append(f"{fixture_id}: repeated LiteRT output differs")

        comparison = raw_comparison(mapped_runs[0], reference_raw, tolerances)
        before_nms, after_nms = decode(
            mapped_runs[0],
            int(web_fixture["image"]["width"]),
            int(web_fixture["image"]["height"]),
            frozen_fixture["preprocessing"],
        )
        expected_decoded = frozen_fixture["runs"][0]["decoded"]
        decoded = decoded_comparison(after_nms, expected_decoded)
        if comparison["mismatchCount"]:
            failures.append(f"{fixture_id}: raw tolerance mismatch")
        if decoded["classMismatchCount"] or decoded["anchorMismatchCount"] or decoded["countMismatch"]:
            failures.append(f"{fixture_id}: decoded identity mismatch")
        if decoded["maximumConfidenceAbsolute"] > tolerances["confidenceAbsolute"]:
            failures.append(f"{fixture_id}: confidence tolerance mismatch")
        if decoded["minimumBboxIou"] is not None and decoded["minimumBboxIou"] < tolerances["boxIouMinimum"]:
            failures.append(f"{fixture_id}: bbox IoU tolerance mismatch")
        if decoded["maximumDecodedBboxAbsolute"] > tolerances["decodedBoxAbsolute"]:
            failures.append(f"{fixture_id}: decoded bbox absolute tolerance mismatch")
        fixtures.append(
            {
                "decoded": {
                    "beforeNmsCount": len(before_nms),
                    "nmsIouThreshold": 0.45,
                    "results": after_nms,
                },
                "decodedComparison": decoded,
                "determinism": {
                    "mappedRawExact": deterministic,
                    "runDigests": [tensor_sha256(item) for item in mapped_runs],
                },
                "id": fixture_id,
                "mappedOutput": tensor_summary(mapped_runs[0]),
                "rawComparison": comparison,
                "runtimeOutput": runtime_runs[0],
                "webReferenceDecoded": expected_decoded,
                "webReferenceRaw": tensor_summary(reference_raw),
            }
        )

    finite_values = all(
        item["runtimeOutput"]["finiteCount"] == item["runtimeOutput"]["elementCount"] for item in fixtures
    )
    report = {
        "artifact": {
            "bytes": args.model.stat().st_size,
            "format": "TFLite FlatBuffer with deterministic Ultralytics metadata ZIP trailer",
            "logicalPath": ".evidence/litert/artifacts/yolov8n-fp32.tflite",
            "sha256": artifact_sha256,
            "trackedByGit": False,
        },
        "contract": {
            "input": input_contract,
            "output": output_contract,
            "outputRole": "detections",
        },
        "fixtures": fixtures,
        "mapping": {
            "input": "identity: runtime input already uses NCHW FP32 [1,3,640,640]",
            "outputCoordinates": "multiply attributes 0..3 by 640 to reverse the official Ultralytics LiteRT _NormalizeCoords wrapper",
            "outputLayout": "identity: runtime output already uses [batch,attribute,anchor] [1,84,8400]",
        },
        "ownership": {
            "nms": "operator postprocess; no NMS op is present in the model graph",
            "postprocess": "operator decode, confidence threshold and NMS",
            "preprocess": "runtime adapter letterbox/RGB/FP32 [0,1]; not fused into the model",
        },
        "runtime": {
            "name": "ai-edge-litert Interpreter",
            "operatorCount": len(op_names),
            "operatorNames": sorted(set(op_names)),
            "nmsOperators": nms_ops,
            "threads": 1,
            "version": runtime_version,
        },
        "schemaVersion": 1,
        "summary": {
            "allFinite": finite_values,
            "allShapesMatched": True,
            "classMismatchCount": sum(item["decodedComparison"]["classMismatchCount"] for item in fixtures),
            "deterministic": all(item["determinism"]["mappedRawExact"] for item in fixtures),
            "fixtureCount": len(fixtures),
            "maximumConfidenceAbsolute": max(
                item["decodedComparison"]["maximumConfidenceAbsolute"] for item in fixtures
            ),
            "maximumDecodedBboxAbsolute": max(
                item["decodedComparison"]["maximumDecodedBboxAbsolute"] for item in fixtures
            ),
            "maximumRawAbsolute": max(item["rawComparison"]["maxAbsolute"] for item in fixtures),
            "maximumRawRelative": max(item["rawComparison"]["maxRelative"] for item in fixtures),
            "minimumBboxIou": min(
                (
                    item["decodedComparison"]["minimumBboxIou"]
                    for item in fixtures
                    if item["decodedComparison"]["minimumBboxIou"] is not None
                ),
                default=None,
            ),
            "rawToleranceMismatchCount": sum(item["rawComparison"]["mismatchCount"] for item in fixtures),
        },
        "tolerances": tolerances,
        "webReference": {
            "runtime": frozen["runtime"],
            "sha256": web_manifest["sourceReferenceSha256"],
        },
    }
    if nms_ops:
        failures.append("model graph unexpectedly contains NMS")
    if not finite_values:
        failures.append("runtime output contains non-finite values")
    report["passed"] = not failures
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "summary": report["summary"]}, sort_keys=True))
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
