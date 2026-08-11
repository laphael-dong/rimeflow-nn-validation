#!/usr/bin/env python3
"""从锁定 YOLOv8n checkpoint 导出 MindSpore 兼容性候选 ONNX。"""

import argparse
import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import onnx
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import ultralytics
from ultralytics import YOLO


EXPECTED_PT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
RAW_ATOL = 1.0e-5
RAW_RTOL = 1.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path: Path) -> dict[str, int | str]:
    metadata = path.stat()
    return {
        "bytes": metadata.st_size,
        "mtimeNs": metadata.st_mtime_ns,
        "sha256": sha256(path),
    }


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value.type.tensor_type.shape.dim
    ]


def tensor_summary(value: onnx.ValueInfoProto) -> dict[str, object]:
    return {
        "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
        "name": value.name,
        "shape": tensor_shape(value),
    }


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def comparison_cases(web_reference_dir: Path | None) -> dict[str, np.ndarray]:
    cases = {
        "zeros": np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32),
        "gray114": np.full(EXPECTED_INPUT_SHAPE, 114 / 255, dtype=np.float32),
        "seed20260811": np.random.default_rng(20260811).random(EXPECTED_INPUT_SHAPE, dtype=np.float32),
    }
    if web_reference_dir is not None:
        manifest = json.loads((web_reference_dir / "manifest.json").read_text())
        for item in manifest["fixtures"]:
            values = np.fromfile(web_reference_dir / item["id"] / "input.f32le", dtype="<f4")
            cases[f"fixture:{item['id']}"] = values.reshape(EXPECTED_INPUT_SHAPE)
    return cases


def compare_onnx(reference: Path, candidate: Path, web_reference_dir: Path | None) -> dict[str, object]:
    reference_session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    results = []
    for case_id, input_values in comparison_cases(web_reference_dir).items():
        expected = reference_session.run(None, {reference_session.get_inputs()[0].name: input_values})[0]
        actual = candidate_session.run(None, {candidate_session.get_inputs()[0].name: input_values})[0]
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allclose = bool(np.allclose(actual, expected, atol=RAW_ATOL, rtol=RAW_RTOL))
        results.append(
            {
                "allclose": allclose,
                "candidateSha256Float32Le": tensor_sha256(actual),
                "id": case_id,
                "inputSha256Float32Le": tensor_sha256(input_values),
                "maxAbsoluteDifference": float(difference.max()),
                "meanAbsoluteDifference": float(difference.mean()),
                "referenceSha256Float32Le": tensor_sha256(expected),
                "shape": list(actual.shape),
            }
        )
    if not all(item["allclose"] for item in results):
        raise RuntimeError("重导出 ONNX 超出冻结 raw tensor 容差")
    return {
        "allCasesWithinTolerance": True,
        "cases": results,
        "tolerances": {"absolute": RAW_ATOL, "relative": RAW_RTOL},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", default=17, type=int)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--web-reference-dir", type=Path)
    args = parser.parse_args()

    source = args.pt.resolve(strict=True)
    output = args.output.resolve()
    reference = args.reference.resolve(strict=True)
    before = snapshot(source)
    if before["sha256"] != EXPECTED_PT_SHA256:
        raise RuntimeError("checkpoint SHA-256 与锁定输入不一致")

    work = output.parent / "export-work"
    work.mkdir(parents=True, exist_ok=True)
    local_checkpoint = work / "yolov8n.pt"
    shutil.copyfile(source, local_checkpoint)
    if sha256(local_checkpoint) != EXPECTED_PT_SHA256:
        raise RuntimeError("checkpoint 工作副本 SHA-256 不一致")

    model = YOLO(str(local_checkpoint))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=False,
            opset=args.opset,
            nms=False,
            device="cpu",
            half=False,
            optimize=False,
        )
    ).resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, output)

    graph = onnx.load(output, load_external_data=False)
    normalized_metadata = {item.key: item.value for item in graph.metadata_props}
    normalized_metadata["date"] = "normalized-for-deterministic-mindspore-replay"
    onnx.helper.set_model_props(graph, normalized_metadata)
    onnx.save_model(graph, output)
    graph = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(graph)
    after = snapshot(source)
    if before != after:
        raise RuntimeError("锁定 checkpoint 在导出期间发生变化")
    operators: dict[str, int] = {}
    for node in graph.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    metadata = {item.key: item.value for item in graph.metadata_props}
    report = {
        "comparison": compare_onnx(
            reference,
            output,
            args.web_reference_dir.resolve(strict=True) if args.web_reference_dir else None,
        ),
        "commandParameters": {
            "batch": 1,
            "device": "cpu",
            "dynamic": False,
            "format": "onnx",
            "half": False,
            "imgsz": 640,
            "nms": False,
            "opset": args.opset,
            "optimize": False,
            "simplify": False,
        },
        "environment": {
            "numpy": version("numpy"),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "output": {
            "bytes": output.stat().st_size,
            "inputs": [tensor_summary(item) for item in graph.graph.input],
            "irVersion": graph.ir_version,
            "metadata": dict(sorted(metadata.items())),
            "nmsInGraph": any(node.op_type == "NonMaxSuppression" for node in graph.graph.node),
            "operators": dict(sorted(operators.items())),
            "opsets": [
                {"domain": item.domain, "version": item.version}
                for item in graph.opset_import
            ],
            "outputs": [tensor_summary(item) for item in graph.graph.output],
            "sha256": sha256(output),
        },
        "source": {"after": after, "before": before, "unchanged": before == after},
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
