#!/usr/bin/env python3
"""使用结构化 ONNX API 改写 DFL Conv，并证明与规范 ONNX 数值等价。"""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


DFL_CONV_NAME = "/model.22/dfl/conv/Conv"
EXPECTED_INPUT_SHAPE = [1, 3, 640, 640]
EXPECTED_OUTPUT_SHAPE = [1, 84, 8400]
RAW_ATOL = 1.0e-5
RAW_RTOL = 1.0e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value.type.tensor_type.shape.dim
    ]


def summarize_model(path: Path) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    operators: dict[str, int] = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    return {
        "bytes": path.stat().st_size,
        "inputs": [{"name": item.name, "shape": shape(item)} for item in model.graph.input],
        "nmsInGraph": any(node.op_type == "NonMaxSuppression" for node in model.graph.node),
        "operators": dict(sorted(operators.items())),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "outputs": [{"name": item.name, "shape": shape(item)} for item in model.graph.output],
        "sha256": sha256(path),
    }


def assert_supported_conv(node: onnx.NodeProto, weights: np.ndarray) -> None:
    if node.op_type != "Conv" or len(node.input) != 2 or len(node.output) != 1:
        raise RuntimeError("DFL 节点不是预期的无 bias 单输出 Conv")
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    expected_defaults = {
        "dilations": [1, 1],
        "group": 1,
        "kernel_shape": [1, 1],
        "pads": [0, 0, 0, 0],
        "strides": [1, 1],
    }
    for name, expected in expected_defaults.items():
        actual = attributes.get(name, expected)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise RuntimeError(f"DFL Conv 属性 {name}={actual!r}，不满足受支持改写前提")
    expected_weights = np.arange(16, dtype=np.float32).reshape(1, 16, 1, 1)
    if weights.shape != expected_weights.shape or not np.array_equal(weights, expected_weights):
        raise RuntimeError("DFL Conv 权重不是精确的 [0..15]")


def rewrite(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source, load_external_data=False)
    onnx.checker.check_model(model)
    matches = [(index, node) for index, node in enumerate(model.graph.node) if node.name == DFL_CONV_NAME]
    if len(matches) != 1:
        raise RuntimeError(f"预期恰好一个 {DFL_CONV_NAME}，实际 {len(matches)}")
    index, node = matches[0]
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if node.input[1] not in initializers:
        raise RuntimeError("DFL Conv 权重不是 initializer")
    assert_supported_conv(node, initializers[node.input[1]])

    axes_name = "/model.22/dfl/conv/ReduceSum_axes"
    product_name = "/model.22/dfl/conv/Mul_output_0"
    model.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), axes_name))
    replacement = [
        helper.make_node(
            "Mul",
            [node.input[0], node.input[1]],
            [product_name],
            name="/model.22/dfl/conv/Mul",
        ),
        helper.make_node(
            "ReduceSum",
            [product_name, axes_name],
            list(node.output),
            name="/model.22/dfl/conv/ReduceSum",
            keepdims=1,
        ),
    ]
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes[:index] + replacement + nodes[index + 1 :])
    helper.set_model_props(
        model,
        {
            **{item.key: item.value for item in model.metadata_props},
            "rimeflow.mindsporeCompatibilityTransform": "DFL Conv -> Mul + ReduceSum(axis=1, keepdims=1)",
        },
    )
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    onnx.checker.check_model(onnx.load(output, load_external_data=False))
    return {
        "insertedNodes": [item.name for item in replacement],
        "removedNode": node.name,
        "weightInitializer": node.input[1],
        "weightSha256Float32Le": hashlib.sha256(initializers[node.input[1]].astype("<f4").tobytes()).hexdigest(),
    }


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


def compare(reference: Path, candidate: Path, web_reference_dir: Path | None) -> dict[str, object]:
    reference_session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    candidate_session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
    if [item.shape for item in reference_session.get_inputs()] != [EXPECTED_INPUT_SHAPE]:
        raise RuntimeError("规范 ONNX 输入 Shape 漂移")
    if [item.shape for item in candidate_session.get_outputs()] != [EXPECTED_OUTPUT_SHAPE]:
        raise RuntimeError("派生 ONNX 输出 Shape 漂移")
    results = []
    for case_id, input_values in comparison_cases(web_reference_dir).items():
        expected = reference_session.run(None, {reference_session.get_inputs()[0].name: input_values})[0]
        actual = candidate_session.run(None, {candidate_session.get_inputs()[0].name: input_values})[0]
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), 1.0e-12)
        allclose = bool(np.allclose(actual, expected, atol=RAW_ATOL, rtol=RAW_RTOL))
        results.append(
            {
                "allclose": allclose,
                "candidateSha256Float32Le": tensor_sha256(actual),
                "id": case_id,
                "inputSha256Float32Le": tensor_sha256(input_values),
                "maxAbsoluteDifference": float(difference.max()),
                "maxRelativeDifference": float((difference / denominator).max()),
                "meanAbsoluteDifference": float(difference.mean()),
                "referenceSha256Float32Le": tensor_sha256(expected),
                "shape": list(actual.shape),
            }
        )
    if not all(item["allclose"] for item in results):
        raise RuntimeError("派生 ONNX 超出冻结 raw tensor 容差")
    if not all(math.isfinite(item["maxAbsoluteDifference"]) for item in results):
        raise RuntimeError("数值等价报告包含非有限值")
    return {
        "allCasesWithinTolerance": True,
        "cases": results,
        "tolerances": {"absolute": RAW_ATOL, "relative": RAW_RTOL},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--web-reference-dir", type=Path)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    transform = rewrite(source, output)
    report = {
        "comparison": compare(
            reference,
            output,
            args.web_reference_dir.resolve(strict=True) if args.web_reference_dir else None,
        ),
        "derived": summarize_model(output),
        "reference": summarize_model(reference),
        "schemaVersion": 1,
        "source": summarize_model(source),
        "toolchain": {
            "numpy": version("numpy"),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "transform": transform,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
