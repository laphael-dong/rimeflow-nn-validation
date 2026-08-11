#!/usr/bin/env python3
"""审计外部 YOLOv8n .pt/ONNX handoff，不复制或发布模型文件。"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
import torchvision
import ultralytics
from onnx import numpy_helper
from ultralytics import YOLO


EXPECTED_PT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_ONNX_SHA256 = "71002056f43781f2d26681c56e7ec3686d918951c5c8ae70ca55de10409a2a45"
PT_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
PT_RELEASE_API = "https://api.github.com/repos/ultralytics/assets/releases/177482232"
PT_RELEASE_ID = 177482232
PT_ASSET_ID = 195719301
PT_EXPECTED_BYTES = 6549796
PT_ONNX_RTOL = 1e-4
PT_ONNX_ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        for dimension in value.type.tensor_type.shape.dim
    ]


def onnx_summary(path: Path) -> tuple[onnx.ModelProto, dict]:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    operators: dict[str, int] = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    summary = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "irVersion": model.ir_version,
        "opsets": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [
            {
                "name": item.name,
                "shape": tensor_shape(item),
                "dtype": onnx.TensorProto.DataType.Name(
                    item.type.tensor_type.elem_type
                ),
            }
            for item in model.graph.input
        ],
        "outputs": [
            {
                "name": item.name,
                "shape": tensor_shape(item),
                "dtype": onnx.TensorProto.DataType.Name(
                    item.type.tensor_type.elem_type
                ),
            }
            for item in model.graph.output
        ],
        "nodeCount": len(model.graph.node),
        "initializerCount": len(model.graph.initializer),
        "valueInfoCount": len(model.graph.value_info),
        "operators": dict(sorted(operators.items())),
        "metadata": dict(sorted(metadata.items())),
        "nmsInGraph": any(node.op_type == "NonMaxSuppression" for node in model.graph.node),
    }
    return model, summary


def compare_initializers(candidate: onnx.ModelProto, reference: onnx.ModelProto) -> dict:
    candidate_values = {
        item.name: numpy_helper.to_array(item) for item in candidate.graph.initializer
    }
    reference_values = {
        item.name: numpy_helper.to_array(item) for item in reference.graph.initializer
    }
    common = sorted(candidate_values.keys() & reference_values.keys())
    mismatches = []
    for name in common:
        left = candidate_values[name]
        right = reference_values[name]
        if not np.array_equal(left, right):
            mismatches.append(name)
    return {
        "commonCount": len(common),
        "exactEqualCount": len(common) - len(mismatches),
        "mismatches": mismatches,
        "onlyCandidate": sorted(candidate_values.keys() - reference_values.keys()),
        "onlyReference": sorted(reference_values.keys() - candidate_values.keys()),
    }


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--reference-onnx", required=True, type=Path)
    parser.add_argument("--exported-onnx", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for path in [args.pt, args.onnx, args.reference_onnx]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.pt) != EXPECTED_PT_SHA256 or args.pt.stat().st_size != PT_EXPECTED_BYTES:
        raise RuntimeError("handoff .pt 与锁定的 Ultralytics release asset 不一致")
    if sha256(args.onnx) != EXPECTED_ONNX_SHA256:
        raise RuntimeError("handoff ONNX 与指定候选 SHA-256 不一致")

    candidate_model, candidate_summary = onnx_summary(args.onnx)
    reference_model, reference_summary = onnx_summary(args.reference_onnx)
    wrapper = YOLO(str(args.pt))
    torch_model = wrapper.model.eval().float()
    checkpoint = wrapper.ckpt or {}

    candidate_session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    reference_session = ort.InferenceSession(
        str(args.reference_onnx), providers=["CPUExecutionProvider"]
    )
    cases = {
        "zeros": np.zeros((1, 3, 640, 640), dtype=np.float32),
        "gray114": np.full((1, 3, 640, 640), 114 / 255, dtype=np.float32),
        "seed20260811": np.random.default_rng(20260811).random(
            (1, 3, 640, 640), dtype=np.float32
        ),
    }
    inference = []
    with torch.inference_mode():
        for case_id, input_values in cases.items():
            torch_runs = []
            for _ in range(2):
                output = torch_model(torch.from_numpy(input_values))
                if isinstance(output, (list, tuple)):
                    output = output[0]
                torch_runs.append(output.detach().cpu().numpy())
            candidate_output = candidate_session.run(
                None, {candidate_session.get_inputs()[0].name: input_values}
            )[0]
            reference_output = reference_session.run(
                None, {reference_session.get_inputs()[0].name: input_values}
            )[0]
            difference = np.abs(
                torch_runs[0].astype(np.float64)
                - candidate_output.astype(np.float64)
            )
            inference.append(
                {
                    "id": case_id,
                    "inputSha256Float32Le": tensor_sha256(input_values),
                    "shape": list(candidate_output.shape),
                    "torchRuns": [tensor_sha256(item) for item in torch_runs],
                    "torchRepeatExact": bool(
                        np.array_equal(torch_runs[0], torch_runs[1])
                    ),
                    "candidateOnnxSha256Float32Le": tensor_sha256(candidate_output),
                    "referenceOnnxSha256Float32Le": tensor_sha256(reference_output),
                    "candidateReferenceExact": bool(
                        np.array_equal(candidate_output, reference_output)
                    ),
                    "ptCandidateAllclose": bool(
                        np.allclose(
                            torch_runs[0],
                            candidate_output,
                            rtol=PT_ONNX_RTOL,
                            atol=PT_ONNX_ATOL,
                        )
                    ),
                    "ptCandidateMaxAbsoluteDifference": float(difference.max()),
                    "ptCandidateMeanAbsoluteDifference": float(difference.mean()),
                }
            )

    exported = None
    if args.exported_onnx:
        if not args.exported_onnx.is_file():
            raise FileNotFoundError(args.exported_onnx)
        exported_model, exported_summary = onnx_summary(args.exported_onnx)
        exported_session = ort.InferenceSession(
            str(args.exported_onnx), providers=["CPUExecutionProvider"]
        )
        input_values = cases["seed20260811"]
        exported_output = exported_session.run(
            None, {exported_session.get_inputs()[0].name: input_values}
        )[0]
        candidate_output = candidate_session.run(
            None, {candidate_session.get_inputs()[0].name: input_values}
        )[0]
        exported = {
            "ephemeralArtifact": exported_summary,
            "initializerComparison": compare_initializers(
                exported_model, candidate_model
            ),
            "seed20260811OutputExact": bool(
                np.array_equal(exported_output, candidate_output)
            ),
            "publication": "test-evidence-only",
        }

    lock_path = Path(__file__).resolve().parents[1] / "tooling" / "model-audit-requirements.lock"
    train_args = checkpoint.get("train_args") or {}
    report = {
        "schemaVersion": 1,
        "checkedOn": "2026-08-11",
        "scope": {
            "verified": ["source .pt", "candidate ONNX", "Linux x86_64 ORT CPU"],
            "delegated": ["Core ML macOS/iOS runner", "LiteRT v2 Android runner", "Windows ML", "MindSpore Lite", "其他真实平台 runner"],
            "doesNotReplaceFrozenGolden": True,
        },
        "sourceCheckpoint": {
            "logicalPath": "$HANDOFF_ASSETS/yolov8n.pt",
            "bytes": args.pt.stat().st_size,
            "sha256": sha256(args.pt),
            "upstream": {
                "repository": "https://github.com/ultralytics/assets",
                "tag": "v8.3.0",
                "releaseId": PT_RELEASE_ID,
                "releaseApi": PT_RELEASE_API,
                "assetId": PT_ASSET_ID,
                "downloadUrl": PT_SOURCE_URL,
                "releaseImmutable": False,
                "downloadedBytesSha256Matched": True,
                "immutabilityConclusion": "官方 release asset 当前字节已匹配；本任务只把指定 SHA-256 当作内部框架验证输入，不把该 URL 作为产品发布来源。",
            },
            "checkpoint": {
                "date": checkpoint.get("date"),
                "version": checkpoint.get("version"),
                "license": checkpoint.get("license"),
                "docs": checkpoint.get("docs"),
                "epoch": checkpoint.get("epoch"),
                "training": {
                    "model": train_args.get("model"),
                    "data": train_args.get("data"),
                    "epochs": train_args.get("epochs"),
                    "imgsz": train_args.get("imgsz"),
                    "seed": train_args.get("seed"),
                    "deterministic": train_args.get("deterministic"),
                    "source": train_args.get("source"),
                },
            },
            "model": {
                "class": f"{type(torch_model).__module__}.{type(torch_model).__name__}",
                "parameterCount": sum(item.numel() for item in torch_model.parameters()),
                "stateDictSha256": model_state_sha256(torch_model),
                "stride": torch_model.stride.tolist(),
                "classCount": len(torch_model.names),
            },
        },
        "candidateOnnx": {
            "logicalPath": "$HANDOFF_ASSETS/yolov8n.onnx",
            **candidate_summary,
        },
        "referenceOnnx": {
            "logicalPath": "models/yolov8n.onnx",
            **reference_summary,
        },
        "comparisons": {
            "initializers": compare_initializers(candidate_model, reference_model),
            "inference": inference,
            "ptOnnxTolerance": {
                "relative": PT_ONNX_RTOL,
                "absolute": PT_ONNX_ATOL,
                "purpose": "仅用于 PyTorch 与 ONNX exporter 数值同源审计，不修改任务 1.3 已冻结的平台 golden 容差。",
            },
            "conclusion": "候选 ONNX、仓库参考 ONNX 和官方 checkpoint 使用同一组权重；ONNX 图的 opset/export metadata 不同，但固定输入结果等价。",
        },
        "reExport": {
            "command": "$AUDIT_PYTHON -m ultralytics export model=$WORK/yolov8n.pt format=onnx imgsz=640 batch=1 simplify=True dynamic=False nms=False",
            "toolVersions": {
                "ultralytics": ultralytics.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "onnx": onnx.__version__,
                "onnxslim": onnxslim.__version__,
            },
            "result": exported,
            "byteReproducible": False,
            "byteReproducibilityBlocker": "Ultralytics 写入导出时间，且 handoff 未提供原始完整依赖锁；当前重导出仅证明 I/O、权重和推理等价，不能复现候选 ONNX 的原始 SHA-256。",
        },
        "tooling": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "providers": candidate_session.get_providers(),
            "requirementsLock": "evidence/tooling/model-audit-requirements.lock",
            "requirementsLockSha256": sha256(lock_path),
        },
        "licensing": {
            "rightsHolder": "Ultralytics",
            "declaredByCheckpoint": checkpoint.get("license"),
            "declaredByCandidateOnnx": candidate_summary["metadata"].get("license"),
            "intendedUse": "internal-onnx-base-framework-validation-only",
            "commercialUse": "not-evaluated-out-of-scope",
            "clientRedistribution": "none",
            "formatConversion": "ephemeral-test-artifacts-only",
            "derivedArtifacts": "test-evidence-only",
            "productPackaging": "excluded",
            "authorizationBlocker": False,
            "reason": "模型许可证声明按 checkpoint 和 ONNX metadata 如实记录；本任务只验证 onnx-base 框架，不评估产品商业授权，模型及派生产物不进入 RimeCut 产品包。",
            "publication": "test-evidence-only",
        },
        "decision": {
            "ptAndOnnxTechnicalVerification": "passed",
            "sourceTraceability": "sha-pinned-for-framework-validation",
            "task14": "blocked",
            "task14Complete": False,
            "blockers": [
                "LiteRT host artifact/inference/golden 与 Core ML artifact/spec 已单独验证；Core ML macOS/iOS runtime、Windows ML、MindSpore Lite、其余 Linux provider 和 LiteRT Android runner evidence 尚未闭环",
            ],
        },
    }
    if not all(item["torchRepeatExact"] for item in inference):
        raise RuntimeError("PyTorch 两轮推理不确定")
    if not all(item["candidateReferenceExact"] for item in inference):
        raise RuntimeError("候选 ONNX 与仓库参考 ONNX 输出不一致")
    if not all(item["ptCandidateAllclose"] for item in inference):
        raise RuntimeError("checkpoint 与候选 ONNX 超出同源审计容差")
    if report["comparisons"]["initializers"]["mismatches"]:
        raise RuntimeError("候选 ONNX 与仓库参考 ONNX 的共同 initializer 不一致")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
