#!/usr/bin/env python3
"""两轮重放 MindSpore Lite 2.7.0 转换、host Run 与生产后处理证据。"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED = {
    "archive": "8bb1097100c9fec12675670ba2d4264a2cd6da3a9be093eb56631d00fc0c455b",
    "handoffOnnx": "71002056f43781f2d26681c56e7ec3686d918951c5c8ae70ca55de10409a2a45",
    "pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "referenceOnnx": "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad",
}
EXPECTED_CONVERTER_COMMIT = "d2b243f75f33a7a896483b09e567d845155cad06"
INPUT_SHAPE = [1, 3, 640, 640]
RUNTIME_INPUT_SHAPE = [1, 640, 640, 3]
OUTPUT_SHAPE = [1, 84, 8400]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def sanitize(value: str, root: Path, handoff_dir: Path) -> str:
    return value.replace(str(handoff_dir), "$HANDOFF_ASSETS").replace(str(root), "$REPO")


def normalize_converter_log(value: str, root: Path, handoff_dir: Path) -> str:
    normalized = sanitize(value, root, handoff_dir)
    return re.sub(
        r"LITE\(\d+,[0-9a-f]+,converter_lite\):\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d+(?:\.\d+)?",
        "LITE(<pid>,<thread>,converter_lite):<timestamp>",
        normalized,
    )


def execute(
    command: list[str],
    root: Path,
    handoff_dir: Path,
    *,
    env: dict[str, str] | None = None,
    converter_log: bool = False,
) -> dict[str, object]:
    started_at = utc_now()
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    ended_at = utc_now()
    normalize = normalize_converter_log if converter_log else sanitize
    stdout = normalize(result.stdout, root, handoff_dir)
    stderr = normalize(result.stderr, root, handoff_dir)
    return {
        "command": [sanitize(item, root, handoff_dir) for item in command],
        "endedAt": ended_at,
        "exitCode": result.returncode,
        "startedAt": started_at,
        "stderr": stderr,
        "stdout": stdout,
    }


def parse_last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("command stdout did not contain a JSON object")


def artifact(path: Path, logical_path: str | None = None) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": logical_path or str(path),
        "sha256": sha256(path),
    }


def failure_signature(stderr: str) -> dict[str, object] | None:
    match = re.search(r"InferShape failed, name: ([^,\n]+), type: ([^\n]+)", stderr)
    if not match:
        return None
    resize_warnings = sorted(
        set(re.findall(r"Cannot find input:  of node: ([^\n]+)", stderr))
    )
    return {
        "failedOperator": match.group(1),
        "inputShape": [1, 16, 4, 8400],
        "operatorType": match.group(2).strip(),
        "outputShape": [1, 1, 4, 8400],
        "phase": "legacy_optimizer/InferSubgraph -> Conv2DFusion infer-shape -> graph pass",
        "resizeOptionalEmptyInputWarnings": resize_warnings,
        "weightShape": [1, 16, 1, 1],
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


def compare_decoded(
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
    tolerances: dict[str, object],
) -> dict[str, object]:
    comparisons = []
    for candidate, reference in zip(actual, expected):
        bbox_iou = iou(candidate["bbox"], reference["bbox"])
        bbox_max_absolute = max(
            abs(left - right) for left, right in zip(candidate["bbox"], reference["bbox"])
        )
        confidence_absolute = abs(candidate["score"] - reference["score"])
        class_equal = candidate["classId"] == reference["classId"]
        passed = (
            class_equal
            and confidence_absolute <= tolerances["confidenceAbsolute"]
            and bbox_iou >= tolerances["boxIouMinimum"]
            and bbox_max_absolute <= tolerances["decodedBoxAbsolute"]
        )
        comparisons.append(
            {
                "bboxIou": bbox_iou,
                "bboxMaxAbsoluteDifference": bbox_max_absolute,
                "classEqual": class_equal,
                "confidenceAbsoluteDifference": confidence_absolute,
                "passed": passed,
            }
        )
    return {
        "actualCount": len(actual),
        "comparisons": comparisons,
        "expectedCount": len(expected),
        "passed": len(actual) == len(expected) and all(item["passed"] for item in comparisons),
    }


def compare_raw(actual: np.ndarray, expected: np.ndarray, tolerances: dict[str, object]) -> dict[str, object]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    passed = bool(
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(
            actual,
            expected,
            atol=tolerances["rawTensorAbsolute"],
            rtol=tolerances["rawTensorRelative"],
        )
    )
    return {
        "actualSha256Float32Le": tensor_sha256(actual),
        "elementCount": int(actual.size),
        "expectedSha256Float32Le": tensor_sha256(expected),
        "finiteCount": int(np.count_nonzero(np.isfinite(actual))),
        "maxAbsoluteDifference": float(difference.max()),
        "meanAbsoluteDifference": float(difference.mean()),
        "passed": passed,
    }


def matrix_definition(
    reference: Path,
    handoff: Path,
    exported: Path,
    derived: Path,
) -> list[dict[str, object]]:
    return [
        {
            "id": "reference-baseline-general",
            "inputShape": None,
            "model": reference,
            "optimize": "general",
            "rationale": "先逐字重放既有规范 ONNX 失败，确认失败阶段、算子和 Resize 警告没有漂移。",
        },
        {
            "id": "reference-static-general",
            "inputShape": "images:1,3,640,640",
            "model": reference,
            "optimize": "general",
            "rationale": "规范 ONNX 已声明静态 Shape，但 converter 明确支持 inputShape；用于排除解析后静态 Shape 未传播。",
        },
        {
            "id": "handoff-static-general",
            "inputShape": "images:1,3,640,640",
            "model": handoff,
            "optimize": "general",
            "rationale": "handoff ONNX 与规范模型权重及固定输入输出等价，但由 Ultralytics 8.4.104/opset 20 生成；用于隔离 exporter/opset 图表示差异。",
        },
        {
            "id": "pt-reexport-opset17-unsimplified-static-general",
            "inputShape": "images:1,3,640,640",
            "model": exported,
            "optimize": "general",
            "rationale": "从锁定 .pt 以静态、nms=false、opset 17、simplify=false 重导出；规范模型也是 opset 17，关闭 simplify 用于隔离图简化重写。",
        },
        {
            "id": "reference-static-none",
            "inputShape": "images:1,3,640,640",
            "model": reference,
            "optimize": "none",
            "rationale": "已知错误位于 converter graph optimization/infer-shape；官方 optimize=none 是有边界的故障阶段隔离，不遍历其他优化模式。",
        },
        {
            "id": "pt-reexport-opset17-dfl-reduced-static-general",
            "inputShape": "images:1,3,640,640",
            "model": derived,
            "optimize": "general",
            "rationale": "所有未改图路径均锁定在 DFL 1x1 Conv；其权重精确为 0..15，故用结构化 ONNX API 等价替换为 Mul+ReduceSum，并先通过冻结 raw 容差。",
        },
    ]


def run_conversion_matrix(
    root: Path,
    handoff_dir: Path,
    round_root: Path,
    converter: Path,
    converter_env: dict[str, str],
    matrix: list[dict[str, object]],
) -> list[dict[str, object]]:
    results = []
    for candidate in matrix:
        output_prefix = round_root / "conversions" / candidate["id"]
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_artifact = output_prefix.with_suffix(".ms")
        output_artifact.unlink(missing_ok=True)
        arguments = [
            "--fmk=ONNX",
            f"--modelFile={candidate['model']}",
            f"--outputFile={output_prefix}",
        ]
        if candidate["inputShape"]:
            arguments.append(f"--inputShape={candidate['inputShape']}")
        arguments.append(f"--optimize={candidate['optimize']}")
        attempt = execute(
            [str(converter), *arguments],
            root,
            handoff_dir,
            env=converter_env,
            converter_log=True,
        )
        attempt["parameters"] = {
            "fmk": "ONNX",
            "inputShape": candidate["inputShape"],
            "optimize": candidate["optimize"],
            "saveType": "MINDIR_LITE (converter default)",
        }
        attempt["failureSignature"] = failure_signature(attempt["stderr"])
        attempt["artifact"] = (
            artifact(output_artifact, f".evidence/mindspore/replay/round-N/conversions/{candidate['id']}.ms")
            if output_artifact.is_file()
            else None
        )
        attempt["id"] = candidate["id"]
        attempt["model"] = artifact(candidate["model"], sanitize(str(candidate["model"]), root, handoff_dir))
        attempt["rationale"] = candidate["rationale"]
        attempt["result"] = "success" if attempt["exitCode"] == 0 and attempt["artifact"] else "failed"
        results.append(attempt)
    return results


def build_host_runner(
    root: Path,
    handoff_dir: Path,
    round_root: Path,
    mindspore_root: Path,
) -> tuple[dict[str, object], Path]:
    binary = round_root / "host-runner" / "mindspore-host-runner"
    binary.parent.mkdir(parents=True, exist_ok=True)
    runtime = mindspore_root / "runtime"
    glog = runtime / "third_party/glog/libmindspore_glog.so.0"
    command = [
        "c++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        f"-I{runtime}",
        "evidence/tooling/mindspore-host-runner/main.cpp",
        f"-L{runtime / 'lib'}",
        f"-Wl,-rpath,{runtime / 'lib'}",
        f"-Wl,-rpath,{runtime / 'third_party/glog'}",
        "-lmindspore-lite",
        str(glog),
        "-o",
        str(binary),
    ]
    result = execute(command, root, handoff_dir)
    result["binary"] = artifact(binary, ".evidence/mindspore/replay/round-N/host-runner/mindspore-host-runner") if binary.is_file() else None
    return result, binary


def run_host_validation(
    root: Path,
    handoff_dir: Path,
    round_root: Path,
    mindspore_root: Path,
    model: Path,
    web_root: Path,
    fixture_manifest: dict[str, object],
    frozen_reference: dict[str, object],
) -> dict[str, object]:
    runtime = mindspore_root / "runtime"
    runtime_env = {
        "LD_LIBRARY_PATH": ":".join(
            [
                str(runtime / "lib"),
                str(runtime / "third_party/glog"),
                str(runtime / "third_party/libjpeg-turbo/lib"),
            ]
        )
    }
    compile_attempt, host_runner = build_host_runner(root, handoff_dir, round_root, mindspore_root)
    if compile_attempt["exitCode"] != 0:
        raise RuntimeError(f"MindSpore host runner compile failed: {compile_attempt['stderr']}")
    rust_build = execute(
        ["cargo", "build", "--offline", "--manifest-path", "evidence/tooling/raw-golden/Cargo.toml"],
        root,
        handoff_dir,
    )
    if rust_build["exitCode"] != 0:
        raise RuntimeError(f"production Rust harness build failed: {rust_build['stderr']}")
    rust_runner = root / "evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden"
    benchmark = mindspore_root / "tools/benchmark/benchmark"
    tolerances = frozen_reference["tolerances"]
    fixtures = []
    discovered_contract = None
    for entry in fixture_manifest["images"]:
        fixture_id = entry["id"]
        canonical_path = web_root / fixture_id / "input.f32le"
        canonical = np.fromfile(canonical_path, dtype="<f4").reshape(INPUT_SHAPE)
        runtime_input = np.transpose(canonical, (0, 2, 3, 1)).astype("<f4", copy=False)
        runtime_input_path = round_root / "runtime-inputs" / f"{fixture_id}.f32le"
        runtime_input_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_input.tofile(runtime_input_path)
        raw_path = round_root / "runtime-outputs" / f"{fixture_id}.f32le"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_attempt = execute(
            [
                str(benchmark),
                f"--modelFile={model}",
                "--modelType=MindIR",
                "--device=CPU",
                f"--inDataFile={runtime_input_path}",
                "--loopCount=1",
                "--warmUpLoopCount=0",
                "--numThreads=1",
            ],
            root,
            handoff_dir,
            env=runtime_env,
        )
        host_attempt = execute(
            [str(host_runner), str(model), str(runtime_input_path), str(raw_path)],
            root,
            handoff_dir,
            env=runtime_env,
        )
        if benchmark_attempt["exitCode"] != 0 or host_attempt["exitCode"] != 0:
            raise RuntimeError(f"{fixture_id}: MindSpore benchmark/runtime failed")
        contract = parse_last_json(host_attempt["stdout"])
        if discovered_contract is None:
            discovered_contract = contract
        elif discovered_contract != contract:
            raise RuntimeError(f"{fixture_id}: runtime I/O contract drift")
        decoded_path = round_root / "decoded" / f"{fixture_id}.json"
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        rust_attempt = execute(
            [
                str(rust_runner),
                str(raw_path),
                str(entry["width"]),
                str(entry["height"]),
                str(decoded_path),
            ],
            root,
            handoff_dir,
        )
        if rust_attempt["exitCode"] != 0:
            raise RuntimeError(f"{fixture_id}: production Rust decode/NMS failed")
        actual_raw = np.fromfile(raw_path, dtype="<f4").reshape(OUTPUT_SHAPE)
        expected_raw = np.fromfile(web_root / fixture_id / "raw.f32le", dtype="<f4").reshape(OUTPUT_SHAPE)
        actual_decoded = json.loads(decoded_path.read_text())
        frozen_fixture = next(item for item in frozen_reference["fixtures"] if item["id"] == fixture_id)
        decoded_comparison = compare_decoded(actual_decoded, frozen_fixture["runs"][0]["decoded"], tolerances)
        raw_comparison = compare_raw(actual_raw, expected_raw, tolerances)
        passed = decoded_comparison["passed"] and raw_comparison["passed"]
        if not passed:
            raise RuntimeError(f"{fixture_id}: MindSpore golden comparison failed")
        fixtures.append(
            {
                "benchmark": benchmark_attempt,
                "canonicalInput": artifact(canonical_path, f".evidence/mindspore/replay/round-N/web-reference/{fixture_id}/input.f32le"),
                "decoded": actual_decoded,
                "decodedComparison": decoded_comparison,
                "id": fixture_id,
                "passed": passed,
                "productionRust": rust_attempt,
                "raw": artifact(raw_path, f".evidence/mindspore/replay/round-N/runtime-outputs/{fixture_id}.f32le"),
                "rawComparison": raw_comparison,
                "runtime": host_attempt,
                "runtimeInput": {
                    **artifact(runtime_input_path, f".evidence/mindspore/replay/round-N/runtime-inputs/{fixture_id}.f32le"),
                    "mapping": "transpose NCHW [0,1,2,3] -> NHWC [0,2,3,1]; no value transform",
                    "shape": RUNTIME_INPUT_SHAPE,
                },
            }
        )
    return {
        "compile": compile_attempt,
        "discoveredContract": discovered_contract,
        "fixtures": fixtures,
        "passed": all(item["passed"] for item in fixtures),
        "productionRustBuild": rust_build,
    }


def validate_inputs(paths: dict[str, Path]) -> dict[str, object]:
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        item = artifact(path)
        item["expectedSha256"] = EXPECTED[name]
        item["verified"] = item["sha256"] == EXPECTED[name]
        if not item["verified"]:
            raise RuntimeError(f"{name} SHA-256 mismatch")
        result[name] = item
    return result


def sanitize_input_snapshot(
    snapshot: dict[str, object], root: Path, handoff_dir: Path
) -> dict[str, object]:
    cleaned = json.loads(json.dumps(snapshot))
    for item in cleaned.values():
        item["path"] = sanitize(item["path"], root, handoff_dir)
    return cleaned


def matrix_comparison(rounds: list[dict[str, object]]) -> dict[str, object]:
    first = {item["id"]: item for item in rounds[0]["matrix"]}
    second = {item["id"]: item for item in rounds[1]["matrix"]}
    paths = {}
    for candidate_id in first:
        left = first[candidate_id]
        right = second[candidate_id]
        paths[candidate_id] = {
            "artifactDigestEqual": left["artifact"] == right["artifact"],
            "exitCodeEqual": left["exitCode"] == right["exitCode"],
            "failureSignatureEqual": left["failureSignature"] == right["failureSignature"],
            "resultEqual": left["result"] == right["result"],
        }
    return {
        "allDeterministic": all(all(value.values()) for value in paths.values()),
        "paths": paths,
    }


def stable_manifest(
    inputs: dict[str, object],
    rounds: list[dict[str, object]],
    tools: dict[str, object],
) -> dict[str, object]:
    final = rounds[1]
    success = next(item for item in final["matrix"] if item["result"] == "success")
    contract = final["hostValidation"]["discoveredContract"]
    return {
        "artifact": {
            **success["artifact"],
            "format": "MindIR Lite / MINDIR_LITE FlatBuffer (.ms)",
            "location": ".evidence/mindspore/artifacts/yolov8n-fp32.ms",
            "trackedByGit": False,
        },
        "conversion": {
            "candidateId": success["id"],
            "command": success["command"],
            "parameters": success["parameters"],
            "sourceGraph": success["model"],
        },
        "derivedOnnx": {
            "equivalence": final["derivationReport"]["comparison"],
            "graph": final["derivationReport"]["derived"],
            "transform": final["derivationReport"]["transform"],
        },
        "differencesFromReferenceOnnx": [
            "输入 runtime Shape/layout 从 NCHW [1,3,640,640] 变为 NHWC [1,640,640,3]；输入名称 images、index 0、FP32 保持。",
            "DFL 的单个 1x1 Conv(weight=0..15) 由结构化 ONNX API 改写为 Mul+ReduceSum(axis=1, keepdims=1)。",
            "输出名称 output0、index 0、attributes-first Shape [1,84,8400]、FP32 和 640x640 输入像素 xywh 坐标保持。",
            "产物格式从 ONNX 变为 MindIR Lite .ms；输入输出均无量化参数。",
        ],
        "ioContract": {
            "inputs": contract["inputs"],
            "outputs": contract["outputs"],
        },
        "ownership": {
            "coordinates": "output attributes 0..3 are xywh in 640x640 letterboxed input pixels; no coordinate remapping in the model",
            "nms": "operator; source/derived ONNX contain no NonMaxSuppression and host runner performs no decode/NMS",
            "postprocessing": "production src/postprocess.rs through evidence/tooling/raw-golden",
            "preprocessing": "runtime adapter owns letterbox, RGB, /255 normalization and NCHW-to-NHWC transpose; model artifact folds none of them",
        },
        "quantization": {
            "input": contract["inputs"][0]["quantization"],
            "mode": "FP32; converter input/output type defaults, fp16 off, no quantization requested",
            "output": contract["outputs"][0]["quantization"],
        },
        "schemaVersion": 1,
        "sourceInputs": inputs,
        "status": {
            "artifactVerified": True,
            "harmonyOsDeviceVerified": False,
            "hostInferenceVerified": True,
            "supported": False,
            "task14Complete": False,
            "value": "host-inference-verified",
        },
        "toolchain": tools,
        "usageScope": {
            "artifactRedistributionAllowed": False,
            "modelLicenseMetadata": "AGPL-3.0 License (https://ultralytics.com/license)",
            "productPackaging": "excluded",
            "purpose": "internal-onnx-base-framework-validation-only",
            "trackedByGit": False,
        },
    }


def stable_golden(round_data: dict[str, object], frozen_reference: dict[str, object]) -> dict[str, object]:
    fixtures = []
    for item in round_data["hostValidation"]["fixtures"]:
        fixtures.append(
            {
                "decoded": item["decoded"],
                "decodedComparison": item["decodedComparison"],
                "id": item["id"],
                "passed": item["passed"],
                "raw": item["raw"],
                "rawComparison": item["rawComparison"],
                "runtimeInput": item["runtimeInput"],
            }
        )
    return {
        "artifact": next(item["artifact"] for item in round_data["matrix"] if item["result"] == "success"),
        "fixtures": fixtures,
        "passed": all(item["passed"] for item in fixtures),
        "productionPostprocess": {
            "implementation": "src/postprocess.rs",
            "invocation": "evidence/tooling/raw-golden/src/main.rs",
            "platformSpecificImplementationAdded": False,
        },
        "runtimeContract": round_data["hostValidation"]["discoveredContract"],
        "schemaVersion": 1,
        "summary": {"fixtureCount": len(fixtures), "passedCount": sum(item["passed"] for item in fixtures)},
        "tolerances": frozen_reference["tolerances"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--handoff-onnx", required=True, type=Path)
    parser.add_argument("--workspace", default=".evidence/mindspore/replay", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    handoff_dir = args.pt.resolve(strict=True).parent
    workspace = (root / args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    mindspore_root = root / ".evidence/mindspore/mindspore-lite-2.7.0-linux-x64"
    archive = root / ".evidence/mindspore/mindspore-lite-2.7.0-linux-x64.tar.gz"
    reference = root / "models/yolov8n.onnx"
    paths = {
        "archive": archive,
        "handoffOnnx": args.handoff_onnx.resolve(strict=True),
        "pt": args.pt.resolve(strict=True),
        "referenceOnnx": reference,
    }
    inputs = sanitize_input_snapshot(validate_inputs(paths), root, handoff_dir)
    commit_id = (mindspore_root / ".commit_id").read_text().strip().removeprefix("commit ")
    if commit_id != EXPECTED_CONVERTER_COMMIT:
        raise RuntimeError("MindSpore Lite archive commit id drift")
    converter = mindspore_root / "tools/converter/converter/converter_lite"
    benchmark = mindspore_root / "tools/benchmark/benchmark"
    converter_env = {
        "LD_LIBRARY_PATH": f"{mindspore_root / 'tools/converter/lib'}:{mindspore_root / 'runtime/lib'}"
    }
    version_probe = execute([str(converter), "--version"], root, handoff_dir, env=converter_env)
    help_probe = execute([str(converter), "--help"], root, handoff_dir, env=converter_env)
    tools = {
        "archive": inputs["archive"],
        "benchmark": artifact(benchmark, ".evidence/mindspore/mindspore-lite-2.7.0-linux-x64/tools/benchmark/benchmark"),
        "commitId": commit_id,
        "converter": artifact(converter, ".evidence/mindspore/mindspore-lite-2.7.0-linux-x64/tools/converter/converter/converter_lite"),
        "converterHelpProbe": help_probe,
        "converterVersion": "2.7.0",
        "converterVersionProbe": {
            **version_probe,
            "interpretation": "converter_lite 2.7.0 does not implement --version; version is pinned by official archive name/SHA and internal .commit_id",
        },
        "officialDownloadPage": "https://www.mindspore.cn/lite/docs/en/r2.7.0/use/downloads.html",
        "platform": platform.platform(),
        "pythonRequirements": artifact(
            root / "evidence/tooling/mindspore-python-addons.lock",
            "evidence/tooling/mindspore-python-addons.lock",
        ),
        "runtimeLibrary": artifact(
            mindspore_root / "runtime/lib/libmindspore-lite.so",
            ".evidence/mindspore/mindspore-lite-2.7.0-linux-x64/runtime/lib/libmindspore-lite.so",
        ),
    }
    fixture_manifest = json.loads((root / "evidence/fixtures/manifest.json").read_text())
    frozen_reference = json.loads((root / "evidence/golden/web-reference.json").read_text())
    rounds = []
    for round_number in (1, 2):
        round_root = workspace / f"round-{round_number}"
        round_root.mkdir(parents=True, exist_ok=True)
        before = git_status(root)
        source_before_raw = validate_inputs(paths)
        web_root = round_root / "web-reference"
        web = execute(
            ["node", "evidence/scripts/export_web_reference_tensors.mjs", str(web_root)],
            root,
            handoff_dir,
        )
        if web["exitCode"] != 0:
            raise RuntimeError(f"round {round_number}: Web reference export failed: {web['stderr']}")
        exported = round_root / "onnx/yolov8n-opset17-unsimplified.onnx"
        exported.parent.mkdir(parents=True, exist_ok=True)
        export_attempt = execute(
            [
                sys.executable,
                "evidence/scripts/export_mindspore_onnx.py",
                "--pt",
                str(paths["pt"]),
                "--reference",
                str(reference),
                "--web-reference-dir",
                str(web_root),
                "--output",
                str(exported),
                "--opset",
                "17",
            ],
            root,
            handoff_dir,
        )
        if export_attempt["exitCode"] != 0:
            raise RuntimeError(f"round {round_number}: .pt export failed: {export_attempt['stderr']}")
        export_report = parse_last_json(export_attempt["stdout"])
        derived = round_root / "onnx/yolov8n-opset17-dfl-reduced.onnx"
        derivation_report_path = round_root / "onnx/dfl-equivalence.json"
        derive_attempt = execute(
            [
                sys.executable,
                "evidence/scripts/derive_mindspore_onnx.py",
                "--source",
                str(exported),
                "--reference",
                str(reference),
                "--web-reference-dir",
                str(web_root),
                "--output",
                str(derived),
                "--report",
                str(derivation_report_path),
            ],
            root,
            handoff_dir,
        )
        if derive_attempt["exitCode"] != 0:
            raise RuntimeError(f"round {round_number}: ONNX derivation failed: {derive_attempt['stderr']}")
        derivation_report = json.loads(derivation_report_path.read_text())
        matrix = run_conversion_matrix(
            root,
            handoff_dir,
            round_root,
            converter,
            converter_env,
            matrix_definition(reference, paths["handoffOnnx"], exported, derived),
        )
        successes = [item for item in matrix if item["result"] == "success"]
        if len(successes) != 1 or successes[0]["id"] != "pt-reexport-opset17-dfl-reduced-static-general":
            raise RuntimeError(f"round {round_number}: unexpected conversion matrix result")
        success_model = round_root / "conversions/pt-reexport-opset17-dfl-reduced-static-general.ms"
        host_validation = run_host_validation(
            root,
            handoff_dir,
            round_root,
            mindspore_root,
            success_model,
            web_root,
            fixture_manifest,
            frozen_reference,
        )
        source_after_raw = validate_inputs(paths)
        if source_before_raw != source_after_raw:
            raise RuntimeError(f"round {round_number}: locked inputs changed")
        source_before = sanitize_input_snapshot(source_before_raw, root, handoff_dir)
        source_after = sanitize_input_snapshot(source_after_raw, root, handoff_dir)
        after = git_status(root)
        rounds.append(
            {
                "derivation": derive_attempt,
                "derivationReport": derivation_report,
                "export": export_attempt,
                "exportReport": export_report,
                "hostValidation": host_validation,
                "matrix": matrix,
                "round": round_number,
                "sourceAfter": source_after,
                "sourceBefore": source_before,
                "webReference": web,
                "worktreeAfter": after,
                "worktreeBefore": before,
            }
        )

    comparison = matrix_comparison(rounds)
    comparison.update(
        {
            "derivedOnnxDigestEqual": rounds[0]["derivationReport"]["derived"]["sha256"]
            == rounds[1]["derivationReport"]["derived"]["sha256"],
            "fixtureResultsEqual": stable_golden(rounds[0], frozen_reference)
            == stable_golden(rounds[1], frozen_reference),
            "reexportOnnxDigestEqual": rounds[0]["exportReport"]["output"]["sha256"]
            == rounds[1]["exportReport"]["output"]["sha256"],
            "trackedWorktreeStateStable": rounds[0]["worktreeAfter"]["tracked"]
            == rounds[1]["worktreeAfter"]["tracked"],
        }
    )
    if not all(
        [
            comparison["allDeterministic"],
            comparison["derivedOnnxDigestEqual"],
            comparison["fixtureResultsEqual"],
            comparison["reexportOnnxDigestEqual"],
            comparison["trackedWorktreeStateStable"],
        ]
    ):
        raise RuntimeError(f"MindSpore two-round determinism failed: {comparison}")

    final_artifact_source = workspace / "round-2/conversions/pt-reexport-opset17-dfl-reduced-static-general.ms"
    final_artifact = root / ".evidence/mindspore/artifacts/yolov8n-fp32.ms"
    final_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final_artifact_source, final_artifact)
    tools["pythonEnvironment"] = rounds[1]["exportReport"]["environment"]
    manifest = stable_manifest(inputs, rounds, tools)
    manifest["artifact"].update(artifact(final_artifact, ".evidence/mindspore/artifacts/yolov8n-fp32.ms"))
    golden = stable_golden(rounds[1], frozen_reference)
    golden["artifact"] = artifact(final_artifact, ".evidence/mindspore/artifacts/yolov8n-fp32.ms")
    replay = {
        "artifact": artifact(final_artifact, ".evidence/mindspore/artifacts/yolov8n-fp32.ms"),
        "comparison": comparison,
        "inputs": inputs,
        "recorded": args.record,
        "rounds": rounds,
        "schemaVersion": 1,
        "tools": tools,
    }
    replay_path = workspace / "mindspore-replay.json"
    replay_path.write_bytes(stable_bytes(replay))
    if args.record:
        (root / "evidence/conversions/mindspore-artifact-manifest.json").write_bytes(stable_bytes(manifest))
        (root / "evidence/reports/mindspore-golden-report.json").write_bytes(stable_bytes(golden))
        (root / "evidence/reports/mindspore-conversion-report.json").write_bytes(stable_bytes(replay))
    else:
        expected_outputs = [
            (root / "evidence/conversions/mindspore-artifact-manifest.json", stable_bytes(manifest)),
            (root / "evidence/reports/mindspore-golden-report.json", stable_bytes(golden)),
        ]
        for path, expected_bytes in expected_outputs:
            if not path.is_file() or path.read_bytes() != expected_bytes:
                raise RuntimeError(f"tracked MindSpore evidence drift: {path.relative_to(root)}")
    print(
        json.dumps(
            {
                "artifact": artifact(final_artifact, ".evidence/mindspore/artifacts/yolov8n-fp32.ms"),
                "comparison": comparison,
                "recorded": args.record,
                "replay": str(replay_path.relative_to(root)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
