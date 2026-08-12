#!/usr/bin/env python3
"""在真实 Linux x86_64 NVIDIA runner 上验证 ORT TensorRT EP。"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "evidence/tensorrt/contract.json"
REPORT_PATH = ROOT / "evidence/reports/tensorrt-ep-report.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact(path: Path, logical_path: str | None = None) -> dict:
    item = path.stat()
    return {
        "path": logical_path or str(path),
        "bytes": item.st_size,
        "sha256": digest(path),
        "mtimeNs": str(item.st_mtime_ns),
        "ctimeNs": str(item.st_ctime_ns),
    }


def git_source_identity(path: str, commit: str) -> dict:
    source = ROOT / path
    current = source.read_bytes()
    blob_result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False)
    blob = blob_result.stdout
    blob_oid = subprocess.run(["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if blob_result.returncode != 0 or blob_oid.returncode != 0:
        raise RuntimeError(f"tracked source missing from evidence source commit: {path}")
    if current != blob:
        raise RuntimeError(f"tracked source differs from evidence source commit: {path}")
    return {"path": path, "bytes": len(current), "sha256": hashlib.sha256(current).hexdigest(), "gitBlobOid": blob_oid.stdout.strip()}


def run(command: list[str]) -> dict:
    started = time.time()
    started_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": subprocess.list2cmdline(command),
        "startedAt": started_iso,
        "finishedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "startedAtEpochSeconds": started,
        "finishedAtEpochSeconds": time.time(),
        "exitCode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def record_call(report: dict, command: str, stage: str, function):
    started = time.time()
    entry = {"command": command, "startedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(), "startedAtEpochSeconds": started, "stage": stage}
    try:
        result = function()
        entry.update({"exitCode": 0, "stdout": "", "stderr": ""})
        return result
    except BaseException as error:
        entry.update({"exitCode": 1, "stdout": "", "stderr": str(error)})
        raise
    finally:
        entry.update({"finishedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(), "finishedAtEpochSeconds": time.time()})
        report["commands"].append(entry)


@contextlib.contextmanager
def capture_native_stderr(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = os.dup(2)
    with path.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def mapped_library_identities(names: tuple[tuple[str, str], ...], work: Path, versions: dict, maps_text: str) -> list[dict]:
    mapped: dict[Path, list[str]] = {}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5].removesuffix(" (deleted)"))
            if path.is_file():
                mapped.setdefault(path, []).append(line)
    identities = []
    library_root = work / "runtime-libraries"
    library_root.mkdir(parents=True, exist_ok=True)
    version_sources = {
        "onnxruntime": "onnxruntime.__version__",
        "tensorrt": "dpkg-query libnvinfer10",
        "cuda": "/usr/local/cuda/version.json",
        "cudnn": "dpkg-query libcudnn9-cuda-12",
        "driver": "nvidia-smi driver_version",
    }
    for name, component in names:
        candidates = [path for path in mapped if path.name == name or path.name.startswith(f"{name}.")]
        resolved = candidates[0] if len(candidates) == 1 else None
        if resolved is None:
            identities.append({"requested": name, "component": component, "mappedPath": None, "candidateCount": len(candidates)})
            continue
        canonical = resolved.resolve()
        captured = library_root / f"{len(identities):02d}-{canonical.name}"
        shutil.copyfile(canonical, captured)
        source_stat = canonical.stat()
        soname = None
        readelf = run(["readelf", "-d", str(canonical)])
        for line in readelf["stdout"].splitlines():
            if "Library soname:" in line:
                soname = line.split("[", 1)[1].split("]", 1)[0]
                break
        identities.append({
            "requested": name,
            "component": component,
            "componentVersion": versions[component] if component != "driver" else versions["driver"],
            "versionSource": version_sources[component],
            "mappedPath": str(resolved),
            "mapLines": mapped[resolved],
            "realpath": str(canonical),
            "basename": canonical.name,
            "soname": soname,
            "sourceDevice": source_stat.st_dev,
            "sourceInode": source_stat.st_ino,
            "artifact": artifact(captured, str(captured.relative_to(work))),
        })
    return identities


def artifact_identities(paths: list[Path], root: Path) -> list[dict]:
    identities = []
    for path in sorted(set(paths)):
        if path.is_file():
            identities.append(artifact(path, str(path.relative_to(root))))
    return identities


def cache_identities(cache_root: Path, timing_root: Path, work: Path, run_id: str | None = None, transaction_id: str | None = None) -> list[dict]:
    paths = [path for root in (cache_root, timing_root) for path in root.rglob("*")]
    identities = artifact_identities(paths, work)
    if run_id is not None:
        generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for item in identities:
            item.update({"runId": run_id, "transactionId": transaction_id, "generatedAt": generated_at})
    return identities


def profile_counts(profile_path: Path) -> tuple[dict, dict, str, list[dict]]:
    counts = {"TensorrtExecutionProvider": 0, "CUDAExecutionProvider": 0, "CPUExecutionProvider": 0, "unknown": 0}
    unique = {key: set() for key in counts}
    nodes = []
    for event in load(profile_path):
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        key = provider if provider in counts else "unknown"
        counts[key] += 1
        unique[key].add(f"{args.get('node_index', '')}:{args.get('op_name', '')}:{str(event.get('name')).removesuffix('_kernel_time')}")
        nodes.append({"name": event.get("name"), "opName": args.get("op_name"), "provider": provider})
    unique_counts = {key: len(values) for key, values in unique.items()}
    execution_plan = "unknown"
    if unique_counts["TensorrtExecutionProvider"]:
        execution_plan = "partitioned" if unique_counts["CUDAExecutionProvider"] + unique_counts["CPUExecutionProvider"] else "full"
    return counts, unique_counts, execution_plan, nodes


def compare_raw(actual, expected, absolute: float, relative: float) -> dict:
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return {"passed": False, "shapeMatch": actual.shape == expected.shape, "finite": bool(np.isfinite(actual).all())}
    if not np.isfinite(expected).all():
        return {"passed": False, "shapeMatch": True, "finite": True, "referenceFinite": False}
    difference = np.abs(actual - expected)
    allowed = absolute + relative * np.abs(expected)
    flat = int(np.argmax(difference))
    reference_at_max = float(expected.flat[flat])
    near_zero = np.abs(expected) < 1e-6
    return {
        "passed": bool(np.all(difference <= allowed)),
        "elementCount": int(actual.size),
        "finiteCount": int(np.count_nonzero(np.isfinite(actual))),
        "referenceFiniteCount": int(np.count_nonzero(np.isfinite(expected))),
        "mismatchCount": int(np.count_nonzero(difference > allowed)),
        "maxAbsoluteDifference": float(difference.max(initial=0)),
        "meanAbsoluteDifference": float(difference.mean()),
        "maxAbsoluteDifferenceLocation": {
            "flatIndex": flat, "attribute": flat // 8400 % 84, "anchor": flat % 8400,
            "actual": float(actual.flat[flat]), "reference": reference_at_max,
            "tolerance": absolute + relative * abs(reference_at_max),
        },
        "nearZero": {
            "referenceAbsoluteThreshold": 1e-6,
            "elementCount": int(np.count_nonzero(near_zero)),
            "maxAbsoluteDifference": float(difference[near_zero].max(initial=0)),
            "mismatchCount": int(np.count_nonzero((difference > allowed) & near_zero)),
        },
    }


def box_iou(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return 0.0 if union <= 0 else intersection / union


def compare_decoded(actual: list[dict], expected: list[dict], tolerances: dict) -> dict:
    if len(actual) != len(expected):
        return {"passed": False, "countMatch": False, "actualCount": len(actual), "expectedCount": len(expected)}
    comparisons = []
    for got, want in zip(actual, expected):
        score_delta = abs(float(got["score"]) - float(want["score"]))
        box_delta = max(abs(float(a) - float(b)) for a, b in zip(got["bbox"], want["bbox"]))
        iou = box_iou(got["bbox"], want["bbox"])
        passed = (got["classId"] == want["classId"] and score_delta <= tolerances["confidenceAbsolute"] and box_delta <= tolerances["decodedBoxAbsolute"] and iou >= tolerances["boxIouMinimum"])
        comparisons.append({"classExact": got["classId"] == want["classId"], "confidenceAbsoluteDifference": score_delta, "bboxMaxAbsoluteDifference": box_delta, "bboxIou": iou, "passed": passed})
    return {"passed": all(item["passed"] for item in comparisons), "actualCount": len(actual), "expectedCount": len(expected), "detections": comparisons}


def classify(error: BaseException, current_stage: str, log_tail: list[str]) -> tuple[str, dict]:
    message = str(error)
    lower = "\n".join([message, *log_tail]).lower()
    buckets = {"parserErrors": [], "buildErrors": [], "operatorErrors": [], "profileErrors": []}
    log_tokens = {
        "parserErrors": ("parser error", "failed to parse", "modelparser", "onnxparser"),
        "buildErrors": ("builder error", "build engine", "engine build", "tactic", "workspace size"),
        "operatorErrors": ("unsupported operator", "unsupported node", "no importer registered", "kernel not found"),
        "profileErrors": ("profile parse", "profiling file", "end_profiling"),
    }
    for line in log_tail:
        lowered = line.lower()
        for key, tokens in log_tokens.items():
            if any(token in lowered for token in tokens):
                buckets[key].append(line)
    if any(token in lower for token in ("parser error", "failed to parse", "modelparser", "onnxparser")):
        stage, bucket = "parser", "parserErrors"
    elif any(token in lower for token in ("unsupported operator", "unsupported node", "no importer registered", "kernel not found")):
        stage, bucket = "operator", "operatorErrors"
    elif current_stage in ("profile-parse", "profile-validation") or any(token in lower for token in ("profile parse", "profiling file", "end_profiling")):
        stage, bucket = "profile", "profileErrors"
    elif any(token in lower for token in ("builder error", "build engine", "engine build", "tactic", "workspace size")):
        stage, bucket = "build", "buildErrors"
    elif current_stage in ("session-create/parser-build", "engine-build/inference"):
        stage, bucket = "build", "buildErrors"
    elif current_stage == "golden-compare":
        stage, bucket = "golden-compare", "operatorErrors"
    else:
        stage, bucket = "runtime-unknown", "buildErrors"
    if message not in buckets[bucket]:
        buckets[bucket].append(message)
    return stage, buckets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    args = parser.parse_args()
    contract = load(CONTRACT_PATH)
    allowed_path = ROOT / ".evidence/tensorrt"
    if allowed_path.is_symlink():
        raise SystemExit("TensorRT evidence root must not be a symlink")
    work = (args.workspace or allowed_path).resolve()
    allowed = allowed_path.resolve()
    if work != allowed and allowed not in work.parents:
        raise SystemExit("TensorRT workspace must stay under .evidence/tensorrt")
    run_id = str(uuid.uuid4())
    transaction_id = str(uuid.uuid4())
    input_root = work / "web-reference"
    cache_root = work / "engine-cache" / contract["canonicalModel"]["sha256"] / transaction_id
    timing_root = work / "timing-cache" / contract["canonicalModel"]["sha256"] / transaction_id
    profile_prefix = work / "profiles" / contract["canonicalModel"]["sha256"] / transaction_id / "ort-tensorrt"
    output = (args.output.resolve() if args.output else work / "tensorrt-ep-report.json")
    if output != work / "tensorrt-ep-report.json" and work not in output.parents:
        raise SystemExit("TensorRT output must stay inside workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started_at_epoch_ns = time.time_ns()
    discarded_cache = cache_identities(cache_root, timing_root, work)
    for root in (cache_root, timing_root):
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    model = ROOT / contract["canonicalModel"]["path"]
    if model.stat().st_size != contract["canonicalModel"]["bytes"] or digest(model) != contract["canonicalModel"]["sha256"]:
        raise SystemExit("canonical ONNX identity mismatch")

    report = {
        "schemaVersion": 2, "taskId": contract["taskId"], "status": "blocked", "supported": False,
        "runtimeExecuted": False, "hostInferenceVerified": False, "goldenExecuted": False,
        "task14Complete": False, "openspecTask1_4Checked": False,
        "failureStage": "runner-preflight", "failureReason": None,
        "sourceCommit": contract["sourceCommit"], "evidenceSourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model": {"path": contract["canonicalModel"]["path"], "bytes": model.stat().st_size, "sha256": digest(model)},
        "runner": {"realNvidiaRunner": False, "host": platform.node(), "os": platform.platform(), "arch": platform.machine(), "gpu": None, "driver": None, "containerImageId": os.environ.get("RIMEFLOW_TENSORRT_IMAGE_ID")},
        "versions": {"onnxruntime": None, "tensorrt": None, "cuda": None, "cudnn": None, "python": platform.python_version(), "matchesContract": False},
        "providers": {"requested": contract["officialPath"]["requestedProviders"], "available": [], "session": [], "options": {"requested": {}, "session": {}}, "profileArtifact": None, "profileExecutionEventCounts": {"TensorrtExecutionProvider": 0, "CUDAExecutionProvider": 0, "CPUExecutionProvider": 0, "unknown": 0}, "profileUniqueNodeCounts": {"TensorrtExecutionProvider": 0, "CUDAExecutionProvider": 0, "CPUExecutionProvider": 0, "unknown": 0}, "executionPlan": "unknown", "fallback": {"cudaExecutionEvents": 0, "cpuExecutionEvents": 0, "cudaUniqueNodes": 0, "cpuUniqueNodes": 0, "hidden": False}},
        "dynamicLibraries": [], "runtimeEvidence": {"mapsArtifact": None, "versionProbeArtifact": None, "containerImageInspectArtifact": None},
        "engineBuild": {"attempted": False, "succeeded": False, "stage": "not-started", "runId": run_id, "transactionId": transaction_id, "startedAt": started_at, "startedAtEpochNs": str(started_at_epoch_ns), "finishedAt": None, "finishedAtEpochNs": None, "generatedAt": None, "logArtifact": None, "errors": {"parserErrors": [], "buildErrors": [], "operatorErrors": [], "profileErrors": []}},
        "cache": {"root": str(work.relative_to(ROOT)), "modelSha256": digest(model), "freshBuildRequired": True, "discardedBeforeRun": discarded_cache, "before": [], "after": [], "generated": []},
        "commands": [], "fixtures": [],
        "webReference": {"frozenReferenceSha256": digest(ROOT / "evidence/golden/web-reference.json"), "exporterManifest": None},
        "productionPostprocess": {"implementation": "src/postprocess.rs", "platformSpecificDecodeOrNmsAdded": False, "sourceArtifacts": [], "containerBinary": None, "build": {"command": "CARGO_TARGET_DIR=/build/tensorrt-target CARGO_INCREMENTAL=0 cargo build --locked --offline --release --manifest-path evidence/tooling/tensorrt-postprocess/Cargo.toml --bin rimeflow-tensorrt-postprocess", "rustBuilderIndex": contract["compatibility"]["rustBuilderIndex"], "rustBuilderAmd64Manifest": contract["compatibility"]["rustBuilderAmd64Manifest"]}},
    }
    try:
        identity = run(["uname", "-a"]); report["commands"].append({**identity, "stage": "runner-preflight"})
        locate_gpu = run(["bash", "-lc", "command -v nvidia-smi"]); report["commands"].append({**locate_gpu, "stage": "runner-preflight"})
        if platform.system() != "Linux" or platform.machine() != "x86_64" or locate_gpu["exitCode"] != 0:
            raise RuntimeError("需要 Linux x86_64 且可执行 nvidia-smi 的真实 NVIDIA runner")
        smi = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"])
        report["commands"].append({**smi, "stage": "runner-preflight"})
        if smi["exitCode"] != 0 or not smi["stdout"].strip():
            raise RuntimeError("nvidia-smi 未发现可用 NVIDIA GPU")
        gpu = [part.strip() for part in smi["stdout"].splitlines()[0].split(",")]
        report["runner"].update({"realNvidiaRunner": True, "gpu": {"name": gpu[0], "uuid": gpu[1], "computeCapability": gpu[3]}, "driver": gpu[2]})
        report["versions"]["driver"] = gpu[2]

        import numpy as np
        import onnxruntime as ort
        package_versions = run(["dpkg-query", "-W", "-f=${Package}=${Version}\\n", "libnvinfer10", "libcudnn9-cuda-12"])
        report["commands"].append({**package_versions, "stage": "runtime-load"})
        packages = dict(line.split("=", 1) for line in package_versions["stdout"].splitlines() if "=" in line)
        cuda_identity = load(Path("/usr/local/cuda/version.json"))
        report["versions"].update({"onnxruntime": ort.__version__, "tensorrt": packages.get("libnvinfer10", "").split("-", 1)[0], "cuda": cuda_identity.get("cuda", {}).get("version"), "cudnn": packages.get("libcudnn9-cuda-12", "").split("-", 1)[0]})
        report["providers"]["available"] = ort.get_available_providers()
        nvcc = run(["nvcc", "--version"]); report["commands"].append({**nvcc, "stage": "runtime-load"})
        expected = contract["compatibility"]
        report["versions"]["matchesContract"] = all((report["versions"][key] or "").startswith(expected[key]) for key in ("onnxruntime", "tensorrt", "cuda", "cudnn", "python"))
        if not report["versions"]["matchesContract"]:
            raise RuntimeError(f"ORT/TensorRT/CUDA/cuDNN/Python version drift: {report['versions']} != {expected}")
        if "TensorrtExecutionProvider" not in report["providers"]["available"]:
            raise RuntimeError("TensorrtExecutionProvider 不在 available providers")

        inputs = load(input_root / "manifest.json")
        frozen = load(ROOT / "evidence/golden/web-reference.json")
        report["webReference"]["exporterManifest"] = artifact(input_root / "manifest.json", "web-reference/manifest.json")
        options = dict(contract["providerOptions"])
        options["trt_engine_cache_path"] = str(cache_root)
        options["trt_timing_cache_path"] = str(timing_root)
        report["providers"]["options"]["requested"] = options
        report["cache"]["before"] = cache_identities(cache_root, timing_root, work)
        session_options = ort.SessionOptions(); session_options.enable_profiling = True; session_options.profile_file_prefix = str(profile_prefix)
        session_options.log_severity_level = 0
        report["engineBuild"].update({"attempted": True, "stage": "session-create/parser-build"})
        ort_log = work / "logs" / contract["canonicalModel"]["sha256"] / "ort-tensorrt.log"
        with capture_native_stderr(ort_log):
            session = record_call(report, "onnxruntime.InferenceSession(TensorRT,CUDA,CPU)", "session-create/parser-build", lambda: ort.InferenceSession(str(model), sess_options=session_options, providers=[("TensorrtExecutionProvider", options), "CUDAExecutionProvider", "CPUExecutionProvider"]))
            report["providers"]["session"] = session.get_providers()
            report["providers"]["options"]["session"] = session.get_provider_options()
            report["engineBuild"]["stage"] = "engine-build/inference"
            cpu = record_call(report, "onnxruntime.InferenceSession(CPU reference only)", "reference-session", lambda: ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]))
            for fixture in inputs["fixtures"]:
                values = np.fromfile(input_root / fixture["id"] / "input.f32le", dtype="<f4").reshape(1, 3, 640, 640)
                actual = np.asarray(record_call(report, f"session.run TensorRT candidate {fixture['id']}", "engine-build/inference", lambda: session.run(None, {"images": values}))[0], dtype=np.float32)
                cpu_raw = np.asarray(record_call(report, f"session.run CPU reference {fixture['id']}", "reference-inference", lambda: cpu.run(None, {"images": values}))[0], dtype=np.float32)
                web_raw = np.fromfile(input_root / fixture["id"] / "raw.f32le", dtype="<f4").reshape(1, 84, 8400)
                raw_path = work / "raw" / f"{fixture['id']}.f32le"; raw_path.parent.mkdir(parents=True, exist_ok=True); actual.astype("<f4").tofile(raw_path)
                decoded_path = work / "decoded" / f"{fixture['id']}.json"; decoded_path.parent.mkdir(parents=True, exist_ok=True)
                decode = run(["/usr/local/bin/rimeflow-tensorrt-postprocess", str(raw_path), str(fixture["image"]["width"]), str(fixture["image"]["height"]), str(decoded_path)])
                report["commands"].append({**decode, "stage": "golden-compare"})
                expected_fixture = next(item for item in frozen["fixtures"] if item["id"] == fixture["id"])
                decoded = load(decoded_path) if decode["exitCode"] == 0 else None
                raw_comparison = compare_raw(actual, web_raw, frozen["tolerances"]["rawTensorAbsolute"], frozen["tolerances"]["rawTensorRelative"])
                decoded_comparison = compare_decoded(decoded or [], expected_fixture["runs"][0]["decoded"], frozen["tolerances"])
                input_path = input_root / fixture["id"] / "input.f32le"
                reference_path = input_root / fixture["id"] / "raw.f32le"
                report["fixtures"].append({
                    "id": fixture["id"], "image": fixture["image"],
                    "artifacts": {
                        "input": {**artifact(input_path, str(input_path.relative_to(work))), "shape": [1, 3, 640, 640], "dtype": "float32", "elementCount": 1228800},
                        "referenceRaw": {**artifact(reference_path, str(reference_path.relative_to(work))), "shape": [1, 84, 8400], "dtype": "float32", "elementCount": 705600},
                        "actualRaw": {**artifact(raw_path, str(raw_path.relative_to(work))), "shape": [1, 84, 8400], "dtype": "float32", "elementCount": 705600},
                        "decoded": {**artifact(decoded_path, str(decoded_path.relative_to(work))), "dtype": "json"},
                    },
                    "rawComparison": raw_comparison,
                    "cudaOrCpuReferenceOnly": compare_raw(actual, cpu_raw, frozen["tolerances"]["rawTensorAbsolute"], frozen["tolerances"]["rawTensorRelative"]),
                    "decodedComparison": decoded_comparison,
                })
            maps_path = work / "runtime" / "proc-self-maps.txt"; maps_path.parent.mkdir(parents=True, exist_ok=True); maps_text = Path("/proc/self/maps").read_text(encoding="utf-8"); maps_path.write_text(maps_text, encoding="utf-8")
            report["runtimeEvidence"]["mapsArtifact"] = artifact(maps_path, str(maps_path.relative_to(work)))
            report["runtimeEvidence"]["pid"] = os.getpid()
            report["dynamicLibraries"] = mapped_library_identities((
                ("libonnxruntime_providers_tensorrt.so", "onnxruntime"), ("libonnxruntime_providers_cuda.so", "onnxruntime"),
                ("libnvinfer.so.10", "tensorrt"), ("libnvonnxparser.so.10", "tensorrt"),
                ("libcudart.so.12", "cuda"), ("libcudnn.so.9", "cudnn"),
            ), work, report["versions"], maps_text)
            probes_path = work / "runtime" / "version-probes.json"
            probes_path.write_text(json.dumps({"resolvedVersions": report["versions"], "commands": {"nvidiaSmi": smi, "dpkgQuery": package_versions, "ort": {"version": ort.__version__, "availableProviders": report["providers"]["available"]}, "python": {"version": platform.python_version(), "executable": sys.executable}, "cuda": cuda_identity, "nvcc": nvcc}}, indent=2) + "\n", encoding="utf-8")
            report["runtimeEvidence"]["versionProbeArtifact"] = artifact(probes_path, str(probes_path.relative_to(work)))
            image_inspect_path = work / "runtime" / "container-image-inspect.json"
            image_inspect = load(image_inspect_path)
            if not isinstance(image_inspect, list) or len(image_inspect) != 1 or image_inspect[0].get("Id") != report["runner"]["containerImageId"]:
                raise RuntimeError("本轮 container image inspect identity 与实际 runner image ID 不一致")
            report["runtimeEvidence"]["containerImageInspectArtifact"] = artifact(image_inspect_path, str(image_inspect_path.relative_to(work)))
            report["engineBuild"]["stage"] = "profile-parse"
            profile = Path(record_call(report, "session.end_profiling()", "profile-parse", session.end_profiling)); counts, unique_counts, execution_plan, _nodes = profile_counts(profile)
            report["providers"]["profileArtifact"] = artifact(profile, str(profile.relative_to(work)))
        report["engineBuild"]["logArtifact"] = artifact(ort_log, str(ort_log.relative_to(work)))
        report["providers"]["profileExecutionEventCounts"] = counts
        report["providers"]["profileUniqueNodeCounts"] = unique_counts
        report["providers"]["executionPlan"] = execution_plan
        report["providers"]["fallback"] = {"cudaExecutionEvents": counts["CUDAExecutionProvider"], "cpuExecutionEvents": counts["CPUExecutionProvider"], "cudaUniqueNodes": unique_counts["CUDAExecutionProvider"], "cpuUniqueNodes": unique_counts["CPUExecutionProvider"], "hidden": False}
        report["cache"]["after"] = cache_identities(cache_root, timing_root, work, run_id, transaction_id)
        before_paths = {item["path"] for item in report["cache"]["before"]}
        report["cache"]["generated"] = [item for item in report["cache"]["after"] if item["path"] not in before_paths]
        report["engineBuild"]["succeeded"] = any(Path(item["path"]).suffix.lower() in (".engine", ".plan") for item in report["cache"]["generated"])
        if not report["engineBuild"]["succeeded"]:
            report["engineBuild"]["stage"] = "engine-build-verification"
            raise RuntimeError("TensorRT engine build did not generate a fresh .engine or .plan artifact")
        report["engineBuild"]["stage"] = "profile-validation"
        if unique_counts["TensorrtExecutionProvider"] < 1:
            raise RuntimeError("configured TensorRT session has no TensorrtExecutionProvider profile node")
        report["engineBuild"]["stage"] = "profile-verified"
        if not all(item["rawComparison"].get("passed") and item["decodedComparison"].get("passed") for item in report["fixtures"]):
            report["engineBuild"]["stage"] = "golden-compare"
            raise RuntimeError("TensorRT fixture raw/decoded golden comparison failed")
        report["productionPostprocess"]["sourceArtifacts"] = [git_source_identity(path, report["evidenceSourceCommit"]) for path in (
            "evidence/tooling/tensorrt-postprocess/Cargo.toml", "evidence/tooling/tensorrt-postprocess/Cargo.lock",
            "evidence/tooling/tensorrt-postprocess/main.rs", "src/postprocess.rs", "evidence/tooling/tensorrt/Dockerfile",
        )]
        binary = Path("/usr/local/bin/rimeflow-tensorrt-postprocess")
        captured_binary = work / "runtime" / "rimeflow-tensorrt-postprocess"
        shutil.copyfile(binary, captured_binary)
        captured_binary.chmod(0o755)
        report["productionPostprocess"]["containerBinary"] = {**artifact(captured_binary, str(captured_binary.relative_to(work))), "containerPath": str(binary), "elf": "ELF64-little-endian-x86_64", "containerImageId": report["runner"]["containerImageId"]}
        report["engineBuild"].update({"finishedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(), "finishedAtEpochNs": str(time.time_ns()), "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        report.update({"status": "host-inference-verified", "runtimeExecuted": True, "hostInferenceVerified": True, "goldenExecuted": True, "supported": False, "failureStage": None, "failureReason": None})
    except BaseException as error:
        if report["engineBuild"]["attempted"]:
            report["cache"]["after"] = cache_identities(cache_root, timing_root, work, run_id, transaction_id)
            before_paths = {item["path"] for item in report["cache"]["before"]}
            report["cache"]["generated"] = [item for item in report["cache"]["after"] if item["path"] not in before_paths]
        if 'ort_log' in locals() and ort_log.is_file():
            report["engineBuild"]["logArtifact"] = artifact(ort_log, str(ort_log.relative_to(work)))
        if report["engineBuild"]["attempted"]:
            current_stage = report["engineBuild"]["stage"]
            log_tail = ort_log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:] if 'ort_log' in locals() and ort_log.is_file() else []
            stage, buckets = classify(error, current_stage, log_tail)
            report["engineBuild"]["stage"] = stage
        else:
            stage = "runner-preflight"
            buckets = {"parserErrors": [], "buildErrors": [], "operatorErrors": [], "profileErrors": []}
        report["failureStage"] = stage if report["runner"]["realNvidiaRunner"] else "runner-preflight"
        report["failureReason"] = str(error)
        for key, values in buckets.items(): report["engineBuild"]["errors"][key].extend(values)
        report["engineBuild"]["finishedAt"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["engineBuild"]["finishedAtEpochNs"] = str(time.time_ns())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if report["status"] == "host-inference-verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
