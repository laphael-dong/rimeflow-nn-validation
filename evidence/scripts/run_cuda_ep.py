#!/usr/bin/env python3
"""T14 Linux x86_64 ORT CUDA EP runner；不允许 CPU fallback 冒充 CUDA。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_SHA256 = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
MODEL_BYTES = 12851098
ORT_VERSION = "1.26.0"
LOCKED_PACKAGES = {
    "flatbuffers": "25.2.10",
    "numpy": "2.2.3",
    "onnxruntime-gpu": ORT_VERSION,
    "packaging": "24.2",
    "protobuf": "5.29.3",
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-nvjitlink-cu12": "12.8.93",
}
PROVIDER_OPTIONS = {
    "device_id": "0",
    "arena_extend_strategy": "kNextPowerOfTwo",
    "cudnn_conv_algo_search": "EXHAUSTIVE",
    "cudnn_conv_use_max_workspace": "1",
    "do_copy_in_default_stream": "1",
}
INPUT = {"name": "images", "shape": [1, 3, 640, 640], "dtype": "tensor(float)"}
OUTPUT = {"name": "output0", "shape": [1, 84, 8400], "dtype": "tensor(float)"}
REQUIRED_LIBRARIES = ["libonnxruntime_providers_cuda.so", "libcuda.so", "libcudart.so", "libcudnn.so", "libcublas.so"]
LIBRARY_BASENAME_PATTERNS = {
    "libonnxruntime_providers_cuda.so": r"^libonnxruntime_providers_cuda\.so(?:\.\d+)*$",
    "libcuda.so": r"^libcuda\.so(?:\.\d+)*$",
    "libcudart.so": r"^libcudart\.so(?:\.\d+)*$",
    "libcudnn.so": r"^libcudnn\.so(?:\.\d+)*$",
    "libcublas.so": r"^libcublas\.so(?:\.\d+)*$",
}
EXIT = {"ok": 0, "host": 20, "hardware": 21, "runtime": 22, "session": 23, "profile": 24, "golden": 25}


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd):
    started = now()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {"command": command, "startedAt": started, "endedAt": now(), "exitCode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def logical_path(path, root):
    path = path.resolve()
    return str(path.relative_to(root)) if path == root or root in path.parents else str(path)


def artifact(path, root, **metadata):
    return {
        "path": logical_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **metadata,
    }


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    lines = []
    paths = set()
    for line in maps.read_text(errors="replace").splitlines():
        path = line.rsplit(maxsplit=1)[-1]
        if path.startswith("/") and ("onnxruntime" in path or re.search(r"/(libcuda|libcud|libcublas|libcufft|libcurand|libnvrtc)", path)):
            canonical = str(Path(path).resolve())
            lines.append(f"{line.rsplit(maxsplit=1)[0]} {canonical}")
            paths.add(canonical)
    return sorted(paths), lines


def library_presence(paths):
    return {name: any(re.fullmatch(LIBRARY_BASENAME_PATTERNS[name], Path(path).name) for path in paths) for name in REQUIRED_LIBRARIES}


def library_component_version(name, report):
    if name == "libonnxruntime_providers_cuda.so":
        return report["versions"]["onnxruntime"], "onnxruntime-gpu package"
    if name == "libcuda.so":
        return report["versions"]["nvidiaDriver"], "nvidia-smi driver_version"
    if name == "libcudart.so":
        return report["versions"]["cudaRuntime"], "nvidia-cuda-runtime-cu12 package"
    if name == "libcudnn.so":
        return report["versions"]["cudnn"], "nvidia-cudnn-cu12 package"
    if name == "libcublas.so":
        return report["lockedPackages"]["nvidia-cublas-cu12"], "nvidia-cublas-cu12 package"
    raise AssertionError(name)


def library_evidence(paths, report, root):
    entries = {}
    for required in REQUIRED_LIBRARIES:
        matches = [Path(path).resolve(strict=True) for path in paths if re.fullmatch(LIBRARY_BASENAME_PATTERNS[required], Path(path).name)]
        if not matches:
            continue
        path = matches[0]
        version, source = library_component_version(required, report)
        header = subprocess.run(["readelf", "-h", str(path)], capture_output=True, text=True, check=True).stdout
        dynamic = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, check=True).stdout
        soname_match = re.search(r"\(SONAME\).*\[(.+?)\]", dynamic)
        entry = artifact(
            path,
            root,
            realpath=str(path),
            basename=path.name,
            elfClass="ELF64" if re.search(r"^\s*Class:\s+ELF64\s*$", header, re.MULTILINE) else None,
            elfMachine="Advanced Micro Devices X86-64" if re.search(r"^\s*Machine:\s+Advanced Micro Devices X86-64\s*$", header, re.MULTILINE) else None,
            soname=soname_match.group(1) if soname_match else None,
            componentVersion=version,
            versionSource=source,
        )
        entry["path"] = str(path)
        entries[required] = entry
    return entries


def compare_raw_arrays(actual, reference, absolute, relative):
    import numpy as np

    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowed = absolute + relative * np.abs(reference.astype(np.float64))
    mismatch = difference > allowed
    maximum_index = int(np.argmax(difference)) if difference.size else None
    maximum_allowed_index = int(np.argmax(difference - allowed)) if difference.size else None
    denominator = np.abs(reference.astype(np.float64))
    relative_difference = np.zeros_like(difference)
    np.divide(difference, denominator, out=relative_difference, where=denominator > 0)
    return {
        "passed": int(np.count_nonzero(mismatch)) == 0,
        "mismatchCount": int(np.count_nonzero(mismatch)),
        "maximumAbsolute": float(difference[maximum_index]) if maximum_index is not None else 0.0,
        "maximumRelative": float(np.max(relative_difference)) if relative_difference.size else 0.0,
        "maximumDifferenceFlatIndex": maximum_index,
        "maximumExcessFlatIndex": maximum_allowed_index,
        "nearZeroReferenceCount": int(np.count_nonzero(denominator <= 1.0e-12)),
        "rule": "abs(actual-reference) <= rawTensorAbsolute + rawTensorRelative * abs(reference)",
    }


def bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return 0.0 if union <= 0 else intersection / union


def compare_decoded(actual, expected, tolerances):
    class_mismatches = abs(len(actual) - len(expected))
    confidence_max = 0.0
    box_absolute_max = 0.0
    iou_min = 1.0 if actual or expected else None
    for left, right in zip(actual, expected):
        class_mismatches += int(left["classId"] != right["classId"])
        confidence_max = max(confidence_max, abs(left["score"] - right["score"]))
        box_absolute_max = max(box_absolute_max, *(abs(a - b) for a, b in zip(left["bbox"], right["bbox"])))
        iou_min = min(iou_min, bbox_iou(left["bbox"], right["bbox"]))
    passed = (
        len(actual) == len(expected)
        and class_mismatches == 0
        and confidence_max <= tolerances["confidenceAbsolute"]
        and box_absolute_max <= tolerances["decodedBoxAbsolute"]
        and (iou_min is None or iou_min >= tolerances["boxIouMinimum"])
    )
    return {"passed": passed, "countMismatch": len(actual) != len(expected), "classMismatchCount": class_mismatches, "maximumConfidenceAbsolute": confidence_max, "maximumDecodedBboxAbsolute": box_absolute_max, "minimumBboxIou": iou_min}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.onnx"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--web-reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model = (root / args.model).resolve() if not args.model.is_absolute() else args.model.resolve()
    workspace = (root / args.workspace).resolve() if not args.workspace.is_absolute() else args.workspace.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    allowed = (root / ".evidence/cuda").resolve()
    if workspace != allowed and allowed not in workspace.parents:
        raise SystemExit("CUDA workspace must stay under .evidence/cuda")
    if output != allowed and allowed not in output.parents:
        raise SystemExit("CUDA runtime report must stay under .evidence/cuda")
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": 1,
        "task": "T14-LNX-CUDA-01",
        "state": "blocked",
        "supported": False,
        "task14Complete": False,
        "staticVerified": False,
        "buildVerified": False,
        "runtimeExecuted": False,
        "hostInferenceVerified": False,
        "goldenExecuted": False,
        "startedAt": now(),
        "endedAt": None,
        "failureStage": None,
        "failure": None,
        "host": {"os": platform.system(), "architecture": platform.machine(), "platform": platform.platform(), "python": platform.python_version()},
        "model": {"path": "models/yolov8n.onnx", "bytes": model.stat().st_size if model.exists() else None, "sha256": sha256(model) if model.exists() else None, "converted": False, "copied": False},
        "versions": {"onnxruntime": None, "cudaRuntime": None, "cudnn": None, "nvidiaDriver": None},
        "lockedPackages": LOCKED_PACKAGES,
        "requestedProviders": ["CUDAExecutionProvider"],
        "providerOptionsRequested": PROVIDER_OPTIONS,
        "availableProviders": [],
        "sessionProviders": [],
        "sessionProviderOptions": {},
        "profile": {"artifact": None, "nodeProviderCounts": {}, "cudaNodeCount": 0, "cpuNodeCount": 0},
        "sharedLibraries": {"mapsArtifact": None, "entries": {}, "requiredLoaded": {name: False for name in REQUIRED_LIBRARIES}},
        "ioContract": {"input": INPUT, "output": OUTPUT},
        "commands": [],
        "fixtures": [],
        "productionPostprocess": {"implementation": "src/postprocess.rs", "runner": "evidence/tooling/raw-golden", "cudaSpecificDecodeOrNms": False},
    }

    def stop(stage, code, message, missing=None):
        report["failureStage"] = stage
        report["failure"] = {"message": message, "missing": missing or []}
        report["endedAt"] = now()
        write_report(output, report)
        logical_output = logical_path(output, root)
        print(json.dumps({"state": "blocked", "failureStage": stage, "exitCode": code, "report": logical_output}))
        return code

    if report["host"]["os"] != "Linux" or report["host"]["architecture"] != "x86_64":
        return stop("host-preflight", EXIT["host"], "runner requires Linux x86_64")
    if report["model"]["bytes"] != MODEL_BYTES or report["model"]["sha256"] != MODEL_SHA256:
        return stop("model-identity", EXIT["host"], "canonical model identity mismatch")
    report["staticVerified"] = True
    for command in (
        ["uname", "-a"],
        ["cat", "/etc/os-release"],
        ["bash", "-lc", "lspci -nn | grep -Ei 'VGA|3D controller'"],
        ["bash", "-lc", "ldconfig -p | grep -E 'lib(cuda|cudnn|cublas|cudart|onnxruntime_providers_cuda)'"],
        [sys.executable, "-c", "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers()); print(ort.__file__)"],
    ):
        if shutil_which(command[0]):
            report["commands"].append(run(command, root))
    lspci_probe = next((item for item in report["commands"] if "lspci -nn" in " ".join(item["command"])), None)
    if lspci_probe:
        report["host"]["displayControllers"] = [line for line in lspci_probe["stdout"].splitlines() if re.search(r"VGA|3D controller", line, re.IGNORECASE)]
    probe = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], root) if shutil_which("nvidia-smi") else {"command": ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], "startedAt": now(), "endedAt": now(), "exitCode": 127, "stdout": "", "stderr": "nvidia-smi: command not found\n"}
    report["commands"].append(probe)
    if probe["exitCode"] != 0 or not probe["stdout"].strip():
        return stop("nvidia-hardware-driver-preflight", EXIT["hardware"], "NVIDIA GPU/driver unavailable; CUDA inference was not attempted", ["NVIDIA GPU", "NVIDIA driver / nvidia-smi", "onnxruntime-gpu", *REQUIRED_LIBRARIES])
    first_gpu = [item.strip() for item in probe["stdout"].splitlines()[0].split(",")]
    report["host"]["nvidiaGpu"] = {"name": first_gpu[0], "uuid": first_gpu[1], "computeCapability": first_gpu[3]}
    report["versions"]["nvidiaDriver"] = first_gpu[2]
    if int(first_gpu[2].split(".")[0]) < 525:
        return stop("nvidia-driver-version", EXIT["hardware"], f"CUDA 12.x requires NVIDIA driver major >= 525, got {first_gpu[2]}")
    try:
        actual_packages = {name: importlib.metadata.version(name) for name in LOCKED_PACKAGES}
    except importlib.metadata.PackageNotFoundError as error:
        return stop("runtime-packages", EXIT["runtime"], str(error), [error.name])
    if actual_packages != LOCKED_PACKAGES:
        return stop("runtime-version-lock", EXIT["runtime"], f"package version drift: {actual_packages}")
    report["versions"]["cudaRuntime"] = actual_packages["nvidia-cuda-runtime-cu12"]
    report["versions"]["cudnn"] = actual_packages["nvidia-cudnn-cu12"]
    try:
        import numpy as np
        import onnxruntime as ort
        ort.preload_dlls(directory="")
    except Exception as error:
        return stop("runtime-load", EXIT["runtime"], repr(error))
    report["versions"]["onnxruntime"] = ort.__version__
    report["availableProviders"] = ort.get_available_providers()
    if ort.__version__ != ORT_VERSION or "CUDAExecutionProvider" not in report["availableProviders"]:
        return stop("provider-availability", EXIT["runtime"], "locked CUDAExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(workspace / "ort-profile")
    try:
        session = ort.InferenceSession(str(model), sess_options=options, providers=[("CUDAExecutionProvider", PROVIDER_OPTIONS)])
        report["sessionProviders"] = session.get_providers()
        report["sessionProviderOptions"] = session.get_provider_options()
        actual_cuda_options = report["sessionProviderOptions"].get("CUDAExecutionProvider", {})
        for key, expected in PROVIDER_OPTIONS.items():
            if key not in actual_cuda_options or str(actual_cuda_options[key]).upper() != expected.upper():
                return stop("session-provider-options", EXIT["session"], f"CUDA provider option drift: {key}={actual_cuda_options.get(key)!r}, expected {expected!r}")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        actual_input = {"name": inputs[0].name, "shape": inputs[0].shape, "dtype": inputs[0].type} if len(inputs) == 1 else None
        actual_output = {"name": outputs[0].name, "shape": outputs[0].shape, "dtype": outputs[0].type} if len(outputs) == 1 else None
        if actual_input != INPUT or actual_output != OUTPUT or "CUDAExecutionProvider" not in report["sessionProviders"]:
            return stop("session-contract", EXIT["session"], f"session provider/I/O drift: {actual_input}, {actual_output}")
        report["buildVerified"] = True
        smoke = np.zeros(INPUT["shape"], dtype=np.float32)
        smoke_output = session.run([OUTPUT["name"]], {INPUT["name"]: smoke})[0]
        if list(smoke_output.shape) != OUTPUT["shape"] or not np.isfinite(smoke_output).all():
            return stop("smoke-inference", EXIT["session"], "smoke output Shape/non-finite failure")
        profile_path = Path(session.end_profiling())
    except Exception as error:
        paths, _ = loaded_libraries()
        report["sharedLibraries"]["entries"] = library_evidence(paths, report, root)
        report["sharedLibraries"]["requiredLoaded"] = library_presence(paths)
        return stop("cuda-session-or-run", EXIT["session"], repr(error), [name for name, present in report["sharedLibraries"]["requiredLoaded"].items() if not present])
    events = json.loads(profile_path.read_text())
    counts = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider", "unassigned")
        counts[provider] = counts.get(provider, 0) + 1
    report["profile"] = {"artifact": artifact(profile_path, root, format="ort-chrome-trace-json"), "nodeProviderCounts": counts, "cudaNodeCount": counts.get("CUDAExecutionProvider", 0), "cpuNodeCount": counts.get("CPUExecutionProvider", 0)}
    mapped_paths, mapped_lines = loaded_libraries()
    maps_path = workspace / "proc-self-maps.txt"
    maps_path.write_text("\n".join(mapped_lines) + "\n")
    report["sharedLibraries"]["mapsArtifact"] = artifact(maps_path, root, format="proc-self-maps-filtered")
    report["sharedLibraries"]["entries"] = library_evidence(mapped_paths, report, root)
    report["sharedLibraries"]["requiredLoaded"] = library_presence(mapped_paths)
    missing_libraries = [name for name, present in report["sharedLibraries"]["requiredLoaded"].items() if not present]
    if report["profile"]["cudaNodeCount"] == 0:
        return stop("profile-provider-proof", EXIT["profile"], "configured CUDA provider executed no profiled CUDA Node")
    if missing_libraries:
        return stop("shared-library-proof", EXIT["profile"], "required CUDA shared libraries were not observed in /proc/self/maps", missing_libraries)
    report["runtimeExecuted"] = True
    report["hostInferenceVerified"] = True
    if args.web_reference_dir is None:
        return stop("golden-inputs", EXIT["golden"], "five-fixture Web tensor directory is required after CUDA runtime verification")
    web_root = args.web_reference_dir.resolve()
    if web_root != allowed and allowed not in web_root.parents:
        return stop("golden-inputs", EXIT["golden"], "CUDA Web reference tensors must stay under .evidence/cuda")
    frozen = json.loads((root / "evidence/golden/web-reference.json").read_text())
    rust_runner = root / "evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden"
    if not rust_runner.is_file():
        return stop("production-postprocess-runner", EXIT["golden"], "production raw-golden runner is missing; build it before CUDA golden validation")
    report["productionPostprocess"]["runnerArtifact"] = artifact(rust_runner, root)
    report["productionPostprocess"]["sourceArtifacts"] = [
        artifact(root / path, root)
        for path in [
            "evidence/tooling/raw-golden/Cargo.toml",
            "evidence/tooling/raw-golden/Cargo.lock",
            "evidence/tooling/raw-golden/src/main.rs",
            "evidence/tooling/raw-golden/src/lib.rs",
            "src/postprocess.rs",
        ]
    ]
    tolerances = frozen["tolerances"]
    fixture_meta = json.loads((web_root / "manifest.json").read_text())
    report["webReferenceManifest"] = artifact(web_root / "manifest.json", root)
    for fixture in frozen["fixtures"]:
        fixture_id = fixture["id"]
        fixture_dir = web_root / fixture_id
        input_values = np.fromfile(fixture_dir / "input.f32le", dtype="<f4").reshape(INPUT["shape"])
        raw = session.run([OUTPUT["name"]], {INPUT["name"]: input_values})[0].astype("<f4", copy=False)
        reference_raw = np.fromfile(fixture_dir / "raw.f32le", dtype="<f4").reshape(OUTPUT["shape"])
        if not np.isfinite(input_values).all() or not np.isfinite(raw).all() or not np.isfinite(reference_raw).all():
            return stop("frozen-golden-finite", EXIT["golden"], f"{fixture_id}: input/reference/CUDA raw contains NaN or Infinity")
        raw_comparison = compare_raw_arrays(raw, reference_raw, tolerances["rawTensorAbsolute"], tolerances["rawTensorRelative"])
        raw_path = workspace / f"{fixture_id}.raw.f32le"
        decoded_path = workspace / f"{fixture_id}.decoded.json"
        raw.tofile(raw_path)
        image = next(item["image"] for item in fixture_meta["fixtures"] if item["id"] == fixture_id)
        rust = run([str(rust_runner), str(raw_path), str(image["width"]), str(image["height"]), str(decoded_path)], root)
        if rust["exitCode"] != 0:
            return stop("production-postprocess", EXIT["golden"], rust["stderr"])
        decoded = json.loads(decoded_path.read_text())
        decoded_comparison = compare_decoded(decoded, fixture["runs"][0]["decoded"], tolerances)
        passed = raw_comparison["passed"] and decoded_comparison["passed"]
        input_path = fixture_dir / "input.f32le"
        reference_path = fixture_dir / "raw.f32le"
        report["fixtures"].append({
            "id": fixture_id,
            "image": image,
            "input": artifact(input_path, root, shape=INPUT["shape"], dtype="float32-le", elementCount=int(np.prod(INPUT["shape"]))),
            "referenceRaw": artifact(reference_path, root, shape=OUTPUT["shape"], dtype="float32-le", elementCount=int(np.prod(OUTPUT["shape"]))),
            "cudaRaw": artifact(raw_path, root, shape=OUTPUT["shape"], dtype="float32-le", elementCount=int(np.prod(OUTPUT["shape"]))),
            "decoded": artifact(decoded_path, root, format="rimeflow-detections-json", detectionCount=len(decoded)),
            "rawComparison": raw_comparison,
            "decodedComparison": decoded_comparison,
            "productionRust": rust,
            "passed": passed,
        })
        if not passed:
            return stop("frozen-golden-tolerance", EXIT["golden"], f"{fixture_id}: CUDA output exceeded frozen tolerance")
    report["goldenExecuted"] = len(report["fixtures"]) == 5
    report["state"] = "host-inference-verified"
    report["endedAt"] = now()
    write_report(output, report)
    logical_output = logical_path(output, root)
    print(json.dumps({"state": report["state"], "exitCode": 0, "report": logical_output}))
    return EXIT["ok"]


def shutil_which(name):
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    sys.exit(main())
