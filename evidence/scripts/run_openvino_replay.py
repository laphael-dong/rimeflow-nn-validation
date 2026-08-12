#!/usr/bin/env python3
"""记录或重放 Linux x86_64 ONNX Runtime OpenVINO EP 真实推理证据。"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_BYTES = 12_851_098
MODEL_SHA256 = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
INPUT_SHAPE = [1, 3, 640, 640]
OUTPUT_SHAPE = [1, 84, 8400]
TRACKED = {
    "manifest": "evidence/conversions/openvino-ep-manifest.json",
    "report": "evidence/reports/openvino-ep-report.json",
}
WHEELS = [
    ("flatbuffers", "25.12.19", "flatbuffers-25.12.19-py2.py3-none-any.whl", "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4", "Apache-2.0", "https://files.pythonhosted.org/packages/e8/2d/d2a548598be01649e2d46231d151a6c56d10b964d94043a335ae56ea2d92/flatbuffers-25.12.19-py2.py3-none-any.whl"),
    ("mpmath", "1.3.0", "mpmath-1.3.0-py3-none-any.whl", "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c", "BSD-3-Clause", "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"),
    ("numpy", "2.5.2", "numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8", "BSD-3-Clause and bundled notices", "https://files.pythonhosted.org/packages/3a/5f/62d28cf019460c7f1394105b4d49d9911a9c444cb77ab0bd95a204c5a6de/numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"),
    ("onnxruntime-openvino", "1.24.1", "onnxruntime_openvino-1.24.1-cp312-cp312-manylinux_2_28_x86_64.whl", "d617fac2f59a6ab5ea59a788c3e1592240a129642519aaeaa774761dfe35150e", "MIT; bundled OpenVINO and third-party notices apply", "https://files.pythonhosted.org/packages/50/cf/17ba72de2df0fcba349937d2788f154397bbc2d1a2d67772a97e26f6bc5f/onnxruntime_openvino-1.24.1-cp312-cp312-manylinux_2_28_x86_64.whl"),
    ("packaging", "26.3", "packaging-26.3-py3-none-any.whl", "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c", "Apache-2.0 OR BSD-2-Clause", "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl"),
    ("protobuf", "7.35.1", "protobuf-7.35.1-cp310-abi3-manylinux2014_x86_64.whl", "74758715c53d7158fb76caf4f0cfdacc5329a4b1bb994f865d6cf302d413a1c4", "BSD-3-Clause", "https://files.pythonhosted.org/packages/e4/be/5b3cfe508bfab6761414ff944e3366eb13be4fd71efcd69450f89ba39f43/protobuf-7.35.1-cp310-abi3-manylinux2014_x86_64.whl"),
    ("sympy", "1.14.0", "sympy-1.14.0-py3-none-any.whl", "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5", "BSD-3-Clause", "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def artifact(path: Path, logical_path: str | None = None) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "path": logical_path or str(path), "sha256": sha256(path)}


def execute(command: list[str], root: Path) -> dict[str, object]:
    def sanitize(value: str) -> str:
        normalized = value.replace(str(root), "$REPO")
        return re.sub(r"(?:\$REPO/|\./)?\.evidence/openvino/[^/\s]*venv", "$OPENVINO_VENV", normalized)

    started = utc_now()
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    return {
        "command": [sanitize(item) for item in command],
        "endedAt": utc_now(),
        "exitCode": result.returncode,
        "startedAt": started,
        "stderr": sanitize(result.stderr),
        "stdout": sanitize(result.stdout),
    }


def file_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"available": False, "bytes": None, "sha256": None}
    return {"available": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def tracked_snapshot(root: Path, required: bool) -> dict[str, dict[str, object]]:
    result = {}
    for key, relative in TRACKED.items():
        item = file_snapshot(root / relative)
        item["path"] = relative
        if required and not item["available"]:
            raise RuntimeError(f"missing tracked OpenVINO evidence: {relative}")
        result[key] = item
    return result


def publish(payloads: dict[Path, bytes]) -> None:
    staged = {}
    try:
        for target, payload in payloads.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.recording-")
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged[target] = Path(temporary)
        for target, temporary in staged.items():
            os.replace(temporary, target)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def validate_workspace(root: Path, workspace: Path) -> None:
    allowed = (root / ".evidence/openvino").resolve()
    if workspace == allowed or allowed not in workspace.parents:
        raise RuntimeError("OpenVINO workspace must be a child of .evidence/openvino")
    for relative in TRACKED.values():
        tracked = (root / relative).resolve()
        if workspace == tracked or workspace in tracked.parents or tracked in workspace.parents:
            raise RuntimeError("OpenVINO workspace overlaps tracked evidence")


def parse_cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def host_identity() -> dict[str, object]:
    os_release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    glibc = platform.libc_ver()
    return {
        "architecture": platform.machine(),
        "cpuModel": parse_cpu_model(),
        "glibc": {"name": glibc[0], "version": glibc[1]},
        "kernel": platform.release(),
        "os": {"id": os_release.get("ID"), "prettyName": os_release.get("PRETTY_NAME"), "versionId": os_release.get("VERSION_ID")},
        "runnerId": platform.node(),
    }


class OvVersion(ctypes.Structure):
    _fields_ = [("buildNumber", ctypes.c_char_p), ("description", ctypes.c_char_p)]


class OvCoreVersion(ctypes.Structure):
    _fields_ = [("device_name", ctypes.c_char_p), ("version", OvVersion)]


class OvCoreVersionList(ctypes.Structure):
    _fields_ = [("versions", ctypes.POINTER(OvCoreVersion)), ("size", ctypes.c_size_t)]


class OvAvailableDevices(ctypes.Structure):
    _fields_ = [("devices", ctypes.POINTER(ctypes.c_char_p)), ("size", ctypes.c_size_t)]


def openvino_introspection(capi: Path) -> dict[str, object]:
    library = ctypes.CDLL(str(capi / "libopenvino_c.so"))
    version = OvVersion()
    if library.ov_get_openvino_version(ctypes.byref(version)) != 0:
        raise RuntimeError("ov_get_openvino_version failed")
    runtime = {"buildNumber": version.buildNumber.decode(), "description": version.description.decode()}
    library.ov_version_free(ctypes.byref(version))
    core = ctypes.c_void_p()
    if library.ov_core_create(ctypes.byref(core)) != 0:
        raise RuntimeError("ov_core_create failed")
    devices = OvAvailableDevices()
    versions = OvCoreVersionList()
    try:
        if library.ov_core_get_available_devices(core, ctypes.byref(devices)) != 0:
            raise RuntimeError("ov_core_get_available_devices failed")
        names = [devices.devices[index].decode() for index in range(devices.size)]
        if library.ov_core_get_versions_by_device_name(core, b"CPU", ctypes.byref(versions)) != 0:
            raise RuntimeError("ov_core_get_versions_by_device_name failed")
        device_versions = [
            {
                "buildNumber": versions.versions[index].version.buildNumber.decode(),
                "description": versions.versions[index].version.description.decode(),
                "device": versions.versions[index].device_name.decode(),
            }
            for index in range(versions.size)
        ]
    finally:
        library.ov_core_versions_free(ctypes.byref(versions))
        library.ov_available_devices_free(ctypes.byref(devices))
        library.ov_core_free(core)
    return {"availableDevices": names, "requestedDevice": "CPU", "runtime": runtime, "versions": device_versions}


def loaded_libraries(capi: Path) -> list[dict[str, object]]:
    wanted = {
        "libonnxruntime_providers_openvino.so",
        "libonnxruntime_providers_shared.so",
        "libopenvino.so.2541",
        "libopenvino_c.so",
        "libopenvino_intel_cpu_plugin.so",
        "libopenvino_onnx_frontend.so.2541",
        "onnxruntime_pybind11_state.cpython-312-x86_64-linux-gnu.so",
    }
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.rsplit(" ", 1)[-1]
        if candidate.startswith("/") and Path(candidate).name in wanted:
            mapped.add(Path(candidate).resolve())
    result = []
    for name in sorted(wanted):
        candidates = [item for item in mapped if item.name == name]
        if not candidates:
            raise RuntimeError(f"required ORT/OpenVINO library was not mapped by the process: {name}")
        path = candidates[0]
        item = artifact(path, f"$VENV/{path.relative_to(capi.parent.parent.parent)}")
        item["actualPath"] = str(path)
        item["componentVersion"] = "1.24.1" if name.startswith("libonnxruntime") or name.startswith("onnxruntime_") else "2025.4.1"
        item["name"] = name
        item["mappedByProcess"] = True
        result.append(item)
    return result


def compare_raw(actual: np.ndarray, expected: np.ndarray, tolerances: dict[str, object]) -> dict[str, object]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "allClose": bool(np.allclose(actual, expected, atol=tolerances["rawTensorAbsolute"], rtol=tolerances["rawTensorRelative"])),
        "elementCount": int(actual.size),
        "finiteCount": int(np.isfinite(actual).sum()),
        "maxAbsoluteDifference": float(difference.max()),
        "meanAbsoluteDifference": float(difference.mean()),
        "referenceSha256Float32Le": tensor_sha256(expected),
        "sha256Float32Le": tensor_sha256(actual),
    }


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return 0.0 if union <= 0 else intersection / union


def compare_decoded(actual: list[dict[str, object]], expected: list[dict[str, object]], tolerances: dict[str, object]) -> dict[str, object]:
    comparisons = []
    for candidate, reference in zip(actual, expected):
        box_iou = iou(candidate["bbox"], reference["bbox"])
        box_absolute = max(abs(left - right) for left, right in zip(candidate["bbox"], reference["bbox"]))
        confidence = abs(candidate["score"] - reference["score"])
        comparisons.append({
            "bboxIou": box_iou,
            "bboxMaxAbsoluteDifference": box_absolute,
            "classEqual": candidate["classId"] == reference["classId"],
            "confidenceAbsoluteDifference": confidence,
            "passed": candidate["classId"] == reference["classId"] and confidence <= tolerances["confidenceAbsolute"] and box_iou >= tolerances["boxIouMinimum"] and box_absolute <= tolerances["decodedBoxAbsolute"],
        })
    return {"actualCount": len(actual), "comparisons": comparisons, "expectedCount": len(expected), "passed": len(actual) == len(expected) and all(item["passed"] for item in comparisons)}


def profile_evidence(profile_path: Path) -> dict[str, object]:
    events = json.loads(profile_path.read_text())
    nodes = [event for event in events if event.get("cat") == "Node"]
    execution_counts = {"CPUExecutionProvider": 0, "OpenVINOExecutionProvider": 0, "unknown": 0}
    unique = {"CPUExecutionProvider": set(), "OpenVINOExecutionProvider": set(), "unknown": set()}
    normalized = []
    for event in nodes:
        provider = event.get("args", {}).get("provider")
        key = provider if provider in execution_counts else "unknown"
        execution_counts[key] += 1
        unique[key].add(event.get("name"))
        normalized.append({"name": event.get("name"), "provider": provider})
    unique_counts = {key: len(value) for key, value in unique.items()}
    execution_plan = "unknown"
    if unique_counts["OpenVINOExecutionProvider"] > 0:
        execution_plan = "partitioned" if unique_counts["CPUExecutionProvider"] > 0 else "full"
    return {
        **artifact(profile_path, str(profile_path)),
        "executionEventCounts": execution_counts,
        "executionPlan": execution_plan,
        "nodeEvents": normalized,
        "uniqueNodeCounts": unique_counts,
    }


def io_metadata(items: list[object]) -> list[dict[str, object]]:
    result = []
    for item in items:
        count = 1
        for dimension in item.shape:
            count *= dimension
        result.append({"dtype": "float32" if item.type == "tensor(float)" else item.type, "elementCount": count, "name": item.name, "shape": item.shape})
    return result


def semantic_digest(rounds: list[dict[str, object]]) -> str:
    core = []
    for round_data in rounds:
        core.append({
            "availableProviders": round_data["availableProviders"],
            "executionPlan": round_data["profile"]["executionPlan"],
            "fixtures": [{"id": item["id"], "runs": [{"decoded": run["decoded"], "raw": run["rawComparison"]} for run in item["runs"]]} for item in round_data["fixtures"]],
            "inputs": round_data["inputs"],
            "outputs": round_data["outputs"],
            "profileCounts": round_data["profile"]["executionEventCounts"],
            "sessionProviders": round_data["sessionProviders"],
        })
    return hashlib.sha256(stable_bytes(core)).hexdigest()


def comparable_manifest(value: dict[str, object]) -> dict[str, object]:
    comparable = json.loads(json.dumps(value))
    for key in ("install", "pipCheck", "pythonVersionCommand"):
        comparable["toolchain"][key].pop("startedAt", None)
        comparable["toolchain"][key].pop("endedAt", None)
    for library in comparable["runtime"]["libraries"]:
        library["actualPath"] = "$OPENVINO_VENV/" + library["actualPath"].split("/site-packages/", 1)[1]
    return comparable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".evidence/openvino/replay", type=Path)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    workspace = (root / args.workspace).resolve()
    validate_workspace(root, workspace)
    before = tracked_snapshot(root, required=not args.record)
    recorded_manifest = json.loads((root / TRACKED["manifest"]).read_text()) if not args.record else None
    recorded_report = json.loads((root / TRACKED["report"]).read_text()) if not args.record else None
    model = root / "models/yolov8n.onnx"
    if model.stat().st_size != MODEL_BYTES or sha256(model) != MODEL_SHA256:
        raise RuntimeError("canonical ONNX bytes/SHA-256 drift")
    if platform.machine() != "x86_64" or platform.system() != "Linux":
        raise RuntimeError("OpenVINO spike requires Linux x86_64")
    if ort.__version__ != "1.24.1" or np.__version__ != "2.5.2":
        raise RuntimeError("OpenVINO Python environment version drift")
    lock = root / "evidence/tooling/openvino-requirements.lock"
    wheel_root = root / ".evidence/openvino/wheels"
    wheels = []
    for name, version, filename, digest, license_name, url in WHEELS:
        wheel = wheel_root / filename
        if not wheel.is_file() or sha256(wheel) != digest:
            raise RuntimeError(f"wheel missing or drifted: {filename}")
        wheels.append({"filename": filename, "license": license_name, "name": name, "sha256": digest, "source": url, "version": version})
    install = execute([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheel_root), "--require-hashes", "--only-binary=:all:", "-r", str(lock)], root)
    pip_check = execute([sys.executable, "-m", "pip", "check"], root)
    if install["exitCode"] or pip_check["exitCode"]:
        raise RuntimeError("hash-locked OpenVINO environment verification failed")
    rust_build = execute(["cargo", "build", "--offline", "--manifest-path", "evidence/tooling/raw-golden/Cargo.toml"], root)
    if rust_build["exitCode"]:
        raise RuntimeError("production raw-golden harness build failed")
    rust_runner = root / "evidence/tooling/raw-golden/target/debug/rimeflow-raw-golden"
    fixture_manifest = json.loads((root / "evidence/fixtures/manifest.json").read_text())
    frozen = json.loads((root / "evidence/golden/web-reference.json").read_text())
    workspace.mkdir(parents=True, exist_ok=True)
    rounds = []
    model_before = artifact(model, "models/yolov8n.onnx")
    capi = Path(ort.__file__).resolve().parent / "capi"
    ov_device = openvino_introspection(capi)
    for round_number in (1, 2):
        round_root = workspace / f"round-{round_number}"
        if round_root.exists():
            shutil.rmtree(round_root)
        round_root.mkdir(parents=True)
        web_root = round_root / "web-reference"
        web = execute(["node", "evidence/scripts/export_web_reference_tensors.mjs", str(web_root)], root)
        if web["exitCode"]:
            raise RuntimeError(f"round {round_number}: Web reference export failed")
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(round_root / "ort-profile")
        started = utc_now()
        session = ort.InferenceSession(
            str(model),
            sess_options=options,
            providers=[("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"],
        )
        available = ort.get_available_providers()
        session_providers = session.get_providers()
        inputs, outputs = io_metadata(session.get_inputs()), io_metadata(session.get_outputs())
        if "OpenVINOExecutionProvider" not in available or session_providers[0] != "OpenVINOExecutionProvider":
            raise RuntimeError("OpenVINO provider unavailable or not first in session")
        if inputs != [{"dtype": "float32", "elementCount": 1_228_800, "name": "images", "shape": INPUT_SHAPE}]:
            raise RuntimeError("runtime input contract drift")
        if outputs != [{"dtype": "float32", "elementCount": 705_600, "name": "output0", "shape": OUTPUT_SHAPE}]:
            raise RuntimeError("runtime output contract drift")
        fixtures = []
        for entry in fixture_manifest["images"]:
            fixture_id = entry["id"]
            canonical_path = web_root / fixture_id / "input.f32le"
            reference_path = web_root / fixture_id / "raw.f32le"
            canonical = np.fromfile(canonical_path, dtype="<f4").reshape(INPUT_SHAPE)
            expected_raw = np.fromfile(reference_path, dtype="<f4").reshape(OUTPUT_SHAPE)
            expected_fixture = next(item for item in frozen["fixtures"] if item["id"] == fixture_id)
            runs = []
            for repeat in (1, 2):
                raw = session.run(None, {"images": canonical})
                if len(raw) != 1 or raw[0].shape != tuple(OUTPUT_SHAPE) or raw[0].dtype != np.float32:
                    raise RuntimeError(f"{fixture_id}: output contract drift")
                output = raw[0]
                raw_path = round_root / "raw" / f"{fixture_id}-{repeat}.f32le"
                decoded_path = round_root / "decoded" / f"{fixture_id}-{repeat}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                decoded_path.parent.mkdir(parents=True, exist_ok=True)
                output.astype("<f4", copy=False).tofile(raw_path)
                production = execute([str(rust_runner), str(raw_path), str(entry["width"]), str(entry["height"]), str(decoded_path)], root)
                if production["exitCode"]:
                    raise RuntimeError(f"{fixture_id}: production decode/NMS failed")
                decoded = json.loads(decoded_path.read_text())
                raw_comparison = compare_raw(output, expected_raw, frozen["tolerances"])
                decoded_comparison = compare_decoded(decoded, expected_fixture["runs"][0]["decoded"], frozen["tolerances"])
                passed = raw_comparison["allClose"] and raw_comparison["finiteCount"] == 705_600 and decoded_comparison["passed"]
                if not passed:
                    raise RuntimeError(f"{fixture_id}: frozen golden failed")
                runs.append({"decoded": decoded, "decodedComparison": decoded_comparison, "passed": passed, "productionPostprocess": production, "raw": artifact(raw_path, f".evidence/openvino/{args.workspace.name}/round-{round_number}/raw/{fixture_id}-{repeat}.f32le"), "rawComparison": raw_comparison, "repeat": repeat})
            deterministic = runs[0]["raw"]["sha256"] == runs[1]["raw"]["sha256"] and runs[0]["decoded"] == runs[1]["decoded"]
            if not deterministic:
                raise RuntimeError(f"{fixture_id}: same-round determinism failed")
            fixtures.append({"canonicalInput": artifact(canonical_path), "deterministic": deterministic, "id": fixture_id, "runs": runs})
        profile_path = Path(session.end_profiling())
        ended = utc_now()
        profile = profile_evidence(profile_path)
        if profile["uniqueNodeCounts"]["OpenVINOExecutionProvider"] < 1:
            raise RuntimeError("profile contains no OpenVINO graph node")
        rounds.append({
            "availableProviders": available,
            "command": ["$OPENVINO_VENV/bin/python", "evidence/scripts/run_openvino_replay.py", "--workspace", str(args.workspace), *( ["--record"] if args.record else [])],
            "endedAt": ended,
            "exitCode": 0,
            "fixtures": fixtures,
            "inputs": inputs,
            "outputs": outputs,
            "profile": profile,
            "providerOptions": session.get_provider_options(),
            "round": round_number,
            "sessionProviders": session_providers,
            "startedAt": started,
            "webReference": web,
        })
    if semantic_digest([rounds[0]]) != semantic_digest([rounds[1]]):
        raise RuntimeError("two-round replay results differ")
    model_after = artifact(model, "models/yolov8n.onnx")
    if model_before != model_after:
        raise RuntimeError("canonical ONNX changed during replay")
    libraries = loaded_libraries(capi)
    version_output = execute([sys.executable, "--version"], root)
    manifest = {
        "artifact": {**model_after, "format": "ONNX", "noConversion": True, "sameCanonicalFileLoaded": True},
        "ioContract": {"inputs": rounds[0]["inputs"], "outputs": rounds[0]["outputs"], "outputLayout": "N_ATTRIBUTES_ANCHORS"},
        "ownership": {"nms": "operator", "postprocessing": "src/postprocess.rs through evidence/tooling/raw-golden", "preprocessing": "runtime caller owns letterbox, RGB, NCHW and /255; canonical ONNX unchanged"},
        "provider": {"configured": ["OpenVINOExecutionProvider", "CPUExecutionProvider"], "deviceType": "CPU", "fallbackVisible": True, "requested": "OpenVINOExecutionProvider"},
        "runtime": {"buildInfo": ort.get_build_info(), "device": ort.get_device(), "libraries": libraries, "numpy": np.__version__, "onnxruntimeOpenvino": ort.__version__, "openvino": ov_device, "pip": "24.0", "python": platform.python_version()},
        "schemaVersion": 1,
        "status": {"adapterImplemented": False, "artifactVerified": True, "hostInferenceVerified": True, "packagingVerified": False, "performanceVerified": False, "state": "host-inference-verified", "supported": False, "targetPlatformClosed": False, "task14Complete": False},
        "toolchain": {"environmentCreationCommand": ["python3", "-m", "venv", ".evidence/openvino/venv"], "install": install, "lock": artifact(lock, "evidence/tooling/openvino-requirements.lock"), "pipCheck": pip_check, "pythonVersionCommand": version_output, "wheels": wheels},
        "usageScope": {"cache": "ignored .evidence/openvino only", "licenseAndRedistribution": "ONNX Runtime wheel is MIT and includes third-party notices; bundled OpenVINO runtime is Apache-2.0 with bundled notices. Preserve notices when redistributing. This spike does not redistribute the runtime or model.", "modelLicenseMetadata": "AGPL-3.0 License (https://ultralytics.com/license)", "productPackaging": "excluded", "rawTensorPublication": "prohibited", "runtimePublication": "prohibited"},
    }
    report = {
        "executionPlan": rounds[0]["profile"]["executionPlan"],
        "host": host_identity(),
        "mode": "record",
        "modelAfter": model_after,
        "modelBefore": model_before,
        "noConversion": True,
        "productionPostprocess": {"implementation": "src/postprocess.rs", "platformSpecificImplementationAdded": False},
        "recordDigest": semantic_digest(rounds),
        "rounds": rounds,
        "schemaVersion": 1,
        "status": manifest["status"],
        "tolerances": frozen["tolerances"],
    }
    if args.record:
        publish({root / TRACKED["manifest"]: stable_bytes(manifest), root / TRACKED["report"]: stable_bytes(report)})
        tracked = tracked_snapshot(root, required=True)
        result = {"mode": "record", "recordDigest": report["recordDigest"], "trackedEvidence": tracked}
    else:
        comparable = comparable_manifest(manifest)
        recorded_comparable = comparable_manifest(recorded_manifest)
        if comparable != recorded_comparable:
            keys = [key for key in comparable if comparable[key] != recorded_comparable.get(key)]
            if keys == ["toolchain"]:
                subkeys = [key for key in comparable["toolchain"] if comparable["toolchain"][key] != recorded_comparable["toolchain"].get(key)]
                (workspace / "manifest-drift.json").write_bytes(stable_bytes({"actual": {key: comparable["toolchain"][key] for key in subkeys}, "recorded": {key: recorded_comparable["toolchain"][key] for key in subkeys}}))
                raise RuntimeError(f"OpenVINO replay manifest drift: toolchain {subkeys}")
            raise RuntimeError(f"OpenVINO replay manifest drift: {keys}")
        if report["recordDigest"] != recorded_report["recordDigest"]:
            raise RuntimeError("OpenVINO replay digest differs from tracked record")
        after = tracked_snapshot(root, required=True)
        preservation = {}
        for key in TRACKED:
            unchanged = before[key]["bytes"] == after[key]["bytes"] and before[key]["sha256"] == after[key]["sha256"]
            preservation[key] = {"before": before[key], "after": after[key], "unchanged": unchanged}
            if not unchanged:
                raise RuntimeError(f"ordinary OpenVINO replay changed tracked {key}")
        replay = {"mode": "replay", "recordDigest": report["recordDigest"], "rounds": rounds, "schemaVersion": 1, "trackedEvidence": preservation}
        replay_path = workspace / "openvino-replay.json"
        replay_path.write_bytes(stable_bytes(replay))
        result = {"mode": "replay", "recordDigest": report["recordDigest"], "report": str(replay_path), "trackedEvidence": preservation}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
