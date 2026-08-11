#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path


EXPECTED_PT_BYTES = 6_549_796
EXPECTED_PT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_PYTHON_VERSION = "3.12.3"
EXPECTED_VERSIONS = {
    "coremltools": "9.0",
    "numpy": "2.3.5",
    "torch": "2.7.0+cpu",
    "torchvision": "0.22.0+cpu",
    "ultralytics": "8.4.104",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def enum_name(message: object, field_name: str, value: int) -> str:
    field = message.DESCRIPTOR.fields_by_name[field_name]
    return field.enum_type.values_by_number[value].name


def tensor_shape(tensor_type: object) -> list[int | str]:
    result = []
    for dimension in tensor_type.dimensions:
        kind = dimension.WhichOneof("dimension")
        result.append(int(dimension.constant.size) if kind == "constant" else kind)
    return result


def immediate_floats(operation: object) -> list[float]:
    value = operation.attributes.get("val")
    if value is None or value.WhichOneof("value") != "immediateValue":
        return []
    tensor = value.immediateValue.tensor
    return list(tensor.floats.values) if tensor.WhichOneof("value") == "floats" else []


def operation_inputs(operation: object) -> dict[str, list[str]]:
    return {
        key: [binding.name for binding in argument.arguments if binding.WhichOneof("binding") == "name"]
        for key, argument in operation.inputs.items()
    }


def package_tree(package: Path) -> dict[str, object]:
    files = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(package).as_posix(),
                "sha256": sha256(path),
            }
        )
    return {
        "canonicalization": "sorted POSIX relative path + bytes + SHA-256; compact JSON with sorted keys; directory timestamps excluded",
        "digest": stable_digest(files),
        "fileCount": len(files),
        "files": files,
        "totalFileBytes": sum(item["bytes"] for item in files),
    }


def normalize_package_manifest(package: Path) -> dict[str, object]:
    manifest = json.loads((package / "Manifest.json").read_text())
    entries = sorted(manifest["itemInfoEntries"].values(), key=lambda item: item["path"])
    root_id = manifest["rootModelIdentifier"]
    return {
        "fileFormatVersion": manifest["fileFormatVersion"],
        "itemsByPath": entries,
        "rootModelPath": manifest["itemInfoEntries"][root_id]["path"],
    }


def inspect_spec(package: Path) -> dict[str, object]:
    import coremltools as ct
    from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
    from coremltools.proto import MIL_pb2, Model_pb2

    model = ct.models.MLModel(str(package), skip_model_load=True)
    spec = model.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        raise SystemExit(f"expected mlProgram, got {spec.WhichOneof('Type')}")
    if len(spec.description.input) != 1 or len(spec.description.output) != 1:
        raise SystemExit("expected exactly one Core ML input and output")

    feature_input = spec.description.input[0]
    feature_output = spec.description.output[0]
    input_kind = feature_input.type.WhichOneof("Type")
    output_kind = feature_output.type.WhichOneof("Type")
    if input_kind != "imageType" or output_kind != "multiArrayType":
        raise SystemExit(f"unexpected Core ML feature types: {input_kind} -> {output_kind}")

    function = spec.mlProgram.functions["main"]
    if len(function.block_specializations) != 1:
        raise SystemExit("expected one main block specialization")
    opset, block = next(iter(function.block_specializations.items()))
    operations = list(block.operations)
    op_counts = Counter(operation.type for operation in operations)
    output_dtypes = Counter()
    blob_dtypes = Counter()
    for operation in operations:
        for output in operation.outputs:
            if output.type.WhichOneof("type") == "tensorType":
                output_dtypes[MIL_pb2.DataType.Name(output.type.tensorType.dataType)] += 1
        value = operation.attributes.get("val")
        if value is not None and value.WhichOneof("value") == "blobFileValue":
            if operation.outputs and operation.outputs[0].type.WhichOneof("type") == "tensorType":
                dtype = MIL_pb2.DataType.Name(operation.outputs[0].type.tensorType.dataType)
                blob_dtypes[dtype] += 1

    function_input = function.inputs[0]
    function_input_tensor = function_input.type.tensorType
    scale_constant = None
    scale_operation = None
    for operation in operations:
        values = immediate_floats(operation)
        if len(values) == 1 and abs(values[0] - 1 / 255) < 1e-9 and operation.outputs:
            candidate = operation.outputs[0].name
            for consumer in operations:
                inputs = operation_inputs(consumer)
                if consumer.type == "mul" and candidate in inputs.get("y", []) and feature_input.name in inputs.get("x", []):
                    scale_constant = values[0]
                    scale_operation = consumer
                    break
        if scale_operation is not None:
            break
    if scale_operation is None:
        raise SystemExit("Core ML image scale 1/255 was not found in the real ML Program graph")

    dbox_operation = next(
        (operation for operation in operations if any(output.name == "dbox" for output in operation.outputs)),
        None,
    )
    if dbox_operation is None or dbox_operation.type != "mul":
        raise SystemExit("decoded bbox stride multiplication was not found")
    dbox_inputs = operation_inputs(dbox_operation)
    stride_name = next(name for name in dbox_inputs["y"] if name != "var_907")
    stride_const = next(
        operation for operation in operations if any(output.name == stride_name for output in operation.outputs)
    )
    stride_value = stride_const.attributes["val"]
    weights_path = package / "Data/com.apple.CoreML/weights/weight.bin"
    strides = BlobReader(str(weights_path)).read_float_data(stride_value.blobFileValue.offset)
    stride_counts = Counter(float(value) for value in strides.tolist())

    output_concat = next(
        operation for operation in operations if any(output.name == feature_output.name for output in operation.outputs)
    )
    output_concat_inputs = operation_inputs(output_concat)
    if output_concat.type != "concat" or "dbox" not in output_concat_inputs.get("values", []):
        raise SystemExit("Core ML output is not the expected raw bbox/class concat")

    image = feature_input.type.imageType
    multi_array = feature_output.type.multiArrayType
    nms_ops = sorted(
        operation.type for operation in operations
        if "nms" in operation.type.lower() or "non_maximum" in operation.type.lower()
    )
    normalized_spec = Model_pb2.Model()
    normalized_spec.CopyFrom(spec)
    volatile_metadata = {}
    for key in ["date"]:
        if key in normalized_spec.description.metadata.userDefined:
            volatile_metadata[key] = normalized_spec.description.metadata.userDefined[key]
            del normalized_spec.description.metadata.userDefined[key]

    input_dtype = MIL_pb2.DataType.Name(function_input_tensor.dataType)
    output_dtype = enum_name(multi_array, "dataType", multi_array.dataType)
    return {
        "computePrecision": {
            "actual": "FLOAT32",
            "blobConstantDtypes": dict(sorted(blob_dtypes.items())),
            "float16Present": output_dtypes.get("FLOAT16", 0) > 0 or blob_dtypes.get("FLOAT16", 0) > 0,
            "operationOutputDtypes": dict(sorted(output_dtypes.items())),
            "requestedHalf": False,
            "requestedQuantization": None,
        },
        "coordinates": {
            "bboxEncoding": "xywh in 640x640 model-input pixel units",
            "evidence": "dbox is multiplied by the real stride tensor and concatenated directly with 80 sigmoid class scores; no later coordinate normalization op exists",
            "outputConcatInputs": output_concat_inputs.get("values", []),
            "strideTensor": {
                "counts": {str(int(key)): value for key, value in sorted(stride_counts.items())},
                "dtype": "FLOAT32",
                "name": stride_name,
                "shape": [1, 8400],
                "uniqueValues": sorted(stride_counts),
            },
        },
        "input": {
            "bias": [0.0, 0.0, 0.0],
            "colorSpace": enum_name(image, "colorSpace", image.colorSpace),
            "featureType": "IMAGE",
            "functionTensorDtype": input_dtype,
            "height": int(image.height),
            "index": 0,
            "layout": "RGB image feature; ML Program function tensor NCHW",
            "name": feature_input.name,
            "scale": scale_constant,
            "shape": tensor_shape(function_input_tensor),
            "width": int(image.width),
        },
        "minimumDeploymentTarget": {
            "derivation": "real specificationVersion 6 / CoreML5 ML Program",
            "iOS": "15.0",
            "macOS": "12.0",
            "tvOS": "15.0",
            "watchOS": "8.0",
        },
        "modelType": "mlProgram",
        "nms": {
            "fused": len(nms_ops) > 0,
            "operators": nms_ops,
            "responsibility": "operator postprocess",
        },
        "opset": opset,
        "operations": {
            "count": len(operations),
            "types": dict(sorted(op_counts.items())),
        },
        "output": {
            "count": len(spec.description.output),
            "dtype": output_dtype,
            "featureType": "MULTI_ARRAY",
            "index": 0,
            "layout": "N_ATTRIBUTES_ANCHORS",
            "name": feature_output.name,
            "shape": [int(value) for value in multi_array.shape],
        },
        "preprocessing": {
            "colorConversion": "Core ML RGB Image feature",
            "fused": ["RGB image feature conversion", "multiply by 1/255"],
            "notFused": ["letterbox resize", "letterbox padding"],
            "scaleGraphEvidence": {
                "constant": scale_constant,
                "input": feature_input.name,
                "operation": scale_operation.type,
                "output": scale_operation.outputs[0].name,
            },
        },
        "specificationVersion": int(spec.specificationVersion),
        "volatileMetadata": volatile_metadata,
        "normalizedSpecSha256": hashlib.sha256(
            normalized_spec.SerializeToString(deterministic=True)
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = args.pt.resolve(strict=True)
    before = source.stat()
    source_sha256 = sha256(source)
    if before.st_size != EXPECTED_PT_BYTES or source_sha256 != EXPECTED_PT_SHA256:
        raise SystemExit(
            f"checkpoint mismatch: expected {EXPECTED_PT_BYTES}/{EXPECTED_PT_SHA256}, "
            f"got {before.st_size}/{source_sha256}"
        )
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise SystemExit(
            f"Python version mismatch: expected {EXPECTED_PYTHON_VERSION}, got {platform.python_version()}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(output_dir / "ultralytics-config")

    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    versions = {name: version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise SystemExit(f"conversion environment version mismatch: {versions}")

    model = YOLO(str(source))
    model.model.pt_path = str(output_dir / "yolov8n.pt")
    result = Path(
        model.export(
            format="coreml",
            batch=1,
            imgsz=640,
            dynamic=False,
            nms=False,
            device="cpu",
            half=False,
            quantize=None,
        )
    ).resolve(strict=True)
    if result.suffix != ".mlpackage" or not result.is_dir():
        raise SystemExit(f"expected .mlpackage directory, got {result}")

    spec = inspect_spec(result)
    tree = package_tree(result)
    package_manifest = normalize_package_manifest(result)
    after = source.stat()
    after_sha256 = sha256(source)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or source_sha256 != after_sha256
    ):
        raise SystemExit("source checkpoint changed during Core ML conversion")

    print(
        json.dumps(
            {
                "artifact": {
                    "format": "Core ML ML Program .mlpackage",
                    "path": str(result),
                    "trackedByGit": False,
                    "tree": tree,
                },
                "commandParameters": {
                    "batch": 1,
                    "device": "cpu",
                    "dynamic": False,
                    "format": "coreml",
                    "half": False,
                    "imgsz": 640,
                    "nms": False,
                    "quantize": None,
                },
                "conversion": {
                    "computePrecisionRequestedByExporter": "coremltools.precision.FLOAT32",
                    "intermediates": [
                        "torch.jit.trace(model.eval(), example_input, strict=False) in memory",
                        "coremltools MIL Program in memory",
                    ],
                    "minimumDeploymentTargetArgument": None,
                    "path": "Ultralytics YOLO.export(format='coreml') -> torch.jit.trace -> coremltools.convert(convert_to='mlprogram')",
                },
                "host": {
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "packageManifest": {
                    "normalized": package_manifest,
                    "normalizedSha256": stable_digest(package_manifest),
                },
                "source": {
                    "after": {
                        "bytes": after.st_size,
                        "mtimeNs": after.st_mtime_ns,
                        "sha256": after_sha256,
                    },
                    "before": {
                        "bytes": before.st_size,
                        "mtimeNs": before.st_mtime_ns,
                        "sha256": source_sha256,
                    },
                    "logicalPath": "$HANDOFF_ASSETS/yolov8n.pt",
                },
                "spec": spec,
                "versions": versions,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
