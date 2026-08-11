#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import struct
import sys
import zipfile
from pathlib import Path


EXPECTED_PT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
EXPECTED_PYTHON_VERSION = "3.12.3"
EXPECTED_VERSIONS = {
    "ai-edge-litert": "2.1.6",
    "ai-edge-quantizer": "0.8.0",
    "litert-converter": "0.3.1",
    "litert-torch": "0.9.3",
    "torch": "2.12.1+cpu",
    "torchvision": "0.27.1+cpu",
    "ultralytics": "8.4.104",
}
FIXED_METADATA_DATE = "2026-08-11T00:00:00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def normalize_metadata_timestamp(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "metadata.json":
            raise SystemExit("unexpected Ultralytics metadata archive layout")
        graph_bytes = entries[0].header_offset

    payload = bytearray(path.read_bytes())
    central_offset = payload.find(b"PK\x01\x02", graph_bytes)
    if payload[graph_bytes : graph_bytes + 4] != b"PK\x03\x04" or central_offset < 0:
        raise SystemExit("unable to locate Ultralytics metadata ZIP headers")

    year, month, day, hour, minute, second = (2026, 8, 11, 0, 0, 0)
    dos_time = (hour << 11) | (minute << 5) | (second // 2)
    dos_date = ((year - 1980) << 9) | (month << 5) | day
    timestamp = struct.pack("<HH", dos_time, dos_date)
    payload[graph_bytes + 10 : graph_bytes + 14] = timestamp
    payload[central_offset + 12 : central_offset + 16] = timestamp
    graph_sha256 = hashlib.sha256(payload[:graph_bytes]).hexdigest()
    path.write_bytes(payload)
    return {
        "fixedDosTimestamp": "2026-08-11T00:00:00",
        "flatbufferPrefixBytes": graph_bytes,
        "flatbufferPrefixSha256": graph_sha256,
        "modifiedFields": [
            "metadata ZIP local-header DOS timestamp",
            "metadata ZIP central-directory DOS timestamp",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = args.pt.resolve(strict=True)
    before = source.stat()
    actual_sha256 = sha256(source)
    if actual_sha256 != EXPECTED_PT_SHA256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch: expected {EXPECTED_PT_SHA256}, got {actual_sha256}"
        )
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise SystemExit(
            f"Python version mismatch: expected {EXPECTED_PYTHON_VERSION}, got {platform.python_version()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output_dir / "ultralytics-config"))
    os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "false")

    import ai_edge_litert
    import litert_torch
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    versions = {
        "ai-edge-litert": package_version("ai-edge-litert"),
        "ai-edge-quantizer": package_version("ai-edge-quantizer"),
        "litert-converter": package_version("litert-converter"),
        "litert-torch": litert_torch.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
    }
    if versions != EXPECTED_VERSIONS:
        raise SystemExit(f"conversion environment version mismatch: {versions}")

    model = YOLO(str(source))
    output_prefix = (args.output_dir / "yolov8n.pt").resolve()
    model.model.pt_path = str(output_prefix)

    def freeze_metadata(exporter):
        exporter.metadata["date"] = FIXED_METADATA_DATE

    model.add_callback("on_export_start", freeze_metadata)
    result = Path(
        model.export(
            format="litert",
            batch=1,
            imgsz=640,
            device="cpu",
            dynamic=False,
            quantize=32,
            nms=False,
        )
    ).resolve(strict=True)
    metadata_normalization = normalize_metadata_timestamp(result)

    after = source.stat()
    after_sha256 = sha256(source)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or actual_sha256 != after_sha256
    ):
        raise SystemExit("source checkpoint changed during conversion")

    print(
        json.dumps(
            {
                "artifact": {
                    "path": str(result),
                    "bytes": result.stat().st_size,
                    "sha256": sha256(result),
                },
                "commandParameters": {
                    "batch": 1,
                    "dynamic": False,
                    "format": "litert",
                    "imgsz": 640,
                    "nms": False,
                    "quantize": 32,
                },
                "converter": {
                    "component": "litert-torch",
                    "intermediate": "torch.export ExportedProgram and LiteRT compiler IR (in-memory)",
                    "metadataNormalization": metadata_normalization,
                },
                "host": {
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                },
                "runtimeModule": str(Path(ai_edge_litert.__file__).name),
                "source": {
                    "bytes": before.st_size,
                    "logicalPath": "$HANDOFF_ASSETS/yolov8n.pt",
                    "sha256": actual_sha256,
                },
                "versions": versions,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
