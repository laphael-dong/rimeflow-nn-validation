#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "evidence/reports/android-litert-device-validation-report.json"
SCRIPT_PATH = ROOT / "evidence/scripts/android_litert_device_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("android_litert_device_report", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Android LiteRT device report validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_rejected(module, report: dict[str, object], label: str) -> None:
    try:
        module.validate_report(report)
    except ValueError:
        return
    raise AssertionError(f"guard did not reject {label}")


def main() -> int:
    module = load_module()
    report = json.loads(REPORT_PATH.read_text())
    module.validate_report(report)

    changed_tolerance = copy.deepcopy(report)
    changed_tolerance["goldenComparison"]["tolerances"]["rawTensorAbsolute"] = 0.1
    expect_rejected(module, changed_tolerance, "changed frozen tolerance")

    hidden_mismatch = copy.deepcopy(report)
    hidden_mismatch["goldenComparison"]["summary"]["rawMismatchCount"] = 1
    expect_rejected(module, hidden_mismatch, "hidden raw mismatch")

    performance_overclaim = copy.deepcopy(report)
    performance_overclaim["performance"]["performancePassed"] = True
    expect_rejected(module, performance_overclaim, "performance pass overclaim")

    supported_overclaim = copy.deepcopy(report)
    supported_overclaim["status"]["supported"] = True
    expect_rejected(module, supported_overclaim, "supported-status overclaim")

    fallback_overclaim = copy.deepcopy(report)
    fallback_overclaim["faultContract"]["initialization"]["fallbackExecutionClaimed"] = True
    expect_rejected(module, fallback_overclaim, "fallback execution overclaim")

    print("Android LiteRT device report guard tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
