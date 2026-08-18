#!/usr/bin/env python3

import copy
import json
from pathlib import Path

from android_litert_validation_model import (
    MANIFEST_PATH,
    REPORT_PATH,
    validate_manifest,
    validate_report,
)


def rejected(name: str, action) -> None:
    try:
        action()
    except (KeyError, TypeError, ValueError):
        return
    raise AssertionError(f"negative guard did not reject {name}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    report = json.loads((root / REPORT_PATH).read_text())

    cases = []
    changed = copy.deepcopy(manifest)
    changed["artifact"]["sha256"] = "0" * 64
    cases.append(("model digest drift", changed))
    changed = copy.deepcopy(manifest)
    changed["artifact"]["architectureNeutral"] = False
    cases.append(("ABI-neutral claim removed", changed))
    changed = copy.deepcopy(manifest)
    changed["artifact"]["abiSpecificBytes"] = True
    cases.append(("ABI-specific model bytes claimed", changed))
    changed = copy.deepcopy(manifest)
    changed["consumption"]["consumers"][0]["modelPath"] = "models/x86_64/yolov8n.tflite"
    cases.append(("ABI-specific consumer path", changed))
    changed = copy.deepcopy(manifest)
    changed["runtime"]["version"] = "2.1.5"
    cases.append(("LiteRT version drift", changed))
    changed = copy.deepcopy(manifest)
    changed["replay"]["rounds"] = 1
    cases.append(("single replay round", changed))
    changed = copy.deepcopy(manifest)
    changed["ioContract"]["input"]["dtype"] = "uint8"
    cases.append(("input dtype drift", changed))
    changed = copy.deepcopy(manifest)
    changed["releaseBoundary"]["targetAccepted"] = True
    cases.append(("target accepted overclaim", changed))
    changed = copy.deepcopy(manifest)
    changed["releaseBoundary"]["supported"] = True
    cases.append(("supported overclaim", changed))
    changed = copy.deepcopy(manifest)
    changed["releaseBoundary"]["productPackaging"] = "included"
    cases.append(("product packaging permission", changed))

    for name, candidate in cases:
        rejected(name, lambda candidate=candidate: validate_manifest(root, candidate))

    changed_report = copy.deepcopy(report)
    changed_report["contractToEvidence"].pop("AC-VAL-X64-4")
    rejected("missing Contract-to-Evidence key", lambda: validate_report(root, changed_report, manifest))
    changed_report = copy.deepcopy(report)
    changed_report["remoteParity"]["statusAtRecord"] = "verified"
    rejected("premature remote parity claim", lambda: validate_report(root, changed_report, manifest))
    changed_report = copy.deepcopy(report)
    changed_report["deviationLog"] = "unreported"
    rejected("deviation log drift", lambda: validate_report(root, changed_report, manifest))

    validate_manifest(root, manifest)
    validate_report(root, report, manifest)
    print(json.dumps({"negativeCases": len(cases) + 3, "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
