#!/usr/bin/env python3

import copy
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "evidence/conversions/android-litert-candidate-manifest.json"
VALIDATOR = ROOT / "evidence/scripts/android_litert_candidate.py"


def run_case(name: str, manifest: dict[str, object], expected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rimeflow-litert-candidate-guard-") as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        result = subprocess.run(
            ["python3", str(VALIDATOR), "--manifest", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    if result.returncode == 0 or expected not in result.stderr:
        raise SystemExit(f"{name}: expected failure containing {expected!r}, got {result.stderr!r}")


def main() -> int:
    source = json.loads(MANIFEST_PATH.read_text())
    cases = []

    changed = copy.deepcopy(source)
    changed["model"]["artifact"]["sha256"] = "0" * 64
    cases.append(("artifact identity drift", changed, "historical TFLite SHA-256 mismatch"))

    changed = copy.deepcopy(source)
    changed["fixtures"][0]["input"]["sha256"] = "0" * 64
    cases.append(("input identity drift", changed, "input SHA-256 mismatch"))

    changed = copy.deepcopy(source)
    changed["tolerances"]["rawTensorAbsolute"] *= 2
    cases.append(("tolerance relaxation", changed, "frozen tolerances changed"))

    changed = copy.deepcopy(source)
    changed["status"]["androidTargetAccepted"] = True
    cases.append(("target acceptance overclaim", changed, "overclaims Android target acceptance"))

    changed = copy.deepcopy(source)
    changed["usageScope"]["artifactRedistributionAllowed"] = True
    cases.append(("redistribution overclaim", changed, "redistribution must remain prohibited"))

    for name, manifest, expected in cases:
        run_case(name, manifest, expected)
    print(json.dumps({"guardCases": len(cases), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
