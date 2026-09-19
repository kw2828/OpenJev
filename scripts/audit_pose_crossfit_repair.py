"""Saved-output-only repair for a signed scalar check in the frozen crossfit audit.

The original auditor bytes, numerical body, tolerances, gates and budgets are
unchanged. A private exact-source module receives one scalar-check replacement
and a receipt writer that discloses this repair before exclusive file writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts/audit_pose_crossfit.py"
ORIGINAL_SHA256 = "d3c16029bc27790673140d078e4fa482d50865f66759780592c9524507851955"
ORIGINAL_TEST = ROOT / "tests/test_audit_pose_crossfit.py"
ORIGINAL_TEST_SHA256 = "2a235fd8cb07cf6d77deca2739d13e325c13bc0072c3b20ac6bb52f68d845173"
REPAIR_TEST = ROOT / "tests/test_audit_pose_crossfit_repair.py"
FAILED_SHA256 = "c24b40c4dbbb69cbdba595bf3bb433185be709c9758043d1aab682ab7d522cda"
COMPLETED_SHA256 = "3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close_number(actual, expected, label):
    if (type(actual) not in (int, float) or type(expected) not in (int, float)
            or not math.isfinite(actual) or not math.isfinite(expected)):
        raise ValueError("invalid " + label)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11):
        raise ValueError(label)


def load_original():
    payload = ORIGINAL.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ORIGINAL_SHA256 or sha(ORIGINAL_TEST) != ORIGINAL_TEST_SHA256:
        raise ValueError("frozen auditor source/test changed")
    module = ModuleType("_crossfit_signed_scalar_repair")
    module.__file__ = str(ORIGINAL)
    exec(compile(payload, str(ORIGINAL), "exec"), module.__dict__)  # noqa: S102 - execute exactly pinned bytes
    return module


def metadata(failed_audit):
    if failed_audit.is_symlink() or sha(failed_audit) != FAILED_SHA256:
        raise ValueError("original failed audit digest")
    failure = json.loads(failed_audit.read_text())
    if (failure.get("status") != "failed" or failure.get("auditor_sha256") != ORIGINAL_SHA256
            or failure.get("error") != "ValueError('invalid constant position numerator')"):
        raise ValueError("original failure identity")
    return {"version": "pose-crossfit-signed-scalar-repair-v1",
        "change": "Accept signed finite scalar values in close_number; preserve rtol1e-10/atol1e-11 and all original audit/gate/search checks.",
        "original_failure_path": str(failed_audit.resolve()), "original_failure_sha256": FAILED_SHA256,
        "original_auditor_sha256": ORIGINAL_SHA256, "original_test_sha256": ORIGINAL_TEST_SHA256,
        "repair_source_path": str(Path(__file__).resolve()), "repair_source_sha256": sha(__file__),
        "repair_test_path": str(REPAIR_TEST), "repair_test_sha256": sha(REPAIR_TEST),
        "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0, "new_random_draws": 0,
        "measurement_repeated": False, "automatic_retry": False}


def repaired_writer(writer, repair, *, validate_bindings=None):
    def write(path, value):
        if Path(path).name in ("summary.json", "receipt.json", "failed.json"):
            if "audit_repair" in value:
                raise ValueError("repair metadata already present")
            if validate_bindings is not None and Path(path).name != "failed.json":
                validate_bindings()
            return writer(path, {**value, "audit_repair": repair})
        return writer(path, value)
    return write


def audit(experiment, out, *, protocol_sha256, completed_sha256, failed_audit):
    if completed_sha256 != COMPLETED_SHA256:
        raise ValueError("repair requires the original completed execution")
    if out.resolve().is_relative_to(experiment.resolve()) or out.resolve().is_relative_to(failed_audit.parent.resolve()):
        raise ValueError("repair output must be separate from original evidence")
    original = load_original()
    repair = metadata(failed_audit)
    def stable():
        if (sha(ORIGINAL) != ORIGINAL_SHA256 or sha(ORIGINAL_TEST) != ORIGINAL_TEST_SHA256
                or sha(failed_audit) != FAILED_SHA256 or sha(__file__) != repair["repair_source_sha256"]
                or sha(REPAIR_TEST) != repair["repair_test_sha256"]):
            raise ValueError("repair/source evidence changed")
    original.close_number = close_number
    original.write_json = repaired_writer(original.write_json, repair, validate_bindings=stable)
    result = original.audit(experiment, out, protocol_sha256=protocol_sha256, completed_sha256=completed_sha256)
    return {**result, "audit_repair": repair}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--failed-audit", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256,
                   completed_sha256=args.completed_sha256, failed_audit=args.failed_audit)
    print(json.dumps({k: result[k] for k in ("status", "qualification_passed", "constant_certification_pass",
        "requirements_passed", "total_requirements", "checks_passed", "total_checks", "wall_seconds")}))
