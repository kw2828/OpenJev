"""Verify terminal saved artifacts without rendering, inference or native replay.

The caller supplies actual execution/audit subprocess return codes and external
completion hashes. This helper authenticates the pinned renderer before using
its pure saved-artifact checks. Scientific qualification is recorded unchanged:
a completed experiment may have qualification_passed=False.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
RENDERER = Path(__file__).with_name("render.py")
RENDERER_SHA = "d8c79dd3442c10ad6aeaceb3007b0e1ed0d9f97f5c74dda69566310bcb07c659"
PLAN_SHA = "23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee"
STUDY = "reacher-geometry-memory-v1"
FALLBACK_FAILURE_ROOT = Path(__file__).parent / "terminal-verification-failures"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_renderer():
    require(RENDERER.is_file() and not RENDERER.is_symlink(),
            "Exact independently reviewed renderer source required")
    source = RENDERER.read_bytes()
    require(hashlib.sha256(source).hexdigest() == RENDERER_SHA,
            "Exact independently reviewed renderer source required")
    # Execute precisely the authenticated source bytes, never a timestamp-valid
    # bytecode cache that could contain different code of the same source size.
    module = ModuleType("geometry_memory_saved_artifact_checks")
    module.__file__ = str(RENDERER)
    exec(compile(source, str(RENDERER), "exec"), module.__dict__)  # noqa: S102 - exact reviewed SHA-pinned bytes
    require(module.PLAN_SHA == PLAN_SHA and module.STUDY == STUDY, "Pinned renderer study identity")
    return module


def safe_output(args):
    out = Path(args.out)
    require(out.name == "terminal-verification.json", "Explicit terminal-verification.json filename")
    for source in (Path(args.execution), Path(args.audit), Path(args.plan).parent):
        require(not out.resolve().is_relative_to(source.resolve()), "Verification must not modify source artifact trees")
    require(not any(path.is_symlink() for path in (out, *out.parents)), "No output symlink paths")
    return out


def preserve_failure(args, error, started, out):
    directory = out.parent if out is not None else FALLBACK_FAILURE_ROOT
    failure_path = directory / f"terminal-verification.failed-{time.time_ns()}.json"
    try:
        write_exclusive(failure_path, {"status": "failed", "study": STUDY,
            "scope": "terminal_saved_artifact_verification_only", "intended_output": str(args.out),
            "execution_exit_code": args.execution_exit_code, "audit_exit_code": args.audit_exit_code,
            "expected_plan_sha256": args.expected_plan_sha256,
            "expected_execution_completed_sha256": args.expected_completion_sha256,
            "expected_audit_receipt_sha256": args.expected_audit_receipt_sha256,
            "error": repr(error), "wall_seconds": time.perf_counter() - started,
            "new_model_calls": 0, "new_native_calls": 0, "overwritten_files": 0,
            "retry_policy": "No automatic retry, no changed qualification or audit scope."})
    except BaseException as preservation_error:  # noqa: BLE001
        error.add_note(f"Could not preserve separate terminal verification failure: {preservation_error!r}")
    else:
        error.add_note(f"Terminal verification failure preserved at {failure_path}")


def verify(args):
    started, out = time.perf_counter(), None
    verifier_sha = sha(__file__)
    try:
        out = safe_output(args)
        require(not out.exists(), "Terminal verification output already exists; no overwrite")
        require(type(args.execution_exit_code) is int and args.execution_exit_code == 0
                and type(args.audit_exit_code) is int and args.audit_exit_code == 0,
                "Both actual subprocess exit codes must be exactly zero")
        require(args.expected_plan_sha256 == PLAN_SHA
                and digest(args.expected_completion_sha256) and digest(args.expected_audit_receipt_sha256),
                "Frozen plan and explicit external completion/audit SHA256 values required")
        renderer = load_renderer()
        renderer.checked(Path(args.execution) / "completed.json", args.expected_completion_sha256)
        # The exit/hash preconditions above authorize only completed-artifact
        # reads. This does not call the renderer's plotting or replay functions.
        plan, receipt, summary, inputs = renderer.authenticate(SimpleNamespace(
            completed_authorized=True, plan=args.plan, expected_plan_sha256=PLAN_SHA,
            audit=args.audit, expected_audit_receipt_sha256=args.expected_audit_receipt_sha256,
            execution=args.execution))
        require(receipt["execution_completed_sha256"] == args.expected_completion_sha256,
                "Independently supplied completion SHA must match the completed audit")
        completed_path = Path(args.execution) / "completed.json"
        renderer.checked(completed_path, args.expected_completion_sha256)
        completed = renderer.read(completed_path)
        gate, costs = summary["continuation_gate"], summary["costs"]
        require(type(gate["passed"]) is bool and len(gate["checks"]) == 25,
                "Preserve all25 audit checks, including failed qualification")
        gate_hash = canonical_hash(gate)
        payload = {"status": "completed", "study": STUDY, "engineering": False,
            "scope": "terminal_saved_artifact_verification_only_no_reaudit",
            "plan_sha256": PLAN_SHA, "execution_completed_sha256": args.expected_completion_sha256,
            "audit_receipt_sha256": args.expected_audit_receipt_sha256,
            "audit_summary_sha256": inputs["audit_summary_sha256"],
            "execution_exit_code": 0, "audit_exit_code": 0,
            "exit_code_source": "Explicit caller-supplied subprocess return codes; this helper does not supervise or rerun either process.",
            "bound_sources_verified": len(plan["sources"]), "restored_checkpoints": completed["restored_models"],
            "control_rows": completed["control_rows"],
            "native_control_transitions_checked": summary["native_control_transitions_checked"],
            "native_nominal_candidate_transitions_checked": summary["native_nominal_candidate_transitions_checked"],
            "native_nominal_selected_transitions_checked": summary["native_nominal_selected_transitions_checked"],
            "public_observer_transitions_checked": summary["public_observer_transitions_checked"],
            "native_max_abs_error": summary["native_max_abs_error"],
            "new_fits": summary["new_fits"], "new_optimizer_steps": costs["new_optimizer_steps"],
            "audit_model_calls": summary["new_model_calls"], "new_model_calls": 0, "new_native_calls": 0,
            "qualification_passed": gate["passed"], "checks_passed": sum(row["passed"] for row in gate["checks"]),
            "total_checks": 25, "continuation_gate": gate, "gate_sha256": gate_hash,
            "gate_hash_algorithm": "SHA256 of UTF8 compact sorted-key JSON with finite numeric values",
            "execution_wall_seconds": costs["execution_wall_seconds"],
            "audit_wall_seconds": costs["audit_validation_wall_seconds"],
            "execution_cap_seconds": plan["cap_seconds"], "audit_cap_seconds": plan["audit_cap_seconds"],
            "execution_member_count": inputs["execution_member_count"],
            "source_sha256": plan["sources"], "renderer_source_sha256": RENDERER_SHA,
            "verifier_source_sha256": verifier_sha}
        require(payload["bound_sources_verified"] == 90 and payload["restored_checkpoints"] == 12
                and payload["control_rows"] == 51, "Exact publication coverage")
        # Check the original external bindings again; never replace a bound
        # digest with a digest read from a potentially changed output file.
        renderer.checked(args.plan, PLAN_SHA)
        renderer.checked(Path(args.audit) / "receipt.json", args.expected_audit_receipt_sha256)
        renderer.checked(Path(args.audit) / "summary.json", inputs["audit_summary_sha256"])
        renderer.checked(completed_path, args.expected_completion_sha256)
        for name, expected in plan["sources"].items():
            renderer.checked(renderer.member(ROOT, name), expected)
        require(canonical_hash(renderer.read(Path(args.audit) / "summary.json")["continuation_gate"]) == gate_hash,
                "All25 audit gate fields unchanged through verification")
        require(sha(RENDERER) == RENDERER_SHA and sha(__file__) == verifier_sha, "Verification sources remained unchanged")
        payload.update(wall_seconds=time.perf_counter() - started,
                       created_utc=datetime.datetime.now(datetime.UTC).isoformat())
        write_exclusive(out, payload)
        return payload
    except BaseException as error:
        preserve_failure(args, error, started, out)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--execution-exit-code", required=True, type=int)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--audit-exit-code", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args)
    print(json.dumps({"status": result["status"], "qualification_passed": result["qualification_passed"],
                      "checks_passed": result["checks_passed"], "total_checks": result["total_checks"],
                      "verification_sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
