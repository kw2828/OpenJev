"""Saved-only repair v1 for the completed capacity measurement's audit.

The original measurement and failed audit remain immutable. The only numerical
schema change is legacy model_work versus history aggregate_model_work. All
training/native validators and paid-work/projection checks are reused. This
helper never calls the measurement backend, constructs a model or fits weights.
No work happens at import. Execution requires explicit audit authorization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "reacher-two-observation-capacity-audit-repair-v1"
HELPER = "output/reacher-two-observation-control-v1/capacity_audit_repair.py"
TEST_HELPER = "output/reacher-two-observation-control-v1/test_capacity_audit_repair.py"
ORIGINAL_HELPER = "output/reacher-two-observation-control-v1/capacity_probe.py"
ORIGINAL_TESTS = "output/reacher-two-observation-control-v1/test_capacity_probe.py"
ORIGINAL_HELPER_SHA = "adc72736ff277b0dee80fbeb644b30e3e1bebbb0fae03be36fde94a70728c063"
ORIGINAL_TESTS_SHA = "90b76bb5dbfcefe63da4a3ff4411cf138ba1a208a3b242e19c741b4273a29a88"
COMPLETED_SHA = "416d8b10fd8bd36758db2a03f9b62b4270cfcc079b73558b87072cf593739e13"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def load_original(root):
    """Execute only exact original source bytes, avoiding stale bytecode."""
    path = Path(root) / ORIGINAL_HELPER
    require(sha(path) == ORIGINAL_HELPER_SHA and sha(Path(root) / ORIGINAL_TESTS) == ORIGINAL_TESTS_SHA,
            "Original as-run helper and tests must remain unchanged")
    module = types.ModuleType("_original_capacity_as_run")
    module.__file__ = str(path)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ORIGINAL_HELPER_SHA, "Exact original bytes before import")
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def aggregate_model_work(state, arm):
    """Normalize exactly two already-validated return schemas, never guess."""
    require(type(state) is dict, "State-work object required")
    require(arm in ("residual_gru", "cached_gru", "two_observation_gru"), "Exact declared capacity class")
    key = "aggregate_model_work" if arm == "two_observation_gru" else "model_work"
    other = "model_work" if key == "aggregate_model_work" else "aggregate_model_work"
    require(key in state and other not in state and type(state[key]) is dict and state[key],
            "Exact unambiguous class-specific state-work schema")
    return copy.deepcopy(state[key])


def audit_measurements(original, root, execution, out, binding, deadline, progress):
    """Original saved audit body, with one explicit state-work field adapter."""
    require, read, write = original.require, original.read, original.write
    check, bound, row_manifest = original.check, original.bound, original.row_manifest
    NAMESPACE, ARMS, FIT_NAME, SCOPE = original.NAMESPACE, original.ARMS, original.FIT_NAME, original.SCOPE
    if str(Path(root) / "scripts") not in sys.path:
        sys.path.insert(0, str(Path(root) / "scripts"))
    import audit_reacher_two_observation_study as audit
    import torch

    from openjev.research import reacher_two_observation_streams as streams

    check(deadline)
    plan = read(execution / "capacity-plan.json")
    settings = plan["settings"]
    require(settings["rng_namespace"] == NAMESPACE and settings["engineering"] is True
            and plan["rows"] == row_manifest() and plan["source_sha256"] == binding["scientific_sources"]
            and plan["profile_source_sha256"] == binding["sources"] and plan["runtime"] == binding["runtime"], "Capacity plan identity")
    stream = streams.validate_stream_contract(settings, plan["random_stream_contract"], history=plan["random_stream_contract"]["history"])
    kernel = audit.kernel_plan(settings, stream)
    contexts = [audit._DEADLINE, audit.memory._DEADLINE, audit.inherited._DEADLINE, audit.previous._DEADLINE]
    tokens = [context.set(deadline) for context in contexts]
    outer = torch.get_rng_state().clone()
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    try:
        progress.update(phase="saved_training_audit")
        tick = time.perf_counter()
        payload = load(execution / "training-inputs.pt")
        folder = execution / "fits" / FIT_NAME
        fit = read(folder / "completed.json")
        require(fit["status"] == "completed" and fit["updates"] == fit["optimizer_steps"] == 24, "Full synthetic epoch completed")
        require(set(fit["files"]) == {"initial-weights.pt", "training.jsonl", "weights.pt", "checkpoint.pt"}, "Exact fit payloads")
        for name, digest in fit["files"].items():
            bound(folder, name, digest)
        logs = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
        require(audit.tensor_hash(load(folder / "initial-weights.pt")) == payload["initial_hash"], "Capacity original tensor binding")
        training = audit.training_audit.audit_training(load(folder / "checkpoint.pt"), logs,
            initial_weights=payload["initial_weights"], orders=payload["orders"], public_data=payload["public_data"],
            settings=payload["settings"], expected_initial_sha256=payload["initial_hash"],
            expected_orders_sha256=payload["orders_hash"], expected_data_sha256=payload["data_hash"],
            expected_checkpoint_sha256=fit["checkpoint_integrity_sha256"], provenance=payload["provenance"],
            source_sha256=binding["sources"], runtime=binding["runtime"], final_weights=load(folder / "weights.pt"),
            deadline_check=lambda: check(deadline))
        require(fit["settings"] == payload["settings"] and fit["source_sha256"] == binding["sources"]
                and fit["runtime"] == binding["runtime"] and fit["provenance"] == payload["provenance"], "Fit settings/source/runtime")
        body = audit.training_audit._unseal(load(execution / "full48-setup-only.pt"))
        require(body["settings"] == payload["full_settings"] and body["settings"]["epochs"] == 48
                and body["successful_updates"] == body["optimizer_steps"] == 0
                and body["cursor"] == {"epoch": 0, "batch": 0} and body["optimizer_state"]["state"] == {}
                and body["failed"] is False and body["failure"] is body["last_attempt"] is None
                and body["training_wall_seconds"] == 0
                and audit.tensor_hash(body["student_state"]) == payload["initial_hash"]
                and audit.state_hash(body["orders"]) == audit.state_hash(payload["full_orders"])
                and body["source_sha256"] == binding["sources"] and body["runtime"] == binding["runtime"], "Full48 setup performed zero updates")
        audit.training_audit.replay_orders(payload["full_orders"], payload["full_settings"], deadline_check=lambda: check(deadline))
        train_seconds = time.perf_counter() - tick
        write(out / "training-audit.json", {"scope": SCOPE, "saved_training": training, "full48_setup_updates": 0,
            "audit_seconds": train_seconds, "fit_timing": {key: value for key, value in fit.items() if key.endswith("seconds")},
            "update_phase_seconds": [row["costs"] for row in logs]})
        metadata = read(execution / "synthetic-models.json")
        require(set(metadata) == {f"{arm}-capacity" for arm in ARMS}, "All three untrained model identities")
        for arm, cls in zip(ARMS, ("GRUResidualRewardWorldModel", "TwoObservationHistoryGRUWorldModel", "CachedObservationGRUWorldModel"), strict=True):
            saved = load(execution / f"{arm}.pt")
            require(saved["class"] == cls and saved["seed"] == 410 and saved["edited_key"] == "observation_head.2.bias"
                    and saved["edited_value"] == [1., 1., 0., 0.], "Explicit synthetic geometry bias initialization")
            before, after = saved["before_bias_edit"], saved["state"]
            audit.training_audit._weights(after, audit.training_audit.parameter_shapes(64), "Synthetic control tensors")
            require(set(before) == set(after) and all(torch.equal(before[key], value) for key, value in after.items() if key != saved["edited_key"])
                    and torch.equal(after[saved["edited_key"]], torch.tensor(saved["edited_value"]))
                    and audit.tensor_hash(after) == metadata[f"{arm}-capacity"]["student_tensor_sha256"], "Only declared bias modified")
        inputs = [audit.audit_innovations(kernel, execution / "innovations" / "control" / f"{step:03d}", step) for step in range(50)]
        rows = []
        allowed = ("setup_seconds", "decision_wall_seconds", "native_step_seconds", "row_wall_seconds", "decision_seconds",
                   "search_seconds", "candidate_evaluations", "imagined_transitions", "scoring_work", "physics_work", "public_observer")
        for row in row_manifest():
            check(deadline)
            progress.update(phase="saved_row_audit", row=row["path"])
            tick = time.perf_counter()
            folder = execution / row["path"]
            records = audit.base.load_records(folder / "episodes", 64)
            cohort = audit.audit_cohort(kernel, records, row["panel"])
            if row.get("arm") == "two_observation_gru":
                checked = audit.audit_history_control(kernel, execution, folder, row, records, inputs, metadata[row["fit"]])
            elif "arm" in row:
                checked = audit.memory.audit_learned_control(kernel, folder, row, records, inputs, metadata[row["fit"]])
            else:
                checked = audit.audit_reference_control(kernel, folder, row, records, inputs)
            result = {**row, "audit_seconds": time.perf_counter() - tick, "native_replay": cohort,
                      "measured_control_work": {key: checked[key] for key in allowed if key in checked}}
            if "state_and_work" in checked:
                state = checked["state_and_work"]
                result["aggregate_model_work"] = aggregate_model_work(state, row["arm"])
                if row["arm"] == "two_observation_gru":
                    result["controller_reconstruction_seconds"] = sum(step["reconstruction_seconds"] for step in state["steps"])
            rows.append(result)
            write(out / "rows" / (row["path"].replace("/", "__") + ".json"), result)
        require(sum(row["native_replay"]["transitions"] for row in rows) == 44800
                and max(row["native_replay"]["max_abs_error"] for row in rows) == 0, "All measured native episodes replay exactly")
        physical = [row["measured_control_work"]["physics_work"] for row in rows if "reference" in row]
        require(sum(row["candidate_native_transitions_replayed"] for row in physical) == 26247168
                and sum(row["selected_native_transitions_replayed"] for row in physical) == 9600
                and max(row["max_abs_error"] for row in physical) == 0, "All measured nominal candidates and selected predictions replay exactly")
        rng = load(execution / "outer-rng.pt")
        require(torch.equal(rng["before"], rng["after"]) and torch.equal(outer, torch.get_rng_state()), "Saved/run audit ambient RNG preserved")
        return {"rows": rows, "training_audit_seconds": train_seconds, "training": training, "fit": fit,
                "settings": settings, "native_control_transitions_checked": 44800,
                "native_nominal_candidate_transitions_checked": 26247168, "native_nominal_selected_transitions_checked": 9600,
                "new_model_calls": 0, "new_optimizer_steps": 0,
                "utility_reporting": "Existing saved audit kernels validate reward arithmetic internally; no costs, prediction errors or qualification results are reported."}
    finally:
        for context, token in zip(contexts, tokens, strict=True):
            context.reset(token)


def audit_saved(attempt, completed_sha256, out, *, root, failed_audit, authorize_saved_audit=False):
    require(authorize_saved_audit is True, "Explicit saved-only audit authorization required")
    attempt, out, root = Path(attempt).resolve(), Path(out).resolve(), Path(root).resolve(strict=True)
    require(out.is_relative_to(root) and not out.is_relative_to(attempt), "Exclusive separate audit directory under repository")
    require(type(failed_audit) is dict and set(failed_audit) == {"path", "sha256"}
            and type(failed_audit["path"]) is str, "External failed audit receipt reference")
    require(not out.is_relative_to((root / failed_audit["path"]).resolve().parent), "Repair cannot write within the failed audit")
    out.mkdir(parents=True, exist_ok=False)
    begin, progress = time.monotonic(), {"phase": "admission"}
    try:
        write(out / "started.json", {"version": VERSION, "attempt": str(attempt), "completed_sha256": completed_sha256,
            "previous_failed_audit": failed_audit, "automatic_retry": False, "new_model_calls": 0, "new_optimizer_steps": 0})
        require(completed_sha256 == COMPLETED_SHA, "Only the exact completed original capacity measurement is admitted")
        original = load_original(root)
        completed = original.read(original.bound(attempt, "completed.json", completed_sha256))
        require(completed["status"] == "completed" and completed["engineering"] is True
                and completed["namespace"] == original.NAMESPACE and completed["scope"] == original.SCOPE
                and type(completed["audit_cap_seconds"]) is int and completed["audit_cap_seconds"] > 0
                and 0 < completed["wall_seconds"] < completed["cap_seconds"]
                and completed["qualification_ready"] is False
                and all(type(completed[key]) is int and completed[key] == wanted for key, wanted in
                        (("row_count", 14), ("new_fits", 1), ("optimizer_updates", 24),
                         ("full48_setup_only_count", 1), ("native_control_transitions", 44800))), "Completed bounded original engineering measurement")
        deadline = begin + completed["audit_cap_seconds"]
        require(type(failed_audit) is dict and set(failed_audit) == {"path", "sha256"}, "External failed audit receipt reference")
        failed_path = original.bound(root, failed_audit["path"], failed_audit["sha256"])
        require(not out.is_relative_to(failed_path.parent), "Repair cannot write within the failed audit")
        failed = original.read(failed_path)
        started = original.read(failed_path.parent / "started.json")
        require(failed["status"] == "failed" and failed["exception_type"] == "KeyError"
                and "aggregate_model_work" in failed["error"]
                and failed["scope"] == started["scope"] == original.SCOPE
                and failed["automatic_retry"] is False
                and started["completed_sha256"] == completed_sha256, "Exact original failed schema audit retained")
        previous = {**failed_audit, "files": original.inventory(failed_path.parent, deadline)}
        repair_sources = {name: sha(root / name) for name in (HELPER, TEST_HELPER)}
        for name, digest in repair_sources.items():
            source = original.bound(root, name, digest)
            target = out / "repair-source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as saved:
                shutil.copyfileobj(incoming, saved)
            require(sha(target) == digest, "Exact additive audit source snapshot")
        provenance = {"version": VERSION, "original_completion_sha256": completed_sha256,
            "original_helper_sha256": ORIGINAL_HELPER_SHA, "original_tests_sha256": ORIGINAL_TESTS_SHA,
            "repair_source_sha256": repair_sources, "previous_failed_audit": previous,
            "change": "Only class-specific aggregate field normalization: residual/cached model_work; history aggregate_model_work.",
            "measurement_rerun": False, "original_measurement_sources_changed": False}
        write(out / "repair-provenance.json", provenance)
        tick = time.perf_counter()
        original.verify_members(attempt, completed["files"], deadline, extra=("completed.json",))
        original.bound(attempt, "completed.json", completed_sha256)
        entry_hash_seconds = time.perf_counter() - tick
        binding = original.bind_rehearsal(root, completed["rehearsal"], deadline)
        require(completed["source_sha256"] == binding["scientific_sources"]
                and completed["profile_source_sha256"] == binding["sources"] and completed["runtime"] == binding["runtime"], "Original source/runtime retained")
        result = audit_measurements(original, root, attempt / "execution", out, binding, deadline, progress)
        tick = time.perf_counter()
        original.verify_members(attempt, completed["files"], deadline, extra=("completed.json",))
        original.bound(attempt, "completed.json", completed_sha256)
        original.verify_members(failed_path.parent, previous["files"], deadline)
        for name, digest in (binding["sources"] | repair_sources).items():
            original.check(deadline)
            original.bound(root, name, digest)
        exit_hash_seconds = time.perf_counter() - tick
        result["wall_seconds_before_report"] = time.monotonic() - begin
        result["storage_hash_seconds"] = entry_hash_seconds + exit_hash_seconds
        measured = original.read(attempt / "execution/measurement.json")
        estimates = original.projection(measured, result, result["fit"], completed)
        write(out / "projection.json", estimates)
        from openjev.research import reacher_two_observation_protocol as protocol
        components = {
            "training": {"wall_seconds": measured["phases"]["fit_seconds"], "units": 24, "artifact": "training-audit.json"},
            "learned_control": {"wall_seconds": sum(row["whole_call_seconds"] for row in measured["rows"] if "arm" in row), "units": 9, "artifact": "measurement-audit.json"},
            "physics_references": {"wall_seconds": sum(row["whole_call_seconds"] for row in measured["rows"] if "reference" in row), "units": 5, "artifact": "measurement-audit.json"},
            "audit": {"wall_seconds": result["wall_seconds_before_report"], "units": 14, "artifact": "measurement-audit.json"},
            "storage_and_hashing": {"wall_seconds": completed["hashing_seconds"] + result["storage_hash_seconds"], "units": len(completed["files"]), "artifact": "measurement-audit.json"}}
        write(out / "measurement-audit.json", {**result, "scope": original.SCOPE, "measured_components": components,
            "whole_measurement": completed, "projection_sha256": sha(out / "projection.json"), "audit_repair": provenance})
        files = {name: value["sha256"] for name, value in original.inventory(out, deadline).items()}
        write(out / "qualification-template.json", {"schema": "reacher-two-observation-capacity-qualification-v1",
            "status": "pending_independent_sizing_review", "engineering": True, "study": protocol.STUDY,
            "source_sha256": binding["scientific_sources"], "runtime": binding["runtime"],
            "full_shape": {key: result["settings"][key] for key in ("hidden_size", "train_episodes", "batch_size", "steps", "control_episodes", "planning_horizon", "action_block")},
            "coverage": protocol.coverage(result["settings"]), "measured_components": components,
            "projected_seconds": estimates["nominal_projected_seconds"], "files": files,
            "review_sha256": None, "wall_seconds": completed["wall_seconds"] + time.monotonic() - begin,
            "missing": ["Independent sizing review, unmeasured-work allowances and justified phase caps.",
                        "Final qualification must retain original measurement, failed phases, repaired audit and source snapshots."],
            "qualification_ready": False, "audit_repair": provenance})
        files = original.inventory(out, deadline)
        write(out / "completed.json", {"status": "completed", "version": VERSION, "scope": original.SCOPE, "engineering": True,
            "execution_completed_sha256": completed_sha256, "source_sha256": binding["scientific_sources"],
            "profile_source_sha256": binding["sources"], "runtime": binding["runtime"], "files": files,
            "wall_seconds": time.monotonic() - begin, "cap_seconds": completed["audit_cap_seconds"],
            "new_model_calls": 0, "new_optimizer_steps": 0, "qualification_ready": False, "audit_repair": provenance})
        original.check(deadline)
        return out
    except BaseException as error:
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "invalid-completion.json")
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Could not invalidate completion: {preservation_error!r}")
        try:
            write(out / "failed.json", {"status": "failed", "version": VERSION, "error": repr(error),
                "exception_type": type(error).__name__, "progress": progress, "automatic_retry": False,
                "wall_seconds": time.monotonic() - begin, "notes": list(getattr(error, "__notes__", ()))})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Could not preserve failure: {preservation_error!r}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--failed-audit-receipt", required=True)
    parser.add_argument("--failed-audit-sha256", required=True)
    parser.add_argument("--authorize-saved-audit", action="store_true")
    args = parser.parse_args()
    audit_saved(args.attempt, args.completed_sha256, args.out, root=args.root,
        failed_audit={"path": args.failed_audit_receipt, "sha256": args.failed_audit_sha256},
        authorize_saved_audit=args.authorize_saved_audit)


if __name__ == "__main__":
    main()
