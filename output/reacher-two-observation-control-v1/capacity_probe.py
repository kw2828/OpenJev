"""Retained full-shape engineering capacity measurement, never a scored run.

No numerical work occurs at import. Explicit execution authorization, separate
positive phase caps, and an externally pinned completed source-matched whole-tree
rehearsal are mandatory. One synthetic epoch has 24 real optimizer updates; nine
full 64-case learned rows and five shifted reference rows are measured. The
scientific target is three 1152-update fits and 42 rows. No fitted production
weights are read.

Run and audit preserve failures and cannot overwrite an earlier attempt. Raw
projections are estimates from measured units, not cap recommendations. A separate
reviewed sizing receipt must turn the template into a qualification receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "two_observation_full_shape_engineering_capacity_no_effectiveness_claim"
NAMESPACE = "reacher-two-observation-engineering-capacity-v1"
ARMS = ("residual_gru", "two_observation_gru", "cached_gru")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "public_kinematic", "zero", "uniform")
HELPER = "output/reacher-two-observation-control-v1/capacity_probe.py"
TEST_HELPER = "output/reacher-two-observation-control-v1/test_capacity_probe.py"
FIT_NAME = "two_observation_gru-capacity"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Explicit engineering capacity phase cap exceeded")


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"Positive finite {name}")
    return float(value)


def safe(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and "\\" not in relative and all(part not in ("", ".", "..") for part in relative.split("/")), "Safe relative file path")
    value = Path(root).resolve(strict=True)
    for part in relative.split("/"):
        value /= part
        require(not value.is_symlink(), "No symlink capacity/source members")
    require(value.is_file(), f"Missing required file: {relative}")
    return value


def bound(root, relative, digest):
    require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), "External SHA256 required")
    path = safe(root, relative)
    require(sha(path) == digest, f"Digest changed: {relative}")
    return path


def inventory(folder, deadline):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        check(deadline)
        require(not path.is_symlink(), "No symlink artifact members")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def verify_members(folder, members, deadline, *, extra=()):
    require(type(members) is dict and members, "Exact nonempty retained member inventory")
    actual = inventory(folder, deadline)
    require(set(actual) == set(members) | set(extra), "Exact retained artifact membership")
    require(all(actual[name] == value for name, value in members.items()), "Retained artifact byte/hash mismatch")


def row_manifest():
    learned = [{"panel": panel, "arm": arm, "fit": f"{arm}-capacity", "label": f"{arm}-capacity",
                "path": f"control/{panel}/{arm}-capacity", "score_mode": "geometry"}
               for panel in PANELS for arm in ARMS]
    return learned + [{"panel": "shift", "reference": name, "label": name,
                       "path": f"control/shift/{name}"} for name in REFERENCES]


def failure(out, error, begin, progress):
    try:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "invalid-completion.json")
        write(out / "failed.json", {"status": "failed", "scope": SCOPE, "error": repr(error),
            "exception_type": type(error).__name__, "progress": progress,
            "wall_seconds": time.monotonic() - begin, "automatic_retry": False,
            "notes": list(getattr(error, "__notes__", ()))})
    except BaseException as preservation_error:  # noqa: BLE001
        error.add_note(f"Failure receipt could not be retained: {preservation_error!r}")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def bind_rehearsal(root, reference, deadline):
    """Read only authenticated engineering metadata and member bytes, not weights."""
    from openjev.research import reacher_two_observation_experiment as experiment

    require(type(reference) is dict and set(reference) == {"plan_path", "audit_path", "audit_sha256", "execution_path", "qualification_path", "qualification_sha256"}, "Complete external rehearsal reference")
    qualification = read(bound(root, reference["qualification_path"], reference["qualification_sha256"]))
    require(qualification["schema"] == "reacher-two-observation-rehearsal-qualification-v1"
            and qualification["status"] == "completed" and qualification["engineering"] is True
            and qualification["saved_output_audit_completed"] is True
            and qualification["scientific_draws"] == 0 and qualification["native_replay_max_abs_error"] == 0
            and qualification["new_fits"] == 3 and qualification["inherited_models"] == 6
            and qualification["restored_final_models"] == 9 and qualification["control_rows"] == 42
            and qualification["audit_receipt_sha256"] == reference["audit_sha256"], "Completed independently qualified rehearsal")
    qualification_folder = Path(reference["qualification_path"]).parent
    for name, digest in qualification["files"].items():
        check(deadline)
        bound(root, (qualification_folder / name).as_posix(), digest)
    audit = read(bound(root, reference["audit_path"], reference["audit_sha256"]))
    plan = read(bound(root, reference["plan_path"], audit["plan_sha256"]))
    completed = read(bound(root, reference["execution_path"] + "/completed.json", audit["execution_completed_sha256"]))
    require(audit["status"] == completed["status"] == "completed" and audit["engineering"] is True
            and audit["saved_output_only"] is True and plan["engineering"] is True
            and plan["schema"] == experiment.SCHEMA and audit["source_sha256"] == plan["sources"]
            and audit["execution_members"] == completed["files"]
            and completed["plan_sha256"] == audit["plan_sha256"], "Completed engineering rehearsal identity")
    require(qualification["plan_sha256"] == audit["plan_sha256"]
            and qualification["execution_completed_sha256"] == audit["execution_completed_sha256"]
            and qualification["source_sha256"] == plan["sources"]
            and qualification["runtime"] == plan["runtime"], "Qualification exact rehearsal bindings")
    require(audit["runtime"] == plan["runtime"] == experiment.runtime(), "Source-matched current rehearsal runtime")
    require(len(plan["sources"]) == 116 and set(experiment.SOURCE_PATHS_NEW) <= set(plan["sources"]), "Complete116-source rehearsal closure")
    require(set(completed["files"]) == set(experiment.expected_members(plan["settings"])), "Complete42-row rehearsal membership")
    require({p.relative_to(Path(root) / reference["execution_path"]).as_posix()
             for p in (Path(root) / reference["execution_path"]).rglob("*") if p.is_file()}
            == set(completed["files"]) | {"completed.json"}, "No extra rehearsal execution files")
    sources = {**plan["sources"], **plan["historical_registries"]["engineering_sources"]}
    for name, digest in sources.items():
        check(deadline)
        bound(root, name, digest)
    folder = Path(reference["audit_path"]).parent
    require(set(audit["files"]) == {"summary.json", "README.md"}, "Exact completed saved-audit payload")
    for name, digest in audit["files"].items():
        bound(root, (folder / name).as_posix(), digest)
    summary = read(Path(root) / folder / "summary.json")
    # Only coverage/error fields are consulted. The engineering utility/gate is
    # irrelevant to capacity admission and is not copied into profile output.
    require(summary["saved_output_only"] is True and summary["execution_new_fits"] == 3
            and summary["coverage"]["control_rows"] == 42 and summary["coverage"]["evaluated_models"] == 9
            and summary["native_max_abs_error"] == summary["public_observer_max_abs_error"] == 0
            and summary["new_model_calls"] == summary["new_optimizer_steps"] == 0, "Completed full-tree engineering audit required")
    for name, digest in completed["files"].items():
        check(deadline)
        bound(root, reference["execution_path"] + "/" + name, digest)
    for name in (HELPER, TEST_HELPER):
        sources[name] = sha(safe(root, name))
    return {"reference": copy.deepcopy(reference), "plan": plan, "audit": audit,
            "sources": sources, "scientific_sources": dict(plan["sources"]), "runtime": plan["runtime"]}


def snapshot_sources(root, out, binding, deadline):
    for name, digest in binding["sources"].items():
        check(deadline)
        source = bound(root, name, digest)
        target = out / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as saved:
            shutil.copyfileobj(incoming, saved)
        require(sha(target) == digest, "Exact source snapshot")


def _imports(root):
    if str(Path(root) / "scripts") not in sys.path:
        sys.path.insert(0, str(Path(root) / "scripts"))
    import reacher_two_observation_study as runner
    return runner


def _measure(root, out, binding, deadline, progress):
    """Authorized numerical backend, loaded only after all prerequisite checks."""
    from dataclasses import replace

    import torch

    from openjev.research import reacher_two_observation_protocol as protocol
    from openjev.research import reacher_two_observation_streams as streams
    from openjev.research.reacher_cached_observation import CachedObservationGRUWorldModel
    from openjev.research.reacher_memory_training import make_orders
    from openjev.research.reacher_objective_training import canonical_state_hash, canonical_tensor_hash
    from openjev.research.reacher_two_observation_history import TwoObservationHistoryGRUWorldModel
    from openjev.research.reacher_world_models import GRUResidualRewardWorldModel

    runner = _imports(root)
    settings = protocol.settings(engineering=True)
    settings["rng_namespace"] = NAMESPACE
    history = copy.deepcopy(binding["plan"]["random_stream_contract"]["history"])
    literal = {"source_path": HELPER, "source_sha256": binding["sources"][HELPER],
        "numpy_registry": {}, "numpy_generators": {}, "torch_registry": {"synthetic_constructors_and_orders": 410},
        "torch_generators": streams.torch_generator_manifest({"synthetic_constructors_and_orders": 410})}
    require(not any(row["source_path"] == HELPER for row in history["literal_calls"]), "New capacity source attribution must be explicit once")
    history["literal_calls"].append(literal)
    contract = streams.stream_contract(settings, history=history)
    stream = streams.validate_stream_contract(settings, contract, history=history)
    plan = {"scope": SCOPE, "settings": settings, "random_stream_contract": contract,
            "source_sha256": binding["scientific_sources"], "profile_source_sha256": binding["sources"],
            "runtime": binding["runtime"], "rows": row_manifest(), "new_training_epochs_measured": 1,
            "target_training_epochs": 48, "training_data": "deterministic synthetic public tensors; no native or historical corpus"}
    write(out / "capacity-plan.json", plan)
    original_threads, original_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    outer = torch.get_rng_state().clone()
    models, fitted_metadata, phases = {}, {}, {}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        tick = time.perf_counter()
        for arm, cls in zip(ARMS, (GRUResidualRewardWorldModel, TwoObservationHistoryGRUWorldModel, CachedObservationGRUWorldModel), strict=True):
            progress.update(phase="synthetic_control_construction", arm=arm)
            check(deadline)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(410)
                model = cls(hidden_size=64, dt=.02, noise_std=.05, residual_reward=True).cpu().float().eval()
            before = {key: value.detach().clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                model.observation_head[2].bias.copy_(torch.tensor([1., 1., 0., 0.]))
            after = {key: value.detach().clone() for key, value in model.state_dict().items()}
            torch.save({"class": cls.__name__, "seed": 410, "before_bias_edit": before, "state": after,
                        "edited_key": "observation_head.2.bias", "edited_value": [1., 1., 0., 0.]}, out / f"{arm}.pt")
            models[arm] = model
            fitted_metadata[f"{arm}-capacity"] = {"arm": arm, "student_tensor_sha256": canonical_tensor_hash(after),
                "parameters": sum(value.numel() for value in after.values())}
        require(len({row["student_tensor_sha256"] for row in fitted_metadata.values()}) == 1, "Exactly paired synthetic GRU tensor initialization")
        phases["synthetic_control_setup_seconds"] = time.perf_counter() - tick
        write(out / "synthetic-models.json", fitted_metadata)
        tick = time.perf_counter()
        n, points = 768, 51
        clock = torch.arange(points, dtype=torch.float32)[None, :].expand(n, -1)
        case = torch.arange(n, dtype=torch.float32)[:, None]
        theta0, theta1 = .3 * torch.sin(.1 * clock + .01 * case), .2 * torch.cos(.15 * clock + .02 * case)
        packets = torch.zeros(n, points, 8, dtype=torch.float32)
        packets[..., :4] = torch.stack((theta0.cos(), theta1.cos(), theta0.sin(), theta1.sin()), -1)
        packets[..., 4], packets[..., 5], packets[..., 6] = .12, .04, 1.
        for index in range(n):
            offset = index % 4
            for start in (8 + offset, 28 + offset):
                packets[index, start:start+6, :4] = 0.
                packets[index, start:start+6, 6] = 0.
                packets[index, start:start+6, 7] = torch.arange(1, 7, dtype=torch.float32) * .02
        commands = torch.stack((.1 * torch.sin(clock[:, :50] * .2), .1 * torch.cos(clock[:, :50] * .15)), -1)
        public = {"packets": packets, "commands": commands, "rewards": -.1 - commands.square().sum(-1)}
        full_cfg = runner.training_settings(settings)
        cfg = replace(full_cfg, epochs=1)
        orders = make_orders(410, cfg, role_name="capacity/synthetic_order")
        full_orders = make_orders(410, full_cfg, role_name="capacity/synthetic_full48_order")
        initial = {key: value.detach().clone() for key, value in models["residual_gru"].state_dict().items()}
        data_hash, initial_hash = canonical_tensor_hash(public), canonical_tensor_hash(initial)
        provenance = {"scope": SCOPE, "initialization": "isolated literal410 synthetic, not original scientific pairing",
                      "public_data": "deterministic synthetic cosine/sine packets, six-step missing intervals", "new_scored_draws": 0}
        training_inputs = {"initial_weights": initial, "orders": orders, "full_orders": full_orders,
            "public_data": public, "settings": cfg.configuration(), "full_settings": full_cfg.configuration(),
            "provenance": provenance, "source_sha256": binding["sources"], "runtime": binding["runtime"],
            "initial_hash": initial_hash, "data_hash": data_hash, "orders_hash": canonical_state_hash(orders)}
        torch.save(training_inputs, out / "training-inputs.pt")
        phases["synthetic_data_order_generation_storage_seconds"] = time.perf_counter() - tick
        check(deadline)
        progress.update(phase="full48_setup_only")
        tick = time.perf_counter()
        full_trainer = runner.history_training.TwoObservationTrainer(initial, full_orders, public, settings=full_cfg,
            expected_initial_sha256=initial_hash, expected_orders_sha256=canonical_state_hash(full_orders),
            expected_data_sha256=data_hash, provenance=provenance, source_sha256=binding["sources"], runtime=binding["runtime"])
        phases["full48_trainer_setup_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        torch.save(full_trainer.export_checkpoint(), out / "full48-setup-only.pt")
        phases["full48_setup_checkpoint_storage_seconds"] = time.perf_counter() - tick
        del full_trainer
        check(deadline)
        progress.update(phase="full_shape_one_epoch_fit")
        tick = time.perf_counter()
        fit = runner.fitting.fit_one(initial, orders, public, out / "fits" / FIT_NAME, settings=cfg,
            expected_initial_sha256=initial_hash, expected_orders_sha256=canonical_state_hash(orders),
            expected_data_sha256=data_hash, provenance=provenance, source_sha256=binding["sources"], runtime=binding["runtime"],
            pair="capacity", name=FIT_NAME, deadline=deadline, progress=progress)
        phases["fit_seconds"] = time.perf_counter() - tick
        require(fit["updates"] == fit["optimizer_steps"] == 24, "Exactly24 real engineering updates")
        tick = time.perf_counter()
        inputs = []
        for step in range(50):
            check(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            runner.artifacts.save_inputs(stem, streams.draw_control_inputs(settings, step, contract=stream), f"planner/control/{step}")
            inputs.append(stem)
        phases["innovation_generation_storage_seconds"] = time.perf_counter() - tick
        rows = []
        for row in row_manifest():
            check(deadline)
            progress.update(phase="capacity_control", row=row["path"])
            cases = runner.control_cases(settings, row["panel"], stream)
            tick = time.perf_counter()
            if "arm" in row:
                helper = runner.episode.learned_control if row["arm"] == "two_observation_gru" else runner.control.learned_control
                timing = helper(settings, models[row["arm"]], row["panel"], inputs, out / row["path"], deadline, progress, cases=cases)
            else:
                timing = runner.reference_control(settings, row["panel"], row["reference"], inputs,
                    out / row["path"], deadline, progress, cases=cases, bound=stream)
            rows.append({**row, "whole_call_seconds": time.perf_counter() - tick,
                         "timing": timing, "native_control_transitions": 3200})
            write(out / "row-measurements" / (row["path"].replace("/", "__") + ".json"), rows[-1])
            check(deadline)
        require(all(canonical_tensor_hash(model.state_dict()) == fitted_metadata[f"{arm}-capacity"]["student_tensor_sha256"] for arm, model in models.items()), "Untrained control tensors unchanged; fitted capacity model never deployed")
        require(torch.equal(outer, torch.get_rng_state()), "All literal construction/order validation kept ambient Torch RNG")
        torch.save({"before": outer, "after": torch.get_rng_state()}, out / "outer-rng.pt")
        write(out / "measurement.json", {"scope": SCOPE, "rows": rows, "phases": phases,
            "actual_training_updates": 24, "actual_new_fit_count": 1, "additional_full48_setup_only_count": 1,
            "control_rows": 14, "native_control_transitions": 44800, "effectiveness_summary_computed": False,
            "limits": "Synthetic one-epoch training and untrained control workloads. Full48 setup is separately measured; no fitted production checkpoint or scientific outcome is used."})
        return {"rows": rows, "phases": phases, "fit": fit, "settings": settings}
    finally:
        torch.set_num_threads(original_threads)
        torch.use_deterministic_algorithms(original_deterministic)


def run_profile(attempt, *, root, rehearsal, cap_seconds, audit_cap_seconds, execute_engineering=False):
    require(execute_engineering is True, "Explicit engineering execution authorization required")
    attempt = Path(attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    begin, progress = time.monotonic(), {"phase": "admission"}
    try:
        write(attempt / "started.json", {"scope": SCOPE, "requested_cap_seconds": repr(cap_seconds),
            "requested_audit_cap_seconds": repr(audit_cap_seconds), "rehearsal": rehearsal,
            "namespace": NAMESPACE, "shared_host": True, "automatic_retry": False})
        require(all(type(value) is int and value > 0 for value in (cap_seconds, audit_cap_seconds)), "Explicit positive integer phase caps")
        deadline = begin + cap_seconds
        binding = bind_rehearsal(root, rehearsal, deadline)
        snapshot_sources(root, attempt, binding, deadline)
        write(attempt / "source-binding.json", {key: value for key, value in binding.items() if key not in ("plan", "audit")})
        admission_seconds = time.monotonic() - begin
        execution = attempt / "execution"
        execution.mkdir()
        result = _measure(root, execution, binding, deadline, progress)
        for name, digest in binding["sources"].items():
            check(deadline)
            bound(root, name, digest)
        bound(root, rehearsal["qualification_path"], rehearsal["qualification_sha256"])
        tick = time.perf_counter()
        files = inventory(attempt, deadline)
        hashing = time.perf_counter() - tick
        write(attempt / "completed.json", {"status": "completed", "scope": SCOPE, "engineering": True,
            "namespace": NAMESPACE, "cap_seconds": cap_seconds, "audit_cap_seconds": audit_cap_seconds,
            "rehearsal": rehearsal, "source_sha256": binding["scientific_sources"], "profile_source_sha256": binding["sources"],
            "runtime": binding["runtime"], "files": files, "row_count": len(result["rows"]),
            "new_fits": 1, "optimizer_updates": 24, "full48_setup_only_count": 1, "native_control_transitions": 44800,
            "hashing_seconds": hashing, "wall_seconds": time.monotonic() - begin,
            "admission_source_snapshot_seconds": admission_seconds,
            "bytes_retained_before_completion": sum(row["bytes"] for row in files.values()),
            "peak_process_rss_bytes": peak_rss_bytes(), "qualification_ready": False,
            "audit_status": "not_started", "sizing_review_status": "not_started"})
        check(deadline)
    except BaseException as error:
        failure(attempt, error, begin, progress)
        raise


def projection(measured, audited, fit, completed):
    """Transparent nominal scaling, not an upper bound or approved phase cap."""
    rows, checks = measured["rows"], audited["rows"]
    require(len(rows) == len(checks) == 14, "All fourteen measured and audited rows")
    for actual, saved, wanted in zip(rows, checks, row_manifest(), strict=True):
        require(all(actual[key] == value and saved[key] == value for key, value in wanted.items()), "Fixed row order/identity")
        positive(actual["whole_call_seconds"], "measured row wall")
        positive(saved["audit_seconds"], "audited row wall")
    require(measured["actual_training_updates"] == fit["updates"] == 24, "Twenty-four measured updates")
    phases = measured["phases"]
    update = positive(fit["update_call_seconds"], "update wall") + positive(fit["log_validation_flush_seconds"], "log I/O wall")
    fit_wall, constructor = positive(fit["wall_seconds"], "whole fit wall"), positive(fit["constructor_seconds"], "fit constructor")
    require(fit_wall >= update + constructor, "Nested fit time cannot exceed whole fit")
    fit_other = fit_wall - update - constructor
    full_setup = positive(phases["full48_trainer_setup_seconds"], "full48 setup")
    training = update * 144 + 3 * (full_setup + fit_other)
    learned = 3 * sum(row["whole_call_seconds"] for row in rows if "arm" in row)
    refs = 3 * sum(row["whole_call_seconds"] for row in rows if "reference" in row)
    measured_calls = sum(row["whole_call_seconds"] for row in rows) + phases["fit_seconds"]
    execution_overhead = positive(completed["wall_seconds"], "whole execution") - measured_calls - full_setup
    require(execution_overhead >= 0, "Complete execution includes every measured call")
    audit_rows = sum(row["audit_seconds"] for row in checks)
    audit_training = positive(audited["training_audit_seconds"], "training audit")
    audit_fixed = positive(audited["wall_seconds_before_report"], "audit wall") - audit_rows - audit_training
    require(audit_fixed >= 0, "Complete audit includes every measured kernel")
    return {"status": "pending_independent_sizing_review", "scope": SCOPE,
        "measured": {"training_updates": 24, "learned_rows": 9, "reference_rows": 5, "control_cases_per_row": 64, "actions_per_case": 50},
        "target": {"training_updates": 3456, "new_fits": 3, "learned_rows": 27, "reference_rows": 15, "control_rows": 42},
        "nominal_projected_seconds": {"execution": training + learned + refs + execution_overhead,
                                      "audit": 3 * audit_rows + 144 * audit_training + audit_fixed},
        "execution_components_seconds": {"training": training, "learned_control": learned, "references": refs,
                                         "measured_outer_overhead_proxy": execution_overhead},
        "audit_components_seconds": {"row_replay": 3 * audit_rows, "saved_training": 144 * audit_training,
                                     "fixed_and_hash_proxy": audit_fixed},
        "formulas": {"training": "144*(24-update call+log I/O) + 3*(full48 setup + measured finalization/other fit cost)",
            "learned_control": "3 * sum(all nine full64 rows across three classes and panels)",
            "references": "3 * sum(all five full64 shifted reference rows)",
            "audit": "3*all14 row audits + 144*oneepoch saved training audit + measured fixed/hash overhead"},
        "unmeasured_or_nonbinding": [
            "No statistical upper bound: shared-host timing, thermal effects, data and future optimizer state may change speed.",
            "Only one shifted reference row per method was measured; other sensing schedules are extrapolated.",
            "Actual scientific parent copying, six inherited/all-nine restoration and complete study receipts are not directly measured by this proxy.",
            "Training audit scaling repeats fixed tensor checks; final large-log, order and metadata costs need reviewed allowance.",
            "Outer hashing proxy covers measured bytes only; target bytes and final whole-tree rehashing need reviewed sizing.",
            "No execution/audit cap is selected here; measured raw files and independent review must justify explicit caps."],
        "qualification_ready": False}


def _audit_measurements(root, execution, out, binding, deadline, progress):
    """Saved tensor arithmetic and full native replay; never neural inference."""
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
                result["aggregate_model_work"] = state["aggregate_model_work"]
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


def audit_profile(attempt, completed_sha256, *, root, execute_engineering=False):
    require(execute_engineering is True, "Explicit engineering audit authorization required")
    attempt = Path(attempt).resolve()
    out = attempt.with_name(attempt.name + "-audit")
    out.mkdir(parents=True, exist_ok=False)
    begin, progress = time.monotonic(), {"phase": "admission"}
    try:
        write(out / "started.json", {"scope": SCOPE, "attempt": str(attempt), "completed_sha256": completed_sha256,
                                     "automatic_retry": False, "new_model_calls": 0, "new_optimizer_steps": 0})
        completed = read(bound(attempt, "completed.json", completed_sha256))
        require(completed["status"] == "completed" and completed["engineering"] is True
                and completed["namespace"] == NAMESPACE and completed["scope"] == SCOPE
                and type(completed["audit_cap_seconds"]) is int and completed["audit_cap_seconds"] > 0
                and 0 < completed["wall_seconds"] < completed["cap_seconds"]
                and completed["qualification_ready"] is False
                and all(type(completed[key]) is int and completed[key] == wanted for key, wanted in
                        (("row_count", 14), ("new_fits", 1), ("optimizer_updates", 24),
                         ("full48_setup_only_count", 1), ("native_control_transitions", 44800))), "Completed bounded engineering measurement")
        deadline = begin + completed["audit_cap_seconds"]
        tick = time.perf_counter()
        verify_members(attempt, completed["files"], deadline, extra=("completed.json",))
        bound(attempt, "completed.json", completed_sha256)
        entry_hash_seconds = time.perf_counter() - tick
        binding = bind_rehearsal(root, completed["rehearsal"], deadline)
        require(completed["source_sha256"] == binding["scientific_sources"]
                and completed["profile_source_sha256"] == binding["sources"] and completed["runtime"] == binding["runtime"], "Same source/runtime in both phases")
        result = _audit_measurements(root, attempt / "execution", out, binding, deadline, progress)
        tick = time.perf_counter()
        verify_members(attempt, completed["files"], deadline, extra=("completed.json",))
        bound(attempt, "completed.json", completed_sha256)
        for name, digest in binding["sources"].items():
            check(deadline)
            bound(root, name, digest)
        exit_hash_seconds = time.perf_counter() - tick
        result["wall_seconds_before_report"] = time.monotonic() - begin
        result["storage_hash_seconds"] = entry_hash_seconds + exit_hash_seconds
        measured = read(attempt / "execution" / "measurement.json")
        estimates = projection(measured, result, result["fit"], completed)
        write(out / "projection.json", estimates)
        from openjev.research import reacher_two_observation_protocol as protocol
        components = {
            "training": {"wall_seconds": measured["phases"]["fit_seconds"], "units": 24, "artifact": "training-audit.json"},
            "learned_control": {"wall_seconds": sum(row["whole_call_seconds"] for row in measured["rows"] if "arm" in row), "units": 9, "artifact": "measurement-audit.json"},
            "physics_references": {"wall_seconds": sum(row["whole_call_seconds"] for row in measured["rows"] if "reference" in row), "units": 5, "artifact": "measurement-audit.json"},
            "audit": {"wall_seconds": result["wall_seconds_before_report"], "units": 14, "artifact": "measurement-audit.json"},
            "storage_and_hashing": {"wall_seconds": completed["hashing_seconds"] + result["storage_hash_seconds"], "units": len(completed["files"]), "artifact": "measurement-audit.json"}}
        write(out / "measurement-audit.json", {**result, "scope": SCOPE, "measured_components": components,
            "whole_measurement": completed, "projection_sha256": sha(out / "projection.json")})
        files = {name: value["sha256"] for name, value in inventory(out, deadline).items()}
        write(out / "qualification-template.json", {"schema": "reacher-two-observation-capacity-qualification-v1",
            "status": "pending_independent_sizing_review", "engineering": True, "study": protocol.STUDY,
            "source_sha256": binding["scientific_sources"], "runtime": binding["runtime"],
            "full_shape": {key: result["settings"][key] for key in ("hidden_size", "train_episodes", "batch_size", "steps", "control_episodes", "planning_horizon", "action_block")},
            "coverage": protocol.coverage(result["settings"]), "measured_components": components,
            "projected_seconds": estimates["nominal_projected_seconds"], "files": files,
            "review_sha256": None, "wall_seconds": completed["wall_seconds"] + time.monotonic() - begin,
            "missing": ["Independent retained sizing review, including explicit unmeasured-work allowances and justified phase caps.",
                        "Final qualification must hash-bind the review, every retained measurement member, and both completed phase receipts."],
            "qualification_ready": False})
        files = inventory(out, deadline)
        write(out / "completed.json", {"status": "completed", "scope": SCOPE, "engineering": True,
            "execution_completed_sha256": completed_sha256, "source_sha256": binding["scientific_sources"],
            "profile_source_sha256": binding["sources"], "runtime": binding["runtime"], "files": files,
            "wall_seconds": time.monotonic() - begin, "cap_seconds": completed["audit_cap_seconds"],
            "new_model_calls": 0, "new_optimizer_steps": 0, "qualification_ready": False})
        check(deadline)
        return out
    except BaseException as error:
        failure(out, error, begin, progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "audit"))
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute-engineering", action="store_true")
    parser.add_argument("--rehearsal-plan")
    parser.add_argument("--rehearsal-audit")
    parser.add_argument("--rehearsal-audit-sha256")
    parser.add_argument("--rehearsal-execution")
    parser.add_argument("--rehearsal-qualification")
    parser.add_argument("--rehearsal-qualification-sha256")
    parser.add_argument("--cap-seconds", type=int)
    parser.add_argument("--audit-cap-seconds", type=int)
    parser.add_argument("--completed-sha256")
    args = parser.parse_args()
    if args.phase == "run":
        run_profile(args.attempt, root=args.root,
            rehearsal={"plan_path": args.rehearsal_plan, "audit_path": args.rehearsal_audit,
                       "audit_sha256": args.rehearsal_audit_sha256, "execution_path": args.rehearsal_execution,
                       "qualification_path": args.rehearsal_qualification,
                       "qualification_sha256": args.rehearsal_qualification_sha256},
            cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds, execute_engineering=args.execute_engineering)
    else:
        audit_profile(args.attempt, args.completed_sha256, root=args.root, execute_engineering=args.execute_engineering)


if __name__ == "__main__":
    main()
