"""Small prospective capacity profile. No work occurs on import.

Explicit --execute-engineering is mandatory. Uses only full-width synthetic
students and the declared capacity namespace, never a fitted scored student.
Execution and saved-output replay have independent fixed 600-second caps.
No effectiveness summaries, method comparisons or scientific gate is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
CAP = 600
NAMESPACE = "reacher-geometry-memory-engineering-capacity-v1"
STEPS = (0, 12, 32, 44, 49)
ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
PANELS = ("full", "ordinary", "shift")
REHEARSAL = ROOT / "output/reacher-geometry-memory-rehearsal-v1/attempt-01"
RECEIPT_SHA = "4793f27b9839239abd48c063dfe0b56a2f63581fed9d14cb09ece7cab84779ee"
SCOPE = "engineering_capacity_only_no_effectiveness_metrics"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Fixed 600-second engineering capacity phase cap")


def inventory(folder, deadline):
    result = {}
    for path in sorted(folder.rglob("*")):
        check(deadline)
        require(not path.is_symlink(), "No symlink capacity artifacts")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def size(folder):
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def bind_sources():
    require(sha(REHEARSAL / "audit/receipt.json") == RECEIPT_SHA, "Pinned completed engineering receipt")
    receipt = read(REHEARSAL / "audit/receipt.json")
    require(receipt["status"] == "completed" and receipt["engineering"] is True, "Engineering terminal state")
    require(sha(REHEARSAL / "plan.json") == receipt["plan_sha256"], "Rehearsal plan identity")
    require(sha(REHEARSAL / "execution/completed.json") == receipt["execution_completed_sha256"], "Rehearsal completion identity")
    sources = receipt["source_sha256"]
    require(len(sources) == 90 and all(sha(ROOT / name) == digest for name, digest in sources.items()), "Current unchanged ninety sources")
    return receipt, {**sources, Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(__file__)}


def authenticated_rehearsal_overhead(receipt):
    """Only bound timing/physics-wall metadata, never scientific result fields."""
    execution = REHEARSAL / "execution"
    completed = read(execution / "completed.json")
    members = completed["files"]
    require(members == receipt["execution_members"], "Rehearsal member identity")
    residual = 0.
    for panel in PANELS:
        for reference in ("known_state", "particle", "public_kinematic"):
            relative = f"control/{panel}/{reference}/timings.json"
            require(sha(execution / relative) == members[relative], "Reference timing authentication")
            timing = read(execution / relative)
            paid = 0.
            for step in range(50):
                relative = f"control/{panel}/{reference}/physics/{step:03d}.json"
                require(sha(execution / relative) == members[relative], "Physics wall metadata authentication")
                paid += read(execution / relative)["wall_seconds"]
            residual += max(0., sum(timing["decision_seconds"]) - paid)
    return {"reference_nonsearch_decision_reserve_seconds": 64 * residual,
            "fixed_execution_reserve_seconds": completed["evaluation_started_elapsed_seconds"] + 10.,
            "fixed_audit_reserve_seconds": receipt["costs"]["audit_validation_wall_seconds"],
            "limit": "64x reference nonsearch residual includes observer and serialization overhead; adding it to compressed probe time deliberately double counts some I/O. Whole tiny audit is a conservative fixed reserve, not an isolated fixed-cost estimate."}


def horizon_total(rows, field, *, maximum=False):
    require({row["step"] for row in rows} == set(STEPS) and len(rows) == 5, "All five predeclared probes")
    values = {row["step"]: float(row[field]) for row in rows}
    require(all(math.isfinite(value) and value >= 0 for value in values.values()), "Finite probe measurements")
    twelve = sorted(values[step] for step in (0, 12, 32))[1]
    if maximum:
        return 50 * max(values.values())
    six, one = values[44], values[49]
    total = 39 * twelve
    for h in range(1, 12):
        total += one + (six - one) * (h - 1) / 5 if h <= 6 else six + (twelve - six) * (h - 6) / 6
    return total


def projection(rows, field, *, maximum=False):
    learned = sum(3 * horizon_total([row for row in rows if row["kind"] == arm and row["panel"] == panel], field, maximum=maximum)
                  for arm in ARMS for panel in PANELS)
    native = 9 * horizon_total([row for row in rows if row["kind"] == "physics"], field, maximum=maximum)
    return {"learned": learned, "physics": native, "total": learned + native}


def failure(out, error, begin, progress):
    try:
        completed = out / "completed.json"
        if completed.exists():
            completed.rename(out / "over-cap-or-finalization-failure-completion.json")
        write(out / "failed.json", {"status": "failed", "scope": SCOPE, "exception_type": type(error).__name__,
              "message": str(error), "notes": list(getattr(error, "__notes__", ())), "progress": progress,
              "wall_seconds": time.monotonic() - begin, "cap_seconds": CAP,
              "retry_policy": "No automatic retry or coverage reduction."})
    except BaseException as preservation_error:  # noqa: BLE001
        error.add_note(f"Capacity failure receipt could not be written: {preservation_error!r}")


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def close_template(template, original_error, out, begin, progress):
    if template is None:
        return
    try:
        template.close()
    except BaseException as close_error:
        if original_error is None:
            failure(out, close_error, begin, progress)
            raise
        original_error.add_note(f"Capacity template close failed: {close_error!r}")


def run_profile(attempt):
    import numpy as np
    import reacher_geometry_memory_study as runner
    import torch

    from openjev.research import reacher_geometry_memory_control as control
    from openjev.research import reacher_geometry_memory_protocol as protocol
    from openjev.research import reacher_search_protocol as artifacts
    from openjev.research.reacher_cache_training import REGISTRY
    from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
    from openjev.research.reacher_objective_training import canonical_tensor_hash
    from openjev.research.robotics_reacher import ReacherEpisode, make_env

    begin, progress = time.monotonic(), {"phase": "authenticate"}
    deadline = begin + CAP
    attempt = Path(attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    out = attempt / "execution"
    out.mkdir()
    template, active_physics, records, original_error = None, None, [], None
    active_folder, active_diagnostic, active_selected, active_root = None, None, None, None
    try:
        write(out / "started.json", {"scope": SCOPE, "cap_seconds": CAP, "started_unix_time": time.time(),
            "namespace": NAMESPACE, "synthetic_model_seed": 410, "source_models": "four untrained full-width actual classes",
            "probes": {"learned": 60, "physics": 5, "cases": 64, "steps": list(STEPS)},
            "free_disk_bytes": shutil.disk_usage(out).free, "shared_host": True,
            "cpu_brand": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=10).stdout.strip()})
        receipt, sources = bind_sources()
        overhead = authenticated_rehearsal_overhead(receipt)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        plan = protocol.settings(engineering=True)
        plan.update(engineering=True, rng_namespace=NAMESPACE,
                    engineering_rng_namespaces=[name for name in protocol.ENGINEERING_NAMESPACES if name != NAMESPACE],
                    sources={name: digest for name, digest in sources.items() if not name.startswith("output/")},
                    runtime=runner.base.runtime())
        plan["random_stream_contract"] = runner.stream_contract(plan)
        protocol.validate_settings(plan, engineering=True)
        write(out / "capacity-plan.json", plan)
        write(out / "source-binding.json", {"sources": sources, "rehearsal_receipt_sha256": RECEIPT_SHA,
            "runtime": plan["runtime"], "overhead": overhead,
            "literal_rng_scope": "Four isolated seed410 constructors; already excluded state, explicit capacity reuse. All other random draws use existing capacity protocol roles.",
            "stream_contract_scope": "The reused prospective protocol is a namespace/exclusion template. Its discarded-constructor description refers to the future12-checkpoint study, not this capacity profile. Actual constructors:4 synthetic models, seed410, outer RNG restored; no checkpoint restorations."})
        inputs = {}
        for step in STEPS:
            check(deadline)
            inputs[step] = protocol.draw_control_inputs(plan, step)
            artifacts.save_inputs(out / "innovations" / f"{step:03d}", inputs[step], f"planner/control/{step}")
        models, fits = {}, {}
        before_rng = torch.get_rng_state().clone()
        for arm in ARMS:
            progress.update(phase="construct_full_width_synthetic", arm=arm)
            check(deadline)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(410)
                initial_rng = torch.get_rng_state().clone()
                width = {"width": 107} if arm == "cached_mlp" else {"hidden_size": 64}
                model = REGISTRY[arm](**width, dt=.02, noise_std=.05, residual_reward=True).cpu().eval()
                constructor_rng = torch.get_rng_state().clone()
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            linear_layers = [(name, layer) for name, layer in model.observation_head.named_modules()
                             if isinstance(layer, torch.nn.Linear)]
            require(bool(linear_layers), "Observation head has an explicit final Linear layer")
            layer_name, final_layer = linear_layers[-1]
            require(final_layer.out_features == 4 and final_layer.bias is not None
                    and tuple(final_layer.bias.shape) == (4,), "Four-angle head bias shape")
            edited_key = ".".join(name for name in ("observation_head", layer_name, "bias") if name)
            with torch.no_grad():
                final_layer.bias.copy_(torch.tensor([1., 1., 0., 0.]))
            models[arm] = model
            fits[arm] = {"arm": arm, "parameters": sum(value.numel() for value in model.parameters()),
                         "student_tensor_sha256": canonical_tensor_hash(model.state_dict())}
            path = out / "models" / (arm + ".pt")
            path.parent.mkdir(exist_ok=True)
            torch.save({"before_bias_edit": initial, "state": model.state_dict(), "initial_rng": initial_rng,
                        "constructor_rng": constructor_rng, "seed": 410, "bias_edit": [1., 1., 0., 0.],
                        "edited_key": edited_key, "edited_shape": [4]}, path)
        require(torch.equal(before_rng, torch.get_rng_state()), "Constructors restored ambient Torch state")
        torch.save({"outer_before": before_rng, "outer_after": torch.get_rng_state()}, out / "constructor-outer-rng.pt")
        write(out / "synthetic-models.json", fits)
        collection_start = time.perf_counter()
        for index in range(64):
            progress.update(phase="zero_source_collection", case=index)
            check(deadline)
            env = ReacherEpisode(noise_std=.05)
            source_error = None
            try:
                env.reset(protocol.seed(plan, f"control/reset/{index}"), protocol.schedule(plan, index, "ordinary"),
                          noise_seed=protocol.seed(plan, f"control/actuator_noise/{index}"))
                for _ in range(50):
                    check(deadline)
                    env.step(np.zeros(2, np.float32))
                records.append(env.episode_record())
            except BaseException as error:
                source_error = error
                try:
                    control._save_records(out / f"partial-source-case-{index:03d}", [env.episode_record()])
                except BaseException as preservation_error:  # noqa: BLE001
                    error.add_note(f"Partial native source preservation failed: {preservation_error!r}")
                raise
            finally:
                close_template(env, source_error, out, begin, progress)
        control._save_records(out / "source-episodes", records)
        collection_seconds = time.perf_counter() - collection_start
        rows, prefix_seconds = [], 0.
        for panel in PANELS:
            histories = [artifacts.public_history(record, protocol.schedule(plan, index, panel), 49)
                         for index, record in enumerate(records)]
            packets = np.stack([item["packets"] for item in histories])
            commands = np.stack([item["commands"] for item in histories])
            for arm, model in models.items():
                for step in STEPS:
                    progress = {"phase": "learned_probe", "panel": panel, "arm": arm, "step": step}
                    check(deadline)
                    folder = out / "learned" / panel / arm / f"{step:03d}"
                    folder.mkdir(parents=True)
                    active_folder, active_diagnostic, active_selected, active_root = folder, None, None, None
                    tick = time.perf_counter()
                    with torch.no_grad():
                        state = model.initial(64)
                        for t in range(step + 1):
                            state = model.assimilate(state, torch.from_numpy(packets[:, t].copy()))
                            if t < step:
                                state, _, _ = model.advance(state, torch.from_numpy(commands[:, t].copy()))
                        root = state
                        active_root = {key: value.numpy().copy() for key, value in root.items()}
                        prefix_seconds += time.perf_counter() - tick
                        tick = time.perf_counter()
                        result, raw, sizes, seconds, _, diagnostic = control.score_search(plan, model, root, inputs[step], step,
                            deadline, failure_out=folder / "partial-scoring")
                        selected = {}
                        active_diagnostic, active_selected = diagnostic, selected
                        carried, angles, learned = control._selected_prediction(plan, model, root,
                            result.selected_actions, "geometry", selected)
                        diagnostic["arrays"].update(selected["arrays"])
                        diagnostic["metadata"]["selected_advance"] = selected["metadata"]
                    artifacts.save_trace(folder / "trace", result, raw, sizes, seconds)
                    artifacts.save_npz(folder / "scoring.npz", **diagnostic["arrays"])
                    write(folder / "scoring.json", diagnostic["metadata"])
                    artifacts.save_npz(folder / "states.npz", **{"root__" + k: v.numpy().copy() for k, v in root.items()},
                                       **{"carried__" + k: v.numpy().copy() for k, v in carried.items()},
                                       angles=angles.numpy(), rewards=learned.numpy())
                    rows.append({"kind": arm, "panel": panel, "step": step,
                        "wall_seconds": time.perf_counter() - tick, "bytes": size(folder), "path": folder.relative_to(out).as_posix()})
                    check(deadline)
                    active_folder, active_diagnostic, active_selected, active_root = None, None, None, None
        template = make_env()  # Static model only, no extra reset seed.
        for step in STEPS:
            progress = {"phase": "physics_probe", "step": step}
            check(deadline)
            folder = out / "physics" / f"{step:03d}"
            folder.mkdir(parents=True)
            roots = {key: np.stack([record["audit"][key][step] for record in records]) for key in ("qpos", "qvel")}
            roots["public_target"] = roots["qpos"][:, 2:].astype(np.float32)
            tick = time.perf_counter()
            active_physics = PhysicsGeometryCEM(template.unwrapped.model)
            result, journal = active_physics.plan(**roots, inputs=inputs[step], step=step, deadline=deadline)
            raw = np.concatenate([bank["arrays"]["geometry_reward"] for bank in journal["banks"]], axis=1)
            artifacts.save_trace(folder / "trace", result, raw, [64] * 4, journal["wall_seconds"])
            runner.save_physics(folder / "journal", journal)
            rows.append({"kind": "physics", "panel": "shared_native_roots", "step": step,
                "wall_seconds": time.perf_counter() - tick, "bytes": size(folder), "path": folder.relative_to(out).as_posix()})
            active_physics = None
            check(deadline)
        require(len(rows) == 65, "All60 learned and5 physics probes")
        require(all(canonical_tensor_hash(model.state_dict()) == fits[arm]["student_tensor_sha256"] for arm, model in models.items()),
                "Synthetic weights unchanged by probes")
        require(all(sha(ROOT / name) == value for name, value in sources.items()), "Sources unchanged after profile")
        hash_start = time.perf_counter()
        files = inventory(out, deadline)
        hashing_seconds = time.perf_counter() - hash_start
        point, conservative = projection(rows, "wall_seconds"), projection(rows, "wall_seconds", maximum=True)
        byte_projection = projection(rows, "bytes", maximum=True)
        extra = 51 * collection_seconds + prefix_seconds * (115200 / 109056) + overhead["reference_nonsearch_decision_reserve_seconds"] + overhead["fixed_execution_reserve_seconds"]
        expected_files = len(receipt["execution_members"])
        hash_reserve = hashing_seconds * max(byte_projection["total"] / max(1, sum(item["bytes"] for item in files.values())),
                                             expected_files / len(files))
        storage_reserve = 51 * sum((out / ("source-episodes" + extension)).stat().st_size for extension in (".npz", ".json"))
        storage_reserve += expected_files * 4096 + 100 * 2**20
        write(out / "completed.json", {"status": "completed", "scope": SCOPE, "cap_seconds": CAP,
            "plan_sha256": sha(out / "capacity-plan.json"), "source_sha256": sources, "rows": rows,
            "files": files, "new_fits": 0, "optimizer_updates": 0, "outcome_summaries_computed": False,
            "source_collection_seconds": collection_seconds, "prefix_reconstruction_seconds": prefix_seconds,
            "overhead": overhead, "hashing_seconds": hashing_seconds, "point": point, "conservative": conservative,
            "projected_probe_storage_bytes": byte_projection, "additional_execution_reserve_seconds": extra + hash_reserve,
            "expected_scientific_file_count": expected_files, "scaled_hashing_reserve_seconds": hash_reserve,
            "additional_storage_reserve_bytes": storage_reserve,
            "suggested_free_disk_bytes": max(100 * 2**30, 4 * (byte_projection["total"] + storage_reserve)),
            "peak_process_rss_bytes": peak_rss_bytes(),
            "suggested_execution_cap_seconds": 300 * math.ceil((2 * (max(point["total"], conservative["total"]) + extra + hash_reserve) + 120) / 300),
            "wall_seconds": time.monotonic() - begin, "audit_status": "not_started",
            "limits": "Synthetic untrained predictions and zero-action native roots; interpolation and twice-cost margin are descriptive estimates, not calibrated guarantees. Prefix/observer/fixed reserves may overcount work. No scientific readiness conclusion."})
        check(deadline)
    except BaseException as error:
        original_error = error
        try:
            if records and not (out / "source-episodes.npz").exists():
                control._save_records(out / "partial-completed-source-episodes", records)
            if active_folder is not None:
                if active_root is not None:
                    artifacts.save_npz(active_folder / "partial-root.npz", **active_root)
                if active_diagnostic is not None:
                    arrays = dict(active_diagnostic["arrays"])
                    if active_selected is not None:
                        arrays.update(active_selected.get("arrays", {}))
                        active_diagnostic["metadata"]["partial_selected_advance"] = active_selected.get("metadata", {})
                        if "carried_state" in active_selected:
                            artifacts.save_npz(active_folder / "partial-carried.npz", **active_selected["carried_state"])
                    artifacts.save_npz(active_folder / "partial-active-scoring.npz", **arrays)
                    write(active_folder / "partial-active-scoring.json", active_diagnostic["metadata"])
            if active_physics is not None:
                snapshot = active_physics.snapshot()
                if snapshot["last_operation"] is not None:
                    runner.save_physics(out / "partial-physics", snapshot.pop("last_operation"))
                write(out / "partial-physics-final.json", snapshot)
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Capacity partial preservation failed: {preservation_error!r}")
        failure(out, error, begin, progress)
        raise
    finally:
        close_template(template, original_error, out, begin, progress)


def audit_profile(attempt, expected_completion):
    import audit_reacher_geometry_memory_study as audit
    import numpy as np

    from openjev.research import reacher_geometry_memory_protocol as protocol
    from openjev.research.robotics_reacher import make_env

    begin, deadline = time.monotonic(), time.monotonic() + CAP
    attempt = Path(attempt).resolve()
    execution, out = attempt / "execution", attempt / "audit"
    out.mkdir(parents=True, exist_ok=False)
    progress, template, tokens, original_error = {"phase": "authenticate"}, None, [], None
    try:
        write(out / "started.json", {"scope": SCOPE, "cap_seconds": CAP, "expected_completed_sha256": expected_completion,
                                    "new_model_calls": 0, "started_unix_time": time.time()})
        require(sha(execution / "completed.json") == expected_completion, "Explicit completed profile SHA")
        completed = read(execution / "completed.json")
        require(completed["status"] == "completed" and completed["scope"] == SCOPE and completed["cap_seconds"] == CAP
                and math.isfinite(completed["wall_seconds"]) and 0 < completed["wall_seconds"] <= CAP
                and not (execution / "failed.json").exists(), "Completed within-cap engineering profile required")
        _, sources = bind_sources()
        require(completed["source_sha256"] == sources, "Capacity sources unchanged")
        hashing_start = time.perf_counter()
        for name, item in completed["files"].items():
            check(deadline)
            require(sha(execution / name) == item["sha256"] and (execution / name).stat().st_size == item["bytes"], "Capacity member hash/size")
        require({p.relative_to(execution).as_posix() for p in execution.rglob('*') if p.is_file()}
                == set(completed["files"]) | {"completed.json"}, "Exact complete capacity members")
        hashing_seconds = time.perf_counter() - hashing_start
        plan, fits = read(execution / "capacity-plan.json"), read(execution / "synthetic-models.json")
        require(sha(execution / "capacity-plan.json") == completed["plan_sha256"] and plan["rng_namespace"] == NAMESPACE,
                "Capacity plan namespace/binding")
        for context in (audit._DEADLINE, audit.inherited._DEADLINE, audit.previous._DEADLINE):
            tokens.append((context, context.set(deadline)))
        protocol.validate_settings(plan, engineering=True)
        require(plan["control_episodes"] == 64 and plan["hidden_size"] == 64 and plan["mlp_width"] == 107
                and plan["threads"] == 2, "Full-width, full64 capacity settings")
        require(plan["random_stream_contract"] == protocol.stream_contract(plan, *audit.prior_streams()),
                "Independently reconstructed namespace/exclusion contract")
        require(plan["runtime"] == read(execution / "source-binding.json")["runtime"], "Capacity runtime receipt binding")
        audit.validate_runtime_sources(plan)
        require(set(fits) == set(ARMS), "All four synthetic class identities")
        for arm in ARMS:
            payload = audit.torch.load(execution / "models" / (arm + ".pt"), map_location="cpu", weights_only=True)
            initial, state, key = payload["before_bias_edit"], payload["state"], payload["edited_key"]
            require(payload["seed"] == 410 and payload["edited_shape"] == [4]
                    and key.startswith("observation_head.") and key.endswith(".bias")
                    and set(state) == set(initial) and key in state, "Explicit synthetic bias-edit contract")
            require(audit.torch.equal(state[key], audit.torch.tensor([1., 1., 0., 0.]))
                    and all(audit.torch.equal(state[name], initial[name]) for name in state if name != key),
                    "Only final observation bias changed before profiling")
            require(audit.tensor_hash(state) == fits[arm]["student_tensor_sha256"]
                    and sum(value.numel() for value in state.values()) == fits[arm]["parameters"],
                    "Saved synthetic tensor/parameter identity")
        rng = audit.torch.load(execution / "constructor-outer-rng.pt", map_location="cpu", weights_only=True)
        require(audit.torch.equal(rng["outer_before"], rng["outer_after"]), "Ambient constructor RNG restoration witness")
        inputs = {step: audit.search_audit.audit_innovations(plan, execution / "innovations" / f"{step:03d}",
                    f"planner/control/{step}", 64) for step in STEPS}
        authentication_seconds = time.monotonic() - begin
        expected_rows = {(arm, panel, step, f"learned/{panel}/{arm}/{step:03d}")
                         for arm in ARMS for panel in PANELS for step in STEPS}
        expected_rows |= {("physics", "shared_native_roots", step, f"physics/{step:03d}") for step in STEPS}
        require(len(completed["rows"]) == 65 and {(r["kind"], r["panel"], r["step"], r["path"])
                for r in completed["rows"]} == expected_rows, "Exactly all65 declared probe rows")
        rows, physics_transitions, selected_transitions = [], 0, 0
        template = make_env()
        for row in completed["rows"]:
            progress = {"phase": "saved_probe_audit", "path": row["path"]}
            check(deadline)
            tick = time.perf_counter()
            folder, step = execution / row["path"], row["step"]
            trace = audit.search_audit.audit_trace({**plan, "planners": ["cem256"]}, inputs[step], folder / "trace", step=step)
            if row["kind"] == "physics":
                arrays = audit.base.load_npz(folder / "journal.npz")
                root = np.concatenate((arrays["root__qpos"], arrays["root__qvel"]), 1)
                checked = audit.audit_physics_trace(plan, template.unwrapped.model, folder / "journal", trace,
                    root, arrays["root__public_target"], step=step)
                physics_transitions += checked["candidate_transitions"]
                selected_transitions += checked["selected_transitions"]
            else:
                states = audit.base.load_npz(folder / "states.npz")
                root = {key.removeprefix("root__"): value for key, value in states.items() if key.startswith("root__")}
                carried = {key.removeprefix("carried__"): value for key, value in states.items() if key.startswith("carried__")}
                audit.audit_scoring_trace(plan, folder / "scoring", "geometry", root, fits[row["kind"]], trace, step,
                    carried=carried, executed_angles=states["angles"], executed_rewards=states["rewards"])
            rows.append({**row, "wall_seconds": time.perf_counter() - tick})
        require(physics_transitions == 704512 and selected_transitions == 320, "All nominal candidate and selected replay")
        tick = time.perf_counter()
        native = audit.audit_cohort(plan, audit.base.load_records(execution / "source-episodes", 64), "ordinary")
        native_seconds = time.perf_counter() - tick
        require(native["transitions"] == 3200 and native["max_abs_error"] == 0., "Complete engineering native source replay")
        point, conservative = projection(rows, "wall_seconds"), projection(rows, "wall_seconds", maximum=True)
        projected_bytes = completed["projected_probe_storage_bytes"]["total"]
        hash_reserve = hashing_seconds * max(projected_bytes / max(1, sum(item["bytes"] for item in completed["files"].values())),
                                             completed["expected_scientific_file_count"] / len(completed["files"]))
        reserve = 51 * native_seconds + completed["overhead"]["fixed_audit_reserve_seconds"] + hash_reserve
        require(sha(execution / "completed.json") == expected_completion
                and all(sha(ROOT / name) == value for name, value in sources.items()), "Final source/completion stability")
        audit.validate_runtime_sources(plan)
        write(out / "completed.json", {"status": "completed", "scope": SCOPE, "cap_seconds": CAP,
            "execution_completed_sha256": expected_completion, "rows": rows,
            "candidate_nominal_transitions_replayed": physics_transitions, "selected_nominal_transitions_replayed": selected_transitions,
            "native_source_transitions_replayed": native["transitions"], "native_source_audit_seconds": native_seconds,
            "hashing_seconds": hashing_seconds, "scaled_hashing_reserve_seconds": hash_reserve,
            "authentication_seconds_including_hashing": authentication_seconds,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "new_model_calls": 0, "new_fits": 0, "outcome_summaries_computed": False,
            "point": point, "conservative": conservative, "additional_audit_reserve_seconds": reserve,
            "suggested_audit_cap_seconds": 300 * math.ceil((2 * (max(point["total"], conservative["total"]) + reserve) + 120) / 300),
            "wall_seconds": time.monotonic() - begin,
            "limits": "Capacity kernels plus conservative complete engineering audit reserve. This is not the full scientific51-row audit; no neural replay or efficacy check. Hashing/storage/shared-host variance remains a prospective risk."})
        check(deadline)
    except BaseException as error:
        original_error = error
        failure(out, error, begin, progress)
        raise
    finally:
        try:
            close_template(template, original_error, out, begin, progress)
        finally:
            for context, token in reversed(tokens):
                context.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "audit"))
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--execute-engineering", action="store_true")
    parser.add_argument("--completed-sha256")
    args = parser.parse_args()
    require(args.execute_engineering, "Explicit --execute-engineering required; default performs no probes")
    require(args.phase == "run" or args.completed_sha256 is not None, "Audit needs externally supplied completion SHA")
    if args.phase == "run":
        run_profile(args.attempt)
    else:
        audit_profile(args.attempt, args.completed_sha256)


if __name__ == "__main__":
    main()
