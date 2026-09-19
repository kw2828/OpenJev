"""Prospective zero-fit geometry-scored comparison of twelve memory students.

No import-time draws or model construction. The enclosing protocol authenticates
completed cache and geometry evidence; the independent auditor is a mandatory
source dependency. No new training, exposed-root diagnostic or retry exists.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path, PurePosixPath

import numpy as np
import reacher_cache_study as cache
import reacher_geometry_study as geometry
import torch

from openjev.research import reacher_geometry_memory_control as control
from openjev.research import reacher_geometry_memory_protocol as protocol
from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.reacher_kinematic_control import KinematicObserver
from openjev.research.reacher_physics_control import ParticleFilter
from openjev.research.robotics_reacher import ReacherEpisode, make_env

ROOT = Path(__file__).resolve().parents[1]
STUDY = protocol.STUDY
base, search_study, training = cache.base, cache.search_study, cache.training
sha, read, write, require = cache.sha, cache.read, cache.write, cache.require
checked, check_cap, file_members = cache.checked, cache.check_cap, cache.file_members
SOURCES = geometry.SOURCES + protocol.NEW_SOURCES
CACHE_EXECUTION = "runs/reacher-cache-ablation-v1/attempt"
GEOMETRY_EXECUTION = "runs/reacher-geometry-score-v1/attempt"


def _relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == name, "Safe relative artifact path")
    return path


def authenticate_evidence(plan_path, plan_digest, audit_path, audit_digest, execution_path,
                          *, kind, engineering=False):
    """Saved-byte/source authentication only, never model or optimizer work."""
    require(kind in ("cache", "geometry"), "Exact parent role")
    parent = read(checked(ROOT / plan_path, plan_digest))
    audit = read(checked(ROOT / audit_path, audit_digest))
    execution = ROOT / execution_path
    done = read(checked(execution / "completed.json", audit["execution_completed_sha256"]))
    expected_sources = cache.SOURCES if kind == "cache" else geometry.SOURCES
    require(audit["status"] == done["status"] == "completed" and audit["saved_output_only"] is True
            and audit["engineering"] is engineering and parent["engineering"] is engineering
            and audit["plan_sha256"] == done["plan_sha256"] == plan_digest
            and audit["source_sha256"] == parent["sources"]
            and audit["runtime"] == parent["runtime"]
            and audit["execution_members"] == done["files"], "Completed prior plan/audit/execution identity")
    require(set(parent["sources"]) == set(expected_sources), "Exact inherited source membership")
    require(np.isfinite(done["wall_seconds"]) and 0 <= done["wall_seconds"] <= parent["cap_seconds"], "Completed parent within cap")
    for name, digest in parent["sources"].items():
        _relative(name)
        checked(ROOT / name, digest)
    for name, digest in audit["files"].items():
        _relative(name)
        checked((ROOT / audit_path).parent / name, digest)
    names = protocol.inherited_members(protocol.settings()) if kind == "cache" else []
    require(set(names) <= set(done["files"]), "Every inherited file was audited")
    members = {name: done["files"][name] for name in names}
    for name, digest in members.items():
        _relative(name)
        checked(execution / name, digest)
    summary = read((ROOT / audit_path).parent / "summary.json")
    gate = summary["continuation_gate"]["passed"]
    if not engineering:
        require(gate is (kind == "geometry"), "Pinned prior gate interpretation")
    source = {"plan_path": str(plan_path), "plan_sha256": plan_digest, "audit_path": str(audit_path),
        "audit_receipt_sha256": audit_digest, "execution_path": str(execution_path),
        "completed_sha256": audit["execution_completed_sha256"], "summary_sha256": audit["files"]["summary.json"],
        "members": members, "prior_costs": audit["costs"], "engineering": engineering,
        "previous_scientific_gate_passed": gate if not engineering else None,
        "training_weights_reused": kind == "cache", "new_fits": 0}
    return parent, source


def authenticated_sources():
    # This validates older training/data lineage through the existing immutable
    # source chain. No learned forward, trainer restore or native rollout occurs.
    cache.authenticated_sources()
    bindings = protocol.lineage_contract()
    parent, source = authenticate_evidence(bindings["cache"]["plan_path"], bindings["cache"]["plan_sha256"],
        bindings["cache"]["audit_path"], bindings["cache"]["audit_receipt_sha256"], CACHE_EXECUTION, kind="cache")
    previous, geometry_source = authenticate_evidence(bindings["geometry"]["plan_path"], bindings["geometry"]["plan_sha256"],
        bindings["geometry"]["audit_path"], bindings["geometry"]["audit_receipt_sha256"], GEOMETRY_EXECUTION, kind="geometry")
    require(previous["parent_source"]["plan_sha256"] == source["plan_sha256"]
            and previous["parent_source"]["audit_receipt_sha256"] == source["audit_receipt_sha256"],
            "Geometry context uses the same completed cache parent")
    return parent, source, previous, geometry_source


def prior_streams():
    # Exact frozen numerical registries, validated independently by the new
    # pure protocol; it separately excludes geometry and all engineering roles.
    return geometry.prior_streams()


def stream_contract(plan):
    return protocol.stream_contract(plan, *prior_streams())


def source_hashes(inherited_sources):
    names = protocol.source_membership(inherited_sources)
    require(set(names) == set(SOURCES), "Complete runner dependency membership")
    result = {name: sha(ROOT / name) for name in names}
    protocol.validate_source_hashes(result, inherited_sources)
    return result


def prepare(out, *, cap_seconds, audit_cap_seconds):
    """Write a prospective plan only when all actual source dependencies exist.

    This is not a launch/readiness receipt. Independent completed engineering
    validation and the external freeze remain required before a scored launch.
    """
    require(all(type(value) is int and value > 0 for value in (cap_seconds, audit_cap_seconds)), "Explicit positive caps")
    parent, cache_source, previous, geometry_source = authenticated_sources()
    sources = source_hashes(previous["sources"])  # Missing auditor/test blocks here.
    plan = protocol.settings()
    plan.update(cache_source=cache_source, geometry_source=geometry_source, cap_seconds=cap_seconds,
        audit_cap_seconds=audit_cap_seconds, engineering=False, sources=sources, runtime=base.runtime())
    require(all(plan[key] == parent[key] for key in ("hidden_size", "mlp_width", "dt", "noise_std", "residual_reward")),
            "Unchanged inherited numerical configuration")
    plan["random_stream_contract"] = stream_contract(plan)
    protocol.validate_settings(plan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json"), "ready_to_launch": False}


def validate(path, expected):
    plan = read(checked(path, expected))
    require(plan["engineering"] is False, "Scored protocol required")
    protocol.validate_settings(plan)
    _, source, previous, geometry_source = authenticated_sources()
    require(plan["sources"] == source_hashes(previous["sources"]), "Frozen sources changed")
    require(plan["runtime"] == base.runtime(), "Frozen runtime changed")
    require(plan["cache_source"] == source and plan["geometry_source"] == geometry_source
            and plan["random_stream_contract"] == stream_contract(plan), "Frozen lineage and random streams")
    require(all(type(plan[key]) is int and plan[key] > 0 for key in ("cap_seconds", "audit_cap_seconds")), "Frozen caps")
    return plan


def copy_inherited(plan, out, deadline):
    for kind in ("cache", "geometry"):
        source = plan[kind + "_source"]
        if kind == "cache":
            require(set(source["members"]) == set(protocol.inherited_members(plan)), "All twelve fits and shared inputs")
            for name, digest in source["members"].items():
                _relative(name)
                search_study.copy_verified(ROOT / source["execution_path"] / name,
                    out / "inherited" / name, digest, deadline)
        else:
            require(source["members"] == {}, "No exposed prior geometry trajectories")
        for suffix, path, digest in (
            ("plan", ROOT / source["plan_path"], source["plan_sha256"]),
            ("audit", ROOT / source["audit_path"], source["audit_receipt_sha256"]),
            ("completed", ROOT / source["execution_path"] / "completed.json", source["completed_sha256"]),
            ("summary", (ROOT / source["audit_path"]).parent / "summary.json", source["summary_sha256"]),
        ):
            search_study.copy_verified(path, out / "inherited" / f"{kind}-{suffix}.json", digest, deadline)
    write(out / "inheritance.json", {kind: plan[kind + "_source"] for kind in ("cache", "geometry")})


def restore_students(plan, out, deadline):
    """Construct only deployment students; no trainer or optimizer restoration."""
    started, models, receipts = time.perf_counter(), {}, {}
    for name, digest in plan["cache_source"]["members"].items():
        check_cap(deadline)
        checked(out / "inherited" / name, digest)
    parent = read(out / "inherited" / "cache-plan.json")
    cfg = cache.settings_from_plan(parent)
    data = cache.public_training_data(base.load_records(out / "inherited" / "train"))
    data_hash = training.canonical_tensor_hash(data)
    require(all(plan[key] == getattr(cfg, key) for key in ("hidden_size", "mlp_width", "dt", "noise_std", "residual_reward")),
            "Exact deployment dimensions and scalar behavior")
    require(sha(out / "inherited" / "cache-plan.json") == plan["cache_source"]["plan_sha256"], "Copied parent plan")
    for row in protocol.fit_manifest(plan):
        check_cap(deadline)
        folder = out / "inherited" / "fits" / row["name"]
        done = read(folder / "completed.json")
        require(set(done["files"]) == set(protocol.FIT_MEMBERS) - {"completed.json"}, "Exact inherited fit members")
        for name, digest in done["files"].items():
            checked(folder / name, digest)
        checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
        integrity = training.canonical_state_hash({k: v for k, v in checkpoint.items() if k != "integrity_sha256"})
        config = training.model_configuration(row["arm"], cfg)
        require(integrity == checkpoint["integrity_sha256"] == done["checkpoint_integrity_sha256"]
                and checkpoint["version"] == training.VERSION and checkpoint["failed"] is False
                and checkpoint["kind"] == row["arm"]
                and checkpoint["settings"] == done["settings"] == cfg.configuration()
                and checkpoint["model_configuration"] == done["model_configuration"] == config
                and checkpoint["source_sha256"] == parent["sources"]
                and checkpoint["runtime"] == parent["runtime"] == plan["runtime"]
                and checkpoint["data_sha256"] == done["data_sha256"] == data_hash,
                "Authenticated unchanged completed checkpoint configuration")
        updates = cfg.epochs * cfg.batches_per_epoch
        require(done["name"] == row["name"] and done["arm"] == row["arm"] and done["pair"] == row["pair"]
                and checkpoint["successful_updates"] == checkpoint["optimizer_steps"]
                == done["updates"] == done["optimizer_steps"] == updates
                and checkpoint["cursor"] == {"epoch": cfg.epochs, "batch": 0}, "Complete inherited fit schedule")
        for field, directory in (("initialization", "initializations"), ("orders", "orders")):
            original = torch.load(out / "inherited" / directory / f"{row['pair']}.pt",
                                  weights_only=True, map_location="cpu")
            require(training.canonical_state_hash(original) == training.canonical_state_hash(checkpoint[field])
                    and original["integrity_sha256"] == done[field + "_sha256"], "Inherited fit input identity")
        initial = torch.load(folder / "initial-weights.pt", weights_only=True, map_location="cpu")
        family = "mlp" if row["arm"] == "cached_mlp" else "gru"
        require(training.canonical_tensor_hash(initial) == checkpoint["initialization"]["tensor_hashes"][family]
                == training.canonical_tensor_hash(checkpoint["initialization"]["states"][family]),
                "Actual paired family initialization")
        orders = checkpoint["orders"]["orders"]
        require(orders.dtype == torch.int64 and tuple(orders.shape) == (cfg.epochs, cfg.train_episodes)
                and torch.equal(orders.sort(dim=1).values, torch.arange(cfg.train_episodes).repeat(cfg.epochs, 1)),
                "Complete paired permutations without regenerating training RNG")
        weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
        digest = training.canonical_tensor_hash(weights)
        require(digest == done["student_tensor_sha256"] == training.canonical_tensor_hash(checkpoint["student_state"]),
                "Exact saved student tensors")
        # Isolated constructor seed0 is an excluded engineering role. Every
        # parameter is overwritten by the authenticated checkpoint before use.
        model = training._construct(row["arm"], cfg, 0)
        training._load_weights(model, checkpoint["student_state"])
        model.eval().requires_grad_(False)
        require(training.canonical_tensor_hash(model.state_dict()) == digest, "Deployment tensor restoration")
        control.model_identity(model, plan)
        snapshot = out / "model-states" / f"{row['name']}-before.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cache.save_torch(snapshot, model.state_dict())
        models[row["name"]] = model
        receipts[row["name"]] = {"kind": row["arm"], "model_class": type(model).__name__,
            "checkpoint_sha256": sha(folder / "checkpoint.pt"), "weights_sha256": sha(folder / "weights.pt"),
            "student_tensor_sha256": digest, "successful_updates": updates, "configuration": config,
            "snapshot_path": snapshot.relative_to(out).as_posix(), "snapshot_sha256": sha(snapshot)}
    write(out / "all-models-restored.json", {"models": receipts, "optimizer_constructed": False,
        "new_updates": 0, "constructor_seed": 0, "constructor_rng_isolated_and_weights_overwritten": True,
        "wall_seconds": time.perf_counter() - started, "unix_time": time.time()})
    check_cap(deadline)
    return models


def control_cases(plan, panel):
    return [control.ControlCase(protocol.seed(plan, f"control/reset/{index}"),
        protocol.seed(plan, f"control/actuator_noise/{index}"), protocol.schedule(plan, index, panel))
        for index in range(plan["control_episodes"])]


def save_physics(stem, journal):
    """Lossless detached physics journal; arrays are explicit, no neural calls.

    NPZ roots use root__key, callback banks bank0__key through bank3__key,
    selected one-step prediction selected__key. JSON preserves all scalar
    metadata, search identities/selection and any failed native scratch state.
    """
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    arrays = {"root__" + key: value for key, value in journal["roots"].items()}
    metadata = {key: value for key, value in journal.items() if key not in ("roots", "banks", "selected")}
    metadata["banks"] = []
    for index, record in enumerate(journal["banks"]):
        arrays.update({f"bank{index}__{key}": value for key, value in record["arrays"].items()})
        metadata["banks"].append({key: value for key, value in record.items() if key != "arrays"})
    metadata["selected"] = None
    if journal["selected"] is not None:
        arrays.update({"selected__" + key: value for key, value in journal["selected"]["arrays"].items()})
        metadata["selected"] = {key: value for key, value in journal["selected"].items() if key != "arrays"}
    artifacts.save_npz(Path(stem).with_suffix(".npz"), **arrays)
    write(Path(stem).with_suffix(".json"), cache.json_arrays(metadata))


def reference_control(plan, panel, arm, inputs_by_step, out, deadline=float("inf"), progress=None, *, cases):
    """Matched geometry CEM for explicit public or privileged reference states.

    Only known_state reads native audit qpos/qvel. Particle and kinematic
    observers receive only real public packets and previously issued commands.
    They never receive candidate terminal states or realized disturbances.
    Full row timing charges observer work, snapshots, copies and trace writes.
    """
    require(arm in protocol.REFERENCES and panel in protocol.PANELS, "Known reference row")
    require(len(cases) == plan["control_episodes"] and all(type(case) is control.ControlCase for case in cases),
            "Complete explicit cases")
    require(len(inputs_by_step) == plan["steps"] == 50, "Complete fixed50 inputs")
    out, begin = Path(out), time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    envs, initialized, template, physics, observers = [], [], None, None, []
    original_error, active_journal = None, None
    decisions, native_times, scores, roots, public, issued = [], [], [], [], [], []
    estimates_before, previous_actions = [], np.zeros((len(cases), 2), np.float32)
    observer_updates = 0
    try:
        packets = []
        for case in cases:
            check_cap(deadline)
            env = ReacherEpisode(noise_std=plan["noise_std"])
            envs.append(env)
            require(env.dt == plan["dt"], "Native/reference timestep alignment")
            packets.append(env.reset(case.reset_seed, case.sensor_schedule, noise_seed=case.noise_seed))
            initialized.append(env)
        packets = np.stack(packets)
        planned = arm in protocol.PHYSICS_REFERENCES
        if planned:
            # No reset, no RNG draw: only static constructor-loaded model data.
            template = make_env()
            native_model = template.unwrapped.model
            physics = PhysicsGeometryCEM(native_model, frame_skip=2, noise_std=plan["noise_std"], steps=50,
                planning_horizon=plan["planning_horizon"], action_block=plan["action_block"])
            if arm == "particle":
                observers = [ParticleFilter(native_model, packet,
                    seed=protocol.seed(plan, f"planner/particle_filter/{index}"), particles=plan["particles"],
                    noise_std=plan["noise_std"], measurement_std=plan["filter_bandwidth"], frame_skip=2)
                    for index, packet in enumerate(packets)]
            elif arm == "public_kinematic":
                observers = [KinematicObserver(native_model, packet, frame_skip=2) for packet in packets]
        uniform = np.random.default_rng(protocol.seed(plan, "floor/uniform/0")) if arm == "uniform" else None
        setup = time.perf_counter() - begin
        for step in range(50):
            active_journal = None
            if progress is not None:
                progress["step"] = step
            check_cap(deadline)
            tick = time.perf_counter()
            if planned:
                estimates = []
                for index, env in enumerate(envs):
                    check_cap(deadline)
                    if arm == "known_state":
                        actual = env.audit_record()
                        position, velocity = actual["qpos"], actual["qvel"]
                    else:
                        if step:
                            observers[index].update(previous_actions[index], packets[index])
                            observer_updates += 1
                        position, velocity = observers[index].estimate()
                    estimates.append((position, velocity))
                qpos = np.stack([row[0] for row in estimates]).astype(np.float64)
                qvel = np.stack([row[1] for row in estimates]).astype(np.float64)
                root_state = np.concatenate((qpos, qvel), axis=1)
                roots.append(root_state.copy())
                public.append(packets.copy())
                issued.append(previous_actions.copy())
                value = inputs_by_step[step]
                inputs = value if isinstance(value, artifacts.SearchInputs) else artifacts.load_inputs(value)
                result, active_journal = physics.plan(qpos, qvel, packets[:, 4:6].copy(), inputs,
                    step=step, deadline=deadline)
                actions = result.selected_actions.copy()
                raw = np.concatenate([bank["arrays"]["geometry_reward"] for bank in active_journal["banks"]], axis=1)
                artifacts.save_trace(out / "decisions" / f"{step:03d}", result, raw, [64] * 4,
                    active_journal["wall_seconds"])
                save_physics(out / "physics" / f"{step:03d}", active_journal)
                scores.append(result.scores.copy())
                if observers:
                    after = np.stack([np.concatenate(observer.estimate()) for observer in observers])
                    require(np.array_equal(after, root_state), "Search must not overwrite observer roots")
                estimates_before.append(root_state.copy())
            else:
                actions = np.zeros((len(envs), 2), np.float32) if arm == "zero" else uniform.uniform(-1, 1, (len(envs), 2)).astype(np.float32)
                scores.append(np.zeros((len(envs), 256), np.float64))
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
            previous_actions = actions
        require(all(env.finished and env.step_index == 50 for env in envs), "Complete reference native50")
        control._save_records(out / "episodes", [env.episode_record() for env in envs])
        arrays = {"candidate_scores": np.stack(scores, 1), "planner_used": np.array(planned)}
        if planned:
            arrays.update(root_estimates=np.stack(roots, 1), public_packets=np.stack(public, 1),
                          previous_commands=np.stack(issued, 1))
            snapshot = physics.snapshot()
            snapshot.pop("last_operation")
            write(out / "physics-final.json", cache.json_arrays(snapshot))
        artifacts.save_npz(out / "planning.npz", **arrays)
        if arm == "public_kinematic":
            write(out / "observer-final.json", cache.json_arrays([observer.snapshot() for observer in observers]))
        elif arm == "particle":
            write(out / "observer-final.json", {"reference": arm, "public_inputs_only": True,
                "updates": observer_updates, "particles": plan["particles"],
                "estimates": cache.json_arrays(estimates_before[-1]),
                "counts": {"setup_forward_calls": len(cases) * plan["particles"],
                    "nominal_transition_calls": observer_updates * plan["particles"],
                    "native_substeps": observer_updates * plan["particles"] * 2,
                    "measurement_reanchor_forward_calls": sum(int(case.sensor_schedule[1:50].sum()) for case in cases) * plan["particles"]},
                "scope": "Detached estimate after final decision assimilation, before terminal packet; not a resumable particle/RNG state."})
        timing = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
            "row_wall_seconds": time.perf_counter() - begin,
            "candidate_evaluations_per_decision": 256 if planned else 0,
            "selected_nominal_advances": len(envs) * 50 if planned else 0,
            "observer_updates": observer_updates, "privileged_state_reads": len(envs) * 50 if arm == "known_state" else 0,
            "information": "Current native qpos/qvel, public goal; no future noise" if arm == "known_state" else "Public packets and issued commands only, supplied nominal model for planned references",
            "score_mode": "geometry" if planned else None, "planner": "cem256" if planned else None,
            "floors_zero_scores_are_predictions": False}
        write(out / "timings.json", timing)
        check_cap(deadline)
        return timing
    except BaseException as error:
        original_error = error
        try:
            if initialized and not (out / "episodes.npz").exists():
                control._save_partial(out / "partial-episodes", initialized)
            if physics is not None:
                snapshot = physics.snapshot()
                if snapshot["last_operation"] is not None:
                    save_physics(out / "partial-physics", snapshot.pop("last_operation"))
                write(out / "partial-physics-final.json", cache.json_arrays(snapshot))
            write(out / "failed.json", {"status": "failed", "exception_type": type(error).__name__,
                "message": str(error), "reference": arm, "completed_steps_by_case": [env.step_index for env in initialized],
                "decision_seconds": decisions, "native_step_seconds": native_times,
                "observer_updates": observer_updates, "wall_seconds": time.perf_counter() - begin})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Artifact preservation failed: {preservation_error!r}")
        raise
    finally:
        errors = []
        for env in [*envs, *([] if template is None else [template])]:
            try:
                env.close()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            if original_error is not None:
                original_error.add_note(f"Close failures: {[repr(error) for error in errors]}")
            else:
                raise errors[0]


def run(path, expected, out):
    begin, out = time.monotonic(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = validate(path, expected)
        check_cap(begin + plan["cap_seconds"])
        return run_validated(plan, expected, out, begin=begin, output_created=True,
                             final_validate=lambda: validate(path, expected))
    except BaseException as error:
        if not (out / "failed.json").exists():
            try:
                write(out / "failed.json", {"status": "failed", "plan_sha256": expected, "error": repr(error),
                    "progress": {"phase": "validate-or-setup"}, "wall_seconds": time.monotonic() - begin})
            except BaseException as preservation_error:  # noqa: BLE001
                error.add_note(f"Failure receipt preservation failed: {preservation_error!r}")
        raise


def run_validated(plan, expected, out, *, begin=None, final_validate=None, output_created=False):
    begin = time.monotonic() if begin is None else begin
    protocol.validate_settings(plan, engineering=plan.get("engineering") is True)
    require(callable(final_validate), "Every execution requires source/lineage validation")
    deadline, out = begin + plan["cap_seconds"], Path(out)
    if output_created:
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), "Exclusive empty attempt")
    else:
        out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "initial-validation"}
    try:
        final_validate()
        check_cap(deadline)
        torch.set_num_threads(plan["threads"])
        torch.use_deterministic_algorithms(True)
        write(out / "started.json", {"plan_sha256": expected, "unix_time": time.time()})
        write(out / "random-streams.json", plan["random_stream_contract"])
        progress = {"phase": "copy-inherited"}
        tick = time.perf_counter()
        copy_inherited(plan, out, deadline)
        copy_seconds = time.perf_counter() - tick
        progress = {"phase": "restore-all-models"}
        models = restore_students(plan, out, deadline)
        require(set(models) == {row["name"] for row in protocol.fit_manifest(plan)}, "All twelve models restored before draws")
        evaluation_started = time.monotonic() - begin
        write(out / "evaluation-started.json", {"plan_sha256": expected,
            "all_models_restored_sha256": sha(out / "all-models-restored.json"),
            "unix_time": time.time(), "elapsed_seconds": evaluation_started})
        inputs, tick = [], time.perf_counter()
        for step in range(50):
            check_cap(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            artifacts.save_inputs(stem, protocol.draw_control_inputs(plan, step), f"planner/control/{step}")
            inputs.append(stem)
        innovation_seconds, control_times = time.perf_counter() - tick, []
        for row in protocol.execution_order(plan):
            progress = {"phase": "control", "panel": row["panel"], "model": row["label"]}
            cases = control_cases(plan, row["panel"])
            if "fit" in row:
                timing = control.learned_control(plan, models[row["fit"]], row["panel"], inputs,
                    out / row["path"], deadline, progress, cases=cases)
            else:
                timing = reference_control(plan, row["panel"], row["reference"], inputs,
                    out / row["path"], deadline, progress, cases=cases)
            control_times.append(timing)
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        write(out / "control-completed.json", {"rows": len(control_times), "unix_time": time.time(),
            "plan_sha256": expected, "evaluation_started_sha256": sha(out / "evaluation-started.json")})
        progress = {"phase": "final-validation"}
        final_validate()
        restored = read(out / "all-models-restored.json")
        final_hashes = {name: training.canonical_tensor_hash(model.state_dict()) for name, model in models.items()}
        require(all(digest == restored["models"][name]["student_tensor_sha256"] for name, digest in final_hashes.items()),
                "Model weights unchanged across every control row")
        snapshots = {}
        for name, model in models.items():
            path = out / "model-states" / f"{name}-after.pt"
            cache.save_torch(path, model.state_dict())
            snapshots[path.relative_to(out).as_posix()] = sha(path)
        write(out / "final-models.json", {"student_tensor_sha256": final_hashes, "files": snapshots, "new_updates": 0})
        costs = {"inheritance_copy_wall_seconds": copy_seconds, "restore_wall_seconds": restored["wall_seconds"],
            "new_fits": 0, "new_optimizer_steps": 0, "new_prediction_episodes": 0, "diagnostic_roots": 0,
            "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_times),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_times),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_times),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_times),
            "control_native_transitions": len(control_times) * plan["control_episodes"] * 50,
            "astra_calls": 0, "cache_parent_costs": plan["cache_source"]["prior_costs"],
            "geometry_context_costs": plan["geometry_source"]["prior_costs"], "compute_matched": False,
            "accounting": "Full execution includes source/checkpoint validation and copies, twelve restorations, all heads, "
                "geometry/CEM, public observers or privileged reference roots, native steps, diagnostics storage and hashes. "
                "Row/controller costs include only their internal validation/copies. Nested times overlap. "
                "Historical geometry cumulative costs already include its cache ancestry; do not add the two histories."}
        write(out / "costs.json", costs)
        members = file_members(out, deadline)
        check_cap(deadline)
        total = time.monotonic() - begin
        prior_total = plan["geometry_source"]["prior_costs"]["cumulative_attempt_wall_seconds"]
        write(out / "completed.json", {"status": "completed", "study": STUDY, "plan_sha256": expected,
            "new_fits": 0, "restored_models": len(models), "control_rows": len(control_times),
            "diagnostic_roots": 0, "astra_calls": 0, "wall_seconds": total,
            "evaluation_started_elapsed_seconds": evaluation_started,
            "cumulative_attempt_wall_seconds": prior_total + total, "files": members})
        check_cap(deadline)
        return {"status": "completed", "control_rows": len(control_times), "wall_seconds": total}
    except BaseException as error:
        if (out / "completed.json").exists():
            (out / "completed.json").rename(out / "over-cap-completion.json")
        try:
            write(out / "failed.json", {"status": "failed", "plan_sha256": expected, "error": repr(error),
                "progress": progress, "wall_seconds": time.monotonic() - begin, "files": file_members(out)})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Failure receipt preservation failed: {preservation_error!r}")
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--cap-seconds", type=int, required=True)
    prep.add_argument("--audit-cap-seconds", type=int, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--expected-plan-sha256", required=True)
    execution.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out, cap_seconds=args.cap_seconds, audit_cap_seconds=args.audit_cap_seconds) \
        if args.command == "prepare" else run(args.plan, args.expected_plan_sha256, args.out)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
