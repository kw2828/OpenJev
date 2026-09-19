"""One exclusive two-observation study attempt with explicit authenticated inputs.

No work at import time. Original initialization/order/data bytes are inherited;
six deployment models are checked before three fresh history fits. All nine
final deployment classes are restored without optimizers before fresh draws.
The independent auditor, preparation and external readiness remain separate.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import reacher_cache_study as cache
import torch

from openjev.research import reacher_geometry_memory_control as control
from openjev.research import reacher_search_protocol as artifacts
from openjev.research import reacher_two_observation_control as history_control
from openjev.research import reacher_two_observation_episode as episode
from openjev.research import reacher_two_observation_fitting as fitting
from openjev.research import reacher_two_observation_protocol as protocol
from openjev.research import reacher_two_observation_results as results
from openjev.research import reacher_two_observation_streams as streams
from openjev.research import reacher_two_observation_training as history_training
from openjev.research.reacher_geometry_physics import PhysicsGeometryCEM
from openjev.research.reacher_kinematic_control import KinematicObserver
from openjev.research.reacher_physics_control import ParticleFilter
from openjev.research.robotics_reacher import ReacherEpisode, make_env

ROOT = Path(__file__).resolve().parents[1]
STUDY = protocol.STUDY
VERSION = "reacher-two-observation-runner-v1"
base, search_study, training = cache.base, cache.search_study, cache.training
sha, read, write, require = cache.sha, cache.read, cache.write, cache.require
checked, check_cap, file_members = cache.checked, cache.check_cap, cache.file_members


def _experiment():
    # Missing preparation/auditor dependencies fail before any attempt draws.
    from openjev.research import reacher_two_observation_experiment
    return reacher_two_observation_experiment


def validate(path, expected, *, engineering=False, freeze_ref=None):
    plan = read(checked(Path(path), expected))
    context = _experiment().validate_plan(plan, expected, root=ROOT, runtime=_experiment().runtime(),
        engineering=engineering, plan_bytes=Path(path).read_bytes(), freeze_ref=freeze_ref)
    return plan, context


def prepare(out, settings, *, lineage_refs, historical_registries, cap_seconds,
            audit_cap_seconds, engineering_evidence, engineering=False):
    """Write only an unrun plan. No readiness or launch receipt is invented."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = _experiment().prepare(ROOT, settings, lineage_refs=lineage_refs,
            historical_registries=historical_registries, runtime=_experiment().runtime(),
            cap_seconds=cap_seconds, audit_cap_seconds=audit_cap_seconds,
            engineering_evidence=engineering_evidence, engineering=engineering)
        write(out / "plan.json", plan)
        return {"path": str(out / "plan.json"), "sha256": sha(out / "plan.json"), "ready_to_launch": False}
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "phase": "prepare", "error": repr(error)})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Preparation failure preservation failed: {preservation_error!r}")
        raise


def copy_inherited(plan, context, out, deadline):
    settings = context.settings
    source = context.cache_source
    require(set(source["members"]) == set(protocol.inherited_members(settings)), "Exact44 original members")
    for name, digest in source["members"].items():
        search_study.copy_verified(ROOT / source["execution_path"] / name,
            out / "inherited" / name, digest, deadline)
    for label, source in (("cache", context.cache_source), ("prerequisite", context.prerequisite_source)):
        pairs = [("plan", ROOT / source["plan_path"], source["plan_sha256"]),
                 ("audit", ROOT / source["audit_path"], source["audit_receipt_sha256"]),
                 ("completed", ROOT / source["execution_path"] / "completed.json", source["completed_sha256"]),
                 ("summary", (ROOT / source["audit_path"]).parent / "summary.json", source["summary_sha256"])]
        if label == "prerequisite":
            pairs.append(("terminal", ROOT / source["terminal_path"], source["terminal_sha256"]))
        for suffix, path, digest in pairs:
            search_study.copy_verified(path, out / "inherited" / "lineage" / f"{label}-{suffix}.json", digest, deadline)
    write(out / "inheritance.json", {"cache": context.cache_source, "prerequisite": context.prerequisite_source})


def training_settings(settings):
    return history_training.TrainingSettings(**{field.name: settings[field.name]
        for field in fields(history_training.TrainingSettings)})


def load_training_inputs(plan, context, out, deadline):
    """Authenticate saved inputs only. Never create initials or new permutations."""
    begin = time.perf_counter()
    settings, parent = context.settings, context.cache_plan
    cfg = cache.settings_from_plan(parent)
    require(all(getattr(cfg, field.name) == settings[field.name] for field in fields(cfg)),
            "Unchanged original training recipe")
    for name, digest in context.cache_source["members"].items():
        check_cap(deadline)
        checked(out / "inherited" / name, digest)
    data = cache.public_training_data(base.load_records(out / "inherited" / "train"))
    digest = training._public_data(data, cfg)
    initials, orders, bindings = {}, {}, {}
    original_seeds = parent["random_stream_contract"]["torch_registry"]
    for pair in protocol.PAIRS:
        check_cap(deadline)
        initial_path = out / "inherited" / "initializations" / f"{pair}.pt"
        order_path = out / "inherited" / "orders" / f"{pair}.pt"
        initial = torch.load(initial_path, weights_only=True, map_location="cpu")
        order = torch.load(order_path, weights_only=True, map_location="cpu")
        for payload, path in ((initial, initial_path), (order, order_path)):
            training._unseal(payload)
            require(read(path.with_suffix(".json")) == {"pair": pair, "file_sha256": sha(path),
                "integrity_sha256": payload["integrity_sha256"]}, "Original paired payload sidecar")
        roles = {family: f"fit/{family}/{pair}" for family in ("gru", "mlp")}
        require(initial["version"] == training.VERSION and initial["settings"] == cfg.configuration()
            and initial["roles"] == roles
            and initial["seeds"] == {family: original_seeds[role] for family, role in roles.items()}
            and set(initial["states"]) == set(initial["tensor_hashes"]) == {"gru", "mlp"},
            "Original initialization semantic identity")
        for family in ("gru", "mlp"):
            require(training.canonical_tensor_hash(initial["states"][family]) == initial["tensor_hashes"][family],
                    "Original family tensor identity")
        order_role = f"fit/minibatch/{pair}"
        require(order["version"] == training.VERSION and order["role"] == order_role
            and order["seed"] == original_seeds[order_role], "Original minibatch role/seed")
        require(order["orders"].dtype == torch.int64 and order["orders"].device.type == "cpu"
            and tuple(order["orders"].shape) == (cfg.epochs, cfg.train_episodes)
            and torch.equal(order["orders"].sort(1).values, torch.arange(cfg.train_episodes).repeat(cfg.epochs, 1)),
            "Complete original paired epoch orders")
        initials[pair], orders[pair] = initial, order
        bindings[pair] = {"initial_file_sha256": sha(initial_path), "order_file_sha256": sha(order_path),
            "initial_tensor_sha256": initial["tensor_hashes"]["gru"],
            "initialization_integrity_sha256": initial["integrity_sha256"],
            "orders_integrity_sha256": order["integrity_sha256"],
            "orders_sha256": training.canonical_state_hash(order)}
    receipt = {"data_sha256": digest, "pairs": bindings, "settings": cfg.configuration(),
        "new_initializations": 0, "new_orders": 0, "wall_seconds": time.perf_counter() - begin,
        "scope": "Authenticated copied original tensors/orders; no initialization or permutation generation."}
    write(out / "training-inputs.json", receipt)
    check_cap(deadline)
    return data, initials, orders, receipt


def fit_provenance(plan, expected, row):
    source = plan["lineage"]["cache"]
    paths = [row["initialization_path"], row["order_path"], "train.npz", "train.json"]
    return {"study": STUDY, "plan_sha256": expected, "name": row["name"], "pair": row["pair"],
        "cache_plan_sha256": source["plan_sha256"], "cache_audit_receipt_sha256": source["audit_receipt_sha256"],
        "original_members": {name: source["members"][name] for name in paths},
        "initial_state": "original paired GRU tensors, not inherited fitted weights"}


def _inherited_student(settings, parent, out, row, data_hash, initials, orders):
    folder = out / "inherited" / "fits" / row["name"]
    done = read(folder / "completed.json")
    require(set(done["files"]) == set(protocol.FIT_MEMBERS) - {"completed.json"}, "Exact inherited fit payloads")
    for name, digest in done["files"].items():
        checked(folder / name, digest)
    checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
    training._unseal(checkpoint)
    cfg = cache.settings_from_plan(parent)
    config = training.model_configuration(row["arm"], cfg)
    updates = cfg.epochs * cfg.batches_per_epoch
    require(checkpoint["integrity_sha256"] == done["checkpoint_integrity_sha256"]
        and checkpoint["version"] == training.VERSION and checkpoint["failed"] is False
        and checkpoint["kind"] == row["arm"] and checkpoint["settings"] == done["settings"] == cfg.configuration()
        and checkpoint["model_configuration"] == done["model_configuration"] == config
        and checkpoint["source_sha256"] == parent["sources"] and checkpoint["runtime"] == parent["runtime"]
        and checkpoint["data_sha256"] == done["data_sha256"] == data_hash,
        "Authenticated inherited checkpoint configuration")
    require(done["name"] == row["name"] and done["arm"] == row["arm"] and done["pair"] == row["pair"]
        and checkpoint["successful_updates"] == checkpoint["optimizer_steps"] == done["updates"]
        == done["optimizer_steps"] == updates and checkpoint["cursor"] == {"epoch": cfg.epochs, "batch": 0},
        "Complete inherited training schedule")
    for field, original in (("initialization", initials[row["pair"]]), ("orders", orders[row["pair"]])):
        require(training.canonical_state_hash(checkpoint[field]) == training.canonical_state_hash(original)
            and done[field + "_sha256"] == original["integrity_sha256"], "Original paired fit inputs")
    initial = torch.load(folder / "initial-weights.pt", weights_only=True, map_location="cpu")
    require(training.canonical_tensor_hash(initial) == initials[row["pair"]]["tensor_hashes"]["gru"],
            "Original shared GRU initial weights")
    weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
    digest = training.canonical_tensor_hash(weights)
    require(digest == done["student_tensor_sha256"] == training.canonical_tensor_hash(checkpoint["student_state"]),
            "Inherited deployment weight equality")
    # Excluded isolated constructor0; every tensor is overwritten. No optimizer.
    model = training._construct(row["arm"], cfg, 0)
    training._load_weights(model, weights)
    model.eval().requires_grad_(False)
    identity = control.model_identity(model, settings)
    require(identity["kind"] == row["arm"] and type(model).__name__ == row["model_class"], "Exact inherited class")
    return model, folder, config, digest, updates


def _history_student(plan, expected, out, row, data_hash, initials, orders):
    folder, cfg = out / "fits" / row["name"], training_settings(plan["settings"])
    done = read(folder / "completed.json")
    require(done["status"] == "completed" and set(done["files"]) == fitting.PAYLOAD_FILES, "Complete fresh fit")
    for name, digest in done["files"].items():
        checked(folder / name, digest)
    checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
    history_training._unseal(checkpoint)
    initial_hash = initials[row["pair"]]["tensor_hashes"]["gru"]
    order_hash = training.canonical_state_hash(orders[row["pair"]])
    updates, provenance = cfg.epochs * cfg.batches_per_epoch, fit_provenance(plan, expected, row)
    require(checkpoint["version"] == history_training.VERSION and checkpoint["kind"] == history_training.KIND
        and checkpoint["integrity_sha256"] == done["checkpoint_integrity_sha256"]
        and checkpoint["failed"] is False and checkpoint["failure"] is None
        and checkpoint["settings"] == done["settings"] == cfg.configuration()
        and checkpoint["data_sha256"] == done["data_sha256"] == data_hash,
        "Fresh actual class/settings/data checkpoint")
    for key, value in (("source_sha256", plan["sources"]), ("runtime", plan["runtime"]), ("provenance", provenance)):
        require(training.canonical_state_hash(checkpoint[key]) == training.canonical_state_hash(done[key])
            == training.canonical_state_hash(value), "Fresh external " + key)
    require(done["name"] == row["name"] and done["pair"] == row["pair"] and done["arm"] == row["arm"]
        and checkpoint["successful_updates"] == checkpoint["optimizer_steps"] == done["updates"]
        == done["optimizer_steps"] == done["flushed_updates"] == updates
        and checkpoint["cursor"] == done["cursor"] == {"epoch": cfg.epochs, "batch": 0}
        and checkpoint["log_chain_sha256"] == done["log_chain_sha256"], "Complete fresh fit boundary")
    require(checkpoint["initialization"]["tensor_sha256"] == done["initialization_sha256"] == initial_hash
        and training.canonical_tensor_hash(checkpoint["initialization"]["weights"]) == initial_hash
        and training.canonical_state_hash(checkpoint["orders"]) == done["orders_sha256"] == order_hash,
        "Fresh fit original inputs")
    require(training.canonical_tensor_hash(torch.load(folder / "initial-weights.pt", weights_only=True,
        map_location="cpu")) == initial_hash, "Actual fresh initial tensors")
    weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
    digest = training.canonical_tensor_hash(weights)
    require(digest == done["student_tensor_sha256"] == training.canonical_tensor_hash(checkpoint["student_state"]),
            "Fresh deployment tensors")
    # This constructs only the class, never restore_checkpoint/Adam.
    model = history_training._construct(cfg)
    history_training._load_weights(model, weights)
    model.eval().requires_grad_(False)
    config = model.configuration()
    require(config == checkpoint["model_configuration"] == done["model_configuration"], "Actual history semantics")
    history_control.model_identity(model, plan["settings"])
    return model, folder, config, digest, updates


def restore_students(plan, expected, context, out, deadline, inputs, *, stage):
    require(stage in ("prefit", "before"), "Explicit restoration phase")
    begin, models, receipts = time.perf_counter(), {}, {}
    data, initials, orders, input_receipt = inputs
    if stage == "before":
        boundary = read(out / "all-fits-completed.json")
        expected_receipts = {f"fits/{row['name']}/completed.json"
            for row in protocol.training_manifest(context.settings)}
        require(boundary["plan_sha256"] == expected and boundary["new_fits"] == 3
            and set(boundary["files"]) == expected_receipts, "All three fresh fits bound before deployment")
        for name, digest in boundary["files"].items():
            checked(out / name, digest)
    for name, digest in context.cache_source["members"].items():
        check_cap(deadline)
        checked(out / "inherited" / name, digest)
    require(training.canonical_tensor_hash(data) == input_receipt["data_sha256"], "Public data unchanged")
    for row in protocol.fit_manifest(context.settings):
        if stage == "prefit" and row["new_fit"]:
            continue
        check_cap(deadline)
        if row["new_fit"]:
            model, folder, config, digest, updates = _history_student(plan, expected, out, row,
                input_receipt["data_sha256"], initials, orders)
        else:
            model, folder, config, digest, updates = _inherited_student(context.settings, context.cache_plan,
                out, row, input_receipt["data_sha256"], initials, orders)
        require(training.canonical_tensor_hash(model.state_dict()) == digest, "Restored student exact tensors")
        path = out / "model-states" / f"{row['name']}-{stage}.pt"
        path.parent.mkdir(exist_ok=True)
        cache.save_torch(path, model.state_dict())
        models[row["name"]] = model
        receipts[row["name"]] = {"kind": row["arm"], "model_class": type(model).__name__, "configuration": config,
            "checkpoint_path": (folder / "checkpoint.pt").relative_to(out).as_posix(),
            "checkpoint_sha256": sha(folder / "checkpoint.pt"), "weights_sha256": sha(folder / "weights.pt"),
            "student_tensor_sha256": digest, "successful_updates": updates,
            "snapshot_path": path.relative_to(out).as_posix(), "snapshot_sha256": sha(path)}
    name = "inherited-models-restored.json" if stage == "prefit" else "all-models-restored.json"
    receipt = {"plan_sha256": expected, "stage": stage, "models": receipts, "optimizer_constructed": False,
        "restoration_optimizer_updates": 0, "constructor_rng_isolated_and_weights_overwritten": True,
        "constructor_seeds": {"inherited": 0, "history": 410}, "wall_seconds": time.perf_counter() - begin,
        "unix_time": time.time()}
    write(out / name, receipt)
    check_cap(deadline)
    return models, receipt


def control_cases(settings, panel, bound):
    return [control.ControlCase(streams.seed(settings, f"control/reset/{index}", contract=bound),
        streams.seed(settings, f"control/actuator_noise/{index}", contract=bound),
        streams.schedule(settings, index, panel, contract=bound)) for index in range(settings["control_episodes"])]


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


def reference_control(plan, panel, arm, inputs_by_step, out, deadline=float("inf"), progress=None, *, cases, bound):
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
                    seed=streams.seed(plan, f"planner/particle_filter/{index}", contract=bound), particles=plan["particles"],
                    noise_std=plan["noise_std"], measurement_std=plan["filter_bandwidth"], frame_skip=2)
                    for index, packet in enumerate(packets)]
            elif arm == "public_kinematic":
                observers = [KinematicObserver(native_model, packet, frame_skip=2) for packet in packets]
        uniform = np.random.default_rng(streams.seed(plan, "floor/uniform/0", contract=bound)) if arm == "uniform" else None
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
            actual_actions = np.stack([env.episode_record()["policy"]["commands"][-1] for env in envs])
            require(np.array_equal(actual_actions, actions), "Native issued commands differ from selection")
            previous_actions = actual_actions
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


def saved_results(settings, out, deadline):
    rows = []
    for row in protocol.execution_order(settings):
        check_cap(deadline)
        records = base.load_records(out / row["path"] / "episodes")
        require(len(records) == settings["control_episodes"], "Complete saved result cases")
        rows.append({"panel": row["panel"], "label": row["label"],
            "case_ids": [f"control/{index}" for index in range(len(records))],
            "native_rewards": np.stack([record["audit"]["rewards"] for record in records])})
    value = results.evaluate_rows(settings, rows)
    write(out / "results.json", value)
    return value


def _failure(out, expected, error, progress, begin):
    if (out / "completed.json").exists():
        try:
            (out / "completed.json").rename(out / "invalid-completion.json")
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Invalid completion demotion failed: {preservation_error!r}")
    try:
        write(out / "failed.json", {"status": "failed", "version": VERSION, "plan_sha256": expected,
            "error": repr(error), "exception_type": type(error).__name__, "progress": progress,
            "wall_seconds": time.monotonic() - begin, "files": file_members(out)})
    except BaseException as preservation_error:  # noqa: BLE001
        error.add_note(f"Failure receipt preservation failed: {preservation_error!r}")


def run(path, expected, out, *, engineering=False, freeze_ref=None):
    """Exclusive CLI boundary. Authentication failures leave a terminal receipt."""
    begin, out = time.monotonic(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan, context = validate(path, expected, engineering=engineering, freeze_ref=freeze_ref)
        return run_validated(plan, expected, out, context, begin=begin, output_created=True,
            final_validate=lambda: validate(path, expected, engineering=engineering, freeze_ref=freeze_ref))
    except BaseException as error:
        if not (out / "failed.json").exists():
            _failure(out, expected, error, {"phase": "validate-or-setup"}, begin)
        raise


def run_validated(plan, expected, out, context, *, begin=None, final_validate=None, output_created=False):
    """Internal execution path, also used with authentic engineering fixtures.

    It requires complete entry/exit validation supplied by the outer byte-bound
    runner. Test doubles may replace numerical work, never authorize a study.
    """
    begin = time.monotonic() if begin is None else begin
    out = Path(out)
    if output_created:
        require(out.is_dir() and not out.is_symlink() and not any(out.iterdir()), "Exclusive empty attempt")
    else:
        out.mkdir(parents=True, exist_ok=False)
    progress = {"phase": "initial-validation"}
    try:
        require(callable(final_validate), "Mandatory entry/exit source and lineage authentication")
        verified_plan, verified_context = final_validate()
        require(training.canonical_state_hash(verified_plan) == training.canonical_state_hash(plan), "Plan identity drift")
        require(verified_context.settings == context.settings, "Validated settings drift")
        context = verified_context
        settings, bound = context.settings, context.streams
        protocol.validate_settings(settings, engineering=plan["engineering"])
        deadline = begin + plan["cap_seconds"]
        check_cap(deadline)
        torch.set_num_threads(settings["threads"])
        torch.use_deterministic_algorithms(True)
        write(out / "started.json", {"version": VERSION, "study": STUDY, "plan_sha256": expected,
            "engineering": plan["engineering"], "unix_time": time.time(), "elapsed_seconds": time.monotonic() - begin})
        write(out / "random-streams.json", plan["random_stream_contract"])
        progress = {"phase": "copy-inherited"}
        tick = time.perf_counter()
        copy_inherited(plan, context, out, deadline)
        copy_seconds = time.perf_counter() - tick
        progress = {"phase": "training-input-authentication"}
        inputs = load_training_inputs(plan, context, out, deadline)
        data, initials, orders, input_receipt = inputs
        progress = {"phase": "restore-six-before-fitting"}
        prefit_models, prefit_receipt = restore_students(plan, expected, context, out, deadline, inputs, stage="prefit")
        inherited_names = {row["name"] for row in protocol.restore_manifest(settings)}
        require(set(prefit_models) == inherited_names and len(prefit_models) == 6, "All6 inherited models before fitting")
        del prefit_models
        write(out / "training-started.json", {"plan_sha256": expected, "unix_time": time.time(),
            "elapsed_seconds": time.monotonic() - begin,
            "training_inputs_sha256": sha(out / "training-inputs.json"),
            "inherited_models_restored_sha256": sha(out / "inherited-models-restored.json"),
            "fit_order": [row["name"] for row in protocol.training_manifest(settings)]})
        cfg, fit_receipts = training_settings(settings), []
        for row in protocol.training_manifest(settings):
            progress = {"phase": "fit", "name": row["name"], "pair": row["pair"]}
            check_cap(deadline)
            pair = row["pair"]
            receipt = fitting.fit_one(initials[pair]["states"]["gru"], orders[pair], data,
                out / "fits" / row["name"], settings=cfg,
                expected_initial_sha256=input_receipt["pairs"][pair]["initial_tensor_sha256"],
                expected_orders_sha256=input_receipt["pairs"][pair]["orders_sha256"],
                expected_data_sha256=input_receipt["data_sha256"], provenance=fit_provenance(plan, expected, row),
                source_sha256=plan["sources"], runtime=plan["runtime"], pair=pair, name=row["name"],
                deadline=deadline, progress=progress)
            require(read(out / "fits" / row["name"] / "completed.json") == receipt, "Fit receipt return/file equality")
            fit_receipts.append(receipt)
            check_cap(deadline)
        write(out / "all-fits-completed.json", {"plan_sha256": expected, "new_fits": 3,
            "fit_order": [row["name"] for row in protocol.training_manifest(settings)],
            "files": {f"fits/{row['name']}/completed.json": sha(out / "fits" / row["name"] / "completed.json")
                      for row in protocol.training_manifest(settings)},
            "optimizer_steps": sum(row["optimizer_steps"] for row in fit_receipts),
            "training_started_sha256": sha(out / "training-started.json"),
            "unix_time": time.time(), "elapsed_seconds": time.monotonic() - begin})
        progress = {"phase": "restore-all-nine-before-fresh-draws"}
        models, restored = restore_students(plan, expected, context, out, deadline, inputs, stage="before")
        require(set(models) == {row["name"] for row in protocol.fit_manifest(settings)} and len(models) == 9,
                "All9 final actual classes restored before fresh draws")
        require(all(restored["models"][name]["student_tensor_sha256"] == prefit_receipt["models"][name]["student_tensor_sha256"]
                    for name in inherited_names), "Inherited weights unchanged during new fitting")
        evaluation_started = time.monotonic() - begin
        write(out / "evaluation-started.json", {"plan_sha256": expected, "unix_time": time.time(),
            "elapsed_seconds": evaluation_started, "all_fits_completed_sha256": sha(out / "all-fits-completed.json"),
            "all_models_restored_sha256": sha(out / "all-models-restored.json"),
            "random_streams_sha256": sha(out / "random-streams.json")})
        progress = {"phase": "fresh-input-generation"}
        innovation_stems, tick = [], time.perf_counter()
        for step in range(settings["steps"]):
            check_cap(deadline)
            stem = out / "innovations" / "control" / f"{step:03d}"
            artifacts.save_inputs(stem, streams.draw_control_inputs(settings, step, contract=bound), f"planner/control/{step}")
            innovation_stems.append(stem)
        innovation_seconds, control_times = time.perf_counter() - tick, []
        for row in protocol.execution_order(settings):
            progress = {"phase": "control", "panel": row["panel"], "model": row["label"]}
            check_cap(deadline)
            cases = control_cases(settings, row["panel"], bound)
            if "fit" in row:
                helper = episode.learned_control if row["arm"] == history_training.KIND else control.learned_control
                timing = helper(settings, models[row["fit"]], row["panel"], innovation_stems,
                    out / row["path"], deadline, progress, cases=cases)
            else:
                timing = reference_control(settings, row["panel"], row["reference"], innovation_stems,
                    out / row["path"], deadline, progress, cases=cases, bound=bound)
            check_cap(deadline)
            control_times.append({"path": row["path"], **timing})
            print(json.dumps({"completed_control": f"{row['panel']}/{row['label']}"}), flush=True)
        require(len(control_times) == 42, "Complete42 rows")
        write(out / "control-completed.json", {"plan_sha256": expected, "rows": len(control_times),
            "row_order": [row["path"] for row in control_times], "unix_time": time.time(),
            "elapsed_seconds": time.monotonic() - begin, "evaluation_started_sha256": sha(out / "evaluation-started.json")})
        progress = {"phase": "final-authentication-and-snapshots"}
        verified_plan, _ = final_validate()
        require(training.canonical_state_hash(verified_plan) == training.canonical_state_hash(plan), "Final plan drift")
        final_hashes, snapshots = {}, {}
        for name, model in models.items():
            check_cap(deadline)
            digest = training.canonical_tensor_hash(model.state_dict())
            binding = restored["models"][name]
            require(digest == binding["student_tensor_sha256"], "Evaluation changed learned weights")
            checked(out / binding["checkpoint_path"], binding["checkpoint_sha256"])
            checked((out / binding["checkpoint_path"]).with_name("weights.pt"), binding["weights_sha256"])
            path = out / "model-states" / f"{name}-after.pt"
            cache.save_torch(path, model.state_dict())
            snapshots[path.relative_to(out).as_posix()] = sha(path)
            final_hashes[name] = digest
        write(out / "final-models.json", {"student_tensor_sha256": final_hashes, "files": snapshots,
            "evaluation_optimizer_updates": 0, "all_models_restored_sha256": sha(out / "all-models-restored.json")})
        progress = {"phase": "saved-result-arithmetic"}
        saved_results(settings, out, deadline)
        coverage = protocol.coverage(settings)
        costs = {"inheritance_copy_wall_seconds": copy_seconds,
            "training_input_validation_wall_seconds": input_receipt["wall_seconds"],
            "prefit_restore_wall_seconds": prefit_receipt["wall_seconds"],
            "final_restore_wall_seconds": restored["wall_seconds"],
            "new_fit_wall_seconds": sum(row["wall_seconds"] for row in fit_receipts),
            "fit_wall_seconds": {row["name"]: row["wall_seconds"] for row in fit_receipts},
            "new_fits": 3, "new_optimizer_steps": sum(row["optimizer_steps"] for row in fit_receipts),
            "inherited_optimizer_steps": 0, "evaluation_optimizer_steps": 0,
            "new_prediction_episodes": 0, "diagnostic_roots": 0, "astra_calls": 0,
            "innovation_generation_and_storage_seconds": innovation_seconds,
            "control_row_wall_seconds": sum(row["row_wall_seconds"] for row in control_times),
            "control_setup_seconds": sum(row["setup_seconds"] for row in control_times),
            "control_decision_seconds": sum(sum(row["decision_seconds"]) for row in control_times),
            "control_native_step_seconds": sum(sum(row["native_step_seconds"]) for row in control_times),
            "control_row_times": control_times, "coverage": coverage,
            "cache_parent_costs": context.cache_source["prior_costs"],
            "prerequisite_costs": context.prerequisite_source["prior_costs"], "compute_matched": False,
            "accounting": "Whole attempt includes source/input authentication, copies, six prefit and nine final deployment restorations, three fresh fits, every row and storage/hash work. Row sub-times overlap. Historical prerequisite cumulative costs already include cache ancestry; add that total only once. Audit wall is separate."}
        require(costs["new_optimizer_steps"] == coverage["new_optimizer_updates"], "Complete declared fresh training updates")
        write(out / "costs.json", costs)
        progress = {"phase": "terminal-manifest"}
        members = file_members(out, deadline)
        expected_members = set(_experiment().expected_members(settings)) - {"completed.json"}
        require(set(members) == expected_members, "Exact complete execution artifact membership")
        check_cap(deadline)
        total = time.monotonic() - begin
        previous = context.prerequisite_source["prior_costs"]["cumulative_attempt_wall_seconds"]
        receipt = {"status": "completed", "version": VERSION, "study": STUDY, "plan_sha256": expected,
            "engineering": plan["engineering"], "new_fits": 3, "inherited_fits": 6, "restored_models": 9,
            "prefit_restored_models": 6, "control_rows": 42, "diagnostic_roots": 0, "astra_calls": 0,
            "new_optimizer_steps": costs["new_optimizer_steps"], "wall_seconds": total,
            "evaluation_started_elapsed_seconds": evaluation_started,
            "cumulative_attempt_wall_seconds": previous + total, "files": members}
        write(out / "completed.json", receipt)
        check_cap(deadline)
        return receipt
    except BaseException as error:
        _failure(out, expected, error, progress, begin)
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--request", type=Path, required=True,
        help="Explicit settings, lineage_refs, historical_registries, caps and engineering_evidence JSON")
    prep.add_argument("--out", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--plan", type=Path, required=True)
    execution.add_argument("--expected-plan-sha256", required=True)
    execution.add_argument("--out", type=Path, required=True)
    execution.add_argument("--engineering", action="store_true")
    execution.add_argument("--freeze", type=Path)
    execution.add_argument("--expected-freeze-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        request = read(args.request)
        result = prepare(args.out, **request)
    else:
        require((args.freeze is None) == (args.expected_freeze_sha256 is None), "Both freeze path and SHA required")
        freeze = None if args.freeze is None else {"path": str(args.freeze), "sha256": args.expected_freeze_sha256}
        result = run(args.plan, args.expected_plan_sha256, args.out, engineering=args.engineering, freeze_ref=freeze)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
