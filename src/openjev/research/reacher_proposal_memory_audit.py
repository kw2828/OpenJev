"""Saved-output audit for additive proposal memory, without new search or learning.

The original shared innovation files remain authenticated. Only independently
rederived inputs enter the unchanged CEM arithmetic checker. The inherited
native, observer and accounting checks are reused without global rebinding.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from openjev.research.reacher_tracking_audit import (
    COMMON,
    COUNT_NAMES,
    PLANNED,
    array,
    array_hash,
    audit_episode,
    authenticate_members,
    bank_counts,
    budget,
    checked_model,
    equal,
    file_hash,
    finite_times,
    load_inputs,
    model_hash,
    public_roots,
    replay_candidates,
    replay_identification,
    require,
    verify_bank_metadata,
    verify_planner_configuration,
    verify_search,
    verify_snapshot,
)

VERSION = "reacher-proposal-memory-saved-audit-v1"
ARMS = ("nominal", "public_gain", "true_state")
MODES = ("cold", "repeat_last", "shift_plan")


def derive_inputs(inputs, *, mode, step, steps, horizon, block, target,
                  previous_target=None, previous_sequence=None, previous_command=None):
    """Independent causal centering; no proposal-memory implementation import."""
    require(mode in MODES and type(step) is int and type(steps) is int and 0 <= step < steps
            and type(horizon) is int and type(block) is int and 1 <= block <= horizon <= steps,
            "proposal cursor/configuration")
    target = array(target, (2,), np.float32, "current public target")
    h = min(horizon, steps-step); chunks = (horizon+block-1)//block
    require(set(inputs) == {"initial", "random_extra", "cem/1", "cem/2", "cem/3"}, "base input fieldset")
    for name, count in zip(("initial", "random_extra", "cem/1", "cem/2", "cem/3"),
                           (64, 192, 64, 64, 63), strict=True):
        array(inputs[name], (1, count, chunks, 2), np.float64, "base innovations " + name)
    if step == 0:
        require(previous_target is None and previous_sequence is None and previous_command is None,
                "startup without invented history")
        reason = "startup"
    else:
        previous_target = array(previous_target, (2,), np.float32, "previous observed target")
        previous_sequence = array(previous_sequence, (min(horizon, steps-step+1), 2), np.float32,
                                  "previous complete selected sequence")
        previous_command = array(previous_command, (2,), np.float32, "previous issued command")
        equal(previous_sequence[0], previous_command, "previous selected first action acknowledged")
        require((np.abs(previous_sequence) <= 1).all(), "cached normalized actions")
        reason = "none" if np.array_equal(target, previous_target) else "observed_target_change"
    center = np.zeros((chunks, 2), np.float64); shifted = None
    if reason == "none" and mode == "repeat_last":
        center[:(h+block-1)//block] = previous_command.astype(np.float64)
    elif reason == "none" and mode == "shift_plan":
        shifted = previous_sequence[1:].copy()
        if len(shifted) < h:
            shifted = np.concatenate((shifted, np.repeat(previous_sequence[-1:], h-len(shifted), axis=0)))
        shifted = shifted[:h].copy()
        for index, first in enumerate(range(0, h, block)):
            segment = shifted[first:min(first+block, h)]
            for column in range(2):
                center[index, column] = np.mean(segment[:, column], dtype=np.float64)
    transformed = dict(inputs)
    reused = not bool(np.any(center != 0))
    if not reused:
        initial = inputs["initial"].copy()
        scales = np.r_[np.full(32, .25), np.full(32, .75)]
        initial[:, 7:] += center[None, None]/scales[None, 7:, None, None]
        require(np.isfinite(initial).all(), "finite derived innovations")
        transformed["initial"] = initial
    return transformed, {"horizon": h, "center": center, "shifted_sequence": shifted,
                         "reset_reason": reason, "input_object_reused": reused}


MEMORY_VERSION = "reacher-cem-proposal-memory-v1"
ARITHMETIC = "initial[:,7:64] + center / scales[7:64]; unchanged kernel then multiplies scales"
MEMORY_COUNTS = (
    "prepare_attempted", "prepare_completed", "commit_attempted", "commit_completed",
    "failed_operations", "startup_prepares", "target_change_prepares", "zero_center_prepares",
    "transformed_prepares", "projected_blocks", "column_mean_calls", "tail_actions_repeated",
    "transformed_initial_scalars", "input_payload_bytes_copied", "input_identity_bytes_hashed",
    "snapshot_calls",
)
MEMORY_TIMES = ("setup_wall_seconds", "prepare_wall_seconds", "commit_wall_seconds", "snapshot_wall_seconds")


def new_counts():
    return dict.fromkeys(MEMORY_COUNTS, 0)


def memory_configuration(mode, steps, horizon, block):
    return {"version": MEMORY_VERSION, "mode": mode, "steps": steps,
            "planning_horizon": horizon, "action_block": block, "cases_per_instance": 1,
            "rng_draws": 0, "native_calls": 0,
            "reset_rule": "startup or observed current public target change only",
            "cache_boundary": "commit after actual issued-command acknowledgement",
            "projection": "one-action shift, hold-pad, explicit float64 column means in current blocks",
            "transform_arithmetic": ARITHMETIC,
            "payload_bytes_scope": "unique retained ndarray payloads; excludes Python/JSON overhead and returned copies",
            "timing_scope": "setup/prepare/commit/snapshot separately; caller adds native work and trace storage",
            "run_status_authority": "enclosing protocol and execution receipts"}


def verify_memory_costs(costs, expected, retained):
    require(set(costs) == set(MEMORY_COUNTS) | set(MEMORY_TIMES) | {"retained_numpy_payload_bytes"},
            "exact proposal cost fields")
    for name in MEMORY_COUNTS:
        require(type(costs[name]) is int and costs[name] == expected[name], "proposal count " + name)
    require(type(costs["retained_numpy_payload_bytes"]) is int
            and costs["retained_numpy_payload_bytes"] == retained, "retained unique proposal payload bytes")
    finite_times(costs, MEMORY_TIMES)


def audit_prepare(trace, inputs, *, mode, step, steps, horizon, block, target, prior_counts,
                  prior_costs, previous_target=None, previous_sequence=None, previous_command=None):
    """Validate one pre-commit trace from prior acknowledged native commands."""
    transformed, info = derive_inputs(inputs, mode=mode, step=step, steps=steps, horizon=horizon,
        block=block, target=target, previous_target=previous_target,
        previous_sequence=previous_sequence, previous_command=previous_command)
    expected_trace = {"version": MEMORY_VERSION, "mode": mode, "step": step, "horizon": info["horizon"],
        "action_block": block, "reset_reason": info["reset_reason"], "public_target": target.tolist(),
        "center": info["center"].tolist(),
        "shifted_sequence": None if info["shifted_sequence"] is None else info["shifted_sequence"].tolist(),
        "base_input_identities": {k: array_hash(v) for k, v in inputs.items()},
        "transformed_input_identities": {k: array_hash(v) for k, v in transformed.items()},
        "input_object_reused": info["input_object_reused"], "transform_arithmetic": ARITHMETIC}
    require(set(trace) == set(expected_trace) | {"costs"}, "exact proposal trace fields")
    require(type(trace["step"]) is int and type(trace["horizon"]) is int
            and type(trace["action_block"]) is int and type(trace["input_object_reused"]) is bool,
            "proposal trace scalar types")
    require({k: trace[k] for k in expected_trace} == expected_trace, "causal proposal trace/identity reconstruction")
    delta = new_counts()
    delta["prepare_attempted"] = delta["prepare_completed"] = 1
    delta["commit_attempted"] = delta["commit_completed"] = int(step > 0)
    delta["startup_prepares"] = int(info["reset_reason"] == "startup")
    delta["target_change_prepares"] = int(info["reset_reason"] == "observed_target_change")
    delta["zero_center_prepares"] = int(info["input_object_reused"])
    delta["transformed_prepares"] = int(not info["input_object_reused"])
    chunks = (horizon+block-1)//block
    payload = sum(v.nbytes for v in inputs.values())
    delta["input_identity_bytes_hashed"] = payload*(1+delta["transformed_prepares"])
    delta["input_payload_bytes_copied"] = (payload+inputs["initial"].nbytes)*delta["transformed_prepares"]
    delta["transformed_initial_scalars"] = 57*chunks*2*delta["transformed_prepares"]
    shifted_bytes = 0
    if info["shifted_sequence"] is not None:
        delta["projected_blocks"] = (info["horizon"]+block-1)//block
        delta["column_mean_calls"] = 2*delta["projected_blocks"]
        delta["tail_actions_repeated"] = max(0, info["horizon"]-(len(previous_sequence)-1))
        shifted_bytes = 8*info["horizon"]
    retained = 8+16*chunks if step == 0 else 24+8*len(previous_sequence)+16*chunks+shifted_bytes
    expected_counts = {k: prior_counts[k]+delta[k] for k in MEMORY_COUNTS}
    verify_memory_costs(trace["costs"], expected_counts, retained)
    for key in MEMORY_TIMES:
        require(trace["costs"][key] >= prior_costs[key], "monotonic proposal time " + key)
    require(trace["costs"]["setup_wall_seconds"] == prior_costs["setup_wall_seconds"]
            and trace["costs"]["snapshot_wall_seconds"] == prior_costs["snapshot_wall_seconds"],
            "no repeated setup or hidden proposal snapshot")
    if step == 0:
        require(trace["costs"]["commit_wall_seconds"] == 0., "no unacknowledged startup commit time")
    return transformed, {"counts": delta,
        "wall_seconds": trace["costs"]["prepare_wall_seconds"]-prior_costs["prepare_wall_seconds"]}


def audit_memory_snapshots(initial, final, *, mode, steps, horizon, block, counts,
                           last_sequence, last_command, last_target, last_trace,
                           prepare_seconds, enclosing_seconds):
    fields = {"configuration", "next_step", "pending", "last_target", "last_selected_sequence",
              "last_issued_command", "failed", "failure", "last_trace", "costs"}
    require(set(initial) == set(final) == fields, "exact proposal snapshot fields")
    config = memory_configuration(mode, steps, horizon, block)
    require(initial["configuration"] == final["configuration"] == config, "fixed proposal configuration")
    require(initial["next_step"] == 0 and final["next_step"] == steps
            and type(initial["next_step"]) is int and type(final["next_step"]) is int,
            "acknowledged proposal cursor")
    for snap in (initial, final):
        require(snap["pending"] is None and snap["failed"] is False and snap["failure"] is None,
                "successful proposal snapshot with no pending action")
    for key in ("last_target", "last_selected_sequence", "last_issued_command", "last_trace"):
        require(initial[key] is None, "empty initial proposal cache")
    initial_counts = new_counts(); initial_counts["snapshot_calls"] = 1
    verify_memory_costs(initial["costs"], initial_counts, 0)
    require(initial["costs"]["prepare_wall_seconds"] == initial["costs"]["commit_wall_seconds"] == 0.,
            "no unreported startup proposal work")
    equal(final["last_target"], last_target, "final acknowledged proposal target")
    equal(final["last_selected_sequence"], last_sequence, "final acknowledged proposal sequence")
    equal(final["last_issued_command"], last_command, "final acknowledged proposal command")
    require(final["last_trace"] == {k: v for k, v in last_trace.items() if k != "costs"},
            "final retained proposal trace")
    expected = dict(counts)
    expected["commit_attempted"] += 1; expected["commit_completed"] += 1; expected["snapshot_calls"] += 1
    chunks = (horizon+block-1)//block
    retained = 16+8*len(last_sequence)+16*chunks
    if last_trace["shifted_sequence"] is not None:
        retained += 8*len(last_trace["shifted_sequence"])
    verify_memory_costs(final["costs"], expected, retained)
    require(final["costs"]["setup_wall_seconds"] == initial["costs"]["setup_wall_seconds"],
            "proposal setup unchanged")
    for key in MEMORY_TIMES:
        require(final["costs"][key] >= last_trace["costs"][key], "final proposal time " + key)
    require(final["costs"]["prepare_wall_seconds"] == last_trace["costs"]["prepare_wall_seconds"]
            and abs(final["costs"]["prepare_wall_seconds"]-prepare_seconds) <= 1e-12,
            "no omitted or repeated proposal preparation time")
    require(np.isfinite(enclosing_seconds) and enclosing_seconds >= 0
            and final["costs"]["prepare_wall_seconds"]+final["costs"]["commit_wall_seconds"] <= enclosing_seconds+1e-8,
            "nested proposal calls charged inside row proposal time")


def audit_row(folder, *, nominal_model, inputs_by_step, deadline=float("inf")):
    """Audit one authenticated completed row; no result threshold or new action.

    Caller binds source/runtime, completion bytes and external shared-input files
    to its prospective run manifest. Return counts, recorded cost and max errors.
    """
    started_at = time.perf_counter(); folder = Path(folder)
    read = lambda name: json.loads((folder/name).read_text())
    started = read("started.json"); completed = read("completed.json")
    cfg = started
    arm, steps, horizon, block = cfg["arm"], cfg["steps"], cfg["planning_horizon"], cfg["action_block"]
    mode = cfg["proposal_mode"]
    require(mode in MODES and completed["proposal_mode"] == mode, "declared proposal mode")
    require(arm in ARMS and type(steps) is int and steps > 0 and len(inputs_by_step) == steps,
            "complete row role/horizon/inputs")
    require(type(cfg["window"]) is int and cfg["window"] > 0 and type(cfg["freeze_after"]) is int
            and 1 <= cfg["freeze_after"] <= steps, "identifier window/freeze boundary")
    gains = np.asarray(cfg["gain_grid"], np.float64)
    require(gains.ndim == 1 and len(gains) >= 2 and np.isfinite(gains).all() and np.all(gains > 0)
            and np.all(np.diff(gains) > 0) and np.any(gains == 1.), "explicit global gain grid")
    expected = {"started.json", "initial-controller.json", "initial-proposal-memory.json", "proposal-memory.json", "episode.json", "controller.json"}
    expected |= {f"decisions/{t:03d}.{suffix}" for t in range(steps) for suffix in ("npz", "json")}
    expected |= {f"observations/{t+1:03d}.json" for t in range(steps)}
    authenticate_members(folder, completed, expected, deadline)
    completion_sha = file_hash(folder/"completed.json", deadline)
    require(cfg["frame_skip"] == 2 and cfg["automatic_retry"] is False
            and type(cfg["engineering"]) is bool and completed["engineering"] is cfg["engineering"]
            and completed["arm"] == arm and completed["steps"] == steps
            and completed["native_decisions"] == completed["policy_decisions"] == completed["observation_updates"]
            == steps, "row receipt/settings identity")
    require(len(cfg["inputs_by_step"]) == steps, "full external input manifest")
    for bound, supplied in zip(cfg["inputs_by_step"], inputs_by_step, strict=True):
        require(Path(bound["stem"]).resolve() == Path(supplied).resolve(), "caller explicit innovation stem")
        for suffix in ("npz", "json"):
            path = Path(supplied).with_suffix("."+suffix)
            require(not path.is_symlink() and file_hash(path, deadline) == bound[suffix+"_sha256"],
                    "authenticated external innovation bytes")
    initial = read("initial-controller.json")
    require(initial["arm"] == arm and initial["steps"] == steps and initial["step"] == 0
            and initial["failed"] is False and initial["pending_command"] is None
            and initial["gain_estimate"] == 1., "initial public controller boundary")
    episode = read("episode.json")
    require(episode["metadata"]["horizon"] == steps and episode["metadata"]["seed"] == cfg["reset_seed"],
            "row episode reset/horizon")
    native = audit_episode(episode, nominal_model, deadline=deadline)
    packets = np.asarray(episode["policy"]["packets"], np.float32)
    commands = np.asarray(episode["policy"]["commands"], np.float32)
    public_q, public_v = public_roots(packets)
    equal(initial["observer"]["packets"], packets[:1], "initial public packet only")
    equal(initial["observer"]["qpos_estimate"], public_q[0], "initial public qpos")
    equal(initial["observer"]["qvel_estimate"], public_v[0], "initial zero velocity")
    require(initial["freeze_after"] == cfg["freeze_after"], "initial freeze boundary")
    for name in ("decisions_completed", "observations_completed", "identifier_updates_completed"):
        require(initial["costs"][name] == 0, "no unreported pre-row control work")
    if arm == "zero":
        require(initial["planner"] is None, "zero startup no planner")
    else:
        bank = initial["planner"]
        require(bank["failed"] is False and bank["last_operation"] is None
                and bank["last_native_journal"] is None and len(bank["members"]) == len(gains),
                "unrun eager planning bank")
        equal(bank["configuration"]["allowed_gains"], gains, "startup global allowed grid")
        for member, g in zip(bank["members"], gains, strict=True):
            require(member["gain"] == g and member["failed"] is False, "startup model member")
            verify_planner_configuration(member["configuration"], model_sha=model_hash(checked_model(nominal_model, float(g))),
                                         steps=steps, horizon=horizon, block=block, noise_std=cfg["noise_std"])
            for key in (*COUNT_NAMES, "operations_completed", "operations_failed", "operation_wall_seconds"):
                require(member["lifetime"][key] == 0, "zero startup lifetime")
    cumulative = [{**dict.fromkeys(COUNT_NAMES, 0), "operations_completed": 0} for _ in gains]
    gain = 1.; retained = []; identifier_steps = 0; candidate_steps = 0; selected_steps = 0
    max_native = max_geometry = max_identifier = 0.; last_snapshot = None
    memory_initial = read("initial-proposal-memory.json")
    memory_counts = new_counts()
    memory_counts["snapshot_calls"] = 1
    previous_costs = memory_initial["costs"]
    previous_sequence = previous_command = previous_target = None
    prepare_seconds = 0.
    for step in range(steps):
        budget(deadline)
        with np.load(folder/f"decisions/{step:03d}.npz", allow_pickle=False) as saved:
            require(set(saved.files) == COMMON | (set() if arm == "zero" else PLANNED), "exact decision NPZ fields")
            values = {key: saved[key] for key in saved.files}
        meta = read(f"decisions/{step:03d}.json")
        require(set(meta) == {"step", "arm", "input_identities", "planner_configuration", "bank_snapshot",
                              "journalcallbackmetadata", "selectedmetadata", "proposal_memory"}, "exact decision metadata fields")
        require(meta["step"] == step and meta["arm"] == arm, "decision role/cursor")
        for key, expected_value in (("public_packet", packets[step]), ("public_qpos", public_q[step]),
                                    ("public_qvel", public_v[step]), ("action", commands[step])):
            equal(values[key], expected_value, "real public/command boundary " + key)
        root_q, root_v = public_q[step].copy(), public_v[step]
        actual_gain = episode["metadata"]["gear_multiplier"][step]
        if arm == "true_state":
            state = episode["audit"]["decision_states"][step]
            root_q, root_v = np.asarray(state["qpos"], np.float64).copy(), np.asarray(state["qvel"], np.float64)
            root_q[2:] = packets[step, 4:6].astype(np.float64)
        equal(values["root_qpos"], root_q, "allowed state provenance qpos")
        equal(values["root_qvel"], root_v, "allowed state provenance qvel")
        allowed_gain = actual_gain if arm in ("public_gain", "true_state") else gain
        require(float(array(values["planning_gain"], (), np.float64, "gain scalar")) == allowed_gain
                and np.any(gains == allowed_gain), "causal allowed current gain")
        inputs = load_inputs(inputs_by_step[step], horizon, block)
        inputs, info = audit_prepare(meta["proposal_memory"], inputs, mode=mode, step=step, steps=steps,
            horizon=horizon, block=block, target=packets[step, 4:6], previous_target=previous_target,
            previous_sequence=previous_sequence, previous_command=previous_command,
            prior_counts=memory_counts, prior_costs=previous_costs)
        previous_costs = meta["proposal_memory"]["costs"]
        for key, value in info["counts"].items():
            memory_counts[key] += value
        prepare_seconds += info["wall_seconds"]
        require(meta["input_identities"] == {key: array_hash(value) for key, value in inputs.items()},
                "scoring inputs equal independently derived proposal inputs")
        if arm == "zero":
            equal(commands[step], np.zeros(2, np.float32), "zero action")
            require(meta["bank_snapshot"] is None and meta["planner_configuration"] is None
                    and meta["journalcallbackmetadata"] == [] and meta["selectedmetadata"] is None,
                    "zero row no hidden planning")
        else:
            search = verify_search(values, inputs, step=step, steps=steps, horizon=horizon, block=block)
            require(meta["input_identities"] == search["input_identities"], "decision input binding")
            replay = replay_candidates(nominal_model, values, step=step, noise_std=cfg["noise_std"], deadline=deadline)
            max_native = max(max_native, replay["max_native_abs_error"])
            max_geometry = max(max_geometry, replay["max_geometry_abs_error"])
            candidate_steps += replay["candidate_transitions"]; selected_steps += 1
            roots = {"qpos": array_hash(root_q[None]), "qvel": array_hash(root_v[None]),
                     "public_target": array_hash(packets[step, None, 4:6])}
            require(len(meta["journalcallbackmetadata"]) == 4, "four paid callbacks")
            selected_gain_index = int(np.flatnonzero(gains == allowed_gain)[0])
            for j, callback in enumerate(meta["journalcallbackmetadata"]):
                verify_bank_metadata(callback, values["sequences"][j*64:(j+1)*64], roots, start=j*64)
                for key, count in bank_counts(64, search["horizon"]).items():
                    cumulative[selected_gain_index][key] += count
            verify_bank_metadata(meta["selectedmetadata"], values["action"][None, None], roots, start=None)
            for key, count in bank_counts(1, 1).items():
                cumulative[selected_gain_index][key] += count
            cumulative[selected_gain_index]["operations_completed"] += 1
            verify_planner_configuration(meta["planner_configuration"], model_sha=replay["model_binary_sha256"],
                                         steps=steps, horizon=horizon, block=block, noise_std=cfg["noise_std"])
            last_snapshot = meta["bank_snapshot"]
            verify_snapshot(last_snapshot, nominal=nominal_model, gains=gains, gain=allowed_gain, step=step,
                            cumulative=cumulative, selected_model_sha=replay["model_binary_sha256"], steps=steps,
                            horizon=horizon, block=block, noise_std=cfg["noise_std"])
        observed = read(f"observations/{step+1:03d}.json")
        require(observed["step"] == step+1, "post-action observer cursor")
        equal(observed["public_qpos"], public_q[step+1], "public observer endpoint qpos")
        equal(observed["public_qvel"], public_v[step+1], "public observer endpoint qvel")
        update = arm == "adaptive" or (arm == "frozen" and step < cfg["freeze_after"])
        require(observed["identifier_updated"] is update, "exact frozen update boundary")
        if update:
            gain, retained, replay_id = replay_identification(nominal_model, observed["identifier_trace"],
                root_qpos=public_q[step], root_qvel=public_v[step], command=commands[step],
                next_packet=packets[step+1], gains=gains, retained=retained, prior_gain=gain, window=cfg["window"],
                step=step, deadline=deadline)
            identifier_steps += replay_id["transitions"]
            max_identifier = max(max_identifier, replay_id["max_abs_error"])
        else:
            require(observed["identifier_trace"] is None, "no phantom identification")
        require(observed["identified_gain"] == (gain if arm in ("adaptive", "frozen") else None),
                "recorded causal estimate")
        previous_sequence = values["sequences"][int(values["selected_id"])].copy()
        previous_command = commands[step].copy()
        previous_target = packets[step, 4:6].copy()
        equal(previous_sequence[0], previous_command, "proposal cache acknowledged issued action")
    proposal_snapshot = read("proposal-memory.json")
    audit_memory_snapshots(memory_initial, proposal_snapshot, mode=mode, steps=steps, horizon=horizon,
        block=block, counts=memory_counts, last_sequence=previous_sequence, last_command=previous_command,
        last_target=previous_target, last_trace=meta["proposal_memory"], prepare_seconds=prepare_seconds,
        enclosing_seconds=completed["proposal_memory_seconds"])
    require(memory_initial["costs"]["setup_wall_seconds"]+memory_initial["costs"]["snapshot_wall_seconds"]
            <= completed["setup_seconds"]+1e-8, "proposal setup inside row setup")
    require(proposal_snapshot["costs"]["snapshot_wall_seconds"] <= completed["wall_seconds"]+1e-8,
            "proposal snapshots inside row wall")
    controller = read("controller.json")
    require(controller["arm"] == arm and controller["steps"] == controller["step"] == steps
            and controller["pending_command"] is None and controller["failed"] is False
            and controller["freeze_after"] == cfg["freeze_after"] and controller["gain_estimate"] == gain,
            "terminal policy state")
    equal(controller["observer"]["qpos_estimate"], public_q[-1], "final public qpos")
    equal(controller["observer"]["qvel_estimate"], public_v[-1], "final public qvel")
    costs = controller["costs"]
    require(costs["decisions_completed"] == costs["observations_completed"] == steps
            and costs["identifier_updates_completed"] == identifier_steps//len(gains), "policy work coverage")
    finite_times(costs, ("setup_wall_seconds", "decision_wall_seconds", "observation_wall_seconds"))
    if last_snapshot is None:
        require(controller["planner"] is None, "zero final planner")
    else:
        final = dict(controller["planner"]); final.pop("last_native_journal", None)
        require({key: final[key] for key in ("costs", "last_operation")} == last_snapshot,
                "final planner cumulative accounting")
        require(initial["planner"]["configuration"] == final["configuration"], "startup/final model configuration identity")
        for first, last in zip(initial["planner"]["members"], final["members"], strict=True):
            require(first["configuration"] == last["configuration"] and first["setup_seconds"] == last["setup_seconds"],
                    "eager member identity unchanged")
        for key in ("setup_wall_seconds", "temporary_gain_model_copies", "retained_adapter_model_copies",
                    "native_data_instances", "nested_adapter_setup_seconds"):
            require(initial["planner"]["costs"][key] == final["costs"][key], "no omitted repeated bank setup")
        verify_snapshot(final, nominal=nominal_model, gains=gains, gain=allowed_gain, step=steps-1,
                        cumulative=cumulative, selected_model_sha=replay["model_binary_sha256"], steps=steps,
                        horizon=horizon, block=block, noise_std=cfg["noise_std"])
    identifier = controller["identifier"]
    if arm in ("adaptive", "frozen"):
        require(identifier["failed"] is False and identifier["step_index"] == identifier_steps//len(gains)
                and identifier["gain_estimate"] == gain, "final identifier cursor/gain")
        require(initial["identifier"] is not None and initial["identifier"]["step_index"] == 0
                and initial["identifier"]["gain_estimate"] == 1. and initial["identifier"]["window_residuals"] == [],
                "zero-history prior identifier")
        require(initial["identifier"]["configuration"] == identifier["configuration"], "identifier fixed configuration")
        config = identifier["configuration"]
        equal(config["gains"], gains, "identifier global grid")
        require(config["window"] == cfg["window"] and config["prior_gain"] == 1. and config["frame_skip"] == 2
                and config["dt"] == .02 and config["learned"] is False and config["privileged_live_state"] is False,
                "public identifier configuration")
        identifier_end = identifier_steps//len(gains)
        equal(identifier["observer"]["qpos_estimate"], public_q[identifier_end], "frozen identifier endpoint qpos")
        equal(identifier["observer"]["qvel_estimate"], public_v[identifier_end], "frozen identifier endpoint qvel")
        equal(identifier["window_residuals"], np.stack(retained), "final residual ring")
        for name in ("native_transition_attempts", "native_transitions_completed", "root_forward_calls_completed",
                     "postconstraint_refresh_calls_completed"):
            require(identifier["costs"][name] == identifier_steps, "identifier work " + name)
        for name in ("native_substeps_requested", "native_substeps_completed"):
            require(identifier["costs"][name] == 2*identifier_steps, "identifier substeps")
        for name in ("native_model_copies", "native_data_instances"):
            require(identifier["costs"][name] == initial["identifier"]["costs"][name] == len(gains),
                    "all identifier setup copies charged")
        require(identifier["costs"]["retained_residual_scalars"] == len(retained)*len(gains)
                and initial["identifier"]["costs"]["retained_residual_scalars"] == 0,
                "identifier retained residual storage")
        for name in ("native_transition_attempts", "native_transitions_completed", "native_substeps_requested",
                     "native_substeps_completed", "root_forward_calls_completed", "postconstraint_refresh_calls_completed",
                     "update_wall_seconds"):
            require(initial["identifier"]["costs"][name] == 0, "no unreported startup identification")
        require(identifier["costs"]["setup_wall_seconds"] == initial["identifier"]["costs"]["setup_wall_seconds"],
                "unchanged identifier setup")
        finite_times(identifier["costs"], ("setup_wall_seconds", "update_wall_seconds"))
        require(identifier["costs"]["setup_wall_seconds"] <= costs["setup_wall_seconds"]+1e-8
                and identifier["costs"]["update_wall_seconds"] <= costs["observation_wall_seconds"]+1e-8,
                "nested identifier paid times")
    else:
        require(identifier is None and initial["identifier"] is None, "no hidden identifier for nonidentifying row")
    finite_times(completed, ("setup_seconds", "decision_seconds", "native_seconds", "observation_seconds",
                             "decision_serialization_seconds", "proposal_memory_seconds", "wall_seconds"))
    require(sum(completed[key] for key in ("setup_seconds", "decision_seconds", "native_seconds",
                "observation_seconds", "decision_serialization_seconds", "proposal_memory_seconds")) <= completed["wall_seconds"]+1e-8,
            "row disjoint measured components inside inclusive wall")
    require(costs["setup_wall_seconds"] <= completed["setup_seconds"]+1e-8
            and costs["decision_wall_seconds"] <= completed["decision_seconds"]+1e-8
            and costs["observation_wall_seconds"] <= completed["observation_seconds"]+1e-8,
            "nested policy times inside row calls")
    require(completed["native_cost"] == native["mean_cost"], "row native cost arithmetic")
    for bound, supplied in zip(cfg["inputs_by_step"], inputs_by_step, strict=True):
        for suffix in ("npz", "json"):
            require(file_hash(Path(supplied).with_suffix("."+suffix), deadline) == bound[suffix+"_sha256"],
                    "external innovations unchanged after replay")
    authenticate_members(folder, completed, expected, deadline)
    require(file_hash(folder/"completed.json", deadline) == completion_sha, "completion unchanged during audit")
    return {"version": VERSION, "arm": arm, "proposal_mode": mode, "status": "completed", "row_completed_sha256": completion_sha,
            "native_control": native, "candidate_transitions_checked": candidate_steps,
            "selected_transitions_checked": selected_steps, "identifier_transitions_checked": identifier_steps,
            "max_candidate_state_abs_error": max_native, "max_geometry_abs_error": max_geometry,
            "max_identifier_abs_error": max_identifier, "controller_costs": costs,
            "proposal_memory": proposal_snapshot, "proposal_memory_seconds": completed["proposal_memory_seconds"],
            "row_wall_seconds": completed["wall_seconds"], "audit_wall_seconds": time.perf_counter()-started_at,
            "new_model_calls": 0, "new_planner_calls": 0, "new_rng_draws": 0,
            "scope": "Initial seed/schedule/source/runtime authenticity belongs to caller; saved arithmetic and native replay only."}
