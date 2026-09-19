"""Independent saved common-root searches, cross scores and native branches.

The caller authenticates source episodes, source/runtime identity, innovation
roles and branch noise. No search, controller, learned model or RNG is invoked.
Nominal candidates reset solver state; branches restore full integration state.
These different conventions are retained and explicitly counted, not conflated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from openjev.research.reacher_tracking_audit import (
    COMMON,
    COUNT_NAMES,
    PLANNED,
    _native_state,
    array,
    array_hash,
    authenticate_members,
    bank_counts,
    budget,
    checked_model,
    close,
    equal,
    file_hash,
    finite_times,
    geometry_reward,
    input_arrays,
    model_hash,
    native_step,
    replay_candidates,
    require,
    verify_bank_metadata,
    verify_planner_configuration,
    verify_search,
)

VERSION = "reacher-common-root-saved-audit-v1"
ASSUMPTIONS = ("nominal", "actual")
SEARCHES = ("short_a", "short_b", "long_a")
ROOT_FIELDS = {"source_role", "step", "root", "true_gain", "public_packet",
               "source_episode_sha256", "source_completed_sha256"}


def read_npz(path, fields):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(fields), "exact saved array fields")
        return {key: saved[key] for key in saved.files}


def validate_root(saved, expected, nominal):
    require(set(saved) == set(expected) == ROOT_FIELDS and saved == expected,
            "externally authenticated root metadata")
    require(saved["source_role"] in ("nominal", "public_gain", "true_state")
            and type(saved["step"]) is int and saved["step"] in (0, 50, 100, 150),
            "prospective event-onset root")
    for name in ("source_episode_sha256", "source_completed_sha256"):
        value = saved[name]
        require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "source root hash " + name)
    require(type(saved["true_gain"]) in (float, int) and np.isfinite(saved["true_gain"])
            and saved["true_gain"] > 0, "actual positive root gain")
    state = saved["root"]
    require(set(state) == {"qpos", "qvel", "raw_obs", "integration_state", "time"}, "complete source root state")
    q = array(state["qpos"], (4,), np.float64, "root qpos", json_value=True)
    v = array(state["qvel"], (4,), np.float64, "root qvel", json_value=True)
    array(state["raw_obs"], (10,), np.float64, "root raw observation", json_value=True)
    array(state["integration_state"], (mujoco.mj_stateSize(nominal, mujoco.mjtState.mjSTATE_INTEGRATION),),
          np.float64, "complete integration root", json_value=True)
    close(state["time"], saved["step"]*.02, "source decision time")
    equal(v[2:], [0., 0.], "fixed current target velocity")
    packet = array(saved["public_packet"], (8,), np.float32, "root public packet", json_value=True)
    equal(packet, np.r_[np.cos(q[:2]), np.sin(q[:2]), q[2:], 1., 0.].astype(np.float32),
          "post-target-event packet from same native root")
    return q, v, packet


def rebuild_union(searches):
    """All70 identity slots, tie-A restart selection, +0 canonical byte dedup."""
    require(set(searches) == {a+"--"+s for a in ASSUMPTIONS for s in SEARCHES}, "six complete searches")
    initial = array(searches["nominal--long_a"]["sequences"][:64], (64, 24, 2), np.float32,
                    "common first64 long proposals")
    equal(initial, searches["actual--long_a"]["sequences"][:64], "gain-independent initial bank")
    slots = [initial.copy()]; chosen = []
    for assumption in ASSUMPTIONS:
        a, b, long = (searches[assumption+"--"+name] for name in SEARCHES)
        for name, row in zip(SEARCHES, (a, b, long), strict=True):
            h = 24 if name == "long_a" else 12
            array(row["sequences"], (256, h, 2), np.float32, "complete saved winner bank")
            scores = array(row["scores"], (256,), np.float64, "complete saved winner scores")
            selected = int(array(row["selected_id"], (), np.int64, "saved selected id"))
            require(selected == int(np.argmax(scores)), "union uses scored global best")
        restart = b if b["scores"][int(b["selected_id"])] > a["scores"][int(a["selected_id"])] else a
        chosen.extend((a, restart, long))
    for row in chosen:
        sequence = row["sequences"][int(row["selected_id"])].copy()
        require((np.abs(sequence) <= 1).all(), "normalized union commands")
        if len(sequence) == 12:
            sequence = np.concatenate((sequence, np.repeat(sequence[-1:], 12, axis=0)))
        slots.append(sequence[None])
    identities = np.concatenate(slots); identities[identities == 0] = np.float32(0.)
    unique, first, mapping, seen = [], [], [], {}
    for index, sequence in enumerate(identities):
        key = sequence.tobytes(order="C")
        if key not in seen:
            seen[key] = len(unique); unique.append(sequence.copy()); first.append(index)
        mapping.append(seen[key])
    return {"slots": identities, "unique": np.stack(unique), "slot_to_unique": np.array(mapping, np.int64),
            "first_slot": np.array(first, np.int64)}


def verify_union(saved, searches):
    expected = rebuild_union(searches)
    require(set(saved) == set(expected), "exact union fields")
    for key, value in expected.items():
        actual = array(saved[key], value.shape, value.dtype, "union " + key)
        require(actual.tobytes(order="C") == value.tobytes(order="C"), "exact canonical first-occurrence union " + key)
    return expected


def verify_fresh_bank(meta, *, gain, step, horizon, model_sha):
    """A single-gain bank is constructed afresh for each paid CEM search."""
    require(set(meta) == {"last_operation", "costs"}, "compact fresh search snapshot")
    last, costs = meta["last_operation"], meta["costs"]
    require(set(last) == {"gain_index", "gain", "step", "status", "model_binary_sha256", "wall_seconds"},
            "exact completed fresh operation")
    require(last["gain_index"] == 0 and last["gain"] == gain and last["step"] == step
            and last["status"] == "completed" and last["model_binary_sha256"] == model_sha,
            "common root current gain and native model")
    expected_fields = {"setup_wall_seconds", "plan_call_wall_seconds", "plan_calls", "plan_calls_completed",
                      "input_rejections", "plan_calls_failed", "temporary_gain_model_copies",
                      "retained_adapter_model_copies", "native_data_instances", "nested_adapter_setup_seconds",
                      "aggregate_adapter_lifetime"}
    require(set(costs) == expected_fields, "complete fresh bank cost fields")
    for key in ("plan_calls", "plan_calls_completed", "temporary_gain_model_copies",
                "retained_adapter_model_copies", "native_data_instances"):
        require(type(costs[key]) is int and costs[key] == 1, "one fresh bank " + key)
    require(costs["input_rejections"] == costs["plan_calls_failed"] == 0, "no hidden search failure or rejection")
    total = costs["aggregate_adapter_lifetime"]
    require(set(total) == set(COUNT_NAMES) | {"operations_completed", "operations_failed", "operation_wall_seconds"},
            "complete aggregate native work")
    for key in COUNT_NAMES:
        expected = 4*bank_counts(64, horizon)[key]+bank_counts(1, 1)[key]
        require(type(total[key]) is int and total[key] == expected, "fresh search work " + key)
    require(total["operations_completed"] == 1 and total["operations_failed"] == 0, "one completed native search")
    finite_times(costs, ("setup_wall_seconds", "plan_call_wall_seconds", "nested_adapter_setup_seconds"))
    finite_times(total, ("operation_wall_seconds",)); finite_times(last, ("wall_seconds",))
    require(last["wall_seconds"] == costs["plan_call_wall_seconds"]
            and total["operation_wall_seconds"] <= costs["plan_call_wall_seconds"]+1e-8
            and costs["nested_adapter_setup_seconds"] <= costs["setup_wall_seconds"]+1e-8,
            "fresh search nested paid times")


def audit_search(values, meta, *, nominal, root, gain, horizon, inputs, deadline):
    q, v, packet = validate_root(root, root, nominal)
    step = root["step"]
    require(set(values) == COMMON | PLANNED, "all saved search arrays")
    for key, value in (("root_qpos", q), ("root_qvel", v), ("public_packet", packet),
                       ("public_qpos", q), ("public_qvel", v)):
        equal(values[key], value, "common-root search provenance " + key)
    require(float(array(values["planning_gain"], (), np.float64, "search gain")) == gain,
            "declared search gain")
    require(set(meta) == {"step", "arm", "input_identities", "planner_configuration", "bank_snapshot",
                          "journalcallbackmetadata", "selectedmetadata"}, "exact search metadata fields")
    require(meta["step"] == step, "search root cursor")
    result = verify_search(values, inputs, step=step, steps=200, horizon=horizon, block=3)
    require(meta["input_identities"] == result["input_identities"], "authenticated search innovations")
    replay = replay_candidates(nominal, values, step=step, noise_std=.05, deadline=deadline)
    roots = {"qpos": array_hash(q[None]), "qvel": array_hash(v[None]), "public_target": array_hash(packet[None, 4:6])}
    require(len(meta["journalcallbackmetadata"]) == 4, "four paid CEM callbacks")
    for index, row in enumerate(meta["journalcallbackmetadata"]):
        verify_bank_metadata(row, values["sequences"][64*index:64*(index+1)], roots, start=64*index)
    verify_bank_metadata(meta["selectedmetadata"], values["action"][None, None], roots, start=None)
    verify_planner_configuration(meta["planner_configuration"], model_sha=replay["model_binary_sha256"],
                                 steps=200, horizon=horizon, block=3, noise_std=.05)
    verify_fresh_bank(meta["bank_snapshot"], gain=gain, step=step, horizon=horizon, model_sha=replay["model_binary_sha256"])
    callback_seconds = sum(row["wall_seconds"] for row in meta["journalcallbackmetadata"])+meta["selectedmetadata"]["wall_seconds"]
    require(callback_seconds <= meta["bank_snapshot"]["costs"]["aggregate_adapter_lifetime"]["operation_wall_seconds"]+1e-8,
            "callbacks and selected advance paid inside search")
    return replay


def audit_cross(values, meta, *, nominal, root, gain, sequences, deadline=float("inf")):
    q, v, packet = validate_root(root, root, nominal)
    k = len(sequences)
    require(set(values) == {"sequences", "predicted_angles", "raw_rewards", "scores_12", "scores_24", "qpos", "qvel"},
            "complete cross-scoring arrays")
    equal(array(values["sequences"], (k, 24, 2), np.float32, "cross sequences"), sequences,
          "same deduplicated union under each gain")
    angles = array(values["predicted_angles"], (k, 24, 4), np.float32, "cross angles")
    position = array(values["qpos"], (k, 25, 4), np.float64, "cross qpos")
    velocity = array(values["qvel"], (k, 25, 4), np.float64, "cross qvel")
    rewards = array(values["raw_rewards"], (k, 24), np.float32, "cross rewards")
    accumulated = np.zeros(k, np.float32)
    for offset in range(24):
        accumulated += np.clip(rewards[:, offset], -2.5, 0.)
        if offset in (11, 23):
            equal(array(values[f"scores_{offset+1}"], (k,), np.float32, "cross prefix scores"), accumulated,
                  "exact sequential float32 prefix score")
    model = checked_model(nominal, gain); data = mujoco.MjData(model); maximum = 0.
    for candidate in range(k):
        budget(deadline); mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.time = q, v, root["step"]*.02
        mujoco.mj_forward(model, data)
        for offset in range(25):
            maximum = max(maximum, close(position[candidate, offset], data.qpos, "cross native qpos"),
                          close(velocity[candidate, offset], data.qvel, "cross native qvel"))
            if offset:
                equal(angles[candidate, offset-1], np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])].astype(np.float32),
                      "cross exact float32 native angles")
            if offset < 24:
                native_step(model, data, sequences[candidate, offset], deadline)
    error = close(rewards, geometry_reward(angles, packet[4:6], sequences, .05),
                  "independent cross geometry", geometry=True)
    require(set(meta) == {"configuration", "journalmetadata"}, "cross metadata fields")
    verify_planner_configuration(meta["configuration"], model_sha=model_hash(model), steps=200,
                                 horizon=24, block=3, noise_std=.05)
    roots = {"qpos": array_hash(q[None]), "qvel": array_hash(v[None]), "public_target": array_hash(packet[None, 4:6])}
    verify_bank_metadata(meta["journalmetadata"], sequences, roots, start=0)
    return {"native_transitions": k*24, "max_native_abs_error": maximum, "max_geometry_abs_error": error}


def audit_branches(folder, *, nominal, root, sequences, noise, deadline=float("inf")):
    """Full integration-state replay of fixed sequences with externally bound noise."""
    folder = Path(folder)
    receipt = json.loads((folder/"completed.json").read_text())
    authenticate_members(folder, receipt, {"started.json", "data.npz"}, deadline)
    receipt_sha = file_hash(folder/"completed.json", deadline)
    started = json.loads((folder/"started.json").read_text())
    expected_root = root["root"]
    require(set(started) == {"status", "configuration", "root", "root_sha256", "input_sha256", "automatic_retry"}
            and started["status"] == "started" and started["automatic_retry"] is False,
            "branch initial receipt")
    require(started["root"] == expected_root, "branch full root provenance")
    require(started["root_sha256"] == {key: array_hash(np.asarray(value, np.float64)) for key, value in expected_root.items()},
            "all branch original root typed hashes")
    k, h = sequences.shape[:2]; b = len(noise)
    require((b, h) == (4, 24), "fixed four full24 noise branches")
    noise = array(noise, (4, 24, 2), np.float64, "externally supplied branch noise")
    inputs = {"commands": array_hash(sequences), "noise": array_hash(noise)}
    require(started["input_sha256"] == receipt["input_sha256"] == inputs, "externally bound branch commands/noise")
    model = checked_model(nominal, float(root["true_gain"])); data = mujoco.MjData(model)
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    configuration = {"version": "reacher-common-root-branches-v1", "nominal_model_sha256": model_hash(nominal),
        "model_sha256": model_hash(model), "state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "state_size": state_size, "frame_skip": 2, "dt": .02, "branch_count": b, "candidate_count": k,
        "horizon": h, "true_gain": root["true_gain"], "reward_dist_weight": 1., "reward_control_weight": 1.,
        "root_semantics": "reset then complete integration restore then forward; verify qpos/qvel/time",
        "reward_semantics": "cached 3D xpos after two RK4 steps and RNE; no extra forward; applied effort once",
        "event_semantics": "constant gain/target; event-free task horizon eligibility is caller-bound",
        "rng_draws": 0, "new_model_calls": 0, "run_status_authority": "enclosing protocol and execution receipts"}
    require(started["configuration"] == receipt["configuration"] == configuration,
            "exact native branch model/gain/scoring configuration")
    require(set(receipt) == {"status", "version", "configuration", "counters", "input_sha256", "files",
                             "wall_seconds", "time_scope", "automatic_retry"}
            and receipt["version"] == configuration["version"] and receipt["automatic_retry"] is False
            and receipt["time_scope"] == "entry validation, setup, all native work/copies, data serialization and hashes; excludes final receipt write",
            "complete successful branch receipt")
    fields = {"commands", "noise", "applied", "qpos", "qvel", "integration_states", "time", "distance",
              "effort", "rewards", "initialized", "native_completed", "recorded", "substeps_completed"}
    values = read_npz(folder/"data.npz", fields)
    equal(array(values["commands"], (k, h, 2), np.float32, "branch commands"), sequences, "same union branch commands")
    equal(array(values["noise"], (b, h, 2), np.float64, "saved branch noise"), noise, "shared external branch noise")
    for name in ("qpos", "qvel"):
        array(values[name], (b, k, h+1, 4), np.float64, "branch " + name)
    array(values["integration_states"], (b, k, h+1, state_size), np.float64, "full branch integration states")
    array(values["time"], (b, k, h+1), np.float64, "branch times")
    array(values["applied"], (b, k, h, 2), np.float64, "branch applied actions")
    for name in ("distance", "effort", "rewards"):
        array(values[name], (b, k, h), np.float64, "branch native " + name)
    for name, shape in (("initialized", (b, k)), ("native_completed", (b, k, h)), ("recorded", (b, k, h))):
        require(array(values[name], shape, np.bool_, "branch mask " + name).all(), "all paid branches completed " + name)
    equal(array(values["substeps_completed"], (b, k, h), np.int64, "branch substeps"),
          np.full((b, k, h), 2, np.int64), "two substeps per branch transition")
    root_integration = np.array(expected_root["integration_state"], np.float64)
    maximum = 0.
    for branch in range(b):
        for candidate in range(k):
            budget(deadline); mujoco.mj_resetData(model, data)
            mujoco.mj_setState(model, data, root_integration, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, data)
            equal(data.qpos, expected_root["qpos"], "restored exact root position")
            equal(data.qvel, expected_root["qvel"], "restored exact root velocity")
            equal(data.time, expected_root["time"], "restored exact root time")
            for offset in range(h+1):
                actual, _ = _native_state(model, data)
                for name, native_name in (("qpos", "qpos"), ("qvel", "qvel"),
                                          ("integration_states", "integration_state"), ("time", "time")):
                    maximum = max(maximum, close(values[name][branch, candidate, offset], actual[native_name],
                                                 "independent full native branch " + name))
                if offset == h:
                    continue
                applied = np.clip(sequences[candidate, offset].astype(np.float64)+noise[branch, offset], -1., 1.)
                equal(values["applied"][branch, candidate, offset], applied, "issued command plus common noise then clipping")
                native_step(model, data, applied, deadline)
                _, reward_distance = _native_state(model, data)
                effort = float(np.square(applied).sum())
                for name, value in (("distance", -reward_distance), ("effort", effort), ("rewards", reward_distance-effort)):
                    maximum = max(maximum, close(values[name][branch, candidate, offset], value,
                                                 "cached native distance and once-only effort " + name))
    expected_counts = {"model_copies": 1, "data_allocations": 1, "roots_initialized": b*k,
                       "transitions_recorded": b*k*h}
    for prefix, amount in (("reset_calls", b*k), ("restore_calls", b*k), ("forward_calls", b*k),
                           ("native_transitions", b*k*h), ("native_substeps", 2*b*k*h),
                           ("postconstraint_calls", b*k*h)):
        expected_counts.update({prefix+"_"+suffix: amount for suffix in ("attempted", "completed")})
    require(set(receipt["counters"]) == set(expected_counts), "complete branch work counters")
    for name, expected in expected_counts.items():
        require(type(receipt["counters"][name]) is int and receipt["counters"][name] == expected,
                "native branch paid work " + name)
    finite_times(receipt, ("wall_seconds",))
    authenticate_members(folder, receipt, {"started.json", "data.npz"}, deadline)
    require(file_hash(folder/"completed.json", deadline) == receipt_sha, "branch completion unchanged during replay")
    return {"native_transitions": b*k*h, "native_substeps": 2*b*k*h, "max_native_abs_error": maximum,
            "receipt_sha256": receipt_sha, "wall_seconds": receipt["wall_seconds"], "counters": expected_counts}


def audit_slot(folder, *, nominal_model, inputs_a, inputs_b, expected_root, expected_noise,
               deadline=float("inf")):
    """Authenticate and replay one full slot. External provenance is mandatory.

    Caller supplies the authenticated source decision root and noise, verifies
    the original A-prefix lineage and no target/gain event during the interval,
    binds this auditor/runtime and enforces the outer audit cap. No native cost
    can select a winner here: union winners are derived only from saved scores.
    """
    began = time.perf_counter(); folder = Path(folder)
    read = lambda path: json.loads((folder/path).read_text())
    budget(deadline)
    root = read("root.json"); validate_root(root, expected_root, nominal_model)
    noise = array(expected_noise, (4, 24, 2), np.float64, "explicit authenticated common noise")
    noise_before = array_hash(noise)
    a, b = input_arrays(inputs_a, 8), input_arrays(inputs_b, 8)
    inputs_before = [{k: array_hash(v) for k, v in bank.items()} for bank in (a, b)]
    expected_members = {"root.json", "union.npz", "branches/started.json", "branches/data.npz", "branches/completed.json"}
    expected_members |= {f"searches/{g}--{s}/decisions/{root['step']:03d}.{ext}"
                         for g in ASSUMPTIONS for s in SEARCHES for ext in ("npz", "json")}
    expected_members |= {f"cross/{g}.{ext}" for g in ASSUMPTIONS for ext in ("npz", "json")}
    receipt = read("completed.json")
    require(set(receipt) == {"status", "files", "wall_seconds", "work"}, "exact slot completion fields")
    authenticate_members(folder, receipt, expected_members, deadline)
    receipt_sha = file_hash(folder/"completed.json", deadline)
    searches = {}; search_reports = {}; search_paid = 0.
    for assumption in ASSUMPTIONS:
        gain = 1. if assumption == "nominal" else float(root["true_gain"])
        for name in SEARCHES:
            label = assumption+"--"+name
            stem = f"searches/{label}/decisions/{root['step']:03d}"
            values = read_npz(folder/(stem+".npz"), COMMON | PLANNED); meta = read(stem+".json")
            require(meta["arm"] == assumption, "search dynamics assumption label")
            horizon = 24 if name == "long_a" else 12
            source = b if name == "short_b" else a
            innovations = {key: value[:, :, :horizon//3].copy() for key, value in source.items()}
            search_reports[label] = audit_search(values, meta, nominal=nominal_model, root=root, gain=gain,
                                                 horizon=horizon, inputs=innovations, deadline=deadline)
            searches[label] = values
            costs = meta["bank_snapshot"]["costs"]
            search_paid += costs["setup_wall_seconds"]+costs["plan_call_wall_seconds"]
        equal(searches[assumption+"--short_a"]["sequences"][:64],
              searches[assumption+"--long_a"]["sequences"][:64, :12], "shared A initial prefix")
        equal(searches[assumption+"--short_a"]["predicted_angles"][:64],
              searches[assumption+"--long_a"]["predicted_angles"][:64, :12], "same-root native initial prefix")
    union = verify_union(read_npz(folder/"union.npz", {"slots", "unique", "slot_to_unique", "first_slot"}), searches)
    cross_reports = {}; cross_paid = 0.
    for assumption in ASSUMPTIONS:
        gain = 1. if assumption == "nominal" else float(root["true_gain"])
        values = read_npz(folder/f"cross/{assumption}.npz",
                          {"sequences", "predicted_angles", "raw_rewards", "scores_12", "scores_24", "qpos", "qvel"})
        meta = read(f"cross/{assumption}.json")
        cross_reports[assumption] = audit_cross(values, meta, nominal=nominal_model, root=root,
            gain=gain, sequences=union["unique"], deadline=deadline)
        cross_paid += meta["journalmetadata"]["wall_seconds"]
    branches = audit_branches(folder/"branches", nominal=nominal_model, root=root,
                               sequences=union["unique"], noise=noise, deadline=deadline)
    k = len(union["unique"])
    work = {"search_candidate": 24576, "selected_advance": 6, "cross_score": 2*k*24, "native_branch": 4*k*24}
    require(receipt["work"] == work and all(type(value) is int for value in receipt["work"].values()),
            "complete nominal and noisy native work, without nested A double-counting")
    finite_times(receipt, ("wall_seconds",))
    require(search_paid+cross_paid+branches["wall_seconds"] <= receipt["wall_seconds"]+1e-8,
            "disjoint search, cross-bank, branch work inside inclusive slot wall")
    require([{key: array_hash(value) for key, value in bank.items()} for bank in (a, b)] == inputs_before,
            "caller innovation arrays unchanged")
    require(array_hash(expected_noise) == noise_before, "caller noise unchanged")
    authenticate_members(folder, receipt, expected_members, deadline)
    require(file_hash(folder/"completed.json", deadline) == receipt_sha, "slot completion unchanged during audit")
    maximum_native = max([branches["max_native_abs_error"]]+
        [x["max_native_abs_error"] for x in search_reports.values()]+[x["max_native_abs_error"] for x in cross_reports.values()])
    maximum_geometry = max([x["max_geometry_abs_error"] for x in search_reports.values()]+
                            [x["max_geometry_abs_error"] for x in cross_reports.values()])
    return {"status": "completed", "version": VERSION, "source_role": root["source_role"], "step": root["step"],
        "slot_completed_sha256": receipt_sha, "root_sha256": file_hash(folder/"root.json", deadline),
        "identity_slots": 70, "unique_sequences": k, "work": work,
        "native_transitions_checked": sum(work.values()), "native_substeps_checked": 2*sum(work.values()),
        "searches": search_reports, "cross": cross_reports, "branches": branches,
        "max_native_abs_error": maximum_native, "max_geometry_abs_error": maximum_geometry,
        "recorded_slot_wall_seconds": receipt["wall_seconds"], "audit_wall_seconds": time.perf_counter()-began,
        "new_planner_calls": 0, "new_model_calls": 0, "new_rng_draws": 0,
        "limits": ["Exposed common native roots are privileged, not deployable observer comparisons.",
                   "Nominal reset-state scoring differs from full-state noisy native branch continuation.",
                   "External source provenance, A-prefix pairing, event-free interval and RNG identities remain caller-bound.",
                   "Saved score arithmetic and fixed native replay do not establish optimal search or closed-loop improvement."]}
