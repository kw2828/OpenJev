"""Independent saved-output arithmetic/native replay for tracking engineering.

No controller, search, identifier, learned model, optimizer or RNG is invoked.
The caller authenticates the experiment/source/runtime and explicit input roles;
this module authenticates a completed row and its compact numerical evidence.
Native replay is paid audit work, not a new policy evaluation. Failed executions
are preserved by the writer and are not accepted as completed rows here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

VERSION = "reacher-tracking-saved-audit-v1"
ARMS = ("nominal", "adaptive", "frozen", "public_gain", "true_state", "zero")
INPUT_NAMES = ("initial", "random_extra", "cem/1", "cem/2", "cem/3")
COMMON = {"root_qpos", "root_qvel", "public_packet", "public_qpos", "public_qvel", "planning_gain", "action"}
PLANNED = {"sequences", "scores", "selected_id", "predicted_angles", "raw_rewards", "selected_qpos",
           "selected_qvel", "selected_angles", "selected_reward"}


def require(condition, label):
    if not condition:
        raise ValueError(label)


def budget(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError("tracking saved-output audit deadline")


def array(value, shape, dtype, label, *, json_value=False):
    result = np.asarray(value, dtype=dtype if json_value else None)
    require(result.shape == shape and result.dtype == dtype and np.isfinite(result).all(), label)
    return result


def equal(actual, expected, label):
    require(np.array_equal(actual, expected), label)


def close(actual, expected, label, *, geometry=False):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label)
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.allclose(a, b, rtol=2e-6 if geometry else 0., atol=5e-7 if geometry else 1e-10), label)
    return error


def array_hash(value):
    head = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(head + b"\n" + value.tobytes(order="C")).hexdigest()


def file_hash(path, deadline=float("inf")):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024*1024):
            budget(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def model_hash(model):
    payload = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, buffer=payload)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def checked_model(nominal, gain):
    require(isinstance(nominal, mujoco.MjModel) and (nominal.nq, nominal.nv, nominal.nu) == (4, 4, 2),
            "native Reacher nominal model")
    gear = np.zeros((2, 6)); gear[:, 0] = 200.
    equal(nominal.actuator_gear, gear, "independent nominal gear200")
    equal(nominal.dof_damping, [1., 1., 0., 0.], "unchanged damping")
    require(nominal.opt.timestep == .01 and int(nominal.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4),
            "pinned RK4 timestep")
    require(math.isfinite(gain) and gain > 0 and math.isfinite(200*gain), "finite positive gain")
    model = copy.copy(nominal)
    model.actuator_gear[:] = gear*gain
    return model


def input_arrays(values, chunks):
    """Validate a caller-authenticated mapping, without constructing SearchInputs."""
    require(set(values) == set(INPUT_NAMES), "innovation fieldset")
    return {key: array(values[key], (1, k, chunks, 2), np.float64, "innovations " + key)
            for key, k in zip(INPUT_NAMES, (64, 192, 64, 64, 63), strict=True)}


def verify_search(values, inputs, *, step, steps, horizon, block):
    """Reconstruct stored CEM proposals/selection with independent NumPy math."""
    require(type(step) is int and type(steps) is int and 0 <= step < steps
            and type(horizon) is int and type(block) is int and 1 <= block <= horizon <= steps,
            "search horizon/terminal boundary")
    h = min(horizon, steps-step); c = (h+block-1)//block
    inputs = input_arrays(inputs, (horizon+block-1)//block)
    sequences = array(values["sequences"], (256, h, 2), np.float32, "all256 sequences")
    require((np.abs(sequences) <= 1).all(), "normalized commands")
    scores = array(values["scores"], (256,), np.float64, "all256 scores")
    rewards = array(values["raw_rewards"], (256, h), np.float32, "all candidate offset rewards")
    summed = np.zeros(256, np.float32)
    for t in range(h):
        summed += np.clip(rewards[:, t], -2.5, 0.)
    equal(scores, summed.astype(np.float64), "exact sequential clipped float32 scores")
    proposal = inputs["initial"][0, :, :c]*np.r_[np.full(32, .25), np.full(32, .75)][:, None, None]
    for i, anchor in enumerate(((0., 0.), (.1, 0.), (-.1, 0.), (0., .1), (0., -.1), (.2, .2), (-.2, -.2))):
        proposal[i] = anchor
    previous = None
    for generation in range(4):
        if generation:
            elite_ids = np.argsort(-scores[(generation-1)*64:generation*64], kind="stable")[:8]
            elites = previous[elite_ids].astype(np.float64)
            mean = elites.mean(0); std = np.maximum(elites.std(0, ddof=0), .001)
            with np.errstate(over="ignore", invalid="ignore"):
                proposal = mean[None] + std[None]*inputs[f"cem/{generation}"][0, :, :c]
            if generation == 3:
                proposal = np.concatenate((proposal, mean[None]))
        previous = np.clip(proposal, -1., 1.).astype(np.float32)
        equal(sequences[generation*64:(generation+1)*64], np.repeat(previous, block, axis=1)[:, :h],
              "CEM scored quantization/elites/paid mean")
    selected = int(array(values["selected_id"], (), np.int64, "selected int64 scalar"))
    require(selected == int(np.argmax(scores)), "earliest global argmax")
    equal(values["action"], sequences[selected, 0], "issued selected command")
    return {"horizon": h, "selected_id": selected,
            "input_identities": {key: array_hash(value) for key, value in inputs.items()}}


def geometry_reward(angles, target, commands, sigma):
    """Independent float32 FK and analytic clipped-normal second moment."""
    require(type(sigma) in (int, float) and math.isfinite(sigma) and 0 <= sigma <= 1,
            "supported declared actuator noise scale [0,1]")
    require(angles.dtype == commands.dtype == np.float32 and angles.shape[:-1] == commands.shape[:-1]
            and angles.shape[-1] == 4 and commands.shape[-1] == 2, "geometry float32 shapes")
    require(np.isfinite(angles).all() and np.isfinite(commands).all()
            and (np.abs(commands) <= 1).all(), "finite geometry inputs")
    q = np.arctan2(angles[..., 2:], angles[..., :2]); q01 = q.sum(-1)
    tip = np.stack((np.float32(.10)*np.cos(q[..., 0])+np.float32(.11)*np.cos(q01),
                    np.float32(.10)*np.sin(q[..., 0])+np.float32(.11)*np.sin(q01)), -1)
    distance = np.sqrt(np.sum((tip-target)**2, axis=-1))
    u = commands.astype(np.float64)
    if sigma < math.sqrt(np.finfo(np.float64).tiny):
        moment = u*u
    else:
        lo = np.clip((-1-u)/sigma, -38, 38); hi = np.clip((1-u)/sigma, -38, 38)
        erf = np.vectorize(math.erf, otypes=[np.float64])
        mass = .5*(erf(hi/math.sqrt(2))-erf(lo/math.sqrt(2)))
        phi_lo = np.exp(-.5*lo**2)/math.sqrt(2*math.pi)
        phi_hi = np.exp(-.5*hi**2)/math.sqrt(2*math.pi)
        moment = ((u*u+sigma*sigma)*mass + 2*u*sigma*(phi_lo-phi_hi)
                  + sigma*sigma*(lo*phi_lo-hi*phi_hi) + 1-mass)
    return -distance-np.clip(moment, 0, 1).sum(-1).astype(np.float32)


def native_step(model, data, command, deadline):
    data.ctrl[:] = command
    for _ in range(2):
        budget(deadline); mujoco.mj_step(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "finite native replay")


def replay_candidates(nominal, values, *, step, noise_std, deadline=float("inf")):
    """Replay fixed saved sequences; no scoring callback or online selection."""
    gain = float(array(values["planning_gain"], (), np.float64, "planning gain"))
    model = checked_model(nominal, gain); data = mujoco.MjData(model)
    root_q = array(values["root_qpos"], (4,), np.float64, "candidate root qpos")
    root_v = array(values["root_qvel"], (4,), np.float64, "candidate root qvel")
    packet = array(values["public_packet"], (8,), np.float32, "root packet")
    equal(root_q[2:].astype(np.float32), packet[4:6], "current public target")
    equal(root_v[2:], [0., 0.], "target velocity zero")
    bank = values["sequences"]; k, h, _ = bank.shape
    saved_angles = array(values["predicted_angles"], (k, h, 4), np.float32, "saved candidate angles")
    first_q = first_v = None
    for candidate in range(k):
        budget(deadline); mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:], data.time = root_q, root_v, step*.02
        mujoco.mj_forward(model, data)
        for offset in range(h):
            native_step(model, data, bank[candidate, offset], deadline)
            angles = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])].astype(np.float32)
            equal(angles, saved_angles[candidate, offset], "exact independently replayed candidate angles")
            if candidate == int(values["selected_id"]) and offset == 0:
                first_q, first_v = data.qpos.copy(), data.qvel.copy()
    reward = geometry_reward(saved_angles, packet[4:6], bank, noise_std)
    max_reward = close(values["raw_rewards"], reward, "independent candidate geometry", geometry=True)
    selected_q = array(values["selected_qpos"], (2, 4), np.float64, "selected qpos")
    selected_v = array(values["selected_qvel"], (2, 4), np.float64, "selected qvel")
    equal(selected_q[0], root_q, "selected restarted qpos")
    equal(selected_v[0], root_v, "selected restarted qvel")
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:], data.time = root_q, root_v, step*.02
    mujoco.mj_forward(model, data)
    native_step(model, data, values["action"], deadline)
    maximum = max(close(selected_q[1], data.qpos, "selected native qpos"),
                  close(selected_v[1], data.qvel, "selected native qvel"))
    equal(selected_q[1], first_q, "chosen candidate first qpos identity")
    equal(selected_v[1], first_v, "chosen candidate first qvel identity")
    angles = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])].astype(np.float32)
    equal(array(values["selected_angles"], (4,), np.float32, "selected angles"), angles,
          "selected angle identity")
    selected_reward = array(values["selected_reward"], (), np.float32, "selected reward")
    max_reward = max(max_reward, close(selected_reward, geometry_reward(angles, packet[4:6], values["action"],
                                                                       noise_std), "selected geometry", geometry=True))
    return {"candidate_transitions": k*h, "selected_transitions": 1, "native_substeps": 2*(k*h+1),
            "max_native_abs_error": maximum, "max_geometry_abs_error": max_reward,
            "model_binary_sha256": model_hash(model)}


def public_roots(packets, dt=.02):
    packets = array(packets, (len(packets), 8), np.float32, "public sequence")
    require((packets[:, 6] == 1).all() and (packets[:, 7] == 0).all(), "full visible zero-age qualification")
    require(np.allclose(packets[:, :2]**2+packets[:, 2:4]**2, 1., atol=1e-6), "unit-circle observations")
    measured = np.arctan2(packets[:, 2:4].astype(np.float64), packets[:, :2].astype(np.float64))
    q = np.empty((len(packets), 4)); v = np.zeros_like(q); q[0, :2] = measured[0]
    q[:, 2:] = packets[:, 4:6]
    previous = None
    for t in range(1, len(packets)):
        delta = (measured[t]-measured[t-1]+np.pi) % (2*np.pi)-np.pi
        q[t, :2] = q[t-1, :2]+delta
        v[t, :2] = delta/dt if previous is None else (3*delta-previous)/(2*dt)
        previous = delta
    return q, v


def replay_identification(nominal, trace, *, root_qpos, root_qvel, command, next_packet,
                          gains, retained, prior_gain, window, step, deadline=float("inf")):
    """Validate the real update before endpoint assimilation; retained is caller-owned."""
    require(trace["step_index"] == step, "identifier completed-transition cursor")
    for key, expected in (("root_qpos", root_qpos), ("root_qvel", root_qvel),
                          ("issued_command", command), ("next_packet", next_packet)):
        equal(trace[key], expected, "identifier " + key)
    predictions = array(trace["candidate_predictions"], (len(gains), 4), np.float64,
                        "identifier predictions", json_value=True)
    maximum = 0.
    for i, gain in enumerate(gains):
        budget(deadline); model = checked_model(nominal, float(gain)); data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = root_qpos, root_qvel
        mujoco.mj_forward(model, data)
        native_step(model, data, command, deadline)
        expected = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2])]
        maximum = max(maximum, close(predictions[i], expected, "identifier native prediction"))
    residual = np.mean((predictions-next_packet[:4])**2, axis=1)
    retained = [*retained, residual][-window:]
    summed = np.sum(np.stack(retained), axis=0, dtype=np.float64)
    equal(trace["candidate_residuals"], residual, "identifier residual arithmetic")
    equal(trace["window_residual_sums"], summed, "identifier rolling residual arithmetic")
    flat = bool(np.all(summed == summed[0]))
    gain = prior_gain if flat else float(gains[int(np.argmin(summed))])
    require(trace["retained_transitions"] == len(retained) and trace["exactly_flat_bank"] is flat
            and trace["gain_before"] == prior_gain and trace["gain_after"] == gain
            and trace["residual_spread"] == float(np.max(summed)-np.min(summed)), "identifier selected gain/flat rule")
    return gain, retained, {"transitions": len(gains), "max_abs_error": maximum}


def _native_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec)
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    delta = data.xpos[finger]-data.xpos[target]
    raw = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2]), data.qpos[2:], data.qvel[:2], delta[:2]]
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "raw_obs": raw,
            "integration_state": state, "time": float(data.time)}, -float(np.linalg.norm(delta))


def audit_episode(record, nominal, *, deadline=float("inf")):
    """Continuous native replay, preserving reward-before-target-event ordering.

    Initial state/seed allocation is caller-authenticated; no reset RNG is replayed.
    Native cached body positions are used at the same RK4 point as Gymnasium.
    """
    meta = record["metadata"]; t_count = meta["horizon"]
    require(type(t_count) is int and t_count > 0 and meta["status"] == "completed"
            and meta["completed_decisions"] == meta["attempted_transitions"] == meta["returned_native_transitions"]
            == t_count and meta["failure"] is None, "complete native episode")
    require(meta["dt"] == .02 and meta["frame_skip"] == 2 and meta["base_gear"] == 200.
            and meta["reward_dist_weight"] == meta["reward_control_weight"] == 1., "native identity")
    packets = array(record["policy"]["packets"], (t_count+1, 8), np.float32, "packets", json_value=True)
    commands = array(record["policy"]["commands"], (t_count, 2), np.float32, "commands", json_value=True)
    require((np.abs(commands) <= 1).all(), "native normalized commands")
    targets = array(meta["target_path"], (t_count+1, 2), np.float64, "target schedule", json_value=True)
    gains = array(meta["gear_multiplier"], (t_count,), np.float64, "gain schedule", json_value=True)
    noise = array(meta["noise"], (t_count, 2), np.float64, "noise schedule", json_value=True)
    require(np.all(np.abs(targets) <= .27) and np.all(gains > 0), "native schedule domain")
    equal(meta["sensor_schedule"], np.ones(t_count+1, bool), "full sensing qualification")
    states = record["audit"]["decision_states"]; transitions = record["audit"]["transitions"]
    require(len(states) == t_count+1 and len(transitions) == t_count, "native state/transition coverage")
    model = checked_model(nominal, 1.); data = mujoco.MjData(model)
    initial = array(states[0]["integration_state"], (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),),
                    np.float64, "initial integration state", json_value=True)
    mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    maximum = 0.; rewards = []
    def verify_state(saved):
        actual, _ = _native_state(model, data)
        require(set(saved) == set(actual), "native state fields")
        return max(close(saved[key], value, "native state " + key) for key, value in actual.items())
    for step in range(t_count+1):
        budget(deadline); maximum = max(maximum, verify_state(states[step]))
        close(data.time, step*.02, "continuous decision time")
        equal(data.qpos[2:], targets[step], "current target qpos")
        equal(data.qvel[2:], [0., 0.], "target velocity")
        packet = np.r_[np.cos(data.qpos[:2]), np.sin(data.qpos[:2]), targets[step], 1., 0.].astype(np.float32)
        equal(packets[step], packet, "post-event public packet")
        if step == t_count:
            break
        row = transitions[step]; applied = np.clip(commands[step].astype(np.float64)+noise[step], -1., 1.)
        require(row["step"] == step and row["status"] == "completed" and row["native_returned"] is True
                and row["gear_multiplier"] == gains[step] and row["actuator_gear"] == 200*gains[step],
                "actual transition/gain identity")
        for key, value in (("command", commands[step]), ("actuator_noise", noise[step]), ("applied_action", applied),
                           ("reward_target", targets[step]), ("next_target", targets[step+1])):
            equal(row[key], value, "transition " + key)
        changed = not np.array_equal(targets[step], targets[step+1])
        require(row["target_changed"] is changed and row["terminated"] is False
                and row["truncated"] is (step+1 == t_count), "target event/terminal flags")
        model.actuator_gear[:, 0] = 200*gains[step]
        native_step(model, data, applied, deadline)
        maximum = max(maximum, verify_state(row["native_after"]))
        _, dist = _native_state(model, data); ctrl = -float(np.square(applied).sum())
        for key, value in (("reward_dist", dist), ("reward_ctrl", ctrl), ("reward", dist+ctrl)):
            maximum = max(maximum, close(row[key], value, "pre-target-event " + key))
        rewards.append(float(row["reward"]))
        if changed:
            data.qpos[2:], data.qvel[2:] = targets[step+1], 0.
            mujoco.mj_forward(model, data)
    return {"transitions": t_count, "substeps": 2*t_count, "max_abs_error": maximum,
            "native_rewards": rewards, "mean_cost": -float(np.sum(rewards))}


COUNT_NAMES = tuple(f"{prefix}_{suffix}" for prefix in (
    "native_transitions", "native_substeps", "reset_calls", "forward_calls", "postconstraint_calls",
    "geometry_calls", "geometry_samples") for suffix in ("attempted", "completed")) + (
    "candidate_sequences_requested", "candidate_sequences_initialized", "candidate_sequences_native_completed",
    "candidate_sequences_scored")


def bank_counts(k, h):
    result = {}
    for key in COUNT_NAMES:
        if key.startswith(("candidate_sequences_", "reset_calls_", "forward_calls_")):
            result[key] = k
        elif key.startswith("native_substeps_"):
            result[key] = 2*k*h
        elif key.startswith("geometry_calls_"):
            result[key] = 1
        else:
            result[key] = k*h
    return result


def finite_times(mapping, names):
    for name in names:
        value = mapping[name]
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "finite paid time " + name)


def verify_bank_metadata(meta, commands, root_hashes, *, start):
    k, h = commands.shape[:2]
    for key, expected in bank_counts(k, h).items():
        require(type(meta[key]) is int and meta[key] == expected, "completed bank work " + key)
    require(meta["status"] == "completed" and meta["candidate_start"] == start
            and meta["bank_shape"] == [1, k, h, 2] and meta["sum_offsets_completed"] == h,
            "complete bank boundary")
    require(meta["purpose"] == ("selected_root_advance" if start is None else "candidate_scoring"),
            "candidate/selected accounting separation")
    require(meta["bank_sha256"] == array_hash(commands[None]) and meta["root_sha256"] == root_hashes,
            "bank/root hash identity")
    equal(meta["cursor"], {"case": 0, "candidate": k-1, "offset": h-1}, "final completed bank cursor")
    finite_times(meta, ("native_seconds", "geometry_seconds", "wall_seconds"))
    require(meta["native_seconds"]+meta["geometry_seconds"] <= meta["wall_seconds"]+1e-8,
            "bank component times nested in wall")


def verify_planner_configuration(config, *, model_sha, steps, horizon, block, noise_std):
    expected = {"version": "reacher-nominal-geometry-cem-v1", "model_binary_sha256": model_sha,
                "model_copied": True, "native_timestep": .01, "frame_skip": 2, "decision_dt": .02,
                "steps": steps, "planning_horizon": horizon, "action_block": block, "planner": "cem256",
                "callback_sizes": [64]*4, "warm_start": False, "reward_clip": [-2.5, 0.],
                "score_sum": "sequential float32, per-step clipping", "rng_draws": 0}
    for key, value in expected.items():
        require(config[key] == value, "planner config " + key)
    geom = config["geometry"]
    for key, value in {"noise_std": noise_std, "distance_weight": 1., "control_weight": 1.,
                       "link_offsets_meters": [.10, .11], "native_reward_exact": False,
                       "reward_clipping": None, "joint_limit_penalty": None}.items():
        require(geom[key] == value, "geometry config " + key)


def verify_snapshot(snapshot, *, nominal, gains, gain, step, cumulative, selected_model_sha,
                    steps, horizon, block, noise_std):
    full = "configuration" in snapshot
    if full:
        require(snapshot["failed"] is False, "completed bank not failed")
        cfg = snapshot["configuration"]
        equal(cfg["allowed_gains"], gains, "global allowed gain bank")
        require(cfg["nominal_model_binary_sha256"] == model_hash(nominal), "nominal binary identity")
        require(cfg["steps"] == steps and cfg["planning_horizon"] == horizon and cfg["action_block"] == block
                and cfg["frame_skip"] == 2 and cfg["noise_std"] == noise_std
                and cfg["cases_per_call"] == 1 and cfg["rng_draws"] == cfg["estimator_updates"] == 0,
                "gain bank configuration")
        require(len(snapshot["members"]) == len(gains), "all eagerly constructed gain members")
        sha = []
        for i, (member, g) in enumerate(zip(snapshot["members"], gains, strict=True)):
            require(member["gain"] == g and member["failed"] is False, "gain member identity")
            member_sha = model_hash(checked_model(nominal, float(g))); sha.append(member_sha)
            verify_planner_configuration(member["configuration"], model_sha=member_sha, steps=steps,
                                         horizon=horizon, block=block, noise_std=noise_std)
            for key in COUNT_NAMES:
                require(type(member["lifetime"][key]) is int and member["lifetime"][key] == cumulative[i][key],
                        "per-gain cumulative " + key)
            require(member["lifetime"]["operations_completed"] == cumulative[i]["operations_completed"]
                    and member["lifetime"]["operations_failed"] == 0, "per-gain operation counts")
            finite_times(member, ("setup_seconds",))
            finite_times(member["lifetime"], ("operation_wall_seconds",))
        equal(cfg["member_model_binary_sha256"], sha, "all eager model fingerprints")
    last = snapshot["last_operation"]; index = int(np.flatnonzero(gains == gain)[0])
    require(last["gain"] == gain and last["gain_index"] == index and last["step"] == step
            and last["status"] == "completed" and last["model_binary_sha256"] == selected_model_sha,
            "selected gain/member/step")
    costs = snapshot["costs"]
    for key in ("temporary_gain_model_copies", "retained_adapter_model_copies", "native_data_instances"):
        require(costs[key] == len(gains), "paid eager setup " + key)
    require(costs["plan_calls"] == costs["plan_calls_completed"] == step+1
            and costs["input_rejections"] == costs["plan_calls_failed"] == 0, "one successful plan per real decision")
    for key in (*COUNT_NAMES, "operations_completed"):
        require(costs["aggregate_adapter_lifetime"][key] == sum(row[key] for row in cumulative),
                "aggregate lifetime " + key)
    require(costs["aggregate_adapter_lifetime"]["operations_failed"] == 0, "no hidden failed plan")
    finite_times(costs, ("setup_wall_seconds", "plan_call_wall_seconds", "nested_adapter_setup_seconds"))
    if full:
        close(costs["nested_adapter_setup_seconds"], sum(m["setup_seconds"] for m in snapshot["members"]),
              "nested eager setup sum")
        close(costs["aggregate_adapter_lifetime"]["operation_wall_seconds"],
              sum(m["lifetime"]["operation_wall_seconds"] for m in snapshot["members"]), "nested plan wall sum")
    require(costs["nested_adapter_setup_seconds"] <= costs["setup_wall_seconds"]+1e-8
            and costs["aggregate_adapter_lifetime"]["operation_wall_seconds"] <= costs["plan_call_wall_seconds"]+1e-8,
            "nested adapter setup/plan times")
    return costs


def load_inputs(stem, horizon, block):
    stem = Path(stem)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
        require(set(saved.files) == {name.replace("/", "_") for name in INPUT_NAMES}, "input NPZ members")
        values = {name: saved[name.replace("/", "_")] for name in INPUT_NAMES}
    values = input_arrays(values, (horizon+block-1)//block)
    meta = json.loads(stem.with_suffix(".json").read_text())
    require(meta["input_identities"] == {name: array_hash(value) for name, value in values.items()},
            "innovation sidecar identity")
    return values


def authenticate_members(folder, completed, expected, deadline):
    require(completed["status"] == "completed" and isinstance(completed["files"], dict), "completed row marker")
    paths = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
    require(set(paths) == expected | {"completed.json"} and set(completed["files"]) == expected,
            "exact completed row membership")
    for name in expected:
        path = paths[name]
        require(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()), "contained ordinary payload")
        entry = completed["files"][name]
        # Enclosing writer uses digest strings. A richer receipt must be versioned explicitly.
        require(type(entry) is str and file_hash(path, deadline) == entry, "row file hash " + name)


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
    require(arm in ARMS and type(steps) is int and steps > 0 and len(inputs_by_step) == steps,
            "complete row role/horizon/inputs")
    require(type(cfg["window"]) is int and cfg["window"] > 0 and type(cfg["freeze_after"]) is int
            and 1 <= cfg["freeze_after"] <= steps, "identifier window/freeze boundary")
    gains = np.asarray(cfg["gain_grid"], np.float64)
    require(gains.ndim == 1 and len(gains) >= 2 and np.isfinite(gains).all() and np.all(gains > 0)
            and np.all(np.diff(gains) > 0) and np.any(gains == 1.), "explicit global gain grid")
    expected = {"started.json", "initial-controller.json", "episode.json", "controller.json"}
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
    for step in range(steps):
        budget(deadline)
        with np.load(folder/f"decisions/{step:03d}.npz", allow_pickle=False) as saved:
            require(set(saved.files) == COMMON | (set() if arm == "zero" else PLANNED), "exact decision NPZ fields")
            values = {key: saved[key] for key in saved.files}
        meta = read(f"decisions/{step:03d}.json")
        require(set(meta) == {"step", "arm", "input_identities", "planner_configuration", "bank_snapshot",
                              "journalcallbackmetadata", "selectedmetadata"}, "exact decision metadata fields")
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
        require(meta["input_identities"] == {key: array_hash(value) for key, value in inputs.items()},
                "every row innovation identity")
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
                             "decision_serialization_seconds", "wall_seconds"))
    require(sum(completed[key] for key in ("setup_seconds", "decision_seconds", "native_seconds",
                "observation_seconds", "decision_serialization_seconds")) <= completed["wall_seconds"]+1e-8,
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
    return {"version": VERSION, "arm": arm, "status": "completed", "row_completed_sha256": completion_sha,
            "native_control": native, "candidate_transitions_checked": candidate_steps,
            "selected_transitions_checked": selected_steps, "identifier_transitions_checked": identifier_steps,
            "max_candidate_state_abs_error": max_native, "max_geometry_abs_error": max_geometry,
            "max_identifier_abs_error": max_identifier, "controller_costs": costs,
            "row_wall_seconds": completed["wall_seconds"], "audit_wall_seconds": time.perf_counter()-started_at,
            "new_model_calls": 0, "new_planner_calls": 0, "new_rng_draws": 0,
            "scope": "Initial seed/schedule/source/runtime authenticity belongs to caller; saved arithmetic and native replay only."}
