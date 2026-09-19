"""Explicit common candidate union and full-state noisy native branches.

No search, RNG, policy, or learned model runs here. The enclosing protocol owns
root provenance and guarantees that gain and goal stay constant over the branch.
Restoration deliberately differs from a nominal planner's qpos/qvel reset.
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

from openjev.research.reacher_physics_control import _model

VERSION = "reacher-common-root-branches-v1"
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frozen(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _commands(value, shape, label):
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype == np.float32 and np.isfinite(value).all()
             and (np.abs(value) <= 1).all(), f"{label}: finite normalized float32{shape} required")
    return value.copy()


def build_union(initial64, selected):
    """Preserve 70 slots, extend short winners, and deduplicate first occurrences.

    Signed zeros are canonicalized to +0 in every slot before byte comparison.
    For finite float32 this makes byte equality agree with np.array_equal.
    Returned arrays are detached, bytes-backed and cannot be made writable.
    The caller owns the mutable containing dictionary.
    """
    initial = _commands(initial64, (64, 24, 2), "initial64")
    _require(isinstance(selected, (list, tuple)) and len(selected) == 6,
             "Exactly six ordered selected arrays required")
    winners = []
    for index, value in enumerate(selected):
        _require(isinstance(value, np.ndarray) and value.shape in ((12, 2), (24, 2)),
                 f"selected[{index}] must have 12 or 24 actions")
        value = _commands(value, value.shape, f"selected[{index}]")
        if len(value) == 12:
            value = np.concatenate((value, np.repeat(value[-1:], 12, axis=0)), axis=0)
        winners.append(value)
    slots = np.concatenate((initial, np.stack(winners)), axis=0)
    slots[slots == 0] = np.float32(0.)
    lookup, first, indices = {}, [], []
    for index, sequence in enumerate(slots):
        key = sequence.tobytes(order="C")
        if key not in lookup:
            lookup[key] = len(first)
            first.append(index)
        indices.append(lookup[key])
    return {"slots": _frozen(slots), "unique": _frozen(slots[first]),
            "slot_to_unique": _frozen(np.asarray(indices, np.int64)),
            "first_slot": _frozen(np.asarray(first, np.int64))}


def _deadline(deadline):
    _require(type(deadline) in (int, float) and not math.isnan(deadline), "Explicit monotonic deadline required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Common-root branch deadline exceeded")


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    return value


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _save(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _hash_array(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()


def _file_hash(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _deadline(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _root_array(value, shape, label):
    raw = np.asarray(value)
    _require(raw.shape == shape and raw.dtype.kind in "iuf" and np.isfinite(raw).all(), label)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)
    _require(np.isfinite(result).all(), label)
    return result


def _root(root, size):
    required = {"qpos", "qvel", "integration_state", "time"}
    _require(isinstance(root, dict) and required <= set(root) <= required | {"raw_obs"},
             "Explicit saved native decision-state dictionary required")
    result = {name: _root_array(root[name], shape, "root " + name) for name, shape in
              (("qpos", (4,)), ("qvel", (4,)), ("integration_state", (size,)))}
    result["time"] = float(_root_array(root["time"], (), "root time"))
    _require(result["time"] >= 0 and (result["qvel"][2:] == 0).all()
             and (np.abs(result["qpos"][2:]) <= .27).all(), "Nonnegative time and static native target required")
    if "raw_obs" in root:
        result["raw_obs"] = _root_array(root["raw_obs"], (10,), "root raw_obs")
    return result


def _state(model, data):
    integration = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), np.float64)
    mujoco.mj_getState(model, data, integration, STATE_SPEC)
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "integration_states": integration, "time": np.asarray(data.time, np.float64)}


def _record(arrays, model, data, b, k, t):
    values = _state(model, data)
    _require(all(np.isfinite(value).all() for value in values.values()), "Nonfinite native branch state")
    for name, value in values.items():
        arrays[name][b, k, t] = value


def run_branches(nominal_model, *, root, true_gain, sequences, noise, out, deadline):
    """Execute explicit branches from one complete native state, once per folder.

    Shapes: commands[K,H,2]float32, noise[B,H,2]float64. Every (branch,candidate)
    resets private MjData, restores mjSTATE_INTEGRATION, calls mj_forward, and
    verifies qpos/qvel/time exactly. Saved initial states are *after* forward;
    started.json retains the original integration state. Optional raw_obs is
    retained as provenance, not compared across that cache refresh.

    Each transition sets clip(float64(command)+noise), runs two mj_step calls,
    then mj_rnePostConstraint. Reward uses the cached 3D body positions at that
    RK4 point without another forward, and charges applied squared control once.
    No TimeLimit or event schedule is synthesized. The caller establishes that
    the entire interval has fixed target/gain and is within its task horizon.

    Pure malformed-input/expired-entry checks precede attempt allocation. All
    later failures retain partial arrays/counters, preserve BaseException, and
    prohibit reuse of the exclusive folder. Native-completed means two steps
    plus postconstraint returned; recorded separately means capture succeeded.
    """
    begin = time.perf_counter()
    _deadline(deadline)
    nominal = _model(nominal_model)
    _require(np.array_equal(nominal.actuator_gear, [[200., 0., 0., 0., 0., 0.]] * 2)
             and np.array_equal(nominal.dof_damping, [1., 1., 0., 0.])
             and nominal.opt.timestep == .01 and nominal.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4,
             "Independent nominal gear200, fixed damping, .01 RK4 model required")
    gain = float(_root_array(true_gain, (), "true_gain"))
    _require(0 < gain <= np.finfo(np.float64).max / 200., "Positive finite gear multiplier required")
    _require(isinstance(sequences, np.ndarray) and sequences.ndim == 3
             and sequences.shape[-1] == 2 and min(sequences.shape[:2]) > 0, "Nonempty sequences[K,H,2] required")
    commands = _commands(sequences, sequences.shape, "sequences")
    k_count, horizon, _ = commands.shape
    _require(isinstance(noise, np.ndarray) and noise.ndim == 3
             and noise.shape[0] > 0 and noise.shape[1:] == (horizon, 2)
             and noise.dtype == np.float64 and np.isfinite(noise).all(), "Finite noise[B,H,2]float64 required")
    noise = noise.copy()
    b_count = len(noise)
    size = mujoco.mj_stateSize(nominal, STATE_SPEC)
    root = _root(root, size)
    out = Path(out)
    _deadline(deadline)
    out.mkdir(parents=True, exist_ok=False)
    counters = dict.fromkeys(("model_copies", "data_allocations", "reset_calls_attempted", "reset_calls_completed",
        "restore_calls_attempted", "restore_calls_completed", "forward_calls_attempted", "forward_calls_completed",
        "roots_initialized", "native_transitions_attempted", "native_transitions_completed",
        "native_substeps_attempted", "native_substeps_completed", "postconstraint_calls_attempted",
        "postconstraint_calls_completed", "transitions_recorded"), 0)
    arrays, model, data, config = {}, None, None, None
    phase, cursor = "setup", None
    try:
        original_hash = _model_hash(nominal)
        model = copy.copy(nominal)
        counters["model_copies"] += 1
        model.actuator_gear[:, 0] = 200. * gain
        model_hash = _model_hash(model)
        data = mujoco.MjData(model)
        counters["data_allocations"] += 1
        finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
        target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        config = {"version": VERSION, "nominal_model_sha256": original_hash, "model_sha256": model_hash,
            "state_spec": int(STATE_SPEC), "state_size": size, "frame_skip": 2, "dt": .02,
            "branch_count": b_count, "candidate_count": k_count, "horizon": horizon,
            "true_gain": gain, "reward_dist_weight": 1., "reward_control_weight": 1.,
            "root_semantics": "reset then complete integration restore then forward; verify qpos/qvel/time",
            "reward_semantics": "cached 3D xpos after two RK4 steps and RNE; no extra forward; applied effort once",
            "event_semantics": "constant gain/target; event-free task horizon eligibility is caller-bound",
            "rng_draws": 0, "new_model_calls": 0,
            "run_status_authority": "enclosing protocol and execution receipts"}
        arrays = {"commands": commands, "noise": noise,
            "applied": np.zeros((b_count, k_count, horizon, 2), np.float64),
            "qpos": np.zeros((b_count, k_count, horizon + 1, 4), np.float64),
            "qvel": np.zeros((b_count, k_count, horizon + 1, 4), np.float64),
            "integration_states": np.zeros((b_count, k_count, horizon + 1, size), np.float64),
            "time": np.zeros((b_count, k_count, horizon + 1), np.float64),
            "initialized": np.zeros((b_count, k_count), bool),
            "substeps_completed": np.zeros((b_count, k_count, horizon), np.int64)}
        arrays.update({name: np.zeros((b_count, k_count, horizon), np.float64)
                       for name in ("distance", "effort", "rewards")})
        arrays.update({name: np.zeros((b_count, k_count, horizon), bool)
                       for name in ("native_completed", "recorded")})
        write_inputs = {"commands": _hash_array(commands), "noise": _hash_array(noise)}
        _write(out / "started.json", {"status": "started", "configuration": config, "root": root,
            "root_sha256": {key: _hash_array(np.asarray(value)) for key, value in root.items()},
            "input_sha256": write_inputs, "automatic_retry": False})
        for b in range(b_count):
            for k in range(k_count):
                cursor = {"branch": b, "candidate": k, "transition": None}
                phase = "restore"
                _deadline(deadline)
                counters["reset_calls_attempted"] += 1
                mujoco.mj_resetData(model, data)
                counters["reset_calls_completed"] += 1
                counters["restore_calls_attempted"] += 1
                mujoco.mj_setState(model, data, root["integration_state"], STATE_SPEC)
                counters["restore_calls_completed"] += 1
                _deadline(deadline)
                counters["forward_calls_attempted"] += 1
                mujoco.mj_forward(model, data)
                counters["forward_calls_completed"] += 1
                _require(np.array_equal(data.qpos, root["qpos"]) and np.array_equal(data.qvel, root["qvel"])
                         and float(data.time) == root["time"], "Restored qpos/qvel/time disagree with supplied root")
                _record(arrays, model, data, b, k, 0)
                arrays["initialized"][b, k] = True
                counters["roots_initialized"] += 1
                for t in range(horizon):
                    cursor["transition"] = t
                    phase = "native"
                    _deadline(deadline)
                    applied = np.clip(commands[k, t].astype(np.float64) + noise[b, t], -1., 1.)
                    arrays["applied"][b, k, t] = applied
                    data.ctrl[:] = applied
                    counters["native_transitions_attempted"] += 1
                    for _ in range(2):
                        _deadline(deadline)
                        counters["native_substeps_attempted"] += 1
                        mujoco.mj_step(model, data)
                        counters["native_substeps_completed"] += 1
                        arrays["substeps_completed"][b, k, t] += 1
                    _deadline(deadline)
                    counters["postconstraint_calls_attempted"] += 1
                    mujoco.mj_rnePostConstraint(model, data)
                    counters["postconstraint_calls_completed"] += 1
                    arrays["native_completed"][b, k, t] = True
                    counters["native_transitions_completed"] += 1
                    phase = "capture"
                    _record(arrays, model, data, b, k, t + 1)
                    distance = float(np.linalg.norm(data.xpos[finger] - data.xpos[target]))
                    effort = float(np.sum(applied ** 2))
                    _require(np.isfinite([distance, effort]).all(), "Nonfinite branch reward")
                    arrays["distance"][b, k, t], arrays["effort"][b, k, t] = distance, effort
                    arrays["rewards"][b, k, t] = -distance - effort
                    arrays["recorded"][b, k, t] = True
                    counters["transitions_recorded"] += 1
                    _deadline(deadline)
        phase = "serialization"
        _deadline(deadline)
        _require(_model_hash(nominal) == original_hash and _model_hash(model) == model_hash,
                 "Nominal or private model changed")
        _save(out / "data.npz", arrays)
        _deadline(deadline)
        files = {name: _file_hash(out / name, deadline) for name in ("started.json", "data.npz")}
        phase = "completion"
        receipt = {"status": "completed", "version": VERSION, "configuration": config,
            "counters": counters, "input_sha256": write_inputs, "files": files,
            "wall_seconds": time.perf_counter() - begin,
            "time_scope": "entry validation, setup, all native work/copies, data serialization and hashes; excludes final receipt write",
            "automatic_retry": False}
        _write(out / "completed.json", receipt)
        _deadline(deadline)
        return receipt
    except BaseException as error:
        failures = []
        if (out / "completed.json").exists():
            try:
                (out / "completed.json").rename(out / "partial-completed.json")
            except BaseException as secondary:  # noqa: BLE001 - original cause takes precedence
                failures.append({"phase": "demote_completion", "error": repr(secondary)})
        if data is not None:
            try:
                arrays.update({"failure_" + name: value for name, value in _state(model, data).items()})
            except BaseException as secondary:  # noqa: BLE001 - preserve even partial native capture
                failures.append({"phase": "capture_native", "error": repr(secondary)})
        try:
            _save(out / "partial.npz", arrays)
        except BaseException as secondary:  # noqa: BLE001
            failures.append({"phase": "save_partial", "error": repr(secondary)})
        try:
            _write(out / "failed.json", {"status": "failed", "version": VERSION, "phase": phase,
                "cursor": cursor, "error": repr(error), "counters": counters, "configuration": config,
                "cleanup_errors": failures, "wall_seconds": time.perf_counter() - begin, "automatic_retry": False})
        except BaseException as secondary:  # noqa: BLE001
            error.add_note("Failure receipt could not be written: " + repr(secondary))
        for failure in failures:
            error.add_note("Failure preservation: " + repr(failure))
        raise
