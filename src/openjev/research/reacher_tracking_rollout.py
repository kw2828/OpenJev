"""Explicit-input tracking row execution and compact, replayable evidence.

This module chooses no case, seed, target, gain or innovation. Its caller owns
the prospective protocol. Current oracle inputs are selected outside the public
policy. Every output is exclusive; failures retain the completed prefix.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_tracking_dynamics import TrackingDynamicsEpisode, nominal_model
from openjev.research.reacher_tracking_policy import ARMS, TrackingPolicy


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Tracking row cap exceeded")


def save_decision(folder, trace, inputs):
    step = trace["step"]
    arrays = {key: np.asarray(trace[key]).copy() for key in
              ("root_qpos", "root_qvel", "public_packet", "public_qpos", "public_qvel", "action")}
    arrays["planning_gain"] = np.array(trace["planning_gain"], dtype=np.float64)
    meta = {"step": step, "arm": trace["arm"], "input_identities": dict(inputs.identities()),
            "planner_configuration": None, "bank_snapshot": None,
            "journalcallbackmetadata": [], "selectedmetadata": None}
    if trace["search"] is not None:
        search, journal = trace["search"], trace["journal"]
        arrays.update(sequences=search.sequences[0].copy(), scores=search.scores[0].copy(),
            selected_id=np.asarray(search.selected_ids[0], dtype=np.int64),
            predicted_angles=np.concatenate([bank["arrays"]["predicted_angles"][0]
                                             for bank in journal["banks"]]),
            raw_rewards=np.concatenate([bank["arrays"]["geometry_reward"][0]
                                        for bank in journal["banks"]]))
        selected = journal["selected"]["arrays"]
        arrays.update(selected_qpos=selected["qpos"][0, 0].copy(),
            selected_qvel=selected["qvel"][0, 0].copy(),
            selected_angles=selected["predicted_angles"][0, 0, 0].copy(),
            selected_reward=np.asarray(selected["geometry_reward"][0, 0, 0], dtype=np.float32))
        meta.update(planner_configuration=journal["configuration"],
            journalcallbackmetadata=[bank["metadata"] for bank in journal["banks"]],
            selectedmetadata=journal["selected"]["metadata"],
            bank_snapshot={key: trace["bank_snapshot"][key] for key in ("last_operation", "costs")})
    artifacts.save_npz(folder / "decisions" / f"{step:03d}.npz", **arrays)
    write(folder / "decisions" / f"{step:03d}.json", meta)


def run_row(*, arm, target_path, gain_schedule, noise, reset_seed, inputs_by_step,
            gain_grid, window, freeze_after, noise_std, planning_horizon, action_block,
            out, deadline, engineering):
    """One complete row, with inclusive wall time and no automatic recovery.

    Shared innovations must exist before this call. Their bytes are hashed and
    checked on load, and their identities are written per decision. Even the
    zero-command row loads the declared inputs; it performs no planning.
    """
    begin = time.perf_counter()
    out = Path(out)
    if arm not in ARMS or type(engineering) is not bool:
        raise ValueError("Explicit valid role and engineering classification required")
    if not isinstance(inputs_by_step, (list, tuple)) or not inputs_by_step:
        raise ValueError("Explicit nonempty input stems required")
    steps = len(inputs_by_step)
    check_deadline(deadline)
    stems = [Path(path) for path in inputs_by_step]
    inputs = [{"stem": str(stem), "npz_sha256": sha(stem.with_suffix(".npz")),
               "json_sha256": sha(stem.with_suffix(".json"))} for stem in stems]
    if any(stem.with_suffix(".npz").is_symlink() or stem.with_suffix(".json").is_symlink() for stem in stems):
        raise ValueError("Shared inputs must be ordinary files")
    out.mkdir(parents=True, exist_ok=False)
    settings = {"arm": arm, "steps": steps, "reset_seed": reset_seed, "window": window,
        "freeze_after": freeze_after, "noise_std": noise_std, "gain_grid": gain_grid,
        "planning_horizon": planning_horizon, "action_block": action_block, "frame_skip": 2,
        "engineering": engineering, "inputs_by_step": inputs, "automatic_retry": False}
    episode = controller = None
    phase, current_step, closed = "setup", 0, False
    native_seconds = decision_seconds = observe_seconds = serialization_seconds = 0.
    try:
        (out / "decisions").mkdir()
        (out / "observations").mkdir()
        write(out / "started.json", settings)
        gain_schedule = np.asarray(gain_schedule).copy()
        gain_schedule.flags.writeable = False
        episode = TrackingDynamicsEpisode(horizon=steps, target_path=target_path,
            gear_multiplier=gain_schedule, sensor_schedule=np.ones(steps+1, dtype=bool), noise=noise)
        initial = episode.reset(reset_seed)
        controller = TrackingPolicy(nominal_model(), initial, arm=arm, steps=steps,
            gain_grid=gain_grid, window=window, freeze_after=freeze_after, noise_std=noise_std,
            planning_horizon=planning_horizon, action_block=action_block)
        write(out / "initial-controller.json", controller.snapshot())
        setup_seconds = time.perf_counter()-begin
        for step, stem in enumerate(stems):
            current_step = step
            check_deadline(deadline)
            phase = "input"
            bound = inputs[step]
            if sha(stem.with_suffix(".npz")) != bound["npz_sha256"] or sha(stem.with_suffix(".json")) != bound["json_sha256"]:
                raise ValueError("Shared input bytes changed")
            search_inputs = artifacts.load_inputs(stem)
            reference = {}
            if arm in ("public_gain", "true_state"):
                reference["current_gain"] = float(gain_schedule[step])
            if arm == "true_state":
                state = episode.audit_record()["decision_state"]
                reference["true_state"] = (state["qpos"].copy(), state["qvel"].copy())
            phase = "decision"
            tick = time.perf_counter()
            action, trace = controller.decide(search_inputs, deadline=deadline, **reference)
            decision_seconds += time.perf_counter()-tick
            check_deadline(deadline)
            tick = time.perf_counter()
            save_decision(out, trace, search_inputs)
            serialization_seconds += time.perf_counter()-tick
            phase = "native"
            check_deadline(deadline)
            tick = time.perf_counter()
            returned = episode.step(action)
            native_seconds += time.perf_counter()-tick
            public = episode.episode_record()["policy"]
            if not np.array_equal(returned, public["packets"][-1]) or not np.array_equal(action, public["commands"][-1]):
                raise RuntimeError("Native returned packet or issued command differs from selected public record")
            phase = "observation"
            check_deadline(deadline)
            tick = time.perf_counter()
            observation = controller.observe(public["commands"][-1].copy(), returned.copy())
            observe_seconds += time.perf_counter()-tick
            check_deadline(deadline)
            tick = time.perf_counter()
            write(out / "observations" / f"{step+1:03d}.json", observation)
            serialization_seconds += time.perf_counter()-tick
        phase = "finalize"
        record = episode.episode_record()
        if record["metadata"]["status"] != "completed":
            raise RuntimeError("Native row did not complete")
        write(out / "episode.json", record)
        write(out / "controller.json", controller.snapshot())
        phase = "close"
        closed = True
        episode.close()
        check_deadline(deadline)
        files = {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
        check_deadline(deadline)
        rewards = [transition["reward"] for transition in record["audit"]["transitions"]]
        receipt = {"status": "completed", "engineering": engineering, "arm": arm, "steps": steps,
            "native_decisions": steps, "policy_decisions": steps, "observation_updates": steps,
            "native_cost": -float(np.sum(rewards, dtype=np.float64)),
            "setup_seconds": setup_seconds, "decision_seconds": decision_seconds,
            "native_seconds": native_seconds, "observation_seconds": observe_seconds,
            "decision_serialization_seconds": serialization_seconds,
            "wall_seconds": time.perf_counter()-begin,
            "time_scope": "inclusive row validation, setup, all calls/copies, serialization, closure and payload hashing; excludes this final receipt write",
            "files": files}
        write(out / "completed.json", receipt)
        check_deadline(deadline)
        return receipt
    except BaseException as error:
        cleanup_errors = []
        if (out / "completed.json").exists():
            try:
                (out / "completed.json").rename(out / "partial-completed.json")
            except BaseException as capture_error:  # noqa: BLE001 - preserve the primary failure
                cleanup_errors.append({"phase": "demote_completion", "error": repr(capture_error)})
        if episode is not None:
            try:
                write(out / "partial-episode.json", episode.episode_record())
            except BaseException as capture_error:  # noqa: BLE001 - preserve the primary failure
                cleanup_errors.append({"phase": "capture_episode", "error": repr(capture_error)})
        if controller is not None:
            try:
                write(out / "partial-controller.json", controller.snapshot())
            except BaseException as capture_error:  # noqa: BLE001 - preserve the primary failure
                cleanup_errors.append({"phase": "capture_controller", "error": repr(capture_error)})
        if episode is not None and not closed:
            closed = True
            try:
                episode.close()
            except BaseException as close_error:  # noqa: BLE001 - preserve the primary failure
                cleanup_errors.append({"phase": "close", "error": repr(close_error)})
        try:
            write(out / "failed.json", {"status": "failed", "engineering": engineering, "arm": arm,
                "phase": phase, "step": current_step, "error": repr(error), "cleanup_errors": cleanup_errors,
                "wall_seconds": time.perf_counter()-begin, "automatic_retry": False})
        except BaseException as receipt_error:  # noqa: BLE001 - preserve the primary failure
            error.add_note("Failure receipt could not be written: " + repr(receipt_error))
        for cleanup_error in cleanup_errors:
            error.add_note("Failure cleanup: " + repr(cleanup_error))
        raise
