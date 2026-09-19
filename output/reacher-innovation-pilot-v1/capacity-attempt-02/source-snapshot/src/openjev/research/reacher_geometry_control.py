"""Frozen-transition learned/geometry scoring, with saved formula evidence.

This component defines no study, fit selection or random streams. Both modes
call the actual registered model's unchanged advance, including its learned
reward head. Only geometry mode additionally computes approximate geometry.
The caller provides every candidate innovation and native reset/noise seed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_cache_control import (
    PANELS,
    ControlCase,
    _public_arrays,
    _save_partial,
    _save_records,
    _state_arrays,
    _state_bytes,
    check_cap,
    model_identity,
    require,
    work_accounting,
)
from openjev.research.reacher_geometry_reward import (
    geometry_reward_components,
    geometry_reward_configuration,
)
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.reacher_world_models import repeat_index
from openjev.research.robotics_reacher import HORIZON, ReacherEpisode

VERSION = "reacher-geometry-control-v1"
SCORE_MODES = ("learned", "geometry")
KINDS = ("residual_gru", "cached_mlp")
_GEOMETRY = ("joint_angles", "pair_norms", "fingertip", "distance", "action_cost")


def scoring_configuration(score_mode, noise_std):
    require(score_mode in SCORE_MODES, "Exactly learned or geometry scoring")
    return {
        "version": VERSION, "score_mode": score_mode,
        "reward_clipping": [-2.5, 0.0], "sum_dtype": "float32_sequential",
        "geometry": geometry_reward_configuration(noise_std),
        "geometry_computed": score_mode == "geometry",
        "learned_head_work_retained": True,
        "joint_limit_diagnostic": "abs(projected q1) > 3 radians; counted only, no penalty",
        "timing_scope": "Geometry seconds time its function only; copies, validation and diagnostics remain charged in total search/decision/row wall time.",
        "run_status_authority": "enclosing protocol and execution receipts",
    }


def _identity(plan, model, score_mode):
    scoring_configuration(score_mode, plan["noise_std"])
    identity = model_identity(model, plan)
    require(identity["kind"] in KINDS, "Only original residual_gru and cached_mlp classes")
    require(plan.get("engineering") is True or
            (plan["planning_horizon"] == 12 and plan["action_block"] == 3),
            "Production scorer uses fixed horizon12/block3")
    return identity


def _array_sha(value):
    # Dtype and shape are separately recorded; bytes are always C-order.
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _predictions(angles, reward, count):
    require(
        isinstance(angles, torch.Tensor) and isinstance(reward, torch.Tensor)
        and angles.shape == (count, 4) and reward.shape == (count,)
        and angles.dtype == reward.dtype == torch.float32
        and angles.device.type == reward.device.type == "cpu"
        and bool(torch.isfinite(angles).all()) and bool(torch.isfinite(reward).all()),
        "Finite CPU float32 native-shaped predictions",
    )


def _journal(root, bank, score_mode):
    require(isinstance(bank, np.ndarray) and bank.dtype == np.float32 and bank.ndim == 4
            and bank.shape[-1] == 2 and all(size > 0 for size in bank.shape)
            and bank.shape[1] <= 256 and np.isfinite(bank).all() and (np.abs(bank) <= 1).all(),
            "Finite issued candidate bank [N,K<=256,H,2] in [-1,1]")
    _state_bytes(root)
    n, _, _, _ = bank.shape
    require(all(len(value) == n for value in root.values()) and root["packet"].shape == (n, 8),
            "Candidate/root case alignment")
    _public_arrays(root["packet"].detach().numpy())
    return {
        "bank": bank.copy(), "root": _state_arrays(root),
        "root_sha256": canonical_tensor_hash(root), "score_mode": score_mode,
        "root_target": root["packet"][:, 4:6].detach().numpy().copy(),
        "values": {key: [] for key in ("predicted_angles", "learned_rewards", "selected_rewards")},
        "model_advance_attempted_samples": 0, "model_advance_samples": 0,
        "geometry_attempted_samples": 0, "geometry_samples": 0,
        "model_advance_seconds": 0.0, "geometry_seconds": 0.0,
        "max_state_bytes": 0, "step": 0, "prefix_state_sha256": [],
    }


def _append(record, key, value, n, k):
    value = value.detach().numpy().copy()
    record["values"].setdefault(key, []).append(value.reshape(n, k, *value.shape[1:]))


def _geometry_score(record, angles, target, actions, noise_std, n, k):
    record["geometry_attempted_samples"] += n * k
    tick = time.perf_counter()
    try:
        result = geometry_reward_components(angles, target, actions, noise_std)
    finally:
        record["geometry_seconds"] += time.perf_counter() - tick
    record["geometry_samples"] += n * k
    for key in _GEOMETRY:
        _append(record, "geometry_" + key, getattr(result, key), n, k)
    return result.reward


def _diagnostics(record):
    arrays = {"commands": record["bank"], "root_target": record["root_target"]}
    arrays.update({key: np.stack(values, axis=2) for key, values in record["values"].items() if values})
    if record.get("failed_predictions"):
        arrays.update(record["failed_predictions"])
    norms = arrays.get("geometry_pair_norms")
    angles = arrays.get("geometry_joint_angles")
    return {"arrays": arrays, "metadata": {
        "score_mode": record["score_mode"], "root_sha256": record["root_sha256"],
        "bank_shape": list(record["bank"].shape), "bank_dtype": "float32",
        "bank_sha256": _array_sha(record["bank"]),
        "completed_model_offsets": len(record["values"]["learned_rewards"]),
        "completed_scored_offsets": len(record["values"]["selected_rewards"]),
        "model_advance_attempted_samples": record["model_advance_attempted_samples"],
        "model_advance_samples": record["model_advance_samples"],
        "geometry_attempted_samples": record["geometry_attempted_samples"],
        "geometry_samples": record["geometry_samples"],
        "model_advance_seconds": record["model_advance_seconds"],
        "geometry_seconds": record["geometry_seconds"],
        "bank_wall_seconds": record.get("wall_seconds", 0.0),
        "maximum_candidate_state_tensor_bytes": record["max_state_bytes"],
        "minimum_pair_norm": None if norms is None else float(norms.min()),
        "joint_limit_violation_samples": None if angles is None else int((np.abs(angles[..., 1]) > 3).sum()),
        "terminal_state_sha256": record.get("terminal_state_sha256"),
        "prefix_state_sha256": record["prefix_state_sha256"],
        "geometry_diagnostics_omitted_in_learned_mode": record["score_mode"] == "learned",
    }}


def _preserve_failure(out, records, error):
    if out is None:
        return
    try:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        metadata = []
        for index, record in enumerate(records):
            diagnostic = _diagnostics(record)
            arrays = {**diagnostic["arrays"], **{f"root__{k}": v for k, v in record["root"].items()}}
            artifacts.save_npz(out / f"bank-{index:03d}.npz", **arrays)
            metadata.append(diagnostic["metadata"])
        artifacts.write(out / "failed.json", {"status": "failed", "exception_type": type(error).__name__,
            "message": str(error), "banks": metadata, "scope": "Completed and partial candidate prefixes; no retry"})
    except BaseException as preservation_error:  # noqa: BLE001
        error.add_note(f"Scoring preservation failed: {type(preservation_error).__name__}: {preservation_error}")


def _score_bank(plan, model, root, bank, score_mode, deadline, record):
    start = time.perf_counter()
    n, k, horizon, _ = bank.shape
    require(horizon <= plan["planning_horizon"], "Bank horizon exceeds declared planning horizon")
    imagined = repeat_index(root, torch.arange(n).repeat_interleave(k))
    target = torch.from_numpy(np.repeat(record["root_target"], k, axis=0))
    record["max_state_bytes"] = _state_bytes(imagined)
    total = torch.zeros(n * k, dtype=torch.float32)
    try:
        for offset in range(horizon):
            record["step"] = offset
            check_cap(deadline)
            actions = torch.from_numpy(bank[:, :, offset].reshape(n * k, 2).copy())
            record["model_advance_attempted_samples"] += n * k
            tick = time.perf_counter()
            try:
                imagined, angles, learned = model.advance(imagined, actions)
            finally:
                record["model_advance_seconds"] += time.perf_counter() - tick
            record["failed_predictions"] = {key: value.detach().numpy().copy()
                for key, value in (("failed_angles", angles), ("failed_learned_reward", learned))
                if isinstance(value, torch.Tensor) and value.device.type == "cpu"}
            _predictions(angles, learned, n * k)
            record["max_state_bytes"] = max(record["max_state_bytes"], _state_bytes(imagined))
            record["prefix_state_sha256"].append(canonical_tensor_hash(imagined))
            record["model_advance_samples"] += n * k
            _append(record, "predicted_angles", angles, n, k)
            _append(record, "learned_rewards", learned, n, k)
            selected = learned if score_mode == "learned" else _geometry_score(
                record, angles, target, actions, plan["noise_std"], n, k)
            _append(record, "selected_rewards", selected, n, k)
            total += selected.clamp(-2.5, 0.0)
            record.pop("failed_predictions", None)
        require(canonical_tensor_hash(root) == record["root_sha256"], "Candidate scoring mutated real root")
        record["terminal_state_sha256"] = canonical_tensor_hash(imagined)
        check_cap(deadline)
        return total.reshape(n, k).numpy().copy()
    finally:
        record["wall_seconds"] = time.perf_counter() - start


@torch.no_grad()
def score_bank(plan, model, root, bank, score_mode, deadline=float("inf"), *, failure_out=None):
    """Score one explicit shared bank; return (float32 scores, diagnostics).

    Neither real assimilation nor RNG draws occur. Diagnostics retain commands,
    original rewards and every predicted angle. The terminal state hash proves
    the recorded branch identity; it is never installed as real state.
    """
    start = time.perf_counter()
    identity = _identity(plan, model, score_mode)
    records = []
    try:
        record = _journal(root, bank, score_mode)
        records.append(record)
        scores = _score_bank(plan, model, root, bank, score_mode, deadline, record)
        diagnostic = _diagnostics(record)
        diagnostic["metadata"].update({"model": identity,
            "configuration": scoring_configuration(score_mode, plan["noise_std"]),
            "wall_seconds": time.perf_counter() - start})
        return scores, diagnostic
    except BaseException as error:
        _preserve_failure(failure_out, records, error)
        raise


@torch.no_grad()
def score_search(plan, model, root, inputs, step, deadline=float("inf"), *, score_mode, failure_out=None):
    """Return the original five cache-search fields plus scoring diagnostics.

    raw_rewards contains the chosen mode's unclipped score; original learned
    rewards remain separately available in diagnostics. Four paid batches of64
    retain the unchanged CEM256 global-best and paid-mean semantics.
    """
    start = time.perf_counter()
    identity = _identity(plan, model, score_mode)
    require(isinstance(inputs, SearchInputs), "Explicit saved innovations required")
    records = []

    def score(bank):
        record = _journal(root, bank, score_mode)
        records.append(record)
        return _score_bank(plan, model, root, bank, score_mode, deadline, record)

    try:
        result = search("cem256", inputs, score, step=step, steps=plan["steps"],
                        planning_horizon=plan["planning_horizon"], action_block=plan["action_block"])
        sizes = [record["bank"].shape[1] for record in records]
        require(sizes == [64] * 4, "Exact CEM256 callback schedule")
        parts = [_diagnostics(record) for record in records]
        arrays = {key: np.concatenate([part["arrays"][key] for part in parts], axis=1)
                  for key in parts[0]["arrays"] if key != "root_target"}
        arrays["root_target"] = parts[0]["arrays"]["root_target"]
        arrays["selected_ids"] = result.selected_ids.copy()
        arrays["selected_actions"] = result.selected_actions.copy()
        work = {"candidate_evaluations": result.candidate_evaluations,
            "imagined_transitions": result.imagined_transitions, "root_tensor_bytes": _state_bytes(root),
            "max_single_candidate_state_tensor_bytes": max(record["max_state_bytes"] for record in records),
            "tensor_bytes_are_not_peak_process_memory": True,
            "model_work": work_accounting(model, assimilate_samples=0, advance_samples=result.imagined_transitions),
            "geometry_samples": sum(record["geometry_samples"] for record in records),
            "geometry_seconds": sum(record["geometry_seconds"] for record in records),
            "model_advance_seconds": sum(record["model_advance_seconds"] for record in records),
            "diagnostic_array_bytes": sum(value.nbytes for value in arrays.values())}
        metadata = {"version": VERSION, "model": identity, "step": step, "score_mode": score_mode,
            "configuration": scoring_configuration(score_mode, plan["noise_std"]),
            "root_sha256": records[0]["root_sha256"],
            "callbacks": [{"candidate_start": i * 64, "candidate_stop": (i + 1) * 64,
                           **part["metadata"]} for i, part in enumerate(parts)], "work": work}
        check_cap(deadline)
        seconds = time.perf_counter() - start
        metadata["search_seconds"] = seconds
        return result, arrays["selected_rewards"], sizes, seconds, work, {"arrays": arrays, "metadata": metadata}
    except BaseException as error:
        _preserve_failure(failure_out, records, error)
        raise


def _selected_prediction(plan, model, root, action, score_mode, journal):
    """One selected-action advance from the real root, including scoring work."""
    n = len(action)
    root_hash = canonical_tensor_hash(root)
    metadata = {"model_advance_attempted_samples": n, "model_advance_samples": 0,
        "geometry_attempted_samples": 0, "geometry_samples": 0, "geometry_seconds": 0.0,
        "root_sha256": root_hash, "minimum_pair_norm": None, "joint_limit_violation_samples": None}
    journal["metadata"] = metadata
    tick = time.perf_counter()
    try:
        state, angles, learned = model.advance(root, torch.from_numpy(action.copy()))
    finally:
        metadata["model_advance_seconds"] = time.perf_counter() - tick
    journal["carried_state"] = _state_arrays(state)
    journal["arrays"] = {key: value.detach().numpy().copy()
        for key, value in (("selected_angles", angles), ("selected_learned_reward", learned))
        if isinstance(value, torch.Tensor) and value.device.type == "cpu"}
    _predictions(angles, learned, n)
    metadata["model_advance_samples"] = n
    metadata["carried_sha256"] = canonical_tensor_hash(state)
    chosen = learned
    if score_mode == "geometry":
        metadata["geometry_attempted_samples"] = n
        tick = time.perf_counter()
        try:
            result = geometry_reward_components(angles, root["packet"][:, 4:6],
                                                torch.from_numpy(action.copy()), plan["noise_std"])
        finally:
            metadata["geometry_seconds"] = time.perf_counter() - tick
        metadata["geometry_samples"] = n
        metadata["minimum_pair_norm"] = float(result.pair_norms.min())
        metadata["joint_limit_violation_samples"] = int((result.joint_angles[:, 1].abs() > 3).sum())
        chosen = result.reward
        journal["arrays"].update({"selected_geometry_" + key: getattr(result, key).numpy().copy()
                                  for key in _GEOMETRY})
    journal["arrays"]["selected_reward"] = chosen.numpy().copy()
    require(canonical_tensor_hash(root) == root_hash, "Selected advance mutated real root")
    return state, angles, learned


@torch.no_grad()
def learned_control(plan, model, panel, inputs_by_step, out, deadline=float("inf"), progress=None,
                    *, cases, score_mode):
    """Run a row with explicit inputs and unchanged native/cache state semantics."""
    begin = time.perf_counter()
    identity = _identity(plan, model, score_mode)
    require(panel in PANELS, "Unknown sensing panel")
    require(type(plan["control_episodes"]) is int and plan["control_episodes"] > 0
            and len(cases) == plan["control_episodes"] and all(type(case) is ControlCase for case in cases),
            "Explicit complete control cases")
    require(len(inputs_by_step) == HORIZON, "One innovation bank for each real decision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "scoring").mkdir()
    envs, initialized, packets = [], [], []
    original_error, active_root, active_diagnostic, active_selected = None, None, None, None
    decisions, native_times, angles, rewards = [], [], [], []
    root_states, carried_states, state_work = [], [], []
    try:
        for case in cases:
            check_cap(deadline)
            env = ReacherEpisode(noise_std=plan["noise_std"])
            envs.append(env)
            require(env.dt == plan["dt"], "Native/model timestep alignment")
            packets.append(env.reset(case.reset_seed, case.sensor_schedule, noise_seed=case.noise_seed))
            initialized.append(env)
        packets = np.stack(packets)
        state = model.initial(len(envs))
        setup = time.perf_counter() - begin
        for step in range(HORIZON):
            active_root, active_diagnostic, active_selected = None, None, None
            if progress is not None:
                progress["step"] = step
            check_cap(deadline)
            tick = time.perf_counter()
            _public_arrays(packets)
            value = inputs_by_step[step]
            inputs = value if isinstance(value, SearchInputs) else artifacts.load_inputs(value)
            require(inputs.initial.shape[0] == len(envs), "Innovation case count")
            root = model.assimilate(state, torch.from_numpy(packets.copy()))
            active_root = _state_arrays(root)
            root_hash = canonical_tensor_hash(root)
            result, raw, sizes, search_seconds, work, active_diagnostic = score_search(
                plan, model, root, inputs, step, deadline, score_mode=score_mode,
                failure_out=out / "partial-scoring" / f"{step:03d}")
            active_selected = {}
            state, predicted_angles, predicted_reward = _selected_prediction(
                plan, model, root, result.selected_actions, score_mode, active_selected)
            active_diagnostic["arrays"].update(active_selected["arrays"])
            active_diagnostic["metadata"]["selected_advance"] = active_selected["metadata"]
            root_states.append(active_root)
            carried_states.append(_state_arrays(state))
            angles.append(predicted_angles.numpy().copy())
            rewards.append(predicted_reward.numpy().copy())
            state_work.append({"step": step, "root_sha256": root_hash,
                "carried_sha256": canonical_tensor_hash(state), "root_tensor_bytes": _state_bytes(root),
                "carried_tensor_bytes": _state_bytes(state), "search": work,
                "model_work": work_accounting(model, assimilate_samples=len(envs),
                                               advance_samples=len(envs) + result.imagined_transitions)})
            artifacts.save_trace(out / "decisions" / f"{step:03d}", result, raw, sizes, search_seconds)
            artifacts.save_npz(out / "scoring" / f"{step:03d}.npz", **active_diagnostic["arrays"])
            artifacts.write(out / "scoring" / f"{step:03d}.json", active_diagnostic["metadata"])
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack([env.step(action) for env, action in zip(envs, result.selected_actions, strict=True)])
            native_times.append(time.perf_counter() - tick)
        require(all(env.finished and env.step_index == HORIZON for env in envs), "Complete native50 termination")
        require(model_identity(model, plan) == identity, "Inference changed model tensors/configuration")
        _save_records(out / "episodes", [env.episode_record() for env in envs])
        artifacts.save_npz(out / "executed_predictions.npz", angles=np.stack(angles, 1), rewards=np.stack(rewards, 1))
        states = {f"{label}__{key}": np.stack([row[key] for row in rows], axis=1)
                  for label, rows in (("root", root_states), ("carried", carried_states)) for key in rows[0]}
        artifacts.save_npz(out / "states.npz", **states)
        artifacts.write(out / "state-work.json", {"model": identity,
            "steps": state_work,
            "state_arrays_bytes": sum(value.nbytes for value in states.values()),
            "max_single_candidate_state_tensor_bytes": max(row["search"]["max_single_candidate_state_tensor_bytes"] for row in state_work),
            "tensor_bytes_are_not_peak_process_memory": True,
            "aggregate_model_work": work_accounting(model, assimilate_samples=len(envs) * HORIZON,
                advance_samples=len(envs) * HORIZON + sum(row["search"]["imagined_transitions"] for row in state_work)),
            "auxiliary_teacher_calls": 0, "reset_calls": 0})
        timing = {"setup_seconds": setup, "decision_seconds": decisions, "native_step_seconds": native_times,
            "row_wall_seconds": time.perf_counter() - begin,
            "observation_assimilations": len(envs) * HORIZON, "executed_action_advances": len(envs) * HORIZON}
        artifacts.write(out / "timings.json", timing)
        check_cap(deadline)
        return timing
    except BaseException as error:
        original_error = error
        try:
            if initialized and not (out / "episodes.npz").exists():
                _save_partial(out / "partial-episodes", initialized)
            if active_root is not None:
                artifacts.save_npz(out / "partial-root.npz", **active_root)
            if active_diagnostic is not None:
                arrays = dict(active_diagnostic["arrays"])
                if active_selected is not None:
                    arrays.update(active_selected.get("arrays", {}))
                    if "metadata" in active_selected:
                        active_diagnostic["metadata"]["selected_advance"] = active_selected["metadata"]
                    if "carried_state" in active_selected:
                        artifacts.save_npz(out / "partial-carried.npz", **active_selected["carried_state"])
                artifacts.save_npz(out / "partial-active-scoring.npz", **arrays)
                artifacts.write(out / "partial-active-scoring.json", active_diagnostic["metadata"])
            if root_states:
                states = {f"{label}__{key}": np.stack([row[key] for row in rows], axis=1)
                          for label, rows in (("root", root_states), ("carried", carried_states)) for key in rows[0]}
                artifacts.save_npz(out / "partial-states.npz", **states)
            artifacts.write(out / "failed.json", {"status": "failed", "score_mode": score_mode,
                "exception_type": type(error).__name__, "message": str(error),
                "completed_steps_by_case": [env.step_index for env in initialized],
                "completed_decision_seconds": decisions, "completed_native_step_seconds": native_times,
                "completed_state_work": state_work, "wall_seconds": time.perf_counter() - begin})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Artifact preservation failed: {type(preservation_error).__name__}: {preservation_error}")
        raise
    finally:
        close_errors = []
        for env in envs:
            try:
                env.close()
            except BaseException as error:  # noqa: BLE001
                close_errors.append(error)
        if close_errors:
            if original_error is not None:
                original_error.add_note(f"Environment close failures: {[str(e) for e in close_errors]}")
            else:
                raise close_errors[0]
