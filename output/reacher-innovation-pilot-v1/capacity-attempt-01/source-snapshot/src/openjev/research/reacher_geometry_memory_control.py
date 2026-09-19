"""Geometry-only control for four exact, already-defined memory model classes.

Prospective engineering adapter, not a study or new model. The enclosing runner
must authenticate all three saved fit pairs and actual checkpoint classes. This
module does not load/select weights, draw evaluation seeds or define a protocol.

The scalar geometry math, candidate-transition journal, selected-action kernel,
CEM implementation and cache/state accounting are reused unchanged. Only the
outer exact-class admission and geometry-only interface are new. No module
constant, class registry or model behavior is patched. All original reward-head
work remains executed and recorded even though geometry determines actions.

Real observations assimilate once per decision; every candidate has private
state. The selected issued action advances once from the real root. Original
learned rewards remain in executed_predictions; chosen geometry rewards and all
candidate predictions remain in scoring/*. Native replay files are unchanged.
"""
from __future__ import annotations

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
from openjev.research.reacher_geometry_control import (
    VERSION as SCORING_KERNEL_VERSION,
)
from openjev.research.reacher_geometry_control import (
    _diagnostics,
    _journal,
    _preserve_failure,
    _score_bank,
    _selected_prediction,
)
from openjev.research.reacher_geometry_control import (
    scoring_configuration as _kernel_configuration,
)
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.robotics_reacher import HORIZON, ReacherEpisode

VERSION = "reacher-geometry-memory-control-v1"
KINDS = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")


def scoring_configuration(noise_std):
    """Bind new admission semantics while preserving the exact geometry recipe."""
    return {**_kernel_configuration("geometry", noise_std), "version": VERSION,
            "scoring_kernel_version": SCORING_KERNEL_VERSION,
            "permitted_model_kinds": list(KINDS)}


def _identity(plan, model):
    scoring_configuration(plan["noise_std"])
    identity = model_identity(model, plan)
    require(identity["kind"] in KINDS, "Only the four registered memory comparators")
    require("score_modes" not in plan or plan["score_modes"] == ["geometry"],
            "The memory comparison supports geometry scoring only")
    require("score_mode" not in plan or plan["score_mode"] == "geometry",
            "The memory comparison supports geometry scoring only")
    require(plan.get("engineering") is True or
            (plan["planning_horizon"] == 12 and plan["action_block"] == 3),
            "Production scorer uses fixed horizon12/block3")
    return identity


@torch.no_grad()
def score_bank(plan, model, root, bank, deadline=float("inf"), *, failure_out=None):
    """Score one explicit shared bank; return (float32 scores, diagnostics).

    Neither real assimilation nor RNG draws occur. Diagnostics retain commands,
    original rewards and every predicted angle. The terminal state hash proves
    the recorded branch identity; it is never installed as real state.
    """
    start = time.perf_counter()
    score_mode = "geometry"
    identity = _identity(plan, model)
    records = []
    try:
        record = _journal(root, bank, score_mode)
        records.append(record)
        scores = _score_bank(plan, model, root, bank, score_mode, deadline, record)
        diagnostic = _diagnostics(record)
        diagnostic["metadata"].update({"model": identity,
            "configuration": scoring_configuration(plan["noise_std"]),
            "wall_seconds": time.perf_counter() - start})
        return scores, diagnostic
    except BaseException as error:
        _preserve_failure(failure_out, records, error)
        raise


@torch.no_grad()
def score_search(plan, model, root, inputs, step, deadline=float("inf"), *, failure_out=None):
    """Return the original five cache-search fields plus scoring diagnostics.

    raw_rewards contains the chosen mode's unclipped score; original learned
    rewards remain separately available in diagnostics. Four paid batches of64
    retain the unchanged CEM256 global-best and paid-mean semantics.
    """
    start = time.perf_counter()
    score_mode = "geometry"
    identity = _identity(plan, model)
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
            "configuration": scoring_configuration(plan["noise_std"]),
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


@torch.no_grad()
def learned_control(plan, model, panel, inputs_by_step, out, deadline=float("inf"), progress=None,
                    *, cases):
    """Run a row with explicit inputs and unchanged native/cache state semantics."""
    begin = time.perf_counter()
    score_mode = "geometry"
    identity = _identity(plan, model)
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
                plan, model, root, inputs, step, deadline,
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
