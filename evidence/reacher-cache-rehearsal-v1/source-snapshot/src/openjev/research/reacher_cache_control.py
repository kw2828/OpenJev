"""Isolated actual-class control/prediction for five public-cache comparators.

Derived from the frozen memory controller in a separate module.

No study, checkpoint selection, fitting, RNG namespace or scientific result is
defined here. Caller supplies every reset/noise seed, sensing schedule and CEM
innovation. Each decision assimilates its REAL public packet once, searches
private branches, then advances the selected action from that real root once.
Candidate terminal states never become real state. The terminal observation is
recorded by the environment but has no further policy decision/assimilation.

Native episodes and search traces preserve the existing replay formats. New
root/carried state arrays expose public-memory/action alignment to a later
saved-output auditor. Tensor byte counts are payload sizes, not peak process
memory; dense-affine counts are estimates, not total FLOPs or compute matching.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from openjev.research import reacher_search_protocol as artifacts
from openjev.research.reacher_adaptive_search import SearchInputs, search
from openjev.research.reacher_cache_training import REGISTRY, SEMANTICS
from openjev.research.reacher_cached_observation import configuration_for
from openjev.research.reacher_cached_observation import operation_counts as cache_operation_counts
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.reacher_observation_baseline import operation_counts
from openjev.research.reacher_world_models import repeat_index
from openjev.research.robotics_reacher import HORIZON, ReacherEpisode

VERSION = "reacher-cache-control-engineering-v1"
PANELS = ("full", "ordinary", "shift")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_cap(deadline):
    require(type(deadline) in (float, int) and not math.isnan(deadline), "Valid monotonic deadline")
    if time.monotonic() >= deadline:
        raise TimeoutError("Cooperative control/prediction cap")


@dataclass(frozen=True, eq=False)
class ControlCase:
    reset_seed: int
    noise_seed: int
    sensor_schedule: np.ndarray

    def __post_init__(self):
        require(
            all(type(s) is int and 0 <= s < 2**64 for s in (self.reset_seed, self.noise_seed)),
            "Explicit uint64 reset/noise seeds",
        )
        schedule = np.asarray(self.sensor_schedule)
        require(
            schedule.dtype == np.bool_ and schedule.shape == (HORIZON + 1,) and bool(schedule[0]),
            "Initial-valid native50 sensing schedule",
        )
        object.__setattr__(self, "sensor_schedule", np.frombuffer(schedule.tobytes(), dtype=np.bool_))


def _plan(plan):
    require(
        isinstance(plan, dict) and type(plan["steps"]) is int and plan["steps"] == HORIZON,
        "Native50-step plan required",
    )
    for key in ("planning_horizon", "action_block"):
        require(type(plan[key]) is int and plan[key] > 0, f"Positive integer {key}")
    for key in ("dt", "noise_std"):
        require(
            type(plan[key]) in (float, int) and math.isfinite(plan[key]) and plan[key] >= 0,
            f"Finite nonnegative {key}",
        )
    require(plan["dt"] > 0, "Positive dt")


def model_identity(model, plan):
    """Actual class and scalars, not merely compatible tensor names."""
    _plan(plan)
    kinds = [kind for kind, cls in REGISTRY.items() if type(model) is cls]
    require(len(kinds) == 1, "Exact registered model class required")
    kind = kinds[0]
    require(
        not model.training
        and model.residual_reward is True
        and model.dt == plan["dt"]
        and model.noise_std == plan["noise_std"],
        "Eval-mode physical/residual configuration",
    )
    require(not list(model.named_buffers()), "Unexpected model buffers")
    require(
        all(
            p.dtype == torch.float32 and p.device.type == "cpu" and bool(torch.isfinite(p).all())
            for p in model.parameters()
        ),
        "Finite CPU float32 model",
    )
    width = model.width if kind in ("packet_mlp", "cached_mlp") else model.hidden_size
    width_key = "mlp_width" if kind in ("packet_mlp", "cached_mlp") else "hidden_size"
    if width_key in plan:
        require(width == plan[width_key], "Actual model width differs from plan")
    public_memory = None
    if kind in ("encoded_current_gru", "cached_gru", "cached_mlp"):
        public_memory = configuration_for(type(model), width=width, dt=plan["dt"],
            noise_std=plan["noise_std"], residual_reward=True)
        require(model.configuration() == public_memory, "Live public-cache configuration drift")
    return {
        "version": VERSION,
        "kind": kind,
        "model_class": type(model).__name__,
        "width": width,
        "dt": model.dt,
        "noise_std": model.noise_std,
        "residual_reward": True,
        "real_assimilation": SEMANTICS[kind],
        "public_memory_configuration": public_memory,
        "weight_tensor_sha256": canonical_tensor_hash(model.state_dict()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


def _public_arrays(packets, commands=None):
    require(
        isinstance(packets, np.ndarray)
        and packets.dtype == np.float32
        and packets.shape[-1] == 8
        and np.isfinite(packets).all(),
        "Finite float32 public packets",
    )
    require(np.isin(packets[..., 6], (0, 1)).all() and (packets[..., 7] >= 0).all(), "Public validity/age")
    require((packets[..., :4][packets[..., 6] == 0] == 0).all(), "Unavailable angular fields must be masked")
    if commands is not None:
        require(
            isinstance(commands, np.ndarray)
            and commands.dtype == np.float32
            and commands.shape[-1] == 2
            and np.isfinite(commands).all()
            and (np.abs(commands) <= 1).all(),
            "Finite clipped float32 issued commands",
        )


def _state_bytes(state):
    require(isinstance(state, dict) and state, "Named state required")
    for value in state.values():
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.ndim >= 1
            and (not value.is_floating_point() or bool(torch.isfinite(value).all())),
            "Finite CPU state tensor",
        )
    return sum(value.numel() * value.element_size() for value in state.values())


def _state_arrays(state):
    _state_bytes(state)
    return {key: value.detach().numpy().copy() for key, value in state.items()}


def work_accounting(model, *, assimilate_samples, advance_samples):
    require(
        all(type(n) is int and n >= 0 for n in (assimilate_samples, advance_samples)),
        "Nonnegative sample counts",
    )
    kind = next((key for key, cls in REGISTRY.items() if type(model) is cls), None)
    require(kind is not None, "Exact registered model class required")
    if kind in ("encoded_current_gru", "cached_gru", "cached_mlp"):
        result = cache_operation_counts(
            model, assimilate_samples=assimilate_samples, advance_samples=advance_samples
        )
    elif kind == "packet_mlp":
        result = operation_counts(
            model, assimilate_samples=assimilate_samples, advance_samples=advance_samples
        )
    else:
        h, a, d = model.hidden_size, assimilate_samples, advance_samples
        result = {
            "gru_cell_sample_calls": a + d,
            "linear_layer_sample_calls": 4 * d,
            "analytic_reward_sample_calls": d,
            "dense_affine_macs": a * 3 * h * (h + 8) + d * (5 * h * h + 25 * h),
            "counts_are_not_total_flops_or_measured_wall_time": True,
        }
    return {
        "assimilate_samples": assimilate_samples,
        "advance_samples": advance_samples,
        "operations": result,
        "compute_matched": False,
        "limits": "Affine estimates exclude validation, copies, masks, nonlinearities, gate arithmetic and analytic reward operations; all executed work remains charged in wall time.",
    }


@torch.no_grad()
def score_search(plan, model, root, inputs, step, deadline=float("inf")):
    """CEM256 with independent copies of the same real root for every stage.

    Returns (SearchResult, raw_rewards, callback_sizes, search_seconds, work).
    Root identity is checked after every callback; no real assimilation occurs.
    """
    started = time.perf_counter()
    model_identity(model, plan)
    require(isinstance(inputs, SearchInputs), "Explicit saved innovations required")
    root_bytes = _state_bytes(root)
    root_hash = canonical_tensor_hash(root)
    raw, sizes = [], []
    state_bytes = []

    def score(bank):
        check_cap(deadline)
        n, k, horizon, _ = bank.shape
        require(all(len(value) == n for value in root.values()), "Candidate/root case alignment")
        imagined = repeat_index(root, torch.arange(n).repeat_interleave(k))
        state_bytes.append(_state_bytes(imagined))
        total, rewards = torch.zeros(n * k, dtype=torch.float32), []
        for offset in range(horizon):
            check_cap(deadline)
            actions = torch.from_numpy(bank[:, :, offset].reshape(n * k, 2).copy())
            imagined, angles, reward = model.advance(imagined, actions)
            require(
                angles.shape == (n * k, 4)
                and angles.dtype == torch.float32
                and reward.shape == (n * k,)
                and reward.dtype == torch.float32
                and bool(torch.isfinite(angles).all())
                and bool(torch.isfinite(reward).all()),
                "Finite native-shaped imagined predictions",
            )
            state_bytes.append(_state_bytes(imagined))
            total += reward.clamp(-2.5, 0.0)
            rewards.append(reward.reshape(n, k).numpy().copy())
        require(canonical_tensor_hash(root) == root_hash, "Candidate search mutated real root")
        sizes.append(k)
        raw.append(np.stack(rewards, axis=2))
        return total.reshape(n, k).numpy()

    result = search(
        "cem256",
        inputs,
        score,
        step=step,
        steps=plan["steps"],
        planning_horizon=plan["planning_horizon"],
        action_block=plan["action_block"],
    )
    require(sizes == [64] * 4, "Exact CEM256 callback schedule")
    check_cap(deadline)
    work = {
        "candidate_evaluations": result.candidate_evaluations,
        "imagined_transitions": result.imagined_transitions,
        "root_tensor_bytes": root_bytes,
        "max_single_candidate_state_tensor_bytes": max(state_bytes),
        "tensor_bytes_are_not_peak_process_memory": True,
        "model_work": work_accounting(
            model, assimilate_samples=0, advance_samples=result.imagined_transitions
        ),
    }
    return result, np.concatenate(raw, axis=1), sizes, time.perf_counter() - started, work


def _save_records(stem, records):
    """Exact legacy category__field NPZ plus metadata-list JSON convention."""
    arrays = {}
    for category in ("policy", "audit"):
        keys = set(records[0][category])
        require(all(set(record[category]) == keys for record in records), "Consistent episode arrays")
        for key in sorted(keys):
            arrays[f"{category}__{key}"] = np.stack([record[category][key] for record in records])
    artifacts.save_npz(Path(stem).with_suffix(".npz"), **arrays)
    artifacts.write(Path(stem).with_suffix(".json"), [record["metadata"] for record in records])


def _save_partial(stem, envs):
    records = [env.episode_record() for env in envs]
    lengths = [len(record["policy"]["commands"]) for record in records]
    if len(set(lengths)) == 1:
        _save_records(stem, records)
    else:
        stem.mkdir(parents=True, exist_ok=False)
        for index, record in enumerate(records):
            _save_records(stem / f"{index:03d}", [record])
        artifacts.write(stem / "manifest.json", {"completed_steps_by_case": lengths})


@torch.no_grad()
def learned_control(plan, model, panel, inputs_by_step, out, deadline=float("inf"), progress=None, *, cases):
    """Run one native50 row; no resets, seeds, innovations or model selection.

    Setup, each complete decision (including trace/state collection), native
    stepping and total row wall time are recorded separately. A failed attempt
    preserves available native history without changing its original exception.
    """
    begin = time.perf_counter()
    identity = model_identity(model, plan)
    require(panel in PANELS, "Unknown sensing panel")
    require(
        type(plan["control_episodes"]) is int
        and plan["control_episodes"] > 0
        and len(cases) == plan["control_episodes"]
        and all(type(case) is ControlCase for case in cases),
        "Explicit complete control cases",
    )
    require(len(inputs_by_step) == HORIZON, "One innovation bank for each real decision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    envs, initialized, packets = [], [], []
    original_error = None
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
        decisions, native_times, angles, rewards = [], [], [], []
        root_states, carried_states, state_work = [], [], []
        for step in range(HORIZON):
            if progress is not None:
                progress["step"] = step
            check_cap(deadline)
            tick = time.perf_counter()
            _public_arrays(packets)
            value = inputs_by_step[step]
            inputs = value if isinstance(value, SearchInputs) else artifacts.load_inputs(value)
            require(inputs.initial.shape[0] == len(envs), "Innovation case count")
            # Exactly one real assimilation, using only the current public packet.
            root = model.assimilate(state, torch.from_numpy(packets.copy()))
            root_state = _state_arrays(root)
            root_hash = canonical_tensor_hash(root)
            result, raw, sizes, search_seconds, work = score_search(plan, model, root, inputs, step, deadline)
            # Re-advance the issued action from the real root, never a candidate
            # terminal state. This is the cache models' one-advance real phase.
            state, predicted_angles, predicted_reward = model.advance(
                root, torch.from_numpy(result.selected_actions.copy())
            )
            require(
                predicted_angles.shape == (len(envs), 4)
                and predicted_reward.shape == (len(envs),)
                and bool(torch.isfinite(predicted_angles).all())
                and bool(torch.isfinite(predicted_reward).all()),
                "Finite selected-action predictions",
            )
            require(canonical_tensor_hash(root) == root_hash, "Selected advance mutated real root")
            root_states.append(root_state)
            carried_states.append(_state_arrays(state))
            angles.append(predicted_angles.numpy().copy())
            rewards.append(predicted_reward.numpy().copy())
            state_work.append(
                {
                    "step": step,
                    "root_sha256": root_hash,
                    "carried_sha256": canonical_tensor_hash(state),
                    "root_tensor_bytes": _state_bytes(root),
                    "carried_tensor_bytes": _state_bytes(state),
                    "search": work,
                    "model_work": work_accounting(
                        model,
                        assimilate_samples=len(envs),
                        advance_samples=len(envs) + result.imagined_transitions,
                    ),
                }
            )
            artifacts.save_trace(out / "decisions" / f"{step:03d}", result, raw, sizes, search_seconds)
            decisions.append(time.perf_counter() - tick)
            check_cap(deadline)
            tick = time.perf_counter()
            packets = np.stack(
                [env.step(action) for env, action in zip(envs, result.selected_actions, strict=True)]
            )
            native_times.append(time.perf_counter() - tick)
        require(
            all(env.finished and env.step_index == HORIZON for env in envs), "Complete native50 termination"
        )
        require(model_identity(model, plan) == identity, "Inference changed model tensors/configuration")
        _save_records(out / "episodes", [env.episode_record() for env in envs])
        artifacts.save_npz(
            out / "executed_predictions.npz", angles=np.stack(angles, 1), rewards=np.stack(rewards, 1)
        )
        states = {
            f"{label}__{key}": np.stack([row[key] for row in rows], axis=1)
            for label, rows in (("root", root_states), ("carried", carried_states))
            for key in rows[0]
        }
        artifacts.save_npz(out / "states.npz", **states)
        work = {
            "model": identity,
            "steps": state_work,
            "state_arrays_bytes": sum(value.nbytes for value in states.values()),
            "max_single_candidate_state_tensor_bytes": max(
                row["search"]["max_single_candidate_state_tensor_bytes"] for row in state_work
            ),
            "tensor_bytes_are_not_peak_process_memory": True,
            "aggregate_model_work": work_accounting(
                model,
                assimilate_samples=len(envs) * HORIZON,
                advance_samples=len(envs) * HORIZON
                + sum(row["search"]["imagined_transitions"] for row in state_work),
            ),
            "auxiliary_teacher_calls": 0,
            "reset_calls": 0,
        }
        artifacts.write(out / "state-work.json", work)
        timing = {
            "setup_seconds": setup,
            "decision_seconds": decisions,
            "native_step_seconds": native_times,
            "row_wall_seconds": time.perf_counter() - begin,
            "observation_assimilations": len(envs) * HORIZON,
            "executed_action_advances": len(envs) * HORIZON,
        }
        artifacts.write(out / "timings.json", timing)
        check_cap(deadline)
        return timing
    except BaseException as error:
        original_error = error
        try:
            if initialized and not (out / "episodes.npz").exists():
                _save_partial(out / "partial-episodes", initialized)
            artifacts.write(
                out / "failed.json",
                {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "completed_steps_by_case": [env.step_index for env in initialized],
                    "wall_seconds": time.perf_counter() - begin,
                },
            )
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original interrupt/failure
            error.add_note(
                f"Artifact preservation also failed: {type(preservation_error).__name__}: {preservation_error}"
            )
        raise
    finally:
        close_errors = []
        for env in envs:
            try:
                env.close()
            except BaseException as error:  # noqa: BLE001 - close every environment, then propagate
                close_errors.append(error)
        if close_errors:
            if original_error is not None:
                original_error.add_note(f"Environment close failures: {[str(e) for e in close_errors]}")
            else:
                raise close_errors[0]


@torch.no_grad()
def prediction_record(plan, model, records, deadline=float("inf")):
    """Only policy packets and issued commands enter held-out predictions.

    One-step targets are next observations. hN arrays contain all complete
    N-transition windows; hN_valid means root AND endpoint observed. Auditors
    may additionally derive endpoint-only masks from the saved public records.
    """
    begin = time.perf_counter()
    identity = model_identity(model, plan)
    horizons = plan["prediction_horizons"]
    require(
        isinstance(horizons, (tuple, list))
        and horizons
        and all(type(h) is int and 1 <= h <= HORIZON for h in horizons)
        and tuple(sorted(set(horizons))) == tuple(horizons),
        "Sorted unique complete prediction horizons",
    )
    require(
        len(records) > 0 and all(set(record["policy"]) == {"packets", "commands"} for record in records),
        "Only public record inputs",
    )
    packets = np.stack([record["policy"]["packets"] for record in records])
    commands = np.stack([record["policy"]["commands"] for record in records])
    require(
        packets.shape == (len(records), HORIZON + 1, 8) and commands.shape == (len(records), HORIZON, 2),
        "Complete native prediction histories",
    )
    _public_arrays(packets, commands)
    require((packets[:, 0, 6] == 1).all(), "Observed initial packet")
    packets, commands = torch.from_numpy(packets.copy()), torch.from_numpy(commands.copy())
    state = model.initial(len(records))
    roots, one, rewards = [], [], []
    for step in range(HORIZON):
        check_cap(deadline)
        state = model.assimilate(state, packets[:, step])
        roots.append(state)
        state, angles, reward = model.advance(state, commands[:, step])
        one.append(angles.numpy().copy())
        rewards.append(reward.numpy().copy())
    endpoints = {h: [] for h in horizons}
    imagined_advances = 0
    root_hashes = [canonical_tensor_hash(root) for root in roots]
    for root_index, root in enumerate(roots):
        future = {key: value.clone() for key, value in root.items()}
        for offset in range(1, min(max(horizons), HORIZON - root_index) + 1):
            check_cap(deadline)
            future, angles, _ = model.advance(future, commands[:, root_index + offset - 1])
            imagined_advances += len(records)
            if offset in endpoints:
                endpoints[offset].append(angles.numpy().copy())
        require(
            canonical_tensor_hash(root) == root_hashes[root_index],
            "Prediction imagination mutated a real root",
        )
    values = {"one_step_angles": np.stack(one, 1), "one_step_rewards": np.stack(rewards, 1)}
    observed = packets[:, :, 6].numpy() > 0.5
    for horizon, predictions in endpoints.items():
        values[f"h{horizon}_angles"] = np.stack(predictions, 1)
        values[f"h{horizon}_valid"] = observed[:, :-horizon] & observed[:, horizon:]
    require(all(np.isfinite(value).all() for value in values.values()), "Nonfinite predictions")
    require(model_identity(model, plan) == identity, "Prediction changed model tensors/configuration")
    check_cap(deadline)
    return values, {
        "wall_seconds": time.perf_counter() - begin,
        "model": identity,
        "student_assimilations": len(records) * HORIZON,
        "prefix_advances": len(records) * HORIZON,
        "imagined_advances": imagined_advances,
        "teacher_or_auxiliary_calls": 0,
        "retained_root_state_tensor_bytes": sum(_state_bytes(root) for root in roots),
        "output_array_bytes": sum(value.nbytes for value in values.values()),
        "tensor_bytes_are_not_peak_process_memory": True,
        "model_work": work_accounting(
            model,
            assimilate_samples=len(records) * HORIZON,
            advance_samples=len(records) * HORIZON + imagined_advances,
        ),
    }
