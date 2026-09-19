"""Small deterministic fit and development-prediction helper, outside studies.

The caller authenticates source identities, initial tensors, full epoch orders
and public data, controls the data split and completes ALL fits before calling
development evaluation. This module cannot certify those external facts. It
never loads a dataset, allocates study seeds, resumes a fit or selects a winner.
Saved checkpoint arithmetic is not an independent numerical training audit.

Every construction uses isolated engineering seed410, then overwrites every
parameter with the supplied named tensors. CPU float32 and one Torch thread
are used locally. Evaluation reports prediction errors, not native returns,
calibrated uncertainty or evidence that a gate improves control.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from openjev.research.reacher_innovation_context import (
    VARIANTS,
    InnovationContextWorldModel,
    capture_module_work,
)
from openjev.research.reacher_innovation_loss import (
    diagonal_residual_moment_score,
    innovation_sequence_loss,
)
from openjev.research.reacher_objective_training import canonical_state_hash, canonical_tensor_hash, clone
from openjev.research.reacher_world_models import _validate_sequence

VERSION = "reacher-innovation-pilot-v1"
RECIPE = {"learning_rate": 0.001, "gradient_clip": {"backbone": 10., "variance_head": 10.}, "rollout_horizon": 5,
          "rollout_weight": 0.5, "reward_scale": 4., "torch_threads": 1}
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
        "foreach": False, "fused": False, "maximize": False, "capturable": False,
        "differentiable": False, "decoupled_weight_decay": False}


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class PilotConfig:
    variant: str
    hidden_size: int
    context_size: int
    dt: float
    noise_std: float
    eta: float
    variance_min: float
    variance_max: float
    epochs: int
    batch_size: int
    variance_score_weight: float

    def validate(self):
        require(self.variant in VARIANTS, "Explicit known variant required")
        require(all(type(x) is int and x > 0 for x in
                    (self.hidden_size, self.context_size, self.epochs, self.batch_size)), "Positive integer sizes")
        require(all(type(x) in (float, int) and math.isfinite(x) for x in
                    (self.dt, self.noise_std, self.eta, self.variance_min, self.variance_max, self.variance_score_weight))
                and self.dt > 0 and self.noise_std >= 0 and 0 < self.eta < 1
                and 0 < self.variance_min < self.variance_max and self.variance_score_weight >= 0,
                "Explicit finite model/loss configuration")

    def model_kwargs(self):
        self.validate()
        return {key: getattr(self, key) for key in
                ("variant", "hidden_size", "context_size", "dt", "noise_std", "eta", "variance_min", "variance_max")}


def _check(deadline):
    require(type(deadline) in (int, float) and math.isfinite(deadline), "Explicit finite monotonic deadline required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Innovation pilot deadline exceeded")


def _sha(value):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"), "SHA256 required")
    return value


def _file_sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _json(path, value):
    with path.open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def _save(path, value):
    with path.open("xb") as file:
        torch.save(value, file)


def _members(out, deadline):
    result = {}
    for path in sorted(out.iterdir()):
        _check(deadline)
        require(path.is_file() and not path.is_symlink(), "Plain exclusive artifact files only")
        result[path.name] = {"sha256": _file_sha(path), "bytes": path.stat().st_size}
    _check(deadline)
    return result


@contextmanager
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _configuration(config, sources, runtime):
    require(type(config) is PilotConfig, "Actual PilotConfig required")
    config.validate()
    require(type(sources) is dict and sources and all(isinstance(k, str) and k for k in sources), "Caller-bound sources required")
    for digest in sources.values():
        _sha(digest)
    require(type(runtime) is dict and runtime, "Caller-bound runtime required")
    return {"version": VERSION, "config": asdict(config), "recipe": RECIPE, "adam": ADAM,
            "source_sha256": dict(sources), "runtime": clone(runtime), "source_authentication": "caller responsibility",
            "constructor_seed": 410, "constructor_tensors_overwritten": True}


def _data(public_data, config, expected_sha):
    require(type(public_data) is dict and set(public_data) == {"packets", "commands", "rewards"}, "Exact public tensor fields")
    require(all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and v.dtype == torch.float32
                and not v.requires_grad and bool(torch.isfinite(v).all()) for v in public_data.values()),
            "Finite detached CPU float32 public data required")
    require(canonical_tensor_hash(public_data) == _sha(expected_sha), "Public data identity")
    data = clone(public_data)
    p, a, r = (data[k] for k in ("packets", "commands", "rewards"))
    _validate_sequence(p, a, r)
    require(5 <= a.shape[1] <= 50 and bool((a.abs() <= 1).all()), "Five to fifty already-clipped issued actions")
    require(bool((p[:, 0, 6] == 1).all()) and torch.equal(p[..., 4:6], p[:, :1, 4:6].expand_as(p[..., 4:6])),
            "Visible startup and static public target required")
    valid = p[..., 6] == 1
    index = torch.arange(p.shape[1])[None].expand(len(p), -1)
    latest = torch.where(valid, index, -1).cummax(1).values
    require(bool((p[..., 7][valid] == 0).all())
            and bool(torch.isclose(p[..., 7], (index - latest).float() * config.dt, rtol=1e-5, atol=1e-6).all()),
            "Actual public observation ages required")
    require(bool((p[..., :4][~valid] == 0).all()), "Missing public angles must be zero")
    return data


def _model(config, weights, expected_sha):
    require(canonical_tensor_hash(weights) == _sha(expected_sha), "Named weight identity")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        model = InnovationContextWorldModel(**config.model_kwargs()).cpu().float()
    expected = model.state_dict()
    require(set(weights) == set(expected), "Exact initial parameter membership")
    for name, value in weights.items():
        require(value.device.type == "cpu" and value.dtype == torch.float32
                and value.shape == expected[name].shape and not value.requires_grad,
                "Exact detached initial parameter schema: " + name)
    model.load_state_dict(weights, strict=True)
    require(canonical_tensor_hash(model.state_dict()) == expected_sha, "Every constructor parameter overwritten")
    return model


def _work(counts, batch, steps, *, rollout=True):
    advances = batch * (steps + (steps - 4) * 5 if rollout else steps)
    return {"modules": clone(counts), "expected_action_cost_samples": advances,
            "completed_forward_only": True, "excludes": "nonmodule arithmetic, validation, copies, backward and Adam"}


def _checkpoint(model, optimizer, binding, counters):
    return {"version": VERSION, "binding": binding, "model_configuration": model.configuration(),
            "weights": clone(model.state_dict()), "optimizer": clone(optimizer.state_dict()) if optimizer else None,
            "parameter_names": list(dict(model.named_parameters())), "counts": dict(counters), "resume_authorized": False}


def _failed(out, error, begin, phase, counters, model=None, optimizer=None, binding=None):
    # Each preservation action is independent and cannot hide the original error.
    for action in (
        lambda: (out / "completed.json").rename(out / "invalid-completion.json") if (out / "completed.json").exists() else None,
        lambda: _save(out / "partial-checkpoint.pt", _checkpoint(model, optimizer, binding, counters)) if model else None,
        lambda: _json(out / "failed.json", {"status": "failed", "phase": phase, "error": repr(error),
            "exception_type": type(error).__name__, "counts": counters,
            "wall_seconds": time.monotonic() - begin, "automatic_retry": False, "resume_authorized": False}),
    ):
        try:
            action()
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note(f"Pilot failure preservation error: {preservation_error!r}")


def fit_one(initial_weights, epoch_orders, public_data, out, *, config, source_sha256, runtime,
            expected_initial_sha256, expected_orders_sha256, expected_data_sha256, deadline):
    """Fit once using exact supplied orders; refuse existing output, never resume."""
    begin, out = time.monotonic(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    counters = {"attempted_updates": 0, "optimizer_steps": 0, "flushed_updates": 0}
    model = optimizer = binding = None
    phase = "admission"
    try:
        _check(deadline)
        binding = _configuration(config, source_sha256, runtime) | {
            "initial_sha256": _sha(expected_initial_sha256), "orders_sha256": _sha(expected_orders_sha256),
            "data_sha256": _sha(expected_data_sha256)}
        _json(out / "started.json", {"status": "started", **binding, "deadline_monotonic": deadline,
                                    "automatic_retry": False, "evaluation_performed": False})
        data = _data(public_data, config, expected_data_sha256)
        count, steps = data["commands"].shape[:2]
        require(isinstance(epoch_orders, torch.Tensor) and epoch_orders.device.type == "cpu"
                and epoch_orders.dtype == torch.int64 and epoch_orders.shape == (config.epochs, count), "Full epoch order tensor")
        require(canonical_tensor_hash({"orders": epoch_orders}) == expected_orders_sha256, "Full supplied order identity")
        require(torch.equal(epoch_orders.sort(dim=1).values, torch.arange(count).expand(config.epochs, -1)),
                "Every epoch must be a complete permutation")
        orders = epoch_orders.clone()
        with _one_thread():
            model = _model(config, initial_weights, expected_initial_sha256).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE["learning_rate"], **ADAM)
            # Separate clipping prevents auxiliary variance gradients from
            # indirectly changing backbone updates through a shared norm.
            groups = {"backbone": [p for name, p in model.named_parameters() if not name.startswith("variance_head.")],
                      "variance_head": list(model.variance_head.parameters())}
            _save(out / "initial-weights.pt", clone(model.state_dict()))
            _save(out / "epoch-orders.pt", orders)
            with (out / "training.jsonl").open("x") as log:
                for epoch in range(config.epochs):
                    for batch, start in enumerate(range(0, count, config.batch_size)):
                        _check(deadline)
                        phase, tick = "training", time.monotonic()
                        counters["attempted_updates"] += 1
                        indices = orders[epoch, start:start + config.batch_size]
                        values = {name: value[indices] for name, value in data.items()}
                        optimizer.zero_grad(set_to_none=True)
                        with capture_module_work(model) as work:
                            loss, metrics = innovation_sequence_loss(model, **values,
                                variance_score_weight=config.variance_score_weight,
                                rollout_horizon=5, rollout_weight=0.5, reward_scale=4.)
                        require(bool(torch.isfinite(loss)) and all(math.isfinite(v) for v in metrics.values()), "Finite loss components")
                        _check(deadline)
                        loss.backward()
                        norms = {name: torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True, foreach=False)
                                 for name, parameters in groups.items()}
                        _check(deadline)
                        optimizer.step()
                        counters["optimizer_steps"] += 1
                        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite updated parameters")
                        row = {"epoch": epoch, "batch": batch, "update": counters["optimizer_steps"],
                            "indices": indices.tolist(), "indices_sha256": canonical_tensor_hash({"indices": indices}),
                            "metrics": metrics, "gradient_norm_before_clip": {name: float(norm) for name, norm in norms.items()},
                            "gradient_clipped": {name: bool(norm > 10.) for name, norm in norms.items()},
                            "gradient_clip": RECIPE["gradient_clip"], "work": _work(work, len(indices), steps),
                            "update_seconds_before_log": time.monotonic() - tick,
                            "fit_elapsed_seconds_before_log": time.monotonic() - begin}
                        log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                        counters["flushed_updates"] += 1
                        _check(deadline)
            phase = "final-artifacts"
            _save(out / "weights.pt", clone(model.state_dict()))
            checkpoint = _checkpoint(model, optimizer, binding, counters)
            _save(out / "checkpoint.pt", checkpoint)
            expected_updates = config.epochs * math.ceil(count / config.batch_size)
            require(all(v == expected_updates for v in counters.values()), "All supplied batches completed and flushed")
            final_hash = canonical_tensor_hash(model.state_dict())
        files = _members(out, deadline)
        completed = {"status": "completed", **binding, "counts": counters, "episodes": count, "steps": steps,
            "final_weights_sha256": final_hash, "checkpoint_sha256": canonical_state_hash(checkpoint),
            "wall_seconds": time.monotonic() - begin, "files": files,
            "timing_scope": "Whole call through hashing; excludes own terminal write, checked against deadline afterward",
            "evaluation_performed": False, "automatic_retry": False, "resume_authorized": False}
        _json(out / "completed.json", completed)
        _check(deadline)
        return completed
    except BaseException as error:
        _failed(out, error, begin, phase, counters, model, optimizer, binding)
        raise


def _metric(error, mask):
    values = error[mask]
    return float(values.mean()) if values.numel() else None


def _episode_metrics(predictions, data, config):
    p, commands, rewards = (data[key] for key in ("packets", "commands", "rewards"))
    steps, starts = commands.shape[1], commands.shape[1] - 4
    target_valid = p[:, 1:, 6] == 1
    reacquired = target_valid & (p[:, :-1, 6] == 0)
    root_valid = p[:, :starts, 6] == 1
    future = torch.stack([p[:, offset + 1:offset + starts + 1] for offset in range(5)], 2)
    future_rewards = torch.stack([rewards[:, offset:offset + starts] for offset in range(5)], 2)
    future_valid = (future[..., 6] == 1) & root_valid[..., None]
    recovery_roots = torch.zeros_like(root_valid)
    recovery_roots[:, 1:] = root_valid[:, 1:] & (p[:, :starts - 1, 6] == 0)
    recovery_errors = (predictions["open_loop_mean"][:, :, [0, 2]] - future[:, :, [0, 2], :4]).square().mean(-1)
    recovery_valid = recovery_roots[..., None] & (future[:, :, [0, 2], 6] == 1)
    predictions.update({"one_step_target_valid": target_valid, "reacquisition_prior_mask": reacquired,
        "post_reacquisition_root_mask": recovery_roots, "post_reacquisition_target_valid": recovery_valid,
        "post_reacquisition_squared_error": recovery_errors})
    rows = []
    for case in range(len(p)):
        error = (predictions["one_step_mean"][case] - p[case, 1:, :4]).square().mean(-1)
        score, moment = diagonal_residual_moment_score(predictions["one_step_mean"][case],
            predictions["one_step_variance"][case], p[case, 1:],
            variance_min=config.variance_min, variance_max=config.variance_max)
        visible = int(target_valid[case].sum())
        total_reacquisitions = int(reacquired[case].sum())
        complete_roots = recovery_valid[case].all(-1)
        full_coverage = total_reacquisitions == 2 and int(complete_roots.sum()) == 2
        per_horizon = [_metric(recovery_errors[case, :, offset], recovery_valid[case, :, offset]) for offset in range(2)]
        rows.append({"episode": case, "one_step_angle_mse": _metric(error, target_valid[case]),
            "one_step_reward_mse": float((predictions["one_step_reward"][case] - rewards[case]).square().mean()),
            "reacquisition_prior_angle_mse": _metric(error, reacquired[case]),
            "post_reacquisition_h1_mse": per_horizon[0], "post_reacquisition_h3_mse": per_horizon[1],
            "post_reacquisition_mean_mse": float(recovery_errors[case, complete_roots].mean()) if full_coverage else None,
            "post_reacquisition_complete_two_roots": full_coverage,
            "open_loop_angle_mse": _metric((predictions["open_loop_mean"][case] - future[case, ..., :4]).square().mean(-1), future_valid[case]),
            "open_loop_reward_mse": _metric((predictions["open_loop_reward"][case] - future_rewards[case]).square(),
                                            root_valid[case, :, None].expand(starts, 5)),
            "residual_moment_score": float(score) if visible else None,
            "variance_lower_bound_fraction": moment["variance_lower_bound_fraction"] if visible else None,
            "variance_upper_bound_fraction": moment["variance_upper_bound_fraction"] if visible else None,
            "mean_predicted_residual_variance": moment["mean_predicted_residual_variance"] if visible else None,
            "counts": {"one_step_actions": steps, "visible_targets": visible,
                "reacquisition_targets": total_reacquisitions, "open_loop_windows": starts,
                "post_reacquisition_roots": int(recovery_roots[case].sum()),
                "post_reacquisition_complete_roots": int(complete_roots.sum()),
                "post_reacquisition_h1_targets": int(recovery_valid[case, :, 0].sum()),
                "post_reacquisition_h3_targets": int(recovery_valid[case, :, 1].sum()),
                "visible_open_loop_roots": int(root_valid[case].sum()),
                "visible_open_loop_targets": int(future_valid[case].sum()),
                "open_loop_reward_targets": int(root_valid[case].sum()) * 5,
                "moment_coordinate_targets": visible * 4}})
    return rows


def evaluate_development(weights, public_data, out, *, config, source_sha256, runtime,
                         expected_weights_sha256, expected_data_sha256, deadline):
    """Prediction-only development evaluation; caller enforces all-fit boundary.

    Save all one-step and complete private H5 forecasts. Missing endpoints and
    missing roots follow the training masks; no eligible target yields null and
    a zero count. Reacquisition means visible packet t after missing packet t-1.
    Its prior error is separate from h1/h3 recovery AFTER assimilating t. The
    primary recovery mean requires exactly two complete roots. No final target
    is assimilated, and no future target enters a private forecast.
    """
    begin, out = time.monotonic(), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    phase, model, binding = "admission", None, None
    counters = {"optimizer_steps": 0, "completed_batches": 0}
    try:
        _check(deadline)
        binding = _configuration(config, source_sha256, runtime) | {
            "weights_sha256": _sha(expected_weights_sha256), "data_sha256": _sha(expected_data_sha256)}
        _json(out / "started.json", {"status": "started", "split": "development", **binding,
            "caller_controls_split_and_all_fits_boundary": True, "deadline_monotonic": deadline})
        data = _data(public_data, config, expected_data_sha256)
        count, steps = data["commands"].shape[:2]
        starts = steps - 4
        chunks, workloads = [], []
        with _one_thread(), torch.no_grad():
            model = _model(config, weights, expected_weights_sha256).eval()
            for first in range(0, count, config.batch_size):
                _check(deadline)
                phase = "development-predictions"
                p, a = (data[key][first:first + config.batch_size] for key in ("packets", "commands"))
                batch, state = len(p), model.initial(len(p))
                roots, means, variances, rewards = [], [], [], []
                with capture_module_work(model) as work:
                    for step in range(steps):
                        _check(deadline)
                        state = model.assimilate(state, p[:, step])
                        roots.append(state)
                        state, mean, reward = model.advance(state, a[:, step])
                        means.append(mean)
                        variances.append(state["prior_variance"])
                        rewards.append(reward)
                    imagined = {key: torch.stack([root[key] for root in roots[:starts]], 1).reshape(
                        batch * starts, *roots[0][key].shape[1:]) for key in roots[0]}
                    future_means, future_variances, future_rewards = [], [], []
                    for offset in range(5):
                        _check(deadline)
                        imagined, mean, reward = model.advance(imagined, a[:, offset:offset + starts].reshape(-1, 2))
                        future_means.append(mean.reshape(batch, starts, 4))
                        future_variances.append(imagined["prior_variance"].reshape(batch, starts, 4))
                        future_rewards.append(reward.reshape(batch, starts))
                chunks.append({"one_step_mean": torch.stack(means, 1), "one_step_variance": torch.stack(variances, 1),
                    "one_step_reward": torch.stack(rewards, 1), "open_loop_mean": torch.stack(future_means, 2),
                    "open_loop_variance": torch.stack(future_variances, 2), "open_loop_reward": torch.stack(future_rewards, 2)})
                workloads.append(_work(work, batch, steps))
                counters["completed_batches"] += 1
            predictions = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
            per_episode = _episode_metrics(predictions, data, config)
            require(canonical_tensor_hash(model.state_dict()) == expected_weights_sha256, "Development evaluation changed weights")
        phase = "development-artifacts"
        _save(out / "predictions.pt", predictions)
        summary = {"split": "development", "episodes": count, "steps": steps, "horizon": 5,
            "per_episode": per_episode, "work_by_batch": workloads, "counts": counters,
            "angle_units": "MSE over four cosine/sine coordinates, not radians",
            "moment_scope": "uncalibrated diagonal residual moments; not a joint likelihood or calibrated coverage",
            "reacquisition_scope": "Prior error is separate; primary recovery is equal h1/h3 mean after assimilating each of exactly two returning observations, equally weighted across roots; null if coverage is incomplete",
            "claims": "Prediction-only development diagnostics; no environment return or model selection"}
        _json(out / "summary.json", summary)
        files = _members(out, deadline)
        completed = {"status": "completed", **binding, "split": "development", "counts": counters,
            "files": files, "wall_seconds": time.monotonic() - begin, "new_optimizer_steps": 0,
            "timing_scope": "Whole call through hashing; excludes own terminal write, checked against deadline afterward"}
        _json(out / "completed.json", completed)
        _check(deadline)
        return summary
    except BaseException as error:
        _failed(out, error, begin, phase, counters, model, None, binding)
        raise
