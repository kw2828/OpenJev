"""Public-target supervision for a prospective predictive error-scale head.

The diagonal residual moment score estimates E[(y-mu)^2 | public history].
With a biased mean predictor this includes its bias, so it is not automatically
variance around the true conditional mean. It is not a joint Gaussian density
on the constrained cosine/sine manifold, nor evidence of calibrated coverage.
Fixed bounds constrain the target moment estimate. No scientific study uses
this module yet.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from openjev.research.reacher_world_models import _masked_mean, _validate_sequence


def diagonal_residual_moment_score(
    prior_mean: Tensor,
    prior_variance: Tensor,
    public_targets: Tensor,
    *,
    variance_min: float,
    variance_max: float,
) -> tuple[Tensor, dict[str, float]]:
    """Score a pre-observation prediction against actually visible targets.

    Inputs have matching leading dimensions and final widths 4, 4 and 8.
    The caller must construct the prior before assimilating its target. Only
    this loss's variance argument receives gradients: observed targets and
    predicted means are detached. A model must also detach the features entering
    its variance head if auxiliary supervision must not train its backbone.

    The loss is one half the visible-target mean of the sum over coordinates
    of ``log(v) + stopgrad(y-mu)^2 / v``. It can be negative. Missing angular
    placeholders, including NaN/inf, contribute neither information nor loss.
    All-absent targets give a defined zero gradient to variance. Bound fractions
    count values within one percent of each bound's absolute value; they are
    diagnostics, not a test of calibration.
    """
    if any(isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) for v in (variance_min, variance_max)):
        raise ValueError("Finite scalar variance bounds required")
    if not 0 < variance_min < variance_max:
        raise ValueError("Require 0 < variance_min < variance_max")
    if not isinstance(prior_mean, Tensor) or prior_mean.ndim < 2 or prior_mean.numel() == 0:
        raise ValueError("Nonempty batched prior required")
    for value, width in ((prior_mean, 4), (prior_variance, 4), (public_targets, 8)):
        if (not isinstance(value, Tensor)
                or value.dtype not in (torch.float32, torch.float64)
                or value.dtype != prior_mean.dtype or value.device != prior_mean.device
                or value.shape != (*prior_mean.shape[:-1], width)):
            raise ValueError("Matching floating tensor shapes, dtypes and devices required")
    if not bool(torch.isfinite(prior_mean).all() and torch.isfinite(prior_variance).all()):
        raise ValueError("Finite prior mean and variance required")
    if not bool(((prior_variance >= variance_min) & (prior_variance <= variance_max)).all()):
        raise ValueError("Prior variance outside declared bounds")
    valid = public_targets[..., 6] == 1
    if not bool(((public_targets[..., 6] == 0) | valid).all()):
        raise ValueError("Binary public validity required")
    if (not bool(torch.isfinite(public_targets[..., 4:]).all())
            or not bool((public_targets[..., 7] >= 0).all())
            or not bool((public_targets[..., 7][valid] == 0).all())):
        raise ValueError("Finite public metadata and consistent visible age required")
    observed = torch.where(valid[..., None], public_targets[..., :4], 0).detach()
    if not bool(torch.isfinite(observed).all()):
        raise ValueError("Visible observation must be finite")
    error_squared = torch.where(valid[..., None], observed - prior_mean.detach(), 0).square()
    per_target = 0.5 * (prior_variance.log() + error_squared / prior_variance).sum(-1)
    if not bool(torch.isfinite(per_target[valid]).all()):
        raise ValueError("Visible residual moment score overflowed")
    count = valid.sum()
    score = torch.where(valid, per_target, 0).sum() / count.clamp_min(1)
    coordinate_mask = valid[..., None].expand_as(prior_variance)
    coordinate_count = coordinate_mask.sum().clamp_min(1)

    def fraction(condition):
        return float((condition & coordinate_mask).sum().detach() / coordinate_count)

    return score, {
        "residual_moment_score": float(score.detach()),
        "valid_variance_targets": float(count),
        "mean_predicted_residual_variance": float(
            torch.where(coordinate_mask, prior_variance.detach(), 0).sum() / coordinate_count
        ),
        "variance_lower_bound_fraction": fraction(prior_variance <= 1.01 * variance_min),
        "variance_upper_bound_fraction": fraction(prior_variance >= 0.99 * variance_max),
    }


def innovation_sequence_loss(
    model,
    packets: Tensor,
    commands: Tensor,
    rewards: Tensor,
    *,
    variance_score_weight: float,
    rollout_horizon: int = 5,
    rollout_weight: float = 0.5,
    reward_scale: float = 4.0,
) -> tuple[Tensor, dict[str, float]]:
    """Existing mean/reward objectives plus pre-observation error-scale fitting.

    The deterministic innovation-context model has no latent KL objective.
    One-step and open-loop MSE reductions match the existing sequence loss.
    The additional score uses one-step priors from the real public-history
    sequence, including a returned observation after a blackout. Packet T is
    a target only; no terminal assimilation or extra action is performed.
    Imagination computes its variance head but has no auxiliary variance loss.

    The score's weight is deliberately required from the caller. This isolated
    function specifies neither a training protocol nor a calibrated forecast.
    """
    from openjev.research.reacher_innovation_context import InnovationContextWorldModel

    if type(model) is not InnovationContextWorldModel:
        raise ValueError("Exact innovation-context model required")
    _validate_sequence(packets, commands, rewards)
    if type(rollout_horizon) is not int or rollout_horizon < 1:
        raise ValueError("Positive integer rollout horizon required")
    if any(isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) or v < 0
           for v in (variance_score_weight, rollout_weight, reward_scale)):
        raise ValueError("Finite nonnegative loss weights required")
    batch, steps, _ = commands.shape
    state = model.initial(batch, packets.device)
    roots, means, variances, predicted_rewards = [], [], [], []
    for step in range(steps):
        state = model.assimilate(state, packets[:, step])
        roots.append(state)
        state, mean, reward = model.advance(state, commands[:, step])
        means.append(mean)
        variances.append(state["prior_variance"])
        predicted_rewards.append(reward)
    predicted_mean = torch.stack(means, 1)
    observation_loss = _masked_mean(
        (predicted_mean - packets[:, 1:, :4]).square(), packets[:, 1:, 6:7],
    )
    reward_loss = F.mse_loss(torch.stack(predicted_rewards, 1), rewards)
    moment_loss, moment_metrics = diagonal_residual_moment_score(
        predicted_mean, torch.stack(variances, 1), packets[:, 1:],
        variance_min=model.variance_min, variance_max=model.variance_max,
    )
    rollout_observation, rollout_reward = observation_loss * 0, observation_loss * 0
    starts = steps - rollout_horizon + 1
    if rollout_weight and starts > 0:
        imagined = {
            key: torch.stack([root[key] for root in roots[:starts]], 1).reshape(
                batch * starts, *roots[0][key].shape[1:],
            ) for key in roots[0]
        }
        observations, forecasts, targets, target_rewards, masks = [], [], [], [], []
        prefix_valid = packets[:, :starts, 6].reshape(-1)
        for offset in range(rollout_horizon):
            action = commands[:, offset:offset + starts].reshape(batch * starts, 2)
            imagined, mean, reward = model.advance(imagined, action)
            observations.append(mean)
            forecasts.append(reward)
            targets.append(packets[:, offset + 1:offset + starts + 1, :4].reshape(-1, 4))
            target_rewards.append(rewards[:, offset:offset + starts].reshape(-1))
            masks.append(packets[:, offset + 1:offset + starts + 1, 6].reshape(-1) * prefix_valid)
        rollout_observation = _masked_mean(
            (torch.stack(observations, 1) - torch.stack(targets, 1)).square(),
            torch.stack(masks, 1)[..., None],
        )
        rollout_reward = _masked_mean(
            (torch.stack(forecasts, 1) - torch.stack(target_rewards, 1)).square(),
            prefix_valid[:, None],
        )
    prediction_loss = observation_loss + reward_scale * reward_loss
    prediction_loss = prediction_loss + rollout_weight * (rollout_observation + reward_scale * rollout_reward)
    total = prediction_loss + variance_score_weight * moment_loss
    metrics = {
        "loss": float(total.detach()), "prediction_objective": float(prediction_loss.detach()),
        "observation_mse": float(observation_loss.detach()), "reward_mse": float(reward_loss.detach()),
        "rollout_observation_mse": float(rollout_observation.detach()),
        "rollout_reward_mse": float(rollout_reward.detach()),
        "valid_observation_targets": float(packets[:, 1:, 6].sum().detach()),
        "valid_rollout_starts": float(packets[:, :max(starts, 0), 6].sum().detach()) if starts > 0 else 0.,
        **moment_metrics,
    }
    return total, metrics
