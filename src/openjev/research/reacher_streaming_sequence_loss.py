"""Optional fixed-weight streamed form of the existing public sequence loss.

This uses the exact residual GRU parameters and two-observation root function,
not a new model or training recipe. Equations, masks, window order, reductions
and metrics follow ``reacher_world_models.sequence_loss``. Root reuse changes
autograd accumulation order, so equivalent derivatives need not be bitwise
equal and fitted trajectories cannot be assumed identical. No trainer,
checkpoint, registry or frozen-study integration is provided.

Packet T supervises action T-1 and is never assimilated. Real roots 0..T-1
are streamed once; every forecast uses its own private recurrent state. This
does not remove the cost of validation, selection, retained graphs, copies,
backward or Adam. Work counts are completed forward calls/samples, not FLOPs
or measured speed. No timing is performed here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_anchor_streaming import stream_roots
from openjev.research.reacher_two_observation_history import MAX_STEPS
from openjev.research.reacher_world_models import (
    ACTION_SIZE,
    ANGLE_SIZE,
    _masked_mean,
    _validate_sequence,
)


def _settings(rollout_horizon, rollout_weight, reward_scale, kl_weight, kl_balance, free_nats):
    # Preserve the existing loss's scalar validation, including its supported
    # zero weights and horizons longer than the sequence.
    if not isinstance(rollout_horizon, int) or rollout_horizon < 1:
        raise ValueError("rollout_horizon must be a positive integer")
    if any(not math.isfinite(v) or v < 0 for v in (rollout_weight, reward_scale, kl_weight)):
        raise ValueError("loss weights must be nonnegative and finite")
    if not 0 <= kl_balance <= 1 or not math.isfinite(free_nats) or free_nats < 0:
        raise ValueError("invalid KL balance or free nats")


def streaming_sequence_work(batch: int, steps: int, *, rollout_horizon: int = 5,
                            rollout_weight: float = 0.5, residual_reward: bool = True) -> dict[str, int]:
    """Completed forward work for a successful call, excluding backward/Adam.

    The streamer sees T packets and T-1 issued commands, not the final target.
    One-step forecasts preserve the original B-sized batches. Open-loop batches
    contain all B*(T-H+1) roots, including roots masked out of the loss.
    """
    if type(batch) is not int or batch < 1 or type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError("Positive batch and one to fifty actions required")
    if type(residual_reward) is not bool:
        raise ValueError("Boolean residual reward configuration required")
    _settings(rollout_horizon, rollout_weight, 4., 0.01, 0.8, 1.)
    starts = max(steps - rollout_horizon + 1, 0)
    active = bool(rollout_weight) and starts > 0
    root_observations, root_advances = 1 + 3 * (steps - 1), 2 * (steps - 1)
    rollout_calls = rollout_horizon if active else 0
    rollout_samples = batch * starts * rollout_calls
    advance_samples = batch * (root_advances + steps) + rollout_samples
    return {
        "batch_size": batch, "real_roots_per_case": steps, "terminal_assimilations": 0,
        "root_assimilate_calls": root_observations, "root_advance_calls": root_advances,
        "one_step_advance_calls": steps, "rollout_advance_calls": rollout_calls,
        "parent_assimilate_calls": root_observations,
        "parent_advance_calls": root_advances + steps + rollout_calls,
        "observation_update_samples": batch * root_observations,
        "transition_samples": advance_samples, "observation_head_samples": advance_samples,
        "reward_head_samples": advance_samples,
        "expected_action_cost_samples": advance_samples if residual_reward else 0,
        "one_step_samples": batch * steps, "rollout_samples": rollout_samples,
        "posterior_kl_calls": steps, "posterior_kl_samples": batch * steps,
    }


def streaming_sequence_loss(model: GRUResidualRewardWorldModel, packets: Tensor,
                            commands: Tensor, rewards: Tensor, *, rollout_horizon: int = 5,
                            rollout_weight: float = 0.5, reward_scale: float = 4.0,
                            kl_weight: float = 0.01, kl_balance: float = 0.8,
                            free_nats: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    """The original loss interface, with reusable causal two-observation roots.

    Accepts only the exact parent ``GRUResidualRewardWorldModel``. The streamed
    public prefix must satisfy the reviewed twelve-packet bound, fixed target,
    actual observation age and clipped-issued-command contract. Full sequences
    retain the original finite-input requirement, even at masked targets. As
    in the original loss, the terminal target is not another real assimilation.
    """
    _validate_sequence(packets, commands, rewards)
    _settings(rollout_horizon, rollout_weight, reward_scale, kl_weight, kl_balance, free_nats)
    batch, steps, _ = commands.shape
    if steps > MAX_STEPS:
        raise ValueError("At most fifty actions are supported")
    # The final action is forecast below, not part of the root reconstruction.
    # Validate it too, since stream_roots receives only the intervening actions.
    if not bool((commands.abs() <= 1).all()):
        raise ValueError("Issued commands must already be within [-1,1]")
    streamed = stream_roots(model, packets[:, :-1], commands[:, :-1])
    roots = [{key: value[:, step] for key, value in streamed.roots.items()} for step in range(steps)]
    angles, predicted_rewards, kls = [], [], []
    for step, root in enumerate(roots):
        kls.append(model.posterior_kl(root, kl_balance, free_nats))
        _, observation, reward = model.advance(root, commands[:, step])
        angles.append(observation)
        predicted_rewards.append(reward)
    observation_loss = _masked_mean(
        (torch.stack(angles, 1) - packets[:, 1:, :ANGLE_SIZE]).square(), packets[:, 1:, 6:7],
    )
    reward_loss = F.mse_loss(torch.stack(predicted_rewards, 1), rewards)
    valid_posteriors = packets[:, :-1, 6]
    kl_loss = torch.stack(kls, 1).sum() / valid_posteriors.sum().clamp_min(1)
    zero = observation_loss * 0
    rollout_observation, rollout_reward = zero, zero
    starts = steps - rollout_horizon + 1
    if rollout_weight and starts > 0:
        imagined = {
            key: torch.stack([root[key] for root in roots[:starts]], 1).reshape(
                batch * starts, *roots[0][key].shape[1:],
            )
            for key in roots[0]
        }
        observations, predictions, targets, target_rewards, masks = [], [], [], [], []
        prefix_valid = packets[:, :starts, 6].reshape(-1)
        for offset in range(rollout_horizon):
            action = commands[:, offset:offset + starts].reshape(batch * starts, ACTION_SIZE)
            imagined, observation, reward = model.advance(imagined, action)
            observations.append(observation)
            predictions.append(reward)
            targets.append(packets[:, offset + 1:offset + starts + 1, :ANGLE_SIZE].reshape(-1, ANGLE_SIZE))
            target_rewards.append(rewards[:, offset:offset + starts].reshape(-1))
            masks.append(packets[:, offset + 1:offset + starts + 1, 6].reshape(-1) * prefix_valid)
        rollout_observation = _masked_mean(
            (torch.stack(observations, 1) - torch.stack(targets, 1)).square(),
            torch.stack(masks, 1)[..., None],
        )
        rollout_reward = _masked_mean(
            (torch.stack(predictions, 1) - torch.stack(target_rewards, 1)).square(), prefix_valid[:, None],
        )
    loss = observation_loss + reward_scale * reward_loss + kl_weight * kl_loss
    loss = loss + rollout_weight * (rollout_observation + reward_scale * rollout_reward)
    metrics = {
        "loss": float(loss.detach()), "observation_mse": float(observation_loss.detach()),
        "reward_mse": float(reward_loss.detach()), "kl_nats": float(kl_loss.detach()),
        "rollout_observation_mse": float(rollout_observation.detach()),
        "rollout_reward_mse": float(rollout_reward.detach()),
        "valid_observation_targets": float(packets[:, 1:, 6].sum().detach()),
        "valid_rollout_starts": float(packets[:, :max(starts, 0), 6].sum().detach()) if starts > 0 else 0.0,
    }
    return loss, metrics
