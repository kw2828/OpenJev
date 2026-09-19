"""Known actuator-cost residual for the conventional public-input GRU.

This is an engineering ablation, not a novel architecture. The environment
clips the issued command, adds independent zero-mean Gaussian actuator noise,
then clips the applied action. Its control penalty is the applied squared norm
with weight one. Only the known noise standard deviation and issued command
enter the analytical term; no realized noise or component reward labels enter.
The unchanged sequence loss still supervises the total executed native reward.
"""

from __future__ import annotations

import math
import sys

import torch
from torch import Tensor

from openjev.research.reacher_world_models import GRUWorldModel, State


def _validate_noise_std(noise_std: float) -> float:
    try:
        value = float(noise_std)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("noise_std must be a finite nonnegative scalar") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("noise_std must be a finite nonnegative scalar")
    return value


def expected_clipped_action_cost(command: Tensor, noise_std: float) -> Tensor:
    """Return E[sum(clip(clip(command) + noise, -1, 1)**2)] over the last axis.

    Commands must be finite floating tensors with last dimension two. The
    result has the input dtype/device and remains differentiable in commands,
    including through their initial clipping. ``noise_std`` is known, scalar
    configuration, not a trainable input. No random numbers are consumed.

    Normal partial moments give the closed form. Computation uses float64 for
    tail cancellation. Above standard deviation eight, an integrated Gaussian
    power series avoids subtraction of large moments (the first omitted term
    is below 2e-20). Below sqrt(float64 tiny), the deterministic machine limit
    avoids undefined intermediate divisions. These are numerical evaluation
    branches, not learned approximations or sampled estimates. Float64 support
    is required by the tensor backend; this pilot targets CPU.
    """
    sigma = _validate_noise_std(noise_std)
    if not isinstance(command, Tensor) or not command.is_floating_point():
        raise ValueError("command must be a floating tensor")
    if command.ndim < 1 or command.shape[-1] != 2:
        raise ValueError("command must have last dimension two")
    if not torch.isfinite(command).all():
        raise ValueError("command must be finite")
    issued = command.to(torch.float64).clamp(-1.0, 1.0)
    if sigma < math.sqrt(sys.float_info.min):
        per_action = issued.square()
    elif sigma > 8.0:
        # 1 - E[(1-X**2) 1{|X|<1}]. Expand the Gaussian density on [-1, 1].
        # Integral (1-x**2)*(x-u)**(2k) is a sum of nonnegative even moments.
        inv_sigma = 1.0 / sigma
        correction = torch.zeros_like(issued)
        coefficient = inv_sigma / math.sqrt(2.0 * math.pi)
        for k in range(9):
            moment = torch.zeros_like(issued)
            for power in range(0, 2 * k + 1, 2):
                moment = moment + (
                    math.comb(2 * k, power) * 4.0 / ((power + 1) * (power + 3))
                    * issued.pow(2 * k - power)
                )
            correction = correction + coefficient * moment
            coefficient *= -(inv_sigma * inv_sigma) / (2.0 * (k + 1))
        per_action = 1.0 - correction
    else:
        a = ((-1.0 - issued) / sigma).clamp(-38.0, 38.0)
        b = ((1.0 - issued) / sigma).clamp(-38.0, 38.0)
        probability = 0.5 * (torch.erf(b / math.sqrt(2.0)) - torch.erf(a / math.sqrt(2.0)))
        phi_a = torch.exp(-0.5 * a.square()) / math.sqrt(2.0 * math.pi)
        phi_b = torch.exp(-0.5 * b.square()) / math.sqrt(2.0 * math.pi)
        per_action = (
            (issued.square() + sigma * sigma) * probability
            + 2.0 * issued * sigma * (phi_a - phi_b)
            + sigma * sigma * (a * phi_a - b * phi_b)
            + 1.0 - probability
        )
    return per_action.clamp(0.0, 1.0).sum(-1).to(command.dtype)


class GRUResidualRewardWorldModel(GRUWorldModel):
    """Exactly the base GRU parameters plus a parameter-free reward skip.

    ``residual_reward=False`` gives the base forward behavior exactly. With
    True, the reward head learns the residual after removing the known expected
    actuator cost. Calling the base constructor once preserves parameter names,
    count, ordering, and seeded initialization in both arms. Store noise_std and
    residual_reward in the experiment configuration: they are Python scalar
    settings, deliberately absent from the compatible parameter state dict.
    """

    def __init__(self, hidden_size: int = 64, dt: float = 0.02, *,
                 noise_std: float = 0.05, residual_reward: bool = True) -> None:
        noise_std = _validate_noise_std(noise_std)
        if not isinstance(residual_reward, bool):
            raise TypeError("residual_reward must be a boolean")
        super().__init__(hidden_size=hidden_size, dt=dt)
        self.noise_std = noise_std
        self.residual_reward = residual_reward

    def advance(self, state: State, action: Tensor) -> tuple[State, Tensor, Tensor]:
        next_state, angles, reward = super().advance(state, action)
        if self.residual_reward:
            reward = reward - expected_clipped_action_cost(action, self.noise_std)
        return next_state, angles, reward
