"""Small residual correction to analytic GP deletion priorities.

Training uses fresh on-policy groups and a leave-one-out reward baseline.
The policy never receives requests, target exposures or an external archive.
"""
from __future__ import annotations

import numpy as np
import torch


class RetentionPolicy(torch.nn.Module):
    """33 float64 parameters; zero initial correction preserves analytic scores."""

    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(6, 4, dtype=torch.float64)
        self.output = torch.nn.Linear(4, 1, dtype=torch.float64)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, features, kl):
        correction = 2 * torch.tanh(self.output(torch.tanh(self.hidden(features))).squeeze(-1))
        return torch.log(kl + 1e-12) + correction

    @property
    def parameter_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.parameters())

    def arrays(self):
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in self.state_dict().items()}


def leave_one_out_advantage(reward):
    """Independent other-trajectory baseline, fixed reward scale, no fitted critic."""
    if reward.ndim != 2 or reward.shape[1] < 2 or not torch.isfinite(reward).all():
        raise ValueError("finite field-by-trajectory rewards with at least two trajectories")
    baseline = (reward.sum(1, keepdim=True)-reward)/(reward.shape[1]-1)
    return ((reward-baseline)/.05).detach()


def select(policy, statistics, stochastic):
    features = torch.from_numpy(np.array(statistics['features'], copy=True))
    kl = torch.from_numpy(np.array(statistics['kl'], copy=True))
    scores = policy(features, kl)
    if not torch.isfinite(scores).all():
        raise ValueError("finite retention priorities")
    if stochastic:
        distribution = torch.distributions.Categorical(logits=-scores)
        action = distribution.sample()
        return action.detach().numpy(), distribution.log_prob(action), distribution.entropy()
    return scores.detach().numpy().argmin(-1), None, None
