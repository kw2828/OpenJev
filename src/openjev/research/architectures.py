"""Small trainable research modules, not pretrained or deployed controllers.

Inputs are numeric observations and candidate vectors supplied by the experiment.
They do not encode arbitrary English. State is explicit and owned by an episode.
"""

import torch
from torch import nn


class CandidateReadout(nn.Module):
    def __init__(self, hidden, candidate_dim):
        super().__init__()
        self.project = nn.Linear(candidate_dim, hidden, bias=False)
        self.scale = hidden**-0.5

    def forward(self, state, candidates):
        return torch.einsum("bh,bkh->bk", state, self.project(candidates)) * self.scale


class SparseCircuit(nn.Module):
    """Sensory/inter/command/motor motif, NOT a reconstructed connectome or LTC.

    Uses dense masked matrix operations. Sparsity alone does not imply speedup.
    Compare against shuffled masks with exactly the same edge count.
    """

    def __init__(self, observation_dim, candidate_dim, hidden=48, seed=0):
        super().__init__()
        if hidden < 6 or hidden % 3:
            raise ValueError("hidden must be a multiple of three and at least six")
        self.hidden = hidden
        self.input = nn.Linear(observation_dim, hidden // 3)
        self.recurrent = nn.Linear(hidden, hidden)
        mask = torch.zeros(hidden, hidden)
        width = hidden // 3
        mask[width:2 * width, :width] = 1
        mask[2 * width:, width:2 * width] = 1
        generator = torch.Generator().manual_seed(seed)
        mask[width:2 * width, width:2 * width] = (
            torch.rand(width, width, generator=generator) < 0.25
        ).float()
        self.register_buffer("mask", mask)
        self.readout = CandidateReadout(width, candidate_dim)

    def forward(self, observation, candidates, state=None):
        if state is None:
            state = observation.new_zeros(observation.shape[0], self.hidden)
        drive = nn.functional.pad(self.input(observation), (0, self.hidden * 2 // 3))
        # Three synchronous updates carry this observation through the three groups.
        for _ in range(3):
            state = torch.tanh(nn.functional.linear(state, self.recurrent.weight * self.mask,
                                                  self.recurrent.bias) + drive)
        return self.readout(state[:, self.hidden * 2 // 3:], candidates), state


class LatentTransformer(nn.Module):
    """One shared block repeated in depth, with optional explicit temporal state."""

    def __init__(self, observation_dim, candidate_dim, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.observation = nn.Linear(observation_dim, hidden)
        self.candidate = nn.Linear(candidate_dim, hidden)
        self.block = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0,
                                               batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.readout = CandidateReadout(hidden, candidate_dim)

    def forward(self, observation, candidates, state=None, depth=2):
        if not isinstance(depth, int) or not 1 <= depth <= 16:
            raise ValueError("depth must be an integer in [1, 16]")
        current = self.observation(observation)
        if state is not None:
            current = current + state
        tokens = torch.cat([current[:, None], self.candidate(candidates)], dim=1)
        for _ in range(depth):
            tokens = self.block(tokens)
        state = self.norm(tokens[:, 0])
        return self.readout(state, candidates), state


class RecurrentWorldModel(nn.Module):
    """Deterministic GRU dynamics scaffold, not a Dreamer/RSSM reproduction.

    Observe actual transitions; imagine predicts features, reward, and termination
    from state/action without seeing the next observation. Requires training.
    """

    def __init__(self, observation_dim, action_dim, candidate_dim, hidden=48):
        super().__init__()
        self.hidden = hidden
        self.posterior = nn.GRUCell(observation_dim + action_dim, hidden)
        self.transition = nn.GRUCell(action_dim, hidden)
        self.observation_head = nn.Linear(hidden, observation_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)
        self.readout = CandidateReadout(hidden, candidate_dim)

    def forward(self, observation, previous_action, candidates, state=None):
        if state is None:
            state = observation.new_zeros(observation.shape[0], self.hidden)
        state = self.posterior(torch.cat([observation, previous_action], dim=-1), state)
        return self.readout(state, candidates), state

    def imagine(self, state, action):
        future = self.transition(action, state)
        return {
            "state": future,
            "observation": self.observation_head(future),
            "reward": self.reward_head(future).squeeze(-1),
            "termination_logit": self.done_head(future).squeeze(-1),
        }
