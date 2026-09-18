# SPDX-License-Identifier: GPL-3.0-only
"""Untrained information and capacity controls for the frozen pin-factor head.

The frozen v1 head is imported unchanged. These controls isolate the extra
branch, not all successor information or all interactions in the whole policy.
"""
import math

import torch
from torch import nn

from openjev.research.chess_pin_factor_head import ChessPinFactorHead
from openjev.research.wldn_baseline import native_edge_list
from openjev.research.wldn_interventions import validate

VERSION = 'wldn-pin-factor-controls-v1'
ARMS = ('root_only', 'counts', 'graph_mlp')
PARAMETERS = {'root_only': 16658, 'counts': 16656, 'graph_mlp': 16658}


def validate_factors(factors, count, device):
    """Validate even fields a control will discard, before filtering any row."""
    if factors.ndim != 2 or factors.shape[1] != 6 or factors.dtype != torch.long or factors.device != device:
        raise ValueError('Invalid pin-factor table')
    if len(factors):
        bounds = factors.new_tensor([count, 2, 64, 64, 64, 3])
        if ((factors < 0) | (factors >= bounds)).any():
            raise ValueError('Pin-factor index out of range')
        if any((factors[:, a] == factors[:, b]).any() for a, b in ((2, 3), (2, 4), (3, 4))):
            raise ValueError('A witness requires three distinct squares')


def pin_counts(factors, count, *, dtype, reference=False):
    """Six raw counts: owner-major, then retained/added/removed status.

Inputs must already be validated. Counts intentionally retain successor pin
changes while discarding square identities and all neural role features.
"""
    if reference:
        rows = [[0]*6 for _ in range(count)]
        for candidate, owner, _king, _blocker, _attacker, status in factors.tolist():
            rows[candidate][3*owner+status] += 1
        return torch.tensor(rows, dtype=dtype, device=factors.device).reshape(count, 6)
    indices = 6*factors[:, 0]+3*factors[:, 1]+factors[:, 5]
    return torch.bincount(indices, minlength=6*count).reshape(count, 6).to(dtype)


class ChessPinFactorControl(ChessPinFactorHead):
    """Three separately trained controls; none is a quality result.

root_only keeps root witnesses (retained or removed), resets their status to
retained, and uses root-encoded roles with zero node deltas. Its entire factor
branch is root-only, although the existing WLDN graph/action paths still use
    candidate information. All v1 tensors and dimensions are preserved, but
    1,568 first-layer weights multiply zeroed delta/status channels. Stored
    parameter equality does not match effective capacity for this ablation.

counts maps six counts through 6->80->32 and reads out at width32. It has two
fewer parameters than v1; no unused padding parameters create nominal parity.

graph_mlp adds a conventional 32->28->32 MLP of the existing pooled WLDN graph
vector, with readout width39. It receives no pin-derived information and has
exactly the v1 parameter count. All arms retain identical encoder/difference
initial tensors. Replacement branches use a separate, fixed seed stream.
"""
    def __init__(self, arm, *, seed=1109):
        if arm not in ARMS:
            raise ValueError('Unknown pin-factor control')
        super().__init__('joint', seed=seed)
        self.control = arm
        if arm == 'root_only':
            return
        input_width, factor_width, readout_width = (6, 80, 32) if arm == 'counts' else (32, 28, 39)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(500000+seed)
            self.factor = nn.Sequential(nn.Linear(input_width, factor_width), nn.ReLU(), nn.Linear(factor_width, 32))
            self.readout = nn.Linear(184, readout_width)
            self.output = nn.Linear(readout_width, 1, bias=False)
            for module in (self.factor, self.readout, self.output):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=min(1/math.sqrt(layer.in_features), .1))
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.output.weight)

    def branch(self, root, delta, graph, factors, *, reference=False):
        if root.shape != delta.shape or delta.ndim != 3 or delta.shape[1:] != (64, 32) or graph.shape != (len(delta), 32):
            raise ValueError('Invalid control features')
        validate_factors(factors, len(delta), delta.device)
        if self.control == 'graph_mlp':
            # Reference uses individual affine calls, sharing the neural layers.
            return torch.stack([self.factor(row[None])[0] for row in graph]) if reference else self.factor(graph)
        if self.control == 'counts':
            counts = pin_counts(factors, len(delta), dtype=delta.dtype, reference=reference)
            return torch.stack([self.factor(row[None])[0] for row in counts]) if reference else self.factor(counts)
        if reference:
            kept = [(c, o, k, b, a, 0) for c, o, k, b, a, status in factors.tolist() if status in (0, 2)]
            root_factors = factors.new_tensor(kept).reshape(-1, 6)
        else:
            root_factors = factors[factors[:, 5] != 1].clone()
            root_factors[:, 5] = 0
        return super().pool(root, torch.zeros_like(delta), root_factors, reference=reference)

    def forward(self, hidden, actions, candidates, mask, base, root_edges, child_edges, factors, *, reference=False):
        args = (hidden, actions, candidates, mask, base, root_edges, child_edges)
        owner, slot, child_edges = validate(self, args)
        root_index, root_flags = native_edge_list(root_edges, hidden.dtype)
        child_index, child_flags = native_edge_list(child_edges, hidden.dtype)
        before = self.encoder(hidden, root_index, root_flags)
        after = self.encoder(hidden[owner], child_index, child_flags)
        delta = after-before[owner]
        graph = self.difference(delta, child_index, child_flags).sum(1)
        branch = self.branch(before[owner], delta, graph, factors, reference=reference)
        correction = self.output(self.readout(torch.cat((graph, branch, actions[owner, slot]), -1)).relu()).flatten()
        batch, moves = mask.shape
        dense = base.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base+dense).masked_fill(~mask, -torch.inf)
