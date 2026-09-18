# SPDX-License-Identifier: GPL-3.0-only
"""Untrained WLDN-plus-pin-factor candidate architecture and a matched control.

The constraint factors are hand-derived absolute pins, not learned chess rules.
WLDN and higher-order factor networks are established methods. This prototype
does not establish architectural novelty, superior quality or legal reasoning.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from openjev.research.wldn_baseline import WLEncoder, WLStep, native_edge_list
from openjev.research.wldn_interventions import validate

VERSION = 'wldn-pin-factor-prototype-v1'
ARMS = ('joint', 'separable')


def pack_factors(records, *, device=None):
    """Pack complete root menus in compact legal-candidate order.

Columns: compact candidate, owner, king, blocker, attacker, edit status.
The caller must align each record menu to the same neural graph/action menu.
"""
    packed = []; candidate = 0
    for record in records:
        for row in record['candidates']:
            for factor in row['factors']:
                if len(factor) != 5:
                    raise ValueError('Expected five pin-factor fields')
                packed.append((candidate, *factor))
            candidate += 1
    return torch.tensor(packed, dtype=torch.long, device=device).reshape(-1, 6)


class ChessPinFactorHead(nn.Module):
    """16,658 stored parameters in both arms, including original WLDN pathways.

Each factor combines three root node states and three encoded node differences
with ownership/edit indicators. The joint arm applies a shared factor MLP to
all roles together. The separable arm keeps only anchored first-order role
terms, using the same tensors and metadata. It needs four factor evaluations,
versus one for joint; parameter equality does not mean equal compute.

The extra32 pooled factor channels reduce the final readout from59 to29 units
to keep total stored parameters near original WLDN's16,638. This changes the
capacity allocation and is not an exact initialization match to that baseline.
"""
    def __init__(self, arm='joint', *, seed=1109):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown pin-factor arm')
        self.arm, self.seed = arm, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = WLEncoder(32, 4, 32, 3)
            self.difference = WLStep(32, 4)
            self.factor = nn.Sequential(nn.Linear(197, 16), nn.ReLU(), nn.Linear(16, 32))
            self.readout = nn.Linear(184, 29)
            self.output = nn.Linear(29, 1, bias=False)
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=min(1/math.sqrt(layer.in_features), .1))
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.output.weight)

    def factor_value(self, values):
        """Role blocks are(root32,delta32), repeated king/blocker/attacker."""
        if values.ndim != 2 or values.shape[-1] != 197:
            raise ValueError('Expected three64-channel roles and five metadata channels')
        if self.arm == 'joint':
            return self.factor(values)
        metadata_only = torch.cat((torch.zeros_like(values[:, :192]), values[:, 192:]), -1)
        result = -2*self.factor(metadata_only)
        for role in range(3):
            mask = torch.zeros_like(values[:, :192]); mask[:, role*64:(role+1)*64] = 1
            result = result+self.factor(torch.cat((values[:, :192]*mask, values[:, 192:]), -1))
        return result

    def pool(self, root, delta, factors, *, reference=False):
        count = len(delta)
        if (root.shape != delta.shape or delta.ndim != 3 or delta.shape[1:] != (64, 32)
                or factors.ndim != 2 or factors.shape[1] != 6 or factors.dtype != torch.long
                or factors.device != delta.device):
            raise ValueError('Invalid pin-factor inputs')
        if factors.numel():
            bounds = factors.new_tensor([count, 2, 64, 64, 64, 3])
            if ((factors < 0) | (factors >= bounds)).any():
                raise ValueError('Pin-factor index out of range')
            if (factors[:, 2] == factors[:, 3]).any() or (factors[:, 2] == factors[:, 4]).any() or (factors[:, 3] == factors[:, 4]).any():
                raise ValueError('A witness requires three distinct squares')
        if not len(factors):
            return delta.new_zeros(count, 32)
        if reference:
            outputs = []
            for candidate in range(count):
                total = delta.new_zeros(32)
                for row in factors.tolist():
                    owner, color, king, blocker, attacker, status = row
                    if owner != candidate:
                        continue
                    roles = [torch.cat((root[owner, s], delta[owner, s])) for s in (king, blocker, attacker)]
                    metadata = delta.new_tensor([int(color == x) for x in range(2)]+[int(status == x) for x in range(3)])
                    x = torch.cat((*roles, metadata))[None]
                    if self.arm == 'joint':
                        value = self.factor(x)
                    else:
                        baseline = torch.cat((torch.zeros_like(x[:, :192]), x[:, 192:]), -1)
                        value = -2*self.factor(baseline)
                        for role in range(3):
                            parts = [x[:, k*64:(k+1)*64] if k == role else torch.zeros_like(x[:, k*64:(k+1)*64]) for k in range(3)]
                            value = value+self.factor(torch.cat((*parts, x[:, 192:]), -1))
                    total = total+value[0]
                outputs.append(total)
            return torch.stack(outputs)
        owner, color, king, blocker, attacker, status = factors.T
        roles = [torch.cat((root[owner, square], delta[owner, square]), -1) for square in (king, blocker, attacker)]
        metadata = torch.cat((F.one_hot(color, 2), F.one_hot(status, 3)), -1).to(delta.dtype)
        values = self.factor_value(torch.cat((*roles, metadata), -1))
        return delta.new_zeros(count, 32).index_add(0, owner, values)

    def forward(self, hidden, actions, candidates, mask, base, root_edges, child_edges, factors, *, reference=False):
        args = (hidden, actions, candidates, mask, base, root_edges, child_edges)
        owner, slot, child_edges = validate(self, args)
        root_index, root_flags = native_edge_list(root_edges, hidden.dtype)
        child_index, child_flags = native_edge_list(child_edges, hidden.dtype)
        before = self.encoder(hidden, root_index, root_flags)
        after = self.encoder(hidden[owner], child_index, child_flags)
        delta = after-before[owner]
        graph = self.difference(delta, child_index, child_flags).sum(1)
        pins = self.pool(before[owner], delta, factors, reference=reference)
        correction = self.output(self.readout(torch.cat((graph, pins, actions[owner, slot]), -1)).relu()).flatten()
        batch, moves = mask.shape
        dense = base.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base+dense).masked_fill(~mask, -torch.inf)
