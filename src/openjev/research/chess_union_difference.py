# SPDX-License-Identifier: GPL-3.0-only
"""Experimental union/edit difference layer on the established WLDN encoder.

This combines established nodewise difference and reaction-union ingredients;
no algorithmic novelty or quality advantage is claimed by implementation.
The rotated control preserves union flags and per-candidate/channel edit counts,
but changes which ordered square pairs receive the edit classes. Unlike a global
class renaming it cannot be undone by permuting the three input weight blocks.
"""
import math
import torch
from torch import nn

from openjev.research.wldn_baseline import WLEncoder, WLStep, native_edge_list
from openjev.research.wldn_interventions import validate

ARMS = ('child', 'union', 'edits', 'rotated')
VERSION = 'wldn-union-edit-difference-v1'


def relation_flags(raw):
    # Validation and the direction convention are shared with the frozen WLDN.
    native_edge_list(raw, torch.float32)
    return torch.stack((raw[:, 0], raw[:, 0].transpose(1, 2),
                        raw[:, 1], raw[:, 1].transpose(1, 2)), -1).bool()


def rotate_classes(labels):
    """Rotate nonzero classes by ceil(n/2) in each candidate/direction group.

labels has Cx64x64x4 integer entries:0 absent,1 retained,2 added,3 removed.
Stable square-pair order deliberately anchors this negative control to board
coordinates; it is not an equivariant alternative architecture. Empty/singleton
or homogeneous groups can remain unchanged. No cross-candidate mixing.
"""
    if labels.ndim != 4 or labels.shape[1:] != (64, 64, 4) or labels.dtype != torch.uint8:
        raise ValueError('Expected Cx64x64x4 byte edit classes')
    if (labels > 3).any(): raise ValueError('Unknown edit class')
    result = labels.clone()
    for relation in range(4):
        x = labels[..., relation]; owner, receiving, neighbor = x.nonzero(as_tuple=True)
        counts = torch.bincount(owner, minlength=len(labels))
        starts = counts.cumsum(0)-counts
        rank = torch.arange(len(owner), device=labels.device)-starts[owner]
        donor = starts[owner]+(rank+(counts[owner]+1)//2) % counts[owner]
        result[owner, receiving, neighbor, relation] = x[owner[donor], receiving[donor], neighbor[donor]]
    return result


def difference_edges(root, child, arm, dtype):
    if arm not in ARMS: raise ValueError('Unknown difference arm')
    if root.shape != child.shape or root.device != child.device:
        raise ValueError('Paired compact root/child graphs required')
    r, c = relation_flags(root), relation_flags(child)
    if arm in ('child', 'union'):
        flags = c if arm == 'child' else r | c
        dense = torch.cat((flags, torch.zeros_like(flags), torch.zeros_like(flags)), -1)
    else:
        labels = (r & c).to(torch.uint8)+2*(c & ~r).to(torch.uint8)+3*(r & ~c).to(torch.uint8)
        if arm == 'rotated': labels = rotate_classes(labels)
        dense = torch.cat((labels == 1, labels == 2, labels == 3), -1)
    active = dense.any(-1)
    return active.nonzero().T.contiguous(), dense[active].to(dtype)


def reference_edges(root, child, arm, dtype):
    """Slow CPU Python extraction, independent of the vectorized packing/rotation."""
    if arm not in ARMS: raise ValueError('Unknown difference arm')
    r, c = root.tolist(), child.tolist(); triples, features = [], []
    for b in range(len(r)):
        values = {}
        for u in range(64):
            for v in range(64):
                before = [r[b][0][u][v], r[b][0][v][u], r[b][1][u][v], r[b][1][v][u]]
                after = [c[b][0][u][v], c[b][0][v][u], c[b][1][u][v], c[b][1][v][u]]
                classes = [1 if x and y else 2 if y else 3 if x else 0 for x, y in zip(before, after)]
                if any(classes): values[u, v] = classes
        if arm == 'rotated':
            for rel in range(4):
                pairs = [pair for pair, value in values.items() if value[rel]]
                n = len(pairs)
                if n:
                    original = [values[pair][rel] for pair in pairs]
                    for j, pair in enumerate(pairs): values[pair][rel] = original[(j+(n+1)//2) % n]
        for (u, v), classes in values.items():
            if arm == 'child': flags = [int(x in (1, 2)) for x in classes]+[0]*8
            elif arm == 'union': flags = [int(x != 0) for x in classes]+[0]*8
            else: flags = [int(x == role) for role in (1, 2, 3) for x in classes]
            if any(flags): triples.append([b, u, v]); features.append(flags)
    return (torch.tensor(triples, dtype=torch.long).reshape(-1, 3).T.contiguous(),
            torch.tensor(features, dtype=dtype).reshape(-1, 12))


class ChessUnionDifferenceHead(nn.Module):
    """16,740 stored parameters in every arm, width32,58-unit action readout.

Child/union arms use only4 of12 edge channels. Stored-parameter matching does
not equate active capacity or graph-dependent compute. Every arm shares an
identical initialized state; the arm only changes difference-layer graph input.
"""
    def __init__(self, arm='edits', *, seed=1109):
        super().__init__()
        if arm not in ARMS: raise ValueError('Unknown difference arm')
        self.arm, self.seed = arm, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = WLEncoder(32, 4, 32, 3)
            self.difference = WLStep(32, 12)
            self.readout = nn.Linear(152, 58)
            self.output = nn.Linear(58, 1, bias=False)
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=min(1/math.sqrt(layer.in_features), .1))
                    if layer.bias is not None: nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.output.weight)

    def forward(self, hidden, actions, candidates, mask, base, root, child, *, reference=False):
        args = (hidden, actions, candidates, mask, base, root, child)
        owner, slot, child = validate(self, args)
        root_index, root_flags = native_edge_list(root, hidden.dtype)
        child_index, child_flags = native_edge_list(child, hidden.dtype)
        before = self.encoder(hidden, root_index, root_flags)
        after = self.encoder(hidden[owner], child_index, child_flags)
        pack = reference_edges if reference else difference_edges
        index, flags = pack(root[owner], child, self.arm, hidden.dtype)
        delta = self.difference(after-before[owner], index, flags).sum(1)
        correction = self.output(self.readout(torch.cat((delta, actions[owner, slot]), -1)).relu()).flatten()
        batch, moves = mask.shape
        dense = base.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base+dense).masked_fill(~mask, -torch.inf)
