# SPDX-License-Identifier: GPL-3.0-only
"""Research adaptation of the published Weisfeiler-Lehman Difference Network.

Reference: Coley et al., Chemical Science 2019, doi:10.1039/C8SC04228D.
The selected reference is connorcoley/rexgen_direct at
5905475ae279a49fb318d7cf8d33a93b9a0c4e49, rank_diff_wln/models.py and
nntrain_direct_useScores.py. Its GPL license is retained in
evidence/chess-wldn-reference-v1/sources/LICENSE. This module is separately
identified as GPL research code; it is not part of a claimed MIT-only release.

The reference's shared node encoding, nodewise subtraction, subsequent graph
propagation and sum readout are retained. Chess adaptations use frozen root
features for both branches, four directed/color attack flags, every legal move,
an action-feature readout and a zero-initialized correction to the backbone.
This is not an official TensorFlow reproduction or a novel model family.
"""
import math

import torch
from torch import nn

VERSION = 'wldn-root-feature-chess-adaptation-v1'


def check_sparse(nodes, index, features, node_width, edge_width):
    if nodes.ndim != 3 or nodes.shape[-1] != node_width or not nodes.shape[0] or not nodes.shape[1]:
        raise ValueError('Invalid node features')
    if index.ndim != 2 or index.shape[0] != 3 or index.dtype != torch.long:
        raise ValueError('Expected batch, receiving node, neighbor indices')
    if features.shape != (index.shape[1], edge_width):
        raise ValueError('Invalid edge feature shape')
    if features.device != nodes.device or index.device != nodes.device or features.dtype != nodes.dtype:
        raise ValueError('Graph dtype or device mismatch')
    if not torch.isfinite(nodes).all() or not torch.isfinite(features).all():
        raise ValueError('Nonfinite graph features')
    if index.numel() and ((index < 0).any() or (index[0] >= len(nodes)).any()
                          or (index[1:] >= nodes.shape[1]).any()):
        raise ValueError('Graph index out of range')


class WLStep(nn.Module):
    """One tied WL node update; edge order affects only floating-point summation."""
    def __init__(self, width, edge_width):
        super().__init__()
        self.width, self.edge_width = width, edge_width
        self.message = nn.Linear(width+edge_width, width)
        self.update = nn.Linear(2*width, width)

    def forward(self, nodes, index, features):
        check_sparse(nodes, index, features, self.width, self.edge_width)
        owner, receiving, neighbor = index
        messages = self.message(torch.cat((nodes[owner, neighbor], features), -1)).relu()
        total = nodes.new_zeros(nodes.shape[0]*nodes.shape[1], self.width)
        total.index_add_(0, owner*nodes.shape[1]+receiving, messages)
        return self.update(torch.cat((nodes, total.reshape_as(nodes)), -1)).relu()


class WLEncoder(nn.Module):
    def __init__(self, input_width, edge_width, width, steps):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (input_width, edge_width, width, steps)):
            raise ValueError('Positive integer dimensions and steps required')
        self.input_width, self.edge_width, self.steps = input_width, edge_width, steps
        self.embedding = nn.Linear(input_width, width, bias=False)
        self.step = WLStep(width, edge_width)

    def forward(self, nodes, index, features):
        check_sparse(nodes, index, features, self.input_width, self.edge_width)
        hidden = self.embedding(nodes).relu()
        for _ in range(self.steps): hidden = self.step(hidden, index, features)
        return hidden


def native_edge_list(relations, dtype):
    """Four flags per ordered square pair, preserving overlapping relation types.

Rows receive features from their flagged neighbors. Flags are own outgoing,
own incoming, opponent outgoing and opponent incoming attacks. A pair occurs
once with a multi-hot feature vector, not once per active flag. Empty rows have
no messages; there are no implicit self-loops or degree normalization.
"""
    if relations.ndim != 4 or relations.shape[1:] != (2, 64, 64):
        raise ValueError('Expected native square attack graphs')
    if not ((relations == 0) | (relations == 1)).all() or relations.diagonal(dim1=-2, dim2=-1).any():
        raise ValueError('Expected binary attacks without self edges')
    flags = torch.stack((relations[:, 0], relations[:, 0].transpose(1, 2),
                         relations[:, 1], relations[:, 1].transpose(1, 2)), -1)
    active = flags.bool().any(-1)
    return active.nonzero().T.contiguous(), flags[active].to(dtype)


class ChessWLDNHead(nn.Module):
    """16,638 parameter candidate ranker, with one shared root encoding per board.

Three tied representation updates precede nodewise child-minus-root subtraction.
One separate difference update uses the child graph, followed by a sum over
all64 squares and a59-unit readout with the original120 action features.
Unlike scalar graph contrast, the published difference encoder includes edge
features and biases, so identical graph branches do not require a zero learned
score. Only the fresh zero output preserves the original policy exactly.
"""
    def __init__(self, *, seed=1109):
        super().__init__()
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = WLEncoder(32, 4, 32, 3)
            self.difference = WLStep(32, 4)
            self.readout = nn.Linear(152, 59)
            self.output = nn.Linear(59, 1, bias=False)
            # Reference-style normal affine weights and zero biases. The final
            # output is separately zeroed to preserve this study's backbone.
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=min(1./math.sqrt(layer.in_features), .1))
                    if layer.bias is not None: nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.output.weight)

    def forward(self, hidden, action_features, candidates, legal_mask, base_logits,
                root_edges, child_edges):
        if hidden.ndim != 3 or hidden.shape[1:] != (64, 32):
            raise ValueError('Expected frozen width32 root features')
        batch = len(hidden)
        if candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[2] != 5:
            raise ValueError('Invalid candidate features')
        moves = candidates.shape[1]
        if (candidates.dtype != torch.long or ((candidates[..., :2] < 0) | (candidates[..., :2] >= 64)).any()
                or legal_mask.shape != (batch, moves) or legal_mask.dtype != torch.bool
                or not legal_mask.any(-1).all() or base_logits.shape != (batch, moves)
                or action_features.shape != (batch, moves, 120)
                or root_edges.shape != (batch, 2, 64, 64)):
            raise ValueError('Invalid features or legal menu')
        owner, slot = legal_mask.nonzero(as_tuple=True)
        if child_edges.shape == (batch, moves, 2, 64, 64): child_edges = child_edges[owner, slot]
        if child_edges.shape != (len(owner), 2, 64, 64):
            raise ValueError('Invalid compact child graphs')
        parameter = next(self.parameters())
        if any(x.device != parameter.device for x in (hidden, action_features, candidates, legal_mask, base_logits, root_edges, child_edges)):
            raise ValueError('Device mismatch')
        if any(x.dtype != parameter.dtype for x in (hidden, action_features, base_logits)):
            raise ValueError('Dtype mismatch')
        if not torch.isfinite(action_features).all() or not torch.isfinite(base_logits[legal_mask]).all():
            raise ValueError('Nonfinite action features or logits')
        root_index, root_features = native_edge_list(root_edges, hidden.dtype)
        child_index, child_features = native_edge_list(child_edges, hidden.dtype)
        root = self.encoder(hidden, root_index, root_features)
        child = self.encoder(hidden[owner], child_index, child_features)
        delta = self.difference(child-root[owner], child_index, child_features).sum(1)
        selected = torch.cat((delta, action_features[owner, slot]), -1)
        correction = self.output(self.readout(selected).relu()).flatten()
        dense = base_logits.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base_logits+dense).masked_fill(~legal_mask, -torch.inf)
