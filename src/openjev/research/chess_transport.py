"""Two-endpoint relation transport for a frozen chess policy's move head.

Research hypothesis, not an established novel method. Query-conditioned graph
walks and pair representations have close predecessors in Neural LP, MINERVA
and NBFNet. All relations here come from the root board, never a successor or
teacher label. The model performs internal message propagation, not game search.
"""
import hashlib
import random

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('transport', 'no_overlap', 'uniform', 'static', 'rewired')
VERSION = 'two-endpoint-root-relation-transport-v1'


def root_relations(board):
    """Binary [own/opponent, source, target] geometric attacks, in mover frame."""
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('A valid standard chess board is required')
    canonical = lambda s: s if board.turn else chess.square_mirror(s)
    edges = np.zeros((2, 64, 64), dtype=np.uint8)
    for source, piece in board.piece_map().items():
        side = int(piece.color != board.turn)
        for target in board.attacks(source):
            edges[side, canonical(source), canonical(target)] = 1
    return edges


def degree_rewire(edges, *, seed, swaps_per_edge=5, attempts_per_swap=20):
    """Fixed-attempt directed double-edge swaps, preserving each node's degrees.

No self edges or duplicate edges are introduced. We record achieved swaps and
overlap: small/rigid graphs can remain unchanged. This is not a uniform sample
from all degree-matched graphs. No outcome, legal move or piece feature changes.
"""
    if edges.shape != (2, 64, 64) or not np.isin(edges, [0, 1]).all():
        raise ValueError('Expected binary attack relations')
    if np.diagonal(edges, axis1=1, axis2=2).any():
        raise ValueError('Self attacks are not valid input relations')
    result = edges.copy()
    rng = random.Random(seed)
    records = []
    for relation in range(2):
        pairs = list(zip(*np.nonzero(edges[relation])))
        pairs = [(int(a), int(b)) for a,b in pairs]
        members = set(pairs)
        accepted, attempted = 0, 0
        target = swaps_per_edge * len(pairs)
        if len(pairs) >= 2:
            for _ in range(target * attempts_per_swap):
                if accepted >= target:
                    break
                attempted += 1
                i, j = rng.sample(range(len(pairs)), 2)
                a,b = pairs[i]; c,d = pairs[j]
                if a == c or b == d or a == d or c == b:
                    continue
                if (a,d) in members or (c,b) in members:
                    continue
                members.remove((a,b)); members.remove((c,d))
                members.add((a,d)); members.add((c,b))
                pairs[i], pairs[j] = (a,d), (c,b)
                accepted += 1
        result[relation].fill(0)
        for a,b in pairs:
            result[relation,a,b] = 1
        assert np.array_equal(result[relation].sum(0), edges[relation].sum(0))
        assert np.array_equal(result[relation].sum(1), edges[relation].sum(1))
        records.append({'edges': len(pairs), 'accepted_swaps': accepted,
                        'attempted_swaps': attempted, 'target_swaps': target,
                        'retained_edges': int((result[relation] & edges[relation]).sum())})
    return result, records


def graph_seed(fen):
    # Literal FEN has no labels. The same root gets one fixed rewire in every fit.
    return int.from_bytes(hashlib.sha256(('transport-rewire-v1|' + fen).encode()).digest()[:8], 'big')


def transitions(edges, dtype):
    """Outgoing own/opponent and reverse channels, with absorbing empty rows."""
    raw = edges.to(dtype=dtype)
    raw = torch.stack((raw[:,0], raw[:,0].transpose(1,2),
                       raw[:,1], raw[:,1].transpose(1,2)), dim=1)
    degrees = raw.sum(-1, keepdim=True)
    eye = torch.eye(64, dtype=dtype, device=edges.device)[None,None]
    return raw / degrees.clamp_min(1) + (degrees == 0).to(dtype) * eye


class TransportHead(nn.Module):
    def __init__(self, arm='transport', *, seed=1109, width=32, steps=3):
        super().__init__()
        if arm not in ARMS or type(width) is not int or width < 1 or type(steps) is not int or steps < 1:
            raise ValueError('Invalid transport configuration')
        self.arm, self.width, self.steps, self.seed = arm, width, steps, seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.initialize = nn.Linear(3*width+24, width)
            self.router = nn.Linear(width, 8)
            self.cell = nn.GRUCell(3*width+1, width)
            self.output = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, hidden, action_features, candidates, legal_mask, base_logits, edges, *, trace=False):
        b, n, w = hidden.shape
        l = candidates.shape[1]
        if n != 64 or w != self.width or action_features.shape != (b,l,3*w+24):
            raise ValueError('Invalid node or action feature shape')
        if candidates.shape != (b,l,5) or candidates.dtype != torch.long:
            raise ValueError('Invalid candidate features')
        if legal_mask.shape != (b,l) or legal_mask.dtype != torch.bool or not legal_mask.any(-1).all():
            raise ValueError('Expected nonempty legal menus')
        if base_logits.shape != (b,l) or edges.shape != (b,2,64,64):
            raise ValueError('Invalid base logits or relation shape')
        if ((candidates[...,:2] < 0) | (candidates[...,:2] >= 64)).any():
            raise ValueError('Invalid endpoint')
        parameter = next(self.parameters())
        if any(t.device != parameter.device for t in (hidden, action_features, candidates, legal_mask, base_logits, edges)):
            raise ValueError('Inputs and model must share a device')
        if any(t.dtype != parameter.dtype for t in (hidden, action_features, base_logits)):
            raise ValueError('Feature dtype mismatch')
        if not torch.isfinite(hidden).all() or not torch.isfinite(action_features).all() or not torch.isfinite(base_logits[legal_mask]).all():
            raise ValueError('Nonfinite model inputs')
        if not ((edges == 0) | (edges == 1)).all():
            raise ValueError('Expected binary relations')
        adjacency = transitions(edges, hidden.dtype)
        left = F.one_hot(candidates[...,0], 64).to(hidden.dtype)
        right = F.one_hot(candidates[...,1], 64).to(hidden.dtype)
        query = torch.tanh(self.initialize(action_features))
        history = []
        for _ in range(self.steps):
            gates = self.router(query).reshape(b,l,2,4).softmax(-1)
            if self.arm == 'uniform':
                gates = torch.full_like(gates, .25)
            if self.arm != 'static':
                left = sum(torch.bmm(left, adjacency[:,r]) * gates[:,:,0,r,None] for r in range(4))
                right = sum(torch.bmm(right, adjacency[:,r]) * gates[:,:,1,r,None] for r in range(4))
            overlap = left * right
            mass = overlap.sum(-1, keepdim=True)
            overlap = overlap / mass.clamp_min(1e-8)
            if self.arm == 'no_overlap':
                overlap = torch.zeros_like(overlap)
                mass = torch.zeros_like(mass)
            context = torch.cat((torch.bmm(left, hidden), torch.bmm(right, hidden),
                                 torch.bmm(overlap, hidden), mass), dim=-1)
            query = self.cell(context.reshape(b*l,-1), query.reshape(b*l,w)).reshape(b,l,w)
            if trace:
                history.append({'left': left, 'right': right, 'gates': gates, 'overlap_mass': mass})
        logits = (base_logits + self.output(query).squeeze(-1)).masked_fill(~legal_mask, -torch.inf)
        return (logits, history) if trace else logits


@torch.no_grad()
def candidate_features(backbone, nodes, candidates):
    """Reconstruct the frozen policy's existing move features from cached nodes."""
    batch = torch.arange(len(nodes), device=nodes.device)[:,None]
    features = torch.cat((nodes[batch,candidates[...,0]], nodes[batch,candidates[...,1]],
                          nodes.mean(1)[:,None].expand(-1,candidates.shape[1],-1),
                          backbone.promotion_embedding(candidates[...,2]),
                          backbone.dx_embedding(candidates[...,3]),
                          backbone.dy_embedding(candidates[...,4])), dim=-1)
    return features


@torch.no_grad()
def frozen_features(backbone, observations, candidates, legal_mask):
    """The frozen direct model supplies node features and its unchanged logits."""
    if backbone.arm != 'direct':
        raise ValueError('Only a direct-policy backbone is allowed')
    backbone.eval()
    logits, value, hidden = backbone(observations, candidates, legal_mask)
    nodes = hidden.flatten(2).transpose(1,2)
    return nodes, candidate_features(backbone, nodes, candidates), logits, value
