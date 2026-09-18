"""Shared-weight before/after graph contrast for a frozen chess policy.

Native transitions produce candidate attack graphs in the root-player frame.
The representation at every square remains the frozen ROOT representation.
This is a graph-change hypothesis, not a learned world model, a game-value
counterfactual, or an established novel contribution. Dense operators are the
reference; the earlier sparse-operator screen remains a separate failed study.
"""
import hashlib

import chess
import torch
from torch.nn import functional as F

from openjev.research.chess_counterfactual_transport import fixed_frame_relations
from openjev.research.chess_transport import TransportHead, transitions

ARMS = ('contrast', 'child', 'root', 'permuted_contrast')


def candidate_graphs(boards):
    """Root/child binary attacks, preserving caller boards and all legal moves."""
    boards = list(boards)
    if not boards:
        raise ValueError('Expected at least one board')
    roots, children, menus, permutations = [], [], [], []
    for board in boards:
        root = fixed_frame_relations(board, board.turn)
        names = sorted(move.uci() for move in board.legal_moves)
        if not names or board.is_game_over(claim_draw=False):
            raise ValueError('Expected a nonterminal legal menu')
        graphs = []
        for name in names:
            child = board.copy(stack=True); child.push_uci(name)
            graphs.append(torch.from_numpy(fixed_frame_relations(child, board.turn)))
        count = len(names)
        if count == 1:
            permutation = torch.tensor([0])
        else:
            value = int.from_bytes(hashlib.sha256(
                ('graph-contrast-permutation-v1|' + board.fen(en_passant='fen')).encode()).digest()[:8], 'big')
            offset = 1 + value % (count - 1)
            permutation = (torch.arange(count) + offset) % count
        roots.append(torch.from_numpy(root)); children.append(torch.stack(graphs))
        menus.append(names); permutations.append(permutation)
    width = max(map(len, menus)); root = torch.stack(roots)
    child = root[:, None].expand(-1, width, -1, -1, -1).clone()
    mask = torch.zeros(len(boards), width, dtype=torch.bool)
    permute = torch.arange(width).expand(len(boards), -1).clone()
    for b, graphs in enumerate(children):
        child[b, :len(graphs)] = graphs; mask[b, :len(graphs)] = True
        permute[b, :len(graphs)] = permutations[b]
    return {'root': root, 'children': child, 'mask': mask, 'menus': menus,
            'permutation': permute}


class GraphContrastHead(TransportHead):
    """The same16,744 weights evaluate root and child graph representations.

Contrast is f(child; frozen root features, action)-f(root; same inputs).
The two controllers share weights but can take different paths as their pooled
contexts diverge. This is not the fixed-gate linear operator-difference identity.
Root and child reference arms use only their respective f. No branch depth,
seed or checkpoint is selected by the module.
"""
    def __init__(self, arm='contrast', *, seed=1109):
        if arm not in ARMS:
            raise ValueError('Unknown graph-contrast arm')
        super().__init__('transport', seed=seed, width=32, steps=3)
        self.contrast_arm = arm

    def _queries(self, hidden, action_features, candidates, operators):
        batch, moves, _ = action_features.shape
        left = F.one_hot(candidates[..., 0], 64).to(hidden.dtype)
        right = F.one_hot(candidates[..., 1], 64).to(hidden.dtype)
        query = torch.tanh(self.initialize(action_features))
        for _ in range(self.steps):
            gates = self.router(query).reshape(batch, moves, 2, 4).softmax(-1)
            # Identical operation ordering in root/child branches makes the
            # zero-change invariant exact, including floating-point execution.
            next_left, next_right = [], []
            for relation in range(4):
                operator = operators[:, :, relation].reshape(batch*moves, 64, 64)
                next_left.append(torch.bmm(left.reshape(-1, 1, 64), operator).reshape(batch, moves, 64)
                                 * gates[:, :, 0, relation, None])
                next_right.append(torch.bmm(right.reshape(-1, 1, 64), operator).reshape(batch, moves, 64)
                                  * gates[:, :, 1, relation, None])
            left, right = sum(next_left), sum(next_right)
            overlap = left * right
            mass = overlap.sum(-1, keepdim=True)
            overlap = overlap / mass.clamp_min(1e-8)
            context = torch.cat((torch.bmm(left, hidden), torch.bmm(right, hidden),
                                 torch.bmm(overlap, hidden), mass), -1)
            query = self.cell(context.reshape(batch*moves, -1),
                              query.reshape(batch*moves, 32)).reshape(batch, moves, 32)
        return query

    def forward(self, hidden, action_features, candidates, legal_mask, base_logits,
                root_edges, child_edges, *, permutation=None):
        if hidden.ndim != 3 or hidden.shape[1:] != (64, 32):
            raise ValueError('Expected frozen width32 square features')
        batch = len(hidden)
        if candidates.ndim != 3 or candidates.shape[0] != batch or candidates.shape[-1] != 5:
            raise ValueError('Invalid candidates')
        moves = candidates.shape[1]
        if (candidates.dtype != torch.long or ((candidates[..., :2] < 0) | (candidates[..., :2] >= 64)).any()
                or action_features.shape != (batch, moves, 120)
                or legal_mask.shape != (batch, moves) or legal_mask.dtype != torch.bool
                or not legal_mask.any(-1).all() or base_logits.shape != (batch, moves)
                or root_edges.shape != (batch, 2, 64, 64)
                or child_edges.shape != (batch, moves, 2, 64, 64)):
            raise ValueError('Invalid input shapes or legal menu')
        parameter = next(self.parameters())
        if any(x.device != parameter.device for x in (hidden, action_features, candidates,
                legal_mask, base_logits, root_edges, child_edges)):
            raise ValueError('Device mismatch')
        if any(x.dtype != parameter.dtype for x in (hidden, action_features, base_logits)):
            raise ValueError('Dtype mismatch')
        if (not torch.isfinite(hidden).all() or not torch.isfinite(action_features).all()
                or not torch.isfinite(base_logits[legal_mask]).all()
                or not ((root_edges == 0) | (root_edges == 1)).all()
                or not ((child_edges == 0) | (child_edges == 1)).all()):
            raise ValueError('Invalid finite features or binary graphs')
        if self.contrast_arm == 'permuted_contrast':
            if (permutation is None or permutation.shape != (batch, moves)
                    or permutation.dtype != torch.long or permutation.device != hidden.device
                    or ((permutation < 0) | (permutation >= moves)).any()):
                raise ValueError('Invalid candidate graph permutation')
            for b in range(batch):
                selected = legal_mask[b].nonzero().flatten()
                if not torch.equal(permutation[b, selected].sort().values, selected):
                    raise ValueError('Permutation must preserve the legal menu')
        # Work on legal candidates only. Padding must not allocate dense child
        # operators or change reductions. A single candidate dimension also
        # avoids copying expanded root matrices at every recurrent step.
        owner, slot = legal_mask.nonzero(as_tuple=True)
        node_features = hidden[owner]
        features = action_features[owner, slot][:, None]
        menu = candidates[owner, slot][:, None]
        root_operators = transitions(root_edges, hidden.dtype)[owner, None]
        if self.contrast_arm == 'root':
            query = self._queries(node_features, features, menu, root_operators)
        else:
            child_slot = permutation[owner, slot] if self.contrast_arm == 'permuted_contrast' else slot
            child_operators = transitions(child_edges[owner, child_slot], hidden.dtype)[:, None]
            query = self._queries(node_features, features, menu, child_operators)
            if self.contrast_arm in ('contrast', 'permuted_contrast'):
                query = query - self._queries(node_features, features, menu, root_operators)
                # Separate backward paths can leave cancellation roundoff even
                # when their forward outputs are bitwise equal. Enforce the
                # structural zero at identical operators in both directions.
                changed = (child_operators != root_operators).flatten(1).any(-1)
                query = query * changed[:, None, None]
        correction = self.output(query).flatten()
        correction = base_logits.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base_logits + correction).masked_fill(~legal_mask, -torch.inf)
