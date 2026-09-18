"""Graph-contrast implementation that shares root propagation across candidates.

The earlier dense engineering implementation remains unchanged. This version
reuses the original transport root kernel and supports compact legal-child
inputs. It has the same mathematical score, not a new architectural hypothesis.
Numerical/gradient parity and actual runtime require separate checks.
"""
import torch

from openjev.research.chess_graph_contrast import GraphContrastHead
from openjev.research.chess_transport import TransportHead, transitions


class SharedRootGraphContrastHead(GraphContrastHead):
    def forward(self, hidden, action_features, candidates, legal_mask, base_logits,
                root_edges, child_edges=None, *, permutation=None):
        if self.contrast_arm == 'root':
            return TransportHead.forward(self, hidden, action_features, candidates,
                                         legal_mask, base_logits, root_edges)
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
                or root_edges.shape != (batch, 2, 64, 64)):
            raise ValueError('Invalid feature shapes or legal menu')
        owner, slot = legal_mask.nonzero(as_tuple=True)
        if child_edges is None:
            raise ValueError('Child graphs required for this arm')
        if child_edges.shape == (batch, moves, 2, 64, 64):
            child_edges = child_edges[owner, slot]
        elif child_edges.shape != (len(owner), 2, 64, 64):
            raise ValueError('Invalid compact child graphs')
        parameter = next(self.parameters())
        if any(x.device != parameter.device for x in (hidden, action_features, candidates,
                legal_mask, base_logits, root_edges, child_edges)):
            raise ValueError('Device mismatch')
        if any(x.dtype != parameter.dtype for x in (hidden, action_features, base_logits)):
            raise ValueError('Dtype mismatch')
        if (not torch.isfinite(hidden).all() or not torch.isfinite(action_features).all()
                or not torch.isfinite(base_logits[legal_mask]).all()
                or not ((root_edges == 0) | (root_edges == 1)).all()
                or not ((child_edges == 0) | (child_edges == 1)).all()
                or root_edges.diagonal(dim1=-2, dim2=-1).any()
                or child_edges.diagonal(dim1=-2, dim2=-1).any()):
            raise ValueError('Expected finite features and binary native attacks without self edges')
        if self.contrast_arm == 'permuted_contrast':
            if (permutation is None or permutation.shape != (batch, moves)
                    or permutation.dtype != torch.long or permutation.device != hidden.device
                    or ((permutation < 0) | (permutation >= moves)).any()):
                raise ValueError('Invalid candidate permutation')
            for b in range(batch):
                selected = legal_mask[b].nonzero().flatten()
                if not torch.equal(permutation[b, selected].sort().values, selected):
                    raise ValueError('Permutation must preserve the legal menu')
            lookup = torch.full((batch, moves), -1, dtype=torch.long, device=hidden.device)
            lookup[owner, slot] = torch.arange(len(owner), device=hidden.device)
            child_edges = child_edges[lookup[owner, permutation[owner, slot]]]
        child_operators = transitions(child_edges, hidden.dtype)[:, None]
        query = self._queries(hidden[owner], action_features[owner, slot][:, None],
                              candidates[owner, slot][:, None], child_operators)
        correction = self.output(query).flatten()
        if self.contrast_arm in ('contrast', 'permuted_contrast'):
            root_correction = TransportHead.forward(self, hidden, action_features,
                candidates, legal_mask, torch.zeros_like(base_logits), root_edges)[owner, slot]
            # For binary native attacks without self edges, identical normalized
            # operators and identical raw relations are equivalent. Avoid an
            # expanded floating root-operator tensor just to check this identity.
            changed = (child_edges != root_edges[owner]).flatten(1).any(-1)
            correction = (correction - root_correction) * changed
        dense = base_logits.new_zeros(batch*moves).scatter(0, owner*moves+slot, correction).reshape(batch, moves)
        return (base_logits + dense).masked_fill(~legal_mask, -torch.inf)
