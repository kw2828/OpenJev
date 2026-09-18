# SPDX-License-Identifier: GPL-3.0-only
"""Anchored second-order control for the pin-role MLP, not the whole policy."""
from itertools import combinations

import torch

from openjev.research.chess_pin_factor_head import ChessPinFactorHead
from openjev.research.chess_pin_factor_controls import validate_factors

VERSION = 'wldn-pin-anchored-pairwise-v1'


class ChessPinPairwiseHead(ChessPinFactorHead):
    """Same 16,658 tensors as joint; seven factor calls rather than one.

With metadata fixed and role vectors anchored at zero, retain all terms of
order at most two: sum(pair slices)-sum(single slices)+empty slice. Earlier
neural encoding, witness selection and the final readout still mix information.
This is an established anchored functional decomposition, not causal isolation
of chess pieces or a novel network family.
"""
    def __init__(self, *, seed=1109):
        super().__init__('joint', seed=seed)
        self.arm = 'pairwise'

    def factor_value(self, values):
        if values.ndim != 2 or values.shape[-1] != 197:
            raise ValueError('Expected three64-channel roles and five metadata channels')
        roles = [values[:, i*64:(i+1)*64] for i in range(3)]
        zero = torch.zeros_like(roles[0]); metadata = values[:, 192:]
        result = self.factor(torch.cat((zero, zero, zero, metadata), -1))
        for omit in range(3):
            pair = [zero if i == omit else roles[i] for i in range(3)]
            single = [roles[i] if i == omit else zero for i in range(3)]
            result = result+self.factor(torch.cat((*pair, metadata), -1))-self.factor(torch.cat((*single, metadata), -1))
        return result

    def reference_factor(self, values):
        """Explicit subset sum with independently constructed role masks."""
        if values.ndim != 2 or values.shape[-1] != 197:
            raise ValueError('Expected factor values')
        terms = []
        for size in (0, 1, 2):
            for subset in combinations(range(3), size):
                mask = values.new_ones(197)
                for role in range(3):
                    if role not in subset: mask[role*64:(role+1)*64] = 0
                terms.append((1 if size % 2 == 0 else -1)*self.factor(values*mask))
        return torch.stack(terms).sum(0)

    def pool(self, root, delta, factors, *, reference=False):
        if not reference:
            return super().pool(root, delta, factors)
        if root.shape != delta.shape or delta.ndim != 3 or delta.shape[1:] != (64, 32):
            raise ValueError('Invalid role node features')
        validate_factors(factors, len(delta), delta.device)
        outputs = []
        for candidate in range(len(delta)):
            total = delta.new_zeros(32)
            for owner, color, king, blocker, attacker, status in factors.tolist():
                if owner != candidate: continue
                roles = [torch.cat((root[owner, square], delta[owner, square])) for square in (king, blocker, attacker)]
                metadata = delta.new_tensor([int(color == i) for i in range(2)]+[int(status == i) for i in range(3)])
                total = total+self.reference_factor(torch.cat((*roles, metadata))[None])[0]
            outputs.append(total)
        return torch.stack(outputs) if outputs else delta.new_zeros(0, 32)
