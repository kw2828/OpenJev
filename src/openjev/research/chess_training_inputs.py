# SPDX-License-Identifier: GPL-3.0-only
"""Join authenticated feature/pin blocks and select aligned training batches.

File provenance belongs to the caller. This module validates tensor structure
and cross-input identity; it neither generates features nor reads labels.
"""
import torch

from openjev.research import chess_native_feature_cache as native
from openjev.research import chess_pin_cache as pins

VERSION = 'native-feature-pin-training-loader-v1'


def validate_features(cache):
    if set(cache) != {'version', 'nodes', 'action_features', 'base_logits', 'candidates', 'offsets', 'menus', 'fens'} or cache['version'] != native.VERSION:
        raise ValueError('Invalid feature fields or version')
    count = len(cache['menus']); lengths = [len(m) for m in cache['menus']]
    if not count or len(cache['fens']) != count or min(lengths) <= 0:
        raise ValueError('Invalid feature root membership')
    if any(tuple(sorted(set(m))) != m for m in cache['menus']):
        raise ValueError('Invalid feature menus')
    total = sum(lengths)
    shapes = {'nodes': (count, 64, 32), 'action_features': (total, 120),
              'base_logits': (total,), 'candidates': (total, 5), 'offsets': (count+1,)}
    for name, shape in shapes.items():
        value = cache[name]; dtype = torch.long if name in ('candidates', 'offsets') else torch.float32
        if value.shape != shape or value.dtype != dtype or value.device.type != 'cpu' or value.requires_grad:
            raise ValueError('Invalid feature tensor')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite feature tensor')
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lengths).cumsum(0)))
    if not torch.equal(offsets, cache['offsets']): raise ValueError('Invalid feature offsets')
    return cache


def merge_features(blocks):
    blocks = [validate_features(b) for b in blocks]
    if not blocks: raise ValueError('Expected feature blocks')
    offsets = [torch.zeros(1, dtype=torch.long)]; candidate = 0
    for block in blocks:
        offsets.append(block['offsets'][1:]+candidate); candidate += int(block['offsets'][-1])
    return {'version': native.VERSION,
            **{k: torch.cat([b[k] for b in blocks]) for k in ('nodes', 'action_features', 'base_logits', 'candidates')},
            'offsets': torch.cat(offsets), 'menus': tuple(m for b in blocks for m in b['menus']),
            'fens': tuple(f for b in blocks for f in b['fens'])}


def merge_pins(blocks):
    blocks = [pins.validate(b) for b in blocks]
    if not blocks: raise ValueError('Expected pin blocks')
    factors = []; candidates = [torch.zeros(1, dtype=torch.long)]; offsets = [torch.zeros(1, dtype=torch.long)]
    candidate = factor = 0
    for block in blocks:
        rows = block['factors'].clone(); rows[:, 0] += candidate; factors.append(rows)
        candidates.append(block['candidate_offsets'][1:]+candidate)
        offsets.append(block['factor_offsets'][1:]+factor)
        candidate += int(block['candidate_offsets'][-1]); factor += len(rows)
    return {'version': pins.VERSION, 'factors': torch.cat(factors),
            'candidate_offsets': torch.cat(candidates), 'factor_offsets': torch.cat(offsets),
            'menus': tuple(m for b in blocks for m in b['menus']),
            'fens': tuple(f for b in blocks for f in b['fens'])}


class TrainingInputs:
    def __init__(self, feature_blocks, pin_blocks):
        self.features = merge_features(feature_blocks); self.pins = merge_pins(pin_blocks)
        if (self.features['menus'] != self.pins['menus'] or self.features['fens'] != self.pins['fens']
                or not torch.equal(self.features['offsets'], self.pins['candidate_offsets'])):
            raise ValueError('Feature and pin identities differ')

    def batch(self, indices, graph):
        args = native.arguments(self.features, indices, graph)
        return args, pins.gather(self.pins, indices)
