# SPDX-License-Identifier: GPL-3.0-only
"""Compact absolute-pin inputs with explicit root/candidate offsets."""
import torch

from openjev.research.chess_pin_factor_head import pack_factors

VERSION = 'absolute-pin-training-blocks-v1'


def pack(records, fens):
    if not records or len(records) != len(fens): raise ValueError('Expected aligned nonempty roots')
    candidate_offsets, factor_offsets = [0], [0]
    for row in records:
        candidate_offsets.append(candidate_offsets[-1]+len(row['candidates']))
        factor_offsets.append(factor_offsets[-1]+sum(len(c['factors']) for c in row['candidates']))
    return {'version': VERSION, 'factors': pack_factors(records),
            'candidate_offsets': torch.tensor(candidate_offsets, dtype=torch.long),
            'factor_offsets': torch.tensor(factor_offsets, dtype=torch.long),
            'menus': tuple(tuple(c['uci'] for c in r['candidates']) for r in records), 'fens': tuple(fens)}


def validate(cache):
    if set(cache) != {'version', 'factors', 'candidate_offsets', 'factor_offsets', 'menus', 'fens'} or cache['version'] != VERSION:
        raise ValueError('Invalid pin-cache version or fields')
    factors, candidates, offsets = cache['factors'], cache['candidate_offsets'], cache['factor_offsets']
    count = len(cache['menus'])
    if not count or len(cache['fens']) != count:
        raise ValueError('Invalid pin-cache root count')
    if any(t.dtype != torch.long or t.device.type != 'cpu' for t in (factors, candidates, offsets)):
        raise ValueError('Pin-cache arrays must be CPU int64')
    if factors.ndim != 2 or factors.shape[1] != 6 or candidates.shape != (count+1,) or offsets.shape != (count+1,):
        raise ValueError('Invalid pin-cache array shape')
    if candidates[0] != 0 or offsets[0] != 0 or offsets[-1] != len(factors) or (offsets[1:] < offsets[:-1]).any():
        raise ValueError('Invalid pin-factor offsets')
    lengths = torch.tensor([len(m) for m in cache['menus']])
    if (lengths <= 0).any() or not torch.equal(candidates[1:]-candidates[:-1], lengths):
        raise ValueError('Candidate offsets and menus disagree')
    if any(tuple(sorted(set(menu))) != menu for menu in cache['menus']):
        raise ValueError('Menus must be sorted and unique')
    if len(factors):
        bounds = factors.new_tensor([int(candidates[-1]), 2, 64, 64, 64, 3])
        if ((factors < 0) | (factors >= bounds)).any(): raise ValueError('Invalid factor index')
        if any((factors[:, a] == factors[:, b]).any() for a, b in ((2, 3), (2, 4), (3, 4))):
            raise ValueError('Repeated witness square')
    for root in range(count):
        rows = factors[int(offsets[root]):int(offsets[root+1])]
        if len(rows) and ((rows[:, 0] < candidates[root]) | (rows[:, 0] >= candidates[root+1])).any():
            raise ValueError('Factor attached to the wrong root')
    return cache


def gather(cache, indices):
    """Gather already validated roots in arbitrary order, with repeats allowed."""
    if (indices.ndim != 1 or indices.dtype != torch.long or indices.device.type != 'cpu'
            or not len(indices) or (indices < 0).any() or (indices >= len(cache['menus'])).any()):
        raise ValueError('Invalid pin-cache selection')
    pieces = []; candidate = 0
    for root in indices.tolist():
        lo, hi = cache['factor_offsets'][root:root+2].tolist()
        a, b = cache['candidate_offsets'][root:root+2].tolist()
        rows = cache['factors'][lo:hi].clone(); rows[:, 0] += candidate-a
        pieces.append(rows); candidate += b-a
    return torch.cat(pieces)
