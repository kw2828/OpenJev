# SPDX-License-Identifier: GPL-3.0-only
"""Shared mechanics for a future matched pin study; no quality run is launched."""
import hashlib
import math
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from openjev.research.wldn_baseline import ChessWLDNHead
from openjev.research.chess_pin_factor_head import ChessPinFactorHead
from openjev.research.chess_pin_factor_controls import ChessPinFactorControl
from openjev.research.chess_pin_pairwise import ChessPinPairwiseHead
from openjev.research.chess_union_difference import ChessUnionDifferenceHead

VERSION = 'canonical-pin-study-mechanics-v1'
SEEDS = (97, 109, 127)
CORE_ARMS = ('wldn', 'joint', 'separable', 'pairwise', 'root_only', 'counts', 'graph_mlp')
UNION_ARMS = ('child', 'union', 'edits', 'rotated')
PARAMETERS = {'wldn': 16638, 'joint': 16658, 'separable': 16658, 'pairwise': 16658,
              'root_only': 16658, 'counts': 16656, 'graph_mlp': 16658,
              **{'union:'+arm: 16740 for arm in UNION_ARMS}}


def make_head(arm, backbone_seed):
    if arm not in PARAMETERS or backbone_seed not in SEEDS: raise ValueError('Unknown arm or backbone seed')
    seed = backbone_seed+1100
    if arm == 'wldn': model = ChessWLDNHead(seed=seed)
    elif arm in ('joint', 'separable'): model = ChessPinFactorHead(arm, seed=seed)
    elif arm == 'pairwise': model = ChessPinPairwiseHead(seed=seed)
    elif arm in ('root_only', 'counts', 'graph_mlp'): model = ChessPinFactorControl(arm, seed=seed)
    else: model = ChessUnionDifferenceHead(arm.split(':')[1], seed=seed)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]: raise AssertionError('Parameter count changed')
    return model


def logits(model, arm, args, factors, *, reference=False):
    if arm not in PARAMETERS: raise ValueError('Unknown arm')
    expected = (ChessWLDNHead if arm == 'wldn' else ChessUnionDifferenceHead if arm.startswith('union:')
                else ChessPinPairwiseHead if arm == 'pairwise' else ChessPinFactorControl if arm in ('root_only', 'counts', 'graph_mlp')
                else ChessPinFactorHead)
    if type(model) is not expected: raise ValueError('Arm/model type mismatch')
    if arm in ('joint', 'separable') and model.arm != arm: raise ValueError('Factor arm mismatch')
    if arm in ('root_only', 'counts', 'graph_mlp') and model.control != arm: raise ValueError('Control arm mismatch')
    if arm.startswith('union:') and model.arm != arm.split(':')[1]: raise ValueError('Union arm mismatch')
    if arm == 'wldn': return model(*args)
    if arm.startswith('union:'): return model(*args, reference=reference)
    return model(*args, factors, reference=reference)


def schedule(roots, epochs, batch_size, seed):
    if any(type(v) is not int or v < 1 for v in (roots, epochs, batch_size)) or seed not in SEEDS:
        raise ValueError('Invalid update schedule')
    if roots % batch_size: raise ValueError('All matched-study batches must be complete')
    for epoch in range(epochs):
        order = np.random.default_rng(700000+100*seed+epoch).permutation(roots)
        for start in range(0, roots, batch_size):
            yield epoch, torch.from_numpy(order[start:start+batch_size].copy())


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)


def update(model, arm, opt, args, factors, targets, *, reference=False):
    """One measured update; validate legal targets before any parameter change."""
    if (targets.dtype != torch.long or targets.device != args[3].device or targets.shape != (len(args[3]),)
            or (targets < 0).any() or (targets >= args[3].shape[1]).any()
            or not args[3][torch.arange(len(targets), device=targets.device), targets].all()):
        raise ValueError('Targets must identify one legal candidate per root')
    begin = time.perf_counter(); opt.zero_grad(set_to_none=True)
    values = logits(model, arm, args, factors, reference=reference)
    loss = F.cross_entropy(values, targets)
    if not torch.isfinite(loss): raise AssertionError('Nonfinite loss')
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError('Missing/nonfinite gradient')
    opt.step()
    return {'loss': float(loss.detach()), 'gradient_norm': float(norm), 'update_seconds': time.perf_counter()-begin}


def index_digest(indices):
    if indices.device.type != 'cpu' or indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError('Expected CPU int64 root indices')
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()


def select_comparators(metrics):
    """Caller must authenticate completed union outcomes and their full audit.

Select the highest three-seed mean on each old panel among original WLDN and
the four union-study arms, regardless of that study's acceptance gate. Exact
ties prefer WLDN, then child,union,edits,rotated. A maximum of two distinct
union arms is added to the seven mandatory retrained arms. This adaptive
selection is not confirmation or a result under the new input contract.
"""
    candidates = ('wldn', *UNION_ARMS); selected = set(); panels = {}
    for split in ('dev', 'shift'):
        means = {}
        for arm in candidates:
            values = []
            for seed in SEEDS:
                result = metrics[f'{arm}-{seed}'][split]; value = result['agreement']
                if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                        or result['examples'] != 2048): raise ValueError('Invalid completed panel metrics')
                values.append(value)
            means[arm] = statistics.mean(values)
        best = max(means.values()); winner = next(a for a in candidates if means[a] == best)
        panels[split] = {'means': means, 'winner': winner}
        if winner != 'wldn': selected.add(winner)
    extras = tuple('union:'+arm for arm in UNION_ARMS if arm in selected)
    return {'panels': panels, 'additional_union_arms': list(extras), 'fit_arms': [*CORE_ARMS, *extras]}
