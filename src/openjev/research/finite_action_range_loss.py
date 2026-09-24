"""Blind-cost loss intervention around the unchanged random-head model.

The range penalty is mean((max(error)-min(error))**2/4) over case/horizon
rows. For four action-centered errors it lies between MSE and twice MSE.
Its square root times two bounds the greedy action's true regret. These are
pointwise algebraic bounds, not guarantees about optimization or held-out
decisions. torch.amax/amin share gradients equally among tied extrema.

The frozen joint graph is built once. Only its blind MSE is replaced, with
the original full-TRAIN/minibatch weight. Ordinary MSE returns the original
loss tensor unchanged; empty endpoint batches do too. Extra scalar work is
reported separately from the five frozen model-work blocks, including the
recomputed MSE needed for replacement. Counters describe operations, not FLOPs,
and exclude validation/backward and Python metadata bookkeeping.
"""
from __future__ import annotations

from types import MappingProxyType

import torch

from openjev.research.finite_head_initialization import make_model as _head_model
from openjev.research.finite_head_initialization import model_metadata as _head_metadata
from openjev.research.finite_joint_reuse import TARGET_FIELDS, WORK_ROUTES
from openjev.research.finite_joint_reuse import joint_objective as _joint_objective
from openjev.research.otto_observation_operator_model import _finite, require

VERSION = 'finite-action-range-loss-v1'
__all__ = ['ARMS', 'HEAD_ARMS', 'LOSS_KINDS', 'LOSS_WORK_KEYS', 'TARGET_FIELDS',
           'TRANSPORT_ARMS', 'WORK_ROUTES', 'action_range_loss', 'joint_objective',
           'loss_work_counts', 'make_model', 'model_metadata']
ARMS = ('rounded_mse', 'rounded_double', 'rounded_range', 'free_mse', 'free_double', 'free_range')
HEAD_ARMS = MappingProxyType({arm: 'rounded_random' if arm.startswith('rounded_') else 'matched_free_random'
                              for arm in ARMS})
TRANSPORT_ARMS = MappingProxyType({arm: 'rounded' if arm.startswith('rounded_') else 'matched_free'
                                   for arm in ARMS})
LOSS_KINDS = MappingProxyType({arm: arm.rsplit('_', 1)[1] for arm in ARMS})
LOSS_WORK_KEYS = ('wrapper_calls', 'zero_endpoint_batches', 'blind_error_entries',
    'mse_square_entries', 'mse_mean_calls', 'range_max_rows', 'range_min_rows',
    'range_subtract_entries', 'range_square_entries', 'range_scale_entries', 'range_mean_calls',
    'replacement_subtractions', 'weight_scalar_divisions', 'weight_scalar_multiplications',
    'loss_adjustment_multiplications', 'loss_adjustment_additions')


def loss_work_counts():
    return dict.fromkeys(LOSS_WORK_KEYS, 0)


def make_model(arm, seed, *, check=lambda: None):
    require(type(arm) is str and arm in ARMS, 'declared action-range arm')
    return _head_model(HEAD_ARMS[arm], seed, check=check)


def model_metadata(model, arm):
    require(type(arm) is str and arm in ARMS, 'declared action-range arm')
    parent = _head_metadata(model, HEAD_ARMS[arm])
    return {**parent, 'version': VERSION, 'head_factory_version': parent['version'],
        'arm': arm, 'head_arm': HEAD_ARMS[arm], 'loss_kind': LOSS_KINDS[arm],
        'blind_cost_objective': {'mse': 'mean(error.square())',
            'double': '2*mean(error.square())',
            'range': 'mean((amax(error)-amin(error)).square()/4)'}[LOSS_KINDS[arm]],
        'loss_scope': 'Only joint blind cost changes; observed cost, survival, event and public-prefix losses are unchanged.',
        'range_tie_gradient': 'torch.amax/amin equally share gradient among tied extrema'}


def action_range_loss(error, *, work=None):
    """CPU float64 [case,horizon,4] errors; no centering, clipping or detach."""
    require(isinstance(error, torch.Tensor) and error.device.type == 'cpu'
            and error.dtype == torch.float64 and error.ndim == 3 and error.shape[-1] == 4
            and error.shape[0] > 0 and error.shape[1] > 0, 'nonempty CPU float64 four-action errors')
    _finite(error, 'finite cost errors')
    if work is None:
        work = loss_work_counts()
    require(type(work) is dict and set(work) == set(LOSS_WORK_KEYS)
            and all(type(value) is int and value >= 0 for value in work.values()), 'exact scalar work counters')
    rows = error.shape[0] * error.shape[1]
    maximum = error.amax(-1)
    work['range_max_rows'] += rows
    minimum = error.amin(-1)
    work['range_min_rows'] += rows
    span = maximum - minimum
    work['range_subtract_entries'] += rows
    squared = span.square()
    work['range_square_entries'] += rows
    scaled = squared / 4
    work['range_scale_entries'] += rows
    result = scaled.mean()
    work['range_mean_calls'] += 1
    _finite(result, 'finite action-range loss')
    return result


def joint_objective(model, prefix, lengths, endpoint_positions, actions, observations, targets,
                    *, total_attempts, total_survivors, total_events, loss_kind):
    """Frozen reuse API plus required loss_kind; loss_work is an extra disjoint block."""
    require(type(loss_kind) is str and loss_kind in ('mse', 'double', 'range'), 'declared blind loss kind')
    work = loss_work_counts()
    work['wrapper_calls'] += 1
    result = None
    try:
        result = _joint_objective(model, prefix, lengths, endpoint_positions, actions, observations, targets,
            total_attempts=total_attempts, total_survivors=total_survivors, total_events=total_events)
        eligible = len(endpoint_positions)
        loss = result['loss']
        if not eligible:
            work['zero_endpoint_batches'] += 1
        elif loss_kind != 'mse':
            error = result['blind']['cost_contrasts'] - targets['blind_costs']
            work['blind_error_entries'] += error.numel()
            squared = error.square()
            work['mse_square_entries'] += error.numel()
            mse = squared.mean()
            work['mse_mean_calls'] += 1
            adjustment = mse
            if loss_kind == 'range':
                adjustment = action_range_loss(error, work=work) - mse
                work['replacement_subtractions'] += 1
            weight = eligible / total_survivors * total_attempts / len(prefix)
            work['weight_scalar_divisions'] += 2
            work['weight_scalar_multiplications'] += 1
            weighted = adjustment * weight
            work['loss_adjustment_multiplications'] += 1
            loss = loss + weighted
            work['loss_adjustment_additions'] += 1
            _finite(loss, 'finite replacement joint objective')
        return {**result, 'loss': loss, 'loss_kind': loss_kind, 'loss_work': dict(work)}
    except BaseException as error:
        error.action_range_loss_work = dict(work)
        if result is not None:
            error.joint_reuse_work = {route: dict(block) for route, block in result['work'].items()}
            error.rounded_model_work = dict(model.last_work)
        raise
