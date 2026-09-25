"""Differentiable output-history probes for the unchanged tanh feedback model.

This is a conditional, data-free implementation of the prospective sensitivity
study. It neither fits a model nor reads benchmark data. Sampled finite-horizon
output gains are not worst-case norms or stability certificates.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from openjev.research.fsm_residual import CHANNELS, FSMResidual, _require

HORIZONS = (8, 32, 128)
ARMS = (
    'unregularized', 'l2', 'residual_magnitude', 'total_sensitivity',
    'max_envelope', 'relative_sensitivity',
)


def sample_directions(batch_size, order, *, generator, count=2):
    """Draw unit Rademacher probes using only the caller's explicit CPU stream."""
    _require(type(batch_size) is int and batch_size > 0, 'positive integer batch size')
    _require(type(order) is int and 1 <= order <= 99, 'integer order in1..99')
    _require(type(count) is int and count > 0, 'positive integer direction count')
    _require(isinstance(generator, torch.Generator) and generator.device.type == 'cpu',
             'explicit CPU random generator required')
    width = CHANNELS * order
    signs = torch.randint(0, 2, (batch_size, count, width), generator=generator,
                          device='cpu', dtype=torch.int64)
    return (2 * signs.to(torch.float64) - 1) / math.sqrt(width)


def _model(model):
    _require(type(model) is FSMResidual and model.mode == 'feedback'
             and model.residual_kind == 'tanh', 'unchanged tanh feedback FSMResidual required')
    model._validate_parameters()


def _finite(value, label):
    _require(bool(torch.isfinite(value).all()), 'nonfinite sensitivity ' + label + '; no repair')


def _primal_step(model, current, current_u):
    split = CHANNELS * model.order
    first, last = model.residual[0], model.residual[2]
    features = torch.cat((current[:, :split], current_u, current[:, split:]), dim=-1)
    linear = F.linear(features, model.coefficients[:, :-1], model.coefficients[:, -1])
    hidden = torch.tanh(F.linear(features, first.weight, first.bias))
    residual = F.linear(hidden, last.weight, last.bias)
    prediction = linear + residual
    next_state = torch.cat((current[:, CHANNELS:split], prediction,
                            current[:, split + CHANNELS:], current_u), dim=-1)
    return prediction, next_state, residual, hidden


def _inputs(model, future_u, state):
    _model(model)
    model._state(state)
    _require(isinstance(future_u, torch.Tensor) and future_u.ndim == 3
             and future_u.shape[1] > 0, 'positive forecast horizon required')
    model._tensor(future_u, (len(state), future_u.shape[1], CHANNELS), 'future inputs')


def rollout_primal(model, future_u, state):
    """Unchanged primal predictions and corrections, without tangent computation."""
    _inputs(model, future_u, state)
    current, predictions, residuals = state, [], []
    for step in range(future_u.shape[1]):
        prediction, current, residual, _ = _primal_step(model, current, future_u[:, step])
        predictions.append(prediction)
        residuals.append(residual)
    result = {'prediction': torch.stack(predictions, dim=1), 'final_state': current,
              'residual': torch.stack(residuals, dim=1)}
    for name, value in result.items():
        _finite(value, name)
    return result


def rollout_tangents(model, future_u, state, directions):
    """Return the full primal and directional recurrence without detached states.

    directions[B,K,3*p] perturb only chronological output lags. Their magnitude
    is caller supplied; sample_directions supplies the unit probes in the draft.
    Input lags and executed future inputs have zero tangents. A separate linear
    tangent recurrence uses the same full feature layout and linear arithmetic.
    All returned tensors are temporary training/diagnostic outputs, not a cache.
    """
    _inputs(model, future_u, state)
    batch, horizon, split = len(state), future_u.shape[1], CHANNELS * model.order
    _require(isinstance(directions, torch.Tensor) and directions.ndim == 3
             and directions.shape[1] > 0, 'directions shape [B,positive K,3*p] required')
    count = directions.shape[1]
    model._tensor(directions, (batch, count, split), 'output-history directions')
    current = state
    tangent = torch.cat((directions, torch.zeros_like(directions)), dim=-1)
    baseline_tangent = tangent.clone()
    zero_input = directions.new_zeros(batch, count, CHANNELS)
    first, last = model.residual[0], model.residual[2]
    linear_weight = model.coefficients[:, :-1]
    predictions, residuals, tangents, baseline_tangents = [], [], [], []
    for step in range(horizon):
        current_u = future_u[:, step]
        prediction, current, residual, hidden = _primal_step(model, current, current_u)

        feature_tangent = torch.cat((tangent[:, :, :split], zero_input,
                                     tangent[:, :, split:]), dim=-1)
        linear_tangent = F.linear(feature_tangent, linear_weight)
        hidden_tangent = F.linear(feature_tangent, first.weight) * (1 - hidden.square())[:, None]
        prediction_tangent = linear_tangent + F.linear(hidden_tangent, last.weight)
        baseline_features = torch.cat((baseline_tangent[:, :, :split], zero_input,
                                       baseline_tangent[:, :, split:]), dim=-1)
        baseline_prediction_tangent = F.linear(baseline_features, linear_weight)

        tangent = torch.cat((tangent[:, :, CHANNELS:split], prediction_tangent,
                             tangent[:, :, split + CHANNELS:], zero_input), dim=-1)
        baseline_tangent = torch.cat((baseline_tangent[:, :, CHANNELS:split],
                                      baseline_prediction_tangent,
                                      baseline_tangent[:, :, split + CHANNELS:], zero_input), dim=-1)
        predictions.append(prediction)
        residuals.append(residual)
        tangents.append(prediction_tangent)
        baseline_tangents.append(baseline_prediction_tangent)
    result = {
        'prediction': torch.stack(predictions, dim=1),
        'final_state': current,
        'residual': torch.stack(residuals, dim=1),
        'output_tangents': torch.stack(tangents, dim=2),
        'linear_output_tangents': torch.stack(baseline_tangents, dim=2),
        'final_tangents': tangent,
        'linear_final_tangents': baseline_tangent,
    }
    for name, value in result.items():
        _finite(value, name)
    return result


def _gains(value, horizons):
    _require(isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[0] > 0
             and value.shape[1] > 0 and value.shape[2] > 0 and value.shape[3] == CHANNELS,
             'tangent outputs shape [B,K,H,3] required')
    FSMResidual._tensor(value, tuple(value.shape), 'output tangents')
    _require(isinstance(horizons, tuple) and len(horizons) > 0
             and all(type(h) is int and 1 <= h <= value.shape[2] for h in horizons)
             and tuple(sorted(set(horizons))) == horizons,
             'distinct increasing integer horizons within rollout required')
    gain = torch.stack([value[:, :, :h].square().sum(dim=(-2, -1)) / (CHANNELS * h)
                        for h in horizons], dim=-1)
    _finite(gain, 'gain')
    return gain


def gain_profiles(result, horizons=HORIZONS):
    """Return learned/baseline gains [B,K,L], each prefix energy divided by3h."""
    _require(isinstance(result, dict), 'rollout result mapping required')
    a, b = result['output_tangents'], result['linear_output_tangents']
    gain, baseline = _gains(a, horizons), _gains(b, horizons)
    _require(a.shape == b.shape, 'matching learned and baseline tangent shapes required')
    return gain, baseline


def regularizer(model, result, arm, horizons=HORIZONS):
    """Compute only one arm's unweighted penalty, without unused-term failures.

    Max-envelope reduction is over directions only, independently per window and
    horizon. Residual magnitude concerns the per-step neural correction, not the
    accumulated difference between the nonlinear and linear forecasts.

    Unregularized/L2 accept result=None. Residual magnitude requires only a
    residual tensor; total sensitivity needs only learned output tangents. Simple
    arms should use a primal-only rollout, not pay for rollout_tangents.
    """
    _model(model)
    _require(type(arm) is str and arm in ARMS, 'declared regularizer arm required')
    if arm == 'unregularized':
        value = model.coefficients.new_zeros(())
    elif arm == 'l2':
        parameters = tuple(model.parameters())
        value = sum(p.square().sum() for p in parameters) / sum(p.numel() for p in parameters)
    else:
        _require(isinstance(result, dict), 'rollout result mapping required')
        if arm == 'residual_magnitude':
            correction = result['residual']
            _require(isinstance(correction, torch.Tensor) and correction.ndim == 3
                     and correction.shape[0] > 0 and correction.shape[1] > 0
                     and correction.shape[2] == CHANNELS, 'per-step residual shape [B,H,3] required')
            FSMResidual._tensor(correction, tuple(correction.shape), 'per-step residual')
            value = correction.square().mean()
        elif arm == 'total_sensitivity':
            value = _gains(result['output_tangents'], horizons).square().mean()
        else:
            gain, baseline = gain_profiles(result, horizons)
            allowance = baseline if arm == 'relative_sensitivity' else baseline.max(dim=1, keepdim=True).values
            value = torch.relu(gain - allowance).square().mean()
    _finite(value, arm)
    return value


def regularizers(model, result, horizons=HORIZONS):
    """Diagnostic-only bulk evaluation; training should call regularizer per arm.

    This intentionally evaluates all six terms and fails if any is nonfinite.
    It must not impose unused derivative work or failures on simple comparators.
    """
    return {arm: regularizer(model, result, arm, horizons) for arm in ARMS}
