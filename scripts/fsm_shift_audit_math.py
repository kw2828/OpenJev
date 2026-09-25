# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent, array-only forecast arithmetic for the prospective FSM shift check.

Inputs and predictions are physical units. VARX/residual caller states contain
normalized chronological output lags followed by input lags. Author states are
in each saved model's coordinates. The first supplied context input belongs to
the second supplied observation; no unavailable first input is invented.

There are no file decoders, production model imports, fitted models, or targets
in this interface. Only the two named, qualified independent source modules are
loaded. BLA reuse retains its fixed order-28, C100/H128 geometry. Other branches
also allow smaller fabricated contexts/horizons for mathematical qualification.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC_KEYS = {'instance_id', 'family', 'seed', 'kind', 'order', 'alpha',
             'residual_kind', 'mode', 'hidden_width', 'context_iterations'}
NORMALIZER_KEYS = {'u_mean', 'u_scale', 'y_mean', 'y_scale'}
HELPERS = {
    'bla': ('audit_fsm_author_bla.py', '0e3784f668afe69da29182c95665c5fb83c63df5c202a40b7e9cc74e61e7b1bd'),
    'nllfr': ('fsm_nllfr_budget_audit_math.py', 'd4454df6f8854d1895d7edfee75dd1e36a58ec616fb12537a7424359a538f8c6'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == np.float64
            and value.shape == shape and np.isfinite(value).all(),
            f'{name}: finite float64 array with shape {shape}')


def _helper(kind):
    name, expected = HELPERS[kind]
    path = ROOT/'scripts'/name
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            'qualified independent source changed')
    spec = importlib.util.spec_from_file_location('shift_independent_'+kind, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(value):
    """Own finite JSON trees, including qualified helpers' diagnostic arrays."""
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), 'diagnostic object keys')
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    require(value is None or type(value) in (str, bool, int, float), 'diagnostic JSON type')
    require(type(value) is not float or math.isfinite(value), 'nonfinite diagnostic')
    return value


def _validate(spec, arrays, normalizer, y, u, future):
    require(isinstance(spec, dict) and set(spec) == SPEC_KEYS, 'exact specification keys')
    require(all(type(spec[k]) is str and spec[k] for k in ('instance_id', 'family')), 'identity strings')
    seed = spec['seed']
    require(seed is None or type(seed) is int and 0 <= seed < 2**63, 'seed')
    require(spec['instance_id'] == spec['family']+(f'-{seed}' if seed is not None else ''),
            'instance identity disagrees with family and seed')
    kind, p = spec['kind'], spec['order']
    require(kind in ('varx', 'residual', 'bla', 'nllfr'), 'model kind')
    require(type(p) is int and p > 0, 'positive integer order')
    if kind in ('varx', 'residual'):
        require(p <= 99, 'lag order at most 99')
    if kind == 'varx':
        require(type(spec['alpha']) in (int, float) and math.isfinite(spec['alpha'])
                and spec['alpha'] > 0, 'positive VARX alpha')
    else:
        require(spec['alpha'] is None, 'irrelevant alpha must be null')
    if kind == 'residual':
        require(seed is not None and spec['residual_kind'] in ('affine', 'tanh')
                and spec['mode'] in ('output_only', 'feedback'), 'residual kind/mode')
        width = spec['hidden_width']
        require(width is None if spec['residual_kind'] == 'affine'
                else type(width) is int and width > 0, 'residual hidden width')
    else:
        require(all(spec[k] is None for k in ('residual_kind', 'mode', 'hidden_width')),
                'irrelevant residual fields must be null')
    require(type(spec['context_iterations']) is int and spec['context_iterations'] in (16, 64)
            if kind == 'nllfr' else spec['context_iterations'] is None, 'context iterations')
    require(kind not in ('bla', 'nllfr') or spec['seed'] is None, 'author seed is provenance only')
    require(isinstance(arrays, dict), 'model arrays mapping')
    require(isinstance(normalizer, dict) and set(normalizer) == NORMALIZER_KEYS, 'normalizer keys')
    for k in NORMALIZER_KEYS:
        finite(normalizer[k], (3,), k)
    require((normalizer['u_scale'] > 0).all() and (normalizer['y_scale'] > 0).all(),
            'positive common scales')
    require(isinstance(y, np.ndarray) and y.ndim == 3 and y.shape[0] > 0
            and y.shape[1] >= 2, 'nonempty batched context')
    batch, context = y.shape[:2]
    require(isinstance(future, np.ndarray) and future.ndim == 3, 'future rank')
    horizon = future.shape[1]
    finite(y, (batch, context, 3), 'y_context')
    finite(u, (batch, context-1, 3), 'u_context')
    finite(future, (batch, horizon, 3), 'future_u')
    require(kind not in ('varx', 'residual') or context >= p+1, 'sufficient known input history')
    return batch, context, horizon


def _lags(spec, arrays, norm, y, u, future):
    p, batch = spec['order'], len(y)
    width = 6*p+3
    shapes = {'coefficients': (3, width+1)}
    if spec['kind'] == 'residual':
        if spec['residual_kind'] == 'affine':
            shapes.update({'residual.weight': (3, width), 'residual.bias': (3,)})
        else:
            h = spec['hidden_width']
            shapes.update({'residual.0.weight': (h, width), 'residual.0.bias': (h,),
                           'residual.2.weight': (3, h), 'residual.2.bias': (3,)})
    require(set(arrays) == set(shapes), 'exact lag model array roster')
    for k, shape in shapes.items():
        finite(arrays[k], shape, k)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y = (y-norm['y_mean'])/norm['y_scale']
        u = (u-norm['u_mean'])/norm['u_scale']
        future = (future-norm['u_mean'])/norm['u_scale']
    if not all(np.isfinite(v).all() for v in (y, u, future)):
        raise FloatingPointError('nonfinite common normalization')
    state = np.concatenate((y[:, -p:].reshape(batch, -1), u[:, -p:].reshape(batch, -1)), axis=1)
    initial = state.copy()
    prediction = np.empty((batch, future.shape[1], 3), dtype=np.float64)
    coefficients = arrays['coefficients']
    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(future.shape[1]):
            features = np.concatenate((state[:, :3*p], future[:, t], state[:, 3*p:]), axis=1)
            if spec['kind'] == 'varx':
                design = np.concatenate((features, np.ones((batch, 1))), axis=1)
                linear = design@coefficients.T
                output = linear
            else:
                linear = features@coefficients[:, :-1].T+coefficients[:, -1]
                if spec['residual_kind'] == 'affine':
                    correction = features@arrays['residual.weight'].T+arrays['residual.bias']
                else:
                    hidden = np.tanh(features@arrays['residual.0.weight'].T+arrays['residual.0.bias'])
                    correction = hidden@arrays['residual.2.weight'].T+arrays['residual.2.bias']
                output = linear+correction
            prediction[:, t] = output
            inserted = linear if spec['kind'] == 'residual' and spec['mode'] == 'output_only' else output
            state = np.concatenate((state[:, 3:3*p], inserted, state[:, 3*p+3:], future[:, t]), axis=1)
        physical = prediction*norm['y_scale']+norm['y_mean']
    if not all(np.isfinite(v).all() for v in (prediction, physical, state)):
        raise FloatingPointError('nonfinite lag forecast')
    return physical, initial, state, {'state_units': 'common-normalized output lags then input lags',
                                      'order': p, 'context_inputs_used': p}


def _bla(arrays, y, u, future):
    helper = _helper('bla')
    # The held replay omits this state, so reconstruct it at the same instant.
    # Its own replay separately checks geometry and all forecast outputs.
    helper.model(arrays)
    require(y.shape[1] == 100 and future.shape[1] == 128, 'BLA replay requires C100/H128')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized_y = (y-arrays['y_mean'])/arrays['y_std']
        normalized_u = (u-arrays['u_mean'])/arrays['u_std']
        a, b, c, d = (arrays[k] for k in ('A', 'B_u', 'C_y', 'D_yu'))
        power, forced = np.eye(28), np.zeros((len(y), 28))
        blocks, rhs = [], []
        for t in range(99):
            blocks.append(c@power)
            rhs.append(normalized_y[:, t+1]-forced@c.T-normalized_u[:, t]@d.T)
            forced = forced@a.T+normalized_u[:, t]@b.T
            power = a@power
        design, target = np.concatenate(blocks), np.stack(rhs, axis=1).reshape(len(y), -1).T
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise FloatingPointError('nonfinite BLA state system')
    start = np.linalg.lstsq(design, target, rcond=1e-12)[0].T.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for t in range(99):
            start = start@a.T+normalized_u[:, t]@b.T
    output, final, singular, residual, rank = helper.replay(arrays, y, u, future)
    return output, start, final, {'state_units': 'author coordinates', 'rank': rank,
                                  'singular_values': singular, 'context_residual_norm': residual}


def replay(spec, arrays, normalizer, y_context, u_context, future_u):
    """Replay owned physical forecasts and caller states without targets or data I/O.

    Structural failures raise ValueError. Nonfinite arithmetic fails explicitly;
    author branches preserve their qualified helpers' numeric failure policy.
    No clipping, repair, alternate initialization, or checkpoint selection occurs.
    """
    batch, _, horizon = _validate(spec, arrays, normalizer, y_context, u_context, future_u)
    kind = spec['kind']
    if kind in ('varx', 'residual'):
        prediction, forecast, final, diagnostics = _lags(spec, arrays, normalizer, y_context, u_context, future_u)
        state_size = 6*spec['order']
    elif kind == 'bla':
        require(spec['order'] == 28, 'BLA order 28')
        prediction, forecast, final, diagnostics = _bla(arrays, y_context, u_context, future_u)
        state_size = 28
    else:
        helper = _helper('nllfr')
        state_size = helper.validate_model(arrays, production=False)
        require(spec['order'] == state_size, 'NL state order disagreement')
        outputs, forecasts, finals, requests = [], [], [], []
        for i in range(batch):
            output, end, start, detail = helper.replay_request(
                arrays, y_context[i:i+1], u_context[i:i+1], future_u[i:i+1],
                iterations=spec['context_iterations'])
            outputs.append(output); forecasts.append(start); finals.append(end); requests.append(detail)
        prediction, forecast, final = (np.concatenate(v, axis=0) for v in (outputs, forecasts, finals))
        diagnostics = {'state_units': 'author coordinates', 'per_request': requests}
    for name, value, shape in (('prediction', prediction, (batch, horizon, 3)),
                               ('forecast_state', forecast, (batch, state_size)),
                               ('final_state', final, (batch, state_size))):
        finite(value, shape, name)
    return {'prediction': prediction.copy(), 'forecast_state': forecast.copy(),
            'final_state': final.copy(), 'diagnostics': _json(diagnostics)}
