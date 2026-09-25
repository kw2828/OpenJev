# SPDX-License-Identifier: GPL-3.0-or-later
"""FIT-only tensor assembly and causal physical-unit BLA requests."""
from __future__ import annotations

import numpy as np

from .linear_context import predict

NAMES = ('A', 'B_u', 'C_y', 'D_yu', 'u_mean', 'u_std', 'y_mean', 'y_std', 'ts')
FIT_IDS = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV')
                for r in range(3) for p in range(2))
DEV_IDS = tuple(f'{a}-realization-{r}-period-{p}' for a in ('100mV', '200mV')
                for r in range(3, 6) for p in range(2))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def assemble_fit(records):
    """Preserve period axes and each amplitude's orthogonal FIT triplet."""
    records = tuple(records)
    require(len(records) == 12 and {r.record_id for r in records} == set(FIT_IDS),
            'exact duplicate-free twelve-record FIT roster required')
    lengths = {len(r.u) for r in records}
    require(len(lengths) == 1, 'FIT record lengths differ')
    n = lengths.pop()
    require(n >= 2, 'FIT records too short')
    by_id = {r.record_id: r for r in records}
    u = np.empty((n, 3, 6, 2), dtype=np.float64)
    y = np.empty_like(u)
    for ai, amplitude in enumerate(('100mV', '200mV')):
        for realization in range(3):
            for period in range(2):
                rid = f'{amplitude}-realization-{realization}-period-{period}'
                r = by_id[rid]
                require((r.partition, r.amplitude, r.realization, r.period, r.fs_hz)
                        == ('fit', amplitude, realization, period, 6400.0), 'FIT identity mismatch')
                for target, value in ((u, r.u), (y, r.y)):
                    require(value.shape == (n, 3) and value.dtype == np.float64
                            and np.isfinite(value).all(), 'FIT array contract')
                    target[:, :, ai*3+realization, period] = value
    for value in (u, y):
        scale = value.std(axis=(0, 2, 3))
        require(np.isfinite(scale).all() and (scale > 0).all(), 'positive finite FIT scales required')
    return u, y


def export_model(model):
    arrays = {name: np.array(getattr(model, name), dtype=np.float64, copy=True)
              for name in ('A', 'B_u', 'C_y', 'D_yu', 'ts')}
    arrays.update({name: np.array(getattr(model.norm, name), dtype=np.float64, copy=True)
                   for name in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    validate_model(arrays)
    return arrays


def validate_model(arrays):
    require(set(arrays) == set(NAMES), 'exact model array roster required')
    a = arrays['A']
    require(isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[0] > 0, 'invalid A dimensions')
    n = a.shape[0]
    shapes = {'A': (n, n), 'B_u': (n, 3), 'C_y': (3, n), 'D_yu': (3, 3), 'ts': ()}
    shapes.update({key: (3,) for key in ('u_mean', 'u_std', 'y_mean', 'y_std')})
    for key, shape in shapes.items():
        value = arrays[key]
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all(), 'invalid model '+key)
    require((arrays['u_std'] > 0).all() and (arrays['y_std'] > 0).all()
            and arrays['ts'] > 0, 'positive scales and sample time required')


def physical_request(arrays, y_context, u_context, future_u):
    """No targets or request cache; every call includes a fresh state solve."""
    validate_model(arrays)
    require(isinstance(y_context, np.ndarray) and y_context.ndim == 3
            and y_context.shape[0] > 0 and y_context.shape[1] >= 2, 'invalid context shape')
    batch, length = y_context.shape[:2]
    require(isinstance(future_u, np.ndarray) and future_u.ndim == 3, 'invalid future input')
    for value, shape in ((y_context, (batch, length, 3)),
                         (u_context, (batch, length-1, 3)),
                         (future_u, (batch, future_u.shape[1], 3))):
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all(), 'invalid physical request array')
    yc = (y_context-arrays['y_mean'])/arrays['y_std']
    uc = (u_context-arrays['u_mean'])/arrays['u_std']
    fu = (future_u-arrays['u_mean'])/arrays['u_std']
    forecast, state, diagnostics = predict(*(arrays[n] for n in NAMES[:4]), yc, uc, fu)
    physical = forecast*arrays['y_std']+arrays['y_mean']
    require(np.isfinite(physical).all(), 'nonfinite physical forecast')
    return physical, state, diagnostics


def requests(record, starts, context=100, horizon=128):
    require(isinstance(starts, np.ndarray) and starts.ndim == 1 and starts.dtype == np.int64
            and len(starts) > 0, 'nonempty int64 starts required')
    require(type(context) is int and context >= 2 and type(horizon) is int and horizon > 0,
            'invalid context/horizon')
    require((starts >= 0).all() and (starts+context+horizon <= len(record.u)).all(),
            'request crosses period boundary')
    yc = np.stack([record.y[s:s+context] for s in starts])
    uc = np.stack([record.u[s+1:s+context] for s in starts])
    fu = np.stack([record.u[s+context:s+context+horizon] for s in starts])
    return yc, uc, fu


def score(prediction, target, scale):
    require(all(isinstance(v, np.ndarray) and v.dtype == np.float64
                for v in (prediction, target, scale)), 'float64 NumPy score arrays required')
    require(prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[2] == 3
            and prediction.shape[0] > 0 and prediction.shape[1] > 0, 'matched nonempty forecasts required')
    require(scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(), 'invalid scoring scale')
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'nonfinite score inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        errors = ((prediction-target)/scale)**2
        mse = float(np.mean(errors))
        channel = np.sqrt(errors.mean(axis=(0, 1)))
        native = channel*scale
    require(np.isfinite(mse) and np.isfinite(channel).all() and np.isfinite(native).all(), 'score overflow')
    return {'rmse': float(np.sqrt(mse)), 'mse': mse, 'per_channel_rmse': channel.tolist(),
            'native_output_per_channel_rmse': native.tolist(),
            'requests': len(prediction), 'horizon': prediction.shape[1]}
