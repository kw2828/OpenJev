# SPDX-License-Identifier: GPL-3.0-or-later
"""One physical C100/H128 request for frozen amplitude-shift deployments.

Checkpoint authentication belongs to the registered caller. This module accepts
already admitted arrays, constructs no optimizer and reads no files. Deployment
construction is outside inference timing; request copies, normalization, state
inference, rollout and output conversion are inside request(). Author adapters
are the existing qualified NumPy implementations, with no JAX runtime needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch
from openjev_fsm_author import benchmark, linear_context, nllfr, nllfr_context_budget

from .fsm_linear import FIT_IDS, VARXModel
from .fsm_linear import predict as varx_predict
from .fsm_residual import FSMResidual

SPEC_KEYS = frozenset(('instance_id', 'family', 'seed', 'kind', 'order', 'alpha',
                       'residual_kind', 'mode', 'hidden_width', 'context_iterations'))
NORM_KEYS = ('u_mean', 'u_scale', 'y_mean', 'y_scale')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == np.float64
            and value.shape == shape and bool(np.isfinite(value).all()),
            'finite float64 array required: '+name)
    return value


def owned(value):
    return np.frombuffer(value.tobytes(order='C'), dtype=np.float64).reshape(value.shape)


def validate_spec(spec):
    require(type(spec) is dict and set(spec) == SPEC_KEYS, 'exact deployment specification')
    require(all(type(spec[k]) is str and spec[k] for k in ('instance_id', 'family')),
            'nonempty deployment identity')
    seed, kind, order = spec['seed'], spec['kind'], spec['order']
    require(seed is None or (type(seed) is int and 0 <= seed < 2**63), 'plain seed or None')
    require(spec['instance_id'] == spec['family']+(f'-{seed}' if seed is not None else ''),
            'instance identity disagrees with family and seed')
    require(kind in ('varx', 'residual', 'bla', 'nllfr'), 'declared deployment kind')
    require(type(order) is int and order > 0, 'positive integer order')
    if kind in ('varx', 'residual'):
        require(order <= 99, 'lag order fits observed context')
    if kind == 'varx':
        require(type(spec['alpha']) in (float, int) and np.isfinite(spec['alpha'])
                and spec['alpha'] > 0, 'positive historical VARX penalty metadata')
    else:
        require(spec['alpha'] is None, 'alpha applies only to VARX')
    if kind == 'residual':
        require(seed is not None and spec['residual_kind'] in ('affine', 'tanh')
                and spec['mode'] in ('feedback', 'output_only'), 'declared residual architecture')
        if spec['residual_kind'] == 'tanh':
            require(type(spec['hidden_width']) is int and spec['hidden_width'] > 0, 'positive tanh width')
        else:
            require(spec['hidden_width'] is None, 'affine has no hidden width')
    else:
        require(all(spec[k] is None for k in ('residual_kind', 'mode', 'hidden_width')),
                'residual-only specification fields')
    if kind == 'nllfr':
        require(type(spec['context_iterations']) is int and spec['context_iterations'] in (16, 64),
                'qualified nonlinear context budget')
    else:
        require(spec['context_iterations'] is None, 'context budget applies only to NL-LFR')
    if kind in ('bla', 'nllfr'):
        require(seed is None, 'author deployment seed is provenance, not a seed-matched slot')
    return spec


def normalize_spec(normalizer):
    require(type(normalizer) is dict and set(normalizer) == set(NORM_KEYS), 'exact common FIT scales')
    values = {k: owned(array(normalizer[k], (3,), k)) for k in NORM_KEYS}
    require((values['u_scale'] > 0).all() and (values['y_scale'] > 0).all(), 'positive common FIT scales')
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class Deployment:
    spec: MappingProxyType
    model: object
    normalizer: MappingProxyType | None
    storage: MappingProxyType


def make_deployment(spec, arrays, normalizer):
    """Copy admitted weights into one frozen deployment, without file access."""
    spec = validate_spec(spec)
    require(type(arrays) is dict and all(type(k) is str for k in arrays), 'named model arrays required')
    kind, p = spec['kind'], spec['order']
    norm = normalize_spec(normalizer)
    if kind == 'varx':
        require(set(arrays) == {'coefficients'}, 'exact VARX array roster')
        model = VARXModel(arrays['coefficients'], p, spec['alpha'], 12*(8192-p), FIT_IDS)
        numeric, metadata, state, normalization = model.coefficients.nbytes, 24, 6*p*8, 96
    elif kind == 'residual':
        model = FSMResidual(array(arrays.get('coefficients'), (3, 6*p+4), 'coefficients'),
                            order=p, mode=spec['mode'], residual_kind=spec['residual_kind'],
                            hidden_width=spec['hidden_width'] or 24, seed=spec['seed'])
        shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        require(set(arrays) == set(shapes), 'exact residual array roster')
        tensors = {k: torch.from_numpy(array(arrays[k], shape, k).copy()) for k, shape in shapes.items()}
        model.load_state_dict(tensors, strict=True)
        model.requires_grad_(False)
        model.eval()
        info = model.model_spec()
        numeric = info['parameter_bytes'] + info['buffer_bytes']
        metadata, state, normalization = info['numeric_metadata_bytes'], info['state_bytes_per_stream'], 96
    else:
        expected = benchmark.NAMES if kind == 'bla' else nllfr.NAMES
        require(set(arrays) == set(expected), 'exact author array roster')
        model = {k: owned(array(v, v.shape if isinstance(v, np.ndarray) else (), k)) for k, v in arrays.items()}
        (benchmark.validate_model if kind == 'bla' else nllfr.validate)(model)
        require(model['A'].shape == (p, p), 'declared author state dimension')
        numeric = sum(v.nbytes for v in model.values())
        metadata, state, normalization = (24 if kind == 'bla' else 72), 8*p, 0
        norm = None  # Author normalization is already held and counted in model arrays.
    storage = {'model_numeric_bytes': numeric, 'policy_metadata_bytes': metadata,
               'state_bytes': state, 'external_normalizer_bytes': normalization,
               'persistent_numeric_bytes': numeric+metadata+state+normalization,
               'scope': 'One deployment plus one stream state and required scales; no request cache. '
                        'Requests, temporary workspace and Python/string overhead excluded.'}
    return Deployment(MappingProxyType(spec.copy()), model, norm, MappingProxyType(storage))


def json_tree(value):
    if isinstance(value, np.ndarray):
        return json_tree(value.tolist())
    if isinstance(value, np.generic):
        return json_tree(value.item())
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), 'diagnostic keys must be strings')
        return {k: json_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_tree(v) for v in value]
    require(value is None or type(value) in (str, bool, int, float), 'unsupported diagnostic value')
    require(type(value) is not float or np.isfinite(value), 'nonfinite diagnostic value')
    return value


@torch.inference_mode()
def request(deployment, y_context, u_context, future_u):
    """Return physical outputs and owned states; targets are not accepted.

    Lag-model states use common FIT-standardized units. Author states use their
    native latent coordinates. Every call starts from the supplied context.
    """
    require(type(deployment) is Deployment, 'prepared deployment required')
    require(isinstance(y_context, np.ndarray) and y_context.ndim == 3 and len(y_context) > 0,
            'positive request batch')
    batch = len(y_context)
    y, u, future = (array(v, (batch, n, 3), name).copy(order='C') for v, n, name in
                    ((y_context, 100, 'y_context'), (u_context, 99, 'u_context'), (future_u, 128, 'future_u')))
    spec, model, norm = deployment.spec, deployment.model, deployment.normalizer
    kind, p = spec['kind'], spec['order']
    if kind in ('varx', 'residual'):
        yc, uc, fu = ((y-norm['y_mean'])/norm['y_scale'], (u-norm['u_mean'])/norm['u_scale'],
                      (future-norm['u_mean'])/norm['u_scale'])
        if kind == 'varx':
            state = np.concatenate((yc[:, -p:].reshape(batch, -1), uc[:, -p:].reshape(batch, -1)), axis=1)
            prediction = varx_predict(model, yc, uc, fu)
            final = np.concatenate((np.concatenate((yc[:, -p:], prediction), axis=1)[:, -p:].reshape(batch, -1),
                                    np.concatenate((uc[:, -p:], fu), axis=1)[:, -p:].reshape(batch, -1)), axis=1)
        else:
            state_tensor = model.condition(torch.from_numpy(yc), torch.from_numpy(uc))
            forecast_tensor, final_tensor = model.rollout(torch.from_numpy(fu), state_tensor)
            state, prediction, final = (t.numpy().copy() for t in (state_tensor, forecast_tensor, final_tensor))
        physical = prediction*norm['y_scale']+norm['y_mean']
        diagnostics = {'method': 'observed_lag_state', 'lag_order': p, 'context_rows_supplied': 100}
    elif kind == 'bla':
        yc, uc, fu = ((y-model['y_mean'])/model['y_std'], (u-model['u_mean'])/model['u_std'],
                      (future-model['u_mean'])/model['u_std'])
        matrices = tuple(model[k] for k in ('A', 'B_u', 'C_y', 'D_yu'))
        state, diagnostics = linear_context.condition(*matrices, yc, uc)
        prediction, final = linear_context.rollout(*matrices, fu, state)
        physical = prediction*model['y_std']+model['y_mean']
    else:
        state, diagnostics = nllfr_context_budget.condition(model, y, u, iterations=spec['context_iterations'])
        physical, final = nllfr.physical_rollout(model, future, state)
    state_width = 6*p if kind in ('varx', 'residual') else p
    return {'prediction': array(physical, (batch, 128, 3), 'physical prediction').copy(),
            'forecast_state': array(state, (batch, state_width), 'forecast state').copy(),
            'final_state': array(final, (batch, state_width), 'final state').copy(),
            'diagnostics': json_tree(diagnostics)}
