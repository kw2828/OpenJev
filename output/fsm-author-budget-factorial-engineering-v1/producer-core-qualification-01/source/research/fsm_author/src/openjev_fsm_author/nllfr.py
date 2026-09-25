# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit NumPy two-hidden-layer ReLU NL-LFR and state sensitivities.

This is a conventional deployment adapter for the pinned author model.
Outputs precede transitions. There is no observation injection or hidden cache.
"""
from __future__ import annotations

import numpy as np

from .benchmark import NAMES as LINEAR_NAMES
from .benchmark import require, validate_model

EXTRA_NAMES = ('B_w', 'C_z', 'D_yw', 'D_zu', 'W0', 'b0', 'W1', 'b1', 'W2', 'b2')
NAMES = (*LINEAR_NAMES, *EXTRA_NAMES)


def validate(arrays):
    require(isinstance(arrays, dict) and set(arrays) == set(NAMES), 'exact NL-LFR array roster required')
    validate_model({name: arrays[name] for name in LINEAR_NAMES})
    nx = len(arrays['A'])
    for name in EXTRA_NAMES:
        value = arrays[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and np.isfinite(value).all(), 'invalid NL-LFR '+name)
    require(arrays['C_z'].ndim == arrays['B_w'].ndim == arrays['W0'].ndim == 2,
            'NL-LFR matrix ranks')
    nz, nw, width = arrays['C_z'].shape[0], arrays['B_w'].shape[1], arrays['W0'].shape[0]
    require(min(nz, nw, width) > 0, 'positive NL-LFR dimensions')
    shapes = {'B_w': (nx, nw), 'C_z': (nz, nx), 'D_yw': (3, nw), 'D_zu': (nz, 3),
              'W0': (width, nz), 'b0': (width,), 'W1': (width, width), 'b1': (width,),
              'W2': (nw, width), 'b2': (nw,)}
    require(all(arrays[k].shape == shape for k, shape in shapes.items()), 'NL-LFR matrix shapes')
    return nx, nz, nw, width


def export(model):
    """Copy numeric parameters without converting lower precision to float64."""
    from equinox.nn._mlp import _identity

    from .benchmark import export_model

    raw = [np.asarray(getattr(model, name)) for name in ('A', 'B_u', 'C_y', 'D_yu', 'ts')]
    raw += [np.asarray(getattr(model.norm, name)) for name in ('u_mean', 'u_std', 'y_mean', 'y_std')]
    require(all(value.dtype == np.float64 for value in raw), 'author base parameters must be float64')
    arrays = export_model(model)
    arrays.update({name: np.array(getattr(model, name), copy=True)
                   for name in ('B_w', 'C_z', 'D_yw', 'D_zu')})
    network = model.func_static
    require(network.layers == 2 and network.bias is True, 'two hidden layers with bias required')
    require(getattr(network.activation, '__name__', None) == 'relu', 'ReLU required')
    require(network.model.final_activation is _identity, 'qualified identity final activation required')
    require(len(network.model.layers) == 3, 'three affine layers required')
    for i, layer in enumerate(network.model.layers):
        arrays[f'W{i}'] = np.array(layer.weight, copy=True)
        arrays[f'b{i}'] = np.array(layer.bias, copy=True)
    validate(arrays)
    return arrays


def _step(arrays, state, u, sensitivity):
    z = state @ arrays['C_z'].T + u @ arrays['D_zu'].T
    h0 = z @ arrays['W0'].T + arrays['b0']
    h1 = np.maximum(h0, 0.) @ arrays['W1'].T + arrays['b1']
    w = np.maximum(h1, 0.) @ arrays['W2'].T + arrays['b2']
    y = state @ arrays['C_y'].T + u @ arrays['D_yu'].T + w @ arrays['D_yw'].T
    next_state = state @ arrays['A'].T + u @ arrays['B_u'].T + w @ arrays['B_w'].T
    if sensitivity is None:
        return y, next_state, None, None
    # JAX ReLU selects derivative zero exactly at zero. Final layer is linear.
    j0 = (h0 > 0.)[:, :, None] * (arrays['W0'] @ arrays['C_z'])[None]
    j1 = (h1 > 0.)[:, :, None] * (arrays['W1'][None] @ j0)
    jw = arrays['W2'][None] @ j1
    y_jacobian = (arrays['C_y'][None] + arrays['D_yw'][None] @ jw) @ sensitivity
    next_sensitivity = (arrays['A'][None] + arrays['B_w'][None] @ jw) @ sensitivity
    return y, next_state, y_jacobian, next_sensitivity


def rollout(arrays, inputs, state, *, jacobian=False):
    """Normalized trajectory and optional derivative with respect to initial state.

    Inputs [batch,time,3], state [batch,nx]; returns outputs, final state and
    output Jacobian [batch,time,3,nx] (or None). No future observations exist in
    this interface. Transient Jacobian workspace is recomputed for every call.
    """
    nx, _, _, _ = validate(arrays)
    require(isinstance(inputs, np.ndarray) and inputs.ndim == 3
            and inputs.shape[0] > 0 and inputs.shape[2] == 3, 'invalid rollout inputs')
    batch, horizon = inputs.shape[:2]
    require(type(jacobian) is bool, 'boolean Jacobian flag required')
    require(isinstance(state, np.ndarray) and state.shape == (batch, nx), 'invalid rollout state')
    require(all(v.dtype == np.float64 and np.isfinite(v).all() for v in (state, inputs)),
            'finite float64 rollout arrays required')
    current = state.copy()
    values = np.empty((batch, horizon, 3), dtype=np.float64)
    derivatives = np.empty((batch, horizon, 3, nx), dtype=np.float64) if jacobian else None
    sensitivity = np.broadcast_to(np.eye(nx), (batch, nx, nx)).copy() if jacobian else None
    with np.errstate(over='ignore', invalid='ignore'):
        for j in range(horizon):
            values[:, j], current, derivative, sensitivity = _step(arrays, current, inputs[:, j], sensitivity)
            if derivatives is not None:
                derivatives[:, j] = derivative
    if not np.isfinite(values).all() or not np.isfinite(current).all():
        raise FloatingPointError('nonfinite NL-LFR rollout')
    if derivatives is not None:
        if not np.isfinite(derivatives).all() or not np.isfinite(sensitivity).all():
            raise FloatingPointError('nonfinite NL-LFR sensitivities')
    return values, current.copy(), derivatives


def physical_rollout(arrays, inputs, state):
    validate(arrays)
    require(isinstance(inputs, np.ndarray) and inputs.dtype == np.float64, 'physical inputs must be float64')
    output, final, _ = rollout(arrays, (inputs-arrays['u_mean'])/arrays['u_std'], state)
    physical = output*arrays['y_std'] + arrays['y_mean']
    require(np.isfinite(physical).all(), 'nonfinite physical outputs')
    return physical, final
