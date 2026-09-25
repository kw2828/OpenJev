# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded observation-only NL-LFR state inference, without future inputs.

A fixed damped Gauss-Newton/Armijo solve starts from the final model's linear
submodel. This is an inference adapter, not a new learning algorithm. Only
nonfinite line-search trials are rejected; a bad initial state/Jacobian/solve
fails rather than switching methods. See the frozen causal contract.
"""
from __future__ import annotations

import numpy as np

from .benchmark import require
from .nllfr import physical_rollout, rollout, validate

ITERATIONS = 16
TRIALS = 8
DAMPING = 1e-3
ARMIJO = 1e-4
GRADIENT_TOL = 1e-8
SCALE_FLOOR = 1e-8
RCOND = 1e-12


def _histories(y, u):
    require(isinstance(y, np.ndarray) and y.ndim == 3 and y.shape[0] > 0
            and y.shape[1] >= 2 and y.shape[2] == 3, 'context shape')
    require(isinstance(u, np.ndarray) and u.shape == (y.shape[0], y.shape[1]-1, 3), 'paired input shape')
    require(all(a.dtype == np.float64 and np.isfinite(a).all() for a in (y, u)),
            'finite float64 contexts required')


def linear_seed(arrays, y, u):
    """Estimate x_(s+1), not forecast-time state, using final linear matrices."""
    nx, _, _, _ = validate(arrays)
    _histories(y, u)
    batch, length = u.shape[:2]
    a, b, c, d = (arrays[name] for name in ('A', 'B_u', 'C_y', 'D_yu'))
    observability = np.empty((length*3, nx))
    rhs = np.empty((batch, length, 3))
    forced = np.zeros((batch, nx))
    block = c.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for k in range(length):
            observability[k*3:(k+1)*3] = block
            rhs[:, k] = y[:, k+1]-forced @ c.T-u[:, k] @ d.T
            forced = forced @ a.T+u[:, k] @ b.T
            if k+1 < length:
                block = block @ a
    require(np.isfinite(observability).all() and np.isfinite(rhs).all(), 'nonfinite linear seed system')
    state, _, rank, singular = np.linalg.lstsq(observability, rhs.transpose(1, 2, 0).reshape(length*3, batch), rcond=RCOND)
    require(np.isfinite(state).all() and np.isfinite(singular).all(), 'nonfinite linear seed')
    return state.T.copy(), {'rank': int(rank), 'singular_values': singular.copy(),
                            'rcond': RCOND, 'time': 'second observed output'}


def _loss(predicted, target):
    with np.errstate(over='ignore', invalid='ignore'):
        residual = (predicted-target).reshape(-1)
        standardized = residual/np.sqrt(len(residual))
        value = float(np.dot(standardized, standardized)/2.)
    if not np.isfinite(value) or not np.isfinite(standardized).all():
        raise FloatingPointError('nonfinite context objective')
    return value, standardized


def _solve_one(arrays, target, inputs, seed):
    current = seed.copy()
    trace, accepted, evaluations, jacobian_evaluations = [], 0, 0, 0
    trial_attempts = 0
    initial_loss = None
    status = 'ITERATION_CAP'
    final_rank = None
    for iteration in range(ITERATIONS):
        predicted, _, derivative = rollout(arrays, inputs, current, jacobian=True)
        evaluations += 1
        jacobian_evaluations += 1
        objective, residual = _loss(predicted, target)
        if initial_loss is None:
            initial_loss = objective
        jac = derivative.reshape(len(residual), current.shape[1])/np.sqrt(len(residual))
        column_norm = np.linalg.norm(jac, axis=0)
        if not np.isfinite(column_norm).all():
            raise FloatingPointError('nonfinite Jacobian column norm')
        maximum = float(np.max(column_norm))
        row = {'iteration': iteration, 'objective': objective, 'trials': []}
        trace.append(row)
        if maximum == 0:
            status, final_rank = 'ZERO_JACOBIAN', 0
            break
        scaling = np.maximum(column_norm, SCALE_FLOOR*maximum)
        scaled_jac = jac/scaling
        gradient = scaled_jac.T @ residual
        require(np.isfinite(gradient).all() and np.isfinite(scaled_jac).all(), 'nonfinite scaled gradient')
        gradient_inf = float(np.max(np.abs(gradient)))
        row['scaled_gradient_inf'] = gradient_inf
        singular = np.linalg.svd(scaled_jac, compute_uv=False)
        final_rank = int(np.sum(singular > RCOND*singular[0]))
        row['scaled_jacobian_rank'] = final_rank
        if gradient_inf <= GRADIENT_TOL*(1.+np.linalg.norm(residual)):
            status = 'GRADIENT_TOL'
            break
        system = np.vstack((scaled_jac, np.sqrt(DAMPING)*np.eye(current.shape[1])))
        rhs = np.concatenate((-residual, np.zeros(current.shape[1])))
        direction, _, _, _ = np.linalg.lstsq(system, rhs, rcond=RCOND)
        delta = direction/scaling
        slope = float(np.dot(gradient, direction))
        require(np.isfinite(delta).all() and np.isfinite(slope), 'nonfinite context direction')
        if slope >= 0:
            status = 'STALLED'
            row['reason'] = 'nonnegative_directional_derivative'
            break
        selected = False
        for trial in range(TRIALS):
            alpha = 2.**(-trial)
            with np.errstate(over='ignore', invalid='ignore'):
                proposal = current+alpha*delta[None]
            item = {'alpha': alpha, 'accepted': False, 'nonfinite': False}
            trial_attempts += 1
            try:
                if not np.isfinite(proposal).all():
                    raise FloatingPointError('nonfinite proposal')
                evaluations += 1
                prediction, _, _ = rollout(arrays, inputs, proposal)
                trial_loss, _ = _loss(prediction, target)
                item['objective'] = trial_loss
                if trial_loss < objective and trial_loss <= objective+ARMIJO*alpha*slope:
                    current = proposal
                    selected = True
                    accepted += 1
                    item['accepted'] = True
            except FloatingPointError:
                item['nonfinite'] = True
            row['trials'].append(item)
            if selected:
                break
        if not selected:
            status = 'STALLED'
            break
    prediction, final, derivative = rollout(arrays, inputs, current, jacobian=True)
    evaluations += 1
    jacobian_evaluations += 1
    final_loss, _ = _loss(prediction, target)
    require(initial_loss is not None and final_loss <= initial_loss, 'context objective worsened')
    final_jacobian = derivative.reshape(-1, current.shape[1])/np.sqrt(derivative.shape[1]*3)
    final_singular = np.linalg.svd(final_jacobian, compute_uv=False)
    require(np.isfinite(final_singular).all(), 'nonfinite final Jacobian diagnostics')
    final_rank = int(np.sum(final_singular > RCOND*final_singular[0]))
    return final, current, {'status': status, 'initial_objective': initial_loss,
        'final_objective': final_loss, 'directions_considered': len(trace), 'accepted_steps': accepted,
        'trajectory_evaluations': evaluations, 'jacobian_evaluations': jacobian_evaluations,
        'trial_attempts': trial_attempts,
        'final_jacobian_rank': final_rank, 'final_jacobian_singular_values': final_singular.copy(), 'trace': trace,
        'status_scope': 'bounded context solve; no optimality or forecasting-accuracy certificate'}


def condition(arrays, y_context, u_context):
    """Physical contexts only. Returns forecast state plus per-request diagnostics."""
    validate(arrays)
    _histories(y_context, u_context)
    y = (y_context-arrays['y_mean'])/arrays['y_std']
    u = (u_context-arrays['u_mean'])/arrays['u_std']
    states, solved, seeds, linear_diagnostics, diagnostics = [], [], [], [], []
    # Each request is independent, including the least-squares seed reduction.
    for i in range(len(y)):
        own_seed, own_linear = linear_seed(arrays, y[i:i+1], u[i:i+1])
        final, state, diagnostic = _solve_one(arrays, y[i:i+1, 1:], u[i:i+1], own_seed)
        states.append(final[0])
        solved.append(state[0])
        seeds.append(own_seed[0])
        linear_diagnostics.append(own_linear)
        diagnostic['linear_rank'] = own_linear['rank']
        diagnostics.append(diagnostic)
    return np.stack(states), {'requests': diagnostics, 'linear_seed_states': np.stack(seeds),
        'solved_context_start_states': np.stack(solved), 'linear_seed_diagnostics': linear_diagnostics,
        'policy': {'iterations': ITERATIONS, 'trials': TRIALS, 'damping': DAMPING,
                   'armijo': ARMIJO, 'gradient_tol': GRADIENT_TOL, 'scale_floor': SCALE_FLOOR, 'rcond': RCOND}}


def predict(arrays, y_context, u_context, future_u):
    """No future input participates in state inference."""
    forecast_state, diagnostic = condition(arrays, y_context, u_context)
    prediction, final = physical_rollout(arrays, future_u, forecast_state)
    return prediction, final, diagnostic
