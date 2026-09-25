# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed16/64 context-solve budgets, separate from the frozen16-step adapter.

Only the outer iteration bound differs. Seed construction, scaled damped GN,
Armijo trials, stopping tests, final diagnostics and physical rollout retain
exactly the original arithmetic. The full request recomputes normalization,
linear seed, context solve and forecast; no prepared state or cache is retained.

A larger budget may reduce context loss. It does not guarantee better forecast
error, convergence, control stability or completion of the model's training.
"""
from __future__ import annotations

import numpy as np

from .benchmark import require
from .nllfr import physical_rollout, rollout, validate
from .nllfr_context import (
    ARMIJO,
    DAMPING,
    GRADIENT_TOL,
    RCOND,
    SCALE_FLOOR,
    TRIALS,
    _histories,
    _loss,
    linear_seed,
)

BUDGETS = (16, 64)


def _check_budget(iterations):
    require(type(iterations) is int and iterations in BUDGETS, 'fixed context budget must be int16 or64')


def _solve_one(arrays, target, inputs, seed, *, iterations):
    _check_budget(iterations)
    current = seed.copy()
    trace, accepted, evaluations, jacobian_evaluations = [], 0, 0, 0
    trial_attempts = 0
    initial_loss = None
    status = 'ITERATION_CAP'
    final_rank = None
    for iteration in range(iterations):
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


def condition(arrays, y_context, u_context, *, iterations=16):
    """Physical contexts only. Returns forecast state plus per-request diagnostics."""
    _check_budget(iterations)
    validate(arrays)
    _histories(y_context, u_context)
    y = (y_context-arrays['y_mean'])/arrays['y_std']
    u = (u_context-arrays['u_mean'])/arrays['u_std']
    states, solved, seeds, linear_diagnostics, diagnostics = [], [], [], [], []
    # Each request is independent, including the least-squares seed reduction.
    for i in range(len(y)):
        own_seed, own_linear = linear_seed(arrays, y[i:i+1], u[i:i+1])
        final, state, diagnostic = _solve_one(arrays, y[i:i+1, 1:], u[i:i+1], own_seed, iterations=iterations)
        states.append(final[0])
        solved.append(state[0])
        seeds.append(own_seed[0])
        linear_diagnostics.append(own_linear)
        diagnostic['linear_rank'] = own_linear['rank']
        diagnostics.append(diagnostic)
    return np.stack(states), {'requests': diagnostics, 'linear_seed_states': np.stack(seeds),
        'solved_context_start_states': np.stack(solved), 'linear_seed_diagnostics': linear_diagnostics,
        'policy': {'iterations': iterations, 'trials': TRIALS, 'damping': DAMPING,
                   'armijo': ARMIJO, 'gradient_tol': GRADIENT_TOL, 'scale_floor': SCALE_FLOOR, 'rcond': RCOND}}


def predict(arrays, y_context, u_context, future_u, *, iterations=16):
    """No future input participates in state inference."""
    forecast_state, diagnostic = condition(arrays, y_context, u_context, iterations=iterations)
    prediction, final = physical_rollout(arrays, future_u, forecast_state)
    return prediction, final, diagnostic
