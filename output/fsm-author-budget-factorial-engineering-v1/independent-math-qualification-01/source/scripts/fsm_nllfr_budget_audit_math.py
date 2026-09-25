# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent 16/64-direction context replay, with no producer imports.

The pinned original independent auditor supplies the recurrence, derivative,
linear seed and loss. This module only parameterizes its literal GN policy.
It performs no study admission, file decoding, training or timing replay.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'scripts/audit_fsm_author_nllfr.py'
HELPER_SHA = '43c4787e316bc5c0a89653fece0c0694cd9f88041628b798c02382feb09dfa32'
if hashlib.sha256((ROOT/HELPER).read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('qualified independent helper changed')
_spec = importlib.util.spec_from_file_location('held_nllfr_budget_replay_math', ROOT/HELPER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
require, finite, validate_model = old.require, old.finite, old.validate_model
trajectory, linear_seed, context_loss = old.trajectory, old.linear_seed, old.context_loss


def policy(*, iterations=16):
    """Return a new policy mapping without sharing or mutating the held policy."""
    require(type(iterations) is int and iterations in (16, 64), 'iterations must be exactly 16 or 64')
    return {'iterations': iterations, 'trials': 8, 'damping': .001, 'armijo': .0001,
            'gradient_tol': 1e-8, 'scale_floor': 1e-8, 'rcond': 1e-12}


def solve_context(m, target, u, seed, *, iterations=16):
    """Estimate one normalized context-start state, then advance it to forecast time.

    A capped or stalled solve returns its last accepted finite iterate. Only a
    nonfinite line-search trial is rejected locally; no seed/state fallback is
    applied when the current trajectory, Jacobian or factorization fails.
    """
    policy(iterations=iterations)
    n = validate_model(m, production=False)
    require(isinstance(u, np.ndarray) and u.ndim == 3 and u.shape[0] == 1
            and u.shape[1] > 0, 'one nonempty independent context')
    finite(u, (1, u.shape[1], 3)); finite(target, u.shape); finite(seed, (1, n))
    x = seed.copy()
    trace, accepted, calls, jac_calls, proposals = [], 0, 0, 0, 0
    status, first = 'ITERATION_CAP', None
    for iteration in range(iterations):
        prediction, _, sensitivity = trajectory(m, u, x, jacobian=True)
        calls += 1; jac_calls += 1
        loss, r = context_loss(prediction, target)
        if first is None:
            first = loss
        j = sensitivity.reshape(target.size, x.size)/np.sqrt(target.size)
        norms = np.linalg.norm(j, axis=0)
        if not np.isfinite(norms).all():
            raise FloatingPointError('nonfinite Jacobian column norm')
        row = {'iteration': iteration, 'objective': loss, 'trials': []}
        trace.append(row)
        if norms.max() == 0:
            status = 'ZERO_JACOBIAN'; break
        scale = np.maximum(norms, 1e-8*norms.max())
        k = j/scale
        g = k.T@r
        require(np.isfinite(g).all() and np.isfinite(k).all(), 'nonfinite scaled gradient')
        row['scaled_gradient_inf'] = float(np.max(abs(g)))
        singular = np.linalg.svd(k, compute_uv=False)
        row['scaled_jacobian_rank'] = int(np.sum(singular > 1e-12*singular[0]))
        if row['scaled_gradient_inf'] <= 1e-8*(1+np.linalg.norm(r)):
            status = 'GRADIENT_TOL'; break
        augmented = np.concatenate([k, np.sqrt(.001)*np.eye(x.size)], axis=0)
        p = np.linalg.lstsq(augmented, np.concatenate([-r, np.zeros(x.size)]), rcond=1e-12)[0]
        delta, slope = p/scale, float(np.dot(g, p))
        require(np.isfinite(delta).all() and math.isfinite(slope), 'nonfinite context direction')
        if slope >= 0:
            row['reason'] = 'nonnegative_directional_derivative'; status = 'STALLED'; break
        accepted_here = False
        for trial in range(8):
            alpha = 2.**(-trial)
            with np.errstate(over='ignore', invalid='ignore'):
                proposed = x+alpha*delta[None]
            detail = {'alpha': alpha, 'accepted': False, 'nonfinite': False}
            proposals += 1
            try:
                if not np.isfinite(proposed).all():
                    raise FloatingPointError('nonfinite proposal')
                calls += 1
                trial_prediction = trajectory(m, u, proposed)[0]
                trial_loss = context_loss(trial_prediction, target)[0]
                detail['objective'] = trial_loss
                if trial_loss < loss and trial_loss <= loss+.0001*alpha*slope:
                    x = proposed; accepted += 1; accepted_here = True; detail['accepted'] = True
            except FloatingPointError:
                detail['nonfinite'] = True
            row['trials'].append(detail)
            if accepted_here:
                break
        if not accepted_here:
            status = 'STALLED'; break
    prediction, final, derivative = trajectory(m, u, x, jacobian=True)
    calls += 1; jac_calls += 1
    last, _ = context_loss(prediction, target)
    require(first is not None and last <= first, 'context objective worsened')
    s = np.linalg.svd(derivative.reshape(target.size, x.size)/np.sqrt(target.size), compute_uv=False)
    require(np.isfinite(s).all(), 'nonfinite final Jacobian diagnostics')
    record = {'status': status, 'initial_objective': first, 'final_objective': last,
              'directions_considered': len(trace), 'accepted_steps': accepted,
              'trajectory_evaluations': calls, 'jacobian_evaluations': jac_calls,
              'trial_attempts': proposals, 'final_jacobian_rank': int(np.sum(s > 1e-12*s[0])),
              'final_jacobian_singular_values': s, 'trace': trace,
              'status_scope': 'bounded context solve; no optimality or forecasting-accuracy certificate'}
    return final, x, record


def replay_request(m, y_context, u_context, future_u, *, iterations=16):
    """One independent physical request. Future inputs never enter conditioning."""
    selected_policy = policy(iterations=iterations)
    n = validate_model(m, production=False)
    require(isinstance(y_context, np.ndarray) and y_context.ndim == 3
            and y_context.shape[0] == 1 and y_context.shape[1] >= 2, 'one independent context')
    require(isinstance(future_u, np.ndarray) and future_u.ndim == 3, 'future input rank')
    finite(y_context, (1, y_context.shape[1], 3)); finite(u_context, (1, y_context.shape[1]-1, 3))
    finite(future_u, (1, future_u.shape[1], 3))
    y, u = (y_context-m['y_mean'])/m['y_std'], (u_context-m['u_mean'])/m['u_std']
    seed, linear = linear_seed(m, y, u)
    forecast, solved, detail = solve_context(m, y[:, 1:], u, seed, iterations=iterations)
    detail['linear_rank'] = linear['rank']
    output, final, _ = trajectory(m, (future_u-m['u_mean'])/m['u_std'], forecast)
    physical = output*m['y_std']+m['y_mean']
    require(np.isfinite(physical).all(), 'nonfinite physical outputs')
    finite(final, (1, n))
    return physical, final, forecast, {'requests': [detail], 'linear_seed_states': seed,
        'solved_context_start_states': solved, 'linear_seed_diagnostics': [linear], 'policy': selected_policy}
