# SPDX-License-Identifier: GPL-3.0-or-later
"""Causal context initialization of an explicit linear state-space model.

The convention is y[k] = C x[k] + D u[k], followed by
x[k+1] = A x[k] + B u[k]. Context input j belongs to observed output
j+1. Output 0 has no supplied matching input and is excluded from fitting.
The returned context state belongs to the first forecast output, whose input
has not been used. These are positional contracts, not a physical delay claim.

All operations use finite CPU NumPy float64 arrays. Matrices, histories and
states are caller supplied; nothing is retained between requests. This is a
standard observability least-squares initializer, not a stability guarantee.
"""
from __future__ import annotations

import numpy as np

RCOND = 1e-12
CHANNELS = 3


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.float64:
        raise ValueError(f'{name} must be a CPU NumPy float64 array')
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name} has invalid shape or nonfinite values')


def _matrices(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> int:
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] < 1:
        raise ValueError('A must be a nonempty square matrix')
    n = A.shape[0]
    for value, shape, name in ((A, (n, n), 'A'), (B, (n, CHANNELS), 'B'),
                               (C, (CHANNELS, n), 'C'), (D, (CHANNELS, CHANNELS), 'D')):
        _array(value, shape, name)
    return n


def _context(y_context: np.ndarray, u_context: np.ndarray) -> tuple[int, int]:
    if not isinstance(y_context, np.ndarray) or y_context.ndim != 3:
        raise ValueError('y_context must have shape [batch, context, 3]')
    batch, length = y_context.shape[:2]
    if batch < 1 or length < 2:
        raise ValueError('context needs a nonempty batch and at least two outputs')
    _array(y_context, (batch, length, CHANNELS), 'y_context')
    _array(u_context, (batch, length-1, CHANNELS), 'u_context')
    return batch, length


def _future(future_u: np.ndarray, batch: int) -> int:
    if not isinstance(future_u, np.ndarray) or future_u.ndim != 3:
        raise ValueError('future_u must have shape [batch, horizon, 3]')
    horizon = future_u.shape[1]
    _array(future_u, (batch, horizon, CHANNELS), 'future_u')
    return horizon


def condition(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
              y_context: np.ndarray, u_context: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit x at output 1, then advance every observed pair to forecast time.

    ``u_context[:, j]`` matches ``y_context[:, j+1]``. The fit stacks
    ``C @ A**j`` and subtracts the known forced response, including D u[j].
    ``numpy.linalg.lstsq(..., rcond=1e-12)`` gives the minimum Euclidean norm
    solution if the context is rank deficient. There is no jitter, alternative
    solve or future-input warmup. Minimum norm is coordinate dependent when
    the state is not fully identifiable from this context.

    Diagnostics contain the numerical rank, singular values and residual
    L2 norm for each batch member. Diagnostic arrays and the state are owned.
    """
    n = _matrices(A, B, C, D)
    batch, length = _context(y_context, u_context)
    pairs = length-1
    observability = np.empty((pairs*CHANNELS, n), dtype=np.float64)
    rhs = np.empty((batch, pairs, CHANNELS), dtype=np.float64)
    forced = np.zeros((batch, n), dtype=np.float64)
    block = C.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for j in range(pairs):
            observability[j*CHANNELS:(j+1)*CHANNELS] = block
            rhs[:, j] = y_context[:, j+1] - forced @ C.T - u_context[:, j] @ D.T
            forced = forced @ A.T + u_context[:, j] @ B.T
            if j+1 < pairs:
                block = block @ A
    if not np.isfinite(observability).all() or not np.isfinite(rhs).all():
        raise ValueError('nonfinite observability system')
    targets = rhs.transpose(1, 2, 0).reshape(pairs*CHANNELS, batch)
    solution, _, rank, singular = np.linalg.lstsq(observability, targets, rcond=RCOND)
    with np.errstate(over='ignore', invalid='ignore'):
        residual_norm = np.linalg.norm(observability @ solution-targets, axis=0)
        state = solution.T.copy()
        for j in range(pairs):
            state = state @ A.T + u_context[:, j] @ B.T
    if not all(np.isfinite(value).all() for value in (solution, singular, residual_norm, state)):
        raise ValueError('nonfinite least-squares state or diagnostics')
    diagnostics = {'rcond': RCOND, 'state_dimension': n, 'context_pairs': pairs,
                   'equations': pairs*CHANNELS, 'rank': int(rank),
                   'rank_deficient': bool(rank < n),
                   'solution_convention': 'minimum Euclidean norm at the second observed output',
                   'singular_values': singular.copy(), 'residual_norm': residual_norm.copy()}
    return state.copy(), diagnostics


def _step(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
          state: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return state @ C.T + u @ D.T, state @ A.T + u @ B.T


def step(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
         state: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read y[k] using u[k], then advance and return the owned x[k+1]."""
    n = _matrices(A, B, C, D)
    if not isinstance(state, np.ndarray) or state.ndim != 2 or state.shape[0] < 1:
        raise ValueError('state must have shape [nonempty batch, state dimension]')
    _array(state, (state.shape[0], n), 'state')
    _array(u, (state.shape[0], CHANNELS), 'u')
    with np.errstate(over='ignore', invalid='ignore'):
        output, next_state = _step(A, B, C, D, state, u)
    if not np.isfinite(output).all() or not np.isfinite(next_state).all():
        raise ValueError('nonfinite linear prediction or state')
    return output, next_state


def rollout(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
            future_u: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast using only future inputs and the explicit caller state.

    Validation is outside the step loop. The complete returned forecast and
    final state must be finite. An empty horizon returns owned empty outputs
    and a copy of the unchanged caller state.
    """
    n = _matrices(A, B, C, D)
    if not isinstance(state, np.ndarray) or state.ndim != 2 or state.shape[0] < 1:
        raise ValueError('state must have shape [nonempty batch, state dimension]')
    batch = state.shape[0]
    _array(state, (batch, n), 'state')
    horizon = _future(future_u, batch)
    result = np.empty((batch, horizon, CHANNELS), dtype=np.float64)
    current = state.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for j in range(horizon):
            result[:, j], current = _step(A, B, C, D, current, future_u[:, j])
    if not np.isfinite(result).all() or not np.isfinite(current).all():
        raise ValueError('nonfinite linear prediction or state')
    return result, current.copy()


def predict(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
            y_context: np.ndarray, u_context: np.ndarray,
            future_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Condition causally, returning forecast, final state and fit diagnostics."""
    batch, _ = _context(y_context, u_context)
    _future(future_u, batch)
    state, diagnostics = condition(A, B, C, D, y_context, u_context)
    forecast, final = rollout(A, B, C, D, future_u, state)
    return forecast, final, diagnostics
