"""Fixed finite Sinkhorn normalization for learned eight-state transport.

The convention is T[action,next,current]. A complete sweep normalizes rows
(dim2), then columns (dim1), in the log domain. The default is exactly64
sweeps, with no early stopping, retry, clipping, temperature or final repair.
The returned transition is approximately doubly stochastic only within the
declared residual tolerance. Some positive matrices converge arbitrarily
slowly, so a finite cap cannot guarantee admission for every finite input.

Autograd differentiates the executed finite sequence of logsumexp operations,
not an infinite balancing limit or an implicit projection. All intermediates
remain attached to that graph. Only reported scalar diagnostics are detached.
No parameters, model, optimizer, random draws or persistent state live here.

Work counters describe actual named forward operations, not FLOPs, backward
work, validation, tensor copies, elementwise exponentials inside logsumexp,
or scalar diagnostic reductions. A caller-owned work dictionary is updated
in place even when a later guard or callback fails. Other integer counters
are preserved. Counter increments precede attempted named Torch operations;
completed_sweeps advances only after both normalizations pass finite guards.
"""
from __future__ import annotations

import math

import torch

VERSION = 'finite-balanced-transition-v1'
DEFAULT_SWEEPS = 64
MAX_SWEEPS = 64
MAX_TOLERANCE = 1e-12
WORK_KEYS = (
    'balanced_transition_calls',
    'row_logsumexp_calls',
    'column_logsumexp_calls',
    'normalization_vectors',
    'normalization_entries',
    'completed_sweeps',
    'exponential_calls',
    'exponential_entries',
    'residual_sum_calls',
    'check_calls',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(bool(torch.isfinite(value).all()), name + ': finite values required')


def _work(value):
    if value is None:
        value = {}
    _require(type(value) is dict and all(type(count) is int and count >= 0 for count in value.values()),
             'work must be a dictionary of nonnegative Python integer counts')
    for name in WORK_KEYS:
        value.setdefault(name, 0)
    return value


def balanced_transition(logits, *, sweeps=DEFAULT_SWEEPS, tolerance=MAX_TOLERANCE,
                        check=lambda: None, work=None):
    """Return owned differentiable probabilities and their normalized logs.

    Input must be a finite CPU strided float64 tensor of shape(4,8,8).
    sweeps must be an exact Python int in1..64; the smaller values support
    explicit engineering failure fixtures. tolerance must be an exact finite
    Python float in(0,1e-12]. A scientific wrapper must freeze its own config.

    Returns transition, log_transition, diagnostics and the supplied work
    dictionary (or a newly allocated dictionary). Probabilities and logs do
    not share input storage or each other's storage. The input is never
    mutated. All valid configurations use fixed work, without convergence-
    based early exit. Final residual failure raises ValueError and retains
    the performed work in a supplied dictionary; there is no output fallback.
    External check exceptions propagate unchanged.

    For an admitted default call:64 row and64 column logsumexp calls,
    4096 normalized length8 vectors,32768 normalized entries,1 explicit
    exponential over256 entries,2 residual sums and66 callback checks.
    Basic config validation precedes the first callback; tensor validation
    follows it. Each completed sweep and the final measured residuals are
    followed by a callback before their respective next operation/admission.
    """
    counts = _work(work)
    counts['balanced_transition_calls'] += 1
    _require(type(sweeps) is int and 1 <= sweeps <= MAX_SWEEPS,
             'sweeps must be a Python int in1..64')
    _require(type(tolerance) is float and math.isfinite(tolerance) and 0 < tolerance <= MAX_TOLERANCE,
             'tolerance must be a finite Python float in(0,1e-12]')
    _require(callable(check), 'check must be callable')

    def tick():
        counts['check_calls'] += 1
        check()

    tick()
    _require(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu'
             and logits.layout == torch.strided and logits.dtype == torch.float64
             and tuple(logits.shape) == (4, 8, 8),
             'logits must be CPU strided float64 with shape(4,8,8)')
    _finite(logits, 'logits')
    log_transition = logits.clone()
    for _ in range(sweeps):
        counts['row_logsumexp_calls'] += 1
        counts['normalization_vectors'] += 32
        row_normalizer = torch.logsumexp(log_transition, dim=2, keepdim=True)
        counts['normalization_entries'] += 256
        log_transition = log_transition - row_normalizer
        _finite(log_transition, 'row-normalized logs')

        counts['column_logsumexp_calls'] += 1
        counts['normalization_vectors'] += 32
        column_normalizer = torch.logsumexp(log_transition, dim=1, keepdim=True)
        counts['normalization_entries'] += 256
        log_transition = log_transition - column_normalizer
        _finite(log_transition, 'column-normalized logs')
        counts['completed_sweeps'] += 1
        tick()

    counts['exponential_calls'] += 1
    counts['exponential_entries'] += 256
    transition = log_transition.exp()
    _finite(transition, 'transition')
    _require(bool(((transition > 0) & (transition < 1)).all()),
             'strict transition probabilities required; underflow or saturation is not repaired')
    counts['residual_sum_calls'] += 1
    row_residual = (transition.sum(dim=2) - 1).abs().amax()
    counts['residual_sum_calls'] += 1
    column_residual = (transition.sum(dim=1) - 1).abs().amax()
    row_error, column_error = float(row_residual.detach()), float(column_residual.detach())
    _require(math.isfinite(row_error) and math.isfinite(column_error), 'finite residuals required')
    diagnostics = {'sweeps': sweeps, 'tolerance': tolerance,
                   'row_residual_max': row_error, 'column_residual_max': column_error,
                   'minimum_probability': float(transition.detach().amin()),
                   'maximum_probability': float(transition.detach().amax()),
                   'converged': row_error <= tolerance and column_error <= tolerance}
    tick()
    _require(diagnostics['converged'],
             f'fixed-sweep residual exceeded tolerance: row={row_error!r}, column={column_error!r}, '
             f'tolerance={tolerance!r}, sweeps={sweeps}')
    return {'transition': transition, 'log_transition': log_transition,
            'diagnostics': diagnostics, 'work': counts}
