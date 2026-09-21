"""One-parameter output calibration; no actor state, corpus or model access."""
from __future__ import annotations

import math

import numpy as np

BETA_BOUNDS = (0.125, 8.0)
BISECTION_STEPS = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(logs, target):
    logs = np.asarray(logs)
    target = np.asarray(target)
    require(logs.ndim == 2 and logs.shape[0] > 0 and logs.shape[1] >= 2
            and logs.dtype in (np.dtype('float32'), np.dtype('float64')), 'Finite candidate log matrix')
    require(target.dtype == np.int64 and target.shape == (len(logs),)
            and np.all((target >= 0) & (target < logs.shape[1])), 'Int64 target indices')
    support = np.isfinite(logs)
    require(np.all(support.sum(1) >= 2) and np.isneginf(logs[~support]).all(), 'Finite support and negative-infinite padding')
    require(support[np.arange(len(target)), target].all(), 'Target in candidate support')
    values = logs.astype(np.float64)
    require(np.abs(np.exp(values).sum(1)-1).max() <= 2e-6, 'Original probability mass tolerance')
    return values, target, support


def normalize(logs, beta):
    require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0, 'Positive finite inverse temperature')
    scaled = logs.astype(np.float64) * beta
    centered = scaled - scaled.max(1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(1, keepdims=True))


def output_logs(logs, beta):
    """Normalized calibrated outputs at every beta, including one."""
    require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0, 'Positive finite inverse temperature')
    return normalize(logs, beta)


def objective_and_derivative(logs, target, beta):
    """Convex NLL in beta. Center before products to avoid 0 * -infinity."""
    centered = logs - logs.max(1, keepdims=True)
    adjusted = normalize(centered, beta)
    finite = np.isfinite(centered)
    expected = (np.exp(adjusted) * np.where(finite, centered, 0)).sum(1)
    observed = centered[np.arange(len(target)), target]
    nll = -adjusted[np.arange(len(target)), target]
    return float(nll.mean()), float((expected-observed).mean())


def fit_temperature(logs, target):
    """Fit solely on the supplied calibration endpoints with a fixed recipe.

    Convex inverse-temperature optimization has two endpoint checks and exactly
    64 bisections for an interior optimum. If every row is exactly uniform,
    preserve beta=1 as the predeclared unidentifiable-objective rule.
    """
    values, target, support = validate(logs, target)
    low, high = BETA_BOUNDS
    _, dlo = objective_and_derivative(values, target, low)
    _, dhi = objective_and_derivative(values, target, high)
    require(math.isfinite(dlo) and math.isfinite(dhi) and dlo <= dhi+1e-12, 'Convex finite endpoint derivatives')
    maximum = values.max(1, keepdims=True)
    uniform = np.all(np.where(support, values == maximum, True))
    steps = 0
    if uniform:
        beta, location = 1.0, 'unidentified_uniform_identity'
    elif dlo >= 0:
        beta, location = low, 'lower_bound'
    elif dhi <= 0:
        beta, location = high, 'upper_bound'
    else:
        left, right = low, high
        for _ in range(BISECTION_STEPS):
            middle = (left+right)/2
            _, derivative = objective_and_derivative(values, target, middle)
            require(math.isfinite(derivative), 'Finite optimization derivative')
            if derivative < 0:
                left = middle
            else:
                right = middle
        beta, location, steps = (left+right)/2, 'interior', BISECTION_STEPS
    before, _ = objective_and_derivative(values, target, 1.0)
    after, derivative = objective_and_derivative(values, target, beta)
    require(after <= before+1e-12, 'Calibration objective cannot increase at fitted optimum')
    adjusted = output_logs(values, beta)
    require(np.array_equal(adjusted.argmax(1), values.argmax(1)), 'Positive temperature preserves first argmax')
    return {'beta': beta, 'temperature': 1.0/beta, 'beta_bounds': list(BETA_BOUNDS),
            'location': location, 'bisection_steps': steps, 'endpoints': len(target),
            'raw_nll_before': float(-values[np.arange(len(target)), target].mean()),
            'normalized_nll_before': before, 'normalized_nll_after': after,
            'normalization_only_nll_drift': before-float(-values[np.arange(len(target)), target].mean()),
            'derivative_at_lower_bound': dlo, 'derivative_at_upper_bound': dhi,
            'derivative_at_solution': derivative, 'selection_data': 'calibration_only',
            'choices_unchanged': True}
