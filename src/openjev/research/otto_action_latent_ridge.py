"""Fixed causal action moments and solved ridge baselines for action gaps.

Brier-target least squares is followed by Euclidean probability-simplex projection
and fixed smoothing. This is a small transparent baseline, not the previously
trained P4 readout (whose refreshed observations violate this task's boundary).
"""
from __future__ import annotations

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def design(prefix, lengths, actions, continuation=None, found=None):
    require(prefix.ndim == 3 and prefix.shape[2] == 31 and np.isfinite(prefix).all(), 'prefix')
    n, h = actions.shape
    require(n == len(prefix) and 1 <= h <= 8 and actions.dtype == np.int64
            and np.all((actions >= 0) & (actions < 4)), 'actions')
    require(lengths.shape == (n,) and lengths.dtype == np.int64 and np.all((lengths >= 1) & (lengths <= prefix.shape[1])), 'prefix lengths')
    normal = continuation is not None
    require((found is not None) == normal, 'paired normal inputs')
    if normal:
        require(continuation.shape == (n, h, 31) and found.shape == (n, h)
                and found.dtype == np.bool_ and np.isfinite(continuation).all(), 'normal observations')
    context = prefix[np.arange(n), lengths - 1].astype(np.float64)
    require(prefix.shape[1] == 9, 'nine-slot observed prefix')
    history = np.where(np.arange(9)[None, :, None] < lengths[:, None, None], prefix, 0).reshape(n, 279).astype(np.float64)
    terminated = np.zeros(n, dtype=bool)
    rows = []
    for t in range(h):
        a = actions[:, t:t + 1] if normal else actions[:, :t + 1]
        length = a.shape[1]
        pos = np.arange(1, length + 1, dtype=np.float64)
        basis = np.stack([np.ones(length), np.sin(pos * np.pi / 8), np.cos(pos * np.pi / 8),
                          np.sin(pos * np.pi / 4), np.cos(pos * np.pi / 4)], axis=1)
        onehot = np.eye(4)[a]
        moments = np.einsum('bta,tf->baf', onehot, basis).reshape(n, 20) / length
        row = np.concatenate([history, context, moments, np.eye(4)[a[:, -1]],
                              np.full((n, 1), length / 8), np.ones((n, 1))], axis=1)
        if normal and rows:
            row = np.where(terminated[:, None], rows[-1], row)
        rows.append(row)
        if normal:
            terminated |= found[:, t]
            context = np.where(terminated[:, None], context, continuation[:, t])
    result = np.stack(rows, axis=1)
    require(result.shape == (n, h, 336) and np.isfinite(result).all(), 'fixed causal design')
    return result


def solve(x, targets, weights, lam=.0001):
    require(x.ndim == 2 and x.shape[1] == 336 and targets.ndim == 2 and len(targets) == len(x)
            and weights.shape == (len(x),) and np.isfinite(x).all() and np.isfinite(targets).all()
            and np.isfinite(weights).all() and np.all(weights >= 0) and lam > 0, 'ridge inputs')
    # All coefficients, including intercept, receive the fixed ridge penalty.
    a, b = x * np.sqrt(weights[:, None]), targets * np.sqrt(weights[:, None])
    coefficient = np.linalg.solve(a.T @ a + lam * np.eye(336), a.T @ b)
    require(np.isfinite(coefficient).all(), 'finite solved coefficients')
    return coefficient


def project_probabilities(value, smoothing=.000001):
    require(value.ndim >= 2 and value.shape[-1] == 5 and np.isfinite(value).all()
            and 0 < smoothing < 1, 'finite probability regression and smoothing')
    ordered = -np.sort(-value, axis=-1)
    cumulative = np.cumsum(ordered, axis=-1) - 1.
    active = ordered - cumulative / np.arange(1, 6) > 0
    rho = active.sum(axis=-1) - 1
    theta = np.take_along_axis(cumulative, rho[..., None], axis=-1)[..., 0] / (rho + 1)
    projected = np.maximum(value - theta[..., None], 0.)
    return (1 - smoothing) * projected + smoothing / 5
