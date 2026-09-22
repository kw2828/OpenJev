"""Matched analytic-preference and continuation-cost regression primitives.

This component constructs targets and a common loss only. It reads no files,
fits no parameters, chooses no actions and admits no training experiment. A
caller supplies the same frozen TRAIN rows and episode identities to both arms.
The global target scale is TRAIN-derived; validation/evaluation targets must not
be used to recompute it. Continuous cost gaps are never normalized per panel.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

VERSION = "otto-cost-regression-v1"
KINDS = ("analytic", "continuation")
TEMPERATURE, RANGE_FLOOR, SCALE_FLOOR = 0.25, 1e-8, 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _immutable(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def episode_weights(episode_ids):
    """Owned float64 weights N/(E*n_episode), with equal total episode weight.

    An anchor is one row regardless of its number of continuation replicates or
    eligible actions. No support/precision filtering or target-dependent weights
    are introduced. Positive weights have mean one up to float64 roundoff.
    """
    _require(isinstance(episode_ids, (tuple, list)) and len(episode_ids) > 0,
             "nonempty episode-ID sequence required")
    _require(all(type(value) is str and value for value in episode_ids),
             "episode IDs must be nonempty strings")
    counts = Counter(episode_ids)
    n, episodes = len(episode_ids), len(counts)
    weights = np.asarray([n / (episodes * counts[key]) for key in episode_ids], dtype=np.float64)
    _require(np.isfinite(weights).all() and (weights > 0).all(), "invalid episode weights")
    return _immutable(weights)


def _inputs(costs, allowed, episode_ids):
    _require(isinstance(costs, np.ndarray) and costs.dtype == np.float64
             and costs.ndim == 2 and costs.shape[0] > 0 and costs.shape[1] == 4,
             "costs must be nonempty float64[N,4]")
    _require(isinstance(allowed, np.ndarray) and allowed.dtype == np.bool_
             and allowed.shape == costs.shape and allowed.any(axis=1).all(),
             "nonempty boolean eligible-action rows required")
    _require(np.isfinite(costs[allowed]).all() and np.isposinf(costs[~allowed]).all(),
             "costs must be finite on eligible actions and positive infinity only on blocked actions")
    weights = episode_weights(episode_ids)
    _require(len(weights) == len(costs), "cost rows and episode IDs must align")
    return weights


def build_targets(costs, allowed, episode_ids, *, kind):
    """Construct one arm's fixed TRAIN targets without changing its action gaps.

    Analytic: (h-eligible_mean(h))/(0.25*max(eligible_range(h),1e-8)).
    This is the centered negative-logit representation of the historical
    analytic preference distribution, not its previous cross-entropy objective.

    Continuation: Q-eligible_mean(Q), where the supplied Q is the mean capped
    continuation cost in moves. The caller authenticates complete paired panels;
    this function does not infer panel provenance, uncertainty or optimality.

    The single scale for this arm is max(1e-8, sqrt(sum_i(w_i*mean_a(d_i^2))/N)).
    Means over actions use only each row's eligible set. Return the unscaled
    float64 targets, scale, float32 scaled targets, both weight dtypes and mask.
    Blocked target entries are exactly zero. Every returned array is an owned,
    immutable copy. Float32 casting can round very small gaps; original float64
    targets are retained, with no clipping, hard labels or confidence threshold.
    """
    _require(kind in KINDS, "unknown regression target kind")
    weights = _inputs(costs, allowed, episode_ids)
    count = allowed.sum(axis=1, dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        finite_costs = np.where(allowed, costs, 0.0)
        mean = np.sum(finite_costs, axis=1, dtype=np.float64) / count
        centered = np.where(allowed, finite_costs - mean[:, None], 0.0)
        if kind == "analytic":
            maximum = np.max(np.where(allowed, costs, -np.inf), axis=1)
            minimum = np.min(np.where(allowed, costs, np.inf), axis=1)
            centered /= (TEMPERATURE * np.maximum(maximum - minimum, RANGE_FLOOR))[:, None]
        row_squares = np.sum(centered * centered, axis=1, dtype=np.float64) / count
        rms = float(np.sqrt(np.sum(weights * row_squares, dtype=np.float64) / len(costs)))
        scale = max(SCALE_FLOOR, rms)
        scaled = np.asarray(centered / scale, dtype=np.float32)
        training_weights = weights.astype(np.float32)
    _require(np.isfinite(centered).all() and np.isfinite(scale) and scale > 0
             and np.isfinite(scaled).all(), "nonfinite regression targets or scale")
    _require(np.isfinite(training_weights).all() and (training_weights > 0).all(),
             "invalid float32 episode weights")
    return {"kind": kind, "centered_float64": _immutable(centered), "scale": scale,
            "scaled_float32": _immutable(scaled), "weights_float64": weights,
            "weights_float32": _immutable(training_weights), "allowed": _immutable(allowed)}


def targets(analytic_costs, continuation_means, allowed, episode_ids):
    """Construct the two matched TRAIN arms on exactly the same supplied rows.

    Both inputs use float64[N,4] with +inf on blocked actions. Continuation means
    remain continuous; neither their sign nor their magnitude is clipped. The
    two separate global scales match weighted target RMS without imposing equal
    per-state gaps. No validation/evaluation data are accepted implicitly.
    """
    analytic = build_targets(analytic_costs, allowed, episode_ids, kind="analytic")
    continuation = build_targets(continuation_means, allowed, episode_ids, kind="continuation")
    _require(np.array_equal(analytic["weights_float64"], continuation["weights_float64"]),
             "the two arms require identical row weighting")
    return {"version": VERSION, "analytic": analytic, "continuation": continuation}


def training_losses(model, x, scaled_targets, allowed):
    """Return N unweighted losses for the qualified dense D4 training route.

    For every original row: evaluate all eight qualified feature views, transform
    targets/masks with the same action permutations, center predictions AND
    targets over eligible actions, average squared error over eligible actions,
    then average the eight views. The caller applies weights_float32 and its
    batch mean identically to both arms. No optimizer, detach, gradient clipping,
    entropy objective or inference-time ensemble is introduced here.
    """
    import torch

    from openjev.research.otto_symmetry_head import INPUT_DIM, MAX_BATCH

    _require(isinstance(model, torch.nn.Module) and getattr(model, "kind", None) == "dense_augmented",
             "qualified dense_augmented head required")
    _require(isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] == INPUT_DIM
             and 0 < x.shape[0] <= MAX_BATCH and x.dtype == torch.float32 and x.device.type == "cpu"
             and bool(torch.isfinite(x).all()), "bounded finite CPU float32 features required")
    _require(isinstance(scaled_targets, torch.Tensor) and isinstance(allowed, torch.Tensor)
             and scaled_targets.shape == allowed.shape == (len(x), 4)
             and scaled_targets.dtype == torch.float32 and allowed.dtype == torch.bool
             and scaled_targets.device == allowed.device == x.device
             and bool(torch.isfinite(scaled_targets).all()) and bool(allowed.any(dim=-1).all())
             and bool((scaled_targets[~allowed] == 0).all()),
             "finite float32 targets, zero blocked targets and nonempty boolean masks required")
    _require(model.permutations.shape == (8, 4) and model.permutations.dtype == torch.int64
             and model.permutations.device == x.device, "qualified D4 action permutations required")
    inverse = torch.argsort(model.permutations, dim=-1)
    target, mask = scaled_targets[:, inverse], allowed[:, inverse]
    predicted = model.core(model.views(x))
    _require(predicted.shape == target.shape and predicted.dtype == torch.float32
             and predicted.device == x.device and bool(torch.isfinite(predicted).all()),
             "finite four-action costs on all eight views required")
    counts = mask.sum(dim=-1, keepdim=True)
    centered_prediction = predicted - predicted.masked_fill(~mask, 0).sum(dim=-1, keepdim=True) / counts
    centered_target = target - target.masked_fill(~mask, 0).sum(dim=-1, keepdim=True) / counts
    errors = (centered_prediction - centered_target).masked_fill(~mask, 0)
    losses = ((errors * errors).sum(dim=-1) / counts.squeeze(-1)).mean(dim=-1)
    _require(losses.shape == (len(x),) and bool(torch.isfinite(losses).all()),
             "finite per-row centered regression losses required")
    return losses
