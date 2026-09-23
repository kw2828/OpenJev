"""Standalone CPU float32 SPO+ loss for four legal-action costs.

Elmachtoub and Grigas, "Smart Predict, then Optimize", Definition 3.5:
https://arxiv.org/html/1710.08005v5#S3.SS2
    max_{w in S} (c - 2*z)^T w + 2*z^T w_star(c) - min_{w in S} c^T w
Here S is the simplex over legal actions, z=prediction/64, c=teacher/64,
and w_star is uniform over the legal EXACT minima of the original float32
teacher. The implementation uses the algebraically equivalent expression
    max_legal [(c_a - min_legal c) - 2*(z_a - mean_teacher_minima z)].
Prediction differences are formed around the first teacher minimum before
averaging. Scaling precedes subtraction to keep finite float32 extremes safe.

The maximum uses torch.max(dim=-1), whose backward selects the FIRST maximizing
action in numeric action order. It does not average maximizing subgradients.
Exact teacher ties are determined before scaling; signed zeros compare equal.
No 1e-10 near-tie tolerance is used. A deployment selector using that tolerance
is a different decision rule from the exact optimization oracle in this loss.
These are implementation choices, not calibration or downstream-performance
claims. The component does not choose a combined objective or change a model.
"""
from __future__ import annotations

import torch

VERSION = "otto-spo-plus-loss-v1"
SCALE = 64
EPISODES = 54
MAX_BATCH = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _inputs(prediction, targets, legal):
    require(isinstance(prediction, torch.Tensor) and prediction.device.type == "cpu"
            and prediction.dtype == torch.float32 and prediction.ndim == 3
            and prediction.shape[-1] == 4 and all(d > 0 for d in prediction.shape),
            "CPU float32 nonempty batch/time/four prediction scores")
    require(isinstance(targets, torch.Tensor) and targets.device == prediction.device
            and targets.dtype == torch.float32 and targets.shape == prediction.shape,
            "matching CPU float32 teacher scores")
    require(isinstance(legal, torch.Tensor) and legal.device == prediction.device
            and legal.dtype == torch.bool and legal.shape == prediction.shape,
            "matching CPU bool legal-action mask")
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(targets).all()),
            "finite prediction and teacher scores, including padding and illegal slots")
    return targets.detach()


def spo_plus_rows(prediction, targets, legal):
    """Return float32 [B,T] rows; empty legal masks give connected exact zeros.

All teacher values are detached. Illegal actions have zero prediction gradient.
Single-action rows have exact zero loss and gradient. Inputs are not mutated.
    """
    targets = _inputs(prediction, targets, legal)
    valid = legal.any(dim=-1, keepdim=True)
    minimum = targets.masked_fill(~legal, float("inf")).min(dim=-1, keepdim=True).values
    minimum = torch.where(valid, minimum, torch.zeros_like(minimum))
    best = legal & (targets == minimum)
    count = best.sum(dim=-1, keepdim=True).clamp(min=1)
    anchor = best.to(torch.int64).argmax(dim=-1, keepdim=True)

    pred = prediction / SCALE
    centered = pred - pred.gather(dim=-1, index=anchor)
    reference_offset = (centered * best.to(torch.float32)).sum(dim=-1, keepdim=True) / count
    teacher_gap = targets / SCALE - minimum / SCALE
    margins = teacher_gap - 2 * (centered - reference_offset)
    candidates = margins.masked_fill(~legal, float("-inf"))
    # A neutral finite maximum for padding avoids an all-negative-infinity row.
    candidates = torch.where(valid, candidates, torch.zeros_like(candidates))
    rows = candidates.max(dim=-1).values
    connected_zero = (prediction * 0).sum(dim=-1)
    rows = torch.where(valid[:, :, 0], rows, connected_zero)
    require(bool(torch.isfinite(rows).all()), "finite SPO+ row losses")
    return rows


def weighted_spo_plus_loss(prediction, targets, legal, weights):
    """Return sum(row_loss * detached_float32_weight) * (54 / B), B <= 6.

Weights must be finite nonnegative CPU float32/float64 [B,T]. Their declared
importance weights and the full 54-episode denominator are preserved; there is
no normalization by realized weight mass or number of nonempty rows. Float64
weights follow the existing float32 training cast. Positive weights must remain
positive and finite after that cast, and must have at least one legal action.
An all-zero-weight batch retains a zero-gradient connection to predictions.
    """
    rows = spo_plus_rows(prediction, targets, legal)
    require(prediction.shape[0] <= MAX_BATCH, "at most six episode lanes")
    require(isinstance(weights, torch.Tensor) and weights.device.type == "cpu"
            and weights.dtype in (torch.float32, torch.float64) and weights.shape == rows.shape
            and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()),
            "finite nonnegative matching CPU float32/float64 weights")
    detached = weights.detach()
    positive = detached > 0
    require(bool(legal.any(dim=-1)[positive].all()), "positive-weight rows have legal actions")
    weight = detached.to(torch.float32)
    require(bool(torch.isfinite(weight).all()) and bool((weight[positive] > 0).all()),
            "positive weights remain finite and positive in float32")
    total = (rows * weight).sum() * (EPISODES / prediction.shape[0])
    require(bool(torch.isfinite(total)), "finite weighted SPO+ objective")
    return total
