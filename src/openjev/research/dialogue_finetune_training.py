"""Endpoint-weighted loss for complete-dialogue gradient accumulation.

No corpus, encoder, optimizer or state updates are performed here. The caller
runs each full public stream and accumulates this loss across the effective
batch before one optimizer update. Every microbatch uses the same denominator:
the total number of eligible endpoints in that actual effective batch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

VERSION = "dialogue-finetune-training-v1"


def supervised_loss(logp, rows, weights, batch_endpoint_count):
    """Weighted endpoint NLL sum divided by the whole-batch endpoint count.

    logp is [1,T,Q,C], including all public unscored turns. Rows are separate
    evaluator records with time/query_position/label_index/stratum_index. They
    are gathered in their supplied source order. Strata are unmentioned
    retention, assigned retention, changed, in that order. No per-dialogue mean
    or sum-of-weights normalization is applied. The caller, which owns batch
    membership, must validate the shared denominator against all its rows.

    Padded candidates may have -inf log probability. Selecting one, or another
    zero-probability target, fails the finite training-loss requirement instead
    of introducing a probability floor. Saved-output reporting may separately
    report infinite NLL; it must not use this training-only rejection as a filter.
    """
    if not (isinstance(logp, torch.Tensor) and logp.is_floating_point()
            and logp.ndim == 4 and logp.shape[0] == 1 and all(n > 0 for n in logp.shape)):
        raise ValueError("Floating log probabilities must have nonempty [1,T,Q,C] shape")
    if not (isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and len(rows)):
        raise ValueError("Nonempty scored endpoint rows required")
    if type(batch_endpoint_count) is not int or batch_endpoint_count < len(rows):
        raise ValueError("Positive whole-batch endpoint count must cover this dialogue")
    weight = torch.as_tensor(weights, dtype=logp.dtype, device=logp.device)
    if weight.shape != (3,) or not torch.isfinite(weight).all() or not (weight > 0).all():
        raise ValueError("Three finite positive stratum weights required")
    if torch.isnan(logp).any() or torch.isposinf(logp).any():
        raise ValueError("NaN or positive-infinite log probabilities")

    indices, strata, seen = [], [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Scored endpoint mapping required")
        values = tuple(row.get(key) for key in ("time", "query_position", "label_index", "stratum_index"))
        bounds = (*logp.shape[1:], 3)
        if any(type(value) is not int or not 0 <= value < bound
               for value, bound in zip(values, bounds, strict=True)):
            raise ValueError("Endpoint or stratum index outside supported dimensions")
        time, query, label, stratum = values
        if (time, query) in seen:
            raise ValueError("Duplicate scored dialogue/query endpoint")
        seen.add((time, query))
        indices.append((time, query, label))
        strata.append(stratum)
    selection = torch.tensor(indices, dtype=torch.long, device=logp.device)
    selected = logp[0, selection[:, 0], selection[:, 1], selection[:, 2]]
    if not torch.isfinite(selected).all():
        raise ValueError("Selected target must have finite log probability")
    stratum_indices = torch.tensor(strata, dtype=torch.long, device=logp.device)
    result = -(selected * weight[stratum_indices]).sum() / batch_endpoint_count
    if not torch.isfinite(result):
        raise ValueError("Nonfinite accumulated training loss")
    return result
