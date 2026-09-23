"""Expose the existing pre-assimilation forecast without changing its model.

The caller must freeze ``otto_cross_query_scores.py`` at BASE_SOURCE_SHA256.
Its forward, validation, initialization, correction and readout arithmetic are
inherited unchanged. In that held implementation, rank-two readouts occur only
at later queries; nonquery sequence readouts have rank three. This adapter
captures those existing tensors, with their gradient graphs, and maps them to
the validated chronological query positions. It performs no extra model call.

Only genuine query scores enter the inherited model. The caller owns the
objective: any prior target must be detached and scored before assimilation,
over all four centered coordinates divided by64, matching the innovation.
No prior target or skipped teacher label is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from openjev.research import otto_cross_query_scores as base

VERSION = "otto-prequery-scores-v1"
BASE_SOURCE_SHA256 = "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
KINDS = ("innovation", "innovation_gru")
PARAMETERS = {kind: base.PARAMETERS[kind] for kind in KINDS}
Carry = base.Carry
detach_carry = base.detach_carry


@dataclass(frozen=True)
class Forecast:
    """Owned output containers; floating tensors retain their gradient graphs.

    ``prediction`` and ``carry`` are the unchanged inherited results.
    ``prior`` is raw float32[B,T,4], with positive zeros outside ``prior_mask``.
    ``prior_mask`` selects active absolute steps4,8,... only. First queries,
    nonqueries and padding never supply a prior. Frozen fields do not make
    tensors read-only; callers must detach carry before the next TBPTT chunk.
    """

    prediction: torch.Tensor
    prior: torch.Tensor
    prior_mask: torch.Tensor
    carry: Carry


def parameter_count(kind):
    base.require(kind in KINDS, "declared prequery family")
    return PARAMETERS[kind]


class PrequeryHead(base.CrossQueryHead):
    """Single active forward per instance, without persistent episode state.

    Temporary readout references are always cleared in ``finally``, including
    failed forwards. They are neither a model input nor saved module state.
    Reentrant/concurrent use of one instance is rejected; use separate heads.
    """

    def __init__(self, kind, seed):
        base.require(kind in KINDS, "declared prequery family")
        super().__init__(kind, seed)
        self._captured: list[torch.Tensor] | None = None

    def _prediction(self, hidden, raw_anchor):
        value = super()._prediction(hidden, raw_anchor)
        if self._captured is not None and hidden.ndim == 2:
            self._captured.append(value)
        return value

    def forward(self, features, query_scores, lengths, query_mask, *, carry=None, episode_ends):
        """Use the inherited signature and return a :class:`Forecast`.

        The complete inherited validation runs before any capture mapping.
        Numerical causality follows the original forward: a prior at time t
        cannot depend on Q_t or any later query. Validation may still reject
        malformed later active inputs before any result can be returned.
        """
        base.require(self._captured is None, "prequery forward cannot be reentered")
        self._captured = []
        try:
            prediction, next_carry = super().forward(
                features, query_scores, lengths, query_mask,
                carry=carry, episode_ends=episode_ends,
            )
            batch, span, _ = prediction.shape
            starts = torch.zeros(batch, dtype=torch.int64, device="cpu") if carry is None else carry.absolute_step
            absolute = starts[:, None] + torch.arange(span, device="cpu")[None, :]
            prior_mask = query_mask & (absolute >= base.PERIOD)
            rows = [torch.zeros(batch, base.SCORE_DIM, dtype=torch.float32, device="cpu")
                    for _ in range(span)]
            used = 0
            for offset in range(0, span, base.PERIOD):
                index = torch.nonzero(prior_mask[:, offset], as_tuple=False).flatten()
                if not index.numel():
                    continue
                base.require(used < len(self._captured), "all later-query readouts captured")
                value = self._captured[used]
                base.require(tuple(value.shape) == (index.numel(), base.SCORE_DIM),
                             "exact later-query readout geometry")
                rows[offset] = rows[offset].index_copy(0, index, value)
                used += 1
            base.require(used == len(self._captured), "no unassigned rank-two readout")
            return Forecast(prediction, torch.stack(rows, dim=1), prior_mask, next_carry)
        finally:
            self._captured = None


def make_head(kind, seed):
    """Identically initialized held model, preserving the caller's Torch RNG."""
    base.require(kind in KINDS and type(seed) is int and 0 <= seed < 2**32,
                 "declared family and uint32 seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = PrequeryHead(kind, seed)
    base.require(sum(p.numel() for p in model.parameters()) == parameter_count(kind), "exact parameter count")
    return model
