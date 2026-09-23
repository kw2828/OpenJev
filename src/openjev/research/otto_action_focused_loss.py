"""Fixed auxiliary calibration objective with an optional unit-weight SPO+ term.

Both cells use the unchanged legal-centered nonquery MSE and all-four later-query
prior MSE from otto_prequery_loss, each scaled by 64 before squaring. Only the
``spo`` treatment adds the qualified exact-oracle SPO+ nonquery loss. It uses the
same declared importance weights and 54/B multiplier, with coefficient one.
There is no realized-weight normalization, new label, or adaptive coefficient.
The ``aux`` control never evaluates SPO+ during optimization.
"""
from __future__ import annotations

import torch

from openjev.research import otto_prequery_loss as base
from openjev.research import otto_spo_plus_loss as spo_base

VERSION = "otto-action-focused-loss-v1"
OBJECTIVES = ("aux", "spo")
SCALE, EPISODES = base.SCALE, base.EPISODES
nonquery_rows = base.nonquery_rows
prior_rows = base.prior_rows
spo_plus_rows = spo_base.spo_plus_rows


def weighted_loss(prediction, prior, targets, legal, weights,
                  prior_targets, prior_weights, prior_mask, *, objective):
    """Return scalar tensors total/nonquery/prior/spo for the declared objective.

The three AUX tensors are the original query_aux result objects, including the
original addition order. SPO treatment total is ``base_total + spo``. Targets
and importance weights are detached inside the unchanged component functions.
The caller owns query-versus-nonquery support validation and the 54-episode
cohort. Row helpers are re-exported for final rescore without another 54/B factor.
    """
    base.require(objective in OBJECTIVES, "fixed action-focused objective")
    result = base.weighted_loss(prediction, prior, targets, legal, weights,
                               prior_targets, prior_weights, prior_mask, objective="query_aux")
    extra = result["total"].new_zeros(())
    total = result["total"]
    if objective == "spo":
        extra = spo_base.weighted_spo_plus_loss(prediction, targets, legal, weights)
        total = total + extra
    base.require(bool(torch.isfinite(total)), "finite combined action-focused objective")
    return {"total": total, "nonquery": result["nonquery"], "prior": result["prior"], "spo": extra}
