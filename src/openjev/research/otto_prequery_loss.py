"""Fixed CPU float32 nonquery and pre-assimilation score calibration losses.

The nonquery arithmetic is the original separately eligible-centered expression.
The additional prior term centers all four actions, matching the model's query
innovation coordinates. Teacher tensors and importance weights are detached.
No targets are passed into a model, and no parameters or state are changed here.
"""
from __future__ import annotations

import torch

VERSION = "otto-prequery-loss-v1"
OBJECTIVES = ("mse", "query_aux")
SCALE = 64
EPISODES = 54


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _scores(prediction, targets):
    require(isinstance(prediction, torch.Tensor) and prediction.device.type == "cpu"
            and prediction.dtype == torch.float32 and prediction.ndim == 3
            and prediction.shape[-1] == 4 and all(d > 0 for d in prediction.shape),
            "CPU float32 batch/time/four scores")
    require(isinstance(targets, torch.Tensor) and targets.device == prediction.device
            and targets.dtype == torch.float32 and targets.shape == prediction.shape,
            "matching CPU float32 targets")
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(targets).all()),
            "finite prediction and teacher scores")
    return targets.detach()


def nonquery_rows(prediction, targets, legal):
    """Original per-row legal-centered MSE /64, with neutral padding divisor one."""
    targets = _scores(prediction, targets)
    require(isinstance(legal, torch.Tensor) and legal.dtype == torch.bool
            and legal.device == prediction.device and legal.shape == prediction.shape,
            "matching legal-action mask")
    allowed = legal.to(torch.float32)
    count = allowed.sum(dim=-1, keepdim=True).clamp(min=1)
    pred = prediction / SCALE
    target = targets / SCALE
    difference = (pred - (pred * allowed).sum(dim=-1, keepdim=True) / count
                  - target + (target * allowed).sum(dim=-1, keepdim=True) / count)
    return (difference.square() * allowed).sum(dim=-1) / count[:, :, 0]


def prior_rows(prior, targets):
    """All-four separately centered MSE /64, before assimilating that query."""
    targets = _scores(prior, targets)
    pred = prior / SCALE
    target = targets / SCALE
    difference = pred - pred.mean(dim=-1, keepdim=True) - target + target.mean(dim=-1, keepdim=True)
    return difference.square().mean(dim=-1)


def _weights(weights, shape, name):
    require(isinstance(weights, torch.Tensor) and weights.device.type == "cpu"
            and weights.dtype in (torch.float32, torch.float64) and weights.shape == shape
            and bool(torch.isfinite(weights).all()) and bool((weights >= 0).all()),
            f"finite nonnegative {name}")
    return weights.detach().to(torch.float32)


def weighted_loss(prediction, prior, targets, legal, weights,
                  prior_targets, prior_weights, prior_mask, *, objective):
    """Return total/nonquery/prior scalar terms after the common 54/B multiplier.

The mse control has exactly the old nonquery objective and no prior gradient.
query_aux adds the prior term with fixed coefficient one. Episode weights retain
the fixed full denominator even when an episode has no later query. The caller
must compare the model's prior_mask with the data mask before invoking this loss.
    """
    require(objective in OBJECTIVES, "declared fixed objective")
    row_losses = nonquery_rows(prediction, targets, legal)
    require(prediction.shape[0] <= 6, "at most six full episode lanes")
    weight = _weights(weights, row_losses.shape, "nonquery weights")
    require(bool(legal.any(dim=-1)[weight > 0].all()), "weighted nonqueries have eligible actions")
    multiplier = EPISODES / prediction.shape[0]
    nonquery = (row_losses * weight).sum() * multiplier
    extra = nonquery.new_zeros(())
    if objective == "query_aux":
        require(isinstance(prior_mask, torch.Tensor) and prior_mask.device.type == "cpu"
                and prior_mask.dtype == torch.bool and prior_mask.shape == row_losses.shape,
                "matching prior mask")
        require(bool((weight[prior_mask] == 0).all()), "disjoint weighted prior and nonquery rows")
        for name, value in (("prediction", prior), ("target", prior_targets)):
            require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                    and value.dtype == torch.float32 and value.shape == prediction.shape,
                    f"matching prior {name} shape and dtype")
        require(isinstance(prior_weights, torch.Tensor) and prior_weights.device.type == "cpu"
                and prior_weights.dtype in (torch.float32, torch.float64)
                and prior_weights.shape == row_losses.shape, "matching prior weights shape and dtype")
        if bool(prior_mask.any()):
            # Select before every numerical read: all unscored prior slots may be poison.
            selected = prior[prior_mask].unsqueeze(1)
            selected_target = prior_targets[prior_mask].detach().unsqueeze(1)
            selected_weight = _weights(prior_weights[prior_mask], selected.shape[:1], "selected prior weights")
            require(bool((selected_weight > 0).all()), "positive selected prior weights")
            extra = (prior_rows(selected, selected_target)[:, 0] * selected_weight).sum() * multiplier
    total = nonquery + extra
    require(bool(torch.isfinite(total)), "finite weighted objective")
    return {"total": total, "nonquery": nonquery, "prior": extra}
