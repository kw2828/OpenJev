"""Pure targets and horizon-eight loss for a fixed 32-history TRAIN bank.

This component does not collect, fit, schedule, load files or use randomness.
Each row is an originating public prefix, not an independent history draw.
Found histories have zero costs and remain in every denominator. All four
actions must share the same legal support, as enforced by the caller's data
contract. The teacher's raw costs are nonnegative, in original teacher units.

Targets are centered Q/64 in float64. Model outputs retain the existing CPU
float32 interface, and only their eighth forecast is scored. The shared scale
comes from the second moment of all sampled TRAIN contrasts, including zeros,
not the lower second moment of mean labels. Its float32 rounding matches the
model's fixed scale buffer. No claim of exact conditional expectation is made:
the mean32 target is an empirical conditional mean from a finite shared bank.

The mean of squared errors to all 32 draws and squared error to their mean
differ by target variance, independent of the prediction. Their gradients are
therefore equal at a fixed prediction. Sampling one draw changes gradient
noise under a fixed optimizer budget, not that finite-bank optimum.
"""
from __future__ import annotations

import math

import torch

VERSION = 'otto-conditional-label-v1'
BANK_SIZE, HORIZON, ACTIONS = 32, 8, 4
ARMS = ('sampled', 'mean32')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor(value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.layout == torch.strided and value.dtype == dtype
            and tuple(value.shape) == tuple(shape), 'CPU tensor with exact dtype/shape: ' + name)
    require(bool(torch.isfinite(value).all()), 'finite ' + name)


def centered(value, name):
    # All target centering is done in float64. This only allows its roundoff.
    tolerance = 8 * torch.finfo(torch.float64).eps * value.abs().amax(-1).clamp_min(1.)
    require(bool((value.sum(-1).abs() <= tolerance).all()), 'centered ' + name)


def bank_valid(bank):
    require(isinstance(bank, torch.Tensor) and bank.ndim == 3 and bank.shape[0] > 0,
            'nonempty centered bank [B,32,4]')
    tensor(bank, torch.float64, (bank.shape[0], BANK_SIZE, ACTIONS), 'centered bank')
    require(not bank.requires_grad, 'fixed target bank without gradient history')
    centered(bank, 'bank')


def centered_bank(raw_costs, alive):
    """Return owned float64[B,32,4] contrasts, retaining found-zero draws."""
    require(isinstance(raw_costs, torch.Tensor) and raw_costs.ndim == 3
            and raw_costs.shape[0] > 0, 'nonempty raw bank [B,32,4]')
    n = raw_costs.shape[0]
    tensor(raw_costs, torch.float32, (n, BANK_SIZE, ACTIONS), 'raw costs')
    tensor(alive, torch.bool, (n, BANK_SIZE), 'alive')
    require(not raw_costs.requires_grad and bool((raw_costs >= 0).all()), 'fixed nonnegative teacher costs')
    require(bool((raw_costs[~alive] == 0).all()), 'found histories have literal zero costs')
    normalized = raw_costs.to(torch.float64) / 64.
    result = normalized - normalized.mean(-1, keepdim=True)
    bank_valid(result)
    return result


def select_target(bank, arm, *, indices=None):
    """Select one draw per prefix or average all draws; scheduling is external.

    The returned tensor owns its storage. The sampled arm requires explicit
    indices; the mean arm rejects them to catch accidental schedule confusion.
    """
    bank_valid(bank)
    require(type(arm) is str and arm in ARMS, 'declared target arm')
    if arm == 'mean32':
        require(indices is None, 'mean32 has no draw selection')
        return bank.mean(1)
    tensor(indices, torch.int64, (len(bank),), 'sampled draw indices')
    require(bool(((indices >= 0) & (indices < BANK_SIZE)).all()), 'draw indices in 0..31')
    return bank[torch.arange(len(bank)), indices].clone()


def train_scale(bank, *, variance_floor):
    """Return the common TRAIN sampled-bank second moment and rounded scale."""
    bank_valid(bank)
    require(type(variance_floor) in (int, float) and math.isfinite(variance_floor)
            and variance_floor > 0, 'explicit finite positive variance floor')
    moment = float(bank.square().mean())
    scale = float(torch.tensor(math.sqrt(max(moment, variance_floor)), dtype=torch.float32))
    require(math.isfinite(moment) and math.isfinite(scale) and scale > 0,
            'finite second moment and positive represented float32 scale')
    return {'cost_scale': scale, 'second_moment_before_floor': moment,
            'variance_floor': float(variance_floor)}


def horizon8_loss(cost_contrasts, target, *, cost_scale):
    """Scalar float64 squared error, equal originating prefixes and actions.

    Only output index7 receives gradients. Earlier latent transitions remain
    in the model's graph. There is no realized-survival loss mask and no
    outcome, feature, counterfactual or normal-observation auxiliary objective.
    """
    require(isinstance(cost_contrasts, torch.Tensor) and cost_contrasts.ndim == 3
            and cost_contrasts.shape[0] > 0, 'nonempty forecast [B,8,4]')
    n = len(cost_contrasts)
    tensor(cost_contrasts, torch.float32, (n, HORIZON, ACTIONS), 'cost contrasts')
    tensor(target, torch.float64, (n, ACTIONS), 'target')
    require(not target.requires_grad, 'fixed target without gradient history')
    centered(target, 'target')
    require(type(cost_scale) in (int, float) and math.isfinite(cost_scale) and cost_scale > 0,
            'explicit finite positive cost scale')
    loss = ((cost_contrasts[:, HORIZON - 1].to(torch.float64) - target) / cost_scale).square().mean()
    require(bool(torch.isfinite(loss)), 'finite horizon-eight loss')
    return loss
