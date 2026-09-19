"""Small cached-error predictors for a training-only residual-order diagnostic.

No backbone, optimizer, data loader, random permutation, or external recurrent
cache is retained. Callers supply frozen one-step predictions made before each
successor assimilation. Shape checks cannot authenticate that provenance.
Rotations map body to world and are assumed proper, as in rigid_motion; no
projection is performed and the principal logarithm inherits its pi cut.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from openjev.research.rigid_motion import so3_exp, so3_log

TOKEN_DIM = 55
HORIZON = 25


def _check(value: Tensor, shape: tuple[int, ...], reference: Tensor, name: str):
    if not (isinstance(value, Tensor) and tuple(value.shape) == shape
            and value.dtype in (torch.float32, torch.float64)
            and value.dtype == reference.dtype and value.device == reference.device
            and bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must have finite shape {shape} and matching float32/64 dtype/device")


def _sequence(value: Tensor, name: str) -> tuple[int, int]:
    if not isinstance(value, Tensor) or value.ndim != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} requires positive batch and sequence dimensions")
    batch, length = value.shape[:2]
    _check(value, (batch, length, 3), value, name)
    return batch, length


def _scales(value: Tensor, count: int, reference: Tensor, name: str):
    _check(value, (count,), reference, name)
    if not bool((value > 0).all()):
        raise ValueError(f"{name} must be strictly positive")


def _rotate(rotation: Tensor, vector: Tensor) -> Tensor:
    return (rotation @ vector.unsqueeze(-1)).squeeze(-1)


def residual_targets(base_p: Tensor, base_R: Tensor, target_p: Tensor, target_R: Tensor,
                     root_R: Tensor, error_scales: Tensor) -> Tensor:
    """Return [B,H,6] normalized signed errors in the fixed root body frame.

    Translation is Rroot.T*(target_p-base_p)/scale_p. Rotation is
    Rroot.T*Log(target_R*base_R.T)/scale_R, a LEFT correction expressed at root.
    This target-only helper is never called by either predictor's forward.
    H is positive and generic; the diagnostic heads below emit exactly25 leads.
    """
    batch, horizon = _sequence(base_p, "base positions")
    _check(base_R, (batch, horizon, 3, 3), base_p, "base rotations")
    _check(target_p, (batch, horizon, 3), base_p, "target positions")
    _check(target_R, (batch, horizon, 3, 3), base_p, "target rotations")
    _check(root_R, (batch, 3, 3), base_p, "root rotations")
    _scales(error_scales, 2, base_p, "error scales")
    inverse = root_R.transpose(-1, -2)[:, None]
    position = _rotate(inverse, target_p - base_p) / error_scales[0]
    rotation = _rotate(inverse, so3_log(target_R @ base_R.transpose(-1, -2))) / error_scales[1]
    result = torch.cat((position, rotation), -1)
    _check(result, (batch, horizon, 6), base_p, "residual targets")
    return result


def apply_correction(base_p: Tensor, base_R: Tensor, root_R: Tensor,
                     predicted_residual: Tensor, error_scales: Tensor) -> tuple[Tensor, Tensor]:
    """Apply normalized root-frame corrections independently at every lead.

    p'=base_p+Rroot*(scale_p*c_p), R'=Exp(Rroot*(scale_R*c_R))*base_R.
    The corrected poses do not feed back into the frozen predictor. This is a
    direct forecast-error head, not an online dynamics or state-update adapter.
    No detach is used: a downstream physical pose loss can train the head.
    """
    batch, horizon = _sequence(base_p, "base positions")
    _check(base_R, (batch, horizon, 3, 3), base_p, "base rotations")
    _check(root_R, (batch, 3, 3), base_p, "root rotations")
    _check(predicted_residual, (batch, horizon, 6), base_p, "predicted residuals")
    _scales(error_scales, 2, base_p, "error scales")
    frame = root_R[:, None]
    position = base_p + _rotate(frame, predicted_residual[..., :3] * error_scales[0])
    rotation = so3_exp(_rotate(frame, predicted_residual[..., 3:] * error_scales[1])) @ base_R
    _check(position, (batch, horizon, 3), base_p, "corrected positions")
    _check(rotation, (batch, horizon, 3, 3), base_p, "corrected rotations")
    return position, rotation


def innovation_tokens(p: Tensor, R: Tensor, actions: Tensor, predicted_p: Tensor,
                      predicted_R: Tensor, motion_scales: Tensor, error_scales: Tensor) -> Tensor:
    """Completed transitions only: [B,T+1] actual poses -> [B,T,55], 1<=T<=31.

    Token t uses actual pose t+1 and the action t that produced it. The first49
    coordinates match pose_coordination: current-body world vertical (3),
    measured backward displacement/angular increment in that body frame (6),
    and40 caller-normalized recorded torques. The final6 are the signed error
    of the caller's pre-assimilation one-step prediction, expressed in the
    FINAL context/root frame. Tanh follows scaling and concatenation once.

    Every input is available at the final root. Root-frame errors are a
    retrospective encoding, not a claim that the same token existed earlier.
    Input error scales are separate from the output/application scales.
    """
    batch, length = _sequence(p, "context positions")
    if not 2 <= length <= 32:
        raise ValueError("context must contain2..32 actual poses")
    transitions = length - 1
    _check(R, (batch, length, 3, 3), p, "context rotations")
    _check(actions, (batch, transitions, 40), p, "completed actions")
    _check(predicted_p, (batch, transitions, 3), p, "one-step positions")
    _check(predicted_R, (batch, transitions, 3, 3), p, "one-step rotations")
    _scales(motion_scales, 4, p, "motion scales")
    _scales(error_scales, 2, p, "input error scales")
    current = R[:, 1:]
    inverse = current.transpose(-1, -2)
    dp = p[:, 1:] - p[:, :-1]
    w = so3_log(current @ R[:, :-1].transpose(-1, -2))
    errors = residual_targets(predicted_p, predicted_R, p[:, 1:], current, R[:, -1], error_scales)
    features = torch.cat((current[:, :, 2, :],
                          _rotate(inverse, dp) / motion_scales[0],
                          _rotate(inverse, w) / motion_scales[1], actions, errors), -1)
    _check(features, (batch, transitions, TOKEN_DIM), p, "unbounded innovation features")
    return features.tanh()


def _constructor(dtype):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("predictor dtype must be float32 or float64")


def _tokens(tokens: Tensor, reference: Tensor):
    if not (isinstance(tokens, Tensor) and tokens.ndim == 3 and tokens.shape[0] > 0
            and 1 <= tokens.shape[1] <= 31):
        raise ValueError("tokens require positive batch and1..31 completed transitions")
    _check(tokens, (tokens.shape[0], tokens.shape[1], TOKEN_DIM), reference, "cached tokens")


class RecurrentInnovation(nn.Module):
    """55->GRU8->150, 2,910 parameters; zero head, no retained recurrent state.

    The shuffled-history and zero-error controls use this EXACT class/weights.
    Caller permutes interior whole tokens (first/last fixed), or zeros only
    coordinates49:55, before forward. There is no internal random operation.
    Output [B,25,6] uses caller-chosen normalized root-frame residual units.
    """

    def __init__(self, *, dtype=torch.float32, device=None):
        super().__init__()
        _constructor(dtype)
        self.gru = nn.GRU(TOKEN_DIM, 8, batch_first=True, dtype=dtype, device=device)
        self.head = nn.Linear(8, HORIZON * 6, dtype=dtype, device=device)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.head.weight)
        _, hidden = self.gru(tokens)
        result = self.head(hidden[0]).reshape(tokens.shape[0], HORIZON, 6)
        _check(result, (tokens.shape[0], HORIZON, 6), tokens, "predicted residuals")
        return result


class SummaryInnovation(nn.Module):
    """165->9(tanh)->150, 2,994 parameters, zero head (2.9% more than GRU).

    Concatenates last token, mean over all tokens, and last-minus-first.
    Mathematically invariant to interior permutations with endpoints fixed;
    floating-point summation may change by rounding. No exact parameter-match
    or compute-match claim is made. Output units match RecurrentInnovation.
    """

    def __init__(self, *, dtype=torch.float32, device=None):
        super().__init__()
        _constructor(dtype)
        self.network = nn.Sequential(nn.Linear(3 * TOKEN_DIM, 9, dtype=dtype, device=device), nn.Tanh(),
                                     nn.Linear(9, HORIZON * 6, dtype=dtype, device=device))
        nn.init.zeros_(self.network[2].weight)
        nn.init.zeros_(self.network[2].bias)

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.network[0].weight)
        summary = torch.cat((tokens[:, -1], tokens.mean(1), tokens[:, -1] - tokens[:, 0]), -1)
        result = self.network(summary).reshape(tokens.shape[0], HORIZON, 6)
        _check(result, (tokens.shape[0], HORIZON, 6), tokens, "predicted residuals")
        return result
