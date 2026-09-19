"""Small context-only gates over independently computed pose forecasts.

No expert model or trajectory state is retained. A gate supplies two fractions
per context, fixed over the forecast horizon. Blended rotations never feed back
into either expert. This is conventional pose ensembling, not a joint dynamics
model or a claim of physical consistency. Matrix inputs are assumed proper
rotations, following rigid_motion; no projection or orthogonality repair occurs.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from openjev.research.rigid_motion import so3_exp, so3_log


def _check(value: Tensor, shape: tuple[int, ...], reference: Tensor, name: str):
    if not (isinstance(value, Tensor) and tuple(value.shape) == shape
            and value.dtype in (torch.float32, torch.float64)
            and value.dtype == reference.dtype and value.device == reference.device
            and bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must have finite shape {shape} and matching float32/64 dtype/device")


def context_tokens(p: Tensor, rotation: Tensor, past_actions: Tensor, scales: Tensor) -> Tensor:
    """Return [B,31,49] tokens from exactly32 observed poses and31 actions.

    Token t-1, t=1..31, uses R[t], measured backward displacement and spatial
    angular increment from t-1 to t, and the action[t-1] that produced it.
    Inputs are current-body world vertical, current-body motion divided by
    scales[0:2], and40 already normalized torques. Tanh is applied to all49.
    No future observation or action argument is accepted.
    """
    if not isinstance(p, Tensor) or p.ndim != 3 or p.shape[0] < 1:
        raise ValueError("context positions require positive batch size")
    batch = p.shape[0]
    _check(p, (batch, 32, 3), p, "context positions")
    _check(rotation, (batch, 32, 3, 3), p, "context rotations")
    _check(past_actions, (batch, 31, 40), p, "completed actions")
    _check(scales, (4,), p, "training scales")
    if not bool((scales > 0).all()):
        raise ValueError("training scales must be positive")
    current = rotation[:, 1:]
    inverse = current.transpose(-1, -2)
    dp = p[:, 1:] - p[:, :-1]
    w = so3_log(current @ rotation[:, :-1].transpose(-1, -2))
    inputs = torch.cat((current[:, :, 2, :],
                        (inverse @ dp.unsqueeze(-1)).squeeze(-1) / scales[0],
                        (inverse @ w.unsqueeze(-1)).squeeze(-1) / scales[1],
                        past_actions), -1)
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("nonfinite context token features")
    return inputs.tanh()


def _check_dtype(dtype):
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("gate dtype must be float32 or float64")


def _tokens(tokens: Tensor, parameter: Tensor):
    if not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] < 1:
        raise ValueError("tokens require positive batch size")
    _check(tokens, (tokens.shape[0], 31, 49), parameter, "context tokens")


class ConstantGate(nn.Module):
    """Two global fast-expert logits, initialized to the exact half blend."""

    def __init__(self, *, dtype=torch.float32, device=None):
        super().__init__()
        _check_dtype(dtype)
        self.logits = nn.Parameter(torch.zeros(2, dtype=dtype, device=device))

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.logits)
        return self.logits.sigmoid().unsqueeze(0).expand(tokens.shape[0], -1).clone()


class SummaryGate(nn.Module):
    """147->22->2 MLP on last token, mean and last-minus-first, in that order."""

    def __init__(self, *, dtype=torch.float32, device=None):
        super().__init__()
        _check_dtype(dtype)
        self.network = nn.Sequential(nn.Linear(147, 22, dtype=dtype, device=device), nn.Tanh(),
                                     nn.Linear(22, 2, dtype=dtype, device=device))
        nn.init.zeros_(self.network[2].weight)
        nn.init.zeros_(self.network[2].bias)

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.network[0].weight)
        summary = torch.cat((tokens[:, -1], tokens.mean(1), tokens[:, -1] - tokens[:, 0]), -1)
        return self.network(summary).sigmoid()


class RecurrentGate(nn.Module):
    """49-input,16-hidden GRU reset from zero for every complete context."""

    def __init__(self, *, dtype=torch.float32, device=None):
        super().__init__()
        _check_dtype(dtype)
        self.gru = nn.GRU(49, 16, batch_first=True, dtype=dtype, device=device)
        self.head = nn.Linear(16, 2, dtype=dtype, device=device)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        _tokens(tokens, self.head.weight)
        _, hidden = self.gru(tokens)
        return self.head(hidden[0]).sigmoid()


def blend(fast_p: Tensor, fast_rotation: Tensor, slow_p: Tensor, slow_rotation: Tensor,
          alpha: Tensor) -> tuple[Tensor, Tensor]:
    """Two per-context fast fractions [B,2], held fixed across positive H.

    p=alpha_p*p_fast+(1-alpha_p)*p_slow;
    R=Exp(alpha_R*Log(R_fast R_slow^T))*R_slow. The SO(3) principal-log cut
    inherits rigid_motion's convention; smooth gradients are not promised at
    the pi cut. Exact zero/one fractions select the original endpoints with
    torch.where. Interior fractions remain differentiable, without detaching.
    """
    if (not isinstance(fast_p, Tensor) or fast_p.ndim != 3
            or fast_p.shape[0] < 1 or fast_p.shape[1] < 1):
        raise ValueError("expert positions require positive batch and horizon")
    batch, horizon = fast_p.shape[:2]
    _check(fast_p, (batch, horizon, 3), fast_p, "fast positions")
    _check(slow_p, (batch, horizon, 3), fast_p, "slow positions")
    _check(fast_rotation, (batch, horizon, 3, 3), fast_p, "fast rotations")
    _check(slow_rotation, (batch, horizon, 3, 3), fast_p, "slow rotations")
    _check(alpha, (batch, 2), fast_p, "gate fractions")
    if not bool(((alpha >= 0) & (alpha <= 1)).all()):
        raise ValueError("gate fractions must be in [0,1]")
    ap = alpha[:, 0, None, None]
    position = ap * fast_p + (1 - ap) * slow_p
    position = torch.where(ap == 0, slow_p, torch.where(ap == 1, fast_p, position))
    ar = alpha[:, 1, None, None]
    relative = so3_log(fast_rotation @ slow_rotation.transpose(-1, -2))
    rotation = so3_exp(ar * relative) @ slow_rotation
    ar = ar.unsqueeze(-1)
    rotation = torch.where(ar == 0, slow_rotation, torch.where(ar == 1, fast_rotation, rotation))
    if not bool(torch.isfinite(position).all() and torch.isfinite(rotation).all()):
        raise ValueError("nonfinite blended pose")
    return position, rotation
