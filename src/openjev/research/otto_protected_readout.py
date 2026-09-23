"""Protect the held ordinary GRU forecast while exposing an action-only readout.

Engineering adapter only; no training rule, fitted advantage or novelty claim.
Both immutable source files are SHA256-checked before construction. The model
inherits the ordinary ``innovation_gru`` PrequeryHead: its predictions, later
query priors, query assimilation and Carry are unchanged. Exactly one inherited
forward runs. Its existing rank-three ``_prediction`` calls reveal nonquery
hidden sequences; rank-two prequery calls retain their original behavior.

For each active nonquery hidden vector h (width 28), define raw-score residual
z = W h + b and delta = z - mean(z, axis=-1). ``action_prediction = prediction +
delta`` only on those positions. There is no extra factor of64: W and b produce
raw-score units. The 4x28 weight and 4-element bias add116 parameters, for6112
total. Both tensors start at exact zero. Genuine query outputs and padding are
copied without addition, retaining their original bytes, including signed zero.
Centering is ordinary float32 subtraction; it is not a probabilistic softmax.

The two modes have identical residual capacity. Bitwise base preservation is
relative to the original with matching parameter requires_grad flags AND
gradient context. Equal weights and shared no_grad alone do not imply bitwise
cross-mode equality: differences from parameter flags also occur in the held
original implementation. The adapter preserves each corresponding original's
values and does not normalize these differences. ``frozen`` locks all5996
backbone/readout parameters and runs that inherited forward under no_grad;
only the116 residual parameters receive action-loss gradients, including when
caller inputs require gradients. ``joint`` leaves all6112 parameters trainable.
Residual values never enter the inherited ``_prediction`` result,
prior, innovation, recurrent input or Carry. They affect no later base forecast.

Capture is transient and exclusive to one instance. A nonblocking lock rejects
concurrent/reentrant calls before touching an active capture; ``finally`` always
clears this adapter's references and releases its lock. The inherited prequery
capture has its own unchanged guard/finally. No hidden state or hook is retained
between forwards; save/load parameters through state_dict, not pickling the
instance's synchronization lock. TBPTT carries still require explicit detach.
"""
from __future__ import annotations

import hashlib
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from openjev.research import otto_prequery_scores as base

VERSION = "otto-protected-readout-v1"
PREQUERY_SOURCE_SHA256 = "6dfff08284004a134abf47b4ab4dc425f5d9c96f63c58475007295a666b24f4e"
CROSS_QUERY_SOURCE_SHA256 = "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
KIND = "innovation_gru"
MODES = ("frozen", "joint")
WIDTH, SCORE_DIM, PERIOD = 28, 4, 4
BACKBONE_PARAMETERS, RESIDUAL_PARAMETERS = 5996, 116
PARAMETERS = BACKBONE_PARAMETERS + RESIDUAL_PARAMETERS
Carry, detach_carry = base.Carry, base.detach_carry
require = base.base.require


def verify_sources():
    """Admit only the unchanged qualified parent implementations."""
    for module, expected in ((base, PREQUERY_SOURCE_SHA256), (base.base, CROSS_QUERY_SOURCE_SHA256)):
        path = Path(module.__file__)
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                "immutable protected-readout base source pin")
    require(base.BASE_SOURCE_SHA256 == CROSS_QUERY_SOURCE_SHA256,
            "prequery adapter declares the same immutable cross-query source")


@dataclass(frozen=True)
class Forecast:
    """Original base containers plus a separate differentiable action forecast.

    prediction/prior/prior_mask/carry are the exact inherited output objects.
    action_prediction is a separately owned CPU float32[B,T,4] tensor. No tensor
    immutability is implied by the frozen dataclass.
    """

    prediction: torch.Tensor
    prior: torch.Tensor
    prior_mask: torch.Tensor
    carry: Carry
    action_prediction: torch.Tensor


def parameter_count(mode, *, trainable_only=False):
    require(mode in MODES and type(trainable_only) is bool, "declared protected-readout mode")
    return RESIDUAL_PARAMETERS if trainable_only and mode == "frozen" else PARAMETERS


class ProtectedReadout(base.PrequeryHead):
    def __init__(self, mode, seed):
        require(mode in MODES and type(seed) is int and 0 <= seed < 2**32,
                "declared mode and uint32 seed")
        verify_sources()
        super().__init__(KIND, seed)
        self._mode = mode
        self.action_residual = torch.nn.Linear(WIDTH, SCORE_DIM, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.action_residual.weight.zero_()
            self.action_residual.bias.zero_()
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(mode == "joint" or name.startswith("action_residual."))
        self._action_hidden: list[torch.Tensor] | None = None
        self._action_lock = threading.Lock()

    @property
    def mode(self):
        return self._mode

    def _prediction(self, hidden, raw_anchor):
        # The inherited return value is never replaced by the residual branch.
        value = super()._prediction(hidden, raw_anchor)
        if self._action_hidden is not None and hidden.ndim == 3:
            self._action_hidden.append(hidden)
        return value

    def forward(self, features, query_scores, lengths, query_mask, *, carry=None, episode_ends):
        require(self._action_lock.acquire(blocking=False), "protected readout cannot be reentered or used concurrently")
        try:
            require(self._action_hidden is None, "no stale action hidden capture")
            require(self.mode in MODES and all(
                parameter.requires_grad == (self.mode == "joint" or name.startswith("action_residual."))
                for name, parameter in self.named_parameters()
            ), "protected mode parameter locks unchanged")
            self._action_hidden = []
            with torch.no_grad() if self.mode == "frozen" else nullcontext():
                forecast = super().forward(features, query_scores, lengths, query_mask,
                                           carry=carry, episode_ends=episode_ends)
            # The held implementation emits nonquery readouts in offset/count
            # order, with ascending lane indices, after all input validation.
            batch, span, _ = forecast.prediction.shape
            rows = [forecast.prediction[:, step].clone() for step in range(span)]
            used = 0
            for offset in range(0, span, PERIOD):
                for count in range(1, min(PERIOD, span - offset)):
                    index = torch.nonzero((lengths - offset - 1).clamp(0, PERIOD - 1) == count,
                                          as_tuple=False).flatten()
                    if not index.numel():
                        continue
                    require(used < len(self._action_hidden), "all active nonquery hidden sequences captured")
                    hidden = self._action_hidden[used]
                    require(tuple(hidden.shape) == (index.numel(), count, WIDTH),
                            "exact active nonquery hidden geometry")
                    raw = self.action_residual(hidden)
                    residual = raw - raw.mean(dim=-1, keepdim=True)
                    require(bool(torch.isfinite(residual).all()), "finite centered action residual")
                    for age in range(count):
                        step = offset + age + 1
                        value = forecast.prediction[index, step] + residual[:, age]
                        rows[step] = rows[step].index_copy(0, index, value)
                    used += 1
            require(used == len(self._action_hidden), "no unassigned nonquery hidden sequence")
            action = torch.stack(rows, dim=1)
            require(tuple(action.shape) == (batch, span, SCORE_DIM) and bool(torch.isfinite(action).all()),
                    "finite protected action forecast")
            return Forecast(forecast.prediction, forecast.prior, forecast.prior_mask, forecast.carry, action)
        finally:
            self._action_hidden = None
            self._action_lock.release()


def make_head(mode, seed):
    """Deterministic held initialization with zero residual and unchanged caller RNG."""
    require(mode in MODES and type(seed) is int and 0 <= seed < 2**32,
            "declared mode and uint32 seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ProtectedReadout(mode, seed)
    require(sum(p.numel() for p in model.parameters()) == parameter_count(mode), "exact total parameter count")
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)
            == parameter_count(mode, trainable_only=True), "exact trainable parameter count")
    return model
