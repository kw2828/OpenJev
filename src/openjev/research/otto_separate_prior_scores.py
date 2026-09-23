"""Separate prequery calibration from the nonquery decision readout.

This engineering component changes only the readout used at later queries.
It inherits the held public-input validation, recurrent updates, correction,
query override, capture mapping and detached-carry contract unchanged. The
caller must bind both dependencies to the source hashes below.

In that exact base implementation, rank-two readouts are later-query priors;
rank-three readouts are nonquery sequences. Dispatch is restricted to an active
inherited forward, after its input validation. This is a pinned internal call
contract, not a general inference-mode interface based on arbitrary shapes.

The new prior head changes the innovation supplied to correction. Both heads
still share the GRU, and future nonquery losses can reach the prior head through
correction. Separate readouts do not isolate backbone gradients or guarantee
preservation of first-window decisions after training. No efficacy is claimed.
"""
from __future__ import annotations

import torch

from openjev.research import otto_prequery_scores as shared

VERSION = "otto-separate-prior-scores-v1"
BASE_SOURCE_SHA256 = "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
PREQUERY_SOURCE_SHA256 = "6dfff08284004a134abf47b4ab4dc425f5d9c96f63c58475007295a666b24f4e"
KINDS = shared.KINDS
PARAMETERS = {"innovation": 6098, "innovation_gru": 6112}
Forecast, Carry, detach_carry = shared.Forecast, shared.Carry, shared.detach_carry
base = shared.base


def parameter_count(kind):
    base.require(kind in KINDS, "declared separate-prior family")
    return PARAMETERS[kind]


class SeparatePriorHead(shared.PrequeryHead):
    """Same chronological forward API, with one additional zero-initialized head.

    Existing named parameters retain the shared model's seeded initialization.
    ``output`` remains the decision readout; ``prior_output`` forecasts later
    queries. Zero initialization of both gives the exact original hold policy.
    Copying output into prior_output recovers the shared model's forward values
    for identical existing parameters, including its correction and carry.
    """

    def __init__(self, kind, seed):
        super().__init__(kind, seed)
        self.prior_output = torch.nn.Linear(self.width, base.SCORE_DIM,
                                            device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.prior_output.weight.zero_()
            self.prior_output.bias.zero_()

    def _prediction(self, hidden, raw_anchor):
        base.require(self._captured is not None, "readouts require an active inherited forward")
        if hidden.ndim == 3:
            return super()._prediction(hidden, raw_anchor)
        base.require(hidden.ndim == 2 and hidden.shape[1] == self.width
                     and tuple(raw_anchor.shape) == (hidden.shape[0], base.SCORE_DIM),
                     "held later-query readout geometry")
        value = (raw_anchor / base.SCALE + base.centered(self.prior_output(hidden))) * base.SCALE
        self._captured.append(value)
        return value


def make_head(kind, seed):
    """Preserve caller RNG and all existing parameters for the same paired seed."""
    base.require(kind in KINDS and type(seed) is int and 0 <= seed < 2**32,
                 "declared family and uint32 seed")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = SeparatePriorHead(kind, seed)
    base.require(sum(p.numel() for p in model.parameters()) == parameter_count(kind), "exact parameter count")
    return model
