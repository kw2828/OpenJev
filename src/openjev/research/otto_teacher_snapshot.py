"""Public-only analytic teacher initialized at an already-assimilated anchor.

This is initialization engineering, not a rollout sampler or study admission.
The caller establishes the posterior's public provenance. No simulator, source,
random stream, seed, model or hidden trajectory state is accepted or retained.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from openjev.research.otto_reference_control import SpaceAwareActor, _SpaceAwarePolicy
from openjev.research.otto_released_policy import EPSILON, PUBLIC_FIELDS, N, PublicBeliefView, _packet

VERSION = "otto-public-teacher-snapshot-v1"
NORMALIZATION_ATOL = EPSILON


class TeacherSnapshot(SpaceAwareActor):
    """Copy a normalized nonterminal public belief without applying its hit again.

    ``public`` retains its arbitrary valid position and absolute public step.
    ``belief`` must be finite nonnegative float64[53,53], with exact zero mass
    at that position and total mass within the existing categorical tolerance
    1e-10 of one. The tolerance permits roundoff, not a subfloor/empty anchor.
    Accepted entries are copied exactly; no normalization or support repair.
    The kernel uses the unchanged owned, immutable PublicBeliefView checks.

    All subsequent choices and observations use the inherited in-bounds analytic
    policy and public filter. A prescribed first update is accepted while no
    choice is pending. The caller counts that action as continuation move one
    and owns the horizon; the final unsuccessful packet is still assimilated.

    Inherited ``reset(initial_public)`` starts a NEW center-prior episode with
    step0 and a positive initial hit. It does not restore this anchor. Construct
    a fresh TeacherSnapshot to restart an arbitrary continuation. No additional
    copy of the anchor, previous hits, or trajectory is retained.
    """

    __slots__ = ()

    def __init__(self, public, belief, kernel):
        if isinstance(public, tuple) and hasattr(public, "_fields") and hasattr(public, "_asdict"):
            public = public._asdict()
        if not isinstance(public, Mapping) or set(public) != PUBLIC_FIELDS:
            raise ValueError("exact public anchor packet fields required")
        packet = _packet(public, public["step"])
        if packet["done"]:
            raise ValueError("teacher anchor must be nonterminal")
        if not isinstance(belief, np.ndarray) or belief.dtype != np.float64 or belief.shape != (N, N):
            raise ValueError("anchor belief must be float64[53,53]")
        owned = belief.copy(order="C")
        if not np.isfinite(owned).all() or (owned < 0).any():
            raise ValueError("anchor belief must be finite and nonnegative")
        mass = float(np.sum(owned, dtype=np.float64))
        if not np.isfinite(mass) or abs(mass-1.0) > NORMALIZATION_ATOL:
            raise ValueError("anchor belief must already be normalized; no repair")
        if owned[packet["position"]] != 0:
            raise ValueError("nonterminal anchor must have exactly zero current-cell mass")
        self._view = PublicBeliefView(kernel)
        self._view._probability = owned
        self._view._agent = packet["position"]
        self._public, self._pending_action = packet, None
        self._policy = _SpaceAwarePolicy(env=self._view, model=None, sym_avg=False, allow_stay=False)
