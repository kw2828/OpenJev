"""Matched public-history reconstruction for classical RockSample controls.

All completed primitive transitions count toward observation age. Every sampling
action is retained, including those older than the check window. Replay remains
chronological: applying the final depletion ledger before earlier checks would
leak future actions into those checks' likelihoods. No rewards, true maps, latent
qualities, episode seeds or simulator objects belong in this history.

History covers one episode. The caller starts a new history at each public reset.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Protocol, Self


class PublicBelief(Protocol):
    """The public subset of the particle/quality belief interface used here."""

    size: int
    rocks: int

    def copy(self) -> Self: ...
    def reset(self) -> None: ...
    def sample(self, position: tuple[int, int]) -> None: ...
    def condition_check(self, rock: int, position: tuple[int, int], positive: bool) -> Self: ...


@dataclass(frozen=True, slots=True)
class PublicTransition:
    """One completed action at its previous public position; CHECK reading is +/-1.

    CHECK and SAMPLE do not move, so this is also their observation position.
    Movements have no reading. Coordinates are copied into an immutable tuple.
    """

    position: tuple[int, int]
    action: int
    reading: int | None = None

    def __post_init__(self) -> None:
        try:
            coordinates = tuple(self.position)
        except TypeError as exc:
            raise ValueError("position must contain two nonnegative integer coordinates") from exc
        if (len(coordinates) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0
                                       for v in coordinates)):
            raise ValueError("position must contain two nonnegative integer coordinates")
        if isinstance(self.action, bool) or not isinstance(self.action, Integral) or self.action < 0:
            raise ValueError("action must be a nonnegative integer")
        if self.action < 5:
            if self.reading is not None:
                raise ValueError("movement and SAMPLE actions have no check reading")
        elif isinstance(self.reading, bool) or not isinstance(self.reading, Integral) or self.reading not in (-1, 1):
            raise ValueError("CHECK requires a signed -1 or +1 reading")
        object.__setattr__(self, "position", tuple(int(v) for v in coordinates))
        object.__setattr__(self, "action", int(self.action))
        if self.reading is not None:
            object.__setattr__(self, "reading", int(self.reading))


def _validate(belief: PublicBelief, event: PublicTransition) -> None:
    if not isinstance(event, PublicTransition):
        raise TypeError("history entries must be PublicTransition objects")
    row, column = event.position
    if row >= belief.size or column >= belief.size - 1:
        raise ValueError("an action must start at a nonterminal public board position")
    if event.action >= belief.rocks + 5:
        raise ValueError("action is outside the public discrete action space")


def apply_transition(belief: PublicBelief, event: PublicTransition) -> PublicBelief:
    """Apply public evidence; SAMPLE mutates belief, CHECK returns a conditioned copy."""
    _validate(belief, event)
    if event.action == 4:
        belief.sample(event.position)
    elif event.action >= 5:
        return belief.condition_check(event.action - 5, event.position, event.reading == 1)
    return belief


def reconstruct(prior: PublicBelief, history,
                mode: Literal["recent128", "latest", "full"] = "recent128") -> PublicBelief:
    """Reset a copy of prior, retaining all depletion and the selected check evidence.

    ``recent128`` retains checks with primitive index >= len(history)-128.
    ``latest`` retains the most recent check separately for every rock. ``full``
    retains every check. Samples are never dropped or moved out of chronology.
    Ignored checks are still structurally validated and count toward the window.
    """
    if mode not in ("recent128", "latest", "full"):
        raise ValueError("mode must be recent128, latest or full")
    events = tuple(history)
    for event in events:
        _validate(prior, event)
    latest = {event.action: index for index, event in enumerate(events) if event.action >= 5}
    cutoff = max(0, len(events) - 128)
    belief = prior.copy()
    belief.reset()
    for index, event in enumerate(events):
        keep_check = (event.action >= 5 and (mode == "full" or mode == "recent128" and index >= cutoff
                                            or mode == "latest" and latest[event.action] == index))
        if event.action == 4 or keep_check:
            belief = apply_transition(belief, event)
    return belief
