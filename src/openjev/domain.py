import math
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np


class Directive(StrEnum):
    HUNT = "hunt"
    CONSERVE = "conserve"
    PACIFIST = "pacifist"


STEERING = ("left", "hold", "right")
SCENARIOS = ("defend_the_center", "defend_the_line", "basic")
FEATURES = (
    "target_visible",
    "aim_error_x10",
    "abs_aim_error_x10",
    "target_half_width_x10",
    "distance_1000",
    "health_100",
    "ammo_30",
    "hunt",
    "conserve",
    "pacifist",
    "basic",
)


@dataclass(frozen=True)
class Observation:
    visible: bool = False
    aim_error: float = 0.0  # -1 left edge, +1 right edge of the screen
    half_width: float = 0.0  # target half-width in the same coordinates
    distance: float = 0.0
    health: float = 100.0
    ammo: float = 0.0
    enemies: int = 0
    target: str = "none"
    scenario: str = "defend_the_center"
    directive: str = "hunt"

    def features(self) -> np.ndarray:
        return np.array(
            [
                float(self.visible),
                self.aim_error * 10,
                abs(self.aim_error) * 10,
                self.half_width * 10,
                min(self.distance / 1000, 2),
                self.health / 100,
                min(self.ammo / 30, 2),
                *[float(self.directive == d) for d in Directive],
                float(self.scenario == "basic"),
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    steer: str
    fire: bool
    probabilities: dict[str, float]
    fire_probability: float
    source: str
    latency_ms: float = 0.0
    provider_confidence: float | None = None

    def __post_init__(self):
        if self.steer not in STEERING or type(self.fire) is not bool:
            raise ValueError("Invalid typed action")
        if set(self.probabilities) != set(STEERING):
            raise ValueError("Incomplete steering distribution")
        values = list(self.probabilities.values())
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError("Invalid probability")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        if not math.isfinite(self.fire_probability) or not 0 <= self.fire_probability <= 1:
            raise ValueError("Invalid fire probability")
        if self.provider_confidence is not None and (
            not math.isfinite(self.provider_confidence) or not 0 <= self.provider_confidence <= 1
        ):
            raise ValueError("Invalid provider confidence")

    def buttons(self, observation: Observation) -> list[int]:
        # These hard constraints apply to every backend, including remote and manual.
        fire_allowed = observation.directive != "pacifist" and observation.ammo > 0
        return [int(self.steer == "left"), int(self.steer == "right"), int(self.fire and fire_allowed)]

    def to_dict(self) -> dict:
        return asdict(self)


def hard_decision(steer: str, fire: bool, source: str) -> Decision:
    return Decision(steer, fire, {s: float(s == steer) for s in STEERING}, float(fire), source)


def teacher_action(obs: Observation) -> tuple[str, bool]:
    """Transparent hand-written teacher, also used as an evaluation baseline."""
    if not obs.visible:
        return ("right", False)
    tolerance = 0.07
    steer = "left" if obs.aim_error < -tolerance else "right" if obs.aim_error > tolerance else "hold"
    margin = 0.04 if obs.directive == "conserve" else 0.10
    fire = (
        abs(obs.aim_error) < max(margin, obs.half_width * 0.65)
        and obs.ammo > 0
        and obs.directive != "pacifist"
    )
    return steer, fire
