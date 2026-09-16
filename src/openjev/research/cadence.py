"""Small causal firing-cadence controls, not a learned world model."""
from dataclasses import dataclass

from openjev.domain import teacher_action


@dataclass
class CadenceControl:
    margin: float = .10
    event: str = 'ammo'
    rest_windows: int = 0
    remaining: int = 0
    previous_ammo: float | None = None

    def decide(self, observation):
        # Only observations available before this decision may trigger a rest.
        if (self.event == 'ammo' and self.previous_ammo is not None
                and observation.ammo < self.previous_ammo):
            self.remaining = self.rest_windows
        self.previous_ammo = observation.ammo
        steer, _ = teacher_action(observation)
        aligned = (observation.visible
                   and abs(observation.aim_error) < max(self.margin, observation.half_width*.65)
                   and observation.ammo > 0 and observation.directive != 'pacifist')
        fire = bool(aligned and self.remaining == 0)
        self.remaining = max(0, self.remaining-1)
        if fire and self.event == 'command':
            self.remaining = self.rest_windows
        return steer, fire
