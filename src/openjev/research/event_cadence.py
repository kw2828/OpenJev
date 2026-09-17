"""Combine two observable event channels when ammo replenishment hides shots."""
from dataclasses import dataclass

from openjev.research.cadence import CadenceControl


@dataclass
class EventCadenceControl(CadenceControl):
    previous_hit: bool = False

    def observe_outcome(self, hit_count_change):
        # Called after engine advancement, for use by the following decision only.
        self.previous_hit = hit_count_change > 0

    def decide(self, observation):
        hit_event = self.event in ('hit', 'ammo_or_hit') and self.previous_hit
        ammo_event = (self.event == 'ammo_or_hit' and self.previous_ammo is not None
                      and observation.ammo < self.previous_ammo)
        if hit_event or ammo_event:
            self.remaining = self.rest_windows
        self.previous_hit = False
        return super().decide(observation)
