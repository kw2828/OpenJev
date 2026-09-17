from dataclasses import replace

from openjev.domain import Observation
from openjev.research.event_cadence import EventCadenceControl


def test_hit_feedback_handles_replenished_ammo_without_future_leakage():
    obs = Observation(visible=True, ammo=100.)
    control = EventCadenceControl(event='ammo_or_hit', rest_windows=1)
    first = control.decide(obs)
    assert first[1]
    control.observe_outcome(1)
    assert first[1]  # A later outcome cannot change the already-issued decision.
    assert not control.decide(obs)[1]  # Same displayed ammo, but a hit was observed.
    control.observe_outcome(0)
    assert control.decide(obs)[1]


def test_ammo_feedback_handles_misses_and_combined_events_do_not_double_rest():
    obs = Observation(visible=True, ammo=30.)
    control = EventCadenceControl(event='ammo_or_hit', rest_windows=1)
    assert control.decide(obs)[1]
    control.observe_outcome(0)
    assert not control.decide(replace(obs, ammo=29.))[1]
    control.observe_outcome(0)
    assert control.decide(replace(obs, ammo=29.))[1]
    control.observe_outcome(1)
    assert not control.decide(replace(obs, ammo=28.))[1]
    control.observe_outcome(0)
    assert control.decide(replace(obs, ammo=28.))[1]
    assert EventCadenceControl(event='ammo_or_hit',rest_windows=1).decide(obs)[1]
