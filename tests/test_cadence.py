from dataclasses import replace

from openjev.domain import Observation, teacher_action
from openjev.research.cadence import CadenceControl


def test_zero_rest_matches_rules_and_constraints():
    policy = CadenceControl()
    for visible in (False, True):
        for aim in (-.3, -.05, 0., .05, .3):
            for ammo in (0., 30., 100.):
                for directive in ('hunt', 'pacifist'):
                    obs = Observation(visible=visible, aim_error=aim, ammo=ammo, directive=directive)
                    assert policy.decide(obs) == teacher_action(obs)


def test_ammo_event_is_causal_and_not_clipped_at_high_ammo():
    obs = Observation(visible=True, ammo=100.)
    policy = CadenceControl(rest_windows=1)
    assert policy.decide(obs)[1]
    # Command alone is not an expenditure event.
    assert policy.decide(obs)[1]
    assert not policy.decide(replace(obs, ammo=99.))[1]
    assert policy.decide(replace(obs, ammo=99.))[1]
    # A fresh episode cannot inherit a cooldown.
    assert CadenceControl(rest_windows=1).decide(replace(obs, ammo=99.))[1]


def test_two_window_rest_and_command_control_have_distinct_semantics():
    obs = Observation(visible=True, ammo=30.)
    policy = CadenceControl(rest_windows=2)
    assert policy.decide(obs)[1]
    assert not policy.decide(replace(obs, ammo=29.))[1]
    assert not policy.decide(replace(obs, ammo=29.))[1]
    assert policy.decide(replace(obs, ammo=29.))[1]
    command = CadenceControl(event='command', rest_windows=1)
    assert [command.decide(obs)[1] for _ in range(4)] == [True, False, True, False]
