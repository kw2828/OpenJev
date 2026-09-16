from dataclasses import replace

import numpy as np
import pytest

from openjev.domain import Decision, Observation, hard_decision
from openjev.policies import LocalPolicy
from openjev.train import dataset


@pytest.mark.parametrize("error,expected", [(-0.6, "left"), (0.0, "hold"), (0.6, "right")])
def test_learned_steering(error, expected):
    obs = Observation(visible=True, aim_error=error, half_width=0.03, distance=300, ammo=20)
    assert LocalPolicy().decide(obs).steer == expected


def test_learned_fire_and_pacifist():
    obs = Observation(visible=True, ammo=20, half_width=0.03, distance=300)
    policy = LocalPolicy()
    assert policy.decide(obs).fire
    assert not policy.decide(replace(obs, directive="pacifist")).fire


@pytest.mark.parametrize("directive,ammo", [("pacifist", 30), ("hunt", 0)])
def test_constraints_override_any_backend(directive, ammo):
    obs = Observation(directive=directive, ammo=ammo)
    malicious = hard_decision("left", True, "test")
    assert malicious.buttons(obs) == [1, 0, 0]


@pytest.mark.parametrize(
    "probabilities",
    [
        {"left": 0.1, "hold": 0.1, "right": 0.1},
        {"left": float("nan"), "hold": 0.0, "right": 1.0},
        {"left": -1.0, "hold": 1.0, "right": 1.0},
        {"left": 1.0},
    ],
)
def test_invalid_distributions_fail_closed(probabilities):
    with pytest.raises(ValueError):
        Decision("left", False, probabilities, 0.0, "test")


def test_no_conflicting_buttons():
    policy = LocalPolicy()
    for error in np.linspace(-1, 1, 50):
        obs = Observation(visible=True, aim_error=float(error), ammo=20)
        buttons = policy.decide(obs).buttons(obs)
        assert buttons[0] + buttons[1] <= 1


def test_held_out_synthetic_agreement():
    x, y = dataset(1000, 913)
    logits = LocalPolicy().logits(x)
    assert np.mean(logits[:, :3].argmax(1) == y[:, 0]) > 0.97
    assert np.mean(logits[:, 3:].argmax(1) == y[:, 1]) > 0.97
