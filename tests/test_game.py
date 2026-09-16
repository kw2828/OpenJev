import pytest

from openjev.domain import SCENARIOS, hard_decision
from openjev.evaluate import run_episode
from openjev.game import Doom


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_engine_loads_and_advances(scenario):
    with Doom(scenario, 81) as doom:
        obs = doom.observe()
        assert doom.frame().shape == (480, 640, 3)
        before = doom.stats()["game_seconds"]
        doom.step(hard_decision("right", False, "test"), obs)
        assert doom.stats()["game_seconds"] > before


def test_learned_policy_scores_in_real_game():
    result = run_episode("local", "defend_the_center", 42)
    assert result["finished"]
    assert result["kills"] >= 5


def test_pacifist_never_spends_ammo():
    with Doom(seed=42) as doom:
        initial = doom.observe("pacifist").ammo
        for _ in range(30):
            obs = doom.observe("pacifist")
            doom.step(hard_decision("hold", True, "test"), obs)
        assert doom.observe("pacifist").ammo == initial
