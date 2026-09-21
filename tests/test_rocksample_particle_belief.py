"""Synthetic finite-mixture oracles only; no environment imports or episodes."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from openjev.research.rocksample_particle_belief import ImpossibleObservationError, ParticleRockBelief


def tiny_belief(qualities=0.5, weights=None):
    return ParticleRockBelief.from_hypotheses([[[0, 0], [1, 0]], [[1, 0], [0, 0]]], qualities,
                                             weights, size=2, half_efficiency_distance=1)


def snapshot(belief):
    return belief._log_weights.copy(), belief._log_quality.copy(), belief.maps.copy()


def assert_unchanged(belief, before):
    for actual, expected in zip((belief._log_weights, belief._log_quality, belief.maps), before, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_seeded_prior_is_deterministic_distinct_and_normalized():
    belief = ParticleRockBelief()
    other = ParticleRockBelief()
    np.testing.assert_array_equal(belief.maps, other.maps)
    assert belief.maps.shape == (256, 11, 2)
    assert np.all((belief.maps[:, :, 0] >= 0) & (belief.maps[:, :, 0] < 11))
    assert np.all((belief.maps[:, :, 1] >= 0) & (belief.maps[:, :, 1] < 10))
    assert all(len(set(map(tuple, locations))) == 11 for locations in belief.maps)
    np.testing.assert_array_equal(belief.map_probabilities, np.full(256, 1 / 256))
    np.testing.assert_array_equal(belief.conditional_quality_probabilities, np.full((256, 11), 0.5))
    np.testing.assert_allclose(belief.quality_probabilities(), 0.5)
    np.testing.assert_array_equal(belief.expected_rewards(), np.zeros((11, 10)))
    assert belief.diagnostics() == {"ess": 256, "max_weight": 1 / 256,
                                     "alive_particles": 256, "particles": 256}
    assert not np.array_equal(belief.maps, ParticleRockBelief(seed=530002).maps)


def test_immutable_shared_maps_but_independent_mutable_state_and_audit_exports():
    belief = tiny_belief()
    cloned = belief.copy()
    assert np.shares_memory(cloned.maps, belief.maps)
    assert not np.shares_memory(cloned._log_weights, belief._log_weights)
    assert not np.shares_memory(cloned._log_quality, belief._log_quality)
    with pytest.raises(ValueError):
        cloned.maps[0, 0] = [1, 0]
    before = snapshot(belief)
    cloned.sample((0, 0))
    assert_unchanged(belief, before)
    cloned.map_probabilities[:] = 99
    cloned.conditional_quality_probabilities[:] = 99
    assert np.all(cloned.map_probabilities <= 1)
    assert np.all(cloned.conditional_quality_probabilities <= 1)


def test_source_hypothesis_arrays_cannot_mutate_initialization():
    maps = np.array([[[0, 0]], [[1, 0]]])
    qualities, weights = np.array([[0.2], [0.8]]), np.array([1.0, 3.0])
    belief = ParticleRockBelief.from_hypotheses(maps, qualities, weights, size=2)
    expected = snapshot(belief)
    maps[:] = 0
    qualities[:] = 0
    weights[:] = 0
    assert_unchanged(belief, expected)
    np.testing.assert_allclose(belief.map_probabilities, [0.25, 0.75])


def test_map_weights_use_pre_update_quality_likelihood_once():
    belief = ParticleRockBelief.from_hypotheses([[[0, 0]], [[1, 0]]], [[0.9], [0.2]],
                                               [0.3, 0.7], size=2, half_efficiency_distance=1)
    before = snapshot(belief)
    # At A, map0 has perfect sensing; map1 has eta=.75.
    evidence = np.array([0.9, 0.2 * 0.75 + 0.8 * 0.25])
    expected_probability = float(np.dot([0.3, 0.7], evidence))
    assert belief.check_probability(0, (0, 0)) == pytest.approx(expected_probability)
    conditioned = belief.condition_check(0, (0, 0), True)
    np.testing.assert_allclose(conditioned.map_probabilities, np.array([0.3, 0.7]) * evidence / expected_probability)
    np.testing.assert_allclose(conditioned.conditional_quality_probabilities[:, 0], [1, 0.15 / 0.35])
    assert_unchanged(belief, before)


def test_sampling_depletes_every_matching_hypothesis_without_weight_update():
    belief = tiny_belief(qualities=[[0.8, 0.3], [0.2, 0.9]], weights=[0.2, 0.8])
    before = belief._log_weights.copy()
    belief.sample((0, 0))
    np.testing.assert_array_equal(belief._log_weights, before)
    np.testing.assert_allclose(belief.conditional_quality_probabilities, [[0, 0.3], [0.2, 0]])
    assert belief.expected_rewards()[0, 0] == pytest.approx(-10)
    once = snapshot(belief)
    belief.sample((0, 0))
    assert_unchanged(belief, once)
    belief.sample((0, 1))  # exit column has no rock in any valid map
    assert_unchanged(belief, once)


def test_reset_restores_supplied_prior_without_redrawing_maps():
    belief = tiny_belief(qualities=[[0.8, 0.3], [0.2, 0.9]], weights=[0.2, 0.8])
    initial = snapshot(belief)
    conditioned = belief.condition_check(0, (0, 0), True)
    conditioned.sample((1, 0))
    conditioned.reset()
    assert_unchanged(conditioned, initial)
    assert np.shares_memory(belief.maps, conditioned.maps)


def test_exact_eight_world_native_oracle_through_checks_and_depletion():
    """Exhaust all two-map/four-quality worlds, using scalar probability arithmetic."""
    belief = tiny_belief()
    locations = [(0, 0), (1, 0)]
    worlds = [{"map": list(positions), "quality": list(qualities), "weight": 1 / 8}
              for positions in itertools.permutations(locations)
              for qualities in itertools.product((0, 1), repeat=2)]

    def probability(world, rock, coordinate):
        eta = (1 + 2 ** (-math.dist(coordinate, world["map"][rock]))) / 2
        return eta if world["quality"][rock] else 1 - eta

    actions = [("check", 0, (0, 0), True), ("check", 0, (0, 0), False),
               ("check", 1, (0, 0), True), ("sample", None, (0, 0), None),
               ("check", 1, (1, 0), False), ("check", 0, (1, 0), True)]
    for action, rock, coordinate, positive in actions:
        if action == "check":
            expected = sum(w["weight"] * probability(w, rock, coordinate) for w in worlds)
            assert belief.check_probability(rock, coordinate) == pytest.approx(expected, abs=1e-14)
            before = snapshot(belief)
            belief = belief.condition_check(rock, coordinate, positive)
            for world in worlds:
                p = probability(world, rock, coordinate)
                world["weight"] *= p if positive else 1 - p
            total = sum(w["weight"] for w in worlds)
            for world in worlds:
                world["weight"] /= total
            assert not np.shares_memory(belief._log_weights, before[0])
        else:
            belief.sample(coordinate)
            for world in worlds:
                for index in range(2):
                    if world["map"][index] == coordinate:
                        world["quality"][index] = 0
        expected_weights = np.zeros(2)
        expected_quality = np.zeros((2, 2))
        expected_rewards = np.zeros((2, 1))
        for world in worlds:
            index = int(world["map"][0] != (0, 0))
            expected_weights[index] += world["weight"]
            expected_quality[index] += world["weight"] * np.array(world["quality"])
            for coordinate, quality in zip(world["map"], world["quality"], strict=True):
                expected_rewards[coordinate] += world["weight"] * (10 if quality else -10)
        np.testing.assert_allclose(belief.map_probabilities, expected_weights, atol=1e-14)
        np.testing.assert_allclose(belief.quality_probabilities(), expected_quality.sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(belief.expected_rewards(), expected_rewards, atol=1e-14)
        for index, weight in enumerate(expected_weights):
            if weight > 0:
                np.testing.assert_allclose(belief.conditional_quality_probabilities[index],
                                           expected_quality[index] / weight, atol=1e-14)
        assert np.isfinite(belief.conditional_quality_probabilities).all()
    assert belief.diagnostics()["alive_particles"] == 1


def test_occupancy_and_reward_bounds_follow_distinct_maps_after_conditioning():
    belief = ParticleRockBelief(size=4, rocks=7, particles=32, seed=17)
    for rock, coordinate, positive in [(0, (0, 0), True), (1, (3, 2), False), (2, (1, 1), True)]:
        belief = belief.condition_check(rock, coordinate, positive)
    occupancy = np.zeros((4, 3))
    for weight, positions in zip(belief.map_probabilities, belief.maps, strict=True):
        for row, column in positions:
            occupancy[row, column] += weight
    assert np.all(occupancy <= 1 + 1e-14)
    assert occupancy.sum() == pytest.approx(7)
    assert np.all(np.abs(belief.expected_rewards()) <= 10 * occupancy + 1e-14)
    diagnostics = belief.diagnostics()
    assert 1 <= diagnostics["ess"] <= 32 + 1e-14
    assert 1 / 32 <= diagnostics["max_weight"] <= 1


def test_known_map_true_quality_reference_has_exact_reward_and_impossible_evidence():
    belief = ParticleRockBelief.from_hypotheses([[[0, 0], [2, 1]]], [1, 0], size=3)
    expected = np.zeros((3, 2))
    expected[0, 0], expected[2, 1] = 10, -10
    np.testing.assert_array_equal(belief.expected_rewards(), expected)
    assert belief.check_probability(0, (0, 0)) == 1
    assert belief.check_probability(1, (2, 1)) == 0
    before = snapshot(belief)
    with pytest.raises(ImpossibleObservationError, match="zero probability"):
        belief.condition_check(0, (0, 0), False)
    assert_unchanged(belief, before)
    belief.sample((0, 0))
    assert belief.expected_rewards()[0, 0] == -10
    assert belief.check_probability(0, (0, 0)) == 0


def test_dead_hypothesis_quality_is_normalized_and_never_revived_by_evidence():
    belief = ParticleRockBelief.from_hypotheses([[[0, 0]], [[0, 0]]], [[1], [0]], size=2)
    belief = belief.condition_check(0, (0, 0), True)
    np.testing.assert_array_equal(belief.map_probabilities, [1, 0])
    assert np.isfinite(belief.conditional_quality_probabilities).all()
    before = snapshot(belief)
    with pytest.raises(ImpossibleObservationError):
        belief.condition_check(0, (0, 0), False)
    assert_unchanged(belief, before)
    assert belief.diagnostics() == {"ess": 1, "max_weight": 1, "alive_particles": 1, "particles": 2}


def test_log_quality_survives_underflow_then_recovers_without_probability_floor():
    belief = ParticleRockBelief.from_hypotheses([[[0, 0]]], size=3, half_efficiency_distance=1)
    for _ in range(1000):
        belief = belief.condition_check(0, (0, 1), True)
    assert np.exp(belief._log_quality[0, 0, 0]) == 0
    assert np.isfinite(belief._log_quality[0, 0, 0])
    for _ in range(1000):
        belief = belief.condition_check(0, (0, 1), False)
    assert belief.quality_probabilities()[0] == pytest.approx(0.5, abs=1e-11)


def test_ess_uses_weights_and_alive_count_uses_log_support():
    belief = tiny_belief(weights=[1, 3])
    assert belief.diagnostics()["ess"] == pytest.approx(1 / (0.25**2 + 0.75**2))
    assert belief.diagnostics()["max_weight"] == pytest.approx(0.75)
    # Test-only extreme finite log posterior: no probability floor or fake death.
    belief._log_weights = np.array([0.0, -1000.0])
    assert belief.map_probabilities[1] == 0
    assert belief.diagnostics() == {"ess": 1, "max_weight": 1, "alive_particles": 2, "particles": 2}


@pytest.mark.parametrize("kwargs", [{"size": 1}, {"size": True}, {"rocks": 0}, {"size": 2, "rocks": 3},
                                    {"particles": 0}, {"particles": 1.5}, {"seed": -1}, {"seed": False},
                                    {"half_efficiency_distance": 0}, {"half_efficiency_distance": np.inf},
                                    {"half_efficiency_distance": np.nan}])
def test_invalid_prior_constants_fail(kwargs):
    with pytest.raises(ValueError):
        ParticleRockBelief(**kwargs)


@pytest.mark.parametrize("maps", [[], [[[0, 0], [0, 0]]], [[[0, 1]]], [[[2, 0]]], [[[-1, 0]]],
                                  [[[0.5, 0]]], [[[np.nan, 0]]], [[0, 0]], [[[True, False]]]])
def test_invalid_explicit_maps_fail(maps):
    with pytest.raises(ValueError):
        ParticleRockBelief.from_hypotheses(maps, size=2)


@pytest.mark.parametrize("qualities,weights", [(-0.1, None), (1.1, None), (np.nan, None),
                                              ([0.1, 0.2, 0.3], None), (0.5, [0, 0]),
                                              (0.5, [-1, 2]), (0.5, [np.inf, 1]), (0.5, [1]),
                                              (0.5, [np.nan, 1])])
def test_invalid_explicit_probabilities_fail(qualities, weights):
    with pytest.raises(ValueError):
        tiny_belief(qualities, weights)


@pytest.mark.parametrize("rock,position,positive,exception", [(-1, (0, 0), True, ValueError),
                                                            (2, (0, 0), True, ValueError),
                                                            (True, (0, 0), True, ValueError),
                                                            (0, (-1, 0), True, ValueError),
                                                            (0, (2, 0), True, ValueError),
                                                            (0, (0.5, 0), True, ValueError),
                                                            (0, (np.nan, 0), True, ValueError),
                                                            (0, (0, 0), 1, TypeError),
                                                            (0, (0, 0), -1, TypeError)])
def test_invalid_check_inputs_fail_atomically(rock, position, positive, exception):
    belief = tiny_belief()
    before = snapshot(belief)
    with pytest.raises(exception):
        belief.condition_check(rock, position, positive)
    assert_unchanged(belief, before)


def test_invalid_sampling_input_fails_atomically():
    belief = tiny_belief()
    before = snapshot(belief)
    with pytest.raises(ValueError):
        belief.sample((0, np.inf))
    assert_unchanged(belief, before)
