"""Synthetic public-input checks; no simulator imports, real maps or episodes."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from openjev.research.rocksample_public_belief import PublicRockBelief


def observation(belief, position, rock=None, sign=1):
    values = np.zeros(2 * belief.size + belief.rocks)
    values[position[0]] = 1
    values[belief.size + position[1]] = 1
    if rock is not None:
        values[2 * belief.size + rock] = sign
    return values


def install_test_only_table(belief, table):
    """Inject a degenerate synthetic prior without adding a privileged production API."""
    assert table.shape == belief.posterior.shape
    np.testing.assert_allclose(table.sum(axis=(1, 2)), 1)
    with np.errstate(divide="ignore"):
        belief._log_belief = np.log(table)


def point_location(belief, position, probability=0.5):
    table = np.zeros_like(belief.posterior)
    index = next(i for i, value in enumerate(belief.positions) if tuple(value) == position)
    table[:, index, :] = [1 - probability, probability]
    install_test_only_table(belief, table)


def test_uniform_native_grid_prior_and_copy_isolation():
    belief = PublicRockBelief()
    assert belief.posterior.shape == (11, 110, 2)
    assert belief.posterior.dtype == np.float64
    assert list(map(tuple, belief.positions)) == list(itertools.product(range(11), range(10)))
    np.testing.assert_allclose(belief.posterior, 1 / 220, rtol=0, atol=1e-18)
    np.testing.assert_allclose(belief.quality_probabilities(), 0.5)
    for position in ((0, 0), (10, 9), (3, 10)):
        np.testing.assert_allclose(belief.probabilities(position), 0.5)
    belief.posterior[:] = 0
    belief.positions[:] = 99
    assert np.all(belief.posterior > 0) and belief.positions.max() == 10


def test_known_location_matches_closed_form_bayes_and_prediction_precedes_update():
    belief = PublicRockBelief(size=3, rocks=1, half_efficiency_distance=1)
    point_location(belief, (0, 0))
    position = (0, 1)
    before = belief.posterior.copy()
    assert belief.probabilities(position)[0] == 0.5
    np.testing.assert_array_equal(belief.posterior, before)
    belief.update(position, 5, observation(belief, position, 0, 1), False)
    eta = (1 + 2 ** (-1)) / 2
    np.testing.assert_allclose(belief.quality_probabilities(), eta, rtol=0, atol=1e-15)
    np.testing.assert_allclose(belief.probabilities(position), eta**2 + (1 - eta)**2)
    belief.update(position, 5, observation(belief, position, 0, -1), False)
    np.testing.assert_allclose(belief.quality_probabilities(), 0.5, atol=1e-15)
    np.testing.assert_allclose(belief.posterior.sum(axis=(1, 2)), 1, atol=1e-15)


def test_repeated_evidence_survives_probability_underflow_and_recovers_without_floor():
    belief = PublicRockBelief(size=3, rocks=1, half_efficiency_distance=1)
    point_location(belief, (0, 0))
    position = (0, 1)
    for _ in range(1000):
        belief.update(position, 5, observation(belief, position, 0, 1), False)
    assert belief.posterior[0, 0, 0] == 0  # audit conversion underflows; log state does not
    assert np.isfinite(belief._log_belief[0, 0, 0])
    for _ in range(1000):
        belief.update(position, 5, observation(belief, position, 0, -1), False)
    np.testing.assert_allclose(belief.quality_probabilities(), 0.5, rtol=0, atol=1e-11)


def test_sampling_depletes_only_matching_location_hypotheses_for_every_rock():
    belief = PublicRockBelief(size=2, rocks=2)
    before = belief.posterior
    belief.update((0, 0), 4, observation(belief, (0, 0)), False)
    after = belief.posterior
    np.testing.assert_allclose(after[:, 0, :], [[0.5, 0], [0.5, 0]])
    np.testing.assert_allclose(after[:, 1, :], before[:, 1, :])
    np.testing.assert_allclose(after.sum(axis=2), before.sum(axis=2))
    np.testing.assert_allclose(belief.quality_probabilities(), 0.25)
    belief.update((0, 0), 4, observation(belief, (0, 0)), False)
    np.testing.assert_array_equal(belief.posterior, after)


def test_perfect_sensor_preserves_exact_zeros_and_rejects_impossible_evidence_atomically():
    belief = PublicRockBelief(size=2, rocks=1)
    point_location(belief, (0, 0))
    belief.update((0, 0), 5, observation(belief, (0, 0), 0, 1), False)
    assert belief.quality_probabilities()[0] == 1
    assert belief.probabilities((0, 0))[0] == 1
    before = belief._log_belief.copy()
    with pytest.raises(ValueError, match="zero probability"):
        belief.update((0, 0), 5, observation(belief, (0, 0), 0, -1), False)
    np.testing.assert_array_equal(belief._log_belief, before)
    belief.update((0, 0), 4, observation(belief, (0, 0)), False)
    assert belief.quality_probabilities()[0] == 0
    assert belief.probabilities((0, 0))[0] == 0


@pytest.mark.parametrize("old_action", [4, 5, 999, None])
def test_done_ignores_old_action_and_coordinate_and_restores_fresh_prior(old_action):
    belief = PublicRockBelief(size=3, rocks=2)
    belief.update((0, 0), 5, observation(belief, (0, 0), 0, 1), False)
    belief.update(object(), old_action, observation(belief, (2, 1)), np.bool_(True))
    np.testing.assert_array_equal(belief.posterior, PublicRockBelief(size=3, rocks=2).posterior)
    belief.update((2, 1), 6, observation(belief, (2, 1), 1, -1), False)
    assert belief.quality_probabilities()[1] < 0.5
    assert belief.quality_probabilities()[0] == 0.5


@pytest.mark.parametrize("position,action,next_position", [((0, 0), 0, (0, 0)), ((1, 0), 0, (0, 0)),
                                                          ((1, 0), 1, (1, 1)), ((1, 0), 2, (2, 0)),
                                                          ((1, 1), 3, (1, 0)), ((2, 0), 2, (2, 0))])
def test_movement_preserves_all_evidence(position, action, next_position):
    belief = PublicRockBelief(size=3, rocks=2)
    belief.update(position, 5, observation(belief, position, 0, 1), False)
    before = belief._log_belief.copy()
    belief.update(position, action, observation(belief, next_position), False)
    np.testing.assert_array_equal(belief._log_belief, before)


def test_exhaustive_independent_location_oracle_matches_full_transition_sequence():
    """Enumerate the approximation's joint worlds, including overlapping locations.

    This is deliberately not a native no-overlap map oracle. Oracle arithmetic
    uses ordinary probability weights and independent scalar sensor evaluation.
    """
    belief = PublicRockBelief(size=2, rocks=2, half_efficiency_distance=1)
    locations = [(0, 0), (1, 0)]
    hypotheses = [(location, quality) for location in locations for quality in (0, 1)]
    worlds = [{"locations": [x[0] for x in choices], "qualities": [x[1] for x in choices],
               "weight": 1 / 16} for choices in itertools.product(hypotheses, repeat=2)]
    sequence = [((0, 0), 5, (0, 0), 0, 1), ((0, 0), 6, (0, 0), 1, -1),
                ((0, 0), 2, (1, 0), None, 0), ((1, 0), 5, (1, 0), 0, -1),
                ((1, 0), 4, (1, 0), None, 0), ((1, 0), 0, (0, 0), None, 0),
                ((0, 0), 6, (0, 0), 1, 1), ((0, 0), 5, (0, 0), 0, 1)]

    def positive(world, rock, position):
        eta = (1 + 2 ** (-math.dist(position, world["locations"][rock]))) / 2
        return eta if world["qualities"][rock] else 1 - eta

    for previous, action, position, checked, sign in sequence:
        expected_prediction = [sum(w["weight"] * positive(w, rock, position) for w in worlds)
                               for rock in range(2)]
        np.testing.assert_allclose(belief.probabilities(position), expected_prediction, atol=1e-14)
        for world in worlds:
            if action == 4:
                for rock in range(2):
                    if world["locations"][rock] == previous:
                        world["qualities"][rock] = 0
            elif checked is not None:
                probability = positive(world, checked, position)
                world["weight"] *= probability if sign == 1 else 1 - probability
        normalizer = sum(w["weight"] for w in worlds)
        for world in worlds:
            world["weight"] /= normalizer
        expected = np.zeros((2, 2, 2))
        for world in worlds:
            for rock in range(2):
                expected[rock, locations.index(world["locations"][rock]), world["qualities"][rock]] += (
                    world["weight"])
        belief.update(previous, action, observation(belief, position, checked, sign), False)
        np.testing.assert_allclose(belief.posterior, expected, atol=1e-14)
        np.testing.assert_allclose(belief.posterior.sum(axis=(1, 2)), 1, atol=1e-14)


def test_native_no_overlap_oracle_exposes_the_declared_factorization_approximation():
    """Eight native worlds show why the public table is not a native exact filter.

    Conflicting reads at A rule out rock 0 being at A (its sensor would be
    perfect), so native exclusion forces rock 1 to A. The independent filter
    cannot transfer that location evidence between rocks.
    """
    belief = PublicRockBelief(size=2, rocks=2, half_efficiency_distance=1)
    locations = [(0, 0), (1, 0)]
    worlds = [{"locations": positions, "qualities": qualities, "weight": 1 / 8}
              for positions in itertools.permutations(locations)
              for qualities in itertools.product((0, 1), repeat=2)]
    for rock, sign in [(0, 1), (0, -1), (1, 1)]:
        for world in worlds:
            eta = (1 + 2 ** (-math.dist((0, 0), world["locations"][rock]))) / 2
            predicted = eta if world["qualities"][rock] else 1 - eta
            world["weight"] *= predicted if sign == 1 else 1 - predicted
        total = sum(world["weight"] for world in worlds)
        for world in worlds:
            world["weight"] /= total
        belief.update((0, 0), 5 + rock, observation(belief, (0, 0), rock, sign), False)
        native = np.zeros((2, 2, 2))
        for world in worlds:
            for index in range(2):
                native[index, locations.index(world["locations"][index]), world["qualities"][index]] += (
                    world["weight"])
        if rock == 0:
            # The observed rock's marginal is exact until cross-rock evidence is used.
            np.testing.assert_allclose(belief.posterior[0], native[0], atol=1e-14)
    assert native[1, 0, 1] == 1
    assert belief.quality_probabilities()[1] == pytest.approx((1 + 0.75) / 2)
    assert not np.allclose(belief.posterior[1], native[1])


@pytest.mark.parametrize("kwargs", [{"size": True}, {"size": 1}, {"size": 3.5}, {"rocks": False},
                                    {"rocks": 0}, {"size": 2, "rocks": 3},
                                    {"half_efficiency_distance": 0}, {"half_efficiency_distance": -1},
                                    {"half_efficiency_distance": np.inf},
                                    {"half_efficiency_distance": np.nan},
                                    {"half_efficiency_distance": True}])
def test_invalid_public_constants_rejected(kwargs):
    with pytest.raises(ValueError):
        PublicRockBelief(**kwargs)


@pytest.mark.parametrize("position", [(-1, 0), (3, 0), (0, 3), (0.1, 0), (np.nan, 0),
                                      (0, np.inf), (0,), (True, False), ("0", "0")])
def test_prediction_rejects_invalid_public_coordinates(position):
    with pytest.raises(ValueError, match="position"):
        PublicRockBelief(size=3, rocks=2).probabilities(position)


@pytest.mark.parametrize("defect", ["shape", "nan", "not_onehot", "no_onehot", "bad_sign", "wrong_rock",
                                    "two_readings", "check_missing", "move_reading", "sample_reading",
                                    "wrong_position", "bad_action", "bool_action", "float_action",
                                    "bad_done", "done_check", "done_exit", "previous_exit", "next_exit"])
def test_invalid_public_transition_rejected_without_state_change(defect):
    belief = PublicRockBelief(size=3, rocks=2)
    belief.update((0, 0), 5, observation(belief, (0, 0), 0, 1), False)
    previous, action, done = (0, 0), 5, False
    obs = observation(belief, previous, 0, 1)
    if defect == "shape":
        obs = obs[:-1]
    elif defect == "nan":
        obs[0] = np.nan
    elif defect == "not_onehot":
        obs[1] = 1
    elif defect == "no_onehot":
        obs[0] = 0
    elif defect == "bad_sign":
        obs[-2] = 0.5
    elif defect == "wrong_rock":
        obs[-2:] = [0, 1]
    elif defect == "two_readings":
        obs[-1] = -1
    elif defect == "check_missing":
        obs[-2:] = 0
    elif defect == "move_reading":
        action = 0
    elif defect == "sample_reading":
        action = 4
    elif defect == "wrong_position":
        obs = observation(belief, (1, 0), 0, 1)
    elif defect == "bad_action":
        action = 7
    elif defect == "bool_action":
        action = True
    elif defect == "float_action":
        action = 5.0
    elif defect == "bad_done":
        done = 1
    elif defect == "done_check":
        done = True
    elif defect == "done_exit":
        done, obs = True, observation(belief, (0, 2))
    elif defect == "previous_exit":
        previous, action, obs = (0, 2), 3, observation(belief, (0, 1))
    elif defect == "next_exit":
        previous, action, obs = (0, 1), 1, observation(belief, (0, 2))
    before = belief._log_belief.copy()
    with pytest.raises(TypeError if defect == "bad_done" else ValueError):
        belief.update(previous, action, obs, done)
    np.testing.assert_array_equal(belief._log_belief, before)
