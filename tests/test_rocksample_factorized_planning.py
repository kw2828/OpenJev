"""Tiny synthetic belief/planner checks; no simulator, real maps or episodes."""
from __future__ import annotations

import inspect
import itertools
import json
import math

import numpy as np
import pytest

from openjev.research.rocksample_factorized_planning import FactorizedRockBelief
from openjev.research.rocksample_memory_controls import PublicTransition, reconstruct
from openjev.research.rocksample_route_planner import plan_action


def point_table(belief, position, good):
    """Test-only degenerate marginals, without a production hidden-map interface."""
    table = np.zeros_like(belief.posterior)
    index = list(map(tuple, belief.positions)).index(position)
    table[:, index, :] = [1.0 - good, good]
    with np.errstate(divide="ignore"):
        belief._core._log_belief = np.log(table)


def test_complete_prior_and_public_only_constructor():
    belief = FactorizedRockBelief()
    assert set(inspect.signature(FactorizedRockBelief).parameters) == {
        "size", "rocks", "half_efficiency_distance"}
    assert belief.posterior.shape == (11, 110, 2)
    np.testing.assert_allclose(belief.posterior, 1 / 220, rtol=0, atol=1e-18)
    assert np.all(belief.posterior > 0)
    np.testing.assert_array_equal(belief.expected_rewards(), np.zeros((11, 10)))
    diagnostic = belief.diagnostics()
    assert diagnostic["ess"] is None and diagnostic["particles"] == 0
    assert diagnostic["locations_per_rock"] == 110
    assert diagnostic["kind"] == "factorized_full_support"
    assert diagnostic["maximum_cell_occupancy"] == pytest.approx(0.1)
    assert diagnostic["overfull_cells"] == 0
    json.dumps(diagnostic, allow_nan=False)


def test_clone_branch_audit_arrays_and_reset_are_independent():
    belief = FactorizedRockBelief(size=3, rocks=2)
    before = belief.posterior
    clone = belief.copy()
    clone.sample((0, 0))
    branch = belief.condition_check(0, (1, 0), np.bool_(True))
    np.testing.assert_array_equal(belief.posterior, before)
    assert branch.quality_probabilities()[0] > 0.5
    assert branch.quality_probabilities()[1] == 0.5
    branch.posterior[:] = 0
    branch.positions[:] = 99
    assert np.any(branch.posterior) and branch.positions.max() == 2
    branch.reset()
    np.testing.assert_array_equal(branch.posterior, before)
    assert not np.array_equal(clone.posterior, before)


def test_exhaustive_collision_allowed_worlds_match_predictions_branches_and_rewards():
    """Scalar enumeration is independent of the core's log-table arithmetic.

    Worlds intentionally allow both rocks in one cell, matching this approximation
    rather than the native exclusion rule. Its additive reward can be non-native.
    """
    belief = FactorizedRockBelief(size=2, rocks=2, half_efficiency_distance=1)
    locations = [(0, 0), (1, 0)]
    atoms = list(itertools.product(locations, (0, 1)))
    worlds = [{"locations": [x[0] for x in choices], "quality": [x[1] for x in choices],
               "mass": 1 / 16} for choices in itertools.product(atoms, repeat=2)]

    def likelihood(world, rock, position):
        accuracy = (1.0 + 2.0 ** (-math.dist(position, world["locations"][rock]))) / 2
        return accuracy if world["quality"][rock] else 1.0 - accuracy

    for operation, rock, position, positive in [
        ("check", 0, (0, 0), True), ("check", 1, (1, 0), False),
        ("check", 0, (0, 0), False), ("sample", None, (1, 0), None),
        ("check", 1, (0, 0), True), ("check", 0, (1, 0), False),
    ]:
        before = belief.posterior
        for query in range(2):
            expected = math.fsum(w["mass"] * likelihood(w, query, position) for w in worlds)
            assert belief.check_probability(query, position) == pytest.approx(expected, abs=1e-14)
        np.testing.assert_array_equal(belief.posterior, before)
        if operation == "sample":
            belief.sample(position)
            for world in worlds:
                for i in range(2):
                    if world["locations"][i] == position:
                        world["quality"][i] = 0
        else:
            belief = belief.condition_check(rock, position, positive)
            for world in worlds:
                probability = likelihood(world, rock, position)
                world["mass"] *= probability if positive else 1.0 - probability
            normalizer = math.fsum(w["mass"] for w in worlds)
            for world in worlds:
                world["mass"] /= normalizer
        expected = np.zeros((2, 2, 2))
        for world in worlds:
            for i in range(2):
                expected[i, locations.index(world["locations"][i]), world["quality"][i]] += world["mass"]
        np.testing.assert_allclose(belief.posterior, expected, rtol=0, atol=1e-14)
        rewards = [math.fsum(w["mass"] * sum(10 if w["quality"][i] else -10 for i in range(2)
                                            if w["locations"][i] == target) for w in worlds)
                   for target in locations]
        np.testing.assert_allclose(belief.expected_rewards().ravel(), rewards, rtol=0, atol=1e-13)


def test_branch_mixture_recovers_parent_and_tiny_planner_uses_copies():
    belief = FactorizedRockBelief(size=2, rocks=1, half_efficiency_distance=1)
    point_table(belief, (0, 0), 0.5)
    prior = belief.posterior
    p = belief.check_probability(0, (0, 0))
    plus = belief.condition_check(0, (0, 0), True)
    minus = belief.condition_check(0, (0, 0), False)
    np.testing.assert_allclose(p * plus.posterior + (1 - p) * minus.posterior, prior)
    result = plan_action(belief, (0, 0), np.zeros((2, 1), dtype=bool), 3, 1)
    assert result["kind"] == "sense" and result["actions"] == [5]
    assert result["expected_return"] == pytest.approx(15.0)
    np.testing.assert_array_equal(belief.posterior, prior)


def test_sample_changes_quality_only_is_idempotent_and_reset_restores_support():
    belief = FactorizedRockBelief(size=3, rocks=2)
    initial = belief.posterior
    belief.sample((0, 0))
    after = belief.posterior
    np.testing.assert_array_equal(after.sum(axis=2), initial.sum(axis=2))
    assert np.all(after[:, 0, 1] == 0)
    np.testing.assert_allclose(belief.expected_rewards()[0, 0], -10 * 2 / 6)
    belief.sample((0, 0))
    np.testing.assert_array_equal(belief.posterior, after)
    belief.reset()
    np.testing.assert_array_equal(belief.posterior, initial)


def test_noncoherent_occupancy_and_reward_forecasts_remain_visible_unclipped():
    belief = FactorizedRockBelief(size=2, rocks=2, half_efficiency_distance=1)
    # Contradictory perfect-at-A checks exclude A separately for each rock.
    # This public history retains mass in the independent model but is impossible
    # under the native two-rock/two-cell exclusion constraint.
    for rock in range(2):
        belief = belief.condition_check(rock, (0, 0), True).condition_check(rock, (0, 0), False)
    diagnostic = belief.diagnostics()
    assert diagnostic["maximum_cell_occupancy"] == pytest.approx(2)
    assert diagnostic["overfull_cells"] == 1
    assert diagnostic["occupancy_excess_mass"] == pytest.approx(1)
    for rock in range(2):
        belief = belief.condition_check(rock, (1, 0), True)
    assert belief.expected_rewards()[1, 0] == 20
    assert belief.diagnostics()["maximum_expected_reward"] == 20
    belief.sample((1, 0))
    assert belief.expected_rewards()[1, 0] == -20
    assert belief.diagnostics()["minimum_expected_reward"] == -20


def test_history_controls_keep_old_depletion_and_original_prior_unchanged():
    prior = FactorizedRockBelief(size=2, rocks=1)
    history = [PublicTransition((0, 0), 5, 1), PublicTransition((0, 0), 4)]
    history += [PublicTransition((1, 0), 0)] * 129
    history += [PublicTransition((1, 0), 5, 1)]
    before = prior.posterior
    for mode in ("full", "recent128", "latest"):
        belief = reconstruct(prior, history, mode)
        assert belief.posterior[0, 0, 1] == 0
        assert belief.expected_rewards()[0, 0] <= 0
    np.testing.assert_array_equal(prior.posterior, before)
    recent = prior.copy()
    recent.sample((0, 0))
    recent = recent.condition_check(0, (1, 0), True)
    np.testing.assert_array_equal(reconstruct(prior, history, "recent128").posterior, recent.posterior)


def test_impossible_branch_and_invalid_mutations_leave_parent_untouched():
    belief = FactorizedRockBelief(size=2, rocks=1)
    point_table(belief, (0, 0), 1)
    before = belief.posterior
    assert belief.check_probability(0, (0, 0)) == 1
    with pytest.raises(ValueError, match="zero probability"):
        belief.condition_check(0, (0, 0), False)
    np.testing.assert_array_equal(belief.posterior, before)


@pytest.mark.parametrize("position", [(0, 2), (-1, 0), (3, 0), (0.5, 0), (np.nan, 0),
                                      (True, False), ("0", "0"), (0,)])
def test_invalid_and_exit_positions_are_rejected_atomically(position):
    belief = FactorizedRockBelief(size=3, rocks=2)
    before = belief.posterior
    for method in (lambda: belief.check_probability(0, position),
                   lambda: belief.condition_check(0, position, True), lambda: belief.sample(position)):
        with pytest.raises(ValueError):
            method()
    np.testing.assert_array_equal(belief.posterior, before)


@pytest.mark.parametrize("rock", [True, np.bool_(False), -1, 2, 0.0, "0"])
def test_invalid_rock_indices_rejected(rock):
    belief = FactorizedRockBelief(size=3, rocks=2)
    for method in (lambda: belief.check_probability(rock, (0, 0)),
                   lambda: belief.condition_check(rock, (0, 0), True)):
        with pytest.raises(ValueError):
            method()


@pytest.mark.parametrize("positive", [1, -1, 0, "True", None, np.nan])
def test_reading_requires_boolean(positive):
    with pytest.raises(TypeError, match="boolean"):
        FactorizedRockBelief().condition_check(0, (0, 0), positive)
