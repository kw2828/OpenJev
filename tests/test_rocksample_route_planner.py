"""Closed finite-hypothesis planner cases; no environment or model imports."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openjev.research.rocksample_route_planner import plan_action


class FiniteBelief:
    """Artificial exact mixture with a declared binary sensor, independent of position."""

    def __init__(self, grids, signs, weights=None, accuracy=1.):
        self.grids = np.asarray(grids, dtype=float)
        self.signs = np.asarray(signs, dtype=bool)
        self.size = self.grids.shape[1]
        self.rocks = self.signs.shape[1]
        self.weights = (np.ones(len(grids)) / len(grids) if weights is None else np.asarray(weights, dtype=float))
        self.accuracy = accuracy

    def expected_rewards(self):
        return np.tensordot(self.weights, self.grids, axes=1)

    def check_probability(self, rock, position):
        likelihoods = np.where(self.signs[:, rock], self.accuracy, 1 - self.accuracy)
        return float(self.weights @ likelihoods)

    def condition_check(self, rock, position, positive):
        assert type(positive) is bool
        likelihoods = np.where(self.signs[:, rock] == positive, self.accuracy, 1 - self.accuracy)
        joint = self.weights * likelihoods
        if not joint.sum():
            raise ValueError("impossible branch")
        return FiniteBelief(self.grids, self.signs, joint / joint.sum(), self.accuracy)


def uncertain(*, accuracy=1., good_probability=.5, reward=10.):
    grids = np.zeros((2, 3, 2))
    grids[0, 0, 0], grids[1, 0, 0] = reward, -reward
    return FiniteBelief(grids, [[True], [False]], [good_probability, 1 - good_probability], accuracy)


def ledger(size=3):
    return np.zeros((size, size - 1), dtype=bool)


def test_known_good_samples_then_exits_without_redundant_sensing():
    belief = uncertain(good_probability=1.)
    result = plan_action(belief, (0, 0), ledger(), 12, 64)
    assert result["kind"] == "exploit"
    assert result["actions"] == [4]
    assert result["expected_return"] == 20.
    assert result["forecast_tour_steps"] == 3
    assert result["work"]["zero_probability_branches_skipped"] > 0
    assert result["sensing_gain"] == 0


def test_known_bad_and_empty_beliefs_choose_exit():
    belief = uncertain(good_probability=0.)
    result = plan_action(belief, (0, 0), ledger(), 12, 64)
    assert result["kind"] == "exit" and result["actions"] == [1, 1]
    assert result["expected_return"] == 10.
    belief.grids[:] = 0
    assert plan_action(belief, (0, 0), ledger(), 12, 64)["kind"] == "exit"


def test_perfect_check_has_closed_form_collectable_value():
    # A fair unknown good/bad rock at the current cell: inspect, sample iff good,
    # exit. Expected return is .5*20 + .5*10 = 15, not 20.
    belief = uncertain()
    original = belief.weights.copy()
    result = plan_action(belief, (0, 0), ledger(), 12, 64)
    assert result["kind"] == "sense" and result["actions"] == [5]
    assert result["expected_return"] == 15.
    assert result["sensing_gain"] == 5.
    assert result["check_rocks"] == [0]
    np.testing.assert_array_equal(belief.weights, original)


def test_two_noisy_checks_can_help_when_one_cannot():
    # Prior good=.2, sensor=.75. A single positive leaves P(good)<.5.
    # Two positives have joint good=.1125, bad=.05, producing +.625 expected
    # sampling reward. All other outcomes exit. Four checks are unaffordable.
    belief = uncertain(accuracy=.75, good_probability=.2)
    one = plan_action(belief, (0, 0), ledger(), 12, 1)
    two = plan_action(belief, (0, 0), ledger(), 12, 2)
    assert one["kind"] == "exit" and one["expected_return"] == 10.
    assert two["kind"] == "sense" and two["check_rocks"] == [0, 0]
    assert two["expected_return"] == pytest.approx(10.625, abs=1e-12)
    assert two["actions"] == [5, 5]


def test_repeated_observation_branches_are_conditionally_weighted():
    # Four .75-reliable readings of a fair bit give positive expected reward
    # 5*((.75**4-.25**4)+4*(.75**3*.25-.25**3*.75)) = 3.4375.
    result = plan_action(uncertain(accuracy=.75), (0, 0), ledger(), 12, 4)
    assert result["check_rocks"] == [0, 0, 0, 0]
    assert result["expected_return"] == pytest.approx(13.4375, abs=1e-12)
    assert result["work"]["maximum_branch_mass_error"] <= 1e-12


def test_known_quality_diffuse_location_is_not_certain_reward_at_a_cell():
    grids = np.zeros((2, 3, 2))
    grids[0, 0, 0] = 10
    grids[1, 2, 1] = 10
    belief = FiniteBelief(grids, [[True], [True]])
    result = plan_action(belief, (0, 0), ledger(), 3, 0)
    assert result["first_sample_target"] == [0, 0]
    assert result["expected_return"] == 15.  # .5 occupancy * 10 plus exit
    assert result["forecast_tour_steps"] == 3


def test_perfect_branches_choose_different_sample_targets():
    grids = np.zeros((2, 3, 2))
    grids[0, 0, 0], grids[0, 0, 1] = 10, -10
    grids[1, 0, 0], grids[1, 0, 1] = -10, 10
    belief = FiniteBelief(grids, [[True], [False]])
    result = plan_action(belief, (0, 0), ledger(), 8, 1)
    assert result["kind"] == "sense" and result["expected_return"] == 20.
    for positive, target in ((True, [0, 0]), (False, [0, 1])):
        posterior = belief.condition_check(0, (0, 0), positive)
        option = plan_action(posterior, (0, 0), ledger(), 7, 0)
        assert option["first_sample_target"] == target


def test_positive_cell_prefix_and_first_sample_commitment():
    grids = np.zeros((1, 3, 2))
    grids[0, 0, 0], grids[0, 0, 1] = 2, 3
    belief = FiniteBelief(grids, [[True]])
    result = plan_action(belief, (0, 0), ledger(), 4, 0)
    assert result["expected_return"] == 15.
    assert result["actions"] == [4]  # Not the whole four-step route.
    assert result["forecast_tour_steps"] == 4
    shorter = plan_action(belief, (0, 0), ledger(), 3, 0)
    assert shorter["expected_return"] == 13.
    assert shorter["actions"] == [1, 4]


def test_persistent_public_ledger_prevents_repeat_sampling():
    belief = uncertain(good_probability=1.)
    sampled = ledger()
    sampled[0, 0] = True  # Still true after arbitrarily many public transitions.
    before = sampled.copy()
    result = plan_action(belief, (0, 0), sampled, 200, 64)
    assert result["kind"] == "exit" and result["expected_return"] == 10.
    assert result["actions"] == [1, 1]
    np.testing.assert_array_equal(sampled, before)


@pytest.mark.parametrize("remaining,actions,value", [(0, [], 0.), (1, [1], 0.), (2, [1, 1], 10.)])
def test_exit_reservation_and_short_horizon(remaining, actions, value):
    result = plan_action(uncertain(good_probability=1.), (0, 0), ledger(), remaining, 64)
    assert result["kind"] == "exit" and result["actions"] == actions
    assert result["expected_return"] == value
    assert result["checks_committed"] == 0


def test_no_partial_sensing_bundle_and_exact_margin_tie():
    # The one perfect check can consume the final spare step, but then no sample
    # fits before exit: its continuation is worth only10.
    result = plan_action(uncertain(), (0, 0), ledger(), 3, 4)
    assert result["kind"] == "exit"
    margin = plan_action(uncertain(reward=.1), (0, 0), ledger(), 12, 4)
    assert margin["best_sensing_return"] == 10.05
    assert margin["kind"] == "exit"  # Strictly more than the fixed .05 margin.


class Uninformative:
    size, rocks = 3, 4

    def expected_rewards(self):
        return np.zeros((3, 2))

    def check_probability(self, rock, position):
        return .5

    def condition_check(self, rock, position, positive):
        return Uninformative()


def test_exact_bounded_family_entropy_ties_and_all_four_tours():
    result = plan_action(Uninformative(), (1, 0), ledger(), 100, 64)
    work = result["work"]
    assert work["vantages_considered"] == work["vantages_feasible"] == 5
    assert work["bundles_evaluated"] == work["bundle_candidates"] == 25
    assert work["outcome_leaves"] == 210
    assert work["maximum_leaves_per_bundle"] == 16
    assert work["tour_evaluations"] == 4 * 211
    assert work["condition_check_calls"] == 370
    assert work["entropy_probability_calls"] == 20
    assert result["kind"] == "exit"
    assert result == plan_action(Uninformative(), (1, 0), ledger(), 100, 64)


def test_duplicate_current_corner_is_evaluated_once():
    result = plan_action(Uninformative(), (0, 0), ledger(), 100, 64)
    assert result["work"]["vantages_considered"] == 4
    assert result["work"]["bundles_evaluated"] == 20


@pytest.mark.parametrize("position,sampled,remaining,checks", [
    ((0, 2), ledger(), 20, 4), ((0., 0.), ledger(), 20, 4),
    ((0, 0), np.zeros((3, 2)), 20, 4), ((0, 0), ledger(), -1, 4),
    ((0, 0), ledger(), True, 4), ((0, 0), ledger(), 20, 65),
])
def test_reject_invalid_public_inputs(position, sampled, remaining, checks):
    with pytest.raises(ValueError):
        plan_action(Uninformative(), position, sampled, remaining, checks)


def test_reject_nonfinite_probability_and_mutating_alias_contract():
    class Invalid(Uninformative):
        def check_probability(self, rock, position):
            return math.nan
    with pytest.raises(ValueError, match="probability"):
        plan_action(Invalid(), (0, 0), ledger(), 20, 4)

    class Aliased(Uninformative):
        def condition_check(self, rock, position, positive):
            return self
    with pytest.raises(ValueError, match="copy"):
        plan_action(Aliased(), (0, 0), ledger(), 20, 4)


def test_positive_mass_branch_errors_are_not_renormalized_away():
    class Broken(Uninformative):
        def condition_check(self, rock, position, positive):
            if not positive:
                raise ValueError("failed branch")
            return Uninformative()
    with pytest.raises(ValueError, match="failed branch"):
        plan_action(Broken(), (0, 0), ledger(), 20, 4)
