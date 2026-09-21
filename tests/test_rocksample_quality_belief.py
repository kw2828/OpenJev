import numpy as np
import pytest

from openjev.research.rocksample_quality_belief import QualityRockBelief


def test_uniform_quality_prior_and_zero_sampling_value():
    b = QualityRockBelief()
    assert np.array_equal(b.quality_probabilities(), np.full(11, .5))
    assert np.array_equal(b.expected_rewards(), np.zeros((11, 10)))
    assert b.check_probability(0, (0, 0)) == .5


def test_tiny_initial_check_matches_direct_enumerated_marginal():
    b = QualityRockBelief(size=2, rocks=1)
    correctness = (1 + 2 ** (-1 / 20)) / 2
    prior_probability = .5
    expected_good_posterior = (.25 * 1 + .25 * correctness) / prior_probability
    after = b.condition_check(0, (0, 0), True)
    assert after.quality_probabilities()[0] == pytest.approx(expected_good_posterior)
    assert b.quality_probabilities()[0] == .5
    assert after.expected_rewards()[0, 0] == pytest.approx(5 * (2 * expected_good_posterior - 1))
    assert after.expected_rewards()[1, 0] == pytest.approx(after.expected_rewards()[0, 0])


def test_depletion_changes_likelihood_without_observing_reward():
    b = QualityRockBelief(size=2, rocks=1)
    b.sample((0, 0))
    correctness = (1 + 2 ** (-1 / 20)) / 2
    # At the sampled location every initial quality is now bad. Only the other
    # half of the location prior can emit a positive signal with probability .5.
    assert b.check_probability(0, (0, 0)) == pytest.approx(.25)
    after = b.condition_check(0, (0, 0), True)
    assert after.quality_probabilities()[0] == pytest.approx(.5 * correctness)
    assert b.expected_rewards()[0, 0] == -5
    assert b.expected_rewards()[1, 0] == 0


def test_complete_depletion_prevents_quality_learning_from_readings():
    b = QualityRockBelief(size=2, rocks=1).condition_check(0, (0, 0), True)
    b.sample((0, 0)); b.sample((1, 0))
    after = b.condition_check(0, (0, 0), True)
    np.testing.assert_allclose(after._log_quality, b._log_quality, atol=1e-15)
    assert np.array_equal(after.quality_probabilities(), [0.0])
    assert np.array_equal(after.expected_rewards(), np.full((2, 1), -5.0))


def test_copies_and_reset_preserve_independence():
    b = QualityRockBelief(size=2, rocks=1)
    clone = b.copy()
    clone.sample((0, 0))
    assert b.diagnostics()["sampled_cells"] == 0
    assert clone.diagnostics()["sampled_cells"] == 1
    clone.reset()
    assert np.array_equal(clone.expected_rewards(), b.expected_rewards())


@pytest.mark.parametrize("position", [(-1, 0), (0, 10), (11, 0), (0.1, 1), (np.nan, 0), (0, 0, 0)])
def test_reject_invalid_public_position(position):
    b = QualityRockBelief()
    with pytest.raises(ValueError):
        b.sample(position)


@pytest.mark.parametrize("rock", [-1, 11, True, .2])
def test_reject_invalid_rock(rock):
    with pytest.raises(ValueError):
        QualityRockBelief().check_probability(rock, (0, 0))
