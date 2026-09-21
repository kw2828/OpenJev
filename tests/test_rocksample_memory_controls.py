"""Synthetic completed public histories only; no environment calls or imports."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from openjev.research.rocksample_memory_controls import PublicTransition, apply_transition, reconstruct
from openjev.research.rocksample_particle_belief import ParticleRockBelief
from openjev.research.rocksample_quality_belief import QualityRockBelief


def prior(quality=0.5):
    return ParticleRockBelief.from_hypotheses([[[0, 0], [2, 0]]], quality,
                                             size=3, half_efficiency_distance=1)


def move():
    return PublicTransition((0, 1), 0)


def direct_replay(belief, events):
    result = belief.copy()
    result.reset()
    for event in events:
        if event.action == 4:
            result.sample(event.position)
        elif event.action >= 5:
            result = result.condition_check(event.action - 5, event.position, event.reading == 1)
    return result


def assert_same_belief(left, right):
    np.testing.assert_allclose(left.expected_rewards(), right.expected_rewards(), atol=1e-14)
    np.testing.assert_allclose(left.quality_probabilities(), right.quality_probabilities(), atol=1e-14)
    if hasattr(left, "map_probabilities"):
        np.testing.assert_allclose(left.map_probabilities, right.map_probabilities, atol=1e-14)


def test_history_schema_has_only_immutable_public_fields():
    assert [field.name for field in fields(PublicTransition)] == ["position", "action", "reading"]
    position = [np.int64(0), np.int64(1)]
    event = PublicTransition(position, np.int64(5), np.int64(1))
    position[0] = 2
    assert event.position == (0, 1) and type(event.action) is int and type(event.reading) is int
    with pytest.raises(FrozenInstanceError):
        event.action = 4
    with pytest.raises(TypeError):
        PublicTransition((0, 0), 4, reward=10)


@pytest.mark.parametrize("mode", ["recent128", "latest", "full"])
def test_depletion_older_than_window_is_retained_before_new_check(mode):
    belief = prior(quality=1)
    history = [PublicTransition((0, 0), 4)] + [move() for _ in range(128)]
    history.append(PublicTransition((0, 0), 5, -1))
    # Without the first SAMPLE, this perfect negative reading would be impossible.
    reconstructed = reconstruct(belief, history, mode)
    assert reconstructed.quality_probabilities()[0] == 0
    assert reconstructed.quality_probabilities()[1] == 1
    assert_same_belief(reconstructed, direct_replay(belief, history))
    assert belief.quality_probabilities()[0] == 1


@pytest.mark.parametrize("mode", ["recent128", "latest", "full"])
def test_future_sample_is_not_preapplied_before_an_earlier_positive_check(mode):
    belief = prior()
    history = [PublicTransition((0, 0), 5, 1), PublicTransition((0, 0), 4)]
    # Preapplying the final sample ledger would make the retained positive impossible.
    reconstructed = reconstruct(belief, history, mode)
    assert reconstructed.quality_probabilities()[0] == 0
    assert reconstructed.expected_rewards()[0, 0] == -10
    assert_same_belief(reconstructed, direct_replay(belief, history))


@pytest.mark.parametrize("length,included", [(1, True), (127, True), (128, True), (129, False), (256, False)])
def test_recent128_counts_all_primitive_actions_and_includes_exact_boundary(length, included):
    belief = prior()
    history = [PublicTransition((0, 1), 5, 1)] + [move() for _ in range(length - 1)]
    reconstructed = reconstruct(belief, history)
    assert reconstructed.quality_probabilities()[0] == pytest.approx(0.75 if included else 0.5)
    assert reconstructed.quality_probabilities()[1] == pytest.approx(0.5)
    assert reconstruct(belief, history, "full").quality_probabilities()[0] == pytest.approx(0.75)


def test_latest_retains_one_check_for_each_rock_in_original_chronology():
    belief = prior()
    history = [PublicTransition((0, 1), 5, 1), PublicTransition((0, 1), 6, -1),
               PublicTransition((0, 1), 5, -1), move(), PublicTransition((0, 1), 6, 1)]
    actual = reconstruct(belief, history, "latest")
    expected = direct_replay(belief, [history[2], history[4]])
    assert_same_belief(actual, expected)
    assert actual.quality_probabilities()[0] < 0.5
    assert actual.quality_probabilities()[1] > 0.5
    np.testing.assert_allclose(reconstruct(belief, history, "full").quality_probabilities(), 0.5)


def test_latest_keeps_samples_before_between_and_after_retained_checks():
    belief = ParticleRockBelief.from_hypotheses([[[0, 0]], [[1, 0]]], size=3,
                                               half_efficiency_distance=1)
    history = [PublicTransition((0, 0), 4), PublicTransition((0, 1), 5, -1),
               PublicTransition((1, 0), 4), PublicTransition((0, 1), 5, -1),
               PublicTransition((2, 0), 4)]
    actual = reconstruct(belief, history, "latest")
    expected = direct_replay(belief, [history[0], history[2], history[3], history[4]])
    assert_same_belief(actual, expected)
    assert actual.quality_probabilities()[0] == 0


@pytest.mark.parametrize("factory", [prior, lambda: QualityRockBelief(size=3, rocks=2,
                                                                     half_efficiency_distance=1)])
def test_full_matches_incremental_public_updates_for_both_belief_families(factory):
    belief = factory()
    history = [PublicTransition((0, 1), 5, 1), PublicTransition((0, 0), 4),
               PublicTransition((0, 1), 6, -1), move(), PublicTransition((0, 1), 5, -1)]
    incremental = belief.copy()
    for event in history:
        incremental = apply_transition(incremental, event)
    assert_same_belief(reconstruct(belief, history, "full"), incremental)


def test_reconstruction_resets_a_copy_and_preserves_input_prior_and_history():
    belief = prior().condition_check(0, (0, 1), True)
    belief.sample((2, 0))
    state_before = (belief._log_weights.copy(), belief._log_quality.copy())
    history = [PublicTransition((0, 1), 6, 1), move()]
    history_before = copy.deepcopy(history)
    actual = reconstruct(belief, history, "full")
    assert_same_belief(actual, direct_replay(prior(), history))
    np.testing.assert_array_equal(belief._log_weights, state_before[0])
    np.testing.assert_array_equal(belief._log_quality, state_before[1])
    assert history == history_before
    actual.sample((0, 0))
    np.testing.assert_array_equal(belief._log_quality, state_before[1])


def test_apply_transition_ownership_and_signed_reading_conversion():
    belief = prior()
    before = belief._log_quality.copy()
    assert apply_transition(belief, move()) is belief
    checked = apply_transition(belief, PublicTransition((0, 1), 5, -1))
    assert checked is not belief
    assert checked.quality_probabilities()[0] == pytest.approx(0.25)
    np.testing.assert_array_equal(belief._log_quality, before)
    assert apply_transition(checked, PublicTransition((0, 0), 4)) is checked
    assert checked.quality_probabilities()[0] == 0


@pytest.mark.parametrize("mode", ["recent128", "latest", "full"])
def test_empty_history_returns_independent_reset_prior(mode):
    initial = prior()
    changed = initial.condition_check(0, (0, 1), True)
    actual = reconstruct(changed, [], mode)
    assert actual is not initial and actual is not changed
    assert_same_belief(actual, initial)


@pytest.mark.parametrize("kwargs", [{"position": None, "action": 0}, {"position": (0,), "action": 0},
                                    {"position": (-1, 0), "action": 0}, {"position": (True, 0), "action": 0},
                                    {"position": (0.0, 0), "action": 0}, {"position": (0, 0), "action": True},
                                    {"position": (0, 0), "action": -1}, {"position": (0, 0), "action": 5},
                                    {"position": (0, 0), "action": 5, "reading": True},
                                    {"position": (0, 0), "action": 5, "reading": 0},
                                    {"position": (0, 0), "action": 5, "reading": 0.5},
                                    {"position": (0, 0), "action": 4, "reading": 1},
                                    {"position": (0, 0), "action": 0, "reading": -1}])
def test_invalid_transition_schema_rejected(kwargs):
    with pytest.raises(ValueError):
        PublicTransition(**kwargs)


@pytest.mark.parametrize("event", [PublicTransition((3, 0), 0), PublicTransition((0, 2), 4),
                                   PublicTransition((0, 0), 7, 1)])
def test_even_discarded_out_of_bounds_evidence_is_validated(event):
    belief = prior()
    before = belief._log_quality.copy()
    history = [event] + [move() for _ in range(129)]
    with pytest.raises(ValueError):
        reconstruct(belief, history)
    np.testing.assert_array_equal(belief._log_quality, before)


def test_invalid_mode_and_nonpublic_history_entries_fail():
    with pytest.raises(ValueError, match="mode"):
        reconstruct(prior(), [], "recent32")
    with pytest.raises(TypeError, match="PublicTransition"):
        reconstruct(prior(), [{"position": (0, 0), "action": 4, "reward": 10}])
