"""Exhaustive two-state expected-count oracles, without model/data generation.

Every hidden path is enumerated independently of the forward/backward learner.
These engineering fixtures use only deterministic rational constants.
"""
from __future__ import annotations

import copy
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from openjev.research import finite_expected_count as module


def parameters():
    # T[action,next,current] is deliberately asymmetric and column stochastic.
    transition = np.asarray([[[.8, .3], [.2, .7]], [[.1, .6], [.9, .4]]], np.float64)
    emission = np.asarray([[.5, .1], [.2, .2], [.2, .3], [.1, .4]], np.float64)
    hazard = np.asarray([[.05, .15], [.25, .1]], np.float64)
    return transition, emission, hazard


def fraction(value):
    return Fraction(str(float(value)))


def enumerate_paths(transition, emission, hazard, actions, observations):
    """Exact rational sum over z0,...,zK; no production arithmetic helpers."""
    k, states = len(actions), transition.shape[1]
    weights = []
    for path in itertools.product(range(states), repeat=k + 1):
        probability = Fraction(1, states) * fraction(emission[int(observations[0]), path[0]])
        for step, action in enumerate(actions):
            old, new = path[step], path[step + 1]
            probability *= fraction(transition[int(action), new, old])
            risk = fraction(hazard[int(action), new])
            if int(observations[step + 1]) == 4:
                probability *= risk
            else:
                probability *= (1 - risk) * fraction(emission[int(observations[step + 1]), new])
        weights.append((path, probability))
    evidence = sum((weight for _, weight in weights), Fraction(0))
    assert evidence > 0
    counts = {'transition': np.zeros_like(transition), 'emission': np.zeros_like(emission),
              'survive': np.zeros_like(hazard), 'found': np.zeros_like(hazard),
              'reset': np.zeros(states, np.float64)}
    smoothed = np.zeros((k + 1, states), np.float64)
    joints = np.zeros((k, states, states), np.float64)
    for path, weight in weights:
        posterior = float(weight / evidence)
        counts['reset'][path[0]] += posterior
        counts['emission'][int(observations[0]), path[0]] += posterior
        for step, state in enumerate(path):
            smoothed[step, state] += posterior
        for step, action in enumerate(actions):
            old, new = path[step], path[step + 1]
            counts['transition'][int(action), new, old] += posterior
            joints[step, new, old] += posterior
            if int(observations[step + 1]) == 4:
                counts['found'][int(action), new] += posterior
            else:
                counts['survive'][int(action), new] += posterior
                counts['emission'][int(observations[step + 1]), new] += posterior
    return {'likelihood': evidence, 'log_likelihood': math.log(float(evidence)),
            'smoothed': smoothed, 'transition_posteriors': joints, 'counts': counts}


def oracle(transition, emission, hazard, actions, observations):
    result = enumerate_paths(transition, emission, hazard, actions, observations)
    filtered, scales = [], []
    preceding = Fraction(1)
    for stop in range(len(actions) + 1):
        prefix = enumerate_paths(transition, emission, hazard, actions[:stop], observations[:stop + 1])
        filtered.append(prefix['smoothed'][-1])
        scales.append(float(prefix['likelihood'] / preceding))
        preceding = prefix['likelihood']
    return {**result, 'filtered': np.asarray(filtered), 'scales': np.asarray(scales)}


def assert_counts(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].dtype == np.float64 and actual[key].shape == expected[key].shape
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=2e-14, err_msg=key)


@pytest.mark.parametrize(('actions', 'observations'), [
    ([], [0]),
    ([0], [0, 1]),
    ([1], [2, 4]),
    ([0, 1], [0, 2, 3]),
    ([1, 0], [3, 1, 4]),
    ([0, 1, 0], [0, 1, 2, 3]),
    ([1, 0, 1], [2, 0, 3, 4]),
])
def test_all_hidden_paths_match_likelihood_posteriors_and_every_expected_count(actions, observations):
    transition, emission, hazard = parameters()
    actions, observations = np.asarray(actions, np.int64), np.asarray(observations, np.int64)
    expected = oracle(transition, emission, hazard, actions, observations)
    actual = module.forward_backward(transition, emission, hazard, actions, observations)
    assert actual['log_likelihood'] == pytest.approx(expected['log_likelihood'], abs=2e-13)
    assert actual['terminal_found'] is bool(observations[-1] == 4)
    for key in ('scales', 'filtered', 'smoothed', 'transition_posteriors'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=2e-14, err_msg=key)
    assert_counts(actual['counts'], expected['counts'])
    assert sum(math.log(float(value)) for value in actual['scales']) == pytest.approx(expected['log_likelihood'], abs=2e-13)
    found = int(observations[-1] == 4)
    assert actual['counts']['transition'].sum() == pytest.approx(len(actions))
    assert actual['counts']['emission'].sum() == pytest.approx(len(actions) + 1 - found)
    assert actual['counts']['survive'].sum() == pytest.approx(len(actions) - found)
    assert actual['counts']['found'].sum() == pytest.approx(found)
    assert actual['counts']['reset'].sum() == pytest.approx(1.)


def test_filter_is_causal_while_smoother_uses_future_labels():
    transition, emission, hazard = parameters()
    actions = np.asarray([0, 1, 0], np.int64)
    left = module.forward_backward(transition, emission, hazard, actions, np.asarray([0, 1, 2, 0], np.int64))
    right = module.forward_backward(transition, emission, hazard, actions, np.asarray([0, 1, 2, 3], np.int64))
    np.testing.assert_array_equal(left['filtered'][:3], right['filtered'][:3])
    np.testing.assert_array_equal(left['scales'][:3], right['scales'][:3])
    assert np.max(np.abs(left['smoothed'][0] - right['smoothed'][0])) > 1e-5
    prefix = module.forward_backward(transition, emission, hazard, actions[:1], np.asarray([0, 1], np.int64))
    np.testing.assert_array_equal(prefix['filtered'], left['filtered'][:2])


def test_terminal_destination_has_hazard_and_transition_counts_but_no_emission():
    transition, emission, hazard = parameters()
    actions, observations = np.asarray([1], np.int64), np.asarray([0, 4], np.int64)
    result = module.forward_backward(transition, emission, hazard, actions, observations)
    expected = oracle(transition, emission, hazard, actions, observations)
    assert_counts(result['counts'], expected['counts'])
    np.testing.assert_array_equal(result['counts']['emission'][1:], np.zeros((3, 2)))
    np.testing.assert_allclose(result['counts']['found'][1], result['smoothed'][-1], atol=2e-14, rtol=2e-13)
    assert result['counts']['survive'].sum() == 0.
    assert result['filtered'][-1].sum() == pytest.approx(1.)  # Diagnostic pre-absorption hidden destination.


def sequences():
    return [(np.asarray(a, np.int64), np.asarray(o, np.int64)) for a, o in (
        ([0, 1, 0], [0, 1, 2, 4]), ([1, 0], [3, 2, 4]),
        ([0, 1], [1, 2, 0]), ([], [2]))]


def aggregate_oracle(transition, emission, hazard):
    records = [enumerate_paths(transition, emission, hazard, a, o) for a, o in sequences()]
    counts = {key: sum((record['counts'][key] for record in records), np.zeros_like(records[0]['counts'][key]))
              for key in records[0]['counts']}
    return counts, sum(record['log_likelihood'] for record in records)


def penalized_likelihood(transition, emission, hazard, pseudocount):
    _, likelihood = aggregate_oracle(transition, emission, hazard)
    probabilities = itertools.chain(transition.ravel(), emission.ravel(), hazard.ravel(), (1 - hazard).ravel())
    return likelihood + pseudocount * math.fsum(math.log(float(value)) for value in probabilities)


def test_mle_update_matches_hand_normalization_and_nondecreasing_observed_likelihood():
    transition, emission, hazard = parameters()
    counts, before = aggregate_oracle(transition, emission, hazard)
    retained = {key: value.copy() for key, value in counts.items()}
    updated = module.mle_update(transition, emission, hazard, counts)
    np.testing.assert_allclose(updated['transition'], counts['transition'] / counts['transition'].sum(axis=1, keepdims=True), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(updated['emission'], counts['emission'] / counts['emission'].sum(axis=0, keepdims=True), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(updated['hazard'], counts['found'] / (counts['found'] + counts['survive']), rtol=2e-13, atol=2e-14)
    _, after = aggregate_oracle(updated['transition'], updated['emission'], updated['hazard'])
    assert after >= before - 1e-12
    assert_counts(counts, retained)
    assert_counts(updated['raw_counts'], retained)


def test_map_adds_pseudocount_to_each_categorical_entry_and_both_hazard_outcomes():
    transition, emission, hazard = parameters()
    pc = 1e-3
    for _ in range(3):
        counts, _ = aggregate_oracle(transition, emission, hazard)
        before = penalized_likelihood(transition, emission, hazard, pc)
        updated = module.map_update(transition, emission, hazard, counts, pseudocount=pc)
        t = counts['transition'] + pc
        o = counts['emission'] + pc
        np.testing.assert_allclose(updated['transition'], t / t.sum(axis=1, keepdims=True), rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(updated['emission'], o / o.sum(axis=0, keepdims=True), rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(updated['hazard'], (counts['found'] + pc) / (counts['found'] + counts['survive'] + 2 * pc), rtol=2e-13, atol=2e-14)
        transition, emission, hazard = updated['transition'], updated['emission'], updated['hazard']
        assert penalized_likelihood(transition, emission, hazard, pc) >= before - 1e-12
        assert updated['pseudocount'] == pc


def padded_sequences():
    cases = sequences()
    actions = np.full((len(cases), 3), -1, np.int64)
    observations = np.full((len(cases), 4), -1, np.int64)
    lengths = np.asarray([len(o) for _, o in cases], np.int64)
    for i, (a, o) in enumerate(cases):
        actions[i, :len(a)], observations[i, :len(o)] = a, o
    return actions, observations, lengths


def test_padded_batch_equals_independent_variable_length_paths_with_zero_inactive_scales():
    transition, emission, hazard = parameters()
    actions, observations, lengths = padded_sequences()
    result = module.expected_counts(transition, emission, hazard, actions, observations, lengths)
    expected_counts, expected_ll = aggregate_oracle(transition, emission, hazard)
    assert_counts(result['counts'], expected_counts)
    assert result['log_likelihood'] == pytest.approx(expected_ll, abs=3e-13)
    for i, (a, o) in enumerate(sequences()):
        independent = oracle(transition, emission, hazard, a, o)
        assert result['sequence_log_likelihood'][i] == pytest.approx(independent['log_likelihood'], abs=2e-13)
        np.testing.assert_allclose(result['scales'][i, :lengths[i]], independent['scales'], rtol=2e-13, atol=2e-14)
        np.testing.assert_array_equal(result['scales'][i, lengths[i]:], np.zeros(4 - lengths[i]))


@pytest.mark.parametrize('field', ['actions', 'observations', 'lengths'])
def test_padding_cannot_silently_contribute_or_hide_events(field):
    transition, emission, hazard = parameters()
    actions, observations, lengths = padded_sequences()
    if field == 'actions':
        actions[-1, 0] = 0
    elif field == 'observations':
        observations[-1, 1] = 0
    else:
        lengths[-1] = 0
    with pytest.raises(ValueError):
        module.expected_counts(transition, emission, hazard, actions, observations, lengths)


@pytest.mark.parametrize(('actions', 'observations'), [([0], [4, 0]), ([0, 1], [0, 4, 1]), ([0], [0]), ([2], [0, 1]), ([-1], [0, 1])])
def test_invalid_reset_terminal_order_lengths_and_actions_fail(actions, observations):
    transition, emission, hazard = parameters()
    with pytest.raises(ValueError):
        module.forward_backward(transition, emission, hazard, np.asarray(actions, np.int64), np.asarray(observations, np.int64))


def test_impossible_event_fails_without_mutating_parameters():
    transition, emission, hazard = parameters()
    emission[0] += emission[3]
    emission[3] = 0.
    before = copy.deepcopy((transition, emission, hazard))
    with pytest.raises(ValueError):
        module.forward_backward(transition, emission, hazard, np.asarray([0], np.int64), np.asarray([0, 3], np.int64))
    for current, saved in zip((transition, emission, hazard), before, strict=True):
        np.testing.assert_array_equal(current, saved)


def test_reset_only_sequence_has_emission_count_and_no_transition_or_hazard_count():
    transition, emission, hazard = parameters()
    result = module.forward_backward(transition, emission, hazard, np.empty(0, np.int64), np.asarray([0], np.int64))
    np.testing.assert_allclose(result['counts']['reset'], [5 / 6, 1 / 6], atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(result['counts']['emission'][0], [5 / 6, 1 / 6], atol=2e-14, rtol=2e-13)
    assert result['counts']['transition'].sum() == result['counts']['survive'].sum() == result['counts']['found'].sum() == 0.
