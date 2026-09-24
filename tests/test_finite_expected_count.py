"""Fabricated engineering contracts; no task generator, Torch or empirical IO."""
import math

import numpy as np
import pytest

from openjev.research.finite_expected_count import (
    expected_counts,
    forward_backward,
    map_update,
    mle_update,
    tokens_from_prefix,
)


def parameters():
    return (np.full((2, 2, 2), .5, dtype=np.float64),
            np.full((4, 2), .25, dtype=np.float64),
            np.full((2, 2), .25, dtype=np.float64))


def sequence():
    return np.array([0, 1], dtype=np.int64), np.array([0, 1, 4], dtype=np.int64)


def empty_batch():
    return (np.empty((0, 3), dtype=np.int64), np.empty((0, 4), dtype=np.int64),
            np.empty(0, dtype=np.int64))


def public_prefixes():
    p = np.zeros((2, 9, 31), dtype=np.float32)
    sizes = np.array([9, 3], dtype=np.int64)
    for row, size in enumerate(sizes):
        p[row, 0, 4 + row] = p[row, 0, 9] = 1
        for step in range(1, size):
            p[row, step, (row + step) % 4] = 1
            p[row, step, 8 if row == 1 and step == 2 else 4 + step % 4] = 1
    return p, sizes


def test_hand_symmetric_counts_include_reset_but_not_found_emission():
    result = forward_backward(*parameters(), *sequence())
    assert result['log_likelihood'] == pytest.approx(math.log(3 / 256), abs=1e-14)
    np.testing.assert_array_equal(result['scales'], [.25, 3 / 16, .25])
    np.testing.assert_array_equal(result['filtered'], np.full((3, 2), .5))
    np.testing.assert_array_equal(result['smoothed'], np.full((3, 2), .5))
    counts = result['counts']
    np.testing.assert_array_equal(counts['transition'], np.full((2, 2, 2), .25))
    np.testing.assert_array_equal(counts['reset'], [.5, .5])
    np.testing.assert_array_equal(counts['emission'], [[.5, .5], [.5, .5], [0, 0], [0, 0]])
    np.testing.assert_array_equal(counts['survive'], [[.5, .5], [0, 0]])
    np.testing.assert_array_equal(counts['found'], [[0, 0], [.5, .5]])
    assert result['work'] == {'sequences': 1, 'reset_events': 1, 'action_events': 2,
        'ordinary_events': 1, 'found_events': 1, 'forward_matvecs': 2, 'backward_matvecs': 2,
        'transition_posterior_matrices': 2, 'state_posterior_rows': 3, 'check_calls': 11}


def test_batch_has_variable_lengths_no_padding_counts_and_equal_sequence_weights():
    a = np.array([[0, 1], [1, -1], [-1, -1]], dtype=np.int64)
    y = np.array([[0, 1, 4], [3, 4, -1], [2, -1, -1]], dtype=np.int64)
    lengths = np.array([3, 2, 1], dtype=np.int64)
    result = expected_counts(*parameters(), a, y, lengths)
    np.testing.assert_allclose(result['sequence_log_likelihood'],
                               np.log([3 / 256, 1 / 16, 1 / 4]), rtol=0, atol=1e-14)
    assert result['log_likelihood'] == pytest.approx(math.log(3 / 16384), abs=1e-14)
    assert result['counts']['reset'].sum() == 3
    assert result['counts']['transition'].sum() == 3
    assert result['counts']['found'].sum() == 2
    assert result['counts']['emission'].sum() == 4
    np.testing.assert_array_equal(result['scales'], [[.25, 3 / 16, .25], [.25, .25, 0], [.25, 0, 0]])
    assert result['work']['reset_events'] == 3 and result['work']['action_events'] == 3


def test_mle_boundary_solution_and_no_emission_for_absorption():
    old = parameters()
    counts = forward_backward(*old, *sequence())['counts']
    before = {name: value.copy() for name, value in counts.items()}
    result = mle_update(*old, counts)
    np.testing.assert_array_equal(result['transition'], old[0])
    np.testing.assert_array_equal(result['emission'], [[.5, .5], [.5, .5], [0, 0], [0, 0]])
    np.testing.assert_array_equal(result['hazard'], [[0, 0], [1, 1]])
    assert result['pseudocount'] == 0
    for name, values in before.items():
        np.testing.assert_array_equal(counts[name], values)
        np.testing.assert_array_equal(result['raw_counts'][name], values)
        assert not np.shares_memory(result['raw_counts'][name], counts[name])
    new_likelihood = forward_backward(result['transition'], result['emission'], result['hazard'], *sequence())
    assert new_likelihood['log_likelihood'] == pytest.approx(math.log(.25), abs=1e-14)


def test_reset_only_updates_emission_and_retains_exactly_unexposed_dynamics():
    old = parameters()
    old[0][0] = [[.75, .125], [.25, .875]]
    old[2][:] = [[0, 1], [.125, .875]]
    counts = forward_backward(*old, np.empty(0, dtype=np.int64), np.array([3], dtype=np.int64))['counts']
    result = mle_update(*old, counts)
    np.testing.assert_array_equal(result['transition'], old[0])
    np.testing.assert_array_equal(result['hazard'], old[2])
    np.testing.assert_array_equal(result['emission'], [[0, 0], [0, 0], [0, 0], [1, 1]])
    assert result['zero_exposure']['transition'].all() and result['zero_exposure']['hazard'].all()
    assert not result['zero_exposure']['emission'].any()
    assert result['work'] == {'transition_columns_updated': 0, 'emission_columns_updated': 2,
                              'hazard_entries_updated': 0}


def test_empty_counts_mle_retains_old_but_map_uses_prior_only():
    old = parameters()
    old[0][0] = [[.75, .125], [.25, .875]]
    counts = expected_counts(*old, *empty_batch())
    assert counts['log_likelihood'] == 0 and not any(counts['work'].values())
    mle = mle_update(*old, counts['counts'])
    for value, name in zip(old, ('transition', 'emission', 'hazard'), strict=True):
        np.testing.assert_array_equal(mle[name], value)
        assert not np.shares_memory(mle[name], value)
    update = map_update(*old, counts['counts'], pseudocount=.001)
    np.testing.assert_array_equal(update['transition'], np.full((2, 2, 2), .5))
    np.testing.assert_array_equal(update['emission'], np.full((4, 2), .25))
    np.testing.assert_array_equal(update['hazard'], np.full((2, 2), .5))
    assert all(value.all() for value in update['zero_exposure'].values())


def test_map_prior_matches_hand_count_normalization_without_mutation():
    old = parameters()
    counts = forward_backward(*old, *sequence())['counts']
    result = map_update(*old, counts, pseudocount=.125)
    np.testing.assert_array_equal(result['transition'], old[0])
    np.testing.assert_array_equal(result['emission'], [[5 / 12, 5 / 12], [5 / 12, 5 / 12],
                                                     [1 / 12, 1 / 12], [1 / 12, 1 / 12]])
    np.testing.assert_array_equal(result['hazard'], [[1 / 6, 1 / 6], [5 / 6, 5 / 6]])
    assert result['pseudocount'] == .125
    assert counts['emission'][2].sum() == 0


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, np.bool_(False), [1e-3]])
def test_map_rejects_invalid_explicit_prior(value):
    old = parameters()
    counts = expected_counts(*old, *empty_batch())['counts']
    with pytest.raises(ValueError):
        map_update(*old, counts, pseudocount=value)


@pytest.mark.parametrize('corruption', ['dtype', 'negative', 'nan', 'normalization', 'hazard', 'shape'])
def test_model_contract_rejects_invalid_arrays(corruption):
    t, o, h = parameters()
    if corruption == 'dtype':
        t = t.astype(np.float32)
    elif corruption == 'negative':
        o[0, 0] = -.1
    elif corruption == 'nan':
        h[0, 0] = np.nan
    elif corruption == 'normalization':
        t[0, 0, 0] = .75
    elif corruption == 'hazard':
        h[0, 0] = 1.01
    else:
        o = o[:3]
    with pytest.raises(ValueError):
        forward_backward(t, o, h, *sequence())


@pytest.mark.parametrize('corruption', ['negative', 'missing', 'shape', 'flow', 'reset'])
def test_m_step_rejects_invalid_expected_counts(corruption):
    old = parameters()
    counts = forward_backward(*old, *sequence())['counts']
    if corruption == 'negative':
        counts['transition'][0, 0, 0] = -.25
    elif corruption == 'missing':
        del counts['reset']
    elif corruption == 'shape':
        counts['reset'] = counts['reset'][:1]
    elif corruption == 'flow':
        counts['found'][1, 0] += 1
    else:
        counts['reset'][0] += 1
    with pytest.raises(ValueError):
        mle_update(*old, counts)


def test_batch_preflight_rejects_late_malformed_row_before_any_callback():
    callbacks = []
    a = np.array([[0, 1], [1, -1]], dtype=np.int64)
    y = np.array([[0, 1, 4], [3, 4, 0]], dtype=np.int64)  # Non-sentinel padding.
    with pytest.raises(ValueError, match='padding'):
        expected_counts(*parameters(), a, y, np.array([3, 2], dtype=np.int64),
                        check=lambda: callbacks.append(1))
    assert callbacks == []


def test_callback_failure_propagates_without_mutating_parameters_or_tokens():
    params, tokens = parameters(), sequence()
    before = [array.copy() for array in (*params, *tokens)]
    counter = 0

    class Deadline(Exception):
        pass

    def check():
        nonlocal counter
        counter += 1
        if counter == 4:
            raise Deadline('fabricated bound')

    with pytest.raises(Deadline):
        forward_backward(*params, *tokens, check=check)
    for value, snapshot in zip((*params, *tokens), before, strict=True):
        np.testing.assert_array_equal(value, snapshot)


def test_public_tokens_preserve_valid_labels_and_own_all_returned_arrays():
    prefix, lengths = public_prefixes()
    result = tokens_from_prefix(prefix, lengths)
    np.testing.assert_array_equal(result['actions'][0], [1, 2, 3, 0, 1, 2, 3, 0])
    np.testing.assert_array_equal(result['observations'][0], [0, 1, 2, 3, 0, 1, 2, 3, 0])
    np.testing.assert_array_equal(result['actions'][1], [2, 3, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(result['observations'][1], [1, 1, 4, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(result['event_mask'].sum(axis=1), lengths)
    result['lengths'][0] = 2
    result['actions'][0, 0] = 0
    assert lengths[0] == 9 and prefix[0, 1, 1] == 1


@pytest.mark.parametrize('corruption', ['padding', 'future_found', 'short_without_found',
                                      'reset_found', 'reset_action', 'nonbinary', 'hidden_feature', 'nan'])
def test_public_grammar_rejects_nonpublic_or_noncausal_rows(corruption):
    p, sizes = public_prefixes()
    if corruption == 'padding':
        p[1, 8, 0] = 1
    elif corruption == 'future_found':
        p[0, 1, 4:9] = 0
        p[0, 1, 8] = 1
    elif corruption == 'short_without_found':
        p[1, 2, 8], p[1, 2, 4] = 0, 1
    elif corruption == 'reset_found':
        p[0, 0, 4], p[0, 0, 8] = 0, 1
    elif corruption == 'reset_action':
        p[0, 0, 0] = 1
    elif corruption == 'nonbinary':
        p[0, 1, 1] = .5
    elif corruption == 'hidden_feature':
        p[0, 1, 10] = 1
    else:
        p[1, 8, 30] = np.nan
    with pytest.raises(ValueError):
        tokens_from_prefix(p, sizes)
