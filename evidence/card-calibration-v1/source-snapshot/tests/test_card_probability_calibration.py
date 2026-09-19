"""Hand-computable probability arrays only; no RNG, model, or native calls."""
import copy
import json
import math

import numpy as np
import pytest

from openjev.research import card_probability_calibration as calibration


def episode(steps=5):
    return {'raw_probabilities': np.full((steps, 52, 13), 1 / 13, dtype=np.float64),
            'targets': np.full((steps, 52), -1, dtype=np.int64),
            'target_mask': np.zeros((steps, 52), dtype=np.bool_),
            'ages': np.full((steps, 52), -1, dtype=np.int32)}


def query(ep, step, position, *, probability=.5, target=0, age=1):
    ep['raw_probabilities'][step, position] = (1 - probability) / 12
    ep['raw_probabilities'][step, position, target] = probability
    ep['target_mask'][step, position] = True
    ep['targets'][step, position] = target
    ep['ages'][step, position] = age


def binary_episode(fraction_correct):
    ep = episode(steps=2)
    for pos in range(10):
        ep['raw_probabilities'][1, pos] = 0
        ep['raw_probabilities'][1, pos, :2] = [.8, .2]
        ep['targets'][1, pos] = 0 if pos < fraction_correct * 10 else 1
        ep['target_mask'][1, pos] = True
        ep['ages'][1, pos] = 1
    return ep


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_identity_is_exact_raw_copy_with_mass_drift_and_no_alias(dtype):
    raw = np.full((52, 13), 1 / 13, dtype=dtype)
    raw[0, 0] += dtype(1e-7)
    before = raw.copy()
    for mode in ('baseline', 'temperature'):
        actual = calibration.transform_probabilities(raw, mode, 1.0)
        np.testing.assert_array_equal(actual, raw.astype(np.float64))
        assert not actual.flags.writeable and not np.shares_memory(actual, raw)
    np.testing.assert_array_equal(raw, before)


def test_power_transform_handles_zeros_and_hard_uses_lowest_exact_argmax():
    raw = np.zeros((52, 13), dtype=np.float64)
    raw[:, :3] = [.4, .4, .2]
    actual = calibration.transform_probabilities(raw, 'temperature', 2.0)
    np.testing.assert_allclose(actual[:, :3], np.tile([4 / 9, 4 / 9, 1 / 9], (52, 1)), atol=1e-15)
    assert not actual[:, 3:].any()
    hard = calibration.transform_probabilities(raw, 'hard')
    assert np.array_equal(hard[:, 0], np.ones(52)) and not hard[:, 1:].any()
    np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-15)


def test_power_is_invariant_to_near_unit_row_scaling_and_preserves_argmax():
    raw = np.zeros((3, 52, 13), dtype=np.float64)
    raw[..., :2] = [.8, .2]
    scaled = raw * (1 + 1e-7)
    a = calibration.transform_probabilities(raw, 'temperature', .05)
    b = calibration.transform_probabilities(scaled, 'temperature', .05)
    np.testing.assert_allclose(a, b, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(a.argmax(-1), raw.argmax(-1))


def test_fit_has_analytic_interior_solution_without_opaque_optimizer():
    fit = calibration.fit_temperature([binary_episode(.6)])
    expected = math.log(.6 / .4) / math.log(.8 / .2)
    assert fit['beta'] == pytest.approx(expected, abs=2e-15)
    assert fit['temperature'] == pytest.approx(1 / expected)
    assert fit['oracle_evaluations'] == 68 and len(fit['trace']) == 68
    assert [row['point'] for row in fit['trace'][:3]] == ['lower', 'upper', 'identity']
    assert fit['trace'][-1]['point'] == 'final'
    assert fit['calibrated_nll'] == pytest.approx(-.6 * math.log(.6) - .4 * math.log(.4))
    assert fit['calibrated_nll'] < fit['baseline_nll'] and not fit['at_bound']


@pytest.mark.parametrize('correct,beta', [(1.0, 20.0), (0.0, .05)])
def test_convex_endpoint_optima_and_exact_pass_count(correct, beta):
    fit = calibration.fit_temperature([binary_episode(correct)])
    assert fit['beta'] == beta and fit['at_bound'] and fit['oracle_evaluations'] == 3


def test_uniform_flat_loss_keeps_identity_exactly():
    ep = episode()
    query(ep, 3, 0, probability=1 / 13)
    fit = calibration.fit_temperature([ep])
    assert fit['beta'] == fit['temperature'] == 1
    assert fit['calibrated_nll'] == fit['baseline_nll']
    assert fit['oracle_evaluations'] == 3


def test_original_softmax_zero_at_target_rejects_fit_without_smoothing():
    ep = binary_episode(.6)
    ep['raw_probabilities'][1, 0, 0] = 0
    ep['raw_probabilities'][1, 0, 1] = 1
    with pytest.raises(ValueError, match='infinite NLL.*no floor'):
        calibration.fit_temperature([ep])


def test_hand_computable_episode_boundary_query_weighting():
    a, b, empty = episode(), episode(), episode()
    query(a, 2, 0, probability=.8)
    query(a, 3, 0, probability=.4)
    query(a, 3, 1, probability=.4)
    for pos in range(10):
        query(b, 3, pos, probability=.2)
    result = calibration.score_episodes([a, b, empty])
    nll = ((-math.log(.8) - math.log(.4)) / 2 - math.log(.2)) / 2
    assert result['all']['nll'] == pytest.approx(nll, abs=1e-15)
    assert result['all']['eligible_episodes'] == 2
    assert result['all']['query_boundaries'] == 3 and result['all']['query_cards'] == 13
    assert result['per_episode'][2]['all']['nll'] is None
    fit = calibration.fit_temperature([a, b, empty])
    assert fit['baseline_nll'] == pytest.approx(nll, abs=1e-15)
    assert fit['counts'] == {'episodes': 3, 'eligible_episodes': 2, 'query_boundaries': 3, 'query_cards': 13}
    pooled = (-math.log(.8) - 2 * math.log(.4) - 10 * math.log(.2)) / 13
    assert abs(result['all']['nll'] - pooled) > .1


def test_hard_mistake_nll_is_infinite_brier_is_two_not_smoothed_or_dropped():
    ep = binary_episode(.6)
    result = calibration.score_episodes([ep], 'hard')['all']
    assert result['nll'] is None and result['nll_is_infinite'] is True
    assert result['infinite_nll_queries'] == 4 and result['accuracy'] == .6
    assert result['brier'] == pytest.approx(.8)
    json.dumps(result, allow_nan=False)


def test_hard_correct_nll_zero_and_actual_zero_soft_target_infinite():
    ep = binary_episode(1.0)
    result = calibration.score_episodes([ep], 'hard')['all']
    assert result['nll'] == result['brier'] == 0 and result['accuracy'] == 1
    ep['raw_probabilities'][1, 0, :2] = [0, 1]
    for mode, beta in (('baseline', 1), ('temperature', .5)):
        scores = calibration.score_episodes([ep], mode, beta)['all']
        assert scores['nll'] is None and scores['nll_is_infinite']
        assert scores['infinite_nll_queries'] == 1


def test_temperature_log_nll_stays_finite_when_exposed_probability_underflows():
    ep = episode(2)
    query(ep, 1, 0, probability=1e-200)
    result = calibration.score_episodes([ep], 'temperature', 20)['all']
    assert result['nll'] > 9000 and math.isfinite(result['nll'])
    assert not result['nll_is_infinite'] and result['infinite_nll_queries'] == 0
    assert result['underflowed_target_probabilities'] == 1
    assert calibration.transform_probabilities(ep['raw_probabilities'], 'temperature', 20)[1, 0, 0] == 0


def test_age_stratum_uses_its_own_nonempty_episode_and_boundary_denominators():
    a, b = episode(35), episode(35)
    query(a, 33, 0, probability=.8, age=33)
    query(a, 34, 0, probability=.2, age=1)
    query(b, 34, 0, probability=.4, age=1)
    scores = calibration.score_episodes([a, b])
    assert scores['age_gt32']['eligible_episodes'] == 1
    assert scores['age_gt32']['nll'] == pytest.approx(-math.log(.8))
    assert scores['all']['eligible_episodes'] == 2
    assert scores['per_episode'][1]['age_gt32']['nll'] is None
    assert not scores['per_episode'][1]['age_gt32']['nll_is_infinite']


def test_no_target_mask_rejection_for_fit_but_descriptive_empty_metrics():
    values = [episode()]
    with pytest.raises(ValueError, match='No eligible'):
        calibration.fit_temperature(values)
    result = calibration.score_episodes(values)['all']
    assert result['eligible_episodes'] == 0 and result['nll'] is result['brier'] is result['accuracy'] is None
    assert not result['nll_is_infinite']


def test_no_input_mutation_or_target_leak_into_transform():
    values = [binary_episode(.6)]
    original = copy.deepcopy(values)
    fit = calibration.fit_temperature(values)
    calibration.score_episodes(values, 'temperature', fit['beta'])
    for before, after in zip(original, values, strict=True):
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
    first = calibration.transform_probabilities(values[0]['raw_probabilities'], 'hard')
    values[0]['targets'][values[0]['target_mask']] = 7
    second = calibration.transform_probabilities(values[0]['raw_probabilities'], 'hard')
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('beta', [True, 0, -.1, .04999, 20.01, math.inf, math.nan, '1', np.float32(1)])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match='beta'):
        calibration.transform_probabilities(episode()['raw_probabilities'], 'temperature', beta)


@pytest.mark.parametrize('kind', ['nan', 'negative', 'zero_row', 'bad_mass', 'integer', 'shape'])
def test_invalid_raw_probability_arrays_rejected(kind):
    raw = episode()['raw_probabilities']
    if kind == 'nan':
        raw[0, 0, 0] = np.nan
    elif kind == 'negative':
        raw[0, 0, 0] = -.1
    elif kind == 'zero_row':
        raw[0, 0] = 0
    elif kind == 'bad_mass':
        raw[0, 0] = .1
    elif kind == 'integer':
        raw = raw.astype(np.int64)
    else:
        raw = raw[:, :51]
    with pytest.raises(ValueError):
        calibration.transform_probabilities(raw)


@pytest.mark.parametrize('kind', ['field', 'target_dtype', 'mask_dtype', 'unmasked_label', 'rank',
                                 'negative_age', 'future_age', 'startup', 'too_long'])
def test_invalid_episode_schema_and_public_age_bounds_rejected(kind):
    ep = episode(105 if kind == 'too_long' else 5)
    query(ep, 2, 0, probability=.8)
    if kind == 'field':
        ep['hidden_deck'] = np.arange(52)
    elif kind == 'target_dtype':
        ep['targets'] = ep['targets'].astype(np.int32)
    elif kind == 'mask_dtype':
        ep['target_mask'] = ep['target_mask'].astype(np.int64)
    elif kind == 'unmasked_label':
        ep['targets'][0, 0] = 1
    elif kind == 'rank':
        ep['targets'][2, 0] = 13
    elif kind == 'negative_age':
        ep['ages'][2, 0] = 0
    elif kind == 'future_age':
        ep['ages'][2, 0] = 3
    elif kind == 'startup':
        query(ep, 0, 0, probability=.8)
    with pytest.raises(ValueError):
        calibration.fit_temperature([ep])


@pytest.mark.parametrize('mode', ['temperature2', None, True])
def test_unknown_mode_rejected(mode):
    with pytest.raises(ValueError, match='mode'):
        calibration.score_episodes([episode()], mode)


@pytest.mark.parametrize('mode', ['baseline', 'hard'])
def test_other_modes_cannot_silently_accept_temperature(mode):
    with pytest.raises(ValueError, match='Only temperature'):
        calibration.transform_probabilities(episode()['raw_probabilities'], mode, 2)
