"""Hand-calculated diagnostic identities; no empirical files or models."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finite_decision_error_diagnostic as diagnostic

SEEDS = tuple(range(947101, 947106))


def fixture():
    target = np.array([[0., 0., 1., 2.], [0., .001, 2., 3.],
                       [0., .01, .04, 2.], [0., .1, .2, .3], [0., .2, .3, 1.]], np.float64)
    truth = np.repeat(target[:, None, :], 8, axis=1)
    predictions = {}
    for arm in diagnostic.ARMS:
        chosen = [1, 0, 1, 0, 3] if arm == 'rounded_random' else [0, 1, 0, 1, 0]
        values = np.full_like(truth, 2.)
        for index, action in enumerate(chosen):
            values[index, :, action] = -1.
        for seed in SEEDS:
            predictions[arm, seed] = values.copy()
    return truth, predictions, ['tie', 'b001', 'b01', 'b1', 'large']


def group(summary, *, control='matched_free_random', horizon=4, seed=SEEDS[0]):
    return next(row for row in summary['contrasts'] if (row['control'], row['horizon'], row['seed'])
                == (control, horizon, seed))


def test_exact_ties_true_bin_boundaries_and_asymmetric_misselection():
    result = diagnostic.analyze(*fixture())
    records, summary = result['records'], result['summary']
    assert len(records) == 60 * 5
    assert len(summary['means']) == 60 and len(summary['arm_means']) == 12
    assert len(summary['contrasts']) == 20 and len(summary['contrast_means']) == 4
    tie = next(row for row in records if row['arm'] == 'rounded_random' and row['case_id'] == 'tie')
    assert tie['true_action'] == 0 and tie['predicted_action'] == 1
    assert tie['true_margin'] == tie['regret'] == tie['true_chosen_gap'] == 0.
    assert tie['predicted_contrast'] == -3.
    # Error = [2,-1,1,0]; MSE=6/4, squared centered error=5/4.
    assert tie['mse'] == 1.5 and tie['centered_mse'] == 1.25
    comparison = group(summary)
    assert [r['bin_index'] for r in comparison['records']] == [0, 1, 2, 3, 3]
    assert [r['category'] for r in comparison['records']] == [
        'equal', 'candidate_better', 'control_better', 'candidate_better', 'control_better']
    assert comparison['actions_differ'] == 5  # The zero-regret tie also chooses a different action.
    assert comparison['means']['candidate_regret'] == pytest.approx(1.01 / 5, abs=1e-15)
    assert comparison['means']['control_regret'] == pytest.approx(.101 / 5, abs=1e-15)
    assert comparison['means']['delta_regret'] == pytest.approx(.909 / 5, abs=1e-15)
    categories = {row['category']: row for row in comparison['categories']}
    assert categories['candidate_better']['count'] == categories['control_better']['count'] == 2
    assert categories['equal']['count'] == 1
    assert categories['candidate_better']['population_contributions']['delta_regret'] == pytest.approx(-.101 / 5)
    assert categories['candidate_better']['within_means']['delta_regret'] == pytest.approx(-.101 / 2)
    assert categories['control_better']['population_contributions']['delta_regret'] == pytest.approx(1.01 / 5)
    bins = comparison['margin_bins']
    assert [row['count'] for row in bins] == [1, 1, 1, 2]
    assert bins[-1]['upper'] is None and bins[-1]['within_means']['delta_regret'] == pytest.approx(.9 / 2)
    assert bins[-1]['population_contributions']['delta_regret'] == pytest.approx(.9 / 5)
    # Equal counts of better/worse cases do not imply equal magnitudes or a zero mean difference.
    assert comparison['means']['delta_regret'] > 0
    json.dumps(result, allow_nan=False)
    assert summary['descriptive_only'] and not summary['failed_rule_rescued']


def test_predicted_tie_always_chooses_first_index_and_does_not_clip_tiny_differences():
    truth, predictions, ids = fixture()
    truth[0] = [0., 1e-14, 1., 2.]
    for seed in SEEDS:
        predictions['rounded_random', seed][0] = 0.
        predictions['matched_free_random', seed][0] = [1., 0., 1., 1.]
    result = diagnostic.analyze(truth, predictions, ids)
    row = next(r for r in result['records'] if r['arm'] == 'rounded_random' and r['case_index'] == 0)
    assert row['predicted_action'] == row['true_action'] == 0
    assert row['predicted_contrast'] == row['regret'] == 0.
    paired = group(result['summary'])['records'][0]
    assert paired['delta_regret'] == -1e-14 and paired['category'] == 'candidate_better'


def test_full_population_decomposition_equal_fit_means_and_empty_subgroups():
    truth, predictions, ids = fixture()
    # One seed has a different policy; aggregation must keep all five, not select the best seed.
    predictions['rounded_random', SEEDS[-1]] = truth.copy()
    summary = diagnostic.analyze(truth, predictions, ids)['summary']
    for row in summary['contrasts'] + summary['contrast_means']:
        for partition in ('categories', 'margin_bins'):
            population = row.get('cases', row.get('case_views'))
            assert sum(part['count'] for part in row[partition]) == population
            assert math.fsum(part['fraction'] for part in row[partition]) == pytest.approx(1.)
            for metric in diagnostic.COMPARISON_METRICS:
                assert math.fsum(part['population_contributions'][metric] for part in row[partition]) == pytest.approx(
                    row['means'][metric], abs=1e-15)
    average = next(r for r in summary['arm_means'] if r['arm'] == 'rounded_random' and r['horizon'] == 4)
    assert average['means']['regret'] == pytest.approx(4 / 5 * 1.01 / 5)
    assert average['case_views'] == 25 and average['cases_per_fit'] == 5 and average['fits'] == 5
    aggregate = next(r for r in summary['contrast_means'] if r['control'] == 'matched_free_random' and r['horizon'] == 4)
    assert aggregate['means']['delta_regret'] == pytest.approx((4 * 1.01 / 5 - .101) / 5)
    # All-zero true costs give only the zero-margin bin and equal-regret category.
    truth.fill(0.)
    empty = group(diagnostic.analyze(truth, predictions, ids)['summary'])
    for row in empty['margin_bins'][1:] + empty['categories'][:2]:
        assert row['count'] == row['actions_differ'] == 0 and row['fraction'] == 0.
        assert row['within_means'] is None
        assert row['population_contributions'] == dict.fromkeys(diagnostic.COMPARISON_METRICS, 0.)


def test_centered_error_removes_only_action_constant_bias_and_outputs_own_data():
    truth = np.repeat(np.array([[[0., .25, .5, .75]]], np.float64), 8, axis=1)
    predictions = {(arm, seed): truth + 2. for arm in diagnostic.ARMS for seed in SEEDS}
    snapshots = {key: value.copy() for key, value in predictions.items()}
    saved_truth = truth.copy()
    result = diagnostic.analyze(truth, predictions, ['one'])
    assert all(r['mse'] == 4. and r['centered_mse'] == r['regret'] == 0. for r in result['records'])
    assert np.array_equal(truth, saved_truth)
    assert all(np.array_equal(value, snapshots[key]) for key, value in predictions.items())
    result['records'][0]['true_costs'][0] = 99.
    result['records'][0]['predicted_costs'][0] = 99.
    result['summary']['definitions']['regret'] = 'changed'
    assert np.array_equal(truth, saved_truth)
    assert all(np.array_equal(value, snapshots[key]) for key, value in predictions.items())
    assert diagnostic.DEFINITIONS['regret'] != 'changed'


def test_case_permutation_is_equivariant_and_does_not_change_aggregate_values():
    truth, predictions, ids = fixture()
    original = diagnostic.analyze(truth, predictions, ids)
    order = [4, 0, 3, 2, 1]
    shuffled = diagnostic.analyze(truth[order], {key: value[order] for key, value in predictions.items()},
                                  [ids[index] for index in order])
    assert original['summary']['means'] == shuffled['summary']['means']
    assert original['summary']['arm_means'] == shuffled['summary']['arm_means']
    assert original['summary']['contrast_means'] == shuffled['summary']['contrast_means']
    a = {(r['arm'], r['seed'], r['horizon'], r['case_id']): r for r in original['records']}
    for row in shuffled['records']:
        previous = copy.deepcopy(a[row['arm'], row['seed'], row['horizon'], row['case_id']])
        previous['case_index'] = row['case_index']
        assert row == previous


@pytest.mark.parametrize('fault', ['duplicate_id', 'integer_id', 'short_ids', 'missing_fit', 'extra_arm',
    'bool_seed', 'sixth_seed', 'nonfinite_true', 'nonfinite_prediction', 'wrong_shape', 'short_horizon',
    'integer_costs', 'empty_cases', 'alternate_horizons'])
def test_malformed_or_selected_inputs_fail(fault):
    truth, predictions, ids = fixture()
    kwargs = {}
    if fault == 'duplicate_id':
        ids[-1] = ids[0]
    elif fault == 'integer_id':
        ids[0] = 1
    elif fault == 'short_ids':
        ids.pop()
    elif fault == 'missing_fit':
        predictions.pop(('rounded_random', SEEDS[-1]))
    elif fault == 'extra_arm':
        predictions['selected_winner', SEEDS[0]] = truth.copy()
    elif fault == 'bool_seed':
        predictions['rounded_random', True] = truth.copy()
    elif fault == 'sixth_seed':
        for arm in diagnostic.ARMS:
            predictions[arm, 947106] = truth.copy()
    elif fault == 'nonfinite_true':
        truth[0, 0, 0] = np.nan
    elif fault == 'nonfinite_prediction':
        predictions['rounded_random', SEEDS[0]][0, 0, 0] = np.inf
    elif fault == 'wrong_shape':
        predictions['rounded_random', SEEDS[0]] = truth[:1]
    elif fault == 'short_horizon':
        truth = truth[:, :2]
    elif fault == 'integer_costs':
        truth = truth.astype(np.int64)
    elif fault == 'empty_cases':
        truth = truth[:0]
    else:
        kwargs['horizons'] = (4, 8)
    with pytest.raises(ValueError):
        diagnostic.analyze(truth, predictions, ids, **kwargs)


def test_finite_inputs_that_overflow_arithmetic_fail_without_nonfinite_json():
    truth, predictions, ids = fixture()
    predictions['rounded_random', SEEDS[0]].fill(1e308)
    with pytest.raises(ValueError, match='finite'):
        diagnostic.analyze(truth, predictions, ids)


def test_exact_callback_count_and_external_failure_propagation():
    calls = []
    result = diagnostic.analyze(*fixture(), check=lambda: calls.append(None))
    assert result['summary']['work']['check_calls'] == len(calls) == 82
    assert result['summary']['work']['model_calls'] == result['summary']['work']['array_file_decodes'] == 0
    marker = TimeoutError('fabricated external deadline')
    def fail():
        raise marker
    with pytest.raises(TimeoutError) as caught:
        diagnostic.analyze(*fixture(), check=fail)
    assert caught.value is marker
