"""Fabricated scalar oracles and corruptions; no models or empirical inputs."""
from __future__ import annotations

import ast
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_decision_error as a
import finite_decision_error_diagnostic as producer

SEEDS = (947101, 947102, 947103, 947104, 947105)


def fixture():
    """Five fixed cases span ties, all four bins, bias and both error signs."""
    true = [[0., 1., 2., 3.], [0., 0., .5, 1.], [0., .001, .02, .2],
            [0., .01, .1, .2], [0., .1, .2, .3]]
    random = [[1., 0., 2., 3.], [1., 0., .5, 1.], [5., 5.001, 5.02, 5.2],
              [.01, 0., .1, .2], true[4]]
    control = [[1., 2., 0., 3.], true[1], [.001, 0., .02, .2], true[3], true[4]]
    inputs = {'rounded_anchor': true, 'rounded_random': random, 'matched_free_random': control}
    truth = np.repeat(np.array(true, np.float64)[:, None, :], 8, axis=1)
    predictions = {(arm, seed): np.repeat(np.array(inputs[arm], np.float64)[:, None, :], 8, axis=1)
                   for arm in a.ARMS for seed in SEEDS}
    ids = [f'fabricated-case-{i}' for i in range(5)]
    result = producer.analyze(truth, predictions, ids)
    parent = []
    # This parent oracle is derived directly from the hand constants, not from
    # either implementation's saved metrics or reduction helpers.
    for arm in a.ARMS:
        regret, square = [], []
        for target, prediction in zip(true, inputs[arm], strict=True):
            selected = min(range(4), key=lambda i: (prediction[i], i))
            regret.append(target[selected] - min(target))
            square.extend((prediction[i] - target[i]) ** 2 for i in range(4))
        for seed in SEEDS:
            for horizon in a.HORIZONS:
                parent.append({'arm': arm, 'seed': seed, 'horizon': horizon, 'regime': 'base',
                    'cases': 5, 'blind_regret': math.fsum(regret) / 5,
                    'blind_cost_mse': math.fsum(square) / 20})
    return result['records'], result['summary'], parent, ids


def run(values):
    records, summary, parent, ids = values
    return a.audit(records, summary, parent, expected_case_ids=ids)


def test_actual_fabricated_producer_output_matches_independent_full_roster_oracle():
    values = fixture()
    before = copy.deepcopy(values)
    result = run(values)
    assert result['agreement'] and not result['technical_complete']
    assert len(result['means']) == 60 and len(result['arm_means']) == 12
    assert len(result['contrasts']) == 20 and len(result['contrast_means']) == 4
    assert result['counts']['case_records_recomputed'] == 300
    assert result['counts']['paired_case_contrasts'] == 100
    assert all(result['counts'][key] == 0 for key in ('array_decodes', 'checkpoint_decodes',
        'model_calls', 'optimizer_calls', 'world_or_generator_calls'))
    assert values == before
    group = result['contrasts'][0]
    assert group['control'] == 'matched_free_random' and group['seed'] == SEEDS[0] and group['horizon'] == 4
    assert [r['count'] for r in group['categories']] == [2, 1, 2]
    assert [r['count'] for r in group['margin_bins']] == [1, 1, 1, 2]
    assert group['actions_differ'] == 4  # A different exact-tie action has zero regret difference.
    assert group['means']['delta_regret'] == pytest.approx(-.991 / 5)
    assert group['categories'][0]['population_contributions']['delta_regret'] == pytest.approx(-1.001 / 5)
    assert group['categories'][0]['within_means']['delta_regret'] == pytest.approx(-1.001 / 2)


def test_cost_witnesses_and_centering_have_hand_derived_values():
    wrong = a.cost_metrics([0., 1., 2., 3.], [1., 0., 2., 3.])
    assert wrong == {'true_action': 0, 'predicted_action': 1, 'regret': 1., 'mse': .5,
                     'centered_mse': .5, 'true_margin': 1., 'true_chosen_gap': 1., 'predicted_contrast': -1.}
    bias = a.cost_metrics([0., 1., 2., 3.], [5., 6., 7., 8.])
    assert bias['mse'] == 25. and bias['centered_mse'] == bias['regret'] == 0.
    tied = a.cost_metrics([0., 0., .5, 1.], [1., 0., .5, 1.])
    assert tied['true_action'] == 0 and tied['predicted_action'] == 1
    assert tied['true_margin'] == tied['regret'] == 0.
    assert tied['mse'] == .25 and tied['centered_mse'] == .1875
    assert tied['predicted_contrast'] == -1.
    raw_tie = a.cost_metrics([0., 1., 2., 3.], [0., 0., 4., 5.])
    assert raw_tie['predicted_action'] == 0


@pytest.mark.parametrize('value,expected', [(0., 0), (.000999, 0), (.001, 1), (.00999, 1),
                                          (.01, 2), (.09999, 2), (.1, 3), (4., 3)])
def test_fixed_margin_bin_boundaries(value, expected):
    assert a.margin_bin(value) == expected


def test_partition_uses_all_cases_not_subgroup_size_and_reports_empty_groups():
    records = [{'candidate_regret': .25, 'control_regret': .75, 'delta_regret': -.5,
                'actions_differ': True, 'category': 'candidate_better', 'bin_index': 0},
               {'candidate_regret': 1., 'control_regret': 0., 'delta_regret': 1.,
                'actions_differ': True, 'category': 'control_better', 'bin_index': 0},
               {'candidate_regret': 0., 'control_regret': 0., 'delta_regret': 0.,
                'actions_differ': False, 'category': 'equal', 'bin_index': 0}]
    report = a.partition(records)
    assert report['means']['delta_regret'] == pytest.approx(1 / 6)
    assert report['categories'][0]['population_contributions']['delta_regret'] == pytest.approx(-1 / 6)
    assert report['categories'][0]['within_means']['delta_regret'] == -.5
    for row in report['margin_bins'][1:]:
        assert row['count'] == row['fraction'] == row['actions_differ'] == 0
        assert row['within_means'] is None
        assert row['population_contributions'] == dict.fromkeys(a.CONTRAST_METRICS, 0.)


def test_exact_sign_classification_does_not_hide_a_small_positive_difference():
    candidate = {'case_id': 'a', 'case_index': 0, 'regret': 1e-15, 'true_margin': 0., 'predicted_action': 1}
    control = {'case_id': 'a', 'case_index': 0, 'regret': 0., 'true_margin': 0., 'predicted_action': 0}
    assert a.case_contrast(candidate, control)['category'] == 'control_better'


@pytest.mark.parametrize('fault', ['record_drop', 'record_duplicate', 'case_id', 'case_index', 'head_arm',
    'target', 'prediction', 'choice', 'regret', 'margin', 'centered', 'witness', 'category', 'bin',
    'contribution_denominator', 'within_mean', 'contrast_drop', 'arm_mean', 'parent_mean', 'parent_drop',
    'expected_ids', 'whole_case_drop', 'rescue', 'definitions', 'work'])
def test_scalar_or_identity_corruptions_fail_closed(fault):
    values = fixture()
    records, summary, parent, ids = values
    if fault == 'record_drop':
        records.pop()
    elif fault == 'record_duplicate':
        records[-1] = copy.deepcopy(records[0])
    elif fault == 'case_id':
        records[0]['case_id'] = 'unbound'
    elif fault == 'case_index':
        records[0]['case_index'] = True
    elif fault == 'head_arm':
        records[0]['arm'] = 'rounded'
    elif fault == 'target':
        records[0]['true_costs'][1] += .25
    elif fault == 'prediction':
        records[0]['predicted_costs'][0] += .25
    elif fault == 'choice':
        records[0]['predicted_action'] = 1
    elif fault == 'regret':
        records[0]['regret'] += .01
    elif fault == 'margin':
        records[0]['true_margin'] += .01
    elif fault == 'centered':
        records[0]['centered_mse'] += .01
    elif fault == 'witness':
        records[0]['predicted_contrast'] -= .01
    elif fault == 'category':
        summary['contrasts'][0]['records'][0]['category'] = 'equal'
    elif fault == 'bin':
        summary['contrasts'][0]['records'][0]['bin_index'] = 0
    elif fault == 'contribution_denominator':
        block = summary['contrasts'][0]['categories'][0]
        block['population_contributions']['delta_regret'] = block['within_means']['delta_regret']
    elif fault == 'within_mean':
        summary['contrasts'][0]['categories'][0]['within_means']['delta_regret'] = 0.
    elif fault == 'contrast_drop':
        summary['contrasts'].pop()
    elif fault == 'arm_mean':
        summary['arm_means'][0]['means']['mse'] += .01
    elif fault == 'parent_mean':
        parent[0]['blind_cost_mse'] += .01
    elif fault == 'parent_drop':
        parent.pop()
    elif fault == 'expected_ids':
        ids[0] = 'other'
    elif fault == 'whole_case_drop':
        records[:] = [r for r in records if r['case_index'] != 4]
        summary['roster']['cases'] -= 1
        summary['roster']['case_ids'].pop()
    elif fault == 'rescue':
        summary['failed_rule_rescued'] = True
    elif fault == 'definitions':
        summary['definitions']['category'] = 'Use a tolerance to discard small losses.'
    else:
        summary['work']['case_records'] -= 1
    with pytest.raises(ValueError):
        run(values)


@pytest.mark.parametrize('bad', [math.inf, math.nan, True, '0', None])
def test_cost_vectors_reject_nonfinite_or_non_numeric_entries(bad):
    with pytest.raises(ValueError):
        a.cost_metrics([0., 1., 2., 3.], [bad, 1., 2., 3.])


def test_external_check_failure_is_not_swallowed():
    records, summary, parent, ids = fixture()
    marker = TimeoutError('fabricated admission deadline')
    def stop():
        raise marker
    with pytest.raises(TimeoutError) as caught:
        a.audit(records, summary, parent, expected_case_ids=ids, check=stop)
    assert caught.value is marker


def test_independent_auditor_source_imports_only_standard_library():
    tree = ast.parse(Path(a.__file__).read_text())
    imports = [node.module if isinstance(node, ast.ImportFrom) else alias.name
               for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in (node.names if isinstance(node, ast.Import) else [None])]
    assert set(imports) == {'__future__', 'math'}
