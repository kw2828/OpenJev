"""Fabricated prospective action-effect gate and immutable legacy-gate checks."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research import otto_action_effect_metrics as metrics
from openjev.research import otto_belief_distillation_metrics as base

FAMILIES = ('effect_recurrent', 'paired_recurrent', 'paired_blind', 'paired_direct')
THRESHOLDS = {'long_effect_relative_gain': .1, 'long_gap_relative_gain': .05,
              'long_log_relative_tolerance': .01, 'normal_log_relative_tolerance': .01,
              'normal_gap_relative_tolerance': .01, 'minimum_supported_cases': 2}
OPTIONS = {'candidate': FAMILIES[0], 'controls': FAMILIES[1:], 'fit_seeds': (1, 2, 3),
           'regimes': ('lambda3', 'lambda4'), 'thresholds': THRESHOLDS}


def fixture(*, zero_signal_case=False):
    target = np.zeros((4, 8), np.int64)
    raw = np.broadcast_to([0., 1., 2., 3.], (4, 8, 4)).copy()
    legal = np.ones((4, 8, 4), np.bool_)
    p = np.broadcast_to([.3, .2, .2, .2, .1], (4, 8, 5)).copy()
    opposite = np.broadcast_to([.1, .2, .2, .2, .3], (4, 8, 5)).copy()
    if zero_signal_case:
        opposite[0] = p[0]
    actions = np.tile([0, 1, 2, 3] * 2, (4, 1)).astype(np.int64)
    ids, regimes = ['a', 'b', 'c', 'd'], ['lambda3'] * 2 + ['lambda4'] * 2
    reports, sensitivity = [], []
    for family in FAMILIES:
        for seed in (1, 2, 3):
            for condition in ('gap', 'normal'):
                cost = raw.copy()
                if family != FAMILIES[0] and condition == 'gap':
                    cost[:] = [1, 0, 2, 3]
                reports.append(base.score(target, p, cost, raw, legal, oracle_probabilities=p,
                    case_ids=ids, regimes=regimes, family=family, fit_seed=seed, condition=condition,
                    prediction_kind='probabilities'))
            report = base.action_sensitivity(p, opposite, case_ids=ids, regimes=regimes, actions=actions,
                alternate_actions=actions ^ 1, predicted_original=p,
                predicted_alternate=opposite if family == FAMILIES[0] else p)
            sensitivity.append({'family': family, 'fit_seed': seed, 'report': report})
    return reports, sensitivity


def set_effect(sensitivity, value, *, family=FAMILIES[0], seed=None):
    for row in sensitivity:
        if row['family'] != family or (seed is not None and row['fit_seed'] != seed):
            continue
        for h in ('5', '6', '7', '8'):
            for leaf in row['report']['per_horizon'][h]['by_regime'].values():
                for case in leaf['cases']:
                    case['model_effect_error'] = value
                leaf['case_weighted_model_effect_error'] = value


def test_new_hypothesis_can_pass_without_relabeling_legacy_failure():
    reports, sensitivity = fixture()
    before = copy.deepcopy((reports, sensitivity))
    result = metrics.evaluate_reports(reports, sensitivity, **OPTIONS)
    assert result['passed'] is True and result['passed_cells'] == result['total_cells'] == 18
    assert all(len(cell['conditions']) == 6 for cell in result['cells'])
    assert result['admits_execution'] is False
    legacy = base.evaluate_reports(reports, **{**OPTIONS, 'thresholds': {
        'long_log_relative_gain': .01, 'long_gap_relative_gain': .05,
        'normal_log_relative_tolerance': .01, 'normal_gap_relative_tolerance': .01, 'minimum_supported_cases': 2}})
    assert legacy['passed'] is False
    assert (reports, sensitivity) == before


@pytest.mark.parametrize('candidate,passed', [(.9, True), (np.nextafter(.9, 1.).item(), False), (0., True)])
def test_ten_percent_effect_boundary_is_inclusive_and_exact(candidate, passed):
    reports, sensitivity = fixture()
    for family in FAMILIES[1:]:
        set_effect(sensitivity, 1., family=family)
    set_effect(sensitivity, candidate)
    result = metrics.evaluate_reports(reports, sensitivity, **OPTIONS)
    assert result['passed'] is passed


def test_equal_zero_effects_cannot_meet_gain_and_every_seed_can_veto():
    reports, sensitivity = fixture()
    for family in FAMILIES:
        set_effect(sensitivity, 0., family=family)
    result = metrics.evaluate_reports(reports, sensitivity, **OPTIONS)
    assert result['passed_cells'] == 0
    reports, sensitivity = fixture()
    set_effect(sensitivity, 1., seed=3)
    result = metrics.evaluate_reports(reports, sensitivity, **OPTIONS)
    assert result['passed'] is False and result['passed_cells'] == 12
    assert all(not row['passed'] for row in result['cells'] if row['fit_seed'] == 3)


@pytest.mark.parametrize('condition,group,metric', [('gap', 'long', 'log_score'),
    ('normal', 'all', 'log_score'), ('normal', 'all', 'decision_gap')])
def test_each_noninferiority_condition_is_required(condition, group, metric):
    reports, sensitivity = fixture()
    for report in reports:
        if report['family'] == FAMILIES[0] and report['condition'] == condition:
            for leaf in report['groups'][group]['by_regime'].values():
                leaf['case_weighted_' + metric] = 100.
    assert metrics.evaluate_reports(reports, sensitivity, **OPTIONS)['passed'] is False


def test_support_failure_cannot_be_hidden_by_good_effects():
    reports, sensitivity = fixture()
    result = metrics.evaluate_reports(reports, sensitivity, **{**OPTIONS, 'thresholds': {
        **THRESHOLDS, 'minimum_supported_cases': 3}})
    assert result['passed'] is False
    assert all(row['conditions'][-1]['passed'] is False for row in result['cells'])


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'mapping', 'case', 'rows', 'nan', 'oracle', 'mean'])
def test_malformed_or_unpaired_effect_views_fail_closed(change):
    reports, sensitivity = fixture()
    if change == 'missing':
        sensitivity.pop()
    elif change == 'duplicate':
        sensitivity[-1] = copy.deepcopy(sensitivity[0])
    elif change == 'mapping':
        sensitivity[0]['report']['action_mapping'] = 'reverse'
    else:
        leaf = sensitivity[0]['report']['per_horizon']['5']['by_regime']['lambda3']
        if change == 'case':
            leaf['cases'][0]['case_id'] = 'missing'
        elif change == 'rows':
            leaf['cases'][0]['rows'] = 0
        elif change == 'nan':
            leaf['cases'][0]['model_effect_error'] = float('nan')
        elif change == 'oracle':
            leaf['cases'][0]['oracle_signal'] += .2
            leaf['case_weighted_oracle_signal'] += .1
        else:
            leaf['case_weighted_model_effect_error'] += .1
    with pytest.raises(ValueError):
        metrics.evaluate_reports(reports, sensitivity, **OPTIONS)


def test_zero_signal_case_remains_in_equal_case_effect_denominator():
    reports, sensitivity = fixture(zero_signal_case=True)
    result = metrics.evaluate_reports(reports, sensitivity, **OPTIONS)
    row = next(row for row in result['long_effects'] if row['family'] == FAMILIES[1]
               and row['fit_seed'] == 1 and row['regime'] == 'lambda3')
    assert row['declared_cases'] == 2 and row['rows'] == 8
    assert row['cases'][0]['oracle_signal'] == row['cases'][0]['model_effect_error'] == 0.
    assert row['case_weighted_oracle_signal'] == pytest.approx(.04)
    assert row['case_weighted_model_effect_error'] == pytest.approx(.04)
    assert result['passed'] is True


@pytest.mark.parametrize('gap,passed', [(.95, True), (np.nextafter(.95, 1.).item(), False)])
def test_five_percent_decision_gap_boundary(gap, passed):
    reports, sensitivity = fixture()
    for report in reports:
        if report['family'] == FAMILIES[0] and report['condition'] == 'gap':
            for leaf in report['groups']['long']['by_regime'].values():
                leaf['case_weighted_decision_gap'] = gap
    assert metrics.evaluate_reports(reports, sensitivity, **OPTIONS)['passed'] is passed
