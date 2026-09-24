"""Fabricated decision selection, unconditional weights and bootstrap contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_otto_cost_information as r


def fixture_data(n=4):
    m = r.c.CONFIG['mc_draws']
    root = np.zeros((n, 2809), np.float64)
    root[:, 0] = 1.
    return {'case_ids': np.asarray([f'fake:{i}' for i in range(n)]),
            'regimes': np.asarray(['lambda3'] * (n // 2) + ['lambda4'] * (n - n // 2)),
            'prefix_position': np.full((n, 2), 26, np.int64),
            'actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)), 'root_grid': root,
            'mc_costs8': np.tile(np.array([1., 3., 5., 7.], np.float32), (n, 2, m, 1)),
            'mc_alive8': np.ones((n, 2, m), np.bool_)}


def predictions(n=4, actions=(1, 2, 3)):
    result = {}
    for f in r.FAMILIES:
        for j, s in enumerate(r.c.FIT_SEEDS):
            value = np.ones((n, 8, 4), np.float32)
            value[:, :, actions[j]] = 0.
            result[f'{f}__{s}__cost'] = value
    return result


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setitem(r.c.CONFIG, 'bootstrap_replicates', 64)
    monkeypatch.setitem(r.c.CONFIG, 'min_cases', 2)
    monkeypatch.setitem(r.c.CONFIG, 'min_mc_survivors', 1)


def test_separate_decisions_are_averaged_without_new_ensemble_policy():
    report = r.primary_gate(fixture_data(), predictions(), np)
    assert report['passed']
    for group in report['groups']:
        assert group['control_gap'] == 4.
        assert group['reference_gap'] == 0.
        assert group['gain'] == 4.
        assert group['approximate_95_percent_interval'][1] == [4., 4.]
    assert all(x['control_actions'] == [1, 2, 3] and x['reference_action'] == 0 for x in report['cases'])


def test_selection_does_not_peek_at_evaluation_argmin_and_negative_gain_is_preserved():
    d = fixture_data()
    d['mc_costs8'][:, 1] = [8., 1., 1., 1.]
    report = r.primary_gate(d, predictions(), np)
    assert not report['passed']
    assert all(row['reference_action'] == 0 for row in report['cases'])
    assert all(g['gain'] == -7. and g['reference_gap'] == 7. for g in report['groups'])
    assert all(not g['conditions']['resolved_gain'] for g in report['groups'])


def test_found_draws_and_zero_support_prefixes_stay_in_unconditional_denominator():
    d = fixture_data()
    d['mc_costs8'][[0, 2]] = 0
    d['mc_alive8'][[0, 2]] = False
    report = r.primary_gate(d, predictions(), np)
    for g in report['groups']:
        assert g['cases'] == 2 and g['control_gap'] == 2. and g['gain'] == 2.
        assert not g['conditions']['mc_support']  # exact S is positive; no replacement draws.
    assert all(report['cases'][i]['control_gap'] == 0. for i in (0, 2))
    assert all(report['cases'][i]['reference_action'] == 0 for i in (0, 2))


def test_bootstrap_is_reproducible_and_uses_both_levels():
    d = fixture_data()
    d['mc_costs8'][0, 1, :64, 1:] += 1.
    a = r.primary_gate(d, predictions(), np)
    b = r.primary_gate(d, predictions(), np)
    assert a == b
    assert len({tuple(v) for v in a['groups'][0]['bootstrap_replicates']}) > 1
    assert a['groups'][0]['bootstrap_index_sha256'] != a['groups'][1]['bootstrap_index_sha256']


def test_survival_counts_unique_visited_source_cells_once():
    d = fixture_data()
    d['root_grid'][:] = 0
    # Action sequence visits (25,26), (26,26) repeatedly.
    d['root_grid'][:, 25 * 53 + 26] = .25
    d['root_grid'][:, 26 * 53 + 26] = .5
    d['root_grid'][:, 0] = .25
    np.testing.assert_array_equal(r.survival(d, np, 8), np.full(4, .25))


@pytest.mark.parametrize('defect', ('missing', 'nan', 'shape', 'dtype'))
def test_all_frozen_prediction_views_required(defect):
    d, p = fixture_data(), predictions()
    key = next(iter(p))
    if defect == 'missing':
        del p[key]
    elif defect == 'nan':
        p[key][0, 0, 0] = np.nan
    elif defect == 'shape':
        p[key] = p[key][:, :4]
    else:
        p[key] = p[key].astype(np.float64)
    with pytest.raises(ValueError, match='forecasts'):
        r.primary_gate(d, p, np)


def test_analyze_multiplies_each_case_mass_before_equal_case_averaging():
    d = fixture_data()
    d['exact_weights'] = np.zeros((4, 256), np.float64)
    d['exact_weights'][:, 0] = [1., .5, 1., .5]
    d['exact_costs'] = np.zeros((4, 256, 4), np.float32)
    d['exact_costs'][:, 0] = [[0, 4, 4, 4], [0, 8, 8, 8]] * 2
    d['mc_costs4'] = d['mc_costs8'].copy()
    report = r.analyze(d, predictions(actions=(1, 1, 1)), np)
    for row in report['rows']:
        if row['horizon'] == 4:
            assert row['total_regret'] == 4.
            assert row['information_advantage'] == 0.
            assert row['approximation_regret'] == 4.
    # mean mass * mean conditional regret would incorrectly give4.5.
    assert len(report['rows']) == 48
    assert len(report['h4_sampling_validation']) == 8
