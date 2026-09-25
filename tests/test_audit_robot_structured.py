"""Fabricated scalar checks only; no saved study input or checkpoint access."""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_robot_structured as audit
from test_robot_structured_study import resources_fixture, rows_fixture


def conditions(result):
    return {row['name']: row['passed'] for row in result['conditions']}


def test_independent_metrics_have_closed_form_horizon_and_joint_values():
    prediction = np.ones((2, 128, 6), dtype=np.float64)
    prediction[:, 64:] = 3.
    target = np.zeros_like(prediction)
    scale = np.arange(1., 7.)
    short = audit.scored(prediction, target, scale, 64, np)
    long = audit.scored(prediction, target, scale, 128, np)
    assert short == {
        'standardized_rmse': 1., 'standardized_sse': 768., 'scalars': 768,
        'physical_rmse_deg': math.sqrt(91 / 6),
        'per_joint_rmse_deg': scale.tolist(), 'windows': 2, 'horizon': 64,
    }
    assert long['standardized_sse'] == 7680.
    assert long['scalars'] == 1536 and long['standardized_rmse'] == math.sqrt(5)
    assert long['physical_rmse_deg'] == math.sqrt(455 / 6)
    np.testing.assert_allclose(long['per_joint_rmse_deg'], scale * math.sqrt(5), rtol=1e-15)


def test_nonfinite_predictions_and_overflow_remain_failed_rows():
    target = np.zeros((1, 128, 6), dtype=np.float64)
    for value, reason in [(float('nan'), 'nonfinite prediction/target'),
                          (1e300, 'nonfinite metric arithmetic')]:
        prediction = np.full_like(target, value)
        rows = audit.metric_rows({'arm': 'householder'}, prediction, target, np.ones(6), None, np)
        assert [r['horizon'] for r in rows] == [64, 128]
        assert all(r['status'] == 'FAILED' and r['metrics'] is None for r in rows)
        assert all(r['error'] == {'type': 'NonfiniteEvaluation', 'message': reason} for r in rows)
    rows = audit.metric_rows({}, None, target, np.ones(6), {'type': 'FailedTrainingAttempt'}, np)
    assert all(r['error'] == {'type': 'FailedTrainingAttempt'} for r in rows)


def test_complete_independent_160_row_rule_passes_all61():
    cfg, rows = rows_fixture()
    before = copy.deepcopy(rows)
    selection, result = audit.decisions(rows, resources_fixture(), cfg)
    assert len(rows) == 160 and rows == before
    assert set(selection['selected_rates']) == set(audit.ARMS)
    assert set(selection['selected_rates'].values()) == {.001}
    assert selection['selected_ridge'] == 'causal_ridge_1'
    assert result['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'
    assert result['passed'] == result['total'] == len(result['conditions']) == 61
    assert len(conditions(result)) == 61 and all(conditions(result).values())


@pytest.mark.parametrize('arm,expected_passes', [('householder', 0), ('legacy_instant', 52)])
def test_failed_family_is_ineligible_and_never_dropped_from_rule(arm, expected_passes):
    cfg, rows = rows_fixture()
    for row in rows:
        if row['arm'] == arm:
            row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    resources = [row for row in resources_fixture() if row['arm'] != arm]
    selection, result = audit.decisions(rows, resources, cfg)
    assert selection['selected_rates'][arm] is None
    assert not any(option['eligible'] for option in selection['options'][arm])
    assert len(rows) == 160 and result['total'] == 61
    assert result['passed'] == expected_passes
    assert result['status'] == 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION'
    assert conditions(result)['all_selected_families_and_causal_ridge_eligible'] is False


def test_one_failed_recipe_forces_other_rate_across_all_seeds():
    cfg, rows = rows_fixture()
    row = next(row for row in rows if row['arm'] == 'householder' and row['learning_rate'] == .001
               and row['horizon'] == 128)
    row.update(status='FAILED', metrics=None, error={'type': 'FailedTrainingAttempt'})
    selection, result = audit.decisions(rows, resources_fixture(), cfg)
    assert selection['selected_rates']['householder'] == .003
    assert selection['options']['householder'][0]['eligible'] is False
    assert result['passed'] == 61


def test_accuracy_failures_remain_failed_despite_passing_costs():
    cfg, rows = rows_fixture()
    for row in rows:
        if row['arm'] == 'householder':
            row['metrics'].update(standardized_rmse=1.03, standardized_sse=1.03**2*120,
                                  physical_rmse_deg=1.03, per_joint_rmse_deg=[1.03]*6)
    _, result = audit.decisions(rows, resources_fixture(), cfg)
    # Five mean comparisons plus two simple references fail on each recording.
    assert result['passed'] == 47 and result['total'] == 61
    failed = [name for name, passed in conditions(result).items() if not passed]
    assert len(failed) == 14 and all('/mean_' in name for name in failed)


@pytest.mark.parametrize('name', ['at_most_80pct_dense_bounded_latency',
                                'at_most_80pct_dense_bounded_numeric_storage'])
def test_cost_failure_cannot_be_rescued_by_accuracy(name):
    cfg, rows = rows_fixture()
    resources = resources_fixture()
    for row in resources:
        if row['arm'] == 'householder':
            if name.endswith('latency'):
                row['timing']['median_seconds'] = .00801
            else:
                row['buffer_bytes'] = 12
    _, result = audit.decisions(rows, resources, cfg)
    assert result['passed'] == 60 and result['total'] == 61
    assert [key for key, passed in conditions(result).items() if not passed] == [name]


@pytest.mark.parametrize('damage', ['missing', 'duplicate'])
def test_incomplete_or_duplicate_scalar_grid_rejected(damage):
    cfg, rows = rows_fixture()
    if damage == 'missing':
        rows.pop()
    else:
        rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match='complete unique metric roster'):
        audit.decisions(rows, resources_fixture(), cfg)


def test_saved_scalar_comparison_has_explicit_absolute_and_relative_limits():
    audit.close({'x': .1 + .2}, {'x': .3}, 'ordinary roundoff')
    audit.close(0., 1e-12, 'absolute boundary')
    with pytest.raises(ValueError, match='scalar'):
        audit.close(0., np.nextafter(1e-12, math.inf).item(), 'beyond absolute')
    audit.close(1., 1. + .9e-10, 'relative inside')
    with pytest.raises(ValueError, match='scalar'):
        audit.close(1., 1. + 1.1e-10, 'relative outside')


@pytest.mark.parametrize('left,right', [(8101, 8101.), ('PASS', 'FAILED'), (True, 1),
                                      ({'seed': 8101}, {'other': 8101}), ([1, 2], [2, 1])])
def test_identity_and_schema_do_not_receive_numeric_tolerance(left, right):
    with pytest.raises(ValueError):
        audit.close(left, right, 'identity')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_saved_scalar_is_never_equal(value):
    with pytest.raises(ValueError, match='scalar'):
        audit.close(value, value, 'nonfinite')
