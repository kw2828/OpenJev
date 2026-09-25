"""Fabricated saved-output arithmetic only; no science data or model calls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/audit_fsm_correction.py'
spec = importlib.util.spec_from_file_location('audit_fsm_correction', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    fits, evaluations = [], []
    for family in audit.FAMILIES:
        for seed in audit.SEEDS if family in audit.NEURAL else (None,):
            fits.append({'family': family, 'seed': seed, 'status': 'complete', 'accepted_updates': 1024})
            evaluations.append({'family': family, 'seed': seed,
                                'rows': [{'record_id': r, 'status': 'complete', 'rmse': 8. if family == 'selective' else 10.} for r in audit.DEV],
                                'timing_error': None, 'median_request_ms': 1.,
                                'persistent_numeric_bytes': 100,
                                'routing_nonzero_counts': [90, 5, 5] if family == 'selective' else None})
    return fits, evaluations


def test_scalar_metrics_and_common_reductions():
    p = np.broadcast_to(np.array([1., 2., 3.]), (32, 128, 3)).copy()
    actual = audit.metrics(p, np.zeros_like(p))
    assert actual['mse'] == 14/3
    assert actual['rmse'] == np.sqrt(14/3)
    assert actual['per_channel_rmse'] == [1., 2., 3.]
    assert (actual['requests'], actual['horizon']) == (32, 128)


@pytest.mark.parametrize('kind', ['nan', 'overflow', 'shape', 'dtype'])
def test_bad_forecast_rejected(kind):
    p = np.ones((32, 128, 3))
    if kind == 'nan':
        p[0, 0, 0] = np.nan
    elif kind == 'overflow':
        p.fill(1e308)
    elif kind == 'shape':
        p = p[:1]
    else:
        p = p.astype(np.float32)
    with pytest.raises(ValueError):
        audit.metrics(p, np.zeros_like(p))


def test_all_nine_rules_and_equal_record_mean():
    fits, evaluations = fixture()
    actual = audit.decisions(evaluations, fits)
    assert actual['status'] == 'DEVELOPMENT_PASS'
    assert actual['passed'] == actual['total'] == 9
    assert actual['strongest_control'] == 'dense'
    assert actual['families']['selective']['mean_rmse'] == 8.
    assert len(actual['families']) == 16


@pytest.mark.parametrize('mutation,failed', [
    ('fit', 'all_declared_fits_complete'), ('absent', 'all_declared_evaluations_complete'),
    ('seed', 'candidate_eligible'), ('timing', 'latency_within_ten_percent_dense'),
    ('storage', 'storage_within_ten_percent_dense'), ('record', 'no_record_over_two_percent_dense'),
    ('routing', 'two_blocks_at_least_five_percent_each_seed'),
    ('zero_routes', 'two_blocks_at_least_five_percent_each_seed'),
    ('strong_control', 'five_percent_below_strongest_control'), ('paired', 'every_seed_beats_dense'),
])
def test_failure_never_filtered(mutation, failed):
    fits, evaluations = fixture()
    candidate = [e for e in evaluations if e['family'] == 'selective']
    if mutation == 'fit':
        fits[-1]['status'] = 'failed'
    elif mutation == 'absent':
        evaluations.pop()
    elif mutation == 'seed':
        candidate[-1]['seed'] = candidate[0]['seed']
    elif mutation == 'timing':
        for e in candidate:
            e['median_request_ms'] = 1.100001
    elif mutation == 'storage':
        for e in candidate:
            e['persistent_numeric_bytes'] = 111
    elif mutation == 'record':
        for e in candidate:
            e['rows'][0]['rmse'] = 10.200001
    elif mutation in ('routing', 'zero_routes'):
        candidate[0]['routing_nonzero_counts'] = [100, 0, 0] if mutation == 'routing' else [0, 0, 0]
    elif mutation == 'strong_control':
        for r in evaluations[-1]['rows']:
            r['rmse'] = 7.
    else:
        for r in candidate[0]['rows']:
            r['rmse'] = 10.
    result = audit.decisions(evaluations, fits)
    assert result['conditions'][failed] is False
    assert result['status'] == 'DEVELOPMENT_FAIL'


def test_exact_cost_boundaries_and_strict_seed_condition():
    fits, evaluations = fixture()
    for e in evaluations:
        if e['family'] == 'selective':
            e['median_request_ms'] = 1.1
            e['persistent_numeric_bytes'] = 110
    assert audit.decisions(evaluations, fits)['passed'] == 9


def test_original_failure_rejected_before_study_read(tmp_path, monkeypatch):
    process = tmp_path/'process.json'
    process.write_text(json.dumps({'observed_exit_code': 1}))
    real_read = audit.read
    calls = []
    def safe_read(path):
        calls.append(Path(path))
        return real_read(path) if Path(path) == process else {}
    monkeypatch.setattr(audit, 'read', safe_read)
    with pytest.raises(ValueError, match='successful process'):
        audit.audit(tmp_path/'nonexistent-study', process)
    assert all('nonexistent-study' not in str(p) for p in calls)


def test_scalar_join_is_tight_and_identities_exact():
    audit.close({'value': 1.+5e-13, 'count': 1}, {'value': 1., 'count': 1})
    with pytest.raises(ValueError):
        audit.close({'value': 1.+1e-8}, {'value': 1.})
    with pytest.raises(ValueError):
        audit.close({'count': True}, {'count': 1})
