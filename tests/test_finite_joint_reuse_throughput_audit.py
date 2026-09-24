"""Fabricated saved-evidence oracles; no model or world-generator calls."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_joint_reuse_throughput as a


def records(ratio=1.2):
    return [{**row, 'call_seconds': ratio if row['implementation'] == 'separate' else 1.}
            for row in a.schedule()]


def test_independent_schedule_retains_all_pairs_and_declared_order():
    rows = a.schedule()
    assert len(rows) == len({row['run_id'] for row in rows}) == 24
    assert sum(row['warmup'] for row in rows) == 6
    assert [(row['arm'], row['implementation']) for row in rows[:6]] == [
        ('original_free', 'separate'), ('original_free', 'reuse'),
        ('matched_free', 'reuse'), ('matched_free', 'separate'),
        ('rounded', 'separate'), ('rounded', 'reuse')]
    assert [rows[i]['arm'] for i in (6, 12, 18)] == ['original_free', 'matched_free', 'rounded']
    assert len(a.expected_files()) == 247
    assert all(len(a.fit_files(arm, 943301)) == 10 for arm in a.ARMS)


def test_all_nine_ratios_and_each_arm_median_are_required():
    rows = records()
    # A slow warmup remains reported but is prospectively excluded from the gate.
    rows[0]['call_seconds'] = .01
    report = a.timing_gate(rows)
    assert report['passed'] and len(report['conditions']) == 12
    assert report['median_ratios'] == dict.fromkeys(a.ARMS, 1.2)
    assert len(report['paired']) == 12 and sum(not r['warmup'] for r in report['paired']) == 9
    assert report['warmup_call_seconds'] == pytest.approx(5.41)
    for row in rows:
        if row['round'] == 2 and row['arm'] == 'rounded' and row['implementation'] == 'separate':
            row['call_seconds'] = 1.
    result = a.timing_gate(rows)
    assert not result['passed']
    assert result['median_ratios']['rounded'] == 1.2  # Mean/median cannot rescue the tie.
    assert result['conditions']['rounded_round2_ratio_gt1'] is False


def test_exact_median_threshold_is_not_success():
    result = a.timing_gate(records(1.05))
    assert not result['passed']
    assert all(value for key, value in result['conditions'].items() if 'round' in key)
    assert not any(value for key, value in result['conditions'].items() if 'median' in key)


@pytest.mark.parametrize('mutation', [
    lambda rows: rows.pop(),
    lambda rows: rows.__setitem__(3, copy.deepcopy(rows[2])),
    lambda rows: rows.reverse(),
    lambda rows: rows[7].update(warmup=True),
    lambda rows: rows[7].update(call_seconds=0.),
    lambda rows: rows[7].update(call_seconds=float('nan')),
    lambda rows: rows[7].update(call_seconds=True),
])
def test_timing_roster_and_positive_duration_corruptions_fail(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        a.timing_gate(rows)


def test_symmetric_final_parity_and_exact_signed_zero_boundaries():
    x = np.array([0., 2., -3.], np.float64)
    y = x + np.array([.5e-7, 2e-7, -3e-7], np.float64)
    report = a.compare_arrays(x, y, np)
    assert report == a.compare_arrays(y, x, np)
    assert 0 < report['maximum_tolerance_fraction'] < 1
    with pytest.raises(ValueError, match='exact boundary'):
        a.compare_arrays(x, y, np, exact=True)
    positive = np.array([0.], np.float64)
    negative = np.array([-0.], np.float64)
    a.compare_arrays(positive, negative, np)
    with pytest.raises(ValueError, match='exact boundary'):
        a.compare_arrays(positive, negative, np, exact=True)


@pytest.mark.parametrize('other', [
    np.array([1e-6], np.float64), np.array([np.inf], np.float64),
    np.array([np.nan], np.float64), np.array([0.], np.float32), np.zeros((1, 1), np.float64),
])
def test_final_array_guard_does_not_treat_nonfinite_or_wrong_dtype_as_parity(other):
    with pytest.raises(ValueError):
        a.compare_arrays(np.array([0.], np.float64), other, np)


def adam():
    tensor = lambda values: {'kind': 'tensor', 'dtype': 'torch.float64', 'shape': [2], 'values': values}
    return {'param_groups': [{'params': [0], 'lr': .003}], 'state': {'0': {
        'step': {'kind': 'tensor', 'dtype': 'torch.float32', 'shape': [], 'values': 64.},
        'exp_avg': tensor([.2, -.3]), 'exp_avg_sq': tensor([.1, .4])}}}


def test_adam_moments_receive_tolerance_but_steps_and_groups_remain_exact():
    first, second = adam(), adam()
    second['state']['0']['exp_avg']['values'][0] += 1e-8
    assert a.compare_optimizer(first, second, np, exact=False)['maximum_absolute_error'] > 0
    with pytest.raises(ValueError):
        a.compare_optimizer(first, second, np, exact=True)
    second = adam()
    second['state']['0']['step']['values'] = 63.
    with pytest.raises(ValueError, match='step'):
        a.compare_optimizer(first, second, np, exact=False)
    second = adam()
    second['param_groups'][0]['lr'] = .004
    with pytest.raises(ValueError, match='groups'):
        a.compare_optimizer(first, second, np, exact=False)


def small_prefix():
    # Three independent fabricated histories: survivor, found at step1, found at step3.
    prefix = np.zeros((3, 9, 31), np.float32)
    lengths = np.array([9, 2, 4], np.int64)
    for index, length in enumerate(lengths):
        prefix[index, 0, 4] = prefix[index, 0, 9] = 1
        for step in range(1, length):
            prefix[index, step, 0] = 1
            prefix[index, step, 8 if index and step == length - 1 else 4] = 1
    return {'prefix': prefix, 'lengths': lengths}


@pytest.mark.parametrize('arm', a.ARMS)
def test_disjoint_reuse_operation_oracle_counts_shared_fields_only_once(arm):
    prefix = small_prefix()
    # Two endpoint lanes are allowed here solely to test standalone geometry.
    observations = np.array([[0, 4], [4, 4]], np.int64)
    blocks = a.reuse_work(arm, prefix, np.arange(3), observations, np)
    shared, endpoint, nll, blind, observed = (blocks[name] for name in a.REUSE_ROUTES)
    assert all(set(block) == a.WORK_KEYS for block in blocks.values())
    assert shared['factorization_calls'] == shared['probability_field_calls'] == 1
    assert shared['operator_marginal_sum_calls'] == 1
    assert (endpoint['reset_emission_rows'], endpoint['prefix_filter_calls'], endpoint['prefix_filter_rows']) == (2, 8, 16)
    assert (nll['reset_emission_rows'], nll['prefix_probability_rows'], nll['prefix_nll_rows']) == (3, 15, 15)
    assert nll['prefix_filter_calls'] == 8 and nll['prefix_filter_rows'] == 10
    assert blind['blind_transition_calls'] == observed['blind_transition_calls'] == 2
    assert blind['blind_transition_rows'] == observed['blind_transition_rows'] == 4
    assert observed['observed_conditioning_rows'] == observed['absorbed_rows'] == 1
    assert observed['event_probability_rows'] == 4
    assert blind['cost_head_softmax_calls'] == observed['cost_head_softmax_calls'] == 2
    assert sum(block['probability_field_calls'] for block in blocks.values()) == 1
    assert all(block['rounded_transition_calls'] == 0 for block in (endpoint, nll, blind, observed))
    assert shared['rounded_transition_calls'] == int(arm == 'rounded')
    assert sum(block['prior_transition_log_softmax_calls'] for block in blocks.values()) == 0


def test_empty_endpoint_still_has_shared_fields_and_all_attempt_nll():
    blocks = a.reuse_work('rounded', small_prefix(), np.arange(3), np.empty((0, 2), np.int64), np)
    assert blocks['joint_reuse_shared']['factorization_calls'] == 1
    assert blocks['joint_reuse_shared']['operator_marginal_sum_calls'] == 0
    assert blocks['joint_reuse_prefix_nll']['prefix_nll_rows'] == 15
    for key in ('joint_reuse_endpoint_prefix', 'joint_reuse_blind', 'joint_reuse_observed'):
        assert not any(blocks[key].values())


def test_original_closure_failure_precedes_any_numpy_load(monkeypatch, tmp_path):
    calls = []
    def rejected(_study):
        raise ValueError('original producer did not close')
    def forbidden(*args, **kwargs):
        calls.append('decode')
        raise AssertionError('premature decode')
    monkeypatch.setattr(a, 'admit', rejected)
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='did not close'):
        a.audit(tmp_path)
    assert calls == []


def test_incomplete_inventory_is_rejected_before_numeric_import_path(tmp_path):
    summary = {'version': a.PRODUCER_VERSION, 'status': 'PASS', 'config': a.SPEC.config,
               'schedule': a.schedule(), 'scientific_admission': False}
    with pytest.raises(ValueError, match='payload roster'):
        a.validate_metadata(tmp_path, summary)
