"""Fabricated benchmark plumbing only; never load real checkpoints or a library."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_robot_native as bench


def fits_fixture():
    return [{'key': f'{arm}-{seed}-lr{ri}', 'arm': arm, 'seed': seed,
             'learning_rate': rate, 'origin': 'fresh' if arm in bench.ARMS else 'cached_parent',
             'fit': {'status': 'PASS', 'completed_updates': 4096}}
            for arm in (*bench.ARMS, 'legacy_instant') for seed in bench.SEEDS
            for ri, rate in enumerate((.001, .003))]


def test_exact_selected_roster_keeps_every_family_seed_and_original_rate():
    fits = fits_fixture()
    rates = {arm: .003 if i % 2 else .001 for i, arm in enumerate(bench.ARMS)}
    chosen = bench.selected_fits(fits, rates)
    assert len(chosen) == 15
    assert [(r['arm'], r['seed']) for r in chosen] == [(a, s) for a in bench.ARMS for s in bench.SEEDS]
    assert all(r['learning_rate'] == rates[r['arm']] for r in chosen)


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'failed', 'incomplete', 'cached', 'none_rate', 'key'])
def test_bad_rosters_cannot_silently_reduce_native_comparison(damage):
    fits = fits_fixture()
    rates = dict.fromkeys(bench.ARMS, .001)
    if damage == 'duplicate': fits[-1] = copy.deepcopy(fits[0])
    elif damage == 'missing': fits.pop()
    elif damage == 'failed': fits[0]['fit']['status'] = 'FAILED'
    elif damage == 'incomplete': fits[0]['fit']['completed_updates'] = 4095
    elif damage == 'cached': fits[0]['origin'] = 'cached_parent'
    elif damage == 'none_rate': rates[bench.ARMS[0]] = None
    else: fits[0]['key'] = 'different'
    with pytest.raises(ValueError): bench.selected_fits(fits, rates)


def test_inputs_have_exact_torque_alignment_without_future_position_extraction():
    record = {'q': np.arange(200*6, dtype=np.float64).reshape(200, 6),
              'u': np.arange(200*6, dtype=np.float64).reshape(200, 6)+10000}
    starts = np.array([3], dtype=np.int64)
    result = bench.physical_contexts(record, starts)
    assert set(result) == {'q_context', 'u_context', 'future_u'}
    assert np.array_equal(result['q_context'][0], record['q'][3:35])
    assert np.array_equal(result['future_u'][0], record['u'][34:162])
    changed = {k: v.copy() for k, v in record.items()}
    changed['q'][35:] += 999
    other = bench.physical_contexts(changed, starts)
    assert all(np.array_equal(result[k], other[k]) for k in result)
    assert all(not np.shares_memory(v, record['q']) and not np.shares_memory(v, record['u']) for v in result.values())


def test_selected_array_load_retains_layout_and_does_not_decode_target(tmp_path):
    path = tmp_path / 'arrays.npz'
    value = np.arange(150, dtype=np.float32).reshape(6, 25).copy(order='F')
    np.savez(path, coefficient=value, target=np.array([object()], dtype=object))
    inputs = {str(path): bench.pin(path)}
    loaded = bench.load_pinned_arrays(path, inputs, ('coefficient',))['coefficient']
    assert loaded.flags.f_contiguous and not loaded.flags.c_contiguous
    assert np.array_equal(loaded, value) and not np.shares_memory(loaded, value)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='authenticated'):
        bench.load_pinned_arrays(path, inputs, ('coefficient',))


def test_standardized_check_catches_error_hidden_by_large_physical_offset():
    expected = np.zeros((1, 128, 6), dtype=np.float64)
    actual = expected + .01
    assert not bench.comparison(actual, expected)['passed']
    assert bench.comparison(actual+100000., expected+100000.)['passed']


@pytest.mark.parametrize('batch1_error', [False, True])
def test_both_execution_shapes_and_both_native_states_are_checked(monkeypatch, batch1_error):
    physical = {'q_context': np.ones((22, 32, 6)), 'u_context': np.zeros((22, 32, 6)),
                'future_u': np.ones((22, 128, 6))}
    norm = {k: np.ones(6) if k.endswith('std') else np.zeros(6)
            for k in ('q_mean', 'q_std', 'u_mean', 'u_std')}
    calls = []
    def reference(model, data, normalizers):
        n = len(data['q_context'])
        return {'standardized': np.ones((n, 128, 6)), 'prediction': np.ones((n, 128, 6)),
                'final_state': np.ones((n, 12), dtype=np.float32)}
    class Native:
        def request(self, q, u, future, normalizers):
            n = len(q)
            calls.append(n)
            state = np.ones((n, 12), dtype=np.float32)
            if n == 1 and batch1_error: state[:, -1] += .1
            return {'prediction': np.ones((n, 128, 6)), 'final_state': state, 'work': {'ffi_calls': 1}}
    monkeypatch.setattr(bench, 'torch_request', reference)
    arrays, record = bench.parity_pair(None, Native(), physical, norm, lambda: None)
    assert calls == [22, 22, 1, 1] and len(arrays) == 14 and len(record['checks']) == 8
    assert record['passed'] is (not batch1_error)
    assert record['checks']['batch22/physical_final_state']['passed']
    assert record['checks']['batch1/physical_final_state']['passed'] is (not batch1_error)
    assert record['checks']['batch1/standardized_final_state']['passed'] is (not batch1_error)


def test_parity_tolerance_and_nonfinite_records_are_json_finite():
    expected = np.array([0., 1.], dtype=np.float64)
    assert bench.comparison(expected + np.array([.9e-5, 1.9e-5]), expected)['passed']
    assert not bench.comparison(expected + np.array([1.1e-5, 2.1e-5]), expected)['passed']
    record = bench.comparison(np.array([np.nan, np.inf]), expected)
    assert record['nonfinite'] == record['violations'] == 2 and record['max_absolute_error'] is None
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('damage', ['none', 'failure', 'missing', 'duplicate'])
def test_all30_gate_is_required_before_any_timing(damage):
    rows = [{'fit_key': str(i), 'recording': str(j), 'passed': True} for i in range(15) for j in range(2)]
    if damage == 'none':
        bench.require_all_parity(rows)
        return
    if damage == 'failure': rows[-1]['passed'] = False
    elif damage == 'missing': rows.pop()
    else: rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match='all30'): bench.require_all_parity(rows)


def test_timing_keeps_every_warmup_pair_order_duration_and_median():
    calls, checks = [], []
    ticks = iter(np.arange(132, dtype=np.float64)+1)
    rows = bench.timed_pairs(lambda: calls.append('torch'), lambda: calls.append('native'),
                             lambda: checks.append(True), clock=lambda: next(ticks))
    assert len(rows) == 33 and len(checks) == 132 and len(calls) == 66
    assert [r['phase'] for r in rows] == ['warmup']*3+['timed']*30
    for group in (rows[:3], rows[3:]):
        assert all(r['order'] == (['torch', 'native'] if i % 2 == 0 else ['native', 'torch']) for i, r in enumerate(group))
    summary = bench.summarize_timings(rows)
    assert summary['median_seconds'] == {'torch': 1., 'native': 1.}
    assert summary['torch_over_native'] == 1 and summary['paired_ratios'] == [1.]*30


def test_failed_request_preserves_completed_first_half_of_pair():
    ticks = iter([1., 2., 3.])
    error = RuntimeError('fabricated failure')
    def fail(): raise error
    with pytest.raises(RuntimeError) as caught:
        bench.timed_pairs(lambda: None, fail, lambda: None, clock=lambda: next(ticks))
    assert caught.value is error
    assert error.timing_records == [{'phase': 'warmup', 'repetition': 0,
                                    'order': ['torch', 'native'], 'seconds': {'torch': 1.}}]


def test_torch_whole_request_uses_current_torque_and_normalizes_inside_call():
    import torch
    class Model:
        def condition(self, q, u):
            assert q.dtype == u.dtype == torch.float32
            return q[:, -1]
        def __call__(self, future, state):
            return state[:, None]+future, state
    physical = {'q_context': np.full((2, 32, 6), 12.), 'u_context': np.full((2, 32, 6), 3.),
                'future_u': np.full((2, 128, 6), 5.)}
    norm = {'q_mean': np.full(6, 10.), 'q_std': np.full(6, 2.),
            'u_mean': np.full(6, 3.), 'u_std': np.full(6, 4.)}
    result = bench.torch_request(Model(), physical, norm)
    assert np.array_equal(result['standardized'], np.full((2, 128, 6), 1.5))
    assert np.array_equal(result['prediction'], np.full((2, 128, 6), 13.))
    assert np.array_equal(result['final_state'], np.ones((2, 6), dtype=np.float32))


def test_failed_admission_preserved_without_array_decode(tmp_path, monkeypatch):
    def fail(*args, **kwargs): raise ValueError('fabricated admission failure')
    monkeypatch.setattr(bench.proof, 'authenticate', fail)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('must not decode before admission'))
    output = tmp_path / 'attempt'
    result = bench.benchmark(tmp_path / 'study', None, None, tmp_path / 'qual.json', output)
    assert result['status'] == 'FAILED'
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['parity'] == summary['timings'] == []
    assert summary['error']['message'] == 'fabricated admission failure'
    assert (output / 'manifest.json').is_file() and (output / 'receipt.json').is_file()
    with pytest.raises(FileExistsError):
        bench.benchmark(tmp_path / 'study', None, None, tmp_path / 'qual.json', output)
