"""Synthetic qualification helpers only; no HDF5, TensorFlow or released tensors."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_pretrained_value as PORT

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/qualify_otto_pretrained.py'
SPEC = importlib.util.spec_from_file_location('qualify_otto_pretrained_fixture', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def kernels():
    first = np.zeros((4, 107, 107), dtype=np.float64)
    first[0], first[1], first[2], first[3] = .1, .2, .3, .4
    second = first[::-1].copy()
    return {'base': first, 'shift': second}


def test_fixed_fixture_population_is_reproducible_and_paired():
    values = M.policy_fixtures(kernels(), np)
    assert len(values) == 36
    assert len({v['id'] for v in values}) == 36
    for left, right in zip(values[:16], values[16:32], strict=True):
        assert left['position'] == right['position']
        np.testing.assert_array_equal(left['belief'], right['belief'])
        assert left['belief'][left['position']] == 0
        assert abs(left['belief'].sum() - 1) < 1e-14
    repeat = M.policy_fixtures(kernels(), np)
    for a, b in zip(values, repeat, strict=True):
        np.testing.assert_array_equal(a['belief'], b['belief'])
        np.testing.assert_array_equal(a['kernel'], b['kernel'])


@pytest.mark.parametrize('index,expected_mass', [(0, 0.), (1, .5), (2, 1.), (3, 1.)])
def test_mechanical_fixtures_isolate_each_side_of_branch_floor(index, expected_mass):
    case = M.policy_fixtures(kernels(), np)[32 + index]
    inputs, masses = PORT.policy_inputs(case['belief'], case['position'], case['kernel'])
    np.testing.assert_allclose(inputs[::4].sum(axis=(1, 2)), expected_mass, atol=0, rtol=0)
    np.testing.assert_array_equal(masses[:, 0], np.full(4, max(case['mass'], 1e-10), dtype=np.float32))
    if index == 0:
        assert not inputs[::4].any()
        assert (masses[:, 0] > 0).all()


def test_raw_inputs_cover_zero_subnormalized_and_distinct_d4_orbit():
    names, raw = M.raw_fixtures(np)
    assert len(names) == len(set(names)) == 16
    assert raw.shape == (16, 105, 105) and raw.dtype == np.float32
    assert np.isfinite(raw).all() and (raw >= 0).all()
    assert not raw[0].any()
    assert raw[1].sum() == .5
    assert len({x.tobytes() for x in raw[6:14]}) == 8
    assert all(abs(x.sum(dtype=np.float64) - 1) < 1e-6 for x in raw[6:14])
    np.testing.assert_array_equal(raw, M.raw_fixtures(np)[1])


@pytest.mark.parametrize('position', M.POSITIONS)
def test_fake_environment_centering_and_boundary_are_independent_exact_helpers(position):
    p = np.arange(1, 2810, dtype=np.float64).reshape(53, 53)
    p /= p.sum()
    env = M.SyntheticEnvironment(p, position, kernels()['base'], np)
    actual = env._centeragent(p, position)
    assert actual.shape == (105, 105)
    for i, j in ((0, 0), (17, 28), (52, 52)):
        assert actual[52 + i - position[0], 52 + j - position[1]] == p[i, j]
    assert np.count_nonzero(actual) == p.size
    np.testing.assert_array_equal(actual.astype(np.float32), PORT.center_beliefs(p[None], [position])[0])
    for action in range(4):
        moved, possible = env._move(action, list(position))
        delta = -1 if action % 2 == 0 else 1
        expected = list(position)
        axis = action // 2
        if 0 <= expected[axis] + delta < 53:
            expected[axis] += delta
        assert moved == expected and possible is (moved != list(position))
    assert not hasattr(env, 'step') and not hasattr(env, 'source') and not hasattr(env, 'reward')


def test_kernel_crop_uses_absolute_displacement_not_agent_center_roll():
    k = np.arange(4 * 107 * 107, dtype=np.float64).reshape(4, 107, 107)
    env = M.SyntheticEnvironment(np.zeros((53, 53)), (0, 0), k, np)
    crop = env._extract_N_from_2N(k, [7, 41])
    assert crop.shape == (4, 53, 53)
    assert crop[2, 11, 33] == k[2, 53 - 7 + 11, 53 - 41 + 33]


def test_numeric_rule_has_no_epsilon_beyond_frozen_tolerance():
    zero = np.array([0.], dtype=np.float64)
    bound = np.array([1e-4], dtype=np.float64)
    assert M.compare_arrays(bound, zero, np)['passed']
    assert not M.compare_arrays(np.nextafter(bound, np.inf), zero, np)['passed']
    assert M.compare_arrays(np.array([10000.1]), np.array([10000.]), np)['passed']
    assert not M.compare_arrays(np.array([10000.2]), np.array([10000.]), np)['passed']


def test_exact_comparison_catches_dtype_and_signed_zero_bytes():
    a = np.zeros(2, dtype=np.float32)
    assert M.compare_arrays(a.copy(), a, np, exact=True)['passed']
    b = a.copy()
    b[0] = -0.
    assert not M.compare_arrays(b, a, np, exact=True)['passed']
    assert not M.compare_arrays(a.astype(np.float64), a, np, exact=True)['passed']
    assert not M.compare_arrays(np.array([np.nan]), np.array([1.]), np)['passed']
    assert not M.compare_arrays(a[None], a, np)['passed']


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def test_recording_wrapper_binds_actual_caller_and_copies_original_mass():
    def original_call(recorder):
        probs = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        inputs = FakeTensor(np.zeros((16, 105, 105), dtype=np.float32))
        returned = recorder(inputs, sym_avg=True)
        probs[:] = -1
        return returned

    def fake_model(inputs, *, training, sym_avg):
        assert training is False and sym_avg is True
        return FakeTensor(np.zeros((16, 1), dtype=np.float32))

    recorder = M.RecordingModel(fake_model, original_call.__code__, np)
    original_call(recorder)
    assert len(recorder.records) == 1
    np.testing.assert_array_equal(recorder.records[0]['masses'], np.arange(16, dtype=np.float32).reshape(4, 4) / 16)
    with pytest.raises(ValueError, match='caller identity'):
        recorder(FakeTensor(np.zeros((16, 105, 105))), sym_avg=True)


def comparison_rows():
    names, _ = M.raw_fixtures(np)
    weights = [{'id': f'{kind}_{i}', 'passed': True} for i in range(4) for kind in ('kernel', 'bias')]
    values = [{'passed': True, 'sym_avg': sym, 'batch_size': size, 'offset': offset,
               'ids': names[offset:offset + size]} for sym in (False, True)
              for size in (1, 3, 16) for offset in range(0, 16, size)]
    cases = M.policy_fixtures(kernels(), np)
    policies = [{'id': c['id'], 'kind': c['kind'], 'passed': True, 'action_equal': True,
                 'comparisons': {k: {'passed': True} for k in ('inputs', 'masses', 'numpy_policy_inputs', 'values', 'scores')}}
                for c in cases]
    return weights, values, policies


def test_all_comparisons_required_and_value_only_pass_never_admits_policy():
    rows = comparison_rows()
    result = M.summarize(*rows)
    assert result['qualified'] and result['raw_value_predictions_compared'] == 96
    rows[2][0].update(action_equal=False, passed=False)
    result = M.summarize(*rows)
    assert result['value_parity'] and not result['qualified']
    assert result['action_disagreements'] == [rows[2][0]['id']]
    assert result['near_tie_exemptions'] is False


@pytest.mark.parametrize('group', ['weight', 'raw_value', 'inputs', 'masses', 'numpy_policy_inputs', 'values', 'scores'])
def test_each_independent_failure_is_retained_without_stopping_summary(group):
    weights, values, policies = comparison_rows()
    if group == 'weight':
        weights[3]['passed'] = False
    elif group == 'raw_value':
        values[17]['passed'] = False
    else:
        policies[35]['comparisons'][group]['passed'] = False
        policies[35]['passed'] = False
    result = M.summarize(weights, values, policies)
    assert not result['qualified'] and result['policy_fixtures_compared'] == 36


@pytest.mark.parametrize('group', [0, 1, 2])
def test_missing_or_duplicate_comparison_is_never_silently_accepted(group):
    rows = list(comparison_rows())
    rows[group].pop()
    with pytest.raises(ValueError, match='counts'):
        M.summarize(*rows)
    rows = list(comparison_rows())
    rows[group][1] = copy.deepcopy(rows[group][0])
    with pytest.raises(ValueError, match='identit|route'):
        M.summarize(*rows)


def test_bad_plan_pin_rejects_before_runtime_metadata_or_payload_decode(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    monkeypatch.setattr(M.importlib.metadata, 'version', lambda name: pytest.fail('runtime read before external pin'))
    with pytest.raises(ValueError, match='external plan pin'):
        M.authenticate(SimpleNamespace(plan=path, plan_sha256='0' * 64))


def test_exclusive_output_collision_never_binds_or_modifies_existing_file(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    marker = out / 'existing'
    marker.write_text('unchanged')
    run = M.Run(argparse.Namespace(output=out))
    monkeypatch.setattr(run, 'bind', lambda: pytest.fail('bound colliding output'))
    with pytest.raises(FileExistsError):
        run.execute()
    assert marker.read_text() == 'unchanged' and list(out.iterdir()) == [marker]


def test_primary_failure_survives_secondary_clock_failure_with_receipt(tmp_path, monkeypatch):
    run = M.Run(argparse.Namespace(output=tmp_path / 'out'))

    def fail_bind():
        def bad_clock():
            raise RuntimeError('clock unavailable')
        run.clock, run.start = SimpleNamespace(now_ns=bad_clock), 0
        raise ValueError('primary pin failure')

    monkeypatch.setattr(run, 'bind', fail_bind)
    with pytest.raises(ValueError, match='primary pin failure'):
        run.execute()
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and 'primary pin failure' in receipt['error']
    assert 'clock unavailable' in receipt['failure_clock_error']
    assert receipt['simulator_calls'] == 0


def test_call_journal_preserves_completed_work_and_uncertain_pending_call(tmp_path):
    work = M.WorkLedger(tmp_path)
    work.context('raw_value', 'fixture_0')

    def successful_call():
        journal = [json.loads(line) for line in work.path.read_text().splitlines()]
        assert journal[-1]['event'] == 'call_attempt'
        assert journal[-1]['calls']['tensorflow_value'] == {'attempted': 1, 'returned': 0}
        return 17

    assert work.call('tensorflow_value', successful_call) == 17
    work.completed('value', {'id': 'fixture_0', 'passed': False})
    work.context('raw_value', 'fixture_1')

    def interrupted_call():
        raise RuntimeError('backend interruption')

    with pytest.raises(RuntimeError, match='backend interruption'):
        work.call('tensorflow_value', interrupted_call)
    assert work.state['calls']['tensorflow_value'] == {'attempted': 2, 'returned': 1}
    assert work.state['pending_call'] == {'channel': 'tensorflow_value', 'ordinal': 2,
                                          'phase': 'raw_value', 'case': 'fixture_1'}
    assert work.state['completed_comparisons'] == {'weight': 0, 'value': 1, 'policy': 0}
    assert [json.loads(line) for line in (tmp_path / 'value-checks.jsonl').read_text().splitlines()] == [
        {'id': 'fixture_0', 'passed': False}]
    with pytest.raises(ValueError, match='pending call'):
        work.context('policy')


def test_failure_receipt_includes_exact_work_counters(tmp_path, monkeypatch):
    run = M.Run(argparse.Namespace(output=tmp_path / 'out'))

    def fail_bind():
        run.work.context('model_construction')
        run.work.call('tensorflow_construction', lambda: object())
        run.work.call('tensorflow_build', lambda: None)

        def failed_load():
            raise RuntimeError('synthetic load failure')
        run.work.call('tensorflow_load', failed_load)

    monkeypatch.setattr(run, 'bind', fail_bind)
    with pytest.raises(RuntimeError, match='synthetic load failure'):
        run.execute()
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['work']['calls']['tensorflow_build'] == {'attempted': 1, 'returned': 1}
    assert receipt['work']['calls']['tensorflow_load'] == {'attempted': 1, 'returned': 0}
    assert receipt['work']['calls']['tensorflow_value'] == {'attempted': 0, 'returned': 0}
    assert receipt['work']['pending_call']['channel'] == 'tensorflow_load'
