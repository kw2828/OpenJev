"""Fabricated capacity-runner contracts; no scientific inputs or model calls."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('capacity_study_test', ROOT / 'scripts/study_otto_capacity.py')
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def fitting_metrics():
    return {f'{kind}@{seed}': {'train': {'mse_normalized': 1. if kind == 'mlp8' else .8},
                              'valid': {'mse_normalized': .5 if kind == 'mlp8' else .45}}
            for seed in S.SEEDS for kind in S.KINDS}


def test_alias_floor_is_row_weighted_and_preserves_conflicting_repeats():
    features = np.array([[1, 0], [1, 0], [1, 0], [2, 0]], dtype=np.float32)
    target = np.array([0, 1, 2, 10], dtype=np.float32)
    result = S.alias_floor(features, target)
    assert result == {'rows': 4, 'unique_groups': 2, 'duplicate_groups': 1,
                      'duplicate_rows': 3, 'conflicting_groups': 1,
                      'irreducible_mse_normalized': .5}
    permutation = [3, 1, 0, 2]
    assert S.alias_floor(features[permutation], target[permutation]) == result
    assert np.array_equal(target, [0, 1, 2, 10])


def test_alias_groups_follow_exact_float32_bytes_including_signed_zero():
    features = np.array([[0., 0.], [0., -0.]], dtype=np.float32)
    result = S.alias_floor(features, np.array([0, 1], dtype=np.float32))
    assert result['unique_groups'] == 2
    assert result['irreducible_mse_normalized'] == 0.


@pytest.mark.parametrize('kind', ['mlp128', 'deep128'])
@pytest.mark.parametrize('seed', [10101, 10102, 10103])
@pytest.mark.parametrize('metric', ['train_excess', 'valid_mse'])
def test_all_twelve_exact_gates_reject_one_ulp_failure(kind, seed, metric):
    metrics = fitting_metrics()
    original = S.gates(metrics, 0.)
    assert len(original) == 12 and all(row['passes'] for row in original)
    split, threshold = ('train', .8) if metric == 'train_excess' else ('valid', .45)
    metrics[f'{kind}@{seed}'][split]['mse_normalized'] = math.nextafter(threshold, math.inf)
    failed = [row for row in S.gates(metrics, 0.) if not row['passes']]
    assert len(failed) == 1
    assert failed[0]['name'] == f'{kind}.{seed}.{metric}'


def test_zero_excess_and_negative_rounding_residual_are_not_clipped():
    metrics = fitting_metrics()
    for panels in metrics.values():
        panels['train']['mse_normalized'] = .5
    rows = S.gates(metrics, .5)
    assert all(row['passes'] for row in rows)
    metrics['mlp128@10101']['train']['mse_normalized'] = math.nextafter(.5, math.inf)
    assert not S.gates(metrics, .5)[0]['passes']
    metrics['mlp128@10101']['train']['mse_normalized'] = math.nextafter(.5, -math.inf)
    row = S.gates(metrics, .5)[0]
    assert row['value'] < 0 and row['threshold'] == 0 and row['passes']


def test_missing_fit_cannot_silently_pass_the_screen():
    metrics = fitting_metrics()
    del metrics['deep128@10103']
    with pytest.raises((ValueError, KeyError)):
        S.gates(metrics, 0.)


def test_qualification_geometry_work_and_payload_counts():
    qualification, study = S.configuration('qualify'), S.configuration('study')
    assert (qualification['rows'], qualification['batch_size'], qualification['epochs']) == (256, 128, 6)
    assert qualification['seeds'] == [10101]
    assert '256' in qualification['qualification_selection']
    assert study['rows'] == 5589 and study['batch_size'] == 128 and study['epochs'] == 80
    assert 3 * math.ceil(256 / 128) * 6 == S.limits('qualify')['optimizer_updates'] == 36
    assert 9 * math.ceil(5589 / 128) * 80 == S.limits('study')['optimizer_updates'] == 31680
    assert len(S.payload_names('qualify')) == 20
    assert len(S.payload_names('study')) == 38
    for mode in ('qualify', 'study'):
        assert all('eval' not in name and 'policy' not in name for name in S.payload_names(mode))
        assert S.limits(mode)['native_steps'] == S.limits(mode)['native_resets'] == 0


def test_qualification_loads_only_train_and_uses_full_train_baseline(tmp_path, monkeypatch):
    # Broadcast storage avoids allocating a full dense fabricated cache.
    n = 5589
    x = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 11028))
    y = (np.arange(n, dtype=np.float32) % 64 + 1) / 64
    rows = [{'row_index': i, 'episode_id': f'train-{i % 192}', 'stage': 'train',
             'public': {'done': False}, 'target': float(y[i]), 'total_steps': i % 64 + 1,
             'prefix_index': 0} for i in range(n)]
    row_path = tmp_path / 'train-rows.jsonl'
    row_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    opened = []

    class Cache:
        def __enter__(self):
            return {'features': x, 'target': y}

        def __exit__(self, *_args):
            return False

    def fake_load(path, *, allow_pickle):
        assert Path(path) == tmp_path / 'train-data.npz' and allow_pickle is False
        opened.append(Path(path).name)
        return Cache()

    fake_torch = SimpleNamespace(set_num_threads=lambda _n: None,
                                 set_num_interop_threads=lambda _n: None,
                                 use_deterministic_algorithms=lambda _enabled: None,
                                 get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
                                 are_deterministic_algorithms_enabled=lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(np, 'load', fake_load)
    monkeypatch.setattr(S, 'regular', lambda name: tmp_path / name)
    for name, value in S.THREADS.items():
        monkeypatch.setenv(name, value)
    # prepare inserts an import path; restore it after this synthetic call.
    monkeypatch.setattr(sys, 'path', list(sys.path))
    out = tmp_path / 'out'
    out.mkdir()
    runner = S.Run(SimpleNamespace(output=out))
    runner.plan = {'mode': 'qualify', 'inputs': {'train_data': {'path': 'train-data.npz'},
                                               'train_rows': {'path': 'train-rows.jsonl'}}}
    runner.prepare()
    expected = np.linspace(0, 5588, 256, dtype=np.int64)
    assert opened == ['train-data.npz'] and set(runner.data) == {'train'}
    assert runner.indices['train'] == expected.tolist()
    assert np.array_equal(runner.data['train'][0][:, 0], expected.astype(np.float32))
    assert np.array_equal(runner.data['train'][1], y[expected])
    assert runner.c0 == float(np.float32(np.mean(y, dtype=np.float64)))


@pytest.mark.parametrize(('mode', 'change'), [
    ('qualify', 'missing_train'), ('qualify', 'extra_eval'), ('qualify', 'extra_valid'),
    ('study', 'missing_train'), ('study', 'missing_valid'), ('study', 'extra_eval'),
])
def test_wrong_input_roles_fail_before_lineage_or_array_reads(tmp_path, monkeypatch, mode, change):
    roles = {'prior_plan', 'prior_receipt', 'prior_terminal', 'prior_audit_receipt', 'train_data', 'train_rows'}
    if mode == 'study':
        roles |= {'valid_data', 'valid_rows'}
    if change == 'missing_train':
        roles.remove('train_data')
    elif change == 'missing_valid':
        roles.remove('valid_rows')
    else:
        roles.add('eval_data' if change == 'extra_eval' else 'valid_data')
    sources = dict.fromkeys(S.NEW, 'source-pin') | {S.OLD: S.OLD_PIN}
    plan = {'version': S.VERSION, 'status': 'frozen_before_execution', 'mode': mode,
            'configuration': S.configuration(mode), 'limits': S.limits(mode), 'sources': sources,
            'inputs': {role: {} for role in roles}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    pin = S.sha(path)
    monkeypatch.setattr(S, 'sha', lambda p: pin if Path(p) == path else sources[str(p)])
    monkeypatch.setattr(S, 'regular', Path)

    def forbidden(*_args, **_kwargs):
        pytest.fail('wrong input roles crossed the authentication boundary')

    monkeypatch.setattr(S.P, 'authenticate', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError, match='exact input roles'):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256=pin))


def test_external_plan_mismatch_prevents_decoding(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text('deliberately not JSON')
    monkeypatch.setattr(S, 'read', lambda _path: pytest.fail('decoded an unauthenticated plan'))
    with pytest.raises(ValueError, match='external plan pin'):
        S.authenticate(SimpleNamespace(plan=path, plan_sha256='0' * 64))


def test_historical_and_current_command_order_are_preserved(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text('{}')
    receipt = tmp_path / 'run' / 'receipt.json'
    common = ['/python', '--plan', str(plan), '--plan-sha256', S.sha(plan)]
    for runner, tail in (
        (S.OLD, ['--supervision', '/launch', '--output', str(receipt.parent)]),
        ('scripts/study_otto_capacity.py', ['--output', str(receipt.parent), '--supervision', '/launch']),
    ):
        command = S.expected_command({'python_executable': '/python'}, plan, receipt, '/launch', runner)
        assert command == [common[0], str(S.ROOT / runner), *common[1:], *tail]


def test_failed_readout_preserves_attempt_and_pending_identity(tmp_path):
    runner = S.Run(SimpleNamespace(output=tmp_path))
    runner.check = lambda: None
    runner.context = {'fit_id': 'mlp128@10101', 'split': 'train', 'offset': 256}

    def operation():
        raise RuntimeError('synthetic readout failure')

    with pytest.raises(RuntimeError, match='synthetic readout'):
        runner.call('saved_prediction', operation)
    assert runner.calls['saved_prediction']['attempted'] == 1
    assert runner.calls['saved_prediction']['returned'] == 0
    rows = [json.loads(line) for line in (tmp_path / 'work.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['event'] == 'attempt'
    assert runner.pending == [{key: value for key, value in rows[0].items() if key != 'event'}]


def test_exclusive_collision_preserves_existing_files(tmp_path):
    existing = tmp_path / 'keep.txt'
    existing.write_text('original')
    with pytest.raises(ValueError, match='exclusive output'):
        S.Run(SimpleNamespace(output=tmp_path)).execute()
    assert existing.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [existing]


def test_failure_publication_cannot_replace_primary_error(tmp_path, monkeypatch):
    runner = S.Run(SimpleNamespace(output=tmp_path / 'run'))
    primary = RuntimeError('primary bind failure')

    def bind():
        raise primary

    def write_failure(*_args):
        raise OSError('synthetic failed-receipt storage error')

    runner.bind = bind
    monkeypatch.setattr(S, 'write', write_failure)
    with pytest.raises(RuntimeError, match='primary bind failure') as captured:
        runner.execute()
    assert captured.value is primary
    assert any('failed-receipt storage error' in note for note in primary.__notes__)


def test_late_cap_failure_demotes_completed_receipt(tmp_path, monkeypatch):
    runner = S.Run(SimpleNamespace(output=tmp_path / 'run'))

    def bind():
        runner.plan = {'mode': 'qualify', 'configuration': S.configuration('qualify'),
                       'limits': S.limits('qualify')}
        runner.clock = SimpleNamespace(now_ns=lambda: 2, backend='synthetic')
        runner.start = 1

    def prepare():
        runner.data = {'train': ([0], [0])}
        runner.floor = {'irreducible_mse_normalized': 0.}
        runner.calls = {'optimizer_update': {'attempted': 36, 'returned': 36, 'seconds': 1.}}

    def fit(kind, seed):
        runner.fits.append({'fit_id': f'{kind}@{seed}', 'fit_seconds': 1., 'metrics': {}})

    def check():
        if (runner.out / 'receipt.json').exists():
            raise ValueError('synthetic late output cap')

    runner.bind, runner.prepare, runner.fit, runner.check = bind, prepare, fit, check
    monkeypatch.setattr(S, 'authenticate', lambda _args: runner.plan)
    monkeypatch.setattr(S, 'payload_names', lambda _mode: {'summary.json'})
    with pytest.raises(ValueError, match='synthetic late output cap'):
        runner.execute()
    assert not (runner.out / 'receipt.json').exists()
    assert S.read(runner.out / 'invalid-completed-receipt.json')['status'] == 'completed'
    failed = S.read(runner.out / 'failed.json')
    assert failed['status'] == 'failed' and 'synthetic late output cap' in failed['error']
