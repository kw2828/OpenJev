"""Fabricated arrays and saved records only; no scientific inputs or training."""
from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_capacity_audit_test', ROOT/'scripts/audit_otto_capacity.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_duplicate_floor_is_row_weighted_not_equal_group_weighted():
    x = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], np.float32)
    y = np.asarray([0, 2, 0, 3, 6], np.float32)
    result = A.duplicate_floor(x, y, np)
    assert result == {'rows': 5, 'unique_groups': 2, 'duplicate_groups': 2, 'duplicate_rows': 5,
                      'conflicting_groups': 2, 'irreducible_mse_normalized': 4.}
    predictions = np.asarray([1, 1, 3, 3, 3], np.float64)
    assert A.scalar_metrics(predictions, y, np)['mse_normalized'] == result['irreducible_mse_normalized']


def test_duplicate_floor_uses_training_bytes_and_preserves_labels():
    x = np.asarray([[1, 0], [1, 0], [1, -0.0], [np.nextafter(np.float32(1), np.float32(2)), 0]], np.float32)
    y = np.asarray([2, 4, 10, 20], np.float32)
    original = x.tobytes(), y.tobytes()
    result = A.duplicate_floor(x, y, np)
    assert result['unique_groups'] == 3 and result['irreducible_mse_normalized'] == .5
    assert (x.tobytes(), y.tobytes()) == original


def test_floor_centered_sums_preserve_small_variance_on_large_offset():
    x = np.ones((3, 2), np.float32)
    y = np.asarray([2**20, 2**20+1, 2**20+2], np.float32)
    assert A.duplicate_floor(x, y, np)['irreducible_mse_normalized'] == pytest.approx(2/3)


def test_singletons_and_equal_targets_have_zero_floor():
    x = np.asarray([[0], [1], [1]], np.float32)
    result = A.duplicate_floor(x, np.asarray([7, 9, 9], np.float32), np)
    assert result['irreducible_mse_normalized'] == 0 and result['conflicting_groups'] == 0


def test_metrics_keep_signed_values_and_physical_scale():
    result = A.scalar_metrics(np.asarray([-1, 3], np.float64), np.asarray([1, 2], np.float32), np)
    assert result == {'rows': 2, 'mse_normalized': 2.5, 'mae_physical': 96.,
                      'negative_predictions': 1, 'minimum_normalized': -1., 'maximum_normalized': 3.}


def test_dense_prediction_has_independent_scalar_oracle_and_correct_orientation():
    x = np.zeros((3, A.DIMENSION), np.float64)
    x[:, :2] = [[.25, .75], [1., 0.], [0., 0.]]
    x[:, -1] = [.6, .8, 0.]
    w = np.zeros((2, A.DIMENSION), np.float64)
    w[:, [0, 1, A.DIMENSION-1]] = [[2, -3, 4], [-5, 6, -7]]
    layers = [(w, np.asarray([.5, -.25])), (np.asarray([[3., -2.]]), np.asarray([-.75]))]
    expected = []
    for row in x:
        h0 = max(2*row[0]-3*row[1]+4*row[-1]+.5, 0)
        h1 = max(-5*row[0]+6*row[1]-7*row[-1]-.25, 0)
        expected.append(.4*(row[0]+row[1])+3*h0-2*h1-.75)
    result = A.dense_predict(x, layers, .4, np)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    assert result[-1] == .75  # A biased model at zero is not silently repaired.


def test_three_hidden_layers_and_batch_partition_agree():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, A.DIMENSION)).astype(np.float64)
    widths = (A.DIMENSION, 3, 4, 2, 1)
    layers = [(rng.normal(size=(right, left)), rng.normal(size=right)) for left, right in pairwise(widths)]
    together = A.dense_predict(x, layers, .125, np)
    singles = np.concatenate([A.dense_predict(row[None], layers, .125, np) for row in x])
    np.testing.assert_allclose(together, singles, atol=1e-10, rtol=1e-10)
    assert A.count_parameters(layers) == sum(right*(left+1) for left, right in pairwise(widths))


@pytest.mark.parametrize('defect', ['empty', 'dtype', 'nan', 'target_length', 'target_infinite'])
def test_invalid_floor_inputs_rejected(defect):
    x, y = np.ones((2, 2), np.float32), np.ones(2, np.float32)
    if defect == 'empty':
        x, y = x[:0], y[:0]
    elif defect == 'dtype':
        x = x.astype(np.float64)
    elif defect == 'nan':
        x[0, 0] = np.nan
    elif defect == 'target_length':
        y = y[:1]
    else:
        y[0] = np.inf
    with pytest.raises(ValueError):
        A.duplicate_floor(x, y, np)


def exported(kind):
    result = {'version': np.asarray('otto-capacity-value-v1'), 'kind': np.asarray(kind),
              'input_dim': np.asarray(A.DIMENSION), 'c0': np.asarray(.5, np.float32)}
    dims = A.WIDTHS[kind]
    for i, (left, right) in enumerate(pairwise(dims)):
        result[f'weight_{i}'] = np.zeros((right, left), np.float32)
        result[f'bias_{i}'] = np.zeros(right, np.float32)
    return result


@pytest.mark.parametrize(('kind', 'parameters'), [('mlp8', 88241), ('mlp128', 1411841), ('deep128', 1444865)])
def test_checkpoint_exact_counts_and_owned_upcasts(kind, parameters):
    original = exported(kind)
    layers = A.checkpoint(original, kind, .5, np)
    assert A.count_parameters(layers) == parameters
    original['weight_0'][0, 0] = 10
    assert layers[0][0][0, 0] == 0 and layers[0][0].dtype == np.float64


@pytest.mark.parametrize('defect', ['extra', 'missing', 'transpose', 'double', 'nan', 'c0', 'kind'])
def test_malformed_checkpoint_fails_before_prediction(defect):
    original = exported('mlp8')
    if defect == 'extra':
        original['optimizer'] = np.zeros(1)
    elif defect == 'missing':
        del original['bias_1']
    elif defect == 'transpose':
        original['weight_0'] = original['weight_0'].T
    elif defect == 'double':
        original['weight_0'] = original['weight_0'].astype(np.float64)
    elif defect == 'nan':
        original['weight_0'][0, 0] = np.nan
    elif defect == 'c0':
        original['c0'] = np.asarray(.25, np.float32)
    else:
        original['kind'] = np.asarray('deep128')
    with pytest.raises(ValueError):
        A.checkpoint(original, 'mlp8', .5, np)


def fit_rows():
    return [{'fit_id': f'{kind}@{seed}', 'metrics': {
        'train': {'rows': 5589, 'mse_normalized': .35 if kind == 'mlp8' else .3},
        'valid': {'rows': 1109, 'mse_normalized': 1. if kind == 'mlp8' else .9}}}
        for seed in A.SEEDS for kind in A.WIDTHS]


def test_all12_rules_use_floor_adjusted_error_and_every_seed():
    result = A.aggregate(fit_rows(), .1)
    assert len(result['checks']) == 12 and result['capacity_screen_passed']
    assert all(result['family_admission'].values())
    rows = fit_rows()
    rows[1]['metrics']['train']['mse_normalized'] = .31
    result = A.aggregate(rows, .1)
    assert not result['capacity_screen_passed'] and not result['family_admission']['mlp128']
    assert result['family_admission']['deep128']
    assert result['checks'][0]['threshold'] == .8*(.35-.1)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'rows', 'nonfinite'])
def test_incomplete_or_malformed_fit_membership_rejected(defect):
    rows = fit_rows()
    if defect == 'missing':
        rows.pop()
    elif defect == 'duplicate':
        rows[-1] = rows[-2]
    elif defect == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == 'rows':
        rows[0]['metrics']['valid']['rows'] = 1108
    else:
        rows[0]['metrics']['train']['mse_normalized'] = float('nan')
    with pytest.raises(ValueError):
        A.aggregate(rows, .1)


def work_rows(channel='optimizer_update', context=None, seconds=.125, identifier=1):
    shared = {'id': identifier, 'channel': channel, 'context': context or {'fit_id': 'mlp8@10101'}}
    return [{**shared, 'event': 'attempt'}, {**shared, 'event': 'return', 'seconds': seconds}]


def test_work_preserves_order_context_and_disjoint_seconds():
    rows = work_rows(seconds=.125)+work_rows('checkpoint_export', seconds=.25, identifier=2)
    work = A.Work(rows)
    assert work.take('optimizer_update', {'fit_id': 'mlp8@10101'}) == .125
    assert work.take('checkpoint_export', {'fit_id': 'mlp8@10101'}) == .25
    assert work.counts['optimizer_update'] == {'attempted': 1, 'returned': 1, 'seconds': .125}
    with pytest.raises(ValueError, match='channel counts'):
        work.finish()


@pytest.mark.parametrize('defect', ['pending', 'nested', 'context', 'id', 'bool_id', 'extra', 'nan', 'negative'])
def test_work_rejects_unresolved_or_forged_atomic_events(defect):
    rows = work_rows()
    if defect == 'pending':
        rows.pop()
    elif defect == 'nested':
        rows.insert(1, work_rows(identifier=2)[0])
    elif defect == 'context':
        rows[1]['context'] = {'fit_id': 'mlp128@10101'}
    elif defect == 'id':
        rows[1]['id'] = 2
    elif defect == 'bool_id':
        rows[0]['id'] = True
    elif defect == 'extra':
        rows[1]['retry'] = 1
    elif defect == 'nan':
        rows[1]['seconds'] = float('nan')
    else:
        rows[1]['seconds'] = -1
    with pytest.raises((ValueError, StopIteration)):
        A.Work(rows).take('optimizer_update', {'fit_id': 'mlp8@10101'})


def epoch_fixture(epoch=1):
    import hashlib
    rng = np.random.default_rng(30101)
    for _ in range(epoch):
        order = rng.permutation(5589)
    fit_id = 'mlp8@10101'
    order_sha = hashlib.sha256(order.tobytes()).hexdigest()
    ordered = {'fit_id': fit_id, 'epoch': epoch, 'order': order.tolist(), 'sha256': order_sha}
    updates, journal = [], []
    for batch, offset in enumerate(range(0, 5589, 128)):
        context = {'fit_id': fit_id, 'epoch': epoch, 'batch': batch}
        updates.append({**context, 'rows': min(128, 5589-offset), 'loss': .25, 'gradient_norm': 10.,
                        'update_index': (epoch-1)*44+batch+1})
        journal.extend(work_rows(context=context, identifier=batch+1))
    curve = {'fit_id': fit_id, 'epoch': epoch, 'rows': 5589, 'updates': 44,
             'training_mse_normalized': .25, 'order_sha256': order_sha}
    return order, ordered, updates, journal, curve


def test_epoch_exact_rng_short_batch_and_preclip_gradient_norm():
    for epoch in (1, 80):
        order, ordered, updates, journal, curve = epoch_fixture(epoch)
        assert updates[-1]['rows'] == 85
        assert A.epoch_records('mlp8@10101', epoch, order, ordered, iter(updates), curve, A.Work(journal), np) == 5.5


@pytest.mark.parametrize('defect', ['order', 'short_batch', 'index', 'curve', 'negative_gradient'])
def test_epoch_rejects_wrong_row_and_optimizer_witnesses(defect):
    order, ordered, updates, journal, curve = epoch_fixture()
    if defect == 'order':
        ordered['order'][0], ordered['order'][1] = ordered['order'][1], ordered['order'][0]
    elif defect == 'short_batch':
        updates[-1]['rows'] = 128
    elif defect == 'index':
        updates[-1]['update_index'] = 43
    elif defect == 'curve':
        curve['training_mse_normalized'] = .3
    else:
        updates[0]['gradient_norm'] = -1
    with pytest.raises(ValueError):
        A.epoch_records('mlp8@10101', 1, order, ordered, iter(updates), curve, A.Work(journal), np)


def test_exact38_payloads_include_every_initial_final_and_prediction():
    names = A.payload_names()
    assert len(names) == 38
    for seed in A.SEEDS:
        for kind in A.WIDTHS:
            assert {f'{phase}-{kind}-{seed}.npz' for phase in ('initial', 'final', 'predictions')} <= names


def test_manifest_rejects_extra_file_and_tampered_bound_member(tmp_path):
    A.write(tmp_path/'data.json', {'x': 1})
    receipt = {'status': 'completed', 'files': {'data.json': A.digest(tmp_path/'data.json')}}
    A.write(tmp_path/'receipt.json', receipt)
    pin = A.digest(tmp_path/'receipt.json')['sha256']
    assert A.manifest(tmp_path, pin, {'data.json'}, lambda: None) == receipt
    (tmp_path/'extra').write_text('unexpected')
    with pytest.raises(ValueError, match='closed'):
        A.manifest(tmp_path, pin, {'data.json'}, lambda: None)
    (tmp_path/'extra').unlink()
    (tmp_path/'data.json').write_text('{}')
    with pytest.raises(ValueError, match='bound payload'):
        A.manifest(tmp_path, pin, {'data.json'}, lambda: None)


def terminal_fixture(tmp_path):
    plan = {'python_executable': '/fixed/python', 'limits': {'native_seconds': 1800},
            'sources': {'scripts/supervise_dialogue_observation_v2.py': 'watchdog'}}
    plan_path = tmp_path/'plan.json'
    A.write(plan_path, plan)
    run = tmp_path/'run'
    run.mkdir()
    launch_path = tmp_path/'launch.json'
    request = {'plan': str(plan_path), 'plan_sha256': A.digest(plan_path)['sha256'],
               'output': str(run), 'supervision': str(launch_path)}
    launch = {'command': ['/fixed/python', '-u', str(A.ROOT/A.RUNNER), '--plan', str(plan_path),
              '--plan-sha256', request['plan_sha256'], '--output', str(run), '--supervision', str(launch_path)],
              'pid': 102, 'pgid': 102, 'parent_pid': 101, 'cwd': str(A.ROOT), 'started_ns': 100,
              'deadline_ns': 1800*10**9+100, 'clock_backend': 'mach_continuous_time', 'cap_seconds': 1800,
              'watchdog_sha256': 'watchdog', 'clock_source_sha256': A.CLOCK_PIN}
    A.write(launch_path, launch)
    worker = {'supervision_sha256': A.digest(launch_path)['sha256'], 'started_ns': 200, 'finished_ns': 400,
              'clock_backend': 'mach_continuous_time', 'wall_seconds': 200/1e9}
    terminal = {**launch, 'finished_ns': 500, 'elapsed_ns': 400, 'wall_seconds': 400/1e9,
                'status': 'completed', 'returncode': 0, 'timed_out': False, 'group_absent': True,
                'error': None, 'clock_error': None, 'cleanup': {'group_absent': True, 'reaped': True, 'errors': []}}
    started = {'launch': launch, 'request': request, 'started_ns': 200}
    return plan, plan_path, worker, started, terminal, run


def test_terminal_encloses_worker_and_uses_native_not_civil_time(tmp_path):
    values = terminal_fixture(tmp_path)
    values[4]['started_unix'], values[4]['finished_unix'] = 200., 1.
    A.check_terminal(*values, lambda: None)


@pytest.mark.parametrize('defect', ['exit', 'cleanup', 'deadline', 'backend', 'worker_outside', 'clock', 'launch_hash'])
def test_terminal_rejects_failed_late_or_mismatched_execution(tmp_path, defect):
    values = terminal_fixture(tmp_path)
    worker, terminal = values[2], values[4]
    if defect == 'exit':
        terminal['returncode'] = 1
    elif defect == 'cleanup':
        terminal['cleanup']['errors'] = ['reap failed']
    elif defect == 'deadline':
        terminal['finished_ns'] = terminal['deadline_ns']
    elif defect == 'backend':
        worker['clock_backend'] = 'CLOCK_BOOTTIME'
    elif defect == 'worker_outside':
        worker['finished_ns'] = terminal['finished_ns']+1
    elif defect == 'clock':
        terminal['clock_error'] = 'unavailable'
    else:
        worker['supervision_sha256'] = 'incorrect'
    with pytest.raises(ValueError):
        A.check_terminal(*values, lambda: None)


def audit_args(tmp_path):
    from argparse import Namespace
    return Namespace(output=tmp_path/'audit', plan=tmp_path/'plan.json', plan_sha256='plan',
                     run=tmp_path/'run', receipt_sha256='worker', terminal=tmp_path/'terminal.json', terminal_sha256='terminal')


def test_clock_initialization_failure_is_retained_with_null_timing(tmp_path, monkeypatch):
    def broken(*_):
        raise RuntimeError('clock unavailable')
    monkeypatch.setattr(A, 'digest', lambda *args: {'sha256': A.CLOCK_PIN})
    monkeypatch.setattr(A, 'load', broken)
    audit = A.Audit(audit_args(tmp_path))
    with pytest.raises(RuntimeError, match='clock unavailable'):
        audit.execute()
    failed = A.read(audit.out/'failed.json')
    assert failed['started_ns'] is None and failed['wall_seconds'] is None
    assert not (audit.out/'receipt.json').exists()


def test_late_failure_demotes_completed_receipt_and_preserves_primary(tmp_path, monkeypatch):
    from types import SimpleNamespace
    class FakeClock:
        backend = 'injected'
        def now_ns(self):
            return 100
    monkeypatch.setattr(A, 'digest', lambda *args: {'sha256': A.CLOCK_PIN, 'bytes': 1})
    monkeypatch.setattr(A, 'load', lambda *args: SimpleNamespace(SuspendClock=FakeClock))
    audit = A.Audit(audit_args(tmp_path))
    def authenticate():
        audit.plan = {'sources': {A.RUNNER: 'producer'}}
    def check():
        audit.rss = 100
        if (audit.out/'receipt.json').exists():
            raise TimeoutError('post-publication deadline')
    monkeypatch.setattr(audit, 'authenticate', authenticate)
    monkeypatch.setattr(audit, 'compute', lambda: {'agreement': True})
    monkeypatch.setattr(audit, 'check', check)
    with pytest.raises(TimeoutError, match='post-publication'):
        audit.execute()
    assert A.read(audit.out/'invalid-completed-receipt.json')['status'] == 'completed'
    assert A.read(audit.out/'failed.json')['status'] == 'failed'
    assert not (audit.out/'receipt.json').exists()
