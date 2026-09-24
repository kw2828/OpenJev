"""Fabricated metadata and orchestration checks; no empirical evidence is read."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('direct_readout_runner_tests', ROOT / 'scripts/run_otto_direct_readout.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_freeze_is_metadata_only_and_preserves_diagnostic_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'sys', SimpleNamespace(modules={}))
    auth = {'sources': {'fabricated.py': 'fixed'}, 'train': {'stage': 'train'}, 'dev': {'stage': 'dev'}}
    pin = {'path': 'qualification.json', 'sha256': 'fixed', 'bytes': 1}
    monkeypatch.setattr(runner, 'authenticate', lambda: copy.deepcopy(auth))
    monkeypatch.setattr(runner, 'engineering', lambda _: pin)
    args = SimpleNamespace(engineering=tmp_path / 'engineering.json', output=tmp_path / 'plan.json')
    runner.freeze(args)
    saved = json.loads(args.output.read_text())
    assert saved['status'] == 'registered_before_empirical_decodes'
    assert saved['configuration']['dev_reused'] is True
    assert saved['configuration']['held_out_from_training'] is True
    assert saved['configuration']['fresh_dev'] is saved['configuration']['held_out_evidence'] is False
    assert saved['configuration']['test_admitted'] is saved['configuration']['confirmation_admitted'] is False
    assert saved['engineering'] == pin and saved['sources'] == auth['sources']
    runner.validate_plan(saved)
    with pytest.raises(FileExistsError):
        runner.freeze(args)
    for name in ('numpy', 'torch', 'tensorflow'):
        runner.sys.modules[name] = object()
        args.output = tmp_path / (name + '.json')
        with pytest.raises(ValueError, match='metadata-only'):
            runner.freeze(args)
        assert not args.output.exists()
        del runner.sys.modules[name]
    saved['configuration']['ridge'] = .001
    with pytest.raises(ValueError, match='fixed new diagnostic recipe'):
        runner.validate_plan(saved)
    saved['configuration']['ridge'] = runner.CONFIG['ridge']
    saved['sources']['fabricated.py'] = 'changed'
    with pytest.raises(ValueError, match='unchanged registered evidence'):
        runner.validate_plan(saved)


@pytest.mark.parametrize('change', [None, 'timeout', 'unreaped', 'live_group', 'scientific_call',
                                  'source_drift', 'capacity', 'failed_capacity', 'capacity_decode'])
def test_engineering_requires_closed_fabricated_work_and_capacity(monkeypatch, tmp_path, change):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'NEW', {'source.py'})
    (tmp_path / 'source.py').write_text('# fabricated\n')
    sources = {'source.py': runner.descriptor('source.py')['sha256']}
    capacity = {'status': 'passed', 'projected_seconds': 1350, 'audit_projected_seconds': 450,
                'empirical_array_decodes': 0, 'checkpoint_decodes': 0}
    if change == 'capacity':
        capacity['projected_seconds'] = 1350.001
    elif change == 'failed_capacity':
        capacity['status'] = 'failed'
    elif change == 'capacity_decode':
        capacity['empirical_array_decodes'] = 1
    runner.write(tmp_path / 'capacity.json', capacity)
    record = {'status': 'passed', 'sources_before': sources.copy(), 'sources_after': sources.copy(),
              'commands': [{'returncode': 0, 'timed_out': False, 'reaped': True, 'group_absent': True}
                           for _ in range(3)],
              'empirical_array_decodes': 0, 'checkpoint_decodes': 0, 'scientific_calls': 0,
              'files': {'capacity.json': runner.descriptor('capacity.json')}}
    if change in ('timeout', 'unreaped', 'live_group'):
        key = {'timeout': 'timed_out', 'unreaped': 'reaped', 'live_group': 'group_absent'}[change]
        record['commands'][0][key] = change == 'timeout'
    elif change == 'scientific_call':
        record['scientific_calls'] = 1
    elif change == 'source_drift':
        (tmp_path / 'source.py').write_text('# changed\n')
    runner.write(tmp_path / 'engineering.json', record)
    if change is None:
        assert runner.engineering('engineering.json') == runner.descriptor('engineering.json')
    else:
        with pytest.raises(ValueError):
            runner.engineering('engineering.json')


def test_authentication_uses_only_fixed_parent_and_comparator_checkpoints(monkeypatch):
    called, verified = [], []
    sources = {'old.py': 'old'}
    old_plan = {'sources': sources, 'train': {'stage': 'train'}, 'dev': {'stage': 'dev'},
                'lineage': {'checkpoints': {'pretrained': {'fixture': True}}}, 'runtime': {'fixture': True}}
    files = {'fits.json': {'path': 'fits.json'}, 'views.json': {'path': 'views.json'}}
    for seed in runner.SEEDS:
        for family in ('action_residual_only', 'full_joint'):
            name = f'checkpoint-{family}-{seed}.npz'
            files[name] = {'path': name}
    closure = {'technical_complete': True, 'status': 'DEV_FAIL', 'plan': {'path': 'old-plan'},
               'producer_receipt': {'path': 'producer'}, 'producer_terminal': {'path': 'producer-terminal'},
               'audit_receipt': {'path': 'audit'}, 'audit_terminal': {'path': 'audit-terminal'}}

    def closed(*args):
        called.append(args)
        return copy.deepcopy(old_plan), {'files': copy.deepcopy(files)}, Path('/fabricated')

    def descriptor(path):
        sha = runner.PRIOR_SHA if path == runner.PRIOR else runner.CLOSURE_SHA if path == runner.CLOSURE else 'new'
        return {'path': str(path), 'sha256': sha, 'bytes': 1}

    monkeypatch.setattr(runner, 'descriptor', descriptor)
    monkeypatch.setattr(runner, 'read', lambda _: closure)
    monkeypatch.setattr(runner, 'load', lambda *_: SimpleNamespace(process_closure=closed))
    monkeypatch.setattr(runner, 'verify', lambda pin: verified.append(pin['path']))
    actual = runner.authenticate()
    assert called == [('old-plan', 'producer', 'producer-terminal'), ('old-plan', 'audit', 'audit-terminal')]
    assert len(verified) == 6 and set(verified) == set(files) - {'fits.json', 'views.json'}
    assert actual['prior_views'] == files['views.json']
    assert actual['lineage'] == old_plan['lineage'] and sources == {'old.py': 'old'}
    assert actual['sources'].keys() == {'old.py'} | runner.NEW
    assert set(actual['comparators']) == {'action_residual_only', 'full_joint'}
    assert not {'test', 'confirmation'} & actual.keys()
    closure['status'] = 'DEV_PASS'
    with pytest.raises(ValueError, match='prior failure retained'):
        runner.authenticate()


@pytest.mark.parametrize('stage,solves,views,barrier', [
    ('test', 6, 12, True), ('confirm', 6, 12, True), ('validation', 6, 12, True),
    ('dev', 5, 12, True), ('dev', 6, 11, True), ('dev', 6, 12, False),
    ('dev', 7, 12, True), ('dev', 6, 13, True)])
def test_history_rejects_wrong_split_or_incomplete_barrier_before_import(stage, solves, views, barrier, tmp_path):
    worker = runner.BoundRun.__new__(runner.BoundRun)
    worker.out = tmp_path
    worker.receipt = {'solves_completed': solves, 'train_views_completed': views}
    if barrier:
        (tmp_path / 'training-barrier.json').write_text('{}')
    # No plan, array module or decoder is installed: rejection must precede them.
    with pytest.raises(ValueError, match='TRAIN and reused DEV only|all solves and parity checks'):
        worker.history(stage)


@pytest.mark.parametrize('change', [None, 'command', 'pid', 'deadline', 'plan_pin', 'watchdog'])
def test_bind_requires_original_runtime_command_and_external_plan_pin(monkeypatch, tmp_path, change):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'sys', SimpleNamespace(executable='/fabricated/python', argv=['runner.py', 'train']))
    for name, value in (('getpid', 10), ('getpgrp', 10), ('getppid', 9)):
        monkeypatch.setattr(runner.os, name, lambda value=value: value)
    for key in runner.THREADS:
        monkeypatch.setenv(key, '1')
    planpath, launchpath = tmp_path / 'plan.json', tmp_path / 'launch.json'
    runner.write(planpath, {'sources': {runner.CLOCK: 'clock', runner.SUPERVISOR: 'watchdog'},
                            'runtime': {'fabricated': True}})
    launch = {'command': ['/fabricated/python', 'runner.py', 'train'], 'pid': 10, 'pgid': 10,
              'parent_pid': 9, 'cap_seconds': 1800, 'clock_backend': 'mach_continuous_time',
              'started_ns': 0, 'deadline_ns': 1800 * 10**9,
              'clock_source_sha256': 'clock', 'watchdog_sha256': 'watchdog'}
    if change == 'command':
        launch['command'][0] = '/other/python'
    elif change == 'pid':
        launch['pid'] = 11
    elif change == 'deadline':
        launch['deadline_ns'] += 1
    elif change == 'watchdog':
        launch['watchdog_sha256'] = 'other'
    runner.write(launchpath, launch)
    args = SimpleNamespace(mode='train', output=tmp_path, supervision=launchpath, plan=planpath,
                           plan_sha256='other' if change == 'plan_pin' else runner.descriptor(planpath)['sha256'])
    worker = runner.BoundRun(args)
    worker.clock, worker.start = SimpleNamespace(backend='mach_continuous_time', now_ns=lambda: 2), 1
    worker.check = lambda: None
    published = []
    worker.publish = lambda name, value: published.append(name)
    monkeypatch.setattr(runner, 'validate_plan', lambda _: None)
    if change is None:
        worker.bind()
        assert published == ['started.json', 'runtime.json']
        assert worker.receipt['plan'] == runner.descriptor(planpath)
    else:
        with pytest.raises(ValueError):
            worker.bind()
        assert published == []


@pytest.fixture
def orchestration(monkeypatch, tmp_path):
    """Use tiny synthetic arrays and fake modules, never a numerical model or SVD."""
    import numpy as np

    from openjev import research

    class Value:
        def __init__(self, array):
            self.array = np.array(array, copy=True)

        def clone(self):
            return Value(self.array)

        def numpy(self):
            return self.array

    fake_torch = SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None,
        use_deterministic_algorithms=lambda _: None, from_numpy=Value,
        equal=lambda a, b: np.array_equal(a.array, b.array))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    trace, published, saved, designs = [], {}, {}, []
    fail = {'parity': None, 'solve': None}
    history = {'legal': np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 0, 0, 0]], bool),
               'prior_mask': np.array([False, False, True]),
               'nonquery_weights': np.array([0., .5, 0.]), 'prior_weights': np.array([0., 0., .25]),
               'targets': np.arange(12, dtype=np.float32).reshape(3, 4)}
    before = {k: v.copy() for k, v in history.items()}
    cache_fields = {'z', 'support_mask', 'seconds', 'parent_prediction'}

    def extract(parent, seed, history, check):
        check()
        trace.append(('cache', seed))
        z = np.zeros((3, 29), np.float64)
        z[:, -1] = 1
        return {'z': z, 'support_mask': np.array([False, True, True]), 'seconds': .1,
                'parent_prediction': np.full((3, 4), 999, np.float32)}

    def affine(cache, theta):
        assert set(cache) == cache_fields, 'design arrays must not contaminate the validated cache'
        result = cache['z'] @ theta.T
        result -= result.mean(1, keepdims=True)
        result[~cache['support_mask']] = 0
        return result

    def design(z, error, legal, weights):
        designs.append((error.copy(), legal.copy(), weights.copy()))
        return np.zeros((4, 87)), np.ones(4)

    def solve(A, b, mode):
        trace.append(('solve', mode))
        if sum(item[0] == 'solve' for item in trace) == fail['solve']:
            raise RuntimeError('fabricated solver failure')
        return {'increment': np.zeros((4, 29)), 'diagnostics': {'mode': mode}}

    solver = SimpleNamespace(build_design=design, solve_design=solve,
        contrast_basis=lambda: np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * .5)
    cache_module = SimpleNamespace(extract=extract, affine_prediction=affine)
    for name, module in (('otto_direct_readout', solver), ('otto_direct_readout_cache', cache_module)):
        monkeypatch.setattr(research, name, module, raising=False)
        monkeypatch.setitem(sys.modules, 'openjev.research.' + name, module)
    worker = runner.BoundRun(SimpleNamespace(mode='train', output=tmp_path))
    worker.check = lambda: None
    ticks = iter(range(10000))
    worker.clock = SimpleNamespace(now_ns=lambda: next(ticks))
    worker.plan = {'lineage': {'checkpoints': {'pretrained': {str(s): ('pretrained', s) for s in runner.SEEDS}}},
                   'comparators': {a: {str(s): (a, s) for s in runner.SEEDS}
                                   for a in ('action_residual_only', 'full_joint')}}

    def state(pin):
        trace.append(('state', *pin))
        worker.receipt['array_decodes'] += 1
        worker.receipt['checkpoint_decodes'] += 1
        return {**{f'fixed-{i}': Value([i]) for i in range(6)},
                'action_residual.weight': Value(np.zeros((4, 28), np.float32)),
                'action_residual.bias': Value(np.arange(4, dtype=np.float32) / 8)}

    def open_history(stage):
        trace.append(('history', stage))
        if stage == 'dev':
            assert worker.receipt['solves_completed'] == 6
            assert worker.receipt['train_views_completed'] == 12
            assert published['training-barrier.json']['dev_decodes'] == 0
            assert len(published['training-barrier.json']['checkpoints']) == 6
            assert all(row['parity']['passed'] for row in published['fits.json'])
        worker.receipt['array_decodes'] += 1
        return history

    def save(name, arrays):
        assert name not in saved
        saved[name] = {k: v.copy() for k, v in arrays.items()}
        return {'path': name, 'sha256': 'fabricated', 'bytes': 1}

    def publish(name, value):
        assert name not in published
        published[name] = copy.deepcopy(value)
        trace.append(('publish', name))
        return {'path': name, 'sha256': 'fabricated', 'bytes': 1}

    def predict(family, seed, state, history, stage):
        trace.append(('view', stage, family, seed))
        cache = {'z': saved[f'cache-{seed}.npz']['z'],
                 'support_mask': saved[f'cache-{seed}.npz']['support_mask'], 'seconds': 0,
                 'parent_prediction': saved[f'cache-{seed}.npz']['parent_prediction']}
        scores = (64 * affine(cache, worker.theta(state))).astype(np.float32)
        if fail['parity'] == (family, seed):
            scores[1, 0] += 1
        key = 'train_views_completed' if stage == 'train' else 'views_completed'
        worker.receipt[key] += 1
        return {'stage': stage, 'family': family, 'seed': seed, 'seconds': .2}, {
            'action_prediction': scores.copy(), 'corrected_shadow_prior': scores.copy()}

    worker.state, worker.history, worker.save_arrays, worker.publish, worker.predict = state, open_history, save, publish, predict
    return SimpleNamespace(worker=worker, trace=trace, published=published, saved=saved,
                           fail=fail, designs=designs, history=history, before=before, np=np)


def test_all_six_solves_and_twelve_train_parity_views_precede_any_dev(orchestration):
    f = orchestration
    f.worker.train()
    boundary = f.trace.index(('history', 'dev'))
    assert sum(row[0] == 'solve' for row in f.trace[:boundary]) == 6
    train = [row[2:] for row in f.trace[:boundary] if row[:2] == ('view', 'train')]
    assert train == [(a, s) for s in runner.SEEDS for a in runner.TRAIN_VIEWS]
    dev = [row[2:] for row in f.trace[boundary:] if row[:2] == ('view', 'dev')]
    assert dev == [(a, s) for s in runner.SEEDS for a in runner.VIEWS]
    assert len(f.published['fits.json']) == 6 and len(f.published['views.json']) == 27
    assert f.worker.receipt['array_decodes'] == 11 and f.worker.receipt['checkpoint_decodes'] == 9
    assert f.worker.receipt['optimizer_steps'] == f.worker.receipt['teacher_calls'] == 0
    assert f.worker.receipt['native_calls'] == f.worker.receipt['test_array_decodes'] == 0
    assert all(row[0] != 'history' or row[1] in ('train', 'dev') for row in f.trace)
    for error, legal, weights in f.designs:
        f.np.testing.assert_array_equal(legal[2], True)
        f.np.testing.assert_allclose(weights, [0, .25, .0625])
        expected = f.history['targets'].astype(f.np.float64) / 64
        expected[1:] -= f.np.arange(4) / 8 - 3 / 16
        f.np.testing.assert_array_equal(error, expected)
    for name in f.before:
        f.np.testing.assert_array_equal(f.before[name], f.history[name])
    for name, arrays in f.saved.items():
        if name.startswith('checkpoint-'):
            assert set(arrays) == {f'slow.fixed-{i}' for i in range(6)} | {
                'slow.action_residual.weight', 'slow.action_residual.bias'}
            for i in range(6):
                f.np.testing.assert_array_equal(arrays[f'slow.fixed-{i}'], [i])


@pytest.mark.parametrize('where', ['first_control', 'last_export', 'solve'])
def test_any_failed_export_or_solve_keeps_dev_unopened(orchestration, where):
    f = orchestration
    if where == 'solve':
        f.fail['solve'] = 3
    else:
        f.fail['parity'] = ('pretrained', runner.SEEDS[0]) if where == 'first_control' else ('ridge', runner.SEEDS[-1])
    with pytest.raises((ValueError, RuntimeError), match='cached affine surrogate|fabricated solver failure'):
        f.worker.train()
    assert ('history', 'dev') not in f.trace
    assert 'training-barrier.json' not in f.published
    assert 'summary.json' not in f.published
    assert f.worker.receipt['views_completed'] == 0
    if where == 'solve':
        assert f.worker.receipt['solves_completed'] == 2
        assert f.worker.receipt['pending']['stage'] == 'solve'


@pytest.fixture
def closed_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'validate_plan', lambda _: None)

    def make(phase):
        directory = tmp_path / phase
        directory.mkdir()
        planpath, terminalpath = tmp_path / f'{phase}-plan.json', tmp_path / f'{phase}-terminal.json'
        launchpath = tmp_path / f'{phase}-launch.json'
        sources = {runner.SUPERVISOR: 'watchdog', runner.CLOCK: 'clock'}
        runner.write(planpath, {'sources': sources})
        extras = []
        producer_pins = {}
        if phase == 'audit':
            for kind in ('receipt', 'terminal'):
                path = tmp_path / f'producer-{kind}.json'
                runner.write(path, {'fabricated': kind})
                producer_pins['producer_' + kind] = runner.descriptor(path)
                extras += ['--producer-' + kind, str(path)]
        command = [str(tmp_path / '.venv/bin/python'), str(tmp_path / runner.SELF), phase,
                   '--plan', str(planpath), '--plan-sha256', runner.descriptor(planpath)['sha256'],
                   '--supervision', str(launchpath), '--output', str(directory), *extras]
        cap = runner.LIMITS[phase]['seconds']
        launch = {'command': command, 'pid': 10, 'pgid': 10, 'parent_pid': 9, 'cwd': str(tmp_path),
                  'started_ns': 0, 'deadline_ns': cap * 10**9, 'cap_seconds': cap,
                  'watchdog_sha256': 'watchdog', 'clock_source_sha256': 'clock', 'clock_backend': 'mach_continuous_time'}
        runner.write(launchpath, launch)
        for name in runner.expected_payloads(phase):
            runner.write(directory / name, {'launch': launch, 'started_ns': 1} if name == 'started.json' else {})
        receipt = {'phase': phase, 'version': runner.VERSION, 'status': 'completed', 'pending': None,
                   'plan': runner.descriptor(planpath), 'supervision': runner.descriptor(launchpath),
                   'started_ns': 1, 'finished_ns': 50, 'sources': sources, 'limits': runner.LIMITS[phase],
                   'teacher_calls': 0, 'native_calls': 0, 'test_array_decodes': 0, 'optimizer_steps': 0,
                   'fits_completed': 6, 'solves_completed': 6, 'train_views_completed': 12,
                   'views_completed': 15, 'checkpoint_decodes': 9, 'array_decodes': 11,
                   'files': {p.name: runner.descriptor(p) for p in directory.iterdir()}}
        if phase == 'audit':
            receipt.update(agreement=True, extra_arguments=extras, **producer_pins,
                array_decodes=49, checkpoint_decodes=15, views_completed=27,
                audit_counts={'array_decodes': 49, 'checkpoint_decodes': 15, 'views_completed': 27,
                              'fits_checked': 6, 'train_views_completed': 12, 'dev_views_completed': 15})
        terminal = {**copy.deepcopy(launch), 'finished_ns': 60, 'status': 'completed', 'returncode': 0,
                    'timed_out': False, 'error': None, 'clock_error': None, 'group_absent': True,
                    'timing_available': True, 'cleanup': {'reaped': True, 'group_absent': True, 'errors': []}}
        return planpath, directory / 'receipt.json', terminalpath, receipt, terminal

    return make


@pytest.mark.parametrize('phase', ['train', 'audit'])
def test_complete_original_process_closure(closed_process, phase):
    plan, receiptpath, terminalpath, receipt, terminal = closed_process(phase)
    runner.write(receiptpath, receipt)
    runner.write(terminalpath, terminal)
    assert runner.process_closure(plan, receiptpath, terminalpath)[1] == receipt
    assert len(runner.expected_payloads(phase)) == (47 if phase == 'train' else 3)


@pytest.mark.parametrize('change', ['solve_count', 'train_view_count', 'dev_view_count', 'decode_count',
    'checkpoint_count', 'optimizer', 'test_decode', 'pending', 'missing_payload', 'extra_payload',
    'payload_drift', 'outside_time', 'unreaped', 'timeout', 'command', 'limits', 'sources', 'watchdog'])
def test_original_train_closure_rejects_incomplete_or_drifted_evidence(closed_process, change):
    plan, receiptpath, terminalpath, receipt, terminal = closed_process('train')
    keys = {'solve_count': 'solves_completed', 'train_view_count': 'train_views_completed',
            'dev_view_count': 'views_completed', 'decode_count': 'array_decodes',
            'checkpoint_count': 'checkpoint_decodes', 'optimizer': 'optimizer_steps', 'test_decode': 'test_array_decodes'}
    if change in keys:
        receipt[keys[change]] += 1
    elif change == 'pending':
        receipt['pending'] = {'stage': 'uncertain solve'}
    elif change == 'missing_payload':
        receipt['files'].pop('fits.json')
    elif change == 'extra_payload':
        (receiptpath.parent / 'undeclared.json').write_text('{}')
    elif change == 'payload_drift':
        (receiptpath.parent / 'fits.json').write_text('{"changed":true}')
    elif change == 'outside_time':
        receipt['finished_ns'] = 61
    elif change == 'unreaped':
        terminal['cleanup']['reaped'] = False
    elif change == 'timeout':
        terminal['timed_out'] = True
    elif change == 'command':
        terminal['command'][2] = 'audit'
    elif change == 'limits':
        receipt['limits'] = {**receipt['limits'], 'seconds': 1801}
    elif change == 'sources':
        receipt['sources'] = {**receipt['sources'], 'injected.py': 'new'}
    else:
        terminal['watchdog_sha256'] = 'changed'
    runner.write(receiptpath, receipt)
    runner.write(terminalpath, terminal)
    with pytest.raises(ValueError):
        runner.process_closure(plan, receiptpath, terminalpath)


@pytest.mark.parametrize('change', ['audit_count', 'top_count', 'agreement', 'producer_arguments'])
def test_original_audit_closure_binds_work_and_producer_arguments(closed_process, change):
    plan, receiptpath, terminalpath, receipt, terminal = closed_process('audit')
    if change == 'audit_count':
        receipt['audit_counts']['dev_views_completed'] = 14
    elif change == 'top_count':
        receipt['checkpoint_decodes'] = 0
    elif change == 'agreement':
        receipt['agreement'] = False
    else:
        # Keep the launched command valid while substituting a different opaque
        # producer descriptor. The explicit argument-to-descriptor join must fail.
        path = receiptpath.parent.parent / 'other-producer.json'
        runner.write(path, {'fabricated': 'other'})
        receipt['producer_receipt'] = runner.descriptor(path)
    runner.write(receiptpath, receipt)
    runner.write(terminalpath, terminal)
    with pytest.raises(ValueError):
        runner.process_closure(plan, receiptpath, terminalpath)


def test_failed_clock_preserves_original_failure_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)

    class FailedClock:
        def now_ns(self):
            raise RuntimeError('permanently failed native clock')

    worker = runner.BoundRun(SimpleNamespace(mode='train', output=tmp_path))
    worker.owns_output, worker.clock, worker.start = True, FailedClock(), 1
    worker.receipt['pending'] = {'stage': 'solve'}
    worker.finish('original failure')
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['error'] == 'original failure'
    assert receipt['wall_seconds'] is receipt['finished_ns'] is None
    assert 'permanently failed' in receipt['clock_error']
    assert receipt['pending'] == {'stage': 'solve'}
