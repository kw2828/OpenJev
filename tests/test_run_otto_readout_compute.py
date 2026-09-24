"""Fail-closed new-runner metadata and split-boundary checks, fabricated only."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('readout_compute_runner_tests', ROOT / 'scripts/run_otto_readout_compute.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def plan():
    evidence = {'path': 'fabricated', 'sha256': '123', 'bytes': 1}
    return {'version': runner.VERSION, 'status': 'registered_before_gradients',
            'configuration': copy.deepcopy(runner.CONFIG), 'limits': copy.deepcopy(runner.LIMITS),
            'sources': {'fabricated': '123'}, 'engineering': evidence, 'collection_engineering': evidence}


def test_plan_requires_identical_configuration_and_evidence(monkeypatch):
    value = plan()
    auth = {'sources': value['sources']}
    monkeypatch.setattr(runner, 'authenticate', lambda: auth)
    monkeypatch.setattr(runner, 'engineering', lambda path: value['engineering'])
    assert runner.validate_plan(value) == auth
    value['configuration']['epochs']['action_residual_only'] = 75
    with pytest.raises(ValueError, match='recipe'):
        runner.validate_plan(value)
    value = plan()
    value['sources']['fabricated'] = '456'
    with pytest.raises(ValueError, match='sources'):
        runner.validate_plan(value)


@pytest.mark.parametrize('stage', ['test', 'confirm', 'validation', 'dev'])
def test_split_barrier_rejects_before_import_or_decode(stage, tmp_path):
    worker = runner.BoundRun.__new__(runner.BoundRun)
    worker.receipt = {'fits_completed': 5, 'optimizer_steps': 2412}
    worker.out = tmp_path
    with pytest.raises(ValueError, match='TRAIN and DEV only|all final fits'):
        worker.history(stage)


def test_descriptor_rejects_external_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    target = tmp_path / 'input.json'
    target.write_text('{}')
    assert runner.descriptor(target)['bytes'] == 2
    link = tmp_path / 'link.json'
    link.symlink_to(target)
    with pytest.raises(ValueError, match='regular contained'):
        runner.descriptor(link)
    with pytest.raises(ValueError, match='regular contained'):
        runner.descriptor(Path(__file__).resolve())


def test_engineering_cannot_admit_failed_or_changed_qualification(monkeypatch):
    record = {'status': 'failed'}
    monkeypatch.setattr(runner, 'read', lambda path: record)
    with pytest.raises(ValueError, match='qualification'):
        runner.engineering('fabricated')
    record.update(status='passed', sources_before={'x': 'a'}, sources_after={'x': 'b'}, commands=[])
    with pytest.raises(ValueError, match='qualification'):
        runner.engineering('fabricated')


@pytest.mark.parametrize('change', [None, 'missing_native_manifest', 'native_call', 'capacity'])
def test_engineering_shared_descriptor_schema_and_native_boundary(monkeypatch, tmp_path, change):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'NEW', {'bound.py'})
    (tmp_path / 'bound.py').write_text('# fabricated source\n')
    capacity = {'status': 'passed', 'empirical_array_decodes': 0, 'checkpoint_decodes': 0,
                'projected_seconds': 4050 if change != 'capacity' else 4051}
    names = ('array_decodes', 'checkpoint_decodes', 'model_calls', 'teacher_calls', 'native_calls',
             'optimizer_steps', 'confirmation_decodes', 'old_test_decodes')
    native = {'status': 'passed', 'metadata_import_guard': True, 'counts': dict.fromkeys(names, 0)}
    if change == 'native_call':
        native['counts']['native_calls'] = 1
    runner.write(tmp_path / 'capacity.json', capacity)
    runner.write(tmp_path / 'native-preflight.json', native)
    sources = {'bound.py': runner.descriptor(tmp_path / 'bound.py')['sha256']}
    files = {name: {k: runner.descriptor(tmp_path / name)[k] for k in ('sha256', 'bytes')}
             for name in ('capacity.json', 'native-preflight.json')}
    if change == 'missing_native_manifest':
        files.pop('native-preflight.json')
    receipt = {'status': 'passed', 'sources_before': sources, 'sources_after': sources,
               'metadata_handoff_passed': True, 'files': files,
               'commands': [{'returncode': 0, 'timed_out': False, 'reaped': True, 'group_absent': True} for _ in range(4)],
               'empirical_array_decodes': 0, 'checkpoint_decodes': 0, 'scientific_calls': 0}
    path = tmp_path / 'receipt.json'
    runner.write(path, receipt)
    if change is None:
        assert runner.engineering(path) == runner.descriptor(path)
    else:
        with pytest.raises(ValueError):
            runner.engineering(path)


def test_fixed_experiment_roster_and_caps():
    assert runner.CONFIG['fits'] == len(runner.ARMS) * len(runner.SEEDS) == 6
    assert runner.CONFIG['views'] == len(runner.VIEWS) * len(runner.SEEDS) == 9
    assert runner.EPOCHS == {'action_residual_only': 74, 'full_joint': 40}
    assert runner.CONFIG['updates_per_fit'] == {'action_residual_only': 666, 'full_joint': 360}
    assert 3 * (666 + 360) == runner.UPDATES == runner.CONFIG['optimizer_steps'] == 3078
    assert 6 * runner.UPDATES == runner.EXPOSURES == runner.CONFIG['episode_exposures'] == 18468
    assert runner.CONFIG['dev_episodes'] == 36
    assert runner.CONFIG['dev_reused'] is False and runner.CONFIG['fresh_dev'] is True
    assert runner.CONFIG['held_out_from_training'] is runner.CONFIG['development_only'] is True
    assert runner.CONFIG['test_admitted'] is runner.CONFIG['held_out_evidence'] is False
    assert runner.LIMITS['train']['seconds'] == 5400
    assert runner.LIMITS['audit']['seconds'] == 600
    assert len(runner.expected_payloads('train')) == 27
    assert len(runner.expected_payloads('audit')) == 3
    assert runner.CONFIG['fit_time_ratio'] == {
        'numerator': 'action_residual_only', 'denominator': 'full_joint',
        'inclusive_lower': .9, 'inclusive_upper': 1.1, 'every_seed': True}


def test_plan_rejects_qualification_different_from_collection(monkeypatch):
    value = plan()
    monkeypatch.setattr(runner, 'authenticate', lambda: {'sources': value['sources']})
    monkeypatch.setattr(runner, 'engineering', lambda path: value['engineering'])
    value['collection_engineering'] = {**value['engineering'], 'sha256': 'different'}
    with pytest.raises(ValueError, match='qualification receipt'):
        runner.validate_plan(value)


def test_qualification_prior_authentication_never_requires_fresh_collection(monkeypatch):
    old = {'sources': {runner.ORIGINAL: runner.ORIGINAL_SHA256}, 'runtime': {'fixture': True},
           'train': {'identities': []}, 'dev': {'identities': []}}
    loaded = []

    def load(path, name):
        loaded.append(path)
        assert path == runner.ORIGINAL
        return SimpleNamespace(authenticate=lambda: old)

    monkeypatch.setattr(runner, 'load', load)
    monkeypatch.setattr(runner, 'descriptor', lambda p: {'sha256': runner.ORIGINAL_SHA256 if p == runner.ORIGINAL else 'new'})
    actual = runner.authenticate_prior()
    assert loaded == [runner.ORIGINAL]
    assert actual['sources'].keys() == {runner.ORIGINAL} | runner.NEW
    assert old['sources'] == {runner.ORIGINAL: runner.ORIGINAL_SHA256}
    assert actual['runtime'] == old['runtime']


@pytest.fixture
def fresh_bridge(monkeypatch):
    prior = {'sources': {'source.py': 'pin'}, 'runtime': {'fixture': True},
             'train': {'identities': [{'seed': 11}]}, 'dev': {'identities': [{'seed': 22}]},
             'bridge_inputs': {'old': 'replaced'}, 'lineage': {'checkpoints': 'prior'}}
    identities = [{'stage': 'dev', 'episode_id': str(i), 'episode_index': i,
                   'seed': 100 + i // 3} for i in range(36)]
    collection = {'execution_cohort': identities, 'inputs': {'engineering': {'path': 'qualified.json'}}}
    receipt = {'files': {'dev.npz': {'sha256': 'pin', 'bytes': 1}}}
    calls = []

    def descriptor(path):
        return {'path': str(path), 'sha256': 'pin', 'bytes': 1}

    def verify_bridge(inputs):
        calls.append(inputs)
        return collection, receipt, Path('/fixture/fresh')

    monkeypatch.setattr(runner, 'authenticate_prior', lambda: copy.deepcopy(prior))
    monkeypatch.setattr(runner, 'descriptor', descriptor)
    monkeypatch.setattr(runner, 'read', lambda path: {'sources': {'source.py': 'pin'}})
    monkeypatch.setattr(runner, 'load', lambda path, name: SimpleNamespace(verify_bridge=verify_bridge))
    return prior, collection, receipt, calls


def test_fresh_bridge_replaces_only_dev_and_binds_collection_qualification(fresh_bridge):
    prior, collection, _, calls = fresh_bridge
    actual = runner.authenticate()
    assert len(calls) == 1 and set(calls[0]) == {
        'bridge_receipt', 'bridge_terminal', 'collection_plan', 'collection_receipt', 'collection_terminal'}
    assert actual['dev']['identities'] == collection['execution_cohort']
    assert actual['dev']['descriptor']['path'] == '/fixture/fresh/dev.npz'
    assert actual['train'] == prior['train'] and actual['lineage'] == prior['lineage']
    assert actual['bridge_inputs'] == calls[0] and actual['bridge_inputs'] != prior['bridge_inputs']
    assert actual['collection_engineering']['path'] == 'qualified.json'


@pytest.mark.parametrize('change', ['short', 'old_dev_seed', 'train_seed', 'duplicate_id', 'wrong_stage', 'reordered', 'payload'])
def test_fresh_bridge_cannot_admit_reused_or_incomplete_paths(fresh_bridge, change):
    _, collection, receipt, _ = fresh_bridge
    rows = collection['execution_cohort']
    if change == 'short':
        rows.pop()
    elif change == 'old_dev_seed':
        rows[0]['seed'] = 22
    elif change == 'train_seed':
        rows[0]['seed'] = 11
    elif change == 'duplicate_id':
        rows[0]['episode_id'] = rows[1]['episode_id']
    elif change == 'wrong_stage':
        rows[0]['stage'] = 'test'
    elif change == 'reordered':
        rows.reverse()
    else:
        receipt['files']['dev.npz']['sha256'] = 'changed'
    with pytest.raises(ValueError):
        runner.authenticate()


def test_fresh_bridge_rejects_sources_outside_qualified_map(fresh_bridge, monkeypatch):
    monkeypatch.setattr(runner, 'read', lambda path: {'sources': {'unknown.py': 'pin'}})
    with pytest.raises(ValueError, match='bridge sources'):
        runner.authenticate()


def test_fixed_epoch_fit_orders_and_optimizer_counts(monkeypatch, tmp_path):
    """Run only orchestration with fake models/updates, never a numerical model."""
    import numpy as np

    from openjev import research

    class Value:
        def detach(self):
            return self

        def clone(self):
            return Value()

        def numpy(self):
            return np.zeros(1, np.float32)

    class Model:
        def __init__(self):
            self.frozen, self.head = Value(), Value()
            self.slow = SimpleNamespace(state_dict=lambda: {'frozen': self.frozen, 'head': self.head})

        def named_parameters(self):
            return [('slow.frozen', self.frozen), ('slow.head', self.head)]

        def effective_named_parameters(self):
            return [('slow.head', self.head)]

        def state_dict(self):
            return dict(self.named_parameters())

        def parameter_metadata(self):
            return {'fabricated': True}

    def construct_optimizer(model):
        return SimpleNamespace(state={model.head: {'step': 0}})

    observed = []

    def batch_update(model, optimizer, history, indices, *, check, stage):
        assert len(indices) == len(set(indices)) == 6
        check()
        stage('fabricated_update', None)
        observed.append(indices)
        optimizer.state[model.head]['step'] += 1
        return {'optimizer_step': optimizer.state[model.head]['step'], 'optimizer_updates': 1}

    fake_models = SimpleNamespace(from_state=lambda family, seed, parent: Model())
    fake_training = SimpleNamespace(construct_optimizer=construct_optimizer, batch_update=batch_update)
    for name, value in (('otto_readout_ablation_model', fake_models), ('otto_readout_ablation_training', fake_training)):
        monkeypatch.setattr(research, name, value, raising=False)
        monkeypatch.setitem(sys.modules, 'openjev.research.' + name, value)
    old = SimpleNamespace(state_witness=lambda model: {'tensors': {'slow.frozen': 'fixed', 'slow.head': 'fixed'}})
    monkeypatch.setattr(runner, 'load', lambda path, name: old)
    rows = {}
    for family in runner.ARMS:
        worker = runner.BoundRun.__new__(runner.BoundRun)
        worker.np, worker.out = np, tmp_path
        worker.clock = SimpleNamespace(now_ns=lambda: 1)
        worker.receipt = {'optimizer_steps': 0, 'episode_exposures': 0, 'fits_completed': 0}
        events, observed = [], []
        worker.event = lambda value, filename='work.jsonl', events=events: events.append((filename, value))
        worker.check = lambda: None
        worker.save_arrays = lambda name, arrays: {'path': name, 'sha256': 'fabricated', 'bytes': 1}
        _, row = worker.fit(family, 309000001, {}, {'fabricated': True})
        rows[family] = row
        expected = 666 if family == 'action_residual_only' else 360
        assert len(observed) == row['steps'] == worker.receipt['optimizer_steps'] == expected
        assert row['exposures'] == worker.receipt['episode_exposures'] == expected * 6
        assert row['epochs'] == len(row['orders']) == (74 if family == 'action_residual_only' else 40)
        assert row['optimizer_steps'] == {'slow.head': expected}
        assert row['frozen_names'] == ['slow.frozen'] and row['frozen_unchanged'] is True
        assert len(events) == 2 * expected + 1 and events[-1][0] == 'progress.jsonl'
        assert all(sorted(order) == list(range(54)) for order in row['orders'])
        assert worker.receipt['fits_completed'] == 1 and worker.receipt['pending'] is None
    assert rows['action_residual_only']['orders'][:40] == rows['full_joint']['orders']


def test_check_enforces_resource_and_deadline(monkeypatch, tmp_path):
    worker = runner.BoundRun.__new__(runner.BoundRun)
    worker.clock = SimpleNamespace(now_ns=lambda: 10)
    worker.launch = {'deadline_ns': 10}
    worker.phase, worker.out, worker.receipt = 'train', tmp_path, {}
    with pytest.raises(ValueError, match='deadline'):
        worker.check()


@pytest.fixture
def closed_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'validate_plan', lambda value: None)
    directory = tmp_path / 'train'
    directory.mkdir()
    planpath, receiptpath, terminalpath = tmp_path / 'plan.json', directory / 'receipt.json', tmp_path / 'terminal.json'
    sources = {runner.SUPERVISOR: 'watchdog', runner.CLOCK: 'clock'}
    runner.write(planpath, {'sources': sources})
    launchpath = tmp_path / 'launch.json'
    command = [str(tmp_path / '.venv/bin/python'), str(tmp_path / runner.SELF), 'train',
               '--plan', str(planpath), '--plan-sha256', runner.descriptor(planpath)['sha256'],
               '--supervision', str(launchpath), '--output', str(directory)]
    launch = {'command': command, 'pid': 10, 'pgid': 10, 'parent_pid': 9, 'cwd': str(tmp_path),
              'started_ns': 0, 'deadline_ns': 5400 * 10**9, 'cap_seconds': 5400,
              'watchdog_sha256': 'watchdog', 'clock_source_sha256': 'clock', 'clock_backend': 'mach_continuous_time'}
    runner.write(launchpath, launch)
    for name in runner.expected_payloads('train'):
        runner.write(directory / name, {'launch': launch, 'started_ns': 1} if name == 'started.json' else {})
    receipt = {'phase': 'train', 'version': runner.VERSION, 'status': 'completed', 'pending': None,
               'plan': runner.descriptor(planpath), 'supervision': runner.descriptor(launchpath),
               'started_ns': 1, 'finished_ns': 50, 'sources': sources, 'limits': runner.LIMITS['train'],
               'teacher_calls': 0, 'native_calls': 0, 'test_array_decodes': 0, 'fits_completed': 6,
               'optimizer_steps': 3078, 'episode_exposures': 18468, 'views_completed': 9,
               'checkpoint_decodes': 3, 'array_decodes': 5,
               'files': {p.name: runner.descriptor(p) for p in directory.iterdir()}}
    terminal = {**launch, 'finished_ns': 60, 'status': 'completed', 'returncode': 0, 'timed_out': False,
                'error': None, 'clock_error': None, 'group_absent': True, 'timing_available': True,
                'cleanup': {'reaped': True, 'group_absent': True, 'errors': []}}
    return planpath, receiptpath, terminalpath, receipt, terminal


def test_original_process_closure_accepts_complete_fixture(closed_process):
    planpath, receiptpath, terminalpath, receipt, terminal = closed_process
    runner.write(receiptpath, receipt)
    runner.write(terminalpath, terminal)
    assert runner.process_closure(planpath, receiptpath, terminalpath)[1] == receipt


@pytest.mark.parametrize('change', ['partial_fit', 'missing_payload', 'outside_time', 'cleanup_error', 'changed_command'])
def test_original_process_closure_rejects_incomplete_or_mismatched_fixture(closed_process, change):
    planpath, receiptpath, terminalpath, receipt, terminal = closed_process
    if change == 'partial_fit':
        receipt['fits_completed'] = 5
    elif change == 'missing_payload':
        receipt['files'].pop('fits.json')
    elif change == 'outside_time':
        receipt['finished_ns'] = 61
    elif change == 'cleanup_error':
        terminal['cleanup']['errors'] = ['failed']
    else:
        terminal['command'][2] = 'audit'
    runner.write(receiptpath, receipt)
    runner.write(terminalpath, terminal)
    with pytest.raises(ValueError):
        runner.process_closure(planpath, receiptpath, terminalpath)


def test_failed_clock_still_writes_failure_receipt(tmp_path):
    class FailedClock:
        def now_ns(self):
            raise RuntimeError('permanently failed native clock')
    worker = runner.BoundRun.__new__(runner.BoundRun)
    worker.out, worker.owns_output, worker.clock, worker.start = tmp_path, True, FailedClock(), 1
    worker.receipt = {'status': 'started'}
    worker.finish('original failure')
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['error'] == 'original failure'
    assert receipt['wall_seconds'] is receipt['finished_ns'] is None
    assert 'permanently failed' in receipt['clock_error']
