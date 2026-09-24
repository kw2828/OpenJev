"""Fail-closed new-runner metadata and split-boundary checks, fabricated only."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('readout_runner_tests', ROOT / 'scripts/run_otto_readout_ablation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def plan():
    return {'version': runner.VERSION, 'status': 'registered_before_gradients',
            'configuration': copy.deepcopy(runner.CONFIG), 'limits': copy.deepcopy(runner.LIMITS),
            'sources': {'fabricated': '123'}, 'engineering': {'path': 'fabricated', 'sha256': '123', 'bytes': 1}}


def test_plan_requires_identical_configuration_and_evidence(monkeypatch):
    value = plan()
    auth = {'sources': value['sources']}
    monkeypatch.setattr(runner, 'authenticate', lambda: auth)
    monkeypatch.setattr(runner, 'engineering', lambda path: value['engineering'])
    assert runner.validate_plan(value) == auth
    value['configuration']['epochs'] = 41
    with pytest.raises(ValueError, match='recipe'):
        runner.validate_plan(value)
    value = plan()
    value['sources']['fabricated'] = '456'
    with pytest.raises(ValueError, match='sources'):
        runner.validate_plan(value)


@pytest.mark.parametrize('stage', ['test', 'confirm', 'validation', 'dev'])
def test_split_barrier_rejects_before_import_or_decode(stage, tmp_path):
    worker = runner.BoundRun.__new__(runner.BoundRun)
    worker.receipt = {'fits_completed': 8, 'optimizer_steps': 2880}
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


def test_fixed_experiment_roster_and_caps():
    assert runner.CONFIG['fits'] == len(runner.ARMS) * len(runner.SEEDS) == 9
    assert runner.CONFIG['views'] == len(runner.VIEWS) * len(runner.SEEDS) == 12
    assert runner.CONFIG['epochs'] * 54 // 6 == runner.CONFIG['updates_per_fit'] == 360
    assert runner.CONFIG['test_admitted'] is runner.CONFIG['held_out_evidence'] is False
    assert runner.LIMITS['train']['seconds'] == 5400
    assert runner.LIMITS['audit']['seconds'] == 600


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
               'teacher_calls': 0, 'native_calls': 0, 'test_array_decodes': 0, 'fits_completed': 9,
               'optimizer_steps': 3240, 'episode_exposures': 19440, 'views_completed': 12,
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
        receipt['fits_completed'] = 8
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
