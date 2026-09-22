"""Preparation failure lifecycle using fabricated files and controlled stubs.

The fixture loads the runner only during pytest execution. Its numerical imports,
clock construction, metadata closure, and input paths are replaced before run().
No historical rows, arrays, real cohort, simulator, sampler, or model are used.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    source = Path(__file__).resolve().parents[1] / 'scripts/prepare_otto_teacher_cohort.py'
    spec = importlib.util.spec_from_file_location('_fabricated_cohort_preparation_tests', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(runner, tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.chdir(root)
    monkeypatch.setattr(runner, 'ROOT', root)
    case = SimpleNamespace(budgets=[], handlers={}, imported=[], on_check=None, on_reconstruct=None)
    args = SimpleNamespace(output=root / 'run', plan=root / 'plan.json',
                           supervision=root / 'launch.json')
    case.args = args
    sources = {runner.CLOCK: '1' * 64, 'scripts/supervise_dialogue_observation_v2.py': '2' * 64}
    selections = [{'anchor_id': 0, 'prefix_index': 0}]
    plan = {'status': 'frozen_before_reconstruction', 'sources': sources,
            'cohort': {'selections': selections}}
    runner.write(args.plan, plan)
    args.plan_sha256 = runner.digest(args.plan)['sha256']
    launch = {'command': [sys.executable, *sys.argv], 'cwd': str(root),
              'pid': os.getpid(), 'pgid': os.getpgrp(), 'parent_pid': os.getppid(),
              'cap_seconds': 600, 'clock_backend': 'fabricated',
              'clock_source_sha256': sources[runner.CLOCK],
              'watchdog_sha256': sources['scripts/supervise_dialogue_observation_v2.py'],
              'started_ns': 10_000, 'deadline_ns': 10_000 + 600 * 10**9}
    runner.write(args.supervision, launch)
    monkeypatch.setattr(runner, 'metadata', lambda _check: {'sources': sources})

    def install_handler(number, handler):
        assert number == signal.SIGTERM
        case.handlers[number] = handler

    monkeypatch.setattr(runner.signal, 'signal', install_handler)

    class FakeBudget(runner.Budget):
        def __init__(self, out):
            self.out = out
            self.clock = SimpleNamespace(backend='fabricated', now_ns=lambda: 30_000)
            self.start, self.deadline = 20_000, launch['deadline_ns']
            self.bytes = self.events = 0
            self.pending, self.calls = {}, {}
            case.budgets.append(self)

        def check(self):
            if case.on_check is not None:
                case.on_check(self)

    monkeypatch.setattr(runner, 'Budget', FakeBudget)
    transition = root / 'fabricated-transitions.jsonl'
    transition.write_bytes(b'fabricated public record\n')
    paths = {f'{runner.SYMM}/collection-transitions.jsonl': transition,
             **{f'{runner.SYMM}/kernel-{regime}.npz': root / f'fake-{regime}.npz'
                for regime in ('lambda3', 'lambda4')}}

    def public_path(name):
        assert name in paths, 'runner attempted an undeclared or historical input'
        return paths[name]

    monkeypatch.setattr(runner, 'path', public_path)

    class FakeKernel:
        files = ('likelihood', 'initial_hit_weights')

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, key):
            assert key == 'likelihood'
            return [0.25, 0.75]

    def load_kernel(path, *, allow_pickle):
        assert path in paths.values() and path != transition
        assert allow_pickle is False
        return FakeKernel()

    def project(lines, selected):
        assert selected == selections
        assert list(lines) == [b'fabricated public record\n']
        return ('projected fabricated record',)

    def reconstruct(selected, records, kernels, *, check, emit):
        assert selected == selections and records == ('projected fabricated record',)
        assert set(kernels) == {'lambda3', 'lambda4'}
        check()
        event = {'event': 'attempt', 'operation_id': 0, 'operation': 'public_reset',
                 'episode_id': 'fabricated-episode', 'step': 0}
        emit(event)
        if case.on_reconstruct is not None:
            case.on_reconstruct()
        emit({**event, 'event': 'return'})
        return {'anchors': ({'anchor_id': 0, 'belief': [1.0]},),
                'counts': {'anchors': 1, 'episodes': 1, 'public_resets': 1, 'public_updates': 0}}

    def support(anchors, kernels, *, check, emit):
        assert anchors == ({'anchor_id': 0, 'belief': [1.0]},)
        assert set(kernels) == {'lambda3', 'lambda4'}
        check()
        event = {'event': 'attempt', 'operation_id': 0, 'operation': 'anchor_support', 'anchor_id': 0}
        emit(event)
        emit({**event, 'event': 'return'})
        return {'assessments': [{'anchor_id': 0, 'supported': True, 'error': None}],
                'counts': {'anchors': 1, 'supported': 1, 'unsupported': 0,
                           'snapshot_attempts': 1, 'snapshot_returns': 1}}

    numpy_stub = SimpleNamespace(load=load_kernel, stack=list,
                                 savez=lambda stream, **_arrays: stream.write(b'fabricated array bytes\n'))
    helper_stub = SimpleNamespace(iter_public_records=project, reconstruct_cohort=reconstruct,
                                  assess_support=support)
    substitutes = {'numpy': numpy_stub,
                   'openjev.research': SimpleNamespace(otto_teacher_cohort=helper_stub)}
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in substitutes:
            case.imported.append(name)
            return substitutes[name]
        assert name != 'numpy' and not name.startswith(('numpy.', 'openjev.'))
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', isolated_import)
    return case


def failed_receipt(case):
    saved = json.loads((case.args.output / 'receipt.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['eligible_for_sampling'] is False
    for counter in ('native_calls', 'source_draws', 'analytic_choices', 'learned_model_calls', 'optimizer_calls'):
        assert saved[counter] == 0
    return saved


def test_budget_construction_failure_is_published_without_masking_original(runner, tmp_path, monkeypatch):
    case = make_case(runner, tmp_path, monkeypatch)
    original = RuntimeError('fabricated clock construction failure')

    def failed_budget(_out):
        raise original

    monkeypatch.setattr(runner, 'Budget', failed_budget)
    with pytest.raises(RuntimeError) as failure:
        runner.run(case.args)
    assert failure.value is original
    saved = failed_receipt(case)
    assert saved['error'] == repr(original)
    assert saved['calls'] == {} and saved['pending'] == []
    assert saved['failure_elapsed_seconds'] is None
    assert case.imported == [] and case.budgets == []
    assert not (case.args.output / 'started.json').exists()


def test_metadata_failure_is_preserved_before_any_numerical_work(runner, tmp_path, monkeypatch):
    case = make_case(runner, tmp_path, monkeypatch)
    original = ValueError('fabricated closure mismatch')

    def failed_metadata(check):
        check()
        raise original

    monkeypatch.setattr(runner, 'metadata', failed_metadata)
    with pytest.raises(ValueError) as failure:
        runner.run(case.args)
    assert failure.value is original
    saved = failed_receipt(case)
    assert saved['error'] == repr(original)
    assert saved['calls'] == {} and saved['pending'] == [] and saved['files'] == {}
    assert case.imported == []
    assert not (case.args.output / 'receipt.invalid.json').exists()


def test_sigterm_in_body_preserves_unfinished_attempt_and_failed_receipt(runner, tmp_path, monkeypatch):
    case = make_case(runner, tmp_path, monkeypatch)
    case.on_reconstruct = lambda: case.handlers[signal.SIGTERM](signal.SIGTERM, None)
    with pytest.raises(InterruptedError, match='supervisor terminated cohort preparation') as failure:
        runner.run(case.args)
    saved = failed_receipt(case)
    assert saved['error'] == repr(failure.value)
    assert saved['calls'] == {'public_reset': {'attempted': 1, 'returned': 0}}
    journal = case.args.output / 'reconstruction.jsonl'
    ledger = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(ledger) == 1 and ledger[0]['event'] == 'attempt'
    assert saved['pending'] == ledger
    assert saved['files']['reconstruction.jsonl'] == runner.digest(journal)
    assert not (case.args.output / 'anchors.npz').exists()
    assert not (case.args.output / 'support.json').exists()
    assert not (case.args.output / 'receipt.invalid.json').exists()


def test_late_budget_failure_demotes_completed_receipt_and_keeps_payloads(runner, tmp_path, monkeypatch):
    case = make_case(runner, tmp_path, monkeypatch)
    original = TimeoutError('fabricated budget expiration after publication')
    published = []

    def fail_after_publication(budget):
        target = budget.out / 'receipt.json'
        if target.exists():
            published.append(target.read_bytes())
            raise original

    case.on_check = fail_after_publication
    with pytest.raises(TimeoutError) as failure:
        runner.run(case.args)
    assert failure.value is original
    saved = failed_receipt(case)
    invalid = case.args.output / 'receipt.invalid.json'
    assert len(published) == 1 and invalid.read_bytes() == published[0]
    prior = json.loads(published[0])
    assert prior['status'] == 'completed' and prior['eligible_for_sampling'] is True
    assert prior['pending'] == [] and prior['events'] == 4
    assert saved['error'] == repr(original) and saved['pending'] == []
    assert saved['calls'] == {'public_reset': {'attempted': 1, 'returned': 1},
                              'anchor_support': {'attempted': 1, 'returned': 1}}
    assert saved['files']['receipt.invalid.json'] == runner.digest(invalid)
    for name, descriptor in prior['files'].items():
        assert saved['files'][name] == descriptor == runner.digest(case.args.output / name)
