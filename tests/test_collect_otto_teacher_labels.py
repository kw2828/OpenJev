"""Collector failure boundaries with fabricated streams and controlled stubs.

The source is loaded only by the test fixture. No historical records, numerical
arrays, real sampler, actor, native environment or model are used. The body test
replaces every numerical/import/input seam and stops at its first sampler stub.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def collector():
    path = Path(__file__).resolve().parents[1] / 'scripts/collect_otto_teacher_labels.py'
    spec = importlib.util.spec_from_file_location('_fabricated_teacher_collector_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    backend = 'fabricated'

    def __init__(self):
        self.tick = 0

    def now_ns(self):
        self.tick += 1_000
        return self.tick


def make_run(collector, tmp_path):
    root = tmp_path.resolve()
    run = collector.Run(SimpleNamespace(output=root / 'run', plan=root / 'unused-plan.json',
                                        supervision=root / 'unused-launch.json', plan_sha256='0' * 64))
    run.clock, run.start = Clock(), 0
    run.bind = lambda: None
    run.check = lambda: None
    return run


def operation(event='attempt', *, anchor_id=0, operation_id=0, name='teacher_update'):
    return {'event': event, 'anchor_id': anchor_id, 'operation_id': operation_id, 'operation': name}


def receipt(run):
    result = json.loads((run.out / 'receipt.json').read_text())
    assert result['status'] == 'failed' and result['technical_completion'] is False
    assert result['requires_successful_original_supervisor'] is True
    assert not (run.out / 'summary.json').exists()
    return result


def test_oversized_return_preserves_durable_attempt_without_fabricated_completion(collector, tmp_path):
    run = make_run(collector, tmp_path)
    attempted = operation()
    oversized = {**operation('return'), 'public': 'x' * (collector.EVENT_BYTES + 1)}

    def body():
        run.emit('events.jsonl', attempted, sampler=True)
        run.emit('events.jsonl', oversized, sampler=True)

    run.body = body
    with pytest.raises(ValueError, match='event exceeds frozen byte allowance'):
        run.execute()
    saved = receipt(run)
    assert saved['calls'] == {'sampler:teacher_update': {'attempted': 1, 'returned': 0}}
    assert saved['pending'] == [{'channel': 'sampler', **attempted}]
    assert saved['pending_emission'] == {'file': 'events.jsonl', 'event': oversized}
    assert saved['sampler_events'] == 1
    assert [json.loads(line) for line in (run.out / 'events.jsonl').read_text().splitlines()] == [attempted]


class WriteFault:
    def __init__(self, stream, failure, fail_at):
        self.stream, self.failure, self.fail_at = stream, failure, fail_at
        self.write_calls = 0

    @property
    def closed(self):
        return self.stream.closed

    def fileno(self):
        return self.stream.fileno()

    def write(self, value):
        self.write_calls += 1
        if self.write_calls == self.fail_at:
            raise self.failure
        return self.stream.write(value)

    def close(self):
        self.stream.close()


@pytest.mark.parametrize('failure_stage', ['attempt_write', 'return_write', 'return_fsync'])
def test_journal_failure_preserves_original_error_and_unfinished_operation(
        collector, tmp_path, monkeypatch, failure_stage):
    run = make_run(collector, tmp_path)
    original = OSError(f'fabricated {failure_stage} failure')
    attempted, returned = operation(), operation('return')
    opened = []
    real_fsync = collector.os.fsync

    def fsync(fd):
        if (opened and not opened[0].closed and fd == opened[0].fileno()
                and failure_stage == 'return_fsync' and opened[0].write_calls == 2):
            raise original
        return real_fsync(fd)

    monkeypatch.setattr(collector.os, 'fsync', fsync)

    def body():
        fail_at = {'attempt_write': 1, 'return_write': 2, 'return_fsync': None}[failure_stage]
        journal = WriteFault((run.out / 'events.jsonl').open('xb', buffering=0), original, fail_at)
        opened.append(journal)
        run.streams['events.jsonl'] = journal
        run.emit('events.jsonl', attempted, sampler=True)
        run.emit('events.jsonl', returned, sampler=True)

    run.body = body
    with pytest.raises(OSError) as failure:
        run.execute()
    assert failure.value is original
    saved = receipt(run)
    assert saved['error'] == repr(original)
    assert saved['calls'] == {'sampler:teacher_update': {'attempted': 1, 'returned': 0}}
    assert saved['pending'] == [{'channel': 'sampler', **attempted}]
    failed_event = attempted if failure_stage == 'attempt_write' else returned
    assert saved['pending_emission'] == {'file': 'events.jsonl', 'event': failed_event}
    assert saved['sampler_events'] == (0 if failure_stage == 'attempt_write' else 1)
    ledger = [json.loads(line) for line in (run.out / 'events.jsonl').read_text().splitlines()]
    expected = {'attempt_write': [], 'return_write': [attempted], 'return_fsync': [attempted, returned]}
    # Written return bytes survive a failed fsync, but are never counted as a
    # durably acknowledged return or used to clear the pending operation.
    assert ledger == expected[failure_stage]
    assert opened[0].closed


class CloseFault:
    def __init__(self, failure=None):
        self.failure, self.closed, self.close_calls = failure, False, 0

    def close(self):
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure
        self.closed = True


@pytest.mark.parametrize('previous_completed_receipt', [False, True])
def test_cleanup_error_cannot_skip_failed_receipt_or_demotion(
        collector, tmp_path, previous_completed_receipt):
    run = make_run(collector, tmp_path)
    original = ArithmeticError('original controlled body failure')
    cleanup_error = OSError('fabricated bad stream close')
    bad, healthy = CloseFault(cleanup_error), CloseFault()
    prior = {'status': 'completed', 'technical_completion': True, 'marker': 'previous publication'}

    def body():
        run.streams.update({'bad.log': bad, 'healthy.log': healthy})
        if previous_completed_receipt:
            collector.write(run.out / 'receipt.json', prior)
        raise original

    run.body = body
    with pytest.raises(ArithmeticError) as failure:
        run.execute()
    assert failure.value is original
    saved = receipt(run)
    assert saved['error'] == repr(original)
    assert bad.close_calls == healthy.close_calls == 1 and healthy.closed
    assert saved['cleanup_errors'] == [{'file': 'bad.log', 'error': repr(cleanup_error)}]
    invalid = run.out / 'receipt.invalid.json'
    assert invalid.exists() is previous_completed_receipt
    if previous_completed_receipt:
        assert json.loads(invalid.read_text()) == prior
        assert 'receipt.invalid.json' in saved['files']


@pytest.mark.parametrize('sixth_validation_returned', [False, True])
def test_body_blocks_sampling_until_all_six_validation_returns(
        collector, tmp_path, monkeypatch, sixth_validation_returned):
    run = make_run(collector, tmp_path)
    selections = [{'anchor_id': i, 'prefix_index': 1} for i in range(6)]
    run.plan = {'selected_anchors': selections}
    sampled, visited = [], []
    stopped = RuntimeError('stop at first fabricated sampler entry')
    anchors = [{'anchor_id': i, 'regime': 'lambda3' if i < 3 else 'lambda4',
                'public': {'valid_actions': [0, 1]}, 'belief': [i]} for i in range(6)]
    transition_file = tmp_path.resolve() / 'fabricated-transitions.jsonl'
    transition_file.write_bytes(b'fabricated public record\n')
    paths = {f'{collector.SYMM}/run-01/collection-transitions.jsonl': transition_file}
    for regime in ('lambda3', 'lambda4'):
        paths[f'{collector.SYMM}/run-01/kernel-{regime}.npz'] = tmp_path.resolve() / f'fake-{regime}.npz'

    def relative(name):
        assert name in paths, 'body tried to read an undeclared or real study input'
        return paths[name]

    monkeypatch.setattr(collector, 'relative', relative)
    monkeypatch.setattr(collector.sys, 'path', list(collector.sys.path))

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
        assert path in paths.values() and path != transition_file
        assert allow_pickle is False
        return FakeKernel()

    def project(lines, selected):
        assert selected is selections
        assert list(lines) == [b'fabricated public record\n']
        return ('projected fabricated record',)

    def reconstruct(selected, projected, kernels, *, check, emit):
        assert selected is selections and projected == ('projected fabricated record',)
        assert set(kernels) == {'lambda3', 'lambda4'}
        for row in selected:
            for ordinal, name in enumerate(('public_reset', 'public_update', 'anchor_validation')):
                check()
                event = operation(anchor_id=row['anchor_id'], operation_id=ordinal, name=name)
                emit(event)
                if name == 'anchor_validation' and row['anchor_id'] == 5 and not sixth_validation_returned:
                    continue
                emit({**event, 'event': 'return'})
                if name == 'anchor_validation':
                    visited.append(row['anchor_id'])
        return anchors

    def sample(*_args, **kwargs):
        sampled.append(kwargs['anchor_id'])
        assert visited == list(range(6))
        assert run.calls['reconstruction:anchor_validation'] == {'attempted': 6, 'returned': 6}
        assert not run.pending
        raise stopped

    def forbidden_reduce(*_args, **_kwargs):
        raise AssertionError('test must stop at the first sampler stub')

    numpy_stub = SimpleNamespace(load=load_kernel, stack=list,
                                 savez=lambda stream, **_arrays: stream.write(b'fabricated array bytes\n'))
    helper_stub = SimpleNamespace(iter_public_records=project, reconstruct_anchors=reconstruct)
    substitutes = {
        'numpy': numpy_stub,
        'openjev.research': SimpleNamespace(otto_teacher_anchors=helper_stub),
        'openjev.research.otto_teacher_costs': SimpleNamespace(summarize_costs=forbidden_reduce),
        'openjev.research.otto_teacher_rollouts': SimpleNamespace(sample_teacher_panel=sample),
    }
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in substitutes:
            return substitutes[name]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', isolated_import)
    expected = RuntimeError if sixth_validation_returned else ValueError
    with pytest.raises(expected) as failure:
        run.execute()
    saved = receipt(run)
    assert sampled == ([0] if sixth_validation_returned else [])
    assert visited == list(range(6 if sixth_validation_returned else 5))
    if sixth_validation_returned:
        assert failure.value is stopped and saved['active_panel'] == 0
    else:
        assert 'all six anchors validated before source generation' in str(failure.value)
        assert saved['pending'][0]['anchor_id'] == 5
        assert saved['pending'][0]['operation'] == 'anchor_validation'
        assert not (run.out / 'anchors.npz').exists()
