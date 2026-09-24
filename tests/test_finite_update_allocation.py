"""Exact-update controller qualification using fabricated clocks and states only."""
import copy
import hashlib
import json

import pytest

import openjev.research.finite_update_allocation as allocation
from openjev.research.finite_training_allocation import Hooks, Snapshot


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Clock:
    def __init__(self):
        self.value = 0.
        self.calls = 0
        self.bad = None

    def __call__(self):
        self.calls += 1
        if self.bad is not None:
            value, self.bad = self.bad, None
            return value
        return self.value


class Backend:
    """Owned two-parameter state and nested momentum, without numerical imports."""
    def __init__(self, durations, *, setup=0., optimizer_seconds=0., callback_seconds=None,
                 restore_seconds=0.):
        self.clock = Clock()
        self.durations = list(durations)
        self.model = {'dynamic': 1., 'head': 2.}
        self.optimizers, self.calls, self.checkpoints = [], [], []
        self.validation_calls = 0
        self.setup, self.optimizer_seconds = setup, optimizer_seconds
        self.callback_seconds = dict(callback_seconds or {})
        self.restore_seconds = restore_seconds
        self.failure = None
        self.update_error = None
        self.callback_error = None
        self.callback_mutation = None
        self.bad_result = False
        self.bad_clock = None
        self.restore_error = None
        self.summary_calls = 0

    def validate(self):
        self.validation_calls += 1
        self.clock.value += self.setup
        return {'fixture': 'fabricated exact update schedule'}

    def new_optimizer(self, kind):
        optimizer = {'kind': kind, 'step': 0, 'state': {'moment': 0., 'square': 0.}}
        self.optimizers.append(optimizer)
        self.clock.value += self.optimizer_seconds
        return optimizer

    def model_hash(self):
        return digest(self.model)

    def optimizer_hash(self, optimizer):
        return digest(optimizer)

    def snapshot_model(self):
        return Snapshot(copy.deepcopy(self.model), self.model_hash(), 2, 16)

    def snapshot_optimizer(self, optimizer):
        return Snapshot(copy.deepcopy(optimizer), self.optimizer_hash(optimizer), 3, 24)

    def restore_model(self, state):
        self.clock.value += self.restore_seconds
        if self.restore_error is not None:
            raise self.restore_error
        self.model = copy.deepcopy(state)

    def restore_optimizer(self, optimizer, state):
        self.clock.value += self.restore_seconds
        optimizer.clear()
        optimizer.update(copy.deepcopy(state))

    def update(self, kind, optimizer, cursor):
        index = len(self.calls)
        assert index < len(self.durations), 'unexpected extra or retried update'
        assert (cursor is None) == (kind == 'prefix')
        self.calls.append({'kind': kind, 'cursor': cursor, 'optimizer_identity': id(optimizer)})
        gradient = 2. if kind == 'prefix' else float(cursor + 1)
        optimizer['state']['moment'] = .9 * optimizer['state']['moment'] + .1 * gradient
        optimizer['state']['square'] += gradient * gradient
        optimizer['step'] += 1
        self.model['dynamic'] -= optimizer['state']['moment']
        if kind == 'joint':
            self.model['head'] -= optimizer['state']['moment']
        self.clock.value += self.durations[index]
        if self.bad_clock is not None:
            self.clock.bad = self.bad_clock
        if self.update_error is not None:
            raise self.update_error
        return {'loss': float('nan') if self.bad_result else self.model['dynamic'] ** 2,
                'diagnostics': {'cursor': cursor, 'kind': kind},
                'work': {'gradient_passes': 1, 'event_exposures': 7}}

    def callback(self, label, optimizer, cursor):
        self.clock.value += self.callback_seconds.get(label, 0.)
        if self.callback_mutation == label:
            self.model['head'] = -99.
            optimizer['state']['moment'] = -99.
        if self.failure == label:
            raise self.callback_error
        return {'synthetic_record': label, 'cursor': cursor}

    def checkpoint(self, label, optimizer, cursor):
        self.checkpoints.append({'label': label, 'model': copy.deepcopy(self.model),
            'optimizer': copy.deepcopy(optimizer), 'cursor': cursor})
        return self.callback(label, optimizer, cursor)

    def final_summary(self, optimizer, cursor):
        self.summary_calls += 1
        return self.callback('summary', optimizer, cursor)

    def hooks(self):
        return Hooks(**{name: getattr(self, name) for name in Hooks.__dataclass_fields__})

    def run(self, prefix_updates=2, joint_updates=3, **kwargs):
        return allocation.run_update_allocation(self.hooks(), prefix_updates=prefix_updates,
            joint_updates=joint_updates, clock=self.clock, **kwargs)


def test_exact_exposure_cursor_fresh_optimizer_and_owned_evidence():
    backend = Backend([1., 2., 3., 4., 5.], setup=.5, optimizer_seconds=.25,
                      callback_seconds={'initial': .5, 'boundary': .5, 'final': .5, 'summary': .5})
    result = backend.run(max_seconds=20.)
    assert allocation.Hooks is Hooks and allocation.Snapshot is Snapshot
    assert result['status'] == 'PASS' and result['termination'] == 'completed_updates'
    assert result['version'] == 'finite-update-allocation-v1'
    assert result['arm'] == 'prefix_then_joint' and result['compute_matched'] is False
    assert result['accepted_prefix_updates'] == 2 and result['accepted_joint_updates'] == 3
    assert result['attempted_updates'] == result['accepted_updates'] == 5
    assert [row['cursor'] for row in backend.calls] == [None, None, 0, 1, 2]
    assert len({row['optimizer_identity'] for row in backend.calls}) == 2
    assert backend.optimizers[0]['step'] == 2 and backend.optimizers[1]['step'] == 3
    assert result['joint_cursor'] == 3
    assert result['boundary']['optimizer_reset'] and result['boundary']['joint_cursor'] == 0
    assert result['boundary']['optimizer_before_sha256'] != result['boundary']['optimizer_after_sha256']
    assert backend.checkpoints[0]['model']['head'] == backend.checkpoints[1]['model']['head'] == 2.
    assert [row['target_updates'] for row in result['stages']] == [2, 3]
    assert [row['deadline_seconds'] for row in result['stages']] == [20., 20.]
    assert all(row['termination'] == 'completed_updates' for row in result['stages'])
    assert all(row['accepted'] and not row['rolled_back'] and row['error'] is None for row in result['trace'])
    assert result['update_work'] == {'gradient_passes': 5, 'event_exposures': 35}
    assert result['timed_seconds'] == 18. and result['overrun_seconds'] == 0.
    assert result['final_summary_seconds'] == .5
    assert result['final_summary_record']['completed_elapsed'] == result['timed_seconds']
    assert result['checkpoints'][-1]['model_sha256'] == result['final_model_sha256'] == backend.model_hash()
    preserved = copy.deepcopy(result)
    backend.model['dynamic'] = 123.
    backend.optimizers[-1]['state']['moment'] = 456.
    assert result == preserved
    json.dumps(result, allow_nan=False)


def test_success_receipt_counts_every_snapshot_hash_clock_and_summary_once():
    backend = Backend([0.] * 5)
    result = backend.run()
    counts = result['work']
    assert set(counts) == set(allocation.WORK_KEYS)
    assert counts['model_hash_calls'] == counts['optimizer_hash_calls'] == 25
    assert counts['check_calls'] == 30
    assert counts['clock_calls'] == backend.clock.calls == 27
    assert counts['model_snapshot_calls'] == counts['optimizer_snapshot_calls'] == 9
    assert counts['model_snapshot_tensors'] == 18 and counts['model_snapshot_bytes'] == 144
    assert counts['optimizer_snapshot_tensors'] == 27 and counts['optimizer_snapshot_bytes'] == 216
    assert counts['prefix_update_calls'] == 2 and counts['joint_update_calls'] == 3
    assert counts['optimizer_constructions'] == 2 and counts['checkpoint_calls'] == 3
    assert counts['validation_calls'] == counts['final_summary_calls'] == 1
    assert counts['model_restore_calls'] == counts['optimizer_restore_calls'] == 0
    summary = result['final_summary_work']
    expected = dict.fromkeys(allocation.WORK_KEYS, 0)
    expected.update(check_calls=3, clock_calls=2, model_hash_calls=2, optimizer_hash_calls=2,
                    model_snapshot_calls=1, model_snapshot_tensors=2, model_snapshot_bytes=16,
                    optimizer_snapshot_calls=1, optimizer_snapshot_tensors=3, optimizer_snapshot_bytes=24,
                    final_summary_calls=1)
    assert summary == expected
    assert all(value <= counts[key] for key, value in summary.items())
    assert all(row['work_delta']['model_snapshot_calls'] == 1 for row in result['trace'])


def test_exact_counts_do_not_change_with_hook_duration_or_available_time():
    fast, slow = Backend([0.] * 5), Backend([10.] * 5)
    a, b = fast.run(max_seconds=1.), slow.run(max_seconds=60.)
    assert fast.model == slow.model and fast.optimizers == slow.optimizers
    assert a['accepted_updates'] == b['accepted_updates'] == 5
    assert a['timed_seconds'] == 0. and b['timed_seconds'] == 50.


@pytest.mark.parametrize('prefix,joint', [(True, 1), (1, False), (0, 1), (1, -1),
                                       (1., 1), (1, 2.), (100000, 1)])
def test_invalid_update_counts_rejected_before_hooks(prefix, joint):
    backend = Backend([])
    with pytest.raises(ValueError):
        backend.run(prefix, joint)
    assert backend.validation_calls == 0
    assert not backend.optimizers and not backend.calls and not backend.checkpoints


@pytest.mark.parametrize('limit', [True, 0., -1., float('nan'), float('inf'), 10 ** 1000])
def test_invalid_safety_limit_is_json_safe_and_precedes_hooks(limit):
    backend = Backend([])
    with pytest.raises(ValueError) as caught:
        backend.run(max_seconds=limit)
    assert backend.validation_calls == 0 and not backend.optimizers
    json.dumps(caught.value.allocation_result, allow_nan=False)


@pytest.mark.parametrize('where', ['validation', 'optimizer'])
def test_setup_consumes_original_safety_deadline(where):
    backend = Backend([], setup=2. if where == 'validation' else 0.,
                      optimizer_seconds=2. if where == 'optimizer' else 0.)
    with pytest.raises(TimeoutError) as caught:
        backend.run(max_seconds=1.)
    result = caught.value.allocation_result
    assert result['status'] == 'FAILED_TIMEOUT' and result['timed_seconds'] == 2.
    assert result['attempted_updates'] == 0 and not backend.checkpoints
    assert result['work']['validation_calls'] == 1


@pytest.mark.parametrize('stage', ['prefix', 'joint'])
def test_late_attempt_rolls_back_moments_cursor_and_charges_rollback_time(stage):
    backend = Backend([2.] if stage == 'prefix' else [.25, 2.], restore_seconds=.25)
    original = copy.deepcopy(backend.model)
    with pytest.raises(TimeoutError) as caught:
        backend.run(1, 1, max_seconds=1.)
    result = caught.value.allocation_result
    row = result['trace'][-1]
    assert result['status'] == 'FAILED_TIMEOUT' and row['rolled_back'] and not row['accepted']
    assert row['cursor_retained'] == result['joint_cursor'] == 0
    assert row['model_retained_sha256'] == row['model_before_sha256'] == backend.model_hash()
    assert row['optimizer_retained_sha256'] == row['optimizer_before_sha256']
    assert backend.optimizers[-1]['step'] == 0
    assert backend.optimizers[-1]['state'] == {'moment': 0., 'square': 0.}
    assert row['rollback_completed_elapsed'] == row['completed_elapsed'] + .5
    assert result['timed_seconds'] == row['rollback_completed_elapsed']
    assert result['overrun_seconds'] == result['timed_seconds'] - 1.
    assert result['update_work']['gradient_passes'] == (1 if stage == 'prefix' else 2)
    assert result['work']['model_restore_bytes'] == 16 and result['work']['optimizer_restore_bytes'] == 24
    assert backend.summary_calls == 0
    if stage == 'prefix':
        assert backend.model == original and len(backend.optimizers) == 1
    else:
        assert backend.model == backend.checkpoints[1]['model']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['exception', 'result', 'clock', 'check'])
def test_failed_update_preserves_original_exception_and_atomic_state(failure):
    backend = Backend([.25])
    original = copy.deepcopy(backend.model)
    sentinel = RuntimeError('fabricated update/check exception')
    backend.update_error = sentinel if failure == 'exception' else None
    backend.bad_result = failure == 'result'
    backend.bad_clock = -1. if failure == 'clock' else None

    def check():
        if failure == 'check' and backend.calls:
            raise sentinel

    with pytest.raises((RuntimeError, ValueError)) as caught:
        backend.run(1, 1, check=check)
    if failure in ('exception', 'check'):
        assert caught.value is sentinel
    result = caught.value.allocation_result
    assert result['status'] == 'FAILED_EXCEPTION'
    assert backend.model == original and backend.optimizers[0]['step'] == 0
    assert result['trace'][0]['rolled_back'] and result['accepted_updates'] == 0
    assert len(backend.calls) == 1 and backend.summary_calls == 0
    assert result['work']['model_restore_calls'] == result['work']['optimizer_restore_calls'] == 1


@pytest.mark.parametrize('label', ['initial', 'boundary', 'final', 'summary'])
@pytest.mark.parametrize('failure', ['late', 'raise', 'mutate'])
def test_every_checkpoint_and_summary_is_protected_and_inside_cap(label, failure):
    backend = Backend([.25, .25], callback_seconds={label: 2. if failure == 'late' else .125})
    sentinel = RuntimeError('fabricated persistence failure')
    backend.failure = label if failure == 'raise' else None
    backend.callback_error = sentinel
    backend.callback_mutation = label if failure == 'mutate' else None
    with pytest.raises((TimeoutError, RuntimeError, ValueError)) as caught:
        backend.run(1, 1, max_seconds=1.)
    if failure == 'raise':
        assert caught.value is sentinel
    result = caught.value.allocation_result
    row = result['final_summary_record'] if label == 'summary' else result['checkpoints'][-1]
    assert row['label'] == label and row['rolled_back'] and not row['accepted']
    assert row['model_sha256'] == backend.model_hash()
    assert row['optimizer_sha256'] == backend.optimizer_hash(backend.optimizers[-1])
    assert result['work']['model_restore_calls'] == result['work']['optimizer_restore_calls'] == 1
    assert len(backend.calls) == {'initial': 0, 'boundary': 1, 'final': 2, 'summary': 2}[label]
    assert backend.summary_calls == int(label == 'summary')
    assert result['status'] != 'PASS'
    if failure == 'late':
        assert result['timed_seconds'] > 1. and result['overrun_seconds'] > 0.
    json.dumps(result, allow_nan=False)


def test_summary_completion_exactly_at_deadline_is_admitted_without_extension():
    backend = Backend([1., 1.], callback_seconds={'summary': 1.})
    result = backend.run(1, 1, max_seconds=3.)
    assert result['status'] == 'PASS' and result['timed_seconds'] == 3.
    assert result['final_summary_record']['completed_elapsed'] == result['max_seconds']


def test_no_next_operation_may_start_at_deadline_even_if_last_update_timely():
    backend = Backend([1.])
    with pytest.raises(TimeoutError) as caught:
        backend.run(1, 1, max_seconds=1.)
    result = caught.value.allocation_result
    assert result['accepted_prefix_updates'] == 1 and result['accepted_joint_updates'] == 0
    assert result['trace'][0]['accepted'] and not result['trace'][0]['rolled_back']
    assert [row['label'] for row in backend.checkpoints] == ['initial']
    assert result['status'] == 'FAILED_TIMEOUT'


def test_failed_restore_never_claims_atomic_restoration_or_masks_original_error():
    backend = Backend([.25])
    sentinel = RuntimeError('update failed')
    backend.update_error = sentinel
    backend.restore_error = RuntimeError('restore failed')
    with pytest.raises(RuntimeError) as caught:
        backend.run(1, 1)
    assert caught.value is sentinel
    result = caught.value.allocation_result
    assert 'restore failed' in result['rollback_error']
    assert not result['trace'][0]['rolled_back']
    assert result['trace'][0]['model_retained_sha256'] is None
    assert result['status'] == 'FAILED_EXCEPTION'


def test_invalid_initial_clock_does_not_invoke_any_state_hook():
    backend = Backend([])
    backend.clock.bad = float('nan')
    with pytest.raises(ValueError) as caught:
        backend.run()
    assert backend.validation_calls == 0 and not backend.optimizers
    assert caught.value.allocation_result['timed_seconds'] is None
