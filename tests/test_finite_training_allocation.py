"""Fabricated clocks/states only; no scientific world, data or model calls."""
import copy
import hashlib
import json

import pytest

import openjev.research.finite_training_allocation as allocation


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Clock:
    def __init__(self):
        self.value = 0.
        self.calls = 0
        self.reverse_once = False

    def __call__(self):
        self.calls += 1
        if self.reverse_once:
            self.reverse_once = False
            return self.value - 100.
        return self.value


class Backend:
    """Small explicit momentum state, with owned snapshots and charged hooks."""
    def __init__(self, durations, *, setup=0., checkpoint_seconds=0., summary_seconds=0.):
        self.clock = Clock()
        self.durations = list(durations)
        self.model = {'dynamic': 1., 'head': 2.}
        self.optimizers, self.calls, self.checkpoints = [], [], []
        self.setup = setup
        self.checkpoint_seconds = checkpoint_seconds
        self.summary_seconds = summary_seconds
        self.invalid_result = False
        self.bad_clock = False

    def validate(self):
        self.clock.value += self.setup
        return {'fixture': 'no actual training data'}

    def new_optimizer(self, kind):
        optimizer = {'kind': kind, 'step': 0, 'moment': 0., 'learning_rate': .003}
        self.optimizers.append(optimizer)
        return optimizer

    def model_hash(self):
        return digest(self.model)

    def optimizer_hash(self, optimizer):
        return digest(optimizer)

    def snapshot_model(self):
        return allocation.Snapshot(copy.deepcopy(self.model), self.model_hash(), 2, 16)

    def snapshot_optimizer(self, optimizer):
        return allocation.Snapshot(copy.deepcopy(optimizer), self.optimizer_hash(optimizer), 2, 16)

    def restore_model(self, state):
        self.model = copy.deepcopy(state)

    def restore_optimizer(self, optimizer, state):
        optimizer.clear()
        optimizer.update(copy.deepcopy(state))

    def update(self, kind, optimizer, cursor):
        index = len(self.calls)
        assert index < len(self.durations), 'unplanned extra attempt'
        assert (cursor is None) == (kind == 'prefix')
        self.calls.append({'kind': kind, 'cursor': cursor, 'optimizer_identity': id(optimizer)})
        gradient = 2. if kind == 'prefix' else float(cursor + 1)
        optimizer['moment'] = .9 * optimizer['moment'] + .1 * gradient
        optimizer['step'] += 1
        self.model['dynamic'] -= optimizer['moment']
        if kind == 'joint':
            self.model['head'] -= optimizer['moment']
        self.clock.value += self.durations[index]
        if self.bad_clock:
            self.clock.reverse_once = True
        return {'loss': float('nan') if self.invalid_result else self.model['dynamic'] ** 2,
                'diagnostics': {'cursor': cursor, 'kind': kind}, 'work': {'gradient_passes': 1, 'event_exposures': 7}}

    def checkpoint(self, label, optimizer, cursor):
        self.checkpoints.append({'label': label, 'model': copy.deepcopy(self.model),
            'optimizer': copy.deepcopy(optimizer), 'cursor': cursor})
        self.clock.value += self.checkpoint_seconds
        return {'synthetic_record': label}

    def final_summary(self, optimizer, cursor):
        self.clock.value += self.summary_seconds
        return {'retained_cursor': cursor}

    def hooks(self):
        return allocation.Hooks(**{name: getattr(self, name) for name in allocation.Hooks.__dataclass_fields__})

    def run(self, arm='joint_continuous', **kwargs):
        return allocation.run_allocation(arm, self.hooks(), clock=self.clock, **kwargs)


def test_exact_boundaries_charge_setup_checkpoints_and_outside_summary():
    backend = Backend([8., 29.], setup=1., checkpoint_seconds=1., summary_seconds=2.)
    result = backend.run()
    assert result['status'] == 'PASS'
    assert [r['completed_elapsed'] for r in result['trace']] == [10., 40.]
    assert [r['accepted_updates'] for r in result['stages']] == [1, 1]
    assert [r['label'] for r in result['checkpoints']] == ['initial', 'boundary', 'final']
    assert result['timed_seconds'] == 41. and result['overrun_seconds'] == 1.
    assert result['final_summary_seconds'] == 2.
    assert result['work']['checkpoint_calls'] == 3
    assert result['work']['optimizer_constructions'] == 1
    assert result['work']['clock_calls'] + result['final_summary_work']['clock_calls'] == backend.clock.calls
    assert result['work']['model_restore_calls'] == result['work']['optimizer_restore_calls'] == 0
    assert result['boundary']['optimizer_before_sha256'] == result['boundary']['optimizer_after_sha256']
    assert result['checkpoints'][1]['optimizer_sha256'] == result['boundary']['optimizer_before_sha256']
    assert result['checkpoints'][-1]['model_sha256'] == result['final_model_sha256']
    json.dumps(result, allow_nan=False)


def test_late_stage_one_retries_same_batch_only_in_stage_two_with_restored_moments():
    backend = Backend([4., 7., 8., 21.])
    result = backend.run()
    assert result['status'] == 'PASS' and result['joint_cursor'] == 3
    assert [r['cursor'] for r in backend.calls] == [0, 1, 1, 2]
    assert [r['accepted'] for r in result['trace']] == [True, False, True, True]
    first, rejected, retry, _ = result['trace']
    assert rejected['model_retained_sha256'] == first['model_retained_sha256'] == retry['model_before_sha256']
    assert rejected['optimizer_retained_sha256'] == first['optimizer_retained_sha256'] == retry['optimizer_before_sha256']
    assert rejected['cursor_retained'] == retry['cursor_before'] == 1
    assert result['stages'][0]['termination'] == 'late_update_rolled_back'
    assert result['stages'][1]['start_elapsed'] == 11.
    assert backend.optimizers[0]['step'] == 3
    assert result['update_work'] == {'gradient_passes': 4, 'event_exposures': 28}
    assert result['work']['model_restore_bytes'] == result['work']['optimizer_restore_bytes'] == 16


@pytest.mark.parametrize('arm', allocation.ARMS)
def test_second_stage_uses_global_deadline_not_fresh_budget(arm):
    backend = Backend([10., 31.])
    result = backend.run(arm)
    assert result['status'] == 'FAILED_ZERO_ACCEPTED'
    assert result['trace'][1]['deadline_seconds'] == 40.
    assert result['trace'][1]['completed_elapsed'] == 41.
    assert result['trace'][1]['rolled_back'] and not result['trace'][1]['accepted']
    assert result['stages'][1]['accepted_updates'] == 0
    assert backend.checkpoints[-1]['model'] == backend.checkpoints[1]['model']
    assert result['final_model_sha256'] == result['boundary']['model_sha256']


def test_restart_changes_only_optimizer_carry_and_preserves_next_cursor():
    continuous, restart = Backend([10., 30.]), Backend([10., 30.])
    a, b = continuous.run(), restart.run('joint_restart')
    assert a['status'] == b['status'] == 'PASS'
    assert a['boundary']['joint_cursor'] == b['boundary']['joint_cursor'] == 1
    assert continuous.checkpoints[1]['model'] == restart.checkpoints[1]['model']
    assert continuous.checkpoints[1]['optimizer'] == restart.checkpoints[1]['optimizer']
    assert [r['cursor'] for r in restart.calls] == [0, 1]
    assert len(continuous.optimizers) == 1 and len(restart.optimizers) == 2
    assert a['boundary']['optimizer_reset'] is False and b['boundary']['optimizer_reset'] is True
    assert b['boundary']['optimizer_after_sha256'] == b['initial_optimizer_sha256']
    assert b['boundary']['optimizer_before_sha256'] != b['boundary']['optimizer_after_sha256']
    assert continuous.optimizers[-1]['step'] == 2 and restart.optimizers[-1]['step'] == 1
    assert continuous.model != restart.model


def test_prefix_does_not_consume_joint_batches_or_change_head():
    backend = Backend([4., 6., 30.])
    result = backend.run('prefix_then_joint')
    assert result['status'] == 'PASS' and result['accepted_prefix_updates'] == 2
    assert result['accepted_joint_updates'] == result['joint_cursor'] == 1
    assert [r['cursor'] for r in backend.calls] == [None, None, 0]
    assert backend.checkpoints[1]['model']['head'] == backend.checkpoints[0]['model']['head']
    assert result['boundary']['stage1_optimizer_kind'] == 'prefix'
    assert result['boundary']['optimizer_reset'] and result['boundary']['joint_cursor'] == 0
    assert result['stages'][1]['joint_cursor_start'] == 0


def test_late_final_attempt_restores_prior_accepted_model_optimizer_and_cursor():
    backend = Backend([10., 10., 21.])
    result = backend.run()
    assert result['status'] == 'PASS' and result['joint_cursor'] == 2
    assert result['attempted_updates'] == 3 and result['accepted_updates'] == 2
    rejected, retained = result['trace'][2], result['trace'][1]
    assert rejected['model_retained_sha256'] == retained['model_retained_sha256'] == result['final_model_sha256']
    assert rejected['optimizer_retained_sha256'] == retained['optimizer_retained_sha256'] == result['final_optimizer_sha256']
    assert backend.optimizers[-1]['step'] == 2
    assert result['overrun_seconds'] == 1.


def test_setup_exhaustion_preserves_three_boundaries_without_stage_two():
    backend = Backend([], setup=11.)
    result = backend.run()
    assert result['status'] == 'FAILED_ZERO_ACCEPTED' and len(result['stages']) == 1
    assert not result['trace'] and result['joint_cursor'] == 0
    assert [r['label'] for r in backend.checkpoints] == ['initial', 'boundary', 'final']
    assert result['initial_model_sha256'] == result['final_model_sha256']
    assert result['boundary']['optimizer_after_sha256'] is None


def test_secondary_attempt_cap_is_failure_without_retry(monkeypatch):
    monkeypatch.setattr(allocation, 'MAX_ATTEMPTS', 1)
    backend = Backend([1.])
    result = backend.run()
    assert result['status'] == 'FAILED_ATTEMPT_CAP'
    assert result['attempted_updates'] == result['accepted_updates'] == 1
    assert len(result['stages']) == 1 and result['boundary']['optimizer_after_sha256'] is None
    assert len(backend.checkpoints) == 3


@pytest.mark.parametrize('failure', ('nonfinite', 'clock', 'check'))
def test_exception_rolls_back_model_and_optimizer_and_attaches_trace(failure):
    backend = Backend([1.])
    original = copy.deepcopy(backend.model)
    backend.invalid_result = failure == 'nonfinite'
    backend.bad_clock = failure == 'clock'

    def check():
        if failure == 'check' and backend.calls:
            raise TimeoutError('outer phase deadline')

    with pytest.raises((ValueError, TimeoutError)) as caught:
        backend.run(check=check)
    assert backend.model == original
    assert backend.optimizers[-1]['step'] == 0 and backend.optimizers[-1]['moment'] == 0.
    partial = caught.value.allocation_result
    assert partial['status'] == 'FAILED_EXCEPTION' and partial['joint_cursor'] == 0
    assert len(partial['trace']) == 1 and partial['trace'][0]['rolled_back']
    assert partial['trace'][0]['model_retained_sha256'] == digest(original)
    json.dumps(partial, allow_nan=False)


@pytest.mark.parametrize('first,total', ((0., 40.), (10., 10.), (40., 10.), (float('nan'), 40.), (True, 40.)))
def test_invalid_deadlines_fail_before_any_model_hook(first, total):
    backend = Backend([])
    with pytest.raises(ValueError):
        backend.run(stage1_seconds=first, total_seconds=total)
    assert not backend.optimizers and not backend.calls and not backend.checkpoints


def test_actual_adam_late_restore_matches_three_uninterrupted_toy_updates():
    """A two-scalar toy, not the scientific model; verify real Adam moments."""
    import torch

    def encode(value):
        if isinstance(value, torch.Tensor):
            return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.detach().tolist()}
        if isinstance(value, dict):
            return {str(key): encode(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(child) for child in value]
        return value

    class TorchBackend(Backend):
        def __init__(self, durations):
            super().__init__(durations)
            self.model = {'dynamic': torch.nn.Parameter(torch.tensor(1., dtype=torch.float64)),
                          'head': torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))}

        def new_optimizer(self, kind):
            assert kind == 'joint'
            value = torch.optim.Adam(self.model.values(), lr=.003)
            self.optimizers.append(value)
            return value

        def model_hash(self):
            return digest(encode(self.model))

        def optimizer_hash(self, optimizer):
            return digest(encode(optimizer.state_dict()))

        def snapshot_optimizer(self, optimizer):
            state = copy.deepcopy(optimizer.state_dict())
            tensors = [v for row in state['state'].values() for v in row.values() if isinstance(v, torch.Tensor)]
            return allocation.Snapshot(state, self.optimizer_hash(optimizer), len(tensors),
                                       sum(v.numel() * v.element_size() for v in tensors))

        def restore_model(self, state):
            with torch.no_grad():
                for name, value in state.items():
                    self.model[name].copy_(value)

        def restore_optimizer(self, optimizer, state):
            optimizer.load_state_dict(copy.deepcopy(state))

        def update(self, kind, optimizer, cursor):
            assert kind == 'joint'
            index = len(self.calls)
            assert index < len(self.durations)
            self.calls.append(cursor)
            optimizer.zero_grad(set_to_none=True)
            loss = (self.model['dynamic'] - cursor - 2).square() + self.model['head'].square()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.values(), 5., error_if_nonfinite=True)
            optimizer.step()
            self.clock.value += self.durations[index]
            return {'loss': float(loss.detach()), 'diagnostics': {'cursor': cursor}, 'work': {'adam_updates': 1}}

    actual, reference = TorchBackend([4., 7., 8., 21.]), TorchBackend([0., 0., 0.])
    result = actual.run()
    optimizer = reference.new_optimizer('joint')
    for cursor in range(3):
        reference.update('joint', optimizer, cursor)
    assert result['status'] == 'PASS' and actual.calls == [0, 1, 1, 2]
    assert actual.model_hash() == reference.model_hash()
    assert actual.optimizer_hash(actual.optimizers[0]) == reference.optimizer_hash(optimizer)
    assert all(torch.equal(actual.model[k], reference.model[k]) for k in actual.model)
