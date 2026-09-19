"""Handwritten fake rows only: no model, native step, or random draw."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import reacher_proposal_memory_rollout as row


class Inputs:
    def __init__(self, step, transformed=False):
        self.step, self.transformed = step, transformed

    def identities(self):
        return (("initial", str(self.step) * 63 + str(int(self.transformed))),)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spy = SimpleNamespace(events=[], episodes=[], policies=[], memories=[], inputs=[],
        failure=None, fail_at=None, fail_step=1, wrong_ack=None)
    targets = np.array([[.1, -.1], [.1, -.1], [-.12, .03], [-.12, .03]])

    def fault(where, step):
        if spy.fail_at == where and spy.fail_step == step:
            raise spy.failure

    class Episode:
        def __init__(self, **kwargs):
            self.targets = np.array(kwargs['target_path'], copy=True)
            self.gains = np.array(kwargs['gear_multiplier'], copy=True)
            self.steps = kwargs['horizon']
            self.index, self.closes = 0, 0
            self.packets, self.commands, self.audit_reads = [], [], []
            spy.episodes.append(self)

        def packet(self):
            return np.array([1., 1., 0., 0., *self.targets[self.index], 1., 0.], np.float32)

        def reset(self, seed):
            assert seed == 410
            self.packets.append(self.packet())
            return self.packets[-1].copy()

        def step(self, action):
            fault('native', self.index)
            spy.events.append(('native', self.index))
            command = action.copy()
            if spy.wrong_ack == 'command':
                command[0] += np.float32(.1)
            self.commands.append(command)
            self.index += 1
            self.packets.append(self.packet())
            returned = self.packets[-1].copy()
            if spy.wrong_ack == 'packet':
                returned[0] = 0.
            return returned

        def audit_record(self):
            self.audit_reads.append(self.index)
            return {'decision_state': {'qpos': np.array([7., -8., *self.targets[self.index]]),
                                       'qvel': np.array([9., -10., 0., 0.])}}

        def episode_record(self):
            return {'policy': {'packets': np.asarray(self.packets, np.float32).reshape(-1, 8),
                               'commands': np.asarray(self.commands, np.float32).reshape(-1, 2)},
                    'metadata': {'status': 'completed' if self.index == self.steps else 'running'},
                    'audit': {'transitions': [{'reward': -float(t+1)} for t in range(self.index)]}}

        def close(self):
            self.closes += 1

    class Policy:
        def __init__(self, nominal, initial, **kwargs):
            assert nominal == 'independent nominal sentinel'
            self.arm, self.packet, self.step = kwargs['arm'], initial.copy(), 0
            self.calls, self.observations = [], []
            spy.policies.append(self)

        def decide(self, inputs, **kwargs):
            assert inputs is spy.memories[0].prepared[-1]
            assert inputs.transformed and inputs.step == self.step
            fault('decision', self.step)
            spy.events.append(('decision', self.step))
            self.calls.append(copy.deepcopy(kwargs))
            h = min(2, 3-self.step)
            action = np.array([.015 + .005*self.step, -.005], np.float32)
            sequences = np.broadcast_to(action, (1, 2, h, 2)).copy()
            sequences[0, 0] = 0.
            selected = sequences[0, 1].copy()
            angles = np.broadcast_to(np.array([1., 1., 0., 0.], np.float32), (1, 2, h, 4)).copy()
            reward = np.full((1, 2, h), -.25, np.float32)
            search = SimpleNamespace(sequences=sequences, scores=np.array([[-2., -1.]]),
                selected_ids=np.array([1]), selected_sequences=selected[None].copy())
            public_q = np.r_[0., 0., self.packet[4:6]].astype(np.float64)
            public_v = np.zeros(4, np.float64)
            root_q, root_v = kwargs.get('true_state', (public_q, public_v))
            journal = {'configuration': {'fixture': True}, 'banks': [{'arrays': {
                'predicted_angles': angles, 'geometry_reward': reward}, 'metadata': {'candidate_start': 0}}],
                'selected': {'arrays': {'qpos': np.zeros((1, 1, 2, 4)),
                    'qvel': np.zeros((1, 1, 2, 4)), 'predicted_angles': angles[:, 1:2, :1],
                    'geometry_reward': reward[:, 1:2, :1]}, 'metadata': {'purpose': 'selected_root_advance'}}}
            return action, {'step': self.step, 'arm': self.arm, 'public_packet': self.packet.copy(),
                'public_qpos': public_q, 'public_qvel': public_v,
                'root_qpos': root_q.copy(), 'root_qvel': root_v.copy(), 'action': action.copy(),
                'planning_gain': kwargs.get('current_gain', 1.), 'search': search, 'journal': journal,
                'bank_snapshot': {'last_operation': {'fixture': True}, 'costs': {'calls': self.step+1}}}

        def observe(self, command, packet):
            fault('observation', self.step)
            episode = spy.episodes[0]
            np.testing.assert_array_equal(command, episode.commands[-1])
            np.testing.assert_array_equal(packet, episode.packets[-1])
            assert not np.shares_memory(command, episode.commands[-1])
            assert not np.shares_memory(packet, episode.packets[-1])
            spy.events.append(('observe', self.step))
            self.observations.append((command.copy(), packet.copy()))
            self.packet, self.step = packet.copy(), self.step+1
            return {'step': self.step, 'identifier_updated': False}

        def snapshot(self):
            return {'step': self.step, 'decisions': len(self.calls), 'observations': len(self.observations)}

    class Memory:
        def __init__(self, **kwargs):
            self.settings, self.prepared, self.commits, self.targets = kwargs, [], [], []
            spy.memories.append(self)

        def prepare(self, inputs, target, *, step):
            assert len(self.commits) == step
            assert isinstance(inputs, Inputs) and not inputs.transformed
            assert target.dtype == np.float32 and target.shape == (2,)
            assert not np.shares_memory(target, spy.episodes[0].packets[-1])
            fault('proposal_memory_prepare', step)
            spy.events.append(('prepare', step))
            changed = bool(self.targets and not np.array_equal(self.targets[-1], target))
            self.targets.append(target.copy())
            transformed = Inputs(step, transformed=True)
            self.prepared.append(transformed)
            return transformed, {'mode': self.settings['mode'], 'step': step, 'target_changed': changed,
                                 'original_inputs': dict(inputs.identities())}

        def commit(self, selected, issued):
            step = len(self.commits)
            # A returned native packet is not sufficient: public observation must succeed first.
            assert spy.policies[0].step == step+1
            assert len(spy.episodes[0].commands) == step+1
            np.testing.assert_array_equal(issued, spy.episodes[0].commands[-1])
            np.testing.assert_array_equal(selected[0], issued)
            assert not np.shares_memory(issued, spy.episodes[0].commands[-1])
            fault('proposal_memory_commit', step)
            spy.events.append(('commit', step))
            self.commits.append((selected.copy(), issued.copy()))

        def snapshot(self):
            return {'prepared': len(self.prepared), 'committed': len(self.commits),
                    'settings': self.settings, 'targets': self.targets}

    def load(stem):
        inputs = Inputs(int(Path(stem).name))
        spy.inputs.append(inputs)
        return inputs

    monkeypatch.setattr(row, 'TrackingDynamicsEpisode', Episode)
    monkeypatch.setattr(row, 'TrackingPolicy', Policy)
    monkeypatch.setattr(row, 'CEMProposalMemory', Memory)
    monkeypatch.setattr(row, 'nominal_model', lambda: 'independent nominal sentinel')
    monkeypatch.setattr(row.artifacts, 'load_inputs', load)
    stems = [tmp_path / 'inputs' / f'{step:03d}' for step in range(3)]
    stems[0].parent.mkdir()
    for stem in stems:
        stem.with_suffix('.npz').write_bytes(b'fake bound innovations')
        stem.with_suffix('.json').write_text('{}\n')
    arguments = {'arm': 'nominal', 'proposal_mode': 'cold', 'target_path': targets,
        'gain_schedule': np.array([.7, 1.3, 1.3]), 'noise': np.zeros((3, 2)), 'reset_seed': 410,
        'inputs_by_step': stems, 'gain_grid': np.array([.7, 1., 1.3]), 'window': 2,
        'freeze_after': 1, 'noise_std': 0., 'planning_horizon': 2, 'action_block': 1,
        'out': tmp_path / 'row', 'deadline': float('inf'), 'engineering': True}
    return spy, arguments


@pytest.mark.parametrize('arm', row.ARMS)
@pytest.mark.parametrize('mode', ['cold', 'repeat_last', 'shift_plan'])
def test_all_nine_cells_transform_before_policy_and_commit_after_observation(harness, arm, mode):
    spy, arguments = harness
    receipt = row.run_row(**{**arguments, 'arm': arm, 'proposal_mode': mode})
    assert spy.events == [(phase, t) for t in range(3)
                          for phase in ('prepare', 'decision', 'native', 'observe', 'commit')]
    policy, episode, memory = spy.policies[0], spy.episodes[0], spy.memories[0]
    assert memory.settings['mode'] == mode
    assert len(memory.commits) == 3 and episode.closes == 1
    for t, kwargs in enumerate(policy.calls):
        expected = {'deadline'}
        if arm != 'nominal':
            expected.add('current_gain')
            assert kwargs['current_gain'] == [.7, 1.3, 1.3][t]
        if arm == 'true_state':
            expected.add('true_state')
            np.testing.assert_array_equal(kwargs['true_state'][1], [9., -10., 0., 0.])
        assert set(kwargs) == expected
        meta = json.loads((arguments['out'] / f'decisions/{t:03d}.json').read_text())
        assert meta['input_identities'] == dict(memory.prepared[t].identities())
        assert meta['proposal_memory']['original_inputs'] == dict(spy.inputs[t].identities())
        # Gain change at t1 is not a target event. New goal is available only at t2.
        assert meta['proposal_memory']['target_changed'] == (t == 2)
        with np.load(arguments['out'] / f'decisions/{t:03d}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['action'], episode.commands[t])
            np.testing.assert_array_equal(saved['sequences'][1], memory.commits[t][0])
            assert saved['selected_id'] == 1
    assert episode.audit_reads == ([0, 1, 2] if arm == 'true_state' else [])
    assert receipt['native_cost'] == 6. and receipt['proposal_mode'] == mode
    assert receipt['proposal_memory_seconds'] >= 0
    expected = {'started.json', 'initial-controller.json', 'initial-proposal-memory.json',
                'episode.json', 'controller.json', 'proposal-memory.json'}
    expected |= {f'decisions/{t:03d}.{suffix}' for t in range(3) for suffix in ('npz', 'json')}
    expected |= {f'observations/{t:03d}.json' for t in range(1, 4)}
    assert set(receipt['files']) == expected
    assert all(row.sha(arguments['out'] / name) == digest for name, digest in receipt['files'].items())
    assert receipt['wall_seconds'] >= sum(receipt[key] for key in
        ('decision_seconds', 'native_seconds', 'observation_seconds', 'proposal_memory_seconds'))


@pytest.mark.parametrize('bad_ack', ['command', 'packet'])
def test_native_mismatch_never_updates_observer_or_memory(harness, bad_ack):
    spy, arguments = harness
    spy.wrong_ack = bad_ack
    with pytest.raises(RuntimeError, match='differs'):
        row.run_row(**arguments)
    assert not spy.policies[0].observations and not spy.memories[0].commits
    assert len(spy.episodes[0].commands) == 1
    assert json.loads((arguments['out'] / 'failed.json').read_text())['phase'] == 'native'


@pytest.mark.parametrize('where', ['proposal_memory_prepare', 'decision', 'native',
                                  'observation', 'proposal_memory_commit'])
def test_failed_second_step_retains_ragged_prefix_and_original_exception(harness, where):
    spy, arguments = harness
    original = KeyboardInterrupt('synthetic interrupted ' + where)
    spy.failure, spy.fail_at = original, where
    with pytest.raises(KeyboardInterrupt) as caught:
        row.run_row(**arguments)
    assert caught.value is original
    out = arguments['out']
    assert (out / 'observations/001.json').is_file()
    assert not (out / 'observations/002.json').exists()
    assert (out / 'decisions/001.json').exists() == (where in ('native', 'observation', 'proposal_memory_commit'))
    assert not (out / 'completed.json').exists()
    failure = json.loads((out / 'failed.json').read_text())
    assert failure['phase'] == where and failure['step'] == 1 and not failure['automatic_retry']
    partial = json.loads((out / 'partial-proposal-memory.json').read_text())
    assert partial['committed'] == 1
    assert partial['prepared'] == (1 if where == 'proposal_memory_prepare' else 2)
    assert len(spy.policies[0].observations) == (2 if where == 'proposal_memory_commit' else 1)
    assert spy.episodes[0].closes == 1


def test_memory_failure_survives_failed_snapshot_and_failure_receipt(harness, monkeypatch):
    spy, arguments = harness
    original = KeyboardInterrupt('primary commit failure')
    spy.failure, spy.fail_at = original, 'proposal_memory_commit'
    real_write = row.write

    def broken(path, value):
        if Path(path).name in ('partial-proposal-memory.json', 'failed.json'):
            raise OSError('secondary disk failure')
        return real_write(path, value)

    monkeypatch.setattr(row, 'write', broken)
    with pytest.raises(KeyboardInterrupt) as caught:
        row.run_row(**arguments)
    assert caught.value is original
    assert any('Failure receipt' in note for note in original.__notes__)
    assert any('capture_proposal_memory' in note for note in original.__notes__)
    assert (arguments['out'] / 'partial-episode.json').exists()
    assert spy.episodes[0].closes == 1


def test_late_receipt_cap_demotes_completed_row(harness, monkeypatch):
    spy, arguments = harness
    clock = [0.]
    monkeypatch.setattr(row.time, 'monotonic', lambda: clock[0])
    real_write = row.write

    def delayed(path, value):
        result = real_write(path, value)
        if Path(path).name == 'completed.json':
            clock[0] = 11.
        return result

    monkeypatch.setattr(row, 'write', delayed)
    with pytest.raises(TimeoutError):
        row.run_row(**{**arguments, 'deadline': 10.})
    assert not (arguments['out'] / 'completed.json').exists()
    assert (arguments['out'] / 'partial-completed.json').exists()
    assert (arguments['out'] / 'failed.json').exists()
    assert len(spy.memories[0].commits) == 3 and spy.episodes[0].closes == 1


def test_existing_row_cannot_be_overwritten(harness):
    spy, arguments = harness
    arguments['out'].mkdir()
    sentinel = arguments['out'] / 'sentinel'
    sentinel.write_bytes(b'original attempt')
    with pytest.raises(FileExistsError):
        row.run_row(**arguments)
    assert list(arguments['out'].iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'original attempt'
    assert not spy.episodes and not spy.memories


@pytest.mark.parametrize('role', ['adaptive', 'frozen', 'zero'])
def test_disallowed_roles_fail_before_any_attempt(harness, role):
    spy, arguments = harness
    with pytest.raises(ValueError, match='role'):
        row.run_row(**{**arguments, 'arm': role})
    assert not arguments['out'].exists() and not spy.episodes
