"""Fake orchestration and compact byte artifacts only; no native/model calls."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import reacher_tracking_rollout as rollout


class FakeInputs:
    def identities(self):
        return (('initial', 'a' * 64), ('cem/1', 'b' * 64))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spy = SimpleNamespace(episodes=[], policies=[], loaded=[], model_calls=0,
        fail_native_at=None, fail_decision_at=None, fail_observation_at=None, failure=None,
        mismatch_command=False, mismatch_packet=False, mutate_schedule=False)
    targets = np.array([[.1, -.1], [.1, -.1], [-.12, .03]], np.float64)
    gains = np.array([.7, 1.3], np.float64)
    noise = np.zeros((2, 2), np.float64)
    model = object()

    class Episode:
        def __init__(self, *, horizon, target_path, gear_multiplier, sensor_schedule, noise):
            self.horizon = horizon
            self.targets = np.array(target_path, copy=True)
            self.gains = np.array(gear_multiplier, copy=True)
            self.noise = np.array(noise, copy=True)
            self.schedule = np.array(sensor_schedule, copy=True)
            self.index, self.status, self.closes = 0, 'created', 0
            self.packets, self.commands, self.transitions = [], [], []
            self.audit_reads = []
            spy.episodes.append(self)

        def packet(self):
            return np.array([1., 1., 0., 0., *self.targets[self.index], 1., 0.], np.float32)

        def reset(self, seed):
            assert seed == 410
            self.status = 'running'
            first = self.packet()
            self.packets.append(first.copy())
            return first

        def step(self, action):
            if self.index == spy.fail_native_at:
                self.status = 'failed'
                raise spy.failure
            command = action.copy()
            if spy.mismatch_command:
                command[0] += np.float32(.1)
            self.commands.append(command)
            self.transitions.append({'reward': -float(self.index + 1),
                                     'gear_multiplier': float(self.gains[self.index])})
            self.index += 1
            if self.index == self.horizon:
                self.status = 'completed'
            observed = self.packet()
            self.packets.append(observed.copy())
            if spy.mismatch_packet:
                observed[0] = 0.
            return observed

        def audit_record(self):
            self.audit_reads.append(self.index)
            return {'decision_state': {'qpos': np.array([7., -8., *self.targets[self.index]]),
                                       'qvel': np.array([9., -10., 0., 0.])}}

        def metadata(self):
            return {'status': self.status, 'gear_multiplier': self.gains.tolist(),
                    'target_path': self.targets.tolist(), 'horizon': self.horizon}

        def episode_record(self):
            return {'policy': {'packets': np.array(self.packets, np.float32).reshape(-1, 8),
                               'commands': np.array(self.commands, np.float32).reshape(-1, 2)},
                    'metadata': self.metadata(), 'audit': {'transitions': copy.deepcopy(self.transitions)}}

        def close(self):
            self.closes += 1

    class Policy:
        def __init__(self, nominal, initial, **settings):
            assert nominal is model
            assert initial.shape == (8,) and initial.dtype == np.float32
            self.arm, self.settings = settings['arm'], settings
            self.packet = initial.copy()
            self.calls, self.observations = [], []
            self.step = 0
            self.failed = False
            spy.policies.append(self)
            if spy.mutate_schedule:
                gains[:] = 9.  # The caller's buffer, after native schedule capture.

        def decide(self, inputs, **kwargs):
            assert isinstance(inputs, FakeInputs)
            if self.step == spy.fail_decision_at:
                self.failed = True
                raise spy.failure
            self.calls.append(copy.deepcopy(kwargs))
            action = np.array([0., 0.] if self.arm == 'zero' else [.015, -.005], np.float32)
            public_q = np.r_[0., 0., self.packet[4:6]].astype(np.float64)
            public_v = np.zeros(4, np.float64)
            root_q, root_v = kwargs.get('true_state', (public_q, public_v))
            trace = {'step': self.step, 'arm': self.arm, 'public_packet': self.packet.copy(),
                     'public_qpos': public_q, 'public_qvel': public_v,
                     'root_qpos': root_q.copy(), 'root_qvel': root_v.copy(),
                     'planning_gain': kwargs.get('current_gain', 1.), 'action': action.copy(),
                     'search': None, 'journal': None, 'bank_snapshot': None}
            return action, trace

        def observe(self, issued, packet):
            if self.step == spy.fail_observation_at:
                self.failed = True
                raise spy.failure
            native = spy.episodes[-1]
            np.testing.assert_array_equal(issued, native.commands[-1])
            np.testing.assert_array_equal(packet, native.packets[-1])
            assert not np.shares_memory(issued, native.commands[-1])
            assert not np.shares_memory(packet, native.packets[-1])
            self.observations.append((issued.copy(), packet.copy()))
            self.packet = packet.copy()
            self.step += 1
            return {'step': self.step, 'identifier_updated': self.arm in ('adaptive', 'frozen')}

        def snapshot(self):
            return {'arm': self.arm, 'step': self.step, 'failed': self.failed,
                    'decisions': len(self.calls), 'observations': len(self.observations)}

    def factory():
        spy.model_calls += 1
        return model

    def load(stem):
        spy.loaded.append(Path(stem))
        return FakeInputs()

    monkeypatch.setattr(rollout, 'TrackingDynamicsEpisode', Episode)
    monkeypatch.setattr(rollout, 'TrackingPolicy', Policy)
    monkeypatch.setattr(rollout, 'nominal_model', factory)
    monkeypatch.setattr(rollout.artifacts, 'load_inputs', load)
    stems = [tmp_path / 'inputs' / f'{step:03d}' for step in range(2)]
    stems[0].parent.mkdir()
    for step, stem in enumerate(stems):
        stem.with_suffix('.npz').write_bytes(f'fake prebound input {step}'.encode())
        stem.with_suffix('.json').write_text('{}\n')
    arguments = {'arm': 'nominal', 'target_path': targets, 'gain_schedule': gains,
                 'noise': noise, 'reset_seed': 410, 'inputs_by_step': stems,
                 'gain_grid': np.array([.7, 1., 1.3]), 'window': 2, 'freeze_after': 1,
                 'noise_std': 0., 'planning_horizon': 2, 'action_block': 1,
                 'out': tmp_path / 'row', 'deadline': float('inf'), 'engineering': True}
    return spy, arguments


@pytest.mark.parametrize('arm', rollout.ARMS)
def test_all_six_roles_receive_only_their_allowed_current_inputs(harness, arm):
    spy, arguments = harness
    receipt = rollout.run_row(**{**arguments, 'arm': arm})
    controller, episode = spy.policies[0], spy.episodes[0]
    assert episode.closes == 1 and episode.index == 2
    assert spy.model_calls == 1 and len(spy.loaded) == 2
    for step, call in enumerate(controller.calls):
        expected = {'deadline'}
        if arm in ('public_gain', 'true_state'):
            expected.add('current_gain')
            assert call['current_gain'] == [.7, 1.3][step]
        if arm == 'true_state':
            expected.add('true_state')
            np.testing.assert_array_equal(call['true_state'][0][:2], [7., -8.])
            np.testing.assert_array_equal(call['true_state'][1], [9., -10., 0., 0.])
        assert set(call) == expected
    assert episode.audit_reads == ([0, 1] if arm == 'true_state' else [])
    assert len(controller.observations) == 2
    assert receipt['status'] == 'completed' and receipt['native_cost'] == 3.
    assert receipt['native_decisions'] == receipt['policy_decisions'] == receipt['observation_updates'] == 2
    out = arguments['out']
    expected_names = {'started.json', 'initial-controller.json', 'episode.json', 'controller.json',
                      'decisions/000.npz', 'decisions/000.json', 'decisions/001.npz', 'decisions/001.json',
                      'observations/001.json', 'observations/002.json'}
    assert set(receipt['files']) == expected_names
    for name, digest in receipt['files'].items():
        assert rollout.sha(out / name) == digest
    assert json.loads((out / 'completed.json').read_text()) == receipt
    saved = json.loads((out / 'episode.json').read_text())
    assert len(saved['policy']['packets']) == 3 and len(saved['policy']['commands']) == 2
    assert receipt['wall_seconds'] >= receipt['decision_seconds'] + receipt['native_seconds'] + receipt['observation_seconds']


def test_oracle_gain_is_bound_to_copied_native_schedule_not_mutated_caller_buffer(harness):
    spy, arguments = harness
    spy.mutate_schedule = True
    rollout.run_row(**{**arguments, 'arm': 'public_gain'})
    assert arguments['gain_schedule'].tolist() == [9., 9.]
    assert spy.episodes[0].gains.tolist() == [.7, 1.3]
    assert [call['current_gain'] for call in spy.policies[0].calls] == [.7, 1.3]


@pytest.mark.parametrize('kind', ['mismatch_command', 'mismatch_packet'])
def test_wrong_native_acknowledgement_fails_before_policy_observation(harness, kind):
    spy, arguments = harness
    setattr(spy, kind, True)
    with pytest.raises(RuntimeError, match='differs'):
        rollout.run_row(**arguments)
    assert not spy.policies[0].observations
    assert spy.episodes[0].closes == 1
    failed = json.loads((arguments['out'] / 'failed.json').read_text())
    assert failed['phase'] == 'native' and failed['step'] == 0
    assert not (arguments['out'] / 'completed.json').exists()
    partial = json.loads((arguments['out'] / 'partial-episode.json').read_text())
    assert len(partial['policy']['commands']) == 1


@pytest.mark.parametrize('where', ['decision', 'native', 'observation'])
def test_failed_second_step_retains_complete_ragged_prefix_and_original_error(harness, where):
    spy, arguments = harness
    original = KeyboardInterrupt('synthetic second step failure')
    spy.failure = original
    setattr(spy, f'fail_{where}_at', 1)
    with pytest.raises(KeyboardInterrupt) as caught:
        rollout.run_row(**arguments)
    assert caught.value is original
    out = arguments['out']
    assert (out / 'decisions/000.npz').is_file() and (out / 'observations/001.json').is_file()
    assert (out / 'decisions/001.npz').exists() == (where != 'decision')
    assert not (out / 'observations/002.json').exists()
    assert not (out / 'completed.json').exists()
    failed = json.loads((out / 'failed.json').read_text())
    assert failed['phase'] == where and failed['step'] == 1 and not failed['automatic_retry']
    partial = json.loads((out / 'partial-episode.json').read_text())
    native_count = 2 if where == 'observation' else 1
    assert len(partial['policy']['packets']) == native_count + 1
    assert len(partial['policy']['commands']) == native_count
    assert spy.episodes[0].closes == 1


def test_existing_output_is_never_overwritten_or_marked_failed(harness):
    spy, arguments = harness
    out = arguments['out']
    out.mkdir()
    (out / 'sentinel').write_bytes(b'prior attempt must remain exact')
    with pytest.raises(FileExistsError):
        rollout.run_row(**arguments)
    assert list(out.iterdir()) == [out / 'sentinel']
    assert (out / 'sentinel').read_bytes() == b'prior attempt must remain exact'
    assert not spy.episodes and not spy.policies


def test_expired_entry_cap_allocates_no_attempt_or_backend(harness):
    spy, arguments = harness
    with pytest.raises(TimeoutError):
        rollout.run_row(**{**arguments, 'deadline': -1.})
    assert not arguments['out'].exists()
    assert not spy.episodes and not spy.policies


def test_input_change_after_binding_is_rejected_before_decision(harness, monkeypatch):
    spy, arguments = harness
    original = rollout.TrackingPolicy

    def mutate(*args, **kwargs):
        policy = original(*args, **kwargs)
        arguments['inputs_by_step'][0].with_suffix('.npz').write_bytes(b'changed input')
        return policy

    monkeypatch.setattr(rollout, 'TrackingPolicy', mutate)
    with pytest.raises(ValueError, match='bytes changed'):
        rollout.run_row(**arguments)
    assert not spy.policies[0].calls and not spy.loaded
    assert spy.episodes[0].closes == 1
    assert json.loads((arguments['out'] / 'failed.json').read_text())['phase'] == 'input'


def test_started_serialization_failure_retains_attempt_failure(harness, monkeypatch):
    spy, arguments = harness
    original_write = rollout.write
    original = OSError('synthetic started write failure')

    def fail(path, value):
        if Path(path).name == 'started.json':
            raise original
        return original_write(path, value)

    monkeypatch.setattr(rollout, 'write', fail)
    with pytest.raises(OSError) as caught:
        rollout.run_row(**arguments)
    assert caught.value is original
    assert (arguments['out'] / 'failed.json').is_file()
    assert not spy.episodes and not spy.policies


def test_late_completed_serialization_demotes_success_and_keeps_failure(harness, monkeypatch):
    spy, arguments = harness
    original_write = rollout.write
    clock = [0.]
    monkeypatch.setattr(rollout.time, 'monotonic', lambda: clock[0])

    def late(path, value):
        result = original_write(path, value)
        if Path(path).name == 'completed.json':
            clock[0] = 11.
        return result

    monkeypatch.setattr(rollout, 'write', late)
    with pytest.raises(TimeoutError):
        rollout.run_row(**{**arguments, 'deadline': 10.})
    assert not (arguments['out'] / 'completed.json').exists()
    failed = json.loads((arguments['out'] / 'failed.json').read_text())
    assert failed['status'] == 'failed'
    assert spy.episodes[0].closes == 1
    assert len(spy.policies[0].observations) == 2


def test_failed_receipt_write_cannot_replace_original_native_baseexception(harness, monkeypatch):
    spy, arguments = harness
    original = KeyboardInterrupt('primary native failure')
    spy.failure, spy.fail_native_at = original, 0
    original_write = rollout.write

    def fail(path, value):
        if Path(path).name == 'failed.json':
            raise OSError('secondary failed receipt failure')
        return original_write(path, value)

    monkeypatch.setattr(rollout, 'write', fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        rollout.run_row(**arguments)
    assert caught.value is original
    assert (arguments['out'] / 'partial-episode.json').exists()
    assert (arguments['out'] / 'partial-controller.json').exists()
    assert spy.episodes[0].closes == 1


def test_compact_decision_keeps_all_scores_angles_and_selected_native_advance(tmp_path):
    (tmp_path / 'decisions').mkdir()
    h = 2
    sequences = np.arange(256*h*2, dtype=np.float32).reshape(1, 256, h, 2) / 10000
    scores = np.arange(256, dtype=np.float64)[None]
    angles = np.arange(256*h*4, dtype=np.float32).reshape(256, h, 4) / 1000
    rewards = -np.arange(256*h, dtype=np.float32).reshape(256, h) / 100
    journal = {'banks': [{'arrays': {'predicted_angles': angles[i:i+64][None],
                                     'geometry_reward': rewards[i:i+64][None]},
                           'metadata': {'candidate_start': i}} for i in range(0, 256, 64)],
               'configuration': {'version': 'explicit fake schema'},
               'selected': {'arrays': {'qpos': np.arange(8, dtype=np.float64).reshape(1, 1, 2, 4),
                                       'qvel': np.arange(8, 16, dtype=np.float64).reshape(1, 1, 2, 4),
                                       'predicted_angles': angles[255:256, 0:1][None],
                                       'geometry_reward': rewards[255:256, 0:1][None]},
                            'metadata': {'purpose': 'selected_root_advance'}}}
    trace = {'step': 0, 'arm': 'adaptive', 'root_qpos': np.zeros(4), 'root_qvel': np.zeros(4),
             'public_packet': np.zeros(8, np.float32), 'public_qpos': np.zeros(4),
             'public_qvel': np.zeros(4), 'action': sequences[0, 255, 0], 'planning_gain': .7,
             'search': SimpleNamespace(sequences=sequences, scores=scores, selected_ids=np.array([255])),
             'journal': journal, 'bank_snapshot': {'last_operation': {'gain': .7}, 'costs': {'calls': 1}}}
    rollout.save_decision(tmp_path, trace, FakeInputs())
    with np.load(tmp_path / 'decisions/000.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'root_qpos', 'root_qvel', 'public_packet', 'public_qpos', 'public_qvel',
            'action', 'planning_gain', 'sequences', 'scores', 'selected_id', 'predicted_angles',
            'raw_rewards', 'selected_qpos', 'selected_qvel', 'selected_angles', 'selected_reward'}
        np.testing.assert_array_equal(saved['predicted_angles'], angles)
        np.testing.assert_array_equal(saved['raw_rewards'], rewards)
        np.testing.assert_array_equal(saved['sequences'], sequences[0])
        np.testing.assert_array_equal(saved['selected_qpos'], journal['selected']['arrays']['qpos'][0, 0])
        assert saved['selected_id'] == 255 and saved['selected_reward'] == rewards[255, 0]
    metadata = json.loads((tmp_path / 'decisions/000.json').read_text())
    assert metadata['input_identities'] == dict(FakeInputs().identities())
    assert metadata['journalcallbackmetadata'] == [{'candidate_start': i} for i in range(0, 256, 64)]
    original = (tmp_path / 'decisions/000.npz').read_bytes()
    with pytest.raises(FileExistsError):
        rollout.save_decision(tmp_path, trace, FakeInputs())
    assert (tmp_path / 'decisions/000.npz').read_bytes() == original
