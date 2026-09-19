"""Fake environments/models and byte-tree orchestration only; no native games."""

import hashlib
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openjev.research.card_memory_task import PublicTeacher, Reveal

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_card_memory.py'
SPEC = importlib.util.spec_from_file_location('card_evaluator_test', PATH)
study = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    SPEC.loader.exec_module(study)


@pytest.fixture(autouse=True)
def synthetic_runtime(monkeypatch):
    monkeypatch.setattr(study, 'runtime', lambda: {'synthetic': True})


def hidden():
    return np.full(52, 13, dtype=np.int64)


class HandwrittenGame:
    """A deterministic table fixture, not POPGym or a random deck generator."""

    def __init__(self, *, bad_info=False, close_error=False):
        self.bad_info, self.close_error = bad_info, close_error
        self.actions, self.closed = [], False

    @property
    def state(self):
        raise AssertionError('Hidden state must never be accessed')

    def get_state(self):
        raise AssertionError('Hidden state must never be accessed')

    def reset(self, *, seed):
        self.seed = seed
        self.matched, self.pending = {}, None
        return hidden(), {}

    def step(self, action):
        self.actions.append(action)
        observation = hidden()
        for pos, rank in self.matched.items():
            observation[pos] = rank
        if self.pending is not None:
            observation[self.pending] = self.pending // 4
        observation[action] = action // 4
        if action in self.matched:
            reward = -(1 if self.pending is None else 2) / 104
            self.pending = None
        elif self.pending is None:
            reward, self.pending = 0, action
        else:
            if action != self.pending and action // 4 == self.pending // 4:
                reward = 2 / 52
                self.matched[action] = action // 4
                self.matched[self.pending] = self.pending // 4
            else:
                reward = -2 / 104
            self.pending = None
        return observation, reward, len(self.matched) == 52, len(self.actions) == 104, ({'bad': True} if self.bad_info else {})

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError('cleanup failure')


class FakeMemory:
    def __init__(self, fail_write=False):
        self.calls, self.fail_write = [], fail_write

    def init_state(self, batch, device):
        assert (batch, device) == (1, 'cpu')
        return {'events': torch.zeros(1)}

    def predict(self, state, queries):
        self.calls.append(('predict', int(state['events'].item())))
        assert queries.tolist() == [list(range(52))]
        return torch.zeros(1, 52, 13)

    def write(self, state, pos, rank, valid):
        self.calls.append(('write', int(pos.item()), int(rank.item())))
        assert valid.tolist() == [True]
        if self.fail_write:
            raise RuntimeError('write failure')
        return {'events': state['events'] + 1}, {'diagnostic': torch.zeros(1)}


@pytest.mark.parametrize('controller', ['exact', 'last32', 'gru-pair0'])
def test_native_boundary_uses_only_public_frames_and_one_selected_write(controller):
    env = HandwrittenGame()
    model = FakeMemory() if controller == 'gru-pair0' else None
    arrays, receipt = study.evaluate_episode(env, controller=controller, seed=410, model=model)
    assert receipt['status'] == 'complete' and receipt['success']
    assert receipt['native_steps'] == 52 and receipt['matched_pairs'] == 26
    assert receipt['return'] == 1
    assert arrays['observations'].shape == (53, 52)
    assert arrays['raw_probabilities'].shape == arrays['picker_probabilities'].shape == (52, 52, 13)
    assert arrays['actions'].tolist() == env.actions == list(range(52))
    assert arrays['terminated'].sum() == 1 and not arrays['truncated'].any()
    assert receipt['counts']['native_returned'] == 52
    if model:
        assert len(model.calls) == 104
        for step in range(52):
            assert model.calls[2 * step] == ('predict', step)
            assert model.calls[2 * step + 1] == ('write', step, step // 4)
        assert receipt['counts']['predict_returned'] == receipt['counts']['write_returned'] == 52
    else:
        assert receipt['counts']['predict_returned'] == receipt['counts']['write_returned'] == 0
    assert not env.closed  # The wrapper owns cleanup.


def test_last32_only_counts_selected_reveals_and_newest_replaces_same_position():
    events = deque(maxlen=32)
    events.append(Reveal(40, 7))
    for i in range(32):
        events.append(Reveal(i % 2, i % 13))
    probability = study._reference_probs('last32', None, events)
    np.testing.assert_array_equal(probability[40], np.full(13, 1 / 13))
    assert probability[0, 30 % 13] == probability[1, 31 % 13] == 1
    teacher = PublicTeacher()
    teacher.reset(hidden())
    observation = hidden()
    observation[40] = 7
    teacher.observe(observation)
    teacher.observe(hidden())
    assert study._reference_probs('exact', teacher, events)[40, 7] == 1


def test_failed_model_write_keeps_completed_native_transition_and_single_write_attempt():
    model = FakeMemory(fail_write=True)
    with pytest.raises(RuntimeError, match='write failure') as caught:
        study.evaluate_episode(HandwrittenGame(), controller='gru-pair0', seed=410, model=model)
    arrays, receipt = caught.value.partial_card_episode, caught.value.partial_card_receipt
    assert arrays['actions'].tolist() == [0] and arrays['observations'].shape == (2, 52)
    assert receipt['counts']['native_returned'] == receipt['counts']['write_attempted'] == 1
    assert receipt['counts']['write_returned'] == 0 and receipt['status'] == 'failed'


def test_bad_native_info_keeps_raw_return_before_public_validation():
    with pytest.raises(ValueError, match='flags/info') as caught:
        study.evaluate_episode(HandwrittenGame(bad_info=True), controller='exact', seed=410)
    assert caught.value.partial_card_episode['actions'].tolist() == [0]
    assert caught.value.partial_card_receipt['counts']['native_returned'] == 1


def test_expired_deadline_prevents_reset():
    env = HandwrittenGame()
    with pytest.raises(TimeoutError):
        study.evaluate_episode(env, controller='exact', seed=410, deadline=0)
    assert not hasattr(env, 'seed')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixtures(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    checkpoint_map = {}
    for mode in study.MODES:
        for pair in range(3):
            name = f'{mode}-pair{pair}'
            directory = root / 'fits' / name
            directory.mkdir(parents=True)
            checkpoint = directory / 'final-checkpoint.pt'
            checkpoint.write_bytes(b'synthetic opaque tensor payload')
            completed = directory / 'completed.json'
            write_json(completed, {'status': 'complete', 'evaluation_calls': 0,
                       'counts': {'successful_updates': 1, 'completed_epochs': 1},
                       'recipe': {'expected_updates': 1, 'epochs': 1},
                       'files': {checkpoint.name: {'sha256': study._sha(checkpoint), 'bytes': checkpoint.stat().st_size}}})
            checkpoint_map[name] = {'mode': mode, 'pair': pair,
                                   'checkpoint_path': str(checkpoint.relative_to(root)), 'checkpoint_sha256': study._sha(checkpoint),
                                   'completed_path': str(completed.relative_to(root)), 'completed_sha256': study._sha(completed)}
    protocol, inputs, bindings = root / 'protocol.json', root / 'inputs.json', root / 'bindings.json'
    write_json(protocol, {'bindings_file': 'bindings.json', 'evaluation': study.EVALUATION})
    write_json(inputs, {'evaluation': [{'seed': 410 + index} for index in range(64)]})
    write_json(bindings, {'runtime': {'synthetic': True},
                         'files': {'protocol.json': study._sha(protocol), 'inputs.json': study._sha(inputs)}})
    return {'protocol_path': protocol, 'expected_protocol_sha256': study._sha(protocol),
            'inputs_path': inputs, 'expected_inputs_sha256': study._sha(inputs),
            'checkpoint_map': checkpoint_map, 'out': tmp_path / 'evaluation',
            'expected_bindings_sha256': study._sha(bindings), 'root': root}


def fake_episode(env, *, controller, seed, model, deadline):
    arrays = {'observations': np.full((2, 52), 13, dtype=np.int64), 'actions': np.array([0]),
              'rewards': np.array([0.]), 'terminated': np.array([False]), 'truncated': np.array([True])}
    receipt = {'status': 'complete', 'controller': controller, 'seed': seed, 'wall_seconds': 0.,
               'native_steps': 1, 'return': 0., 'success': False}
    return arrays, receipt


def test_all18_authenticate_before_any_loader_and_full20x64_coverage(tmp_path, monkeypatch):
    arguments = fixtures(tmp_path)
    loads, episodes, envs = [], [], []

    def loader(entry, receipt, root):
        assert (arguments['out'] / 'all-fits-ready.json').exists()
        assert len(study._read(arguments['out'] / 'all-fits-ready.json')['fits']) == 18
        loads.append((entry['mode'], entry['pair']))
        return object()

    def episode(env, **kwargs):
        episodes.append((kwargs['controller'], kwargs['seed']))
        return fake_episode(env, **kwargs)

    def factory():
        env = HandwrittenGame()
        envs.append(env)
        return env

    monkeypatch.setattr(study, 'evaluate_episode', episode)
    result = study.evaluate(**arguments, env_factory=factory, model_loader=loader)
    assert result['status'] == 'complete' and result['episodes'] == 1280
    assert len(loads) == 18 and len(episodes) == 1280 and all(e.closed for e in envs)
    assert {name for name, _ in episodes} == set(arguments['checkpoint_map']) | {'exact', 'last32'}
    for name in result['controller_wall_seconds']:
        assert [seed for row, seed in episodes if row == name] == list(range(410, 474))
    assert len(result['files']) == 2582  # Two outer payloads +20*(128 episode files+one receipt).
    for relative, binding in result['files'].items():
        assert study._sha(arguments['out'] / relative) == binding['sha256']


@pytest.mark.parametrize('corruption', ['missing_fit', 'checkpoint', 'failed_marker', 'not_complete', 'wrong_pair'])
def test_any_incomplete_fit_stops_all_native_or_model_evaluation(tmp_path, corruption):
    arguments = fixtures(tmp_path)
    entry = arguments['checkpoint_map']['gru-pair2']
    if corruption == 'missing_fit':
        arguments['checkpoint_map'].pop('gru-pair2')
    elif corruption == 'checkpoint':
        (arguments['root'] / entry['checkpoint_path']).write_bytes(b'changed')
    elif corruption == 'failed_marker':
        (arguments['root'] / entry['completed_path']).with_name('failed.json').write_text('{}')
    elif corruption == 'not_complete':
        path = arguments['root'] / entry['completed_path']
        data = study._read(path)
        data['counts']['successful_updates'] = 0
        write_json(path, data)
        entry['completed_sha256'] = study._sha(path)
    else:
        entry['pair'] = 0

    def forbidden(*args):
        raise AssertionError('No evaluation before every fit is complete')

    with pytest.raises(ValueError):
        study.evaluate(**arguments, env_factory=forbidden, model_loader=forbidden)
    assert not (arguments['out'] / 'all-fits-ready.json').exists()
    assert (arguments['out'] / 'failed.json').exists()


@pytest.mark.parametrize('mutation', ['protocol', 'inputs', 'bindings'])
def test_bad_external_hash_stops_before_model_reset(tmp_path, mutation):
    arguments = fixtures(tmp_path)
    arguments[f'expected_{mutation}_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        study.evaluate(**arguments, env_factory=lambda: pytest.fail('native construction'),
                       model_loader=lambda *a: pytest.fail('model construction'))
    assert study._read(arguments['out'] / 'failed.json')['phase'] == 'authentication'


def test_original_episode_error_survives_close_and_failed_receipt_failure(tmp_path, monkeypatch):
    arguments = fixtures(tmp_path)
    original_write = study._write

    def write(path, value):
        if Path(path).name == 'failed.json':
            raise OSError('failure receipt unavailable')
        return original_write(path, value)

    monkeypatch.setattr(study, '_write', write)
    with pytest.raises(RuntimeError, match='write failure'):
        study.evaluate(**arguments, env_factory=lambda: HandwrittenGame(close_error=True),
                       model_loader=lambda *a: FakeMemory(fail_write=True))
    assert not (arguments['out'] / 'completed.json').exists()
    with np.load(arguments['out'] / 'partial-episode.npz', allow_pickle=False) as data:
        assert data['actions'].tolist() == [0]


def test_successful_episode_still_preserved_if_close_fails(tmp_path, monkeypatch):
    arguments = fixtures(tmp_path)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    with pytest.raises(RuntimeError, match='cleanup failure'):
        study.evaluate(**arguments, env_factory=lambda: HandwrittenGame(close_error=True), model_loader=lambda *a: object())
    assert (arguments['out'] / 'partial-episode.npz').exists()
    assert study._read(arguments['out'] / 'failed.json')['current']['case'] == 0


def test_late_completed_serialization_demotes_success_and_preserves_failure(tmp_path, monkeypatch):
    arguments = fixtures(tmp_path)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    original_write, late = study._write, [False]

    def write(path, value):
        original_write(path, value)
        if Path(path) == arguments['out'] / 'completed.json':
            late[0] = True

    def check(deadline):
        if late[0]:
            raise TimeoutError('late serialization')

    monkeypatch.setattr(study, '_write', write)
    monkeypatch.setattr(study, '_check', check)
    with pytest.raises(TimeoutError, match='late serialization'):
        study.evaluate(**arguments, env_factory=HandwrittenGame, model_loader=lambda *a: object())
    assert not (arguments['out'] / 'completed.json').exists()
    assert (arguments['out'] / 'invalid-completion.json').exists()
    assert (arguments['out'] / 'failed.json').exists()


def test_existing_output_never_overwritten(tmp_path):
    arguments = fixtures(tmp_path)
    arguments['out'].mkdir()
    sentinel = arguments['out'] / 'sentinel'
    sentinel.write_text('retained')
    with pytest.raises(FileExistsError):
        study.evaluate(**arguments)
    assert list(arguments['out'].iterdir()) == [sentinel]


@pytest.mark.parametrize('change_after_start', [False, True])
def test_runtime_binding_at_both_boundaries(tmp_path, monkeypatch, change_after_start):
    arguments = fixtures(tmp_path)
    calls = [0]

    def runtime():
        calls[0] += 1
        return {'synthetic': bool(change_after_start and calls[0] == 1)}

    monkeypatch.setattr(study, 'runtime', runtime)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    with pytest.raises(ValueError, match='Runtime'):
        study.evaluate(**arguments, env_factory=HandwrittenGame, model_loader=lambda *a: object())
    assert not (arguments['out'] / 'completed.json').exists()
    assert calls[0] == (2 if change_after_start else 1)


def test_checkpoint_mode_binding_rejects_same_tensor_schema_without_construction(tmp_path, monkeypatch):
    checkpoint = {'model_class': 'CardAssociativeMemory', 'resume_authorized': False,
                  'configuration': {'mode': 'innovation_local'}}
    monkeypatch.setattr(study.torch, 'load', lambda *a, **kw: checkpoint)
    with pytest.raises(ValueError, match='mode mismatch'):
        study.load_model({'mode': 'innovation_matched', 'checkpoint_path': 'synthetic.pt'}, {}, tmp_path)


def test_tensor_hash_uses_training_compatible_metadata_and_bytes():
    value = torch.tensor([1., 2.], dtype=torch.float32)
    digest = hashlib.sha256()
    digest.update(json.dumps(['weight', 'torch.float32', [2]], separators=(',', ':')).encode())
    digest.update(value.numpy().tobytes())
    assert study._weights_hash({'weight': value}) == digest.hexdigest()
