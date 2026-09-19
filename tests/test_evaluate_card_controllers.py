"""Synthetic byte trees and handwritten games only; no native/model/RNG calls."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_card_controllers.py'
SPEC = importlib.util.spec_from_file_location('card_controllers_test', PATH)
study = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    SPEC.loader.exec_module(study)


class Game:
    """Fixed handwritten ranks and public transitions, not a native simulator."""

    def __init__(self, *, close_error=False, bad_info=False):
        self.closed, self.close_error, self.bad_info = False, close_error, bad_info
        self.actions, self.identity_reads = [], 0

    def reset(self, *, seed):
        self.seed = seed
        self.matched, self.pending = {}, None
        return np.full(52, 13, dtype=np.int64), {}

    def get_state(self):
        assert not self.actions  # Privileged identity reads only after reset, before action.
        self.identity_reads += 1
        return np.arange(52, dtype=np.int64) // 4, None, None, None

    def step(self, action):
        self.actions.append(action)
        after = np.full(52, 13, dtype=np.int64)
        for pos, rank in self.matched.items():
            after[pos] = rank
        if self.pending is not None:
            after[self.pending] = self.pending // 4
        after[action] = action // 4
        if self.pending is None:
            reward, self.pending = 0.0, action
        else:
            if self.pending != action and self.pending // 4 == action // 4:
                reward = 1 / 26
                self.matched[action] = action // 4
                self.matched[self.pending] = self.pending // 4
            else:
                reward = -2 / 104
            self.pending = None
        return after, reward, len(self.matched) == 52, len(self.actions) == 104, ({'bad': True} if self.bad_info else {})

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError('close failure')


class FakeMemory:
    def __init__(self, *, fail_write=False):
        self.calls, self.fail_write, self.initializations = [], fail_write, 0

    def state_dict(self):
        return {'weight': torch.tensor([1.0])}

    def init_state(self, batch, device):
        assert (batch, device) == (1, 'cpu')
        self.initializations += 1
        return {'events': torch.zeros(1)}

    def predict(self, state, queries):
        assert queries.tolist() == [list(range(52))]
        self.calls.append(('predict', int(state['events'].item())))
        return torch.zeros(1, 52, 13)

    def write(self, state, pos, rank, valid):
        assert valid.tolist() == [True]
        self.calls.append(('write', int(pos.item()), int(rank.item())))
        if self.fail_write:
            raise RuntimeError('write failure')
        return {'events': state['events'] + 1}, {'diagnostic': torch.zeros(1)}


@pytest.mark.parametrize('policy', study.POLICIES)
@pytest.mark.parametrize('controller', ['exact', 'last32', 'gru-pair0'])
def test_episode_public_read_before_write_single_decision_and_identity(controller, policy, monkeypatch):
    model = FakeMemory() if controller == 'gru-pair0' else None
    env = Game()
    calls = []
    tracker_type = study.CardPolicyTracker

    class Spy(tracker_type):
        def decision(self, raw):
            calls.append(raw.copy())
            return super().decision(raw)

        def choose(self, *args, **kwargs):
            pytest.fail('Driver must use decision exactly once, not separate choose')

    monkeypatch.setattr(study, 'CardPolicyTracker', Spy)
    arrays, receipt = study.evaluate_episode(env, controller=controller, policy=policy, seed=410, model=model)
    assert receipt['status'] == 'complete' and receipt['return'] == 1 and receipt['success']
    assert receipt['native_steps'] == 52 and receipt['counts']['decision_returned'] == 52
    assert len(calls) == len(receipt['decisions']) == 52 and env.identity_reads == 1
    assert arrays['actions'].tolist() == list(range(52))
    assert arrays['observations'].shape == (53, 52)
    assert arrays['raw_probabilities'].shape == arrays['picker_probabilities'].shape == (52, 52, 13)
    for name in arrays:
        assert 'layout' not in name and 'deck' not in name
    if model is not None:
        assert model.initializations == 1
        for step in range(52):
            assert model.calls[2 * step] == ('predict', step)
            assert model.calls[2 * step + 1] == ('write', step, step // 4)
        assert receipt['counts']['predict_returned'] == receipt['counts']['write_returned'] == 52
    else:
        assert receipt['counts']['predict_returned'] == receipt['counts']['write_returned'] == 0


def test_a_episode_exact_old_public_actions_raw_and_picker_arrays():
    old_arrays, old_receipt = study.old.evaluate_episode(Game(), controller='gru-pair0', seed=410, model=FakeMemory())
    new_arrays, new_receipt = study.evaluate_episode(Game(), controller='gru-pair0', policy='A',
                                                   seed=410, model=FakeMemory())
    for key in old_arrays:
        np.testing.assert_array_equal(new_arrays[key], old_arrays[key])
    assert new_receipt['return'] == old_receipt['return']


def test_failed_write_preserves_returned_native_prefix():
    with pytest.raises(RuntimeError, match='write failure') as caught:
        study.evaluate_episode(Game(), controller='gru-pair0', policy='C', seed=410,
                               model=FakeMemory(fail_write=True))
    receipt = caught.value.partial_card_receipt
    assert receipt['counts']['native_returned'] == receipt['counts']['write_attempted'] == 1
    assert receipt['counts']['write_returned'] == 0 and len(receipt['decisions']) == 1
    assert caught.value.partial_card_episode['observations'].shape == (2, 52)


def test_bad_info_preserves_actual_return_and_expired_deadline_prevents_reset():
    with pytest.raises(ValueError, match='flags/info') as caught:
        study.evaluate_episode(Game(bad_info=True), controller='exact', policy='B', seed=410)
    assert caught.value.partial_card_receipt['counts']['native_returned'] == 1
    env = Game()
    with pytest.raises(TimeoutError):
        study.evaluate_episode(env, controller='exact', policy='B', seed=410, deadline=0)
    assert not hasattr(env, 'seed') and env.identity_reads == 0


def test_malformed_reset_info_preserves_returned_reset_before_tracker_ready():
    env = Game()
    env.reset = lambda seed: (np.full(52, 13, dtype=np.int64), {'unexpected': True})
    with pytest.raises(ValueError, match='reset info') as caught:
        study.evaluate_episode(env, controller='exact', policy='A', seed=410)
    receipt = caught.value.partial_card_receipt
    assert receipt['counts']['reset_returned'] == receipt['counts']['reset_attempted'] == 1
    assert receipt['counts']['identity_read_attempted'] == receipt['matched_pairs'] == 0
    assert caught.value.partial_card_episode['observations'].shape == (1, 52)


@pytest.mark.parametrize('bad', [np.zeros(52, dtype=np.int64), np.arange(52, dtype=np.int32) // 4,
                               np.arange(51, dtype=np.int64) // 4])
def test_bad_hidden_identity_prevents_model_init_or_actions(bad):
    env, model = Game(), FakeMemory()
    env.get_state = lambda: (bad,)
    with pytest.raises(ValueError, match='52 int64') as caught:
        study.evaluate_episode(env, controller='gru-pair0', policy='B', seed=410, model=model)
    assert not env.actions and not model.calls and model.initializations == 0
    assert caught.value.partial_card_receipt['counts']['identity_read_attempted'] == 1
    assert caught.value.partial_card_receipt['counts']['identity_read_returned'] == 1


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': path.name, 'sha256': study._sha(path)}


def fixtures(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    monkeypatch.setattr(study, 'runtime', lambda: {'synthetic': True})
    checkpoint_map, fit_rows = {}, []
    weight_hash = study.old._weights_hash(FakeMemory().state_dict())
    for mode in study.MODES:
        for pair in range(3):
            name = f'{mode}-pair{pair}'
            folder = root / 'fits' / name
            folder.mkdir(parents=True)
            checkpoint = folder / 'final-checkpoint.pt'
            checkpoint.write_bytes(b'opaque synthetic checkpoint')
            completed = folder / 'completed.json'
            put(completed, {'status': 'complete', 'evaluation_calls': 0,
                            'counts': {'successful_updates': 128, 'completed_epochs': 16},
                            'recipe': {'expected_updates': 128, 'epochs': 16}, 'final_weights_sha256': weight_hash,
                            'files': {checkpoint.name: {'sha256': study._sha(checkpoint),
                                                       'bytes': checkpoint.stat().st_size}}})
            entry = {'mode': mode, 'pair': pair, 'checkpoint_path': str(checkpoint.relative_to(root)),
                     'checkpoint_sha256': study._sha(checkpoint), 'completed_path': str(completed.relative_to(root)),
                     'completed_sha256': study._sha(completed)}
            checkpoint_map[name] = entry
            fit_rows.append({'id': name, **entry})
    lineage = {}
    lineage['checkpoint_map'] = put(root / 'checkpoint-map.json', checkpoint_map)
    lineage['old_protocol'] = put(root / 'old-protocol.json', {'version': 'card-memory-pilot-v1'})
    lineage['old_inputs'] = put(root / 'old-inputs.json', {'data': [{'seed': 410, 'behavior_seed': 411}],
                                                       'evaluation': [{'seed': 412}]})
    old_files = {lineage[key]['path']: lineage[key]['sha256'] for key in ('old_protocol', 'old_inputs')}
    for index in range(76):
        path = root / f'old-source{index:02d}.txt'
        path.write_text('synthetic frozen source')
        old_files[path.name] = study._sha(path)
    lineage['old_bindings'] = put(root / 'old-bindings.json', {'version': 'card-memory-pilot-v1',
        'runtime': {'synthetic': True}, 'files': old_files})
    started = put(root / 'started.json', {'status': 'started'})
    lineage['training_boundary'] = put(root / 'training-boundary.json', {'status': 'complete',
        'development_or_native_evaluations': 0, 'fits': fit_rows,
        'checkpoint_map_sha256': lineage['checkpoint_map']['sha256']})
    lineage['training_completed'] = put(root / 'training.json', {'status': 'complete', 'fits': 18,
        'updates': 2304, 'native_evaluation_calls': 0, 'started_sha256': started['sha256'],
        'checkpoint_map_sha256': lineage['checkpoint_map']['sha256'],
        'training_completed_sha256': lineage['training_boundary']['sha256']})
    lineage['independent_review'] = put(root / 'review.json', {'status': 'complete', 'bindings': {
        'source_count': 78, 'source_bindings_sha256': lineage['old_bindings']['sha256'],
        'training_completed_sha256': lineage['training_completed']['sha256'],
        'protocol_sha256': lineage['old_protocol']['sha256']}})
    protocol = root / 'protocol.json'
    put(protocol, {'version': study.VERSION, 'evaluation': study.EVALUATION,
                   'bindings_file': 'bindings.json', 'lineage': lineage})
    inputs = root / 'inputs.json'
    put(inputs, {'version': study.VERSION, 'evaluation': [
        {'seed': 1000 + i, 'policy_order': list(study.POLICIES[i % 3:] + study.POLICIES[:i % 3])}
        for i in range(64)]})
    bindings = root / 'bindings.json'
    sources = {}
    for name in ('scripts/evaluate_card_controllers.py', 'src/openjev/research/card_memory_picker.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic runtime source fixture')
        sources[name] = study._sha(path)
    put(bindings, {'version': study.VERSION, 'runtime': {'synthetic': True},
                   'files': {'protocol.json': study._sha(protocol), 'inputs.json': study._sha(inputs), **sources}})
    return {'protocol_path': protocol, 'expected_protocol_sha256': study._sha(protocol),
            'inputs_path': inputs, 'expected_inputs_sha256': study._sha(inputs),
            'checkpoint_map_path': root / 'checkpoint-map.json',
            'expected_checkpoint_map_sha256': lineage['checkpoint_map']['sha256'],
            'expected_bindings_sha256': study._sha(bindings), 'root': root, 'out': tmp_path / 'attempt'}


def fake_episode(env, *, controller, policy, seed, model, deadline):
    arrays = {'actions': np.array([0], dtype=np.int64), 'observations': np.full((2, 52), 13, dtype=np.int64)}
    counts = dict.fromkeys(study.COUNTERS, 0)
    counts.update(reset_attempted=1, reset_returned=1, identity_read_attempted=1, identity_read_returned=1,
                  decision_attempted=1, decision_returned=1, native_attempted=1, native_returned=1)
    if model is not None:
        for key in ('init_attempted', 'init_returned', 'predict_attempted', 'predict_returned',
                    'write_attempted', 'write_returned'):
            counts[key] = 1
    return arrays, {'status': 'complete', 'controller': controller, 'policy': policy, 'seed': seed,
                    'layout_sha256': 'a' * 64, 'counts': counts, 'native_steps': 1, 'return': 0.0,
                    'success': False, 'whole_episode_seconds': 0.0}


def test_full_fake_3840_coverage_balanced_order18_restores_and_exact_manifest(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    calls, loads, environments = [], [], []

    def loader(entry, receipt, root):
        assert (args['out'] / 'all-fits-ready.json').is_file()
        assert len(study._read(args['out'] / 'all-fits-ready.json')['fits']) == 18
        loads.append((entry['mode'], entry['pair']))
        return FakeMemory()

    def episode(env, **kwargs):
        calls.append((kwargs['controller'], kwargs['seed'], kwargs['policy'], id(kwargs['model'])))
        return fake_episode(env, **kwargs)

    def factory():
        env = Game()
        environments.append(env)
        return env

    monkeypatch.setattr(study, 'evaluate_episode', episode)
    result = study.evaluate(**args, env_factory=factory, model_loader=loader)
    assert result['episodes'] == 3840 and len(calls) == len(environments) == 3840 and len(loads) == 18
    assert result['model_restores'] == 18 and result['counts']['identity_read_returned'] == 3840
    assert all(env.closed for env in environments) and result['distinct_layouts'] == 1
    assert len(result['files']) == 7742 and result['new_fits'] == result['native_replay_calls'] == 0
    for row, controller in enumerate(study.CONTROLLERS):
        actual = calls[row * 192:(row + 1) * 192]
        expected = [(controller, 1000 + i, p) for i in range(64)
                    for p in study.POLICIES[i % 3:] + study.POLICIES[:i % 3]]
        assert [event[:3] for event in actual] == expected
        assert len({event[3] for event in actual}) == 1
    for relative, binding in result['files'].items():
        assert study._sha(args['out'] / relative) == binding['sha256']
    assert result['counts']['predict_returned'] == result['counts']['write_returned'] == 3456


@pytest.mark.parametrize('what', ['old_source', 'old_runtime', 'new_runtime', 'review', 'training',
                                'last_checkpoint', 'failed_fit', 'schedule', 'old_seed', 'duplicate_seed'])
def test_authentication_corruption_stops_before_any_model_or_native(tmp_path, monkeypatch, what):
    args = fixtures(tmp_path, monkeypatch)
    root = args['root']
    if what == 'old_source':
        (root / 'old-source00.txt').write_text('changed')
    elif what in ('old_runtime', 'new_runtime'):
        file = root / ('old-bindings.json' if what == 'old_runtime' else 'bindings.json')
        value = study._read(file)
        value['runtime'] = {'changed': True}
        put(file, value)
    elif what in ('review', 'training'):
        file = root / ('review.json' if what == 'review' else 'training.json')
        value = study._read(file)
        value['status'] = 'failed'
        put(file, value)
    elif what == 'last_checkpoint':
        (root / 'fits/gru-pair2/final-checkpoint.pt').write_bytes(b'corrupt')
    elif what == 'failed_fit':
        (root / 'fits/gru-pair2/failed.json').write_text('{}')
    else:
        value = study._read(args['inputs_path'])
        if what == 'schedule':
            value['evaluation'][0]['policy_order'] = ['B', 'C', 'A']
        else:
            value['evaluation'][0]['seed'] = 410 if what == 'old_seed' else 1001
        put(args['inputs_path'], value)
        args['expected_inputs_sha256'] = study._sha(args['inputs_path'])
        value = study._read(root / 'bindings.json')
        value['files']['inputs.json'] = args['expected_inputs_sha256']
        put(root / 'bindings.json', value)
        args['expected_bindings_sha256'] = study._sha(root / 'bindings.json')
    with pytest.raises(ValueError):
        study.evaluate(**args, env_factory=lambda: pytest.fail('Native factory'),
                       model_loader=lambda *a: pytest.fail('Model construction'))
    assert (args['out'] / 'failed.json').is_file() and not (args['out'] / 'all-fits-ready.json').exists()


@pytest.mark.parametrize('external', ['protocol', 'inputs', 'checkpoint_map', 'bindings'])
def test_external_hashes_are_mandatory(tmp_path, monkeypatch, external):
    args = fixtures(tmp_path, monkeypatch)
    args[f'expected_{external}_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('Native factory'))
    assert study._read(args['out'] / 'failed.json')['phase'] == 'authentication'


@pytest.mark.parametrize('missing', ['scripts/evaluate_card_controllers.py',
                                   'src/openjev/research/card_memory_picker.py'])
def test_runtime_critical_membership_cannot_be_dropped_from_resealed_bindings(tmp_path, monkeypatch, missing):
    args = fixtures(tmp_path, monkeypatch)
    path = args['root'] / 'bindings.json'
    value = study._read(path)
    del value['files'][missing]
    put(path, value)
    args['expected_bindings_sha256'] = study._sha(path)
    with pytest.raises(ValueError, match='driver and picker'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('Native factory'),
                       model_loader=lambda *a: pytest.fail('Model construction'))


def test_failure_keeps_native_prefix_and_original_error_with_bad_close(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='write failure'):
        study.evaluate(**args, env_factory=lambda: Game(close_error=True),
                       model_loader=lambda *a: FakeMemory(fail_write=True))
    receipt = study._read(args['out'] / 'failed.json')
    assert receipt['partial_receipt']['counts']['native_returned'] == 1
    assert receipt['partial_receipt']['counts']['write_returned'] == 0
    assert (args['out'] / 'partial-episode.npz').exists() and not (args['out'] / 'completed.json').exists()


def test_close_failure_after_complete_episode_keeps_payload(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    with pytest.raises(RuntimeError, match='close failure'):
        study.evaluate(**args, env_factory=lambda: Game(close_error=True), model_loader=lambda *a: FakeMemory())
    assert (args['out'] / 'partial-episode.npz').exists()
    assert study._read(args['out'] / 'failed.json')['partial_receipt']['status'] == 'complete'


def test_postcompletion_cap_demotes_success_and_preserves_everything(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    original, late = study._write, [False]

    def write(path, value):
        original(path, value)
        if path == args['out'] / 'completed.json':
            late[0] = True

    def check(deadline):
        if late[0]:
            raise TimeoutError('late output hash deadline')

    monkeypatch.setattr(study, '_write', write)
    monkeypatch.setattr(study, '_check', check)
    with pytest.raises(TimeoutError, match='late output'):
        study.evaluate(**args, env_factory=Game, model_loader=lambda *a: FakeMemory())
    assert not (args['out'] / 'completed.json').exists()
    assert (args['out'] / 'invalid-completion.json').exists() and (args['out'] / 'failed.json').exists()
    failed = study._read(args['out'] / 'failed.json')
    assert failed['completed_episodes'] == 3840 and failed['completed_episode_counts']['native_returned'] == 3840
    assert failed['partial_receipt'] is None and not (args['out'] / 'partial-episode.npz').exists()


def test_storage_cap_failure_prevents_first_native_call(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    monkeypatch.setitem(study.EVALUATION, 'output_cap_bytes', 1)
    with pytest.raises(ValueError, match='storage cap'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('Native factory'))
    assert study._read(args['out'] / 'failed.json')['completed_episodes'] == 0


def test_layout_mismatch_stops_after_second_fake_episode_and_retains_both(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    calls = []

    def episode(env, **kwargs):
        arrays, receipt = fake_episode(env, **kwargs)
        calls.append(kwargs['policy'])
        receipt['layout_sha256'] = str(len(calls)) * 64
        return arrays, receipt

    monkeypatch.setattr(study, 'evaluate_episode', episode)
    with pytest.raises(ValueError, match='different native layout'):
        study.evaluate(**args, env_factory=Game, model_loader=lambda *a: FakeMemory())
    assert calls == ['A', 'B'] and (args['out'] / 'controllers/delta-pair0/A/episodes/000.npz').exists()
    assert (args['out'] / 'partial-episode.npz').exists()


def test_existing_attempt_never_overwritten(tmp_path, monkeypatch):
    args = fixtures(tmp_path, monkeypatch)
    args['out'].mkdir()
    sentinel = args['out'] / 'retained'
    sentinel.write_text('untouched')
    with pytest.raises(FileExistsError):
        study.evaluate(**args)
    assert list(args['out'].iterdir()) == [sentinel]
