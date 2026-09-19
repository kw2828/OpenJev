"""Handwritten fake games/models and byte-only orchestration; no native or trained model calls."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/evaluate_card_calibration.py'
SPEC = importlib.util.spec_from_file_location('card_calibration_evaluation_test', PATH)
study = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(PATH.parent), *sys.path]):
    SPEC.loader.exec_module(study)
    import evaluate_card_controllers as prior


class Game:
    """Explicit fixed ranks and legal public timing, never a native environment."""

    def __init__(self, *, bad_info=False, close_error=False):
        self.bad_info, self.close_error = bad_info, close_error
        self.closed, self.actions, self.identity_reads = False, [], 0

    def reset(self, *, seed):
        self.seed, self.matched, self.pending = seed, {}, None
        return np.full(52, 13, np.int64), {}

    def get_state(self):
        assert not self.actions
        self.identity_reads += 1
        return np.arange(52, dtype=np.int64) // 4, None

    def step(self, action):
        self.actions.append(action)
        after = np.full(52, 13, np.int64)
        for pos, rank in self.matched.items():
            after[pos] = rank
        if self.pending is not None:
            after[self.pending] = self.pending // 4
        after[action] = action // 4
        if self.pending is None:
            reward, self.pending = 0., action
        else:
            if action != self.pending and action // 4 == self.pending // 4:
                reward = 2 / 52
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
        self.calls, self.initializations, self.fail_write = [], 0, fail_write

    def state_dict(self):
        return {'weight': torch.tensor([1.0])}

    def init_state(self, batch, device):
        assert (batch, device) == (1, 'cpu')
        self.initializations += 1
        return {'events': torch.zeros(1)}

    def predict(self, state, queries):
        assert queries.tolist() == [list(range(52))]
        self.calls.append(('predict', int(state['events'].item())))
        return torch.arange(13, dtype=torch.float32)[None, None, :].expand(1, 52, 13).clone()

    def write(self, state, pos, rank, valid):
        assert valid.tolist() == [True]
        self.calls.append(('write', int(pos.item()), int(rank.item())))
        if self.fail_write:
            raise RuntimeError('write failure')
        return {'events': state['events'] + 1}, {'diagnostic': torch.zeros(1)}


@pytest.mark.parametrize(('controller', 'policy', 'beta'), [
    ('gru-pair0', 'baseline', None), ('gru-pair0', 'temperature', 2.0),
    ('gru-pair0', 'hard', None), ('exact', 'baseline', None), ('last32', 'baseline', None)])
def test_public_episode_preserves_original_raw_and_uses_fixed_c_once(controller, policy, beta, monkeypatch):
    model = FakeMemory() if controller not in study.REFERENCES else None
    env, transformed, queried = Game(), [], []
    original_transform, original_tracker = study.transform_probabilities, study.CardPolicyTracker

    def transform(raw, mode, beta):
        result = original_transform(raw, mode, beta)
        transformed.append((raw.copy(), result.copy(), mode, beta))
        return result

    class Spy(original_tracker):
        def __init__(self, selected):
            assert selected == 'C'
            super().__init__(selected)

        def decision(self, raw):
            queried.append(raw.copy())
            return super().decision(raw)

        def choose(self, *args, **kwargs):
            pytest.fail('One decision call required')

    monkeypatch.setattr(study, 'transform_probabilities', transform)
    monkeypatch.setattr(study, 'CardPolicyTracker', Spy)
    arrays, receipt = study.evaluate_episode(env, controller=controller, policy=policy,
                                             seed=410, beta=beta, model=model)
    assert receipt['status'] == 'complete' and receipt['success'] and receipt['return'] == 1
    assert receipt['policy'] == policy and receipt['tracker_policy'] == 'C' and receipt['beta'] == beta
    assert receipt['native_steps'] == len(queried) == len(transformed) == 52
    assert receipt['counts']['transform_attempted'] == receipt['counts']['transform_returned'] == 52
    assert receipt['timings']['transform_seconds'] >= 0 and env.identity_reads == 1
    assert arrays['actions'].tolist() == list(range(52)) and arrays['observations'].shape == (53, 52)
    assert arrays['raw_probabilities'].dtype == arrays['picker_probabilities'].dtype == np.float64
    for i, (raw, changed, mode, used_beta) in enumerate(transformed):
        np.testing.assert_array_equal(arrays['raw_probabilities'][i], raw)
        np.testing.assert_array_equal(queried[i], changed)
        assert mode == policy and used_beta == (beta if beta is not None else 1.0)
    if model is not None:
        expected = torch.arange(13, dtype=torch.float32).softmax(-1).numpy().astype(np.float64)
        np.testing.assert_array_equal(arrays['raw_probabilities'][0], np.broadcast_to(expected, (52, 13)))
        assert model.initializations == 1 and len(model.calls) == 104
        assert model.calls[:2] == [('predict', 0), ('write', 0, 0)]
        if policy != 'baseline':
            assert not np.array_equal(transformed[0][0], transformed[0][1])
    else:
        assert receipt['counts']['predict_returned'] == receipt['counts']['write_returned'] == 0


def test_baseline_is_bitwise_prior_c_episode():
    before, old_receipt = prior.evaluate_episode(Game(), controller='gru-pair0', policy='C', seed=410, model=FakeMemory())
    after, new_receipt = study.evaluate_episode(Game(), controller='gru-pair0', policy='baseline', seed=410, model=FakeMemory())
    assert set(before) == set(after)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    assert new_receipt['decisions'] == old_receipt['decisions']
    assert all(new_receipt['counts'][key] == value for key, value in old_receipt['counts'].items())


@pytest.mark.parametrize(('policy', 'beta'), [('baseline', 1.), ('hard', 2.), ('temperature', None),
                                            ('temperature', True), ('temperature', 0.), ('temperature', 21.)])
def test_invalid_beta_stops_before_reset(policy, beta):
    env = Game()
    with pytest.raises(ValueError, match='beta'):
        study.evaluate_episode(env, controller='gru-pair0', policy=policy, beta=beta, seed=410, model=FakeMemory())
    assert not hasattr(env, 'seed')


@pytest.mark.parametrize('policy', ['temperature', 'hard'])
def test_references_only_baseline(policy):
    env = Game()
    with pytest.raises(ValueError, match='baseline-only'):
        study.evaluate_episode(env, controller='exact', policy=policy, beta=2 if policy == 'temperature' else None, seed=410)
    assert not hasattr(env, 'seed')


def test_failed_transform_retains_original_prediction_before_native(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('transform failure')

    monkeypatch.setattr(study, 'transform_probabilities', fail)
    with pytest.raises(RuntimeError, match='transform failure') as caught:
        study.evaluate_episode(Game(), controller='gru-pair0', policy='temperature', beta=2, seed=410, model=FakeMemory())
    receipt, arrays = caught.value.partial_card_receipt, caught.value.partial_card_episode
    assert receipt['phase'] == 'transform' and receipt['counts']['predict_returned'] == 1
    assert receipt['counts']['transform_attempted'] == 1 and receipt['counts']['transform_returned'] == 0
    assert receipt['counts']['native_attempted'] == 0
    assert arrays['raw_probabilities'].shape == (1, 52, 13) and arrays['picker_probabilities'].shape == (0, 52, 13)


def test_failed_write_and_bad_info_preserve_returned_native():
    for env, model, message in ((Game(), FakeMemory(fail_write=True), 'write failure'),
                                 (Game(bad_info=True), FakeMemory(), 'flags/info')):
        with pytest.raises((RuntimeError, ValueError), match=message) as caught:
            study.evaluate_episode(env, controller='gru-pair0', policy='hard', seed=410, model=model)
        r = caught.value.partial_card_receipt
        assert r['counts']['native_returned'] == 1 and r['counts']['write_returned'] == 0
        assert caught.value.partial_card_episode['observations'].shape == (2, 52)


def test_early_reset_and_identity_failures_keep_counters():
    env = Game()
    env.reset = lambda seed: (np.full(52, 13, dtype=np.int64), {'bad': True})
    with pytest.raises(ValueError, match='reset info') as caught:
        study.evaluate_episode(env, controller='exact', policy='baseline', seed=410)
    assert caught.value.partial_card_receipt['counts']['reset_returned'] == 1
    assert caught.value.partial_card_receipt['matched_pairs'] == 0
    env = Game()
    env.get_state = lambda: (np.zeros(52, dtype=np.int64),)
    with pytest.raises(ValueError, match='52 int64') as caught:
        study.evaluate_episode(env, controller='exact', policy='baseline', seed=410)
    assert caught.value.partial_card_receipt['counts']['identity_read_returned'] == 1
    assert not env.actions


def test_expired_deadline_stops_before_reset():
    env = Game()
    with pytest.raises(TimeoutError):
        study.evaluate_episode(env, controller='exact', policy='baseline', seed=410, deadline=0)
    assert not hasattr(env, 'seed')


def fixture(tmp_path, monkeypatch):
    """Only enclosing authentication is substituted; real serialization is used."""
    root = tmp_path / 'repo'
    root.mkdir()
    args = {'root': root, 'out': tmp_path / 'attempt', 'calibration_dir': root / 'calibration',
            'expected_calibration_completed_sha256': 'c' * 64}
    for name in ('protocol', 'inputs', 'checkpoint_map', 'bindings'):
        path = root / (name + '.json')
        path.write_text('{}')
        args[name + '_path'] = path
        args['expected_' + name + '_sha256'] = study._sha(path)
    checkpoints = {name: {'controller': name, 'checkpoint_sha256': 'd' * 64} for name in study.CONTROLLERS if name not in study.REFERENCES}
    fits = {name: {'entry': entry, 'receipt': {'final_weights_sha256': study.old._weights_hash(FakeMemory().state_dict())}}
            for name, entry in checkpoints.items()}
    calibration = {'status': 'complete', 'fits': {name: {'beta': 1.5 + i / 100, 'temperature': 1 / (1.5 + i / 100),
                   'checkpoint_sha256': entry['checkpoint_sha256'], 'receipt_path': f'fits/{name}/completed.json',
                   'receipt_sha256': str(i % 10) * 64} for i, (name, entry) in enumerate(checkpoints.items())}}
    inputs = {'evaluation': [{'seed': 410 + i, 'policy_order': list(study.POLICIES[i % 3:] + study.POLICIES[:i % 3])}
                             for i in range(64)]}
    auth_calls = []

    def authenticate(*received):
        expected = (root, args['protocol_path'], args['expected_protocol_sha256'], args['inputs_path'], args['expected_inputs_sha256'],
                    args['bindings_path'], args['expected_bindings_sha256'], args['checkpoint_map_path'], args['expected_checkpoint_map_sha256'])
        assert received == expected
        for path, digest in zip(received[1::2], received[2::2], strict=True):
            if study._sha(path) != digest:
                raise ValueError('external fixture hash mismatch')
        auth_calls.append('source')
        return {'lineage': {'synthetic': True}}, inputs, checkpoints, fits

    def authenticate_calibration(*received):
        assert received == (args['calibration_dir'], args['expected_calibration_completed_sha256'],
                            args['expected_protocol_sha256'], args['expected_inputs_sha256'],
                            args['expected_bindings_sha256'], args['expected_checkpoint_map_sha256'])
        auth_calls.append('calibration')
        return calibration

    monkeypatch.setattr(study, 'authenticate', authenticate)
    monkeypatch.setattr(study, 'authenticate_calibration', authenticate_calibration)
    return args, calibration, auth_calls


def fake_episode(env, *, controller, policy, beta, seed, model, deadline):
    counts = dict.fromkeys(study.COUNTERS, 0)
    for key in ('reset_attempted', 'reset_returned', 'identity_read_attempted', 'identity_read_returned',
                'transform_attempted', 'transform_returned', 'decision_attempted', 'decision_returned', 'native_attempted', 'native_returned'):
        counts[key] = 1
    if model is not None:
        for key in ('init_attempted', 'init_returned', 'predict_attempted', 'predict_returned', 'write_attempted', 'write_returned'):
            counts[key] = 1
    arrays = {'actions': np.array([0], np.int64), 'observations': np.full((2, 52), 13, np.int64)}
    return arrays, {'status': 'complete', 'controller': controller, 'policy': policy, 'tracker_policy': 'C', 'beta': beta,
                    'seed': seed, 'layout_sha256': 'a' * 64, 'counts': counts, 'native_steps': 1,
                    'return': 0., 'success': False, 'whole_episode_seconds': 0.}


def test_full_fake_3584_episodes_balanced_policies_and7226_payloads(tmp_path, monkeypatch):
    args, calibration, auth_calls = fixture(tmp_path, monkeypatch)
    calls, loads, environments = [], [], []

    def loader(entry, receipt, root):
        assert auth_calls == ['source', 'calibration']
        ready = study._read(args['out'] / 'all-fits-ready.json')
        assert ready['calibration_fits'] == calibration['fits'] and len(ready['fits']) == 18
        loads.append(entry['controller'])
        return FakeMemory()

    def factory():
        env = Game()
        environments.append(env)
        return env

    def episode(env, **kwargs):
        calls.append((kwargs['controller'], kwargs['seed'], kwargs['policy'], kwargs['beta'], id(kwargs['model'])))
        return fake_episode(env, **kwargs)

    monkeypatch.setattr(study, 'evaluate_episode', episode)
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    result = study.evaluate(**args, env_factory=factory, model_loader=loader)
    assert result['episodes'] == len(environments) == len(calls) == 3584 and len(loads) == result['model_restores'] == 18
    assert result['tracker_policy'] == 'C' and result['calibration_completed_sha256'] == args['expected_calibration_completed_sha256']
    assert result['new_fits'] == result['new_optimizer_steps'] == result['native_replay_calls'] == 0
    assert auth_calls == ['source', 'calibration', 'source', 'calibration'] and all(env.closed for env in environments)
    assert len(result['files']) == 7226 and len(list(args['out'].rglob('*.json'))) == 3643
    assert result['counts']['native_returned'] == result['counts']['transform_returned'] == 3584
    assert result['counts']['predict_returned'] == result['counts']['write_returned'] == 3456
    offset = 0
    for controller in study.CONTROLLERS:
        expected = [(controller, 410 + i, policy, calibration['fits'][controller]['beta'] if policy == 'temperature' else None)
                    for i in range(64) for policy in
                    (('baseline',) if controller in study.REFERENCES else study.POLICIES[i % 3:] + study.POLICIES[:i % 3])]
        actual = calls[offset:offset + len(expected)]
        assert [event[:4] for event in actual] == expected and len({event[4] for event in actual}) == 1
        offset += len(expected)
        for policy in (('baseline',) if controller in study.REFERENCES else study.POLICIES):
            episode_receipt = study._read(args['out'] / 'controllers' / controller / policy / 'episodes/000.json')
            assert episode_receipt['calibration_completed_sha256'] == args['expected_calibration_completed_sha256']
            assert episode_receipt['calibration_receipt_sha256'] == (None if controller in study.REFERENCES else calibration['fits'][controller]['receipt_sha256'])
    for relative, item in result['files'].items():
        assert study._sha(args['out'] / relative) == item['sha256']
    assert torch.get_num_threads() == before_threads and torch.are_deterministic_algorithms_enabled() == before_deterministic


def test_failed_calibration_stops_before_restore_and_reset(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError('unbound calibration')

    monkeypatch.setattr(study, 'authenticate_calibration', reject)
    with pytest.raises(ValueError, match='unbound calibration'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('reset factory'), model_loader=lambda *a: pytest.fail('model loader'))
    assert not (args['out'] / 'all-fits-ready.json').exists()
    assert study._read(args['out'] / 'failed.json')['completed_episodes'] == 0


@pytest.mark.parametrize('corruption', ['checkpoint', 'beta', 'membership'])
def test_calibration_must_match_all_authenticated_models_before_restore(tmp_path, monkeypatch, corruption):
    args, calibration, _ = fixture(tmp_path, monkeypatch)
    last = next(reversed(calibration['fits']))
    if corruption == 'checkpoint':
        calibration['fits'][last]['checkpoint_sha256'] = 'e' * 64
    elif corruption == 'beta':
        calibration['fits'][last]['beta'] = float('nan')
    else:
        calibration['fits'].pop(last)
    with pytest.raises(ValueError, match='[Cc]alibration'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('native'), model_loader=lambda *a: pytest.fail('model'))
    assert not (args['out'] / 'all-fits-ready.json').exists()


@pytest.mark.parametrize('name', ['protocol', 'inputs', 'bindings', 'checkpoint_map'])
def test_external_arguments_reach_authentication_before_work(tmp_path, monkeypatch, name):
    args, _, _ = fixture(tmp_path, monkeypatch)
    args[name + '_path'].write_text('changed bytes')
    with pytest.raises(ValueError, match='hash mismatch'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('native'), model_loader=lambda *a: pytest.fail('model'))


def test_failed_write_keeps_original_error_even_bad_close_and_receipt_failure(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    original = study._write

    def write(path, value):
        if Path(path).name == 'failed.json':
            raise OSError('failed receipt blocked')
        original(path, value)

    monkeypatch.setattr(study, '_write', write)
    with pytest.raises(RuntimeError, match='write failure') as caught:
        study.evaluate(**args, env_factory=lambda: Game(close_error=True), model_loader=lambda *a: FakeMemory(fail_write=True))
    assert (args['out'] / 'partial-episode.npz').exists() and not (args['out'] / 'completed.json').exists()
    assert caught.value.partial_card_receipt['counts']['native_returned'] == 1


def test_close_failure_after_completed_episode_preserves_payload(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    with pytest.raises(RuntimeError, match='close failure'):
        study.evaluate(**args, env_factory=lambda: Game(close_error=True), model_loader=lambda *a: FakeMemory())
    assert (args['out'] / 'partial-episode.npz').exists()
    assert study._read(args['out'] / 'failed.json')['partial_receipt']['status'] == 'complete'


def test_late_completion_cap_demotes_without_double_charging_last_episode(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(study, 'evaluate_episode', fake_episode)
    original, late = study._write, [False]

    def write(path, value):
        original(path, value)
        if path == args['out'] / 'completed.json':
            late[0] = True

    def check(deadline):
        if late[0]:
            raise TimeoutError('late completed serialization')

    monkeypatch.setattr(study, '_write', write)
    monkeypatch.setattr(study, '_check', check)
    with pytest.raises(TimeoutError, match='late completed'):
        study.evaluate(**args, env_factory=Game, model_loader=lambda *a: FakeMemory())
    assert not (args['out'] / 'completed.json').exists() and (args['out'] / 'invalid-completion.json').exists()
    failed = study._read(args['out'] / 'failed.json')
    assert failed['completed_episodes'] == failed['completed_episode_counts']['native_returned'] == 3584
    assert failed['partial_receipt'] is None and not (args['out'] / 'partial-episode.npz').exists()


def test_storage_guard_and_exclusive_output(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    with patch.dict(study.EVALUATION, output_cap_bytes=1), pytest.raises(ValueError, match='storage cap'):
        study.evaluate(**args, env_factory=lambda: pytest.fail('native'))
    failed_bytes = (args['out'] / 'failed.json').read_bytes()
    with pytest.raises(FileExistsError):
        study.evaluate(**args)
    assert (args['out'] / 'failed.json').read_bytes() == failed_bytes


def test_first_pair_layout_mismatch_retains_second_prefix(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)
    calls = []

    def episode(env, **kwargs):
        arrays, receipt = fake_episode(env, **kwargs)
        calls.append(kwargs['policy'])
        receipt['layout_sha256'] = str(len(calls)) * 64
        return arrays, receipt

    monkeypatch.setattr(study, 'evaluate_episode', episode)
    with pytest.raises(ValueError, match='different native layout'):
        study.evaluate(**args, env_factory=Game, model_loader=lambda *a: FakeMemory())
    assert calls == ['baseline', 'temperature']
    assert (args['out'] / 'controllers/delta-pair0/baseline/episodes/000.npz').exists()
    assert (args['out'] / 'partial-episode.npz').exists()
