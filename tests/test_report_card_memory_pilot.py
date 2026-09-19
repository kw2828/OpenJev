"""Saved-array and receipt arithmetic only; no native, model, or RNG calls."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def reporter(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('card_report_test', scripts / 'report_card_memory_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('left', 'right', 'margin', 'passed'), [
    (.13, .10, .03, True), (.52, .50, .02, True), (.129999, .10, .03, False),
    (None, .50, .02, False), (.50, None, .02, False),
])
def test_frozen_inclusive_decimal_margin(reporter, left, right, margin, passed):
    assert reporter.margin_pass(left, right, margin) is passed


def saved_episode(reporter, tmp_path):
    tracker = reporter.PublicCardTracker()
    obs = np.full(52, 13, dtype=np.int64)
    tracker.reset(obs)
    frames, actions, ranks, rewards, picker = [obs.copy()], [], [], [], []
    raw = np.full((52, 13), 1/13, dtype=np.float64)
    # Handwritten public trace: adjacent pairs are equal, all 52 cards eventually matched.
    for t in range(52):
        action = tracker.choose(raw)
        assert action == t
        picker.append(tracker.probabilities(raw))
        obs = obs.copy()
        obs[action] = action // 4
        reward = 0.0 if t % 2 == 0 else 2/52
        reveal = tracker.observe(action, reward, obs)
        frames.append(obs.copy())
        actions.append(action)
        ranks.append(reveal.rank)
        rewards.append(reward)
    arrays = {'actions': np.array(actions, dtype=np.int64), 'ranks': np.array(ranks, dtype=np.int64),
                  'rewards': np.array(rewards, dtype=np.float64), 'observations': np.array(frames),
                  'terminated': np.array([False]*51+[True]), 'truncated': np.zeros(52, dtype=bool),
                  'raw_probabilities': np.repeat(raw[None], 52, axis=0), 'picker_probabilities': np.array(picker)}
    path = tmp_path / 'episode.npz'
    np.savez(path, **arrays)
    receipt = dict(status='complete', npz_sha256=reporter.sha(path), native_steps=52, success=True,
                   matched_pairs=26, controller='delta-pair0', **{'return':1.0}, counts={
                       'reset_attempted':1, 'reset_returned':1, 'native_attempted':52, 'native_returned':52,
                       'predict_attempted':52, 'predict_returned':52, 'write_attempted':52, 'write_returned':52})
    return path, receipt, arrays


def test_saved_public_trace_complete(reporter, tmp_path):
    path, receipt, _ = saved_episode(reporter, tmp_path)
    assert reporter.audit_episode(path, receipt) == 52


@pytest.mark.parametrize('field', ['ranks', 'actions', 'picker_probabilities', 'terminated'])
def test_resealed_semantic_corruption_rejected(reporter, tmp_path, field):
    path, receipt, arrays = saved_episode(reporter, tmp_path)
    arrays[field][0] = 1 if field != 'picker_probabilities' else 0
    np.savez(path, **arrays)
    receipt['npz_sha256'] = reporter.sha(path)
    with pytest.raises(ValueError):
        reporter.audit_episode(path, receipt)


def test_reference_cannot_claim_neural_work(reporter, tmp_path):
    path, receipt, _ = saved_episode(reporter, tmp_path)
    receipt['controller'] = 'exact'
    with pytest.raises(ValueError, match='learned calls'):
        reporter.audit_episode(path, receipt)


def test_training_authentication_rejects_unbound_development(reporter, tmp_path):
    training = tmp_path / 'train'
    training.mkdir()
    names = [f'{mode}-pair{i}' for mode in reporter.MODES for i in range(3)]
    checkpoints, fits, development = {}, [], {}
    for name in names:
        mode, pair = name.rsplit('-pair', 1)
        folder = training / 'fits' / name
        folder.mkdir(parents=True)
        checkpoint = folder / 'final-checkpoint.pt'
        checkpoint.write_bytes(b'handwritten unused tensor placeholder')
        done = {'status':'complete', 'counts':{'successful_updates':128},
                'files':{'final-checkpoint.pt':{'sha256':reporter.sha(checkpoint), 'bytes':checkpoint.stat().st_size}}}
        reporter.write_json(folder / 'completed.json', done)
        initial = training / (name+'-initial.pt')
        initial.write_bytes(b'handwritten unused initial placeholder')
        entry = {'mode':mode, 'pair':int(pair), 'checkpoint_path':str(checkpoint.relative_to(tmp_path)),
                 'checkpoint_sha256':reporter.sha(checkpoint),
                 'completed_path':str((folder/'completed.json').relative_to(tmp_path)),
                 'completed_sha256':reporter.sha(folder/'completed.json')}
        fitted = {'id':name, 'initial_sha256':reporter.sha(initial), **entry}
        reporter.write_json(training/(name+'-receipt.json'), fitted)
        dev = {'all':{'accuracy':.5}, 'age_gt32':{'accuracy':.4, 'eligible_episodes':32}}
        reporter.write_json(training/(name+'-development.json'), {**dev, 'per_episode':[]})
        development[name] = dev
        checkpoints[name] = entry
        fits.append(fitted)
    reporter.write_json(training/'started.json', {})
    reporter.write_json(training/'checkpoint-map.json', checkpoints)
    reporter.write_json(training/'training-completed.json', {'status':'complete', 'fits':fits,
                        'development_or_native_evaluations':0, 'checkpoint_map_sha256':reporter.sha(training/'checkpoint-map.json')})
    train = {'fits': 18, 'updates': 2304, 'development': development,
                 'started_sha256': reporter.sha(training/'started.json'),
                 'training_completed_sha256': reporter.sha(training/'training-completed.json'),
                 'checkpoint_map_sha256': reporter.sha(training/'checkpoint-map.json')}
    assert reporter.authenticate_training(tmp_path, training, train) == checkpoints
    path = training/'innovation_local-pair0-development.json'
    altered = json.loads(path.read_text())
    altered['age_gt32']['accuracy'] = .99
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='Unbound development'):
        reporter.authenticate_training(tmp_path, training, train)
