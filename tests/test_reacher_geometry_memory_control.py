"""Fresh seed410 engineering tensors/cases only; no inherited or scored models."""

import json

import numpy as np
import pytest
import torch

from openjev.research import reacher_geometry_control as frozen
from openjev.research import reacher_geometry_memory_control as control
from openjev.research.reacher_adaptive_search import SearchInputs
from openjev.research.reacher_cache_training import REGISTRY
from openjev.research.reacher_geometry_reward import geometry_reward
from openjev.research.reacher_objective_training import canonical_tensor_hash
from openjev.research.robotics_reacher import native_replay


@pytest.fixture(autouse=True)
def isolated_engineering_rng():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def plan():
    return {'engineering': True, 'rng_namespace': 'reacher-geometry-memory-engineering-unit-v1',
            'steps': 50, 'control_episodes': 1, 'noise_std': .05, 'dt': .02,
            'hidden_size': 4, 'mlp_width': 7, 'planning_horizon': 2, 'action_block': 1,
            'score_modes': ['geometry']}


def model(kind):
    torch.manual_seed(410)
    size = {'width': 7} if kind in ('cached_mlp', 'packet_mlp') else {'hidden_size': 4}
    return REGISTRY[kind](**size, noise_std=.05).eval()


def packet(n=1, *, valid=True, age=0.):
    return torch.tensor([[1., 1., 0., 0., .05, -.03, float(valid), age]]).repeat(n, 1)


def root(student, n=1):
    with torch.no_grad():
        return student.assimilate(student.initial(n), packet(n))


def innovations(horizon=2, block=1, n=1):
    chunks = (horizon + block - 1) // block
    rng = np.random.default_rng(410)
    arrays = [rng.normal(size=(n, k, chunks, 2)) for k in (64, 192, 64, 64, 63)]
    return SearchInputs(arrays[0], arrays[1], tuple(arrays[2:]))


def cases(n=1):
    schedule = np.ones(51, dtype=bool)
    schedule[8:14] = schedule[28:34] = False
    return [control.ControlCase(410, 410, schedule) for _ in range(n)]


def records(stem):
    metadata = json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
        result = []
        for index, meta in enumerate(metadata):
            row = {'metadata': meta, 'policy': {}, 'audit': {}}
            for name in saved.files:
                category, key = name.split('__', 1)
                row[category][key] = saved[name][index].copy()
            result.append(row)
        return result


@pytest.mark.parametrize('kind', control.KINDS)
def test_all_classes_score_exact_geometry_preserve_original_heads_and_private_candidates(kind):
    cfg, student = plan(), model(kind)
    state = root(student, 2)
    state_hash, weights = canonical_tensor_hash(state), canonical_tensor_hash(student.state_dict())
    bank = np.linspace(-.3, .3, 2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    unchanged = bank.copy()
    scores, diagnostic = control.score_bank(cfg, student, state, bank)
    arrays, meta = diagnostic['arrays'], diagnostic['metadata']
    with torch.no_grad():
        imagined = {name: tensor.repeat_interleave(3, dim=0) for name, tensor in state.items()}
        angles, rewards = [], []
        for t in range(2):
            imagined, predicted, learned = student.advance(imagined, torch.from_numpy(bank[:, :, t].reshape(6, 2).copy()))
            angles.append(predicted.reshape(2, 3, 4).numpy())
            rewards.append(learned.reshape(2, 3).numpy())
    np.testing.assert_array_equal(arrays['predicted_angles'], np.stack(angles, 2))
    np.testing.assert_array_equal(arrays['learned_rewards'], np.stack(rewards, 2))
    target = torch.from_numpy(np.broadcast_to(state['packet'][:, None, None, 4:6].numpy(), (2, 3, 2, 2)).copy())
    expected = geometry_reward(torch.from_numpy(arrays['predicted_angles']), target, torch.from_numpy(bank.copy()), .05).numpy()
    np.testing.assert_array_equal(arrays['selected_rewards'], expected)
    sums = np.zeros((2, 3), np.float32)
    for t in range(2):
        sums += np.clip(expected[:, :, t], -2.5, 0)
    np.testing.assert_array_equal(scores, sums)
    changed = bank.copy()
    changed[:, 0] *= -1
    other_scores, other = control.score_bank(cfg, student, state, changed)
    np.testing.assert_array_equal(other_scores[:, 1:], scores[:, 1:])
    np.testing.assert_array_equal(other['arrays']['predicted_angles'][:, 1:], arrays['predicted_angles'][:, 1:])
    assert canonical_tensor_hash(state) == state_hash
    assert canonical_tensor_hash(student.state_dict()) == weights
    np.testing.assert_array_equal(bank, unchanged)
    assert all(parameter.grad is None for parameter in student.parameters())
    assert meta['configuration']['version'] == control.VERSION
    assert meta['model']['kind'] == kind
    assert meta['geometry_samples'] == meta['model_advance_samples'] == 12
    assert meta['configuration']['learned_head_work_retained'] is True


@pytest.mark.parametrize('kind', frozen.KINDS)
def test_existing_two_classes_match_frozen_geometry_exactly_without_changing_gate(kind):
    cfg, student, draws = plan(), model(kind), innovations()
    state, old_kinds = root(student), frozen.KINDS
    old = frozen.score_search(cfg, student, state, draws, 0, score_mode='geometry')
    new = control.score_search(cfg, student, state, draws, 0)
    for field in ('sequences', 'scores', 'selected_actions', 'selected_ids'):
        np.testing.assert_array_equal(getattr(new[0], field), getattr(old[0], field))
    for key in old[5]['arrays']:
        np.testing.assert_array_equal(new[5]['arrays'][key], old[5]['arrays'][key])
    assert old[2] == new[2] == [64] * 4
    assert old[4]['model_work'] == new[4]['model_work']
    assert frozen.KINDS == old_kinds == ('residual_gru', 'cached_mlp')
    with pytest.raises(ValueError, match='Only original'):
        frozen.score_search(cfg, model('cached_gru'), root(model('cached_gru')), draws, 0, score_mode='geometry')


@pytest.mark.parametrize('kind', control.KINDS)
def test_missing_public_packet_preserves_actual_assimilation_semantics(kind):
    student = model(kind)
    initial = root(student)
    with torch.no_grad():
        carried, _, _ = student.advance(initial, torch.tensor([[.1, -.1]]))
        missing = packet(valid=False, age=.02)
        missing[:, :4] = 9876.  # Finite unavailable placeholders must be sanitized.
        observed = student.assimilate(carried, missing)
        clean = missing.clone()
        clean[:, :4] = 0
        identical = student.assimilate(carried, clean)
        assert canonical_tensor_hash(observed) == canonical_tensor_hash(identical)
        assert torch.equal(observed['packet'], clean)
        if kind == 'residual_gru':
            assert torch.equal(observed['hidden'], carried['hidden'])
        elif kind in ('encoded_current_gru', 'cached_gru'):
            features = clean.clone()
            if kind == 'cached_gru':
                features[:, :4] = initial['cached_angles']
                assert torch.equal(observed['cached_angles'], initial['cached_angles'])
            assert torch.equal(observed['hidden'], student.observation_update(features, torch.zeros_like(observed['hidden'])))
        else:
            assert torch.equal(observed['encoder_features'][:, :4], initial['cached_angles'])
            assert torch.equal(observed['encoder_features'][:, 4:], clean[:, 4:])
        if kind != 'residual_gru':
            assert observed['real_index'].item() == 1 and observed['imagined_depth'].item() == 0
            twice, _, _ = student.advance(carried, torch.zeros(1, 2))
            with pytest.raises(ValueError, match='exactly one'):
                student.assimilate(twice, clean)
        control.score_bank(plan(), student, observed, np.zeros((1, 2, 2, 2), np.float32))


@pytest.mark.parametrize('kind', control.KINDS)
def test_cem_uses_exact_paid_budget_and_terminal_boundary(kind):
    cfg, student = plan(), model(kind)
    cfg.update(engineering=False, planning_horizon=12, action_block=3)
    state = root(student)
    original = canonical_tensor_hash(state)
    result, raw, sizes, _, work, diagnostic = control.score_search(cfg, student, state, innovations(12, 3), 47)
    assert result.horizon == 3 and sizes == [64] * 4
    assert raw.shape == (1, 256, 3)
    assert work['geometry_samples'] == work['imagined_transitions'] == 256 * 3
    assert len(diagnostic['metadata']['callbacks']) == 4
    assert canonical_tensor_hash(state) == original
    with pytest.raises(ValueError, match='at or beyond'):
        control.score_search(cfg, student, state, innovations(12, 3), 50)


@pytest.mark.parametrize('kind', control.KINDS)
def test_complete_engineering_row_replays_native_and_selected_advance_from_each_real_root(kind, tmp_path):
    cfg, student, draws = plan(), model(kind), innovations()
    initial_hash = canonical_tensor_hash(student.state_dict())
    out = tmp_path / kind
    timing = control.learned_control(cfg, student, 'ordinary', [draws] * 50, out, cases=cases())
    record = records(out / 'episodes')[0]
    assert native_replay(record)['transitions'] == 50
    assert timing['observation_assimilations'] == timing['executed_action_advances'] == 50
    assert timing['row_wall_seconds'] >= timing['setup_seconds'] + sum(timing['decision_seconds']) + sum(timing['native_step_seconds'])
    work = json.loads((out / 'state-work.json').read_text())
    assert work['model']['kind'] == kind
    assert work['aggregate_model_work']['assimilate_samples'] == 50
    assert work['aggregate_model_work']['advance_samples'] == 50 + 256 * (49 * 2 + 1)
    with np.load(out / 'states.npz', allow_pickle=False) as states, np.load(out / 'executed_predictions.npz', allow_pickle=False) as executed:
        np.testing.assert_array_equal(states['root__packet'][0], record['policy']['packets'][:-1])
        for t in range(50):
            saved_root = {key.removeprefix('root__'): torch.from_numpy(states[key][:, t].copy())
                          for key in states.files if key.startswith('root__')}
            with torch.no_grad():
                carried, angles, reward = student.advance(saved_root, torch.from_numpy(record['policy']['commands'][None, t].copy()))
                prior = student.initial(1) if t == 0 else {
                    key.removeprefix('carried__'): torch.from_numpy(states[key][:, t - 1].copy())
                    for key in states.files if key.startswith('carried__')}
                actual_root = student.assimilate(prior, torch.from_numpy(record['policy']['packets'][None, t].copy()))
            assert canonical_tensor_hash(actual_root) == canonical_tensor_hash(saved_root)
            for key, tensor in carried.items():
                np.testing.assert_array_equal(tensor.numpy(), states['carried__' + key][:, t])
            np.testing.assert_array_equal(angles.numpy(), executed['angles'][:, t])
            np.testing.assert_array_equal(reward.numpy(), executed['rewards'][:, t])
        for t in (0, 8, 13, 14, 49):
            with np.load(out / 'scoring' / f'{t:03d}.npz', allow_pickle=False) as score, np.load(out / 'decisions' / f'{t:03d}.npz', allow_pickle=False) as trace:
                np.testing.assert_array_equal(score['selected_rewards'], trace['raw_rewards'])
                np.testing.assert_array_equal(score['selected_angles'], executed['angles'][:, t])
                np.testing.assert_array_equal(score['selected_learned_reward'], executed['rewards'][:, t])
                np.testing.assert_array_equal(score['commands'][np.arange(1), trace['selected_ids'], 0], record['policy']['commands'][None, t])
        if kind != 'residual_gru':
            assert (states['root__imagined_depth'] == 0).all() and (states['carried__imagined_depth'] == 1).all()
            if kind.startswith('cached'):
                np.testing.assert_array_equal(states['root__cached_angles'], states['carried__cached_angles'])
    assert canonical_tensor_hash(student.state_dict()) == initial_hash


@pytest.mark.parametrize('kind', control.KINDS)
def test_interrupted_candidate_prefix_preserved_and_original_baseexception_raised(kind, tmp_path, monkeypatch):
    cfg, student = plan(), model(kind)
    state = root(student)
    before = canonical_tensor_hash(state)
    advance, calls = student.advance, 0
    sentinel = KeyboardInterrupt('synthetic interrupted second prediction')
    def stopped(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sentinel
        return advance(*args)
    monkeypatch.setattr(student, 'advance', stopped)
    out = tmp_path / 'partial'
    with pytest.raises(KeyboardInterrupt) as error:
        control.score_bank(cfg, student, state, np.zeros((1, 3, 2, 2), np.float32), failure_out=out)
    assert error.value is sentinel and canonical_tensor_hash(state) == before
    receipt = json.loads((out / 'failed.json').read_text())
    assert receipt['banks'][0]['completed_scored_offsets'] == 1
    assert receipt['banks'][0]['model_advance_attempted_samples'] == 6
    with np.load(out / 'bank-000.npz', allow_pickle=False) as saved:
        assert saved['predicted_angles'].shape == (1, 3, 1, 4)
        np.testing.assert_array_equal(saved['root__packet'], state['packet'].numpy())


def test_selected_failure_keeps_full_search_and_carried_state(tmp_path, monkeypatch):
    cfg, student, draws = plan(), model('cached_gru'), innovations()
    selected = control._selected_prediction
    sentinel = RuntimeError('synthetic selected boundary failure')
    def stopped(*args):
        selected(*args)
        raise sentinel
    monkeypatch.setattr(control, '_selected_prediction', stopped)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError) as error:
        control.learned_control(cfg, student, 'ordinary', [draws] * 50, out, cases=cases())
    assert error.value is sentinel
    assert json.loads((out / 'failed.json').read_text())['completed_steps_by_case'] == [0]
    assert (out / 'partial-root.npz').exists() and (out / 'partial-carried.npz').exists()
    with np.load(out / 'partial-active-scoring.npz', allow_pickle=False) as saved:
        assert saved['predicted_angles'].shape == (1, 256, 2, 4)
        assert saved['selected_angles'].shape == (1, 4)


@pytest.mark.parametrize('kind', control.KINDS)
def test_invalid_angle_predictions_fail_with_evidence_and_no_fallback(kind, tmp_path):
    student = model(kind)
    with torch.no_grad():
        student.observation_head[-1].weight.zero_()
        student.observation_head[-1].bias.zero_()
    out = tmp_path / 'failed'
    with pytest.raises(ValueError, match='norms'):
        control.score_bank(plan(), student, root(student), np.zeros((1, 2, 2, 2), np.float32), failure_out=out)
    row = json.loads((out / 'failed.json').read_text())['banks'][0]
    assert row['completed_model_offsets'] == 1 and row['completed_scored_offsets'] == 0
    assert row['geometry_attempted_samples'] == 2 and row['geometry_samples'] == 0


@pytest.mark.parametrize('invalid', ['wrong_class', 'subclass', 'training', 'mode', 'horizon', 'dtype', 'action', 'case_count'])
def test_strict_identity_configuration_and_inputs(invalid):
    cfg, student = plan(), model('residual_gru')
    bank = np.zeros((1, 2, 2, 2), np.float32)
    if invalid == 'wrong_class':
        student = model('packet_mlp')
    elif invalid == 'subclass':
        class Pretender(REGISTRY['residual_gru']):
            pass
        student = Pretender(hidden_size=4, noise_std=.05).eval()
    elif invalid == 'training':
        student.train()
    elif invalid == 'mode':
        cfg['score_modes'] = ['learned', 'geometry']
    elif invalid == 'horizon':
        cfg['engineering'] = False
    elif invalid == 'dtype':
        bank = bank.astype(np.float64)
    elif invalid == 'action':
        bank[0, 0, 0, 0] = 1.01
    else:
        bank = np.zeros((2, 2, 2, 2), np.float32)
    with pytest.raises(ValueError):
        control.score_bank(cfg, student, root(student), bank)
