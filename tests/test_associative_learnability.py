import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('minigrid', reason='Install minigrid==3.0.0 for the learnability diagnostic')

from openjev.research.associative_policy import MODES, AssociativePolicy
from openjev.research.cue_memory_env import make_cue_env

spec = importlib.util.spec_from_file_location(
    '_associative_learnability_test', Path(__file__).resolve().parents[1]/'scripts/associative_learnability.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_training_only_search_finds_four_unique_balanced_contexts():
    data, metadata = study.training_packet()
    assert data['observations'].shape == (9, 4, 984)
    assert {(row['cue'], row['upper_branch']) for row in metadata['contexts']} == set(study.CONTEXTS)
    assert metadata['unique_observation_sequences'] == 4
    assert metadata['target_counts'] == {'0': 2, '1': 2}
    assert all(row['seed'] >= 70_000_000 for row in metadata['contexts'])
    assert np.all(data['previous'][0] == 0) and np.all(data['previous'][1:] == 2)
    assert data['resets'][0].all() and not data['resets'][1:].any()


def test_observation_teacher_agrees_with_independent_native_goal_oracle():
    _, metadata = study.training_packet()
    for row in metadata['contexts']:
        env = make_cue_env(11, row['seed'])
        try:
            initial = env.gen_obs()
            for _ in range(8):
                fork, _, terminated, truncated, _ = env.step(2)
                assert not terminated and not truncated
            label, _ = study.teacher_choice(initial, fork)
            oracle = 0 if env.success_pos[1] < 11//2 else 1
            assert label == oracle == row['target_turn']
            # Hidden metadata never reaches the observation-based teacher.
            changed_initial = {**initial, 'success_pos': (-99, -99), 'answer': 1-label, 'mission': 'ignore'}
            changed_fork = {**fork, 'success_pos': (99, 99), 'answer': 1-label, 'mission': 'ignore'}
            assert study.teacher_choice(changed_initial, changed_fork)[0] == label
        finally:
            env.close()


def test_opposite_cues_share_every_later_observation_and_require_opposite_turns():
    data, metadata = study.training_packet()
    for upper in (study.KEY, study.BALL):
        indices = [i for i, row in enumerate(metadata['contexts']) if row['upper_branch'] == upper]
        a, b = indices
        assert data['labels'][a] != data['labels'][b]
        assert not np.array_equal(data['observations'][0, a], data['observations'][0, b])
        np.testing.assert_array_equal(data['observations'][1:, a], data['observations'][1:, b])
        np.testing.assert_array_equal(data['previous'][:, a], data['previous'][:, b])


@pytest.mark.parametrize('mode', MODES)
def test_label_changes_cannot_change_policy_inputs_or_logits_and_future_cannot_change_prefix(mode):
    data, _ = study.training_packet()
    torch.manual_seed(713)
    model = AssociativePolicy(mode, hidden=8, store_dim=4, adapter_hidden=5)
    original = study.final_logits(model, data)
    changed = {**data, 'labels': 1-data['labels']}
    torch.testing.assert_close(study.final_logits(model, changed), original, rtol=0, atol=0)
    obs = torch.from_numpy(data['observations'])
    previous = torch.from_numpy(data['previous'])
    reset = torch.from_numpy(data['resets'])
    full = model.sequence(obs, previous, model.initial_state(4), reset)[0]
    altered = obs.clone()
    altered[4:] = torch.randn_like(altered[4:])
    replay = model.sequence(altered, previous, model.initial_state(4), reset)[0]
    torch.testing.assert_close(full[:4], replay[:4], rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_state_erasure_cannot_solve_balanced_opposite_cues(mode):
    data, _ = study.training_packet()
    torch.manual_seed(971)
    model = AssociativePolicy(mode, hidden=8, store_dim=4, adapter_hidden=5)
    logits = study.final_logits(model, data, 'reset_all')
    # Canonical order is key/key, key/ball, ball/key, ball/ball.
    torch.testing.assert_close(logits[0], logits[2], rtol=0, atol=0)
    torch.testing.assert_close(logits[1], logits[3], rtol=0, atol=0)
    assert float((logits.argmax(-1) == torch.from_numpy(data['labels'])).float().mean()) <= .5


def test_nonfast_store_erasure_is_a_noop():
    data, _ = study.training_packet()
    for mode in ('gru', 'feedforward'):
        torch.manual_seed(197)
        model = AssociativePolicy(mode, hidden=8, store_dim=4, adapter_hidden=5)
        torch.testing.assert_close(study.final_logits(model, data, 'intact'),
                                   study.final_logits(model, data, 'reset_store'), rtol=0, atol=0)
