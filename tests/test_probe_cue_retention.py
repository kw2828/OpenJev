import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('minigrid', reason='Install minigrid==3.0.0 for the memory retention probe')

from openjev.research.predictive_memory import PredictiveMemory

spec = importlib.util.spec_from_file_location(
    '_retention_probe_test', Path(__file__).resolve().parents[1]/'scripts/probe_cue_retention.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize('size', [11, 17, 23])
def test_swap_changes_only_cue_with_identical_native_goals_and_rng(size):
    envs = probe.paired_environments(size, [21, 22])
    try:
        for original, swapped in zip(envs[::2], envs[1::2], strict=True):
            changed = np.argwhere(original.grid.encode() != swapped.grid.encode())
            assert changed.tolist() == [[1, size//2-1, 0]]
            assert original.success_pos == swapped.success_pos
            assert original.failure_pos == swapped.failure_pos
            assert original.np_random.bit_generator.state == swapped.np_random.bit_generator.state
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize('size', [11, 17, 23])
def test_forced_path_hides_cue_before_fork_and_current_state_difference_is_exactly_zero(size):
    torch.manual_seed(713)
    model = PredictiveMemory(hidden=8).eval()
    result = probe.probe_size(model, 'current_ppo', size, seeds=[21, 22])
    assert result['forced_actions_per_pair_member'] == [2]*(size-3)
    assert len(result['per_step']) == size-2
    assert result['last_cue_visible_steps'] == [0, 0]
    assert result['initial']['hidden_l2_distance'] > 0
    assert result['current_only_equality_check_passed']
    for row in result['per_step'][1:]:
        assert row['observations_identical'] == 1
        assert row['cue_visible_fraction'] == 0
        assert row['hidden_l2_distance'] == 0
        assert row['policy_probability_l1_distance'] == 0
        assert row['argmax_disagreement'] == 0
    assert result['per_pair'][0][-1]['x'] == size-2


def test_policy_preferences_cannot_change_forced_observations_or_actions():
    torch.manual_seed(193)
    model = PredictiveMemory(hidden=8).eval()
    first = probe.probe_size(model, 'recurrent_ppo', 11, seeds=[21, 22])
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.bias[6] = 100  # Strongly prefer no-op, but the probe still imposes forward.
    second = probe.probe_size(model, 'recurrent_ppo', 11, seeds=[21, 22])
    assert first['paired_observations_sha256'] == second['paired_observations_sha256']
    assert second['forced_actions_per_pair_member'] == [2]*8
    assert all(row['native_argmax'] == row['swapped_argmax'] == 6
               for pair in second['per_pair'] for row in pair)
    assert first['fork']['hidden_l2_distance'] == second['fork']['hidden_l2_distance']
    assert first['all_post_initial_observations_identical']
