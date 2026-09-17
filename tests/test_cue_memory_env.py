import numpy as np
import pytest

pytest.importorskip('minigrid', reason='Install minigrid==3.0.0 for the cue-visible memory study')

from minigrid.core.world_object import Ball, Key

from openjev.research.cue_memory_env import CueMemoryBatch, make_cue_env
from openjev.research.memory_env import OBS_SIZE, encode_obs, make_env


@pytest.mark.parametrize('size', [11, 17, 23])
def test_native_map_rng_and_task_unchanged_except_start_position(size):
    combinations = set()
    for seed in range(24):
        native, intervention = make_env(size, seed), make_cue_env(size, seed)
        try:
            native_obs, native_info = native.reset(seed=seed)
            obs, info = intervention.reset(seed=seed)
            np.testing.assert_array_equal(native.grid.encode(), intervention.grid.encode())
            assert intervention.np_random.bit_generator.state == native.np_random.bit_generator.state
            assert intervention.success_pos == native.success_pos
            assert intervention.failure_pos == native.failure_pos
            assert intervention.mission == native.mission == obs['mission'] == native_obs['mission']
            assert info == native_info
            assert intervention.step_count == native.step_count == 0
            assert intervention.carrying is native.carrying is None
            assert intervention.max_steps == native.max_steps == 128
            assert intervention.action_space.n == native.action_space.n == 7
            assert intervention.observation_space == native.observation_space
            assert intervention.agent_view_size == native.agent_view_size == 7
            assert tuple(intervention.agent_pos) == (1, size//2)
            assert intervention.agent_dir == native.agent_dir == 0
            assert encode_obs(obs).shape == (OBS_SIZE,)
            # Actual native visibility, including wall occlusion, verifies the intervention.
            assert intervention.agent_sees(1, size//2-1)
            assert not intervention.agent_sees(size-2, size//2-2)
            assert not intervention.agent_sees(size-2, size//2+2)
            combinations.add((intervention.grid.get(1, size//2-1).type, intervention.success_pos))
        finally:
            native.close()
            intervention.close()
    assert len(combinations) == 4  # Both cue identities and both branch mappings use the same pose.


@pytest.mark.parametrize('size', [11, 17, 23])
def test_initial_observation_hides_branch_mapping_and_fork_hides_cue(size):
    env = make_cue_env(size, 19)
    try:
        initial = encode_obs(env.gen_obs())
        upper = env.grid.get(size-2, size//2-2)
        lower = env.grid.get(size-2, size//2+2)
        env.grid.set(size-2, size//2-2, lower)
        env.grid.set(size-2, size//2+2, upper)
        np.testing.assert_array_equal(initial, encode_obs(env.gen_obs()))
        env.agent_pos = np.array((size-2, size//2))
        for direction in range(4):
            env.agent_dir = direction
            env.grid.set(1, size//2-1, Key('green'))
            key_obs = encode_obs(env.gen_obs())
            env.grid.set(1, size//2-1, Ball('green'))
            np.testing.assert_array_equal(key_obs, encode_obs(env.gen_obs()))
    finally:
        env.close()


def test_every_explicit_and_automatic_reset_presents_the_cue_without_advancing_time():
    with CueMemoryBatch(3, 11, 201, max_steps=2) as batch:
        batch.reset()
        for episode in range(4):
            for i, env in enumerate(batch.envs):
                assert tuple(env.agent_pos) == (1, 5) and env.agent_dir == 0
                assert env.step_count == 0 and env.agent_sees(1, 4)
                assert batch.episode_seeds[i] == 201+i+episode*3
            _, _, terminal, timeout, _ = batch.step([0, 0, 0])
            assert not terminal.any() and not timeout.any()
            _, _, terminal, timeout, infos = batch.step([0, 0, 0])
            assert not terminal.any() and timeout.all()
            for info in infos:
                assert info['episode']['length'] == 2
                assert info['episode']['return'] == 0
                assert not info['episode']['success']
        # An explicit reset advances the same per-slot stream and uses the same pose.
        batch.reset()
        assert batch.episode_seeds.tolist() == [216, 217, 218]
        assert all(env.agent_sees(1, 4) and env.step_count == 0 for env in batch.envs)


def test_timeout_terminal_observation_is_not_replaced_by_cue_visible_reset():
    with CueMemoryBatch(1, 11, 21, max_steps=2) as batch:
        batch.reset()
        native = make_cue_env(11, 21, max_steps=2)
        try:
            native.step(0)
            terminal_obs, _, _, native_timeout, _ = native.step(0)
        finally:
            native.close()
        batch.step([0])
        next_obs, _, terminal, timeout, infos = batch.step([0])
        assert native_timeout and timeout[0] and not terminal[0]
        np.testing.assert_array_equal(infos[0]['terminal_observation'], encode_obs(terminal_obs))
        assert not np.array_equal(next_obs[0], infos[0]['terminal_observation'])
        assert batch.envs[0].agent_dir == 0 and batch.envs[0].step_count == 0
        assert batch.envs[0].agent_sees(1, 4)


@pytest.mark.parametrize('correct', [False, True])
def test_native_goal_rewards_and_terminal_capture_survive_intervention(correct):
    with CueMemoryBatch(1, 11, 41) as batch:
        batch.reset()
        native = make_cue_env(11, 41)
        try:
            for env in (native, batch.envs[0]):
                target = env.success_pos if correct else env.failure_pos
                env.agent_pos = np.array((9, 5))
                env.agent_dir = 3 if target[1] < 5 else 1
            terminal_obs, reward, expected_terminal, expected_timeout, _ = native.step(2)
        finally:
            native.close()
        next_obs, rewards, terminal, timeout, infos = batch.step([2])
        assert expected_terminal and terminal[0] and not expected_timeout and not timeout[0]
        assert rewards[0] == pytest.approx(reward)
        assert reward == pytest.approx(1-0.9/128 if correct else 0)
        assert infos[0]['episode']['success'] == correct
        np.testing.assert_array_equal(infos[0]['terminal_observation'], encode_obs(terminal_obs))
        assert not np.array_equal(next_obs[0], infos[0]['terminal_observation'])
        assert tuple(batch.envs[0].agent_pos) == (1, 5)
        assert batch.envs[0].step_count == 0 and batch.envs[0].agent_sees(1, 4)


@pytest.mark.parametrize('size', [7, 9, 10])
def test_small_or_even_grids_are_rejected(size):
    with pytest.raises(ValueError, match='at least eleven'):
        make_cue_env(size, 21)
    with pytest.raises(ValueError, match='at least eleven'):
        CueMemoryBatch(1, size, 21)
