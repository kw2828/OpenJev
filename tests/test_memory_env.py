import numpy as np
import pytest

pytest.importorskip("minigrid", reason="Install minigrid==3.0.0 for the memory study")

from minigrid.core.world_object import Ball, Key

from openjev.research.memory_env import ACTION_COUNT, OBS_SIZE, MemoryBatch, encode_obs, make_env


def test_encoding_only_uses_partial_image_and_direction():
    env = make_env(7, 19)
    try:
        obs, _ = env.reset(seed=19)
        encoded = encode_obs(obs)
        assert encoded.shape == (OBS_SIZE,) == (984,)
        assert encoded.dtype == np.float32
        assert encoded.sum() == 49 * 3 + 1
        changed = {**obs, "mission": "unrelated", "hidden_target": "secret", "seed": 10000}
        np.testing.assert_array_equal(encoded, encode_obs(changed))
        assert env.action_space.n == ACTION_COUNT == 7
    finally:
        env.close()


def test_alternative_view_encoding_is_explicit():
    env = make_env(9, 19, view_size=3)
    try:
        obs, _ = env.reset(seed=19)
        assert encode_obs(obs).shape == (184,)
    finally:
        env.close()


@pytest.mark.parametrize("size", [11, 17, 23])
def test_cue_is_unobservable_at_the_fork_for_all_orientations(size):
    env = make_env(size, 19)
    try:
        env.agent_pos = np.array((size - 2, size // 2))
        for direction in range(4):
            env.agent_dir = direction
            env.grid.set(1, size // 2 - 1, Key("green"))
            key_obs = encode_obs(env.gen_obs())
            env.grid.set(1, size // 2 - 1, Ball("green"))
            np.testing.assert_array_equal(key_obs, encode_obs(env.gen_obs()))
    finally:
        env.close()


def test_small_grid_has_reactive_cue_shortcut_with_native_view():
    env = make_env(7, 19)
    try:
        env.agent_pos = np.array((5, 3))
        env.agent_dir = 2
        env.grid.set(1, 2, Key("green"))
        key_obs = encode_obs(env.gen_obs())
        env.grid.set(1, 2, Ball("green"))
        assert not np.array_equal(key_obs, encode_obs(env.gen_obs()))
    finally:
        env.close()


def test_timeout_preserves_terminal_observation_before_autoreset():
    with MemoryBatch(1, 7, 21, max_steps=1) as batch:
        first = batch.reset()
        native = make_env(7, 21, max_steps=1)
        try:
            terminal_obs, native_reward, native_term, native_trunc, _ = native.step(0)
        finally:
            native.close()
        next_obs, rewards, terminated, truncated, infos = batch.step([0])
        assert not native_term and native_trunc
        assert not terminated[0] and truncated[0]
        assert rewards[0] == native_reward == 0
        np.testing.assert_array_equal(infos[0]["terminal_observation"], encode_obs(terminal_obs))
        assert not np.array_equal(infos[0]["terminal_observation"], next_obs[0])
        assert first.shape == next_obs.shape
        assert infos[0]["episode"] == {"return": 0.0, "success": False, "length": 1, "seed": 21}
        assert batch.episode_seeds[0] == 22
        assert batch.returns[0] == batch.lengths[0] == 0


@pytest.mark.parametrize("correct", [False, True])
def test_native_success_and_failure_terminate_without_timeout(correct):
    with MemoryBatch(1, 7, 71) as batch:
        batch.reset()
        env = batch.envs[0]
        target = env.success_pos if correct else env.failure_pos
        # Test-only positioning isolates the real environment's terminal rule.
        env.agent_pos = np.array((5, 3))
        env.agent_dir = 3 if target[1] < 3 else 1
        _, rewards, terminated, truncated, infos = batch.step([2])
        assert terminated[0] and not truncated[0]
        assert infos[0]["episode"]["success"] == correct
        expected = 1 - 0.9 / 128 if correct else 0.0
        assert rewards[0] == pytest.approx(expected)
        assert infos[0]["episode"]["return"] == pytest.approx(expected)
        assert infos[0]["episode"]["length"] == 1


def test_seed_streams_and_controls_are_deterministic_and_nonoverlapping():
    with MemoryBatch(3, 7, 110, max_steps=2) as left, MemoryBatch(3, 7, 110, max_steps=2) as right:
        np.testing.assert_array_equal(left.reset(), right.reset())
        completed_seeds = []
        for step in range(8):
            actions = np.array([step % 2, (step + 1) % 2, 6])
            a, b = left.step(actions), right.step(actions)
            for first, second in zip(a[:4], b[:4], strict=True):
                np.testing.assert_array_equal(first, second)
            for first, second in zip(a[4], b[4], strict=True):
                if "episode" in first:
                    assert first["episode"] == second["episode"]
                    completed_seeds.append(first["episode"]["seed"])
                    np.testing.assert_array_equal(first["terminal_observation"], second["terminal_observation"])
        assert completed_seeds == list(range(110, 122))
        left.reset()
        assert left.episode_seeds.tolist() == [125, 126, 127]


def test_reset_required_and_invalid_actions_do_not_step_environment():
    with MemoryBatch(1, 7, 4) as batch:
        with pytest.raises(RuntimeError, match="reset"):
            batch.step([0])
        batch.reset()
        for invalid in ([0.5], [7], [-1], [0, 1]):
            with pytest.raises(ValueError):
                batch.step(invalid)
        assert batch.envs[0].step_count == 0
