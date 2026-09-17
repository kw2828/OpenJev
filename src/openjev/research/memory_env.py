"""Native MiniGrid Memory with a small, explicitly partial observation.

Only the observation's image and direction enter the model input. The image is
one-hot encoded per cell in object/color/state order (11/6/3 categories), then
flattened and followed by the four direction indicators. The native default
7x7 view has 984 features. Mission text is constant and omitted.

``MemoryBatch.reset()`` returns a float32 ``[count, observation_size]`` array.
``step(actions)`` returns ``(observations, rewards, terminated, truncated, infos)``.
The arrays have leading dimension ``count`` and infos is a list of dictionaries.
Completed slots automatically reset: their returned observation starts the next
episode, while ``info['terminal_observation']`` preserves the encoded final
observation of the completed episode. ``info['episode']`` contains its native
reward sum (``return``), positive-reward ``success``, ``length`` and ``seed``.
Callers must reset recurrent state on *either* termination or truncation and use
the terminal observation for any timeout bootstrap or dynamics target.

Slot i, episode j uses seed ``seed_start + i + j * count``. Explicit batch resets
also advance this stream, so seeds are never silently reused within a batch.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from minigrid.envs import MemoryEnv

ACTION_COUNT = 7
VIEW_SIZE = 7
OBS_SIZE = VIEW_SIZE * VIEW_SIZE * 20 + 4
_ONE_HOT = tuple(np.eye(size, dtype=np.float32) for size in (11, 6, 3))


def make_env(size: int, seed: int, max_steps: int = 128, *, view_size: int = VIEW_SIZE) -> MemoryEnv:
    """Create and seed the unmodified Farama task, with no reward shaping.

    Its native starting position is random along the corridor. The agent may
    need to navigate back to inspect the cue; we do not teleport it to the cue.
    All seven native action IDs are retained (including the task's no-op IDs).
    """
    if size < 7 or size % 2 != 1:
        raise ValueError("Memory requires an odd grid size of at least seven")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if view_size < 3 or view_size % 2 != 1:
        raise ValueError("view_size must be odd and at least three")
    env = MemoryEnv(size=size, random_length=False, max_steps=max_steps, agent_view_size=view_size)
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    return env


def encode_obs(obs: Mapping) -> np.ndarray:
    """Encode the supplied partial image and direction, ignoring other fields."""
    image = np.asarray(obs["image"])
    if image.ndim != 3 or image.shape[0] != image.shape[1] or image.shape[2] != 3:
        raise ValueError("Expected a square MiniGrid image with three channels")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError("MiniGrid image channels must contain integer category IDs")
    channels = []
    for channel, encoding in enumerate(_ONE_HOT):
        index = image[..., channel]
        if np.any(index < 0) or np.any(index >= len(encoding)):
            raise ValueError(f"Invalid MiniGrid category in channel {channel}")
        channels.append(encoding[index])
    direction = obs["direction"]
    if not isinstance(direction, (int, np.integer)) or not 0 <= direction < 4:
        raise ValueError("Direction must be an integer from zero to three")
    direction_features = np.zeros(4, dtype=np.float32)
    direction_features[direction] = 1.0
    return np.concatenate((np.concatenate(channels, axis=-1).reshape(-1), direction_features))


class MemoryBatch:
    """Synchronous auto-reset batch with native rewards and distinct done flags."""

    def __init__(self, count: int, size: int, seed_start: int, max_steps: int = 128,
                 *, view_size: int = VIEW_SIZE):
        if count < 1:
            raise ValueError("count must be positive")
        self.count = count
        self.seed_start = int(seed_start)
        self.observation_size = view_size * view_size * 20 + 4
        self.envs = [make_env(size, self.seed_start + i, max_steps, view_size=view_size)
                     for i in range(count)]
        self.episode_indices = np.full(count, -1, dtype=np.int64)
        self.episode_seeds = np.zeros(count, dtype=np.int64)
        self.returns = np.zeros(count, dtype=np.float64)
        self.lengths = np.zeros(count, dtype=np.int64)
        self._ready = False

    def _reset_slot(self, index: int) -> np.ndarray:
        self.episode_indices[index] += 1
        seed = self.seed_start + index + int(self.episode_indices[index]) * self.count
        self.episode_seeds[index] = seed
        self.returns[index] = 0.0
        self.lengths[index] = 0
        observation, _ = self.envs[index].reset(seed=seed)
        return encode_obs(observation)

    def reset(self) -> np.ndarray:
        observations = np.stack([self._reset_slot(i) for i in range(self.count)])
        self._ready = True
        return observations

    def step(self, actions: Sequence[int] | np.ndarray):
        if not self._ready:
            raise RuntimeError("Call reset before step")
        actions = np.asarray(actions)
        if actions.shape != (self.count,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Expected one integer action per environment")
        if np.any(actions < 0) or np.any(actions >= ACTION_COUNT):
            raise ValueError("Action IDs must be between zero and six")
        observations = np.empty((self.count, self.observation_size), dtype=np.float32)
        rewards = np.empty(self.count, dtype=np.float32)
        terminated = np.empty(self.count, dtype=bool)
        truncated = np.empty(self.count, dtype=bool)
        infos = []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            obs, reward, terminal, timeout, native_info = env.step(int(action))
            encoded = encode_obs(obs)
            info = dict(native_info)
            self.returns[i] += reward
            self.lengths[i] += 1
            rewards[i], terminated[i], truncated[i] = reward, terminal, timeout
            if terminal or timeout:
                info["terminal_observation"] = encoded.copy()
                info["episode"] = {"return": float(self.returns[i]), "success": bool(self.returns[i] > 0),
                                   "length": int(self.lengths[i]), "seed": int(self.episode_seeds[i])}
                encoded = self._reset_slot(i)
            observations[i] = encoded
            infos.append(info)
        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._ready = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
