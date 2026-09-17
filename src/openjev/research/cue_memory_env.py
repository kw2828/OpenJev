"""Cue-visible reset intervention for the native MiniGrid Memory task.

Native map generation runs unchanged, including its original random starting-x
draw. Afterwards only the agent pose is pinned to (x=1, y=height//2), facing east
(direction=0, already the native orientation). This is a legitimate native start
pose independent of cue identity and branch mapping. Consequently a given seed
preserves the exact native map, cue, branch assignment and RNG state.

The native seven actions, 7x7 partial observation, rewards, reward denominator,
termination rules and step counter are unchanged. Sizes below eleven are rejected
because the cue-visible start must not expose the distant branch choices. This
module is an explicit task intervention, not the original random-start benchmark.
"""

import numpy as np
from minigrid.envs import MemoryEnv

from openjev.research.memory_env import MemoryBatch


class CueVisibleMemoryEnv(MemoryEnv):
    """Apply a constant legitimate starting pose after every native reset."""

    def __init__(self, size: int, max_steps: int = 128):
        if size < 11 or size % 2 != 1:
            raise ValueError('Cue-visible Memory requires an odd size of at least eleven')
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        super().__init__(size=size, random_length=False, max_steps=max_steps, agent_view_size=7)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        # Do not condition the pose on cue type, target branch or any other draw.
        self.agent_pos = np.array((1, self.height // 2))
        self.agent_dir = 0
        return self.gen_obs(), info


def make_cue_env(size: int, seed: int, max_steps: int = 128) -> CueVisibleMemoryEnv:
    """Create the native task with only the documented reset-position change."""
    env = CueVisibleMemoryEnv(size, max_steps)
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    return env


class CueMemoryBatch(MemoryBatch):
    """MemoryBatch's exact transition, receipt and auto-reset behavior, cue-visible.

    The base constructor initializes its counters and unique per-slot seed stream.
    Replace its freshly created environments before the first public batch reset.
    Each replacement uses native reset logic with the pose intervention, so the
    inherited automatic resets apply it too. No frozen base source is modified.
    """

    def __init__(self, count: int, size: int, seed_start: int, max_steps: int = 128):
        if size < 11 or size % 2 != 1:
            raise ValueError('Cue-visible Memory requires an odd size of at least eleven')
        super().__init__(count, size, seed_start, max_steps)
        for env in self.envs:
            env.close()
        self.envs = [make_cue_env(size, self.seed_start+i, max_steps) for i in range(count)]
