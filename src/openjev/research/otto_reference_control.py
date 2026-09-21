"""Public-only space-aware infotaxis under a declared four-action contract.

The scalar objective and its arithmetic follow the pinned OTTO HeuristicPolicy
at a6aaef6507cffd2aff79291c1019f506f616bbef. The default additionally evaluates
blocked directions as stay-and-observe actions, matching released RLPolicy's
action space. ``allow_stay=False`` retains the heuristic's in-bounds selection.
Both variants use the already-qualified public belief and lifecycle unchanged.
No simulator, trained model, source location, random seed or history is accepted.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from openjev.research.otto_released_policy import EPSILON, ReleasedPolicyActor

VERSION = "otto-public-space-aware-reference-v1"


def _entropy(probability):
    # Preserve upstream's zero contribution at and below EPSILON, including
    # subnormalized branches. This is not an unconditional entropy repair.
    logs = np.zeros(probability.shape)
    keep = probability > EPSILON
    logs[keep] = -np.log2(probability[keep])
    return np.sum(probability * logs)


class _SpaceAwarePolicy:
    __slots__ = ("_allow_stay", "distance_array", "env")

    def __init__(self, *, env, model, sym_avg, allow_stay):
        if model is not None or sym_avg is not False:
            raise ValueError("analytic control accepts no model or symmetry ensemble")
        self.env, self._allow_stay = env, allow_stay
        grid = np.indices((107, 107), dtype=np.float64)
        distance = np.zeros((107, 107), dtype=np.float64)
        for axis in range(2):
            distance += np.abs(grid[axis] - 53)
        self.distance_array = np.frombuffer(distance.tobytes(), dtype=np.float64).reshape(107, 107)

    def _value_policy(self):
        # env is the owned PublicBeliefView supplied by ReleasedPolicyActor,
        # never the simulator. Its returned belief is already an owned copy.
        probability, position = self.env.p_source, self.env.agent
        scores = np.full(4, np.inf, dtype=np.float64)
        for action in range(4):
            target, possible = self.env._move(action, position)
            if not possible and not self._allow_stay:
                continue
            distance = self.env._extract_N_from_2N(self.distance_array, target)
            surviving = probability.copy()
            end_mass = surviving[tuple(target)]
            if end_mass > 1 - EPSILON:
                expected = -EPSILON
            else:
                surviving[tuple(target)] = 0
                if np.sum(surviving) > EPSILON:
                    surviving /= np.sum(surviving)
                branches = surviving * self.env._extract_N_from_2N(self.env.p_Poisson, target)
                hit_mass = np.sum(branches, axis=(1, 2))
                for hit in range(4):
                    if hit_mass[hit] > EPSILON:
                        branches[hit] /= hit_mass[hit]
                expected = 0.0
                for hit in range(4):
                    mean_distance = np.sum(branches[hit] * distance)
                    entropy = _entropy(branches[hit])
                    value = mean_distance + 2 ** (entropy - 1) - 1 / 2
                    if value > 0.0:
                        value = np.log2(value)
                    expected += (1.0 - end_mass) * hit_mass[hit] * value
            scores[action] = expected
        permitted = np.ones(4, dtype=bool) if self._allow_stay else np.asarray(
            [self.env._move(action, position)[1] for action in range(4)], dtype=bool
        )
        if not np.isfinite(scores[permitted]).all() or not np.isposinf(scores[~permitted]).all():
            raise FloatingPointError("invalid analytic action cost; no score repair is permitted")
        chosen = int(np.flatnonzero(np.abs(scores - scores.min()) < EPSILON)[0])
        return chosen, scores


class SpaceAwareActor(ReleasedPolicyActor):
    """Analytic controller with the exact released actor's public state update.

    choose() returns an action and float64[4] costs. In-bounds mode alone uses
    positive infinity for blocked directions. Each actor owns an immutable
    Manhattan-distance table, charged in initialization and reported storage.
    Hypothetical posteriors and entropy arrays are temporary planning work.
    """

    __slots__ = ()

    def __init__(self, initial_public, kernel, *, allow_stay=True):
        if not isinstance(allow_stay, bool):
            raise TypeError("allow_stay must be boolean")
        policy = partial(_SpaceAwarePolicy, allow_stay=allow_stay)
        super().__init__(initial_public, kernel, None, policy, sym_avg=False)

    @property
    def allowed_actions(self):
        if self.public["done"]:
            return ()
        return tuple(range(4)) if self._policy._allow_stay else self.public["valid_actions"]

    def choose(self):
        if self.public["done"]:
            raise RuntimeError("cannot choose after source found")
        if self._pending_action is not None:
            raise RuntimeError("a chosen action is already awaiting its public outcome")
        action, scores = self._policy._value_policy()
        if action not in self.allowed_actions:
            raise ValueError("analytic choice outside declared action set")
        self._pending_action = action
        return action, scores.copy()

    def storage_bytes(self):
        storage = super().storage_bytes()
        kernel_bytes = storage["immutable_array_bytes"]
        distance_bytes = self._policy.distance_array.nbytes
        storage["immutable_array_bytes"] += distance_bytes
        storage["immutable_arrays"] = {"observation_kernel": kernel_bytes, "manhattan_distance_table": distance_bytes}
        storage["scope"] = ("Owned posterior, immutable kernel and per-actor cached Manhattan table; excludes "
                            "scalar metadata, temporary initialization arrays, snapshots and planning workspace.")
        return storage
