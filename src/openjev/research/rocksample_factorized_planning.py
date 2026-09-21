"""Planner adapter for the frozen independent-rock public-history filter.

Every rock starts with support on every non-exit cell and both qualities. Native
without-replacement location coupling is deliberately absent. Consequently the
summed occupancy of a cell can exceed one, and the additive SAMPLE forecast can
leave the native [-10, 10] reward range. Diagnostics expose both without clipping
or repairing them. This is an approximate planner belief, not native exact Bayes.

Only public board/sensor constants, positions, selected rocks and check signs are
accepted. SAMPLE applies deterministic depletion, without its realized reward.
Episode boundaries belong to the caller, which must reset the belief.
"""
from __future__ import annotations

import copy
from numbers import Integral

import numpy as np

from openjev.research.rocksample_public_belief import PublicRockBelief


class FactorizedRockBelief:
    """Full-location-support approximation with the existing planner interface."""

    def __init__(self, size: int = 11, rocks: int = 11,
                 half_efficiency_distance: float = 20.0):
        self._core = PublicRockBelief(size, rocks, half_efficiency_distance)
        self.size = self._core.size
        self.rocks = self._core.rocks
        self.half_efficiency_distance = self._core.half_efficiency_distance

    @property
    def positions(self) -> np.ndarray:
        """Copied public location support, never an observed or privileged map."""
        return self._core.positions

    @property
    def posterior(self) -> np.ndarray:
        """Copied marginal table, axes (rock, location, bad/good)."""
        return self._core.posterior

    def reset(self) -> None:
        self._core.reset()

    def copy(self) -> FactorizedRockBelief:
        return copy.deepcopy(self)

    def _position(self, position) -> np.ndarray:
        coordinates = self._core._position(position)
        if coordinates[1] == self.size - 1:
            raise ValueError("a nonterminal public board position is required")
        return coordinates

    def _rock(self, rock) -> int:
        if (isinstance(rock, (bool, np.bool_)) or not isinstance(rock, Integral)
                or not 0 <= rock < self.rocks):
            raise ValueError("invalid rock index")
        return int(rock)

    def _observation(self, position) -> np.ndarray:
        observation = np.zeros(2 * self.size + self.rocks, dtype=np.int8)
        observation[position[0]] = 1
        observation[self.size + position[1]] = 1
        return observation

    def check_probability(self, rock, position) -> float:
        """Predict the selected positive reading before incorporating any outcome."""
        rock = self._rock(rock)
        position = self._position(position)
        return float(self._core.probabilities(position)[rock])

    def condition_check(self, rock, position, positive: bool) -> FactorizedRockBelief:
        """Return an independent branch; the source belief is never changed."""
        rock = self._rock(rock)
        position = self._position(position)
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("reading sign must be boolean")
        observation = self._observation(position)
        observation[2 * self.size + rock] = 1 if positive else -1
        result = self.copy()
        result._core.update(position, rock + 5, observation, False)
        return result

    def sample(self, position) -> None:
        """Make every matching location hypothesis bad, regardless of occupancy."""
        position = self._position(position)
        self._core.update(position, 4, self._observation(position), False)

    def quality_probabilities(self) -> np.ndarray:
        return self._core.quality_probabilities()

    def expected_rewards(self) -> np.ndarray:
        """Unclipped additive SAMPLE forecast under the independent-rock table."""
        joint = self.posterior
        return (10.0 * (joint[:, :, 1] - joint[:, :, 0]).sum(axis=0)).reshape(
            self.size, self.size - 1)

    def diagnostics(self) -> dict:
        joint = self.posterior
        occupancy = joint.sum(axis=(0, 2))
        rewards = 10.0 * (joint[:, :, 1] - joint[:, :, 0]).sum(axis=0)
        return {
            "kind": "factorized_full_support",
            "ess": None, "max_weight": None, "alive_particles": None, "particles": 0,
            "locations_per_rock": int(joint.shape[1]),
            "maximum_cell_occupancy": float(occupancy.max()),
            "overfull_cells": int(np.count_nonzero(occupancy > 1.0)),
            "occupancy_excess_mass": float(np.maximum(occupancy - 1.0, 0.0).sum()),
            "minimum_expected_reward": float(rewards.min()),
            "maximum_expected_reward": float(rewards.max()),
        }
