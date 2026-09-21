"""Location-blind quality inference control with a shared public depletion ledger.

The conditional distribution of rock locations remains uniform. Checks update
each rock's initial quality using sensor likelihoods averaged over that fixed
location prior and the known sampled cells. This discards correlations between
location and quality induced by past readings. It is an explicit approximation,
not an exact map-marginalized Bayesian filter. No reward is observed.
"""
from __future__ import annotations

import numpy as np


class QualityRockBelief:
    def __init__(self, size=11, rocks=11, half_efficiency_distance=20.0):
        if type(size) is not int or size < 2 or type(rocks) is not int or not 1 <= rocks <= size * (size - 1):
            raise ValueError("invalid board or rock count")
        if not np.isfinite(half_efficiency_distance) or half_efficiency_distance <= 0:
            raise ValueError("sensor distance must be finite and positive")
        self.size, self.rocks = size, rocks
        self.half_efficiency_distance = float(half_efficiency_distance)
        self._positions = np.array([(r, c) for r in range(size) for c in range(size - 1)])
        self.reset()

    def reset(self):
        self._log_quality = np.full((self.rocks, 2), -np.log(2.0))
        self._sampled = np.zeros((self.size, self.size - 1), dtype=bool)

    def copy(self):
        other = object.__new__(type(self))
        other.__dict__ = self.__dict__.copy()
        other._log_quality = self._log_quality.copy()
        other._sampled = self._sampled.copy()
        return other

    def _position(self, position):
        pos = np.asarray(position)
        if (pos.shape != (2,) or pos.dtype.kind not in "iuf" or not np.isfinite(pos).all()
                or np.any(pos != np.floor(pos)) or np.any(pos < 0)
                or pos[0] >= self.size or pos[1] >= self.size - 1):
            raise ValueError("a nonterminal public board position is required")
        return pos.astype(int)

    def _rock(self, rock):
        if isinstance(rock, (bool, np.bool_)) or not isinstance(rock, (int, np.integer)) or not 0 <= rock < self.rocks:
            raise ValueError("invalid rock index")
        return int(rock)

    def _positive_likelihoods(self, position):
        pos = self._position(position)
        distance = np.linalg.norm(self._positions - pos, axis=1)
        flip = -0.5 * np.expm1(-np.log(2.0) * distance / self.half_efficiency_distance)
        sampled = self._sampled.reshape(-1)
        # Given an initially bad quality the rock remains bad everywhere.
        # An initially good quality remains good only at unsampled locations.
        return np.array([np.mean(flip), np.mean(np.where(sampled, flip, 1.0 - flip))])

    def check_probability(self, rock, position):
        rock = self._rock(rock)
        return float(np.exp(self._log_quality[rock]) @ self._positive_likelihoods(position))

    def condition_check(self, rock, position, positive):
        rock = self._rock(rock)
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("reading sign must be boolean")
        likelihood = self._positive_likelihoods(position)
        if not positive:
            likelihood = 1.0 - likelihood
        with np.errstate(divide="ignore"):
            updated = self._log_quality[rock] + np.log(likelihood)
        normalizer = np.logaddexp(*updated)
        if not np.isfinite(normalizer):
            raise ValueError("impossible public observation")
        result = self.copy()
        result._log_quality[rock] = updated - normalizer
        return result

    def sample(self, position):
        row, col = self._position(position)
        self._sampled[row, col] = True

    def quality_probabilities(self):
        return np.exp(self._log_quality[:, 1]) * (1.0 - self._sampled.mean())

    def expected_rewards(self):
        n = len(self._positions)
        value = 10.0 * np.sum(2.0 * np.exp(self._log_quality[:, 1]) - 1.0) / n
        return np.where(self._sampled, -10.0 * self.rocks / n, value)

    def diagnostics(self):
        return {"kind": "location_blind_quality_approximation", "sampled_cells": int(self._sampled.sum()),
                "ess": None, "max_weight": None, "alive_particles": None, "particles": 0}
