"""Public-history approximate filtering for POBAX RockSample.

Each rock has a joint distribution over its unknown location and current bad/good
quality. Rocks are independent here: the native map's exclusion of overlapping
locations is NOT represented. This is not an exact filter for the native task.
Only board/sensor rules and public coordinates, actions, readings and boundaries
are consumed. There are no simulator-state, map, seed or reward inputs.

The observation contract is the raw ``2 * size + rocks`` vector: one-hot row,
one-hot column, then signed check readings. An adapter must supply a fresh reset
observation on ``done``; the old action is then ignored. Predict before updating
with the outcome being scored. There is no probability floor or random state.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class PublicRockBelief:
    """Independent per-rock float64 log beliefs, with axes (rock, location, quality).

    Quality index 0 means bad and 1 means good. Candidate locations are row-major,
    excluding the east exit column. All constants describe public task rules.
    Returned audit arrays are copies and cannot change the filter's state.
    """

    def __init__(self, size: int = 11, rocks: int = 11,
                 half_efficiency_distance: float = 20.0):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 2:
            raise ValueError("size must be an integer >= 2")
        if (isinstance(rocks, (bool, np.bool_)) or not isinstance(rocks, Integral)
                or not 1 <= rocks <= size * (size - 1)):
            raise ValueError("rocks must fit the native distinct-location map")
        if (isinstance(half_efficiency_distance, (bool, np.bool_))
                or not isinstance(half_efficiency_distance, Real)
                or not np.isfinite(half_efficiency_distance) or half_efficiency_distance <= 0):
            raise ValueError("half_efficiency_distance must be finite and positive")
        self.size = int(size)
        self.rocks = int(rocks)
        self.half_efficiency_distance = float(half_efficiency_distance)
        self._positions = np.column_stack((np.repeat(np.arange(self.size), self.size - 1),
                                           np.tile(np.arange(self.size - 1), self.size)))
        self.reset()

    def reset(self) -> None:
        """Forget all episode evidence and restore independent uniform priors."""
        self._log_belief = np.full((self.rocks, len(self._positions), 2),
                                  -np.log(2.0 * len(self._positions)), dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        """Public candidate-location grid, not an observed or true map."""
        return self._positions.copy()

    @property
    def posterior(self) -> np.ndarray:
        """Copied normalized probability table for audits, without probability floors."""
        weights = np.exp(self._log_belief - np.max(self._log_belief, axis=(1, 2), keepdims=True))
        return weights / weights.sum(axis=(1, 2), keepdims=True)

    def _position(self, position) -> np.ndarray:
        values = np.asarray(position)
        if (values.shape != (2,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                or np.any(values != np.floor(values)) or np.any(values < 0)
                or np.any(values >= self.size)):
            raise ValueError("position must contain two integer board coordinates")
        return values.astype(np.int64)

    def _observation(self, observation) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(observation)
        if (values.shape != (2 * self.size + self.rocks,) or values.dtype.kind not in "biuf"
                or not np.isfinite(values).all()):
            raise ValueError("observation must be the finite raw public observation vector")
        coordinates = values[:2 * self.size].reshape(2, self.size)
        if not np.isin(coordinates, (0, 1)).all() or not np.all(coordinates.sum(axis=1) == 1):
            raise ValueError("observation coordinates must each be one-hot")
        readings = values[2 * self.size:]
        if not np.isin(readings, (-1, 0, 1)).all():
            raise ValueError("check readings must be signed -1, 0 or 1")
        return np.argmax(coordinates, axis=1), readings

    def _flip_probabilities(self, position) -> np.ndarray:
        coordinate = self._position(position)
        distance = np.linalg.norm(self._positions - coordinate, axis=1)
        # expm1 retains nonzero error probabilities close to a perfect sensor.
        with np.errstate(over="ignore"):
            return -0.5 * np.expm1(-np.log(2.0) * distance / self.half_efficiency_distance)

    def probabilities(self, position) -> np.ndarray:
        """Predict P(positive reading | CHECK rock i at position) for every rock.

        Call before feeding that check's result to update(). The function is pure
        with respect to filter state and accepts any public board coordinate.
        """
        flip = self._flip_probabilities(position)
        joint = self.posterior
        positive = (joint[:, :, 0] * flip + joint[:, :, 1] * (1.0 - flip)).sum(axis=1)
        negative = (joint[:, :, 0] * (1.0 - flip) + joint[:, :, 1] * flip).sum(axis=1)
        return positive / (positive + negative)

    def quality_probabilities(self) -> np.ndarray:
        """Return P(current quality is good), marginalized over unknown location."""
        quality = self.posterior.sum(axis=1)
        return quality[:, 1] / quality.sum(axis=1)

    @staticmethod
    def _normalize(log_joint: np.ndarray) -> np.ndarray:
        maximum = np.max(log_joint)
        if not np.isfinite(maximum):
            raise ValueError("public observation has zero probability under the current belief")
        centered = log_joint - maximum
        return centered - np.log(np.exp(centered).sum())

    def update(self, previous_position, action: int, observation, done: bool) -> None:
        """Apply one public transition, atomically rejecting inconsistent inputs.

        On done, require a zero-reading reset observation, ignore the terminating
        action and previous position, and restore the prior. Otherwise movement
        must match the public coordinates, and only a CHECK may carry a reading.
        SAMPLE changes every rock's matching location hypothesis to bad. It does
        not use a hidden occupancy lookup or the realized sampling reward.
        """
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be a boolean episode-boundary flag")
        position, readings = self._observation(observation)
        if done:
            if np.any(readings) or position[1] == self.size - 1:
                raise ValueError("done requires a fresh nonterminal, zero-reading reset observation")
            self.reset()
            return
        previous = self._position(previous_position)
        if previous[1] == self.size - 1 or position[1] == self.size - 1:
            raise ValueError("non-done transition cannot enter or leave the terminal east column")
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral)
                or not 0 <= action < self.rocks + 5):
            raise ValueError("action must be an integer in the public discrete action space")
        action = int(action)
        expected = previous.copy()
        if action < 4:
            expected += np.array(((-1, 0), (0, 1), (1, 0), (0, -1)))[action]
            expected = np.clip(expected, 0, self.size - 1)
        if not np.array_equal(position, expected):
            raise ValueError("observation position disagrees with the public action")
        active = np.flatnonzero(readings)
        if action >= 5:
            rock = action - 5
            if not np.array_equal(active, [rock]):
                raise ValueError("CHECK requires exactly the selected rock's signed reading")
            flip = self._flip_probabilities(position)
            with np.errstate(divide="ignore"):
                correct = np.log1p(-flip)
                incorrect = np.log(flip)
            log_likelihood = np.column_stack((incorrect, correct))
            if readings[rock] < 0:
                log_likelihood = log_likelihood[:, ::-1]
            updated = self._normalize(self._log_belief[rock] + log_likelihood)
            self._log_belief[rock] = updated
        else:
            if len(active):
                raise ValueError("movement and SAMPLE observations cannot contain check readings")
            if action == 4:
                indices = np.flatnonzero(np.all(self._positions == previous, axis=1))
                if len(indices):
                    index = int(indices[0])
                    self._log_belief[:, index, 0] = np.logaddexp(
                        self._log_belief[:, index, 0], self._log_belief[:, index, 1])
                    self._log_belief[:, index, 1] = -np.inf
