"""Public-history finite-map filtering with native RockSample occupancy exclusion.

Every fixed map hypothesis has distinct rock locations and independent conditional
bad/good quality probabilities. Checks update map weights using the *pre-update*
predictive likelihood, then condition that map's selected quality. Sampling uses
only the public coordinate and known depletion rule, never a realized reward.

This is exact conditioning of a finite map mixture, not exact inference over all
native maps. Maps never resample or rejuvenate; missing prior support cannot be
recovered. No simulator, environment seed, hidden state or reward is consumed.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


class ImpossibleObservationError(ValueError):
    """The supplied check result has zero probability under every live hypothesis."""


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sensor_distance(value) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError("half_efficiency_distance must be finite and positive")
    return float(value)


def _real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64)


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


class ParticleRockBelief:
    """A fixed mixture of distinct maps with conditional Bernoulli quality bits.

    ``seed`` is the public inference algorithm's randomness, independent of any
    environment constructor/reset seed. ``reset`` restores this initialization's
    map weights and quality priors without drawing new maps.
    """

    def __init__(self, size: int = 11, rocks: int = 11, particles: int = 256,
                 seed: int = 530001, half_efficiency_distance: float = 20.0):
        size = _integer(size, "size", 2)
        rocks = _integer(rocks, "rocks", 1)
        particles = _integer(particles, "particles", 1)
        seed = _integer(seed, "seed", 0)
        if rocks > size * (size - 1):
            raise ValueError("rocks exceed the distinct non-exit locations")
        half_efficiency_distance = _sensor_distance(half_efficiency_distance)
        rng = np.random.default_rng(seed)
        locations = np.stack([rng.choice(size * (size - 1), size=rocks, replace=False)
                              for _ in range(particles)])
        maps = np.stack((locations // (size - 1), locations % (size - 1)), axis=-1)
        self._initialize(maps, 0.5, None, size, half_efficiency_distance)
        self.seed = seed

    @classmethod
    def from_hypotheses(cls, maps, qualities=0.5, weights=None, *, size: int = 11,
                        half_efficiency_distance: float = 20.0) -> ParticleRockBelief:
        """Construct an explicit finite mixture for classical references or tests.

        maps has shape (particles, rocks, 2), with no repeated cell inside a map.
        qualities broadcasts to (particles, rocks); weights has shape (particles,).
        Nonnegative weights are normalized. Supplying actual environment maps or
        qualities creates a PRIVILEGED reference, which the caller must label and
        keep separate from public actors. This method never obtains such data.
        """
        result = cls.__new__(cls)
        result._initialize(maps, qualities, weights, _integer(size, "size", 2),
                           _sensor_distance(half_efficiency_distance))
        result.seed = None
        return result

    def _initialize(self, maps, qualities, weights, size, half_efficiency_distance) -> None:
        coordinates = _real_array(maps, "maps")
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2 or min(coordinates.shape[:2]) < 1:
            raise ValueError("maps must have shape (positive particles, positive rocks, 2)")
        if (np.any(coordinates != np.floor(coordinates)) or np.any(coordinates < 0)
                or np.any(coordinates[:, :, 0] >= size) or np.any(coordinates[:, :, 1] >= size - 1)):
            raise ValueError("map coordinates must be integer non-exit board locations")
        self.size, self.particles, self.rocks = size, coordinates.shape[0], coordinates.shape[1]
        if self.rocks > size * (size - 1):
            raise ValueError("rocks exceed the distinct non-exit locations")
        self.half_efficiency_distance = half_efficiency_distance
        self._maps = coordinates.astype(np.int64)
        indices = self._maps[:, :, 0] * (size - 1) + self._maps[:, :, 1]
        if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
            raise ValueError("each map must exclude overlapping rock locations")
        self._maps.flags.writeable = False
        self._indices = indices
        self._indices.flags.writeable = False
        probabilities = _real_array(qualities, "qualities")
        try:
            probabilities = np.broadcast_to(probabilities, (self.particles, self.rocks))
        except ValueError as exc:
            raise ValueError("qualities must broadcast to (particles, rocks)") from exc
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("quality probabilities must lie in [0, 1]")
        masses = np.ones(self.particles) if weights is None else _real_array(weights, "weights")
        if masses.shape != (self.particles,) or np.any(masses < 0) or not np.any(masses > 0):
            raise ValueError("weights must be a nonnegative vector with positive total mass")
        with np.errstate(divide="ignore"):
            self._initial_log_quality = np.stack((np.log1p(-probabilities), np.log(probabilities)), axis=-1)
            logs = np.log(masses)
        self._initial_log_weights = logs - np.logaddexp.reduce(logs)
        self._initial_log_quality.flags.writeable = False
        self._initial_log_weights.flags.writeable = False
        self.reset()

    @property
    def maps(self) -> np.ndarray:
        """Read-only inference hypotheses, not an observation of the actual map."""
        result = self._maps.view()
        result.flags.writeable = False
        return result

    @property
    def map_probabilities(self) -> np.ndarray:
        """Copied posterior map weights. Tiny masses can underflow on export."""
        return _normalized_weights(self._log_weights)

    @property
    def conditional_quality_probabilities(self) -> np.ndarray:
        """Copied P(good | map) array, including unused normalized dead-map states."""
        normalizers = np.logaddexp(self._log_quality[:, :, 0], self._log_quality[:, :, 1])
        return np.exp(self._log_quality[:, :, 1] - normalizers)

    def reset(self) -> None:
        """Restore the initial mixture and qualities, retaining identical maps."""
        self._log_weights = self._initial_log_weights.copy()
        self._log_quality = self._initial_log_quality.copy()

    def copy(self) -> ParticleRockBelief:
        """Share immutable maps/priors and copy all mutable filtering state."""
        result = type(self).__new__(type(self))
        for name in ("size", "rocks", "particles", "seed", "half_efficiency_distance", "_maps", "_indices",
                     "_initial_log_weights", "_initial_log_quality"):
            setattr(result, name, getattr(self, name))
        result._log_weights = self._log_weights.copy()
        result._log_quality = self._log_quality.copy()
        return result

    def _position(self, position) -> np.ndarray:
        values = _real_array(position, "position")
        if (values.shape != (2,) or np.any(values != np.floor(values)) or np.any(values < 0)
                or np.any(values >= self.size)):
            raise ValueError("position must contain two integer board coordinates")
        return values.astype(np.int64)

    def _check_likelihood(self, rock: int, position, positive: bool) -> np.ndarray:
        rock = _integer(rock, "rock", 0)
        if rock >= self.rocks:
            raise ValueError("rock index is outside the public action space")
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError("positive must be a boolean signed-reading outcome")
        coordinate = self._position(position)
        distance = np.linalg.norm(self._maps[:, rock, :] - coordinate, axis=1)
        with np.errstate(over="ignore", divide="ignore"):
            flip = -0.5 * np.expm1(-np.log(2.0) * distance / self.half_efficiency_distance)
            correct, incorrect = np.log1p(-flip), np.log(flip)
        likelihood = np.column_stack((incorrect, correct))
        return likelihood if positive else likelihood[:, ::-1]

    def check_probability(self, rock: int, position) -> float:
        """Predict the positive CHECK outcome before incorporating that outcome."""
        likelihood = self._check_likelihood(rock, position, True)
        positive = np.logaddexp.reduce(self._log_quality[:, rock, :] + likelihood, axis=1)
        negative = np.logaddexp.reduce(self._log_quality[:, rock, :] + likelihood[:, ::-1], axis=1)
        log_positive = np.logaddexp.reduce(self._log_weights + positive)
        log_negative = np.logaddexp.reduce(self._log_weights + negative)
        return float(np.exp(log_positive - np.logaddexp(log_positive, log_negative)))

    def condition_check(self, rock: int, position, positive: bool) -> ParticleRockBelief:
        """Return a conditioned copy; impossible evidence raises without mutation.

        A map's likelihood is computed from its old quality distribution exactly
        once. Zero-likelihood maps become dead; their unused quality pair is kept
        normalized rather than forming undefined 0/0 conditional probabilities.
        """
        likelihood = self._check_likelihood(rock, position, positive)
        joint = self._log_quality[:, rock, :] + likelihood
        evidence = np.logaddexp.reduce(joint, axis=1)
        posterior_weights = self._log_weights + evidence
        normalizer = np.logaddexp.reduce(posterior_weights)
        if not np.isfinite(normalizer):
            raise ImpossibleObservationError("check outcome has zero probability under every live map")
        result = self.copy()
        result._log_weights = posterior_weights - normalizer
        supported = np.isfinite(evidence)
        result._log_quality[supported, rock, :] = joint[supported] - evidence[supported, None]
        return result

    def sample(self, position) -> None:
        """Deplete matching locations in every hypothesis without reweighting maps."""
        coordinate = self._position(position)
        matches = np.all(self._maps == coordinate, axis=2)
        self._log_quality[:, :, 0][matches] = 0.0
        self._log_quality[:, :, 1][matches] = -np.inf

    def expected_rewards(self) -> np.ndarray:
        """Native expected SAMPLE reward on every non-exit cell, including empty cells."""
        good = self.conditional_quality_probabilities
        normalizers = np.logaddexp(self._log_quality[:, :, 0], self._log_quality[:, :, 1])
        bad = np.exp(self._log_quality[:, :, 0] - normalizers)
        contributions = 10.0 * self.map_probabilities[:, None] * (good - bad)
        flat = np.zeros(self.size * (self.size - 1), dtype=np.float64)
        np.add.at(flat, self._indices.ravel(), contributions.ravel())
        return flat.reshape(self.size, self.size - 1)

    def quality_probabilities(self) -> np.ndarray:
        """Marginal P(good) for each rock under the current map mixture."""
        return np.sum(self.map_probabilities[:, None] * self.conditional_quality_probabilities, axis=0)

    def diagnostics(self) -> dict[str, float | int]:
        """ESS and max weight; alive means finite log weight, even if export underflows."""
        weights = self.map_probabilities
        return {"ess": float(1.0 / np.sum(weights**2)), "max_weight": float(np.max(weights)),
                "alive_particles": int(np.isfinite(self._log_weights).sum()), "particles": self.particles}
