"""Inference-only Bayesian linear memory for four centered score coordinates.

This NumPy float64 reference is a future control, independent of the currently
frozen experiment. The caller owns a fixed cue projection and supplies labels.
There is no episode, query schedule, score normalization, first-query exclusion,
forgetting, gating, noise fitting or model training in this component.

With zero prior mean and isotropic prior covariance, full mode uses
    v = P z; s = noise_variance + z.T v; k = v / s
    W_new = W + outer(target_centered - W z, k)
    P_new = P - outer(v, k).
All right-hand quantities use the prewrite state. Diagonal mode uses the same
scalar innovation denominator with diagonal P, then discards the off-diagonal
entries of the posterior covariance. It is a moment-projection approximation,
not diagonal precision accumulation. For key dimension one the two modes use
exactly the same arithmetic.

Four output rows represent a three-dimensional contrast space. The scalar
returned by predict is epistemic variance: output covariance is that scalar
multiplied by C = I_4 - 11.T/4. Observation noise is noise_variance * C. These
quantities are conditional linear-Gaussian estimates, not calibrated action
probabilities or a guarantee of correctness under misspecification.

The full rank-one update costs O(d^2), but this deliberately conservative
reference ALSO runs a Cholesky positive-definiteness validation after every
prospective full update and state import, costing O(d^3). That validation cost
must be charged in a benchmark. Diagonal updates/validation cost O(d). No
speed advantage, biological mechanism or empirical benefit is claimed.

Array inputs are copied. Reads have no state effect. State exports are copies;
imports, resets and labeled updates commit only after all validation succeeds.
Overflow, nonfinite arithmetic, or lost positive definiteness is rejected with
ValueError and leaves the old state and update count unchanged. There is no
jitter, clipping, variance floor or silent numerical recovery.
"""
from __future__ import annotations

import numbers
from typing import NamedTuple

import numpy as np

VERSION = "bayesian-score-memory-v1"
MODES = ("full", "diagonal")
_CENTER_TOLERANCE = 64 * np.finfo(np.float64).eps


class Prediction(NamedTuple):
    """Owned centered mean and scalar epistemic variance, before any write."""

    mean: np.ndarray
    epistemic_variance: float


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_scalar(value, name):
    _require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)),
             name + " must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(name + " must be representable as float64") from error
    _require(np.isfinite(result) and result > 0, name + " must be finite and positive")
    return result


def _array(value, shape, name, *, state=False):
    try:
        raw = np.asarray(value)
        _require(raw.shape == shape and raw.dtype.kind in "fiu", name + " must have the declared real shape")
        if state:
            _require(isinstance(value, np.ndarray) and raw.dtype == np.dtype(np.float64),
                     name + " state must be a float64 ndarray")
        with np.errstate(over="raise", invalid="raise"):
            result = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (FloatingPointError, OverflowError, TypeError) as error:
        raise ValueError(name + " cannot be represented in float64") from error
    _require(bool(np.isfinite(result).all()), name + " must be finite")
    return result


def _center(value):
    # Divide first, so a finite average does not overflow solely in the sum.
    return value - np.sum(value * .25, axis=0, keepdims=value.ndim == 2)


def _require_centered(value, name):
    # Scaling also prevents a validation sum from overflowing. Nonzero columns
    # receive a relative tolerance only; a constant tiny target still fails.
    scale = np.max(np.abs(value), axis=0)
    safe_scale = np.where(scale == 0, 1., scale)
    normalized_mean = np.sum((value / safe_scale) * .25, axis=0)
    _require(bool((np.abs(normalized_mean) <= _CENTER_TOLERANCE).all()), name + " must be centered across four outputs")


def _covariance(value, key_dim, mode, *, state=False):
    shape = (key_dim, key_dim) if mode == "full" else (key_dim,)
    result = _array(value, shape, "covariance", state=state)
    if mode == "full":
        # Exported states are exactly symmetric. Reject asymmetric imports
        # instead of silently using one triangle or replacing user evidence.
        _require(np.array_equal(result, result.T), "full covariance must be symmetric")
        try:
            factor = np.linalg.cholesky(result)
        except np.linalg.LinAlgError as error:
            raise ValueError("full covariance must be positive definite") from error
        _require(bool(np.isfinite(factor).all()) and bool((np.diag(factor) > 0).all()),
                 "full covariance factor must be finite and positive")
    else:
        _require(bool((result > 0).all()), "diagonal covariance must be positive")
    return result


class BayesianScoreMemory:
    """A caller-managed posterior over centered linear score corrections.

    ``prior_variance`` and ``noise_variance`` are mandatory positive scalars;
    this component supplies no empirical hyperparameter defaults.

    ``predict(cue)`` returns an owned ``Prediction`` and never changes state.
    ``observe(cue, target_centered)`` is one labeled update and returns that
    observation's prewrite ``Prediction``. The caller decides when to read or
    write and must not pass unavailable future labels. Centered targets may
    have only tiny relative floating-point centering error; arbitrary offsets
    are rejected. Tiny accepted centering error is projected out explicitly.
    """

    def __init__(self, key_dim, *, prior_variance, noise_variance, mode="full"):
        _require(isinstance(key_dim, numbers.Integral) and not isinstance(key_dim, (bool, np.bool_))
                 and key_dim > 0, "key_dim must be a positive integer")
        _require(type(mode) is str and mode in MODES, "mode must be full or diagonal")
        self._key_dim = int(key_dim)
        self._mode = mode
        self._prior_variance = _positive_scalar(prior_variance, "prior_variance")
        self._noise_variance = _positive_scalar(noise_variance, "noise_variance")
        self.reset()

    @property
    def key_dim(self):
        return self._key_dim

    @property
    def mode(self):
        return self._mode

    @property
    def prior_variance(self):
        return self._prior_variance

    @property
    def noise_variance(self):
        return self._noise_variance

    def reset(self):
        """Restore the zero mean, original isotropic prior and zero updates."""
        weights = np.zeros((4, self.key_dim), dtype=np.float64)
        covariance = (np.eye(self.key_dim, dtype=np.float64) * self.prior_variance
                      if self.mode == "full" else np.full(self.key_dim, self.prior_variance, dtype=np.float64))
        covariance = _covariance(covariance, self.key_dim, self.mode)
        self._weights, self._covariance, self._labeled_updates = weights, covariance, 0

    def _read(self, cue):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                if self.key_dim == 1:
                    diagonal = self._covariance.reshape(-1)
                    direction = diagonal * cue
                elif self.mode == "full":
                    direction = self._covariance @ cue
                else:
                    direction = self._covariance * cue
                variance = float(cue @ direction)
                mean = _center(self._weights @ cue)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("nonfinite posterior prediction arithmetic") from error
        _require(bool(np.isfinite(direction).all()) and np.isfinite(variance) and variance >= 0
                 and bool(np.isfinite(mean).all()), "finite nonnegative posterior prediction")
        return Prediction(mean.copy(), variance), direction

    def predict(self, cue):
        """Read prewrite mean and epistemic variance without changing state."""
        cue = _array(cue, (self.key_dim,), "cue")
        prediction, _ = self._read(cue)
        return prediction

    def observe(self, cue, target_centered):
        """Atomically incorporate exactly one explicitly labeled observation."""
        cue = _array(cue, (self.key_dim,), "cue")
        target = _array(target_centered, (4,), "target_centered")
        _require_centered(target, "target_centered")
        prediction, direction = self._read(cue)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                target = _center(target)
                innovation_variance = self.noise_variance + prediction.epistemic_variance
                _require(np.isfinite(innovation_variance) and innovation_variance > 0,
                         "finite positive innovation variance")
                gain = direction / innovation_variance
                weights = _center(self._weights + np.outer(target - prediction.mean, gain))
                if self.key_dim == 1:
                    covariance = (self._covariance.reshape(-1) - direction * gain).reshape(self._covariance.shape)
                elif self.mode == "full":
                    covariance = self._covariance - np.outer(direction, gain)
                    # Rank-one subtraction is mathematically symmetric; average
                    # the two computed triangles to remove multiplication-order
                    # roundoff. This is not eigenvalue repair or added noise.
                    covariance = covariance * .5 + covariance.T * .5
                else:
                    covariance = self._covariance - direction * gain
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("nonfinite labeled posterior update arithmetic") from error
        _require(bool(np.isfinite(weights).all()), "finite posterior mean weights")
        _require_centered(weights, "posterior mean weights")
        covariance = _covariance(covariance, self.key_dim, self.mode)
        # No mutation above this line: rejected observations retain all bytes
        # and the labeled observation count of the previous posterior.
        self._weights, self._covariance = weights, covariance
        self._labeled_updates += 1
        return prediction

    def counts(self):
        """State allocation and observation counts, not FLOPs or speed claims."""
        return {"mode": self.mode, "covariance_entries": self.key_dim**2 if self.mode == "full" else self.key_dim,
                "mean_entries": 4 * self.key_dim, "labeled_updates": self._labeled_updates}

    def state_dict(self):
        """Export owned posterior arrays plus the explicit fixed configuration."""
        return {"version": VERSION, "key_dim": self.key_dim, "mode": self.mode,
                "prior_variance": self.prior_variance, "noise_variance": self.noise_variance,
                "weights": self._weights.copy(), "covariance": self._covariance.copy(),
                "labeled_updates": self._labeled_updates}

    def load_state(self, state):
        """Atomically import validated copies under this exact configuration.

        Covariance must be positive definite in full mode or strictly positive
        in diagonal mode. Full covariance import incurs O(d^3) Cholesky
        validation. This validates a posterior representation, not its provenance
        or compatibility with an unverifiable observation history.
        """
        required = {"version", "key_dim", "mode", "prior_variance", "noise_variance", "weights", "covariance", "labeled_updates"}
        _require(isinstance(state, dict) and set(state) == required, "exact exported state fields")
        _require(state["version"] == VERSION and type(state["key_dim"]) is int and state["key_dim"] == self.key_dim
                 and state["mode"] == self.mode, "matching posterior dimension, version and mode")
        _require(_positive_scalar(state["prior_variance"], "prior_variance") == self.prior_variance
                 and _positive_scalar(state["noise_variance"], "noise_variance") == self.noise_variance,
                 "matching fixed prior and noise variances")
        _require(type(state["labeled_updates"]) is int and state["labeled_updates"] >= 0,
                 "nonnegative integer labeled update count")
        weights = _array(state["weights"], (4, self.key_dim), "weights", state=True)
        _require_centered(weights, "imported mean weights")
        covariance = _covariance(state["covariance"], self.key_dim, self.mode, state=True)
        self._weights, self._covariance, self._labeled_updates = weights, covariance, state["labeled_updates"]

    @classmethod
    def from_state(cls, state):
        """Construct and atomically import an explicit exported configuration."""
        _require(isinstance(state, dict) and {"key_dim", "prior_variance", "noise_variance", "mode"} <= set(state),
                 "explicit exported posterior configuration")
        result = cls(state["key_dim"], prior_variance=state["prior_variance"],
                     noise_variance=state["noise_variance"], mode=state["mode"])
        result.load_state(state)
        return result
