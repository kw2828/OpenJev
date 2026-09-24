"""Bounded raw-observation and fixed FIC controls for retention learning.

Raw memory retains only ordered coordinates and noisy labels. FIC retains a
Gaussian over nine fixed inducing values; its residual is independent per
observation event, including repeated coordinates inside a requested path.
Neither control retains factors, predictions, a full archive, or a random state.
Array storage plus four float64 hyperparameters and an int64 step is reported
per context. Python object overhead and temporary computation are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import ndtr

from openjev.research.retention_data import (
    AMPLITUDE,
    EXPOSURE_THRESHOLD,
    LENGTH,
    NOISE_VARIANCE,
    SPATIAL_NUGGET,
    exact_reference,
    kernel,
)

VERSION = "retention-controls-v1"
RAW_CAPACITY = 41
FIC_CAPACITY = 9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64")
             and value.shape == shape and np.isfinite(value).all(),
             name + " must be a finite float64 array of the declared shape")


def _owned(value):
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


def _metadata(state):
    _require(type(state.step) is int and state.step >= 0, "nonnegative integer event count")
    values = (state.length, state.amplitude, state.nugget, state.noise_variance)
    _require(all(type(value) is float for value in values)
             and values == (LENGTH, AMPLITUDE, SPATIAL_NUGGET, NOISE_VARIANCE),
             "fixed kernel and noise scalars required")


def _anchors():
    axis = np.array([-1.8, .1, 1.8], dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(9, 2)


@dataclass(frozen=True, eq=False)
class RawMemory:
    Z: np.ndarray
    y: np.ndarray
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = SPATIAL_NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _require(isinstance(self.Z, np.ndarray) and self.Z.ndim == 3
                 and self.Z.shape[0] > 0 and 1 <= self.Z.shape[1] <= RAW_CAPACITY
                 and self.Z.shape[2] == 2, "raw coordinates [B,M,2], 1 <= M <= 41")
        batch, size = self.Z.shape[:2]
        _array(self.Z, (batch, size, 2), "raw coordinates")
        _array(self.y, (batch, size), "raw labels")
        _metadata(self)
        used = min(self.step, size)
        _require(not np.any(self.Z[:, used:]) and not np.any(self.y[:, used:]),
                 "unused raw slots must be zero")
        object.__setattr__(self, "Z", _owned(self.Z))
        object.__setattr__(self, "y", _owned(self.y))

    @property
    def array_bytes(self):
        return int(self.Z.nbytes + self.y.nbytes)

    @property
    def resident_bytes_per_context(self):
        return self.array_bytes // self.Z.shape[0] + 40


@dataclass(frozen=True, eq=False)
class FICMemory:
    Z: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = SPATIAL_NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _require(isinstance(self.Z, np.ndarray) and self.Z.ndim == 3
                 and self.Z.shape[0] > 0, "FIC coordinates [B,9,2]")
        batch = self.Z.shape[0]
        _array(self.Z, (batch, 9, 2), "FIC coordinates")
        _array(self.mean, (batch, 9), "FIC mean")
        _array(self.cov, (batch, 9, 9), "FIC covariance")
        _metadata(self)
        _require(np.array_equal(self.Z, np.broadcast_to(_anchors(), self.Z.shape)),
                 "fixed off-grid FIC anchors required")
        _require(np.allclose(self.cov, self.cov.swapaxes(-1, -2), rtol=0., atol=1e-12),
                 "symmetric FIC covariance required")
        np.linalg.cholesky(self.cov)
        for name in ("Z", "mean", "cov"):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    @property
    def array_bytes(self):
        return int(self.Z.nbytes + self.mean.nbytes + self.cov.nbytes)

    @property
    def resident_bytes_per_context(self):
        return self.array_bytes // self.Z.shape[0] + 40


def initial_raw(batch, capacity=RAW_CAPACITY):
    _require(type(batch) is int and batch > 0 and type(capacity) is int
             and 1 <= capacity <= RAW_CAPACITY, "positive batch and raw capacity <= 41")
    return RawMemory(np.zeros((batch, capacity, 2)), np.zeros((batch, capacity)))


def write_raw(state, x, y, mode="coverage"):
    """Process one public observation per context; exact ties remove oldest.

    Coverage drops the first point attaining the smallest nearest-neighbor
    squared distance in the augmented buffer, preserving the remaining order.
    The capacity is encoded only by the allocated array shape.
    """
    _require(type(state) is RawMemory, "RawMemory required")
    _require(type(mode) is str and mode in ("coverage", "recent"), "unknown raw retention mode")
    batch, size = state.Z.shape[:2]
    _array(x, (batch, 2), "new coordinates")
    _array(y, (batch,), "new labels")
    if state.step < size:
        coordinates, labels = state.Z.copy(), state.y.copy()
        coordinates[:, state.step] = x
        labels[:, state.step] = y
    else:
        augmented = np.concatenate((state.Z, x[:, None]), axis=1)
        augmented_y = np.concatenate((state.y, y[:, None]), axis=1)
        if mode == "recent":
            drop = np.zeros(batch, dtype=np.int64)
        else:
            with np.errstate(over="raise", invalid="raise"):
                difference = augmented[:, :, None] - augmented[:, None, :]
                distances = np.sum(difference*difference, axis=-1)
            _require(np.isfinite(distances).all(), "finite raw pair distances required")
            distances[:, np.arange(size+1), np.arange(size+1)] = np.inf
            drop = np.argmin(np.min(distances, axis=-1), axis=1)
        keep = np.arange(size+1)[None] != drop[:, None]
        coordinates = augmented[keep].reshape(batch, size, 2)
        labels = augmented_y[keep].reshape(batch, size)
    return RawMemory(coordinates, labels, state.step+1)


def _paths(state, paths):
    _require(isinstance(paths, np.ndarray) and paths.ndim == 5 and paths.shape[1] > 0,
             "nonempty paths [B,Q,4,4,2]")
    _array(paths, (state.Z.shape[0], paths.shape[1], 4, 4, 2), "paths")


def predict_raw(state, paths):
    """Exact known-law GP conditioned only on retained noisy observations."""
    _require(type(state) is RawMemory and state.step > 0, "nonempty RawMemory required")
    _paths(state, paths)
    used = min(state.step, state.Z.shape[1])
    result = exact_reference(state.Z[:, :used], state.y[:, :used], paths)
    return {key: result[key] for key in ("mean", "variance", "risk")}


def initial_fic(batch):
    _require(type(batch) is int and batch > 0, "positive integer batch required")
    anchors = _anchors()
    return FICMemory(np.broadcast_to(anchors, (batch, 9, 2)), np.zeros((batch, 9)),
                     np.broadcast_to(kernel(anchors, anchors), (batch, 9, 9)))


def _coefficients(anchors, points):
    prior = kernel(anchors, anchors)
    cross = kernel(anchors, points)
    coefficients = cho_solve((np.linalg.cholesky(prior), True), cross, check_finite=False).T
    residual = AMPLITUDE**2 + SPATIAL_NUGGET - np.sum(coefficients*cross.T, axis=1)
    _require(np.isfinite(coefficients).all() and np.isfinite(residual).all()
             and (residual >= 0).all(), "nonnegative finite FIC residual; no clipping or jitter")
    return coefficients, residual


def write_fic(state, x, y):
    """One Kalman update with noise plus the per-event conditional residual."""
    _require(type(state) is FICMemory, "FICMemory required")
    batch = state.Z.shape[0]
    _array(x, (batch, 2), "new coordinates")
    _array(y, (batch,), "new labels")
    mean, covariance = np.empty_like(state.mean), np.empty_like(state.cov)
    for context in range(batch):
        coefficient, residual = _coefficients(state.Z[context], x[context:context+1])
        a = coefficient[0]
        likelihood_variance = NOISE_VARIANCE + residual[0]
        cross = state.cov[context] @ a
        denominator = likelihood_variance + a @ cross
        _require(np.isfinite(denominator) and denominator > 0, "positive FIC innovation variance")
        gain = cross / denominator
        mean[context] = state.mean[context] + gain*(y[context]-a @ state.mean[context])
        transform = np.eye(9) - np.outer(gain, a)
        updated = transform @ state.cov[context] @ transform.T + likelihood_variance*np.outer(gain, gain)
        covariance[context] = .5*(updated+updated.T)
    return FICMemory(state.Z, mean, covariance, state.step+1)


def predict_fic(state, paths):
    """FIC latent-average risk, with independent residuals per path event.

    No observation noise is added to latent exposure. Even identical path
    coordinates have distinct residual events in this approximation; inducing
    covariance remains shared. No target or private exposure is accepted.
    """
    _require(type(state) is FICMemory, "FICMemory required")
    _paths(state, paths)
    mean = np.empty(paths.shape[:3], np.float64)
    variance = np.empty_like(mean)
    for context in range(paths.shape[0]):
        coefficients, residual = _coefficients(state.Z[context], paths[context].reshape(-1, 2))
        average = coefficients.reshape(paths.shape[1], 4, 4, 9).mean(axis=2)
        mean[context] = average @ state.mean[context]
        variance[context] = np.einsum("qai,ij,qaj->qa", average, state.cov[context], average)
        variance[context] += residual.reshape(paths.shape[1], 4, 4).sum(axis=2)/16
    _require(np.isfinite(mean).all() and np.isfinite(variance).all() and (variance > 0).all(),
             "positive finite FIC path variance required")
    risk = ndtr((mean-EXPOSURE_THRESHOLD)/np.sqrt(variance))
    _require(np.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all(), "finite FIC risk")
    return {"mean": mean, "variance": variance, "risk": risk}
