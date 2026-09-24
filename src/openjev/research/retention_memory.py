"""Bounded projected Gaussian-process memory, using only public observations.

The state defines q(f) = p(f | f(Z)) N(f(Z); mean, cov). Each expansion
conditions on one new noisy observation. Compression retains a posterior
marginal and restores the prior conditional outside that dictionary. This is
an approximate posterior, not exact conditioning on the complete history.

The spatial white nugget is part of the fixed kernel, shared at exactly equal
coordinates. It is not adaptive numerical jitter or observation noise. Query
covariances describe latent function values and include cross-location terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LENGTH = 1.0
AMPLITUDE = 1.0
NUGGET = 1e-5
NOISE_VARIANCE = .09
KL_ROUNDOFF = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64")
             and value.shape == shape and np.isfinite(value).all(),
             name + " must be a finite float64 array of the declared shape")


def _owned(value):
    # Immutable bytes prevent re-enabling writes through setflags(). No caller
    # array or historical observation buffer is retained.
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def kernel(x, y):
    """Fixed RBF plus exact-coordinate white nugget; [B,n,2] or [n,2]."""
    _require(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
             and x.ndim in (2, 3) and y.ndim == x.ndim and x.shape[-1] == y.shape[-1] == 2,
             "kernel coordinates have matching rank and final dimension two")
    _array(x, x.shape, "left coordinates")
    _array(y, y.shape, "right coordinates")
    if x.ndim == 3:
        _require(x.shape[0] == y.shape[0] and x.shape[0] > 0, "matching nonempty kernel batches")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        differences = x[..., :, None, :] - y[..., None, :, :]
        distance = np.square(differences / LENGTH).sum(axis=-1)
        equal = np.all(x[..., :, None, :] == y[..., None, :, :], axis=-1)
        result = AMPLITUDE**2 * np.exp(-.5 * distance) + NUGGET * equal
    _require(np.isfinite(result).all(), "finite kernel values")
    return result


def _factor(matrix, name):
    _require(np.isfinite(matrix).all(), "finite " + name)
    try:
        result = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(name + " must be positive definite; no extra jitter") from error
    _require(np.isfinite(result).all(), "finite " + name + " Cholesky factor")
    return result


def _solve(factor, right):
    return np.linalg.solve(factor.swapaxes(-1, -2), np.linalg.solve(factor, right))


@dataclass(frozen=True, eq=False)
class State:
    Z: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    step: int = 0
    length: float = LENGTH
    amplitude: float = AMPLITUDE
    nugget: float = NUGGET
    noise_variance: float = NOISE_VARIANCE

    def __post_init__(self):
        _require(isinstance(self.Z, np.ndarray) and self.Z.ndim == 3
                 and self.Z.shape[0] > 0 and self.Z.shape[2] == 2, "state coordinates [B,M,2]")
        batch, size = self.Z.shape[:2]
        _array(self.Z, (batch, size, 2), "state coordinates")
        _array(self.mean, (batch, size), "state mean")
        _array(self.cov, (batch, size, size), "state covariance")
        _require(type(self.step) is int and self.step >= size, "integer step counts all assimilated events")
        scalars = (self.length, self.amplitude, self.nugget, self.noise_variance)
        _require(all(type(v) is float for v in scalars)
                 and scalars == (LENGTH, AMPLITUDE, NUGGET, NOISE_VARIANCE), "fixed kernel and noise scalars")
        if size:
            identical = np.all(self.Z[:, :, None, :] == self.Z[:, None, :, :], axis=-1)
            _require(not np.any(identical & ~np.eye(size, dtype=bool)), "distinct retained coordinates")
            _require(np.allclose(self.cov, self.cov.swapaxes(-1, -2), rtol=0., atol=1e-12),
                     "symmetric state covariance")
            _factor(self.cov, "state covariance")
        for name in ("Z", "mean", "cov"):
            object.__setattr__(self, name, _owned(getattr(self, name)))

    @property
    def array_bytes(self):
        return int(self.Z.nbytes + self.mean.nbytes + self.cov.nbytes)

    @property
    def resident_bytes_per_context(self):
        """Logical per-context state: arrays plus four float64s and int64 step.

        Excludes Python object overhead, policy weights and transient workspace.
        The batched implementation shares the scalar metadata across contexts.
        """
        return self.array_bytes // self.Z.shape[0] + 40


def initial(batch, capacity):
    """Empty state. The caller enforces capacity after temporary expansion."""
    _require(type(batch) is int and batch > 0 and type(capacity) is int and capacity > 0,
             "positive integer batch and capacity")
    return State(np.empty((batch, 0, 2)), np.empty((batch, 0)), np.empty((batch, 0, 0)))


def _state(state):
    _require(type(state) is State, "owned State required")


def predict(state, points):
    """Return latent mean[B,L], cov[B,L,L]; never assimilates a query."""
    _state(state)
    batch, size = state.Z.shape[:2]
    _require(isinstance(points, np.ndarray) and points.ndim == 3 and points.shape[1] > 0,
             "nonempty query coordinates [B,L,2]")
    _array(points, (batch, points.shape[1], 2), "query coordinates")
    covariance = kernel(points, points)
    if not size:
        return {"mean": np.zeros(points.shape[:2]), "cov": covariance}
    prior = kernel(state.Z, state.Z)
    cross = kernel(points, state.Z)
    coefficient = _solve(_factor(prior, "dictionary prior"), cross.swapaxes(-1, -2)).swapaxes(-1, -2)
    mean = np.einsum("bli,bi->bl", coefficient, state.mean)
    covariance = covariance - coefficient @ cross.swapaxes(-1, -2)
    covariance += coefficient @ state.cov @ coefficient.swapaxes(-1, -2)
    covariance = .5 * (covariance + covariance.swapaxes(-1, -2))
    _require(np.isfinite(mean).all() and np.isfinite(covariance).all(), "finite joint prediction")
    _require(np.all(np.linalg.eigvalsh(covariance) >= -1e-10), "PSD joint predictive covariance")
    return {"mean": mean, "cov": covariance}


def expand(state, x, y):
    """Expand the dictionary and apply exactly one new observation likelihood."""
    _state(state)
    batch, size = state.Z.shape[:2]
    _array(x, (batch, 2), "new coordinates")
    _array(y, (batch,), "new observation")
    _require(not np.any(np.all(state.Z == x[:, None, :], axis=-1)),
             "new observation repeats retained coordinates")
    if size:
        prior = kernel(state.Z, state.Z)
        cross = kernel(state.Z, x[:, None, :])
        coefficient = _solve(_factor(prior, "dictionary prior"), cross)
        new_mean = np.einsum("bi,bi->b", coefficient[:, :, 0], state.mean)
        posterior_cross = state.cov @ coefficient
        new_variance = AMPLITUDE**2 + NUGGET - np.sum(cross * coefficient, axis=(1, 2))
        new_variance += np.sum(coefficient * posterior_cross, axis=(1, 2))
    else:
        new_mean = np.zeros(batch)
        posterior_cross = np.empty((batch, 0, 1))
        new_variance = np.full(batch, AMPLITUDE**2 + NUGGET)
    mean = np.concatenate((state.mean, new_mean[:, None]), axis=1)
    covariance = np.empty((batch, size + 1, size + 1))
    covariance[:, :size, :size] = state.cov
    covariance[:, :size, size:] = posterior_cross
    covariance[:, size:, :size] = posterior_cross.swapaxes(-1, -2)
    covariance[:, size, size] = new_variance
    _require(np.isfinite(new_variance).all() and (new_variance > 0).all(), "positive new latent variance")
    innovation_variance = new_variance + NOISE_VARIANCE
    gain = covariance[:, :, -1] / innovation_variance[:, None]
    mean = mean + gain * (y - new_mean)[:, None]
    # Joseph form preserves the same Gaussian update with explicit PSD terms.
    transform = np.broadcast_to(np.eye(size + 1), covariance.shape).copy()
    transform[:, :, -1] -= gain
    covariance = transform @ covariance @ transform.swapaxes(-1, -2)
    covariance += NOISE_VARIANCE * gain[:, :, None] * gain[:, None, :]
    covariance = .5 * (covariance + covariance.swapaxes(-1, -2))
    return State(np.concatenate((state.Z, x[:, None, :]), axis=1), mean, covariance, state.step + 1)


def compress(aug, drop):
    """Retain an exact marginal of the augmented posterior, with no likelihood."""
    _state(aug)
    batch, size = aug.Z.shape[:2]
    _require(size > 0 and isinstance(drop, np.ndarray) and drop.dtype == np.dtype("int64")
             and drop.shape == (batch,) and ((drop >= 0) & (drop < size)).all(), "valid int64 drop index per context")
    keep = np.arange(size)[None, :] != drop[:, None]
    indices = np.broadcast_to(np.arange(size), (batch, size))[keep].reshape(batch, size - 1)
    batch_indices = np.arange(batch)[:, None]
    return State(aug.Z[batch_indices, indices], aug.mean[batch_indices, indices],
                 aug.cov[np.arange(batch)[:, None, None], indices[:, :, None], indices[:, None, :]], aug.step)


def deletion_statistics(aug):
    """Forward KL to each retained-marginal/prior-conditional projection.

    Features: x/2, y/2, prior-conditional mean residual/sqrt(r), log(v/r),
    log1p(KL), and posterior marginal variance/prior marginal variance.
    Only negative KL in [-1e-10,0) is floored, with an explicit count.
    """
    _state(aug)
    batch, size = aug.Z.shape[:2]
    _require(size > 0, "deletion statistics require a nonempty dictionary")
    prior = kernel(aug.Z, aug.Z)
    eye = np.broadcast_to(np.eye(size), (batch, size, size))
    inverse_prior = _solve(_factor(prior, "dictionary prior"), eye)
    inverse_posterior = _solve(_factor(aug.cov, "posterior covariance"), eye)
    prior_diagonal = np.diagonal(inverse_prior, axis1=1, axis2=2)
    posterior_diagonal = np.diagonal(inverse_posterior, axis1=1, axis2=2)
    r, v = 1. / prior_diagonal, 1. / posterior_diagonal
    _require(np.isfinite(r).all() and np.isfinite(v).all() and (r > 0).all() and (v > 0).all(),
             "positive conditional variances")
    normalized = inverse_prior / prior_diagonal[:, :, None]
    residual_mean = np.einsum("bij,bj->bi", normalized, aug.mean)
    residual_variance = np.einsum("bij,bjk,bik->bi", normalized, aug.cov, normalized)
    raw = .5 * (np.log(r / v) + (residual_variance + residual_mean**2) / r - 1.)
    _require(np.isfinite(raw).all() and (raw >= -KL_ROUNDOFF).all(), "nonnegative KL within declared roundoff")
    floor_count = int(np.count_nonzero(raw < 0))
    kl = np.maximum(raw, 0.)
    features = np.stack((aug.Z[:, :, 0] / 2., aug.Z[:, :, 1] / 2., residual_mean / np.sqrt(r),
                         np.log(v / r), np.log1p(kl),
                         np.diagonal(aug.cov, axis1=1, axis2=2) / (AMPLITUDE**2 + NUGGET)), axis=-1)
    _require(np.isfinite(features).all(), "finite deletion features")
    return {"kl": kl, "features": features, "negative_kl_floor_count": floor_count}
