"""Static RBF memory under four declared Gaussian covariance approximations.

SoR uses a rank16 inducing feature prior. query_only adds residual variance to
the final query only, as a diagnostic, without changing SoR routing or means.
FIC uses independent residuals per observation event throughout. In particular,
two events at identical coordinates have independent approximate residuals;
this is not the full GP's shared latent value at duplicate coordinates.
full is ordinary cached exact GP conditioning at the supplied kernel.

Every request starts from immutable public archive statistics and incorporates
its few shots once. Repeating a call is computation, not new evidence. No
private identity, query target, fitted neural model or empirical data is read.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp, ndtr

VERSION = "residual-memory-v1"
MODES = ("sor", "query_only", "fic", "full")
JITTER = 1e-6
RESIDUAL_TOLERANCE = 1e-12
NOISE_VARIANCE = .0225
DIAGNOSTIC_KEYS = ("residual_points", "minimum_raw_residual", "residual_floor_count")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, name):
    _require(type(value) in (int, float), name + " must be a positive finite scalar")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + " must be representable") from error
    _require(math.isfinite(value) and value > 0, name + " must be positive and finite")
    return value


def _array(value, ndim, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64")
             and value.ndim == ndim and min(value.shape) > 0 and np.isfinite(value).all(),
             name + " must be a nonempty finite float64 array with the declared dimensions")


def _owned(value):
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    _require(np.isfinite(result).all(), "finite owned cache array")
    result.flags.writeable = False
    return result


def _kernel(x, y, length, amplitude):
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        differences = (x[..., :, None, :] - y[..., None, :, :]) / length
        result = amplitude**2 * np.exp(-.5 * np.square(differences).sum(-1))
    _require(np.isfinite(result).all(), "finite RBF covariance")
    return result


def _factor(matrix):
    _require(np.isfinite(matrix).all(), "finite covariance before Cholesky")
    try:
        result = cholesky(matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("positive-definite covariance required; no retry or extra jitter") from error
    _require(np.isfinite(result).all(), "finite Cholesky factor")
    return result


def _features(points, grid, factor, length, amplitude):
    shape = points.shape[:-1]
    cross = _kernel(points.reshape(-1, 2), grid, length, amplitude)
    return solve_triangular(factor, cross.T, lower=True, check_finite=False).T.reshape(*shape, 16)


def _new_diagnostics():
    return {"residual_points": 0, "minimum_raw_residual": None, "residual_floor_count": 0}


def _residual(features, amplitude, diagnostics):
    raw = amplitude**2 - np.square(features).sum(-1)
    _require(np.isfinite(raw).all(), "finite diagonal residual")
    minimum = float(raw.min())
    diagnostics["residual_points"] += raw.size
    old = diagnostics["minimum_raw_residual"]
    diagnostics["minimum_raw_residual"] = minimum if old is None else min(old, minimum)
    _require(minimum >= -RESIDUAL_TOLERANCE, "materially negative diagonal residual")
    diagnostics["residual_floor_count"] += int(np.count_nonzero(raw < 0))
    # Only this explicitly declared roundoff operation is permitted. No
    # conditional variance, covariance matrix or final probability is clipped.
    return np.maximum(raw, 0.)


def accumulate_information(features, targets, variances):
    """Sequential rank-one information updates, starting from N(0,I).

    Shapes are [K,N,R], [K,N], [K,N]. Each of N events contributes exactly
    once to precision and eta. Blocks are vectorized, observations are an
    explicit recurrent loop. Returns owned precision[K,R,R], eta[K,R].
    """
    for value, ndim, name in ((features, 3, "features"), (targets, 2, "targets"),
                              (variances, 2, "variances")):
        _array(value, ndim, name)
    blocks, points, rank = features.shape
    _require(targets.shape == variances.shape == (blocks, points) and (variances > 0).all(),
             "matched information inputs with positive event variances")
    precision = np.broadcast_to(np.eye(rank), (blocks, rank, rank)).copy()
    eta = np.zeros((blocks, rank), np.float64)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for index in range(points):
            phi, y, variance = features[:, index], targets[:, index], variances[:, index]
            precision += phi[:, :, None] * phi[:, None, :] / variance[:, None, None]
            eta += phi * (y / variance)[:, None]
    _require(np.isfinite(precision).all() and np.isfinite(eta).all(), "finite information statistics")
    return {"precision": precision, "eta": eta}


@dataclass(frozen=True)
class ArchiveCache:
    mode: str
    length: float
    amplitude: float
    noise_variance: float
    block_count: int
    point_count: int
    arrays: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]

    @property
    def array_bytes(self):
        """All retained arrays; excludes Python metadata and transient workspace."""
        return sum(value.nbytes for value in self.arrays.values())


def build_archive(bx, by, *, length, amplitude, mode="sor", noise_variance=NOISE_VARIANCE):
    """Fit public blocks once; no raw archive is retained by low-rank modes."""
    _array(bx, 3, "bx")
    _array(by, 2, "by")
    blocks, points, dim = bx.shape
    _require(dim == 2 and by.shape == (blocks, points), "matched two-dimensional archive shapes")
    _require(type(mode) is str and mode in MODES, "declared covariance mode")
    length, amplitude, noise = (_positive(value, name) for value, name in
                                ((length, "length"), (amplitude, "amplitude"),
                                 (noise_variance, "noise_variance")))
    _require(math.isfinite(amplitude * amplitude), "finite amplitude squared")
    diagnostics = _new_diagnostics()
    if mode == "full":
        factors, alphas = [], []
        for x, y in zip(bx, by, strict=True):
            factor = _factor(_kernel(x, x, length, amplitude) + noise * np.eye(points))
            factors.append(factor)
            alphas.append(cho_solve((factor, True), y, check_finite=False))
        arrays = {"archive_x": bx, "archive_factor": np.stack(factors), "archive_alpha": np.stack(alphas)}
    else:
        coordinates = np.linspace(-2., 2., 4)
        grid = np.array([(a, b) for a in coordinates for b in coordinates], np.float64)
        factor = _factor(_kernel(grid, grid, length, amplitude) + JITTER * np.eye(16))
        features = _features(bx, grid, factor, length, amplitude)
        event_variance = np.full((blocks, points), noise)
        if mode == "fic":
            event_variance += _residual(features, amplitude, diagnostics)
        information = accumulate_information(features, by, event_variance)
        covariance, means = [], []
        for precision, eta in zip(information["precision"], information["eta"], strict=True):
            posterior_factor = _factor(precision)
            covariance.append(cho_solve((posterior_factor, True), np.eye(16), check_finite=False))
            means.append(cho_solve((posterior_factor, True), eta, check_finite=False))
        arrays = {"inducing_grid": grid, "feature_factor": factor,
                  "posterior_covariance": np.stack(covariance), "weight_mean": np.stack(means)}
    return ArchiveCache(mode, length, amplitude, noise, blocks, points,
                        MappingProxyType({key: _owned(value) for key, value in arrays.items()}),
                        MappingProxyType(diagnostics.copy()))


def predict(cache, fx, fy, qx, *, diagnostics=None):
    """Public-only request prediction; optional empty diagnostics dict is filled.

    Joint few-shot likelihood selects uniformly-prior blocks before conditioning
    each query on those F labels once. Outputs have K components and a scalar
    noisy-response positive probability. Cache state is never updated.
    """
    _require(type(cache) is ArchiveCache, "ArchiveCache required")
    for value, ndim, name in ((fx, 2, "fx"), (fy, 1, "fy"), (qx, 1, "qx")):
        _array(value, ndim, name)
    count = len(fx)
    _require(fx.shape == (count, 2) and fy.shape == (count,) and qx.shape == (2,), "request shapes")
    if diagnostics is None:
        diagnostics = {}
    _require(type(diagnostics) is dict and not diagnostics, "diagnostics must be an empty dict")
    diagnostics.update(_new_diagnostics())
    points = np.concatenate((fx, qx[None]), axis=0)
    arrays, noise = cache.arrays, cache.noise_variance
    means, variances, evidence = [], [], []
    if cache.mode != "full":
        features = _features(points, arrays["inducing_grid"], arrays["feature_factor"],
                              cache.length, cache.amplitude)
        event_variance = np.full(count + 1, noise)
        if cache.mode == "fic":
            event_variance += _residual(features, cache.amplitude, diagnostics)
        elif cache.mode == "query_only":
            event_variance[-1:] += _residual(features[-1:], cache.amplitude, diagnostics)
    else:
        joint_prior = _kernel(points, points, cache.length, cache.amplitude) + noise * np.eye(count + 1)
    for block in range(cache.block_count):
        if cache.mode == "full":
            cross = _kernel(arrays["archive_x"][block], points, cache.length, cache.amplitude)
            prior_mean = cross.T @ arrays["archive_alpha"][block]
            whitened = solve_triangular(arrays["archive_factor"][block], cross, lower=True, check_finite=False)
            prior_covariance = joint_prior - whitened.T @ whitened
        else:
            prior_mean = features @ arrays["weight_mean"][block]
            prior_covariance = features @ arrays["posterior_covariance"][block] @ features.T
            prior_covariance += np.diag(event_variance)
        _require(np.isfinite(prior_mean).all() and np.isfinite(prior_covariance).all(), "finite request Gaussian")
        factor = _factor(prior_covariance[:count, :count])
        residual = fy - prior_mean[:count]
        standardized = solve_triangular(factor, residual, lower=True, check_finite=False)
        evidence.append(-.5 * (count * math.log(2 * math.pi) + 2 * np.log(factor.diagonal()).sum()
                              + standardized @ standardized))
        query_cross = prior_covariance[:count, -1]
        means.append(prior_mean[-1] + query_cross @ cho_solve((factor, True), residual, check_finite=False))
        whitened_query = solve_triangular(factor, query_cross, lower=True, check_finite=False)
        variances.append(prior_covariance[-1, -1] - whitened_query @ whitened_query)
    means, variances, evidence = map(np.asarray, (means, variances, evidence))
    _require(np.isfinite(means).all() and np.isfinite(variances).all() and (variances > 0).all()
             and np.isfinite(evidence).all(), "finite means/evidence and positive query variance; no clamp")
    log_weights = evidence - logsumexp(evidence)
    probability = float(np.sum(np.exp(log_weights) * ndtr(means / np.sqrt(variances))))
    _require(np.isfinite(log_weights).all() and math.isfinite(probability)
             and 0 <= probability <= 1 + 1e-12, "finite mixture probabilities with declared roundoff")
    return {"component_mean": means, "component_variance": variances,
            "log_weights": log_weights, "prob_positive": probability}
