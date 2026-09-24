"""Seeded grid fields and exact public-history Gaussian path-risk reference.

The spatial nugget is a latent value shared by every occurrence of the same
coordinate. Independent observation noise is added only to the public stream.
Path exposure is the mean of four latent values, with no extra measurement
noise. Private realized exposures are generated for evaluation, never accepted
by the predictor. Contexts are independent; requests within a context are not.

This is known-law Gaussian conditioning, not a learned uncertainty guarantee.
There is no outcome filtering, inverse, jitter, variance clipping or fallback.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtr

VERSION = 'retention-data-v1'
GRID_SIDE = 17
GRID_POINTS = GRID_SIDE**2
LENGTH = 1.
AMPLITUDE = 1.
SPATIAL_NUGGET = 1e-5
NOISE_VARIANCE = .09
EXPOSURE_THRESHOLD = .5
PATH_COST = .02
DEFER_COST = .20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum, maximum=None):
    _require(type(value) is int and value >= minimum
             and (maximum is None or value <= maximum), name + ' outside integer domain')


def _array(value, ndim, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype('float64')
             and value.ndim == ndim and all(size > 0 for size in value.shape)
             and np.isfinite(value).all(), name + ' must be a nonempty finite float64 array')


def grid_points():
    """Owned coordinates, x index outermost and y index innermost."""
    axis = np.linspace(-2., 2., GRID_SIDE, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2).copy()


def kernel(x, z):
    """Fixed RBF plus shared-coordinate white latent component, without noise."""
    _array(x, 2, 'kernel x')
    _array(z, 2, 'kernel z')
    _require(x.shape[1] == z.shape[1] == 2, 'kernel coordinates must have dimension two')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        differences = (x[:, None, :] - z[None, :, :])/LENGTH
        result = AMPLITUDE**2*np.exp(-.5*np.sum(differences*differences, axis=-1))
        result += SPATIAL_NUGGET*np.all(x[:, None, :] == z[None, :, :], axis=-1)
    _require(np.isfinite(result).all(), 'finite kernel required')
    return result


def _path_indices(geometry):
    _require(type(geometry) is str and geometry in ('axial', 'diagonal'), 'unknown path geometry')
    directions = ((1, 0), (0, 1)) if geometry == 'axial' else ((1, 1), (1, -1))
    result = []
    for dx, dy in directions:
        for i in range(GRID_SIDE):
            for j in range(GRID_SIDE):
                points = [(i+2*t*dx, j+2*t*dy) for t in range(4)]
                if all(0 <= u < GRID_SIDE and 0 <= v < GRID_SIDE for u, v in points):
                    result.append([u*GRID_SIDE+v for u, v in points])
    return np.asarray(result, dtype=np.int64)


def path_library(geometry='axial'):
    """All 374 axial or 242 diagonal undirected four-point paths.

    Adjacent points differ by two grid steps in each moving coordinate.
    Reverse order is not a second action because exposure is an unordered mean.
    """
    return grid_points()[_path_indices(geometry)].copy()


@lru_cache(maxsize=1)
def _generation_factor():
    # Fixed prior factor only. The inference path never uses this cache.
    points = grid_points()
    factor = np.linalg.cholesky(kernel(points, points))
    factor.setflags(write=False)
    return factor


def generate(seed, contexts, observations, geometry='axial', queries=4):
    """Generate x[C,N,2], y[C,N], paths[C,Q,4,4,2], private exposure[C,Q,4].

    SeedSequence([seed, context]) spawns order, latent, observation-noise, then
    request streams. A full grid permutation and full noise vector are drawn,
    making shorter observation streams prefixes of longer ones. Four distinct
    paths are sampled uniformly per request after the public stream; requests
    may repeat paths. Context and request extensions preserve existing data.
    No global RNG or private outcome is used to select inputs or requests.
    """
    _integer(seed, 'seed', 0, 2**32-1)
    _integer(contexts, 'contexts', 1)
    _integer(observations, 'observations', 1, GRID_POINTS)
    _integer(queries, 'queries', 1)
    choices = _path_indices(geometry)
    grid = grid_points()
    factor = _generation_factor()
    x = np.empty((contexts, observations, 2), np.float64)
    y = np.empty((contexts, observations), np.float64)
    paths = np.empty((contexts, queries, 4, 4, 2), np.float64)
    exposure = np.empty((contexts, queries, 4), np.float64)
    for context in range(contexts):
        order_rng, field_rng, noise_rng, request_rng = [np.random.default_rng(child) for child in
            np.random.SeedSequence([seed, context]).spawn(4)]
        order = order_rng.permutation(GRID_POINTS)
        latent = factor @ field_rng.standard_normal(GRID_POINTS)
        noise = np.sqrt(NOISE_VARIANCE)*noise_rng.standard_normal(GRID_POINTS)
        selected = order[:observations]
        x[context] = grid[selected]
        y[context] = latent[selected]+noise[:observations]
        for query in range(queries):
            indices = choices[request_rng.choice(len(choices), size=4, replace=False)]
            paths[context, query] = grid[indices]
            exposure[context, query] = latent[indices].mean(axis=-1)
    result = {'x': x, 'y': y, 'paths': paths, 'exposure': exposure}
    _require(all(np.isfinite(value).all() for value in result.values()), 'finite generated values required')
    return result


def exact_reference(x, y, paths):
    """Target-free latent-average moments/risk under all supplied observations.

    Inputs x[C,N,2], y[C,N], paths[C,Q,4,4,2]. Outputs mean, variance and risk
    have shape [C,Q,4]. diag_variance and diag_risk discard posterior covariance
    between path points ONLY as an explicitly privileged diagnostic. They are
    not a second exact posterior. The reference accepts repeated coordinates,
    preserving shared latent values but independent noise on each observation.
    """
    for value, ndim, name in ((x, 3, 'x'), (y, 2, 'y'), (paths, 5, 'paths')):
        _array(value, ndim, name)
    contexts, observations, dimension = x.shape
    _require(dimension == 2 and y.shape == (contexts, observations)
             and paths.shape[0] == contexts and paths.shape[2:] == (4, 4, 2),
             'consistent public input shapes required')
    output_shape = paths.shape[:3]
    mean = np.empty(output_shape, np.float64)
    variance = np.empty_like(mean)
    diagonal = np.empty_like(mean)
    for context in range(contexts):
        covariance = kernel(x[context], x[context])+NOISE_VARIANCE*np.eye(observations)
        factor = np.linalg.cholesky(covariance)
        points = paths[context].reshape(-1, 2)
        cross = kernel(x[context], points)
        whitened = solve_triangular(factor, cross, lower=True, check_finite=False)
        alpha = cho_solve((factor, True), y[context], check_finite=False)
        point_means = (cross.T @ alpha).reshape(paths.shape[1], 4, 4)
        point_variances = AMPLITUDE**2+SPATIAL_NUGGET-np.sum(whitened*whitened, axis=0)
        _require(np.isfinite(point_variances).all() and (point_variances > 0).all(),
                 'positive finite point posterior variances required')
        mean[context] = point_means.mean(axis=-1)
        diagonal[context] = point_variances.reshape(paths.shape[1], 4, 4).sum(axis=-1)/16
        for query in range(paths.shape[1]):
            for action in range(4):
                start = (query*4+action)*4
                projected = whitened[:, start:start+4].mean(axis=1)
                prior_variance = kernel(paths[context, query, action], paths[context, query, action]).sum()/16
                variance[context, query, action] = prior_variance-projected @ projected
    _require(np.isfinite(mean).all() and np.isfinite(variance).all() and (variance > 0).all(),
             'positive finite exposure posterior variances required')
    risk = ndtr((mean-EXPOSURE_THRESHOLD)/np.sqrt(variance))
    diag_risk = ndtr((mean-EXPOSURE_THRESHOLD)/np.sqrt(diagonal))
    _require(np.isfinite(risk).all() and np.isfinite(diag_risk).all()
             and ((risk >= 0) & (risk <= 1)).all() and ((diag_risk >= 0) & (diag_risk <= 1)).all(),
             'finite tail probabilities required')
    return {'mean': mean, 'variance': variance, 'risk': risk,
            'diag_variance': diagonal, 'diag_risk': diag_risk}
