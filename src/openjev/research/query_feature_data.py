"""Noisy RBF function archives and a public-information exact GP reference.

Each block is one coherent zero-mean Gaussian process. Coordinates and request
identities are independently drawn, but responses for requests using the same
block can remain correlated conditional on its archive. Requests are scored
separately from the archive and their own few-shots; earlier query answers are
never predictor inputs. Contexts, not requests, are independent sampling units.

Amplitude is the GP standard deviation. Observation noise is independent at
every sampled point, including query targets. The reference uses a uniform
block prior and the joint few-shot Gaussian likelihood, conditions once on
those observations, and includes observation noise in query variance. It is
established known-law Gaussian inference, not a learned-model calibration claim.
No Cholesky jitter, clipping, inverse, or numerical fallback is used.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp, ndtr

VERSION = 'query-feature-data-v1'
PREDICTION_KEYS = {'component_mean', 'component_variance', 'log_weights', 'prob_positive'}
PROBABILITY_ROUNDOFF_TOLERANCE = 1e-12


def _require(value, message):
    if not value:
        raise ValueError(message)


def _positive(value, name):
    _require(type(value) in (int, float), name + ' must be a positive finite scalar')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + ' must be representable as float64') from error
    _require(math.isfinite(result) and result > 0, name + ' must be a positive finite scalar')
    return result


def _law(length, amplitude, noise_variance):
    return tuple(_positive(value, name) for value, name in
                 ((length, 'length'), (amplitude, 'amplitude'), (noise_variance, 'noise_variance')))


def _array(value, ndim, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype('float64')
             and value.ndim == ndim and all(size > 0 for size in value.shape)
             and np.isfinite(value).all(), name + ' must be a nonempty finite float64 array')


def _kernel(x, y, length, amplitude):
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        differences = (x[..., :, None, :] - y[..., None, :, :])/length
        squared = np.sum(differences*differences, axis=-1)
        result = amplitude*amplitude*np.exp(-.5*squared)
    _require(np.isfinite(result).all(), 'finite kernel required')
    return result


def _chol(matrix):
    try:
        result = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('positive-definite covariance required; no jitter or fallback') from error
    _require(np.isfinite(result).all(), 'finite Cholesky factor required')
    return result


def _solve(factor, right):
    return np.linalg.solve(factor.swapaxes(-1, -2), np.linalg.solve(factor, right))


def generate_contexts(seed, contexts, blocks=4, basis_points=16, queries=2, fewshots=4,
                      extent=2., length=1., amplitude=1., noise_variance=.0225):
    """Generate owned arrays, keeping private IDs and noisy targets separate.

    Per-context SeedSequence([seed, context]) spawns coordinate, request, ID,
    then one function-value stream per block. Each block is jointly sampled at
    its basis and all request coordinates before private IDs select responses.
    Increasing context count preserves earlier contexts; global RNG is unused.
    """
    _require(type(seed) is int and 0 <= seed < 2**32, 'seed must be a uint32 Python integer')
    for name, value in (('contexts', contexts), ('blocks', blocks), ('basis_points', basis_points),
                        ('queries', queries), ('fewshots', fewshots)):
        _require(type(value) is int and value > 0, name + ' must be a positive Python integer')
    extent = _positive(extent, 'extent')
    length, amplitude, noise_variance = _law(length, amplitude, noise_variance)
    bx = np.empty((contexts, blocks, basis_points, 2), np.float64)
    by = np.empty((contexts, blocks, basis_points), np.float64)
    fx = np.empty((contexts, queries, fewshots, 2), np.float64)
    fy = np.empty((contexts, queries, fewshots), np.float64)
    qx = np.empty((contexts, queries, 2), np.float64)
    target = np.empty((contexts, queries), np.float64)
    identity = np.empty((contexts, queries), np.int64)
    total = basis_points+queries*(fewshots+1)
    for context in range(contexts):
        streams = [np.random.default_rng(child) for child in
                   np.random.SeedSequence([seed, context]).spawn(3+blocks)]
        bx[context] = streams[0].uniform(-extent, extent, (blocks, basis_points, 2))
        requests = streams[1].uniform(-extent, extent, (queries, fewshots+1, 2))
        fx[context], qx[context] = requests[:, :fewshots], requests[:, fewshots]
        identity[context] = streams[2].integers(0, blocks, queries, dtype=np.int64)
        points = np.concatenate((bx[context], np.broadcast_to(requests.reshape(-1, 2),
                                (blocks, queries*(fewshots+1), 2))), axis=1)
        covariance = _kernel(points, points, length, amplitude)+noise_variance*np.eye(total)
        factors = _chol(covariance)
        innovations = np.stack([stream.standard_normal(total) for stream in streams[3:]])
        values = np.einsum('kij,kj->ki', factors, innovations)
        by[context] = values[:, :basis_points]
        request_values = values[:, basis_points:].reshape(blocks, queries, fewshots+1)
        selected = request_values[identity[context], np.arange(queries)]
        fy[context], target[context] = selected[:, :fewshots], selected[:, fewshots]
    output = {'bx': bx, 'by': by, 'fx': fx, 'fy': fy, 'qx': qx, 'target': target,
              'private_selected_block': identity}
    _require(all(np.isfinite(value).all() for value in output.values()), 'finite generated context required')
    return output


def exact_gp_mixture(bx, by, fx, fy, qx, *, length=1., amplitude=1., noise_variance=.0225):
    """Target-free exact public prediction for flat batch B.

    Inputs: bx[B,K,N,2], by[B,K,N], fx[B,F,2], fy[B,F], qx[B,2].
    Outputs: mean/variance/log_weights[B,K], prob_positive[B]. Every block is
    conditioned on the archive; its JOINT F-dimensional few-shot likelihood
    determines posterior identity weights. Query responses are noisy draws.
    """
    length, amplitude, noise_variance = _law(length, amplitude, noise_variance)
    for value, ndim, name in ((bx, 4, 'bx'), (by, 3, 'by'), (fx, 3, 'fx'), (fy, 2, 'fy'), (qx, 2, 'qx')):
        _array(value, ndim, name)
    batch, blocks, basis, dimension = bx.shape
    _require(dimension == 2 and by.shape == (batch, blocks, basis)
             and fx.shape[0] == batch and fx.shape[2] == 2
             and fy.shape == fx.shape[:2] and qx.shape == (batch, 2), 'consistent public input shapes')
    fewshots = fx.shape[1]
    component_mean = np.empty((batch, blocks), np.float64)
    component_variance = np.empty((batch, blocks), np.float64)
    evidence = np.empty((batch, blocks), np.float64)
    for case in range(batch):
        points = np.concatenate((fx[case], qx[case:case+1]), axis=0)
        joint_prior = _kernel(points, points, length, amplitude)+noise_variance*np.eye(fewshots+1)
        for block in range(blocks):
            x, y = bx[case, block], by[case, block]
            archive_covariance = _kernel(x, x, length, amplitude)+noise_variance*np.eye(basis)
            archive_factor = _chol(archive_covariance)
            cross = _kernel(x, points, length, amplitude)
            conditional_mean = cross.T @ _solve(archive_factor, y)
            whitened = np.linalg.solve(archive_factor, cross)
            conditional_covariance = joint_prior-whitened.T @ whitened
            few_factor = _chol(conditional_covariance[:fewshots, :fewshots])
            residual = fy[case]-conditional_mean[:fewshots]
            standardized = np.linalg.solve(few_factor, residual)
            evidence[case, block] = -.5*(fewshots*math.log(2*math.pi)
                + 2*np.log(np.diag(few_factor)).sum()+standardized @ standardized)
            query_cross = conditional_covariance[:fewshots, -1]
            component_mean[case, block] = conditional_mean[-1]+query_cross @ _solve(few_factor, residual)
            query_whitened = np.linalg.solve(few_factor, query_cross)
            component_variance[case, block] = conditional_covariance[-1, -1]-query_whitened @ query_whitened
    _require(np.isfinite(component_mean).all() and np.isfinite(component_variance).all()
             and (component_variance > 0).all() and np.isfinite(evidence).all(), 'finite positive conditional Gaussian moments')
    log_weights = evidence-logsumexp(evidence, axis=-1, keepdims=True)
    probability = np.sum(np.exp(log_weights)*ndtr(component_mean/np.sqrt(component_variance)), axis=-1)
    # A sum of rounded normalized weights may exceed one slightly when every
    # component CDF saturates at one. Permit only the declared roundoff guard;
    # retain the computed value without clipping or a hidden renormalization.
    _require(np.isfinite(log_weights).all() and np.isfinite(probability).all()
             and ((probability >= 0) & (probability <= 1+PROBABILITY_ROUNDOFF_TOLERANCE)).all(),
             'finite normalized mixture probabilities')
    return {'component_mean': component_mean, 'component_variance': component_variance,
            'log_weights': log_weights, 'prob_positive': probability}


def mixture_log_prob(prediction, target):
    """Score targets after prediction, with no assimilation or routing change."""
    _require(type(prediction) is dict and set(prediction) == PREDICTION_KEYS, 'exact mixture prediction schema')
    mean, variance, weights = (prediction[key] for key in ('component_mean', 'component_variance', 'log_weights'))
    for value, name in ((mean, 'component_mean'), (variance, 'component_variance'), (weights, 'log_weights')):
        _array(value, 2, name)
    _array(target, 1, 'target')
    _array(prediction['prob_positive'], 1, 'prob_positive')
    _require(mean.shape == variance.shape == weights.shape and target.shape == mean.shape[:1]
             and prediction['prob_positive'].shape == target.shape and (variance > 0).all()
             and np.allclose(logsumexp(weights, axis=-1), 0., atol=PROBABILITY_ROUNDOFF_TOLERANCE, rtol=0.)
             and ((prediction['prob_positive'] >= 0)
                  & (prediction['prob_positive'] <= 1+PROBABILITY_ROUNDOFF_TOLERANCE)).all(),
             'valid normalized Gaussian mixture with declared roundoff guard')
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        components = -.5*(math.log(2*math.pi)+np.log(variance)+(target[:, None]-mean)**2/variance)
        result = logsumexp(weights+components, axis=-1)
    _require(np.isfinite(result).all(), 'finite target log density')
    return result


def reference_predict(data, *, length=1., amplitude=1., noise_variance=.0225):
    """Convenient C-by-Q scored reference; private block identities are unread.

    Calls the target-free predictor on public arrays, then scores target only.
    No other request's observed answer is supplied to the current request.
    """
    _require(isinstance(data, dict) and {'bx', 'by', 'fx', 'fy', 'qx', 'target'} <= set(data), 'public arrays plus separate target required')
    bx, by, fx, fy, qx, target = (data[name] for name in ('bx', 'by', 'fx', 'fy', 'qx', 'target'))
    for value, ndim, name in ((bx, 4, 'bx'), (by, 3, 'by'), (fx, 4, 'fx'), (fy, 3, 'fy'), (qx, 3, 'qx'), (target, 2, 'target')):
        _array(value, ndim, name)
    contexts, queries, fewshots, dimension = fx.shape
    _require(dimension == 2 and bx.shape[0] == contexts and bx.shape[-1] == 2
             and by.shape == bx.shape[:3] and fy.shape == (contexts, queries, fewshots)
             and qx.shape == (contexts, queries, 2) and target.shape == (contexts, queries), 'context/query dimensions')
    prediction = exact_gp_mixture(np.repeat(bx, queries, axis=0), np.repeat(by, queries, axis=0),
        fx.reshape(contexts*queries, fewshots, 2), fy.reshape(contexts*queries, fewshots),
        qx.reshape(contexts*queries, 2), length=length, amplitude=amplitude, noise_variance=noise_variance)
    score = mixture_log_prob(prediction, target.reshape(-1))
    output = {name: value.reshape((contexts, queries)+value.shape[1:]).copy() for name, value in prediction.items()}
    output['log_prob'] = score.reshape(contexts, queries).copy()
    return output
