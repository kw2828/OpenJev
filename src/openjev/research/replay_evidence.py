"""Exact Bayesian linear regression with caller-owned evidence identities.

This established Gaussian reference treats output columns independently, with
one shared coefficient precision and known scalar observation-noise variance.
An ID denotes one physical measurement. Re-reading that same ID and byte-exact
payload changes no sufficient statistic or observation step. A different ID
denotes independent evidence even when its values happen to be identical.
The caller must supply truthful identities; independence cannot be inferred
statistically from equal values. This does not establish calibration for learned
or misspecified features.

All arrays are finite NumPy float64. Prior precision must be exactly symmetric
(absolute and relative symmetry tolerances both zero) and positive definite.
There is no symmetrization, jitter, clipping, inverse, or solver fallback.
Queries return output-space covariance; coefficient columns are independent.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'replay-evidence-v1'
SYMMETRY_ABSOLUTE_TOLERANCE = 0.
SYMMETRY_RELATIVE_TOLERANCE = 0.


def _require(value, message):
    if not value:
        raise ValueError(message)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype('float64')
             and value.shape == shape and np.isfinite(value).all(),
             name + ' must have the declared shape and finite float64 values')


def _owned(value):
    result = np.array(value, dtype=np.float64, order='C', copy=True)
    result.setflags(write=False)
    return result


def _factor(precision, information):
    try:
        factor = np.linalg.cholesky(precision)
        mean = np.linalg.solve(factor.T, np.linalg.solve(factor, information))
    except np.linalg.LinAlgError as error:
        raise ValueError('precision must admit a positive-definite Cholesky factorization') from error
    _require(np.isfinite(factor).all() and np.isfinite(mean).all(), 'finite posterior factor and mean required')
    return _owned(factor), _owned(mean)


class ReplayEvidence:
    """Owned Gaussian sufficient statistics with atomic idempotent updates.

    prior_mean[D,O], prior_precision[D,D], and scalar noise_variance specify
    theta[:,o] ~ N(prior_mean[:,o], prior_precision^-1) independently for each
    output o. For each unique (ID,x,y), y = x @ theta + independent noise.
    observation_steps counts accepted unique measurements, not wall-clock time.
    Replay, queries, snapshots and rejected inputs do not advance it.
    """

    def __init__(self, prior_mean, prior_precision, *, noise_variance):
        _require(isinstance(prior_mean, np.ndarray) and prior_mean.ndim == 2
                 and all(size > 0 for size in prior_mean.shape), 'positive input and output dimensions')
        self._dimension, self._outputs = prior_mean.shape
        _array(prior_mean, (self._dimension, self._outputs), 'prior_mean')
        _array(prior_precision, (self._dimension, self._dimension), 'prior_precision')
        _require(np.array_equal(prior_precision, prior_precision.T),
                 'prior_precision must be exactly symmetric; no hidden repair')
        _require(type(noise_variance) in (int, float) and math.isfinite(noise_variance)
                 and noise_variance > 0, 'noise_variance must be a positive finite real scalar')
        self._noise_variance = float(noise_variance)
        self._prior_mean, self._prior_precision = _owned(prior_mean), _owned(prior_precision)
        self._precision = _owned(prior_precision)
        try:
            with np.errstate(over='raise', invalid='raise'):
                self._information = _owned(prior_precision @ prior_mean)
        except FloatingPointError as error:
            raise ValueError('finite prior information required') from error
        _require(np.isfinite(self._information).all(), 'finite prior information required')
        self._cholesky, self._mean = _factor(self._precision, self._information)
        self._evidence = {}
        self._replay_events = 0

    @property
    def counts(self):
        """An owned dictionary; only accepted unique IDs advance the step."""
        unique = len(self._evidence)
        return {'unique_evidence': unique, 'replay_events': self._replay_events,
                'observation_steps': unique}

    def add(self, evidence_id, x, y):
        """Add one measurement or acknowledge its exact replay.

        Nonempty string IDs are immutable keys. Payload equality is byte-exact
        in contiguous native float64, including signed zero. Conflicts and all
        validation/numerical failures leave posterior, IDs and counts unchanged.
        """
        _require(type(evidence_id) is str and bool(evidence_id), 'evidence_id must be a nonempty string')
        _array(x, (self._dimension,), 'x')
        _array(y, (self._outputs,), 'y')
        x, y = _owned(x), _owned(y)
        if evidence_id in self._evidence:
            previous = self._evidence[evidence_id]
            _require(previous['x'].tobytes() == x.tobytes() and previous['y'].tobytes() == y.tobytes(),
                     'same evidence_id has a contradictory payload')
            self._replay_events += 1
            return {'status': 'replay', **self.counts}
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                precision = self._precision + np.outer(x, x)/self._noise_variance
                information = self._information + np.outer(x, y)/self._noise_variance
        except FloatingPointError as error:
            raise ValueError('finite measurement information required') from error
        _require(np.isfinite(precision).all() and np.isfinite(information).all(),
                 'finite measurement information required')
        factor, mean = _factor(precision, information)
        # Commit only after every operation that can reject this update.
        self._precision, self._information = _owned(precision), _owned(information)
        self._cholesky, self._mean = factor, mean
        self._evidence[evidence_id] = {'x': x, 'y': y}
        return {'status': 'added', **self.counts}

    def query(self, x):
        """Predict from x alone; no query target, hidden state or evidence ID."""
        _array(x, (self._dimension,), 'query x')
        try:
            with np.errstate(over='raise', invalid='raise'):
                mean = x @ self._mean
                whitened = np.linalg.solve(self._cholesky, x)
                variance = float(whitened @ whitened)
                predictive_variance = variance+self._noise_variance
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise ValueError('finite query moments required') from error
        _require(np.isfinite(mean).all() and math.isfinite(variance) and variance >= 0
                 and math.isfinite(predictive_variance), 'finite nonnegative query moments required')
        return {'mean': _owned(mean),
                'latent_covariance': _owned(np.eye(self._outputs)*variance),
                'predictive_covariance': _owned(np.eye(self._outputs)*predictive_variance)}

    def snapshot(self):
        """Return detached, owned arrays/dictionaries without internal aliases."""
        return {'version': VERSION, 'prior_mean': _owned(self._prior_mean),
                'prior_precision': _owned(self._prior_precision), 'noise_variance': self._noise_variance,
                'precision': _owned(self._precision), 'information': _owned(self._information),
                'posterior_mean': _owned(self._mean), 'counts': self.counts,
                'evidence': {key: {name: _owned(value) for name, value in row.items()}
                             for key, row in self._evidence.items()}}
