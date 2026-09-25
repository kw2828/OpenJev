"""Conventional causal multi-horizon ridge with no future-output input.

For horizon h (one based), predict q[C+h-1] from q[C-16:C],
u[C-17:C-1], and u[C-1:C+h-1]. The last block is future_u[:h].
Each horizon has its own six-output ridge; the intercept is penalized too.
One shared float64 Gram/cross-product is computed, then the appropriate leading
feature columns and separate final intercept are selected for 128 Cholesky
solves. There is no adaptive penalty, added jitter, inverse, RNG, or Torch.

The immutable bank retains only allowed coefficients, not a padded dense bank
or training examples. Byte counts exclude Python objects, request arrays and
transient designs/factorizations. Callers own normalization and input provenance.
This is an established regression baseline, not a novel recurrent model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

VERSION = 'causal-robot-ridge-v1'
HORIZONS, JOINTS, CONTEXT, HISTORY = 128, 6, 32, 16
HISTORY_COLUMNS = 2 * HISTORY * JOINTS
MAX_COLUMNS = HISTORY_COLUMNS + HORIZONS * JOINTS + 1


def _require(value, message):
    if not value:
        raise ValueError(message)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape and np.isfinite(value).all(),
             name + ' must be a finite float64 array with the declared shape')


def _inputs(q_context, u_context, future_u):
    _require(isinstance(q_context, np.ndarray) and q_context.ndim == 3
             and q_context.shape[0] > 0, 'nonempty context batch required')
    batch = q_context.shape[0]
    _array(q_context, (batch, CONTEXT, JOINTS), 'q_context')
    _array(u_context, (batch, CONTEXT, JOINTS), 'u_context')
    _array(future_u, (batch, HORIZONS, JOINTS), 'future_u')
    return batch


def _design(q_context, u_context, future_u):
    batch = len(q_context)
    return np.concatenate((q_context[:, -HISTORY:].reshape(batch, -1),
                           u_context[:, -HISTORY-1:-1].reshape(batch, -1),
                           future_u.reshape(batch, -1), np.ones((batch, 1))), axis=1)


def _indices(horizon):
    return np.r_[np.arange(HISTORY_COLUMNS + JOINTS * horizon), MAX_COLUMNS - 1]


def features(q_context, u_context, future_u, *, horizon):
    """Owned [B,193+6h] causal design, useful for independent verification."""
    _inputs(q_context, u_context, future_u)
    _require(type(horizon) is int and 1 <= horizon <= HORIZONS, 'integer horizon in1..128 required')
    return _design(q_context, u_context, future_u)[:, _indices(horizon)]


@dataclass(frozen=True, slots=True)
class CausalRobotRidge:
    """Owned immutable coefficients[h-1] with shape [6,193+6h]."""

    coefficients: tuple[np.ndarray, ...]
    penalty: float
    fit_rows: int

    def __post_init__(self):
        _require(type(self.penalty) in (float, int) and np.isfinite(self.penalty)
                 and self.penalty > 0, 'finite positive penalty required')
        _require(type(self.fit_rows) is int and self.fit_rows > 0, 'positive integer fit_rows required')
        _require(type(self.coefficients) is tuple and len(self.coefficients) == HORIZONS,
                 'complete tuple of128 horizon coefficients required')
        owned = []
        for h, value in enumerate(self.coefficients, 1):
            _array(value, (JOINTS, HISTORY_COLUMNS + JOINTS * h + 1), 'coefficient')
            # An immutable bytes owner prevents re-enabling writes via setflags.
            owned.append(np.frombuffer(value.tobytes(order='C'), dtype=np.float64).reshape(value.shape))
        object.__setattr__(self, 'coefficients', tuple(owned))
        object.__setattr__(self, 'penalty', float(self.penalty))

    def metadata(self):
        count = sum(value.size for value in self.coefficients)
        size = sum(value.nbytes for value in self.coefficients)
        return {'version': VERSION, 'fit_rows': self.fit_rows, 'penalty': self.penalty,
                'horizons': HORIZONS, 'outputs': JOINTS, 'dtype': 'float64',
                'coefficient_count': count, 'coefficient_bytes': size,
                'scalar_metadata_bytes': 16, 'retained_numeric_bytes': size + 16,
                'required_history_values': HISTORY_COLUMNS,
                'required_history_bytes': HISTORY_COLUMNS * 8,
                'normalizer_bytes_included': 0, 'training_rows_retained': False,
                'intercept_penalized': True, 'padded_future_coefficients_stored': False,
                'solver': 'shared Gram, indexed primal Cholesky for each horizon',
                'scope': 'numeric coefficient and scalar metadata payload; Python overhead, caller normalization, request arrays and solver workspace excluded'}


def fit(q_context, u_context, future_u, target, *, penalty=1.):
    """Fit using aligned target[B,h-1]=q[C+h-1]; target is never a predictor.

    Penalty is applied to all coefficients, including each horizon's intercept.
    A Cholesky failure or nonfinite arithmetic fails explicitly without repair.
    """
    batch = _inputs(q_context, u_context, future_u)
    _array(target, (batch, HORIZONS, JOINTS), 'target')
    _require(type(penalty) in (float, int) and np.isfinite(penalty) and penalty > 0,
             'finite positive penalty required')
    design = _design(q_context, u_context, future_u)
    with np.errstate(over='ignore', invalid='ignore'):
        gram = design.T @ design
        cross = design.T @ target.reshape(batch, HORIZONS * JOINTS)
    _require(np.isfinite(gram).all() and np.isfinite(cross).all(), 'nonfinite ridge sufficient statistics')
    coefficients = []
    for h in range(1, HORIZONS + 1):
        indices = _indices(h)
        system = gram[np.ix_(indices, indices)]
        system.flat[::len(indices)+1] += penalty
        rhs = cross[indices, (h-1)*JOINTS:h*JOINTS]
        _require(np.isfinite(system).all(), 'nonfinite penalized ridge system')
        try:
            factor = cho_factor(system, lower=True, overwrite_a=False, check_finite=False)
            result = cho_solve(factor, rhs, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise ValueError('ridge Cholesky failed; no jitter or penalty change') from error
        _require(np.isfinite(result).all(), 'nonfinite ridge coefficient')
        coefficients.append(result.T)
    return CausalRobotRidge(tuple(coefficients), float(penalty), batch)


def predict(model, q_context, u_context, future_u):
    """Return owned float64[B,128,6]; later torques cannot alter earlier outputs."""
    _require(type(model) is CausalRobotRidge, 'CausalRobotRidge model required')
    batch = _inputs(q_context, u_context, future_u)
    design = _design(q_context, u_context, future_u)
    result = np.empty((batch, HORIZONS, JOINTS), dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        for h, coefficients in enumerate(model.coefficients, 1):
            result[:, h-1] = design[:, _indices(h)] @ coefficients.T
    _require(np.isfinite(result).all(), 'nonfinite ridge prediction')
    return result
