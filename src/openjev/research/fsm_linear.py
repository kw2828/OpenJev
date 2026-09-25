"""Conventional causal VARX ridge controls for the CubeSpec development split.

Callers supply the common FIT-normalized twelve FSM FIT records. At time t the
features are y[t-p:t], u[t], u[t-p:t], and an intercept, in chronological order.
The objective is mean squared residual plus alpha times squared non-intercept
coefficients. One float64 primal Cholesky solve fits all three outputs. There
is no pole projection, jitter, clipping, period wraparound, or stability repair.

Prediction starts from the final p observed outputs and p available past inputs;
u_context contains C-1 inputs, excluding the first context sample's input, just
like the paired neural interface. Future u[h] predicts future y[h]. Every later
output uses only earlier predictions, never measured future outputs. This is
not the authors' 28-state frequency-domain BLA or a novel model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .fsm_data import AMPLITUDES, FIT_REALIZATIONS, FS_HZ, FSMRecord

VERSION = "fsm-linear-varx-v1"
ORDERS = (8, 16, 32)
ALPHAS = (1e-6, 1e-3, .1)
CHANNELS = 3
FIT_IDS = tuple(f"{a}-realization-{r}-period-{p}"
                for a in AMPLITUDES for r in FIT_REALIZATIONS for p in range(2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _order(order):
    _require(type(order) is int and 1 <= order <= 99, "integer VARX order in 1..99 required")


def _alpha(alpha):
    _require(type(alpha) in (int, float) and np.isfinite(alpha) and alpha > 0,
             "finite positive VARX alpha required")


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape and bool(np.isfinite(value).all()),
             name + " must be a finite float64 array with the declared shape")


def _owned(value):
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class VARXModel:
    coefficients: np.ndarray
    order: int
    alpha: float
    fit_rows: int
    fit_record_ids: tuple[str, ...]

    def __post_init__(self):
        _order(self.order)
        _alpha(self.alpha)
        _array(self.coefficients, (CHANNELS, 6*self.order+4), "coefficients")
        _require(type(self.fit_rows) is int and self.fit_rows > 0, "positive integer fit_rows required")
        _require(type(self.fit_record_ids) is tuple and len(self.fit_record_ids) == len(FIT_IDS)
                 and set(self.fit_record_ids) == set(FIT_IDS), "complete twelve-record FIT identity roster required")
        object.__setattr__(self, "coefficients", _owned(self.coefficients))
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def state_scalars(self):
        return 6*self.order

    def parameter_count(self):
        return int(self.coefficients.size)

    def model_spec(self):
        return {"version": VERSION, "order": self.order, "alpha": self.alpha,
                "dtype": "float64", "input_channels": 3, "output_channels": 3,
                "fit_rows": self.fit_rows, "fit_record_ids": list(self.fit_record_ids),
                "parameter_count": self.parameter_count(), "parameter_bytes": self.coefficients.nbytes,
                "scalar_metadata_bytes": 24, "retained_numeric_bytes": self.coefficients.nbytes+24,
                "state_scalars": self.state_scalars, "state_bytes_per_stream": self.state_scalars*8,
                "intercept_penalized": False, "ridge_scale": "mean Gram over all within-record fit rows",
                "feature_order": "chronological last-p y, current u, chronological last-p u, intercept",
                "solver": "float64 primal Cholesky; no jitter or stability repair",
                "training_rows_retained": False, "persistent_cache": False,
                "storage_scope": "coefficient and logical order/alpha/fit_rows payload plus separately reported request state; identifiers, Python overhead, normalization, request/output arrays and temporary fitting/prediction workspace excluded"}


def _fit_records(records, order):
    records = tuple(records)
    _require(len(records) == len(FIT_IDS) and all(type(r) is FSMRecord for r in records),
             "complete twelve FSMRecord FIT records required")
    ids = [r.record_id for r in records]
    _require(len(set(ids)) == len(FIT_IDS) and set(ids) == set(FIT_IDS),
             "complete duplicate-free twelve-record FIT identity roster required")
    for record in records:
        _require(record.amplitude in AMPLITUDES and type(record.realization) is int
                 and record.realization in FIT_REALIZATIONS and type(record.period) is int
                 and record.period in (0, 1) and record.partition == "fit" and record.fs_hz == FS_HZ
                 and record.record_id == f"{record.amplitude}-realization-{record.realization}-period-{record.period}",
                 "FIT record identity/partition/frequency mismatch")
        _require(isinstance(record.u, np.ndarray) and record.u.ndim == 2 and len(record.u) > order,
                 "FIT record must have more samples than VARX order")
        _array(record.u, (len(record.u), CHANNELS), "record u")
        _array(record.y, (len(record.u), CHANNELS), "record y")
    _require(len({len(r.u) for r in records}) == 1, "equal FIT record lengths required")
    # Canonical order avoids caller-order-dependent floating-point summation.
    by_id = {r.record_id: r for r in records}
    return tuple(by_id[key] for key in FIT_IDS)


def fit_varx(records, *, order, alpha):
    """Fit only complete normalized FIT records, without joining their boundaries.

    The nine registered candidates are ORDERS x ALPHAS. Other orders up to 99
    are supported for fabricated qualification; the experiment owns its roster.
    Normalization and source admission are the caller's responsibility.
    """
    _order(order)
    _alpha(alpha)
    records = _fit_records(records, order)
    width = 6*order+4
    gram, cross = np.zeros((width, width)), np.zeros((width, CHANNELS))
    rows = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for record in records:
            count = len(record.u)-order
            design = np.concatenate(
                [record.y[j:j+count] for j in range(order)]
                + [record.u[order:]]
                + [record.u[j:j+count] for j in range(order)]
                + [np.ones((count, 1))], axis=1)
            gram += design.T @ design
            cross += design.T @ record.y[order:]
            rows += count
        gram /= rows
        cross /= rows
    _require(bool(np.isfinite(gram).all()) and bool(np.isfinite(cross).all()),
             "nonfinite VARX sufficient statistics")
    gram[np.arange(width-1), np.arange(width-1)] += alpha
    _require(bool(np.isfinite(gram).all()), "nonfinite VARX regularized system")
    try:
        coefficients = cho_solve(cho_factor(gram, lower=True, check_finite=False), cross, check_finite=False).T
    except np.linalg.LinAlgError as error:
        raise ValueError("VARX Cholesky failed; no jitter or alpha change") from error
    _require(bool(np.isfinite(coefficients).all()), "nonfinite VARX coefficients")
    return VARXModel(coefficients, order, float(alpha), rows, FIT_IDS)


def predict(model, y_context, u_context, future_u):
    """Return owned [B,H,3] forecasts using current-input direct feedthrough.

    No model or caller input is mutated. Nonfinite recursion fails explicitly;
    finite but very large forecasts are retained for the experiment to score.
    """
    _require(type(model) is VARXModel, "VARXModel required")
    _require(isinstance(y_context, np.ndarray) and y_context.ndim == 3
             and y_context.shape[0] > 0 and y_context.shape[1] >= model.order+1,
             "context shape [positive B,C>=p+1,3] required")
    batch, context = y_context.shape[:2]
    _array(y_context, (batch, context, CHANNELS), "y_context")
    _array(u_context, (batch, context-1, CHANNELS), "u_context")
    _require(isinstance(future_u, np.ndarray) and future_u.ndim == 3,
             "future_u shape [B,H,3] required")
    _array(future_u, (batch, future_u.shape[1], CHANNELS), "future_u")
    past_y, past_u = y_context[:, -model.order:].copy(), u_context[:, -model.order:].copy()
    result = np.empty((batch, future_u.shape[1], CHANNELS), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for h in range(future_u.shape[1]):
            current_u = future_u[:, h]
            design = np.concatenate((past_y.reshape(batch, -1), current_u,
                                     past_u.reshape(batch, -1), np.ones((batch, 1))), axis=1)
            current_y = design @ model.coefficients.T
            _require(bool(np.isfinite(current_y).all()), "nonfinite VARX prediction")
            result[:, h] = current_y
            past_y = np.concatenate((past_y[:, 1:], current_y[:, None]), axis=1)
            past_u = np.concatenate((past_u[:, 1:], current_u[:, None]), axis=1)
    return result
