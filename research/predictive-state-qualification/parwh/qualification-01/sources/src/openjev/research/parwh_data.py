"""Estimation-only Parallel Wiener-Hammerstein records and causal windows.

Native axes are (sample, period, phase, amplitude). Periods are separate
chronological records, never flattened together. The fixed development split
groups both periods and every amplitude by phase: FIT 0..14, DEV 15..19.
There is no official-test loader. The MAT reader requests only estimation
variables; original bytes may be hashed opaquely for provenance.

The physical convention is y(k) depends on u(k). Initializing from y[s] omits
u[s]; each later conditioning transition receives u[s+1], ..., u[s+C-1].
The first forecast receives u[s+C] and predicts y[s+C].
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

VERSION = "parwh-data-v1"
NATIVE_SHAPE = (16384, 2, 20, 5)
ESTIMATION_VARIABLES = ("uEst", "yEst", "fs", "amp")
FIT_PHASES = tuple(range(15))
DEV_PHASES = tuple(range(15, 20))
_MAT_METADATA = frozenset(("__header__", "__version__", "__globals__"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(value):
    """Independent immutable bytes backing, including against setflags(write=True)."""
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(order="C"), dtype=np.float64).reshape(a.shape)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.dtype.itemsize == 8,
             f"{name} must be a real float64 array")
    _require(value.shape == shape, f"unexpected {name} shape")
    _require(bool(np.isfinite(value).all()), f"nonfinite {name}")
    return value


@dataclass(frozen=True, slots=True)
class ParWHRecord:
    u: np.ndarray
    y: np.ndarray
    phase: int
    amplitude_index: int
    period: int
    amplitude: float
    fs_hz: float
    partition: str
    record_id: str


@dataclass(frozen=True, slots=True)
class EstimationData:
    records: tuple[ParWHRecord, ...]
    amplitudes: np.ndarray
    fs_hz: float
    native_shape: tuple[int, ...]
    source_sha256: str | None
    source_bytes: int | None

    def partition(self, name):
        _require(name in ("fit", "dev"), "only fit/dev estimation partitions exist")
        return tuple(r for r in self.records if r.partition == name)


@dataclass(frozen=True, slots=True)
class Normalizer:
    u_mean: float
    u_scale: float
    y_mean: float
    y_scale: float
    fit_record_ids: tuple[str, ...]
    fit_samples: int


@dataclass(frozen=True, slots=True)
class Window:
    y_context: np.ndarray
    transition_context_u: np.ndarray
    future_u: np.ndarray
    target: np.ndarray
    record_id: str
    start: int

    def public_inputs(self):
        """No target or original recording in the predictor boundary."""
        return MappingProxyType({"y_context": self.y_context,
                                 "transition_context_u": self.transition_context_u,
                                 "future_u": self.future_u})


def _records(mapping, shape, source_sha256=None, source_bytes=None):
    _require(isinstance(mapping, dict), "estimation variables must be a dict")
    _require(set(mapping) - _MAT_METADATA == set(ESTIMATION_VARIABLES),
             "exact estimation variables required; unknown or official-test keys forbidden")
    u, y = (_array(mapping[key], shape, key) for key in ("uEst", "yEst"))
    _require(u.dtype == y.dtype, "uEst/yEst dtypes must match")
    fs = float(_array(mapping["fs"], (1, 1), "fs")[0, 0])
    amps = _array(mapping["amp"], (1, 5), "amp")[0]
    _require(fs > 0 and bool((amps > 0).all()) and bool((np.diff(amps) > 0).all()),
             "positive sampling frequency and increasing positive amplitudes required")
    records = []
    for phase in range(20):
        for amplitude_index in range(5):
            for period in range(2):
                record_id = f"phase-{phase:02d}-amp-{amplitude_index:02d}-period-{period}"
                records.append(ParWHRecord(
                    _owned(u[:, period, phase, amplitude_index, None]),
                    _owned(y[:, period, phase, amplitude_index, None]),
                    phase, amplitude_index, period, float(amps[amplitude_index]), fs,
                    "fit" if phase in FIT_PHASES else "dev", record_id))
    return EstimationData(tuple(records), _owned(amps), fs, shape, source_sha256, source_bytes)


def records_from_fixture(mapping):
    """Fabricated fixtures only: small sample axis, unchanged period/phase/amplitude axes.

    This helper is never called by the production reader and cannot relax its
    fixed 16,384-sample shape check. It must not be used for measured admission.
    """
    _require(isinstance(mapping, dict) and isinstance(mapping.get("uEst"), np.ndarray),
             "fixture requires uEst array")
    shape = mapping["uEst"].shape
    _require(len(shape) == 4 and shape[0] >= 2 and shape[1:] == NATIVE_SHAPE[1:],
             "fixture geometry must be [N>=2,2,20,5]")
    return _records(mapping, shape)


def read_mat_estimation(path):
    """Decode exactly uEst/yEst/fs/amp with fixed production geometry.

    No validation/test measurement key is requested or returned. Load from the
    same immutable byte snapshot that is hashed, preventing a hash/read race.
    No interpolation, period averaging, resampling or missing-row removal.
    """
    from scipy.io import loadmat

    blob = Path(path).read_bytes()
    mapping = loadmat(io.BytesIO(blob), variable_names=ESTIMATION_VARIABLES,
                      appendmat=False, verify_compressed_data_integrity=True)
    return _records(mapping, NATIVE_SHAPE, hashlib.sha256(blob).hexdigest(), len(blob))


def _record(record):
    _require(isinstance(record, ParWHRecord), "ParWHRecord required")
    _require(type(record.phase) is int and 0 <= record.phase < 20
             and type(record.amplitude_index) is int and 0 <= record.amplitude_index < 5
             and type(record.period) is int and 0 <= record.period < 2, "invalid record identity")
    expected_id = f"phase-{record.phase:02d}-amp-{record.amplitude_index:02d}-period-{record.period}"
    _require(record.record_id == expected_id
             and record.partition == ("fit" if record.phase in FIT_PHASES else "dev"),
             "record identity/partition mismatch")
    _require(isinstance(record.u, np.ndarray) and record.u.ndim == 2 and record.u.shape[0] >= 2,
             "record inputs must have shape [N>=2,1]")
    _array(record.u, (len(record.u), 1), "record u")
    _array(record.y, (len(record.u), 1), "record y")
    _require(np.isfinite(record.fs_hz) and record.fs_hz > 0
             and np.isfinite(record.amplitude) and record.amplitude > 0, "invalid record units")


def fit_normalizer(records):
    """Population mean/std over exactly all FIT records; DEV values are not read.

    All samples, periods and amplitudes contribute. Zero scales fail instead of
    receiving an arbitrary floor. No validation-derived statistics are allowed.
    """
    records = tuple(records)
    _require(all(isinstance(r, ParWHRecord) for r in records), "records required")
    chosen = tuple(r for r in records if r.partition == "fit")
    expected = {(p, a, t) for p in FIT_PHASES for a in range(5) for t in range(2)}
    identities = [(r.phase, r.amplitude_index, r.period) for r in chosen]
    _require(len(identities) == len(expected) and set(identities) == expected,
             "complete duplicate-free FIT record roster required")
    for r in chosen:
        _record(r)
    _require(len({(len(r.u), r.fs_hz) for r in chosen}) == 1, "FIT record geometry/sampling mismatch")
    u, y = (np.concatenate([getattr(r, key) for r in chosen], axis=0) for key in ("u", "y"))
    values = (float(u.mean()), float(u.std(ddof=0)), float(y.mean()), float(y.std(ddof=0)))
    _require(all(np.isfinite(v) for v in values) and values[1] > 0 and values[3] > 0,
             "finite FIT normalization with positive scales required")
    return Normalizer(*values, tuple(r.record_id for r in chosen), len(u))


def make_window(record, start, *, context=50, horizon=256, normalizer=None):
    """Return one within-period window; predictors receive public_inputs() only."""
    _record(record)
    _require(type(start) is int and start >= 0 and type(context) is int and context >= 1
             and type(horizon) is int and horizon >= 1, "integer nonnegative start/positive lengths required")
    _require(start + context + horizon <= len(record.u), "window crosses period boundary")
    u_mean, u_scale, y_mean, y_scale = 0.0, 1.0, 0.0, 1.0
    if normalizer is not None:
        _require(isinstance(normalizer, Normalizer), "FIT Normalizer required")
        u_mean, u_scale, y_mean, y_scale = (normalizer.u_mean, normalizer.u_scale,
                                          normalizer.y_mean, normalizer.y_scale)
        _require(all(np.isfinite(v) for v in (u_mean, u_scale, y_mean, y_scale))
                 and u_scale > 0 and y_scale > 0, "invalid normalization")
    end = start + context
    y_context = (record.y[start:end] - y_mean) / y_scale
    context_u = (record.u[start+1:end] - u_mean) / u_scale
    future_u = (record.u[end:end+horizon] - u_mean) / u_scale
    target = (record.y[end:end+horizon] - y_mean) / y_scale
    _require(all(np.isfinite(a).all() for a in (y_context, context_u, future_u, target)),
             "nonfinite normalized window")
    return Window(*map(_owned, (y_context, context_u, future_u, target)), record.record_id, start)
