"""Two-amplitude estimation-only CubeSpec Fine Steering Mirror adapter.

Native axes are (sample, channel, realization, period). Each period remains a
separate chronological record. Both amplitudes use realizations 0..2 for FIT
and 3..5 for DEV, preserving the author's orthogonal realization triplets.
No 300mV or official-test measurement member is decoded. NPZ names and original
bytes may be inspected opaquely for schema and provenance.

The prospective convention pairs physical u(k) with y(k). Initializing from
y[s] skips u[s]; context transitions use u[s+1:s+C], then future u[s+C:s+C+H]
predicts y[s+C:s+C+H]. No period averaging, resampling or imputation is used.
The documented shape is fixed; float64 is an admission requirement, not a
claim established by inspecting the actual measurement array headers.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

VERSION = "fsm-data-v1"
REPOSITORY_COMMIT = "539a12fef384b086a8562b500498b2fa3899ef70"
DOWNLOAD_URL = ("https://raw.githubusercontent.com/merijnfloren/fsm-benchmark-data/"
                + REPOSITORY_COMMIT + "/data/combined_data.npz")
DATA_LICENSE = "CC-BY-4.0"
NATIVE_SHAPE = (8192, 3, 6, 2)
FS_HZ = 6400.0
AMPLITUDES = ("100mV", "200mV")
ESTIMATION_VARIABLES = tuple(f"{v}_{a}_train" for a in AMPLITUDES for v in ("u", "y"))
ARCHIVE_VARIABLES = frozenset(f"{v}_{a}_{p}" for a in (*AMPLITUDES, "300mV")
                              for p in ("train", "test") for v in ("u", "y"))
FIT_REALIZATIONS = (0, 1, 2)
DEV_REALIZATIONS = (3, 4, 5)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(value):
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(order="C"), dtype=np.float64).reshape(a.shape)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.dtype.itemsize == 8,
             f"{name} must be a real float64 array")
    _require(value.shape == shape, f"unexpected {name} shape")
    _require(bool(np.isfinite(value).all()), f"nonfinite {name}")
    return value


@dataclass(frozen=True, slots=True)
class FSMRecord:
    u: np.ndarray
    y: np.ndarray
    amplitude: str
    realization: int
    period: int
    partition: str
    record_id: str
    fs_hz: float = FS_HZ


@dataclass(frozen=True, slots=True)
class EstimationData:
    records: tuple[FSMRecord, ...]
    native_shape: tuple[int, ...]
    source_sha256: str | None
    source_bytes: int | None
    decoded_keys: tuple[str, ...] = ESTIMATION_VARIABLES
    fs_hz: float = FS_HZ

    def partition(self, name):
        _require(name in ("fit", "dev"), "only fit/dev estimation partitions exist")
        return tuple(r for r in self.records if r.partition == name)


@dataclass(frozen=True, slots=True)
class Normalizer:
    u_mean: np.ndarray
    u_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
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
        """Only arrived observations and correctly aligned applied inputs."""
        return MappingProxyType({"y_context": self.y_context,
                                 "transition_context_u": self.transition_context_u,
                                 "future_u": self.future_u})


def _records(mapping, shape, source_sha256=None, source_bytes=None):
    _require(isinstance(mapping, dict) and set(mapping) == set(ESTIMATION_VARIABLES),
             "exact four permitted estimation variables required")
    for key in ESTIMATION_VARIABLES:
        _array(mapping[key], shape, key)
    _require(len({mapping[k].dtype for k in ESTIMATION_VARIABLES}) == 1,
             "all estimation dtypes must match")
    records = []
    for amplitude in AMPLITUDES:
        u, y = mapping[f"u_{amplitude}_train"], mapping[f"y_{amplitude}_train"]
        for realization in range(6):
            for period in range(2):
                record_id = f"{amplitude}-realization-{realization}-period-{period}"
                records.append(FSMRecord(_owned(u[:, :, realization, period]),
                                         _owned(y[:, :, realization, period]),
                                         amplitude, realization, period,
                                         "fit" if realization in FIT_REALIZATIONS else "dev", record_id))
    return EstimationData(tuple(records), shape, source_sha256, source_bytes)


def records_from_fixture(mapping):
    """Fabricated small-N helper only; production never calls this geometry path."""
    _require(isinstance(mapping, dict) and isinstance(mapping.get(ESTIMATION_VARIABLES[0]), np.ndarray),
             "fixture requires first estimation array")
    shape = mapping[ESTIMATION_VARIABLES[0]].shape
    _require(len(shape) == 4 and 2 <= shape[0] < NATIVE_SHAPE[0] and shape[1:] == NATIVE_SHAPE[1:],
             "fixture geometry must be [2<=N<8192,3,6,2]")
    return _records(mapping, shape)


def read_npz_estimation(path):
    """Lazy-decode only the four allowed TRAIN arrays from one hashed snapshot.

    Known 300mV and test member names may appear in the central directory, but
    __getitem__ is never called for them. Pickle loading is disabled. A dtype or
    native geometry discrepancy fails instead of silently coercing the source.
    """
    blob = Path(path).read_bytes()
    archive = np.load(io.BytesIO(blob), allow_pickle=False)
    _require(isinstance(archive, np.lib.npyio.NpzFile), "an NPZ archive is required")
    with archive:
        names = archive.files
        _require(len(names) == len(set(names)) and set(names) <= ARCHIVE_VARIABLES
                 and set(ESTIMATION_VARIABLES) <= set(names), "unexpected or duplicate NPZ member roster")
        mapping = {key: archive[key] for key in ESTIMATION_VARIABLES}
    return _records(mapping, NATIVE_SHAPE, hashlib.sha256(blob).hexdigest(), len(blob))


def _record(record):
    _require(isinstance(record, FSMRecord) and record.amplitude in AMPLITUDES,
             "permitted FSMRecord required")
    _require(type(record.realization) is int and 0 <= record.realization < 6
             and type(record.period) is int and 0 <= record.period < 2, "invalid record identity")
    expected = f"{record.amplitude}-realization-{record.realization}-period-{record.period}"
    _require(record.record_id == expected
             and record.partition == ("fit" if record.realization in FIT_REALIZATIONS else "dev")
             and record.fs_hz == FS_HZ, "record identity/partition/frequency mismatch")
    _require(isinstance(record.u, np.ndarray) and record.u.ndim == 2 and len(record.u) >= 2,
             "record inputs must have shape [N>=2,3]")
    _array(record.u, (len(record.u), 3), "record u")
    _array(record.y, (len(record.u), 3), "record y")


def fit_normalizer(records):
    """Channel-wise population mean/std from the complete twelve FIT records only."""
    records = tuple(records)
    _require(all(isinstance(r, FSMRecord) for r in records), "FSM records required")
    chosen = tuple(r for r in records if r.partition == "fit")
    expected = {(a, r, p) for a in AMPLITUDES for r in FIT_REALIZATIONS for p in range(2)}
    identities = [(r.amplitude, r.realization, r.period) for r in chosen]
    _require(len(identities) == len(expected) and set(identities) == expected,
             "complete duplicate-free twelve-record FIT roster required")
    for r in chosen:
        _record(r)
    _require(len({len(r.u) for r in chosen}) == 1, "FIT sample counts must match")
    u, y = (np.concatenate([getattr(r, key) for r in chosen], axis=0) for key in ("u", "y"))
    values = (u.mean(axis=0), u.std(axis=0, ddof=0), y.mean(axis=0), y.std(axis=0, ddof=0))
    _require(all(np.isfinite(v).all() for v in values)
             and (values[1] > 0).all() and (values[3] > 0).all(), "finite positive FIT scales required")
    return Normalizer(*map(_owned, values), tuple(r.record_id for r in chosen), len(u))


def make_window(record, start, *, context=100, horizon=128, normalizer=None):
    """One chronological window within one period, with no future target inputs."""
    _record(record)
    _require(type(start) is int and start >= 0 and type(context) is int and context >= 1
             and type(horizon) is int and horizon >= 1, "integer nonnegative start/positive lengths required")
    _require(start + context + horizon <= len(record.u), "window crosses period boundary")
    u_mean, u_scale, y_mean, y_scale = np.zeros(3), np.ones(3), np.zeros(3), np.ones(3)
    if normalizer is not None:
        _require(isinstance(normalizer, Normalizer), "FIT Normalizer required")
        u_mean, u_scale, y_mean, y_scale = tuple(_array(getattr(normalizer, k), (3,), k)
                                               for k in ("u_mean", "u_scale", "y_mean", "y_scale"))
        _require((u_scale > 0).all() and (y_scale > 0).all(), "positive normalization scales required")
    end = start + context
    arrays = ((record.y[start:end]-y_mean)/y_scale,
              (record.u[start+1:end]-u_mean)/u_scale,
              (record.u[end:end+horizon]-u_mean)/u_scale,
              (record.y[end:end+horizon]-y_mean)/y_scale)
    _require(all(np.isfinite(a).all() for a in arrays), "nonfinite normalized window")
    return Window(*map(_owned, arrays), record.record_id, start)
