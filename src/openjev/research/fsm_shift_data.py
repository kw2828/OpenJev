"""Restricted physical-value reader for a preregistered 300 mV amplitude check.

Authentication covers the whole opaque snapshot before NumPy inspects its ZIP
directory or any array header. Only the two 300 mV TRAIN members are decoded;
their names do not authorize training. No normalizer is fitted or applied here.
Public windows and targets have separate APIs and independently owned storage.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

VERSION = "fsm-shift-data-v1"
NATIVE_SHAPE = (8192, 3, 6, 2)
FS_HZ = 6400.0
CONTEXT = 100
HORIZON = 128
SHIFT_VARIABLES = ("u_300mV_train", "y_300mV_train")
ARCHIVE_VARIABLES = frozenset(f"{v}_{a}_{p}" for a in ("100mV", "200mV", "300mV")
                              for p in ("train", "test") for v in ("u", "y"))
RECORD_IDS = tuple(f"300mV-realization-{r}-period-{p}" for r in range(6) for p in range(2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _schema(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(np.float64),
             f"{name} must be a native float64 array")
    _require(value.shape == shape, f"unexpected {name} shape")
    return value


def _finite(value, name):
    _require(bool(np.isfinite(value).all()), f"nonfinite {name}")
    return value


def _owned(value):
    # An immutable bytes owner prevents callers from re-enabling writes.
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class ShiftRecord:
    u: np.ndarray
    y: np.ndarray
    amplitude: str
    realization: int
    period: int
    partition: str
    record_id: str
    fs_hz: float = FS_HZ


@dataclass(frozen=True, slots=True)
class ShiftData:
    records: tuple[ShiftRecord, ...]
    source_sha256: str
    source_bytes: int
    native_shape: tuple[int, ...] = NATIVE_SHAPE
    decoded_keys: tuple[str, ...] = SHIFT_VARIABLES
    fs_hz: float = FS_HZ


@dataclass(frozen=True, slots=True)
class ShiftInputs:
    y_context: np.ndarray
    transition_context_u: np.ndarray
    future_u: np.ndarray
    record_id: str
    start: int

    def public_inputs(self):
        """Forecast inputs only; pass the two context fields to an initializer."""
        return MappingProxyType({"y_context": self.y_context,
                                 "transition_context_u": self.transition_context_u,
                                 "future_u": self.future_u})


def read_npz_shift(path, *, expected_sha256, expected_bytes):
    """Authenticate once, then decode exactly the two allowed snapshot members.

    Expectations must be supplied by the caller's frozen registration. Tests may
    supply expectations for fabricated archives; no default admits a real source.
    Other known member names may be listed, but their headers/values stay closed.
    """
    _require(isinstance(expected_sha256, str)
             and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "expected_sha256 must be a lowercase SHA256 digest")
    _require(type(expected_bytes) is int and expected_bytes > 0, "positive expected_bytes required")
    blob = Path(path).read_bytes()
    _require(len(blob) == expected_bytes, "snapshot byte-size mismatch before decoding")
    actual_sha256 = hashlib.sha256(blob).hexdigest()
    _require(actual_sha256 == expected_sha256, "snapshot SHA256 mismatch before decoding")
    _require(zipfile.is_zipfile(io.BytesIO(blob)), "an NPZ archive is required")
    archive = np.load(io.BytesIO(blob), allow_pickle=False)
    _require(isinstance(archive, np.lib.npyio.NpzFile), "an NPZ archive is required")
    with archive:
        names = archive.files
        _require(len(names) == len(set(names)) and set(names) <= ARCHIVE_VARIABLES
                 and set(SHIFT_VARIABLES) <= set(names), "unexpected or duplicate NPZ member roster")
        values = {key: _finite(_schema(archive[key], NATIVE_SHAPE, key), key) for key in SHIFT_VARIABLES}
    records = tuple(ShiftRecord(_owned(values[SHIFT_VARIABLES[0]][:, :, r, p]),
                                _owned(values[SHIFT_VARIABLES[1]][:, :, r, p]),
                                "300mV", r, p, "confirmation", RECORD_IDS[2*r+p])
                    for r in range(6) for p in range(2))
    return ShiftData(records, actual_sha256, len(blob))


def _window_bounds(record, start):
    _require(isinstance(record, ShiftRecord) and record.amplitude == "300mV",
             "a 300mV ShiftRecord is required")
    _require(type(record.realization) is int and 0 <= record.realization < 6
             and type(record.period) is int and 0 <= record.period < 2, "invalid record identity")
    _require(record.record_id == RECORD_IDS[2*record.realization+record.period]
             and record.partition == "confirmation" and record.fs_hz == FS_HZ,
             "record identity/partition/frequency mismatch")
    _schema(record.u, NATIVE_SHAPE[:2], "record u")
    _schema(record.y, NATIVE_SHAPE[:2], "record y")
    _require(type(start) is int and start >= 0, "nonnegative integer start required")
    _require(start + CONTEXT + HORIZON <= NATIVE_SHAPE[0], "window crosses period boundary")
    return start + CONTEXT


def public_window(record, start):
    """C100 observed outputs, 99 aligned past inputs, and H128 future inputs.

    The unavailable u[start] is omitted. This function never slices or checks
    future output values. Values are physical, with no fitted normalization.
    """
    end = _window_bounds(record, start)
    y = _owned(_finite(record.y[start:end], "arrived context outputs"))
    past = _owned(_finite(record.u[start+1:end], "past context inputs"))
    future = _owned(_finite(record.u[end:end+HORIZON], "future inputs"))
    return ShiftInputs(y, past, future, record.record_id, start)


def target_window(record, start):
    """H128 future outputs for a scorer, to be called after prediction returns."""
    end = _window_bounds(record, start)
    return _owned(_finite(record.y[end:end+HORIZON], "future target outputs"))
