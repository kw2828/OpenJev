"""Raw-only, causal loader for the Industrial Robot conditional pilot.

This deliberately differs from the official concatenated, zero-phase prepared
benchmark. Each recording has its own causal filter state. Positions are degrees;
the six measured total motor torques are Nm. No velocities or reference signals
are predictors. Only the five required variables are numerically decoded.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import butter, sosfilt, sosfilt_zi

VERSION = "industrial-robot-data-v1"
RAW_SAMPLES = 90_881
RAW_HZ = 250.0
STRIDE = 25
FILTER_ORDER = 4
CUTOFF_HZ = 4.0
TIME_ATOL_SECONDS = 1e-10
_TIMES = (
    "20H_29M", "20H_38M", "20H_45M", "21H_11M", "21H_32M",
    "21H_3M", "21H_41M", "21H_54M", "22H_10M", "22H_41M", "22H_50M",
)
TRAIN_RECORDINGS = tuple(f"recording_2021_12_15_{t}.mat" for t in _TIMES)
OFFICIAL_TEST = "recording_2021_12_15_22H_58M.mat"
PARTITIONS = MappingProxyType({
    "fit": TRAIN_RECORDINGS[:7],
    "dev": TRAIN_RECORDINGS[7:9],
    "confirm": TRAIN_RECORDINGS[9:],
})
_MATRIX_NAMES = (
    "q_mot_meas", "q_se_meas", "qd_mot_meas", "qd_se_meas", "tau_meas",
    "tau_fb_meas", "q_ref", "qd_ref", "tau_ref_ff",
)
_HEADERS = {name: ((6, RAW_SAMPLES), "double") for name in _MATRIX_NAMES}
_HEADERS.update({
    "recording_ok": ((1, 1), "logical"),
    "time": ((1, RAW_SAMPLES), "double"),
    "READ_ME": ((17, 2), "cell"),
})
_DECODE = ("q_se_meas", "q_mot_meas", "tau_meas", "time", "recording_ok")
PREPROCESSING = MappingProxyType({
    "version": VERSION,
    "raw_samples": RAW_SAMPLES,
    "raw_hz": RAW_HZ,
    "filter": "Butterworth lowpass SOS, scipy.signal.butter/sosfilt",
    "filter_order": FILTER_ORDER,
    "cutoff_hz": CUTOFF_HZ,
    "initial_state": "sosfilt_zi scaled by each channel's first sample",
    "stride": STRIDE,
    "first_raw_index": 0,
    "q": "q_se_meas first three rows; q_mot_meas last three rows; degrees",
    "torque": "tau_meas all six rows; total measured motor torque; Nm",
    "time_atol_seconds": TIME_ATOL_SECONDS,
})
PREPROCESSING_SHA256 = hashlib.sha256(
    json.dumps(dict(PREPROCESSING), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _owned_readonly(value: np.ndarray) -> np.ndarray:
    """Use an immutable bytes owner, so callers cannot re-enable writes."""
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class RobotRecording:
    q: np.ndarray
    torque: np.ndarray
    raw_indices: np.ndarray
    name: str
    partition: str
    pins: Mapping[str, str | int]


def recording_partition(name: str, *, allow_confirmation: bool = False) -> str:
    """Check the explicit roster before any source file is opened."""
    _require(type(allow_confirmation) is bool, "allow_confirmation must be bool")
    _require(type(name) is str and name in TRAIN_RECORDINGS,
             "name must be an allowed raw TRAIN basename; official TEST is forbidden")
    partition = next(key for key, names in PARTITIONS.items() if name in names)
    _require(partition != "confirm" or allow_confirmation,
             "confirmation requires explicit allow_confirmation=True")
    return partition


def _source_bytes(source: bytes | str | Path) -> bytes:
    if type(source) is bytes:
        blob = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        _require(not path.is_symlink() and path.is_file(), "source must be a regular file")
        blob = path.read_bytes()
    else:
        raise ValueError("source must be MAT bytes or a file path")
    _require(bool(blob), "source MAT bytes are empty")
    return blob


def _causal_filter(channels: np.ndarray) -> np.ndarray:
    """Filter [sample, channel] using only each channel's own past and present."""
    sos = butter(FILTER_ORDER, CUTOFF_HZ, btype="lowpass", fs=RAW_HZ, output="sos")
    zi = sosfilt_zi(sos)[:, :, None] * channels[0][None, None, :]
    filtered, _ = sosfilt(sos, channels, axis=0, zi=zi)
    _require(np.isfinite(filtered).all(), "filtered signal is nonfinite")
    return filtered


def load_recording(
    source: bytes | str | Path,
    name: str,
    *,
    allow_confirmation: bool = False,
) -> RobotRecording:
    """Load one whitelisted recording, with owned immutable decimated arrays.

    The caller must authenticate these returned source pins against its registered
    source inventory. A name alone does not authenticate arbitrary supplied bytes.
    Confirmation access is explicit; the official TEST has no supported API.
    Unused variables are checked by header only and never numerically decoded.
    """
    partition = recording_partition(name, allow_confirmation=allow_confirmation)
    blob = _source_bytes(source)
    try:
        headers = whosmat(io.BytesIO(blob))
    except Exception as exc:
        raise ValueError("invalid MAT header") from exc
    _require(len(headers) == len(_HEADERS), "unexpected raw MAT variable roster")
    actual = {key: (tuple(shape), kind) for key, shape, kind in headers}
    _require(actual == _HEADERS, "unexpected raw MAT fields, shapes or classes")
    try:
        # mat_dtype=False preserves complex values for explicit rejection. The
        # header independently establishes that recording_ok is MATLAB logical.
        data = loadmat(io.BytesIO(blob), variable_names=list(_DECODE),
                       mat_dtype=False, squeeze_me=False, verify_compressed_data_integrity=True)
    except Exception as exc:
        raise ValueError("invalid selected MAT variables") from exc
    for key in _DECODE[:-1]:
        value = data.get(key)
        _require(isinstance(value, np.ndarray) and value.dtype.kind == "f"
                 and value.dtype.itemsize == 8 and value.shape == _HEADERS[key][0],
                 f"{key} must be a real float64 array of the registered shape")
        _require(np.isfinite(value).all(), f"{key} must be finite")
    flag = data.get("recording_ok")
    _require(isinstance(flag, np.ndarray) and flag.shape == (1, 1)
             and flag.dtype.kind in "bu" and flag.item() == 1,
             "recording_ok must be logical true")
    time = data["time"][0]
    # This bound accommodates float64 timestamp subtraction at this recording
    # length; it is not a fitted tolerance from measured timestamp values.
    _require(np.allclose(np.diff(time), 1.0 / RAW_HZ, rtol=0.0,
                         atol=TIME_ATOL_SECONDS),
             "time must be strictly uniform at 250 Hz")
    q = np.concatenate((data["q_se_meas"][:3], data["q_mot_meas"][3:]), axis=0).T
    signals = np.concatenate((q, data["tau_meas"].T), axis=1)
    filtered = _causal_filter(signals)
    indices = np.arange(0, RAW_SAMPLES, STRIDE, dtype=np.int64)
    selected = filtered[indices]
    pins = MappingProxyType({
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        "source_bytes": len(blob),
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "loader_version": VERSION,
    })
    return RobotRecording(
        q=_owned_readonly(selected[:, :6]),
        torque=_owned_readonly(selected[:, 6:]),
        raw_indices=_owned_readonly(indices), name=name, partition=partition, pins=pins,
    )
