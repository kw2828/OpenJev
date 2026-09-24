"""Fixed TRAIN-only Silverbox CSV partitions, with no official-test loader.

Raw indices are zero-based data-row positions, excluding the CSV header. The
official v1.0 loader defines TRAIN/validation as [40650:105712]. This module
converts measurements only within one of three prospectively fixed subsets of
that interval. All other measurement strings are checked only for CSV structure,
never converted to numbers, normalized, summarized or returned.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

VERSION = "silverbox-data-v1"
EXPECTED_ROWS = 131072
SAMPLING_FREQUENCY = 610.35
OFFICIAL_TRAIN = (40650, 105712)
OFFICIAL_TESTS = MappingProxyType({
    "multisine": (105712, 127400),
    "arrow_full": (100, 40575),
    "arrow_no_extrapolation": (100, 32100),
})
PARTITIONS = MappingProxyType({
    "fit": (40650, 83946),
    "dev_a": (84446, 92638),
    "dev_b": (93138, 101330),
})
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(bytes(bytearray(array.tobytes())), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class SilverboxRows:
    raw_indices: np.ndarray
    u: np.ndarray
    y: np.ndarray
    partition: str
    csv_sha256: str
    sampling_time: float


def _scan(csv_bytes):
    _require(type(csv_bytes) is bytes and bool(csv_bytes), "CSV must be nonempty bytes")
    try:
        reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader)
        _require(header == ["V1", "V2", ""], "unexpected Silverbox CSV header")
        rows, trailing_blanks = [], 0
        for source in reader:
            if not any(cell.strip() for cell in source):
                trailing_blanks += 1
                continue
            _require(trailing_blanks == 0, "blank row inside data sequence")
            _require(len(source) == 3 and source[2].strip() == "", "unexpected CSV row structure")
            rows.append((source[0].strip(), source[1].strip()))
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError("invalid UTF-8 comma CSV") from error
    _require(len(rows) == EXPECTED_ROWS, "unexpected Silverbox data-row count")
    return rows, trailing_blanks


def inspect_metadata(csv_bytes):
    """Return only schema, fixed row boundaries and file identity, no values."""
    rows, trailing = _scan(csv_bytes)
    return {"version": VERSION, "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "csv_bytes": len(csv_bytes), "columns": ["V1", "V2"],
            "empty_trailing_columns": 1, "data_rows": len(rows), "blank_trailing_rows": trailing,
            "sampling_frequency_hz": SAMPLING_FREQUENCY, "raw_index_origin": "zero-based after header",
            "official_train_bounds": list(OFFICIAL_TRAIN),
            "official_test_bounds": {key: list(bounds) for key, bounds in OFFICIAL_TESTS.items()},
            "partitions": {key: {"start": lo, "stop": hi, "rows": hi - lo}
                           for key, (lo, hi) in PARTITIONS.items()},
            "measurement_inspection": "none; CSV structure and row count only"}


def _number(text, index, column):
    _require(bool(_DECIMAL.fullmatch(text)), f"raw row {index}: invalid or missing {column}")
    value = float(text)
    _require(np.isfinite(value), f"raw row {index}: nonfinite {column}")
    return value


def load_partition(csv_bytes, partition):
    """Load only fit/dev_a/dev_b; this API cannot return official test rows.

    No missing-value removal, interpolation, centering or scaling is performed.
    Invalid or nonfinite values inside the selected partition fail explicitly.
    Each returned array owns an immutable buffer independent of the input.
    """
    _require(type(partition) is str and partition in PARTITIONS, "unknown TRAIN partition")
    rows, _ = _scan(csv_bytes)
    lo, hi = PARTITIONS[partition]
    u, y = [], []
    for index in range(lo, hi):
        u_text, y_text = rows[index]
        u.append(_number(u_text, index, "V1"))
        y.append(_number(y_text, index, "V2"))
    return SilverboxRows(_owned(np.arange(lo, hi), np.int64), _owned(u, np.float64),
                         _owned(y, np.float64), partition, hashlib.sha256(csv_bytes).hexdigest(),
                         1.0 / SAMPLING_FREQUENCY)
