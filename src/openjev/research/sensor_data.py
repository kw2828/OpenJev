"""Guarded UCI Air Quality parsing, without fitting or implicit row removal.

Dates are naive source-local wall-clock hours since 1970-01-01. No timezone or
DST conversion is inferred. Source row IDs are 1-based data rows after the
header; missing measurements and missing hours are never silently dropped.

``inspect_metadata`` exposes no measurement values. ``load_train`` converts
only the fixed March-June 2004 segment to numerical data. Evaluation requires
an explicit, byte-pinned registration with the admission contract below. That
contract is an access guard, not proof of a completed experimental qualification;
the caller must authenticate any required original process closures separately.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np

VERSION = "sensor-data-v1"
INPUT_COLUMNS = ("PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
                 "PT08.S5(O3)", "T", "RH")
TARGET_COLUMN = "C6H6(GT)"
COLUMNS = (
    "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
    "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
    "PT08.S5(O3)", "T", "RH", "AH",
)
SEGMENTS = MappingProxyType({
    "train": ("2004-03-01T00:00:00", "2004-07-01T00:00:00"),
    "jul_aug": ("2004-07-01T00:00:00", "2004-09-01T00:00:00"),
    "sep_oct": ("2004-09-01T00:00:00", "2004-11-01T00:00:00"),
    "nov_dec": ("2004-11-01T00:00:00", "2005-01-01T00:00:00"),
    "jan_final": ("2005-01-01T00:00:00", None),
})
_EPOCH = datetime(1970, 1, 1)  # noqa: DTZ001 - source timezone is unspecified, not UTC.
_NUMBER = re.compile(r"[+-]?\d+(?:,\d+)?\Z")
_MISSING = re.compile(r"-200(?:,0+)?\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _owned(value, dtype):
    array = np.asarray(value, dtype=dtype)
    # bytearray forces a distinct allocation even for Python's one-byte singleton bytes.
    buffer = bytes(bytearray(array.tobytes()))
    return np.frombuffer(buffer, dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class SensorRows:
    timestamp_hours: np.ndarray
    row_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    valid_inputs: np.ndarray
    valid_target: np.ndarray
    segment: str
    csv_sha256: str
    registration_sha256: str | None

    @property
    def valid_rows(self):
        """Owned joint validity mask; callers must explicitly decide gap handling."""
        return _owned(self.valid_inputs.all(axis=1) & self.valid_target, np.bool_)


def _scan(csv_bytes):
    _require(type(csv_bytes) is bytes and bool(csv_bytes), "CSV must be nonempty bytes")
    try:
        reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline=""),
                            delimiter=";", strict=True)
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("invalid UTF-8 semicolon CSV") from error
    _require(bool(rows), "CSV has no header")
    header = tuple(cell.strip() for cell in rows[0])
    _require(header[:len(COLUMNS)] == COLUMNS, "unexpected CSV column schema")
    _require(all(not cell for cell in header[len(COLUMNS):]), "unexpected extra columns")
    records, trailing_blanks, gaps = [], 0, []
    previous = None
    for row_id, source in enumerate(rows[1:], start=1):
        cells = tuple(cell.strip() for cell in source)
        if not any(cells):
            trailing_blanks += 1
            continue
        _require(trailing_blanks == 0, "blank row inside dated sequence")
        _require(len(cells) == len(header), f"row {row_id}: column count differs")
        _require(all(not cell for cell in cells[len(COLUMNS):]), f"row {row_id}: extra values")
        _require(bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", cells[0])) and
                 bool(re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", cells[1])),
                 f"row {row_id}: invalid date/time format")
        try:
            stamp = datetime.strptime(cells[0] + " " + cells[1], "%d/%m/%Y %H.%M.%S")  # noqa: DTZ007
        except ValueError as error:
            raise ValueError(f"row {row_id}: invalid date/time") from error
        _require(stamp.minute == stamp.second == 0, f"row {row_id}: timestamp is not an hour")
        if previous is not None:
            delta = int((stamp - previous).total_seconds() // 3600)
            _require(delta > 0, f"row {row_id}: duplicate or nonchronological timestamp")
            if delta != 1:
                gaps.append({"previous_row_id": row_id - 1, "next_row_id": row_id,
                             "previous_timestamp": previous.isoformat(),
                             "next_timestamp": stamp.isoformat(), "missing_hours": delta - 1})
        previous = stamp
        records.append((row_id, stamp, cells))
    _require(bool(records), "CSV has no dated rows")
    return header, records, trailing_blanks, gaps


def _is_missing(text):
    return not text or bool(_MISSING.fullmatch(text))


def _in_segment(stamp, segment):
    lower, upper = SEGMENTS[segment]
    return stamp >= datetime.fromisoformat(lower) and (upper is None or stamp < datetime.fromisoformat(upper))


def inspect_metadata(csv_bytes):
    """Only schema, chronology and missingness; no label conversion/distribution."""
    header, records, trailing, gaps = _scan(csv_bytes)
    missing = {name: sum(_is_missing(cells[i]) for _, _, cells in records)
               for i, name in enumerate(COLUMNS[2:], start=2)}
    segments = {}
    indices = tuple(COLUMNS.index(name) for name in (*INPUT_COLUMNS, TARGET_COLUMN))
    for segment in SEGMENTS:
        selected = [r for r in records if _in_segment(r[1], segment)]
        segments[segment] = {
            "rows": len(selected),
            "valid_inputs_and_target": sum(not any(_is_missing(cells[i]) for i in indices)
                                           for _, _, cells in selected),
        }
    return {"version": VERSION, "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "csv_bytes": len(csv_bytes), "columns": list(COLUMNS),
            "empty_trailing_columns": len(header) - len(COLUMNS),
            "dated_rows": len(records), "blank_trailing_rows": trailing,
            "first_timestamp": records[0][1].isoformat(), "last_timestamp": records[-1][1].isoformat(),
            "source_timezone": "unspecified naive local time", "row_id_origin": "1-based after header",
            "gaps": gaps, "missing_measurements": missing, "segments": segments,
            "unassigned_rows": sum(not any(_in_segment(stamp, s) for s in SEGMENTS)
                                   for _, stamp, _ in records)}


def _number(text, row_id, name):
    if _is_missing(text):
        return np.nan, False
    _require(bool(_NUMBER.fullmatch(text)), f"row {row_id}: invalid decimal-comma value in {name}")
    value = float(text.replace(",", "."))
    _require(np.isfinite(value), f"row {row_id}: nonfinite value in {name}")
    return value, True


def _load(csv_bytes, segment, registration_sha256=None):
    _, records, _, _ = _scan(csv_bytes)
    selected = [r for r in records if _in_segment(r[1], segment)]
    _require(bool(selected), f"segment {segment} has no rows")
    hours, ids, inputs, targets, valid_x, valid_y = [], [], [], [], [], []
    names = (*INPUT_COLUMNS, TARGET_COLUMN)
    for row_id, stamp, cells in selected:
        converted = [_number(cells[COLUMNS.index(name)], row_id, name) for name in names]
        hours.append(int((stamp - _EPOCH).total_seconds() // 3600))
        ids.append(row_id)
        inputs.append([p[0] for p in converted[:-1]])
        targets.append(converted[-1][0])
        valid_x.append([p[1] for p in converted[:-1]])
        valid_y.append(converted[-1][1])
    return SensorRows(_owned(hours, np.int64), _owned(ids, np.int64), _owned(inputs, np.float64),
                      _owned(targets, np.float64), _owned(valid_x, np.bool_), _owned(valid_y, np.bool_),
                      segment, hashlib.sha256(csv_bytes).hexdigest(), registration_sha256)


def load_train(csv_bytes):
    """Expose only 2004-03-01 <= timestamp < 2004-07-01, with all rows retained."""
    return _load(csv_bytes, "train")


def load_registered_evaluation(csv_bytes, *, segment, registration_path, registration_sha256):
    """Load one held-out segment only after matching an explicit registration.

    The byte-pinned JSON must contain ``sensor_data_admission`` with exactly
    ``mode: registered_evaluation``, ``csv_sha256`` and a unique ``segments`` list
    chosen from jul_aug/sep_oct/nov_dec/jan_final. No TRAIN loader option enables
    this path, and there is no implicit/default held-out segment.
    """
    _require(type(segment) is str and segment in SEGMENTS and segment != "train",
             "invalid evaluation segment")
    _require(type(csv_bytes) is bytes, "CSV must be bytes")
    _require(type(registration_sha256) is str and bool(_HASH.fullmatch(registration_sha256)),
             "invalid registration SHA256")
    path = Path(registration_path)
    _require(path.is_file() and not path.is_symlink(), "registration must be a regular nonsymlink file")
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == registration_sha256, "registration hash mismatch")
    try:
        plan = json.loads(data)
    except (UnicodeError, ValueError) as error:
        raise ValueError("invalid registration JSON") from error
    admission = plan.get("sensor_data_admission") if type(plan) is dict else None
    _require(type(admission) is dict and set(admission) == {"mode", "csv_sha256", "segments"},
             "missing or invalid sensor_data_admission")
    _require(admission["mode"] == "registered_evaluation", "evaluation mode is not registered")
    _require(admission["csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest(), "CSV hash mismatch")
    segments = admission["segments"]
    _require(type(segments) is list and bool(segments) and all(type(s) is str for s in segments),
             "invalid registered segment list")
    _require(len(set(segments)) == len(segments) and all(s in SEGMENTS and s != "train" for s in segments),
             "invalid registered segments")
    _require(segment in segments, "segment is not admitted")
    return _load(csv_bytes, segment, registration_sha256)
