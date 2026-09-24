"""Fabricated CSV only; no downloaded measurements or official-test decoding."""
from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research import silverbox_data as data


def fixture(partition="dev_a", overrides=None, trailing="\n", header="V1,V2,\n"):
    rows = ["poison-u,poison-y,\n"] * data.EXPECTED_ROWS
    lo, hi = data.PARTITIONS[partition]
    rows[lo:hi] = ["+1.25,-2.5e-2,\n"] * (hi - lo)
    for index, text in (overrides or {}).items():
        rows[index] = text
    return (header + "".join(rows) + trailing).encode()


@pytest.mark.parametrize("partition,lo,hi", [("fit", 40650, 83946), ("dev_a", 84446, 92638),
                                           ("dev_b", 93138, 101330)])
def test_fixed_selected_indices_only_convert_and_boundaries_are_half_open(partition, lo, hi):
    blob = fixture(partition, {lo: "3.5,-4.25,\n", hi - 1: "-6.25,7.125,\n"})
    got = data.load_partition(blob, partition)
    assert got.partition == partition
    np.testing.assert_array_equal(got.raw_indices, np.arange(lo, hi))
    np.testing.assert_array_equal(got.u[[0, -1]], [3.5, -6.25])
    np.testing.assert_array_equal(got.y[[0, -1]], [-4.25, 7.125])
    np.testing.assert_array_equal(got.u[1:-1], np.full(hi - lo - 2, 1.25))
    assert got.u.dtype == got.y.dtype == np.float64 and got.raw_indices.dtype == np.int64
    assert got.csv_sha256 == hashlib.sha256(blob).hexdigest()
    assert got.sampling_time == 1 / 610.35
    assert data.OFFICIAL_TRAIN[0] <= lo < hi <= data.OFFICIAL_TRAIN[1]


def test_metadata_exposes_no_numeric_values_and_preserves_official_roster(monkeypatch):
    blob = fixture()
    monkeypatch.setattr(data, "_number", lambda *args: pytest.fail("metadata numeric conversion"))
    got = data.inspect_metadata(blob)
    assert got["data_rows"] == 131072 and got["blank_trailing_rows"] == 1
    assert got["official_train_bounds"] == [40650, 105712]
    assert got["official_test_bounds"] == {"multisine": [105712, 127400],
                                           "arrow_full": [100, 40575],
                                           "arrow_no_extrapolation": [100, 32100]}
    assert got["partitions"] == {"fit": {"start": 40650, "stop": 83946, "rows": 43296},
                                  "dev_a": {"start": 84446, "stop": 92638, "rows": 8192},
                                  "dev_b": {"start": 93138, "stop": 101330, "rows": 8192}}


@pytest.mark.parametrize("partition", ["test", "train", "arrow_full", "all", "dev", None, 0])
def test_no_official_test_or_all_rows_access_path(partition, monkeypatch):
    monkeypatch.setattr(data, "_scan", lambda *args: pytest.fail("invalid partition scanned"))
    with pytest.raises(ValueError, match="TRAIN partition"):
        data.load_partition(b"ignored", partition)


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "", "missing", "1,25", "9" * 400])
def test_invalid_selected_values_rejected_without_dropping_rows(text):
    blob = fixture(overrides={84446: f"1.0,{text},\n"})
    with pytest.raises(ValueError, match="invalid|nonfinite|structure"):
        data.load_partition(blob, "dev_a")


def test_no_numeric_parse_of_official_test_unused_train_or_other_dev(monkeypatch):
    blob = fixture("dev_a")
    calls = []
    original = data._number

    def checked(text, index, column):
        assert 84446 <= index < 92638
        calls.append((index, column))
        return original(text, index, column)

    monkeypatch.setattr(data, "_number", checked)
    got = data.load_partition(blob, "dev_a")
    assert len(calls) == 2 * 8192 and got.y.shape == (8192,)
    assert calls[0] == (84446, "V1") and calls[-1] == (92637, "V2")


def test_structure_checks_do_not_silently_shift_indices():
    with pytest.raises(ValueError, match="blank row inside"):
        data.inspect_metadata(fixture(overrides={123: "\n"}))
    with pytest.raises(ValueError, match="structure"):
        data.inspect_metadata(fixture(overrides={123: "u,y,unexpected\n"}))
    with pytest.raises(ValueError, match="structure"):
        data.inspect_metadata(fixture(overrides={123: "u,y\n"}))
    with pytest.raises(ValueError, match="header"):
        data.inspect_metadata(fixture(header="V2,V1,\n"))
    with pytest.raises(ValueError, match="row count"):
        data.inspect_metadata(b"V1,V2,\n1,2,\n")


def test_trailing_blank_lines_bom_and_crlf_are_supported():
    blob = b"\xef\xbb\xbf" + fixture(trailing="\n,,\n").replace(b"\n", b"\r\n")
    assert data.inspect_metadata(blob)["blank_trailing_rows"] == 2
    assert len(data.load_partition(blob, "dev_a").u) == 8192


def test_arrays_are_immutable_owned_and_not_normalized():
    blob = fixture()
    a, b = data.load_partition(blob, "dev_a"), data.load_partition(blob, "dev_a")
    for name in ("raw_indices", "u", "y"):
        first, second = getattr(a, name), getattr(b, name)
        assert not np.shares_memory(first, second)
        with pytest.raises(ValueError):
            first.setflags(write=True)
    assert a.u[0] == 1.25 and a.y[0] == -.025
    with pytest.raises(FrozenInstanceError):
        a.partition = "test"
    with pytest.raises(TypeError):
        data.PARTITIONS["test"] = (0, 10)


@pytest.mark.parametrize("blob", [b"", "not bytes", b"\xff"])
def test_empty_nonbytes_and_invalid_encoding_rejected(blob):
    with pytest.raises(ValueError):
        data.inspect_metadata(blob)
