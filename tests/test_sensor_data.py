"""Fabricated CSV and admission fixtures; no empirical file or held-out values."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research import sensor_data as data


def row(date="10/03/2004", time="18.00.00", **overrides):
    fields = dict(zip(data.COLUMNS, [date, time, "91", "11", "92", "2,5", "12", "93",
                                   "13", "94", "14", "15", "16,5", "17,5", "0,4"], strict=True))
    fields.update(overrides)
    return [fields[name] for name in data.COLUMNS] + ["", ""]


def encoded(rows, header=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";", lineterminator="\r\n")
    writer.writerow(list(data.COLUMNS) + ["", ""] if header is None else header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def register(tmp_path, blob, **overrides):
    admission = {"mode": "registered_evaluation", "csv_sha256": hashlib.sha256(blob).hexdigest(),
                 "segments": ["jul_aug"]}
    admission.update(overrides)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({"sensor_data_admission": admission}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_decimal_comma_seven_public_columns_and_source_hours():
    rows = data.load_train(encoded([row(), row(time="19.00.00")]))
    np.testing.assert_array_equal(rows.x, [[11, 12, 13, 14, 15, 16.5, 17.5]] * 2)
    np.testing.assert_array_equal(rows.y, [2.5, 2.5])
    np.testing.assert_array_equal(rows.row_ids, [1, 2])
    np.testing.assert_array_equal(np.diff(rows.timestamp_hours), [1])
    expected = np.datetime64("2004-03-10T18", "h").astype(np.int64)
    assert rows.timestamp_hours[0] == expected
    assert rows.x.dtype == rows.y.dtype == np.float64
    assert rows.row_ids.dtype == rows.timestamp_hours.dtype == np.int64
    assert rows.valid_inputs.shape == (2, 7) and rows.valid_rows.all()
    assert rows.segment == "train" and rows.registration_sha256 is None
    assert not any("(GT)" in name for name in data.INPUT_COLUMNS)


def test_sentinel_and_empty_rows_preserved_and_masks_separate():
    blob = encoded([row(**{"PT08.S2(NMHC)": "-200", "T": "", "C6H6(GT)": "-200,00"}),
                    row(time="19.00.00", **{"C6H6(GT)": "-199,9"}),
                    row(time="20.00.00", **{name: "-200" for name in data.COLUMNS[2:]})])
    got = data.load_train(blob)
    assert got.x.shape == (3, 7)
    np.testing.assert_array_equal(got.valid_inputs[0], [True, False, True, True, True, False, True])
    np.testing.assert_array_equal(got.valid_target, [False, True, False])
    np.testing.assert_array_equal(got.valid_rows, [False, True, False])
    assert np.isnan(got.x[0, 1]) and np.isnan(got.y[0]) and got.y[1] == -199.9
    metadata = data.inspect_metadata(blob)
    assert metadata["missing_measurements"]["C6H6(GT)"] == 2
    assert metadata["segments"]["train"] == {"rows": 3, "valid_inputs_and_target": 1}


def test_gaps_preserved_as_elapsed_hours_with_original_row_ids():
    blob = encoded([row(), row(time="21.00.00")])
    got = data.load_train(blob)
    np.testing.assert_array_equal(got.row_ids, [1, 2])
    np.testing.assert_array_equal(np.diff(got.timestamp_hours), [3])
    assert data.inspect_metadata(blob)["gaps"] == [{
        "previous_row_id": 1, "next_row_id": 2,
        "previous_timestamp": "2004-03-10T18:00:00", "next_timestamp": "2004-03-10T21:00:00",
        "missing_hours": 2}]


def test_only_blank_trailing_rows_skipped():
    blob = encoded([row(), [""] * 17, [], [" "] * 17])
    assert data.inspect_metadata(blob)["blank_trailing_rows"] == 3
    assert data.load_train(blob).x.shape == (1, 7)
    with pytest.raises(ValueError, match="blank row inside"):
        data.inspect_metadata(encoded([row(), [], row(time="19.00.00")]))


def test_metadata_never_converts_measurements_or_returns_them(monkeypatch):
    blob = encoded([row(), row("01/07/2004", "00.00.00", **{"C6H6(GT)": "hidden-label"})])
    monkeypatch.setattr(data, "_number", lambda *args: pytest.fail("metadata converted a measurement"))
    result = data.inspect_metadata(blob)
    assert result["dated_rows"] == 2 and result["segments"]["jul_aug"]["rows"] == 1
    assert "hidden-label" not in json.dumps(result)


def test_train_does_not_convert_or_return_heldout_labels_or_features():
    heldout = {name: "forbidden-heldout-number" for name in (*data.INPUT_COLUMNS, data.TARGET_COLUMN)}
    blob = encoded([row("29/02/2004", "23.00.00", **heldout), row("01/03/2004", "00.00.00"),
                    row("30/06/2004", "23.00.00"), row("01/07/2004", "00.00.00", **heldout)])
    got = data.load_train(blob)
    np.testing.assert_array_equal(got.row_ids, [2, 3])
    np.testing.assert_array_equal(got.y, [2.5, 2.5])
    assert got.x.shape == (2, 7)


def test_calendar_segments_are_fixed_half_open_and_january_extends_to_actual_end():
    dates = ["30/06/2004", "01/07/2004", "31/08/2004", "01/09/2004", "31/10/2004",
             "01/11/2004", "31/12/2004", "01/01/2005", "04/04/2005"]
    metadata = data.inspect_metadata(encoded([row(date, "00.00.00") for date in dates]))
    assert {key: entry["rows"] for key, entry in metadata["segments"].items()} == {
        "train": 1, "jul_aug": 2, "sep_oct": 2, "nov_dec": 2, "jan_final": 2}
    assert metadata["unassigned_rows"] == 0


@pytest.mark.parametrize("second", [row(), row(time="17.00.00")])
def test_duplicate_and_reversed_dates_rejected(second):
    with pytest.raises(ValueError, match="duplicate or nonchronological"):
        data.load_train(encoded([row(), second]))


@pytest.mark.parametrize("date,time", [("31/02/2004", "00.00.00"), ("10/03/2004", "18.30.00"),
                                      ("2004-03-10", "18.00.00"), ("10/03/2004", "18:00:00")])
def test_invalid_or_nonhour_date_rejected(date, time):
    with pytest.raises(ValueError, match="date|timestamp"):
        data.inspect_metadata(encoded([row(date, time)]))


@pytest.mark.parametrize("text", ["nan", "inf", "1.25", "1e3", "1,2,3", "9" * 400])
def test_train_invalid_numeric_values_fail_instead_of_coercion(text):
    with pytest.raises(ValueError, match="value"):
        data.load_train(encoded([row(**{"C6H6(GT)": text})]))


def test_schema_and_row_width_are_exact():
    wrong = list(data.COLUMNS) + ["", ""]
    wrong[10] = "PT08.S4(NO)"
    with pytest.raises(ValueError, match="schema"):
        data.inspect_metadata(encoded([row()], header=wrong))
    with pytest.raises(ValueError, match="column count"):
        data.inspect_metadata(encoded([row()[:-1]]))
    extra = row(); extra[-1] = "unexpected"
    with pytest.raises(ValueError, match="extra values"):
        data.inspect_metadata(encoded([extra]))


def test_owned_immutable_arrays_and_no_invented_imputation():
    blob = encoded([row()])
    first, second = data.load_train(blob), data.load_train(blob)
    for name in ("timestamp_hours", "row_ids", "x", "y", "valid_inputs", "valid_target", "valid_rows"):
        a, b = getattr(first, name), getattr(second, name)
        assert not np.shares_memory(a, b)
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        first.segment = "jul_aug"
    with pytest.raises(TypeError):
        data.load_train(blob, segment="jul_aug")


def test_explicit_registered_evaluation_returns_only_admitted_segment(tmp_path):
    blob = encoded([row(), row("01/07/2004", "00.00.00", **{"C6H6(GT)": "3,5"}),
                    row("01/09/2004", "00.00.00", **{"C6H6(GT)": "4,5"})])
    path, digest = register(tmp_path, blob)
    got = data.load_registered_evaluation(blob, segment="jul_aug", registration_path=path,
                                          registration_sha256=digest)
    np.testing.assert_array_equal(got.row_ids, [2])
    np.testing.assert_array_equal(got.y, [3.5])
    assert got.registration_sha256 == digest
    with pytest.raises(ValueError, match="not admitted"):
        data.load_registered_evaluation(blob, segment="sep_oct", registration_path=path,
                                        registration_sha256=digest)


@pytest.mark.parametrize("change,message", [({"mode": "preview"}, "mode"),
                                             ({"csv_sha256": "0" * 64}, "CSV hash"),
                                             ({"segments": ["jul_aug", "jul_aug"]}, "segments"),
                                             ({"segments": ["train"]}, "segments"),
                                             ({"segments": []}, "segment list"),
                                             ({"extra": True}, "admission")])
def test_registration_guards_run_before_numeric_conversion(tmp_path, monkeypatch, change, message):
    blob = encoded([row("01/07/2004", "00.00.00")])
    path, digest = register(tmp_path, blob, **change)
    monkeypatch.setattr(data, "_load", lambda *args: pytest.fail("unadmitted numeric access"))
    with pytest.raises(ValueError, match=message):
        data.load_registered_evaluation(blob, segment="jul_aug", registration_path=path,
                                        registration_sha256=digest)


def test_registration_hash_and_nonsymlink_identity_required(tmp_path):
    blob = encoded([row("01/07/2004", "00.00.00")])
    path, digest = register(tmp_path, blob)
    with pytest.raises(ValueError, match="registration hash"):
        data.load_registered_evaluation(blob, segment="jul_aug", registration_path=path,
                                        registration_sha256="0" * 64)
    alias = tmp_path / "link.json"; alias.symlink_to(path)
    with pytest.raises(ValueError, match="nonsymlink"):
        data.load_registered_evaluation(blob, segment="jul_aug", registration_path=alias,
                                        registration_sha256=digest)
    path.write_text("{}")
    with pytest.raises(ValueError, match="registration hash"):
        data.load_registered_evaluation(blob, segment="jul_aug", registration_path=path,
                                        registration_sha256=digest)


def test_empty_segment_and_nonbytes_rejected():
    with pytest.raises(ValueError, match="no rows"):
        data.load_train(encoded([row("01/07/2004", "00.00.00")]))
    with pytest.raises(ValueError, match="bytes"):
        data.inspect_metadata("not bytes")
    with pytest.raises(ValueError, match="no dated rows"):
        data.inspect_metadata(encoded([]))
