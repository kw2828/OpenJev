"""Fabricated raw MAT files only; no official measurement values are loaded."""

from __future__ import annotations

import hashlib
import io
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.io import savemat
from scipy.signal import butter, lfilter, lfilter_zi

from openjev.research import industrial_robot_data as subject

FIT = subject.TRAIN_RECORDINGS[0]


def encode(fields):
    stream = io.BytesIO()
    savemat(stream, fields, do_compression=True)
    return stream.getvalue()


@pytest.fixture(scope="module")
def fields():
    n = subject.RAW_SAMPLES
    zeros = np.zeros((6, n), dtype=np.float64)
    result = {name: zeros for name in (
        "qd_mot_meas", "qd_se_meas", "tau_fb_meas", "q_ref", "qd_ref", "tau_ref_ff",
    )}
    result.update({
        "q_mot_meas": np.broadcast_to(np.arange(11.0, 17.0)[:, None], (6, n)).copy(),
        "q_se_meas": np.broadcast_to(np.arange(21.0, 27.0)[:, None], (6, n)).copy(),
        "tau_meas": np.broadcast_to(np.arange(31.0, 37.0)[:, None], (6, n)).copy(),
        "time": np.arange(n, dtype=np.float64)[None, :] / 250.0,
        "recording_ok": np.array([[True]], dtype=np.bool_),
        "READ_ME": np.full((17, 2), "fabricated documentation", dtype=object),
    })
    return result


@pytest.fixture(scope="module")
def blob(fields):
    return encode(fields)


def test_sensor_selection_units_shape_stride_and_dc_initialization(blob):
    result = subject.load_recording(blob, FIT)
    assert result.name == FIT and result.partition == "fit"
    assert result.q.shape == result.torque.shape == (3636, 6)
    assert result.q.dtype == result.torque.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result.raw_indices, np.arange(0, 90881, 25))
    expected_q = np.broadcast_to([21., 22., 23., 14., 15., 16.], result.q.shape)
    expected_tau = np.broadcast_to(np.arange(31., 37.), result.torque.shape)
    np.testing.assert_allclose(result.q, expected_q, atol=2e-11, rtol=0)
    np.testing.assert_allclose(result.torque, expected_tau, atol=2e-11, rtol=0)


def test_independent_direct_form_filter_and_current_sample_decimation(fields):
    modified = dict(fields)
    signal = np.zeros(subject.RAW_SAMPLES)
    signal[125:500] = 1.0
    signal[500:] = -0.5
    modified["tau_meas"] = fields["tau_meas"].copy()
    modified["tau_meas"][0] = signal
    result = subject.load_recording(encode(modified), FIT)
    b, a = butter(4, 4.0, fs=250.0, btype="lowpass")
    reference, _ = lfilter(b, a, signal, zi=lfilter_zi(b, a) * signal[0])
    np.testing.assert_allclose(result.torque[:, 0], reference[::25], atol=3e-10, rtol=3e-10)
    assert result.torque[4, 0] == 0.0
    assert result.torque[5, 0] > 0.0  # The current sample may affect its own output.


def test_future_values_do_not_change_past_and_recordings_reset(fields, blob):
    modified = dict(fields)
    modified["tau_meas"] = fields["tau_meas"].copy()
    modified["tau_meas"][:, 1000:] += 17.0
    original = subject.load_recording(blob, FIT)
    changed = subject.load_recording(encode(modified), FIT)
    np.testing.assert_array_equal(original.torque[:40], changed.torque[:40])
    assert not np.array_equal(original.torque[40:], changed.torque[40:])
    again = subject.load_recording(blob, subject.TRAIN_RECORDINGS[7])
    assert again.partition == "dev"
    np.testing.assert_array_equal(original.q, again.q)
    np.testing.assert_array_equal(original.torque, again.torque)


def test_exact_partition_roster_and_lexical_order():
    assert tuple(map(len, subject.PARTITIONS.values())) == (7, 2, 2)
    assert subject.TRAIN_RECORDINGS == tuple(sorted(subject.TRAIN_RECORDINGS))
    assert subject.TRAIN_RECORDINGS[4].endswith("21H_32M.mat")
    assert subject.TRAIN_RECORDINGS[5].endswith("21H_3M.mat")
    for index, name in enumerate(subject.TRAIN_RECORDINGS):
        expected = "fit" if index < 7 else "dev" if index < 9 else "confirm"
        assert subject.recording_partition(name, allow_confirmation=True) == expected


@pytest.mark.parametrize("name", [subject.OFFICIAL_TEST, "unknown.mat", "../" + FIT, "raw_data/" + FIT])
def test_forbidden_names_rejected_before_source_open(name, tmp_path):
    with pytest.raises(ValueError, match="TRAIN basename"):
        subject.load_recording(tmp_path / "does-not-exist", name, allow_confirmation=True)


def test_confirmation_requires_explicit_boolean_before_source_open(blob, tmp_path):
    name = subject.TRAIN_RECORDINGS[-1]
    with pytest.raises(ValueError, match="confirmation requires"):
        subject.load_recording(tmp_path / "does-not-exist", name)
    with pytest.raises(ValueError, match="must be bool"):
        subject.load_recording(blob, FIT, allow_confirmation=1)
    assert subject.load_recording(blob, name, allow_confirmation=True).partition == "confirm"


def test_only_five_required_fields_are_decoded(blob, monkeypatch):
    original = subject.loadmat
    requests = []

    def checked(*args, **kwargs):
        requests.append(tuple(kwargs["variable_names"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "loadmat", checked)
    subject.load_recording(blob, FIT)
    assert requests == [("q_se_meas", "q_mot_meas", "tau_meas", "time", "recording_ok")]


@pytest.mark.parametrize("change", ["missing", "extra", "selected_shape", "unused_shape", "float32", "nonlogical_flag"])
def test_header_schema_rejection(fields, change):
    modified = dict(fields)
    if change == "missing":
        del modified["q_ref"]
    elif change == "extra":
        modified["unexpected"] = np.zeros((1, 1))
    elif change == "selected_shape":
        modified["q_mot_meas"] = fields["q_mot_meas"][:, :-1]
    elif change == "unused_shape":
        modified["qd_ref"] = fields["qd_ref"][:, :-1]
    elif change == "float32":
        modified["tau_meas"] = fields["tau_meas"].astype(np.float32)
    else:
        modified["recording_ok"] = np.ones((1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="raw MAT"):
        subject.load_recording(encode(modified), FIT)


@pytest.mark.parametrize("key,value", [("q_se_meas", np.nan), ("q_mot_meas", np.inf),
                                       ("tau_meas", -np.inf), ("time", np.nan)])
def test_selected_nonfinite_rejected(fields, key, value):
    modified = dict(fields)
    modified[key] = fields[key].copy()
    modified[key][0, 17] = value
    with pytest.raises(ValueError, match="finite"):
        subject.load_recording(encode(modified), FIT)


def test_complex_signal_rejected_without_silently_discarding_imaginary_part(fields):
    modified = dict(fields)
    modified["tau_meas"] = fields["tau_meas"].astype(np.complex128) + 1j
    with pytest.raises(ValueError, match="real float64"):
        subject.load_recording(encode(modified), FIT)


def test_recording_error_flag_rejected(fields):
    modified = dict(fields, recording_ok=np.array([[False]]))
    with pytest.raises(ValueError, match="logical true"):
        subject.load_recording(encode(modified), FIT)


@pytest.mark.parametrize("mode", ["one_gap", "reversed", "wrong_rate"])
def test_bad_time_rejected(fields, mode):
    modified = dict(fields)
    modified["time"] = fields["time"].copy()
    if mode == "one_gap":
        modified["time"][0, 1000:] += 0.004
    elif mode == "reversed":
        modified["time"] = modified["time"][:, ::-1]
    else:
        modified["time"] *= 2.0
    with pytest.raises(ValueError, match="250 Hz"):
        subject.load_recording(encode(modified), FIT)


def test_time_origin_does_not_change_filter(fields, blob):
    modified = dict(fields, time=fields["time"] + 1000.0)
    shifted = subject.load_recording(encode(modified), FIT)
    original = subject.load_recording(blob, FIT)
    np.testing.assert_array_equal(shifted.q, original.q)
    np.testing.assert_array_equal(shifted.torque, original.torque)
    assert shifted.pins["source_sha256"] != original.pins["source_sha256"]


def test_owned_immutable_results_source_pins_and_path_parity(blob, tmp_path):
    path = tmp_path / FIT
    path.write_bytes(blob)
    left = subject.load_recording(blob, FIT)
    right = subject.load_recording(path, FIT)
    assert dict(left.pins) == dict(right.pins) == {
        "source_sha256": hashlib.sha256(blob).hexdigest(),
        "source_bytes": len(blob),
        "preprocessing_sha256": subject.PREPROCESSING_SHA256,
        "loader_version": subject.VERSION,
    }
    for key in ("q", "torque", "raw_indices"):
        a, b = getattr(left, key), getattr(right, key)
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(a, b)
        with pytest.raises(ValueError):
            a.setflags(write=True)
        with pytest.raises(ValueError):
            a.flat[0] = 0
    with pytest.raises(TypeError):
        left.pins["source_bytes"] = 0
    with pytest.raises(FrozenInstanceError):
        left.partition = "confirm"


@pytest.mark.parametrize("blob", [b"", b"not a MATLAB file", bytearray(b"x"), 17])
def test_invalid_sources_rejected(blob):
    with pytest.raises(ValueError):
        subject.load_recording(blob, FIT)


def test_symlink_source_rejected(blob, tmp_path):
    path = tmp_path / "original.mat"
    path.write_bytes(blob)
    linked = tmp_path / FIT
    linked.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        subject.load_recording(linked, FIT)
