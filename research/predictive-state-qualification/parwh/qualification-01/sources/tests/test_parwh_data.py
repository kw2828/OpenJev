"""Fabricated ParWH axis, input-time and estimation-boundary qualification."""
from __future__ import annotations

import hashlib
import io
import random
from dataclasses import replace

import numpy as np
import pytest
import scipy.io

from openjev.research import parwh_data as data


def fixture(samples=64):
    n, period, phase, amp = np.indices((samples, 2, 20, 5))
    u = (1_000_000*phase + 10_000*amp + 100*period + n).astype(np.float64)
    return {"uEst": u, "yEst": 2*u+7, "fs": np.array([[78125.0]]),
            "amp": np.array([[.1, .325, .55, .775, 1.0]])}


def test_native_axes_and_complete_phase_groups_do_not_interleave_periods():
    records = data.records_from_fixture(fixture(8))
    assert records.native_shape == (8, 2, 20, 5)
    assert len(records.records) == 200
    assert len(records.partition("fit")) == 150
    assert len(records.partition("dev")) == 50
    assert len({r.record_id for r in records.records}) == 200
    for r in records.records:
        offset = 1_000_000*r.phase+10_000*r.amplitude_index+100*r.period
        np.testing.assert_array_equal(r.u[:, 0], offset+np.arange(8))
        np.testing.assert_array_equal(r.y[:, 0], 2*(offset+np.arange(8))+7)
        assert r.partition == ("fit" if r.phase <= 14 else "dev")
    for phase in range(20):
        group = [r for r in records.records if r.phase == phase]
        assert {(r.amplitude_index, r.period) for r in group} == {
            (a, p) for a in range(5) for p in range(2)}
        assert len({r.partition for r in group}) == 1
    assert records.source_sha256 is None and records.source_bytes is None


@pytest.mark.parametrize("name", ["test", "val", "official", "FIT", None])
def test_no_official_partition_api(name):
    with pytest.raises(ValueError, match="only fit/dev"):
        data.records_from_fixture(fixture(2)).partition(name)


def test_record_arrays_owned_immutable_and_no_global_rng_mutation():
    raw = fixture(4)
    before_numpy = np.random.get_state()
    before_python = random.getstate()
    loaded = data.records_from_fixture(raw)
    np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
    assert np.random.get_state()[0:1] == before_numpy[0:1]
    assert np.random.get_state()[2:] == before_numpy[2:]
    assert random.getstate() == before_python
    original = loaded.records[0].u.copy()
    raw["uEst"][:] = -900
    raw["amp"][:] = 123
    np.testing.assert_array_equal(loaded.records[0].u, original)
    assert loaded.amplitudes[0] == .1
    arrays = [loaded.amplitudes, loaded.records[0].u, loaded.records[0].y]
    for a in arrays:
        assert a.dtype == np.float64 and not a.flags.writeable
        with pytest.raises(ValueError):
            a.setflags(write=True)
        with pytest.raises(ValueError):
            a.flat[0] = 1
    assert not np.shares_memory(loaded.records[0].u, loaded.records[1].u)
    with pytest.raises((AttributeError, TypeError)):
        loaded.records[0].partition = "dev"


@pytest.mark.parametrize("key", ["uVal", "yVal", "uValArr", "unknown"])
def test_unknown_or_official_measurement_variables_reject(key):
    raw = fixture(2)
    raw[key] = object()
    with pytest.raises(ValueError, match="unknown or official-test"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("key", data.ESTIMATION_VARIABLES)
def test_missing_required_variables_reject(key):
    raw = fixture(2)
    del raw[key]
    with pytest.raises(ValueError):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("key", data.ESTIMATION_VARIABLES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_reject(key, bad):
    raw = fixture(2)
    raw[key].flat[0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, object])
def test_real_float64_required_and_asymmetric_dtype_rejects(dtype):
    raw = fixture(2)
    raw["yEst"] = raw["yEst"].astype(dtype)
    with pytest.raises(ValueError, match="float64"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("key,shape", [("uEst", (2, 20, 5, 2)), ("yEst", (2, 2, 19, 5)),
                                     ("fs", (1,)), ("amp", (5,)), ("amp", (5, 1))])
def test_exact_native_geometry_and_metadata_shapes(key, shape):
    raw = fixture(2)
    raw[key] = np.zeros(shape, dtype=np.float64)
    with pytest.raises(ValueError):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("kind", ["fs_zero", "fs_negative", "amp_zero", "amp_duplicate", "amp_reverse"])
def test_invalid_physical_metadata(kind):
    raw = fixture(2)
    if kind == "fs_zero":
        raw["fs"][0, 0] = 0
    elif kind == "fs_negative":
        raw["fs"][0, 0] = -1
    elif kind == "amp_zero":
        raw["amp"][0, 0] = 0
    elif kind == "amp_duplicate":
        raw["amp"][0, 1] = raw["amp"][0, 0]
    else:
        raw["amp"] = raw["amp"][:, ::-1]
    with pytest.raises(ValueError, match="positive"):
        data.records_from_fixture(raw)


def test_loader_whitelists_only_estimation_and_binds_same_bytes(tmp_path, monkeypatch):
    # Opaque fabricated bytes, not a measured MAT file. Decoder and validator are spies.
    path = tmp_path / "fabricated.mat"
    blob = b"opaque fixture bytes with a pretend official-test marker"
    path.write_bytes(blob)
    mapping = fixture(2)
    marker = object()
    calls = []

    def fake_loadmat(stream, **kwargs):
        assert isinstance(stream, io.BytesIO) and stream.getvalue() == blob
        assert kwargs == {"variable_names": ("uEst", "yEst", "fs", "amp"),
                          "appendmat": False, "verify_compressed_data_integrity": True}
        # Mutation after snapshot cannot change the decoded/hash identity.
        path.write_bytes(b"replaced after snapshot")
        calls.append("load")
        return mapping

    def fake_records(actual, shape, digest, count):
        assert actual is mapping and shape == (16384, 2, 20, 5)
        assert digest == hashlib.sha256(blob).hexdigest() and count == len(blob)
        calls.append("validate")
        return marker

    monkeypatch.setattr(scipy.io, "loadmat", fake_loadmat)
    monkeypatch.setattr(data, "_records", fake_records)
    assert data.read_mat_estimation(path) is marker
    assert calls == ["load", "validate"]


def test_production_loader_cannot_take_small_fixture_geometry(tmp_path, monkeypatch):
    path = tmp_path / "small.mat"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(scipy.io, "loadmat", lambda *a, **kw: fixture(2))
    with pytest.raises(ValueError, match="unexpected uEst shape"):
        data.read_mat_estimation(path)


def test_mat_metadata_is_not_treated_as_a_measurement():
    raw = fixture(2)
    raw.update(__header__=b"fabricated", __version__="1.0", __globals__=[])
    assert len(data.records_from_fixture(raw).records) == 200


def test_hand_population_normalizer_and_dev_values_never_contribute():
    raw = fixture(2)
    raw["uEst"][:] = np.array([0., 2.])[:, None, None, None]
    raw["yEst"][:] = np.array([4., 8.])[:, None, None, None]
    records = data.records_from_fixture(raw).records
    norm = data.fit_normalizer(records)
    assert (norm.u_mean, norm.u_scale, norm.y_mean, norm.y_scale) == (1., 1., 6., 2.)
    assert norm.fit_samples == 300 and len(norm.fit_record_ids) == 150
    # Even invalid DEV measurements are not inspected by FIT normalization.
    changed = tuple(replace(r, u=np.array([[np.nan]]), y=np.array([[np.inf]]))
                    if r.partition == "dev" else r for r in records)
    assert data.fit_normalizer(changed) == norm
    assert data.fit_normalizer(r for r in records if r.partition == "fit") == norm


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_partition", "zero_scale"])
def test_normalizer_requires_complete_fit_and_nonzero_scales(change):
    raw = fixture(2)
    if change == "zero_scale":
        raw["uEst"][:] = 1
    records = list(data.records_from_fixture(raw).partition("fit"))
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = records[0]
    elif change == "wrong_partition":
        records[0] = replace(records[0], partition="dev")
    with pytest.raises(ValueError):
        data.fit_normalizer(records)


def test_window_same_time_input_alignment_and_boundary():
    record = data.records_from_fixture(fixture(12)).records[0]
    window = data.make_window(record, 3, context=4, horizon=5)
    np.testing.assert_array_equal(window.y_context[:, 0], 2*np.arange(3, 7)+7)
    np.testing.assert_array_equal(window.transition_context_u[:, 0], np.arange(4, 7))
    np.testing.assert_array_equal(window.future_u[:, 0], np.arange(7, 12))
    np.testing.assert_array_equal(window.target[:, 0], 2*np.arange(7, 12)+7)
    assert window.record_id == record.record_id and window.start == 3
    assert set(window.public_inputs()) == {"y_context", "transition_context_u", "future_u"}
    one = data.make_window(record, 0, context=1, horizon=1)
    assert one.transition_context_u.shape == (0, 1)
    assert one.future_u[0, 0] == 1 and one.target[0, 0] == 9
    with pytest.raises(ValueError, match="period boundary"):
        data.make_window(record, 4, context=4, horizon=5)


def test_default_context_horizon_are_50_256():
    record = data.records_from_fixture(fixture(306)).records[0]
    window = data.make_window(record, 0)
    assert window.y_context.shape == (50, 1)
    assert window.transition_context_u.shape == (49, 1)
    assert window.future_u.shape == window.target.shape == (256, 1)
    assert window.transition_context_u[0, 0] == 1 and window.future_u[0, 0] == 50


def test_targets_are_isolated_and_window_arrays_are_owned():
    record = data.records_from_fixture(fixture(20)).records[0]
    original = data.make_window(record, 2, context=4, horizon=5)
    changed_y = record.y.copy()
    changed_y[6:] = -999
    altered = data.make_window(replace(record, y=changed_y), 2, context=4, horizon=5)
    for key, value in original.public_inputs().items():
        np.testing.assert_array_equal(value, altered.public_inputs()[key])
    assert not np.array_equal(original.target, altered.target)
    for array in (*original.public_inputs().values(), original.target):
        assert not np.shares_memory(array, record.u) and not np.shares_memory(array, record.y)
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        original.public_inputs()["target"] = original.target


def test_window_normalization_hand_calculation():
    record = data.records_from_fixture(fixture(8)).records[0]
    norm = data.Normalizer(1., 2., 7., 4., (), 0)
    window = data.make_window(record, 0, context=2, horizon=2, normalizer=norm)
    np.testing.assert_array_equal(window.y_context[:, 0], [0., .5])
    np.testing.assert_array_equal(window.transition_context_u[:, 0], [0.])
    np.testing.assert_array_equal(window.future_u[:, 0], [.5, 1.])
    np.testing.assert_array_equal(window.target[:, 0], [1., 1.5])


@pytest.mark.parametrize("kwargs", [{"start": -1}, {"start": True}, {"start": .5},
                                    {"context": 0}, {"context": 2.0}, {"horizon": 0},
                                    {"horizon": False}, {"normalizer": {}}])
def test_malformed_window_requests_reject(kwargs):
    record = data.records_from_fixture(fixture(20)).records[0]
    args = {"start": 0, "context": 2, "horizon": 2, **kwargs}
    with pytest.raises(ValueError):
        data.make_window(record, **args)
