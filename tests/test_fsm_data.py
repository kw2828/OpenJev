"""Fabricated FSM adapter qualification; no real archive or model is used."""
import hashlib
import io
import random
from dataclasses import replace

import numpy as np
import pytest

from openjev.research import fsm_data as data


def fixture(samples=16):
    n, channel, realization, period = np.indices((samples, 3, 6, 2))
    output = {}
    for a, amplitude in enumerate(("100mV", "200mV")):
        u = (a*1_000_000+realization*10_000+channel*1000+period*100+n).astype(np.float64)
        output[f"u_{amplitude}_train"] = u
        output[f"y_{amplitude}_train"] = 2*u+7
    return output


def test_native_axes_preserve_each_period_and_orthogonal_triplets():
    result = data.records_from_fixture(fixture(8))
    assert len(result.records) == 24
    assert len(result.partition("fit")) == len(result.partition("dev")) == 12
    assert len({r.record_id for r in result.records}) == 24
    assert result.fs_hz == 6400.0 and result.native_shape == (8, 3, 6, 2)
    assert result.decoded_keys == ("u_100mV_train", "y_100mV_train", "u_200mV_train", "y_200mV_train")
    for r in result.records:
        a = 0 if r.amplitude == "100mV" else 1
        offset = a*1_000_000+r.realization*10_000+r.period*100
        expected = offset+np.arange(8)[:, None]+1000*np.arange(3)[None]
        np.testing.assert_array_equal(r.u, expected)
        np.testing.assert_array_equal(r.y, 2*expected+7)
        assert r.partition == ("fit" if r.realization <= 2 else "dev")
    for realization in range(6):
        group = [r for r in result.records if r.realization == realization]
        assert {(r.amplitude, r.period) for r in group} == {
            (a, p) for a in ("100mV", "200mV") for p in range(2)}
        assert len({r.partition for r in group}) == 1


@pytest.mark.parametrize("name", ["test", "300mV", "official", "FIT", None])
def test_no_reserved_partition_api(name):
    with pytest.raises(ValueError, match="only fit/dev"):
        data.records_from_fixture(fixture(2)).partition(name)


def test_owned_immutable_records_and_no_rng_mutation():
    raw = fixture(4)
    before_np, before_py = np.random.get_state(), random.getstate()
    result = data.records_from_fixture(raw)
    np.testing.assert_array_equal(np.random.get_state()[1], before_np[1])
    assert np.random.get_state()[0:1] == before_np[0:1] and np.random.get_state()[2:] == before_np[2:]
    assert random.getstate() == before_py
    original = result.records[0].u.copy()
    raw["u_100mV_train"][:] = -77
    np.testing.assert_array_equal(result.records[0].u, original)
    for array in (result.records[0].u, result.records[0].y):
        assert array.dtype == np.float64 and not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0, 0] = 1
    assert not np.shares_memory(result.records[0].u, result.records[1].u)
    with pytest.raises((TypeError, AttributeError)):
        result.records[0].partition = "dev"


@pytest.mark.parametrize("key", ["u_300mV_train", "y_100mV_test", "unknown"])
def test_decoded_mapping_may_not_include_reserved_or_unknown_keys(key):
    raw = fixture(2)
    raw[key] = object()
    with pytest.raises(ValueError, match="exact four"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("key", data.ESTIMATION_VARIABLES)
def test_missing_permitted_variable_rejects(key):
    raw = fixture(2)
    del raw[key]
    with pytest.raises(ValueError):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, object])
def test_real_float64_and_matching_source_dtype_required(dtype):
    raw = fixture(2)
    raw["y_200mV_train"] = raw["y_200mV_train"].astype(dtype)
    with pytest.raises(ValueError, match="float64"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("key", data.ESTIMATION_VARIABLES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_each_permitted_array_requires_finite_values(key, bad):
    raw = fixture(2)
    raw[key].flat[0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        data.records_from_fixture(raw)


@pytest.mark.parametrize("shape", [(2, 6, 3, 2), (2, 3, 6, 1), (2, 3, 5, 2), (1, 3, 6, 2)])
def test_exact_axis_geometry(shape):
    raw = fixture(2)
    raw["u_100mV_train"] = np.zeros(shape)
    with pytest.raises(ValueError):
        data.records_from_fixture(raw)


class SpyArchive(np.lib.npyio.NpzFile):
    def __init__(self, mapping, names=None):
        self.mapping = mapping
        self.files = list(data.ARCHIVE_VARIABLES) if names is None else names
        self.reads = []
        self.closed = False

    def __getitem__(self, name):
        self.reads.append(name)
        if name not in data.ESTIMATION_VARIABLES:
            raise AssertionError("reserved member was decoded")
        return self.mapping[name]

    def close(self):
        self.closed = True


def test_lazy_npz_only_four_getitems_and_same_snapshot_hash(tmp_path, monkeypatch):
    blob = b"opaque synthetic archive snapshot"
    path = tmp_path / "fabricated.npz"
    path.write_bytes(blob)
    archive = SpyArchive(fixture(2))
    marker = object()

    def load(stream, **kwargs):
        assert isinstance(stream, io.BytesIO) and stream.getvalue() == blob
        assert kwargs == {"allow_pickle": False}
        path.write_bytes(b"changed after snapshot")
        return archive

    def records(mapping, shape, digest, count):
        assert set(mapping) == set(data.ESTIMATION_VARIABLES)
        assert shape == (8192, 3, 6, 2)
        assert digest == hashlib.sha256(blob).hexdigest() and count == len(blob)
        return marker

    monkeypatch.setattr(data.np, "load", load)
    monkeypatch.setattr(data, "_records", records)
    assert data.read_npz_estimation(path) is marker
    assert archive.reads == list(data.ESTIMATION_VARIABLES) and archive.closed


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "path"])
def test_member_roster_failures_precede_any_array_decode(tmp_path, monkeypatch, change):
    names = list(data.ESTIMATION_VARIABLES)
    if change == "missing":
        names.pop()
    elif change == "duplicate":
        names.append(names[0])
    elif change == "unknown":
        names.append("unknown")
    else:
        names.append("../y_100mV_train")
    archive = SpyArchive(fixture(2), names)
    path = tmp_path / "opaque.npz"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(data.np, "load", lambda *a, **kw: archive)
    with pytest.raises(ValueError, match="member roster"):
        data.read_npz_estimation(path)
    assert archive.reads == [] and archive.closed


def test_production_shape_cannot_be_relaxed_by_fixture_reader(tmp_path, monkeypatch):
    archive = SpyArchive(fixture(2))
    path = tmp_path / "small.npz"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(data.np, "load", lambda *a, **kw: archive)
    with pytest.raises(ValueError, match="unexpected .* shape"):
        data.read_npz_estimation(path)
    assert archive.closed and archive.reads == list(data.ESTIMATION_VARIABLES)


def test_real_fabricated_npz_reserved_object_members_remain_undecoded(tmp_path, monkeypatch):
    # These fabricated objects would raise under allow_pickle=False if accessed.
    mapping = fixture(2)
    for key in data.ARCHIVE_VARIABLES-set(data.ESTIMATION_VARIABLES):
        mapping[key] = np.array([{"never": "decode"}], dtype=object)
    path = tmp_path / "poisoned-fabricated.npz"
    np.savez_compressed(path, **mapping)
    original = data._records

    def validate_fixture(mapping, shape, digest, count):
        assert shape == (8192, 3, 6, 2)
        return original(mapping, (2, 3, 6, 2), digest, count)

    monkeypatch.setattr(data, "_records", validate_fixture)
    result = data.read_npz_estimation(path)
    assert len(result.records) == 24
    assert result.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.source_bytes == path.stat().st_size


def test_channel_normalizer_hand_means_stds_and_dev_is_never_inspected():
    raw = fixture(2)
    for key in raw:
        values = np.array([[0., 10., -4.], [2., 14., 2.]])
        raw[key][:] = (values if key.startswith("u_") else 2*values+7)[:, :, None, None]
    records = data.records_from_fixture(raw).records
    norm = data.fit_normalizer(records)
    for actual, expected in ((norm.u_mean, [1, 12, -1]), (norm.u_scale, [1, 2, 3]),
                             (norm.y_mean, [9, 31, 5]), (norm.y_scale, [2, 4, 6])):
        np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    assert norm.fit_samples == 24 and len(norm.fit_record_ids) == 12
    changed = [replace(r, u=np.array([[np.nan]]), y=np.array([[np.inf]]))
               if r.partition == "dev" else r for r in records]
    altered = data.fit_normalizer(changed)
    for key in ("u_mean", "u_scale", "y_mean", "y_scale"):
        np.testing.assert_array_equal(getattr(norm, key), getattr(altered, key))
    assert altered.fit_record_ids == norm.fit_record_ids


@pytest.mark.parametrize("change", ["missing", "duplicate", "partition", "zero_channel"])
def test_normalizer_requires_complete_twelve_fit_records_and_positive_channel_scales(change):
    raw = fixture(2)
    if change == "zero_channel":
        for key in raw:
            raw[key][:, 1] = 0
    records = list(data.records_from_fixture(raw).partition("fit"))
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = records[0]
    elif change == "partition":
        records[0] = replace(records[0], partition="dev")
    with pytest.raises(ValueError):
        data.fit_normalizer(records)


def test_same_time_input_alignment_and_period_boundary():
    record = data.records_from_fixture(fixture(12)).records[0]
    window = data.make_window(record, 3, context=4, horizon=5)
    channels = 1000*np.arange(3)[None]
    np.testing.assert_array_equal(window.y_context, 2*(np.arange(3, 7)[:, None]+channels)+7)
    np.testing.assert_array_equal(window.transition_context_u, np.arange(4, 7)[:, None]+channels)
    np.testing.assert_array_equal(window.future_u, np.arange(7, 12)[:, None]+channels)
    np.testing.assert_array_equal(window.target, 2*(np.arange(7, 12)[:, None]+channels)+7)
    assert set(window.public_inputs()) == {"y_context", "transition_context_u", "future_u"}
    with pytest.raises(ValueError, match="period boundary"):
        data.make_window(record, 4, context=4, horizon=5)
    single = data.make_window(record, 0, context=1, horizon=1)
    assert single.transition_context_u.shape == (0, 3)
    assert single.future_u[0, 0] == 1 and single.target[0, 0] == 9


def test_default_100_context_128_horizon_and_skipped_initial_input():
    record = data.records_from_fixture(fixture(228)).records[0]
    original = data.make_window(record, 0)
    assert original.y_context.shape == (100, 3)
    assert original.transition_context_u.shape == (99, 3)
    assert original.future_u.shape == original.target.shape == (128, 3)
    assert original.future_u[0, 0] == 100
    changed = record.u.copy()
    changed[0] = -1e9
    altered = data.make_window(replace(record, u=changed), 0)
    for key in original.public_inputs():
        np.testing.assert_array_equal(original.public_inputs()[key], altered.public_inputs()[key])


def test_target_isolation_future_input_prefix_and_owned_window_arrays():
    record = data.records_from_fixture(fixture(20)).records[0]
    original = data.make_window(record, 2, context=4, horizon=5)
    changed_y, changed_u = record.y.copy(), record.u.copy()
    changed_y[6:] = -123
    altered = data.make_window(replace(record, y=changed_y), 2, context=4, horizon=5)
    for key in original.public_inputs():
        np.testing.assert_array_equal(original.public_inputs()[key], altered.public_inputs()[key])
    assert not np.array_equal(original.target, altered.target)
    changed_u[8:] = 456
    suffix = data.make_window(replace(record, u=changed_u), 2, context=4, horizon=5)
    np.testing.assert_array_equal(suffix.future_u[:2], original.future_u[:2])
    for array in (*original.public_inputs().values(), original.target):
        assert not np.shares_memory(array, record.u) and not np.shares_memory(array, record.y)
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        original.public_inputs()["target"] = original.target


def test_window_applies_distinct_input_and_output_channel_scales():
    record = data.records_from_fixture(fixture(8)).records[0]
    norm = data.Normalizer(np.array([1., 1001., 2001.]), np.array([2., 4., 8.]),
                           np.array([9., 2009., 4009.]), np.array([4., 8., 16.]), (), 0)
    window = data.make_window(record, 0, context=2, horizon=2, normalizer=norm)
    np.testing.assert_array_equal(window.y_context, [[-.5, -.25, -.125], [0, 0, 0]])
    np.testing.assert_array_equal(window.transition_context_u, [[0, 0, 0]])
    np.testing.assert_array_equal(window.future_u, [[.5, .25, .125], [1, .5, .25]])
    np.testing.assert_array_equal(window.target, [[.5, .25, .125], [1, .5, .25]])
    with pytest.raises(ValueError, match="positive"):
        data.make_window(record, 0, context=2, horizon=2,
                         normalizer=replace(norm, y_scale=np.array([1., 0., 1.])))


@pytest.mark.parametrize("kwargs", [{"start": -1}, {"start": True}, {"start": .5},
                                    {"context": 0}, {"horizon": False}, {"horizon": 0},
                                    {"normalizer": {}}, {"context": 2.0}])
def test_invalid_window_arguments(kwargs):
    record = data.records_from_fixture(fixture(20)).records[0]
    with pytest.raises(ValueError):
        data.make_window(record, **{"start": 0, "context": 2, "horizon": 2, **kwargs})
