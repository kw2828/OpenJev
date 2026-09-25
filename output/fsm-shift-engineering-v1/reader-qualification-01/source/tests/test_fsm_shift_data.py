"""Fabricated-only amplitude-reader qualification, with poisoned closed members."""
import hashlib
import io
import random
import warnings
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from openjev.research import fsm_shift_data as data


def npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def archive(mapping, *, poison=True, extra=()):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, value in mapping.items():
            output.writestr(name + ".npy", npy(value))
        if poison:
            for name in sorted(data.ARCHIVE_VARIABLES - set(data.SHIFT_VARIABLES)):
                # Not even a valid NPY header: accessing a closed member must fail.
                output.writestr(name + ".npy", b"CLOSED MEMBER: NO ARRAY HEADER")
        for name, value in extra:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                output.writestr(name, value)
    return stream.getvalue()


def admit(tmp_path, blob, **overrides):
    path = tmp_path / "fabricated.npz"
    path.write_bytes(blob)
    pins = {"expected_sha256": hashlib.sha256(blob).hexdigest(), "expected_bytes": len(blob)}
    pins.update(overrides)
    return data.read_npz_shift(path, **pins)


@pytest.fixture(scope="module")
def raw():
    u = (np.arange(8192, dtype=np.float64)[:, None, None, None]
         + 10_000*np.arange(3)[None, :, None, None]
         + 100_000*np.arange(6)[None, None, :, None]
         + 1_000_000*np.arange(2)[None, None, None, :])
    return {"u_300mV_train": u, "y_300mV_train": -2*u + 17}


@pytest.fixture(scope="module")
def blob(raw):
    return archive(raw)


@pytest.fixture
def records(tmp_path, blob):
    return admit(tmp_path, blob).records


def test_exact_two_decodes_leave_all_other_headers_closed(tmp_path, blob, monkeypatch):
    reads, opened = [], []
    original_get = np.lib.npyio.NpzFile.__getitem__
    original_open = zipfile.ZipFile.open

    def get(self, name):
        reads.append(name)
        assert name in ("u_300mV_train", "y_300mV_train")
        return original_get(self, name)

    def open_member(self, name, *args, **kwargs):
        key = name.filename if isinstance(name, zipfile.ZipInfo) else name
        opened.append(key)
        assert key in ("u_300mV_train.npy", "y_300mV_train.npy")
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", get)
    monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
    result = admit(tmp_path, blob)
    assert reads == ["u_300mV_train", "y_300mV_train"]
    assert set(opened) == {"u_300mV_train.npy", "y_300mV_train.npy"}
    assert result.decoded_keys == tuple(reads)
    assert result.source_bytes == len(blob)
    assert result.source_sha256 == hashlib.sha256(blob).hexdigest()
    assert result.native_shape == (8192, 3, 6, 2) and result.fs_hz == 6400.


@pytest.mark.parametrize("pins,error", [
    ({"expected_sha256": "0"*64}, "SHA256 mismatch"),
    ({"expected_bytes": 1}, "byte-size mismatch"),
    ({"expected_sha256": "A"*64}, "lowercase SHA256"),
    ({"expected_bytes": True}, "positive expected_bytes"),
    ({"expected_bytes": 0}, "positive expected_bytes"),
])
def test_bad_snapshot_expectations_fail_before_np_load(tmp_path, blob, monkeypatch, pins, error):
    def forbidden(*args, **kwargs):
        raise AssertionError("decode attempted before authentication")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match=error):
        admit(tmp_path, blob, **pins)


def test_decode_uses_the_authenticated_bytes_even_if_path_changes(tmp_path, blob, monkeypatch):
    original_load = np.load

    def load(source, *, allow_pickle):
        assert isinstance(source, io.BytesIO) and allow_pickle is False
        assert source.getvalue() == blob
        (tmp_path / "fabricated.npz").write_bytes(b"changed after snapshot authentication")
        return original_load(source, allow_pickle=allow_pickle)
    monkeypatch.setattr(np, "load", load)
    assert len(admit(tmp_path, blob).records) == 12


@pytest.mark.parametrize("attack", ["unknown", "duplicate", "missing_u", "missing_y"])
def test_member_roster_rejects_before_any_member_is_opened(tmp_path, raw, monkeypatch, attack):
    mapping = dict(raw)
    extra = ()
    if attack.startswith("missing"):
        del mapping["u_300mV_train" if attack == "missing_u" else "y_300mV_train"]
    elif attack == "unknown":
        extra = (("hidden.npy", b"never open"),)
    else:
        extra = (("u_300mV_train.npy", b"duplicate"),)
    payload = archive(mapping, extra=extra)

    def forbidden(*args, **kwargs):
        raise AssertionError("roster failure opened an array")
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden)
    with pytest.raises(ValueError, match="member roster"):
        admit(tmp_path, payload)


@pytest.mark.parametrize("attack,error", [
    ("float32", "float64"), ("integer", "float64"), ("complex", "float64"),
    ("big_endian", "float64"), ("wrong_axis", "shape"), ("nan", "nonfinite"),
    ("infinity", "nonfinite"),
])
def test_allowed_array_schema_has_no_coercion_or_repair(tmp_path, raw, attack, error):
    value = raw["y_300mV_train"].copy()
    if attack == "wrong_axis":
        value = value.transpose(0, 2, 1, 3)
    elif attack in ("nan", "infinity"):
        value[-1, -1, -1, -1] = np.nan if attack == "nan" else np.inf
    else:
        dtype = {"float32": np.float32, "integer": np.int64,
                 "complex": np.complex128, "big_endian": ">f8"}[attack]
        value = value.astype(dtype)
    with pytest.raises(ValueError, match=error):
        admit(tmp_path, archive({"u_300mV_train": raw["u_300mV_train"], "y_300mV_train": value}))


def test_hash_matching_npy_is_rejected_without_numpy_decode(tmp_path, monkeypatch):
    payload = npy(np.zeros((2, 3), dtype=np.float64))

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected standalone array decode")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="NPZ archive"):
        admit(tmp_path, payload)


def test_twelve_independent_records_preserve_literal_native_axes(records):
    expected_ids = []
    for realization in range(6):
        for period in range(2):
            expected_ids.append(f"300mV-realization-{realization}-period-{period}")
            record = records[2*realization+period]
            expected = (np.arange(8192)[:, None] + 10_000*np.arange(3)[None, :]
                        + 100_000*realization + 1_000_000*period)
            np.testing.assert_array_equal(record.u, expected)
            np.testing.assert_array_equal(record.y, -2*expected+17)
            assert (record.realization, record.period, record.amplitude, record.partition) == (
                realization, period, "300mV", "confirmation")
    assert [record.record_id for record in records] == expected_ids
    assert len(set(expected_ids)) == len(records) == 12


def test_reader_and_window_storage_are_owned_immutable_without_rng_effect(tmp_path, raw, blob):
    before_numpy, before_python = np.random.get_state(), random.getstate()
    result = admit(tmp_path, blob)
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
    assert random.getstate() == before_python
    record = result.records[0]
    public, target = data.public_window(record, 256), data.target_window(record, 256)
    arrays = (record.u, record.y, public.y_context, public.transition_context_u, public.future_u, target)
    for index, array in enumerate(arrays):
        assert array.dtype == np.float64 and array.flags.c_contiguous and not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0, 0] = -1
        for other in arrays[index+1:]:
            assert not np.shares_memory(array, other)
        assert not np.shares_memory(array, raw["u_300mV_train"])
    assert not np.shares_memory(record.u, result.records[1].u)
    with pytest.raises(AttributeError):
        record.period = 1
    with pytest.raises(TypeError):
        public.public_inputs()["target"] = target


@pytest.mark.parametrize("start", [0, 256, 7936, 7964])
def test_public_and_target_alignment_matches_current_input_convention(records, start):
    public = data.public_window(records[0], start)
    target = data.target_window(records[0], start)
    assert set(public.public_inputs()) == {"y_context", "transition_context_u", "future_u"}
    assert not hasattr(public, "target")
    channels = 10_000*np.arange(3)[None, :]
    np.testing.assert_array_equal(public.y_context, -2*(np.arange(start, start+100)[:, None]+channels)+17)
    np.testing.assert_array_equal(public.transition_context_u, np.arange(start+1, start+100)[:, None]+channels)
    np.testing.assert_array_equal(public.future_u, np.arange(start+100, start+228)[:, None]+channels)
    np.testing.assert_array_equal(target, -2*(np.arange(start+100, start+228)[:, None]+channels)+17)


def test_public_inputs_do_not_touch_future_outputs_or_unavailable_first_input(records):
    accessed = []

    class ArrivedOnly(np.ndarray):
        def __getitem__(self, index):
            accessed.append(index)
            assert isinstance(index, slice) and index.start == 0 and index.stop == 100
            return super().__getitem__(index)

    y = records[0].y.copy()
    y[100:] = np.nan
    u = records[0].u.copy()
    u[0] = np.nan  # Dropped unavailable input must not be numerically inspected.
    record = replace(records[0], y=y.view(ArrivedOnly), u=u)
    public = data.public_window(record, 0)
    assert accessed == [slice(0, 100)] and np.isfinite(public.y_context).all()
    with pytest.raises(ValueError, match="future target"):
        data.target_window(replace(record, y=y), 0)
    # A future-output change has no effect on any returned model input.
    changed_y = records[0].y.copy()
    changed_y[100:] = 1e100
    changed = data.public_window(replace(records[0], y=changed_y), 0)
    for key, value in public.public_inputs().items():
        np.testing.assert_array_equal(value, changed.public_inputs()[key])


@pytest.mark.parametrize("start", [-1, True, 1.5, np.int64(0), 7965])
def test_start_type_and_period_boundary_fail_explicitly(records, start):
    for helper in (data.public_window, data.target_window):
        with pytest.raises(ValueError, match="start|boundary"):
            helper(records[0], start)


@pytest.mark.parametrize("change", [
    {"amplitude": "100mV"}, {"realization": 6}, {"realization": True}, {"period": -1},
    {"partition": "test"}, {"record_id": "300mV-realization-5-period-1"}, {"fs_hz": 1.},
])
def test_false_record_identity_is_rejected_by_both_apis(records, change):
    for helper in (data.public_window, data.target_window):
        with pytest.raises(ValueError):
            helper(replace(records[0], **change), 0)


@pytest.mark.parametrize("field,index", [("y", 99), ("u", 1), ("u", 100)])
def test_nonfinite_legal_inputs_are_failures_without_imputation(records, field, index):
    changed = getattr(records[0], field).copy()
    changed[index, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        data.public_window(replace(records[0], **{field: changed}), 0)
