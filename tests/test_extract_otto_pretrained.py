"""Artificial tiny HDF5 files only. No official weights, framework or simulator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_otto_pretrained.py"
SPEC = importlib.util.spec_from_file_location("extract_otto_fixture", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture
def small_shapes(monkeypatch):
    shapes = {"kernel_0": (6, 4), "bias_0": (4,),
              "kernel_1": (4, 4), "bias_1": (4,),
              "kernel_2": (4, 4), "bias_2": (4,),
              "kernel_3": (4, 1), "bias_3": (1,)}
    monkeypatch.setattr(M, "SHAPES", shapes)
    return shapes


def fixture_file(path, shapes):
    expected = {}
    with h5py.File(path, "w") as handle:
        handle.attrs["layer_names"] = np.asarray(M.LAYERS, dtype="S")
        handle.attrs["backend"] = np.bytes_("tensorflow")
        handle.attrs["keras_version"] = np.bytes_("fixture-not-a-real-model")
        flatten = handle.create_group(M.LAYERS[0])
        flatten.attrs["weight_names"] = np.array([], dtype="S1")
        for index, layer in enumerate(M.LAYERS[1:]):
            group = handle.create_group(layer)
            group.attrs["weight_names"] = np.asarray(
                [f"{layer}/kernel:0", f"{layer}/bias:0"], dtype="S")
            nested = group.create_group(layer)
            for kind in ("kernel", "bias"):
                key = f"{kind}_{index}"
                shape = shapes[key]
                # Asymmetry and negatives detect transposition, casting or clipping.
                array = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 5) / 8
                nested.create_dataset(f"{kind}:0", data=array)
                expected[key] = array
    return expected


def bind_synthetic_assets(directory, monkeypatch):
    assets = {}
    entries = []
    for name in ("zoo_model_2_3_2", "zoo_model_2_3_2.config"):
        data = (directory / name).read_bytes()
        item = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
        assets[name] = item
        entries.append({"path": name, **item, "expected_bytes": len(data),
                        "expected_git_blob_sha1": item["git_blob_sha1"], "exit_code": 0})
    receipt = {"status": "completed", "repository": "C0PEP0D/otto", "commit": M.COMMIT,
               "assets": entries}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(M, "ASSETS", assets)
    monkeypatch.setattr(M, "RECEIPT_PIN", M.sha(directory / "receipt.json"))


@pytest.fixture
def assets(tmp_path, small_shapes, monkeypatch):
    directory = tmp_path / "artificial-assets"
    directory.mkdir()
    expected = fixture_file(directory / "zoo_model_2_3_2", small_shapes)
    # Serialization is fixture generation only; neither extraction nor test executes it.
    (directory / "zoo_model_2_3_2.config").write_bytes(pickle.dumps(dict(M.CONFIG), protocol=4))
    bind_synthetic_assets(directory, monkeypatch)
    return directory, expected


def test_complete_extraction_preserves_exact_tensor_bytes_and_orientation(assets, tmp_path, monkeypatch):
    directory, expected = assets
    monkeypatch.setattr(pickle, "load", lambda *a, **kw: pytest.fail("pickle execution"))
    monkeypatch.setattr(pickle, "loads", lambda *a, **kw: pytest.fail("pickle execution"))
    destination = tmp_path / "result"
    receipt = M.execute(argparse.Namespace(assets=str(directory), out=str(destination)))
    assert receipt["status"] == "completed" and receipt["model_calls"] == 0
    assert receipt["pickle_executed"] is False and receipt["inference_qualified"] is False
    assert {p.name for p in destination.iterdir()} == set(receipt["files"]) | {"receipt.json"}
    assert receipt["source_sha256"][str(SCRIPT)] == M.sha(SCRIPT)
    metadata = json.loads((destination / "tensor-metadata.json").read_text())
    with np.load(destination / "tensors.npz", allow_pickle=False) as packet:
        assert set(packet.files) == set(expected)
        for key, array in expected.items():
            assert packet[key].dtype == np.dtype("float32")
            assert packet[key].tobytes() == array.tobytes()
            record = metadata["datasets"][key]
            kind, index = key.split("_")
            layer = M.LAYERS[int(index) + 1]
            assert record["dataset"] == f"/{layer}/{layer}/{kind}:0"
            assert record["sha256_c_order"] == hashlib.sha256(array.tobytes()).hexdigest()
            assert record["negative_values"] > 0
    for name, item in receipt["files"].items():
        assert item == M.descriptor(destination / name)


@pytest.mark.parametrize("corruption", ["weight", "config", "receipt", "git_blob", "extra", "symlink"])
def test_authentication_rejects_before_hdf5_reader(assets, tmp_path, monkeypatch, corruption):
    directory, _ = assets
    if corruption in ("weight", "config", "receipt"):
        name = {"weight": "zoo_model_2_3_2", "config": "zoo_model_2_3_2.config",
                "receipt": "receipt.json"}[corruption]
        with (directory / name).open("ab") as stream:
            stream.write(b"tamper")
    elif corruption == "git_blob":
        monkeypatch.setitem(M.ASSETS["zoo_model_2_3_2"], "git_blob_sha1", "0" * 40)
    elif corruption == "extra":
        (directory / "extra").write_text("unexpected")
    else:
        config = directory / "zoo_model_2_3_2.config"
        moved = tmp_path / "config-target"
        config.rename(moved)
        config.symlink_to(moved)
    monkeypatch.setattr(M, "read_tensors", lambda *a, **kw: pytest.fail("HDF5 reader called before auth"))
    destination = tmp_path / "failure"
    with pytest.raises(ValueError):
        M.execute(argparse.Namespace(assets=str(directory), out=str(destination)))
    assert (destination / "failed.json").exists()
    assert not (destination / "receipt.json").exists()
    assert not (destination / "tensors.npz").exists()


@pytest.mark.parametrize("corruption", ["soft", "external", "alias", "extra_group", "extra_dataset",
                                       "datatype_object", "extra_attribute", "shape", "float64",
                                       "big_endian", "nan", "infinity", "compressed", "external_storage",
                                       "virtual", "weight_order"])
def test_invalid_hdf5_rejected_without_repair(tmp_path, small_shapes, corruption):
    path = tmp_path / "artificial.h5"
    fixture_file(path, small_shapes)
    with h5py.File(path, "r+") as handle:
        layer = M.LAYERS[1]
        nested = handle[layer][layer]
        name = "kernel:0"
        if corruption in ("soft", "external", "alias"):
            del handle[M.LAYERS[2]]
            if corruption == "soft":
                handle[M.LAYERS[2]] = h5py.SoftLink(f"/{layer}")
            elif corruption == "external":
                handle[M.LAYERS[2]] = h5py.ExternalLink("must-not-open.h5", "/")
            else:
                handle[M.LAYERS[2]] = handle[layer]
        elif corruption == "extra_group":
            handle.create_group("unexpected")
        elif corruption == "extra_dataset":
            nested.create_dataset("unexpected", data=np.zeros(1, dtype="f4"))
        elif corruption == "datatype_object":
            del nested[name]
            nested[name] = np.dtype("float32")
        elif corruption == "extra_attribute":
            nested.attrs["unexpected"] = "metadata"
        elif corruption in ("nan", "infinity"):
            nested[name][0, 0] = np.nan if corruption == "nan" else np.inf
        elif corruption == "weight_order":
            handle[layer].attrs["weight_names"] = np.asarray(
                [f"{layer}/bias:0", f"{layer}/kernel:0"], dtype="S")
        else:
            del nested[name]
            shape = small_shapes["kernel_0"]
            if corruption == "shape":
                nested.create_dataset(name, data=np.zeros(shape[::-1], dtype="f4"))
            elif corruption in ("float64", "big_endian"):
                nested.create_dataset(name, data=np.zeros(shape, dtype="f8" if corruption == "float64" else ">f4"))
            elif corruption == "compressed":
                nested.create_dataset(name, data=np.zeros(shape, dtype="f4"), compression="gzip")
            elif corruption == "external_storage":
                nested.create_dataset(name, shape=shape, dtype="f4", external=[("must-not-open.bin", 0, h5py.h5f.UNLIMITED)])
            else:
                layout = h5py.VirtualLayout(shape=shape, dtype="f4")
                layout[:] = h5py.VirtualSource("must-not-open.h5", "/data", shape=shape)
                nested.create_virtual_dataset(name, layout)
    with pytest.raises(ValueError):
        M.read_tensors(path)
    assert not (tmp_path / "must-not-open.h5").exists()
    assert not (tmp_path / "must-not-open.bin").exists()


def test_nonfinite_failure_preserved_and_retry_cannot_overwrite(assets, tmp_path, monkeypatch):
    directory, _ = assets
    with h5py.File(directory / "zoo_model_2_3_2", "r+") as handle:
        handle["dense_84/dense_84/kernel:0"][0, 0] = np.nan
    bind_synthetic_assets(directory, monkeypatch)
    destination = tmp_path / "failed-result"
    args = argparse.Namespace(assets=str(directory), out=str(destination))
    with pytest.raises(ValueError, match="finite tensor"):
        M.execute(args)
    evidence = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        M.execute(args)
    assert evidence == {p.name: p.read_bytes() for p in destination.iterdir()}


def test_failure_writing_receipt_does_not_replace_original(assets, tmp_path, monkeypatch):
    directory, _ = assets
    original = M.write

    def write(path, value):
        if path.name == "failed.json":
            raise OSError("synthetic failure-receipt IO error")
        return original(path, value)

    monkeypatch.setattr(M, "write", write)
    monkeypatch.setattr(M, "read_tensors", lambda *a: (_ for _ in ()).throw(ValueError("original extraction error")))
    with pytest.raises(ValueError, match="original extraction error") as caught:
        M.execute(argparse.Namespace(assets=str(directory), out=str(tmp_path / "result")))
    assert any("receipt could not" in note for note in caught.value.__notes__)


def test_late_success_is_demoted_with_payloads_preserved(assets, tmp_path, monkeypatch):
    directory, _ = assets
    original = M.write

    def write(path, value):
        result = original(path, value)
        if path.name == "receipt.json":
            monkeypatch.setitem(M.LIMITS, "wall_seconds", -1)
        return result

    monkeypatch.setattr(M, "write", write)
    destination = tmp_path / "late"
    with pytest.raises(TimeoutError):
        M.execute(argparse.Namespace(assets=str(directory), out=str(destination)))
    assert not (destination / "receipt.json").exists()
    assert (destination / "late-completion.json").exists()
    assert (destination / "failed.json").exists()
    assert (destination / "tensors.npz").exists()
