"""Extract the pinned OTTO HDF5 tensors only; never execute pickle or a model.

Keras orientation is retained: x @ kernel_i + bias_i. Configuration values are
known from the authenticated config's static opcode inspection. This utility
does not establish numerical forward, policy, or benchmark-checkpoint parity.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickletools
import platform
import resource
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-pretrained-extraction-v1"
COMMIT = "1467029f399dc5eeac8652499a9c8326ecab4575"
RECEIPT_PIN = "c4c55ff316346d9165f4d814582911acefd3b3b5a0bd4a301cffe1d01cab81b6"
ASSETS = {
    "zoo_model_2_3_2": {
        "bytes": 53581916,
        "sha256": "1efb73aa38e0fd8b08d6d03059c0db4664da3afb8eaff1d7ab9b363f8e7ad37d",
        "git_blob_sha1": "ab27e6db2fdbaec6cdc61d876d471aabf979005c",
    },
    "zoo_model_2_3_2.config": {
        "bytes": 147,
        "sha256": "ca12567f2e0333192a7a0b0c177d519bdae60749ddb8005a1456608fd6782b4c",
        "git_blob_sha1": "89aea1154e2f9a3266695735ac31f77a882e622a",
    },
}
LAYERS = ("flatten_21", "dense_84", "dense_85", "dense_86", "dense_87")
SHAPES = {
    "kernel_0": (11025, 1024), "bias_0": (1024,),
    "kernel_1": (1024, 1024), "bias_1": (1024,),
    "kernel_2": (1024, 1024), "bias_2": (1024,),
    "kernel_3": (1024, 1), "bias_3": (1,),
}
CONFIG = {"Ndim": 2, "FC_layers": 3, "FC_units": 1024,
          "regularization_factor": 0.0, "loss_function": "mean_squared_error"}
LIMITS = {"wall_seconds": 60, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
CLOCK_PATH = ROOT / "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def authenticate(directory):
    """All input identities are verified before importing/opening HDF5."""
    require(not directory.is_symlink() and directory.is_dir(), "regular asset directory")
    require({p.name for p in directory.iterdir()} == set(ASSETS) | {"receipt.json"},
            "exact retrieved asset membership")
    receipt_path = directory / "receipt.json"
    require(not receipt_path.is_symlink() and sha(receipt_path) == RECEIPT_PIN,
            "retrieval receipt pin")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "completed" and receipt["repository"] == "C0PEP0D/otto"
            and receipt["commit"] == COMMIT, "completed official retrieval")
    require(len(receipt["assets"]) == len(ASSETS)
            and {x["path"] for x in receipt["assets"]} == set(ASSETS), "receipt asset identities")
    items = {item["path"]: item for item in receipt["assets"]}
    for name, expected in ASSETS.items():
        path = directory / name
        require(not path.is_symlink() and path.is_file(), f"regular input: {name}")
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                f"input bytes/SHA256: {name}")
        digest = hashlib.sha1(f"blob {expected['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        require(digest.hexdigest() == expected["git_blob_sha1"], f"Git blob: {name}")
        item = items[name]
        require(all(item[k] == value for k, value in expected.items())
                and item["expected_bytes"] == expected["bytes"]
                and item["expected_git_blob_sha1"] == expected["git_blob_sha1"]
                and item["exit_code"] == 0, f"retrieval witness: {name}")
    with (directory / "zoo_model_2_3_2").open("rb") as stream:
        require(stream.read(8) == b"\x89HDF\r\n\x1a\n", "HDF5 magic")
    return {"receipt": {"path": str(receipt_path), **descriptor(receipt_path)},
            "assets": {name: {"path": str(directory / name), **item}
                       for name, item in ASSETS.items()}}


def config_opcodes(path):
    """Disassemble bytes; no pickle.load/loads or opcode execution."""
    stream = io.StringIO()
    pickletools.dis(path.read_bytes(), out=stream)
    return stream.getvalue()


def read_tensors(path, check=lambda: None):
    import h5py
    import numpy as np

    def text(value):
        require(isinstance(value, (bytes, str, np.bytes_, np.str_)), "string attribute")
        result = value.decode("ascii") if isinstance(value, bytes) else str(value)
        require(len(result) <= 128 and "\x00" not in result, "bounded string attribute")
        return result

    def names(value):
        values = np.asarray(value)
        require(values.ndim == 1 and len(values) <= 8, "bounded name vector")
        return [text(value) for value in values]

    def attributes(obj, expected):
        require(set(obj.attrs) == set(expected), f"exact attributes: {obj.name}")

    seen = set()

    def child(group, name, kind):
        require(isinstance(group.get(name, getlink=True), h5py.HardLink),
                f"only local hard links: {group.name}/{name}")
        obj = group[name]
        require(isinstance(obj, kind), f"object type: {obj.name}")
        address = h5py.h5o.get_info(obj.id).addr
        require(address not in seen, f"aliased/cyclic object: {obj.name}")
        seen.add(address)
        return obj

    arrays, records = {}, {}
    with h5py.File(path, "r") as handle:
        seen.add(h5py.h5o.get_info(handle.id).addr)
        attributes(handle, ("layer_names", "backend", "keras_version"))
        require(names(handle.attrs["layer_names"]) == list(LAYERS), "exact ordered layers")
        require(text(handle.attrs["backend"]) == "tensorflow", "TensorFlow backend")
        keras_version = text(handle.attrs["keras_version"])
        require(set(handle) == set(LAYERS), "exact root objects")
        for index, layer in enumerate(LAYERS):
            check()
            group = child(handle, layer, h5py.Group)
            attributes(group, ("weight_names",))
            if index == 0:
                require(names(group.attrs["weight_names"]) == [] and len(group) == 0,
                        "empty flatten layer")
                continue
            expected_names = [f"{layer}/kernel:0", f"{layer}/bias:0"]
            require(names(group.attrs["weight_names"]) == expected_names,
                    f"exact ordered weights: {layer}")
            require(set(group) == {layer}, f"exact layer group: {layer}")
            nested = child(group, layer, h5py.Group)
            attributes(nested, ())
            require(set(nested) == {"kernel:0", "bias:0"}, f"exact datasets: {layer}")
            for kind in ("kernel", "bias"):
                key = f"{kind}_{index - 1}"
                dataset = child(nested, f"{kind}:0", h5py.Dataset)
                attributes(dataset, ())
                require(dataset.shape == SHAPES[key], f"tensor shape: {key}")
                require(dataset.dtype == np.dtype("<f4")
                        and dataset.id.get_type().equal(h5py.h5t.IEEE_F32LE), f"tensor dtype: {key}")
                properties = dataset.id.get_create_plist()
                require(not dataset.is_virtual and not dataset.external
                        and properties.get_nfilters() == 0 and dataset.chunks is None
                        and properties.get_layout() == h5py.h5d.CONTIGUOUS,
                        f"plain internal contiguous storage: {key}")
                offset = dataset.id.get_offset()
                require(isinstance(offset, int) and offset >= 0
                        and offset + dataset.size * 4 <= path.stat().st_size,
                        f"allocated internal tensor: {key}")
                check()
                values = dataset[...]
                require(np.isfinite(values).all(), f"finite tensor: {key}")
                arrays[key] = values
                records[key] = {"dataset": dataset.name, "shape": list(values.shape),
                                "dtype": values.dtype.str, "bytes": values.nbytes,
                                "sha256_c_order": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
                                "negative_values": int(np.count_nonzero(values < 0)),
                                "minimum": float(values.min()), "maximum": float(values.max())}
    require(set(arrays) == set(SHAPES), "complete eight tensors")
    return arrays, {"datasets": records, "backend": "tensorflow", "keras_version": keras_version,
                    "numpy_version": np.__version__, "h5py_version": h5py.__version__,
                    "hdf5_version": h5py.version.hdf5_version,
                    "parameters": sum(value.size for value in arrays.values()),
                    "tensor_bytes": sum(value.nbytes for value in arrays.values()),
                    "orientation": "Keras input-output; no transpose, clipping, or numeric conversion"}


def execute(args):
    destination = Path(args.out).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    source, inputs = {}, None
    clock, started_ns, elapsed_ns = None, None, None
    old_handler = signal.getsignal(signal.SIGALRM)

    def alarm(_signum, _frame):
        raise TimeoutError("extraction wall cap")

    def check():
        nonlocal elapsed_ns
        elapsed_ns = None  # A failing native read must not leave a valid terminal time.
        elapsed_ns = clock.now_ns() - started_ns
        if elapsed_ns >= LIMITS["wall_seconds"] * 10**9:
            raise TimeoutError("extraction suspend-inclusive wall cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == "darwin" else rss * 1024
        require(rss <= LIMITS["rss_bytes"], "extraction RSS cap")
        require(sum(p.stat().st_size for p in destination.iterdir() if p.is_file())
                <= LIMITS["output_bytes"], "extraction output cap")

    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        source = {str(Path(__file__).resolve()): sha(Path(__file__))}
        require(sha(CLOCK_PATH) == CLOCK_PIN, "qualified clock source")
        source[str(CLOCK_PATH)] = CLOCK_PIN
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.suspend_clock import SuspendClock
        clock = SuspendClock()
        started_ns = clock.now_ns()
        write(destination / "started.json", {"status": "started", "version": VERSION,
              "started_at": datetime.now(UTC).isoformat(), "request": vars(args),
              "source_sha256": source, "limits": LIMITS,
              "clock_backend": clock.backend, "started_ns": started_ns})
        directory = Path(args.assets).absolute()
        inputs = authenticate(directory)
        check()
        (destination / "config-opcodes.txt").write_text(config_opcodes(directory / "zoo_model_2_3_2.config"))
        arrays, metadata = read_tensors(directory / "zoo_model_2_3_2", check)
        import numpy as np
        with (destination / "tensors.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        write(destination / "tensor-metadata.json", {"version": VERSION, "configuration": CONFIG,
              "configuration_scope": "Known values from pinned config; opcode inspection only.", **metadata})
        check()
        require(authenticate(directory) == inputs, "unchanged inputs")
        require(all(sha(Path(path)) == pin for path, pin in source.items()), "unchanged extractor source")
        files = {p.name: descriptor(p) for p in destination.iterdir() if p.is_file()}
        require(set(files) == {"started.json", "config-opcodes.txt", "tensors.npz", "tensor-metadata.json"},
                "exact extraction payloads")
        receipt = {"status": "completed", "version": VERSION, "source_sha256": source,
                   "inputs": inputs, "files": files, "python": platform.python_version(),
                   "runtime": {k: metadata[k] for k in ("numpy_version", "h5py_version", "hdf5_version")},
                   "limits": LIMITS, "wall_seconds": elapsed_ns / 10**9,
                   "clock_backend": clock.backend, "started_ns": started_ns,
                   "elapsed_ns": elapsed_ns, "timing_available": True,
                   "tensor_keys": list(SHAPES), "parameters": metadata["parameters"],
                   "tensor_bytes": metadata["tensor_bytes"], "model_calls": 0, "simulator_calls": 0,
                   "pickle_executed": False, "inference_qualified": False,
                   "scope": "Byte-authenticated tensor extraction only. No inference or training; "
                            "not evidence of numerical policy parity or benchmark asset identity."}
        write(destination / "receipt.json", receipt)
        check()
        print(json.dumps({"status": "completed", "receipt_sha256": sha(destination / "receipt.json")}))
        return receipt
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (destination / "receipt.json").exists():
                (destination / "receipt.json").rename(destination / "late-completion.json")
            write(destination / "failed.json", {"status": "failed", "version": VERSION,
                  "error": repr(exc), "source_sha256": source, "inputs": inputs,
                  "wall_seconds": None if elapsed_ns is None else elapsed_ns / 10**9,
                  "timing_available": elapsed_ns is not None,
                  "clock_backend": None if clock is None else clock.backend,
                  "model_calls": 0, "simulator_calls": 0, "inference_qualified": False})
        except BaseException as cleanup:  # noqa: BLE001 - preserve the primary failure
            exc.add_note(f"Failure receipt could not be written: {cleanup!r}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default=str(ROOT / "tmp/otto-pretrained-reference-01"))
    parser.add_argument("--out", required=True)
    return execute(parser.parse_args())


if __name__ == "__main__":
    main()
