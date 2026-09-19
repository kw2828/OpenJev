"""One fixed necessary control-opportunity screen; no training or sweeps."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from openjev.research.drive_task import (
    HORIZON,
    METHODS,
    PANELS,
    SEEDS,
    EpisodeFailure,
    make_streams,
    run_episode,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/openjev/research/drive_task.py", "src/openjev/research/drive_observers.py",
           "tests/test_drive_task.py", "tests/test_drive_observers.py",
           "scripts/run_drive_qualification.py", "research/drive-qualification-protocol.md",
           "scripts/report_drive_qualification.py", "tests/test_report_drive_qualification.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "streams").mkdir()
    for seed in SEEDS:
        np.savez_compressed(output / f"streams/seed-{seed}.npz", **make_streams(seed))
    write(output / "protocol.json", {
        "version": "drive-qualification-v1", "seeds": list(SEEDS), "panels": list(PANELS),
        "methods": list(METHODS), "horizon": HORIZON, "episodes": 252,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "streams_sha256": {p.name: sha(p) for p in sorted((output / "streams").iterdir())},
        "rule": {"required_panels": list(PANELS[1:]), "relative_improvement": .10,
                 "absolute_mse_improvement": .001, "paired_wins": 9, "maximum_divergent_episodes": 0},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": importlib.metadata.version("numpy")},
    })
    print(json.dumps({"prepared": 12, "protocol_sha256": sha(output / "protocol.json")}), flush=True)


def run(output, protocol_pin):
    if not protocol_pin or sha(output / "protocol.json") != protocol_pin:
        raise ValueError("An exact externally supplied protocol SHA-256 is required")
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["runtime"] != {"python": platform.python_version(), "platform": platform.platform(),
                               "numpy": importlib.metadata.version("numpy")}:
        raise ValueError("Runtime changed since freeze")
    for name, digest in protocol["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    for name, digest in protocol["streams_sha256"].items():
        if sha(output / "streams" / name) != digest:
            raise ValueError(f"Frozen stream changed: {name}")
    run_dir = output / "run-01"
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    records = []
    stem = None
    pending_trace = None
    pending_native_steps = 0
    try:
        for seed in SEEDS:
            with np.load(output / f"streams/seed-{seed}.npz") as archive:
                streams = {k: archive[k].copy() for k in archive.files}
            for value in streams.values():
                value.flags.writeable = False
            for panel in PANELS:
                for method in METHODS:
                    stem = f"{panel}-{seed}-{method}"
                    pending_trace = None
                    pending_native_steps = 0
                    before = time.perf_counter()
                    trace, metrics, costs = run_episode(method, panel, streams)
                    pending_trace = trace
                    pending_native_steps = HORIZON
                    wall = time.perf_counter() - before
                    np.savez_compressed(run_dir / f"{stem}.npz", **trace)
                    record = {"seed": seed, "panel": panel, "method": method,
                              "trace": f"{stem}.npz", "trace_sha256": sha(run_dir / f"{stem}.npz"),
                              "metrics": metrics, "costs": costs,
                              "episode_wall_seconds": wall, "native_steps": HORIZON}
                    write(run_dir / f"{stem}.json", record)
                    records.append(record)
                    pending_trace = None
                    pending_native_steps = 0
            print(json.dumps({"seed": seed, "episodes": len(records), "of": 252}), flush=True)
        if sha(output / "protocol.json") != protocol_pin:
            raise ValueError("Protocol changed during execution")
        for name, digest in protocol["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Source changed during execution: {name}")
        for name, digest in protocol["streams_sha256"].items():
            if sha(output / "streams" / name) != digest:
                raise ValueError(f"Stream changed during execution: {name}")
        write(run_dir / "records.json", records)
        write(run_dir / "completed.json", {
            "status": "complete", "episodes": len(records), "native_steps": HORIZON * len(records),
            "wall_seconds": time.perf_counter() - started, "protocol_sha256": sha(output / "protocol.json"),
            "records_sha256": sha(run_dir / "records.json"), "neural_fits": 0, "model_api_calls": 0,
        })
    except BaseException as exc:
        try:
            prefix = None
            partial_steps = pending_native_steps
            if isinstance(exc, EpisodeFailure):
                prefix = f"{stem}-failed-prefix.npz"
                np.savez_compressed(run_dir / prefix, **exc.trace)
                partial_steps = exc.native_steps
            elif pending_trace is not None:
                prefix = f"{stem}-failed-prefix.npz"
                np.savez_compressed(run_dir / prefix, **pending_trace)
            write(run_dir / "failed.json", {
                "status": "failed", "finished_episodes": len(records), "failing_episode": stem,
                "failed_prefix": prefix, "partial_native_steps": partial_steps,
                "finished_episode_native_steps": HORIZON * len(records),
                "error_type": type(exc).__name__, "error": str(exc), "protocol_sha256": protocol_pin})
        except BaseException as preservation_error:  # noqa: BLE001 - retain the original failure
            exc.add_note(f"Failure preservation also failed: {preservation_error!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/drive-qualification-v1")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    else:
        run(args.output, args.protocol_sha256)
