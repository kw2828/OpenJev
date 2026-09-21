"""Fixed twelve-operation diagnostic; no checkpoint, model, or simulator imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else 1024 * value)


def child(name, plan):
    out = HERE / name
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {"status": "started", "runtime": name, "pid": os.getpid(),
             "started_at": datetime.now(UTC).isoformat(), "plan_sha256": sha(HERE / "plan.json"),
             "source_sha256": sha(__file__), "results": [], "attempted_matmuls": 0,
             "model_calls": 0, "checkpoint_reads": 0, "simulator_calls": 0}
    write(out / "started.json", {**state, "executable": sys.executable})

    def check():
        if time.monotonic() - started >= 60 or rss() > 2 * 1024**3:
            raise RuntimeError("child time/RSS cap")

    def event(data):
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
            f.flush()

    try:
        import numpy as np
        expected_version = "2.2.6" if name == "isolated" else "2.5.3"
        if np.__version__ != expected_version:
            raise ValueError("runtime NumPy version differs from plan")
        state.update(numpy_version=np.__version__, python=platform.python_version(),
                     numpy_build=np.__config__.CONFIG,
                     environment={key: os.environ.get(key) for key in plan["environment"]})
        phases = ["fresh_numpy", "after_tensorflow_import"] if name == "isolated" else ["fresh_numpy"]
        for phase in phases:
            if phase == "after_tensorflow_import":
                event({"event": "tensorflow_import_attempt"})
                import tensorflow as tf
                tf.config.set_visible_devices([], "GPU")
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
                if tf.__version__ != "2.20.0" or tf.config.get_visible_devices("GPU"):
                    raise ValueError("TensorFlow runtime/device mismatch")
                state["tensorflow_version"] = tf.__version__
                event({"event": "tensorflow_import_return", "version": tf.__version__})
            check()
            for width in (11025, 1024):
                for left_value in (0.0, 1 / 16):
                    check()
                    left = np.full((1, width), left_value, dtype=np.float32)
                    right = np.full((width, 1024), 1 / 16, dtype=np.float32)
                    array_bytes = left.nbytes + right.nbytes + 1024 * 4
                    if array_bytes > 512 * 1024**2:
                        raise RuntimeError("array allocation cap")
                    record = {"phase": phase, "width": width, "left_value": left_value,
                              "right_value": 1 / 16, "left_shape": [1, width],
                              "right_shape": [width, 1024], "dtype": "float32",
                              "expected": 0.0 if left_value == 0 else width / 256,
                              "allocated_array_bytes_including_result": array_bytes}
                    state["attempted_matmuls"] += 1
                    event({"event": "matmul_attempt", "ordinal": state["attempted_matmuls"], **record})
                    begin = time.monotonic()
                    result = None
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            with np.errstate(over="warn", invalid="warn", divide="warn", under="warn"):
                                result = left @ right  # The only matrix multiplication in this diagnostic.
                            record["exception"] = None
                        except BaseException as exc:
                            record["exception"] = repr(exc)
                    record["seconds"] = time.monotonic() - begin
                    record["warnings"] = [{"category": type(w.message).__name__, "message": str(w.message)}
                                          for w in caught]
                    if result is not None:
                        finite = bool(np.isfinite(result).all())
                        error = float(np.max(np.abs(result.astype(np.float64) - record["expected"]))) if finite else None
                        record.update(returned=True, shape=list(result.shape), finite=finite,
                                      exact=bool(np.all(result == np.float32(record["expected"]))),
                                      maximum_absolute_error=error,
                                      minimum=float(result.min()) if finite else None,
                                      maximum=float(result.max()) if finite else None,
                                      output_sha256=hashlib.sha256(result.tobytes()).hexdigest())
                    else:
                        record.update(returned=False, finite=False, exact=False, maximum_absolute_error=None)
                    state["results"].append(record)
                    event({"event": "matmul_result", "ordinal": state["attempted_matmuls"], **record})
                    del result, left, right
                    check()
        expected = 8 if name == "isolated" else 4
        if state["attempted_matmuls"] != expected or len(state["results"]) != expected:
            raise ValueError("exact primitive coverage")
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    state.update(wall_seconds=time.monotonic() - started, peak_rss_bytes=rss(),
                 finished_at=datetime.now(UTC).isoformat())
    write(out / "receipt.json", state)
    return 0 if state["status"] == "completed" else 1


def parent(plan, plan_pin):
    if sha(HERE / "plan.json") != plan_pin or sha(__file__) != plan["source_sha256"]:
        raise ValueError("frozen plan/source mismatch")
    write(HERE / "started.json", {"plan_sha256": plan_pin, "source_sha256": sha(__file__),
                                   "started_at": datetime.now(UTC).isoformat(), "status": "started"})
    records = []
    for name, executable in plan["runtimes"].items():
        command = [executable, str(Path(__file__).resolve()), "--child", name,
                   "--plan-sha256", plan_pin]
        started = time.monotonic()
        launch = {"command": command, "cwd": str(ROOT), "environment": plan["environment"],
                  "timeout_seconds": 60, "started_at": datetime.now(UTC).isoformat()}
        write(HERE / f"{name}-launch.json", launch)
        try:
            completed = subprocess.run(command, cwd=ROOT, env={**os.environ, **plan["environment"]},
                                       text=True, capture_output=True, timeout=60)
            stdout, stderr, code, timed_out = completed.stdout, completed.stderr, completed.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code, timed_out = None, True
        (HERE / f"{name}.stdout.txt").write_text(stdout)
        (HERE / f"{name}.stderr.txt").write_text(stderr)
        record = {"runtime": name, "exit_code": code, "timed_out": timed_out,
                  "wall_seconds": time.monotonic() - started,
                  "finished_at": datetime.now(UTC).isoformat()}
        records.append(record)
        write(HERE / f"{name}-terminal.json", record)
    receipt = {"status": "completed" if all(r["exit_code"] == 0 for r in records) else "failed",
               "plan_sha256": plan_pin, "source_sha256": sha(__file__), "children": records,
               "scope": plan["scope"], "model_calls": 0, "checkpoint_reads": 0, "simulator_calls": 0,
               "files": {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                         for p in sorted(HERE.rglob("*")) if p.is_file()}}
    if sha(__file__) != plan["source_sha256"] or sha(HERE / "plan.json") != plan_pin:
        receipt.update(status="failed", error="plan/source changed")
    write(HERE / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "children": records,
                      "receipt_sha256": sha(HERE / "receipt.json")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("isolated", "main"))
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    with (HERE / "plan.json").open() as stream:
        frozen = json.load(stream)
    if sha(HERE / "plan.json") != args.plan_sha256 or sha(__file__) != frozen["source_sha256"]:
        raise ValueError("plan/source identity before imports")
    sys.exit(child(args.child, frozen) if args.child else parent(frozen, args.plan_sha256))
