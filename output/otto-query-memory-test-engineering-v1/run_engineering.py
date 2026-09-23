"""Qualify only the new conditional TEST consumer on fabricated inputs."""
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SELF = "output/otto-query-memory-test-engineering-v1/run_engineering.py"
PRIOR = "output/otto-query-memory-training-engineering-v1/attempt-02/receipt.json"
PRIOR_SHA256 = "a8c05d697b62ae80ad21973a24fd327cdf8afd29fc759172c93ce8edc02dd169"
NEW = [
    "scripts/evaluate_otto_query_memory.py",
    "scripts/audit_otto_query_memory_test.py",
    "tests/test_evaluate_otto_query_memory.py",
    "tests/test_audit_otto_query_memory_test.py",
]
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS", "PYTHONDONTWRITEBYTECODE",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}


def descriptor(path):
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def group_absent(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def main():
    if (len(sys.argv) != 2 or not sys.argv[1].startswith("attempt-")
            or Path(sys.argv[1]).name != sys.argv[1]):
        raise ValueError("one exclusive attempt directory name required")
    output = BASE / sys.argv[1]
    output.mkdir(exist_ok=False)
    prior_pin = descriptor(ROOT / PRIOR)
    if prior_pin["sha256"] != PRIOR_SHA256:
        raise ValueError("immutable successful training qualification required")
    prior = json.loads((ROOT / PRIOR).read_text())
    if prior["status"] != "passed" or prior["source_before"] != prior["source_after"]:
        raise ValueError("unchanged qualified inherited sources required")
    bound = sorted(set(prior["source_after"]) | set(NEW) | {SELF})

    def pins():
        return {name: descriptor(ROOT / name) for name in bound}

    before = pins()
    if any(before[name] != pin for name, pin in prior["source_after"].items()):
        raise ValueError("live training source changed; qualification forbidden")
    commands = [
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(output / "pytest-temp"),
         *[name for name in NEW if name.startswith("tests/")]],
        [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *NEW, SELF],
    ]
    started = {
        "scope": "fabricated conditional TEST engineering only; no empirical arrays or checkpoints",
        "source_before": before, "prior_qualification": {"path": PRIOR, **prior_pin},
        "commands": commands, "thread_environment": THREAD_ENV, "python": sys.version,
        "platform": platform.platform(), "per_command_timeout_seconds": 180,
        "cwd": str(ROOT), "unix_time": time.time(),
    }
    write(output / "started.json", started)
    outcomes = []
    for index, command in enumerate(commands):
        log = output / f"command-{index + 1}.log"
        begin = time.monotonic()
        timed_out = False
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **THREAD_ENV},
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        absent = group_absent(child.pid)
        if not absent:
            os.killpg(child.pid, signal.SIGKILL)
        outcomes.append({
            "command": command, "returncode": code, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - begin, "pid": child.pid,
            "reaped": True, "group_absent": absent, "log": log.name, **descriptor(log),
        })
    probes = list((output / "pytest-temp").rglob("test-audit-capacity.json"))
    capacity, capacity_error = None, None
    try:
        if len(probes) != 1 or not probes[0].is_file() or probes[0].is_symlink():
            raise ValueError("exactly one fabricated TEST scalar-audit capacity probe required")
        payload = probes[0].read_bytes()
        with (output / "test-audit-capacity.json").open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        capacity = json.loads(payload)
        periods = capacity["periods"]
        if not (isinstance(periods, list) and len(periods) == 2
                and {row["query_period"] for row in periods} == {4, 8}
                and all(row["episodes"] == 6 and row["rows"] == 13128
                        and math.isfinite(row["seconds"]) and row["seconds"] > 0
                        for row in periods)):
            raise ValueError("both fabricated P4/P8 audit timings required")
        projected = max(row["seconds"] / row["rows"] for row in periods) * 3780864 * 2
        if not (capacity["admitted"] is True
                and capacity["empirical_reads"] == capacity["model_calls"] == 0
                and capacity["worst_case_rows"] == 3780864
                and capacity["multiplier"] == 2
                and capacity["reserve_seconds"] == 120
                and capacity["audit_cap_seconds"] == 600
                and capacity["threshold_seconds"] == 480
                and math.isfinite(capacity["projected_seconds"])
                and math.isclose(capacity["projected_seconds"], projected, rel_tol=1e-12)
                and 0 < capacity["projected_seconds"] <= 480):
            raise ValueError("fabricated TEST scalar-audit capacity did not admit its frozen cap")
    except (ValueError, KeyError, TypeError, OSError) as error:
        capacity_error = repr(error)
    after = pins()
    passed = (before == after and descriptor(ROOT / PRIOR) == prior_pin
              and capacity_error is None
              and all(row["returncode"] == 0 and not row["timed_out"]
                      and row["group_absent"] for row in outcomes))
    receipt = {
        "scope": started["scope"], "status": "passed" if passed else "failed",
        "source_before": before, "source_after": after,
        "prior_qualification": started["prior_qualification"], "commands": outcomes,
        "files": {path.name: descriptor(path) for path in output.iterdir() if path.is_file()},
        "empirical_data_reads": 0, "scientific_optimizer_steps": 0,
        "scientific_model_calls": 0, "fabricated_model_calls_exercised": True,
        "audit_capacity": capacity, "audit_capacity_error": capacity_error,
        "scientific_performance_claim": False,
    }
    write(output / "receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"], "receipt": str(output / "receipt.json"),
        "commands": [{key: row[key] for key in
                      ("returncode", "timed_out", "group_absent", "elapsed_seconds")}
                     for row in outcomes],
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
