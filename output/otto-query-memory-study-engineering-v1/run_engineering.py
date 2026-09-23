"""Bounded fabricated qualification; outputs never count as scientific results."""
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
NEW = [
    "src/openjev/research/otto_query_memory.py",
    "src/openjev/research/otto_scheduled_predictor.py",
    "src/openjev/research/otto_query_memory_model.py",
    "src/openjev/research/otto_query_memory_data.py",
    "src/openjev/research/otto_query_memory_metrics.py",
    "scripts/collect_otto_query_memory.py",
    "tests/test_otto_query_memory.py",
    "tests/test_otto_scheduled_predictor.py",
    "tests/test_otto_query_memory_integration.py",
    "tests/test_otto_query_memory_model.py",
    "tests/test_otto_query_memory_data.py",
    "tests/test_otto_query_memory_metrics.py",
    "tests/test_collect_otto_query_memory.py",
]
BOUND = [*NEW, "pyproject.toml", "uv.lock",
         "research/otto-query-memory-protocol.md",
         "src/openjev/__init__.py", "src/openjev/research/__init__.py",
         "src/openjev/research/otto_cross_query_scores.py",
         "src/openjev/research/otto_prequery_scores.py",
         "src/openjev/research/otto_prequery_loss.py",
         "src/openjev/research/otto_score_forecast_data.py",
         "src/openjev/research/otto_protected_readout.py",
         "src/openjev/research/otto_protected_training_model.py",
         "output/otto-query-memory-study-engineering-v1/run_engineering.py"]
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}


def descriptor(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def pins():
    return {name: descriptor(ROOT / name) for name in BOUND}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if len(sys.argv) != 2 or not sys.argv[1].startswith("attempt-") or "/" in sys.argv[1]:
        raise ValueError("one exclusive attempt directory name required")
    output = BASE / sys.argv[1]
    output.mkdir(exist_ok=False)
    commands = [
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(output / "pytest-temp"), *[name for name in NEW if name.startswith("tests/")]],
        [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *NEW,
         "output/otto-query-memory-study-engineering-v1/run_engineering.py"],
    ]
    before = pins()
    started = {"scope": "fabricated engineering only; no empirical arrays, checkpoints or fitting",
               "source_before": before, "commands": commands, "thread_environment": THREAD_ENV,
               "python": sys.version, "platform": platform.platform(),
               "per_command_timeout_seconds": 180, "cwd": str(ROOT), "unix_time": time.time()}
    write(output / "started.json", started)
    outcomes = []
    environment = {**os.environ, **THREAD_ENV}
    for index, command in enumerate(commands):
        log = output / f"command-{index + 1}.log"
        begin = time.monotonic()
        timed_out = False
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        outcomes.append({"command": command, "returncode": code, "timed_out": timed_out,
                         "elapsed_seconds": time.monotonic() - begin, "pid": child.pid,
                         "reaped": True, "log": log.name, **descriptor(log)})
    after = pins()
    passed = before == after and all(row["returncode"] == 0 and not row["timed_out"] for row in outcomes)
    receipt = {"scope": started["scope"], "status": "passed" if passed else "failed",
               "source_before": before, "source_after": after, "commands": outcomes,
               "files": {path.name: descriptor(path) for path in output.iterdir() if path.is_file()},
               "empirical_data_reads": 0, "optimizer_steps": 0,
               "scientific_performance_claim": False}
    write(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json"),
                      "commands": [{k: r[k] for k in ("returncode", "timed_out", "elapsed_seconds")}
                                   for r in outcomes]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
