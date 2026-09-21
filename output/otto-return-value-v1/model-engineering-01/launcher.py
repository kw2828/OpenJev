"""One bounded synthetic model qualification; no science inputs or native calls."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = ("src/openjev/research/otto_return_value.py", "tests/test_otto_return_value.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    pins = {name: digest(ROOT / name) for name in SOURCES}
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    commands = (("pytest", [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", SOURCES[1]]),
                ("ruff", [str(ROOT / ".venv/bin/python"), "-m", "ruff", "check", *SOURCES]))
    results = []
    status = "failed"
    error = None
    try:
        for label, command in commands:
            phase_start = time.monotonic()
            remaining = 60 - (phase_start - started)
            if remaining <= 0:
                raise TimeoutError("whole synthetic qualification cap")
            with (OUT / f"{label}.log").open("x") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=remaining, check=False)
            results.append({"name": label, "command": command, "returncode": completed.returncode,
                            "seconds": time.monotonic() - phase_start,
                            "log_sha256": digest(OUT / f"{label}.log")})
        if all(item["returncode"] == 0 for item in results):
            status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    pins_after = {name: digest(ROOT / name) for name in SOURCES}
    if pins != pins_after or elapsed >= 60:
        status = "failed"
        error = error or "source changed or whole cap exceeded"
    receipt = {"status": status, "source_sha256": pins, "source_sha256_after": pins_after,
               "launcher_sha256": digest(Path(__file__)), "wall_seconds": elapsed, "wall_cap_seconds": 60,
               "cpu_threads": 1, "checks": results, "error": error,
               "peak_child_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
               * (1 if sys.platform == "darwin" else 1024),
               "scope": "Synthetic NumPy/Torch model unit fixtures and static lint only; tiny synthetic gradient updates included.",
               "science_data_reads": 0, "checkpoint_loads": 0, "simulator_calls": 0, "api_calls": 0}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    for item in results:
        print((OUT / f"{item['name']}.log").read_text()[-6000:])
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
