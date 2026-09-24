"""Qualify fresh frozen-feature estimator integration with fabricated tests only."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SELF = str(Path(__file__).resolve().relative_to(ROOT))
PLAN = ROOT / "output/otto-query-memory-v1/training-plan-01.json"
PLAN_HASH = "9ea6b7422116ca3a708e93d2339f4209ddad8a695feac9d31c590f758aac140a"
NEW = [
    "src/openjev/research/otto_residual_contract.py",
    "src/openjev/research/otto_residual_features.py",
    "src/openjev/research/otto_residual_replay.py",
    "src/openjev/research/otto_residual_gate.py",
    "tests/test_otto_residual_contract.py",
    "tests/test_otto_residual_features.py",
    "tests/test_otto_residual_replay.py",
    "tests/test_otto_residual_gate.py",
    "src/openjev/research/bayesian_score_memory.py",
    "tests/test_bayesian_score_memory.py",
]
DESIGN = "research/otto-residual-estimator-design.md"
TESTS = [name for name in NEW if name.startswith("tests/")]
ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)}


def descriptor(path):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def absent(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def main():
    if (len(sys.argv) != 2 or not sys.argv[1].startswith("attempt-")
            or Path(sys.argv[1]).name != sys.argv[1]):
        raise ValueError("exclusive attempt directory required")
    if Path.cwd() != ROOT:
        raise ValueError("run from the OpenJev checkout")
    if descriptor(PLAN)["sha256"] != PLAN_HASH:
        raise ValueError("unchanged frozen training plan required")
    sources = json.loads(PLAN.read_text())["sources"]
    if set(NEW) & set(sources):
        raise ValueError("engineering must remain separate from the closed study")
    bound = sorted(set(sources) | set(NEW) | {SELF, DESIGN})

    def pins():
        return {name: descriptor(ROOT / name) for name in bound}

    before = pins()
    if any(before[name]["sha256"] != expected for name, expected in sources.items()):
        raise ValueError("frozen source changed")
    output = BASE / sys.argv[1]
    output.mkdir(exist_ok=False)
    commands = [
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(output / "pytest-temp"), *TESTS],
        [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *NEW, SELF],
    ]
    scope = "prospective frozen-feature residual estimators; fabricated qualification only; no empirical execution"
    write(output / "started.json", {
        "scope": scope, "source_before": before, "commands": commands,
        "per_command_cap_seconds": 180, "environment_overrides": ENV,
        "python": sys.version, "created_unix_ns": time.time_ns(),
    })
    outcomes = []
    for index, command in enumerate(commands):
        log = output / f"command-{index + 1}.log"
        begin = time.monotonic()
        timeout = False
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **ENV},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        group_absent = absent(child.pid)
        if not group_absent:
            os.killpg(child.pid, signal.SIGKILL)
        outcomes.append({"command": command, "returncode": code, "timed_out": timeout,
                         "elapsed_seconds": time.monotonic() - begin, "pid": child.pid,
                         "reaped": True, "group_absent": group_absent, "log": log.name,
                         **descriptor(log)})
    after = pins()
    unchanged = before == after and descriptor(PLAN)["sha256"] == PLAN_HASH
    passed = unchanged and all(r["returncode"] == 0 and not r["timed_out"]
                               and r["group_absent"] for r in outcomes)
    receipt = {
        "scope": scope, "status": "passed" if passed else "failed",
        "source_before": before, "source_after": after,
        "frozen_source_count": len(sources), "sources_unchanged": unchanged,
        "commands": outcomes, "scientific_performance_claim": False,
        "empirical_data_access": "none by test-source review; not OS-sandbox enforcement",
        "files": {p.name: descriptor(p) for p in output.iterdir() if p.is_file()},
    }
    write(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json"),
                      "commands": [{k: r[k] for k in ("returncode", "timed_out", "group_absent")}
                                   for r in outcomes]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
