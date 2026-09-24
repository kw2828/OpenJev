"""Bound all new DEV-runner tests, lint and worst-horizon synthetic capacity."""
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
OLD_PLAN = ROOT / "output/otto-query-memory-v1/training-plan-01.json"
OLD_PLAN_PIN = "9ea6b7422116ca3a708e93d2339f4209ddad8a695feac9d31c590f758aac140a"
PRIOR = ROOT / "output/otto-residual-estimator-engineering-v1/attempt-02/receipt.json"
PRIOR_PIN = "2a9aab7e8d33fb6904335eeb38a22df6ec3a24982f5d6456c29ee54d45d143f8"
NEW = ["scripts/collect_otto_residual.py", "tests/test_collect_otto_residual.py",
       "scripts/otto_residual_lineage.py", "tests/test_otto_residual_lineage.py",
       "src/openjev/research/otto_residual_audit.py", "tests/test_otto_residual_audit.py",
       "scripts/run_otto_residual.py", "tests/test_run_otto_residual.py",
       "scripts/qualify_otto_residual_capacity.py"]
EXTRA = ["research/otto-residual-estimator-protocol.md", "research/otto-residual-estimator-engineering.md",
         "output/otto-residual-estimator-v1/seed-reservation-01.json",
         "output/otto-residual-estimator-v1/seed-reservation-02.json",
         "output/otto-residual-estimator-v1/seed-review-01.json"]
PREVIOUS_TESTS = [f"tests/test_otto_residual_{name}.py" for name in ("contract", "features", "replay", "gate")]
PREVIOUS_TESTS += ["tests/test_bayesian_score_memory.py"]
ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
       "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}


def descriptor(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


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
            or Path(sys.argv[1]).name != sys.argv[1] or Path.cwd() != ROOT):
        raise ValueError("exclusive attempt from OpenJev checkout required")
    if descriptor(OLD_PLAN)["sha256"] != OLD_PLAN_PIN or descriptor(PRIOR)["sha256"] != PRIOR_PIN:
        raise ValueError("immutable prior source and engineering anchors")
    frozen = json.loads(OLD_PLAN.read_text())["sources"]
    prior = json.loads(PRIOR.read_text())["source_after"]
    hits = json.loads((ROOT / EXTRA[-1]).read_text())["resolved_block_hits"]
    bound = sorted(set(frozen) | set(prior) | set(NEW) | set(EXTRA) | {SELF} | {r["path"] for r in hits})

    def pins():
        return {name: descriptor(ROOT / name) for name in bound}

    before = pins()
    if any(before[name]["sha256"] != pin for name, pin in frozen.items()) or any(before[name] != pin for name, pin in prior.items()):
        raise ValueError("prior145 frozen sources and qualified numerical integration stay unchanged")
    output = BASE / sys.argv[1]
    output.mkdir(exist_ok=False)
    tests = PREVIOUS_TESTS + [name for name in NEW if name.startswith("tests/")]
    lint = NEW + [SELF]
    commands = [
        (180, [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--basetemp", str(output / "pytest-temp"), *tests]),
        (60, [str(ROOT / ".venv/bin/ruff"), "check", "--no-cache", *lint]),
        (240, [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/qualify_otto_residual_capacity.py"),
               "--output", str(output / "capacity.json"), "--temp", str(output / "pytest-temp/capacity")]),
    ]
    write(output / "started.json", {"scope": "fabricated DEV-runner qualification only", "source_before": before,
          "commands": [{"cap_seconds": cap, "command": command} for cap, command in commands],
          "environment_overrides": ENV, "python": sys.version, "created_unix_ns": time.time_ns()})
    outcomes = []
    for index, (cap, command) in enumerate(commands):
        if index == 2 and any(row["returncode"] != 0 or row["timed_out"] or not row["group_absent"] for row in outcomes):
            break
        log = output / f"command-{index + 1}.log"
        begin, timeout = time.monotonic(), False
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **ENV},
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        group_absent = absent(child.pid)
        if not group_absent:
            os.killpg(child.pid, signal.SIGKILL)
        outcomes.append({"command": command, "cap_seconds": cap, "returncode": code, "timed_out": timeout,
                         "elapsed_seconds": time.monotonic() - begin, "pid": child.pid, "reaped": True,
                         "group_absent": group_absent, "log": log.name, **descriptor(log)})
        print(json.dumps({"command": index + 1, "returncode": code, "timed_out": timeout, "group_absent": group_absent}), flush=True)
    after = pins()
    unchanged = before == after and descriptor(OLD_PLAN)["sha256"] == OLD_PLAN_PIN and descriptor(PRIOR)["sha256"] == PRIOR_PIN
    passed = unchanged and len(outcomes) == 3 and all(row["returncode"] == 0 and not row["timed_out"] and row["group_absent"] for row in outcomes)
    receipt = {"scope": "fabricated runner and capacity qualification; no empirical execution", "status": "passed" if passed else "failed",
               "source_before": before, "source_after": after, "sources_unchanged": unchanged,
               "frozen_source_count": len(frozen), "commands": outcomes,
               "scientific_performance_claim": False,
               "empirical_data_access": "none by reviewed test/capacity sources; not OS-sandbox enforcement",
               "files": {p.name: descriptor(p) for p in output.iterdir() if p.is_file()}}
    write(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
