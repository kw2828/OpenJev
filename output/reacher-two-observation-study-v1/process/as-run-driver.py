"""Run and audit one already published frozen study; retain actual process exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = "evidence/reacher-two-observation-study-v1/protocol/plan.json"
FREEZE = "evidence/reacher-two-observation-study-v1/protocol/freeze.json"
EXECUTION = "runs/reacher-two-observation-study-v1/attempt"
AUDIT = "evidence/reacher-two-observation-study-v1/audit"
PROCESS = "output/reacher-two-observation-study-v1/process"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--published-commit", required=True)
    parser.add_argument("--execute-frozen-study", action="store_true")
    args = parser.parse_args()
    if not args.execute_frozen_study:
        raise ValueError("Explicit frozen-study execution required")
    commit = args.published_commit
    if len(commit) != 40 or not set(commit) <= set("0123456789abcdef"):
        raise ValueError("Exact published Git commit required")
    for name, expected in ((PLAN, args.plan_sha256), (FREEZE, args.freeze_sha256)):
        payload = (ROOT / name).read_bytes()
        blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        if digest(payload) != expected or digest(blob) != expected:
            raise ValueError("Published plan or freeze bytes differ: " + name)
    # Verify the supplied commit is public before any scientific process starts.
    remote = subprocess.run(["gh", "api", f"repos/kw2828/OpenJev/commits/{commit}",
                             "--jq", ".sha"], cwd=ROOT, capture_output=True, check=True)
    if remote.stdout.decode().strip() != commit:
        raise ValueError("Published commit not independently confirmed")
    for name in (EXECUTION, AUDIT, PROCESS):
        if (ROOT / name).exists():
            raise FileExistsError("Never overwrite or retry study output: " + name)
    out = ROOT / PROCESS
    out.mkdir(parents=True, exist_ok=False)
    with (out / "as-run-driver.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    write(out / "started.json", {"status": "started", "engineering": False,
        "plan_sha256": args.plan_sha256, "freeze_sha256": args.freeze_sha256,
        "published_commit": commit, "driver_sha256": digest(Path(__file__).read_bytes()),
        "automatic_retry": False, "unix_time": time.time()})
    env = {**os.environ, "PYTHONPATH": "src:scripts:tests", "PYTHONUNBUFFERED": "1"}
    commands = [
        ("execution", [sys.executable, "scripts/reacher_two_observation_study.py", "run",
                       "--plan", PLAN, "--expected-plan-sha256", args.plan_sha256,
                       "--out", EXECUTION, "--freeze", FREEZE,
                       "--expected-freeze-sha256", args.freeze_sha256]),
        ("audit", [sys.executable, "scripts/audit_reacher_two_observation_study.py",
                   "--plan", PLAN, "--expected-plan-sha256", args.plan_sha256,
                   "--execution", EXECUTION, "--out", AUDIT,
                   "--freeze-path", FREEZE, "--freeze-sha256", args.freeze_sha256]),
    ]
    begin, phases = time.monotonic(), []
    for phase, command in commands:
        write(out / f"{phase}-invocation.json", {"command": command, "cwd": str(ROOT)})
        tick = time.monotonic()
        with (out / f"{phase}.stdout").open("xb") as stdout, (out / f"{phase}.stderr").open("xb") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        receipt = {"phase": phase, "exit_code": result.returncode,
                   "wall_seconds": time.monotonic() - tick, "command": command}
        phases.append(receipt)
        write(out / f"{phase}-process.json", receipt)
        print(json.dumps(receipt), flush=True)
        if result.returncode:
            write(out / "failed.json", {"status": "failed", "phase": phase,
                "phases": phases, "wall_seconds": time.monotonic() - begin,
                "automatic_retry": False})
            return result.returncode
    write(out / "completed.json", {"status": "completed", "engineering": False,
        "phases": phases, "wall_seconds": time.monotonic() - begin,
        "limits": "Successful process exits do not by themselves establish a positive scientific result."})
    return 0


if __name__ == "__main__":
    sys.exit(main())
