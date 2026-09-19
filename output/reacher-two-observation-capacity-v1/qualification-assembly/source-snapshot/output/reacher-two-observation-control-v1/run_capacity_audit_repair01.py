"""Run one reviewed saved-only audit repair and retain its actual process exit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "output/reacher-two-observation-control-v1/capacity_audit_repair.py"
ATTEMPT = ROOT / "output/reacher-two-observation-capacity-v1/attempt-02"
OUT = ATTEMPT.with_name("attempt-02-audit-repair-01")
PROCESS = ATTEMPT.with_name("attempt-02-audit-repair-01-process")
COMPLETED_SHA = "416d8b10fd8bd36758db2a03f9b62b4270cfcc079b73558b87072cf593739e13"
FAILED_PATH = "output/reacher-two-observation-capacity-v1/attempt-02-audit/failed.json"
FAILED_SHA = "5806bf336dc6a76e11b24d9d39e10351ea24ac2cb1c3de48b4f8295da023d138"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--expected-helper-sha256":
        raise ValueError("Supply independently reviewed helper SHA256")
    expected = sys.argv[2]
    if sha(HELPER) != expected or sha(ATTEMPT / "completed.json") != COMPLETED_SHA:
        raise ValueError("Reviewed helper or original measurement changed")
    if sha(ROOT / FAILED_PATH) != FAILED_SHA:
        raise ValueError("Original failed audit changed")
    if OUT.exists() or PROCESS.exists():
        raise FileExistsError("Never overwrite or retry an existing audit repair")
    PROCESS.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(HELPER), str(ATTEMPT), str(OUT),
               "--completed-sha256", COMPLETED_SHA,
               "--failed-audit-receipt", FAILED_PATH,
               "--failed-audit-sha256", FAILED_SHA, "--authorize-saved-audit"]
    env = {**os.environ, "PYTHONPATH": "src:scripts:tests", "PYTHONUNBUFFERED": "1"}
    with (PROCESS / "as-run-driver.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    write(PROCESS / "started.json", {
        "scope": "engineering_capacity_saved_audit_only_no_efficacy",
        "driver_sha256": sha(Path(__file__)), "helper_sha256": expected,
        "command": command, "cwd": str(ROOT), "unix_time": time.time(),
        "audit_cap_seconds": 3600, "automatic_retry": False,
        "new_training_updates": 0, "new_model_calls": 0,
    })
    begin = time.monotonic()
    with (PROCESS / "audit.stdout").open("xb") as stdout, (PROCESS / "audit.stderr").open("xb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    receipt = {"phase": "saved_audit_repair", "exit_code": result.returncode,
               "wall_seconds": time.monotonic() - begin, "command": command,
               "automatic_retry": False}
    write(PROCESS / "process.json", receipt)
    write(PROCESS / ("completed.json" if result.returncode == 0 else "failed.json"), {
        "status": "completed" if result.returncode == 0 else "failed", **receipt,
        "limits": "Saved-only capacity audit; independent sizing review remains required.",
    })
    print(json.dumps(receipt), flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
