"""Launch one reviewed engineering probe and retain both actual process exits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = ROOT / "output/reacher-two-observation-capacity-v1/attempt-01"
HELPER = ROOT / "output/reacher-two-observation-control-v1/capacity_probe.py"
REHEARSAL = "output/reacher-two-observation-rehearsal-v1/attempt-01"
QUALIFICATION_SHA = "6066f071cbe3ec48f45efdbfce46d2c397da14dce39063188ac838e213cafce9"
AUDIT_SHA = "0b82b728033078665537e5c79f5e6ef52c94ed8feb8451cff04a20e7a763e60b"


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--expected-helper-sha256":
        raise ValueError("Supply independently reviewed helper SHA256")
    expected = sys.argv[2]
    assert digest(HELPER) == expected, "Reviewed helper changed before launch"
    assert not ATTEMPT.exists(), "Never overwrite or retry an existing attempt"
    assert not ATTEMPT.with_name(ATTEMPT.name + "-audit").exists()
    external = ATTEMPT.with_name(ATTEMPT.name + "-process")
    external.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    env = {**os.environ, "PYTHONPATH": "src:scripts:tests", "PYTHONUNBUFFERED": "1"}
    phases = []
    write(external / "started.json", {
        "scope": "engineering_capacity_only_no_scientific_efficacy",
        "driver_sha256": digest(Path(__file__)), "helper_sha256": expected,
        "execution_cap_seconds": 3600, "audit_cap_seconds": 3600,
        "cap_basis": "Engineering pilot limits; scientific limits await measured projection and review.",
        "automatic_retry": False, "unix_time": time.time(),
    })
    for phase in ("run", "audit"):
        assert digest(HELPER) == expected, "Reviewed helper changed between phases"
        command = [sys.executable, str(HELPER), phase, str(ATTEMPT), "--execute-engineering"]
        if phase == "run":
            command += [
                "--rehearsal-plan", REHEARSAL + "/plan.json",
                "--rehearsal-audit", REHEARSAL + "/audit/receipt.json",
                "--rehearsal-audit-sha256", AUDIT_SHA,
                "--rehearsal-execution", REHEARSAL + "/execution",
                "--rehearsal-qualification", REHEARSAL + "/qualification.json",
                "--rehearsal-qualification-sha256", QUALIFICATION_SHA,
                "--cap-seconds", "3600", "--audit-cap-seconds", "3600",
            ]
        else:
            command += ["--completed-sha256", digest(ATTEMPT / "completed.json")]
        phase_begin = time.monotonic()
        write(external / f"{phase}-invocation.json", {"command": command, "cwd": str(ROOT)})
        with (external / f"{phase}.stdout").open("xb") as stdout, (external / f"{phase}.stderr").open("xb") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        receipt = {"phase": phase, "exit_code": result.returncode,
                   "wall_seconds": time.monotonic() - phase_begin, "command": command}
        write(external / f"{phase}-process.json", receipt)
        phases.append(receipt)
        print(json.dumps(receipt), flush=True)
        if result.returncode:
            write(external / "failed.json", {"status": "failed", "phase": phase,
                  "wall_seconds": time.monotonic() - begin, "phases": phases,
                  "automatic_retry": False})
            return result.returncode
    write(external / "completed.json", {"status": "completed", "engineering": True,
          "wall_seconds": time.monotonic() - begin, "phases": phases,
          "limits": "Capacity measurement only; independent scientific sizing review remains required."})
    return 0


if __name__ == "__main__":
    sys.exit(main())
