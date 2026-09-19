"""Retain actual process exits for a single explicit engineering rehearsal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = ROOT / "output/reacher-two-observation-rehearsal-v1/attempt-01"


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    begin = time.monotonic()
    assert not ATTEMPT.exists(), "Retain prior attempts; never overwrite or retry implicitly"
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    external = ATTEMPT.parent / "attempt-01-process"
    external.mkdir(exist_ok=False)
    phases = []
    env = {**os.environ, "PYTHONPATH": "src:scripts:tests", "PYTHONUNBUFFERED": "1"}
    write(external / "started.json", {
        "scope": "engineering_only_no_scientific_efficacy",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_cap_seconds": 600, "audit_cap_seconds": 600,
        "automatic_retry": False, "unix_time": time.time(),
    })
    commands = [
        ("prepare", "from reacher_two_observation_fixture import prepare_fixture; "
         f"prepare_fixture({str(ATTEMPT)!r},cap_seconds=600,audit_cap_seconds=600)"),
        ("execution", "from pathlib import Path; import json; "
         "from reacher_two_observation_fixture import HistoryFixture,run_fixture; "
         f"p=Path({str(ATTEMPT)!r}); "
         "d=json.loads((p/'preparation-completed.json').read_text())['plan_sha256']; "
         "run_fixture(HistoryFixture(p,p/'plan.json',d,p/'execution'))"),
        ("audit", "from pathlib import Path; import json; "
         "from reacher_two_observation_fixture import HistoryFixture,audit_fixture; "
         f"p=Path({str(ATTEMPT)!r}); "
         "d=json.loads((p/'preparation-completed.json').read_text())['plan_sha256']; "
         "audit_fixture(HistoryFixture(p,p/'plan.json',d,p/'execution'),p/'audit')"),
    ]
    for phase, code in commands:
        command = [sys.executable, "-c", code]
        phase_begin = time.monotonic()
        write(external / f"{phase}-invocation.json", {"command": command, "cwd": str(ROOT)})
        with (external / f"{phase}.stdout").open("xb") as stdout, (external / f"{phase}.stderr").open("xb") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
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
          "limits": "Actual full-tree engineering integration; no scientific performance result."})
    return 0


if __name__ == "__main__":
    sys.exit(main())
