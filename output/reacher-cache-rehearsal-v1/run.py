"""One complete retained engineering rehearsal, with no scientific outcomes."""
import hashlib
import json
import time
from pathlib import Path

from reacher_cache_fixture import prepare_fixture, run_fixture, audit_fixture

BASE=Path(__file__).resolve().parent
started=time.monotonic()

def write(name,value):
    with (BASE/name).open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n")

write("started.json",{"scope":"engineering only; tiny synthetic cohort; no efficacy claim",
    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"unix_time":time.time()})
try:
    case=prepare_fixture(BASE/"attempt-01",cap_seconds=300,audit_cap_seconds=300)
    result=run_fixture(case)
    print(json.dumps({"engineering_execution":result}),flush=True)
    audit=audit_fixture(case,case.root/"audit")
    write("completed.json",{"status":"completed","scope":"engineering only; no scientific qualification",
        "plan_sha256":case.digest,"execution":result,"audit_receipt_sha256":hashlib.sha256((case.root/"audit/receipt.json").read_bytes()).hexdigest(),
        "wall_seconds":time.monotonic()-started})
    print(json.dumps({"status":"engineering_rehearsal_completed","wall_seconds":time.monotonic()-started}),flush=True)
except BaseException as e:
    write("failed.json",{"status":"failed","error":repr(e),"wall_seconds":time.monotonic()-started,
        "scope":"engineering only; preserve attempted execution and audit"})
    raise
