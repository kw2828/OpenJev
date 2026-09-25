"""One original fabricated-only qualification, preserving commands and source bytes."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "output/fsm-shift-engineering-v1/reader-qualification-01"
SOURCES = ("src/openjev/research/fsm_shift_data.py", "tests/test_fsm_shift_data.py")
ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
COMMANDS = [
    [".venv/bin/ruff", "check", *SOURCES],
    [".venv/bin/python", "-m", "pytest", "--noconftest", "-q", SOURCES[1]],
]


def pin(path):
    payload = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.UTC).isoformat()
    sources = {name: pin(ROOT / name) for name in SOURCES}
    old_reader = pin(ROOT / "src/openjev/research/fsm_data.py")
    for name in SOURCES:
        snapshot = DEST / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
        assert pin(snapshot) == sources[name]
    (DEST / "qualify.py").write_bytes(Path(__file__).read_bytes())
    save(DEST / "preflight.json", {"created_utc": started, "sources": sources,
         "unchanged_old_reader": old_reader, "commands": COMMANDS, "thread_env": ENV,
         "scope": "Fabricated NPZ archives only; no measured inputs, models, training or evaluation."})
    records = []
    error = None
    try:
        for index, command in enumerate(COMMANDS, 1):
            log = DEST / f"command-{index:02d}.log"
            begin = time.perf_counter()
            with log.open("xb") as handle:
                process = subprocess.run(command, cwd=ROOT, env={**os.environ, **ENV},
                                         stdout=handle, stderr=subprocess.STDOUT, check=False)
            records.append({"command": command, "returncode": process.returncode,
                            "seconds": time.perf_counter()-begin, "log": str(log), "log_pin": pin(log)})
            if process.returncode:
                break
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        unchanged = all(pin(ROOT / name) == value for name, value in sources.items())
        old_unchanged = old_reader == pin(ROOT / "src/openjev/research/fsm_data.py")
        passed = (error is None and unchanged and old_unchanged and len(records) == len(COMMANDS)
                  and all(record["returncode"] == 0 for record in records))
        receipt = {"status": "PASS" if passed else "FAIL", "started_utc": started,
                   "finished_utc": dt.datetime.now(dt.UTC).isoformat(), "sources": sources,
                   "sources_unchanged": unchanged, "old_reader_unchanged": old_unchanged,
                   "thread_env": ENV, "commands": records, "error": error,
                   "scope": "Only fabricated reader qualification; no empirical shift was launched."}
        save(DEST / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
