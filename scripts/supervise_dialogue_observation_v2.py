"""Suspend-aware process supervision; no model, data, or metric operations.

Only native continuous/BOOTTIME clocks admit success. Civil timestamps are
provenance. Cleanup may use suspend-excluding bounded waits after failure;
those waits can never turn a failure into success. No child artifact is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research import suspend_clock

VERSION = "dialogue-observation-supervision-v2"
POLL_SECONDS = 0.25
TERM_WAIT_SECONDS = 1.0
KILL_WAIT_SECONDS = 1.0
GROUP_POLL_SECONDS = 0.025
GROUP_POLLS = 10
NATIVE_BACKENDS = {"mach_continuous_time", "CLOCK_BOOTTIME"}


def artifact_paths(prefix):
    prefix = Path(prefix).resolve()
    return {key: Path(f"{prefix}.{suffix}") for key, suffix in (
        ("log", "log"), ("launch", "launch.json"), ("terminal", "terminal.json"))}


def publish(path, value):
    """Atomically publish complete JSON without replacing an existing path."""
    temporary = Path(f"{path}.tmp")
    owned = False
    try:
        with temporary.open("x") as stream:
            owned = True
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if owned:
            temporary.unlink()


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup(process):
    """Bounded failure cleanup that never needs the scientific clock."""
    record = {"signals": [], "errors": [], "reaped": process is None,
              "group_absent": process is None, "group_poll_attempts": 0}
    if process is None:
        return record

    def alive():
        try:
            return group_exists(process.pid)
        except Exception as error:  # noqa: BLE001 - cleanup must preserve all failures
            record["errors"].append(f"group check: {error!r}")
            return True

    def send(sig):
        try:
            os.killpg(process.pid, sig)
            record["signals"].append(signal.Signals(sig).name)
        except ProcessLookupError:
            pass
        except Exception as error:  # noqa: BLE001 - cleanup must preserve all failures
            record["errors"].append(f"signal {sig}: {error!r}")

    def reap(timeout):
        try:
            process.wait(timeout=timeout)
            record["reaped"] = True
        except subprocess.TimeoutExpired:
            pass
        except Exception as error:  # noqa: BLE001 - cleanup must preserve all failures
            record["errors"].append(f"reap: {error!r}")

    if alive():
        send(signal.SIGTERM)
        reap(TERM_WAIT_SECONDS)
    else:
        reap(0)
    if alive():
        send(signal.SIGKILL)
        reap(KILL_WAIT_SECONDS)
    for _ in range(GROUP_POLLS):
        record["group_poll_attempts"] += 1
        if not alive():
            record["group_absent"] = True
            break
        time.sleep(GROUP_POLL_SECONDS)
    return record


def validate_command(command, launch_path):
    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("a nonempty command is required")
    if command.count("--supervision") != 1 or any(x.startswith("--supervision=") for x in command):
        raise ValueError("command must contain exactly one --supervision PATH")
    position = command.index("--supervision")
    if position + 1 >= len(command) or Path(command[position + 1]).resolve() != launch_path:
        raise ValueError("command --supervision must name this supervisor's launch path")


def supervise(prefix, cap_seconds, command):
    """Run once. Tests replace dependencies; CLI always selects a native clock."""
    paths = artifact_paths(prefix)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    for path in [*paths.values(), Path(f"{paths['launch']}.tmp"), Path(f"{paths['terminal']}.tmp")]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    process = clock = deadline = None
    error = clock_error = None
    timed_out = False
    timing_available = False
    common = {"version": VERSION, "command": list(command), "cwd": str(Path.cwd()),
              "parent_pid": os.getpid(), "pid": None, "pgid": None,
              "cap_seconds": cap_seconds, "clock_backend": None,
              "started_ns": None, "deadline_ns": None, "started_unix": time.time()}
    # Exclusive log creation reserves this prefix before any process is started.
    with paths["log"].open("x") as log:
        try:
            if type(cap_seconds) is not int or cap_seconds <= 0:
                raise ValueError("cap_seconds must be a positive integer")
            validate_command(command, paths["launch"])
            clock = suspend_clock.SuspendClock()
            if clock.backend not in NATIVE_BACKENDS:
                raise suspend_clock.ClockError("a native suspend-inclusive backend is required")
            deadline = clock.deadline_after(cap_seconds)
            timing_available = True
            common.update(clock_backend=clock.backend, started_ns=deadline.started_ns,
                          deadline_ns=deadline.expires_ns,
                          watchdog_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          clock_source_sha256=hashlib.sha256(Path(suspend_clock.__file__).read_bytes()).hexdigest())
            deadline.check()
            process = subprocess.Popen(command, cwd=Path.cwd(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.update(pid=process.pid, pgid=process.pid)
            deadline.check()
            publish(paths["launch"], common)
            while True:
                deadline.check()
                if process.poll() is not None:
                    deadline.check()
                    break
                remaining = deadline.remaining_ns()
                if remaining <= 0:
                    raise TimeoutError("deadline expired before wait")
                try:
                    process.wait(timeout=min(POLL_SECONDS, remaining / 1_000_000_000))
                except subprocess.TimeoutExpired:
                    continue
                deadline.check()
                break
        except suspend_clock.ClockError as caught:
            clock_error = repr(caught)
            error = clock_error
            timing_available = False
        except TimeoutError as caught:
            timed_out = True
            error = repr(caught)
        except BaseException as caught:  # noqa: BLE001 - preserve interrupts and cleanup
            error = repr(caught)
        finally:
            cleaned = cleanup(process)
            finished_ns = elapsed_ns = wall_seconds = None
            if timing_available:
                try:
                    finished_ns = clock.now_ns()
                    elapsed_ns = finished_ns - deadline.started_ns
                    wall_seconds = elapsed_ns / 1_000_000_000
                    if finished_ns >= deadline.expires_ns:
                        timed_out = True
                        error = error or "deadline expired after cleanup"
                except suspend_clock.ClockError as caught:
                    clock_error = repr(caught)
                    error = error or clock_error
                    timing_available = False
                    finished_ns = elapsed_ns = wall_seconds = None
            terminal = {**common, "returncode": process.returncode if process else None,
                        "group_absent": cleaned["group_absent"], "timed_out": timed_out,
                        "error": error, "clock_error": clock_error, "timing_available": timing_available,
                        "finished_ns": finished_ns, "elapsed_ns": elapsed_ns, "wall_seconds": wall_seconds,
                        "finished_unix": time.time(), "cleanup": cleaned,
                        "timing_scope": "native elapsed from before Popen through child exit and group cleanup; "
                                        "terminal JSON publication follows the final reading"}
            successful = (timing_available and not timed_out and error is None
                          and terminal["returncode"] == 0 and cleaned["group_absent"]
                          and cleaned["reaped"] and not cleaned["errors"])
            terminal["status"] = "completed" if successful else "failed"
            publish(paths["terminal"], terminal)
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--cap-seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    def interrupted(signum, _frame):
        raise InterruptedError(f"supervisor received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        terminal = supervise(args.prefix, args.cap_seconds, command)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(terminal, sort_keys=True, allow_nan=False), flush=True)
    return 0 if terminal["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
