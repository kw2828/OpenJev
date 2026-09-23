"""Launch one pinned supervisor in a detached session, without claiming completion.

Use --prefix ABS --cap-seconds N -- COMMAND... . Prefixes must be canonical
paths under this checkout's output directory. COMMAND must explicitly pass
--supervision ABS.launch.json for that prefix. A reserved prefix is never reused,
even if launch acknowledgement is missing after an interruption. Inspect the
original supervisor launch/terminal files; this launcher never reconstructs them.

Detachment protects the supervisor from a signal to the originating process
group. It does not protect against host shutdown or a signal addressed directly
to the detached supervisor. The immutable supervisor remains the only writer of
the worker's original launch, terminal, deadline and reaping evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SUPERVISOR = ROOT / "scripts/supervise_dialogue_observation_v2.py"
CLOCK = ROOT / "src/openjev/research/suspend_clock.py"
SUPERVISOR_SHA256 = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
CLOCK_SHA256 = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
VERSION = "otto-detached-phase-launch-v1"
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_paths(prefix):
    return {name: Path(f"{prefix}.{suffix}") for name, suffix in (
        ("intent", "detached-intent.json"),
        ("acknowledgement", "detached-launch.json"),
        ("failure", "detached-failure.json"),
        ("supervisor_log", "supervisor.log"),
        ("worker_log", "log"),
        ("launch", "launch.json"),
        ("terminal", "terminal.json"),
    )}


def publish(path, value):
    """Publish atomically and exclusively, preserving any interrupted reservation."""
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


def native_clock_backend():
    # Load the exact pinned clock file, without importing model or data modules.
    name = "_openjev_detached_suspend_clock"
    spec = importlib.util.spec_from_file_location(name, CLOCK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    clock = module.SuspendClock()
    if clock.backend not in {"mach_continuous_time", "CLOCK_BOOTTIME"}:
        raise ValueError("a native suspend-inclusive clock is required")
    clock.now_ns()
    return clock.backend


def validate(prefix, cap_seconds, command):
    prefix = Path(prefix)
    if not prefix.is_absolute() or prefix != prefix.resolve():
        raise ValueError("prefix must be an absolute canonical path without symlinks")
    output = ROOT / "output"
    if not prefix.is_relative_to(output) or prefix == output:
        raise ValueError("prefix must be contained under this checkout's output directory")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("launch from the OpenJev checkout root")
    if type(cap_seconds) is not int or cap_seconds <= 0:
        raise ValueError("cap_seconds must be a positive integer")
    if not isinstance(command, (tuple, list)) or not command or not all(
        isinstance(item, str) and item and "\0" not in item for item in command
    ):
        raise ValueError("a nonempty structured command is required")
    executable = Path(command[0])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("command executable must be an absolute executable file")
    paths = artifact_paths(prefix)
    if command.count("--supervision") != 1 or any(
        item.startswith("--supervision=") for item in command
    ):
        raise ValueError("command must contain exactly one --supervision PATH")
    position = command.index("--supervision")
    if position + 1 >= len(command) or command[position + 1] != str(paths["launch"]):
        raise ValueError("--supervision must name the original canonical supervisor launch path")
    for path in (prefix, *paths.values(), *(Path(f"{p}.tmp") for p in paths.values())):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"reserved prefix cannot be retried or resumed: {path}")
    for path, expected in ((SUPERVISOR, SUPERVISOR_SHA256), (CLOCK, CLOCK_SHA256)):
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"immutable source pin differs: {path}")
    return paths, native_clock_backend()


def launch(prefix, cap_seconds, command):
    """Spawn exactly once; return launch provenance, never a completion verdict."""
    paths, backend = validate(prefix, cap_seconds, command)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    interpreter = str(Path(sys.executable).absolute())
    supervisor_command = [interpreter, str(SUPERVISOR), "--prefix", str(prefix),
                          "--cap-seconds", str(cap_seconds), "--", *command]
    environment = os.environ.copy()
    environment.update(THREAD_ENV, PYTHONDONTWRITEBYTECODE="1")
    intent = {
        "version": VERSION, "status": "launch_intent", "created_unix_ns": time.time_ns(),
        "launcher_pid": os.getpid(), "launcher_pgid": os.getpgrp(),
        "cwd": str(ROOT), "prefix": str(prefix), "cap_seconds": cap_seconds,
        "command": list(command), "supervisor_command": supervisor_command,
        "interpreter": interpreter, "launcher_sha256": digest(SELF),
        "watchdog_sha256": SUPERVISOR_SHA256, "clock_source_sha256": CLOCK_SHA256,
        "native_clock_preflight_backend": backend,
        "environment_overrides": {**THREAD_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        "original_launch": str(paths["launch"]), "original_terminal": str(paths["terminal"]),
        "retry_allowed": False, "resume_allowed": False,
        "scope": "Process launch only. Completion requires the original supervisor terminal.",
    }
    # The exclusive intent reserves this attempt before any child can be spawned.
    publish(paths["intent"], intent)
    process = None
    try:
        with paths["supervisor_log"].open("x") as log:
            process = subprocess.Popen(
                supervisor_command, cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        acknowledgement = {
            "version": VERSION, "status": "launched", "observed_unix_ns": time.time_ns(),
            "supervisor_pid": process.pid, "expected_supervisor_pgid": process.pid,
            "start_new_session": True, "supervisor_command": supervisor_command,
            "intent_sha256": digest(paths["intent"]),
            "launcher_sha256": intent["launcher_sha256"],
            "watchdog_sha256": SUPERVISOR_SHA256, "clock_source_sha256": CLOCK_SHA256,
            "original_launch": str(paths["launch"]), "original_terminal": str(paths["terminal"]),
            "worker_completion": "not_observed", "observation_not_completion": True,
        }
        publish(paths["acknowledgement"], acknowledgement)
    except BaseException as error:
        # Do not signal or restart a possibly running detached supervisor. If an
        # uncatchable interruption leaves only intent, this prefix stays spent.
        publish(paths["failure"], {
            "version": VERSION, "status": "launch_or_acknowledgement_failed",
            "observed_unix_ns": time.time_ns(), "error": repr(error),
            "supervisor_pid": process.pid if process is not None else None,
            "intent_sha256": digest(paths["intent"]),
            "original_launch": str(paths["launch"]), "original_terminal": str(paths["terminal"]),
            "worker_completion": "unknown", "retry_allowed": False,
        })
        raise
    return acknowledgement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--cap-seconds", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    acknowledgement = launch(args.prefix, args.cap_seconds, command)
    print(json.dumps(acknowledgement, sort_keys=True, allow_nan=False), flush=True)
    return 0  # Accepted launch only. No inference about worker outcome is made.


if __name__ == "__main__":
    raise SystemExit(main())
