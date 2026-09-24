"""Authenticate completed native collection under its original interpreter.

This separate metadata process never enters a collector Run/setup method. Its
consumer uses only standard-library metadata reads and opaque byte hashes. It
does not admit evaluation, confirmation, or a retry of the closed first screen.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/otto_residual_native_bridge.py"
OLD_RUNNER = "scripts/run_otto_residual.py"
COLLECTOR = "scripts/collect_otto_residual.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
INTERPRETER = ".venv-otto-released-native/bin/python"
VERSION = "otto-residual-native-bridge-v1"
CAP_SECONDS = 60
PINS = {
    OLD_RUNNER: "8b2936b77a6821ed795206ce366a0f97ac9fb1a45a815efa68117b826bc966e8",
    COLLECTOR: "62455365dd7a7f56662a2a2a02c10668ebe16bc4fbe173924a1af83a2a7f2815",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
    SUPERVISOR: "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
}
ROLES = ("collection_plan", "collection_receipt", "collection_terminal", "engineering")
CONSUMER_ROLES = {"bridge_receipt", "bridge_terminal", *ROLES[:3]}
PAYLOADS = {"started.json", "runtime.json"}
BLOCKED_IMPORTS = frozenset({"numpy", "torch", "tensorflow", "tf_keras", "scipy", "h5py",
                             "jax", "jaxlib", "keras", "mlx", "pandas", "safetensors", "onnxruntime"})
ZERO_COUNTS = dict.fromkeys(("array_decodes", "checkpoint_decodes", "model_calls", "teacher_calls",
                             "native_calls", "optimizer_steps", "confirmation_decodes", "old_test_decodes"), 0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular file")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def record(value):
    path = regular(value)
    return {"path": str(path), **descriptor(path)}


def check_record(row):
    require(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes"}
            and type(row["bytes"]) is int and row["bytes"] >= 0
            and descriptor(row["path"]) == {k: row[k] for k in ("sha256", "bytes")}, "exact descriptor")
    return regular(row["path"])


def same_record(a, b):
    return check_record(a) == check_record(b) and a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"]


def read(value):
    path = regular(value)
    require(path.suffix == ".json", "metadata JSON only")
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def exclusive_output(value):
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and path.parent.is_dir() and not path.exists() and not path.is_symlink()
            and not any(p.is_symlink() for p in path.parents), "exclusive contained output")
    return path


def check_sources(sources):
    require(isinstance(sources, dict) and sources, "nonempty source closure")
    for name, pin in sources.items():
        require(not Path(name).is_absolute() and descriptor(name)["sha256"] == pin,
                "unchanged source: " + name)


def bridge_sources():
    check_sources(PINS)
    return {**PINS, SELF: descriptor(SELF)["sha256"]}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, regular(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def metadata_imports_only():
    require(not any(name.split(".")[0] in BLOCKED_IMPORTS for name in sys.modules),
            "clean metadata-only process")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        require(name.split(".")[0] not in BLOCKED_IMPORTS, "forbidden numerical import: " + name)
        return original(name, *args, **kwargs)

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            require(fullname.split(".")[0] not in BLOCKED_IMPORTS,
                    "forbidden numerical import: " + fullname)

    blocker = Blocker()
    builtins.__import__ = guarded
    sys.meta_path.insert(0, blocker)
    try:
        yield
        require(not any(name.split(".")[0] in BLOCKED_IMPORTS for name in sys.modules),
                "metadata boundary remains clean")
    finally:
        builtins.__import__ = original
        sys.meta_path.remove(blocker)


def command_options(command, script, flags):
    command = list(command)
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / INTERPRETER), str(ROOT / script), "run"]
            and len(command) == 3 + 2 * len(flags), "exact native command")
    names, values = command[3::2], command[4::2]
    require(len(set(names)) == len(flags) and set(names) == flags, "exact unique command flags")
    return dict(zip(names, values, strict=True))


def launch_identity(launch, seconds):
    require(launch["version"] == "dialogue-observation-supervision-v2"
            and launch["cwd"] == str(ROOT) and launch["cap_seconds"] == seconds
            and launch["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
            and launch["clock_source_sha256"] == PINS[CLOCK]
            and launch["watchdog_sha256"] == PINS[SUPERVISOR]
            and type(launch["started_ns"]) is int and launch["started_ns"] >= 0
            and launch["deadline_ns"] == launch["started_ns"] + seconds * 10**9
            and all(type(launch[k]) is int and launch[k] > 0 for k in ("pid", "pgid", "parent_pid"))
            and launch["pid"] == launch["pgid"] != launch["parent_pid"], "original native launch identity")


def successful_parent(terminal, launch, receipt, seconds):
    launch_identity(launch, seconds)
    require(all(terminal.get(k) == v for k, v in launch.items())
            and terminal["status"] == "completed" and type(terminal["returncode"]) is int
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["timing_available"] is True
            and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == []
            and launch["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= terminal["finished_ns"] < launch["deadline_ns"], "successful original parent closure")


def closed_payloads(directory, receipt, expected):
    require(set(receipt["files"]) == set(expected)
            and {p.name for p in directory.iterdir()} == set(expected) | {"receipt.json"}, "closed payload roster")
    result = {}
    for name, pin in receipt["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == pin, "opaque payload pin")
        result[name] = record(directory / name)
    return result


def collection_closure(inputs, sources):
    """Check the saved native evidence, without importing native authentication."""
    require(set(inputs) == set(ROLES), "original collection evidence roles")
    for row in inputs.values():
        check_record(row)
    plan = read(inputs["collection_plan"]["path"])
    receipt = read(inputs["collection_receipt"]["path"])
    require(plan["version"] == receipt["version"] == "otto-residual-collection-v1"
            and plan["status"] == "frozen_before_collection" and plan["phase"] == "dev"
            and receipt["phase"] == "dev" and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["plan_sha256"] == inputs["collection_plan"]["sha256"]
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"]
            and receipt["native_inputs"] == plan["native_inputs"] and receipt["limits"] == plan["limits"]
            and all(sources.get(k) == v for k, v in plan["sources"].items())
            and same_record(plan["inputs"]["engineering"], inputs["engineering"])
            and receipt["completed_episodes"] == receipt["dev_episodes"] == 18
            and receipt["confirm_episodes"] == 0 and receipt["master_episodes"] == 54
            and receipt["training_updates"] == receipt["old_test_array_decodes"] == 0
            and receipt["pending"] == [] and all(receipt[k] is None for k in
                ("pending_episode", "pending_action", "pending_emission"))
            and receipt["requires_successful_original_supervisor"] is True, "completed DEV-only collection")
    for row in plan["inputs"].values():
        check_record(row)
    for row in plan["native_inputs"].values():
        check_record(row)
    engineering = read(inputs["engineering"]["path"])
    require(engineering["status"] == "passed" and engineering["source_before"] == engineering["source_after"]
            and engineering["sources_unchanged"] is True and engineering["commands"]
            and all(row["returncode"] == 0 and row["timed_out"] is False and row["reaped"] is True
                    and row["group_absent"] is True for row in engineering["commands"]), "original engineering closure")
    require(sources == {name: pin["sha256"] for name, pin in engineering["source_after"].items()},
            "same old qualified source roster")
    for name, pin in engineering["source_after"].items():
        require(descriptor(name) == pin, "original engineering source bytes")
    for name, pin in engineering["files"].items():
        require(Path(name).name == name and descriptor(regular(inputs["engineering"]["path"]).parent / name) == pin,
                "original engineering opaque payload")
    directory = regular(inputs["collection_receipt"]["path"]).parent
    payloads = closed_payloads(directory, receipt, plan["payloads"])
    started = read(directory / "started.json")
    launch = started["launch"]
    terminal = read(inputs["collection_terminal"]["path"])
    successful_parent(terminal, launch, receipt, 3600)
    options = command_options(launch["command"], COLLECTOR, {"--plan", "--plan-sha256", "--supervision", "--output"})
    require(regular(options["--plan"]) == regular(inputs["collection_plan"]["path"])
            and options["--plan-sha256"] == inputs["collection_plan"]["sha256"]
            and options["--output"] == str(directory)
            and descriptor(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and read(options["--supervision"]) == launch and started["started_ns"] == receipt["started_ns"]
            and started["request"] == {"mode": "run", **{key[2:].replace("-", "_"): value
                                                           for key, value in options.items()}}, "original collection joins")
    return plan, receipt, directory, payloads


def bridge_flags():
    return {"--supervision", "--output"} | {"--" + role.replace("_", "-") + suffix
                                             for role in ROLES for suffix in ("", "-sha256")}


def verify_bridge(inputs):
    """Verify five pinned inputs under the evaluator runtime; no module loads."""
    require(isinstance(inputs, dict) and set(inputs) == CONSUMER_ROLES, "exact consumer bridge roles")
    for row in inputs.values():
        check_record(row)
    receipt = read(inputs["bridge_receipt"]["path"])
    require(receipt["version"] == VERSION and receipt["status"] == "completed" and receipt["complete"] is True
            and receipt["pending"] is None and receipt["counts"] == ZERO_COUNTS
            and all(type(v) is int for v in receipt["counts"].values())
            and receipt["requires_successful_original_supervisor"] is True
            and receipt["bridge_sources"] == bridge_sources(), "complete source-bound metadata bridge")
    check_sources(receipt["sources"])
    for role in ROLES[:3]:
        require(same_record(inputs[role], receipt["inputs"][role]), "same original collection input")
    plan, collection, directory, payloads = collection_closure(receipt["inputs"], receipt["sources"])
    require(receipt["native_runtime"] == plan["runtime"] and receipt["native_inputs"] == plan["native_inputs"]
            and receipt["collection_directory"] == str(directory) and receipt["collection_payloads"] == payloads,
            "same opaque native collection export")
    out = regular(inputs["bridge_receipt"]["path"]).parent
    closed_payloads(out, receipt, PAYLOADS)
    started, runtime = read(out / "started.json"), read(out / "runtime.json")
    terminal = read(inputs["bridge_terminal"]["path"])
    launch = started["launch"]
    successful_parent(terminal, launch, receipt, CAP_SECONDS)
    require(read(receipt["inputs"]["collection_terminal"]["path"])["finished_ns"] <= launch["started_ns"],
            "bridge starts after original collection closure")
    options = command_options(launch["command"], SELF, bridge_flags())
    for role in ROLES:
        flag = "--" + role.replace("_", "-")
        require(regular(options[flag]) == check_record(receipt["inputs"][role])
                and options[flag + "-sha256"] == receipt["inputs"][role]["sha256"], "bridge command input joins")
    require(options["--output"] == str(out)
            and descriptor(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and read(options["--supervision"]) == launch and started["started_ns"] == receipt["started_ns"]
            and started["inputs"] == receipt["inputs"] and runtime == receipt["native_runtime"]
            and runtime["python_executable"] == str(ROOT / INTERPRETER), "original bridge launch/runtime joins")
    return plan, collection, directory


def execute(args):
    out = exclusive_output(args.output)
    out.mkdir()
    clock = None
    receipt = {"version": VERSION, "status": "failed", "complete": False, "pending": "admission",
               "counts": dict(ZERO_COUNTS), "files": {}, "requires_successful_original_supervisor": True}
    try:
        require(Path(sys.executable).absolute() == ROOT / INTERPRETER and Path.cwd() == ROOT,
                "exact original native interpreter and cwd")
        pins = bridge_sources()
        with metadata_imports_only():
            clock = load(CLOCK, "_residual_bridge_clock").SuspendClock()
            start = clock.now_ns()
            receipt["started_ns"] = start
            require(args.supervision.is_absolute(), "absolute original supervision path")
            while not args.supervision.exists():
                require(clock.now_ns() - start < 5 * 10**9, "original supervisor launch missing")
                time.sleep(.01)
            launch = read(args.supervision)
            launch_identity(launch, CAP_SECONDS)
            actual_command = list(launch["command"])
            if actual_command[1:2] == ["-u"]:
                actual_command.pop(1)
            require(actual_command == [sys.executable, *sys.argv]
                    and launch["pid"] == os.getpid() and launch["pgid"] == os.getpgrp()
                    and launch["parent_pid"] == os.getppid() and launch["clock_backend"] == clock.backend
                    and launch["started_ns"] <= start < launch["deadline_ns"], "genuine current worker identity")
            options = command_options(launch["command"], SELF, bridge_flags())
            inputs = {}
            for role in ROLES:
                row = record(getattr(args, role))
                flag = "--" + role.replace("_", "-")
                require(row["sha256"] == getattr(args, role + "_sha256") == options[flag + "-sha256"]
                        and regular(options[flag]) == regular(row["path"]), "external native evidence pin")
                inputs[role] = row
            require(options["--supervision"] == str(args.supervision) and options["--output"] == str(out),
                    "actual worker paths")
            receipt.update(inputs=inputs, bridge_sources=pins, supervision_sha256=descriptor(args.supervision)["sha256"])
            write(out / "started.json", {"launch": launch, "inputs": inputs, "started_ns": start})
            receipt["pending"] = "native_metadata_authentication"
            old = load(OLD_RUNNER, "_residual_bridge_immutable_runner")
            sources = old.engineering(inputs["engineering"]["path"])
            plan, collected, directory = old.authenticate_collection(inputs, sources)
            require(clock.now_ns() < launch["deadline_ns"], "fixed bridge deadline")
            require(read(inputs["collection_terminal"]["path"])["finished_ns"] <= launch["started_ns"],
                    "bridge follows original collection closure")
            check_sources(sources)
            verified_plan, verified_receipt, verified_directory, payloads = collection_closure(inputs, sources)
            require((verified_plan, verified_receipt, verified_directory) == (plan, collected, directory)
                    and pins == bridge_sources(), "unchanged bridge and collection evidence")
            write(out / "runtime.json", plan["runtime"])
            receipt.update(sources=sources, native_runtime=plan["runtime"], native_inputs=plan["native_inputs"],
                           collection_directory=str(directory), collection_payloads=payloads)
            require(clock.now_ns() < launch["deadline_ns"], "fixed bridge deadline after opaque verification")
        receipt.update(status="completed", complete=True, pending=None)
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        if clock is not None:
            try:
                receipt["finished_ns"] = clock.now_ns()
            except BaseException as error:  # noqa: BLE001 - clock failure cannot produce a successful receipt
                receipt.update(status="failed", complete=False, clock_error=repr(error))
        receipt["files"] = {path.name: descriptor(path) for path in out.iterdir() if path.is_file()}
        write(out / "receipt.json", receipt)
    require(receipt["complete"], "bridge did not complete")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run",))
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        parser.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, _frame):
        raise InterruptedError(f"metadata bridge received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        result = execute(args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
