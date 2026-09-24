"""Real-process, metadata-only qualification of the native-to-Torch handoff.

This invokes no numerical worker. A successful consumer output is a preflight
plan only, not execution authorization. All environments are inherited unchanged.
Zero scientific-call counts are source-based assertions plus import guards, not
an operating-system sandbox claim. Native clock reads are supervision, not OTTO
simulator calls. Every child has an original suspend-inclusive 60-second cap.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/qualify_otto_residual_handoff.py"
VERSION = "otto-residual-handoff-qualification-v1"
BRIDGE = "scripts/otto_residual_native_bridge.py"
CONSUMER = "scripts/run_otto_residual_reanalysis.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK = "src/openjev/research/suspend_clock.py"
CAP_SECONDS = 60
NATIVE = ROOT / ".venv-otto-released-native/bin/python"
NUMERICAL = ROOT / ".venv/bin/python"
FIXED = {
    "collection_plan": (
        "output/otto-residual-estimator-v1/dev-collection-plan-01.json",
        "a9b6e35095e86e45d9beab43e573d33754c1dc85e7480da2bfaa7624626f142d",
    ),
    "collection_receipt": (
        "output/otto-residual-estimator-v1/dev-collection-01/receipt.json",
        "245c635dac36cad4b4e3d0b074e623c2bb7e01f12adff5031e93a56d5a73baba",
    ),
    "collection_terminal": (
        "output/otto-residual-estimator-v1/dev-collection-native-01.terminal.json",
        "54f99ef42df4dd8412a893e8997f95de9c55da32815a59a47db4bdf5cf1d82f1",
    ),
    "engineering": (
        "output/otto-residual-runner-engineering-v1/attempt-02/receipt.json",
        "fbe77c205e9717c6ff7531befeeccd2d8f076c6b1b3a3fb1b8c2a25c228872f0",
    ),
}
FIXED_SOURCES = {
    SUPERVISOR: "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144",
    CLOCK: "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124",
}
BLOCKED = frozenset({"numpy", "torch", "tensorflow", "tf_keras", "scipy", "h5py",
                     "jax", "jaxlib", "keras", "mlx", "pandas", "safetensors", "onnxruntime"})
ZERO_COUNTS = dict.fromkeys(("array_decodes", "checkpoint_decodes", "model_calls", "teacher_calls",
                            "native_calls", "optimizer_steps", "confirmation_decodes", "old_test_decodes"), 0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contained(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained nonsymlink path")
    return path


def descriptor(value):
    path = contained(value)
    require(path.is_file(), "regular file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    path = contained(value)
    require(path.suffix == ".json" and path.is_file(), "metadata JSON only")
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def record(value):
    return {"path": str(contained(value)), **descriptor(value)}


@contextmanager
def metadata_only():
    def clean():
        require(not any(name.split(".")[0] in BLOCKED for name in sys.modules),
                "no numerical imports in handoff harness")

    clean()
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        require(name.split(".")[0] not in BLOCKED, "forbidden numerical import: " + name)
        return original(name, *args, **kwargs)

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            require(fullname.split(".")[0] not in BLOCKED, "forbidden numerical import: " + fullname)

    blocker = Blocker()
    builtins.__import__ = guarded
    sys.meta_path.insert(0, blocker)
    try:
        yield
        clean()
    finally:
        builtins.__import__ = original
        sys.meta_path.remove(blocker)


def load_supervisor():
    spec = importlib.util.spec_from_file_location("_residual_handoff_supervisor", ROOT / SUPERVISOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_inputs():
    rows = {}
    for role, (path, pin) in FIXED.items():
        rows[role] = record(path)
        require(rows[role]["sha256"] == pin, "immutable original input: " + role)
    return rows


def sources(engineering):
    old = read(FIXED["engineering"][0])
    new = read(engineering)
    for value in (old, new):
        require(value["status"] == "passed" and value["sources_unchanged"] is True
                and value["source_before"] == value["source_after"], "passed source-bound engineering")
    expected = dict(new["source_after"])
    require({SELF, BRIDGE, CONSUMER, SUPERVISOR, CLOCK, "src/openjev/__init__.py",
             "src/openjev/research/__init__.py"} <= set(expected), "complete handoff source binding")
    require(all(expected.get(name) == pin for name, pin in old["source_after"].items()),
            "all original qualified sources remain immutable")
    actual = {name: descriptor(name) for name in sorted(expected)}
    require(actual == expected, "current new engineering source identity")
    require(all(actual[name]["sha256"] == pin for name, pin in FIXED_SOURCES.items()),
            "original supervisor and native clock")
    return actual


def flags(inputs):
    result = []
    for role, row in inputs.items():
        flag = "--" + role.replace("_", "-")
        result.extend((flag, row["path"], flag + "-sha256", row["sha256"]))
    return result


def plan_process(supervisor, prefix, command):
    """The plan CLI has no supervision argument; apply the same clock/cleanup."""
    child = clock = deadline = None
    error = None
    timed_out = False
    common = {"command": command, "cwd": str(ROOT), "cap_seconds": CAP_SECONDS,
              "parent_pid": os.getpid(), "pid": None, "pgid": None,
              "clock_backend": None, "started_ns": None, "deadline_ns": None}
    with Path(str(prefix) + ".log").open("x") as log:
        try:
            clock = supervisor.suspend_clock.SuspendClock()
            require(clock.backend in supervisor.NATIVE_BACKENDS, "native suspend-inclusive process clock")
            deadline = clock.deadline_after(CAP_SECONDS)
            common.update(clock_backend=clock.backend, started_ns=deadline.started_ns,
                          deadline_ns=deadline.expires_ns)
            deadline.check()
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.update(pid=child.pid, pgid=child.pid)
            write(Path(str(prefix) + ".launch.json"), common)
            while child.poll() is None:
                deadline.check()
                try:
                    child.wait(timeout=min(.25, deadline.remaining_ns() / 1e9))
                except subprocess.TimeoutExpired:
                    pass
            deadline.check()
        except TimeoutError as caught:
            timed_out, error = True, repr(caught)
        except BaseException as caught:  # noqa: BLE001 - preserve failure and always reap
            error = repr(caught)
        finally:
            cleaned = supervisor.cleanup(child)
            finished = None
            try:
                if clock is not None and deadline is not None:
                    finished = clock.now_ns()
                    if finished >= deadline.expires_ns:
                        timed_out, error = True, error or "deadline expired through cleanup"
            except BaseException as caught:  # noqa: BLE001 - clock failure is terminal
                error = error or repr(caught)
            result = {**common, "returncode": child.returncode if child else None,
                      "timed_out": timed_out, "error": error, "finished_ns": finished,
                      "wall_seconds": ((finished - deadline.started_ns) / 1e9
                                       if finished is not None else None),
                      "cleanup": cleaned, "group_absent": cleaned["group_absent"],
                      "timing_available": finished is not None}
            write(Path(str(prefix) + ".terminal.json"), result)
    return result


def qualified_result(directory, command, terminal, expected_error):
    cleanup = terminal["cleanup"]
    clean = (terminal["timing_available"] is True and terminal["timed_out"] is False
             and terminal["error"] is None and terminal["group_absent"] is True
             and cleanup["reaped"] is True and cleanup["group_absent"] is True
             and cleanup["errors"] == [] and terminal["finished_ns"] < terminal["deadline_ns"])
    code = terminal["returncode"]
    rejected = (type(code) is int and code > 0
                and expected_error in (directory / "process.log").read_text()) if expected_error else False
    passed = clean and (rejected if expected_error else code == 0)
    result = {"command": command, "expected": "rejection" if expected_error else "success",
              "expected_error": expected_error, "passed": passed, "process": terminal,
              "files": {str(path.relative_to(directory)): descriptor(path)
                        for path in sorted(directory.rglob("*")) if path.is_file()}}
    write(directory / "outcome.json", result)
    return result


def run_bridge(supervisor, out, name, inputs, interpreter=NATIVE, expected_error=None):
    directory = out / name
    directory.mkdir()
    prefix = directory / "process"
    command = [str(interpreter), str(ROOT / BRIDGE), "run", *flags(inputs),
               "--supervision", str(prefix) + ".launch.json", "--output", str(directory / "bridge")]
    write(directory / "command.json", {"command": command, "supervisor_source": record(SUPERVISOR)})
    terminal = supervisor.supervise(prefix, CAP_SECONDS, command)
    return qualified_result(directory, command, terminal, expected_error)


def run_plan(supervisor, out, name, inputs, expected_error=None):
    directory = out / name
    directory.mkdir()
    command = [str(NUMERICAL), str(ROOT / CONSUMER), "plan", "--phase", "evaluate",
               *flags(inputs), "--output", str(directory / "preflight-plan.json")]
    write(directory / "command.json", {"command": command, "numerical_worker": False})
    terminal = plan_process(supervisor, directory / "process", command)
    result = qualified_result(directory, command, terminal, expected_error)
    if result["passed"]:
        require((directory / "preflight-plan.json").exists() is (expected_error is None),
                "plan emitted only for successful admission")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering", type=Path, required=True)
    parser.add_argument("--engineering-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(Path.cwd() == ROOT and args.engineering.is_absolute() and args.output.is_absolute(),
            "repository cwd and absolute inputs/output required")
    out = contained(args.output)
    require(out.parent.is_dir() and not out.exists(), "exclusive output with existing parent")
    out.mkdir()
    receipt = {"version": VERSION, "status": "failed", "real_runtime_handoff_passed": False,
               "counts": dict(ZERO_COUNTS), "counts_basis": "reviewed metadata-only sources and import guards; no OS sandbox",
               "numerical_worker_started": False, "admits_execution": False,
               "scientific_performance_claim": False, "environment_overrides": {}, "cases": {},
               "source_before": {}, "source_after": {}, "sources_unchanged": False}

    def interrupted(signum, _frame):
        raise InterruptedError(f"handoff harness received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with metadata_only():
            engineering = record(args.engineering)
            require(engineering["sha256"] == args.engineering_sha256, "external new engineering pin")
            original = fixed_inputs()
            before = sources(args.engineering)
            receipt.update(engineering=engineering, original_inputs=original, source_before=before)
            write(out / "started.json", {**receipt, "python": sys.version, "executable": sys.executable,
                                       "per_child_cap_seconds": CAP_SECONDS, "number_of_cases": 5})
            supervisor = load_supervisor()

            def keep(name, result):
                receipt["cases"][name] = result
                require(sources(args.engineering) == before and fixed_inputs() == original
                        and record(args.engineering) == engineering, "unchanged evidence after case")
                require(result["passed"], "handoff case failed: " + name)

            keep("positive_native_bridge", run_bridge(supervisor, out, "positive_native_bridge", original))
            positive = out / "positive_native_bridge"
            consumer = {role: original[role] for role in ("collection_plan", "collection_receipt", "collection_terminal")}
            consumer.update(engineering=engineering, bridge_receipt=record(positive / "bridge/receipt.json"),
                            bridge_terminal=record(positive / "process.terminal.json"))
            keep("positive_consumer_plan", run_plan(supervisor, out, "positive_consumer_plan", consumer))
            keep("negative_wrong_interpreter", run_bridge(
                supervisor, out, "negative_wrong_interpreter", original, interpreter=NUMERICAL,
                expected_error="exact original native interpreter and cwd"))
            changed = {role: dict(row) for role, row in original.items()}
            changed["collection_plan"]["sha256"] = "0" * 64
            keep("negative_changed_plan_pin", run_bridge(
                supervisor, out, "negative_changed_plan_pin", changed,
                expected_error="external native evidence pin"))
            copied = out / "incomplete_bridge"
            copied.mkdir()
            for name in ("started.json", "runtime.json"):
                with (copied / name).open("xb") as stream:
                    stream.write((positive / "bridge" / name).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
            incomplete = read(positive / "bridge/receipt.json")
            incomplete["complete"] = False
            write(copied / "receipt.json", incomplete)
            bad_consumer = dict(consumer, bridge_receipt=record(copied / "receipt.json"))
            keep("negative_incomplete_bridge", run_plan(
                supervisor, out, "negative_incomplete_bridge", bad_consumer,
                expected_error="complete source-bound metadata bridge"))
            receipt.update(status="passed", real_runtime_handoff_passed=True)
    except BaseException as error:  # noqa: BLE001 - preserve partial qualification outcomes
        receipt.update(status="failed", real_runtime_handoff_passed=False, error=repr(error))
    finally:
        signal.signal(signal.SIGTERM, previous)
        try:
            receipt["source_after"] = {name: descriptor(name) for name in receipt["source_before"]}
            receipt["sources_unchanged"] = bool(receipt["source_before"]) and receipt["source_before"] == receipt["source_after"]
            require(receipt["sources_unchanged"], "source closure unchanged through qualification")
            if "original_inputs" in receipt:
                require(fixed_inputs() == receipt["original_inputs"] and record(args.engineering) == receipt["engineering"],
                        "unchanged final external evidence")
        except BaseException as error:  # noqa: BLE001 - failed final pins cannot pass
            receipt.update(status="failed", real_runtime_handoff_passed=False, final_error=repr(error))
        receipt["files"] = {str(path.relative_to(out)): descriptor(path)
                            for path in sorted(out.rglob("*")) if path.is_file()}
        write(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": record(out / "receipt.json")}), flush=True)
    return 0 if receipt["real_runtime_handoff_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
