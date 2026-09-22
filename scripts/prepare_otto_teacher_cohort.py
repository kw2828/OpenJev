"""Freeze and assess a representative public-only learner-TRAIN cohort."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VERSION = "otto-teacher-cohort-v1"
HISTORY = "output/otto-teacher-label-pilot-v1/plan-01.json"
HISTORY_PIN = "6956fd2658fc8920c277a6a4585ecb75d0b1639730ead5158af0cc7835269380"
SYMM = "output/otto-symmetry-head-v1/run-01"
SOURCES = ("scripts/prepare_otto_teacher_cohort.py", "src/openjev/research/otto_teacher_cohort.py",
           "tests/test_otto_teacher_cohort.py", "tests/test_prepare_otto_teacher_cohort.py",
           "research/otto-teacher-cohort-protocol.md")
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
LIMITS = {"native_seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 1024**3}
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
QUALIFICATIONS = {
    "output/otto-teacher-cohort-v1/engineering-01/receipt.json": "tests/test_otto_teacher_cohort.py",
    "output/otto-teacher-cohort-v1/engineering-02/receipt.json": "tests/test_prepare_otto_teacher_cohort.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path, check=lambda: None):
    require(not any(p.is_symlink() for p in (path, *path.parents)), "nonsymlink evidence path")
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            check()
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def path(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "relative evidence name")
    return ROOT / p


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def write(target, value):
    with target.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def metadata(check):
    require(digest(path(HISTORY), check)["sha256"] == HISTORY_PIN, "frozen historical plan")
    history = json.loads(path(HISTORY).read_text())
    require(len(history["sources"]) == 138 and len(history["inputs"]) == 495, "complete historical closure")
    for name, expected in history["sources"].items():
        require(digest(path(name), check)["sha256"] == expected, f"frozen source: {name}")
    for name, expected in history["inputs"].items():
        require(digest(path(name), check) == expected, f"frozen input: {name}")
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    require(all(runtime[k] == history[k] for k in runtime), "unchanged qualified runtime")
    require(all(os.environ.get(k) == v for k, v in THREADS.items()), "one numerical thread")
    qualifications = {}
    for qualification_name, test in QUALIFICATIONS.items():
        qualification = json.loads(path(qualification_name).read_text())
        require(qualification["commands"][0]["returncode"] == 0
                and qualification["commands"][0]["command"] == [sys.executable, "-m", "pytest", "-q", test]
                and qualification["source_before"] == qualification["source_after"], "successful fabricated tests")
        for name, pin in qualification["source_after"].items():
            require(digest(path(name))["sha256"] == pin, "unchanged tested component")
        for name, desc in qualification["files"].items():
            require(Path(name).name == name and digest(path(qualification_name).parent / name) == desc,
                    "qualification log integrity")
        qualifications[qualification_name] = digest(path(qualification_name))
        if qualification_name.endswith("engineering-02/receipt.json"):
            require(qualification["status"] == "passed", "final lifecycle tests and lint passed")
    return {"version": VERSION, "history": {"path": HISTORY, **digest(path(HISTORY))},
            "qualifications": qualifications,
            "sources": {**history["sources"], **{p: digest(path(p))["sha256"] for p in SOURCES}},
            "inputs": history["inputs"], "limits": LIMITS, "environment": THREADS, **runtime}


class Budget:
    def __init__(self, out):
        require(digest(path(CLOCK))["sha256"] == CLOCK_PIN, "qualified clock identity")
        from openjev.research.suspend_clock import SuspendClock
        self.out, self.clock = out, SuspendClock()
        require(self.clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native deadline clock")
        self.start = self.clock.now_ns()
        self.deadline = self.start + LIMITS["native_seconds"] * 10**9
        self.bytes = self.events = 0
        self.pending, self.calls = {}, {}

    def check(self):
        require(self.clock.now_ns() < self.deadline, "suspend-inclusive deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(rss <= LIMITS["rss_bytes"] and self.bytes <= LIMITS["output_bytes"], "resource limit")

    def emit(self, channel, stream, event):
        self.check()
        raw = encoded(event)
        require(len(raw) <= 1024 and self.events < 631296, "event serialization allocation")
        require(self.bytes + len(raw) <= LIMITS["output_bytes"] - 64 * 1024**2, "journal allocation")
        key = (channel, event["operation_id"])
        count = self.calls.setdefault(event["operation"], {"attempted": 0, "returned": 0})
        if event["event"] == "attempt":
            require(key not in self.pending, "unique operation")
            self.pending[key] = event
            count["attempted"] += 1
        else:
            require(event["event"] == "return" and key in self.pending
                    and all(event[k] == v for k, v in self.pending[key].items() if k != "event"), "joined return")
        require(stream.write(raw) == len(raw), "complete journal append")
        os.fsync(stream.fileno())
        self.bytes += len(raw)
        self.events += 1
        if event["event"] == "return":
            count["returned"] += 1
            del self.pending[key]


def prepare(args):
    budget = Budget(args.output.parent)
    result = metadata(budget.check)
    from openjev.research import otto_teacher_anchors as original
    from openjev.research.otto_teacher_cohort import select_cohort

    def rows(stream):
        for i, line in enumerate(stream):
            budget.check()
            raw = original._decode_record(line)
            require(raw["row_index"] == str(i), "complete metadata row order")
            yield {**{k: original._numbers(v) for k, v in raw.items() if k != "teacher_costs"},
                   "teacher_costs": None}

    with path(f"{SYMM}/dagger-rows.jsonl").open("rb") as stream:
        result["cohort"] = select_cohort(rows(stream))
    result.update(status="frozen_before_reconstruction", started_ns=budget.start,
                  finished_ns=budget.clock.now_ns(), clock_backend=budget.clock.backend)
    write(args.output, result)
    budget.check()
    print(json.dumps({"status": result["status"], "plan": digest(args.output), "counts": result["cohort"]["counts"]}))


def run(args):
    args.output.mkdir(parents=False, exist_ok=False)
    budget = None
    receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
               "native_calls": 0, "source_draws": 0, "analytic_choices": 0,
               "learned_model_calls": 0, "optimizer_calls": 0}

    def interrupted(_signum, _frame):
        raise InterruptedError("supervisor terminated cohort preparation")

    def finalize():
        files = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
        receipt.update(started_ns=budget.start, finished_ns=budget.clock.now_ns(),
                       clock_backend=budget.clock.backend, events=budget.events, calls=budget.calls,
                       pending=list(budget.pending.values()),
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                       * (1 if sys.platform == "darwin" else 1024),
                       files=files,
                       timing_scope="Worker through payload hashing; original supervisor covers receipt publication and process exit.")
        receipt["wall_seconds"] = (receipt["finished_ns"] - budget.start) / 1e9
        budget.bytes = sum(p.stat().st_size for p in args.output.iterdir()) + len(encoded(receipt))
        budget.check()
        write(args.output / "receipt.json", receipt)
        budget.check()

    signal.signal(signal.SIGTERM, interrupted)
    try:
        budget = Budget(args.output)
        require(digest(args.plan, budget.check)["sha256"] == args.plan_sha256, "external plan pin")
        plan = json.loads(args.plan.read_text())
        expected = metadata(budget.check)
        require(plan["status"] == "frozen_before_reconstruction"
                and all(plan[k] == v for k, v in expected.items()), "unchanged preparation")
        while not args.supervision.exists():
            require(budget.clock.now_ns() - budget.start < 5 * 10**9, "original launch missing")
            time.sleep(.01)
        launch = json.loads(args.supervision.read_text())
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch["cwd"] == str(ROOT) == str(Path.cwd())
                and launch["pid"] == os.getpid() and launch["pgid"] == os.getpgrp()
                and launch["parent_pid"] == os.getppid() and launch["cap_seconds"] == 600,
                "original process identity")
        require(launch["clock_backend"] == budget.clock.backend
                and launch["clock_source_sha256"] == plan["sources"]["src/openjev/research/suspend_clock.py"]
                and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and launch["started_ns"] <= budget.start < launch["deadline_ns"]
                and launch["deadline_ns"] == launch["started_ns"] + 600 * 10**9, "original deadline")
        budget.deadline = launch["deadline_ns"]
        receipt.update(plan_sha256=args.plan_sha256, supervision_sha256=digest(args.supervision)["sha256"])
        write(args.output / "started.json", {"started_ns": budget.start, "launch": launch,
              "request": {k: str(v) for k, v in vars(args).items()}})
        import numpy as np

        from openjev.research import otto_teacher_cohort as helper
        kernels = {}
        for regime in ("lambda3", "lambda4"):
            with np.load(path(f"{SYMM}/kernel-{regime}.npz"), allow_pickle=False) as arrays:
                require(set(arrays.files) == {"likelihood", "initial_hit_weights"}, "public kernel schema")
                kernels[regime] = arrays["likelihood"].copy()
        with path(f"{SYMM}/collection-transitions.jsonl").open("rb") as stream, \
                (args.output / "reconstruction.jsonl").open("xb", buffering=0) as journal:
            def lines():
                for line in stream:
                    budget.check()
                    yield line
            records = helper.iter_public_records(lines(), plan["cohort"]["selections"])
            captured = helper.reconstruct_cohort(plan["cohort"]["selections"], records, kernels,
                check=budget.check, emit=lambda e: budget.emit("reconstruction", journal, e))
        with (args.output / "anchors.npz").open("xb") as stream:
            np.savez(stream, beliefs=np.stack([r["belief"] for r in captured["anchors"]]),
                     kernel_lambda3=kernels["lambda3"], kernel_lambda4=kernels["lambda4"])
            stream.flush()
            os.fsync(stream.fileno())
        with (args.output / "support.jsonl").open("xb", buffering=0) as journal:
            support = helper.assess_support(captured["anchors"], kernels, check=budget.check,
                emit=lambda e: budget.emit("support", journal, e))
        write(args.output / "support.json", support)
        require(not budget.pending and all(v["attempted"] == v["returned"] for v in budget.calls.values()),
                "every reconstruction and assessment returned")
        receipt.update(status="completed", reconstruction=captured["counts"], support=support["counts"],
                       eligible_for_sampling=support["counts"]["unsupported"] == 0,
                       requires_successful_original_supervisor=True)
        finalize()
        print(json.dumps({"status": receipt["status"], "support": receipt["support"],
                          "receipt": digest(args.output / "receipt.json")}))
    except BaseException as failure:
        receipt.update(status="failed", error=repr(failure), traceback=traceback.format_exc(),
                       eligible_for_sampling=False, calls=budget.calls if budget else {},
                       pending=list(budget.pending.values()) if budget else [])
        try:
            receipt["failure_elapsed_seconds"] = (budget.clock.now_ns() - budget.start) / 1e9 if budget else None
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            receipt["failure_clock_error"] = repr(secondary)
        try:
            target = args.output / "receipt.json"
            if target.exists():
                target.rename(args.output / "receipt.invalid.json")
            receipt["files"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
            write(target, receipt)
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            failure.add_note(f"Failure publication error: {secondary!r}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--supervision", type=Path)
    args = parser.parse_args()
    require(args.output.is_absolute(), "absolute output path")
    if args.mode == "plan":
        prepare(args)
    else:
        require(args.plan and args.plan.is_absolute() and args.plan_sha256
                and args.supervision and args.supervision.is_absolute(), "run inputs required")
        run(args)


if __name__ == "__main__":
    main()
