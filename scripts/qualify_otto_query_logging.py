"""One bounded, fabricated comparison of old and new durable event emission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-logging-qualification-v1"
LIMITS = {"seconds": 120, "rss_bytes": 1024**3, "output_bytes": 64 * 1024**2}
BASELINE = "scripts/study_otto_query_advantage.py"
BASELINE_PIN = "ab67e579a7db5147e8cc61cc6e467a85da65edc8c1fc9d121c0180393a1ca6ff"
CANDIDATE = "scripts/study_otto_query_advantage_v2.py"
JOURNAL = "src/openjev/research/durable_jsonl.py"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SOURCES = (BASELINE, CANDIDATE, JOURNAL, "tests/test_durable_jsonl.py",
           "tests/test_study_otto_query_advantage_v2.py", "scripts/qualify_otto_query_logging.py",
           "research/otto-query-logging-qualification-protocol.md", "src/openjev/research/suspend_clock.py",
           "scripts/supervise_dialogue_observation_v2.py")
N, BLOCKS = 4096, 6


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trace():
    operations = ("movement", "hit_draw", "teacher_choose", "teacher_update")
    return [{"anchor_id": i // 256, "seed": 19300001, "replicate_id": (i // 8) % 16,
             "operation_id": i // 2 + 1, "operation": operations[(i // 2) % 4],
             "event": "attempt" if i % 2 == 0 else "return",
             "first_action": (i // 128) % 4, "step": (i // 8) % 32,
             "public": {"position": [20 + i % 9, 20 + (i // 9) % 9],
                        "hit": i % 4, "done": False, "valid_actions": [0, 1, 2, 3]},
             "fixture_only": True} for i in range(N)]


class Qualification:
    def __init__(self, args):
        self.args, self.out = args, args.output
        require(digest(ROOT / CLOCK)["sha256"] == CLOCK_PIN, "clock source before import")
        self.clock = load(CLOCK, "_logging_clock").SuspendClock()
        self.start = self.clock.now_ns()
        self.results, self.pending = [], None
        self.receipt = {"version": VERSION, "status": "started", "admitted": False,
                        "limits": LIMITS, "model_calls": 0, "native_calls": 0,
                        "sampler_calls": 0, "optimizer_calls": 0, "outcome_reads": 0}

    def check(self):
        require(self.clock.now_ns() - self.start < LIMITS["seconds"] * 10**9, "qualification deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "qualification RSS cap")
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
                <= LIMITS["output_bytes"] - 1024**2, "qualification output with failure reserve")

    def arm(self, name, block, modules, events):
        self.check()
        directory = self.out / f"block-{block:02d}-{name}"
        directory.mkdir()
        module = modules[name]
        files = sorted(n for n in module.PAYLOADS if n.endswith(".jsonl"))
        files += ["started.json", "runtime.json", "setup.json", "anchors.npz"]
        require(len(files) == 14, "fixed directory population")
        for filename in files:
            (directory / filename).touch(exist_ok=False)
        initialization_tick = time.perf_counter()
        run = module.Run(types.SimpleNamespace(output=directory))
        run.clock = self.clock
        run.launch = {"deadline_ns": self.start + LIMITS["seconds"] * 10**9}
        run.ledger.context = {"phase": "fabricated_qualification"}
        if name == "new":
            run.journal = modules["journal"].DurableJSONL(directory,
                total_limit=module.LIMITS["output_bytes"],
                non_sampler_limit=module.LIMITS["non_sampler_output_bytes"],
                failure_reserve=module.FAILURE_RESERVE,
                per_file_record_limits={"sampler-events.jsonl": 1024})
        initialization_seconds = time.perf_counter() - initialization_tick
        original_sync, syncs, acknowledged = os.fsync, 0, 0

        def sync(fd):
            nonlocal syncs
            original_sync(fd)
            syncs += 1

        self.pending = {"block": block, "implementation": name, "acknowledged": 0}
        tick = time.perf_counter()
        try:
            os.fsync = sync
            for event in events:
                if name == "old":
                    require(len((json.dumps(event, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False) + "\n").encode()) <= 1024,
                            "old sampler record size check")
                run.ledger.emit("sampler-events.jsonl", event)
                acknowledged += 1
                self.pending["acknowledged"] = acknowledged
            append_seconds, append_syncs = time.perf_counter() - tick, syncs
            close_tick = time.perf_counter()
            if name == "new":
                run.journal.close()
                require(not run.journal.pending and not run.journal.errors and not run.journal.poisoned,
                        "candidate closed without uncertainty")
            else:
                run.check()
            close_seconds = time.perf_counter() - close_tick
            require(acknowledged == append_syncs == N and not run.ledger.pending_publications,
                    "every append durably acknowledged exactly once")
        finally:
            os.fsync = original_sync
            if name == "new" and not run.journal.closed:
                primary = sys.exception()
                try:
                    run.journal.close(suppress=True)
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(f"Journal cleanup: {cleanup_error!r}")
                self.pending["journal"] = {"pending": run.journal.pending,
                    "errors": run.journal.errors, "poisoned": run.journal.poisoned}
        result = {"block": block, "implementation": name, "acknowledged": acknowledged,
                  "append_fsyncs": append_syncs, "total_fsyncs": syncs,
                  "initialization_seconds": initialization_seconds,
                  "append_seconds": append_seconds, "close_seconds": close_seconds,
                  "total_seconds": initialization_seconds + append_seconds + close_seconds,
                  "journal": digest(directory / "sampler-events.jsonl")}
        require(all(math.isfinite(result[k]) and result[k] >= 0 for k in
                    ("append_seconds", "close_seconds", "total_seconds")), "finite actual timing")
        self.results.append(result)
        write(directory / "receipt.json", result)
        self.pending = None
        self.check()

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("qualification deadline")

        previous = signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
        try:
            while not self.args.supervision.exists():
                require(self.clock.now_ns() - self.start < 5 * 10**9, "original logger launch available")
                time.sleep(.01)
            launch = json.loads(self.args.supervision.read_text())
            require(launch["command"] == [sys.executable, *sys.argv]
                    and launch["pid"] == os.getpid() and launch["pgid"] == os.getpgrp()
                    and launch["parent_pid"] == os.getppid() and launch["cap_seconds"] == 120
                    and launch["clock_backend"] == self.clock.backend
                    and launch["started_ns"] <= self.start < launch["deadline_ns"]
                    and launch["deadline_ns"] == launch["started_ns"] + 120 * 10**9
                    and Path(launch["cwd"]) == ROOT == Path.cwd(), "original bounded logger process")
            plan = json.loads(self.args.plan.read_text())
            require(digest(self.args.plan)["sha256"] == self.args.plan_sha256, "external plan pin")
            sources = {name: digest(ROOT / name) for name in SOURCES}
            require(plan["version"] == VERSION and plan["status"] == "frozen_before_timing"
                    and plan["sources"] == sources and plan["limits"] == LIMITS
                    and plan["events_per_arm"] == N and plan["paired_blocks"] == BLOCKS,
                    "exact prospective workload and source closure")
            require(sources[BASELINE]["sha256"] == BASELINE_PIN, "original baseline implementation")
            require(Path(sys.executable).absolute() == ROOT / ".venv-otto-released-native/bin/python",
                    "unchanged original native runtime")
            require(launch["clock_source_sha256"] == sources["src/openjev/research/suspend_clock.py"]["sha256"]
                    and launch["watchdog_sha256"] == sources["scripts/supervise_dialogue_observation_v2.py"]["sha256"],
                    "qualified timing supervisor")
            self.receipt.update(plan_sha256=self.args.plan_sha256, sources=sources,
                supervision_sha256=digest(self.args.supervision)["sha256"], launch=launch,
                started_ns=self.start)
            write(self.out / "started.json", self.receipt)
            modules = {"old": load(BASELINE, "_logging_old"),
                       "new": load(CANDIDATE, "_logging_new"),
                       "journal": load(JOURNAL, "_logging_journal")}
            events = trace()
            expected = b"".join((json.dumps(e, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                 + "\n").encode() for e in events)
            wanted = {"bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()}
            for block in range(BLOCKS):
                for name in (("old", "new") if block % 2 == 0 else ("new", "old")):
                    self.arm(name, block, modules, events)
            ratios = []
            for block in range(BLOCKS):
                arms = {r["implementation"]: r for r in self.results if r["block"] == block}
                require(arms["old"]["journal"] == arms["new"]["journal"] == wanted, "byte-identical complete traces")
                ratios.append(arms["old"]["total_seconds"] / arms["new"]["total_seconds"])
            admitted = statistics.median(ratios) >= 2 and sum(r > 1 for r in ratios) >= 5
            require(sources == {name: digest(ROOT / name) for name in SOURCES}, "unchanged measured sources")
            self.receipt.update(status="completed", admitted=admitted, results=self.results,
                paired_speedups=ratios, median_speedup=statistics.median(ratios),
                faster_blocks=sum(r > 1 for r in ratios), expected_journal=wanted,
                scope="Fabricated event emission only, not a scientific completion or architecture result.")
            self.check()
            self.receipt["files"] = {str(p.relative_to(self.out)): digest(p)
                                     for p in self.out.rglob("*") if p.is_file()}
            self.check()
            finished = self.clock.now_ns()
            require(finished < launch["deadline_ns"], "closure inside original logger deadline")
            self.receipt.update(finished_ns=finished, wall_seconds=(finished-self.start)/1e9)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": self.receipt["status"], "admitted": self.receipt["admitted"],
                              "receipt": digest(self.out / "receipt.json")}))
        except BaseException as error:
            self.receipt.update(status="failed", admitted=False, error=repr(error),
                                traceback=traceback.format_exc(), results=self.results, pending=self.pending)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {str(p.relative_to(self.out)): digest(p)
                                         for p in self.out.rglob("*") if p.is_file()}
                finished = self.clock.now_ns()
                self.receipt.update(finished_ns=finished, wall_seconds=(finished-self.start)/1e9)
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication_error:  # noqa: BLE001 - keep primary failure and partial files
                error.add_note(f"Failure publication: {publication_error!r}")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    args = parser.parse_args()
    require(args.plan.is_absolute() and args.output.is_absolute() and args.supervision.is_absolute(),
            "absolute qualification paths")
    Qualification(args).execute()


if __name__ == "__main__":
    main()
