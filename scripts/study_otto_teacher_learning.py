"""Frozen matched teacher-cost collection, ordinary-head fitting and fresh search.

All scientific components are authenticated before import. Buffered sampler
events become durable only at a completed gzip-panel boundary. An interrupted
panel fails the run; there is no resume, replacement, horizon extension or retry.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-teacher-learning-v1"
BASE = "output/otto-teacher-cohort-v1"
SYMM = "output/otto-symmetry-head-v1/run-01"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
PINS = {f"{BASE}/plan-01.json": "dcae4eeed5e72c1352f5e32bba359c601dd2442cf7f62586774ca74499b38c94",
        f"{BASE}/run-01/receipt.json": "da32d79052c0bdbd665bb409c543cfef7b2cec278b15eda080f40b1730093121",
        f"{BASE}/supervision-01.terminal.json": "14409a676ab65d7751248c088dacd0bf22c03498847de1ec1a7d514e998bff9a",
        f"{BASE}/audit-01/receipt.json": "694b4f92d69df8f7dc256dfb0b142227324bda0c3514c8eae499c6c5ea6cfccf"}
NEW = {"scripts/study_otto_teacher_learning.py", "src/openjev/research/otto_teacher_learning.py",
       "tests/test_otto_teacher_learning.py", "src/openjev/research/otto_cost_regression.py",
       "tests/test_otto_cost_regression.py", "scripts/audit_otto_teacher_cohort.py",
       "tests/test_study_otto_teacher_learning.py", "research/otto-teacher-learning-protocol.md"}
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
SEEDS, KINDS = (10101, 10102, 10103), ("analytic", "continuation")
ARMS = tuple(f"{k}@{s}" for s in SEEDS for k in KINDS) + ("analytic_inbounds",)
FIRST = {"lambda3": 1020001, "lambda4": 1030001, "lambda5": 1040001}
HORIZON, CASES, LABEL_SEED = 2188, 24, 19000002
LIMITS = {"native_seconds": 7200, "rss_bytes": 4 * 1024**3, "output_bytes": 16 * 1024**3}
UPSTREAM = "tmp/otto-source-review-01/isotropic/classes/sourcetracking.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def path(name):
    p = Path(name)
    require(not p.is_absolute() and ".." not in p.parts, "relative manifest path")
    return ROOT / p


def digest(p, check=lambda: None):
    require(p.is_file() and not any(v.is_symlink() for v in (p, *p.parents)), "regular evidence file")
    h, n = hashlib.sha256(), 0
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            check()
            h.update(block)
            n += len(block)
    return {"sha256": h.hexdigest(), "bytes": n}


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    with p.open("xb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def configuration():
    return {"anchors": 558, "label_seed": LABEL_SEED, "replicate_ids": list(range(16)),
            "horizon": HORIZON, "fit_seeds": list(SEEDS), "target_arms": list(KINDS),
            "epochs": 80, "batch_size": 128, "learning_rate": .0003, "gradient_clip": 5.,
            "shuffle_seed_offset": 20000, "evaluation_first_seeds": FIRST, "cases_per_regime": CASES,
            "evaluation_arms": list(ARMS), "evaluation_episodes": 504, "native_resets": 504,
            "native_step_cap": 504 * HORIZON, "learned_forward_cap": 432 * HORIZON,
            "optimizer_updates": 2400, "head_allocation_episodes": 72, "module_allocation_episodes": 432,
            "Ngrid": 53, "Nhits": 4, "R_dt": 2., "event_bytes": 512, "gzip_level": 1,
            "timing": "Complete actor costs exclude measured journal I/O; cold inference is included.",
            "checkpoint": "fixed final epoch80; no validation selection or inference ensemble"}


def parent_closure(check):
    for name, pin in PINS.items():
        require(digest(path(name), check)["sha256"] == pin, "closed cohort external pin")
    prior = read(path(f"{BASE}/plan-01.json"))
    worker = read(path(f"{BASE}/run-01/receipt.json"))
    audit = read(path(f"{BASE}/audit-01/receipt.json"))
    terminal = read(path(f"{BASE}/supervision-01.terminal.json"))
    require(worker["status"] == audit["status"] == terminal["status"] == "completed"
            and audit["agreement"] is True and worker["eligible_for_sampling"] is True
            and worker["support"] == {"anchors": 558, "supported": 558, "unsupported": 0,
                 "snapshot_attempts": 558, "snapshot_returns": 558}
            and worker["plan_sha256"] == audit["plan_sha256"] == PINS[f"{BASE}/plan-01.json"]
            and audit["worker_sha256"] == PINS[f"{BASE}/run-01/receipt.json"]
            and audit["terminal_sha256"] == PINS[f"{BASE}/supervision-01.terminal.json"]
            and terminal["returncode"] == 0 and terminal["group_absent"] is True
            and terminal["timed_out"] is False and terminal["error"] is None, "supported audited cohort")
    require(digest(path("scripts/audit_otto_teacher_cohort.py"), check) == audit["source"], "prior audit source identity")
    require(len(prior["sources"]) == 143 and len(prior["inputs"]) == 495, "complete inherited closure")
    inputs = dict(prior["inputs"])
    inputs.update({name: digest(path(name), check) for name in PINS})
    for relative, receipt in ((f"{BASE}/run-01", worker), (f"{BASE}/audit-01", audit)):
        require({p.name for p in path(relative).iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "closed cohort directory membership")
        inputs.update({f"{relative}/{name}": desc for name, desc in receipt["files"].items()})
    for name, desc in prior["qualifications"].items():
        inputs[name] = desc
        receipt = read(path(name))
        inputs.update({str(Path(name).parent / file): value for file, value in receipt["files"].items()})
    launch_name = f"{BASE}/supervision-01.launch.json"
    require(digest(path(launch_name), check)["sha256"] == worker["supervision_sha256"], "prior launch pin")
    launch = read(path(launch_name))
    require(all(terminal[k] == v for k, v in launch.items()), "prior parent identity join")
    inputs[launch_name] = digest(path(launch_name), check)
    inputs[prior["history"]["path"]] = {k: prior["history"][k] for k in ("sha256", "bytes")}
    for name, desc in inputs.items():
        require(digest(path(name), check) == desc, f"complete input hash: {name}")
    for name, pin in prior["sources"].items():
        require(digest(path(name), check)["sha256"] == pin, f"inherited source: {name}")
    rows = prior["cohort"]["selections"]
    require(len(rows) == 558 and [r["anchor_id"] for r in rows] == list(range(558)), "all canonical anchors")
    continuations = 16 * sum(len(r["public"]["valid_actions"]) for r in rows)
    steps = continuations * HORIZON
    bounds = {"continuations": continuations, "continuation_steps": steps,
              "sampler_events": 67 * len(rows) + 4 * continuations + 8 * steps}
    require(continuations == 32304 and steps == 70681152 and bounds["sampler_events"] == 565615818,
            "frozen geometry-derived label bounds")
    bounds.update(uncompressed_event_bytes=512 * bounds["sampler_events"],
                  compression_guaranteed=False, compressed_output_hard_cap=LIMITS["output_bytes"])
    return prior, inputs, bounds


class ClockBudget:
    def __init__(self, output):
        require(digest(path(CLOCK))["sha256"] == CLOCK_PIN, "clock pin")
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.suspend_clock import SuspendClock
        self.clock, self.output = SuspendClock(), output
        require(self.clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME"), "native clock")
        self.start = self.clock.now_ns()
        self.deadline = self.start + LIMITS["native_seconds"] * 10**9
        self.last_size_check = 0

    def check(self, *, force=False):
        now = self.clock.now_ns()
        require(now < self.deadline, "original suspend-inclusive deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(rss <= LIMITS["rss_bytes"], "RSS cap")
        if self.output.is_dir() and (force or now - self.last_size_check >= 100_000_000):
            require(sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) <= LIMITS["output_bytes"],
                    "hard compressed/output byte cap")
            self.last_size_check = now


def make_plan(args):
    budget = ClockBudget(args.output.parent)
    require(all(os.environ.get(k) == v for k, v in THREADS.items()), "single-thread environment")
    prior, inputs, bounds = parent_closure(budget.check)
    sources = dict(prior["sources"])
    for name in NEW:
        pin = digest(path(name), budget.check)["sha256"]
        require(name not in sources or sources[name] == pin, "no inherited source replacement")
        sources[name] = pin
    required_engineering = {ROOT / "output/otto-cost-regression-component-v1/engineering-01",
                            ROOT / "output/otto-teacher-learning-v1/engineering-02",
                            ROOT / "output/otto-teacher-learning-v1/engineering-03"}
    require(set(args.engineering) == required_engineering and len(args.engineering) == 3,
            "exact loss/helper/runner qualifications")
    covered = set()
    for directory in args.engineering:
        require(directory.is_absolute() and directory.is_dir(), "explicit engineering directory")
        receipt = read(directory / "receipt.json")
        require(receipt["status"] == "passed" and receipt["source_before"] == receipt["source_after"],
                "qualified unchanged new sources required")
        for name, pin in receipt["source_after"].items():
            require(sources[name] == pin, "engineering source bound to current planned bytes")
            covered.add(name)
        require(receipt["commands"] and all(type(c["returncode"]) is int and c["returncode"] == 0
                for c in receipt["commands"]), "all final qualification commands passed")
        require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "complete flat engineering attempt")
        for name, desc in receipt["files"].items():
            require(Path(name).name == name and digest(directory / name, budget.check) == desc,
                    "captured engineering log descriptor")
        for p in directory.iterdir():
            require(p.is_file(), "flat engineering evidence")
            inputs[str(p.relative_to(ROOT))] = digest(p, budget.check)
    require(NEW - {"scripts/audit_otto_teacher_cohort.py", "research/otto-teacher-learning-protocol.md"} <= covered,
            "every new numerical/runner source and test was qualified")
    failed = path("output/otto-teacher-learning-v1/engineering-01")
    original_failure = read(failed / "receipt.json")
    require(original_failure["status"] == "failed" and {p.name for p in failed.iterdir()}
            == set(original_failure["files"]) | {"receipt.json"}, "preserved first helper lint failure")
    for name, desc in original_failure["files"].items():
        require(Path(name).name == name and digest(failed / name, budget.check) == desc, "failed-attempt original log")
    inputs.update({str(p.relative_to(ROOT)): digest(p, budget.check) for p in failed.iterdir()})
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    require(all(runtime[k] == prior[k] for k in runtime), "unchanged qualified runtime")
    plan = {"version": VERSION, "configuration": configuration(), "limits": LIMITS, "bounds": bounds,
            "sources": sources, "inputs": inputs, "environment": THREADS, **runtime,
            "selections": prior["cohort"]["selections"], "status": "frozen_before_label_generation"}
    write(args.output, plan)
    budget.check()
    print(json.dumps({"status": plan["status"], "plan": digest(args.output)}))


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.calls, self.pending, self.streams = {}, {}, {}
        self.io_seconds = 0.
        self.context, self.pending_panel, self.pending_emission, self.pending_episode = {}, None, None, None
        self.sampler_calls, self.sampler_pending, self.sampler_events = {}, {}, 0
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
                        "completed_panels": 0, "completed_fits": 0, "completed_episodes": 0}

    def check(self):
        self.budget.check()

    def emit(self, name, value, *, durable=False):
        tick = time.perf_counter()
        if name not in self.streams:
            self.streams[name] = (self.out / name).open("xb", buffering=0)
        stream = self.streams[name]
        raw = encode(value)
        require(stream.write(raw) == len(raw), "complete journal write")
        if durable:
            os.fsync(stream.fileno())
        self.io_seconds += time.perf_counter() - tick

    def call(self, operation, function, *, context=None, checked=False):
        if not checked:
            self.check()
        counts = self.calls.setdefault(operation, {"attempted": 0, "returned": 0, "seconds": 0.})
        caps = {"native_reset": 504, "actor_initialization": 504, "native_step": 504 * HORIZON,
                "public_update": 504 * HORIZON, "head_predict": 432 * HORIZON, "analytic_choose": 72 * HORIZON}
        require(operation not in caps or counts["attempted"] < caps[operation], "pre-call operation cap")
        identity = {"operation": operation, "index": counts["attempted"], **(context or self.context)}
        key = (operation, identity["index"])
        self.pending[key] = identity
        counts["attempted"] += 1
        self.emit("work.jsonl", {"event": "attempt", **identity})
        tick = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - tick
        self.emit("work.jsonl", {"event": "return", **identity, "seconds": elapsed})
        counts["returned"] += 1
        counts["seconds"] += elapsed
        del self.pending[key]
        return result, elapsed

    def array_file(self, name, arrays):
        with (self.out / name).open("xb") as stream:
            self.np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": name, **digest(self.out / name, self.check)}

    def checkpoint(self, arm, seed, phase, exported):
        name = f"{arm}-{seed}-{phase}.npz"
        desc = self.array_file(name, exported)
        with self.np.load(self.out / name, allow_pickle=False) as saved:
            require(set(saved.files) == set(exported), "saved checkpoint key identity")
            for key, expected in exported.items():
                actual = saved[key]
                require((actual.dtype == expected.dtype and actual.shape == expected.shape
                         and actual.tobytes() == expected.tobytes()) if isinstance(expected, self.np.ndarray)
                        else actual.ndim == 0 and actual.item() == expected, "saved checkpoint exact export identity")
        return desc

    def bind(self):
        require(digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256, "external plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["configuration"] == configuration()
                and self.plan["limits"] == LIMITS and self.plan["environment"] == THREADS,
                "fixed study configuration")
        prior, inherited, bounds = parent_closure(self.check)
        require(all(self.plan["inputs"][k] == v for k, v in inherited.items())
                and self.plan["selections"] == prior["cohort"]["selections"] and self.plan["bounds"] == bounds,
                "prior cohort and bound joins")
        require(set(self.plan["sources"]) == set(prior["sources"]) | NEW, "exact scientific source set")
        for name, pin in self.plan["sources"].items():
            require(digest(path(name), self.check)["sha256"] == pin, "frozen source")
        for name, desc in self.plan["inputs"].items():
            require(digest(path(name), self.check) == desc, "frozen input")
        runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
                   "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
        require(all(self.plan[k] == v for k, v in runtime.items())
                and all(os.environ.get(k) == v for k, v in THREADS.items()), "runtime and CPU threads")
        launch = read(self.args.supervision)
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and launch["pid"] == os.getpid()
                and launch["pgid"] == os.getpgrp() and launch["parent_pid"] == os.getppid()
                and launch["cwd"] == str(ROOT) == str(Path.cwd()) and launch["cap_seconds"] == 7200
                and launch["clock_backend"] == self.budget.clock.backend
                and launch["clock_source_sha256"] == CLOCK_PIN
                and launch["watchdog_sha256"] == self.plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and launch["deadline_ns"] == launch["started_ns"] + 7200 * 10**9
                and launch["started_ns"] <= self.budget.start < launch["deadline_ns"], "original supervision")
        self.budget.deadline = launch["deadline_ns"]
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=digest(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()}, "launch": launch})

    def setup(self):
        tick = time.perf_counter()
        import numpy as np
        self.np = np
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.otto_teacher_costs import summarize_costs
        from openjev.research.otto_teacher_rollouts import sample_teacher_panel
        self.sample, self.reduce = sample_teacher_panel, summarize_costs
        with np.load(path(f"{BASE}/run-01/anchors.npz"), allow_pickle=False) as archive:
            require(set(archive.files) == {"beliefs", "kernel_lambda3", "kernel_lambda4"}, "captured cohort schema")
            self.beliefs = archive["beliefs"]
            require(self.beliefs.dtype == np.float64 and self.beliefs.shape == (558, 53, 53), "all captured anchors")
        self.beliefs.setflags(write=False)
        self.kernels, self.mixtures = {}, {}
        for regime in FIRST:
            with np.load(path(f"{SYMM}/kernel-{regime}.npz"), allow_pickle=False) as archive:
                require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "only public kernel arrays")
                kernel, weights = archive["likelihood"], archive["initial_hit_weights"]
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64
                    and weights.shape == (4,) and weights[0] == 0 and (weights[1:] > 0).all(), "public kernel/mixture")
            kernel.setflags(write=False)
            self.kernels[regime] = kernel
            self.mixtures[regime] = weights.tolist()
        self.rows = self.plan["selections"]
        for row, belief in zip(self.rows, self.beliefs, strict=True):
            require(row["posterior"] == {"sha256": hashlib.sha256(belief.tobytes()).hexdigest(),
                    "mass": float(belief.sum())}, "exact frozen anchor witness")
        self.setup_seconds = time.perf_counter() - tick

    def collect(self):
        np = self.np
        means = np.full((558, 4), np.inf, dtype=np.float64)
        tick = time.perf_counter()
        total_records = total_steps = total_found = 0
        for row, belief in zip(self.rows, self.beliefs, strict=True):
            anchor = row["anchor_id"]
            self.context = {"phase": "sampling", "anchor_id": anchor}
            self.pending_panel = self.context.copy()
            self.emit("panels.jsonl", {"event": "attempt", **self.context}, durable=True)
            panel_tick = time.perf_counter()
            self.sampler_pending = {}
            filename = f"panel-{anchor:03d}.jsonl.gz"
            with (self.out / filename).open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as compressed:
                    def emit(event, anchor=anchor, compressed=compressed):
                        payload = encode(event)
                        self.pending_emission = {"anchor_id": anchor, "event": event["event"],
                                                 "operation_id": event.get("operation_id"), "bytes": len(payload)}
                        require(len(payload) <= 512 and self.sampler_events < self.plan["bounds"]["sampler_events"],
                                "fixed sampler event byte/count cap")
                        if event["event"] in ("attempt", "return"):
                            key = event["operation_id"]
                            count = self.sampler_calls.setdefault(event["operation"], {"attempted": 0, "returned": 0})
                            if event["event"] == "attempt":
                                require(key not in self.sampler_pending, "unique sampler attempt")
                                self.sampler_pending[key] = event
                                count["attempted"] += 1
                            else:
                                require(key in self.sampler_pending and all(event[k] == v for k, v in
                                        self.sampler_pending[key].items() if k != "event"), "paired sampler return")
                        require(compressed.write(payload) == len(payload), "complete gzip event write")
                        if event["event"] == "return":
                            count["returned"] += 1
                            del self.sampler_pending[key]
                        self.sampler_events += 1
                        self.pending_emission = None
                    records = self.sample(row["public"], belief, self.kernels[row["regime"]], seed=LABEL_SEED,
                        anchor_id=anchor, replicate_ids=range(16), horizon=HORIZON, check=self.check, emit=emit)
                raw.flush()
                os.fsync(raw.fileno())
            require(not self.sampler_pending, "complete sampler panel")
            reduced = self.reduce(records, eligible_actions=row["public"]["valid_actions"],
                                   replicate_ids=range(16), horizon=HORIZON)
            for action in reduced["actions"]:
                means[anchor, action["action"]] = action["mean_cost"]
            panel = {"event": "return", **self.context, "seconds": time.perf_counter() - panel_tick,
                     "file": {"path": filename, **digest(self.out / filename, self.check)},
                     "records": [{"replicate_id": r.replicate_id, "first_action": r.first_action,
                                  "steps": r.steps, "found": r.found} for r in records], "summary": reduced}
            self.emit("panels.jsonl", panel, durable=True)
            self.pending_panel = None
            self.receipt["completed_panels"] += 1
            total_records += len(records)
            total_steps += sum(r.steps for r in records)
            total_found += sum(r.found for r in records)
            self.budget.check(force=True)
            if anchor % 24 == 0:
                print(json.dumps({"phase": "sampling", "panels": anchor + 1, "moves": total_steps}), flush=True)
        require(total_records == self.plan["bounds"]["continuations"], "every prescribed continuation")
        expected_calls = {"anchor_snapshot": 558, "source_generator": 558 * 16, "source_draw": 558 * 16,
                          "teacher_snapshot": total_records, "hit_generator": total_records,
                          "teacher_choose": total_steps - total_records, "movement": total_steps,
                          "hit_draw": total_steps - total_found, "teacher_update": total_steps}
        require(self.sampler_calls == {k: {"attempted": v, "returned": v} for k, v in expected_calls.items()}
                and self.sampler_events == 67 * 558 + 4 * total_records + 8 * total_steps - 2 * total_found,
                "complete sampler event/count identities")
        self.collection_seconds = time.perf_counter() - tick
        write(self.out / "collection.json", {"panels": 558, "records": total_records, "steps": total_steps,
              "found": total_found, "censored": total_records - total_found, "sampler_events": self.sampler_events,
              "calls": self.sampler_calls, "seconds": self.collection_seconds,
              "timing_scope": "All panel sampling, reduction, serialization, gzip, hashes and fsyncs."})
        return means

    def training_data(self, continuation):
        np = self.np
        tick = time.perf_counter()
        from openjev.research import otto_symmetry_head as model
        self.model = model
        self.model_module_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_cost_regression import targets
        selected = {r["row_index"]: r for r in self.rows}
        analytic = np.full((558, 4), np.inf, dtype=np.float64)
        mask = np.zeros((558, 4), dtype=np.bool_)
        found = set()
        # Parse unselected numerical values lexically. Only the frozen selected
        # teacher scores become labels; no historical EVAL rows are decoded.
        with path(f"{SYMM}/dagger-rows.jsonl").open("rb") as stream:
            for index, line in enumerate(stream):
                self.check()
                lexical = json.loads(line, parse_int=str, parse_float=str)
                require(lexical["row_index"] == str(index), "original retained row order")
                if index not in selected:
                    continue
                original = json.loads(line)
                row = selected[index]
                require({k: v for k, v in original.items() if k != "teacher_costs"}
                        == {k: v for k, v in row.items() if k != "anchor_id"}, "exact selected analytic label join")
                costs = original["teacher_costs"]
                require(len(costs) == 4, "four recorded analytic costs")
                for action in range(4):
                    allowed = action in row["public"]["valid_actions"]
                    require((type(costs[action]) in (int, float) and math.isfinite(costs[action]))
                            if allowed else costs[action] is None, "recorded eligible/blocked analytic scores")
                    mask[row["anchor_id"], action] = allowed
                    if allowed:
                        analytic[row["anchor_id"], action] = costs[action]
                found.add(index)
        require(index + 1 == 4596 and found == set(selected), "all selected historical labels")
        maps = {regime: model.PublicFeatureMap(self.kernels[regime], float(regime[-1])) for regime in FIRST}
        features = np.stack([maps[r["regime"]].features(b, r["public"])
                             for r, b in zip(self.rows, self.beliefs, strict=True)])
        bundle = targets(analytic, continuation, mask, [r["episode_id"] for r in self.rows])
        arrays = {"features": features, "analytic_costs": analytic, "continuation_means": continuation,
                  "allowed": mask, "anchor_ids": np.arange(558, dtype=np.int64)}
        for kind in KINDS:
            arrays.update({f"{kind}_{k}": v for k, v in bundle[kind].items() if isinstance(v, np.ndarray)})
        data_file = self.array_file("training-data.npz", arrays)
        self.preparation_seconds = time.perf_counter() - tick
        write(self.out / "training-data.json", {"file": data_file, "rows": self.rows,
              "scales": {k: bundle[k]["scale"] for k in KINDS}, "episodes": 144,
              "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
              "seconds": self.preparation_seconds, "model_module_seconds": self.model_module_seconds,
              "scope": "Same selected TRAIN rows and weights; historical analytic label generation is reused work."})
        return features, bundle

    def json_value(self, value):
        if isinstance(value, self.np.ndarray):
            return value.tolist()
        if isinstance(value, self.np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: self.json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.json_value(v) for v in value]
        return value

    def train(self, features, bundle):
        tick = time.perf_counter()
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from openjev.research.otto_teacher_learning import train_pair
        self.training_setup_seconds = time.perf_counter() - tick
        self.training_pending, self.training_calls, self.fit_costs = {}, {}, {}
        fits, starts, operation_starts = [], {}, {}

        def emit(event, payload):
            value = self.json_value(payload)
            seed, arm = payload.get("seed"), payload.get("arm")
            if event == "targets":
                value = {"seed": seed, "arm": arm, "scale": bundle[arm]["scale"],
                         "arrays": {k: hashlib.sha256(v.tobytes()).hexdigest()
                                    for k, v in bundle[arm].items() if isinstance(v, self.np.ndarray)}}
            if event == "attempt":
                key = seed, payload["operation_id"]
                require(key not in self.training_pending, "unique training attempt")
                self.training_pending[key] = value
                count = self.training_calls.setdefault(payload["channel"], {"attempted": 0, "returned": 0})
                count["attempted"] += 1
            elif event == "return":
                value["operation_seconds"] = time.perf_counter() - operation_starts[(seed, payload["operation_id"])]
            self.emit("training.jsonl", {"event": event, **value}, durable=True)
            if event == "attempt":
                operation_starts[key] = time.perf_counter()
            elif event == "return":
                key = seed, payload["operation_id"]
                require(key in self.training_pending and all(payload[k] == v for k, v in
                        self.training_pending[key].items()), "paired training return")
                self.training_calls[payload["channel"]]["returned"] += 1
                del self.training_pending[key]
                operation_starts.pop(key)
            elif event == "fit_start":
                starts[(seed, arm)] = time.perf_counter()
            elif event == "fit_end":
                self.fit_costs[f"{arm}@{seed}"] = {"fit_seconds": time.perf_counter() - starts[(seed, arm)]}
                self.receipt["completed_fits"] += 1

        self.pair_costs = []
        for seed in SEEDS:
            self.context = {"phase": "fitting", "seed": seed}
            tick = time.perf_counter()
            result = train_pair(features, bundle, seed, check=self.check, emit=emit, checkpoint=self.checkpoint)
            pair_seconds = time.perf_counter() - tick
            fit_sum = math.fsum(self.fit_costs[f"{kind}@{seed}"]["fit_seconds"] for kind in KINDS)
            overhead = pair_seconds - fit_sum
            require(overhead >= 0 and result["progress"]["completed_updates"] == 800
                    and result["progress"]["pending"] == {}, "complete paired training")
            for kind in KINDS:
                self.fit_costs[f"{kind}@{seed}"].update(pair_setup_allocation_seconds=overhead / 2,
                    total_training_seconds=self.fit_costs[f"{kind}@{seed}"]["fit_seconds"] + overhead / 2,
                    label_allocation_seconds=self.collection_seconds / 3 if kind == "continuation" else 0.)
            self.pair_costs.append({"seed": seed, "seconds": pair_seconds, "nonfit_seconds": overhead})
            for fit in result["fits"]:
                clean = {k: v for k, v in fit.items() if k != "final_export"}
                clean["costs"] = self.fit_costs[f'{fit["arm"]}@{seed}']
                self.emit("fits.jsonl", self.json_value(clean), durable=True)
                fits.append(clean)
            print(json.dumps({"phase": "fitting", "seed": seed, "completed_fits": len(fits)}), flush=True)
        require(not self.training_pending and self.training_calls["optimizer_update"] == {"attempted": 2400, "returned": 2400},
                "all actual matched optimizer operations")
        write(self.out / "training-summary.json", {"calls": self.training_calls, "pairs": self.pair_costs,
              "fits": self.fit_costs, "torch_setup_seconds": self.training_setup_seconds,
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "initializations": 6, "final_exports": 6, "parity_numpy_calls": 6, "parity_torch_calls": 6,
              "historical_analytic_label_cost": "Reused authenticated work, not measured again or claimed zero."})
        return fits

    def evaluation_setup(self, fits):
        tick = time.perf_counter()
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        spec = importlib.util.spec_from_file_location("_teacher_learning_native", path(UPSTREAM))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.native_class = module.SourceTracking
        self.native_setup_seconds = time.perf_counter() - tick
        self.heads, self.head_setup = {}, {}
        for fit in fits:
            name = f'{fit["arm"]}@{fit["seed"]}'
            checkpoint = self.out / fit["final_checkpoint"]["path"]
            self.check()
            tick = time.perf_counter()
            require(digest(checkpoint) == {k: fit["final_checkpoint"][k] for k in ("sha256", "bytes")},
                    "final evaluation checkpoint pin")
            with self.np.load(checkpoint, allow_pickle=False) as archive:
                exported = {key: archive[key] for key in archive.files}
            for key in ("version", "kind", "input_dim"):
                exported[key] = exported[key].item()
            self.heads[name] = self.model.FrozenHead(exported)
            self.head_setup[name] = {"seconds": time.perf_counter() - tick,
                                     "checkpoint": fit["final_checkpoint"], "storage": self.heads[name].storage_bytes()}
        write(self.out / "deployment.json", {"heads": self.head_setup, "native_setup_seconds": self.native_setup_seconds,
              "model_module_seconds": self.model_module_seconds, "head_allocation_episodes": 72,
              "module_allocation_episodes": 432,
              "scope": "Six independent file loads; no inference warmup. Public policy modules may already be imported during sampling."})

    def witness(self, actor, env, packet):
        belief = actor.belief
        require(actor.public == packet and belief.dtype == self.np.float64 and belief.shape == (53, 53)
                and self.np.isfinite(belief).all() and belief.tobytes() == env.p_source.tobytes(),
                "exact public/native posterior and packet")
        return {"sha256": hashlib.sha256(belief.tobytes()).hexdigest(), "mass": float(belief.sum())}

    def episode(self, regime, seed, case, arm, compressed):
        np = self.np
        episode_id = f"eval:{regime}:{seed}:{arm}"
        self.context = {"phase": "evaluation", "episode_id": episode_id, "step": 0}
        self.pending_episode = episode_id
        identity = {"episode_id": episode_id, "regime": regime, "seed": seed, "case": case,
                    "initial_hit": 1 + case % 3, "block": case // 3, "arm": arm}
        config = {"Ndim": 2, "lambda_over_dx": float(regime[-1]), "R_dt": 2., "Ngrid": 53,
                  "Nhits": 4, "draw_source": True, "norm_Poisson": "Euclidean"}
        env, reset_seconds = self.call("native_reset", lambda: self.seeded(self.native_class, seed, config,
                                                                       initial_hit=identity["initial_hit"]))
        require(np.array_equal(env.p_Poisson, self.kernels[regime]) and env.N == 53 and env.Nhits == 4,
                "every native reset matches authenticated known kernel")
        packet = dict(self.observation(env, 0)._asdict())
        head = self.heads.get(arm)
        # The same qualified public actor supplies exact filtering to every arm.
        # Learned arms never invoke its analytic scorer; its initialization/storage
        # cost is retained rather than silently omitted.
        def initialize():
            return (self.actor_class(packet, self.kernels[regime], allow_stay=False),
                    self.model.PublicFeatureMap(self.kernels[regime], float(regime[-1])) if head else None)
        (actor, feature_map), init_seconds = self.call("actor_initialization", initialize)
        state = self.witness(actor, env, packet)

        def save(value):
            tick = time.perf_counter()
            raw = encode(value)
            require(compressed.write(raw) == len(raw), "complete evaluation event write")
            self.io_seconds += time.perf_counter() - tick

        save({"kind": "reset", **identity, "public": packet, "posterior_after": state,
              "source_evaluation_only": env.source.tolist()})
        chooses = updates = environment = raw_chooses = excluded = 0.
        for step in range(1, HORIZON + 1):
            self.context["step"] = step
            self.check()
            tick, old_io = time.perf_counter(), self.io_seconds
            if head is None:
                (action, scores), _ = self.call("analytic_choose", actor.choose, checked=True)
            else:
                feature = feature_map.features(actor.belief, packet)
                scores, _ = self.call("head_predict", lambda feature=feature: head.scores(feature), checked=True)
                allowed = packet["valid_actions"]
                minimum = min(float(scores[a]) for a in allowed)
                action = next(a for a in allowed if abs(float(scores[a]) - minimum) < 1e-10)
            raw = time.perf_counter() - tick
            removed = self.io_seconds - old_io
            choice = raw - removed
            require(choice >= 0, "nonnegative measured choice excluding only I/O")
            native, elapsed = self.call("native_step", lambda action=action: env.step(action, quiet=True))
            after = dict(self.observation(env, step)._asdict())
            require((int(native[0]), bool(native[2])) == (after["hit"], after["done"]), "native return packet")
            _, update = self.call("public_update", lambda action=action, after=after: actor.update(action, after))
            next_state = self.witness(actor, env, after)
            require(after["position"] != packet["position"], "inbounds movement")
            save({"kind": "step", "episode_id": episode_id, "step": step, "action": action,
                  "scores": [float(x) if np.isfinite(x) else None for x in scores],
                  "allowed_actions": list(packet["valid_actions"]), "public": after,
                  "posterior_before": state, "posterior_after": next_state, "choose_seconds": choice,
                  "choose_instrumented_seconds": raw, "excluded_io_seconds": removed, "update_seconds": update,
                  "environment_seconds": elapsed, "native_p_end": float(native[1])})
            chooses += choice
            updates += update
            environment += elapsed
            raw_chooses += raw
            excluded += removed
            packet, state = after, next_state
            if packet["done"]:
                break
        allocation = self.head_setup[arm]["seconds"] / 72 + self.model_module_seconds / 432 if head else 0.
        row = {**identity, "steps": step, "found": packet["done"], "updates": step, "blocked_steps": 0,
               "init_seconds": init_seconds, "choose_seconds": chooses, "update_seconds": updates,
               "setup_allocation_seconds": allocation, "controller_seconds": init_seconds + chooses + updates + allocation,
               "choose_instrumented_seconds": raw_chooses, "excluded_io_seconds": excluded,
               "environment_seconds": environment, "environment_reset_seconds": reset_seconds,
               "source_evaluation_only": env.source.tolist(), "final_public": packet, "final_update_assimilated": True,
               "draws_evaluation_only": [{k: r[k] for k in ("channel", "index", "uniform", "selected_index", "cdf_mass")}
                                        for r in env.draw_log],
               "storage": {"actor": actor.storage_bytes(), "features": feature_map.storage_bytes() if feature_map else None,
                           "head": head.storage_bytes() if head else None}}
        return row

    def evaluate(self, fits):
        self.evaluation_setup(fits)
        rows, paired = [], {}
        tick = time.perf_counter()
        with (self.out / "eval-transitions.jsonl.gz").open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as stream:
                for ri, (regime, first) in enumerate(FIRST.items()):
                    for case in range(CASES):
                        offset = (ri * CASES + case) % len(ARMS)
                        for arm in ARMS[offset:] + ARMS[:offset]:
                            row = self.episode(regime, first + case, case, arm, stream)
                            key = regime, case
                            source = row["source_evaluation_only"]
                            draws = {(d["channel"], d["index"]): d["uniform"] for d in row["draws_evaluation_only"]}
                            require(len(draws) == len(row["draws_evaluation_only"]), "unique channel/index draw witness")
                            if key in paired:
                                original_source, original_draws = paired[key]
                                require(original_source == source and all(original_draws[k] == draws[k]
                                        for k in original_draws.keys() & draws.keys()), "matched source and common uniform prefixes")
                                original_draws.update(draws)
                            else:
                                paired[key] = source, draws
                            stream.flush()
                            raw.flush()
                            os.fsync(raw.fileno())
                            for journal in self.streams.values():
                                os.fsync(journal.fileno())
                            self.emit("evaluation.jsonl", row, durable=True)
                            self.receipt["completed_episodes"] += 1
                            self.pending_episode = None
                            rows.append(row)
                            self.budget.check(force=True)
                        print(json.dumps({"phase": "evaluation", "completed": len(rows), "total": 504}), flush=True)
            raw.flush()
            os.fsync(raw.fileno())
        self.evaluation_seconds = time.perf_counter() - tick
        return aggregate(rows, self.mixtures)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        self.budget = None

        def interrupted(_signal, _frame):
            raise InterruptedError("original supervisor interrupted study")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.budget = ClockBudget(self.out)
            tick = time.perf_counter()
            self.bind()
            authentication_seconds = time.perf_counter() - tick
            self.setup()
            means = self.collect()
            features, targets = self.training_data(means)
            fits = self.train(features, targets)
            result = self.evaluate(fits)
            require(self.receipt["completed_panels"] == 558 and self.receipt["completed_fits"] == 6
                    and self.receipt["completed_episodes"] == 504 and not self.pending
                    and not self.sampler_pending and not self.training_pending
                    and self.pending_panel is None and self.pending_episode is None
                    and all(c["attempted"] == c["returned"] for c in self.calls.values()), "complete end-to-end work")
            require(self.calls["native_reset"]["returned"] == self.calls["actor_initialization"]["returned"] == 504
                    and self.calls["native_step"]["returned"] == self.calls["public_update"]["returned"]
                    <= 504 * HORIZON and self.calls["head_predict"]["returned"] <= 432 * HORIZON,
                    "bounded actual native and controller work")
            costs = {"authentication_seconds": authentication_seconds, "common_input_setup_seconds": self.setup_seconds,
                     "continuation_collection_seconds": self.collection_seconds, "training_data_seconds": self.preparation_seconds,
                     "model_module_seconds": self.model_module_seconds, "torch_setup_seconds": self.training_setup_seconds,
                     "training_pairs_seconds": math.fsum(p["seconds"] for p in self.pair_costs),
                     "native_setup_seconds": self.native_setup_seconds,
                     "head_restore_seconds": math.fsum(v["seconds"] for v in self.head_setup.values()),
                     "evaluation_seconds": self.evaluation_seconds}
            result.update(costs=costs, physical_stage_seconds=math.fsum(costs.values()),
                          measured_journal_io_seconds=self.io_seconds, fit_costs=self.fit_costs,
                          timing_scope="Stage totals are disjoint. The measured journal timer covers work/training/panel "
                          "journal writes and evaluation transition writes, not every NPZ/gzip/serialization operation. "
                          "It overlaps stage totals and is not added again. "
                          "Actual controller intervals exclude only measured work-journal I/O; monitoring is outside them. "
                          "Existing analytic-label generation is inherited work. This is one CPU timing pass.")
            write(self.out / "summary.json", result)
            for stream in self.streams.values():
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
            for name, pin in self.plan["sources"].items():
                require(digest(path(name), self.check)["sha256"] == pin, "frozen sources unchanged after execution")
            for name, desc in self.plan["inputs"].items():
                require(digest(path(name), self.check) == desc, "frozen inputs unchanged after execution")
            require(digest(self.args.plan, self.check)["sha256"] == self.args.plan_sha256
                    and digest(self.args.supervision, self.check)["sha256"] == self.receipt["supervision_sha256"],
                    "unchanged plan and original launch")
            files = {p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()}
            self.receipt.update(status="completed", pilot_continuation=result["pilot_continuation"],
                requires_successful_original_supervisor=True, calls=self.calls, sampler_calls=self.sampler_calls,
                training_calls=self.training_calls, sampler_events=self.sampler_events, pending=[], pending_panel=None,
                pending_episode=None, started_ns=self.budget.start, finished_ns=self.budget.clock.now_ns(),
                clock_backend=self.budget.clock.backend, files=files,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                timing_scope="Worker through final payload hashing; original parent covers receipt publication and process exit.")
            self.receipt["wall_seconds"] = (self.receipt["finished_ns"] - self.budget.start) / 1e9
            self.check()
            write(self.out / "receipt.json", self.receipt)
            self.budget.check(force=True)
            print(json.dumps({"status": "completed", "pilot_continuation": result["pilot_continuation"],
                              "receipt": digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            cleanup = []
            for stream in self.streams.values():
                try:
                    stream.close()
                except BaseException as secondary:  # noqa: BLE001 - try all handles without replacing primary failure
                    cleanup.append(repr(secondary))
            self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                cleanup_errors=cleanup, calls=self.calls, sampler_calls=self.sampler_calls,
                training_calls=getattr(self, "training_calls", {}), sampler_events=self.sampler_events,
                pending=list(self.pending.values()), sampler_pending=list(self.sampler_pending.values()),
                training_pending=list(getattr(self, "training_pending", {}).values()), pending_panel=self.pending_panel,
                pending_episode=self.pending_episode, pending_emission=self.pending_emission, context=self.context)
            if hasattr(error, "progress"):
                self.receipt["helper_failure_progress"] = self.json_value(error.progress)
            try:
                self.receipt["failure_elapsed_seconds"] = ((self.budget.clock.now_ns() - self.budget.start) / 1e9
                                                           if self.budget is not None else None)
            except BaseException as secondary:  # noqa: BLE001 - preserve clock errors as secondary evidence
                self.receipt["failure_clock_error"] = repr(secondary)
            try:
                target = self.out / "receipt.json"
                if target.exists():
                    target.rename(self.out / "receipt.invalid.json")
                self.receipt["partial_files"] = {p.name: {"bytes": p.stat().st_size} for p in self.out.iterdir() if p.is_file()}
                self.receipt["failure_file_scope"] = "Sizes only; no potentially unbounded hashing after a resource failure."
                write(target, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - no retry; retain publication failure on original
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def aggregate(rows, mixtures):
    expected = []
    for ri, (regime, first) in enumerate(FIRST.items()):
        for case in range(24):
            offset = (ri * 24 + case) % 7
            expected.extend((regime, first + case, case, 1 + case % 3, case // 3, arm)
                            for arm in ARMS[offset:] + ARMS[:offset])
    require([(r["regime"], r["seed"], r["case"], r["initial_hit"], r["block"], r["arm"]) for r in rows]
            == expected, "exact 504-episode order and coverage")
    metrics = ("found", "steps", "init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds",
               "controller_seconds", "environment_seconds")
    for row in rows:
        require(type(row["steps"]) is int and 1 <= row["steps"] <= HORIZON and type(row["found"]) is bool
                and (row["found"] or row["steps"] == HORIZON) and row["updates"] == row["steps"]
                and row["blocked_steps"] == 0 and all(math.isfinite(row[k]) and row[k] >= 0 for k in metrics),
                "complete finite episode outcome/cost")
        require(abs(row["controller_seconds"] - math.fsum(row[k] for k in
                    ("init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds"))) <= 1e-9,
                "complete controller cost sum")
    regimes, positive, competence, relative = {}, [], [], []

    def criterion(group, name, value, threshold, passed):
        group.append({"name": name, "value": value, "threshold": threshold, "passes": bool(passed)})

    for regime in FIRST:
        weights = {h: float(mixtures[regime][h]) for h in (1, 2, 3)}
        require(all(v > 0 for v in weights.values()) and abs(math.fsum(weights.values()) - 1) <= 1e-12,
                "positive-hit case mixture")
        local = [r for r in rows if r["regime"] == regime]

        def weighted(subset, metric, weights=weights):
            return math.fsum(weights[h] * math.fsum(float(r[metric]) for r in subset if r["initial_hit"] == h)
                             / sum(r["initial_hit"] == h for r in subset) for h in (1, 2, 3))

        means = {a: {m: weighted([r for r in local if r["arm"] == a], m) for m in metrics} for a in ARMS}
        families = {k: {m: math.fsum(means[f"{k}@{s}"][m] for s in SEEDS) / 3 for m in metrics} for k in KINDS}
        blocks = [{a: weighted([r for r in local if r["arm"] == a and r["block"] == block], "steps")
                   for a in ARMS} for block in range(8)]
        teacher = means["analytic_inbounds"]
        criterion(positive, f"{regime}.analytic_control.success", teacher["found"], .95, teacher["found"] >= .95)
        for seed in SEEDS:
            model = means[f"continuation@{seed}"]
            criterion(competence, f"{regime}.{seed}.success", model["found"], .95, model["found"] >= .95)
            threshold = 1.05 * teacher["steps"]
            criterion(competence, f"{regime}.{seed}.moves", model["steps"], threshold, model["steps"] <= threshold)
        candidate, reference = families["continuation"], families["analytic"]
        criterion(relative, f"{regime}.success", candidate["found"], reference["found"], candidate["found"] >= reference["found"])
        criterion(relative, f"{regime}.moves", candidate["steps"], .95 * reference["steps"], candidate["steps"] <= .95 * reference["steps"])
        gains = [math.fsum(b[f"analytic@{s}"] - b[f"continuation@{s}"] for s in SEEDS) / 3 for b in blocks]
        count = sum(v > 0 for v in gains)
        criterion(relative, f"{regime}.positive_blocks", count, 6, count >= 6)
        criterion(relative, f"{regime}.controller_cost", candidate["controller_seconds"], 1.05 * reference["controller_seconds"],
                  candidate["controller_seconds"] <= 1.05 * reference["controller_seconds"])
        regimes[regime] = {"weights": weights, "means": means, "family_means": families, "blocks": blocks,
            "paired_family_block_gains": gains,
            "strata": {h: {a: {m: math.fsum(float(r[m]) for r in local if r["arm"] == a and r["initial_hit"] == h) / 8
                                for m in metrics} for a in ARMS} for h in (1, 2, 3)},
            "raw_counts": {a: {"found": sum(r["found"] for r in local if r["arm"] == a), "episodes": 24} for a in ARMS}}
    require((len(positive), len(competence), len(relative)) == (3, 18, 12), "all 33 frozen conditions")
    return {"version": VERSION, "episodes": 504, "paired_cases": 72, "regimes": regimes,
            "positive_control_checks": positive, "competence_checks": competence, "relative_checks": relative,
            "pilot_continuation": all(r["passes"] for r in positive + competence + relative),
            "architecture_advantage_established": False, "inherited_results_revised": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--supervision", type=Path)
    parser.add_argument("--engineering", type=Path, action="append", default=[])
    args = parser.parse_args()
    require(args.output.is_absolute(), "absolute output path")
    if args.mode == "plan":
        make_plan(args)
    else:
        require(args.plan and args.plan.is_absolute() and args.plan_sha256 and args.supervision
                and args.supervision.is_absolute() and not args.engineering, "run frozen absolute inputs")
        Run(args).execute()


if __name__ == "__main__":
    main()
