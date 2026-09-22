"""Frozen appended-replicate precision comparison, paired fitting and fresh search.

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
VERSION = "otto-target-precision-v1"
BASE = "output/otto-teacher-learning-v1"
COHORT = "output/otto-teacher-cohort-v1"
OUTPUT = "output/otto-target-precision-v1"
SYMM = "output/otto-symmetry-head-v1/run-01"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
PINS = {f"{BASE}/plan-01.json": "7c155b10b630ee6fb7b16e302accfc0883b4a39ff7a0cb6f94fe4b6ffb0bec90",
        f"{BASE}/run-01/receipt.json": "f55f19acc8b8b22ca936b78a1023835556b1ecb80b9bb84bbaa1b3dc661a0bf3",
        f"{BASE}/supervision-01.terminal.json": "4dd9a826c8bfd0747e429ac0f3471b5e1614b03fe78f16f87e6a7834e19ebdf7",
        f"{BASE}/audit-02/receipt.json": "3ef858eb393a2896a48e9ec7bd9a9feb4600a3c4c3111b8ec3484829a56826d0",
        f"{BASE}/audit-supervision-02.terminal.json": "b0278cdd9d97f20683c9026da8ba5083db71d1661b81f768a2cf8fc3254c0270"}
NEW = {"scripts/study_otto_target_precision.py", "tests/test_study_otto_target_precision.py",
       "src/openjev/research/otto_teacher_precision.py", "tests/test_otto_teacher_precision.py",
       "research/otto-target-precision-protocol.md"}
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
SEEDS, KINDS = (20101, 20102, 20103), ("r16", "r64")
ARMS = tuple(f"{k}@{s}" for s in SEEDS for k in KINDS) + ("analytic_inbounds",)
FIRST = {"lambda3": 1050001, "lambda4": 1060001, "lambda5": 1070001}
HORIZON, CASES, LABEL_SEED = 2188, 24, 19000002
LIMITS = {"native_seconds": 10800, "rss_bytes": 4 * 1024**3, "output_bytes": 32 * 1024**3}
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
    return {"anchors": 558, "label_seed": LABEL_SEED, "replicate_ids": list(range(16, 64)), "reused_replicate_ids": list(range(16)),
            "target_scale": "immutable original R16 continuation global TRAIN scale for both arms",
            "horizon": HORIZON, "fit_seeds": list(SEEDS), "target_arms": list(KINDS),
            "epochs": 80, "batch_size": 128, "learning_rate": .0003, "gradient_clip": 5.,
            "shuffle_seed_offset": 20000, "evaluation_first_seeds": FIRST, "cases_per_regime": CASES,
            "evaluation_arms": list(ARMS), "evaluation_episodes": 504, "native_resets": 504,
            "native_step_cap": 504 * HORIZON, "learned_forward_cap": 432 * HORIZON,
            "optimizer_updates": 2400, "head_allocation_episodes": 72, "module_allocation_episodes": 432,
            "Ngrid": 53, "Nhits": 4, "R_dt": 2., "event_bytes": 512, "gzip_level": 1,
            "timing": "Complete actor costs exclude measured journal I/O; first deployment forward is included; shared imports may be reused.",
            "checkpoint": "fixed final epoch80; no validation selection or inference ensemble"}


def parent_closure(check):
    for name, pin in PINS.items():
        require(digest(path(name), check)["sha256"] == pin, "closed R16 study external pin")
    prior = read(path(f"{BASE}/plan-01.json"))
    worker = read(path(f"{BASE}/run-01/receipt.json"))
    audit = read(path(f"{BASE}/audit-02/receipt.json"))
    require(len(prior["sources"]) == 151 and len(prior["inputs"]) == 526,
            "complete original R16 source/input closure")
    require(worker["status"] == audit["status"] == "completed" and audit["agreement"] is True
            and worker["plan_sha256"] == PINS[f"{BASE}/plan-01.json"]
            and audit["version"] == "otto-teacher-learning-saved-audit-v2"
            and worker["completed_panels"] == 558 and worker["completed_fits"] == 6
            and worker["completed_episodes"] == 504 and len(worker["files"]) == 583,
            "complete audited R16 study, irrespective of scientific failure")
    sources, inputs = dict(prior["sources"]), dict(prior["inputs"])

    def add(name, expected=None):
        desc = digest(path(name), check)
        require(expected is None or desc == expected, "inherited payload descriptor")
        require(name not in inputs or inputs[name] == desc, "immutable input identity")
        inputs[name] = desc
        return desc

    def closure(directory, receipt):
        require({p.name for p in path(directory).iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "complete inherited flat payload closure")
        add(f"{directory}/receipt.json")
        for name, desc in receipt["files"].items():
            require(Path(name).name == name, "flat payload member")
            add(f"{directory}/{name}", desc)

    for name in PINS:
        add(name)
    closure(f"{BASE}/run-01", worker)
    closure(f"{BASE}/audit-02", audit)
    for receipt, prefix in ((worker, "supervision-01"), (audit, "audit-supervision-02")):
        launch_name, terminal_name = f"{BASE}/{prefix}.launch.json", f"{BASE}/{prefix}.terminal.json"
        require(add(launch_name)["sha256"] == receipt["supervision_sha256"], "original launch pin")
        launch, terminal = read(path(launch_name)), read(path(terminal_name))
        require(all(terminal[k] == v for k, v in launch.items())
                and terminal["status"] == "completed" and terminal["returncode"] == 0
                and terminal["group_absent"] is True and terminal["timed_out"] is False
                and terminal["error"] is None and terminal["clock_error"] is None
                and terminal["cleanup"]["errors"] == [] and terminal["cleanup"]["reaped"] is True
                and terminal["cleanup"]["group_absent"] is True
                and launch["started_ns"] <= receipt["started_ns"] <= receipt["finished_ns"]
                <= terminal["finished_ns"] < launch["deadline_ns"], "successful original parent closure")
        add(f"{BASE}/{prefix}.log")
    audit_plan_name = f"{BASE}/audit-plan-02.json"
    require(add(audit_plan_name)["sha256"] == audit["audit_plan_sha256"], "successful V2 audit plan pin")
    audit_plan = read(path(audit_plan_name))
    require(audit_plan["inputs"] == audit["producer_inputs"]
            and all(v["sha256"] == PINS[str(Path(v["path"]).relative_to(ROOT))]
                    for v in audit_plan["inputs"].values()), "V2 external producer identity joins")
    for name, pin in audit_plan["sources"].items():
        require(name not in sources or sources[name] == pin, "immutable inherited source union")
        sources[name] = pin
    require(digest(path("scripts/audit_otto_teacher_learning_v2.py"), check) == audit["source"], "V2 auditor bytes")
    require(audit_plan["preserved_inputs"] == audit["preserved_inputs"], "failed audit lineage join")
    for name, pin in audit_plan["preserved_inputs"].items():
        require(add(name)["sha256"] == pin, "preserved V1 failure/diagnosis")
    failed = read(path(f"{BASE}/audit-01/receipt.json"))
    require(failed["status"] == "failed" and failed["agreement"] is False, "original failed audit retained")
    closure(f"{BASE}/audit-01", failed)
    add(f"{BASE}/audit-supervision-01.log")
    for name, desc in inputs.items():
        require(digest(path(name), check) == desc, "complete inherited input hash")
    for name, pin in sources.items():
        require(digest(path(name), check)["sha256"] == pin, "inherited source unchanged")
    rows = prior["selections"]
    require(len(rows) == 558 and [r["anchor_id"] for r in rows] == list(range(558)), "canonical fixed cohort")
    continuations = 48 * sum(len(r["public"]["valid_actions"]) for r in rows)
    steps = continuations * HORIZON
    bounds = {"continuations": continuations, "reused_continuations": 32304, "combined_continuations": 129216,
              "continuation_steps": steps, "sampler_events": 195 * len(rows) + 4 * continuations + 8 * steps}
    require((continuations, steps, bounds["sampler_events"]) == (96912, 212043456, 1696844106),
            "geometry-derived new R48 bounds")
    bounds.update(uncompressed_event_bytes=512 * bounds["sampler_events"], compression_guaranteed=False,
                  compressed_output_hard_cap=LIMITS["output_bytes"])
    return {**prior, "sources": sources}, inputs, bounds


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
    seed_name = f"{OUTPUT}/seed-reservation-01.json"
    reservation = read(path(seed_name))
    require(reservation["status"] == "clear" and all(reservation[key] == configuration()[key]
            for key in ("fit_seeds", "evaluation_first_seeds", "cases_per_regime")), "reviewed seed reservation")
    inputs[seed_name] = digest(path(seed_name), budget.check)
    noise_base = "output/otto-target-noise-v1"
    noise_pins = {f"{noise_base}/plan-01.json": "8e8041d43f4e7ab1a56163d8cd2730137aa5fdcc415705d330b161a0f3f90031",
                  f"{noise_base}/run-01/receipt.json": "7a9bece4a8e54915e69f5198134fc671996bb80d66da5ba9ac137aa7dcff41ca"}
    for name, pin in noise_pins.items():
        inputs[name] = digest(path(name), budget.check)
        require(inputs[name]["sha256"] == pin, "closed exploratory noise evidence")
    noise = read(path(f"{noise_base}/run-01/receipt.json"))
    require(noise["status"] == "completed" and noise["plan_sha256"] == noise_pins[f"{noise_base}/plan-01.json"],
            "completed exploratory diagnostic, not efficacy admission")
    require({p.name for p in path(f"{noise_base}/run-01").iterdir()} == set(noise["files"]) | {"receipt.json"},
            "closed exploratory payloads")
    for name, desc in {**noise["sources"], **{f"{noise_base}/run-01/{n}": d for n, d in noise["files"].items()}}.items():
        require(digest(path(name), budget.check) == desc, "noise source/payload descriptor")
        inputs[name] = desc
    for name in (f"{noise_base}/engineering-review-01.json", "research/otto-target-noise-results.md"):
        inputs[name] = digest(path(name), budget.check)
    directories = sorted(p for p in path(OUTPUT).glob("engineering-[0-9][0-9]") if p.is_dir())
    require(directories and set(args.engineering) == set(directories) and len(args.engineering) == len(directories),
            "all direct precision engineering attempts required")
    covered = set()
    for directory in directories:
        receipt = read(directory / "receipt.json")
        require(receipt["status"] in ("passed", "failed") and receipt["commands"], "complete engineering attempt")
        require({p.name for p in directory.iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "flat complete engineering closure")
        for name, desc in receipt["files"].items():
            require(Path(name).name == name and digest(directory / name, budget.check) == desc, "engineering log descriptor")
        if receipt["status"] == "passed":
            require(receipt["source_before"] == receipt["source_after"] and all(type(c["returncode"]) is int
                    and c["returncode"] == 0 for c in receipt["commands"]), "passed unchanged engineering")
            for name, pin in receipt["source_after"].items():
                require(sources[name] == pin, "qualification current source pin")
                covered.add(name)
        for p in directory.iterdir():
            inputs[str(p.relative_to(ROOT))] = digest(p, budget.check)
    require(NEW - {"research/otto-target-precision-protocol.md"} <= covered, "qualified new helper/runner and tests")
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    require(all(runtime[k] == prior[k] for k in runtime), "unchanged qualified runtime")
    plan = {"version": VERSION, "configuration": configuration(), "limits": LIMITS, "bounds": bounds,
            "sources": sources, "inputs": inputs, "environment": THREADS, **runtime,
            "selections": prior["selections"], "status": "frozen_before_label_generation"}
    write(args.output, plan)
    budget.check()
    print(json.dumps({"status": plan["status"], "plan": digest(args.output)}))


def checked_records(records, actions, replicates):
    """Canonical complete saved-record panel; no missing work becomes censoring."""
    actions, replicates = tuple(actions), tuple(replicates)
    require(actions == tuple(sorted(set(actions))) and actions
            and all(type(a) is int and 0 <= a < 4 for a in actions), "canonical eligible action IDs")
    by_key = {}
    for row in records:
        require(isinstance(row, dict) and set(row) == {"replicate_id", "first_action", "steps", "found"},
                "exact continuation record schema")
        replicate, action, steps, found = (row[k] for k in ("replicate_id", "first_action", "steps", "found"))
        require(type(replicate) is int and replicate in replicates and type(action) is int and action in actions
                and type(steps) is int and 1 <= steps <= HORIZON and type(found) is bool
                and (found or steps == HORIZON), "complete valid continuation cost")
        require((replicate, action) not in by_key, "duplicate continuation identity")
        by_key[replicate, action] = dict(row)
    expected = [(replicate, action) for replicate in replicates for action in actions]
    require(set(by_key) == set(expected), "complete prescribed replica/action coverage")
    return [by_key[key] for key in expected]


def merge_record_panels(old_records, new_records, actions):
    """Merge 16+48 integer costs, preserving every replicate and censored flag."""
    old = checked_records(old_records, actions, range(16))
    new = checked_records(new_records, actions, range(16, 64))
    combined = old + new
    sums = {a: sum(r["steps"] for r in combined if r["first_action"] == a) for a in actions}
    means = [sums[a] / 64 if a in sums else None for a in range(4)]
    return {"records": combined, "integer_sums": {str(a): sums[a] for a in actions}, "means": means,
            "reused_records": len(old), "new_records": len(new), "total_records": len(combined)}


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
                and self.plan["selections"] == prior["selections"] and self.plan["bounds"] == bounds,
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
                and launch["cwd"] == str(ROOT) == str(Path.cwd()) and launch["cap_seconds"] == LIMITS["native_seconds"]
                and launch["clock_backend"] == self.budget.clock.backend
                and launch["clock_source_sha256"] == CLOCK_PIN
                and launch["watchdog_sha256"] == self.plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and launch["deadline_ns"] == launch["started_ns"] + LIMITS["native_seconds"] * 10**9
                and launch["started_ns"] <= self.budget.start < launch["deadline_ns"], "original supervision")
        self.budget.deadline = launch["deadline_ns"]
        self.receipt.update(plan_sha256=self.args.plan_sha256, supervision_sha256=digest(self.args.supervision)["sha256"])
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()}, "launch": launch})

    def setup(self):
        tick = time.perf_counter()
        import numpy as np
        self.np = np
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research.otto_teacher_costs import ContinuationRecord, summarize_costs
        from openjev.research.otto_teacher_rollouts import sample_teacher_panel
        self.sample, self.reduce, self.record_class = sample_teacher_panel, summarize_costs, ContinuationRecord
        with np.load(path(f"{COHORT}/run-01/anchors.npz"), allow_pickle=False) as archive:
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
        with np.load(path(f"{BASE}/run-01/training-data.npz"), allow_pickle=False) as archive:
            expected = {"features", "analytic_costs", "continuation_means", "allowed", "anchor_ids"}
            expected |= {f"{kind}_{key}" for kind in ("analytic", "continuation") for key in
                         ("centered_float64", "scaled_float32", "weights_float64", "weights_float32", "allowed")}
            require(set(archive.files) == expected, "exact inherited training cache schema")
            self.features, self.old_costs = archive["features"], archive["continuation_means"]
            self.allowed = archive["allowed"]
            reference_arrays = {key: archive[f"continuation_{key}"] for key in
                                ("centered_float64", "scaled_float32", "weights_float64", "weights_float32", "allowed")}
            require(np.array_equal(archive["anchor_ids"], np.arange(558, dtype=np.int64)), "inherited feature row order")
        info = read(path(f"{BASE}/run-01/training-data.json"))
        require(info["rows"] == self.rows and self.features.dtype == np.float32
                and self.features.shape == (558, 2836) and np.isfinite(self.features).all()
                and hashlib.sha256(self.features.tobytes()).hexdigest() == info["features_sha256"], "unaltered original features")
        self.reference = {"kind": "continuation", "scale": info["scales"]["continuation"], **reference_arrays}
        expected_mask = np.array([[a in r["public"]["valid_actions"] for a in range(4)] for r in self.rows], dtype=np.bool_)
        require(self.allowed.dtype == np.bool_ and np.array_equal(self.allowed, expected_mask)
                and self.old_costs.dtype == np.float64 and self.old_costs.shape == (558, 4), "inherited cost/mask identity")
        for value in (self.features, self.old_costs, self.allowed, *reference_arrays.values()):
            value.setflags(write=False)
        self.old_panels = []
        with path(f"{BASE}/run-01/panels.jsonl").open("rb") as stream:
            for row in self.rows:
                self.check()
                raw = [stream.readline(128 * 1024 + 1) for _ in range(2)]
                require(all(line.endswith(b"\n") and len(line) <= 128 * 1024 for line in raw), "complete old panel boundary")
                attempt, panel = (json.loads(line) for line in raw)
                identity = {"phase": "sampling", "anchor_id": row["anchor_id"]}
                require(attempt == {"event": "attempt", **identity} and panel["event"] == "return"
                        and all(panel[k] == v for k, v in identity.items()), "ordered immutable R16 panel")
                name = f"panel-{row['anchor_id']:03d}.jsonl.gz"
                require(panel["file"] == {"path": name, **self.plan["inputs"][f"{BASE}/run-01/{name}"]},
                        "inherited panel bytes joined to saved records")
                records = checked_records(panel["records"], row["public"]["valid_actions"], range(16))
                for action in row["public"]["valid_actions"]:
                    require(self.old_costs[row["anchor_id"], action] == sum(r["steps"] for r in records
                            if r["first_action"] == action) / 16, "immutable R16 mean matches original integer costs")
                self.old_panels.append({"records": records, "file": {"path": f"{BASE}/run-01/{name}",
                    **self.plan["inputs"][f"{BASE}/run-01/{name}"]}})
            require(stream.read(1) == b"", "exact inherited panel closure")
        self.historical_collection_seconds = read(path(f"{BASE}/run-01/collection.json"))["seconds"]
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
                        anchor_id=anchor, replicate_ids=range(16, 64), horizon=HORIZON, check=self.check, emit=emit)
                raw.flush()
                os.fsync(raw.fileno())
            require(not self.sampler_pending, "complete sampler panel")
            reduced = self.reduce(records, eligible_actions=row["public"]["valid_actions"],
                                   replicate_ids=range(16, 64), horizon=HORIZON)
            panel = {"event": "return", **self.context, "seconds": time.perf_counter() - panel_tick,
                     "file": {"path": filename, **digest(self.out / filename, self.check)},
                     "records": [{"replicate_id": r.replicate_id, "first_action": r.first_action,
                                  "steps": r.steps, "found": r.found} for r in records], "summary": reduced}
            merged = merge_record_panels(self.old_panels[anchor]["records"], panel["records"], row["public"]["valid_actions"])
            objects = [self.record_class(**r) for r in merged["records"]]
            combined_summary = self.reduce(objects, eligible_actions=row["public"]["valid_actions"],
                                           replicate_ids=range(64), horizon=HORIZON)
            blocks = [self.reduce([r for r in objects if start <= r.replicate_id < start + 16],
                       eligible_actions=row["public"]["valid_actions"], replicate_ids=range(start, start + 16),
                       horizon=HORIZON) for start in (0, 16, 32, 48)]
            block_actions = []
            for block in blocks:
                minimum = min(r["mean_cost"] for r in block["actions"])
                block_actions.append(next(r["action"] for r in block["actions"] if abs(r["mean_cost"] - minimum) < 1e-10))
            for action in combined_summary["actions"]:
                require(action["mean_cost"] == merged["means"][action["action"]], "integer merged R64 reduction")
                means[anchor, action["action"]] = action["mean_cost"]
            self.emit("combined-labels.jsonl", {"anchor_id": anchor, "source_r16": self.old_panels[anchor]["file"],
                      "source_r48": panel["file"], **merged, "summary_r64": combined_summary,
                      "disjoint16": [{"replicate_ids": list(range(start, start + 16)), "summary": block}
                                     for start, block in zip((0, 16, 32, 48), blocks, strict=True)],
                      "block16_actions": block_actions,
                      "block16_action_disagreements": sum(block_actions[i] != block_actions[j]
                                                            for i in range(4) for j in range(i + 1, 4)),
                      "scope": "All four disjoint blocks retained descriptively; no filtering or confidence admission."}, durable=True)
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
        expected_calls = {"anchor_snapshot": 558, "source_generator": 558 * 48, "source_draw": 558 * 48,
                          "teacher_snapshot": total_records, "hit_generator": total_records,
                          "teacher_choose": total_steps - total_records, "movement": total_steps,
                          "hit_draw": total_steps - total_found, "teacher_update": total_steps}
        require(self.sampler_calls == {k: {"attempted": v, "returned": v} for k, v in expected_calls.items()}
                and self.sampler_events == 195 * 558 + 4 * total_records + 8 * total_steps - 2 * total_found,
                "complete sampler event/count identities")
        self.collection_seconds = time.perf_counter() - tick
        write(self.out / "collection.json", {"panels": 558, "records": total_records, "steps": total_steps,
              "found": total_found, "censored": total_records - total_found, "sampler_events": self.sampler_events,
              "calls": self.sampler_calls, "seconds": self.collection_seconds,
              "reused_records": 32304, "combined_records": 129216, "replicate_ids": list(range(16, 64)),
              "reducer_calls": 558 * 6, "historical_r16_collection_seconds": self.historical_collection_seconds,
              "timing_scope": "All new R48 sampling, merging, reductions, serialization, gzip, hashes and fsyncs; R16 reused."})
        return means

    def training_data(self, continuation):
        tick = time.perf_counter()
        from openjev.research import otto_symmetry_head as model
        self.model = model
        self.model_module_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_teacher_precision import build_precision_targets
        bundle = build_precision_targets(self.old_costs, continuation, self.allowed,
            [row["episode_id"] for row in self.rows], reference_continuation=self.reference)
        counts = self.allowed.sum(axis=1, dtype=self.np.int64)
        diagnostic_r64_rms = float(self.np.sqrt(self.np.sum(bundle["r64"]["weights_float64"]
            * self.np.sum(bundle["r64"]["centered_float64"] ** 2, axis=1, dtype=self.np.float64) / counts,
            dtype=self.np.float64) / len(self.rows)))
        require(bundle["r16"]["scale"] == bundle["r64"]["scale"] == self.reference["scale"], "fixed shared R16 scale")
        for key, value in self.reference.items():
            if isinstance(value, self.np.ndarray):
                require(bundle["r16"][key].dtype == value.dtype and bundle["r16"][key].shape == value.shape
                        and bundle["r16"][key].tobytes() == value.tobytes(), "byte-identical original R16 training arrays")
        arrays = {"features": self.features, "r16_costs": self.old_costs, "r64_costs": continuation,
                  "allowed": self.allowed, "anchor_ids": self.np.arange(558, dtype=self.np.int64)}
        for kind in KINDS:
            arrays.update({f"{kind}_{k}": v for k, v in bundle[kind].items() if isinstance(v, self.np.ndarray)})
        data_file = self.array_file("training-data.npz", arrays)
        self.preparation_seconds = time.perf_counter() - tick
        write(self.out / "training-data.json", {"file": data_file, "rows": self.rows, "episodes": 144,
              "scales": {k: bundle[k]["scale"] for k in KINDS},
              "r64_unfloored_rms_diagnostic_only": diagnostic_r64_rms,
              "original_training_cache": {"path": f"{BASE}/run-01/training-data.npz",
                                          **self.plan["inputs"][f"{BASE}/run-01/training-data.npz"]},
              "features_sha256": hashlib.sha256(self.features.tobytes()).hexdigest(),
              "seconds": self.preparation_seconds, "model_module_seconds": self.model_module_seconds,
              "scope": "Original features/R16 arrays unchanged; combined R64 integer means, common original R16 TRAIN scale."})
        return self.features, bundle

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
        from openjev.research.otto_teacher_precision import train_pair
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
                    label_allocation_seconds=self.collection_seconds / 3 if kind == "r64" else 0.,
                    inherited_r16_label_allocation_seconds=self.historical_collection_seconds / 3)
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
              "historical_r16_label_cost": "Common inherited R16 work; paid before this run, not claimed zero."})
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
                     "additional_r48_collection_seconds": self.collection_seconds, "training_data_seconds": self.preparation_seconds,
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
                          "Original R16 label generation is common inherited work; new R48 work is attributed only to R64. This is one CPU timing pass.")
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
            model = means[f"r64@{seed}"]
            criterion(competence, f"{regime}.{seed}.success", model["found"], .95, model["found"] >= .95)
            threshold = 1.05 * teacher["steps"]
            criterion(competence, f"{regime}.{seed}.moves", model["steps"], threshold, model["steps"] <= threshold)
        candidate, reference = families["r64"], families["r16"]
        criterion(relative, f"{regime}.success", candidate["found"], reference["found"], candidate["found"] >= reference["found"])
        criterion(relative, f"{regime}.moves", candidate["steps"], .95 * reference["steps"], candidate["steps"] <= .95 * reference["steps"])
        gains = [math.fsum(b[f"r16@{s}"] - b[f"r64@{s}"] for s in SEEDS) / 3 for b in blocks]
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
