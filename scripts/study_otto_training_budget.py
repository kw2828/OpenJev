"""Fixed 80-versus-320 update allocation on unchanged R64 labels.

No sampler or new TRAIN data. All sources and inherited evidence are checked
before numerical imports. Final TRAIN diagnostics precede fresh deployment
loads and every autonomous episode. No selection, retries or budget extension.
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
VERSION = "otto-training-budget-v1"
BASE = "output/otto-target-precision-v1"
OUTPUT = "output/otto-training-budget-v1"
SYMM = "output/otto-symmetry-head-v1/run-01"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
PINS = {f"{BASE}/plan-01.json": "921318b9c978b208989ad66210a06adad8f104c827351eda1850f080d9f8a46a",
        f"{BASE}/run-01/receipt.json": "34b331507fe8f53058baecdbeed0e3448b91202a6971762949d1ae7017f4bb2a",
        f"{BASE}/supervision-01.terminal.json": "fc3312338d04b5a0955727651e83a62f5907cedff7be007e165d4f4637a71e37",
        f"{BASE}/audit-01/receipt.json": "549eb00f8a04904f22bb9ff351c35c4e5f965b88a6f1944d0186886ac693baa7",
        f"{BASE}/audit-supervision-01.terminal.json": "bf756bbb82639b78de692bf7bc0cba818306ce82d7ca599c99d68ff45b272a34"}
NEW = {"scripts/study_otto_training_budget.py", "tests/test_study_otto_training_budget.py",
       "src/openjev/research/otto_training_budget.py", "tests/test_otto_training_budget.py",
       "research/otto-training-budget-protocol.md"}
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
SEEDS, KINDS = (30101, 30102, 30103), ("short", "long")
ARMS = tuple(f"{k}@{s}" for s in SEEDS for k in KINDS) + ("analytic_inbounds",)
FIRST = {"lambda3": 1080001, "lambda4": 1090001, "lambda5": 1100001}
HORIZON, CASES = 2188, 24
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
    return {"anchors": 558, "fit_seeds": list(SEEDS), "target_arms": list(KINDS),
            "epochs": {"short": 80, "long": 320}, "batch_size": 128, "learning_rate": .0003,
            "gradient_clip": 5., "shuffle_seed_offset": 20000, "optimizer_updates": 6000,
            "target_scale": "unchanged prior R64 training scale and all target bytes for both arms",
            "checkpoints": 15, "matched_prefix_epoch": 80, "new_label_calls": 0, "payloads": 34,
            "training_operations": 6060, "training_events": 14550,
            "diagnostic_checkpoints": 6, "diagnostic_batch": 128, "diagnostic_views": 8,
            "diagnostic_forwards": 240, "diagnostic_view_rows": 26784,
            "evaluation_first_seeds": FIRST, "cases_per_regime": CASES, "horizon": HORIZON,
            "evaluation_arms": list(ARMS), "evaluation_episodes": 504, "native_resets": 504,
            "native_step_cap": 504 * HORIZON, "learned_forward_cap": 432 * HORIZON,
            "head_allocation_episodes": 72, "module_allocation_episodes": 432,
            "Ngrid": 53, "Nhits": 4, "R_dt": 2., "gzip_level": 1,
            "timing": "Separate final TRAIN diagnostics; fresh deployment loads; complete controller costs.",
            "checkpoint": "fixed epoch80/320, longprefix80 matched exactly; no checkpoint selection"}


def payload_names():
    roots = {"started.json", "training-data.npz", "training-data.json", "training.jsonl", "fits.jsonl",
             "training-summary.json", "diagnostics.jsonl", "diagnostic-summary.json", "deployment.json",
             "evaluation.jsonl", "eval-transitions.jsonl.gz", "work.jsonl", "summary.json"}
    return roots | {f"{arm}-{seed}-{phase}.npz" for arm in KINDS for seed in SEEDS
                    for phase in ("initial", "final", "diagnostics")} | {f"long-{seed}-prefix.npz" for seed in SEEDS}


def parent_closure(check):
    for name, pin in PINS.items():
        require(digest(path(name), check)["sha256"] == pin, "completed precision study external pin")
    prior = read(path(f"{BASE}/plan-01.json"))
    worker = read(path(f"{BASE}/run-01/receipt.json"))
    audit = read(path(f"{BASE}/audit-01/receipt.json"))
    require(len(prior["sources"]) == 158 and len(prior["inputs"]) == 1144,
            "complete precision source/input closure")
    require(worker["status"] == audit["status"] == "completed" and audit["agreement"] is True
            and worker["plan_sha256"] == PINS[f"{BASE}/plan-01.json"]
            and audit["version"] == "otto-target-precision-saved-audit-v1"
            and worker["completed_panels"] == 558 and worker["completed_fits"] == 6
            and worker["completed_episodes"] == 504 and len(worker["files"]) == 584,
            "complete audited precision study, with its scientific failure preserved")
    sources, inputs = dict(prior["sources"]), dict(prior["inputs"])

    def add(name, expected=None):
        desc = digest(path(name), check)
        require(expected is None or desc == expected, "inherited payload descriptor")
        require(name not in inputs or inputs[name] == desc, "immutable input identity")
        inputs[name] = desc
        return desc

    for name in PINS:
        add(name)
    for directory, receipt in ((f"{BASE}/run-01", worker), (f"{BASE}/audit-01", audit)):
        require({p.name for p in path(directory).iterdir()} == set(receipt["files"]) | {"receipt.json"},
                "complete inherited flat payload closure")
        for name, desc in receipt["files"].items():
            require(Path(name).name == name, "flat payload member")
            add(f"{directory}/{name}", desc)
    for receipt, prefix in ((worker, "supervision-01"), (audit, "audit-supervision-01")):
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
    audit_plan_name = f"{BASE}/audit-plan-01.json"
    require(add(audit_plan_name)["sha256"] == audit["audit_plan_sha256"], "completed independent audit plan pin")
    audit_plan = read(path(audit_plan_name))
    require(audit_plan["inputs"] == audit["producer_inputs"]
            and all(v["sha256"] == PINS[str(Path(v["path"]).relative_to(ROOT))]
                    for v in audit_plan["inputs"].values()), "external producer identity joins")
    for name, pin in audit_plan["sources"].items():
        require(name not in sources or sources[name] == pin, "immutable inherited source union")
        sources[name] = pin
    require(digest(path("scripts/audit_otto_target_precision.py"), check) == audit["source"], "completed auditor source")
    for name, desc in inputs.items():
        require(digest(path(name), check) == desc, "complete inherited input hash, including all failed lineage")
    for name, pin in sources.items():
        require(digest(path(name), check)["sha256"] == pin, "inherited source unchanged")
    rows = prior["selections"]
    require(len(rows) == 558 and [r["anchor_id"] for r in rows] == list(range(558)), "canonical fixed cohort")
    return {**prior, "sources": sources}, inputs, {
        "sampler_calls": 0, "optimizer_updates": 6000, "diagnostic_forwards": 240,
        "diagnostic_view_rows": 26784, "native_resets": 504, "native_steps": 504 * HORIZON}


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
    raw_seed_name = f"{OUTPUT}/seed-reservation-01.json"
    seed_name = f"{OUTPUT}/seed-reservation-02.json"
    raw_descriptor = digest(path(raw_seed_name), budget.check)
    reservation = read(path(seed_name))
    require(reservation["status"] == "clear" and all(reservation[key] == configuration()[key]
            for key in ("fit_seeds", "evaluation_first_seeds", "cases_per_regime"))
            and reservation["raw_scan"] == {"path": raw_seed_name, **raw_descriptor},
            "reviewed RNG-namespace seed reservation and preserved original scan")
    inputs[raw_seed_name] = raw_descriptor
    inputs[seed_name] = digest(path(seed_name), budget.check)
    directories = sorted(p for p in path(OUTPUT).glob("engineering-[0-9][0-9]") if p.is_dir())
    require(directories and set(args.engineering) == set(directories) and len(args.engineering) == len(directories),
            "all direct training-budget engineering attempts required")
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
    require(NEW - {"research/otto-training-budget-protocol.md"} <= covered, "qualified new helper/runner and tests")
    runtime = {"python_executable": sys.executable, "python_version": platform.python_version(),
               "all_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    require(all(runtime[k] == prior[k] for k in runtime), "unchanged qualified runtime")
    plan = {"version": VERSION, "configuration": configuration(), "limits": LIMITS, "bounds": bounds,
            "sources": sources, "inputs": inputs, "environment": THREADS, **runtime,
            "selections": prior["selections"], "status": "frozen_before_training"}
    write(args.output, plan)
    budget.check()
    print(json.dumps({"status": plan["status"], "plan": digest(args.output)}))



def diagnostic_metrics(raw_scores, scaled_targets, allowed, raw_costs, weights_float64,
                       weights_float32, permutations):
    """Fixed-final saved-score reductions, without fitting or state selection.

    MSE upcasts raw f32 predictions, rounded f32 targets and f32 training weights
    to f64. Physical regret/argmin rates use original f64 episode weights.
    Every weighted sum divides by N, never by a rounded sum of weights.
    """
    import numpy as np
    n = len(raw_scores)
    require(raw_scores.dtype == np.float32 and raw_scores.shape == (n, 8, 4) and n > 0
            and np.isfinite(raw_scores).all(), "finite float32[N,8,4] predictions")
    require(scaled_targets.dtype == np.float32 and scaled_targets.shape == (n, 4)
            and np.isfinite(scaled_targets).all() and allowed.dtype == np.bool_
            and allowed.shape == (n, 4) and allowed.any(axis=1).all(), "common diagnostic targets/masks")
    require(raw_costs.dtype == np.float64 and raw_costs.shape == (n, 4)
            and np.isfinite(raw_costs[allowed]).all() and np.isposinf(raw_costs[~allowed]).all(), "raw R64 label costs")
    require(weights_float64.dtype == np.float64 and weights_float32.dtype == np.float32
            and weights_float64.shape == weights_float32.shape == (n,)
            and np.isfinite(weights_float64).all() and np.isfinite(weights_float32).all()
            and (weights_float64 > 0).all() and (weights_float32 > 0).all(), "positive diagnostic weights")
    perms = np.asarray(permutations)
    require(perms.shape == (8, 4) and np.issubdtype(perms.dtype, np.integer)
            and all(sorted(p.tolist()) == list(range(4)) for p in perms)
            and perms[0].tolist() == list(range(4)), "qualified action permutations with identity first")
    inverse = np.argsort(perms, axis=1)
    mask = allowed[:, inverse]
    target = scaled_targets.astype(np.float64)[:, inverse]
    predicted = raw_scores.astype(np.float64)
    counts = mask.sum(axis=-1, keepdims=True, dtype=np.int64)
    centered = predicted - np.where(mask, predicted, 0.).sum(axis=-1, keepdims=True) / counts
    target = target - np.where(mask, target, 0.).sum(axis=-1, keepdims=True) / counts
    errors = np.where(mask, centered - target, 0.)
    per_view = np.sum(errors * errors, axis=-1, dtype=np.float64) / counts[..., 0]
    single, eight = per_view[:, 0], np.mean(per_view, axis=1, dtype=np.float64)
    actions, regrets, optimal_hits, first_matches = [], [], [], []
    for scores, eligible, costs in zip(raw_scores[:, 0], allowed, raw_costs, strict=True):
        valid = np.flatnonzero(eligible).tolist()
        minimum = min(float(scores[a]) for a in valid)
        chosen = next(a for a in valid if abs(float(scores[a]) - minimum) < 1e-10)
        label_minimum = min(float(costs[a]) for a in valid)
        label_first = next(a for a in valid if float(costs[a]) == label_minimum)
        actions.append(chosen)
        regrets.append(float(costs[chosen]) - label_minimum)
        optimal_hits.append(float(costs[chosen]) == label_minimum)
        first_matches.append(chosen == label_first)
    arrays = {"raw_scores": raw_scores, "per_view_mse": per_view, "single_view_mse": single,
              "eight_view_mse": eight, "selected_action": np.asarray(actions, dtype=np.int64),
              "label_regret": np.asarray(regrets, dtype=np.float64),
              "optimal_set_hit": np.asarray(optimal_hits, dtype=np.bool_),
              "first_argmin_match": np.asarray(first_matches, dtype=np.bool_)}
    require(np.isfinite(per_view).all() and np.isfinite(arrays["label_regret"]).all(), "finite diagnostic reductions")
    metrics = {"single_view_weighted_mse": float(np.sum(single * weights_float32.astype(np.float64), dtype=np.float64) / n),
               "eight_view_weighted_mse": float(np.sum(eight * weights_float32.astype(np.float64), dtype=np.float64) / n)}
    metrics.update({key: float(np.sum(arrays[key].astype(np.float64) * weights_float64, dtype=np.float64) / n)
                    for key in ("label_regret", "optimal_set_hit", "first_argmin_match")})
    return arrays, metrics


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.calls, self.pending, self.streams = {}, {}, {}
        self.io_seconds = 0.
        self.checkpoints, self.prefix_pairs = {}, []
        self.context, self.pending_episode = {}, None
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
                        "completed_fits": 0, "completed_diagnostics": 0, "completed_episodes": 0,
                        "sampler_calls": 0, "new_label_calls": 0}

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
                "public_update": 504 * HORIZON, "head_predict": 432 * HORIZON, "analytic_choose": 72 * HORIZON,
                "diagnostic_restore": 6, "diagnostic_forward": 240}
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
        self.checkpoints[arm, seed, phase] = desc
        if phase == "prefix":
            require(arm == "long" and ("short", seed, "final") in self.checkpoints,
                    "short final exists before the long prefix")
            previous = self.checkpoints["short", seed, "final"]
            with self.np.load(self.out / previous["path"], allow_pickle=False) as short, \
                    self.np.load(self.out / name, allow_pickle=False) as long:
                require(set(short.files) == set(long.files) and all(short[k].dtype == long[k].dtype
                        and short[k].shape == long[k].shape and short[k].tobytes() == long[k].tobytes()
                        for k in short.files), "long prefix checkpoint must exactly equal paired short final")
            self.prefix_pairs.append({"seed": seed, "short_final": previous, "long_prefix": desc, "byte_identical": True})
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
        self.rows, self.kernels, self.mixtures = self.plan["selections"], {}, {}
        for regime in FIRST:
            with np.load(path(f"{SYMM}/kernel-{regime}.npz"), allow_pickle=False) as archive:
                require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "only public kernel arrays")
                kernel, weights = archive["likelihood"], archive["initial_hit_weights"]
            require(kernel.shape == (4, 107, 107) and kernel.dtype == np.float64
                    and np.isfinite(kernel).all() and weights.shape == (4,) and weights[0] == 0
                    and (weights[1:] > 0).all(), "public kernel/mixture")
            self.kernels[regime] = np.frombuffer(kernel.tobytes(), dtype=np.float64).reshape(kernel.shape)
            self.mixtures[regime] = weights.tolist()
        fields = ("centered_float64", "scaled_float32", "weights_float64", "weights_float32", "allowed")
        with np.load(path(f"{BASE}/run-01/training-data.npz"), allow_pickle=False) as archive:
            expected = {"features", "r16_costs", "r64_costs", "allowed", "anchor_ids"} | {
                f"{kind}_{key}" for kind in ("r16", "r64") for key in fields}
            require(set(archive.files) == expected, "exact inherited precision cache schema")
            self.features, self.raw_costs = archive["features"], archive["r64_costs"]
            self.allowed = archive["allowed"]
            arrays = {key: archive[f"r64_{key}"] for key in fields}
            require(np.array_equal(archive["anchor_ids"], np.arange(558, dtype=np.int64)), "inherited row order")
        info = read(path(f"{BASE}/run-01/training-data.json"))
        require(info["rows"] == self.rows and info["episodes"] == 144 and self.features.dtype == np.float32
                and self.features.shape == (558, 2836) and np.isfinite(self.features).all()
                and hashlib.sha256(self.features.tobytes()).hexdigest() == info["features_sha256"], "unchanged original features")
        expected_mask = np.array([[a in row["public"]["valid_actions"] for a in range(4)] for row in self.rows])
        require(self.allowed.dtype == np.bool_ and np.array_equal(self.allowed, expected_mask)
                and self.raw_costs.dtype == np.float64 and self.raw_costs.shape == (558, 4)
                and np.isfinite(self.raw_costs[self.allowed]).all() and np.isposinf(self.raw_costs[~self.allowed]).all(),
                "inherited cost/mask identity")
        self.reference = {"kind": "r64", "scale": info["scales"]["r64"], **arrays}
        self.expected_reference = {"scale": self.reference["scale"],
                                   "arrays": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}}
        for value in (self.features, self.raw_costs, self.allowed, *arrays.values()):
            value.setflags(write=False)
        collection = read(path(f"{BASE}/run-01/collection.json"))
        self.inherited_label_seconds = collection["seconds"] + collection["historical_r16_collection_seconds"]
        self.setup_seconds = time.perf_counter() - tick

    def training_data(self):
        tick = time.perf_counter()
        from openjev.research import otto_symmetry_head as model
        self.model = model
        self.model_module_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        from openjev.research.otto_training_budget import build_training_targets
        bundle = build_training_targets(self.reference, expected=self.expected_reference)
        arrays = {"features": self.features, "r64_costs": self.raw_costs, "allowed": self.allowed,
                  "anchor_ids": self.np.arange(558, dtype=self.np.int64)}
        for kind in KINDS:
            require(bundle[kind]["scale"] == self.reference["scale"], "unchanged common R64 scale")
            for key, value in self.reference.items():
                if isinstance(value, self.np.ndarray):
                    require(bundle[kind][key].dtype == value.dtype and bundle[kind][key].shape == value.shape
                            and bundle[kind][key].tobytes() == value.tobytes(), "every original R64 target byte retained")
                    arrays[f"{kind}_{key}"] = bundle[kind][key]
        desc = self.array_file("training-data.npz", arrays)
        self.preparation_seconds = time.perf_counter() - tick
        write(self.out / "training-data.json", {"rows": self.rows, "episodes": 144, "file": desc,
              "scales": {kind: bundle[kind]["scale"] for kind in KINDS}, "reference": self.expected_reference,
              "original_training_cache": {"path": f"{BASE}/run-01/training-data.npz",
                  **self.plan["inputs"][f"{BASE}/run-01/training-data.npz"]},
              "features_sha256": hashlib.sha256(self.features.tobytes()).hexdigest(),
              "seconds": self.preparation_seconds, "model_module_seconds": self.model_module_seconds,
              "scope": "Exact inherited R64 features/targets/masks/weights/scale; no new labels or target reconstruction."})
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
        from openjev.research.otto_training_budget import train_pair
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
            require(overhead >= 0 and result["progress"]["completed_updates"] == 2000
                    and result["progress"]["pending"] == {}, "complete paired training")
            for kind in KINDS:
                self.fit_costs[f"{kind}@{seed}"].update(pair_setup_allocation_seconds=overhead / 2,
                    total_training_seconds=self.fit_costs[f"{kind}@{seed}"]["fit_seconds"] + overhead / 2,
                    label_allocation_seconds=0.,
                    inherited_r64_label_allocation_seconds=self.inherited_label_seconds / 3)
            self.pair_costs.append({"seed": seed, "seconds": pair_seconds, "nonfit_seconds": overhead})
            for fit in result["fits"]:
                clean = {k: v for k, v in fit.items() if k != "final_export"}
                clean["costs"] = self.fit_costs[f'{fit["arm"]}@{seed}']
                self.emit("fits.jsonl", self.json_value(clean), durable=True)
                fits.append(clean)
            print(json.dumps({"phase": "fitting", "seed": seed, "completed_fits": len(fits)}), flush=True)
        expected_calls = {"model_initialization": 6, "checkpoint_export": 15, "checkpoint_publication": 15,
                          "optimizer_initialization": 6, "optimizer_update": 6000, "inference_setup": 6,
                          "parity_numpy": 6, "parity_torch": 6}
        require(not self.training_pending and self.training_calls == {key: {"attempted": n, "returned": n}
                for key, n in expected_calls.items()} and len(self.prefix_pairs) == 3,
                "all actual matched optimizer/checkpoint/parity operations")
        write(self.out / "training-summary.json", {"calls": self.training_calls, "pairs": self.pair_costs,
              "fits": self.fit_costs, "torch_setup_seconds": self.training_setup_seconds,
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "initializations": 6, "final_exports": 6, "prefix_exports": 3, "prefix_pairs": self.prefix_pairs, "parity_numpy_calls": 6, "parity_torch_calls": 6,
              "historical_r64_label_seconds": self.inherited_label_seconds,
              "historical_cost_scope": "Common inherited R64 labels for either standalone family; no new collection."})
        return fits

    def load_head(self, checkpoint):
        filename = self.out / checkpoint["path"]
        require(digest(filename, self.check) == {k: checkpoint[k] for k in ("sha256", "bytes")}, "saved final head pin")
        with self.np.load(filename, allow_pickle=False) as archive:
            exported = {key: archive[key] for key in archive.files}
        for key in ("version", "kind", "input_dim"):
            exported[key] = exported[key].item()
        return self.model.FrozenHead(exported)

    def diagnostics(self, fits, bundle):
        require(len(fits) == self.receipt["completed_fits"] == 6 and len(self.prefix_pairs) == 3,
                "all fits and identical prefixes precede any diagnostic")
        np, tick, records = self.np, time.perf_counter(), []
        permutations = [self.model.action_permutation(g) for g in range(8)]
        for fit in fits:
            arm = f'{fit["arm"]}@{fit["seed"]}'
            self.context = {"phase": "final_train_diagnostics", "fit_id": arm}
            self.check()
            fit_tick = time.perf_counter()
            head, restored = self.call("diagnostic_restore", lambda fit=fit: self.load_head(fit["final_checkpoint"]))
            predictions = np.empty((558, 8, 4), dtype=np.float32)
            prediction_seconds = 0.
            for offset in range(0, 558, 128):
                stop = min(offset + 128, 558)
                for group in range(8):
                    self.context = {"phase": "final_train_diagnostics", "fit_id": arm,
                                    "offset": offset, "rows": stop - offset, "view": group}
                    self.check()
                    transformed = self.model.transform_features(self.features[offset:stop], group)
                    value, elapsed = self.call("diagnostic_forward", lambda head=head, transformed=transformed: head.scores(transformed))
                    require(value.dtype == np.float32 and value.shape == (stop - offset, 4)
                            and np.isfinite(value).all(), "complete saved diagnostic scores")
                    predictions[offset:stop, group] = value
                    prediction_seconds += elapsed
            target = bundle[fit["arm"]]
            arrays, metrics = diagnostic_metrics(predictions, target["scaled_float32"], target["allowed"], self.raw_costs,
                target["weights_float64"], target["weights_float32"], permutations)
            descriptor = self.array_file(f'{fit["arm"]}-{fit["seed"]}-diagnostics.npz', arrays)
            record = {"fit_id": arm, "checkpoint": fit["final_checkpoint"], "file": descriptor,
                      "metrics": metrics, "restore_seconds": restored, "prediction_seconds": prediction_seconds,
                      "seconds": time.perf_counter() - fit_tick, "forwards": 40, "view_rows": 4464,
                      "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
            self.emit("diagnostics.jsonl", record, durable=True)
            self.receipt["completed_diagnostics"] += 1
            records.append(record)
            del head
            self.budget.check(force=True)
        self.diagnostic_seconds = time.perf_counter() - tick
        require(self.calls["diagnostic_forward"]["attempted"] == self.calls["diagnostic_forward"]["returned"] == 240
                and self.calls["diagnostic_restore"]["attempted"] == self.calls["diagnostic_restore"]["returned"] == 6,
                "exact diagnostic work, no hidden initial or prefix forwards")
        write(self.out / "diagnostic-summary.json", {"fits": records, "seconds": self.diagnostic_seconds,
              "forwards": 240, "view_rows": 26784, "checkpoint_restores": 6,
              "action_permutations": permutations, "mse_weights": "weights_float32 upcast to float64, sum/N",
              "decision_weights": "weights_float64, sum/N", "selection": "eligible Python-float first within strict 1e-10",
              "scope": "All558 TRAIN rows, six fixed finals. NumPy f32 predictions and rounded targets reduced in f64; "
              "same mathematical objective, not bit-exact Torch training arithmetic. Regret uses the sampled R64 "
              "costs, not true action values. No gradients or selection. All diagnostic heads discarded before deployment."})

    def evaluation_setup(self, fits):
        tick = time.perf_counter()
        from openjev.research.otto_public import observation, seeded_environment
        from openjev.research.otto_reference_control import SpaceAwareActor
        self.observation, self.seeded, self.actor_class = observation, seeded_environment, SpaceAwareActor
        spec = importlib.util.spec_from_file_location("_training_budget_native", path(UPSTREAM))
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
              "scope": "Six new independent deployment file loads after diagnostics; no deployment-instance warmup. Shared modules are already imported."})

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
            features, targets = self.training_data()
            fits = self.train(features, targets)
            self.diagnostics(fits, targets)
            result = self.evaluate(fits)
            require(self.receipt["completed_diagnostics"] == 6 and self.receipt["completed_fits"] == 6
                    and self.receipt["completed_episodes"] == 504 and not self.pending
                    and not self.training_pending and self.pending_episode is None
                    and all(c["attempted"] == c["returned"] for c in self.calls.values()), "complete end-to-end work")
            require(self.calls["native_reset"]["returned"] == self.calls["actor_initialization"]["returned"] == 504
                    and self.calls["native_step"]["returned"] == self.calls["public_update"]["returned"]
                    <= 504 * HORIZON and self.calls["head_predict"]["returned"] <= 432 * HORIZON,
                    "bounded actual native and controller work")
            costs = {"authentication_seconds": authentication_seconds, "common_input_setup_seconds": self.setup_seconds,
                     "training_data_seconds": self.preparation_seconds, "train_diagnostic_seconds": self.diagnostic_seconds,
                     "model_module_seconds": self.model_module_seconds, "torch_setup_seconds": self.training_setup_seconds,
                     "training_pairs_seconds": math.fsum(p["seconds"] for p in self.pair_costs),
                     "native_setup_seconds": self.native_setup_seconds,
                     "head_restore_seconds": math.fsum(v["seconds"] for v in self.head_setup.values()),
                     "evaluation_seconds": self.evaluation_seconds}
            result.update(costs=costs, physical_stage_seconds=math.fsum(costs.values()),
                          measured_journal_io_seconds=self.io_seconds, fit_costs=self.fit_costs,
                          timing_scope="Stage totals are disjoint. The measured journal timer covers work/training/diagnostic "
                          "journal writes and evaluation transition writes, not every NPZ/gzip/serialization operation. "
                          "It overlaps stage totals and is not added again. "
                          "Actual controller intervals exclude only measured work-journal I/O; monitoring is outside them. "
                          "Both arms reuse paid historical R64 labels. Final TRAIN diagnostics are a separate cost, before fresh deployment loads. This is one CPU timing pass.")
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
            require({p.name for p in self.out.iterdir()} == payload_names(), "exact completed 34-payload closure")
            files = {p.name: digest(p, self.check) for p in self.out.iterdir() if p.is_file()}
            self.receipt.update(status="completed", pilot_continuation=result["pilot_continuation"],
                requires_successful_original_supervisor=True, calls=self.calls,
                training_calls=self.training_calls, pending=[],
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
                cleanup_errors=cleanup, calls=self.calls, training_calls=getattr(self, "training_calls", {}),
                pending=list(self.pending.values()), training_pending=list(getattr(self, "training_pending", {}).values()),
                pending_episode=self.pending_episode, context=self.context)
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
            model = means[f"long@{seed}"]
            criterion(competence, f"{regime}.{seed}.success", model["found"], .95, model["found"] >= .95)
            threshold = 1.05 * teacher["steps"]
            criterion(competence, f"{regime}.{seed}.moves", model["steps"], threshold, model["steps"] <= threshold)
        candidate, reference = families["long"], families["short"]
        criterion(relative, f"{regime}.success", candidate["found"], reference["found"], candidate["found"] >= reference["found"])
        criterion(relative, f"{regime}.moves", candidate["steps"], .95 * reference["steps"], candidate["steps"] <= .95 * reference["steps"])
        gains = [math.fsum(b[f"short@{s}"] - b[f"long@{s}"] for s in SEEDS) / 3 for b in blocks]
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
