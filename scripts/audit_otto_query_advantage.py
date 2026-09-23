"""Independent saved-only query-advantage audit and prespecified signal screen.

No producer, reducer, actor, policy or model is imported. NumPy is used only for
saved arrays and independent reconstruction of declared PCG64 categorical draws.
Analytic and neural scores remain inherited evidence, never regenerated here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_query_advantage.py"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-query-advantage-audit-v1"
LIMITS = {"seconds": 120, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
REGIMES, SCHEDULES, PREFIXES = ("base", "shift"), ("always", "never", "period2"), (0, 4, 8, 16, 32)
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 19400001, 2000
IDENTITY_KEYS = ("episode_id", "episode_index", "regime", "sensing_length", "case", "seed", "initial_hit", "schedule")
PAYLOADS = {name + ".jsonl" for name in ("work", "weights", "forwards", "gate-operations",
    "public-transitions", "native-truth", "episodes", "anchors", "sampler-events", "panels")} | {
    "started.json", "runtime.json", "setup.json", "anchors.npz", "costs.json", "summary.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def panel_statistics(records, analytic_action, neural_action, *, horizon=32, replicate_count=16):
    """Independent integer sufficient statistics; one physical same-action row."""
    require(integer(horizon, 1, 2188) and integer(replicate_count, 4, 10000)
            and replicate_count % 2 == 0, "bounded even panel")
    require(integer(analytic_action, 0, 3) and integer(neural_action, 0, 3), "endpoint action IDs")
    actions, panel = sorted({analytic_action, neural_action}), {}
    for row in records:
        require(type(row) is dict and set(row) == {"replicate_id", "first_action", "steps", "found"}, "record schema")
        rep, action, steps, found = (row[k] for k in ("replicate_id", "first_action", "steps", "found"))
        require(integer(rep, 0, replicate_count - 1) and integer(action, 0, 3) and action in actions
                and integer(steps, 1, horizon) and type(found) is bool and (found or steps == horizon), "complete capped record")
        require((rep, action) not in panel, "duplicate physical continuation")
        panel[rep, action] = row
    require(len(panel) == replicate_count * len(actions), "every declared physical continuation")
    differences = [panel[r, analytic_action]["steps"] - panel[r, neural_action]["steps"] for r in range(replicate_count)]
    n, half = replicate_count, replicate_count // 2
    total, squares = sum(differences), sum(d * d for d in differences)
    numerator = n * squares - total * total
    halves = [sum(differences[:half]), sum(differences[half:])]
    branches = {}
    for name, action in (("analytic", analytic_action), ("neural", neural_action)):
        selected = [panel[r, action] for r in range(n)]
        found = sum(r["found"] for r in selected)
        capped = sum(r["steps"] == horizon for r in selected)
        branches[name] = {"action": action, "mean_capped_moves": sum(r["steps"] for r in selected) / n,
            "found_count": found, "censored_count": n - found, "at_cap_count": capped,
            "found_fraction": found / n, "censored_fraction": (n - found) / n, "at_cap_fraction": capped / n}
    signs = [(v > 0) - (v < 0) for v in halves]
    return {"horizon": horizon, "replicate_ids": list(range(n)), "replicate_count": n,
        "analytic_action": analytic_action, "neural_action": neural_action,
        "physical_record_count": len(panel), "structural_zero": analytic_action == neural_action,
        "records": [panel[r, a].copy() for r in range(n) for a in actions], "differences": differences,
        "sum_difference": total, "sum_squared_difference": squares, "variance_numerator": numerator,
        "mean_advantage": total / n, "sample_variance": numerator / (n * (n - 1)),
        "variance_of_mean": numerator / (n * n * (n - 1)),
        "paired_standard_error": math.sqrt(numerator / (n * n * (n - 1))), "branches": branches,
        "positive_count": sum(d > 0 for d in differences), "negative_count": sum(d < 0 for d in differences),
        "tie_count": differences.count(0),
        "both_censored_count": sum(not panel[r, analytic_action]["found"] and not panel[r, neural_action]["found"] for r in range(n)),
        "halves": [{"replicate_ids": list(range(i * half, (i + 1) * half)), "sum_difference": v,
                    "mean_advantage": v / half, "sign": signs[i]} for i, v in enumerate(halves)],
        "split_same_sign": signs[0] == signs[1], "split_same_nonzero_sign": signs[0] == signs[1] != 0}


def weighted_statistics(rows, weights):
    """Rows contain independently reduced panels; subset weights are inherited."""
    require(len(rows) == len(weights), "aligned rows/weights")
    if not rows:
        return {"anchor_count": 0, "episode_count": 0, "original_weight_mass": 0., "statistics": None}
    require(all(math.isfinite(w) and w > 0 for w in weights), "positive finite original weights")
    mass = math.fsum(weights)
    weights = [w / mass for w in weights]
    panels = [r["reduction"] for r in rows]

    def mean(values):
        return math.fsum(w * x for w, x in zip(weights, values, strict=True))

    a = [p["halves"][0]["mean_advantage"] for p in panels]
    b = [p["halves"][1]["mean_advantage"] for p in panels]
    constant_a, constant_b = len(set(a)) == 1, len(set(b)) == 1
    ma, mb = a[0] if constant_a else mean(a), b[0] if constant_b else mean(b)
    va, vb = mean([(x - ma)**2 for x in a]), mean([(x - mb)**2 for x in b])
    covariance = mean([(x - ma) * (y - mb) for x, y in zip(a, b, strict=True)])
    second = mean([p["mean_advantage"]**2 for p in panels])
    noise = mean([p["variance_of_mean"] for p in panels])
    variance = (va + vb) / 2
    stats = {"mean_advantage": mean([p["mean_advantage"] for p in panels]), "second_moment": second,
        "estimated_mean_noise": noise, "untruncated_signal": second - noise,
        "uncentered_reliability": (second - noise) / second if second > 0 else None,
        "half_means": [ma, mb], "half_variances": [va, vb], "cross_half_second_moment": mean([x * y for x, y in zip(a, b, strict=True)]),
        "cross_half_covariance": covariance, "pooled_half_variance": variance,
        "repeatability": covariance / variance if variance > 0 else None,
        "half_correlation": covariance / math.sqrt(va * vb) if va > 0 and vb > 0 else None,
        "structural_zero_fraction": mean([int(p["structural_zero"]) for p in panels]),
        "zero_difference_fraction": mean([p["tie_count"] / p["replicate_count"] for p in panels]),
        "both_censored_fraction": mean([p["both_censored_count"] / p["replicate_count"] for p in panels]),
        "split_same_sign_fraction": mean([int(p["split_same_sign"]) for p in panels]),
        "split_same_nonzero_sign_fraction": mean([int(p["split_same_nonzero_sign"]) for p in panels]),
        "branch_rates": {role: {k: mean([p["branches"][role][k] for p in panels]) for k in
            ("found_fraction", "censored_fraction", "at_cap_fraction")} for role in ("analytic", "neural")}}
    return {"anchor_count": len(rows), "episode_count": len({r["episode_id"] for r in rows}),
            "original_weight_mass": mass, "statistics": stats}


def signal_statistics(rows, episode_ids):
    episodes = tuple(episode_ids)
    require(len(episodes) == len(set(episodes)) and episodes, "nonempty declared episode set")
    counts = collections.Counter(r["episode_id"] for r in rows)
    require(set(counts) == set(episodes), "all episodes contribute anchors")
    require(len({r["anchor_id"] for r in rows}) == len(rows), "unique anchors")
    ordered = sorted(rows, key=lambda r: (r["episode_id"], r["anchor_id"]))
    weights = [1 / (len(episodes) * counts[r["episode_id"]]) for r in ordered]
    selected = [(r, w) for r, w in zip(ordered, weights, strict=True) if not r["reduction"]["structural_zero"]]
    return {"episode_ids": sorted(episodes), "episode_count": len(episodes), "anchor_count": len(rows),
        "horizon": 32, "replicate_ids": list(range(16)),
        "weights": [{"episode_id": r["episode_id"], "anchor_id": r["anchor_id"], "weight": w}
                    for r, w in zip(ordered, weights, strict=True)],
        "all": weighted_statistics(ordered, weights),
        "different_action": weighted_statistics([r for r, _ in selected], [w for _, w in selected])}


def bootstrap_admission(rows, *, check=lambda: None, emit=lambda row: None, technical_complete=True):
    """Fixed case-cluster bootstrap: every resample retained, nearest rank 200."""
    by = {}
    results, checks = {}, []
    for regime in REGIMES:
        local = [r for r in rows if r["regime"] == regime]
        episode_ids = sorted({r["episode_id"] for r in local})
        require(len(episode_ids) == 36 and {r["case"] for r in local} == set(range(12)), "36 episodes/12 cases per setting")
        for case in range(12):
            entries = [r for r in local if r["case"] == case]
            require({r["schedule"] for r in entries} == set(SCHEDULES)
                    and len({r["episode_id"] for r in entries}) == 3, "all three schedules per originating case")
            by[regime, case] = entries
        results[regime] = signal_statistics(local, episode_ids)
    samples = {"all": [], "different_action": []}
    degenerate = {regime: {"empty_different": 0, "zero_variance_different": 0, "zero_variance_all": 0} for regime in REGIMES}
    rng = random.Random(BOOTSTRAP_SEED)
    for draw in range(BOOTSTRAP_DRAWS):
        check()
        sampled, covariances = {}, {}
        for regime in REGIMES:
            sampled[regime] = [rng.randrange(12) for _ in range(12)]
            expanded, weights = [], []
            for case in sampled[regime]:
                entries = by[regime, case]
                counts = collections.Counter(r["episode_id"] for r in entries)
                for r in entries:
                    expanded.append(r)
                    weights.append(1 / (12 * 3 * counts[r["episode_id"]]))
            covariances[regime] = {}
            for group in samples:
                pairs = [(r, w) for r, w in zip(expanded, weights, strict=True)
                         if group == "all" or not r["reduction"]["structural_zero"]]
                stats = weighted_statistics([r for r, _ in pairs], [w for _, w in pairs])["statistics"]
                if stats is None:
                    degenerate[regime]["empty_different"] += 1
                    value = 0.
                elif stats["pooled_half_variance"] == 0:
                    degenerate[regime]["zero_variance_" + ("all" if group == "all" else "different")] += 1
                    value = 0.
                else:
                    value = stats["cross_half_covariance"]
                covariances[regime][group] = value
        pooled = {group: math.fsum(covariances[r][group] for r in REGIMES) / 2 for group in samples}
        for group, value in pooled.items():
            samples[group].append(value)
        emit({"draw": draw, "sampled_cases": sampled, "covariances": covariances, "pooled_covariances": pooled})

    def condition(name, value, threshold, relation):
        checks.append({"name": name, "value": value, "threshold": threshold, "relation": relation,
                       "passes": value >= threshold if relation == ">=" else value > threshold})

    condition("technical_complete", int(technical_complete), 1, ">=")
    support = {}
    for regime in REGIMES:
        different = [r for r in rows if r["regime"] == regime and not r["reduction"]["structural_zero"]]
        support[regime] = {"different_anchors": len(different), "different_cases": len({r["case"] for r in different})}
        condition(regime + ".different_anchors", len(different), 12, ">=")
        condition(regime + ".different_cases", support[regime]["different_cases"], 6, ">=")
        for group in samples:
            stats = results[regime][group]["statistics"]
            condition(regime + ".covariance." + group, stats["cross_half_covariance"] if stats else 0., 0., ">")
    lower = {group: sorted(values)[199] for group, values in samples.items()}
    for group, value in lower.items():
        condition("pooled_lower." + group, value, 0., ">")
    require(len(checks) == 11, "exact eleven admission conditions")
    contributions = {}
    for regime in REGIMES:
        contributions[regime] = {}
        local = {r["anchor_id"]: r for r in rows if r["regime"] == regime}
        for group in samples:
            selected = [(local[w["anchor_id"]], w["weight"]) for w in results[regime]["weights"]
                        if group == "all" or not local[w["anchor_id"]]["reduction"]["structural_zero"]]
            mass = math.fsum(w for _, w in selected)
            stats = results[regime][group]["statistics"]
            means = [0., 0.] if stats is None else stats["half_means"]
            contributions[regime][group] = [{"case": case,
                "original_weight_mass": math.fsum(w for r, w in selected if r["case"] == case),
                "covariance_contribution": math.fsum(w / mass * (r["reduction"]["halves"][0]["mean_advantage"] - means[0])
                    * (r["reduction"]["halves"][1]["mean_advantage"] - means[1]) for r, w in selected if r["case"] == case)}
                for case in range(12)]
    return {"regimes": results, "support": support, "case_contributions": contributions, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_rng": "random.Random, base then shift, twelve randrange(12) calls each",
        "lower_bound_rule": "sorted pooled covariance values[199]; no interpolation or discarded draws",
        "lower_bounds": lower, "degenerate_resamples": degenerate, "conditions": checks,
        "conditions_passed": sum(c["passes"] for c in checks), "conditions_total": 11,
        "signal_admitted": all(c["passes"] for c in checks),
        "scope": "Descriptive fixed-pilot repeatability admission, not formal coverage, efficacy or architecture evidence."}


def cohort():
    rows = []
    for ri, regime in enumerate(REGIMES):
        for case in range(12):
            shift = (ri * 12 + case) % 3
            for schedule in SCHEDULES[shift:] + SCHEDULES[:shift]:
                seed = (19100001 if ri == 0 else 19200001) + case
                rows.append({"episode_id": f"train:{regime}:{seed}:{schedule}", "episode_index": len(rows),
                    "regime": regime, "sensing_length": float(ri + 3), "case": case, "seed": seed,
                    "initial_hit": 1 + case % 3, "schedule": schedule})
    return rows


def eligible(position):
    x, y = position
    return [a for a, (dx, dy) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1)))
            if 0 <= x + dx < 53 and 0 <= y + dy < 53]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


LIMITATIONS = [
    "Independent saved arithmetic and PCG64 categorical replay; no new policy, native, filter, model or optimizer calls.",
    "Original neural/analytic score correctness and posterior evolution remain inherited numerical evidence; saved actions, public continuity and captured posterior hashes are checked.",
    "Timing truth and historical qualification are inherited from authenticated sources, receipts and original process records.",
    "Case-cluster bootstrap is a descriptive small-pilot screen, not a formal confidence guarantee or control/architecture efficacy result.",
]


class Audit:
    def __init__(self, args):
        self.args, self.out, self.run = args, args.output, args.run
        self.clock = self.launch = self.start = None
        self.recording_failure = False
        self.counts = collections.Counter()
        self.bound = {}
        self.receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
            "model_calls": 0, "native_calls": 0, "sampler_calls": 0, "optimizer_calls": 0, "limitations": LIMITATIONS}

    def require(self, value, message):
        self.counts["checks"] += 1
        require(value, message)
        if self.launch and not self.recording_failure and self.counts["checks"] % 2048 == 0:
            self.check()

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original audit deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir())
                <= LIMITS["output_bytes"], "saved audit resource limits")

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        self.require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
                     and not any(q.is_symlink() for q in (p, *p.parents)), "contained regular input")
        return p

    def digest(self, value):
        p, h = self.path(value), hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
                if self.launch and not self.recording_failure:
                    self.check()
        return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}

    def bind(self, value, expected):
        p = self.path(value)
        actual = self.digest(p)
        self.require(actual == expected if isinstance(expected, dict) else actual["sha256"] == expected,
                     "external or frozen descriptor " + str(p))
        self.bound[str(p)] = actual
        return p

    def read(self, value):
        return json.loads(self.path(value).read_text())

    def rows(self, value, limit=1024 * 1024):
        with self.path(value).open() as stream:
            for line in stream:
                self.require(len(line.encode()) <= limit and line.endswith("\n"), "bounded complete JSON row")
                row = json.loads(line)
                self.require(type(row) is dict, "JSON object row")
                yield row

    def same(self, actual, expected, name):
        if isinstance(expected, dict):
            self.require(type(actual) is dict and set(actual) == set(expected), name + " keys")
            for key, value in expected.items():
                self.same(actual[key], value, name + "." + key)
        elif isinstance(expected, list):
            self.require(type(actual) is list and len(actual) == len(expected), name + " length")
            for a, b in zip(actual, expected, strict=True):
                self.same(a, b, name)
        elif isinstance(expected, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), name)
        else:
            self.require(type(actual) is type(expected) and actual == expected, name)

    def exhaust(self, rows):
        sentinel = object()
        self.require(next(rows, sentinel) is sentinel, "no omitted trailing records")

    def admit(self):
        self.bind(CLOCK, CLOCK_PIN)
        self.bind(SUPERVISOR, SUPERVISOR_PIN)
        spec = importlib.util.spec_from_file_location("_advantage_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original audit launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
            and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
            and Path.cwd() == ROOT == Path(self.launch["cwd"]) and self.launch["cap_seconds"] == 120
            and self.launch["clock_backend"] == self.clock.backend and self.launch["clock_source_sha256"] == CLOCK_PIN
            and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9, "original audit process")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(key) == "1", "single-thread numerical audit")
        for path, pin in ((self.args.plan, self.args.plan_sha256), (self.run / "receipt.json", self.args.receipt_sha256),
                          (self.args.terminal, self.args.terminal_sha256)):
            self.bind(path, pin)
        self.plan, self.worker, self.parent = self.read(self.args.plan), self.read(self.run / "receipt.json"), self.read(self.args.terminal)
        self.authenticate()
        self.receipt.update(plan_sha256=self.args.plan_sha256, worker_sha256=self.args.receipt_sha256,
                            terminal_sha256=self.args.terminal_sha256,
                            supervision_sha256=self.digest(self.args.supervision)["sha256"], sources=self.plan["sources"])
        self.bind(self.args.supervision, self.receipt["supervision_sha256"])
        write(self.out / "started.json", {"request": {k: str(v) for k, v in vars(self.args).items()},
                                         "launch": self.launch, "started_ns": self.start})

    def authenticate(self):
        p, w, t = self.plan, self.worker, self.parent
        self.require(p["version"] == w["version"] == "otto-query-advantage-study-v1"
            and p["status"] == "frozen_before_collection" and w["status"] == "completed" and w["complete"] is True
            and w["requires_successful_original_supervisor"] is True and w["pending"] == []
            and w["pending_publications"] == [] and w["sampler_pending"] == []
            and w["completed_episodes"] == 72 and w["training_updates"] == w["evaluation_episodes"] == 0
            and w["plan_sha256"] == self.args.plan_sha256 and p["cohort"] == cohort(), "fixed complete TRAIN-only pilot")
        c = p["configuration"]
        self.require(c["schedules"] == list(SCHEDULES) and c["prefixes"] == list(PREFIXES)
            and c["label_seed"] == 19300001 and c["label_horizon"] == 32 and c["replicate_ids"] == list(range(16))
            and c["episodes"] == 72 and c["horizon"] == 2188 and c["cases_per_regime"] == 12, "fixed label/schedule allocation")
        self.require(w["sources"] == p["sources"] and w["inputs"] == p["inputs"]
            and w["native_inputs"] == p["native_inputs"] and w["limits"] == p["limits"]
            and SELF in p["sources"] and "tests/test_audit_otto_query_advantage.py" in p["sources"], "exact frozen provenance")
        for name, pin in p["sources"].items():
            self.bind(name, pin)
        for records in (p["inputs"], p["native_inputs"]):
            for desc in records.values():
                self.bind(desc["path"], {k: desc[k] for k in ("sha256", "bytes")})
        self.require(set(w["files"]) == set(p["payloads"]) == PAYLOADS
            and {f.name for f in self.run.iterdir()} == PAYLOADS | {"receipt.json"}, "exact sixteen-payload closure")
        for name, desc in w["files"].items():
            self.bind(self.run / name, desc)
        receipt_bytes = self.bound[str(self.run / "receipt.json")]["bytes"]
        self.require(integer(w["peak_rss_bytes"], 1, 4 * 1024**3)
            and sum(d["bytes"] for d in w["files"].values()) + receipt_bytes <= 6 * 1024**3
            and sum(d["bytes"] for n, d in w["files"].items() if n != "sampler-events.jsonl")
                + receipt_bytes <= 2 * 1024**3, "closed worker RSS/total/non-sampler bounds")
        command = t["command"]
        normalized_command = list(command)
        if normalized_command[1:2] == ["-u"]:
            normalized_command.pop(1)
        self.require(normalized_command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"),
            str(ROOT / "scripts/study_otto_query_advantage.py"), "run"] and len(normalized_command) == 11
            and set(normalized_command[3::2]) == {"--plan", "--plan-sha256", "--output", "--supervision"},
            "exact original interpreter/script/run invocation")
        launch_path = Path(command[command.index("--supervision") + 1])
        self.bind(launch_path, w["supervision_sha256"])
        launch = self.read(launch_path)
        self.require(all(t[k] == v for k, v in launch.items()) and t["status"] == "completed" and t["returncode"] == 0
            and t["timed_out"] is False and t["error"] is t["clock_error"] is None
            and t["group_absent"] is t["cleanup"]["group_absent"] is t["cleanup"]["reaped"] is True
            and t["cleanup"]["errors"] == [] and t["pid"] != t["parent_pid"] and t["pid"] == t["pgid"]
            and t["cap_seconds"] == 900 and t["clock_source_sha256"] == CLOCK_PIN and t["watchdog_sha256"] == SUPERVISOR_PIN
            and t["deadline_ns"] == t["started_ns"] + 900 * 10**9
            and t["started_ns"] <= w["started_ns"] < w["finished_ns"] <= t["finished_ns"] <= t["deadline_ns"], "successful original pilot parent")
        self.require(Path(command[command.index("--plan") + 1]) == self.args.plan
            and command[command.index("--plan-sha256") + 1] == self.args.plan_sha256
            and Path(command[command.index("--output") + 1]) == self.run
            and Path(t["cwd"]) == ROOT, "original invocation joins")
        self.same(w["wall_seconds"], (w["finished_ns"] - w["started_ns"]) / 1e9, "worker time")
        self.same(t["wall_seconds"], (t["finished_ns"] - t["started_ns"]) / 1e9, "parent time")
        started = self.read(self.run / "started.json")
        self.require(started["launch"] == launch and started["request"]["plan_sha256"] == self.args.plan_sha256,
                     "embedded original launch")
        self.summary = self.read(self.run / "summary.json")
        self.require(self.summary["complete"] is True and self.summary["configuration"] == c
            and self.summary["training_updates"] == self.summary["evaluation_episodes"] == 0, "saved complete scope")
        self.counts["source_files"], self.counts["producer_payloads"] = len(p["sources"]), 16

    def packet(self, public):
        self.require(set(public) == {"position", "step", "hit", "done", "valid_actions"}
            and type(public["position"]) is list and len(public["position"]) == 2
            and all(integer(x, 0, 52) for x in public["position"]) and integer(public["step"], 0, 2220)
            and type(public["done"]) is bool, "public packet geometry")
        self.require(public["valid_actions"] == ([] if public["done"] else eligible(public["position"]))
            and (public["hit"] == -2 if public["done"] else integer(public["hit"], 0, 3)), "public terminal/eligibility semantics")

    def move(self, position, action):
        self.require(action in eligible(position), "eligible actual action")
        dx, dy = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        return [position[0] + dx, position[1] + dy]

    def anchors_and_paths(self):
        import numpy as np

        self.np = np
        self.episodes = list(self.rows(self.run / "episodes.jsonl"))
        self.anchors = list(self.rows(self.run / "anchors.jsonl"))
        self.require(len(self.episodes) == 72 and len(self.anchors) == 360
            and [a["anchor_id"] for a in self.anchors] == list(range(360)), "all episode/anchor slots in fixed order")
        self.available = [a for a in self.anchors if a["status"] == "available"]
        with np.load(self.run / "anchors.npz", allow_pickle=False) as archive:
            self.require(set(archive.files) == {"beliefs", "features", "anchor_ids"}, "exact public anchor array schema")
            self.arrays = {k: archive[k] for k in archive.files}
        for name, shape, dtype in (("beliefs", (len(self.available), 53, 53), np.float64),
            ("features", (len(self.available), 31), np.float32), ("anchor_ids", (len(self.available),), np.int64)):
            a = self.arrays[name]
            self.require(a.shape == shape and a.dtype == dtype and np.isfinite(a).all(), "public array " + name)
        self.kernels = {}
        for regime in REGIMES:
            with np.load(self.path(self.plan["native_inputs"][regime + "_kernel"]["path"]), allow_pickle=False) as archive:
                self.require(set(archive.files) == {"likelihood", "initial_hit_weights"}, "qualified kernel keys")
                k = archive["likelihood"]
            self.require(k.dtype == np.float64 and k.shape == (4, 107, 107) and np.isfinite(k).all()
                         and (k >= 0).all() and (k[:, 53, 53] == 0).all(), "immutable public likelihood kernel")
            self.kernels[regime] = k
        paths, truth = self.rows(self.run / "public-transitions.jsonl"), self.rows(self.run / "native-truth.jsonl")
        total = queries = annotations = 0
        paired = {}
        available_index = 0
        for identity, episode in zip(cohort(), self.episodes, strict=True):
            self.same({k: episode[k] for k in identity}, identity, "episode identity")
            self.require(integer(episode["rows"], 1, 2188) and episode["start_row"] == total
                and episode["end_row"] == total + episode["rows"] and episode["censored"] is (not episode["found"])
                and (episode["found"] or episode["rows"] == 2188), "complete uncropped episode")
            reset, hidden = next(paths), next(truth)
            self.require(reset["kind"] == hidden["kind"] == "reset", "one reset")
            for value in (reset, hidden):
                self.same({k: value[k] for k in identity}, identity, "stream episode identity")
            current, posterior = reset["public"], reset["posterior"]
            self.packet(current)
            self.require(current["step"] == 0 and not current["done"] and current["hit"] == identity["initial_hit"]
                         and posterior["exact"] is True, "public initial packet")
            source = hidden["source_evaluation_only"]
            pairkey = (identity["regime"], identity["case"])
            pair = paired.setdefault(pairkey, {"source": source, "draws": {}})
            self.require(pair["source"] == source, "schedule-paired native source")
            self.require(len(hidden["draws"]) == 1 and hidden["draws"][0]["channel"] == "source"
                and hidden["draws"][0]["index"] == 0
                and list(divmod(hidden["draws"][0]["selected_index"], 53)) == source, "actual initial source witness")
            self.native_draws(hidden["draws"], pair)
            last_query, qc, ec, ages = None, 0, 0, collections.Counter()
            for step in range(episode["rows"]):
                row, hidden = next(paths), next(truth)
                for value in (row, hidden):
                    self.same({k: value[k] for k in identity}, identity, "step identity")
                    self.require(value["kind"] == "step" and value["step"] == step + 1, "chronological actual step")
                self.require(row["public_before"] == current and row["row_index"] == total, "public continuity/global row")
                queried = identity["schedule"] == "always" or identity["schedule"] == "period2" and step % 2 == 0
                is_anchor, age = step in PREFIXES, step if last_query is None else step - last_query
                features = np.asarray(row["features"], dtype=np.float32)
                self.require(features.shape == (31,) and np.isfinite(features).all()
                    and float(features[16]) == float(np.float32(age / 2188))
                    and float(features[17]) == float(last_query is not None)
                    and row["scheduled_query"] is queried and row["anchor"] is is_anchor
                    and row["external_annotation"] is (is_anchor and not queried)
                    and row["query_age"] == age and row["state_after"] == [step + 1.], "actual schedule and unaltered query age")
                self.require(row["analytic_action"] in current["valid_actions"], "analytic eligible endpoint")
                if not queried:
                    self.require(row["action"] == row["analytic_action"], "unqueried analytic action")
                if is_anchor:
                    anchor = self.anchors[identity["episode_index"] * 5 + PREFIXES.index(step)]
                    self.anchor(anchor, identity, current, posterior, features, available_index)
                    self.require(row["analytic_action"] == anchor["analytic_action"]
                        and (not queried or row["action"] == anchor["neural_action"]), "anchor endpoint/action join")
                    available_index += 1
                public = row["public"]
                self.packet(public)
                self.require(public["step"] == step + 1 and public["position"] == self.move(current["position"], row["action"])
                    and public["done"] is (public["position"] == source) and row["posterior"]["exact"] is True
                    and (not public["done"] or step + 1 == episode["rows"]), "native saved outcome/public continuity")
                self.require(len(hidden["draws"]) == int(not public["done"]), "found skips odor")
                self.native_draws(hidden["draws"], pair)
                if hidden["draws"]:
                    self.require(hidden["draws"][0]["channel"] == "hit"
                        and hidden["draws"][0]["index"] == step
                        and hidden["draws"][0]["selected_index"] == public["hit"], "saved native hit")
                    moved = public["position"]
                    probs = self.kernels[identity["regime"]][:, 53 + source[0] - moved[0], 53 + source[1] - moved[1]]
                    self.require(hidden["draws"][0]["probabilities"] == probs.tolist(), "native source-conditioned immutable kernel")
                current, posterior = public, row["posterior"]
                ages[age] += 1
                if queried:
                    last_query, qc = step, qc + 1
                ec += is_anchor and not queried
                total += 1
            self.require(current["done"] is episode["found"] and qc == episode["query_count"]
                and ec == episode["external_annotation_calls"] and qc + ec == episode["neural_calls"]
                and episode["available_anchor_steps"] == [p for p in PREFIXES if p < episode["rows"]]
                and episode["maximum_query_age"] == max(ages)
                and episode["query_age_counts"] == {str(k): v for k, v in ages.items()}, "episode accounting")
            for slot, prefix in enumerate(PREFIXES):
                anchor = self.anchors[identity["episode_index"] * 5 + slot]
                if prefix >= episode["rows"]:
                    self.require(anchor == {**identity, "anchor_id": identity["episode_index"] * 5 + slot,
                        "prefix_index": slot, "preaction_step": prefix, "status": "unavailable_found",
                        "found_after_moves": episode["rows"]}, "missing anchor explicitly after discovery")
            queries, annotations = queries + qc, annotations + ec
        self.exhaust(paths); self.exhaust(truth)
        for key, value in {"trajectory_rows": total, "queries": queries, "external_annotations": annotations,
                           "available_anchors": available_index}.items():
            self.require(self.worker[key] == self.summary[key] == value, "complete accounting " + key)
        self.require(self.summary["episodes"] == self.episodes and self.summary["anchor_slots"] == 360
            and self.summary["unavailable_anchors"] == 360 - available_index
            and self.summary["anchors_file"] == self.worker["files"]["anchors.npz"], "saved summary joins")
        self.counts.update(episodes=72, anchors=available_index, unavailable_anchors=360-available_index, native_saved_steps=total)

    def native_draws(self, draws, pair):
        for draw in draws:
            key, uniform = (draw["channel"], draw["index"]), draw["uniform"]
            self.require(draw["channel"] in ("source", "hit") and integer(draw["index"], 0, 2188)
                         and 0 <= uniform < 1, "original draw identity")
            if key in pair["draws"]:
                self.require(pair["draws"][key] == uniform, "native schedules retain shared uniform prefix")
            pair["draws"][key] = uniform
            index, mass = self.categorical(draw["probabilities"], uniform)
            self.require(draw["selected_index"] == index and draw["cdf_mass"] == mass, "saved native categorical witness")
            self.counts["native_saved_categorical_checks"] += 1

    def anchor(self, anchor, identity, public, posterior, features, index):
        np = self.np
        slot = PREFIXES.index(public["step"])
        for key, value in {**identity, "anchor_id": identity["episode_index"] * 5 + slot, "prefix_index": slot,
            "preaction_step": public["step"], "status": "available", "array_index": index, "public": public}.items():
            self.require(anchor[key] == value, "selected public anchor " + key)
        b, f = self.arrays["beliefs"][index], self.arrays["features"][index]
        self.require(self.arrays["anchor_ids"][index] == anchor["anchor_id"] and (b >= 0).all()
            and b[tuple(public["position"])] == 0 and abs(float(b.sum()) - 1.) <= 1e-10
            and anchor["belief_mass"] == float(b.sum()) == posterior["mass"]
            and hashlib.sha256(b.tobytes()).hexdigest() == anchor["belief_sha256"] == posterior["sha256"]
            and f.tobytes() == features.tobytes() and anchor["features"] == features.tolist(), "captured immutable public state")
        allowed = public["valid_actions"]
        analytic = anchor["analytic_costs"]
        self.require(len(analytic) == 4 and all(analytic[a] is None for a in range(4) if a not in allowed)
            and all(type(analytic[a]) in (int, float) and math.isfinite(analytic[a]) for a in allowed), "analytic score support")
        minimum = min(analytic[a] for a in allowed)
        self.require(anchor["analytic_action"] == next(a for a in allowed if abs(analytic[a] - minimum) < 1e-10),
                     "analytic eligible saved-score selection")
        costs = np.asarray(anchor["neural_costs"], dtype=np.float32)
        self.require(costs.shape == (4,) and np.isfinite(costs).all() and costs.tolist() == anchor["neural_costs"], "exact saved f32 neural costs")
        minimum = costs[allowed].min()
        self.require(anchor["neural_action"] == next(a for a in allowed if abs(costs[a] - minimum) < 1e-10),
                     "original float32 near-tie endpoint")

    def categorical(self, probabilities, uniform):
        np = self.np
        p = np.asarray(probabilities, dtype=np.float64)
        self.require(p.ndim == 1 and len(p) and np.isfinite(p).all() and (p >= 0).all()
            and abs(float(np.sum(p, dtype=np.float64)) - 1.) <= 1e-10 and 0 <= uniform < 1, "saved categorical support")
        cdf = np.cumsum(p, dtype=np.float64)
        mass = float(cdf[-1])
        self.require(mass > 0, "positive CDF mass")
        cdf /= mass
        index = int(np.searchsorted(cdf, uniform, side="right"))
        self.require(index < len(p) and p[index] > 0, "categorical support selected")
        return index, mass

    def sampler(self):
        np, stream = self.np, self.rows(self.run / "sampler-events.jsonl", 1024)
        panels = self.rows(self.run / "panels.jsonl")
        calls, events, all_rows = collections.Counter(), 0, []
        for anchor in self.available:
            base = {"version": "otto-teacher-rollouts-v1", "seed": 19300001, "anchor_id": anchor["anchor_id"]}
            operation_id = 0

            def event(kind, fields, base=base):
                nonlocal events
                row = next(stream)
                events += 1
                self.require({k: row[k] for k in (*base, "event")} == {**base, "event": kind}, "sampler panel identity")
                for k, v in fields.items():
                    self.require(row[k] == v and type(row[k]) is type(v), "sampler context " + k)
                self.require(set(row) == set(base) | {"event"} | set(fields), "complete sampler event schema")
                return row

            def call(operation, context, result_keys=(), base=base):
                nonlocal operation_id, events
                operation_id += 1
                fields = {**context, "operation_id": operation_id, "operation": operation}
                event("attempt", fields)
                row = next(stream)
                events += 1
                self.require(set(row) == set(base) | {"event"} | set(fields) | set(result_keys)
                    and {k: row[k] for k in base} == base and row["event"] == "return"
                    and {k: row[k] for k in fields} == fields, "complete sampler operation return")
                calls[operation] += 1
                return row

            self.require(call("anchor_snapshot", {}, ("public",))["public"] == anchor["public"], "validation anchor snapshot")
            actions = sorted({anchor["analytic_action"], anchor["neural_action"]})
            event("pair_declaration", {"adapter_version": "otto-query-pair-rollouts-v1",
                "analytic_action": anchor["analytic_action"], "neural_action": anchor["neural_action"],
                "distinct_actions": actions, "replicate_ids": list(range(16)), "horizon": 32})
            belief = self.arrays["beliefs"][anchor["array_index"]]
            kernel = self.kernels[anchor["regime"]]
            records = []
            for rep in range(16):
                repctx = {"replicate_id": rep}
                call("source_generator", repctx)
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F54544F, 19300001, anchor["anchor_id"], rep, 0])))
                uniform = float(rng.random())
                index, mass = self.categorical(belief.reshape(-1), uniform)
                source = list(divmod(index, 53))
                row = call("source_draw", repctx, ("uniform", "cdf_mass", "selected_index", "source"))
                self.require({k: row[k] for k in ("uniform", "cdf_mass", "selected_index", "source")} == {
                    "uniform": uniform, "cdf_mass": mass, "selected_index": index, "source": source}, "independent paired source draw")
                self.counts["replayed_source_draws"] += 1
                for first in actions:
                    context = {**repctx, "first_action": first}
                    self.require(call("teacher_snapshot", context, ("public",))["public"] == anchor["public"], "fresh identical branch snapshot")
                    call("hit_generator", context)
                    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F54544F, 19300001, anchor["anchor_id"], rep, 1])))
                    position = anchor["public"]["position"]
                    for step in range(1, 33):
                        ctx = {**context, "local_step": step, "from_step": anchor["preaction_step"] + step - 1}
                        if step == 1:
                            action = first
                            event("forced_action", {**ctx, "action": first})
                        else:
                            row = call("teacher_choose", ctx, ("action", "scores"))
                            scores, allowed = row["scores"], eligible(position)
                            self.require(len(scores) == 4 and all(scores[a] is None for a in range(4) if a not in allowed)
                                and all(type(scores[a]) in (int, float) and math.isfinite(scores[a]) for a in allowed), "saved analytic teacher scores")
                            minimum = min(scores[a] for a in allowed)
                            action = next(a for a in allowed if abs(scores[a] - minimum) < 1e-10)
                            self.require(row["action"] == action, "first eligible teacher near-minimum")
                        moved = self.move(position, action)
                        found = moved == source
                        row = call("movement", ctx, ("position", "found"))
                        self.require(row["position"] == moved and row["found"] is found, "private source-conditioned transport")
                        hit = -2
                        if not found:
                            probs = kernel[:, 53 + source[0] - moved[0], 53 + source[1] - moved[1]]
                            uniform = float(rng.random())
                            hit, mass = self.categorical(probs, uniform)
                            row = call("hit_draw", ctx, ("uniform", "cdf_mass", "selected_index", "probabilities", "draw_index"))
                            self.require(row["probabilities"] == probs.tolist() and row["uniform"] == uniform
                                and row["cdf_mass"] == mass and row["selected_index"] == hit and row["draw_index"] == step - 1,
                                "shared replica hit stream with branch-specific source-conditioned probabilities")
                            self.counts["replayed_hit_draws"] += 1
                        public = {"position": moved, "hit": hit, "done": found,
                            "step": anchor["preaction_step"] + step, "valid_actions": [] if found else eligible(moved)}
                        self.require(call("teacher_update", ctx, ("public",))["public"] == public, "every final/censored public update")
                        position = moved
                        if found or step == 32:
                            record = {**context, "steps": step, "found": found}
                            event("record", record)
                            records.append(record)
                            break
            event("panel_complete", {"record_count": len(records), "operation_count": operation_id,
                                      "adapter_version": "otto-query-pair-rollouts-v1"})
            expected = panel_statistics(records, anchor["analytic_action"], anchor["neural_action"])
            panel = next(panels)
            self.require(set(panel) == {"episode_id", "anchor_id", "reduction"}
                and panel["episode_id"] == anchor["episode_id"] and panel["anchor_id"] == anchor["anchor_id"], "complete panel join")
            self.same({k: panel["reduction"][k] for k in expected}, expected, "independent integer panel reduction")
            all_rows.append({k: anchor[k] for k in ("episode_id", "anchor_id", "regime", "case", "schedule", "preaction_step")}
                            | {"reduction": expected})
        self.exhaust(stream); self.exhaust(panels)
        self.require(self.worker["panels"] == len(all_rows) and self.worker["sampler_events"] == events <= 3019680,
                     "all physical panels/events")
        for channel in self.plan["sampler_caps"]:
            n = calls[channel]
            self.require(n <= self.plan["sampler_caps"][channel]
                and self.worker["sampler_calls"][channel] == {"attempted": n, "returned": n}, "sampler operation accounting")
        records = sum(r["reduction"]["physical_record_count"] for r in all_rows)
        moves = sum(p["steps"] for r in all_rows for p in r["reduction"]["records"])
        found = sum(p["found"] for r in all_rows for p in r["reduction"]["records"])
        self.require(records == self.worker["sampler_records"] and events == 68 * len(all_rows) + 4 * records + 8 * moves - 2 * found,
                     "complete exact sampler serialization count")
        for group in ("all", *REGIMES):
            selected = [r for r in all_rows if group == "all" or r["regime"] == group]
            ids = [r["episode_id"] for r in self.episodes if group == "all" or r["regime"] == group]
            expected = signal_statistics(selected, ids)
            saved = self.summary["signals"][group]
            self.same({k: saved[k] for k in expected}, expected, "independent saved signal reduction " + group)
        self.counts.update(sampler_events=events, physical_continuations=records, continuation_moves=moves,
                           continuation_found=found, sampler_operations=sum(calls.values()))
        return all_rows

    def journals(self):
        calls = {k: {"attempted": 0, "returned": 0, "seconds": 0.} for k in self.plan["call_caps"]}
        stack, sequence, root_sequence = [], 0, []
        episode_seconds = {e["episode_id"]: {k: [] for k in calls} for e in self.episodes}
        byid = {e["episode_id"]: e for e in self.episodes}
        for row in self.rows(self.run / "work.jsonl"):
            channel, context = row["channel"], row["context"]
            self.require(channel in calls, "known work channel")
            identity = {k: row[k] for k in ("call_id", "channel", "context", "parent_call_id")}
            if row["event"] == "attempt":
                sequence += 1
                self.require(set(row) == set(identity) | {"event"} and row["call_id"] == sequence
                    and row["parent_call_id"] == (stack[-1]["call_id"] if stack else None), "nested work attempt")
                if not stack:
                    # Keep only compact root chronology; fewer than 0.5 million short tuples.
                    root_sequence.append((channel, context.get("episode_id"), context.get("step"), context.get("anchor_id")))
                else:
                    self.require(channel == "tensorflow_value" and stack[-1]["channel"] in
                                 ("actor_choose", "annotation_score"), "single nested neural readout")
                stack.append(identity)
                calls[channel]["attempted"] += 1
            else:
                self.require(row["event"] == "return" and stack and stack.pop() == identity
                    and set(row) == set(identity) | {"event", "seconds", "instrumented_seconds", "excluded_io_seconds"}
                    and all(math.isfinite(row[k]) and row[k] >= 0 for k in ("seconds", "instrumented_seconds", "excluded_io_seconds"))
                    and row["seconds"] == row["instrumented_seconds"] - row["excluded_io_seconds"], "durable work return/timing")
                calls[channel]["returned"] += 1
                calls[channel]["seconds"] += row["seconds"]
                if context.get("episode_id") in episode_seconds and context["phase"] != "paired_labels":
                    episode_seconds[context["episode_id"]][channel].append(row["seconds"])
            if "episode_id" in context:
                e = byid[context["episode_id"]]
                if context["phase"] != "paired_labels":
                    self.require(all(context[k] == e[k] for k in IDENTITY_KEYS), "work episode identity")
                    if "step" in context:
                        self.require(integer(context["step"], 0, e["rows"] - 1), "work decision step")
        self.require(not stack, "all work returned")
        self.same(calls, self.worker["calls"], "complete work counts/time")
        for channel, count in calls.items():
            self.require(count["attempted"] == count["returned"] <= self.plan["call_caps"][channel], "work cap")
        expected_roots = [(k, None, None, None) for k in ("tensorflow_construction", "tensorflow_build", "tensorflow_load")]
        for e in self.episodes:
            eid = e["episode_id"]
            expected_roots.extend((k, eid, None, None) for k in ("native_reset", "actor_construction", "backend_binding"))
            for step in range(e["rows"]):
                query = e["schedule"] == "always" or e["schedule"] == "period2" and step % 2 == 0
                expected_roots.append(("actor_choose", eid, step, None))
                if step in PREFIXES:
                    if not query:
                        expected_roots.append(("annotation_score", eid, step, None))
                    expected_roots.append(("anchor_validate", eid, step, None))
                expected_roots.extend((k, eid, step, None) for k in ("native_step", "actor_update"))
            for channel in calls:
                self.same(e["operation_seconds"][channel], math.fsum(episode_seconds[eid][channel]), "episode scoped work time")
        expected_roots.append(("anchor_save", None, None, None))
        for a in self.available:
            expected_roots.extend((k, a["episode_id"], None, a["anchor_id"]) for k in ("pair_panel", "panel_reduce"))
        # Producer deliberately retains the final anchor context for the three aggregate calls.
        last = self.available[-1]
        expected_roots.extend(("signal_reduce", last["episode_id"], None, last["anchor_id"]) for _ in range(3))
        self.require(root_sequence == expected_roots, "all72 paths/validations/save precede any paired labels")
        self.gate_journal()
        forward_counts = collections.Counter()
        for ordinal, row in enumerate(self.rows(self.run / "forwards.jsonl"), 1):
            ctx = row["context"]
            e = byid[ctx["episode_id"]]
            self.require(row["ordinal"] == ordinal and row["input_shape"] == [16, 105, 105]
                and row["symmetry_average"] is True and len(row["values"]) == 16
                and all(math.isfinite(x) for x in row["values"]) and ctx["phase"] == "decision"
                and integer(ctx["step"], 0, e["rows"] - 1), "original counted neural branch values")
            query = e["schedule"] == "always" or e["schedule"] == "period2" and ctx["step"] % 2 == 0
            self.require(query or ctx["step"] in PREFIXES, "only deployed or anchor-annotation readouts")
            forward_counts[e["episode_id"], ctx["step"]] += 1
            self.require(forward_counts[e["episode_id"], ctx["step"]] == 1, "no repeated hidden neural annotation")
        self.require(sum(forward_counts.values()) == self.worker["queries"] + self.worker["external_annotations"]
            == calls["tensorflow_value"]["returned"], "all neural calls paid once")
        weights = list(self.rows(self.run / "weights.jsonl"))
        self.require(weights and len({w["id"] for w in weights}) == len(weights)
            and all(w["exact"] is True and len(w["sha256"]) == 64 for w in weights), "saved exact original model restore witnesses")
        costs = self.read(self.run / "costs.json")
        self.require(costs == self.summary["costs"] and costs["operation_seconds"] == self.worker["calls"]
            and costs["sampler_counts"] == self.worker["sampler_calls"], "disjoint phase/nested operation cost joins")
        for key in ("setup_wall_seconds", "collection_wall_seconds", "paired_labels_wall_seconds", "anchor_save_seconds", "journal_io_seconds"):
            self.require(type(costs[key]) in (int, float) and math.isfinite(costs[key]) and costs[key] >= 0, "nonnegative paid cost")
        self.require(sum(costs[k] for k in ("setup_wall_seconds", "collection_wall_seconds", "paired_labels_wall_seconds", "anchor_save_seconds"))
            <= self.worker["wall_seconds"] + 1e-6, "physical phases fit worker wall without double charging nested operations")
        self.counts["work_operations"] = sequence
        return costs

    def gate_journal(self):
        stream = self.rows(self.run / "gate-operations.jsonl")
        channels = ("view_initialization", "public_reset", "analytic_score", "feature_build", "gate", "neural_score", "public_update")
        for e in self.episodes:
            counts = {k: {"attempted": 0, "returned": 0} for k in channels}
            expected = [("view_initialization", None, -1), ("public_reset", 0, 0)]
            for step in range(e["rows"]):
                expected.extend((k, step, 0) for k in ("analytic_score", "feature_build", "gate"))
                query = e["schedule"] == "always" or e["schedule"] == "period2" and step % 2 == 0
                if query:
                    expected.append(("neural_score", step, 0))
                expected.append(("public_update", step, 0))
            for operation, (channel, step, episode) in enumerate(expected, 1):
                for event in ("attempt", "return"):
                    row = next(stream)
                    wanted = {k: e[k] for k in IDENTITY_KEYS} | {"id": operation, "channel": channel, "episode": episode, "step": step, "event": event}
                    self.require(row == wanted, "every gate/filter operation including final update")
                counts[channel]["attempted"] += 1
                counts[channel]["returned"] += 1
            self.require(e["gate_progress"] == {"episode": 0, "failed": False, "pending_action": None,
                "calls": counts, "pending_operations": [], "errors": []}, "closed gate lifecycle")
        self.exhaust(stream)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("saved query-advantage audit terminated")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            self.anchors_and_paths()
            rows = self.sampler()
            costs = self.journals()
            with (self.out / "anchors.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            with (self.out / "bootstrap.jsonl").open("x") as stream:
                result = bootstrap_admission(rows, check=self.check, emit=lambda row: stream.write(
                    json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"))
                stream.flush(); os.fsync(stream.fileno())
            result.update(costs=costs, counts=dict(self.counts), worker_seconds=self.worker["wall_seconds"],
                          parent_seconds=self.parent["wall_seconds"])
            write(self.out / "audit.json", {"version": VERSION, "agreement": True, "summary": result, "limitations": LIMITATIONS})
            for path, desc in self.bound.items():
                self.require(self.digest(path) == desc, "unchanged authenticated input/source")
            names = {"started.json", "anchors.jsonl", "bootstrap.jsonl", "audit.json"}
            self.require({p.name for p in self.out.iterdir()} == names, "exclusive completed audit inventory")
            files = {name: self.digest(self.out / name) for name in sorted(names)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", agreement=True, counts=dict(self.counts), files=files,
                signal_admitted=result["signal_admitted"], conditions_passed=result["conditions_passed"], conditions_total=11,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.recording_failure = True
            self.receipt.update(status="failed", agreement=False, counts=dict(self.counts),
                                error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.digest(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "supervision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan", "receipt", "terminal"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    if not all(getattr(args, k).is_absolute() for k in ("plan", "run", "terminal", "supervision", "output")):
        parser.error("absolute paths required")
    Audit(args).execute()


if __name__ == "__main__":
    main()
