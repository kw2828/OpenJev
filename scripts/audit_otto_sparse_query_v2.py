"""Bounded saved-record arithmetic and journal audit; never scientific replay.

No evaluator, model, sampler, optimizer, Torch, NumPy or native simulator imports.
The only imported project code is the pinned nonscientific suspend clock.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import struct
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_sparse_query_v2.py"
TEST = "tests/test_audit_otto_sparse_query_v2.py"
ORIGINAL = "scripts/audit_otto_sparse_query.py"
ORIGINAL_PIN = "e73a94a8927fea330c65f0b006ade1d82f79693e21336c91bf5acd742b1d7925"
REPAIR_PROTOCOL = "research/otto-sparse-query-audit-repair.md"
REPAIR_PROTOCOL_PIN = "24d8f189680052887e5bea3fb3d3c0afc3849b029c446f5052fafc9b413d78fb"
REPAIR_VERSION = "otto-sparse-query-audit-repair-v2"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-sparse-query-saved-audit-v2"
LIMITS = {"seconds": 120, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
PRODUCER = "scripts/study_otto_sparse_query.py"
PRODUCER_VERSION = "otto-sparse-query-study-v1"
ARMS = ("analytic", "neural", "period2", "random_pair", "entropy")
SPARSE = ARMS[2:]
VALID_FIRST = {"lambda3": 20100001, "lambda4": 20200001}
FIRST = {"lambda3": 20300001, "lambda4": 20400001, "lambda5": 20500001}
METRICS = ("found", "steps", "queries", "init_seconds", "choose_seconds", "update_seconds",
           "setup_allocation_seconds", "controller_seconds", "paid_controller_seconds", "environment_seconds")
PAYLOADS = {f"{name}.jsonl.gz" for name in ("work", "weights", "forwards", "gate-operations", "gate-decisions",
    "transitions", "validation-decisions")} | {"started.json", "runtime.json", "setup.json", "deployment.json",
    "episode-boundaries.jsonl", "episodes.jsonl", "threshold.json", "costs.json", "summary.json"}
LIMITATIONS = [
    "Original neural values, entropy extraction and native/public posterior equality remain inherited numerical evidence; no model/filter/array replay.",
    "Saved endpoint decisions, sparse schedules, calibration median, paired witnesses, reductions and cost accounting are independently checked.",
    "Instrumentation timing and original initial-hit probability extraction remain inherited authenticated evidence.",
    "Immediate frozen inputs/process closure are authenticated; ancestral experiments are not re-audited.",
    "Agreement verifies this fixed opportunity screen, not recurrence, world-model or architectural novelty.",
]


PHYSICAL_SCALARS = ("original_setup_wall_seconds", "validation_wall_seconds", "threshold_wall_seconds",
    "validation_setup_seconds", "calibration_paid_seconds", "evaluation_wall_seconds", "journal_io_seconds")


def valid_physical_scalars(costs):
    """Exact scalar schema; operation_seconds is independently audited as a ledger."""
    return (isinstance(costs, dict) and set(costs) == {*PHYSICAL_SCALARS, "operation_seconds", "scope"}
        and isinstance(costs["operation_seconds"], dict) and isinstance(costs["scope"], str)
        and all(type(costs[key]) in (int, float) and math.isfinite(costs[key]) and costs[key] >= 0
                for key in PHYSICAL_SCALARS))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cohort():
    result = []
    for stage, starts, size in (("valid", VALID_FIRST, 12), ("eval", FIRST, 24)):
        for ri, (regime, first) in enumerate(starts.items()):
            for case in range(size):
                order = ("period2",) if stage == "valid" else ARMS
                if stage == "eval":
                    shift = (ri * size + case) % len(ARMS)
                    order = order[shift:] + order[:shift]
                for arm in order:
                    result.append({"stage": stage, "episode_index": len(result), "regime": regime,
                        "seed": first + case, "case": case, "initial_hit": case % 3 + 1, "block": case // 3,
                        "arm": arm, "episode_id": f"{stage}:{regime}:{first + case}:{arm}"})
    return result


def expected_query(kind, seed, step, used, entropy, threshold):
    """Independent scalar schedule, including every prefix and exact pair bit."""
    if not (type(step) is int and 0 <= step < 2188 and type(used) is int and 0 <= used <= (step + 1) // 2):
        raise ValueError("invalid chronological query state")
    offset = digest = None
    if kind == "period2":
        if used != (step + 1) // 2:
            raise ValueError("period2 prior count")
        query = step % 2 == 0
    elif kind == "random_pair":
        digest = hashlib.sha256(b"openjev.otto.sparse-query.random-pair.v1\x00" + struct.pack(">II", seed, step // 2)).hexdigest()
        offset = int(digest[:2], 16) & 1
        if used != step // 2 + int(step % 2 == 1 and offset == 0):
            raise ValueError("random pair prior count")
        query = step % 2 == offset
    elif kind == "entropy":
        if threshold is None or not math.isfinite(entropy) or not math.isfinite(threshold):
            raise ValueError("finite frozen entropy threshold")
        query = entropy >= threshold and used < (step + 2) // 2
    else:
        raise ValueError("unknown sparse schedule")
    return query, offset, digest


def selected_action(scores, allowed, queried):
    if not allowed or sorted(set(allowed)) != allowed or any(a not in range(4) for a in allowed):
        raise ValueError("canonical eligible actions")
    if len(scores) != 4 or any(type(scores[a]) not in (int, float) or not math.isfinite(scores[a]) for a in allowed):
        raise ValueError("finite eligible scores")
    if queried and any(f32(scores[a]) != scores[a] for a in allowed):
        raise ValueError("original float32 neural scores")
    minimum = min(scores[a] for a in allowed)
    return next(a for a in allowed if abs(f32(scores[a] - minimum) if queried else scores[a] - minimum)
                < (f32(1e-10) if queried else 1e-10))


def median_record(decisions, episodes):
    lengths = {r["episode_index"]: r["steps"] for r in episodes}
    if len(episodes) != 24 or set(lengths) != set(range(24)) or any(type(n) is not int or not 1 <= n <= 2188 for n in lengths.values()):
        raise ValueError("complete VALID episode lengths")
    ordered, seen = [], set()
    for row in decisions:
        index, step, value = row["episode_index"], row["step"], row["entropy"]
        if index not in lengths or type(step) is not int or not 0 <= step < lengths[index] or (index, step) in seen:
            raise ValueError("unique VALID decision")
        if type(value) is not float or not math.isfinite(value) or f32(value) != value:
            raise ValueError("saved float32 entropy")
        ordered.append((value, index, step)); seen.add((index, step))
    if len(seen) != sum(lengths.values()):
        raise ValueError("all VALID decisions")
    mass = Fraction(0)
    for value, index, _step in sorted(ordered):
        mass += Fraction(1, 24 * lengths[index])
        if mass >= Fraction(1, 2):
            return {"value": value, "threshold": value,
                "method": "episode_balanced_lower_weighted_median", "episodes": 24,
                "decisions": len(ordered), "weighting": "1/(24*episode_steps)",
                "direction": "entropy >= threshold", "total_weight": 1.0,
                "selection_cumulative_weight": float(mass),
                "selection_cumulative_weight_exact": [mass.numerator, mass.denominator],
                "source": "VALID period2 public float32 entropy only; no outcome optimization"}
    raise ValueError("undefined weighted median")



def reduce_rows(rows, mixtures, calibration):
    if len(rows) != 360 or len({(r["regime"], r["case"], r["arm"]) for r in rows}) != 360:
        raise ValueError("complete unique EVAL cohort")
    rows = [{**r, "paid_controller_seconds": r["controller_seconds"] + (calibration / 72 if r["arm"] == "entropy" else 0.)} for r in rows]
    primary = [{"name": "technical_complete", "value": 1, "threshold": 1, "relation": ">=", "passes": True},
               {"name": "causal_quotas", "value": int(all(r["query_quota_valid"] for r in rows if r["arm"] in SPARSE)),
                "threshold": 1, "relation": ">=", "passes": all(r["query_quota_valid"] for r in rows if r["arm"] in SPARSE)}]
    diagnostic, regimes = [], {}
    for setting in FIRST:
        local = [r for r in rows if r["regime"] == setting]
        weights = {str(h): float(mixtures[setting][str(h)]) for h in (1, 2, 3)}
        if len(local) != 120 or any(not math.isfinite(v) or v <= 0 for v in weights.values()) or abs(math.fsum(weights.values()) - 1) > 1e-12:
            raise ValueError("complete setting and positive-hit mixture")
        def means(selected, weights=weights):
            answer = {}
            for metric in METRICS:
                strata = [[float(r[metric]) for r in selected if r["initial_hit"] == h] for h in (1, 2, 3)]
                if any(not values for values in strata):
                    raise ValueError("complete hit strata")
                answer[metric] = math.fsum(weights[str(h)] * math.fsum(strata[h-1]) / len(strata[h-1]) for h in (1, 2, 3))
            return answer
        arms = {arm: means([r for r in local if r["arm"] == arm]) for arm in ARMS}
        blocks = [{arm: means([r for r in local if r["arm"] == arm and r["block"] == block]) for arm in ARMS} for block in range(8)]
        def rule(name, value, bound, relation):
            return {"name": name, "value": value, "threshold": bound, "relation": relation,
                    "passes": value >= bound if relation == ">=" else value <= bound}
        if setting != "lambda5":
            primary.extend(rule(f"{setting}.{arm}.success", arms[arm]["found"], .95, ">=") for arm in ("analytic", "neural"))
        for arm in SPARSE:
            own, neural, analytic = arms[arm], arms["neural"], arms["analytic"]
            comparisons = [rule(f"{setting}.{arm}.{name}", value, bound, relation) for name, value, bound, relation in (
                ("success", own["found"], .95, ">="),
                ("success_vs_references", own["found"], max(neural["found"], analytic["found"]), ">="),
                ("moves_vs_neural", own["steps"], 1.05 * neural["steps"], "<="),
                ("moves_vs_analytic", own["steps"], .95 * analytic["steps"], "<="),
                ("paid_cost_vs_neural", own["paid_controller_seconds"], .6 * neural["paid_controller_seconds"], "<="))]
            diagnostic.extend(comparisons)
            if setting != "lambda5" and arm == "period2":
                primary.extend(comparisons)
            diagnostic.append(rule(f"{setting}.{arm}.old_half_cost_descriptive", own["paid_controller_seconds"], .5 * neural["paid_controller_seconds"], "<="))
        regimes[setting] = {"weights": weights, "means": arms, "blocks": blocks,
            "raw_counts": {arm: {"episodes": 24, **{key: sum(r[key] for r in local if r["arm"] == arm) for key in ("found", "steps", "queries")}} for arm in ARMS}}
    return {"version": "otto-sparse-metrics-v1", "episodes": 360, "paired_cases": 72, "primary_arm": "period2",
        "regimes": regimes, "required": primary, "diagnostic": diagnostic,
        "required_conditions": 16, "required_passed": sum(r["passes"] for r in primary),
        "diagnostic_conditions": 54, "diagnostic_passed": sum(r["passes"] for r in diagnostic),
        "pilot_continuation": all(r["passes"] for r in primary), "calibration_paid_seconds": calibration,
        "calibration_per_entropy_episode_seconds": calibration / 72,
        "scope": "Period2 only primary; random/entropy and length5 diagnostic. Technical admission also requires "
                 "original process closure and independent saved audit. No recurrence or architecture efficacy claim."}

def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.start = self.launch = None
        self.recording_failure = False
        self.counts = collections.Counter()
        self.receipt = {"version": VERSION, "status": "started", "agreement": False, "limits": LIMITS,
                        "model_calls": 0, "native_calls": 0, "sampler_calls": 0, "optimizer_calls": 0, "binary_array_decodes": 0,
                        "limitations": LIMITATIONS, "failures": []}

    def require(self, value, message):
        self.counts["checks"] += 1
        if not value:
            raise ValueError(message)
        if not self.recording_failure and self.clock is not None and self.launch is not None and self.counts["checks"] % 256 == 0:
            self.check()

    def check(self):
        if self.clock.now_ns() >= self.launch["deadline_ns"]:
            raise TimeoutError("original saved-audit deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        if rss > LIMITS["rss_bytes"] or sum(p.stat().st_size for p in self.out.iterdir()) > LIMITS["output_bytes"]:
            raise MemoryError("saved-audit resource cap")

    def path(self, value):
        p = Path(value)
        p = p if p.is_absolute() else ROOT / p
        self.require(p.is_file() and p.is_relative_to(ROOT) and ".." not in p.parts
                     and not any(q.is_symlink() for q in (p, *p.parents)), "contained regular input")
        return p

    def digest(self, value):
        p = self.path(value)
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
                if not self.recording_failure and self.clock is not None and self.launch is not None:
                    self.check()
        return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}

    def read(self, value):
        return json.loads(self.path(value).read_text())

    def rows(self, path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(self.path(path), "rt") as stream:
            for line in stream:
                self.require(len(line) <= 2 * 1024**2, "bounded saved JSON row")
                yield json.loads(line)

    def same(self, actual, expected, name):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), name + " keys")
            for k, v in expected.items():
                self.same(actual[k], v, f"{name}.{k}")
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), name + " length")
            for i, value in enumerate(expected):
                self.same(actual[i], value, f"{name}[{i}]")
        elif isinstance(expected, float):
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), name)
        else:
            self.require(type(actual) is type(expected) and actual == expected, name)

    def admit(self):
        self.require(self.digest(CLOCK)["sha256"] == CLOCK_PIN, "qualified clock before import")
        spec = importlib.util.spec_from_file_location("_sparse_saved_audit_clock", ROOT / CLOCK)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.clock = module.SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            self.require(self.clock.now_ns() - self.start < 5 * 10**9, "original launch available")
            time.sleep(.01)
        self.launch = self.read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
            and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
            and Path.cwd() == ROOT == Path(self.launch["cwd"]) and self.launch["cap_seconds"] == 120
            and self.launch["clock_backend"] == self.clock.backend
            and self.launch["started_ns"] <= self.start < self.launch["deadline_ns"]
            and self.launch["deadline_ns"] == self.launch["started_ns"] + 120 * 10**9
            and self.launch["clock_source_sha256"] == CLOCK_PIN
            and self.launch["watchdog_sha256"] == SUPERVISOR_PIN, "original bounded audit process")
        self.inputs = {}
        for role in ("plan", "worker", "terminal"):
            path = getattr(self.args, role)
            record = self.digest(path)
            self.require(record["sha256"] == getattr(self.args, role + "_sha256"), "external original " + role)
            self.inputs[role] = {"path": str(path), **record}
        self.plan = self.read(self.args.plan)
        self.require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_validation"
            and self.plan["sources"][ORIGINAL] == self.digest(ORIGINAL)["sha256"] == ORIGINAL_PIN
            and self.plan["sources"][CLOCK] == CLOCK_PIN
            and self.plan["sources"][SUPERVISOR] == SUPERVISOR_PIN, "audit frozen before scientific work")
        self.admit_repair()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.require(os.environ.get(name) == "1", "single-thread audit environment")
        self.receipt.update(plan_sha256=self.args.plan_sha256, producer_inputs=self.inputs,
            supervision_sha256=self.digest(self.args.supervision)["sha256"],
            sources=self.repair["sources"], repair_plan={"path": str(self.args.repair_plan), **self.digest(self.args.repair_plan)},
            preserved_failure={k: self.repair["inputs"][k] for k in ("failed_audit", "failed_audit_terminal")})
        write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start, "producer_inputs": self.inputs})

    def admit_repair(self):
        self.require(self.digest(self.args.repair_plan)["sha256"] == self.args.repair_plan_sha256, "external prospective repair plan")
        self.repair = self.read(self.args.repair_plan)
        r = self.repair
        self.require(r["version"] == REPAIR_VERSION and r["status"] == "frozen_before_saved_arithmetic"
            and r["limits"] == LIMITS
            and set(r["sources"]) == {SELF, TEST, ORIGINAL, CLOCK, SUPERVISOR, REPAIR_PROTOCOL}
            and set(r["inputs"]) == {"plan", "worker", "terminal", "failed_audit", "failed_audit_terminal"}, "exact separate engineering repair scope")
        self.require(r["sources"][ORIGINAL] == ORIGINAL_PIN and r["sources"][CLOCK] == CLOCK_PIN
            and r["sources"][SUPERVISOR] == SUPERVISOR_PIN and r["sources"][REPAIR_PROTOCOL] == REPAIR_PROTOCOL_PIN, "unchanged original audit and repair protocol")
        for name, pin in r["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "repair source " + name)
        for record in r["inputs"].values():
            self.require(Path(record["path"]).is_absolute() and self.digest(record["path"]) == {
                k: record[k] for k in ("sha256", "bytes")}, "repair preserved input")
        self.same({k: r["inputs"][k] for k in self.inputs}, self.inputs, "identical scientific allocation")
        failed = self.read(r["inputs"]["failed_audit"]["path"])
        terminal = self.read(r["inputs"]["failed_audit_terminal"]["path"])
        self.require(failed["version"] == "otto-sparse-query-saved-audit-v1" and failed["status"] == "failed"
            and failed["agreement"] is False and failed["plan_sha256"] == self.args.plan_sha256
            and failed["sources"] == {ORIGINAL: ORIGINAL_PIN, CLOCK: CLOCK_PIN, SUPERVISOR: SUPERVISOR_PIN}
            and failed["failures"] and failed["failures"][0]["error"] == "ValueError('finite physical stage costs')", "preserved exact original audit failure")
        self.same(failed["producer_inputs"], self.inputs, "failed original audited the same inputs")
        directory = self.path(r["inputs"]["failed_audit"]["path"]).parent
        self.require({p.name for p in directory.iterdir()} == set(failed["files"]) | {"receipt.json"}, "preserved failed audit closure")
        for name, pin in failed["files"].items():
            self.require(Path(name).name == name and self.digest(directory / name) == pin, "original failed payload")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        self.require(command[:2] == [str(ROOT / ".venv/bin/python"), str(ROOT / ORIGINAL)]
            and len(command[2:]) == 16, "original failed invocation")
        options = dict(zip(command[2::2], command[3::2], strict=True))
        expected = {"--plan": str(self.args.plan), "--plan-sha256": self.args.plan_sha256,
            "--worker": str(self.args.worker), "--worker-sha256": self.args.worker_sha256,
            "--terminal": str(self.args.terminal), "--terminal-sha256": self.args.terminal_sha256,
            "--output": str(directory)}
        self.require(set(options) == set(expected) | {"--supervision"}
            and all(options[k] == v for k, v in expected.items()), "failed attempt exact argument coverage")
        self.failed_launch = self.path(options["--supervision"])
        self.require(self.digest(self.failed_launch)["sha256"] == failed["supervision_sha256"], "original failed launch pin")
        launch = self.read(self.failed_launch)
        self.require(terminal["status"] == "failed" and terminal["returncode"] == 1 and terminal["timed_out"] is False
            and terminal["error"] is terminal["clock_error"] is None and terminal["group_absent"] is True
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == []
            and terminal["cap_seconds"] == 120 and terminal["clock_source_sha256"] == CLOCK_PIN
            and terminal["watchdog_sha256"] == SUPERVISOR_PIN
            and terminal["started_ns"] < terminal["finished_ns"] <= terminal["deadline_ns"]
            and terminal["deadline_ns"] == terminal["started_ns"] + 120 * 10**9
            and terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
            and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9, "original failed parent fully closed")
        self.same(self.read(directory / "started.json")["launch"], launch, "failed original started witness")
        for key, value in launch.items():
            self.same(terminal[key], value, "failed original launch " + key)

    def authenticate(self):
        self.worker, self.parent = self.read(self.args.worker), self.read(self.args.terminal)
        w, t, p = self.worker, self.parent, self.plan
        command = t["command"]
        self.require(command[:4] == [str(ROOT / ".venv-otto-released-native/bin/python"), "-u", str(ROOT / PRODUCER), "run"]
            or command[:3] == [str(ROOT / ".venv-otto-released-native/bin/python"), str(ROOT / PRODUCER), "run"], "actual absolute original command")
        self.producer_launch_path = self.path(command[command.index("--supervision") + 1])
        launch = self.read(self.producer_launch_path)
        self.require(w["status"] == "completed" and w["complete"] is True and w["version"] == PRODUCER_VERSION
            and w["pending"] == [] and w["pending_emission"] is None and w["pending_episode"] is None
            and not w.get("cleanup_errors") and w["completed_episodes"] == 384 and w["validation_episodes"] == 24
            and w["evaluation_episodes"] == 360 and w["training_updates"] == w["annotations"] == 0
            and w["plan_sha256"] == self.args.plan_sha256
            and w["supervision_sha256"] == self.digest(self.producer_launch_path)["sha256"], "complete original worker")
        self.require(t["status"] == "completed" and t["returncode"] == 0 and t["timed_out"] is False
            and t["error"] is None and t["clock_error"] is None and t["group_absent"] is True
            and t["cleanup"]["reaped"] is True and t["cleanup"]["errors"] == [], "successful original parent")
        for key, value in launch.items():
            self.same(t[key], value, "original launch " + key)
        self.require(launch["cwd"] == str(ROOT) and launch["cap_seconds"] == 1800
            and launch["clock_source_sha256"] == CLOCK_PIN and launch["watchdog_sha256"] == SUPERVISOR_PIN
            and launch["deadline_ns"] == launch["started_ns"] + 1800 * 10**9
            and launch["started_ns"] <= w["started_ns"] < w["finished_ns"] <= t["finished_ns"] <= launch["deadline_ns"]
            and t["elapsed_ns"] == t["finished_ns"] - t["started_ns"]
            and t["wall_seconds"] == t["elapsed_ns"] / 1e9
            and w["wall_seconds"] == (w["finished_ns"] - w["started_ns"]) / 1e9, "complete physical clocks")
        self.run = self.args.worker.parent
        self.require(Path(command[command.index("--plan") + 1]) == self.args.plan
            and command[command.index("--plan-sha256") + 1] == self.args.plan_sha256
            and Path(command[command.index("--output") + 1]) == self.run, "original invocation joins")
        self.require(set(w["files"]) == set(p["payloads"]) == PAYLOADS
            and {x.name for x in self.run.iterdir()} == PAYLOADS | {"receipt.json"}, "exact closed payloads")
        for name, pin in w["files"].items():
            self.require(self.digest(self.run / name) == pin, "payload " + name)
        for name, pin in p["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "frozen source " + name)
        self.require(w["sources"] == p["sources"] and w["inputs"] == p["inputs"] and w["native_inputs"] == p["native_inputs"], "worker provenance")
        for record in [*p["inputs"].values(), *p["native_inputs"].values()]:
            self.require(self.digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "immediate inherited input")
        self.same(p["cohort"], cohort(), "all 384 fixed identities")
        self.same(p["limits"], {"native_seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 6 * 1024**3}, "fixed original cap")
        self.require(w["peak_rss_bytes"] <= p["limits"]["rss_bytes"]
            and sum(x["bytes"] for x in w["files"].values()) + self.args.worker.stat().st_size <= p["limits"]["output_bytes"], "original recorded resource limits")
        self.saved = self.read(self.run / "summary.json")
        self.deploy, self.setup, self.cost = (self.read(self.run / (name + ".json")) for name in ("deployment", "setup", "costs"))
        self.threshold = self.read(self.run / "threshold.json")
        self.same(self.saved["deployment"], self.deploy, "deployment join")
        self.same(self.saved["costs"], self.cost, "cost join")
        self.same(self.saved["threshold"], self.threshold, "threshold join")
        self.require(w["threshold"] == self.digest(self.run / "threshold.json"), "closed fixed threshold bytes")
        self.mixtures = self.deploy["mixtures"]
        self.require(set(self.mixtures) == set(FIRST) and self.deploy["warmup_forwards"] == 0, "all inherited mixtures and no warmup")
        self.counts["payloads"], self.counts["source_files"] = len(PAYLOADS), len(p["sources"])

    def episodes(self):
        rows, paired = [], {}
        boundaries = iter(self.rows(self.run / "episode-boundaries.jsonl"))
        for row, identity in zip(self.rows(self.run / "episodes.jsonl"), cohort(), strict=True):
            if identity["episode_index"] == 24:
                self.same(next(boundaries), {"event": "threshold_published", "threshold": self.worker["threshold"],
                    "completed_validation_episodes": 24, "completed_evaluation_episodes": 0}, "durable threshold barrier before first EVAL")
            self.same({k: row[k] for k in identity}, identity, "exact episode identity")
            self.same(next(boundaries), {"event": "attempt", **identity}, "durable episode attempt")
            self.same(next(boundaries), {"event": "return", **identity, "steps": row["steps"]}, "durable full episode acknowledgement")
            steps, found, arm = row["steps"], row["found"], row["arm"]
            self.require(type(steps) is int and 1 <= steps <= 2188 and type(found) is bool and (found or steps == 2188)
                and row["updates"] == steps and row["blocked_steps"] == 0 and row["final_update_assimilated"] is True
                and row["final_public"]["step"] == steps and row["final_public"]["done"] is found, "terminal or complete censored tail")
            self.require(type(row["queries"]) is int and 0 <= row["queries"] <= steps and type(row["query_quota_valid"]) is bool
                and (arm != "analytic" or row["queries"] == 0) and (arm != "neural" or row["queries"] == steps)
                and (arm not in SPARSE or row["queries"] <= (steps + 1) // 2), "actual endpoint/query counts")
            self.require(all(math.isfinite(row[k]) and row[k] >= 0 for k in METRICS if k != "paid_controller_seconds"), "finite nonnegative episode metrics")
            self.same(row["controller_seconds"], math.fsum(row[k] for k in ("init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds")), "controller component sum")
            draws = row.pop("draws_evaluation_only")
            self.require([x["index"] for x in draws if x["channel"] == "source"] == [0]
                and [x["index"] for x in draws if x["channel"] == "hit"] == list(range(steps - int(found)))
                and all(x["channel"] in ("source", "hit") and type(x["selected_index"]) is int
                    and 0 <= x["uniform"] < 1 and math.isfinite(x["cdf_mass"]) and x["cdf_mass"] > 0 for x in draws), "complete categorical saved witnesses")
            source = row["source_evaluation_only"]
            self.require(len(source) == 2 and all(type(x) is int and 0 <= x < 53 for x in source)
                and next(x["selected_index"] for x in draws if x["channel"] == "source") == source[0] * 53 + source[1], "original source coordinate")
            stream = {(x["channel"], x["index"]): x["uniform"] for x in draws}
            self.require(len(stream) == len(draws), "unique categorical identities")
            row["_hit_choices"] = [x["selected_index"] for x in draws if x["channel"] == "hit"]
            if row["stage"] == "eval":
                key = row["regime"], row["case"]
                if key in paired:
                    old_source, old_stream = paired[key]
                    self.require(old_source == source and all(old_stream[k] == stream[k] for k in old_stream.keys() & stream.keys()), "paired source and overlapping uniforms")
                    old_stream.update(stream)
                else:
                    paired[key] = source, stream
            rows.append(row)
        self.require(next(boundaries, None) is None, "no extra completion boundaries")
        self.counts["episodes"], self.counts["paired_cases"] = len(rows), len(paired)
        return rows

    def costs(self, rows):
        setup, d = self.setup, self.deploy
        cold = {"common_setup_allocation_seconds": setup["common_import_seconds"] + setup["shared_evaluator_setup_seconds"] + d["extra_common_seconds"],
                "tf_setup_allocation_seconds": setup["model_setup_seconds"], "gate_setup_allocation_seconds": d["gate_module_seconds"]}
        self.require(all(math.isfinite(v) and v >= 0 for v in cold.values()), "finite cold setup")
        for row in rows:
            shares = {"common_setup_allocation_seconds": cold["common_setup_allocation_seconds"] / 384,
                "tf_setup_allocation_seconds": cold["tf_setup_allocation_seconds"] / 312 if row["arm"] != "analytic" else 0.,
                "gate_setup_allocation_seconds": cold["gate_setup_allocation_seconds"] / 240 if row["arm"] in SPARSE else 0.}
            for key, value in shares.items():
                self.same(row[key], value, "exact cold allocation " + key)
            self.same(row["setup_allocation_seconds"], sum(shares.values()), "complete episode setup")
        for key, value in cold.items():
            self.same(math.fsum(r[key] for r in rows), value, "cold physical conservation " + key)
        c = self.cost
        self.same(c["original_setup_wall_seconds"], d["original_setup_wall_seconds"], "physical setup join")
        self.require(valid_physical_scalars(c), "exact finite scalar physical stage costs and separate operation ledger")
        self.same(c["validation_setup_seconds"], math.fsum(r["setup_allocation_seconds"] for r in rows[:24]), "24 paid VALID setup shares")
        self.same(c["calibration_paid_seconds"], sum(c[k] for k in ("validation_wall_seconds", "threshold_wall_seconds", "validation_setup_seconds")), "full one-time calibration bill")
        self.require(math.fsum(c[k] for k in ("original_setup_wall_seconds", "validation_wall_seconds", "threshold_wall_seconds", "evaluation_wall_seconds")) <= self.worker["wall_seconds"], "physical stages contained within worker wall")
        self.same(math.fsum(c["calibration_paid_seconds"] / 72 for r in rows[24:] if r["arm"] == "entropy"), c["calibration_paid_seconds"], "calibration charged exactly once across all 72 entropy episodes")
    def journals(self, rows):
        by_id = {r["episode_id"]: r for r in rows}
        identities = cohort()
        calls, elapsed = collections.Counter(), collections.defaultdict(float)
        per_case, per_time = collections.defaultdict(collections.Counter), collections.defaultdict(lambda: collections.defaultdict(float))
        outer, stack, sequence, resets, threshold_returned = collections.defaultdict(list), [], 0, 0, False
        tf_work = hashlib.sha256()
        for event in self.rows(self.run / "work.jsonl.gz"):
            self.counts["work_events"] += 1
            channel, context = event["channel"], event["context"]
            self.require(channel in self.plan["call_caps"], "declared work operation")
            ep = context.get("episode_id")
            if ep is not None:
                self.require(ep in by_id, "declared work episode")
                self.same({k: context[k] for k in identities[by_id[ep]["episode_index"]]}, identities[by_id[ep]["episode_index"]], "work episode identity")
            if event["event"] == "attempt":
                sequence += 1
                self.require(event["call_id"] == sequence and event["parent_call_id"] == (stack[-1]["call_id"] if stack else None), "unique nested operation attempt")
                if channel == "native_reset":
                    self.require(not stack and resets < 384 and ep == rows[resets]["episode_id"]
                        and (resets < 24 or threshold_returned), "fixed native reset chronology after calibration")
                    if resets:
                        previous = rows[resets-1]
                        self.require(per_case[previous["episode_id"]]["actor_update"] == previous["steps"], "previous full tail before next reset")
                    resets += 1
                if channel == "threshold_selection":
                    self.require(not stack and resets == 24 and not threshold_returned
                        and sum(per_case[r["episode_id"]]["actor_update"] for r in rows[:24]) == sum(r["steps"] for r in rows[:24]), "one threshold after all VALID updates")
                if channel in ("tensorflow_value", "sparse_gate"):
                    self.require(stack and stack[-1]["channel"] == "actor_choose" and ep is not None
                        and context["step"] == per_case[ep]["actor_choose"] + 1, "query computation inside actual choice")
                elif ep is not None:
                    self.require(not stack, "episode primitive is top-level")
                    outer[ep].append((channel, context.get("step")))
                stack.append(event)
            else:
                self.require(event["event"] == "return" and bool(stack), "returned pending operation")
                attempt = stack.pop()
                self.same({k: event[k] for k in attempt if k != "event"}, {k: v for k, v in attempt.items() if k != "event"}, "exact return identity")
                seconds = event["seconds"]
                self.require(math.isfinite(seconds) and seconds >= 0 and seconds == event["instrumented_seconds"] - event["excluded_io_seconds"], "original net operation timer")
                calls[channel] += 1; elapsed[channel] += seconds
                if channel == "threshold_selection":
                    threshold_returned = True
                if ep is not None:
                    per_case[ep][channel] += 1; per_time[ep][channel] += seconds
                    if channel == "tensorflow_value":
                        tf_work.update(json.dumps([ep, context["step"]]).encode() + b"\n")
        self.require(not stack and resets == 384 and threshold_returned, "closed complete work ledger")
        self.require(all(calls[c] == 1 for c in ("tensorflow_construction", "tensorflow_build", "tensorflow_load", "threshold_selection")), "single cold model and threshold computation")
        self.require(set(self.worker["calls"]) == set(self.plan["call_caps"]), "complete operation channels")
        for channel, record in self.worker["calls"].items():
            self.same(record, {"attempted": calls[channel], "returned": calls[channel], "seconds": elapsed[channel]}, "global work accounting")
            self.require(calls[channel] <= self.plan["call_caps"][channel], "frozen call cap")
        self.same(self.cost["operation_seconds"], self.worker["calls"], "cost ledger join")
        for row in rows:
            ep, steps, queries, arm = row["episode_id"], row["steps"], row["queries"], row["arm"]
            expected = {"native_reset": 1, "actor_construction": 1, "actor_choose": steps, "actor_update": steps,
                        "native_step": steps, "tensorflow_value": queries}
            ordered = [("native_reset", None), ("actor_construction", None)]
            if arm in SPARSE:
                expected.update(backend_binding=1, sparse_gate=steps)
                ordered.append(("backend_binding", None))
            ordered.extend((channel, step) for step in range(1, steps + 1) for channel in ("actor_choose", "native_step", "actor_update"))
            self.require(outer[ep] == ordered, "all ordered episode primitive calls")
            self.same(dict(per_case[ep]), {k: v for k, v in expected.items() if v}, "full episode operation counts")
            for field, channels in {"init_seconds": ("actor_construction", "backend_binding"), "choose_seconds": ("actor_choose",),
                "update_seconds": ("actor_update",), "environment_seconds": ("native_step",), "reset_seconds": ("native_reset",),
                "model_forward_seconds": ("tensorflow_value",)}.items():
                self.same(row[field], sum(per_time[ep][c] for c in channels), "actual episode time " + field)
        self.transitions(rows, tf_work, calls)
        gate_counts, gate_stack, ids = collections.defaultdict(collections.Counter), {}, collections.Counter()
        for event in self.rows(self.run / "gate-operations.jsonl.gz"):
            ep = event["episode_id"]
            self.require(ep in by_id and by_id[ep]["arm"] in SPARSE, "only sparse gate operations")
            if event["event"] == "attempt":
                ids[ep] += 1
                self.require(ep not in gate_stack and event["id"] == ids[ep], "unique gate attempt")
                gate_stack[ep] = event
            else:
                self.require(event["event"] == "return" and ep in gate_stack, "gate return")
                previous = gate_stack.pop(ep)
                self.same({k: v for k, v in event.items() if k != "event"}, {k: v for k, v in previous.items() if k != "event"}, "same gate operation")
                gate_counts[ep][event["channel"]] += 1
            self.counts["gate_operation_events"] += 1
        self.require(not gate_stack, "closed gate journal")
        for row in rows:
            if row["arm"] in SPARSE:
                expected = {"view_initialization": 1, "public_reset": 1, "analytic_score": row["steps"], "feature_build": row["steps"],
                    "gate": row["steps"], "neural_score": row["queries"], "public_update": row["steps"]}
                self.same(dict(gate_counts[row["episode_id"]]), {k: v for k, v in expected.items() if v}, "all public filter/gate calls")
                self.same(row["gate_progress"]["calls"], {k: {"attempted": v, "returned": v} for k, v in expected.items()}, "gate acknowledged counters")
                self.require(row["gate_progress"]["failed"] is False and row["gate_progress"]["pending_operations"] == []
                    and row["gate_progress"]["pending_action"] is None and row["gate_progress"]["errors"] == [], "closed sparse lifecycle")

    def transitions(self, rows, tf_work, calls):
        transitions = iter(self.rows(self.run / "transitions.jsonl.gz"))
        gates = iter(self.rows(self.run / "gate-decisions.jsonl.gz"))
        validation = iter(self.rows(self.run / "validation-decisions.jsonl.gz"))
        self.valid_decisions = []
        tf_transition, tf_record = hashlib.sha256(), hashlib.sha256()
        identities = cohort()
        for row in rows:
            identity = identities[row["episode_index"]]
            reset = next(transitions)
            self.same({k: reset[k] for k in identity}, identity, "reset identity")
            self.require(reset["kind"] == "reset" and reset["source_evaluation_only"] == row["source_evaluation_only"], "original reset witness")
            current, posterior = reset["public"], reset["posterior_after"]
            self.require(current["step"] == 0 and current["position"] == [26, 26] and current["hit"] == row["initial_hit"]
                and current["done"] is False and posterior["exact"] is True, "reset public state")
            queries, last_query, last_action, ages = 0, None, None, collections.Counter()
            for step in range(1, row["steps"] + 1):
                event = next(transitions)
                self.same({k: event[k] for k in identity}, identity, "transition identity")
                self.require(event["kind"] == "step" and event["step"] == step and event["public"]["step"] == step
                    and event["allowed_actions"] == current["valid_actions"] and event["action"] in current["valid_actions"]
                    and current["done"] is False and type(event["queried"]) is bool, "public step chronology")
                if row["arm"] in SPARSE:
                    gate = next(gates)
                    self.same({k: gate[k] for k in identity}, identity, "gate episode identity")
                    self.require(gate["step"] == step - 1 and gate["kind"] == row["arm"] and gate["episode_seed"] == row["seed"]
                        and gate["version"] == "otto-sparse-query-v1" and gate["state_before"] == [float(step - 1), float(queries)], "carried gate state")
                    feature = gate["features"]
                    self.require(len(feature) == 31 and all(type(v) is float and math.isfinite(v) and f32(v) == v for v in feature), "original finite float32 public features")
                    age = step - 1 if last_query is None else step - 1 - last_query
                    public_features = {0: current["position"][0] / 52, 1: current["position"][1] / 52,
                        6: int(row["regime"][-1]) / 5, 15: (step - 1) / 2188, 16: age / 2188,
                        17: float(last_query is not None), 18: posterior["mass"]}
                    public_features.update({2 + a: float(a in current["valid_actions"]) for a in range(4)})
                    public_features.update({7 + h: float(current["hit"] == h) for h in range(4)})
                    public_features.update({11 + a: float(last_action == a) for a in range(4)})
                    self.require(all(feature[k] == f32(v) for k, v in public_features.items()) and gate["query_age"] == age
                        and gate["entropy_feature"] == feature[19], "scalar public feature and query-age joins")
                    threshold = self.threshold["value"] if row["arm"] == "entropy" else None
                    query, offset, digest = expected_query(row["arm"], row["seed"], step - 1, queries, feature[19], threshold)
                    self.require(gate["entropy_threshold"] == threshold and gate["pair_index"] == ((step - 1) // 2 if row["arm"] == "random_pair" else None)
                        and gate["pair_offset"] == offset and gate["pair_sha256"] == digest and gate["query"] is query
                        and event["queried"] is query and gate["queries_before"] == queries
                        and gate["queries_after"] == queries + int(query) and gate["query_limit"] == (step + 1) // 2
                        and gate["state_after"] == [float(step), float(queries + int(query))], "independent sparse decision and exact quota")
                    if row["stage"] == "valid":
                        saved = next(validation)
                        self.same(saved, {**identity, "step": step - 1, "entropy": feature[19], "queried": query}, "all VALID entropy witnesses")
                        self.valid_decisions.append(saved)
                    ages[str(age)] += 1
                    self.counts["gate_decisions"] += 1
                else:
                    self.require(event["queried"] is (row["arm"] == "neural"), "standalone endpoint choice")
                self.require(event["cost_dtype"] == ("float32" if event["queried"] else "float64")
                    and selected_action(event["costs"], current["valid_actions"], event["queried"]) == event["action"]
                    and event["model_forward_calls"] == int(event["queried"]), "dtype-specific selected endpoint action")
                self.require(event["posterior_before"]["exact"] is True and event["posterior_after"]["exact"] is True, "inherited exact native/filter comparisons")
                self.same(event["posterior_before"], posterior, "public posterior witness continuity")
                position = list(current["position"])
                position[event["action"] // 2] += -1 if event["action"] % 2 == 0 else 1
                self.same(event["public"]["position"], position, "in-bounds action displacement")
                self.require(all(0 <= v < 53 for v in position), "bounded grid position")
                if event["public"]["done"]:
                    self.require(step == row["steps"] and position == row["source_evaluation_only"]
                        and event["posterior_after"]["mass"] == 1., "terminal source and inherited point-mass witness")
                else:
                    self.require(event["public"]["hit"] == row["_hit_choices"][step - 1], "saved nonterminal odor category")
                if event["queried"]:
                    queries += 1; last_query = step - 1
                    tf_transition.update(json.dumps([row["episode_id"], step]).encode() + b"\n")
                if row["arm"] in SPARSE:
                    self.require(event["query_quota_valid"] is True and queries <= (step + 1) // 2, "every executed sparse prefix")
                self.same(event["choose_seconds"], event["choose_instrumented_seconds"] - event["choose_excluded_io_seconds"], "step choice timer subtraction")
                current, posterior, last_action = event["public"], event["posterior_after"], event["action"]
                self.counts["transitions"] += 1
            self.same(current, row["final_public"], "last assimilated packet")
            self.require(queries == row["queries"] and row["query_quota_valid"] is True, "full actual query count and quota")
            self.same(dict(ages), row["query_age_histogram"], "full query-age histogram")
            self.require(row["maximum_query_age"] == max(map(int, ages), default=None), "maximum observed age")
        self.require(next(transitions, None) is None and next(gates, None) is None and next(validation, None) is None, "no extra saved transitions/decisions")
        for ordinal, event in enumerate(self.rows(self.run / "forwards.jsonl.gz"), 1):
            self.require(event["ordinal"] == ordinal and event["input_shape"] == [16, 105, 105]
                and event["symmetry_average"] is True and len(event["values"]) == 16
                and all(math.isfinite(v) for v in event["values"]), "original bounded neural forward witness")
            tf_record.update(json.dumps([event["context"]["episode_id"], event["context"]["step"]]).encode() + b"\n")
            self.counts["forwards"] += 1
        self.require(tf_record.digest() == tf_transition.digest() == tf_work.digest()
            and self.counts["forwards"] == calls["tensorflow_value"], "query iff exactly one actual neural forward")
        independent = median_record(self.valid_decisions, rows[:24])
        self.same(self.threshold, independent, "independent exact rational VALID median")
        self.counts["validation_decisions"] = len(self.valid_decisions)
    def body(self):
        self.admit()
        self.authenticate()
        rows = self.episodes()
        self.costs(rows)
        self.journals(rows)
        result = reduce_rows(rows[24:], self.mixtures, self.cost["calibration_paid_seconds"])
        result.update(version=PRODUCER_VERSION, costs=self.cost, threshold=self.threshold, deployment=self.deploy,
            validation_episodes=24, evaluation_episodes=360, training_updates=0, annotations=0)
        self.require(len(result["required"]) == 16 and len(result["diagnostic"]) == 54, "all fixed conditions")
        self.same(self.saved, result, "complete independent scientific summary")
        self.counts["required_conditions"], self.counts["diagnostic_conditions"] = 16, 54
        write(self.out / "audit.json", {"version": VERSION, "agreement": True, "summary": result, "limitations": LIMITATIONS})
        for name, pin in self.plan["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "unchanged original source")
        for record in [*self.plan["inputs"].values(), *self.plan["native_inputs"].values(), *self.inputs.values()]:
            self.require(self.digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged original input")
        for name, pin in self.worker["files"].items():
            self.require(self.digest(self.run / name) == pin, "unchanged original payload")
        for name, pin in self.repair["sources"].items():
            self.require(self.digest(name)["sha256"] == pin, "unchanged repair source")
        for record in self.repair["inputs"].values():
            self.require(self.digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "unchanged repair lineage")
        self.require(self.digest(self.args.repair_plan)["sha256"] == self.args.repair_plan_sha256, "unchanged frozen repair plan")
        self.require(self.digest(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged own launch")

    def execute(self):
        self.require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
            and not any(p.is_symlink() for p in self.out.parents), "contained exclusive audit output")
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("saved sparse audit terminated")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.body()
            files = {name: self.digest(self.out / name) for name in ("started.json", "audit.json")}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", agreement=True, counts=dict(self.counts), files=files,
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.recording_failure = True
            self.receipt.update(status="failed", agreement=False, counts=dict(self.counts),
                failures=[{"error": repr(error), "traceback": traceback.format_exc()}])
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.digest(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure and all evidence
                error.add_note(f"Failure publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "worker", "terminal", "repair-plan", "supervision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan", "worker", "terminal", "repair-plan"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    if not all(getattr(args, key).is_absolute() for key in ("plan", "worker", "terminal", "repair_plan", "supervision", "output")):
        parser.error("absolute paths required")
    Audit(args).execute()


if __name__ == "__main__":
    main()
