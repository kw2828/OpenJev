"""Fixed additive spectral compression on saved OTTO public trajectories.

No simulator, new gameplay, model training, or learned-performance admission.
All twelve reconstructions are scored on the same recorded full-policy prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

AUDITOR = "scripts/audit_otto_large_memory.py"
AUDITOR_PIN = "1ad5a080f58177751dadf0eaab0cd9e2c7b37d7e9d84b50cc32f4046a66155ec"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
MODEL = "src/openjev/research/otto_spectral_memory.py"
SOURCES = {"scripts/study_otto_spectral_memory.py", MODEL,
           "tests/test_otto_spectral_memory.py", "tests/test_otto_spectral_study.py",
           "research/otto-spectral-memory-protocol.md", AUDITOR,
           "src/openjev/research/suspend_clock.py"}
LIMITS = {"native_seconds": 300, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}
PUBLIC = {"position", "hit", "done", "step", "valid_actions"}
RANKS = (4, 8, 16)
EXTENSIONS = ("neutral", "nearest")
CANDIDATES = tuple(f"dct{q}_{extension}" for q in RANKS for extension in EXTENSIONS)
QUALIFICATION = ("exact_log", "dct53_neutral", "dct53_nearest")
ARMS = ("full_bayes", "exact_log", *CANDIDATES, "dct53_neutral", "dct53_nearest",
        "recent32", "recent32_hard")
MEASURES = ("tv_to_full", "kl_full_to_approx", "full_objective_excess", "matches_full_action")
THREAD_ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
SCOPE = ("Fixed nonlearned additive spectral compression on all 96 original full-policy paths. "
         "Every proposed action is teacher-forced, never executed. Both fill conventions and all "
         "three ranks are retained. Full-rank checks qualify arithmetic only. Original 9/10 "
         "continuation failure remains unchanged; no new quality gate or learned admission.")
CONFIGURATION = {"ranks": list(RANKS), "extensions": list(EXTENSIONS), "qualification_rank": 53,
                 "arms": list(ARMS), "cases": 96, "decisions": 2164, "eligible_prefixes": 538,
                 "eligibility": "completed_steps > 32", "score_tolerance": 1e-8,
                 "qualification_tv_tolerance": 1e-10,
                 "rotation": "(case_index + completed_steps) modulo12"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def locate(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_analytic():
    path = ROOT / AUDITOR
    require(sha(path) == AUDITOR_PIN, "pinned independent analytic source")
    spec = importlib.util.spec_from_file_location("otto_evidence_analytic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_pins(plan):
    require(set(plan["sources"]) == SOURCES and plan["sources"][AUDITOR] == AUDITOR_PIN
            and plan["sources"]["src/openjev/research/suspend_clock.py"] == CLOCK_PIN, "exact seven-source closure")
    for name, pin in plan["sources"].items():
        require(not (ROOT / name).is_symlink() and sha(ROOT / name) == pin, f"diagnostic source: {name}")


def authenticate(plan, analytic, budget):
    require(plan["version"] == "otto-spectral-memory-v1" and plan["status"] == "frozen_before_saved_study" and plan["limits"] == LIMITS, "frozen diagnostic limits")
    require(plan["configuration"] == CONFIGURATION, "fixed rank/fill/cohort configuration")
    source_pins(plan)
    inputs = plan["inputs"]
    require(set(inputs) == {"run", "receipt_sha256", "summary_sha256", "plan", "plan_sha256", "terminal",
                           "terminal_sha256", "audit", "audit_receipt_sha256", "audit_summary_sha256"}, "input closure")
    args = SimpleNamespace(run=locate(inputs["run"]), plan=locate(inputs["plan"]),
                           plan_sha256=inputs["plan_sha256"], receipt_sha256=inputs["receipt_sha256"],
                           terminal=locate(inputs["terminal"]), terminal_sha256=inputs["terminal_sha256"])
    original_plan, done, terminal = analytic.authenticate(args, budget)
    require(done["completed_episodes"] == 768 and done["learned_pilot_opportunity"] is False, "completed unchanged failed study")
    require(sha(args.run / "summary.json") == inputs["summary_sha256"] == done["files"]["summary.json"]["sha256"],
            "original summary pin")
    audit = locate(inputs["audit"])
    require(not Path(inputs["audit"]).is_symlink() and {p.name for p in audit.iterdir()} == {"started.json", "summary.json", "receipt.json"},
            "exact agreeing audit closure")
    require(sha(audit / "receipt.json") == inputs["audit_receipt_sha256"], "external audit receipt")
    receipt = read(audit / "receipt.json")
    require(receipt["status"] == "completed" and receipt["agreement"] is True and receipt["source_sha256"] == AUDITOR_PIN
            and receipt["plan_sha256"] == inputs["plan_sha256"]
            and receipt["producer_receipt_sha256"] == inputs["receipt_sha256"]
            and receipt["producer_summary_sha256"] == inputs["summary_sha256"]
            and receipt["terminal_sha256"] == inputs["terminal_sha256"], "same completed independently audited study")
    require(set(receipt["files"]) == {"started.json", "summary.json"}, "audit payload closure")
    for name, witness in receipt["files"].items():
        path = audit / name
        require(path.is_file() and not path.is_symlink() and sha(path) == witness["sha256"]
                and path.stat().st_size == witness["bytes"], f"audit payload: {name}")
    require(sha(audit / "summary.json") == inputs["audit_summary_sha256"], "external audit summary")
    result = read(audit / "summary.json")
    require(result["agreement"] is True and result["episodes"] == 768 and result["recomputed_summary"]["learned_pilot_opportunity"] is False,
            "complete audit scope and unchanged failure")
    budget()
    return args.run, original_plan, done, terminal


def validate_public(packet, step, analytic):
    require(isinstance(packet, dict) and set(packet) == PUBLIC, "public packet whitelist")
    require(type(packet["step"]) is int and packet["step"] == step and packet["done"] is False, "completed nonterminal chronology")
    require(type(packet["hit"]) is int and 0 <= packet["hit"] < analytic.NHITS, "ordinary hit category, never terminal sentinel")
    position = packet["position"]
    require(isinstance(position, (tuple, list)) and len(position) == 2
            and all(type(x) is int and 0 <= x < analytic.N for x in position), "public position")
    position = tuple(position)
    require(packet["valid_actions"] == [a for a in range(4) if analytic.moved(position, a) != position], "valid action IDs")
    return position


def public_traces(run, analytic, budget):
    selected = {}
    groups = analytic.grouped(run / "transitions.jsonl")
    for case in range(96):
        shift = case % 8
        for arm in analytic.ARMS[shift:] + analytic.ARMS[:shift]:
            key, rows = next(groups, (None, None))
            require(key == (610001 + case, arm), "exact 768-case trace rotation")
            require(rows and rows[0]["kind"] == "reset" and all(r["kind"] == "step" for r in rows[1:]), "trace row kinds")
            if arm == "space_full":
                # Explicit projection excludes source, seed streams, native posterior and costs from actor reconstruction.
                trace = [{"public": rows[0]["public"]}]
                for i, row in enumerate(rows[1:], 1):
                    require(row["step"] == row["public"]["step"] == i, "source public time join")
                    trace.append({"public": row["public"], "action": row["action"], "scores": row["scores"]})
                selected[key] = trace
            budget()
    require(next(groups, None) is None and len(selected) == 96, "complete selected public pairs")
    return selected


def arm_order(case_index, completed_steps):
    require(type(case_index) is int and type(completed_steps) is int
            and case_index >= 0 and completed_steps >= 0, "nonnegative rotation indices")
    shift = (case_index + completed_steps) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def load_spectral(plan):
    path = ROOT / MODEL
    require(sha(path) == plan["sources"][MODEL], "spectral model source pin")
    spec = importlib.util.spec_from_file_location("qualified_otto_spectral", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_logs(np, values):
    require(not np.isnan(values).any() and not np.isposinf(values).any(), "valid log evidence")
    finite = np.isfinite(values)
    require(bool(finite.any()), "nonempty posterior support")
    peak = float(values[finite].max())
    normalizer = peak + math.log(float(np.exp(values[finite] - peak).sum()))
    logs = values - normalizer
    return logs, np.exp(logs)


class PublicBaseline:
    """Original Bayes or 32 soft observations, with a declared persistent mask.

    Recent controls preserve the retained-evidence definition using stable log
    reconstruction, not bitwise upstream probability-product arithmetic.
    """

    def __init__(self, initial, kernel, analytic, np, mode):
        require(mode in ("full_bayes", "recent32", "recent32_hard"), "baseline identity")
        self.mode, self.kernel, self.analytic, self.np = mode, kernel, analytic, np
        self.position = validate_public(initial, 0, analytic)
        self.step = 0
        prior = analytic.prior(kernel, initial["hit"])
        self.excluded = prior == 0
        self.excluded[self.position] = True
        if mode == "full_bayes":
            self.probabilities = prior
        else:
            self.initial_logs = np.full_like(prior, -np.inf)
            np.log(prior, out=self.initial_logs, where=prior > 0)
            self.initial_logs.setflags(write=False)
            self.history = np.zeros((32, 3), dtype=np.int64)
            self.count = 0

    def update(self, public):
        position = validate_public(public, self.step + 1, self.analytic)
        require(sum(abs(a - b) for a, b in zip(position, self.position, strict=True)) <= 1,
                "public movement continuity")
        likelihood = self.analytic.likelihood_at(self.kernel, position)[public["hit"]]
        self.excluded[position] = True
        if self.mode != "recent32":
            self.excluded |= likelihood == 0
        if self.mode == "full_bayes":
            self.probabilities[position] = 0
            self.probabilities = self.analytic.normalized(self.probabilities * likelihood)
        else:
            self.history[self.count % 32] = (*position, public["hit"])
            self.count += 1
        self.position, self.step = position, public["step"]

    def decode(self):
        np = self.np
        if self.mode == "full_bayes":
            probabilities = self.probabilities.copy()
            logs = np.full_like(probabilities, -np.inf)
            np.log(probabilities, out=logs, where=probabilities > 0)
            return logs, probabilities
        logits = self.initial_logs.copy()
        for index in range(max(0, self.count - 32), self.count):
            x, y, hit = self.history[index % 32]
            likelihood = self.analytic.likelihood_at(self.kernel, (int(x), int(y)))[int(hit)]
            increment = np.full_like(likelihood, -np.inf)
            np.log(likelihood, out=increment, where=likelihood > 0)
            logits += increment
        logits[self.excluded] = -np.inf
        return normalized_logs(np, logits)

    def storage_bytes(self):
        mutable = {"support_or_visited_mask": int(self.excluded.nbytes)}
        immutable = {}
        if self.mode == "full_bayes":
            mutable["probability_grid"] = int(self.probabilities.nbytes)
        else:
            mutable["32_public_observation_ring"] = int(self.history.nbytes)
            immutable["initial_log_prior"] = int(self.initial_logs.nbytes)
        return {"mutable_arrays": mutable, "mutable_array_bytes": sum(mutable.values()),
                "immutable_arrays": immutable, "immutable_array_bytes": sum(immutable.values()),
                "python_metadata_included": False,
                "mask_scope": ("initial prior zeros and visited cells only" if self.mode == "recent32"
                               else "initial prior zeros, visited cells and every historical likelihood zero"),
                "decoder_return_array_bytes": int(2 * self.excluded.size * 8),
                "temporary_workspace": "Full-grid float64 log/probability arrays; recent reconstruction additionally uses a full-grid increment."}


def choose(scores):
    valid = [a for a, score in enumerate(scores) if score is not None]
    require(len(scores) == 4 and valid and all(math.isfinite(scores[a]) for a in valid), "four finite supported scores")
    best = min(scores[a] for a in valid)
    ties = [a for a in valid if abs(scores[a] - best) < 1e-10]
    return ties[0], len(ties)


def validate_decoded(np, logs, probabilities, shape):
    require(logs.shape == probabilities.shape == shape and logs.dtype == probabilities.dtype == np.float64,
            "float64 full-grid decoder outputs")
    require(not np.isnan(logs).any() and not np.isposinf(logs).any()
            and np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            and abs(float(probabilities.sum()) - 1) <= 1e-10, "normalized supported decoder")
    require(np.all(np.isfinite(logs[probabilities > 0]))
            and float(np.max(np.abs(np.exp(logs) - probabilities))) <= 1e-12, "stable logs agree with probabilities")


def belief_metrics(np, full_logs, full_probs, logs, probabilities, scores, full_scores):
    action, ties = choose(scores)
    full_action, _ = choose(full_scores)
    positive = full_probs > 0
    require(np.isfinite(logs[positive]).all(), "approximation has log support wherever reference has positive mass")
    # Finite log q remains usable even when exp(log q) underflows to zero.
    kl = float(np.sum(full_probs[positive] * (full_logs[positive] - logs[positive])))
    require(math.isfinite(kl), "finite KL without floors")
    return {"tv_to_full": float(np.abs(full_probs - probabilities).sum() / 2),
            "kl_full_to_approx": kl,
            "full_objective_excess": full_scores[action] - min(x for x in full_scores if x is not None),
            "matches_full_action": action == full_action, "action": action, "tie_count": ties,
            "probability_underflow_cells": int(np.sum(np.isfinite(logs) & (probabilities == 0))),
            "finite_log_support_cells": int(np.isfinite(logs).sum()),
            "positive_probability_cells": int((probabilities > 0).sum())}


def parity(scores, recorded):
    require(len(scores) == len(recorded["scores"]) == 4, "saved score count")
    require(all((a is None) == (b is None) for a, b in zip(scores, recorded["scores"], strict=True)), "saved score support")
    error = max(abs(a - b) for a, b in zip(scores, recorded["scores"], strict=True) if a is not None)
    require(math.isfinite(error) and error <= 1e-8, "saved score parity")
    require(choose(scores)[0] == recorded["action"], "exact saved selected action")
    return error


def mean_records(records, weights=None):
    if not records:
        return None
    weights = [1.0] * len(records) if weights is None else weights
    require(len(weights) == len(records) and all(math.isfinite(w) and w > 0 for w in weights), "positive aggregation weights")
    keys = set(records[0])
    require(all(set(row) == keys for row in records), "metric key consistency")
    require(all(math.isfinite(float(value)) for row in records for value in row.values()), "finite aggregated metrics")
    denominator = math.fsum(weights)
    return {key: math.fsum(float(row[key]) * w for row, w in zip(records, weights, strict=True)) / denominator
            for key in sorted(keys)}


def flatten(modes, sensitivity):
    result = {f"{arm}.{metric}": float(modes[arm][metric]) for arm in ARMS for metric in MEASURES}
    result.update({f"fill.q{q}.{metric}": float(value) for q, fields in sensitivity.items() for metric, value in fields.items()})
    return result


def population(cases, mixture, eligible=False):
    mean_key, count_key = ("eligible_means", "eligible_prefixes") if eligible else ("all_means", "decisions")
    selected = [case for case in cases if case[count_key] > 0]
    records = [case[mean_key] for case in selected]
    weights = [mixture[case["initial_hit"]] / 32 for case in selected]
    counts = [case[count_key] for case in selected]
    return {"all_cases": len(cases), "selected_cases": len(selected), "prefixes": sum(counts),
            "selected_case_mixture_mass": math.fsum(weights),
            "by_initial_hit": {h: {"all_cases": sum(c["initial_hit"] == h for c in cases),
                                   "selected_cases": sum(c["initial_hit"] == h for c in selected),
                                   "prefixes": sum(c[count_key] for c in selected if c["initial_hit"] == h),
                                   "case_means": mean_records([c[mean_key] for c in selected if c["initial_hit"] == h]),
                                   "prefix_means": mean_records([c[mean_key] for c in selected if c["initial_hit"] == h],
                                                                [c[count_key] for c in selected if c["initial_hit"] == h])}
                               for h in (1, 2, 3)},
            "unweighted_case_means": mean_records(records),
            "mixture_weighted_case_means": mean_records(records, weights),
            "mixture_weighted_prefix_means": mean_records(records, [w * n for w, n in zip(weights, counts, strict=True)])}


def make_actor(arm, initial, model, kernel, analytic, np):
    if arm in ("full_bayes", "recent32", "recent32_hard"):
        return PublicBaseline(initial, kernel, analytic, np, arm)
    if arm == "exact_log":
        return model.start_exact(initial)
    rank, extension = arm.split("_")
    return model.start(initial, int(rank.removeprefix("dct")), extension=extension)


def eligible_prefix(completed_steps):
    require(type(completed_steps) is int and completed_steps >= 0, "nonnegative prefix age")
    return completed_steps > 32


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    own_pin, start, last = sha(__file__), None, None
    progress = {"completed_cases": 0, "prefixes": 0, "eligible_prefixes": 0,
                "qualified_prefixes": {arm: 0 for arm in ("full_bayes", *QUALIFICATION)},
                "maximum_score_error": {arm: 0.0 for arm in ("full_bayes", *QUALIFICATION)},
                "maximum_qualification_tv": {arm: 0.0 for arm in QUALIFICATION},
                "reference_zero_probability_on_hard_support_prefixes": 0,
                "maximum_reference_zero_probability_on_hard_support_cells": 0}
    try:
        require(sha(ROOT / "src/openjev/research/suspend_clock.py") == CLOCK_PIN, "clock source")
        clock = SuspendClock()
        start = clock.now_ns()
        deadline = start + LIMITS["native_seconds"] * 10**9

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last < deadline, "spectral study deadline")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    <= LIMITS["rss_bytes"], "spectral study RSS cap")
            require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= LIMITS["output_bytes"],
                    "spectral study output cap")

        write(output / "started.json", {"source_sha256": own_pin, "request": {k: str(v) for k, v in vars(args).items()},
              "clock_backend": clock.backend, "started_ns": start, "deadline_ns": deadline, "scope": SCOPE})
        require(sha(args.plan) == args.plan_sha256, "external spectral study plan pin")
        plan = read(args.plan)
        source_pins(plan)
        # Set before importing either numerical library into this fresh process.
        for name, value in THREAD_ENV.items():
            os.environ[name] = value
        import numpy as np

        analytic = load_analytic()
        spectral = load_spectral(plan)
        run, original_plan, done, terminal = authenticate(plan, analytic, budget)
        scipy_version = importlib.metadata.version("scipy")
        require(sys.version.split()[0] == original_plan["python_version"]
                and {"numpy": np.__version__, "scipy": scipy_version} == original_plan["runtime_versions"],
                "same qualified Python and numerical library versions")
        with np.load(run / "public-kernel.npz", allow_pickle=False) as saved:
            kernel = saved["likelihood"].copy()
            mixture = {h: float(saved["initial_hit_weights"][h]) for h in (1, 2, 3)}
        kernel.setflags(write=False)
        require(kernel.shape == (4, 107, 107), "unchanged53x53 numerical sensor")
        tick = time.perf_counter()
        model = spectral.SpectralModel(kernel)
        shared_initialization_seconds = time.perf_counter() - tick
        require(not model.kernel.flags.writeable and not model.initial_log_priors.flags.writeable,
                "immutable shared model arrays")
        model_pins = {"kernel": hashlib.sha256(model.kernel.tobytes()).hexdigest(),
                      "initial_log_priors": hashlib.sha256(model.initial_log_priors.tobytes()).hexdigest()}
        traces = public_traces(run, analytic, budget)
        cases, storage = [], None
        total_measured_seconds = shared_initialization_seconds
        with (output / "prefixes.jsonl").open("x") as prefix_stream, (output / "cases.jsonl").open("x") as case_stream:
            for case_index in range(96):
                seed = 610001 + case_index
                trace = traces[seed, "space_full"]
                initial, actors, timers = trace[0]["public"], {}, {}
                require(initial["hit"] == 1 + (case_index % 12) // 4, "original initial-hit stratum")
                for arm in arm_order(case_index, 0):
                    budget()
                    tick = time.perf_counter()
                    actors[arm] = make_actor(arm, initial, model, kernel, analytic, np)
                    timers[arm] = {"initialization_seconds": time.perf_counter() - tick,
                                   "update_seconds": 0.0, "decode_seconds": 0.0, "planner_seconds": 0.0,
                                   "update_calls": 0, "decode_calls": 0, "planner_calls": 0}
                initial_storage = {arm: actors[arm].storage_bytes() for arm in ARMS}
                if storage is None:
                    storage = initial_storage
                all_records, eligible_records = [], []
                underflow = {arm: {"prefixes_with_probability_underflow": 0, "maximum_underflow_cells": 0} for arm in ARMS}
                for completed_steps, event in enumerate(trace[1:]):
                    budget()
                    public = initial if completed_steps == 0 else trace[completed_steps]["public"]
                    position = validate_public(public, completed_steps, analytic)
                    decoded, score_vectors, measurements = {}, {}, {}
                    order = arm_order(case_index, completed_steps)
                    for arm in order:
                        actor, timing = actors[arm], timers[arm]
                        update_seconds = 0.0
                        if completed_steps:
                            tick = time.perf_counter()
                            actor.update(public)
                            update_seconds = time.perf_counter() - tick
                            timing["update_calls"] += 1
                        tick = time.perf_counter()
                        logs, probabilities = actor.decode()
                        decode_seconds = time.perf_counter() - tick
                        validate_decoded(np, logs, probabilities, (53, 53))
                        tick = time.perf_counter()
                        scores = analytic.action_scores(probabilities, position, kernel, True)
                        planner_seconds = time.perf_counter() - tick
                        decoded[arm] = logs, probabilities
                        score_vectors[arm] = scores
                        measurements[arm] = {"update_seconds": update_seconds, "decode_seconds": decode_seconds,
                                             "planner_seconds": planner_seconds}
                        for name, value in measurements[arm].items():
                            timing[name] += value
                        timing["decode_calls"] += 1
                        timing["planner_calls"] += 1
                    full_logs, full_probs = decoded["full_bayes"]
                    common_mask = actors["full_bayes"].excluded
                    for arm in ("exact_log", *CANDIDATES, "dct53_neutral", "dct53_nearest", "recent32_hard"):
                        require(np.array_equal(actors[arm].excluded, common_mask), "identical persistent hard support")
                    native_zero = int(np.sum(~common_mask & (full_probs == 0)))
                    progress["reference_zero_probability_on_hard_support_prefixes"] += int(native_zero > 0)
                    progress["maximum_reference_zero_probability_on_hard_support_cells"] = max(
                        progress["maximum_reference_zero_probability_on_hard_support_cells"], native_zero)
                    modes = {}
                    for arm in ARMS:
                        logs, probabilities = decoded[arm]
                        metrics = belief_metrics(np, full_logs, full_probs, logs, probabilities,
                                                 score_vectors[arm], score_vectors["full_bayes"])
                        modes[arm] = {**metrics, "scores": score_vectors[arm], "timing": measurements[arm]}
                        underflow[arm]["prefixes_with_probability_underflow"] += int(metrics["probability_underflow_cells"] > 0)
                        underflow[arm]["maximum_underflow_cells"] = max(underflow[arm]["maximum_underflow_cells"],
                                                                       metrics["probability_underflow_cells"])
                        if arm in ("full_bayes", *QUALIFICATION):
                            error = parity(score_vectors[arm], event)
                            progress["maximum_score_error"][arm] = max(progress["maximum_score_error"][arm], error)
                            progress["qualified_prefixes"][arm] += 1
                        if arm in QUALIFICATION:
                            require(metrics["tv_to_full"] <= 1e-10, "exact/full-rank posterior qualification")
                            progress["maximum_qualification_tv"][arm] = max(progress["maximum_qualification_tv"][arm],
                                                                          metrics["tv_to_full"])
                    sensitivity = {q: {"actions_disagree": modes[f"dct{q}_neutral"]["action"] != modes[f"dct{q}_nearest"]["action"],
                                       "tv_between_extensions": float(np.abs(decoded[f"dct{q}_neutral"][1]
                                                                            - decoded[f"dct{q}_nearest"][1]).sum() / 2)}
                                   for q in RANKS}
                    row = {"seed": seed, "initial_hit": initial["hit"], "block": case_index // 12,
                           "completed_steps": completed_steps, "position": list(position), "arm_order": list(order),
                           "modes": modes, "fill_sensitivity": sensitivity,
                           "persistent_excluded_cells": int(common_mask.sum()),
                           "reference_zero_probability_on_retained_support_cells": native_zero}
                    prefix_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    flat = flatten(modes, sensitivity)
                    all_records.append(flat)
                    if eligible_prefix(completed_steps):
                        eligible_records.append(flat)
                        progress["eligible_prefixes"] += 1
                    progress["prefixes"] += 1
                    if event["public"]["done"]:
                        require(completed_steps + 1 == len(trace) - 1 and event["public"]["hit"] == -2,
                                "terminal never enters a subsequent actor update")
                count = len(trace) - 1
                for arm in ARMS:
                    require(timers[arm]["update_calls"] == count - 1
                            and timers[arm]["decode_calls"] == timers[arm]["planner_calls"] == count,
                            "complete update/decode/planning accounting")
                    timers[arm]["total_seconds"] = math.fsum(timers[arm][name] for name in (
                        "initialization_seconds", "update_seconds", "decode_seconds", "planner_seconds"))
                    total_measured_seconds += timers[arm]["total_seconds"]
                    require(actors[arm].storage_bytes() == initial_storage[arm], "no growing history or model-array cache")
                require(hashlib.sha256(model.kernel.tobytes()).hexdigest() == model_pins["kernel"]
                        and hashlib.sha256(model.initial_log_priors.tobytes()).hexdigest() == model_pins["initial_log_priors"],
                        "shared model arrays unchanged")
                case = {"seed": seed, "block": case_index // 12, "initial_hit": initial["hit"],
                        "decisions": count, "eligible_prefixes": len(eligible_records),
                        "all_means": mean_records(all_records), "eligible_means": mean_records(eligible_records),
                        "timing": timers, "underflow": underflow, "storage": initial_storage}
                case_stream.write(json.dumps(case, sort_keys=True, allow_nan=False) + "\n")
                prefix_stream.flush()
                case_stream.flush()
                cases.append(case)
                progress["completed_cases"] += 1
        require(progress["completed_cases"] == 96 and progress["prefixes"] == 2164
                and progress["eligible_prefixes"] == 538
                and all(count == 2164 for count in progress["qualified_prefixes"].values()), "complete fixed cohort")
        case_weights = [mixture[c["initial_hit"]] / 32 for c in cases]
        timing_summary = {arm: mean_records([{name: value for name, value in c["timing"][arm].items()
                                              if name.endswith("_seconds")} for c in cases], case_weights) for arm in ARMS}
        budget()
        summary = {"scope": SCOPE, "configuration": CONFIGURATION, "progress": progress,
                   "qualification_rates": {arm: count / progress["prefixes"]
                                           for arm, count in progress["qualified_prefixes"].items()},
                   "underflow_by_arm": {arm: {"prefixes_with_probability_underflow": sum(
                       case["underflow"][arm]["prefixes_with_probability_underflow"] for case in cases),
                       "maximum_underflow_cells": max(case["underflow"][arm]["maximum_underflow_cells"] for case in cases)}
                       for arm in ARMS},
                   "original_learned_pilot_opportunity": False, "new_quality_gate": None,
                   "initial_hit_weights": mixture, "all_prefixes": population(cases, mixture),
                   "after32_prefixes": population(cases, mixture, True), "storage_by_arm": storage,
                   "arm_roles": {arm: "compression candidate" if arm in CANDIDATES else "qualification only" if arm in QUALIFICATION
                                 else "reference" if arm == "full_bayes" else "recent-history control" for arm in ARMS},
                   "shared_model": model.storage_bytes(), "shared_model_array_sha256": model_pins,
                   "additional_readonly_planner_kernel_bytes": int(kernel.nbytes),
                   "decoder_return_arrays_bytes_per_arm": 53 * 53 * 8 * 2,
                   "planner_workspace": "Every proposal calls the same full-grid planner, allocating coordinate grids, distance/survival grids and four full-grid hit branches. Array payload counts are not per-arm allocator peak measurements.",
                   "mixture_weighted_per_case_timing_seconds": timing_summary,
                   "shared_model_initialization_seconds": shared_initialization_seconds,
                   "sum_measured_controller_seconds": total_measured_seconds,
                   "whole_elapsed_before_summary_seconds": (last - start) / 1e9,
                   "unattributed_elapsed_before_summary_seconds": (last - start) / 1e9 - total_measured_seconds,
                   "timing_scope": "One rotated pass of elapsed initialization/update/decode/planning on CPU, not a deployment speed benchmark. Whole time includes imports, authentication, trace loading, checks and output serialization.",
                   "recent_control_arithmetic": "Both recent controls use the declared retained-evidence definition with stable log reconstruction. No bitwise equivalence to upstream probability-product reconstruction is claimed.",
                   "numerical_thread_environment": {name: os.environ[name] for name in THREAD_ENV},
                   "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                               "scipy": scipy_version, "architecture": platform.machine()},
                   "weighting": "Original initial-hit weight/32 per case, renormalized over eligible cases or prefixes; all ranks and fills retained."}
        write(output / "summary.json", summary)
        source_pins(plan)
        require(sha(args.plan) == args.plan_sha256 and sha(__file__) == own_pin, "unchanged frozen study sources")
        for name in ("transitions.jsonl", "public-kernel.npz", "summary.json"):
            require(sha(run / name) == done["files"][name]["sha256"], "unchanged saved inputs")
        budget()
        receipt = {"status": "completed", "source_sha256": own_pin, "plan_sha256": args.plan_sha256,
                   "inputs": plan["inputs"], "scope": SCOPE, "progress": progress,
                   "environment_calls": 0, "model_calls": 0, "training_updates": 0,
                   "parent_original_wall_seconds": terminal["wall_seconds"], "clock_backend": clock.backend,
                   "started_ns": start, "finished_ns": last, "elapsed_ns": last - start, "wall_seconds": (last - start) / 1e9,
                   "limits": LIMITS, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if sys.platform == "darwin" else 1024),
                   "timing_scope": "Native start through pre-receipt check; final publication checked before return.",
                   "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}}
        write(output / "receipt.json", receipt)
        budget()
        print(json.dumps({"status": "completed", **progress}), flush=True)
        return receipt
    except BaseException as error:
        try:
            if (output / "receipt.json").exists():
                (output / "receipt.json").rename(output / "invalid-receipt.json")
            write(output / "failed.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                  "source_sha256": own_pin, "progress": progress, "scope": SCOPE,
                  "last_clock_elapsed_ns": None if last is None or start is None else last - start,
                  "timing_available": False, "wall_seconds": None})
        except BaseException as secondary:  # noqa: BLE001 - preserve the primary failure.
            error.add_note(f"Failure evidence publication also failed: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
