"""Post hoc saved-public-history decomposition of old positive and zero evidence.

No simulator, actor, policy, producer runner, environment or neural imports.
The pinned independent auditor supplies authentication and public analytic math.
Hypothetical decisions are never executed; the original failed gate is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

MODES = ("recent32", "old_positive", "old_zero", "full")
MEASURES = ("tv_to_full", "full_objective_excess", "matches_full_action")
PUBLIC = {"position", "hit", "done", "step", "valid_actions"}
AUDITOR = "scripts/audit_otto_large_memory.py"
AUDITOR_PIN = "1ad5a080f58177751dadf0eaab0cd9e2c7b37d7e9d84b50cc32f4046a66155ec"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SOURCES = {"scripts/diagnose_otto_memory_evidence.py", "tests/test_otto_memory_evidence.py",
           "research/otto-memory-evidence-protocol.md", AUDITOR, "src/openjev/research/suspend_clock.py"}
LIMITS = {"native_seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}
SCOPE = ("Post hoc public-history likelihood ablation on all 96 primary space_full paths and their recent32 pairs. "
         "All full decisions require saved-score/action parity; four-mode summaries use t>32 and first shared-history "
         "divergences. Other actions are teacher-forced proposals, not executed controls. Heuristic score excess is "
         "dimensionless, not realized search regret. No new pass condition or reversal of the original failure.")


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
            and plan["sources"]["src/openjev/research/suspend_clock.py"] == CLOCK_PIN, "exact five-source closure")
    for name, pin in plan["sources"].items():
        require(not (ROOT / name).is_symlink() and sha(ROOT / name) == pin, f"diagnostic source: {name}")


def authenticate(plan, analytic, budget):
    require(plan["status"] == "frozen_before_saved_diagnostic" and plan["limits"] == LIMITS, "frozen diagnostic limits")
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


def evidence_beliefs(initial_public, completed_public, kernel, analytic):
    """Inputs are public packets and public likelihood math, never evaluator fields."""
    initial_position = validate_public(initial_public, 0, analytic)
    require(initial_position == analytic.CENTER and initial_public["hit"] in (1, 2, 3), "one conditioned initial positive hit")
    require(kernel.shape == (analytic.NHITS, 2 * analytic.N + 1, 2 * analytic.N + 1)
            and np.isfinite(kernel).all() and np.all(kernel >= 0), "public kernel support")
    visited, history, previous = {initial_position}, [], initial_position
    for index, packet in enumerate(completed_public, 1):
        position = validate_public(packet, index, analytic)
        require(sum(abs(a - b) for a, b in zip(position, previous, strict=True)) <= 1, "public movement continuity")
        visited.add(position)
        history.append((position, packet["hit"]))
        previous = position
    count = len(history)
    old_count = max(0, count - 32)
    initial = analytic.prior(kernel, initial_public["hit"])
    for position in visited:
        initial[position] = 0
    initial = analytic.normalized(initial)
    beliefs = {}
    for mode in MODES:
        p = initial.copy()
        for index, (position, hit) in enumerate(history):
            retain = index >= old_count or mode == "full" or mode == "old_positive" and hit > 0 or mode == "old_zero" and hit == 0
            if retain:
                p = analytic.normalized(p * analytic.likelihood_at(kernel, position)[hit])
        require(np.isfinite(p).all() and np.all(p >= 0) and abs(float(p.sum()) - 1) <= 1e-10
                and all(p[position] == 0 for position in visited), "normalized belief and shared exclusion mask")
        beliefs[mode] = p
    last_positive = max([0, *(j for j, (_, hit) in enumerate(history, 1) if hit > 0)])
    return beliefs, {"completed_steps": count, "old_positive_count": sum(hit > 0 for _, hit in history[:old_count]),
                     "old_zero_count": sum(hit == 0 for _, hit in history[:old_count]),
                     "moves_since_last_positive": count - last_positive, "visited_cells": len(visited)}


def choose(scores):
    require(isinstance(scores, list) and len(scores) == 4, "four scores")
    valid = [a for a, score in enumerate(scores) if score is not None]
    require(valid and all(math.isfinite(scores[a]) for a in valid), "finite valid scores")
    best = min(scores[a] for a in valid)
    ties = [a for a in valid if abs(scores[a] - best) < 1e-10]
    return ties[0], len(ties)


def factorial(values):
    r, p, z, f = (values[mode] for mode in MODES)
    return {"positive_without_old_zero": p - r, "positive_with_old_zero": f - z,
            "zero_without_old_positive": z - r, "zero_with_old_positive": f - p,
            "interaction": f - p - z + r}


def describe(beliefs, metadata, position, kernel, analytic):
    scores = {mode: analytic.action_scores(p, position, kernel, True) for mode, p in beliefs.items()}
    choices = {mode: choose(values) for mode, values in scores.items()}
    full_scores = scores["full"]
    full_best = min(x for x in full_scores if x is not None)
    modes = {}
    for mode in MODES:
        action, ties = choices[mode]
        modes[mode] = {"action": action, "scores": scores[mode], "tie_count": ties,
                       "tv_to_full": float(np.abs(beliefs[mode] - beliefs["full"]).sum() / 2),
                       "full_objective_excess": full_scores[action] - full_best,
                       "matches_full_action": action == choices["full"][0]}
    contrasts = {metric: factorial({mode: float(modes[mode][metric]) for mode in MODES}) for metric in MEASURES}
    return {**metadata, "modes": modes, "factorial": contrasts}


def parity(description, event):
    saved = event["scores"]
    scores = description["modes"]["full"]["scores"]
    require(len(saved) == 4 and all((a is None) == (b is None) for a, b in zip(saved, scores, strict=True)), "saved score support")
    error = max(abs(a - b) for a, b in zip(saved, scores, strict=True) if a is not None)
    require(math.isfinite(error) and error <= 1e-8, "full saved score parity")
    require(choose(saved)[0] == event["action"] == description["modes"]["full"]["action"], "exact full selected-action parity")
    return error


def first_disagreement(full, recent):
    require(full[0]["public"] == recent[0]["public"], "same public initialization")
    for index, (left, right) in enumerate(zip(full[1:], recent[1:]), 1):
        require(left["public"]["step"] == right["public"]["step"] == index, "paired step chronology")
        if left["action"] != right["action"]:
            return index - 1
        require(left["public"] == right["public"], "identical public history before first differing action")
    require(len(full) == len(recent), "same actions cannot produce unmatched paired termination")
    return None


def metrics(description):
    values = {f"{mode}.{metric}": float(description["modes"][mode][metric]) for mode in MODES for metric in MEASURES}
    values.update({f"factorial.{metric}.{name}": value for metric, contrasts in description["factorial"].items()
                   for name, value in contrasts.items()})
    return values


def means(records, weights=None):
    if not records:
        return None
    weights = [1.0] * len(records) if weights is None else weights
    require(len(weights) == len(records) and all(math.isfinite(w) and w > 0 for w in weights), "positive summary weights")
    keys = set(records[0])
    require(all(set(record) == keys for record in records), "metric key consistency")
    denominator = math.fsum(weights)
    return {key: math.fsum(w * record[key] for w, record in zip(weights, records, strict=True)) / denominator for key in sorted(keys)}


def population(cases, mixture, first=False):
    key = "first_disagreement" if first else "prefix_means"
    eligible = [case for case in cases if case[key] is not None]
    records = [metrics(case[key]) if first else case[key] for case in eligible]
    case_weights = [mixture[case["initial_hit"]] / 32 for case in eligible]
    counts = [1 if first else case["eligible_prefixes"] for case in eligible]
    per_stratum = {str(h): {"all_cases": sum(c["initial_hit"] == h for c in cases),
                           "eligible_cases": sum(c["initial_hit"] == h for c in eligible),
                           "eligible_prefixes": sum(n for c, n in zip(eligible, counts, strict=True) if c["initial_hit"] == h)}
                    for h in (1, 2, 3)}
    result = {"all_cases": len(cases), "eligible_cases": len(eligible), "eligible_prefixes": sum(counts),
              "by_initial_hit": per_stratum, "eligible_case_mixture_mass": math.fsum(case_weights),
              "unweighted_eligible_case_means": means(records),
              "stratum_weighted_eligible_case_means": means(records, case_weights),
              "stratum_weighted_eligible_prefix_means": means(records, [w * n for w, n in zip(case_weights, counts, strict=True)])}
    if first:
        labels = ("positive_only", "zero_only", "either_alone", "joint_only")
        result["action_recovery_counts"] = {label: sum(c["recovery"] == label for c in eligible) for label in labels}
        total = math.fsum(case_weights)
        result["action_recovery_conditional_mixture_rates"] = None if not total else {
            label: math.fsum(w for c, w in zip(eligible, case_weights, strict=True) if c["recovery"] == label) / total for label in labels}
    return result


def public_traces(run, analytic, budget):
    selected = {}
    groups = analytic.grouped(run / "transitions.jsonl")
    for case in range(96):
        shift = case % 8
        for arm in analytic.ARMS[shift:] + analytic.ARMS[:shift]:
            key, rows = next(groups, (None, None))
            require(key == (610001 + case, arm), "exact 768-case trace rotation")
            require(rows and rows[0]["kind"] == "reset" and all(r["kind"] == "step" for r in rows[1:]), "trace row kinds")
            if arm in ("space_full", "space_recent32"):
                # Explicit projection excludes source, seed streams, native posterior and costs from actor reconstruction.
                trace = [{"public": rows[0]["public"]}]
                for i, row in enumerate(rows[1:], 1):
                    require(row["step"] == row["public"]["step"] == i, "source public time join")
                    trace.append({"public": row["public"], "action": row["action"], "scores": row["scores"]})
                selected[key] = trace
            budget()
    require(next(groups, None) is None and len(selected) == 192, "complete selected public pairs")
    return selected


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = last = None
    progress = {"completed_cases": 0, "parity_decisions": 0, "eligible_prefixes": 0, "maximum_score_error": 0.0}
    own_pin = sha(__file__)
    try:
        require(sha(ROOT / "src/openjev/research/suspend_clock.py") == CLOCK_PIN, "clock source")
        clock = SuspendClock()
        start = clock.now_ns()
        deadline = start + LIMITS["native_seconds"] * 10**9

        def budget():
            nonlocal last
            last = clock.now_ns()
            require(last < deadline, "native diagnostic deadline")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
                    <= LIMITS["rss_bytes"], "diagnostic RSS cap")
            require(sum(p.stat().st_size for p in output.iterdir()) <= LIMITS["output_bytes"], "diagnostic output cap")

        write(output / "started.json", {"source_sha256": own_pin, "request": {k: str(v) for k, v in vars(args).items()},
              "clock_backend": clock.backend, "started_ns": start, "deadline_ns": deadline, "scope": SCOPE})
        require(sha(args.plan) == args.plan_sha256, "external diagnostic plan pin")
        plan = read(args.plan)
        source_pins(plan)
        analytic = load_analytic()
        run, _original_plan, done, terminal = authenticate(plan, analytic, budget)
        with np.load(run / "public-kernel.npz", allow_pickle=False) as saved:
            kernel = saved["likelihood"].copy()
            mixture = {h: float(saved["initial_hit_weights"][h]) for h in (1, 2, 3)}
        traces = public_traces(run, analytic, budget)
        cases = []
        with (output / "prefixes.jsonl").open("x") as prefix_file, (output / "cases.jsonl").open("x") as case_file:
            for case in range(96):
                seed, hit, block = 610001 + case, 1 + (case % 12) // 4, case // 12
                full, recent = traces[seed, "space_full"], traces[seed, "space_recent32"]
                require(full[0]["public"]["hit"] == recent[0]["public"]["hit"] == hit, "fixed initial stratum")
                divergence = first_disagreement(full, recent)
                require(divergence is None or divergence > 32, "no old-evidence divergence before first discarded observation")
                completed, selected, first = [], [], None
                max_error = 0.0
                for t, event in enumerate(full[1:]):
                    budget()
                    beliefs, metadata = evidence_beliefs(full[0]["public"], completed, kernel, analytic)
                    position = tuple((completed[-1] if completed else full[0]["public"])["position"])
                    description = describe(beliefs, metadata, position, kernel, analytic)
                    error = parity(description, event)
                    max_error = max(max_error, error)
                    progress["maximum_score_error"] = max(progress["maximum_score_error"], error)
                    progress["parity_decisions"] += 1
                    if t > 32:
                        row = {"seed": seed, "block": block, "initial_hit": hit, **description}
                        prefix_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        selected.append(metrics(description))
                        progress["eligible_prefixes"] += 1
                    if t == divergence:
                        first = description
                        recent_description = {"modes": {"full": description["modes"]["recent32"]}}
                        parity(recent_description, recent[t + 1])
                    packet = event["public"]
                    if packet["done"]:
                        require(t + 1 == len(full) - 1 and packet["hit"] == -2, "terminal cannot enter next decision")
                    else:
                        completed.append(packet)
                require(divergence is None or first is not None, "first disagreement represented")
                recovery = None
                if first is not None:
                    p, z = (first["modes"][m]["matches_full_action"] for m in ("old_positive", "old_zero"))
                    recovery = "either_alone" if p and z else "positive_only" if p else "zero_only" if z else "joint_only"
                row = {"seed": seed, "block": block, "initial_hit": hit, "full_decisions": len(full) - 1,
                       "recent32_decisions": len(recent) - 1, "eligible_prefixes": len(selected),
                       "maximum_full_score_error": max_error, "prefix_means": means(selected),
                       "first_disagreement": first, "recovery": recovery}
                cases.append(row)
                case_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                case_file.flush()
                prefix_file.flush()
                progress["completed_cases"] += 1
        require(progress["completed_cases"] == 96 and progress["parity_decisions"] == sum(c["full_decisions"] for c in cases)
                and progress["eligible_prefixes"] == sum(c["eligible_prefixes"] for c in cases), "complete case/decision accounting")
        summary = {"scope": SCOPE, "original_learned_pilot_opportunity": False, "progress": progress,
                   "initial_hit_weights": mixture, "first_disagreement": population(cases, mixture, True),
                   "all_full_path_prefixes": population(cases, mixture),
                   "interpretation": {"old_positive": "recent32 plus all old categories 1,2,3; category3 is saturated >=3",
                      "old_zero": "recent32 plus all old category0", "full_objective_excess": "full-belief score of chosen action minus minimum full-belief score; dimensionless heuristic",
                      "factorial": "differences in diagnostic metrics, not episode return attribution",
                      "weights": "case weight=initial-hit mixture/32, renormalized over eligible cases or prefixes; no imputation",
                      "joint_only": "neither single old category reproduces the full action at this prefix; not a global necessity claim"}}
        write(output / "summary.json", summary)
        source_pins(plan)
        require(sha(args.plan) == args.plan_sha256 and sha(__file__) == own_pin, "unchanged analysis sources")
        for name in ("transitions.jsonl", "public-kernel.npz", "summary.json"):
            require(sha(run / name) == done["files"][name]["sha256"], "unchanged diagnostic inputs")
        budget()
        receipt = {"status": "completed", "source_sha256": own_pin, "plan_sha256": args.plan_sha256, "inputs": plan["inputs"],
                   "scope": SCOPE, "progress": progress, "environment_calls": 0, "model_calls": 0,
                   "parent_original_wall_seconds": terminal["wall_seconds"], "clock_backend": clock.backend,
                   "started_ns": start, "finished_ns": last, "elapsed_ns": last - start, "wall_seconds": (last - start) / 1e9,
                   "limits": LIMITS, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   "timing_scope": "Native start through last pre-receipt check; final publication rechecked before return.",
                   "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}}
        write(output / "receipt.json", receipt)
        budget()
        print(json.dumps({"status": "completed", **progress}), flush=True)
    except BaseException as exc:
        try:
            if (output / "receipt.json").exists():
                (output / "receipt.json").rename(output / "invalid-receipt.json")
            write(output / "failed.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                  "source_sha256": own_pin, "progress": progress, "last_clock_elapsed_ns": None if last is None else last - start,
                  "timing_available": False, "wall_seconds": None})
        except BaseException as publication_error:  # noqa: BLE001 - preserve original error
            if hasattr(exc, "add_note"):
                exc.add_note(f"Failure evidence publication also failed: {publication_error!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())
