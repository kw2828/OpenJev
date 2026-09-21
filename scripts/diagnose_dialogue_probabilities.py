"""Saved-only, post hoc probability diagnosis of the complete V2 experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import resource
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-probability-diagnostic-v1"
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
TAILS = (1e-2, 1e-4, 1e-6)
BIN_LOWER = (0.0, 0.5, 0.7, 0.9, 0.95, 0.99)
CAPS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
AUDITOR = "output/dialogue-observation-learning-v2/result-audit-01/audit.py"
AUDITOR_PIN = "eb3c0839faf50741cd8175d40ee9631626402aed04cc0405d0151d8706b5dbef"
DIAGNOSTIC_SOURCES = {"scripts/diagnose_dialogue_probabilities.py",
                      "tests/test_diagnose_dialogue_probabilities.py",
                      "research/dialogue-probability-diagnostic-protocol.md"}
SCOPE = ("Post hoc description of all completed V2 predictions on exposed DEV. "
         "No temperature fitting, selection, model calls, actor-state changes, new predictions or TEST access. "
         "Original continuation remains failed: 6 of 7 conditions passed; no architecture or calibration advantage claimed.")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def describe(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def close(a, b):
    need(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), "Additive or baseline disagreement")


def temperature_logs(logs, temperature):
    need(math.isfinite(temperature) and temperature > 0, "Positive finite temperature")
    if temperature == 1:
        return logs.copy()
    scaled = logs.astype(np.float64) / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    centered = scaled - maximum
    return centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))


def row_stats(logs, target):
    logs = logs.astype(np.float64)
    choices = np.argmax(logs, axis=1)
    probability = np.exp(logs)
    target_logs = logs[np.arange(len(target)), target]
    brier = np.sum((probability - (np.arange(logs.shape[1])[None, :] == target[:, None]))**2, axis=1)
    return {"choice": choices, "correct": choices == target, "nll": -target_logs,
            "brier": brier, "confidence": probability[np.arange(len(target)), choices],
            "target_logs": target_logs}


def cell(stats, mask, denominator):
    count = int(mask.sum())
    total = math.fsum(float(x) for x in stats["nll"][mask])
    confidence = math.fsum(float(x) for x in stats["confidence"][mask])
    correct = int(np.count_nonzero(stats["correct"] & mask))
    return {"count": count, "correct": correct, "nll_sum": total,
            "nll_mean": total / count if count else None,
            "nll_contribution": total / denominator if denominator else None,
            "accuracy": correct / count if count else None,
            "mean_confidence": confidence / count if count else None,
            "confidence_minus_accuracy": (confidence-correct) / count if count else None}


def partition(stats, selected, masks):
    denominator = int(selected.sum())
    need(np.array_equal(sum(mask.astype(int) for mask in masks.values()), selected.astype(int)),
         "Exhaustive disjoint partition")
    result = {name: cell(stats, mask, denominator) for name, mask in masks.items()}
    need(sum(v["count"] for v in result.values()) == denominator, "Partition counts")
    close(math.fsum(v["nll_sum"] for v in result.values()), math.fsum(float(x) for x in stats["nll"][selected]))
    return result


def group_masks(info):
    groups = {}
    for panel in ("all", "seen", "unseen"):
        selected = np.ones(info["rows"], bool) if panel == "all" else info["unseen"] if panel == "unseen" else ~info["unseen"]
        groups[panel + "/micro"] = selected
        for stratum in ("unmentioned_retention", "assigned_retention", "changed"):
            groups[panel + "/" + stratum] = selected & (info["strata"] == stratum)
    return groups


def confidence_masks(confidence, selected):
    result = {}
    for i, lower in enumerate(BIN_LOWER):
        upper = BIN_LOWER[i+1] if i+1 < len(BIN_LOWER) else None
        name = f"[{lower},{upper if upper is not None else 'inf'})"
        result[name] = selected & (confidence >= lower) & ((confidence < upper) if upper is not None else True)
    return result


def fit_diagnostics(logs, info):
    stats = row_stats(logs, info["target"])
    groups, temperatures = {}, {}
    masks = group_masks(info)
    for name, selected in masks.items():
        denominator = int(selected.sum())
        groups[name] = {
            "raw": cell(stats, selected, denominator),
            "correctness": partition(stats, selected, {
                "correct": selected & stats["correct"], "incorrect": selected & ~stats["correct"]}),
            "confidence_bins": partition(stats, selected, confidence_masks(stats["confidence"], selected)),
            "confidence_above_one": int(np.count_nonzero(selected & (stats["confidence"] > 1))),
            "target_probability_tails": {str(t): cell(stats, selected & (stats["target_logs"] < math.log(t)), denominator) for t in TAILS}}
    for temperature in TEMPERATURES:
        scaled = temperature_logs(logs, temperature)
        supported = info["mask"]
        need(np.isfinite(scaled[supported]).all() and np.isneginf(scaled[~supported]).all(), "Transformed support")
        need(float(np.abs(np.exp(scaled).sum(axis=1)-1).max()) <= 2e-6, "Transformed mass")
        values = row_stats(scaled, info["target"])
        need(np.array_equal(values["choice"], stats["choice"]), "Temperature changed canonical choices")
        if temperature == 1:
            need(np.array_equal(scaled, logs) and np.array_equal(values["nll"], stats["nll"]), "T=1 identity")
        temperatures[str(temperature)] = {name: {
            "count": int(mask.sum()),
            "nll": math.fsum(float(x) for x in values["nll"][mask]) / int(mask.sum()) if mask.any() else None,
            "brier": math.fsum(float(x) for x in values["brier"][mask]) / int(mask.sum()) if mask.any() else None,
            "choices_unchanged": True} for name, mask in masks.items()}
    return {"groups": groups, "temperature_grid": temperatures, "temperature_selected": None}, stats


def paired_cell(control, treatment, mask, denominator):
    a, b = cell(control, mask, denominator), cell(treatment, mask, denominator)
    return {"count": a["count"], "control": a, "treatment": b,
            "signed_gap_sum": b["nll_sum"]-a["nll_sum"],
            "signed_gap_contribution": (b["nll_sum"]-a["nll_sum"]) / denominator if denominator else None}


def paired_diagnostics(control, treatment, info):
    result = {}
    for name, selected in group_masks(info).items():
        denominator = int(selected.sum())
        masks = {f"{a}_to_{b}": selected & (control["correct"] == a) & (treatment["correct"] == b)
                 for a in (False, True) for b in (False, True)}
        partition(control, selected, masks)
        partition(treatment, selected, masks)
        transitions = {label: paired_cell(control, treatment, mask, denominator) for label, mask in masks.items()}
        raw = paired_cell(control, treatment, selected, denominator)
        close(math.fsum(v["signed_gap_sum"] for v in transitions.values()), raw["signed_gap_sum"])
        tails = {}
        for threshold in TAILS:
            common = (control["target_logs"] < math.log(threshold)) | (treatment["target_logs"] < math.log(threshold))
            tail_masks = {"either_below": selected & common, "neither_below": selected & ~common}
            partition(control, selected, tail_masks)
            partition(treatment, selected, tail_masks)
            values = {label: paired_cell(control, treatment, mask, denominator) for label, mask in tail_masks.items()}
            close(math.fsum(v["signed_gap_sum"] for v in values.values()), raw["signed_gap_sum"])
            tails[str(threshold)] = values
        result[name] = {"raw": raw, "correctness_transitions": transitions, "common_target_probability_tails": tails}
    return result


def authenticate(plan):
    need(set(plan["sources"]) == DIAGNOSTIC_SOURCES, "Complete diagnostic source closure")
    request = plan["audit_request"]
    input_bindings = {
        str(Path(request["run"])/"completed.json"): request["completed_sha256"],
        request["plan"]: request["plan_sha256"],
        request["launch"]: request["launch_sha256"],
        request["terminal"]: request["terminal_sha256"],
        str(Path(request["report"])/"receipt.json"): request["report_receipt_sha256"],
        str(Path(request["report"])/"summary.json"): request["report_summary_sha256"],
        str(Path(plan["independent_audit"])/"receipt.json"): plan["independent_audit_receipt_sha256"],
        str(Path(plan["independent_audit"])/"summary.json"): plan["independent_audit_summary_sha256"],
        plan["actual_exit_review"]: plan["actual_exit_review_sha256"]}
    need(plan["inputs"] == input_bindings, "Complete exact external input bindings")
    need(plan["environment"] == {"python": platform.python_version(), "numpy": np.__version__}, "Runtime version identity")
    for name, pin in plan["sources"].items():
        need(sha(ROOT/name) == pin, "Diagnostic source changed: " + name)
    for name, pin in plan["inputs"].items():
        need(sha(ROOT/name) == pin, "Input pin: " + name)
    need(sha(ROOT/AUDITOR) == AUDITOR_PIN, "Qualified auditor source")
    spec = importlib.util.spec_from_file_location("observation_probability_input_audit", ROOT/AUDITOR)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    args = SimpleNamespace(**{k: ROOT/v if k in ("run", "plan", "launch", "terminal", "report") else v
                              for k, v in plan["audit_request"].items()})
    _, scientific_plan, _, published, _ = auditor.authenticate(args)
    need(published["continuation"]["passed"] is False and published["continuation"]["checks_passed"] == 6,
         "Preserved original continuation failure")
    directory = ROOT/plan["independent_audit"]
    receipt = read(directory/"receipt.json")
    need(receipt["status"] == "completed" and receipt["agreement"] is True
         and receipt["continuation_passed"] is False and receipt["scientific_checks_passed"] == 6
         and receipt["execution_completed_sha256"] == args.completed_sha256
         and receipt["plan_sha256"] == args.plan_sha256
         and receipt["producer_receipt_sha256"] == args.report_receipt_sha256
         and receipt["producer_summary_sha256"] == args.report_summary_sha256, "Successful independent audit binding")
    need(set(receipt["files"]) == {"started.json", "summary.json"}, "Independent audit closure")
    auditor.manifest(directory, receipt["files"], "receipt.json")
    need(receipt["files"]["summary.json"]["sha256"] == plan["independent_audit_summary_sha256"], "Independent summary pin join")
    review = read(ROOT/plan["actual_exit_review"])
    need(review["status"] == "completed" and review["independent_agreement"] is True
         and review["scientific_continuation_passed"] is False, "Successful actual reader exits")
    witnessed = {v["sha256"]: v for v in review["processes"]}
    for pin in (args.report_receipt_sha256, plan["independent_audit_receipt_sha256"]):
        need(pin in witnessed and witnessed[pin]["actual_exit_code"] == 0, "Actual matching reader exit")
    return auditor, args, scientific_plan, published


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value*1024


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    clock, deadline, handler, elapsed = None, None, None, None
    try:
        clock = SuspendClock()
        deadline = clock.deadline_after(CAPS["wall_seconds"])
        def timeout(*_):
            raise TimeoutError("Diagnostic wall cap")
        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
        def budget():
            nonlocal elapsed
            now = clock.now_ns()
            elapsed = now-deadline.started_ns
            need(now < deadline.expires_ns and peak_rss() <= CAPS["rss_bytes"], "Diagnostic time/RSS cap")
            need(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "Diagnostic output cap")
        need(sha(args.plan) == args.plan_sha256, "Diagnostic plan pin")
        plan = read(args.plan)
        need(plan["version"] == VERSION and plan["limits"] == CAPS
             and plan["temperatures"] == list(TEMPERATURES) and plan["tails"] == list(TAILS)
             and plan["confidence_lower_bounds"] == list(BIN_LOWER), "Fixed diagnostic recipe")
        write(args.out/"started.json", {"version": VERSION, "plan_sha256": args.plan_sha256,
              "sources": plan["sources"], "inputs": plan["inputs"], "limits": CAPS, "scope": SCOPE})
        auditor, prior, scientific_plan, published = authenticate(plan)
        budget()
        rows = [json.loads(line) for line in (prior.run/"evaluation-rows.jsonl").read_text().splitlines()]
        info = auditor.layout(rows)
        need(info["rows"] == 62329, "Complete canonical DEV endpoints")
        fits, retained, comparison = {}, {}, auditor.Comparison()
        for name in auditor.FIT_ORDER:
            with np.load(prior.run/name/"predictions.npz", allow_pickle=False) as data:
                need(set(data.files) == {"row_indices", "log_probs"}, "Prediction fields")
                logs = data["log_probs"]
                baseline = auditor.score(logs, data["row_indices"], info)
            comparison.check(baseline, published["fits"][name], name)
            fits[name], values = fit_diagnostics(logs.astype(np.float64), info)
            for group, selected in group_masks(info).items():
                panel, stratum = group.split("/")
                expected = baseline["panels"][panel]["micro"] if stratum == "micro" else baseline["panels"][panel]["strata"][stratum]
                observed = fits[name]["groups"][group]["raw"]
                need(observed["count"] == expected["count"] and observed["correct"] == expected["correct"], "Raw count baseline")
                if selected.any():
                    close(observed["nll_mean"], expected["nll"])
                    close(fits[name]["temperature_grid"]["1.0"][group]["brier"], expected["brier"])
            if name.startswith(("frozen_numbers", "trainable_numbers")):
                retained[name] = values
            budget()
        pairs = {str(seed): paired_diagnostics(retained[f"frozen_numbers-{seed}"], retained[f"trainable_numbers-{seed}"], info)
                 for seed in auditor.SEEDS}
        summary = {"version": VERSION, "status": "completed", "scope": SCOPE, "fits": fits,
                   "primary_pairs": pairs, "complete_fits": 12, "rows_per_fit": info["rows"],
                   "plan_sha256": args.plan_sha256, "temperature_grid": list(TEMPERATURES),
                   "temperature_selected": None, "original_continuation_passed": False,
                   "original_conditions_passed": 6, "original_conditions_total": 7,
                   "baseline_scalar_comparisons": comparison.count, "model_calls": 0}
        write(args.out/"summary.json", summary)
        for name, pin in plan["inputs"].items():
            need(sha(ROOT/name) == pin, "End input stability")
        for name, pin in {**scientific_plan["source_sha256"], **plan["sources"]}.items():
            need(sha(ROOT/name) == pin, "End source stability")
        auditor.manifest(prior.run, read(prior.run/"completed.json")["files"], "completed.json")
        budget()
        receipt = {"version": VERSION, "status": "completed", "scope": SCOPE, "plan_sha256": args.plan_sha256,
                   "source_sha256": plan["sources"], "input_sha256": plan["inputs"],
                   "files": {name: describe(args.out/name) for name in ("started.json", "summary.json")},
                   "limits": CAPS, "clock_backend": clock.backend, "started_ns": deadline.started_ns,
                   "finished_ns": deadline.started_ns+elapsed, "deadline_ns": deadline.expires_ns,
                   "elapsed_ns": elapsed, "wall_seconds": elapsed/1e9, "peak_rss_bytes": peak_rss(),
                   "model_calls": 0, "complete_fits": 12, "original_continuation_passed": False}
        write(args.out/"receipt.json", receipt)
        budget()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if (args.out/"receipt.json").exists():
            (args.out/"receipt.json").rename(args.out/"invalid-receipt.json")
        write(args.out/"failed.json", {"version": VERSION, "status": "failed", "error": repr(error),
              "elapsed_ns": None, "last_successful_elapsed_ns": elapsed, "scope": SCOPE,
              "model_calls": 0, "source": describe(Path(__file__))})
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({"status": result["status"], "complete_fits": result["complete_fits"], "model_calls": 0}))
