"""One-shot independent arithmetic audit of the complete saved probability diagnosis.

Only authentication and canonical layout are inherited. No diagnostic arithmetic,
model, training, tokenizer or checkpoint-deserialization functions are called.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[3]
VERSION = "dialogue-probability-independent-audit-v1"
PRODUCER = "scripts/diagnose_dialogue_probabilities.py"
PRODUCER_PIN = "a99c68b2101f299d1118bb9c69f568578d46519ec6f0387b18576263a7f8d78f"
CLOCK = "src/openjev/research/suspend_clock.py"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
FITS = tuple(f"{arm}-{seed}" for seed in SEEDS for arm in ARMS)
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
TAILS = (0.01, 0.0001, 0.000001)
EDGES = (0.0, 0.5, 0.7, 0.9, 0.95, 0.99)
CAPS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
SCOPE = (
    "All numerical fields in all12 fit diagnostics, all12 groups perfit, all8 temperatures, "
    "and all3 primary paired decompositions independently reconstructed from raw saved float32 logs. "
    "Qualified producer.authenticate and qualified auditor.layout are inherited, including full artifact/source "
    "authentication and canonical metadata semantics; diagnostic arithmetic is not imported or called. "
    "No independent replay of training, encoder/state witnesses, raw corpus, inference or terminal clocks. "
    "The original seven-condition FAIL remains unchanged; no temperature selection or calibration claim."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def descriptor(path):
    return {"sha256": digest(path), "bytes": Path(path).stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def rss():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return size if sys.platform == "darwin" else size * 1024


class Compare:
    def __init__(self):
        self.counts = {"integer": 0, "float": 0, "boolean": 0, "none": 0, "string": 0}
        self.sections = {}
        self.maximum_float_absolute_difference = 0.0

    @property
    def total(self):
        return sum(self.counts.values())

    def check(self, expected, saved, where):
        if isinstance(expected, dict):
            require(type(saved) is dict and set(expected) == set(saved), "Field membership: " + where)
            for key in expected:
                self.check(expected[key], saved[key], where + "/" + key)
        elif isinstance(expected, list):
            require(type(saved) is list and len(saved) == len(expected), "List membership: " + where)
            for i, value in enumerate(expected):
                self.check(value, saved[i], where + "/" + str(i))
        else:
            category = {int: "integer", float: "float", bool: "boolean", type(None): "none", str: "string"}[type(expected)]
            self.counts[category] += 1
            section = where.split("/")[0]
            self.sections[section] = self.sections.get(section, 0) + 1
            if category == "float":
                require(type(saved) in (int, float) and math.isfinite(saved) and math.isfinite(expected), "Finite scalar: " + where)
                delta = abs(expected - saved)
                self.maximum_float_absolute_difference = max(self.maximum_float_absolute_difference, delta)
                require(math.isclose(expected, saved, rel_tol=1e-12, abs_tol=1e-12), f"Scalar disagreement {where}: {expected!r} != {saved!r}")
            else:
                require(type(expected) is type(saved) and expected == saved, f"Exact disagreement {where}: {expected!r} != {saved!r}")


def total(values):
    return math.fsum(map(float, values))


def select_baseline_fields(expected, observed):
    """The original report also has out-of-scope type/service fields."""
    if isinstance(expected, dict):
        require(type(observed) is dict and set(expected) <= set(observed), "Original baseline fields")
        return {key: select_baseline_fields(value, observed[key]) for key, value in expected.items()}
    return observed


def vectors(logs, targets):
    """Candidate-sum Brier independently formed as sum(p²)-2p_y+1."""
    probability = np.exp(logs)
    choice = np.argmax(logs, axis=1)
    index = np.arange(len(targets))
    return {"choice": choice, "correct": choice == targets, "nll": -logs[index, targets],
            "confidence": probability[index, choice],
            "brier": np.einsum("ij,ij->i", probability, probability) - 2 * probability[index, targets] + 1}


def selections(info):
    output = {}
    for panel, selected in (("all", np.ones(info["rows"], dtype=bool)), ("seen", ~info["unseen"]), ("unseen", info["unseen"])):
        output[panel + "/micro"] = selected
        for stratum in STRATA:
            output[panel + "/" + stratum] = selected & np.equal(info["strata"], stratum)
    return output


def describe_cell(v, selected, full_count):
    n = int(np.count_nonzero(selected))
    hits = int(np.count_nonzero(v["correct"][selected]))
    nll_sum = total(v["nll"][selected])
    confidence_sum = total(v["confidence"][selected])
    return {"count": n, "correct": hits, "nll_sum": nll_sum,
            "nll_mean": nll_sum / n if n else None,
            "nll_contribution": nll_sum / full_count if full_count else None,
            "accuracy": hits / n if n else None,
            "mean_confidence": confidence_sum / n if n else None,
            "confidence_minus_accuracy": confidence_sum / n - hits / n if n else None}


def exhaustive(v, selected, parts):
    coverage = np.zeros(len(selected), dtype=np.int64)
    for part in parts.values():
        coverage += part
    require(np.array_equal(coverage, selected.astype(np.int64)), "Independent disjoint/exhaustive partition")
    n = int(selected.sum())
    cells = {key: describe_cell(v, part, n) for key, part in parts.items()}
    require(sum(c["count"] for c in cells.values()) == n, "Independent partition count")
    require(math.isclose(total(c["nll_sum"] for c in cells.values()), total(v["nll"][selected]), rel_tol=1e-12, abs_tol=1e-12), "Independent NLL partition identity")
    return cells


def new_temperature(logs, temperature):
    if temperature == 1.0:
        return logs.copy()
    scaled = logs / temperature
    return scaled - np.logaddexp.reduce(scaled, axis=1)[:, None]


def raw_baseline(logs, info, v):
    groups = selections(info)
    result = {}
    def score(mask):
        n = int(mask.sum())
        correct = int(v["correct"][mask].sum())
        return {"count": n, "correct": correct, "incorrect": n - correct,
                "accuracy": correct / n if n else None, "error": (n-correct) / n if n else None,
                "nll": total(v["nll"][mask]) / n if n else None,
                "brier": total(v["brier"][mask]) / n if n else None}
    for panel in ("all", "seen", "unseen"):
        strata = {s: score(groups[panel + "/" + s]) for s in STRATA}
        result[panel] = {"micro": score(groups[panel + "/micro"]), "strata": strata,
                         "macro_three": {metric: total(strata[s][metric] for s in STRATA) / 3
                                         if all(strata[s]["count"] for s in STRATA) else None
                                         for metric in ("accuracy", "nll", "brier")}}
    p = np.exp(logs)
    return {"panels": result, "validation": {
        "rows": info["rows"], "supported_candidate_positions": int(info["mask"].sum()),
        "maximum_mass_error": float(abs(p.sum(axis=1)-1).max()), "tolerance": 2e-6,
        "exact_top1_tie_rows": int(((logs == logs.max(axis=1)[:, None]).sum(axis=1) > 1).sum()),
        "float64_exponent_underflow_positions": int((p[info["mask"]] == 0).sum())}}


def fit_result(logs, info, check_budget):
    v = vectors(logs, info["target"])
    masks = selections(info)
    groups, grid = {}, {}
    bins = np.searchsorted(np.array(EDGES), v["confidence"], side="right") - 1
    for name, selected in masks.items():
        n = int(selected.sum())
        bin_masks = {f"[{lower},{EDGES[i+1] if i+1 < len(EDGES) else 'inf'})": selected & (bins == i)
                     for i, lower in enumerate(EDGES)}
        groups[name] = {"raw": describe_cell(v, selected, n),
                        "correctness": exhaustive(v, selected, {"correct": selected & v["correct"], "incorrect": selected & ~v["correct"]}),
                        "confidence_bins": exhaustive(v, selected, bin_masks),
                        "confidence_above_one": int((selected & (v["confidence"] > 1)).sum()),
                        "target_probability_tails": {str(t): describe_cell(v, selected & (v["nll"] > -math.log(t)), n) for t in TAILS}}
    for t in TEMPERATURES:
        adjusted = new_temperature(logs, t)
        require(np.isfinite(adjusted[info["mask"]]).all() and np.isneginf(adjusted[~info["mask"]]).all(), "Temperature support/padding")
        require(float(abs(np.exp(adjusted).sum(axis=1)-1).max()) <= 2e-6, "Temperature mass tolerance")
        values = vectors(adjusted, info["target"])
        require(np.array_equal(values["choice"], v["choice"]), "First-argmax preservation")
        if t == 1.0:
            require(np.array_equal(adjusted, logs) and np.array_equal(values["nll"], v["nll"]) and np.array_equal(values["brier"], v["brier"]), "Exact raw T1 identity")
        grid[str(t)] = {name: {"count": int(mask.sum()),
                              "nll": total(values["nll"][mask]) / int(mask.sum()) if mask.any() else None,
                              "brier": total(values["brier"][mask]) / int(mask.sum()) if mask.any() else None,
                              "choices_unchanged": True} for name, mask in masks.items()}
        check_budget()
    return {"groups": groups, "temperature_grid": grid, "temperature_selected": None}, v


def paired_result(a, b, info):
    output = {}
    for name, selected in selections(info).items():
        n = int(selected.sum())
        def paired(mask):
            # Sum pointwise raw-log differences, rather than subtract aggregate totals.
            gap = total((b["nll"] - a["nll"])[mask])
            return {"count": int(mask.sum()), "control": describe_cell(a, mask, n), "treatment": describe_cell(b, mask, n),
                    "signed_gap_sum": gap, "signed_gap_contribution": gap / n if n else None}
        def partition(parts):
            exhaustive(a, selected, parts)
            exhaustive(b, selected, parts)
            result = {key: paired(mask) for key, mask in parts.items()}
            require(math.isclose(total(x["signed_gap_sum"] for x in result.values()), paired(selected)["signed_gap_sum"], rel_tol=1e-12, abs_tol=1e-12), "Independent paired additive identity")
            return result
        transitions = {f"{bool(i)}_to_{bool(j)}": selected & (a["correct"] == i) & (b["correct"] == j) for i in (0, 1) for j in (0, 1)}
        tails = {}
        for threshold in TAILS:
            either = np.maximum(a["nll"], b["nll"]) > -math.log(threshold)
            tails[str(threshold)] = partition({"either_below": selected & either, "neither_below": selected & ~either})
        output[name] = {"raw": paired(selected), "correctness_transitions": partition(transitions), "common_target_probability_tails": tails}
    return output


def manual_fixtures():
    # Closed-form equal candidates, first tie, exact Brier, empty support and tail underflow.
    logs = np.array([[math.log(0.5), math.log(0.5), -np.inf], [0.0, -1000.0, -np.inf]], dtype=np.float64)
    v = vectors(logs, np.array([0, 1]))
    require(v["choice"].tolist() == [0, 0] and v["correct"].tolist() == [True, False], "Fixture first tie")
    require(v["brier"].tolist() == [0.5, 2.0] and v["nll"][1] == 1000, "Fixture Brier/underflow NLL")
    require(np.array_equal(new_temperature(logs, 1), logs), "Fixture identity")
    require(np.isclose(new_temperature(logs, 2)[1, 1], -500), "Fixture temperature in log space")
    empty = describe_cell(v, np.array([False, False]), 0)
    require(empty["count"] == 0 and empty["nll_sum"] == 0 and empty["nll_mean"] is None and empty["nll_contribution"] is None, "Fixture empty denominator")
    require(np.searchsorted(np.array(EDGES), np.array([0.5, 0.99, 1.000001]), side="right").tolist() == [2, 6, 6], "Fixture inclusive bins/raw overshoot")
    require((v["nll"] > -math.log(1e-6)).tolist() == [False, True], "Fixture strict tail")
    return 7


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    clock, deadline, old_handler, elapsed = None, None, None, None
    comparisons = Compare()
    try:
        clock = SuspendClock()
        deadline = clock.deadline_after(CAPS["wall_seconds"])
        def alarm(*_):
            raise TimeoutError("Independent audit awake emergency timer")
        old_handler = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
        def budget():
            nonlocal elapsed
            now = clock.now_ns()
            elapsed = now-deadline.started_ns
            require(now < deadline.expires_ns, "Independent audit native deadline")
            require(rss() <= CAPS["rss_bytes"], "Independent audit RSS cap")
            require(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "Independent audit output cap")
        request = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        source_hashes = {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__), CLOCK: digest(ROOT/CLOCK)}
        write(args.out/"started.json", {"version": VERSION, "request": request, "limits": CAPS, "source_sha256": source_hashes,
              "started_utc": datetime.now(timezone.utc).isoformat(), "clock_backend": clock.backend,
              "started_ns": deadline.started_ns, "deadline_ns": deadline.expires_ns,
              "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}, "scope": SCOPE})
        fixtures = manual_fixtures()
        require(digest(args.plan) == args.plan_sha256 and digest(args.diagnostic/"receipt.json") == args.receipt_sha256
                and digest(args.diagnostic/"summary.json") == args.summary_sha256, "Externally pinned diagnostic inputs")
        plan, receipt = read(args.plan), read(args.diagnostic/"receipt.json")
        require(plan["temperatures"] == list(TEMPERATURES) and plan["tails"] == list(TAILS)
                and plan["confidence_lower_bounds"] == list(EDGES) and plan["limits"] == CAPS, "Fixed independent recipe")
        require(digest(ROOT/PRODUCER) == PRODUCER_PIN == plan["sources"][PRODUCER], "Authentication-only producer source")
        spec = importlib.util.spec_from_file_location("probability_authentication_only", ROOT/PRODUCER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        auditor, inherited, scientific_plan, original = module.authenticate(plan)
        require(receipt["status"] == "completed" and receipt["version"] == plan["version"]
                and receipt["plan_sha256"] == args.plan_sha256 and receipt["source_sha256"] == plan["sources"]
                and receipt["input_sha256"] == plan["inputs"] and receipt["limits"] == CAPS
                and receipt["complete_fits"] == 12 and receipt["model_calls"] == 0
                and receipt["original_continuation_passed"] is False, "Successful diagnostic identity")
        require(set(receipt["files"]) == {"started.json", "summary.json"}, "Diagnostic file membership")
        auditor.manifest(args.diagnostic, receipt["files"], "receipt.json")
        require(receipt["files"]["summary.json"]["sha256"] == args.summary_sha256, "Summary receipt join")
        require(all(type(receipt[k]) is int for k in ("started_ns", "finished_ns", "deadline_ns", "elapsed_ns"))
                and receipt["started_ns"] <= receipt["finished_ns"] < receipt["deadline_ns"]
                and receipt["deadline_ns"] - receipt["started_ns"] == CAPS["wall_seconds"] * 10**9
                and receipt["elapsed_ns"] == receipt["finished_ns"] - receipt["started_ns"]
                and receipt["wall_seconds"] == receipt["elapsed_ns"] / 10**9
                and receipt["peak_rss_bytes"] <= CAPS["rss_bytes"]
                and sum(v["bytes"] for v in receipt["files"].values()) + (args.diagnostic/"receipt.json").stat().st_size <= CAPS["output_bytes"], "Recorded diagnostic bounds")
        published = read(args.diagnostic/"summary.json")
        expected_top = {"version", "status", "scope", "fits", "primary_pairs", "complete_fits", "rows_per_fit", "plan_sha256",
                        "temperature_grid", "temperature_selected", "original_continuation_passed", "original_conditions_passed",
                        "original_conditions_total", "baseline_scalar_comparisons", "model_calls"}
        require(set(published) == expected_top and set(published["fits"]) == set(FITS)
                and set(published["primary_pairs"]) == {str(s) for s in SEEDS}, "Complete exact diagnostic summary")
        budget()
        rows = [json.loads(line) for line in (inherited.run/"evaluation-rows.jsonl").read_text().splitlines()]
        info = auditor.layout(rows)
        require(info["rows"] == 62329, "Full canonical cohort")
        retained, compact, baseline_checks = {}, {}, 0
        for fit in FITS:
            with np.load(inherited.run/fit/"predictions.npz", allow_pickle=False) as data:
                require(set(data.files) == {"row_indices", "log_probs"}, "NPZ membership")
                raw, ids = data["log_probs"], data["row_indices"]
                require(raw.dtype == np.float32 and raw.shape == (info["rows"], 12), "Raw log format")
                require(ids.dtype == np.int64 and np.array_equal(ids, np.arange(info["rows"])), "Canonical row alignment")
            require(np.isfinite(raw[info["mask"]]).all() and np.isneginf(raw[~info["mask"]]).all(), "Raw support/padding")
            logs = raw.astype(np.float64)
            require(float(abs(np.exp(logs).sum(axis=1)-1).max()) <= 2e-6, "Raw unmodified mass")
            expected, v = fit_result(logs, info, budget)
            comparisons.check(expected, published["fits"][fit], "diagnostic/" + fit)
            baseline = raw_baseline(logs, info, v)
            before = comparisons.total
            comparisons.check(baseline, select_baseline_fields(baseline, original["fits"][fit]), "baseline/" + fit)
            baseline_checks += comparisons.total - before
            compact[fit] = {"unseen_raw": expected["groups"]["unseen/micro"]["raw"],
                            "unseen_temperature_grid": {t: values["unseen/micro"] for t, values in expected["temperature_grid"].items()}}
            if fit.startswith(("frozen_numbers-", "trainable_numbers-")):
                retained[fit] = v
            budget()
        primary = {}
        for seed in SEEDS:
            value = paired_result(retained[f"frozen_numbers-{seed}"], retained[f"trainable_numbers-{seed}"], info)
            comparisons.check(value, published["primary_pairs"][str(seed)], "pairs/" + str(seed))
            primary[str(seed)] = value["unseen/micro"]
            budget()
        metadata = {"version": plan["version"], "status": "completed", "scope": receipt["scope"], "complete_fits": 12,
                    "rows_per_fit": info["rows"], "plan_sha256": args.plan_sha256, "temperature_grid": list(TEMPERATURES),
                    "temperature_selected": None, "original_continuation_passed": False,
                    "original_conditions_passed": 6, "original_conditions_total": 7,
                    "baseline_scalar_comparisons": baseline_checks, "model_calls": 0}
        comparisons.check(metadata, {key: published[key] for key in metadata}, "metadata")
        for path, pin in {**plan["sources"], **scientific_plan["source_sha256"], **plan["inputs"], **source_hashes}.items():
            require(digest(ROOT/path) == pin, "End source/input identity: " + path)
        require(digest(args.plan) == args.plan_sha256 and digest(args.diagnostic/"receipt.json") == args.receipt_sha256
                and digest(args.diagnostic/"summary.json") == args.summary_sha256, "End external identity")
        auditor.manifest(inherited.run, read(inherited.run/"completed.json")["files"], "completed.json")
        result = {"version": VERSION, "status": "completed", "agreement": True, "scope": SCOPE,
                  "checks": {"scalar_checks": comparisons.total, "scalar_types": comparisons.counts, "sections": comparisons.sections,
                             "maximum_float_absolute_difference": comparisons.maximum_float_absolute_difference,
                             "relative_tolerance": 1e-12, "absolute_tolerance": 1e-12,
                             "manual_closed_form_fixtures": fixtures, "fits": 12, "groups_per_fit": 12,
                             "fit_group_cells": 144, "confidence_bin_cells": 864, "correctness_cells": 288,
                             "target_tail_cells": 432, "temperature_group_cells": 1152,
                             "paired_groups": 36, "paired_correctness_cells": 144, "paired_common_tail_cells": 216,
                             "paired_raw_cells": 36, "raw_baseline_scalar_checks": baseline_checks},
                  "unseen_fit_reconstruction": compact, "primary_unseen_reconstruction": primary,
                  "temperature_selected": None, "original_continuation_passed": False,
                  "model_calls": 0, "prediction_arrays_read": 12}
        write(args.out/"summary.json", result)
        budget()
        terminal = {"version": VERSION, "status": "completed", "agreement": True, "scope": SCOPE, "request": request,
                    "source_sha256": source_hashes, "authentication_source_sha256": {PRODUCER: PRODUCER_PIN, module.AUDITOR: module.AUDITOR_PIN},
                    "producer_receipt_sha256": args.receipt_sha256, "producer_summary_sha256": args.summary_sha256,
                    "plan_sha256": args.plan_sha256, "execution_completed_sha256": inherited.completed_sha256,
                    "files": {name: descriptor(args.out/name) for name in ("started.json", "summary.json")},
                    "limits": CAPS, "clock_backend": clock.backend, "started_ns": deadline.started_ns,
                    "finished_ns": deadline.started_ns + elapsed, "deadline_ns": deadline.expires_ns,
                    "elapsed_ns": elapsed, "wall_seconds": elapsed/1e9, "peak_rss_bytes": rss(),
                    "timing_scope": "Native start through last pre-receipt budget check; final hashing/publication rechecked before return.",
                    "scalar_checks": comparisons.total, "model_calls": 0, "prediction_arrays_read": 12,
                    "process_exit_status": "External tool completion must be recorded separately after exit."}
        write(args.out/"receipt.json", terminal)
        budget()
        print(json.dumps({"status": "completed", "agreement": True, "scalar_checks": comparisons.total,
                          "wall_seconds": terminal["wall_seconds"], "peak_rss_bytes": terminal["peak_rss_bytes"],
                          "receipt_sha256": digest(args.out/"receipt.json"), "summary_sha256": digest(args.out/"summary.json")}))
        return terminal
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if (args.out/"receipt.json").exists():
            (args.out/"receipt.json").rename(args.out/"invalid-receipt.json")
        write(args.out/"failed.json", {"version": VERSION, "status": "failed", "agreement": False, "error": repr(error),
              "elapsed_ns": None, "last_successful_elapsed_ns": elapsed, "scalar_checks_before_failure": comparisons.total,
              "source_sha256": digest(__file__), "model_calls": 0, "scope": SCOPE})
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    execute(parser.parse_args())
