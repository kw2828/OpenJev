"""Independent primary numerical audit; no model or producer metric calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import resource
import signal
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
READER = "scripts/report_dialogue_warm_pooling.py"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
ARMS = ("pooled", "schema_attention", "belief_query", "state_token")
SEEDS = (6901, 6902, 6903)
FITS = [f"{arm}-{seed}" for seed in SEEDS for arm in (*ARMS, "untouched")]
STUDY_PIN = "b86c64cc5cbf60dd751eb6a60545e4e253ef5de660740a5c1e74939a2aa481f7"
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
LIMITS = {"wall_seconds": 600, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
SCOPE = ("Independent numerical replay of twelve adapted fits and three untouched references, all/seen/unseen micro "
         "and three-stratum metrics (180 cells), three-stratum macro scores and all thirty-two fixed conditions. "
         "Execution provenance and file authentication are inherited from the source-pinned qualified reader's read-only "
         "authenticate/manifest functions. No producer numerical functions, model execution, checkpoint deserialization, "
         "probability repair or temperature fitting. Secondary service/type/bin/recovery trees, family/contrast trees, "
         "cost counters and qualification parity are authenticated but not independently recomputed. "
         "Saved raw arrays do not independently establish inference or training execution.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


class Budget:
    def __init__(self, out):
        require(sha(ROOT / CLOCK) == CLOCK_PIN, "Qualified clock pin")
        self.clock = importlib.import_module("openjev.research.suspend_clock").SuspendClock()
        require(self.clock.backend in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "Native clock required")
        self.deadline = self.clock.deadline_after(600)
        self.out, self.last, self.fit_id = out, None, None

    def check(self):
        now = self.clock.now_ns()
        require(now < self.deadline.expires_ns, "Audit native deadline")
        require(rss() <= LIMITS["rss_bytes"], "Audit RSS cap")
        self.last = now - self.deadline.started_ns

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")


class Checks:
    def __init__(self):
        self.count, self.maximum_error = 0, 0.

    def compare(self, actual, expected, path):
        if isinstance(expected, dict):
            require(type(actual) is dict and set(expected) <= set(actual), "Missing audited keys: " + path)
            for key, value in expected.items():
                self.compare(actual[key], value, path + "/" + key)
        elif isinstance(expected, list):
            require(type(actual) is list and len(actual) == len(expected), "Complete list: " + path)
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                self.compare(a, b, f"{path}/{i}")
        else:
            self.count += 1
            if type(expected) is float:
                require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                        and abs(actual-expected) <= 1e-12, "Numeric mismatch: " + path)
                self.maximum_error = max(self.maximum_error, abs(actual-expected))
            else:
                require(type(actual) is type(expected) and actual == expected, "Exact mismatch: " + path)


def authenticate(args, budget):
    require(sha(args.plan) == args.plan_sha256, "External complete report plan pin")
    plan = read(args.plan)
    require(READER in plan["sources"] and plan["study_plan"]["sha256"] == STUDY_PIN,
            "Independently fixed study and pinned reporting source")
    for name, pin in plan["sources"].items():
        require(sha(ROOT/name) == pin, "Report source pin before import")
    reader = importlib.import_module("report_dialogue_warm_pooling")
    accepted, study, done, terminal, run = reader.authenticate(
        SimpleNamespace(plan=args.plan, plan_sha256=args.plan_sha256, out=args.report), budget)
    require(accepted == plan and run.resolve() == args.run.resolve()
            and sha(run/"completed.json") == args.completed_sha256 == plan["run"]["completed_sha256"],
            "External entire completed campaign identity")
    require(sha(args.report/"receipt.json") == args.report_sha256, "External completed report receipt pin")
    receipt = read(args.report/"receipt.json")
    require(receipt["status"] == "completed" and receipt["version"] == plan["version"]
            and receipt["plan_sha256"] == args.plan_sha256 and receipt["source_sha256"] == plan["sources"]
            and receipt["study_plan"] == plan["study_plan"] and receipt["run"] == plan["run"]
            and receipt["complete_adapted_fits"] == 12 and receipt["complete_untouched_references"] == 3
            and receipt["scored_fits"] == 15 and receipt["model_calls"] == 0
            and receipt["temperature_fitted"] is False and receipt["official_test_opened"] is False
            and receipt["clock_backend"] in {"mach_continuous_time", "CLOCK_BOOTTIME"}
            and 0 < receipt["wall_seconds"] < 600 and set(receipt["files"]) == {"summary.json", "report.md"},
            "Complete qualified report identity before aggregate decoding")
    reader.manifest(args.report, receipt["files"], "receipt.json")
    summary = read(args.report/"summary.json")
    require(summary["status"] == "completed" and summary["technical_validity_passed"] is True
            and summary["version"] == plan["version"] and summary["complete_adapted_fits"] == 12
            and summary["complete_untouched_references"] == 3 and summary["scored_fits"] == 15
            and summary["fit_order"] == FITS and set(summary["fits"]) == set(FITS)
            and summary["bindings"] == {"study_plan": plan["study_plan"], "run": plan["run"]}
            and summary["model_calls"] == 0 and summary["temperature_fitted"] is False
            and summary["probabilities_transformed"] is False and summary["official_test_opened"] is False,
            "All fifteen raw report results with exact study/experiment bindings")
    return plan, study, done, terminal, receipt, summary


def metadata(args, study, done, budget):
    rows = []
    with (args.run/"rows-dev.jsonl").open() as stream:
        for line in stream:
            budget.check()
            rows.append(json.loads(line))
    require(len(rows) == done["row_counts"]["dev"]
            and list(dict.fromkeys(r["dialogue_id"] for r in rows)) == study["selected"]["dev"], "Full canonical selected DEV cohort")
    original = []
    with (Path(study["checkpoints"]["run"])/"evaluation-rows.jsonl").open() as stream:
        for line in stream:
            budget.check()
            original.append(json.loads(line))
    require(rows == original and len(rows) == 62329, "Full unchanged original canonical DEV ledger")
    labels, widths, strata, unseen, endpoints = [], [], [], [], set()
    for i, row in enumerate(rows):
        ids, target = row["candidate_ids"], row["label_index"]
        require(type(row["row_index"]) is int and row["row_index"] == i and row["split"] == "dev"
                and 3 <= len(ids) <= 12 and len(set(ids)) == len(ids) and type(target) is int
                and 0 <= target < len(ids) and row["label_id"] == ids[target]
                and ids.count("reserved:NOT_MENTIONED") == ids.count("reserved:DONTCARE") == 1, "Canonical target and candidate order")
        group = row["bin"] if row["bin"] in STRATA[:2] else "changed"
        require(row["bin"] in (*STRATA[:2], "first_assignment", "revision", "clear")
                and row["stratum"] == group and type(row["unseen"]) is bool, "Fixed exhaustive strata/panels")
        identity = (row["dialogue_id"], row["time"], row["query_index"])
        require(identity not in endpoints, "No duplicate endpoint")
        endpoints.add(identity)
        labels.append(target)
        widths.append(len(ids))
        strata.append(group)
        unseen.append(row["unseen"])
    return np.asarray(labels, np.int64), np.arange(12)[None, :] < np.asarray(widths)[:, None], np.asarray(strata), np.asarray(unseen, bool)


def fraction(value):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def score(path, meta):
    labels, mask, strata, unseen = meta
    with np.load(path, allow_pickle=False) as packet:
        require(set(packet.files) == {"log_probs", "row_indices"}, "Exact prediction fields")
        logs, indices = packet["log_probs"], packet["row_indices"]
    require(logs.dtype == np.float32 and logs.shape == mask.shape and indices.dtype == np.int64
            and np.array_equal(indices, np.arange(len(labels), dtype=np.int64)), "Complete raw canonical row alignment")
    require(np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(), "Finite supported logs and exact padding")
    raw = logs.astype(np.float64)
    p = np.exp(raw)
    require(float(np.max(np.abs(p.sum(1)-1))) <= 2e-6 and float(raw[mask].max()) <= math.log1p(2e-6), "Raw mass without repair")
    correct = np.argmax(raw, axis=1) == labels
    target = np.arange(len(labels)), labels
    losses = {"nll": -raw[target], "brier": np.square(p).sum(1)-2*p[target]+1}
    require(all(np.isfinite(v).all() for v in losses.values()), "Finite proper scores")

    def cell(selected):
        n, hits = int(selected.sum()), int(correct[selected].sum())
        return {"count": n, "correct": hits, "incorrect": n-hits, "accuracy": hits/n if n else None,
                "error": (n-hits)/n if n else None,
                **{k: math.fsum(map(float, v[selected]))/n if n else None for k, v in losses.items()}}

    panels = {}
    for name in ("all", "seen", "unseen"):
        selected = np.ones(len(labels), bool) if name == "all" else unseen if name == "unseen" else ~unseen
        groups = {s: cell(selected & (strata == s)) for s in STRATA}
        macro = {metric: math.fsum(groups[s][metric] for s in STRATA)/3
                 if all(groups[s][metric] is not None for s in STRATA) else None for metric in ("accuracy", "nll", "brier")}
        panels[name] = {"micro": cell(selected), "strata": groups, "macro_three": macro}
    return {"panels": panels, "validation": {"rows": len(labels), "supported_candidate_positions": int(mask.sum()),
        "maximum_mass_error": float(np.max(np.abs(p.sum(1)-1))), "tolerance": 2e-6,
        "exact_top1_tie_rows": int(((logs == logs.max(1, keepdims=True)).sum(1) > 1).sum()),
        "float64_exponent_underflow_positions": int((p[mask] == 0).sum())}}


def gate(fits):
    require(set(fits) == set(FITS), "Every fixed adapted/reference fit participates")

    def accuracy(name, panel, stratum):
        cell = fits[name]["panels"][panel]["strata"][stratum]
        require(type(cell["count"]) is int and cell["count"] > 0
                and type(cell["correct"]) is int and 0 <= cell["correct"] <= cell["count"],
                "Positive complete primary support")
        return Fraction(cell["correct"], cell["count"])

    def macro(name, panel):
        return sum((accuracy(name, panel, s) for s in STRATA), Fraction())/len(STRATA)

    def seed_map(values, exact=False):
        return {str(seed): fraction(value) if exact else float(value)
                for seed, value in zip(SEEDS, values, strict=True)}

    checks = []
    for comparator in ("pooled", "schema_attention", "state_token", "untouched"):
        new = [f"belief_query-{seed}" for seed in SEEDS]
        old = [f"{comparator}-{seed}" for seed in SEEDS]
        unseen_gain = [macro(a, "unseen")-macro(b, "unseen") for a, b in zip(new, old, strict=True)]
        seen_gain = [macro(a, "seen")-macro(b, "seen") for a, b in zip(new, old, strict=True)]
        changed_gain = [accuracy(a, "unseen", "changed")-accuracy(b, "unseen", "changed")
                        for a, b in zip(new, old, strict=True)]

        def rate_check(name, values, threshold, comparison, *, comparator=comparator):
            mean = sum(values, Fraction())/len(SEEDS)
            passed = mean >= threshold if comparison == "ge" else mean <= threshold
            return {"comparator": comparator, "name": name, "passed": passed, "comparison": comparison,
                    "threshold": float(threshold), "threshold_exact": fraction(threshold),
                    "paired_seed_changes": seed_map(values), "paired_seed_changes_exact": seed_map(values, True),
                    "mean_paired_change": float(mean), "mean_paired_change_exact": fraction(mean)}

        checks.append(rate_check("unseen_macro_gain_at_least_1pp", unseen_gain, Fraction(1, 100), "ge"))
        positive = sum(value > 0 for value in unseen_gain)
        checks.append({"comparator": comparator, "name": "unseen_macro_positive_at_least_two_seeds",
                       "passed": positive >= 2, "positive_seeds": positive, "required_positive_seeds": 2,
                       "paired_seed_changes": seed_map(unseen_gain), "paired_seed_changes_exact": seed_map(unseen_gain, True)})
        checks.append(rate_check("unseen_changed_gain_at_least_1pp", changed_gain, Fraction(1, 100), "ge"))
        for metric in ("nll", "brier"):
            after = [fits[name]["panels"]["unseen"]["micro"][metric] for name in new]
            before = [fits[name]["panels"]["unseen"]["micro"][metric] for name in old]
            require(all(type(value) is float and math.isfinite(value) for value in [*after, *before]),
                    "Finite unseen proper scores for every paired seed")
            after_mean, before_mean = math.fsum(after)/len(SEEDS), math.fsum(before)/len(SEEDS)
            changes = [a-b for a, b in zip(after, before, strict=True)]
            checks.append({"comparator": comparator, "name": "unseen_"+metric+"_nonworse",
                           "passed": after_mean <= before_mean, "belief_query_mean": after_mean,
                           "comparator_mean": before_mean, "difference_of_arm_means": after_mean-before_mean,
                           "paired_seed_changes": seed_map(changes), "mean_paired_change": math.fsum(changes)/len(SEEDS)})
        checks.append(rate_check("seen_macro_decline_at_most_1pp", seen_gain, Fraction(-1, 100), "ge"))
        for panel in ("seen", "unseen"):
            error_increase = [(1-accuracy(a, panel, "assigned_retention"))-(1-accuracy(b, panel, "assigned_retention"))
                              for a, b in zip(new, old, strict=True)]
            checks.append(rate_check(panel+"_assigned_retention_error_increase_at_most_half_pp",
                                     error_increase, Fraction(1, 200), "le"))
    require(len(checks) == 32, "Exactly thirty-two fixed comparisons")
    return {"checks": checks, "checks_total": 32, "checks_passed": sum(c["passed"] for c in checks),
            "passed": all(c["passed"] for c in checks)}


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget = handler = None
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source_pin = None
    try:
        source_pin = sha(__file__)
        budget = Budget(args.out)

        def expired(*_):
            raise TimeoutError("Supplementary independent audit alarm")

        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 600)
        write(args.out/"started.json", {"request": request, "source_sha256": source_pin, "limits": LIMITS, "model_calls": 0})
        plan, study, done, _, receipt, summary = authenticate(args, budget)
        meta = metadata(args, study, done, budget)
        fits, checks = {}, Checks()
        for name in FITS:
            budget.fit_id = name
            directory = args.run/"qualification"/name if name.startswith("untouched-") else args.run/name
            fits[name] = score(directory/"predictions.npz", meta)
            checks.compare(summary["fits"][name], fits[name], name)
            budget.check()
        decision = gate(fits)
        checks.compare(summary["continuation"], decision, "all32conditions")
        checks.compare(receipt["continuation"], decision, "report-receipt")
        result = {"status": "completed", "agreement": True, "fits": fits, "continuation": decision,
                  "scalar_checks": checks.count, "cells": 180, "maximum_score_difference": checks.maximum_error,
                  "score_absolute_tolerance": 1e-12, "gate_policy": "Exact counts/rationals and booleans; proper-score decisions use no epsilon.", "scope": SCOPE}
        write(args.out/"summary.json", result)
        authenticate(args, budget)
        require(sha(__file__) == source_pin, "Stable audit source")
        budget.storage()
        elapsed = budget.last
        output = {"status": "completed", "agreement": True, "request": request, "source_sha256": source_pin,
                  "inherited_auth_reader_sha256": plan["sources"][READER], "producer_receipt_sha256": args.report_sha256,
                  "producer_summary_sha256": sha(args.report/"summary.json"), "plan_sha256": args.plan_sha256,
                  "execution_completed_sha256": args.completed_sha256, "scalar_checks": checks.count, "cells": 180,
                  "continuation_passed": decision["passed"], "checks_passed": decision["checks_passed"], "checks_total": 32,
                  "limits": LIMITS, "clock_backend": budget.clock.backend, "elapsed_ns": elapsed, "wall_seconds": elapsed/1e9,
                  "timing_available": True, "started_ns": budget.deadline.started_ns,
                  "finished_ns": budget.deadline.started_ns+elapsed, "deadline_ns": budget.deadline.expires_ns,
                  "peak_rss_bytes": rss(), "model_calls": 0, "scope": SCOPE,
                  "files": {n: {"sha256": sha(args.out/n), "bytes": (args.out/n).stat().st_size} for n in ("started.json", "summary.json")}}
        write(args.out/"receipt.json", output)
        budget.storage()
        return output
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out/"receipt.json").exists():
                (args.out/"receipt.json").rename(args.out/"invalid-receipt.json")
            write(args.out/"failed.json", {"status": "failed", "agreement": False, "request": request, "source_sha256": source_pin,
                  "error_type": type(error).__name__, "error": str(error), "fit_id": None if budget is None else budget.fit_id,
                  "wall_seconds": None, "timing_available": False, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "report", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("plan-sha256", "completed-sha256", "report-sha256"):
        parser.add_argument("--"+name, required=True)
    answer = execute(parser.parse_args())
    print(json.dumps({"status": answer["status"], "agreement": answer["agreement"], "checks_passed": answer["checks_passed"]}))
