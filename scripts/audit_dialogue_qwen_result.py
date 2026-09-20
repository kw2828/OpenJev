"""Independent complete-result arithmetic, without producer/reporter imports.

Run only after the two-arm inference and main report are terminal. This audit
reconstructs supported-label distributions, fixed primary metrics and decisions.
Inference, tokenization, source closure, public-input provenance and historical
model metrics inherit the externally authenticated run/report/preparation chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-result-audit-v1"
ARMS = ("current", "history4")
STRATA = ("all", "changed", "retained")
WEIGHTINGS = ("row", "equal_service")
SUPPORT = {"all": 7819, "changed": 578, "retained": 7241}
REFERENCE = "output/dialogue-objective-v1/report-01/summary.json"
REFERENCE_SHA = "d744753d9edb545b9060867500e3c390d03da69937ffd6d8b3fd1e0c12c4cf6b"
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
SCOPE = (
    "Independent supported-label log-softmax, exact prompt-order choices, both-arm all/changed/retained "
    "row/equal-service accuracy/error/NLL/Brier and per-service tables, plus both four-check behavioral "
    "and four-check proper-score decisions. Original corrected-flat metrics are inherited from a separately "
    "pinned historical summary. This is not inference, tokenization, full-vocabulary mass, cost, public "
    "actor provenance, source/runtime-closure, rare-category, equal-dialogue or paired-repair re-auditing. "
    "Those checks remain within the authenticated main report/preparation and source-bound execution."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
            check()
    return value.hexdigest()


def item(path, check=lambda: None):
    return {"sha256": sha(path, check), "bytes": Path(path).stat().st_size}


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def authenticate(args, check):
    prepared, run, report = (Path(getattr(args, name)).resolve() for name in ("prepared", "run", "report"))
    for path, digest in ((prepared/"plan.json", args.plan_sha256), (run/"completed.json", args.run_sha256),
                         (report/"receipt.json", args.report_receipt_sha256),
                         (report/"summary.json", args.report_summary_sha256)):
        require(sha(path, check) == digest, "External artifact pin: "+str(path))
    plan, prep, done, receipt = (read(path) for path in (prepared/"plan.json", prepared/"completed.json",
                                                      run/"completed.json", report/"receipt.json"))
    require(plan["version"] == "dialogue-qwen-observation-v1" and plan["method"] == "batch"
            and plan["row_count"] == SUPPORT["all"] and plan["decisions"] == 2*SUPPORT["all"], "Fixed complete scope")
    require(prep["status"] == done["status"] == receipt["status"] == "completed"
            and prep["plan_sha256"] == done["plan_sha256"] == receipt["plan_sha256"] == args.plan_sha256
            and done["phase"] == "run" and done["version"] == plan["version"]
            and done["labels_accessed"] is False and done["quality_outputs_saved"] is True
            and receipt["version"] == "dialogue-qwen-observation-report-v1"
            and receipt["technical_validity_passed"] is True
            and receipt["execution_completed_sha256"] == args.run_sha256
            and done["source_sha256"] == receipt["study_source_sha256"] == plan["source_sha256"],
            "Completed run/report lineage")
    bindings = {}
    def bind(path, descriptor):
        require(set(descriptor) == {"sha256", "bytes"} and item(path, check) == descriptor,
                "Payload digest/size: "+str(path))
        bindings[path] = descriptor
    for directory, document, members, terminal in (
        (prepared, prep, {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}, "completed.json"),
        (run, done, {"started.json", "plan.json", "timings.jsonl", "scores.jsonl"}, "completed.json"),
        (report, receipt, {"started.json", "summary.json", "report.md"}, "receipt.json"),
    ):
        require(set(document["files"]) == members
                and {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
                == members | {terminal}, "Exact artifact closure")
        for name, descriptor in document["files"].items():
            bind(directory/name, descriptor)
        bindings[directory/terminal] = item(directory/terminal, check)
    require(sha(run/"plan.json", check) == args.plan_sha256 and set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
            and all(prep["files"][key] == value for key, value in plan["files"].items()), "Prepared payload identity")
    reference = ROOT/REFERENCE
    require(plan["inputs"][REFERENCE]["sha256"] == REFERENCE_SHA, "Fixed historical reference pin")
    bind(reference, plan["inputs"][REFERENCE])
    historical, summary = read(reference), read(report/"summary.json")
    require(historical["status"] == "completed" and historical["technical_validity_passed"] is True
            and summary["status"] == "completed" and summary["technical_validity_passed"] is True
            and summary["version"] == receipt["version"] and summary["plan_sha256"] == args.plan_sha256
            and summary["execution_completed_sha256"] == args.run_sha256
            and summary["source_sha256"] == plan["source_sha256"], "Summary identity")
    return prepared, run, plan, historical, summary, bindings


def reconstruct(prepared, run, plan, check):
    labels = [decode(line) for line in (prepared/"labels.jsonl").read_text().splitlines()]
    require(len(labels) == len({r["row_index"] for r in labels}) == SUPPORT["all"], "Canonical evaluator membership")
    by_id = {row["row_index"]: row for row in labels}
    records, request_ids = {arm: {} for arm in ARMS}, set()
    with (prepared/"requests.jsonl").open() as requests, (run/"scores.jsonl").open() as scores:
        for request_line in requests:
            request = decode(request_line)
            line = scores.readline()
            require(bool(line), "Missing score request")
            score = decode(line)
            require(all(score[key] == request[key] for key in ("request_id", "arm", "dialogue_id", "time")),
                    "Score request identity/order")
            arm = request["arm"]
            require(arm in ARMS and request["request_id"] not in request_ids, "Request membership")
            request_ids.add(request["request_id"])
            n = len(request["row_indices"])
            require(1 <= n <= 4 and len(score["questions"]) == len(request["request"]["questions"])
                    == len(request["canonical_id_maps"]) == len(request["ordered_candidate_ids"]) == n,
                    "Question row alignment")
            for index, question, mapping, order, saved in zip(request["row_indices"], request["request"]["questions"],
                    request["canonical_id_maps"], request["ordered_candidate_ids"], score["questions"], strict=True):
                require(type(index) is int and index in by_id and index not in records[arm], "Exact saved row membership")
                label = by_id[index]
                ids = [mapping[c["id"]] for c in question["candidates"]]
                require(len(ids) == len(set(ids)) == label["candidate_count"] and 2 <= len(ids) <= 12
                        and set(ids) == set(order) and len(order) == len(ids)
                        and order == [mapping[c["id"]] for c in sorted(question["candidates"], key=lambda c: c["description"])],
                        "Canonical/prompt candidate order")
                require(label["dialogue_id"] == request["dialogue_id"] and label["time"] == request["time"]
                        and saved["row_index"] == index and saved["question_id"] == question["id"] == f"r{index}"
                        and saved["candidate_ids"] == ids and request["label_ids"] == list(range(32, 44))
                        and saved["label_ids"] == [32+order.index(cid) for cid in ids]
                        and saved["tie_break"] == "first frozen prompt label", "Saved canonical/label join")
                z = np.asarray(saved["candidate_logits"], dtype=np.float64)
                require(z.shape == (len(ids),) and np.isfinite(z).all(), "Finite supported logits")
                log_probs = z-z.max()
                log_probs -= np.logaddexp.reduce(log_probs)
                saved_logs, saved_probs = (np.asarray(saved[key], dtype=np.float64) for key in ("log_probs", "probabilities"))
                require(saved_logs.shape == saved_probs.shape == z.shape and np.isfinite(log_probs).all()
                        and np.allclose(saved_logs, log_probs, rtol=2e-12, atol=2e-12)
                        and np.allclose(saved_probs, np.exp(log_probs), rtol=2e-12, atol=2e-12),
                        "Saved conditional probabilities/logs")
                selected = next(cid for cid in order if z[ids.index(cid)] == z.max())
                require(saved["selected_id"] == selected, "Frozen prompt-label maximum/tie")
                target, previous = ids.index(label["current_candidate_id"]), ids.index(label["previous_candidate_id"])
                p = np.exp(log_probs)
                p[target] -= 1
                correct = selected == ids[target]
                records[arm][index] = {"service": label["service"], "changed": target != previous,
                    "accuracy": int(correct), "error": int(not correct), "nll": float(-log_probs[target]),
                    "brier": math.fsum(float(v*v) for v in p)}
            check()
        require(scores.readline() == "", "Unexpected extra score")
    require(len(request_ids) == sum(plan["request_counts"].values())
            and all(set(v) == set(by_id) for v in records.values()), "Complete two-arm coverage")
    return records


def calculate(records):
    result = {}
    for arm, records_by_id in records.items():
        rows = [records_by_id[index] for index in sorted(records_by_id)]
        services = sorted({r["service"] for r in rows})
        require(len(services) == 6, "Six primary services")
        def cell(values, services=services):
            groups = {service: [r for r in values if r["service"] == service] for service in services}
            groups = {k: v for k, v in groups.items() if v}
            metrics = {}
            for metric in ("accuracy", "error", "nll", "brier"):
                metrics[metric] = {
                    "row": math.fsum(r[metric] for r in values)/len(values) if values else None,
                    "equal_service": math.fsum(math.fsum(r[metric] for r in group)/len(group)
                                               for group in groups.values())/len(groups) if groups else None}
            return {"rows": len(values), "counts": {"correct": sum(r["accuracy"] for r in values),
                                                       "error": sum(r["error"] for r in values)}, "metrics": metrics}
        selectors = {"all": lambda _r: True, "changed": lambda r: r["changed"], "retained": lambda r: not r["changed"]}
        result[arm] = {"cells": {name: cell([r for r in rows if choose(r)]) for name, choose in selectors.items()},
                       "services": {s: {name: cell([r for r in rows if r["service"] == s and choose(r)])
                                          for name, choose in selectors.items()} for s in services}}
        require({name: c["rows"] for name, c in result[arm]["cells"].items()} == SUPPORT, "Primary stratum counts")
    return result


def rules(candidate, controls):
    def rational(fit, stratum, key, weighting):
        def rate(c):
            n, d = c["counts"][key], c["rows"]
            require(type(n) is int and type(d) is int and 0 <= n <= d, "Integral decision counts")
            return Fraction(n, d) if d else None
        if weighting == "row":
            return rate(fit["cells"][stratum])
        values = [rate(s[stratum]) for s in fit["services"].values() if s[stratum]["rows"]]
        return sum(values, Fraction())/len(values) if values else None
    behavioral, scores = {}, {}
    for stratum, metric, key, threshold in (("changed", "accuracy", "correct", Fraction(1, 50)),
                                          ("retained", "error", "error", Fraction(0))):
        for weighting in WEIGHTINGS:
            a = rational(candidate, stratum, key, weighting)
            bs = [rational(c, stratum, key, weighting) for c in controls]
            delta = a-sum(bs, Fraction())/len(bs) if a is not None and all(b is not None for b in bs) else None
            passed = delta is not None and (delta >= threshold if metric == "accuracy" else delta <= threshold)
            behavioral[f"{stratum}_{metric}_{weighting}"] = {"passed": passed,
                "difference": None if delta is None else float(delta), "threshold": float(threshold),
                "relation": ">=" if metric == "accuracy" else "<="}
    for metric in ("nll", "brier"):
        for weighting in WEIGHTINGS:
            a = candidate["cells"]["all"]["metrics"][metric][weighting]
            bs = [c["cells"]["all"]["metrics"][metric][weighting] for c in controls]
            delta = a-math.fsum(bs)/len(bs) if a is not None and all(b is not None for b in bs) else None
            scores[f"{metric}_{weighting}"] = {"passed": delta is not None and delta <= 0, "difference": delta,
                                               "threshold": 0., "relation": "<="}
    return {"behavioral": {"passed": all(v["passed"] for v in behavioral.values()), "checks": behavioral},
            "proper_score_nonregression": {"passed": all(v["passed"] for v in scores.values()), "checks": scores}}


def verify(fits, historical, reported):
    checks = 0
    def compare(a, b, location):
        nonlocal checks
        if isinstance(b, dict):
            require(isinstance(a, dict) and set(b) <= set(a), "Missing audit fields: "+location)
            for key, value in b.items():
                compare(a[key], value, location+"/"+key)
        else:
            checks += 1
            if type(b) is float:
                require(type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12),
                        "Arithmetic disagreement: "+location)
            else:
                require(type(a) is type(b) and a == b, "Exact disagreement: "+location)
    compare(reported["arms"], fits, "arms")
    flat = []
    for seed in (6201, 6202, 6203):
        source = historical["historical"]["fits"][f"flat_stratum-corrected-{seed}"]
        fit = {"cells": {name: source["cells"]["heldout_service/"+name] for name in STRATA},
               "services": {service: source["services"][service] for service in fits["current"]["services"]}}
        for name in STRATA:
            require(fit["cells"][name]["rows"] == SUPPORT[name], "Historical primary support")
            for service, cells in fits["current"]["services"].items():
                require(fit["services"][service][name]["rows"] == cells[name]["rows"], "Historical service support")
        flat.append(fit)
    decisions = {"semantic_strength": rules(fits["current"], flat),
                 "added_history": rules(fits["history4"], [fits["current"]])}
    compare(reported["decisions"], decisions, "decisions")
    return decisions, checks


def execute(args):
    started = time.monotonic()
    out = Path(args.out).resolve()
    for name in ("prepared", "run", "report"):
        source = Path(getattr(args, name)).resolve()
        require(not out.is_relative_to(source) and not source.is_relative_to(out), "Separate audit output")
    out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) for k, v in vars(args).items()}
    source_hash, prior = None, None
    def check():
        if time.monotonic()-started > LIMITS["wall_seconds"]:
            raise TimeoutError("Independent audit wall cap")
        require(rss() <= LIMITS["rss_bytes"], "Independent audit RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
    def expired(*_):
        raise TimeoutError("Independent audit wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        source_hash = sha(__file__, check)
        write(out/"started.json", {"version": VERSION, "request": request, "limits": LIMITS, "source_sha256": source_hash})
        prepared, run, plan, historical, reported, bindings = authenticate(args, check)
        fits = calculate(reconstruct(prepared, run, plan, check))
        decisions, checks = verify(fits, historical, reported)
        summary = {"status": "completed", "agreement": True, "version": VERSION, "scope": SCOPE,
                   "fits": fits, "decisions": decisions, "scalar_checks": checks, "primary_rows_per_arm": SUPPORT["all"],
                   "metric_cells": 2*3*(1+6), "decision_checks": 16, "historical_summary_sha256": REFERENCE_SHA}
        write(out/"summary.json", summary)
        for path, descriptor in bindings.items():
            require(item(path, check) == descriptor, "End artifact identity: "+str(path))
        require(sha(__file__, check) == source_hash, "Audit source identity")
        files = {name: item(out/name, check) for name in ("started.json", "summary.json")}
        write(out/"receipt.json", {"status": "completed", "agreement": True, "version": VERSION, "scope": SCOPE,
              "source_sha256": source_hash, "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
              "report_receipt_sha256": args.report_receipt_sha256, "report_summary_sha256": args.report_summary_sha256,
              "request": request, "files": files, "authenticated_inputs": {str(p): d for p, d in bindings.items()},
              "scalar_checks": checks, "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": rss(),
              "limits": LIMITS, "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0,
              "wall_scope": "Whole saved-artifact audit through output hashing; terminal write/return cap-checked"})
        check()
    except BaseException as error:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "source_sha256": source_hash, "error": repr(error), "wall_seconds": time.monotonic()-started})
        except BaseException as secondary:  # noqa: BLE001 - preserve original audit failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "plan-sha256", "run", "run-sha256", "report", "report-receipt-sha256", "report-summary-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
