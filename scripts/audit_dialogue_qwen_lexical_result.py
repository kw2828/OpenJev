"""Independent primary saved-result arithmetic for the lexical-input ablation.

Uses only the pinned independent original auditor's authentication/reconstruction
and primary aggregation helpers. No producer, reporter, tokenizer or model imports.
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
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-lexical-result-audit-v1"
EXPERIMENT = "dialogue-qwen-lexical-ablation-v1"
REPORT_VERSION = "dialogue-qwen-lexical-ablation-report-v1"
HELPER = "scripts/audit_dialogue_qwen_result.py"
HELPER_SHA = "ccc24ad9555ff5a45281fcb02521d7c0ee0d0b45697042edb4d69947a7245b70"
BASE_PLAN = "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111"
BASE_PREP = "6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f"
BASE_RUN = "872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c"
BASE_SUMMARY = "931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c"
BASE_REPORT_RECEIPT = "3106ca10cfa939fb47af41560772ae66a9308911cbfb5aa290dbe213f93e4ce2"
NEW_PLAN = "89b90d4a7decddf35e2dfbfacd53ce15a61e455cb32be9a719092839367caa36"
BASE_SOURCES = {"src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "src/openjev/research/dialogue_qwen_observation.py", "scripts/prepare_dialogue_qwen_observation.py",
    "scripts/run_dialogue_qwen_observation.py", "tests/test_dialogue_qwen_observation.py",
    "tests/test_run_dialogue_qwen_observation.py", "tests/test_prepare_dialogue_qwen_observation.py",
    "research/dialogue-qwen-observation-protocol.md"}
PROTOCOL = "research/dialogue-qwen-lexical-ablation-protocol.md"
ADDED_SOURCES = {"src/openjev/research/dialogue_qwen_lexical_ablation.py",
    "scripts/prepare_dialogue_qwen_lexical_ablation.py", "tests/test_dialogue_qwen_lexical_ablation.py",
    "tests/test_prepare_dialogue_qwen_lexical_ablation.py", PROTOCOL}
SOURCES = {"scripts/audit_dialogue_qwen_lexical_result.py", "tests/test_audit_dialogue_qwen_lexical_result.py", HELPER}
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
ARMS, STRATA = ("current", "history4"), ("all", "changed", "retained")
SCOPE = (
    "Independent arithmetic using the SHA-pinned independent original auditor's supported-log-softmax, "
    "prompt-label tie reconstruction and primary aggregation helpers; not producer/reporter metric code. "
    "Checks 84 original/no-flags primary all/changed/retained cells (overall plus six services), row/equal-service "
    "accuracy/error/NLL/Brier, and all 16 exact count/score continuation checks. Authenticates complete saved "
    "preparation/run/report payloads, source maps and paired row/candidate identities. Actual inference, "
    "exact public prompt subtraction, tokenization, full-vocabulary mass, cost accounting, rare/subtype, "
    "equal-dialogue and paired secondary metrics inherit the authenticated main report; they are not "
    "independently replayed here. No weights, model or tokenizer are loaded. Original outcomes remain unchanged."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            check()
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_helper(check=lambda: None):
    path = ROOT/HELPER
    require(sha(path, check) == HELPER_SHA, "Pinned independent helper source")
    spec = importlib.util.spec_from_file_location("independent_lexical_audit_helpers", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def safe(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Safe repository-relative source")
    return ROOT/path


def authenticate(args, h, check):
    """All closure/hash checks finish before any labels, requests or scores decode."""
    baseline_args = SimpleNamespace(prepared=args.baseline_prepared, plan_sha256=BASE_PLAN,
        run=args.baseline_run, run_sha256=BASE_RUN, report=args.baseline_report,
        report_summary_sha256=BASE_SUMMARY, report_receipt_sha256=BASE_REPORT_RECEIPT)
    require(h.sha(Path(args.baseline_prepared)/"completed.json", check) == BASE_PREP, "Fixed original preparation")
    old_prepared, old_run, original, _, old_report, bindings = h.authenticate(baseline_args, check)
    prepared, run, report = (Path(getattr(args, name)).resolve() for name in ("prepared", "run", "report"))
    require(args.plan_sha256 == NEW_PLAN, "Fixed new plan pin")
    for path, digest in ((prepared/"plan.json", args.plan_sha256), (run/"completed.json", args.run_sha256),
                         (report/"receipt.json", args.report_receipt_sha256),
                         (report/"summary.json", args.report_summary_sha256)):
        require(h.sha(path, check) == digest, "External new artifact pin")
    plan, prep, done, receipt = (h.read(path) for path in (prepared/"plan.json", prepared/"completed.json",
                                                       run/"completed.json", report/"receipt.json"))
    require(plan["version"] == done["version"] == "dialogue-qwen-observation-v1"
            and plan["experiment_id"] == prep["experiment_id"] == receipt["experiment_id"] == EXPERIMENT
            and receipt["version"] == REPORT_VERSION and done["phase"] == "run", "New experiment identity")
    require(prep["status"] == done["status"] == receipt["status"] == "completed"
            and prep["plan_sha256"] == done["plan_sha256"] == receipt["plan_sha256"] == args.plan_sha256
            and receipt["execution_completed_sha256"] == args.run_sha256 and receipt["technical_validity_passed"] is True
            and prep["model_calls"] == 0 and prep["tokenizer_only"] is True and prep["labels_decoded"] is False
            and prep["baseline_scores_accessed"] is False and prep["checkpoint_deserializations"] == 0
            and done["labels_accessed"] is False and done["quality_outputs_saved"] is True,
            "Complete new preparation/run/report lineage")
    def bind(path, descriptor):
        require(set(descriptor) == {"sha256", "bytes"} and h.item(path, check) == descriptor, "New payload hash/size")
        bindings[path] = descriptor
    for directory, document, members, terminal in (
        (prepared, prep, {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}, "completed.json"),
        (run, done, {"started.json", "plan.json", "timings.jsonl", "scores.jsonl"}, "completed.json"),
        (report, receipt, {"started.json", "summary.json", "report.md"}, "receipt.json"),
    ):
        require(set(document["files"]) == members
                and {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
                == members | {terminal}, "Exact successful new closure")
        for name, descriptor in document["files"].items():
            bind(directory/name, descriptor)
        bindings[directory/terminal] = h.item(directory/terminal, check)
    require(h.sha(run/"plan.json", check) == args.plan_sha256 and set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
            and all(prep["files"][name] == descriptor for name, descriptor in plan["files"].items()), "New plan payload identity")
    expected_parent = {"prepared_path": str(old_prepared), "plan_sha256": BASE_PLAN, "completed_sha256": BASE_PREP,
                       "files": original["files"], "source_sha256": original["source_sha256"]}
    require(plan["parent"] == expected_parent and prep["parent_plan_sha256"] == BASE_PLAN
            and prep["parent_completed_sha256"] == BASE_PREP, "Exact paired preparation lineage")
    mutable = {"source_sha256", "files", "pilot_request_ids", "input_token_slots", "request_counts", "prompt_tokens"}
    require(set(plan) == set(original) | {"experiment_id", "protocol_sha256", "parent", "transformation"}
            and all(plan[k] == v for k, v in original.items() if k not in mutable), "Unchanged recipe and input descriptors")
    require(plan["request_counts"] == original["request_counts"]
            and plan["files"]["labels.jsonl"] == original["files"]["labels.jsonl"], "Byte-identical full evaluator cohort")
    require(set(original["source_sha256"]) == BASE_SOURCES
            and set(plan["source_sha256"]) == BASE_SOURCES | ADDED_SOURCES
            and all(plan["source_sha256"][k] == v for k, v in original["source_sha256"].items())
            and done["source_sha256"] == plan["source_sha256"]
            and done["runtime"] == plan["runtime"] and done["model"] == plan["model"], "Fixed source/runtime/model bindings")
    for mapping in (plan["source_sha256"], receipt["source_sha256"]):
        for name, digest in mapping.items():
            path = safe(name)
            require(h.sha(path, check) == digest, "Live source binding")
            bindings[path] = h.item(path, check)
    require(plan["protocol_sha256"] == plan["source_sha256"][PROTOCOL], "Protocol source identity")
    summary = h.read(report/"summary.json")
    require(summary["status"] == "completed" and summary["technical_validity_passed"] is True
            and summary["version"] == REPORT_VERSION and summary["experiment_id"] == EXPERIMENT
            and summary["plan_sha256"] == args.plan_sha256 and summary["execution_completed_sha256"] == args.run_sha256
            and summary["baseline_plan_sha256"] == BASE_PLAN and summary["baseline_run_sha256"] == BASE_RUN
            and summary["baseline_report_sha256"] == receipt["baseline_report_sha256"] == BASE_SUMMARY
            and summary["protocol_sha256"] == plan["protocol_sha256"], "Reported paired lineage")
    return old_prepared, old_run, original, prepared, run, plan, old_report, summary, receipt, bindings


def paired_membership(original, changed, h, check):
    count = 0
    with (original/"requests.jsonl").open() as a, (changed/"requests.jsonl").open() as b:
        for first, second in zip(a, b, strict=True):
            old, new = h.decode(first), h.decode(second)
            require({k: v for k, v in old.items() if k not in ("tokens", "request")}
                    == {k: v for k, v in new.items() if k not in ("tokens", "request")}, "Paired request/row/map membership")
            require(old["request"]["context"] == new["request"]["context"], "Paired public context identity")
            for q, r in zip(old["request"]["questions"], new["request"]["questions"], strict=True):
                require(q["id"] == r["id"] and [c["id"] for c in q["candidates"]]
                        == [c["id"] for c in r["candidates"]], "Paired candidate identity")
            count += 1
            check()
    require(count > 0, "Nonempty paired membership")
    return count


def continuation(original, no_flags):
    def rate(fit, stratum, metric, weighting):
        key = "correct" if metric == "accuracy" else "error"
        def cell(value):
            n, d = value["counts"][key], value["rows"]
            require(type(n) is int and type(d) is int and 0 <= n <= d, "Integral rate counts")
            return Fraction(n, d) if d else None
        if weighting == "row":
            return cell(fit["cells"][stratum])
        values = [cell(group[stratum]) for group in fit["services"].values() if group[stratum]["rows"]]
        return sum(values, Fraction())/len(values) if values else None
    checks = {}
    for arm in ARMS:
        require(set(original[arm]["services"]) == set(no_flags[arm]["services"]), "Same supported services")
        for stratum, metric, threshold in (("retained", "error", Fraction(-1, 50)),
                ("changed", "accuracy", Fraction(-1, 100)), ("all", "nll", 0.), ("all", "brier", 0.)):
            for weighting in ("row", "equal_service"):
                if metric in ("accuracy", "error"):
                    a, b = (rate(fit[arm], stratum, metric, weighting) for fit in (no_flags, original))
                else:
                    a, b = (fit[arm]["cells"][stratum]["metrics"][metric][weighting] for fit in (no_flags, original))
                delta = a-b if a is not None and b is not None else None
                passed = delta is not None and (delta >= threshold if metric == "accuracy" else delta <= threshold)
                checks[f"{arm}/{stratum}/{metric}/{weighting}"] = {"difference": None if delta is None else float(delta),
                    "threshold": float(threshold), "relation": ">=" if metric == "accuracy" else "<=", "passed": passed}
    arms = {arm: all(v["passed"] for k, v in checks.items() if k.startswith(arm+"/")) for arm in ARMS}
    return {"passed": all(arms.values()), "arms": arms, "checks_passed": sum(v["passed"] for v in checks.values()),
            "total_checks": 16, "checks": checks}


def compare(actual, expected, path=""):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), "Missing metric fields: "+path)
        return sum(compare(actual[k], value, path+"/"+k) for k, value in expected.items())
    if type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), "Arithmetic disagreement: "+path)
    else:
        require(type(actual) is type(expected) and actual == expected, "Exact disagreement: "+path)
    return 1


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def execute(args):
    start, handler, sources = time.monotonic(), None, {}
    out = Path(args.out).resolve()
    for name in ("baseline_prepared", "baseline_run", "baseline_report", "prepared", "run", "report"):
        path = Path(getattr(args, name)).resolve()
        require(not out.is_relative_to(path) and not path.is_relative_to(out), "Separate audit output")
    out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        if time.monotonic()-start > LIMITS["wall_seconds"]:
            raise TimeoutError("Saved-only audit wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Saved-only audit RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
    def expired(*_):
        raise TimeoutError("Saved-only audit wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        sources = {name: sha(ROOT/name, check) for name in SOURCES}
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
                                  "limits": LIMITS, "scope": SCOPE, "cpu_threads": 1})
        h = load_helper(check)
        op, orun, original, np, nrun, plan, old_report, report, receipt, bindings = authenticate(args, h, check)
        request_count = paired_membership(op, np, h, check)
        old_fit = h.calculate(h.reconstruct(op, orun, original, check))
        new_fit = h.calculate(h.reconstruct(np, nrun, plan, check))
        checks = compare(old_report["arms"], old_fit, "baseline_report")
        checks += compare(report["original"], old_fit, "original") + compare(report["no_flags"], new_fit, "no_flags")
        decision = continuation(old_fit, new_fit)
        checks += compare(report["continuation"], decision, "continuation")
        require(receipt["continuation_passed"] is decision["passed"], "Report receipt continuation")
        summary = {"status": "completed", "version": VERSION, "agreement": True, "scope": SCOPE,
            "original": old_fit, "no_flags": new_fit, "continuation": decision, "metric_cells": 84,
            "decision_checks": 16, "scalar_checks": checks, "paired_requests": request_count,
            "rows_per_arm": h.SUPPORT["all"], "plan_sha256": args.plan_sha256,
            "execution_completed_sha256": args.run_sha256, "producer_summary_sha256": args.report_summary_sha256}
        write(out/"summary.json", summary)
        for path, descriptor in bindings.items():
            require(h.item(path, check) == descriptor, "End input identity")
        require({name: sha(ROOT/name, check) for name in sources} == sources, "End audit source identity")
        files = {name: h.item(out/name, check) for name in ("started.json", "summary.json")}
        write(out/"receipt.json", {"status": "completed", "agreement": True, "version": VERSION, "scope": SCOPE,
            "continuation_passed": decision["passed"], "source_sha256": sources, "request": request, "files": files,
            "producer_receipt_sha256": args.report_receipt_sha256, "producer_summary_sha256": args.report_summary_sha256,
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
            "original_plan_sha256": BASE_PLAN, "original_execution_completed_sha256": BASE_RUN,
            "authenticated_inputs": {str(path): descriptor for path, descriptor in bindings.items()},
            "metric_cells": 84, "decision_checks": 16, "scalar_checks": checks,
            "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0, "cpu_threads": 1,
            "limits": LIMITS, "wall_seconds": time.monotonic()-start, "process_lifetime_peak_rss_bytes": peak_rss(),
            "wall_scope": "Entire audit through closing input/output hashes; terminal receipt write/return cap-checked."})
        check()
        return summary
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                "source_sha256": sources, "error": repr(error), "wall_seconds": time.monotonic()-start, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - retain the original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-prepared", "baseline-run", "baseline-report", "prepared", "plan-sha256", "run",
                 "run-sha256", "report", "report-receipt-sha256", "report-summary-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
