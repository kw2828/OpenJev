"""Paired saved-only lexical ablation report, reusing explicitly pinned old arithmetic."""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import time
from array import array
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-lexical-ablation-report-v1"
EXPERIMENT = "dialogue-qwen-lexical-ablation-v1"
HELPER = "scripts/report_dialogue_qwen_observation.py"
HELPER_SHA = "a387c0f3dbba40553b9fe24942dda30ed030c278a18303e761cfc6a89a7ccd1c"
TRANSFORM = "src/openjev/research/dialogue_qwen_lexical_ablation.py"
PROTOCOL = "research/dialogue-qwen-lexical-ablation-protocol.md"
ADDED_SOURCES = {TRANSFORM, PROTOCOL, "scripts/prepare_dialogue_qwen_lexical_ablation.py",
                 "tests/test_dialogue_qwen_lexical_ablation.py", "tests/test_prepare_dialogue_qwen_lexical_ablation.py"}
BASE_PLAN = "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111"
BASE_PREP = "6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f"
BASE_RUN = "872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c"
BASE_SUMMARY = "931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c"
BASE_REPORT_RECEIPT = "3106ca10cfa939fb47af41560772ae66a9308911cbfb5aa290dbe213f93e4ce2"
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
SCOPE = ("Paired saved-output development comparison on exposed official TRAIN with correct previous gold. "
         "Authentication, score reconstruction and metrics inherit the SHA-pinned original reporter; this is "
         "not an independent audit. Exact public subtraction is replayed, but tokenization, inference and "
         "unsaved full-vocabulary logits are not. Original outcomes are unchanged. Literal-register controls "
         "are original-input context and were unavailable to the no-flags actor. No architecture, calibration, "
         "autonomous-memory or paired speed claim.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_old(extended=False):
    require(sha(ROOT/HELPER) == HELPER_SHA, "Pinned original reporter source")
    module = load_module(ROOT/HELPER, "inherited_qwen_report_"+str(extended))
    if extended:
        # Only this newly loaded private instance changes its exact source set.
        module.SOURCES = module.SOURCES | ADDED_SOURCES
    return module


def same_tree(a, b, h):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), "Baseline report schema")
        for key in a:
            same_tree(a[key], b[key], h)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "Baseline report list")
        for x, y in zip(a, b, strict=True):
            same_tree(x, y, h)
    elif type(a) is float:
        h.close(a, b, "Baseline report agreement")
    else:
        require(type(a) is type(b) and a == b, "Baseline report exact agreement")


def validate_plan_pair(original, plan, prep, args, h):
    require(plan.get("experiment_id") == prep.get("experiment_id") == EXPERIMENT
            and plan.get("protocol_sha256") == args.protocol_sha256
            and plan["source_sha256"][PROTOCOL] == args.protocol_sha256, "New experiment/protocol identity")
    expected_parent = {"prepared_path": str(Path(args.baseline_prepared).resolve()), "plan_sha256": BASE_PLAN,
                       "completed_sha256": BASE_PREP, "files": original["files"], "source_sha256": original["source_sha256"]}
    require(plan["parent"] == expected_parent and prep["parent_plan_sha256"] == BASE_PLAN
            and prep["parent_completed_sha256"] == BASE_PREP, "Exact baseline preparation lineage")
    require(prep["labels_decoded"] is False and prep["baseline_scores_accessed"] is False
            and prep["checkpoint_deserializations"] == 0, "Preparation actor isolation")
    changed = {"source_sha256", "files", "pilot_request_ids", "input_token_slots", "request_counts", "prompt_tokens"}
    added = {"experiment_id", "protocol_sha256", "parent", "transformation"}
    require(set(plan) == set(original) | added, "Exact compatible plan schema")
    require(all(plan[k] == v for k, v in original.items() if k not in changed), "Unchanged original input/recipe descriptors")
    require(plan["request_counts"] == original["request_counts"]
            and plan["files"]["labels.jsonl"] == original["files"]["labels.jsonl"], "Complete byte-identical label cohort")
    require(all(plan["source_sha256"][k] == v for k, v in original["source_sha256"].items()), "Unchanged inherited source hashes")
    require(h.pin(args.protocol_sha256), "External protocol pin")


def authenticate(args, h, extended, check):
    base_args = SimpleNamespace(prepared=args.baseline_prepared, plan_sha256=BASE_PLAN,
                                run=args.baseline_run, run_sha256=BASE_RUN)
    require(h.sha(Path(args.baseline_prepared)/"completed.json", check) == BASE_PREP, "Fixed baseline preparation completion")
    base = h.authenticate(base_args, check)
    new = extended.authenticate(args, check)
    validate_plan_pair(base[0], new[0], new[1], args, h)
    require(h.sha(ROOT/PROTOCOL, check) == args.protocol_sha256, "External protocol source")
    report_path = Path(args.baseline_report).resolve()
    require(h.sha(report_path/"summary.json", check) == BASE_SUMMARY
            and h.sha(report_path/"receipt.json", check) == BASE_REPORT_RECEIPT, "Fixed original report pins")
    receipt, report = h.read(report_path/"receipt.json"), h.read(report_path/"summary.json")
    bindings = {**base[-1], **new[-1]}
    h.manifest(report_path, receipt["files"], {"started.json", "summary.json", "report.md"}, "receipt.json", bindings, check)
    require(receipt["status"] == report["status"] == "completed"
            and receipt["technical_validity_passed"] is report["technical_validity_passed"] is True
            and receipt["version"] == report["version"] == h.VERSION
            and receipt["plan_sha256"] == report["plan_sha256"] == BASE_PLAN
            and receipt["execution_completed_sha256"] == report["execution_completed_sha256"] == BASE_RUN
            and receipt["study_source_sha256"] == report["source_sha256"] == base[0]["source_sha256"]
            and receipt["model_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0,
            "Authenticated original report declarations")
    require(receipt["limits"] == h.LIMITS and receipt["no_retry"] is True
            and 0 < h.number(receipt["wall_seconds"]) <= h.LIMITS["wall_seconds"]
            and 0 < h.number(receipt["process_lifetime_peak_rss_bytes"]) <= h.LIMITS["rss_bytes"]
            and receipt["source_sha256"][HELPER] == HELPER_SHA,
            "Original report resource/source declarations")
    for name, digest in receipt["source_sha256"].items():
        require(h.sha(h.safe(ROOT, name), check) == digest, "Original report source binding")
        bindings[ROOT/name] = h.item(ROOT/name, check)
    transform_path = ROOT/TRANSFORM
    require(h.sha(transform_path, check) == new[0]["source_sha256"][TRANSFORM], "Sealed transform source")
    transform = load_module(transform_path, "sealed_lexical_subtraction")
    return base, new, report, transform, bindings


def lines(path, h, check):
    with Path(path).open() as stream:
        for n, line in enumerate(stream):
            if n % 100 == 0:
                check()
            yield h.decode(line)


def paired_requests(base_path, new_path, transform, h, check):
    """Compare exact actor records before lossless token compaction and dropping compared context."""
    original, new = [], []
    for old, changed in zip(lines(base_path, h, check), lines(new_path, h, check), strict=True):
        require(transform.verify_transform(old, changed), "Exact prescribed lexical subtraction")
        for record, destination in ((old, original), (changed, new)):
            require(all(type(v) is list and 1 <= len(v) <= 4096
                        and all(type(t) is int and 0 <= t < 151936 for t in v) for v in record["tokens"]), "Finite legal token IDs")
            record["tokens"] = [array("I", v) for v in record["tokens"]]
            # Context already compared byte-for-character. Metrics require only question identity.
            record["request"]["context"] = ""
            destination.append(record)
    require(len(original) > 0, "Nonempty paired requests")
    return original, new


def validate_new_requests(requests, plan, h):
    require({a: sum(h.work_for(r)["input_token_slots"] for r in requests if r["arm"] == a) for a in h.ARMS}
            == plan["input_token_slots"] and {a: sum(r["arm"] == a for r in requests) for a in h.ARMS} == plan["request_counts"],
            "Retokenized full workload")
    profile = {a: {"min": min(len(v) for r in requests if r["arm"] == a for v in r["tokens"]),
                   "max": max(len(v) for r in requests if r["arm"] == a for v in r["tokens"]),
                   "sum": sum(len(v) for r in requests if r["arm"] == a for v in r["tokens"])} for a in h.ARMS}
    require(profile == plan["prompt_tokens"], "Retokenized prompt profile")
    groups = sorted({(r["dialogue_id"], r["time"]) for r in requests})
    selected = set(groups[:12])
    for arm in h.ARMS:
        longest = min((r for r in requests if r["arm"] == arm), key=lambda r: (-max(map(len, r["tokens"])), r["dialogue_id"], r["time"], r["request_id"]))
        selected.add((longest["dialogue_id"], longest["time"]))
    require(plan["pilot_request_ids"] == [r["request_id"] for r in requests if (r["dialogue_id"], r["time"]) in selected], "Fresh pilot membership")


def reconstruction(chain, requests, layouts, rows, run, h, check):
    plan, _, done, pilot, pilot_path, _, _ = chain
    timings = list(lines(Path(run)/"timings.jsonl", h, check))
    h.validate_timings(timings, requests, done)
    pilot_timings = list(lines(pilot_path/"timings.jsonl", h, check))
    h.validate_timings(pilot_timings, [r for r in requests if r["request_id"] in plan["pilot_request_ids"]], pilot)
    h.validate_projection(pilot, pilot_timings, plan)
    return h.reconstruct(requests, list(lines(Path(run)/"scores.jsonl", h, check)), timings, layouts, rows)


def continuation(original, new, h):
    checks, arms = {}, {}
    for arm in h.ARMS:
        for stratum, metric, threshold in (("retained", "error", Fraction(-1, 50)), ("changed", "accuracy", Fraction(-1, 100)),
                                           ("all", "nll", 0.), ("all", "brier", 0.)):
            for weighting in ("row", "equal_service"):
                if metric in ("accuracy", "error"):
                    delta = h.exact_rate(new[arm], stratum, metric, weighting)-h.exact_rate(original[arm], stratum, metric, weighting)
                else:
                    delta = new[arm]["cells"][stratum]["metrics"][metric][weighting]-original[arm]["cells"][stratum]["metrics"][metric][weighting]
                passed = delta >= threshold if metric == "accuracy" else delta <= threshold
                checks[f"{arm}/{stratum}/{metric}/{weighting}"] = {"difference": float(delta), "threshold": float(threshold),
                     "relation": ">=" if metric == "accuracy" else "<=", "passed": bool(passed)}
        arms[arm] = all(v["passed"] for k, v in checks.items() if k.startswith(arm+"/"))
    return {"passed": all(arms.values()), "arms": arms, "checks_passed": sum(v["passed"] for v in checks.values()),
            "total_checks": 16, "checks": checks}


def paired(rows, original, new, h):
    groups, result = h.masks(rows), {}
    services = sorted({r["service"] for r in rows})
    target = np.asarray([r["target"] for r in rows])
    for arm in h.ARMS:
        old_logs, old_choice = original[0][arm], original[1][arm]
        logs, choice = new[0][arm], new[1][arm]
        a, b = h.vectors(rows, logs, choice), h.vectors(rows, old_logs, old_choice)
        def cell(mask, a=a, b=b, choice=choice, old_choice=old_choice):
            ac, bc = choice == target, old_choice == target
            return {"rows": int(mask.sum()), "wrong_to_correct": int((mask & ac & ~bc).sum()),
                    "correct_to_wrong": int((mask & ~ac & bc).sum()), "both_correct": int((mask & ac & bc).sum()),
                    "both_wrong": int((mask & ~ac & ~bc).sum()),
                    "metric_differences": {k: h.means(a[k]-b[k], rows, mask) for k in a}}
        type_axes = {"previous": [r["types"][r["previous"]] for r in rows], "target": [r["types"][r["target"]] for r in rows],
                     "original_selected": [r["types"][c] for r, c in zip(rows, old_choice, strict=True)],
                     "no_flags_selected": [r["types"][c] for r, c in zip(rows, choice, strict=True)]}
        result[arm] = {"cells": {g: cell(m) for g, m in groups.items()},
                      "services": {s: {g: cell(m & np.asarray([r["service"] == s for r in rows])) for g, m in groups.items()} for s in services},
                      "candidate_types": {axis: {name: {g: cell(m & (np.asarray(types) == code)) for g, m in groups.items()}
                                          for code, name in enumerate(h.TYPE_NAMES)} for axis, types in type_axes.items()}}
    return result


def costs(chain):
    plan, prep, done, pilot, _, _, _ = chain
    return {"preparation_seconds": prep["wall_seconds"], "pilot_seconds": pilot["wall_seconds"], "run_seconds": done["wall_seconds"],
            "combined_seconds": prep["wall_seconds"]+pilot["wall_seconds"]+done["wall_seconds"],
            "run_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"], "pilot_projection": pilot["projection"],
            "pilot_work": pilot["work_totals"], "run_work": done["work_totals"], "input_token_slots": plan["input_token_slots"]}


def render_text(summary):
    rule = summary["continuation"]
    lines = ["# Fixed lexical-input ablation", "", SCOPE, "",
             f"Development continuation {'PASS' if rule['passed'] else 'FAIL'}: {rule['checks_passed']}/16 components.", "",
             "| Arm | Component | Difference | Requirement | Result |", "|---|---|---:|---:|---|"]
    for key, v in rule["checks"].items():
        arm, component = key.split("/", 1)
        lines.append(f"| {arm} | {component} | {v['difference']:+.8f} | {v['relation']} {v['threshold']} | {'PASS' if v['passed'] else 'FAIL'} |")
    lines.extend(("", "Rates above are fractions, not percentage points. Full stratum/service/type metrics and paired repairs/harms are saved in summary.json.",
                  "Baseline timing is contextual, not interleaved or a speed benchmark. No earlier scientific decision is replaced."))
    return "\n".join(lines)+"\n"


def execute(args):
    out = Path(args.out).resolve()
    require(all(not out.is_relative_to(Path(p).resolve()) and not Path(p).resolve().is_relative_to(out)
                for p in (args.prepared, args.run, args.baseline_prepared, args.baseline_run, args.baseline_report)), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    start, previous, sources, h = time.monotonic(), None, {}, None
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        require(time.monotonic()-start <= LIMITS["wall_seconds"], "Report wall cap")
        if h is not None:
            require(h.peak_rss() <= LIMITS["rss_bytes"], "Report RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Report output cap")
    def timeout(*_):
        raise TimeoutError("Report wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing timer")
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        h, extended = load_old(), load_old(True)
        source_names = {"scripts/report_dialogue_qwen_lexical_ablation.py", "tests/test_report_dialogue_qwen_lexical_ablation.py", HELPER, TRANSFORM, PROTOCOL}
        sources = {n: h.sha(ROOT/n, check) for n in source_names}
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources, "scope": SCOPE, "limits": LIMITS})
        original, new, reported, transform, bindings = authenticate(args, h, extended, check)
        old_requests, new_requests = paired_requests(Path(args.baseline_prepared)/"requests.jsonl", Path(args.prepared)/"requests.jsonl", transform, h, check)
        labels = list(lines(Path(args.baseline_prepared)/"labels.jsonl", h, check))
        rows, layouts = h.request_layout(old_requests, labels, original[0])
        validate_new_requests(new_requests, new[0], h)
        old_values = reconstruction(original, old_requests, layouts, rows, args.baseline_run, h, check)
        new_values = reconstruction(new, new_requests, layouts, rows, args.run, h, check)
        old_metrics = h.aggregate(rows, old_values[0], old_values[1], original[5])
        new_metrics = h.aggregate(rows, new_values[0], new_values[1], original[5])
        for key in ("arms", "controls", "paired", "decisions", "support", "services"):
            same_tree(old_metrics[key], reported[key], h)
        same_tree(old_values[2], reported["validations"], h)
        summary = {"status": "completed", "version": VERSION, "experiment_id": EXPERIMENT, "technical_validity_passed": True,
             "scope": SCOPE, "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
             "protocol_sha256": args.protocol_sha256, "baseline_plan_sha256": BASE_PLAN, "baseline_run_sha256": BASE_RUN,
             "baseline_report_sha256": BASE_SUMMARY, "support": old_metrics["support"], "services": old_metrics["services"],
             "original": old_metrics["arms"], "no_flags": new_metrics["arms"], "paired": paired(rows, old_values, new_values, h),
             "original_input_descriptive_controls": old_metrics["controls"], "historical_context": old_metrics["historical"],
             "original_study_decisions_unchanged": reported["decisions"], "continuation": continuation(old_metrics["arms"], new_metrics["arms"], h),
             "validations": {"original": old_values[2], "no_flags": new_values[2]},
             "costs": {"original_context_only": costs(original), "no_flags": costs(new),
                       "token_slot_reduction": {a: original[0]["input_token_slots"][a]-new[0]["input_token_slots"][a] for a in h.ARMS},
                       "scope": "Whole phase costs include load/auth/IO; pilot repetitions separately paid. Earlier baseline not a paired timing benchmark."}}
        write(out/"summary.json", summary)
        (out/"report.md").write_text(render_text(summary))
        for path, descriptor in bindings.items():
            require(h.item(path, check) == descriptor, "End input identity")
        require(sources == {n: h.sha(ROOT/n, check) for n in sources}, "End source identity")
        check()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "experiment_id": EXPERIMENT,
             "technical_validity_passed": True, "continuation_passed": summary["continuation"]["passed"], "scope": SCOPE,
             "request": request, "source_sha256": sources, "inherited_reporter_sha256": HELPER_SHA,
             "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256, "baseline_report_sha256": BASE_SUMMARY,
             "files": {n: h.item(out/n, check) for n in ("started.json", "summary.json", "report.md")},
             "input_members": {str(p): v for p, v in bindings.items()}, "limits": LIMITS,
             "wall_seconds": time.monotonic()-start, "process_lifetime_peak_rss_bytes": h.peak_rss(), "cpu_threads": 1,
             "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0, "no_retry": True})
        check()
        return summary
    except BaseException as error:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request, "source_sha256": sources,
                  "error": repr(error), "wall_seconds": time.monotonic()-start, "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original error
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-prepared", "baseline-run", "baseline-report", "prepared", "plan-sha256", "run", "run-sha256", "protocol-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
