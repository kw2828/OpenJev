"""Bounded post-result retention diagnosis; no inference or independent-audit claim.

Authentication and distribution reconstruction deliberately reuse the pinned
independent result auditor. This script adds descriptive partitions and fixed
blinded packets. It does not change a prediction, threshold or original result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import resource
import signal
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-retention-diagnostic-v1"
HELPER = "scripts/audit_dialogue_qwen_result.py"
HELPER_SHA = "ccc24ad9555ff5a45281fcb02521d7c0ee0d0b45697042edb4d69947a7245b70"
PROTOCOL = "research/dialogue-qwen-retention-diagnostic-protocol.md"
INSTRUCTIONS = "research/dialogue-qwen-review-instructions.md"
INSTRUCTIONS_SHA = "c1fe23c47ae415e43697b3fea86399e17bcc714d531119f39fc68a53d7a8d878"
SOURCES = ("scripts/diagnose_dialogue_qwen_retention.py", "tests/test_diagnose_dialogue_qwen_retention.py",
           PROTOCOL, INSTRUCTIONS, HELPER)
PINS = {"plan_sha256": "2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111",
        "run_sha256": "872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c",
        "report_summary_sha256": "931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c"}
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
ARMS = ("current", "history4")
NONE, DC = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
TYPES = ("NOT_MENTIONED", "DONTCARE", "TRUE", "FALSE", "OTHER")
SUBTYPES = ("first_assignment", "revision", "clear", "unmentioned_retention", "assigned_retention")
BINS = ("[0,.01)", "[.01,.1)", "[.1,.5)", "[.5,.9)", "[.9,.99)", "[.99,1]")
FLAGS = ("latest_user", "latest_system", "older_history4_user", "older_history4_system")
REVIEW_STRATA = ("current_error_retained_true", "current_error_retained_other",
                "current_error_retained_dontcare", "unmentioned_current_correct_history4_wrong",
                "current_error_changed_true", "current_error_changed_other")
SCOPE = (
    "Post-result descriptive analysis of the complete frozen cohort. Authentication, canonical score "
    "reconstruction and main-report reconciliation reuse the pinned independent Qwen result auditor; "
    "this diagnostic is not another independent audit. Inference, tokenization, public-input provenance "
    "and historical metrics inherit that authenticated chain. No threshold fitting, probability repair, "
    "new model, reviewer execution or achieved-performance claim. Original failed decisions remain unchanged."
)
VISIBILITY = {arm: {flag: arm == "history4" or not flag.startswith("older_") for flag in FLAGS} for arm in ARMS}


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


def item(path, check=lambda: None):
    return {"sha256": sha(path, check), "bytes": Path(path).stat().st_size}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def json_lines(path, records, check=lambda: None):
    with Path(path).open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
            check()


def load_helper(check=lambda: None):
    path = ROOT/HELPER
    require(sha(path, check) == HELPER_SHA, "Pinned inherited auditor")
    spec = importlib.util.spec_from_file_location("qwen_retention_inherited_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_type(cid):
    if cid == NONE:
        return "NOT_MENTIONED"
    if cid == DC:
        return "DONTCARE"
    require(isinstance(cid, str) and cid.startswith("value:") and len(cid) > 6, "Canonical literal candidate ID")
    value = cid[6:].casefold()
    return value.upper() if value in ("true", "false") else "OTHER"


def subtype(previous, target):
    if previous == target:
        return "unmentioned_retention" if target == NONE else "assigned_retention"
    return "first_assignment" if previous == NONE else "clear" if target == NONE else "revision"


def prior_bin(p):
    require(math.isfinite(p) and 0 <= p <= 1, "Previous probability range")
    return BINS[next((i for i, upper in enumerate((.01, .1, .5, .9, .99)) if p < upper), 5)]


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def context_exchanges(context, arm):
    value = json.loads(context)
    require(isinstance(value, dict) and set(value) == {"exchanges_oldest_first"}, "Public context envelope")
    exchanges = value["exchanges_oldest_first"]
    require(isinstance(exchanges, list) and 1 <= len(exchanges) <= (1 if arm == "current" else 4),
            "Frozen arm context length")
    require(all(isinstance(x, dict) and set(x) == {"USER", "SYSTEM"}
                and all(isinstance(v, str) for v in x.values()) for x in exchanges), "Public roles/text only")
    return exchanges


def literal_flags(cid, exchanges):
    if candidate_type(cid) != "OTHER":
        return dict.fromkeys(FLAGS, None)
    literal = normalized(cid[6:])
    require(bool(literal), "Nonempty normalized literal")
    pattern = re.compile(r"(?<!\w)"+re.escape(literal)+r"(?!\w)")
    def occurs(text):
        return bool(pattern.search(normalized(text)))
    # Never join utterances: a literal spanning two different turns is not an occurrence.
    return {"latest_user": occurs(exchanges[-1]["USER"]),
            "latest_system": occurs(exchanges[-1]["SYSTEM"]),
            "older_history4_user": any(occurs(x["USER"]) for x in exchanges[:-1]),
            "older_history4_system": any(occurs(x["SYSTEM"]) for x in exchanges[:-1])}


def derive(prepared, run, helper, validated, check=lambda: None):
    """Called only after byte authentication and inherited full score validation."""
    labels = {r["row_index"]: r for r in map(helper.decode, (prepared/"labels.jsonl").read_text().splitlines())}
    public, rows = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    with (prepared/"requests.jsonl").open() as requests, (run/"scores.jsonl").open() as scores:
        for request_line, score_line in zip(requests, scores, strict=True):
            request, score = helper.decode(request_line), helper.decode(score_line)
            arm = request["arm"]
            require(score["request_id"] == request["request_id"], "Repeated request identity")
            exchanges = context_exchanges(request["request"]["context"], arm)
            for index, question, mapping, saved in zip(request["row_indices"], request["request"]["questions"],
                                                     request["canonical_id_maps"], score["questions"], strict=True):
                label, base = labels[index], validated[arm][index]
                ids, logits, probabilities = (saved[k] for k in ("candidate_ids", "candidate_logits", "probabilities"))
                previous, target, selected = label["previous_candidate_id"], label["current_candidate_id"], saved["selected_id"]
                kind = subtype(previous, target)
                target_group = "none" if target == NONE else candidate_type(target).lower()
                require(label["derived_bin"] == kind and label["current_value_group"] == target_group,
                        "Evaluator transition/type identity")
                pi, si = ids.index(previous), ids.index(selected)
                require(base["accuracy"] == int(selected == target) and base["changed"] == (previous != target),
                        "Inherited reconstruction agreement")
                public[arm][index] = {"context": request["request"]["context"], "question": question["question"],
                                      "candidates": question["candidates"], "canonical_id_map": mapping,
                                      "exchanges": exchanges}
                rows[arm][index] = {"arm": arm, "row_index": index, "dialogue_id": label["dialogue_id"],
                    "time": label["time"], "service": label["service"], "previous_id": previous,
                    "target_id": target, "selected_id": selected, "previous_type": candidate_type(previous),
                    "target_type": candidate_type(target), "selected_type": candidate_type(selected),
                    "changed": previous != target, "subtype": kind, "correct": selected == target,
                    "previous_probability": probabilities[pi], "previous_probability_bin": prior_bin(probabilities[pi]),
                    "previous_competition_rank": 1+sum(z > logits[pi] for z in logits),
                    "selected_probability": probabilities[si]}
            check()
    require(set(rows["current"]) == set(rows["history4"]) == set(labels), "Both complete diagnostic arms")
    for index in labels:
        c, h = public["current"][index], public["history4"][index]
        require(c["exchanges"][-1] == h["exchanges"][-1], "Latest public exchange shared by arms")
        require(all(c[k] == h[k] for k in ("question", "candidates", "canonical_id_map")), "Matched public question/candidates")
        for arm in ARMS:
            rows[arm][index]["literal_flags"] = literal_flags(rows[arm][index]["selected_id"], h["exchanges"])
            rows[arm][index]["literal_actor_visibility"] = VISIBILITY[arm]
        check()
    return rows, public


def describe(rows):
    n = len(rows)
    bins = {key: 0 for key in BINS}
    ranks = {str(i): 0 for i in range(1, 13)}
    flags = {key: {"true": 0, "false": 0, "not_applicable": 0} for key in FLAGS}
    for row in rows:
        bins[row["previous_probability_bin"]] += 1
        ranks[str(row["previous_competition_rank"])] += 1
        for key, value in row["literal_flags"].items():
            flags[key]["not_applicable" if value is None else "true" if value else "false"] += 1
    correct = sum(r["correct"] for r in rows)
    return {"rows": n, "correct": correct, "error": n-correct,
            "previous_probability_mean": math.fsum(r["previous_probability"] for r in rows)/n if n else None,
            "previous_competition_rank_mean": math.fsum(r["previous_competition_rank"] for r in rows)/n if n else None,
            "selected_probability_mean": math.fsum(r["selected_probability"] for r in rows)/n if n else None,
            "previous_probability_bins": bins, "previous_competition_ranks": ranks, "literal_flags": flags}


def aggregate(rows, reported):
    services = sorted({r["service"] for r in rows["current"].values()})
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for service in ["__all__", *services]:
            group = [r for r in rows[arm].values() if service == "__all__" or r["service"] == service]
            buckets = defaultdict(list)
            for row in group:
                key = (row["previous_type"], row["target_type"], row["selected_type"], row["subtype"], row["correct"])
                buckets[key].append(row)
            cells = {"all": describe(group), "changed": describe([r for r in group if r["changed"]]),
                     "retained": describe([r for r in group if not r["changed"]])}
            cells.update({name: describe([r for r in group if r["subtype"] == name]) for name in SUBTYPES})
            original = reported["arms"][arm]["cells"] if service == "__all__" else reported["arms"][arm]["services"][service]
            for name in ("all", "changed", "retained", "unmentioned_retention", "assigned_retention"):
                require(cells[name]["rows"] == original[name]["rows"]
                        and all(cells[name][k] == original[name]["counts"][k] for k in ("correct", "error")),
                        "Main report count reconciliation")
            result[arm][service] = {"cells": cells, "typed_groups": [
                {"previous_type": key[0], "target_type": key[1], "selected_type": key[2],
                 "subtype": key[3], "correct": key[4], "statistics": describe(values)}
                for key, values in sorted(buckets.items())]}
            require(sum(x["statistics"]["rows"] for x in result[arm][service]["typed_groups"]) == len(group),
                    "Exhaustive sparse partition")
    return result


def review_stratum(current, history):
    if current["subtype"] == "unmentioned_retention" and current["correct"] and not history["correct"]:
        return REVIEW_STRATA[3]
    if current["correct"]:
        return None
    if not current["changed"]:
        return {"TRUE": REVIEW_STRATA[0], "OTHER": REVIEW_STRATA[1], "DONTCARE": REVIEW_STRATA[2]}.get(current["target_type"])
    return {"TRUE": REVIEW_STRATA[4], "OTHER": REVIEW_STRATA[5]}.get(current["target_type"])


def selection(rows, public):
    eligible = {s: [] for s in REVIEW_STRATA}
    for index, current in rows["current"].items():
        name = review_stratum(current, rows["history4"][index])
        if name:
            eligible[name].append(index)
    def order(index):
        return hashlib.sha256(f"openjev-retention-diagnostic-v1:{index}".encode()).hexdigest()
    packets = {arm: {"arm": arm, "review_instructions_sha256": INSTRUCTIONS_SHA, "cases": []} for arm in ARMS}
    ledger, counts = [], {}
    for name, indices in eligible.items():
        chosen, dialogues = [], set()
        for index in sorted(indices, key=order):
            dialogue = rows["current"][index]["dialogue_id"]
            if dialogue not in dialogues:
                chosen.append(index)
                dialogues.add(dialogue)
            if len(chosen) == 2:
                break
        counts[name] = {"eligible_rows": len(indices),
            "eligible_dialogues": len({rows["current"][i]["dialogue_id"] for i in indices}), "selected_rows": len(chosen)}
        for index in chosen:
            case_id = hashlib.sha256(f"openjev-retention-review-case-v1:{index}".encode()).hexdigest()
            ledger.append({"case_id": case_id, "selection_stratum": name, "selection_sha256": order(index),
                           "rows": {arm: rows[arm][index] for arm in ARMS},
                           "canonical_id_map": public["current"][index]["canonical_id_map"]})
            for arm in ARMS:
                actor = public[arm][index]
                packets[arm]["cases"].append({"case_id": case_id, **{k: actor[k] for k in ("context", "question", "candidates")}})
    for packet in packets.values():
        packet["cases"].sort(key=lambda c: c["case_id"])
    ledger.sort(key=lambda c: c["case_id"])
    require(len({x["case_id"] for x in ledger}) == len(ledger), "Unique opaque review cases")
    return packets, ledger, counts


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def execute(args):
    started, sources, old_handler = time.monotonic(), {}, None
    out = Path(args.out).resolve()
    for key in ("prepared", "run", "report"):
        path = Path(getattr(args, key)).resolve()
        require(not out.is_relative_to(path) and not path.is_relative_to(out), "Separate diagnostic output")
    out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        if time.monotonic()-started > LIMITS["wall_seconds"]:
            raise TimeoutError("Diagnostic wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Diagnostic RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Diagnostic output cap")
    def expired(*_):
        raise TimeoutError("Diagnostic wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        old_handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        sources = {name: sha(ROOT/name, check) for name in SOURCES}
        require(sources[PROTOCOL] == args.protocol_sha256, "External protocol pin")
        require(sources[INSTRUCTIONS] == INSTRUCTIONS_SHA, "Fixed neutral review instructions")
        require(all(getattr(args, name) == digest for name, digest in PINS.items()), "Fixed study pins")
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
                                  "limits": LIMITS, "scope": SCOPE})
        helper = load_helper(check)
        prepared, run, plan, historical, reported, bindings = helper.authenticate(args, check)
        validated = helper.reconstruct(prepared, run, plan, check)
        _, inherited_checks = helper.verify(helper.calculate(validated), historical, reported)
        rows, public = derive(prepared, run, helper, validated, check)
        groups = aggregate(rows, reported)
        packets, ledger, counts = selection(rows, public)
        summary = {"version": VERSION, "status": "completed", "scope": SCOPE,
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
            "report_summary_sha256": args.report_summary_sha256, "services": sorted(set(groups["current"])-{"__all__"}),
            "arms": groups, "row_records": sum(map(len, rows.values())), "selection": counts,
            "literal_actor_visibility": VISIBILITY, "original_decisions": reported["decisions"],
            "inherited_reconciliation_scalar_checks": inherited_checks,
            "sparse_table_convention": {"missing_group_count": 0, "missing_group_means": None,
                "type_order": TYPES, "subtype_order": SUBTYPES, "correctness": [False, True]},
            "literal_scope": "OTHER only; overlapping occurrence flags are not entailment. Older history4 text is "
                "diagnostic-only for current. No occurrence does not exclude evidence in prior state or earlier dialogue.",
            "review_scope": "Error-conditioned fixed sample, not prevalence; two separate arm packets. "
                "Evaluator ledger and full row ledger must not be shown to blinded reviewers. No reviewers executed."}
        json_lines(out/"rows.jsonl", (rows[a][i] for a in ARMS for i in sorted(rows[a])), check)
        json_lines(out/"evaluator-selection.jsonl", ledger, check)
        for arm in ARMS:
            write(out/f"{arm}-review.json", packets[arm])
        write(out/"summary.json", summary)
        for path, descriptor in bindings.items():
            require(helper.item(path, check) == descriptor, "End input identity")
        require({name: sha(ROOT/name, check) for name in SOURCES} == sources, "End source identity")
        files = {p.name: item(p, check) for p in out.iterdir() if p.is_file()}
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "scope": SCOPE,
            "request": request, "source_sha256": sources, "files": files,
            "authenticated_inputs": {str(p): d for p, d in bindings.items()},
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.run_sha256,
            "report_summary_sha256": args.report_summary_sha256, "report_receipt_sha256": args.report_receipt_sha256,
            "row_records": summary["row_records"], "selected_cases": len(ledger),
            "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0, "reviewer_calls": 0,
            "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": peak_rss(), "limits": LIMITS,
            "wall_scope": "Whole diagnosis including authentication, reconstruction, writing and closing input/output hashes; "
                "terminal receipt write and return are cap-checked."})
        check()
        return summary
    except BaseException as error:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "source_sha256": sources, "error": repr(error), "wall_seconds": time.monotonic()-started})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "plan-sha256", "run", "run-sha256", "report", "report-receipt-sha256",
                 "report-summary-sha256", "protocol-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
