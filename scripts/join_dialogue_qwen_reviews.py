"""Join four terminal blinded reviews to saved evaluator metadata, with no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-review-join-v1"
DISPATCH_VERSION = "dialogue-qwen-blinded-review-v1"
DIAGNOSTIC_VERSION = "dialogue-qwen-retention-diagnostic-v1"
ARMS = ("current", "history4")
REVIEW_IDS = tuple(f"{arm}-{n:02}" for arm in ARMS for n in (1, 2))
INSTRUCTIONS = "research/dialogue-qwen-review-instructions.md"
PROTOCOL = "research/dialogue-qwen-retention-diagnostic-protocol.md"
SOURCES = ("scripts/join_dialogue_qwen_reviews.py", "tests/test_join_dialogue_qwen_reviews.py", INSTRUCTIONS, PROTOCOL)
DIAGNOSTIC_FILES = {"started.json", "summary.json", "rows.jsonl", "evaluator-selection.jsonl",
                    "current-review.json", "history4-review.json"}
LIMITS = {"wall_seconds": 60, "rss_bytes": 2*1024**3, "output_bytes": 64*1024**2}
SCOPE = ("Post-result interpretation of a fixed error-conditioned sample, not prevalence, annotation truth, "
         "model efficacy or a new success gate. All structurally valid non-null answers from completed reviews count in "
         "agreement denominators; ambiguous answers remain included and are separately identified. Missing, "
         "null, invalid and failed-review answers are preserved, without replacement. Quote matching checks "
         "literal presence, not entailment, and does not filter agreement denominators. Pre-dispatch freezing, reviewer independence and read isolation "
         "are parent orchestration witnesses, not independently reconstructed from saved files.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def read(path):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def item(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            check()
    return {"sha256": h.hexdigest(), "bytes": Path(path).stat().st_size}


def local(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe relative path")
    path = root/relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Escaping path")
    return path


def bind(path, descriptor, bindings, check):
    require(set(descriptor) == {"sha256", "bytes"} and pin(descriptor["sha256"])
            and type(descriptor["bytes"]) is int and descriptor["bytes"] >= 0, "File descriptor")
    require(item(path, check) == descriptor, "File identity: "+str(path))
    bindings[path] = descriptor


def terminals_first(reviews, diagnostic_pin, bindings, check):
    """This gate precedes every response/evaluator read, including byte hashing."""
    dispatch_path = local(reviews, "dispatch.json")
    bindings[dispatch_path] = item(dispatch_path, check)
    dispatch = read(dispatch_path)
    require(set(dispatch) == {"version", "model", "instructions", "diagnostic_receipt_sha256", "reviews"}
            and dispatch["version"] == DISPATCH_VERSION and dispatch["model"] == "gpt-6-astra"
            and dispatch["diagnostic_receipt_sha256"] == diagnostic_pin, "Frozen dispatch identity")
    entries = dispatch["reviews"]
    require(type(entries) is list and len(entries) == 4
            and {e["review_id"] for e in entries} == set(REVIEW_IDS), "Exactly four assigned reviews")
    for e in entries:
        rid = e["review_id"]
        require(set(e) == {"review_id", "arm", "packet_path", "packet_sha256", "response_file", "terminal_file"}
                and e["arm"] == rid.rsplit("-", 1)[0] and pin(e["packet_sha256"])
                and Path(e["packet_path"]).is_absolute() and e["response_file"] == rid+".json"
                and e["terminal_file"] == rid+"-terminal.json", "Assigned review schema")
    require(all(local(reviews, e["terminal_file"]).is_file() for e in entries), "All four terminal records required")
    terminals = {}
    for e in entries:
        path = local(reviews, e["terminal_file"])
        bindings[path] = item(path, check)
        terminal = read(path)
        require(set(terminal) == {"review_id", "agent_name", "status", "note"}
                and terminal["review_id"] == e["review_id"] and terminal["status"] in ("completed", "failed")
                and ((type(terminal["agent_name"]) is str and bool(terminal["agent_name"]))
                     or terminal["status"] == "failed" and terminal["agent_name"] is None)
                and type(terminal["note"]) is str, "Valid parent terminal record")
        terminals[e["review_id"]] = terminal
    return dispatch, terminals


def authenticate(args, check):
    diagnostic, reviews = Path(args.diagnostic).resolve(), Path(args.reviews).resolve()
    require(pin(args.diagnostic_receipt_sha256), "External diagnostic pin")
    bindings = {}
    dispatch, terminals = terminals_first(reviews, args.diagnostic_receipt_sha256, bindings, check)
    receipt_path = local(diagnostic, "receipt.json")
    descriptor = item(receipt_path, check)
    require(descriptor["sha256"] == args.diagnostic_receipt_sha256, "External diagnostic receipt identity")
    bindings[receipt_path] = descriptor
    receipt = read(receipt_path)
    require(receipt["status"] == "completed" and receipt["version"] == DIAGNOSTIC_VERSION
            and receipt["model_calls"] == receipt["tokenizer_calls"] == receipt["checkpoint_deserializations"]
            == receipt["reviewer_calls"] == 0, "Completed saved diagnostic")
    require(set(receipt["files"]) == DIAGNOSTIC_FILES
            and {p.relative_to(diagnostic).as_posix() for p in diagnostic.rglob("*") if p.is_file()}
            == DIAGNOSTIC_FILES | {"receipt.json"}, "Exact diagnostic closure")
    for name, desc in receipt["files"].items():
        bind(local(diagnostic, name), desc, bindings, check)
    for name, digest in receipt["source_sha256"].items():
        path = local(ROOT, name)
        require(pin(digest) and item(path, check)["sha256"] == digest, "Diagnostic source identity")
        bindings[path] = item(path, check)
    instruction = dispatch["instructions"]
    require(set(instruction) == {"path", "sha256"} and Path(instruction["path"]).is_absolute()
            and Path(instruction["path"]).resolve() == (ROOT/INSTRUCTIONS).resolve()
            and instruction["sha256"] == receipt["source_sha256"][INSTRUCTIONS], "Neutral instructions identity")
    packets = {}
    for arm in ARMS:
        path = local(diagnostic, arm+"-review.json")
        for e in (e for e in dispatch["reviews"] if e["arm"] == arm):
            require(Path(e["packet_path"]).resolve() == path and e["packet_sha256"] == bindings[path]["sha256"],
                    "Assigned packet identity")
        packet = read(path)
        require(set(packet) == {"arm", "review_instructions_sha256", "cases"} and packet["arm"] == arm
                and packet["review_instructions_sha256"] == instruction["sha256"], "Packet identity")
        require(type(packet["cases"]) is list and len(packet["cases"]) == receipt["selected_cases"] <= 12,
                "Fixed selected case count")
        require([c["case_id"] for c in packet["cases"]] == sorted({c["case_id"] for c in packet["cases"]}),
                "Unique sorted cases")
        for case in packet["cases"]:
            require(set(case) == {"case_id", "context", "question", "candidates"} and pin(case["case_id"])
                    and type(case["context"]) is str and type(case["question"]) is str, "Blinded case schema")
            require(type(case["candidates"]) is list and len(case["candidates"]) > 1
                    and all(set(c) == {"id", "description"} and all(type(v) is str for v in c.values()) for c in case["candidates"])
                    and len({c["id"] for c in case["candidates"]}) == len(case["candidates"]), "Public candidates")
        packets[arm] = packet
    require([c["case_id"] for c in packets["current"]["cases"]] == [c["case_id"] for c in packets["history4"]["cases"]],
            "Matched arm cases")
    return dispatch, terminals, packets, receipt, bindings


def answer_status(raw, case):
    required = {"case_id", "candidate_id", "ambiguous", "support", "reason", "state_definition_issue"}
    if type(raw) is not dict or set(raw) != required or raw.get("case_id") != case["case_id"]:
        return "invalid", "case_schema", None
    if type(raw["ambiguous"]) is not bool or any(type(raw[k]) is not str for k in ("support", "reason", "state_definition_issue")):
        return "invalid", "field_types", None
    if not raw["reason"].strip() or raw["candidate_id"] is not None and (
            type(raw["candidate_id"]) is not str or raw["candidate_id"] not in {c["id"] for c in case["candidates"]}):
        return "invalid", "reason_or_candidate", None
    # A quote witness is literal presence only, never a judgment about its meaning.
    context = json.loads(case["context"])
    strings = [v for exchange in context["exchanges_oldest_first"] for v in (exchange["SYSTEM"], exchange["USER"])]
    previous = case["question"].rsplit("\nPrevious committed value:\n", 1)[1].split("\nLexical flag order:", 1)[0]
    quoted = bool(raw["support"]) and any(raw["support"] in text for text in strings+[previous])
    issue = "support_not_exact_quote" if (raw["support"] or raw["candidate_id"] is not None) and not quoted else None
    return ("unresolved" if raw["candidate_id"] is None else "valid"), issue, quoted


def response(entry, terminal, packet, reviews, bindings, absent, check):
    path = local(reviews, entry["response_file"])
    envelope_error, document = None, None
    if not path.exists():
        absent.add(path)
        envelope_error = "missing_response"
    else:
        bindings[path] = item(path, check)
        try:
            document = read(path)
            require(type(document) is dict and set(document) == {"review_id", "packet_sha256", "cases"}
                    and document["review_id"] == entry["review_id"] and document["packet_sha256"] == entry["packet_sha256"]
                    and type(document["cases"]) is list
                    and [c.get("case_id") if isinstance(c, dict) else None for c in document["cases"]]
                    == [c["case_id"] for c in packet["cases"]], "Response identity/case order")
        except (ValueError, KeyError, TypeError) as error:
            envelope_error = str(error)
    answers = []
    for i, case in enumerate(packet["cases"]):
        raw = None if envelope_error else document["cases"][i]
        status, reason, quote = ("missing" if envelope_error == "missing_response" else "invalid", envelope_error, None) if envelope_error else answer_status(raw, case)
        answers.append({"case_id": case["case_id"], "status": status, "issue": reason, "interpretation": raw,
                        "exact_quote_present": quote, "eligible": status == "valid" and terminal["status"] == "completed"})
    return {"review_id": entry["review_id"], "arm": entry["arm"], "terminal": terminal,
            "response_present": path.exists(), "response_sha256": bindings[path]["sha256"] if path in bindings else None,
            "envelope_error": envelope_error, "answers": answers}


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator, "rate": numerator/denominator if denominator else None}


def join(packets, reviews, ledger):
    ids = [c["case_id"] for c in packets["current"]["cases"]]
    require([x["case_id"] for x in ledger] == ids, "Exact evaluator case order")
    cases = []
    for i, row in enumerate(ledger):
        mapping = row["canonical_id_map"]
        require(type(mapping) is dict and len(set(mapping.values())) == len(mapping), "Canonical bijection")
        case = {"case_id": row["case_id"], "selection_stratum": row["selection_stratum"], "arms": {}}
        for arm in ARMS:
            public = packets[arm]["cases"][i]
            require(set(mapping) == {c["id"] for c in public["candidates"]}, "Canonical public join")
            evaluator = row["rows"][arm]
            require(evaluator["arm"] == arm and all(evaluator[k] in mapping.values() for k in ("previous_id", "target_id", "selected_id")),
                    "Evaluator candidate membership")
            answers = {}
            for n in (1, 2):
                review_id = f"{arm}-{n:02}"
                answer = dict(reviews[review_id]["answers"][i])
                canonical = mapping[answer["interpretation"]["candidate_id"]] if answer["eligible"] else None
                answer.update(canonical_candidate_id=canonical,
                              target_agreement=(canonical == evaluator["target_id"]) if answer["eligible"] else None,
                              model_agreement=(canonical == evaluator["selected_id"]) if answer["eligible"] else None)
                answers[review_id] = answer
            values = list(answers.values())
            comparable = all(a["eligible"] for a in values)
            case["arms"][arm] = {"evaluator": evaluator, "reviews": answers, "both_valid_nonnull": comparable,
                                  "agreement": values[0]["canonical_candidate_id"] == values[1]["canonical_candidate_id"] if comparable else None}
        cases.append(case)
    aggregates = {}
    for arm in ARMS:
        rows = [c["arms"][arm] for c in cases]
        comparable = [c for c in rows if c["both_valid_nonnull"]]
        arm_results = {"cases": len(rows), "paired_valid_denominator": len(comparable),
                       "agree": sum(c["agreement"] for c in comparable), "disagree": sum(not c["agreement"] for c in comparable),
                       "not_comparable": len(rows)-len(comparable), "reviews": {}}
        for n in (1, 2):
            rid = f"{arm}-{n:02}"
            answers = [c["reviews"][rid] for c in rows]
            eligible = [a for a in answers if a["eligible"]]
            arm_results["reviews"][rid] = {
                "terminal_status": reviews[rid]["terminal"]["status"], "answer_status_counts": dict(Counter(a["status"] for a in answers)),
                "valid_nonnull_denominator": len(eligible),
                "target_agreement": fraction(sum(a["target_agreement"] for a in eligible), len(eligible)),
                "model_agreement": fraction(sum(a["model_agreement"] for a in eligible), len(eligible)),
                "ambiguous_valid_answers": sum(a["interpretation"]["ambiguous"] for a in eligible),
                "support_issue_flags": sum(a["issue"] == "support_not_exact_quote" for a in answers),
                "ambiguity_flags_in_all_interpretations": sum(a["interpretation"].get("ambiguous") is True for a in answers if isinstance(a["interpretation"], dict)),
                "state_definition_issue_flags": sum(bool(a["interpretation"].get("state_definition_issue")) for a in answers if isinstance(a["interpretation"], dict))}
        aggregates[arm] = arm_results
    return {"cases": cases, "arms": aggregates, "reviews": {rid: {k: v for k, v in row.items() if k != "answers"} for rid, row in reviews.items()}}


def peak_rss():
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n*1024)


def execute(args):
    out = Path(args.out).resolve()
    require(all(not out.is_relative_to(Path(p).resolve()) and not Path(p).resolve().is_relative_to(out)
                for p in (args.diagnostic, args.reviews)), "Separate output directory")
    out.mkdir(parents=True, exist_ok=False)
    started, prior, sources = time.monotonic(), None, {}
    request = {k: str(v) for k, v in vars(args).items()}
    def check():
        require(time.monotonic()-started <= LIMITS["wall_seconds"], "Join wall cap")
        require(peak_rss() <= LIMITS["rss_bytes"], "Join RSS cap")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Join output cap")
    def timeout(*_):
        raise TimeoutError("Join wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing timer")
        prior = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        sources = {name: item(ROOT/name, check)["sha256"] for name in SOURCES}
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
                                  "limits": LIMITS, "scope": SCOPE, "cpu_threads": 1})
        dispatch, terminals, packets, diagnostic, bindings = authenticate(args, check)
        absent = set()
        reviews = {e["review_id"]: response(e, terminals[e["review_id"]], packets[e["arm"]], Path(args.reviews).resolve(), bindings, absent, check)
                   for e in dispatch["reviews"]}
        ledger = [json.loads(line) for line in (Path(args.diagnostic)/"evaluator-selection.jsonl").read_text().splitlines()]
        summary = join(packets, reviews, ledger)
        summary.update(status="completed", version=VERSION, scope=SCOPE, selected_cases=diagnostic["selected_cases"],
                       diagnostic_receipt_sha256=args.diagnostic_receipt_sha256,
                       dispatch_sha256=bindings[Path(args.reviews).resolve()/"dispatch.json"]["sha256"],
                       model_calls=0, replacement_reviews=0)
        write(out/"summary.json", summary)
        for path, descriptor in bindings.items():
            require(item(path, check) == descriptor, "End input identity")
        require(all(not p.exists() for p in absent), "A missing response appeared during join")
        require(sources == {n: item(ROOT/n, check)["sha256"] for n in SOURCES}, "End source identity")
        check()
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "request": request, "scope": SCOPE,
              "source_sha256": sources, "diagnostic_receipt_sha256": args.diagnostic_receipt_sha256,
              "dispatch_sha256": summary["dispatch_sha256"], "input_members": {str(p): d for p, d in bindings.items()},
              "missing_responses": [str(p) for p in sorted(absent)], "files": {n: item(out/n, check) for n in ("started.json", "summary.json")},
              "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": peak_rss(), "limits": LIMITS,
              "cpu_threads": 1, "model_calls": 0, "tokenizer_calls": 0, "checkpoint_deserializations": 0,
              "replacement_reviews": 0, "no_retry": True})
        check()
        return summary
    except BaseException as error:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"late-receipt.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "source_sha256": sources, "error": repr(error), "wall_seconds": time.monotonic()-started,
                  "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "diagnostic-receipt-sha256", "reviews", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
