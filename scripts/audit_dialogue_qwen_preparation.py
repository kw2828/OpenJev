"""Independent saved-packet audit; no producer, tokenizer or model imports.

Reconstructs public actor text and mappings, but inherits tokenization and model
weight authentication from the externally pinned preparation. No quality scores.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-qwen-preparation-audit-v1"
ARMS = ("current", "history4")
LIMITS = {"wall_seconds": 60, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
FEATURES = ("user_match", "system_match", "unique_longest_user", "unique_longest_system",
            "literal_current", "literal_previous", "is_none", "is_dontcare",
            "affirmative_cue_for_true", "negative_cue_for_false")
TASK = (
    "What is the user's currently committed value for this slot after the final USER turn? "
    "Use the supplied previous value as the state immediately before that turn. "
    "Update it only when the dialogue supports a change; a SYSTEM proposal alone is not "
    "a user commitment. NOT_MENTIONED means no constraint has been stated; DONTCARE "
    "means the user explicitly has no preference. Literal ontology values, including "
    "a value named None, remain distinct from these reserved states. "
    "Lexical flags are noisy public string-match observations, not answers. "
    "The literal register carries the last unique longest USER mention across the "
    "public prefix; it does not understand negation or relevance. "
)
SOURCES = {
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "src/openjev/research/dialogue_qwen_observation.py",
    "scripts/prepare_dialogue_qwen_observation.py", "scripts/run_dialogue_qwen_observation.py",
    "tests/test_dialogue_qwen_observation.py", "tests/test_run_dialogue_qwen_observation.py",
    "tests/test_prepare_dialogue_qwen_observation.py", "research/dialogue-qwen-observation-protocol.md",
}
PINS = {
    "runs/sgd-state-v1/data/completed.json": "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
    "runs/dialogue-copy-v1/lexical-01/completed.json": "2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54",
    "output/dialogue-objective-v1/training-01/evaluation-rows.jsonl": "56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0",
    "output/dialogue-objective-v1/report-01/summary.json": "d744753d9edb545b9060867500e3c390d03da69937ffd6d8b3fd1e0c12c4cf6b",
    "output/shared-prefix-v1/protocol.json": "199a48fdcabb2240f09769bc14fa37b64f4783e6c2b03002252500c51643fc57",
}
CATALOG = "runs/dialogue-conditional-v1/preparation-01/catalog.json"
CATALOG_SHA = "835dbf5bb45d84487c6bb9cce4443f98dae295aa3199be3e0107be94fb57b723"
LABEL_FIELDS = ("row_index", "dialogue_id", "time", "query_id", "service", "slot",
                "current_candidate_id", "previous_candidate_id", "derived_bin",
                "current_value_group", "boolean_slot", "candidate_count")


def require(value, message):
    if not value:
        raise ValueError(message)


def decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            check()
    return digest.hexdigest()


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def audit(args, check):
    prepared = Path(args.prepared)
    require(sha(prepared / "completed.json", check) == args.completed_sha256, "External completion pin")
    require(sha(prepared / "plan.json", check) == args.plan_sha256, "External plan pin")
    done, plan = read(prepared / "completed.json"), read(prepared / "plan.json")
    require(done["status"] == "completed" and done["plan_sha256"] == args.plan_sha256
            and done["model_calls"] == done["encoder_calls"] == 0 and done["tokenizer_only"] is True
            and done["official_dev_inference"] is False and done["official_dev_dialogues_accessed"] is False
            and done["test_contents_accessed"] is False, "Preparation completed scope")
    members = {"started.json", "plan.json", "requests.jsonl", "labels.jsonl"}
    require(set(done["files"]) == members and {str(p.relative_to(prepared)) for p in prepared.rglob("*")
            if p.is_file()} == members | {"completed.json"}, "Exact completed preparation closure")
    authenticated = {}
    def bind(path, expected, size=None):
        path = Path(path)
        require(path.is_file() and (size is None or path.stat().st_size == size)
                and sha(path, check) == expected, "Artifact identity: " + str(path))
        authenticated[str(path.resolve())] = expected
        return path
    for name, entry in done["files"].items():
        require(set(entry) == {"sha256", "bytes"}, "File descriptor fields")
        bind(prepared / name, entry["sha256"], entry["bytes"])
    started = read(prepared / "started.json")
    require(started == {"version": "dialogue-qwen-observation-preparation-v1",
            "limits": {"wall_seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 512 * 1024**2},
            "model_calls": 0}, "Preparation initial contract")
    require(math.isfinite(done["wall_seconds"]) and 0 < done["wall_seconds"] <= 180
            and 0 < done["peak_rss_bytes"] <= 2 * 1024**3
            and sum(p.stat().st_size for p in prepared.iterdir() if p.is_file()) <= 512 * 1024**2,
            "Preparation recorded caps")
    require(plan["version"] == "dialogue-qwen-observation-v1" and plan["method"] == "batch"
            and plan["row_count"] == 7819 and plan["decisions"] == 15638, "Plan scope")
    require(set(plan["source_sha256"]) == SOURCES, "Nine source paths")
    for name, digest in plan["source_sha256"].items():
        bind(ROOT / name, digest)
    require(set(plan["files"]) == {"requests.jsonl", "labels.jsonl"}
            and all(done["files"][k] == v for k, v in plan["files"].items()), "Payload receipt binding")
    for name, digest in PINS.items():
        require(plan["inputs"][name]["sha256"] == digest, "Inherited input pin")
    for name, entry in plan["inputs"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Input path")
        bind(ROOT / name, entry["sha256"], entry["bytes"])
    def inherited(parent, name):
        receipt = read(ROOT / parent / "completed.json")
        entry = receipt["files"][name]
        digest = entry["sha256"] if isinstance(entry, dict) else entry
        require(plan["inputs"][parent + "/" + name]["sha256"] == digest, "Parent payload binding")
        return ROOT / parent / name
    catalog_path = inherited("runs/sgd-state-v1/data", "catalog.json")
    public_path = inherited("runs/sgd-state-v1/data", "train-dialogues.jsonl")
    lexical_path = inherited("runs/dialogue-copy-v1/lexical-01", "lexical.npy")
    index_path = inherited("runs/dialogue-copy-v1/lexical-01", "index.json")
    require(set(plan["inputs"]) == set(PINS) | {
        str(p.relative_to(ROOT)) for p in (catalog_path, public_path, lexical_path, index_path)},
        "Exact inherited input set")
    old_model = read(ROOT / "output/shared-prefix-v1/protocol.json")
    require(plan["model"]["id"] == old_model["model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
            and plan["model"]["revision"] == old_model["revision"] == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
            and plan["model"]["files_sha256"] == old_model["local_model_files_sha256"], "Inherited model manifest")
    rows = [r for line in (ROOT / "output/dialogue-objective-v1/training-01/evaluation-rows.jsonl").read_text().splitlines()
            if (r := decode(line))["heldout_service"]]
    require(len(rows) == len({r["row_index"] for r in rows}) == 7819
            and all(r["split"] == "train" and r["admission"] == "admitted" for r in rows), "Canonical primary rows")
    by_id = {r["row_index"]: r for r in rows}
    require(dict(Counter(r["derived_bin"] for r in rows)) == plan["strata"], "Stratum counts")
    with (prepared / "labels.jsonl").open() as stream:
        labels = [decode(line) for line in stream]
    require(labels == [{k: r[k] for k in LABEL_FIELDS} for r in rows], "Exact evaluator rows/order")
    queries = {q["query_id"]: q for q in read(catalog_path)["train"]}
    query_indices = {q["query_index"]: q for q in read(bind(ROOT / CATALOG, CATALOG_SHA))["queries"]}
    needed = {r["dialogue_id"] for r in rows}
    dialogues = {}
    with public_path.open() as stream:
        for line in stream:
            d = decode(line)
            if d["dialogue_id"] in needed:
                require(d["dialogue_id"] not in dialogues, "Duplicate public dialogue")
                dialogues[d["dialogue_id"]] = d
    require(set(dialogues) == needed, "Public dialogue coverage")
    timeline = {}
    for identifier, d in dialogues.items():
        pairs = []
        for i, turn in enumerate(d["turns"]):
            require(set(turn) == {"speaker", "utterance"} and turn["speaker"] in ("USER", "SYSTEM")
                    and type(turn["utterance"]) is str, "Public text whitelist")
            if turn["speaker"] == "USER":
                system = d["turns"][i-1]["utterance"] if i and d["turns"][i-1]["speaker"] == "SYSTEM" else ""
                pairs.append({"SYSTEM": system, "USER": turn["utterance"]})
        timeline[identifier] = pairs
    lex_index = read(index_path)
    require(lex_index["features"] == list(FEATURES), "Lexical flag order")
    layouts = {d["id"]: d for d in lex_index["cohorts"]["train"]}
    lexical = np.load(lexical_path, mmap_mode="r", allow_pickle=False)
    require(lexical.ndim == 1 and lexical.dtype == np.float32, "Lexical float32 layout")
    expected_questions, mappings, groups = {}, {}, defaultdict(list)
    type_names = ("NOT_MENTIONED", "DONTCARE", "TRUE", "FALSE", "OTHER")
    for r in rows:
        q, indexed = queries[r["query_id"]], query_indices[r["query_index"]]
        candidates = q["candidates"]
        ids = [c["id"] for c in candidates]
        require(indexed["split"] == "train" and indexed["query_id"] == r["query_id"]
                and indexed["candidate_ids"] == ids and q["service"] == r["service"]
                and q["slot"] == r["slot"] and len(ids) == r["candidate_count"]
                and 2 <= len(ids) <= 12 and len(set(ids)) == len(ids), "Query/candidate lineage")
        require(ids[r["previous_current_index"]] == r["previous_candidate_id"]
                and ids[r["current_label_index"]] == r["current_candidate_id"], "Canonical state identities")
        require(indexed["boolean_slot"] == r["boolean_slot"], "Public Boolean schema")
        types = [0 if c["id"] == "reserved:NOT_MENTIONED" else 1 if c["id"] == "reserved:DONTCARE"
                 else (2 if c["value"].strip().casefold() == "true" else 3) if indexed["boolean_slot"] else 4
                 for c in candidates]
        require(types == r["candidate_types"], "Public candidate types")
        layout, t = layouts[r["dialogue_id"]], r["time"]
        require(type(t) is int and 0 <= t < len(timeline[r["dialogue_id"]]) == layout["shape"][0]
                and layout["shape"][3] == 10 and layout["shape"][1] == len(layout["query_ids"]), "Public time/layout")
        local_q = layout["query_ids"].index(r["query_index"])
        start = layout["offset"] + 10 * layout["shape"][2] * (t * layout["shape"][1] + local_q)
        require(start == r["cache"]["lexical_start"] and r["cache"]["lexical_candidates"] == len(ids)
                and r["cache"]["lexical_stride"] == 10 and layout["shape"][2] >= len(ids), "Lexical offset/stride")
        flags = lexical[start:start+10*len(ids)].reshape(len(ids), 10)
        require(np.isfinite(flags).all() and ((flags == 0) | (flags == 1)).all(), "Binary public flags")
        descriptions = [c["text"] + "\nCandidate type: " + type_names[k] + "\nPublic lexical flags: "
                        + ",".join(str(int(v)) for v in f) for c, k, f in zip(candidates, types, flags, strict=True)]
        expected_questions[r["row_index"]] = {"id": "r" + str(r["row_index"]),
            "question": (TASK + "\n" + q["query_text"] + "\nPrevious committed value:\n"
                         + candidates[r["previous_current_index"]]["text"]
                         + "\nLexical flag order: " + ",".join(FEATURES)).strip(),
            "candidates": [{"id": f"c{i:02d}", "description": text.strip()} for i, text in enumerate(descriptions)]}
        mappings[r["row_index"]] = {f"c{i:02d}": value for i, value in enumerate(ids)}
        groups[(r["dialogue_id"], t)].append(r["row_index"])
    schedule = []
    for i, ((identifier, t), group) in enumerate(sorted(groups.items())):
        group = sorted(group)
        for arm in ARMS if i % 2 == 0 else ARMS[::-1]:
            for offset in range(0, len(group), 4):
                schedule.append((f"{arm}:{identifier}:{t}:{offset//4}", arm, identifier, t, group[offset:offset+4]))
    stats, counts, slots, seen, metadata = {a: [] for a in ARMS}, Counter(), Counter(), Counter(), []
    packet_fields = {"request_id", "arm", "dialogue_id", "time", "row_indices", "request", "tokens",
                     "ordered_candidate_ids", "canonical_id_maps", "label_ids"}
    with (prepared / "requests.jsonl").open() as stream:
        for index, line in enumerate(stream):
            require(index < len(schedule), "Unexpected extra request")
            record = decode(line)
            request_id, arm, identifier, t, row_ids = schedule[index]
            require(set(record) == packet_fields and (record["request_id"], record["arm"], record["dialogue_id"],
                    record["time"], record["row_indices"]) == (request_id, arm, identifier, t, row_ids), "Exact request order/rows")
            first = t if arm == "current" else max(0, t-3)
            context = json.dumps({"exchanges_oldest_first": timeline[identifier][first:t+1]}, ensure_ascii=False, sort_keys=True)
            require(record["request"] == {"context": context, "questions": [expected_questions[r] for r in row_ids]},
                    "Public actor text/previous/flags mismatch")
            require(record["canonical_id_maps"] == [mappings[r] for r in row_ids]
                    and record["label_ids"] == list(range(32, 44)), "Canonical maps/single-token label witness")
            ordered = [[mappings[r][c["id"]] for c in sorted(expected_questions[r]["candidates"],
                        key=lambda c: c["description"])] for r in row_ids]
            require(record["ordered_candidate_ids"] == ordered and len(record["tokens"]) == len(row_ids), "Label order/token rows")
            lengths = []
            for r, tokens in zip(row_ids, record["tokens"], strict=True):
                require(type(tokens) is list and 0 < len(tokens) <= 4096
                        and all(type(x) is int and 0 <= x < 151936 for x in tokens), "Token structural bounds")
                lengths.append(len(tokens))
                seen[(arm, r)] += 1
            stats[arm].extend(lengths)
            counts[arm] += 1
            slots[arm] += len(lengths) * max(lengths)
            metadata.append((request_id, arm, identifier, t, max(lengths)))
            if index % 100 == 0:
                check()
    require(len(metadata) == len(schedule) and seen == Counter({(a, r): 1 for a in ARMS for r in by_id}), "All 15638 decisions exactly once")
    require(dict(counts) == plan["request_counts"] and dict(slots) == plan["input_token_slots"], "Request/padded token counts")
    token_summary = {a: {"min": min(v), "max": max(v), "sum": sum(v)} for a, v in stats.items()}
    require(token_summary == plan["prompt_tokens"], "Recorded prompt length summaries")
    pilot_groups = set(sorted(groups)[:12])
    for arm in ARMS:
        longest = min((m for m in metadata if m[1] == arm), key=lambda m: (-m[4], m[2], m[3], m[0]))
        pilot_groups.add((longest[2], longest[3]))
    require(plan["pilot_request_ids"] == [m[0] for m in metadata if (m[2], m[3]) in pilot_groups], "Exact pilot union/numeric tie/order")
    for path, digest in authenticated.items():
        require(sha(path, check) == digest, "End identity stability: " + path)
    require(sha(prepared / "completed.json", check) == args.completed_sha256, "End completion stability")
    return {"status": "completed", "agreement": True, "version": VERSION,
            "prepared_completed_sha256": args.completed_sha256, "plan_sha256": args.plan_sha256,
            "canonical_rows": len(rows), "actor_decisions_checked": sum(seen.values()), "requests_checked": len(metadata),
            "dialogues": len(needed), "schema_queries": len({r["query_id"] for r in rows}),
            "services": sorted({r["service"] for r in rows}), "strata": plan["strata"],
            "candidate_counts": dict(sorted(Counter(r["candidate_count"] for r in rows).items())),
            "request_counts": dict(counts), "prompt_tokens": token_summary, "input_token_slots": dict(slots),
            "pilot_groups": len(pilot_groups), "pilot_requests": len(plan["pilot_request_ids"]),
            "source_sha256": plan["source_sha256"], "authenticated_files": authenticated,
            "model_calls": 0, "tokenizer_calls": 0, "encoder_calls": 0, "quality_scored": False,
            "scope": "Independent exact public actor/evaluator reconstruction and structural token/count validation. "
                     "Model-weight content and tokenization semantics inherit the pinned preparation; no weight rehash, "
                     "token decoding or neural replay. Runner must retokenize each actual prompt before inference. "
                     "Equality to whitelisted reconstruction does not prove arbitrary execution ignored labels."}


def execute(args):
    start = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    source_hash = sha(source)
    old_handler = None
    def check():
        require(time.monotonic()-start <= LIMITS["wall_seconds"], "Audit wall cap")
        require(rss() <= LIMITS["rss_bytes"], "Audit RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
    def timeout(*_):
        raise TimeoutError("Audit wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        write(out / "started.json", {"version": VERSION, "request": vars(args), "limits": LIMITS,
                                    "audit_source_sha256": source_hash})
        with (out / "audit-source.py").open("xb") as stream:
            stream.write(source.read_bytes())
        result = audit(args, check)
        require(sha(source) == source_hash, "Audit source changed")
        write(out / "summary.json", result)
        check()
        files = {p.name: {"sha256": sha(p, check), "bytes": p.stat().st_size}
                 for p in sorted(out.iterdir()) if p.is_file()}
        write(out / "receipt.json", {"status": "completed", "agreement": True, "version": VERSION,
              "audit_source_sha256": source_hash, "prepared_completed_sha256": args.completed_sha256,
              "plan_sha256": args.plan_sha256, "wall_seconds": time.monotonic()-start,
              "peak_rss_bytes": rss(), "limits": LIMITS, "files": files, "model_calls": 0, "tokenizer_calls": 0})
        check()
        return {"receipt_sha256": sha(out / "receipt.json", check), "summary_sha256": sha(out / "summary.json", check)}
    except BaseException as error:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "receipt.json").exists():
                (out / "receipt.json").rename(out / "late-receipt.json")
            write(out / "failed.json", {"status": "failed", "version": VERSION, "request": vars(args),
                  "error": repr(error), "wall_seconds": time.monotonic()-start, "audit_source_sha256": source_hash})
        except BaseException as secondary:  # noqa: BLE001 - preserve original audit failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", required=True)
    print(json.dumps(execute(parser.parse_args())))
