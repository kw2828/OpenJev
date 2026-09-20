"""One frozen, model-free literal-history diagnostic on the exposed TRAIN cohort."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from openjev.research.dialogue_history_support import aggregate, audit_query

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/audit_dialogue_history_support.py",
    "src/openjev/research/dialogue_history_support.py",
    "tests/test_audit_dialogue_history_support.py",
    "tests/test_dialogue_history_support.py",
    "research/dialogue-history-support-protocol.md",
)
INPUTS = {
    "runs/sgd-state-v1/features-02/completed.json":
        "e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308",
    "runs/sgd-state-v1/features-02/packet.json":
        "14dd89a633bfa406394945d279c44ce6796d3b8dc412f772e2abda794ec943cb",
    "runs/sgd-state-v1/data/completed.json":
        "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
    "runs/sgd-state-v1/data/train-dialogues.jsonl":
        "04bf8797b35495d02c9a809a9b83259aefc24ff04a560f13aec989ba117b15aa",
}
CONFIG = {"cohort": "train", "dialogues": 2017, "rows": 51741,
          "wall_seconds": 180, "rss_bytes": 2 * 1024**3,
          "output_bytes": 128 * 1024**2, "model_calls": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def authenticate(root, expected):
    for name, digest in expected.items():
        require(sha(root / name) == digest, "Hash mismatch: " + name)


def selected_labels(dialogue, public, queries):
    """Join full packet times to public USER indices; gold stays evaluator-side."""
    require(dialogue["id"] == public["dialogue_id"], "Dialogue identity")
    users = public["user_turns"]
    require(len(dialogue["turns"]) == len(users), "Complete stream length")
    text = [public["turns"][u["turn_index"]]["utterance"] for u in users]
    require(text == dialogue["user_text"], "Public/encoded USER alignment")
    grouped = defaultdict(list)
    for row in dialogue["queries"]:
        qi, ti = row["query"], row["time"]
        require(type(qi) is int and 0 <= qi < len(queries), "Query index")
        require(type(ti) is int and 0 <= ti < len(users), "Time index")
        query = queries[qi]
        li = row["label"]
        require(type(li) is int and 0 <= li < len(query["candidate_ids"]), "Label index")
        grouped[qi].append({"dialogue_id": dialogue["id"],
                            "query_id": query["id"],
                            "turn_index": users[ti]["turn_index"],
                            "label_index": li, "label_id": query["candidate_ids"][li],
                            "bin": row["bin"]})
    result = []
    for qi in sorted(grouped):
        entry = queries[qi]
        require(json.loads(entry["id"]) == [entry["service"], entry["slot"]], "Query identity")
        query = {"query_id": entry["id"], "service": entry["service"], "slot": entry["slot"],
                 "candidates": [{"id": cid, "value": value} for cid, value in
                                zip(entry["candidate_ids"], entry["candidate_values"], strict=True)]}
        result.append((query, grouped[qi]))
    return result


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    authenticate(ROOT, INPUTS)
    plan = {"study": "dialogue-history-support-v1", "config": CONFIG,
            "input_sha256": INPUTS, "source_sha256": {s: sha(ROOT / s) for s in SOURCES},
            "python": platform.python_version(), "platform": platform.platform()}
    write(out / "plan.json", plan)
    print(json.dumps({"plan_sha256": sha(out / "plan.json")}))


def run(plan_path, plan_sha256, out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    completed_dialogues = 0

    def limits():
        require(time.perf_counter() - started <= CONFIG["wall_seconds"], "Wall cap")
        require(rss() <= CONFIG["rss_bytes"], "RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <=
                CONFIG["output_bytes"], "Output cap")

    try:
        write(out / "started.json", {"plan_sha256": plan_sha256,
                                    "scope": "Exposed TRAIN literal evidence; no model execution"})
        require(sha(plan_path) == plan_sha256, "Plan hash")
        plan = json.loads(plan_path.read_text())
        require(plan["study"] == "dialogue-history-support-v1" and plan["config"] == CONFIG,
                "Frozen study/configuration")
        require(plan["input_sha256"] == INPUTS and set(plan["source_sha256"]) == set(SOURCES),
                "Frozen input/source closure")
        require(plan["python"] == platform.python_version() and
                plan["platform"] == platform.platform(), "Frozen runtime")
        authenticate(ROOT, plan["source_sha256"])
        authenticate(ROOT, INPUTS)
        limits()
        packet = json.loads((ROOT / "runs/sgd-state-v1/features-02/packet.json").read_text())
        cohort = packet["cohorts"]["train"]
        requested = {d["id"] for d in cohort}
        require(len(requested) == len(cohort) == CONFIG["dialogues"], "Fixed TRAIN population")
        require(sum(len(d["queries"]) for d in cohort) == CONFIG["rows"], "Fixed TRAIN rows")
        public = {}
        seen = set()
        with (ROOT / "runs/sgd-state-v1/data/train-dialogues.jsonl").open() as f:
            for line in f:
                item = json.loads(line)
                identifier = item["dialogue_id"]
                require(identifier not in seen, "Duplicate public dialogue")
                seen.add(identifier)
                if identifier in requested:
                    public[identifier] = item
        require(set(public) == requested, "Complete public cohort")
        limits()
        rows = []
        supplied_queries = user_steps = query_steps = 0
        consecutive_system_pairs = consecutive_user_pairs = 0
        for dialogue in cohort:
            visible = public[dialogue["id"]]
            for left, right in zip(visible["turns"], visible["turns"][1:]):
                consecutive_system_pairs += left["speaker"] == right["speaker"] == "SYSTEM"
                consecutive_user_pairs += left["speaker"] == right["speaker"] == "USER"
            grouped = selected_labels(dialogue, visible, packet["queries"])
            supplied_queries += len(grouped)
            user_steps += len(visible["user_turns"])
            query_steps += len(visible["user_turns"]) * len(grouped)
            for query, labels in grouped:
                rows.extend(audit_query(visible, query, labels))
            completed_dialogues += 1
            limits()
        require(len(rows) == CONFIG["rows"], "Complete endpoint coverage")
        summary = aggregate(rows)
        summary["cohort"] = {"split": "train", "dialogues": completed_dialogues,
                             "rows": len(rows), "supplied_queries": supplied_queries,
                             "available_public_user_steps": user_steps,
                             "available_public_query_steps": query_steps,
                             "consecutive_system_pairs": consecutive_system_pairs,
                             "consecutive_user_pairs": consecutive_user_pairs}
        summary["scope"] = ("Literal proxies on exposed TRAIN, not semantic or causal memory evidence. "
                            "No prediction, accuracy estimate, fit or automatic training admission.")
        with (out / "endpoint-metadata.jsonl.gz").open("xb") as raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
        write(out / "summary.json", summary)
        authenticate(ROOT, plan["source_sha256"])
        authenticate(ROOT, INPUTS)
        limits()
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in out.iterdir()}
        write(out / "completed.json", {"status": "completed", "plan_sha256": plan_sha256,
              "files": files, "input_sha256": INPUTS, "source_sha256": plan["source_sha256"],
              "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": rss(),
              "dialogues": completed_dialogues, "rows": len(rows), "model_calls": 0,
              "feature_arrays_opened": False, "dev_dialogue_file_opened": False,
              "official_test_opened": False, "scope": summary["scope"]})
        limits()
        print(json.dumps({"status": "completed", "completed_sha256": sha(out / "completed.json")}))
    except BaseException as error:
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "incomplete-completion.json")
            write(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                  "error": str(error), "completed_dialogues": completed_dialogues,
                  "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": rss(), "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original execution failure
            error.add_note("Failure receipt could not be closed: " + str(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("freeze")
    a.add_argument("--out", required=True, type=Path)
    a = sub.add_parser("run")
    a.add_argument("--out", required=True, type=Path)
    a.add_argument("--plan", required=True, type=Path)
    a.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    else:
        run(args.plan, args.plan_sha256, args.out)
