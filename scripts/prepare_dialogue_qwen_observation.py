"""Freeze complete, target-separated Qwen prompts using local tokenization only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from openjev.decisions import LABELS, MAX_TOKENS, MODEL_ID, MODEL_REVISION, messages_for
from openjev.research.dialogue_qwen_observation import (
    ARMS,
    FEATURES,
    candidate_type,
    make_request,
    pilot_ids,
    public_context,
    question_from_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
ROW_PATH = "output/dialogue-objective-v1/training-01/evaluation-rows.jsonl"
ROW_SHA = "56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0"
DATA = "runs/sgd-state-v1/data"
LEX = "runs/dialogue-copy-v1/lexical-01"
DATA_SHA = "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12"
LEX_SHA = "2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54"
REFERENCE = "output/dialogue-objective-v1/report-01/summary.json"
REFERENCE_SHA = "d744753d9edb545b9060867500e3c390d03da69937ffd6d8b3fd1e0c12c4cf6b"
SOURCES = (
    "src/openjev/decisions.py", "src/openjev/research/shared_prefix.py",
    "src/openjev/research/dialogue_qwen_observation.py",
    "scripts/prepare_dialogue_qwen_observation.py", "scripts/run_dialogue_qwen_observation.py",
    "tests/test_dialogue_qwen_observation.py", "tests/test_run_dialogue_qwen_observation.py",
    "research/dialogue-qwen-observation-protocol.md",
)
PREP_LIMITS = {"wall_seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 512 * 1024**2}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "started.json", {"version": "dialogue-qwen-observation-preparation-v1",
                                "limits": PREP_LIMITS, "model_calls": 0})
    def alarm(*_):
        raise TimeoutError("Preparation wall cap")
    previous_handler = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, PREP_LIMITS["wall_seconds"])
    def check():
        require(time.monotonic() - started <= PREP_LIMITS["wall_seconds"], "Preparation wall cap")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= PREP_LIMITS["rss_bytes"],
                "Preparation RSS cap (macOS bytes)")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= PREP_LIMITS["output_bytes"],
                "Preparation output cap")
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        require(platform.system() == "Darwin", "RSS contract requires macOS")
        sources = {p: sha(ROOT / p) for p in SOURCES}
        inputs = {}
        def authenticated(name, expected):
            p = ROOT / name
            require(sha(p) == expected, "Input identity: " + name)
            inputs[name] = {"sha256": expected, "bytes": p.stat().st_size}
            check()
            return p
        data_receipt = json.loads(authenticated(DATA + "/completed.json", DATA_SHA).read_text())
        lex_receipt = json.loads(authenticated(LEX + "/completed.json", LEX_SHA).read_text())
        row_path = authenticated(ROW_PATH, ROW_SHA)
        authenticated(REFERENCE, REFERENCE_SHA)  # Opaque reference: not read for prompt construction.
        def payload(parent, receipt, name):
            expected = receipt["files"][name]
            if isinstance(expected, dict):
                expected = expected["sha256"]
            return authenticated(parent + "/" + name, expected)
        catalog = json.loads(payload(DATA, data_receipt, "catalog.json").read_text())["train"]
        queries = {q["query_id"]: q for q in catalog}
        rows = [r for line in row_path.read_text().splitlines() if (r := json.loads(line))["heldout_service"]]
        require(len(rows) == 7819 and len({r["row_index"] for r in rows}) == 7819, "Exact primary cohort")
        require(all(r["split"] == "train" and r["admission"] == "admitted" for r in rows), "TRAIN-only rows")
        needed = {r["dialogue_id"] for r in rows}
        dialogues = {}
        with payload(DATA, data_receipt, "train-dialogues.jsonl").open() as stream:
            for line in stream:
                d = json.loads(line)
                if d["dialogue_id"] in needed:
                    require(d["dialogue_id"] not in dialogues, "Duplicate public dialogue")
                    dialogues[d["dialogue_id"]] = d
        require(set(dialogues) == needed, "Complete public dialogue coverage")
        lexical = np.load(payload(LEX, lex_receipt, "lexical.npy"), mmap_mode="r", allow_pickle=False)
        lex_index = json.loads(payload(LEX, lex_receipt, "index.json").read_text())
        require(tuple(lex_index["features"]) == FEATURES, "Inherited lexical feature order")
        cohort = {x["id"]: x for x in lex_index["cohorts"]["train"]}

        model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
        # Reuse the existing sealed model manifest; compare every local file.
        old_protocol_path = authenticated("output/shared-prefix-v1/protocol.json",
                              "199a48fdcabb2240f09769bc14fa37b64f4783e6c2b03002252500c51643fc57")
        old = json.loads(old_protocol_path.read_text())
        require(old["model"] == MODEL_ID and old["revision"] == MODEL_REVISION, "Model identity")
        model_files = old["local_model_files_sha256"]
        for name, digest in model_files.items():
            require(sha(model_path / name) == digest, "Local model content: " + name)
            check()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        ids = [tokenizer.encode(x, add_special_tokens=False) for x in LABELS]
        require(all(len(x) == 1 for x in ids) and len({x[0] for x in ids}) == len(LABELS), "Unique labels")
        label_ids = [x[0] for x in ids]
        grouped = defaultdict(list)
        labels = []
        for row in rows:
            # Current truth is written only to the evaluator file, never passed to the builder.
            labels.append({k: row[k] for k in ("row_index", "dialogue_id", "time", "query_id", "service",
                           "slot", "current_candidate_id", "previous_candidate_id", "derived_bin",
                           "current_value_group", "boolean_slot", "candidate_count")})
            grouped[(row["dialogue_id"], row["time"])].append(row)
        records = []
        for gi, ((dialogue_id, ordinal), group) in enumerate(sorted(grouped.items())):
            group = sorted(group, key=lambda r: r["row_index"])
            contexts = {a: public_context(dialogues[dialogue_id], ordinal, a) for a in ARMS}
            questions, maps = [], []
            for row in group:
                query = queries[row["query_id"]]
                require(len(query["candidates"]) == row["candidate_count"], "Candidate coverage")
                type_order = ("NOT_MENTIONED", "DONTCARE", "TRUE", "FALSE", "OTHER")
                require([type_order.index(candidate_type(c["id"])) for c in query["candidates"]]
                        == row["candidate_types"], "Inherited candidate types")
                idx = cohort[dialogue_id]
                require(row["query_index"] in idx["query_ids"], "Inherited query index")
                local_q = idx["query_ids"].index(row["query_index"])
                start = idx["offset"] + (ordinal * idx["shape"][1] + local_q) * idx["shape"][2] * 10
                require(start == row["cache"]["lexical_start"], "Inherited lexical offset")
                flags = lexical[start:start + len(query["candidates"]) * 10].reshape(-1, 10).tolist()
                q, mapping = question_from_metadata(row, query, flags)
                questions.append(q); maps.append(mapping)
            for arm in ARMS if gi % 2 == 0 else reversed(ARMS):
                for chunk_start in range(0, len(group), 4):
                    chunk = slice(chunk_start, chunk_start + 4)
                    request = make_request(contexts[arm], questions[chunk])
                    tokens, ordered_ids = [], []
                    for q, mapping in zip(request.questions, maps[chunk], strict=True):
                        messages, ordered = messages_for(request.context, q)
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        seq = tokenizer.encode(prompt, add_special_tokens=False)
                        require(0 < len(seq) <= MAX_TOKENS, "Prompt overflow; no truncation")
                        tokens.append(seq); ordered_ids.append([mapping[c.id] for c in ordered])
                    records.append({"request_id": f"{arm}:{dialogue_id}:{ordinal}:{chunk_start // 4}",
                                    "arm": arm, "dialogue_id": dialogue_id, "time": ordinal,
                                    "row_indices": [r["row_index"] for r in group[chunk]],
                                    "request": request.model_dump(), "tokens": tokens,
                                    "ordered_candidate_ids": ordered_ids, "canonical_id_maps": maps[chunk],
                                    "label_ids": label_ids})
            if gi % 100 == 0:
                check()
        require(sum(len(r["row_indices"]) for r in records) == 15638, "Complete two-arm decisions")
        for filename, values in (("requests.jsonl", records), ("labels.jsonl", labels)):
            with (out / filename).open("x") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            check()
        plan = {
            "version": "dialogue-qwen-observation-v1", "method": "batch", "source_sha256": sources,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(model_path),
                      "files_sha256": model_files},
            "runtime": {"python": platform.python_version(), **{x: importlib.metadata.version(x)
                        for x in ("numpy", "mlx", "mlx-lm", "transformers", "tokenizers")}},
            "inputs": inputs,
            "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                      for name in ("requests.jsonl", "labels.jsonl")},
            "limits": {phase: {"wall_seconds": seconds, "rss_bytes": 12 * 1024**3,
                               "output_bytes": 512 * 1024**2} for phase, seconds in (("pilot", 300), ("run", 7200))},
            "pilot_request_ids": pilot_ids(records), "row_count": len(rows), "decisions": 15638,
            "input_token_slots": {a: sum(len(r["tokens"]) * max(map(len, r["tokens"]))
                                         for r in records if r["arm"] == a) for a in ARMS},
            "request_counts": {a: sum(r["arm"] == a for r in records) for a in ARMS},
            "prompt_tokens": {a: {"min": min(len(t) for r in records if r["arm"] == a for t in r["tokens"]),
                                    "max": max(len(t) for r in records if r["arm"] == a for t in r["tokens"]),
                                    "sum": sum(len(t) for r in records if r["arm"] == a for t in r["tokens"])}
                              for a in ARMS},
            "strata": dict(Counter(r["derived_bin"] for r in rows)),
            "scope": "Exposed official TRAIN; gold previous supplied; no training or autonomous memory claim.",
        }
        write(out / "plan.json", plan)
        check()
        require({p: sha(ROOT / p) for p in SOURCES} == sources, "Sources changed during preparation")
        for name, entry in inputs.items():
            require(sha(ROOT / name) == entry["sha256"], "Input changed during preparation: " + name)
            check()
        write(out / "completed.json", {"status": "completed", "model_calls": 0, "encoder_calls": 0,
              "tokenizer_only": True, "official_dev_inference": False,
              "official_dev_dialogues_accessed": False, "test_contents_accessed": False,
              "wall_seconds": time.monotonic() - started,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "plan_sha256": sha(out / "plan.json"),
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in out.iterdir() if p.is_file()}})
        check()
        print(json.dumps({"status": "completed", "plan_sha256": sha(out / "plan.json"),
                          "requests": len(records), "prompt_tokens": plan["prompt_tokens"]}), flush=True)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "late-completion.json")
            write(out / "failed.json", {"status": "failed", "exception": type(exc).__name__,
                  "message": str(exc), "wall_seconds": time.monotonic() - started, "model_calls": 0})
        except BaseException as receipt_error:  # noqa: BLE001 - preserve the original terminal exception
            if callable(getattr(exc, "add_note", None)):
                exc.add_note("Could not preserve terminal receipt: " + repr(receipt_error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
