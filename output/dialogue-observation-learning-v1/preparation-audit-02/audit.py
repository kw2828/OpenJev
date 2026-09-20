"""Independent saved-preparation audit. No project, tokenizer or model imports."""
import hashlib
import json
import math
import platform
import resource
import signal
import sys
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREP = OUT.parent / "preparation-02"
COMPLETED = "d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83"
PLAN = "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e"
ORIGINAL_LEXICAL = "d70dce8074fb8f5fcf8f8174cbbade325397873cc227204f73d96faa3e613591"
OLD_PLAN = "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905"
NONE, DC = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
STRATA = ("unmentioned_retention", "assigned_retention", "changed")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def integer(value):
    return type(value) is int and value >= 0


def transitions(previous, current):
    if previous == current:
        return "unmentioned_retention" if current == NONE else "assigned_retention"
    if previous == NONE:
        return "first_assignment"
    return "clear" if current == NONE else "revision"


def profile(actor):
    tokens = actor["tokens"]
    need(type(tokens) is list and tokens and all(type(x) is list and x and all(integer(v) for v in x) for x in tokens), "Token list geometry")
    turns, queries, candidates = actor["turn_text_ids"], actor["query_text_ids"], actor["candidate_text_ids"]
    need(turns and queries and len(candidates) == len(queries), "Actor map geometry")
    indices = turns + queries + [v for row in candidates for v in row]
    need(all(integer(v) and v < len(tokens) for v in indices) and set(indices) == set(range(len(tokens))), "Complete supported text map")
    original = actor["original_feature_ids"]
    need(len(original) == len(tokens) == len(set(original)) and all(integer(v) for v in original), "Original feature identity")
    need(len(actor["candidate_ids"]) == len(candidates), "Candidate identity count")
    for positions, ids in zip(candidates, actor["candidate_ids"], strict=True):
        need(len(positions) == len(ids) > 0 and len(set(ids)) == len(ids), "Distinct candidate support")
        need(ids.count(NONE) == ids.count(DC) == 1 and all(type(v) is str for v in ids), "Reserved candidate support")
    t, q, c = len(turns), len(queries), max(map(len, candidates))
    need(actor["lexical_shape"] == [t, q, c, 10], "Actor lexical shape")
    need(len(actor["query_ids"]) == q and actor["query_ids"] == sorted(set(actor["query_ids"])), "Sorted global query IDs")
    user_positions = actor["user_turn_indices"]
    need(len(user_positions) == t and all(integer(v) for v in user_positions) and user_positions == sorted(set(user_positions)), "Public USER chronology positions")
    lengths = []
    for text in tokens:
        remainder = len(text)
        while remainder:
            used = min(254, remainder)
            lengths.append(used + 2)
            remainder -= used
    padded = attention = calls = 0
    for begin in range(0, len(lengths), 32):
        chunk = lengths[begin:begin + 32]
        padded += len(chunk) * max(chunk)
        attention += len(chunk) * max(chunk) ** 2
        calls += 1
    valid = sum(lengths)
    occurrences = sum(map(len, candidates))
    return {"unique_texts": len(tokens), "input_texts": len(tokens), "content_tokens": sum(map(len, tokens)),
            "chunks": len(lengths), "encoder_sequences": len(lengths), "special_token_positions": 2 * len(lengths),
            "valid_token_positions": valid, "padded_token_positions": padded, "padding_token_positions": padded - valid,
            "padded_attention_positions": attention, "encoder_calls": calls, "chunk_tokens": 254, "chunk_batch_size": 32,
            "overlength_texts_chunked": sum(len(x) > 254 for x in tokens), "truncated_tokens": 0,
            "max_chunk_tokens_with_special": max(lengths), "public_user_turns": t, "queries": q,
            "schema_text_occurrences": q, "candidate_text_occurrences": occurrences, "max_candidates": c,
            "real_question_updates": t * q, "real_candidate_updates": t * occurrences,
            "padded_candidate_positions": t * q * c, "lexical_scalars": t * q * c * 10,
            "lexical_bytes": t * q * c * 40, "max_content_tokens_per_text": max(map(len, tokens))}


def execute():
    need(sha(PREP / "completed.json") == COMPLETED, "External completion identity")
    done = read(PREP / "completed.json")
    payloads = {"started.json", "source-pins.json", "actors-train.jsonl", "actors-dev.jsonl",
                "targets-train.jsonl", "targets-dev.jsonl", "lexical-original.npy", "lexical-numbers.npy",
                "workloads.json", "orders.json", "index.json", "effective-batches.json", "plan.json"}
    need(done["status"] == "completed" and done["phase"] == "prepare", "Completed preparation only")
    need(set(done["files"]) == payloads and {p.name for p in PREP.iterdir()} == payloads | {"completed.json"}, "Exact fourteen-file closure")
    for name, entry in done["files"].items():
        need(sha(PREP / name) == entry["sha256"] and (PREP / name).stat().st_size == entry["bytes"], "Payload identity: " + name)
    need(sha(PREP / "plan.json") == PLAN == done["plan_sha256"], "External plan identity")
    plan, started = read(PREP / "plan.json"), read(PREP / "started.json")
    need(plan["preparation_version"] == 2 and plan["study"] == started["study"] == done["study"] == "dialogue-observation-learning-v1", "Study/version identity")
    need(plan["runtime"] == started["runtime"] and plan["config"] == started["config"] and plan["caps"] == started["caps"], "Configuration/runtime identity")
    source = read(PREP / "source-pins.json")
    need(source == plan["source_sha256"] and len(source) == 39, "Complete 39-source map")
    for name, pin in source.items():
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts and sha(ROOT / path) == pin, "Frozen source: " + name)
    expected_data = payloads - {"plan.json"}
    need(set(plan["data_files"]) == expected_data, "Plan data membership")
    for name in expected_data:
        need(plan["data_files"][name] == done["files"][name], "Plan/completion data manifest agreement")
    need(done["files"]["lexical-original.npy"]["sha256"] == ORIGINAL_LEXICAL == plan["input_sha256"]["runs/dialogue-copy-v1/lexical-01/lexical.npy"], "Byte-identical original lexical array")
    lexical = {k: np.load(PREP / ("lexical-" + k + ".npy"), allow_pickle=False, mmap_mode="r") for k in ("original", "numbers")}
    need(all(v.dtype == np.float32 and v.shape == (20007650,) for v in lexical.values()), "Lexical dtype/size")
    for name, value in lexical.items():
        for begin in range(0, value.size, 1000000):
            chunk = value[begin:begin + 1000000]
            need(np.isfinite(chunk).all() and ((chunk == 0) | (chunk == 1)).all(), "Binary finite lexical array " + name)

    reconstructed, actor_ids, reconstructed_index = [], {}, []
    counts, feature_hashes, query_metadata = {}, {}, {}
    offset = 0
    changes = np.zeros(10, dtype=np.int64)
    actor_keys = {"split", "dialogue_id", "query_ids", "user_turn_indices", "tokens", "original_feature_ids",
                  "turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical_offset", "lexical_shape"}
    for split, expected_dialogues, expected_rows in (("train", 2017, 51741), ("dev", 2363, 62329)):
        seen, row_count = set(), 0
        actor_ids[split], counts[split] = [], Counter()
        with (PREP / ("actors-" + split + ".jsonl")).open() as actors, (PREP / ("targets-" + split + ".jsonl")).open() as targets:
            for actor_line, target_line in zip_longest(actors, targets):
                need(actor_line is not None and target_line is not None, "Complete actor/target line join")
                a, target = json.loads(actor_line), json.loads(target_line)
                did = a["dialogue_id"]
                need(set(a) == actor_keys and a["split"] == target["split"] == split and did == target["dialogue_id"] and did not in seen, "Split-qualified actor/target identity and whitelist")
                seen.add(did)
                actor_ids[split].append(did)
                w = profile(a)
                need(a["lexical_offset"] == offset, "Contiguous original lexical offset")
                views = {k: v[offset:offset + w["lexical_scalars"]].reshape(a["lexical_shape"]) for k, v in lexical.items()}
                need(np.array_equal(views["original"][..., 6:], views["numbers"][..., 6:]), "Unchanged reserved/Boolean features")
                changes += np.count_nonzero(views["original"] != views["numbers"], axis=(0, 1, 2))
                for j, ids in enumerate(a["candidate_ids"]):
                    for mode, view in views.items():
                        supported = view[:, j, :len(ids)]
                        need(not view[:, j, len(ids):].any(), "Zero padded lexical support")
                        need((supported[:, :, 4].sum(-1) == 1).all() and (supported[:, :, 5].sum(-1) == 1).all(), "One public literal current/previous state")
                        need((supported[:, ids.index(NONE), 6] == 1).all() and (supported[:, :, 6].sum(-1) == 1).all(), "NONE public flag")
                        need((supported[:, ids.index(DC), 7] == 1).all() and (supported[:, :, 7].sum(-1) == 1).all(), "DONTCARE public flag")
                        need(target["literal_registers"][mode][j] == supported[:, :, 4].argmax(-1).tolist(), "Saved literal register corresponds to actor, not scored")
                for feature, tokens in zip(a["original_feature_ids"], a["tokens"], strict=True):
                    identity = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
                    need(feature not in feature_hashes or feature_hashes[feature] == identity, "Global feature/token consistency")
                    feature_hashes[feature] = identity
                rows = target["rows"]
                need(rows and len(rows) == len({(r["time"], r["query_index"]) for r in rows}), "Unique nonempty loss rows")
                covered = set()
                for n, r in enumerate(rows):
                    ti, j, label = r["time"], r["query_position"], r["label_index"]
                    need(all(integer(v) for v in (ti, j, label)) and ti < w["public_user_turns"] and j < w["queries"] and label < len(a["candidate_ids"][j]), "Supported endpoint indices")
                    qi, cid = a["query_ids"][j], a["candidate_ids"][j][label]
                    need(r["split"] == split and r["dialogue_id"] == did and r["source_row_index"] == n and r["query_index"] == qi and r["label_id"] == cid and r["turn_index"] == a["user_turn_indices"][ti], "Exact evaluator/actor endpoint join")
                    need(type(r["unseen"]) is bool and (split != "train" or not r["unseen"]) and type(r["dontcare"]) is bool and r["dontcare"] == (cid == DC), "Evaluator support flags")
                    stratum = r["bin"] if r["bin"] in STRATA[:2] else "changed"
                    need(r["bin"] in (*STRATA[:2], "first_assignment", "revision", "clear") and r["stratum"] == stratum and r["stratum_index"] == STRATA.index(stratum), "Bin/stratum mapping")
                    metadata = (split, r["query_id"], r["service"], r["slot"], r["unseen"], tuple(a["candidate_ids"][j]))
                    need(qi not in query_metadata or query_metadata[qi] == metadata, "Consistent global schema identity")
                    query_metadata[qi] = metadata
                    covered.add(qi)
                    counts[split][stratum] += 1
                need(covered == set(a["query_ids"]), "Complete supplied-query evaluator support")
                previous = {}
                for r in sorted(rows, key=lambda x: (x["time"], x["query_index"])):
                    qi = r["query_index"]
                    need(r["bin"] == transitions(previous.get(qi, NONE), r["label_id"]), "Chronological annotation transition")
                    previous[qi] = r["label_id"]
                reconstructed.append({"split": split, "dialogue_id": did, "work": w})
                reconstructed_index.append({"split": split, "dialogue_id": did, "shape": a["lexical_shape"], "offset": offset, "scored_rows": len(rows)})
                offset += w["lexical_scalars"]
                row_count += len(rows)
        need(len(seen) == expected_dialogues and row_count == expected_rows, "Complete split membership/endpoints")
    need(offset == 20007650 and len(set(actor_ids["train"]) & set(actor_ids["dev"])) == 298, "Complete split-qualified lexical geometry")
    need(read(PREP / "index.json") == reconstructed_index and plan["lexical_changed_positions_by_feature"] == changes.tolist(), "Independent index and lexical changes")
    workloads = read(PREP / "workloads.json")
    need(workloads["profiles"] == reconstructed and workloads["version"] == "dialogue-finetune-dataset-v2", "Exact reconstructed per-dialogue work")
    summed = [k for k in reconstructed[0]["work"] if k not in ("chunk_tokens", "chunk_batch_size") and not k.startswith("max_")]
    maximized = [k for k in reconstructed[0]["work"] if k not in ("chunk_tokens", "chunk_batch_size")]
    def reduce(group):
        return {"dialogues": len(group), "totals": {k: sum(x["work"][k] for x in group) for k in summed},
                "maxima": {k: max(x["work"][k] for x in group) for k in maximized}}
    need(workloads["all"] == reduce(reconstructed), "All sums/maxima")
    for split in actor_ids:
        need(workloads["splits"][split] == reduce([x for x in reconstructed if x["split"] == split]), "Split sums/maxima")
        need(plan["loss_counts"][split] == {str(i): counts[split][s] for i, s in enumerate(STRATA)}, "All stratum counts")
    weights = [51741 / (3 * counts["train"][s]) for s in STRATA]
    old_path = ROOT / "runs/dialogue-copy-v2/study-01/plan.json"
    need(sha(old_path) == OLD_PLAN == plan["input_sha256"]["runs/dialogue-copy-v2/study-01/plan.json"], "Historical objective identity")
    need(weights == plan["loss_weights"] == read(old_path)["loss_weights"], "Original training-only stratum weights")
    orders = read(PREP / "orders.json")
    need(orders["dialogue_ids"] == actor_ids["train"] and orders["seeds"] == [6901, 6902, 6903] and orders["epochs"] == 20, "Fixed paired order metadata")
    work_by_id = {p["dialogue_id"]: p["work"] for p in reconstructed if p["split"] == "train"}
    metrics = ("encoder_calls", "padded_token_positions", "padded_attention_positions", "real_question_updates", "padded_candidate_positions", "encoder_sequences")
    maxima, sizes = {}, Counter()
    for seed in (6901, 6902, 6903):
        generator = np.random.Generator(np.random.PCG64(seed))
        sequence = orders["orders"][str(seed)]
        need(len(sequence) == 20, "Twenty epochs")
        for epoch, order in enumerate(sequence):
            need(order == generator.permutation(2017).tolist(), "Exact seeded PCG64 permutation")
            for begin in range(0, 2017, 32):
                indices = order[begin:begin + 32]
                dids = [actor_ids["train"][i] for i in indices]
                sizes[str(len(indices))] += 1
                for k in metrics:
                    total = sum(work_by_id[did][k] for did in dids)
                    if k not in maxima or total > maxima[k]["value"]:
                        maxima[k] = {"value": total, "seed": seed, "epoch": epoch, "batch_start": begin,
                                     "dialogue_indices": indices, "dialogue_ids": dids}
    batch = read(PREP / "effective-batches.json")
    need(batch["sizes"] == dict(sizes) == {"32": 3780, "1": 60} and batch["batches"] == 3840 and batch["maxima"] == maxima, "All 3840 batches and earliest tied maximum records")
    need(plan["updates_per_fit"] == 1280 and plan["fits"] == 12 and plan["training_dialogue_visits"] == 484080 and plan["training_supervised_presentations"] == 51741 * 20 * 12, "Complete prospective allocation counts")
    need(done["encoder_calls"] == 0 and done["model_weights_loaded"] is False and done["official_test_opened"] is False, "Preparation scope witnesses")
    need(math.isfinite(done["wall_seconds"]) and done["wall_seconds"] <= plan["caps"]["wall_seconds"] and done["peak_rss_bytes"] <= plan["caps"]["rss_bytes"], "Recorded preparation caps")
    need(sum(p.stat().st_size for p in PREP.iterdir()) <= plan["caps"]["output_bytes"], "Actual output cap")
    need(sha(PREP / "completed.json") == COMPLETED, "Terminal identity unchanged")
    return {"status": "clear", "agreement": True, "payload_files_verified": 13, "source_files_verified": 39,
            "dialogues": {s: len(v) for s, v in actor_ids.items()}, "endpoints": {s: sum(v.values()) for s, v in counts.items()},
            "split_qualified_identity": True, "cross_split_bare_id_overlap": 298, "lexical_positions": offset,
            "original_lexical_byte_parity": True, "reserved_boolean_unchanged": True,
            "profiles_reconstructed": len(reconstructed), "epoch_orders_reconstructed": 60,
            "effective_batches_reconstructed": 3840, "largest_batch_measures_checked": len(metrics),
            "stratum_counts": {s: dict(v) for s, v in counts.items()}, "loss_weights": weights,
            "lexical_changes_by_feature": changes.tolist(), "work_totals": workloads["all"]["totals"],
            "limits": ["No raw public text, tokenizer, model, score or accuracy evaluation. Actor tokenization and the new lexical matches are inherited preparation witnesses, not rederived from text.",
                       "Original lexical equality is established by the exact original input SHA, without rereading its duplicate bytes.",
                       "Target labels are read only to validate support, joins, transition bins and training weights; deterministic registers are checked only against actor columns, never scored.",
                       "Only metadata PCG64 order generation is reproduced; no neural or training random process is run.",
                       "Identity and work agreement do not establish elapsed training cost, memory admission or task quality."]}


if __name__ == "__main__":
    began = time.perf_counter()
    script_hash = sha(Path(__file__))
    def timeout(*_):
        raise TimeoutError("Saved-preparation audit exceeded 60 seconds")
    need(not (OUT / "receipt.json").exists() and not (OUT / "failed.json").exists(), "Exclusive audit terminal")
    write(OUT / "started.json", {"script_sha256": script_hash, "preparation_completed_sha256": COMPLETED,
          "preparation_plan_sha256": PLAN, "cap_seconds": 60, "runtime": {"python": platform.python_version(), "numpy": np.__version__}})
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        result = execute()
        signal.setitimer(signal.ITIMER_REAL, 0)
        need(sha(Path(__file__)) == script_hash and time.perf_counter() - began <= 60, "Audit identity/deadline")
        result.update(script_sha256=script_hash, preparation_completed_sha256=COMPLETED, preparation_plan_sha256=PLAN,
                      started_sha256=sha(OUT / "started.json"), wall_seconds=time.perf_counter() - began,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      experimental_model_calls=0, tokenizer_calls=0, literal_accuracy_computed=False)
        write(OUT / "receipt.json", result)
        print(json.dumps({"status": result["status"], "receipt_sha256": sha(OUT / "receipt.json"), "script_sha256": script_hash,
                          "dialogues": result["dialogues"], "endpoints": result["endpoints"], "wall_seconds": result["wall_seconds"]}))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            write(OUT / "failed.json", {"status": "failed", "error": repr(error), "script_sha256": script_hash,
                  "wall_seconds": time.perf_counter() - began})
        except BaseException as secondary:
            error.add_note("Failure receipt: " + repr(secondary))
        raise
