"""Metadata-only preparation of the privileged previous-value observation probe.

Mixed JSON containers are decoded, then projected onto explicit metadata fields.
Text values are never interpreted, emitted or used for admission. Feature files
are opaque hash inputs; only whitelisted int64 index arrays are numerically read.
No experiment loaders, neural libraries, tokenizers or model code are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-conditional-preparation-v1"
PROTOCOL = "research/dialogue-conditional-preparation-protocol.md"
DESIGN = "research/dialogue-conditional-observation-design.md"
SELF = "scripts/prepare_dialogue_conditional.py"
TEST = "tests/test_prepare_dialogue_conditional.py"
WALL_CAP = 120.
OUTPUT_CAP = 256 * 1024**2
NONE, DONTCARE = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
REASONS = ("admitted", "first_public_turn", "missing_adjacent_scored_predecessor", "previous_candidate_absent")
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
FIELDS = ("user_match", "system_match", "unique_longest_user", "unique_longest_system", "literal_current",
          "literal_previous", "is_none", "is_dontcare", "affirmative_cue_for_true", "negative_cue_for_false")
REFERENCE_SOURCES = {
    "scripts/study_dialogue_memory.py": "51a46d35c3d9dd2857bd3ded5fc289652c7e18f05b85b8f82dd245ade7626e2c",
    "scripts/study_dialogue_copy.py": "d628616f24d973f40a43d68b18758010547449877bdc569324e2cd920cd49996",
    "scripts/study_dialogue_tokens.py": "88e28a1455862c02f562798b31c1b9992c844b7abcb35c4a8649e7ff37075bee",
    "scripts/prepare_dialogue_tokens.py": "fa630dfb96d3b37184045703981dad6054813e03c2fcb948c14ca98ffda9b409",
    "src/openjev/research/dialogue_state_data.py": "7749e97e64575ae02c9fc843573e93e82d5e9cbb301410e47881e18190f0c102",
    "src/openjev/research/dialogue_copy_features.py": "78a2c2e26c7fae4a30fad0380a6832b998b1f696c8d0edbb7b53cd3aaa1ac503",
}
INPUTS = {
    "data": ("runs/sgd-state-v1/data/completed.json", "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12"),
    "packet": ("runs/sgd-state-v1/features-02/completed.json", "e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308"),
    "lexical": ("runs/dialogue-copy-v1/lexical-01/completed.json", "2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54"),
    "token_preparation": ("runs/dialogue-token-v1/preparation-01/completed.json", "094b31d5cf7895ad700d24caa7e3f449dc5781a0f2a5a997b1a4d1262a3d8912"),
    "token_plan": ("runs/dialogue-token-v1/preparation-01/plan.json", "c2120622a0b11f0ffb1a9e4925ca1f2619f193a9f228a6d0728759f62dceadf4"),
    "token_capacity": ("runs/dialogue-token-v1/capacity-01/completed.json", "3de6725bae68443f1326e9fdd1a3506ef355d936bfa98e9ef377aedf6e6f9dbc"),
    "tokens": ("runs/dialogue-token-v1/features-01/completed.json", "45d63e97698d1d549ad68c9c331cad2154bb17b9c70535392d00bd78edcd56ee"),
    "cache_audit": ("output/dialogue-token-v1/cache-audit-01/result-01/receipt.json", "3752c520ab58ee72e47f8a74fdd0ae99a58670c2bdd4b4803451100514c44808"),
    "cache_audit_summary": ("output/dialogue-token-v1/cache-audit-01/result-01/summary.json", "d6053c9cf552c20fb8bdc1400e2d27e1c99a43f47f31777b2d6a7fe5132b0d0c"),
}
INTEGER_FILES = {"context-indices.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy", "text-token-counts.npy"}
FLOAT_FILES = {"features.npy", "lexical.npy", "tokens.npy", "priors.npy"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(x, minimum=0):
    return type(x) is int and x >= minimum


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object, parse_constant=invalid)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def safe(base, name):
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe member path")
    path = base / name
    require(not path.is_symlink() and path.resolve().is_relative_to(base.resolve()), "Escaped/symlink member")
    return path


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            check()
    return digest.hexdigest()


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy")}


def source_map(check=lambda: None):
    result = {name: sha(safe(ROOT, name), check) for name in (*REFERENCE_SOURCES, SELF, TEST, PROTOCOL, DESIGN)}
    require(all(result[k] == v for k, v in REFERENCE_SOURCES.items()), "Reference source drift")
    return result


class Budget:
    def __init__(self, out, started):
        self.out, self.started = out, started
        self.progress = {"phase": "initializing", "hashed_files": 0, "rows_written": 0, "admitted_rows": 0}

    def check(self):
        if time.monotonic() - self.started > WALL_CAP:
            raise TimeoutError("Whole preparation cap exceeded")

    def storage(self, extra=0):
        self.check()
        total = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(total + extra <= OUTPUT_CAP, "Output storage cap exceeded")
        return total


def binding(path, expected, budget, verified):
    require(path.is_file() and not path.is_symlink(), "Missing or symlink input")
    if str(path) in verified:
        require(verified[str(path)]["sha256"] == expected, "Conflicting input hashes")
        return
    require(sha(path, budget.check) == expected, "Input hash mismatch: " + str(path))
    verified[str(path)] = {"sha256": expected, "bytes": path.stat().st_size}
    budget.progress["hashed_files"] += 1


def file_entry(item):
    if type(item) is str:
        return item, None
    require(type(item) is dict and set(item) == {"sha256", "bytes"} and integer(item["bytes"]), "File record schema")
    return item["sha256"], item["bytes"]


def receipts(budget, *, payloads):
    docs, verified = {}, {}
    for key, (name, pin) in INPUTS.items():
        path = safe(ROOT, name)
        binding(path, pin, budget, verified)
        docs[key] = read(path)
        require(not any((path.parent / p).exists() for p in ("failed.json", "late-completion.json")), "Failed input")
        if key != "token_plan":
            require(docs[key]["status"] == "completed", "Incomplete input")
        if payloads and "files" in docs[key]:
            for member, item in docs[key]["files"].items():
                digest, size = file_entry(item)
                target = safe(path.parent, member)
                binding(target, digest, budget, verified)
                require(size is None or target.stat().st_size == size, "Input file bytes")
    packet_dir = safe(ROOT, INPUTS["packet"][0]).parent
    ep = packet_dir / "encoder-plan.json"
    binding(ep, docs["packet"]["files"]["encoder-plan.json"], budget, verified)
    encoder = read(ep)
    plan = docs["token_plan"]
    require(encoder["input_sha256"]["completed.json"] == INPUTS["data"][1], "Pooled data lineage")
    models = encoder["model_files_sha256"]
    require(set(models) == {"config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json",
                           "special_tokens_map.json", "vocab.txt"}, "Six encoder files")
    require(docs["packet"]["model_files_sha256"] == models == plan["model_files_sha256"], "Encoder file identity")
    require(encoder["revision"] == plan["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
            and encoder["encoder"] == plan["encoder"] == "sentence-transformers/all-MiniLM-L6-v2", "Encoder recipe identity")
    for key in ("data", "lexical"):
        require(docs[key]["test_contents_accessed"] is False, "Test boundary")
    for key in ("packet", "data"):
        require(docs["lexical"][key + "_completed_sha256"] == INPUTS[key][1], "Lexical parent identity")
        require(plan["inputs"][key]["completed_sha256"] == INPUTS[key][1], "Token plan parent identity")
    require(plan["inputs"]["lexical"]["completed_sha256"] == INPUTS["lexical"][1], "Token lexical identity")
    require(docs["token_preparation"]["plan_sha256"] == INPUTS["token_plan"][1]
            and docs["token_preparation"]["source_sha256"] == plan["source_sha256"], "Token freeze identity")
    for key, phase in (("token_capacity", "capacity"), ("tokens", "encode")):
        d = docs[key]
        require(d["phase"] == phase and d["version"] == "dialogue-token-minilm-v1" and d["no_retry"] is True
                and d["test_contents_accessed"] is False and d["parameter_training_steps"] == 0, "Token cache scope")
        require(d["plan_sha256"] == INPUTS["token_plan"][1] and d["source_sha256"] == plan["source_sha256"]
                and d["model_files_sha256"] == models and d["runtime"] == plan["runtime"], "Token cache lineage")
        for parent in ("packet", "lexical", "data"):
            require(d[parent + "_completed_sha256"] == INPUTS[parent][1], "Token parent mismatch")
    require(docs["tokens"]["capacity_completed_sha256"] == INPUTS["token_capacity"][1]
            and docs["tokens"]["capacity_path"] == str(Path(INPUTS["token_capacity"][0]).parent), "Capacity parent mismatch")
    require(docs["token_capacity"]["projection"]["encoding_permitted"] is True
            and 0 < docs["tokens"]["wall_seconds"] <= 900
            and docs["tokens"]["pooling_equivalence"]["tolerance"] == 2e-5
            and 0 <= docs["tokens"]["pooling_equivalence"]["max_abs_error"] <= 2e-5, "Successful cache preparation witness")
    require(docs["cache_audit"]["cache_completed_sha256"] == INPUTS["tokens"][1]
            and docs["cache_audit_summary"]["cache_completed_sha256"] == INPUTS["tokens"][1]
            and docs["cache_audit"]["capacity_completed_sha256"] == INPUTS["token_capacity"][1]
            and docs["cache_audit"]["files"]["summary.json"]["sha256"] == INPUTS["cache_audit_summary"][1], "Independent audit binding")
    if payloads:
        for name, digest in {**docs["data"]["implementation_sha256"], **docs["lexical"]["source_sha256"],
                             **plan["source_sha256"]}.items():
            binding(safe(ROOT, name), digest, budget, verified)
    return docs, verified


def array_header(path):
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), "Unsupported array header")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
    require(not fortran and not dtype.hasobject and all(integer(n) for n in shape), "Unsafe array layout")
    require(Path(path).stat().st_size == offset + math.prod(shape) * dtype.itemsize, "Array payload size")
    return shape, dtype


def int_array(path):
    require(Path(path).name in INTEGER_FILES, "Only integer metadata arrays may be decoded")
    shape, dtype = array_header(path)
    require(len(shape) == 1 and dtype == np.dtype("int64"), "Integer metadata dtype/rank")
    return np.load(path, allow_pickle=False)


def float_header(path, expected_shape=None):
    require(Path(path).name in FLOAT_FILES, "Unlisted float header")
    shape, dtype = array_header(path)
    require(dtype == np.dtype("float32") and (expected_shape is None or shape == expected_shape), "Float header shape/dtype")
    return shape


def transition(previous, current):
    return ("unmentioned_retention" if previous == current == NONE else "assigned_retention" if previous == current
            else "first_assignment" if previous == NONE else "clear" if current == NONE else "revision")


def catalog_metadata(packet, catalog, feature_rows):
    require(set(catalog) == {"train", "dev"}, "Catalog split membership")
    lookups = {}
    for split in ("train", "dev"):
        items = catalog[split]
        lookups[split] = {q["query_id"]: q for q in items}
        require(len(lookups[split]) == len(items), "Duplicate catalog query")
    queries, identities = [], set()
    for i, q in enumerate(packet["queries"]):
        split = q["split"]
        require(split in lookups and all(type(q[k]) is str and q[k] for k in ("id", "service", "slot")), "Query identity")
        identity = (split, q["service"], q["slot"])
        require(identity not in identities and json.loads(q["id"]) == [q["service"], q["slot"]], "Ambiguous query identity")
        identities.add(identity)
        c = lookups[split][q["id"]]
        ids, values = q["candidate_ids"], q["candidate_values"]
        require(ids == [x["id"] for x in c["candidates"]] and values == [x.get("value") for x in c["candidates"]]
                and c["service"] == q["service"] and c["slot"] == q["slot"], "Catalog candidate identity")
        require(type(ids) is list and 3 <= len(ids) <= 12 and len(set(ids)) == len(ids)
                and ids[:2] == [NONE, DONTCARE] and values[:2] == [None, None]
                and all(type(v) is str and v and cid == "value:" + v for cid, v in zip(ids[2:], values[2:], strict=True)), "Candidate IDs/values")
        require(len(q["candidates"]) == len(ids) and all(integer(x) and x < feature_rows for x in [q["text"], *q["candidates"]]), "Embedding index range")
        queries.append({"query_index": i, "query_id": q["id"], "split": split, "service": q["service"], "slot": q["slot"],
                        "query_feature_index": q["text"], "candidate_feature_indices": q["candidates"],
                        "candidate_ids": ids, "candidate_values": values,
                        "boolean_slot": {v.strip().casefold() for v in values[2:]} == {"true", "false"}})
    return queries


def validate_metadata(packet, lexical, token, arrays, queries, float_shapes):
    require(set(packet["cohorts"]) == set(lexical["cohorts"]) == set(token["cohorts"]) == {"train", "dev"}, "Only train/dev cohorts")
    require(lexical["features"] == list(FIELDS), "Lexical field order")
    lengths, offsets = arrays["text-token-counts.npy"], arrays["offsets.npy"]
    chunks, chunk_offsets, expected_offsets = [], [0], [0]
    for n in lengths:
        require(int(n) > 0, "Empty token context")
        pieces = [min(254, int(n) - start) for start in range(0, int(n), 254)]
        chunks.extend(pieces)
        chunk_offsets.append(len(chunks))
        expected_offsets.append(expected_offsets[-1] + int(n) + 2 * len(pieces))
    require(np.array_equal(offsets, expected_offsets) and np.array_equal(arrays["chunk-offsets.npy"], chunk_offsets)
            and np.array_equal(arrays["chunk-lengths.npy"], chunks), "Chunk/offset arithmetic")
    original = token["original_feature_indices"]
    require(token["version"] == "dialogue-token-minilm-v1" and token["width"] == 384
            and len(original) == token["unique_contexts"] == len(lengths) and len(set(original)) == len(original)
            and all(integer(x) and x < float_shapes["features.npy"][0] for x in original), "Token original feature identities")
    require(float_shapes["tokens.npy"] == (int(offsets[-1]), 384) and float_shapes["priors.npy"] == (int(offsets[-1]),), "Token/prior header geometry")
    ids = arrays["context-indices.npy"]
    require(len(ids) == token["context_occurrences"] and ((ids >= 0) & (ids < len(original))).all(), "Context index range")
    layouts, lc, tc, all_ids = {}, 0, 0, set()
    for split in ("train", "dev"):
        ds, ls, ts = packet["cohorts"][split], lexical["cohorts"][split], token["cohorts"][split]
        require(len(ds) == len(ls) == len(ts), "Cohort size mismatch")
        for d, lex, tok in zip(ds, ls, ts, strict=True):
            require(type(d["id"]) is str and d["id"] and (split, d["id"]) not in all_ids, "Duplicate/invalid dialogue")
            all_ids.add((split, d["id"]))
            turns, records = d["turns"], d["queries"]
            require(type(turns) is list and turns and type(records) is list and records
                    and all(integer(x) and x < float_shapes["features.npy"][0] for x in turns), "Public turn indices")
            require(all(integer(r["query"]) and r["query"] < len(queries) and queries[r["query"]]["split"] == split for r in records), "Query range/split")
            qids = sorted({r["query"] for r in records})
            shape = [len(turns), len(qids), max(len(queries[q]["candidate_ids"]) for q in qids), 10]
            for entry in (lex, tok):
                require(integer(entry["offset"]) and type(entry["shape"]) is list
                        and all(integer(x, 1) for x in entry["shape"]) and type(entry["query_ids"]) is list
                        and all(integer(x) for x in entry["query_ids"]), "Strict layout integer types")
            require(lex == {"id": d["id"], "query_ids": qids, "offset": lc, "shape": shape}, "Lexical layout mismatch")
            require(tok == {"id": d["id"], "query_ids": qids, "offset": tc, "shape": [len(turns)]}, "Token layout mismatch")
            require([original[int(x)] for x in ids[tc:tc+len(turns)]] == turns, "Pooled/token context identity")
            layouts[(split, d["id"])] = {"lexical": lex, "token": tok}
            lc += math.prod(shape)
            tc += len(turns)
    require(lc == float_shapes["lexical.npy"][0] and tc == len(ids), "Unused metadata payload")
    return layouts


def row_ledger(packet, queries, layouts, arrays):
    """Yield all scored rows; labels enter admission/targets, never feature construction."""
    number = 0
    panel_flags = {}
    service_flags = {}
    training_services = {q["service"] for q in queries if q["split"] == "train"}
    for split in ("train", "dev"):
        for d in packet["cohorts"][split]:
            seen, previous = set(), {}
            layout = layouts[(split, d["id"])]
            lex, tok = layout["lexical"], layout["token"]
            require(all(integer(r["time"]) and integer(r["query"]) and r["query"] < len(queries)
                        for r in d["queries"]), "Scored time/query integer")
            for source_index, r in sorted(enumerate(d["queries"]), key=lambda x: (x[1]["time"], x[1]["query"])):
                t, qi, label = r["time"], r["query"], r["label"]
                q = queries[qi]
                require(integer(t) and t < len(d["turns"]) and integer(label) and label < len(q["candidate_ids"]), "Scored time/label range")
                key = (q["service"], q["slot"])
                require((t, key) not in seen, "Duplicate scored query/time")
                seen.add((t, key))
                require(type(r["unseen"]) is bool and type(r["dontcare"]) is bool and r["bin"] in BINS, "Scored metadata types")
                require(panel_flags.setdefault((split, q["query_id"]), r["unseen"]) == r["unseen"], "Inconsistent query panel flag")
                require(service_flags.setdefault((split, q["service"]), r["unseen"]) == r["unseen"], "Inconsistent service panel flag")
                require(not r["unseen"] if split == "train" or q["service"] in training_services else True, "Known training service marked unseen")
                current = q["candidate_ids"][label]
                require(r["dontcare"] == (current == DONTCARE), "DONTCARE metadata disagreement")
                old = previous.get(key)
                adjacent = old is not None and old["time"] == t - 1
                old_id = old["current_candidate_id"] if adjacent else None
                mapped = q["candidate_ids"].index(old_id) if old_id in q["candidate_ids"] else None
                reason = ("first_public_turn" if t == 0 else "missing_adjacent_scored_predecessor" if not adjacent
                          else "previous_candidate_absent" if mapped is None else "admitted")
                derived = transition(old_id, current) if reason == "admitted" else None
                if reason == "admitted":
                    require(r["bin"] == derived, "Inherited adjacent transition bin differs")
                context = int(arrays["context-indices.npy"][tok["offset"]+t])
                start, stop = map(int, arrays["offsets.npy"][context:context+2])
                local_q = lex["query_ids"].index(qi)
                lexical_start = lex["offset"] + (t*lex["shape"][1]+local_q)*lex["shape"][2]*10
                value = q["candidate_values"][label]
                group = ("dontcare" if current == DONTCARE else value.strip().casefold()
                         if q["boolean_slot"] and label >= 2 else "none" if current == NONE else "other")
                row = {"row_index": number, "source_row_index": source_index,
                       "split": split, "dialogue_id": d["id"], "time": t,
                       "query_index": qi, "query_id": q["query_id"], "service": q["service"], "slot": q["slot"],
                       "unseen": r["unseen"], "current_label_index": label, "current_candidate_id": current,
                       "inherited_bin": r["bin"], "admission": reason,
                       "previous_row_index": old["row_index"] if adjacent else None,
                       "previous_query_index": old["query_index"] if adjacent else None,
                       "previous_candidate_id": old_id, "previous_current_index": mapped,
                       "derived_bin": derived, "candidate_count": len(q["candidate_ids"]),
                       "boolean_slot": q["boolean_slot"], "current_value_group": group,
                       "cache": {"pooled_index": d["turns"][t], "query_feature_index": q["query_feature_index"],
                                 "candidate_feature_indices": q["candidate_feature_indices"], "token_context_index": context,
                                 "token_start": start, "token_stop": stop, "lexical_start": lexical_start,
                                 "lexical_candidates": len(q["candidate_ids"]), "lexical_stride": 10}}
                previous[key] = row
                number += 1
                yield row


def aggregate(rows):
    """All summaries are cohort metadata, never prediction accuracy or loss."""
    groups = {}
    for row in rows:
        for panel in ("all", "unseen" if row["unseen"] else "seen"):
            key = row["split"] + "/" + panel
            g = groups.setdefault(key, {"scored_rows": 0, "reasons": Counter(), "bins": Counter(), "values": Counter(),
                                       "bin_values": Counter(), "token_candidate_histogram": Counter(), "dialogues": set(),
                                       "admitted_dialogues": set(), "admitted_queries": set(), "unique_contexts": set(),
                                       "sum_score_positions": 0, "max_score_positions": 0,
                                       "sum_float_input_scalars": Counter(), "max_float_input_scalars": Counter()})
            g["scored_rows"] += 1
            g["reasons"][row["admission"]] += 1
            g["dialogues"].add(row["dialogue_id"])
            if row["admission"] != "admitted":
                continue
            g["admitted_dialogues"].add(row["dialogue_id"])
            g["admitted_queries"].add((row["dialogue_id"], row["query_id"]))
            g["unique_contexts"].add(row["cache"]["token_context_index"])
            g["bins"][row["derived_bin"]] += 1
            g["values"][row["current_value_group"]] += 1
            g["bin_values"][row["derived_bin"] + "/" + row["current_value_group"]] += 1
            length = row["cache"]["token_stop"] - row["cache"]["token_start"]
            c = row["candidate_count"]
            g["token_candidate_histogram"][f"{length},{c}"] += 1
            g["sum_score_positions"] += length*c
            g["max_score_positions"] = max(g["max_score_positions"], length*c)
            # Per-row logical input payloads, not allocations or cache traffic.
            values = {"mean": 384+384+c*384+c*10+c,
                      "tokens": length*384+length+384+c*384+c*10+c}
            for arm, n in values.items():
                g["sum_float_input_scalars"][arm] += n
                g["max_float_input_scalars"][arm] = max(g["max_float_input_scalars"][arm], n)
    for split in ("train", "dev"):
        for panel in ("all", "seen", "unseen"):
            key = split + "/" + panel
            if key not in groups:
                groups[key] = {"scored_rows": 0, "reasons": {}, "bins": {}, "values": {}, "bin_values": {},
                               "token_candidate_histogram": {}, "dialogues": set(), "admitted_dialogues": set(),
                               "admitted_queries": set(), "unique_contexts": set(), "sum_score_positions": 0,
                               "max_score_positions": 0, "sum_float_input_scalars": {}, "max_float_input_scalars": {}}
    for g in groups.values():
        for key in ("dialogues", "admitted_dialogues", "admitted_queries", "unique_contexts"):
            g[key] = len(g[key])
        g["reasons"] = {k: g["reasons"].get(k, 0) for k in REASONS}
        g["adjacent_eligible_rows"] = g["reasons"]["admitted"] + g["reasons"]["previous_candidate_absent"]
        g["bins"] = {k: g["bins"].get(k, 0) for k in BINS}
        g["values"] = {k: g["values"].get(k, 0) for k in ("none", "true", "false", "dontcare", "other")}
        g["changed"] = sum(g["bins"][k] for k in BINS[2:])
        g["retained"] = sum(g["bins"][k] for k in BINS[:2])
        for prefix in ("sum", "max"):
            g[prefix+"_float_input_bytes"] = {k: 4*v for k, v in g[prefix+"_float_input_scalars"].items()}
        require(sum(g["reasons"].values()) == g["scored_rows"] and g["changed"]+g["retained"] == g["reasons"]["admitted"], "Admission accounting")
    return groups


def manifest(out, check=lambda: None):
    return {p.relative_to(out).as_posix(): {"sha256": sha(p, check), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file()}


def validate_plan(path, pin, budget):
    require(sha(path, budget.check) == pin, "External plan hash")
    plan = read(path)
    require(plan["version"] == VERSION and plan["runtime"] == runtime() and plan["inputs"] == {k: {"path": v[0], "sha256": v[1]} for k, v in INPUTS.items()}
            and plan["limits"] == {"wall_seconds": WALL_CAP, "output_bytes": OUTPUT_CAP}, "Plan recipe/runtime")
    require(plan["source_sha256"] == source_map(budget.check), "Plan source identity")
    frozen = read(path.parent / "completed.json")
    require(frozen["status"] == "completed" and frozen["phase"] == "freeze" and frozen["plan_sha256"] == pin
            and frozen["source_sha256"] == plan["source_sha256"] and frozen["no_retry"] is True, "Freeze completion")
    require(not (path.parent / "failed.json").exists() and not (path.parent / "late-completion.json").exists(), "Failed freeze")
    expected = {"started.json", "plan.json"} | {"sources/"+k for k in plan["source_sha256"]} | {"inputs/"+k+".json" for k in INPUTS}
    require(set(frozen["files"]) == expected, "Freeze source/input snapshot membership")
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}
            == set(frozen["files"]) | {"completed.json"}, "Freeze exact membership")
    for name, item in frozen["files"].items():
        member = safe(path.parent, name)
        require(sha(member, budget.check) == item["sha256"] and member.stat().st_size == item["bytes"], "Freeze payload mismatch")
    for name, digest in plan["source_sha256"].items():
        require(frozen["files"]["sources/"+name]["sha256"] == digest, "Frozen source snapshot identity")
    for name, (_, digest) in INPUTS.items():
        require(frozen["files"]["inputs/"+name+".json"]["sha256"] == digest, "Frozen receipt snapshot identity")
    return plan


def freeze_body(out, budget, _args):
    docs, verified = receipts(budget, payloads=False)
    sources = source_map(budget.check)
    for name in sources:
        target = out / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(ROOT, name), target)
    for key, (name, _) in INPUTS.items():
        (out / "inputs").mkdir(exist_ok=True)
        shutil.copyfile(safe(ROOT, name), out / "inputs" / (key + ".json"))
    plan = {"version": VERSION, "runtime": runtime(), "source_sha256": sources,
            "inputs": {k: {"path": v[0], "sha256": v[1]} for k, v in INPUTS.items()},
            "limits": {"wall_seconds": WALL_CAP, "output_bytes": OUTPUT_CAP},
            "model_files_sha256": docs["token_plan"]["model_files_sha256"],
            "scope": "Whitelisted metadata only; no float feature decoding, encoder, model, corpus-text interpretation or official test access"}
    write(out / "plan.json", plan)
    require(source_map(budget.check) == sources, "Freeze source changed during snapshots")
    for name, item in verified.items():
        require(sha(name, budget.check) == item["sha256"], "Freeze metadata changed during snapshots")
    return {"plan_sha256": sha(out / "plan.json", budget.check), "source_sha256": sources,
            "authenticated_inputs": verified, "metadata_rows_extracted": 0}


def prepare_body(out, budget, args):
    plan = validate_plan(Path(args.plan), args.plan_sha256, budget)
    docs, verified = receipts(budget, payloads=True)
    dirs = {k: safe(ROOT, INPUTS[k][0]).parent for k in ("data", "packet", "lexical", "tokens")}
    # Explicitly permitted mixed-container decode. Only metadata fields below are used.
    packet, catalog = read(dirs["packet"] / "packet.json"), read(dirs["data"] / "catalog.json")
    lexical, token = read(dirs["lexical"] / "index.json"), read(dirs["tokens"] / "index.json")
    float_shapes = {"features.npy": float_header(dirs["packet"] / "features.npy"),
                    "lexical.npy": float_header(dirs["lexical"] / "lexical.npy"),
                    "tokens.npy": float_header(dirs["tokens"] / "tokens.npy"),
                    "priors.npy": float_header(dirs["tokens"] / "priors.npy")}
    require(len(float_shapes["features.npy"]) == 2 and float_shapes["features.npy"][1] == 384
            and len(float_shapes["lexical.npy"]) == 1, "Pooled/lexical header geometry")
    arrays = {name: int_array(dirs["tokens"] / name) for name in INTEGER_FILES}
    queries = catalog_metadata(packet, catalog, float_shapes["features.npy"][0])
    layouts = validate_metadata(packet, lexical, token, arrays, queries, float_shapes)
    require(len(arrays["text-token-counts.npy"]) == docs["tokens"]["unique_contexts_encoded"] == docs["token_plan"]["unique_contexts"], "Unique context receipt count")
    write(out / "catalog.json", {"version": VERSION, "queries": queries})
    budget.progress["phase"] = "row_ledger"
    def stream():
        with (out / "rows.jsonl").open("x") as handle:
            for row in row_ledger(packet, queries, layouts, arrays):
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                budget.progress["rows_written"] += 1
                budget.progress["admitted_rows"] += row["admission"] == "admitted"
                if budget.progress["rows_written"] % 256 == 0:
                    handle.flush()
                    budget.storage()
                yield row
    groups = aggregate(stream())
    for split in ("train", "dev"):
        require(groups[split+"/all"]["scored_rows"] == docs["packet"]["cohorts"][split]["queries"]
                and groups[split+"/all"]["dialogues"] == docs["packet"]["cohorts"][split]["dialogues"], "Original packet coverage")
    summary = {"version": VERSION, "groups": groups, "rows": budget.progress["rows_written"],
               "admitted_rows": budget.progress["admitted_rows"], "source_sha256": plan["source_sha256"],
               "work_scope": "One unbatched row: score positions L*C; float32 inputs include evidence, query, candidates, ten lexical features, previous onehot and token priors. Logical repeated payload, not allocations/traffic/latency.",
               "validity_boundary": "Float feature contents authenticated by opaque byte hashes; numerical finiteness and pooling validity inherited from the pinned cache audit, not recomputed.",
               "panel_boundary": "Unseen flags inherited from the pinned full-schema parser; per-query/service consistency and known training-service negatives checked. Absent categorical service does not imply unseen; full schema membership is not reconstructed.",
               "no_accuracy_or_loss_computed": True, "no_training_authorized": True}
    write(out / "summary.json", summary)
    require(source_map(budget.check) == plan["source_sha256"], "End source stability")
    for name, item in verified.items():
        require(Path(name).stat().st_size == item["bytes"] and sha(name, budget.check) == item["sha256"], "End input stability")
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "authenticated_inputs": verified,
            "rows": summary["rows"], "admitted_rows": summary["admitted_rows"], "no_training_authorized": True}


def attempt(phase, out, body, args):
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, started)
    request = {"phase": phase, "out": str(out), "plan": str(getattr(args, "plan", "")),
               "plan_sha256": getattr(args, "plan_sha256", None), "protocol": PROTOCOL,
               "inputs": {k: {"path": v[0], "sha256": v[1]} for k, v in INPUTS.items()}}
    def timeout(_signum, _frame):
        raise TimeoutError("Whole preparation cap exceeded")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, WALL_CAP)
    try:
        request["protocol_sha256"] = sha(safe(ROOT, PROTOCOL), budget.check)
        request["preparer_sha256"] = sha(Path(__file__), budget.check)
        write(out / "started.json", {"version": VERSION, "phase": phase, "runtime": runtime(), "no_retry": True,
                                     "limits": {"wall_seconds": WALL_CAP, "output_bytes": OUTPUT_CAP}, "request": request})
        budget.progress["phase"] = phase
        result = body(out, budget, args)
        result.update(status="completed", version=VERSION, phase=phase, runtime=runtime(), no_retry=True,
                      model_calls=0, encoder_calls=0, optimizer_steps=0, float_feature_arrays_decoded=0,
                      test_contents_accessed=False, progress=dict(budget.progress), files=manifest(out, budget.check))
        result["wall_seconds"] = time.monotonic() - started
        result["wall_scope"] = "Recorded through payload manifest, before completion serialization/write/hash; whole cap remains checked through return."
        budget.storage(len(encoded(result)))
        write(out / "completed.json", result)
        budget.storage()
        return sha(out / "completed.json", budget.check)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if (out / "completed.json").exists():
            try:
                (out / "completed.json").rename(out / "late-completion.json")
            except BaseException as preservation_error:  # noqa: BLE001 - retain the original failure
                if callable(getattr(error, "add_note", None)):
                    error.add_note("Completion demotion failed: " + repr(preservation_error))
        try:
            write(out / "failed.json", {"status": "failed", "phase": phase, "version": VERSION, "no_retry": True,
                                        "error_type": type(error).__name__, "error": str(error),
                                        "wall_seconds": time.monotonic()-started, "progress": budget.progress, "request": request})
        except BaseException as preservation_error:  # noqa: BLE001 - retain the original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt preservation failed: " + repr(preservation_error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--plan-sha256", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = attempt(args.phase, args.out, freeze_body if args.phase == "freeze" else prepare_body, args)
    print(json.dumps({"status": "completed", "completed_sha256": digest}))


if __name__ == "__main__":
    main()
