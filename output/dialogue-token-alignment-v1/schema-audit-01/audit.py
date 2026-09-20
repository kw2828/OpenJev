"""Independent saved schema-cache audit. No encoder, preparer or model imports."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import signal
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PINS = {"packet": "e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308",
        "data": "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
        "prepared": "960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3",
        "study": "65bd57084dbc7f7b99b782e140bae2e18f25b9e237e89a0132cead5fbfe2a095"}
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.txt"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts, "Relative input path")
    result = root / name
    require(result.resolve().is_relative_to(root.resolve()), "Input escaped root")
    return result


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def reconstruct_layout(packet, catalog, rows, study):
    fit, evaluation = study["fit_row_indices"], study["evaluation_row_indices"]
    require(len(fit) == 29211 and len(evaluation) == 13599 and not set(fit) & set(evaluation), "Original split sizes")
    wanted = set(fit) | set(evaluation)
    found, qids = set(), set()
    for r in rows:
        if r["row_index"] in wanted:
            require(r["row_index"] not in found and r["split"] == "train" and r["admission"] == "admitted", "Exact TRAIN scope")
            found.add(r["row_index"]); qids.add(r["query_index"])
        else:
            require(not (r["split"] == "train" and r["admission"] == "admitted"), "TRAIN row omission")
    require(found == wanted and len(qids) == 53, "Complete schema selection")
    catalogs = {q["query_id"]: q for q in catalog["train"]}
    texts, original_ids, mapping, entries = [], [], {}, []
    def add(text, original):
        if text not in mapping:
            mapping[text] = len(texts); texts.append(text); original_ids.append(original)
        result = mapping[text]
        require(original_ids[result] == original, "Exact text/source index identity")
        return result
    for qi in sorted(qids):
        query = packet["queries"][qi]; c = catalogs[query["id"]]
        require(query["split"] == "train" and (query["service"], query["slot"]) == (c["service"], c["slot"]), "TRAIN schema identity")
        prefix = "Service: " + c["service_description"] + "\nSlot: " + c["slot_description"]
        require(c["query_text"] == prefix, "Original exact query string")
        ids = [x["id"] for x in c["candidates"]]
        require(ids == query["candidate_ids"] and ids[:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"]
                and [x["value"] for x in c["candidates"]] == query["candidate_values"], "All candidate identities")
        expected = [prefix + "\nValue: NOT_MENTIONED (no constraint stated)", prefix + "\nValue: DONTCARE (no preference)"]
        expected.extend(prefix + "\nValue: " + x["value"] for x in c["candidates"][2:])
        require(expected == [x["text"] for x in c["candidates"]], "Full candidate strings retain schema")
        query_token = add(prefix, query["text"])
        candidate_tokens = [add(t, i) for t, i in zip(expected, query["candidates"], strict=True)]
        entries.append({"query_index": qi, "query_id": query["id"], "service": query["service"], "slot": query["slot"],
                        "query_token_id": query_token, "candidate_token_ids": candidate_tokens,
                        "query_feature_index": query["text"], "candidate_feature_indices": query["candidates"], "candidate_ids": ids})
    require(len(texts) == 360 and sum(len(q["candidate_ids"]) for q in entries) == 307, "All 360 strings and 307 candidates")
    return texts, original_ids, entries


def audit(args, inputs):
    cache = Path(args.cache).resolve()
    def bind(path, expected, size=None):
        path = Path(path)
        require(path.is_file() and sha(path) == expected and (size is None or path.stat().st_size == size), "Digest/size: " + str(path))
        inputs[str(path)] = {"sha256": expected, "bytes": path.stat().st_size}
    bind(cache / "completed.json", args.completed_sha256)
    done = read(cache / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "encode" and done["no_retry"] is True
            and done["parameter_training_steps"] == 0 and done["test_contents_accessed"] is False, "Completed nontraining cache")
    bind(cache / "plan.json", done["plan_sha256"])
    plan = read(cache / "plan.json")
    require(plan["version"] == done["version"] == "dialogue-schema-token-v1" and plan["revision"] == REVISION
            and plan["encoder"] == "sentence-transformers/all-MiniLM-L6-v2"
            and plan["device"] == "mps" and plan["dtype"] == "float32" and plan["width"] == 384
            and plan["chunk_tokens"] == 254 and plan["batch_size"] == 128
            and plan["wall_cap_seconds"] == 120 and plan["disk_cap_bytes"] == 256 * 1024**2
            and plan["pooling_max_abs_tolerance"] == 2e-5, "Frozen numerical recipe")
    require(done["inputs"] == plan["inputs"] and done["source_sha256"] == plan["source_sha256"]
            and done["runtime"] == plan["runtime"] and done["model_files_sha256"] == plan["model_files_sha256"], "Plan/completion identity")
    sources = plan["source_sha256"]
    require(len(sources) == 33, "33 frozen source members")
    expected = {"started.json", "plan.json", "original-encoder-source.py", "index.json", "token-profile.json", "text-token-counts.npy",
                "tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy"} | {"sources/" + n for n in sources}
    require(set(done["files"]) == expected and {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file()} == expected | {"completed.json"}, "Exact cache closure")
    for name, item in done["files"].items(): bind(safe(cache, name), item["sha256"], item["bytes"])
    for name, digest in sources.items():
        bind(safe(ROOT, name), digest); bind(safe(cache / "sources", name), digest)
    bind(cache / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    require(set(plan["model_files_sha256"]) == MODEL_FILES, "Six model/tokenizer files")
    for name, digest in plan["model_files_sha256"].items(): bind(Path(args.model_snapshot) / name, digest)
    parents = {}
    for name, pin in PINS.items():
        item = plan["inputs"][name]
        require(item["completed_sha256"] == pin, "Fixed inherited input pin")
        path = safe(ROOT, item["path"])
        bind(path / "completed.json", pin)
        parents[name] = (path, read(path / "completed.json"))
    def parent_file(parent, name):
        folder, record = parents[parent]; item = record["files"][name]
        bind(folder / name, item if type(item) is str else item["sha256"], None if type(item) is str else item["bytes"])
        return folder / name
    study = read(parent_file("study", "plan.json"))
    require(study["prepared_completed_sha256"] == PINS["prepared"]
            and all(sources.get(n) == v for n, v in study["source_sha256"].items()), "Inherited split/source closure")
    packet = read(parent_file("packet", "packet.json"))
    encoder = read(parent_file("packet", "encoder-plan.json"))
    require(encoder["model_files_sha256"] == plan["model_files_sha256"] and encoder["revision"] == REVISION
            and encoder["input_sha256"]["completed.json"] == PINS["data"], "Original feature lineage")
    catalog = read(parent_file("data", "catalog.json"))
    rows_path = parent_file("prepared", "rows.jsonl")
    with rows_path.open() as stream:
        texts, source_ids, entries = reconstruct_layout(packet, catalog, (json.loads(line) for line in stream), study)
    index = read(cache / "index.json")
    require(index["queries"] == entries and index["original_feature_indices"] == source_ids
            and index["query_indices"] == [q["query_index"] for q in entries]
            and index["unique_texts"] == done["unique_texts"] == plan["unique_texts"] == 360
            and index["query_count"] == done["query_count"] == plan["query_count"] == 53
            and index["candidate_occurrences"] == done["candidate_occurrences"] == plan["candidate_occurrences"] == 307, "Exact complete schema mapping")
    serialized = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts).encode()
    require(hashlib.sha256(serialized).hexdigest() == plan["layout_files"]["texts.jsonl"], "Frozen exact text sequence")
    bind(cache / "index.json", plan["layout_files"]["index.json"])
    arrays = {name: np.load(cache / name, mmap_mode="r", allow_pickle=False) for name in
              ("text-token-counts.npy", "tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy")}
    lengths = arrays["text-token-counts.npy"]
    require(lengths.dtype == np.int64 and lengths.shape == (360,) and np.all(lengths > 0), "Positive token lengths")
    expected_offsets, chunk_offsets, chunks, expected_priors = [0], [0], [], []
    for n in map(int, lengths):
        parts = [min(254, n - start) for start in range(0, n, 254)]
        chunks.extend(parts); chunk_offsets.append(len(chunks))
        expected_offsets.append(expected_offsets[-1] + n + 2 * len(parts))
        for part in parts: expected_priors.extend([np.float32(part / (n * (part + 2)))] * (part + 2))
    for name, values in (("offsets.npy", expected_offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        require(arrays[name].dtype == np.int64 and np.array_equal(arrays[name], np.asarray(values, np.int64)), "Exact integer chunk geometry")
    raw, priors = arrays["tokens.npy"], arrays["priors.npy"]
    require(raw.dtype == priors.dtype == np.float32 and raw.shape == (expected_offsets[-1], 384)
            and priors.shape == (expected_offsets[-1],) and np.isfinite(raw).all()
            and np.array_equal(priors, np.asarray(expected_priors, np.float32)), "Raw tokens and exact priors")
    sizes = [n + 2 for n in chunks]
    profile = {"unique_texts": 360, "input_tokens": sum(map(int, lengths)), "encoder_sequences": len(chunks),
               "encoder_calls": math.ceil(len(chunks) / 128), "encoder_tokens_with_special": sum(sizes),
               "padded_token_slots": sum(len(sizes[i:i + 128]) * max(sizes[i:i + 128]) for i in range(0, len(sizes), 128)),
               "overlength_texts_chunked": sum(int(n) > 254 for n in lengths), "truncated_tokens": 0,
               "minimum_text_tokens": int(min(lengths)), "maximum_text_tokens": int(max(lengths))}
    saved_profile = read(cache / "token-profile.json")
    require(all(saved_profile[k] == v for k, v in profile.items()), "Full token profile")
    progress, work = done["progress"], done["work"]
    require(progress["encoder_calls_attempted"] == progress["encoder_calls_returned"] == profile["encoder_calls"]
            and progress["encoded_sequences"] == len(chunks) and progress["encoded_texts"] == 360, "Recorded encoder coverage")
    require(all(work[k] == profile[k] for k in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "truncated_tokens"))
            and work["retained_token_rows"] == expected_offsets[-1] and work["overlength_contexts_chunked"] == profile["overlength_texts_chunked"], "Recorded work geometry")
    vectors = np.load(parent_file("packet", "features.npy"), mmap_mode="r", allow_pickle=False)
    require(vectors.dtype == np.float32 and vectors.ndim == 2 and vectors.shape[1] == 384, "Original pooled vector shape")
    maximum32 = maximum64 = prior_sum_error = 0.
    for i, (a, b) in enumerate(itertools.pairwise(expected_offsets)):
        pooled32 = np.sum(raw[a:b] * priors[a:b, None], axis=0, dtype=np.float32)
        norm32 = float(np.linalg.norm(pooled32))
        pooled64 = np.sum(raw[a:b].astype(np.float64) * priors[a:b, None].astype(np.float64), axis=0)
        norm64 = float(np.linalg.norm(pooled64))
        require(math.isfinite(norm32) and norm32 > 1e-12 and math.isfinite(norm64) and norm64 > 1e-12, "Finite pooled norms")
        maximum32 = max(maximum32, float(np.abs(pooled32 / norm32 - vectors[source_ids[i]]).max()))
        maximum64 = max(maximum64, float(np.abs(pooled64 / norm64 - vectors[source_ids[i]]).max()))
        prior_sum_error = max(prior_sum_error, abs(math.fsum(map(float, priors[a:b])) - 1))
    require(maximum32 <= 2e-5 and maximum64 <= 2e-5 and prior_sum_error <= 2e-6, "All saved pooled parity and prior sums")
    eq = done["pooling_equivalence"]
    require(eq["contexts_checked"] == 360 and eq["tolerance"] == 2e-5 and eq["max_abs_error"] == maximum32, "Exact producer parity witness")
    actual_bytes = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    require(done["payload_bytes"] == sum(v["bytes"] for v in done["files"].values())
            and actual_bytes <= 256 * 1024**2 and 0 < done["wall_seconds"] <= 120, "Recorded time and actual disk caps")
    return {"status": "completed", "all_scoped_checks_passed": True, "files": len(expected) + 1, "sources": len(sources),
            "queries": 53, "candidate_occurrences": 307, "unique_strings": 360, "token_profile": profile,
            "maximum_float32_pooling_error": maximum32, "maximum_float64_pooling_error": maximum64,
            "maximum_prior_sum_error": prior_sum_error, "actual_cache_bytes": actual_bytes,
            "recorded_encoder_wall_seconds": done["wall_seconds"], "recorded_encoder_calls": profile["encoder_calls"],
            "recorded_process_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"],
            "scope": "Independent stdlib/NumPy reconstruction of exact TRAIN schema text/index mappings, all candidates, token chunk geometry/priors and every pooled-vector parity. No preparer/model imports or encoder calls. Tokenizer execution, raw transformer states and recorded wall/RSS are authenticated producer witnesses, not rerun. Existing dataset JSON records are decoded, but labels do not select schema strings. Official DEV/TEST is not encoded. Existing packet JSON contains text/annotations but only schema fields are used; no dialogue text is interpreted and no weights are deserialized."}


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); source = sha(__file__); inputs = {}
    def timeout(_signal, _frame): raise TimeoutError("Independent cache audit 60-second cap")
    previous = None
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing audit timer")
        previous = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, 60)
        result = audit(args, inputs)
        for path, item in inputs.items(): require(sha(path) == item["sha256"], "End input stability")
        require(sha(__file__) == source, "Auditor source stability")
        write(out / "summary.json", result)
        write(out / "receipt.json", {"status": "completed", "source_sha256": source, "cache_completed_sha256": args.completed_sha256,
              "inputs": inputs, "files": {"summary.json": {"sha256": sha(out / "summary.json"), "bytes": (out / "summary.json").stat().st_size}},
              "wall_seconds": time.monotonic() - start, "audit_wall_cap_seconds": 60, "model_calls": 0, "encoder_calls": 0,
              "scope": result["scope"]})
        print(json.dumps({"status": "completed", "source_sha256": source, "summary_sha256": sha(out / "summary.json"),
                          "receipt_sha256": sha(out / "receipt.json")}))
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error": repr(error), "source_sha256": source,
                  "cache_completed_sha256": args.completed_sha256, "wall_seconds": time.monotonic() - start, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "completed-sha256", "out", "model-snapshot"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
