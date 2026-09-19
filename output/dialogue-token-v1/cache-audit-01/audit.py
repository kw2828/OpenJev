"""Independent saved-token audit. No encoder, preparer, Torch or corpus parser."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
VERSION = "dialogue-token-minilm-v1"
CAP_BYTES = 4 * 1024**3
TOLERANCE = 2e-5
PINS = {
    "packet": "e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308",
    "lexical": "2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54",
    "data": "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12",
}
SOURCES = {
    "scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py",
    "scripts/prepare_dialogue_tokens.py", "tests/test_prepare_dialogue_tokens.py",
    "scripts/study_dialogue_memory.py", "scripts/prepare_sgd_state.py",
    "src/openjev/research/dialogue_state_data.py", "tests/test_dialogue_state_data.py",
    "scripts/prepare_dialogue_copy.py", "src/openjev/research/dialogue_copy_features.py",
    "tests/test_dialogue_copy_features.py", "research/dialogue-token-protocol.md",
}
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.json", "vocab.txt"}
TOKEN_FILES = {"tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "Member path")
    path = Path(root) / name
    require(path.resolve().is_relative_to(Path(root).resolve()), "Member escapes root")
    return path


def bind(path, digest, inventory, *, model=False):
    path = Path(path)
    require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Digest schema")
    require(path.is_file() and (model or not path.is_symlink()) and sha(path) == digest, "Changed file: " + str(path))
    inventory[str(path.resolve())] = digest


def tree(folder, pin, inventory, expected=None):
    bind(folder / "completed.json", pin, inventory)
    receipt = read(folder / "completed.json")
    require(receipt["status"] == "completed" and type(receipt["files"]) is dict, "Incomplete receipt")
    if expected is not None:
        paths = list(folder.rglob("*"))
        require(not any(p.is_symlink() for p in paths) and set(receipt["files"]) == expected
                and {p.relative_to(folder).as_posix() for p in paths if p.is_file()} == expected | {"completed.json"}, "Exact phase membership")
    for name, item in receipt["files"].items():
        path = safe(folder, name)
        digest = item if isinstance(item, str) else item["sha256"]
        if isinstance(item, dict):
            require(type(item["bytes"]) is int and path.stat().st_size == item["bytes"], "Payload bytes")
        bind(path, digest, inventory)
    return receipt


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), name)
    return value


def geometry(lengths):
    require(lengths.dtype == np.int64 and lengths.ndim == 1 and len(lengths) and (lengths > 0).all(), "Context lengths")
    chunks, offsets, chunk_offsets = [], [0], [0]
    for n in map(int, lengths):
        whole, tail = divmod(n, 254)
        parts = [254] * whole + ([tail] if tail else [])
        chunks.extend(parts)
        offsets.append(offsets[-1] + n + 2*len(parts))
        chunk_offsets.append(len(chunks))
    sizes = [n+2 for n in chunks]
    profile = {"unique_texts": len(lengths), "input_tokens": sum(map(int, lengths)),
        "encoder_sequences": len(chunks), "encoder_calls": (len(chunks)+127)//128,
        "encoder_tokens_with_special": offsets[-1],
        "padded_token_slots": sum(max(sizes[i:i+128])*len(sizes[i:i+128]) for i in range(0, len(sizes), 128)),
        "overlength_texts_chunked": int((lengths > 254).sum()), "truncated_tokens": 0,
        "minimum_text_tokens": int(lengths.min()), "maximum_text_tokens": int(lengths.max())}
    return profile, np.array(offsets, np.int64), np.array(chunk_offsets, np.int64), np.array(chunks, np.int64)


def phase_arrays(folder, receipt, lengths, original_ids, features):
    profile, offsets, chunk_offsets, chunks = geometry(lengths)
    for name, expected in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        actual = np.load(folder / name, mmap_mode="r", allow_pickle=False)
        require(actual.dtype == np.int64 and np.array_equal(actual, expected), "Chunk geometry: " + name)
    tokens = np.load(folder / "tokens.npy", mmap_mode="r", allow_pickle=False)
    priors = np.load(folder / "priors.npy", mmap_mode="r", allow_pickle=False)
    require(tokens.dtype == priors.dtype == np.float32 and tokens.shape == (int(offsets[-1]), 384)
            and priors.shape == (int(offsets[-1]),), "Raw token/prior shape")
    maximum = 0.
    for i, (lo, hi) in enumerate(pairwise(offsets)):
        states, weights = tokens[lo:hi], priors[lo:hi]
        require(np.isfinite(states).all() and np.isfinite(weights).all() and (weights > 0).all(), "Nonfinite token/prior")
        cursor = 0
        for n in chunks[chunk_offsets[i]:chunk_offsets[i+1]]:
            expected = np.float32(int(n)/(int(lengths[i])*(int(n)+2)))
            require(np.all(weights[cursor:cursor+n+2] == expected), "Chunk-weighted prior")
            cursor += int(n)+2
        require(cursor == len(weights) and abs(weights.sum(dtype=np.float64)-1) <= 2e-6, "Context prior mass")
        pooled = np.sum(states * weights[:, None], axis=0, dtype=np.float32)
        norm = np.linalg.norm(pooled)
        require(np.isfinite(norm) and norm > 1e-12, "Invalid pooled vector")
        pooled = pooled / norm
        target = features[int(original_ids[i])]
        require(np.isfinite(target).all() and abs(np.linalg.norm(target)-1) <= 2e-6, "Original sentence vector")
        error = float(np.max(np.abs(pooled-target)))
        require(error <= TOLERANCE, "Sentence-vector equivalence")
        maximum = max(maximum, error)
    eq = receipt["pooling_equivalence"]
    require(eq == {"contexts_checked": len(lengths), "max_abs_error": maximum, "tolerance": TOLERANCE}, "Recorded equivalence maximum")
    work, progress = receipt["work"], receipt["progress"]
    for key in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "truncated_tokens"):
        require(work[key] == profile[key], "Work count: " + key)
    require(work["retained_token_rows"] == profile["encoder_tokens_with_special"]
            and work["overlength_contexts_chunked"] == profile["overlength_texts_chunked"]
            and progress["encoded_texts"] == len(lengths) and progress["encoded_sequences"] == profile["encoder_sequences"]
            and progress["encoder_calls_attempted"] == progress["encoder_calls_returned"] == profile["encoder_calls"], "Execution witness counts")
    require(progress["work"] == work, "Terminal work witness differs")
    neural = finite(work["forward_transfer_seconds"], "Neural-transfer time", True)
    retention = finite(work["retention_seconds"], "Retention time", True)
    tokenization = finite(work["tokenization_seconds"], "Tokenization time")
    setup = finite(receipt["setup_seconds"], "Setup time")
    require(neural+retention+tokenization+setup <= receipt["wall_seconds"], "Disjoint time witnesses exceed wall")
    return {"contexts": len(lengths), "profile": profile, "pooling_equivalence": eq,
            "recorded_wall_seconds": receipt["wall_seconds"], "independently_recomputed_token_rows": int(offsets[-1])}


def audit(args):
    start = time.perf_counter()
    root, capacity, cache = args.root.resolve(), args.capacity.resolve(), args.cache.resolve()
    inventory = {}
    common = TOKEN_FILES | {"plan.json", "started.json", "original-encoder-source.py", "token-profile.json", "text-token-counts.npy"} | {"sources/"+p for p in SOURCES}
    cap = tree(capacity, args.capacity_sha256, inventory, common)
    full = tree(cache, args.cache_sha256, inventory, common | {"index.json", "context-indices.npy"})
    plan = read(cache / "plan.json")
    bind(cache / "plan.json", full["plan_sha256"], inventory)
    bind(capacity / "plan.json", full["plan_sha256"], inventory)
    require(set(plan["source_sha256"]) == SOURCES and set(plan["model_files_sha256"]) == MODEL_FILES, "Source/model closure")
    require(plan["version"] == VERSION and plan["encoder"] == ENCODER and plan["revision"] == REVISION
            and plan["device"] == "mps" and plan["dtype"] == "float32" and plan["width"] == 384
            and plan["chunk_tokens"] == 254 and plan["batch_size"] == 128 and plan["capacity_unique_contexts"] == 512
            and plan["capacity_seconds"] == 120 and plan["encoding_seconds"] == 900 and plan["admission_seconds"] == 720
            and plan["disk_cap_bytes"] == CAP_BYTES and plan["pooling_max_abs_tolerance"] == TOLERANCE, "Frozen recipe")
    require(plan["protocol_path"] == "research/dialogue-token-protocol.md"
            and plan["protocol_sha256"] == plan["source_sha256"][plan["protocol_path"]], "Protocol identity")
    for name, pin in plan["source_sha256"].items():
        for source in (safe(root, name), safe(capacity / "sources", name), safe(cache / "sources", name)):
            bind(source, pin, inventory)
    for name, pin in plan["model_files_sha256"].items():
        bind(args.encoder_snapshot / name, pin, inventory, model=True)
    require(set(plan["inputs"]) == set(PINS), "Parent membership")
    parents = {}
    for name, pin in PINS.items():
        require(plan["inputs"][name]["completed_sha256"] == pin, "Original input pin")
        folder = safe(root, plan["inputs"][name]["path"])
        parents[name] = tree(folder, pin, inventory)
    require(parents["data"]["test_contents_accessed"] is False
            and parents["lexical"]["packet_completed_sha256"] == PINS["packet"]
            and parents["lexical"]["data_completed_sha256"] == PINS["data"], "Parent scope/lineage")
    for mapping in (parents["data"]["implementation_sha256"], parents["lexical"]["source_sha256"]):
        require(all(plan["source_sha256"].get(k) == v for k, v in mapping.items()), "Preparation implementation lineage")
    packet = safe(root, plan["inputs"]["packet"]["path"])
    encoder_plan = read(packet / "encoder-plan.json")
    require(encoder_plan["encoder"] == ENCODER and encoder_plan["revision"] == REVISION
            and encoder_plan["model_files_sha256"] == plan["model_files_sha256"]
            and encoder_plan["source_sha256"] == plan["original_encoder_source_sha256"]
            and encoder_plan["input_sha256"]["completed.json"] == PINS["data"], "Original encoder identity")
    for folder, receipt, phase, seconds in ((capacity, cap, "capacity", 120), (cache, full, "encode", 900)):
        require(receipt["phase"] == phase and receipt["no_retry"] is True and receipt["test_contents_accessed"] is False
                and receipt["parameter_training_steps"] == 0 and 0 < finite(receipt["wall_seconds"], "Phase wall") <= seconds, "Phase status/scope/cap")
        for key in ("version", "runtime", "device", "encoder", "revision", "source_sha256", "model_files_sha256", "counts"):
            require(receipt[key] == plan[key], "Receipt identity: " + key)
        require(receipt["plan_sha256"] == full["plan_sha256"] and receipt["full_unique_contexts"] == plan["unique_contexts"], "Context/plan identity")
        require(all(receipt[n+"_completed_sha256"] == pin for n, pin in PINS.items()), "Receipt input identity")
        bind(folder / "original-encoder-source.py", plan["original_encoder_source_sha256"], inventory)
        require(receipt["cache_file_bytes"] == sum(v["bytes"] for v in receipt["files"].values())
                and receipt["cache_file_bytes"]+(folder / "completed.json").stat().st_size <= CAP_BYTES, "Actual disk bytes/cap")
        started = read(folder / "started.json")
        require(started["status"] == "started" and started["phase"] == phase and started["cap_seconds"] == seconds
                and started["disk_cap_bytes"] == CAP_BYTES and started["no_retry"] is True
                and started["source_sha256"] == plan["source_sha256"]["scripts/prepare_dialogue_joint.py"]
                and started["request"]["plan_sha256"] == full["plan_sha256"], "Started phase identity")
    require(cap["capacity_path"] is None and cap["capacity_completed_sha256"] is None
            and full["capacity_completed_sha256"] == args.capacity_sha256
            and safe(root, full["capacity_path"]).resolve() == capacity, "External capacity binding")
    for name in ("token-profile.json", "text-token-counts.npy"):
        bind(cache / name, sha(capacity / name), inventory)
    lengths = np.load(cache / "text-token-counts.npy", allow_pickle=False)
    profile, offsets, _, chunks = geometry(lengths)
    recorded_profile = read(cache / "token-profile.json")
    require(all(recorded_profile[k] == v for k, v in profile.items()) and len(lengths) == plan["unique_contexts"] >= 512, "Full profile arithmetic")
    require(cap["unique_contexts_encoded"] == 512 and full["unique_contexts_encoded"] == len(lengths), "Fixed sample/full coverage")
    index = read(cache / "index.json")
    for name in ("index.json", "context-indices.npy"):
        bind(cache / name, plan["layout_files"][name], inventory)
    require(index["version"] == VERSION and index["template"] == plan["template"] and index["width"] == 384
            and index["unique_contexts"] == len(lengths) and index["counts"] == plan["counts"], "Context index identity")
    features = np.load(packet / "features.npy", mmap_mode="r", allow_pickle=False)
    ids = index["original_feature_indices"]
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == 384 and len(ids) == len(lengths)
            and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < len(features) for i in ids), "Original feature indexing")
    context_ids = np.load(cache / "context-indices.npy", allow_pickle=False)
    require(context_ids.dtype == np.int64 and context_ids.shape == (index["context_occurrences"],)
            and ((context_ids >= 0) & (context_ids < len(ids))).all(), "Public context index support")
    lexical_index = read(safe(root, plan["inputs"]["lexical"]["path"]) / "index.json")
    cursor = 0
    for split in ("train", "dev"):
        entries, original = index["cohorts"][split], lexical_index["cohorts"][split]
        require(len(entries) == len(original), "Dialogue coverage")
        turns = questions = 0
        for entry, prior in zip(entries, original, strict=True):
            t = prior["shape"][0]
            require(entry == {"id": prior["id"], "query_ids": prior["query_ids"], "offset": cursor, "shape": [t]}, "Public full-turn layout")
            cursor += t
            turns += t
            questions += t*len(prior["query_ids"])
        require(index["counts"][split] == {"dialogues": len(entries), "public_turns": turns, "public_question_steps": questions}, "Public layout counts")
    require(cursor == len(context_ids) == plan["context_occurrences"], "Public context coverage")
    sample_result = phase_arrays(capacity, cap, lengths[:512], ids[:512], features)
    full_result = phase_arrays(cache, full, lengths, ids, features)
    projected_bytes = (int(offsets[-1])*1540 + (len(lengths)+1)*16 + len(chunks)*8
        + sum(v["bytes"] for n, v in cap["files"].items() if n not in TOKEN_FILES)
        + sum((cache / n).stat().st_size for n in ("index.json", "context-indices.npy")) + 8*1024**2)
    p, w = cap["projection"], cap["work"]
    neural = w["forward_transfer_seconds"] / w["padded_token_slots"] * profile["padded_token_slots"]
    retention = w["retention_seconds"] / w["retained_token_rows"] * profile["encoder_tokens_with_special"]
    overhead = finite(p["observed_other_seconds"], "Remaining capacity overhead")
    require(w["forward_transfer_seconds"]+w["retention_seconds"]+overhead <= cap["wall_seconds"], "Projection buckets exceed measured wall")
    require(p["projected_forward_transfer_seconds"] == neural and p["projected_retention_seconds"] == retention
            and p["projected_total_seconds"] == neural+retention+overhead and p["maximum_projected_seconds"] == 720.
            and p["projected_cache_bytes"] == projected_bytes and p["maximum_cache_bytes"] == CAP_BYTES
            and p["encoding_permitted"] is True and neural+retention+overhead <= 720 and projected_bytes <= CAP_BYTES, "Admission arithmetic")
    for path, pin in inventory.items():
        require(sha(path) == pin, "Input changed during audit")
    summary = {"status": "completed", "study": "dialogue-token-v1", "capacity_completed_sha256": args.capacity_sha256,
        "cache_completed_sha256": args.cache_sha256, "plan_sha256": full["plan_sha256"], "capacity": sample_result,
        "full": full_result, "projection": p, "source_files": len(SOURCES), "encoder_files": len(MODEL_FILES),
        "raw_corpus_parsed": False, "encoder_calls": 0, "model_calls": 0, "first_512_identity": "Shared frozen layout IDs[0:512] and exact prefix length/chunk schedule, with independent pooled-vector checks",
        "scope": "Hashes, layouts, all saved token/prior geometry and pooled means are independently recomputed. Encoder execution, tokenizer/text correspondence, raw-text assembly and measured timing are authenticated source-bound witnesses, not replayed or externally attested.",
        "wall_seconds": time.perf_counter()-start}
    write(args.out / "summary.json", summary)
    write(args.out / "receipt.json", {"status": "completed", "auditor_sha256": sha(__file__), "input_sha256": inventory,
        "files": {"summary.json": {"sha256": sha(args.out / "summary.json"), "bytes": (args.out / "summary.json").stat().st_size}},
        "capacity_completed_sha256": args.capacity_sha256, "cache_completed_sha256": args.cache_sha256,
        "no_neural_replay": True, "automatic_retry": False, "wall_seconds": time.perf_counter()-start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capacity", "cache", "out", "encoder-snapshot"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("capacity-sha256", "cache-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        audit(args)
    except BaseException as error:
        try:
            write(args.out / "failed.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                "auditor_sha256": sha(__file__), "automatic_retry": False})
        except BaseException as secondary:  # noqa: BLE001 - original error remains primary
            error.add_note("Failure receipt: " + repr(secondary))
        raise


if __name__ == "__main__":
    main()
