"""Saved-only capacity audit. No encoder, tokenizer, model or corpus parser."""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
CAP = ROOT / "runs/dialogue-joint-v1/capacity-01"
PREP = ROOT / "runs/dialogue-joint-v1/preparation-01"
CAP_SHA = "7f8174f3afb7c81e29740e373f83224d0ca723a14d847f667e12c59aa1d779b3"
PLAN_SHA = "fd419ec9772ff614ac6bb6aa13ab2df7baa9121f709d0d5a50fe8b7299179225"
PREP_SHA = "0d6a160a9d72d424e12b69da87e0ae4d48447762e9d8e77199c30d625ab09b62"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SOURCE_NAMES = {"research/dialogue-joint-protocol.md", "scripts/prepare_dialogue_copy.py",
    "scripts/prepare_dialogue_joint.py", "scripts/prepare_sgd_state.py", "scripts/study_dialogue_memory.py",
    "src/openjev/research/dialogue_copy_features.py", "src/openjev/research/dialogue_state_data.py",
    "tests/test_dialogue_copy_features.py", "tests/test_dialogue_state_data.py", "tests/test_prepare_dialogue_joint.py"}
MODEL_NAMES = {"config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.txt"}


def require(condition, message):
    if not condition:
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
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def bind(path, expected, inventory, *, cache_symlink=False):
    require(path.is_file() and (cache_symlink or not path.is_symlink()) and sha(path) == expected, "Hash mismatch: " + str(path))
    inventory[str(path)] = expected


def payloads(folder, receipt, inventory):
    require(receipt["status"] == "completed", "Incomplete receipt")
    for name, item in receipt["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe member")
        path = folder / name
        require(path.stat().st_size == item["bytes"], "Member byte mismatch")
        bind(path, item["sha256"], inventory)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(actual == set(receipt["files"]) | {"completed.json"} and not any(p.is_symlink() for p in folder.rglob("*")),
            "Exact retained payload membership")


def profile(lengths):
    """Independent chunk sequence and padding reconstruction from saved lengths."""
    require(lengths.ndim == 1 and lengths.dtype == np.int64 and lengths.size and (lengths > 0).all(), "Token lengths")
    chunk_lengths = np.asarray([min(254, int(n) - start) + 2 for n in lengths for start in range(0, int(n), 254)], dtype=np.int64)
    full = len(chunk_lengths) // 128
    padded = int(chunk_lengths[:full * 128].reshape(full, 128).max(1).sum()) * 128 if full else 0
    tail = chunk_lengths[full * 128:]
    if len(tail):
        padded += len(tail) * int(tail.max())
    return {"unique_texts": len(lengths), "input_tokens": int(lengths.sum()), "encoder_sequences": len(chunk_lengths),
        "encoder_calls": (len(chunk_lengths) + 127) // 128, "encoder_tokens_with_special": int(chunk_lengths.sum()),
        "padded_token_slots": padded, "overlength_texts_chunked": int((lengths > 254).sum()), "truncated_tokens": 0,
        "minimum_text_tokens": int(lengths.min()), "maximum_text_tokens": int(lengths.max())}


def main():
    require(not any((OUT / n).exists() for n in ("started.json", "summary.json", "receipt.json", "failed.json")), "No overwrite or retry")
    start, inventory = time.perf_counter(), {}
    stage = "authentication"
    write(OUT / "started.json", {"status": "started", "auditor_sha256": sha(Path(__file__)),
        "capacity_completed_sha256": CAP_SHA, "preparation_completed_sha256": PREP_SHA, "plan_sha256": PLAN_SHA})
    try:
        # Hand-calculated arithmetic sanity cases, no random or model operations.
        assert profile(np.array([1, 255, 510], np.int64))["padded_token_slots"] == 1536
        assert profile(np.array([254] * 128 + [1], np.int64))["padded_token_slots"] == 32771
        bind(CAP / "completed.json", CAP_SHA, inventory)
        bind(PREP / "completed.json", PREP_SHA, inventory)
        bind(PREP / "plan.json", PLAN_SHA, inventory)
        done, prep, plan = read(CAP / "completed.json"), read(PREP / "completed.json"), read(PREP / "plan.json")
        payloads(CAP, done, inventory)
        payloads(PREP, prep, inventory)
        require(done["phase"] == "capacity" and prep["phase"] == "freeze" and prep["encoder_calls"] == 0
                and done["plan_sha256"] == prep["plan_sha256"] == PLAN_SHA, "Phase/plan identity")
        require(done["no_retry"] is True and done["test_contents_accessed"] is prep["test_contents_accessed"] is False
                and done["parameter_training_steps"] == 0 and 0 < done["wall_seconds"] <= 120, "Execution bounds")
        require(done["source_sha256"] == plan["source_sha256"] == prep["source_sha256"]
                and set(plan["source_sha256"]) == SOURCE_NAMES, "Ten-source closure")
        for name, digest in plan["source_sha256"].items():
            for file in (ROOT / name, CAP / "sources" / name, PREP / "sources" / name):
                bind(file, digest, inventory)
        for folder in (CAP, PREP):
            bind(folder / "original-encoder-source.py", plan["original_encoder_source_sha256"], inventory)
        require(read(CAP / "plan.json") == plan and done["runtime"] == plan["runtime"] and done["device"] == plan["device"] == "mps",
                "Runtime/copied plan identity")
        packet = ROOT / plan["inputs"]["packet"]["path"]
        bind(packet / "completed.json", plan["inputs"]["packet"]["completed_sha256"], inventory)
        packet_receipt = read(packet / "completed.json")
        bind(packet / "encoder-plan.json", packet_receipt["files"]["encoder-plan.json"], inventory)
        inherited = read(packet / "encoder-plan.json")
        require(done["model_files_sha256"] == plan["model_files_sha256"] == inherited["model_files_sha256"]
                and set(plan["model_files_sha256"]) == MODEL_NAMES and plan["revision"] == inherited["revision"] == REVISION,
                "Original six-file encoder identity")
        model_cache = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots" / REVISION
        for name, digest in plan["model_files_sha256"].items():
            bind(model_cache / name, digest, inventory, cache_symlink=True)
        require(done["unique_texts_encoded"] == 512 and done["full_unique_texts"] == plan["unique_texts"], "Sample coverage")
        stage = "token_and_embedding_arithmetic"
        lengths = np.load(CAP / "text-token-counts.npy", allow_pickle=False)
        require(len(lengths) == plan["unique_texts"], "Full length coverage")
        full, sample = profile(lengths), profile(lengths[:512])
        saved_profile = read(CAP / "token-profile.json")
        require({k: saved_profile[k] for k in full} == full, "Full token-profile arithmetic mismatch")
        work, progress = done["work"], done["progress"]
        for key in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "overlength_texts_chunked", "truncated_tokens"):
            require(work[key] == sample[key], "Sample work mismatch: " + key)
        require(progress["encoder_calls_attempted"] == progress["encoder_calls_returned"] == sample["encoder_calls"] == 4
                and progress["encoded_sequences"] == sample["encoder_sequences"] == 512
                and progress["encoded_texts"] == 512 and progress["encoder_work"] == work, "Recorded call prefix")
        vectors = np.load(CAP / "embeddings.npy", allow_pickle=False)
        require(vectors.dtype == np.float32 and vectors.shape == (512, 384) and np.isfinite(vectors).all(), "Sample embeddings")
        norm_error = float(np.max(np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1)))
        require(norm_error <= 2e-6, "Sample embeddings not normalized")
        stage = "projection"
        projection = done["projection"]
        require(work["encoder_seconds"] > 0 and saved_profile["tokenization_seconds"] > 0, "Timing units")
        overhead = projection["observed_nonencoding_seconds"]
        measured_fixed = done["setup_seconds"] + saved_profile["tokenization_seconds"] + work["tokenization_seconds"] + work["pool_accumulation_seconds"]
        require(measured_fixed <= overhead <= done["wall_seconds"] - work["encoder_seconds"], "Disjoint timing scopes")
        encoder_seconds = work["encoder_seconds"] / sample["padded_token_slots"] * full["padded_token_slots"]
        total_seconds = encoder_seconds + overhead
        require(math.isclose(projection["projected_encoder_seconds"], encoder_seconds, rel_tol=1e-12, abs_tol=1e-10)
                and math.isclose(projection["projected_total_seconds"], total_seconds, rel_tol=1e-12, abs_tol=1e-10), "Time projection arithmetic")
        require(plan["projected_embedding_bytes"] == plan["unique_texts"] * 384 * 4
                and plan["padded_index_bytes"] == plan["index_count"] * 8, "Array byte projection")
        source_bytes = sum(item["bytes"] for name, item in done["files"].items()
                           if name.startswith("sources/") or name in ("started.json", "original-encoder-source.py"))
        projected_bytes = plan["projected_embedding_bytes"] + plan["padded_index_bytes"] + (PREP / "index.json").stat().st_size + source_bytes + 3 * 1024**2
        require(projection["projected_cache_bytes"] == projected_bytes and projection["maximum_cache_bytes"] == 4 * 1024**3
                and projection["maximum_projected_seconds"] == 720., "Disk/admission thresholds")
        allowed = total_seconds <= 720 and projected_bytes <= 4 * 1024**3
        require(projection["encoding_permitted"] is allowed, "Admission decision mismatch")
        require(done["cache_file_bytes"] == sum(item["bytes"] for item in done["files"].values()), "Retained capacity bytes")
        for file, digest in tuple(inventory.items()):
            require(sha(Path(file)) == digest, "Input changed during audit")
        result = {"status": "passed", "measurement_valid": True, "encoding_permitted": allowed,
            "capacity_completed_sha256": CAP_SHA, "preparation_completed_sha256": PREP_SHA, "plan_sha256": PLAN_SHA,
            "sample": sample, "full_token_workload": full, "sample_embedding_shape": list(vectors.shape),
            "maximum_sample_norm_error": norm_error, "encoder_calls_attempted": 4, "encoder_calls_returned": 4,
            "capacity_wall_seconds": done["wall_seconds"], "sample_encoder_seconds": work["encoder_seconds"],
            "setup_seconds": done["setup_seconds"], "full_tokenization_seconds": saved_profile["tokenization_seconds"],
            "projected_encoder_seconds": encoder_seconds, "observed_nonencoding_seconds": overhead,
            "projected_total_seconds": total_seconds, "maximum_projected_seconds": 720.,
            "projected_cache_bytes": projected_bytes, "maximum_cache_bytes": 4 * 1024**3,
            "time_admission_passed": total_seconds <= 720, "disk_admission_passed": projected_bytes <= 4 * 1024**3,
            "preparation_sources_checked": 10, "model_files_hashed": 6, "authenticated_files": inventory,
            "new_model_calls": 0, "new_tokenizer_calls": 0, "new_corpus_parses": 0,
            "limits": ["Tokenization lengths, elapsed times and actual forward execution are source-bound producer records; this audit reconstructs their arithmetic and hashes without rerunning them.",
                "The deterministic cold first-512 sample does not measure full-workload encoder time or establish general hardware/architecture inefficiency.",
                "The 720-second forecast threshold is a prospective admission rule, distinct from the 900-second full-run cap. Failed admission authorizes no full encoding or fitting.",
                "MPS memory is a sampled live allocation maximum; RSS is process-lifetime high water. Neither is a task-isolated allocator peak."]}
        write(OUT / "summary.json", result)
        write(OUT / "receipt.json", {"status": "passed", "measurement_valid": True, "encoding_permitted": allowed,
            "auditor_sha256": sha(Path(__file__)), "summary_sha256": sha(OUT / "summary.json"),
            "started_sha256": sha(OUT / "started.json"), "wall_seconds": time.perf_counter() - start,
            "capacity_completed_sha256": CAP_SHA, "new_model_calls": 0, "new_tokenizer_calls": 0})
        print(json.dumps({"status": "passed", "measurement_valid": True, "encoding_permitted": allowed,
                          "projected_seconds": total_seconds, "projected_bytes": projected_bytes}))
    except BaseException as error:
        write(OUT / "failed.json", {"status": "failed", "stage": stage, "error": str(error),
            "traceback": traceback.format_exc(), "authenticated_files": inventory, "wall_seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    main()
