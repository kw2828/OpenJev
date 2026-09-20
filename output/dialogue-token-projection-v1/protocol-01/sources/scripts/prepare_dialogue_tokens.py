"""Frozen shared-context token cache. All corpus/encoder work is explicit CLI work.

The prior joint-encoding helper supplies authenticated I/O and cap primitives;
its exact bytes are pinned. It is never monkeypatched or used to encode joint
candidate strings here. Downstream cache loading imports no Torch/Transformers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/prepare_dialogue_joint.py"
HELPER_SHA = "d57f9f647203606ce047b7dfbf5f42396adb829ca029adc39601ef385e9430d3"
_raw = (ROOT / HELPER).read_bytes()
if hashlib.sha256(_raw).hexdigest() != HELPER_SHA:
    raise ValueError("Immutable preparation helper differs")
old = types.ModuleType("pinned_joint_preparation_primitives")
old.__file__ = str(ROOT / HELPER)
exec(compile(_raw, old.__file__, "exec"), old.__dict__)  # noqa: S102 - exact authenticated local bytes
require, sha, read, write, bind, safe = old.require, old.sha, old.read, old.write, old.bind, old.safe
VERSION = "dialogue-token-minilm-v1"
TEMPLATE = "'System: ' + previous_system_text + '\\nUser: ' + current_user_text"
SOURCES = (HELPER, "tests/test_prepare_dialogue_joint.py", "scripts/prepare_dialogue_tokens.py", "tests/test_prepare_dialogue_tokens.py")
TOKEN_FILES = ("tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy")
TOLERANCE = 2e-5
CAP_BYTES = 4 * 1024**3


def build_contexts(packet, lexical_index, public):
    """Only public turn strings and supplied query layout; no gold/time/bin read."""
    require(set(packet["cohorts"]) == set(lexical_index["cohorts"]) == set(public) == {"train", "dev"}, "Only train/dev")
    texts, lookup, original_ids, inverse, indices, cohorts, counts = [], {}, [], {}, [], {}, {}
    for split in ("train", "dev"):
        ds, entries = packet["cohorts"][split], lexical_index["cohorts"][split]
        require(len(ds) == len(entries) and len({d["id"] for d in ds}) == len(ds)
                and set(public[split]) == {d["id"] for d in ds}, "Public dialogue closure")
        cohorts[split] = []
        counts[split] = {"dialogues": len(ds), "public_turns": 0, "public_question_steps": 0}
        for d, layout in zip(ds, entries, strict=True):
            qids = layout["query_ids"]
            require(layout["id"] == d["id"] and qids and qids == sorted(set(qids))
                    and all(type(q) is int and 0 <= q < len(packet["queries"]) and packet["queries"][q]["split"] == split for q in qids),
                    "Supplied public question layout")
            raw = public[split][d["id"]]
            require(len(raw["user_turns"]) == len(d["turns"]) == len(d["user_text"]) and d["turns"], "Public turn count")
            require(layout["shape"][:2] == [len(d["turns"]), len(qids)] and layout["shape"][-1] == 10, "Lexical turn layout")
            entry = {"id": d["id"], "query_ids": qids, "offset": len(indices), "shape": [len(d["turns"])]}
            for t, step in enumerate(raw["user_turns"]):
                ti, si = step["turn_index"], step["previous_system_turn_index"]
                require(type(ti) is int and 0 <= ti < len(raw["turns"]) and raw["turns"][ti]["speaker"] == "USER", "USER chronology")
                require(si is None or (type(si) is int and 0 <= si < ti and raw["turns"][si]["speaker"] == "SYSTEM"), "SYSTEM chronology")
                system = "" if si is None else raw["turns"][si]["utterance"]
                user = raw["turns"][ti]["utterance"]
                require(type(system) is str and type(user) is str and user == d["user_text"][t], "Original public text differs")
                text, source_id = "System: " + system + "\nUser: " + user, d["turns"][t]
                require(type(source_id) is int and source_id >= 0, "Original sentence index")
                require(inverse.setdefault(source_id, text) == text, "One source index maps to different contexts")
                if text not in lookup:
                    lookup[text] = len(texts)
                    texts.append(text)
                    original_ids.append(source_id)
                require(original_ids[lookup[text]] == source_id, "Identical text has different original sentence indices")
                indices.append(lookup[text])
            cohorts[split].append(entry)
            counts[split]["public_turns"] += len(d["turns"])
            counts[split]["public_question_steps"] += len(d["turns"]) * len(qids)
    require(texts, "No public contexts")
    return texts, np.asarray(indices, np.int64), {"version": VERSION, "template": TEMPLATE, "width": 384,
        "unique_contexts": len(texts), "context_occurrences": len(indices), "original_feature_indices": original_ids,
        "cohorts": cohorts, "counts": counts, "token_mask": "implicit: every stored token is valid; runner pads with a false mask"}


def chunks_for_lengths(lengths):
    require(lengths.dtype == np.int64 and lengths.ndim == 1 and len(lengths) and (lengths > 0).all(), "Token-count vector")
    chunks, chunk_offsets, offsets = [], [0], [0]
    for n in lengths:
        local = [min(254, int(n) - i) for i in range(0, int(n), 254)]
        chunks.extend(local)
        chunk_offsets.append(len(chunks))
        offsets.append(offsets[-1] + sum(local) + 2 * len(local))
    return np.asarray(offsets, np.int64), np.asarray(chunk_offsets, np.int64), np.asarray(chunks, np.int64)


def projection(work, full_profile, overhead, cache_bytes):
    require(all(math.isfinite(work[k]) and work[k] > 0 for k in ("forward_transfer_seconds", "retention_seconds"))
            and work["padded_token_slots"] > 0 and work["retained_token_rows"] > 0
            and math.isfinite(overhead) and overhead >= 0 and type(cache_bytes) is int and cache_bytes > 0, "Invalid capacity units")
    encoder = work["forward_transfer_seconds"] / work["padded_token_slots"] * full_profile["padded_token_slots"]
    retention = work["retention_seconds"] / work["retained_token_rows"] * full_profile["encoder_tokens_with_special"]
    total = encoder + retention + overhead
    return {"projected_forward_transfer_seconds": encoder, "projected_retention_seconds": retention,
        "observed_other_seconds": overhead, "projected_total_seconds": total, "maximum_projected_seconds": 720.,
        "projected_cache_bytes": cache_bytes, "maximum_cache_bytes": CAP_BYTES,
        "encoding_permitted": total <= 720. and cache_bytes <= CAP_BYTES,
        "formula": "forward+CPU-transfer scaled by padded tokens; retention+flush+parity+payloadhash scaled by valid token rows; plus remaining observed overhead",
        "scope": "Deterministic cold first512 extrapolation, not measured full latency; flush is not durable-disk timing"}


def source_map(parents, protocol, protocol_sha):
    mapping = {name: sha(ROOT / name) for name in SOURCES}
    mapping["scripts/study_dialogue_memory.py"] = sha(ROOT / "scripts/study_dialogue_memory.py")
    for inherited in (parents["data"]["implementation_sha256"], parents["lexical"]["source_sha256"]):
        for name, digest in inherited.items():
            require(name not in mapping or mapping[name] == digest, "Conflicting source hash")
            mapping[name] = digest
    require(mapping[HELPER] == HELPER_SHA, "Pinned helper identity")
    name = str(Path(protocol).resolve().relative_to(ROOT))
    require(name == "research/dialogue-token-protocol.md", "Wrong prospective token protocol")
    bind(ROOT / name, protocol_sha)
    mapping[name] = protocol_sha
    return mapping


def freeze(args):
    with old.attempt(args.out, "freeze", {k: str(v) for k, v in vars(args).items()}) as (out, start, _progress, check):
        require(args.device == "mps", "Frozen protocol requires MPS")
        parents = {name: old.authenticate(Path(getattr(args, name)), getattr(args, name + "_sha256")) for name in ("packet", "lexical", "data")}
        require(parents["data"]["test_contents_accessed"] is False and parents["lexical"]["packet_completed_sha256"] == args.packet_sha256
                and parents["lexical"]["data_completed_sha256"] == args.data_sha256, "Parent lineage/test boundary")
        packet_path = Path(args.packet)
        inherited = read(packet_path / "encoder-plan.json")
        require(inherited["encoder"] == old.ENCODER and inherited["revision"] == old.REVISION and inherited["dtype"] == "float32"
                and inherited["chunk_tokens"] == 254 and inherited["batch_size"] == 128
                and inherited["input_sha256"]["completed.json"] == args.data_sha256, "Original encoding recipe")
        bind(packet_path / "encoder-source.py", inherited["source_sha256"])
        mapping = source_map(parents, args.protocol, args.protocol_sha256)
        old.sources(out, mapping)
        shutil.copyfile(packet_path / "encoder-source.py", out / "original-encoder-source.py")
        old.model_paths(inherited["model_files_sha256"])
        packet, lexical = read(packet_path / "packet.json"), read(Path(args.lexical) / "index.json")
        texts, indices, index = build_contexts(packet, lexical, old.public_dialogues(Path(args.data), packet))
        with (out / "texts.jsonl").open("x") as stream:
            for text in texts:
                stream.write(json.dumps(text, ensure_ascii=False) + "\n")
        np.save(out / "context-indices.npy", indices, allow_pickle=False)
        write(out / "index.json", index)
        plan = {"version": VERSION, "template": TEMPLATE, "encoder": old.ENCODER, "revision": old.REVISION,
            "device": args.device, "dtype": "float32", "chunk_tokens": 254, "batch_size": 128, "width": 384,
            "capacity_unique_contexts": 512, "capacity_seconds": 120, "encoding_seconds": 900,
            "admission_seconds": 720, "disk_cap_bytes": CAP_BYTES, "pooling_max_abs_tolerance": TOLERANCE,
            "source_sha256": mapping, "runtime": old.runtime(), "model_files_sha256": inherited["model_files_sha256"],
            "original_encoder_source_sha256": inherited["source_sha256"],
            "inputs": {n: {"path": str(Path(getattr(args, n)).resolve().relative_to(ROOT)),
                            "completed_sha256": getattr(args, n + "_sha256")} for n in parents},
            "protocol_path": "research/dialogue-token-protocol.md", "protocol_sha256": args.protocol_sha256,
            "layout_files": {n: sha(out / n) for n in ("texts.jsonl", "index.json", "context-indices.npy")},
            "unique_contexts": len(texts), "counts": index["counts"], "context_occurrences": len(indices),
            "prior": "n_chunk / (sum_content_tokens * (n_chunk+2)) for every valid token, including each CLS/SEP",
            "scope": "All existing public System/User contexts once; no candidate tokens or label-dependent selection"}
        write(out / "plan.json", plan)
        check()
        for name in parents:
            old.authenticate(Path(getattr(args, name)), getattr(args, name + "_sha256"))
        for name, digest in mapping.items():
            bind(ROOT / name, digest)
        write(out / "completed.json", {"status": "completed", "phase": "freeze", "version": VERSION,
            "plan_sha256": sha(out / "plan.json"), "source_sha256": mapping, "files": old.manifest(out),
            "wall_seconds": time.perf_counter() - start, "encoder_calls": 0, "test_contents_accessed": False})
    return sha(out / "completed.json")


def validate_plan(path, expected):
    path = Path(path)
    bind(path, expected)
    plan = read(path)
    require(plan["version"] == VERSION and plan["template"] == TEMPLATE and plan["encoder"] == old.ENCODER
            and plan["revision"] == old.REVISION and plan["dtype"] == "float32" and plan["device"] == "mps"
            and plan["chunk_tokens"] == 254 and plan["batch_size"] == 128 and plan["width"] == 384
            and plan["capacity_unique_contexts"] == 512 and plan["capacity_seconds"] == 120
            and plan["encoding_seconds"] == 900 and plan["admission_seconds"] == 720 and plan["disk_cap_bytes"] == CAP_BYTES
            and plan["pooling_max_abs_tolerance"] == TOLERANCE and plan["runtime"] == old.runtime(), "Frozen token recipe/runtime")
    require(set(SOURCES) <= set(plan["source_sha256"]) and plan["source_sha256"][HELPER] == HELPER_SHA, "Preparation source closure")
    for name, digest in plan["source_sha256"].items():
        bind(ROOT / name, digest)
    require(set(plan["inputs"]) == {"packet", "lexical", "data"}, "Input closure")
    for item in plan["inputs"].values():
        old.authenticate(safe(ROOT, item["path"]), item["completed_sha256"])
    require(set(plan["layout_files"]) == {"texts.jsonl", "index.json", "context-indices.npy"}, "Layout closure")
    for name, digest in plan["layout_files"].items():
        bind(safe(path.parent, name), digest)
    bind(path.parent / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    bind(ROOT / plan["protocol_path"], plan["protocol_sha256"])
    done = read(path.parent / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == expected
            and not any((path.parent / x).exists() for x in old.FAILURES), "Freeze not completed")
    return plan


class TokenEncoder(old.Encoder):
    def tokens(self, sequences, progress):
        encoded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            progress["encoder_calls_attempted"] += 1
            hidden = self.model(**encoded).last_hidden_state
            progress["encoder_calls_returned"] += 1
            self.sync()
            memory = self.memory()
            values = hidden.cpu().numpy()
            self.sync()
        return values, int(encoded["input_ids"].numel()), memory


def profile_counts(lengths):
    """Independent integer reconstruction of the sequential 254/128 schedule."""
    offsets, _, chunks = chunks_for_lengths(lengths)
    sizes = chunks + 2
    return {"unique_texts": len(lengths), "input_tokens": int(lengths.sum()),
        "encoder_sequences": len(chunks), "encoder_calls": math.ceil(len(chunks) / 128),
        "encoder_tokens_with_special": int(offsets[-1]),
        "padded_token_slots": sum(len(sizes[i:i + 128]) * int(sizes[i:i + 128].max()) for i in range(0, len(sizes), 128)),
        "overlength_texts_chunked": int((lengths > 254).sum()), "truncated_tokens": 0,
        "minimum_text_tokens": int(lengths.min()), "maximum_text_tokens": int(lengths.max())}


def validate_work(receipt, lengths):
    counts = profile_counts(lengths)
    w, p, eq = receipt["work"], receipt["progress"], receipt["pooling_equivalence"]
    require(p["encoder_calls_attempted"] == p["encoder_calls_returned"] == counts["encoder_calls"]
            and p["encoded_sequences"] == counts["encoder_sequences"] and p["encoded_texts"] == len(lengths), "Encoder call coverage")
    require(all(w[k] == counts[k] for k in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "truncated_tokens"))
            and w["retained_token_rows"] == counts["encoder_tokens_with_special"]
            and w["overlength_contexts_chunked"] == counts["overlength_texts_chunked"], "Encoder token coverage")
    require(eq["contexts_checked"] == len(lengths) and eq["tolerance"] == TOLERANCE
            and math.isfinite(eq["max_abs_error"]) and 0 <= eq["max_abs_error"] <= TOLERANCE, "Pooling-equivalence witness")


def receipt_identity(receipt, plan, plan_sha):
    require(receipt["version"] == plan["version"] == VERSION and receipt["plan_sha256"] == plan_sha
            and receipt["source_sha256"] == plan["source_sha256"] and receipt["runtime"] == plan["runtime"]
            and receipt["model_files_sha256"] == plan["model_files_sha256"] and len(plan["model_files_sha256"]) == 6
            and receipt["encoder"] == plan["encoder"] == old.ENCODER and receipt["revision"] == plan["revision"] == old.REVISION
            and receipt["device"] == plan["device"] == "mps" and receipt["no_retry"] is True
            and receipt["test_contents_accessed"] is False and receipt["parameter_training_steps"] == 0
            and all(receipt[n + "_completed_sha256"] == v["completed_sha256"] for n, v in plan["inputs"].items()), "Cache receipt identity")
    require(plan["template"] == TEMPLATE and plan["dtype"] == "float32" and plan["chunk_tokens"] == 254
            and plan["batch_size"] == 128 and plan["width"] == 384 and plan["capacity_unique_contexts"] == 512
            and plan["capacity_seconds"] == 120 and plan["encoding_seconds"] == 900 and plan["admission_seconds"] == 720
            and plan["disk_cap_bytes"] == CAP_BYTES and plan["pooling_max_abs_tolerance"] == TOLERANCE, "Cache frozen recipe")


def projected_bytes(lengths, cap_files, layout_dir):
    offsets, _, chunks = chunks_for_lengths(lengths)
    variable = int(offsets[-1]) * (384 * 4 + 4) + (len(lengths) + 1) * 8 * 2 + len(chunks) * 8
    fixed = sum(v["bytes"] for n, v in cap_files.items() if n not in TOKEN_FILES)
    return variable + fixed + sum((layout_dir / n).stat().st_size for n in ("context-indices.npy", "index.json")) + 8 * 1024**2


def validate_capacity(path, expected, plan, plan_sha, layout_dir):
    """Authenticate sizing before any full-encoding model construction."""
    path = Path(path)
    cap = old.authenticate(path, expected)
    receipt_identity(cap, plan, plan_sha)
    require(cap["phase"] == "capacity" and 0 < cap["wall_seconds"] <= 120, "Capacity completion")
    expected_files = set(TOKEN_FILES) | {"started.json", "plan.json", "original-encoder-source.py", "token-profile.json",
        "text-token-counts.npy"} | {"sources/" + k for k in plan["source_sha256"]}
    require(set(cap["files"]) == expected_files and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            == expected_files | {"completed.json"}, "Capacity payload closure")
    bind(path / "plan.json", plan_sha)
    bind(path / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    for name, digest in plan["source_sha256"].items():
        bind(safe(path / "sources", name), digest)
    lengths = np.load(path / "text-token-counts.npy", allow_pickle=False)
    full = read(path / "token-profile.json")
    require(lengths.shape == (plan["unique_contexts"],) and all(full[k] == v for k, v in profile_counts(lengths).items()), "Capacity full token profile")
    sample = min(512, len(lengths))
    require(cap["unique_contexts_encoded"] == sample and cap["full_unique_contexts"] == len(lengths)
            and cap["counts"] == plan["counts"], "Fixed capacity sample")
    validate_work(cap, lengths[:sample])
    offsets, chunk_offsets, chunks = chunks_for_lengths(lengths[:sample])
    for name, exact in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        value = np.load(path / name, allow_pickle=False, mmap_mode="r")
        require(value.dtype == np.int64 and np.array_equal(value, exact), "Capacity token geometry")
    for name, shape in (("tokens.npy", (int(offsets[-1]), 384)), ("priors.npy", (int(offsets[-1]),))):
        value = np.load(path / name, allow_pickle=False, mmap_mode="r")
        require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(), "Capacity raw token shape")
    admitted = cap["projection"]
    require(admitted == projection(cap["work"], full, admitted["observed_other_seconds"], projected_bytes(lengths, cap["files"], layout_dir))
            and admitted["encoding_permitted"] is True, "Capacity admission failed")
    return cap, full, lengths


def encode_tokens(texts, lengths, backend, out, source_vectors, progress, check):
    require(len(texts) == len(lengths) == len(source_vectors) and texts, "Context/token/vector coverage")
    require(source_vectors.dtype == np.float32 and source_vectors.shape == (len(texts), 384)
            and np.isfinite(source_vectors).all(), "Original sentence vectors")
    retain_start = time.perf_counter()
    offsets, chunk_offsets, chunks = chunks_for_lengths(lengths)
    require(int(offsets[-1]) * (384 * 4 + 4) < CAP_BYTES, "Raw token payload exceeds cap")
    tokens = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 384))
    priors = np.lib.format.open_memmap(out / "priors.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]),))
    for name, value in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        np.save(out / name, value, allow_pickle=False)
    work = {"forward_transfer_seconds": 0., "retention_seconds": time.perf_counter() - retain_start,
        "input_tokens": 0, "encoder_tokens_with_special": 0, "padded_token_slots": 0, "retained_token_rows": 0,
        "tokenization_seconds": 0., "truncated_tokens": 0, "overlength_contexts_chunked": int((lengths > 254).sum()),
        "mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}
    pending, chunk_cursor, row_cursor = [], 0, 0
    def flush():
        nonlocal row_cursor, chunk_cursor
        batch = pending[:128]
        del pending[:128]
        backend.sync()
        stamp = time.perf_counter()
        values, padded, memory = backend.tokens([x[1] for x in batch], progress)
        backend.sync()
        work["forward_transfer_seconds"] += time.perf_counter() - stamp
        require(values.dtype == np.float32 and values.shape == (len(batch), max(len(x[1]) for x in batch), 384), "Raw hidden shape/type")
        work["padded_token_slots"] += padded
        for key, value in memory.items():
            work[key] = max(work[key], value)
        stamp = time.perf_counter()
        for j, (context, ids, content_length) in enumerate(batch):
            block = values[j, :len(ids)]
            require(np.isfinite(block).all() and chunks[chunk_cursor] == content_length, "Nonfinite/misaligned token states")
            tokens[row_cursor:row_cursor + len(ids)] = block
            priors[row_cursor:row_cursor + len(ids)] = content_length / (int(lengths[context]) * len(ids))
            row_cursor += len(ids)
            chunk_cursor += 1
        work["retention_seconds"] += time.perf_counter() - stamp
        work["encoder_tokens_with_special"] += sum(len(x[1]) for x in batch)
        work["retained_token_rows"] = row_cursor
        progress["encoded_sequences"] += len(batch)
        progress["work"] = dict(work)
        check()
    for start in range(0, len(texts), 512):
        stamp = time.perf_counter()
        tokenized = backend.tokenize(texts[start:start + 512])
        work["tokenization_seconds"] += time.perf_counter() - stamp
        require(len(tokenized) == len(texts[start:start + 512]), "Tokenization coverage")
        for local, ids in enumerate(tokenized):
            context = start + local
            require(len(ids) == lengths[context] and ids and all(type(i) is int and i >= 0 for i in ids), "Profile/current tokenization differs")
            work["input_tokens"] += len(ids)
            for j in range(0, len(ids), 254):
                part = ids[j:j + 254]
                pending.append((context, [backend.tokenizer.cls_token_id, *part, backend.tokenizer.sep_token_id], len(part)))
                if len(pending) == 128:
                    flush()
        check()
    if pending:
        flush()
    require(row_cursor == int(offsets[-1]) and chunk_cursor == len(chunks), "Incomplete token writes")
    stamp = time.perf_counter()
    tokens.flush()
    priors.flush()
    max_error = 0.
    for i in range(len(texts)):
        a, b = int(offsets[i]), int(offsets[i + 1])
        # This is also the zero-logit attention result, with only f32 rounding.
        pooled = (tokens[a:b] * priors[a:b, None]).sum(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        require(math.isfinite(norm) and norm > 1e-12, "Invalid pooled token mean")
        pooled /= norm
        error = float(np.max(np.abs(pooled - source_vectors[i])))
        max_error = max(max_error, error)
        require(error <= TOLERANCE, "Original sentence-vector equivalence exceeded fixed tolerance")
        if i % 512 == 0:
            check()
    payloads = {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in TOKEN_FILES}
    work["retention_seconds"] += time.perf_counter() - stamp
    progress["encoded_texts"] = len(texts)
    progress["work"] = dict(work)
    return work, {"contexts_checked": len(texts), "max_abs_error": max_error, "tolerance": TOLERANCE}, payloads


def encode(args, *, backend_factory=TokenEncoder):
    phase = args.phase
    require(phase in ("capacity", "encode"), "Encoding phase")
    with old.attempt(args.out, phase, {k: str(v) for k, v in vars(args).items()}) as (out, start, progress, check):
        plan = validate_plan(args.plan, args.plan_sha256)
        old.sources(out, plan["source_sha256"])
        shutil.copyfile(args.plan, out / "plan.json")
        shutil.copyfile(Path(args.plan).parent / "original-encoder-source.py", out / "original-encoder-source.py")
        if phase == "encode":
            require(args.capacity and args.capacity_sha256, "Completed capacity required")
            _, full_profile, lengths = validate_capacity(Path(args.capacity), args.capacity_sha256, plan, args.plan_sha256, Path(args.plan).parent)
        plan_dir = Path(args.plan).parent
        texts = [json.loads(line) for line in (plan_dir / "texts.jsonl").read_text().splitlines()]
        index = read(plan_dir / "index.json")
        require(len(texts) == plan["unique_contexts"] == index["unique_contexts"] and len(set(texts)) == len(texts), "Unique contexts")
        source = np.load(safe(ROOT, plan["inputs"]["packet"]["path"]) / "features.npy", allow_pickle=False, mmap_mode="r")
        selected = min(512, len(texts)) if phase == "capacity" else len(texts)
        vectors = source[np.asarray(index["original_feature_indices"][:selected], np.int64)]
        check()
        stamp = time.perf_counter()
        backend = backend_factory(plan)
        backend.sync()
        setup_seconds, initial_memory = time.perf_counter() - stamp, backend.memory()
        if phase == "capacity":
            full_profile, lengths = old.token_profile(texts, backend, check)
            np.save(out / "text-token-counts.npy", lengths, allow_pickle=False)
            write(out / "token-profile.json", full_profile)
        else:
            for name in ("text-token-counts.npy", "token-profile.json"):
                shutil.copyfile(Path(args.capacity) / name, out / name)
        require(lengths.shape == (len(texts),) and lengths.dtype == np.int64, "Full token lengths")
        require(all(full_profile[k] == v for k, v in profile_counts(lengths).items()), "Full token profile arithmetic")
        work, equivalence, variable_files = encode_tokens(texts[:selected], lengths[:selected], backend, out, vectors, progress, check)
        validate_work({"work": work, "pooling_equivalence": equivalence, "progress": progress}, lengths[:selected])
        if phase == "encode":
            for name in ("context-indices.npy", "index.json"):
                shutil.copyfile(plan_dir / name, out / name)
        progress["phase"] = "final_authentication"
        validate_plan(args.plan, args.plan_sha256)
        old.model_paths(plan["model_files_sha256"])
        files = dict(variable_files)
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.relative_to(out).as_posix() not in files:
                files[p.relative_to(out).as_posix()] = {"sha256": sha(p), "bytes": p.stat().st_size}
        check()
        receipt = {"status": "completed", "phase": phase, "version": VERSION, "plan_sha256": args.plan_sha256,
            "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "device": plan["device"],
            "model_files_sha256": plan["model_files_sha256"], "encoder": old.ENCODER, "revision": old.REVISION,
            **{n + "_completed_sha256": item["completed_sha256"] for n, item in plan["inputs"].items()},
            "capacity_path": str(Path(args.capacity).resolve().relative_to(ROOT)) if phase == "encode" else None,
            "capacity_completed_sha256": args.capacity_sha256 if phase == "encode" else None,
            "unique_contexts_encoded": selected, "full_unique_contexts": len(texts), "counts": plan["counts"],
            "work": work, "progress": progress, "pooling_equivalence": equivalence,
            "setup_seconds": setup_seconds, "initial_device_memory": initial_memory,
            "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
            "device_memory_scope": "Sampled live MPS allocation maxima, not allocator high water",
            "retention_scope": "Raw token/prior writes, mmap flush, weighted-mean equivalence, and token-payload hashes; not durable disk timing",
            "files": files, "cache_file_bytes": sum(v["bytes"] for v in files.values()),
            "parameter_training_steps": 0, "test_contents_accessed": False, "no_retry": True}
        if phase == "capacity":
            overhead = time.perf_counter() - start - work["forward_transfer_seconds"] - work["retention_seconds"]
            receipt["projection"] = projection(work, full_profile, overhead, projected_bytes(lengths, files, plan_dir))
        receipt["wall_seconds"] = time.perf_counter() - start
        # Include the serialized completion itself in admission before writing it.
        completion_bytes = len((json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode())
        require(receipt["cache_file_bytes"] + completion_bytes <= CAP_BYTES, "Actual cache exceeds four GiB")
        write(out / "completed.json", receipt)
    return sha(out / "completed.json")


def authenticate_cache(path, completed_sha256):
    """Return index, raw tokens, context offsets, priors, flat turn context IDs."""
    path = Path(path)
    receipt = old.authenticate(path, completed_sha256)
    require(receipt["phase"] == "encode" and receipt["version"] == VERSION and receipt["no_retry"] is True
            and receipt["test_contents_accessed"] is False and receipt["parameter_training_steps"] == 0
            and 0 < receipt["wall_seconds"] <= 900, "Incomplete full token cache")
    bind(path / "plan.json", receipt["plan_sha256"])
    plan = read(path / "plan.json")
    receipt_identity(receipt, plan, receipt["plan_sha256"])
    require(receipt["source_sha256"] == plan["source_sha256"] and receipt["runtime"] == plan["runtime"]
            and receipt["model_files_sha256"] == plan["model_files_sha256"] and plan["source_sha256"][HELPER] == HELPER_SHA,
            "Cache provenance")
    expected = set(TOKEN_FILES) | {"started.json", "plan.json", "original-encoder-source.py", "token-profile.json",
        "text-token-counts.npy", "context-indices.npy", "index.json"} | {"sources/" + k for k in plan["source_sha256"]}
    require(set(receipt["files"]) == expected and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            == expected | {"completed.json"}, "Cache payload closure")
    for name, digest in plan["source_sha256"].items():
        bind(safe(path / "sources", name), digest)
    bind(path / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    capacity_path = safe(ROOT, receipt["capacity_path"])
    _, full, _ = validate_capacity(capacity_path, receipt["capacity_completed_sha256"], plan, receipt["plan_sha256"], path)
    for name in ("token-profile.json", "text-token-counts.npy"):
        bind(path / name, sha(capacity_path / name))
    for name in ("index.json", "context-indices.npy"):
        bind(path / name, plan["layout_files"][name])
    require(all(full[k] == v for k, v in profile_counts(np.load(path / "text-token-counts.npy", allow_pickle=False)).items()), "Full profile changed")
    index = read(path / "index.json")
    arrays = {n: np.load(path / n, allow_pickle=False, mmap_mode="r") for n in (*TOKEN_FILES, "context-indices.npy", "text-token-counts.npy")}
    tokens, priors = arrays["tokens.npy"], arrays["priors.npy"]
    offsets, chunk_offsets, chunks = chunks_for_lengths(arrays["text-token-counts.npy"])
    require(index["version"] == VERSION and index["template"] == TEMPLATE and index["width"] == 384
            and index["unique_contexts"] == plan["unique_contexts"] == receipt["unique_contexts_encoded"] == receipt["full_unique_contexts"]
            and index["counts"] == plan["counts"] == receipt["counts"], "Cache context identity")
    for name, value in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        require(arrays[name].dtype == np.int64 and np.array_equal(arrays[name], value), "Token offsets/chunks")
    require(tokens.dtype == priors.dtype == np.float32 and tokens.shape == (int(offsets[-1]), 384)
            and priors.shape == (int(offsets[-1]),), "Token dtype/shape")
    context_ids = arrays["context-indices.npy"]
    require(context_ids.dtype == np.int64 and context_ids.shape == (index["context_occurrences"],)
            and ((context_ids >= 0) & (context_ids < len(offsets) - 1)).all(), "Context IDs")
    for start in range(0, len(tokens), 8192):
        require(np.isfinite(tokens[start:start + 8192]).all() and np.isfinite(priors[start:start + 8192]).all()
                and (priors[start:start + 8192] > 0).all(), "Nonfinite raw cache")
    row = 0
    for i, length in enumerate(arrays["text-token-counts.npy"]):
        for n in chunks[chunk_offsets[i]:chunk_offsets[i + 1]]:
            require(np.all(priors[row:row + n + 2] == np.float32(n / (int(length) * (n + 2)))), "Chunk prior differs")
            row += n + 2
    validate_work(receipt, arrays["text-token-counts.npy"])
    require(receipt["cache_file_bytes"] == sum(v["bytes"] for v in receipt["files"].values()), "Cache payload byte count")
    require(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= CAP_BYTES, "Disk cap")
    return index, tokens, arrays["offsets.npy"], priors, context_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = sub.add_parser("freeze")
    for key in ("packet", "lexical", "data", "protocol", "packet-sha256", "lexical-sha256", "data-sha256", "protocol-sha256", "out"):
        freeze_parser.add_argument("--" + key, required=True)
    freeze_parser.add_argument("--device", choices=("mps",), required=True)
    for phase in ("capacity", "encode"):
        p = sub.add_parser(phase)
        for key in ("plan", "plan-sha256", "out"):
            p.add_argument("--" + key, required=True)
        p.add_argument("--capacity", required=phase == "encode")
        p.add_argument("--capacity-sha256", required=phase == "encode")
    args = parser.parse_args()
    print(json.dumps({"status": "completed", "completed_sha256": freeze(args) if args.phase == "freeze" else encode(args)}))
