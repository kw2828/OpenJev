"""Freeze, size and encode public candidate-conditioned MiniLM observations.

No inference at import time. The downstream cache loader needs only NumPy and
stdlib. Every execution is exclusive; failures are retained without retry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TEMPLATE = "candidate_text + '\\n' + 'System: ' + previous_system_text + '\\nUser: ' + current_user_text"
VERSION = "dialogue-joint-minilm-v1"
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.json", "vocab.txt"}
SOURCES = ("scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py")
LIMITS = {"freeze": 120, "capacity": 120, "encode": 900}
DISK_CAP = 4 * 1024**3
CHUNK = 254
BATCH = 128
WIDTH = 384
FAILURES = ("failed.json", "late-completion.json", "cleanup-error.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe member path")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "Member escapes root")
    return path


def bind(path, expected):
    require(type(expected) is str and len(expected) == 64 and all(x in "0123456789abcdef" for x in expected), "Invalid hash")
    require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == expected, "Hash mismatch: " + str(path))


def authenticate(folder, expected):
    folder = Path(folder)
    require(not any((folder / n).exists() for n in FAILURES), "Failed input attempt")
    bind(folder / "completed.json", expected)
    receipt = read(folder / "completed.json")
    require(receipt["status"] == "completed" and type(receipt["files"]) is dict and receipt["files"], "Incomplete input")
    for name, item in receipt["files"].items():
        path = safe(folder, name)
        if type(item) is dict:
            require(path.stat().st_size == item["bytes"], "Payload byte count")
            item = item["sha256"]
        bind(path, item)
    return receipt


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: str(importlib.metadata.version(name)) for name in ("numpy", "torch", "transformers", "huggingface-hub")}}


def template(candidate, system, user):
    require(all(type(x) is str for x in (candidate, system, user)) and bool(candidate.strip()), "Invalid public text")
    return candidate + "\nSystem: " + system + "\nUser: " + user


def build_layout(packet, lexical_index, catalogs, public):
    """Pure public layout; gold labels, scored times and transition bins unused."""
    require(set(packet["cohorts"]) == set(lexical_index["cohorts"]) == set(catalogs) == set(public) == {"train", "dev"},
            "Only train/dev permitted")
    lookup, texts, arrays, offset, cohorts, counts = {}, [], [], 0, {}, {}
    text_identity = {}
    def original_text(index, value):
        require(type(index) is int and index >= 0 and type(value) is str, "Original text identity")
        require(text_identity.setdefault(index, value) == value, "Original embedding/text alignment")
    catalogs_by_id = {}
    for split in ("train", "dev"):
        catalogs_by_id[split] = {q["query_id"]: q for q in catalogs[split]}
        require(len(catalogs_by_id[split]) == len(catalogs[split]), "Duplicate catalog identity")
    candidates = []
    for q in packet["queries"]:
        require(q["split"] in catalogs_by_id, "Foreign query split")
        catalog = catalogs_by_id[q["split"]][q["id"]]
        cs = catalog["candidates"]
        require([c["id"] for c in cs] == q["candidate_ids"] and [c.get("value") for c in cs] == q["candidate_values"]
                and len(cs) == len(q["candidates"]) and 3 <= len(cs) <= 12, "Original candidate catalog differs")
        original_text(q["text"], catalog["query_text"])
        for ti, c in zip(q["candidates"], cs, strict=True):
            original_text(ti, c["text"])
        candidates.append([c["text"] for c in cs])
    for split in ("train", "dev"):
        ds, entries = packet["cohorts"][split], lexical_index["cohorts"][split]
        require(len(ds) == len(entries) and {d["id"] for d in ds} == set(public[split]), "Public dialogue membership")
        cohorts[split] = []
        totals = {"dialogues": len(ds), "public_turns": 0, "public_question_steps": 0, "real_candidate_steps": 0}
        for d, entry in zip(ds, entries, strict=True):
            qids = entry["query_ids"]
            require(entry["id"] == d["id"] and qids == sorted({r["query"] for r in d["queries"]}) and qids,
                    "Requested question layout")
            require(all(type(qi) is int and 0 <= qi < len(candidates) and packet["queries"][qi]["split"] == split for qi in qids),
                    "Invalid requested question")
            raw = public[split][d["id"]]
            turns = []
            for step in raw["user_turns"]:
                ti, si = step["turn_index"], step["previous_system_turn_index"]
                require(type(ti) is int and 0 <= ti < len(raw["turns"]) and raw["turns"][ti]["speaker"] == "USER",
                        "USER chronology")
                require(si is None or (type(si) is int and 0 <= si < ti and raw["turns"][si]["speaker"] == "SYSTEM"),
                        "Previous SYSTEM chronology")
                turns.append(("" if si is None else raw["turns"][si]["utterance"], raw["turns"][ti]["utterance"]))
            require(turns and len(turns) == len(d["turns"]) and [u for _, u in turns] == d["user_text"], "Original public turns differ")
            cmax = max(len(candidates[q]) for q in qids)
            shape = [len(turns), len(qids), cmax]
            require(entry["shape"] == shape + [10], "Lexical/public layout shape")
            block = np.full(shape, -1, np.int64)
            for t, (system, user) in enumerate(turns):
                original_text(d["turns"][t], "System: " + system + "\nUser: " + user)
                for j, qi in enumerate(qids):
                    for c, candidate in enumerate(candidates[qi]):
                        text = template(candidate, system, user)
                        if text not in lookup:
                            lookup[text] = len(texts)
                            texts.append(text)
                        block[t, j, c] = lookup[text]
            arrays.append(block.ravel())
            cohorts[split].append({"id": d["id"], "query_ids": qids, "offset": offset, "shape": shape})
            offset += block.size
            totals["public_turns"] += len(turns)
            totals["public_question_steps"] += len(turns) * len(qids)
            totals["real_candidate_steps"] += len(turns) * sum(len(candidates[q]) for q in qids)
        counts[split] = totals
    require(texts and arrays, "Empty public corpus")
    indices = np.concatenate(arrays)
    return texts, indices, {"version": VERSION, "template": TEMPLATE, "embedding_width": WIDTH,
        "padding_index": -1, "unique_texts": len(texts), "index_count": int(indices.size), "cohorts": cohorts, "counts": counts}


def public_dialogues(data, packet):
    public = {}
    for split in ("train", "dev"):
        requested = {d["id"] for d in packet["cohorts"][split]}
        chosen = {}
        with (data / (split + "-dialogues.jsonl")).open() as stream:
            for line in stream:
                item = json.loads(line)
                if item["dialogue_id"] in requested:
                    require(item["dialogue_id"] not in chosen, "Duplicate public dialogue")
                    chosen[item["dialogue_id"]] = item
        require(set(chosen) == requested, "Missing public dialogue")
        public[split] = chosen
    return public


def model_paths(expected):
    from huggingface_hub import hf_hub_download
    require(set(expected) == MODEL_FILES, "Exact six encoder files required")
    result = {n: Path(hf_hub_download(ENCODER, n, revision=REVISION, local_files_only=True)) for n in sorted(MODEL_FILES)}
    for name, path in result.items():
        # The HF snapshot cache uses legitimate symlinks to immutable blob files.
        require(path.is_file() and sha(path) == expected[name], "Cached encoder changed: " + name)
    return result


def sources(out, mapping):
    for name, digest in mapping.items():
        path = safe(ROOT, name)
        bind(path, digest)
        target = safe(out / "sources", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())


def manifest(out):
    return {p.relative_to(out).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "completed.json"}


@contextmanager
def attempt(out, phase, request):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    progress = {"phase": phase, "encoder_calls_attempted": 0, "encoder_calls_returned": 0,
                "encoded_sequences": 0, "encoded_texts": 0}
    def check():
        if time.perf_counter() - start > LIMITS[phase]:
            raise TimeoutError("Fixed " + phase + " cap exceeded")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= DISK_CAP, "Four GiB output cap exceeded")
    def alarm(_signal, _frame):
        raise TimeoutError("Fixed " + phase + " alarm expired")
    previous = None
    try:
        write(out / "started.json", {"status": "started", "phase": phase, "request": request,
            "cap_seconds": LIMITS[phase], "disk_cap_bytes": DISK_CAP, "source_sha256": sha(__file__), "no_retry": True})
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        previous = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(.001, LIMITS[phase] - (time.perf_counter() - start)))
        yield out, start, progress, check
        check()
    except BaseException as error:
        if (out / "completed.json").exists():
            try:
                (out / "completed.json").rename(out / "late-completion.json")
            except OSError as secondary:
                if callable(getattr(error, "add_note", None)):
                    error.add_note("Completion demotion: " + repr(secondary))
        try:
            write(out / "failed.json", {"status": "failed", "error": str(error), "error_type": type(error).__name__,
                "wall_seconds": time.perf_counter() - start, "progress": progress, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - original failure remains primary
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def freeze(args):
    request = {k: str(v) for k, v in vars(args).items()}
    with attempt(args.out, "freeze", request) as (out, start, progress, check):
        parents = {}
        for label in ("packet", "lexical", "data"):
            directory = Path(getattr(args, label)).resolve()
            pin = getattr(args, label + "_sha256")
            parents[label] = authenticate(directory, pin)
            check()
        require(parents["data"]["test_contents_accessed"] is False, "Test data not permitted")
        require(parents["lexical"]["packet_completed_sha256"] == args.packet_sha256
                and parents["lexical"]["data_completed_sha256"] == args.data_sha256, "Input lineage differs")
        packet_path, data_path = Path(args.packet), Path(args.data)
        inherited = read(packet_path / "encoder-plan.json")
        require(inherited["encoder"] == ENCODER and inherited["revision"] == REVISION
                and inherited["chunk_tokens"] == CHUNK and inherited["batch_size"] == BATCH
                and inherited["dtype"] == "float32", "Original encoder recipe differs")
        require(inherited["input_sha256"]["completed.json"] == args.data_sha256, "Encoder data lineage differs")
        bind(packet_path / "encoder-source.py", inherited["source_sha256"])
        bind(Path(args.protocol), args.protocol_sha256)
        mapping = {name: sha(ROOT / name) for name in SOURCES}
        mapping["scripts/study_dialogue_memory.py"] = sha(ROOT / "scripts/study_dialogue_memory.py")
        for source_map in (parents["data"]["implementation_sha256"], parents["lexical"]["source_sha256"]):
            for name, digest in source_map.items():
                require(name not in mapping or mapping[name] == digest, "Conflicting source identity")
                mapping[name] = digest
        protocol_name = str(Path(args.protocol).resolve().relative_to(ROOT))
        mapping[protocol_name] = args.protocol_sha256
        sources(out, mapping)
        shutil.copyfile(packet_path / "encoder-source.py", out / "original-encoder-source.py")
        paths = model_paths(inherited["model_files_sha256"])
        packet = read(packet_path / "packet.json")
        index = read(Path(args.lexical) / "index.json")
        public = public_dialogues(data_path, packet)
        texts, indices, layout = build_layout(packet, index, read(data_path / "catalog.json"), public)
        check()
        with (out / "texts.jsonl").open("x") as stream:
            for text in texts:
                stream.write(json.dumps(text, ensure_ascii=False) + "\n")
        np.save(out / "indices.npy", indices, allow_pickle=False)
        write(out / "index.json", layout)
        plan = {"version": VERSION, "encoder": ENCODER, "revision": REVISION, "template": TEMPLATE,
            "device": args.device, "dtype": "float32", "chunk_tokens": CHUNK, "batch_size": BATCH,
            "capacity_unique_texts": 512, "capacity_seconds": 120, "encoding_seconds": 900, "disk_cap_bytes": DISK_CAP,
            "source_sha256": mapping, "runtime": runtime(), "model_files_sha256": inherited["model_files_sha256"],
            "original_encoder_source_sha256": inherited["source_sha256"],
            "inputs": {name: {"path": str(Path(getattr(args, name)).resolve().relative_to(ROOT)),
                              "completed_sha256": getattr(args, name + "_sha256")} for name in parents},
            "protocol_path": protocol_name, "protocol_sha256": args.protocol_sha256,
            "layout_files": {n: sha(out / n) for n in ("texts.jsonl", "indices.npy", "index.json")},
            "counts": layout["counts"], "unique_texts": len(texts), "index_count": int(indices.size),
            "projected_embedding_bytes": len(texts) * WIDTH * 4, "padded_index_bytes": int(indices.nbytes),
            "label_boundary": "All public turns x requested question x valid candidate; no label/time/bin selects texts"}
        write(out / "plan.json", plan)
        for label in parents:
            authenticate(Path(getattr(args, label)), getattr(args, label + "_sha256"))
        for name, digest in mapping.items():
            bind(ROOT / name, digest)
        for name, path in paths.items():
            require(sha(path) == plan["model_files_sha256"][name], "Encoder file changed")
        check()
        progress["unique_texts"] = len(texts)
        write(out / "completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": sha(out / "plan.json"),
            "source_sha256": mapping, "files": manifest(out), "wall_seconds": time.perf_counter() - start,
            "encoder_calls": 0, "test_contents_accessed": False})
    return sha(out / "completed.json")


def validate_plan(path, expected):
    path = Path(path)
    bind(path, expected)
    plan = read(path)
    require(plan["version"] == VERSION and plan["encoder"] == ENCODER and plan["revision"] == REVISION
            and plan["template"] == TEMPLATE and plan["dtype"] == "float32" and plan["chunk_tokens"] == CHUNK
            and plan["batch_size"] == BATCH and plan["device"] in ("cpu", "mps")
            and plan["capacity_unique_texts"] == 512 and plan["capacity_seconds"] == 120
            and plan["encoding_seconds"] == 900 and plan["disk_cap_bytes"] == DISK_CAP, "Frozen recipe mismatch")
    require(plan["runtime"] == runtime(), "Runtime differs from preparation")
    require(set(SOURCES) <= set(plan["source_sha256"]), "Preparation sources missing")
    for name, digest in plan["source_sha256"].items():
        bind(ROOT / name, digest)
    for name, item in plan["inputs"].items():
        require(name in ("packet", "lexical", "data"), "Unknown parent")
        authenticate(safe(ROOT, item["path"]), item["completed_sha256"])
    require(set(plan["inputs"]) == {"packet", "lexical", "data"}, "Missing parent")
    bind(path.parent / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    bind(safe(ROOT, plan["inputs"]["packet"]["path"]) / "encoder-source.py", plan["original_encoder_source_sha256"])
    for name, digest in plan["layout_files"].items():
        bind(safe(path.parent, name), digest)
    require(set(plan["layout_files"]) == {"texts.jsonl", "index.json", "indices.npy"}, "Layout membership")
    bind(ROOT / plan["protocol_path"], plan["protocol_sha256"])
    require(not any((path.parent / n).exists() for n in FAILURES), "Failed freeze")
    completion = read(path.parent / "completed.json")
    require(completion["status"] == "completed" and completion["phase"] == "freeze"
            and completion["plan_sha256"] == expected, "Freeze completion missing")
    return plan


class Encoder:
    """Local-only fixed MiniLM, preserving the original pooling arithmetic."""
    def __init__(self, plan):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, plan["device"]
        require(self.device != "mps" or torch.backends.mps.is_available(), "MPS unavailable")
        model_paths(plan["model_files_sha256"])
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION, local_files_only=True)
        with torch.random.fork_rng(devices=[]):
            self.model = AutoModel.from_pretrained(ENCODER, revision=REVISION, local_files_only=True,
                attn_implementation="eager").to(device=self.device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.sync()

    def sync(self):
        if self.device == "mps":
            self.torch.mps.synchronize()

    def memory(self):
        if self.device == "mps":
            return {"mps_current_allocated_bytes": self.torch.mps.current_allocated_memory(),
                    "mps_driver_allocated_bytes": self.torch.mps.driver_allocated_memory()}
        return {"mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}

    def tokenize(self, texts):
        return self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]

    def pooled(self, sequences):
        encoded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            self.sync()
            memory = self.memory()  # Includes live hidden/pooling allocations.
            values = pooled.cpu().numpy()
        return values, int(encoded["input_ids"].numel()), memory


def token_profile(texts, backend, check):
    """Full sequential batch-128 workload, tokenization only, no forward calls."""
    lengths, pending = [], []
    total_tokens = total_with_special = padded = chunks = 0
    started = time.perf_counter()
    for offset in range(0, len(texts), 512):
        batch = backend.tokenize(texts[offset:offset + 512])
        require(len(batch) == len(texts[offset:offset + 512]), "Profile token coverage")
        for ids in batch:
            require(ids and all(type(i) is int and i >= 0 for i in ids), "Profile empty/invalid tokens")
            n = len(ids)
            lengths.append(n)
            total_tokens += n
            for j in range(0, n, CHUNK):
                size = min(CHUNK, n - j) + 2
                total_with_special += size
                chunks += 1
                pending.append(size)
                if len(pending) == BATCH:
                    padded += BATCH * max(pending)
                    pending.clear()
        check()
    if pending:
        padded += len(pending) * max(pending)
    return {"unique_texts": len(texts), "input_tokens": total_tokens,
        "encoder_sequences": chunks, "encoder_calls": math.ceil(chunks / BATCH),
        "encoder_tokens_with_special": total_with_special, "padded_token_slots": padded,
        "overlength_texts_chunked": sum(x > CHUNK for x in lengths), "truncated_tokens": 0,
        "minimum_text_tokens": min(lengths), "maximum_text_tokens": max(lengths),
        "tokenization_seconds": time.perf_counter() - started}, np.asarray(lengths, dtype=np.int64)


def capacity_projection(work, full_profile, nonencoding_seconds, estimated_cache_bytes):
    require(work["encoder_seconds"] > 0 and work["padded_token_slots"] > 0
            and math.isfinite(nonencoding_seconds) and nonencoding_seconds >= 0, "Invalid sizing units")
    encoder = work["encoder_seconds"] / work["padded_token_slots"] * full_profile["padded_token_slots"]
    total = encoder + nonencoding_seconds
    return {"projected_encoder_seconds": encoder, "observed_nonencoding_seconds": nonencoding_seconds,
        "projected_total_seconds": total, "maximum_projected_seconds": 720.,
        "projected_cache_bytes": estimated_cache_bytes, "maximum_cache_bytes": DISK_CAP,
        "encoding_permitted": total <= 720. and estimated_cache_bytes <= DISK_CAP,
        "formula": "sample synchronized encoder_seconds / sample padded_token_slots * full padded_token_slots + observed capacity nonencoding_seconds",
        "scope": "Heuristic extrapolation from deterministic first 512 unique texts, not measured full encoding time"}


def encode_texts(texts, backend, destination, progress, check):
    require(texts and all(type(t) is str and t.strip() for t in texts), "Empty encoding input")
    require(not destination.exists(), "Embedding path already exists")
    vectors = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=(len(texts), WIDTH))
    vectors[:] = 0
    weights = np.zeros(len(texts), np.int64)
    pending = []
    counts = {"input_tokens": 0, "encoder_tokens_with_special": 0, "padded_token_slots": 0,
              "overlength_texts_chunked": 0, "truncated_tokens": 0, "tokenization_seconds": 0., "encoder_seconds": 0.,
              "pool_accumulation_seconds": 0., "mps_current_allocated_bytes": 0, "mps_driver_allocated_bytes": 0}
    def flush():
        batch = pending[:BATCH]
        del pending[:BATCH]
        backend.sync()
        started = time.perf_counter()
        progress["encoder_calls_attempted"] += 1
        values, slots, memory = backend.pooled([x[1] for x in batch])
        progress["encoder_calls_returned"] += 1
        backend.sync()
        counts["encoder_seconds"] += time.perf_counter() - started
        require(values.shape == (len(batch), WIDTH) and values.dtype == np.float32 and np.isfinite(values).all(), "Invalid encoder return")
        counts["padded_token_slots"] += slots
        for k, value in memory.items():
            counts[k] = max(counts[k], value)
        started = time.perf_counter()
        for (i, ids, w), value in zip(batch, values, strict=True):
            vectors[i] += w * value
            weights[i] += w
            counts["encoder_tokens_with_special"] += len(ids)
        progress["encoded_sequences"] += len(batch)
        counts["pool_accumulation_seconds"] += time.perf_counter() - started
        progress["encoder_work"] = dict(counts)
        check()
    for offset in range(0, len(texts), 512):
        started = time.perf_counter()
        tokens = backend.tokenize(texts[offset:offset + 512])
        counts["tokenization_seconds"] += time.perf_counter() - started
        require(len(tokens) == len(texts[offset:offset + 512]), "Tokenization coverage")
        for local, ids in enumerate(tokens):
            require(ids and all(type(i) is int and i >= 0 for i in ids), "Empty/invalid tokens")
            counts["input_tokens"] += len(ids)
            counts["overlength_texts_chunked"] += int(len(ids) > CHUNK)
            for j in range(0, len(ids), CHUNK):
                part = ids[j:j + CHUNK]
                pending.append((offset + local, [backend.tokenizer.cls_token_id, *part, backend.tokenizer.sep_token_id], len(part)))
                if len(pending) == BATCH:
                    flush()
        check()
    if pending:
        flush()
    require((weights > 0).all() and int(weights.sum()) == counts["input_tokens"], "All-token coverage")
    for offset in range(0, len(texts), 8192):
        block = vectors[offset:offset + 8192]
        block /= weights[offset:offset + 8192, None]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        require(np.isfinite(block).all() and (norms > 1e-12).all(), "Nonfinite/zero frozen embedding")
        block /= norms
        check()
    vectors.flush()
    progress["encoded_texts"] = len(texts)
    return counts


def encode(args, *, backend_factory=Encoder):
    phase = args.phase
    require(phase in ("capacity", "encode"), "Unknown encoding phase")
    request = {k: str(v) for k, v in vars(args).items()}
    with attempt(args.out, phase, request) as (out, start, progress, check):
        plan = validate_plan(args.plan, args.plan_sha256)
        sources(out, plan["source_sha256"])
        shutil.copyfile(args.plan, out / "plan.json")
        shutil.copyfile(Path(args.plan).parent / "original-encoder-source.py", out / "original-encoder-source.py")
        if phase == "encode":
            require(args.capacity and args.capacity_sha256, "Completed capacity is required")
            cap = authenticate(Path(args.capacity), args.capacity_sha256)
            require(cap["phase"] == "capacity" and cap["plan_sha256"] == args.plan_sha256
                    and cap["no_retry"] is True and 0 < cap["wall_seconds"] <= LIMITS["capacity"]
                    and cap["runtime"] == plan["runtime"] and cap["source_sha256"] == plan["source_sha256"]
                    and cap["unique_texts_encoded"] == min(512, plan["unique_texts"]), "Capacity lineage differs")
            require(cap["projection"]["encoding_permitted"] is True
                    and cap["projection"]["projected_total_seconds"] <= 720.
                    and cap["projection"]["projected_cache_bytes"] <= DISK_CAP, "Capacity denied full encoding")
        plan_dir = Path(args.plan).parent
        texts = [json.loads(line) for line in (plan_dir / "texts.jsonl").read_text().splitlines()]
        require(len(texts) == plan["unique_texts"] and len(set(texts)) == len(texts), "Unique-text coverage")
        selected = texts[:512] if phase == "capacity" else texts
        require(len(texts) * WIDTH * 4 + plan["padded_index_bytes"] < DISK_CAP, "Projected cache exceeds fixed disk cap")
        check()
        setup = time.perf_counter()
        backend = backend_factory(plan)
        backend.sync()
        setup_seconds, initial_memory = time.perf_counter() - setup, backend.memory()
        check()
        if phase == "capacity":
            full_profile, token_counts = token_profile(texts, backend, check)
            np.save(out / "text-token-counts.npy", token_counts, allow_pickle=False)
            write(out / "token-profile.json", full_profile)
        work = encode_texts(selected, backend, out / "embeddings.npy", progress, check)
        if phase == "encode":
            for name in ("indices.npy", "index.json"):
                shutil.copyfile(plan_dir / name, out / name)
            full_profile = read(Path(args.capacity) / "token-profile.json")
            for key in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "overlength_texts_chunked", "truncated_tokens"):
                require(work[key] == full_profile[key], "Full token profile differs: " + key)
            require(progress["encoded_sequences"] == full_profile["encoder_sequences"], "Full chunk count differs")
        progress["phase"] = "final_hashing"
        validate_plan(args.plan, args.plan_sha256)
        model_paths(plan["model_files_sha256"])
        check()
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        receipt = {"status": "completed", "version": VERSION, "phase": phase, "plan_sha256": args.plan_sha256,
            "source_sha256": plan["source_sha256"], "runtime": plan["runtime"],
            **{label + "_completed_sha256": item["completed_sha256"] for label, item in plan["inputs"].items()},
            "capacity_completed_sha256": args.capacity_sha256 if phase == "encode" else None,
            "capacity_path": str(Path(args.capacity).resolve().relative_to(ROOT)) if phase == "encode" else None,
            "unique_texts_encoded": len(selected), "full_unique_texts": plan["unique_texts"], "counts": plan["counts"],
            "encoder": ENCODER, "revision": REVISION, "model_files_sha256": plan["model_files_sha256"],
            "device": plan["device"], "setup_seconds": setup_seconds, "work": work, "progress": progress,
            "process_lifetime_peak_rss_bytes": int(peak_rss), "initial_device_memory": initial_memory,
            "device_memory_scope": "Maximum sampled live MPS allocations after forwards; not an allocator high-water guarantee",
            "projected_full_embedding_bytes": plan["projected_embedding_bytes"], "parameter_training_steps": 0,
            "test_contents_accessed": False, "no_retry": True,
            "capacity_scope": "Fixed first 512 unique strings in train/dev encounter order; timing is not a representative sampling guarantee",
            "files": manifest(out)}
        receipt["cache_file_bytes"] = sum(v["bytes"] for v in receipt["files"].values())
        if phase == "capacity":
            # All final hashing is included in observed fixed overhead. Reserve a
            # conservative 1 MiB for each NPY header/final receipt in disk sizing.
            source_bytes = sum(v["bytes"] for k, v in receipt["files"].items()
                               if k.startswith("sources/") or k in ("started.json", "original-encoder-source.py"))
            estimated_bytes = (plan["projected_embedding_bytes"] + plan["padded_index_bytes"]
                               + (plan_dir / "index.json").stat().st_size + source_bytes + 3 * 1024**2)
            receipt["projection"] = capacity_projection(work, full_profile,
                max(0., time.perf_counter() - start - work["encoder_seconds"]), estimated_bytes)
        receipt["wall_seconds"] = time.perf_counter() - start
        write(out / "completed.json", receipt)
    return sha(out / "completed.json")


def authenticate_cache(path, expected_completed_sha256):
    """Saved-only loader returning index, embeddings mmap, flat index mmap."""
    path = Path(path)
    receipt = authenticate(path, expected_completed_sha256)
    require(receipt["phase"] == "encode" and receipt["version"] == VERSION and receipt["test_contents_accessed"] is False,
            "Not a full completed joint cache")
    require(receipt["no_retry"] is True and 0 < receipt["wall_seconds"] <= LIMITS["encode"]
            and receipt["parameter_training_steps"] == 0, "Encoding phase/cap/training scope")
    expected_files = {"started.json", "plan.json", "original-encoder-source.py", "embeddings.npy", "indices.npy", "index.json"} | {
        "sources/" + name for name in receipt["source_sha256"]}
    require(set(receipt["files"]) == expected_files and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            == expected_files | {"completed.json"}, "Cache exact file membership")
    require(not any(p.is_symlink() for p in path.rglob("*")), "Symlink in cache")
    bind(path / "plan.json", receipt["plan_sha256"])
    plan = read(path / "plan.json")
    require(receipt["source_sha256"] == plan["source_sha256"] and receipt["runtime"] == plan["runtime"]
            and receipt["model_files_sha256"] == plan["model_files_sha256"] and set(plan["model_files_sha256"]) == MODEL_FILES
            and receipt["encoder"] == plan["encoder"] == ENCODER and receipt["revision"] == plan["revision"] == REVISION,
            "Cache plan/model/source identity")
    for label, item in plan["inputs"].items():
        require(receipt[label + "_completed_sha256"] == item["completed_sha256"], "Cache parent identity")
    for name, digest in plan["source_sha256"].items():
        bind(safe(path / "sources", name), digest)
    bind(path / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    capacity_path = safe(ROOT, receipt["capacity_path"])
    capacity = authenticate(capacity_path, receipt["capacity_completed_sha256"])
    require(capacity["phase"] == "capacity" and capacity["no_retry"] is True and 0 < capacity["wall_seconds"] <= 120
            and capacity["plan_sha256"] == receipt["plan_sha256"] and capacity["source_sha256"] == receipt["source_sha256"]
            and capacity["runtime"] == receipt["runtime"] and capacity["model_files_sha256"] == receipt["model_files_sha256"]
            and capacity["unique_texts_encoded"] == min(512, plan["unique_texts"]), "Parent capacity identity/scope")
    profile = read(capacity_path / "token-profile.json")
    projection = capacity["projection"]
    require(projection == capacity_projection(capacity["work"], profile, projection["observed_nonencoding_seconds"],
            projection["projected_cache_bytes"]) and projection["encoding_permitted"] is True, "Capacity admission arithmetic")
    for key in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "overlength_texts_chunked", "truncated_tokens"):
        require(receipt["work"][key] == profile[key], "Cache token coverage")
    require(receipt["progress"]["encoder_calls_attempted"] == receipt["progress"]["encoder_calls_returned"] == profile["encoder_calls"]
            and receipt["progress"]["encoded_sequences"] == profile["encoder_sequences"]
            and receipt["progress"]["encoded_texts"] == plan["unique_texts"], "Cache call/text coverage")
    require({"index.json", "indices.npy", "embeddings.npy", "started.json"} <= set(receipt["files"]), "Missing cache members")
    index = read(path / "index.json")
    embeddings = np.load(path / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    indices = np.load(path / "indices.npy", allow_pickle=False, mmap_mode="r")
    require(index["version"] == VERSION and index["template"] == TEMPLATE and index["embedding_width"] == WIDTH
            and index["padding_index"] == -1, "Cache schema")
    require(embeddings.dtype == np.float32 and embeddings.shape == (index["unique_texts"], WIDTH)
            and receipt["unique_texts_encoded"] == index["unique_texts"], "Embedding shape/type")
    require(indices.dtype == np.int64 and indices.shape == (index["index_count"],)
            and ((indices >= -1) & (indices < len(embeddings))).all(), "Index shape/range")
    require(index["counts"] == receipt["counts"] == plan["counts"] and index["unique_texts"] == plan["unique_texts"]
            and index["index_count"] == plan["index_count"] and receipt["full_unique_texts"] == plan["unique_texts"], "Cache aggregate counts")
    require(sum(v["bytes"] for v in receipt["files"].values()) == receipt["cache_file_bytes"]
            and sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= DISK_CAP, "Cache disk count/cap")
    for offset in range(0, len(embeddings), 8192):
        block = embeddings[offset:offset + 8192]
        require(np.isfinite(block).all() and np.allclose(np.linalg.norm(block, axis=1), 1., atol=2e-6, rtol=0),
                "Invalid normalized cache embeddings")
    return index, embeddings, indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    f = subs.add_parser("freeze")
    for name in ("packet", "lexical", "data", "protocol", "packet-sha256", "lexical-sha256", "data-sha256", "protocol-sha256", "out"):
        f.add_argument("--" + name, required=True)
    f.add_argument("--device", choices=("cpu", "mps"), required=True)
    for phase in ("capacity", "encode"):
        command = subs.add_parser(phase)
        for name in ("plan", "plan-sha256", "out"):
            command.add_argument("--" + name, required=True)
        command.add_argument("--capacity", required=phase == "encode")
        command.add_argument("--capacity-sha256", required=phase == "encode")
    args = parser.parse_args()
    digest = freeze(args) if args.phase == "freeze" else encode(args)
    print(json.dumps({"status": "completed", "completed_sha256": digest}))
