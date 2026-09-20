"""Freeze and encode exact existing TRAIN schema strings; no inference on import.

Raw-token encoding reuses hash-pinned primitives. Schema/candidate selection
depends only on the fixed typed-study row membership and public query identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import signal
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/prepare_dialogue_tokens.py"
HELPER_SHA = "fa630dfb96d3b37184045703981dad6054813e03c2fcb948c14ca98ffda9b409"
raw = (ROOT / HELPER).read_bytes()
if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
    raise ValueError("Frozen token primitive changed")
tokens = types.ModuleType("pinned_schema_token_primitives")
tokens.__file__ = str(ROOT / HELPER)
exec(compile(raw, tokens.__file__, "exec"), tokens.__dict__)  # noqa: S102 - exact pinned local source
old = tokens.old
require, sha, read, write, bind, safe = old.require, old.sha, old.read, old.write, old.bind, old.safe
VERSION = "dialogue-schema-token-v1"
SOURCES = (*tokens.SOURCES, "scripts/prepare_dialogue_schema_tokens.py", "tests/test_prepare_dialogue_schema_tokens.py")
WALL_CAP = 120.
DISK_CAP = 256 * 1024**2
QUERY_COUNT = 53
LAYOUT_FILES = {"texts.jsonl", "index.json"}


def query_scope(rows, fit_ids, evaluation_ids):
    """Whole JSON records may be decoded; only these public/membership keys read."""
    require(all(type(v) is list and v == sorted(set(v)) and all(type(i) is int and i >= 0 for i in v)
                for v in (fit_ids, evaluation_ids)), "Strict frozen row membership")
    wanted = set(fit_ids) | set(evaluation_ids)
    require(not set(fit_ids) & set(evaluation_ids) and wanted, "Disjoint nonempty split")
    found, queries = set(), set()
    for row in rows:
        i = row["row_index"]
        require(type(i) is int and i >= 0, "Strict public row index")
        if i not in wanted:
            require(not (row["split"] == "train" and row["admission"] == "admitted"), "Uncovered admitted TRAIN row")
            continue
        require(i not in found and row["split"] == "train" and row["admission"] == "admitted", "Fixed TRAIN admission")
        require(type(row["query_index"]) is int and row["query_index"] >= 0, "Strict supplied query index")
        found.add(i)
        queries.add(row["query_index"])
    require(found == wanted, "Exact split row coverage")
    return sorted(queries)


def build_layout(packet, train_catalog, query_ids):
    """Every exact query string and ALL declared candidates, never target-selected."""
    require(type(query_ids) is list and query_ids == sorted(set(query_ids)) and query_ids
            and all(type(i) is int and 0 <= i < len(packet["queries"]) for i in query_ids), "Query identity set")
    catalog = {q["query_id"]: q for q in train_catalog}
    require(len(catalog) == len(train_catalog), "Unique TRAIN catalog")
    texts, lookup, original, inverse, entries = [], {}, [], {}, []
    def add(text, feature_id):
        require(type(text) is str and text.strip() and type(feature_id) is int and feature_id >= 0, "Public text/feature identity")
        require(inverse.setdefault(feature_id, text) == text, "One feature ID has multiple strings")
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
            original.append(feature_id)
        index = lookup[text]
        require(original[index] == feature_id, "Exact strings must share original feature ID")
        return index
    for qi in query_ids:
        q = packet["queries"][qi]
        require(q["split"] == "train", "No DEV/test schema encoding")
        c = catalog[q["id"]]
        prefix = f"Service: {c['service_description']}\nSlot: {c['slot_description']}"
        require(c["query_text"] == prefix and (q["service"], q["slot"]) == (c["service"], c["slot"]), "Original query string")
        candidates = c["candidates"]
        require(3 <= len(candidates) <= 12 and len(q["candidates"]) == len(candidates)
                and q["candidate_ids"] == [v["id"] for v in candidates]
                and q["candidate_values"] == [v["value"] for v in candidates]
                and q["candidate_ids"][:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"], "Original candidate order")
        expected = [prefix + "\nValue: NOT_MENTIONED (no constraint stated)", prefix + "\nValue: DONTCARE (no preference)"]
        expected += [prefix + "\nValue: " + v["value"] for v in candidates[2:]]
        require([v["text"] for v in candidates] == expected, "Exact full schema/value strings")
        query_text_index = add(c["query_text"], q["text"])
        candidate_text_indices = [add(c["text"], i) for c, i in zip(candidates, q["candidates"], strict=True)]
        entries.append({"query_index": qi, "query_id": q["id"], "service": q["service"], "slot": q["slot"],
                        "query_token_id": query_text_index, "candidate_token_ids": candidate_text_indices,
                        "query_feature_index": q["text"], "candidate_feature_indices": q["candidates"],
                        "candidate_ids": q["candidate_ids"]})
    return texts, {"version": VERSION, "width": 384, "split": "train", "queries": entries,
                   "query_indices": query_ids, "unique_texts": len(texts), "original_feature_indices": original,
                   "query_count": len(entries), "candidate_occurrences": sum(len(q["candidate_ids"]) for q in entries),
                   "scope": "Exact supplied query and full schema/value candidate strings, all candidates; no target-based text or selection"}


@contextmanager
def attempt(out, phase, request):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    progress = {"phase": phase, "encoder_calls_attempted": 0, "encoder_calls_returned": 0,
                "encoded_sequences": 0, "encoded_texts": 0}
    def check():
        if time.perf_counter() - start > WALL_CAP:
            raise TimeoutError("Fixed schema-cache wall cap exceeded")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= DISK_CAP, "Schema-cache disk cap")
    def expired(_signal, _frame):
        raise TimeoutError("Schema-cache alarm expired")
    previous = None
    try:
        write(out / "started.json", {"status": "started", "phase": phase, "request": request,
              "wall_cap_seconds": WALL_CAP, "disk_cap_bytes": DISK_CAP, "source_sha256": sha(__file__), "no_retry": True})
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.001, WALL_CAP - (time.perf_counter() - start)))
        yield out, start, progress, check
        check()
    except BaseException as error:
        try:
            if (out / "completed.json").exists():
                (out / "completed.json").rename(out / "late-completion.json")
            write(out / "failed.json", {"status": "failed", "error": repr(error), "request": request,
                  "wall_seconds": time.perf_counter() - start, "progress": progress, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain original error
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation: " + repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def parents(inputs):
    result = {key: old.authenticate(safe(ROOT, item["path"]), item["completed_sha256"]) for key, item in inputs.items()}
    require(set(result) == {"packet", "data", "prepared", "study"}, "Fixed input classes")
    require(result["data"]["test_contents_accessed"] is False, "No official test content")
    study = read(safe(ROOT, inputs["study"]["path"]) / "plan.json")
    require(sha(safe(ROOT, inputs["study"]["path"]) / "plan.json") == result["study"]["plan_sha256"]
            and study["prepared_completed_sha256"] == inputs["prepared"]["completed_sha256"], "Fixed typed split lineage")
    for key in ("packet", "data"):
        identity = str((safe(ROOT, inputs[key]["path"]) / "completed.json").resolve())
        require(result["prepared"]["authenticated_inputs"][identity]["sha256"] == inputs[key]["completed_sha256"],
                "Prepared metadata parent identity")
    encoder = read(safe(ROOT, inputs["packet"]["path"]) / "encoder-plan.json")
    require(encoder["encoder"] == old.ENCODER and encoder["revision"] == old.REVISION
            and encoder["dtype"] == "float32" and encoder["chunk_tokens"] == 254 and encoder["batch_size"] == 128
            and encoder["input_sha256"]["completed.json"] == inputs["data"]["completed_sha256"], "Original encoder lineage")
    bind(safe(ROOT, inputs["packet"]["path"]) / "encoder-source.py", encoder["source_sha256"])
    return result, study, encoder


def layout_from_inputs(inputs, study):
    prepared = safe(ROOT, inputs["prepared"]["path"])
    with (prepared / "rows.jsonl").open() as stream:
        query_ids = query_scope((json.loads(line) for line in stream), study["fit_row_indices"], study["evaluation_row_indices"])
    packet = read(safe(ROOT, inputs["packet"]["path"]) / "packet.json")
    catalog = read(safe(ROOT, inputs["data"]["path"]) / "catalog.json")["train"]
    texts, index = build_layout(packet, catalog, query_ids)
    require(index["query_count"] == QUERY_COUNT, "Exact admitted TRAIN schema count")
    return texts, index


def finish(out, start, record, check):
    record["files"] = old.manifest(out)
    record["payload_bytes"] = sum(v["bytes"] for v in record["files"].values())
    record["wall_seconds"] = time.perf_counter() - start
    record["wall_scope"] = "Exclusive creation through validation/encoding/flush/parity/payload hashing; final completion write excluded, terminal cap check still applies"
    completion_size = len((json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    require(record["payload_bytes"] + completion_size <= DISK_CAP, "Final cache disk cap")
    check()
    write(out / "completed.json", record)


def freeze(args):
    with attempt(args.out, "freeze", {k: str(v) for k, v in vars(args).items()}) as (out, start, _progress, check):
        inputs = {key: {"path": str(Path(getattr(args, key)).resolve().relative_to(ROOT)),
                        "completed_sha256": getattr(args, key + "_sha256")} for key in ("packet", "data", "prepared", "study")}
        docs, study, encoder = parents(inputs)
        require(args.device == "mps", "MPS fixed encoder device")
        protocol = str(Path(args.protocol).resolve().relative_to(ROOT))
        bind(ROOT / protocol, args.protocol_sha256)
        mapping = dict(study["source_sha256"])
        for name in SOURCES:
            digest = sha(ROOT / name)
            require(name not in mapping or mapping[name] == digest, "Inherited source drift")
            mapping[name] = digest
        for name, digest in docs["data"]["implementation_sha256"].items():
            require(name not in mapping or mapping[name] == digest, "Data implementation drift")
            mapping[name] = digest
        mapping[protocol] = args.protocol_sha256
        old.sources(out, mapping)
        shutil.copyfile(safe(ROOT, inputs["packet"]["path"]) / "encoder-source.py", out / "original-encoder-source.py")
        old.model_paths(encoder["model_files_sha256"])
        texts, index = layout_from_inputs(inputs, study)
        with (out / "texts.jsonl").open("x") as stream:
            for value in texts:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        write(out / "index.json", index)
        plan = {"version": VERSION, "device": "mps", "dtype": "float32", "width": 384,
                "encoder": old.ENCODER, "revision": old.REVISION, "chunk_tokens": 254, "batch_size": 128,
                "wall_cap_seconds": WALL_CAP, "disk_cap_bytes": DISK_CAP, "pooling_max_abs_tolerance": 2e-5,
                "inputs": inputs, "source_sha256": mapping, "runtime": old.runtime(),
                "model_files_sha256": encoder["model_files_sha256"], "original_encoder_source_sha256": encoder["source_sha256"],
                "protocol_path": protocol, "protocol_sha256": args.protocol_sha256,
                "layout_files": {name: sha(out / name) for name in LAYOUT_FILES},
                "query_count": QUERY_COUNT, "unique_texts": len(texts), "candidate_occurrences": index["candidate_occurrences"],
                "no_retry": True, "scope": index["scope"]}
        write(out / "plan.json", plan)
        parents(inputs)
        for name, digest in mapping.items():
            bind(ROOT / name, digest)
        finish(out, start, {"status": "completed", "phase": "freeze", "version": VERSION,
              "plan_sha256": sha(out / "plan.json"), "source_sha256": mapping, "model_calls": 0, "no_retry": True}, check)
    return sha(out / "completed.json")


def validate_plan(path, expected):
    path = Path(path)
    bind(path, expected)
    plan = read(path)
    require(plan["version"] == VERSION and plan["device"] == "mps" and plan["dtype"] == "float32"
            and plan["width"] == 384 and plan["chunk_tokens"] == 254 and plan["batch_size"] == 128
            and plan["encoder"] == old.ENCODER and plan["revision"] == old.REVISION
            and plan["wall_cap_seconds"] == WALL_CAP and plan["disk_cap_bytes"] == DISK_CAP
            and plan["pooling_max_abs_tolerance"] == 2e-5 and plan["query_count"] == QUERY_COUNT
            and plan["no_retry"] is True and plan["runtime"] == old.runtime(), "Frozen schema-cache recipe/runtime")
    done = read(path.parent / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == expected,
            "Completed exclusive metadata freeze")
    old.authenticate(path.parent, sha(path.parent / "completed.json"))
    require(set(plan["layout_files"]) == LAYOUT_FILES, "Fixed layout fields")
    expected_files = {"started.json", "plan.json", "original-encoder-source.py", *LAYOUT_FILES} | {"sources/" + n for n in plan["source_sha256"]}
    require(set(done["files"]) == expected_files and {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}
            == expected_files | {"completed.json"}, "Exact freeze closure")
    for name, digest in plan["source_sha256"].items():
        bind(ROOT / name, digest)
        bind(safe(path.parent / "sources", name), digest)
    require(all(name in plan["source_sha256"] for name in SOURCES) and plan["source_sha256"][HELPER] == HELPER_SHA, "Required source closure")
    for name, digest in plan["layout_files"].items():
        bind(path.parent / name, digest)
    bind(path.parent / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    bind(ROOT / plan["protocol_path"], plan["protocol_sha256"])
    _, study, encoder = parents(plan["inputs"])
    require(plan["model_files_sha256"] == encoder["model_files_sha256"], "Original six encoder identities")
    texts, index = layout_from_inputs(plan["inputs"], study)
    require(index == read(path.parent / "index.json") and texts == [json.loads(s) for s in (path.parent / "texts.jsonl").read_text().splitlines()],
            "Frozen exact schema layout reproduction")
    return plan


def encode(args, *, backend_factory=tokens.TokenEncoder):
    with attempt(args.out, "encode", {k: str(v) for k, v in vars(args).items()}) as (out, start, progress, check):
        plan = validate_plan(args.plan, args.plan_sha256)
        old.model_paths(plan["model_files_sha256"])
        old.sources(out, plan["source_sha256"])
        for name in ("plan.json", "original-encoder-source.py", "index.json"):
            shutil.copyfile(Path(args.plan).parent / name, out / name)
        index = read(out / "index.json")
        texts = [json.loads(line) for line in (Path(args.plan).parent / "texts.jsonl").read_text().splitlines()]
        source = np.load(safe(ROOT, plan["inputs"]["packet"]["path"]) / "features.npy", mmap_mode="r", allow_pickle=False)
        require(source.dtype == np.float32 and source.ndim == 2 and source.shape[1] == 384, "Original feature matrix")
        vectors = source[np.asarray(index["original_feature_indices"], np.int64)]
        check()
        stamp = time.perf_counter()
        backend = backend_factory(plan)
        backend.sync()
        setup_seconds = time.perf_counter() - stamp
        profile, lengths = old.token_profile(texts, backend, check)
        offsets, _, chunks = tokens.chunks_for_lengths(lengths)
        require(int(offsets[-1]) * (384 * 4 + 4) + len(chunks) * 8 + 1024**2 < DISK_CAP, "Pre-forward raw output admission")
        np.save(out / "text-token-counts.npy", lengths, allow_pickle=False)
        write(out / "token-profile.json", profile)
        work, equivalence, _ = tokens.encode_tokens(texts, lengths, backend, out, vectors, progress, check)
        tokens.validate_work({"work": work, "pooling_equivalence": equivalence, "progress": progress}, lengths)
        validate_plan(args.plan, args.plan_sha256)
        old.model_paths(plan["model_files_sha256"])
        finish(out, start, {"status": "completed", "phase": "encode", "version": VERSION,
              "plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "inputs": plan["inputs"],
              "runtime": plan["runtime"], "model_files_sha256": plan["model_files_sha256"],
              "query_count": index["query_count"], "candidate_occurrences": index["candidate_occurrences"],
              "unique_texts": len(texts), "work": work, "progress": progress, "pooling_equivalence": equivalence,
              "setup_seconds": setup_seconds, "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
              "no_retry": True, "parameter_training_steps": 0, "test_contents_accessed": False,
              "scope": plan["scope"]}, check)
    return sha(out / "completed.json")


def authenticate_cache_metadata(path, completed_sha256):
    """Authenticate opaque float files and return index/offsets; decode integers only."""
    path = Path(path)
    receipt = old.authenticate(path, completed_sha256)
    bind(path / "plan.json", receipt["plan_sha256"])
    plan = read(path / "plan.json")
    require(receipt["status"] == "completed" and receipt["phase"] == "encode" and receipt["version"] == plan["version"] == VERSION
            and 0 < receipt["wall_seconds"] <= WALL_CAP and receipt["no_retry"] is True
            and receipt["parameter_training_steps"] == 0 and receipt["test_contents_accessed"] is False
            and receipt["source_sha256"] == plan["source_sha256"] and receipt["inputs"] == plan["inputs"]
            and receipt["model_files_sha256"] == plan["model_files_sha256"] and receipt["runtime"] == plan["runtime"], "Cache receipt identity")
    require(plan["encoder"] == old.ENCODER and plan["revision"] == old.REVISION
            and plan["device"] == "mps" and plan["dtype"] == "float32" and plan["width"] == 384
            and plan["chunk_tokens"] == 254 and plan["batch_size"] == 128
            and plan["wall_cap_seconds"] == WALL_CAP and plan["disk_cap_bytes"] == DISK_CAP
            and plan["pooling_max_abs_tolerance"] == 2e-5 and plan["no_retry"] is True,
            "Cache frozen numerical recipe")
    expected = {"started.json", "plan.json", "original-encoder-source.py", "index.json", "token-profile.json", "text-token-counts.npy", *tokens.TOKEN_FILES}
    expected |= {"sources/" + n for n in plan["source_sha256"]}
    require(set(receipt["files"]) == expected and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} == expected | {"completed.json"}, "Exact schema-token payload closure")
    require(all(name in plan["source_sha256"] for name in SOURCES)
            and plan["source_sha256"][HELPER] == HELPER_SHA, "Cache source closure")
    for name, digest in plan["source_sha256"].items():
        bind(ROOT / name, digest)
        bind(safe(path / "sources", name), digest)
    bind(path / "original-encoder-source.py", plan["original_encoder_source_sha256"])
    bind(path / "index.json", plan["layout_files"]["index.json"])
    _, study, encoder = parents(plan["inputs"])
    require(plan["model_files_sha256"] == encoder["model_files_sha256"], "Cache original model identities")
    require(all(plan["source_sha256"].get(n) == digest for n, digest in study["source_sha256"].items()),
            "Inherited study source closure")
    _, exact_index = layout_from_inputs(plan["inputs"], study)
    index = read(path / "index.json")
    require(index == exact_index and receipt["query_count"] == index["query_count"] == QUERY_COUNT
            and receipt["unique_texts"] == index["unique_texts"] == plan["unique_texts"]
            and receipt["candidate_occurrences"] == index["candidate_occurrences"], "Exact schema identity")
    arrays = {name: np.load(path / name, mmap_mode="r", allow_pickle=False) for name in ("offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy", "text-token-counts.npy")}
    lengths = arrays["text-token-counts.npy"]
    require(lengths.shape == (index["unique_texts"],), "All unique schema strings")
    offsets, chunk_offsets, chunks = tokens.chunks_for_lengths(lengths)
    for name, exact in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        require(arrays[name].dtype == np.int64 and np.array_equal(arrays[name], exact), "Chunk geometry")
    profile = read(path / "token-profile.json")
    require(all(profile[k] == v for k, v in tokens.profile_counts(lengths).items())
            and type(profile["tokenization_seconds"]) in (int, float) and math.isfinite(profile["tokenization_seconds"])
            and profile["tokenization_seconds"] >= 0, "Sequential token profile")
    for name, shape in (("tokens.npy", (int(offsets[-1]), 384)), ("priors.npy", (int(offsets[-1]),))):
        with (path / name).open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            require(version == (1, 0), "NumPy cache header version")
            actual, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            require(actual == shape and dtype == np.dtype("float32") and not fortran
                    and stream.tell() + math.prod(shape) * 4 == (path / name).stat().st_size, "Opaque float header/byte closure")
    tokens.validate_work(receipt, lengths)
    require(receipt["payload_bytes"] == sum(v["bytes"] for v in receipt["files"].values())
            and sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= DISK_CAP, "Actual output bytes")
    return index, arrays["offsets.npy"]


def authenticate_cache(path, completed_sha256):
    """Return index, raw tokens, offsets, priors after full numerical parity checks."""
    path = Path(path)
    index, offsets = authenticate_cache_metadata(path, completed_sha256)
    receipt, plan = read(path / "completed.json"), read(path / "plan.json")
    arrays = {name: np.load(path / name, mmap_mode="r", allow_pickle=False) for name in tokens.TOKEN_FILES}
    lengths = np.load(path / "text-token-counts.npy", allow_pickle=False)
    chunk_offsets, chunks = arrays["chunk-offsets.npy"], arrays["chunk-lengths.npy"]
    raw, priors = arrays["tokens.npy"], arrays["priors.npy"]
    require(raw.dtype == priors.dtype == np.float32 and raw.shape == (int(offsets[-1]), 384)
            and priors.shape == (int(offsets[-1]),) and np.isfinite(raw).all() and np.isfinite(priors).all(), "Finite raw tokens")
    source = np.load(safe(ROOT, plan["inputs"]["packet"]["path"]) / "features.npy", mmap_mode="r", allow_pickle=False)
    cursor, maximum = 0, 0.
    for i, length in enumerate(lengths):
        for n in chunks[chunk_offsets[i]:chunk_offsets[i + 1]]:
            require(np.all(priors[cursor:cursor + n + 2] == np.float32(n / (int(length) * (n + 2)))), "Exact chunk priors")
            cursor += n + 2
        a, b = int(offsets[i]), int(offsets[i + 1])
        pooled = (raw[a:b] * priors[a:b, None]).sum(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        require(math.isfinite(norm) and norm > 1e-12, "Pooled mean norm")
        maximum = max(maximum, float(np.abs(pooled / norm - source[index["original_feature_indices"][i]]).max()))
    tokens.validate_work(receipt, lengths)
    require(maximum <= 2e-5 and maximum == receipt["pooling_equivalence"]["max_abs_error"], "Recomputed pooled parity")
    require(receipt["payload_bytes"] == sum(v["bytes"] for v in receipt["files"].values())
            and sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) <= DISK_CAP, "Actual output bytes")
    return index, raw, arrays["offsets.npy"], priors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    frozen = sub.add_parser("freeze")
    for name in ("packet", "data", "prepared", "study", "protocol"):
        frozen.add_argument("--" + name, required=True)
        frozen.add_argument("--" + name + "-sha256", required=True)
    frozen.add_argument("--device", choices=("mps",), required=True)
    frozen.add_argument("--out", required=True)
    encoded = sub.add_parser("encode")
    for name in ("plan", "plan-sha256", "out"):
        encoded.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps({"completed_sha256": freeze(args) if args.phase == "freeze" else encode(args)}))
