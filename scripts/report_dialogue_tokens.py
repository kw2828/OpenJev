"""Saved-only paired observation report; no encoder/model/weight deserialization.

Exact pinned V2 ledger and original metric helpers are reused. Cache geometry,
observation work, subgroup definitions and fifteen criteria are checked here.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOINT_PATH = "scripts/report_dialogue_joint.py"
JOINT_SHA = "f09f85a5c2bfad37d9c21aeab8886b07bd9f75f7f117408c01c159479151eef8"
_source = (ROOT / JOINT_PATH).read_bytes()
if hashlib.sha256(_source).hexdigest() != JOINT_SHA:
    raise ValueError("Frozen joint reporter changed")
joint_report = types.ModuleType("token_frozen_joint_report")
joint_report.__file__ = str(ROOT / JOINT_PATH)
exec(compile(_source, joint_report.__file__, "exec"), joint_report.__dict__)  # noqa: S102 - exact pinned local source
v2, base, np = joint_report.v2, joint_report.base, joint_report.np
V2_PATH, V2_SHA = joint_report.V2_PATH, joint_report.V2_SHA
STUDY, VERSION = "dialogue-token-v1", "dialogue-token-memory-v1"
ARMS = ("slot_readout", "slot_scalar", "candidate_readout", "candidate_scalar")
HEADS, SEEDS, PANELS = ("readout", "scalar"), v2.SEEDS, v2.PANELS
CONFIG = {**v2.CONFIG, "methods": list(ARMS), "heads": list(HEADS), "observations": ["slot", "candidate"]}
CHECKS = {"heads": list(HEADS), "required_complete_fits": 12, "unseen_true_gain": .10,
          "unseen_dontcare_gain": .10, "unseen_macro_gain": .02, "maximum_seen_macro_deficit": .01,
          "unseen_micro_nll_nonworse": True, "strict_unseen_macro_wins": 2,
          "maximum_unseen_revision_deficit": .01, "total_head_checks": 14}
CAP, TOLERANCE = 3600., 2e-6
OLD_PLAN = "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905"
OLD_COMPLETED = "768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4"
DATA_COMPLETED = "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12"
SOURCES = v2.SOURCES | {"src/openjev/research/dialogue_joint_memory.py", "tests/test_dialogue_joint_memory.py",
    "scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py", JOINT_PATH, "tests/test_report_dialogue_joint.py",
    "src/openjev/research/dialogue_token_memory.py", "tests/test_dialogue_token_memory.py",
    "scripts/prepare_dialogue_tokens.py", "tests/test_prepare_dialogue_tokens.py", "scripts/study_dialogue_tokens.py",
    "tests/test_study_dialogue_tokens.py", "scripts/report_dialogue_tokens.py", "tests/test_report_dialogue_tokens.py",
    "research/dialogue-token-protocol.md"}
ENCODER, REVISION = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CACHE_VERSION = "dialogue-token-minilm-v1"
TEMPLATE = "'System: ' + previous_system_text + '\\nUser: ' + current_user_text"
MODEL_FILES = {"config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.txt"}
DISK_CAP = 4 * 1024**3
GROUPS = ("assigned_true", "assigned_false", "dontcare")


def bind(path, digest, inventory):
    base.bind(Path(path), digest)
    inventory[str(Path(path).resolve())] = digest


def bind_tree(directory, digest, inventory, *, exact=None):
    directory = Path(directory)
    base.clean(directory)
    base.require(not any((directory / n).exists() for n in ("late-completion.json", "cleanup-error.json")), "Failed parent")
    bind(directory / "completed.json", digest, inventory)
    done = base.read(directory / "completed.json")
    base.require(done["status"] == "completed" and done["files"], "Incomplete parent")
    for name, entry in done["files"].items():
        path = base.safe(directory, name)
        if isinstance(entry, dict):
            base.require(set(entry) == {"sha256", "bytes"} and type(entry["bytes"]) is int
                         and path.stat().st_size == entry["bytes"], "Manifest size/schema")
            entry = entry["sha256"]
        bind(path, entry, inventory)
    if exact is not None:
        base.require(set(done["files"]) == exact, "Parent payload membership")
    return done


def execution_members(run):
    expected = {"plan.json", "started.json", "completed.json", "references.npz"} | {
        f"fits/{a}-{s}/{n}" for s in SEEDS for a in ARMS
        for n in ("completed.json", "weights.pt", "dev-predictions.npz", "batches.jsonl")}
    paths = list(run.rglob("*"))
    base.require(not any(p.is_symlink() for p in paths)
                 and {p.relative_to(run).as_posix() for p in paths if p.is_file()} == expected,
                 "Exact 52 execution files required")
    return expected


def original_records(plan, root, inventory):
    base.require(plan["old_plan_sha256"] == OLD_PLAN and plan["old_completed_sha256"] == OLD_COMPLETED, "V2 lineage pins")
    folder = base.safe(root, plan["paths"]["old_study"])
    bind(folder / "plan.json", OLD_PLAN, inventory)
    bind(folder / "completed.json", OLD_COMPLETED, inventory)
    prior, done = base.read(folder / "plan.json"), base.read(folder / "completed.json")
    base.require(prior["study"] == "dialogue-copy-v2" and prior["config"] == v2.CONFIG
                 and prior["practical_checks"] == v2.PRACTICAL_CHECKS and done["status"] == "completed"
                 and done["plan_sha256"] == OLD_PLAN and done["fit_count"] == 15, "V2 parent recipe/completion")
    for key in ("runtime", "packet_completed_sha256", "lexical_completed_sha256", "loss_counts", "loss_weights", "optimizer_updates_per_fit"):
        base.require(plan[key] == prior[key], "V2 input/recipe differs: " + key)
    base.require(set(prior["source_sha256"]) == v2.SOURCES
                 and all(plan["source_sha256"][k] == v for k, v in prior["source_sha256"].items()), "V2 source identity")
    names = [f"{m}-{s}" for s in SEEDS for m in v2.METHODS]
    base.require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "V2 parent fit order")
    result = {}
    for name, r in zip(names, done["fits"], strict=True):
        path = folder / "fits" / name / "completed.json"
        # The external completed digest authenticates this embedded receipt too.
        base.require(not path.is_symlink() and base.read(path) == r and r["status"] == "completed", "V2 fit receipt")
        bind(path, base.sha(path), inventory)
        result[name] = r
    return result


def cache_geometry(directory, packet, data, index, receipt):
    names = ("tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy", "context-indices.npy", "text-token-counts.npy")
    arrays = {n: np.load(directory/n, allow_pickle=False, mmap_mode="r") for n in names}
    raw, prior, lengths = arrays["tokens.npy"], arrays["priors.npy"], arrays["text-token-counts.npy"]
    base.require(index["version"] == CACHE_VERSION and index["template"] == TEMPLATE and index["width"] == 384
                 and lengths.dtype == np.int64 and lengths.shape == (index["unique_contexts"],) and (lengths > 0).all(), "Token length/schema")
    offsets, chunk_offsets, chunks = [0], [0], []
    for length in lengths:
        local = [min(254, int(length)-j) for j in range(0, int(length), 254)]
        chunks.extend(local)
        chunk_offsets.append(len(chunks))
        offsets.append(offsets[-1]+sum(local)+2*len(local))
    for n, expected in (("offsets.npy", offsets), ("chunk-offsets.npy", chunk_offsets), ("chunk-lengths.npy", chunks)):
        base.require(arrays[n].dtype == np.int64 and np.array_equal(arrays[n], expected), "Token offset/chunk reconstruction")
    base.require(raw.dtype == prior.dtype == np.float32 and raw.shape == (offsets[-1], 384) and prior.shape == (offsets[-1],), "Token payload geometry")
    for start in range(0, len(raw), 8192):
        base.require(np.isfinite(raw[start:start+8192]).all() and np.isfinite(prior[start:start+8192]).all()
                     and (prior[start:start+8192] > 0).all(), "Nonfinite/invalid raw token payload")
    source_ids = index["original_feature_indices"]
    source = np.load(packet / "features.npy", allow_pickle=False, mmap_mode="r")
    base.require(source.dtype == np.float32 and source.ndim == 2 and source.shape[1] == 384
                 and len(source_ids) == len(lengths) and len(set(source_ids)) == len(source_ids)
                 and all(type(i) is int and 0 <= i < len(source) for i in source_ids), "Original context identities")
    max_error, cursor = 0., 0
    for i, length in enumerate(lengths):
        for n in chunks[chunk_offsets[i]:chunk_offsets[i+1]]:
            expected_prior = np.float32(n/(int(length)*(n+2)))
            base.require(np.all(prior[cursor:cursor+n+2] == expected_prior), "Exact chunk pooling prior")
            cursor += n+2
        a, z = offsets[i:i+2]
        base.require(abs(float(prior[a:z].sum(dtype=np.float64))-1) <= TOLERANCE, "Context prior mass")
        pooled = (raw[a:z]*prior[a:z, None]).sum(0, dtype=np.float32)
        norm = float(np.linalg.norm(pooled))
        base.require(math.isfinite(norm) and norm > 1e-12, "Invalid pooled token mean")
        pooled /= norm
        base.require(np.isfinite(source[source_ids[i]]).all(), "Invalid original sentence vector")
        error = float(np.max(np.abs(pooled-source[source_ids[i]])))
        max_error = max(max_error, error)
        base.require(error <= 2e-5, "Original pooling equivalence")
    eq = receipt["pooling_equivalence"]
    base.require(eq == {"contexts_checked": len(lengths), "max_abs_error": max_error, "tolerance": 2e-5}, "Pooling witness differs from saved arithmetic")
    ids = arrays["context-indices.npy"]
    base.require(ids.dtype == np.int64 and ids.shape == (index["context_occurrences"],)
                 and ((ids >= 0) & (ids < len(lengths))).all() and set(index["cohorts"]) == {"train", "dev"}, "Context IDs")
    cursor, counts = 0, {}
    for split in ("train", "dev"):
        ds, entries = data["cohorts"][split], index["cohorts"][split]
        base.require(len(ds) == len(entries), "Token dialogue closure")
        count = Counter(dialogues=len(ds), public_turns=0, public_question_steps=0)
        for d, entry in zip(ds, entries, strict=True):
            qids = sorted({r["query"] for r in d["queries"]})
            n = len(d["turns"])
            base.require(entry == {"id": d["id"], "query_ids": qids, "offset": cursor, "shape": [n]}
                         and [source_ids[int(i)] for i in ids[cursor:cursor+n]] == d["turns"], "Original public context mapping")
            cursor += n
            count.update(public_turns=n, public_question_steps=n*len(qids))
        counts[split] = dict(count)
    base.require(cursor == len(ids) and counts == index["counts"] == receipt["counts"], "Context coverage/counts")
    sizes = [n+2 for n in chunks]
    full = base.read(directory / "token-profile.json")
    expected_profile = {"input_tokens": sum(map(int, lengths)), "encoder_tokens_with_special": sum(sizes),
        "encoder_sequences": len(sizes), "encoder_calls": math.ceil(len(sizes)/128),
        "padded_token_slots": sum(max(sizes[j:j+128])*len(sizes[j:j+128]) for j in range(0, len(sizes), 128)),
        "overlength_texts_chunked": int((lengths > 254).sum()), "truncated_tokens": 0}
    base.require(all(full[k] == v for k, v in expected_profile.items()), "Token profile does not match all context lengths")
    return {int(i): offsets[j+1]-offsets[j] for j, i in enumerate(source_ids)}


def authenticate_tokens(tokens, packet, lexical, data, plan, root, inventory):
    receipt = bind_tree(tokens, plan["tokens_completed_sha256"], inventory)
    expected = {"tokens.npy", "priors.npy", "offsets.npy", "chunk-offsets.npy", "chunk-lengths.npy", "started.json", "plan.json",
                "original-encoder-source.py", "token-profile.json", "text-token-counts.npy", "context-indices.npy", "index.json"} | {
        "sources/"+n for n in receipt["source_sha256"]}
    base.require(set(receipt["files"]) == expected and not any(p.is_symlink() for p in tokens.rglob("*"))
                 and {p.relative_to(tokens).as_posix() for p in tokens.rglob("*") if p.is_file()} == expected | {"completed.json"}, "Token cache exact membership")
    bind(tokens / "plan.json", receipt["plan_sha256"], inventory)
    prep, inherited = base.read(tokens / "plan.json"), base.read(packet / "encoder-plan.json")
    base.require(receipt["phase"] == "encode" and receipt["version"] == prep["version"] == CACHE_VERSION
                 and receipt["encoder"] == prep["encoder"] == ENCODER and receipt["revision"] == prep["revision"] == REVISION
                 and receipt["test_contents_accessed"] is False and receipt["parameter_training_steps"] == 0
                 and receipt["no_retry"] is True and 0 < receipt["wall_seconds"] <= 900, "Token encoding identity/scope/cap")
    base.require(receipt["model_files_sha256"] == prep["model_files_sha256"] == inherited["model_files_sha256"]
                 and set(prep["model_files_sha256"]) == MODEL_FILES, "Original six encoder files differ")
    base.require(prep["template"] == TEMPLATE and prep["dtype"] == "float32" and prep["device"] == receipt["device"] == "mps"
                 and prep["chunk_tokens"] == 254 and prep["batch_size"] == 128 and prep["width"] == 384
                 and prep["capacity_unique_contexts"] == 512 and prep["capacity_seconds"] == 120 and prep["encoding_seconds"] == 900
                 and prep["admission_seconds"] == 720 and prep["disk_cap_bytes"] == DISK_CAP and prep["pooling_max_abs_tolerance"] == 2e-5,
                 "Token preparation recipe")
    base.require(receipt["source_sha256"] == prep["source_sha256"] and receipt["runtime"] == prep["runtime"], "Preparation identity")
    for n, digest in prep["source_sha256"].items():
        bind(base.safe(root, n), digest, inventory)
        bind(base.safe(tokens / "sources", n), digest, inventory)
        if n in plan["source_sha256"]:
            base.require(digest == plan["source_sha256"][n], "Preparation/study source mismatch")
    base.require({"scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py", "scripts/prepare_dialogue_tokens.py",
                  "tests/test_prepare_dialogue_tokens.py", "research/dialogue-token-protocol.md"} <= set(prep["source_sha256"]), "Preparation source closure")
    bind(tokens / "original-encoder-source.py", inherited["source_sha256"], inventory)
    base.require(prep["original_encoder_source_sha256"] == inherited["source_sha256"], "Original encoder source")
    base.require(set(prep["inputs"]) == {"packet", "lexical", "data"}, "Preparation parents")
    for label, directory, digest in (("packet", packet, plan["packet_completed_sha256"]), ("lexical", lexical, plan["lexical_completed_sha256"]),
            ("data", base.safe(root, prep["inputs"]["data"]["path"]), DATA_COMPLETED)):
        base.require(base.safe(root, prep["inputs"][label]["path"]).resolve() == directory.resolve()
                     and prep["inputs"][label]["completed_sha256"] == receipt[label+"_completed_sha256"] == digest, "Preparation parent differs")
        if label == "data":
            bind_tree(directory, digest, inventory)
    bind(base.safe(root, prep["protocol_path"]), prep["protocol_sha256"], inventory)
    base.require(prep["protocol_path"] == "research/dialogue-token-protocol.md"
                 and prep["protocol_sha256"] == plan["source_sha256"][prep["protocol_path"]], "Preparation protocol differs")
    cap_path = base.safe(root, receipt["capacity_path"])
    cap = bind_tree(cap_path, receipt["capacity_completed_sha256"], inventory)
    base.require(cap["phase"] == "capacity" and cap["version"] == CACHE_VERSION and cap["no_retry"] is True
                 and 0 < cap["wall_seconds"] <= 120 and cap["unique_contexts_encoded"] == min(512, prep["unique_contexts"]), "Capacity scope")
    for k in ("plan_sha256", "source_sha256", "runtime", "model_files_sha256"):
        base.require(cap[k] == receipt[k], "Capacity/cache identity")
    for n in ("token-profile.json", "text-token-counts.npy"):
        base.require(n in cap["files"] and cap["files"][n]["sha256"] == receipt["files"][n]["sha256"], "Unbound or changed full profile")
    full, projection = base.read(tokens / "token-profile.json"), cap["projection"]
    for k in ("forward_transfer_seconds", "retention_seconds", "padded_token_slots", "retained_token_rows"):
        base.positive(cap["work"][k], "capacity units")
    forward = cap["work"]["forward_transfer_seconds"]/cap["work"]["padded_token_slots"]*full["padded_token_slots"]
    retention = cap["work"]["retention_seconds"]/cap["work"]["retained_token_rows"]*full["encoder_tokens_with_special"]
    overhead = projection["observed_other_seconds"]
    base.require(type(overhead) in (float, int) and math.isfinite(overhead) and overhead >= 0
                 and projection["projected_forward_transfer_seconds"] == forward and projection["projected_retention_seconds"] == retention
                 and projection["projected_total_seconds"] == forward+retention+overhead <= 720.
                 and projection["maximum_projected_seconds"] == 720. and projection["maximum_cache_bytes"] == DISK_CAP
                 and 0 < projection["projected_cache_bytes"] <= DISK_CAP and projection["encoding_permitted"] is True, "Three-bucket capacity arithmetic/admission")
    base.require(receipt["progress"]["encoder_calls_attempted"] == receipt["progress"]["encoder_calls_returned"] == full["encoder_calls"]
                 and receipt["progress"]["encoded_sequences"] == full["encoder_sequences"]
                 and receipt["progress"]["encoded_texts"] == receipt["unique_contexts_encoded"] == receipt["full_unique_contexts"] == prep["unique_contexts"], "Encoded context/call coverage")
    base.require(receipt["work"]["retained_token_rows"] == full["encoder_tokens_with_special"]
                 and receipt["work"]["input_tokens"] == full["input_tokens"] and receipt["work"]["truncated_tokens"] == 0
                 and receipt["work"]["padded_token_slots"] == full["padded_token_slots"], "Encoder token coverage")
    index = base.read(tokens / "index.json")
    base.require(index["unique_contexts"] == prep["unique_contexts"] and index["counts"] == prep["counts"]
                 and index["context_occurrences"] == prep["context_occurrences"], "Frozen token layout")
    for n in ("index.json", "context-indices.npy"):
        bind(tokens / n, prep["layout_files"][n], inventory)
    lengths = cache_geometry(tokens, packet, data, index, receipt)
    base.require(receipt["cache_file_bytes"] == sum(e["bytes"] for e in receipt["files"].values())
                 and sum(p.stat().st_size for p in tokens.rglob("*") if p.is_file()) <= DISK_CAP, "Cache bytes/cap")
    return {"encoding": receipt, "capacity": cap, "context_lengths": lengths}


def authenticate_inputs(run, packet, lexical, tokens, plan_sha, completed_sha, root, inventory):
    members = execution_members(run)
    bind(run / "plan.json", plan_sha, inventory)
    done = bind_tree(run, completed_sha, inventory, exact=members - {"completed.json"})
    plan = base.read(run / "plan.json")
    base.require(plan["study"] == done["study"] == STUDY and plan["implementation_version"] == VERSION
                 and plan["config"] == CONFIG and plan["practical_checks"] == CHECKS
                 and plan["wall_cap_seconds"] == CAP and plan["normalization_tolerance"] == TOLERANCE
                 and plan["execution_files"] == 52 and plan["expected_fits"] == 12, "Frozen study recipe")
    base.require(set(plan["source_sha256"]) == SOURCES and plan["source_sha256"][V2_PATH] == V2_SHA
                 and plan["source_sha256"]["scripts/report_dialogue_tokens.py"] == base.sha(__file__)
                 and plan["source_sha256"][JOINT_PATH] == JOINT_SHA, "Study source closure")
    for n, digest in plan["source_sha256"].items():
        bind(base.safe(root, n), digest, inventory)
    base.require(done["plan_sha256"] == plan_sha and done["fit_count"] == 12 and done["external_model_api_calls"] == 0
                 and done["encoder_calls"] == 0 and done["normalization_tolerance"] == TOLERANCE
                 and done["resume_authorized"] is False and done["tokens_completed_sha256"] == plan["tokens_completed_sha256"], "Completion identity")
    base.positive(done["wall_seconds"], "whole execution wall")
    base.require(done["wall_seconds"] <= CAP and base.read(run / "started.json") == {"status": "started",
        "plan_sha256": plan_sha, "runtime": plan["runtime"], "wall_cap_seconds": CAP}, "Execution timing/start identity")
    base.require(set(plan["paths"]) == {"packet", "lexical", "tokens", "old_study"}, "Study paths")
    for k, directory in (("packet", packet), ("lexical", lexical), ("tokens", tokens)):
        base.require(base.safe(root, plan["paths"][k]).resolve() == directory.resolve(), "Input path differs")
    parents = {}
    for label, directory, names in (("packet", packet, {"encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}),
                                  ("lexical", lexical, {"started.json", "index.json", "lexical.npy"})):
        parents[label] = bind_tree(directory, plan[label+"_completed_sha256"], inventory, exact=names)
    base.require(parents["lexical"]["packet_completed_sha256"] == plan["packet_completed_sha256"]
                 and parents["lexical"]["source_sha256"] and all(plan["source_sha256"].get(k) == v
                 for k, v in parents["lexical"]["source_sha256"].items()), "Lexical parent/source")
    data = base.read(packet / "packet.json")
    expected, counts = base.expected_records(data)
    base.require(parents["packet"]["cohorts"]["dev"]["queries"] == len(counts)
                 and parents["packet"]["cohorts"]["dev"]["dialogues"] == len(data["cohorts"]["dev"]), "Packet cohort counts")
    originals = original_records(plan, root, inventory)
    parents["tokens"] = authenticate_tokens(tokens, packet, lexical, data, plan, root, inventory)
    return plan, done, parents, data, expected, counts, originals


def subgroup_masks(data, expected):
    boolean = []
    values = []
    for qi, label in zip(expected["query"], expected["labels"], strict=True):
        q = data["queries"][int(qi)]
        candidates = q["candidate_values"]
        base.require(len(candidates) == len(q["candidate_ids"]), "Candidate value alignment")
        ontology = {v.casefold() for v in candidates[2:] if isinstance(v, str)}
        boolean.append(ontology == {"true", "false"})
        value = candidates[int(label)]
        values.append(value.casefold() if isinstance(value, str) else None)
    boolean, values = np.asarray(boolean), np.asarray(values, dtype=object)
    return {"assigned_true": boolean & (expected["labels"] >= 2) & (values == "true"),
            "assigned_false": boolean & (expected["labels"] >= 2) & (values == "false"),
            "dontcare": expected["labels"] == 1}


def metrics(arrays, masks):
    result = base.metrics(arrays)
    for panel in PANELS:
        selected = np.ones(len(arrays["labels"]), bool) if panel == "all" else arrays["unseen"] == (panel == "unseen")
        result[panel]["subgroups"] = {g: base.score(arrays["probabilities"], arrays["labels"], selected & mask)
                                       for g, mask in masks.items()}
    return result


def family_metrics(rows):
    def pool(items):
        base.require(len({r["count"] for r in items}) == 1, "Unequal fit support")
        return {"count_per_fit": items[0]["count"], "fit_count": len(SEEDS),
            **{k: base.mean_defined([r[k] for r in items]) for k in ("accuracy", "nll", "brier")},
            "zero_target_probabilities_all_fits": sum(r["zero_target_probabilities"] for r in items)}
    families = {}
    for arm in ARMS:
        family = {}
        for panel in PANELS:
            items = [rows[f"{arm}-{seed}"]["metrics"][panel] for seed in SEEDS]
            family[panel] = {"micro": pool([r["micro"] for r in items]), "revision": pool([r["revision"] for r in items]),
                "macro_three": {k: base.mean_defined([r["macro_three"][k] for r in items]) for k in ("accuracy", "nll", "brier")},
                "bins": {g: pool([r["bins"][g] for r in items]) for g in base.BINS},
                "strata": {g: pool([r["strata"][g] for r in items]) for g in v2.old.STRATA},
                "subgroups": {g: pool([r["subgroups"][g] for r in items]) for g in GROUPS}}
        families[arm] = family
    return families


def criteria(rows):
    """Exact equal-seed integer-ratio comparisons; NLL has no epsilon/floor."""
    checks = [{"name": "all_12_fits_completed", "passed": set(rows) == {f"{a}-{s}" for a in ARMS for s in SEEDS}}]
    def exact(row, panel, kind):
        r = row["metrics"][panel]
        if kind == "macro":
            return base.macro_exact(r)
        item = r["revision"] if kind == "revision" else r["subgroups"][kind]
        return Fraction(item["correct"], item["count"]) if item["count"] else None
    def means(arm, panel, kind):
        values = [exact(rows[f"{arm}-{s}"], panel, kind) for s in SEEDS]
        return None if any(x is None for x in values) else sum(values, Fraction()) / len(values)
    for head in HEADS:
        independent, tokens = "slot_"+head, "candidate_"+head
        complete = all(f"{a}-{s}" in rows for a in (independent, tokens) for s in SEEDS)
        for name, panel, kind, margin in (("unseen_true_gain", "unseen", "assigned_true", Fraction(1, 10)),
                ("unseen_dontcare_gain", "unseen", "dontcare", Fraction(1, 10)),
                ("unseen_macro_gain", "unseen", "macro", Fraction(1, 50)),
                ("seen_macro_noninferiority", "seen", "macro", Fraction(-1, 100)),
                ("unseen_revision_noninferiority", "unseen", "revision", Fraction(-1, 100))):
            left = means(tokens, panel, kind) if complete else None
            right = means(independent, panel, kind) if complete else None
            checks.append({"name": head+"/"+name, "passed": left is not None and right is not None and left-right >= margin,
                           "candidate": None if left is None else float(left), "slot": None if right is None else float(right),
                           "required_difference": float(margin)})
        values = [[rows[f"{a}-{s}"]["metrics"]["unseen"]["micro"]["nll"] for s in SEEDS]
                  for a in (independent, tokens)] if complete else [[None], [None]]
        a, b = map(base.mean_defined, values)
        checks.append({"name": head+"/unseen_micro_nll_nonworse", "passed": a is not None and b is not None and b <= a,
                       "candidate": b, "slot": a})
        paired = [(exact(rows[f"{tokens}-{s}"], "unseen", "macro"), exact(rows[f"{independent}-{s}"], "unseen", "macro"))
                  for s in SEEDS] if complete else []
        wins = sum(a is not None and b is not None and a > b for a, b in paired)
        checks.append({"name": head+"/strict_paired_unseen_macro_wins", "passed": complete and wins >= 2, "wins": wins, "required": 2})
    return {"passed": all(r["passed"] for r in checks), "checks_passed": sum(r["passed"] for r in checks),
            "checks_total": 15, "checks": checks}


def observation_work(dialogs, queries, mode, lengths):
    base.require(mode in ("slot", "candidate"), "Attention mode")
    b, t = len(dialogs), max(len(d["turns"]) for d in dialogs)
    qids = [sorted({r["query"] for r in d["queries"]}) for d in dialogs]
    q = max(map(len, qids))
    c = max(len(queries[i]["candidate_ids"]) for ids in qids for i in ids)
    sizes = [lengths[i] for d in dialogs for i in d["turns"]]
    length, valid = max(sizes), sum(sizes)
    padded, evidence = b*t*length, b*t*q*c
    return {"emitted_float32_scalars": padded*384, "emitted_bytes": padded*384*4,
        "emitted_token_mask_bytes": padded, "emitted_token_prior_bytes": padded*4,
        "raw_cache_token_bytes_read": valid*384*4, "raw_cache_prior_bytes_read": valid*4,
        "valid_token_positions": valid, "padded_token_positions": padded,
        "pooling_score_positions": evidence*length, "pooling_evidence_positions": evidence,
        "token_key_projection_positions": padded, "evidence_query_projection_positions": b*q*c,
        "turn_projection_positions": evidence, "schema_query_projection_positions": b*q,
        "schema_candidate_projection_positions": b*q*c,
        "real_public_turns": sum(len(d["turns"]) for d in dialogs),
        "real_question_updates": sum(len(d["turns"])*len(ids) for d, ids in zip(dialogs, qids, strict=True)),
        "real_candidate_updates": sum(len(d["turns"])*sum(len(queries[i]["candidate_ids"]) for i in ids)
                                      for d, ids in zip(dialogs, qids, strict=True))}


def audit_observations(ledger, dialogs, queries, mode, lengths):
    total = Counter()
    for row in ledger:
        expected = observation_work([dialogs[i] for i in row["indices"]], queries, mode, lengths)
        base.require(row["observation_work"] == expected, "Token observation work differs from layout")
        total.update(expected)
    return dict(total)


def configuration_from_parent(parent, mode):
    result = joint_report.configuration_from_parent(parent)
    result.update({"class": "MonitoredTokenMemory", "parameters": 173186,
        "implementation_version": VERSION, "normalized_head_parent_version": "dialogue-joint-memory-v1",
        "attention_mode": mode, "attention_width": 64,
        "adapter_parameter_names": ["token_key.weight", "evidence_query.weight"], "adapter_parameters": 73728,
        "parent_parameters": 99458,
        "attention": "softmax(log(token_prior) + dot(Wq[schema_pair], Wk[token])/sqrt(64)) over tokens only",
        "schema_pair": "[query;candidate]" if mode == "candidate" else "[query;query]",
        "pooling": "L2-normalize weighted raw token sum, eps=1e-12; zero vector remains zero",
        "token_prior": "Positive on token_mask, zero elsewhere; real-turn sum within 2e-6 of one",
        "observation_paths": {"tokens": "[B,T,L,D]; shared raw tokens, candidate-specific pooled evidence"},
        "observation_mode_binding": "Caller authenticates raw contextual tokens and chunk-weighted priors",
        "representation_control": "Same normalized head; two added bias-free projections; no encoder calls",
        "attention_work": "Forward: schema query B*Q*C once, token keys B*T*L, token logits B*T*Q*C*L; step repeats schema projection"})
    return result


MEMORY_SCOPE = "Process lifetime RSS high-water sampled at receipt; not a per-fit allocation or unique working set"
OBSERVATION_WORK_SCOPE = "Cumulative emitted/gathered tensor payload, not peak memory or measured storage I/O/cache misses"


def audit_saved(run, packet, lexical, tokens, plan_sha, completed_sha, root=ROOT):
    inventory = {}
    plan, done, parents, data, expected, counts, originals = authenticate_inputs(
        run, packet, lexical, tokens, plan_sha, completed_sha, root, inventory)
    train, dev = data["cohorts"]["train"], data["cohorts"]["dev"]
    loss_counts = Counter("0" if r["bin"] == "unmentioned_retention" else "1" if r["bin"] == "assigned_retention" else "2"
                          for d in train for r in d["queries"])
    updates = math.ceil(len(train)/CONFIG["batch_size"])*CONFIG["epochs"]
    base.require(set(loss_counts) == {"0", "1", "2"} and plan["loss_counts"] == dict(loss_counts)
                 and plan["loss_weights"] == [sum(loss_counts.values())/(3*loss_counts[str(i)]) for i in range(3)]
                 and plan["optimizer_updates_per_fit"] == updates, "Loss/update recipe")
    names = [f"{a}-{s}" for s in SEEDS for a in ARMS]
    base.require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names
                 and set(plan["expected_configurations"]) == set(ARMS), "Fit order/configuration membership")
    rows, paired_init, paired_orders = {}, {}, {}
    train_total, eval_total = v2.empty_invariants(), v2.empty_invariants()
    masks = subgroup_masks(data, expected)
    for name, record in zip(names, done["fits"], strict=True):
        folder = run / "fits" / name
        arm, seed = record["method"], record["seed"]
        mode, head = arm.rsplit("_", 1)
        base.require(base.read(folder / "completed.json") == record and record["status"] == "completed"
                     and record["head"] == head and record["observation"] == mode, "Fit receipt/arm identity")
        base.require(record["epochs"] == CONFIG["epochs"] and record["updates"] == updates
                     and record["training_queries"] == sum(loss_counts.values())*CONFIG["epochs"], "Fit completion work")
        for file, key in (("weights.pt", "weights_sha256"), ("dev-predictions.npz", "predictions_sha256"), ("batches.jsonl", "batches_sha256")):
            base.require(done["files"][f"fits/{name}/{file}"]["sha256"] == record[key], "Fit payload binding")
        original = originals[f"{head}-{seed}"]
        for key in ("initial_tensors_sha256", "common_initial_tensors_sha256", "parent_initial_tensors_sha256", "original_initial_tensors_sha256"):
            value = record[key]
            base.require(type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Invalid initialization digest")
        base.require(record["parent_initial_tensors_sha256"] == record["original_initial_tensors_sha256"] == original["initial_tensors_sha256"]
                     and record["initial_tensors_sha256"] == record["common_initial_tensors_sha256"], "V2 full initialization differs")
        base.require(paired_init.setdefault(seed, record["initial_tensors_sha256"]) == record["initial_tensors_sha256"], "Four-arm initialization differs")
        configuration = configuration_from_parent(original["configuration"], mode)
        base.require(record["configuration"] == plan["expected_configurations"][arm] == configuration
                     and record["parameters"] == 173186 and original["parameters"] == 99458, "Head configuration differs")
        base.positive(record["parameters_with_final_gradient"], "gradient parameter count", integer=True)
        base.require(record["parameters_with_final_gradient"] <= record["parameters"], "Invalid active count")
        ledger = [base.json.loads(line) for line in (folder / "batches.jsonl").read_text().splitlines()]
        inv, shapes, orders, losses = v2.audit_batches(ledger, train, data["queries"], head, training=True)
        work = audit_observations(ledger, train, data["queries"], mode, parents["tokens"]["context_lengths"])
        base.require(record["training_invariants"] == inv and record["training_actor_shapes"] == shapes
                     and record["training_losses"] == losses and record["training_observation_work"] == work, "Training ledger aggregates")
        base.require(paired_orders.setdefault(seed, orders) == orders, "Epoch orders unpaired")
        evaluation = record["evaluation"]
        einv, eshapes, _, _ = v2.audit_batches(evaluation["batches"], dev, data["queries"], head, training=False)
        ework = audit_observations(evaluation["batches"], dev, data["queries"], mode, parents["tokens"]["context_lengths"])
        base.require(evaluation["queries"] == len(counts) and evaluation["invariants"] == einv
                     and evaluation["actor_shapes"] == eshapes and evaluation["observation_work"] == ework, "Evaluation coverage/work")
        v2.merge_invariants(train_total, inv)
        v2.merge_invariants(eval_total, einv)
        for v in (record["train_wall_seconds"], record["fit_wall_seconds_before_receipt"], evaluation["wall_seconds"]):
            base.positive(v, "fit wall time")
        base.require(record["train_wall_seconds"]+evaluation["wall_seconds"] <= record["fit_wall_seconds_before_receipt"], "Fit timing scope")
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        base.validate_predictions(arrays, expected, counts)
        base.require(arrays["probabilities"].dtype == np.float32, "Prediction dtype")
        rows[name] = {**record, "metrics": metrics(arrays, masks), "evaluation": {k: v for k, v in evaluation.items() if k != "batches"}}
    base.require(done["process_memory_scope"] == MEMORY_SCOPE and done["observation_work_scope"] == OBSERVATION_WORK_SCOPE,
                 "Process memory/work scope")
    base.positive(done["process_lifetime_peak_rss_bytes"], "Whole peak RSS", integer=True)
    last_peak = 0
    for record in rows.values():
        peak = record["process_lifetime_peak_rss_bytes"]
        base.require(type(peak) is int and 0 < peak and last_peak <= peak <= done["process_lifetime_peak_rss_bytes"]
                     and record["process_memory_scope"] == MEMORY_SCOPE and record["observation_work_scope"] == OBSERVATION_WORK_SCOPE,
                     "Lifetime memory/work receipt")
        last_peak = peak
    reference = done["references"]
    base.require(reference["file"] == "references.npz" and reference["queries"] == len(counts)
                 and reference["sha256"] == done["files"]["references.npz"]["sha256"], "Reference coverage/binding")
    base.positive(reference["wall_seconds"], "reference time")
    base.require(math.fsum(r["fit_wall_seconds_before_receipt"] for r in rows.values())+reference["wall_seconds"] <= done["wall_seconds"], "Whole cost")
    with np.load(run / "references.npz", allow_pickle=False) as saved:
        ref_arrays = {k: saved[k] for k in saved.files}
    references = base.reference_metrics(ref_arrays, expected, counts)
    for ref, panels in references.items():
        correct = ref_arrays["pred_"+ref] == expected["labels"]
        for panel, r in panels.items():
            selected = np.ones(len(counts), bool) if panel == "all" else expected["unseen"] == (panel == "unseen")
            r["subgroups"] = {}
            for g, mask in masks.items():
                selected_group = selected & mask
                n, hit = int(selected_group.sum()), int(correct[selected_group].sum())
                r["subgroups"][g] = {"count": n, "correct": hit, "accuracy": hit/n if n else None}
    summary = {"study": STUDY, "status": "completed", "development_only": True, "method_order": list(ARMS), "seeds": list(SEEDS),
        "rows": rows, "families": family_metrics(rows), "references": references, "continuation_gate": criteria(rows),
        "technical_validity": {"passed": True, "exact_execution_files": 52, "scientific_sources": 35,
            "training": train_total, "evaluation": eval_total,
            "scope": "Saved witnesses authenticate internal normalization/update coverage, including dummy slots. "
                     "Neural arithmetic, encoder semantics, initialization and optimizer execution remain source/test-bound, not independently replayed."},
        "source_sha256": plan["source_sha256"], "execution_runtime": plan["runtime"], "execution_wall_seconds": done["wall_seconds"],
        "process_lifetime_peak_rss_bytes": done["process_lifetime_peak_rss_bytes"], "process_memory_scope": MEMORY_SCOPE,
        "observation_work_scope": OBSERVATION_WORK_SCOPE, "reference_wall_seconds": reference["wall_seconds"], "optimizer_updates": sum(r["updates"] for r in rows.values()),
        "supervised_presentations": sum(r["training_queries"] for r in rows.values()),
        "preparation": {"original_encoder": {k: parents["packet"].get(k) for k in ("wall_seconds", "unique_texts", "encoder_sequences", "encoder_tokens_with_special")},
            "lexical": {k: parents["lexical"].get(k) for k in ("wall_seconds", "counts", "float32_scalars", "bytes")}, **{k: v for k, v in parents["tokens"].items() if k != "context_lengths"}},
        "zero_probability_policy": "No epsilon or floor. Null NLL plus positive zero-target count is infinite; zero-support metrics are undefined.",
        "scope": "Exposed development representation control, not a new memory algorithm, causal diagnosis of past errors, calibrated belief, or untouched confirmation. FALSE support is zero in the actual fixed panel.",
        "timing_scope": "Capacity encoding is additional to full encoding. Cached head costs include monitored computation/assembly/I/O and exclude encoding; not end-to-end serving latency. Observation-work counts are not runtime speedups."}
    return summary, inventory, plan, done


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (title, key, group) in zip(axes.flat, (("Unseen macro", "macro_three", None), ("Unseen TRUE", "subgroups", "assigned_true"),
            ("Unseen DONTCARE", "subgroups", "dontcare"), ("Unseen revisions", "revision", None)), strict=True):
        for x, arm in enumerate(ARMS):
            def get(m, key=key, group=group):
                return m[key][group]["accuracy"] if group else m[key]["accuracy"]
            mean = get(summary["families"][arm]["unseen"])
            if mean is not None:
                ax.bar(x, mean*100, color=("#829bb0" if arm.startswith("slot") else "#008577"), alpha=.65)
            for j, seed in enumerate(SEEDS):
                v = get(summary["rows"][f"{arm}-{seed}"]["metrics"]["unseen"])
                if v is not None:
                    ax.scatter(x+(j-1)*.09, v*100, color="black", s=22, zorder=4)
        reference = get(summary["references"]["literal"]["unseen"])
        if reference is not None:
            ax.axhline(reference*100, color="#a34027", linestyle=":", label="Literal carry")
            ax.legend(fontsize=8)
        ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 100))
        ax.set_xticks(range(4), [a.replace("_", "\n") for a in ARMS], fontsize=9)
    fig.suptitle("Shared-token attention: all three seeds, exposed development")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def report_text(summary):
    gate = summary["continuation_gate"]
    lines = ["# Shared-token evidence control", "", f"Technical validity: PASS. Fixed continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['checks_passed']}/15).", "",
             "| Arm | Unseen macro % | TRUE % (n/fit) | FALSE % (n/fit) | DONTCARE % (n/fit) | Revision % | NLL |", "|---|---:|---:|---:|---:|---:|---:|"]
    def show(v, scale=1):
        return "undefined/infinite" if v is None else f"{v*scale:.4f}"
    for arm in ARMS:
        r = summary["families"][arm]["unseen"]
        groups = [f"{show(r['subgroups'][g]['accuracy'], 100)} ({r['subgroups'][g]['count_per_fit']})" for g in GROUPS]
        lines.append(f"| {arm} | {show(r['macro_three']['accuracy'], 100)} | {' | '.join(groups)} | {show(r['revision']['accuracy'], 100)} | {show(r['micro']['nll'])} |")
    lines += ["", "| Arm | Registered parameters | Training seconds, 3 fits | Evaluation seconds, 3 fits |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        rs = [summary["rows"][f"{arm}-{s}"] for s in SEEDS]
        lines.append(f"| {arm} | {rs[0]['parameters']} | {math.fsum(r['train_wall_seconds'] for r in rs):.3f} | {math.fsum(r['evaluation']['wall_seconds'] for r in rs):.3f} |")
    prep = summary["preparation"]
    lines += ["", f"Shared-token capacity work: {prep['capacity']['wall_seconds']:.3f}s. Full encoding: {prep['encoding']['wall_seconds']:.3f}s. Cache payload bytes: {prep['encoding']['cache_file_bytes']}.", "",
              summary["technical_validity"]["scope"], "", summary["zero_probability_policy"], "", summary["timing_scope"], "", summary["scope"]]
    return "\n".join(lines)+"\n"


def report(run, packet, lexical, tokens, plan_sha256, completed_sha256, out, *, root=ROOT,
           historical_summary=None, historical_summary_sha256=None):
    run, packet, lexical, tokens, out = map(Path, (run, packet, lexical, tokens, out))
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        base.write(out / "started.json", {"study": STUDY, "plan_sha256": plan_sha256, "completed_sha256": completed_sha256,
                   "reporter_sha256": base.sha(__file__), "reused_joint_reporter_sha256": JOINT_SHA, "reused_v2_reporter_sha256": V2_SHA})
        summary, inventory, plan, done = audit_saved(run, packet, lexical, tokens, plan_sha256, completed_sha256, root)
        if historical_summary is not None or historical_summary_sha256 is not None:
            base.require(historical_summary is not None and historical_summary_sha256 is not None, "Both historical path and pin required")
            bind(Path(historical_summary), historical_summary_sha256, inventory)
            historical = base.read(historical_summary)
            base.require(historical["study"] == "dialogue-copy-v2" and historical["status"] == "completed", "Historical V2 summary")
            summary["historical_v2_descriptive"] = {"sha256": historical_summary_sha256,
                "families": {h: historical["families"][h] for h in HEADS}, "scope": "Historical reference only, outside all fifteen checks; no cross-study causal attribution."}
        base.write(out / "summary.json", summary)
        (out / "report.md").write_text(report_text(summary))
        render(summary, out)
        for path, digest in inventory.items():
            base.bind(Path(path), digest)
        execution_members(run)
        receipt = {"study": STUDY, "status": "completed", "plan_sha256": plan_sha256, "execution_completed_sha256": completed_sha256,
            "source_sha256": plan["source_sha256"], "reporter_sha256": base.sha(__file__), "reused_joint_reporter_sha256": JOINT_SHA, "reused_v2_reporter_sha256": V2_SHA,
            "feature_packet_completed_sha256": plan["packet_completed_sha256"], "lexical_completed_sha256": plan["lexical_completed_sha256"],
            "tokens_completed_sha256": plan["tokens_completed_sha256"], "input_sha256": inventory,
            "technical_validity_passed": True, "continuation_passed": summary["continuation_gate"]["passed"],
            "checks_passed": summary["continuation_gate"]["checks_passed"], "checks_total": 15, "fit_count": 12,
            "neural_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0,
            "execution_wall_seconds": done["wall_seconds"], "wall_seconds": time.perf_counter()-started,
            "files": {n: {"sha256": base.sha(out/n), "bytes": (out/n).stat().st_size}
                      for n in ("started.json", "summary.json", "report.md", "comparison.png")}}
        base.write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            base.write(out / "failed.json", {"status": "failed", "error": str(error), "error_type": type(error).__name__, "wall_seconds": time.perf_counter()-started})
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            error.add_note("Failure receipt: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "packet", "lexical", "tokens", "plan-sha256", "completed-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--historical-summary")
    parser.add_argument("--historical-summary-sha256")
    args = parser.parse_args()
    result = report(args.run, args.packet, args.lexical, args.tokens, args.plan_sha256, args.completed_sha256, args.out,
                    historical_summary=args.historical_summary, historical_summary_sha256=args.historical_summary_sha256)
    print(base.json.dumps({"status": result["status"], "continuation_passed": result["continuation_passed"]}))
