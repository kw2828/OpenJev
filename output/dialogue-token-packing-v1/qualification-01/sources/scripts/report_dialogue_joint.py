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
V2_PATH = "scripts/report_dialogue_copy_v2.py"
V2_SHA = "3182086abbd32f7d3dfe0555d26fe1ee2ff108f5f18445099b32061fd9634788"
_source = (ROOT / V2_PATH).read_bytes()
if hashlib.sha256(_source).hexdigest() != V2_SHA:
    raise ValueError("Frozen V2 reporter changed")
v2 = types.ModuleType("joint_frozen_v2_report")
v2.__file__ = str(ROOT / V2_PATH)
exec(compile(_source, v2.__file__, "exec"), v2.__dict__)  # noqa: S102 - exact pinned local source
base, np = v2.base, v2.np
STUDY, VERSION = "dialogue-joint-v1", "dialogue-joint-memory-v1"
ARMS = ("independent_readout", "independent_scalar", "joint_readout", "joint_scalar")
HEADS, SEEDS, PANELS = ("readout", "scalar"), v2.SEEDS, v2.PANELS
CONFIG = {**v2.CONFIG, "methods": list(ARMS), "heads": list(HEADS), "observations": ["independent", "joint"]}
CHECKS = {"heads": list(HEADS), "required_complete_fits": 12, "unseen_true_gain": .10,
          "unseen_dontcare_gain": .10, "unseen_macro_gain": .02, "maximum_seen_macro_deficit": .01,
          "unseen_micro_nll_nonworse": True, "strict_unseen_macro_wins": 2,
          "maximum_unseen_revision_deficit": .01, "total_head_checks": 14}
CAP, TOLERANCE = 3600., 2e-6
OLD_PLAN = "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905"
OLD_COMPLETED = "768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4"
DATA_COMPLETED = "677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12"
SOURCES = v2.SOURCES | {"src/openjev/research/dialogue_joint_memory.py", "tests/test_dialogue_joint_memory.py",
    "scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py", "scripts/study_dialogue_joint.py",
    "tests/test_study_dialogue_joint.py", "scripts/report_dialogue_joint.py", "tests/test_report_dialogue_joint.py",
    "research/dialogue-joint-protocol.md"}
ENCODER, REVISION = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CACHE_VERSION = "dialogue-joint-minilm-v1"
TEMPLATE = "candidate_text + '\\n' + 'System: ' + previous_system_text + '\\nUser: ' + current_user_text"
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


def cache_geometry(joint, data, index, receipt):
    embeddings = np.load(joint / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    indices = np.load(joint / "indices.npy", allow_pickle=False, mmap_mode="r")
    base.require(index["version"] == CACHE_VERSION and index["template"] == TEMPLATE
                 and index["embedding_width"] == 384 and index["padding_index"] == -1, "Joint index schema")
    base.require(embeddings.dtype == np.float32 and embeddings.shape == (index["unique_texts"], 384)
                 and indices.dtype == np.int64 and indices.shape == (index["index_count"],), "Joint array geometry")
    for start in range(0, len(embeddings), 8192):
        block = embeddings[start:start+8192]
        base.require(np.isfinite(block).all() and np.allclose(np.linalg.norm(block, axis=1), 1., atol=2e-6, rtol=0),
                     "Joint embedding normalization")
    offset, counts = 0, {}
    base.require(set(index["cohorts"]) == {"train", "dev"}, "Joint split membership")
    for split in ("train", "dev"):
        dialogs, entries = data["cohorts"][split], index["cohorts"][split]
        base.require(len(dialogs) == len(entries), "Joint dialogue membership")
        count = Counter(dialogues=len(dialogs), public_turns=0, public_question_steps=0, real_candidate_steps=0)
        for d, entry in zip(dialogs, entries, strict=True):
            qids = sorted({r["query"] for r in d["queries"]})
            sizes = [len(data["queries"][q]["candidate_ids"]) for q in qids]
            shape = [len(d["turns"]), len(qids), max(sizes)]
            base.require(entry == {"id": d["id"], "query_ids": qids, "offset": offset, "shape": shape}, "Joint public layout")
            size = math.prod(shape)
            block = indices[offset:offset+size].reshape(shape)
            mask = np.broadcast_to(np.arange(shape[2])[None, :] < np.asarray(sizes)[:, None], shape)
            base.require((block[~mask] == -1).all() and ((block[mask] >= 0) & (block[mask] < len(embeddings))).all(),
                         "Joint candidate padding/support")
            offset += size
            count.update(public_turns=shape[0], public_question_steps=shape[0]*shape[1], real_candidate_steps=shape[0]*sum(sizes))
        counts[split] = dict(count)
    base.require(offset == len(indices) and counts == index["counts"] == receipt["counts"], "Joint work/layout counts")


def authenticate_joint(joint, packet, lexical, data, plan, root, inventory):
    receipt = bind_tree(joint, plan["joint_completed_sha256"], inventory)
    expected = {"started.json", "plan.json", "original-encoder-source.py", "embeddings.npy", "indices.npy", "index.json"} | {
        "sources/" + n for n in receipt["source_sha256"]}
    base.require(set(receipt["files"]) == expected and not any(p.is_symlink() for p in joint.rglob("*"))
                 and {p.relative_to(joint).as_posix() for p in joint.rglob("*") if p.is_file()} == expected | {"completed.json"},
                 "Joint cache exact membership")
    bind(joint / "plan.json", receipt["plan_sha256"], inventory)
    prep = base.read(joint / "plan.json")
    inherited = base.read(packet / "encoder-plan.json")
    base.require(receipt["phase"] == "encode" and receipt["version"] == prep["version"] == CACHE_VERSION
                 and receipt["encoder"] == prep["encoder"] == ENCODER and receipt["revision"] == prep["revision"] == REVISION
                 and receipt["test_contents_accessed"] is False and receipt["parameter_training_steps"] == 0
                 and receipt["no_retry"] is True and 0 < receipt["wall_seconds"] <= 900, "Joint encoding identity/scope/cap")
    base.require(receipt["model_files_sha256"] == prep["model_files_sha256"] == inherited["model_files_sha256"]
                 and set(prep["model_files_sha256"]) == MODEL_FILES, "Original six encoder files differ")
    base.require(prep["template"] == TEMPLATE and prep["dtype"] == "float32" and prep["device"] == receipt["device"] == "mps"
                 and prep["chunk_tokens"] == 254 and prep["batch_size"] == 128 and prep["capacity_unique_texts"] == 512
                 and prep["capacity_seconds"] == 120 and prep["encoding_seconds"] == 900 and prep["disk_cap_bytes"] == DISK_CAP,
                 "Preparation recipe differs")
    base.require(receipt["source_sha256"] == prep["source_sha256"] and receipt["runtime"] == prep["runtime"], "Preparation source/runtime")
    for name, digest in prep["source_sha256"].items():
        bind(base.safe(root, name), digest, inventory)
        bind(base.safe(joint / "sources", name), digest, inventory)
        if name in plan["source_sha256"]:
            base.require(digest == plan["source_sha256"][name], "Preparation/study source mismatch")
    base.require({"scripts/prepare_dialogue_joint.py", "tests/test_prepare_dialogue_joint.py", "research/dialogue-joint-protocol.md"}
                 <= set(prep["source_sha256"]), "Missing preparation sources")
    bind(joint / "original-encoder-source.py", inherited["source_sha256"], inventory)
    base.require(prep["original_encoder_source_sha256"] == inherited["source_sha256"], "Original encoder source differs")
    base.require(set(prep["inputs"]) == {"packet", "lexical", "data"}, "Preparation parents")
    for label, directory, digest in (("packet", packet, plan["packet_completed_sha256"]),
            ("lexical", lexical, plan["lexical_completed_sha256"]),
            ("data", base.safe(root, prep["inputs"]["data"]["path"]), DATA_COMPLETED)):
        base.require(base.safe(root, prep["inputs"][label]["path"]).resolve() == directory.resolve()
                     and prep["inputs"][label]["completed_sha256"] == receipt[label+"_completed_sha256"] == digest,
                     "Preparation parent differs")
        if label == "data":
            bind_tree(directory, digest, inventory)
    bind(base.safe(root, prep["protocol_path"]), prep["protocol_sha256"], inventory)
    base.require(prep["protocol_path"] == "research/dialogue-joint-protocol.md"
                 and prep["protocol_sha256"] == plan["source_sha256"][prep["protocol_path"]], "Preparation protocol differs")
    cap_path = base.safe(root, receipt["capacity_path"])
    cap = bind_tree(cap_path, receipt["capacity_completed_sha256"], inventory)
    base.require(cap["phase"] == "capacity" and cap["no_retry"] is True and 0 < cap["wall_seconds"] <= 120
                 and cap["unique_texts_encoded"] == min(512, prep["unique_texts"]), "Capacity scope")
    for k in ("plan_sha256", "source_sha256", "runtime", "model_files_sha256"):
        base.require(cap[k] == receipt[k], "Capacity/cache identity: " + k)
    base.require("token-profile.json" in cap["files"], "Unbound capacity profile")
    profile, projection = base.read(cap_path / "token-profile.json"), cap["projection"]
    base.positive(cap["work"]["encoder_seconds"], "capacity encoder time")
    base.positive(cap["work"]["padded_token_slots"], "capacity padded tokens", integer=True)
    seconds = cap["work"]["encoder_seconds"] / cap["work"]["padded_token_slots"] * profile["padded_token_slots"]
    overhead = projection["observed_nonencoding_seconds"]
    base.require(type(overhead) in (int, float) and math.isfinite(overhead) and overhead >= 0
                 and projection["projected_encoder_seconds"] == seconds and projection["projected_total_seconds"] == seconds+overhead
                 and projection["maximum_projected_seconds"] == 720. and seconds+overhead <= 720.
                 and projection["maximum_cache_bytes"] == DISK_CAP and 0 < projection["projected_cache_bytes"] <= DISK_CAP
                 and projection["encoding_permitted"] is True, "Capacity admission arithmetic")
    for key in ("input_tokens", "encoder_tokens_with_special", "padded_token_slots", "overlength_texts_chunked", "truncated_tokens"):
        base.require(receipt["work"][key] == profile[key], "Full token coverage")
    base.require(receipt["work"]["truncated_tokens"] == 0 and receipt["progress"]["encoder_calls_attempted"]
                 == receipt["progress"]["encoder_calls_returned"] == profile["encoder_calls"]
                 and receipt["progress"]["encoded_sequences"] == profile["encoder_sequences"]
                 and receipt["progress"]["encoded_texts"] == receipt["unique_texts_encoded"]
                 == receipt["full_unique_texts"] == prep["unique_texts"], "Full encoding work coverage")
    index = base.read(joint / "index.json")
    base.require(index["unique_texts"] == prep["unique_texts"] and index["index_count"] == prep["index_count"]
                 and index["counts"] == prep["counts"], "Frozen cache geometry differs")
    for name in ("index.json", "indices.npy"):
        bind(joint / name, prep["layout_files"][name], inventory)
    cache_geometry(joint, data, index, receipt)
    base.require(receipt["cache_file_bytes"] == sum(e["bytes"] for e in receipt["files"].values())
                 and sum(p.stat().st_size for p in joint.rglob("*") if p.is_file()) <= DISK_CAP, "Cache bytes/cap")
    return {"encoding": receipt, "capacity": cap}


def authenticate_inputs(run, packet, lexical, joint, plan_sha, completed_sha, root, inventory):
    members = execution_members(run)
    bind(run / "plan.json", plan_sha, inventory)
    done = bind_tree(run, completed_sha, inventory, exact=members - {"completed.json"})
    plan = base.read(run / "plan.json")
    base.require(plan["study"] == done["study"] == STUDY and plan["implementation_version"] == VERSION
                 and plan["config"] == CONFIG and plan["practical_checks"] == CHECKS
                 and plan["wall_cap_seconds"] == CAP and plan["normalization_tolerance"] == TOLERANCE
                 and plan["execution_files"] == 52 and plan["expected_fits"] == 12, "Frozen study recipe")
    base.require(set(plan["source_sha256"]) == SOURCES and plan["source_sha256"][V2_PATH] == V2_SHA
                 and plan["source_sha256"]["scripts/report_dialogue_joint.py"] == base.sha(__file__), "Study source closure")
    for n, digest in plan["source_sha256"].items():
        bind(base.safe(root, n), digest, inventory)
    base.require(done["plan_sha256"] == plan_sha and done["fit_count"] == 12 and done["external_model_api_calls"] == 0
                 and done["encoder_calls"] == 0 and done["normalization_tolerance"] == TOLERANCE
                 and done["resume_authorized"] is False and done["joint_completed_sha256"] == plan["joint_completed_sha256"], "Completion identity")
    base.positive(done["wall_seconds"], "whole execution wall")
    base.require(done["wall_seconds"] <= CAP and base.read(run / "started.json") == {"status": "started",
        "plan_sha256": plan_sha, "runtime": plan["runtime"], "wall_cap_seconds": CAP}, "Execution timing/start identity")
    base.require(set(plan["paths"]) == {"packet", "lexical", "joint", "old_study"}, "Study paths")
    for k, directory in (("packet", packet), ("lexical", lexical), ("joint", joint)):
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
    parents["joint"] = authenticate_joint(joint, packet, lexical, data, plan, root, inventory)
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
        independent, joint = "independent_"+head, "joint_"+head
        complete = all(f"{a}-{s}" in rows for a in (independent, joint) for s in SEEDS)
        for name, panel, kind, margin in (("unseen_true_gain", "unseen", "assigned_true", Fraction(1, 10)),
                ("unseen_dontcare_gain", "unseen", "dontcare", Fraction(1, 10)),
                ("unseen_macro_gain", "unseen", "macro", Fraction(1, 50)),
                ("seen_macro_noninferiority", "seen", "macro", Fraction(-1, 100)),
                ("unseen_revision_noninferiority", "unseen", "revision", Fraction(-1, 100))):
            left = means(joint, panel, kind) if complete else None
            right = means(independent, panel, kind) if complete else None
            checks.append({"name": head+"/"+name, "passed": left is not None and right is not None and left-right >= margin,
                           "joint": None if left is None else float(left), "independent": None if right is None else float(right),
                           "required_difference": float(margin)})
        values = [[rows[f"{a}-{s}"]["metrics"]["unseen"]["micro"]["nll"] for s in SEEDS]
                  for a in (independent, joint)] if complete else [[None], [None]]
        a, b = map(base.mean_defined, values)
        checks.append({"name": head+"/unseen_micro_nll_nonworse", "passed": a is not None and b is not None and b <= a,
                       "joint": b, "independent": a})
        paired = [(exact(rows[f"{joint}-{s}"], "unseen", "macro"), exact(rows[f"{independent}-{s}"], "unseen", "macro"))
                  for s in SEEDS] if complete else []
        wins = sum(a is not None and b is not None and a > b for a, b in paired)
        checks.append({"name": head+"/strict_paired_unseen_macro_wins", "passed": complete and wins >= 2, "wins": wins, "required": 2})
    return {"passed": all(r["passed"] for r in checks), "checks_passed": sum(r["passed"] for r in checks),
            "checks_total": 15, "checks": checks}


def observation_work(dialogs, queries, mode):
    shapes = v2.work_counts(dialogs, queries)
    b, t = len(dialogs), max(len(d["turns"]) for d in dialogs)
    qids = [sorted({r["query"] for r in d["queries"]}) for d in dialogs]
    q = max(map(len, qids))
    c = max(len(queries[i]["candidate_ids"]) for ids in qids for i in ids)
    positions = b*t if mode == "independent" else b*t*q*c
    base.require(mode in ("independent", "joint"), "Observation mode")
    return {"emitted_float32_scalars": positions*384, "emitted_bytes": positions*384*4,
            "turn_projection_positions": positions, "schema_query_projection_positions": b*q,
            "schema_candidate_projection_positions": b*q*c, "real_public_turns": shapes["real_turns"],
            "real_question_updates": shapes["real_question_steps"],
            "real_candidate_updates": sum(len(d["turns"])*sum(len(queries[i]["candidate_ids"]) for i in ids)
                                          for d, ids in zip(dialogs, qids, strict=True))}


def audit_observations(ledger, dialogs, queries, mode):
    total = Counter()
    for row in ledger:
        expected = observation_work([dialogs[i] for i in row["indices"]], queries, mode)
        base.require(row["observation_work"] == expected, "Observation work differs from layout")
        total.update(expected)
    return dict(total)


def configuration_from_parent(parent):
    return {**parent, "class": "MonitoredJointMemory", "implementation_version": VERSION,
        "normalized_parent_version": "dialogue-copy-memory-v2-normalized",
        "observation_paths": {"independent": "[B,T,D]; exact V2 super path", "joint": "[B,T,Q,C,D]; candidate-specific turn projection"},
        "observation_mode_binding": "Caller must bind independent/joint mode and embedding provenance",
        "supported_methods": list(HEADS),
        "representation_control": "Same heads/parameters/normalization; only supplied turn embedding indexing differs"}


def audit_saved(run, packet, lexical, joint, plan_sha, completed_sha, root=ROOT):
    inventory = {}
    plan, done, parents, data, expected, counts, originals = authenticate_inputs(
        run, packet, lexical, joint, plan_sha, completed_sha, root, inventory)
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
        base.require(record["initial_tensors_sha256"] == record["original_initial_tensors_sha256"]
                     == record["common_initial_tensors_sha256"] == original["initial_tensors_sha256"], "V2 full initialization differs")
        base.require(paired_init.setdefault(seed, record["initial_tensors_sha256"]) == record["initial_tensors_sha256"], "Four-arm initialization differs")
        configuration = configuration_from_parent(original["configuration"])
        base.require(record["configuration"] == plan["expected_configurations"][arm] == configuration
                     and record["parameters"] == original["parameters"] == 99458, "Head configuration differs")
        base.positive(record["parameters_with_final_gradient"], "gradient parameter count", integer=True)
        base.require(record["parameters_with_final_gradient"] <= record["parameters"], "Invalid active count")
        ledger = [base.json.loads(line) for line in (folder / "batches.jsonl").read_text().splitlines()]
        inv, shapes, orders, losses = v2.audit_batches(ledger, train, data["queries"], head, training=True)
        work = audit_observations(ledger, train, data["queries"], mode)
        base.require(record["training_invariants"] == inv and record["training_actor_shapes"] == shapes
                     and record["training_losses"] == losses and record["training_observation_work"] == work, "Training ledger aggregates")
        base.require(paired_orders.setdefault(seed, orders) == orders, "Epoch orders unpaired")
        evaluation = record["evaluation"]
        einv, eshapes, _, _ = v2.audit_batches(evaluation["batches"], dev, data["queries"], head, training=False)
        ework = audit_observations(evaluation["batches"], dev, data["queries"], mode)
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
        "technical_validity": {"passed": True, "exact_execution_files": 52, "scientific_sources": 29,
            "training": train_total, "evaluation": eval_total,
            "scope": "Saved witnesses authenticate internal normalization/update coverage, including dummy slots. "
                     "Neural arithmetic, encoder semantics, initialization and optimizer execution remain source/test-bound, not independently replayed."},
        "source_sha256": plan["source_sha256"], "execution_runtime": plan["runtime"], "execution_wall_seconds": done["wall_seconds"],
        "reference_wall_seconds": reference["wall_seconds"], "optimizer_updates": sum(r["updates"] for r in rows.values()),
        "supervised_presentations": sum(r["training_queries"] for r in rows.values()),
        "preparation": {"original_encoder": {k: parents["packet"].get(k) for k in ("wall_seconds", "unique_texts", "encoder_sequences", "encoder_tokens_with_special")},
            "lexical": {k: parents["lexical"].get(k) for k in ("wall_seconds", "counts", "float32_scalars", "bytes")}, **parents["joint"]},
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
                ax.bar(x, mean*100, color=("#829bb0" if arm.startswith("independent") else "#008577"), alpha=.65)
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
    fig.suptitle("Candidate-conditioned observations: all three seeds, exposed development")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def report_text(summary):
    gate = summary["continuation_gate"]
    lines = ["# Candidate-conditioned observation control", "", f"Technical validity: PASS. Fixed continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['checks_passed']}/15).", "",
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
    lines += ["", f"Joint capacity work: {prep['capacity']['wall_seconds']:.3f}s. Full encoding: {prep['encoding']['wall_seconds']:.3f}s. Cache payload bytes: {prep['encoding']['cache_file_bytes']}.", "",
              summary["technical_validity"]["scope"], "", summary["zero_probability_policy"], "", summary["timing_scope"], "", summary["scope"]]
    return "\n".join(lines)+"\n"


def report(run, packet, lexical, joint, plan_sha256, completed_sha256, out, *, root=ROOT,
           historical_summary=None, historical_summary_sha256=None):
    run, packet, lexical, joint, out = map(Path, (run, packet, lexical, joint, out))
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        base.write(out / "started.json", {"study": STUDY, "plan_sha256": plan_sha256, "completed_sha256": completed_sha256,
                   "reporter_sha256": base.sha(__file__), "reused_v2_reporter_sha256": V2_SHA})
        summary, inventory, plan, done = audit_saved(run, packet, lexical, joint, plan_sha256, completed_sha256, root)
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
            "source_sha256": plan["source_sha256"], "reporter_sha256": base.sha(__file__), "reused_v2_reporter_sha256": V2_SHA,
            "feature_packet_completed_sha256": plan["packet_completed_sha256"], "lexical_completed_sha256": plan["lexical_completed_sha256"],
            "joint_completed_sha256": plan["joint_completed_sha256"], "input_sha256": inventory,
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
    for name in ("run", "packet", "lexical", "joint", "plan-sha256", "completed-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--historical-summary")
    parser.add_argument("--historical-summary-sha256")
    args = parser.parse_args()
    result = report(args.run, args.packet, args.lexical, args.joint, args.plan_sha256, args.completed_sha256, args.out,
                    historical_summary=args.historical_summary, historical_summary_sha256=args.historical_summary_sha256)
    print(base.json.dumps({"status": result["status"], "continuation_passed": result["continuation_passed"]}))
