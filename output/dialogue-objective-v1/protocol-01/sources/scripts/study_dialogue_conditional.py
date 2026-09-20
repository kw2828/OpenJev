"""Exclusive nine-fit privileged conditional-observation probe.

Freeze reads authenticated metadata, headers and opaque hashes only. Torch and
the standalone scorer are imported only by the training backend. This runner
saves final predictions, never development quality metrics or selected fits.
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
import time
from collections import Counter
from pathlib import Path

import numpy as np
import prepare_dialogue_conditional as prep

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-conditional-training-v1"
PREPARER_SHA256 = "ec3fb56feee3e8ecd517472cfd6edecaa407e3a8f3a72850f0db459431b44a69"
COPY_SHA256 = "182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717"
MODES, SEEDS = ("mean", "slot", "candidate"), (5301, 5302, 5303)
ARM_ORDERS = (MODES, ("slot", "candidate", "mean"), ("candidate", "mean", "slot"))
CONFIG = {"methods": list(MODES), "seeds": list(SEEDS), "epochs": 20, "batch_size": 256,
          "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.,
          "input_dim": 384, "projection_dim": 64, "hidden_dim": 64,
          "threads": 4, "interop_threads": 1, "dtype": "float32", "deterministic": True}
LIMITS = {"wall_seconds": 3600., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
ADD_SOURCES = ("src/openjev/research/dialogue_conditional_observation.py",
               "tests/test_dialogue_conditional_observation.py",
               "src/openjev/research/dialogue_copy_memory.py",
               "scripts/study_dialogue_conditional.py", "tests/test_study_dialogue_conditional.py",
               "scripts/report_dialogue_conditional.py", "tests/test_report_dialogue_conditional.py",
               "research/dialogue-conditional-training-protocol.md")
EXPECTED_FITS = [f"{mode}-{seed}" for seed, modes in zip(SEEDS, ARM_ORDERS, strict=True) for mode in modes]
COUNTERS = ("forward_attempted", "forward_returned", "backward_attempted", "backward_returned",
            "optimizer_attempted", "optimizer_returned", "training_rows", "evaluation_rows")
require, sha, read, write, safe = prep.require, prep.sha, prep.read, prep.write, prep.safe


def runtime():
    return {**prep.runtime(), "torch": importlib.metadata.version("torch")}


def source_map(check=lambda: None):
    require(sha(ROOT/prep.SELF, check) == PREPARER_SHA256, "Pinned preparer source changed")
    require(sha(ROOT/"src/openjev/research/dialogue_copy_memory.py", check) == COPY_SHA256, "Frozen lexical-field dependency changed")
    return {**prep.source_map(check), **{name: sha(safe(ROOT, name), check) for name in ADD_SOURCES}}


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if platform.system() == "Darwin" else raw*1024)


class Budget:
    def __init__(self, out, start):
        self.out, self.start = Path(out), start
        self.progress = {"phase": "initializing", "hashed_files": 0, "completed_fits": [],
                         "totals": dict.fromkeys(COUNTERS, 0), "active_fit": None}
        self.partial_model = None
        self.partial_predictions = None
        self.partial_path = None

    def check(self):
        if time.monotonic()-self.start > LIMITS["wall_seconds"]:
            raise TimeoutError("Whole conditional study wall cap exceeded")
        require(peak_rss() <= LIMITS["rss_bytes"], "Process-lifetime RSS cap exceeded")

    def storage(self, extra=0):
        self.check()
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(size+extra <= LIMITS["output_bytes"], "Output storage cap exceeded")
        return size


def read_rows(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            rows.append(json.loads(line, object_pairs_hook=prep.unique_object,
                                   parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
    return rows


def authenticate_prepared(path, pin, budget, *, payloads):
    path = Path(path)
    require(sha(path/"completed.json", budget.check) == pin, "External prepared completion hash")
    done = read(path/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "prepare" and done["version"] == prep.VERSION
            and done["no_retry"] is True and done["runtime"] == prep.runtime(), "Prepared completion scope/runtime")
    names = {"started.json", "catalog.json", "rows.jsonl", "summary.json"}
    require(set(done["files"]) == names and {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
            == names | {"completed.json"}, "Prepared exact file closure")
    for name, item in done["files"].items():
        p = safe(path, name)
        require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Prepared payload identity")
    require(done["source_sha256"] == prep.source_map(budget.check), "Prepared source identity")
    request = read(path/"started.json")["request"]
    prep.validate_plan(Path(request["plan"]), done["plan_sha256"], budget)
    docs, verified = prep.receipts(budget, payloads=False)
    if payloads:
        for name, item in done["authenticated_inputs"].items():
            p = Path(name)
            require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(), "Prepared input escapes repository")
            require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Prepared inherited input drift")
    for name, item in verified.items():
        require(done["authenticated_inputs"].get(name) == item, "Prepared external input bindings differ")
    return done, docs


def metadata(path, budget):
    path = Path(path)
    catalog, summary, rows = read(path/"catalog.json"), read(path/"summary.json"), read_rows(path/"rows.jsonl")
    require(catalog["version"] == prep.VERSION and summary["version"] == prep.VERSION, "Prepared metadata version")
    queries = catalog["queries"]
    require([q["query_index"] for q in queries] == list(range(len(queries))), "Catalog index order")
    for i, row in enumerate(rows):
        require(type(row["row_index"]) is int and row["row_index"] == i, "Global prepared row order")
        qi = row["query_index"]
        require(prep.integer(qi) and qi < len(queries), "Query index range")
        q = queries[qi]
        require(row["split"] == q["split"] and row["query_id"] == q["query_id"]
                and row["service"] == q["service"] and row["slot"] == q["slot"], "Row/catalog identity")
        ids, label = q["candidate_ids"], row["current_label_index"]
        require(prep.integer(label) and label < len(ids) and ids[label] == row["current_candidate_id"], "Current label identity")
        require(row["candidate_count"] == len(ids) and 3 <= len(ids) <= 12, "Candidate support count")
        if row["admission"] == "admitted":
            pi, previous = row["previous_current_index"], row["previous_row_index"]
            require(prep.integer(pi) and pi < len(ids) and ids[pi] == row["previous_candidate_id"], "Previous candidate remap")
            require(prep.integer(previous) and previous < i, "Previous row identity")
            old = rows[previous]
            require(old["split"] == row["split"] and old["dialogue_id"] == row["dialogue_id"]
                    and old["service"] == row["service"] and old["slot"] == row["slot"]
                    and old["time"] == row["time"]-1 and old["current_candidate_id"] == ids[pi], "Adjacent previous value")
            require(row["derived_bin"] == prep.transition(ids[pi], ids[label]), "Admitted stratum identity")
        if i % 256 == 0:
            budget.check()
    require(summary["groups"] == prep.aggregate(iter(rows)) and summary["rows"] == len(rows), "Prepared aggregate mismatch")
    admitted = {split: [r for r in rows if r["split"] == split and r["admission"] == "admitted"] for split in ("train", "dev")}
    require(all(admitted.values()), "Empty admitted train or development cohort")
    return queries, admitted, summary


def stratum(row):
    name = row["derived_bin"]
    require(name in prep.BINS, "Unknown admitted transition")
    return 0 if name == "unmentioned_retention" else 1 if name == "assigned_retention" else 2


def objective(rows):
    counts = Counter(stratum(r) for r in rows)
    require(set(counts) == {0, 1, 2}, "Missing admitted training stratum")
    return {"counts": [counts[i] for i in range(3)], "weights": [len(rows)/(3*counts[i]) for i in range(3)]}


def batch_work(rows, mode):
    require(mode in MODES and rows, "Batch mode/rows")
    b, c = len(rows), max(r["candidate_count"] for r in rows)
    length = 0 if mode == "mean" else max(r["cache"]["token_stop"]-r["cache"]["token_start"] for r in rows)
    floats = b*(384 if mode == "mean" else length*384+length)+b*384+b*c*(384+10+1)
    return {"rows": b, "supported_candidate_positions": sum(r["candidate_count"] for r in rows),
            "padded_candidate_positions": b*c, "supported_token_positions": 0 if mode == "mean" else sum(r["cache"]["token_stop"]-r["cache"]["token_start"] for r in rows),
            "padded_token_positions": b*length, "attention_score_positions": b*c*length,
            "token_key_positions": b*length, "attention_query_positions": 0 if mode == "mean" else b*c,
            "scorer_positions": b*c, "float_input_scalars": floats, "float_input_bytes": 4*floats,
            "boolean_input_bytes": b*c+b*length, "max_token_length": length, "max_candidate_count": c}


def work_schedule(rows, orders, mode):
    totals, max_float_bytes, batches = Counter(), 0, 0
    for order in orders:
        for start in range(0, len(order), CONFIG["batch_size"]):
            work = batch_work([rows[int(i)] for i in order[start:start+CONFIG["batch_size"]]], mode)
            totals.update({k: v for k, v in work.items() if not k.startswith("max_")})
            max_float_bytes = max(max_float_bytes, work["float_input_bytes"])
            batches += 1
    return {"batches": batches, "totals": dict(totals), "maximum_batch_float_input_bytes": max_float_bytes}


def float_paths():
    return {"features": safe(ROOT, prep.INPUTS["packet"][0]).parent/"features.npy",
            "lexical": safe(ROOT, prep.INPUTS["lexical"][0]).parent/"lexical.npy",
            "tokens": safe(ROOT, prep.INPUTS["tokens"][0]).parent/"tokens.npy",
            "priors": safe(ROOT, prep.INPUTS["tokens"][0]).parent/"priors.npy"}


def headers():
    result = {key: {"path": str(path), "shape": list(prep.float_header(path))} for key, path in float_paths().items()}
    require(len(result["features"]["shape"]) == 2 and result["features"]["shape"][1] == 384
            and len(result["lexical"]["shape"]) == 1 and len(result["tokens"]["shape"]) == 2
            and result["tokens"]["shape"][1] == 384 and result["priors"]["shape"] == result["tokens"]["shape"][:1], "Feature headers")
    return result


def validate_addresses(rows, shapes):
    nf, nl, nt = shapes["features"]["shape"][0], shapes["lexical"]["shape"][0], shapes["tokens"]["shape"][0]
    for row in rows:
        c, a = row["candidate_count"], row["cache"]
        require(prep.integer(c, 1) and c <= 12, "Candidate count")
        require(all(prep.integer(i) and i < nf for i in [a["pooled_index"], a["query_feature_index"], *a["candidate_feature_indices"]])
                and len(a["candidate_feature_indices"]) == c, "Feature address range")
        require(prep.integer(a["token_start"]) and prep.integer(a["token_stop"], 1)
                and a["token_start"] < a["token_stop"] <= nt, "Token address range")
        require(prep.integer(a["lexical_start"]) and a["lexical_start"]+10*c <= nl
                and a["lexical_candidates"] == c and a["lexical_stride"] == 10, "Lexical address range")
        require(prep.integer(row["previous_current_index"]) and row["previous_current_index"] < c, "Previous index range")


def make_actor(rows, arrays, mode):
    """Only addresses, supplied support and declared previous value are consumed.

    Current target/bin/unseen/value-category fields are neither joined nor read.
    All source arrays are copied into new, writable batch storage before Torch.
    """
    shapes = {key: {"shape": list(value.shape)} for key, value in arrays.items()}
    validate_addresses(rows, shapes)
    work = batch_work(rows, mode)
    b, c, length = len(rows), work["max_candidate_count"], work["max_token_length"]
    actor = {"observation": np.zeros((b, 384) if mode == "mean" else (b, length, 384), np.float32),
             "query": np.zeros((b, 384), np.float32), "candidates": np.zeros((b, c, 384), np.float32),
             "candidate_mask": np.zeros((b, c), bool), "lexical": np.zeros((b, c, 10), np.float32),
             "previous_onehot": np.zeros((b, c), np.float32)}
    if mode != "mean":
        actor.update(token_mask=np.zeros((b, length), bool), token_prior=np.zeros((b, length), np.float32))
    for i, row in enumerate(rows):
        a, n = row["cache"], row["candidate_count"]
        actor["query"][i] = arrays["features"][a["query_feature_index"]]
        actor["candidates"][i, :n] = arrays["features"][a["candidate_feature_indices"]]
        actor["candidate_mask"][i, :n] = True
        actor["lexical"][i, :n] = arrays["lexical"][a["lexical_start"]:a["lexical_start"]+10*n].reshape(n, 10)
        actor["previous_onehot"][i, row["previous_current_index"]] = 1.
        if mode == "mean":
            actor["observation"][i] = arrays["features"][a["pooled_index"]]
        else:
            start, stop = a["token_start"], a["token_stop"]
            actor["observation"][i, :stop-start] = arrays["tokens"][start:stop]
            actor["token_mask"][i, :stop-start] = True
            actor["token_prior"][i, :stop-start] = arrays["priors"][start:stop]
    require(sum(a.size for a in actor.values() if a.dtype == np.float32) == work["float_input_scalars"], "Actual actor/work mismatch")
    require(all(np.isfinite(a).all() for a in actor.values() if a.dtype == np.float32), "Nonfinite actor inputs")
    return actor, work


def supervision(rows):
    labels = []
    for row in rows:
        value = row["current_label_index"]
        require(prep.integer(value) and value < row["candidate_count"], "Current target range")
        labels.append(value)
    return np.array(labels, np.int64), np.array([stratum(r) for r in rows], np.int64)


def reference_arrays(rows, lexical):
    literal = []
    for row in rows:
        c, start = row["candidate_count"], row["cache"]["lexical_start"]
        values = lexical[start:start+10*c].reshape(c, 10)[:, 4]
        require(np.isfinite(values).all() and np.isin(values, [0., 1.]).all() and (values == 1).sum() == 1,
                "Literal current must be exact supported one-hot")
        literal.append(int(np.flatnonzero(values == 1)[0]))
    return {"row_indices": np.array([r["row_index"] for r in rows], np.int64),
            "previous_indices": np.array([r["previous_current_index"] for r in rows], np.int64),
            "literal_indices": np.array(literal, np.int64)}


def freeze_body(out, budget, args):
    done, _ = authenticate_prepared(args.prepared, args.prepared_sha256, budget, payloads=True)
    _, rows, _ = metadata(args.prepared, budget)
    shape = headers()
    validate_addresses(rows["train"]+rows["dev"], shape)
    sources = source_map(budget.check)
    for name in sources:
        target = out/"sources"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(ROOT, name), target)
    shutil.copyfile(Path(args.prepared)/"completed.json", out/"prepared-completed.json")
    order_records, schedules = {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        orders = np.stack([rng.permutation(len(rows["train"])) for _ in range(CONFIG["epochs"])]).astype(np.int64)
        name = f"orders-{seed}.npy"
        np.save(out/name, orders, allow_pickle=False)
        order_records[str(seed)] = {"file": name, "sha256": sha(out/name, budget.check), "shape": list(orders.shape)}
        for mode in MODES:
            schedules[f"{mode}-{seed}"] = {"training": work_schedule(rows["train"], orders, mode),
                                            "evaluation": work_schedule(rows["dev"], [range(len(rows["dev"]))], mode)}
        budget.storage()
    n, d = len(rows["train"]), len(rows["dev"])
    plan = {"version": VERSION, "config": CONFIG, "limits": LIMITS, "runtime": runtime(), "source_sha256": sources,
            "prepared_path": str(Path(args.prepared).resolve()), "prepared_completed_sha256": args.prepared_sha256,
            "prepared_plan_sha256": done["plan_sha256"], "prepared_files": done["files"],
            "feature_headers": shape, "objective": objective(rows["train"]), "orders": order_records,
            "admitted_train_rows": n, "admitted_dev_rows": d, "expected_fits": EXPECTED_FITS,
            "updates_per_fit": CONFIG["epochs"]*math.ceil(n/CONFIG["batch_size"]),
            "evaluation_batches_per_fit": math.ceil(d/CONFIG["batch_size"]), "work_schedules": schedules,
            "primary": "unseen changed mean row NLL; candidate-slot paired differences must each be strictly negative",
            "no_quality_metrics_in_runner": True, "no_retry": True}
    write(out/"plan.json", plan)
    require(source_map(budget.check) == sources, "Freeze end source drift")
    return {"plan_sha256": sha(out/"plan.json", budget.check), "prepared_completed_sha256": args.prepared_sha256,
            "source_sha256": sources, "model_calls": 0, "float_feature_arrays_decoded": 0}


def validate_plan(path, pin, budget):
    path = Path(path)
    require(sha(path, budget.check) == pin, "External training plan hash")
    plan = read(path)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == LIMITS
            and plan["runtime"] == runtime() and plan["expected_fits"] == EXPECTED_FITS, "Frozen training recipe/runtime")
    require(plan["source_sha256"] == source_map(budget.check), "Training source closure")
    done = read(path.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == pin, "Training freeze completion")
    expected = {"started.json", "plan.json", "prepared-completed.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+p for p in plan["source_sha256"]}
    require(set(done["files"]) == expected and {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}
            == expected | {"completed.json"}, "Training freeze exact closure")
    for name, item in done["files"].items():
        target = safe(path.parent, name)
        require(target.stat().st_size == item["bytes"] and sha(target, budget.check) == item["sha256"], "Training freeze payload")
    require(sha(path.parent/"prepared-completed.json", budget.check) == plan["prepared_completed_sha256"], "Prepared snapshot pin")
    for name, digest in plan["source_sha256"].items():
        require(done["files"]["sources/"+name]["sha256"] == digest, "Source snapshot identity")
    return plan


def backend():
    import torch

    from openjev.research.dialogue_conditional_observation import (
        ATTENTION_TENSORS,
        COMMON_TENSORS,
        DialogueConditionalObservation,
        copy_initialization,
    )
    return torch, DialogueConditionalObservation, copy_initialization, COMMON_TENSORS, ATTENTION_TENSORS


def tensor_digest(model, names):
    parameters, digest = dict(model.named_parameters()), hashlib.sha256()
    for name in names:
        a = parameters[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode()+b"\0"+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return digest.hexdigest()


def invariants(scores, mask):
    import torch
    require(scores.dtype == torch.float32 and scores.shape == mask.shape and torch.isfinite(scores[mask]).all().item()
            and torch.isneginf(scores[~mask]).all().item(), "Output support/finiteness")
    raw = scores.detach().exp().to(torch.float64).sum(-1)
    error = float((raw-1).abs().max())
    require(math.isfinite(error) and error <= 2e-6, "Raw probability normalization")
    return {"batches": 1, "rows": len(scores), "supported_candidates": int(mask.sum()),
            "masked_candidates": int((~mask).sum()), "max_abs_mass_error": error,
            "min_supported_log_prob": float(scores.detach()[mask].min()),
            "max_supported_log_prob": float(scores.detach()[mask].max())}


def merge_invariants(total, value):
    for key in ("batches", "rows", "supported_candidates", "masked_candidates"):
        total[key] = total.get(key, 0)+value[key]
    for key, op in (("max_abs_mass_error", max), ("min_supported_log_prob", min), ("max_supported_log_prob", max)):
        total[key] = op(total[key], value[key]) if key in total else value[key]


def bump(budget, key, amount=1):
    budget.progress["totals"][key] += amount
    budget.progress["active_fit"]["counts"][key] += amount


def save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_arrays(plan):
    result = {}
    for key, entry in plan["feature_headers"].items():
        require(list(prep.float_header(entry["path"])) == entry["shape"], "Training array header drift")
        value = np.load(entry["path"], mmap_mode="r", allow_pickle=False)
        require(isinstance(value, np.memmap) and not value.flags.writeable and value.dtype == np.float32, "Read-only float32 arrays required")
        result[key] = value
    return result


def evaluate(model, rows, arrays, mode, torch, budget):
    model.eval()
    log_probs = np.full((len(rows), 12), -np.inf, np.float32)
    row_indices = np.array([r["row_index"] for r in rows], np.int64)
    totals, work = {}, Counter()
    budget.partial_predictions = {"log_probs": log_probs, "row_indices": row_indices, "completed_rows": 0}
    with torch.no_grad():
        for start in range(0, len(rows), CONFIG["batch_size"]):
            budget.storage()
            batch = rows[start:start+CONFIG["batch_size"]]
            actor, batch_counts = make_actor(batch, arrays, mode)
            tensors = {k: torch.from_numpy(v) for k, v in actor.items()}
            bump(budget, "forward_attempted")
            scores = model(**tensors)
            bump(budget, "forward_returned")
            stats = invariants(scores, tensors["candidate_mask"])
            merge_invariants(totals, stats)
            log_probs[start:start+len(batch), :scores.shape[1]] = scores.numpy()
            bump(budget, "evaluation_rows", len(batch))
            budget.partial_predictions["completed_rows"] = start+len(batch)
            work.update({k: v for k, v in batch_counts.items() if not k.startswith("max_")})
            budget.check()
    return {"log_probs": log_probs, "row_indices": row_indices}, {"normalization": totals, "work": dict(work), "rows": len(rows)}


def train_body(out, budget, args):
    plan = validate_plan(args.plan, args.plan_sha256, budget)
    authenticate_prepared(plan["prepared_path"], plan["prepared_completed_sha256"], budget, payloads=True)
    _, rows, _ = metadata(plan["prepared_path"], budget)
    require(objective(rows["train"]) == plan["objective"] and len(rows["train"]) == plan["admitted_train_rows"]
            and len(rows["dev"]) == plan["admitted_dev_rows"] and headers() == plan["feature_headers"], "Prepared recipe alignment")
    n, d = len(rows["train"]), len(rows["dev"])
    require(plan["updates_per_fit"] == CONFIG["epochs"]*math.ceil(n/CONFIG["batch_size"])
            and plan["evaluation_batches_per_fit"] == math.ceil(d/CONFIG["batch_size"]), "Operation counts")
    orders_by_seed = {}
    shutil.copyfile(args.plan, out/"plan.json")
    for seed in SEEDS:
        item = plan["orders"][str(seed)]
        source = safe(Path(args.plan).parent, item["file"])
        require(sha(source, budget.check) == item["sha256"], "Frozen row orders hash")
        order = np.load(source, allow_pickle=False)
        require(order.dtype == np.int64 and order.shape == (CONFIG["epochs"], n)
                and all(np.array_equal(np.sort(o), np.arange(n)) for o in order), "Complete epoch permutations")
        orders_by_seed[seed] = order
        shutil.copyfile(source, out/item["file"])
        for mode in MODES:
            expected = {"training": work_schedule(rows["train"], order, mode),
                        "evaluation": work_schedule(rows["dev"], [range(d)], mode)}
            require(plan["work_schedules"][f"{mode}-{seed}"] == expected, "Frozen work schedule")
    arrays = load_arrays(plan)
    save_npz(out/"references.npz", **reference_arrays(rows["dev"], arrays["lexical"]))
    torch, model_type, copy_initialization, common_names, attention_names = backend()
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(CONFIG["interop_threads"])
    torch.use_deterministic_algorithms(True)
    weights = torch.tensor(plan["objective"]["weights"], dtype=torch.float32)
    (out/"fits").mkdir()
    records = []
    for seed, arm_order in zip(SEEDS, ARM_ORDERS, strict=True):
        budget.storage()
        torch.manual_seed(seed)
        models = {mode: model_type(mode, input_dim=384, projection_dim=64, hidden_dim=64) for mode in MODES}
        copy_initialization(models["mean"], models["slot"])
        copy_initialization(models["mean"], models["candidate"])
        copy_initialization(models["slot"], models["candidate"], include_attention=True)
        common = {m: tensor_digest(models[m], common_names) for m in MODES}
        attention = {m: tensor_digest(models[m], attention_names) for m in ("slot", "candidate")}
        require(len(set(common.values())) == len(set(attention.values())) == 1, "Paired initializer tensors")
        for mode in arm_order:
            name, model = f"{mode}-{seed}", models[mode]
            destination = out/"fits"/name
            destination.mkdir()
            started = time.monotonic()
            budget.partial_path, budget.partial_model = destination, model
            budget.progress["active_fit"] = {"name": name, "phase": "training", "counts": dict.fromkeys(COUNTERS, 0)}
            budget.progress["phase"] = "training"
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            normalization, actual_work = {}, Counter()
            model.train()
            with (destination/"updates.jsonl").open("x") as journal:
                for epoch, order in enumerate(orders_by_seed[seed]):
                    for start in range(0, n, CONFIG["batch_size"]):
                        budget.storage()
                        batch_start = time.monotonic()
                        indices = order[start:start+CONFIG["batch_size"]]
                        batch = [rows["train"][int(i)] for i in indices]
                        active = budget.progress["active_fit"]
                        active["batch"] = {"epoch": epoch, "start": start, "row_indices": [r["row_index"] for r in batch], "phase": "assembly"}
                        actor, work = make_actor(batch, arrays, mode)
                        labels, bins = supervision(batch)
                        tensors = {k: torch.from_numpy(v) for k, v in actor.items()}
                        assembly_end = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        bump(budget, "forward_attempted")
                        active["batch"]["phase"] = "forward"
                        scores = model(**tensors)
                        bump(budget, "forward_returned")
                        stats = invariants(scores, tensors["candidate_mask"])
                        merge_invariants(normalization, stats)
                        target, strata = torch.from_numpy(labels), torch.from_numpy(bins)
                        loss = (-scores[torch.arange(len(batch)), target]*weights[strata]).mean()
                        require(bool(torch.isfinite(loss)), "Nonfinite weighted loss")
                        forward_end = time.monotonic()
                        bump(budget, "backward_attempted")
                        active["batch"]["phase"] = "backward"
                        loss.backward()
                        bump(budget, "backward_returned")
                        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), "Missing/nonfinite parameter gradients")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                        backward_end = time.monotonic()
                        bump(budget, "optimizer_attempted")
                        active["batch"]["phase"] = "optimizer"
                        optimizer.step()
                        bump(budget, "optimizer_returned")
                        bump(budget, "training_rows", len(batch))
                        optimizer_end = time.monotonic()
                        actual_work.update({k: v for k, v in work.items() if not k.startswith("max_")})
                        event = {"epoch": epoch, "start": start, "update": active["counts"]["optimizer_returned"],
                                 "row_indices": [r["row_index"] for r in batch], "work": work,
                                 "actor_shapes": {k: list(v.shape) for k, v in actor.items()},
                                 "normalization": stats, "weighted_loss": float(loss.detach()),
                                 "phase_seconds": {"assembly": assembly_end-batch_start, "forward_validation_loss": forward_end-assembly_end,
                                                   "backward_clip": backward_end-forward_end, "optimizer": optimizer_end-backward_end}}
                        journal.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
                        journal.flush()
                        active.pop("batch")
                        budget.storage()
            require(budget.progress["active_fit"]["counts"]["optimizer_returned"] == plan["updates_per_fit"], "Complete optimizer coverage")
            require(dict(actual_work) == plan["work_schedules"][name]["training"]["totals"], "Executed training work")
            with (destination/"weights.pt").open("xb") as stream:
                torch.save(model.state_dict(), stream)
            budget.progress["active_fit"]["phase"] = "evaluation"
            evaluation_start = time.monotonic()
            predictions, evaluation = evaluate(model, rows["dev"], arrays, mode, torch, budget)
            save_npz(destination/"dev-predictions.npz", **predictions)
            evaluation["wall_seconds"] = time.monotonic()-evaluation_start
            require(evaluation["work"] == plan["work_schedules"][name]["evaluation"]["totals"], "Executed evaluation work")
            counts = dict(budget.progress["active_fit"]["counts"])
            expected = {"forward_attempted": plan["updates_per_fit"]+plan["evaluation_batches_per_fit"],
                        "forward_returned": plan["updates_per_fit"]+plan["evaluation_batches_per_fit"],
                        "backward_attempted": plan["updates_per_fit"], "backward_returned": plan["updates_per_fit"],
                        "optimizer_attempted": plan["updates_per_fit"], "optimizer_returned": plan["updates_per_fit"],
                        "training_rows": CONFIG["epochs"]*n, "evaluation_rows": d}
            require(counts == expected and normalization["rows"] == CONFIG["epochs"]*n, "Full fit counters")
            record = {"status": "completed", "method": mode, "seed": seed, "epochs": CONFIG["epochs"], "counts": counts,
                      "initial_common_sha256": common[mode], "initial_attention_sha256": attention.get(mode),
                      "orders_sha256": plan["orders"][str(seed)]["sha256"], "configuration": model.configuration(),
                      "training_normalization": normalization, "training_work": dict(actual_work), "evaluation": evaluation,
                      "files": prep.manifest(destination, budget.check), "wall_seconds": time.monotonic()-started,
                      "process_lifetime_peak_rss_bytes": peak_rss(), "plan_sha256": args.plan_sha256}
            write(destination/"completed.json", record)
            records.append(record)
            budget.progress["completed_fits"].append(name)
            budget.progress["active_fit"] = None
            budget.partial_model = budget.partial_predictions = budget.partial_path = None
            budget.storage()
            del optimizer, predictions
        del models
    require(budget.progress["completed_fits"] == EXPECTED_FITS, "All nine final fits required")
    require(source_map(budget.check) == plan["source_sha256"] and runtime() == plan["runtime"], "End source/runtime drift")
    authenticate_prepared(plan["prepared_path"], plan["prepared_completed_sha256"], budget, payloads=True)
    return {"plan_sha256": args.plan_sha256, "prepared_completed_sha256": plan["prepared_completed_sha256"],
            "source_sha256": plan["source_sha256"], "expected_fits": EXPECTED_FITS, "completed_fits": EXPECTED_FITS,
            "fits": [{"method": r["method"], "seed": r["seed"], "counts": r["counts"]} for r in records],
            "quality_metrics_computed": False, "encoder_calls": 0, "test_contents_accessed": False}


def preserve_failure(out, budget, error):
    def attempt(label, function):
        try:
            function()
        except BaseException as secondary:  # noqa: BLE001 - preserve original exception
            error.add_note(label+": "+repr(secondary))
    if budget.partial_path is not None:
        if budget.partial_model is not None:
            def weights():
                import torch
                with (budget.partial_path/"partial-weights.pt").open("xb") as stream:
                    torch.save(budget.partial_model.state_dict(), stream)
            attempt("Partial weights", weights)
        if budget.partial_predictions is not None:
            def predictions():
                p = budget.partial_predictions
                n = p["completed_rows"]
                save_npz(budget.partial_path/"partial-dev-predictions.npz", log_probs=p["log_probs"][:n], row_indices=p["row_indices"][:n])
            attempt("Partial predictions", predictions)
    if (out/"completed.json").exists():
        attempt("Completion demotion", lambda: (out/"completed.json").rename(out/"late-completion.json"))
    attempt("Failure receipt", lambda: write(out/"failed.json", {"status": "failed", "version": VERSION,
            "error_type": type(error).__name__, "error": str(error), "progress": budget.progress,
            "wall_seconds": time.monotonic()-budget.start, "process_lifetime_peak_rss_bytes": peak_rss(), "no_retry": True}))


def execute(args):
    start, out = time.monotonic(), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, start)
    def alarm(_signum, _frame):
        raise TimeoutError("Whole conditional study wall cap exceeded")
    prior = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
    try:
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "runtime": runtime(), "limits": LIMITS,
              "request": {k: str(v) for k, v in vars(args).items()}, "no_retry": True})
        result = (freeze_body if args.phase == "freeze" else train_body)(out, budget, args)
        result.update(status="completed", version=VERSION, phase=args.phase, runtime=runtime(), no_retry=True,
                      progress=budget.progress, files=prep.manifest(out, budget.check),
                      wall_seconds=time.monotonic()-start, process_lifetime_peak_rss_bytes=peak_rss(),
                      wall_scope="Includes all work through payload hashing; completion write/hash and return are also cap checked",
                      rss_scope="Sampled process-lifetime peak, not per-arm allocation")
        budget.storage(len(prep.encoded(result)))
        write(out/"completed.json", result)
        budget.storage()
        return sha(out/"completed.json", budget.check)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        preserve_failure(out, budget, error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--prepared", type=Path, required=True)
    freeze.add_argument("--prepared-sha256", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--plan-sha256", required=True)
    train.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"phase": args.phase, "completed_sha256": execute(args)}))


if __name__ == "__main__":
    main()
