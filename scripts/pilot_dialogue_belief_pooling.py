"""Bounded eight-fit real-SGD pooling pilot, without evaluation scoring.

Freeze reads authenticated public workload metadata only. Run caches the pinned
pretrained encoder once, then trains all four CPU heads on complete streams.
Targets are separate loss/evaluator records and never enter public forward.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import qualify_dialogue_observation_batches as inherited

from openjev.research.suspend_clock import Deadline, SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-belief-pooling-pilot-v1"
METHODS = ("pooled", "schema_attention", "belief_query", "state_token")
SEEDS = (7101, 7102)
FIT_ORDER = [f"{method}-{seed}" for seed in SEEDS for method in METHODS]
CONFIG = {"methods": list(METHODS), "seeds": list(SEEDS), "dialogues_per_split": 128,
          "epochs": 3, "batch_dialogues": 8, "learning_rate": .001, "weight_decay": .01,
          "clip": 1., "loss": "endpoint_uniform", "input_dim": 384, "attention_dim": 64,
          "projection_dim": 64, "hidden_dim": 64, "chunk_tokens": 254, "encoder_batch": 32,
          "cpu_threads": 1, "encoder_device": "mps", "memory_device": "cpu", "dtype": "float32",
          "pooled_parity_tolerance": 2e-5, "selection_salt": "openjev-belief-pooling-pilot-v1:"}
CAPS = {"freeze": {"wall_seconds": 120, "rss_bytes": 2 * 1024**3, "output_bytes": 32 * 1024**2},
        "run": {"wall_seconds": 1800, "rss_bytes": 8 * 1024**3, "mps_driver_bytes": 8 * 1024**3,
                "output_bytes": 2 * 1024**3}}
PROTOCOL = "research/dialogue-belief-pooling-pilot-protocol.md"
BASE_PLAN = "output/dialogue-observation-learning-v2/scientific-freeze-01/plan.json"
BASE_PIN = "295b4f6857b4f9d625ed1133e0cf0785db40b53f4537c235cdb609f168c98885"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK = "src/openjev/research/suspend_clock.py"
NEW_SOURCES = ("src/openjev/research/dialogue_belief_pooling.py", "tests/test_dialogue_belief_pooling.py",
               "scripts/pilot_dialogue_belief_pooling.py", "tests/test_pilot_dialogue_belief_pooling.py", PROTOCOL)
RUN_OUT = ROOT / "runs/dialogue-belief-pooling-pilot-v1/run-01"
PREPARED = ROOT / "output/dialogue-observation-learning-v1/preparation-02"
FEATURES = ROOT / "runs/sgd-state-v1/features-02/features.npy"
require, sha, read, write = inherited.require, inherited.sha, inherited.read, inherited.write


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "Nonempty JSONL record")
            yield json.loads(line)


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        stream.writelines(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)


def members(path):
    result = {}
    for item in sorted(Path(path).rglob("*")):
        require(not item.is_symlink(), "No output symlinks")
        if item.is_file() and item != Path(path) / "completed.json":
            result[item.relative_to(path).as_posix()] = {"sha256": sha(item), "bytes": item.stat().st_size}
    return result


def select_ids(profiles):
    result = {}
    for split, expected in (("train", 2017), ("dev", 2363)):
        ids = [p["dialogue_id"] for p in profiles if p["split"] == split]
        require(len(ids) == len(set(ids)) == expected, "Complete split-qualified workload inventory")
        result[split] = sorted(ids, key=lambda did: (
            hashlib.sha256((CONFIG["selection_salt"] + split + ":" + did).encode()).hexdigest(), did))[:128]
    return result


def epoch_orders(ids):
    require(len(ids) == len(set(ids)) == 128, "Exactly 128 paired TRAIN dialogues")
    return {str(seed): [sorted(ids, key=lambda did: (
        hashlib.sha256(f"{VERSION}:order:{seed}:{epoch}:{did}".encode()).hexdigest(), did))
        for epoch in range(CONFIG["epochs"])] for seed in SEEDS}


def source_map():
    require(sha(ROOT / BASE_PLAN) == BASE_PIN, "Fixed historical source closure")
    old = read(ROOT / BASE_PLAN)["source_sha256"]
    require(len(old) == 64 and not set(old).intersection(NEW_SOURCES), "Unchanged old64 plus additive sources")
    for name, pin in old.items():
        require(sha(ROOT / name) == pin, "Historical source identity")
    return {**old, **{name: sha(ROOT / name) for name in NEW_SOURCES}}


def freeze(args, budget, progress):
    require(args.prepared.resolve() == PREPARED.resolve(), "Fixed existing preparation")
    parent = inherited.authenticate_prepared(args.prepared, args.prepared_sha256)
    sources = source_map()
    selected = select_ids(read(args.prepared / "workloads.json")["profiles"])
    plan = {"version": VERSION, "config": CONFIG, "limits": CAPS, "prepared": str(args.prepared.resolve()),
            "prepared_completed_sha256": args.prepared_sha256, "prepared_plan_sha256": inherited.PREPARED_PLAN_PIN,
            "prepared_files": read(args.prepared / "completed.json")["files"], "source_sha256": sources,
            "historical_plan": {"path": BASE_PLAN, "sha256": BASE_PIN}, "runtime": inherited.qualified.runtime(),
            "model": {k: parent[k] for k in ("encoder", "revision", "snapshot", "model_files", "tokenizer_ids")},
            "inputs": parent["input_sha256"], "selected": selected, "orders": epoch_orders(selected["train"]),
            "fit_order": FIT_ORDER, "run_out": str(RUN_OUT),
            "expected": {"fits": 8, "optimizer_updates_per_fit": 48, "training_dialogue_visits_per_fit": 384,
                         "evaluation_dialogues_per_fit": 128, "total_training_forwards": 3072, "total_evaluation_forwards": 1024},
            "scope": "Prospective bounded real-SGD development pilot; no quality metrics or outcome-based selection in runner."}
    budget.storage()
    write(args.out / "plan.json", plan)
    inherited.authenticate_prepared(args.prepared, args.prepared_sha256)
    require(source_map() == sources, "Stable source closure after metadata freeze")
    return {"plan_sha256": sha(args.out / "plan.json"), "model_calls": 0, "targets_decoded": False, "official_test_opened": False}


def authenticate(args):
    require(sha(args.plan) == args.plan_sha256, "External pilot plan pin")
    plan = read(args.plan)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == CAPS
            and plan["fit_order"] == FIT_ORDER and Path(plan["run_out"]).resolve() == args.out.resolve() == RUN_OUT.resolve(),
            "Exact immutable pilot configuration and output")
    require(plan["source_sha256"] == source_map() and plan["runtime"] == inherited.qualified.runtime(), "Source/runtime freeze identity")
    prepared = Path(plan["prepared"])
    require(prepared.resolve() == PREPARED.resolve(), "Fixed prepared path")
    parent = inherited.authenticate_prepared(prepared, plan["prepared_completed_sha256"])
    require(plan["prepared_plan_sha256"] == inherited.PREPARED_PLAN_PIN
            and plan["prepared_files"] == read(prepared / "completed.json")["files"]
            and plan["inputs"] == parent["input_sha256"]
            and plan["model"] == {k: parent[k] for k in ("encoder", "revision", "snapshot", "model_files", "tokenizer_ids")},
            "Exact input/model bindings")
    require(plan["selected"] == select_ids(read(prepared / "workloads.json")["profiles"])
            and plan["orders"] == epoch_orders(plan["selected"]["train"]), "Fixed label-free membership and paired orders")
    done = read(args.plan.parent / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == args.plan_sha256
            and set(done["files"]) == {"started.json", "plan.json"} and members(args.plan.parent) == done["files"],
            "Complete unmodified prospective freeze")
    return plan, parent


def load_inputs(plan, budget):
    """Decode actor and evaluator stores separately after complete authentication."""
    import numpy as np

    prepared = Path(plan["prepared"])
    actors, targets = {}, {}
    for split in ("train", "dev"):
        selected = set(plan["selected"][split])
        for prefix, destination in (("actors", actors), ("targets", targets)):
            for item in lines(prepared / f"{prefix}-{split}.jsonl"):
                budget.check()
                if item["dialogue_id"] not in selected:
                    continue
                key = (split, item["dialogue_id"])
                require(item["split"] == split and key not in destination, "Distinct selected split identity")
                destination[key] = item if prefix == "actors" else item["rows"]
    expected = {(s, d) for s in ("train", "dev") for d in plan["selected"][s]}
    require(set(actors) == set(targets) == expected, "Every selected actor and endpoint group")
    lexical = np.load(prepared / "lexical-numbers.npy", mmap_mode="r", allow_pickle=False)
    require(lexical.dtype == np.float32 and lexical.ndim == 1, "Frozen number lexical observation cache")
    queries = read(FEATURES.parent / "packet.json")["queries"]
    rows = {}
    for split in ("train", "dev"):
        current = []
        for did in plan["selected"][split]:
            actor, source = actors[split, did], targets[split, did]
            require(source, "Every selected dialogue has scored endpoints")
            seen = set()
            for endpoint in source:
                t, q, y = (endpoint[k] for k in ("time", "query_position", "label_index"))
                require(all(type(v) is int for v in (t, q, y)) and 0 <= t < len(actor["turn_text_ids"])
                        and 0 <= q < len(actor["query_ids"]) and 0 <= y < len(actor["candidate_ids"][q])
                        and endpoint["query_index"] == actor["query_ids"][q]
                        and endpoint["label_id"] == actor["candidate_ids"][q][y]
                        and (t, q) not in seen, "Exact evaluator/public endpoint join")
                seen.add((t, q))
                candidates = actor["candidate_ids"][q]
                query = queries[endpoint["query_index"]]
                require(query["split"] == split and query["id"] == endpoint["query_id"]
                        and query["service"] == endpoint["service"] and query["slot"] == endpoint["slot"]
                        and query["candidate_ids"] == candidates
                        and len(query["candidate_values"]) == len(candidates), "Authenticated canonical query/candidate join")
                current.append({**endpoint, "row_index": len(current), "candidate_ids": candidates,
                    "candidate_values": query["candidate_values"]})
        rows[split] = current
    return actors, targets, lexical, rows


def unique_texts(actors):
    texts = {}
    for actor in actors.values():
        require(len(actor["tokens"]) == len(actor["original_feature_ids"]), "Original text feature identity")
        for feature, tokens in zip(actor["original_feature_ids"], actor["tokens"], strict=True):
            require(type(feature) is int and feature >= 0 and type(tokens) is list and tokens
                    and all(type(t) is int and t >= 0 for t in tokens), "Public content tokens")
            require(texts.setdefault(feature, tokens) == tokens, "Global feature ID/token identity")
    ids = sorted(texts)
    return ids, [texts[i] for i in ids]


def cache_tokens(encoder, tokens, ids, tokenizer_ids, directory, budget, progress):
    """Capture valid hidden rows from the exact unchanged pooled encoder calls."""
    import numpy as np
    import torch

    from openjev.research.dialogue_trainable_encoder import encode_token_lists

    chunks = [(i, min(254, len(t)-s)+2) for i, t in enumerate(tokens) for s in range(0, len(t), 254)]
    lengths = [len(t) + 2 * math.ceil(len(t)/254) for t in tokens]
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    projected_bytes = int(offsets[-1]) * 384 * 4 + len(tokens) * 384 * 4 + offsets.nbytes
    require(projected_bytes < CAPS["run"]["output_bytes"], "Cache payload alone must fit fixed output cap before encoder call")
    hidden = np.lib.format.open_memmap(directory / "tokens.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 384))
    positions, cursor = offsets[:-1].copy(), 0
    progress.update(encoder_forward_attempts=0, encoder_forward_returns=0)

    def before(*_):
        progress["encoder_forward_attempts"] += 1
        budget.check()

    def after(_module, _args, output):
        nonlocal cursor
        states = output.last_hidden_state
        batch = chunks[cursor:cursor+len(states)]
        require(len(batch) == len(states) and states.shape[-1] == 384 and states.dtype == torch.float32, "Actual cached hidden geometry")
        for j, (owner, valid) in enumerate(batch):
            value = states[j, :valid].detach().cpu().numpy()
            require(np.isfinite(value).all(), "Finite raw token states")
            start = int(positions[owner])
            hidden[start:start+valid] = value
            positions[owner] += valid
        cursor += len(batch)
        progress["encoder_forward_returns"] += 1
        hidden.flush()
        budget.sync(torch)

    first = encoder.register_forward_pre_hook(before)
    second = encoder.register_forward_hook(after)
    try:
        pooled, work = encode_token_lists(encoder, tokens, **tokenizer_ids, trainable=False, chunk_tokens=254, chunk_batch_size=32)
        require(cursor == len(chunks) and np.array_equal(positions, offsets[1:]), "All valid tokens retained, no padding or truncation")
        require(progress["encoder_forward_attempts"] == progress["encoder_forward_returns"] == work["encoder_calls"], "Complete actual encoder dispatch")
        pooled = pooled.detach().cpu().numpy()
        require(pooled.dtype == np.float32 and pooled.shape == (len(tokens), 384) and np.isfinite(pooled).all(), "Frozen pooled schema/turn vectors")
        for name, value in (("pooled.npy", pooled), ("offsets.npy", offsets)):
            with (directory / name).open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
        write(directory / "index.json", {"original_feature_ids": ids, "lengths": lengths, "work": work,
              "raw_token_policy": "Concatenated per-chunk CLS/content/SEP; no padding; encoder inputs unchanged."})
        budget.storage()
        return pooled, hidden, offsets, work
    finally:
        primary = sys.exception()
        first.remove()
        second.remove()
        try:
            hidden.flush()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Cache flush failed: " + repr(error))


def actor_inputs(record, lexical, pooled, hidden, offsets, feature_lookup):
    """Only public fields enter the model, never an evaluator row."""
    import numpy as np
    import torch

    from openjev.research.dialogue_trainable_encoder import build_actor

    local = [feature_lookup[f] for f in record["original_feature_ids"]]
    vectors = torch.from_numpy(np.array(pooled[local], dtype=np.float32, copy=True))
    start, shape = record["lexical_offset"], record["lexical_shape"]
    require(type(start) is int and start >= 0 and start + math.prod(shape) <= len(lexical), "Public lexical slice")
    lex = np.array(lexical[start:start+math.prod(shape)], dtype=np.float32, copy=True).reshape(shape)
    actor, none = build_actor(vectors, **{k: record[k] for k in ("turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids")},
                             lexical=lex, memory_device="cpu")
    turns = [local[i] for i in record["turn_text_ids"]]
    tokens = [torch.from_numpy(np.array(hidden[offsets[i]:offsets[i+1]], dtype=np.float32, copy=True)) for i in turns]
    return tokens, actor[0][0], actor[2][0], actor[3][0], actor[4][0], actor[5][0], none[0]


def forward(model, inputs, budget, counts, *, training, audit):
    import torch

    budget.check()
    counts["forward_attempts"] += 1
    try:
        logs = model(*inputs)
    finally:
        # Partial model counters remain observable if the transition rejects.
        audit.update(model.last_audit)
    counts["forward_returns"] += 1
    t, q, c = inputs[1].shape[0], inputs[2].shape[0], inputs[3].shape[1]
    require(logs.dtype == torch.float32 and logs.device.type == "cpu" and tuple(logs.shape) == (1, t, q, c), "Complete raw CPU trajectory")
    mask = inputs[4][None, None].expand_as(logs)
    require(torch.isfinite(logs[mask]).all().item() and torch.isneginf(logs[~mask]).all().item()
            and (logs.double().exp().sum(-1)-1).abs().max().item() <= 2e-6, "Finite normalized supported trajectory")
    expected = {"forward_attempts": 1, "forward_returns": 1, "observation_attempts": t, "observation_returns": t,
                "step_attempts": t, "step_returns": t, "state_checks": t, "real_question_updates": t*q,
                "attention_positions": 0 if model.method == "pooled" else q*sum(len(x)+1 for x in inputs[0])}
    require(model.last_audit == expected, "Complete internal public stream work and state checks")
    counts["training_forwards" if training else "evaluation_forwards"] += 1
    counts["public_turns"] += t
    counts["question_updates"] += t*q
    budget.check()
    return logs


def fit(model, fit_id, plan, actors, targets, lexical, rows, cache, directory, budget, progress):
    import numpy as np
    import torch

    from openjev.research.dialogue_finetune_training import supervised_loss

    pooled, hidden, offsets, lookup = cache
    seed = int(fit_id.rsplit("-", 1)[1])
    counts = Counter()
    progress["active_counts"] = counts
    train_audit, eval_audit = Counter(), Counter()
    progress.update(active_training_audit=train_audit, active_evaluation_audit=eval_audit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    active, nonzero = set(), set()
    training_start = budget.elapsed()
    with (directory / "updates.jsonl").open("x") as journal:
        for epoch, order in enumerate(plan["orders"][str(seed)]):
            for start in range(0, len(order), 8):
                progress.update(operation="train", epoch=epoch, batch_start=start)
                ids = order[start:start+8]
                denominator = sum(len(targets["train", did]) for did in ids)
                require(denominator > 0, "Batch endpoint denominator")
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total = 0.
                for did in ids:
                    inputs = actor_inputs(actors["train", did], lexical, pooled, hidden, offsets, lookup)
                    logs = forward(model, inputs, budget, counts, training=True, audit=train_audit)
                    loss = supervised_loss(logs, targets["train", did], [1., 1., 1.], denominator)
                    counts["backward_attempts"] += 1
                    loss.backward()
                    counts["backward_returns"] += 1
                    total += float(loss.detach())
                    counts["training_endpoints"] += len(targets["train", did])
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        require(torch.isfinite(parameter.grad).all().item(), "Finite head gradients")
                        active.add(name)
                        if parameter.grad.count_nonzero().item():
                            nonzero.add(name)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
                require(math.isfinite(float(norm)), "Finite clipping norm")
                counts["optimizer_attempts"] += 1
                optimizer.step()
                counts["optimizer_updates"] += 1
                require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Finite head after optimizer update")
                journal.write(json.dumps({"epoch": epoch, "batch_start": start, "dialogue_ids": ids, "endpoint_count": denominator,
                              "row_uniform_nll": total, "gradient_norm_before_clip": float(norm)}, allow_nan=False) + "\n")
                journal.flush()
                budget.storage()
    training_seconds = budget.elapsed() - training_start
    with (directory / "weights.pt").open("xb") as stream:
        torch.save({"memory": model.state_dict()}, stream)
    evaluation_start = budget.elapsed()
    logs_saved = np.lib.format.open_memmap(directory / "partial-log-probs.npy", mode="w+", dtype=np.float32, shape=(len(rows), 12))
    logs_saved[:] = -np.inf
    saved_ids = np.lib.format.open_memmap(directory / "partial-row-indices.npy", mode="w+", dtype=np.int64, shape=(len(rows),))
    saved_ids[:] = -1
    cursor = 0
    try:
        model.eval()
        with torch.no_grad():
            for did in plan["selected"]["dev"]:
                progress.update(operation="save-dev", dialogue_id=did, evaluation_rows=cursor)
                actor = actors["dev", did]
                inputs = actor_inputs(actor, lexical, pooled, hidden, offsets, lookup)
                logs = forward(model, inputs, budget, counts, training=False, audit=eval_audit)
                for endpoint in targets["dev", did]:
                    require(all(rows[cursor][k] == endpoint[k] for k in endpoint), "Exact evaluator row order")
                    t, q = endpoint["time"], endpoint["query_position"]
                    width = len(actor["candidate_ids"][q])
                    logs_saved[cursor, :width] = logs[0, t, q, :width].numpy()
                    saved_ids[cursor] = cursor
                    cursor += 1
                logs_saved.flush()
                saved_ids.flush()
                budget.storage()
        require(cursor == len(rows), "Complete final prediction membership")
        with (directory / "predictions.npz").open("xb") as stream:
            np.savez(stream, log_probs=logs_saved, row_indices=saved_ids)
    finally:
        primary = sys.exception()
        for value in (logs_saved, saved_ids):
            try:
                value.flush()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("Partial prediction flush: " + repr(error))
    del logs_saved, saved_ids
    for name in ("partial-log-probs.npy", "partial-row-indices.npy"):
        (directory / name).unlink()
    require(counts["optimizer_updates"] == counts["optimizer_attempts"] == 48
            and counts["training_forwards"] == counts["backward_returns"] == counts["backward_attempts"] == 384
            and counts["evaluation_forwards"] == 128 and counts["forward_attempts"] == counts["forward_returns"] == 512,
            "Complete fixed training and final evaluation work")
    named = dict(model.named_parameters())
    return {"counts": dict(counts), "training_work": dict(train_audit), "evaluation_work": dict(eval_audit),
            "configuration": model.configuration(), "training_seconds": training_seconds, "evaluation_seconds": budget.elapsed()-evaluation_start,
            "evaluation_rows": cursor, "registered_parameters": sum(p.numel() for p in named.values()),
            "gradient_present_names": sorted(active), "nonzero_gradient_names": sorted(nonzero),
            "gradient_present_parameters": sum(named[n].numel() for n in active),
            "nonzero_gradient_parameters": sum(named[n].numel() for n in nonzero),
            "parameter_activity_scope": "Observed gradient membership across 48 updates, not a general functional parameter count."}


def run(args, budget, progress):
    plan, parent = authenticate(args)
    import numpy as np
    import torch
    from transformers import AutoModel

    from openjev.research.dialogue_belief_pooling import DialogueBeliefPooling

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.backends.mps.is_available(), "Frozen encoder MPS required; no runtime fallback")
    actors, targets, lexical, rows = load_inputs(plan, budget)
    shutil.copyfile(args.plan, args.out / "plan.json")
    for split in ("train", "dev"):
        write_lines(args.out / f"rows-{split}.jsonl", rows[split])
    ids, tokens = unique_texts(actors)
    directory = args.out / "cache"
    directory.mkdir()
    progress.update(operation="load-pretrained-encoder")
    encoder = AutoModel.from_pretrained(parent["snapshot"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager", dtype=torch.float32).to("mps").eval().requires_grad_(False)
    initial_encoder = inherited.qualified.tensor_digest(encoder)
    budget.sync(torch)
    cache_started = budget.elapsed()
    pooled, hidden, offsets, work = cache_tokens(encoder, tokens, ids, parent["tokenizer_ids"], directory, budget, progress)
    reference = np.load(FEATURES, mmap_mode="r", allow_pickle=False)
    require(reference.dtype == np.float32 and reference.ndim == 2 and reference.shape[1] == 384
            and max(ids) < len(reference), "Original pooled vector identity/shape")
    error = float(np.max(np.abs(pooled-reference[ids])))
    require(error <= 2e-5, "Frozen original pooled schema/turn parity")
    require(inherited.qualified.tensor_digest(encoder) == initial_encoder and all(p.grad is None for p in encoder.parameters()), "Frozen encoder unchanged")
    cache_receipt = {"encoder_sha256": initial_encoder, "unique_texts": len(ids), "encoder_work": work,
                     "maximum_pooled_error": error, "wall_seconds": budget.elapsed()-cache_started, "files": members(directory)}
    write(directory / "completed.json", {"status": "completed", **cache_receipt})
    del encoder, reference, tokens
    gc.collect()
    torch.mps.empty_cache()
    budget.sync(torch)
    cache = pooled, hidden, offsets, {f: i for i, f in enumerate(ids)}
    fits = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial = DialogueBeliefPooling("schema_attention")
        state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
        initial_pin = inherited.qualified.tensor_digest(initial)
        del initial
        for method in METHODS:
            name = f"{method}-{seed}"
            progress.update(fit_id=name, active_counts={}, operation="fresh-head", completed_fits=len(fits))
            torch.manual_seed(seed)
            model = DialogueBeliefPooling(method).cpu().float()
            model.load_state_dict(state, strict=True)
            require(inherited.qualified.tensor_digest(model) == initial_pin, "Full paired initialization across all four methods")
            target = args.out / name
            target.mkdir()
            started = budget.elapsed()
            record = fit(model, name, plan, actors, targets, lexical, rows["dev"], cache, target, budget, progress)
            record.update(status="completed", fit_id=name, method=method, seed=seed, initial_state_sha256=initial_pin,
                          final_state_sha256=inherited.qualified.tensor_digest(model), wall_seconds=budget.elapsed()-started,
                          files=members(target))
            write(target / "completed.json", record)
            fits.append({**record, "completed_sha256": sha(target / "completed.json")})
            del model
            gc.collect()
            budget.storage()
            print(json.dumps({"phase": "belief-pooling-pilot", "completed_fits": len(fits), "fit_id": name}), flush=True)
    require([f["fit_id"] for f in fits] == FIT_ORDER, "All eight final fits required")
    authenticate(args)
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "fits": fits, "fit_order": FIT_ORDER,
            "cache": cache_receipt, "selected": plan["selected"], "row_counts": {s: len(v) for s, v in rows.items()},
            "quality_metrics_computed": False, "official_test_opened": False,
            "scope": "All selected complete public streams; final raw DEV endpoints only, no model selection or quality gate in this runner."}


class Budget:
    def __init__(self, out, deadline, limits):
        self.out, self.deadline, self.limits = out, deadline, limits
        self.max_driver = self.max_current = 0

    def check(self):
        self.deadline.check()
        require(inherited.qualified.rss() <= self.limits["rss_bytes"], "Whole-process RSS cap")

    def elapsed(self):
        self.check()
        return (self.deadline.clock.now_ns()-self.deadline.started_ns)/1e9

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= self.limits["output_bytes"], "Complete output cap")

    def sync(self, torch):
        torch.mps.synchronize()
        self.max_driver = max(self.max_driver, torch.mps.driver_allocated_memory())
        self.max_current = max(self.max_current, torch.mps.current_allocated_memory())
        require(self.max_driver <= self.limits["mps_driver_bytes"], "Sampled MPS driver cap")
        self.check()


def await_supervision(args, clock, started):
    wait = Deadline(clock, started, started+5_000_000_000)
    while not args.supervision.exists():
        wait.check()
        time.sleep(.025)
    require(args.supervision.stat().st_size <= 65536, "Bounded supervisor launch")
    pin, launch = sha(args.supervision), read(args.supervision)
    command = launch["command"]
    tail = command[1:]
    if tail and tail[0] == "-u":
        tail = tail[1:]
    require(Path(command[0]).resolve() == Path(sys.executable).resolve() and tail == sys.argv
            and Path(tail[0]).resolve() == Path(__file__).resolve() and tail[1] == "run", "Exact live worker argv")
    flags = dict(zip(tail[2::2], tail[3::2], strict=True))
    require(set(flags) == {"--plan", "--plan-sha256", "--supervision", "--out"}
            and flags["--plan-sha256"] == args.plan_sha256
            and all(Path(flags["--"+k]).resolve() == getattr(args, k).resolve() for k in ("plan", "supervision", "out")), "Supervisor request binding")
    require(launch["version"] == "dialogue-observation-supervision-v2" and Path(launch["cwd"]).resolve() == ROOT.resolve()
            and launch["pid"] == launch["pgid"] == os.getpid() == os.getpgrp() and launch["parent_pid"] == os.getppid()
            and launch["clock_backend"] == clock.backend and launch["cap_seconds"] == 1800
            and 0 <= launch["started_ns"] <= started < launch["deadline_ns"]
            and launch["deadline_ns"]-launch["started_ns"] == 1800_000_000_000
            and launch["watchdog_sha256"] == sha(ROOT / SUPERVISOR) and launch["clock_source_sha256"] == sha(ROOT / CLOCK),
            "Native fixed parent deadline and dedicated child identity")
    require(sha(args.supervision) == pin, "Stable supervisor launch")
    result = Deadline(clock, launch["started_ns"], launch["deadline_ns"])
    result.check()
    return result, pin


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    progress = {"operation": "authenticate"}
    clock = deadline = budget = handler = None
    launch_pin = None
    try:
        clock = SuspendClock()
        require(clock.backend in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "Native suspend-inclusive clock")
        started = clock.now_ns()
        if args.command == "run":
            deadline, launch_pin = await_supervision(args, clock, started)
        else:
            deadline = clock.deadline_after(CAPS["freeze"]["wall_seconds"])
        budget = Budget(args.out, deadline, CAPS[args.command])

        def expired(*_):
            raise TimeoutError("Supplementary emergency alarm expired")

        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.000001, deadline.remaining_ns()/1e9))
        write(args.out / "started.json", {"version": VERSION, "request": request, "limits": CAPS[args.command],
              "clock_backend": clock.backend, "worker_started_ns": started, "parent_started_ns": deadline.started_ns,
              "deadline_ns": deadline.expires_ns, "supervision_sha256": launch_pin})
        result = freeze(args, budget, progress) if args.command == "freeze" else run(args, budget, progress)
        if launch_pin is not None:
            require(sha(args.supervision) == launch_pin, "Stable complete supervision")
        files = members(args.out)
        budget.storage()
        ended = clock.now_ns()
        result.update(version=VERSION, status="completed", phase=args.command, request=request, files=files,
                      limits=CAPS[args.command], clock_backend=clock.backend, started_ns=deadline.started_ns,
                      worker_started_ns=started, finished_ns=ended, deadline_ns=deadline.expires_ns,
                      elapsed_ns=ended-deadline.started_ns, wall_seconds=(ended-deadline.started_ns)/1e9,
                      timing_available=True, supervision_sha256=launch_pin, peak_rss_bytes=inherited.qualified.rss(),
                      sampled_mps_driver_max_bytes=budget.max_driver, sampled_mps_current_max_bytes=budget.max_current,
                      no_retry=True)
        write(args.out / "completed.json", result)
        budget.storage()
        return result
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "completed.json").exists():
                (args.out / "completed.json").rename(args.out / "invalid-completion.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request,
                  "error_type": type(error).__name__, "error": str(error), "progress": progress,
                  "timing_available": False, "wall_seconds": None, "elapsed_ns": None,
                  "supervision_sha256": launch_pin, "quality_metrics_computed": False, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain original failure
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--prepared", type=Path, required=True)
    freeze_parser.add_argument("--prepared-sha256", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--plan-sha256", required=True)
    run_parser.add_argument("--supervision", type=Path, required=True)
    for subparser in (freeze_parser, run_parser):
        subparser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    answer = execute(parse_args())
    print(json.dumps({"status": answer["status"], "phase": answer["phase"]}))
