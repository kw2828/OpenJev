"""Full-cohort adaptation of trained scalar memories with frozen trained encoders.

Freeze is metadata-only. Run authenticates all sources/inputs/checkpoints before
loading, qualifies every zero-residual method against complete saved DEV, then
performs all twelve paired fits. No quality metrics, model selection or scoring.
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

import pilot_dialogue_belief_pooling as shared

from openjev.research.suspend_clock import Deadline, SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-warm-pooling-v1"
METHODS = ("pooled", "schema_attention", "belief_query", "state_token")
SEEDS = (6901, 6902, 6903)
FIT_ORDER = [f"{m}-{s}" for s in SEEDS for m in METHODS]
CONFIG = {"methods": list(METHODS), "seeds": list(SEEDS), "epochs": 5, "batch_dialogues": 32,
          "memory_learning_rate": 1e-4, "attention_learning_rate": 1e-3, "weight_decay": 1e-4,
          "clip": 1., "loss": "original_train_three_stratum_weights", "input_dim": 384,
          "attention_dim": 64, "projection_dim": 64, "hidden_dim": 64,
          "chunk_tokens": 254, "encoder_batch": 32, "cpu_threads": 1,
          "encoder_device": "mps", "memory_device": "cpu", "dtype": "float32",
          "log_parity_tolerance": 1e-5, "probability_parity_tolerance": 1e-6,
          "selected_choice_changes": 0, "top_tie_mask_changes": 0,
          "train_dialogues": 2017, "dev_dialogues": 2363, "dev_endpoints": 62329}
CAPS = {"freeze": {"wall_seconds": 120, "rss_bytes": 2*1024**3, "output_bytes": 32*1024**2},
        "run": {"wall_seconds": 7200, "rss_bytes": 16*1024**3,
                "mps_driver_bytes": 8*1024**3, "output_bytes": 32*1024**3}}
PROTOCOL = "research/dialogue-warm-pooling-protocol.md"
PILOT_PLAN = "output/dialogue-belief-pooling-pilot-v1/freeze-01/plan.json"
PILOT_PIN = "044b9748b18677a0c66df7ee795e336910fa85d0706f35ae4ac070e625c5fae2"
NEW_SOURCES = ("src/openjev/research/dialogue_warm_pooling.py", "tests/test_dialogue_warm_pooling.py",
               "scripts/study_dialogue_warm_pooling.py", "tests/test_study_dialogue_warm_pooling.py", PROTOCOL)
BASE_PLAN, BASE_PIN = shared.BASE_PLAN, shared.BASE_PIN
PREPARED, SUPERVISOR, CLOCK = shared.PREPARED, shared.SUPERVISOR, shared.CLOCK
RUN_OUT = ROOT / "runs/dialogue-warm-pooling-v1/run-01"
OLD_RUN = ROOT / "output/dialogue-observation-learning-v2/scientific-run-01"
OLD_COMPLETED = "454ba05d600d8d8a69726e2e58ee8d337040c2be64c0aa31f4f6854b87eb28f5"
OLD_LAUNCH = "output/dialogue-observation-learning-v2/scientific-process-01.launch.json"
OLD_LAUNCH_PIN = "eefba879336be7ebb3f211174091014f1405741872a80f9934fb461c81c4ada9"
OLD_TERMINAL = "output/dialogue-observation-learning-v2/scientific-process-01.terminal.json"
OLD_TERMINAL_PIN = "7fe9b5e9fae7a83bdd04dceea559a849d78dcd346b97247b4b38d72be5d2b9ef"
inherited = shared.inherited
require, sha, read, write = shared.require, shared.sha, shared.read, shared.write
lines, write_lines, members = shared.lines, shared.write_lines, shared.members
actor_inputs, unique_texts, forward = shared.actor_inputs, shared.unique_texts, shared.forward


def source_map():
    require(sha(ROOT/PILOT_PLAN) == PILOT_PIN, "Immutable helper-source parent")
    old = read(ROOT/PILOT_PLAN)["source_sha256"]
    require(len(old) == 69 and not set(old).intersection(NEW_SOURCES), "Historical69 plus additive warm sources")
    for name, pin in old.items():
        require(sha(ROOT/name) == pin, "Unmodified inherited source: "+name)
    return {**old, **{name: sha(ROOT/name) for name in NEW_SOURCES}}


def checkpoint_lineage():
    """Metadata and opaque hashes only; never deserialize during freeze."""
    require(sha(OLD_RUN/"completed.json") == OLD_COMPLETED and sha(ROOT/BASE_PLAN) == BASE_PIN
            and sha(ROOT/OLD_TERMINAL) == OLD_TERMINAL_PIN and sha(ROOT/OLD_LAUNCH) == OLD_LAUNCH_PIN,
            "Fixed complete checkpoint campaign and successful parent pins")
    done, terminal, launch = read(OLD_RUN/"completed.json"), read(ROOT/OLD_TERMINAL), read(ROOT/OLD_LAUNCH)
    require(done["status"] == "completed" and done["plan_sha256"] == BASE_PIN
            and done["supervision_sha256"] == OLD_LAUNCH_PIN
            and terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["timed_out"] is False and terminal["group_absent"] is True
            and terminal["clock_error"] is None and terminal["cleanup"]["errors"] == []
            and terminal["cleanup"]["reaped"] is True and terminal["timing_available"] is True
            and terminal["started_ns"] <= done["started_ns"] <= done["finished_ns"] <= terminal["finished_ns"] < terminal["deadline_ns"],
            "Successful strict native parent/worker enclosure")
    require(all(terminal[k] == v for k, v in launch.items()), "Original unchanged parent launch")
    require([f["fit_id"] for f in done["fits"]] == read(ROOT/BASE_PLAN)["fit_order"]
            and len(done["fits"]) == 12 and done["source_sha256"] == read(ROOT/BASE_PLAN)["source_sha256"], "Complete old twelve-fit campaign")
    expected_files = {"started.json", "plan.json", "allocation.json", "evaluation-rows.jsonl", "references.json"} | {
        f"{name}/{file}" for name in read(ROOT/BASE_PLAN)["fit_order"] for file in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    require(set(done["files"]) == expected_files and {p.relative_to(OLD_RUN).as_posix() for p in OLD_RUN.rglob("*")
            if p.is_file() and p != OLD_RUN/"completed.json"} == expected_files, "Whole original execution manifest membership")
    selected = {"evaluation-rows.jsonl": done["files"]["evaluation-rows.jsonl"]}
    fits = {}
    for seed in SEEDS:
        name = f"trainable_numbers-{seed}"
        for file in ("completed.json", "weights.pt", "predictions.npz"):
            relative = f"{name}/{file}"
            selected[relative] = done["files"][relative]
        receipt = read(OLD_RUN/name/"completed.json")
        require(receipt["status"] == "completed" and receipt["fit_id"] == name
                and receipt["arm"] == "trainable_numbers" and receipt["seed"] == seed
                and receipt["evaluation"]["rows"] == CONFIG["dev_endpoints"], "Whole final trained checkpoint")
        fits[str(seed)] = {"fit_id": name, "completed_sha256": selected[f"{name}/completed.json"]["sha256"],
                          "checkpoint": receipt["checkpoint"], "encoder_sha256": receipt["final_encoder_sha256"]}
    for name, record in selected.items():
        path = OLD_RUN/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == record["bytes"]
                and sha(path) == record["sha256"], "Opaque checkpoint/reference payload binding")
    return {"run": str(OLD_RUN), "completed_sha256": OLD_COMPLETED, "plan_sha256": BASE_PIN,
            "terminal": {"path": OLD_TERMINAL, "sha256": OLD_TERMINAL_PIN},
            "launch": {"path": OLD_LAUNCH, "sha256": OLD_LAUNCH_PIN}, "files": selected, "fits": fits}


def full_ids(profiles):
    selected = {}
    for split, count in (("train", CONFIG["train_dialogues"]), ("dev", CONFIG["dev_dialogues"])):
        ids = [p["dialogue_id"] for p in profiles if p["split"] == split]
        require(len(ids) == len(set(ids)) == count, "Complete official supplied split inventory")
        selected[split] = ids
    return selected


def epoch_orders(ids):
    require(len(ids) == len(set(ids)) == CONFIG["train_dialogues"], "Complete paired TRAIN order")
    return {str(seed): [sorted(ids, key=lambda did: (
        hashlib.sha256(f"{VERSION}:order:{seed}:{epoch}:{did}".encode()).hexdigest(), did))
        for epoch in range(CONFIG["epochs"])] for seed in SEEDS}


def freeze(args, budget, progress):
    require(args.prepared.resolve() == PREPARED.resolve(), "Fixed original preparation")
    parent = inherited.authenticate_prepared(args.prepared, args.prepared_sha256)
    sources, lineage = source_map(), checkpoint_lineage()
    selected = full_ids(read(args.prepared/"workloads.json")["profiles"])
    plan = {"version": VERSION, "config": CONFIG, "limits": CAPS, "prepared": str(args.prepared.resolve()),
            "prepared_completed_sha256": args.prepared_sha256, "prepared_plan_sha256": inherited.PREPARED_PLAN_PIN,
            "prepared_files": read(args.prepared/"completed.json")["files"], "source_sha256": sources,
            "historical_plan": {"path": BASE_PLAN, "sha256": BASE_PIN}, "helper_plan": {"path": PILOT_PLAN, "sha256": PILOT_PIN},
            "runtime": inherited.qualified.runtime(), "model": {k: parent[k] for k in ("encoder", "revision", "snapshot", "model_files", "tokenizer_ids")},
            "inputs": parent["input_sha256"], "selected": selected, "orders": epoch_orders(selected["train"]),
            "loss_weights": parent["loss_weights"], "loss_counts": parent["loss_counts"], "checkpoints": lineage,
            "fit_order": FIT_ORDER, "run_out": str(RUN_OUT),
            "expected": {"fits": 12, "optimizer_updates_per_fit": 5*math.ceil(2017/32),
                         "training_dialogue_visits_per_fit": 5*2017, "evaluation_dialogues_per_fit": 2363,
                         "qualification_dialogues_per_seed": 5*2363},
            "scope": "Full exposed TRAIN/DEV development comparison; pretrained model replaced only by authenticated final trained encoder and memory; no quality scoring or selection."}
    write(args.out/"plan.json", plan)
    require(source_map() == sources and checkpoint_lineage() == lineage, "Stable frozen lineage")
    budget.storage()
    return {"plan_sha256": sha(args.out/"plan.json"), "model_calls": 0, "targets_decoded": False, "official_test_opened": False}


def authenticate(args):
    require(sha(args.plan) == args.plan_sha256, "External warm plan pin")
    plan = read(args.plan)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == CAPS
            and plan["fit_order"] == FIT_ORDER and Path(plan["run_out"]).resolve() == args.out.resolve() == RUN_OUT.resolve(), "Fixed complete warm recipe")
    require(plan["source_sha256"] == source_map() and plan["runtime"] == inherited.qualified.runtime()
            and plan["checkpoints"] == checkpoint_lineage(), "Source/runtime/checkpoint identity before decoding")
    parent = inherited.authenticate_prepared(Path(plan["prepared"]), plan["prepared_completed_sha256"])
    require(Path(plan["prepared"]).resolve() == PREPARED.resolve()
            and plan["prepared_plan_sha256"] == inherited.PREPARED_PLAN_PIN
            and plan["prepared_files"] == read(PREPARED/"completed.json")["files"]
            and plan["inputs"] == parent["input_sha256"] and plan["loss_weights"] == parent["loss_weights"]
            and plan["loss_counts"] == parent["loss_counts"]
            and plan["model"] == {k: parent[k] for k in ("encoder", "revision", "snapshot", "model_files", "tokenizer_ids")}, "Fixed original public inputs and objective")
    require(plan["selected"] == full_ids(read(PREPARED/"workloads.json")["profiles"])
            and plan["orders"] == epoch_orders(plan["selected"]["train"]), "Complete cohort and paired orders")
    receipt = read(args.plan.parent/"completed.json")
    require(receipt["status"] == "completed" and receipt["phase"] == "freeze"
            and receipt["plan_sha256"] == args.plan_sha256 and set(receipt["files"]) == {"started.json", "plan.json"}
            and members(args.plan.parent) == receipt["files"], "Completed prospective freeze")
    return plan, parent


def strict_state(module, state):
    import torch

    expected = module.state_dict()
    require(type(state) is dict and set(state) == set(expected), "Exact restored state keys")
    for key, target in expected.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == target.dtype
                and value.shape == target.shape and torch.isfinite(value).all().item(), "Exact finite restored state tensor")
    module.load_state_dict(state, strict=True)


def load_checkpoint(seed, plan, parent, budget, progress):
    import torch
    from study_dialogue_copy_v2 import MonitoredCopyMemoryV2
    from transformers import AutoModel

    for key in ("active_counts", "active_training_audit", "active_evaluation_audit", "dialogue_id", "evaluation_rows", "epoch", "batch_start"):
        progress.pop(key, None)
    progress.update(operation="load-trained-checkpoint", seed=seed, load_attempts=0, load_returns=0)
    saved = plan["checkpoints"]["fits"][str(seed)]
    path = OLD_RUN/saved["fit_id"]/"weights.pt"
    require(sha(path) == saved["checkpoint"]["sha256"], "Checkpoint pin before restricted loading")
    progress["load_attempts"] += 1
    state = torch.load(path, weights_only=True, map_location="cpu")
    progress["load_returns"] += 1
    require(type(state) is dict and set(state) == {"memory", "encoder"}, "Final checkpoint only; no optimizer state")
    encoder = AutoModel.from_pretrained(parent["snapshot"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager", dtype=torch.float32).cpu()
    memory = MonitoredCopyMemoryV2("scalar").cpu().float()
    strict_state(encoder, state["encoder"])
    strict_state(memory, state["memory"])
    require(inherited.qualified.tensor_digest(encoder) == saved["encoder_sha256"], "Published trained encoder digest")
    encoder.to("mps").eval().requires_grad_(False)
    memory.eval().requires_grad_(False)
    budget.sync(torch)
    return encoder, memory


def cache_dialogues(encoder, actors, tokenizer_ids, directory, budget, progress):
    """Exact original per-dialogue encoding order; one disk-backed cache per seed."""
    import numpy as np
    import torch

    from openjev.research.dialogue_trainable_encoder import encode_token_lists

    descriptors, sizes, total_texts = [], [], 0
    for (split, did), actor in actors.items():
        raw = set(actor["turn_text_ids"])
        n = len(actor["tokens"])
        require(len(actor["original_feature_ids"]) == n and len(set(actor["original_feature_ids"])) == n,
                "Original local unique text identity")
        lengths = [len(t)+2*math.ceil(len(t)/254) if i in raw else 0 for i, t in enumerate(actor["tokens"])]
        descriptors.append({"split": split, "dialogue_id": did, "pooled_offset": total_texts,
                            "texts": n, "original_feature_ids": actor["original_feature_ids"]})
        sizes.extend(lengths)
        total_texts += n
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(sizes, dtype=np.int64)))
    projected = int(offsets[-1])*384*4+total_texts*384*4+offsets.nbytes
    require(projected + sum(p.stat().st_size for p in budget.out.rglob("*") if p.is_file()) < CAPS["run"]["output_bytes"],
            "Entire new seed cache fits remaining cap before encoding")
    pooled = np.lib.format.open_memmap(directory/"pooled.npy", mode="w+", dtype=np.float32, shape=(total_texts, 384))
    hidden = np.lib.format.open_memmap(directory/"tokens.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), 384))
    with (directory/"offsets.npy").open("xb") as stream:
        np.save(stream, offsets, allow_pickle=False)
    total_work, calls = Counter(), Counter()
    progress.update(cache_work=total_work, cache_calls=calls)
    try:
        for item in descriptors:
            key = item["split"], item["dialogue_id"]
            tokens, base = actors[key]["tokens"], item["pooled_offset"]
            chunks = [(i, min(254, len(t)-s)+2) for i, t in enumerate(tokens) for s in range(0, len(t), 254)]
            cursor, positions = 0, offsets[base:base+len(tokens)].copy()
            progress.update(operation="cache-original-dialogue", split=key[0], dialogue_id=key[1])

            def before(*_):
                calls["encoder_forward_attempts"] += 1
                budget.check()

            def after(_module, _args, output, *, chunks=chunks, base=base, positions=positions):
                nonlocal cursor
                states = output.last_hidden_state
                batch = chunks[cursor:cursor+len(states)]
                require(len(batch) == len(states) and states.dtype == torch.float32 and states.shape[-1] == 384,
                        "Exact encoder hidden batch geometry")
                for j, (owner, valid) in enumerate(batch):
                    if sizes[base+owner] == 0:
                        continue
                    value = states[j, :valid].detach().cpu().numpy()
                    require(np.isfinite(value).all(), "Finite valid token states")
                    start = int(positions[owner])
                    hidden[start:start+valid] = value
                    positions[owner] += valid
                cursor += len(batch)
                calls["encoder_forward_returns"] += 1
                budget.sync(torch)

            first, second = encoder.register_forward_pre_hook(before), encoder.register_forward_hook(after)
            try:
                values, work = encode_token_lists(encoder, tokens, **tokenizer_ids, trainable=False,
                                                  chunk_tokens=254, chunk_batch_size=32)
                require(cursor == len(chunks) and np.array_equal(positions, offsets[base+1:base+len(tokens)+1]),
                        "Complete local valid tokens with no truncation")
                array = values.detach().cpu().numpy()
                require(array.dtype == np.float32 and array.shape == (len(tokens), 384) and np.isfinite(array).all(),
                        "Exact local pooled vectors")
                pooled[base:base+len(tokens)] = array
                total_work.update(work)
                pooled.flush()
                hidden.flush()
                budget.storage()
            finally:
                first.remove()
                second.remove()
        require(calls["encoder_forward_attempts"] == calls["encoder_forward_returns"] == total_work["encoder_calls"],
                "Complete charged encoder calls")
        index = {"dialogues": descriptors, "encoder_work": dict(total_work), "calls": dict(calls),
                 "pooled_text_occurrences": total_texts, "stored_valid_token_positions": int(offsets[-1]),
                 "scope": "Separate original local unique-text sequence per dialogue. Raw valid CLS/content/SEP only for public turns; pooled vectors for all texts. No cross-dialogue rebatching or pretrained-vector comparison."}
        write(directory/"index.json", index)
        return (pooled, hidden, offsets, {(d["split"], d["dialogue_id"]):
            {f: d["pooled_offset"]+i for i, f in enumerate(d["original_feature_ids"])} for d in descriptors}), index
    finally:
        primary = sys.exception()
        for array in (pooled, hidden):
            try:
                array.flush()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("Cache flush: "+repr(error))


def open_cache(directory):
    import numpy as np

    index = read(directory/"index.json")
    values = [np.load(directory/name, mmap_mode="r", allow_pickle=False) for name in ("pooled.npy", "tokens.npy", "offsets.npy")]
    lookup = {(d["split"], d["dialogue_id"]): {f: d["pooled_offset"]+i for i, f in enumerate(d["original_feature_ids"])}
              for d in index["dialogues"]}
    return (*values, lookup)


def inputs_for(key, actors, lexical, cache):
    pooled, hidden, offsets, lookup = cache
    return actor_inputs(actors[key], lexical, pooled, hidden, offsets, lookup[key])


def untouched_forward(memory, inputs, budget, counts, audit):
    import torch
    from study_dialogue_copy_v2 import empty_invariants, merge_invariants

    _, turns, queries, candidates, mask, lexical, none = inputs
    t, q = turns.shape[0], queries.shape[0]
    memory.audit = empty_invariants()
    memory.audit_query_mask = torch.ones((1, q), dtype=torch.bool)
    counts["forward_attempts"] += 1
    budget.check()
    try:
        logs = memory(turns[None], torch.ones((1, t), dtype=torch.bool), queries[None], candidates[None],
                      mask[None], lexical[None], none[None])
    finally:
        merge_invariants(audit, memory.audit)
    counts["forward_returns"] += 1
    counts["evaluation_forwards"] += 1
    counts["public_turns"] += t
    counts["question_updates"] += t*q
    budget.check()
    return logs


def parity(actual, expected, mask):
    import numpy as np

    require(actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape == mask.shape,
            "Original and replay raw float32 endpoint layout")
    require(np.isfinite(actual[mask]).all() and np.isfinite(expected[mask]).all()
            and np.isneginf(actual[~mask]).all() and np.isneginf(expected[~mask]).all(), "Raw supported and padding identity")
    require(np.max(np.abs(np.exp(actual.astype(np.float64)).sum(1)-1)) <= 2e-6
            and np.max(np.abs(np.exp(expected.astype(np.float64)).sum(1)-1)) <= 2e-6, "Raw normalized mass without repair")
    log_error = float(np.max(np.abs(actual[mask].astype(np.float64)-expected[mask].astype(np.float64))))
    prob_error = float(np.max(np.abs(np.exp(actual.astype(np.float64))-np.exp(expected.astype(np.float64)))))
    choices = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
    ties = int(np.count_nonzero((actual == actual.max(1, keepdims=True)) != (expected == expected.max(1, keepdims=True))))
    require(log_error <= CONFIG["log_parity_tolerance"] and prob_error <= CONFIG["probability_parity_tolerance"]
            and choices == 0 and ties == 0, "Frozen full DEV parity including all top ties")
    return {"maximum_log_difference": log_error, "maximum_probability_difference": prob_error,
            "selected_choice_changes": choices, "top_tie_mask_changes": ties, "endpoints": len(actual), "passed": True}


def evaluate(model, actors, targets, lexical, rows, plan, cache, directory, budget, progress, *, original=False):
    import numpy as np
    import torch
    from study_dialogue_copy_v2 import empty_invariants

    counts, audit = Counter(), empty_invariants() if original else Counter()
    progress.update(active_counts=counts, active_evaluation_audit=audit, operation="complete-dev")
    saved = np.lib.format.open_memmap(directory/"partial-log-probs.npy", mode="w+", dtype=np.float32, shape=(len(rows), 12))
    saved[:] = -np.inf
    indices, cursor, started = np.arange(len(rows), dtype=np.int64), 0, budget.elapsed()
    try:
        model.eval()
        with torch.no_grad():
            for did in plan["selected"]["dev"]:
                progress.update(dialogue_id=did, evaluation_rows=cursor)
                inputs = inputs_for(("dev", did), actors, lexical, cache)
                logs = untouched_forward(model, inputs, budget, counts, audit) if original else forward(
                    model, inputs, budget, counts, training=False, audit=audit)
                for row in targets["dev", did]:
                    require(all(rows[cursor][k] == row[k] for k in row), "Original canonical endpoint order")
                    width = len(rows[cursor]["candidate_ids"])
                    saved[cursor, :width] = logs[0, row["time"], row["query_position"], :width].numpy()
                    cursor += 1
                saved.flush()
                budget.storage()
        require(cursor == len(rows) and counts["evaluation_forwards"] == len(plan["selected"]["dev"]), "Complete entire DEV endpoint membership")
        with (directory/"predictions.npz").open("xb") as stream:
            np.savez(stream, log_probs=saved, row_indices=indices)
        result = np.array(saved, copy=True)
    finally:
        primary = sys.exception()
        try:
            saved.flush()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Partial DEV flush: "+repr(error))
    del saved
    (directory/"partial-log-probs.npy").unlink()
    return result, {"counts": dict(counts), "work": dict(audit), "rows": cursor, "wall_seconds": budget.elapsed()-started}


def paired_model(method, seed, memory_state, full_state=None):
    import torch

    from openjev.research.dialogue_warm_pooling import DialogueWarmPooling

    torch.manual_seed(seed)
    model = DialogueWarmPooling(method).cpu().float()
    if full_state is None:
        strict_state(model.memory, memory_state)
    else:
        strict_state(model, full_state)
    model.requires_grad_(True)
    return model


def optimizer_for(model):
    import torch

    memory = list(model.memory.parameters())
    attention = [p for n, p in model.named_parameters() if not n.startswith("memory.")]
    require(memory and attention and not {id(p) for p in memory}.intersection(id(p) for p in attention)
            and len(memory)+len(attention) == len(list(model.parameters())), "Disjoint complete optimizer groups")
    return torch.optim.AdamW([{"params": memory, "lr": CONFIG["memory_learning_rate"]},
                              {"params": attention, "lr": CONFIG["attention_learning_rate"]}],
                             weight_decay=CONFIG["weight_decay"])


def adapt(model, name, plan, actors, targets, lexical, rows, cache, directory, budget, progress):
    import torch

    from openjev.research.dialogue_finetune_training import supervised_loss

    seed, counts, work = int(name.rsplit("-", 1)[1]), Counter(), Counter()
    progress.update(active_counts=counts, active_training_audit=work, optimizer_created=False)
    optimizer = optimizer_for(model)
    progress["optimizer_created"] = True
    present, nonzero, started = set(), set(), budget.elapsed()
    with (directory/"updates.jsonl").open("x") as journal:
        for epoch, order in enumerate(plan["orders"][str(seed)]):
            for start in range(0, len(order), CONFIG["batch_dialogues"]):
                batch = order[start:start+CONFIG["batch_dialogues"]]
                denominator = sum(len(targets["train", did]) for did in batch)
                require(denominator > 0, "Full actual batch endpoint denominator")
                progress.update(operation="adapt", epoch=epoch, batch_start=start)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total = 0.
                for did in batch:
                    logs = forward(model, inputs_for(("train", did), actors, lexical, cache), budget, counts, training=True, audit=work)
                    loss = supervised_loss(logs, targets["train", did], plan["loss_weights"], denominator)
                    counts["backward_attempts"] += 1
                    loss.backward()
                    counts["backward_returns"] += 1
                    counts["training_endpoints"] += len(targets["train", did])
                    total += float(loss.detach())
                for key, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        require(torch.isfinite(parameter.grad).all().item(), "Finite adaptation gradients")
                        present.add(key)
                        if parameter.grad.count_nonzero().item():
                            nonzero.add(key)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"], error_if_nonfinite=True, foreach=False)
                counts["optimizer_attempts"] += 1
                optimizer.step()
                counts["optimizer_updates"] += 1
                require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Finite adapted parameters")
                journal.write(json.dumps({"epoch": epoch, "batch_start": start, "dialogue_ids": batch,
                    "endpoint_count": denominator, "weighted_endpoint_nll": total, "gradient_norm_before_clip": float(norm)}, allow_nan=False)+"\n")
                journal.flush()
                budget.storage()
    train_seconds = budget.elapsed()-started
    with (directory/"weights.pt").open("xb") as stream:
        torch.save({"memory": model.state_dict()}, stream)
    _, evaluation = evaluate(model, actors, targets, lexical, rows, plan, cache, directory, budget, progress)
    n, epochs = len(plan["selected"]["train"]), CONFIG["epochs"]
    require(counts["optimizer_attempts"] == counts["optimizer_updates"] == epochs*math.ceil(n/CONFIG["batch_dialogues"])
            and counts["forward_attempts"] == counts["forward_returns"] == counts["training_forwards"]
            == counts["backward_attempts"] == counts["backward_returns"] == n*epochs, "Complete fixed adaptation work")
    parameters = dict(model.named_parameters())
    return {"training_counts": dict(counts), "training_work": dict(work), "training_seconds": train_seconds,
            "evaluation": evaluation, "configuration": model.configuration(),
            "gradient_present_names": sorted(present), "nonzero_gradient_names": sorted(nonzero),
            "registered_parameters": sum(p.numel() for p in parameters.values()),
            "gradient_present_parameters": sum(parameters[k].numel() for k in present),
            "nonzero_gradient_parameters": sum(parameters[k].numel() for k in nonzero)}


def require_qualification(records):
    names = [f"{method}-{seed}" for seed in SEEDS for method in ("untouched", *METHODS)]
    require([r["fit_id"] for r in records] == names
            and all(r["status"] == "completed" and r["parity"]["passed"] is True
                    and r["rows"] == r["parity"]["endpoints"] == CONFIG["dev_endpoints"]
                    and r["counts"]["evaluation_forwards"] == CONFIG["dev_dialogues"] for r in records),
            "All three seeds and all five complete paths qualify before any adaptation")


def run(args, budget, progress):
    plan, parent = authenticate(args)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.backends.mps.is_available(), "Required frozen encoder MPS, no fallback")
    actors, targets, lexical, rows = shared.load_inputs(plan, budget)
    original_rows = list(lines(OLD_RUN/"evaluation-rows.jsonl"))
    require(rows["dev"] == original_rows and len(rows["dev"]) == CONFIG["dev_endpoints"], "Complete unchanged old DEV row identities")
    shutil.copyfile(args.plan, args.out/"plan.json")
    for split in ("train", "dev"):
        write_lines(args.out/f"rows-{split}.jsonl", rows[split])
    masks = np.arange(12)[None, :] < np.asarray([len(r["candidate_ids"]) for r in rows["dev"]])[:, None]
    initial_states, qualifications, cache_records = {}, [], []
    progress.update(qualification_paths=qualifications, completed_caches=cache_records)
    for seed in SEEDS:
        progress.update(seed=seed, operation="qualification", optimizer_created=False)
        encoder, memory = load_checkpoint(seed, plan, parent, budget, progress)
        encoder_pin = inherited.qualified.tensor_digest(encoder)
        memory_state = {k: v.detach().clone() for k, v in memory.state_dict().items()}
        cache_dir = args.out/f"cache-{seed}"
        cache_dir.mkdir()
        start = budget.elapsed()
        cache, index = cache_dialogues(encoder, actors, parent["tokenizer_ids"], cache_dir, budget, progress)
        require(inherited.qualified.tensor_digest(encoder) == encoder_pin and all(p.grad is None for p in encoder.parameters()),
                "Trained encoder frozen throughout original-order caching")
        cache_record = {"status": "completed", "seed": seed, "encoder_sha256": encoder_pin,
                        "wall_seconds": budget.elapsed()-start, "encoder_work": index["encoder_work"], "files": members(cache_dir)}
        write(cache_dir/"completed.json", cache_record)
        cache_records.append(cache_record)
        del encoder
        gc.collect()
        torch.mps.empty_cache()
        budget.sync(torch)
        with np.load(OLD_RUN/f"trainable_numbers-{seed}"/"predictions.npz", allow_pickle=False) as packet:
            require(set(packet.files) == {"log_probs", "row_indices"} and packet["row_indices"].dtype == np.int64
                    and np.array_equal(packet["row_indices"], np.arange(len(rows["dev"]), dtype=np.int64)), "Original entire saved DEV row order")
            expected = packet["log_probs"]
        initial = paired_model("schema_attention", seed, memory_state)
        state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
        initial_states[seed] = state
        initial_pin = inherited.qualified.tensor_digest(initial)
        del initial
        for method in ("untouched", *METHODS):
            name = f"{method}-{seed}"
            progress.update(fit_id=name, operation="full-dev-parity", optimizer_created=False)
            directory = args.out/"qualification"/name
            directory.mkdir(parents=True)
            model = memory if method == "untouched" else paired_model(method, seed, memory_state, state)
            initial_digest = inherited.qualified.tensor_digest(model)
            if method != "untouched":
                require(initial_digest == initial_pin, "All four exact full paired warm initial states")
            actual, metadata = evaluate(model, actors, targets, lexical, rows["dev"], plan, cache, directory, budget, progress,
                                        original=method == "untouched")
            witness = parity(actual, expected, masks)
            require(inherited.qualified.tensor_digest(model) == initial_digest and all(p.grad is None for p in model.parameters()),
                    "Qualification changes no model state or gradients")
            record = {"status": "completed", "fit_id": name, "initial_state_sha256": initial_digest,
                      "parity": witness, **metadata, "files": members(directory)}
            write(directory/"completed.json", record)
            qualifications.append(record)
            if method != "untouched":
                del model
        del memory, cache, actual, expected
        gc.collect()
        budget.storage()
        print(json.dumps({"phase": "qualification", "seed": seed, "completed_qualification_paths": len(qualifications)}), flush=True)
    require_qualification(qualifications)
    write(args.out/"qualification.json", {"status": "completed", "paths": qualifications, "training_started": False,
                                         "optimizer_created": False, "temperature_applied": False})
    fits = []
    for seed in SEEDS:
        cache = open_cache(args.out/f"cache-{seed}")
        for method in METHODS:
            name = f"{method}-{seed}"
            progress.update(fit_id=name, operation="adaptation", completed_fits=len(fits))
            model = paired_model(method, seed, {}, initial_states[seed])
            initial_pin = inherited.qualified.tensor_digest(model)
            directory = args.out/name
            directory.mkdir()
            start = budget.elapsed()
            record = adapt(model, name, plan, actors, targets, lexical, rows["dev"], cache, directory, budget, progress)
            record.update(status="completed", fit_id=name, method=method, seed=seed, initial_state_sha256=initial_pin,
                          final_state_sha256=inherited.qualified.tensor_digest(model), wall_seconds=budget.elapsed()-start,
                          files=members(directory))
            write(directory/"completed.json", record)
            fits.append({**record, "completed_sha256": sha(directory/"completed.json")})
            del model
            gc.collect()
            budget.storage()
            print(json.dumps({"phase": "warm-pooling", "completed_fits": len(fits), "fit_id": name}), flush=True)
        del cache
    require([r["fit_id"] for r in fits] == FIT_ORDER, "All twelve fits complete")
    authenticate(args)
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "fit_order": FIT_ORDER, "fits": fits,
            "caches": cache_records, "qualification_paths": qualifications, "row_counts": {k: len(v) for k, v in rows.items()},
            "quality_metrics_computed": False, "official_test_opened": False, "temperature_applied": False,
            "scope": "All old final trained encoders and memories, full original TRAIN/DEV, all qualification before fitting, no outcome selection."}


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
            and launch["clock_backend"] == clock.backend and launch["cap_seconds"] == 7200
            and 0 <= launch["started_ns"] <= started < launch["deadline_ns"]
            and launch["deadline_ns"]-launch["started_ns"] == 7200_000_000_000
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
