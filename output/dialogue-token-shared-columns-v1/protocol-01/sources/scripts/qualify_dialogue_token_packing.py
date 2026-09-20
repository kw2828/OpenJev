"""Exclusive synthetic packing parity/cost screen; no corpus, encoder or weights.

The pinned prior harness supplies unchanged dense samples, numerical comparison,
monitored updates and the 300-second failure guard. Its globals are never changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import qualify_dialogue_token_batching as old

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-token-packing-qualification-v1"
PRIOR_SOURCE = "scripts/qualify_dialogue_token_batching.py"
PRIOR_SHA256 = "2fb43630e604d6894454594d5c330838e1c25c5eb854afb5032e74226ddb3985"
GEOMETRY = "output/dialogue-token-packing-v1/geometry-01.json"
GEOMETRY_SHA256 = "286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49"
ADD_SOURCES = ("src/openjev/research/dialogue_token_packed.py", "tests/test_dialogue_token_packed.py",
    "scripts/qualify_dialogue_token_packing.py", "tests/test_qualify_dialogue_token_packing.py",
    "research/dialogue-token-packing-protocol.md", GEOMETRY)
PATHS = ("original", "packed")
CASES = copy.deepcopy(old.CASES)+[{"index": 12+i, "name": f"geometry-{mode}-{head}",
    "shape": [32, 23, 10, 12, 90], "mode": mode, "head": head, "seed": 91301+i}
    for i, (mode, head) in enumerate(old.ARMS)]
RECIPE = {**copy.deepcopy(old.RECIPE), "cases": CASES, "pair_orders": [list(PATHS), list(reversed(PATHS))]*2,
    "optimizer_updates": 160, "parity_forward_backward_passes": 32, "operation_records": 192,
    "geometry_sha256": GEOMETRY_SHA256, "packing_bucket_width": 16}
require, sha, read, write, bind = old.require, old.sha, old.read, old.write, old.bind
timed_update = old.update


def source_path(name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source")
    path = ROOT/name
    require(path.resolve().is_relative_to(ROOT.resolve()), "Source escapes repository")
    return path


def snapshot(out, mapping):
    for name, expected in mapping.items():
        bind(source_path(name), expected)
        target = out/"sources"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(name), target)
        bind(target, expected)


def mask_work(valid, candidate, token):
    """Independent NumPy count reconstruction; no tensor values or target labels."""
    require(valid.dtype == candidate.dtype == token.dtype == np.bool_, "Boolean masks required")
    b, t = valid.shape
    q, c = candidate.shape[1:]
    length = token.shape[-1]
    require(candidate.shape[0] == b and token.shape[:2] == (b, t)
            and candidate.any(-1).all() and token.any(-1)[valid].all(), "Mask geometry")
    supports, lengths = candidate.sum((1, 2)), token.sum(-1)
    turns = np.argwhere(valid)
    groups = {(min(length, 16*((int(lengths[bi, ti])+15)//16)), int(supports[bi])) for bi, ti in turns}
    keys = int(lengths[valid].sum())
    evidence = sum(int(supports[bi]) for bi, _ in turns)
    bucket_rows = sum(min(length, 16*((int(lengths[bi, ti])+15)//16)) for bi, ti in turns)
    scores = sum(min(length, 16*((int(lengths[bi, ti])+15)//16))*int(supports[bi]) for bi, ti in turns)
    return {"real_turns": len(turns), "supported_schema_pairs": int(supports.sum()),
        "packed_token_key_positions": keys, "dense_token_key_positions": b*t*length,
        "packed_score_positions": scores, "dense_score_positions": b*t*q*c*length,
        "packed_evidence_positions": evidence, "dense_evidence_positions": b*t*q*c,
        "pooling_groups": len(groups), "bucket_token_positions": bucket_rows,
        "packed_raw_token_scalars": keys*384, "grouped_raw_token_scalars": bucket_rows*384,
        "grouped_key_scalars": bucket_rows*64, "grouped_schema_scalars": evidence*64,
        "scatter_evidence_scalars": evidence*384, "dense_evidence_scalars": b*t*q*c*384}


def geometry_counts(g):
    require(g["version"] == "dialogue-token-packing-geometry-v1", "Geometry version")
    shape, dialogs = g["maximum_shape"], g["dialogs"]
    require(type(shape) is list and len(shape) == 5 and all(type(x) is int and x > 0 for x in shape), "Maximum shape")
    b, t, q, c, length = shape
    require(type(dialogs) is list and len(dialogs) == b, "Geometry dialogue count")
    for d in dialogs:
        require(set(d) == {"turn_token_lengths", "candidate_counts"} and all(type(d[k]) is list and d[k]
            and all(type(x) is int and x > 0 for x in d[k]) for k in d), "Public geometry lists")
    require([b, max(len(d["turn_token_lengths"]) for d in dialogs), max(len(d["candidate_counts"]) for d in dialogs),
        max(max(d["candidate_counts"]) for d in dialogs), max(max(d["turn_token_lengths"]) for d in dialogs)] == shape, "Geometry maximum differs")
    turns = sum(len(d["turn_token_lengths"]) for d in dialogs)
    tokens = sum(sum(d["turn_token_lengths"]) for d in dialogs)
    real_q = sum(len(d["turn_token_lengths"])*len(d["candidate_counts"]) for d in dialogs)
    real_c = sum(len(d["turn_token_lengths"])*sum(d["candidate_counts"]) for d in dialogs)
    dummy = turns*q-real_q
    interactions = sum(sum(d["turn_token_lengths"])*sum(d["candidate_counts"]) for d in dialogs)
    dummy_interactions = sum(sum(d["turn_token_lengths"])*(q-len(d["candidate_counts"])) for d in dialogs)
    work = {"emitted_bytes": b*t*length*384*4, "emitted_float32_scalars": b*t*length*384,
        "emitted_token_mask_bytes": b*t*length, "emitted_token_prior_bytes": b*t*length*4,
        "evidence_query_projection_positions": b*q*c, "padded_token_positions": b*t*length,
        "pooling_evidence_positions": b*t*q*c, "pooling_score_positions": b*t*q*c*length,
        "raw_cache_prior_bytes_read": tokens*4, "raw_cache_token_bytes_read": tokens*384*4,
        "real_candidate_updates": real_c, "real_public_turns": turns, "real_question_updates": real_q,
        "schema_candidate_projection_positions": b*q*c, "schema_query_projection_positions": b*q,
        "token_key_projection_positions": b*t*length, "turn_projection_positions": b*t*q*c, "valid_token_positions": tokens}
    counts = {"actor_shapes": {"padded_candidate_positions": b*t*q*c, "padded_query_positions": b*t*q,
        "padded_turn_positions": b*t, "real_question_steps": real_q, "real_turns": turns}, "observation_work": work,
        "active_candidate_rows_including_dummy_none": real_c+dummy, "dummy_none_question_updates": dummy,
        "valid_turn_question_slots_including_dummy_none": turns*q,
        "active_candidate_token_interactions_including_dummy_none": interactions+dummy_interactions,
        "real_candidate_token_interactions": interactions, "masked_candidate_rows_in_full_padded_head": b*t*q*c-real_c-dummy,
        "masked_candidate_rows_on_valid_turns": turns*q*c-real_c-dummy, "padded_turn_candidate_rows": (b*t-turns)*q*c}
    require(g["derived_counts"] == counts, "Geometry derived counts differ")
    return counts


def parent_plan(path, expected, *, complete=True):
    bind(source_path(PRIOR_SOURCE), PRIOR_SHA256)
    return old.validate_plan(path, expected, require_complete=complete)


def freeze(args):
    with old.attempt(args.out, "freeze", {k: str(v) for k, v in vars(args).items()}) as (out, start, _, check):
        parent = parent_plan(args.batching_plan, args.batching_plan_sha256)
        require(Path(args.protocol).resolve() == source_path(ADD_SOURCES[-2]).resolve()
                and Path(args.geometry).resolve() == source_path(GEOMETRY).resolve(), "Protocol/geometry path")
        bind(args.protocol, args.protocol_sha256)
        require(args.geometry_sha256 == GEOMETRY_SHA256, "Fixed geometry identity")
        bind(args.geometry, args.geometry_sha256)
        geometry_counts(read(args.geometry))
        mapping = {**parent["source_sha256"], **{n: sha(source_path(n)) for n in ADD_SOURCES}}
        require(len(mapping) == 46 and len(parent["source_sha256"]) == 40, "Expected 46-source closure")
        snapshot(out, mapping)
        for name, origin in (("batching-plan.json", args.batching_plan),
                ("batching-completed.json", Path(args.batching_plan).parent/"completed.json"),
                ("training-plan.json", Path(args.batching_plan).parent/"training-plan.json")):
            shutil.copyfile(origin, out/name)
        plan = {"version": VERSION, "recipe": RECIPE, "runtime": old.runtime(), "source_sha256": mapping,
            "batching_plan_sha256": args.batching_plan_sha256, "batching_completed_sha256": sha(out/"batching-completed.json"),
            "protocol_sha256": args.protocol_sha256, "geometry_sha256": args.geometry_sha256,
            "loss_weights": parent["loss_weights"], "loss_counts": parent["loss_counts"],
            "scope": "Public mask metadata plus artificial values/labels only; no scientific or full-training admission"}
        write(out/"plan.json", plan)
        validate_plan(out/"plan.json", sha(out/"plan.json"), complete=False)
        check()
        write(out/"completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": sha(out/"plan.json"),
            "source_sha256": mapping, "wall_seconds": time.perf_counter()-start, "model_calls": 0, "no_retry": True})
    return sha(out/"plan.json")


def validate_plan(path, expected, *, complete=True):
    path = Path(path)
    bind(path, expected)
    p = read(path)
    require(p["version"] == VERSION and p["recipe"] == RECIPE and p["runtime"] == old.runtime(), "Frozen recipe/runtime differs")
    parent = parent_plan(path.parent/"batching-plan.json", p["batching_plan_sha256"], complete=False)
    bind(path.parent/"batching-completed.json", p["batching_completed_sha256"])
    done = read(path.parent/"batching-completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == p["batching_plan_sha256"]
        and done["source_sha256"] == parent["source_sha256"] and done["model_calls"] == 0 and done["no_retry"] is True, "Parent freeze incomplete")
    require(len(p["source_sha256"]) == 46 and set(p["source_sha256"]) == set(parent["source_sha256"]) | set(ADD_SOURCES)
        and p["loss_weights"] == parent["loss_weights"] and p["loss_counts"] == parent["loss_counts"], "Source/loss closure")
    require(p["protocol_sha256"] == p["source_sha256"][ADD_SOURCES[-2]]
        and p["geometry_sha256"] == p["source_sha256"][GEOMETRY] == GEOMETRY_SHA256, "Protocol/geometry binding")
    for name, expected_sha in p["source_sha256"].items():
        bind(source_path(name), expected_sha)
        bind(path.parent/"sources"/name, expected_sha)
        require(name not in parent["source_sha256"] or parent["source_sha256"][name] == expected_sha, "Inherited source changed")
    geometry_counts(read(path.parent/"sources"/GEOMETRY))
    require(not any((path.parent/n).exists() for n in ("failed.json", "late-completion.json")), "Failed freeze")
    if complete:
        done = read(path.parent/"completed.json")
        require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == expected
            and done["source_sha256"] == p["source_sha256"] and done["model_calls"] == 0 and done["no_retry"] is True, "Freeze incomplete")
    return p


def geometry_sample(case, geometry):
    geometry_counts(geometry)
    require(case["shape"] == geometry["maximum_shape"], "Case geometry shape")
    rng = np.random.default_rng(case["seed"])
    dialogs = geometry["dialogs"]
    lengths = [n for d in dialogs for n in d["turn_token_lengths"]]
    turns = len(lengths)
    features = rng.standard_normal((turns+sum(sum(d["candidate_counts"])+len(d["candidate_counts"]) for d in dialogs), 384), dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    tokens = rng.standard_normal((int(offsets[-1]), 384), dtype=np.float32)
    priors = np.concatenate([np.full(n, 1/n, np.float32) for n in lengths])
    ds, queries, lexical, ti0, lex0, feature0 = [], [], [], 0, 0, turns
    for i, d in enumerate(dialogs):
        dt, dq, dc = len(d["turn_token_lengths"]), len(d["candidate_counts"]), max(d["candidate_counts"])
        ids, rows = [], []
        block = rng.integers(0, 2, size=(dt, dq, dc, 10)).astype(np.float32)
        for qi, nc in enumerate(d["candidate_counts"]):
            key = len(queries)
            ids.append(key)
            queries.append({"text": feature0, "candidates": list(range(feature0+1, feature0+1+nc))})
            feature0 += 1+nc
            block[:, qi, nc:] = 0
            for ti in range(dt):
                if (ti+qi+i) % 4 != 3:
                    rows.append({"query": key, "time": ti, "label": int(rng.integers(nc)), "unseen": False,
                        "bin": ("unmentioned_retention", "assigned_retention", "revision")[(ti+qi+i)%3]})
        ds.append({"id": f"artificial-geometry-{i}", "turns": list(range(ti0, ti0+dt)), "queries": rows,
            "token_contexts": np.arange(ti0, ti0+dt, dtype=np.int64),
            "layout": {"shape": [dt, dq, dc, 10], "query_ids": ids, "offset": lex0}})
        lexical.append(block.reshape(-1))
        lex0 += block.size
        ti0 += dt
    return ds, queries, features, np.concatenate(lexical), (tokens, offsets, priors)


def backend():
    import study_dialogue_tokens as study
    import torch

    from openjev.research.dialogue_token_packed import DialogueTokenPackedMemory
    class MonitoredPacked(study.v2.MonitoredCopyMemoryV2, DialogueTokenPackedMemory):
        pass
    return study, torch, {"original": study.MonitoredTokenMemory, "packed": MonitoredPacked}


def packing_record(model, expected):
    actual = model.last_packing_work
    require(type(actual) is dict and all(type(v) is int for v in actual.values()) and actual == expected, "Actual packing work differs")
    return {"positions_and_scalars": dict(actual), "float32_payload_bytes": {k: 4*v for k, v in actual.items() if k.endswith("_scalars")},
        "scope": "Forward-internal mask/group counters; payload not physical traffic or peak memory"}


def parity(models, data, case, weight, study, torch, progress, emit, check, expected):
    # Explicit argument alias only: reuse the reviewed comparator, not its globals.
    aliased = {"original": models["original"], "batched": models["packed"]}
    def record(row):
        row = dict(row)
        row["path"] = "packed" if row["path"] == "batched" else row["path"]
        if row["path"] == "packed":
            row["packing"] = packing_record(models["packed"], expected)
        emit(row)
    try:
        return old.parity(aliased, data, case, weight, study, torch, progress, record, check)
    finally:
        if progress.get("active", {}).get("path") == "batched":
            progress["active"]["path"] = "packed"


def run_case(case, geometry, weight, study, torch, classes, progress, emit, check):
    start = time.perf_counter()
    data = old.sample(case) if case["index"] < 12 else geometry_sample(case, geometry)
    sample_sha = old.digest(data)
    actor, _, _ = study.make_batch(*data, case["mode"])
    require([*actor[1].shape, *actor[4].shape[1:], actor[0].shape[2]] == case["shape"], "Assembled maximum shape differs")
    expected = mask_work(*(actor[i].numpy() for i in (1, 4, 6)))
    if case["index"] >= 12:
        counts = geometry_counts(geometry)
        require(study.observation_work(data[0], data[1], actor) == counts["observation_work"], "Assembled geometry work differs")
        actual_shapes = study.old.work_counts(data[0], actor)
        require(all(actual_shapes[k] == v for k, v in counts["actor_shapes"].items()), "Assembled geometry coverage differs")
    del actor
    generated = time.perf_counter()
    models = {}
    with torch.random.fork_rng(devices=[]):
        for name, cls in classes.items():
            torch.manual_seed(case["seed"])
            models[name] = cls(case["head"], attention_mode=case["mode"]).train()
    initial = copy.deepcopy(models["original"].state_dict())
    initial_sha = old.digest(initial)
    require(all(sum(p.numel() for p in m.parameters()) == 173186 and old.digest(m.state_dict()) == initial_sha
        and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in m.parameters()) for m in models.values()), "Paired initializer differs")
    result = {"case": case, "sample_sha256": sample_sha, "initial_sha256": initial_sha, "expected_packing_work": expected,
        "geometry_sha256": GEOMETRY_SHA256 if case["index"] >= 12 else None,
        "generation_hash_and_mask_check_seconds": generated-start, "construction_seconds": time.perf_counter()-generated,
        "parity": parity(models, data, case, weight, study, torch, progress, emit, check, expected), "warm": {}, "measured": []}
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for n, m in models.items()}
    def update(name):
        row = timed_update(models[name], optimizers[name], data, case, weight, study, torch, progress, check)
        if name == "packed":
            row["packing"] = packing_record(models[name], expected)
        return row
    for name in PATHS:
        models[name].load_state_dict(initial, strict=True)
        progress["active"] = {"case": case["name"], "path": name, "phase": "warm_update"}
        result["warm"][name] = update(name)
        emit({"kind": "warm", "case": case["name"], "path": name, **result["warm"][name]})
    canonical = copy.deepcopy(models["original"].state_dict()), copy.deepcopy(optimizers["original"].state_dict())
    result["measured_initial_sha256"] = old.digest(canonical)
    reset_seconds = 0.
    for pair, order in enumerate(RECIPE["pair_orders"]):
        rows = {}
        for name in order:
            stamp = time.perf_counter()
            models[name].load_state_dict(canonical[0], strict=True)
            optimizers[name].load_state_dict(copy.deepcopy(canonical[1]))
            require(old.digest((models[name].state_dict(), optimizers[name].state_dict())) == result["measured_initial_sha256"], "Measured reset differs")
            reset_seconds += time.perf_counter()-stamp
            progress["active"] = {"case": case["name"], "path": name, "phase": "measured_update", "pair": pair}
            rows[name] = update(name)
            emit({"kind": "measured", "case": case["name"], "path": name, "pair": pair, **rows[name]})
        result["measured"].append({"pair": pair, "order": order, "paths": rows, "speed_ratio": rows["original"]["seconds"]/rows["packed"]["seconds"]})
    result.update(reset_seconds=reset_seconds, wall_seconds=time.perf_counter()-start, process_lifetime_peak_rss_bytes=old.peak_rss())
    return result


def aggregate(rows, events, rss):
    require([r["case"] for r in rows] == CASES, "Exact sixteen-case identity/order required")
    expected, cells = [], []
    for case, row in zip(CASES, rows, strict=True):
        name = case["name"]
        expected.extend((name, kind, path, None) for kind in ("parity", "warm") for path in PATHS)
        require(set(row["warm"]) == set(PATHS) and len(row["measured"]) == 4, "Timing coverage")
        ratios = []
        for i, pair in enumerate(row["measured"]):
            require(pair["pair"] == i and pair["order"] == RECIPE["pair_orders"][i] and set(pair["paths"]) == set(PATHS), "Pair order")
            expected.extend((name, "measured", path, i) for path in pair["order"])
            a, b = [pair["paths"][path]["seconds"] for path in PATHS]
            require(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in (a, b)), "Invalid timing")
            require(pair["speed_ratio"] == a/b, "Paired ratio differs")
            ratios.append(a/b)
        threshold = .90 if case["index"] < 4 else 1.10
        cells.append({"case": name, "parity_passed": row["parity"]["passed"] is True, "paired_ratios": ratios,
            "median_speed_ratio": statistics.median(ratios), "minimum_ratio": threshold, "speed_passed": statistics.median(ratios) >= threshold})
    require([(e["case"], e["kind"], e["path"], e.get("pair")) for e in events] == expected, "Operation ledger differs")
    require(type(rss) is int and rss > 0, "Invalid process RSS")
    passed = all(c["parity_passed"] and c["speed_passed"] for c in cells) and rss <= RECIPE["maximum_rss_bytes"]
    return {"engineering_admission": passed, "all_parity_passed": all(c["parity_passed"] for c in cells), "cells": cells,
        "optimizer_updates": sum(e["kind"] != "parity" for e in events), "parity_forward_backward_passes": sum(e["kind"] == "parity" for e in events),
        "operation_records": len(events), "process_lifetime_peak_rss_bytes": rss, "rss_passed": rss <= RECIPE["maximum_rss_bytes"],
        "full_training_authorized": False, "scope": "Synthetic implementation/cost qualification only; one metadata geometry is not representative"}


def run(args):
    with old.attempt(args.out, "run", {**{k: str(v) for k, v in vars(args).items()}, "packing_harness_sha256": sha(__file__)}, capped=True) as (out, start, progress, check):
        plan = validate_plan(args.plan, args.plan_sha256)
        snapshot(out, plan["source_sha256"])
        shutil.copyfile(args.plan, out/"plan.json")
        geometry = read(out/"sources"/GEOMETRY)
        study, torch, classes = backend()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        weight = torch.tensor(plan["loss_weights"], dtype=torch.float32)
        results, events = [], []
        (out/"cases").mkdir()
        with (out/"operations.jsonl").open("x") as journal:
            def emit(row):
                journal.write(json.dumps(row, allow_nan=False, sort_keys=True)+"\n")
                journal.flush()
                events.append(row)
            for case in CASES:
                check()
                result = run_case(case, geometry, weight, study, torch, classes, progress, emit, check)
                write(out/"cases"/(case["name"]+".json"), result)
                results.append(result)
                progress["completed_cases"] += 1
                for key in ("active", "active_monitor", "active_actor_shapes", "active_observation_work"):
                    progress.pop(key, None)
                check()
        summary = aggregate(results, events, old.peak_rss())
        for kind, count in (("parity_forward", 32), ("parity_backward", 32), ("update_forward", 160), ("update_backward", 160), ("optimizer_steps", 160)):
            require(progress[kind+"_attempted"] == progress[kind+"_returned"] == count, "Numerical work incomplete")
        write(out/"summary.json", summary)
        validate_plan(args.plan, args.plan_sha256)
        files = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()}
        check()
        write(out/"completed.json", {"status": "completed", "version": VERSION, "plan_sha256": args.plan_sha256,
            "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "files": files, "progress": progress,
            "engineering_admission": summary["engineering_admission"], "full_training_authorized": False,
            "wall_seconds": time.perf_counter()-start, "no_retry": True, "corpus_reads": 0, "encoder_calls": 0,
            "scope": "Original assembler plus packed forward counters included in update wall; external comparison and receipt formatting outside update timing"})
    return sha(out/"completed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("freeze")
    for key in ("batching-plan", "batching-plan-sha256", "protocol", "protocol-sha256", "geometry", "geometry-sha256", "out"):
        f.add_argument("--"+key, required=True)
    r = sub.add_parser("run")
    for key in ("plan", "plan-sha256", "out"):
        r.add_argument("--"+key, required=True)
    args = parser.parse_args()
    print(json.dumps({"phase": args.phase, "sha256": freeze(args) if args.phase == "freeze" else run(args)}))
