"""Exclusive factored-head correctness/cost screen; no corpus or encoders.

Reuse pinned public-geometry and numerical helpers without modifying their globals.
No scientific or full-training admission follows from this engineering screen.
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

import qualify_dialogue_token_batching as old
import qualify_dialogue_token_packing as packing
import qualify_dialogue_token_projection as projection

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-token-factoring-qualification-v1"
PRIOR_SOURCE = "scripts/qualify_dialogue_token_projection.py"
PRIOR_SHA256 = "6bc66c348b413000e5b0e02c2a03d9a345973148231e7ce1baf7092b9d40b70c"
PARENT_PLAN_SHA256 = "442fb17865fc95e9ca4a72ef6721e12f5cdb77f65a02c0ad8eb6da80a2dab2a3"
GEOMETRY, GEOMETRY_SHA256 = packing.GEOMETRY, packing.GEOMETRY_SHA256
ADD_SOURCES = ("src/openjev/research/dialogue_token_factored.py", "tests/test_dialogue_token_factored.py",
    "scripts/qualify_dialogue_token_factoring.py", "tests/test_qualify_dialogue_token_factoring.py",
    "research/dialogue-token-factoring-protocol.md")
PATHS = ("original", "factored")
CASES = copy.deepcopy(projection.CASES)
RECIPE = {**copy.deepcopy(projection.RECIPE), "pair_orders": [list(PATHS), list(reversed(PATHS))]*2,
    "projection": "Groupwise biased turn projection/tanh followed by exogenous first-layer projection and schema-dependent hidden fill",
    "feature_factoring": "First-layer exogenous columns inside groups; one schema fill; two state columns at each ordered head update"}
REFERENCE_FIELDS = (
    "scatter_evidence_scalars", "dense_evidence_scalars", "dense_turn_projection_positions",
    "projected_scatter_scalars", "dense_projected_scalars", "projected_bias_fill_positions",
    "projected_skipped_positions", "projected_scatter_bytes_float32", "dense_projected_bytes_float32",
    "dense_feature_macs")
require, sha, read, write, bind = old.require, old.sha, old.read, old.write, old.bind
timed_update = old.update
geometry_counts, geometry_sample = packing.geometry_counts, packing.geometry_sample


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
    """Independent public-mask work: projection, schema fill and state features."""
    w = projection.mask_work(valid, candidate, token)
    b, t = valid.shape
    q, c = candidate.shape[1:]
    s, n, g = w["packed_evidence_positions"], b*t*q*c, w["pooling_groups"]
    k, p, h = b*q*c, 64, 64
    e = 6*p+10
    return {**w, "exogenous_width": e, "group_feature_positions": s, "fill_feature_positions": k,
        "state_feature_positions": n, "group_feature_calls": g, "fill_feature_calls": 1, "state_feature_calls": t,
        "group_exogenous_input_scalars": s*e, "fill_exogenous_input_scalars": k*e,
        "state_feature_input_scalars": 2*n, "grouped_head_schema_scalars": 2*s*p, "grouped_lexical_scalars": 10*s,
        "scatter_hidden_scalars": s*h, "dense_hidden_scalars": n*h, "fill_hidden_scalars": k*h,
        "group_feature_macs": s*e*h, "fill_feature_macs": k*e*h, "state_feature_macs": 2*n*h,
        "factored_feature_macs": (s+k)*e*h+2*n*h, "dense_feature_macs": n*(e+2)*h,
        "scatter_hidden_bytes_float32": 4*s*h, "dense_hidden_bytes_float32": 4*n*h}


def parent_plan(path, expected, *, complete=True):
    bind(source_path(PRIOR_SOURCE), PRIOR_SHA256)
    require(expected == PARENT_PLAN_SHA256, "Exact frozen projection plan required")
    return projection.validate_plan(path, expected, complete=complete)


def freeze(args):
    with old.attempt(args.out, "freeze", {k: str(v) for k, v in vars(args).items()}) as (out, start, _, check):
        parent = parent_plan(args.projection_plan, args.projection_plan_sha256)
        require(Path(args.protocol).resolve() == source_path(ADD_SOURCES[-1]).resolve(), "Factoring protocol path")
        bind(args.protocol, args.protocol_sha256)
        mapping = {**parent["source_sha256"], **{n: sha(source_path(n)) for n in ADD_SOURCES}}
        require(len(mapping) == 56 and len(parent["source_sha256"]) == 51, "Expected 56-source closure")
        snapshot(out, mapping)
        parent_dir = Path(args.projection_plan).parent
        for name, origin in (("projection-plan.json", args.projection_plan), ("projection-completed.json", parent_dir/"completed.json"),
                *((name, parent_dir/name) for name in ("packing-plan.json", "packing-completed.json", "batching-plan.json", "batching-completed.json", "training-plan.json"))):
            shutil.copyfile(origin, out/name)
        plan = {"version": VERSION, "recipe": RECIPE, "runtime": old.runtime(), "source_sha256": mapping,
            "projection_plan_sha256": args.projection_plan_sha256, "projection_completed_sha256": sha(out/"projection-completed.json"),
            "protocol_sha256": args.protocol_sha256, "geometry_sha256": GEOMETRY_SHA256,
            "loss_weights": parent["loss_weights"], "loss_counts": parent["loss_counts"],
            "scope": "Same sixteen artificial workloads, factored first layer only; no scientific/full-training admission"}
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
    require(p["projection_plan_sha256"] == PARENT_PLAN_SHA256, "Exact frozen projection plan required")
    parent = parent_plan(path.parent/"projection-plan.json", p["projection_plan_sha256"], complete=False)
    bind(path.parent/"projection-completed.json", p["projection_completed_sha256"])
    done = read(path.parent/"projection-completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == p["projection_plan_sha256"]
        and done["source_sha256"] == parent["source_sha256"] and done["model_calls"] == 0 and done["no_retry"] is True, "Parent freeze incomplete")
    require(len(p["source_sha256"]) == 56 and set(p["source_sha256"]) == set(parent["source_sha256"]) | set(ADD_SOURCES)
        and p["loss_weights"] == parent["loss_weights"] and p["loss_counts"] == parent["loss_counts"], "Source/loss closure")
    require(p["protocol_sha256"] == p["source_sha256"][ADD_SOURCES[-1]]
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


def backend():
    import study_dialogue_tokens as study
    import torch

    from openjev.research.dialogue_token_factored import DialogueTokenFactoredMemory
    class MonitoredFactored(study.v2.MonitoredCopyMemoryV2, DialogueTokenFactoredMemory):
        pass
    return study, torch, {"original": study.MonitoredTokenMemory, "factored": MonitoredFactored}


def packing_record(model, expected):
    actual = model.last_packing_work
    require(type(actual) is dict and all(type(v) is int for v in actual.values()) and actual == expected, "Actual packing work differs")
    return {"positions_and_scalars": dict(actual),
        "widths": {"raw_evidence": 384, "attention": 64, "projected_evidence": 64,
            "hidden_preactivation": 64, "exogenous_feature": 394, "state_feature": 2},
        "reference_only_fields": list(REFERENCE_FIELDS),
        "float32_payload_bytes": {k: 4*v for k, v in actual.items() if k.endswith("_scalars")
            and k not in REFERENCE_FIELDS},
        "mac_scope": "First feature linear layer only; multiply-accumulates, not total model FLOPs or measured time",
        "scope": "Grouped/schema-fill/state and hidden-scatter payload; previous dense/projected fields reference-only; not traffic or peak memory"}


def parity(models, data, case, weight, study, torch, progress, emit, check, expected):
    # Explicit argument alias only: reuse the reviewed comparator, not its globals.
    aliased = {"original": models["original"], "batched": models["factored"]}
    def record(row):
        row = dict(row)
        row["path"] = "factored" if row["path"] == "batched" else row["path"]
        if row["path"] == "factored":
            row["packing"] = packing_record(models["factored"], expected)
        emit(row)
    try:
        return old.parity(aliased, data, case, weight, study, torch, progress, record, check)
    finally:
        if progress.get("active", {}).get("path") == "batched":
            progress["active"]["path"] = "factored"


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
    require(all((m.input_dim, m.projection_dim, m.hidden_dim, m.gru_width) == (384, 64, 64, 16)
        and sum(p.numel() for p in m.parameters()) == 173186 and old.digest(m.state_dict()) == initial_sha
        and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in m.parameters()) for m in models.values()), "Paired initializer differs")
    result = {"case": case, "sample_sha256": sample_sha, "initial_sha256": initial_sha, "expected_packing_work": expected,
        "geometry_sha256": GEOMETRY_SHA256 if case["index"] >= 12 else None,
        "generation_hash_and_mask_check_seconds": generated-start, "construction_seconds": time.perf_counter()-generated,
        "parity": parity(models, data, case, weight, study, torch, progress, emit, check, expected), "warm": {}, "measured": []}
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for n, m in models.items()}
    def update(name):
        row = timed_update(models[name], optimizers[name], data, case, weight, study, torch, progress, check)
        if name == "factored":
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
        result["measured"].append({"pair": pair, "order": order, "paths": rows, "speed_ratio": rows["original"]["seconds"]/rows["factored"]["seconds"]})
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
    with old.attempt(args.out, "run", {**{k: str(v) for k, v in vars(args).items()}, "factoring_harness_sha256": sha(__file__)}, capped=True) as (out, start, progress, check):
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
            "scope": "Original assembler plus factored forward counters included in update wall; external comparison and receipt formatting outside update timing"})
    return sha(out/"completed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("freeze")
    for key in ("projection-plan", "projection-plan-sha256", "protocol", "protocol-sha256", "out"):
        f.add_argument("--"+key, required=True)
    r = sub.add_parser("run")
    for key in ("plan", "plan-sha256", "out"):
        r.add_argument("--"+key, required=True)
    args = parser.parse_args()
    print(json.dumps({"phase": args.phase, "sha256": freeze(args) if args.phase == "freeze" else run(args)}))
