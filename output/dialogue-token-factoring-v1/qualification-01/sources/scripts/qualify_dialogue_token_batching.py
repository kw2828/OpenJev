"""Exclusive synthetic parity and CPU update-cost qualification, not scientific admission.

freeze reads only an authenticated old plan, source files and runtime metadata.
run constructs only the twelve declared artificial cases. No corpus/encoder or
trained-weight loader is present; the frozen batch assembler is reused unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-token-batching-qualification-v1"
ADD_SOURCES = (
    "src/openjev/research/dialogue_token_batched.py", "tests/test_dialogue_token_batched.py",
    "scripts/qualify_dialogue_token_batching.py", "tests/test_qualify_dialogue_token_batching.py",
    "research/dialogue-token-batching-protocol.md")
SHAPES = ((1, 6, 1, 7, 53), (32, 15, 10, 12, 80), (32, 23, 10, 12, 90))
ARMS = (("slot", "readout"), ("slot", "scalar"), ("candidate", "readout"), ("candidate", "scalar"))
CASES = [{"index": i*4+j, "name": f"{size}-{mode}-{head}", "shape": list(shape),
          "mode": mode, "head": head, "seed": 91201+i*4+j}
         for i, (size, shape) in enumerate(zip(("minimum", "medium", "large"), SHAPES, strict=True))
         for j, (mode, head) in enumerate(ARMS)]
RECIPE = {"cases": CASES, "device": "cpu", "dtype": "float32", "threads": 4, "interop_threads": 1,
    "deterministic": True, "input_dim": 384, "projection_dim": 64, "hidden_dim": 64, "gru_width": 16,
    "parameters": 173186, "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.,
    "output_atol": 1e-5, "output_rtol": 1e-4, "loss_atol": 1e-6, "loss_rtol": 1e-5,
    "gradient_atol": 1e-5, "gradient_rtol": 1e-4, "warm_updates_per_path": 1, "measured_pairs": 4,
    "pair_orders": [["original", "batched"], ["batched", "original"], ["original", "batched"], ["batched", "original"]],
    "minimum_speed_ratio": .90, "other_speed_ratio": 1.10, "maximum_rss_bytes": 6*1024**3,
    "whole_cap_seconds": 300., "optimizer_updates": 120, "parity_forward_backward_passes": 24,
    "reset": "Both paths restored from original post-warmup parameters and AdamW state outside each measured interval"}
COVERAGE_KEYS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
    "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks", "result_checks", "mass_checks")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, "Duplicate JSON key")
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON: "+v)))


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, allow_nan=False, sort_keys=True, indent=2)
        f.write("\n")


def bind(path, digest):
    require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), "Invalid digest")
    require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == digest, "Source/input hash mismatch: "+str(path))


def runtime():
    return {"python": platform.python_version(), "torch": str(importlib.metadata.version("torch")),
            "numpy": str(np.__version__), "platform": platform.platform()}


def source_path(name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe source path")
    p = ROOT/name
    require(p.resolve().is_relative_to(ROOT.resolve()), "Source escapes repository")
    return p


def parent_plan(path, expected):
    bind(path, expected)
    p = read(path)
    require(p["study"] == "dialogue-token-v1" and len(p["source_sha256"]) == 35
            and {"scripts/study_dialogue_tokens.py", "src/openjev/research/dialogue_token_memory.py"} <= set(p["source_sha256"]), "Original 35-source plan required")
    require(p["runtime"] == runtime(), "Original runtime differs")
    for key in ("learning_rate", "weight_decay", "gradient_clip", "threads", "projection_dim", "hidden_dim", "gru_width"):
        require(p["config"][key] == RECIPE[key], "Original optimization recipe differs")
    counts = p["loss_counts"]
    require(set(counts) == {"0", "1", "2"} and all(type(v) is int and v > 0 for v in counts.values()), "Original loss counts")
    require(p["loss_weights"] == [sum(counts.values())/(3*counts[str(i)]) for i in range(3)], "Original loss weights differ")
    for name, digest in p["source_sha256"].items():
        bind(source_path(name), digest)
    return p


@contextmanager
def attempt(out, phase, request, *, capped=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress = {"phase": phase, "completed_cases": 0, "parity_forward_attempted": 0, "parity_forward_returned": 0,
        "parity_backward_attempted": 0, "parity_backward_returned": 0, "update_forward_attempted": 0,
        "update_forward_returned": 0, "update_backward_attempted": 0, "update_backward_returned": 0,
        "optimizer_steps_attempted": 0, "optimizer_steps_returned": 0}
    previous = None
    def check():
        if capped and time.perf_counter()-started > RECIPE["whole_cap_seconds"]:
            raise TimeoutError("Fixed 300-second qualification cap exceeded")
    def alarm(_signum, _frame):
        raise TimeoutError("Fixed 300-second qualification alarm expired")
    try:
        write(out/"started.json", {"phase": phase, "request": request, "source_sha256": sha(__file__), "no_retry": True})
        if capped:
            require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
            previous = signal.signal(signal.SIGALRM, alarm)
            signal.setitimer(signal.ITIMER_REAL, max(.001, RECIPE["whole_cap_seconds"]-(time.perf_counter()-started)))
        yield out, started, progress, check
        check()
    except BaseException as error:
        if (out/"completed.json").exists():
            try:
                (out/"completed.json").rename(out/"late-completion.json")
            except BaseException as secondary:  # noqa: BLE001 - keep original failure
                error.add_note("Completion demotion: "+repr(secondary))
        try:
            write(out/"failed.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                "wall_seconds": time.perf_counter()-started, "progress": progress, "no_retry": True, "resume_authorized": False})
        except BaseException as secondary:  # noqa: BLE001 - keep original failure
            error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def snapshot(out, mapping):
    for name, digest in mapping.items():
        bind(source_path(name), digest)
        destination = out/"sources"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(name), destination)
        bind(destination, digest)


def freeze(args):
    with attempt(args.out, "freeze", {k: str(v) for k, v in vars(args).items()}) as (out, started, _, check):
        parent = parent_plan(args.training_plan, args.training_plan_sha256)
        require(Path(args.protocol).resolve() == source_path(ADD_SOURCES[-1]).resolve(), "Wrong qualification protocol")
        bind(args.protocol, args.protocol_sha256)
        mapping = {**parent["source_sha256"], **{n: sha(source_path(n)) for n in ADD_SOURCES}}
        require(len(mapping) == 40, "Expected 35 inherited plus five added sources")
        snapshot(out, mapping)
        shutil.copyfile(args.training_plan, out/"training-plan.json")
        plan = {"version": VERSION, "recipe": RECIPE, "runtime": runtime(), "source_sha256": mapping,
            "training_plan_sha256": args.training_plan_sha256, "protocol_sha256": args.protocol_sha256,
            "loss_weights": parent["loss_weights"], "loss_counts": parent["loss_counts"],
            "scope": "Artificial CPU parity/cost only; no corpus or trained weights; no full-training admission"}
        write(out/"plan.json", plan)
        validate_plan(out/"plan.json", sha(out/"plan.json"), require_complete=False)
        check()
        write(out/"completed.json", {"status": "completed", "phase": "freeze", "plan_sha256": sha(out/"plan.json"),
            "source_sha256": mapping, "wall_seconds": time.perf_counter()-started, "model_calls": 0, "no_retry": True})
    return sha(out/"plan.json")


def validate_plan(path, expected, *, require_complete=True):
    path = Path(path)
    bind(path, expected)
    p = read(path)
    require(p["version"] == VERSION and p["recipe"] == RECIPE and p["runtime"] == runtime(), "Frozen recipe/runtime changed")
    parent = parent_plan(path.parent/"training-plan.json", p["training_plan_sha256"])
    require(set(p["source_sha256"]) == set(parent["source_sha256"]) | set(ADD_SOURCES)
            and len(p["source_sha256"]) == 40 and p["loss_weights"] == parent["loss_weights"]
            and p["loss_counts"] == parent["loss_counts"], "Source/loss closure")
    require(p["protocol_sha256"] == p["source_sha256"][ADD_SOURCES[-1]], "Protocol source binding")
    for name, digest in p["source_sha256"].items():
        bind(source_path(name), digest)
        bind(path.parent/"sources"/name, digest)
        require(name not in parent["source_sha256"] or parent["source_sha256"][name] == digest, "Inherited source changed")
    require(not any((path.parent/n).exists() for n in ("failed.json", "late-completion.json")), "Failed freeze")
    if require_complete:
        done = read(path.parent/"completed.json")
        require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == expected
                and done["source_sha256"] == p["source_sha256"] and done["model_calls"] == 0 and done["no_retry"] is True, "Freeze incomplete")
    return p


def sample(case):
    """Full public streams with deliberately ragged lengths; labels only enter CE."""
    b, t, q, c, length = case["shape"]
    rng = np.random.default_rng(case["seed"])
    sizes = [max(1, t-i%4) for i in range(b)]
    turns = sum(sizes)
    features = rng.standard_normal((turns+q*(c+1), 384), dtype=np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    queries = [{"text": turns+i*(c+1), "candidates": list(range(turns+i*(c+1)+1, turns+i*(c+1)+1+max(3, c-i%3)))} for i in range(q)]
    lengths = [max(1, length-i%13) for i in range(turns)]
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    tokens = rng.standard_normal((int(offsets[-1]), 384), dtype=np.float32)
    priors = np.concatenate([np.full(n, 1/n, np.float32) for n in lengths])
    ds, lexical, turn_cursor, lexical_cursor = [], [], 0, 0
    for i, dt in enumerate(sizes):
        dq = max(1, q-i%3)
        dc = max(len(x["candidates"]) for x in queries[:dq])
        block = rng.integers(0, 2, size=(dt, dq, dc, 10)).astype(np.float32)
        rows = []
        for qi in range(dq):
            nc = len(queries[qi]["candidates"])
            block[:, qi, nc:] = 0
            for ti in range(dt):
                # Some unscored public updates remain; no history is truncated.
                if (ti+qi+i) % 4 != 3:
                    binname = ("unmentioned_retention", "assigned_retention", "revision")[(ti+qi+i)%3]
                    rows.append({"query": qi, "time": ti, "label": int(rng.integers(nc)), "bin": binname, "unseen": False})
        ds.append({"id": f"synthetic-{i}", "turns": list(range(turn_cursor, turn_cursor+dt)), "queries": rows,
            "token_contexts": np.arange(turn_cursor, turn_cursor+dt, dtype=np.int64),
            "layout": {"shape": [dt, dq, dc, 10], "query_ids": list(range(dq)), "offset": lexical_cursor}})
        lexical.append(block.reshape(-1))
        lexical_cursor += block.size
        turn_cursor += dt
    return ds, queries, features, np.concatenate(lexical), (tokens, offsets, priors)


def digest(value):
    """Typed recursive input/state identity, without serializing executable objects."""
    h = hashlib.sha256()
    def visit(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().contiguous().numpy()
        if isinstance(x, np.ndarray):
            h.update(json.dumps([str(x.dtype), list(x.shape)]).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        else:
            h.update(json.dumps([type(x).__name__, x], allow_nan=False, sort_keys=True).encode())
    visit(value)
    return h.hexdigest()


def backend():
    import study_dialogue_tokens as study
    import torch

    from openjev.research.dialogue_token_batched import DialogueTokenBatchedMemory
    class MonitoredBatched(study.v2.MonitoredCopyMemoryV2, DialogueTokenBatchedMemory):
        pass
    return study, torch, {"original": study.MonitoredTokenMemory, "batched": MonitoredBatched}


def compare(a, b, atol, rtol, label):
    require(tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype, label+" shape/dtype")
    x, y = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    require(np.array_equal(np.isfinite(x), np.isfinite(y)) and not np.isnan(x).any() and not np.isnan(y).any()
            and np.array_equal(np.isposinf(x), np.isposinf(y)) and np.array_equal(np.isneginf(x), np.isneginf(y)), label+" finite support")
    require(not np.isposinf(x).any() and (label == "Outputs" or np.isfinite(x).all()), label+" nonfinite value")
    valid = np.isfinite(x)
    difference = np.abs(x[valid].astype(np.float64)-y[valid].astype(np.float64))
    bound = atol+rtol*np.abs(x[valid].astype(np.float64))
    require((difference <= bound).all(), label+f" parity tolerance; max_abs={difference.max(initial=0):.9g}; max_scaled={(difference/bound).max(initial=0):.9g}")
    return {"max_abs": float(difference.max(initial=0)), "max_tolerance_fraction": float((difference/bound).max(initial=0)),
            "finite_elements": int(valid.sum()), "negative_infinity_elements": int(np.isneginf(x).sum())}


def monitor(model, actor, ds, head, study):
    _, t = actor[1].shape
    q = actor[4].shape[1]
    n = int(actor[1].sum())*q
    actual = model.audit
    expected = {"forward_calls": 1, "forward_returned": 1, "advance_calls": t, "advance_returned": t,
        "valid_turns": int(actor[1].sum()), "executed_valid_question_slots": n,
        "real_question_updates": sum(d["layout"]["shape"][0]*d["layout"]["shape"][1] for d in ds),
        "incoming_checks": n, "feature_checks": n, "result_checks": n, "mass_checks": n if head == "scalar" else 0}
    require({k: actual[k] for k in COVERAGE_KEYS} == expected, "Actual monitor coverage differs")
    for k in study.MAX_KEYS:
        require(math.isfinite(actual[k]) and 0 <= actual[k] <= 2e-6, "Normalization witness exceeded")
    return dict(actual)


def objective(scores, labels, bins, weight, torch):
    eligible = labels != -100
    loss = (torch.nn.functional.cross_entropy(scores[eligible], labels[eligible], reduction="none")*weight[bins[eligible]]).mean()
    require(bool(torch.isfinite(loss)), "Nonfinite weighted loss")
    return loss


def parity(models, data, case, weight, study, torch, progress, emit, check):
    saved, starts = {}, time.perf_counter()
    masks, input_identity = None, None
    for name, model in models.items():
        actor, labels, bins = study.make_batch(*data, case["mode"])
        actor = [x.detach().clone().requires_grad_(True) if x.is_floating_point() else x.clone() for x in actor]
        current = digest([x for x in actor if not x.is_floating_point()])
        current_inputs = digest(actor)
        require(masks is None or current == masks, "Paired input masks differ")
        require(input_identity is None or current_inputs == input_identity, "Paired actor tensors differ")
        masks = current
        input_identity = current_inputs
        model.zero_grad(set_to_none=True)
        model.begin_batch(actor, data[0])
        progress["active"] = {"case": case["name"], "path": name, "phase": "parity_forward"}
        progress["parity_forward_attempted"] += 1
        try:
            scores = model(*actor)
        finally:
            progress["active_monitor"] = dict(model.audit)
        progress["parity_forward_returned"] += 1
        loss = objective(scores, labels, bins, weight, torch)
        progress["parity_backward_attempted"] += 1
        loss.backward()
        progress["parity_backward_returned"] += 1
        saved[name] = {"scores": scores.detach().clone(), "loss": loss.detach().clone(),
            "inputs": [None if x.grad is None else x.grad.detach().clone() for x in actor],
            "parameters": {k: None if v.grad is None else v.grad.detach().clone() for k, v in model.named_parameters()},
            "monitor": monitor(model, actor, data[0], case["head"], study)}
        emit({"kind": "parity", "case": case["name"], "path": name, "forward_backward_completed": True,
              "loss": float(loss.detach()), "monitor": saved[name]["monitor"]})
        del scores, loss, actor
        check()
    a, b = saved["original"], saved["batched"]
    result = {"passed": True, "mask_sha256": masks, "actor_sha256": input_identity, "seconds": time.perf_counter()-starts,
        "outputs": compare(a["scores"], b["scores"], 1e-5, 1e-4, "Outputs"),
        "loss": compare(a["loss"], b["loss"], 1e-6, 1e-5, "Loss"), "inputs": {}, "parameters": {}}
    for kind in ("inputs", "parameters"):
        keys = range(len(a[kind])) if kind == "inputs" else a[kind]
        require(len(a[kind]) == len(b[kind]), "Gradient key membership")
        if kind == "parameters":
            require(set(a[kind]) == set(b[kind]), "Parameter gradient names differ")
        for key in keys:
            x, y = a[kind][key], b[kind][key]
            require((x is None) == (y is None), f"{kind}/{key} absent gradient differs")
            result[kind][str(key)] = None if x is None else compare(x, y, 1e-5, 1e-4, f"{kind}/{key}")
    return result


def update(model, optimizer, data, case, weight, study, torch, progress, check):
    start = time.perf_counter()
    actor, labels, bins = study.make_batch(*data, case["mode"])
    shapes, work = study.old.work_counts(data[0], actor), study.observation_work(data[0], data[1], actor)
    progress["active_actor_shapes"], progress["active_observation_work"] = shapes, work
    model.begin_batch(actor, data[0])
    optimizer.zero_grad(set_to_none=True)
    assembled = time.perf_counter()
    progress["update_forward_attempted"] += 1
    try:
        scores = model(*actor)
    finally:
        progress["active_monitor"] = dict(model.audit)
    progress["update_forward_returned"] += 1
    inv = monitor(model, actor, data[0], case["head"], study)
    loss = objective(scores, labels, bins, weight, torch)
    forward = time.perf_counter()
    progress["update_backward_attempted"] += 1
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    progress["update_backward_returned"] += 1
    backward = time.perf_counter()
    progress["optimizer_steps_attempted"] += 1
    optimizer.step()
    progress["optimizer_steps_returned"] += 1
    optimized = time.perf_counter()
    scalar = float(loss.detach())
    end = time.perf_counter()
    check()
    return {"seconds": end-start, "phase_seconds": {"assembly": assembled-start, "forward_validation_loss": forward-assembled,
        "backward_clip": backward-forward, "optimizer": optimized-backward, "scalar_readback": end-optimized},
        "loss": scalar, "actor_shapes": shapes, "observation_work": work, "monitor": inv, "supervised_queries": int((labels != -100).sum())}


def run_case(case, weight, study, torch, classes, progress, emit, check):
    start = time.perf_counter()
    data = sample(case)
    input_sha = digest(data)
    generated = time.perf_counter()
    models = {}
    with torch.random.fork_rng(devices=[]):
        for name, cls in classes.items():
            torch.manual_seed(case["seed"])
            models[name] = cls(case["head"], attention_mode=case["mode"]).train()
    initial = copy.deepcopy(models["original"].state_dict())
    initial_sha = digest(initial)
    require(all(sum(p.numel() for p in m.parameters()) == 173186 and digest(m.state_dict()) == initial_sha
                and all(p.dtype == torch.float32 and p.device.type == "cpu" for p in m.parameters()) for m in models.values()), "Paired model initializer differs")
    construction = time.perf_counter()-generated
    result = {"case": case, "sample_sha256": input_sha, "initial_sha256": initial_sha,
        "generation_and_hash_seconds": generated-start, "construction_seconds": construction,
        "parity": parity(models, data, case, weight, study, torch, progress, emit, check), "warm": {}, "measured": []}
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001) for n, m in models.items()}
    for name in classes:
        models[name].load_state_dict(initial, strict=True)
        progress["active"] = {"case": case["name"], "path": name, "phase": "warm_update"}
        row = update(models[name], optimizers[name], data, case, weight, study, torch, progress, check)
        result["warm"][name] = row
        emit({"kind": "warm", "case": case["name"], "path": name, **row})
    canonical_weights = copy.deepcopy(models["original"].state_dict())
    canonical_optimizer = copy.deepcopy(optimizers["original"].state_dict())
    result["measured_initial_sha256"] = digest([canonical_weights, canonical_optimizer])
    reset_seconds = 0.
    for pair, order in enumerate(RECIPE["pair_orders"]):
        rows = {}
        for name in order:
            stamp = time.perf_counter()
            models[name].load_state_dict(canonical_weights, strict=True)
            optimizers[name].load_state_dict(copy.deepcopy(canonical_optimizer))
            require(digest([models[name].state_dict(), optimizers[name].state_dict()]) == result["measured_initial_sha256"], "Measured reset differs")
            reset_seconds += time.perf_counter()-stamp
            progress["active"] = {"case": case["name"], "path": name, "phase": "measured_update", "pair": pair}
            row = update(models[name], optimizers[name], data, case, weight, study, torch, progress, check)
            rows[name] = row
            emit({"kind": "measured", "case": case["name"], "path": name, "pair": pair, **row})
        result["measured"].append({"pair": pair, "order": order, "paths": rows, "speed_ratio": rows["original"]["seconds"]/rows["batched"]["seconds"]})
    result["reset_seconds"] = reset_seconds
    result["wall_seconds"] = time.perf_counter()-start
    result["process_lifetime_peak_rss_bytes"] = peak_rss()
    return result


def peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024))


def aggregate(cases, events, rss):
    require([r["case"] for r in cases] == CASES, "Exact twelve-case order/identity required")
    expected = []
    cells = []
    for case, row in zip(CASES, cases, strict=True):
        name = case["name"]
        expected.extend((name, "parity", path, None) for path in ("original", "batched"))
        expected.extend((name, "warm", path, None) for path in ("original", "batched"))
        require(set(row["warm"]) == {"original", "batched"} and len(row["measured"]) == 4, "Timing coverage")
        ratios = []
        for i, pair in enumerate(row["measured"]):
            require(pair["pair"] == i and pair["order"] == RECIPE["pair_orders"][i] and set(pair["paths"]) == {"original", "batched"}, "Paired timing order")
            for path in pair["order"]:
                expected.append((name, "measured", path, i))
            a, b = [pair["paths"][path]["seconds"] for path in ("original", "batched")]
            require(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in (a, b)), "Invalid measured seconds")
            require(pair["speed_ratio"] == a/b, "Paired ratio differs")
            ratios.append(a/b)
        threshold = .90 if case["index"] < 4 else 1.10
        median = statistics.median(ratios)
        cells.append({"case": name, "parity_passed": row["parity"]["passed"] is True, "paired_ratios": ratios,
            "median_speed_ratio": median, "minimum_ratio": threshold, "speed_passed": median >= threshold})
    require([(r["case"], r["kind"], r["path"], r.get("pair")) for r in events] == expected, "Partial/duplicate/foreign operation ledger")
    require(type(rss) is int and rss > 0, "Invalid process RSS")
    passed = all(r["parity_passed"] and r["speed_passed"] for r in cells) and rss <= RECIPE["maximum_rss_bytes"]
    return {"engineering_admission": passed, "all_parity_passed": all(r["parity_passed"] for r in cells),
        "cells": cells, "optimizer_updates": sum(r["kind"] != "parity" for r in events),
        "parity_forward_backward_passes": sum(r["kind"] == "parity" for r in events),
        "process_lifetime_peak_rss_bytes": rss, "rss_passed": rss <= RECIPE["maximum_rss_bytes"],
        "full_training_authorized": False, "scope": "Synthetic single-update correctness/cost only; not a scientific result or full-study timing forecast"}


def run(args):
    with attempt(args.out, "run", {k: str(v) for k, v in vars(args).items()}, capped=True) as (out, start, progress, check):
        plan = validate_plan(args.plan, args.plan_sha256)
        snapshot(out, plan["source_sha256"])
        shutil.copyfile(args.plan, out/"plan.json")
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
                result = run_case(case, weight, study, torch, classes, progress, emit, check)
                write(out/"cases"/(case["name"]+".json"), result)
                results.append(result)
                progress["completed_cases"] += 1
                for key in ("active", "active_monitor", "active_actor_shapes", "active_observation_work"):
                    progress.pop(key, None)
                check()
        summary = aggregate(results, events, peak_rss())
        for kind, expected in (("parity_forward", 24), ("parity_backward", 24), ("update_forward", 120),
                               ("update_backward", 120), ("optimizer_steps", 120)):
            require(progress[kind+"_attempted"] == progress[kind+"_returned"] == expected, "Completed numerical work differs")
        write(out/"summary.json", summary)
        validate_plan(args.plan, args.plan_sha256)
        files = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()}
        check()
        write(out/"completed.json", {"status": "completed", "version": VERSION, "plan_sha256": args.plan_sha256,
            "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "files": files,
            "progress": progress, "engineering_admission": summary["engineering_admission"], "full_training_authorized": False,
            "wall_seconds": time.perf_counter()-start, "no_retry": True, "corpus_reads": 0, "encoder_calls": 0,
            "scope": "Process lifetime RSS, not per-method peak; construction/reset/parity excluded only from update timings"})
    return sha(out/"completed.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("freeze")
    for key in ("training-plan", "training-plan-sha256", "protocol", "protocol-sha256", "out"):
        f.add_argument("--"+key, required=True)
    r = sub.add_parser("run")
    for key in ("plan", "plan-sha256", "out"):
        r.add_argument("--"+key, required=True)
    args = parser.parse_args()
    print(json.dumps({"phase": args.phase, "sha256": freeze(args) if args.phase == "freeze" else run(args)}))
