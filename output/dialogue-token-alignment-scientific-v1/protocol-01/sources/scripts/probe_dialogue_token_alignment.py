"""Metadata-stratified synthetic capacity, never corpus features or fit quality.

Freeze records fresh study orders and public geometry. Run measures the shared
actor with synthetic index-addressed arrays and the actual model/update path.
The extrapolation is an engineering heuristic, not a measured training runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import dialogue_alignment_common as common
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-token-alignment-capacity-v1"
SELF = "scripts/probe_dialogue_token_alignment.py"
TEST = "tests/test_probe_dialogue_token_alignment.py"
PROTOCOL = "research/dialogue-token-alignment-capacity-protocol.md"
TYPED_PLAN = "output/dialogue-typed-v1/protocol-01/plan.json"
TYPED_PIN = "1558d64b88b981d625ae1b4d001bb46b0c2984d6e5dd78a7e595f35981772c07"
METHODS = ("flat_stratum", "token_mean", "token_aligned")
RECIPE = {"seeds": [6201, 6202, 6203], "epochs": 20, "batch_size": 256,
          "microbatch_size": 32, "strata": 3, "warm_events": 1, "measured_events": 3,
          "synthetic_seed": 410, "dtype": "float32", "threads": 4, "interop_threads": 1,
          "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1.,
          "probe_seconds": 300., "rss_bytes": 6*1024**3, "output_bytes": 64*1024**2,
          "study_ceiling_seconds": 3600., "admission_seconds": 2880.}
require, read, write = common.base.require, common.base.read, common.base.write


def sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            h.update(block)
            check()
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Unsafe member")
    path = Path(root)/name
    require(path.resolve().is_relative_to(Path(root).resolve()), "Member escapes directory")
    return path


def manifest(path, check):
    return {p.relative_to(path).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p, check)}
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


@contextmanager
def attempt(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    progress = {"events_started": 0, "events_completed": 0, "optimizer_attempted": 0,
                "optimizer_returned": 0, "forward_attempted": 0, "forward_returned": 0,
                "backward_attempted": 0, "backward_returned": 0, "active": None}
    def check():
        require(common.base.peak_rss() <= RECIPE["rss_bytes"], "Process-lifetime RSS cap")
        if time.perf_counter()-start > RECIPE["probe_seconds"]:
            raise TimeoutError("Whole probe wall cap")
    def expire(_signal, _frame):
        raise TimeoutError("Whole probe alarm")
    request = {k: str(v) for k, v in vars(args).items()}
    previous = None
    try:
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "request": request,
                                  "runtime": common.base.runtime(), "recipe": RECIPE, "no_retry": True})
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        previous = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, max(.001, RECIPE["probe_seconds"]-(time.perf_counter()-start)))
        yield out, start, progress, check
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"completed.json").exists():
                (out/"completed.json").rename(out/"late-completion.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                  "error": repr(error), "progress": progress, "wall_seconds": time.perf_counter()-start,
                  "process_lifetime_peak_rss_bytes": common.base.peak_rss(), "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original error
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure receipt: "+repr(secondary))
        raise
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def finish(out, start, record, check):
    record.update(status="completed", version=VERSION, no_retry=True,
                  files=manifest(out, check), process_lifetime_peak_rss_bytes=common.base.peak_rss())
    record["wall_seconds"] = time.perf_counter()-start
    record["wall_scope"] = "Start before authentication through payload hashes; terminal write/hash/return cap-checked"
    total = sum(v["bytes"] for v in record["files"].values())
    require(total+len(json.dumps(record).encode())+4096 <= RECIPE["output_bytes"], "Output byte cap")
    write(out/"completed.json", record)
    require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= RECIPE["output_bytes"], "Final byte cap")
    return sha(out/"completed.json", check)


def public_row(row):
    """No actual target, bin, previous label or outcome enters synthetic samples."""
    return {"row_index": row["row_index"], "query_index": row["query_index"],
            "candidate_count": row["candidate_count"], "cache": dict(row["cache"]),
            "previous_current_index": row["row_index"] % row["candidate_count"]}


def workload(g):
    """Fixed composite ordering proxy, not an empirical runtime bound."""
    d = 64
    return (384*d*(g["padded_context_token_positions"]+g["padded_schema_token_positions"])
            + 4*d*d*g["padded_comparison_positions"]+3*d*g["padded_pairwise_positions"])


def effective_geometry(rows, schema):
    g = common.effective_geometry(rows, schema)
    parts = [common.geometry(rows[i:i+RECIPE["microbatch_size"]], schema)
             for i in range(0, len(rows), RECIPE["microbatch_size"])]
    g["padded_comparison_positions"] = sum(p["rows"]*p["max_candidate_count"]*
        (p["max_token_length"]+p["max_schema_token_length"]) for p in parts)
    g["maximum_microbatch_float_bytes"] = max(common.base.batch_work(
        rows[i:i+RECIPE["microbatch_size"]], "candidate")["float_input_bytes"]+p["schema_float_input_bytes"]
        for i, p in zip(range(0, len(rows), RECIPE["microbatch_size"]), parts, strict=True))
    return g


def schedule(fit, evaluation, orders, schema, check=lambda: None):
    result = {"train": [], "evaluation": []}
    b = RECIPE["batch_size"]
    for seed in RECIPE["seeds"]:
        for epoch, order in enumerate(orders[seed]):
            for start in range(0, len(order), b):
                rows = [fit[int(i)] for i in order[start:start+b]]
                g = effective_geometry(rows, schema)
                result["train"].append({"seed": seed, "epoch": epoch, "start": start,
                                         "geometry": g, "workload": workload(g)})
            check()
        for start in range(0, len(evaluation), b):
            g = effective_geometry(evaluation[start:start+b], schema)
            result["evaluation"].append({"seed": seed, "epoch": None, "start": start,
                                         "geometry": g, "workload": workload(g)})
    return result


def stratify(records):
    require(len(records) >= RECIPE["strata"], "Too few workload batches")
    ranked = sorted(range(len(records)), key=lambda i: (records[i]["workload"], i))
    groups = np.array_split(ranked, RECIPE["strata"])
    output = []
    for number, ids in enumerate(groups):
        maximum = max(records[int(i)]["workload"] for i in ids)
        selected = min(int(i) for i in ids if records[int(i)]["workload"] == maximum)
        output.append({"stratum": number, "batches_per_arm": len(ids),
                       "minimum_workload": min(records[int(i)]["workload"] for i in ids),
                       "maximum_workload": maximum, "representative_index": selected,
                       "component_maxima": {k: max(records[int(i)]["geometry"][k] for i in ids)
                                            for k in records[selected]["geometry"]},
                       "representative": records[selected]})
    return output


def schema_metadata(path, pin):
    # This helper hashes float payloads opaquely and decodes integer metadata only.
    from prepare_dialogue_schema_tokens import authenticate_cache_metadata
    index, offsets = authenticate_cache_metadata(path, pin)
    return common.schema_metadata(index, offsets)


def sources(schema_cache, protocol, protocol_pin, check):
    require(Path(protocol).resolve() == (ROOT/PROTOCOL).resolve(), "Protocol path")
    require(sha(protocol, check) == protocol_pin, "Protocol pin")
    inherited = read(Path(schema_cache)/"plan.json")["source_sha256"]
    mapping = common.source_map(check)
    for name, expected in inherited.items():
        require(sha(safe(ROOT, name), check) == expected
                and (name not in mapping or mapping[name] == expected), "Inherited source drift")
        mapping[name] = expected
    for name in (SELF, TEST, PROTOCOL):
        mapping[name] = sha(ROOT/name, check)
    return mapping


def freeze(args):
    with attempt(args) as (out, start, _progress, check):
        require(common.CONFIG["batch_size"] == RECIPE["batch_size"]
                and common.CONFIG["microbatch_size"] == RECIPE["microbatch_size"]
                and common.CONFIG["epochs"] == RECIPE["epochs"] and list(common.SEEDS) == RECIPE["seeds"], "Common recipe drift")
        require(sha(ROOT/TYPED_PLAN, check) == TYPED_PIN, "Original typed plan pin")
        schema = schema_metadata(args.schema_cache, args.schema_completed_sha256)
        queries, fit, evaluation, split = common.load_metadata(check)
        old_plan = read(ROOT/TYPED_PLAN)
        require([r["row_index"] for r in fit] == old_plan["fit_row_indices"]
                and [r["row_index"] for r in evaluation] == old_plan["evaluation_row_indices"]
                and common.typed.objective(fit) == old_plan["objective"], "Fixed split/objective")
        types = common.typed.public_candidate_types(queries)
        fit, evaluation = [public_row(r) for r in fit], [public_row(r) for r in evaluation]
        headers = common.base.headers()
        orders = {}
        for seed in RECIPE["seeds"]:
            rng = np.random.default_rng(seed)
            orders[seed] = np.stack([rng.permutation(len(fit)) for _ in range(RECIPE["epochs"])]).astype(np.int64)
            np.save(out/f"orders-{seed}.npy", orders[seed], allow_pickle=False)
        batches = schedule(fit, evaluation, orders, schema, check)
        strata = {phase: stratify(values) for phase, values in batches.items()}
        write(out/"rows.json", {"fit": fit, "evaluation": evaluation, "candidate_types": types})
        write(out/"schedule.json", batches)
        write(out/"schema-index.json", schema["index"])
        np.save(out/"schema-offsets.npy", schema["offsets"], allow_pickle=False)
        mapping = sources(args.schema_cache, args.protocol, args.protocol_sha256, check)
        for name, expected in mapping.items():
            dest = out/"sources"/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe(ROOT, name), dest)
            require(sha(dest, check) == expected, "Source snapshot")
        payloads = manifest(out, check)
        plan = {"version": VERSION, "recipe": RECIPE, "runtime": common.base.runtime(),
                "source_sha256": mapping, "protocol_sha256": args.protocol_sha256,
                "schema_cache": str(Path(args.schema_cache).resolve()), "schema_completed_sha256": args.schema_completed_sha256,
                "typed_plan_sha256": TYPED_PIN, "prepared_completed_sha256": common.typed.PREPARED_PIN,
                "split_receipt_sha256": common.typed.SPLIT_PIN, "split": split, "feature_headers": headers,
                "objective": old_plan["objective"], "geometry_strata": strata, "payloads": payloads,
                "fit_rows": len(fit), "evaluation_rows": len(evaluation),
                "schema_preparation": {k: read(Path(args.schema_cache)/"completed.json").get(k)
                                       for k in ("wall_seconds", "setup_seconds", "work", "payload_bytes")},
                "scope": "Synthetic values and labels only; uniform positive token priors, no feature-array decoding or quality metrics",
                "projection": "Per-arm stratum count times maximum measured duration; remainder includes warmups/setup/authentication; heuristic",
                "no_retry": True}
        write(out/"plan.json", plan)
        return finish(out, start, {"phase": "freeze", "plan_sha256": sha(out/"plan.json", check),
                                  "model_calls": 0, "encoder_calls": 0}, check)


def validate_plan(path, pin, check):
    path = Path(path)
    require(sha(path, check) == pin, "External plan hash")
    p = read(path)
    require(p["version"] == VERSION and p["recipe"] == RECIPE
            and p["runtime"] == common.base.runtime() and p["no_retry"] is True, "Recipe/runtime drift")
    require(common.CONFIG["batch_size"] == RECIPE["batch_size"]
            and common.CONFIG["microbatch_size"] == RECIPE["microbatch_size"]
            and common.CONFIG["epochs"] == RECIPE["epochs"] and list(common.SEEDS) == RECIPE["seeds"], "Common recipe drift")
    done = read(path.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze"
            and done["plan_sha256"] == pin and done["model_calls"] == 0, "Freeze incomplete")
    expected = set(p["payloads"]) | {"plan.json", "completed.json"}
    require({f.relative_to(path.parent).as_posix() for f in path.parent.rglob("*") if f.is_file()} == expected
            and set(done["files"]) == expected-{"completed.json"}, "Freeze exact closure")
    for name, item in done["files"].items():
        file = safe(path.parent, name)
        require(file.stat().st_size == item["bytes"] and sha(file, check) == item["sha256"], "Freeze payload drift")
        if name in p["payloads"]:
            require(item == p["payloads"][name], "Plan/receipt payload mismatch")
    require(p["source_sha256"] == sources(p["schema_cache"], ROOT/PROTOCOL, p["protocol_sha256"], check), "Source closure drift")
    for name, expected_sha in p["source_sha256"].items():
        require(sha(path.parent/"sources"/name, check) == expected_sha, "Snapshot identity")
    require(p["typed_plan_sha256"] == TYPED_PIN == sha(ROOT/TYPED_PLAN, check)
            and p["prepared_completed_sha256"] == common.typed.PREPARED_PIN
            and p["split_receipt_sha256"] == common.typed.SPLIT_PIN, "Input identities")
    schema = schema_metadata(p["schema_cache"], p["schema_completed_sha256"])
    require(read(path.parent/"schema-index.json") == schema["index"]
            and np.array_equal(np.load(path.parent/"schema-offsets.npy", allow_pickle=False), schema["offsets"]), "Schema metadata drift")
    rows = read(path.parent/"rows.json")
    require(len(rows["fit"]) == p["fit_rows"] and len(rows["evaluation"]) == p["evaluation_rows"], "Row counts")
    orders = {}
    for seed in RECIPE["seeds"]:
        a = np.load(path.parent/f"orders-{seed}.npy", allow_pickle=False)
        require(a.dtype == np.int64 and a.shape == (RECIPE["epochs"], p["fit_rows"])
                and all(np.array_equal(np.sort(v), np.arange(p["fit_rows"])) for v in a), "Complete orders")
        orders[seed] = a
    batches = schedule(rows["fit"], rows["evaluation"], orders, schema, check)
    require(batches == read(path.parent/"schedule.json")
            and {k: stratify(v) for k, v in batches.items()} == p["geometry_strata"], "Geometry reconstruction")
    return p, rows, orders, schema


class SyntheticArray:
    """Small bank indexed by public addresses, never a real feature file.

    Advanced indexing materializes float32 copies inside the timed assembler.
    Uniform-prior slices model valid support without reading stored float priors.
    Bank reuse and generation overhead differ from actual memory-mapped gathering.
    """
    def __init__(self, shape, rng, *, prior=False):
        self.shape, self.prior = tuple(shape), prior
        self.bank = None if prior else rng.normal(0, .2, size=(1024, *self.shape[1:])).astype(np.float32)

    def __getitem__(self, key):
        if isinstance(key, slice):
            begin, end, stride = key.indices(self.shape[0])
            ids = np.arange(begin, end, stride, dtype=np.int64)
        else:
            ids = np.asarray(key, np.int64)
        require(np.all((ids >= 0) & (ids < self.shape[0])), "Synthetic address range")
        if self.prior:
            require(ids.ndim == 1 and len(ids) > 0, "Prior requires nonempty segment")
            return np.full(len(ids), 1/len(ids), np.float32)
        return self.bank[ids % len(self.bank)].copy()


def synthetic_inputs(plan, schema, seed):
    rng = np.random.default_rng(seed)
    arrays = {name: SyntheticArray(header["shape"], rng, prior=name == "priors")
              for name, header in plan["feature_headers"].items()}
    schema = dict(schema)
    schema["tokens"] = SyntheticArray((int(schema["offsets"][-1]), 384), rng)
    schema["priors"] = SyntheticArray((int(schema["offsets"][-1]),), rng, prior=True)
    h = hashlib.sha256()
    for array in [*arrays.values(), schema["tokens"]]:
        if array.bank is not None:
            h.update(array.bank.tobytes())
    return arrays, schema, h.hexdigest()


def sample_rows(record, phase, rows, orders):
    start, b = record["start"], RECIPE["batch_size"]
    if phase == "train":
        return [rows["fit"][int(i)] for i in orders[record["seed"]][record["epoch"], start:start+b]]
    return rows["evaluation"][start:start+b]


def backend():
    import torch
    torch.set_num_threads(RECIPE["threads"])
    torch.set_num_interop_threads(RECIPE["interop_threads"])
    torch.use_deterministic_algorithms(True)
    return torch


def event(torch, model, optimizer, method, phase, rows, arrays, types, schema,
          labels, weights, stream, identity, progress, check):
    """Whole effective batch: assemble/mask/validate, global-C loss and one update."""
    start = time.perf_counter()
    progress["active"] = identity
    progress["events_started"] += 1
    model.train(phase == "train")
    if phase == "train":
        optimizer.zero_grad(set_to_none=True)
    work, invariants, value = Counter(), {}, 0.
    outputs = np.full((len(rows), 12), -np.inf, np.float32) if phase == "evaluation" else None
    context = torch.enable_grad() if phase == "train" else torch.no_grad()
    with context:
        for begin in range(0, len(rows), RECIPE["microbatch_size"]):
            check()
            local = rows[begin:begin+RECIPE["microbatch_size"]]
            actor, charged = common.actor(local, arrays, types, schema, method)
            tensors = {k: torch.from_numpy(v) for k, v in actor.items()}
            progress["forward_attempted"] += 1
            scores = model(**tensors)
            progress["forward_returned"] += 1
            stats = common.base.invariants(scores, tensors["candidate_mask"])
            common.base.merge_invariants(invariants, stats)
            targets = torch.from_numpy(labels[begin:begin+len(local)])
            mass = torch.from_numpy(weights[begin:begin+len(local)])
            loss = (-scores[torch.arange(len(local)), targets]*mass).sum()/len(rows)
            require(bool(torch.isfinite(loss)), "Finite synthetic loss")
            if phase == "train":
                progress["backward_attempted"] += 1
                loss.backward()
                progress["backward_returned"] += 1
            value += float(loss.detach())
            if outputs is not None:
                outputs[begin:begin+len(local), :scores.shape[1]] = scores.detach().numpy()
            work.update({k: v for k, v in charged.items() if not k.startswith("max_") and "_max_" not in k})
    if phase == "train":
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Finite parameter gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["gradient_clip"], error_if_nonfinite=True)
        progress["optimizer_attempted"] += 1
        optimizer.step()
        progress["optimizer_returned"] += 1
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Finite final parameters")
    record = {**identity, "rows": len(rows), "microbatches": math.ceil(len(rows)/RECIPE["microbatch_size"]),
              "synthetic_weighted_loss": value, "normalization": invariants, "work": dict(work)}
    if outputs is not None:
        record["synthetic_output_sha256"] = hashlib.sha256(outputs.tobytes()).hexdigest()
        record["output_materialized_bytes"] = outputs.nbytes
    # The row serializes all work/validation; flush is charged, not durable fsync.
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n"); stream.flush()
    record["seconds"] = time.perf_counter()-start
    progress["events_completed"] += 1
    progress["active"] = None
    check()
    return record


def project(plan, events, overhead, peak_rss):
    require(math.isfinite(overhead) and overhead >= 0, "Finite projection overhead")
    expected = {(phase, st["stratum"], method, rep) for phase, strata in plan["geometry_strata"].items()
                for st in strata for method in METHODS for rep in range(RECIPE["warm_events"]+RECIPE["measured_events"])}
    require(len(events) == len(expected) and {(e["phase"], e["stratum"], e["method"], e["repeat"]) for e in events} == expected,
            "Complete unique timing events")
    require(all(type(e["stratum"]) is int and type(e["repeat"]) is int
                and e["warmup"] is (e["repeat"] < RECIPE["warm_events"])
                and math.isfinite(e["seconds"]) and e["seconds"] > 0 for e in events), "Valid timing event")
    cells, projected = [], overhead
    for phase, strata in plan["geometry_strata"].items():
        for st in strata:
            for method in METHODS:
                values = [e["seconds"] for e in events if (e["phase"], e["stratum"], e["method"]) ==
                          (phase, st["stratum"], method) and e["repeat"] >= RECIPE["warm_events"]]
                require(len(values) == RECIPE["measured_events"] and all(math.isfinite(v) and v > 0 for v in values), "Finite measured times")
                estimate = st["batches_per_arm"]*max(values)
                projected += estimate
                cells.append({"phase": phase, "stratum": st["stratum"], "method": method,
                              "batches": st["batches_per_arm"], "maximum_measured_seconds": max(values), "projected_seconds": estimate})
    return {"admitted": projected <= RECIPE["admission_seconds"] and peak_rss <= RECIPE["rss_bytes"],
            "projected_seconds": projected, "observed_nonmeasured_seconds": overhead, "cells": cells,
            "study_ceiling_seconds": RECIPE["study_ceiling_seconds"], "admission_seconds": RECIPE["admission_seconds"],
            "scope": "Heuristic from synthetic stratum maxima; uniform priors and small bank gathers are not actual cache access or measured full-study time"}


def process_load():
    """Small process snapshot outside cell timing, no command arguments or env."""
    try:
        result = subprocess.run(["ps", "-axo", "pid=,pcpu=,rss=,comm="], capture_output=True,
                                text=True, timeout=2, check=False)
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split(None, 3)
            if len(fields) == 4:
                rows.append({"pid": int(fields[0]), "cpu_percent": float(fields[1]),
                             "rss_kib": int(fields[2]), "command": fields[3]})
        return {"status": "observed", "exit_code": result.returncode,
                "highest_cpu_processes": sorted(rows, key=lambda v: (-v["cpu_percent"], v["pid"]))[:10]}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": repr(error)}


def run(args):
    with attempt(args) as (out, start, progress, check):
        plan, rows, orders, schema = validate_plan(args.plan, args.plan_sha256, check)
        shutil.copyfile(args.plan, out/"plan.json")
        types = {int(k): v for k, v in rows["candidate_types"].items()}
        load_before = process_load()
        torch = backend()
        events, profiles = [], []
        with (out/"events.jsonl").open("x") as journal, (out/"timings.jsonl").open("x") as timing:
            for phase_index, phase in enumerate(("train", "evaluation")):
                strata = plan["geometry_strata"][phase]
                for st in strata:
                    check(); profile_start = time.perf_counter()
                    seed = RECIPE["synthetic_seed"]+phase_index*3+st["stratum"]
                    sample = sample_rows(st["representative"], phase, rows, orders)
                    arrays, synthetic_schema, bank_sha = synthetic_inputs(plan, schema, seed)
                    rng = np.random.default_rng(seed)
                    labels = np.asarray([rng.integers(r["candidate_count"]) for r in sample], np.int64)
                    stratum_weights = [plan["objective"]["rows"]/(3*n) for n in plan["objective"]["stratum_counts"]]
                    weights = np.asarray([stratum_weights[r["row_index"] % 3] for r in sample], np.float32)
                    models, initial = common.init_models(torch, seed)
                    optimizers = {m: torch.optim.AdamW(v.parameters(), lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
                                  for m, v in models.items()}
                    sample_sha = digest({"rows": sample, "labels": labels.tolist(), "weights": weights.tolist(), "bank_sha256": bank_sha})
                    profile = {"phase": phase, "stratum": st["stratum"], "seed": seed, "sample_sha256": sample_sha,
                               "initializers": initial, "geometry": effective_geometry(sample, schema),
                               "configurations": {m: v.configuration() for m, v in models.items()}}
                    write(out/f"profile-{phase}-{st['stratum']}.json", profile)
                    for rep in range(RECIPE["warm_events"]+RECIPE["measured_events"]):
                        order = METHODS[rep % 3:]+METHODS[:rep % 3]
                        for method in order:
                            identity = {"phase": phase, "stratum": st["stratum"], "method": method,
                                        "repeat": rep, "warmup": rep < RECIPE["warm_events"], "sample_sha256": sample_sha}
                            record = event(torch, models[method], optimizers[method], method, phase, sample, arrays,
                                           types, synthetic_schema, labels, weights, journal, identity, progress, check)
                            events.append(record)
                            timing.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n"); timing.flush()
                    profiles.append({"phase": phase, "stratum": st["stratum"], "seconds": time.perf_counter()-profile_start})
                    del models, optimizers, arrays, synthetic_schema
        measured = math.fsum(e["seconds"] for e in events if not e["warmup"])
        expected_updates = sum(e["phase"] == "train" for e in events)
        expected_forward = sum(e["microbatches"] for e in events)
        expected_backward = sum(e["microbatches"] for e in events if e["phase"] == "train")
        require(progress["events_started"] == progress["events_completed"] == len(events)
                and progress["optimizer_attempted"] == progress["optimizer_returned"] == expected_updates
                and progress["forward_attempted"] == progress["forward_returned"] == expected_forward
                and progress["backward_attempted"] == progress["backward_returned"] == expected_backward
                and progress["active"] is None, "Complete operation counts")
        load_after = process_load()
        # Measured-duration duplication serialization/hashes stay in remainder.
        require(plan["source_sha256"] == sources(plan["schema_cache"], ROOT/PROTOCOL, plan["protocol_sha256"], check), "Final source drift")
        for item in manifest(out, check).values():
            require(item["bytes"] >= 0, "Payload size")
        overhead = time.perf_counter()-start-measured
        projection = project(plan, events, overhead, common.base.peak_rss())
        write(out/"projection.json", projection)
        return finish(out, start, {"phase": "run", "plan_sha256": args.plan_sha256,
                      "technical_complete": True, "admitted": projection["admitted"], "projection": projection,
                      "events": len(events), "progress": progress, "profile_durations": profiles,
                      "process_load": {"before": load_before, "after": load_after},
                      "schema_preparation": plan["schema_preparation"],
                      "encoder_calls": 0, "real_feature_arrays_decoded": 0, "quality_metrics": False,
                      "projection_overhead_scope": "Observed whole probe minus measured update/evaluation durations; includes warmups, synthetic banks, authentication, setup and initial payload hashes; projection/final receipt serialization occurs later",
                      "timing_scope": "Synthetic bank gathering through actor, validations, forward, weighted loss, backward/clip/AdamW where applicable, scalar readback and event journal flush; no durable fsync",
                      "source_sha256": plan["source_sha256"]}, check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    frozen = sub.add_parser("freeze")
    for name in ("schema-cache", "schema-completed-sha256", "protocol", "protocol-sha256", "out"):
        frozen.add_argument("--"+name, required=True)
    measured = sub.add_parser("run")
    for name in ("plan", "plan-sha256", "out"):
        measured.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps({"completed_sha256": (freeze if args.phase == "freeze" else run)(args)}))


if __name__ == "__main__":
    main()
