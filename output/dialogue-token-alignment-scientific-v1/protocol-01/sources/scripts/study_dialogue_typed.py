"""Matched typed-output versus rare-support diagnostic on an internal TRAIN split.

Reuse authenticated frozen caches and actor assembly, never earlier fitted weights.
No official development inference or official test access. Torch is training-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np
import study_dialogue_conditional as base

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-typed-v1"
PREPARED = ROOT / "runs/dialogue-conditional-v1/preparation-01"
PREPARED_PIN = "960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3"
SPLIT_METADATA = ROOT / "output/dialogue-typed-v1/split-design-01"
SPLIT_PIN = "fca0db6a135eec3c845fe2b8bc5db151c4acdf0bb4b669ed1ff9ca42b7287118"
METHODS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
SEEDS = (6101, 6102, 6103)
ARM_ORDERS = (METHODS, METHODS[1:] + METHODS[:1], METHODS[2:] + METHODS[:2])
FIT_ORDER = [f"{m}-{s}" for s, methods in zip(SEEDS, ARM_ORDERS, strict=True) for m in methods]
VALUES = ("none", "dontcare", "true", "false", "other")
CONFIG = {**base.CONFIG, "methods": list(METHODS), "seeds": list(SEEDS)}
LIMITS = dict(base.LIMITS)
SPLIT_SALT = "openjev-typed-v1:"
NEW_SOURCES = ("scripts/study_dialogue_typed.py", "tests/test_study_dialogue_typed.py",
               "scripts/report_dialogue_typed.py", "tests/test_report_dialogue_typed.py",
               "src/openjev/research/dialogue_typed_observation.py", "tests/test_dialogue_typed_observation.py",
               "research/dialogue-typed-protocol.md")
require, read, write, sha = base.require, base.read, base.write, base.sha


def source_map(check=lambda: None):
    return {**base.source_map(check), **{p: sha(ROOT/p, check) for p in NEW_SOURCES}}


def split_rows(rows, dialogue_services):
    """Selection uses public service identities, not labels or model outcomes."""
    require(rows and all(r["split"] == "train" and r["admission"] == "admitted" for r in rows), "TRAIN admitted only")
    services = sorted({s for names in dialogue_services.values() for s in names})
    require(services and all(isinstance(s, str) and s for s in services), "Public service identities")
    ranking = sorted(services, key=lambda s: (hashlib.sha256((SPLIT_SALT+s).encode()).hexdigest(), s))
    held = set(ranking[:math.ceil(.2*len(ranking))])
    held_dialogues = {d for d, names in dialogue_services.items() if held.intersection(names)}
    fit, evaluation = [], []
    for row in rows:
        require(row["dialogue_id"] in dialogue_services and row["service"] in dialogue_services[row["dialogue_id"]],
                "Row absent from original public service list")
        target = evaluation if row["dialogue_id"] in held_dialogues else fit
        target.append({**row, "heldout_service": row["service"] in held})
    require(fit and evaluation and any(r["heldout_service"] for r in evaluation), "Empty internal split")
    require(not ({r["dialogue_id"] for r in fit} & {r["dialogue_id"] for r in evaluation}), "Dialogue leakage")
    require(not any(r["heldout_service"] for r in fit), "Held-out service leakage")
    return fit, evaluation, {"salt": SPLIT_SALT, "service_ranking": ranking, "heldout_services": sorted(held),
        "public_dialogues": len(dialogue_services), "heldout_public_dialogues": len(held_dialogues),
        "rule": "First ceil(20% public TRAIN services) by SHA256(salt+service), name tie-break; exclude every containing dialogue from fit",
        "historically_unexposed": False, "official_test_accessed": False}


def public_candidate_types(queries):
    types = {}
    for q in queries:
        values = []
        for cid, value in zip(q["candidate_ids"], q["candidate_values"], strict=True):
            if cid == "reserved:NOT_MENTIONED": code = 0
            elif cid == "reserved:DONTCARE": code = 1
            elif q["boolean_slot"]:
                require(isinstance(value, str) and value.strip().casefold() in ("true", "false"), "Boolean schema")
                code = 2 if value.strip().casefold() == "true" else 3
            else: code = 4
            values.append(code)
        require(values.count(0) == values.count(1) == 1 and any(v >= 2 for v in values), "Schema branch support")
        types[q["query_index"]] = values
    return types


def support(rows):
    return {"rows": len(rows), "dialogues": len({r["dialogue_id"] for r in rows}),
            "services": sorted({r["service"] for r in rows}),
            "schema_queries": len({r["query_id"] for r in rows}),
            "strata": dict(Counter(str(base.stratum(r)) for r in rows)),
            "transition_value": dict(sorted(Counter(r["derived_bin"]+"/"+r["current_value_group"] for r in rows).items()))}


def objective(rows):
    counts = Counter((base.stratum(r), VALUES.index(r["current_value_group"])) for r in rows)
    strata = Counter(base.stratum(r) for r in rows)
    require(set(strata) == {0, 1, 2}, "Missing fit stratum")
    n = len(rows)
    k = {s: sum(counts[s, v] > 0 for v in range(5)) for s in range(3)}
    weights = {"stratum": [[n/(3*strata[s]) if counts[s, v] else None for v in range(5)] for s in range(3)],
               "balanced": [[n/(3*k[s]*counts[s, v]) if counts[s, v] else None for v in range(5)] for s in range(3)]}
    for recipe in weights.values():
        require(abs(sum(recipe[s][v]*count for (s, v), count in counts.items())-n) < 1e-8, "Mean weight one")
    return {"rows": n, "value_order": list(VALUES), "counts": [[counts[s, v] for v in range(5)] for s in range(3)],
            "stratum_counts": [strata[s] for s in range(3)], "nonempty_categories": [k[s] for s in range(3)], "weights": weights}


def sample_weights(rows, recipe, weighting):
    require(weighting in ("stratum", "balanced"), "Unknown weighting")
    values = [recipe["weights"][weighting][base.stratum(r)][VALUES.index(r["current_value_group"])] for r in rows]
    require(all(v is not None and math.isfinite(v) and v > 0 for v in values), "Unsupported training category")
    return np.asarray(values, np.float32)


def actor(rows, arrays, candidate_types):
    result, work = base.make_actor(rows, arrays, "candidate")
    codes = np.full(result["candidate_mask"].shape, -1, np.int64)
    for i, row in enumerate(rows):
        public = candidate_types[row["query_index"]]
        require(len(public) == row["candidate_count"], "Candidate-type alignment")
        codes[i, :len(public)] = public
    result["candidate_types"] = codes
    return result, {**work, "candidate_type_input_bytes": codes.nbytes}


def load_metadata(budget, payloads=True):
    base.authenticate_prepared(PREPARED, PREPARED_PIN, budget, payloads=payloads)
    require(sha(SPLIT_METADATA/"receipt.json", budget.check) == SPLIT_PIN, "Original TRAIN service projection")
    receipt = read(SPLIT_METADATA/"receipt.json")
    require(receipt["status"] == "completed" and receipt["alternate_splits_evaluated"] == 0, "Single fixed split")
    for name, item in receipt["files"].items():
        p = base.safe(SPLIT_METADATA, name)
        require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Split projection payload")
    if payloads:
        for name, item in receipt["inputs"].items():
            p = base.safe(ROOT, name)
            require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Split raw input identity")
    queries, all_rows, _ = base.metadata(PREPARED, budget)
    membership = read(SPLIT_METADATA/"membership.json")
    services = {}
    for row in membership["original_train_dialogues"]:
        require(row["dialogue_id"] not in services, "Duplicate public dialogue")
        services[row["dialogue_id"]] = list(row["services"])
    fit, evaluation, split = split_rows(all_rows["train"], services)
    require([r["row_index"] for r in fit] == membership["admitted_row_indices"]["fit"]
            and [r["row_index"] for r in evaluation] == membership["admitted_row_indices"]["evaluation"], "Original raw projection split agreement")
    return queries, fit, evaluation, split


def freeze(out, budget, args):
    _, fit, evaluation, split = load_metadata(budget)
    sources = source_map(budget.check)
    for name in sources:
        p = out/"sources"/name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, p)
    split.update(fit=support(fit), evaluation=support(evaluation),
                 primary=support([r for r in evaluation if r["heldout_service"]]))
    orders = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = np.stack([rng.permutation(len(fit)) for _ in range(CONFIG["epochs"])]).astype(np.int64)
        path = out/f"orders-{seed}.npy"; np.save(path, order, allow_pickle=False)
        orders[str(seed)] = {"file": path.name, "sha256": sha(path), "shape": list(order.shape)}
    plan = {"version": VERSION, "config": CONFIG, "limits": LIMITS, "runtime": base.runtime(),
            "source_sha256": sources, "prepared_path": str(PREPARED), "prepared_completed_sha256": PREPARED_PIN,
            "split_receipt_sha256": SPLIT_PIN, "feature_headers": base.headers(), "split": split,
            "fit_row_indices": [r["row_index"] for r in fit], "evaluation_row_indices": [r["row_index"] for r in evaluation],
            "objective": objective(fit), "orders": orders, "expected_fits": FIT_ORDER,
            "updates_per_fit": CONFIG["epochs"]*math.ceil(len(fit)/CONFIG["batch_size"]),
            "evaluation_batches_per_fit": math.ceil(len(evaluation)/CONFIG["batch_size"]),
            "initialization": "Full state identical across four arms per seed; all weights fresh",
            "quality_metrics_in_runner": False, "no_retry": True}
    write(out/"plan.json", plan)
    return {"plan_sha256": sha(out/"plan.json"), "source_sha256": sources, "model_calls": 0,
            "fit_rows": len(fit), "evaluation_rows": len(evaluation)}


def validate_plan(path, pin, budget):
    require(sha(path) == pin, "External plan hash")
    path = Path(path); plan = read(path)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == LIMITS
            and plan["runtime"] == base.runtime() and plan["expected_fits"] == FIT_ORDER, "Recipe/runtime drift")
    require(plan["source_sha256"] == source_map(budget.check), "Source drift")
    done = read(path.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == pin, "Freeze incomplete")
    expected = {"started.json", "plan.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+p for p in plan["source_sha256"]}
    require(set(done["files"]) == expected
            and expected == {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}-{"completed.json"}, "Freeze exact files")
    for name, item in done["files"].items():
        p = base.safe(path.parent, name)
        require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Freeze payload drift")
    require(all(done["files"]["sources/"+name]["sha256"] == digest for name, digest in plan["source_sha256"].items()), "Snapshot source identity")
    require(plan["prepared_completed_sha256"] == PREPARED_PIN and plan["split_receipt_sha256"] == SPLIT_PIN, "Input pins")
    return plan


def save_rows(path, rows):
    with path.open("x") as stream:
        for r in rows: stream.write(json.dumps(r, sort_keys=True)+"\n")


def references(rows, fit, arrays, candidate_types):
    result = base.reference_arrays(rows, arrays["lexical"])
    counts = Counter(VALUES.index(r["current_value_group"]) for r in fit)
    choices = []
    for r in rows:
        types = candidate_types[r["query_index"]]; sizes = Counter(types)
        choices.append(int(np.argmax([(counts[t]+1)/sizes[t] for t in types])))
    result["type_frequency_indices"] = np.asarray(choices, np.int64)
    return result


def evaluate(model, rows, arrays, types, torch, budget):
    model.eval(); logs = np.full((len(rows), 12), -np.inf, np.float32)
    totals = {}; work = Counter()
    budget.partial_predictions = {"log_probs": logs, "row_indices": np.asarray([r["row_index"] for r in rows], np.int64), "completed_rows": 0}
    with torch.no_grad():
        for start in range(0, len(rows), CONFIG["batch_size"]):
            budget.check(); batch = rows[start:start+CONFIG["batch_size"]]
            data, batch_work = actor(batch, arrays, types)
            tensors = {k: torch.from_numpy(v) for k, v in data.items()}
            base.bump(budget, "forward_attempted"); scores = model(**tensors); base.bump(budget, "forward_returned")
            base.merge_invariants(totals, base.invariants(scores, tensors["candidate_mask"]))
            logs[start:start+len(batch), :scores.shape[1]] = scores.numpy()
            base.bump(budget, "evaluation_rows", len(batch))
            budget.partial_predictions["completed_rows"] = start+len(batch)
            work.update({k: v for k, v in batch_work.items() if not k.startswith("max_")})
    return {"log_probs": logs, "row_indices": budget.partial_predictions["row_indices"]}, {"normalization": totals, "work": dict(work)}


def train(out, budget, args):
    plan = validate_plan(args.plan, args.plan_sha256, budget)
    queries, fit, evaluation, split = load_metadata(budget)
    require([r["row_index"] for r in fit] == plan["fit_row_indices"]
            and [r["row_index"] for r in evaluation] == plan["evaluation_row_indices"]
            and split["heldout_services"] == plan["split"]["heldout_services"]
            and objective(fit) == plan["objective"] and base.headers() == plan["feature_headers"], "Frozen split/objective drift")
    require(plan["updates_per_fit"] == CONFIG["epochs"]*math.ceil(len(fit)/CONFIG["batch_size"])
            and plan["evaluation_batches_per_fit"] == math.ceil(len(evaluation)/CONFIG["batch_size"]), "Frozen work counts")
    shutil.copyfile(args.plan, out/"plan.json")
    types = public_candidate_types(queries)
    save_rows(out/"evaluation-rows.jsonl", [{**r, "candidate_types": types[r["query_index"]]} for r in evaluation])
    orders = {}
    for seed in SEEDS:
        item = plan["orders"][str(seed)]; p = Path(args.plan).parent/item["file"]
        require(sha(p) == item["sha256"], "Orders hash")
        order = np.load(p, allow_pickle=False)
        require(order.dtype == np.int64 and order.shape == (CONFIG["epochs"], len(fit))
                and all(np.array_equal(np.sort(o), np.arange(len(fit))) for o in order), "Complete paired orders")
        orders[seed] = order; shutil.copyfile(p, out/p.name)
    arrays = base.load_arrays(plan)
    base.save_npz(out/"references.npz", **references(evaluation, fit, arrays, types))
    import torch

    from openjev.research.dialogue_typed_observation import DialogueTypedObservation
    torch.set_num_threads(CONFIG["threads"]); torch.set_num_interop_threads(CONFIG["interop_threads"])
    torch.use_deterministic_algorithms(True)
    (out/"fits").mkdir()
    records = []
    for seed, methods in zip(SEEDS, ARM_ORDERS, strict=True):
        torch.manual_seed(seed)
        models = {m: DialogueTypedObservation(m.split("_")[0]) for m in METHODS}
        for m in METHODS[1:]: models[m].load_state_dict(models[METHODS[0]].state_dict(), strict=True)
        hashes = {m: base.tensor_digest(model, tuple(dict(model.named_parameters()))) for m, model in models.items()}
        require(len(set(hashes.values())) == 1, "Full initializer equality")
        for method in methods:
            started = time.monotonic(); name = f"{method}-{seed}"; model = models[method]
            dest = out/"fits"/name; dest.mkdir()
            budget.partial_path, budget.partial_model = dest, model
            budget.progress["active_fit"] = {"name": name, "phase": "training", "counts": dict.fromkeys(base.COUNTERS, 0)}
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            normalization = {}; work = Counter(); model.train()
            with (dest/"updates.jsonl").open("x") as journal:
                for epoch, order in enumerate(orders[seed]):
                    for start in range(0, len(fit), CONFIG["batch_size"]):
                        budget.check(); begin = time.monotonic()
                        batch = [fit[int(i)] for i in order[start:start+CONFIG["batch_size"]]]
                        budget.progress["active_fit"]["batch"] = {"epoch": epoch, "start": start, "rows": [r["row_index"] for r in batch]}
                        data, batch_work = actor(batch, arrays, types)
                        tensors = {k: torch.from_numpy(v) for k, v in data.items()}
                        labels = torch.tensor([r["current_label_index"] for r in batch], dtype=torch.long)
                        weights = torch.from_numpy(sample_weights(batch, plan["objective"], method.split("_")[1]))
                        optimizer.zero_grad(set_to_none=True)
                        base.bump(budget, "forward_attempted"); scores = model(**tensors); base.bump(budget, "forward_returned")
                        stats = base.invariants(scores, tensors["candidate_mask"]); base.merge_invariants(normalization, stats)
                        loss = (-scores[torch.arange(len(batch)), labels]*weights).mean()
                        require(bool(torch.isfinite(loss)), "Nonfinite weighted loss")
                        base.bump(budget, "backward_attempted"); loss.backward(); base.bump(budget, "backward_returned")
                        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), "Invalid gradient")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
                        base.bump(budget, "optimizer_attempted"); optimizer.step(); base.bump(budget, "optimizer_returned")
                        base.bump(budget, "training_rows", len(batch))
                        work.update({k: v for k, v in batch_work.items() if not k.startswith("max_")})
                        journal.write(json.dumps({"epoch": epoch, "start": start, "row_indices": [r["row_index"] for r in batch],
                            "weighted_loss": float(loss.detach()), "normalization": stats, "work": batch_work,
                            "wall_seconds": time.monotonic()-begin}, sort_keys=True, allow_nan=False)+"\n")
                    journal.flush(); budget.storage()
            require(budget.progress["active_fit"]["counts"]["optimizer_returned"] == plan["updates_per_fit"], "Update coverage")
            with (dest/"weights.pt").open("xb") as f: torch.save(model.state_dict(), f)
            training_seconds = time.monotonic()-started
            budget.progress["active_fit"]["phase"] = "evaluation"
            evaluation_started = time.monotonic()
            predictions, eval_record = evaluate(model, evaluation, arrays, types, torch, budget)
            base.save_npz(dest/"predictions.npz", **predictions)
            eval_record["wall_seconds"] = time.monotonic()-evaluation_started
            counts = dict(budget.progress["active_fit"]["counts"])
            u, e = plan["updates_per_fit"], plan["evaluation_batches_per_fit"]
            require(counts == {"forward_attempted": u+e, "forward_returned": u+e, "backward_attempted": u,
                "backward_returned": u, "optimizer_attempted": u, "optimizer_returned": u,
                "training_rows": CONFIG["epochs"]*len(fit), "evaluation_rows": len(evaluation)}, "Full work counters")
            record = {"status": "completed", "method": method, "seed": seed, "counts": counts,
                "initial_state_sha256": hashes[method], "configuration": model.configuration(),
                "orders_sha256": plan["orders"][str(seed)]["sha256"], "training_normalization": normalization,
                "training_wall_seconds": training_seconds,
                "training_work": dict(work), "evaluation": eval_record, "files": base.prep.manifest(dest, budget.check),
                "wall_seconds": time.monotonic()-started, "plan_sha256": args.plan_sha256}
            write(dest/"completed.json", record); records.append(record)
            budget.progress["completed_fits"].append(name); budget.progress["active_fit"] = None
            budget.partial_model = budget.partial_predictions = budget.partial_path = None
            budget.storage()
            print(json.dumps({"completed_fit": name, "wall_seconds": record["wall_seconds"]}), flush=True)
        del models
    require(budget.progress["completed_fits"] == FIT_ORDER, "All twelve fits required")
    require(source_map(budget.check) == plan["source_sha256"] and base.runtime() == plan["runtime"], "End source/runtime drift")
    base.authenticate_prepared(PREPARED, PREPARED_PIN, budget, payloads=True)
    require(sha(SPLIT_METADATA/"receipt.json", budget.check) == SPLIT_PIN, "End split identity")
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "expected_fits": FIT_ORDER,
        "completed_fits": FIT_ORDER, "quality_metrics_computed": False, "encoder_calls": 0,
        "official_dev_inference": False, "test_contents_accessed": False}


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); budget = base.Budget(out, start)
    budget.progress["phase"] = args.phase
    def timeout(_sig, _frame): raise TimeoutError("Whole typed study wall cap exceeded")
    prior = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
    try:
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "runtime": base.runtime(),
            "request": {k: str(v) for k, v in vars(args).items()}, "no_retry": True})
        result = (freeze if args.phase == "freeze" else train)(out, budget, args)
        result.update(status="completed", version=VERSION, phase=args.phase, runtime=base.runtime(), no_retry=True,
            progress=budget.progress, files=base.prep.manifest(out, budget.check),
            wall_seconds=time.monotonic()-start, process_lifetime_peak_rss_bytes=base.peak_rss(), limits=LIMITS,
            wall_scope="Whole run through payload hashes; completion write/hash/return also cap-checked")
        budget.storage(len(base.prep.encoded(result))); write(out/"completed.json", result); budget.storage()
        return sha(out/"completed.json", budget.check)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if (out/"completed.json").exists():
            try: (out/"completed.json").rename(out/"late-completion.json")
            except BaseException as secondary:  # noqa: BLE001 - retain original error
                error.add_note("Completion demotion: "+repr(secondary))
        if budget.partial_model is not None:
            try:
                import torch
                with (budget.partial_path/"partial-weights.pt").open("xb") as f: torch.save(budget.partial_model.state_dict(), f)
            except BaseException as secondary:  # noqa: BLE001 - retain original error
                error.add_note("Partial weights preservation: "+repr(secondary))
        if budget.partial_predictions is not None and budget.partial_path is not None:
            try:
                p = budget.partial_predictions; n = p["completed_rows"]
                base.save_npz(budget.partial_path/"partial-predictions.npz", log_probs=p["log_probs"][:n], row_indices=p["row_indices"][:n])
            except BaseException as secondary:  # noqa: BLE001 - retain original error
                error.add_note("Partial prediction preservation: "+repr(secondary))
        try:
            write(out/"failed.json", {"status": "failed", "error": repr(error), "progress": budget.progress,
                "wall_seconds": time.monotonic()-start, "version": VERSION, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain original error
            error.add_note("Failure receipt preservation: "+repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train"); t.add_argument("--plan", type=Path, required=True)
    t.add_argument("--plan-sha256", required=True); t.add_argument("--out", type=Path, required=True)
    print(json.dumps({"completed_sha256": execute(parser.parse_args())}))
