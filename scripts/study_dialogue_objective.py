"""Six fresh paired aligned scorers: original stratum CE versus uniform row CE.

Freeze is metadata-only. Train is the only phase loading float caches or Torch.
No quality metrics, old checkpoint loading, encoder calls or objective selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np
import study_dialogue_alignment as old

common, base, typed, schema_prep = old.common, old.base, old.typed, old.schema_prep
ROOT = old.ROOT
VERSION = "dialogue-objective-v1"
MODEL_METHOD = "token_aligned"
METHODS, SEEDS = ("stratum", "uniform"), (6201, 6202, 6203)
ARM_ORDERS = (METHODS, METHODS[::-1], METHODS)
FIT_ORDER = [f"{m}-{s}" for s, order in zip(SEEDS, ARM_ORDERS, strict=True) for m in order]
CONFIG = {**old.CONFIG, "methods": list(METHODS), "seeds": list(SEEDS)}
LIMITS = {"wall_seconds": 6000., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
FREEZE_LIMITS = {"wall_seconds": 60., "rss_bytes": 1024**3, "output_bytes": 128*1024**2}
PARENT_PLAN = ROOT/"output/dialogue-token-alignment-scientific-v1/protocol-01/plan.json"
PARENT_PIN = "e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1"
PARENT_FREEZE_COMPLETED_PIN = "8a2fb2659efed6ac187e92095eef3b3214beacf4ddcbcd7101664c9ef4c57e5f"
NEW_SOURCES = ("scripts/study_dialogue_objective.py", "tests/test_study_dialogue_objective.py",
               "scripts/report_dialogue_objective.py", "tests/test_report_dialogue_objective.py",
               "scripts/audit_dialogue_objective.py", "tests/test_audit_dialogue_objective.py",
               "scripts/audit_dialogue_commitment.py", "tests/test_audit_dialogue_commitment.py",
               "research/dialogue-objective-protocol.md")
ALLOCATION = "Separate six-fit objective comparison, 6000 seconds; prior scientific and cost failures unchanged"
WEIGHTINGS = {"stratum": "Inherited fixed weights; sum weighted CE divided by effective row count",
              "uniform": "Every training row weight exactly one; same effective row denominator"}
INITIALIZATION = "Original common.init_models draw sequence, then two independent full aligned copies; no fitted weights"
require, read, write, sha = old.require, old.read, old.write, old.sha
process_load, additive_work = old.process_load, old.additive_work
load_training_inputs, backend, evaluate = old.load_training_inputs, old.backend, old.evaluate


class Budget(base.Budget):
    def __init__(self, out, start, phase="train"):
        super().__init__(out, start)
        require(phase in ("freeze", "train"), "Unknown budget phase")
        self.limits = FREEZE_LIMITS if phase == "freeze" else LIMITS

    def check(self):
        if time.monotonic()-self.start > self.limits["wall_seconds"]:
            raise TimeoutError("Whole objective campaign wall cap exceeded")
        require(base.peak_rss() <= self.limits["rss_bytes"], "Process-lifetime RSS cap exceeded")

    def storage(self, extra=0):
        self.check()
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(size+extra <= self.limits["output_bytes"], "Output storage cap exceeded")
        return size


def authenticate_parent(budget):
    require(sha(PARENT_PLAN.parent/"completed.json", budget.check) == PARENT_FREEZE_COMPLETED_PIN,
            "External parent freeze completion")
    return old.validate_plan(PARENT_PLAN, PARENT_PIN, budget)


def source_map(parent, check=lambda: None):
    sources = dict(parent["source_sha256"])
    for name in NEW_SOURCES:
        require(name not in sources, "New source collides with frozen source")
        sources[name] = sha(ROOT/name, check)
    require(all(sha(base.safe(ROOT, n), check) == d for n, d in sources.items()), "Source drift")
    return sources


def expected_work(nfit, neval):
    require(nfit > 0 and neval > 0, "Nonempty split")
    def micros(n):
        return sum(math.ceil(min(CONFIG["batch_size"], n-i)/CONFIG["microbatch_size"])
                   for i in range(0, n, CONFIG["batch_size"]))
    return {"updates_per_fit": CONFIG["epochs"]*math.ceil(nfit/CONFIG["batch_size"]),
            "training_microbatches_per_fit": CONFIG["epochs"]*micros(nfit),
            "evaluation_batches_per_fit": math.ceil(neval/CONFIG["batch_size"]),
            "evaluation_microbatches_per_fit": micros(neval)}


def load_orders(directory, nfit, items=None):
    orders = {}
    for seed in SEEDS:
        path = Path(directory)/f"orders-{seed}.npy"
        if items is not None:
            require(items[str(seed)]["file"] == path.name and sha(path) == items[str(seed)]["sha256"], "Orders hash")
        order = np.load(path, allow_pickle=False)
        require(order.dtype == np.int64 and order.shape == (CONFIG["epochs"], nfit)
                and all(np.array_equal(np.sort(row), np.arange(nfit)) for row in order), "Complete paired orders")
        orders[seed] = order
    return orders


def freeze(out, budget, args):
    parent = authenticate_parent(budget)
    index, offsets = schema_prep.authenticate_cache_metadata(parent["schema_cache_path"], parent["schema_completed_sha256"])
    common.schema_metadata(index, offsets)
    _, fit, evaluation, split = common.load_metadata(budget.check)
    require([r["row_index"] for r in fit] == parent["fit_row_indices"]
            and [r["row_index"] for r in evaluation] == parent["evaluation_row_indices"]
            and typed.objective(fit) == parent["objective"] and base.headers() == parent["feature_headers"]
            and split["heldout_services"] == parent["split"]["heldout_services"], "Parent rows/objective/headers")
    load_orders(PARENT_PLAN.parent, len(fit), parent["orders"])
    sources = source_map(parent, budget.check)
    for name, digest in sources.items():
        dest = out/"sources"/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        require(sha(dest, budget.check) == digest, "Snapshot identity")
    for seed in SEEDS:
        shutil.copyfile(PARENT_PLAN.parent/f"orders-{seed}.npy", out/f"orders-{seed}.npy")
    shutil.copyfile(PARENT_PLAN, out/"parent-plan.json")
    shutil.copyfile(PARENT_PLAN.parent/"completed.json", out/"parent-completed.json")
    plan = {**parent, "version": VERSION, "config": CONFIG, "limits": LIMITS, "freeze_limits": FREEZE_LIMITS, "runtime": base.runtime(),
            "source_sha256": sources, "allocation": ALLOCATION, "expected_fits": FIT_ORDER,
            "parent_plan_path": str(PARENT_PLAN), "parent_plan_sha256": PARENT_PIN,
            "parent_freeze_completed_sha256": PARENT_FREEZE_COMPLETED_PIN,
            "initialization": INITIALIZATION, "model_method": MODEL_METHOD,
            "objective_weightings": WEIGHTINGS,
            **expected_work(len(fit), len(evaluation)), "quality_metrics_in_runner": False, "no_retry": True}
    write(out/"plan.json", plan)
    return {"plan_sha256": sha(out/"plan.json"), "source_sha256": sources, "model_calls": 0,
            "encoder_calls": 0, "fit_rows": len(fit), "evaluation_rows": len(evaluation)}


def validate_plan(path, pin, budget):
    path = Path(path)
    require(sha(path, budget.check) == pin, "External plan hash")
    plan, parent = read(path), authenticate_parent(budget)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == LIMITS and plan["freeze_limits"] == FREEZE_LIMITS
            and plan["runtime"] == base.runtime() and plan["expected_fits"] == FIT_ORDER
            and plan["allocation"] == ALLOCATION and plan["initialization"] == INITIALIZATION
            and plan["model_method"] == MODEL_METHOD and plan["objective_weightings"] == WEIGHTINGS
            and plan["quality_metrics_in_runner"] is False
            and plan["no_retry"] is True, "Recipe/runtime/allocation drift")
    # All inherited inputs, row IDs, weights, split, orders and work must be exact.
    changed = {"version", "config", "limits", "source_sha256", "allocation", "expected_fits", "initialization"}
    require(all(plan[k] == v for k, v in parent.items() if k not in changed), "Inherited metadata drift")
    require(plan["source_sha256"] == source_map(parent, budget.check), "Source drift")
    require(plan["parent_plan_path"] == str(PARENT_PLAN) and plan["parent_plan_sha256"] == PARENT_PIN
            and plan["parent_freeze_completed_sha256"] == PARENT_FREEZE_COMPLETED_PIN, "Parent pins")
    done = read(path.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == pin
            and done["version"] == VERSION and done["runtime"] == plan["runtime"]
            and done["source_sha256"] == plan["source_sha256"] and done["limits"] == FREEZE_LIMITS
            and done["model_calls"] == done["encoder_calls"] == 0 and done["no_retry"] is True
            and 0 <= done["wall_seconds"] <= FREEZE_LIMITS["wall_seconds"]
            and 0 <= done["process_lifetime_peak_rss_bytes"] <= FREEZE_LIMITS["rss_bytes"], "Freeze incomplete")
    expected = {"started.json", "plan.json", "parent-plan.json", "parent-completed.json"}
    expected |= {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+p for p in plan["source_sha256"]}
    require(set(done["files"]) == expected and expected | {"completed.json"} ==
            {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}, "Freeze exact files")
    for name, item in done["files"].items():
        p = base.safe(path.parent, name)
        require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Freeze payload drift")
    require(all(done["files"]["sources/"+n]["sha256"] == d for n, d in plan["source_sha256"].items())
            and sha(path.parent/"parent-plan.json", budget.check) == PARENT_PIN
            and sha(path.parent/"parent-completed.json", budget.check) == PARENT_FREEZE_COMPLETED_PIN, "Snapshot identity")
    load_orders(path.parent, len(plan["fit_row_indices"]), plan["orders"])
    return plan


def init_models(torch, seed):
    originals, original_witness = common.init_models(torch, seed)
    models = {m: copy.deepcopy(originals[MODEL_METHOD]) for m in METHODS}
    initial = {m: base.tensor_digest(model, tuple(dict(model.named_parameters()))) for m, model in models.items()}
    expected = original_witness["full_state_sha256"][MODEL_METHOD]
    require(set(initial.values()) == {expected}, "Full paired initializer mismatch")
    for left, right in zip(models[METHODS[0]].parameters(), models[METHODS[1]].parameters(), strict=True):
        require(left.data_ptr() != right.data_ptr(), "Initializer tensor alias")
    return models, {"full_state_sha256": initial,
                    "common_state_sha256": {m: original_witness["common_state_sha256"][MODEL_METHOD] for m in METHODS},
                    "common_tensors": original_witness["common_tensors"], "original_sequence": original_witness}


def loss_weights(rows, objective, weighting):
    require(weighting in METHODS, "Unknown objective weighting")
    return (np.ones(len(rows), np.float32) if weighting == "uniform"
            else typed.sample_weights(rows, objective, "stratum"))


def end_authentication(plan, budget):
    parent = authenticate_parent(budget)
    require(source_map(parent, budget.check) == plan["source_sha256"] and base.runtime() == plan["runtime"],
            "End source/runtime drift")
    base.authenticate_prepared(typed.PREPARED, typed.PREPARED_PIN, budget, payloads=True)
    require(sha(typed.SPLIT_METADATA/"receipt.json", budget.check) == typed.SPLIT_PIN, "End split identity")
    schema_prep.authenticate_cache_metadata(plan["schema_cache_path"], plan["schema_completed_sha256"])
    budget.check()


def update(model, optimizer, weighting, rows, arrays, types, schema, objective, torch, budget):
    """One effective update, including tails. No loss is averaged per microbatch."""
    require(bool(rows), "Empty effective update")
    optimizer.zero_grad(set_to_none=True)
    normalization, work, loss_value, micros = {}, Counter(), 0., 0
    for start in range(0, len(rows), CONFIG["microbatch_size"]):
        budget.check(); local = rows[start:start+CONFIG["microbatch_size"]]
        data, charged = common.actor(local, arrays, types, schema, MODEL_METHOD)
        tensors = {k: torch.from_numpy(v) for k, v in data.items()}
        labels, _ = base.supervision(local)
        labels = torch.from_numpy(labels)
        weights = torch.from_numpy(loss_weights(local, objective, weighting))
        base.bump(budget, "forward_attempted"); scores = model(**tensors); base.bump(budget, "forward_returned")
        stats = base.invariants(scores, tensors["candidate_mask"]); base.merge_invariants(normalization, stats)
        loss = (-scores[torch.arange(len(local)), labels]*weights).sum()/len(rows)
        require(bool(torch.isfinite(loss)), "Nonfinite weighted loss")
        base.bump(budget, "backward_attempted"); loss.backward(); base.bump(budget, "backward_returned")
        base.bump(budget, "training_rows", len(local))
        loss_value += float(loss.detach()); work.update(additive_work(charged)); micros += 1
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Invalid gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
    budget.check()
    base.bump(budget, "optimizer_attempted"); optimizer.step(); base.bump(budget, "optimizer_returned")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Invalid updated parameters")
    return {"weighted_loss": loss_value, "normalization": normalization, "work": dict(work), "microbatches": micros}



def train(out, budget, args):
    plan = validate_plan(args.plan, args.plan_sha256, budget)
    fit, evaluation, types, arrays, schema = load_training_inputs(plan, budget)
    shutil.copyfile(args.plan, out/"plan.json")
    typed.save_rows(out/"evaluation-rows.jsonl", [{**r, "candidate_types": types[r["query_index"]]} for r in evaluation])
    orders = load_orders(Path(args.plan).parent, len(fit), plan["orders"])
    for seed in SEEDS: shutil.copyfile(Path(args.plan).parent/f"orders-{seed}.npy", out/f"orders-{seed}.npy")
    base.save_npz(out/"references.npz", **base.reference_arrays(evaluation, arrays["lexical"]))
    torch = backend(); (out/"fits").mkdir()
    for seed, methods in zip(SEEDS, ARM_ORDERS, strict=True):
        models, witness = init_models(torch, seed)
        for method in methods:
            budget.check(); started = time.monotonic(); name = f"{method}-{seed}"; model = models[method]
            require(base.tensor_digest(model, tuple(dict(model.named_parameters()))) == witness["full_state_sha256"][method], "Fresh untouched initializer")
            dest = out/"fits"/name; dest.mkdir()
            budget.partial_path, budget.partial_model = dest, model
            budget.progress["active_fit"] = {"name": name, "phase": "training", "counts": dict.fromkeys(base.COUNTERS, 0)}
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
            normalization, work = {}, Counter(); model.train()
            with (dest/"updates.jsonl").open("x") as journal:
                for epoch, order in enumerate(orders[seed]):
                    for start in range(0, len(fit), CONFIG["batch_size"]):
                        budget.check(); begin = time.monotonic()
                        batch = [fit[int(i)] for i in order[start:start+CONFIG["batch_size"]]]
                        ids = [r["row_index"] for r in batch]
                        budget.progress["active_fit"]["batch"] = {"epoch": epoch, "start": start, "rows": ids}
                        record = update(model, optimizer, method, batch, arrays, types, schema, plan["objective"], torch, budget)
                        base.merge_invariants(normalization, record["normalization"]); work.update(record["work"])
                        record.update(epoch=epoch, start=start, row_indices=ids, wall_seconds=time.monotonic()-begin)
                        journal.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n"); journal.flush()
                        budget.check()
                    budget.storage()
            require(budget.progress["active_fit"]["counts"]["optimizer_returned"] == plan["updates_per_fit"], "Update coverage")
            with (dest/"weights.pt").open("xb") as stream: torch.save(model.state_dict(), stream)
            training_seconds = time.monotonic()-started
            budget.progress["active_fit"]["phase"] = "evaluation"; evaluation_started = time.monotonic()
            predictions, eval_record = evaluate(model, MODEL_METHOD, evaluation, arrays, types, schema, torch, budget)
            base.save_npz(dest/"predictions.npz", **predictions)
            eval_record["wall_seconds"] = time.monotonic()-evaluation_started
            counts = dict(budget.progress["active_fit"]["counts"])
            u, m, e = plan["updates_per_fit"], plan["training_microbatches_per_fit"], plan["evaluation_microbatches_per_fit"]
            require(counts == {"forward_attempted": m+e, "forward_returned": m+e, "backward_attempted": m,
                    "backward_returned": m, "optimizer_attempted": u, "optimizer_returned": u,
                    "training_rows": CONFIG["epochs"]*len(fit), "evaluation_rows": len(evaluation)}
                    and eval_record["effective_batches"] == plan["evaluation_batches_per_fit"], "Full work counters")
            record = {"status": "completed", "version": VERSION, "method": method, "objective_weighting": method, "model_method": MODEL_METHOD, "seed": seed, "counts": counts,
                "initial_state_sha256": witness["full_state_sha256"][method],
                "initial_common_sha256": witness["common_state_sha256"][method], "initializer_witness": witness,
                "configuration": model.configuration(), "orders_sha256": plan["orders"][str(seed)]["sha256"],
                "epochs": CONFIG["epochs"], "training_effective_batches": u, "training_microbatches": m,
                "training_normalization": normalization, "training_work": dict(work), "training_wall_seconds": training_seconds,
                "evaluation": eval_record, "files": base.prep.manifest(dest, budget.check),
                "wall_seconds": time.monotonic()-started, "plan_sha256": args.plan_sha256,
                "process_lifetime_peak_rss_bytes": base.peak_rss(), "original_capacity_admitted": False,
                "allocation": ALLOCATION}
            write(dest/"completed.json", record)
            budget.progress["completed_fits"].append(name); budget.progress["active_fit"] = None
            budget.partial_model = budget.partial_predictions = budget.partial_path = None
            budget.storage(); print(json.dumps({"completed_fit": name, "wall_seconds": record["wall_seconds"]}), flush=True)
        del models
    require(budget.progress["completed_fits"] == FIT_ORDER, "All six fits required")
    expected_files = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"}
    expected_files |= {f"orders-{seed}.npy" for seed in SEEDS}
    expected_files |= {f"fits/{name}/{file}" for name in FIT_ORDER
                       for file in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    require({p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == expected_files,
            "Execution exact files")
    require(sha(out/"plan.json", budget.check) == args.plan_sha256, "End plan drift")
    load_orders(out, len(fit), plan["orders"])
    end_authentication(plan, budget)
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "expected_fits": FIT_ORDER,
            "completed_fits": FIT_ORDER, "quality_metrics_computed": False, "encoder_calls": 0,
            "schema_completed_sha256": plan["schema_completed_sha256"], "parent_plan_sha256": PARENT_PIN,
            "official_dev_inference": False, "test_contents_accessed": False}



def note(error, message, action):
    try: action()
    except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
        if callable(getattr(error, "add_note", None)): error.add_note(message+repr(secondary))


def preserve_failure(out, budget, error):
    if (out/"completed.json").exists():
        note(error, "Completion demotion: ", lambda: (out/"completed.json").rename(out/"late-completion.json"))
    if budget.partial_model is not None and budget.partial_path is not None:
        def weights():
            import torch
            with (budget.partial_path/"partial-weights.pt").open("xb") as stream: torch.save(budget.partial_model.state_dict(), stream)
        note(error, "Partial weights: ", weights)
    if budget.partial_predictions is not None and budget.partial_path is not None:
        def predictions():
            p = budget.partial_predictions; n = p["completed_rows"]
            base.save_npz(budget.partial_path/"partial-predictions.npz", log_probs=p["log_probs"][:n], row_indices=p["row_indices"][:n])
        note(error, "Partial predictions: ", predictions)
    note(error, "Failure receipt: ", lambda: write(out/"failed.json", {
        "status": "failed", "version": VERSION, "error": repr(error), "progress": budget.progress,
        "wall_seconds": time.monotonic()-budget.start, "no_retry": True,
        "request": getattr(budget, "request", {}), "allocation": ALLOCATION, "limits": budget.limits}))


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, time.monotonic(), args.phase); budget.progress["phase"] = args.phase
    budget.request = {k: str(v) for k, v in vars(args).items()}
    prior = None
    def timeout(_sig, _frame): raise TimeoutError("Whole objective campaign wall cap exceeded")
    try:
        require(args.phase in ("freeze", "train"), "Unknown execution phase")
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, budget.limits["wall_seconds"])
        load_start = process_load()
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "runtime": base.runtime(),
            "request": {k: str(v) for k, v in vars(args).items()}, "no_retry": True, "limits": budget.limits,
            "allocation": ALLOCATION, "process_load": load_start})
        result = (freeze if args.phase == "freeze" else train)(out, budget, args)
        result.update(status="completed", version=VERSION, phase=args.phase, runtime=base.runtime(), no_retry=True,
            allocation=ALLOCATION, progress=budget.progress,
            process_load_start=load_start, process_load_end=process_load(), files=base.prep.manifest(out, budget.check),
            wall_seconds=time.monotonic()-budget.start, process_lifetime_peak_rss_bytes=base.peak_rss(), limits=budget.limits,
            wall_scope="Whole run including authentication, assembly, serialization and hashing; completion write/hash/return cap-checked")
        budget.storage(len(base.prep.encoded(result))); write(out/"completed.json", result); budget.storage()
        return sha(out/"completed.json", budget.check)
    except BaseException as error:
        if prior is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        preserve_failure(out, budget, error)
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="phase", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train"); t.add_argument("--plan", type=Path, required=True)
    t.add_argument("--plan-sha256", required=True); t.add_argument("--out", type=Path, required=True)
    print(json.dumps({"completed_sha256": execute(parser.parse_args())}))
