"""Fresh nine-fit token alignment campaign, with no quality scoring in the runner.

The closed capacity screen remains not admitted. This separately frozen allocation
copies its exact orders and uses authenticated immutable caches, never old weights.
Torch and numeric feature arrays are loaded only by the explicit training phase.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

import dialogue_alignment_common as common
import numpy as np
import prepare_dialogue_schema_tokens as schema_prep

base, typed = common.base, common.typed
ROOT = common.ROOT
VERSION = "dialogue-token-alignment-scientific-v1"
METHODS, SEEDS, ARM_ORDERS, FIT_ORDER = common.METHODS, common.SEEDS, common.ARM_ORDERS, common.FIT_ORDER
CONFIG = dict(common.CONFIG)
LIMITS = {"wall_seconds": 7200., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
CAPACITY_PLAN = ROOT/"output/dialogue-token-alignment-v1/capacity-protocol-02/plan.json"
CAPACITY_PIN = "3b28aac097d9c8c4f3eba38d9ee7ab50f3ed9786cd36249e4772005022f20eea"
CAPACITY_COMPLETED_PIN = "d037a378a62f4c81ae55cf28e1e4e7a1c901c5059c0326f05ae18207fd0f33c3"
NEW_SOURCES = ("scripts/study_dialogue_alignment.py", "tests/test_study_dialogue_alignment.py",
               "scripts/report_dialogue_alignment.py", "tests/test_report_dialogue_alignment.py",
               "scripts/report_dialogue_typed_v2.py", "tests/test_dialogue_typed_report_v2.py",
               "research/dialogue-token-alignment-scientific-protocol.md")
ALLOCATION = "Separate pre-quality 7200-second allocation; original cost screen remains not admitted"
require, read, write, sha = base.require, base.read, base.write, base.sha


class Budget(base.Budget):
    """Own all caps; inherited base limits are deliberately not consulted."""

    def check(self):
        if time.monotonic()-self.start > LIMITS["wall_seconds"]:
            raise TimeoutError("Whole alignment scientific campaign wall cap exceeded")
        require(base.peak_rss() <= LIMITS["rss_bytes"], "Process-lifetime RSS cap exceeded")

    def storage(self, extra=0):
        self.check()
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        require(size+extra <= LIMITS["output_bytes"], "Output storage cap exceeded")
        return size


def process_load():
    """Outside fit timing, inside the whole-run budget; no arguments/environment."""
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


def authenticate_capacity(check=lambda: None):
    require(sha(CAPACITY_PLAN, check) == CAPACITY_PIN
            and sha(CAPACITY_PLAN.parent/"completed.json", check) == CAPACITY_COMPLETED_PIN,
            "External capacity freeze identity")
    plan, done = read(CAPACITY_PLAN), read(CAPACITY_PLAN.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze"
            and done["plan_sha256"] == CAPACITY_PIN and done["model_calls"] == 0, "Capacity freeze scope")
    expected = set(plan["payloads"]) | {"plan.json"}
    require(set(done["files"]) == expected and expected | {"completed.json"} ==
            {p.relative_to(CAPACITY_PLAN.parent).as_posix() for p in CAPACITY_PLAN.parent.rglob("*") if p.is_file()},
            "Capacity freeze closure")
    for name, item in done["files"].items():
        path = base.safe(CAPACITY_PLAN.parent, name)
        require(path.stat().st_size == item["bytes"] and sha(path, check) == item["sha256"], "Capacity payload identity")
        if name in plan["payloads"]:
            require(item == plan["payloads"][name], "Capacity plan payload identity")
    for name, digest in plan["source_sha256"].items():
        require(sha(base.safe(ROOT, name), check) == digest
                and done["files"]["sources/"+name]["sha256"] == digest, "Immutable capacity sources")
    require(plan["runtime"] == base.runtime() and plan["prepared_completed_sha256"] == typed.PREPARED_PIN
            and plan["split_receipt_sha256"] == typed.SPLIT_PIN, "Capacity parent scope/runtime")
    return plan


def source_map(parent, check=lambda: None):
    sources = dict(parent["source_sha256"])
    inherited = read(Path(parent["schema_cache"])/"plan.json")["source_sha256"]
    for name, digest in {**inherited, **common.source_map(check)}.items():
        require(name not in sources or sources[name] == digest, "Conflicting inherited source")
        sources[name] = digest
    for name in NEW_SOURCES:
        sources[name] = sha(ROOT/name, check)
    require(all(sha(base.safe(ROOT, n), check) == d for n, d in sources.items()), "Source drift")
    return sources


def expected_work(nfit, neval):
    def micros(n):
        return sum(math.ceil(min(CONFIG["batch_size"], n-i)/CONFIG["microbatch_size"])
                   for i in range(0, n, CONFIG["batch_size"]))
    require(nfit > 0 and neval > 0, "Nonempty split")
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
    parent = authenticate_capacity(budget.check)
    index, offsets = schema_prep.authenticate_cache_metadata(parent["schema_cache"], parent["schema_completed_sha256"])
    common.schema_metadata(index, offsets)
    _, fit, evaluation, split = common.load_metadata(budget.check)
    public = read(CAPACITY_PLAN.parent/"rows.json")
    require([r["row_index"] for r in fit] == [r["row_index"] for r in public["fit"]]
            and [r["row_index"] for r in evaluation] == [r["row_index"] for r in public["evaluation"]]
            and typed.objective(fit) == parent["objective"] and base.headers() == parent["feature_headers"],
            "Original fixed rows/objective/headers")
    load_orders(CAPACITY_PLAN.parent, len(fit))
    sources = source_map(parent, budget.check)
    for name, digest in sources.items():
        dest = out/"sources"/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
        require(sha(dest, budget.check) == digest, "Snapshot identity")
    orders = {}
    for seed in SEEDS:
        dest = out/f"orders-{seed}.npy"; shutil.copyfile(CAPACITY_PLAN.parent/dest.name, dest)
        orders[str(seed)] = {"file": dest.name, "sha256": sha(dest), "shape": [CONFIG["epochs"], len(fit)]}
    split.update(fit=typed.support(fit), evaluation=typed.support(evaluation),
                 primary=typed.support([r for r in evaluation if r["heldout_service"]]))
    plan = {"version": VERSION, "config": CONFIG, "limits": LIMITS, "runtime": base.runtime(),
            "source_sha256": sources, "capacity_plan_path": str(CAPACITY_PLAN), "capacity_plan_sha256": CAPACITY_PIN,
            "capacity_freeze_completed_sha256": CAPACITY_COMPLETED_PIN,
            "original_capacity_admitted": False, "allocation": ALLOCATION,
            "schema_cache_path": parent["schema_cache"], "schema_completed_sha256": parent["schema_completed_sha256"],
            "schema_preparation": parent["schema_preparation"],
            "prepared_path": str(typed.PREPARED), "prepared_completed_sha256": typed.PREPARED_PIN,
            "split_receipt_sha256": typed.SPLIT_PIN, "feature_headers": parent["feature_headers"], "split": split,
            "fit_row_indices": [r["row_index"] for r in fit], "evaluation_row_indices": [r["row_index"] for r in evaluation],
            "objective": typed.objective(fit), "orders": orders, "expected_fits": FIT_ORDER,
            **expected_work(len(fit), len(evaluation)),
            "initialization": "Fresh common.init_models; full mean/aligned equality, ten common tensors equal in all three arms",
            "quality_metrics_in_runner": False, "no_retry": True}
    write(out/"plan.json", plan)
    return {"plan_sha256": sha(out/"plan.json"), "source_sha256": sources, "model_calls": 0,
            "encoder_calls": 0, "fit_rows": len(fit), "evaluation_rows": len(evaluation)}


def validate_plan(path, pin, budget):
    path = Path(path)
    require(sha(path, budget.check) == pin, "External plan hash")
    plan = read(path); parent = authenticate_capacity(budget.check)
    require(plan["version"] == VERSION and plan["config"] == CONFIG and plan["limits"] == LIMITS
            and plan["runtime"] == base.runtime() and plan["expected_fits"] == FIT_ORDER
            and plan["original_capacity_admitted"] is False and plan["allocation"] == ALLOCATION
            and plan["quality_metrics_in_runner"] is False and plan["no_retry"] is True, "Recipe/runtime/allocation drift")
    require(plan["source_sha256"] == source_map(parent, budget.check), "Source drift")
    done = read(path.parent/"completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == pin, "Freeze incomplete")
    expected = {"started.json", "plan.json"} | {f"orders-{s}.npy" for s in SEEDS} | {"sources/"+p for p in plan["source_sha256"]}
    require(set(done["files"]) == expected and expected ==
            {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_file()}-{"completed.json"}, "Freeze exact files")
    for name, item in done["files"].items():
        p = base.safe(path.parent, name)
        require(p.stat().st_size == item["bytes"] and sha(p, budget.check) == item["sha256"], "Freeze payload drift")
    require(all(done["files"]["sources/"+n]["sha256"] == d for n, d in plan["source_sha256"].items()), "Snapshot source identity")
    require(plan["capacity_plan_path"] == str(CAPACITY_PLAN) and plan["capacity_plan_sha256"] == CAPACITY_PIN
            and plan["capacity_freeze_completed_sha256"] == CAPACITY_COMPLETED_PIN
            and plan["schema_cache_path"] == parent["schema_cache"]
            and plan["schema_completed_sha256"] == parent["schema_completed_sha256"]
            and plan["schema_preparation"] == parent["schema_preparation"]
            and plan["prepared_path"] == str(typed.PREPARED) and plan["prepared_completed_sha256"] == typed.PREPARED_PIN
            and plan["split_receipt_sha256"] == typed.SPLIT_PIN, "Input pins")
    for seed in SEEDS:
        require(plan["orders"][str(seed)]["sha256"] == parent["payloads"][f"orders-{seed}.npy"]["sha256"], "Original order identity")
    return plan


def load_training_inputs(plan, budget):
    queries, fit, evaluation, split = common.load_metadata(budget.check)
    require([r["row_index"] for r in fit] == plan["fit_row_indices"]
            and [r["row_index"] for r in evaluation] == plan["evaluation_row_indices"]
            and split["heldout_services"] == plan["split"]["heldout_services"]
            and typed.objective(fit) == plan["objective"] and base.headers() == plan["feature_headers"], "Frozen rows/objective drift")
    require(all(plan[k] == v for k, v in expected_work(len(fit), len(evaluation)).items()), "Frozen work counts")
    arrays = base.load_arrays(plan)
    index, tokens, offsets, priors = schema_prep.authenticate_cache(plan["schema_cache_path"], plan["schema_completed_sha256"])
    schema = common.schema_metadata(index, offsets); schema.update(tokens=tokens, priors=priors)
    budget.check()
    return fit, evaluation, typed.public_candidate_types(queries), arrays, schema


def backend():
    import torch
    torch.set_num_threads(CONFIG["threads"]); torch.set_num_interop_threads(CONFIG["interop_threads"])
    torch.use_deterministic_algorithms(True)
    return torch


def additive_work(work):
    return {k: v for k, v in work.items() if not k.startswith("max_") and "_max_" not in k}


def update(model, optimizer, method, rows, arrays, types, schema, objective, torch, budget):
    """One effective update, including tails. No loss is averaged per microbatch."""
    require(bool(rows), "Empty effective update")
    optimizer.zero_grad(set_to_none=True)
    normalization, work, loss_value, micros = {}, Counter(), 0., 0
    for start in range(0, len(rows), CONFIG["microbatch_size"]):
        budget.check(); local = rows[start:start+CONFIG["microbatch_size"]]
        data, charged = common.actor(local, arrays, types, schema, method)
        tensors = {k: torch.from_numpy(v) for k, v in data.items()}
        labels, _ = base.supervision(local)
        labels = torch.from_numpy(labels)
        weights = torch.from_numpy(typed.sample_weights(local, objective, "stratum"))
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


def evaluate(model, method, rows, arrays, types, schema, torch, budget):
    model.eval(); logs = np.full((len(rows), 12), -np.inf, np.float32)
    totals, work, batches, micros = {}, Counter(), 0, 0
    budget.partial_predictions = {"log_probs": logs, "row_indices": np.asarray([r["row_index"] for r in rows], np.int64), "completed_rows": 0}
    with torch.no_grad():
        for start in range(0, len(rows), CONFIG["batch_size"]):
            batch = rows[start:start+CONFIG["batch_size"]]
            for offset in range(0, len(batch), CONFIG["microbatch_size"]):
                budget.check(); local = batch[offset:offset+CONFIG["microbatch_size"]]
                data, charged = common.actor(local, arrays, types, schema, method)
                tensors = {k: torch.from_numpy(v) for k, v in data.items()}
                base.bump(budget, "forward_attempted"); scores = model(**tensors); base.bump(budget, "forward_returned")
                base.merge_invariants(totals, base.invariants(scores, tensors["candidate_mask"]))
                index = start+offset
                logs[index:index+len(local), :scores.shape[1]] = scores.numpy()
                base.bump(budget, "evaluation_rows", len(local))
                budget.partial_predictions["completed_rows"] = index+len(local)
                work.update(additive_work(charged)); micros += 1
            batches += 1
    return {"log_probs": logs, "row_indices": budget.partial_predictions["row_indices"]}, {
        "normalization": totals, "work": dict(work), "effective_batches": batches, "microbatches": micros}


def end_authentication(plan, budget):
    parent = authenticate_capacity(budget.check)
    require(source_map(parent, budget.check) == plan["source_sha256"] and base.runtime() == plan["runtime"], "End source/runtime drift")
    base.authenticate_prepared(typed.PREPARED, typed.PREPARED_PIN, budget, payloads=True)
    require(sha(typed.SPLIT_METADATA/"receipt.json", budget.check) == typed.SPLIT_PIN, "End split identity")
    schema_prep.authenticate_cache_metadata(plan["schema_cache_path"], plan["schema_completed_sha256"])
    budget.check()


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
        models, witness = common.init_models(torch, seed)
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
            predictions, eval_record = evaluate(model, method, evaluation, arrays, types, schema, torch, budget)
            base.save_npz(dest/"predictions.npz", **predictions)
            eval_record["wall_seconds"] = time.monotonic()-evaluation_started
            counts = dict(budget.progress["active_fit"]["counts"])
            u, m, e = plan["updates_per_fit"], plan["training_microbatches_per_fit"], plan["evaluation_microbatches_per_fit"]
            require(counts == {"forward_attempted": m+e, "forward_returned": m+e, "backward_attempted": m,
                    "backward_returned": m, "optimizer_attempted": u, "optimizer_returned": u,
                    "training_rows": CONFIG["epochs"]*len(fit), "evaluation_rows": len(evaluation)}
                    and eval_record["effective_batches"] == plan["evaluation_batches_per_fit"], "Full work counters")
            record = {"status": "completed", "version": VERSION, "method": method, "seed": seed, "counts": counts,
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
    require(budget.progress["completed_fits"] == FIT_ORDER, "All nine fits required")
    end_authentication(plan, budget)
    return {"plan_sha256": args.plan_sha256, "source_sha256": plan["source_sha256"], "expected_fits": FIT_ORDER,
            "completed_fits": FIT_ORDER, "quality_metrics_computed": False, "encoder_calls": 0,
            "schema_completed_sha256": plan["schema_completed_sha256"], "capacity_plan_sha256": CAPACITY_PIN,
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
        "wall_seconds": time.monotonic()-budget.start, "no_retry": True, "original_capacity_admitted": False,
        "allocation": ALLOCATION, "limits": LIMITS}))


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, time.monotonic()); budget.progress["phase"] = args.phase
    prior = None
    def timeout(_sig, _frame): raise TimeoutError("Whole alignment scientific campaign wall cap exceeded")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        load_start = process_load()
        write(out/"started.json", {"version": VERSION, "phase": args.phase, "runtime": base.runtime(),
            "request": {k: str(v) for k, v in vars(args).items()}, "no_retry": True, "limits": LIMITS,
            "original_capacity_admitted": False, "allocation": ALLOCATION, "process_load": load_start})
        result = (freeze if args.phase == "freeze" else train)(out, budget, args)
        result.update(status="completed", version=VERSION, phase=args.phase, runtime=base.runtime(), no_retry=True,
            original_capacity_admitted=False, allocation=ALLOCATION, progress=budget.progress,
            process_load_start=load_start, process_load_end=process_load(), files=base.prep.manifest(out, budget.check),
            wall_seconds=time.monotonic()-budget.start, process_lifetime_peak_rss_bytes=base.peak_rss(), limits=LIMITS,
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
