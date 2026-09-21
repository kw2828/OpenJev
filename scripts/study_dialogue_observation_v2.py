"""Twelve fresh autonomous scalar fits with frozen/trainable observation encoders.

Source-only draft until an external, hash-pinned allocation and completed cost
qualification exist. Freeze loads metadata only; train saves final predictions
without quality scoring or checkpoint selection. No implicit resource budget.
V2 changes timing, supervision and provenance only; perf_counter phase
diagnostics and the scientific route remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import qualify_dialogue_observation_batches as cost

from openjev.research.suspend_clock import Deadline, SuspendClock, _duration_ns

ROOT = cost.ROOT
VERSION = "dialogue-observation-scientific-v2"
ALLOCATION_VERSION = "dialogue-observation-allocation-v2"
PROTOCOL = "research/dialogue-observation-learning-protocol-v2.md"
V1_PLAN = "output/dialogue-observation-learning-v1/scientific-freeze-01/plan.json"
V1_PLAN_PIN = "acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237"
FAILED_MANIFEST = "output/dialogue-observation-learning-v1/failed-scientific-publication-01/manifest.json"
FAILED_MANIFEST_PIN = "41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f"
CLOCK_SOURCE = "src/openjev/research/suspend_clock.py"
SUPERVISOR_SOURCE = "scripts/supervise_dialogue_observation_v2.py"
SEEDS = [6901, 6902, 6903]
ARMS = list(cost.ARMS)
FIT_ORDER = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
DEV_ENDPOINTS = 62329
OLD_NEW_SOURCES = ["scripts/study_dialogue_observation.py", "tests/test_study_dialogue_observation.py",
               "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py",
               "scripts/report_dialogue_observation.py", "tests/test_report_dialogue_observation.py",
               "research/dialogue-observation-learning-scoring.md", "research/dialogue-observation-learning-protocol.md",
               "output/dialogue-observation-learning-v1/study-watchdog-01.py"]
NEW_SOURCES = ["scripts/study_dialogue_observation_v2.py", "tests/test_study_dialogue_observation_v2.py",
               "scripts/report_dialogue_observation_v2.py", "tests/test_report_dialogue_observation_v2.py",
               SUPERVISOR_SOURCE, "tests/test_supervise_dialogue_observation_v2.py", CLOCK_SOURCE,
               "tests/test_suspend_clock.py", PROTOCOL,
               "output/dialogue-observation-learning-v2/result-audit-01/audit.py",
               "output/dialogue-observation-learning-v2/result-audit-01/test_audit.py"]
ALLOCATION_KEYS = {"version", "prepared_completed_sha256", "prepared_plan_sha256", "cost_completed",
                   "cost_audit", "protocol", "limits", "freeze_limits", "failed_attempt"}
require, sha, read, write = cost.require, cost.sha, cost.read, cost.write


def members(directory):
    return {p.relative_to(directory).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "completed.json"}


def check_manifest(directory, manifest):
    require({p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
             and p != directory / "completed.json"} == set(manifest), "Exact payload closure")
    for name, entry in manifest.items():
        path = directory / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Safe manifest path")
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Payload identity: " + name)


def allocation(path, pin):
    require(path.stat().st_size <= 65536 and sha(path) == pin, "External bounded allocation identity")
    value = read(path)
    require(set(value) == ALLOCATION_KEYS and value["version"] == ALLOCATION_VERSION, "Exact allocation schema")
    require(value["prepared_completed_sha256"] == cost.PREPARED_PIN
            and value["prepared_plan_sha256"] == cost.PREPARED_PLAN_PIN, "Fixed prepared inputs")
    for field in ("cost_completed", "cost_audit", "protocol", "failed_attempt"):
        ref = value[field]
        require(set(ref) == {"path", "sha256"} and type(ref["path"]) is str
                and len(ref["sha256"]) == 64 and all(c in "0123456789abcdef" for c in ref["sha256"]), "Allocation input reference")
    for field in ("limits", "freeze_limits"):
        limits = value[field]
        keys = {"wall_seconds", "rss_bytes", "output_bytes"} | ({"mps_driver_bytes"} if field == "limits" else set())
        require(set(limits) == keys, "Exact phase-specific resource limits")
        require(type(limits["wall_seconds"]) in (int, float) and math.isfinite(limits["wall_seconds"])
                and limits["wall_seconds"] > 0, "Positive finite external wall allocation")
        require(all(type(limits[k]) is int and limits[k] > 0 for k in keys - {"wall_seconds"}), "Positive byte allocations")
    require(Path(value["protocol"]["path"]).resolve() == (ROOT / PROTOCOL).resolve(), "Scientific protocol path")
    require(Path(value["failed_attempt"]["path"]).resolve() == (ROOT / FAILED_MANIFEST).resolve()
            and value["failed_attempt"]["sha256"] == FAILED_MANIFEST_PIN, "Fixed failed prior attempt")
    failed_attempt(value)
    return value


def failed_attempt(spec):
    ref = spec["failed_attempt"]
    require(sha(ref["path"]) == ref["sha256"] == FAILED_MANIFEST_PIN, "Failed-attempt manifest identity")
    value = read(ref["path"])
    require(value["status"] == "failed_technical_timing" and value["resume_permitted"] is False
            and value["partial_scoring_permitted"] is False and value["quality_metrics_opened"] is False
            and value["individual_predictions_decoded"] is False and value["weights_loaded"] is False,
            "Prior attempt stays failed, unscored and unresumed")
    return value


class Budget:
    def __init__(self, out, deadline, limits):
        self.out, self.deadline, self.limits = out, deadline, limits
        self.max_driver = self.max_current = 0

    def check(self):
        self.deadline.check()
        require(cost.qualified.rss() <= self.limits["rss_bytes"], "Process RSS cap")

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= self.limits["output_bytes"], "Output cap")

    def synchronize(self, torch):
        torch.mps.synchronize()
        self.max_driver = max(self.max_driver, torch.mps.driver_allocated_memory())
        self.max_current = max(self.max_current, torch.mps.current_allocated_memory())
        require(self.max_driver <= self.limits["mps_driver_bytes"], "Sampled MPS driver cap")
        self.check()


def authenticate_prerequisites(prepared, spec):
    failed_attempt(spec)
    parent = cost.authenticate_prepared(prepared, spec["prepared_completed_sha256"])
    for name in ("cost_completed", "cost_audit", "protocol"):
        ref = spec[name]
        require(sha(ref["path"]) == ref["sha256"], "Allocation-bound " + name)
    done_path = Path(spec["cost_completed"]["path"])
    done = read(done_path)
    require(done["status"] == "completed" and done["version"] == cost.VERSION and done["phase"] == "model-pilot"
            and done["prepared_completed_sha256"] == cost.PREPARED_PIN, "Successful matching cost pilot")
    require({k: done["counts"][k] for k in cost.EXPECTED} == cost.EXPECTED, "Complete fixed cost workload")
    check_manifest(done_path.parent, done["files"])
    request = read(done_path.parent / "started.json")["request"]
    cost_path = Path(request["preparation"])
    plan, _ = cost.authenticate_plan(cost_path, request["preparation_sha256"])
    require(sha(cost_path / "plan.json") == done["cost_plan_sha256"], "Cost plan binding")
    audit = read(spec["cost_audit"]["path"])
    require(audit["status"] == "completed" and audit["agreement"] is True
            and audit["execution_completed_sha256"] == spec["cost_completed"]["sha256"], "Independent cost audit join")
    return parent, plan


def expected_counts(parent, workload):
    train, dev = parent["cohort_sizes"]["train"], parent["cohort_sizes"]["dev"]
    epochs, fits = parent["config"]["epochs"], len(FIT_ORDER)
    per_fit = {"optimizer_updates": epochs * math.ceil(train / parent["config"]["effective_batch"]),
               "training_dialogue_visits": epochs * train, "backward_calls": epochs * train,
               "evaluation_forwards": dev, "training_endpoints": epochs * sum(parent["loss_counts"]["train"].values()),
               "evaluation_endpoints": sum(parent["loss_counts"]["dev"].values()),
               "memory_forwards": epochs * train + dev, "encoding_passes": epochs * train + dev}
    per_fit["encoder_calls"] = epochs * workload["splits"]["train"]["totals"]["encoder_calls"] + workload["splits"]["dev"]["totals"]["encoder_calls"]
    totals = workload["splits"]
    encoder_work = {key: epochs * totals["train"]["totals"][key] + totals["dev"]["totals"][key]
                    for key in cost.ENCODER_KEYS}
    question_steps = epochs * totals["train"]["totals"]["real_question_updates"] + totals["dev"]["totals"]["real_question_updates"]
    public_turns = epochs * totals["train"]["totals"]["public_user_turns"] + totals["dev"]["totals"]["public_user_turns"]
    state_counts = {key: question_steps for key in ("incoming_checks", "feature_checks", "result_checks", "mass_checks",
                                                   "executed_valid_question_slots", "real_question_updates")}
    state_counts.update({key: public_turns for key in ("advance_calls", "advance_returned", "valid_turns")})
    state_counts.update({key: per_fit["memory_forwards"] for key in ("forward_calls", "forward_returned")})
    return {"per_fit": per_fit, "all_fits": {key: value * fits for key, value in per_fit.items()}, "fits": fits,
            "encoder_work_per_fit": encoder_work, "state_counts_per_fit": state_counts}


def canonical_evaluation(prepared, packet_queries):
    """Original DEV scored order and exact public candidate values; no inferred type flag."""
    rows, originals, numbers, dialogues = [], [], [], set()
    with (prepared / "targets-dev.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            did = record["dialogue_id"]
            require(record["split"] == "dev" and did not in dialogues, "Canonical DEV dialogue membership")
            dialogues.add(did)
            endpoints = set()
            for source_index, row in enumerate(record["rows"]):
                require(row["split"] == "dev" and row["dialogue_id"] == did
                        and row["source_row_index"] == source_index, "Prepared evaluator source ordering")
                endpoint = (row["time"], row["query_position"])
                require(endpoint not in endpoints, "Duplicate canonical endpoint")
                endpoints.add(endpoint)
                query = packet_queries[row["query_index"]]
                require(query["split"] == "dev" and query["id"] == row["query_id"]
                        and query["service"] == row["service"] and query["slot"] == row["slot"], "Canonical public schema join")
                ids, values = query["candidate_ids"], query["candidate_values"]
                require(3 <= len(ids) <= 12 and len(ids) == len(values)
                        and ids[row["label_index"]] == row["label_id"], "Canonical candidate/label support")
                rows.append({**row, "row_index": len(rows), "candidate_ids": list(ids), "candidate_values": list(values)})
                for name, dest in (("original", originals), ("numbers", numbers)):
                    choice = record["literal_registers"][name][row["query_position"]][row["time"]]
                    require(type(choice) is int and 0 <= choice < len(ids), "Public literal reference support")
                    dest.append(choice)
    return rows, {"row_indices": list(range(len(rows))), "original": originals, "numbers": numbers,
                  "scope": "Deterministic public literal registers, accuracy references only"}


def source_map(cost_plan):
    result = dict(cost_plan["source_sha256"])
    require(sha(ROOT / V1_PLAN) == V1_PLAN_PIN, "Historical V1 plan identity")
    historical = read(ROOT / V1_PLAN)["source_sha256"]
    require(len(historical) == 53 and set(historical) == set(result) | set(OLD_NEW_SOURCES)
            and all(historical[k] == v for k, v in result.items()), "Original 53-source closure preserved")
    result = dict(historical)
    for name in NEW_SOURCES:
        require(name not in result, "New source does not overwrite inherited closure")
        result[name] = sha(ROOT / name)
    require(len(result) == 64, "Exact scientific 64-source closure")
    bind_sources(result)
    return result


def bind_sources(sources):
    for name, digest in sources.items():
        require(sha(ROOT / name) == digest, "Scientific source changed: " + name)


def write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def freeze(args, spec, budget):
    parent, cost_plan = authenticate_prerequisites(args.prepared, spec)
    sources = source_map(cost_plan)
    packet = read(ROOT / "runs/sgd-state-v1/features-02/packet.json")
    rows, references = canonical_evaluation(args.prepared, packet["queries"])
    require(len(rows) == DEV_ENDPOINTS, "Every final DEV endpoint")
    workload = read(args.prepared / "workloads.json")
    for name, digest in sources.items():
        target = args.out / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        require(sha(target) == digest, "Frozen source snapshot")
    shutil.copyfile(args.allocation, args.out / "allocation.json")
    write_rows(args.out / "evaluation-rows.jsonl", rows)
    write(args.out / "references.json", references)
    plan = {"version": VERSION, "runtime": cost.qualified.runtime(), "config": parent["config"],
            "previous_plan_sha256": V1_PLAN_PIN,
            "prepared": str(args.prepared.resolve()), "prepared_completed_sha256": cost.PREPARED_PIN,
            "prepared_plan_sha256": cost.PREPARED_PLAN_PIN, "allocation_sha256": args.allocation_sha256,
            "allocation": spec, "source_sha256": sources, "fit_order": FIT_ORDER,
            "loss_counts": parent["loss_counts"], "loss_weights": parent["loss_weights"],
            "orders_sha256": sha(args.prepared / "orders.json"), "expected": expected_counts(parent, workload),
            "evaluation_rows_sha256": sha(args.out / "evaluation-rows.jsonl"),
            "references_sha256": sha(args.out / "references.json"),
            "quality_scoring_in_runner": False, "no_retry": True,
            "checkpoint_scope": "Final weights only; no optimizer state, resume or epoch selection"}
    write(args.out / "plan.json", plan)
    authenticate_prerequisites(args.prepared, spec)
    bind_sources(sources)
    budget.storage()
    return {"plan_sha256": sha(args.out / "plan.json"), "source_sha256": sources,
            "expected": plan["expected"], "model_calls": 0, "encoder_calls": 0}


def validate_plan(path, pin, spec):
    require(sha(path) == pin, "External scientific plan pin")
    done = read(path.parent / "completed.json")
    require(done["status"] == "completed" and done["phase"] == "freeze" and done["plan_sha256"] == pin,
            "Successful scientific freeze")
    check_manifest(path.parent, done["files"])
    plan = read(path)
    require(plan["version"] == VERSION and plan["allocation"] == spec
            and plan["previous_plan_sha256"] == V1_PLAN_PIN
            and plan["runtime"] == cost.qualified.runtime() and plan["fit_order"] == FIT_ORDER,
            "Frozen scientific version/allocation/runtime/fits")
    require(plan["prepared_completed_sha256"] == cost.PREPARED_PIN
            and plan["prepared_plan_sha256"] == cost.PREPARED_PLAN_PIN
            and plan["quality_scoring_in_runner"] is False and plan["no_retry"] is True,
            "Fixed prepared lineage and scientific execution scope")
    require(sha(path.parent / "allocation.json") == plan["allocation_sha256"], "Frozen allocation bytes")
    parent, cost_plan = authenticate_prerequisites(Path(plan["prepared"]), spec)
    require(plan["source_sha256"] == source_map(cost_plan) and plan["config"] == parent["config"]
            and plan["loss_weights"] == parent["loss_weights"] and plan["loss_counts"] == parent["loss_counts"], "Frozen sources/recipe/objective")
    for name, digest in plan["source_sha256"].items():
        require(sha(path.parent / "sources" / name) == digest, "Frozen source copy")
    prepared = Path(plan["prepared"])
    require(plan["orders_sha256"] == sha(prepared / "orders.json")
            and plan["expected"] == expected_counts(parent, read(prepared / "workloads.json")), "Frozen work/order identity")
    require(sha(path.parent / "evaluation-rows.jsonl") == plan["evaluation_rows_sha256"]
            and sha(path.parent / "references.json") == plan["references_sha256"], "Canonical evaluation artifacts")
    return plan, parent


def load_training_inputs(prepared):
    """Separate actor maps and evaluator records; immutable lexical array views."""
    import numpy as np

    actors, targets = {}, {}
    for split in ("train", "dev"):
        for prefix, destination in (("actors", actors), ("targets", targets)):
            with (prepared / f"{prefix}-{split}.jsonl").open() as stream:
                for line in stream:
                    value = json.loads(line)
                    key = (split, value["dialogue_id"])
                    require(value["split"] == split and key not in destination, "Split-qualified input membership")
                    destination[key] = value if prefix == "actors" else value["rows"]
    require(set(actors) == set(targets), "Actor/evaluator cohort identity")
    lexical = {name: np.load(prepared / f"lexical-{name}.npy", mmap_mode="r", allow_pickle=False)
               for name in ("original", "numbers")}
    for array in lexical.values():
        require(array.dtype == np.float32 and array.ndim == 1, "Prepared lexical cache dtype/geometry")
    return actors, targets, lexical


def payload_actor(record, lexical, arm):
    """Public-only numeric assembly; target records are not accepted."""
    import numpy as np

    shape, offset = record["lexical_shape"], record["lexical_offset"]
    observation = np.array(lexical[arm.split("_", 1)[1]][offset:offset + math.prod(shape)], dtype=np.float32).reshape(shape)
    return {**record, "lexical": observation}


class TorchRoute:
    """The qualified encoder/monitor/loss path, one fresh model pair per fit."""
    def __init__(self, parent, seed, arm, budget):
        import torch
        from study_dialogue_copy_v2 import MonitoredCopyMemoryV2, empty_invariants, merge_invariants
        from transformers import AutoModel

        from openjev.research.dialogue_finetune_training import supervised_loss
        from openjev.research.dialogue_trainable_encoder import build_actor, encode_token_lists

        self.torch, self.budget, self.parent = torch, budget, parent
        self.artifact_dir, self.progress = budget.out, {}
        self.empty, self.merge = empty_invariants, merge_invariants
        self.encode_tokens, self.build_actor, self.loss = encode_token_lists, build_actor, supervised_loss
        self.trainable = arm.startswith("trainable_")
        torch.manual_seed(seed)
        self.encoder = AutoModel.from_pretrained(parent["snapshot"], local_files_only=True,
            use_safetensors=True, attn_implementation="eager", dtype=torch.float32).to("mps").eval()
        torch.manual_seed(seed)
        self.memory = MonitoredCopyMemoryV2("scalar").cpu()
        self.encoder.requires_grad_(self.trainable)
        self.initial = {"encoder": cost.qualified.tensor_digest(self.encoder), "memory": cost.qualified.tensor_digest(self.memory)}
        groups = [{"params": list(self.memory.parameters()), "lr": parent["config"]["memory_lr"]}]
        if self.trainable:
            groups.append({"params": list(self.encoder.parameters()), "lr": parent["config"]["encoder_lr"]})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=parent["config"]["weight_decay"])
        self.parameters = [p for group in groups for p in group["params"]]
        self.counts, self.work, self.audit = Counter(), Counter(), self.empty()
        self.encoder.register_forward_pre_hook(lambda *_: self.counts.update({"encoder_forward_attempts": 1}))
        self.encoder.register_forward_hook(lambda *_: self.counts.update({"encoder_forward_returns": 1}))

    def forward(self, payload, training):
        torch = self.torch
        self.counts["encoding_attempts"] += 1
        vectors, work = self.encode_tokens(self.encoder, payload["tokens"], **self.parent["tokenizer_ids"],
            trainable=self.trainable and training, chunk_tokens=254, chunk_batch_size=32)
        self.counts["encoding_passes"] += 1
        self.work.update(work)
        actor, none = self.build_actor(vectors, **{k: payload[k] for k in
            ("turn_text_ids", "query_text_ids", "candidate_text_ids", "candidate_ids", "lexical")}, memory_device="cpu")
        self.memory.begin_batch(actor, [{"layout": {"shape": payload["lexical_shape"]}}])
        self.counts["memory_forward_attempts"] += 1
        logs = None
        try:
            logs = self.memory(*actor, none_index=none)
            self.counts["memory_forwards"] += 1
            support = actor[4][:, None].expand_as(logs)
            require(torch.isfinite(logs[support]).all().item() and torch.isneginf(logs[~support]).all().item(),
                    "Finite supported log probabilities and exact padding")
        except BaseException as error:
            if logs is not None:
                try:
                    import numpy as np

                    with (self.artifact_dir / "offending-log-probs.npy").open("xb") as stream:
                        np.save(stream, logs.detach().cpu().numpy(), allow_pickle=False)
                    write(self.artifact_dir / "offending-context.json", {
                        "split": payload["split"], "dialogue_id": payload["dialogue_id"], "training": training,
                        "context": {key: self.progress.get(key) for key in ("fit_id", "operation", "epoch", "batch_start", "microbatch")}})
                except BaseException as secondary:  # noqa: BLE001 - retain original numerical failure
                    error.add_note("Offending-output preservation: " + repr(secondary))
            raise
        finally:
            self.merge(self.audit, self.memory.audit)
        return logs, work, dict(self.memory.audit)

    def step(self):
        torch = self.torch
        memory_norm = sum(float(p.grad.detach().double().square().sum()) for p in self.memory.parameters() if p.grad is not None)
        require(0 < memory_norm < float("inf"), "Finite nonzero memory gradient")
        witnesses, named = {}, dict(self.encoder.named_parameters())
        for name in ("embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"):
            grad = named[name].grad
            if self.trainable:
                require(grad is not None, "Trainable encoder gradient present")
                value = float(grad.detach().cpu().double().square().sum())
                require(0 < value < float("inf"), "Finite nonzero encoder gradient")
                witnesses[name] = value
            else:
                require(grad is None, "Frozen encoder gradient absent")
                witnesses[name] = None
        require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in self.parameters), "Finite parameter gradients")
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.parent["config"]["clip"], error_if_nonfinite=True, foreach=False)
        require(torch.isfinite(norm).item(), "Finite global clipping norm")
        self.counts["optimizer_attempts"] += 1
        self.optimizer.step()
        self.counts["optimizer_updates"] += 1
        require(all(torch.isfinite(p).all().item() for p in self.parameters), "Finite updated parameters")
        require(self.trainable or all(p.grad is None for p in self.encoder.parameters()), "All frozen gradients absent")
        return {"memory_gradient_squared_norm": memory_norm, "encoder_gradient_squared_norms": witnesses,
                "gradient_norm_before_clip": float(norm)}

    def checkpoint(self, path):
        start = time.perf_counter()
        self.budget.synchronize(self.torch)
        state = {"memory": {k: v.detach().cpu() for k, v in self.memory.state_dict().items()}}
        if self.trainable:
            state["encoder"] = {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
        with path.open("xb") as stream:
            self.torch.save(state, stream)
        result = {"sha256": sha(path), "bytes": path.stat().st_size, "wall_seconds": time.perf_counter() - start}
        self.budget.storage()
        return result


def update(route, ids, actors, targets, lexical, arm, weights, progress):
    """Actual effective-batch denominator; backward each complete dialogue once."""
    start = time.perf_counter()
    endpoint_count = sum(len(targets[("train", did)]) for did in ids)
    require(endpoint_count > 0, "Nonempty effective batch loss support")
    route.optimizer.zero_grad(set_to_none=True)
    work, audit, value = Counter(), route.empty(), 0.
    for micro, did in enumerate(ids):
        progress["microbatch"] = micro
        key = ("train", did)
        payload = payload_actor(actors[key], lexical, arm)
        logs, observed, invariant = route.forward(payload, True)
        loss = route.loss(logs, targets[key], weights, endpoint_count)
        route.counts["backward_attempts"] += 1
        loss.backward()
        route.counts["backward_calls"] += 1
        route.counts["training_dialogue_visits"] += 1
        route.counts["training_endpoints"] += len(targets[key])
        value += float(loss.detach())
        work.update(observed)
        route.merge(audit, invariant)
        del logs, loss, payload
        route.budget.synchronize(route.torch)
    gradient = route.step()
    route.budget.synchronize(route.torch)
    return {"dialogue_ids": ids, "microbatches": len(ids), "endpoint_count": endpoint_count,
            "weighted_training_loss": value, "encoder_work": dict(work), "invariants": audit,
            "wall_seconds": time.perf_counter() - start, **gradient}


def evaluate(route, actors, targets, lexical, arm, rows, out, progress):
    import numpy as np

    started = time.perf_counter()
    # On-disk raw arrays preserve the produced prefix if a later endpoint fails.
    logs = np.lib.format.open_memmap(out / "partial-log-probs.npy", mode="w+", dtype=np.float32, shape=(len(rows), 12))
    logs[:] = -np.inf
    row_ids = np.lib.format.open_memmap(out / "partial-row-indices.npy", mode="w+", dtype=np.int64, shape=(len(rows),))
    row_ids[:] = -1
    cursor, work, audit = 0, Counter(), route.empty()
    with route.torch.no_grad():
        for key, actor_record in actors.items():
            if key[0] != "dev":
                continue
            progress.update({"operation": "final-dev", "evaluation_dialogue": key[1], "evaluation_rows": cursor})
            payload = payload_actor(actor_record, lexical, arm)
            output, observed, invariant = route.forward(payload, False)
            for endpoint in targets[key]:
                canonical = rows[cursor]
                require(canonical["row_index"] == cursor and canonical["dialogue_id"] == key[1]
                        and all(canonical[k] == endpoint[k] for k in ("source_row_index", "time", "query_index", "label_index", "bin")),
                        "Exact final DEV row membership/order")
                t, q = endpoint["time"], endpoint["query_position"]
                count = len(payload["candidate_ids"][q])
                require(payload["candidate_ids"][q] == canonical["candidate_ids"], "Final candidate identity/order")
                vector = output[0, t, q, :count].detach().cpu().numpy()
                require(vector.dtype == np.float32 and np.isfinite(vector).all(), "Finite raw float32 saved logs")
                require(abs(float(np.exp(vector.astype(np.float64)).sum()) - 1.) <= 2e-6, "Saved supported probability mass")
                logs[cursor, :count] = vector
                row_ids[cursor] = cursor
                cursor += 1
            route.counts["evaluation_forwards"] += 1
            route.counts["evaluation_endpoints"] += len(targets[key])
            work.update(observed)
            route.merge(audit, invariant)
            logs.flush()
            row_ids.flush()
            del output, payload
            route.budget.synchronize(route.torch)
    require(cursor == len(rows), "Every final DEV endpoint saved")
    with (out / "predictions.npz").open("xb") as stream:
        np.savez(stream, row_indices=row_ids, log_probs=logs)
    del logs, row_ids
    (out / "partial-log-probs.npy").unlink()
    (out / "partial-row-indices.npy").unlink()
    route.budget.storage()
    return {"rows": cursor, "wall_seconds": time.perf_counter() - started,
            "encoder_work": dict(work), "invariants": audit,
            "scope": "One final DEV pass, raw logs only; no quality metrics or checkpoint selection"}


def train(args, spec, budget, progress):
    plan, parent = validate_plan(args.plan, args.plan_sha256, spec)
    import gc

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(torch.backends.mps.is_available(), "Qualified MPS path required")
    prepared = Path(plan["prepared"])
    actors, targets, lexical = load_training_inputs(prepared)
    orders = read(prepared / "orders.json")
    require(orders["seeds"] == SEEDS and orders["epochs"] == 20, "Paired prepared seeds/epochs")
    train_ids = orders["dialogue_ids"]
    require(set(train_ids) == {did for split, did in actors if split == "train"}, "Complete training order membership")
    for seed in SEEDS:
        require(len(orders["orders"][str(seed)]) == 20
                and all(sorted(epoch) == list(range(len(train_ids))) for epoch in orders["orders"][str(seed)]), "Exact epoch permutations")
    for name in ("allocation.json", "evaluation-rows.jsonl", "references.json", "plan.json"):
        shutil.copyfile(args.plan.parent / name, args.out / name)
    rows = [json.loads(line) for line in (args.out / "evaluation-rows.jsonl").read_text().splitlines()]
    all_counts, fits, initial_by_seed = Counter(), [], {}
    encoder_initial = None
    progress["counts"] = all_counts
    for seed in SEEDS:
        for arm in ARMS:
            fit_id = f"{arm}-{seed}"
            directory = args.out / fit_id
            directory.mkdir()
            for key in ("active_counts", "active_encoder_work", "active_invariants", "active_initial_sha256",
                        "epoch", "batch_start", "microbatch", "evaluation_dialogue", "evaluation_rows"):
                progress.pop(key, None)
            progress.update({"fit_id": fit_id, "completed_fits": len(fits), "operation": "fresh-model-load"})
            fit_start = time.perf_counter()
            route = TorchRoute(parent, seed, arm, budget)
            route.artifact_dir, route.progress = directory, progress
            progress["active_counts"] = route.counts
            progress["active_encoder_work"] = route.work
            progress["active_invariants"] = route.audit
            progress["active_initial_sha256"] = route.initial
            if encoder_initial is None:
                encoder_initial = route.initial["encoder"]
            require(route.initial["encoder"] == encoder_initial, "Same pinned pretrained encoder in every fit")
            if seed not in initial_by_seed:
                initial_by_seed[seed] = route.initial
            require(route.initial == initial_by_seed[seed], "Four-arm matched fresh initialization")
            training_start = time.perf_counter()
            with (directory / "updates.jsonl").open("x") as journal:
                for epoch, order in enumerate(orders["orders"][str(seed)]):
                    for start in range(0, len(order), 32):
                        ids = [train_ids[i] for i in order[start:start + 32]]
                        progress.update({"epoch": epoch, "batch_start": start, "operation": "training"})
                        event = update(route, ids, actors, targets, lexical, arm, parent["loss_weights"], progress)
                        journal.write(json.dumps({"epoch": epoch, "batch_start": start, **event}, allow_nan=False) + "\n")
                        journal.flush()
                        budget.storage()
            training_seconds = time.perf_counter() - training_start
            evaluation = evaluate(route, actors, targets, lexical, arm, rows, directory, progress)
            final_encoder = cost.qualified.tensor_digest(route.encoder)
            require((final_encoder != route.initial["encoder"]) == route.trainable, "Final frozen/trainable encoder digest")
            checkpoint = route.checkpoint(directory / "weights.pt")
            route.counts["encoder_calls"] = route.work["encoder_calls"]
            expected = plan["expected"]["per_fit"]
            require({k: route.counts[k] for k in expected} == expected, "Complete per-fit operation accounting")
            require(dict(route.work) == plan["expected"]["encoder_work_per_fit"], "Every encoder token/chunk counted")
            require({key: route.audit[key] for key in plan["expected"]["state_counts_per_fit"]}
                    == plan["expected"]["state_counts_per_fit"], "Every public recurrent step monitored")
            for a, b in (("encoding_attempts", "encoding_passes"), ("memory_forward_attempts", "memory_forwards"),
                         ("backward_attempts", "backward_calls"), ("optimizer_attempts", "optimizer_updates"),
                         ("encoder_forward_attempts", "encoder_forward_returns")):
                require(route.counts[a] == route.counts[b], "Per-fit attempt/return coverage")
            require(route.counts["encoder_forward_returns"] == route.work["encoder_calls"], "Actual encoder dispatch accounting")
            fit = {"version": VERSION, "status": "completed", "fit_id": fit_id, "arm": arm, "seed": seed,
                   "initial_sha256": route.initial, "final_encoder_sha256": final_encoder,
                   "counts": dict(route.counts), "encoder_work": dict(route.work), "invariants": route.audit,
                   "training_seconds": training_seconds, "evaluation": evaluation, "checkpoint": checkpoint,
                   "wall_seconds": time.perf_counter() - fit_start, "files": members(directory)}
            write(directory / "completed.json", fit)
            all_counts.update({k: route.counts[k] for k in expected})
            fits.append({"fit_id": fit_id, "completed_sha256": sha(directory / "completed.json")})
            del route
            gc.collect()
            torch.mps.empty_cache()
            budget.synchronize(torch)
    require([fit["fit_id"] for fit in fits] == FIT_ORDER and dict(all_counts) == plan["expected"]["all_fits"], "All twelve complete fits")
    validate_plan(args.plan, args.plan_sha256, spec)
    budget.storage()
    return {"plan_sha256": args.plan_sha256, "allocation_sha256": plan["allocation_sha256"],
            "source_sha256": plan["source_sha256"], "fits": fits, "fit_count": len(fits), "counts": dict(all_counts),
            "sampled_mps_driver_max_bytes": budget.max_driver, "sampled_mps_current_max_bytes": budget.max_current,
            "quality_scoring_in_runner": False, "official_test_opened": False, "external_model_api_calls": 0}


def validate_supervision(args, launch, clock, worker_started_ns, limits):
    """Bind the parent deadline to this exact process, argv, plan and output."""
    require(launch["version"] == "dialogue-observation-supervision-v2", "V2 supervision version")
    command = launch["command"]
    require(type(command) is list and command and all(type(v) is str for v in command), "Supervisor command")
    tail = command[1:]
    if tail and tail[0] == "-u":
        tail = tail[1:]
    require(Path(command[0]).resolve() == Path(sys.executable).resolve() and tail == sys.argv,
            "Exact live worker interpreter/argv")
    require(len(tail) == 10 and Path(tail[0]).resolve() == (ROOT / "scripts/study_dialogue_observation_v2.py").resolve()
            and tail[1] == "train", "Scientific V2 worker command")
    flags = dict(zip(tail[2::2], tail[3::2], strict=True))
    require(set(flags) == {"--plan", "--plan-sha256", "--out", "--supervision"}
            and Path(flags["--plan"]).resolve() == args.plan.resolve()
            and flags["--plan-sha256"] == args.plan_sha256
            and Path(flags["--out"]).resolve() == args.out.resolve()
            and Path(flags["--supervision"]).resolve() == args.supervision.resolve(), "Live command inputs/output")
    require(Path(launch["cwd"]).resolve() == Path.cwd().resolve(), "Supervisor/worker working directory")
    require(all(type(launch[k]) is int and launch[k] > 0 for k in ("pid", "pgid", "parent_pid"))
            and launch["pid"] == os.getpid() and launch["pgid"] == os.getpgrp()
            and launch["pid"] == launch["pgid"] and launch["parent_pid"] == os.getppid(),
            "Live parent and dedicated worker process group")
    require(clock.backend in ("mach_continuous_time", "CLOCK_BOOTTIME")
            and launch["clock_backend"] == clock.backend, "Same native suspend-inclusive clock")
    require(type(launch["started_ns"]) is int and type(launch["deadline_ns"]) is int
            and 0 <= launch["started_ns"] <= worker_started_ns < launch["deadline_ns"]
            and launch["cap_seconds"] == limits["wall_seconds"]
            and launch["deadline_ns"] - launch["started_ns"] == _duration_ns(limits["wall_seconds"]),
            "Exact parent absolute deadline and allocation")
    require(launch["watchdog_sha256"] == sha(ROOT / SUPERVISOR_SOURCE)
            and launch["clock_source_sha256"] == sha(ROOT / CLOCK_SOURCE), "Supervisor/clock source identity")
    deadline = Deadline(clock, launch["started_ns"], launch["deadline_ns"])
    deadline.check()
    return deadline


def await_supervision(args, clock, worker_started_ns, limits):
    """Publication may race worker startup; never wait past five native seconds."""
    wait_limit = Deadline(clock, worker_started_ns, worker_started_ns + 5_000_000_000)
    while True:
        wait_limit.check()
        if args.supervision.exists():
            break
        time.sleep(.025)
    require(args.supervision.stat().st_size <= 65536, "Bounded supervision record")
    pin = sha(args.supervision)
    launch = read(args.supervision)
    require(sha(args.supervision) == pin, "Stable supervision publication")
    return validate_supervision(args, launch, clock, worker_started_ns, limits), pin


def timing_record(clock, started_ns, deadline=None, *, terminal=False, failure=False):
    result = {"clock_backend": None if clock is None else clock.backend, "started_ns": started_ns,
              "parent_started_ns": None if deadline is None else deadline.started_ns,
              "deadline_ns": None if deadline is None else deadline.expires_ns,
              "finished_ns": None, "elapsed_ns": None, "wall_seconds": None,
              "timing_available": clock is not None and started_ns is not None}
    if terminal and result["timing_available"]:
        try:
            end = clock.now_ns()
            require(end >= started_ns, "Terminal clock precedes worker start")
            result.update(finished_ns=end, elapsed_ns=end - started_ns, wall_seconds=(end - started_ns) / 1e9)
        except BaseException as error:  # Failed clock must not erase the original failure.
            if not failure:
                raise
            result.update(timing_available=False, timing_error=repr(error))
    return result


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    request = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    progress = {"phase": args.command, "operation": "allocation-authentication"}
    handler = clock = deadline = started_ns = None
    supervision_pin = None
    try:
        clock = SuspendClock()
        started_ns = clock.now_ns()
        if args.command == "freeze":
            spec = allocation(args.allocation, args.allocation_sha256)
        else:
            require(sha(args.plan) == args.plan_sha256, "External scientific plan pin")
            initial_plan = read(args.plan)
            spec = allocation(args.plan.parent / "allocation.json", initial_plan["allocation_sha256"])
        limits = spec["freeze_limits" if args.command == "freeze" else "limits"]
        if args.command == "train":
            progress["operation"] = "supervision-authentication"
            deadline, supervision_pin = await_supervision(args, clock, started_ns, limits)
        else:
            deadline = Deadline(clock, started_ns, started_ns + _duration_ns(limits["wall_seconds"]))
        budget = Budget(args.out, deadline, limits)
        write(args.out / "started.json", {"version": VERSION, "request": request, "runtime": cost.qualified.runtime(),
              **timing_record(clock, started_ns, deadline), "supervision_sha256": supervision_pin, "no_retry": True})
        budget.storage()

        def timeout(*_):
            # No clock access from a signal handler: SuspendClock uses a lock.
            raise TimeoutError("Supplementary awake-time emergency timeout")
        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.000001, deadline.remaining_ns() / 1e9))
        result = freeze(args, spec, budget) if args.command == "freeze" else train(args, spec, budget, progress)
        if supervision_pin is not None:
            require(sha(args.supervision) == supervision_pin, "End supervision identity")
        payloads = members(args.out)
        timing = timing_record(clock, started_ns, deadline, terminal=True)
        require(timing["timing_available"], "Successful completion requires valid timing")
        budget.check()
        result.update({"version": VERSION, "status": "completed", "phase": args.command,
                       "files": payloads, **timing, "supervision_sha256": supervision_pin,
                       "peak_rss_bytes": cost.qualified.rss(),
                       "wall_scope": "Suspend-inclusive worker elapsed through payload manifest; parent absolute deadline checked through completion write/hash",
                       "phase_timing_scope": "Fit/update/evaluation/checkpoint durations retain V1 perf_counter diagnostics and may exclude suspend; never used for admission",
                       "no_retry": True})
        write(args.out / "completed.json", result)
        pin = sha(args.out / "completed.json")
        budget.storage()
        return pin
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if not (args.out / "started.json").exists():
                write(args.out / "started.json", {"version": VERSION, "request": request,
                      **timing_record(clock, started_ns, deadline), "supervision_sha256": supervision_pin, "no_retry": True})
            if (args.out / "completed.json").exists():
                (args.out / "completed.json").rename(args.out / "invalid-completion.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request,
                  "error_type": type(error).__name__, "error": str(error), "progress": progress,
                  **timing_record(clock, started_ns, deadline, terminal=True, failure=True), "supervision_sha256": supervision_pin,
                  "peak_rss_bytes": cost.qualified.rss(), "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - retain actual failure
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--allocation", type=Path, required=True)
    p.add_argument("--allocation-sha256", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--supervision", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps({"status": "completed", "completed_sha256": execute(parser.parse_args())}))
