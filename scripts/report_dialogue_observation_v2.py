"""Saved-only report for twelve complete autonomous observation-learning fits.

Checkpoint/model assets are hashed, never deserialized. Integer token layouts
and evaluator metadata reconstruct coverage; no lexical/vector arrays, tokenizer,
model or training module is imported. Internal tensor and gradient checks remain
source-bound execution witnesses, not independently replayed computations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import signal
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from openjev.research import dialogue_observation_metrics as metrics
from openjev.research.suspend_clock import SuspendClock

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-observation-report-v2"
STUDY = "dialogue-observation-scientific-v2"
PREPARED_PIN = "d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83"
PREPARED_PLAN_PIN = "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e"
FIT_ORDER = [f"{arm}-{seed}" for seed in metrics.SEEDS for arm in metrics.ARMS]
DEV_ENDPOINTS = 62329
WATCHDOG = "scripts/supervise_dialogue_observation_v2.py"
PROTOCOL = "research/dialogue-observation-learning-protocol-v2.md"
OLD_PLAN = "output/dialogue-observation-learning-v1/scientific-freeze-01/plan.json"
OLD_PLAN_PIN = "acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237"
FAILED_PIN = "41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_TEST = "tests/test_suspend_clock.py"
BACKENDS = {"mach_continuous_time", "CLOCK_BOOTTIME"}
V2_SOURCES = {"scripts/study_dialogue_observation_v2.py", "tests/test_study_dialogue_observation_v2.py",
              "scripts/report_dialogue_observation_v2.py", "tests/test_report_dialogue_observation_v2.py",
              WATCHDOG, "tests/test_supervise_dialogue_observation_v2.py", CLOCK, CLOCK_TEST, PROTOCOL,
              "output/dialogue-observation-learning-v2/result-audit-01/audit.py",
              "output/dialogue-observation-learning-v2/result-audit-01/test_audit.py"}
NEW_SOURCES = {"scripts/study_dialogue_observation.py", "tests/test_study_dialogue_observation.py",
               "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py",
               "scripts/report_dialogue_observation.py", "tests/test_report_dialogue_observation.py",
               "research/dialogue-observation-learning-scoring.md", "research/dialogue-observation-learning-protocol.md",
               "output/dialogue-observation-learning-v1/study-watchdog-01.py"} | V2_SOURCES
COST_SOURCES = {"scripts/qualify_dialogue_observation_batches.py", "tests/test_qualify_dialogue_observation_batches.py",
                "src/openjev/research/dialogue_finetune_training.py", "tests/test_dialogue_finetune_training.py",
                "research/dialogue-observation-learning-cost-protocol.md"}
PREPARED_FILES = {"started.json", "source-pins.json", "plan.json", "actors-train.jsonl", "actors-dev.jsonl",
                  "targets-train.jsonl", "targets-dev.jsonl", "index.json", "lexical-original.npy",
                  "lexical-numbers.npy", "workloads.json", "orders.json", "effective-batches.json"}
ENCODER_KEYS = ("input_texts", "content_tokens", "encoder_sequences", "encoder_calls", "special_token_positions",
                "valid_token_positions", "padded_token_positions", "padded_attention_positions",
                "padding_token_positions", "overlength_texts_chunked", "truncated_tokens")
COUNT_KEYS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
              "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks",
              "result_checks", "mass_checks", "mass_above_one_count")
MAX_KEYS = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")
GRADIENT_KEYS = {"embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"}
CONFIG = {"arms": list(metrics.ARMS), "seeds": list(metrics.SEEDS), "epochs": 20, "effective_batch": 32,
          "microbatch": 1, "encoder_batch": 32, "chunk_tokens": 254, "memory_lr": .001,
          "encoder_lr": .00002, "weight_decay": .0001, "clip": 1., "memory_method": "scalar",
          "dropout": False, "cpu_threads": 1,
          "loss": "sum(weight[stratum] * endpoint NLL) / eligible endpoints in effective batch",
          "selection": "All final fits; no epoch, seed or variant selection"}
CAPS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 256 * 1024**2}
COST_COUNTS = {"cells": 40, "optimizer_updates": 52, "training_dialogue_visits": 1168,
               "evaluation_forwards": 24, "parity_dialogue_visits": 412, "backward_calls": 1168,
               "memory_forwards": 1192, "encoding_passes": 1604, "checkpoints": 4}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "tokenizers")}}


def descriptor(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def manifest(directory, expected, *, terminal="completed.json"):
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "No symbolic links in artifact tree")
        if path.is_file() and path != directory / terminal:
            actual.add(path.relative_to(directory).as_posix())
    require(actual == set(expected), "Exact artifact membership")
    for name, item in expected.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Safe artifact path")
        require(set(item) == {"sha256", "bytes"} and digest(item["sha256"]) and integer(item["bytes"]),
                "File descriptor schema")
        require(descriptor(directory / name) == item, "Artifact changed: " + name)


def bind(mapping, root=ROOT):
    for name, pin in mapping.items():
        require(digest(pin) and sha(root / name) == pin, "Bound source/input changed: " + name)


def terminal(directory, pin, phase):
    require(sha(directory / "completed.json") == pin, "External terminal pin")
    done = read(directory / "completed.json")
    require(done["status"] == "completed" and done["phase"] == phase, "Complete required phase")
    manifest(directory, done["files"])
    return done


def authenticate_failure(reference, sources):
    """Old failure remains opaque provenance, never a source of fitted tensors."""
    require(set(reference) == {"path", "sha256"} and reference["sha256"] == FAILED_PIN
            and sha(reference["path"]) == FAILED_PIN, "Externally pinned failed attempt")
    failed = read(reference["path"])
    require(failed["status"] == "failed_technical_timing" and failed["required_fits"] == 12
            and failed["quality_metrics_opened"] is False and failed["individual_predictions_decoded"] is False
            and failed["weights_loaded"] is False and failed["resume_permitted"] is False
            and failed["partial_scoring_permitted"] is False and failed["process_group_absent"] is True,
            "Preserved unscored, unresumed failed attempt")
    require(sha(ROOT / OLD_PLAN) == OLD_PLAN_PIN, "Externally pinned original source closure")
    old = read(ROOT / OLD_PLAN)["source_sha256"]
    require(len(old) == 53 and len(sources) == 64 and set(sources) == set(old) | V2_SOURCES
            and all(sources[name] == pin for name, pin in old.items()), "All original 53 sources unchanged")


def elapsed_record(value):
    require(value["timing_available"] is True and value["clock_backend"] in BACKENDS
            and all(integer(value[k]) for k in ("started_ns", "finished_ns", "deadline_ns", "elapsed_ns")),
            "Native integer suspend-inclusive timing")
    require(value["started_ns"] <= value["finished_ns"] < value["deadline_ns"]
            and value["elapsed_ns"] == value["finished_ns"] - value["started_ns"]
            and finite(value["wall_seconds"]) and value["wall_seconds"] == value["elapsed_ns"] / 1_000_000_000,
            "Consistent successful suspend-inclusive interval")


def clock_envelope(launch, end, started, done, cap):
    """Civil time may jump in either direction and never determines admission."""
    require(type(cap) is int and cap > 0, "Integer allocated seconds")
    require(launch["clock_backend"] in BACKENDS
            and all(integer(launch[k]) for k in ("started_ns", "deadline_ns"))
            and launch["deadline_ns"] == launch["started_ns"] + cap * 1_000_000_000,
            "Shared absolute deadline")
    elapsed_record(end)
    elapsed_record(done)
    require(end["timing_available"] is True and started["timing_available"] is True
            and end["status"] == "completed", "Successful parent timing")
    require(all(v["clock_backend"] == launch["clock_backend"] and v["deadline_ns"] == launch["deadline_ns"]
                for v in (end, started, done)) and end["started_ns"] == launch["started_ns"]
            and integer(started["started_ns"]) and done["started_ns"] == started["started_ns"],
            "Same parent/worker clock and deadline")
    require(started["parent_started_ns"] == done["parent_started_ns"] == launch["started_ns"],
            "Common inherited parent origin")
    require(launch["started_ns"] <= started["started_ns"] <= done["finished_ns"] <= end["finished_ns"]
            < launch["deadline_ns"], "Worker enclosed within successful parent interval")
    require(finite(launch["started_unix"], positive=True) and finite(end["finished_unix"], positive=True),
            "Finite civil provenance only")


def freeze_timing(started, done, cap):
    elapsed_record(done)
    require(type(cap) is int and cap > 0 and started["timing_available"] is True
            and started["clock_backend"] == done["clock_backend"]
            and started["started_ns"] == done["started_ns"] == started["parent_started_ns"] == done["parent_started_ns"]
            and started["deadline_ns"] == done["deadline_ns"] == done["started_ns"] + cap * 1_000_000_000
            and started["supervision_sha256"] is None and done["supervision_sha256"] is None,
            "Standalone freeze deadline and origin")


def authenticate_lineage(run, done, freeze_path, freeze_pin):
    """Authenticate all opaque payloads before decoding evaluator/token metadata."""
    require(sha(freeze_path) == freeze_pin == done["plan_sha256"] == sha(run / "plan.json"), "External frozen plan")
    plan = read(freeze_path)
    frozen = read(freeze_path.parent / "completed.json")
    require(frozen["status"] == "completed" and frozen["version"] == STUDY and frozen["phase"] == "freeze"
            and frozen["plan_sha256"] == freeze_pin and frozen["model_calls"] == frozen["encoder_calls"] == 0,
            "Successful metadata-only scientific freeze")
    manifest(freeze_path.parent, frozen["files"])
    expected_freeze = {"started.json", "allocation.json", "evaluation-rows.jsonl", "references.json", "plan.json"}
    require(set(frozen["files"]) == expected_freeze | {"sources/" + name for name in plan["source_sha256"]}, "Freeze closure")
    require(plan["version"] == STUDY and plan["previous_plan_sha256"] == OLD_PLAN_PIN
            and plan["config"] == CONFIG and plan["fit_order"] == FIT_ORDER
            and plan["quality_scoring_in_runner"] is False and plan["no_retry"] is True, "Fixed recipe/order")
    require(plan["source_sha256"] == done["source_sha256"] == frozen["source_sha256"]
            and len(plan["source_sha256"]) == 64, "Complete source identity")
    bind(plan["source_sha256"])
    bind(plan["source_sha256"], freeze_path.parent / "sources")
    for name in ("allocation.json", "evaluation-rows.jsonl", "references.json"):
        require(descriptor(run / name) == descriptor(freeze_path.parent / name), "Exact frozen evaluator/allocation copy")
    spec = read(run / "allocation.json")
    require(spec == plan["allocation"] and sha(run / "allocation.json") == plan["allocation_sha256"] == done["allocation_sha256"],
            "Allocation binding")
    require(set(spec) == {"version", "prepared_completed_sha256", "prepared_plan_sha256", "cost_completed", "cost_audit",
                          "protocol", "limits", "freeze_limits", "failed_attempt"}
            and spec["version"] == "dialogue-observation-allocation-v2", "External allocation schema")
    authenticate_failure(spec["failed_attempt"], plan["source_sha256"])
    freeze_timing(read(freeze_path.parent / "started.json"), frozen, spec["freeze_limits"]["wall_seconds"])
    for field in ("limits", "freeze_limits"):
        limits = spec[field]
        require(set(limits) == {"wall_seconds", "rss_bytes", "output_bytes"} | ({"mps_driver_bytes"} if field == "limits" else set()),
                "Allocated resource dimensions")
        require(finite(limits["wall_seconds"], positive=True)
                and all(integer(v) and v > 0 for k, v in limits.items() if k != "wall_seconds"), "Positive allocation")
    require(spec["prepared_completed_sha256"] == plan["prepared_completed_sha256"] == PREPARED_PIN
            and spec["prepared_plan_sha256"] == plan["prepared_plan_sha256"] == PREPARED_PLAN_PIN, "Fixed preparation")
    for name in ("cost_completed", "cost_audit", "protocol"):
        ref = spec[name]
        require(set(ref) == {"path", "sha256"} and sha(ref["path"]) == ref["sha256"], "Allocation parent pin: " + name)
    require(Path(spec["protocol"]["path"]).resolve() == ROOT / PROTOCOL, "Scientific protocol path")
    cost_path = Path(spec["cost_completed"]["path"])
    cost_done = terminal(cost_path.parent, spec["cost_completed"]["sha256"], "model-pilot")
    require(cost_done["version"] == "dialogue-observation-batches-v1" and cost_done["prepared_completed_sha256"] == PREPARED_PIN
            and {k: cost_done["counts"][k] for k in COST_COUNTS} == COST_COUNTS, "Complete qualified cost workload")
    cost_audit = read(spec["cost_audit"]["path"])
    require(cost_audit["status"] == "completed" and cost_audit["agreement"] is True
            and cost_audit["execution_completed_sha256"] == spec["cost_completed"]["sha256"], "Cost audit identity")
    cost_request = read(cost_path.parent / "started.json")["request"]
    cost_freeze = Path(cost_request["preparation"])
    cost_frozen = terminal(cost_freeze, cost_request["preparation_sha256"], "prepare-cost-plan")
    require(sha(cost_freeze / "plan.json") == cost_frozen["plan_sha256"] == cost_done["cost_plan_sha256"], "Cost freeze join")
    cost_plan = read(cost_freeze / "plan.json")
    prepared = Path(plan["prepared"])
    parent_done = terminal(prepared, PREPARED_PIN, "prepare")
    require(set(parent_done["files"]) == PREPARED_FILES and sha(prepared / "plan.json") == PREPARED_PLAN_PIN
            and parent_done["plan_sha256"] == PREPARED_PLAN_PIN and parent_done["encoder_calls"] == 0
            and parent_done["model_weights_loaded"] is False and parent_done["official_test_opened"] is False,
            "Prepared closure/model-free witness")
    parent = read(prepared / "plan.json")
    require(parent["preparation_version"] == 2 and parent["study"] == "dialogue-observation-learning-v1"
            and parent["config"] == CONFIG and parent["cohort_sizes"] == {"train": 2017, "dev": 2363}, "Prepared cohort/recipe")
    require(parent["data_files"] == {k: v for k, v in parent_done["files"].items() if k != "plan.json"}
            and read(prepared / "source-pins.json") == parent["source_sha256"] and len(parent["source_sha256"]) == 39,
            "Prepared source/data closure")
    require(cost_plan["source_sha256"] == {**parent["source_sha256"], **{name: sha(ROOT / name) for name in COST_SOURCES}}
            and plan["source_sha256"] == {**cost_plan["source_sha256"], **{name: sha(ROOT / name) for name in NEW_SOURCES}},
            "Exact transitive source union")
    bind(parent["input_sha256"])
    for item in parent["model_files"].values():
        require(sha(item["path"]) == item["sha256"], "Opaque pinned model asset")
    started = read(run / "started.json")
    require(started["version"] == STUDY and started["request"]["command"] == "train"
            and started["request"]["plan_sha256"] == freeze_pin
            and Path(started["request"]["plan"]).resolve() == freeze_path.resolve(), "Execution request/freeze identity")
    require(started["runtime"] == plan["runtime"] == cost_plan["runtime"] == parent["runtime"] == runtime(), "Runtime identity")
    require(plan["loss_counts"] == parent["loss_counts"] and plan["loss_weights"] == parent["loss_weights"], "Prepared objective")
    require(sha(prepared / "orders.json") == plan["orders_sha256"]
            and sha(run / "evaluation-rows.jsonl") == plan["evaluation_rows_sha256"]
            and sha(run / "references.json") == plan["references_sha256"], "Order/evaluator pins")
    return plan, parent, {"preparation": {"wall_seconds": parent_done["wall_seconds"], "completed_sha256": PREPARED_PIN},
                          "qualification": {"wall_seconds": cost_done["wall_seconds"], "counts": cost_done["counts"],
                                            "completed_sha256": spec["cost_completed"]["sha256"],
                                            "audit_sha256": spec["cost_audit"]["sha256"]}}


def actor_work(actor):
    """Reconstruct token/chunk work from integer metadata, without float arrays."""
    tokens = actor["tokens"]
    require(tokens and all(type(ids) is list and ids and all(integer(i) for i in ids) for ids in tokens), "Content token IDs")
    turns, queries, candidates = actor["turn_text_ids"], actor["query_text_ids"], actor["candidate_text_ids"]
    refs = turns + queries + [i for cs in candidates for i in cs]
    require(turns and queries and len(candidates) == len(queries) and all(cs for cs in candidates)
            and all(integer(i) and i < len(tokens) for i in refs) and set(refs) == set(range(len(tokens))), "Public text maps")
    require(len(actor["original_feature_ids"]) == len(tokens) == len(set(actor["original_feature_ids"])), "Original feature map")
    t, q, c = len(turns), len(queries), max(map(len, candidates))
    require(actor["lexical_shape"] == [t, q, c, 10] and len(actor["candidate_ids"]) == q
            and len(actor["query_ids"]) == q and actor["query_ids"] == sorted(set(actor["query_ids"])), "Actor layout/query inventory")
    for ids, cs in zip(actor["candidate_ids"], candidates, strict=True):
        require(len(ids) == len(cs) and 3 <= len(ids) <= 12 and len(set(ids)) == len(ids)
                and ids.count(metrics.NONE) == ids.count(metrics.DONTCARE) == 1, "Actor candidate support")
    lengths = [min(254, len(ids) - start) + 2 for ids in tokens for start in range(0, len(ids), 254)]
    batches = [lengths[i:i + 32] for i in range(0, len(lengths), 32)]
    padded = sum(len(b) * max(b) for b in batches)
    return {"unique_texts": len(tokens), "input_texts": len(tokens), "content_tokens": sum(map(len, tokens)),
            "chunks": len(lengths), "encoder_sequences": len(lengths), "special_token_positions": 2 * len(lengths),
            "valid_token_positions": sum(lengths), "padded_token_positions": padded,
            "padding_token_positions": padded - sum(lengths),
            "padded_attention_positions": sum(len(b) * max(b)**2 for b in batches), "encoder_calls": len(batches),
            "chunk_tokens": 254, "chunk_batch_size": 32, "overlength_texts_chunked": sum(len(x) > 254 for x in tokens),
            "truncated_tokens": 0, "max_chunk_tokens_with_special": max(lengths), "public_user_turns": t, "queries": q,
            "schema_text_occurrences": q, "candidate_text_occurrences": sum(map(len, candidates)), "max_candidates": c,
            "real_question_updates": t * q, "real_candidate_updates": t * sum(map(len, candidates)),
            "padded_candidate_positions": t * q * c, "lexical_scalars": t * q * c * 10, "lexical_bytes": t * q * c * 40,
            "max_content_tokens_per_text": max(map(len, tokens))}


def work_sum(profiles):
    return {key: sum(p[key] for p in profiles) for key in ENCODER_KEYS}


def state_counts(profiles):
    t, tq = sum(p["public_user_turns"] for p in profiles), sum(p["real_question_updates"] for p in profiles)
    result = {key: tq for key in ("executed_valid_question_slots", "real_question_updates", "incoming_checks",
                                 "feature_checks", "result_checks", "mass_checks")}
    result.update({key: t for key in ("advance_calls", "advance_returned", "valid_turns")})
    result.update({key: len(profiles) for key in ("forward_calls", "forward_returned")})
    return result


def prepared_metadata(prepared, parent, queries):
    actors, targets, profiles, canonical = {}, {}, {}, []
    refs = {"row_indices": [], "original": [], "numbers": [],
            "scope": "Deterministic public literal registers, accuracy references only"}
    counts = {split: Counter() for split in ("train", "dev")}
    for split in ("train", "dev"):
        for actor in lines(prepared / f"actors-{split}.jsonl"):
            key = (split, actor["dialogue_id"])
            require(actor["split"] == split and key not in actors, "Split-qualified public actor membership")
            actors[key], profiles[key] = actor, actor_work(actor)
        for record in lines(prepared / f"targets-{split}.jsonl"):
            key = (split, record["dialogue_id"])
            require(record["split"] == split and key in actors and key not in targets, "Evaluator/actor membership")
            actor = actors[key]
            targets[key] = record["rows"]
            seen = set()
            for i, row in enumerate(record["rows"]):
                require(row["split"] == split and row["dialogue_id"] == key[1] and row["source_row_index"] == i, "Source row ordering")
                t, q, qi = row["time"], row["query_position"], row["query_index"]
                require(integer(t) and t < len(actor["turn_text_ids"]) and integer(q) and q < len(actor["query_ids"])
                        and actor["query_ids"][q] == qi and row["turn_index"] == actor["user_turn_indices"][t]
                        and (t, q) not in seen, "Canonical public endpoint address")
                seen.add((t, q))
                query = queries[qi]
                ids, values = query["candidate_ids"], query["candidate_values"]
                require(query["split"] == split and query["id"] == row["query_id"] and query["service"] == row["service"]
                        and query["slot"] == row["slot"] and ids == actor["candidate_ids"][q]
                        and ids[row["label_index"]] == row["label_id"], "Exact canonical schema/label join")
                stratum = row["bin"] if row["bin"] in metrics.STRATA[:2] else "changed"
                require(row["bin"] in metrics.BINS and row["stratum"] == stratum
                        and row["stratum_index"] == metrics.STRATA.index(stratum), "Prepared loss bin/stratum")
                counts[split][str(row["stratum_index"])] += 1
                if split == "dev":
                    refs["row_indices"].append(len(canonical))
                    canonical.append({**row, "row_index": len(canonical), "candidate_ids": list(ids), "candidate_values": list(values)})
                    for name in ("original", "numbers"):
                        choice = record["literal_registers"][name][q][t]
                        require(integer(choice) and choice < len(ids) and ids[choice] != metrics.DONTCARE, "Literal register support")
                        refs[name].append(choice)
    require(set(actors) == set(targets), "Complete actor/evaluator join")
    require({s: sum(k[0] == s for k in actors) for s in counts} == parent["cohort_sizes"]
            and {s: dict(c) for s, c in counts.items()} == parent["loss_counts"], "Cohort/loss support")
    total = sum(counts["train"].values())
    require(parent["loss_weights"] == [total / (3 * counts["train"][k]) for k in ("0", "1", "2")], "Fit-only objective weights")
    workload = read(prepared / "workloads.json")
    require(workload["profiles"] == [{"split": s, "dialogue_id": d, "work": p} for (s, d), p in profiles.items()], "Every token work profile")
    orders = read(prepared / "orders.json")
    require(orders["seeds"] == list(metrics.SEEDS) and orders["epochs"] == CONFIG["epochs"], "All paired seed/epoch orders")
    dids = orders["dialogue_ids"]
    require(len(set(dids)) == len(dids) and set(dids) == {d for s, d in actors if s == "train"}
            and set(orders["orders"]) == {str(s) for s in metrics.SEEDS}, "Order inventory")
    for epochs in orders["orders"].values():
        require(len(epochs) == CONFIG["epochs"] and all(all(integer(i) for i in order)
                and sorted(order) == list(range(len(dids))) for order in epochs), "Every epoch exact permutation")
    return {"profiles": profiles, "targets": targets, "rows": canonical, "references": refs, "orders": orders}


def expected_work(meta):
    train = [p for (s, _), p in meta["profiles"].items() if s == "train"]
    dev = [p for (s, _), p in meta["profiles"].items() if s == "dev"]
    epochs = CONFIG["epochs"]
    per = {"optimizer_updates": epochs * math.ceil(len(train) / CONFIG["effective_batch"]),
           "training_dialogue_visits": epochs * len(train), "backward_calls": epochs * len(train),
           "evaluation_forwards": len(dev), "training_endpoints": epochs * sum(len(r) for (s, _), r in meta["targets"].items() if s == "train"),
           "evaluation_endpoints": len(meta["rows"]), "memory_forwards": epochs * len(train) + len(dev),
           "encoding_passes": epochs * len(train) + len(dev)}
    work = {k: epochs * work_sum(train)[k] + work_sum(dev)[k] for k in ENCODER_KEYS}
    counts = {k: epochs * state_counts(train)[k] + state_counts(dev)[k] for k in state_counts(train)}
    per["encoder_calls"] = work["encoder_calls"]
    return {"per_fit": per, "all_fits": {k: v * 12 for k, v in per.items()}, "fits": 12,
            "encoder_work_per_fit": work, "state_counts_per_fit": counts}


def validate_invariants(value, expected):
    require(set(value) == set(COUNT_KEYS) | set(MAX_KEYS) | {"mass_min", "mass_max", "tolerance"}, "Invariant schema")
    require(all(integer(value[k]) for k in COUNT_KEYS) and {k: value[k] for k in expected} == expected, "Every public update checked")
    require(value["tolerance"] == 2e-6 and all(finite(value[k]) and value[k] <= 2e-6 for k in MAX_KEYS), "State normalization bounds")
    lo, hi = value["mass_min"], value["mass_max"]
    require(finite(lo) and finite(hi) and lo <= hi <= 1 + 2e-6, "Released mass bounds")
    require(value["mass_max_overshoot"] == max(0., hi - 1.)
            and value["mass_above_one_count"] <= value["mass_checks"]
            and (value["mass_above_one_count"] > 0) == (hi > 1), "Mass roundoff accounting")


def merge_invariants(values):
    return {**{k: sum(v[k] for v in values) for k in COUNT_KEYS},
            **{k: max(v[k] for v in values) for k in MAX_KEYS},
            "mass_min": min(v["mass_min"] for v in values), "mass_max": max(v["mass_max"] for v in values), "tolerance": 2e-6}


def audit_fit(directory, fit, expected, meta, seed, arm):
    require(fit["status"] == "completed" and fit["version"] == STUDY and fit["fit_id"] == directory.name
            and fit["arm"] == arm and fit["seed"] == seed, "Fit identity")
    require(set(fit["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}, "Exact fit payloads")
    manifest(directory, fit["files"])
    init = fit["initial_sha256"]
    require(set(init) == {"encoder", "memory"} and all(digest(x) for x in init.values())
            and digest(fit["final_encoder_sha256"]), "Initialization/final state digests")
    require((fit["final_encoder_sha256"] != init["encoder"]) == arm.startswith("trainable_"), "Frozen/trainable encoder witness")
    require({k: fit["counts"][k] for k in expected["per_fit"]} == expected["per_fit"]
            and all(integer(v) for v in fit["counts"].values()), "Complete fit operations")
    attempts = {"encoding_attempts": "encoding_passes", "memory_forward_attempts": "memory_forwards",
                "backward_attempts": "backward_calls", "optimizer_attempts": "optimizer_updates",
                "encoder_forward_attempts": "encoder_calls", "encoder_forward_returns": "encoder_calls"}
    require(set(fit["counts"]) == set(expected["per_fit"]) | set(attempts)
            and all(fit["counts"][a] == fit["counts"][b] for a, b in attempts.items()), "Attempt/return closure")
    require(fit["encoder_work"] == expected["encoder_work_per_fit"], "Complete encoder work")
    validate_invariants(fit["invariants"], expected["state_counts_per_fit"])
    orders, profiles = meta["orders"], meta["profiles"]
    journal = iter(lines(directory / "updates.jsonl"))
    audits, work, journal_seconds = [], Counter(), 0.
    for epoch, order in enumerate(orders["orders"][str(seed)]):
        for start in range(0, len(order), CONFIG["effective_batch"]):
            event = next(journal, None)
            require(event is not None, "Missing effective update")
            ids = [orders["dialogue_ids"][i] for i in order[start:start + CONFIG["effective_batch"]]]
            batch = [profiles[("train", did)] for did in ids]
            denominator = sum(len(meta["targets"][("train", did)]) for did in ids)
            require(event["epoch"] == epoch and event["batch_start"] == start and event["dialogue_ids"] == ids
                    and event["microbatches"] == len(ids) and event["endpoint_count"] == denominator, "Paired batch order/denominator")
            require(event["encoder_work"] == work_sum(batch), "Actual encoder microbatch coverage")
            validate_invariants(event["invariants"], state_counts(batch))
            require(finite(event["weighted_training_loss"]) and finite(event["wall_seconds"], positive=True)
                    and finite(event["memory_gradient_squared_norm"], positive=True)
                    and finite(event["gradient_norm_before_clip"]), "Finite loss/gradient/time witnesses")
            grads = event["encoder_gradient_squared_norms"]
            require(set(grads) == GRADIENT_KEYS and all(finite(v, positive=True) if arm.startswith("trainable_") else v is None
                    for v in grads.values()), "Actual encoder gradient path witness")
            work.update(event["encoder_work"])
            audits.append(event["invariants"])
            journal_seconds += event["wall_seconds"]
    require(next(journal, None) is None, "Extra effective update")
    evaluation = fit["evaluation"]
    dev = [p for (s, _), p in profiles.items() if s == "dev"]
    require(evaluation["rows"] == len(meta["rows"]) and evaluation["encoder_work"] == work_sum(dev), "Complete final evaluation")
    validate_invariants(evaluation["invariants"], state_counts(dev))
    require(merge_invariants([*audits, evaluation["invariants"]]) == fit["invariants"], "Actual invariant aggregation")
    work.update(evaluation["encoder_work"])
    require(dict(work) == fit["encoder_work"], "Actual encoder work aggregation")
    checkpoint = fit["checkpoint"]
    require({k: checkpoint[k] for k in ("sha256", "bytes")} == fit["files"]["weights.pt"], "Checkpoint byte binding")
    train, evaluate, save, whole = fit["training_seconds"], evaluation["wall_seconds"], checkpoint["wall_seconds"], fit["wall_seconds"]
    require(all(finite(v, positive=True) for v in (train, evaluate, save, whole))
            and journal_seconds <= train and train + evaluate + save <= whole, "Nonoverlapping fit phase timing")
    return {"training_seconds": train, "evaluation_seconds": evaluate, "checkpoint_seconds": save,
            "fit_wall_seconds": whole, "journal_update_seconds": journal_seconds,
            "counts": fit["counts"], "encoder_work": fit["encoder_work"], "invariants": fit["invariants"],
            "initial_sha256": init, "final_encoder_sha256": fit["final_encoder_sha256"]}


def authenticate_run(run, completed_pin, plan_path, plan_pin):
    done = terminal(run, completed_pin, "train")
    require(done["version"] == STUDY and done["fit_count"] == 12
            and done["quality_scoring_in_runner"] is False and done["official_test_opened"] is False
            and done["external_model_api_calls"] == 0, "Complete no-selection scientific execution")
    members = {"started.json", "plan.json", "allocation.json", "evaluation-rows.jsonl", "references.json"}
    members |= {f"{fit}/{name}" for fit in FIT_ORDER for name in ("completed.json", "weights.pt", "updates.jsonl", "predictions.npz")}
    require(set(done["files"]) == members and [v["fit_id"] for v in done["fits"]] == FIT_ORDER, "Exact twelve-fit execution closure")
    plan, parent, inherited = authenticate_lineage(run, done, plan_path, plan_pin)
    packet_path = "runs/sgd-state-v1/features-02/packet.json"
    require(packet_path in parent["input_sha256"], "Canonical public schema bound before decoding")
    meta = prepared_metadata(Path(plan["prepared"]), parent, read(ROOT / packet_path)["queries"])
    require(len(meta["rows"]) == DEV_ENDPOINTS and list(lines(run / "evaluation-rows.jsonl")) == meta["rows"]
            and read(run / "references.json") == meta["references"], "Every original canonical row/reference")
    expected = expected_work(meta)
    require(plan["expected"] == expected and done["counts"] == expected["all_fits"], "Independent full work totals")
    costs, seed_init, encoder_init = {}, {}, None
    for item in done["fits"]:
        fit_id = item["fit_id"]
        path = run / fit_id / "completed.json"
        require(sha(path) == item["completed_sha256"], "Root/fit completion join")
        fit = read(path)
        arm, seed_text = fit_id.rsplit("-", 1)
        seed = int(seed_text)
        costs[fit_id] = audit_fit(path.parent, fit, expected, meta, seed, arm)
        initial = fit["initial_sha256"]
        require(seed_init.setdefault(seed, initial) == initial, "Four-arm paired initialization")
        encoder_init = initial["encoder"] if encoder_init is None else encoder_init
        require(initial["encoder"] == encoder_init, "Identical pretrained encoder across all fits")
    limits = plan["allocation"]["limits"]
    require(finite(done["wall_seconds"], positive=True) and done["wall_seconds"] < limits["wall_seconds"]
            and integer(done["peak_rss_bytes"]) and 0 < done["peak_rss_bytes"] <= limits["rss_bytes"]
            and integer(done["sampled_mps_driver_max_bytes"]) and integer(done["sampled_mps_current_max_bytes"])
            and 0 <= done["sampled_mps_current_max_bytes"] <= done["sampled_mps_driver_max_bytes"] <= limits["mps_driver_bytes"],
            "Allocated execution resources")
    require(sum(v["fit_wall_seconds"] for v in costs.values()) <= done["wall_seconds"]
            and sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) <= limits["output_bytes"], "Nested fit cost/output cap")
    return done, plan, meta, costs, inherited


def aggregate(packets, rows, references):
    """Reusable pure saved-packet scoring; caller owns complete authentication."""
    require(set(packets) == set(FIT_ORDER), "All twelve raw prediction packets required")
    fits = {}
    for name in FIT_ORDER:
        packet = packets[name]
        require(set(packet) == {"row_indices", "log_probs"}, "Exact prediction fields")
        fits[name] = metrics.score_panels(packet["log_probs"], rows, row_indices=packet["row_indices"])
    refs = {name: metrics.literal_metrics(np.asarray(references[name], dtype=np.int64),
                                         np.asarray(references["row_indices"], dtype=np.int64), rows)
            for name in ("original", "numbers")}
    return {"fits": fits, "references": refs, "factorial": metrics.factorial_summary(fits), "continuation": metrics.criteria(fits)}


def authenticate_supervision(args, plan, done):
    require(sha(args.launch) == args.launch_sha256 and sha(args.terminal) == args.terminal_sha256,
            "External process supervisor pins")
    launch, end = read(args.launch), read(args.terminal)
    command = launch["command"]
    require(type(command) is list and command == end["command"] and all(type(x) is str for x in command),
            "Same supervised command")
    require(Path(command[0]).resolve() == Path(sys.executable).resolve(), "Supervised interpreter")
    tail = command[1:]
    if tail and tail[0] == "-u":
        tail = tail[1:]
    require(len(tail) == 10 and (ROOT / tail[0]).resolve() == ROOT / "scripts/study_dialogue_observation_v2.py"
            and tail[1] == "train", "Supervised scientific training command")
    flags = dict(zip(tail[2::2], tail[3::2], strict=True))
    started = read(args.run / "started.json")
    require(set(flags) == {"--plan", "--plan-sha256", "--out", "--supervision"}
            and Path(flags["--plan"]).resolve() == args.plan.resolve()
            and flags["--plan-sha256"] == args.plan_sha256
            and Path(flags["--out"]).resolve() == Path(started["request"]["out"]).resolve()
            and Path(flags["--supervision"]).resolve() == args.launch.resolve()
            == Path(started["request"]["supervision"]).resolve(),
            "Supervised plan/output identity")
    require(integer(launch["pid"]) and launch["pid"] > 0 and launch["pid"] == launch["pgid"] == end["pgid"],
            "Dedicated process group witness")
    require(launch["watchdog_sha256"] == plan["source_sha256"][WATCHDOG]
            and launch["clock_source_sha256"] == plan["source_sha256"][CLOCK]
            and launch["cap_seconds"] == plan["allocation"]["limits"]["wall_seconds"], "Frozen supervisor/cap")
    require(type(end["returncode"]) is int and end["returncode"] == 0 and end["timed_out"] is False
            and end["error"] is None and end["group_absent"] is True, "Successful terminal process-group witness")
    clock_envelope(launch, end, started, done, launch["cap_seconds"])
    require(started["supervision_sha256"] == done["supervision_sha256"] == args.launch_sha256, "Worker launch digest binding")
    require(launch["version"] == end["version"] == "dialogue-observation-supervision-v2"
            and all(end[k] == launch[k] for k in ("parent_pid", "pid", "pgid", "cwd", "cap_seconds",
                                                   "watchdog_sha256", "clock_source_sha256", "started_unix"))
            and integer(launch["parent_pid"]) and launch["parent_pid"] > 0
            and end["clock_error"] is None and end["cleanup"]["reaped"] is True
            and end["cleanup"]["errors"] == [] and end["cleanup"]["group_absent"] is True,
            "Complete supervisor origin and cleanup identity")
    return {"launch_sha256": args.launch_sha256, "terminal_sha256": args.terminal_sha256,
            "watchdog_sha256": launch["watchdog_sha256"], "wall_seconds": end["wall_seconds"],
            "clock_backend": end["clock_backend"], "elapsed_ns": end["elapsed_ns"],
            "deadline_ns": end["deadline_ns"], "civil_time_used_for_admission": False,
            "scope": "Authenticated external process-group and timing witnesses; no live PID query"}


def report_text(summary):
    gate = summary["continuation"]
    text = ["# Observation-learning development comparison", "",
            f"Technical validity: all twelve fits completed and authenticated. Scientific continuation: **{'PASS' if gate['passed'] else 'FAIL'} ({gate['checks_passed']}/7)**.", "",
            "Official DEV is exposed development data. This changes observation learning and lexical matching, not recurrent architecture. No calibration, connectome or world-model advantage is established.", "",
            "| Fit | Unseen macro accuracy | Unseen NLL | Unseen Brier | Seen macro accuracy | Seen assigned error | Unseen assigned error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def value(x, percent=False):
        return "undefined" if x is None else f"{100*x:.4f}%" if percent else f"{x:.6f}"
    for name in FIT_ORDER:
        panels = summary["fits"][name]["panels"]
        seen, unseen = panels["seen"], panels["unseen"]
        values = [value(unseen["macro_three"]["accuracy"], True), value(unseen["micro"]["nll"]), value(unseen["micro"]["brier"]),
                  value(seen["macro_three"]["accuracy"], True), value(seen["strata"]["assigned_retention"]["error"], True),
                  value(unseen["strata"]["assigned_retention"]["error"], True)]
        text.append("| " + name + " | " + " | ".join(values) + " |")
    text.extend(["", "Seven predeclared conditions:", ""])
    for check in gate["checks"]:
        text.append(f"- {check['name']}: {'PASS' if check['passed'] else 'FAIL'}.")
    cost = summary["costs"]
    text.extend(["", f"Whole execution: {cost['execution_wall_seconds']:.3f} s. Training: {cost['training_seconds']:.3f} s; final evaluation: {cost['evaluation_seconds']:.3f} s; checkpoint writing: {cost['checkpoint_seconds']:.3f} s. Root and parent elapsed time include suspend. Fit phases retain V1 perf_counter diagnostics, can exclude suspend, and never determine admission; they are nested, not added to execution.", "",
                 "summary.json retains every arm/seed, literal reference, panel, service, transition/type support and false-positive denominator, four paired contrasts and their factorial interaction. Positive accuracy differences are better; positive NLL/Brier/error differences are worse. Missing support remains undefined.", "",
                 "Coverage and byte identities are checked from saved metadata. Internal normalization, gradients, parameter initialization and encoder execution are authenticated source-bound witnesses; no tensor execution is replayed. Reference semantics inherit the prepared public lexical registers. No model was called by this report."])
    return "\n".join(text) + "\n"


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    handler = None
    sources = {}
    clock, deadline, elapsed_ns = None, None, None
    def check():
        nonlocal elapsed_ns
        now = clock.now_ns()
        elapsed_ns = now - deadline.started_ns
        require(now < deadline.expires_ns and rss() <= CAPS["rss_bytes"], "Report wall/RSS cap")
        require(sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) <= CAPS["output_bytes"], "Report output cap")
    try:
        clock = SuspendClock()
        deadline = clock.deadline_after(CAPS["wall_seconds"])
        def timeout(*_):
            raise TimeoutError("Report wall cap")
        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, CAPS["wall_seconds"])
        sources = {name: sha(ROOT / name) for name in ("scripts/report_dialogue_observation_v2.py", "tests/test_report_dialogue_observation_v2.py",
                                                     "src/openjev/research/dialogue_observation_metrics.py", "tests/test_dialogue_observation_metrics.py",
                                                     CLOCK, CLOCK_TEST)}
        write(args.out / "started.json", {"version": VERSION, "request": request, "source_sha256": sources, "limits": CAPS})
        done, plan, meta, fits_cost, inherited = authenticate_run(args.run, args.completed_sha256, args.plan, args.plan_sha256)
        supervision = authenticate_supervision(args, plan, done)
        check()
        packets = {}
        for name in FIT_ORDER:
            with np.load(args.run / name / "predictions.npz", allow_pickle=False) as packet:
                require(set(packet.files) == {"row_indices", "log_probs"}, "Saved prediction membership")
                packets[name] = {k: packet[k] for k in packet.files}
        summary = aggregate(packets, meta["rows"], meta["references"])
        costs = {"execution_wall_seconds": done["wall_seconds"], "process_lifetime_peak_rss_bytes": done["peak_rss_bytes"],
                 "sampled_mps_driver_max_bytes": done["sampled_mps_driver_max_bytes"], "sampled_mps_current_max_bytes": done["sampled_mps_current_max_bytes"],
                 "fits": fits_cost, "inherited_separate_phases": inherited, "operation_totals": done["counts"],
                 "external_supervision": supervision,
                 "scope": "Root and parent elapsed are suspend-inclusive; fit/training/evaluation/checkpoint times are nested perf_counter diagnostics that may exclude suspend and are never admission clocks; RSS is process lifetime highwater and sampled MPS is not a transient peak or additive to RSS"}
        for key in ("training_seconds", "evaluation_seconds", "checkpoint_seconds", "fit_wall_seconds"):
            costs[key] = math.fsum(v[key] for v in fits_cost.values())
        summary.update({"version": VERSION, "status": "completed", "technical_validity_passed": True,
                        "technical_complete_fits": 12, "execution_completed_sha256": args.completed_sha256,
                        "plan_sha256": args.plan_sha256, "costs": costs,
                        "scope": "All DEV endpoints; exposed development comparison; saved-only scoring with authenticated internal execution witnesses; zero model/tokenizer/encoder calls"})
        write(args.out / "summary.json", summary)
        with (args.out / "report.md").open("x") as stream:
            stream.write(report_text(summary))
        # Authenticate every execution payload again and preserve the original external pins.
        manifest(args.run, done["files"])
        require(sha(args.run / "completed.json") == args.completed_sha256 and sha(args.plan) == args.plan_sha256, "End terminal/plan stability")
        require(sha(args.launch) == args.launch_sha256 and sha(args.terminal) == args.terminal_sha256, "End supervisor stability")
        bind(plan["source_sha256"])
        check()
        files = {name: descriptor(args.out / name) for name in ("started.json", "summary.json", "report.md")}
        receipt = {"version": VERSION, "status": "completed", "technical_validity_passed": True,
                   "execution_completed_sha256": args.completed_sha256, "plan_sha256": args.plan_sha256,
                   "launch_sha256": args.launch_sha256, "terminal_sha256": args.terminal_sha256,
                   "source_sha256": sources, "execution_source_sha256": plan["source_sha256"], "request": request,
                   "continuation_passed": summary["continuation"]["passed"], "scientific_checks_total": 7,
                   "scientific_checks_passed": summary["continuation"]["checks_passed"], "files": files,
                   "wall_seconds": elapsed_ns / 1_000_000_000, "elapsed_ns": elapsed_ns,
                   "clock_backend": clock.backend, "started_ns": deadline.started_ns,
                   "finished_ns": deadline.started_ns + elapsed_ns, "deadline_ns": deadline.expires_ns,
                   "timing_available": True, "peak_rss_bytes": rss(), "model_calls": 0,
                   "scope": summary["scope"], "limits": CAPS}
        write(args.out / "receipt.json", receipt)
        check()
        return receipt
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request,
                  "source_sha256": sources, "error_type": type(error).__name__, "error": str(error),
                  "wall_seconds": None, "elapsed_ns": None, "timing_available": False,
                  "last_successful_elapsed_ns": elapsed_ns,
                  "elapsed_scope": "Terminal timing unavailable; last successful check retained separately",
                  "peak_rss_bytes": rss(), "model_calls": 0,
                  "scientific_result_qualified": False})
        except BaseException as secondary:  # noqa: BLE001 - original failure remains authoritative
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = execute(parser.parse_args())
    print(json.dumps({"status": result["status"], "continuation_passed": result["continuation_passed"]}))
