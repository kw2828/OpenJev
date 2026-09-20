"""Independent saved cost-pilot accounting. Standard library only; never loads tensors.

Invoke only after external completion and process-terminal pins are supplied.
Tokenization, token-profile reconstruction and label isolation inherit the pinned
preparation audit. Numerical witnesses are checked, not experimentally replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "output/dialogue-observation-learning-v1"
VERSION = "dialogue-observation-batches-v1"
COST_PIN = "94ead7a9c7b5cd831e0245055988ac8932c1e87c9a68792c935b04b4aae5c946"
COST_PLAN = "14f15388577448024354839aa9c7c7ef9b25743118f7811aa46dbb65f9b913c4"
PREP_PIN = "d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83"
PREP_PLAN = "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e"
PREP_AUDIT = "4fefa46325c8c8e4a74a6f14dcf8a7961f13c875faeb14b3f03ee35169c2c23c"
WATCHDOG = "d3992f019cc5135d162b074327a73bdbd3b4c160ddc89f990165cd345c578242"
PROTOCOL = "research/dialogue-observation-learning-cost-protocol.md"
PROTOCOL_PIN = "731aad524200b3cf1db4d471760a84341ffde1045bdaeb4d89290fdbd36dc1b2"
NEW = {
    "scripts/qualify_dialogue_observation_batches.py": "ceb97d5c8060353b0bb95886e7862d0c330b3d0767fedafec0b5a208ca0362b0",
    "tests/test_qualify_dialogue_observation_batches.py": "7e108aac3e98a24ad5d0305bae95b733eaf17494a56d04507b037c152b21db3d",
    "src/openjev/research/dialogue_finetune_training.py": "22af61efedc7a5dce91740ceabd7d2736432dca20658c4d4069bd5fb6c5461d8",
    "tests/test_dialogue_finetune_training.py": "1027bf60c321d826378c654efee1d68ce36b2e0df6ff65281cd69006dba7f4a9",
    PROTOCOL: PROTOCOL_PIN,
}
ARMS = ["frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers"]
CAPS = {"wall_seconds": 300, "rss_bytes": 8 * 1024**3,
        "mps_driver_bytes": 8 * 1024**3, "output_bytes": 256 * 1024**2}
AUDIT_CAPS = {"wall_seconds": 60, "rss_bytes": 2 * 1024**3, "output_bytes": 32 * 1024**2}
CONFIG = {"seed": 6901, "arms": ARMS, "encoder_batch": 32, "chunk_tokens": 254,
          "memory_lr": .001, "encoder_lr": .00002, "weight_decay": .0001, "clip": 1.,
          "parity_tolerance": 2e-5, "cpu_threads": 1, "interop_threads": 1,
          "encoder_device": "mps", "memory_device": "cpu", "dtype": "float32"}
PREP_FILES = {"started.json", "source-pins.json", "plan.json", "actors-train.jsonl", "actors-dev.jsonl",
              "targets-train.jsonl", "targets-dev.jsonl", "index.json", "lexical-original.npy",
              "lexical-numbers.npy", "workloads.json", "orders.json", "effective-batches.json"}
ENCODER = ("input_texts", "content_tokens", "encoder_sequences", "encoder_calls", "special_token_positions",
           "valid_token_positions", "padded_token_positions", "padded_attention_positions",
           "padding_token_positions", "overlength_texts_chunked", "truncated_tokens")
BATCH_METRICS = ("encoder_calls", "padded_token_positions", "padded_attention_positions",
                 "real_question_updates", "padded_candidate_positions", "encoder_sequences")
SINGLE_METRICS = ("padded_attention_positions", "encoder_sequences", "max_chunk_tokens_with_special",
                  "padded_candidate_positions", "real_question_updates", "public_user_turns")
INVARIANT_COUNTS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
                    "executed_valid_question_slots", "real_question_updates", "incoming_checks",
                    "feature_checks", "result_checks", "mass_checks", "mass_above_one_count")
INVARIANT_MAX = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")
EXPECTED = {"cells": 40, "optimizer_updates": 52, "training_dialogue_visits": 1168, "evaluation_forwards": 24,
            "parity_dialogue_visits": 412, "backward_calls": 1168, "memory_forwards": 1192,
            "encoding_passes": 1604, "checkpoints": 4}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def integer(value):
    return type(value) is int and value >= 0


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def sha(path):
    need(path.is_file() and not path.is_symlink(), "Non-regular or symlink file: " + str(path))
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON literal: " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def authenticate(folder, pin, names):
    need(digest(pin) and sha(folder / "completed.json") == pin, "External completion identity")
    done = read(folder / "completed.json")
    need(done["status"] == "completed" and set(done["files"]) == names, "Complete payload manifest")
    need({p.name for p in folder.iterdir()} == names | {"completed.json"}, "Exact directory closure")
    for name, entry in done["files"].items():
        need(Path(name).name == name and digest(entry["sha256"]) and integer(entry["bytes"]), "Manifest entry")
        path = folder / name
        need(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Payload bytes: " + name)
    return done


def elapsed(value, ceiling=None):
    need(finite(value), "Nonnegative finite elapsed time")
    if ceiling is not None:
        need(value <= ceiling + 1e-6, "Timing exceeds enclosing interval")
    return value


def recorded_caps(done):
    elapsed(done["wall_seconds"], CAPS["wall_seconds"])
    need(integer(done["peak_rss_bytes"]) and 0 < done["peak_rss_bytes"] <= CAPS["rss_bytes"], "Recorded RSS cap")


def source_map(mapping):
    for name, pin in mapping.items():
        path = ROOT / name
        need(not Path(name).is_absolute() and ".." not in Path(name).parts, "Repository source path")
        need(digest(pin) and sha(path) == pin, "Frozen source identity: " + name)


def reconstruct_cases(profiles, batches, orders):
    """Independently scan every saved epoch and batch, including the one-row tail."""
    ids = orders["dialogue_ids"]
    need(len(ids) == len(set(ids)) == 2017 and orders["epochs"] == 20
         and orders["seeds"] == [6901, 6902, 6903], "Complete saved order dimensions")
    need(set(orders["orders"]) == {"6901", "6902", "6903"}, "Seed membership")
    maxima, sizes = {}, Counter()
    for seed in orders["seeds"]:
        epochs = orders["orders"][str(seed)]
        need(len(epochs) == 20, "All saved epochs")
        for epoch, order in enumerate(epochs):
            need(all(integer(x) for x in order) and sorted(order) == list(range(2017)), "Epoch permutation")
            for start in range(0, 2017, 32):
                indices = order[start:start + 32]
                dids = [ids[i] for i in indices]
                sizes[len(indices)] += 1
                for metric in BATCH_METRICS:
                    value = sum(profiles[("train", did)][metric] for did in dids)
                    if metric not in maxima or value > maxima[metric]["value"]:
                        maxima[metric] = {"seed": seed, "epoch": epoch, "batch_start": start,
                                          "dialogue_indices": indices, "dialogue_ids": dids, "value": value}
    need(batches["maxima"] == maxima and batches["batches"] == sum(sizes.values()) == 3840,
         "Independent full-order maximum reconstruction")
    need(batches["sizes"] == {str(k): v for k, v in sizes.items()}, "All effective batch sizes")
    grouped = {}
    # Metadata was serialized with sorted keys; the published reason-list order is retained.
    for metric in sorted(BATCH_METRICS):
        record = maxima[metric]
        key = (record["seed"], record["epoch"], record["batch_start"])
        entry = grouped.setdefault(key, {"dialogue_ids": record["dialogue_ids"], "maximizes": []})
        need(len(record["dialogue_ids"]) == 32, "Maximum is a complete batch")
        entry["maximizes"].append(metric)
    need(len(grouped) == 3, "Three distinct maximum batches")
    cases = [{"case_id": f"batch-{i}", "kind": "batch", "split": "train", "seed": key[0],
              "epoch": key[1], "batch_start": key[2], "iterations": 3, **value}
             for i, (key, value) in enumerate(sorted(grouped.items()))]
    for split in ("train", "dev"):
        population = [did for s, did in profiles if s == split]
        winners = {}
        for metric in SINGLE_METRICS:
            did = min(population, key=lambda d: (-profiles[(split, d)][metric], d))
            winners.setdefault(did, []).append(metric)
        need(len(winners) == 3, "Three maximum singles per split")
        for i, (did, reasons) in enumerate(sorted(winners.items())):
            cases.append({"case_id": f"{split}-single-{i}", "kind": "single" if split == "train" else "eval",
                          "split": split, "dialogue_ids": [did], "maximizes": reasons,
                          "iterations": 1 if split == "train" else 2})
    cases.insert(6, {"case_id": "tail", "kind": "tail", "split": "train", "seed": 6901, "epoch": 0,
                     "batch_start": 2016, "dialogue_ids": [ids[orders["orders"]["6901"][0][-1]]], "iterations": 1})
    return cases


def metadata():
    prep, cost = BASE / "preparation-02", BASE / "cost-preparation-01"
    done = authenticate(prep, PREP_PIN, PREP_FILES)
    need(done["plan_sha256"] == PREP_PLAN and sha(prep / "plan.json") == PREP_PLAN
         and done["encoder_calls"] == 0 and done["model_weights_loaded"] is False
         and done["official_test_opened"] is False, "Full preparation lineage and scope")
    parent = read(prep / "plan.json")
    need(parent["preparation_version"] == 2 and parent["study"] == "dialogue-observation-learning-v1",
         "Corrected preparation identity")
    need(parent["data_files"] == {k: v for k, v in done["files"].items() if k != "plan.json"}, "Prepared data map")
    need(len(parent["source_sha256"]) == 39
         and read(prep / "source-pins.json") == parent["source_sha256"], "Inherited source closure")
    audit_path = BASE / "preparation-audit-02/receipt.json"
    need(sha(audit_path) == PREP_AUDIT, "Independent full token-profile audit pin")
    prior_audit = read(audit_path)
    need(prior_audit["status"] == "clear" and prior_audit["agreement"] is True
         and prior_audit["preparation_completed_sha256"] == PREP_PIN
         and prior_audit["profiles_reconstructed"] == 4380, "Inherited reconstruction scope")
    need(sha(audit_path.parent / "audit.py") == prior_audit["script_sha256"], "Prior independent audit source")
    cd = authenticate(cost, COST_PIN, {"started.json", "plan.json"})
    recorded_caps(cd)
    need(cd["phase"] == "prepare-cost-plan" and cd["model_calls"] == 0 and cd["plan_sha256"] == COST_PLAN
         and sha(cost / "plan.json") == COST_PLAN, "Fixed cost-plan receipt")
    plan, started = read(cost / "plan.json"), read(cost / "started.json")
    need(plan["version"] == VERSION and plan["config"] == CONFIG and plan["caps"] == CAPS
         and plan["runtime"] == parent["runtime"], "Cost configuration/runtime identity")
    need(plan["prepared"] == str(prep.resolve()) and plan["prepared_completed_sha256"] == PREP_PIN
         and plan["prepared_plan_sha256"] == PREP_PLAN and plan["protocol_sha256"] == PROTOCOL_PIN,
         "Cost plan parent/protocol")
    need(plan["source_sha256"] == {**parent["source_sha256"], **NEW}
         and len(plan["source_sha256"]) == 44, "Exact source union")
    source_map(plan["source_sha256"])
    need(started == {"version": VERSION, "phase": "prepare-cost-plan", "config": CONFIG, "caps": CAPS,
                     "runtime": plan["runtime"], "request": {"prepared": str(prep.resolve()),
                     "prepared_sha256": PREP_PIN, "protocol_sha256": PROTOCOL_PIN}}, "Cost preparation request")
    workloads, index = read(prep / "workloads.json"), read(prep / "index.json")
    profiles = {(x["split"], x["dialogue_id"]): x["work"] for x in workloads["profiles"]}
    layouts = {(x["split"], x["dialogue_id"]): x for x in index}
    need(len(profiles) == len(workloads["profiles"]) == len(layouts) == len(index) == 4380
         and set(profiles) == set(layouts), "All split-qualified work/index rows")
    need(Counter(s for s, _ in profiles) == {"train": 2017, "dev": 2363}, "Complete cohorts")
    for key, work in profiles.items():
        need(all(integer(v) for v in work.values()), "Integer work profile")
        t, q, c, f = layouts[key]["shape"]
        need(f == 10 and (t, q, c) == (work["public_user_turns"], work["queries"], work["max_candidates"])
             and work["real_question_updates"] == t * q and work["padded_candidate_positions"] == t * q * c
             and work["lexical_scalars"] == t * q * c * f and work["lexical_bytes"] == 4 * t * q * c * f,
             "Independent index/profile state geometry")
        need(integer(layouts[key]["scored_rows"]) and 0 < layouts[key]["scored_rows"] <= t * q,
             "Endpoint denominator support")
        need(work["encoder_sequences"] == work["chunks"] and work["encoder_calls"] == (work["chunks"] + 31) // 32
             and work["special_token_positions"] == 2 * work["chunks"]
             and work["valid_token_positions"] == work["content_tokens"] + work["special_token_positions"]
             and work["padded_token_positions"] == work["valid_token_positions"] + work["padding_token_positions"]
             and work["truncated_tokens"] == 0, "Independent encoder-work identities")
    cases = reconstruct_cases(profiles, read(prep / "effective-batches.json"), read(prep / "orders.json"))
    need(plan["cases"] == cases and plan["expected_counts"] == EXPECTED, "Fixed selected case schedule")
    return plan, profiles, layouts, cases


def check_invariants(record, visits, turns, slots):
    keys = set(INVARIANT_COUNTS) | set(INVARIANT_MAX) | {"mass_min", "mass_max", "tolerance"}
    need(set(record) == keys and record["tolerance"] == 2e-6, "Exact invariant schema")
    expected = dict.fromkeys(("forward_calls", "forward_returned"), visits)
    expected.update(dict.fromkeys(("advance_calls", "advance_returned", "valid_turns"), turns))
    expected.update(dict.fromkeys(("executed_valid_question_slots", "real_question_updates", "incoming_checks",
                                   "feature_checks", "result_checks", "mass_checks"), slots))
    need(all(integer(record[k]) and record[k] == value for k, value in expected.items()), "Every executed state step covered")
    need(all(finite(record[k]) and record[k] <= 2e-6 for k in INVARIANT_MAX), "Recorded normalization bounds")
    need(finite(record["mass_min"]) and finite(record["mass_max"])
         and record["mass_min"] <= record["mass_max"] <= 1 + 2e-6, "Released mass range")
    above = record["mass_above_one_count"]
    need(integer(above) and above <= slots and (above > 0) == (record["mass_max"] > 1), "Mass-overshoot count")
    need(record["mass_max_overshoot"] == max(0., record["mass_max"] - 1), "Mass-overshoot maximum")


def folded_invariants(records):
    return {**{k: sum(r[k] for r in records) for k in INVARIANT_COUNTS},
            **{k: max(r[k] for r in records) for k in INVARIANT_MAX},
            "mass_min": min(r["mass_min"] for r in records),
            "mass_max": max(r["mass_max"] for r in records), "tolerance": 2e-6}


def run_audit(args):
    plan, profiles, layouts, cases = metadata()
    run = args.run.resolve()
    names = {"started.json", "events.jsonl", "summary.json"} | {f"checkpoint-{a}.pt" for a in ARMS}
    done = authenticate(run, args.completed_sha256, names)
    recorded_caps(done)
    need(done["version"] == VERSION and done["phase"] == "model-pilot"
         and done["cost_plan_sha256"] == COST_PLAN and done["cost_preparation_completed_sha256"] == COST_PIN
         and done["prepared_completed_sha256"] == PREP_PIN and done["official_targets_used"] is False
         and done["official_test_opened"] is False, "Execution lineage/scope")
    output_bytes = sum(p.stat().st_size for p in run.iterdir())
    need(output_bytes <= CAPS["output_bytes"], "Terminal output-byte cap")
    need(sha(args.launch) == args.launch_sha256 and sha(args.terminal) == args.terminal_sha256,
         "External supervisor pins")
    launch, terminal = read(args.launch), read(args.terminal)
    command = launch["command"]
    expected_suffix = ["run", "--preparation", str((BASE / "cost-preparation-01").resolve()),
                       "--preparation-sha256", COST_PIN, "--out", str(run)]
    need(len(command) == 9 and command[2:] == expected_suffix
         and (ROOT / command[1]).resolve() == (ROOT / "scripts/qualify_dialogue_observation_batches.py").resolve()
         and (ROOT / command[0]).resolve() == (ROOT / ".venv/bin/python").resolve(), "Supervised exact worker command")
    need(launch["cap_seconds"] == 300 and launch["watchdog_sha256"] == WATCHDOG
         and integer(launch["pid"]) and launch["pid"] > 0 and launch["pid"] == launch["pgid"] == terminal["pgid"],
         "Bounded isolated process identity")
    need(terminal["command"] == command and terminal["returncode"] == 0 and terminal["group_absent"] is True
         and terminal["timed_out"] is False and terminal["error"] is None, "Successful recorded terminal cleanup")
    elapsed(terminal["wall_seconds"], 300)
    elapsed(done["wall_seconds"], terminal["wall_seconds"])
    need(finite(launch["started_unix"], True) and finite(terminal["finished_unix"], True)
         and 0 <= terminal["finished_unix"] - launch["started_unix"] <= terminal["wall_seconds"] + 1,
         "Supervisor chronological bounds")
    started = read(run / "started.json")
    need(started == {"version": VERSION, "phase": "model-pilot", "config": CONFIG, "caps": CAPS,
                     "runtime": plan["runtime"], "request": {"preparation": str((BASE / "cost-preparation-01").resolve()),
                     "preparation_sha256": COST_PIN}}, "Execution request/config/runtime")
    summary = read(run / "summary.json")
    need(summary["version"] == VERSION and summary["status"] == "qualified"
         and summary["official_targets_used"] is False and summary["official_test_opened"] is False,
         "Completed synthetic qualification scope")
    need(integer(summary["sampled_mps_driver_max_bytes"])
         and 0 < summary["sampled_mps_driver_max_bytes"] <= CAPS["mps_driver_bytes"]
         and integer(summary["sampled_mps_current_max_bytes"])
         and 0 < summary["sampled_mps_current_max_bytes"] <= summary["sampled_mps_driver_max_bytes"],
         "Sampled device memory bounds")
    journal = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    need(len(summary["cells"]) == 40 and len(journal) == 76, "Complete cells and update/eval events")
    counters, total_work, total_audits = Counter(), Counter(), []
    measured, journal_cursor, initial = [], 0, None
    parity_max, cell_seconds, component_seconds = 0., [], []
    checkpoint_by_arm = {c["arm"]: c for c in summary["checkpoints"]}
    need(len(checkpoint_by_arm) == len(summary["checkpoints"]) == 4 and set(checkpoint_by_arm) == set(ARMS),
         "One checkpoint per arm")
    for arm, c in checkpoint_by_arm.items():
        name = f"checkpoint-{arm}.pt"
        need(c["file"] == name and {k: c[k] for k in ("sha256", "bytes")} == done["files"][name],
             "Checkpoint descriptor matches hashed opaque payload")
        elapsed(c["wall_seconds"])
        counters["checkpoints"] += 1
    for case in cases:
        keys = [(case["split"], did) for did in case["dialogue_ids"]]
        event_work = {k: sum(profiles[x][k] for x in keys) for k in ENCODER}
        turns = sum(profiles[x]["public_user_turns"] for x in keys)
        slots = sum(profiles[x]["real_question_updates"] for x in keys)
        endpoints = sum(layouts[x]["scored_rows"] for x in keys)
        for arm in ARMS:
            cell = summary["cells"][counters["cells"]]
            need(all(cell[k] == case[k] for k in ("case_id", "kind", "split", "dialogue_ids"))
                 and cell["arm"] == arm, "Exact ordered case/arm membership")
            ci = cell["initial_sha256"]
            need(set(ci) == {"encoder", "memory"} and all(digest(v) for v in ci.values()), "Full initialization digests")
            initial = ci if initial is None else initial
            need(ci == initial, "All forty cells exactly paired at initialization")
            need(digest(cell["final_encoder_sha256"])
                 and (cell["final_encoder_sha256"] != ci["encoder"]) == (arm.startswith("trainable_") and case["kind"] != "eval"),
                 "Frozen/trainable and evaluation final encoder identity")
            need(len(cell["parity"]) == len(keys) and len(cell["events"]) == case["iterations"], "Complete per-cell visits")
            parts, measured_times = [], []
            for key, parity in zip(keys, cell["parity"], strict=True):
                need(parity["dialogue_id"] == key[1] and finite(parity["max_abs"])
                     and parity["max_abs"] <= 2e-5, "All selected vector-parity witnesses")
                expected_work = {k: profiles[key][k] for k in ENCODER}
                need(parity["encoder_work"] == expected_work, "Per-dialogue parity encoder work")
                total_work.update(expected_work)
                counters["parity_dialogue_visits"] += 1
                parity_max = max(parity_max, parity["max_abs"])
                parts.append(elapsed(parity["wall_seconds"]))
            for iteration, event in enumerate(cell["events"]):
                phase = "eval" if case["kind"] == "eval" else "update"
                need(event["case_id"] == case["case_id"] and event["arm"] == arm
                     and event["iteration"] == iteration and event["phase"] == phase
                     and event["warm"] is (iteration == 0 and case["kind"] in ("batch", "eval"))
                     and event["dialogue_visits"] == len(keys) and event["batch_endpoint_count"] == endpoints,
                     "Event identity/warmup/complete shared denominator")
                need({k: v for k, v in event.items() if k != "journal_write_seconds"} == journal[journal_cursor],
                     "Journal equals cell event before separately measured journal write")
                journal_cursor += 1
                need(event["encoder_work"] == event_work, "Every timed visit's encoder work")
                total_work.update(event_work)
                check_invariants(event["invariants"], len(keys), turns, slots)
                total_audits.append(event["invariants"])
                parts.extend((elapsed(event["wall_seconds"]), elapsed(event["journal_write_seconds"])))
                if not event["warm"]:
                    measured_times.append(event["wall_seconds"])
                if phase == "update":
                    need(finite(event["memory_gradient_squared_norm"], True)
                         and finite(event["gradient_norm_before_clip"], True)
                         and finite(event["synthetic_weighted_loss"]), "Finite loss and nonzero memory/clip witnesses")
                    gradients = event["encoder_gradient_squared_norms"]
                    need(set(gradients) == {"embeddings.word_embeddings.weight", "encoder.layer.0.attention.self.query.weight"},
                         "Both intended encoder gradient witnesses")
                    need(all(finite(v, True) if arm.startswith("trainable_") else v is None for v in gradients.values()),
                         "Trainable reaches encoder; frozen gradients absent")
                    counters["optimizer_updates"] += 1
                    counters["training_dialogue_visits"] += len(keys)
                    counters["backward_calls"] += len(keys)
                else:
                    need(not any(k in event for k in ("memory_gradient_squared_norm", "encoder_gradient_squared_norms",
                                                     "gradient_norm_before_clip", "synthetic_weighted_loss")),
                         "Evaluation carries no loss/backward/optimizer witness")
                    counters["evaluation_forwards"] += len(keys)
                counters["memory_forwards"] += len(keys)
            if case["case_id"] == "batch-0":
                parts.append(checkpoint_by_arm[arm]["wall_seconds"])
            wall = elapsed(cell["cell_wall_seconds"])
            elapsed(math.fsum(parts), wall)
            cell_seconds.append(wall)
            component_seconds.append(math.fsum(parts))
            measured.append({"case_id": case["case_id"], "arm": arm, "kind": case["kind"],
                             "dialogues_per_event": len(keys), "events": case["iterations"],
                             "batch_endpoint_count": endpoints, "encoder_work_per_event": event_work,
                             "state_checks_per_event_per_kind": slots,
                             "measured_event_count": len(measured_times), "measured_event_seconds": measured_times,
                             "measured_event_median_seconds": statistics.median(measured_times),
                             "cell_wall_seconds": wall, "accounted_component_seconds": math.fsum(parts),
                             "other_cell_seconds": wall - math.fsum(parts)})
            counters["cells"] += 1
    counters["encoding_passes"] = counters["memory_forwards"] + counters["parity_dialogue_visits"]
    need(dict(counters) == EXPECTED, "Independent complete schedule counts")
    for attempted, returned in (("encoding_attempts", "encoding_passes"), ("memory_forward_attempts", "memory_forwards"),
                                ("backward_attempts", "backward_calls"), ("optimizer_attempts", "optimizer_updates")):
        counters[attempted] = counters[returned]
    counters["encoder_forward_attempts"] = counters["encoder_forward_returns"] = total_work["encoder_calls"]
    need(journal_cursor == len(journal) and summary["counts"] == done["counts"] == dict(counters), "All actual dispatch/return counts")
    need(summary["all_encoder_work"] == dict(total_work) and summary["invariants"] == folded_invariants(total_audits),
         "Independent complete work and invariant fold")
    elapsed(math.fsum(cell_seconds), done["wall_seconds"])
    return {"counts": dict(counters), "all_encoder_work": dict(total_work), "invariants": folded_invariants(total_audits),
            "cells": measured, "maximum_initial_vector_error": parity_max, "initial_sha256": initial,
            "source_files_verified": 44, "payload_files_verified": len(names), "checkpoint_payloads_hashed_only": 4,
            "journal_events_verified": len(journal), "full_epoch_batches_reconstructed": 3840,
            "execution_wall_seconds": done["wall_seconds"], "supervisor_wall_seconds": terminal["wall_seconds"],
            "sum_cell_seconds": math.fsum(cell_seconds), "sum_recorded_component_seconds": math.fsum(component_seconds),
            "outside_cells_seconds": done["wall_seconds"] - math.fsum(cell_seconds),
            "execution_peak_rss_bytes": done["peak_rss_bytes"], "execution_output_bytes": output_bytes,
            "sampled_mps_driver_max_bytes": summary["sampled_mps_driver_max_bytes"],
            "sampled_mps_current_max_bytes": summary["sampled_mps_current_max_bytes"],
            "recorded_process_group_absent": terminal["group_absent"], "recorded_returncode": terminal["returncode"],
            "execution_caps": CAPS, "scope": [
                "Independently reconstructs saved schedule, work, denominators, monitor coverage, pairing and timing arithmetic.",
                "Full token/chunk/profile reconstruction and semantic isolation inherit pinned preparation-audit-02; actor and target files are only byte-hashed here.",
                "Encoder assets, runtime execution, gradient values, numerical vector/state checks and clock/RSS/device samples remain producer witnesses, not rerun measurements.",
                "Only checkpoint hashes and byte sizes are checked. No tensor deserialization, model, tokenizer, task score, labels or original public text is decoded.",
                "Process absence is authenticated from the parent terminal receipt, not a fresh process lookup. Sampled MPS allocations do not prove a transient peak and are not added to RSS.",
                "Two measured updates exist only for full-batch cells; single extremes/tail and evaluation have one measured event. These are synthetic costs, not training efficacy or automatic campaign admission."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "terminal", "launch", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("completed", "terminal", "launch"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    identity = {"execution_completed_sha256": args.completed_sha256, "cost_preparation_completed_sha256": COST_PIN,
                "cost_plan_sha256": COST_PLAN, "prepared_completed_sha256": PREP_PIN, "prepared_plan_sha256": PREP_PLAN,
                "prepared_audit_sha256": PREP_AUDIT, "launch_sha256": args.launch_sha256,
                "terminal_sha256": args.terminal_sha256, "script_sha256": sha(Path(__file__))}

    def check():
        need(time.perf_counter() - start <= 60, "Audit wall cap")
        multiplier = 1 if sys.platform == "darwin" else 1024
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier <= AUDIT_CAPS["rss_bytes"], "Audit RSS cap")
        need(sum(p.stat().st_size for p in args.out.iterdir()) <= AUDIT_CAPS["output_bytes"], "Audit output cap")

    def timeout(*_):
        raise TimeoutError("Audit wall cap")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        write(args.out / "started.json", {"request": request, "caps": AUDIT_CAPS, **identity})
        result = run_audit(args)
        check()
        write(args.out / "summary.json", result)
        check()
        write(args.out / "receipt.json", {"status": "completed", "agreement": True, **identity,
              "files": {name: {"sha256": sha(args.out / name), "bytes": (args.out / name).stat().st_size}
                        for name in ("started.json", "summary.json")}, "caps": AUDIT_CAPS,
              "wall_seconds": time.perf_counter() - start,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
              "experimental_model_calls": 0, "tokenizer_calls": 0, "task_metrics_computed": False})
        check()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"status": "failed", "agreement": False, **identity,
                  "error_type": type(error).__name__, "error": str(error), "request": request,
                  "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the first failure
            if hasattr(error, "add_note"):
                error.add_note("Failure receipt error: " + repr(secondary))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps({"status": "completed", "agreement": True, "receipt_sha256": sha(args.out / "receipt.json")}))


if __name__ == "__main__":
    main()
