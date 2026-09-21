"""Authenticated calibration-complement preparation; tokenizer only, no inference.

The shared lifecycle authenticates every source and input before calling body.
Original TRAIN/DEV identities and evaluator row order remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dialogue_calibration_common as common
import numpy as np

from openjev.research.dialogue_calibration_data import (
    SAMPLE_SIZE,
    build_calibration_inputs,
    select_calibration_dialogues,
)
from openjev.research.dialogue_finetune_dataset import aggregate_work_profiles

ROOT, read, write, sha = common.ROOT, common.read, common.write, common.sha
require, write_lines = common.require, common.write_lines
FITTED_DIALOGUES, DEV_DIALOGUES, DEV_ENDPOINTS = 2017, 2363, 62329
VERSION = "dialogue-calibration-preparation-v1"


def read_lines(path, budget):
    rows = []
    with path.open() as stream:
        for line in stream:
            budget.check()
            require(bool(line.strip()), "No empty JSONL records")
            rows.append(json.loads(line))
    return rows


def distinct_index(records, field, message):
    result = {}
    for record in records:
        identity = record[field]
        require(type(identity) is str and identity and identity not in result, message)
        result[identity] = record
    return result


def evaluated_public(dev_actors, public, *, expected_count=DEV_DIALOGUES):
    """Join exactly the originally evaluated DEV membership to public streams."""
    actors = distinct_index(dev_actors, "dialogue_id", "Distinct original DEV actors")
    source = distinct_index(public, "dialogue_id", "Distinct public DEV identities")
    require(len(actors) == expected_count and all(a["split"] == "dev" for a in dev_actors),
            "Complete original evaluated DEV cohort")
    require(set(actors) <= set(source), "Every evaluated DEV public stream exists")
    return [source[did] for did in actors]


def canonical_rows(built, budget):
    """Separate evaluator table with fresh contiguous indices and canonical IDs."""
    actors, targets = built["actors"], built["targets"]
    require(len(actors) == len(targets) == SAMPLE_SIZE, "All 512 actor/target pairs")
    queries, rows = built["packet"]["queries"], []
    for actor, record in zip(actors, targets, strict=True):
        budget.check()
        did = actor["dialogue_id"]
        require(record["dialogue_id"] == did and actor["split"] == record["split"] == "train"
                and actor["analysis_role"] == record["analysis_role"] == "calibration"
                and actor["source_split"] == record["source_split"] == "train",
                "Paired TRAIN calibration identities")
        endpoints = set()
        for index, row in enumerate(record["rows"]):
            require(row["dialogue_id"] == did and row["split"] == "train"
                    and row["source_row_index"] == index and row["unseen"] is False,
                    "Canonical calibration source ordering and split")
            time, position = row["time"], row["query_position"]
            require((time, position) not in endpoints, "Distinct calibration endpoint")
            endpoints.add((time, position))
            qi = actor["query_ids"][position]
            query = queries[qi]
            require(row["query_index"] == qi and row["query_id"] == query["id"]
                    and (row["service"], row["slot"]) == (query["service"], query["slot"])
                    and query["split"] == "train"
                    and actor["user_turn_indices"][time] == row["turn_index"],
                    "Calibration public query and USER chronology join")
            ids, values = query["candidate_ids"], query["candidate_values"]
            require(actor["candidate_ids"][position] == ids and 3 <= len(ids) <= 12
                    and len(ids) == len(values) and ids[row["label_index"]] == row["label_id"],
                    "Canonical calibration candidate and target support")
            rows.append({**row, "row_index": len(rows), "candidate_ids": list(ids),
                         "candidate_values": list(values), "source_split": "train",
                         "analysis_role": "calibration"})
    require(len(rows) == built["counts"]["scored_endpoints"], "All calibration endpoints")
    return rows


def replay_cases(dev_actors, workloads, old_rows, *, expected_count=DEV_DIALOGUES,
                 expected_rows=DEV_ENDPOINTS):
    """First original actor plus the greatest attention workload among the rest.

    Selection uses only fixed public workload geometry. Endpoint references do
    not include target values, bins, predictions or chosen model scores.
    """
    actors = distinct_index(dev_actors, "dialogue_id", "Distinct replay DEV actors")
    require(len(actors) == expected_count and expected_count >= 2
            and all(a["split"] == "dev" for a in dev_actors), "Complete replay DEV cohort")
    dev_profiles = [p for p in workloads["profiles"] if p["split"] == "dev"]
    profiles = distinct_index(dev_profiles, "dialogue_id", "Distinct DEV workload identities")
    require(set(profiles) == set(actors), "Complete original DEV workload membership")
    for profile in profiles.values():
        for key in ("padded_attention_positions", "encoder_calls", "real_question_updates"):
            value = profile["work"][key]
            require(type(value) is int and value > 0, "Positive original replay work geometry")
    first = dev_actors[0]["dialogue_id"]
    largest = min((did for did in actors if did != first),
                  key=lambda did: (-profiles[did]["work"]["padded_attention_positions"], did))
    references = {did: [] for did in (first, largest)}
    seen, candidate_maps, endpoints = set(), {}, set()
    require(len(old_rows) == expected_rows, "Complete original DEV endpoint table")
    for index, row in enumerate(old_rows):
        did = row["dialogue_id"]
        require(row["row_index"] == index and row["split"] == "dev" and did in actors,
                "Canonical original DEV row index and membership")
        seen.add(did)
        actor, position, time = actors[did], row["query_position"], row["time"]
        endpoint = (did, time, position)
        require(endpoint not in endpoints, "Distinct original DEV endpoint")
        endpoints.add(endpoint)
        require(type(position) is int and 0 <= position < len(actor["query_ids"])
                and type(time) is int and 0 <= time < len(actor["user_turn_indices"])
                and row["query_index"] == actor["query_ids"][position]
                and row["turn_index"] == actor["user_turn_indices"][time],
                "Original DEV public query and USER chronology join")
        ids, values = row["candidate_ids"], row["candidate_values"]
        require(ids == actor["candidate_ids"][position] and 3 <= len(ids) <= 12
                and len(values) == len(ids), "Original DEV candidate support and order")
        identity = (row["query_id"], ids, values)
        key = (did, position)
        require(key not in candidate_maps or candidate_maps[key] == identity,
                "Stable original DEV query candidate identity")
        candidate_maps[key] = identity
        if did in references:
            references[did].append({key: row[key] for key in (
                "row_index", "source_row_index", "time", "turn_index", "query_position",
                "query_index", "query_id", "service", "slot", "candidate_ids", "candidate_values")})
    require(seen == set(actors), "Every original DEV dialogue has evaluation endpoints")
    return {
        "selection_rule": "First original DEV actor; maximum attention positions excluding first, ties by ID",
        "source_split": "dev", "analysis_role": "numerical_replay_qualification",
        "cases": [{"dialogue_id": did, "reason": reason, "work": dict(profiles[did]["work"]),
                   "row_indices": [r["row_index"] for r in references[did]], "endpoints": references[did]}
                  for did, reason in ((first, "first_original_actor"), (largest, "largest_remaining_attention"))],
    }


def service_exposure(calibration_rows, old_rows):
    """Describe schema exposure only; never reclassify original DEV panels."""
    services = {row["service"] for row in calibration_rows}
    queries = {row["query_id"] for row in calibration_rows}
    panels = {}
    for row in old_rows:
        service, unseen = row["service"], row["unseen"]
        require(type(unseen) is bool and (service not in panels or panels[service] == unseen),
                "Original DEV service exposure is consistent")
        panels[service] = unseen
    unseen = {service for service, flag in panels.items() if flag}
    seen = set(panels) - unseen
    unseen_queries = {row["query_id"] for row in old_rows if row["unseen"]}
    overlap = services & unseen
    return {
        "calibration_train_services": sorted(services), "calibration_train_queries": sorted(queries),
        "original_dev_seen_services": sorted(seen), "original_dev_unseen_services": sorted(unseen),
        "original_dev_unseen_queries": sorted(unseen_queries),
        "calibration_overlap_with_dev_seen_services": sorted(services & seen),
        "calibration_overlap_with_dev_unseen_services": sorted(overlap),
        "calibration_overlap_with_dev_unseen_queries": sorted(queries & unseen_queries),
        "counts": {"calibration_train_services": len(services), "calibration_train_queries": len(queries),
                   "original_dev_seen_services": len(seen), "original_dev_unseen_services": len(unseen),
                   "overlap_with_dev_seen_services": len(services & seen),
                   "overlap_with_dev_unseen_services": len(overlap),
                   "overlap_with_dev_unseen_queries": len(queries & unseen_queries),
                   "calibration_endpoints_in_dev_unseen_services": sum(r["service"] in overlap for r in calibration_rows),
                   "calibration_dialogues_in_dev_unseen_services": len({r["dialogue_id"] for r in calibration_rows
                                                                         if r["service"] in overlap}),
                   "dev_unseen_endpoints_in_calibration_services": sum(r["unseen"] and r["service"] in services
                                                                        for r in old_rows)},
        "scope": "Original DEV seen/unseen labels retained; calibration selection never uses this exposure summary",
    }


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def body(args, ctx, budget):
    """Called only after the common lifecycle authenticates the complete closure."""
    budget.check()
    source = ROOT / "runs/sgd-state-v1/data"
    prepared = ctx["original_prepared"]
    train_public = read_lines(source / "train-dialogues.jsonl", budget)
    train_labels = read_lines(source / "train-labels.jsonl", budget)
    catalog = read(source / "catalog.json")["train"]
    orders = read(prepared / "orders.json")
    fitted_ids = orders["dialogue_ids"]
    require(type(fitted_ids) is list and len(fitted_ids) == len(set(fitted_ids)) == FITTED_DIALOGUES,
            "All 2017 actual fitted dialogue IDs")
    dev_actors = read_lines(prepared / "actors-dev.jsonl", budget)
    dev_public = evaluated_public(dev_actors, read_lines(source / "dev-dialogues.jsonl", budget),
                                  expected_count=DEV_DIALOGUES)
    budget.check()
    identity_rows = [{key: row[key] for key in ("dialogue_id", "query_id", "turn_index")}
                     for row in train_labels]
    selection = select_calibration_dialogues(train_public, identity_rows, catalog, fitted_ids, dev_public)
    budget.check()
    tokenizer = load_tokenizer(ctx["parent"]["snapshot"])
    expected_ids = ctx["parent"]["tokenizer_ids"]
    require(set(expected_ids) == {"cls_id", "sep_id", "pad_id"}
            and all(getattr(tokenizer, name.removesuffix("_id") + "_token_id") == value
                    for name, value in expected_ids.items()),
            "Original tokenizer special-token identities")
    memo = {}

    def tokenize(text):
        budget.check()
        if text not in memo:
            memo[text] = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        return list(memo[text])

    built = build_calibration_inputs(selection, train_public, train_labels, catalog, tokenize)
    budget.check()
    rows = canonical_rows(built, budget)

    def payloads():
        for actor in built["actors"]:
            budget.check()
            shape, offset = actor["lexical_shape"], actor["lexical_offset"]
            lexical = built["lexical"]["original"][offset:offset + int(np.prod(shape))].reshape(shape)
            yield {**actor, "lexical": lexical}

    workloads = aggregate_work_profiles(payloads())
    require(workloads["all"]["dialogues"] == SAMPLE_SIZE and set(workloads["splits"]) == {"train"},
            "Full 512-dialogue TRAIN geometry")
    old_rows = read_lines(ctx["prior"].run / "evaluation-rows.jsonl", budget)
    replay = replay_cases(dev_actors, read(prepared / "workloads.json"), old_rows,
                          expected_count=DEV_DIALOGUES, expected_rows=DEV_ENDPOINTS)
    replay["evaluation_rows_source"] = str(ctx["prior"].run / "evaluation-rows.jsonl")
    replay["evaluation_rows_sha256"] = sha(ctx["prior"].run / "evaluation-rows.jsonl")
    exposure = service_exposure(rows, old_rows)
    budget.check()
    outputs = {"selection.json": selection, "packet.json": {**built["packet"], "texts": built["texts"],
                                                             "layouts": built["layouts"]},
               "workloads.json": workloads, "replay-cases.json": replay, "service-exposure.json": exposure}
    for name, value in outputs.items():
        budget.check()
        write(args.out / name, value)
        budget.storage()
    for name, values in (("actors.jsonl", built["actors"]), ("targets.jsonl", built["targets"]),
                         ("evaluation-rows.jsonl", rows)):
        write_lines(args.out / name, values)
        budget.storage()
    for name, array in built["lexical"].items():
        with (args.out / f"lexical-{name}.npy").open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        budget.storage()
    budget.check()
    return {"version": VERSION, "counts": built["counts"], "selection_counts": selection["counts"],
            "source_split": "train", "analysis_role": "calibration", "neural_calls": 0,
            "model_calls": 0, "encoder_calls": 0, "model_weights_loaded": False, "tokenizer_only": True,
            "official_test_opened": False, "official_dev_inference": False,
            "fitted_dialogues_excluded": len(fitted_ids), "evaluated_dev_dialogues_excluded": len(dev_public),
            "unique_tokenized_texts": len(memo), "service_exposure": exposure,
            "replay_dialogue_ids": [c["dialogue_id"] for c in replay["cases"]],
            "work_scope": "One visit to all 512 dialogues; inference requires 12 complete visits",
            "original_study_continuation_passed": False,
            "scope": "Preparation only; no new predictions, calibrator fit or evaluation"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(command="prepare")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    return common.execute(parse_args(argv), body)


if __name__ == "__main__":
    main()
