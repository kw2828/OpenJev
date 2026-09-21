"""Synthetic preparation, replay references and lifecycle delegation only."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.dialogue_finetune_dataset import build_actor_payload, build_loss_rows
from openjev.research.dialogue_finetune_inputs import workload_profile
from openjev.research.dialogue_state_data import public_dialogue

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("calibration_preparation", SCRIPTS / "prepare_dialogue_calibration.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "calibration_preparation_original_fixture", Path(__file__).with_name("test_dialogue_finetune_dataset.py")
)
original = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(original)


class Budget:
    def __init__(self):
        self.checks = 0
        self.storage_checks = 0

    def check(self):
        self.checks += 1

    def storage(self):
        self.storage_checks += 1


class FakeTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 101, 102, 0

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": original.tokenizer(text)}


def dev_fixture():
    actors, public, rows, profiles = [], [], [], []
    # First is also the global largest. The remaining maxima tie, so ID breaks
    # the tie rather than encounter order or any scored outcome.
    for did, attention in (("dev-first", 1000), ("dev-z", 500), ("dev-a", 500)):
        packet, record, catalog, queries, layout = original.fixture("dev")
        packet["id"] = record["dialogue_id"] = layout["id"] = did
        record["turns"][0]["utterance"] += " " + did
        payload = build_actor_payload(packet, record, catalog, queries, layout, original.tokenizer, split="dev")
        actor = {key: value for key, value in payload.items() if key != "lexical"}
        actor.update(lexical_shape=layout["shape"], lexical_offset=layout["offset"])
        actors.append(actor)
        public.append(record)
        work = workload_profile(payload)
        work["padded_attention_positions"] = attention
        profiles.append({"split": "dev", "dialogue_id": did, "work": work})
        for row in build_loss_rows(packet, payload, queries):
            query = queries[row["query_index"]]
            rows.append({**row, "row_index": len(rows), "candidate_ids": query["candidate_ids"],
                         "candidate_values": query["candidate_values"]})
    return actors, public, {"profiles": profiles}, rows


def test_fixed_real_membership_counts_and_explicit_cli_lifecycle(monkeypatch, tmp_path):
    assert (prep.FITTED_DIALOGUES, prep.DEV_DIALOGUES, prep.DEV_ENDPOINTS) == (2017, 2363, 62329)
    called = []

    def execute(args, body):
        called.append((args, body))
        return "delegated-without-body"

    monkeypatch.setattr(prep.common, "execute", execute)
    result = prep.main(["--plan", str(tmp_path / "plan.json"), "--plan-sha256", "pinned",
                        "--supervision", str(tmp_path / "launch.json"), "--out", str(tmp_path / "out")])
    assert result == "delegated-without-body"
    args, body = called[0]
    assert body is prep.body and args.command == "prepare" and args.plan_sha256 == "pinned"
    assert args.plan == tmp_path / "plan.json" and args.supervision == tmp_path / "launch.json"
    assert not args.out.exists()


def test_public_dev_join_uses_only_evaluated_membership_and_original_order():
    actors, public, _, _ = dev_fixture()
    extra = public_dialogue({"dialogue_id": "not-evaluated", "turns": [{"speaker": "USER", "utterance": "other"}]})
    result = prep.evaluated_public(actors, [extra, *reversed(public)], expected_count=3)
    assert [r["dialogue_id"] for r in result] == [a["dialogue_id"] for a in actors]
    assert all(set(r) == {"dialogue_id", "turns", "user_turns"} for r in result)
    with pytest.raises(ValueError, match="Complete original evaluated DEV"):
        prep.evaluated_public(actors, public)


@pytest.mark.parametrize("defect", ["missing_public", "duplicate_public", "duplicate_actor", "relabelled_train"])
def test_dev_membership_join_fails_closed(defect):
    actors, public, _, _ = dev_fixture()
    if defect == "missing_public":
        public.pop()
    elif defect == "duplicate_public":
        public.append(public[0])
    elif defect == "duplicate_actor":
        actors.append(actors[0])
    else:
        actors[0]["split"] = "train"
    with pytest.raises(ValueError):
        prep.evaluated_public(actors, public, expected_count=3)


def test_replay_uses_first_then_largest_other_tie_id_and_preserves_original_row_indices():
    actors, _, workloads, rows = dev_fixture()
    result = prep.replay_cases(actors, workloads, rows, expected_count=3, expected_rows=12)
    assert [c["dialogue_id"] for c in result["cases"]] == ["dev-first", "dev-a"]
    assert [c["row_indices"] for c in result["cases"]] == [list(range(4)), list(range(8, 12))]
    for case in result["cases"]:
        for endpoint in case["endpoints"]:
            row = rows[endpoint["row_index"]]
            assert all(row[key] == value for key, value in endpoint.items())
            assert not {"label_index", "label_id", "unseen", "bin", "dontcare"} & set(endpoint)
    assert result["source_split"] == "dev" and result["analysis_role"] == "numerical_replay_qualification"
    # No score or target field participates in replay selection or references.
    for row in rows:
        row["label_index"] = object()
        row["label_id"] = object()
    assert result == prep.replay_cases(actors, workloads, rows, expected_count=3, expected_rows=12)


@pytest.mark.parametrize("defect", ["missing_profile", "duplicate_profile", "nonpositive_work", "missing_row",
                                    "wrong_index", "candidate_order", "duplicate_endpoint", "wrong_query", "wrong_turn"])
def test_replay_rejects_incomplete_or_misaligned_metadata(defect):
    actors, _, workloads, rows = dev_fixture()
    if defect == "missing_profile":
        workloads["profiles"].pop()
    elif defect == "duplicate_profile":
        workloads["profiles"].append(workloads["profiles"][0])
    elif defect == "nonpositive_work":
        workloads["profiles"][0]["work"]["encoder_calls"] = 0
    elif defect == "missing_row":
        rows.pop()
    elif defect == "wrong_index":
        rows[0]["row_index"] = 2
    elif defect == "candidate_order":
        rows[0]["candidate_ids"] = list(reversed(rows[0]["candidate_ids"]))
    elif defect == "duplicate_endpoint":
        rows[1] = {**rows[0], "row_index": 1}
    elif defect == "wrong_query":
        rows[0]["query_index"] = 999
    else:
        rows[0]["turn_index"] = 999
    with pytest.raises(ValueError):
        prep.replay_cases(actors, workloads, rows, expected_count=3, expected_rows=12)


def test_service_exposure_reports_overlap_without_relabelling():
    calibration = [{"service": "shared", "query_id": "shared-query", "dialogue_id": "cal"},
                   {"service": "train-only", "query_id": "train-query", "dialogue_id": "cal"}]
    dev = [{"service": "shared", "query_id": "shared-query", "unseen": True},
           {"service": "dev-only", "query_id": "dev-query", "unseen": True},
           {"service": "train-only", "query_id": "train-query", "unseen": False}]
    before = copy.deepcopy((calibration, dev))
    exposure = prep.service_exposure(calibration, dev)
    assert exposure["calibration_overlap_with_dev_unseen_services"] == ["shared"]
    assert exposure["calibration_overlap_with_dev_unseen_queries"] == ["shared-query"]
    assert exposure["counts"]["calibration_endpoints_in_dev_unseen_services"] == 1
    assert exposure["counts"]["calibration_dialogues_in_dev_unseen_services"] == 1
    assert exposure["counts"]["dev_unseen_endpoints_in_calibration_services"] == 1
    assert (calibration, dev) == before
    with pytest.raises(ValueError, match="service exposure is consistent"):
        prep.service_exposure(calibration, [*dev, {"service": "shared", "query_id": "x", "unseen": False}])


@pytest.fixture
def body_fixture(monkeypatch, tmp_path):
    packet, template, catalog, entries, _ = original.fixture()
    train, labels = [], []
    for index in range(513):
        did = "fitted-only" if index == 512 else f"calibration-{index:04d}"
        record = copy.deepcopy(template)
        record["dialogue_id"] = did
        record["turns"][0]["utterance"] += " distinct " + did
        train.append(record)
        for row in packet["queries"]:
            query = entries[row["query"]]
            labels.append({"dialogue_id": did, "turn_index": record["user_turns"][row["time"]]["turn_index"],
                           "query_id": query["id"], "service": query["service"], "slot": query["slot"],
                           "label_index": row["label"], "label_id": query["candidate_ids"][row["label"]],
                           "bin": row["bin"], "is_dontcare": row["dontcare"], "unseen_service": False})
    dev_actors, dev_public, workloads, old_rows = dev_fixture()
    source, prepared, scientific = tmp_path / "runs/sgd-state-v1/data", tmp_path / "old-prepared", tmp_path / "scientific"
    for directory in (source, prepared, scientific):
        directory.mkdir(parents=True)
    files = {source / "train-dialogues.jsonl": train, source / "train-labels.jsonl": labels,
             source / "dev-dialogues.jsonl": dev_public, prepared / "actors-dev.jsonl": dev_actors,
             scientific / "evaluation-rows.jsonl": old_rows}
    for path, rows in files.items():
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for path, value in ((source / "catalog.json", {"train": catalog}),
                        (prepared / "orders.json", {"dialogue_ids": ["fitted-only"]}),
                        (prepared / "workloads.json", workloads)):
        path.write_text(json.dumps(value))
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "FITTED_DIALOGUES", 1)
    monkeypatch.setattr(prep, "DEV_DIALOGUES", 3)
    monkeypatch.setattr(prep, "DEV_ENDPOINTS", 12)
    loads = []

    def tokenizer(snapshot):
        loads.append(snapshot)
        return FakeTokenizer()

    monkeypatch.setattr(prep, "load_tokenizer", tokenizer)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Preparation must not decode arrays or weights"))
    out = tmp_path / "prepared"
    out.mkdir()
    ctx = {"original_prepared": prepared, "prior": SimpleNamespace(run=scientific),
           "parent": {"snapshot": "synthetic-local-tokenizer", "tokenizer_ids": {"cls_id": 101, "sep_id": 102, "pad_id": 0}},
           "science": {}, "plan": {}, "newplanhash": "synthetic"}
    return SimpleNamespace(out=out), ctx, Budget(), loads, files


def test_body_builds_complete_separate_public_actors_and_canonical_evaluator_outputs(body_fixture):
    args, ctx, budget, loads, inputs = body_fixture
    input_before = {path: path.read_bytes() for path in inputs}
    result = prep.body(args, ctx, budget)
    assert loads == ["synthetic-local-tokenizer"]
    assert result["counts"]["dialogues"] == 512 and result["counts"]["scored_endpoints"] == 2048
    assert result["counts"]["public_user_turns"] == 2048
    assert result["model_calls"] == result["encoder_calls"] == result["neural_calls"] == 0
    assert result["tokenizer_only"] is True and result["model_weights_loaded"] is False
    assert result["official_test_opened"] is result["official_dev_inference"] is False
    assert result["original_study_continuation_passed"] is False
    assert result["source_split"] == "train" and result["analysis_role"] == "calibration"
    assert result["replay_dialogue_ids"] == ["dev-first", "dev-a"]
    assert budget.checks > 512 and budget.storage_checks == 10
    assert {path.name for path in args.out.iterdir()} == {
        "actors.jsonl", "targets.jsonl", "packet.json", "selection.json", "evaluation-rows.jsonl",
        "lexical-original.npy", "lexical-numbers.npy", "workloads.json", "replay-cases.json", "service-exposure.json"}
    actors = prep.read_lines(args.out / "actors.jsonl", budget)
    targets = prep.read_lines(args.out / "targets.jsonl", budget)
    rows = prep.read_lines(args.out / "evaluation-rows.jsonl", budget)
    selection = prep.read(args.out / "selection.json")
    packet, work = prep.read(args.out / "packet.json"), prep.read(args.out / "workloads.json")
    assert selection["counts"]["selected_unique_groups"] == 512
    assert [actor["dialogue_id"] for actor in actors] == selection["selected_ids"]
    assert work["all"]["dialogues"] == 512 and set(work["splits"]) == {"train"}
    assert len(work["profiles"]) == len(packet["layouts"]) == len(packet["cohort"]) == 512
    assert [row["row_index"] for row in rows] == list(range(2048))
    assert all(row["source_split"] == row["split"] == "train" and row["analysis_role"] == "calibration"
               and row["unseen"] is False for row in rows)
    assert all(not {"label", "label_index", "label_id", "bin", "stratum", "unseen", "rows"} & set(a) for a in actors)
    for actor, target in zip(actors, targets, strict=True):
        assert actor["dialogue_id"] == target["dialogue_id"]
        assert len(actor["user_turn_indices"]) == 4 and {row["time"] for row in target["rows"]} == {0, 1, 3}
        for feature, tokens in zip(actor["original_feature_ids"], actor["tokens"], strict=True):
            assert tokens == original.tokenizer(packet["texts"][feature])
    for row in rows:
        query = packet["queries"][row["query_index"]]
        assert row["candidate_ids"] == query["candidate_ids"] and row["candidate_values"] == query["candidate_values"]
        assert row["candidate_ids"][row["label_index"]] == row["label_id"]
    assert all(path.read_bytes() == value for path, value in input_before.items())


def test_incomplete_fitted_membership_fails_before_loading_tokenizer(body_fixture, monkeypatch):
    args, ctx, budget, loads, _ = body_fixture
    monkeypatch.setattr(prep, "FITTED_DIALOGUES", 2)
    with pytest.raises(ValueError, match="actual fitted dialogue IDs"):
        prep.body(args, ctx, budget)
    assert not loads and not list(args.out.iterdir())


def test_tokenizer_identity_mismatch_cannot_write_preparation(body_fixture):
    args, ctx, budget, _, _ = body_fixture
    ctx["parent"]["tokenizer_ids"]["sep_id"] = 999
    with pytest.raises(ValueError, match="tokenizer special-token identities"):
        prep.body(args, ctx, budget)
    assert not list(args.out.iterdir())


def test_budget_failure_precedes_input_decode(body_fixture, monkeypatch):
    args, ctx, _, _, _ = body_fixture
    monkeypatch.setattr(prep, "read_lines", lambda *a: pytest.fail("Decode after exhausted budget"))

    def stop():
        raise TimeoutError("synthetic cap")

    with pytest.raises(TimeoutError, match="synthetic cap"):
        prep.body(args, ctx, SimpleNamespace(check=stop))
    assert not list(args.out.iterdir())


def test_blank_jsonl_line_is_rejected(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text('{}\n\n')
    with pytest.raises(ValueError, match="No empty JSONL"):
        prep.read_lines(path, Budget())
