"""Boundary tests for the public-stream/evaluator join, with synthetic data."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_history", Path(__file__).resolve().parents[1] / "scripts/audit_dialogue_history_support.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    public = {"dialogue_id": "d", "turns": [
        {"speaker": "USER", "utterance": "first"},
        {"speaker": "SYSTEM", "utterance": "proposal"},
        {"speaker": "USER", "utterance": "unscored"},
        {"speaker": "USER", "utterance": "third"}],
        "user_turns": [{"turn_index": 0}, {"turn_index": 2}, {"turn_index": 3}]}
    queries = [{"id": '["s","q"]', "service": "s", "slot": "q",
                "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:red"],
                "candidate_values": [None, None, "red"]}]
    dialogue = {"id": "d", "turns": [10, 11, 12], "user_text": ["first", "unscored", "third"],
                "queries": [{"query": 0, "time": 2, "label": 2, "bin": "first_assignment"}]}
    return dialogue, public, queries


def test_full_stream_time_join_keeps_unscored_gap():
    d, p, q = fixture()
    query, labels = runner.selected_labels(d, p, q)[0]
    assert labels[0]["turn_index"] == 3
    assert query["candidates"][2]["id"] == labels[0]["label_id"] == "value:red"
    assert len(p["user_turns"]) == 3


@pytest.mark.parametrize("field,value", [("time", -1), ("time", 3), ("query", -1),
                                         ("query", True), ("label", 3)])
def test_invalid_packet_indices_rejected(field, value):
    d, p, q = fixture()
    d["queries"][0][field] = value
    with pytest.raises(ValueError):
        runner.selected_labels(d, p, q)


def test_misaligned_public_text_is_rejected():
    d, p, q = fixture()
    p["turns"][2]["utterance"] = "changed"
    with pytest.raises(ValueError, match="alignment"):
        runner.selected_labels(d, p, q)


def test_query_identity_and_candidate_count_are_bound():
    d, p, q = fixture()
    q[0]["slot"] = "other"
    with pytest.raises(ValueError, match="identity"):
        runner.selected_labels(d, p, q)
    d, p, q = fixture()
    q[0]["candidate_values"].pop()
    with pytest.raises(ValueError):
        runner.selected_labels(d, p, q)


def test_join_does_not_mutate_actor_packet():
    d, p, q = fixture()
    before = copy.deepcopy((d, p, q))
    runner.selected_labels(d, p, q)
    assert (d, p, q) == before


def test_authentication_rejects_changed_selected_input(tmp_path):
    f = tmp_path / "input.json"
    f.write_text("{}")
    expected = {f.name: runner.sha(f)}
    runner.authenticate(tmp_path, expected)
    f.write_text('{"altered":true}')
    with pytest.raises(ValueError, match="Hash mismatch"):
        runner.authenticate(tmp_path, expected)


def synthetic_run_inputs(tmp_path, monkeypatch):
    d, p, q = fixture()
    p["user_turns"] = [{"turn_index": 0, "previous_system_turn_index": None},
                       {"turn_index": 2, "previous_system_turn_index": 1},
                       {"turn_index": 3, "previous_system_turn_index": None}]
    payloads = {
        "runs/sgd-state-v1/features-02/packet.json":
            {"queries": q, "cohorts": {"train": [d], "dev": []}},
        "runs/sgd-state-v1/data/train-dialogues.jsonl": p,
        "runs/sgd-state-v1/features-02/completed.json": {"status": "completed"},
        "runs/sgd-state-v1/data/completed.json": {"status": "completed"},
    }
    for name, value in payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCES", ())
    monkeypatch.setattr(runner, "INPUTS", {n: runner.sha(tmp_path / n) for n in payloads})
    monkeypatch.setattr(runner, "CONFIG", {**runner.CONFIG, "dialogues": 1, "rows": 1})
    plan_dir = tmp_path / "plan"
    runner.freeze(plan_dir)
    return plan_dir / "plan.json"


def test_complete_synthetic_run_retains_all_public_steps(tmp_path, monkeypatch):
    plan = synthetic_run_inputs(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    runner.run(plan, runner.sha(plan), out)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cohort"]["rows"] == 1
    assert summary["cohort"]["available_public_query_steps"] == 3
    assert summary["cohort"]["consecutive_user_pairs"] == 1
    with gzip.open(out / "endpoint-metadata.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1 and rows[0]["user_index"] == 2
    receipt = json.loads((out / "completed.json").read_text())
    assert receipt["model_calls"] == 0 and not (out / "failed.json").exists()
    for name, entry in receipt["files"].items():
        assert runner.sha(out / name) == entry["sha256"]


def test_wrong_plan_pin_closes_failure_before_quality(tmp_path, monkeypatch):
    plan = synthetic_run_inputs(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    with pytest.raises(ValueError, match="Plan hash"):
        runner.run(plan, "0" * 64, out)
    failure = json.loads((out / "failed.json").read_text())
    assert failure["completed_dialogues"] == 0
    assert {p.name for p in out.iterdir()} == {"started.json", "failed.json"}


def test_final_cap_failure_demotes_completion(tmp_path, monkeypatch):
    plan = synthetic_run_inputs(tmp_path, monkeypatch)
    original_rss = runner.rss
    out = tmp_path / "audit"

    def rss_after_close():
        return runner.CONFIG["rss_bytes"] + 1 if (out / "completed.json").exists() else original_rss()

    monkeypatch.setattr(runner, "rss", rss_after_close)
    with pytest.raises(ValueError, match="RSS cap"):
        runner.run(plan, runner.sha(plan), out)
    assert not (out / "completed.json").exists()
    assert (out / "incomplete-completion.json").exists()
    assert json.loads((out / "failed.json").read_text())["status"] == "failed"
