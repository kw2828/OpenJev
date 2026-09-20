"""Artificial packets and interpretations only; no live review or evaluator reads."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/join_dialogue_qwen_reviews.py"
SPEC = importlib.util.spec_from_file_location("review_join", PATH)
j = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(j)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, diagnostic, reviews = (tmp_path/n for n in ("repo", "diagnostic", "reviews"))
    monkeypatch.setattr(j, "ROOT", root)
    for name in j.SOURCES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic source "+name)
    diagnostic.mkdir()
    reviews.mkdir()
    instructions_pin = j.item(root/j.INSTRUCTIONS)["sha256"]
    ids = ["a"*64, "b"*64]
    cases = [{"case_id": cid, "context": json.dumps({"exchanges_oldest_first": [{"SYSTEM": "Try blue?", "USER": "Keep red."}]}),
              "question": "Question\nPrevious committed value:\nred\nLexical flag order:x",
              "candidates": [{"id": "c00", "description": "red"}, {"id": "c01", "description": "blue"}]} for cid in ids]
    for arm in j.ARMS:
        j.write(diagnostic/(arm+"-review.json"), {"arm": arm, "review_instructions_sha256": instructions_pin, "cases": cases})
    ledger = [{"case_id": cid, "selection_stratum": "synthetic", "canonical_id_map": {"c00": "value:red", "c01": "value:blue"},
               "rows": {arm: {"arm": arm, "row_index": i, "previous_id": "value:red", "target_id": "value:red",
                              "selected_id": "value:blue"} for arm in j.ARMS}} for i, cid in enumerate(ids)]
    (diagnostic/"evaluator-selection.jsonl").write_text("".join(json.dumps(x)+"\n" for x in ledger))
    for name in ("started.json", "summary.json", "rows.jsonl"):
        (diagnostic/name).write_text('{}\n')
    receipt = {"status": "completed", "version": j.DIAGNOSTIC_VERSION, "model_calls": 0, "tokenizer_calls": 0,
               "checkpoint_deserializations": 0, "reviewer_calls": 0, "selected_cases": 2,
               "source_sha256": {n: j.item(root/n)["sha256"] for n in (j.INSTRUCTIONS, j.PROTOCOL)},
               "files": {n: j.item(diagnostic/n) for n in j.DIAGNOSTIC_FILES}}
    j.write(diagnostic/"receipt.json", receipt)
    pin = j.item(diagnostic/"receipt.json")["sha256"]
    entries = []
    for rid in j.REVIEW_IDS:
        arm = rid.rsplit("-", 1)[0]
        packet = diagnostic/(arm+"-review.json")
        entry = {"review_id": rid, "arm": arm, "packet_path": str(packet), "packet_sha256": j.item(packet)["sha256"],
                 "response_file": rid+".json", "terminal_file": rid+"-terminal.json"}
        entries.append(entry)
        j.write(reviews/entry["terminal_file"], {"review_id": rid, "agent_name": "/synthetic/"+rid,
                                                "status": "completed", "note": "Artificial terminal"})
        answers = [{"case_id": cid, "candidate_id": "c00", "ambiguous": i == 1,
                    "support": "Keep red.", "reason": "The USER retains the previous value.",
                    "state_definition_issue": ""} for i, cid in enumerate(ids)]
        if rid.endswith("02"):
            answers[1].update(candidate_id="c01", support="Try blue?", reason="Synthetic disagreement.")
        j.write(reviews/entry["response_file"], {"review_id": rid, "packet_sha256": entry["packet_sha256"], "cases": answers})
    j.write(reviews/"dispatch.json", {"version": j.DISPATCH_VERSION, "model": "gpt-6-astra",
            "instructions": {"path": str(root/j.INSTRUCTIONS), "sha256": instructions_pin},
            "diagnostic_receipt_sha256": pin, "reviews": entries})
    args = SimpleNamespace(diagnostic=diagnostic, diagnostic_receipt_sha256=pin, reviews=reviews, out=tmp_path/"joined")
    return SimpleNamespace(args=args, cases=cases, entries=entries, ledger=ledger)


def replace(path, update):
    value = j.read(path)
    update(value)
    path.write_text(json.dumps(value))


def test_complete_paired_join_preserves_interpretation_and_denominators(fixture):
    summary = j.execute(fixture.args)
    for arm in j.ARMS:
        result = summary["arms"][arm]
        assert (result["agree"], result["disagree"], result["paired_valid_denominator"]) == (1, 1, 2)
        assert result["reviews"][arm+"-01"]["target_agreement"] == {"numerator": 2, "denominator": 2, "rate": 1.}
        assert result["reviews"][arm+"-02"]["model_agreement"]["numerator"] == 1
        assert result["reviews"][arm+"-01"]["ambiguous_valid_answers"] == 1
    assert summary["cases"][0]["arms"]["current"]["reviews"]["current-01"]["interpretation"]["support"] == "Keep red."
    receipt = j.read(fixture.args.out/"receipt.json")
    assert receipt["model_calls"] == receipt["replacement_reviews"] == 0
    assert sum(Path(p).name.endswith("-terminal.json") for p in receipt["input_members"]) == 4
    with pytest.raises(FileExistsError):
        j.execute(fixture.args)


@pytest.mark.parametrize("terminal", ["missing", "invalid"])
def test_terminal_gate_precedes_any_response_or_evaluator_read(fixture, monkeypatch, terminal):
    path = fixture.args.reviews/"history4-02-terminal.json"
    if terminal == "missing":
        path.unlink()
    else:
        replace(path, lambda x: x.update(status="running"))
    original = Path.open
    blocked = {e["response_file"] for e in fixture.entries} | {"evaluator-selection.jsonl", "rows.jsonl"}
    def guarded(path, *args, **kwargs):
        if path.name in blocked:
            raise AssertionError("Sensitive join payload read before all terminal records")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="terminal"):
        j.execute(fixture.args)
    assert (fixture.args.out/"failed.json").exists()


def test_missing_and_failed_reviews_are_retained_without_replacement(fixture):
    (fixture.args.reviews/"current-01.json").unlink()
    replace(fixture.args.reviews/"current-01-terminal.json", lambda x: x.update(status="failed", agent_name=None, note="Spawn rejected"))
    replace(fixture.args.reviews/"history4-02-terminal.json", lambda x: x.update(status="failed", note="Preserved partial answer"))
    summary = j.execute(fixture.args)
    assert summary["arms"]["current"]["reviews"]["current-01"]["answer_status_counts"] == {"missing": 2}
    assert summary["arms"]["current"]["paired_valid_denominator"] == 0
    failed = summary["arms"]["history4"]["reviews"]["history4-02"]
    assert failed["answer_status_counts"] == {"valid": 2} and failed["valid_nonnull_denominator"] == 0
    assert failed["target_agreement"]["rate"] is None
    assert summary["cases"][0]["arms"]["history4"]["reviews"]["history4-02"]["interpretation"] is not None
    assert len(j.read(fixture.args.out/"receipt.json")["missing_responses"]) == 1


@pytest.mark.parametrize("damage", ["packet_hash", "candidate", "candidate_list", "order", "quote", "empty_quote", "null", "json"])
def test_invalid_and_null_answers_do_not_abort_or_gain_denominators(fixture, damage):
    path = fixture.args.reviews/"current-01.json"
    doc = j.read(path)
    if damage == "packet_hash":
        doc["packet_sha256"] = "0"*64
    elif damage == "order":
        doc["cases"].reverse()
    elif damage == "candidate":
        doc["cases"][0]["candidate_id"] = "not supplied"
    elif damage == "candidate_list":
        doc["cases"][0]["candidate_id"] = ["c00"]
    elif damage in ("quote", "empty_quote"):
        doc["cases"][0]["support"] = "I never said this." if damage == "quote" else ""
    elif damage == "null":
        doc["cases"][0].update(candidate_id=None, support="", ambiguous=True, state_definition_issue="Unclear synthetic state.")
    path.write_text("invalid JSON" if damage == "json" else json.dumps(doc))
    summary = j.execute(fixture.args)
    result = summary["arms"]["current"]["reviews"]["current-01"]
    assert result["valid_nonnull_denominator"] == (0 if damage in ("packet_hash", "order", "json") else 2 if damage in ("quote", "empty_quote") else 1)
    if damage in ("quote", "empty_quote"):
        assert result["support_issue_flags"] == 1 and result["answer_status_counts"] == {"valid": 2}
    if damage == "null":
        assert result["answer_status_counts"] == {"unresolved": 1, "valid": 1}
        assert result["ambiguity_flags_in_all_interpretations"] == 2


@pytest.mark.parametrize("damage", ["diagnostic", "packet", "dispatch"])
def test_input_identity_failure_precedes_response_decoding(fixture, monkeypatch, damage):
    if damage == "diagnostic":
        fixture.args.diagnostic_receipt_sha256 = "0"*64
    elif damage == "packet":
        (fixture.args.diagnostic/"current-review.json").write_text("tampered")
    else:
        replace(fixture.args.reviews/"dispatch.json", lambda x: x.update(model="replacement-model"))
    original = j.response
    def never(*args, **kwargs):
        raise AssertionError("Response decoded before input admission")
    monkeypatch.setattr(j, "response", never)
    with pytest.raises(ValueError):
        j.execute(fixture.args)
    assert original is not None and (fixture.args.out/"failed.json").exists()


def test_evaluator_labels_affect_agreement_not_saved_interpretations(fixture):
    bindings = {}
    dispatch, terminals = j.terminals_first(fixture.args.reviews, fixture.args.diagnostic_receipt_sha256, bindings, lambda: None)
    packets = {arm: j.read(fixture.args.diagnostic/(arm+"-review.json")) for arm in j.ARMS}
    reviews = {e["review_id"]: j.response(e, terminals[e["review_id"]], packets[e["arm"]], fixture.args.reviews, bindings, set(), lambda: None)
               for e in dispatch["reviews"]}
    initial = j.join(packets, reviews, fixture.ledger)
    ledger = copy.deepcopy(fixture.ledger)
    ledger[0]["rows"]["current"]["target_id"] = "value:blue"
    changed = j.join(packets, reviews, ledger)
    a = initial["cases"][0]["arms"]["current"]["reviews"]["current-01"]
    b = changed["cases"][0]["arms"]["current"]["reviews"]["current-01"]
    assert a["interpretation"] == b["interpretation"] and a["canonical_candidate_id"] == b["canonical_candidate_id"]
    assert a["target_agreement"] and not b["target_agreement"]


def test_source_change_during_join_preserves_failure(fixture, monkeypatch):
    original = j.join
    def changed(*args):
        result = original(*args)
        (j.ROOT/j.SOURCES[0]).write_text("changed during join")
        return result
    monkeypatch.setattr(j, "join", changed)
    with pytest.raises(ValueError, match="source identity"):
        j.execute(fixture.args)
    assert (fixture.args.out/"summary.json").exists() and (fixture.args.out/"failed.json").exists()
