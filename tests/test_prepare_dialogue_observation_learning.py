"""Artificial preparation integration; no corpus or pretrained assets."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from openjev.research.dialogue_finetune_dataset import build_actor_payload, paired_epoch_orders
from openjev.research.dialogue_state_data import compile_schema, public_dialogue

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("observation_preparation", SCRIPTS / "prepare_dialogue_observation_learning.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture():
    catalog = compile_schema([{"service_name": "Tickets", "description": "tickets", "slots": [
        {"name": "count", "description": "number of tickets", "is_categorical": True,
         "possible_values": ["1", "2", "True", "False"]}]}]).catalog()
    q = catalog[0]
    entries = [{"id": q["query_id"], "split": "train", "service": q["service"], "slot": q["slot"],
                "text": 10, "candidates": list(range(11, 17)),
                "candidate_ids": [c["id"] for c in q["candidates"]],
                "candidate_values": [c["value"] for c in q["candidates"]]}]
    public = public_dialogue({"dialogue_id": "artificial", "services": ["Tickets"], "turns": [
        {"speaker": "USER", "utterance": "two please"},
        {"speaker": "SYSTEM", "utterance": "one?"},
        {"speaker": "USER", "utterance": "yes please"},
        {"speaker": "SYSTEM", "utterance": "okay"},
        {"speaker": "USER", "utterance": "1"}]})
    packet = {"id": "artificial", "turns": [0, 1, 2], "queries": [{"query": 0}]}
    layout = {"id": "artificial", "query_ids": [0], "offset": 0, "shape": [3, 1, 6, 10]}
    payload = build_actor_payload(packet, public, catalog, entries, layout,
                                  lambda text: [ord(c) for c in text], split="train")
    return payload, public, catalog, entries


def test_paired_lexical_views_preserve_encoder_identity_and_causal_registers():
    payload, public, catalog, entries = fixture()
    before = copy.deepcopy(payload)
    normalized, registers = prep.build_lexical_pair(payload, public, catalog, entries)
    for key in before:
        if key == "lexical":
            np.testing.assert_array_equal(payload[key], before[key])
        else:
            assert payload[key] == before[key]
    np.testing.assert_array_equal(normalized[..., 6:], payload["lexical"][..., 6:])
    ids = payload["candidate_ids"][0]
    none, one, two = (ids.index(x) for x in ("reserved:NOT_MENTIONED", "value:1", "value:2"))
    assert registers == {"original": [[none, none, one]], "numbers": [[two, two, one]]}
    assert normalized[1, 0, one, 1] == 1  # SYSTEM match does not write user register.
    assert normalized[1, 0, two, 4] == 1
    assert not payload["lexical"][0, 0, two, 0] and normalized[0, 0, two, 0]


def test_public_subset_rejects_duplicates_and_missing_members(tmp_path):
    path = tmp_path / "public.jsonl"
    path.write_text('\n'.join(json.dumps({"dialogue_id": d}) for d in ["a", "other", "b"]))
    assert list(prep.public_subset(path, {"a", "b"})) == ["a", "b"]
    with pytest.raises(ValueError, match="Complete public"):
        prep.public_subset(path, {"missing"})
    path.write_text('\n'.join(json.dumps({"dialogue_id": d}) for d in ["a", "a"]))
    with pytest.raises(ValueError, match="Duplicate public"):
        prep.public_subset(path, {"a"})


def test_effective_batches_preserve_tail_and_exclude_dev(monkeypatch):
    monkeypatch.setitem(prep.CONFIG, "seeds", [7, 11])
    ids = [f"d-{i}" for i in range(65)]
    orders = paired_epoch_orders(ids, [7, 11], 2)
    names = ("encoder_calls", "padded_token_positions", "padded_attention_positions",
             "real_question_updates", "padded_candidate_positions", "encoder_sequences")
    profiles = [{"split": "train", "dialogue_id": did,
                 "work": dict.fromkeys(names, i + 1)} for i, did in enumerate(ids)]
    profiles.append({"split": "dev", "dialogue_id": "irrelevant", "work": dict.fromkeys(names, 99999)})
    result = prep.effective_batches(orders, profiles)
    assert result["sizes"] == {"32": 8, "1": 4} and result["batches"] == 12
    all_sums = [sum(i + 1 for i in order[start:start + 32])
                for sequence in orders["orders"].values() for order in sequence for start in range(0, 65, 32)]
    for record in result["maxima"].values():
        assert record["value"] == max(all_sums)
        assert record["dialogue_ids"] == [ids[i] for i in record["dialogue_indices"]]
    orders["orders"]["7"][0][0] = orders["orders"]["7"][0][1]
    with pytest.raises(ValueError, match="Complete epoch"):
        prep.effective_batches(orders, profiles)


def test_final_cap_demotes_success_and_retains_failure(tmp_path, monkeypatch):
    peak = [0]
    monkeypatch.setattr(prep.qualified, "rss", lambda: peak[0])
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="RSS cap"), prep.attempt(out) as (_, progress, _):
        progress["dialogues"] = 2
        prep.write(out / "completed.json", {"status": "too early"})
        peak[0] = prep.CAPS["rss_bytes"] + 1
    assert not (out / "completed.json").exists()
    assert prep.read(out / "failed.json")["progress"]["dialogues"] == 2
    with pytest.raises(FileExistsError), prep.attempt(out):
        pass


def test_failure_receipt_problem_keeps_original_exception(tmp_path, monkeypatch):
    original = prep.write

    def write(path, obj):
        if path.name == "failed.json":
            raise OSError("storage unavailable")
        original(path, obj)

    monkeypatch.setattr(prep, "write", write)
    with pytest.raises(ValueError, match="original") as caught, prep.attempt(tmp_path / "attempt"):
        raise ValueError("original")
    assert "storage unavailable" in caught.value.__notes__[0]


def test_newer_pilot_sources_cannot_drift(monkeypatch):
    monkeypatch.setattr(prep.qualified, "sources", lambda: {"former_new_helper.py": "now"})
    with pytest.raises(ValueError, match="All inherited"):
        prep.inherited_sources({"source_sha256": {"former_new_helper.py": "qualified"}})
    assert prep.inherited_sources({"source_sha256": {"former_new_helper.py": "now"}}) == {"former_new_helper.py": "now"}
