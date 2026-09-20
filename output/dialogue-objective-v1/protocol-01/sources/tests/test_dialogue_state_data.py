"""Synthetic parser, causal whitelist, category and exclusive preparation checks."""

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from openjev.research.dialogue_state_data import (
    DONTCARE,
    NOT_MENTIONED,
    compile_schema,
    parse_dialogue,
    public_dialogue,
    public_prefix,
    text_identity,
    utterance_pair,
)


def raw_schema():
    return [{"service_name": service, "description": f"Description {service}", "slots": [
        {"name": "choice", "description": "Requested choice", "is_categorical": True,
         "possible_values": ["None", "other", "reserved:NOT_MENTIONED"]},
        {"name": "free", "description": "Free text", "is_categorical": False,
         "possible_values": ["example"]}]} for service in ("A", "B")]


def frame(service="A", value=None):
    values = {} if value is None else {"choice": value}
    return {"service": service, "state": {"slot_values": values, "active_intent": "PRIVATE"},
            "actions": [{"values": ["PRIVATE"]}], "slots": [{"private": True}]}


def user(text, *frames):
    return {"speaker": "USER", "utterance": text, "frames": list(frames)}


def system(text):
    return {"speaker": "SYSTEM", "utterance": text, "frames": [{"service_results": ["PRIVATE"]}]}


def dialogue(turns):
    return {"dialogue_id": "synthetic", "services": ["FUTURE_PRIVATE_SERVICE"], "turns": turns}


def test_reserved_ids_literal_none_and_immutable_schema():
    raw = raw_schema()
    schema = compile_schema(raw)
    assert [c.id for c in schema.queries[0].candidates] == [NOT_MENTIONED, DONTCARE,
        "value:None", "value:other", "value:reserved:NOT_MENTIONED"]
    raw[0]["slots"][0]["possible_values"][0] = "mutated"
    assert schema.queries[0].candidates[2].value == "None"
    with pytest.raises(FrozenInstanceError):
        schema.queries[0].slot = "mutated"
    assert len(schema.queries) == 2  # Non-categorical illustrative values never create queries.


def test_candidate_ids_independent_of_ontology_order_and_schema_only_text():
    raw = raw_schema()
    first = compile_schema(raw)
    raw[0]["slots"][0]["possible_values"].reverse()
    second = compile_schema(raw)
    assert {c.id: c.text for c in first.queries[0].candidates} == {
        c.id: c.text for c in second.queries[0].candidates}
    assert first.queries[0].query_text == "Service: Description A\nSlot: Requested choice"
    assert first.queries[0].candidates[2].text.endswith("Value: None")


def test_every_bin_absent_service_does_not_clear_and_dontcare_is_assigned():
    raw = dialogue([user("start", frame()), user("one", frame(value=["None"])),
                    user("other service", frame("B", ["other"])),
                    user("keep", frame(value=["None"])), user("change", frame(value=["other"])),
                    user("clear", frame()), user("no preference", frame(value=["dontcare"]))])
    public, labels = parse_dialogue(raw, compile_schema(raw_schema()), train_services={"A"})
    a = [r for r in labels if r["service"] == "A"]
    assert [r["bin"] for r in a] == ["unmentioned_retention", "first_assignment", "assigned_retention",
                                    "revision", "clear", "first_assignment"]
    assert a[1]["label_id"] == "value:None" and a[1]["label_index"] == 2
    assert a[-1]["label_id"] == DONTCARE and a[-1]["is_dontcare"]
    assert next(r for r in labels if r["service"] == "B")["unseen_service"]
    assert len(public["turns"]) == 7 and len(labels) == 7
    assert not any(r["turn_index"] == 2 and r["service"] == "A" for r in labels)


def test_all_categorical_slots_and_multiframe_user_get_queries():
    raw = raw_schema()
    another = copy.deepcopy(raw[0]["slots"][0])
    another["name"] = "second"
    raw[0]["slots"].append(another)
    _, labels = parse_dialogue(dialogue([user("both", frame(), frame("B"))]),
                               compile_schema(raw), train_services={"A", "B"})
    assert {(r["service"], r["slot"]) for r in labels} == {("A", "choice"), ("A", "second"), ("B", "choice")}


def test_private_metadata_changes_do_not_change_public_features():
    raw = dialogue([user("hello", frame()), system("question"), user("answer", frame(value=["other"]))])
    changed = copy.deepcopy(raw)
    changed["services"] = ["another_future_service"]
    changed["turns"][0]["frames"] = [{"everything": "different"}]
    changed["turns"][1]["frames"] = [{"state": "gold", "service_call": {"answer": "gold"}}]
    assert public_dialogue(raw) == public_dialogue(changed)
    assert set(public_dialogue(raw)) == {"dialogue_id", "turns", "user_turns"}
    assert all(set(t) == {"speaker", "utterance"} for t in public_dialogue(raw)["turns"])


def test_causal_prefix_pair_and_future_poison():
    raw = dialogue([user("first", frame()), system("ask"), user("now", frame()), system("future secret")])
    a = public_dialogue(raw)
    raw["turns"][3]["utterance"] = "a different future secret"
    b = public_dialogue(raw)
    assert public_prefix(a, 2) == public_prefix(b, 2)
    assert utterance_pair(a, 2) == {"previous_system": "ask", "user": "now"}
    assert utterance_pair(a, 0)["previous_system"] == ""
    assert a["user_turns"] == [{"turn_index": 0, "previous_system_turn_index": None},
                               {"turn_index": 2, "previous_system_turn_index": 1}]
    with pytest.raises(ValueError, match="USER"):
        public_prefix(a, 1)


@pytest.mark.parametrize("bad", [[], ["other", "None"], [123], None, ["out of ontology"]])
def test_bad_categorical_labels_fail_with_location(bad):
    f = frame()
    f["state"]["slot_values"]["choice"] = bad
    with pytest.raises(ValueError, match="synthetic:0:A/choice"):
        parse_dialogue(dialogue([user("text", f)]), compile_schema(raw_schema()), train_services={"A"})


def test_non_categorical_multiple_surface_forms_are_not_categorical_errors():
    f = frame()
    f["state"]["slot_values"]["free"] = ["six pm", "6 pm"]
    _, labels = parse_dialogue(dialogue([user("text", f)]), compile_schema(raw_schema()), train_services={"A"})
    assert len(labels) == 1 and labels[0]["label_id"] == NOT_MENTIONED


def test_duplicate_or_unknown_schema_and_frame_rejected():
    schema = raw_schema()
    schema[0]["slots"][0]["possible_values"].append("None")
    with pytest.raises(ValueError, match="Duplicate candidates"):
        compile_schema(schema)
    for frames in ([frame(), frame()], [frame("unknown")]):
        with pytest.raises(ValueError, match="Unknown/duplicate"):
            parse_dialogue(dialogue([user("text", *frames)]), compile_schema(raw_schema()), train_services={"A"})


def test_normalized_dialogue_identity_ignores_id_but_preserves_roles_and_all_turns():
    a = public_dialogue(dialogue([user("Ａ\t B", frame()), system("Okay")]))
    b = public_dialogue(dialogue([user("a b", frame()), system(" okay  ")]))
    b["dialogue_id"] = "another"
    assert text_identity(a) == text_identity(b)
    b["turns"][1]["speaker"] = "USER"
    assert text_identity(a) != text_identity(b)


def test_preparation_cross_split_duplicates_and_receipt(tmp_path, monkeypatch):
    import prepare_sgd_state as driver

    source, out = tmp_path / "source", tmp_path / "data"
    for split in ("train", "dev"):
        (source / split).mkdir(parents=True)
        (source / split / "schema.json").write_text(json.dumps(raw_schema()))
    train = dialogue([user("Hello", frame(value=["None"]))])
    duplicate = copy.deepcopy(train)
    duplicate["dialogue_id"] = "dev-duplicate"
    duplicate["turns"][0]["utterance"] = " hello  "
    unique = dialogue([user("new text", frame(value=["other"]))])
    unique["dialogue_id"] = "dev-unique"
    (source / "train/dialogues_001.json").write_text(json.dumps([train]))
    (source / "dev/dialogues_001.json").write_text(json.dumps([duplicate, unique]))
    (source / "fetch-receipt.json").write_text("{}")
    monkeypatch.setattr(driver, "authenticate_source", lambda _: {"files": {}})
    driver.prepare(source, out)
    done = json.loads((out / "completed.json").read_text())
    assert done["status"] == "completed" and done["new_model_calls"] == 0
    assert done["stats"]["train"]["counts"]["dialogues_retained"] == 1
    assert done["stats"]["dev"]["counts"]["dialogues_retained"] == 1
    excluded = json.loads((out / "duplicates.json").read_text())["excluded_dev"]
    assert excluded[0]["dialogue_id"] == "dev-duplicate"
    assert json.loads((out / "dev-dialogues.jsonl").read_text())["dialogue_id"] == "dev-unique"
    for name, meta in done["files"].items():
        assert driver.sha(out / name) == meta["sha256"]
    with pytest.raises(FileExistsError):
        driver.prepare(source, out)


def test_bad_label_preparation_preserves_context_and_no_completion(tmp_path, monkeypatch):
    import prepare_sgd_state as driver

    source = tmp_path / "source"
    for split in ("train", "dev"):
        (source / split).mkdir(parents=True)
        (source / split / "schema.json").write_text(json.dumps(raw_schema()))
    (source / "fetch-receipt.json").write_text("{}")
    (source / "train/dialogues_001.json").write_text(json.dumps([dialogue([user("bad", frame(value=["other", "None"]))])]))
    monkeypatch.setattr(driver, "authenticate_source", lambda _: {"files": {}})
    out = tmp_path / "data"
    with pytest.raises(ValueError, match="singleton"):
        driver.prepare(source, out)
    failed = json.loads((out / "failed.json").read_text())
    assert failed["context"]["dialogue_id"] == "synthetic"
    assert not (out / "completed.json").exists()
