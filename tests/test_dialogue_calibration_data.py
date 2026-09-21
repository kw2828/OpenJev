"""Artificial dialogue complements and original-builder fixture parity only."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openjev.research import dialogue_calibration_data as c
from openjev.research.dialogue_finetune_dataset import build_actor_payload
from openjev.research.dialogue_number_lexical import lexical_observations
from openjev.research.dialogue_state_data import public_dialogue, text_identity

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "calibration_original_actor_fixture", Path(__file__).with_name("test_dialogue_finetune_dataset.py")
)
original_fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(original_fixture)


def public(did, text):
    return public_dialogue({"dialogue_id": did, "turns": [{"speaker": "USER", "utterance": text}]})


class IdentityOnlyEndpoint(dict):
    def __getitem__(self, key):
        assert key in {"dialogue_id", "query_id", "turn_index"}, f"Selection accessed {key}"
        return super().__getitem__(key)


def selection_fixture():
    catalog = original_fixture.fixture()[2]
    records = [public(f"synthetic-{i:04d}", f"Distinct synthetic conversation {i}") for i in range(514)]
    first, second = sorted(records, key=lambda p: c.rank(p["dialogue_id"]))[:2]
    second["turns"][0]["utterance"] = first["turns"][0]["utterance"].upper() + "  "
    records += [public("fitted", "Training duplicate"), public("fitted-copy", "  TRAINING duplicate  "),
                public("dev-copy", "evaluated development collision"), public("no-questions", "No categorical questions")]
    dev = [public("evaluated-dev", "EVALUATED development collision")]
    metadata = [IdentityOnlyEndpoint(dialogue_id=p["dialogue_id"], query_id=catalog[0]["query_id"],
                                     turn_index=0, label_index=object(), label_id=object(), model_score=object())
                for p in records if p["dialogue_id"] != "no-questions"]
    return records, metadata, catalog, {"fitted"}, dev, first["dialogue_id"], second["dialogue_id"]


def test_selection_is_deterministic_and_excludes_entire_fitted_and_dev_text_groups():
    records, metadata, catalog, fitted, dev, representative, duplicate = selection_fixture()
    selection = c.select_calibration_dialogues(records, metadata, catalog, fitted, dev)
    assert selection == c.select_calibration_dialogues(records[::-1], metadata[::-1], catalog, fitted, dev[::-1])
    ids = selection["selected_ids"]
    assert len(ids) == len(set(ids)) == 512
    assert representative in ids and duplicate not in ids
    assert not {"fitted", "fitted-copy", "dev-copy", "no-questions"} & set(ids)
    assert selection["salt"] == "openjev-calibration-v1:"
    assert selection["source_split"] == "train" and selection["analysis_role"] == "calibration"
    assert ids == sorted(ids, key=c.rank)
    assert len({row["text_sha256"] for row in selection["selected_groups"]}) == 512
    by_id = {p["dialogue_id"]: p for p in records}
    assert all(row["text_sha256"] == text_identity(by_id[row["dialogue_id"]])
               and row["salted_id_sha256"] == c.rank(row["dialogue_id"])[0]
               for row in selection["selected_groups"])
    counts = selection["counts"]
    assert counts["excluded_fitted_group_eligible_dialogues"] == 2
    assert counts["excluded_dev_group_eligible_dialogues"] == 1
    assert counts["excluded_either_group_eligible_dialogues"] == 3
    assert counts["excluded_both_groups_eligible_dialogues"] == 0
    assert counts["ineligible_without_categorical_endpoints"] == 1
    assert counts["duplicate_members_not_represented"] == 1
    assert counts["surviving_unique_groups"] == 513 and counts["unselected_unique_groups"] == 1


def test_same_bare_id_across_train_and_dev_does_not_exclude_unrelated_text():
    records, metadata, catalog, fitted, dev, representative, _ = selection_fixture()
    dev.append(public(representative, "Unrelated DEV public text under the same bare ID"))
    result = c.select_calibration_dialogues(records, metadata, catalog, fitted, dev)
    assert representative in result["selected_ids"]


def test_fitted_and_dev_collision_counts_are_independent_and_union_is_not_double_counted():
    records, metadata, catalog, fitted, dev, _, _ = selection_fixture()
    dev.append(public("also-dev", "Training duplicate"))
    counts = c.select_calibration_dialogues(records, metadata, catalog, fitted, dev)["counts"]
    assert counts["excluded_fitted_group_eligible_dialogues"] == 2
    assert counts["excluded_dev_group_eligible_dialogues"] == 3
    assert counts["excluded_both_groups_eligible_dialogues"] == 2
    assert counts["excluded_either_group_eligible_dialogues"] == 3


def test_selection_fails_instead_of_shrinking_an_insufficient_complement():
    records, metadata, catalog, fitted, dev, _, _ = selection_fixture()
    keep = {p["dialogue_id"] for p in records[:500]} | fitted
    records = [p for p in records if p["dialogue_id"] in keep]
    metadata = [m for m in metadata if m["dialogue_id"] in keep]
    with pytest.raises(ValueError, match="Insufficient eligible unique"):
        c.select_calibration_dialogues(records, metadata, catalog, fitted, dev)


@pytest.mark.parametrize("defect", ["unknown_query", "duplicate_endpoint", "wrong_turn", "missing_fitted", "duplicate_id"])
def test_invalid_selection_membership_is_not_silently_filtered(defect):
    records, metadata, catalog, fitted, dev, _, _ = selection_fixture()
    if defect == "unknown_query":
        metadata[0]["query_id"] = "unknown-categorical-query"
    elif defect == "duplicate_endpoint":
        metadata.append(metadata[0])
    elif defect == "wrong_turn":
        metadata[0]["turn_index"] = 99
    elif defect == "missing_fitted":
        fitted.add("not-in-public-train")
    else:
        records.append(records[0])
    with pytest.raises(ValueError):
        c.select_calibration_dialogues(records, metadata, catalog, fitted, dev)


@pytest.fixture
def builder_fixture():
    original = original_fixture.fixture()
    packet, template, catalog, entries, _ = original
    records, labels = [], []
    for index in range(513):
        did = "fit-only" if index == 512 else f"calibration-{index:04d}"
        p = copy.deepcopy(template)
        p["dialogue_id"] = did
        p["turns"][0]["utterance"] += " synthetic case " + did
        records.append(p)
        for row in packet["queries"]:
            q = entries[row["query"]]
            labels.append({"dialogue_id": did, "turn_index": p["user_turns"][row["time"]]["turn_index"],
                           "query_id": q["id"], "service": q["service"], "slot": q["slot"],
                           "label_index": row["label"], "label_id": q["candidate_ids"][row["label"]],
                           "bin": row["bin"], "is_dontcare": row["dontcare"], "unseen_service": False})
    selection = c.select_calibration_dialogues(records, labels, catalog, {"fit-only"},
                                               [public("evaluated-dev", "Different DEV stream")])
    return selection, records, labels, catalog, original


def build(fixture):
    selection, records, labels, catalog, _ = fixture
    return c.build_calibration_inputs(selection, records, labels, catalog, original_fixture.tokenizer)


def test_new_packet_matches_original_actor_fixture_and_preserves_full_stream(builder_fixture):
    result = build(builder_fixture)
    selection, records, _, catalog, original = builder_fixture
    actor = result["actors"][0]
    did = selection["selected_ids"][0]
    p = next(record for record in records if record["dialogue_id"] == did)
    packet, _, _, entries, layout = copy.deepcopy(original)
    packet["id"] = layout["id"] = did
    expected = build_actor_payload(packet, p, catalog, entries, layout, original_fixture.tokenizer, split="train")
    for key in ("dialogue_id", "query_ids", "user_turn_indices", "tokens", "turn_text_ids",
                "query_text_ids", "candidate_text_ids", "candidate_ids"):
        assert actor[key] == expected[key]
    shape, offset = actor["lexical_shape"], actor["lexical_offset"]
    original_lexical = result["lexical"]["original"][offset:offset + np.prod(shape)].reshape(shape)
    np.testing.assert_array_equal(original_lexical, expected["lexical"])
    normalized = result["lexical"]["numbers"][offset:offset + np.prod(shape)].reshape(shape)
    for j, qi in enumerate(actor["query_ids"]):
        expected_numbers, _ = lexical_observations(p, catalog[qi], actor["candidate_ids"][j])
        np.testing.assert_array_equal(normalized[:, j, :len(actor["candidate_ids"][j])], expected_numbers)
    np.testing.assert_array_equal(normalized[..., 6:], original_lexical[..., 6:])
    assert len(actor["user_turn_indices"]) == 4
    assert {r["time"] for r in result["targets"][0]["rows"]} == {0, 1, 3}
    assert result["counts"]["dialogues"] == 512 and result["counts"]["public_user_turns"] == 2048
    assert result["counts"]["scored_endpoints"] == 2048
    assert result["lexical"]["original"].dtype == result["lexical"]["numbers"].dtype == np.float32
    assert all(a["split"] == a["source_split"] == "train" and a["analysis_role"] == "calibration"
               for a in result["actors"])
    assert all(t["source_split"] == "train" and t["analysis_role"] == "calibration"
               and all(r["split"] == "train" and r["unseen"] is False for r in t["rows"])
               for t in result["targets"])
    for query in result["packet"]["queries"]:
        assert query["split"] == "train"
        assert query["candidate_ids"] == entries[result["packet"]["queries"].index(query)]["candidate_ids"]


def test_fresh_global_indices_offsets_padding_and_actor_label_separation(builder_fixture):
    result = build(builder_fixture)
    texts = result["texts"]
    assert len(texts) == len(set(texts))
    offset = 0
    forbidden = {"label", "label_index", "label_id", "bin", "stratum", "unseen", "dontcare", "rows"}
    for actor, layout, packet in zip(result["actors"], result["layouts"], result["packet"]["cohort"], strict=True):
        assert not forbidden & set(actor) and not forbidden & set(packet)
        assert actor["lexical_offset"] == layout["offset"] == offset
        assert packet["turns"] == [actor["original_feature_ids"][i] for i in actor["turn_text_ids"]]
        for feature, tokens in zip(actor["original_feature_ids"], actor["tokens"], strict=True):
            assert tokens == original_fixture.tokenizer(texts[feature])
        shape = actor["lexical_shape"]
        for lexical in result["lexical"].values():
            array = lexical[offset:offset + np.prod(shape)].reshape(shape)
            for j, ids in enumerate(actor["candidate_ids"]):
                assert not array[:, j, len(ids):].any()
        offset += int(np.prod(shape))
    assert offset == result["counts"]["lexical_positions"]
    assert all(len(array) == offset for array in result["lexical"].values())


def test_valid_target_change_does_not_change_actor_observations(builder_fixture):
    before = build(builder_fixture)
    changed = copy.deepcopy(builder_fixture)
    selection, _, labels, _, _ = changed
    did = selection["selected_ids"][0]
    for row in labels:
        if row["dialogue_id"] == did and row["slot"] == "enabled":
            row.update(label_index=2, label_id="value:True", bin="first_assignment", is_dontcare=False)
    after = build(changed)
    assert before["actors"] == after["actors"]
    assert before["packet"] == after["packet"] and before["texts"] == after["texts"]
    for name in ("original", "numbers"):
        np.testing.assert_array_equal(before["lexical"][name], after["lexical"][name])
    assert before["targets"] != after["targets"]


@pytest.mark.parametrize("defect", ["salt", "role", "duplicate_member", "wrong_text_identity"])
def test_preparation_rejects_tampered_selection(builder_fixture, defect):
    selection = builder_fixture[0]
    if defect == "salt":
        selection["salt"] = "different:"
    elif defect == "role":
        selection["source_split"] = "dev"
    elif defect == "duplicate_member":
        selection["selected_ids"][0] = selection["selected_ids"][1]
    else:
        selection["selected_groups"][0]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build(builder_fixture)


@pytest.mark.parametrize("defect", ["label_identity", "unseen_train", "transition_bin"])
def test_preparation_rejects_invalid_evaluator_semantics(builder_fixture, defect):
    selection, _, labels, _, _ = builder_fixture
    first = next(row for row in labels if row["dialogue_id"] == selection["selected_ids"][0])
    if defect == "label_identity":
        first["label_id"] = "value:unsupported"
    elif defect == "unseen_train":
        first["unseen_service"] = True
    else:
        first["bin"] = "unmentioned_retention"
    with pytest.raises(ValueError):
        build(builder_fixture)
