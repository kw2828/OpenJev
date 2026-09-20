"""Artificial public text, fake content IDs and small local RNG only."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research.dialogue_finetune_dataset import (
    ACTOR_KEYS,
    aggregate_work_profiles,
    assert_lexical_parity,
    build_actor_payload,
    build_loss_rows,
    paired_epoch_orders,
)
from openjev.research.dialogue_finetune_inputs import build_dialogue_inputs, workload_profile
from openjev.research.dialogue_state_data import compile_schema, public_dialogue


def fixture(split="train"):
    catalog = compile_schema([{"service_name": "Paint", "description": "paint service", "slots": [
        {"name": "color", "description": "paint color", "is_categorical": True,
         "possible_values": ["red", "blue"]},
        {"name": "enabled", "description": "enabled", "is_categorical": True,
         "possible_values": ["True", "False", "None"]},
    ]}]).catalog()
    entries = [{"id": q["query_id"], "split": split, "service": q["service"], "slot": q["slot"],
                "text": 20 + 10 * i,
                "candidates": list(range(21 + 10 * i, 21 + 10 * i + len(q["candidates"]))),
                "candidate_ids": [c["id"] for c in q["candidates"]],
                "candidate_values": [c["value"] for c in q["candidates"]]}
               for i, q in enumerate(catalog)]
    public = public_dialogue({"dialogue_id": "fake-dialogue", "services": ["Paint"], "turns": [
        {"speaker": "USER", "utterance": "hello"},
        {"speaker": "SYSTEM", "utterance": "red?"},
        {"speaker": "USER", "utterance": "red"},
        {"speaker": "SYSTEM", "utterance": "blue?"},
        {"speaker": "USER", "utterance": "blue"},
        {"speaker": "SYSTEM", "utterance": "anything else?"},
        {"speaker": "USER", "utterance": "fine"},
    ]})

    def row(q, time, label, bin_name):
        return {"query": q, "time": time, "label": label, "bin": bin_name,
                "unseen": split == "dev", "dontcare": label == 1}

    # Source order is deliberately different from chronological evaluator order.
    packet = {"id": public["dialogue_id"], "turns": [0, 1, 2, 3], "queries": [
        row(0, 3, 3, "revision"), row(1, 0, 0, "unmentioned_retention"),
        row(0, 0, 0, "unmentioned_retention"), row(0, 1, 2, "first_assignment"),
    ]}
    layout = {"id": public["dialogue_id"], "query_ids": [0, 1], "offset": 50,
              "shape": [4, 2, 5, 10]}
    return packet, public, catalog, entries, layout


def tokenizer(text):
    return [ord(c) + 3 for c in text]


def build(items=None, split="train", tokenize=tokenizer):
    return build_actor_payload(*(fixture(split) if items is None else items), tokenize, split=split)


def same_actor(a, b):
    assert set(a) == set(b)
    for key in a:
        if key == "lexical":
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key]


def test_train_matches_immutable_pilot_every_field_and_original_lexical_slice():
    items = fixture()
    expected = build_dialogue_inputs(*items, tokenizer)
    actual = build(items)
    assert actual.pop("split") == "train"
    same_actor(actual, expected)
    witness = assert_lexical_parity(actual, expected["lexical"].copy())
    assert witness == {"matched": True, "positions": 400, "bytes": 1600}


def test_dev_uses_original_global_query_indices_without_train_relabeling():
    items = fixture("dev")
    original = copy.deepcopy(items)
    items[3][:0] = [{"split": "train", "id": "unused"}] * 3
    for row in items[0]["queries"]:
        row["query"] += 3
    items[4]["query_ids"] = [3, 4]
    before = copy.deepcopy(items)
    payload = build(items, split="dev")
    assert payload["split"] == "dev" and payload["query_ids"] == [3, 4]
    assert items == before
    expected = build(original, split="dev")
    for key in ACTOR_KEYS:
        if key == "lexical":
            np.testing.assert_array_equal(payload[key], expected[key])
        else:
            assert payload[key] == expected[key]
    rows = build_loss_rows(items[0], payload, items[3])
    assert all(r["split"] == "dev" and r["unseen"] for r in rows)
    assert rows[0]["query_index"] == 3 and rows[0]["query_position"] == 0


def test_annotations_never_enter_actor_and_unscored_turns_preserve_register():
    class QueryOnly(dict):
        def __getitem__(self, key):
            assert key == "query", f"Actor read annotation {key}"
            return super().__getitem__(key)

    first = build()
    items = fixture()
    items[0]["queries"] = [QueryOnly(query=0, label=object(), time=-1), QueryOnly(query=1), QueryOnly(query=0)]
    for turn in items[1]["turns"]:
        turn["frames"] = object()
    second = build(items)
    same_actor(first, second)
    assert not {"labels", "bins", "previous", "targets", "eligible", "loss_rows"} & second.keys()
    # The unscored blue USER at time 2 updates the literal register before time 3.
    assert second["lexical"][2, 0, 3, 4] == second["lexical"][3, 0, 3, 4] == 1
    assert len(second["turn_text_ids"]) == 4


def test_future_changes_do_not_change_resolved_prefix_or_lexical_history():
    first = build()
    items = fixture()
    items[1]["turns"][-1]["utterance"] = "future red request"
    second = build(items)
    for a, b in zip(first["turn_text_ids"][:3], second["turn_text_ids"][:3], strict=True):
        assert first["tokens"][a] == second["tokens"][b]
    np.testing.assert_array_equal(first["lexical"][:3], second["lexical"][:3])


def test_loss_rows_keep_source_order_full_times_and_separate_validated_targets():
    items = fixture()
    payload = build(items)
    rows = build_loss_rows(items[0], payload, items[3])
    assert [r["source_row_index"] for r in rows] == [0, 1, 2, 3]
    assert [(r["time"], r["turn_index"]) for r in rows] == [(3, 6), (0, 0), (0, 0), (1, 2)]
    assert [r["stratum_index"] for r in rows] == [2, 0, 0, 2]
    assert rows[0]["label_id"] == "value:blue"
    assert len(rows) == 4 < len(payload["turn_text_ids"]) * len(payload["query_ids"])
    before = copy.deepcopy(payload)
    rows[0]["label_id"] = "anything"
    same_actor(payload, before)


def test_candidate_permutation_keeps_identity_and_loss_join():
    items = fixture()
    first = build(items)
    order = [3, 0, 2, 1]
    items[2][0]["candidates"] = [items[2][0]["candidates"][i] for i in order]
    for key in ("candidate_ids", "candidate_values", "candidates"):
        items[3][0][key] = [items[3][0][key][i] for i in order]
    for row in items[0]["queries"]:
        if row["query"] == 0:
            row["label"] = order.index(row["label"])
    second = build(items)
    np.testing.assert_array_equal(second["lexical"][:, 0, :4], first["lexical"][:, 0, order])
    rows = build_loss_rows(items[0], second, items[3])
    assert rows[0]["label_index"] == 0 and rows[0]["label_id"] == "value:blue"


@pytest.mark.parametrize("mutation", ["split", "test", "bool_query", "bool_shape", "missing_public_user"])
def test_bad_public_joins_fail_before_tokenization(mutation):
    items = fixture()
    split = "train"
    if mutation == "split":
        items[3][0]["split"] = "dev"
    elif mutation == "test":
        split = "test"
    elif mutation == "bool_query":
        items[0]["queries"][0]["query"] = False
    elif mutation == "bool_shape":
        items[4]["shape"][0] = True
    else:
        items[1]["user_turns"].pop()
    calls = []
    with pytest.raises(ValueError):
        build(items, split=split, tokenize=lambda text: calls.append(text) or [1])
    assert not calls


@pytest.mark.parametrize("mutation", ["duplicate", "bool_time", "bad_label", "bad_bin", "wrong_bin",
                                      "bad_dontcare", "unseen_train", "missing_query"])
def test_bad_annotations_reject_in_loss_api_only(mutation):
    items = fixture()
    payload = build(items)
    rows = items[0]["queries"]
    if mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "bool_time":
        rows[0]["time"] = True
    elif mutation == "bad_label":
        rows[0]["label"] = 100
    elif mutation == "bad_bin":
        rows[0]["bin"] = "invalid"
    elif mutation == "wrong_bin":
        rows[0]["bin"] = "first_assignment"
    elif mutation == "bad_dontcare":
        rows[0]["dontcare"] = True
    elif mutation == "unseen_train":
        rows[0]["unseen"] = True
    else:
        rows.pop(1)
    with pytest.raises(ValueError):
        build_loss_rows(items[0], payload, items[3])


def test_dev_inconsistent_panel_flags_reject_and_clear_dontcare_transitions_valid():
    items = fixture("dev")
    payload = build(items, split="dev")
    items[0]["queries"][0]["unseen"] = False
    with pytest.raises(ValueError, match="Inconsistent"):
        build_loss_rows(items[0], payload, items[3])
    items = fixture()
    items[0]["queries"][0].update(label=0, bin="clear")
    items[0]["queries"][3].update(label=1, dontcare=True)
    rows = build_loss_rows(items[0], build(items), items[3])
    assert rows[0]["bin"] == "clear" and rows[3]["label_id"] == "reserved:DONTCARE"


@pytest.mark.parametrize("mutation", ["value", "dtype", "shape", "nan"])
def test_original_lexical_parity_is_exact(mutation):
    payload = build()
    observed = payload["lexical"].copy()
    if mutation == "value":
        observed[0, 0, 0, 0] = 1
    elif mutation == "dtype":
        observed = observed.astype(np.float64)
    elif mutation == "shape":
        observed = observed.reshape(-1)
    else:
        observed[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        assert_lexical_parity(payload, observed)


def test_streamed_work_profiles_sum_costs_but_take_dimension_maxima():
    train = build(tokenize=lambda _: [1])
    train["tokens"][0] = [1] * 255
    dev = copy.deepcopy(train)
    dev["split"], dev["dialogue_id"] = "dev", "fake-dev"
    summary = aggregate_work_profiles(p for p in (train, dev))
    expected = workload_profile(train)
    assert summary["all"]["dialogues"] == 2
    assert summary["all"]["totals"]["content_tokens"] == 2 * expected["content_tokens"]
    assert summary["all"]["totals"]["padded_attention_positions"] == 2 * expected["padded_attention_positions"]
    assert summary["all"]["totals"]["overlength_texts_chunked"] == 2
    assert summary["all"]["maxima"]["max_content_tokens_per_text"] == 255
    assert summary["all"]["maxima"]["max_candidates"] == 5
    assert "max_candidates" not in summary["all"]["totals"]
    assert "chunk_tokens" not in summary["all"]["totals"]
    assert set(summary["splits"]) == {"train", "dev"}
    assert summary["splits"]["dev"]["totals"]["truncated_tokens"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "cross_split", "inconsistent_feature", "empty"])
def test_work_aggregation_rejects_membership_or_feature_conflicts(mutation):
    a, b = build(), build()
    if mutation == "cross_split":
        b["split"] = "dev"
    elif mutation == "inconsistent_feature":
        b["dialogue_id"] = "other-dialogue"
        b["tokens"][0].append(999)
    payloads = [] if mutation == "empty" else [a, b]
    with pytest.raises(ValueError):
        aggregate_work_profiles(iter(payloads))


def test_orders_match_original_generator_and_do_not_touch_global_rng_or_ids():
    ids = ["c", "a", "b", "d"]
    state = np.random.get_state()
    first = paired_epoch_orders(ids, [7301, 7302], 3)
    second = paired_epoch_orders(tuple(ids), [7301, 7302], 3)
    assert first == second and first["dialogue_ids"] == ids == ["c", "a", "b", "d"]
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    for seed in (7301, 7302):
        rng = np.random.default_rng(seed)
        expected = [rng.permutation(4).tolist() for _ in range(3)]
        assert first["orders"][str(seed)] == expected
        assert all(sorted(epoch) == list(range(4)) for epoch in expected)


@pytest.mark.parametrize("ids,seeds,epochs", [([], [1], 1), (["a", "a"], [1], 1),
                                            (["a"], [True], 1), (["a"], [1, 1], 1),
                                            (["a"], [1], False)])
def test_invalid_order_contract(ids, seeds, epochs):
    with pytest.raises(ValueError):
        paired_epoch_orders(ids, seeds, epochs)
