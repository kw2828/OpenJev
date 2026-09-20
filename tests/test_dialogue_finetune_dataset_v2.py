"""Artificial token layouts only; no corpus, tokenizer, or model loading."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research import dialogue_finetune_dataset as v1
from openjev.research import dialogue_finetune_dataset_v2 as v2


def payload(split="train", did="shared-id", feature_start=0, token_start=10):
    return {"split": split, "dialogue_id": did, "query_ids": [17],
            "user_turn_indices": [0, 2],
            "tokens": [[token_start, token_start + 1], [token_start + 2],
                       [token_start + 3], [token_start + 4], [token_start + 5]],
            "original_feature_ids": list(range(feature_start, feature_start + 5)),
            "turn_text_ids": [0, 0], "query_text_ids": [1], "candidate_text_ids": [[2, 3, 4]],
            "candidate_ids": [["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:red"]],
            "lexical": np.zeros((2, 1, 3, 10), dtype=np.float32)}


def test_same_bare_id_across_splits_accepts_distinct_public_tokens_and_features():
    train = payload()
    dev = payload("dev", feature_start=100, token_start=30)
    with pytest.raises(ValueError, match="Dialogue shared across splits"):
        v1.aggregate_work_profiles([train, dev])
    result = v2.aggregate_work_profiles(iter([train, dev]))
    assert [(p["split"], p["dialogue_id"]) for p in result["profiles"]] == [
        ("train", "shared-id"), ("dev", "shared-id")]
    assert result["all"]["dialogues"] == 2
    assert result["splits"]["train"]["dialogues"] == result["splits"]["dev"]["dialogues"] == 1
    assert result["all"]["totals"]["content_tokens"] == 12
    assert result["version"] == "dialogue-finetune-dataset-v2"
    assert "(split, dialogue_id)" in result["scope"]


@pytest.mark.parametrize("split", ["train", "dev"])
def test_duplicate_within_split_rejected_even_with_distinct_tokens_and_features(split):
    with pytest.raises(ValueError, match="Duplicate cohort dialogue"):
        v2.aggregate_work_profiles([payload(split), payload(split, feature_start=100, token_start=30)])


def test_global_feature_identity_still_rejects_cross_split_token_mismatch():
    with pytest.raises(ValueError, match="feature ID has inconsistent tokenization"):
        v2.aggregate_work_profiles([payload(), payload("dev", token_start=30)])
    result = v2.aggregate_work_profiles([payload(), payload("dev")])
    assert result["all"]["totals"]["unique_texts"] == 10


def test_arithmetic_identical_to_v1_for_unique_bare_ids_and_input_order_unchanged():
    first = payload("train", "z")
    first["tokens"][0] = [5] * 11
    second = payload("dev", "a", feature_start=100, token_start=30)
    data = [second, first]
    before = copy.deepcopy(data)
    old = v1.aggregate_work_profiles(data, chunk_tokens=4, chunk_batch_size=2)
    new = v2.aggregate_work_profiles(data, chunk_tokens=4, chunk_batch_size=2)
    for key in set(old) - {"version", "scope"}:
        assert new[key] == old[key]
    assert [p["dialogue_id"] for p in new["profiles"]] == ["a", "z"]
    for original, saved in zip(data, before, strict=True):
        for key in original:
            if key == "lexical":
                np.testing.assert_array_equal(original[key], saved[key])
            else:
                assert original[key] == saved[key]


def test_four_functions_are_exact_reexports_and_order_source_ids_unchanged():
    for name in ("build_actor_payload", "build_loss_rows", "assert_lexical_parity", "paired_epoch_orders"):
        assert getattr(v2, name) is getattr(v1, name)
    ids = ["train-b", "train-a", "train-c"]
    expected = v1.paired_epoch_orders(ids, [7301], 2)
    actual = v2.paired_epoch_orders(ids, [7301], 2)
    assert actual == expected and actual["dialogue_ids"] == ids == ["train-b", "train-a", "train-c"]
    assert actual["version"] == v1.VERSION  # Unchanged order helper, no relabeling.


@pytest.mark.parametrize("data", [[], [payload("test")], [payload(did="")]])
def test_existing_empty_and_identity_guards_remain(data):
    with pytest.raises(ValueError):
        v2.aggregate_work_profiles(data)
