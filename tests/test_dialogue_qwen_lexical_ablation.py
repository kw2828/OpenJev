"""Artificial public packets only; exact subtraction, never model execution."""
from __future__ import annotations

import copy

import pytest

from openjev.research.dialogue_qwen_lexical_ablation import (
    EXPLANATION,
    LEGEND,
    TASK,
    transform_request,
    verify_transform,
)
from openjev.research.dialogue_qwen_observation import make_question


def packet(index=0, arm="current"):
    query = {"query_text": "Synthetic query. " + EXPLANATION + LEGEND,
             "candidates": [{"id": "reserved:NOT_MENTIONED", "text": "No previous constraint"},
                            {"id": "value:None", "text": "Literal None"},
                            {"id": "reserved:DONTCARE", "text": "No preference"}]}
    question, mapping = make_question(row_index=index, query=query, previous_candidate_id="value:None",
                                      flags=[[0]*10, [1]*10, [0, 1]*5])
    question = question.model_dump()
    return {"request_id": f"{arm}:synthetic:{index}:0", "arm": arm, "dialogue_id": "synthetic", "time": index,
            "row_indices": [index], "request": {"context": "  Exact public context; do not alter.  ", "questions": [question]},
            "canonical_id_maps": [mapping], "ordered_candidate_ids": [[mapping[c["id"]] for c in sorted(
                question["candidates"], key=lambda c: c["description"])]], "tokens": [[11, 12, 13]],
            "label_ids": list(range(32, 44))}


def test_only_anchored_blocks_removed_and_original_unchanged():
    original = packet()
    before = copy.deepcopy(original)
    result = transform_request(original)
    old, new = original["request"]["questions"][0], result["request"]["questions"][0]
    assert new["question"] == TASK[:-len(EXPLANATION)] + old["question"][len(TASK):-len(LEGEND)]
    assert new["question"].startswith("What is") and "states. \nSynthetic" in new["question"]
    assert EXPLANATION in new["question"] and LEGEND in new["question"]  # Embedded query text preserved.
    assert new["question"].endswith("Previous committed value:\nLiteral None")
    for a, b in zip(old["candidates"], new["candidates"], strict=True):
        assert a["id"] == b["id"] and a["description"].rsplit("\nPublic lexical flags: ", 1)[0] == b["description"]
    assert original == before
    assert result["request"]["context"] == original["request"]["context"]
    assert verify_transform(original, result)


@pytest.mark.parametrize("suffix", ["0,0", "0,0,0,0,0,0,0,0,0,2", "0.0,0,0,0,0,0,0,0,0,0",
                                    "0,0,0,0,0,0,0,0,0,0\n", "0,0,0,0,0,0,0,0,0,0,0"])
def test_malformed_flag_suffix_fails_closed(suffix):
    original = packet()
    candidate = original["request"]["questions"][0]["candidates"][0]
    candidate["description"] = candidate["description"].rsplit("\nPublic lexical flags: ", 1)[0] + "\nPublic lexical flags: " + suffix
    with pytest.raises(ValueError, match="ten binary"):
        transform_request(original)


@pytest.mark.parametrize("damage", ["task", "legend", "type", "label_order"])
def test_mismatched_fixed_prompt_contract_rejected(damage):
    original = packet()
    q = original["request"]["questions"][0]
    if damage == "task":
        q["question"] = q["question"].replace("Lexical flags are noisy", "Lexical flags are perfect", 1)
    elif damage == "legend":
        q["question"] += " "
    elif damage == "type":
        q["candidates"][0]["description"] = q["candidates"][0]["description"].replace("Candidate type: NOT_MENTIONED", "Candidate type: OTHER")
    else:
        original["ordered_candidate_ids"][0].reverse()
    with pytest.raises(ValueError):
        transform_request(original)


def test_reusable_verifier_rejects_unrelated_context_or_previous_edit():
    original = packet()
    result = transform_request(original)
    result["tokens"] = [[3, 4, 5, 6]]
    assert verify_transform(original, result)
    result["request"]["context"] += " extra information"
    with pytest.raises(ValueError, match="prescribed"):
        verify_transform(original, result)
