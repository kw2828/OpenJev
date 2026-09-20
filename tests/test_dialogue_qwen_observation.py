import copy

import pytest

from openjev.decisions import messages_for
from openjev.research.dialogue_qwen_observation import (
    candidate_type,
    make_question,
    pilot_ids,
    public_context,
    question_from_metadata,
)


def query():
    return {"query_text": "Service: travel\nSlot: meal", "candidates": [
        {"id": "reserved:NOT_MENTIONED", "text": "Value: NOT_MENTIONED (no constraint stated)"},
        {"id": "reserved:DONTCARE", "text": "Value: DONTCARE (no preference)"},
        {"id": "value:None", "text": "Value: None"},
        {"id": "value:two meals", "text": "Value: two meals"},
    ]}


def test_candidate_bijection_preserves_reserved_and_spaced_values():
    q, mapping = make_question(row_index=7, query=query(), previous_candidate_id="value:two meals",
                               flags=[[0] * 10 for _ in range(4)])
    _, ordered = messages_for("context", q)
    assert set(mapping.values()) == {x["id"] for x in query()["candidates"]}
    assert [mapping[c.id] for c in ordered] == ["reserved:DONTCARE", "reserved:NOT_MENTIONED",
                                              "value:None", "value:two meals"]
    assert "Value: two meals" in q.question


def test_candidate_permutation_retains_description_to_identity_and_flags():
    cat = query()
    flags = [[int(i == j) for j in range(10)] for i in range(4)]
    def build(c, f):
        q, m = make_question(row_index=1, query=c, previous_candidate_id="value:None", flags=f)
        return [(x.description, m[x.id]) for x in messages_for("same", q)[1]]
    reference = build(cat, flags)
    perm = [3, 1, 0, 2]
    changed = {**cat, "candidates": [cat["candidates"][i] for i in perm]}
    assert build(changed, [flags[i] for i in perm]) == reference


def test_boolean_case_and_literal_none_are_distinct_from_reserved_types():
    assert [candidate_type(x) for x in ("value:True", "value:False", "value:true", "value:None",
                                       "reserved:NOT_MENTIONED", "reserved:DONTCARE")] == [
        "TRUE", "FALSE", "TRUE", "OTHER", "NOT_MENTIONED", "DONTCARE"]


def test_context_uses_public_chronology_and_excludes_future_annotations():
    turns = []
    for i in range(6):
        turns.extend([{"speaker": "SYSTEM", "utterance": f"s{i}", "frames": "SECRET"},
                      {"speaker": "USER", "utterance": f"u{i}", "state": "TARGET"}])
    d = {"turns": turns, "user_turns": "DO NOT USE SCORED INDEX"}
    current = public_context(d, 4, "current")
    history = public_context(d, 4, "history4")
    assert all(x not in current for x in ["s3", "u3", "SECRET", "TARGET", "u5"])
    assert all(x in current for x in ["s4", "u4"])
    assert all(x in history for x in ["u1", "u2", "u3", "u4"])
    assert all(x not in history for x in ["u0", "u5", "SECRET", "TARGET"])
    d2 = copy.deepcopy(d); d2["turns"][-1]["utterance"] = "future mutation"
    assert public_context(d2, 4, "history4") == history


def test_no_current_truth_prompt_argument():
    with pytest.raises(TypeError):
        make_question(row_index=1, query=query(), previous_candidate_id="value:None",
                      flags=[[0] * 10 for _ in range(4)], current_candidate_id="value:two meals")


def test_mixed_row_target_poison_is_ignored_but_allowed_previous_changes():
    row = {"row_index": 1, "previous_candidate_id": "value:None", "current_candidate_id": "value:None",
           "current_label_index": 2, "derived_bin": "retention", "frames": {"target": "SECRET"}}
    flags = [[0] * 10 for _ in range(4)]
    baseline = question_from_metadata(row, query(), flags)
    poisoned = {**row, "current_candidate_id": "value:two meals", "current_label_index": 3,
                "derived_bin": "revision", "frames": {"target": "ANOTHER_SECRET"}}
    assert question_from_metadata(poisoned, query(), flags) == baseline
    changed = question_from_metadata({**row, "previous_candidate_id": "value:two meals"}, query(), flags)
    assert changed[0].question != baseline[0].question
    assert changed[0].candidates == baseline[0].candidates


@pytest.mark.parametrize("previous,flags", [("absent", [[0] * 10 for _ in range(4)]),
                                          ("value:None", [[0] * 9 for _ in range(4)]),
                                          ("value:None", [[float('nan')] * 10 for _ in range(4)])])
def test_invalid_public_input_rejected(previous, flags):
    with pytest.raises(ValueError):
        make_question(row_index=1, query=query(), previous_candidate_id=previous, flags=flags)


def test_pilot_uses_first_groups_and_longest_with_all_chunks():
    rows = [{"request_id": f"{a}-{i}-{j}", "arm": a, "dialogue_id": f"d{i:02d}", "time": 0,
             "tokens": [[0] * (50 if i == 15 else 5)]}
            for i in range(16) for a in ("current", "history4") for j in range(2)]
    result = pilot_ids(rows)
    assert len(result) == 13 * 4
    assert all("-15-" in x or int(x.split('-')[1]) < 12 for x in result)
