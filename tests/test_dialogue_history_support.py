"""Tiny public-text fixtures only, without any corpus or learned model."""
import copy

import pytest

from openjev.research.dialogue_history_support import (
    AGES,
    BINS,
    DONTCARE,
    NONE,
    aggregate,
    audit_query,
)


def query(values=("Blue", "Red", "True", "False", "None"), *, service="service", qid="query"):
    return {"query_id": qid, "service": service, "slot": "slot", "candidates":
        [{"id": NONE, "value": None}, {"id": DONTCARE, "value": None}]
        +[{"id": "value:"+v, "value": v} for v in values]}


def dialogue(pairs, name="dialogue"):
    turns = []
    for system, user in pairs:
        if system is not None:
            turns.append({"speaker": "SYSTEM", "utterance": system})
        turns.append({"speaker": "USER", "utterance": user})
    users = [{"turn_index": i, "previous_system_turn_index": i-1 if i and turns[i-1]["speaker"] == "SYSTEM" else None}
             for i, t in enumerate(turns) if t["speaker"] == "USER"]
    return {"dialogue_id": name, "turns": turns, "user_turns": users}


def label(d, q, ordinal, cid, bin_name):
    return {"dialogue_id": d["dialogue_id"], "query_id": q["query_id"], "service": q["service"], "slot": q["slot"],
        "turn_index": d["user_turns"][ordinal]["turn_index"], "label_id": cid,
        "label_index": [c["id"] for c in q["candidates"]].index(cid), "bin": bin_name}


def delayed_fixture(value="Blue"):
    q = query()
    d = dialogue([(value, "hello"), ("other", "nothing"), (None, "again"), ("okay", "go on"), (None, "accepted")])
    labels = [label(d, q, 0, NONE, "unmentioned_retention"), label(d, q, 4, "value:"+value, "first_assignment")]
    return d, q, labels


def test_distant_system_only_and_unscored_endpoint_chronology():
    d, q, labels = delayed_fixture()
    rows = audit_query(d, q, labels)
    result = rows[1]
    assert result["system_age"] == 4 and result["system_age_category"] == "distant"
    assert result["user_age"] is None and result["user_age_category"] == "never"
    assert result["proxies"]["delayed_system_only"]
    assert result["proxies"]["changed_no_recent_combined_literal"]
    assert result["proxies"]["changed_distant_combined_literal"]
    assert not result["proxies"]["changed_never_combined_literal"]
    assert result["annotation_adjacency"] == "gapped" and not result["annotation_adjacent"]
    assert result["previous_annotated_user_index"] == 0 and result["user_index"] == 4
    changed = copy.deepcopy(d)
    changed["turns"][changed["user_turns"][3]["turn_index"]]["utterance"] = "BLUE please"
    new = audit_query(changed, q, labels)[1]
    assert new["user_age"] == 1 and new["user_age_category"] == "recent"
    assert not any(new["proxies"].values())


def test_future_public_mutation_and_labels_cannot_change_earlier_metadata():
    d = dialogue([("Blue", "nothing"), ("later", "hello"), ("Red", "Red")])
    q = query()
    labels = [label(d, q, 0, "value:Blue", "first_assignment"), label(d, q, 2, "value:Red", "revision")]
    before = copy.deepcopy((d, q, labels))
    expected = audit_query(d, q, labels)[0]
    assert (d, q, labels) == before
    d["turns"][-2]["utterance"] = "Blue Blue"
    d["turns"][-1]["utterance"] = "Blue"
    labels[-1] = label(d, q, 2, "value:Blue", "assigned_retention")
    assert audit_query(d, q, labels)[0] == expected


def test_system_immediately_before_user_and_consecutive_systems_age_zero():
    d = {"dialogue_id": "d", "turns": [{"speaker": "SYSTEM", "utterance": "Blue"},
        {"speaker": "SYSTEM", "utterance": "Red"}, {"speaker": "USER", "utterance": "yes"}]}
    q = query()
    rows = audit_query(d, q, [{"turn_index": 2, "label_id": "value:Blue", "label_index": 2, "bin": "first_assignment"}])
    assert rows[0]["system_age"] == 0 and rows[0]["system_last_mention_turn_index"] == 0
    assert rows[0]["user_age_category"] == "never"


@pytest.mark.parametrize("cid", [NONE, DONTCARE])
def test_reserved_states_are_nonliteral_even_when_their_names_appear(cid):
    q = query()
    d = dialogue([("NOT_MENTIONED DONTCARE", "NOT_MENTIONED DONTCARE")])
    result = audit_query(d, q, [label(d, q, 0, cid, "unmentioned_retention" if cid == NONE else "first_assignment")])[0]
    assert all(result[role+"_age_category"] == "nonliteral" and result[role+"_age"] is None
               for role in ("user", "system", "combined"))
    assert not any(result["proxies"].values())


@pytest.mark.parametrize("value", ["True", "False"])
def test_boolean_literals_have_ages_but_never_primary_proxies(value):
    d, q, labels = delayed_fixture(value)
    result = audit_query(d, q, labels)[1]
    assert result["system_age"] == 4 and result["target_type"] == value.lower()
    assert result["user_age_category"] == "never" and not any(result["proxies"].values())
    d["turns"][-1]["utterance"] = "yes, no, yeah, nope"
    assert audit_query(d, q, labels)[1]["user_age_category"] == "never"
    d["turns"][-1]["utterance"] = value.upper()
    assert audit_query(d, q, labels)[1]["user_age"] == 0


def test_literal_none_remains_other_and_is_distinct_from_reserved_none():
    d, q, labels = delayed_fixture("None")
    result = audit_query(d, q, labels)[1]
    assert result["target_type"] == "other" and result["label_id"] != NONE
    assert result["proxies"]["delayed_system_only"]


@pytest.mark.parametrize("value,text,matched", [
    ("  BLUE  ", "blue!", True), ("Blue", "blueberry", False), ("Blue", "_blue", False),
    ("a+b", "a+b?", True), ("a+b", "aaab", False), ("Straße", "STRASSE", True),
    ("new york", "New York", True), ("new york", "new  york", False), ("Blue", "éblue", False)])
def test_exact_literal_normalization_and_unicode_word_boundaries(value, text, matched):
    q = query((value,))
    d = dialogue([(None, text)])
    result = audit_query(d, q, [label(d, q, 0, "value:"+value, "first_assignment")])[0]
    assert (result["user_age"] == 0) is matched


def test_candidate_permutation_moves_indices_but_preserves_identity_and_results():
    d, q, labels = delayed_fixture()
    before = audit_query(d, q, labels)
    reordered = copy.deepcopy(q)
    reordered["candidates"].reverse()
    new_labels = copy.deepcopy(labels)
    for row in new_labels:
        row["label_index"] = [c["id"] for c in reordered["candidates"]].index(row["label_id"])
    after = audit_query(d, reordered, new_labels)
    for a, b in zip(before, after, strict=True):
        assert a["label_index"] != b["label_index"]
        assert {k: v for k, v in a.items() if k not in ("label_index", "previous_label_index")} == {
            k: v for k, v in b.items() if k not in ("label_index", "previous_label_index")}
    assert aggregate(before) == aggregate(after)


def test_all_five_bins_and_annotated_adjacency():
    d = dialogue([(None, "hello") for _ in range(5)])
    q = query()
    ids = (NONE, "value:Blue", "value:Blue", "value:Red", NONE)
    bins = ("unmentioned_retention", "first_assignment", "assigned_retention", "revision", "clear")
    rows = audit_query(d, q, [label(d, q, i, cid, b) for i, (cid, b) in enumerate(zip(ids, bins, strict=True))])
    assert rows[0]["annotation_adjacency"] == "first_annotation"
    assert all(row["annotation_adjacent"] for row in rows[1:])
    result = aggregate(rows)
    assert result["by_stratum"]["all"]["bins"] == dict.fromkeys(BINS, 1)
    assert result["by_stratum"]["changed"]["rows"] == 3 and result["by_stratum"]["retained"]["rows"] == 2
    cross = result["by_stratum"]["all"]["user_system_age"]
    assert sum(sum(v.values()) for v in cross.values()) == len(rows)
    assert set(cross) == set(AGES)


def test_proxy_counts_include_unique_dialogues_services_and_zero_retention_bins():
    d, q, labels = delayed_fixture()
    rows = audit_query(d, q, labels)
    q2 = query(qid="second-query")
    second_labels = [dict(v, query_id=q2["query_id"]) for v in labels]
    rows += audit_query(d, q2, second_labels)
    d2 = copy.deepcopy(d); d2["dialogue_id"] = "second-dialogue"
    q3 = query(service="second-service", qid="third-query")
    other_labels = [dict(v, dialogue_id=d2["dialogue_id"], query_id=q3["query_id"], service=q3["service"]) for v in labels]
    rows += audit_query(d2, q3, other_labels)
    result = aggregate(rows)
    support = result["proxy_support"]["delayed_system_only"]
    assert (support["rows"], support["unique_dialogues"], support["service_count"]) == (3, 2, 2)
    assert support["services"] == ["second-service", "service"]
    assert support["by_bin"]["first_assignment"]["rows"] == 3
    assert support["by_bin"]["assigned_retention"]["rows"] == 0
    assert sum(v["rows"] for v in result["by_service"].values()) == len(rows)
    assert sum(v["rows"] for v in result["by_type"].values()) == len(rows)


def test_other_distant_support_includes_retained_without_relabeling_it_changed():
    d, q, _ = delayed_fixture()
    labels = [label(d, q, 0, "value:Blue", "first_assignment"), label(d, q, 4, "value:Blue", "assigned_retention")]
    rows = audit_query(d, q, labels)
    assert not rows[1]["changed"] and rows[1]["proxies"]["other_distant_combined_literal"]
    assert not rows[1]["proxies"]["delayed_system_only"]
    support = aggregate(rows)["proxy_support"]["other_distant_combined_literal"]
    assert support["rows"] == support["by_bin"]["assigned_retention"]["rows"] == 1
    assert support["by_stratum"]["retained"]["unique_dialogues"] == 1
    assert support["by_stratum"]["changed"]["rows"] == 0


def test_normalized_literal_collisions_are_preserved_and_explicit():
    q = query(("Blue", " blue "))
    d = dialogue([(None, "BLUE"), (None, "blue")])
    labels = [label(d, q, 0, "value:Blue", "first_assignment"), label(d, q, 1, "value: blue ", "revision")]
    rows = audit_query(d, q, labels)
    assert all(r["user_age"] == 0 and r["target_literal_match_count"] == 2 for r in rows)
    assert [r["label_id"] for r in rows] == ["value:Blue", "value: blue "]
    assert aggregate(rows)["normalized_literal_collisions"] == {
        "rows_with_ambiguous_target": 2, "distinct_queries_with_collisions": 1, "dialogue_queries_with_collisions": 1}


@pytest.mark.parametrize("damage", ["duplicate", "order", "unsupported", "index", "bool_index", "system_turn", "bin", "identity"])
def test_invalid_labels_rejected(damage):
    d, q, labels = delayed_fixture()
    if damage == "duplicate":
        labels.append(copy.deepcopy(labels[-1]))
    elif damage == "order":
        labels.reverse()
    elif damage == "unsupported":
        labels[-1]["label_id"] = "value:missing"
    elif damage == "index":
        labels[-1]["label_index"] = 0
    elif damage == "bool_index":
        labels[0]["label_index"] = False
    elif damage == "system_turn":
        labels[0]["turn_index"] = 0
    elif damage == "bin":
        labels[-1]["bin"] = "revision"
    else:
        labels[-1]["query_id"] = "other-query"
    with pytest.raises(ValueError):
        audit_query(d, q, labels)


def test_wrong_public_index_candidate_values_and_duplicate_aggregate_fail():
    d, q, labels = delayed_fixture()
    broken = copy.deepcopy(d)
    broken["user_turns"][0]["turn_index"] = True
    with pytest.raises(ValueError):
        audit_query(broken, q, labels)
    broken_query = copy.deepcopy(q)
    broken_query["candidates"][2]["value"] = "Red"
    with pytest.raises(ValueError):
        audit_query(d, broken_query, labels)
    rows = audit_query(d, q, labels)
    with pytest.raises(ValueError, match="Duplicate aggregate"):
        aggregate(rows+rows)


def test_empty_annotation_set_retains_zero_denominators():
    d, q, _ = delayed_fixture()
    assert audit_query(d, q, []) == []
    result = aggregate([])
    assert result["rows"] == 0 and result["by_service"] == {}
    assert all(v["rows"] == v["unique_dialogues"] == v["service_count"] == 0 for v in result["proxy_support"].values())
