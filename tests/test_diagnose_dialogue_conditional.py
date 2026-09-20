"""Synthetic checks for the posthoc saved-decision diagnostic."""
import copy

import diagnose_dialogue_conditional as d
import numpy as np
import pytest


def fixture():
    queries = [{"query_id": "q0", "boolean_slot": True,
                "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"],
                "candidate_values": [None, None, "True", "False"]},
               {"query_id": "q1", "boolean_slot": False,
                "candidate_ids": ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:None"],
                "candidate_values": [None, None, "None"]}]
    rows = []
    for i, (target, previous, qi) in enumerate([(2, 0, 0), (1, 0, 0), (3, 0, 0), (2, 2, 0), (2, 0, 1)]):
        q = queries[qi]
        rows.append({"row_index": i, "split": "dev", "admission": "admitted", "query_index": qi,
                     "query_id": q["query_id"], "dialogue_id": "d"+str(i % 2), "candidate_count": len(q["candidate_ids"]),
                     "current_label_index": target, "previous_current_index": previous, "unseen": True,
                     "derived_bin": "assigned_retention" if target == previous else "first_assignment",
                     "current_candidate_id": q["candidate_ids"][target], "previous_candidate_id": q["candidate_ids"][previous],
                     "current_value_group": d.value_class(q["candidate_ids"][target], q["candidate_values"][target], q["boolean_slot"])})
    return rows, queries


def packet(rows, target_probability=.5):
    logs = np.full((len(rows), 12), -np.inf, np.float32)
    for i, row in enumerate(rows):
        n = row["candidate_count"]
        p = np.full(n, (1-target_probability)/(n-1))
        p[row["current_label_index"]] = target_probability
        logs[i, :n] = np.log(p)
    return {"log_probs": logs, "row_indices": np.asarray([r["row_index"] for r in rows], dtype=np.int64)}


def test_changed_prior_correct_and_other_are_exclusive_retained_is_not_copy_error():
    rows, queries = fixture()
    p = packet(rows)
    for i, choice in enumerate([0, 1, 2, 2, 2]):
        n = rows[i]["candidate_count"]
        values = np.full(n, .1/(n-1)); values[choice] = .9
        p["log_probs"][i, :n] = np.log(values)
    q = d.row_quantities(rows, d.classes_for_rows(rows, queries), p)
    result = d.describe(rows, q, np.ones(len(rows), bool))
    assert (result["correct_count"], result["prior_error_count"], result["other_wrong_count"]) == (3, 1, 1)
    assert result["changed_rows"] == 4 and result["retained_rows"] == 1
    assert result["wrong_choice_classes"] == {"none": 1, "dontcare": 0, "true": 1, "false": 0, "other": 0}
    retained = d.describe(rows, q, np.asarray([False, False, False, True, False]))
    assert retained["correct_count"] == 1 and retained["prior_error_count"] == 0
    assert retained["changed_mean_target_minus_prior_log_probability"] is None


def test_literal_none_is_distinct_and_schema_permutation_preserves_canonical_results():
    rows, queries = fixture()
    assert d.classes_for_rows(rows, queries)[-1] == ["none", "dontcare", "other"]
    p = packet(rows)
    original = d.row_quantities(rows, d.classes_for_rows(rows, queries), p)
    changed, qs, permuted = copy.deepcopy(rows), copy.deepcopy(queries), copy.deepcopy(p)
    order = np.array([2, 0, 3, 1])
    for key in ("candidate_ids", "candidate_values"):
        qs[0][key] = [qs[0][key][i] for i in order]
    for i, row in enumerate(changed):
        if row["query_index"] == 0:
            row["current_label_index"] = int(np.flatnonzero(order == row["current_label_index"])[0])
            row["previous_current_index"] = int(np.flatnonzero(order == row["previous_current_index"])[0])
            permuted["log_probs"][i, :4] = p["log_probs"][i, order]
    actual = d.row_quantities(changed, d.classes_for_rows(changed, qs), permuted)
    for key in original:
        assert np.array_equal(original[key], actual[key]), key


def test_underflow_preserves_direct_log_nll_and_margin():
    rows, queries = fixture(); rows = rows[:1]
    p = packet(rows); p["log_probs"][0, :4] = [0., -1000., -1000., -1000.]
    q = d.row_quantities(rows, d.classes_for_rows(rows, queries), p)
    assert q["target_probability"][0] == 0 and q["target_nll"][0] == 1000
    assert q["target_minus_prior_log_probability"][0] == -1000


def test_empty_cells_have_undefined_means_and_no_fabricated_errors():
    rows, queries = fixture()
    q = d.row_quantities(rows, d.classes_for_rows(rows, queries), packet(rows))
    result = d.describe(rows, q, np.zeros(len(rows), bool))
    assert result["rows"] == result["prior_error_count"] == 0
    assert result["mean_target_probability"] is None
    assert result["changed_mean_target_minus_prior_log_probability"] is None


def test_opposing_contributions_reconcile_and_events_are_not_unique_rows():
    rows, queries = fixture(); rows = rows[:2]
    predictions, quantities = {}, {}
    for mode in d.report.MODES:
        for seed in d.report.SEEDS:
            name = f"{mode}-{seed}"
            predictions[name] = packet(rows)
            if mode == "candidate":
                predictions[name]["log_probs"][0] = packet(rows[:1], .25)["log_probs"][0]
                predictions[name]["log_probs"][1] = packet(rows[1:], .75)["log_probs"][0]
            quantities[name] = d.row_quantities(rows, d.classes_for_rows(rows, queries), predictions[name])
    delta = float((quantities['candidate-5301']['target_nll']-quantities['slot-5301']['target_nll']).mean())
    published = {"continuation_allowed": False, "primary": {"rows": 2, "mean_candidate_minus_slot_nll": delta,
        "paired": [{"seed": seed, "candidate_minus_slot_nll": delta} for seed in d.report.SEEDS]}}
    train = [dict(row, split="train") for row in rows]
    result = d.analyze(train, rows, queries, predictions, published)
    values = result["primary_decomposition"]["groups"]
    assert values["true"]["mean_candidate_minus_slot_global_denominator_contribution"] > 0
    assert values["dontcare"]["mean_candidate_minus_slot_global_denominator_contribution"] < 0
    assert sum(v["mean_candidate_minus_slot_global_denominator_contribution"] for v in values.values()) == pytest.approx(delta)
    assert values["false"]["rows"] == 0 and values["false"]["seeds"][0]["conditional_mean_difference"] is None
    assert values["false"]["seeds"][0]["candidate_minus_slot_global_denominator_contribution"] == 0
    assert result["groups"]["unseen/changed"]["rows"] == 2
    assert result["groups"]["unseen/changed"]["prediction_events_across_nine_fits"] == 18
    assert result["original_continuation_allowed"] is False and result["new_primary_or_continuation_rule"] is False
    predictions.pop('mean-5301')
    with pytest.raises(ValueError, match="nine"):
        d.analyze(train, rows, queries, predictions, published)


def test_train_support_exhaustive_partition_and_unique_dialogues():
    rows, queries = fixture()
    rows = [dict(row, split="train") for row in rows]
    result = d.training_support(rows, d.classes_for_rows(rows, queries))
    assert result["rows"] == 5
    assert result["transition_value"]["first_assignment/true"]["rows"] == 1
    assert result["transition_value"]["assigned_retention/true"]["rows"] == 1
    assert result["previous_to_current_value"]["none/other"]["rows"] == 1
    assert result["previous_to_current_value"]["true/true"]["rows"] == 1
    rows[0]["split"] = "dev"
    with pytest.raises(ValueError, match="training"):
        d.training_support(rows, d.classes_for_rows(rows, queries))


def test_exact_top1_ties_are_counted_with_first_index_convention():
    rows, queries = fixture(); rows = rows[:1]
    p = packet(rows, .25)
    q = d.row_quantities(rows, d.classes_for_rows(rows, queries), p)
    assert q["top1_tied"][0] and q["prior_error"][0]
