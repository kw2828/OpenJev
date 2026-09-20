"""Artificial canonical endpoints and saved-format arrays, never real task data."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from openjev.research import dialogue_observation_metrics as m


def row(i, *, unseen=True, bin_name="first_assignment", label=2, values=("True", "False"), service=None):
    service = service or ("new" if unseen else "seen")
    ids = [m.NONE, m.DONTCARE, *["value:" + value for value in values]]
    return {"row_index": i, "split": "dev", "dialogue_id": service + "-dialogue",
            "source_row_index": i, "time": i, "query_index": 1 if unseen else 0,
            "service": service, "slot": "slot", "candidate_ids": ids,
            "candidate_values": [None, None, *values], "label_index": label,
            "label_id": ids[label], "bin": bin_name, "unseen": unseen}


def logs_for(rows, probabilities):
    result = np.full((len(rows), 12), -np.inf, np.float32)
    for i, (r, p) in enumerate(zip(rows, probabilities, strict=True)):
        assert len(p) == len(r["candidate_ids"])
        result[i, :len(p)] = np.log(np.asarray(p, np.float64)).astype(np.float32)
    return result


def test_closed_form_scores_macro_is_stratum_balanced_not_endpoint_balanced():
    rows = [row(i, bin_name="unmentioned_retention", label=0) for i in range(9)]
    rows += [row(9, bin_name="assigned_retention", label=2), row(10, label=3)]
    logs = logs_for(rows, [[.7, .1, .1, .1]] * 11)
    result = m.score_panels(logs, rows, row_indices=np.arange(11, dtype=np.int64))
    panel = result["panels"]["unseen"]
    assert panel["micro"]["correct"] == 9 and panel["micro"]["accuracy"] == 9 / 11
    assert panel["macro_three"]["accuracy"] == 1 / 3
    target_losses = [-float(logs[i, r["label_index"]]) for i, r in enumerate(rows)]
    assert panel["micro"]["nll"] == math.fsum(target_losses) / 11
    expected_brier = (9 * .12 + 2 * 1.32) / 11
    assert panel["micro"]["brier"] == pytest.approx(expected_brier, abs=1e-7)
    assert panel["bins"]["clear"]["count"] == 0 and panel["bins"]["clear"]["nll"] is None
    assert result["panels"]["seen"]["macro_three"]["accuracy"] is None
    assert result["services"]["new"]["metrics"] == panel


def test_changed_pools_endpoints_instead_of_averaging_transition_bins():
    rows = [row(i, bin_name="first_assignment") for i in range(3)] + [row(3, bin_name="revision")]
    logs = logs_for(rows, [[.1, .1, .7, .1]] * 3 + [[.7, .1, .1, .1]])
    panel = m.score_panels(logs, rows)["panels"]["unseen"]
    assert panel["strata"]["changed"]["accuracy"] == .75
    assert panel["bins"]["first_assignment"]["accuracy"] == 1
    assert panel["bins"]["revision"]["accuracy"] == 0


def test_finite_log_underflow_keeps_direct_nll_and_no_probability_repair():
    rows = [row(0, label=2)]
    logs = np.full((1, 12), -np.inf, np.float32)
    logs[0, :4] = [0., -1000., -1000., -1000.]
    result = m.score_panels(logs, rows)
    assert result["panels"]["all"]["micro"]["nll"] == 1000.
    assert result["panels"]["all"]["micro"]["brier"] == 2.
    assert result["validation"]["float64_exponent_underflow_positions"] == 3
    logs[0, 0] = np.float32(1e-7)
    result = m.score_panels(logs, rows)
    assert result["panels"]["all"]["micro"]["brier"] > 2.  # No renormalization.


@pytest.mark.parametrize("mutation", ["supported_neginf", "supported_nan", "supported_posinf",
                                     "padding", "mass", "dtype", "row_order"])
def test_invalid_distribution_or_saved_alignment_is_technical_failure(mutation):
    rows = [row(0), row(1)]
    logs = logs_for(rows, [[.1, .1, .7, .1]] * 2)
    indices = np.arange(2, dtype=np.int64)
    if mutation == "supported_neginf": logs[0, 0] = -np.inf
    elif mutation == "supported_nan": logs[0, 0] = np.nan
    elif mutation == "supported_posinf": logs[0, 0] = np.inf
    elif mutation == "padding": logs[0, 4] = -1000.
    elif mutation == "mass": logs[0, :4] -= 1
    elif mutation == "dtype": logs = logs.astype(np.float64)
    else: indices = indices[::-1]
    with pytest.raises(ValueError):
        m.validate_log_predictions(logs, indices, rows)


def test_canonical_first_tie_and_candidate_permutation_use_exact_ids():
    rows = [row(0, label=0)]
    logs = logs_for(rows, [[.4, .4, .1, .1]])
    score = m.score_panels(logs, rows)
    assert score["panels"]["all"]["micro"]["correct"] == 1
    assert score["validation"]["exact_top1_tie_rows"] == 1
    perm = [2, 0, 3, 1]
    other = copy.deepcopy(rows)
    other[0]["candidate_ids"] = [rows[0]["candidate_ids"][i] for i in perm]
    other[0]["candidate_values"] = [rows[0]["candidate_values"][i] for i in perm]
    other[0]["label_index"] = perm.index(0)
    moved = logs.copy()
    moved[0, :4] = logs[0, perm]
    assert m.score_panels(moved, other)["panels"] == score["panels"]


def test_boolean_and_reserved_type_supports_with_supported_negative_denominators():
    rows = [row(0, label=2), row(1, label=3), row(2, label=0), row(3, label=1),
            row(4, unseen=False, values=("True", "False", "None"), label=2)]
    logs = logs_for(rows, [[.1, .1, .7, .1], [.1, .1, .7, .1], [.1, .1, .7, .1],
                           [.1, .7, .1, .1], [.1, .1, .6, .1, .1]])
    types = m.score_panels(logs, rows)["panels"]["all"]["types"]
    true = types["true"]
    assert (true["count"], true["correct"], true["predicted_support"], true["true_positives"]) == (1, 1, 3, 1)
    assert (true["false_positives"], true["false_positive_denominator"]) == (2, 3)
    assert true["false_positive_rate"] == 2 / 3  # Mixed ontology is not a Boolean slot.
    assert types["other"]["count"] == 1 and types["other"]["correct"] == 1
    assert types["dontcare"]["count"] == 1 and types["dontcare"]["correct"] == 1
    assert types["false"]["predicted_support"] == 0 and types["false"]["type_recall"] == 0
    assert true["bins"]["first_assignment"]["count"] == 1
    only_none = m.score_panels(logs[:1], [row(0, label=0)])["panels"]["all"]["types"]
    assert only_none["true"]["count"] == 0 and only_none["true"]["accuracy"] is None


def test_canonical_endpoint_corruption_rejects_without_inferred_repairs():
    rows = [row(0), row(1)]
    logs = logs_for(rows, [[.1, .1, .7, .1]] * 2)
    for key, bad in (("row_index", True), ("split", "train"), ("source_row_index", 0),
                     ("time", 0), ("label_id", m.NONE), ("unseen", 1)):
        corrupt = copy.deepcopy(rows)
        corrupt[1][key] = bad
        with pytest.raises(ValueError):
            m.score_panels(logs, corrupt)


def test_literal_references_have_no_probabilistic_forecasts_or_dontcare_writes():
    rows = [row(0, label=2), row(1, label=0, bin_name="unmentioned_retention")]
    indices = np.arange(2, dtype=np.int64)
    result = m.literal_metrics(np.asarray([2, 0], np.int64), indices, rows)
    assert result["panels"]["all"]["micro"]["accuracy"] == 1.
    assert "nll" not in result["panels"]["all"]["micro"]
    with pytest.raises(ValueError, match="DONTCARE"):
        m.literal_metrics(np.asarray([1, 0], np.int64), indices, rows)


def gate_fixture():
    def panel(hits):
        cell = {"count": 200, "correct": hits, "incorrect": 200 - hits,
                "accuracy": hits / 200, "error": (200 - hits) / 200, "nll": .5, "brier": .4}
        return {"micro": {**cell, "count": 600, "correct": 3 * hits, "incorrect": 3 * (200 - hits)},
                "strata": {s: copy.deepcopy(cell) for s in m.STRATA},
                "macro_three": {"accuracy": hits / 200, "nll": .5, "brier": .4}}
    fits = {}
    for arm in m.ARMS:
        for seed in m.SEEDS:
            seen = panel(100)
            unseen = panel(102 if arm == "trainable_numbers" else 100)
            all_panel = copy.deepcopy(seen)
            all_panel["micro"]["count"] = 1200
            fits[f"{arm}-{seed}"] = {"panels": {"all": all_panel, "seen": seen, "unseen": unseen},
                                    "validation": {"rows": 1200}}
    return fits


def check(result, name):
    return next(c for c in result["checks"] if c["name"] == name)


def test_seven_exact_checks_include_inclusive_gain_and_retention_boundaries():
    fits = gate_fixture()
    for seed in m.SEEDS:
        panel = fits[f"trainable_numbers-{seed}"]["panels"]["seen"]
        for stratum, hits in zip(m.STRATA, [97, 99, 98], strict=True):
            panel["strata"][stratum]["correct"] = hits
    result = m.criteria(fits)
    assert result["passed"] and result["checks_total"] == result["checks_passed"] == 7
    assert check(result, "unseen_macro_gain_1pp")["mean"] == .01
    assert check(result, "seen_macro_deficit_at_most_1pp")["mean"] == -.01
    assert check(result, "seen_assigned_retention_error_increase_at_most_half_pp")["mean"] == .005
    fits["trainable_numbers-6901"]["panels"]["seen"]["strata"]["assigned_retention"]["correct"] = 98
    assert not check(m.criteria(fits), "seen_assigned_retention_error_increase_at_most_half_pp")["passed"]


def test_one_winning_seed_cannot_pass_even_with_large_mean_gain():
    fits = gate_fixture()
    for seed, hits in zip(m.SEEDS, [120, 100, 100], strict=True):
        for cell in fits[f"trainable_numbers-{seed}"]["panels"]["unseen"]["strata"].values():
            cell["correct"] = hits
    result = m.criteria(fits)
    assert check(result, "unseen_macro_gain_1pp")["passed"]
    assert not check(result, "unseen_macro_strict_paired_wins")["passed"] and not result["passed"]


def test_nll_means_compared_directly_without_epsilon_or_selecting_good_seeds():
    fits = gate_fixture()
    values = [math.nextafter(.5, math.inf)] * 3
    for seed, value in zip(m.SEEDS, values, strict=True):
        fits[f"trainable_numbers-{seed}"]["panels"]["unseen"]["micro"]["nll"] = value
    result = m.criteria(fits)
    assert not check(result, "unseen_micro_nll_nonworse")["passed"]
    for seed in m.SEEDS:
        fits[f"trainable_numbers-{seed}"]["panels"]["unseen"]["micro"]["nll"] = .5
    assert check(m.criteria(fits), "unseen_micro_nll_nonworse")["passed"]


def test_missing_denominator_unevaluable_and_missing_fit_is_technical_failure():
    fits = gate_fixture()
    for fit in fits.values():
        fit["panels"]["unseen"]["strata"]["assigned_retention"].update(count=0, correct=0)
    result = m.criteria(fits)
    assert not result["passed"]
    assert check(result, "unseen_macro_gain_1pp")["mean"] is None
    assert check(result, "unseen_assigned_retention_error_increase_at_most_half_pp")["mean"] is None
    fits.pop("frozen_original-6901")
    with pytest.raises(ValueError, match="Exactly all four arms"):
        m.criteria(fits)
    with pytest.raises(ValueError, match="Exactly all four arms"):
        m.factorial_summary(fits)


def test_factorial_contrasts_and_interaction_have_fixed_paired_signs():
    fits = gate_fixture()
    by_arm = dict(zip(m.ARMS, [.2, .3, .4, .8], strict=True))
    for arm in m.ARMS:
        for j, seed in enumerate(m.SEEDS):
            fits[f"{arm}-{seed}"]["panels"]["unseen"]["micro"]["accuracy"] = by_arm[arm] + .01 * j
    result = m.factorial_summary(fits)
    primary = result["contrasts"]["encoder_numbers"]
    assert primary["mean"]["panels"]["unseen"]["micro"]["accuracy"] == pytest.approx(.5)
    assert result["interaction"]["mean"]["panels"]["unseen"]["micro"]["accuracy"] == pytest.approx(.3)
    assert set(primary["seeds"]) == {"6901", "6902", "6903"}
    assert result["families"]["trainable_numbers"]["mean"]["panels"]["unseen"]["micro"]["accuracy"] == pytest.approx(.81)


def test_all_twelve_actual_synthetic_saved_packets_integrate_and_leave_inputs_unchanged():
    rows = [row(i, unseen=i >= 3, bin_name=m.STRATA[i % 3] if i % 3 < 2 else "revision",
                label=0 if i % 3 == 0 else 2) for i in range(6)]
    logs = logs_for(rows, [[.1, .1, .7, .1]] * 6)
    before = logs.copy()
    fits = {f"{arm}-{seed}": m.score_panels(logs, rows, row_indices=np.arange(6, dtype=np.int64))
            for arm in m.ARMS for seed in m.SEEDS}
    result = m.criteria(fits)
    assert not result["passed"] and result["checks_passed"] == 5
    factorial = m.factorial_summary(fits)
    assert factorial["interaction"]["mean"]["panels"]["unseen"]["micro"]["nll"] == 0.
    assert factorial["interaction"]["mean"]["panels"]["unseen"]["bins"]["clear"]["nll"] is None
    np.testing.assert_array_equal(logs, before)
