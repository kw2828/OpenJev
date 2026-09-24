"""Fabricated proper-score, case-weighting and prospective conjunction tests."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from openjev.research import otto_action_latent_metrics as metrics


def inputs(blocks=2):
    return {"outcomes": np.zeros((blocks, 8), np.int64),
        "outcome_predictions": np.zeros((blocks, 8, 5), np.float64),
        "predicted_costs": np.zeros((blocks, 8, 4), np.float64),
        "raw_costs": np.broadcast_to([2., 1., 3., 4.], (blocks, 8, 4)).copy(),
        "legal": np.ones((blocks, 8, 4), np.bool_),
        "case_ids": [f"case{i}" for i in range(blocks)], "regimes": ["lambda3"] * blocks,
        "family": "candidate", "fit_seed": 1, "condition": "gap"}


def test_uniform_scores_and_exact_legal_tie_rule():
    value = inputs()
    value["legal"][:, :, 0] = False
    value["predicted_costs"][:, :, 0] = -1e6
    report = metrics.score(**value)
    assert report["chosen_actions"] == [[1] * 8] * 2
    for group in ("all", "short", "long"):
        leaf = report["groups"][group]["overall"]
        assert leaf["case_weighted_log_score"] == pytest.approx(math.log(5))
        assert leaf["case_weighted_brier"] == pytest.approx(.8)
        assert leaf["case_weighted_decision_gap"] == 0
    assert set(report["per_horizon"]) == set(map(str, range(1, 9)))
    assert report["groups"]["short"]["overall"]["horizons"] == [1, 2, 3, 4]
    assert report["groups"]["long"]["overall"]["horizons"] == [5, 6, 7, 8]


def test_stable_logits_are_not_clipped_and_equal_probability_scoring():
    value = inputs(1)
    value["outcome_predictions"][:] = [0., 1000., -1000., 0., 0.]
    report = metrics.score(**value)
    assert report["groups"]["all"]["overall"]["case_weighted_log_score"] == 1000
    assert report["groups"]["all"]["overall"]["case_weighted_brier"] == 2
    value["outcome_predictions"][:] = [.4, .2, .1, .2, .1]
    p = metrics.score(**value, prediction_kind="probabilities")
    value["outcome_predictions"] = np.log(value["outcome_predictions"])
    logits = metrics.score(**value)
    for name in ("case_weighted_log_score", "case_weighted_brier"):
        assert p["groups"]["all"]["overall"][name] == pytest.approx(logits["groups"]["all"]["overall"][name])


def test_case_weighting_is_not_block_weighting_and_no_survivor_is_undefined():
    value = inputs(4)
    value["case_ids"] = ["a", "a", "b", "c"]
    value["raw_costs"][:] = [0., 0., 0., 0.]
    value["raw_costs"][:2, :, 0] = 2
    value["raw_costs"][2, :, 0] = 8
    value["outcomes"][3] = 4
    value["legal"][3] = False
    report = metrics.score(**value)
    leaf = report["groups"]["all"]["overall"]
    assert leaf["declared_cases"] == 3 and leaf["blocks"] == 4
    assert leaf["supported_cases"] == 2 and leaf["unsupported_case_ids"] == ["c"]
    assert leaf["case_weighted_decision_gap"] == 5
    assert leaf["full_case_denominator_gap"] == pytest.approx(10 / 3)
    assert leaf["terminal_rows"] == 8 and leaf["decision_rows"] == 24
    assert report["chosen_actions"][3] == [-1] * 8
    value["outcomes"][:] = 4
    value["legal"][:] = False
    report = metrics.score(**value)
    leaf = report["groups"]["all"]["overall"]
    assert leaf["case_weighted_decision_gap"] is leaf["full_case_denominator_gap"] is None
    assert leaf["supported_cases"] == 0 and leaf["terminal_rows"] == 32
    assert leaf["case_weighted_log_score"] == pytest.approx(math.log(5))


def test_absorbed_suffix_has_outcome_loss_but_no_cost_score():
    value = inputs(1)
    value["outcomes"][0, 4:] = 4
    value["legal"][0, 4:] = False
    value["raw_costs"][0, 4:] = 1e200
    report = metrics.score(**value)
    long = report["groups"]["long"]["overall"]
    assert long["outcome_rows"] == long["terminal_rows"] == 4
    assert long["case_weighted_decision_gap"] is None
    assert long["case_weighted_log_score"] == pytest.approx(math.log(5))
    assert report["groups"]["short"]["overall"]["case_weighted_decision_gap"] == 1


@pytest.mark.parametrize("field", ["outcome_predictions", "raw_costs"])
def test_finite_but_overflowing_arithmetic_is_not_silently_clipped(field):
    value = inputs(1)
    value[field][0, 0, 0] = 1e308
    value[field][0, 0, 1] = -1e308
    with pytest.raises(ValueError, match="finite.*arithmetic"):
        metrics.score(**value)


def test_inputs_and_rng_unchanged_and_additive_cost_gauge_invariant():
    value = inputs()
    copies = {k: v.copy() for k, v in value.items() if isinstance(v, np.ndarray)}
    for item in copies:
        value[item].flags.writeable = False
    rng = np.random.get_state()
    before = metrics.score(**value)
    for name, original in copies.items():
        np.testing.assert_array_equal(value[name], original)
    after_rng = np.random.get_state()
    assert rng[0] == after_rng[0] and rng[2:] == after_rng[2:]
    np.testing.assert_array_equal(rng[1], after_rng[1])
    value["predicted_costs"] = value["predicted_costs"] + 10
    after = metrics.score(**value)
    assert before == after


@pytest.mark.parametrize("change", ["float_target", "unknown_class", "nonabsorbing", "nonterminal_no_legal",
    "terminal_legal", "nan_masked_cost", "nan_logits", "infinite_teacher", "shape", "boolean_cost",
    "empty", "case_regime_conflict", "boolean_seed", "wrong_condition", "probability_sum", "probability_zero"])
def test_invalid_contract_fails_explicitly(change):
    value = inputs()
    if change == "float_target":
        value["outcomes"] = value["outcomes"].astype(float)
    elif change == "unknown_class":
        value["outcomes"][0, 0] = 5
    elif change == "nonabsorbing":
        value["outcomes"][0, 0] = 4
    elif change == "nonterminal_no_legal":
        value["legal"][0, 0] = False
    elif change == "terminal_legal":
        value["outcomes"][0] = 4
    elif change == "nan_masked_cost":
        value["outcomes"][0] = 4
        value["legal"][0] = False
        value["predicted_costs"][0, 0, 0] = np.nan
    elif change == "nan_logits":
        value["outcome_predictions"][0, 0, 0] = np.nan
    elif change == "infinite_teacher":
        value["raw_costs"][0, 0, 0] = np.inf
    elif change == "shape":
        value["predicted_costs"] = value["predicted_costs"][:, :7]
    elif change == "boolean_cost":
        value["predicted_costs"] = value["predicted_costs"].astype(bool)
    elif change == "empty":
        value = inputs(0)
    elif change == "case_regime_conflict":
        value["case_ids"] = ["same", "same"]
        value["regimes"] = ["lambda3", "lambda4"]
    elif change == "boolean_seed":
        value["fit_seed"] = True
    elif change == "wrong_condition":
        value["condition"] = "test"
    else:
        value["prediction_kind"] = "probabilities"
        value["outcome_predictions"][:] = [.0, .25, .25, .25, .25] if change == "probability_zero" else [.5] * 5
    with pytest.raises(ValueError):
        metrics.score(**value)


def gate_fixture():
    reports = []
    for condition in ("gap", "normal"):
        for family in ("candidate", "action_blind", "direct_horizon", "ridge"):
            for seed in (1, 2, 3):
                value = inputs()
                value.update(condition=condition, family=family, fit_seed=seed,
                             regimes=["lambda3", "lambda4"])
                if family == "candidate":
                    value["outcome_predictions"][:, :, 0] = 1
                    value["predicted_costs"][:, :, 1] = -1
                reports.append(metrics.score(**value))
    options = {"candidate": "candidate", "controls": ("action_blind", "direct_horizon", "ridge"),
        "fit_seeds": (1, 2, 3), "regimes": ("lambda3", "lambda4"),
        "thresholds": {"long_log_relative_gain": .01, "long_gap_relative_gain": .05,
                       "normal_log_relative_tolerance": .01, "normal_gap_relative_tolerance": .01,
                       "minimum_supported_cases": 1}}
    return reports, options


def test_gate_requires_all_seeds_regimes_controls_and_preserves_input():
    reports, options = gate_fixture()
    original = copy.deepcopy(reports)
    result = metrics.evaluate_reports(reports, **options)
    assert result["passed"] is True and result["admits_execution"] is False
    assert result["passed_cells"] == result["total_cells"] == 18
    assert all(len(c["conditions"]) == 6 for c in result["cells"])
    assert original == reports
    changed = next(r for r in reports if r["condition"] == "gap" and r["family"] == "candidate" and r["fit_seed"] == 3)
    changed["groups"]["long"]["by_regime"]["lambda4"]["case_weighted_decision_gap"] = 2
    failed = metrics.evaluate_reports(reports, **options)
    assert failed["passed"] is False and failed["passed_cells"] == 15
    assert all(c["fit_seed"] == 3 and c["regime"] == "lambda4" for c in failed["cells"] if not c["passed"])


def test_gate_exact_relative_boundaries_normal_regression_and_zero_ties():
    reports, options = gate_fixture()
    for report in reports:
        for regime in options["regimes"]:
            leaf = report["groups"]["long" if report["condition"] == "gap" else "all"]["by_regime"][regime]
            candidate = report["family"] == "candidate"
            normal = report["condition"] == "normal"
            leaf["case_weighted_log_score"] = (1.01 if normal else .99) if candidate else 1.
            leaf["case_weighted_decision_gap"] = (1.01 if normal else .95) if candidate else 1.
    assert metrics.evaluate_reports(reports, **options)["passed"] is True
    report = next(r for r in reports if r["condition"] == "normal" and r["family"] == "candidate")
    report["groups"]["all"]["by_regime"]["lambda3"]["case_weighted_log_score"] = 1.01000001
    assert metrics.evaluate_reports(reports, **options)["passed"] is False
    for report in reports:
        if report["condition"] == "gap":
            for regime in options["regimes"]:
                report["groups"]["long"]["by_regime"][regime]["case_weighted_decision_gap"] = 0.
    result = metrics.evaluate_reports(reports, **options)
    assert not any(c["passed"] for c in result["cells"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "targets", "cases", "nonfinite", "no_survivor", "thresholds"])
def test_gate_missing_or_unpaired_evidence_is_never_promoted(change):
    reports, options = gate_fixture()
    if change == "missing":
        reports.pop()
    elif change == "duplicate":
        reports[-1] = reports[0]
    elif change == "targets":
        reports[-1]["target_sha256"] = "different"
    elif change == "cases":
        reports[-1]["identity_manifest"][0]["case_id"] = "different"
    elif change == "thresholds":
        options["thresholds"].pop("long_log_relative_gain")
    elif change == "nonfinite":
        reports[-1]["groups"]["all"]["by_regime"]["lambda3"]["case_weighted_log_score"] = math.nan
    else:
        for report in reports:
            for name in ("long", "all"):
                for regime in options["regimes"]:
                    leaf = report["groups"][name]["by_regime"][regime]
                    leaf.update(supported_cases=0, unsupported_case_ids=["case0" if regime == "lambda3" else "case1"],
                                case_weighted_decision_gap=None, decision_rows=0)
        result = metrics.evaluate_reports(reports, **options)
        assert result["passed"] is False and result["passed_cells"] == 0
        return
    with pytest.raises(ValueError):
        metrics.evaluate_reports(reports, **options)
