"""Independent scalar oracles for the prospective development/confirmation rule."""
from copy import deepcopy

import numpy as np
import pytest

from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_residual_contract as contract
from openjev.research import otto_residual_gate as gate


def fixture(stage="dev", selected_tau=None, supported_cases=None):
    cases = 3 if stage == "dev" else 6
    support = cases if supported_cases is None else supported_cases
    metric_stage = "dev" if stage == "dev" else "test"
    identities = []
    for regime_index, regime in enumerate(("lambda3", "lambda4")):
        for case in range(cases):
            for arm in ("analytic", "neural", "period4_hold"):
                identities.append({"stage": metric_stage, "episode_id": f"fabricated:{regime}:{case}:{arm}",
                                   "episode_index": len(identities), "seed": 10 + regime_index * 100 + case,
                                   "case": case, "regime": regime, "arm": arm})
    manifest = [{**row, "length": 9 if row["case"] < support else 1, "target_sha256": "0" * 64}
                for row in identities]
    reports = []
    for method, tau, seed in contract.view_specs(stage, selected_tau):
        value = 1. if method == "rls_full" else 10.
        if support == 0:
            value = 0.
        leaf = {"episodes": 3 * cases, "declared_case_count": cases,
                "supported_case_count": support, "case_weighted_raw_gap": value}
        reports.append({"version": metrics.VERSION, "family": contract.view_name(method, tau), "seed": seed,
                        "stage": metric_stage, "query_period": 4, "episodes": len(identities),
                        "identity_manifest": deepcopy(manifest),
                        "scopes": {scope: {"by_regime": {r: deepcopy(leaf) for r in ("lambda3", "lambda4")}}
                                   for scope in ("full", "later")}})
    return reports, identities


def set_gap(reports, family, value, *, scope="later", regime=None, seed=None):
    for row in reports:
        if row["family"] != family or seed is not None and row["seed"] != seed:
            continue
        for r in ([regime] if regime else ("lambda3", "lambda4")):
            row["scopes"][scope]["by_regime"][r]["case_weighted_raw_gap"] = value


@pytest.mark.parametrize("stage,tau,views", [("dev", None, 72), ("confirm", .1, 27)])
def test_all_conditions_and_original_completion_are_required(stage, tau, views):
    reports, identities = fixture(stage, tau)
    result = gate.evaluate(reports, identities, stage=stage, selected_tau=tau)
    assert result["passed_conditions"] == 12 and not result["passed"]
    complete = gate.evaluate(reports, identities, stage=stage, selected_tau=tau, technical_complete=True)
    assert complete["passed"] and complete["passed_conditions"] == complete["total_conditions"] == 13
    assert complete["reports"] == views
    assert complete["selected_tau"] == (.01 if stage == "dev" else tau)


def test_minimax_choice_uses_both_settings_and_only_unattenuated_candidate():
    reports, identities = fixture()
    values = {0.01: (1., 9.), .1: (4., 4.), 1.: (3., 5.), 10.: (2., 8.)}
    for tau, pair in values.items():
        for regime, value in zip(("lambda3", "lambda4"), pair, strict=True):
            set_gap(reports, contract.view_name("rls_full", tau), value, regime=regime)
    # An excellent nuisance control cannot choose the candidate's ratio.
    set_gap(reports, contract.view_name("rls_shrink_025", 10.), 0.)
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert result["selected_tau"] == .1
    assert [row["objective"] for row in result["selection"]["grid"]] == [9., 4., 5., 8.]
    assert result["passed"]
    assert len(result["mechanistic_contrasts"]) == 16


def test_exact_ties_use_ascending_grid_and_do_not_promote_controls():
    reports, identities = fixture()
    for method in contract.RLS_METHODS[1:]:
        for tau in contract.TAUS:
            set_gap(reports, contract.view_name(method, tau), 0.)
    result = gate.evaluate(list(reversed(reports)), identities, stage="dev", technical_complete=True)
    assert result["candidate"] == "rls_full" and result["selected_tau"] == .01
    assert result["passed"]
    assert not any(row.get("confirmed_with_usefulness", False) for row in result["mechanistic_contrasts"])


def test_joint_aux_remains_a_strongest_control_and_can_veto_continuation():
    reports, identities = fixture()
    set_gap(reports, "joint_aux", .5, regime="lambda3")
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    failed = [row["name"] for row in result["conditions"] if not row["passed"]]
    assert failed == ["lambda3:P4:later_gap_10pct", *[f"lambda3:P4:seed_{s}_nonregression" for s in contract.FIT_SEEDS]]


def test_paired_regression_cannot_hide_behind_improved_mean():
    reports, identities = fixture()
    for tau in contract.TAUS:
        set_gap(reports, contract.view_name("rls_full", tau), 11., seed=contract.FIT_SEEDS[-1])
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert all(row["passed"] for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))
    assert [row["name"] for row in result["conditions"] if not row["passed"]] == [
        f"lambda3:P4:seed_{contract.FIT_SEEDS[-1]}_nonregression", f"lambda4:P4:seed_{contract.FIT_SEEDS[-1]}_nonregression"]


@pytest.mark.parametrize("value,passes", [(9., True), (float(np.nextafter(9., np.inf)), False)])
def test_ten_percent_boundary_has_no_tolerance(value, passes):
    reports, identities = fixture()
    for tau in contract.TAUS:
        set_gap(reports, contract.view_name("rls_full", tau), value)
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert all(row["passed"] is passes for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


def test_full_guard_is_independent_of_later_gain():
    reports, identities = fixture()
    set_gap(reports, contract.view_name("rls_full", .01), 12., scope="full", regime="lambda4")
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert [row["name"] for row in result["conditions"] if not row["passed"]] == ["lambda4:P4:full_gap_nonregression"]


@pytest.mark.parametrize("stage,support,passes", [("dev", 1, False), ("dev", 2, True),
                                                ("confirm", 3, False), ("confirm", 4, True)])
def test_support_uses_originating_cases_and_retains_full_denominators(stage, support, passes):
    tau = None if stage == "dev" else .1
    reports, identities = fixture(stage, tau, support)
    result = gate.evaluate(reports, identities, stage=stage, selected_tau=tau, technical_complete=True)
    assert all(row["passed"] is passes for row in result["conditions"] if row["name"].endswith("supported_cases"))


def test_zero_best_control_and_zero_support_cannot_pass():
    reports, identities = fixture(supported_cases=0)
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert not result["passed"]
    assert all(not row["passed"] for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


def test_covariance_contrast_can_fail_while_usefulness_passes():
    reports, identities = fixture("confirm", 1.)
    set_gap(reports, contract.view_name("rls_diagonal", 1.), 1.)
    result = gate.evaluate(reports, identities, stage="confirm", selected_tau=1., technical_complete=True)
    assert result["passed"]
    diag = result["mechanistic_contrasts"][0]
    assert len(diag["conditions"]) == 10
    assert not diag["contrast_passed"] and not diag["confirmed_with_usefulness"]


@pytest.mark.parametrize("technical,ordinary_gap", [(False, 10.), (True, 0.)])
def test_covariance_contrast_cannot_rescue_failed_usefulness_or_completion(technical, ordinary_gap):
    reports, identities = fixture("confirm", 1.)
    set_gap(reports, "joint_aux", ordinary_gap)
    result = gate.evaluate(reports, identities, stage="confirm", selected_tau=1., technical_complete=technical)
    assert not result["passed"]
    diag = result["mechanistic_contrasts"][0]
    assert diag["contrast_passed"] and not diag["confirmed_with_usefulness"]


@pytest.mark.parametrize("length,later_support,full_support", [(1, 0, 0), (2, 0, 3), (5, 0, 3), (6, 3, 3)])
def test_support_boundary_excludes_queries_and_first_episode_row(length, later_support, full_support):
    reports, identities = fixture()
    for report in reports:
        for row in report["identity_manifest"]:
            row["length"] = length
        for scope, count in (("later", later_support), ("full", full_support)):
            for regime in ("lambda3", "lambda4"):
                leaf = report["scopes"][scope]["by_regime"][regime]
                leaf["supported_case_count"] = count
                if not count:
                    leaf["case_weighted_raw_gap"] = 0.
    result = gate.evaluate(reports, identities, stage="dev", technical_complete=True)
    assert [row["actual"] for row in result["conditions"] if row["name"].endswith("supported_cases")] == [later_support] * 2


@pytest.mark.parametrize("defect", ["missing", "duplicate", "wrong_ratio", "wrong_seed", "wrong_stage", "bad_manifest",
                                    "bad_support", "nan_unused_ratio", "negative_gap", "boolean_gap", "wrong_cohort"])
def test_incomplete_mismatched_or_nonfinite_evidence_rejected(defect):
    reports, identities = fixture()
    if defect == "missing":
        reports.pop()
    elif defect == "duplicate":
        reports[-1] = deepcopy(reports[-2])
    elif defect == "wrong_ratio":
        reports[-1]["family"] = "rls_shrink_075@tau=3"
    elif defect == "wrong_seed":
        reports[-1]["seed"] += 7
    elif defect == "wrong_stage":
        reports[-1]["stage"] = "test"
    elif defect == "bad_manifest":
        reports[-1]["identity_manifest"][0]["target_sha256"] = "1" * 64
    elif defect == "bad_support":
        reports[-1]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] -= 1
    elif defect == "nan_unused_ratio":
        set_gap(reports, contract.view_name("rls_shrink_075", 10.), float("nan"))
    elif defect in ("negative_gap", "boolean_gap"):
        set_gap(reports, "pretrained", -1. if defect == "negative_gap" else True)
    else:
        identities[0]["case"] = 10
    with pytest.raises(ValueError):
        gate.evaluate(reports, identities, stage="dev", technical_complete=True)
