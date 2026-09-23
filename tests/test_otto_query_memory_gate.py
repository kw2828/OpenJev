"""Fabricated prospective-gate arithmetic; no empirical evaluation."""
from copy import deepcopy

import numpy as np
import pytest

from openjev.research import otto_query_memory_gate as gate
from openjev.research import otto_query_memory_metrics as metrics


def fixture(stage="dev", *, regression_seed=None, zero_control=False, short=False):
    count = 3 if stage == "dev" else 6
    first = (305000001, 306000001) if stage == "dev" else (307000001, 308000001)
    identities = []
    for regime, seed in zip(("lambda3", "lambda4"), first, strict=True):
        for case in range(count):
            for arm in ("analytic", "neural", "period4_hold"):
                identities.append({"stage": stage, "episode_id": f"{stage}:{regime}:{case}:{arm}",
                                   "episode_index": len(identities), "seed": seed + case,
                                   "case": case, "regime": regime, "arm": arm})
    length = 1 if short else 17
    target = np.tile(np.array([0., 1., 2., 3.], np.float32), (length, 1))
    legal = np.ones_like(target, np.bool_)
    reports = []
    for family in gate.FAMILIES:
        for seed in gate.FIT_SEEDS:
            action = 0 if family == gate.CANDIDATE else 1
            if family == gate.CANDIDATE and seed == regression_seed:
                action = 2
            if zero_control and family == "pretrained":
                action = 0
            prediction = np.ones_like(target)
            prediction[:, action] = 0
            for period in ((4,) if stage == "dev" else (4, 8)):
                rows = [metrics.episode_metrics(identity, period, target, legal, prediction,
                                                prequery_forecast=target) for identity in identities]
                reports.append(metrics.aggregate_episodes(rows, family=family, fit_seed=seed,
                                                          expected_identities=identities))
    return reports, identities


@pytest.mark.parametrize("stage,conditions", (("dev", 13), ("test", 25)))
def test_all_fixed_conditions_are_required_and_producer_never_self_admits(stage, conditions):
    reports, identities = fixture(stage)
    pending = gate.continuation_gate(reports, identities)
    assert not pending["passed"] and pending["passed_conditions"] == conditions - 1
    complete = gate.continuation_gate(reports, identities, technical_complete=True)
    assert complete["passed"] and complete["passed_conditions"] == complete["total_conditions"] == conditions


def test_seed_regression_cannot_be_hidden_by_improved_mean():
    reports, identities = fixture(regression_seed=gate.FIT_SEEDS[-1])
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    failures = [row["name"] for row in result["conditions"] if not row["passed"]]
    assert len(failures) == 2 and all(f"seed_{gate.FIT_SEEDS[-1]}" in name for name in failures)
    assert all(row["passed"] for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


def test_zero_best_control_cannot_pass_strict_improvement_and_no_support_still_fails():
    reports, identities = fixture(zero_control=True)
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    assert not result["passed"]
    assert all(not row["passed"] for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))
    reports, identities = fixture(short=True)
    empty = gate.continuation_gate(reports, identities, technical_complete=True)
    assert not empty["passed"]
    assert all(not row["passed"] for row in empty["conditions"] if row["name"].endswith("supported_cases"))


def test_one_shifted_regime_cannot_be_rescued_by_other_panels():
    reports, identities = fixture("test")
    report = next(r for r in reports if r["family"] == gate.CANDIDATE and r["seed"] == gate.FIT_SEEDS[0]
                  and r["query_period"] == 8)
    # This pure gate intentionally trusts supplied aggregates; the independent
    # audit must recompute them. Here isolate a single scalar comparison.
    report["scopes"]["full"]["by_regime"]["lambda4"]["case_weighted_raw_gap"] = 4
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    failures = [r["name"] for r in result["conditions"] if not r["passed"]]
    assert failures == ["lambda4:P8:full_gap_nonregression"]


def test_mean_comparison_uses_best_control_mean_and_seed_guards_use_paired_best():
    reports, identities = fixture()
    controls = {"pretrained": (0., 10., 10.), "last_error": (10., 0., 10.),
                "instant_delta": (10., 10., 0.), "joint_aux": (10., 10., 10.)}
    for report in reports:
        index = gate.FIT_SEEDS.index(report["seed"])
        value = 6. if report["family"] == gate.CANDIDATE else controls.get(report["family"], (10.,) * 3)[index]
        for regime in ("lambda3", "lambda4"):
            report["scopes"]["later"]["by_regime"][regime]["case_weighted_raw_gap"] = value
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    means = [r for r in result["conditions"] if r["name"].endswith("later_gap_10pct")]
    assert all(r["passed"] and r["best_control"] == pytest.approx(20 / 3) for r in means)
    paired = [r for r in result["conditions"] if "seed_" in r["name"]]
    assert len(paired) == 6 and not any(r["passed"] for r in paired)


@pytest.mark.parametrize("candidate,passes", ((9., True), (float(np.nextafter(9., np.inf)), False)))
def test_exact_ninety_percent_boundary_has_no_tolerance(candidate, passes):
    reports, identities = fixture()
    for report in reports:
        value = candidate if report["family"] == gate.CANDIDATE else 10.
        for regime in ("lambda3", "lambda4"):
            report["scopes"]["later"]["by_regime"][regime]["case_weighted_raw_gap"] = value
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    assert all(row["passed"] is passes for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


@pytest.mark.parametrize("stage,support,passes", (("dev", 1, False), ("dev", 2, True),
                                                 ("test", 3, False), ("test", 4, True)))
def test_support_threshold_counts_originating_cases(stage, support, passes):
    reports, identities = fixture(stage)
    for report in reports:
        for regime in ("lambda3", "lambda4"):
            report["scopes"]["later"]["by_regime"][regime]["supported_case_count"] = support
    result = gate.continuation_gate(reports, identities, technical_complete=True)
    assert all(row["passed"] is passes for row in result["conditions"] if row["name"].endswith("supported_cases"))


@pytest.mark.parametrize("defect", ("missing", "different_seed", "nan", "different_support", "wrong_cohort"))
def test_missing_changed_or_nonfinite_comparisons_fail_closed(defect):
    reports, identities = fixture()
    reports, identities = deepcopy(reports), deepcopy(identities)
    if defect == "missing":
        reports.pop()
    elif defect == "different_seed":
        reports[0]["seed"] += 1
    elif defect == "nan":
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = float("nan")
    elif defect == "different_support":
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] -= 1
    else:
        identities[0]["seed"] += 1
    with pytest.raises(ValueError):
        gate.continuation_gate(reports, identities, technical_complete=True)
