"""Prospective ratio selection and comparison arithmetic, with no IO or admission.

All saved scalar metrics must be reconstructed by an independent audit. This
module validates the complete matched report roster, never substitutes a control
for the named full-RLS candidate, and never opens a held-out split. A caller's
technical_complete flag is not proof of original process closure.
"""
from __future__ import annotations

import math

from openjev.research import otto_query_memory_metrics as metrics
from openjev.research import otto_residual_contract as contract

VERSION = "otto-residual-gate-v1"
REGIMES = ("lambda3", "lambda4")
COLLECTORS = ("analytic", "neural", "period4_hold")
require = contract.require


def evaluate(reports, expected_identities, *, stage, selected_tau=None, technical_complete=False):
    """Select once on DEV, or evaluate one already selected ratio on confirmation.

    The compatibility metrics format labels the new confirmation cohort test.
    Its caller must authenticate the new study's exact independent seed roster;
    this label never permits reading the failed prior study's TEST arrays.
    """
    require(type(technical_complete) is bool, "explicit Boolean technical completion")
    specs = contract.view_specs(stage, selected_tau)
    cases, required_support = (3, 2) if stage == "dev" else (6, 4)
    metric_stage = "dev" if stage == "dev" else "test"
    require(isinstance(expected_identities, (list, tuple)) and len(expected_identities) == 6 * cases,
            "complete expected originating-case cohort")
    require(all(isinstance(row, dict) and row.get("stage") == metric_stage for row in expected_identities),
            "exact new-study compatibility split")
    actual = {(row.get("regime"), row.get("case"), row.get("arm")) for row in expected_identities}
    wanted = {(regime, case, arm) for regime in REGIMES for case in range(cases) for arm in COLLECTORS}
    require(actual == wanted, "complete case and collector roster")
    families = tuple(dict.fromkeys(contract.view_name(method, tau) for method, tau, _ in specs))
    metrics.validate_matched_reports(reports, families=families, fit_seeds=contract.FIT_SEEDS,
                                    query_periods=(4,), expected_identities=expected_identities)
    by_key = {(row["family"], row["seed"]): row for row in reports}

    def leaf(family, seed, scope, regime):
        report = by_key[family, seed]
        require(isinstance(report.get("scopes"), dict) and isinstance(report["scopes"].get(scope), dict)
                and isinstance(report["scopes"][scope].get("by_regime"), dict), "complete scalar metric scopes")
        value = report["scopes"][scope]["by_regime"].get(regime)
        require(isinstance(value, dict) and type(value.get("episodes")) is int
                and value["episodes"] == 3 * cases and type(value.get("declared_case_count")) is int
                and value["declared_case_count"] == cases, "complete case denominators")
        number = value.get("case_weighted_raw_gap")
        require(type(number) in (float, int) and math.isfinite(number) and number >= 0,
                "finite nonnegative case-weighted gap")
        minimum_length = 6 if scope == "later" else 2
        supported = {row["case"] for row in report["identity_manifest"]
                     if row["regime"] == regime and row["length"] >= minimum_length}
        require(type(value.get("supported_case_count")) is int
                and value["supported_case_count"] == len(supported), "support matches preserved episode lengths")
        require(bool(supported) or number == 0, "unsupported scope has zero full-denominator gap")
        return value

    # Validate all controls and all development ratios, not only the winner.
    for family in families:
        for seed in contract.FIT_SEEDS:
            for scope in ("full", "later"):
                for regime in REGIMES:
                    leaf(family, seed, scope, regime)

    def gap(family, seed, scope, regime):
        return leaf(family, seed, scope, regime)["case_weighted_raw_gap"]

    def mean(family, scope, regime):
        # Preserve the existing gate's sum-then-divide arithmetic. Overflow
        # rejects a run rather than changing its numerical comparison rule.
        try:
            return math.fsum(gap(family, seed, scope, regime) for seed in contract.FIT_SEEDS) / len(contract.FIT_SEEDS)
        except OverflowError as error:
            raise ValueError("finite aggregate gap arithmetic") from error

    ratios = contract.TAUS if stage == "dev" else (contract.method_config("rls_full", selected_tau),)
    grid = []
    for tau in ratios:
        family = contract.view_name("rls_full", tau)
        means = {regime: mean(family, "later", regime) for regime in REGIMES}
        grid.append({"tau": tau, "later_means": means, "objective": max(means.values())})
    chosen = min(grid, key=lambda row: row["objective"])["tau"]
    candidate = contract.view_name("rls_full", chosen)
    conditions = [{"name": "technical_completion", "passed": technical_complete}]
    for regime in REGIMES:
        prefix = regime + ":P4"
        support = leaf(candidate, contract.FIT_SEEDS[0], "later", regime)["supported_case_count"]
        conditions.append({"name": prefix + ":supported_cases", "passed": support >= required_support,
                           "actual": support, "required": required_support})
        value = mean(candidate, "later", regime)
        controls = {name: mean(name, "later", regime) for name in contract.BASELINES}
        best = min(controls.values())
        conditions.append({"name": prefix + ":later_gap_10pct", "passed": value <= .9 * best and value < best,
                           "candidate": value, "controls": controls, "best_control": best})
        value = mean(candidate, "full", regime)
        controls = {name: mean(name, "full", regime) for name in contract.BASELINES}
        conditions.append({"name": prefix + ":full_gap_nonregression", "passed": value <= min(controls.values()),
                           "candidate": value, "controls": controls})
        for seed in contract.FIT_SEEDS:
            value = gap(candidate, seed, "later", regime)
            controls = {name: gap(name, seed, "later", regime) for name in contract.BASELINES}
            conditions.append({"name": prefix + f":seed_{seed}_nonregression",
                               "passed": value <= min(controls.values()), "candidate": value, "controls": controls})
    require(len(conditions) == 13, "complete fixed usefulness conjunction")
    passed = all(row["passed"] for row in conditions)

    contrasts = []
    for tau in ratios:
        full = contract.view_name("rls_full", tau)
        for method in contract.RLS_METHODS[1:]:
            other = contract.view_name(method, tau)
            panels, covariance_conditions = [], []
            for regime in REGIMES:
                full_later, other_later = mean(full, "later", regime), mean(other, "later", regime)
                full_gap, other_gap = mean(full, "full", regime), mean(other, "full", regime)
                paired = [{"seed": seed, "candidate_later": gap(full, seed, "later", regime),
                           "control_later": gap(other, seed, "later", regime),
                           "candidate_full": gap(full, seed, "full", regime),
                           "control_full": gap(other, seed, "full", regime)} for seed in contract.FIT_SEEDS]
                panels.append({"regime": regime, "candidate_later": full_later, "control_later": other_later,
                               "candidate_full": full_gap, "control_full": other_gap, "paired": paired})
                if method == "rls_diagonal":
                    covariance_conditions.extend([
                        {"name": regime + ":later_gap_10pct", "passed": full_later <= .9 * other_later and full_later < other_later},
                        {"name": regime + ":full_gap_nonregression", "passed": full_gap <= other_gap},
                        *[{"name": regime + f":seed_{row['seed']}_nonregression",
                           "passed": row["candidate_later"] <= row["control_later"]} for row in paired]])
            record = {"tau": tau, "candidate": full, "control": other, "panels": panels}
            if method == "rls_diagonal":
                contrast_passed = all(row["passed"] for row in covariance_conditions)
                record.update(conditions=covariance_conditions, contrast_passed=contrast_passed,
                              confirmed_with_usefulness=stage == "confirm" and passed and contrast_passed)
            contrasts.append(record)
    return {"version": VERSION, "stage": stage, "candidate": "rls_full", "selected_tau": chosen,
            "selection": {"rule": "minimize worst-setting candidate mean later gap; ascending-grid exact tie",
                          "grid": grid, "uses_confirmation_for_selection": False},
            "reports": len(reports), "technical_complete": technical_complete,
            "conditions": conditions, "passed_conditions": sum(row["passed"] for row in conditions),
            "total_conditions": 13, "passed": passed, "mechanistic_contrasts": contrasts,
            "scope": "fixed-path usefulness screen; no autonomous, calibration, compute or novelty claim"}
