"""The prospective query-memory continuation rule, without audit authority.

Report pairing and scalar arithmetic are checked here. A producer must leave
technical_complete=False. Only an independent saved-output audit that verifies
all phases and original supervisor closures may supply True. This function
cannot authenticate reports, prove causal inputs or admit TEST by itself.
"""
from __future__ import annotations

import math

from openjev.research import otto_query_memory_metrics as metrics

VERSION = "otto-query-memory-gate-v1"
FAMILIES = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta",
            "trace_additive", "trace_scrambled", "trace_no_write")
CONTROLS = ("pretrained", "last_error", "instant_delta", "joint_aux")
CANDIDATE = "trace_delta"
FIT_SEEDS = (309000001, 309000002, 309000003)
require = metrics.require


def continuation_gate(reports, expected_identities, *, technical_complete=False):
    require(type(technical_complete) is bool, "explicit Boolean technical status")
    require(isinstance(expected_identities, (list, tuple)) and bool(expected_identities), "expected cohort")
    stage = expected_identities[0]["stage"]
    require(stage in ("dev", "test"), "only DEV or TEST continuation")
    periods = (4,) if stage == "dev" else (4, 8)
    cases = 3 if stage == "dev" else 6
    required_support = 2 if stage == "dev" else 4
    first = {"lambda3": 305000001, "lambda4": 306000001} if stage == "dev" else {
        "lambda3": 307000001, "lambda4": 308000001}
    require(len(expected_identities) == 6 * cases and all(r["stage"] == stage for r in expected_identities),
            "complete declared split")
    actual = {(r["regime"], r["case"], r["seed"], r["arm"]) for r in expected_identities}
    wanted = {(regime, case, start + case, arm) for regime, start in first.items()
              for case in range(cases) for arm in ("analytic", "neural", "period4_hold")}
    require(actual == wanted, "exact prospective cases, seeds and collectors")
    metrics.validate_matched_reports(reports, families=FAMILIES, fit_seeds=FIT_SEEDS,
                                    query_periods=periods, expected_identities=expected_identities)
    by_key = {(row["family"], row["seed"], row["query_period"]): row for row in reports}

    def leaf(family, seed, period, scope, regime):
        value = by_key[family, seed, period]["scopes"][scope]["by_regime"][regime]
        require(value["episodes"] == 3 * cases and value["declared_case_count"] == cases,
                "complete regime denominators")
        for name in ("case_weighted_raw_gap", "episode_weighted_raw_gap"):
            score = value[name]
            require(type(score) in (float, int) and math.isfinite(score) and score >= 0,
                    "finite nonnegative gap")
        # Three collector paths per case make these equivalent mathematically;
        # retain case weighting as the declared comparison arithmetic.
        support = value["supported_case_count"]
        require(type(support) is int and 0 <= support <= cases, "bounded originating-case support")
        return value

    def gap(family, seed, period, scope, regime):
        return leaf(family, seed, period, scope, regime)["case_weighted_raw_gap"]

    def mean(family, period, scope, regime):
        return math.fsum(gap(family, seed, period, scope, regime) for seed in FIT_SEEDS) / len(FIT_SEEDS)

    conditions = [{"name": "technical_completion", "passed": technical_complete}]
    for period in periods:
        for regime in ("lambda3", "lambda4"):
            prefix = f"{regime}:P{period}"
            support = {leaf(family, seed, period, "later", regime)["supported_case_count"]
                       for family in FAMILIES for seed in FIT_SEEDS}
            require(len(support) == 1, "identical case support across the complete matched set")
            count = support.pop()
            conditions.append({"name": prefix + ":supported_cases", "passed": count >= required_support,
                               "actual": count, "required": required_support})
            candidate = mean(CANDIDATE, period, "later", regime)
            controls = {family: mean(family, period, "later", regime) for family in CONTROLS}
            best = min(controls.values())
            conditions.append({"name": prefix + ":later_gap_10pct", "passed": candidate <= .9 * best and candidate < best,
                               "candidate": candidate, "controls": controls, "best_control": best})
            candidate = mean(CANDIDATE, period, "full", regime)
            controls = {family: mean(family, period, "full", regime) for family in CONTROLS}
            conditions.append({"name": prefix + ":full_gap_nonregression", "passed": candidate <= min(controls.values()),
                               "candidate": candidate, "controls": controls})
            for seed in FIT_SEEDS:
                candidate = gap(CANDIDATE, seed, period, "later", regime)
                controls = {family: gap(family, seed, period, "later", regime) for family in CONTROLS}
                conditions.append({"name": prefix + f":seed_{seed}_nonregression",
                                   "passed": candidate <= min(controls.values()),
                                   "candidate": candidate, "controls": controls})
    expected_count = 13 if stage == "dev" else 25
    require(len(conditions) == expected_count, "complete fixed conjunction")
    return {"version": VERSION, "stage": stage, "candidate": CANDIDATE,
            "technical_complete": technical_complete, "conditions": conditions,
            "passed_conditions": sum(row["passed"] for row in conditions), "total_conditions": expected_count,
            "passed": all(row["passed"] for row in conditions),
            "scope": "prospective mechanism screen; no autonomous, total-compute or novelty claim"}
