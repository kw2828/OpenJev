"""One fixed GRU adaptation gate, using unchanged forecast metric arithmetic.

All fifteen family/seed records and the hold reference retain the same complete
VALID cohort. Equal-seed means do not replace the inherited episode denominators;
zero-support episodes remain in each scope. The candidate must beat every one
of four controls, not a selected weak comparator. This is a forced-path teacher
imitation comparison, not an architecture or autonomous-control claim.

Canonical evaluation metadata is a required caller assertion, not evidence of
execution. The trainer and independent saved audit must verify frozen parameter
flags, no_grad execution, state identity and actual process closure. Nothing in
this pure numerical wrapper loads a model or upgrades technical_complete.
"""
from __future__ import annotations

import copy
import math

from openjev.research import otto_separate_prior_metrics as base

VERSION = "otto-protected-readout-metrics-v1"
FAMILIES = ("pretrained", "frozen_aux", "frozen_spo", "joint_aux", "joint_spo")
SEEDS = (301000001, 301000002, 301000003)
CANDIDATE = "frozen_spo"
COMPARATORS = ("pretrained", "frozen_aux", "joint_aux", "joint_spo")
REGIMES, ARMS, AGES, SCOPES = base.REGIMES, base.ARMS, base.AGES, base.SCOPES
AGREEMENT, GAP, PRIOR = base.AGREEMENT, base.GAP, base.PRIOR
COMMON_CONDITIONS, CANDIDATE_CONDITIONS, REQUIRED_CONDITIONS = 11, 18, 29
EVALUATION_KEYS = ("parameter_mode", "grad_enabled", "backbone_requires_grad", "residual_requires_grad")
forecast_metrics = base.forecast_metrics
prior_metrics = base.prior_metrics
validate_prediction_support = base.validate_prediction_support
require = base.require


def evaluation_metadata(family):
    """Declare canonical flags; the caller must separately prove their use."""
    require(family in FAMILIES, "declared canonical-evaluation family")
    return {"parameter_mode": "frozen", "grad_enabled": False, "backbone_requires_grad": False,
            "residual_requires_grad": None if family == "pretrained" else True}


def _models(models, hold):
    require(isinstance(models, (list, tuple)) and len(models) == 15,
            "all fifteen pretrained and adaptation records")
    by = {}
    for model in models:
        require(isinstance(model, dict) and model.get("family") in FAMILIES
                and type(model.get("seed")) is int and model["seed"] in SEEDS,
                "fixed protected-readout family and seed")
        evaluation = model.get("evaluation")
        expected_evaluation = evaluation_metadata(model["family"])
        require(isinstance(evaluation, dict) and set(evaluation) == set(EVALUATION_KEYS)
                and all(type(evaluation[key]) is type(value) and evaluation[key] == value
                        for key, value in expected_evaluation.items()),
                "declared canonical frozen no_grad evaluation; caller must audit execution")
        key = model["family"], model["seed"]
        require(key not in by, "unique protected-readout family/seed")
        by[key] = model
    require(set(by) == {(family, seed) for family in FAMILIES for seed in SEEDS},
            "complete paired protected-readout membership")
    base._validate_report(hold)
    reference = by["pretrained", SEEDS[0]]["prior_metrics"]
    base._validate_prior(reference)
    for name in ("overall", *REGIMES):
        episodes, cases = (36, 12) if name == "overall" else (18, 6)
        for scope in SCOPES:
            group = hold[scope]["overall"] if name == "overall" else hold[scope]["by_regime"][name]
            for values in (group, *group["by_age"].values()):
                require(values["episodes"] == episodes and values["declared_case_count"] == cases,
                        "complete fixed VALID episode and originating-case denominators")
    for model in by.values():
        report, prior = model["metrics"], model["prior_metrics"]
        base._validate_report(report)
        base._validate_prior(prior)
        for name in ("overall", *REGIMES):
            p = prior["overall"] if name == "overall" else prior["by_regime"][name]
            ref = reference["overall"] if name == "overall" else reference["by_regime"][name]
            require(all(p[key] == ref[key] for key in base.PRIOR_SUPPORT_KEYS),
                    "identical complete prior domains")
            for scope in SCOPES:
                actual = report[scope]["overall"] if name == "overall" else report[scope]["by_regime"][name]
                expected = hold[scope]["overall"] if name == "overall" else hold[scope]["by_regime"][name]
                for left, right in ((actual, expected), *(
                    (actual["by_age"][age], expected["by_age"][age]) for age in AGES
                )):
                    require(all(left[key] == right[key] for key in base.SUPPORT_KEYS),
                            "identical complete nonquery domains")
                require(p["episodes"] == actual["episodes"]
                        and p["declared_case_count"] == actual["declared_case_count"],
                        "same complete episodes for prior and nonquery reports")
    return by


def condition_names():
    """Exact ordered eleven common plus nine candidate conditions per setting."""
    names = ["common.technical_complete"]
    for regime in REGIMES:
        names.append(f"common.{regime}.initial.case_support")
        names.extend(f"common.{regime}.age{age}.case_support" for age in AGES)
        names.append(f"common.{regime}.hold.postcorrection.positive_gap")
    for regime in REGIMES:
        prefix = f"candidate.{regime}"
        names.extend(f"{prefix}.postcorrection.gap_vs_{control}" for control in COMPARATORS)
        names.extend(f"{prefix}.{scope}.agreement" for scope in ("full", "initial"))
        names.extend(f"{prefix}.{seed}.postcorrection.gap_vs_frozen_aux" for seed in SEEDS)
    return tuple(names)


def criteria(models, hold, *, technical_complete):
    """Require all four mean-gap improvements and every declared nonregression.

Each mean-gap condition requires candidate <= 0.9 * control AND candidate <
control. A zero/zero tie therefore fails. Full and initial agreement must each
match the largest of the four control means. Three paired-seed later gaps must
not exceed frozen_aux. No tolerances, fitted thresholds, seed selection or prior
calibration gate are introduced. Closure remains a caller-owned Boolean.
    """
    require(type(technical_complete) is bool, "Boolean technical closure")
    by = _models(models, hold)
    rows = [{"name": "common.technical_complete", "value": technical_complete, "relation": "==",
             "threshold": True, "passes": technical_complete}]

    def add(name, value, relation, threshold, *, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), "finite protected-readout criterion")
        if relation == ">=":
            passes = value >= threshold
        elif relation == ">":
            passes = value > threshold
        else:
            require(relation == "<=", "declared criterion relation")
            passes = value <= threshold
        row = {"name": name, "value": value, "relation": relation, "threshold": threshold,
               "passes": bool(passes)}
        if strict is not None:
            require(relation == "<=" and math.isfinite(strict), "finite strict gap upper bound")
            row.update(relation="<= and <", strict_upper_bound=strict,
                       passes=bool(value <= threshold and value < strict))
        rows.append(row)

    def metric(family, seed, scope, regime, key):
        return by[family, seed]["metrics"][scope]["by_regime"][regime][key]

    def mean(family, scope, regime, key):
        return math.fsum(metric(family, seed, scope, regime, key) for seed in SEEDS) / len(SEEDS)

    for regime in REGIMES:
        add(f"common.{regime}.initial.case_support",
            hold["initial"]["by_regime"][regime]["supported_case_count"], ">=", 4)
        for age in AGES:
            add(f"common.{regime}.age{age}.case_support",
                hold["postcorrection"]["by_regime"][regime]["by_age"][age]["supported_case_count"], ">=", 4)
        add(f"common.{regime}.hold.postcorrection.positive_gap",
            hold["postcorrection"]["by_regime"][regime][GAP], ">", 0.)
    for regime in REGIMES:
        prefix = f"candidate.{regime}"
        gap = mean(CANDIDATE, "postcorrection", regime, GAP)
        for control in COMPARATORS:
            control_gap = mean(control, "postcorrection", regime, GAP)
            add(f"{prefix}.postcorrection.gap_vs_{control}", gap, "<=", .9 * control_gap, strict=control_gap)
        for scope in ("full", "initial"):
            add(f"{prefix}.{scope}.agreement", mean(CANDIDATE, scope, regime, AGREEMENT), ">=",
                max(mean(control, scope, regime, AGREEMENT) for control in COMPARATORS))
        for seed in SEEDS:
            add(f"{prefix}.{seed}.postcorrection.gap_vs_frozen_aux",
                metric(CANDIDATE, seed, "postcorrection", regime, GAP), "<=",
                metric("frozen_aux", seed, "postcorrection", regime, GAP))
    require(tuple(row["name"] for row in rows) == condition_names()
            and len(rows) == COMMON_CONDITIONS + CANDIDATE_CONDITIONS == REQUIRED_CONDITIONS,
            "exact ordered 29-condition inventory")
    return rows


def gate_decisions(rows):
    """Return only the one named 29-condition protected-readout decision."""
    expected = condition_names()
    require(isinstance(rows, (list, tuple)) and len(rows) == len(expected) == REQUIRED_CONDITIONS,
            "all 29 protected-readout conditions")
    require(all(isinstance(row, dict) and type(row.get("passes")) is bool for row in rows)
            and tuple(row.get("name") for row in rows) == expected,
            "exact ordered unique Boolean protected-readout conditions")
    return {"protected_readout": {"conditions": list(expected), "passed": sum(row["passes"] for row in rows),
                                  "total": REQUIRED_CONDITIONS, "passes": all(row["passes"] for row in rows)}}


def summarize(models, hold, *, technical_complete):
    """Own a complete fifteen-record summary without changing caller reports."""
    rows = criteria(models, hold, technical_complete=technical_complete)
    by = {(record["family"], record["seed"]): record for record in models}
    return {"version": VERSION, "scope": "Fixed-path GRU adaptation only; no autonomous or architecture claim",
            "evaluation_scope": "Declared per-record flags; actual execution requires caller audit",
            "candidate": CANDIDATE, "comparators": list(COMPARATORS),
            "seeds": list(SEEDS), "models": [copy.deepcopy(by[family, seed])
                                            for seed in SEEDS for family in FAMILIES],
            "hold": copy.deepcopy(hold), "required": rows, "required_conditions": REQUIRED_CONDITIONS,
            "required_passed": sum(row["passes"] for row in rows), "gates": gate_decisions(rows),
            "technical_complete": technical_complete}
