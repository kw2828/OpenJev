"""Fixed-path action-focused comparisons with three separate named gates.

Forecast and prior arithmetic are unchanged qualified functions, including their
historical report versions, float32 near-tie selection and declared-episode
denominators. SPO+ training's exact-minimum reference does not change evaluation.
All three fit seeds contribute equally; no seed, scope or control is selected.
The 41 unique conditions are coverage records, never an omnibus promotion gate.
These comparisons concern teacher imitation, not autonomous control efficacy.
"""
from __future__ import annotations

import math

from openjev.research import otto_separate_prior_metrics as base

VERSION = "otto-action-focused-metrics-v1"
FAMILIES = ("innovation_aux", "innovation_spo", "gru_aux", "gru_spo")
SEEDS = (295000001, 295000002, 295000003)
ARCHITECTURES = ("innovation", "gru")
REGIMES, ARMS, AGES, SCOPES = base.REGIMES, base.ARMS, base.AGES, base.SCOPES
AGREEMENT, GAP, PRIOR = base.AGREEMENT, base.GAP, base.PRIOR
forecast_metrics = base.forecast_metrics
prior_metrics = base.prior_metrics
validate_prediction_support = base.validate_prediction_support
require = base.require


def _models(models, hold):
    require(isinstance(models, (list, tuple)) and len(models) == 12, "all twelve fixed cells")
    by = {}
    for model in models:
        require(isinstance(model, dict) and model.get("family") in FAMILIES
                and type(model.get("seed")) is int and model["seed"] in SEEDS, "fixed cell and seed")
        key = model["family"], model["seed"]
        require(key not in by, "unique cell/seed")
        by[key] = model
    require(set(by) == {(family, seed) for family in FAMILIES for seed in SEEDS}, "complete paired membership")
    base._validate_report(hold)
    reference = by[FAMILIES[0], SEEDS[0]]["prior_metrics"]
    base._validate_prior(reference)
    for model in by.values():
        report, prior = model["metrics"], model["prior_metrics"]
        base._validate_report(report)
        base._validate_prior(prior)
        for name in ("overall", *REGIMES):
            p = prior["overall"] if name == "overall" else prior["by_regime"][name]
            ref = reference["overall"] if name == "overall" else reference["by_regime"][name]
            require(all(p[key] == ref[key] for key in base.PRIOR_SUPPORT_KEYS), "identical prior domains")
            for scope in SCOPES:
                actual = report[scope]["overall"] if name == "overall" else report[scope]["by_regime"][name]
                expected = hold[scope]["overall"] if name == "overall" else hold[scope]["by_regime"][name]
                for left, right in ((actual, expected), *((actual["by_age"][age], expected["by_age"][age]) for age in AGES)):
                    require(all(left[key] == right[key] for key in base.SUPPORT_KEYS), "identical nonquery domains")
                require(p["episodes"] == actual["episodes"] and p["declared_case_count"] == actual["declared_case_count"],
                        "same complete episodes for prior and nonquery metrics")
    return by


def condition_names():
    """Return the exact ordered 11 common, 12+12 objective and six architecture names."""
    names = ["common.technical_complete"]
    for regime in REGIMES:
        names.append(f"common.{regime}.initial.case_support")
        names.extend(f"common.{regime}.age{age}.case_support" for age in AGES)
        names.append(f"common.{regime}.hold.postcorrection.positive_gap")
    for architecture in ARCHITECTURES:
        for regime in REGIMES:
            prefix = f"objective.{architecture}.{regime}"
            names.extend((f"{prefix}.postcorrection.gap", f"{prefix}.full.agreement", f"{prefix}.initial.agreement"))
            names.extend(f"{prefix}.{seed}.postcorrection.gap_nonregression" for seed in SEEDS)
    for regime in REGIMES:
        names.extend(f"architecture.{regime}.{suffix}" for suffix in
                     ("postcorrection.gap", "full.agreement", "initial.agreement"))
    return tuple(names)


def criteria(models, hold, *, technical_complete):
    """Build 41 records from all 12 fixed fits and the same complete hold cohort.

Each objective gate requires a strict mean primary-gap improvement of at least
10%, full/initial agreement nonregression, and primary-gap nonregression for
every paired seed. Architecture comparisons use the better of BOTH GRU cells
separately for each metric, not a possibly weakened GRU treatment alone.
Technical closure is a caller-owned Boolean and stays false before the original
fitting and saved-audit processes close successfully. There is no numerical
tolerance in any gate, no prior-calibration gate and no learned query policy.
    """
    require(type(technical_complete) is bool, "Boolean technical closure")
    by = _models(models, hold)
    rows = [{"name": "common.technical_complete", "value": technical_complete, "relation": "==",
             "threshold": True, "passes": technical_complete}]

    def add(name, value, relation, threshold, *, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), "finite criterion")
        if relation == ">=":
            passes = value >= threshold
        elif relation == ">":
            passes = value > threshold
        else:
            require(relation == "<=", "declared criterion relation")
            passes = value <= threshold
        row = {"name": name, "value": value, "relation": relation, "threshold": threshold, "passes": bool(passes)}
        if strict is not None:
            require(relation == "<=" and math.isfinite(strict), "finite strict upper bound")
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
    for architecture in ARCHITECTURES:
        candidate, control = f"{architecture}_spo", f"{architecture}_aux"
        for regime in REGIMES:
            prefix = f"objective.{architecture}.{regime}"
            control_gap = mean(control, "postcorrection", regime, GAP)
            add(f"{prefix}.postcorrection.gap", mean(candidate, "postcorrection", regime, GAP), "<=",
                .9 * control_gap, strict=control_gap)
            for scope in ("full", "initial"):
                add(f"{prefix}.{scope}.agreement", mean(candidate, scope, regime, AGREEMENT), ">=",
                    mean(control, scope, regime, AGREEMENT))
            for seed in SEEDS:
                add(f"{prefix}.{seed}.postcorrection.gap_nonregression",
                    metric(candidate, seed, "postcorrection", regime, GAP), "<=",
                    metric(control, seed, "postcorrection", regime, GAP))
    for regime in REGIMES:
        prefix = f"architecture.{regime}"
        better_gap = min(mean(family, "postcorrection", regime, GAP) for family in ("gru_aux", "gru_spo"))
        add(f"{prefix}.postcorrection.gap", mean("innovation_spo", "postcorrection", regime, GAP), "<=",
            .9 * better_gap, strict=better_gap)
        for scope in ("full", "initial"):
            add(f"{prefix}.{scope}.agreement", mean("innovation_spo", scope, regime, AGREEMENT), ">=",
                max(mean(family, scope, regime, AGREEMENT) for family in ("gru_aux", "gru_spo")))
    require(tuple(row["name"] for row in rows) == condition_names() and len(rows) == 41,
            "exact ordered 41-condition inventory")
    return rows


def gate_decisions(rows):
    """Return innovation_objective23, gru_objective23 and architecture29 only."""
    expected = condition_names()
    require(isinstance(rows, (list, tuple)) and len(rows) == len(expected) == 41, "all 41 unique decision rows")
    by = {}
    for row in rows:
        require(isinstance(row, dict) and row.get("name") in expected and type(row.get("passes")) is bool,
                "declared Boolean criterion")
        require(row["name"] not in by, "unique criterion name")
        by[row["name"]] = row
    common = [name for name in expected if name.startswith("common.")]
    objective = {architecture: [name for name in expected if name.startswith(f"objective.{architecture}.")]
                 for architecture in ARCHITECTURES}
    architecture = [name for name in expected if name.startswith("architecture.")]
    memberships = {"innovation_objective": common + objective["innovation"],
                   "gru_objective": common + objective["gru"],
                   "architecture": common + objective["innovation"] + architecture}
    require(tuple(map(len, memberships.values())) == (23, 23, 29), "three fixed gate memberships")
    return {gate: {"conditions": members, "passed": sum(by[name]["passes"] for name in members),
                   "total": len(members), "passes": all(by[name]["passes"] for name in members)}
            for gate, members in memberships.items()}
