"""Fixed-path separate-prior factorial metrics and three named decisions.

The same complete episodes remain in every scope denominator, including zero
support. Initial steps1..3 and postcorrection steps>=5 are separate metrics;
their episode means must not be added to reconstruct the full-path metric.
No model, file, randomness or autonomous-control inference occurs here.
"""
from __future__ import annotations

import math

import numpy as np

from openjev.research import otto_prequery_metrics as prior_base
from openjev.research import otto_score_forecast_data as base
from openjev.research.otto_cross_query_metrics import (
    AGES,
    ARMS,  # noqa: F401
    REGIMES,
    SUPPORT_KEYS,
    _identities,
    require,
)

PRIOR_SUPPORT_KEYS = prior_base.PRIOR_SUPPORT_KEYS
_validate_prior = prior_base._validate_prior
prior_metrics = prior_base.prior_metrics
validate_prediction_support = prior_base.validate_prediction_support

VERSION = "otto-separate-prior-metrics-v1"
FAMILIES = (
    "innovation_shared_mse", "innovation_shared_aux",
    "innovation_separate_mse", "innovation_separate_aux",
    "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux",
)
SEEDS = (285000001, 285000002, 285000003)
SCOPES = ("full", "postcorrection", "initial")
ARCHITECTURES = ("innovation", "gru")
AGREEMENT = "episode_weighted_agreement"
GAP = "episode_weighted_raw_gap"
PRIOR = "episode_weighted_centered_mse"

def forecast_metrics(windows, predictions, identities):
    """Return full, postcorrection and initial-period metrics with the qualified f32 tie rule.

    Inputs use the unchanged complete ``otto_score_forecast_data`` window schema,
    float32 predictions[W,4,4], and ordered identity dicts containing episode_id,
    regime, case and arm. Extra identity metadata is ignored. A case is the pair
    (regime, case), shared across its three collector paths.

    Every overall/regime/age group retains its declared episode denominator,
    including zero-support episodes. Supported-only values are separately named.
    Age groups rebuild within-episode weights. Query and padding rows never
    contribute. Deterministic by_case groups retain every supplied collector
    episode for that originating case, including zero-support paths. Finite
    query/padding predictions are checked but do not score.
    """
    ids, regimes, indices, nonquery, teacher, legal = base._metric_inputs(windows, predictions)
    cases, _ = _identities(identities, ids, regimes)
    rows = [[] for _ in ids]
    for window, age in zip(*np.nonzero(nonquery), strict=True):
        p_near, _ = base._near_minimum(predictions[window, age], legal[window, age])
        t_near, minimum = base._near_minimum(teacher[window, age], legal[window, age])
        action = p_near[0]
        allowed = tuple(int(a) for a in np.flatnonzero(legal[window, age]))
        p = tuple(float(predictions[window, age, a]) for a in allowed)
        t = tuple(float(teacher[window, age, a]) for a in allowed)
        pm, tm = math.fsum(p) / len(p), math.fsum(t) / len(t)
        mse = math.fsum(((a - pm) - (b - tm)) ** 2 for a, b in zip(p, t, strict=True)) / len(p)
        rows[int(indices[window])].append((int(windows["step_offsets"][window]) + int(age), int(age),
            float(action in t_near), float(teacher[window, age, action]) - float(minimum),
            float(action == t_near[0]), mse))

    def summarize(episode_indices, first_step, age=None, last_step=None):
        totals = [[], [], [], []]
        supported, count_rows, zero_ids = [], 0, []
        for index in episode_indices:
            selected = [r for r in rows[index] if r[0] >= first_step and (last_step is None or r[0] <= last_step) and (age is None or r[1] == age)]
            count_rows += len(selected)
            if not selected:
                zero_ids.append(ids[index])
                continue
            supported.append(index)
            for column, values in enumerate(totals, start=2):
                values.append(math.fsum(r[column] for r in selected) / len(selected))
        denominator, support = len(episode_indices), len(supported)
        a, g, f, mse = (math.fsum(values) for values in totals)
        declared_cases = {cases[i] for i in episode_indices}
        supported_cases = {cases[i] for i in supported}

        def case_records(values):
            return [{"regime": regime, "case": case} for regime, case in sorted(values)]

        return {"episodes": denominator, "supported_episodes": support,
            "zero_support_episode_ids": zero_ids, "nonquery_rows": count_rows,
            "weight_mass": support / denominator, "episode_weighted_agreement": a / denominator,
            "episode_weighted_raw_gap": g / denominator, "episode_weighted_first_argmin_match": f / denominator,
            "episode_weighted_centered_mse": mse / denominator,
            "supported_episode_agreement": a / support if support else None,
            "supported_episode_raw_gap": g / support if support else None,
            "supported_episode_centered_mse": mse / support if support else None,
            "declared_case_count": len(declared_cases), "supported_case_count": len(supported_cases),
            "supported_cases": case_records(supported_cases),
            "zero_support_cases": case_records(declared_cases - supported_cases)}

    def scope(first_step, last_step=None):
        def group(episode_indices):
            return {**summarize(episode_indices, first_step, last_step=last_step),
                "by_age": {str(age): summarize(episode_indices, first_step, age, last_step) for age in (1, 2, 3)}}
        return {"overall": group(tuple(range(len(ids)))),
            "by_regime": {regime: group(tuple(i for i, value in enumerate(regimes) if value == regime))
                          for regime in dict.fromkeys(regimes)},
            "by_case": [{"regime": regime, "case": case,
                         **group(tuple(i for i, value in enumerate(cases) if value == (regime, case)))}
                        for regime, case in sorted(set(cases))]}

    return {"version": VERSION, "scope": "saved teacher imitation on fixed paths; no autonomous efficacy",
        "primary_mask": "nonquery and absolute_step >= 5", "full": scope(0), "postcorrection": scope(5), "initial": scope(1, 3)}


def _validate_report(report):
    require(isinstance(report, dict) and report.get("version") == VERSION, "cross-query metric version")
    for scope in SCOPES:
        section = report[scope]
        require(set(section["by_regime"]) == set(REGIMES), "both fixed metric regimes")
        groups = [section["overall"], *section["by_regime"].values()]
        for group in groups:
            require(set(group["by_age"]) == set(AGES), "all three metric ages")
            for values in (group, *group["by_age"].values()):
                for key in ("episodes", "supported_episodes", "nonquery_rows", "declared_case_count", "supported_case_count"):
                    require(type(values[key]) is int and values[key] >= 0, "integer metric support")
                require(values["episodes"] > 0 and values["supported_episodes"] <= values["episodes"]
                        and values["supported_case_count"] <= values["declared_case_count"], "bounded metric support")
                for key in ("weight_mass", "episode_weighted_agreement", "episode_weighted_raw_gap",
                            "episode_weighted_first_argmin_match", "episode_weighted_centered_mse"):
                    value = values[key]
                    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                            "finite nonnegative metric")
                require(values["episode_weighted_agreement"] <= 1 and values["weight_mass"] <= 1,
                        "bounded agreement and support mass")


def _models(models, hold=None):
    require(isinstance(models, (tuple, list)) and len(models) == 24, "all twenty-four fixed cells")
    by = {}
    for model in models:
        require(isinstance(model, dict) and model.get("family") in FAMILIES
                and type(model.get("seed")) is int and model["seed"] in SEEDS, "fixed cell and fit seed")
        key = model["family"], model["seed"]
        require(key not in by, "unique cell/seed")
        by[key] = model
    require(set(by) == {(f, s) for f in FAMILIES for s in SEEDS}, "complete paired membership")
    reference = by[FAMILIES[0], SEEDS[0]]
    expected_report = reference["metrics"] if hold is None else hold
    _validate_report(expected_report)
    for model in models:
        report, prior = model["metrics"], model["prior_metrics"]
        _validate_report(report)
        _validate_prior(prior)
        for name in ("overall", *REGIMES):
            p = prior["overall"] if name == "overall" else prior["by_regime"][name]
            ref = reference["prior_metrics"]
            ref = ref["overall"] if name == "overall" else ref["by_regime"][name]
            require(all(p[k] == ref[k] for k in PRIOR_SUPPORT_KEYS), "identical prior scored domains")
            for scope in SCOPES:
                actual = report[scope]["overall"] if name == "overall" else report[scope]["by_regime"][name]
                expected = expected_report[scope]["overall"] if name == "overall" else expected_report[scope]["by_regime"][name]
                for left, right in ((actual, expected), *((actual["by_age"][a], expected["by_age"][a]) for a in AGES)):
                    require(all(left[k] == right[k] for k in SUPPORT_KEYS), "identical nonquery scored domains")
                require(p["episodes"] == actual["episodes"] and p["declared_case_count"] == actual["declared_case_count"],
                        "same complete episodes for prior and nonquery metrics")
    return by


def criteria(models, hold, *, technical_complete):
    """The 39 unique records feed three named gates, never an omnibus gate.

    All comparisons are equal-three-seed means within each regime. Technical
    closure is a strict caller-owned assertion and remains false until the
    original fitting and saved-audit processes close successfully.
    """
    require(type(technical_complete) is bool, "Boolean technical closure")
    by = _models(models, hold)
    rows = [{"name": "common.technical_complete", "value": technical_complete, "relation": "==",
             "threshold": True, "passes": technical_complete}]

    def add(name, value, relation, threshold, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), "finite criterion")
        row = {"name": name, "value": value, "relation": relation, "threshold": threshold,
               "passes": bool(value >= threshold if relation == ">=" else value <= threshold)}
        if strict is not None:
            require(math.isfinite(strict), "finite strict bound")
            row.update(relation="<= and <", strict_upper_bound=strict,
                       passes=bool(value <= threshold and value < strict))
        rows.append(row)

    def mean(family, scope, regime, key):
        return math.fsum(by[family, s]["metrics"][scope]["by_regime"][regime][key] for s in SEEDS) / 3

    def prior(family, regime):
        return math.fsum(by[family, s]["prior_metrics"]["by_regime"][regime][PRIOR] for s in SEEDS) / 3

    for regime in REGIMES:
        add(f"common.{regime}.initial.case_support", hold["initial"]["by_regime"][regime]["supported_case_count"], ">=", 4)
        for age in AGES:
            add(f"common.{regime}.age{age}.case_support",
                hold["postcorrection"]["by_regime"][regime]["by_age"][age]["supported_case_count"], ">=", 4)
    for architecture in ARCHITECTURES:
        sa, sm, dm, candidate = (f"{architecture}_{suffix}" for suffix in
                                ("shared_aux", "shared_mse", "separate_mse", "separate_aux"))
        for regime in REGIMES:
            prefix = f"mechanism.{architecture}.{regime}"
            for scope in ("full", "initial"):
                threshold = max(mean(sa, scope, regime, AGREEMENT) + .01,
                                mean(sm, scope, regime, AGREEMENT), mean(dm, scope, regime, AGREEMENT))
                add(f"{prefix}.{scope}.agreement_recovery", mean(candidate, scope, regime, AGREEMENT), ">=", threshold)
            add(f"{prefix}.full.gap", mean(candidate, "full", regime, GAP), "<=",
                min(mean(control, "full", regime, GAP) for control in (sa, sm, dm)))
            separate_gap = mean(dm, "postcorrection", regime, GAP)
            add(f"{prefix}.postcorrection.gap", mean(candidate, "postcorrection", regime, GAP), "<=",
                min(mean(sa, "postcorrection", regime, GAP), .9 * separate_gap), strict=separate_gap)
            separate_prior = prior(dm, regime)
            add(f"{prefix}.prior.mse", prior(candidate, regime), "<=",
                min(prior(sa, regime), .8 * separate_prior), strict=separate_prior)
    candidate, control = "innovation_separate_aux", "gru_separate_aux"
    for regime in REGIMES:
        prefix = f"architecture.{regime}"
        add(f"{prefix}.postcorrection.agreement", mean(candidate, "postcorrection", regime, AGREEMENT), ">=",
            mean(control, "postcorrection", regime, AGREEMENT))
        comparison = mean(control, "postcorrection", regime, GAP)
        add(f"{prefix}.postcorrection.gap", mean(candidate, "postcorrection", regime, GAP), "<=", .9 * comparison,
            strict=comparison)
        for scope in ("initial", "full"):
            add(f"{prefix}.{scope}.agreement", mean(candidate, scope, regime, AGREEMENT), ">=",
                mean(control, scope, regime, AGREEMENT))
        add(f"{prefix}.full.gap", mean(candidate, "full", regime, GAP), "<=", mean(control, "full", regime, GAP))
    require(len(rows) == len({r["name"] for r in rows}) == 39, "exact 39-condition inventory")
    return rows


def gate_decisions(rows):
    """Explicit gate membership and counts; no all-39 promotion decision."""
    require(isinstance(rows, (list, tuple)) and len(rows) == 39, "all 39 decision rows")
    names = [r["name"] for r in rows]
    require(len(set(names)) == 39 and all(type(r["passes"]) is bool for r in rows), "unique Boolean conditions")
    common = [n for n in names if n.startswith("common.")]
    mechanism = {a: [n for n in names if n.startswith(f"mechanism.{a}.")] for a in ARCHITECTURES}
    architecture = [n for n in names if n.startswith("architecture.")]
    require(len(common) == 9 and all(len(v) == 10 for v in mechanism.values()) and len(architecture) == 10,
            "fixed common, mechanism and architecture partitions")
    memberships = {"innovation_mechanism": common + mechanism["innovation"],
                   "gru_mechanism": common + mechanism["gru"],
                   "architecture": common + mechanism["innovation"] + architecture}
    by = {r["name"]: r for r in rows}
    return {gate: {"conditions": members, "passed": sum(by[n]["passes"] for n in members),
                   "total": len(members), "passes": all(by[n]["passes"] for n in members)}
            for gate, members in memberships.items()}


def interactions(models):
    """Descriptive (separate aux−MSE)−(shared aux−MSE), signed in metric units."""
    by = _models(models)
    specs = [(scope, key) for scope in SCOPES for key in (AGREEMENT, GAP)] + [("prior", PRIOR)]
    rows, means = [], []

    def value(architecture, regime, scope, metric, seed, readout, objective):
        model = by[f"{architecture}_{readout}_{objective}", seed]
        report = model["prior_metrics"] if scope == "prior" else model["metrics"][scope]
        return report["by_regime"][regime][metric]

    for architecture in ARCHITECTURES:
        for regime in REGIMES:
            for scope, metric in specs:
                values = []
                for seed in SEEDS:
                    args = (architecture, regime, scope, metric, seed)
                    shared = value(*args, "shared", "aux") - value(*args, "shared", "mse")
                    separate = value(*args, "separate", "aux") - value(*args, "separate", "mse")
                    interaction = separate - shared
                    values.append((shared, separate, interaction))
                    rows.append({"architecture": architecture, "regime": regime, "scope": scope, "metric": metric,
                                 "seed": seed, "shared_aux_minus_mse": shared, "separate_aux_minus_mse": separate,
                                 "interaction": interaction})
                means.append({"architecture": architecture, "regime": regime, "scope": scope, "metric": metric,
                    **{key: math.fsum(v[column] for v in values) / 3 for column, key in enumerate(
                        ("shared_aux_minus_mse", "separate_aux_minus_mse", "interaction"))}})
    return {"version": VERSION, "definition": "(separate_aux - separate_mse) - (shared_aux - shared_mse)",
            "scope": "descriptive paired-seed factorial contrast; no isolated gradient-conflict claim",
            "per_seed": rows, "means": means}
