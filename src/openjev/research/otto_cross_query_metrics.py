"""Pure full-path and post-correction teacher-score imitation metrics.

The caller authenticates complete census windows and originating identities.
No models, files, randomness or search-return inference occur here. Additional
query anchors are inputs, never scored rows. The primary domain begins at step5,
after the genuine correction at step4; full-path nonquery metrics are descriptive.
"""
from __future__ import annotations

import math

import numpy as np

from openjev.research import otto_score_forecast_data as base

VERSION = "otto-cross-query-metrics-v1"
FAMILIES = ("innovation", "innovation_gru", "persistent_direct", "reset_direct")
SEEDS = (255000001, 255000002, 255000003)
REGIMES = ("lambda3", "lambda4")
ARMS = ("analytic", "neural", "period4_hold")
AGES = ("1", "2", "3")
SUPPORT_KEYS = ("episodes", "supported_episodes", "zero_support_episode_ids", "nonquery_rows",
                "weight_mass", "declared_case_count", "supported_case_count", "supported_cases",
                "zero_support_cases")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _identities(identities, ids, regimes):
    require(isinstance(identities, (list, tuple)) and len(identities) == len(ids),
            "one originating identity per episode")
    cases, collectors, seen = [], [], set()
    for identity, episode_id, regime in zip(identities, ids, regimes, strict=True):
        require(isinstance(identity, dict) and identity.get("episode_id") == episode_id
                and identity.get("regime") == regime, "aligned episode/regime identity")
        case, arm = identity.get("case"), identity.get("arm")
        require(type(case) is int and case >= 0 and arm in ARMS, "originating case and collector")
        key = (regime, case, arm)
        require(key not in seen, "unique originating case/collector episode")
        seen.add(key)
        cases.append((regime, case))
        collectors.append(arm)
    return tuple(cases), tuple(collectors)


def forecast_metrics(windows, predictions, identities):
    """Return full and postcorrection metrics with the qualified f32 tie rule.

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

    def summarize(episode_indices, first_step, age=None):
        totals = [[], [], [], []]
        supported, count_rows, zero_ids = [], 0, []
        for index in episode_indices:
            selected = [r for r in rows[index] if r[0] >= first_step and (age is None or r[1] == age)]
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

    def scope(first_step):
        def group(episode_indices):
            return {**summarize(episode_indices, first_step),
                "by_age": {str(age): summarize(episode_indices, first_step, age) for age in (1, 2, 3)}}
        return {"overall": group(tuple(range(len(ids)))),
            "by_regime": {regime: group(tuple(i for i, value in enumerate(regimes) if value == regime))
                          for regime in dict.fromkeys(regimes)},
            "by_case": [{"regime": regime, "case": case,
                         **group(tuple(i for i, value in enumerate(cases) if value == (regime, case)))}
                        for regime, case in sorted(set(cases))]}

    return {"version": VERSION, "scope": "saved teacher imitation on fixed paths; no autonomous efficacy",
        "primary_mask": "nonquery and absolute_step >= 5", "full": scope(0), "postcorrection": scope(5)}


def _validate_report(report):
    require(isinstance(report, dict) and report.get("version") == VERSION, "cross-query metric version")
    for scope in ("full", "postcorrection"):
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


def criteria(models, hold, *, technical_complete):
    """All 53 prospective conditions, without selecting a seed or control.

    ``models`` contains exactly twelve {family, seed, metrics} records. Extra fit
    metadata is ignored. ``technical_complete`` is a strict caller-owned Boolean
    covering original successful processes, fixed fits and provenance closure.
    It is an admission assertion, not evidence reconstructed by this pure helper.
    """
    require(type(technical_complete) is bool, "Boolean technical closure")
    require(isinstance(models, (list, tuple)) and len(models) == 12, "all twelve fixed fits")
    by = {}
    for model in models:
        require(isinstance(model, dict) and model.get("family") in FAMILIES
                and type(model.get("seed")) is int and model["seed"] in SEEDS, "fixed family and fit seed")
        key = (model["family"], model["seed"])
        require(key not in by, "unique family/seed")
        by[key] = model["metrics"]
    require(set(by) == {(kind, seed) for kind in FAMILIES for seed in SEEDS}, "complete paired fit membership")
    _validate_report(hold)
    for report in by.values():
        _validate_report(report)
        for scope in ("full", "postcorrection"):
            for name in ("overall", *REGIMES):
                expected = hold[scope]["overall"] if name == "overall" else hold[scope]["by_regime"][name]
                actual = report[scope]["overall"] if name == "overall" else report[scope]["by_regime"][name]
                for left, right in ((actual, expected), *((actual["by_age"][age], expected["by_age"][age]) for age in AGES)):
                    require(all(left[key] == right[key] for key in SUPPORT_KEYS), "identical scored domains across models")
    result = [{"name": "technical_complete", "value": technical_complete, "relation": "==",
               "threshold": True, "passes": technical_complete}]

    def add(name, value, relation, threshold):
        require(math.isfinite(value) and math.isfinite(threshold), "finite criterion")
        result.append({"name": name, "value": value, "relation": relation, "threshold": threshold,
                       "passes": bool(value >= threshold if relation == ">=" else value <= threshold)})

    def mean(kind, scope, regime, key):
        return math.fsum(by[kind, seed][scope]["by_regime"][regime][key] for seed in SEEDS) / len(SEEDS)

    agreement, gap = "episode_weighted_agreement", "episode_weighted_raw_gap"
    for regime in REGIMES:
        held = hold["postcorrection"]["by_regime"][regime]
        for age in AGES:
            add(f"{regime}.age{age}.case_support", held["by_age"][age]["supported_case_count"], ">=", 4)
        for seed in SEEDS:
            candidate = by["innovation", seed]["postcorrection"]["by_regime"][regime]
            add(f"{regime}.{seed}.agreement_vs_hold", candidate[agreement], ">=", held[agreement])
            add(f"{regime}.{seed}.gap_vs_hold", candidate[gap], "<=", .8 * held[gap])
            for age in AGES:
                add(f"{regime}.{seed}.age{age}.gap_vs_hold", candidate["by_age"][age][gap], "<=",
                    held["by_age"][age][gap])
        for control in FAMILIES[1:]:
            add(f"{regime}.mean_agreement_vs_{control}", mean("innovation", "postcorrection", regime, agreement),
                ">=", mean(control, "postcorrection", regime, agreement))
            add(f"{regime}.mean_gap_vs_{control}", mean("innovation", "postcorrection", regime, gap),
                "<=", .9 * mean(control, "postcorrection", regime, gap))
        add(f"{regime}.full.mean_agreement_vs_reset_direct", mean("innovation", "full", regime, agreement),
            ">=", mean("reset_direct", "full", regime, agreement))
        add(f"{regime}.full.mean_gap_vs_reset_direct", mean("innovation", "full", regime, gap),
            "<=", mean("reset_direct", "full", regime, gap))
    require(len(result) == len({row["name"] for row in result}) == 53, "exact 53-condition gate")
    return result
