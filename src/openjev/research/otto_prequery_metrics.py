"""Fixed-path forecast metrics and later-query calibration, never control returns.

Nonquery metrics reuse the unchanged qualified helper. Prior calibration uses
all four raw teacher coordinates only at queries after the initial anchor.
Every grouping retains its complete declared episode denominator. No files,
models, random draws, threshold searches or candidate selection occur here.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from openjev.research import otto_cross_query_metrics as original
from openjev.research.otto_cross_query_metrics import forecast_metrics  # noqa: F401
from openjev.research.otto_score_forecast_data import HORIZON, array, owned, require

VERSION = "otto-prequery-metrics-v1"
FAMILIES = ("innovation_aux", "innovation_mse", "innovation_gru_mse", "innovation_gru_aux")
SEEDS = (275000001, 275000002, 275000003)
REGIMES, ARMS, AGES = original.REGIMES, original.ARMS, original.AGES
PRIOR_MASK = "query and absolute_step >= 4"
PRIOR_SUPPORT_KEYS = tuple("prior_rows" if key == "nonquery_rows" else key for key in original.SUPPORT_KEYS)


def _history(history):
    ids, regimes = history["episode_ids"], history["episode_regimes"]
    require(isinstance(ids, tuple) and bool(ids) and all(type(v) is str and v for v in ids)
            and len(set(ids)) == len(ids), "unique complete episode identities")
    require(isinstance(regimes, tuple) and len(regimes) == len(ids)
            and all(type(v) is str and v for v in regimes), "aligned complete regimes")
    offsets = array(history["episode_offsets"], np.int64, (len(ids) + 1,), "episode offsets")
    require(offsets[0] == 0 and bool(((np.diff(offsets) >= 1) & (np.diff(offsets) <= HORIZON)).all()),
            "complete nonempty episode geometry")
    total = int(offsets[-1])
    mask = np.zeros(total, np.bool_)
    expected_queries = np.zeros(total, np.bool_)
    for low, high in itertools.pairwise(offsets):
        expected_queries[low:high:4] = True
        mask[low + 4:high:4] = True
    array(history["query_mask"], np.bool_, (total,), "true-query mask")
    require(np.array_equal(history["query_mask"], expected_queries), "every original period-four query")
    array(history["query_scores"], np.float32, (total, 4), "true-query scores")
    return ids, regimes, offsets, mask


def validate_prediction_support(history, prior, prior_mask):
    """Exact later-query support; inactive prior values may be NaN poison.

The complete caller-authenticated history defines the support. No query can be
removed because of its prediction, teacher score, target-window sample or mask.
    """
    _, _, offsets, expected = _history(history)
    array(prior, np.float32, (int(offsets[-1]), 4), "flat prior predictions")
    array(prior_mask, np.bool_, expected.shape, "flat prior support")
    require(np.array_equal(prior_mask, expected), "exact complete later-query prediction mask")
    require(bool(np.isfinite(prior[expected]).all()), "finite active prior predictions")
    return owned(expected)


def prior_metrics(history, prior, identities):
    """All-four raw-unit MSE, equal episodes then equal later queries per episode.

TRAIN callers supply all 54 episodes and VALID callers all 36; this pure helper
also accepts an explicitly declared subset for collector breakdowns and tests.
It never replaces that declared denominator with the supported-episode count.
The first query and every nonquery are excluded before any score arithmetic.
    """
    ids, regimes, offsets, expected = _history(history)
    mask = validate_prediction_support(history, prior, expected)
    cases, collectors = original._identities(identities, ids, regimes)
    require(bool(np.isfinite(history["query_scores"][mask]).all()), "finite later-query teacher vectors")
    losses = []
    for low, high in itertools.pairwise(offsets):
        episode = []
        for row in range(int(low) + 4, int(high), 4):
            p = tuple(float(x) for x in prior[row])
            q = tuple(float(x) for x in history["query_scores"][row])
            pm, qm = math.fsum(p) / 4, math.fsum(q) / 4
            episode.append(math.fsum(((a - pm) - (b - qm)) ** 2 for a, b in zip(p, q, strict=True)) / 4)
        losses.append(episode)

    def group(indices):
        supported = [i for i in indices if losses[i]]
        numerator = math.fsum(math.fsum(losses[i]) / len(losses[i]) for i in supported)
        declared_cases, supported_cases = {cases[i] for i in indices}, {cases[i] for i in supported}

        def records(values):
            return [{"regime": regime, "case": case} for regime, case in sorted(values)]

        return {"episodes": len(indices), "supported_episodes": len(supported),
            "zero_support_episode_ids": [ids[i] for i in indices if not losses[i]],
            "prior_rows": sum(len(losses[i]) for i in indices), "weight_mass": len(supported) / len(indices),
            "episode_weighted_centered_mse": numerator / len(indices),
            "supported_episode_centered_mse": numerator / len(supported) if supported else None,
            "declared_case_count": len(declared_cases), "supported_case_count": len(supported_cases),
            "supported_cases": records(supported_cases), "zero_support_cases": records(declared_cases - supported_cases)}

    return {"version": VERSION, "scope": "raw-score calibration on fixed paths; no autonomous efficacy",
        "prior_mask": PRIOR_MASK, "coordinates": "all four actions; no legal mask; separately centered",
        "overall": group(tuple(range(len(ids)))),
        "by_regime": {regime: group(tuple(i for i, r in enumerate(regimes) if r == regime))
                      for regime in dict.fromkeys(regimes)},
        "by_case": [{"regime": regime, "case": case,
                     **group(tuple(i for i, c in enumerate(cases) if c == (regime, case)))}
                    for regime, case in sorted(set(cases))],
        "by_episode": [{"episode_id": ids[i], "regime": regimes[i], "case": cases[i][1],
                        "arm": collectors[i], **group((i,))} for i in range(len(ids))]}


def _validate_prior(report):
    require(isinstance(report, dict) and report.get("version") == VERSION
            and report.get("prior_mask") == PRIOR_MASK, "prequery metric version/domain")
    require(set(report["by_regime"]) == set(REGIMES), "both prior metric regimes")
    for group in (report["overall"], *report["by_regime"].values()):
        for key in ("episodes", "supported_episodes", "prior_rows", "declared_case_count", "supported_case_count"):
            require(type(group[key]) is int and group[key] >= 0, "integer prior support")
        require(group["episodes"] > 0 and group["supported_episodes"] <= group["episodes"]
                and group["supported_case_count"] <= group["declared_case_count"], "bounded prior support")
        value = group["episode_weighted_centered_mse"]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite prior MSE")
        require(group["weight_mass"] == group["supported_episodes"] / group["episodes"], "fixed prior denominator")


def criteria(models, hold, *, technical_complete):
    """All 55 fixed conditions; a caller-owned closure Boolean is not evidence.

Each of 12 records supplies family, seed, metrics (unchanged nonquery report)
and prior_metrics. Six learned-control gap comparisons require both the 0.9
ratio and strict improvement, so a zero/zero tie does not pass. No extra fit or
seed is selected. Technical closure remains False until the original processes
and saved audit have completed successfully, as established by the caller.
    """
    require(type(technical_complete) is bool, "Boolean technical closure")
    require(isinstance(models, (tuple, list)) and len(models) == 12, "all twelve fixed cells")
    by = {}
    for model in models:
        require(isinstance(model, dict) and model.get("family") in FAMILIES
                and type(model.get("seed")) is int and model["seed"] in SEEDS, "fixed cell and fit seed")
        key = model["family"], model["seed"]
        require(key not in by, "unique cell/seed")
        by[key] = model
    require(set(by) == {(kind, seed) for kind in FAMILIES for seed in SEEDS}, "complete paired membership")
    original._validate_report(hold)
    reference_prior = by[FAMILIES[0], SEEDS[0]]["prior_metrics"]
    for model in by.values():
        report, prior = model["metrics"], model["prior_metrics"]
        original._validate_report(report)
        _validate_prior(prior)
        for scope in ("full", "postcorrection"):
            for name in ("overall", *REGIMES):
                expected = hold[scope]["overall"] if name == "overall" else hold[scope]["by_regime"][name]
                actual = report[scope]["overall"] if name == "overall" else report[scope]["by_regime"][name]
                for left, right in ((actual, expected), *((actual["by_age"][a], expected["by_age"][a]) for a in AGES)):
                    require(all(left[k] == right[k] for k in original.SUPPORT_KEYS), "identical nonquery scored domains")
                p = prior["overall"] if name == "overall" else prior["by_regime"][name]
                ref = reference_prior["overall"] if name == "overall" else reference_prior["by_regime"][name]
                require(all(p[k] == ref[k] for k in PRIOR_SUPPORT_KEYS), "identical prior scored domains")
                require(p["episodes"] == actual["episodes"] and p["declared_case_count"] == actual["declared_case_count"],
                        "same complete episodes for prior and nonquery metrics")
    rows = [{"name": "technical_complete", "value": technical_complete, "relation": "==",
             "threshold": True, "passes": technical_complete}]

    def add(name, value, relation, threshold, *, strict=None):
        require(math.isfinite(value) and math.isfinite(threshold), "finite criterion")
        row = {"name": name, "value": value, "relation": relation, "threshold": threshold,
               "passes": bool(value >= threshold if relation == ">=" else value <= threshold)}
        if strict is not None:
            require(math.isfinite(strict), "finite strict comparison")
            row.update(relation="<= and <", strict_upper_bound=strict, passes=bool(value <= threshold and value < strict))
        rows.append(row)

    def mean(kind, scope, regime, key):
        return math.fsum(by[kind, seed]["metrics"][scope]["by_regime"][regime][key] for seed in SEEDS) / len(SEEDS)

    def prior_mean(kind, regime):
        return math.fsum(by[kind, seed]["prior_metrics"]["by_regime"][regime]["episode_weighted_centered_mse"]
                         for seed in SEEDS) / len(SEEDS)

    candidate_kind = FAMILIES[0]
    agreement, gap = "episode_weighted_agreement", "episode_weighted_raw_gap"
    for regime in REGIMES:
        held = hold["postcorrection"]["by_regime"][regime]
        for age in AGES:
            add(f"{regime}.age{age}.case_support", held["by_age"][age]["supported_case_count"], ">=", 4)
        for seed in SEEDS:
            candidate = by[candidate_kind, seed]["metrics"]["postcorrection"]["by_regime"][regime]
            add(f"{regime}.{seed}.agreement_vs_hold", candidate[agreement], ">=", held[agreement])
            add(f"{regime}.{seed}.gap_vs_hold", candidate[gap], "<=", .8 * held[gap])
            for age in AGES:
                add(f"{regime}.{seed}.age{age}.gap_vs_hold", candidate["by_age"][age][gap], "<=", held["by_age"][age][gap])
        for control in FAMILIES[1:]:
            add(f"{regime}.mean_agreement_vs_{control}", mean(candidate_kind, "postcorrection", regime, agreement),
                ">=", mean(control, "postcorrection", regime, agreement))
            comparison = mean(control, "postcorrection", regime, gap)
            add(f"{regime}.mean_gap_vs_{control}", mean(candidate_kind, "postcorrection", regime, gap),
                "<=", .9 * comparison, strict=comparison)
        add(f"{regime}.full.mean_agreement_vs_innovation_mse", mean(candidate_kind, "full", regime, agreement),
            ">=", mean("innovation_mse", "full", regime, agreement))
        add(f"{regime}.full.mean_gap_vs_innovation_mse", mean(candidate_kind, "full", regime, gap),
            "<=", mean("innovation_mse", "full", regime, gap))
        add(f"{regime}.mean_prior_mse_vs_innovation_mse", prior_mean(candidate_kind, regime),
            "<=", .8 * prior_mean("innovation_mse", regime))
    require(len(rows) == len({r["name"] for r in rows}) == 55, "exact 55-condition gate")
    return rows
