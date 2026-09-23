"""Saved-score ranking decomposition on the complete separate-prior VALID cohort.

Pure arithmetic only: no files, models, sampling, or scientific continuation gate.
The caller authenticates saved inputs. Query-prior MSE is a different population
and is deliberately not recomputed or conflated with nonquery action ranking.
"""
from __future__ import annotations

import math

import numpy as np

from openjev.research import otto_score_forecast_data as base

VERSION = "otto-ranking-diagnosis-metrics-v1"
FAMILIES = (
    "innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse", "innovation_separate_aux",
    "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux",
)
SEEDS = (285000001, 285000002, 285000003)
REGIMES = ("lambda3", "lambda4")
ARMS = ("analytic", "neural", "period4_hold")
SCOPES = ("full", "initial", "postcorrection", "post_age1", "post_age2", "post_age3")
CELLS = ("CC", "CW", "WC", "WW")
METRICS = (
    "agreement", "raw_gap", "first_argmin_match", "legal_centered_mse",
    "pair_mse_best_best", "pair_mse_best_rest", "pair_mse_rest_rest",
    "near_all_best_mass", "near_no_best_mass", "near_mixed_correct_mass", "near_mixed_wrong_mass",
    "strict_argmin_raw_gap", "strict_argmin_prediction_margin", "strict_argmin_error_difference",
    "selection_differs_from_strict_mass",
)
PAIRED_METRICS = (
    "agreement", "raw_gap", "legal_centered_mse", "pair_mse_best_best", "pair_mse_best_rest",
    "pair_mse_rest_rest", "near_no_best_mass", "near_mixed_wrong_mass",
)
require = base.require


def contrasts():
    """Twelve fixed directed comparisons; every comparison uses all three seeds."""
    rows = []
    for architecture in ("innovation", "gru"):
        for left, right in (("shared_mse", "shared_aux"), ("separate_mse", "separate_aux"),
                            ("shared_mse", "separate_mse"), ("shared_aux", "separate_aux")):
            rows.append((f"{architecture}.{left}_to_{right}", f"{architecture}_{left}", f"{architecture}_{right}"))
    for suffix in ("shared_mse", "shared_aux", "separate_mse", "separate_aux"):
        rows.append((f"innovation_to_gru.{suffix}", f"innovation_{suffix}", f"gru_{suffix}"))
    return tuple(rows)


def groups(identities):
    """All 21 declared groups, independent of metric values and support."""
    result = [("overall", tuple(range(36)))]
    for regime in REGIMES:
        result.append((regime, tuple(i for i, r in enumerate(identities) if r["regime"] == regime)))
        for arm in ARMS:
            result.append((regime + "/" + arm, tuple(i for i, r in enumerate(identities)
                                                     if r["regime"] == regime and r["arm"] == arm)))
        for case in range(6):
            result.append((f"{regime}/case{case}", tuple(i for i, r in enumerate(identities)
                                                       if r["regime"] == regime and r["case"] == case)))
    return tuple(result)


def _near(scores, legal):
    minimum = np.where(legal, scores, np.float32(np.inf)).min(axis=1)
    with np.errstate(over="ignore"):
        near = legal & ((scores - minimum[:, None]) < np.float32(1e-10))
    require(bool(near.any(axis=1).all()), "nonempty strict float32 near-minimum sets")
    return near, minimum


def _teacher(q, legal):
    near, minimum = _near(q, legal)
    allowed = [tuple(int(a) for a in np.flatnonzero(row)) for row in legal]
    q64 = q.astype(np.float64)
    center = np.asarray([math.fsum(float(row[a]) for a in actions) / len(actions)
                         for row, actions in zip(q, allowed, strict=True)], np.float64)
    alternatives = legal & ~near
    supported = alternatives.any(axis=1)
    margin = np.zeros(len(q), np.float64)
    margin[supported] = np.where(alternatives[supported], q64[supported], np.inf).min(axis=1) - minimum[supported].astype(np.float64)
    return {"q": q, "q64": q64, "legal": legal, "allowed": allowed, "near": near,
            "minimum": minimum.astype(np.float64), "centered": q64 - center[:, None],
            "exact_action": np.where(legal, q, np.float32(np.inf)).argmin(axis=1),
            "margin_support": supported, "margin": margin}


def _row_metrics(p, teacher):
    q, legal, optimal = teacher["q64"], teacher["legal"], teacher["near"]
    near, _ = _near(p, legal)
    chosen = near.argmax(axis=1)
    exact = np.where(legal, p, np.float32(np.inf)).argmin(axis=1)
    first_teacher = optimal.argmax(axis=1)
    index = np.arange(len(p))
    correct = optimal[index, chosen]
    p64, error = p.astype(np.float64), p.astype(np.float64) - q
    means = np.asarray([math.fsum(float(row[a]) for a in actions) / len(actions)
                        for row, actions in zip(p, teacher["allowed"], strict=True)], np.float64)
    centered_error = p64 - means[:, None] - teacher["centered"]
    mse = np.asarray([math.fsum(float(row[a]) ** 2 for a in actions) / len(actions)
                      for row, actions in zip(centered_error, teacher["allowed"], strict=True)], np.float64)
    count = legal.sum(axis=1).astype(np.float64)
    parts = [np.zeros(len(p), np.float64) for _ in range(3)]
    for a in range(4):
        for b in range(a + 1, 4):
            active = legal[:, a] & legal[:, b]
            contribution = ((error[:, b] - error[:, a]) ** 2) / count**2
            both = optimal[:, a] & optimal[:, b]
            one = optimal[:, a] ^ optimal[:, b]
            for part, select in zip(parts, (both, one, ~(both | one)), strict=True):
                part[active & select] += contribution[active & select]
    inside, outside = (near & optimal).any(axis=1), (near & ~optimal).any(axis=1)
    a = teacher["exact_action"]
    strict_gap = q[index, exact] - q[index, a]
    strict_prediction = p64[index, exact] - p64[index, a]
    strict_error = error[index, exact] - error[index, a]
    result = dict(zip(METRICS, (
        correct.astype(np.float64), q[index, chosen] - teacher["minimum"],
        (chosen == first_teacher).astype(np.float64), mse, *parts,
        (inside & ~outside).astype(np.float64), (~inside).astype(np.float64),
        (inside & outside & correct).astype(np.float64), (inside & outside & ~correct).astype(np.float64),
        strict_gap, strict_prediction, strict_error, (chosen != exact).astype(np.float64),
    ), strict=True))
    result["pair_residual"] = np.abs(mse - (parts[0] + parts[1] + parts[2]))
    result["strict_residual"] = np.abs(strict_gap + strict_error - strict_prediction)
    require(all(bool(np.isfinite(v).all()) for v in result.values()), "finite derived score statistics")
    return result


def _average(values, positions, denominator):
    return math.fsum(float(v) for v in values[positions]) / denominator if denominator else 0.


def _margin(teacher, positions, denominator):
    count = int(teacher["margin_support"][positions].sum())
    mass = count / denominator if denominator else 0.
    contribution = _average(teacher["margin"], positions, denominator)
    return {"rows": count, "mass": mass, "contribution": contribution,
            "conditional_mean": contribution / mass if mass else None}


def _individual(values, teacher, positions):
    n = len(positions)
    return {"rows": n, "episodes": 1, "supported_episodes": int(n > 0), "weight_mass": float(n > 0),
            "metrics": {key: _average(values[key], positions, n) for key in METRICS},
            "teacher_margin": _margin(teacher, positions, n),
            "max_residuals": {"pair_mse": float(values["pair_residual"][positions].max()) if n else 0.,
                              "strict_margin": float(values["strict_residual"][positions].max()) if n else 0.}}


def _paired(left, right, teacher, positions, denominator):
    n, cells = len(positions), {}
    for cell in CELLS:
        selected = positions[(left["agreement"][positions] == float(cell[0] == "C"))
                             & (right["agreement"][positions] == float(cell[1] == "C"))]
        baseline = {key: _average(left[key], selected, denominator) for key in PAIRED_METRICS}
        candidate = {key: _average(right[key], selected, denominator) for key in PAIRED_METRICS}
        cells[cell] = {"rows": len(selected), "mass": len(selected) / denominator if denominator else 0.,
                       "baseline": baseline, "candidate": candidate,
                       "delta": {key: candidate[key] - baseline[key] for key in PAIRED_METRICS},
                       "teacher_margin": _margin(teacher, selected, denominator)}
    return {"rows": n, "normalization_rows": denominator, "episodes": 1,
            "supported_episodes": int(n > 0), "normalization_supported_episodes": int(denominator > 0),
            "weight_mass": n / denominator if denominator else 0., "cells": cells}


def _combine_margin(rows, *, seeds=False):
    denominator = len(rows)
    mass = math.fsum(r["mass"] for r in rows) / denominator
    contribution = math.fsum(r["contribution"] for r in rows) / denominator
    counts = math.fsum(r["rows"] for r in rows)
    return {"rows": counts / denominator if seeds else int(counts), "mass": mass,
            "contribution": contribution, "conditional_mean": contribution / mass if mass else None}


def _combine(rows, *, paired=False, seeds=False):
    """Groups average episode contributions; seed means do not create episodes."""
    require(bool(rows), "nonempty declared reduction group")
    n = len(rows)
    count_fields = ["rows", "supported_episodes"]
    if paired:
        count_fields += ["normalization_rows", "normalization_supported_episodes"]
    value = {key: math.fsum(r[key] for r in rows) / n if seeds else sum(r[key] for r in rows)
             for key in count_fields}
    value["episodes"] = rows[0]["episodes"] if seeds else sum(r["episodes"] for r in rows)
    value["weight_mass"] = math.fsum(r["weight_mass"] for r in rows) / n
    if paired:
        cells = {}
        for cell in CELLS:
            selected = [r["cells"][cell] for r in rows]
            cells[cell] = {"rows": math.fsum(r["rows"] for r in selected) / n if seeds else sum(r["rows"] for r in selected),
                           "mass": math.fsum(r["mass"] for r in selected) / n,
                           "teacher_margin": _combine_margin([r["teacher_margin"] for r in selected], seeds=seeds)}
            for field in ("baseline", "candidate", "delta"):
                cells[cell][field] = {key: math.fsum(r[field][key] for r in selected) / n for key in PAIRED_METRICS}
        value["cells"] = cells
    else:
        value["metrics"] = {key: math.fsum(r["metrics"][key] for r in rows) / n for key in METRICS}
        value["teacher_margin"] = _combine_margin([r["teacher_margin"] for r in rows], seeds=seeds)
        value["max_residuals"] = {key: max(r["max_residuals"][key] for r in rows) for key in ("pair_mse", "strict_margin")}
    return value


def _inputs(windows, predictions, identities):
    policies = (("hold", None), *((f, s) for f in FAMILIES for s in SEEDS))
    require(isinstance(predictions, dict) and set(predictions) == set(policies), "all 24 final fits plus one hold")
    ids, regimes, _, _, _, _ = base._metric_inputs(windows, predictions["hold", None])
    require(isinstance(identities, (list, tuple)) and len(identities) == len(ids) == 36, "all 36 declared VALID episodes")
    observed = []
    for i, item in enumerate(identities):
        require(isinstance(item, dict) and item["episode_id"] == ids[i] and item["regime"] == regimes[i]
                and item["regime"] in REGIMES and type(item["case"]) is int and 0 <= item["case"] < 6
                and item["arm"] in ARMS, "aligned originating case and collector identity")
        observed.append((item["regime"], item["case"], item["arm"]))
    require(set(observed) == {(r, c, a) for r in REGIMES for c in range(6) for a in ARMS}, "one path per case and collector")
    target_shape = windows["targets"].shape
    for key in policies:
        p = predictions[key]
        require(isinstance(p, np.ndarray) and p.dtype == np.float32 and p.shape == target_shape
                and bool(np.isfinite(p).all()), "complete finite float32 policy prediction geometry")
        require(p[:, 0].tobytes() == windows["query_scores"].tobytes(), "observed query rows are not forecast evidence")
        padding = p[~windows["valid_mask"]]
        require(padding.tobytes() == np.zeros(padding.shape, np.float32).tobytes(), "exact zero padding")
    held = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
    held[~windows["valid_mask"]] = 0
    require(predictions["hold", None].tobytes() == held.tobytes(), "exact held-query baseline")
    row, age = np.nonzero(windows["nonquery_mask"])
    return policies, row, age, windows["episode_index"][row], windows["step_offsets"][row] + age


def analyze(windows, predictions_by_model, identities):
    """Return JSON-compatible fixed-cohort summaries, without changing inputs.

    ``windows`` is the complete qualified validation-window dict (arrays plus
    metadata). ``predictions_by_model`` maps (family, seed) to float32[W,4,4],
    including ('hold', None). Identities are the original 36 ordered VALID
    episode_id/regime/case/arm records. Source and receipt authentication belong
    to the caller. All scopes, groups, contrasts, zero cells and seeds survive.
    """
    policies, row, age, episode_index, steps = _inputs(windows, predictions_by_model, identities)
    teacher = _teacher(windows["targets"][row, age], windows["legal"][row, age])
    scopes = {"full": np.ones(len(row), np.bool_), "initial": steps <= 3, "postcorrection": steps >= 5}
    scopes.update({f"post_age{a}": (steps >= 5) & (age == a) for a in (1, 2, 3)})
    positions = {(scope, i): np.flatnonzero(mask & (episode_index == i))
                 for scope, mask in scopes.items() for i in range(36)}
    membership = groups(identities)
    output = {name: [] for name in ("individual_episodes", "individual_groups", "individual_seed_means",
        "paired_episodes", "paired_groups", "paired_seed_means", "phase_episodes", "phase_groups", "phase_seed_means")}
    cache, individual_index, paired_index, phase_index = {}, {}, {}, {}
    identity_rows = [{"episode_index": i, **{k: r[k] for k in ("episode_id", "regime", "case", "arm")}}
                     for i, r in enumerate(identities)]
    for family, seed in policies:
        values = _row_metrics(predictions_by_model[family, seed][row, age], teacher)
        cache[family, seed] = values
        for scope in SCOPES:
            stats = [_individual(values, teacher, positions[scope, i]) for i in range(36)]
            output["individual_episodes"].extend({"family": family, "seed": seed, "scope": scope,
                **identity_rows[i], "stats": value} for i, value in enumerate(stats))
            for group, selected in membership:
                value = _combine([stats[i] for i in selected])
                output["individual_groups"].append({"family": family, "seed": seed, "scope": scope, "group": group, "stats": value})
                individual_index[family, seed, scope, group] = value
    for family in FAMILIES:
        for scope in SCOPES:
            for group, _ in membership:
                output["individual_seed_means"].append({"family": family, "seeds": list(SEEDS), "scope": scope,
                    "group": group, "stats": _combine([individual_index[family, s, scope, group] for s in SEEDS], seeds=True)})
    for contrast, left, right in contrasts():
        for seed in SEEDS:
            common = {"contrast": contrast, "baseline_family": left, "candidate_family": right, "seed": seed}
            for scope in SCOPES:
                stats = [_paired(cache[left, seed], cache[right, seed], teacher, positions[scope, i],
                                 len(positions[scope, i])) for i in range(36)]
                output["paired_episodes"].extend({**common, "scope": scope, **identity_rows[i], "stats": value}
                                                 for i, value in enumerate(stats))
                for group, selected in membership:
                    value = _combine([stats[i] for i in selected], paired=True)
                    output["paired_groups"].append({**common, "scope": scope, "group": group, "stats": value})
                    paired_index[contrast, seed, scope, group] = value
            for phase in ("initial", "postcorrection"):
                stats = [_paired(cache[left, seed], cache[right, seed], teacher, positions[phase, i],
                                 len(positions["full", i])) for i in range(36)]
                output["phase_episodes"].extend({**common, "phase": phase, "normalization_scope": "full",
                    **identity_rows[i], "stats": value} for i, value in enumerate(stats))
                for group, selected in membership:
                    value = _combine([stats[i] for i in selected], paired=True)
                    output["phase_groups"].append({**common, "phase": phase, "normalization_scope": "full", "group": group, "stats": value})
                    phase_index[contrast, seed, phase, group] = value
        common = {"contrast": contrast, "baseline_family": left, "candidate_family": right, "seeds": list(SEEDS)}
        for scope in SCOPES:
            for group, _ in membership:
                output["paired_seed_means"].append({**common, "scope": scope, "group": group,
                    "stats": _combine([paired_index[contrast, s, scope, group] for s in SEEDS], paired=True, seeds=True)})
        for phase in ("initial", "postcorrection"):
            for group, _ in membership:
                output["phase_seed_means"].append({**common, "phase": phase, "normalization_scope": "full", "group": group,
                    "stats": _combine([phase_index[contrast, s, phase, group] for s in SEEDS], paired=True, seeds=True)})
    residuals = {"pair_mse": max((float(r["pair_residual"].max()) if len(row) else 0. for r in cache.values()), default=0.),
                 "strict_margin": max((float(r["strict_residual"].max()) if len(row) else 0. for r in cache.values()), default=0.)}
    return {"version": VERSION, "scope": "saved fixed-path nonquery teacher imitation; descriptive only, no continuation gate",
        "definitions": {"scopes": list(SCOPES), "groups": [g for g, _ in membership], "cells": list(CELLS),
            "individual_metrics": list(METRICS), "paired_metrics": list(PAIRED_METRICS),
            "episode_weights": "1/(declared episodes * scope rows in that episode); empty episodes contribute zero",
            "phase_weights": "full-scope denominator, even when numerator selects initial or postcorrection",
            "seed_means": "arithmetic means of all three fits; counts are means, not additional episodes",
            "margin": "nearest teacher nonbest minus exact legal minimum; None conditional mean when no alternative support",
            "strict_witness": "exact argmins are separate from actual float32 near-minimum selection",
            "pair_mse": "sum squared differences of score errors / legal_count**2, partitioned by teacher near-best membership",
            "prior_scope": "query all-four prior MSE is excluded; it is a distinct population and readout"},
        "counts": {"policies": 25, "fits": 24, "episodes": 36, "originating_cases": 12, "groups": 21,
            "scopes": 6, "contrast_types": 12, "paired_contrasts": 36, "windows": len(windows["lengths"]),
            "nonquery_rows": len(row), **{name: len(rows) for name, rows in output.items()}},
        **output, "residuals": residuals}
