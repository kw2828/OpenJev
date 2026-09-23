"""Independently check saved ranking diagnostics; no model or environment calls.

Only the prospective producer's hash-bound byte/process lifecycle is shared.
Numerical calculations below do not import the diagnostic metric reducer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_ranking_diagnosis.py"
RUNNER = "scripts/diagnose_otto_ranking.py"
VERSION = "otto-ranking-diagnosis-saved-audit-v1"
SCOPES = ("full", "initial", "postcorrection", "post_age1", "post_age2", "post_age3")
METRICS = ("agreement", "raw_gap", "first_argmin_match", "legal_centered_mse",
           "pair_mse_best_best", "pair_mse_best_rest", "pair_mse_rest_rest",
           "near_all_best_mass", "near_no_best_mass", "near_mixed_correct_mass",
           "near_mixed_wrong_mass", "strict_argmin_raw_gap", "strict_argmin_prediction_margin",
           "strict_argmin_error_difference", "selection_differs_from_strict_mass")
PAIRED_METRICS = ("agreement", "raw_gap", "legal_centered_mse", "pair_mse_best_best",
                  "pair_mse_best_rest", "pair_mse_rest_rest", "near_no_best_mass", "near_mixed_wrong_mass")
CELLS = ("CC", "CW", "WC", "WW")
FAMILIES = tuple(f"{a}_{r}_{o}" for a in ("innovation", "gru")
                 for r in ("shared", "separate") for o in ("mse", "aux"))
SEEDS = (285000001, 285000002, 285000003)
POLICIES = (("hold", None), *((family, seed) for family in FAMILIES for seed in SEEDS))
MARGIN_COLUMN = len(METRICS)


def require(value, message):
    if not value:
        raise ValueError(message)


def f32(value):
    """Round one scalar subtraction exactly as binary32, including overflow."""
    try:
        return struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def row_statistics(teacher, prediction, legal):
    """Scalar four-action reference, with separate near-tie/exact witnesses."""
    for values in (teacher, prediction):
        require(isinstance(values, np.ndarray) and values.dtype == np.float32 and values.shape == (4,)
                and np.isfinite(values).all(), "finite four-action float32 scores")
    require(isinstance(legal, np.ndarray) and legal.dtype == np.bool_ and legal.shape == (4,)
            and legal.any(), "nonempty four-action Boolean mask")
    ids = [a for a in range(4) if legal[a]]
    q, p = [float(x) for x in teacher], [float(x) for x in prediction]
    qmin, pmin = min(q[a] for a in ids), min(p[a] for a in ids)
    epsilon = f32(1e-10)
    optimum = [a for a in ids if f32(q[a] - qmin) < epsilon]
    predicted = [a for a in ids if f32(p[a] - pmin) < epsilon]
    actual = predicted[0]
    exact_q = next(a for a in ids if q[a] == qmin)
    exact_p = next(a for a in ids if p[a] == pmin)
    qmean = math.fsum(q[a] for a in ids) / len(ids)
    pmean = math.fsum(p[a] for a in ids) / len(ids)
    mse = math.fsum(((p[a] - pmean) - (q[a] - qmean)) ** 2 for a in ids) / len(ids)
    parts = {key: [] for key in ("best_best", "best_rest", "rest_rest")}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            nbest = int(a in optimum) + int(b in optimum)
            key = ("rest_rest", "best_rest", "best_best")[nbest]
            parts[key].append(((p[b] - q[b]) - (p[a] - q[a])) ** 2 / len(ids) ** 2)
    # Preserve the declared six-pair accumulation order for its diagnostic
    # roundoff witness rather than silently changing the reported arithmetic.
    components = {}
    for key, values in parts.items():
        total = 0.
        for value in values:
            total += value
        components[key] = total
    all_best = all(a in optimum for a in predicted)
    no_best = all(a not in optimum for a in predicted)
    correct = actual in optimum
    gap = q[exact_p] - q[exact_q]
    error_difference = (p[exact_p] - q[exact_p]) - (p[exact_q] - q[exact_q])
    prediction_margin = p[exact_p] - p[exact_q]
    outside = [a for a in ids if a not in optimum]
    result = {"agreement": float(correct), "raw_gap": q[actual] - qmin,
        "first_argmin_match": float(actual == optimum[0]), "legal_centered_mse": mse,
        **{"pair_mse_" + key: value for key, value in components.items()},
        "near_all_best_mass": float(all_best), "near_no_best_mass": float(no_best),
        "near_mixed_correct_mass": float(not all_best and not no_best and correct),
        "near_mixed_wrong_mass": float(not all_best and not no_best and not correct),
        "strict_argmin_raw_gap": gap, "strict_argmin_prediction_margin": prediction_margin,
        "strict_argmin_error_difference": error_difference,
        "selection_differs_from_strict_mass": float(actual != exact_p)}
    return {"metrics": result, "action": actual, "correct": correct,
            "teacher_margin": min(q[a] for a in outside) - qmin if outside else None,
            "max_residuals": {"pair_mse": abs(mse - ((components["best_best"] + components["best_rest"])
                                                     + components["rest_rest"])),
                              "strict_margin": abs(gap + error_difference - prediction_margin)}}


def matrix(teacher, prediction, legal, check=lambda: None):
    require(teacher.shape == prediction.shape == legal.shape and teacher.ndim == 2
            and teacher.shape[1] == 4, "aligned row matrices")
    values = np.empty((len(teacher), len(METRICS) + 3), np.float64)
    for i in range(len(teacher)):
        if i % 512 == 0:
            check()
        row = row_statistics(teacher[i], prediction[i], legal[i])
        values[i] = [*(row["metrics"][key] for key in METRICS),
                     np.nan if row["teacher_margin"] is None else row["teacher_margin"],
                     row["max_residuals"]["pair_mse"], row["max_residuals"]["strict_margin"]]
    return values


def scope_rows(steps, scope):
    if scope == "full":
        return np.ones(len(steps), dtype=np.bool_)
    if scope == "initial":
        return steps <= 3
    if scope == "postcorrection":
        return steps >= 5
    require(scope in SCOPES[3:], "known age scope")
    return (steps >= 5) & (steps % 4 == int(scope[-1]))


def groups(identities):
    yield "overall", list(range(len(identities)))
    for regime in ("lambda3", "lambda4"):
        selected = [i for i, row in enumerate(identities) if row["regime"] == regime]
        yield regime, selected
        for arm in ("analytic", "neural", "period4_hold"):
            yield regime + "/" + arm, [i for i in selected if identities[i]["arm"] == arm]
        for case in range(6):
            yield regime + "/case" + str(case), [i for i in selected if identities[i]["case"] == case]


def contrasts():
    for architecture in ("innovation", "gru"):
        for source, target in (("shared_mse", "shared_aux"), ("separate_mse", "separate_aux"),
                               ("shared_mse", "separate_mse"), ("shared_aux", "separate_aux")):
            yield architecture + "." + source + "_to_" + target, architecture + "_" + source, architecture + "_" + target
    for suffix in ("shared_mse", "shared_aux", "separate_mse", "separate_aux"):
        yield "innovation_to_gru." + suffix, "innovation_" + suffix, "gru_" + suffix


def margin(values, denominator):
    supported = [float(value) for value in values if not math.isnan(value)]
    contribution = math.fsum(supported) / denominator if denominator else 0.
    mass = len(supported) / denominator if denominator else 0.
    return {"rows": len(supported), "mass": mass, "contribution": contribution,
            "conditional_mean": contribution / mass if mass else None}


def individual(values):
    n = len(values)
    return {"rows": n, "episodes": 1, "supported_episodes": int(n > 0), "weight_mass": float(n > 0),
            "metrics": {key: math.fsum(values[:, i]) / n if n else 0. for i, key in enumerate(METRICS)},
            "teacher_margin": margin(values[:, MARGIN_COLUMN], n),
            "max_residuals": {"pair_mse": max(values[:, -2], default=0.),
                              "strict_margin": max(values[:, -1], default=0.)}}


def paired(base, candidate, denominator):
    require(base.shape == candidate.shape and type(denominator) is int and denominator >= len(base),
            "paired rows and declared normalization")
    cells = {}
    for cell in CELLS:
        mask = (base[:, 0] == (cell[0] == "C")) & (candidate[:, 0] == (cell[1] == "C"))
        b, c = base[mask], candidate[mask]
        baseline = {key: math.fsum(b[:, METRICS.index(key)]) / denominator if denominator else 0. for key in PAIRED_METRICS}
        proposed = {key: math.fsum(c[:, METRICS.index(key)]) / denominator if denominator else 0. for key in PAIRED_METRICS}
        cells[cell] = {"rows": len(b), "mass": len(b) / denominator if denominator else 0.,
                       "baseline": baseline, "candidate": proposed,
                       "delta": {key: proposed[key] - baseline[key] for key in PAIRED_METRICS},
                       "teacher_margin": margin(b[:, MARGIN_COLUMN], denominator)}
    return {"rows": len(base), "normalization_rows": denominator, "episodes": 1,
            "supported_episodes": int(len(base) > 0), "normalization_supported_episodes": int(denominator > 0),
            "weight_mass": len(base) / denominator if denominator else 0., "cells": cells}


def combine(stats, *, seeds=False):
    """Episode totals and equal means, followed by unweighted seed means."""
    require(bool(stats), "nonempty declared aggregate")
    n = len(stats)

    def mean(values):
        return math.fsum(values) / n

    def combine_margin(values):
        rows = mean([v["rows"] for v in values]) if seeds else sum(v["rows"] for v in values)
        mass = mean([v["mass"] for v in values])
        contribution = mean([v["contribution"] for v in values])
        return {"rows": rows, "mass": mass, "contribution": contribution,
                "conditional_mean": contribution / mass if mass else None}

    result = {}
    for key in ("rows", "normalization_rows", "episodes", "supported_episodes", "normalization_supported_episodes"):
        if key in stats[0]:
            result[key] = mean([s[key] for s in stats]) if seeds else sum(s[key] for s in stats)
    if seeds:
        require(all(s["episodes"] == stats[0]["episodes"] for s in stats), "same episode denominator across seeds")
        result["episodes"] = stats[0]["episodes"]
    result["weight_mass"] = mean([s["weight_mass"] for s in stats])
    if "metrics" in stats[0]:
        result["metrics"] = {key: mean([s["metrics"][key] for s in stats]) for key in METRICS}
        result["teacher_margin"] = combine_margin([s["teacher_margin"] for s in stats])
        result["max_residuals"] = {key: max(s["max_residuals"][key] for s in stats)
                                   for key in ("pair_mse", "strict_margin")}
    else:
        result["cells"] = {}
        for cell in CELLS:
            values = [s["cells"][cell] for s in stats]
            result["cells"][cell] = {
                "rows": mean([v["rows"] for v in values]) if seeds else sum(v["rows"] for v in values),
                "mass": mean([v["mass"] for v in values]),
                **{side: {key: mean([v[side][key] for v in values]) for key in PAIRED_METRICS}
                   for side in ("baseline", "candidate", "delta")},
                "teacher_margin": combine_margin([v["teacher_margin"] for v in values])}
    return result


class Comparison:
    def __init__(self):
        self.checks = 0
        self.numeric_checks = 0
        self.max_absolute_residual = 0.
        self.max_normalized_residual = 0.

    def same(self, expected, actual, path="result"):
        self.checks += 1
        if isinstance(expected, dict):
            require(isinstance(actual, dict) and set(actual) == set(expected), path + ": exact dictionary keys")
            for key, value in expected.items():
                self.same(value, actual[key], path + "." + key)
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, list) and len(actual) == len(expected), path + ": exact list length")
            for i, value in enumerate(expected):
                self.same(value, actual[i], path + f"[{i}]")
        elif expected is None or type(expected) in (str, bool, int):
            require(type(actual) is type(expected) and actual == expected, path + ": exact scalar")
        else:
            require(isinstance(expected, (float, np.floating)) and type(actual) in (int, float)
                    and math.isfinite(expected) and math.isfinite(actual), path + ": finite number")
            residual = abs(float(expected) - actual)
            limit = 1e-12 + 1e-11 * max(abs(float(expected)), abs(actual))
            self.numeric_checks += 1
            self.max_absolute_residual = max(self.max_absolute_residual, residual)
            self.max_normalized_residual = max(self.max_normalized_residual, residual / limit)
            require(residual <= limit, path + ": saved arithmetic agreement")


def geometry(windows, identities):
    require(len(identities) == 36 and len({i["episode_id"] for i in identities}) == 36,
            "all unique declared episodes")
    require({(i["regime"], i["case"], i["arm"]) for i in identities}
            == {(r, c, a) for r in ("lambda3", "lambda4") for c in range(6)
                for a in ("analytic", "neural", "period4_hold")}, "complete originating case/collector cohort")
    require([i["episode_id"] for i in identities] == list(windows["episode_ids"])
            and [i["regime"] for i in identities] == list(windows["episode_regimes"]), "ordered window identities")
    lengths = windows["episode_lengths"]
    require(lengths.dtype == np.int64 and lengths.shape == (36,)
            and ((lengths >= 1) & (lengths <= 2188)).all(), "full original episode horizons")
    expected = [(i, step, min(4, int(length) - step))
                for i, length in enumerate(lengths) for step in range(0, int(length), 4)]
    count = len(expected)
    for key, column in (("episode_index", 0), ("step_offsets", 1), ("lengths", 2)):
        array = windows[key]
        require(array.dtype == np.int64 and array.shape == (count,)
                and array.tolist() == [r[column] for r in expected], "complete chronological windows: " + key)
    active = np.arange(4)[None, :] < windows["lengths"][:, None]
    for key, reference in (("valid_mask", active), ("nonquery_mask", active & (np.arange(4) > 0))):
        array = windows[key]
        require(array.dtype == np.bool_ and array.shape == (count, 4)
                and np.array_equal(array, reference), "exact declared mask: " + key)
    for key, dtype in (("targets", np.float32), ("legal", np.bool_)):
        require(windows[key].shape == (count, 4, 4) and windows[key].dtype == dtype,
                "original target/legal geometry")
    require(np.isfinite(windows["targets"]).all() and windows["legal"][active].any(axis=1).all()
            and not windows["legal"][~active].any(), "finite teacher and only active legal masks")
    require(windows["query_scores"].shape == (count, 4) and windows["query_scores"].dtype == np.float32
            and windows["query_scores"].tobytes() == windows["targets"][:, 0].tobytes(), "exact observed query anchors")
    row, age = np.nonzero(windows["nonquery_mask"])
    steps = windows["step_offsets"][row] + age
    episodes = windows["episode_index"][row]
    require(list(zip(episodes.tolist(), steps.tolist(), strict=True))
            == [(i, t) for i, n in enumerate(lengths) for t in range(int(n)) if t % 4],
            "all disjoint nonquery rows and final tails")
    return row, age, episodes, steps


def published_group(report, scope, group):
    name = "postcorrection" if scope.startswith("post_age") else scope
    if "/" not in group:
        section = report[name]
        value = section["overall"] if group == "overall" else section["by_regime"][group]
    else:
        regime, subgroup = group.split("/", 1)
        if subgroup.startswith("case"):
            value = next(r for r in report[name]["by_case"]
                         if r["regime"] == regime and r["case"] == int(subgroup[4:]))
        else:
            value = report["by_collector"][subgroup][name]["by_regime"][regime]
    return value["by_age"][scope[-1]] if scope.startswith("post_age") else value


def verify_reductions(windows, load_prediction, identities, observed, reference, check=lambda: None):
    """Check every saved record in order; retain compact per-row matrices only."""
    comparison = Comparison()
    row, age, episode_index, steps = geometry(windows, identities)
    table_names = ("individual_episodes", "individual_groups", "individual_seed_means",
                   "paired_episodes", "paired_groups", "paired_seed_means",
                   "phase_episodes", "phase_groups", "phase_seed_means")
    require(set(observed) == {"version", "scope", "definitions", "counts", "residuals", *table_names},
            "complete diagnostic result schema")
    require(observed["version"] == "otto-ranking-diagnosis-metrics-v1"
            and observed["scope"] == "saved fixed-path nonquery teacher imitation; descriptive only, no continuation gate",
            "retrospective descriptive result identity")
    expected_definitions = {"scopes": list(SCOPES), "groups": [g for g, _ in groups(identities)],
        "cells": list(CELLS), "individual_metrics": list(METRICS), "paired_metrics": list(PAIRED_METRICS),
        "episode_weights": "1/(declared episodes * scope rows in that episode); empty episodes contribute zero",
        "phase_weights": "full-scope denominator, even when numerator selects initial or postcorrection",
        "seed_means": "arithmetic means of all three fits; counts are means, not additional episodes",
        "margin": "nearest teacher nonbest minus exact legal minimum; None conditional mean when no alternative support",
        "strict_witness": "exact argmins are separate from actual float32 near-minimum selection",
        "pair_mse": "sum squared differences of score errors / legal_count**2, partitioned by teacher near-best membership",
        "prior_scope": "query all-four prior MSE is excluded; it is a distinct population and readout"}
    comparison.same(expected_definitions, observed["definitions"], "definitions")
    counters = dict.fromkeys(table_names, 0)

    def record(table, value):
        index = counters[table]
        if index % 128 == 0:
            check()
        require(index < len(observed[table]), "missing record: " + table)
        comparison.same(value, observed[table][index], table + f"[{index}]")
        counters[table] += 1

    positions = {(scope, i): np.flatnonzero((episode_index == i) & scope_rows(steps, scope))
                 for scope in SCOPES for i in range(36)}
    membership = list(groups(identities))
    ids = [{"episode_index": i, **{key: item[key] for key in ("episode_id", "regime", "case", "arm")}}
           for i, item in enumerate(identities)]
    q, legal = windows["targets"][row, age], windows["legal"][row, age]
    cache, singles = {}, {}
    published = {(m["family"], m["seed"]): m["metrics"] for m in reference["models"]}
    published["hold", None] = reference["hold"]
    require(set(published) == set(POLICIES), "complete original audited model reports")
    for family, seed in POLICIES:
        check()
        prediction = load_prediction(family, seed)
        require(prediction.dtype == np.float32 and prediction.shape == windows["targets"].shape
                and np.isfinite(prediction).all(), "complete finite saved predictions")
        require(prediction[:, 0].tobytes() == windows["query_scores"].tobytes(), "unchanged actual query readout")
        padding = prediction[~windows["valid_mask"]]
        require(padding.tobytes() == np.zeros_like(padding).tobytes(), "positive-zero prediction padding")
        if family == "hold":
            held = np.repeat(windows["query_scores"][:, None], 4, axis=1)
            held[~windows["valid_mask"]] = 0
            require(held.tobytes() == prediction.tobytes(), "exact held-score control")
        values = matrix(q, prediction[row, age], legal, check)
        cache[family, seed] = values
        for scope in SCOPES:
            stats = [individual(values[positions[scope, i]]) for i in range(36)]
            for i, value in enumerate(stats):
                record("individual_episodes", {"family": family, "seed": seed, "scope": scope, **ids[i], "stats": value})
            for group, selected in membership:
                value = combine([stats[i] for i in selected])
                record("individual_groups", {"family": family, "seed": seed, "scope": scope, "group": group, "stats": value})
                singles[family, seed, scope, group] = value
                original = published_group(published[family, seed], scope, group)
                for name, field in (("agreement", "agreement"), ("raw_gap", "raw_gap"),
                                    ("first_argmin_match", "first_argmin_match"), ("legal_centered_mse", "centered_mse")):
                    comparison.same(value["metrics"][name], original["episode_weighted_" + field], "original audited " + name)
                comparison.same(value["rows"], original["nonquery_rows"], "original audited rows")
                comparison.same(value["supported_episodes"], original["supported_episodes"], "original audited support")
    for family in FAMILIES:
        for scope in SCOPES:
            for group, _ in membership:
                record("individual_seed_means", {"family": family, "seeds": list(SEEDS), "scope": scope,
                    "group": group, "stats": combine([singles[family, seed, scope, group] for seed in SEEDS], seeds=True)})
    for name, baseline, candidate in contrasts():
        paired_groups, phase_groups = {}, {}
        for seed in SEEDS:
            common = {"contrast": name, "baseline_family": baseline, "candidate_family": candidate, "seed": seed}
            for phase_mode in (False, True):
                for scope in (("initial", "postcorrection") if phase_mode else SCOPES):
                    table = "phase" if phase_mode else "paired"
                    identity = {"phase": scope, "normalization_scope": "full"} if phase_mode else {"scope": scope}
                    stats = []
                    for i in range(36):
                        chosen = positions[scope, i]
                        denominator = len(positions["full" if phase_mode else scope, i])
                        value = paired(cache[baseline, seed][chosen], cache[candidate, seed][chosen], denominator)
                        stats.append(value)
                        record(table + "_episodes", {**common, **identity, **ids[i], "stats": value})
                    for group, selected in membership:
                        value = combine([stats[i] for i in selected])
                        record(table + "_groups", {**common, **identity, "group": group, "stats": value})
                        (phase_groups if phase_mode else paired_groups)[seed, scope, group] = value
        common = {"contrast": name, "baseline_family": baseline, "candidate_family": candidate, "seeds": list(SEEDS)}
        for phase_mode in (False, True):
            for scope in (("initial", "postcorrection") if phase_mode else SCOPES):
                identity = {"phase": scope, "normalization_scope": "full"} if phase_mode else {"scope": scope}
                source = phase_groups if phase_mode else paired_groups
                for group, _ in membership:
                    record(("phase" if phase_mode else "paired") + "_seed_means", {**common, **identity,
                        "group": group, "stats": combine([source[seed, scope, group] for seed in SEEDS], seeds=True)})
    for table, count in counters.items():
        require(count == len(observed[table]), "no trailing records: " + table)
    counts = {"policies": 25, "fits": 24, "episodes": 36, "originating_cases": 12, "groups": 21,
              "scopes": 6, "contrast_types": 12, "paired_contrasts": 36,
              "windows": len(windows["lengths"]), "nonquery_rows": len(row), **counters}
    comparison.same(counts, observed["counts"], "counts")
    residuals = {"pair_mse": max((max(v[:, -2], default=0.) for v in cache.values()), default=0.),
                 "strict_margin": max((max(v[:, -1], default=0.) for v in cache.values()), default=0.)}
    comparison.same(residuals, observed["residuals"], "reported identity residuals")
    return {"version": VERSION, "agreement": True, "counts": counts, "identity_residuals": residuals,
            "checks": comparison.checks, "numeric_checks": comparison.numeric_checks,
            "max_absolute_reconciliation_residual": comparison.max_absolute_residual,
            "max_tolerance_fraction": comparison.max_normalized_residual}


def lifecycle(args):
    """Load only the frozen producer's byte/process helpers, never its reducer."""
    require(args.plan.is_absolute() and args.plan.is_relative_to(ROOT)
            and args.plan.is_file() and args.plan.stat().st_size <= 80 * 1024**2
            and not any(p.is_symlink() for p in (args.plan, *args.plan.parents)), "bounded external plan")
    raw = args.plan.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.plan_sha256, "external plan hash")
    plan = json.loads(raw)
    for name in (SELF, RUNNER, "src/openjev/research/suspend_clock.py"):
        path = ROOT / name
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular helper source")
        content = path.read_bytes()
        require({"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} == plan["sources"][name],
                "source authenticated before helper import: " + name)
    spec = importlib.util.spec_from_file_location("_ranking_audit_byte_lifecycle", ROOT / RUNNER)
    require(spec is not None and spec.loader is not None, "byte helper module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diagnostic_closure(helper, job, args):
    """Authenticate the completed diagnostic and its original process first."""
    require(args.worker == Path(job.plan["outputs"]["diagnosis"]) / "receipt.json",
            "predeclared original diagnostic worker")
    expected = {"worker": (args.worker, args.worker_sha256), "terminal": (args.terminal, args.terminal_sha256)}
    descriptors = {}
    for role, (path, digest) in expected.items():
        actual = helper.pin(path)
        require(actual["sha256"] == digest, "external diagnostic " + role + " hash")
        descriptors[role] = {"path": str(path), **actual}
    worker, terminal = helper.read(args.worker), helper.read(args.terminal)
    require(worker["version"] == helper.VERSION and worker["phase"] == "diagnosis"
            and worker["limits"] == helper.LIMITS and worker["requires_successful_original_supervisor"] is True
            and worker["new_scientific_gate"] is False and worker["npz_decodes"] == 26
            and all(worker[k] == 0 for k in ("model_calls", "native_calls", "optimizer_calls")),
            "completed saved-only diagnostic scope")
    require(worker["sources"] == job.plan["sources"] and worker["inputs"] == job.plan["inputs"],
            "same complete frozen sources and inputs")
    options = helper.original_join(worker, terminal, script="diagnose_otto_ranking.py",
        output=args.worker.parent, plan=args.plan, plan_pin=args.plan_sha256, cap=240)
    require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"},
            "exact original diagnostic arguments")
    require(set(worker["files"]) == {"started.json", "diagnosis.json"}
            and {p.name for p in args.worker.parent.iterdir()} == {"started.json", "diagnosis.json", "receipt.json"},
            "complete closed diagnostic payload set")
    for name, pin in worker["files"].items():
        job.check()
        require(helper.pin(args.worker.parent / name) == pin, "closed diagnostic payload: " + name)
    launch = Path(options["--supervision"])
    start = helper.read(args.worker.parent / "started.json")
    require(start == {"phase": "diagnosis", "launch": helper.read(launch), "plan_sha256": args.plan_sha256},
            "original diagnostic startup and launch agree")
    require(worker["peak_rss_bytes"] <= helper.LIMITS["rss_bytes"]
            and sum(v["bytes"] for v in worker["files"].values()) + descriptors["worker"]["bytes"]
            <= helper.LIMITS["output_bytes"], "closed diagnostic resource accounting")
    Comparison().same((worker["finished_ns"] - worker["started_ns"]) / 1e9, worker["wall_seconds"], "worker duration")
    descriptors["launch"] = {"path": str(launch), **helper.pin(launch)}
    result_path = args.worker.parent / "diagnosis.json"
    descriptors["result"] = {"path": str(result_path), **worker["files"]["diagnosis.json"]}
    return worker, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "worker", "terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper = lifecycle(args)
    job = helper.Run(args, kind="audit")
    try:
        worker, descriptors = diagnostic_closure(helper, job, args)
        observed = helper.read(Path(descriptors["result"]["path"]), limit_bytes=helper.LIMITS["output_bytes"])
        require(worker["counts"] == observed["counts"], "acknowledged complete diagnostic counts")
        metadata = helper.read(job.paths["window_metadata"])
        with np.load(job.paths["windows"], allow_pickle=False) as archive:
            windows = {key: archive[key] for key in archive.files}
        job.receipt["npz_decodes"] += 1
        windows.update(episode_ids=tuple(metadata["episode_ids"]), episode_regimes=tuple(metadata["episode_regimes"]))
        identities = [{key: row[key] for key in ("episode_id", "regime", "case", "arm")}
                      for row in helper.read(job.paths["collection_plan"])["cohort"] if row["stage"] == "valid"]

        def load_prediction(family, seed):
            job.check()
            role = "hold" if family == "hold" else f"prediction-{family}-{seed}"
            with np.load(job.paths[role], allow_pickle=False) as archive:
                require(set(archive.files) == ({"predictions"} if family == "hold"
                        else {"predictions", "prior", "prior_mask"}), "original saved prediction container")
                prediction = archive["predictions"]
            job.receipt["npz_decodes"] += 1
            return prediction

        report = verify_reductions(windows, load_prediction, identities, observed, job.reference, job.check)
        require(job.receipt["npz_decodes"] == 26, "exact saved array decode count")
        report.update(producer_inputs=descriptors, new_scientific_gate=False,
            scope="Independent saved f32 arithmetic only; no inference, filtering, training or new efficacy gate.",
            limitations=["Original saved predictions, teacher scores and their model/filter provenance remain inherited.",
                         "SHA-bound original study audit supplies numerical input truth; this audit checks the new reductions.",
                         "Reported identity residuals are descriptive and are never clipped or used as a promotion rule.",
                         "This receipt additionally requires its own successful original supervising parent."])
        for item in descriptors.values():
            job.check()
            require(helper.pin(Path(item["path"])) == {k: item[k] for k in ("sha256", "bytes")},
                    "diagnostic evidence unchanged")
        job.finish({"audit.json": report}, {"agreement": True, "counts": report["counts"],
                   "checks": report["checks"], "producer_inputs": descriptors, "new_scientific_gate": False})
    except BaseException as error:
        job.fail(error)
        raise


if __name__ == "__main__":
    main()
