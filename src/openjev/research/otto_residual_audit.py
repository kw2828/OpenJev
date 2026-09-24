"""Independent in-memory audit of residual-screen predictions and decisions.

No producer metric, gate or replay code is imported. The caller supplies already
authenticated arrays and complete identities; this module does not open files,
admit a phase, call a model, or establish original process closure. Recompute
every view with report_view before evaluating reports or comparing saved JSON.
The optional zero-argument check callback may raise to enforce a caller deadline.

Query rows must reproduce all four observed float32 score bytes, including
signed zeros. They never enter action metrics. Optional prequery diagnostics
exclude the first query and center all four scores; they must come from the
actual model being reported. In particular a pretrained prior is not a Joint
AUX prior. Omitting diagnostics for every method is the common comparison.
"""
from __future__ import annotations

import hashlib
import math
import numbers
from itertools import pairwise

import numpy as np

FIT_SEEDS = (309000001, 309000002, 309000003)
TAUS = (.01, .1, 1., 10.)
BASELINES = ("pretrained", "joint_aux", "last_error", "trace_delta")
RLS = ("rls_full", "rls_diagonal", "rls_shrink_025", "rls_shrink_050", "rls_shrink_075")
REGIMES = ("lambda3", "lambda4")
COLLECTORS = ("analytic", "neural", "period4_hold")
FIELDS = ("stage", "episode_id", "episode_index", "seed", "case", "regime", "arm")
METRICS = ("agreement", "raw_gap", "first_argmin_match", "centered_mse")
METRIC_VERSION = "otto-query-memory-metrics-v1"
SCOPE = "teacher-score imitation on fixed collector paths; no autonomous efficacy"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check(callback):
    _require(callback is None or callable(callback), "optional callable deadline check")
    if callback is not None:
        callback()


def _family(method, tau):
    _require(type(method) is str and method in BASELINES + RLS, "declared method")
    if method in BASELINES:
        _require(tau is None, "ordinary method has no ratio")
        return method
    _require(isinstance(tau, numbers.Real) and not isinstance(tau, (bool, np.bool_)), "explicit real ratio")
    try:
        ratio = float(tau)
    except (ValueError, OverflowError) as error:
        raise ValueError("representable ratio") from error
    _require(ratio in TAUS, "ratio from fixed grid")
    return f"{method}@tau={ratio:g}"


def _identities(values):
    _require(isinstance(values, (list, tuple)) and bool(values), "nonempty identity list")
    result, ids, indices, paths, cases, seeds = [], set(), set(), set(), {}, {}
    for value in values:
        _require(isinstance(value, dict) and set(FIELDS) <= value.keys(), "complete identity")
        row = {key: value[key] for key in FIELDS}
        _require(row["stage"] in ("dev", "test") and row["regime"] in REGIMES and row["arm"] in COLLECTORS,
                 "declared new-cohort split, setting and collector")
        _require(type(row["episode_id"]) is str and bool(row["episode_id"]), "nonempty episode ID")
        _require(all(type(row[k]) is int and row[k] >= 0 for k in ("episode_index", "seed", "case")), "integer identity indices")
        case, path = (row["regime"], row["case"]), (row["regime"], row["case"], row["arm"])
        _require(row["episode_id"] not in ids and row["episode_index"] not in indices and path not in paths,
                 "unique episode, index and case/collector")
        _require(cases.setdefault(case, row["seed"]) == row["seed"] and seeds.setdefault(row["seed"], case) == case,
                 "paired case seeds, distinct between cases")
        ids.add(row["episode_id"])
        indices.add(row["episode_index"])
        paths.add(path)
        result.append(row)
    _require(len({row["stage"] for row in result}) == 1, "single compatibility split")
    return result


def _array(value, dtype, shape, name, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and value.shape == shape,
             name + " exact dtype and shape")
    _require(not finite or bool(np.isfinite(value).all()), name + " finite")


def centered_mse(predicted, target):
    """Scalar legal-action contrast error; prequery callers pass all four."""
    predicted, target = tuple(map(float, predicted)), tuple(map(float, target))
    _require(len(predicted) == len(target) and bool(predicted)
             and all(math.isfinite(v) for v in predicted + target), "finite matched score vectors")
    pmean, tmean = math.fsum(predicted) / len(predicted), math.fsum(target) / len(target)
    value = math.fsum(((p - pmean) - (t - tmean)) ** 2 for p, t in zip(predicted, target, strict=True)) / len(predicted)
    _require(math.isfinite(value), "finite centered score error")
    return value


def scalar_row(raw, legal, prediction):
    """Independent deployed float32 near-minimum choice and scalar metrics."""
    allowed = [i for i in range(4) if legal[i]]
    _require(bool(allowed), "at least one legal action")
    teacher_minimum = min(float(raw[i]) for i in allowed)
    prediction_minimum = min(float(prediction[i]) for i in allowed)
    with np.errstate(over="ignore"):
        predicted = [i for i in allowed if np.float32(float(prediction[i]) - prediction_minimum) < np.float32(1e-10)]
        teacher = [i for i in allowed if np.float32(float(raw[i]) - teacher_minimum) < np.float32(1e-10)]
    chosen = predicted[0]
    return (float(chosen in teacher), float(raw[chosen]) - teacher_minimum, float(chosen == teacher[0]),
            centered_mse([prediction[i] for i in allowed], [raw[i] for i in allowed]))


def _scope_steps(length, stage):
    full = [i for i in range(length) if i % 4]
    result = {"full": full, "initial": [i for i in full if i < 4], "later": [i for i in full if i >= 5]}
    if stage == "test":
        result.update(common_full=full, common_initial=[i for i in full if i < 8],
                      common_later=[i for i in full if i >= 9])
    return result


def _leaf(rows, selected, names=METRICS):
    return {"rows": len(selected), "sums": {name: math.fsum(rows[i][j] for i in selected)
                                             for j, name in enumerate(names)}}


def _summary(records, leaves, names):
    identities = [record["identity"] for record in records]
    cases = sorted({(row["regime"], row["case"]) for row in identities})
    supported = [i for i, leaf in enumerate(leaves) if leaf["rows"]]
    support_cases = {(identities[i]["regime"], identities[i]["case"]) for i in supported}
    result = {"episodes": len(records), "supported_episodes": len(supported),
              "zero_support_episode_ids": [row["episode_id"] for row, leaf in zip(identities, leaves, strict=True) if not leaf["rows"]],
              "rows": sum(leaf["rows"] for leaf in leaves), "weight_mass": len(supported) / len(records),
              "declared_case_count": len(cases), "supported_case_count": len(support_cases),
              "supported_cases": [{"regime": r, "case": c} for r, c in sorted(support_cases)],
              "zero_support_cases": [{"regime": r, "case": c} for r, c in sorted(set(cases) - support_cases)],
              "raw_sums": {name: math.fsum(leaf["sums"][name] for leaf in leaves) for name in names}}
    for name in names:
        episode_means = [leaf["sums"][name] / leaf["rows"] if leaf["rows"] else 0. for leaf in leaves]
        total = math.fsum(episode_means)
        case_means = []
        for case in cases:
            indices = [i for i, row in enumerate(identities) if (row["regime"], row["case"]) == case]
            case_means.append(math.fsum(episode_means[i] for i in indices) / len(indices))
        result.update({"episode_weighted_" + name: total / len(records),
                       "supported_episode_" + name: total / len(supported) if supported else None,
                       "row_weighted_" + name: result["raw_sums"][name] / result["rows"] if result["rows"] else None,
                       "case_weighted_" + name: math.fsum(case_means) / len(cases)})
    return result


def _group(records, scope, names, ages, check):
    def summarize(selected):
        _check(check)
        leaves = [record[scope] for record in selected]
        result = _summary(selected, leaves, names)
        if ages:
            result["by_age"] = {age: _summary(selected, [leaf["by_age"][age] for leaf in leaves], names) for age in ages}
        return result

    return {"overall": summarize(records),
            "by_regime": {regime: summarize([r for r in records if r["identity"]["regime"] == regime])
                          for regime in sorted({r["identity"]["regime"] for r in records})},
            "by_collector": {arm: summarize([r for r in records if r["identity"]["arm"] == arm])
                             for arm in sorted({r["identity"]["arm"] for r in records})},
            "by_case": [{"regime": regime, "case": case, **summarize([
                r for r in records if (r["identity"]["regime"], r["identity"]["case"]) == (regime, case)])}
                for regime, case in sorted({(r["identity"]["regime"], r["identity"]["case"]) for r in records})]}


def report_view(identities, raw_q, legal, action_scores, offsets, method, tau, fit_seed, *, prequery_forecast=None, check=None):
    """Reconstruct the canonical aggregate from complete flat saved arrays.

    Offset blocks follow supplied identity order; results sort by episode index.
    Optional diagnostics are explicit and must belong to this view's model.
    """
    _check(check)
    identities = _identities(identities)
    family = _family(method, tau)
    _require(type(fit_seed) is int and fit_seed in FIT_SEEDS, "fixed fit seed")
    _require(isinstance(raw_q, np.ndarray) and raw_q.ndim == 2, "flat raw scores")
    total = len(raw_q)
    for value, dtype, name in ((raw_q, np.float32, "raw scores"), (legal, np.bool_, "legal actions"),
                               (action_scores, np.float32, "saved action scores")):
        _array(value, dtype, (total, 4), name)
    _array(offsets, np.int64, (len(identities) + 1,), "episode offsets")
    _require(offsets[0] == 0 and offsets[-1] == total and bool((np.diff(offsets) > 0).all())
             and bool((np.diff(offsets) <= 2188).all()), "complete bounded nonempty episodes")
    _require(bool(legal.any(axis=1).all()), "legal support at every row")
    if prequery_forecast is not None:
        _array(prequery_forecast, np.float32, (total, 4), "optional prequery forecast", finite=False)
    records = []
    for identity, (low, high) in zip(identities, pairwise(offsets), strict=True):
        _check(check)
        low, high = int(low), int(high)
        length = high - low
        _require(action_scores[low:high:4].tobytes() == raw_q[low:high:4].tobytes(), "bitwise observed query scores")
        rows, prior_rows = {}, {}
        for step in range(length):
            _check(check)
            if step % 4:
                rows[step] = scalar_row(raw_q[low + step], legal[low + step], action_scores[low + step])
            elif step and prequery_forecast is not None:
                prior_rows[step] = (centered_mse(prequery_forecast[low + step], raw_q[low + step]),)
        digest = hashlib.sha256(b"otto-query-memory-target-v1\0" + length.to_bytes(8, "little"))
        digest.update(raw_q[low:high].astype("<f4", copy=False).tobytes(order="C"))
        digest.update(legal[low:high].tobytes(order="C"))
        record = {"identity": identity, "length": length, "target_sha256": digest.hexdigest()}
        for name, selected in _scope_steps(length, identity["stage"]).items():
            record[name] = _leaf(rows, selected)
            if not name.startswith("common_"):
                record[name]["by_age"] = {str(age): _leaf(rows, [i for i in selected if i % 4 == age]) for age in (1, 2, 3)}
        if prequery_forecast is not None:
            record["prequery"] = _leaf(prior_rows, list(prior_rows), ("centered_mse",))
        records.append(record)
    records.sort(key=lambda record: record["identity"]["episode_index"])
    scopes = {name: _group(records, name, METRICS, () if name.startswith("common_") else ("1", "2", "3"), check)
              for name in _scope_steps(1, identities[0]["stage"])}
    prior = None if prequery_forecast is None else _group(records, "prequery", ("centered_mse",), (), check)
    return {"version": METRIC_VERSION, "scope": SCOPE, "family": family, "seed": fit_seed, "query_period": 4,
            "stage": identities[0]["stage"], "episodes": len(records), "identity_manifest": [
                {**r["identity"], "length": r["length"], "target_sha256": r["target_sha256"]} for r in records],
            "scopes": scopes, "prequery": prior}


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative scalar")
    return value


def _finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif value is not None and not isinstance(value, str):
        _number(value)


def evaluate_reports(reports, expected_identities, *, stage, selected_tau=None, technical_complete=False, check=None):
    """Independently reconstruct selection, 13 usefulness and 10 covariance tests.

    Supply array-reconstructed reports, not unauthenticated producer scalars.
    The technical flag is caller evidence; this pure function cannot certify it.
    """
    _check(check)
    _require(type(stage) is str and stage in ("dev", "confirm") and type(technical_complete) is bool, "explicit phase and technical flag")
    if stage == "dev":
        _require(selected_tau is None, "development selects its own ratio")
        ratios = TAUS
    else:
        _family("rls_full", selected_tau)
        ratios = (float(selected_tau),)
    cases, support_required = (3, 2) if stage == "dev" else (6, 4)
    expected = sorted(_identities(expected_identities), key=lambda row: row["episode_index"])
    metric_stage = "dev" if stage == "dev" else "test"
    _require(len(expected) == cases * 6 and all(row["stage"] == metric_stage for row in expected)
             and {(r["regime"], r["case"], r["arm"]) for r in expected}
             == {(r, c, a) for r in REGIMES for c in range(cases) for a in COLLECTORS}, "complete phase cohort")
    families = BASELINES + tuple(_family(method, tau) for method in RLS for tau in ratios)
    wanted = {(family, seed) for family in families for seed in FIT_SEEDS}
    _require(isinstance(reports, (list, tuple)) and len(reports) == len(wanted), "exact 72 or 27 view roster")
    lookup, reference = {}, None
    for report in reports:
        _check(check)
        _require(isinstance(report, dict) and report.get("version") == METRIC_VERSION
                 and type(report.get("family")) is str and type(report.get("seed")) is int, "canonical view identity")
        key = report["family"], report["seed"]
        _require(key in wanted and key not in lookup, "unique declared method, ratio and seed")
        _require(type(report.get("query_period")) is int and report["query_period"] == 4
                 and report.get("stage") == metric_stage and type(report.get("episodes")) is int
                 and report["episodes"] == len(expected), "complete P4 phase view")
        manifest = report.get("identity_manifest")
        _require(isinstance(manifest, list) and _identities(manifest) == expected, "exact ordered source identities")
        for row in manifest:
            _require(type(row.get("length")) is int and 1 <= row["length"] <= 2188
                     and type(row.get("target_sha256")) is str and len(row["target_sha256"]) == 64
                     and all(c in "0123456789abcdef" for c in row["target_sha256"]), "bounded lengths and target hashes")
        if reference is None:
            reference = manifest
        _require(manifest == reference, "paired target, path and length manifests")
        _require(isinstance(report.get("scopes"), dict) and {"full", "later"} <= report["scopes"].keys(), "primary scopes")
        _finite_tree(report["scopes"])
        _finite_tree(report.get("prequery"))
        lookup[key] = report

    def leaf(family, seed, scope, regime):
        _check(check)
        scopes = lookup[family, seed]["scopes"]
        _require(isinstance(scopes[scope], dict) and isinstance(scopes[scope].get("by_regime"), dict), "setting summaries")
        value = scopes[scope]["by_regime"].get(regime)
        supported = {r["case"] for r in reference if r["regime"] == regime and r["length"] >= (6 if scope == "later" else 2)}
        _require(isinstance(value, dict) and type(value.get("episodes")) is int and value["episodes"] == cases * 3
                 and type(value.get("declared_case_count")) is int and value["declared_case_count"] == cases
                 and type(value.get("supported_case_count")) is int and value["supported_case_count"] == len(supported),
                 "complete case weighting and support")
        gap = _number(value.get("case_weighted_raw_gap"))
        _require(bool(supported) or gap == 0, "zero-support scope retains zero contribution")
        return value

    def gap(family, seed, scope, regime):
        return leaf(family, seed, scope, regime)["case_weighted_raw_gap"]

    def mean(family, scope, regime):
        try:
            return _number(math.fsum(gap(family, seed, scope, regime) for seed in FIT_SEEDS) / 3)
        except OverflowError as error:
            raise ValueError("finite fit-seed aggregate") from error

    for family, seed in wanted:
        for scope in ("full", "later"):
            for regime in REGIMES:
                leaf(family, seed, scope, regime)
    grid = []
    for tau in ratios:
        values = {r: mean(_family("rls_full", tau), "later", r) for r in REGIMES}
        grid.append({"tau": tau, "later_means": values, "objective": max(values.values())})
    chosen = min(grid, key=lambda row: (row["objective"], row["tau"]))["tau"]
    candidate = _family("rls_full", chosen)
    conditions = [{"name": "technical_completion", "passed": technical_complete}]
    for regime in REGIMES:
        prefix = regime + ":P4"
        support = leaf(candidate, FIT_SEEDS[0], "later", regime)["supported_case_count"]
        conditions.append({"name": prefix + ":supported_cases", "passed": support >= support_required,
                           "actual": support, "required": support_required})
        later = mean(candidate, "later", regime)
        controls = {name: mean(name, "later", regime) for name in BASELINES}
        best = min(controls.values())
        conditions.append({"name": prefix + ":later_gap_10pct", "passed": later < best and later <= .9 * best,
                           "candidate": later, "controls": controls, "best_control": best})
        full = mean(candidate, "full", regime)
        controls = {name: mean(name, "full", regime) for name in BASELINES}
        conditions.append({"name": prefix + ":full_gap_nonregression", "passed": full <= min(controls.values()),
                           "candidate": full, "controls": controls})
        for seed in FIT_SEEDS:
            value = gap(candidate, seed, "later", regime)
            controls = {name: gap(name, seed, "later", regime) for name in BASELINES}
            conditions.append({"name": prefix + f":seed_{seed}_nonregression", "passed": value <= min(controls.values()),
                               "candidate": value, "controls": controls})
    passed = all(row["passed"] for row in conditions)
    contrasts = []
    for tau in ratios:
        full = _family("rls_full", tau)
        for other_method in RLS[1:]:
            other = _family(other_method, tau)
            panels, covariance = [], []
            for regime in REGIMES:
                later, comparison = mean(full, "later", regime), mean(other, "later", regime)
                full_gap, comparison_full = mean(full, "full", regime), mean(other, "full", regime)
                paired = [{"seed": seed, "candidate_later": gap(full, seed, "later", regime),
                           "control_later": gap(other, seed, "later", regime), "candidate_full": gap(full, seed, "full", regime),
                           "control_full": gap(other, seed, "full", regime)} for seed in FIT_SEEDS]
                panels.append({"regime": regime, "candidate_later": later, "control_later": comparison,
                               "candidate_full": full_gap, "control_full": comparison_full, "paired": paired})
                if other_method == "rls_diagonal":
                    covariance.extend([{"name": regime + ":later_gap_10pct", "passed": later < comparison and later <= .9 * comparison},
                                       {"name": regime + ":full_gap_nonregression", "passed": full_gap <= comparison_full}])
                    covariance.extend({"name": regime + f":seed_{r['seed']}_nonregression",
                                       "passed": r["candidate_later"] <= r["control_later"]} for r in paired)
            record = {"tau": tau, "candidate": full, "control": other, "panels": panels}
            if other_method == "rls_diagonal":
                _require(len(covariance) == 10, "all covariance contrast conditions")
                contrast_passed = all(row["passed"] for row in covariance)
                record.update(conditions=covariance, contrast_passed=contrast_passed,
                              confirmed_with_usefulness=stage == "confirm" and passed and contrast_passed)
            contrasts.append(record)
    _require(len(conditions) == 13 and len(lookup) == (72 if stage == "dev" else 27), "complete audit decision")
    return {"version": "otto-residual-gate-v1", "stage": stage, "candidate": "rls_full", "selected_tau": chosen,
            "selection": {"rule": "minimize worst-setting candidate mean later gap; ascending-grid exact tie",
                          "grid": grid, "uses_confirmation_for_selection": False},
            "reports": len(reports), "technical_complete": technical_complete, "conditions": conditions,
            "passed_conditions": sum(row["passed"] for row in conditions), "total_conditions": 13, "passed": passed,
            "mechanistic_contrasts": contrasts,
            "scope": "fixed-path usefulness screen; no autonomous, calibration, compute or novelty claim"}
