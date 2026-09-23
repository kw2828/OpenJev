"""Pure complete-path teacher-imitation summaries for P4/P8 observations.

The caller authenticates collection, causal inputs, model state and split access.
This module cannot certify those facts and performs no file/model IO or gates.
All declared episodes, including zero-support paths, retain denominator weight.
Raw teacher costs are minimized using the deployed float32 near-minimum rule.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

VERSION = "otto-query-memory-metrics-v1"
PERIODS = (4, 8)
HORIZON = 2188
EPS = np.float32(1e-10)
IDENTITY_FIELDS = ("stage", "episode_id", "episode_index", "seed", "case", "regime", "arm")
METRICS = ("agreement", "raw_gap", "first_argmin_match", "centered_mse")
SCOPE_DESCRIPTION = "teacher-score imitation on fixed collector paths; no autonomous efficacy"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    require(isinstance(value, dict) and set(IDENTITY_FIELDS) <= value.keys(), "complete episode identity")
    require(value["stage"] in ("train", "dev", "test")
            and value["regime"] in ("lambda3", "lambda4")
            and value["arm"] in ("analytic", "neural", "period4_hold"), "split, regime and collector")
    require(type(value["episode_id"]) is str and bool(value["episode_id"]), "nonempty episode ID")
    require(all(type(value[k]) is int and value[k] >= 0 for k in ("episode_index", "seed", "case")),
            "nonnegative integer identity indices")
    return {key: value[key] for key in IDENTITY_FIELDS}


def _identities(values):
    require(isinstance(values, (list, tuple)) and bool(values), "nonempty expected episode identities")
    result = [_identity(value) for value in values]
    require(len({r["stage"] for r in result}) == 1, "one declared split")
    for key in ("episode_id", "episode_index"):
        require(len({r[key] for r in result}) == len(result), "unique " + key)
    require(len({(r["regime"], r["case"], r["arm"]) for r in result}) == len(result),
            "unique originating case/collector path")
    case_seeds, seed_cases = {}, {}
    for row in result:
        case = (row["regime"], row["case"])
        require(case_seeds.setdefault(case, row["seed"]) == row["seed"], "shared originating case seed")
        require(seed_cases.setdefault(row["seed"], case) == case, "distinct seed for each originating case")
    return sorted(result, key=lambda row: row["episode_index"])


def _array(value, dtype, shape, name, *, finite=True):
    require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape,
            name + " dtype and shape")
    require(not finite or bool(np.isfinite(value).all()), name + " finite")
    return value


def _near_minimum(scores, legal):
    allowed = tuple(int(action) for action in np.flatnonzero(legal))
    minimum = np.min(scores[list(allowed)])
    with np.errstate(over="ignore"):
        near = tuple(action for action in allowed if np.float32(scores[action] - minimum) < EPS)
    return near, minimum


def _centered_mse(predicted, target):
    p, t = tuple(map(float, predicted)), tuple(map(float, target))
    pm, tm = math.fsum(p) / len(p), math.fsum(t) / len(t)
    return math.fsum(((a - pm) - (b - tm)) ** 2 for a, b in zip(p, t, strict=True)) / len(p)


def _masks(length, period, stage):
    steps = np.arange(length)
    nonquery = steps % period != 0
    masks = {"full": nonquery, "initial": nonquery & (steps < period),
             "later": nonquery & (steps >= period + 1)}
    if stage == "test":
        common = steps % 4 != 0
        masks.update(common_full=common, common_initial=common & (steps < 8),
                     common_later=common & (steps >= 9))
    return masks


def _leaf(rows, mask, names=METRICS):
    selected = [rows[step] for step in np.flatnonzero(mask)]
    return {"rows": len(selected), "sums": {
        name: math.fsum(row[index] for row in selected) for index, name in enumerate(names)}}


def episode_metrics(identity, query_period, raw_q, legal, action_scores, *, prequery_forecast=None):
    """Summarize one complete episode, with steps starting at zero.

    Arrays are float32 [T,4] and Boolean [T,4], 1 <= T <= 2188. Query
    action-score rows are validated but never scored. Optional prequery forecasts
    score only actual queries after the first, center all four actions, and may
    contain nonfinite poison elsewhere. Values and gaps remain in raw score
    units. ``first_argmin_match`` means the first *near* minimum, not exact argmin.
    Test-only common scopes exclude every P4 query, use initial step < 8 and
    later step >= 9, so their scored rows agree between P4 and P8.
    """
    identity = _identity(identity)
    require(type(query_period) is int and query_period in PERIODS, "supported integer query period")
    require(isinstance(raw_q, np.ndarray) and raw_q.ndim == 2 and 1 <= len(raw_q) <= HORIZON,
            "complete episode length")
    length = len(raw_q)
    _array(raw_q, np.float32, (length, 4), "raw teacher scores")
    _array(legal, np.bool_, (length, 4), "legal mask")
    _array(action_scores, np.float32, (length, 4), "action scores")
    require(bool(legal.any(-1).all()), "at least one legal action per row")
    rows = {}
    for step in range(length):
        if step % query_period == 0:
            continue
        predicted, _ = _near_minimum(action_scores[step], legal[step])
        teacher, minimum = _near_minimum(raw_q[step], legal[step])
        chosen = predicted[0]
        rows[step] = (float(chosen in teacher), float(raw_q[step, chosen]) - float(minimum),
                      float(chosen == teacher[0]),
                      _centered_mse(action_scores[step, legal[step]], raw_q[step, legal[step]]))
    steps = np.arange(length)
    scopes = {}
    for name, mask in _masks(length, query_period, identity["stage"]).items():
        scopes[name] = _leaf(rows, mask)
        if not name.startswith("common_"):
            scopes[name]["by_age"] = {
                str(age): _leaf(rows, mask & (steps % query_period == age))
                for age in range(1, query_period)}
    prior = None
    if prequery_forecast is not None:
        _array(prequery_forecast, np.float32, (length, 4), "prequery forecast", finite=False)
        prior_mask = (steps > 0) & (steps % query_period == 0)
        require(bool(np.isfinite(prequery_forecast[prior_mask]).all()), "finite consumed prequery forecasts")
        prior_rows = {int(step): (_centered_mse(prequery_forecast[step], raw_q[step]),)
                      for step in np.flatnonzero(prior_mask)}
        prior = _leaf(prior_rows, prior_mask, ("centered_mse",))
    target_hash = hashlib.sha256()
    target_hash.update(b"otto-query-memory-target-v1\0")
    target_hash.update(length.to_bytes(8, "little"))
    target_hash.update(raw_q.astype("<f4", copy=False).tobytes(order="C"))
    target_hash.update(legal.tobytes(order="C"))
    return {"version": VERSION, "identity": identity, "query_period": query_period,
            "length": length, "target_sha256": target_hash.hexdigest(), "scopes": scopes, "prequery": prior}


def _validate_leaf(value, count, names=METRICS):
    require(isinstance(value, dict) and type(value.get("rows")) is int and value["rows"] == count,
            "exact scope row support")
    sums = value.get("sums")
    require(isinstance(sums, dict) and set(sums) == set(names), "exact scope metric sums")
    for name, number in sums.items():
        require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                "finite nonnegative metric sum")
        require(count != 0 or number == 0, "zero support has zero sum")
        if name in ("agreement", "first_argmin_match"):
            require(number <= count and number == int(number), "bounded integer agreement count")
    if "first_argmin_match" in names:
        require(sums["first_argmin_match"] <= sums["agreement"], "first near-minimum implies agreement")


def _validate_episode(report):
    require(isinstance(report, dict) and report.get("version") == VERSION, "episode metric version")
    identity = _identity(report.get("identity"))
    period, length = report.get("query_period"), report.get("length")
    require(type(period) is int and period in PERIODS and type(length) is int and 1 <= length <= HORIZON,
            "episode period and length")
    digest = report.get("target_sha256")
    require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "target and legal-mask digest")
    masks = _masks(length, period, identity["stage"])
    require(isinstance(report.get("scopes"), dict) and set(report["scopes"]) == set(masks),
            "exact split-aware metric scopes")
    steps = np.arange(length)
    for name, mask in masks.items():
        value = report["scopes"][name]
        _validate_leaf(value, int(mask.sum()))
        if not name.startswith("common_"):
            require(isinstance(value.get("by_age"), dict)
                    and set(value["by_age"]) == {str(age) for age in range(1, period)}, "all actual query ages")
            for age in range(1, period):
                _validate_leaf(value["by_age"][str(age)], int((mask & (steps % period == age)).sum()))
            for metric in METRICS:
                total = math.fsum(leaf["sums"][metric] for leaf in value["by_age"].values())
                require(math.isclose(total, value["sums"][metric], rel_tol=1e-14, abs_tol=0),
                        "age sums partition scope sums")
    prior = report.get("prequery")
    if prior is not None:
        _validate_leaf(prior, (length - 1) // period, ("centered_mse",))
    return identity


def _summarize(reports, leaves, names=METRICS):
    identities = [r["identity"] for r in reports]
    cases = sorted({(r["regime"], r["case"]) for r in identities})
    supported = [i for i, leaf in enumerate(leaves) if leaf["rows"]]
    supported_cases = {(identities[i]["regime"], identities[i]["case"]) for i in supported}
    def case_record(values):
        return [{"regime": regime, "case": case} for regime, case in values]

    result = {"episodes": len(reports), "supported_episodes": len(supported),
              "zero_support_episode_ids": [r["episode_id"] for r, leaf in zip(identities, leaves, strict=True)
                                           if not leaf["rows"]],
              "rows": sum(leaf["rows"] for leaf in leaves), "weight_mass": len(supported) / len(reports),
              "declared_case_count": len(cases), "supported_case_count": len(supported_cases),
              "supported_cases": case_record(sorted(supported_cases)),
              "zero_support_cases": case_record(sorted(set(cases) - supported_cases)),
              "raw_sums": {name: math.fsum(leaf["sums"][name] for leaf in leaves) for name in names}}
    for name in names:
        means = [leaf["sums"][name] / leaf["rows"] if leaf["rows"] else 0.0 for leaf in leaves]
        total = math.fsum(means)
        result["episode_weighted_" + name] = total / len(reports)
        result["supported_episode_" + name] = total / len(supported) if supported else None
        result["row_weighted_" + name] = result["raw_sums"][name] / result["rows"] if result["rows"] else None
        case_means = []
        for case in cases:
            indices = [i for i, r in enumerate(identities) if (r["regime"], r["case"]) == case]
            case_means.append(math.fsum(means[i] for i in indices) / len(indices))
        result["case_weighted_" + name] = math.fsum(case_means) / len(cases)
    return result


def _groups(reports, get_leaf, *, names=METRICS, ages=()):
    def summarize(selected):
        leaves = [get_leaf(r) for r in selected]
        result = _summarize(selected, leaves, names)
        if ages:
            result["by_age"] = {age: _summarize(selected, [leaf["by_age"][age] for leaf in leaves], names)
                                for age in ages}
        return result

    identities = [r["identity"] for r in reports]
    return {"overall": summarize(reports),
            "by_regime": {value: summarize([r for r in reports if r["identity"]["regime"] == value])
                          for value in sorted({r["regime"] for r in identities})},
            "by_collector": {value: summarize([r for r in reports if r["identity"]["arm"] == value])
                             for value in sorted({r["arm"] for r in identities})},
            "by_case": [{"regime": regime, "case": case, **summarize([
                r for r in reports if (r["identity"]["regime"], r["identity"]["case"]) == (regime, case)])}
                for regime, case in sorted({(r["regime"], r["case"]) for r in identities})]}


def aggregate_episodes(episode_reports, *, family, fit_seed, expected_identities):
    """Aggregate one family/fit seed/period against an explicit complete roster.

    Reports can arrive in any order. Identities, target digests and lengths are
    retained in the manifest for matching comparisons. Expected identities are
    caller-owned provenance, not a reconstructed proof of census completeness.
    Equal-case metrics first average complete collector episodes within a case;
    equal-episode and pooled-row metrics are separately named, never substituted.
    """
    expected = _identities(expected_identities)
    require(type(family) is str and bool(family) and type(fit_seed) is int and fit_seed >= 0,
            "family and integer fit seed")
    require(isinstance(episode_reports, (list, tuple)) and len(episode_reports) == len(expected),
            "one report per declared episode")
    by_id = {}
    for report in episode_reports:
        identity = _validate_episode(report)
        require(identity["episode_id"] not in by_id, "unique reported episode")
        by_id[identity["episode_id"]] = report
    require(set(by_id) == {r["episode_id"] for r in expected}, "complete declared episode membership")
    reports = [by_id[r["episode_id"]] for r in expected]
    require(all(report["identity"] == identity for report, identity in zip(reports, expected, strict=True)),
            "exact originating identity alignment")
    periods = {r["query_period"] for r in reports}
    require(len(periods) == 1, "one query period per aggregate")
    require(len({r["prequery"] is None for r in reports}) == 1, "consistent optional prequery reporting")
    period = reports[0]["query_period"]
    scopes = {}
    for name in reports[0]["scopes"]:
        ages = tuple(str(age) for age in range(1, period)) if not name.startswith("common_") else ()
        scopes[name] = _groups(reports, lambda r, scope=name: r["scopes"][scope], ages=ages)
    prior = None if reports[0]["prequery"] is None else _groups(
        reports, lambda r: r["prequery"], names=("centered_mse",))
    return {"version": VERSION, "scope": SCOPE_DESCRIPTION, "family": family, "seed": fit_seed,
            "query_period": period, "stage": expected[0]["stage"], "episodes": len(reports),
            "identity_manifest": [{**r["identity"], "length": r["length"], "target_sha256": r["target_sha256"]}
                                  for r in reports], "scopes": scopes, "prequery": prior}


def validate_matched_reports(reports, *, families, fit_seeds, query_periods, expected_identities):
    """Check Cartesian membership and identical input manifests; return no gate.

    This checks pairing of reports produced by ``aggregate_episodes``. It does
    not authenticate their numerical summaries or replace an independent audit.
    Both periods may be compared only on the same declared split and targets.
    """
    expected = _identities(expected_identities)
    for values, name in ((families, "families"), (fit_seeds, "fit seeds"), (query_periods, "query periods")):
        require(isinstance(values, (list, tuple)) and bool(values) and len(set(values)) == len(values),
                "nonempty distinct " + name)
    require(all(type(v) is str and bool(v) for v in families), "family names")
    require(all(type(v) is int and v >= 0 for v in fit_seeds), "integer fit seeds")
    require(all(type(v) is int and v in PERIODS for v in query_periods), "supported query periods")
    wanted = {(family, seed, period) for family in families for seed in fit_seeds for period in query_periods}
    require(isinstance(reports, (list, tuple)) and len(reports) == len(wanted), "complete matched set")
    seen, reference = set(), None
    for report in reports:
        require(isinstance(report, dict) and report.get("version") == VERSION, "aggregate metric version")
        require(type(report.get("family")) is str and type(report.get("seed")) is int
                and type(report.get("query_period")) is int, "strict family/seed/period types")
        key = (report.get("family"), report.get("seed"), report.get("query_period"))
        require(key in wanted and key not in seen, "unique declared family/seed/period")
        seen.add(key)
        require(report.get("stage") == expected[0]["stage"] and report.get("episodes") == len(expected),
                "same declared split and complete episode count")
        manifest = report.get("identity_manifest")
        require(isinstance(manifest, list) and [_identity(row) for row in manifest] == expected,
                "same complete identity manifest")
        require(all(type(row.get("length")) is int and 1 <= row["length"] <= HORIZON
                    and type(row.get("target_sha256")) is str and len(row["target_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in row["target_sha256"]) for row in manifest),
                "manifest lengths and target digests")
        if reference is None:
            reference = manifest
        require(manifest == reference, "identical complete paths, raw targets and legal masks")
    require(seen == wanted, "full matched Cartesian membership")
    return {"matched": True, "reports": len(reports), "episodes_per_report": len(expected),
            "stage": expected[0]["stage"], "query_periods": list(query_periods)}
