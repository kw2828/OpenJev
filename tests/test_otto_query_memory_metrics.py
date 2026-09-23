"""Fabricated hand-calculated and independent scalar checks; no models or data IO."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from openjev.research import otto_query_memory_metrics as metrics


def identity(index=0, *, stage="test", regime="lambda3", case=None, arm="analytic"):
    case = index if case is None else case
    return {"stage": stage, "episode_id": f"{stage}:{regime}:{case}:{arm}",
            "episode_index": index, "seed": (100 if regime == "lambda3" else 200) + case,
            "case": case, "regime": regime, "arm": arm}


def arrays(length):
    teacher = np.tile(np.array([1, 0, 2, 3], np.float32), (length, 1))
    legal = np.ones((length, 4), np.bool_)
    predicted = np.zeros((length, 4), np.float32)
    return teacher, legal, predicted


def episode(length, *, period=4, ident=None, prior=False):
    q, legal, p = arrays(length)
    forecast = None
    if prior:
        forecast = np.full_like(q, np.nan)
        forecast[period::period] = q[period::period]
    return metrics.episode_metrics(identity() if ident is None else ident, period, q, legal, p,
                                   prequery_forecast=forecast)


def aggregate(reports, *, family="candidate", seed=101, expected=None):
    if expected is None:
        expected = [r["identity"] for r in reports]
    return metrics.aggregate_episodes(reports, family=family, fit_seed=seed, expected_identities=expected)


def scalar_rows(q, legal, p, period):
    """Explicit deployment rule independent of the production helpers."""
    result = {}
    for step in range(len(q)):
        if step % period == 0:
            continue
        allowed = [a for a in range(4) if bool(legal[step, a])]
        minimum_p = min(float(p[step, a]) for a in allowed)
        minimum_q = min(float(q[step, a]) for a in allowed)
        chosen = next(a for a in allowed if np.float32(p[step, a] - np.float32(minimum_p)) < np.float32(1e-10))
        teacher = [a for a in allowed if np.float32(q[step, a] - np.float32(minimum_q)) < np.float32(1e-10)]
        pm = sum(float(p[step, a]) for a in allowed) / len(allowed)
        qm = sum(float(q[step, a]) for a in allowed) / len(allowed)
        mse = sum(((float(p[step, a]) - pm) - (float(q[step, a]) - qm)) ** 2
                  for a in allowed) / len(allowed)
        result[step] = (float(chosen in teacher), float(q[step, chosen]) - minimum_q,
                        float(chosen == teacher[0]), mse)
    return result


@pytest.mark.parametrize("period", (4, 8))
def test_scalar_oracle_masks_and_all_actual_ages(period):
    length = 19
    q, legal, p = arrays(length)
    for step in range(length):
        q[step] = [step % 3, 2 - step % 5, 1, -1]
        p[step] = [2, step % 2, -2, 3]
        legal[step, step % 4] = False
    original = [value.copy() for value in (q, legal, p)]
    report = metrics.episode_metrics(identity(), period, q, legal, p)
    oracle = scalar_rows(q, legal, p, period)
    selectors = {
        "full": lambda t: t % period != 0,
        "initial": lambda t: 0 < t < period,
        "later": lambda t: t >= period + 1 and t % period != 0,
        "common_full": lambda t: t % 4 != 0,
        "common_initial": lambda t: t < 8 and t % 4 != 0,
        "common_later": lambda t: t >= 9 and t % 4 != 0,
    }
    for name, keep in selectors.items():
        selected = [values for step, values in oracle.items() if keep(step)]
        actual = report["scopes"][name]
        assert actual["rows"] == len(selected)
        for column, metric in enumerate(metrics.METRICS):
            assert actual["sums"][metric] == pytest.approx(sum(row[column] for row in selected))
        if not name.startswith("common_"):
            assert set(actual["by_age"]) == {str(age) for age in range(1, period)}
            for age in range(1, period):
                rows = [values for step, values in oracle.items() if keep(step) and step % period == age]
                assert actual["by_age"][str(age)]["rows"] == len(rows)
                for column, metric in enumerate(metrics.METRICS):
                    assert actual["by_age"][str(age)]["sums"][metric] == pytest.approx(
                        sum(row[column] for row in rows))
    for value, before in zip((q, legal, p), original, strict=True):
        assert np.array_equal(value, before)


def test_hand_calculated_denominators_include_zero_support_paths():
    reports = [episode(1, ident=identity(0)), episode(6, ident=identity(1))]
    output = aggregate(reports)
    full = output["scopes"]["full"]["overall"]
    assert full["episodes"] == 2
    assert full["supported_episodes"] == 1
    assert full["rows"] == 4
    assert full["weight_mass"] == .5
    assert full["episode_weighted_raw_gap"] == .5
    assert full["supported_episode_raw_gap"] == 1
    assert full["row_weighted_raw_gap"] == 1
    assert full["raw_sums"]["raw_gap"] == 4
    assert full["episode_weighted_centered_mse"] == .625
    assert full["declared_case_count"] == 2
    assert full["supported_case_count"] == 1
    assert full["zero_support_cases"] == [{"regime": "lambda3", "case": 0}]
    assert full["zero_support_episode_ids"] == [identity(0)["episode_id"]]
    later = output["scopes"]["later"]["overall"]
    assert later["rows"] == 1
    assert later["episode_weighted_raw_gap"] == .5
    assert later["by_age"]["2"]["episode_weighted_raw_gap"] == 0
    assert later["by_age"]["2"]["supported_episode_raw_gap"] is None


def test_equal_case_and_equal_episode_are_explicit_when_collector_counts_differ():
    reports = [episode(1, ident=identity(0, case=0, arm="analytic")),
               episode(1, ident=identity(1, case=0, arm="neural")),
               episode(2, ident=identity(2, case=1, arm="analytic"))]
    output = aggregate(reports)
    group = output["scopes"]["full"]["overall"]
    assert group["episode_weighted_raw_gap"] == pytest.approx(1 / 3)
    assert group["case_weighted_raw_gap"] == .5
    assert output["scopes"]["full"]["by_collector"]["analytic"]["episodes"] == 2
    assert output["scopes"]["full"]["by_collector"]["neural"]["supported_episodes"] == 0
    assert output["scopes"]["full"]["by_case"][0]["episodes"] == 2


def test_strict_float32_near_minimum_not_exact_argmin():
    q, legal, p = arrays(2)
    legal[1] = [False, True, False, True]
    q[1] = [-100, np.float32(5e-11), -100, 0]
    p[1] = [-100, np.float32(5e-11), -100, 0]
    leaf = metrics.episode_metrics(identity(), 4, q, legal, p)["scopes"]["full"]
    assert leaf["sums"]["first_argmin_match"] == 1
    assert leaf["sums"]["agreement"] == 1
    assert leaf["sums"]["raw_gap"] == float(np.float32(5e-11))
    p[1, 1] = np.float32(1e-10)
    leaf = metrics.episode_metrics(identity(), 4, q, legal, p)["scopes"]["full"]
    assert leaf["sums"]["first_argmin_match"] == 0
    assert leaf["sums"]["agreement"] == 1
    assert leaf["sums"]["raw_gap"] == 0


def test_centering_ignores_illegal_scores_and_global_offsets():
    q, legal, p = arrays(2)
    legal[1] = [True, True, False, False]
    q[1] = [1, 0, -999, 999]
    p[1] = [4, 3, 1000, -1000]
    leaf = metrics.episode_metrics(identity(), 4, q, legal, p)["scopes"]["full"]
    assert leaf["sums"] == {"agreement": 1, "raw_gap": 0, "first_argmin_match": 1, "centered_mse": 0}


def test_extreme_finite_float32_gap_remains_finite_double():
    q, legal, p = arrays(2)
    q[1] = [np.finfo(np.float32).max, -np.finfo(np.float32).max, 0, 0]
    report = metrics.episode_metrics(identity(), 4, q, legal, p)
    value = report["scopes"]["full"]["sums"]["raw_gap"]
    assert math.isfinite(value) and value == 2 * float(np.finfo(np.float32).max)


@pytest.mark.parametrize("period", (4, 8))
def test_all_four_prequery_scores_only_at_actual_later_queries(period):
    q, legal, p = arrays(17)
    legal[:, 2:] = False
    forecast = np.full_like(q, np.nan)
    forecast[period::period] = q[period::period]
    forecast[period, 3] += 4
    report = metrics.episode_metrics(identity(), period, q, legal, p, prequery_forecast=forecast)
    assert report["prequery"]["rows"] == 16 // period
    assert report["prequery"]["sums"]["centered_mse"] == 3
    output = aggregate([report, episode(1, period=period, ident=identity(1), prior=True)])
    assert output["prequery"]["overall"]["episode_weighted_centered_mse"] == 3 / (16 // period) / 2
    assert output["prequery"]["overall"]["zero_support_episode_ids"] == [identity(1)["episode_id"]]
    forecast[period, 0] = np.nan
    with pytest.raises(ValueError, match="finite consumed"):
        metrics.episode_metrics(identity(), period, q, legal, p, prequery_forecast=forecast)


def test_common_support_equal_between_periods_and_not_p8_full():
    reports = [episode(18, period=period) for period in (4, 8)]
    for name in ("common_full", "common_initial", "common_later"):
        assert reports[0]["scopes"][name] == reports[1]["scopes"][name]
    assert reports[0]["scopes"]["common_initial"]["rows"] == 6
    assert reports[0]["scopes"]["common_later"]["rows"] == 7
    assert reports[0]["scopes"]["full"]["rows"] == 13
    assert reports[1]["scopes"]["full"]["rows"] == 15
    assert reports[0]["target_sha256"] == reports[1]["target_sha256"]


@pytest.mark.parametrize("stage", ("train", "dev"))
def test_common_test_scopes_absent_from_other_splits(stage):
    report = episode(10, ident=identity(stage=stage))
    assert set(report["scopes"]) == {"full", "initial", "later"}


def test_target_digest_binds_teacher_and_legal_but_not_model_predictions():
    q, legal, p = arrays(3)
    first = metrics.episode_metrics(identity(), 4, q, legal, p)
    p[1, 0] += 1
    assert metrics.episode_metrics(identity(), 4, q, legal, p)["target_sha256"] == first["target_sha256"]
    q[0, 0] += 1
    assert metrics.episode_metrics(identity(), 4, q, legal, p)["target_sha256"] != first["target_sha256"]
    q[0, 0] -= 1
    legal[0, 3] = False
    assert metrics.episode_metrics(identity(), 4, q, legal, p)["target_sha256"] != first["target_sha256"]


def test_complete_roster_reorders_outputs_without_mutating_inputs():
    reports = [episode(3, ident=identity(0)), episode(5, ident=identity(1, regime="lambda4"))]
    before = copy.deepcopy(reports)
    output = aggregate(list(reversed(reports)), expected=[r["identity"] for r in reports])
    assert [r["episode_index"] for r in output["identity_manifest"]] == [0, 1]
    assert set(output["scopes"]["full"]["by_regime"]) == {"lambda3", "lambda4"}
    assert reports == before


@pytest.mark.parametrize("period", (True, 3, 4.0, 16))
def test_reject_invalid_period(period):
    with pytest.raises(ValueError):
        metrics.episode_metrics(identity(), period, *arrays(2))


@pytest.mark.parametrize("kind", ("empty", "long", "q_dtype", "p_dtype", "legal_dtype", "no_legal", "q_nan", "p_nan"))
def test_reject_invalid_arrays(kind):
    q, legal, p = arrays(0 if kind == "empty" else 2189 if kind == "long" else 2)
    if kind == "q_dtype":
        q = q.astype(np.float64)
    if kind == "p_dtype":
        p = p.astype(np.float64)
    if kind == "legal_dtype":
        legal = legal.astype(np.int64)
    if kind == "no_legal":
        legal[0] = False
    if kind == "q_nan":
        q[0, 0] = np.nan
    if kind == "p_nan":
        p[0, 0] = np.nan
    with pytest.raises(ValueError):
        metrics.episode_metrics(identity(), 4, q, legal, p)


@pytest.mark.parametrize("kind", ("missing", "duplicate", "split", "seed", "same_path", "period", "prior"))
def test_reject_incomplete_or_mismatched_rosters(kind):
    expected = [identity(0), identity(1)]
    reports = [episode(3, ident=row) for row in expected]
    if kind == "missing":
        reports.pop()
    if kind == "duplicate":
        reports[1] = reports[0]
    if kind == "split":
        expected[1]["stage"] = "dev"
    if kind == "seed":
        reports[1]["identity"]["seed"] += 1
    if kind == "same_path":
        expected[1]["case"] = expected[0]["case"]
    if kind == "period":
        reports[1] = episode(3, period=8, ident=expected[1])
    if kind == "prior":
        reports[1] = episode(3, ident=expected[1], prior=True)
    with pytest.raises(ValueError):
        aggregate(reports, expected=expected)


@pytest.mark.parametrize("kind", ("support", "nan", "negative", "fractional_count", "zero_support_sum", "missing_age"))
def test_reject_corrupted_episode_metrics(kind):
    report = episode(2)
    leaf = report["scopes"]["full"]
    if kind == "support":
        leaf["rows"] += 1
    if kind == "nan":
        leaf["sums"]["raw_gap"] = math.nan
    if kind == "negative":
        leaf["sums"]["raw_gap"] = -1
    if kind == "fractional_count":
        leaf["sums"]["agreement"] = .5
    if kind == "zero_support_sum":
        report["scopes"]["later"]["sums"]["raw_gap"] = 1
    if kind == "missing_age":
        del leaf["by_age"]["3"]
    with pytest.raises(ValueError):
        aggregate([report])


def matched():
    return [aggregate([episode(9, period=period)], family=family, seed=seed)
            for family in ("candidate", "control") for seed in (101, 102) for period in (4, 8)]


def validate(reports):
    return metrics.validate_matched_reports(reports, families=("candidate", "control"),
        fit_seeds=(101, 102), query_periods=(4, 8), expected_identities=[identity()])


def test_matched_set_is_full_cartesian_pairing_not_a_performance_gate():
    output = validate(matched())
    assert output == {"matched": True, "reports": 8, "episodes_per_report": 1,
                      "stage": "test", "query_periods": [4, 8]}
    assert "passes" not in output


@pytest.mark.parametrize("kind", ("missing", "duplicate", "target", "length", "split", "identity", "seed_bool"))
def test_reject_unmatched_comparisons(kind):
    reports = matched()
    if kind == "missing":
        reports.pop()
    if kind == "duplicate":
        reports[1] = reports[0]
    if kind == "target":
        reports[-1]["identity_manifest"][0]["target_sha256"] = "0" * 64
    if kind == "length":
        reports[-1]["identity_manifest"][0]["length"] = 8
    if kind == "split":
        reports[-1]["stage"] = "dev"
    if kind == "identity":
        reports[-1]["identity_manifest"][0]["seed"] += 1
    if kind == "seed_bool":
        reports[-1]["seed"] = True
    with pytest.raises(ValueError):
        validate(reports)
