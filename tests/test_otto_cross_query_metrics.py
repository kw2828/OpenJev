"""Fabricated complete paths and prospective gate boundaries; no learned calls."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from openjev.research import otto_cross_query_metrics as M
from openjev.research import otto_score_forecast_data as base


def fixture(specifications):
    episodes, identities = [], []
    for regime, case, arm, length in specifications:
        episode_id = f"{regime}:{case}:{arm}"
        features = np.zeros((length, 31), dtype=np.float32)
        steps = np.arange(length)
        features[:, 15] = (steps / 2188).astype(np.float32)
        features[:, 16] = ((steps % 4) / 2188).astype(np.float32)
        features[:, 17] = 1
        episodes.append({"id": episode_id, "regime": regime, "features": features,
            "teacher_scores": np.tile(np.asarray([0, 2, 3, 4], np.float32), (length, 1)),
            "legal": np.ones((length, 4), dtype=np.bool_)})
        identities.append({"episode_id": episode_id, "regime": regime, "case": case, "arm": arm})
    windows = base.build_windows(episodes)
    return windows, windows["targets"].copy(), identities


def test_postcorrection_begins_at_five_and_rebalances_each_age():
    windows, predictions, identities = fixture([
        ("lambda3", 0, "analytic", 8), ("lambda3", 1, "analytic", 6),
        ("lambda3", 2, "analytic", 5)])
    # The first three skips are all wrong; they belong only to the full metric.
    predictions[:, 1:] = [5, 0, 6, 7]
    for index, start in enumerate(windows["step_offsets"]):
        if start >= 4:
            predictions[index] = windows["targets"][index]
    result = M.forecast_metrics(windows, predictions, identities)
    full, post = result["full"]["overall"], result["postcorrection"]["overall"]
    assert full["nonquery_rows"] == 13 and post["nonquery_rows"] == 4
    assert full["episode_weighted_agreement"] == pytest.approx((.5 + .25) / 3)
    assert post["episode_weighted_agreement"] == pytest.approx(2 / 3)
    assert post["episode_weighted_raw_gap"] == 0 and post["supported_episode_agreement"] == 1
    assert post["weight_mass"] == pytest.approx(2 / 3)
    assert post["zero_support_episode_ids"] == [identities[2]["episode_id"]]
    assert [post["by_age"][str(age)]["nonquery_rows"] for age in (1, 2, 3)] == [2, 1, 1]
    assert post["by_age"]["1"]["episode_weighted_agreement"] == pytest.approx(2 / 3)
    assert post["by_age"]["2"]["episode_weighted_agreement"] == pytest.approx(1 / 3)
    assert post["by_age"]["3"]["supported_episode_agreement"] == 1


def test_episode_denominator_is_not_row_denominator_or_supported_only():
    windows, predictions, identities = fixture([
        ("lambda3", 0, "analytic", 6), ("lambda3", 1, "analytic", 12),
        ("lambda3", 2, "analytic", 1)])
    predictions[1, 1] = [5, 0, 6, 7]  # One bad primary row in the short path.
    primary = M.forecast_metrics(windows, predictions, identities)["postcorrection"]["overall"]
    assert primary["nonquery_rows"] == 7 and primary["episodes"] == 3
    assert primary["episode_weighted_agreement"] == pytest.approx(1 / 3)
    assert primary["episode_weighted_raw_gap"] == pytest.approx(2 / 3)
    assert primary["supported_episode_agreement"] == .5
    assert primary["supported_episode_raw_gap"] == 1


def test_three_collector_paths_count_as_one_case_and_regimes_are_separate():
    windows, predictions, identities = fixture([
        ("lambda3", 0, arm, length) for arm, length in zip(M.ARMS, (8, 7, 5), strict=True)
    ] + [("lambda3", 1, "analytic", 1), ("lambda4", 0, "analytic", 8)])
    result = M.forecast_metrics(windows, predictions, identities)["postcorrection"]
    assert result["overall"]["declared_case_count"] == 3
    assert result["overall"]["supported_case_count"] == 2
    group = result["by_regime"]["lambda3"]
    assert group["supported_episodes"] == 2 and group["supported_case_count"] == 1
    assert group["by_age"]["3"]["supported_episodes"] == 1
    assert group["by_age"]["3"]["supported_case_count"] == 1
    assert group["zero_support_cases"] == [{"regime": "lambda3", "case": 1}]
    by_case = result["by_case"]
    assert [(r["regime"], r["case"]) for r in by_case] == [("lambda3", 0), ("lambda3", 1), ("lambda4", 0)]
    assert by_case[0]["episodes"] == 3 and by_case[0]["supported_episodes"] == 2
    assert by_case[0]["episode_weighted_agreement"] == pytest.approx(2 / 3)
    assert by_case[0]["by_age"]["3"]["episode_weighted_agreement"] == pytest.approx(1 / 3)
    assert by_case[1]["episode_weighted_agreement"] == 0 and by_case[1]["supported_episode_agreement"] is None


def test_zero_support_retained_with_none_supported_metrics_and_no_fabricated_agreement():
    windows, predictions, identities = fixture([
        ("lambda3", 0, "analytic", 1), ("lambda4", 0, "neural", 5)])
    result = M.forecast_metrics(windows, predictions, identities)
    for group in (result["postcorrection"]["overall"], *result["postcorrection"]["by_regime"].values()):
        for row in (group, *group["by_age"].values()):
            assert row["supported_case_count"] == row["nonquery_rows"] == 0
            assert row["episode_weighted_agreement"] == row["episode_weighted_raw_gap"] == 0
            assert row["supported_episode_agreement"] is row["supported_episode_raw_gap"] is None
    json.dumps(result, allow_nan=False)


def test_strict_float32_near_ties_legal_mask_and_original_unit_gap():
    windows, predictions, identities = fixture([("lambda3", 0, "analytic", 8)])
    windows = dict(windows)
    windows["targets"], windows["legal"] = windows["targets"].copy(), windows["legal"].copy()
    epsilon = np.float32(1e-10)
    below = np.nextafter(epsilon, np.float32(0))
    windows["targets"][1, 1:] = [[0, below, 3, -99], [0, epsilon, 3, -99], [0, 0, 3, -99]]
    windows["legal"][:, :, 3] = False
    predictions[1, 1:] = [2, 0, 3, -100]
    primary = M.forecast_metrics(windows, predictions, identities)["postcorrection"]["overall"]
    assert primary["episode_weighted_agreement"] == pytest.approx(2 / 3)
    assert primary["episode_weighted_first_argmin_match"] == 0
    assert primary["episode_weighted_raw_gap"] == pytest.approx((float(below) + float(epsilon)) / 3)
    predictions[1, 1] = [below, 0, 3, -100]
    predicted_tie = M.forecast_metrics(windows, predictions, identities)["postcorrection"]["overall"]
    assert predicted_tie["by_age"]["1"]["episode_weighted_first_argmin_match"] == 1
    assert predicted_tie["by_age"]["1"]["episode_weighted_raw_gap"] == 0


def test_centered_mse_uses_current_legal_actions_and_full_metrics_match_qualified_helper():
    windows, predictions, identities = fixture([
        ("lambda3", 0, "analytic", 7), ("lambda4", 0, "neural", 1)])
    windows = dict(windows)
    windows["legal"] = windows["legal"].copy()
    windows["legal"][:, :, 2:] = False
    predictions += np.float32(100)
    predictions[1, 2, :2] = [102, 106]  # Centered [-2,2] versus [-1,1]: MSE1.
    before = {key: value.tobytes() for key, value in windows.items() if isinstance(value, np.ndarray)}
    prediction_bytes, identity_copy = predictions.tobytes(), copy.deepcopy(identities)
    result = M.forecast_metrics(windows, predictions, identities)
    assert result["postcorrection"]["by_regime"]["lambda3"]["episode_weighted_centered_mse"] == .5
    qualified = base.forecast_metrics(windows, predictions)

    def compare(actual, expected):
        for key, value in expected.items():
            if isinstance(value, dict):
                compare(actual[key], value)
            else:
                assert actual[key] == value

    compare(result["full"]["overall"], qualified["overall"])
    compare(result["full"]["by_regime"], qualified["by_regime"])
    assert all(windows[key].tobytes() == value for key, value in before.items())
    assert predictions.tobytes() == prediction_bytes and identities == identity_copy


@pytest.mark.parametrize("defect", ["nan", "inf_padding", "float64", "dropped_tail", "bool_case", "wrong_id", "wrong_regime"])
def test_invalid_input_or_identity_rejected(defect):
    windows, predictions, identities = fixture([("lambda3", 0, "analytic", 6)])
    if defect == "nan":
        predictions[1, 1, 0] = np.nan
    elif defect == "inf_padding":
        predictions[1, 3, 0] = np.inf
    elif defect == "float64":
        predictions = predictions.astype(np.float64)
    elif defect == "dropped_tail":
        windows = {**windows, "step_offsets": np.asarray([0, 8], np.int64)}
    elif defect == "bool_case":
        identities[0]["case"] = False
    elif defect == "wrong_id":
        identities[0]["episode_id"] = "other"
    else:
        identities[0]["regime"] = "lambda4"
    with pytest.raises(ValueError):
        M.forecast_metrics(windows, predictions, identities)


def gate_fixture():
    windows, predictions, identities = fixture([
        (regime, case, arm, 8) for regime in M.REGIMES for case in range(4) for arm in M.ARMS])
    template = M.forecast_metrics(windows, predictions, identities)
    hold, models = copy.deepcopy(template), []
    for regime in M.REGIMES:
        hold["postcorrection"]["by_regime"][regime]["episode_weighted_agreement"] = .5
        hold["postcorrection"]["by_regime"][regime]["episode_weighted_raw_gap"] = 11.25
        for age in M.AGES:
            hold["postcorrection"]["by_regime"][regime]["by_age"][age]["episode_weighted_raw_gap"] = 1.
    for kind in M.FAMILIES:
        for seed in M.SEEDS:
            report = copy.deepcopy(template)
            for regime in M.REGIMES:
                for scope in ("full", "postcorrection"):
                    group = report[scope]["by_regime"][regime]
                    group["episode_weighted_agreement"] = .5
                    group["episode_weighted_raw_gap"] = 1. if scope == "full" else 9. if kind == "innovation" else 10.
                    for age in M.AGES:
                        group["by_age"][age]["episode_weighted_raw_gap"] = 1.
            models.append({"family": kind, "seed": seed, "metrics": report})
    return models, hold


def test_all_53_boundaries_pass_with_no_seed_or_controller_selection():
    models, hold = gate_fixture()
    rows = M.criteria(models, hold, technical_complete=True)
    assert len(rows) == len({r["name"] for r in rows}) == 53
    assert all(r["passes"] for r in rows)
    assert all(r["value"] == r["threshold"] for r in rows)
    assert sum("case_support" in r["name"] for r in rows) == 6
    assert sum(".age" in r["name"] and "gap_vs_hold" in r["name"] for r in rows) == 18
    assert sum(".full." in r["name"] for r in rows) == 4
    json.dumps(rows, allow_nan=False)


def boundary_cases():
    result = [("technical", None, None, None)]
    for regime in M.REGIMES:
        result += [("support", regime, age, None) for age in M.AGES]
        for seed in M.SEEDS:
            result += [("hold", regime, seed, metric) for metric in ("agreement", "gap")]
            result += [("age", regime, seed, age) for age in M.AGES]
        for control in M.FAMILIES[1:]:
            result += [("relative", regime, control, metric) for metric in ("agreement", "gap")]
        result += [("full", regime, None, metric) for metric in ("agreement", "gap")]
    return result


@pytest.mark.parametrize("category,regime,item,metric", boundary_cases())
def test_every_condition_can_fail_without_being_replaced_by_another_success(category, regime, item, metric):
    models, hold = gate_fixture()
    by = {(r["family"], r["seed"]): r["metrics"] for r in models}
    complete = category != "technical"
    key = "episode_weighted_agreement" if metric == "agreement" else "episode_weighted_raw_gap"
    if category == "technical":
        name = "technical_complete"
    elif category == "support":
        name = f"{regime}.age{item}.case_support"
        for report in (hold, *by.values()):
            group = report["postcorrection"]["by_regime"][regime]["by_age"][item]
            group["supported_case_count"] = 3
            group["zero_support_cases"] = [group["supported_cases"].pop()]
    elif category == "hold":
        name = f"{regime}.{item}.{metric}_vs_hold"
        by["innovation", item]["postcorrection"]["by_regime"][regime][key] = .4 if metric == "agreement" else 9.1
    elif category == "age":
        name = f"{regime}.{item}.age{metric}.gap_vs_hold"
        by["innovation", item]["postcorrection"]["by_regime"][regime]["by_age"][metric]["episode_weighted_raw_gap"] = 1.1
    elif category == "relative":
        name = f"{regime}.mean_{metric}_vs_{item}"
        for seed in M.SEEDS:
            by[item, seed]["postcorrection"]["by_regime"][regime][key] = .6 if metric == "agreement" else 9.9
    else:
        name = f"{regime}.full.mean_{metric}_vs_reset_direct"
        for seed in M.SEEDS:
            by["reset_direct", seed]["full"]["by_regime"][regime][key] = .6 if metric == "agreement" else .9
    rows = M.criteria(models, hold, technical_complete=complete)
    assert len(rows) == 53 and not next(row for row in rows if row["name"] == name)["passes"]
    assert not all(row["passes"] for row in rows)


@pytest.mark.parametrize("defect", ["nan_metric", "different_support", "missing_fit", "duplicate_fit", "bool_seed", "nonbool_closure"])
def test_gate_rejects_invalid_summaries_or_incomplete_membership(defect):
    models, hold = gate_fixture()
    complete = True
    if defect == "nan_metric":
        models[0]["metrics"]["postcorrection"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = float("nan")
    elif defect == "different_support":
        models[0]["metrics"]["postcorrection"]["by_regime"]["lambda3"]["nonquery_rows"] += 1
    elif defect == "missing_fit":
        models.pop()
    elif defect == "duplicate_fit":
        models[-1] = models[0]
    elif defect == "bool_seed":
        models[0]["seed"] = True
    else:
        complete = 1
    with pytest.raises(ValueError):
        M.criteria(models, hold, technical_complete=complete)
