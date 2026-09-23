"""Fabricated public arrays only; no models, native environments or saved data."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research.otto_score_forecast_data import build_windows, forecast_metrics, model_inputs


def episode(name, length, *, regime="base", split="train"):
    x = np.zeros((length, 31), dtype=np.float32)
    x[:, 0] = np.arange(length)
    x[:, 15] = (np.arange(length) / 2188).astype(np.float32)
    x[:, 16] = ((np.arange(length) % 4) / 2188).astype(np.float32)
    x[:, 17] = 1
    q = np.arange(length * 4, dtype=np.float32).reshape(length, 4)
    return {"id": name, "regime": regime, "split": split, "features": x,
            "teacher_scores": q, "legal": np.ones((length, 4), dtype=np.bool_)}


def test_all_disjoint_tails_and_query_only_windows_preserve_exact_source_bytes():
    source = [episode("one", 1), episode("four", 4), episode("five", 5), episode("seven", 7)]
    source[2]["teacher_scores"][4, 3] = np.float32(-0.0)
    w = build_windows(source)
    assert w["lengths"].tolist() == [1, 4, 4, 1, 4, 3]
    assert w["episode_index"].tolist() == [0, 1, 2, 2, 3, 3]
    assert w["step_offsets"].tolist() == [0, 0, 0, 4, 0, 4]
    assert w["counts"] == {"episodes": 4, "windows": 6, "rows": 17, "query_rows": 6,
                            "nonquery_rows": 11, "zero_support_episodes": 1, "query_only_windows": 2}
    for i, row in enumerate(source):
        chosen = np.flatnonzero(w["episode_index"] == i)
        for name, original in (("features", "features"), ("targets", "teacher_scores"), ("legal", "legal")):
            restored = np.concatenate([w[name][j, :w["lengths"][j]] for j in chosen])
            assert restored.tobytes() == row[original].tobytes()
        for j in chosen:
            assert w["query_scores"][j].tobytes() == row["teacher_scores"][w["step_offsets"][j]].tobytes()
    assert not w["legal"][~w["valid_mask"]].any()
    assert not w["features"][~w["valid_mask"]].any()
    assert not w["nonquery_mask"][:, 0].any()


def test_model_inputs_cannot_observe_skip_labels_or_future_query_scores():
    original = episode("episode", 8)
    changed = copy.deepcopy(original)
    changed["teacher_scores"][[1, 2, 3, 5, 6, 7]] += np.float32(100)
    a, b = build_windows([original]), build_windows([changed])
    assert set(model_inputs(a)) == {"features", "query_scores", "lengths"}
    assert all(model_inputs(a)[key].tobytes() == model_inputs(b)[key].tobytes() for key in model_inputs(a))
    assert a["targets"].tobytes() != b["targets"].tobytes()
    changed["teacher_scores"][4] += np.float32(10)
    c = build_windows([changed])
    assert c["query_scores"][0].tobytes() == a["query_scores"][0].tobytes()
    assert c["query_scores"][1].tobytes() != a["query_scores"][1].tobytes()


def test_detached_immutable_arrays_and_identity_order_without_split_mixing():
    source = [episode("valid-z", 2, split="valid"), episode("train-a", 3, regime="shift")]
    before = copy.deepcopy(source)
    w = build_windows(source)
    assert w["episode_ids"] == ("valid-z", "train-a")
    assert w["episode_splits"] == ("valid", "train")
    assert w["episode_regimes"] == ("base", "shift")
    for array in (value for value in w.values() if isinstance(value, np.ndarray)):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    for i, row in enumerate(source):
        for key in ("features", "teacher_scores", "legal"):
            assert row[key].tobytes() == before[i][key].tobytes()
    source[0]["features"].fill(999)
    source[0]["teacher_scores"].fill(999)
    assert w["features"][0, 0, 0] == 0 and w["query_scores"][0, 0] == 0
    duplicate = episode("valid-z", 1, split="train")
    with pytest.raises(ValueError, match="unique episode"):
        build_windows([before[0], duplicate])


def test_equal_episode_weights_keep_zero_support_and_rebalance_regimes_and_ages():
    source = [episode("short", 2), episode("long", 4), episode("empty", 1, regime="shift")]
    for row in source:
        row["teacher_scores"][:] = np.array([0, 2, 3, 4], dtype=np.float32)
    w = build_windows(source)
    assert w["nonquery_weights"][0].tolist() == [0, 1 / 3, 0, 0]
    assert w["nonquery_weights"][1].tolist() == [0, 1 / 9, 1 / 9, 1 / 9]
    assert w["nonquery_weights"][2].tolist() == [0, 0, 0, 0]
    pred = w["targets"].copy()
    pred[0, 1] = [5, 0, 6, 7]  # Short episode chooses action1, gap2.
    result = forecast_metrics(w, pred)
    all_rows = result["overall"]
    assert all_rows["weight_mass"] == pytest.approx(2 / 3)
    assert all_rows["episode_weighted_agreement"] == pytest.approx(1 / 3)
    assert all_rows["episode_weighted_raw_gap"] == pytest.approx(2 / 3)
    assert all_rows["supported_episode_agreement"] == .5
    assert all_rows["zero_support_episode_ids"] == ["empty"]
    base = result["by_regime"]["base"]
    assert base["episode_weighted_agreement"] == .5 and base["episode_weighted_raw_gap"] == 1
    assert base["by_age"]["1"]["episode_weighted_agreement"] == .5
    assert base["by_age"]["2"]["episode_weighted_agreement"] == .5
    assert base["by_age"]["2"]["supported_episode_agreement"] == 1
    assert base["by_age"]["2"]["zero_support_episode_ids"] == ["short"]
    assert result["by_regime"]["shift"]["supported_episode_agreement"] is None
    assert [all_rows["by_age"][str(age)]["nonquery_rows"] for age in (1, 2, 3)] == [2, 1, 1]


def test_teacher_near_minimum_membership_is_not_first_action_match_and_gap_is_raw():
    source = episode("ties", 4)
    epsilon = np.float32(1e-10)
    inside = np.nextafter(epsilon, np.float32(0))
    source["teacher_scores"][:] = [0, 0, 5, -99]
    source["teacher_scores"][1, 1] = inside
    source["teacher_scores"][2, 1] = epsilon
    source["legal"][:, 3] = False
    w = build_windows([source])
    pred = np.tile(np.asarray([2, 0, 9, -100], dtype=np.float32), (1, 4, 1))
    result = forecast_metrics(w, pred)["overall"]
    assert result["episode_weighted_agreement"] == pytest.approx(2 / 3)
    assert result["episode_weighted_first_argmin_match"] == 0
    assert result["episode_weighted_raw_gap"] == pytest.approx((float(inside) + float(epsilon)) / 3)
    assert result["by_age"]["1"]["episode_weighted_agreement"] == 1
    assert result["by_age"]["2"]["episode_weighted_agreement"] == 0


def test_float32_prediction_tie_uses_first_eligible_action_and_finite_extremes_gap():
    source = episode("precision", 2)
    largest = np.finfo(np.float32).max
    source["teacher_scores"][:] = [largest, -largest, 0, 0]
    w = build_windows([source])
    pred = w["targets"].copy()
    pred[0, 1] = [np.nextafter(np.float32(1e-10), np.float32(0)), 0, 1, 2]
    result = forecast_metrics(w, pred)["overall"]
    assert result["episode_weighted_agreement"] == 0
    assert result["episode_weighted_raw_gap"] == float(largest) - float(-largest)
    assert np.isfinite(result["episode_weighted_raw_gap"])


def test_constant_teacher_hold_query_baseline_has_zero_gap_and_ignores_query_and_padding():
    source = [episode("a", 7), episode("b", 1)]
    for row in source:
        row["teacher_scores"][:] = [-0.0, 3, 2, 1]
    w = build_windows(source)
    predictions = np.repeat(w["query_scores"][:, None, :], 4, axis=1)
    predictions[:, 0] = [9, -8, -7, -6]
    predictions[~w["valid_mask"]] = [9, -8, -7, -6]
    result = forecast_metrics(w, predictions)["overall"]
    assert result["nonquery_rows"] == 5 and result["episode_weighted_raw_gap"] == 0
    assert result["episode_weighted_agreement"] == .5
    assert result["supported_episode_agreement"] == 1
    empty = forecast_metrics(build_windows([episode("only-query", 1)]), np.zeros((1, 4, 4), np.float32))
    assert empty["overall"]["weight_mass"] == 0
    assert empty["overall"]["supported_episode_raw_gap"] is None
    assert set(empty["overall"]["by_age"]) == {"1", "2", "3"}


def test_centered_mse_uses_only_current_legal_actions_and_removes_common_shift():
    source = episode("centered", 3)
    source["teacher_scores"][:] = [1, 3, -999, 999]
    source["legal"][:, 2:] = False
    w = build_windows([source])
    pred = np.tile(np.asarray([101, 103, 9999, -9999], dtype=np.float32), (1, 4, 1))
    zero = forecast_metrics(w, pred)["overall"]
    assert zero["episode_weighted_centered_mse"] == 0
    pred[0, 2, :2] = [102, 106]  # Centered [-2,2] versus [-1,1]: MSE1.
    result = forecast_metrics(w, pred)["overall"]
    assert result["episode_weighted_centered_mse"] == .5
    assert result["by_age"]["1"]["episode_weighted_centered_mse"] == 0
    assert result["by_age"]["2"]["episode_weighted_centered_mse"] == 1


@pytest.mark.parametrize("defect", ["feature_dtype", "score_dtype", "mask_dtype", "nan", "no_legal",
                                    "chronology", "query_age", "has_query", "empty", "extra_field"])
def test_invalid_inputs_rejected_without_casts_or_feature_repairs(defect):
    row = episode("invalid", 3)
    if defect == "feature_dtype":
        row["features"] = row["features"].astype(np.float64)
    elif defect == "score_dtype":
        row["teacher_scores"] = row["teacher_scores"].astype(np.float64)
    elif defect == "mask_dtype":
        row["legal"] = row["legal"].astype(np.int64)
    elif defect == "nan":
        row["teacher_scores"][1, 0] = np.nan
    elif defect == "no_legal":
        row["legal"][2] = False
    elif defect == "chronology":
        row["features"][1, 15] = 0
    elif defect == "query_age":
        row["features"][2, 16] = 0
    elif defect == "has_query":
        row["features"][0, 17] = 0
    elif defect == "empty":
        row = episode("invalid", 0)
    else:
        row["hidden_truth"] = [1, 2]
    with pytest.raises(ValueError):
        build_windows([row])


def test_metric_rejects_wrong_prediction_dtype_and_dropped_tail_support():
    w = build_windows([episode("a", 5)])
    with pytest.raises(ValueError, match="predicted scores"):
        forecast_metrics(w, w["targets"].astype(np.float64))
    broken = dict(w)
    broken["lengths"] = np.asarray([3, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="active-prefix"):
        forecast_metrics(broken, w["targets"])
