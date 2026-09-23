"""Fabricated-only arithmetic and complete-cohort contracts; no saved study data."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from openjev.research import otto_ranking_diagnosis as m
from openjev.research.otto_score_forecast_data import build_windows


def row_values(q, p, legal=None):
    q, p = np.asarray(q, np.float32), np.asarray(p, np.float32)
    allowed = np.ones(q.shape, np.bool_) if legal is None else np.asarray(legal, np.bool_)
    teacher = m._teacher(q, allowed)
    return teacher, m._row_metrics(p, teacher)


def test_pair_partition_has_hand_computed_mse_and_ignores_illegal_actions():
    teacher, value = row_values([[0, 0, 4, 8]], [[0, 2, 3, 7]])
    assert value["legal_centered_mse"].tolist() == [1.5]
    assert value["pair_mse_best_best"].tolist() == [.25]
    assert value["pair_mse_best_rest"].tolist() == [1.25]
    assert value["pair_mse_rest_rest"].tolist() == [0.]
    assert teacher["margin"].tolist() == [4.]
    assert value["pair_residual"].tolist() == [0.]
    _, first = row_values([[0, 2, 4, 1000]], [[1, 0, 5, -1000]], [[1, 1, 1, 0]])
    _, second = row_values([[0, 2, 4, -1e30]], [[1, 0, 5, 1e30]], [[1, 1, 1, 0]])
    assert first["legal_centered_mse"][0] == pytest.approx(2.)
    assert first["raw_gap"].tolist() == [2.]
    for key in m.METRICS:
        np.testing.assert_array_equal(first[key], second[key])


def test_strict_argmin_identity_is_distinct_from_near_minimum_selection():
    tiny = np.float32(5e-11)
    _, value = row_values([[tiny, 0, 1, 2]], [[0, tiny, 1, 2]])
    assert value["agreement"].tolist() == [1.]
    assert value["first_argmin_match"].tolist() == [1.]
    assert value["raw_gap"][0] == float(tiny)
    assert value["strict_argmin_raw_gap"][0] == float(tiny)
    assert value["strict_argmin_prediction_margin"][0] == -float(tiny)
    assert value["strict_argmin_error_difference"][0] == -2 * float(tiny)
    assert value["strict_residual"].tolist() == [0.]
    _, reverse = row_values([[0, tiny, 1, 2]], [[tiny, 0, 1, 2]])
    assert reverse["selection_differs_from_strict_mass"].tolist() == [1.]


@pytest.mark.parametrize("distance,expected", [(np.nextafter(np.float32(1e-10), np.float32(0)), True),
                                             (np.float32(1e-10), False)])
def test_tie_threshold_is_strict_float32(distance, expected):
    near, _ = m._near(np.asarray([[distance, 0, 2, 3]], np.float32), np.ones((1, 4), np.bool_))
    assert bool(near[0, 0]) is expected


def test_near_minimum_classes_partition_every_row_including_mixed_ties():
    teacher, value = row_values(
        [[0, 1, 3, 4], [0, 1, 3, 4], [0, 1, 3, 4], [1, 0, 3, 4]],
        [[0, 1, 3, 4], [1, 0, 3, 4], [5e-11, 0, 3, 4], [5e-11, 0, 3, 4]],
    )
    keys = ("near_all_best_mass", "near_no_best_mass", "near_mixed_correct_mass", "near_mixed_wrong_mass")
    np.testing.assert_array_equal(np.stack([value[k] for k in keys]), np.eye(4))
    assert value["agreement"].tolist() == [1., 0., 1., 0.]
    assert teacher["margin"].tolist() == [1., 1., 1., 1.]


def test_no_alternative_margins_and_zero_support_are_explicit():
    teacher, values = row_values([[2, 2, 2, 2], [1, 20, 30, 40]],
                                [[4, 3, 2, 1], [-2, 8, 9, 10]], [[1, 1, 1, 1], [1, 0, 0, 0]])
    item = m._individual(values, teacher, np.asarray([0, 1]))
    assert item["teacher_margin"] == {"rows": 0, "mass": 0., "contribution": 0., "conditional_mean": None}
    assert item["metrics"]["agreement"] == 1.
    empty = m._individual(values, teacher, np.asarray([], np.int64))
    assert empty["rows"] == empty["supported_episodes"] == empty["weight_mass"] == 0
    assert all(v == 0. for v in empty["metrics"].values())
    assert empty["teacher_margin"]["conditional_mean"] is None


def test_all_four_transition_cells_and_full_denominator_are_preserved():
    teacher, left = row_values([[0, 2, 4, 6]] * 4,
                               [[0, 2, 4, 6], [0, 2, 4, 6], [2, 0, 4, 6], [2, 0, 4, 6]])
    _, right = row_values([[0, 2, 4, 6]] * 4,
                         [[0, 2, 4, 6], [2, 0, 4, 6], [0, 2, 4, 6], [2, 0, 4, 6]])
    result = m._paired(left, right, teacher, np.arange(4), 8)
    assert result["rows"] == 4 and result["normalization_rows"] == 8
    assert result["weight_mass"] == .5
    assert list(result["cells"]) == list(m.CELLS)
    for cell in result["cells"].values():
        assert cell["rows"] == 1 and cell["mass"] == .125
    assert result["cells"]["CW"]["delta"]["agreement"] == -.125
    assert result["cells"]["WC"]["delta"]["agreement"] == .125
    assert result["cells"]["CW"]["delta"]["raw_gap"] == .25
    assert result["cells"]["WC"]["delta"]["raw_gap"] == -.25
    empty = m._paired(left, right, teacher, np.asarray([], np.int64), 0)
    assert all(cell["mass"] == cell["rows"] == 0 for cell in empty["cells"].values())


def test_episode_and_margin_denominators_do_not_condition_away_unsupported_paths():
    teacher, value = row_values([[0, 2, 4, 6], [0, 4, 6, 8]], [[0, 2, 4, 6], [0, 4, 6, 8]])
    one = m._individual(value, teacher, np.asarray([0]))
    two = m._individual(value, teacher, np.asarray([0, 1]))
    empty = m._individual(value, teacher, np.asarray([], np.int64))
    reduced = m._combine([one, two, empty])
    assert reduced["episodes"] == 3 and reduced["supported_episodes"] == 2 and reduced["rows"] == 3
    assert reduced["metrics"]["agreement"] == pytest.approx(2 / 3)
    assert reduced["teacher_margin"]["mass"] == pytest.approx(2 / 3)
    assert reduced["teacher_margin"]["contribution"] == pytest.approx(5 / 3)
    assert reduced["teacher_margin"]["conditional_mean"] == pytest.approx(2.5)


def fixture():
    episodes, identities = [], []
    for regime in m.REGIMES:
        for case in range(6):
            for arm in m.ARMS:
                index = len(episodes)
                length = {0: 8, 1: 6, 18: 5}.get(index, 1)
                features = np.zeros((length, 31), np.float32)
                features[:, 15] = np.arange(length) / 2188
                features[:, 16] = (np.arange(length) % 4) / 2188
                features[:, 17] = 1
                episode_id = f"{regime}:{case}:{arm}"
                episodes.append({"id": episode_id, "regime": regime, "split": "valid", "features": features,
                                 "teacher_scores": np.tile(np.asarray([0, 1, 2, 3], np.float32), (length, 1)),
                                 "legal": np.ones((length, 4), np.bool_)})
                identities.append({"episode_id": episode_id, "regime": regime, "case": case, "arm": arm})
    windows = build_windows(episodes)
    held = np.repeat(windows["query_scores"][:, None, :], 4, axis=1)
    held[~windows["valid_mask"]] = 0
    predictions = {("hold", None): held}
    predictions.update({(family, seed): windows["targets"].copy() for family in m.FAMILIES for seed in m.SEEDS})
    for seed in m.SEEDS:
        for episode in (0, 1):
            w = next(i for i, pair in enumerate(zip(windows["episode_index"], windows["step_offsets"], strict=True))
                     if pair == (episode, 4))
            predictions["innovation_shared_mse", seed][w, 1] = [1, 0, 2, 3]
        predictions["innovation_shared_aux", seed][0, 1] = [1, 0, 2, 3]
    return windows, predictions, identities


@pytest.fixture(scope="module")
def complete_report():
    windows, predictions, identities = fixture()
    before_windows = {k: v.tobytes() for k, v in windows.items() if isinstance(v, np.ndarray)}
    before_predictions = {k: v.tobytes() for k, v in predictions.items()}
    identities_before = copy.deepcopy(identities)
    result = m.analyze(windows, predictions, identities)
    assert before_windows == {k: v.tobytes() for k, v in windows.items() if isinstance(v, np.ndarray)}
    assert before_predictions == {k: v.tobytes() for k, v in predictions.items()}
    assert identities == identities_before
    return result


def test_complete_output_coverage_serialization_and_fixed_comparisons(complete_report):
    result = complete_report
    expected = {"individual_episodes": 5400, "individual_groups": 3150, "individual_seed_means": 1008,
                "paired_episodes": 7776, "paired_groups": 4536, "paired_seed_means": 1512,
                "phase_episodes": 2592, "phase_groups": 1512, "phase_seed_means": 504}
    for key, count in expected.items():
        assert len(result[key]) == result["counts"][key] == count
    assert result["counts"]["policies"] == 25 and result["counts"]["originating_cases"] == 12
    assert len(m.contrasts()) == len({c[0] for c in m.contrasts()}) == 12
    assert len(result["definitions"]["groups"]) == 21
    assert json.loads(json.dumps(result["paired_episodes"][0], allow_nan=False))["stats"]["episodes"] == 1
    assert not any("passes" in row for row in result["individual_groups"])


def test_full_phase_identity_differs_from_reweighted_scope_metrics(complete_report):
    result = complete_report
    name = "innovation.shared_mse_to_shared_aux"
    seed = m.SEEDS[0]
    def value(table, field, label):
        return next(r["stats"] for r in result[table] if r["contrast"] == name and r.get("seed") == seed
                    and r["group"] == "lambda3" and r[field] == label)
    def delta(stats):
        return math.fsum(cell["delta"]["agreement"] for cell in stats["cells"].values())
    full = delta(value("paired_groups", "scope", "full"))
    initial = delta(value("phase_groups", "phase", "initial"))
    post = delta(value("phase_groups", "phase", "postcorrection"))
    assert initial == pytest.approx(-1 / 108)
    assert post == pytest.approx(5 / 216)
    assert full == pytest.approx(1 / 72)
    assert initial + post == pytest.approx(full)
    separately_weighted = delta(value("paired_groups", "scope", "initial")) + delta(value("paired_groups", "scope", "postcorrection"))
    assert separately_weighted != pytest.approx(full)
    post2 = value("paired_groups", "scope", "post_age2")
    assert post2["episodes"] == 18 and post2["supported_episodes"] == 1
    case = next(r["stats"] for r in result["individual_groups"] if r["family"] == "hold"
                and r["scope"] == "postcorrection" and r["group"] == "lambda3/case0")
    assert case["episodes"] == 3 and case["supported_episodes"] == 2
    means = next(r["stats"] for r in result["paired_seed_means"] if r["contrast"] == name
                 and r["scope"] == "full" and r["group"] == "lambda3")
    assert means["episodes"] == 18 and means["rows"] == 10
    assert delta(means) == pytest.approx(full)


@pytest.mark.parametrize("change", ["missing_model", "nonfinite", "duplicate_identity", "bad_query", "padding"])
def test_invalid_input_contract_rejected_before_analysis(change):
    windows, predictions, identities = fixture()
    key = (m.FAMILIES[0], m.SEEDS[0])
    if change == "missing_model":
        predictions.pop(key)
    elif change == "nonfinite":
        predictions[key][0, 1, 0] = np.nan
    elif change == "duplicate_identity":
        identities[1]["case"], identities[1]["arm"] = identities[0]["case"], identities[0]["arm"]
    elif change == "bad_query":
        predictions[key][0, 0, 0] = 1
    else:
        predictions[key][-1, 3, 0] = -0.
    with pytest.raises(ValueError):
        m._inputs(windows, predictions, identities)
