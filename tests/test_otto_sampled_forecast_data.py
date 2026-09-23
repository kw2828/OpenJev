"""Fabricated complete public histories only; no model or environment calls."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from openjev.research.otto_sampled_forecast_data import build_sampled_windows, select_windows
from openjev.research.otto_score_forecast_data import model_inputs


def fixtures(lengths=()):
    lengths = list(lengths) + [1] * (54-len(lengths))
    episodes, selections, scores = [], [], []
    for index, length in enumerate(lengths):
        x = np.zeros((length, 31), dtype=np.float32)
        steps = np.arange(length)
        x[:, 0] = steps
        x[:, 2:6] = 1
        x[:, 15], x[:, 16], x[:, 17] = steps / 2188, (steps % 4) / 2188, 1
        identity = f"train-{index}"
        episodes.append({"id": identity, "regime": "lambda3" if index < 27 else "lambda4", "split": "train",
                         "features": x, "legal": np.ones((length, 4), dtype=np.bool_),
                         "actions": np.zeros(length, dtype=np.int64)})
        selected = {"episode_id": identity, **select_windows(length, 10000+index)}
        selections.append(selected)
        scores.append({start: np.arange(start*4, min(start+4, length)*4, dtype=np.float32).reshape(-1, 4)
                       for start in selected["start_offsets"]})
    return episodes, selections, scores


def test_selection_exact_population_seed_replay_and_global_rng_is_untouched():
    before = np.random.get_state()
    for length in (1, 4, 5, 31, 32, 33, 2188):
        record = select_windows(length, 12345)
        population = (length+3)//4
        count = min(8, population)
        assert record == select_windows(length, 12345)
        assert record["population_windows"] == population and record["selected_windows"] == count
        assert record["inclusion_numerator"] == count and record["inclusion_denominator"] == population
        assert record["inclusion_probability"] == count/population
        assert record["start_offsets"] == sorted(set(record["start_offsets"]))
        assert len(record["start_offsets"]) == count
        assert all(0 <= step < length and step % 4 == 0 for step in record["start_offsets"])
        if population <= 8:
            assert record["start_offsets"] == list(range(0, length, 4))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert select_windows(2188, 1)["start_offsets"] != select_windows(2188, 2)["start_offsets"]


def test_selected_tails_query_only_windows_absolute_features_and_exact_weights():
    episodes, selections, scores = fixtures((1, 4, 5, 7, 33))
    scores[2][4][0, 3] = np.float32(-0.)
    windows = build_sampled_windows(episodes, selections, scores)
    assert windows["counts"]["episodes"] == 54 and windows["counts"]["zero_support_episodes"] == 50
    assert windows["episode_lengths"].tolist() == [1, 4, 5, 7, 33] + [1]*49
    assert windows["episode_nonquery_counts"].tolist() == [0, 3, 3, 5, 24] + [0]*49
    for i, episode in enumerate(episodes):
        chosen = np.flatnonzero(windows["episode_index"] == i)
        assert windows["step_offsets"][chosen].tolist() == selections[i]["start_offsets"]
        for window in chosen:
            start, length = int(windows["step_offsets"][window]), int(windows["lengths"][window])
            assert windows["features"][window, :length].tobytes() == episode["features"][start:start+length].tobytes()
            assert windows["targets"][window, :length].tobytes() == scores[i][start].tobytes()
            assert windows["query_scores"][window].tobytes() == scores[i][start][0].tobytes()
            assert not windows["valid_mask"][window, length:].any()
            assert not windows["legal"][window, length:].any()
    for index, expected in ((1, 1/(54*3)), (2, 1/(54*3)), (3, 1/(54*5)), (4, (9/8)/(54*24))):
        rows = windows["episode_index"] == index
        assert np.all(windows["nonquery_weights"][rows][windows["nonquery_mask"][rows]] == expected)
    assert not windows["nonquery_weights"][~windows["nonquery_mask"]].any()
    assert windows["sampling"]["renormalized"] is False
    assert windows["sampling"]["shared_across_fits"] is True


def test_enumerated_nine_subsets_prove_full_episode_importance_expectation_without_renormalization():
    # T=33 has eight three-label windows and one query-only tail. All nine
    # possible eight-of-nine samples are exercised, not a Monte Carlo estimate.
    seeds = {}
    for seed in range(256):
        offsets = select_windows(33, seed)["start_offsets"]
        missing = tuple(set(range(0, 33, 4)) - set(offsets))
        seeds.setdefault(missing, seed)
    assert set(seeds) == {(offset,) for offset in range(0, 33, 4)}
    estimates, masses = [], []
    for seed in seeds.values():
        episodes, selections, scores = fixtures((33,))
        selections[0] = {"episode_id": episodes[0]["id"], **select_windows(33, seed)}
        scores[0] = {start: np.zeros((min(4, 33-start), 4), dtype=np.float32)
                     for start in selections[0]["start_offsets"]}
        windows = build_sampled_windows(episodes, selections, scores)
        estimate = 0.
        for row in np.flatnonzero(windows["episode_index"] == 0):
            for age in range(1, int(windows["lengths"][row])):
                step = int(windows["step_offsets"][row]) + age
                estimate += windows["nonquery_weights"][row, age] * (step*step + 1)
        estimates.append(estimate)
        masses.append(float(windows["nonquery_weights"].sum()))
    full = sum(step*step+1 for step in range(33) if step % 4) / (54*24)
    assert sum(estimates)/9 == pytest.approx(full, rel=1e-14)
    assert sum(masses)/9 == pytest.approx(1/54, rel=1e-14)
    assert min(masses) < 1/54 < max(masses)  # Realized weights were not renormalized.


def test_unselected_labels_absent_and_nonquery_labels_cannot_enter_model_inputs():
    episodes, selections, scores = fixtures((80,))
    first = build_sampled_windows(episodes, selections, scores)
    changed = copy.deepcopy(scores)
    for matrix in changed[0].values():
        matrix[1:] += np.float32(1000)
    second = build_sampled_windows(episodes, selections, changed)
    assert set(model_inputs(first)) == {"features", "query_scores", "lengths"}
    assert all(model_inputs(first)[k].tobytes() == model_inputs(second)[k].tobytes() for k in model_inputs(first))
    assert first["targets"].tobytes() != second["targets"].tobytes()
    assert len(scores[0]) == 8 < (80+3)//4
    assert first["counts"]["full_rows"] > first["counts"]["rows"]


def test_detached_readonly_arrays_and_selection_metadata_do_not_mutate_inputs():
    episodes, selections, scores = fixtures((40, 7))
    before = copy.deepcopy((episodes, selections, scores))
    windows = build_sampled_windows(episodes, selections, scores)
    for value in windows.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flags.writeable = True
    for i, episode in enumerate(episodes):
        assert all(episode[k].tobytes() == before[0][i][k].tobytes() for k in ("features", "legal", "actions"))
        assert all(scores[i][k].tobytes() == before[2][i][k].tobytes() for k in scores[i])
    assert selections == before[1]
    selections[0]["start_offsets"].clear()
    episodes[0]["features"].fill(999)
    next(iter(scores[0].values())).fill(999)
    assert windows["sampling"]["selections"][0]["start_offsets"]
    assert not np.any(windows["features"] == 999) and not np.any(windows["targets"] == 999)


@pytest.mark.parametrize("defect", ["episode_count", "valid", "duplicate_id", "duplicate_seed", "selection",
                                    "missing_score", "extra_score", "nan_score", "score_dtype", "blocked_action",
                                    "feature_chronology", "query_age", "unselected_score_field"])
def test_malformed_or_selected_only_denominator_inputs_are_rejected(defect):
    episodes, selections, scores = fixtures((40, 7))
    if defect == "episode_count":
        episodes.pop()
    elif defect == "valid":
        episodes[0]["split"] = "valid"
    elif defect == "duplicate_id":
        episodes[1]["id"] = episodes[0]["id"]
    elif defect == "duplicate_seed":
        selections[1] = {"episode_id": episodes[1]["id"], **select_windows(7, selections[0]["seed"])}
    elif defect == "selection":
        selections[0]["inclusion_denominator"] = 8
    elif defect == "missing_score":
        scores[0].pop(next(iter(scores[0])))
    elif defect == "extra_score":
        missing = next(i for i in range(0, 40, 4) if i not in scores[0])
        scores[0][missing] = np.zeros((4, 4), dtype=np.float32)
    elif defect == "nan_score":
        next(iter(scores[0].values()))[0, 0] = np.nan
    elif defect == "score_dtype":
        key = next(iter(scores[0]))
        scores[0][key] = scores[0][key].astype(np.float64)
    elif defect == "blocked_action":
        episodes[0]["legal"][0, 0] = False
    elif defect == "feature_chronology":
        episodes[0]["features"][39, 15] = 0
    elif defect == "query_age":
        episodes[0]["features"][7, 16] = 0
    else:
        episodes[0]["teacher_scores"] = np.zeros((40, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        build_sampled_windows(episodes, selections, scores)


def test_selector_rejects_implicit_or_out_of_bounds_parameters():
    for length, seed in ((0, 1), (2189, 1), (True, 1), (4.0, 1), (4, True), (4, -1), (4, 2**32), (4, np.int64(1))):
        with pytest.raises(ValueError):
            select_windows(length, seed)
