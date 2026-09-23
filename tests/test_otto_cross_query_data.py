"""Fabricated full histories, poison boundaries and sparse target weights."""
import copy

import numpy as np
import pytest

from openjev.research.otto_cross_query_data import batch_chunk, build_training, project_training
from openjev.research.otto_sampled_forecast_data import select_windows


def fixture(lengths=None):
    lengths = lengths or [65] * 54
    episodes, selections, queries, selected = [], [], [], []
    for index, length in enumerate(lengths):
        steps = np.arange(length)
        features = np.zeros((length, 31), dtype=np.float32)
        features[:, 0] = index
        features[:, 15] = steps / 2188
        features[:, 16] = (steps % 4) / 2188
        features[:, 17] = 1
        episode = {"id": f"e{index}", "regime": "a", "split": "train", "features": features,
                   "legal": np.ones((length, 4), dtype=np.bool_), "actions": np.zeros(length, np.int64)}
        score = np.arange(length * 4, dtype=np.float32).reshape(length, 4) + index
        selection = {"episode_id": episode["id"], **select_windows(length, 701 + index)}
        episodes.append(episode)
        selections.append(selection)
        queries.append(score[::4].copy())
        selected.append({start: score[start:min(start + 4, length)].copy() for start in selection["start_offsets"]})
    return episodes, selections, queries, selected


def test_complete_queries_sparse_targets_and_nonrenormalized_weights():
    args = fixture()
    data = build_training(*args)
    assert data["counts"]["query_rows"] == 54 * 17
    for index, selection in enumerate(args[1]):
        low = index * 65
        np.testing.assert_array_equal(data["query_scores"][low:low + 65:4], args[2][index])
        selected = {step for start in selection["start_offsets"] for step in range(start + 1, min(start + 4, 65))}
        for step in range(65):
            assert (data["weights"][low + step] > 0) == (step in selected)
            if step in selected:
                assert data["weights"][low + step] == (17 / 8) / (54 * 48)
            else:
                assert data["targets"][low + step].tobytes() == np.zeros(4, np.float32).tobytes()
    assert not data["features"].flags.writeable
    with pytest.raises(ValueError):
        data["features"].flags.writeable = True


def test_target_changes_do_not_change_model_inputs():
    args = fixture()
    before = build_training(*args)
    for scores in args[3]:
        for score in scores.values():
            score[1:] += 100
    after = build_training(*args)
    for key in ("features", "query_scores", "query_mask", "episode_offsets"):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before["targets"], after["targets"])


def test_anchor_mismatch_and_missing_query_rejected():
    args = fixture()
    args[2][0][args[1][0]["start_offsets"][0] // 4, 0] += 1
    with pytest.raises(ValueError, match="sampled anchor"):
        build_training(*args)
    args = fixture()
    args[2][0] = args[2][0][:-1]
    with pytest.raises(ValueError, match="complete true-query"):
        build_training(*args)


def test_chunk_ends_padding_and_exact_model_input_allowlist():
    data = build_training(*fixture([1, 32, 33, 64, 65, *([65] * 49)]))
    chunk = batch_chunk(data, [0, 1, 2, 3, 4], 0)
    inputs = chunk["model_inputs"]
    assert set(inputs) == {"features", "query_scores", "lengths", "query_mask", "episode_ends"}
    assert inputs["lengths"].tolist() == [1, 32, 32, 32, 32]
    assert inputs["episode_ends"].tolist() == [True, True, False, False, False]
    assert np.isnan(inputs["features"][0, 1:]).all()
    assert np.isnan(inputs["query_scores"][~inputs["query_mask"]]).all()
    next_chunk = batch_chunk(data, [0, 1, 2, 3, 4], 32)["model_inputs"]
    assert next_chunk["lengths"].tolist() == [0, 0, 1, 32, 32]
    assert next_chunk["episode_ends"].tolist() == [False, False, True, True, False]
    final = batch_chunk(data, [0, 1, 2, 3, 4], 64)["model_inputs"]
    assert final["lengths"].tolist() == [0, 0, 0, 0, 1]
    assert final["episode_ends"].tolist() == [False, False, False, False, True]


def test_zero_support_retained_and_query_only_loss_zero():
    data = build_training(*fixture([1] * 54))
    assert data["counts"]["episodes"] == data["counts"]["zero_support_episodes"] == 54
    assert data["weights"].sum() == 0
    assert data["targets"].sum() == 0


def flat_fixture():
    episodes, selections, queries, selected = fixture()
    raw = np.full((54 * 65, 4), np.nan, dtype=np.float32)
    for index in range(54):
        low = index * 65
        raw[low:low + 65:4] = queries[index]
        for start, score in selected[index].items():
            raw[low + start:low + start + len(score)] = score
    flat = {"features": np.concatenate([e["features"] for e in episodes]), "raw_q": raw,
            "legal": np.concatenate([e["legal"] for e in episodes]),
            "actions": np.concatenate([e["actions"] for e in episodes]),
            "correction": np.tile(np.arange(65) % 4 == 0, 54),
            "label_mask": np.ones(54 * 65, np.bool_), "episode_offsets": np.arange(55, dtype=np.int64) * 65}
    identities = [{"stage": "train", "episode_index": index, "episode_id": f"e{index}", "regime": "a"}
                  for index in range(54)]
    return flat, identities, selections


def test_incidental_nonquery_labels_are_never_decoded_as_inputs_or_loss():
    args = flat_fixture()
    before = project_training(*args, selection_start=701)
    assert np.isnan(args[0]["raw_q"]).any()
    args[0]["raw_q"][np.isnan(args[0]["raw_q"])] = 99999
    after = project_training(*args, selection_start=701)
    for key in ("features", "query_scores", "query_mask", "targets", "weights"):
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("fault", ["query_provenance", "target_provenance", "seed", "missing_placeholder", "query_nan"])
def test_projection_rejects_missing_provenance_or_malformed_inputs(fault):
    flat, identities, selections = flat_fixture()
    if fault == "query_provenance":
        flat["label_mask"][0] = False
        flat["raw_q"][0] = 0
    elif fault == "target_provenance":
        step = next(s + 1 for s in selections[0]["start_offsets"] if s + 1 < 65)
        flat["label_mask"][step] = False
        flat["raw_q"][step] = 0
    elif fault == "seed":
        selections = copy.deepcopy(selections)
        selections[0]["seed"] += 1
    elif fault == "missing_placeholder":
        step = int(np.flatnonzero(np.isnan(flat["raw_q"][:, 0]))[0])
        flat["label_mask"][step] = False
        flat["raw_q"][step] = -0.
    else:
        flat["raw_q"][0] = np.nan
    with pytest.raises(ValueError):
        project_training(flat, identities, selections, selection_start=701)


def test_invalid_chunk_boundaries_and_duplicate_lanes_rejected():
    data = build_training(*fixture())
    with pytest.raises(ValueError):
        batch_chunk(data, [0], 4)
    with pytest.raises(ValueError):
        batch_chunk(data, [0, 0], 0)
    with pytest.raises(ValueError):
        batch_chunk(data, [54], 0)
