"""Fabricated complete episodes for separate pre-assimilation targets."""
import copy

import numpy as np
import pytest

from openjev.research import otto_cross_query_data as original
from openjev.research.otto_prequery_data import SELECTION_START, batch_chunk, project_training
from openjev.research.otto_sampled_forecast_data import select_windows


def fixture(lengths=None):
    lengths = lengths or [1, 4, 5, 8, 9, 33, *([1] * 48)]
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    features = np.zeros((total, 31), np.float32)
    raw = np.full((total, 4), np.nan, np.float32)
    correction = np.zeros(total, np.bool_)
    selections, identities = [], []
    for index, length in enumerate(lengths):
        low, high = (int(x) for x in offsets[index:index + 2])
        step = np.arange(length)
        features[low:high, 15] = step / 2188
        features[low:high, 16] = step % 4 / 2188
        features[low:high, 17] = 1
        score = (np.arange(length * 4, dtype=np.float32).reshape(length, 4) + index)
        if length > 4:
            score[4, 0] = -0.
        record = {"episode_id": f"e{index}", **select_windows(length, SELECTION_START + index)}
        raw[low:high:4] = score[::4]
        correction[low:high:4] = True
        for start in record["start_offsets"]:
            stop = min(length, start + 4)
            raw[low + start:low + stop] = score[start:stop]
        selections.append(record)
        identities.append({"stage": "train", "episode_index": index, "episode_id": f"e{index}", "regime": "a"})
    legal = np.ones((total, 4), np.bool_)
    legal[:, 3] = False
    flat = {"features": features, "raw_q": raw, "legal": legal, "actions": np.zeros(total, np.int64),
            "correction": correction, "label_mask": np.ones(total, np.bool_), "episode_offsets": offsets}
    return flat, identities, selections


def test_later_queries_have_full_vectors_and_fixed_episode_weight_with_short_tails():
    flat, identities, selections = fixture()
    data = project_training(flat, identities, selections)
    expected_counts = [0, 0, 1, 1, 2, 8, *([0] * 48)]
    assert data["counts"]["prior_rows"] == 12
    assert data["counts"]["prior_supported_episodes"] == 4
    assert data["counts"]["prior_zero_support_episodes"] == 50
    assert data["counts"]["prior_weight_mass"] == pytest.approx(4 / 54)
    for index, count in enumerate(expected_counts):
        low, high = (int(x) for x in flat["episode_offsets"][index:index + 2])
        expected = np.zeros(high - low, np.bool_)
        expected[4::4] = True
        np.testing.assert_array_equal(data["prior_mask"][low:high], expected)
        assert not data["prior_mask"][low]
        assert data["prior_weights"][low:high].sum() == pytest.approx(1 / 54 if count else 0)
        for row in range(low, high):
            if data["prior_mask"][row]:
                assert data["prior_targets"][row].tobytes() == flat["raw_q"][row].tobytes()
                assert data["prior_weights"][row] == 1 / (54 * count)
                assert not flat["legal"][row, 3] and data["prior_targets"][row, 3] != 0
            else:
                assert data["prior_targets"][row].tobytes() == np.zeros(4, np.float32).tobytes()
                assert data["prior_weights"][row] == 0


def test_old_inputs_targets_weights_and_end_semantics_are_byte_identical():
    args = fixture()
    old = original.project_training(*args, selection_start=SELECTION_START)
    new = project_training(*args)
    for key, value in old.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == new[key].tobytes()
    for start in (0, 32, 64):
        before = original.batch_chunk(old, [0, 2, 5], start)
        after = batch_chunk(new, [0, 2, 5], start)
        assert set(after["model_inputs"]) == {"features", "query_scores", "lengths", "query_mask", "episode_ends"}
        for key, value in before["model_inputs"].items():
            assert value.tobytes() == after["model_inputs"][key].tobytes()
        for key in ("targets", "legal", "weights"):
            assert before[key].tobytes() == after[key].tobytes()
        lengths = after["model_inputs"]["lengths"]
        for lane, length in enumerate(lengths):
            assert not after["prior_mask"][lane, length:].any()
            assert not after["prior_weights"][lane, length:].any()
            assert after["prior_targets"][lane, length:].tobytes() == np.zeros((32 - length, 4), np.float32).tobytes()
    tail = batch_chunk(new, [5], 32)
    assert tail["prior_mask"][0].tolist() == [True, *([False] * 31)]
    assert tail["model_inputs"]["episode_ends"].tolist() == [True]


def test_unselected_nonquery_labels_inert_and_new_arrays_immutable_without_caller_mutation():
    args = fixture([65, *([1] * 53)])
    raw_before = args[0]["raw_q"].tobytes()
    selections_before = copy.deepcopy(args[2])
    before = project_training(*args)
    assert args[0]["raw_q"].tobytes() == raw_before
    assert args[2] == selections_before
    poison = np.isnan(args[0]["raw_q"])
    assert poison.any()
    args[0]["raw_q"][poison] = 123456
    after = project_training(*args)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == after[key].tobytes()
    for key in ("prior_targets", "prior_weights", "prior_mask"):
        with pytest.raises(ValueError):
            after[key].flags.writeable = True
    args[0]["raw_q"][4] = 999
    assert after["prior_targets"][4].tobytes() == before["prior_targets"][4].tobytes()


def test_missing_later_query_provenance_is_rejected_even_when_window_unselected():
    flat, identities, selections = fixture([65, *([1] * 53)])
    row = next(step for step in range(4, 65, 4) if step not in selections[0]["start_offsets"])
    flat["label_mask"][row] = False
    flat["raw_q"][row] = 0
    with pytest.raises(ValueError, match="every scheduled query"):
        project_training(flat, identities, selections)
