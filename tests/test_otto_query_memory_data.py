"""Fabricated census, observation masking and independent loss arithmetic."""
from itertools import pairwise

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory_data as data


def census(lengths=(41, 9, 1), stage="train"):
    total = sum(lengths)
    features = np.linspace(-.5, .5, total * 31, dtype=np.float32).reshape(total, 31)
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    correction = np.zeros(total, np.bool_)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        features[low:high, 15] = steps / 2188
        features[low:high, 16] = (steps % 4) / 2188
        features[low:high, 17] = 1
        correction[low:high] = steps % 4 == 0
    scores = np.arange(total * 4, dtype=np.float32).reshape(total, 4) / 8
    flat = {"features": features, "raw_q": scores, "legal": np.ones((total, 4), np.bool_),
            "actions": np.zeros(total, np.int64), "correction": correction, "episode_offsets": offsets}
    identities = [{"stage": stage, "episode_id": f"{stage}:fabricated:{i}", "episode_index": i + 54,
                   "seed": 100 + i, "case": i, "regime": "lambda3", "arm": "analytic"}
                  for i in range(len(lengths))]
    return flat, identities


@pytest.mark.parametrize("period", (4, 8))
@pytest.mark.parametrize("stage", ("train", "dev", "test"))
def test_projection_owns_bytes_changes_only_clock_and_keeps_complete_denominators(period, stage):
    flat, identities = census(stage=stage)
    originals = {k: v.tobytes() for k, v in flat.items()}
    projected = data.project_census(flat, identities, query_period=period, expected_stage=stage)
    assert projected["episode_count"] == 3
    assert np.isnan(projected["query_scores"][~projected["query_mask"]]).all()
    assert projected["query_scores"][projected["query_mask"]].tobytes() == flat["raw_q"][projected["query_mask"]].tobytes()
    assert projected["features"][:, :16].tobytes() == flat["features"][:, :16].tobytes()
    assert projected["features"][:, 17:].tobytes() == flat["features"][:, 17:].tobytes()
    assert sum(projected["nonquery_weights"]) == pytest.approx(2 / 3)
    assert sum(projected["prior_weights"]) == pytest.approx(2 / 3)
    assert not projected["nonquery_weights"][-1] and not projected["prior_weights"][-1]
    for value in projected.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flags.writeable = True
    assert all(flat[k].tobytes() == value for k, value in originals.items())
    flat["raw_q"][:] = -1
    assert not (projected["targets"] == -1).any()


def test_period8_hides_step4_score_from_inputs_but_retains_it_as_target():
    flat, identities = census((9,))
    first = data.project_census(flat, identities, query_period=8, expected_stage="train")
    flat["raw_q"][4] += 100
    second = data.project_census(flat, identities, query_period=8, expected_stage="train")
    assert first["query_scores"].tobytes() == second["query_scores"].tobytes()
    assert first["features"].tobytes() == second["features"].tobytes()
    assert not first["query_mask"][4] and not first["prior_mask"][4]
    assert not np.array_equal(first["targets"][4], second["targets"][4])
    assert first["features"][4, 16] == np.float32(4 / 2188)


@pytest.mark.parametrize("period", (4, 8))
def test_chunks_cover_every_loss_row_and_pad_ended_lanes_without_query_leakage(period):
    flat, identities = census()
    projected = data.project_census(flat, identities, query_period=period, expected_stage="train")
    first = data.batch_chunk(projected, [0, 1, 2], 0)
    second = data.batch_chunk(projected, [0, 1, 2], 32)
    assert first["model_inputs"]["lengths"].tolist() == [32, 9, 1]
    assert first["model_inputs"]["episode_ends"].tolist() == [False, True, True]
    assert second["model_inputs"]["lengths"].tolist() == [9, 0, 0]
    assert second["model_inputs"]["episode_ends"].tolist() == [True, False, False]
    assert set(first["model_inputs"]) == {"features", "query_scores", "query_mask", "lengths", "episode_ends"}
    for chunk in (first, second):
        assert np.isnan(chunk["model_inputs"]["query_scores"][~chunk["model_inputs"]["query_mask"]]).all()
        assert not np.shares_memory(chunk["targets"], projected["targets"])
    assert first["nonquery_weights"].sum() + second["nonquery_weights"].sum() == pytest.approx(2 / 3)
    assert first["prior_weights"].sum() + second["prior_weights"].sum() == pytest.approx(2 / 3)


@pytest.mark.parametrize("defect", ("stage", "duplicate_id", "index", "age", "correction", "missing", "illegal", "nan"))
def test_malformed_census_fails_before_projection(defect):
    flat, identities = census()
    if defect == "stage":
        identities[0]["stage"] = "test"
    elif defect == "duplicate_id":
        identities[1]["episode_id"] = identities[0]["episode_id"]
    elif defect == "index":
        identities[1]["episode_index"] += 1
    elif defect == "age":
        flat["features"][4, 16] = 4 / 2188
    elif defect == "correction":
        flat["correction"][4] = False
    elif defect == "missing":
        flat.pop("raw_q")
    elif defect == "illegal":
        flat["legal"][0, 0] = False
    else:
        flat["raw_q"][4] = np.nan
    with pytest.raises(ValueError):
        data.project_census(flat, identities, query_period=8, expected_stage="train")


def test_loss_matches_independent_scalar_reference_with_fixed_denominator_and_no_target_gradients():
    prediction = torch.tensor([[[5., 1., 3., 9.], [6., 7., 8., 9.]]], requires_grad=True)
    prior = torch.tensor([[[float("nan")] * 4, [5., 3., 2., 7.]]], requires_grad=True)
    targets = torch.tensor([[[2., 4., 1., 8.], [2., 3., 8., 5.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False, True], [True] * 4]])
    weights = torch.tensor([[1 / 54, 0.]], dtype=torch.float64, requires_grad=True)
    prior_weights = torch.tensor([[0., 1 / 54]], dtype=torch.float64, requires_grad=True)
    query = torch.tensor([[False, True]])
    result = data.weighted_loss(prediction, prior, targets, legal, weights, prior_weights,
                                query, query, episode_count=54)
    def scalar(pred, target, actions):
        left = [float(pred[i]) / 64 for i in actions]
        right = [float(target[i]) / 64 for i in actions]
        lmean, rmean = sum(left) / len(actions), sum(right) / len(actions)
        return sum(((a - lmean) - (b - rmean))**2 for a, b in zip(left, right, strict=True)) / len(actions)
    expected = scalar(prediction.detach()[0, 0], targets.detach()[0, 0], [0, 1, 3])
    expected += scalar(prior.detach()[0, 1], targets.detach()[0, 1], [0, 1, 2, 3])
    assert result["total"].item() == pytest.approx(expected, rel=2e-6)
    result["total"].backward()
    assert prediction.grad[0, 0].abs().sum() > 0 and not prediction.grad[0, 1].any()
    assert prior.grad[0, 1].abs().sum() > 0 and not prior.grad[0, 0].any()
    assert targets.grad is weights.grad is prior_weights.grad is None
