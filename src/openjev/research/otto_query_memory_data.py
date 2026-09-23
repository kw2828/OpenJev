"""Census-only chronological data for explicit period-four/eight observations.

The caller authenticates collection and split provenance before passing arrays.
The collector's period-four behavior stays unchanged. Only the observation-age
feature is rewritten for a different forecasting schedule. Query inputs and
loss targets are separate owned arrays. This module performs no file/model IO.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np

from openjev.research.otto_score_forecast_data import HORIZON, array, owned, require

VERSION = "otto-query-memory-data-v1"
CHUNK = 32
PERIODS = (4, 8)
STAGES = ("train", "dev", "test")
ARRAY_KEYS = {"features", "raw_q", "legal", "actions", "correction", "episode_offsets"}


def project_census(flat, identities, *, query_period, expected_stage):
    """Project a complete authenticated split, retaining unsupported episodes.

    Weights are 1/(number of episodes * eligible rows within this episode).
    Nonquery rows and later-query priors have separate episode denominators.
    There is no sampled-window or realized-support renormalization.
    """
    require(type(query_period) is int and query_period in PERIODS, "explicit supported query period")
    require(expected_stage in STAGES and isinstance(flat, dict) and set(flat) == ARRAY_KEYS,
            "declared split and exact census arrays")
    require(isinstance(identities, (list, tuple)) and bool(identities), "nonempty split identities")
    count = len(identities)
    fields = {"stage", "episode_id", "episode_index", "seed", "case", "regime", "arm"}
    require(all(isinstance(row, dict) and fields <= row.keys() for row in identities), "complete identities")
    require(all(row["stage"] == expected_stage and type(row["episode_id"]) is str and row["episode_id"]
                and row["regime"] in ("lambda3", "lambda4")
                and row["arm"] in ("analytic", "neural", "period4_hold")
                and all(type(row[k]) is int and row[k] >= 0 for k in ("episode_index", "seed", "case"))
                for row in identities), "valid identities in one declared split")
    require(len({row["episode_id"] for row in identities}) == count, "unique episode IDs")
    require([row["episode_index"] for row in identities] == list(range(
        identities[0]["episode_index"], identities[0]["episode_index"] + count)), "ordered complete split indices")
    offsets = array(flat["episode_offsets"], np.int64, (count + 1,), "census offsets")
    lengths = np.diff(offsets)
    require(offsets[0] == 0 and bool(((lengths >= 1) & (lengths <= HORIZON)).all()), "complete episode geometry")
    total = int(offsets[-1])
    for key, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                              ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                              ("correction", np.bool_, (total,))):
        array(flat[key], dtype, shape, key, finite=key in ("features", "raw_q"))
    require(bool(flat["legal"].any(-1).all()) and bool(((flat["actions"] >= 0) & (flat["actions"] < 4)).all()),
            "legal support and bounded actions")
    require(bool(flat["legal"][np.arange(total), flat["actions"]].all()), "actual collector action is legal")
    features = flat["features"].copy()
    query_mask = np.zeros(total, np.bool_)
    prior_mask = np.zeros(total, np.bool_)
    query_scores = np.full((total, 4), np.nan, np.float32)
    nonquery_weights = np.zeros(total, np.float64)
    prior_weights = np.zeros(total, np.float64)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low, dtype=np.int64)
        require(np.array_equal(features[low:high, 15], (steps / HORIZON).astype(np.float32))
                and np.array_equal(features[low:high, 16], ((steps % 4) / HORIZON).astype(np.float32))
                and bool((features[low:high, 17] == 1).all()), "original collector chronology features")
        require(np.array_equal(flat["correction"][low:high], steps % 4 == 0), "original period-four collector schedule")
        features[low:high, 16] = ((steps % query_period) / HORIZON).astype(np.float32)
        local_query = steps % query_period == 0
        local_prior = local_query & (steps > 0)
        local_nonquery = ~local_query
        query_mask[low:high] = local_query
        prior_mask[low:high] = local_prior
        rows = low + steps[local_query]
        query_scores[rows] = flat["raw_q"][rows]
        if bool(local_nonquery.any()):
            nonquery_weights[low + steps[local_nonquery]] = 1 / (count * int(local_nonquery.sum()))
        if bool(local_prior.any()):
            prior_weights[low + steps[local_prior]] = 1 / (count * int(local_prior.sum()))
    visible = query_scores[query_mask]
    require(np.array_equal(visible / np.float32(64) * np.float32(64), visible), "exact query scale roundtrip")
    arrays = {"features": features, "query_scores": query_scores, "targets": flat["raw_q"],
              "legal": flat["legal"], "actions": flat["actions"], "episode_offsets": offsets,
              "query_mask": query_mask, "prior_mask": prior_mask,
              "nonquery_weights": nonquery_weights, "prior_weights": prior_weights}
    return {"version": VERSION, "query_period": query_period, "stage": expected_stage,
            "episode_ids": tuple(row["episode_id"] for row in identities),
            "identities": tuple(tuple((name, row[name]) for name in sorted(fields)) for row in identities),
            "episode_count": count, **{name: owned(value) for name, value in arrays.items()}}


def batch_chunk(data, indices, start, *, span=CHUNK):
    """Copy a chronological chunk; poison only unconsumed model-input slots."""
    require(isinstance(data, dict) and data.get("version") == VERSION, "census data version")
    require(type(start) is int and start >= 0 and start % CHUNK == 0 and span == CHUNK,
            "fixed 32-step chunk boundaries")
    require(isinstance(indices, (list, tuple)) and 1 <= len(indices) <= 6
            and all(type(i) is int and 0 <= i < data["episode_count"] for i in indices)
            and len(set(indices)) == len(indices), "up to six distinct episode indices")
    batch = len(indices)
    features = np.full((batch, span, 31), np.nan, np.float32)
    scores = np.full((batch, span, 4), np.nan, np.float32)
    targets = np.zeros((batch, span, 4), np.float32)
    legal = np.zeros((batch, span, 4), np.bool_)
    query_mask = np.zeros((batch, span), np.bool_)
    prior_mask = np.zeros_like(query_mask)
    weights = np.zeros((batch, span), np.float64)
    prior_weights = np.zeros_like(weights)
    lengths = np.zeros(batch, np.int64)
    ends = np.zeros(batch, np.bool_)
    for lane, index in enumerate(indices):
        low, high = map(int, data["episode_offsets"][index:index + 2])
        length = max(0, min(span, high - low - start))
        lengths[lane] = length
        ends[lane] = length > 0 and start + length == high - low
        if not length:
            continue
        rows = slice(low + start, low + start + length)
        for destination, key in ((features, "features"), (scores, "query_scores"), (targets, "targets"),
                                 (legal, "legal"), (query_mask, "query_mask"), (prior_mask, "prior_mask"),
                                 (weights, "nonquery_weights"), (prior_weights, "prior_weights")):
            destination[lane, :length] = data[key][rows]
    return {"model_inputs": {"features": features, "query_scores": scores, "query_mask": query_mask,
                              "lengths": lengths, "episode_ends": ends},
            "targets": targets, "legal": legal, "nonquery_weights": weights,
            "prior_weights": prior_weights, "prior_mask": prior_mask,
            "episode_count": data["episode_count"]}


def weighted_loss(prediction, corrected_prior, targets, legal, nonquery_weights,
                  prior_weights, query_mask, prior_mask, *, episode_count):
    """Full-forecast AUX: eligible nonquery MSE plus all-four prewrite prior MSE.

    Both terms use /64 score units and coefficient one. Select supported rows
    before arithmetic, detach targets/weights, and retain the fixed full episode
    denominator through episode_count / batch_size. No optimizer calls occur.
    """
    import torch

    from openjev.research.otto_prequery_loss import nonquery_rows, prior_rows

    require(isinstance(prediction, torch.Tensor) and prediction.ndim == 3
            and prediction.shape[-1] == 4 and 1 <= prediction.shape[0] <= 6,
            "one to six episode lanes with four scores")
    batch, span, _ = prediction.shape
    require(span > 0 and type(episode_count) is int and episode_count >= batch, "fixed episode denominator")
    for name, value in (("prediction", prediction), ("corrected_prior", corrected_prior), ("targets", targets)):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.float32 and value.shape == prediction.shape,
                "matching CPU float32 " + name)
    require(isinstance(legal, torch.Tensor) and legal.device.type == "cpu"
            and legal.dtype == torch.bool and legal.shape == prediction.shape, "matching legal masks")
    for name, value in (("query", query_mask), ("prior", prior_mask)):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.bool and value.shape == (batch, span), "matching " + name + " mask")
    for value in (nonquery_weights, prior_weights):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype in (torch.float32, torch.float64) and value.shape == (batch, span)
                and bool(torch.isfinite(value).all()) and bool((value >= 0).all()), "finite nonnegative weights")
    nonquery = nonquery_weights > 0
    require(not bool((nonquery & query_mask).any()) and not bool((prior_mask & ~query_mask).any())
            and torch.equal(prior_weights > 0, prior_mask), "disjoint actual nonquery/prior support")
    first = prediction.new_zeros(())
    second = prediction.new_zeros(())
    if bool(nonquery.any()):
        require(bool(legal[nonquery].any(-1).all()), "positive nonquery weight requires legal actions")
        losses = nonquery_rows(prediction[nonquery][:, None], targets[nonquery].detach()[:, None],
                              legal[nonquery][:, None])[:, 0]
        first = (losses * nonquery_weights[nonquery].detach().to(torch.float32)).sum()
    if bool(prior_mask.any()):
        losses = prior_rows(corrected_prior[prior_mask][:, None], targets[prior_mask].detach()[:, None])[:, 0]
        second = (losses * prior_weights[prior_mask].detach().to(torch.float32)).sum()
    multiplier = episode_count / batch
    first, second = first * multiplier, second * multiplier
    require(bool(torch.isfinite(first + second)), "finite full-forecast AUX loss")
    return {"total": first + second, "nonquery": first, "prior": second}
