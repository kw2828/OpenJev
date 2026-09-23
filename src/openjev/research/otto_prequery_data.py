"""Add later-query supervision to unchanged complete chronological TRAIN inputs.

Only the qualified base projector reads supplied collection arrays. The extra
targets reuse its already authenticated query vectors, never incidental or
unselected nonquery scores. Targets are separate from the unchanged model-input
dictionary. No files, models, environments or additional sampling are used.
"""
from __future__ import annotations

import numpy as np

from openjev.research import otto_cross_query_data as original
from openjev.research.otto_sampled_forecast_data import EPISODES
from openjev.research.otto_score_forecast_data import owned, require

VERSION = "otto-prequery-data-v1"
SELECTION_START = 276000001
CHUNK = original.CHUNK


def project_training(flat, identities, selections, *, selection_start=SELECTION_START):
    """Retain all 54 episodes and give each supported episode prior mass 1/54.

For a complete episode of T rows, K=ceil(T/4)-1. Rows 4,8,... receive
the actual four-action query vector and weight 1/(54*K). The first query,
nonqueries and K=0 episodes receive zero prior targets and weight. There is no
renormalization after removing zero-support episodes, nor a legal-action mask
on this additional supervision. The original nonquery targets remain unchanged.
    """
    data = original.project_training(flat, identities, selections, selection_start=selection_start)
    targets = np.zeros_like(data["query_scores"])
    weights = np.zeros(len(targets), dtype=np.float64)
    mask = np.zeros(len(targets), dtype=np.bool_)
    supported = 0
    for index in range(EPISODES):
        low, high = (int(v) for v in data["episode_offsets"][index:index + 2])
        rows = np.arange(low + 4, high, 4)
        count = len(rows)
        require(count == (high - low + 3) // 4 - 1, "complete later-query geometry")
        if not count:
            continue
        require(bool(data["query_mask"][rows].all()), "later targets are existing true queries")
        targets[rows] = data["query_scores"][rows]
        weights[rows] = 1.0 / (EPISODES * count)
        mask[rows] = True
        supported += 1
    return {**data, "version": VERSION, "prior_targets": owned(targets),
            "prior_weights": owned(weights), "prior_mask": owned(mask),
            "counts": {**data["counts"], "prior_rows": int(mask.sum()),
                       "prior_supported_episodes": supported,
                       "prior_zero_support_episodes": EPISODES - supported,
                       "prior_weight_mass": float(weights.sum())}}


def batch_chunk(data, indices, start, *, span=CHUNK):
    """Delegate original inputs exactly; pad only separate prior arrays with zero.

The three additional arrays are loss inputs, not entries in model_inputs.
The old immutable arrays, episode identity and end/reset semantics are retained.
    """
    require(data["version"] == VERSION, "prequery data version")
    packet = original.batch_chunk({**data, "version": original.VERSION}, indices, start, span=span)
    targets = np.zeros((len(indices), span, 4), dtype=np.float32)
    weights = np.zeros((len(indices), span), dtype=np.float64)
    mask = np.zeros((len(indices), span), dtype=np.bool_)
    for lane, index in enumerate(indices):
        low = int(data["episode_offsets"][index])
        length = int(packet["model_inputs"]["lengths"][lane])
        if length:
            rows = slice(low + start, low + start + length)
            targets[lane, :length] = data["prior_targets"][rows]
            weights[lane, :length] = data["prior_weights"][rows]
            mask[lane, :length] = data["prior_mask"][rows]
    return {**packet, "prior_targets": targets, "prior_weights": weights, "prior_mask": mask}
