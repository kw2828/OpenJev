"""Pure chronological inputs, with query observations separate from targets.

The caller authenticates complete collection artifacts before projection. Only
period-four teacher vectors enter the model. Selected nonquery teacher vectors
enter the loss, never the returned model-input dictionary. No file or model IO.
"""
from __future__ import annotations

import numpy as np

from openjev.research.otto_sampled_forecast_data import EPISODES, build_sampled_windows
from openjev.research.otto_score_forecast_data import HORIZON, array, owned, require

VERSION = "otto-cross-query-data-v1"
CHUNK = 32


def build_training(episodes, selections, query_scores, selected_scores):
    """All 54 complete public episodes, every query, only selected loss targets.

    ``query_scores[e]`` has shape [ceil(T/4),4]. ``selected_scores[e]`` contains
    only the sampled windows, as in the qualified sampled helper. The same
    anchor appearing in both mappings must agree exactly. Returned arrays are
    immutable and own their bytes. Every unselected target slot is positive zero.
    """
    sampled = build_sampled_windows(episodes, selections, selected_scores)
    require(isinstance(query_scores, (list, tuple)) and len(query_scores) == EPISODES,
            "every episode has complete query inputs")
    lengths = sampled["episode_lengths"]
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
    total = int(offsets[-1])
    features = np.concatenate([e["features"] for e in episodes])
    legal = np.concatenate([e["legal"] for e in episodes])
    actions = np.concatenate([e["actions"] for e in episodes])
    queries = np.zeros((total, 4), dtype=np.float32)
    targets = np.zeros_like(queries)
    query_mask = np.zeros(total, dtype=np.bool_)
    weights = np.zeros(total, dtype=np.float64)
    for index, (episode, selection, scores, query) in enumerate(
            zip(episodes, selections, selected_scores, query_scores, strict=True)):
        length = len(episode["features"])
        count = (length + 3) // 4
        array(query, np.float32, (count, 4), "complete true-query scores", finite=True)
        require(np.array_equal(query / np.float32(64) * np.float32(64), query),
                "query scores preserve scale roundtrip")
        low = int(offsets[index])
        local_queries = np.arange(0, length, 4)
        queries[low + local_queries] = query
        query_mask[low + local_queries] = True
        nonquery_count = length - count
        weight = (count / selection["selected_windows"]) / (EPISODES * nonquery_count) if nonquery_count else 0.
        for start in selection["start_offsets"]:
            score = scores[start]
            require(score[0].tobytes() == query[start // 4].tobytes(), "sampled anchor equals exact query input")
            stop = min(length, start + 4)
            targets[low + start + 1:low + stop] = score[1:]
            weights[low + start + 1:low + stop] = weight
    require(bool((targets[weights == 0] == 0).all()), "targets only at selected nonquery rows")
    arrays = {"features": features, "query_scores": queries, "targets": targets,
              "legal": legal, "actions": actions, "query_mask": query_mask,
              "weights": weights, "episode_offsets": offsets}
    return {"version": VERSION, **{k: owned(v) for k, v in arrays.items()},
            "episode_ids": sampled["episode_ids"], "episode_regimes": sampled["episode_regimes"],
            "episode_splits": sampled["episode_splits"], "sampling": sampled["sampling"],
            "counts": {"episodes": EPISODES, "rows": total, "query_rows": int(query_mask.sum()),
                       "selected_nonquery_rows": int((weights > 0).sum()),
                       "selected_windows": sampled["counts"]["windows"],
                       "zero_support_episodes": sampled["counts"]["zero_support_episodes"]}}


def project_training(flat, identities, selections, *, selection_start):
    """Project authenticated saved arrays before building any model inputs.

    No unselected nonquery label is read numerically. Missing scores must have
    the collector's exact positive-zero representation. All queries and all
    selected labels require positive provenance via ``label_mask``.
    """
    require(type(selection_start) is int and 0 <= selection_start < 2**32 - EPISODES,
            "declared selection seed start")
    require(set(flat) == {"features", "raw_q", "legal", "actions", "correction", "episode_offsets", "label_mask"},
            "exact seven TRAIN arrays")
    require(len(identities) == len(selections) == EPISODES, "all complete TRAIN identities")
    offsets = array(flat["episode_offsets"], np.int64, (EPISODES + 1,), "episode offsets")
    require(offsets[0] == 0 and bool(((np.diff(offsets) >= 1) & (np.diff(offsets) <= HORIZON)).all()),
            "nonempty complete episode geometry")
    total = int(offsets[-1])
    for name, dtype, shape in (("features", np.float32, (total, 31)), ("raw_q", np.float32, (total, 4)),
                               ("legal", np.bool_, (total, 4)), ("actions", np.int64, (total,)),
                               ("correction", np.bool_, (total,)), ("label_mask", np.bool_, (total,))):
        array(flat[name], dtype, shape, name)
    missing = flat["raw_q"][~flat["label_mask"]]
    require(missing.tobytes() == np.zeros(missing.shape, np.float32).tobytes(), "exact unscored placeholders")
    episodes, queries, scores = [], [], []
    for index, (identity, selection) in enumerate(zip(identities, selections, strict=True)):
        low, high = int(offsets[index]), int(offsets[index + 1])
        length = high - low
        require(identity["stage"] == "train" and identity["episode_index"] == index
                and selection["seed"] == selection_start + index
                and selection["episode_id"] == identity["episode_id"], "declared episode and seed order")
        mask = np.arange(length) % 4 == 0
        require(np.array_equal(flat["correction"][low:high], mask)
                and bool(flat["label_mask"][low:high][mask].all()), "every scheduled query has a returned score")
        episodes.append({"id": identity["episode_id"], "regime": identity["regime"], "split": "train",
                         "features": flat["features"][low:high], "legal": flat["legal"][low:high],
                         "actions": flat["actions"][low:high]})
        queries.append(flat["raw_q"][low:high][mask].copy())
        selected = {}
        for start in selection["start_offsets"]:
            stop = min(start + 4, length)
            require(bool(flat["label_mask"][low + start:low + stop].all()), "every selected target has a returned score")
            selected[start] = flat["raw_q"][low + start:low + stop].copy()
        scores.append(selected)
    return build_training(episodes, selections, queries, scores)


def batch_chunk(data, indices, start, *, span=CHUNK):
    """Pad one chronological chunk without exposing targets to the model.

    The caller carries hidden state between successive chunks. A lane ends in
    exactly the chunk that consumes its final row; later chunks have length zero
    and do not signal a second end. Numerical padding and skipped query scores
    contain NaN poison; separate targets and weights have zero padding.
    """
    require(data["version"] == VERSION, "chronological data version")
    require(type(start) is int and start >= 0 and start % CHUNK == 0 and span == CHUNK,
            "fixed 32-step chunk boundaries")
    require(isinstance(indices, (tuple, list)) and bool(indices)
            and all(type(i) is int and 0 <= i < len(data["episode_ids"]) for i in indices)
            and len(set(indices)) == len(indices), "distinct episode indices")
    batch = len(indices)
    features = np.full((batch, span, 31), np.nan, dtype=np.float32)
    query = np.full((batch, span, 4), np.nan, dtype=np.float32)
    targets = np.zeros((batch, span, 4), dtype=np.float32)
    weights = np.zeros((batch, span), dtype=np.float64)
    legal = np.zeros((batch, span, 4), dtype=np.bool_)
    mask = np.zeros((batch, span), dtype=np.bool_)
    lengths = np.zeros(batch, dtype=np.int64)
    ends = np.zeros(batch, dtype=np.bool_)
    for lane, index in enumerate(indices):
        low, high = (int(v) for v in data["episode_offsets"][index:index + 2])
        total = high - low
        length = max(0, min(span, total - start))
        lengths[lane] = length
        ends[lane] = length > 0 and start + length == total
        if not length:
            continue
        rows = slice(low + start, low + start + length)
        features[lane, :length] = data["features"][rows]
        targets[lane, :length] = data["targets"][rows]
        weights[lane, :length] = data["weights"][rows]
        legal[lane, :length] = data["legal"][rows]
        mask[lane, :length] = data["query_mask"][rows]
        local = np.flatnonzero(mask[lane])
        query[lane, local] = data["query_scores"][rows][local]
    return {"model_inputs": {"features": features, "query_scores": query, "lengths": lengths,
                              "query_mask": mask, "episode_ends": ends},
            "targets": targets, "legal": legal, "weights": weights}
