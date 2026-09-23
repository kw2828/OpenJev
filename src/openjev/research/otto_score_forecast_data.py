"""Pure period-four saved-score windows and descriptive teacher-score metrics.

No environment, model, file access or randomness. Complete episodes are supplied
by the caller, who authenticates their provenance and admits the train/valid
split. The only teacher scores exposed by ``model_inputs`` are each window's
first queried vector. Subsequent teacher vectors remain separate labels.

Metrics describe imitation of the supplied teacher, not actual search returns.
Every metric group uses equal episode weights, including zero-support episodes;
within an episode its scored rows receive equal weights. Regime and age groups
rebuild these weights, rather than inheriting the overall group's attenuation.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "otto-score-forecast-data-v1"
PERIOD = 4
INPUT_DIM = 31
HORIZON = 2188
EPS = np.float32(1e-10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def owned(array):
    """Detached bytes-backed arrays cannot have writes re-enabled by callers."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def array(value, dtype, shape, name, *, finite=False):
    require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape,
            f"exact {name} dtype/shape")
    if finite:
        require(bool(np.isfinite(value).all()), f"finite {name}")
    return value


def _episode(episode):
    required = {"id", "regime", "features", "teacher_scores", "legal"}
    require(isinstance(episode, dict) and required <= episode.keys()
            and set(episode) <= required | {"split"}, "exact episode fields")
    for key in ("id", "regime"):
        require(type(episode[key]) is str and bool(episode[key]), f"nonempty episode {key}")
    require("split" not in episode or (type(episode["split"]) is str and bool(episode["split"])),
            "optional nonempty split identity")
    features = episode["features"]
    require(isinstance(features, np.ndarray) and features.ndim == 2, "episode feature matrix")
    rows = len(features)
    require(1 <= rows <= HORIZON, "complete episode length within horizon")
    array(features, np.float32, (rows, INPUT_DIM), "episode features", finite=True)
    array(episode["teacher_scores"], np.float32, (rows, PERIOD), "teacher scores", finite=True)
    legal = array(episode["legal"], np.bool_, (rows, PERIOD), "legal actions")
    require(bool(legal.any(axis=1).all()), "at least one legal action at every row")
    steps = np.arange(rows, dtype=np.int64)
    require(np.array_equal(features[:, 15], (steps / HORIZON).astype(np.float32)),
            "episode chronology begins at zero")
    require(np.array_equal(features[:, 16], ((steps % PERIOD) / HORIZON).astype(np.float32)),
            "virtual period-four query age")
    require(bool((features[:, 17] == np.float32(1)).all()), "query anchor available at every row")
    return rows


def _row_weights(mask, episode_index, episode_count):
    counts = np.zeros(episode_count, dtype=np.int64)
    for row, index in zip(mask, episode_index, strict=True):
        counts[index] += int(row.sum())
    weights = np.zeros(mask.shape, dtype=np.float64)
    for index, row in enumerate(mask):
        count = int(counts[episode_index[index]])
        if count:
            weights[index, row] = 1.0 / (episode_count * count)
    return weights, counts


def build_windows(episodes):
    """Keep every disjoint [0:4], [4:8], ... tail in caller episode order.

    Features are supplied public values with step/2188, virtual age/2188 and
    has-queried=1 at columns 15/16/17. They are checked, never rewritten. Real
    teacher values retain their exact float32 bytes, including signed zeros.
    Padding is zero with false legal/valid masks. Query-only windows survive.
    """
    require(isinstance(episodes, (list, tuple)) and bool(episodes), "nonempty episode collection")
    episode_lengths = tuple(_episode(episode) for episode in episodes)
    ids = tuple(episode["id"] for episode in episodes)
    require(len(set(ids)) == len(ids), "unique episode identities across splits")
    count = sum((length + PERIOD - 1) // PERIOD for length in episode_lengths)
    features = np.zeros((count, PERIOD, INPUT_DIM), dtype=np.float32)
    query = np.zeros((count, PERIOD), dtype=np.float32)
    targets = np.zeros((count, PERIOD, PERIOD), dtype=np.float32)
    legal = np.zeros((count, PERIOD, PERIOD), dtype=np.bool_)
    valid = np.zeros((count, PERIOD), dtype=np.bool_)
    lengths = np.empty(count, dtype=np.int64)
    indices = np.empty(count, dtype=np.int64)
    offsets = np.empty(count, dtype=np.int64)
    window = 0
    for index, (episode, total) in enumerate(zip(episodes, episode_lengths, strict=True)):
        for start in range(0, total, PERIOD):
            length = min(PERIOD, total - start)
            stop = start + length
            features[window, :length] = episode["features"][start:stop]
            query[window] = episode["teacher_scores"][start]
            targets[window, :length] = episode["teacher_scores"][start:stop]
            legal[window, :length] = episode["legal"][start:stop]
            valid[window, :length] = True
            lengths[window], indices[window], offsets[window] = length, index, start
            window += 1
    nonquery = valid.copy()
    nonquery[:, 0] = False
    weights, counts = _row_weights(nonquery, indices, len(episodes))
    arrays = {"features": features, "query_scores": query, "targets": targets, "legal": legal,
              "valid_mask": valid, "nonquery_mask": nonquery, "lengths": lengths,
              "episode_index": indices, "step_offsets": offsets, "nonquery_weights": weights,
              "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
              "episode_nonquery_counts": counts}
    return {"version": VERSION, **{name: owned(value) for name, value in arrays.items()},
            "episode_ids": ids, "episode_regimes": tuple(episode["regime"] for episode in episodes),
            "episode_splits": tuple(episode.get("split") for episode in episodes),
            "counts": {"episodes": len(episodes), "windows": count, "rows": sum(episode_lengths),
                       "query_rows": count, "nonquery_rows": int(counts.sum()),
                       "zero_support_episodes": int((counts == 0).sum()),
                       "query_only_windows": int((lengths == 1).sum())}}


def model_inputs(windows):
    """Return only the three accepted model inputs; no labels or legal masks."""
    require(isinstance(windows, dict) and windows.get("version") == VERSION, "forecast window version")
    return {name: windows[name] for name in ("features", "query_scores", "lengths")}


def _metric_inputs(windows, predictions):
    require(isinstance(windows, dict) and windows.get("version") == VERSION, "forecast window version")
    ids = windows["episode_ids"]
    regimes = windows["episode_regimes"]
    require(isinstance(ids, tuple) and bool(ids) and all(type(v) is str and v for v in ids)
            and len(set(ids)) == len(ids), "unique metric episode identities")
    require(isinstance(regimes, tuple) and len(regimes) == len(ids)
            and all(type(v) is str and v for v in regimes), "metric regime identities")
    lengths = windows["lengths"]
    require(isinstance(lengths, np.ndarray) and lengths.ndim == 1, "window lengths vector")
    n = len(lengths)
    array(lengths, np.int64, (n,), "window lengths")
    require(n > 0 and bool(((lengths >= 1) & (lengths <= PERIOD)).all()), "window length bounds")
    valid = array(windows["valid_mask"], np.bool_, (n, PERIOD), "valid mask")
    nonquery = array(windows["nonquery_mask"], np.bool_, (n, PERIOD), "nonquery mask")
    expected = np.arange(PERIOD)[None, :] < lengths[:, None]
    require(np.array_equal(valid, expected) and np.array_equal(nonquery, expected & (np.arange(PERIOD) > 0)),
            "exact active-prefix and nonquery masks")
    index = array(windows["episode_index"], np.int64, (n,), "episode indices")
    require(bool(((index >= 0) & (index < len(ids))).all()) and set(index.tolist()) == set(range(len(ids))),
            "every declared episode has windows")
    offsets = array(windows["step_offsets"], np.int64, (n,), "step offsets")
    episode_lengths = array(windows["episode_lengths"], np.int64, (len(ids),), "episode lengths")
    require(bool(((episode_lengths >= 1) & (episode_lengths <= HORIZON)).all()), "metric episode horizon")
    expected_index, expected_offsets, expected_lengths = [], [], []
    for i, total in enumerate(episode_lengths):
        for start in range(0, int(total), PERIOD):
            expected_index.append(i)
            expected_offsets.append(start)
            expected_lengths.append(min(PERIOD, int(total) - start))
    require(index.tolist() == expected_index and offsets.tolist() == expected_offsets
            and lengths.tolist() == expected_lengths, "complete disjoint episode windows including tails")
    teacher = array(windows["targets"], np.float32, (n, PERIOD, PERIOD), "window targets", finite=True)
    legal = array(windows["legal"], np.bool_, (n, PERIOD, PERIOD), "window legal mask")
    require(bool(legal[valid].any(axis=1).all()) and not bool(legal[~valid].any()), "legal active rows only")
    array(predictions, np.float32, (n, PERIOD, PERIOD), "predicted scores", finite=True)
    return ids, regimes, index, nonquery, teacher, legal


def _near_minimum(scores, legal):
    eligible = tuple(int(action) for action in np.flatnonzero(legal))
    minimum = np.min(scores[list(eligible)])
    # Overflow for separated finite f32 extremes means 'not near', never a tie.
    with np.errstate(over="ignore"):
        near = tuple(action for action in eligible if np.float32(scores[action] - minimum) < EPS)
    return near, minimum


def forecast_metrics(windows, predictions):
    """Scalar, episode-balanced teacher agreement, raw gap and centered MSE.

    Choose the first legal predicted action within strict float32 1e-10 of its
    minimum. Agreement permits any action in the teacher's matching near-minimum
    set. Gap is the exact float64 difference of the original float32 teacher
    scores. For MSE, both vectors are centered over the current legal actions,
    then their squared differences are averaged over those actions. This metric
    uses raw score units; the caller's training-only /64 scaling is not applied.
    Query rows and padding have zero metric/loss support.

    ``episode_weighted_*`` uses the fixed declared episode denominator; absent
    support contributes zero, not an invented correct decision. The separately
    named ``supported_episode_*`` divides by weight_mass and is None at zero
    support. Each age summary rebalances within that age, so age metrics are not
    additive components of the overall metric. All regimes and ages are kept.
    """
    ids, regimes, indices, nonquery, teacher, legal = _metric_inputs(windows, predictions)
    rows = [[] for _ in ids]
    for window, age in zip(*np.nonzero(nonquery), strict=True):
        predicted_near, _ = _near_minimum(predictions[window, age], legal[window, age])
        teacher_near, teacher_minimum = _near_minimum(teacher[window, age], legal[window, age])
        action = predicted_near[0]
        gap = float(teacher[window, age, action]) - float(teacher_minimum)
        eligible = tuple(int(a) for a in np.flatnonzero(legal[window, age]))
        p = tuple(float(predictions[window, age, a]) for a in eligible)
        t = tuple(float(teacher[window, age, a]) for a in eligible)
        p_mean, t_mean = math.fsum(p) / len(p), math.fsum(t) / len(t)
        mse = math.fsum(((a - p_mean) - (b - t_mean)) ** 2 for a, b in zip(p, t, strict=True)) / len(p)
        rows[int(indices[window])].append((int(age), action in teacher_near, gap,
                                          action == teacher_near[0], mse))

    def summarize(episode_indices, age=None):
        agreement, gaps, first, errors = [], [], [], []
        support, total_rows, zero_ids = 0, 0, []
        for index in episode_indices:
            selected = [row for row in rows[index] if age is None or row[0] == age]
            total_rows += len(selected)
            if not selected:
                zero_ids.append(ids[index])
                continue
            support += 1
            agreement.append(math.fsum(float(row[1]) for row in selected) / len(selected))
            gaps.append(math.fsum(row[2] for row in selected) / len(selected))
            first.append(math.fsum(float(row[3]) for row in selected) / len(selected))
            errors.append(math.fsum(row[4] for row in selected) / len(selected))
        count = len(episode_indices)
        a, g, f, error = math.fsum(agreement), math.fsum(gaps), math.fsum(first), math.fsum(errors)
        return {"episodes": count, "supported_episodes": support, "zero_support_episode_ids": zero_ids,
                "nonquery_rows": total_rows, "weight_mass": support / count,
                "episode_weighted_agreement": a / count, "episode_weighted_raw_gap": g / count,
                "episode_weighted_first_argmin_match": f / count,
                "episode_weighted_centered_mse": error / count,
                "supported_episode_agreement": a / support if support else None,
                "supported_episode_raw_gap": g / support if support else None,
                "supported_episode_centered_mse": error / support if support else None}

    def group(episode_indices):
        return {**summarize(episode_indices),
                "by_age": {str(age): summarize(episode_indices, age) for age in (1, 2, 3)}}

    return {"version": VERSION, "scope": "saved teacher-score imitation; no search-return claim",
            "overall": group(tuple(range(len(ids)))),
            "by_regime": {regime: group(tuple(i for i, value in enumerate(regimes) if value == regime))
                          for regime in dict.fromkeys(regimes)}}
