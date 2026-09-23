"""Pure, fixed once-per-episode TRAIN-window selection and importance weights.

All 54 complete TRAIN histories are supplied by the authenticating caller.
The seed is declared independently of observations, scores and fit seeds before
selection. Sample up to eight disjoint period-four windows without replacement;
truncated and query-only tails belong to that same population. They are never
forced into or excluded from a sample. Selection is shared by every fit.

Teacher scores are supplied ONLY for selected windows. Missing scores outside
the sample require no placeholder, reading, imputation or additional query.
VALID remains an unchanged full census through otto_score_forecast_data.
No files, environments, models, optimizer calls or global RNG state are used.
"""
from __future__ import annotations

import numpy as np

from openjev.research.otto_score_forecast_data import (
    HORIZON,
    INPUT_DIM,
    PERIOD,
    array,
    owned,
    require,
)
from openjev.research.otto_score_forecast_data import (
    VERSION as WINDOW_VERSION,
)

VERSION = "otto-sampled-forecast-data-v1"
EPISODES = 54
WINDOW_CAP = 8
ALGORITHM = "numpy.Generator(PCG64(seed)).choice(W,size=k,replace=False); sorted"


def select_windows(length, seed):
    """Return detached JSON metadata for a single fixed, uniform window sample.

The exact inclusion probability is selected_windows/population_windows. The
floating probability is descriptive; numerator and denominator are retained.
Repeat calls with the same declared seed verify one selection, not a new draw.
"""
    require(type(length) is int and 1 <= length <= HORIZON, "complete episode length within horizon")
    require(type(seed) is int and 0 <= seed < 2**32, "independent uint32 Python selection seed")
    population = (length + PERIOD - 1) // PERIOD
    count = min(WINDOW_CAP, population)
    generator = np.random.Generator(np.random.PCG64(seed))
    selected = sorted(int(i) * PERIOD for i in generator.choice(population, size=count, replace=False))
    return {"version": VERSION, "length": length, "seed": seed, "period": PERIOD,
            "population_windows": population, "selected_windows": count, "start_offsets": selected,
            "inclusion_numerator": count, "inclusion_denominator": population,
            "inclusion_probability": count / population, "algorithm": ALGORITHM}


def _episode(episode):
    require(isinstance(episode, dict) and set(episode) == {"id", "regime", "split", "features", "legal", "actions"},
            "exact complete public TRAIN episode fields; no unselected teacher scores")
    require(all(type(episode[k]) is str and episode[k] for k in ("id", "regime"))
            and episode["split"] == "train", "nonempty TRAIN-only episode identities")
    features = episode["features"]
    require(isinstance(features, np.ndarray) and features.ndim == 2, "full public feature matrix")
    length = len(features)
    require(1 <= length <= HORIZON, "complete episode length within horizon")
    array(features, np.float32, (length, INPUT_DIM), "full public features", finite=True)
    legal = array(episode["legal"], np.bool_, (length, PERIOD), "full legal actions")
    actions = array(episode["actions"], np.int64, (length,), "full actual actions")
    require(bool(legal.any(axis=1).all()) and bool(((actions >= 0) & (actions < PERIOD)).all())
            and bool(legal[np.arange(length), actions].all()), "all full-history actual actions are eligible")
    steps = np.arange(length, dtype=np.int64)
    require(np.array_equal(features[:, 15], (steps / HORIZON).astype(np.float32)),
            "full absolute episode chronology begins at zero")
    require(np.array_equal(features[:, 16], ((steps % PERIOD) / HORIZON).astype(np.float32))
            and bool((features[:, 17] == np.float32(1)).all()), "unchanged virtual correction features")
    return length


def build_sampled_windows(episodes, selections, selected_scores):
    """Build original-version model windows from all 54 TRAIN episodes.

``selections[e]`` is {episode_id, **select_windows(full_length, fixed_seed)}.
``selected_scores[e]`` maps each selected integer start offset to an exact
float32[min(4,T-start),4] array, including the query anchor at its first row.
There must be exactly those keys and no unscored placeholder rows. Callers
must additionally verify each supplied row's saved label_mask/provenance.

Every selected nonquery row receives (W/k)/(54*M), where W=ceil(T/4) and M=T-W
refer to the FULL episode, including unselected windows. Query and padding
weights are zero. No renormalization is performed, even when realized weight
mass differs from 1/54 or a sampled query-only tail contributes no support.

The original model_inputs accepts this window version. Full-census metric
functions deliberately do not: do not pass sampled TRAIN windows to them.
    Minibatch objective scaling uses the selected total number of windows divided
    by actual batch size; multiplying by the population count would double-correct.
    Selection expectation matches the full objective for fixed row losses, not
    a claim about fitted-model risk. Performance still requires census VALID.
"""
    require(isinstance(episodes, (list, tuple)) and len(episodes) == EPISODES,
            "exactly 54 complete TRAIN episodes, including zero-support episodes")
    require(isinstance(selections, (list, tuple)) and len(selections) == EPISODES
            and isinstance(selected_scores, (list, tuple)) and len(selected_scores) == EPISODES,
            "one selection and selected score mapping per TRAIN episode")
    episode_lengths = tuple(_episode(e) for e in episodes)
    ids = tuple(e["id"] for e in episodes)
    require(len(set(ids)) == EPISODES, "unique complete TRAIN episode identities")
    verified, seeds = [], []
    for episode, length, record, scores in zip(episodes, episode_lengths, selections, selected_scores, strict=True):
        require(isinstance(record, dict) and "seed" in record, "recorded fixed window selection")
        expected = {"episode_id": episode["id"], **select_windows(length, record["seed"])}
        require(record == expected and all(type(record[k]) is type(v) for k, v in expected.items())
                and all(type(v) is int for v in record["start_offsets"]),
                "exact independently seeded selection and episode identity")
        require(isinstance(scores, dict) and all(type(k) is int for k in scores)
                and set(scores) == set(record["start_offsets"]), "exact selected teacher-score keys only")
        for start in record["start_offsets"]:
            array(scores[start], np.float32, (min(PERIOD, length-start), PERIOD),
                  "selected exact teacher scores", finite=True)
        verified.append(expected)
        seeds.append(record["seed"])
    require(len(set(seeds)) == EPISODES, "distinct independently declared episode selection seeds")
    count = sum(r["selected_windows"] for r in verified)
    features = np.zeros((count, PERIOD, INPUT_DIM), dtype=np.float32)
    query = np.zeros((count, PERIOD), dtype=np.float32)
    targets = np.zeros((count, PERIOD, PERIOD), dtype=np.float32)
    legal = np.zeros((count, PERIOD, PERIOD), dtype=np.bool_)
    valid = np.zeros((count, PERIOD), dtype=np.bool_)
    weights = np.zeros((count, PERIOD), dtype=np.float64)
    lengths, indices, offsets = (np.empty(count, dtype=np.int64) for _ in range(3))
    full_counts = np.asarray([length - (length + PERIOD - 1) // PERIOD for length in episode_lengths], dtype=np.int64)
    window = 0
    for index, (episode, selection, scores) in enumerate(zip(episodes, verified, selected_scores, strict=True)):
        population, selected = selection["population_windows"], selection["selected_windows"]
        full_nonquery = int(full_counts[index])
        row_weight = (population / selected) / (EPISODES * full_nonquery) if full_nonquery else 0.
        for start in selection["start_offsets"]:
            length = min(PERIOD, episode_lengths[index] - start)
            stop = start + length
            features[window, :length] = episode["features"][start:stop]
            query[window] = scores[start][0]
            targets[window, :length] = scores[start]
            legal[window, :length] = episode["legal"][start:stop]
            valid[window, :length] = True
            weights[window, 1:length] = row_weight
            lengths[window], indices[window], offsets[window] = length, index, start
            window += 1
    nonquery = valid.copy()
    nonquery[:, 0] = False
    arrays = {"features": features, "query_scores": query, "targets": targets, "legal": legal,
              "valid_mask": valid, "nonquery_mask": nonquery, "lengths": lengths,
              "episode_index": indices, "step_offsets": offsets, "nonquery_weights": weights,
              "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
              "episode_nonquery_counts": full_counts}
    return {"version": WINDOW_VERSION, **{name: owned(value) for name, value in arrays.items()},
            "episode_ids": ids, "episode_regimes": tuple(e["regime"] for e in episodes),
            "episode_splits": tuple("train" for _ in episodes),
            "counts": {"episodes": EPISODES, "windows": count, "rows": int(valid.sum()),
                "query_rows": count, "nonquery_rows": int(nonquery.sum()),
                "zero_support_episodes": int((full_counts == 0).sum()),
                "query_only_windows": int((lengths == 1).sum()), "full_rows": sum(episode_lengths),
                "population_windows": sum(r["population_windows"] for r in verified),
                "full_nonquery_rows": int(full_counts.sum())},
            "sampling": {"version": VERSION, "unit": "disjoint period-four TRAIN windows",
                "episode_denominator": EPISODES, "window_cap": WINDOW_CAP, "selections": verified,
                "weight_formula": "(population_windows/selected_windows)/(54*full_episode_nonquery_rows)",
                "realized_weight_mass": float(weights.sum()), "renormalized": False,
                "shared_across_fits": True, "validation": "unchanged full census"}}
