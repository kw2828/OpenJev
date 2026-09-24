"""In-memory contracts for a prospective frozen-feature estimator comparison.

This module performs no file IO, empirical decoding, model call or admission.
Shapes and schedules do not authenticate provenance: the eventual worker must
bind source, checkpoint, cohort and phase identities before passing arrays.
Masked query scores, shadow priors and first-step cues may contain poison.
They are never required to be finite where the estimator cannot consume them.
"""
from __future__ import annotations

import numbers
from itertools import pairwise

import numpy as np

CACHE_VERSION = "otto-residual-cache-v1"
HORIZON, QUERY_PERIOD, KEY_DIM, SCORE_DIM, FEATURE_DIM = 2188, 4, 8, 4, 31
FIT_SEEDS = (309000001, 309000002, 309000003)
TAUS = (.01, .1, 1., 10.)
BASELINES = ("pretrained", "joint_aux", "last_error", "trace_delta")
RLS_METHODS = ("rls_full", "rls_diagonal", "rls_shrink_025", "rls_shrink_050", "rls_shrink_075")
METHODS = BASELINES + RLS_METHODS
MULTIPLIERS = {"rls_full": 1., "rls_diagonal": 1., "rls_shrink_025": .25,
               "rls_shrink_050": .5, "rls_shrink_075": .75}
CACHE_FIELDS = {"version", "query_period", "base_action", "joint_action", "shadow_prior", "cues",
                "query_scores", "query_mask", "prior_mask", "key_mask", "episode_offsets", "work_counts"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, dtype, shape, name, *, finite=False):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and value.shape == shape,
            name + " must have the declared ndarray dtype and shape")
    require(not finite or bool(np.isfinite(value).all()), name + " must be finite")
    return value


def masks_for_offsets(episode_offsets, rows):
    """Return fresh masks for complete, nonempty, bounded P4 episodes."""
    require(type(rows) is int and rows > 0, "positive integer row count")
    require(isinstance(episode_offsets, np.ndarray) and episode_offsets.dtype == np.int64
            and episode_offsets.ndim == 1 and len(episode_offsets) >= 2, "int64 episode boundaries")
    require(episode_offsets[0] == 0 and episode_offsets[-1] == rows
            and bool((episode_offsets[1:] > episode_offsets[:-1]).all()), "strict complete episode offsets")
    require(bool((np.diff(episode_offsets) <= HORIZON).all()), "bounded complete episodes")
    query = np.zeros(rows, dtype=np.bool_)
    prior = np.zeros(rows, dtype=np.bool_)
    key = np.ones(rows, dtype=np.bool_)
    for low, high in pairwise(episode_offsets):
        steps = np.arange(int(high - low))
        query[low:high] = steps % QUERY_PERIOD == 0
        prior[low:high] = (steps > 0) & (steps % QUERY_PERIOD == 0)
        key[low] = False
    return {"query_mask": query, "prior_mask": prior, "key_mask": key}


def validate_feature_inputs(features, query_scores, episode_offsets):
    """Check all active input chronology before either frozen forward executes."""
    require(isinstance(features, np.ndarray) and features.ndim == 2, "flat feature matrix")
    rows = len(features)
    array(features, np.float32, (rows, FEATURE_DIM), "features", finite=True)
    array(query_scores, np.float32, (rows, SCORE_DIM), "query-only scores")
    masks = masks_for_offsets(episode_offsets, rows)
    visible = query_scores[masks["query_mask"]]
    require(bool(np.isfinite(visible).all()), "finite visible query scores")
    require(np.array_equal(visible / np.float32(64) * np.float32(64), visible),
            "visible scores must survive exact power-of-two scaling")
    for low, high in pairwise(episode_offsets):
        steps = np.arange(int(high - low), dtype=np.int64)
        require(np.array_equal(features[low:high, 15], (steps / HORIZON).astype(np.float32))
                and np.array_equal(features[low:high, 16], ((steps % QUERY_PERIOD) / HORIZON).astype(np.float32))
                and bool((features[low:high, 17] == 1).all()), "exact per-episode public P4 chronology")
    return masks


def validate_cache(cache):
    """Validate, without modifying or copying, the complete cached interface."""
    require(isinstance(cache, dict) and set(cache) == CACHE_FIELDS, "exact cache fields")
    require(cache["version"] == CACHE_VERSION and type(cache["query_period"]) is int
            and cache["query_period"] == QUERY_PERIOD, "cache version and fixed P4 schedule")
    base = cache["base_action"]
    require(isinstance(base, np.ndarray) and base.ndim == 2, "flat cached action matrix")
    rows = len(base)
    masks = masks_for_offsets(cache["episode_offsets"], rows)
    for name, expected in masks.items():
        array(cache[name], np.bool_, (rows,), name)
        require(np.array_equal(cache[name], expected), "cache " + name + " follows complete episode chronology")
    for name in ("base_action", "joint_action", "shadow_prior", "query_scores"):
        array(cache[name], np.float32, (rows, SCORE_DIM), name, finite=name in ("base_action", "joint_action"))
    array(cache["cues"], np.float32, (rows, KEY_DIM), "cues")
    require(bool(np.isfinite(cache["cues"][masks["key_mask"]]).all()), "finite consumed cues")
    require(bool(np.isfinite(cache["shadow_prior"][masks["prior_mask"]]).all()), "finite eligible shadow forecasts")
    visible = cache["query_scores"][masks["query_mask"]]
    require(bool(np.isfinite(visible).all()), "finite cached query scores")
    for name in ("base_action", "joint_action"):
        require(np.array_equal(cache[name][masks["query_mask"]].view(np.uint32), visible.view(np.uint32)),
                "bitwise exact query answers in " + name)
    counts = cache["work_counts"]
    require(isinstance(counts, dict) and all(type(k) is str and bool(k)
            and type(v) is int and v >= 0 for k, v in counts.items()), "nonnegative integer cache work counts")


def method_config(method, tau):
    """Return the canonical fixed ratio; baselines cannot take an RLS ratio."""
    require(type(method) is str and method in METHODS, "declared estimator method")
    if method in BASELINES:
        require(tau is None, "ordinary baselines have no prior/noise ratio")
        return None
    require(isinstance(tau, numbers.Real) and not isinstance(tau, (bool, np.bool_)), "explicit real RLS ratio")
    try:
        ratio = float(tau)
    except (OverflowError, ValueError) as error:
        raise ValueError("RLS ratio must be representable as a finite grid value") from error
    require(ratio in TAUS, "RLS ratio must be from the prospective grid")
    return ratio


def view_name(method, tau=None):
    tau = method_config(method, tau)
    return method if tau is None else f"{method}@tau={tau:g}"


def view_specs(stage, selected_tau=None):
    """Exact ordered model/ratio/seed roster, without executing any model.

    Public phase names are dev and confirm. A future adapter may use the
    existing metric module's test label for confirm, in this study's namespace.
    That compatibility label never admits the closed prior study's TEST data.
    """
    require(type(stage) is str and stage in ("dev", "confirm"), "declared study phase")
    if stage == "dev":
        require(selected_tau is None, "development evaluates the complete fixed ratio grid")
        ratios = TAUS
    else:
        ratios = (method_config("rls_full", selected_tau),)
    specs = [(method, None, seed) for method in BASELINES for seed in FIT_SEEDS]
    specs.extend((method, tau, seed) for method in RLS_METHODS for tau in ratios for seed in FIT_SEEDS)
    require(len(specs) == (72 if stage == "dev" else 27), "complete fixed view count")
    return tuple(specs)
