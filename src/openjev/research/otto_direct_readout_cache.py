"""Canonical frozen-feature cache for a separately admitted direct-readout study.

This helper performs no file IO, teacher calls, optimizer steps or phase
admission. The caller supplies authenticated in-memory state and history.
Extraction uses the unchanged frozen scheduled predictor at physical batch one
and chronological 32-step chunks. It never reads history targets, legal masks
or training weights. All returned arrays own their storage.

The affine helper defines a float64 normalized-score surrogate from cached
float32 features. It is not a bitwise replica of the float32 readout kernel or
the historical shuffled batch-six training program. Work counts include every
executed slow action/shadow readout, even though only base features are needed.
Elapsed seconds include validation, construction, copies and extraction.
"""
from __future__ import annotations

import math
import time
from itertools import pairwise

import numpy as np
import torch

from openjev.research import otto_query_memory_data as data
from openjev.research import otto_residual_contract as contract
from openjev.research import otto_scheduled_predictor as predictor

VERSION = "otto-direct-readout-cache-v1"
CHUNK = 32
ATOL = RTOL = 1e-5
CACHE_FIELDS = {"version", "z", "base", "parent_prediction", "query_mask", "prior_mask",
                "nonquery_mask", "support_mask", "episode_offsets", "work", "seconds"}
require = contract.require


def _history(history):
    require(isinstance(history, dict) and history.get("version") == data.VERSION
            and type(history.get("query_period")) is int and history["query_period"] == 4,
            "qualified projected P4 history")
    required = {"features", "query_scores", "episode_offsets", "query_mask", "prior_mask", "episode_count"}
    require(required <= history.keys(), "complete feature history fields")
    masks = contract.validate_feature_inputs(history["features"], history["query_scores"],
                                             history["episode_offsets"])
    rows = len(history["features"])
    require(type(history["episode_count"]) is int
            and history["episode_count"] == len(history["episode_offsets"]) - 1,
            "complete episode count")
    for name in ("query_mask", "prior_mask"):
        contract.array(history[name], np.bool_, (rows,), name)
        require(np.array_equal(history[name], masks[name]), "exact history " + name)
    return masks


def _packet(features, scores, query, low, high, start):
    length = min(CHUNK, high - low - start)
    take = slice(low + start, low + start + length)
    values = np.full((1, CHUNK, 31), np.nan, np.float32)
    answers = np.full((1, CHUNK, 4), np.nan, np.float32)
    queries = np.zeros((1, CHUNK), np.bool_)
    values[0, :length], answers[0, :length], queries[0, :length] = features[take], scores[take], query[take]
    return take, {"features": torch.from_numpy(values), "query_scores": torch.from_numpy(answers),
                  "query_mask": torch.from_numpy(queries), "lengths": torch.tensor([length]),
                  "episode_ends": torch.tensor([start + length == high - low])}


def validate_cache(cache):
    require(isinstance(cache, dict) and set(cache) == CACHE_FIELDS and cache["version"] == VERSION,
            "exact direct-readout cache fields and version")
    z = cache["z"]
    require(isinstance(z, np.ndarray) and z.ndim == 2, "flat augmented features")
    rows = len(z)
    expected = contract.masks_for_offsets(cache["episode_offsets"], rows)
    expected["nonquery_mask"] = ~expected["query_mask"]
    expected["support_mask"] = expected.pop("key_mask")
    for name, value in expected.items():
        contract.array(cache[name], np.bool_, (rows,), name)
        require(np.array_equal(cache[name], value), "complete cache " + name)
    contract.array(z, np.float64, (rows, 29), "augmented features", finite=True)
    require(bool((z[:, -1] == 1).all()), "unit augmented bias coordinate")
    unsupported = ~cache["support_mask"]
    require(z[unsupported, :28].tobytes() == np.zeros((int(unsupported.sum()), 28), np.float64).tobytes(),
            "first queries have no hidden feature")
    for name in ("base", "parent_prediction"):
        contract.array(cache[name], np.float32, (rows, 4), name, finite=True)
        require(cache[name][unsupported].tobytes() == np.zeros((int(unsupported.sum()), 4), np.float32).tobytes(),
                "positive zero unsupported " + name)
    require(isinstance(cache["work"], dict) and bool(cache["work"])
            and all(type(k) is str and type(v) is int and v >= 0 for k, v in cache["work"].items()),
            "nonnegative executed work counters")
    require(type(cache["seconds"]) in (int, float) and math.isfinite(cache["seconds"])
            and cache["seconds"] >= 0, "finite elapsed extraction seconds")


def extract(state, seed, history, check=lambda: None):
    """Copy a complete cache from exact unprefixed eight-tensor slow state.

    ``z`` is float64[N,29], with float32 hidden values promoted exactly and a
    unit bias coordinate on every row. Base/parent fields use raw score units:
    nonquery action forecasts, later-query causal priors, zero on first queries.
    Query targets are input only at their actual P4 times. No loss labels are
    copied into this cache. Source and split authorization are caller concerns.
    """
    started = time.perf_counter()
    masks = _history(history)
    check()
    features = history["features"].copy(order="C")
    scores = history["query_scores"].copy(order="C")
    offsets = history["episode_offsets"].copy()
    rows = len(features)
    z = np.zeros((rows, 29), np.float64)
    z[:, -1] = 1
    cache = {"version": VERSION, "z": z, "base": np.zeros((rows, 4), np.float32),
             "parent_prediction": np.zeros((rows, 4), np.float32),
             "query_mask": masks["query_mask"].copy(), "prior_mask": masks["prior_mask"].copy(),
             "nonquery_mask": ~masks["query_mask"], "support_mask": masks["key_mask"].copy(),
             "episode_offsets": offsets, "work": {"model_constructions": 0, "forward_chunks": 0,
                                                      "complete_episodes": 0}, "seconds": 0.}
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = predictor.from_state("frozen", seed, 4, state)
        cache["work"]["model_constructions"] += 1
        model.eval()
        for low, high in pairwise(offsets):
            low, high = int(low), int(high)
            carry = model.initial_carry(1)
            for start in range(0, high - low, CHUNK):
                check()
                take, packet = _packet(features, scores, masks["query_mask"], low, high, start)
                length = int(packet["lengths"][0])
                forecast = model(**packet, carry=carry)
                cache["work"]["forward_chunks"] += 1
                for name, value in forecast.work_counts.items():
                    require(type(value) is int and value >= 0, "actual nonnegative slow work")
                    key = "slow_" + name
                    cache["work"][key] = cache["work"].get(key, 0) + value
                prior = forecast.prior_mask[0, :length].numpy()
                key_mask = forecast.key_mask[0, :length].numpy()
                require(np.array_equal(prior, masks["prior_mask"][take])
                        and np.array_equal(key_mask, masks["key_mask"][take]), "causal forecast masks")
                query = masks["query_mask"][take]
                observed = scores[take][query].tobytes()
                require(forecast.action_prediction[0, :length].numpy()[query].tobytes() == observed
                        and forecast.prediction[0, :length].numpy()[query].tobytes() == observed,
                        "query score bytes unchanged")
                for name in ("keys_hidden", "prediction", "action_prediction", "prior", "shadow_prior"):
                    padding = getattr(forecast, name)[0, length:].numpy()
                    require(padding.tobytes() == np.zeros_like(padding).tobytes(), "positive zero padded " + name)
                cache["z"][take, :28] = forecast.keys_hidden[0, :length].numpy()
                base = cache["base"][take]
                parent = cache["parent_prediction"][take]
                base[~query] = forecast.prediction[0, :length].numpy()[~query]
                parent[~query] = forecast.action_prediction[0, :length].numpy()[~query]
                base[prior] = forecast.prior[0, :length].numpy()[prior]
                parent[prior] = forecast.shadow_prior[0, :length].numpy()[prior]
                carry = predictor.detach_carry(forecast.carry)
            require(bool(carry.base.ended.all()) and int(carry.base.absolute_step[0]) == high - low,
                    "complete episode end and clock")
            cache["work"]["complete_episodes"] += 1
    validate_cache(cache)
    cache["seconds"] = time.perf_counter() - started
    require(math.isfinite(cache["seconds"]) and cache["seconds"] >= 0, "finite extraction clock")
    return cache


def theta_from_state(state):
    """Return an owned float64[4,29] head in normalized residual units."""
    values = []
    for name, shape in (("action_residual.weight", (4, 28)), ("action_residual.bias", (4,))):
        require(name in state, "complete residual head")
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float32
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "finite residual " + name)
        values.append(value.detach().numpy().astype(np.float64))
    return np.concatenate((values[0], values[1][:, None]), axis=1)


def affine_prediction(cache, theta):
    """Normalized float64 surrogate; unsupported first-query rows remain zero."""
    validate_cache(cache)
    contract.array(theta, np.float64, (4, 29), "readout theta", finite=True)
    support = cache["support_mask"]
    result = np.zeros_like(cache["base"], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        residual = cache["z"][support] @ theta.T
        residual -= residual.mean(axis=1, keepdims=True)
        result[support] = cache["base"][support].astype(np.float64) / 64 + residual
    require(bool(np.isfinite(result).all()), "finite normalized affine predictions")
    return result
