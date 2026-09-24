"""Supplied-draw public-belief continuations with one horizon-eight annotation.

No RNG, environment, model implementation or file is accessed here. Sampling
uses the supplied strict 53-bit source/sensor laws; the teacher callback sees
only its separately updated legacy state. Found histories retain zero costs.
"""
from __future__ import annotations

import numpy as np

VERSION = "otto-conditional-label-sampling-v1"
BITS, BINS = 53, 2**53


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, dtype, name):
    require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == shape,
            name + " shape and dtype")
    require(np.isfinite(value).all(), name + " must be finite")
    return value


def integer_cdf(probabilities):
    """Cumulative integer bins, without a second floating normalization."""
    require(isinstance(probabilities, np.ndarray) and probabilities.dtype == np.float64
            and probabilities.ndim == 1 and len(probabilities) > 0, "float64 categorical vector")
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and (probabilities <= 1).all(), "categorical entries in [0,1]")
    scaled = np.ldexp(probabilities, BITS)
    require((scaled == np.floor(scaled)).all(), "categorical law must use exact 53-bit bins")
    counts = scaled.astype(np.uint64)
    require(sum(int(value) for value in counts) == BINS, "categorical integer mass must be 2**53")
    return np.cumsum(counts, dtype=np.uint64)


def sample_endpoint(root, legacy, likelihoods, positions, draws, update, score, *,
                    check=lambda: None, emit=lambda _event: None):
    """Consume a complete32/128-history bank; score H8 survivors only.

    All nine uint53 integers per row are allocated before this function runs.
    Early found makes later integers unused, not unallocated. The returned
    arrays are owned and all four terminal costs are exact positive zero.
    """
    require(isinstance(root, np.ndarray) and root.ndim == 1 and len(root) > 0, "nonempty root vector")
    n = len(root)
    _array(root, (n,), np.float64, "root")
    root_cdf = integer_cdf(root)
    _array(legacy, (n,), np.float64, "legacy root")
    require((legacy >= 0).all(), "nonnegative legacy root; no normalization repair")
    _array(likelihoods, (8, n, 4), np.float64, "likelihoods")
    _array(positions, (8,), np.int64, "positions")
    require(((positions >= 0) & (positions < n)).all(), "in-bounds cell indices")
    require(isinstance(draws, np.ndarray) and draws.dtype == np.uint64
            and draws.shape in ((32, 9), (128, 9)) and (draws < BINS).all(), "complete32/128 uint53 draw rows")
    require(all(callable(value) for value in (update, score, check, emit)), "callable bounded operations")
    sensor_cdfs = np.zeros(likelihoods.shape, np.uint64)
    for h, position in enumerate(positions):
        check()
        for cell, row in enumerate(likelihoods[h]):
            if cell != position or np.any(row):
                sensor_cdfs[h, cell] = integer_cdf(row)
    count = len(draws)
    sources = np.empty(count, np.int64)
    outcomes = np.full((count, 8), 4, np.int64)
    alive = np.zeros(count, np.bool_)
    costs = np.zeros((count, 4), np.float32)
    work = {"allocated_draw_integers": count * 9, "source_draws": 0, "odor_draws": 0,
            "unused_odor_draws": 0, "legacy_updates": 0, "teacher_calls": 0}
    for sample in range(count):
        check()
        source = int(np.searchsorted(root_cdf, draws[sample, 0], side="right"))
        require(source < n and root[source] > 0, "source draw stays on declared support")
        sources[sample] = source
        work["source_draws"] += 1
        state = legacy.copy()
        for h, position in enumerate(positions):
            if source == position:
                break
            odor = int(np.searchsorted(sensor_cdfs[h, source], draws[sample, h + 1], side="right"))
            require(odor < 4 and likelihoods[h, source, odor] > 0, "odor draw stays on declared support")
            outcomes[sample, h] = odor
            work["odor_draws"] += 1
            emit({"mode": "mc", "sample": sample, "horizon": h + 1,
                  "history": outcomes[sample, :h + 1].tolist()})
            check()
            updated = update(state.copy(), int(position), odor)
            _array(updated, (n,), np.float64, "updated legacy state")
            require((updated >= 0).all(), "updated legacy state nonnegative")
            state = updated.copy()
            work["legacy_updates"] += 1
            if h == 7:
                check()
                value = score(state.copy(), int(position))
                _array(value, (4,), np.float32, "teacher endpoint costs")
                require((value >= 0).all(), "nonnegative teacher endpoint costs")
                costs[sample] = value
                alive[sample] = True
                work["teacher_calls"] += 1
    work["unused_odor_draws"] = count * 8 - work["odor_draws"]
    require(work["teacher_calls"] == int(alive.sum()), "one teacher call per horizon-eight survivor")
    return {"source_indices": sources, "outcomes": outcomes, "alive": alive, "costs": costs, "work": work}
