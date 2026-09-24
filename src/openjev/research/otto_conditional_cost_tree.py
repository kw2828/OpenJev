"""Exact short odor tree and fixed-draw long continuations over public beliefs.

This helper has no environment, teacher implementation, file or random stream.
The caller supplies the already-declared 53-bit categorical laws, independent
legacy-filter update and score callbacks, and every Monte Carlo draw integer.
Tree weights follow the sampler law, never the legacy evidence floor. Legacy
state is carried separately and copied before crossing callback boundaries.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = "otto-conditional-cost-tree-v1"
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
    """Integer cumulative bins of a law already quantized onto 2**53 draws.

    Do not re-normalize a CDF here: doing so would define a different law.
    A boundary draw belongs to the next positive category (right-side search).
    """
    require(isinstance(probabilities, np.ndarray) and probabilities.dtype == np.float64
            and probabilities.ndim == 1 and len(probabilities) > 0, "float64 categorical vector")
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all()
            and (probabilities <= 1).all(), "categorical entries in [0,1]")
    scaled = np.ldexp(probabilities, BITS)
    require((scaled == np.floor(scaled)).all(), "categorical law must use exact 53-bit bins")
    counts = scaled.astype(np.uint64)
    require(sum(int(v) for v in counts) == BINS, "categorical integer mass must be 2**53")
    return np.cumsum(counts, dtype=np.uint64)


def _inputs(root, legacy, likelihoods, positions, horizon, update, score, check, emit):
    require(isinstance(root, np.ndarray) and root.ndim == 1 and len(root) > 0, "nonempty root vector")
    n = len(root)
    _array(root, (n,), np.float64, "root")
    integer_cdf(root)
    _array(legacy, (n,), np.float64, "legacy root")
    require((legacy >= 0).all(), "nonnegative legacy root; no normalization repair")
    _array(likelihoods, (horizon, n, 4), np.float64, "likelihoods")
    _array(positions, (horizon,), np.int64, "positions")
    require(((positions >= 0) & (positions < n)).all(), "in-bounds cell indices")
    require(all(callable(v) for v in (update, score, check, emit)), "callable bounded operations")
    for h, position in enumerate(positions):
        for cell, row in enumerate(likelihoods[h]):
            if cell == position and not np.any(row):
                continue
            integer_cdf(row)
    return n


def _updated(update, legacy, position, odor, n):
    result = update(legacy.copy(), int(position), int(odor))
    _array(result, (n,), np.float64, "updated legacy state")
    require((result >= 0).all(), "updated legacy state nonnegative")
    return result.copy()


def _scores(score, legacy, position):
    result = score(legacy.copy(), int(position))
    _array(result, (4,), np.float32, "teacher scores")
    return result.copy()


def histories4():
    """Fixed dense lexicographic roster, including zero-probability histories."""
    ids = np.arange(256, dtype=np.int64)
    return np.stack([(ids // (4**power)) % 4 for power in (3, 2, 1, 0)], axis=1)


def exact_h4(root, legacy, likelihoods, positions, update, score, *, check=lambda: None, emit=lambda _event: None):
    """Score positive-mass four-odor leaves only; found contributes no cost.

    Weights are unconditional from the declared root law. Every teacher score
    follows all four leaf observations. Zero-weight dense entries have zero
    score placeholders and must not be interpreted as teacher evaluations.
    """
    n = _inputs(root, legacy, likelihoods, positions, 4, update, score, check, emit)
    histories = histories4()
    weights = np.zeros(256, np.float64)
    costs = np.zeros((256, 4), np.float32)
    work = {"internal_nodes": 0, "positive_children": 0, "legacy_updates": 0,
            "teacher_calls": 0, "found_positive_branches": 0}
    found = []

    def visit(joint, state, depth, code, history):
        check()
        work["internal_nodes"] += 1
        position = int(positions[depth])
        found_mass = float(joint[position])
        found.append(found_mass)
        work["found_positive_branches"] += int(found_mass > 0)
        surviving = joint.copy()
        surviving[position] = 0.
        for odor in range(4):
            child = surviving * likelihoods[depth, :, odor]
            require(not ((surviving > 0) & (likelihoods[depth, :, odor] > 0) & (child == 0)).any(),
                    "positive tree product underflow")
            mass = float(child.sum(dtype=np.float64))
            if mass == 0:
                continue
            work["positive_children"] += 1
            context = {"mode": "exact", "horizon": depth + 1, "history": [*history, odor]}
            emit(context)
            check()
            after = _updated(update, state, position, odor, n)
            work["legacy_updates"] += 1
            child_code = code * 4 + odor
            if depth == 3:
                check()
                costs[child_code] = _scores(score, after, position)
                work["teacher_calls"] += 1
                weights[child_code] = mass
            else:
                visit(child, after, depth + 1, child_code, context["history"])

    visit(root.copy(), legacy.copy(), 0, 0, [])
    found_mass = math.fsum(found)
    survival_mass = math.fsum(float(v) for v in weights)
    require(abs(found_mass + survival_mass - 1.) <= 1e-12, "exact tree conserves root mass")
    require(work["teacher_calls"] == int(np.count_nonzero(weights)) <= 256, "leaf-only teacher accounting")
    return {"histories": histories, "weights": weights, "costs": costs, "supported": weights > 0,
            "survival_mass": survival_mass, "found_mass": found_mass, "work": work}


def sample_h8(root, legacy, likelihoods, positions, draws, exact, update, score, *,
              check=lambda: None, emit=lambda _event: None):
    """Consume supplied independent integer draws; never generate or retry one.

    All nine draws per row are supplied even when an early found makes later
    draws unused. H4 costs come only from the exact tree. H8 surviving costs
    invoke the unchanged teacher once. Found suffixes and cost placeholders are
    zero; downstream unconditional regret must explicitly use the alive masks.
    """
    n = _inputs(root, legacy, likelihoods, positions, 8, update, score, check, emit)
    require(isinstance(draws, np.ndarray) and draws.dtype == np.uint64 and draws.ndim == 2
            and 1 <= len(draws) <= 128 and draws.shape[1] == 9 and (draws < BINS).all(),
            "one through 128 complete uint64[9] draw rows in [0,2**53)")
    require(isinstance(exact, dict), "exact H4 result required")
    _array(exact["histories"], (256, 4), np.int64, "exact histories")
    require(np.array_equal(exact["histories"], histories4()), "canonical exact H4 history roster")
    _array(exact["weights"], (256,), np.float64, "exact weights")
    _array(exact["costs"], (256, 4), np.float32, "exact costs")
    require((exact["weights"] >= 0).all(), "nonnegative exact weights")
    root_cdf = integer_cdf(root)
    sensor_cdfs = np.zeros(likelihoods.shape, np.uint64)
    for h, position in enumerate(positions):
        for cell, row in enumerate(likelihoods[h]):
            if cell != position or np.any(row):
                sensor_cdfs[h, cell] = integer_cdf(row)
    count = len(draws)
    sources = np.empty(count, np.int64)
    outcomes = np.full((count, 8), 4, np.int64)
    alive4, alive8 = np.zeros(count, np.bool_), np.zeros(count, np.bool_)
    costs4, costs8 = np.zeros((count, 4), np.float32), np.zeros((count, 4), np.float32)
    work = {"allocated_draw_integers": count * 9, "source_draws": 0, "odor_draws": 0,
            "unused_odor_draws": 0, "legacy_updates": 0, "h4_cost_lookups": 0, "teacher_calls": 0}
    for i in range(count):
        check()
        source = int(np.searchsorted(root_cdf, draws[i, 0], side="right"))
        require(source < n and root[source] > 0, "source draw remains on declared support")
        sources[i] = source
        work["source_draws"] += 1
        state, code = legacy.copy(), 0
        for h, position in enumerate(positions):
            if source == position:
                break
            odor = int(np.searchsorted(sensor_cdfs[h, source], draws[i, h + 1], side="right"))
            require(odor < 4 and likelihoods[h, source, odor] > 0, "odor draw remains on declared support")
            work["odor_draws"] += 1
            outcomes[i, h] = odor
            emit({"mode": "mc", "sample": i, "horizon": h + 1, "history": outcomes[i, :h + 1].tolist()})
            check()
            state = _updated(update, state, int(position), odor, n)
            work["legacy_updates"] += 1
            if h < 4:
                code = code * 4 + odor
            if h == 3:
                require(exact["weights"][code] > 0, "sampled H4 leaf has positive exact mass")
                costs4[i] = exact["costs"][code]
                alive4[i] = True
                work["h4_cost_lookups"] += 1
            if h == 7:
                check()
                costs8[i] = _scores(score, state, int(position))
                alive8[i] = True
                work["teacher_calls"] += 1
    work["unused_odor_draws"] = count * 8 - work["odor_draws"]
    require(work["teacher_calls"] == int(alive8.sum()) and work["h4_cost_lookups"] == int(alive4.sum()),
            "one H8 teacher call per survivor; H4 lookup only")
    return {"source_indices": sources, "outcomes": outcomes, "alive4": alive4, "alive8": alive8,
            "costs4": costs4, "costs8": costs8, "work": work}
