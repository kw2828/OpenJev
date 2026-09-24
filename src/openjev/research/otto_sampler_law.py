"""Pure lookup geometry and source-supplied scalar OTTO sampler-law construction.

No SourceTracking instance, random stream, source location or observation enters
table construction. The caller authenticates the supplied original class first.
Native scalar functions retain their own SciPy/NumPy implementation and rounding.
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from openjev.research.otto_predictive_belief import cdf_law

VERSION = "otto-sampler-law-v1"
GRID, OFFSET, SIZE = 53, 52, 105


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_sensor_law(source_class, sensing, *, check=lambda: None):
    """Build scalar sampled-odor laws at every integer relative displacement.

    This mirrors seeded_environment._execute_action: Euclidean scalar distance,
    original mean-hit function, two original Poisson evaluations per category
    0..2, max(0,1-total) tail, then normalized-CDF 53-bit grid probabilities.
    Source=agent is found without an odor draw and retains a zero table row.
    """
    require(type(sensing) in (int, float) and sensing in (3., 4.), "registered sensing length 3 or 4")
    require(callable(check), "callable bound check")
    work = {"mu0_calls": 0, "distance_calls": 0, "mean_calls": 0,
            "poisson_calls": 0, "poisson_unbounded_calls": 0, "cdf_calls": 0}
    context = SimpleNamespace(Ndim=2, Nhits=4, lambda_over_dx=float(sensing), R_dt=2.)

    def unbounded(_self, mu, hit):
        work["poisson_unbounded_calls"] += 1
        return source_class._Poisson_unbounded(context, mu, hit)

    context._Poisson_unbounded = MethodType(unbounded, context)
    check()
    work["mu0_calls"] += 1
    source_class._set_mu0_Poisson(context)
    raw = np.zeros((SIZE, SIZE, 4), np.float64)
    effective = np.zeros_like(raw)
    maximum_adjustment = maximum_quantization = 0.
    removed = 0
    for x in range(SIZE):
        check()
        for y in range(SIZE):
            if x == y == OFFSET:
                continue
            work["distance_calls"] += 1
            distance = np.linalg.norm(np.asarray([x - OFFSET, y - OFFSET]), ord=2)
            work["mean_calls"] += 1
            mu = source_class._mean_number_of_hits(context, distance)
            total = 0
            for hit in range(3):
                work["poisson_calls"] += 1
                raw[x, y, hit] = source_class._Poisson(context, mu, hit)
                work["poisson_calls"] += 1
                total += source_class._Poisson(context, mu, hit)
            raw[x, y, 3] = np.maximum(0, 1. - total)
            work["cdf_calls"] += 1
            record = cdf_law(raw[x, y], uniform_bits=53)
            effective[x, y] = record["probabilities"]
            maximum_adjustment = max(maximum_adjustment, record["max_abs_adjustment"])
            maximum_quantization = max(maximum_quantization, record["max_cdf_quantization_adjustment"])
            removed += record["positive_entries_rounded_to_zero"]
    return {"probabilities": effective, "raw_probabilities": raw,
            "metadata": {"version": VERSION, "sensing_length": float(sensing), "grid": GRID,
                         "shape": [SIZE, SIZE, 4], "offset_origin": [OFFSET, OFFSET],
                         "uniform_bits": 53, "tail_category": "hits >=3", "work": work,
                         "max_abs_adjustment": maximum_adjustment,
                         "max_cdf_quantization_adjustment": maximum_quantization,
                         "positive_entries_rounded_to_zero": removed,
                         "scope": "authenticated scalar sensor functions, 53-bit discrete-uniform CDF law"}}


def _position(position):
    require(isinstance(position, np.ndarray) and position.dtype.kind in "iu" and position.shape == (2,)
            and bool(((position >= 0) & (position < GRID)).all()), "position must be in-bounds integer[2]")
    return position.astype(np.int64, copy=True)


def likelihood_at(table, position):
    """Return owned source-row-major float64[2809,4] likelihoods at a position."""
    require(isinstance(table, np.ndarray) and table.dtype == np.float64 and table.shape == (SIZE, SIZE, 4),
            "sampler table must be float64[105,105,4]")
    q = _position(position)
    # Only selected entries are consumed; full-table validation belongs to its
    # source-authenticated construction or saved-array admission.
    result = table[OFFSET - q[0]:OFFSET - q[0] + GRID,
                   OFFSET - q[1]:OFFSET - q[1] + GRID].reshape(GRID * GRID, 4).copy()
    require(np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all(), "finite categorical lookup")
    return result


def planned_positions(position, actions):
    """Deterministic positions for already committed actions; no observations."""
    q = _position(position)
    require(isinstance(actions, np.ndarray) and actions.dtype == np.int64 and actions.ndim == 1
            and len(actions) in (4, 8) and bool(((actions >= 0) & (actions < 4)).all()),
            "registered int64[4 or 8] action block")
    positions = []
    for action in actions:
        axis, direction = int(action) // 2, -1 if int(action) % 2 == 0 else 1
        q[axis] += direction
        require(bool(((q >= 0) & (q < GRID)).all()), "committed path must remain in bounds")
        positions.append(q.copy())
    return np.stack(positions)
