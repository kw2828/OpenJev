"""Training-cache-only constant pose mixture; no model or data loading.

The rotation certificate concerns the float64, SVD-projected SO(3) objective,
not the original float32 production blend. Bounds use the angular metric's
triangle inequality, with explicit conservative numerical guards. This is a
numerical certificate, not an outward-rounded interval-arithmetic proof.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

ANGLE_GUARD = 1e-12
OBJECTIVE_GUARD = 1e-12


def _inputs(fp, fr, sp, sr, tp, tr):
    arrays = (fp, fr, sp, sr, tp, tr)
    if any(not isinstance(x, np.ndarray) or x.dtype not in (np.float32, np.float64)
           or not np.isfinite(x).all() for x in arrays):
        raise ValueError("finite NumPy float32/float64 cached poses required")
    if fp.ndim != 3 or fp.shape[-1] != 3 or min(fp.shape[:2]) < 1:
        raise ValueError("positions must have positive [batch,horizon,3] shape")
    if sp.shape != fp.shape or tp.shape != fp.shape:
        raise ValueError("position shapes must agree")
    if any(x.shape != (*fp.shape[:-1], 3, 3) for x in (fr, sr, tr)):
        raise ValueError("rotation shapes must match positions")


def _project(rotation):
    original = rotation.astype(np.float64, copy=True).reshape(-1, 3, 3)
    u, singular, vt = np.linalg.svd(original)
    sign = np.where(np.linalg.det(u @ vt) < 0, -1., 1.)
    u[:, :, -1] *= sign[:, None]
    projected = u @ vt
    if not np.isfinite(projected).all():
        raise FloatingPointError("nonfinite SO(3) projection")
    stats = {
        "matrices": len(projected),
        "max_frobenius_change": float(np.linalg.norm(projected - original, axis=(1, 2)).max()),
        "min_input_singular_value": float(singular.min()),
        "reflections_corrected": int(np.sum(sign < 0)),
        "max_orthogonality_error": float(np.abs(projected @ projected.swapaxes(-1, -2) - np.eye(3)).max()),
        "max_determinant_error": float(np.abs(np.linalg.det(projected) - 1).max()),
    }
    if not all(math.isfinite(value) for value in stats.values()):
        raise FloatingPointError("nonfinite projection diagnostics")
    return projected, stats


def _quaternion(rotation):
    """Unit wxyz quaternion; largest component chooses the stable formula."""
    r = rotation
    squared = np.stack((1 + np.trace(r, axis1=-2, axis2=-1),
        1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]), axis=-1)
    a, b, c = r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]
    d, e, f = r[:, 0, 1] + r[:, 1, 0], r[:, 0, 2] + r[:, 2, 0], r[:, 1, 2] + r[:, 2, 1]
    candidates = np.stack((np.stack((squared[:, 0], a, b, c), -1),
        np.stack((a, squared[:, 1], d, e), -1),
        np.stack((b, d, squared[:, 2], f), -1),
        np.stack((c, e, f, squared[:, 3]), -1)), axis=1)
    q = candidates[np.arange(len(r)), np.argmax(squared, axis=1)]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # At exact pi, the largest vector component is positive by construction.
    q *= np.where(q[:, :1] < 0, -1., 1.)
    return q


def _multiply(q, r):
    return np.concatenate((q[:, :1] * r[:, :1] - np.sum(q[:, 1:] * r[:, 1:], axis=1, keepdims=True),
        q[:, :1] * r[:, 1:] + r[:, :1] * q[:, 1:] + np.cross(q[:, 1:], r[:, 1:])), axis=1)


def _rotation_problem(fast, slow, target):
    relative = _quaternion(fast @ slow.swapaxes(-1, -2))
    sine = np.linalg.norm(relative[:, 1:], axis=1)
    half_angle = np.arctan2(sine, relative[:, 0])
    speed = 2 * half_angle
    axis = np.divide(relative[:, 1:], sine[:, None], out=np.zeros_like(relative[:, 1:]), where=sine[:, None] > 0)
    qs, qf = _quaternion(slow), _quaternion(fast)
    target_inverse = _quaternion(target)
    target_inverse[:, 1:] *= -1

    def distances(alpha):
        if alpha == 0:
            q = qs
        elif alpha == 1:
            q = qf
        else:
            phase = alpha * half_angle
            q = _multiply(np.concatenate((np.cos(phase)[:, None], np.sin(phase)[:, None] * axis), axis=1), qs)
        error = _multiply(q, target_inverse)
        result = 2 * np.arctan2(np.linalg.norm(error[:, 1:], axis=1), np.abs(error[:, 0]))
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite rotation objective")
        return result

    return distances, speed


def fit_constant(fp, fr, sp, sr, tp, tr, *, tolerance=1e-7, max_evaluations=4095):
    """Return a JSON-safe coefficient/certificate record for cached [B,H] poses.

    Position minimizes mean squared Euclidean distance analytically. Rotation
    minimizes mean squared geodesic angle on separately SVD-projected inputs.
    Its interpolation is Exp(alpha*Log(Rfast*Rslow.T))*Rslow. Every interval
    remains in the binary leaf partition, including regions worse than the
    incumbent. Endpoints and every evaluated midpoint are retained. Exact
    objective ties choose the smallest evaluated alpha, with no tie tolerance.

    ``certified`` means the guarded upper/lower gap reached ``tolerance``;
    exhaustion returns the best observed value with ``certified=False``.
    Degenerate position differences choose0.5; no external state is retained.
    """
    _inputs(fp, fr, sp, sr, tp, tr)
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite scalar tolerance required")
    if type(max_evaluations) is not int or max_evaluations < 3:
        raise ValueError("integer max_evaluations >=3 required")
    delta = fp.astype(np.float64) - sp.astype(np.float64)
    residual = tp.astype(np.float64) - sp.astype(np.float64)
    denominator = float(np.sum(delta * delta))
    numerator = float(np.sum(delta * residual))
    if not math.isfinite(denominator) or not math.isfinite(numerator):
        raise FloatingPointError("nonfinite position normal equation")
    degenerate = denominator == 0
    raw_alpha = None if degenerate else numerator / denominator
    if raw_alpha is not None and not math.isfinite(raw_alpha):
        raise FloatingPointError("nonfinite unconstrained position solution")
    alpha_p = .5 if degenerate else float(np.clip(raw_alpha, 0., 1.))
    position_objective = float(np.mean(np.sum((alpha_p * delta - residual) ** 2, axis=-1)))
    if not math.isfinite(position_objective):
        raise FloatingPointError("nonfinite position objective")
    fast, fast_projection = _project(fr)
    slow, slow_projection = _project(sr)
    target, target_projection = _project(tr)
    distances, speed = _rotation_problem(fast, slow, target)
    evaluations, intervals, heap = [], [], []
    leaves = set()
    best_value, best_alpha = math.inf, 0.

    def evaluate(alpha, role, radius=0.):
        nonlocal best_value, best_alpha
        errors = distances(alpha)
        value = float(np.mean(errors * errors))
        # Guards cover normal float64 errors, not adversarial interval proof.
        lower = max(0., float(np.mean(np.maximum(0., errors - ANGLE_GUARD
                    - (speed + ANGLE_GUARD) * radius) ** 2)) - OBJECTIVE_GUARD)
        index = len(evaluations)
        evaluations.append({"id": index, "alpha": alpha, "objective": value,
                            "role": role, "radius": radius, "lower_bound": lower})
        best_value, best_alpha = min((best_value, best_alpha), (value, alpha))
        return index, lower

    def interval(left, right, parent):
        center = (left + right) / 2
        evaluation, lower = evaluate(center, "interval_center", (right - left) / 2)
        index = len(intervals)
        intervals.append({"id": index, "parent": parent, "left": left, "right": right,
                          "center_evaluation": evaluation, "lower_bound": lower, "children": None})
        leaves.add(index)
        heapq.heappush(heap, (lower, left, index))
        return index

    evaluate(0., "endpoint")
    evaluate(1., "endpoint")
    interval(0., 1., None)
    status = "evaluation_cap"
    while True:
        lower = heap[0][0]
        upper = best_value + OBJECTIVE_GUARD
        if upper - lower <= tolerance:
            status = "tolerance_reached"
            break
        if len(evaluations) + 2 > max_evaluations:
            break
        _, _, index = heapq.heappop(heap)
        node = intervals[index]
        center = evaluations[node["center_evaluation"]]["alpha"]
        if center == node["left"] or center == node["right"]:
            heapq.heappush(heap, (node["lower_bound"], node["left"], index))
            status = "floating_interval_limit"
            break
        children = [interval(node["left"], center, index), interval(center, node["right"], index)]
        node["children"] = children
        leaves.remove(index)
    lower = heap[0][0]
    upper = best_value + OBJECTIVE_GUARD
    return {
        "version": "pose-constant-v1", "alpha": [alpha_p, best_alpha],
        "input_shape": list(fp.shape), "examples": int(np.prod(fp.shape[:2])),
        "position": {"numerator": numerator, "denominator": denominator, "unclipped_alpha": raw_alpha,
                     "degenerate": degenerate, "objective_m2": position_objective},
        "rotation": {
            "objective": "mean squared geodesic radians on float64 SVD-projected SO(3) inputs",
            "interpolation": "Exp(alpha*Log(Rfast*Rslow.T))*Rslow; principal log, largest-axis positive at exact pi",
            "certificate_scope": "numerical guarded bound for projected interpolation only; not float32 production or formal interval arithmetic",
            "projection_method": "float64 SVD U diag(1,1,det(UVt)) Vt; no input mutation",
            "projection": {"fast": fast_projection, "slow": slow_projection, "target": target_projection},
            "tolerance": float(tolerance), "max_evaluations": max_evaluations,
            "call_count": len(evaluations), "status": status, "certified": status == "tolerance_reached",
            "alpha": best_alpha, "objective_at_alpha": best_value,
            "upper_bound": upper, "lower_bound": lower, "gap": upper - lower,
            "angle_guard": ANGLE_GUARD, "objective_guard": OBJECTIVE_GUARD,
            "speed_min": float(speed.min()), "speed_max": float(speed.max()),
            "tie_rule": "smallest evaluated alpha among exact equal float64 objectives",
            "queue_rule": "smallest interval lower bound, then left endpoint, then interval id",
            "evaluations": evaluations, "intervals": intervals,
            "leaves": sorted(leaves, key=lambda i: (intervals[i]["left"], i)),
        },
    }
