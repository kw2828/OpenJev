"""Public numerical RLPolicy branches, without an environment or learned model.

The float64 route is a reference calculation. The separately named float32
route casts completed float64 branches before value evaluation and reduction;
it is NumPy arithmetic, not a TensorFlow or deployed-policy qualification.
Only nonterminal numerical inputs are supported. No terminal-value constraint
is imposed on a generic value function or on a coefficient at the origin.
"""

from dataclasses import dataclass

import numpy as np

N = 53
CENTERED = 105
EPSILON = 1e-10
MAX_TEMPLATES = 64


def _owned(array):
    """An immutable, independent C-order array backed by immutable bytes."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _float64(array, shape, name, *, nonnegative=False):
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float64"):
        raise ValueError(f"{name} must be a float64 ndarray")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return array


def _position(position):
    if not isinstance(position, (list, tuple, np.ndarray)) or len(position) != 2:
        raise ValueError("position must contain two integer coordinates")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in position):
        raise ValueError("position must contain nonboolean integers")
    if any(not 0 <= v < N for v in position):
        raise ValueError("position must lie in the 53-square grid")
    return tuple(int(v) for v in position)


def _eligible(actions):
    if not isinstance(actions, (tuple, list, np.ndarray)):
        raise TypeError("eligible_actions must be an explicit sequence")
    if not len(actions) or any(
        isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) or not 0 <= a < 4
        for a in actions
    ):
        raise ValueError("eligible_actions must contain IDs in 0..3")
    result = tuple(int(a) for a in actions)
    if len(set(result)) != len(result):
        raise ValueError("duplicate eligible action")
    return tuple(sorted(result))


def _arithmetic(name):
    if name not in ("float64", "float32"):
        raise ValueError("arithmetic must be explicitly float64 or float32")
    return np.dtype(name)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RLBranches:
    """Owned immutable arrays in action-major, hit-minor order (row=4*a+h).

    centered_u and centered_z have shape (16,105,105), successors (16,2),
    raw_masses and weights (4,4), and kernel (4,107,107). All probabilities
    remain float64 here. Eligible IDs affect selection only, never branching.
    """

    centered_u: np.ndarray
    centered_z: np.ndarray
    raw_masses: np.ndarray
    weights: np.ndarray
    successors: np.ndarray
    kernel: np.ndarray
    eligible_actions: tuple


def rl_branches(belief, position, kernel, eligible_actions):
    """Construct all 16 released-policy branches with the original mass floor.

    Crops use 53-successor; centering pads at 52-successor. Each raw 53x53
    product is summed separately, before centering, as in released RLPolicy.
    Kernel origin must be exactly zero, excluding the found source from every
    nonfound branch. Mechanical kernels need not sum to one across categories.
    No posterior repair, weight renormalization or extra mask is applied.
    Eligible IDs are an unordered set supplied as a unique sequence; selection
    always uses original numeric action order, not the sequence's input order.
    """
    belief = _float64(belief, (N, N), "belief", nonnegative=True)
    if belief.sum(dtype=np.float64) > 1 + 1e-6:
        raise ValueError("belief mass exceeds the public probability contract")
    kernel = _float64(kernel, (4, 107, 107), "kernel", nonnegative=True)
    if (kernel > 1).any() or (kernel[:, N, N] != 0).any():
        raise ValueError("kernel must be in [0,1] with an exactly zero origin")
    position, eligible = _position(position), _eligible(eligible_actions)
    unnormalized = np.zeros((16, CENTERED, CENTERED), dtype=np.float64)
    normalized = np.zeros_like(unnormalized)
    masses, weights = np.empty((4, 4)), np.empty((4, 4))
    successors = np.empty((16, 2), dtype=np.int64)
    for action in range(4):
        moved = list(position)
        axis, direction = action // 2, 2 * (action % 2) - 1
        moved[axis] = min(N - 1, max(0, moved[axis] + direction))
        x, y = moved
        crop = kernel[:, N - x:2 * N - x, N - y:2 * N - y]
        for hit in range(4):
            index = 4 * action + hit
            joint = belief * crop[hit]
            mass = np.sum(joint, dtype=np.float64)
            weight = np.maximum(EPSILON, mass)
            target = (index, slice(N - 1 - x, 2 * N - 1 - x), slice(N - 1 - y, 2 * N - 1 - y))
            unnormalized[target], normalized[target] = joint, joint / weight
            masses[action, hit], weights[action, hit] = mass, weight
            successors[index] = moved
    result = object.__new__(RLBranches)
    for name, array in (("centered_u", unnormalized), ("centered_z", normalized),
                        ("raw_masses", masses), ("weights", weights),
                        ("successors", successors), ("kernel", kernel)):
        object.__setattr__(result, name, _owned(array))
    object.__setattr__(result, "eligible_actions", eligible)
    return result


def _scores(branches, values, dtype, *, weighted):
    if not isinstance(values, np.ndarray) or values.shape != (16,) or values.dtype != dtype:
        raise ValueError(f"value function must return {dtype.name}[16]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite branch values")
    with np.errstate(over="raise", invalid="raise"):
        terms = values.reshape(4, 4)
        if weighted:
            terms = branches.weights.astype(dtype) * terms
        scores = dtype.type(1) + np.sum(terms, axis=1, dtype=dtype)
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite action scores")
    return _owned(scores)


def explicit_scores(branches, value, *, arithmetic="float64"):
    """Evaluate one scalar callback on all branches, then retain all four costs.

    value(centered_z, successors, kernel) must return an array of shape (16,)
    and the named arithmetic dtype. The known kernel remains float64. Inputs
    are immutable. A biased value at a zero input is retained, including its
    positive floored weight. No terminal or zero-input shortcut is taken.
    """
    if not isinstance(branches, RLBranches):
        raise TypeError("branches must come from rl_branches")
    dtype = _arithmetic(arithmetic)
    inputs = branches.centered_z if dtype == np.float64 else _owned(branches.centered_z.astype(dtype))
    values = value(inputs, branches.successors, branches.kernel)
    return _scores(branches, values, dtype, weighted=True)


def min_linear_scores(branches, coefficients, *, route="explicit", arithmetic="float64"):
    """Minimum of no-bias linear values, evaluated explicitly or by homogeneity.

    coefficients is float64[J,105,105] or float64[4,J,105,105], J in 1..64.
    The latter bank may be conditioned on each public successor position and
    known kernel, but must be independent of belief and shared across its four
    hit outcomes. That semantic independence is the caller's responsibility.
    Signed coefficients are accepted; nonnegativity/terminal constraints are
    not inferred from the representation. Fused float32 results need not match
    explicit float32 results or selected actions.
    """
    if not isinstance(branches, RLBranches):
        raise TypeError("branches must come from rl_branches")
    if route not in ("explicit", "fused"):
        raise ValueError("route must be explicit or fused")
    dtype = _arithmetic(arithmetic)
    if not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.float64:
        raise ValueError("coefficients must be a float64 ndarray")
    if (coefficients.ndim not in (3, 4) or coefficients.shape[-2:] != (CENTERED, CENTERED)
            or not 1 <= coefficients.shape[-3] <= MAX_TEMPLATES
            or (coefficients.ndim == 4 and coefficients.shape[0] != 4)
            or not np.isfinite(coefficients).all()):
        raise ValueError("invalid coefficient bank")
    values = np.empty(16, dtype=dtype)
    fields = branches.centered_z if route == "explicit" else branches.centered_u
    with np.errstate(over="raise", invalid="raise"):
        alpha = coefficients.astype(dtype, copy=False)
        for action in range(4):
            bank = alpha if alpha.ndim == 3 else alpha[action]
            for hit in range(4):
                index = 4 * action + hit
                products = bank * fields[index].astype(dtype, copy=False)
                values[index] = np.min(np.sum(products, axis=(1, 2), dtype=dtype))
    return _scores(branches, values, dtype, weighted=route == "explicit")


def select_action(scores, eligible_actions):
    """First numeric action ID strictly within 1e-10 of the eligible minimum.

    The dtype of subtraction is preserved, including float32 rounding. Raw
    costs are neither overwritten by a mask nor clipped or repaired.
    Eligible IDs are treated as an unordered set, with duplicates rejected.
    """
    if (not isinstance(scores, np.ndarray) or scores.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or scores.shape != (4,) or not np.isfinite(scores).all()):
        raise ValueError("scores must be finite float32/float64[4]")
    eligible = _eligible(eligible_actions)
    permitted = np.asarray([a in eligible for a in range(4)], dtype=bool)
    best = np.min(scores[permitted])
    choices = np.flatnonzero(permitted & (np.abs(scores - best) < EPSILON))
    return int(choices[0])
