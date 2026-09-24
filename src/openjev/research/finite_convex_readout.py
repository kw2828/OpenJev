"""A bounded convex diagnostic for a frozen eight-state decision readout.

For C = .25 - P, minimize blind cost MSE plus observed cost MSE, each over
N*H*4 entries. Absorbed zero states remain in both denominators. P has four
nonnegative entries per column summing to one. This closed simplex extends the
finite-softmax head's open domain; improvements cannot isolate optimizer choice.

SLSQP uses a Gram quadratic and analytic gradient. Its single returned point is
projected once onto each simplex. Raw feasibility and the maximum repair must
both be <=1e-10. A successful SciPy exit is necessary but insufficient: the
projected point must satisfy the direct-residual Frank-Wolfe gap and objective
checks. The gap is a floating-point numerical certificate, not an interval
proof. Failed results are returned without retries or fallback promotion.

No file, model, generator or admission operations occur here. The caller owns
the phase deadline via check(); exceptions from that callback propagate. NumPy
inputs are copied and no random state is consumed. SciPy is imported only when
solve is called. The fixed 32-coordinate solve adds no regularization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

VERSION = "finite-convex-readout-v1"
MAX_ITERATIONS = 2000
MAX_OBJECTIVE_CALLS = 10000
FTOL = 1e-12
RAW_FEASIBILITY_TOLERANCE = 1e-10
PROJECTION_TOLERANCE = 1e-10
SIMPLEX_TOLERANCE = 1e-12
FW_GAP_TOLERANCE = 1e-8
OBJECTIVE_INCREASE_TOLERANCE = 1e-12
STATE_DIM = 8
ACTION_DIM = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _array(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype == np.float64, f"{name}: float64 required")
    _require(shape is None or raw.shape == shape, f"{name}: incorrect shape")
    _require(bool(np.isfinite(raw).all()), f"{name}: finite values required")
    return np.array(raw, dtype=np.float64, order="C", copy=True)


@dataclass(frozen=True)
class Problem:
    """Owned, read-only raw arrays and their quadratic sufficient statistics."""

    blind_states: np.ndarray
    observed_states: np.ndarray
    blind_targets: np.ndarray
    observed_targets: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    constant: float
    denominator: int
    cases: int
    horizon: int


def build_problem(blind_states, observed_states, blind_targets, observed_targets) -> Problem:
    """Validate and own [N,H,8] masses and centered [N,H,4] targets.

    States are nonnegative subprobabilities; normalized observed states and
    absorbed zeros are both allowed. Targets must be centered to 1e-12. No
    unsupported row is dropped, and no clipping or recentering is performed.
    """
    xb = _array(blind_states, "blind_states")
    _require(xb.ndim == 3 and xb.shape[0] > 0 and xb.shape[1] > 0
             and xb.shape[2] == STATE_DIM, "states: nonempty [N,H,8]")
    n, h, _ = xb.shape
    xo = _array(observed_states, "observed_states", xb.shape)
    yb = _array(blind_targets, "blind_targets", (n, h, ACTION_DIM))
    yo = _array(observed_targets, "observed_targets", yb.shape)
    for name, x in (("blind", xb), ("observed", xo)):
        _require(bool((x >= 0).all()), f"{name} states: nonnegative")
        _require(bool((x.sum(axis=-1) <= 1 + SIMPLEX_TOLERANCE).all()),
                 f"{name} states: mass cannot exceed one")
    for name, y in (("blind", yb), ("observed", yo)):
        _require(bool((np.abs(y.sum(axis=-1)) <= SIMPLEX_TOLERANCE).all()),
                 f"{name} targets: centered action costs required")
    denominator = n * h * ACTION_DIM
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            flat_b, flat_o = xb.reshape(-1, STATE_DIM), xo.reshape(-1, STATE_DIM)
            zb = .25 * flat_b.sum(axis=1, keepdims=True) - yb.reshape(-1, ACTION_DIM)
            zo = .25 * flat_o.sum(axis=1, keepdims=True) - yo.reshape(-1, ACTION_DIM)
            gram = (flat_b.T @ flat_b + flat_o.T @ flat_o) / denominator
            cross = (zb.T @ flat_b + zo.T @ flat_o) / denominator
            constant = float((np.sum(zb * zb) + np.sum(zo * zo)) / denominator)
    except FloatingPointError as error:
        raise ValueError("nonfinite quadratic construction") from error
    for x in (xb, xo, yb, yo, gram, cross):
        _require(bool(np.isfinite(x).all()), "finite quadratic statistics required")
        x.setflags(write=False)
    _require(np.isfinite(constant), "finite quadratic constant required")
    return Problem(xb, xo, yb, yo, gram, cross, constant, denominator, n, h)


def objective_gradient(problem: Problem, probabilities) -> tuple[float, np.ndarray]:
    """Direct full-residual objective and derivative with respect to P.

    This routine does not require P to be feasible, enabling raw-result
    diagnostics. It deliberately does not evaluate the Gram polynomial.
    """
    _require(type(problem) is Problem, "build_problem result required")
    p = _array(probabilities, "probabilities", (ACTION_DIM, STATE_DIM))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            c = .25 - p
            xb = problem.blind_states.reshape(-1, STATE_DIM)
            xo = problem.observed_states.reshape(-1, STATE_DIM)
            eb = xb @ c.T - problem.blind_targets.reshape(-1, ACTION_DIM)
            eo = xo @ c.T - problem.observed_targets.reshape(-1, ACTION_DIM)
            value = float((np.sum(eb * eb) + np.sum(eo * eo)) / problem.denominator)
            gradient = -2 * (eb.T @ xb + eo.T @ xo) / problem.denominator
    except FloatingPointError as error:
        raise ValueError("nonfinite direct objective arithmetic") from error
    _require(np.isfinite(value) and bool(np.isfinite(gradient).all()),
             "finite objective and gradient required")
    return value, gradient


def _quadratic(problem: Problem, p: np.ndarray) -> tuple[float, np.ndarray]:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pg = p @ problem.gram
        value = float(np.sum(pg * p) - 2 * np.sum(p * problem.cross) + problem.constant)
        gradient = 2 * (pg - problem.cross)
    _require(np.isfinite(value) and bool(np.isfinite(gradient).all()),
             "finite solver quadratic required")
    return value, gradient


def simplex_violation(probabilities) -> float:
    """Maximum absolute column-sum error or negative-entry magnitude."""
    p = _array(probabilities, "probabilities", (ACTION_DIM, STATE_DIM))
    return max(float(np.max(np.abs(p.sum(axis=0) - 1))),
               max(0., -float(p.min())))


def project_simplex(probabilities) -> np.ndarray:
    """One deterministic Euclidean projection per column, without retries.

    Sorting is stable for reproducibility; no second renormalization or
    rounding correction is applied. Final feasibility is checked separately.
    """
    p = _array(probabilities, "probabilities", (ACTION_DIM, STATE_DIM))
    result = np.empty_like(p)
    for column in range(STATE_DIM):
        u = np.sort(p[:, column], kind="stable")[::-1]
        sums = np.cumsum(u) - 1
        valid = u - sums / np.arange(1, ACTION_DIM + 1) > 0
        _require(bool(valid.any()), "simplex projection failed")
        rho = int(np.flatnonzero(valid)[-1])
        theta = sums[rho] / (rho + 1)
        result[:, column] = np.maximum(p[:, column] - theta, 0.)
    _require(bool(np.isfinite(result).all()), "finite simplex projection required")
    return result


def certificate(problem: Problem, probabilities) -> dict:
    """Feasibility and direct-gradient Frank-Wolfe numerical certificate.

    For a feasible P, gap = <P,grad> - sum_s min_a grad[a,s]. The raw signed
    floating-point gap is retained; no negative value is silently clipped.
    A negative value below -1e-12 fails the numerical consistency check.
    """
    p = _array(probabilities, "probabilities", (ACTION_DIM, STATE_DIM))
    value, gradient = objective_gradient(problem, p)
    gap = float(np.sum(p * gradient) - np.sum(gradient.min(axis=0)))
    violation = simplex_violation(p)
    passed = (violation <= SIMPLEX_TOLERANCE
              and -SIMPLEX_TOLERANCE <= gap <= FW_GAP_TOLERANCE)
    return {"objective": value, "gradient": gradient, "fw_gap": gap,
            "simplex_violation": violation, "minimum_probability": float(p.min()),
            "maximum_probability": float(p.max()), "passed": bool(passed)}


class _ObjectiveBudget(Exception):
    pass


def solve(problem: Problem, initial_probabilities, *, check: Callable[[], None] = lambda: None) -> dict:
    """One SLSQP attempt and one projection, retaining failed numerical results.

    The analytic function returns both loss and gradient, so objective_calls
    and gradient_calls count the same evaluations. Direct residual passes for
    initial, raw and projected points are separately disclosed. A budget stop
    retains the last attempted point with unsuccessful solver status. External
    callback failures and malformed/nonfinite solver outputs raise, allowing
    the original process supervisor to preserve the technical failure.
    """
    _require(type(problem) is Problem, "build_problem result required")
    _require(callable(check), "check must be callable")
    initial = _array(initial_probabilities, "initial_probabilities", (ACTION_DIM, STATE_DIM))
    _require(simplex_violation(initial) <= SIMPLEX_TOLERANCE,
             "initial probabilities must already be feasible")
    check()
    from scipy.optimize import minimize

    work = {"objective_calls": 0, "gradient_calls": 0, "iteration_callbacks": 0,
            "direct_objective_gradient_passes": 0, "simplex_projection_calls": 0}
    objective_initial, _ = objective_gradient(problem, initial)
    work["direct_objective_gradient_passes"] += 1
    last = initial.copy()

    def value_and_gradient(flat):
        nonlocal last
        check()
        if work["objective_calls"] >= MAX_OBJECTIVE_CALLS:
            raise _ObjectiveBudget
        p = _array(np.asarray(flat).reshape(ACTION_DIM, STATE_DIM), "solver point",
                   (ACTION_DIM, STATE_DIM))
        last = p.copy()
        work["objective_calls"] += 1
        work["gradient_calls"] += 1
        value, gradient = _quadratic(problem, p)
        return value, gradient.ravel()

    def callback(flat):
        nonlocal last
        check()
        last = _array(np.asarray(flat).reshape(ACTION_DIM, STATE_DIM), "solver iterate",
                      (ACTION_DIM, STATE_DIM))
        work["iteration_callbacks"] += 1

    equality_jacobian = np.tile(np.eye(STATE_DIM, dtype=np.float64), (1, ACTION_DIM))

    def equality(flat):
        check()
        return flat.reshape(ACTION_DIM, STATE_DIM).sum(axis=0) - 1

    def equality_jac(flat):
        check()
        return equality_jacobian.copy()

    try:
        result = minimize(value_and_gradient, initial.ravel().copy(), method="SLSQP", jac=True,
                          bounds=[(0., 1.)] * (ACTION_DIM * STATE_DIM),
                          constraints={"type": "eq", "fun": equality, "jac": equality_jac},
                          callback=callback,
                          options={"maxiter": MAX_ITERATIONS, "ftol": FTOL, "disp": False})
        raw = _array(np.asarray(result.x).reshape(ACTION_DIM, STATE_DIM), "solver result",
                     (ACTION_DIM, STATE_DIM))
        solver = {"success": bool(result.success), "status": int(result.status),
                  "message": str(result.message), "iterations": int(result.nit),
                  "nfev": int(result.nfev), "njev": int(result.njev)}
    except _ObjectiveBudget:
        raw = last.copy()
        solver = {"success": False, "status": -10000,
                  "message": "fixed objective-call budget exhausted; no retry",
                  "iterations": work["iteration_callbacks"],
                  "nfev": work["objective_calls"], "njev": work["gradient_calls"]}
    check()
    raw_feasibility = simplex_violation(raw)
    objective_raw, _ = objective_gradient(problem, raw)
    work["direct_objective_gradient_passes"] += 1
    final = project_simplex(raw)
    work["simplex_projection_calls"] += 1
    difference = final - raw
    projection = {"applied": True, "max_abs": float(np.abs(difference).max()),
                  "l2": float(np.linalg.norm(difference))}
    final_certificate = certificate(problem, final)
    work["direct_objective_gradient_passes"] += 1
    failures = []
    if not solver["success"]:
        failures.append("scipy_unsuccessful")
    if raw_feasibility > RAW_FEASIBILITY_TOLERANCE:
        failures.append("raw_simplex_violation")
    if projection["max_abs"] > PROJECTION_TOLERANCE:
        failures.append("projection_repair_exceeds_tolerance")
    if not final_certificate["passed"]:
        failures.append("final_numerical_certificate")
    if final_certificate["objective"] > objective_initial + OBJECTIVE_INCREASE_TOLERANCE:
        failures.append("objective_increased")
    check()
    return {"version": VERSION, "complete": not failures,
            "status": "SOLVE_PASS" if not failures else "SOLVE_FAIL",
            "failure_reasons": failures, "initial_probabilities": initial,
            "raw_probabilities": raw, "probabilities": final, "cost_matrix": .25 - final,
            "objective_initial": objective_initial, "objective_raw": objective_raw,
            "raw_feasibility": raw_feasibility, "projection": projection,
            "certificate": final_certificate, "solver": solver, "work": work}
