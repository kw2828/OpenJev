"""Certificate-driven projected optimization for an existing frozen-state head.

The objective and direct-residual certificate are those of
finite_convex_readout.Problem: blind MSE plus observed MSE, with no factor 1/2
between routes. P has eight four-action simplex columns and C=.25-P.

This is conventional FISTA with a fixed safe Lipschitz bound and a monotonic
restart. It is not a new scientific run or an admission policy. A candidate
whose quadratic exceeds the current value by more than 1e-15 triggers a plain
projected-gradient retry from the current point. If that step also increases,
the solve fails explicitly; there is no backtracking, parameter tuning or
export repair. Certificate checks, not a small update, determine convergence.

Every Gram gradient/value, update projection, direct-residual pass, callback
and restart is counted. The direct certificate is checked at iteration zero,
every ten accepted iterations, and the final retained iterate, without a
duplicate check at the same iteration. Zero curvature requires no update.
The returned certificate is numerical float64 evidence, not an interval proof.
Input arrays and random state are unchanged. No files, models or SciPy calls.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from openjev.research import finite_convex_readout as reference

VERSION = "finite-gap-readout-v1"
METHOD = "fixed-step-monotone-restarted-fista"
MAX_ITERATIONS = 20000
CERT_EVERY = 10
FW_GAP_TOLERANCE = 1e-8
SIMPLEX_TOLERANCE = 1e-12
OBJECTIVE_INCREASE_TOLERANCE = 1e-12
MONOTONICITY_ROUNDOFF = 1e-15


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _point(value):
    raw = np.asarray(value)
    _require(raw.dtype == np.float64 and raw.shape == (4, 8)
             and bool(np.isfinite(raw).all()), "finite float64 P[4,8] required")
    return np.array(raw, dtype=np.float64, order="C", copy=True)


def solve(problem: reference.Problem, initial_probabilities, *,
          check: Callable[[], None] = lambda: None) -> dict:
    """One fixed-budget solve; failed results retain the current feasible head.

    check() exceptions propagate. Numerical overflow/nonfinite arithmetic
    raises ValueError rather than yielding a misleading result. A negative
    direct FW gap below -1e-12 fails, matching the existing certificate guard.
    There is no post-stop projection: raw_probabilities and probabilities are
    equal copies of the retained iterate, and projection.applied is False.
    """
    _require(type(problem) is reference.Problem, "existing build_problem result required")
    _require(callable(check), "check must be callable")
    _require(type(MAX_ITERATIONS) is int and MAX_ITERATIONS >= 0
             and type(CERT_EVERY) is int and CERT_EVERY > 0, "valid fixed iteration settings")
    initial = _point(initial_probabilities)
    _require(reference.simplex_violation(initial) <= SIMPLEX_TOLERANCE
             and bool((initial >= 0).all()), "initial P must already be feasible")
    work = dict.fromkeys(("gradient_calls", "quadratic_value_calls", "projection_calls",
                          "direct_objective_gradient_passes", "check_calls", "monotonicity_restarts",
                          "iterations", "accepted_iterations"), 0)

    def bounded():
        work["check_calls"] += 1
        check()

    def value(p):
        bounded()
        work["quadratic_value_calls"] += 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = float(np.sum((p @ problem.gram) * p)
                               - 2 * np.sum(p * problem.cross) + problem.constant)
        except FloatingPointError as error:
            raise ValueError("nonfinite quadratic value") from error
        _require(math.isfinite(result), "finite quadratic value required")
        return result

    def projected_step(p):
        bounded()
        work["gradient_calls"] += 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                gradient = 2 * (p @ problem.gram - problem.cross)
                candidate = p - gradient / lipschitz
        except FloatingPointError as error:
            raise ValueError("nonfinite fixed-step gradient arithmetic") from error
        _require(bool(np.isfinite(gradient).all() and np.isfinite(candidate).all()), "finite projected-gradient argument")
        bounded()
        work["projection_calls"] += 1
        result = reference.project_simplex(candidate)
        _require(reference.simplex_violation(result) <= SIMPLEX_TOLERANCE,
                 "update projection did not produce a feasible point")
        return result

    history = []
    objective_initial = None

    def assess(p, iteration):
        bounded()
        work["direct_objective_gradient_passes"] += 1
        result = reference.certificate(problem, p)
        # Fixed local conditions prevent an accidental change in imported
        # certificate thresholds from silently changing this solver's stop rule.
        certified = (result["simplex_violation"] <= SIMPLEX_TOLERANCE
                     and -SIMPLEX_TOLERANCE <= result["fw_gap"] <= FW_GAP_TOLERANCE)
        nonincreasing = (objective_initial is None
                         or result["objective"] <= objective_initial + OBJECTIVE_INCREASE_TOLERANCE)
        history.append({"iteration": iteration, "objective": result["objective"],
                        "fw_gap": result["fw_gap"], "simplex_violation": result["simplex_violation"],
                        "certificate_passed": bool(certified), "objective_nonincreasing": bool(nonincreasing),
                        "qualifies": bool(certified and nonincreasing)})
        return result, bool(certified and nonincreasing)

    bounded()
    try:
        with np.errstate(over="raise", invalid="raise"):
            lipschitz = float(2 * np.max(np.sum(np.abs(problem.gram), axis=1)))
    except FloatingPointError as error:
        raise ValueError("nonfinite Gram curvature bound") from error
    _require(math.isfinite(lipschitz) and lipschitz >= 0, "finite nonnegative Gram curvature bound")
    current, extrapolated, t = initial.copy(), initial.copy(), 1.
    final_certificate, qualified = assess(current, 0)
    objective_initial = final_certificate["objective"]
    checked_iteration = 0
    numerical_failure = False
    termination = "initial_certificate" if qualified else "iteration_budget"
    if not qualified and lipschitz == 0:
        termination = "zero_curvature_uncertified"
    elif not qualified and MAX_ITERATIONS:
        current_value = value(current)
        for iteration in range(1, MAX_ITERATIONS + 1):
            bounded()
            work["iterations"] += 1
            candidate = projected_step(extrapolated)
            candidate_value = value(candidate)
            if candidate_value > current_value + MONOTONICITY_ROUNDOFF:
                work["monotonicity_restarts"] += 1
                t = 1.
                extrapolated = current.copy()
                candidate = projected_step(extrapolated)
                candidate_value = value(candidate)
                if candidate_value > current_value + MONOTONICITY_ROUNDOFF:
                    numerical_failure = True
                    termination = "plain_step_nonmonotone"
                    break
            previous = current
            current, current_value = candidate, candidate_value
            work["accepted_iterations"] += 1
            if iteration % CERT_EVERY == 0:
                final_certificate, qualified = assess(current, iteration)
                checked_iteration = iteration
                if qualified:
                    termination = "direct_certificate"
                    break
            next_t = (1 + math.sqrt(1 + 4 * t * t)) / 2
            extrapolated = current + ((t - 1) / next_t) * (current - previous)
            _require(bool(np.isfinite(extrapolated).all()), "finite FISTA extrapolation")
            t = next_t
    retained_iteration = work["accepted_iterations"]
    if checked_iteration != retained_iteration:
        final_certificate, qualified = assess(current, retained_iteration)
        if qualified and not numerical_failure:
            termination = "final_direct_certificate"
    complete = qualified and not numerical_failure
    if numerical_failure:
        reasons = ["plain_step_nonmonotone"]
    elif not qualified:
        reasons = [termination, "final_numerical_certificate"]
    else:
        reasons = []
    bounded()
    # No export repair, projection or replacement occurs after the final check.
    return {"version": VERSION, "method": METHOD, "complete": bool(complete),
        "status": "SOLVE_PASS" if complete else "SOLVE_FAIL", "failure_reasons": reasons,
        "initial_probabilities": initial, "raw_probabilities": current.copy(),
        "probabilities": current.copy(), "cost_matrix": .25 - current,
        "objective_initial": objective_initial, "objective_raw": final_certificate["objective"],
        "raw_feasibility": final_certificate["simplex_violation"],
        "projection": {"applied": False, "max_abs": 0., "l2": 0.},
        "certificate": final_certificate, "certificate_history": history,
        "lipschitz": lipschitz, "termination": termination,
        "solver": {"success": bool(complete), "status": 0 if complete else 1,
                   "message": termination, "iterations": work["iterations"],
                   "nfev": work["quadratic_value_calls"], "njev": work["gradient_calls"]},
        "settings": {"max_iterations": MAX_ITERATIONS, "certificate_every": CERT_EVERY,
                     "fw_gap_tolerance": FW_GAP_TOLERANCE, "simplex_tolerance": SIMPLEX_TOLERANCE,
                     "objective_increase_tolerance": OBJECTIVE_INCREASE_TOLERANCE,
                     "monotonicity_roundoff": MONOTONICITY_ROUNDOFF},
        "work": work}
