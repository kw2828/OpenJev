"""Fabricated convex-head qualification only; no study seeds or data files."""

from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import finite_convex_readout as module


def fixture():
    states = np.eye(8, dtype=np.float64)[:, None, :]
    probability = np.full((4, 8), .1, dtype=np.float64)
    probability[np.arange(8) % 4, np.arange(8)] = .7
    targets = ((.25 - probability) @ states[:, 0, :].T).T[:, None, :]
    problem = module.build_problem(states, states, targets, targets)
    return problem, probability


def scalar_oracle(problem, p):
    """Loop over both complete routes with their original per-route divisor."""
    objective = 0.
    gradient = np.zeros((4, 8), dtype=np.float64)
    for states, targets in ((problem.blind_states, problem.blind_targets),
                            (problem.observed_states, problem.observed_targets)):
        for case in range(problem.cases):
            for horizon in range(problem.horizon):
                for action in range(4):
                    prediction = sum((.25 - p[action, state]) * states[case, horizon, state]
                                     for state in range(8))
                    error = prediction - targets[case, horizon, action]
                    objective += error * error / problem.denominator
                    for state in range(8):
                        gradient[action, state] -= (
                            2 * error * states[case, horizon, state] / problem.denominator)
    return objective, gradient


def test_direct_objective_preserves_sum_of_route_means_and_zero_rows():
    x = np.zeros((2, 1, 8), dtype=np.float64)
    x[0, 0, 0] = 1
    y = np.zeros((2, 1, 4), dtype=np.float64)
    y[0, 0] = [-.75, .25, .25, .25]
    problem = module.build_problem(x, x, y, y)
    value, gradient = module.objective_gradient(problem, np.full((4, 8), .25))
    assert problem.denominator == 8
    assert value == 3 / 16  # Two .75 squared-residual sums, divided by eight.
    np.testing.assert_array_equal(gradient[:, 0], [-.375, .125, .125, .125])
    np.testing.assert_array_equal(gradient[:, 1:], np.zeros((4, 7)))


def test_quadratic_and_analytic_gradient_match_independent_raw_oracle():
    rng = np.random.default_rng(935001)
    xb = rng.dirichlet(np.ones(8), size=(3, 2)).astype(np.float64) * .8
    xo = rng.dirichlet(np.ones(8), size=(3, 2)).astype(np.float64)
    yb, yo = rng.normal(size=(2, 3, 2, 4))
    yb -= yb.mean(axis=-1, keepdims=True)
    yo -= yo.mean(axis=-1, keepdims=True)
    problem = module.build_problem(xb, xo, yb, yo)
    p = rng.dirichlet(np.ones(4), size=8).T.copy()
    value, gradient = module.objective_gradient(problem, p)
    expected, expected_gradient = scalar_oracle(problem, p)
    quadratic, quadratic_gradient = module._quadratic(problem, p)
    assert value == pytest.approx(expected, abs=2e-15)
    assert quadratic == pytest.approx(expected, abs=2e-15)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-16, rtol=2e-15)
    np.testing.assert_allclose(quadratic_gradient, expected_gradient, atol=2e-16, rtol=2e-15)
    direction = rng.normal(size=(4, 8))
    step = 1e-5
    plus = module.objective_gradient(problem, p + step * direction)[0]
    minus = module.objective_gradient(problem, p - step * direction)[0]
    assert (plus - minus) / (2 * step) == pytest.approx(np.sum(gradient * direction), abs=1e-10)


def test_solve_identifies_unique_interior_optimum_and_discloses_work():
    problem, expected = fixture()
    initial = np.full((4, 8), .25, dtype=np.float64)
    result = module.solve(problem, initial)
    assert result["complete"] and result["status"] == "SOLVE_PASS"
    assert result["failure_reasons"] == []
    assert result["solver"]["success"]
    np.testing.assert_allclose(result["probabilities"], expected, atol=1e-8, rtol=0)
    assert result["certificate"]["objective"] < 1e-15
    assert result["certificate"]["fw_gap"] <= module.FW_GAP_TOLERANCE
    assert result["work"]["simplex_projection_calls"] == 1
    assert result["work"]["direct_objective_gradient_passes"] == 3
    assert result["work"]["objective_calls"] == result["work"]["gradient_calls"] > 0
    np.testing.assert_array_equal(initial, np.full((4, 8), .25))


def test_closed_simplex_boundary_optimum_and_linear_mass_readout():
    x = np.eye(8, dtype=np.float64)[:, None, :]
    vertices = np.zeros((4, 8), dtype=np.float64)
    vertices[np.arange(8) % 4, np.arange(8)] = 1
    y = (.25 - vertices).T[:, None, :]
    problem = module.build_problem(x, x, y, y)
    result = module.solve(problem, np.full((4, 8), .25))
    assert result["complete"]
    np.testing.assert_allclose(result["probabilities"], vertices, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result["cost_matrix"].sum(axis=0), 0., atol=1e-15)
    np.testing.assert_array_equal(result["cost_matrix"] @ np.zeros(8), np.zeros(4))


def test_rank_deficiency_and_zero_support_have_no_hidden_regularizer():
    x = np.zeros((2, 3, 8), dtype=np.float64)
    y = np.zeros((2, 3, 4), dtype=np.float64)
    problem = module.build_problem(x, x, y, y)
    initial = np.tile(np.array([.5, .25, .125, .125])[:, None], (1, 8))
    result = module.solve(problem, initial)
    assert result["complete"]
    np.testing.assert_array_equal(result["probabilities"], initial)
    assert result["certificate"]["objective"] == 0
    assert result["certificate"]["fw_gap"] == 0
    assert problem.denominator == 24


def test_ill_conditioned_shared_mass_with_absorbed_rows_has_direct_certificate():
    # The same 95% dominant state hides small but identifiable distinctions.
    # Every second horizon is absorbed and stays in the objective denominator.
    states = np.zeros((8, 2, 8), dtype=np.float64)
    states[:, 0, 0] = .95
    states[:, 0, :] += .05 * np.eye(8, dtype=np.float64)
    optimum = np.full((4, 8), .125, dtype=np.float64)
    optimum[np.arange(8) % 4, np.arange(8)] = .625
    targets = states @ (.25 - optimum).T
    problem = module.build_problem(states, states, targets, targets)
    assert np.linalg.cond(problem.gram) > 1000
    initial = np.full((4, 8), .25, dtype=np.float64)
    assert module.certificate(problem, initial)["fw_gap"] > .001
    result = module.solve(problem, initial)
    assert result["complete"]
    direct_loss, direct_gradient = scalar_oracle(problem, result["probabilities"])
    direct_gap = (sum(float(result["probabilities"][a, s] * direct_gradient[a, s])
                      for a in range(4) for s in range(8))
                  - sum(min(float(direct_gradient[a, s]) for a in range(4))
                        for s in range(8)))
    assert direct_gap <= 1e-8
    assert direct_gap >= -1e-12
    assert direct_loss <= result["objective_initial"] + 1e-12
    assert result["certificate"]["objective"] == pytest.approx(direct_loss, abs=1e-16)
    assert result["certificate"]["fw_gap"] == pytest.approx(direct_gap, abs=1e-15)
    np.testing.assert_array_equal(states[:, 1], np.zeros((8, 8)))
    assert problem.denominator == 64


def test_certificate_gap_is_a_valid_bound_at_a_known_feasible_point():
    problem, optimum = fixture()
    p = np.full((4, 8), .25)
    certificate = module.certificate(problem, p)
    optimum_value = module.objective_gradient(problem, optimum)[0]
    assert not certificate["passed"]
    assert certificate["fw_gap"] >= certificate["objective"] - optimum_value > 0
    assert module.certificate(problem, optimum)["passed"]
    # A zero gradient cannot make an infeasible point certified.
    zeros = np.zeros((1, 1, 8), dtype=np.float64)
    zero_problem = module.build_problem(zeros, zeros, np.zeros((1, 1, 4)), np.zeros((1, 1, 4)))
    assert not module.certificate(zero_problem, np.zeros((4, 8)))["passed"]


def fake_result(p, *, success=True):
    return SimpleNamespace(x=p.ravel().copy(), success=success, status=0 if success else 9,
                           message="fabricated", nit=1, nfev=0, njev=0)


def test_unsuccessful_solver_is_not_promoted_even_at_exact_optimum(monkeypatch):
    import scipy.optimize

    problem, optimum = fixture()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        return fake_result(optimum, success=False)

    monkeypatch.setattr(scipy.optimize, "minimize", fail)
    result = module.solve(problem, np.full((4, 8), .25))
    assert not result["complete"]
    assert result["certificate"]["passed"]
    assert result["failure_reasons"] == ["scipy_unsuccessful"]
    np.testing.assert_array_equal(result["raw_probabilities"], optimum)
    assert len(calls) == 1


def test_tiny_projection_is_reported_but_gross_repair_fails(monkeypatch):
    import scipy.optimize

    problem, optimum = fixture()
    tiny = optimum + 1e-12
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: fake_result(tiny))
    result = module.solve(problem, np.full((4, 8), .25))
    assert result["complete"]
    assert 0 < result["projection"]["max_abs"] <= module.PROJECTION_TOLERANCE
    assert result["raw_feasibility"] > 0
    np.testing.assert_allclose(result["probabilities"], optimum, atol=2e-16)
    gross = optimum + .01
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: fake_result(gross))
    result = module.solve(problem, np.full((4, 8), .25))
    assert not result["complete"]
    assert "raw_simplex_violation" in result["failure_reasons"]
    assert "projection_repair_exceeds_tolerance" in result["failure_reasons"]
    assert result["work"]["simplex_projection_calls"] == 1


def test_projection_is_owned_equivariant_and_has_hand_calculated_solution():
    p = np.tile(np.array([-.1, .2, .3, .9])[:, None], (1, 8))
    before = p.copy()
    projected = module.project_simplex(p)
    np.testing.assert_allclose(projected[:, 0], [0., 1 / 15, 1 / 6, 23 / 30], atol=2e-16)
    order = np.array([2, 0, 3, 1])
    np.testing.assert_array_equal(module.project_simplex(p[order]), projected[order])
    np.testing.assert_array_equal(p, before)
    assert not np.shares_memory(p, projected)


def test_fixed_budget_stops_once_and_retains_failure(monkeypatch):
    import scipy.optimize

    problem, _ = fixture()
    calls = []

    def consume(function, x, **kwargs):
        calls.append(1)
        for _ in range(module.MAX_OBJECTIVE_CALLS + 1):
            function(x)
        pytest.fail("fixed objective budget did not stop")

    monkeypatch.setattr(scipy.optimize, "minimize", consume)
    result = module.solve(problem, np.full((4, 8), .25))
    assert not result["complete"]
    assert result["solver"]["status"] == -10000
    assert result["work"]["objective_calls"] == 10000
    assert result["work"]["gradient_calls"] == 10000
    assert len(calls) == 1


def test_callback_deadline_failure_propagates_without_retry(monkeypatch):
    import scipy.optimize

    problem, _ = fixture()
    invocations = []

    def run(function, x, **kwargs):
        invocations.append(1)
        function(x)
        kwargs["callback"](x)
        pytest.fail("deadline did not propagate")

    checks = []

    def check():
        checks.append(1)
        if len(checks) == 3:
            raise TimeoutError("fabricated phase deadline")

    monkeypatch.setattr(scipy.optimize, "minimize", run)
    with pytest.raises(TimeoutError, match="fabricated phase deadline"):
        module.solve(problem, np.full((4, 8), .25), check=check)
    assert len(invocations) == 1


def test_unsuccessful_certificate_and_objective_increase_are_distinct(monkeypatch):
    import scipy.optimize

    problem, optimum = fixture()
    uniform = np.full((4, 8), .25)
    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: fake_result(uniform))
    result = module.solve(problem, optimum)
    assert not result["complete"]
    assert "final_numerical_certificate" in result["failure_reasons"]
    assert "objective_increased" in result["failure_reasons"]


@pytest.mark.parametrize("change", ["float32", "shape", "empty", "nan", "negative",
                                   "excess_mass", "uncentered", "inf_target"])
def test_invalid_problem_inputs_fail(change):
    x = np.full((2, 1, 8), .125)
    y = np.zeros((2, 1, 4))
    xb, xo, yb, yo = x.copy(), x.copy(), y.copy(), y.copy()
    if change == "float32":
        xb = xb.astype(np.float32)
    elif change == "shape":
        xo = xo[:, :, :7]
    elif change == "empty":
        xb = xb[:0]
    elif change == "nan":
        xb[0, 0, 0] = np.nan
    elif change == "negative":
        xb[0, 0, 0] = -.1
    elif change == "excess_mass":
        xb[0, 0, 0] = .9
    elif change == "uncentered":
        yb[0, 0, 0] = .1
    else:
        yo[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        module.build_problem(xb, xo, yb, yo)


def test_owned_readonly_problem_and_invalid_heads():
    x = np.full((1, 1, 8), .125)
    y = np.zeros((1, 1, 4))
    problem = module.build_problem(x, x, y, y)
    x[:] = 0
    y[:] = 1
    assert problem.blind_states[0, 0, 0] == .125
    assert problem.blind_targets[0, 0, 0] == 0
    assert not problem.blind_states.flags.writeable
    for head in (np.zeros((4, 7)), np.full((4, 8), np.nan), np.zeros((4, 8), dtype=np.float32)):
        with pytest.raises(ValueError):
            module.certificate(problem, head)
    with pytest.raises(ValueError, match="already be feasible"):
        module.solve(problem, np.zeros((4, 8)))
