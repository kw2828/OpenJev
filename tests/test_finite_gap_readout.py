"""Fabricated fixed-step/certificate checks; no empirical inputs or models."""

import math
from dataclasses import replace

import numpy as np
import pytest

from openjev.research import finite_convex_readout as reference
from openjev.research import finite_gap_readout as candidate


def example(*, conditioned=False, boundary=False):
    x = np.zeros((8, 2, 8), dtype=np.float64)
    x[:, 0] = np.eye(8)
    if conditioned:
        x[:, 0] *= .05
        x[:, 0, 0] += .95
    p = np.full((4, 8), .1, dtype=np.float64)
    p[np.arange(8) % 4, np.arange(8)] = .7
    if boundary:
        p[:] = 0
        p[np.arange(8) % 4, np.arange(8)] = 1
    y = x @ (.25 - p).T
    return reference.build_problem(x, x, y, y), p


def scalar(problem, p):
    errors, products = [], [[[] for _ in range(8)] for _ in range(4)]
    for x, y in ((problem.blind_states, problem.blind_targets), (problem.observed_states, problem.observed_targets)):
        for i in range(problem.cases):
            for h in range(problem.horizon):
                for a in range(4):
                    prediction = math.fsum((.25 - float(p[a, s])) * float(x[i, h, s]) for s in range(8))
                    error = prediction - float(y[i, h, a])
                    errors.append(error * error)
                    for s in range(8):
                        products[a][s].append(-2 * error * float(x[i, h, s]))
    gradient = np.array([[math.fsum(values) / problem.denominator for values in row] for row in products])
    gap = (math.fsum(float(p[a, s]) * float(gradient[a, s]) for a in range(4) for s in range(8))
           - math.fsum(min(float(gradient[a, s]) for a in range(4)) for s in range(8)))
    return math.fsum(errors) / problem.denominator, gradient, gap


@pytest.mark.parametrize("conditioned,boundary", [(False, False), (False, True), (True, False), (True, True)])
def test_known_objective_and_direct_certificate(conditioned, boundary):
    problem, optimum = example(conditioned=conditioned, boundary=boundary)
    initial = np.full((4, 8), .25)
    original = initial.copy()
    result = candidate.solve(problem, initial)
    assert result["complete"] and result["status"] == "SOLVE_PASS"
    value, gradient, gap = scalar(problem, result["probabilities"])
    optimum_value, _, _ = scalar(problem, optimum)
    assert -1e-12 <= gap <= 1e-8
    assert value - optimum_value <= gap + 1e-12
    assert value <= result["objective_initial"] + 1e-12
    np.testing.assert_allclose(result["certificate"]["gradient"], gradient, atol=1e-15, rtol=1e-10)
    assert result["certificate"]["objective"] == pytest.approx(value, abs=1e-15)
    assert result["certificate"]["fw_gap"] == pytest.approx(gap, abs=1e-15)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(result["raw_probabilities"], result["probabilities"])
    assert result["projection"] == {"applied": False, "max_abs": 0., "l2": 0.}
    assert result["lipschitz"] == 2 * np.max(np.abs(problem.gram).sum(axis=1))
    assert result["lipschitz"] + 1e-15 >= 2 * np.linalg.eigvalsh(problem.gram).max()


def test_every_operation_and_certificate_schedule_is_disclosed():
    problem, _ = example(conditioned=True)
    calls = []
    result = candidate.solve(problem, np.full((4, 8), .25), check=lambda: calls.append(1))
    work = result["work"]
    assert result["complete"]
    assert work["check_calls"] == len(calls)
    assert work["projection_calls"] == work["gradient_calls"] == work["iterations"] + work["monotonicity_restarts"]
    assert work["quadratic_value_calls"] == 1 + work["gradient_calls"]
    assert work["check_calls"] == (2 + work["direct_objective_gradient_passes"]
                                   + work["quadratic_value_calls"] + 2 * work["gradient_calls"] + work["iterations"])
    assert work["accepted_iterations"] == work["iterations"]
    positions = [row["iteration"] for row in result["certificate_history"]]
    expected = list(range(0, work["accepted_iterations"] + 1, 10))
    if expected[-1] != work["accepted_iterations"]:
        expected.append(work["accepted_iterations"])
    assert positions == expected
    assert work["direct_objective_gradient_passes"] == len(positions)
    assert all(not row["qualifies"] for row in result["certificate_history"][:-1])
    assert result["certificate_history"][-1]["qualifies"]


def test_initial_certificate_has_no_gradient_or_projection_work():
    problem, optimum = example()
    result = candidate.solve(problem, optimum)
    assert result["complete"] and result["termination"] == "initial_certificate"
    assert result["work"] == {"gradient_calls": 0, "quadratic_value_calls": 0, "projection_calls": 0,
        "direct_objective_gradient_passes": 1, "check_calls": 3, "monotonicity_restarts": 0,
        "iterations": 0, "accepted_iterations": 0}


def test_zero_curvature_certifies_nonzero_constant_loss_without_updates():
    x = np.zeros((2, 2, 8), dtype=np.float64)
    y = np.tile(np.array([-.75, .25, .25, .25]), (2, 2, 1))
    problem = reference.build_problem(x, x, y, y)
    initial = np.tile(np.array([.5, .25, .125, .125])[:, None], (1, 8))
    result = candidate.solve(problem, initial)
    assert result["complete"] and result["lipschitz"] == 0
    assert result["certificate"]["objective"] == .375
    assert result["certificate"]["fw_gap"] == 0
    assert result["work"]["projection_calls"] == 0
    np.testing.assert_array_equal(result["probabilities"], initial)


def test_zero_curvature_does_not_override_failed_direct_certificate():
    problem, _ = example()
    inconsistent = replace(problem, gram=np.zeros((8, 8), dtype=np.float64))
    result = candidate.solve(inconsistent, np.full((4, 8), .25))
    assert not result["complete"]
    assert result["termination"] == "zero_curvature_uncertified"
    assert result["work"]["projection_calls"] == result["work"]["gradient_calls"] == 0


def test_zero_budget_preserves_uncertified_initial_point(monkeypatch):
    problem, _ = example()
    monkeypatch.setattr(candidate, "MAX_ITERATIONS", 0)
    initial = np.full((4, 8), .25)
    result = candidate.solve(problem, initial)
    assert not result["complete"] and result["termination"] == "iteration_budget"
    assert result["failure_reasons"] == ["iteration_budget", "final_numerical_certificate"]
    assert len(result["certificate_history"]) == 1
    assert result["work"]["projection_calls"] == 0
    np.testing.assert_array_equal(result["probabilities"], initial)


def test_final_off_schedule_certificate_can_pass_without_extra_projection(monkeypatch):
    problem, _ = example()
    monkeypatch.setattr(candidate, "MAX_ITERATIONS", 1)
    result = candidate.solve(problem, np.full((4, 8), .25))
    assert result["complete"] and result["termination"] == "final_direct_certificate"
    assert [row["iteration"] for row in result["certificate_history"]] == [0, 1]
    assert result["work"]["projection_calls"] == 1


def test_iteration_limit_is_not_a_small_update_success(monkeypatch):
    problem, _ = example(conditioned=True)
    monkeypatch.setattr(candidate, "MAX_ITERATIONS", 1)
    result = candidate.solve(problem, np.full((4, 8), .25))
    assert not result["complete"]
    assert result["termination"] == "iteration_budget"
    assert result["certificate"]["fw_gap"] > 1e-8
    assert result["work"]["accepted_iterations"] == 1


def test_restart_charges_discarded_candidate_and_plain_step(monkeypatch):
    problem, _ = example()
    real_projection = reference.project_simplex
    worst = np.zeros((4, 8), dtype=np.float64)
    worst[(np.arange(8) + 1) % 4, np.arange(8)] = 1
    calls = []

    def first_worse(value):
        calls.append(1)
        return worst.copy() if len(calls) == 1 else real_projection(value)

    monkeypatch.setattr(reference, "project_simplex", first_worse)
    result = candidate.solve(problem, np.full((4, 8), .25))
    assert result["complete"]
    assert result["work"]["monotonicity_restarts"] >= 1
    assert result["work"]["projection_calls"] == len(calls)
    assert result["work"]["gradient_calls"] == result["work"]["iterations"] + result["work"]["monotonicity_restarts"]


def test_rejected_plain_step_is_failure_without_backtracking_or_repair(monkeypatch):
    problem, _ = example()
    worst = np.zeros((4, 8), dtype=np.float64)
    worst[(np.arange(8) + 1) % 4, np.arange(8)] = 1
    projections = []

    def worse(value):
        projections.append(1)
        return worst.copy()

    monkeypatch.setattr(reference, "project_simplex", worse)
    initial = np.full((4, 8), .25)
    result = candidate.solve(problem, initial)
    assert not result["complete"] and result["termination"] == "plain_step_nonmonotone"
    assert result["failure_reasons"] == ["plain_step_nonmonotone"]
    assert result["work"]["iterations"] == 1
    assert result["work"]["accepted_iterations"] == 0
    assert result["work"]["projection_calls"] == len(projections) == 2
    assert result["work"]["direct_objective_gradient_passes"] == 1
    np.testing.assert_array_equal(result["probabilities"], initial)


def test_deadline_exception_propagates_and_no_hidden_retry():
    problem, _ = example()
    checks = []

    def check():
        checks.append(1)
        if len(checks) == 5:
            raise TimeoutError("fabricated fixed phase deadline")

    with pytest.raises(TimeoutError, match="fabricated fixed phase deadline"):
        candidate.solve(problem, np.full((4, 8), .25), check=check)
    assert len(checks) == 5


@pytest.mark.parametrize("invalid", [np.zeros((4, 8)), np.full((4, 8), np.nan),
                                     np.full((4, 8), .25, dtype=np.float32), np.full((8, 4), .25)])
def test_invalid_initial_heads_fail_before_updates(invalid):
    problem, _ = example()
    with pytest.raises(ValueError):
        candidate.solve(problem, invalid)


def test_direct_nonincrease_against_initial_is_required(monkeypatch):
    problem, _ = example()
    real = reference.certificate
    calls = []

    def misleading(p, probabilities):
        result = real(p, probabilities)
        calls.append(1)
        if len(calls) > 1:
            result = {**result, "objective": 1e3, "fw_gap": 0., "passed": True}
        return result

    monkeypatch.setattr(reference, "certificate", misleading)
    monkeypatch.setattr(candidate, "MAX_ITERATIONS", 1)
    result = candidate.solve(problem, np.full((4, 8), .25))
    assert not result["complete"]
    assert result["certificate_history"][-1]["certificate_passed"]
    assert not result["certificate_history"][-1]["objective_nonincreasing"]


def test_closed_simplex_and_global_rng_are_unchanged():
    problem, _ = example(boundary=True)
    before = np.random.get_state()
    result = candidate.solve(problem, np.full((4, 8), .25))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert np.all(result["probabilities"] >= 0)
    np.testing.assert_allclose(result["probabilities"].sum(axis=0), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["cost_matrix"].sum(axis=0), 0., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result["cost_matrix"] @ np.zeros(8), np.zeros(4))
