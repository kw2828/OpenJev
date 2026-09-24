"""Fabricated Sinkhorn and derivative oracles, executed only by qualification."""

from fractions import Fraction

import numpy as np
import pytest
import torch

from openjev.research.finite_balanced_transition import balanced_transition


def asymmetric_logits():
    action, row, column = np.indices((4, 8, 8))
    values = ((3 * action + 5 * row + 7 * column + 3 * row * column + action * column) % 17 - 8) / 16
    return torch.tensor(values, dtype=torch.float64)


def weight_array():
    action, row, column = np.indices((4, 8, 8))
    return (((action + 2) * (row + 1) * (column + 3)) % 19 - 9).astype(np.float64) / 9


def finite_sweep_reference(logits, sweeps):
    """Independent NumPy value oracle for the declared finite sequence."""
    value = np.asarray(logits, dtype=np.float64).copy()
    for _ in range(sweeps):
        for axis in (2, 1):
            maximum = value.max(axis=axis, keepdims=True)
            normalizer = maximum + np.log(np.exp(value - maximum).sum(axis=axis, keepdims=True))
            value = value - normalizer
    return {"transition": np.exp(value), "log_transition": value}


def expected_work(sweeps):
    return {
        "balanced_transition_calls": 1,
        "row_logsumexp_calls": sweeps,
        "column_logsumexp_calls": sweeps,
        "normalization_vectors": 64 * sweeps,
        "normalization_entries": 512 * sweeps,
        "completed_sweeps": sweeps,
        "exponential_calls": 1,
        "exponential_entries": 256,
        "residual_sum_calls": 2,
        "check_calls": sweeps + 2,
    }


@pytest.mark.parametrize("sweeps", [1, 64])
def test_uniform_and_rank_one_inputs_have_uniform_closed_form(sweeps):
    row = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1) / 4
    column = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8) / 7
    for logits in (torch.zeros((4, 8, 8), dtype=torch.float64),
                   (row + column).expand(4, 8, 8).clone()):
        result = balanced_transition(logits, sweeps=sweeps)
        torch.testing.assert_close(result["transition"], torch.full_like(logits, 1 / 8),
                                   atol=1e-12, rtol=0)
        torch.testing.assert_close(result["log_transition"], torch.full_like(logits, -np.log(8)),
                                   atol=1e-12, rtol=0)
        assert result["work"] == expected_work(sweeps)
        assert result["diagnostics"]["converged"] is True


def test_positive_permutation_plus_uniform_is_a_balancing_fixed_point():
    matrices = []
    for action in range(4):
        permutation = np.roll(np.eye(8), action + 1, axis=0)
        matrices.append(3 / 4 * permutation + 1 / 32)
    expected = torch.tensor(np.asarray(matrices), dtype=torch.float64)
    result = balanced_transition(expected.log())
    torch.testing.assert_close(result["transition"], expected, atol=1e-12, rtol=1e-12)
    assert torch.all(result["transition"] > 0)
    assert torch.all(result["transition"] < 1)


def test_asymmetric_values_match_finite_sequence_and_report_actual_residuals():
    logits = asymmetric_logits()
    result = balanced_transition(logits)
    reference = finite_sweep_reference(logits.numpy(), 64)
    for key in ("transition", "log_transition"):
        np.testing.assert_allclose(result[key].numpy(), reference[key], atol=1e-12, rtol=1e-12)
    matrix = result["transition"]
    row_error = float((matrix.sum(2) - 1).abs().max())
    column_error = float((matrix.sum(1) - 1).abs().max())
    assert result["diagnostics"] == {
        "sweeps": 64,
        "tolerance": 1e-12,
        "row_residual_max": row_error,
        "column_residual_max": column_error,
        "minimum_probability": float(matrix.min()),
        "maximum_probability": float(matrix.max()),
        "converged": True,
    }
    assert row_error <= 1e-12 and column_error <= 1e-12


def test_scalar_row_and_column_logit_offsets_preserve_balanced_transport():
    logits = asymmetric_logits()
    action = torch.tensor([-1024., 512., -256., 128.], dtype=torch.float64).reshape(4, 1, 1)
    row = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1) * 8
    column = -torch.arange(8, dtype=torch.float64).reshape(1, 1, 8) * 6
    shifted = logits + action + row + column
    first = balanced_transition(logits)["transition"]
    second = balanced_transition(shifted)["transition"]
    # Large offsets lose a few low bits before normalization. The guard scales
    # with the input magnitude and float64 reduction length, not observed error.
    unit = np.finfo(np.float64).eps / 2
    gamma8 = 8 * unit / (1 - 8 * unit)
    tolerance = 16 * gamma8 * max(1., float(shifted.abs().max()))
    torch.testing.assert_close(second, first, atol=tolerance, rtol=0)


def test_independent_row_column_and_action_permutations_are_equivariant():
    logits = asymmetric_logits()
    actions = [2, 0, 3, 1]
    rows = [3, 0, 7, 1, 5, 2, 6, 4]
    columns = [5, 2, 0, 7, 3, 6, 1, 4]
    transformed = logits[actions][:, rows][:, :, columns]
    expected = balanced_transition(logits)
    actual = balanced_transition(transformed)
    for key in ("transition", "log_transition"):
        torch.testing.assert_close(actual[key], expected[key][actions][:, rows][:, :, columns],
                                   atol=1e-12, rtol=1e-12)


def test_noncontiguous_input_keeps_owned_storage_and_autograd_graph():
    logits = asymmetric_logits().transpose(1, 2).detach().requires_grad_(True)
    before = logits.detach().clone()
    assert not logits.is_contiguous()
    result = balanced_transition(logits)
    pointers = [logits.data_ptr(), result["transition"].data_ptr(), result["log_transition"].data_ptr()]
    assert len(set(pointers)) == 3
    assert result["transition"].grad_fn is not None and result["log_transition"].grad_fn is not None
    weight = torch.tensor(weight_array(), dtype=torch.float64)
    ((result["transition"] + .03 * result["log_transition"]) * weight).sum().backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().max()) > 1e-4
    torch.testing.assert_close(logits.detach(), before, atol=0, rtol=0)
    with torch.no_grad():
        result["transition"].fill_(0)
        result["log_transition"].fill_(1)
    torch.testing.assert_close(logits.detach(), before, atol=0, rtol=0)


@pytest.mark.parametrize("field,scale", [("transition", 1 / 8), ("log_transition", 1.)])
def test_uniform_jacobian_is_analytically_doubly_centered(field, scale):
    logits = torch.zeros((4, 8, 8), dtype=torch.float64, requires_grad=True)
    weights = weight_array()
    weight = torch.tensor(weights, dtype=torch.float64)
    result = balanced_transition(logits)
    gradient, = torch.autograd.grad((result[field] * weight).sum(), logits)
    # At uniform transport, one complete sweep removes row and column means.
    # Further sweeps leave this tangent projection unchanged.
    centered = weights - weights.mean(1, keepdims=True) - weights.mean(2, keepdims=True)
    centered += weights.mean((1, 2), keepdims=True)
    np.testing.assert_allclose(gradient.numpy(), scale * centered, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(gradient.numpy().sum(1), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient.numpy().sum(2), 0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("field", ["transition", "log_transition"])
def test_autograd_matches_finite_difference_of_executed_64_sweeps(field):
    logits = asymmetric_logits().requires_grad_(True)
    weights = weight_array()
    result = balanced_transition(logits)
    gradient, = torch.autograd.grad((result[field] * torch.tensor(weights)).sum(), logits)
    assert float(gradient.abs().max()) > 1e-4
    source = logits.detach().numpy().copy()
    directions = [weights, np.roll(weights, 1, axis=1), np.roll(weights, 2, axis=2)]
    step = 1e-5
    for direction in directions:
        plus = finite_sweep_reference(source + step * direction, 64)[field]
        minus = finite_sweep_reference(source - step * direction, 64)[field]
        difference = float(((plus - minus) * weights).sum() / (2 * step))
        actual = float((gradient.numpy() * direction).sum())
        # Central differences incur O(step^2) truncation plus accumulated
        # float64 rounding across 128 logsumexp reductions; no fitted cutoff.
        assert actual == pytest.approx(difference, rel=2e-6, abs=2e-8)


def test_one_sweep_failure_has_a_rational_nonconvergence_witness_and_no_retry():
    matrix = torch.ones((4, 8, 8), dtype=torch.float64)
    matrix[:, 0, 0] = 16
    matrix[:, 0, 1] = 8
    logits = matrix.log()
    before = logits.clone()
    # After row-then-column normalization, the first row has this exact sum.
    first_row = Fraction(64, 169) + Fraction(32, 137) + 6 * Fraction(4, 109)
    assert abs(float(first_row) - 1) > .1
    work = {}
    with pytest.raises(ValueError, match="fixed-sweep residual"):
        balanced_transition(logits, sweeps=1, work=work)
    assert work == expected_work(1)
    torch.testing.assert_close(logits, before, atol=0, rtol=0)


def test_default_fixed_work_has_no_early_exit_and_preserves_caller_counters():
    work = {"external_counter": 13, "row_logsumexp_calls": 7}
    callbacks = []
    rng = torch.random.get_rng_state().clone()
    result = balanced_transition(torch.zeros((4, 8, 8), dtype=torch.float64),
                                 work=work, check=lambda: callbacks.append(None))
    expected = expected_work(64)
    expected["row_logsumexp_calls"] += 7
    expected["external_counter"] = 13
    assert result["work"] is work and work == expected
    assert len(callbacks) == 66
    torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize("stop", [1, 3, 66])
def test_check_exceptions_propagate_with_performed_work_and_unchanged_input(stop):
    source = asymmetric_logits()
    original = source.clone()
    error = RuntimeError("fabricated checkpoint stop")
    seen = []
    work = {}

    def check():
        seen.append(None)
        if len(seen) == stop:
            raise error

    with pytest.raises(RuntimeError) as caught:
        balanced_transition(source, check=check, work=work)
    assert caught.value is error
    assert work["check_calls"] == stop
    assert work["completed_sweeps"] == min(stop - 1, 64)
    assert work["row_logsumexp_calls"] == work["completed_sweeps"]
    assert work["column_logsumexp_calls"] == work["completed_sweeps"]
    assert work["exponential_calls"] == int(stop == 66)
    assert work["residual_sum_calls"] == 2 * int(stop == 66)
    torch.testing.assert_close(source, original, atol=0, rtol=0)


@pytest.mark.parametrize("change", ["float32", "integer", "shape", "not_tensor", "nan", "inf", "negative_inf", "meta", "sparse"])
def test_invalid_tensor_domain_is_rejected_before_normalization(change):
    source = asymmetric_logits()
    if change == "float32":
        source = source.float()
    elif change == "integer":
        source = source.to(torch.int64)
    elif change == "shape":
        source = source[:3]
    elif change == "not_tensor":
        source = source.tolist()
    elif change == "meta":
        source = torch.empty((4, 8, 8), dtype=torch.float64, device="meta")
    elif change == "sparse":
        source = source.to_sparse()
    else:
        source[0, 0, 0] = {"nan": float("nan"), "inf": float("inf"), "negative_inf": -float("inf")}[change]
    work = {}
    with pytest.raises(ValueError):
        balanced_transition(source, work=work)
    assert work["row_logsumexp_calls"] == work["column_logsumexp_calls"] == 0
    assert work["completed_sweeps"] == work["exponential_calls"] == 0


def test_probability_underflow_or_saturation_fails_without_clipping_or_repair():
    logits = torch.full((4, 8, 8), -1000., dtype=torch.float64)
    for state in range(8):
        logits[:, state, state] = 1000
    original = logits.clone()
    work = {}
    with pytest.raises(ValueError, match="strict transition probabilities"):
        balanced_transition(logits, work=work)
    assert work["completed_sweeps"] == 64 and work["exponential_calls"] == 1
    assert work["residual_sum_calls"] == 0 and work["check_calls"] == 65
    torch.testing.assert_close(logits, original, atol=0, rtol=0)


@pytest.mark.parametrize("kwargs", [
    {"sweeps": 0}, {"sweeps": 65}, {"sweeps": True}, {"sweeps": 1.},
    {"tolerance": 0.}, {"tolerance": -1e-12}, {"tolerance": 1e-11},
    {"tolerance": float("nan")}, {"tolerance": float("inf")}, {"tolerance": True},
    {"tolerance": 0}, {"check": None},
])
def test_invalid_configuration_fails_before_callbacks(kwargs):
    calls = []
    options = {"check": lambda: calls.append(None), **kwargs}
    with pytest.raises(ValueError):
        balanced_transition(asymmetric_logits(), **options)
    assert not calls


@pytest.mark.parametrize("work", [[], {"other": -1}, {"other": True}, {"completed_sweeps": 1.5}])
def test_work_requires_nonnegative_python_integer_counts(work):
    with pytest.raises(ValueError, match="nonnegative Python integer"):
        balanced_transition(asymmetric_logits(), work=work)
