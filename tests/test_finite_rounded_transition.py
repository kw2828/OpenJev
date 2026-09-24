"""Independent fabricated arithmetic and derivative checks; qualification only."""

import json

import numpy as np
import pytest
import torch

from openjev.research.finite_rounded_transition import rounded_transition


def asymmetric_values():
    a, i, j = np.indices((4, 8, 8))
    return ((3 * a + 5 * i + 7 * j + 3 * i * j + a * j) % 19 - 9).astype(np.float64) / 4


def directions():
    a, i, j = np.indices((4, 8, 8))
    return (((a + 2) * (i + 1) * (j + 3)) % 23 - 11).astype(np.float64) / 11


def reference(values):
    """Independent NumPy four-sweep and row/column-deficit construction."""
    value = np.asarray(values, dtype=np.float64).copy()
    for _ in range(4):
        for axis in (2, 1):
            maximum = np.max(value, axis=axis, keepdims=True)
            value -= maximum + np.log(np.sum(np.exp(value - maximum), axis=axis, keepdims=True))
    before = np.exp(value)
    tau = 1 - 1e-8
    rows = before.sum(axis=2)
    after_rows = before * (tau / np.maximum(rows, tau))[:, :, None]
    columns = after_rows.sum(axis=1)
    contracted = after_rows * (tau / np.maximum(columns, tau))[:, None, :]
    r, c = 1 - contracted.sum(axis=2), 1 - contracted.sum(axis=1)
    correction = (r / r.sum(axis=1, keepdims=True))[:, :, None] * c[:, None, :]
    result = contracted + correction
    return {'transition': result, 'log_transition': np.log(result), 'before': before,
            'r': r, 'c': c, 'correction': correction,
            'row_branch': rows > tau, 'column_branch': columns > tau}


def expected_work():
    return {
        'rounded_transition_calls': 1, 'row_logsumexp_calls': 4, 'column_logsumexp_calls': 4,
        'normalization_vectors': 256, 'normalization_entries': 2048, 'completed_sweeps': 4,
        'exponential_calls': 1, 'exponential_entries': 256,
        'matrix_sum_calls': 7, 'vector_sum_calls': 2,
        'contraction_maximum_calls': 2, 'contraction_division_entries': 64,
        'contraction_product_entries': 512, 'deficit_subtraction_entries': 64,
        'deficit_normalization_entries': 32, 'rank_one_product_entries': 256,
        'rank_one_addition_entries': 256, 'correction_mass_sum_calls': 1,
        'logarithm_calls': 1, 'logarithm_entries': 256, 'check_calls': 11,
    }


def test_uniform_and_rank_one_inputs_have_analytic_uniform_output():
    row = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1) / 4
    column = torch.arange(8, dtype=torch.float64).reshape(1, 1, 8) / 7
    for logits in (torch.zeros((4, 8, 8), dtype=torch.float64),
                   (row + column).expand(4, 8, 8).clone()):
        result = rounded_transition(logits)
        torch.testing.assert_close(result['transition'], torch.full_like(logits, 1 / 8),
                                   atol=1e-12, rtol=0)
        torch.testing.assert_close(result['log_transition'], torch.full_like(logits, -np.log(8)),
                                   atol=1e-12, rtol=0)
        assert result['work'] == expected_work()
        assert result['diagnostics']['admitted'] is True
        json.dumps(result['diagnostics'], allow_nan=False)


def test_positive_doubly_stochastic_input_has_declared_smoothing_not_identity():
    x = np.stack([.75 * np.roll(np.eye(8), a + 1, axis=0) + 1 / 32 for a in range(4)])
    result = rounded_transition(torch.tensor(np.log(x), dtype=torch.float64))
    expected = (1 - 1e-8) * x + 1e-8 / 8
    np.testing.assert_allclose(result['transition'].numpy(), expected, atol=1e-12, rtol=0)
    # This fixture's analytical change is above numerical roundoff. The new
    # function intentionally changes an already balanced nonuniform matrix.
    assert result['diagnostics']['pre_to_final_maximum_change'] > 1e-9
    np.testing.assert_allclose(result['diagnostics']['correction_mass'], np.full(4, 8e-8),
                               atol=1e-12, rtol=0)


def test_independent_numpy_values_and_correction_diagnostics():
    values = asymmetric_values()
    result = rounded_transition(torch.tensor(values, dtype=torch.float64))
    oracle = reference(values)
    for key in ('transition', 'log_transition'):
        np.testing.assert_allclose(result[key].numpy(), oracle[key], atol=1e-12, rtol=1e-12)
    diagnostics = result['diagnostics']
    assert diagnostics['sweeps'] == 4 and diagnostics['slack'] == 1e-8
    assert diagnostics['tolerance'] == 1e-12
    expected = {
        'pre_round_row_residual_max': np.max(np.abs(oracle['before'].sum(2) - 1)),
        'pre_round_column_residual_max': np.max(np.abs(oracle['before'].sum(1) - 1)),
        'row_deficit_totals': oracle['r'].sum(1),
        'column_deficit_totals': oracle['c'].sum(1),
        'deficit_total_discrepancies': oracle['r'].sum(1) - oracle['c'].sum(1),
        'correction_mass': oracle['correction'].sum((1, 2)),
        'correction_maximum_entry': oracle['correction'].max(),
        'pre_to_final_maximum_change': np.max(np.abs(oracle['transition'] - oracle['before'])),
    }
    for key, value in expected.items():
        np.testing.assert_allclose(diagnostics[key], value, atol=1e-12, rtol=1e-12)
    assert diagnostics['pre_round_row_residual_max'] > 1e-8
    assert diagnostics['pre_to_final_maximum_change'] > 1e-8
    assert diagnostics['row_residual_max'] <= 1e-12
    assert diagnostics['column_residual_max'] <= 1e-12
    assert diagnostics['minimum_probability'] > 0 and diagnostics['maximum_probability'] < 1
    # The prior must see exactly the final probabilities, not pre-round logs.
    torch.testing.assert_close(result['log_transition'], result['transition'].log(), atol=0, rtol=0)


@pytest.mark.parametrize('log_output', [False, True])
def test_uniform_analytic_jacobian_is_slack_scaled_double_centering(log_output):
    logits = torch.zeros((4, 8, 8), dtype=torch.float64, requires_grad=True)
    weights = directions()
    result = rounded_transition(logits)
    output = result['log_transition' if log_output else 'transition']
    gradient, = torch.autograd.grad((output * torch.tensor(weights)).sum(), logits)
    double_center = weights - weights.mean(1, keepdims=True) - weights.mean(2, keepdims=True)
    double_center += weights.mean((1, 2), keepdims=True)
    expected = (1 - 1e-8) * double_center / (1 if log_output else 8)
    np.testing.assert_allclose(gradient.numpy(), expected, atol=1e-12, rtol=1e-12)


def test_actual_finite_algorithm_gradient_matches_independent_directional_difference():
    values, direction = asymmetric_values(), directions()
    weights = np.flip(direction, axis=2).copy()
    h = 1e-5
    base, plus, minus = reference(values), reference(values + h * direction), reference(values - h * direction)
    # The central difference claims a derivative only away from branch changes.
    for key in ('row_branch', 'column_branch'):
        np.testing.assert_array_equal(plus[key], base[key])
        np.testing.assert_array_equal(minus[key], base[key])
    logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    result = rounded_transition(logits)
    loss = ((result['transition'] + .03 * result['log_transition']) * torch.tensor(weights)).sum()
    gradient, = torch.autograd.grad(loss, logits)
    actual = float((gradient * torch.tensor(direction)).sum())
    high = np.sum((plus['transition'] + .03 * plus['log_transition']) * weights)
    low = np.sum((minus['transition'] + .03 * minus['log_transition']) * weights)
    expected = (high - low) / (2 * h)
    # Fixed before execution: central-difference O(h^2) truncation plus float64
    # cancellation in two scalar reductions, not fitted to observed residuals.
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
    assert abs(actual) > 1e-6


def test_action_row_column_permutations_are_equivariant():
    logits = torch.tensor(asymmetric_values(), dtype=torch.float64)
    actions, rows, columns = [2, 0, 3, 1], [3, 0, 7, 1, 5, 2, 6, 4], [5, 2, 0, 7, 3, 6, 1, 4]
    original = rounded_transition(logits)
    permuted = rounded_transition(logits[actions][:, rows][:, :, columns])
    for key in ('transition', 'log_transition'):
        torch.testing.assert_close(permuted[key], original[key][actions][:, rows][:, :, columns],
                                   atol=1e-12, rtol=1e-12)


def test_row_and_global_offsets_cancel_without_asserting_column_offset_invariance():
    logits = torch.tensor(asymmetric_values(), dtype=torch.float64)
    row = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1) * 8
    action = torch.tensor([-1024., 512., -256., 128.], dtype=torch.float64).reshape(4, 1, 1)
    shifted = logits + row + action
    actual, expected = rounded_transition(shifted), rounded_transition(logits)
    unit = np.finfo(np.float64).eps / 2
    gamma8 = 8 * unit / (1 - 8 * unit)
    bound = 16 * gamma8 * max(1., float(shifted.abs().max()))
    torch.testing.assert_close(actual['transition'], expected['transition'], atol=bound, rtol=0)


def test_selected_underflow_extreme_is_rounded_to_positive_permutation_mixture():
    matrix = np.full((4, 8, 8), -1000., dtype=np.float64)
    for a in range(4):
        for i in range(8):
            matrix[a, i, (i + a) % 8] = 1000.
    logits = torch.tensor(matrix, dtype=torch.float64, requires_grad=True)
    result = rounded_transition(logits)
    expected = np.full((4, 8, 8), 1e-8 / 8, dtype=np.float64)
    for a in range(4):
        for i in range(8):
            expected[a, i, (i + a) % 8] += 1 - 1e-8
    np.testing.assert_allclose(result['transition'].detach().numpy(), expected, atol=1e-12, rtol=1e-12)
    gradient, = torch.autograd.grad(result['log_transition'].sum(), logits)
    assert bool(torch.isfinite(gradient).all())
    assert result['work'] == expected_work()


def test_noncontiguous_input_output_ownership_and_no_mutation():
    logits = torch.tensor(asymmetric_values(), dtype=torch.float64).transpose(1, 2).requires_grad_(True)
    before = logits.detach().clone()
    assert not logits.is_contiguous()
    result = rounded_transition(logits)
    assert len({logits.data_ptr(), result['transition'].data_ptr(), result['log_transition'].data_ptr()}) == 3
    assert result['transition'].grad_fn is not None and result['log_transition'].grad_fn is not None
    ((result['transition'] + .03 * result['log_transition']) * torch.tensor(directions())).sum().backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    torch.testing.assert_close(logits.detach(), before, atol=0, rtol=0)
    saved_logs = result['log_transition'].detach().clone()
    with torch.no_grad():
        result['transition'].fill_(0)
    torch.testing.assert_close(result['log_transition'].detach(), saved_logs, atol=0, rtol=0)
    torch.testing.assert_close(logits.detach(), before, atol=0, rtol=0)


@pytest.mark.parametrize('bad', [None, np.zeros((4, 8, 8)), torch.zeros((4, 8, 8)),
                               torch.zeros((4, 8, 7), dtype=torch.float64),
                               torch.zeros((4, 8, 8), dtype=torch.int64),
                               torch.zeros((4, 8, 8), dtype=torch.complex128),
                               torch.empty((4, 8, 8), device='meta', dtype=torch.float64)])
def test_invalid_input_contract_rejected_before_normalization(bad):
    work = {}
    with pytest.raises(ValueError, match='CPU strided float64'):
        rounded_transition(bad, work=work)
    assert work['rounded_transition_calls'] == work['check_calls'] == 1
    assert work['row_logsumexp_calls'] == work['exponential_calls'] == 0


def test_sparse_layout_rejected():
    value = torch.zeros((4, 8, 8), dtype=torch.float64).to_sparse()
    with pytest.raises(ValueError, match='CPU strided float64'):
        rounded_transition(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_inputs_are_not_sanitized(value):
    logits = torch.zeros((4, 8, 8), dtype=torch.float64)
    logits[0, 0, 0] = value
    work = {}
    with pytest.raises(ValueError, match='finite values'):
        rounded_transition(logits, work=work)
    assert work['row_logsumexp_calls'] == 0


def test_finite_input_overflow_rejects_without_extra_sweep_or_fallback():
    logits = torch.full((4, 8, 8), -1e308, dtype=torch.float64)
    logits[:, :, 0] = 1e308
    before, work = logits.clone(), {}
    with pytest.raises(ValueError, match='row-normalized logs: finite'):
        rounded_transition(logits, work=work)
    assert work['row_logsumexp_calls'] == 1 and work['normalization_entries'] == 256
    assert work['column_logsumexp_calls'] == work['completed_sweeps'] == 0
    assert work['exponential_calls'] == work['logarithm_calls'] == 0
    torch.testing.assert_close(logits, before, atol=0, rtol=0)


@pytest.mark.parametrize('work', [[], {'x': True}, {'x': -1}, {'x': 1.0}])
def test_invalid_caller_work_rejected(work):
    with pytest.raises(ValueError, match='nonnegative Python integer'):
        rounded_transition(torch.zeros((4, 8, 8), dtype=torch.float64), work=work)


def test_accumulated_work_preserves_other_counters_and_callback_identity():
    work, calls = {'caller_prior_work': 7}, []
    check = lambda: calls.append(len(calls))
    logits = torch.zeros((4, 8, 8), dtype=torch.float64)
    first = rounded_transition(logits, check=check, work=work)
    assert first['work'] is work and len(calls) == 11
    rounded_transition(logits, check=check, work=work)
    assert work == {'caller_prior_work': 7, **{k: 2 * v for k, v in expected_work().items()}}
    assert len(calls) == 22


@pytest.mark.parametrize('stop', [1, 3, 6, 7, 8, 9, 10, 11])
def test_external_stop_propagates_and_retains_partial_work(stop):
    error, work, calls = TimeoutError('fabricated deadline'), {}, []

    def check():
        calls.append(None)
        if len(calls) == stop:
            raise error

    with pytest.raises(TimeoutError) as caught:
        rounded_transition(torch.zeros((4, 8, 8), dtype=torch.float64), check=check, work=work)
    assert caught.value is error and work['check_calls'] == stop
    assert work['completed_sweeps'] == min(max(stop - 1, 0), 4)
    assert work['exponential_calls'] == int(stop >= 6)
    assert work['contraction_product_entries'] == 256 * (int(stop >= 7) + int(stop >= 8))
    assert work['rank_one_product_entries'] == 256 * int(stop >= 10)
    assert work['logarithm_calls'] == int(stop >= 11)


def test_fixed_function_rejects_tuning_kwargs_and_noncallable_check():
    logits = torch.zeros((4, 8, 8), dtype=torch.float64)
    for kwargs in ({'sweeps': 5}, {'slack': 1e-7}, {'tolerance': 1e-10}):
        with pytest.raises(TypeError):
            rounded_transition(logits, **kwargs)
    work = {}
    with pytest.raises(ValueError, match='check must be callable'):
        rounded_transition(logits, check=None, work=work)
    assert work['rounded_transition_calls'] == 1 and work['check_calls'] == 0
