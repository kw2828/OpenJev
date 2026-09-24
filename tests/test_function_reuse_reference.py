"""Fabricated independent algebra witnesses, with no study generation."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from openjev.research.function_reuse_reference import (
    fewshot_only,
    fit_blocks,
    predict,
    storage_metadata,
)


def fixture():
    # Axis-aligned rows give an independent scalar solution for every column.
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    matrices = np.array([[[1., 2.], [3., 4.]], [[-2., 1.], [1., -3.]], [[2., 0.], [0., 2.]]])
    bx = np.repeat(x[None], 3, axis=0)
    by = np.array([[[1., 2.], [3., 4.], [-1., -2.], [-3., -4.]],
                   [[-2., 1.], [1., -3.], [2., -1.], [-1., 3.]],
                   [[2., 0.], [0., 2.], [-2., 0.], [0., -2.]]])
    return bx, by, matrices


def test_exact_rational_fit_orientation_rank_and_singular_diagnostics():
    x, y, matrices = fixture()
    fitted = fit_blocks(x, y)
    np.testing.assert_allclose(fitted["maps"], matrices, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(fitted["ranks"], [2, 2, 2])
    np.testing.assert_array_equal(fitted["full_rank"], [True, True, True])
    np.testing.assert_allclose(fitted["singular_values"], np.full((3, 2), np.sqrt(2)), atol=1e-15)


def test_residual_routing_and_prediction_have_independent_rational_witness():
    x, y, _ = fixture()
    result = predict(fit_blocks(x, y), np.array([[.5, 1.]]), np.array([[0., -2.5]]), np.array([2., -.5]))
    # Candidate few-shot values are (3.5,5), (0,-2.5), and (1,2).
    np.testing.assert_allclose(result["block_residuals"], [34.25, 0., 10.625], atol=1e-13)
    assert result["selected_block"] == 1
    np.testing.assert_array_equal(result["tie_indices"], [1])
    np.testing.assert_allclose(result["prediction"], [-4.5, 3.5], atol=1e-14)


def test_exact_ties_choose_lowest_index_and_report_all_ties():
    x = np.repeat(np.eye(2)[None], 3, axis=0)
    y = x.copy()
    result = predict(fit_blocks(x, y), np.eye(2), np.eye(2), np.array([3., 7.]))
    assert result["selected_block"] == 0
    np.testing.assert_array_equal(result["tie_indices"], [0, 1, 2])
    np.testing.assert_array_equal(result["block_residuals"], [0., 0., 0.])
    np.testing.assert_array_equal(result["prediction"], [3., 7.])


def test_non_tied_block_permutation_preserves_prediction_and_permutes_residuals():
    x, y, _ = fixture()
    few_x, few_y, query = np.array([[.5, 1.]]), np.array([[0., -2.5]]), np.array([2., -.5])
    original = predict(fit_blocks(x, y), few_x, few_y, query)
    permutation = [2, 0, 1]
    permuted = predict(fit_blocks(x[permutation], y[permutation]), few_x, few_y, query)
    assert permuted["selected_block"] == 2
    np.testing.assert_array_equal(permuted["prediction"], original["prediction"])
    np.testing.assert_array_equal(permuted["block_residuals"], original["block_residuals"][permutation])


def test_rank_deficient_minimum_norm_solution_is_accepted_and_flagged():
    x = np.array([[[1., 1.], [2., 2.], [-1., -1.]]])
    y = np.array([[[2., 4.], [4., 8.], [-2., -4.]]])
    fitted = fit_blocks(x, y)
    np.testing.assert_allclose(fitted["maps"][0], [[1., 2.], [1., 2.]], atol=1e-14)
    np.testing.assert_array_equal(fitted["ranks"], [1])
    np.testing.assert_array_equal(fitted["full_rank"], [False])
    query = predict(fitted, x[0], y[0], np.array([1., -1.]))
    np.testing.assert_allclose(query["prediction"], [0., 0.], atol=1e-14)


def test_zero_design_minimum_norm_and_nonzero_residual_are_not_fabricated_recovery():
    x = np.zeros((1, 2, 2), np.float64)
    y = np.array([[[1., 2.], [1., 2.]]])
    fitted = fit_blocks(x, y)
    np.testing.assert_array_equal(fitted["maps"], np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(fitted["ranks"], [0])
    result = predict(fitted, x[0], y[0], np.array([3., 4.]))
    np.testing.assert_array_equal(result["block_residuals"], [2.5])


def test_inconsistent_public_rows_return_scalar_least_squares_average():
    x = np.array([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]])
    y = np.array([[[1., 2.], [3., 6.], [2., 0.], [4., 4.]]])
    np.testing.assert_allclose(fit_blocks(x, y)["maps"][0], [[2., 4.], [3., 2.]], atol=1e-14)


def test_fewshot_only_uses_minimum_norm_and_cannot_recover_unseen_direction():
    result = fewshot_only(np.array([[1., 0.]]), np.array([[2., 3.]]), np.array([0., 1.]))
    np.testing.assert_array_equal(result["prediction"], [0., 0.])
    np.testing.assert_array_equal(result["fit"]["maps"][0], [[2., 3.], [0., 0.]])
    np.testing.assert_array_equal(result["fit"]["ranks"], [1])
    assert not result["fit"]["full_rank"][0]
    assert type(result["rank"]) is int and result["rank"] == 1
    np.testing.assert_array_equal(result["singular_values"], [1.])
    assert result["singular_values"].flags.owndata
    assert not np.shares_memory(result["singular_values"], result["fit"]["singular_values"])


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_noiseless_duplicate_rows_do_not_change_minimum_norm_answer(rank_deficient):
    x, y, _ = fixture()
    if rank_deficient:
        x, y = x[:, ::2], y[:, ::2]
    original = fit_blocks(x, y)
    repeated = fit_blocks(np.repeat(x, 3, axis=1), np.repeat(y, 3, axis=1))
    np.testing.assert_allclose(original["maps"], repeated["maps"], atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(original["ranks"], repeated["ranks"])
    np.testing.assert_allclose(repeated["singular_values"], np.sqrt(3) * original["singular_values"], atol=1e-13)


def test_disclosed_dimension_and_storage_account_for_maps_and_diagnostics_separately():
    x = np.repeat(np.tile(np.eye(8), (2, 1))[None], 3, axis=0)
    fitted = fit_blocks(x, x.copy())
    metadata = fitted["metadata"]
    assert metadata == storage_metadata(3, 16, 8)
    assert metadata["learned_parameter_count"] == 0
    assert metadata["fitted_coefficient_count"] == 192
    assert metadata["fitted_coefficient_bytes"] == 1536
    assert metadata["diagnostic_bytes"] == 192 + 24 + 3
    assert metadata["public_demonstration_bytes"] == 6144
    assert metadata["retained_numeric_state_bytes"] == sum(
        fitted[name].nbytes for name in ("maps", "singular_values", "ranks", "full_rank"))
    assert not metadata["raw_demonstrations_retained"] and not metadata["workspace_bytes_measured"]


def test_inputs_and_global_rng_unchanged_owned_results_support_noncontiguous_input():
    x, y, _ = fixture()
    x, y = x[:, ::-1], y[:, ::-1]
    old_x, old_y = x.copy(), y.copy()
    rng_before = np.random.get_state()
    fitted = fit_blocks(x, y)
    result = predict(fitted, x[0], y[0], np.array([1., 2.]))
    fewshot_only(x[0], y[0], np.array([1., 2.]))
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    np.testing.assert_array_equal(x, old_x)
    np.testing.assert_array_equal(y, old_y)
    for item in (fitted["maps"], fitted["singular_values"], fitted["ranks"],
                 fitted["full_rank"], result["prediction"], result["block_residuals"], result["tie_indices"]):
        assert item.flags.owndata and item.flags.c_contiguous and not item.flags.writeable
        assert not np.shares_memory(item, x) and not np.shares_memory(item, y)
    before = fitted["maps"].copy()
    x.fill(17.); y.fill(-3.)
    np.testing.assert_array_equal(fitted["maps"], before)


def test_no_future_targets_or_true_basis_api_and_query_does_not_affect_routing():
    assert tuple(inspect.signature(predict).parameters) == ("fitted", "fewshot_x", "fewshot_y", "query_x")
    x, y, _ = fixture()
    bank = fit_blocks(x, y)
    args = (bank, np.array([[.5, 1.]]), np.array([[0., -2.5]]), np.array([2., -.5]))
    with pytest.raises(TypeError, match="query_target"):
        predict(*args, query_target=np.zeros(2))
    with pytest.raises(TypeError, match="true_index"):
        predict(*args, true_index=0)
    altered = predict(*args[:-1], np.array([100., 200.]))
    assert altered["selected_block"] == predict(*args)["selected_block"] == 1
    np.testing.assert_array_equal(altered["block_residuals"], predict(*args)["block_residuals"])


@pytest.mark.parametrize("bad", [np.zeros((0, 2, 2)), np.zeros((1, 0, 2)), np.zeros((1, 2, 0)),
                                np.zeros((2, 2)), np.zeros((1, 2, 2), np.float32),
                                np.full((1, 2, 2), np.nan), np.full((1, 2, 2), np.inf), [[[1.]]]])
def test_rejects_invalid_block_inputs(bad):
    with pytest.raises(ValueError):
        fit_blocks(bad, bad)


def test_rejects_mismatched_shapes_and_invalid_fewshot_query_inputs():
    x, y, _ = fixture()
    with pytest.raises(ValueError, match="identical"):
        fit_blocks(x, y[:, :2])
    bank = fit_blocks(x, y)
    for fx, fy, query in [(np.empty((0, 2)), np.empty((0, 2)), np.ones(2)),
                          (np.ones((1, 2)), np.ones((2, 2)), np.ones(2)),
                          (np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2))),
                          (np.ones((1, 2)), np.ones((1, 2)), np.ones(3)),
                          (np.ones((1, 2)), np.full((1, 2), np.nan), np.ones(2)),
                          (np.ones((1, 2)), np.ones((1, 2)), np.full(2, np.inf))]:
        with pytest.raises(ValueError):
            predict(bank, fx, fy, query)
        with pytest.raises(ValueError):
            fewshot_only(fx, fy, query)


@pytest.mark.parametrize("field,bad", [("version", "wrong"), ("maps", np.ones((3, 2, 3))),
                                     ("singular_values", np.full((3, 2), -1.)),
                                     ("ranks", np.full(3, 3, np.int64)),
                                     ("full_rank", np.zeros(3, bool)), ("metadata", {})])
def test_rejects_malformed_fitted_diagnostics(field, bad):
    x, y, _ = fixture()
    bank = fit_blocks(x, y)
    bank[field] = bad
    with pytest.raises(ValueError):
        predict(bank, x[0], y[0], np.ones(2))


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, None])
def test_storage_dimensions_are_strict_positive_integers(bad):
    with pytest.raises(ValueError):
        storage_metadata(bad, 16, 8)


def test_solver_nonconvergence_and_nonfinite_results_fail_without_fallback(monkeypatch):
    x, y, _ = fixture()
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("fabricated failure")
    monkeypatch.setattr(np.linalg, "lstsq", fail)
    with pytest.raises(ValueError, match="did not converge"):
        fit_blocks(x, y)
    monkeypatch.setattr(np.linalg, "lstsq", lambda *args, **kwargs: (
        np.full((2, 2), np.nan), np.array([]), 2, np.ones(2)))
    with pytest.raises(ValueError, match="nonfinite"):
        fit_blocks(x, y)
