"""Independent fabricated linear algebra; no empirical inputs or model calls."""

import copy

import numpy as np
import pytest

from openjev.research import otto_direct_readout as M


def reference_basis():
    return np.array([[.5, .5, .5], [.5, -.5, -.5], [-.5, .5, -.5], [-.5, -.5, .5]])


def reference_loss(z, error, legal, weights, B):
    """Direct legal-subset centering, independent of the design construction."""
    total = 0.
    increment = reference_basis() @ B
    for feature, target, mask, weight in zip(z, error, legal, weights, strict=True):
        if weight == 0:
            continue
        difference = (increment @ feature - target)[mask]
        centered = difference - sum(float(x) for x in difference) / len(difference)
        total += float(weight) * sum(float(x) ** 2 for x in centered)
    return total


def fabricated_data():
    z = np.array([[1., 0., 1.], [0., 1., 1.], [2., -1., 1.], [-1., 2., 1.]])
    error = np.array([[1., 2., -3., 4.], [-2., 0., 3., 1.], [0., 5., 2., -1.], [1., 1., 1., 1.]])
    legal = np.array([[True, True, False, True], [True, False, True, False],
                      [True, True, True, True], [False, False, False, False]])
    # Already includes episode/support-count/action-count divisions, once only.
    weights = np.array([1 / (2 * 2 * 3), 1 / (2 * 2 * 2), 1 / (2 * 1 * 4), 0.])
    return z, error, legal, weights


def test_fixed_basis_has_exact_gauge_removal_and_owned_output():
    U = M.contrast_basis()
    assert U.dtype == np.float64 and U.shape == (4, 3)
    np.testing.assert_array_equal(U, reference_basis())
    np.testing.assert_array_equal(U.T @ U, np.eye(3))
    np.testing.assert_array_equal(U @ U.T, np.eye(4) - np.ones((4, 4)) / 4)
    np.testing.assert_array_equal(U.sum(axis=0), np.zeros(3))
    U[:] = 9
    np.testing.assert_array_equal(M.contrast_basis(), reference_basis())


def test_build_design_matches_explicit_nested_coefficients_and_row_major_order():
    z, error, legal, weights = fabricated_data()
    A, b = M.build_design(z, error, legal, weights)
    expected = np.zeros((16, 9))
    expected_target = np.zeros(16)
    U = reference_basis()
    for n in range(4):
        support = [j for j in range(4) if legal[n, j]]
        if not support or weights[n] == 0:
            continue
        for a in range(4):
            if a not in support:
                continue
            for k in range(3):
                centered_u = U[a, k] - sum(U[j, k] for j in support) / len(support)
                for feature in range(3):
                    expected[4 * n + a, 3 * k + feature] = np.sqrt(weights[n]) * centered_u * z[n, feature]
            expected_target[4 * n + a] = np.sqrt(weights[n]) * (
                error[n, a] - sum(error[n, j] for j in support) / len(support))
    np.testing.assert_allclose(A, expected, rtol=0, atol=2e-16)
    np.testing.assert_allclose(b, expected_target, rtol=0, atol=5e-16)
    assert A.dtype == b.dtype == np.float64


@pytest.mark.parametrize("B", (np.zeros((3, 3)), np.arange(9).reshape(3, 3) / 7))
def test_weighted_design_loss_equals_episode_balanced_projected_objective(B):
    z, error, legal, weights = fabricated_data()
    A, b = M.build_design(z, error, legal, weights)
    residual = A @ B.reshape(-1) - b
    np.testing.assert_allclose(residual @ residual, reference_loss(z, error, legal, weights, B), rtol=1e-14, atol=1e-14)
    # A second episode with no supported terms remains in the caller's factor 1/2.
    single_episode_A, single_episode_b = M.build_design(z, error, legal, 2 * weights)
    single_residual = single_episode_A @ B.reshape(-1) - single_episode_b
    np.testing.assert_allclose(single_residual @ single_residual, 2 * (residual @ residual), rtol=1e-14)


def test_ols_identifies_known_full_rank_optimum_for_general_feature_dimension():
    z = np.array([[1., 0., 1.], [0., 1., 1.], [-1., -1., 1.], [2., 1., 1.]])
    B = np.array([[.2, -.1, .3], [2., 1., -.5], [-3., .5, 2.]])
    target = z @ (reference_basis() @ B).T + np.arange(4)[:, None]
    A, b = M.build_design(z, target, np.ones((4, 4), bool), np.ones(4))
    solved = M.solve_design(A, b, mode="ols")
    np.testing.assert_allclose(solved["B"], B, rtol=0, atol=5e-14)
    np.testing.assert_allclose(solved["increment"], reference_basis() @ B, rtol=0, atol=5e-14)
    np.testing.assert_array_equal(solved["coefficients"], solved["B"].reshape(-1))
    d = solved["diagnostics"]
    assert d["rank"] == d["coordinates"] == 9 and d["full_column_rank"]
    assert d["feature_dimension"] == 3 and d["spectrum_scope"] == "data"
    assert d["ridge"] == d["regularization"] == 0
    assert d["data_loss"] < 1e-24 and d["normal_residual"] < 1e-11
    assert d["total_objective"] == d["data_loss"]
    np.testing.assert_allclose(d["objective_before"], b @ b)
    diagnostic = M.design_diagnostics(A)
    assert diagnostic["rank"] == d["rank"]
    np.testing.assert_allclose(diagnostic["singular_values"], solved["singular_values"], rtol=1e-13)


def test_rank_deficiency_uses_minimum_norm_and_reports_absent_full_condition():
    z = np.array([[1., 2.], [2., 4.], [3., 6.]])
    effective = np.array([5., 11., 17.])
    target = np.outer(z[:, 0], reference_basis() @ effective)
    A, b = M.build_design(z, target, np.ones((3, 4), bool), np.ones(3))
    solved = M.solve_design(A, b, mode="ols")
    expected = np.outer(effective, [1., 2.]) / 5
    np.testing.assert_allclose(solved["B"], expected, rtol=1e-13, atol=1e-13)
    assert solved["diagnostics"]["rank"] == 3
    assert solved["diagnostics"]["condition_number"] is None
    assert solved["diagnostics"]["retained_condition_number"] is not None
    assert not solved["diagnostics"]["full_column_rank"]
    null_direction = np.array([2., -1.])
    # Allow scale-aware float64 SVD roundoff in the null-space projection.
    tolerance = 64 * np.finfo(np.float64).eps * np.linalg.norm(solved["B"]) * np.linalg.norm(null_direction)
    np.testing.assert_allclose(solved["B"] @ null_direction, 0., atol=tolerance)


def test_common_action_gauge_and_illegal_targets_do_not_change_design_problem():
    z, error, legal, weights = fabricated_data()
    A, b = M.build_design(z, error, legal, weights)
    gauge = z @ np.array([.25, -.5, 4.])
    shifted = error + gauge[:, None]
    shifted[~legal] += 1000
    other_A, other_b = M.build_design(z, shifted, legal, weights)
    np.testing.assert_array_equal(A, other_A)
    np.testing.assert_allclose(b, other_b, atol=1e-15, rtol=1e-15)
    original = M.solve_design(A, b, mode="ols")
    other = M.solve_design(other_A, other_b, mode="ols")
    np.testing.assert_allclose(original["increment"], other["increment"], atol=2e-14)
    np.testing.assert_allclose(original["increment"].sum(axis=0), 0., atol=1e-15)


def test_fixed_ridge_penalizes_bias_and_matches_closed_form():
    # A single augmented feature represents the bias, so this tests its penalty.
    coefficient = np.array([2., -3., .5])
    z = np.ones((2, 1))
    weights = np.array([.25, .75])
    target = np.tile(reference_basis() @ coefficient, (2, 1))
    A, b = M.build_design(z, target, np.ones((2, 4), bool), weights)
    result = M.solve_design(A, b, mode="ridge")
    expected = coefficient / (1 + 1e-4)
    np.testing.assert_allclose(result["coefficients"], expected, rtol=2e-14, atol=2e-14)
    d = result["diagnostics"]
    assert d["rows"] == A.shape[0] + 3 and d["data_rows"] == A.shape[0]
    assert d["rank"] == 3 and d["ridge"] == 1e-4 and d["rcond"] == 1e-10
    assert d["spectrum_scope"] == "augmented"
    np.testing.assert_allclose(result["singular_values"], np.sqrt(1 + 1e-4), rtol=2e-14)
    residual = A @ result["coefficients"] - b
    np.testing.assert_allclose(d["data_loss"], residual @ residual)
    np.testing.assert_allclose(d["regularization"], 1e-4 * (expected @ expected), rtol=2e-14)
    np.testing.assert_allclose(d["total_objective"], d["data_loss"] + d["regularization"])
    np.testing.assert_allclose(d["normal_residual"], np.linalg.norm(A.T @ residual + 1e-4 * result["coefficients"]))
    assert d["normal_residual"] < 1e-13


@pytest.mark.parametrize("mode", ("ols", "ridge"))
@pytest.mark.parametrize("rows", (0, 2))
def test_zero_support_preserves_parent_with_zero_increment(mode, rows):
    z = np.full((rows, 29), 1e308)
    target = np.full((rows, 4), 1e308)
    A, b = M.build_design(z, target, np.zeros((rows, 4), bool), np.zeros(rows))
    assert A.shape == (4 * rows, 87)
    np.testing.assert_array_equal(A, 0.)
    np.testing.assert_array_equal(b, 0.)
    result = M.solve_design(A, b, mode=mode)
    np.testing.assert_array_equal(result["increment"], np.zeros((4, 29)))
    assert result["diagnostics"]["rank"] == (87 if mode == "ridge" else 0)
    assert result["diagnostics"]["total_objective"] == 0
    data = M.design_diagnostics(A)
    assert data["rank"] == 0 and data["condition_number"] is None


def test_single_legal_action_adds_no_information():
    A, b = M.build_design([[1., 1.]], [[3., 8., 9., 5.]], [[False, True, False, False]], [2.])
    np.testing.assert_array_equal(A, np.zeros((4, 6)))
    np.testing.assert_array_equal(b, np.zeros(4))


def test_inputs_and_global_rng_are_unchanged_and_outputs_are_owned():
    values = fabricated_data()
    for value in values:
        value.flags.writeable = False
    saved = [value.copy() for value in values]
    rng = copy.deepcopy(np.random.get_state())
    A, b = M.build_design(*values)
    saved_A, saved_b = A.copy(), b.copy()
    result = M.solve_design(A, b, mode="ridge")
    M.design_diagnostics(A)
    for current, original in zip(values, saved, strict=True):
        np.testing.assert_array_equal(current, original)
        assert not current.flags.writeable
    np.testing.assert_array_equal(A, saved_A)
    np.testing.assert_array_equal(b, saved_b)
    after = np.random.get_state()
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_array_equal(rng[1], after[1])
    before_B, before_increment = result["B"].copy(), result["increment"].copy()
    result["coefficients"][:] = 0
    np.testing.assert_array_equal(result["B"], before_B)
    np.testing.assert_array_equal(result["increment"], before_increment)


@pytest.mark.parametrize("defect", (
    "feature_rank", "zero_features", "feature_nan", "feature_bool", "feature_complex", "target_shape",
    "target_nan_even_unsupported", "legal_shape", "legal_int", "no_legal", "weight_scalar",
    "weight_negative", "weight_nan", "weight_bool", "weight_shape",
))
def test_invalid_design_inputs_fail_without_mutating_callers(defect):
    z, error, legal, weights = [v.copy() for v in fabricated_data()]
    if defect == "feature_rank":
        z = z.reshape(-1)
    elif defect == "zero_features":
        z = z[:, :0]
    elif defect == "feature_nan":
        z[0, 0] = np.nan
    elif defect == "feature_bool":
        z = z.astype(bool)
    elif defect == "feature_complex":
        z = z.astype(complex)
    elif defect == "target_shape":
        error = error[:, :3]
    elif defect == "target_nan_even_unsupported":
        error[-1, 0] = np.nan
    elif defect == "legal_shape":
        legal = legal[:, :3]
    elif defect == "legal_int":
        legal = legal.astype(int)
    elif defect == "no_legal":
        legal[0] = False
    elif defect == "weight_scalar":
        weights = np.array(1.)
    elif defect == "weight_negative":
        weights[0] = -1
    elif defect == "weight_nan":
        weights[0] = np.nan
    elif defect == "weight_bool":
        weights = weights.astype(bool)
    else:
        weights = weights[:-1]
    before = [v.tobytes() for v in (z, error, legal, weights)]
    with pytest.raises(ValueError):
        M.build_design(z, error, legal, weights)
    assert before == [v.tobytes() for v in (z, error, legal, weights)]


@pytest.mark.parametrize("defect", ("mode", "bool_mode", "rank", "zero_columns", "columns", "nan", "bool",
                                  "complex", "target_shape", "target_nan", "target_bool"))
def test_invalid_solve_inputs_rejected(defect):
    A, b, mode = np.eye(3), np.ones(3), "ols"
    if defect == "mode":
        mode = "automatic"
    elif defect == "bool_mode":
        mode = True
    elif defect == "rank":
        A = A.reshape(-1)
    elif defect == "zero_columns":
        A = A[:, :0]
    elif defect == "columns":
        A = A[:, :2]
    elif defect == "nan":
        A[0, 0] = np.nan
    elif defect == "bool":
        A = A.astype(bool)
    elif defect == "complex":
        A = A.astype(complex)
    elif defect == "target_shape":
        b = b[:, None]
    elif defect == "target_nan":
        b[0] = np.inf
    else:
        b = b.astype(bool)
    with pytest.raises(ValueError):
        M.solve_design(A, b, mode=mode)


def test_fixed_cutoff_is_explicit_and_rank_deficient_ridge_fails(monkeypatch):
    original = np.linalg.lstsq
    calls = []
    def spy(a, b, rcond):
        calls.append((a.copy(), b.copy(), rcond))
        return original(a, b, rcond=rcond)
    monkeypatch.setattr(np.linalg, "lstsq", spy)
    M.solve_design(np.eye(3), np.ones(3), mode="ols")
    M.solve_design(np.eye(3), np.ones(3), mode="ridge")
    assert [c[2] for c in calls] == [1e-10, 1e-10]
    np.testing.assert_array_equal(calls[1][0][-3:], .01 * np.eye(3))
    np.testing.assert_array_equal(calls[1][1][-3:], np.zeros(3))
    with pytest.raises(ValueError, match="ridge matrix numerically rank deficient"):
        M.solve_design(np.diag([1e12, 0., 0.]), np.ones(3), mode="ridge")


def test_numerical_failures_are_explicit_and_do_not_mutate_inputs(monkeypatch):
    z, error = np.array([[1e308]]), np.ones((1, 4))
    before = z.tobytes(), error.tobytes()
    with pytest.raises(ValueError, match="weighted design"):
        M.build_design(z, error, np.ones((1, 4), bool), [1e308])
    assert before == (z.tobytes(), error.tobytes())
    with pytest.raises(ValueError, match="arithmetic|diagnostics"):
        M.solve_design(np.zeros((1, 3)), [1e308], mode="ols")
    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("fabricated SVD failure")
    monkeypatch.setattr(np.linalg, "lstsq", fail)
    with pytest.raises(ValueError, match="least-squares solve"):
        M.solve_design(np.eye(3), np.ones(3), mode="ols")
    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(ValueError, match="design SVD failed"):
        M.design_diagnostics(np.eye(3))
