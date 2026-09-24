"""Fabricated dense-Gaussian and sequential-information qualification only."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from openjev.research import residual_memory as memory


def fixture():
    bx = np.array([[[-1., .5], [0., 0.], [.5, -1.], [1., .5]],
                   [[-.5, -.5], [.25, .75], [1., 0.], [-1., -1.]]])
    by = np.array([[.5, -.25, 1., .75], [-.5, .25, -.75, 1.]])
    fx, fy, qx = np.array([[.2, -.1], [.5, .25], [-.25, .5]]), np.array([.25, -.5, .4]), np.array([.6, -.4])
    return bx, by, fx, fy, qx


def kernel(left, right, length, amplitude):
    return np.array([[amplitude**2 * math.exp(-sum((float(a) - float(b))**2
                                                  for a, b in zip(x, y, strict=True)) / (2 * length**2))
                      for y in right] for x in left])


def surrogate(points, mode, length, amplitude, noise):
    true = kernel(points, points, length, amplitude)
    if mode == "full":
        covariance = true
    else:
        grid = np.array([(a, b) for a in np.linspace(-2, 2, 4) for b in np.linspace(-2, 2, 4)])
        cross = kernel(points, grid, length, amplitude)
        inducing = kernel(grid, grid, length, amplitude) + 1e-6 * np.eye(16)
        covariance = cross @ np.linalg.solve(inducing, cross.T)
        residual = np.diag(true) - np.diag(covariance)
        assert (residual >= -1e-12).all()
        if mode == "fic":
            covariance += np.diag(np.maximum(residual, 0))
        elif mode == "query_only":
            covariance[-1, -1] += max(residual[-1], 0)
    return covariance + noise * np.eye(len(points))


def log_density(values, covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    return -.5 * (len(values) * math.log(2 * math.pi) + logdet
                  + values @ np.linalg.solve(covariance, values))


def dense_reference(bx, by, fx, fy, qx, mode, length=.8, amplitude=1.2, noise=.07):
    means, variances, evidence = [], [], []
    for x, y in zip(bx, by, strict=True):
        covariance = surrogate(np.concatenate((x, fx, qx[None])), mode, length, amplitude, noise)
        observed = np.concatenate((y, fy))
        mean = covariance[-1, :-1] @ np.linalg.solve(covariance[:-1, :-1], observed)
        variance = covariance[-1, -1] - covariance[-1, :-1] @ np.linalg.solve(
            covariance[:-1, :-1], covariance[:-1, -1])
        means.append(mean)
        variances.append(variance)
        evidence.append(log_density(observed, covariance[:-1, :-1])
                        - log_density(y, covariance[:len(y), :len(y)]))
    normalizer = max(evidence) + math.log(sum(math.exp(v - max(evidence)) for v in evidence))
    weights = np.array(evidence) - normalizer
    positive = sum(math.exp(w) * .5 * (1 + math.erf(m / math.sqrt(2 * v)))
                   for w, m, v in zip(weights, means, variances, strict=True))
    return {"component_mean": np.array(means), "component_variance": np.array(variances),
            "log_weights": weights, "prob_positive": positive}


@pytest.mark.parametrize("mode", memory.MODES)
def test_each_covariance_rule_matches_independent_dense_conditioning(mode):
    bx, by, fx, fy, qx = fixture()
    cache = memory.build_archive(bx, by, length=.8, amplitude=1.2, mode=mode, noise_variance=.07)
    diagnostics = {}
    actual = memory.predict(cache, fx, fy, qx, diagnostics=diagnostics)
    expected = dense_reference(bx, by, fx, fy, qx, mode)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=2e-11)
    assert actual["component_mean"].shape == (2,)
    assert actual["component_variance"].shape == actual["log_weights"].shape == (2,)
    assert type(actual["prob_positive"]) is float
    np.testing.assert_allclose(np.exp(actual["log_weights"]).sum(), 1., atol=1e-14)
    assert set(diagnostics) == set(memory.DIAGNOSTIC_KEYS)
    assert diagnostics["residual_points"] == (4 if mode == "fic" else 1 if mode == "query_only" else 0)
    assert cache.diagnostics["residual_points"] == (8 if mode == "fic" else 0)


def test_fic_joint_fewshot_evidence_is_not_independent_marginal_scoring():
    bx, by, fx, fy, qx = fixture()
    cache = memory.build_archive(bx, by, length=1., amplitude=1., mode="fic")
    actual = memory.predict(cache, fx, fy, qx)
    wrong = []
    for x, y in zip(bx, by, strict=True):
        cov = surrogate(np.concatenate((x, fx)), "fic", 1., 1., .0225)
        n = len(x)
        cross = cov[n:, :n]
        mean = cross @ np.linalg.solve(cov[:n, :n], y)
        conditional = cov[n:, n:] - cross @ np.linalg.solve(cov[:n, :n], cross.T)
        assert np.max(np.abs(conditional - np.diag(np.diag(conditional)))) > 1e-3
        wrong.append(-.5 * np.sum(np.log(2 * np.pi * np.diag(conditional)) + (fy - mean)**2 / np.diag(conditional)))
    wrong = np.array(wrong)
    wrong -= max(wrong) + math.log(sum(math.exp(v - max(wrong)) for v in wrong))
    assert not np.allclose(actual["log_weights"], wrong, atol=1e-8, rtol=1e-8)


def test_query_only_preserves_sor_means_and_weights_and_adds_only_query_residual():
    bx, by, fx, fy, qx = fixture()
    sor_cache = memory.build_archive(bx, by, length=.8, amplitude=1.2, mode="sor")
    query_cache = memory.build_archive(bx, by, length=.8, amplitude=1.2, mode="query_only")
    for key in sor_cache.arrays:
        np.testing.assert_array_equal(sor_cache.arrays[key], query_cache.arrays[key])
    sor, corrected = memory.predict(sor_cache, fx, fy, qx), memory.predict(query_cache, fx, fy, qx)
    np.testing.assert_array_equal(sor["component_mean"], corrected["component_mean"])
    np.testing.assert_array_equal(sor["log_weights"], corrected["log_weights"])
    grid = sor_cache.arrays["inducing_grid"]
    cross = kernel(qx[None], grid, .8, 1.2)
    residual = 1.2**2 - (cross @ np.linalg.solve(kernel(grid, grid, .8, 1.2) + 1e-6*np.eye(16), cross.T)).item()
    np.testing.assert_allclose(corrected["component_variance"] - sor["component_variance"], residual, atol=2e-15)


@pytest.mark.parametrize("mode", memory.MODES)
def test_farfield_prior_variance_and_zero_mean_are_explicit(mode):
    bx, by, fx, fy, _ = fixture()
    cache = memory.build_archive(bx, by, length=1., amplitude=1.3, mode=mode, noise_variance=.04)
    pred = memory.predict(cache, fx, fy, np.array([100., 100.]))
    np.testing.assert_array_equal(pred["component_mean"], [0., 0.])
    expected = .04 if mode == "sor" else .04 + 1.3**2
    np.testing.assert_allclose(pred["component_variance"], expected, atol=1e-15)
    assert pred["prob_positive"] == pytest.approx(.5, abs=1e-14)


def test_sequential_information_has_rational_solution_and_matches_batch():
    features = np.array([[[1., 0.], [0., 2.], [1., 1.]]])
    targets, variance = np.array([[1., -1., .5]]), np.array([[1., 2., 4.]])
    saved = [x.copy() for x in (features, targets, variance)]
    result = memory.accumulate_information(features, targets, variance)
    np.testing.assert_array_equal(result["precision"], [[[2.25, .25], [.25, 3.25]]])
    np.testing.assert_array_equal(result["eta"], [[1.125, -.875]])
    expected = np.eye(2) + np.einsum("knr,kns,kn->krs", features, features, 1/variance)
    np.testing.assert_allclose(result["precision"], expected, atol=1e-15)
    np.testing.assert_allclose(result["eta"], np.einsum("knr,kn,kn->kr", features, targets, 1/variance), atol=1e-15)
    reordered = memory.accumulate_information(features[:, ::-1], targets[:, ::-1], variance[:, ::-1])
    for key in result:
        np.testing.assert_array_equal(reordered[key], result[key])
    for before, after in zip(saved, (features, targets, variance), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("mode", memory.MODES)
def test_permutation_stateless_replay_and_no_mutation(mode):
    bx, by, fx, fy, qx = fixture()
    saved = [x.copy() for x in (bx, by, fx, fy, qx)]
    cache = memory.build_archive(bx, by, length=1., amplitude=1., mode=mode)
    arrays_before = {key: value.copy() for key, value in cache.arrays.items()}
    original = memory.predict(cache, fx, fy, qx)
    reordered_cache = memory.build_archive(bx[::-1, ::-1], by[::-1, ::-1], length=1., amplitude=1., mode=mode)
    reordered = memory.predict(reordered_cache, fx[::-1], fy[::-1], qx)
    for key in ("component_mean", "component_variance", "log_weights"):
        np.testing.assert_allclose(reordered[key], original[key][::-1], atol=2e-12, rtol=2e-12)
    assert reordered["prob_positive"] == pytest.approx(original["prob_positive"], abs=2e-12)
    for _ in range(2):
        repeated = memory.predict(cache, fx, fy, qx)
        for key in repeated:
            np.testing.assert_array_equal(repeated[key], original[key])
    for key, before in arrays_before.items():
        np.testing.assert_array_equal(cache.arrays[key], before)
    for before, after in zip(saved, (bx, by, fx, fy, qx), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("mode", memory.MODES)
def test_owned_cache_storage_and_public_boundary(mode):
    bx, by, fx, fy, qx = fixture()
    cache = memory.build_archive(bx, by, length=1., amplitude=1., mode=mode)
    if mode == "full":
        assert set(cache.arrays) == {"archive_x", "archive_factor", "archive_alpha"}
        assert cache.array_bytes == 8 * 2 * (4**2 + 3 * 4)
    else:
        assert set(cache.arrays) == {"inducing_grid", "feature_factor", "posterior_covariance", "weight_mean"}
        assert cache.array_bytes == 8 * (32 + 256 + 2 * (256 + 16))
    for value in cache.arrays.values():
        assert value.dtype == np.float64 and value.flags.owndata and not value.flags.writeable
        assert not np.shares_memory(value, bx) and not np.shares_memory(value, by)
    before = memory.predict(cache, fx, fy, qx)
    bx[:] = 100
    by[:] = -100
    after = memory.predict(cache, fx, fy, qx)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert tuple(inspect.signature(memory.predict).parameters) == ("cache", "fx", "fy", "qx", "diagnostics")
    for name in ("target", "true_index", "private_selected_block"):
        with pytest.raises(TypeError, match=name):
            memory.predict(cache, fx, fy, qx, **{name: 0})


def test_duplicate_coordinate_fic_is_declared_independent_residual_approximation():
    # Same location far from the grid has effectively no shared feature part.
    bx = np.full((1, 2, 2), 100.)
    by, fx, fy, qx = np.array([[1., 1.]]), np.full((2, 2), 100.), np.array([1., 1.]), np.full(2, 100.)
    fic = memory.predict(memory.build_archive(bx, by, length=1., amplitude=1., mode="fic"), fx, fy, qx)
    full = memory.predict(memory.build_archive(bx, by, length=1., amplitude=1., mode="full"), fx, fy, qx)
    np.testing.assert_array_equal(fic["component_mean"], [0.])
    np.testing.assert_allclose(fic["component_variance"], [1.0225], atol=1e-15)
    np.testing.assert_allclose(full["component_mean"], [4 / (4 + .0225)], atol=2e-14)
    np.testing.assert_allclose(full["component_variance"], [.0225 + .0225 / (4 + .0225)], atol=2e-14)


def test_residual_roundoff_is_explicit_counted_and_material_failure_not_repaired():
    diagnostics = {"residual_points": 0, "minimum_raw_residual": None, "residual_floor_count": 0}
    features = np.array([[math.sqrt(1 + 5e-13)], [.5]])
    raw = 1. - np.square(features).sum(-1)
    result = memory._residual(features, 1., diagnostics)
    np.testing.assert_array_equal(result, [0., .75])
    assert diagnostics == {"residual_points": 2, "minimum_raw_residual": float(raw.min()), "residual_floor_count": 1}
    failed = {"residual_points": 0, "minimum_raw_residual": None, "residual_floor_count": 0}
    with pytest.raises(ValueError, match="materially negative"):
        memory._residual(np.array([[math.sqrt(1 + 2e-12)]]), 1., failed)
    assert failed["residual_floor_count"] == 0 and failed["minimum_raw_residual"] < -1e-12


@pytest.mark.parametrize("name,value", [("length", 0.), ("amplitude", -1.), ("noise_variance", 0.),
                                       ("length", math.inf), ("amplitude", math.nan), ("length", True)])
def test_invalid_kernel_scalars_rejected(name, value):
    bx, by, *_ = fixture()
    arguments = {"length": 1., "amplitude": 1., name: value}
    with pytest.raises(ValueError):
        memory.build_archive(bx, by, **arguments)


def test_shapes_nonfinite_values_diagnostic_ownership_and_failure_no_retry(monkeypatch):
    bx, by, fx, fy, qx = fixture()
    for bad_bx, bad_by in ((bx.astype(np.float32), by), (bx, by[:, :1]), (bx, by * np.nan)):
        with pytest.raises(ValueError):
            memory.build_archive(bad_bx, bad_by, length=1., amplitude=1.)
    with pytest.raises(ValueError, match="mode"):
        memory.build_archive(bx, by, length=1., amplitude=1., mode="unknown")
    cache = memory.build_archive(bx, by, length=1., amplitude=1.)
    for args in ((fx, fy[:1], qx), (fx, fy, qx[:1]), (fx.astype(np.float32), fy, qx),
                 (fx, fy * np.nan, qx), (fx[:0], fy[:0], qx)):
        with pytest.raises(ValueError):
            memory.predict(cache, *args)
    with pytest.raises(ValueError, match="empty dict"):
        memory.predict(cache, fx, fy, qx, diagnostics={"old": 1})
    with pytest.raises(ValueError, match="positive event"):
        memory.accumulate_information(np.ones((1, 2, 2)), np.ones((1, 2)), np.zeros((1, 2)))
    calls = []
    def fail(matrix, **kwargs):
        calls.append(matrix.copy())
        raise np.linalg.LinAlgError("fabricated failure")
    monkeypatch.setattr(memory, "cholesky", fail)
    with pytest.raises(ValueError, match="no retry"):
        memory.predict(cache, fx, fy, qx)
    assert len(calls) == 1
