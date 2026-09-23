"""Independent fabricated Bayesian residual-memory reference checks.

No learned checkpoint, scientific example, simulator or experimental study is
used. Batch contrast-space regression supplies the full-covariance oracle.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from openjev.research import bayesian_score_memory as bayes

CONTRAST = np.array([[1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
                     [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
                     [0., -2 / math.sqrt(6), 1 / math.sqrt(12)],
                     [0., 0., -3 / math.sqrt(12)]], dtype=np.float64)
CENTER = np.eye(4, dtype=np.float64) - np.ones((4, 4), dtype=np.float64) / 4


def memory(dim=3, mode="full", prior=1.7, noise=.4):
    return bayes.BayesianScoreMemory(dim, prior_variance=prior, noise_variance=noise, mode=mode)


def batch_posterior(cues, targets, prior, noise):
    """Independent information-form regression in three orthonormal contrasts."""
    dim = cues.shape[1]
    precision = np.eye(dim) / prior + cues.T @ cues / noise
    covariance = np.linalg.solve(precision, np.eye(dim))
    contrast_targets = targets @ CONTRAST
    coefficients = np.linalg.solve(precision, cues.T @ contrast_targets / noise)
    mean = CONTRAST @ coefficients.T
    return mean, covariance


def same_state(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        if isinstance(actual[name], np.ndarray):
            assert actual[name].dtype == expected[name].dtype
            assert actual[name].shape == expected[name].shape
            assert actual[name].tobytes() == expected[name].tobytes()
        else:
            assert actual[name] == expected[name]


def covariance(state, mode):
    return state["covariance"] if mode == "full" else np.diag(state["covariance"])


@pytest.mark.parametrize("dim", (1, 3, 8))
def test_full_sequential_updates_equal_independent_batch_contrast_posterior(dim):
    rng = np.random.default_rng(722)
    cues = rng.normal(size=(23, dim))
    targets = rng.normal(size=(23, 3)) @ CONTRAST.T
    model = memory(dim)
    for count, (cue, target) in enumerate(zip(cues, targets, strict=True), 1):
        model.observe(cue, target)
        expected_mean, expected_covariance = batch_posterior(cues[:count], targets[:count], 1.7, .4)
        state = model.state_dict()
        np.testing.assert_allclose(state["weights"], expected_mean, rtol=5e-12, atol=5e-13)
        np.testing.assert_allclose(state["covariance"], expected_covariance, rtol=5e-12, atol=5e-13)
        assert state["weights"].dtype == state["covariance"].dtype == np.float64


def test_full_update_is_permutation_invariant_and_correlated_cues_are_valid_regressors():
    rng = np.random.default_rng(480)
    latent = rng.normal(size=(31, 1))
    cues = latent @ np.array([[1., -.6, .8]]) + rng.normal(size=(31, 3)) * .03
    coefficients = rng.normal(size=(4, 3))
    coefficients -= coefficients.mean(axis=0)
    targets = cues @ coefficients.T + rng.normal(size=(31, 3)) @ CONTRAST.T * .2
    orders = (np.arange(31), np.arange(30, -1, -1), rng.permutation(31))
    expected_mean, expected_covariance = batch_posterior(cues, targets, 1.7, .4)
    for order in orders:
        model = memory()
        for index in order:
            model.observe(cues[index], targets[index])
        state = model.state_dict()
        np.testing.assert_allclose(state["weights"], expected_mean, rtol=2e-11, atol=2e-12)
        np.testing.assert_allclose(state["covariance"], expected_covariance, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_prediction_reads_only_and_observe_returns_immutable_prewrite_prediction(mode):
    model = memory(mode=mode)
    cue = np.array([1., -.4, .7])
    initial = model.state_dict()
    first = model.predict(cue)
    same_state(model.state_dict(), initial)
    second = model.predict(cue)
    same_state(model.state_dict(), initial)
    np.testing.assert_array_equal(first.mean, second.mean)
    result = model.observe(cue, np.array([1., -1., 2., -2.]))
    np.testing.assert_array_equal(result.mean, first.mean)
    assert result.epistemic_variance == first.epistemic_variance
    snapshot = copy.deepcopy(result)
    learned = model.state_dict()
    shrink = covariance(initial, mode) - covariance(learned, mode)
    assert np.linalg.eigvalsh(shrink).min() >= -1e-13
    assert np.trace(shrink) > 0
    model.predict(cue)
    same_state(model.state_dict(), learned)
    model.observe(np.array([-.2, .3, .9]), np.array([3., 0., -2., -1.]))
    np.testing.assert_array_equal(result.mean, snapshot.mean)
    assert result.epistemic_variance == snapshot.epistemic_variance


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_centered_mean_and_contrast_uncertainty_are_not_action_probabilities(mode):
    model = memory(mode=mode)
    cue = np.array([1., 2., -1.])
    model.observe(cue, np.array([3., 2., -1., -4.]))
    state, prediction = model.state_dict(), model.predict(cue)
    np.testing.assert_allclose(state["weights"].sum(axis=0), 0, rtol=0, atol=2e-14)
    assert abs(prediction.mean.sum()) < 2e-14
    variance = float(cue @ covariance(state, mode) @ cue)
    assert prediction.epistemic_variance == pytest.approx(variance, rel=1e-13, abs=1e-14)
    assert prediction.epistemic_variance != pytest.approx(variance + .4)
    # The scalar is variance in each orthonormal contrast, not four independent
    # action-coordinate variances and not probability of correctness.
    covariance_output = prediction.epistemic_variance * CENTER
    np.testing.assert_allclose(covariance_output @ np.ones(4), 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(covariance_output)[1:], variance, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError):
        model.observe(cue, np.array([103., 102., 99., 96.]))
    same_state(model.state_dict(), state)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_zero_cue_has_no_weight_or_covariance_information(mode):
    model = memory(mode=mode)
    initial = model.state_dict()
    returned = model.observe(np.zeros(3), np.array([-1.5, -.5, .5, 1.5]))
    state = model.state_dict()
    np.testing.assert_array_equal(state["weights"], initial["weights"])
    np.testing.assert_array_equal(state["covariance"], initial["covariance"])
    np.testing.assert_array_equal(returned.mean, np.zeros(4))
    assert returned.epistemic_variance == 0
    model.observe(np.array([1., 0., 0.]), np.zeros(4))
    state = model.state_dict()
    np.testing.assert_array_equal(state["weights"], np.zeros((4, 3)))
    assert covariance(state, mode)[0, 0] < 1.7


def test_one_dimensional_full_and_diagonal_models_match_every_update():
    full, diagonal = memory(1), memory(1, "diagonal")
    for value in (0., 1., -2., .01, 1., 3., -.4):
        cue = np.array([value])
        target = np.array([2. * value, -value, value, -2. * value])
        full.observe(cue, target)
        diagonal.observe(cue, target)
        np.testing.assert_allclose(full.state_dict()["weights"], diagonal.state_dict()["weights"], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(full.state_dict()["covariance"], np.diag(diagonal.state_dict()["covariance"]), rtol=2e-14, atol=2e-14)
        a, b = full.predict(cue), diagonal.predict(cue)
        np.testing.assert_allclose(a.mean, b.mean, rtol=2e-14, atol=2e-14)
        assert a.epistemic_variance == pytest.approx(b.epistemic_variance, rel=2e-14, abs=2e-14)


def test_diagonal_is_moment_projection_and_loses_cross_direction_information():
    full, diagonal = memory(2, prior=1., noise=1.), memory(2, "diagonal", prior=1., noise=1.)
    cue, residual = np.ones(2), np.array([1., -1., 0., 0.])
    for model in (full, diagonal):
        model.observe(cue, residual)
    np.testing.assert_allclose(full.state_dict()["covariance"], np.array([[2., -1.], [-1., 2.]]) / 3, atol=1e-15)
    np.testing.assert_allclose(diagonal.state_dict()["covariance"], np.array([2., 2.]) / 3, atol=1e-15)
    # A diagonal-precision update would produce 1/2 instead of 2/3 and is a
    # different approximation from dropping posterior cross-covariances.
    assert not np.allclose(diagonal.state_dict()["covariance"], [.5, .5])
    next_cue = np.array([1., -1.])
    assert float(next_cue @ full.state_dict()["covariance"] @ next_cue) == pytest.approx(2.)
    assert float(next_cue @ np.diag(diagonal.state_dict()["covariance"]) @ next_cue) == pytest.approx(4 / 3)
    target2 = np.array([0., 1., -1., 0.])
    for model in (full, diagonal):
        model.observe(next_cue, target2)
    np.testing.assert_allclose(full.state_dict()["weights"], np.column_stack(((residual + target2) / 3, (residual - target2) / 3)), atol=1e-15)
    np.testing.assert_allclose(full.state_dict()["covariance"], np.eye(2) / 3, atol=1e-15)
    np.testing.assert_allclose(diagonal.state_dict()["weights"], np.column_stack((residual / 3 + 2 * target2 / 7,
                                                                                 residual / 3 - 2 * target2 / 7)), atol=1e-15)
    np.testing.assert_allclose(diagonal.state_dict()["covariance"], np.full(2, 10 / 21), atol=1e-15)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_export_prediction_and_inputs_do_not_alias_model_state(mode):
    model = memory(mode=mode)
    cue, residual = np.array([1., -.5, .25]), np.array([2., -1., 0., -1.])
    original_cue, original_residual = cue.copy(), residual.copy()
    prediction = model.observe(cue, residual)
    expected = model.state_dict()
    np.testing.assert_array_equal(cue, original_cue)
    np.testing.assert_array_equal(residual, original_residual)
    cue[:] = 99
    residual[:] = 99
    exported = model.state_dict()
    for value in exported.values():
        if isinstance(value, np.ndarray):
            value[:] = 99
    prediction.mean[:] = 99
    same_state(model.state_dict(), expected)
    read = model.predict(original_cue)
    read.mean[:] = -99
    same_state(model.state_dict(), expected)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_reset_restores_explicit_prior_and_zeros_all_learned_state(mode):
    model = memory(mode=mode)
    initial = model.state_dict()
    model.observe(np.array([1., 2., 3.]), np.array([1., -1., -2., 2.]))
    assert np.any(model.state_dict()["weights"] != 0)
    model.reset()
    same_state(model.state_dict(), initial)
    model.reset()
    same_state(model.state_dict(), initial)


def test_repeated_collinear_cues_follow_closed_form_without_collapsing_orthogonal_uncertainty():
    count, prior, noise = 400, 2., .001
    cue = np.array([1., 1., 1.]) / math.sqrt(3)
    residual = np.array([1., -2., 3., -2.])
    model = memory(prior=prior, noise=noise)
    for _ in range(count):
        model.observe(cue, residual)
    expected_covariance = prior * np.eye(3) - count * prior**2 * np.outer(cue, cue) / (noise + count * prior)
    expected_mean = count * prior * np.outer(residual, cue) / (noise + count * prior)
    state = model.state_dict()
    np.testing.assert_allclose(state["covariance"], expected_covariance, rtol=5e-9, atol=5e-11)
    np.testing.assert_allclose(state["weights"], expected_mean, rtol=5e-9, atol=5e-11)
    eigenvalues = np.linalg.eigvalsh(state["covariance"])
    assert eigenvalues.min() > 0
    np.testing.assert_allclose(eigenvalues[1:], [prior, prior], rtol=5e-11, atol=5e-11)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_frozen_export_import_is_owned_atomic_and_replays_the_same_future(mode):
    original = memory(mode=mode)
    original.observe(np.array([1., .25, -.5]), np.array([1., -2., 3., -2.]))
    exported = original.state_dict()
    restored = memory(mode=mode)
    restored.load_state(exported)
    same_state(restored.state_dict(), original.state_dict())
    exported["weights"][:] = 99
    exported["covariance"][:] = 99
    same_state(restored.state_dict(), original.state_dict())
    cues = (np.array([0., 1., 0.]), np.array([.2, -.1, 2.]), np.zeros(3))
    targets = (np.array([2., 1., -1., -2.]), np.array([1., -1., 0., 0.]), np.zeros(4))
    for cue, target in zip(cues, targets, strict=True):
        a, b = original.observe(cue, target), restored.observe(cue, target)
        np.testing.assert_array_equal(a.mean, b.mean)
        assert a.epistemic_variance == b.epistemic_variance
        same_state(restored.state_dict(), original.state_dict())


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_covariance_and_label_counts_distinguish_reads_from_labeled_updates(mode):
    model = memory(mode=mode)
    assert model.counts() == {"mode": mode, "covariance_entries": 9 if mode == "full" else 3,
                              "mean_entries": 12, "labeled_updates": 0}
    for _ in range(5):
        model.predict(np.ones(3))
    assert model.counts()["labeled_updates"] == 0
    model.observe(np.zeros(3), np.zeros(4))
    assert model.counts()["labeled_updates"] == 1
    model.observe(np.ones(3), np.array([1., -1., 0., 0.]))
    assert model.counts()["labeled_updates"] == model.state_dict()["labeled_updates"] == 2
    model.reset()
    assert model.counts()["labeled_updates"] == 0


def test_coordinate_axis_observations_match_full_and_diagonal_posteriors():
    full, diagonal = memory(), memory(mode="diagonal")
    rng = np.random.default_rng(103)
    for index in range(18):
        cue = np.zeros(3)
        cue[index % 3] = (-1)**index * (1 + index / 10)
        target = rng.normal(size=3) @ CONTRAST.T
        full.observe(cue, target)
        diagonal.observe(cue, target)
        np.testing.assert_allclose(full.state_dict()["weights"], diagonal.state_dict()["weights"], rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(full.state_dict()["covariance"], np.diag(diagonal.state_dict()["covariance"]),
                                   rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("key_dim", (0, -1, True, 2.5, "3", None))
def test_key_dimension_is_an_explicit_positive_integer(key_dim):
    with pytest.raises((TypeError, ValueError)):
        bayes.BayesianScoreMemory(key_dim, prior_variance=1., noise_variance=1.)


@pytest.mark.parametrize("name", ("prior_variance", "noise_variance"))
@pytest.mark.parametrize("value", (0., -1., float("nan"), float("inf"), -float("inf"), True, None, "1"))
def test_variances_are_explicit_finite_positive_real_scalars(name, value):
    args = {"prior_variance": 1., "noise_variance": 1., name: value}
    with pytest.raises((TypeError, ValueError)):
        bayes.BayesianScoreMemory(3, **args)


def test_variance_settings_have_no_silent_empirical_defaults():
    with pytest.raises(TypeError):
        bayes.BayesianScoreMemory(3)
    with pytest.raises(TypeError):
        bayes.BayesianScoreMemory(3, prior_variance=1.)
    with pytest.raises(TypeError):
        bayes.BayesianScoreMemory(3, noise_variance=1.)


@pytest.mark.parametrize("mode", ("FULL", "diagonal_precision", "", None, True))
def test_only_the_two_declared_covariance_modes_are_allowed(mode):
    with pytest.raises((TypeError, ValueError)):
        memory(mode=mode)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
@pytest.mark.parametrize("cue", (np.ones(2), np.ones((1, 3)), np.ones((3, 1)),
    np.array([1., float("nan"), 0.]), np.array([0., float("inf"), 1.]),
    np.array([1j, 0j, 0j]), np.array([True, False, True]), np.array(["1", "0", "0"])))
def test_invalid_cue_repeatedly_fails_without_changing_state_or_counts(mode, cue):
    model = memory(mode=mode)
    model.observe(np.ones(3), np.array([1., -1., 0., 0.]))
    before = model.state_dict()
    for _ in range(2):
        with pytest.raises((ValueError, TypeError)):
            model.predict(cue)
        same_state(model.state_dict(), before)
        with pytest.raises((ValueError, TypeError)):
            model.observe(cue, np.zeros(4))
        same_state(model.state_dict(), before)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
@pytest.mark.parametrize("target", (np.ones(3), np.zeros((1, 4)), np.zeros((4, 1)), np.ones(4),
    np.array([1., float("nan"), -1., 0.]), np.array([float("inf"), -float("inf"), 0., 0.])))
def test_invalid_or_materially_uncentered_target_fails_without_partial_covariance_update(mode, target):
    model = memory(mode=mode)
    model.observe(np.ones(3), np.array([1., -1., 0., 0.]))
    before = model.state_dict()
    for _ in range(2):
        with pytest.raises((ValueError, TypeError)):
            model.observe(np.array([0., 1., 0.]), target)
        same_state(model.state_dict(), before)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_finite_inputs_causing_numerical_overflow_fail_atomically(mode):
    model = memory(mode=mode)
    model.observe(np.array([1., 0., 0.]), np.array([1., -1., 0., 0.]))
    before = model.state_dict()
    for _ in range(2):
        with pytest.raises((ValueError, FloatingPointError)):
            model.observe(np.array([1e200, 1e200, 1e200]), np.zeros(4))
        same_state(model.state_dict(), before)
        with pytest.raises((ValueError, FloatingPointError)):
            model.predict(np.array([1e200, 1e200, 1e200]))
        same_state(model.state_dict(), before)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
@pytest.mark.parametrize("defect", ("version", "dimension", "mode", "prior", "noise", "missing", "extra", "weights_shape",
                                   "weights_nan", "weights_uncentered", "covariance_shape", "covariance_nan",
                                   "covariance_nonpositive", "weights_dtype", "covariance_dtype", "count_negative", "count_bool"))
def test_malformed_import_preserves_the_entire_existing_state(mode, defect):
    model = memory(mode=mode)
    model.observe(np.array([1., .5, 0.]), np.array([1., -1., 0., 0.]))
    before = model.state_dict()
    candidate = copy.deepcopy(before)
    if defect == "version":
        candidate["version"] = "wrong"
    elif defect == "dimension":
        candidate["key_dim"] = 4
    elif defect == "mode":
        candidate["mode"] = "diagonal" if mode == "full" else "full"
    elif defect == "prior":
        candidate["prior_variance"] *= 2
    elif defect == "noise":
        candidate["noise_variance"] *= 2
    elif defect == "missing":
        candidate.pop("weights")
    elif defect == "extra":
        candidate["unregistered"] = True
    elif defect == "weights_shape":
        candidate["weights"] = np.zeros((3, 4))
    elif defect == "weights_nan":
        candidate["weights"][0, 0] = float("nan")
    elif defect == "weights_uncentered":
        candidate["weights"][0, 0] += 1
    elif defect == "covariance_shape":
        candidate["covariance"] = np.ones((1, 1, 3))
    elif defect == "covariance_nan":
        candidate["covariance"].flat[0] = float("nan")
    elif defect == "covariance_nonpositive":
        candidate["covariance"].flat[0] = -1
    elif defect == "weights_dtype":
        candidate["weights"] = candidate["weights"].astype(np.float32)
    elif defect == "covariance_dtype":
        candidate["covariance"] = candidate["covariance"].astype(np.float32)
    elif defect == "count_negative":
        candidate["labeled_updates"] = -1
    elif defect == "count_bool":
        candidate["labeled_updates"] = True
    for _ in range(2):
        with pytest.raises((ValueError, TypeError)):
            model.load_state(candidate)
        same_state(model.state_dict(), before)


def test_full_covariance_import_rejects_asymmetry_and_indefiniteness():
    model = memory()
    before = model.state_dict()
    for matrix in (np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]),
                   np.array([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]])):
        candidate = copy.deepcopy(before)
        candidate["covariance"] = matrix
        with pytest.raises(ValueError):
            model.load_state(candidate)
        same_state(model.state_dict(), before)


@pytest.mark.parametrize("mode", ("full", "diagonal"))
def test_from_state_preserves_complete_explicit_configuration_without_array_aliases(mode):
    original = memory(8, mode, prior=2.5, noise=.125)
    original.observe(np.arange(8, dtype=np.float64), np.array([1., -1., 2., -2.]))
    state = original.state_dict()
    reconstructed = bayes.BayesianScoreMemory.from_state(state)
    same_state(reconstructed.state_dict(), state)
    state["weights"][:] = 99
    state["covariance"][:] = 99
    same_state(reconstructed.state_dict(), original.state_dict())
