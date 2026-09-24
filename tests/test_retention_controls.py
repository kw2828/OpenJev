"""Independent fabricated dense-conditioning and bounded-storage witnesses."""
from __future__ import annotations

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from openjev.research import retention_controls as controls


def covariance(left, right):
    return np.array([[math.exp(-sum((float(a)-float(b))**2 for a, b in zip(x, z, strict=True))/2)
                      + (1e-5 if np.array_equal(x, z) else 0.) for z in right] for x in left])


def fixture(batch=2):
    points = np.array([[-1.5, -1.], [-.5, .5], [.5, -.5], [1.5, 1.], [-1., 1.5]])
    x = np.stack([points + np.array([b/8, -b/16]) for b in range(batch)])
    y = np.stack([np.array([.5, -.25, .75, 1., -.5])+b/5 for b in range(batch)])
    path = np.array([[-.75, -.25], [-.25, -.25], [.25, -.25], [.75, -.25]])
    paths = np.empty((batch, 2, 4, 4, 2))
    for b in range(batch):
        for q in range(2):
            for a in range(4):
                paths[b, q, a] = path + np.array([b/8, q/4+a/8])
    return x, y, paths


def tail(mean, variance):
    return .5*math.erfc((.5-float(mean))/math.sqrt(2*float(variance)))


def raw_dense(x, y, path):
    observed = covariance(x, x)+.09*np.eye(len(x))
    cross = covariance(path, x).mean(axis=0)
    mean = cross @ np.linalg.solve(observed, y)
    variance = covariance(path, path).sum()/16-cross @ np.linalg.solve(observed, cross)
    return mean, variance, tail(mean, variance)


def fic_dense(anchors, x, y, path):
    """Batch FIC covariance, independent of the sequential Kalman code."""
    prior = covariance(anchors, anchors)
    observation_cross = covariance(x, anchors)
    query_cross = covariance(path, anchors)
    observation_lowrank = observation_cross @ np.linalg.solve(prior, observation_cross.T)
    query_lowrank = query_cross @ np.linalg.solve(prior, query_cross.T)
    cross = query_cross @ np.linalg.solve(prior, observation_cross.T)
    observation_residual = 1.00001-np.diag(observation_lowrank)
    observed = observation_lowrank+np.diag(observation_residual+.09)
    query_prior = query_lowrank+np.diag(1.00001-np.diag(query_lowrank))
    query_mean = cross @ np.linalg.solve(observed, y)
    query_cov = query_prior-cross @ np.linalg.solve(observed, cross.T)
    mean, variance = query_mean.mean(), query_cov.sum()/16
    return mean, variance, tail(mean, variance)


def test_exact_declared_state_bytes_and_no_hidden_storage():
    raw = controls.initial_raw(3)
    fic = controls.initial_fic(3)
    assert raw.array_bytes == 3*984 and raw.resident_bytes_per_context == 1024
    assert fic.array_bytes == 3*864 and fic.resident_bytes_per_context == 904
    assert set(vars(raw)) == {"Z", "y", "step", "length", "amplitude", "nugget", "noise_variance"}
    assert set(vars(fic)) == {"Z", "mean", "cov", "step", "length", "amplitude", "nugget", "noise_variance"}
    assert raw.step == fic.step == 0


@pytest.mark.parametrize("mode", ["coverage", "recent"])
def test_raw_processes_every_write_and_retains_order(mode):
    state = controls.initial_raw(1, 3)
    positions = [0., 10., 20., 10.1, 40.]
    expected = [0., 10.1, 40.] if mode == "coverage" else [20., 10.1, 40.]
    for i, value in enumerate(positions):
        state = controls.write_raw(state, np.array([[value, 0.]]), np.array([float(i)]), mode)
        assert state.step == i+1
    np.testing.assert_array_equal(state.Z[0, :, 0], expected)
    assert state.resident_bytes_per_context == 112


def test_coverage_exact_tie_drops_first_oldest_and_ignores_labels():
    first = controls.initial_raw(2, 2)
    for value in (0., 1., 2.):
        first = controls.write_raw(first, np.array([[value, 0.], [value, 0.]]),
                                   np.array([value, 1000-value]), "coverage")
    np.testing.assert_array_equal(first.Z[:, :, 0], [[1., 2.], [1., 2.]])
    np.testing.assert_array_equal(first.y, [[1., 2.], [999., 998.]])


def test_raw_repeated_coordinate_keeps_distinct_noisy_rows_until_capacity():
    state = controls.initial_raw(1, 3)
    for value in (1., 2.):
        state = controls.write_raw(state, np.array([[0., 0.]]), np.array([value]))
    assert state.step == 2
    np.testing.assert_array_equal(state.y[0, :2], [1., 2.])
    paths = np.zeros((1, 1, 4, 4, 2))
    output = controls.predict_raw(state, paths)
    expected = raw_dense(np.zeros((2, 2)), np.array([1., 2.]), np.zeros((4, 2)))
    for key, value in zip(("mean", "variance", "risk"), expected, strict=True):
        np.testing.assert_allclose(output[key], value, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("capacity", [2, 5, 41])
@pytest.mark.parametrize("mode", ["coverage", "recent"])
def test_raw_predictions_match_independent_retained_only_gp(capacity, mode):
    x, y, paths = fixture()
    state = controls.initial_raw(2, capacity)
    for t in range(5):
        state = controls.write_raw(state, x[:, t], y[:, t], mode)
    result = controls.predict_raw(state, paths)
    used = min(5, capacity)
    for b in range(2):
        for q in range(2):
            for a in range(4):
                expected = raw_dense(state.Z[b, :used], state.y[b, :used], paths[b, q, a])
                for key, value in zip(("mean", "variance", "risk"), expected, strict=True):
                    assert result[key][b, q, a] == pytest.approx(value, rel=1e-11, abs=1e-12)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_fic_sequential_equals_independent_heteroscedastic_batch(count):
    x, y, paths = fixture()
    state = controls.initial_fic(2)
    for t in range(count):
        state = controls.write_fic(state, x[:, t], y[:, t])
    assert state.step == count and state.resident_bytes_per_context == 904
    result = controls.predict_fic(state, paths)
    for b in range(2):
        anchors = state.Z[b]
        prior = covariance(anchors, anchors)
        observed_cross = covariance(x[b, :count], anchors)
        coefficients = np.linalg.solve(prior, observed_cross.T).T
        residual = 1.00001-np.sum(coefficients*observed_cross, axis=1)
        likelihood = np.diag(residual+.09)
        observed = coefficients @ prior @ coefficients.T+likelihood
        expected_mean = prior @ coefficients.T @ np.linalg.solve(observed, y[b, :count])
        expected_cov = prior-prior @ coefficients.T @ np.linalg.solve(observed, coefficients @ prior)
        np.testing.assert_allclose(state.mean[b], expected_mean, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(state.cov[b], expected_cov, rtol=1e-11, atol=1e-12)
        for q in range(2):
            for a in range(4):
                expected = fic_dense(anchors, x[b, :count], y[b, :count], paths[b, q, a])
                for key, value in zip(("mean", "variance", "risk"), expected, strict=True):
                    assert result[key][b, q, a] == pytest.approx(value, rel=1e-11, abs=1e-12)


def test_fic_repeated_events_have_independent_residuals_not_shared_latent_residual():
    state = controls.initial_fic(1)
    x, y = np.zeros((2, 2)), np.array([1., -.5])
    for t in range(2):
        state = controls.write_fic(state, x[t:t+1], y[t:t+1])
    paths = np.zeros((1, 1, 4, 4, 2))
    output = controls.predict_fic(state, paths)
    expected = fic_dense(state.Z[0], x, y, np.zeros((4, 2)))
    for key, value in zip(("mean", "variance", "risk"), expected, strict=True):
        np.testing.assert_allclose(output[key], value, rtol=1e-11, atol=1e-12)
    prior_state = controls.initial_fic(1)
    prior = controls.predict_fic(prior_state, paths)
    k = covariance(np.zeros((1, 2)), prior_state.Z[0])[0]
    lowrank = k @ np.linalg.solve(covariance(prior_state.Z[0], prior_state.Z[0]), k)
    assert prior["variance"][0, 0, 0] == pytest.approx(lowrank+(1.00001-lowrank)/4, abs=1e-12)
    assert prior["variance"][0, 0, 0] < 1.00001


@pytest.mark.parametrize("kind", ["raw", "fic"])
def test_state_ownership_pure_reads_and_batch_partition(kind):
    x, y, paths = fixture()
    initial = controls.initial_raw if kind == "raw" else controls.initial_fic
    write = controls.write_raw if kind == "raw" else controls.write_fic
    predict = controls.predict_raw if kind == "raw" else controls.predict_fic
    state = initial(2)
    for t in range(5):
        old_step = state.step
        old_coordinates = state.Z.copy()
        state = write(state, x[:, t], y[:, t])
        assert state.step == old_step+1
        assert not np.shares_memory(state.Z, x)
        if old_step == 0:
            assert old_coordinates.shape == state.Z.shape
    before = {name: value.tobytes() for name, value in vars(state).items() if isinstance(value, np.ndarray)}
    actual = predict(state, paths)
    actual["mean"][:] = 999
    repeated = predict(state, paths)
    assert before == {name: value.tobytes() for name, value in vars(state).items() if isinstance(value, np.ndarray)}
    for b in range(2):
        single = initial(1)
        for t in range(5):
            single = write(single, x[b:b+1, t], y[b:b+1, t])
        one = predict(single, paths[b:b+1])
        for key in repeated:
            np.testing.assert_allclose(repeated[key][b], one[key][0], rtol=1e-12, atol=1e-13)
    with pytest.raises(ValueError):
        state.Z.setflags(write=True)


def test_predictors_have_no_target_or_private_input():
    assert list(inspect.signature(controls.predict_raw).parameters) == ["state", "paths"]
    assert list(inspect.signature(controls.predict_fic).parameters) == ["state", "paths"]
    with pytest.raises(TypeError):
        controls.predict_fic(controls.initial_fic(1), np.zeros((1, 1, 4, 4, 2)), exposure=0)


@pytest.mark.parametrize("batch,capacity", [(0, 41), (True, 41), (1, 0), (1, 42), (1, True), (1, 2.5)])
def test_raw_initial_guards(batch, capacity):
    with pytest.raises(ValueError):
        controls.initial_raw(batch, capacity)


@pytest.mark.parametrize("kind", ["raw", "fic"])
def test_invalid_writes_do_not_mutate_state(kind):
    state = controls.initial_raw(1) if kind == "raw" else controls.initial_fic(1)
    write = controls.write_raw if kind == "raw" else controls.write_fic
    before = state.Z.tobytes()
    for x, y in [(np.zeros((1, 2), np.float32), np.ones(1)),
                 (np.zeros((1, 3)), np.ones(1)), (np.zeros((1, 2)), np.array([np.nan]))]:
        with pytest.raises(ValueError):
            write(state, x, y)
    assert state.Z.tobytes() == before and state.step == 0
    with pytest.raises(ValueError):
        replace(state, noise_variance=.1)
    with pytest.raises(ValueError):
        replace(state, step=-1)


def test_remaining_schema_and_empty_guards():
    with pytest.raises(ValueError):
        controls.predict_raw(controls.initial_raw(1), np.zeros((1, 1, 4, 4, 2)))
    with pytest.raises(ValueError):
        controls.write_raw(controls.initial_raw(1), np.zeros((1, 2)), np.zeros(1), "target")
    with pytest.raises(ValueError):
        controls.initial_fic(False)
    fic = controls.initial_fic(1)
    with pytest.raises(ValueError):
        controls.predict_fic(fic, np.zeros((1, 1, 4, 4, 2), np.float32))
    with pytest.raises(ValueError):
        replace(fic, Z=fic.Z+1)
    raw = controls.initial_raw(1, 2)
    with pytest.raises(ValueError):
        replace(raw, y=np.ones((1, 2)))
