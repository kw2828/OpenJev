"""Fabricated scalar Gaussian and deterministic stream qualification."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from openjev.research.retention_data import (
    DEFER_COST,
    PATH_COST,
    exact_reference,
    generate,
    grid_points,
    kernel,
    path_library,
)


def scalar_kernel(left, right):
    distance = sum((float(a)-float(b))**2 for a, b in zip(left, right, strict=True))
    return math.exp(-distance/2)+(1e-5 if all(a == b for a, b in zip(left, right, strict=True)) else 0)


def dense_oracle(x, y, paths):
    # Direct conditional covariance of all requested latent points. Production
    # instead projects whitened cross-covariance before taking each variance.
    points = paths.reshape(-1, 2)
    observed = np.array([[scalar_kernel(a, b) for b in x] for a in x])+.09*np.eye(len(x))
    cross = np.array([[scalar_kernel(a, b) for b in points] for a in x])
    prior = np.array([[scalar_kernel(a, b) for b in points] for a in points])
    point_mean = cross.T @ np.linalg.solve(observed, y)
    conditional = prior-cross.T @ np.linalg.solve(observed, cross)
    values = {name: np.empty(paths.shape[:2]) for name in
              ('mean', 'variance', 'risk', 'diag_variance', 'diag_risk')}
    for query in range(paths.shape[0]):
        for action in range(4):
            start = 16*query+4*action
            block = conditional[start:start+4, start:start+4]
            mean = sum(float(v) for v in point_mean[start:start+4])/4
            variance = sum(float(v) for v in block.flat)/16
            diagonal = sum(float(block[i, i]) for i in range(4))/16
            values['mean'][query, action] = mean
            values['variance'][query, action] = variance
            values['diag_variance'][query, action] = diagonal
            values['risk'][query, action] = .5*math.erfc((.5-mean)/math.sqrt(2*variance))
            values['diag_risk'][query, action] = .5*math.erfc((.5-mean)/math.sqrt(2*diagonal))
    return values


def public_fixture():
    x = np.array([[[-1., .5], [0., -.5], [1., 0.]]])
    y = np.array([[.7, -.2, .4]])
    paths = np.array([[[-1., 0.], [-.5, 0.], [0., 0.], [.5, 0.]],
                      [[0., -1.], [0., -.5], [0., 0.], [0., .5]],
                      [[-.5, -.5], [0., 0.], [.5, .5], [1., 1.]],
                      [[-.5, 1.], [0., .5], [.5, 0.], [1., -.5]]])[None, None]
    return x, y, paths


def test_grid_and_complete_distinct_path_libraries():
    grid = grid_points()
    assert grid.shape == (289, 2) and grid.dtype == np.float64 and grid.flags.owndata
    assert len({tuple(row) for row in grid}) == 289
    np.testing.assert_array_equal(np.unique(grid), np.arange(-2., 2.01, .25))
    for geometry, count in (('axial', 374), ('diagonal', 242)):
        library = path_library(geometry)
        assert library.shape == (count, 4, 2) and library.flags.owndata
        assert len({tuple(sorted(map(tuple, path))) for path in library}) == count
        assert np.all(np.abs(library) <= 2) and np.all(library*4 == np.rint(library*4))
        changes = np.diff(library, axis=1)
        np.testing.assert_array_equal(changes, np.repeat(changes[:, :1], 3, axis=1))
        if geometry == 'axial':
            np.testing.assert_array_equal(np.abs(changes).sum(axis=-1), np.full((count, 3), .5))
            np.testing.assert_array_equal(np.count_nonzero(changes, axis=-1), np.ones((count, 3), int))
        else:
            np.testing.assert_array_equal(np.abs(changes), np.full((count, 3, 2), .5))


def test_kernel_nugget_is_shared_coordinate_not_event_noise():
    x = np.array([[0., 0.], [0., 0.], [1., 0.]])
    expected = np.array([[scalar_kernel(a, b) for b in x] for a in x])
    np.testing.assert_allclose(kernel(x, x), expected, atol=1e-15, rtol=1e-15)
    assert kernel(x, x)[0, 1] == 1.00001
    assert kernel(x, x)[0, 0] == 1.00001


def test_reference_matches_independent_full_conditional_covariance():
    x, y, paths = public_fixture()
    result = exact_reference(x, y, paths)
    expected = dense_oracle(x[0], y[0], paths[0])
    assert set(result) == set(expected)
    for key in result:
        np.testing.assert_allclose(result[key][0], expected[key], atol=2e-12, rtol=2e-12)
        assert result[key].dtype == np.float64 and result[key].flags.owndata


def test_shared_latent_repeated_path_does_not_receive_measurement_noise():
    x = np.zeros((1, 1, 2)); y = np.array([[.7]])
    paths = np.zeros((1, 1, 4, 4, 2))
    result = exact_reference(x, y, paths)
    latent = 1.00001
    expected_mean = latent*.7/(latent+.09)
    expected_variance = latent*.09/(latent+.09)
    np.testing.assert_allclose(result['mean'], expected_mean, atol=1e-14)
    np.testing.assert_allclose(result['variance'], expected_variance, atol=1e-14)
    np.testing.assert_allclose(result['diag_variance'], expected_variance/4, atol=1e-14)
    assert not np.isclose(result['variance'][0, 0, 0], expected_variance+.09/4)
    # A second independent noisy measurement at the same coordinate contributes
    # fresh evidence, while path occurrences remain the same latent variable.
    repeated = exact_reference(np.repeat(x, 2, axis=1), np.repeat(y, 2, axis=1), paths)
    np.testing.assert_allclose(repeated['variance'], 1/(1/latent+2/.09), atol=1e-14)


def test_equal_point_marginals_but_joint_covariance_changes_safe_decision():
    x = np.zeros((1, 1, 2)); y = np.zeros((1, 1))
    spread = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    repeated = np.repeat(spread[:1], 4, axis=0)
    paths = np.stack((spread, repeated, spread, repeated))[None, None]
    result = exact_reference(x, y, paths)
    # All points have the same distance from the observation and the same prior
    # variance; only their within-path correlations differ.
    np.testing.assert_array_equal(result['mean'], np.zeros((1, 1, 4)))
    np.testing.assert_allclose(result['diag_variance'], result['diag_variance'][0, 0, 0], atol=1e-15)
    assert PATH_COST+result['risk'][0, 0, 0] < DEFER_COST
    assert PATH_COST+result['risk'][0, 0, 1] > DEFER_COST
    assert PATH_COST+result['diag_risk'][0, 0, 1] < DEFER_COST


def test_generator_coherence_with_scripted_latent_innovation_and_independent_noise(monkeypatch):
    class Order:
        def permutation(self, count):
            assert count == 289
            return np.arange(count-1, -1, -1)
    class Field:
        def standard_normal(self, count):
            result = np.zeros(count); result[0] = 1.
            return result
    class Noise:
        def standard_normal(self, count):
            return np.full(count, 2.)
    class Request:
        def choice(self, count, *, size, replace):
            assert count == 374 and size == 4 and replace is False
            return np.arange(4)
    streams = iter((Order(), Field(), Noise(), Request()))
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: next(streams))
    data = generate(953101, 1, 5, queries=2)
    # Cholesky's first column is K[:,0]/sqrt(K[0,0]); no production kernel,
    # factorization or sampled full latent field is used to derive this oracle.
    latent = lambda point: scalar_kernel(point, (-2., -2.))/math.sqrt(1.00001)
    np.testing.assert_allclose(data['y'][0], [latent(point)+.6 for point in data['x'][0]], atol=1e-14)
    for query in range(2):
        expected = [sum(latent(point) for point in path)/4 for path in data['paths'][0, query]]
        np.testing.assert_allclose(data['exposure'][0, query], expected, atol=1e-14)
    np.testing.assert_array_equal(data['exposure'][0, 0], data['exposure'][0, 1])


def test_seed_extensions_geometry_pairing_no_global_rng_and_owned_outputs():
    before = np.random.get_state()
    first = generate(953101, 1, 5, queries=1)
    same = generate(953101, 1, 5, queries=1)
    extended = generate(953101, 2, 8, queries=3)
    diagonal = generate(953101, 1, 5, geometry='diagonal', queries=1)
    other = generate(953102, 1, 5, queries=1)
    assert set(first) == {'x', 'y', 'paths', 'exposure'}
    for key, value in first.items():
        assert value.dtype == np.float64 and value.flags.owndata and np.isfinite(value).all()
        np.testing.assert_array_equal(value, same[key])
        selection = extended[key][:1, :5] if key in ('x', 'y') else extended[key][:1, :1]
        np.testing.assert_array_equal(value, selection)
    for key in ('x', 'y'):
        np.testing.assert_array_equal(first[key], diagonal[key])
    assert not np.array_equal(first['y'], other['y'])
    assert len({tuple(point) for point in first['x'][0]}) == 5
    assert len({tuple(path.flat) for path in first['paths'][0, 0]}) == 4
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    first['paths'].fill(100)
    np.testing.assert_array_equal(path_library('axial'), path_library('axial'))
    assert (np.abs(grid_points()) <= 2).all()


def test_reference_chunk_permutation_causality_and_no_input_mutation():
    x, y, paths = public_fixture()
    x = np.repeat(x, 2, axis=0); y = np.concatenate((y, y+.2)); paths = np.repeat(paths, 2, axis=0)
    inputs = (x, y, paths)
    saved = [value.copy() for value in inputs]
    result = exact_reference(*inputs)
    single = exact_reference(x[:1], y[:1], paths[:1])
    reordered = exact_reference(x[:, ::-1], y[:, ::-1], paths[:, :, ::-1, ::-1])
    changed_paths = np.concatenate((paths, paths), axis=1)
    changed_paths[:, 1] += .25
    extra = exact_reference(x, y, changed_paths)
    for key in result:
        np.testing.assert_array_equal(result[key][:1], single[key])
        np.testing.assert_allclose(result[key][:, :, ::-1], reordered[key], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(result[key], extra[key][:, :1], atol=2e-12, rtol=2e-12)
    for value, copy in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(value, copy)
    assert tuple(inspect.signature(exact_reference).parameters) == ('x', 'y', 'paths')
    with pytest.raises(TypeError):
        exact_reference(x, y, paths, exposure=np.zeros((2, 1, 4)))


@pytest.mark.parametrize('overrides', [
    {'seed': True}, {'seed': -1}, {'seed': 2**32}, {'contexts': 0}, {'contexts': 1.},
    {'observations': 0}, {'observations': 290}, {'observations': True},
    {'queries': 0}, {'queries': False}, {'geometry': 'curved'}, {'geometry': None},
])
def test_generator_rejects_malformed_configuration(overrides):
    arguments = {'seed': 953101, 'contexts': 1, 'observations': 4}
    arguments.update(overrides)
    with pytest.raises(ValueError):
        generate(**arguments)


@pytest.mark.parametrize('mutation', ['float32', 'nonfinite', 'missing_context', 'wrong_y', 'wrong_actions', 'empty'])
def test_reference_rejects_malformed_inputs(mutation):
    x, y, paths = public_fixture()
    if mutation == 'float32':
        x = x.astype(np.float32)
    elif mutation == 'nonfinite':
        y[0, 0] = np.nan
    elif mutation == 'missing_context':
        paths = paths[0]
    elif mutation == 'wrong_y':
        y = y[:, :-1]
    elif mutation == 'wrong_actions':
        paths = paths[:, :, :3]
    else:
        x = x[:, :0]; y = y[:, :0]
    with pytest.raises(ValueError):
        exact_reference(x, y, paths)


def test_kernel_rejects_invalid_coordinate_schema_and_cholesky_failure_propagates(monkeypatch):
    with pytest.raises(ValueError):
        kernel(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        kernel(np.zeros((2, 2), np.float32), np.zeros((2, 2)))
    def fail(matrix):
        raise np.linalg.LinAlgError('fabricated factorization stop')
    monkeypatch.setattr(np.linalg, 'cholesky', fail)
    with pytest.raises(np.linalg.LinAlgError, match='fabricated'):
        exact_reference(*public_fixture())
