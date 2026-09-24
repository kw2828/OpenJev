"""Independent Gaussian algebra and small fabricated generator witnesses."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from openjev.research.query_feature_data import (
    exact_gp_mixture,
    generate_contexts,
    mixture_log_prob,
    reference_predict,
)


def covariance(points, length, amplitude, noise):
    # Independent scalar construction, avoiding the production kernel/helper.
    result = np.empty((len(points), len(points)), np.float64)
    for i, left in enumerate(points):
        for j, right in enumerate(points):
            distance = sum((float(a)-float(b))**2 for a, b in zip(left, right, strict=True))
            result[i, j] = amplitude**2*math.exp(-distance/(2*length**2))+(noise if i == j else 0)
    return result


def log_density(values, matrix):
    return -.5*(len(values)*math.log(2*math.pi)+math.log(np.linalg.det(matrix))
                + float(values @ np.linalg.solve(matrix, values)))


def direct_reference(bx, by, fx, fy, qx, *, length=1., amplitude=1., noise=.0225):
    means, variances, evidence = [], [], []
    for block in range(len(bx)):
        points = np.concatenate((bx[block], fx, qx[None]), axis=0)
        matrix = covariance(points, length, amplitude, noise)
        labels = np.concatenate((by[block], fy))
        # Direct conditioning on all evidence at once; routing comes from a
        # difference of joint and archive log densities, not staged updates.
        training = matrix[:-1, :-1]
        means.append(matrix[-1, :-1] @ np.linalg.solve(training, labels))
        variances.append(matrix[-1, -1]-matrix[-1, :-1] @ np.linalg.solve(training, matrix[:-1, -1]))
        evidence.append(log_density(labels, training)-log_density(by[block], matrix[:len(by[block]), :len(by[block])]))
    normalizer = max(evidence)+math.log(sum(math.exp(v-max(evidence)) for v in evidence))
    logs = np.array(evidence)-normalizer
    probability = sum(math.exp(weight)*.5*(1+math.erf(mean/math.sqrt(2*variance)))
                      for weight, mean, variance in zip(logs, means, variances, strict=True))
    return np.array(means), np.array(variances), logs, probability


def fixture():
    bx = np.array([[[[0., 0.], [.3, -.2]], [[-.7, .4], [.2, .9]]]])
    by = np.array([[[.8, -.1], [-.3, .6]]])
    fx = np.array([[[.1, .2], [.4, .5]]])
    fy = np.array([[.2, -.4]])
    qx = np.array([[-.2, .1]])
    return bx, by, fx, fy, qx


def test_full_gaussian_reference_matches_direct_conditioning_and_likelihood_ratio():
    inputs = fixture()
    result = exact_gp_mixture(*inputs, length=.8, amplitude=1.3, noise_variance=.07)
    expected = direct_reference(*(value[0] for value in inputs), length=.8, amplitude=1.3, noise=.07)
    for name, value in zip(('component_mean', 'component_variance', 'log_weights', 'prob_positive'), expected, strict=True):
        np.testing.assert_allclose(result[name][0], value, atol=2e-12, rtol=2e-12)
    target = np.array([.75])
    density = sum(math.exp(weight)/math.sqrt(2*math.pi*variance)*math.exp(-(.75-mean)**2/(2*variance))
                  for weight, mean, variance in zip(expected[2], expected[0], expected[1], strict=True))
    np.testing.assert_allclose(mixture_log_prob(result, target), [math.log(density)], atol=2e-12)


def test_joint_fewshot_correlation_has_exact_rational_routing_witness():
    bx = np.zeros((1, 2, 1, 2)); by = np.array([[[.5], [-.5]]])
    fx = np.zeros((1, 2, 2)); fy = np.array([[.2, .2]]); qx = np.zeros((1, 2))
    result = exact_gp_mixture(bx, by, fx, fy, qx, noise_variance=.5)
    # Conditional F covariance [[5/6,1/3],[1/3,5/6]]. Independence would give
    # log odds 8/25, while the actual joint likelihood gives 8/35.
    np.testing.assert_allclose(result['log_weights'][0, 0]-result['log_weights'][0, 1], 8/35, atol=1e-12)
    assert not math.isclose(result['log_weights'][0, 0]-result['log_weights'][0, 1], 8/25, abs_tol=1e-12)
    np.testing.assert_allclose(result['component_mean'], [[9/35, -1/35]], atol=1e-12)
    np.testing.assert_allclose(result['component_variance'], [[9/14, 9/14]], atol=1e-12)


def test_public_block_permutation_changes_components_not_mixture():
    bx, by, fx, fy, qx = fixture()
    original = exact_gp_mixture(bx, by, fx, fy, qx)
    permuted = exact_gp_mixture(bx[:, ::-1], by[:, ::-1], fx, fy, qx)
    for key in ('component_mean', 'component_variance', 'log_weights'):
        np.testing.assert_allclose(permuted[key], original[key][:, ::-1], atol=1e-13)
    np.testing.assert_allclose(permuted['prob_positive'], original['prob_positive'], atol=1e-13)
    np.testing.assert_allclose(mixture_log_prob(permuted, np.array([.2])), mixture_log_prob(original, np.array([.2])), atol=1e-13)


def test_single_block_zero_labels_and_noisy_query_variance():
    result = exact_gp_mixture(np.zeros((1, 1, 1, 2)), np.zeros((1, 1, 1)),
                             np.zeros((1, 1, 2)), np.zeros((1, 1)), np.zeros((1, 2)), noise_variance=.5)
    # Two observations at one location: latent variance=1/(1+2/.5)=1/5.
    np.testing.assert_allclose(result['component_mean'], [[0.]], atol=1e-14)
    np.testing.assert_allclose(result['component_variance'], [[.7]], atol=1e-14)
    np.testing.assert_array_equal(result['log_weights'], [[0.]])
    np.testing.assert_array_equal(result['prob_positive'], [.5])


def test_saturated_positive_cdf_allows_only_declared_weight_roundoff_without_clipping():
    result = exact_gp_mixture(np.zeros((1, 3, 1, 2)), np.full((1, 3, 1), 100.),
                             np.zeros((1, 2, 2)), np.full((1, 2), 100.), np.zeros((1, 2)),
                             noise_variance=.5)
    # All component CDFs are one; preserve the actual rounded weight sum.
    np.testing.assert_array_equal(result['prob_positive'], np.exp(result['log_weights']).sum(axis=-1))
    assert abs(result['prob_positive'][0]-1.) <= 1e-12
    rounded = {key: value.copy() for key, value in result.items()}
    rounded['log_weights'] += 1e-14
    rounded['prob_positive'][0] = np.nextafter(1., np.inf)
    retained = rounded['prob_positive'].copy()
    assert np.isfinite(mixture_log_prob(rounded, np.array([100.]))).all()
    np.testing.assert_array_equal(rounded['prob_positive'], retained)
    rounded['prob_positive'][0] = 1.+2e-12
    with pytest.raises(ValueError, match='roundoff'):
        mixture_log_prob(rounded, np.array([100.]))


def test_generator_shapes_coherent_selection_and_reproducibility():
    data = generate_contexts(7, 2, blocks=3, basis_points=3, queries=2, fewshots=2)
    shapes = {'bx': (2, 3, 3, 2), 'by': (2, 3, 3), 'fx': (2, 2, 2, 2), 'fy': (2, 2, 2),
              'qx': (2, 2, 2), 'target': (2, 2), 'private_selected_block': (2, 2)}
    assert set(data) == set(shapes)
    again = generate_contexts(7, 2, blocks=3, basis_points=3, queries=2, fewshots=2)
    extended = generate_contexts(7, 3, blocks=3, basis_points=3, queries=2, fewshots=2)
    other = generate_contexts(8, 2, blocks=3, basis_points=3, queries=2, fewshots=2)
    for key, value in data.items():
        assert value.shape == shapes[key] and value.flags.owndata and np.isfinite(value).all()
        assert value.dtype == (np.int64 if key == 'private_selected_block' else np.float64)
        np.testing.assert_array_equal(value, again[key])
        np.testing.assert_array_equal(value, extended[key][:2])
    for key in ('bx', 'fx', 'qx'):
        assert (abs(data[key]) <= 2).all()
    assert ((data['private_selected_block'] >= 0) & (data['private_selected_block'] < 3)).all()
    assert not np.array_equal(data['by'], other['by'])


def test_generator_joint_sampling_uses_noiseless_cross_covariance_and_independent_noise(monkeypatch):
    class Scripted:
        def uniform(self, low, high, shape):
            return np.zeros(shape)
        def integers(self, low, high, size, dtype):
            return np.zeros(size, dtype=dtype)
        def standard_normal(self, size):
            return np.arange(1, size+1, dtype=np.float64)
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: Scripted())
    data = generate_contexts(19, 1, blocks=1, basis_points=1, queries=2, fewshots=1, noise_variance=.5)
    # One archive point and two (few-shot,query) pairs at one common location.
    # Each has a separate noise draw: covariance is all ones plus .5*I.
    expected = np.linalg.cholesky(np.ones((5, 5))+.5*np.eye(5)) @ np.arange(1., 6.)
    np.testing.assert_allclose(data['by'][0, 0], expected[:1], atol=1e-14)
    np.testing.assert_allclose(data['fy'][0, :, 0], expected[[1, 3]], atol=1e-14)
    np.testing.assert_allclose(data['target'][0], expected[[2, 4]], atol=1e-14)


def test_scoring_wrapper_does_not_read_private_identity_or_other_query_answers():
    data = generate_contexts(23, 2, blocks=2, basis_points=3, queries=2, fewshots=2)
    original = reference_predict(data)
    poisoned = {key: value.copy() for key, value in data.items()}
    poisoned['private_selected_block'] = object()
    result = reference_predict(poisoned)
    for key in result:
        np.testing.assert_array_equal(result[key], original[key])
    poisoned['fy'][:, 1] += 20.
    poisoned['target'][:, 1] += 100.
    changed = reference_predict(poisoned)
    for key in result:
        np.testing.assert_array_equal(changed[key][:, 0], original[key][:, 0])
    assert not np.allclose(changed['component_mean'][:, 1], original['component_mean'][:, 1])


def test_batch_chunking_no_mutation_no_global_rng_or_hidden_prediction_api():
    rng_before = np.random.get_state()
    data = generate_contexts(29, 2, blocks=2, basis_points=2, queries=1, fewshots=2)
    inputs = [data['bx'], data['by'], data['fx'][:, 0], data['fy'][:, 0], data['qx'][:, 0]]
    saved = [value.copy() for value in inputs]
    all_rows = exact_gp_mixture(*inputs)
    for index in range(2):
        single = exact_gp_mixture(*(value[index:index+1] for value in inputs))
        for key in single:
            np.testing.assert_array_equal(single[key][0], all_rows[key][index])
    for value, before in zip(inputs, saved, strict=True):
        np.testing.assert_array_equal(value, before)
        assert all(not np.shares_memory(value, output) for output in all_rows.values())
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert tuple(inspect.signature(exact_gp_mixture).parameters)[:5] == ('bx', 'by', 'fx', 'fy', 'qx')
    for key in ('target', 'private_selected_block', 'hidden_function'):
        with pytest.raises(TypeError):
            exact_gp_mixture(*inputs, **{key: np.zeros(1)})


@pytest.mark.parametrize('keyword,value', [('seed', True), ('seed', -1), ('seed', 2**32),
                                        ('contexts', 0), ('blocks', 1.5), ('basis_points', False),
                                        ('queries', 0), ('fewshots', -1), ('extent', np.inf),
                                        ('length', 0.), ('amplitude', -1.), ('noise_variance', 0.)])
def test_generator_rejects_invalid_shape_seed_and_law(keyword, value):
    arguments = {'seed': 1, 'contexts': 1, keyword: value}
    with pytest.raises(ValueError):
        generate_contexts(**arguments)


def test_reference_rejects_malformed_nonfinite_inputs_and_law_without_inverse(monkeypatch):
    original = list(fixture())
    for index, bad in [(0, original[0].astype(np.float32)), (1, np.zeros((1, 1, 2))),
                       (2, np.full((1, 2, 2), np.nan)), (3, np.zeros((1, 0))),
                       (4, np.zeros((1, 3)))]:
        changed = original.copy(); changed[index] = bad
        with pytest.raises(ValueError):
            exact_gp_mixture(*changed)
    for key in ('length', 'amplitude', 'noise_variance'):
        with pytest.raises(ValueError):
            exact_gp_mixture(*original, **{key: 0.})
    monkeypatch.setattr(np.linalg, 'inv', lambda *args: pytest.fail('no matrix inverse'))
    prediction = exact_gp_mixture(*original)
    with pytest.raises(ValueError):
        mixture_log_prob(prediction, np.array([np.nan]))
    with pytest.raises(ValueError):
        mixture_log_prob({**prediction, 'component_variance': -prediction['component_variance']}, np.zeros(1))


def test_cholesky_failure_propagates_without_jitter_or_retry(monkeypatch):
    calls = []
    def fail(matrix):
        calls.append(matrix.copy())
        raise np.linalg.LinAlgError('fabricated stop')
    monkeypatch.setattr(np.linalg, 'cholesky', fail)
    with pytest.raises(ValueError, match='no jitter'):
        exact_gp_mixture(*fixture())
    assert len(calls) == 1
