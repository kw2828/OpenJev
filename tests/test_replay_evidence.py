"""Small deterministic Bayesian witnesses, without empirical data or models."""
from __future__ import annotations

import inspect
import itertools

import numpy as np
import pytest

from openjev.research.replay_evidence import ReplayEvidence


def one_dimension():
    return ReplayEvidence(np.array([[1., -1.]]), np.array([[2.]]), noise_variance=4.)


def assert_snapshot_equal(left, right, *, counts=True):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_array_equal(left[key], right[key])
        elif key == 'evidence':
            assert left[key].keys() == right[key].keys()
            for evidence_id in left[key]:
                for field in ('x', 'y'):
                    np.testing.assert_array_equal(left[key][evidence_id][field], right[key][evidence_id][field])
        elif key != 'counts' or counts:
            assert left[key] == right[key]


def test_prior_and_one_dimensional_two_output_posterior_are_analytic():
    model = one_dimension()
    before = model.query(np.array([3.]))
    np.testing.assert_allclose(before['mean'], [3., -3.], atol=1e-14)
    np.testing.assert_allclose(before['latent_covariance'], 4.5*np.eye(2), atol=1e-14)
    np.testing.assert_allclose(before['predictive_covariance'], 8.5*np.eye(2), atol=1e-14)
    assert model.add('sensor-1', np.array([2.]), np.array([6., 2.])) == {
        'status': 'added', 'unique_evidence': 1, 'replay_events': 0, 'observation_steps': 1}
    state = model.snapshot()
    np.testing.assert_array_equal(state['precision'], [[3.]])
    np.testing.assert_array_equal(state['information'], [[5., -1.]])
    np.testing.assert_allclose(state['posterior_mean'], [[5/3, -1/3]], atol=1e-14)
    prediction = model.query(np.array([3.]))
    np.testing.assert_allclose(prediction['mean'], [5., -1.], atol=1e-14)
    np.testing.assert_allclose(prediction['latent_covariance'], 3*np.eye(2), atol=1e-14)
    np.testing.assert_allclose(prediction['predictive_covariance'], 7*np.eye(2), atol=1e-14)


def test_two_dimensional_correlated_prior_matches_hand_inverse_algebra():
    model = ReplayEvidence(np.array([[1.], [2.]]), np.array([[2., 1.], [1., 2.]]), noise_variance=1.)
    model.add('a', np.array([1., 0.]), np.array([3.]))
    model.add('b', np.array([1., 1.]), np.array([2.]))
    state = model.snapshot()
    # P=[[4,2],[2,3]], det(P)=8, P^-1=1/8*[[3,-2],[-2,4]].
    np.testing.assert_array_equal(state['precision'], [[4., 2.], [2., 3.]])
    np.testing.assert_array_equal(state['information'], [[9.], [7.]])
    np.testing.assert_allclose(state['posterior_mean'], [[13/8], [10/8]], atol=1e-14)
    prediction = model.query(np.array([2., -1.]))
    np.testing.assert_allclose(prediction['mean'], [2.], atol=1e-14)
    np.testing.assert_allclose(prediction['latent_covariance'], [[3.]], atol=1e-14)
    np.testing.assert_allclose(prediction['predictive_covariance'], [[4.]], atol=1e-14)


def test_exact_replays_do_not_tighten_covariance_or_advance_measurement_step():
    model = one_dimension(); x, y = np.array([2.]), np.array([6., 2.])
    model.add('a', x, y)
    before = model.snapshot(); prediction = model.query(np.array([3.]))
    for count in range(1, 6):
        result = model.add('a', x.copy(), y.copy())
        assert result == {'status': 'replay', 'unique_evidence': 1, 'replay_events': count, 'observation_steps': 1}
    assert_snapshot_equal(before, model.snapshot(), counts=False)
    for key, value in prediction.items():
        np.testing.assert_array_equal(value, model.query(np.array([3.]))[key])


def test_new_id_with_identical_payload_counts_independent_measurement():
    model = one_dimension(); x, y = np.array([2.]), np.array([6., 2.])
    model.add('physical-1', x, y); model.add('physical-2', x, y)
    state = model.snapshot()
    np.testing.assert_array_equal(state['precision'], [[4.]])
    np.testing.assert_array_equal(state['information'], [[8., 0.]])
    np.testing.assert_allclose(state['posterior_mean'], [[2., 0.]], atol=1e-14)
    assert state['counts'] == {'unique_evidence': 2, 'replay_events': 0, 'observation_steps': 2}
    np.testing.assert_allclose(model.query(np.array([3.]))['predictive_covariance'], 6.25*np.eye(2), atol=1e-14)


def test_unique_evidence_order_invariance_matches_batch_sum_without_replay_weight():
    rows = [('a', np.array([1., 2.]), np.array([3., -1.])),
            ('b', np.array([-1., 1.]), np.array([2., 4.])),
            ('c', np.array([.5, -.25]), np.array([-2., 1.]))]
    # Dyadic rows give exact sufficient-statistic sums in every order.
    precision = np.array([[3.25, .875], [.875, 6.0625]])
    information = np.array([[0., -4.5], [8.5, 1.75]])
    predictions = []
    for order in itertools.permutations(rows):
        model = ReplayEvidence(np.zeros((2, 2)), np.eye(2), noise_variance=1.)
        for evidence_id, x, y in order:
            model.add(evidence_id, x, y); model.add(evidence_id, x, y)
        state = model.snapshot()
        np.testing.assert_array_equal(state['precision'], precision)
        np.testing.assert_array_equal(state['information'], information)
        assert state['counts'] == {'unique_evidence': 3, 'replay_events': 3, 'observation_steps': 3}
        predictions.append(model.query(np.array([1., -1.])))
    for prediction in predictions[1:]:
        for key in prediction:
            np.testing.assert_allclose(prediction[key], predictions[0][key], atol=1e-14, rtol=1e-14)


def test_conflicting_payload_and_signed_zero_conflict_are_atomic():
    model = one_dimension(); model.add('a', np.array([0.]), np.array([6., 2.]))
    before = model.snapshot()
    for x, y in [(np.array([1.]), np.array([6., 2.])),
                 (np.array([0.]), np.array([6., 3.])),
                 (np.array([-0.]), np.array([6., 2.]))]:
        with pytest.raises(ValueError, match='contradictory'):
            model.add('a', x, y)
        assert_snapshot_equal(model.snapshot(), before)


def test_inputs_outputs_and_nested_dictionaries_have_independent_storage():
    mean, precision = np.zeros((2, 1)), np.eye(2)
    model = ReplayEvidence(mean, precision, noise_variance=1.)
    x, y = np.array([1., 2.]), np.array([3.])
    model.add('a', x, y)
    before = model.snapshot()
    mean.fill(100); precision.fill(100); x.fill(100); y.fill(100)
    snapshot = model.snapshot()
    snapshot['evidence'].clear(); snapshot['counts']['unique_evidence'] = 999
    for value in (snapshot['precision'], snapshot['information'], snapshot['posterior_mean']):
        assert value.flags.owndata and not value.flags.writeable
        value.setflags(write=True); value.fill(-100)
    assert_snapshot_equal(model.snapshot(), before)
    output = model.query(np.array([1., 1.]))
    for value in output.values():
        assert value.flags.owndata and not value.flags.writeable
        value.setflags(write=True); value.fill(999)
    assert_snapshot_equal(model.snapshot(), before)


def test_zero_query_has_zero_latent_uncertainty_and_only_observation_noise():
    model = one_dimension()
    result = model.query(np.array([0.]))
    np.testing.assert_array_equal(result['mean'], [0., 0.])
    np.testing.assert_array_equal(result['latent_covariance'], np.zeros((2, 2)))
    np.testing.assert_array_equal(result['predictive_covariance'], 4*np.eye(2))
    assert model.counts == {'unique_evidence': 0, 'replay_events': 0, 'observation_steps': 0}


@pytest.mark.parametrize('noise', [0., -1., float('nan'), float('inf'), True, '1'])
def test_invalid_known_noise_rejected(noise):
    with pytest.raises(ValueError):
        ReplayEvidence(np.zeros((2, 1)), np.eye(2), noise_variance=noise)


@pytest.mark.parametrize('precision', [np.eye(2, dtype=np.float32), np.eye(3), np.zeros((2, 2)),
                                      np.array([[1., 2.], [2., 1.]]),
                                      np.array([[1., 1e-15], [0., 1.]]),
                                      np.array([[1., 0.], [0., np.nan]])])
def test_precision_requires_finite_exact_symmetry_and_positive_definiteness(precision):
    with pytest.raises(ValueError):
        ReplayEvidence(np.zeros((2, 1)), precision, noise_variance=1.)


@pytest.mark.parametrize('mean', [np.zeros(2), np.zeros((0, 1)), np.zeros((1, 0)),
                                np.zeros((2, 1), np.float32), np.full((2, 1), np.inf)])
def test_prior_dimensions_dtype_and_finiteness_are_strict(mean):
    with pytest.raises(ValueError):
        ReplayEvidence(mean, np.eye(2), noise_variance=1.)


def test_bad_measurement_query_and_overflow_do_not_modify_state():
    model = one_dimension(); before = model.snapshot()
    for evidence_id, x, y in [('', np.ones(1), np.ones(2)), (1, np.ones(1), np.ones(2)),
                              ('a', np.ones(1, np.float32), np.ones(2)),
                              ('a', np.ones(2), np.ones(2)), ('a', np.ones(1), np.ones(1)),
                              ('a', np.array([np.inf]), np.ones(2)),
                              ('a', np.array([1e308]), np.ones(2))]:
        with pytest.raises(ValueError):
            model.add(evidence_id, x, y)
        assert_snapshot_equal(model.snapshot(), before)
    for x in (np.ones(2), np.ones(1, np.float32), np.array([np.nan]), np.array([1e308])):
        with pytest.raises(ValueError):
            model.query(x)
        assert_snapshot_equal(model.snapshot(), before)


def test_no_inverse_no_rng_and_no_hidden_query_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('inverse or random sampling is outside this exact reference')
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    model = one_dimension(); model.add('a', np.array([2.]), np.array([6., 2.]))
    model.query(np.array([3.]))
    assert tuple(inspect.signature(model.query).parameters) == ('x',)
    for key in ('target', 'hidden_state', 'true_id'):
        with pytest.raises(TypeError):
            model.query(np.array([3.]), **{key: np.zeros(2)})


def test_numerical_factorization_failure_is_atomic_without_retry(monkeypatch):
    model = one_dimension(); before = model.snapshot()
    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError('fabricated failure')
    monkeypatch.setattr(np.linalg, 'cholesky', failure)
    with pytest.raises(ValueError, match='positive-definite'):
        model.add('a', np.array([2.]), np.array([6., 2.]))
    assert_snapshot_equal(model.snapshot(), before)
