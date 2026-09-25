"""Fabricated public-context qualification only; no measured files are loaded."""
import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openjev.research import causal_robot_ridge as ridge


@pytest.fixture(scope='module')
def example():
    rng = np.random.default_rng(96101)
    q = rng.normal(size=(5, 32, 6))
    u = rng.normal(size=(5, 32, 6))
    future = rng.normal(size=(5, 128, 6))
    target = rng.normal(size=(5, 128, 6))
    return q, u, future, target


@pytest.fixture(scope='module')
def fitted(example):
    return ridge.fit(*example, penalty=2.)


def direct_design(q, u, future, h):
    # Scalar-loop construction deliberately avoids the production slicing.
    rows = []
    for b in range(len(q)):
        row = [q[b, t, j] for t in range(16, 32) for j in range(6)]
        row += [u[b, t, j] for t in range(15, 31) for j in range(6)]
        row += [future[b, t, j] for t in range(h) for j in range(6)]
        rows.append([*row, 1.])
    return np.array(rows, dtype=np.float64)


@pytest.mark.parametrize('h', [1, 2, 64, 127, 128])
def test_features_alignment(example, h):
    q, u, future, _ = example
    actual = ridge.features(q, u, future, horizon=h)
    np.testing.assert_array_equal(actual, direct_design(q, u, future, h))
    assert actual.shape == (5, 193 + 6*h)
    assert not any(np.shares_memory(actual, value) for value in (q, u, future))


def test_all_horizon_coefficients_against_independent_dual_ridge(example, fitted):
    q, u, future, target = example
    # B=5 dual systems are algebraically independent of the production primal
    # 199..961-dimensional Cholesky systems and shared Gram indexing.
    for h in range(1, 129):
        phi = direct_design(q, u, future, h)
        expected = phi.T @ np.linalg.solve(phi @ phi.T + 2.*np.eye(5), target[:, h-1])
        np.testing.assert_allclose(fitted.coefficients[h-1], expected.T, rtol=1e-10, atol=1e-12)


def test_predictions_against_scalar_feature_oracle(example, fitted):
    q, u, future, _ = example
    actual = ridge.predict(fitted, q, u, future)
    for h in (1, 17, 64, 128):
        expected = direct_design(q, u, future, h) @ fitted.coefficients[h-1].T
        np.testing.assert_allclose(actual[:, h-1], expected, rtol=1e-13, atol=1e-13)
    assert actual.shape == (5, 128, 6) and actual.dtype == np.float64


def test_suffix_future_inputs_never_influence_early_forecasts(example, fitted):
    q, u, future, _ = example
    original = ridge.predict(fitted, q, u, future)
    changed = future.copy()
    changed[:, 7:] += 31.
    after = ridge.predict(fitted, q, u, changed)
    np.testing.assert_array_equal(after[:, :7], original[:, :7])
    assert not np.array_equal(after[:, 7:], original[:, 7:])


def test_unused_context_and_overlap_torque_are_not_features(example, fitted):
    q, u, future, _ = example
    first = ridge.predict(fitted, q, u, future)
    q2, u2 = q.copy(), u.copy()
    q2[:, :16] += 100
    u2[:, :15] -= 100
    u2[:, -1] += 100  # u[C-1] is supplied only through future_u[0].
    np.testing.assert_array_equal(first, ridge.predict(fitted, q2, u2, future))


def test_intercept_is_penalized_and_every_horizon_target_is_separate():
    q = np.zeros((3, 32, 6), np.float64)
    future = np.zeros((3, 128, 6), np.float64)
    target = np.broadcast_to(np.arange(1., 129.)[None, :, None], (3, 128, 6)).copy()
    model = ridge.fit(q, q, future, target, penalty=3.)
    prediction = ridge.predict(model, q, q, future)
    np.testing.assert_allclose(prediction, target / 2, rtol=1e-14, atol=1e-14)
    for h, coef in enumerate(model.coefficients, 1):
        np.testing.assert_array_equal(coef[:, :-1], 0.)
        np.testing.assert_allclose(coef[:, -1], h/2, rtol=1e-14, atol=1e-14)


def test_immutable_owned_bank_and_storage_metadata(example, fitted):
    metadata = fitted.metadata()
    assert sum(value.size for value in fitted.coefficients) == 445440
    assert metadata['coefficient_count'] == 445440
    assert metadata['coefficient_bytes'] == 3563520
    assert metadata['retained_numeric_bytes'] == 3563536
    assert metadata['required_history_bytes'] == 1536
    assert metadata['training_rows_retained'] is False
    with pytest.raises(FrozenInstanceError):
        fitted.penalty = 7.
    for array in fitted.coefficients:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    copied = [value.copy() for value in fitted.coefficients]
    model = ridge.CausalRobotRidge(tuple(copied), 2., 5)
    copied[0][:] = 9
    np.testing.assert_array_equal(model.coefficients[0], fitted.coefficients[0])
    q, u, future, _ = example
    prediction = ridge.predict(model, q, u, future)
    prediction[:] = 99
    assert not np.all(ridge.predict(model, q, u, future) == 99)


def test_does_not_mutate_inputs_or_global_rng(example, fitted):
    before = [value.copy() for value in example]
    random_state = np.random.get_state()
    ridge.predict(fitted, *example[:3])
    for value, expected in zip(example, before, strict=True):
        np.testing.assert_array_equal(value, expected)
    after = np.random.get_state()
    assert random_state[0] == after[0] and random_state[2:] == after[2:]
    np.testing.assert_array_equal(random_state[1], after[1])


def test_chunked_requests_agree(example, fitted):
    q, u, future, _ = example
    together = ridge.predict(fitted, q, u, future)
    separate = np.concatenate([ridge.predict(fitted, q[i:i+1], u[i:i+1], future[i:i+1]) for i in range(5)])
    np.testing.assert_allclose(together, separate, rtol=1e-12, atol=1e-12)


def test_predictor_has_no_target_or_hidden_id_input():
    assert tuple(inspect.signature(ridge.predict).parameters) == ('model', 'q_context', 'u_context', 'future_u')


@pytest.mark.parametrize('penalty', [0., -1., float('nan'), float('inf'), True, '1'])
def test_invalid_penalties_rejected(example, penalty):
    with pytest.raises(ValueError, match='penalty'):
        ridge.fit(*example, penalty=penalty)


@pytest.mark.parametrize('which,kind', [(0, 'dtype'), (1, 'shape'), (2, 'nan'), (3, 'shape'), (3, 'nan')])
def test_invalid_fit_arrays(example, which, kind):
    arrays = [value.copy() for value in example]
    if kind == 'dtype':
        arrays[which] = arrays[which].astype(np.float32)
    elif kind == 'shape':
        arrays[which] = arrays[which][:, :-1]
    else:
        arrays[which].flat[0] = np.nan
    with pytest.raises(ValueError):
        ridge.fit(*arrays)


@pytest.mark.parametrize('horizon', [0, 129, True, 1., '1'])
def test_invalid_horizon(example, horizon):
    with pytest.raises(ValueError, match='horizon'):
        ridge.features(*example[:3], horizon=horizon)


def test_empty_inputs_and_invalid_model(example):
    with pytest.raises(ValueError, match='nonempty'):
        ridge.fit(*(value[:0] for value in example))
    with pytest.raises(ValueError, match='model'):
        ridge.predict({}, *example[:3])


def test_solver_failure_is_not_repaired(example, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise np.linalg.LinAlgError('fabricated')
    monkeypatch.setattr(ridge, 'cho_factor', fail)
    with pytest.raises(ValueError, match='no jitter or penalty change'):
        ridge.fit(*example)
    assert calls == [1]


def test_overflow_is_explicit(example):
    q, u, future, target = (value.copy() for value in example)
    future.fill(1e300)
    with pytest.raises(ValueError, match='sufficient statistics'):
        ridge.fit(q, u, future, target)
