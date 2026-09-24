import numpy as np
import pytest

from openjev.research.otto_action_latent_ridge import design, project_probabilities, solve


def data():
    rng = np.random.default_rng(9)
    return rng.normal(size=(3, 9, 31)).astype(np.float32), np.full(3, 9, np.int64), rng.integers(0, 4, (3, 8), dtype=np.int64)


def test_design_causal_and_normal_no_current_observation():
    p, length, actions = data()
    changed = actions.copy()
    changed[:, 4:] = (changed[:, 4:] + 1) % 4
    assert np.array_equal(design(p, length, actions)[:, :4], design(p, length, changed)[:, :4])
    observations = np.zeros((3, 8, 31), np.float32)
    found = np.zeros((3, 8), bool)
    before = design(p, length, actions, observations, found)
    observations[:, 3] = 100
    after = design(p, length, actions, observations, found)
    assert np.array_equal(before[:, :4], after[:, :4])
    assert not np.array_equal(before[:, 4], after[:, 4])


def test_ridge_normal_equation():
    p, length, actions = data()
    x = design(p, length, actions).reshape(-1, 336)
    y = np.arange(len(x) * 4).reshape(-1, 4) / 100
    w = np.full(len(x), 1 / len(x))
    beta = solve(x, y, w)
    np.testing.assert_allclose(x.T @ (w[:, None] * (x @ beta - y)) + .0001 * beta, 0., atol=1e-14)


def test_simplex_projection_and_smoothing():
    result = project_probabilities(np.array([[5., 0, 0, 0, 0], [-8., -8, -8, -8, -8]]))
    np.testing.assert_allclose(result.sum(-1), 1.)
    np.testing.assert_allclose(result[1], .2)
    assert (result > 0).all()
    with pytest.raises(ValueError):
        project_probabilities(np.array([[np.nan] * 5]))
