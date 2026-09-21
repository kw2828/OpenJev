"""Closed-form calibration checks, independent of any task data."""
import math

import numpy as np
import pytest

from openjev.research.dialogue_temperature import (
    BETA_BOUNDS,
    fit_temperature,
    objective_and_derivative,
    output_logs,
)


def test_binary_constant_score_has_closed_form_interior_optimum():
    logs = np.tile(np.log([0.9, 0.1]), (4, 1))
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    fit = fit_temperature(logs, labels)
    expected = math.log(3)/math.log(9)
    assert fit['beta'] == pytest.approx(expected, abs=1e-12)
    assert fit['normalized_nll_after'] == pytest.approx(-.75*math.log(.75)-.25*math.log(.25))
    assert fit['bisection_steps'] == 64 and fit['location'] == 'interior'


@pytest.mark.parametrize('target,bound,status', [(0, BETA_BOUNDS[1], 'upper_bound'), (1, BETA_BOUNDS[0], 'lower_bound')])
def test_endpoint_optima(target, bound, status):
    fit = fit_temperature(np.log([[.9, .1]]), np.array([target], dtype=np.int64))
    assert fit['beta'] == bound and fit['location'] == status and fit['bisection_steps'] == 0


def test_uniform_scores_choose_beta_one_but_normalize_raw_roundoff():
    logs = np.log(np.array([[.5, .5, 0.], [1/3, 1/3, 1/3]], dtype=np.float64), where=np.array([[1, 1, 0], [1, 1, 1]], bool), out=np.full((2, 3), -np.inf))
    logs[0, :2] += 1e-8
    fit = fit_temperature(logs, np.array([0, 2], dtype=np.int64))
    assert fit['beta'] == 1 and fit['location'] == 'unidentified_uniform_identity'
    adjusted = output_logs(logs, 1)
    np.testing.assert_allclose(np.exp(adjusted).sum(1), 1, atol=1e-15)
    assert adjusted[0, 0] == -math.log(2)
    assert fit['normalization_only_nll_drift'] == pytest.approx(0.5e-8)


def test_padding_underflow_and_ties_stay_valid():
    logs = np.array([[math.log(.5), math.log(.5), -1000, -np.inf], [-1000, math.log(.75), math.log(.25), -np.inf]])
    labels = np.array([2, 0], dtype=np.int64)
    fit = fit_temperature(logs, labels)
    assert fit['raw_nll_before'] == 1000 and math.isfinite(fit['normalized_nll_after'])
    transformed = output_logs(logs, fit['beta'])
    assert np.isneginf(transformed[:, -1]).all()
    np.testing.assert_array_equal(transformed.argmax(1), [0, 1])
    np.testing.assert_allclose(np.exp(transformed).sum(1), 1, atol=1e-12)


def test_derivative_matches_finite_difference_and_is_monotone():
    logs = np.log([[.8, .1, .1], [.1, .6, .3], [.2, .1, .7]])
    labels = np.array([0, 2, 2], dtype=np.int64)
    derivatives = []
    for beta in (.125, .5, 1., 2., 8.):
        _, derivative = objective_and_derivative(logs, labels, beta)
        step = 1e-5
        finite = (objective_and_derivative(logs, labels, beta+step)[0]-objective_and_derivative(logs, labels, beta-step)[0])/(2*step)
        assert derivative == pytest.approx(finite, abs=1e-9)
        derivatives.append(derivative)
    assert derivatives == sorted(derivatives)


@pytest.mark.parametrize('defect', ['nan', 'positive_inf', 'bad_mass', 'masked_target', 'float_target', 'empty'])
def test_invalid_inputs_rejected(defect):
    logs = np.log([[.9, .1]])
    labels = np.array([0], dtype=np.int64)
    if defect == 'nan': logs[0, 0] = np.nan
    elif defect == 'positive_inf': logs[0, 0] = np.inf
    elif defect == 'bad_mass': logs += 1
    elif defect == 'masked_target': logs = np.array([[-np.inf, math.log(.5), math.log(.5)]])
    elif defect == 'float_target': labels = labels.astype(float)
    else: logs, labels = logs[:0], labels[:0]
    with pytest.raises(ValueError): fit_temperature(logs, labels)


@pytest.mark.parametrize('beta', [0, -1, np.nan, np.inf, True])
def test_invalid_inverse_temperature_rejected(beta):
    with pytest.raises(ValueError): output_logs(np.log([[.5, .5]]), beta)
