# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent hand fixtures only: no saved weights, measurements or producers."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/fsm_shift_audit_math.py'
MODULE = importlib.util.spec_from_file_location('shift_audit_math_test', PATH)
audit = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(audit)


def spec(kind='varx', *, order=1, residual_kind='affine', mode='feedback'):
    result = {'instance_id': 'fabricated', 'family': 'fabricated', 'seed': None, 'kind': kind,
              'order': order, 'alpha': None, 'residual_kind': None, 'mode': None,
              'hidden_width': None, 'context_iterations': None}
    if kind == 'varx':
        result['alpha'] = .001
    if kind == 'residual':
        result.update(seed=9201, residual_kind=residual_kind, mode=mode,
                      hidden_width=2 if residual_kind == 'tanh' else None,
                      instance_id='fabricated-9201')
    if kind == 'nllfr':
        result['context_iterations'] = 16
    return result


def norm():
    return {k: np.ones(3) if 'scale' in k else np.zeros(3) for k in audit.NORMALIZER_KEYS}


def lag_arrays(s):
    p = s['order']
    arrays = {'coefficients': np.zeros((3, 6*p+4))}
    arrays['coefficients'][:, 3*(p-1):3*p] = np.eye(3)*.5
    if s['kind'] == 'residual':
        if s['residual_kind'] == 'affine':
            arrays.update({'residual.weight': np.zeros((3, 6*p+3)), 'residual.bias': np.zeros(3)})
        else:
            h = s['hidden_width']
            arrays.update({'residual.0.weight': np.ones((h, 6*p+3))*.01,
                           'residual.0.bias': np.zeros(h), 'residual.2.weight': np.zeros((3, h)),
                           'residual.2.bias': np.zeros(3)})
    return arrays


def inputs(batch=1, context=100, horizon=128):
    return np.zeros((batch, context, 3)), np.zeros((batch, context-1, 3)), np.zeros((batch, horizon, 3))


@pytest.mark.parametrize('kind', ['affine', 'tanh'])
def test_constant_residual_closed_form_distinguishes_propagation_for_h128(kind):
    y, u, future = inputs()
    results = {}
    for mode in ('output_only', 'feedback'):
        s = spec('residual', residual_kind=kind, mode=mode)
        arrays = lag_arrays(s)
        arrays['residual.bias' if kind == 'affine' else 'residual.2.bias'].fill(1)
        results[mode] = audit.replay(s, arrays, norm(), y, u, future)
    np.testing.assert_array_equal(results['output_only']['prediction'], np.ones((1, 128, 3)))
    expected = np.broadcast_to(2*(1-.5**np.arange(1, 129))[None, :, None], (1, 128, 3))
    np.testing.assert_array_equal(results['feedback']['prediction'], expected)
    np.testing.assert_array_equal(results['output_only']['final_state'][:, :3], np.zeros((1, 3)))
    np.testing.assert_array_equal(results['feedback']['final_state'][:, :3], expected[:, -1])
    np.testing.assert_array_equal(results['feedback']['forecast_state'], np.zeros((1, 6)))


def test_chronological_lags_current_input_and_physical_units_hand_sentinel():
    s, n = spec(order=2), norm()
    n.update(u_mean=np.array([10., 20., 30.]), u_scale=np.array([2., 3., 4.]),
             y_mean=np.array([-5., 2., 7.]), y_scale=np.array([.5, 2., 3.]))
    a = lag_arrays(s)
    a['coefficients'].fill(0)
    a['coefficients'][0, [0, 4, 8, 9, 13, 15]] = [1, 2, 3, 4, 5, 6]
    a['coefficients'][1, 7] = 1
    a['coefficients'][2, 5] = 1
    y = np.array([[[999., 999., 999.], [1., 2., 3.], [4., 5., 6.]]])
    u = np.array([[[7., 8., 9.], [10., 11., 12.]]])
    future = np.array([[[13., 14., 15.]]])
    result = audit.replay(s, a, n, y*n['y_scale']+n['y_mean'],
                          u*n['u_scale']+n['u_mean'], future*n['u_scale']+n['u_mean'])
    np.testing.assert_array_equal(result['prediction'], np.array([[[145., 14., 6.]]])*n['y_scale']+n['y_mean'])
    np.testing.assert_array_equal(result['forecast_state'], [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]])
    np.testing.assert_array_equal(result['final_state'], [[4., 5., 6., 145., 14., 6., 10., 11., 12., 13., 14., 15.]])


def test_nonzero_tanh_head_has_hand_derived_recurrent_response():
    s = spec('residual', residual_kind='tanh')
    a = lag_arrays(s)
    a['residual.0.weight'].fill(0)
    a['residual.0.weight'][0, 0] = 1
    a['residual.2.weight'][0, 0] = 2
    a['residual.2.bias'][0] = .1
    actual = audit.replay(s, a, norm(), *inputs(context=2, horizon=3))
    first = .1
    second = .5*first+2*np.tanh(first)+.1
    third = .5*second+2*np.tanh(second)+.1
    np.testing.assert_allclose(actual['prediction'], [[[first, 0, 0], [second, 0, 0], [third, 0, 0]]], atol=1e-15, rtol=0)


@pytest.mark.parametrize('p', [32, 64, 96])
@pytest.mark.parametrize('kind', ['affine', 'tanh'])
@pytest.mark.parametrize('mode', ['output_only', 'feedback'])
def test_zero_head_matches_closed_form_and_varx_at_declared_orders(p, kind, mode):
    y, u, future = inputs()
    y[:, -1] = [1., -2., 3.]
    s = spec('residual', order=p, residual_kind=kind, mode=mode)
    arrays = lag_arrays(s)
    result = audit.replay(s, arrays, norm(), y, u, future)
    expected = y[:, -1:, :]*.5**np.arange(1, 129)[None, :, None]
    np.testing.assert_array_equal(result['prediction'], expected)
    linear = audit.replay(spec(order=p), {'coefficients': arrays['coefficients']}, norm(), y, u, future)
    np.testing.assert_array_equal(result['prediction'], linear['prediction'])
    np.testing.assert_array_equal(result['final_state'], linear['final_state'])


@pytest.mark.parametrize('kind', ['varx', 'residual'])
def test_causal_prefix_batch_ownership_and_empty_horizon(kind):
    s = spec(kind)
    a = lag_arrays(s)
    a['coefficients'][:, 3:6] = np.eye(3)
    y, u, future = inputs(batch=2, context=3, horizon=5)
    y[:, -1] = [[1., 2., 3.], [-1., -2., -3.]]
    future[:] = np.arange(30).reshape(2, 5, 3)*.01
    before = copy.deepcopy((s, a, y, u, future))
    result = audit.replay(s, a, norm(), y, u, future)
    later = future.copy(); later[:, 2:] += 100
    changed_y = y.copy(); changed_y[:, 0] = 10000
    changed = audit.replay(s, a, norm(), changed_y, u, later)
    np.testing.assert_array_equal(changed['prediction'][:, :2], result['prediction'][:, :2])
    np.testing.assert_array_equal(changed['forecast_state'], result['forecast_state'])
    for b in range(2):
        solo = audit.replay(s, a, norm(), y[b:b+1], u[b:b+1], future[b:b+1])
        np.testing.assert_allclose(solo['prediction'], result['prediction'][b:b+1], atol=1e-14, rtol=1e-14)
    empty = audit.replay(s, a, norm(), y, u, future[:, :0])
    assert empty['prediction'].shape == (2, 0, 3)
    np.testing.assert_array_equal(empty['forecast_state'], empty['final_state'])
    assert not np.shares_memory(empty['forecast_state'], empty['final_state'])
    assert s == before[0]
    for k in a:
        np.testing.assert_array_equal(a[k], before[1][k])
    for actual, original in zip((y, u, future), before[2:], strict=True):
        np.testing.assert_array_equal(actual, original)
        assert not any(np.shares_memory(actual, result[k]) for k in ('prediction', 'forecast_state', 'final_state'))
    json.dumps(result['diagnostics'], allow_nan=False)


def author_arrays(n=28, nonlinear=False):
    a = {'A': np.eye(n)*.5, 'B_u': np.zeros((n, 3)), 'C_y': np.zeros((3, n)),
         'D_yu': np.diag([1., 3., 4.]), 'u_mean': np.array([2., -3., .5]),
         'u_std': np.array([.5, 2., 3.]), 'y_mean': np.array([-4., 1., 7.]),
         'y_std': np.array([3., .25, 2.]), 'ts': np.array(1/6400.)}
    a['B_u'][0, 0] = 2
    a['C_y'][:, 0] = [1., 2., -1.]
    if nonlinear:
        a.update(B_w=np.zeros((n, 1)), C_z=np.zeros((1, n)), D_yw=np.zeros((3, 1)),
                 D_zu=np.zeros((1, 3)), W0=np.ones((1, 1)), b0=np.zeros(1),
                 W1=np.ones((1, 1)), b1=np.zeros(1), W2=np.ones((1, 1)), b2=np.zeros(1))
    return a


def test_bla_forecast_state_is_before_first_future_input_not_final_state():
    a = author_arrays()
    y, u, future = inputs(batch=2)
    # x1=2 and every observed input's first coordinate=.5 keep x=2.
    u[:, :, 0] = .5
    y[:, 1:] = [2.5, 4., -2.]
    y[:, 0] = 1e6  # Its unknown input is deliberately unavailable.
    future[:, 0] = [4., 2., 1.]
    args = (y*a['y_std']+a['y_mean'], u*a['u_std']+a['u_mean'], future*a['u_std']+a['u_mean'])
    original = copy.deepcopy((a, args))
    actual = audit.replay(spec('bla', order=28), a, norm(), *args)
    expected_start = np.zeros((2, 28)); expected_start[:, 0] = 2
    np.testing.assert_allclose(actual['forecast_state'], expected_start, atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual['prediction'][:, :1], np.broadcast_to(np.array([[[6., 10., 2.]]])*a['y_std']+a['y_mean'], (2, 1, 3)), atol=1e-13, rtol=0)
    # First update produces x=9; 127 subsequent zero inputs halve it.
    np.testing.assert_allclose(actual['final_state'][:, 0], 9*.5**127, atol=1e-45, rtol=1e-13)
    assert actual['diagnostics']['rank'] == 1
    assert len(actual['diagnostics']['singular_values']) == 28
    json.dumps(actual['diagnostics'], allow_nan=False)
    for k in a:
        np.testing.assert_array_equal(a[k], original[0][k])
    for value, saved in zip(args, original[1], strict=True):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize('iterations', [16, 64])
def test_author_nl_batch_forecast_time_state_and_json_diagnostics(iterations):
    a = author_arrays(n=1, nonlinear=True)
    s = spec('nllfr'); s['context_iterations'] = iterations
    # x1=2, first paired input3 -> x2=7; next input0 -> x3=3.5.
    y = np.array([[[999., -777., 555.], [5., 16., 18.], [7., 17., 1.]]])
    u = np.array([[[3., 4., 5.], [0., 1., 2.]]])
    future = np.array([[[4., 2., 1.], [0., 1., 0.]]])
    args = tuple(np.repeat(v*scale+mean, 2, axis=0) for v, scale, mean in
                 ((y, a['y_std'], a['y_mean']), (u, a['u_std'], a['u_mean']),
                  (future, a['u_std'], a['u_mean'])))
    before = copy.deepcopy((a, args))
    result = audit.replay(s, a, norm(), *args)
    np.testing.assert_allclose(result['forecast_state'], [[3.5], [3.5]], atol=1e-13, rtol=0)
    np.testing.assert_allclose(result['final_state'], [[4.875], [4.875]], atol=1e-13, rtol=0)
    expected = np.array([[[7.5, 13., .5], [9.75, 22.5, -9.75]]])*a['y_std']+a['y_mean']
    np.testing.assert_allclose(result['prediction'], np.repeat(expected, 2, axis=0), atol=1e-12, rtol=0)
    assert len(result['diagnostics']['per_request']) == 2
    for row in result['diagnostics']['per_request']:
        assert row['policy']['iterations'] == iterations
        assert row['requests'][0]['status'] == 'GRADIENT_TOL'
    json.dumps(result['diagnostics'], allow_nan=False)
    for k in a:
        np.testing.assert_array_equal(a[k], before[0][k])
    for value, saved in zip(args, before[1], strict=True):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize('mutation', ['spec_key', 'irrelevant', 'seed', 'seed_range', 'identity', 'order', 'alpha', 'mode',
                                     'model_key', 'coefficient_shape', 'coefficient_nan', 'normalizer_key',
                                     'scale_zero', 'scale_nan', 'dtype', 'context', 'u_shape', 'future_shape', 'input_inf'])
def test_structural_and_finite_contracts_reject_before_forecast(mutation):
    s = spec(); a = lag_arrays(s); n = norm(); y, u, future = inputs(context=3, horizon=2)
    if mutation == 'spec_key': s['extra'] = 1
    elif mutation == 'irrelevant': s['context_iterations'] = 16
    elif mutation == 'seed': s['seed'] = True
    elif mutation == 'seed_range': s['seed'] = 2**63; s['instance_id'] = 'fabricated-'+str(2**63)
    elif mutation == 'identity': s['instance_id'] = 'other'
    elif mutation == 'order': s['order'] = True
    elif mutation == 'alpha': s['alpha'] = 0
    elif mutation == 'mode': s['mode'] = 'feedback'
    elif mutation == 'model_key': a['extra'] = np.zeros(1)
    elif mutation == 'coefficient_shape': a['coefficients'] = np.zeros((3, 9))
    elif mutation == 'coefficient_nan': a['coefficients'][0, 0] = np.nan
    elif mutation == 'normalizer_key': n['extra'] = np.zeros(3)
    elif mutation == 'scale_zero': n['u_scale'][0] = 0
    elif mutation == 'scale_nan': n['y_scale'][0] = np.nan
    elif mutation == 'dtype': y = y.astype(np.float32)
    elif mutation == 'context': y = y[:, :1]; u = u[:, :0]
    elif mutation == 'u_shape': u = np.zeros((1, 3, 3))
    elif mutation == 'future_shape': future = np.zeros((2, 2, 3))
    elif mutation == 'input_inf': future[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        audit.replay(s, a, n, y, u, future)


def test_residual_requires_a_seeded_slot_even_with_consistent_unseeded_identity():
    s = spec('residual'); a = lag_arrays(s)
    s.update(seed=None, instance_id=s['family'])
    with pytest.raises(ValueError, match='residual kind/mode'):
        audit.replay(s, a, norm(), *inputs())


@pytest.mark.parametrize('stage', ['normalization', 'recurrence', 'denormalization'])
def test_finite_inputs_can_overflow_and_are_never_repaired(stage):
    s = spec(); a = lag_arrays(s); n = norm(); y, u, future = inputs(context=2, horizon=3)
    if stage == 'normalization':
        y[:, -1] = 1e308; n['y_scale'].fill(.01)
    elif stage == 'recurrence':
        y[:, -1] = 1e308; a['coefficients'][:, :3] = np.eye(3)*2
    else:
        n['y_scale'].fill(1e308); a['coefficients'][:, -1] = 2
    with pytest.raises(FloatingPointError):
        audit.replay(s, a, n, y, u, future)
