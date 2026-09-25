"""Fabricated deployment qualification; no files containing data or weights read."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from openjev_fsm_author import benchmark, linear_context, nllfr_context, nllfr_context_budget

from openjev.research import fsm_linear
from openjev.research import fsm_shift_inference as shift
from openjev.research.fsm_residual import FSMResidual


def normalizer():
    return {'u_mean': np.array([2., -3., 5.]), 'u_scale': np.array([.5, 2., 3.]),
            'y_mean': np.array([-7., 11., 4.]), 'y_scale': np.array([3., .25, 2.])}


def inputs(batch=2):
    rng = np.random.default_rng(31415)
    norm = normalizer()
    values = [rng.normal(scale=.15, size=(batch, length, 3)) for length in (100, 99, 128)]
    return (values[0]*norm['y_scale']+norm['y_mean'],
            values[1]*norm['u_scale']+norm['u_mean'], values[2]*norm['u_scale']+norm['u_mean'])


def specification(kind, *, order=3, head=None, mode=None, iterations=None, seed=9201, width=2):
    seed = seed if kind == 'residual' else None
    family = kind if kind != 'residual' else f'{head}_{mode}'
    if kind == 'nllfr':
        family += str(iterations)
    return {'instance_id': family+(f'-{seed}' if seed is not None else ''),
            'family': family, 'seed': seed, 'kind': kind, 'order': order,
            'alpha': .001 if kind == 'varx' else None, 'residual_kind': head,
            'mode': mode, 'hidden_width': width if head == 'tanh' else None,
            'context_iterations': iterations}


def coefficients(order):
    value = np.zeros((3, 6*order+4), dtype=np.float64)
    value[:, 3*(order-1):3*order] = np.eye(3)*.35
    value[:, 3*order:3*order+3] = np.diag([.2, -.1, .3])
    value[:, 6*order:6*order+3] = np.eye(3)*.03
    value[:, -1] = [.02, -.01, .03]
    return value


def residual(spec):
    model = FSMResidual(coefficients(spec['order']), order=spec['order'], mode=spec['mode'],
                        residual_kind=spec['residual_kind'], hidden_width=spec['hidden_width'] or 24,
                        seed=spec['seed'])
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.linspace(-.015, .02, parameter.numel(), dtype=torch.float64)
                            .reshape(parameter.shape)+.002*index)
    arrays = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
    return model, arrays


def author_arrays(order=3, *, nonlinear=False):
    a = np.eye(order)*.4
    a[:3, :3] = np.diag([.8, .7, .6])
    bu = np.zeros((order, 3))
    bu[:3] = np.eye(3)*.1
    cy = np.zeros((3, order))
    cy[:, :3] = [[1., .1, 0.], [0., 1., .1], [.1, 0., 1.]]
    arrays = {'A': a, 'B_u': bu, 'C_y': cy, 'D_yu': np.diag([.2, -.1, .3]),
              'u_mean': np.array([1., -2., 4.]), 'u_std': np.array([.25, 1., 2.]),
              'y_mean': np.array([-5., 8., 3.]), 'y_std': np.array([2., .5, 4.]),
              'ts': np.array(1/6400, dtype=np.float64)}
    if nonlinear:
        nz, nw, width = (16, 8, 64) if order == 28 else (3, 3, 3)
        bw, cz = np.zeros((order, nw)), np.zeros((nz, order))
        bw[:3, :3], cz[:3, :3] = np.eye(3)*.01, np.eye(3)
        dyw, dzu = np.zeros((3, nw)), np.zeros((nz, 3))
        dyw[:, :3], dzu[:3] = np.eye(3)*.04, np.eye(3)*.02
        w0, w2 = np.zeros((width, nz)), np.zeros((nw, width))
        w0[:3, :3], w2[:3, :3] = np.eye(3)*.1, np.eye(3)*.05
        arrays.update(B_w=bw, C_z=cz, D_yw=dyw, D_zu=dzu, W0=w0, b0=np.full(width, .1),
                      W1=np.eye(width), b1=np.full(width, .01), W2=w2, b2=np.full(nw, .005))
    return arrays


CASES = ('varx', 'affine_output_only', 'affine_feedback', 'tanh_output_only', 'tanh_feedback',
         'bla', 'nllfr16', 'nllfr64')


def deployment_case(case, *, order=None):
    if case == 'varx':
        spec = specification('varx', order=order or 3)
        arrays = {'coefficients': coefficients(spec['order'])}
    elif case in ('bla', 'nllfr16', 'nllfr64'):
        nl = case != 'bla'
        spec = specification('nllfr' if nl else 'bla', order=order or (3 if nl else 28),
                             iterations=int(case[-2:]) if nl else None)
        arrays = author_arrays(spec['order'], nonlinear=nl)
    else:
        head, mode = case.split('_', 1)
        spec = specification('residual', order=order or 3, head=head, mode=mode)
        _, arrays = residual(spec)
    norm = normalizer()
    return spec, arrays, norm, shift.make_deployment(spec, arrays, norm)


def same_values(actual, expected, *, exact=True):
    for key in ('prediction', 'forecast_state', 'final_state'):
        if exact:
            np.testing.assert_array_equal(actual[key], expected[key])
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('order', [32, 64, 96])
def test_native_varx_physical_forecast_and_final_lags_are_preserved(order):
    _, arrays, norm, deployment = deployment_case('varx', order=order)
    y, u, future = inputs()
    yc, uc, fu = ((y-norm['y_mean'])/norm['y_scale'], (u-norm['u_mean'])/norm['u_scale'],
                  (future-norm['u_mean'])/norm['u_scale'])
    model = fsm_linear.VARXModel(arrays['coefficients'], order, .001,
                                12*(8192-order), fsm_linear.FIT_IDS)
    predicted = fsm_linear.predict(model, yc, uc, fu)
    expected = {'prediction': predicted*norm['y_scale']+norm['y_mean'],
                'forecast_state': np.concatenate((yc[:, -order:].reshape(2, -1),
                                                   uc[:, -order:].reshape(2, -1)), axis=1),
                'final_state': np.concatenate((predicted[:, -order:].reshape(2, -1),
                                                fu[:, -order:].reshape(2, -1)), axis=1)}
    same_values(shift.request(deployment, y, u, future), expected)


@pytest.mark.parametrize('head,mode', [('affine', 'output_only'), ('affine', 'feedback'),
                                      ('tanh', 'output_only'), ('tanh', 'feedback')])
def test_all_residual_primal_and_caller_states_equal_native_core(head, mode):
    spec = specification('residual', head=head, mode=mode)
    native, arrays = residual(spec)
    norm, (y, u, future) = normalizer(), inputs()
    deployment = shift.make_deployment(spec, arrays, norm)
    yc, uc, fu = ((y-norm['y_mean'])/norm['y_scale'], (u-norm['u_mean'])/norm['u_scale'],
                  (future-norm['u_mean'])/norm['u_scale'])
    with torch.inference_mode():
        state = native.condition(torch.from_numpy(yc), torch.from_numpy(uc))
        predicted, final = native.rollout(torch.from_numpy(fu), state)
    same_values(shift.request(deployment, y, u, future),
                {'prediction': predicted.numpy()*norm['y_scale']+norm['y_mean'],
                 'forecast_state': state.numpy(), 'final_state': final.numpy()})
    assert not deployment.model.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in deployment.model.parameters())


def test_bla_matches_qualified_physical_adapter_and_preforecast_state():
    _, arrays, _, deployment = deployment_case('bla')
    y, u, future = inputs()
    output, final, diagnostic = benchmark.physical_request(arrays, y, u, future)
    state, detail = linear_context.condition(*(arrays[key] for key in benchmark.NAMES[:4]),
        (y-arrays['y_mean'])/arrays['y_std'], (u-arrays['u_mean'])/arrays['u_std'])
    actual = shift.request(deployment, y, u, future)
    same_values(actual, {'prediction': output, 'forecast_state': state, 'final_state': final})
    assert actual['diagnostics'] == shift.json_tree(detail) == shift.json_tree(diagnostic)


@pytest.mark.parametrize('iterations', [16, 64])
def test_nonlinear_matches_qualified_budget_and_original16_with_author_scales(iterations):
    _, arrays, _, deployment = deployment_case(f'nllfr{iterations}')
    y, u, future = inputs(batch=1)
    output, final, diagnostic = nllfr_context_budget.predict(arrays, y, u, future, iterations=iterations)
    state, detail = nllfr_context_budget.condition(arrays, y, u, iterations=iterations)
    actual = shift.request(deployment, y, u, future)
    same_values(actual, {'prediction': output, 'forecast_state': state, 'final_state': final})
    assert actual['diagnostics'] == shift.json_tree(detail) == shift.json_tree(diagnostic)
    if iterations == 16:
        old_output, old_final, old_diagnostic = nllfr_context.predict(arrays, y, u, future)
        np.testing.assert_array_equal(actual['prediction'], old_output)
        np.testing.assert_array_equal(actual['final_state'], old_final)
        assert actual['diagnostics'] == shift.json_tree(old_diagnostic)


@pytest.mark.parametrize('case,expected_model,metadata,state,normalization,total', [
    ('varx', 4704, 24, 1536, 96, 6360),
    ('affine_feedback', 9408, 16, 1536, 96, 11056),
    ('tanh_feedback', 42936, 24, 1536, 96, 44592),
    ('bla', 7792, 24, 224, 0, 8040),
    ('nllfr64', 59888, 72, 224, 0, 60184),
])
def test_deployment_storage_uses_actual_arrays_state_and_required_scales(
        case, expected_model, metadata, state, normalization, total):
    order = 28 if case in ('bla', 'nllfr64') else 32
    if case.startswith('tanh'):
        spec = specification('residual', order=32, head='tanh', mode='feedback', width=24)
        _, arrays = residual(spec)
        deployment = shift.make_deployment(spec, arrays, normalizer())
    else:
        spec, arrays, _, deployment = deployment_case(case, order=order)
    assert deployment.spec == spec
    assert deployment.storage['model_numeric_bytes'] == expected_model == sum(v.nbytes for v in arrays.values())
    assert deployment.storage['policy_metadata_bytes'] == metadata
    assert deployment.storage['state_bytes'] == state
    assert deployment.storage['external_normalizer_bytes'] == normalization
    assert deployment.storage['persistent_numeric_bytes'] == total
    assert 'no request cache' in deployment.storage['scope']


@pytest.mark.parametrize('case', ['varx', 'tanh_feedback', 'bla', 'nllfr16'])
def test_requests_own_outputs_inputs_weights_and_repeat_without_state_cache(case):
    spec, arrays, norm, deployment = deployment_case(case)
    y, u, future = inputs(batch=3)
    saved_inputs = [v.copy() for v in (y, u, future)]
    saved_model = (copy.deepcopy(deployment.model.state_dict()) if case == 'tanh_feedback' else None)
    expected = shift.request(deployment, y, u, future)
    assert set(expected) == {'prediction', 'forecast_state', 'final_state', 'diagnostics'}
    json.dumps(expected['diagnostics'], allow_nan=False)
    # Destructive changes to source constructor buffers must not alter deployment.
    for value in arrays.values():
        value[...] = 99.
    for value in norm.values():
        value[...] = 17.
    spec['family'] = 'changed after construction'
    same_values(shift.request(deployment, y, u, future), expected)
    for key in ('prediction', 'forecast_state', 'final_state'):
        assert expected[key].flags.owndata and not any(np.shares_memory(expected[key], v) for v in (y, u, future))
    for left, right in zip((y, u, future), saved_inputs, strict=True):
        np.testing.assert_array_equal(left, right)
    assert deployment.spec['family'] != spec['family']
    with pytest.raises(TypeError):
        deployment.spec['kind'] = 'changed'
    if saved_model is not None:
        for key, value in deployment.model.state_dict().items():
            assert torch.equal(value, saved_model[key])
    for index in (2, 0, 1):
        one = shift.request(deployment, y[index:index+1], u[index:index+1], future[index:index+1])
        for key in ('prediction', 'forecast_state', 'final_state'):
            np.testing.assert_allclose(one[key], expected[key][index:index+1], atol=1e-12, rtol=1e-12)
    for key in ('prediction', 'forecast_state', 'final_state'):
        expected[key][...] = -999.
    repeated = shift.request(deployment, y, u, future)
    assert not np.any(repeated['prediction'] == -999.)


@pytest.mark.parametrize('case', ['varx', 'tanh_feedback', 'bla', 'nllfr64'])
def test_future_input_is_forecast_only_and_targets_have_no_api(case):
    _, _, _, deployment = deployment_case(case)
    y, u, future = inputs(batch=1)
    first = shift.request(deployment, y, u, future)
    changed = future.copy()
    changed[:, 13:] += 5.
    second = shift.request(deployment, y, u, changed)
    np.testing.assert_array_equal(first['forecast_state'], second['forecast_state'])
    np.testing.assert_array_equal(first['prediction'][:, :13], second['prediction'][:, :13])
    assert first['diagnostics'] == second['diagnostics']
    assert not np.array_equal(first['prediction'][:, 13], second['prediction'][:, 13])
    assert tuple(inspect.signature(shift.request).parameters) == ('deployment', 'y_context', 'u_context', 'future_u')
    with pytest.raises(TypeError):
        shift.request(deployment, y, u, future, target=np.zeros_like(future))


def test_finite_large_values_are_preserved_but_overflow_is_not_repaired():
    spec = specification('varx', order=1)
    arrays = {'coefficients': np.zeros((3, 10))}
    arrays['coefficients'][:, -1] = 1e200
    unit = {key: np.zeros(3) if 'mean' in key else np.ones(3) for key in shift.NORM_KEYS}
    values = tuple(np.zeros((1, n, 3)) for n in (100, 99, 128))
    finite = shift.request(shift.make_deployment(spec, arrays, unit), *values)
    np.testing.assert_array_equal(finite['prediction'], np.full((1, 128, 3), 1e200))
    arrays['coefficients'][:, :3] = np.eye(3)*1e200
    with pytest.raises(ValueError, match='nonfinite VARX prediction'):
        shift.request(shift.make_deployment(spec, arrays, unit), *values)


@pytest.mark.parametrize('attack', ['context_length', 'past_length', 'horizon', 'dtype', 'nonfinite'])
def test_public_shape_dtype_and_finiteness_guards_precede_forecast(attack, monkeypatch):
    _, _, _, deployment = deployment_case('varx')
    y, u, future = inputs(batch=1)
    if attack == 'context_length':
        y = y[:, :-1]
    elif attack == 'past_length':
        u = u[:, :-1]
    elif attack == 'horizon':
        future = future[:, :-1]
    elif attack == 'dtype':
        y = y.astype(np.float32)
    else:
        future[0, 0, 0] = np.inf
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid request reached inference')
    monkeypatch.setattr(shift, 'varx_predict', forbidden)
    with pytest.raises(ValueError, match='finite float64'):
        shift.request(deployment, y, u, future)


@pytest.mark.parametrize('change', [{'instance_id': 'different'}, {'alpha': None}, {'order': True},
                                  {'hidden_width': 2}, {'context_iterations': 16}, {'extra': 1}])
def test_wrong_deployment_roles_and_identity_fail_closed(change):
    spec = specification('varx') | change
    with pytest.raises(ValueError):
        shift.make_deployment(spec, {'coefficients': coefficients(3)}, normalizer())


def test_diagnostic_tree_is_owned_finite_json_with_numpy_leaves():
    values = np.array([[1., .5], [-2., 0.]])
    source = {'requests': ({'state': values, 'rank': np.int64(2),
                            'finite': np.bool_(True), 'loss': np.float32(.5)},),
              'optional': None, 'status': 'complete', 'empty': np.empty((0, 3))}
    actual = shift.json_tree(source)
    expected = {'requests': [{'state': [[1., .5], [-2., 0.]], 'rank': 2,
                              'finite': True, 'loss': .5}],
                'optional': None, 'status': 'complete', 'empty': []}
    assert actual == expected
    assert json.loads(json.dumps(actual, allow_nan=False)) == expected
    values[:] = 99.
    source['requests'][0]['rank'] = 42
    assert actual == expected


@pytest.mark.parametrize('leaf', [float('inf'), np.float64('nan'), np.array([[0., -np.inf]])])
def test_nonfinite_diagnostic_leaf_rejected_through_nested_conversion(leaf):
    with pytest.raises(ValueError, match='nonfinite diagnostic value'):
        shift.json_tree({'requests': [{'trace': ({'value': leaf},)}]})


@pytest.mark.parametrize('leaf', [1+2j, {1, 2}, b'not a JSON string', object()])
def test_unsupported_diagnostic_leaf_is_not_stringified_or_dropped(leaf):
    with pytest.raises(ValueError, match='unsupported diagnostic value'):
        shift.json_tree({'requests': [{'trace': ({'value': leaf},)}]})


@pytest.mark.parametrize('key', [1, None, np.str_('numpy key')])
def test_nested_diagnostic_keys_must_be_plain_strings(key):
    with pytest.raises(ValueError, match='diagnostic keys must be strings'):
        shift.json_tree({'requests': [{key: 0.}]})


@pytest.mark.parametrize('case', CASES)
def test_cross_parity_with_independent_numpy_replay(case):
    path = Path(__file__).resolve().parents[1]/'scripts/fsm_shift_audit_math.py'
    loader = importlib.util.spec_from_file_location('qualified_shift_independent_fixture', path)
    independent = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(independent)
    spec, arrays, norm, deployment = deployment_case(case)
    values = inputs(batch=1)
    produced = shift.request(deployment, *values)
    replayed = independent.replay(spec, arrays, norm, *values)
    for key in ('prediction', 'forecast_state', 'final_state'):
        np.testing.assert_allclose(produced[key], replayed[key], atol=1e-8, rtol=1e-8)
    if case.startswith('nllfr'):
        original = produced['diagnostics']['requests'][0]
        audited = replayed['diagnostics']['per_request'][0]['requests'][0]
        assert original['status'] == audited['status']
        for key in ('directions_considered', 'accepted_steps', 'trial_attempts',
                    'jacobian_evaluations', 'trajectory_evaluations'):
            assert original[key] == audited[key]
