"""Fabricated-only algebra, forecast equivalence, ownership and causal checks."""
import copy
import gc
import weakref
from dataclasses import fields

import numpy as np
import pytest
import torch

from openjev.research.fsm_affine_fold import fold_affine_feedback
from openjev.research.fsm_linear import FIT_IDS, VARXModel, predict
from openjev.research.fsm_residual import FSMResidual


def fixture(order=32, seed=11):
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(0, .002/order, (3, 6*order+4)).astype(np.float64)
    coefficients[:, 3*(order-1):3*order] += np.eye(3)*.3
    coefficients[:, 3*order:3*order+3] += np.eye(3)*.2
    coefficients[:, -1] = [.01, -.02, .03]
    parent = VARXModel(np.asfortranarray(coefficients), order, .001,
                       12*(8192-order), FIT_IDS)
    residual = FSMResidual(parent.coefficients, order=order, mode='feedback',
                           residual_kind='affine', seed=seed)
    with torch.no_grad():
        residual.residual.weight.copy_(torch.from_numpy(rng.normal(0, .003/order, (3, 6*order+3))))
        residual.residual.bias.copy_(torch.from_numpy(rng.normal(0, .01, 3)))
    y = rng.normal(0, .3, (2, 100, 3))
    u = rng.normal(0, .4, (2, 99, 3))
    future = rng.normal(0, .5, (2, 128, 3))
    return parent, residual, (y, u, future)


@pytest.mark.parametrize('order', [32, 64, 96])
@pytest.mark.parametrize('seed', [11, 31, 101])
def test_nonzero_random_head_full_h128_forecast_equivalence(order, seed):
    parent, residual, inputs = fixture(order, seed)
    folded = fold_affine_feedback(residual, parent)
    with torch.no_grad():
        original = residual.predict(*(torch.from_numpy(value) for value in inputs)).numpy()
    merged = predict(folded, *inputs)
    np.testing.assert_allclose(merged, original, rtol=2e-13, atol=2e-13)
    assert np.count_nonzero(residual.residual.weight.detach().numpy()) == 3*(6*order+3)
    assert not np.array_equal(folded.coefficients, parent.coefficients)


@pytest.mark.parametrize('order', [32, 64, 96])
def test_zero_head_preserves_all_parent_coefficients_and_forecasts(order):
    parent, _, inputs = fixture(order)
    residual = FSMResidual(parent.coefficients, order=order, mode='feedback', residual_kind='affine')
    folded = fold_affine_feedback(residual, parent)
    np.testing.assert_array_equal(folded.coefficients, parent.coefficients)
    np.testing.assert_array_equal(predict(folded, *inputs), predict(parent, *inputs))


def test_independent_coefficient_and_one_step_feature_order_oracle():
    order = 2
    original = (np.arange(48).reshape(3, 16)-24).astype(np.float64)/100
    parent = VARXModel(original, order, .1, 100, FIT_IDS)
    residual = FSMResidual(original, order=order, mode='feedback', residual_kind='affine')
    head = (np.arange(45).reshape(3, 15)+1).astype(np.float64)/500
    bias = np.array([.25, -.125, .5])
    with torch.no_grad():
        residual.residual.weight.copy_(torch.from_numpy(head))
        residual.residual.bias.copy_(torch.from_numpy(bias))
    folded = fold_affine_feedback(residual, parent)
    expected_coefficients = np.column_stack((original[:, :-1]+head, original[:, -1]+bias))
    np.testing.assert_array_equal(folded.coefficients, expected_coefficients)
    y = np.arange(12, dtype=np.float64).reshape(1, 4, 3)/10
    u = np.arange(9, dtype=np.float64).reshape(1, 3, 3)/20
    future = np.array([[[.7, -.4, .3]]])
    features = np.concatenate((y[:, -2:].reshape(1, -1), future[:, 0], u[:, -2:].reshape(1, -1)), axis=1)
    expected = np.empty((1, 3))
    for output in range(3):
        expected[0, output] = sum(float(features[0, k])*float(expected_coefficients[output, k])
                                   for k in range(15))+float(expected_coefficients[output, -1])
    np.testing.assert_allclose(predict(folded, y, u, future)[:, 0], expected, rtol=2e-14, atol=2e-14)
    state = residual.condition(torch.from_numpy(y), torch.from_numpy(u))
    one, _ = residual.step(state, torch.from_numpy(future[:, 0]))
    np.testing.assert_allclose(one.detach().numpy(), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize('order', [32, 64, 96])
def test_lag_conditioning_prefix_causality_batch_and_segmented_forecasts(order):
    parent, residual, inputs = fixture(order)
    y, u, future = inputs; originals = [value.copy() for value in inputs]
    folded = fold_affine_feedback(residual, parent)
    whole = predict(folded, y, u, future)
    cut = 43
    first = predict(folded, y, u, future[:, :cut])
    # Context u omits its first time point, so extending both observed prefixes
    # with executed future values preserves C-1 exactly at the chunk boundary.
    continued_y = np.concatenate((y, first), axis=1)
    continued_u = np.concatenate((u, future[:, :cut]), axis=1)
    second = predict(folded, continued_y, continued_u, future[:, cut:])
    np.testing.assert_array_equal(np.concatenate((first, second), axis=1), whole)
    for batch in range(2):
        single = predict(folded, y[batch:batch+1], u[batch:batch+1], future[batch:batch+1])
        np.testing.assert_allclose(single, whole[batch:batch+1], rtol=2e-13, atol=2e-13)
    changed_future = future.copy(); changed_future[:, 51:] += 7.
    changed = predict(folded, y, u, changed_future)
    np.testing.assert_array_equal(changed[:, :51], whole[:, :51])
    older_y, older_u = y.copy(), u.copy()
    older_y[:, :-order] += 123.; older_u[:, :-order] -= 123.
    np.testing.assert_array_equal(predict(folded, older_y, older_u, future), whole)
    assert predict(folded, y, u, future[:, :0]).shape == (2, 0, 3)
    state = residual.condition(torch.from_numpy(y), torch.from_numpy(u))
    expected_state = np.concatenate((y[:, -order:].reshape(2, -1), u[:, -order:].reshape(2, -1)), axis=1)
    np.testing.assert_array_equal(state.numpy(), expected_state)
    for value, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(value, original)


def test_no_mutation_alias_or_retained_module_and_exact_storage():
    parent, residual, _ = fixture()
    original_parent = parent.coefficients.copy()
    original_state = {key: value.detach().clone() for key, value in residual.state_dict().items()}
    for parameter in residual.parameters():
        parameter.grad = torch.full_like(parameter, .3)
    original_gradients = [parameter.grad.clone() for parameter in residual.parameters()]
    before_rng = torch.get_rng_state().clone()
    folded = fold_affine_feedback(residual, parent)
    assert torch.equal(before_rng, torch.get_rng_state())
    for key, value in residual.state_dict().items():
        assert torch.equal(value, original_state[key])
    for parameter, gradient in zip(residual.parameters(), original_gradients, strict=True):
        assert torch.equal(parameter.grad, gradient)
    np.testing.assert_array_equal(parent.coefficients, original_parent)
    assert not np.shares_memory(folded.coefficients, parent.coefficients)
    assert not np.shares_memory(folded.coefficients, residual.coefficients.numpy())
    assert not np.shares_memory(folded.coefficients[:, :-1], residual.residual.weight.detach().numpy())
    assert not folded.coefficients.flags.writeable
    expected = folded.coefficients.copy()
    with torch.no_grad():
        residual.coefficients.add_(2.); residual.residual.weight.add_(3.); residual.residual.bias.add_(4.)
    np.testing.assert_array_equal(folded.coefficients, expected)
    assert tuple(field.name for field in fields(folded)) == ('coefficients', 'order', 'alpha', 'fit_rows', 'fit_record_ids')
    assert not hasattr(folded, '__dict__') and not hasattr(folded, 'parameters')
    reference = weakref.ref(residual)
    del residual
    gc.collect()
    assert reference() is None
    spec = folded.model_spec()
    assert spec['parameter_count'] == 588 and spec['parameter_bytes'] == 4704
    assert spec['scalar_metadata_bytes'] == 24 and spec['retained_numeric_bytes'] == 4728
    assert spec['state_scalars'] == 192 and spec['state_bytes_per_stream'] == 1536
    assert spec['retained_numeric_bytes']+spec['state_bytes_per_stream']+96 == 6360
    assert folded.order == parent.order and folded.alpha == parent.alpha
    assert folded.fit_rows == parent.fit_rows and folded.fit_record_ids == parent.fit_record_ids


@pytest.mark.parametrize('kind,mode', [('tanh', 'feedback'), ('tanh', 'output_only'), ('affine', 'output_only')])
def test_different_residual_recurrences_cannot_be_folded(kind, mode):
    parent, _, _ = fixture()
    residual = FSMResidual(parent.coefficients, order=32, residual_kind=kind, mode=mode)
    with pytest.raises(ValueError, match='only affine feedback'):
        fold_affine_feedback(residual, parent)


@pytest.mark.parametrize('case', ['wrong_parent', 'wrong_order', 'parent_type', 'model_type',
                                  'float32', 'nan_weight', 'nan_bias', 'nan_backbone',
                                  'extra_parameter', 'extra_buffer', 'head_shape', 'trainable_backbone'])
def test_invalid_backbone_or_tensor_contract_fails(case):
    parent, residual, _ = fixture()
    if case == 'wrong_parent':
        coefficients = parent.coefficients.copy(); coefficients[0, 0] += .001
        parent = VARXModel(coefficients, 32, .001, parent.fit_rows, FIT_IDS)
    elif case == 'wrong_order':
        parent = VARXModel(np.zeros((3, 388)), 64, .001, 100, FIT_IDS)
    elif case == 'parent_type':
        parent = parent.coefficients
    elif case == 'model_type':
        residual = residual.residual
    elif case == 'float32':
        residual.float()
    elif case == 'nan_weight':
        with torch.no_grad(): residual.residual.weight[0, 0] = float('nan')
    elif case == 'nan_bias':
        with torch.no_grad(): residual.residual.bias[0] = float('nan')
    elif case == 'nan_backbone':
        with torch.no_grad(): residual.coefficients[0, 0] = float('nan')
    elif case == 'extra_parameter':
        residual.register_parameter('hidden', torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
    elif case == 'extra_buffer':
        residual.register_buffer('hidden', torch.zeros(1, dtype=torch.float64))
    elif case == 'head_shape':
        residual.residual = torch.nn.Linear(4, 3, dtype=torch.float64)
    else:
        residual.coefficients.requires_grad_(True)
    with pytest.raises(ValueError):
        fold_affine_feedback(residual, parent)


@pytest.mark.parametrize('field,value', [('alpha', -1.), ('fit_rows', 0), ('fit_record_ids', ('wrong',)),
                                         ('coefficients', np.zeros((3, 196), dtype=np.float32))])
def test_parent_provenance_revalidated_if_frozen_dataclass_was_bypassed(field, value):
    parent, residual, _ = fixture()
    corrupted = copy.copy(parent)
    object.__setattr__(corrupted, field, value)
    with pytest.raises(ValueError):
        fold_affine_feedback(residual, corrupted)


@pytest.mark.parametrize('location', ['weight', 'bias'])
def test_finite_addends_that_overflow_are_rejected_without_repair(location):
    coefficients = np.zeros((3, 196), dtype=np.float64)
    index = 0 if location == 'weight' else -1
    coefficients[0, index] = np.finfo(np.float64).max
    parent = VARXModel(coefficients, 32, .001, 100, FIT_IDS)
    residual = FSMResidual(parent.coefficients, order=32, mode='feedback', residual_kind='affine')
    with torch.no_grad():
        if location == 'weight': residual.residual.weight[0, 0] = np.finfo(np.float64).max
        else: residual.residual.bias[0] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match='nonfinite folded coefficients; no clipping or repair'):
        fold_affine_feedback(residual, parent)
    assert np.isfinite(parent.coefficients).all() and bool(torch.isfinite(residual.residual.weight).all())
