"""Data-free Adam, information-boundary and selected-objective witnesses."""
import copy
import json

import numpy as np
import pytest
import torch

from openjev.research import fsm_sensitivity as sensitivity
from openjev.research import fsm_sensitivity_learning as learning
from openjev.research.fsm_residual import FSMResidual


def fixture(*, horizon=128, active=True):
    coefficients = np.zeros((3, 10), dtype=np.float64)
    coefficients[:, :3] = np.array([[.3, .02, .01], [.03, .25, .02], [.01, .04, .2]])
    coefficients[:, 3:6] = np.eye(3)*.2
    coefficients[:, 6:9] = np.eye(3)*.05
    coefficients[:, -1] = [.01, .02, .03]
    model = FSMResidual(coefficients, order=1, mode='feedback', hidden_width=2, seed=31)
    if active:
        with torch.no_grad():
            first, last = model.residual[0], model.residual[2]
            first.weight.copy_(.04+torch.arange(18, dtype=torch.float64).reshape(2, 9)*.001)
            first.bias.copy_(torch.tensor([.02, .03], dtype=torch.float64))
            last.weight.copy_(.1+torch.arange(6, dtype=torch.float64).reshape(3, 2)*.01)
            last.bias.copy_(torch.tensor([.03, .04, .05], dtype=torch.float64))
    y = torch.linspace(.1, .3, 24, dtype=torch.float64).reshape(2, 4, 3)
    u = torch.linspace(.2, .4, 18, dtype=torch.float64).reshape(2, 3, 3)
    future = torch.linspace(.1, .4, 6*horizon, dtype=torch.float64).reshape(2, horizon, 3)
    target = future*.6+.07
    directions = torch.ones((2, 2, 3), dtype=torch.float64)/np.sqrt(3.)
    return model, (y, u, future, target), directions


def exact_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.detach().cpu().numpy().tobytes() == expected.detach().cpu().numpy().tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            exact_tree(a, b)
    else:
        assert type(actual) is type(expected) and actual == expected


def close_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-14)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            close_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close_tree(a, b)
    else:
        assert actual == expected


def assert_plain_diagnostics(value):
    if isinstance(value, dict):
        assert all(isinstance(k, str) for k in value)
        for item in value.values():
            assert_plain_diagnostics(item)
    elif isinstance(value, list):
        for item in value:
            assert_plain_diagnostics(item)
    else:
        assert value is None or type(value) in (str, int, float, bool)
    json.dumps(value, allow_nan=False)


def native_step(model, optimizer, data):
    optimizer.zero_grad(set_to_none=True)
    y, u, future, target = data
    prediction = model.rollout(future, model.condition(y, u))[0]
    mse = (prediction-target).square().mean()
    mse.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return mse.item(), norm.item()


@pytest.mark.parametrize('populated', [False, True])
def test_h128_unregularized_update_is_bitwise_native_adam_including_existing_moments(populated):
    model, data, _directions = fixture()
    optimizer = learning.make_optimizer(model)
    if populated:
        native_step(model, optimizer, data)
    expected = copy.deepcopy(model)
    expected_optimizer = torch.optim.Adam(expected.parameters(), lr=3e-4, weight_decay=0.)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    inputs_before = tuple(t.clone() for t in data)
    buffer = model.coefficients.clone()
    fields = set(vars(model))
    mse, norm = native_step(expected, expected_optimizer, data)
    report = learning.train_step(model, optimizer, *data, arm='unregularized', penalty_weight=0., directions=None)
    exact_tree(model.state_dict(), expected.state_dict())
    exact_tree(optimizer.state_dict(), expected_optimizer.state_dict())
    exact_tree(model.coefficients, buffer)
    exact_tree(data, inputs_before)
    assert set(vars(model)) == fields
    assert report['mse'] == report['objective'] == mse and report['penalty'] == 0.
    assert report['gradient_norm_before_clip'] == norm
    assert report['completed_updates'] == 1 and report['hinge_active_fraction'] is None
    assert report['work'] == {'primal_window_steps': 256, 'learned_tangent_direction_steps': 0,
        'linear_tangent_direction_steps': 0, 'backward_calls': 1, 'optimizer_steps': 1}
    assert_plain_diagnostics(report)


@pytest.mark.parametrize('arm', ['relative_sensitivity', 'max_envelope'])
def test_zero_head_h128_first_hinge_update_matches_unregularized_bitwise(arm):
    model, data, directions = fixture(active=False)
    expected = copy.deepcopy(model)
    optimizer = learning.make_optimizer(model)
    expected_optimizer = learning.make_optimizer(expected)
    native_step(expected, expected_optimizer, data)
    before = directions.clone()
    report = learning.train_step(model, optimizer, *data, arm=arm, penalty_weight=.001, directions=directions)
    assert report['penalty'] == report['hinge_active_fraction'] == 0.
    exact_tree(model.state_dict(), expected.state_dict())
    exact_tree(optimizer.state_dict(), expected_optimizer.state_dict())
    exact_tree(directions, before)
    assert report['work']['learned_tangent_direction_steps'] == 512
    assert report['work']['linear_tangent_direction_steps'] == 512
    assert_plain_diagnostics(report)


@pytest.mark.parametrize('arm', ['l2', 'relative_sensitivity', 'max_envelope'])
def test_active_selected_penalty_contributes_to_preclip_gradient_and_adam_update(arm, monkeypatch):
    model, data, directions = fixture(horizon=4)
    reference = copy.deepcopy(model)
    reference_optimizer = learning.make_optimizer(reference)
    optimizer = learning.make_optimizer(model)
    parameters = tuple(reference.parameters())
    y, u, future, target = data
    mse = (reference.rollout(future, reference.condition(y, u))[0]-target).square().mean()
    mse_gradients = torch.autograd.grad(mse, parameters)
    if arm == 'l2':
        penalty = sum(p.square().sum() for p in parameters)/29
    else:
        result = sensitivity.rollout_tangents(reference, future, reference.condition(y, u), directions)
        learned, baseline = [], []
        for h in (1, 2, 4):
            learned.append(result['output_tangents'][:, :, :h].square().sum(dim=(2, 3))/(3*h))
            baseline.append(result['linear_output_tangents'][:, :, :h].square().sum(dim=(2, 3))/(3*h))
        gain, base = torch.stack(learned, dim=-1), torch.stack(baseline, dim=-1)
        allowance = base if arm == 'relative_sensitivity' else base.max(dim=1, keepdim=True).values
        assert bool((gain > allowance).all())
        penalty = (gain-allowance).square().mean()
    penalty_gradients = torch.autograd.grad(penalty, parameters)
    weight = .2
    expected_gradients = [a+weight*b for a, b in zip(mse_gradients, penalty_gradients, strict=True)]
    assert penalty.item() > 0. and any(bool(torch.count_nonzero(g)) for g in penalty_gradients)
    for p, gradient in zip(parameters, expected_gradients, strict=True):
        p.grad = gradient.clone()
    original_clip = torch.nn.utils.clip_grad_norm_
    original_clip(parameters, 1., error_if_nonfinite=True)
    reference_optimizer.step()
    captured = []
    def capture(parameters, *args, **kwargs):
        parameters = tuple(parameters)
        captured.extend(p.grad.clone() for p in parameters)
        return original_clip(parameters, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture)
    report = learning.train_step(model, optimizer, *data, arm=arm, penalty_weight=weight,
        directions=None if arm == 'l2' else directions, horizons=(1, 2, 4))
    assert len(captured) == 4
    for actual, expected in zip(captured, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-14)
    assert any(not torch.equal(a, b) for a, b in zip(captured, mse_gradients, strict=True))
    close_tree(model.state_dict(), reference.state_dict())
    close_tree(optimizer.state_dict(), reference_optimizer.state_dict())
    assert report['penalty'] == pytest.approx(penalty.item(), rel=2e-14)
    assert report['objective'] == pytest.approx(report['mse']+weight*report['penalty'], rel=2e-14)
    assert_plain_diagnostics(report)


def test_all_cheap_arms_avoid_tangent_work_and_all_sensitivity_arms_use_one_pass(monkeypatch):
    original_primal, original_tangent = sensitivity.rollout_primal, sensitivity.rollout_tangents
    calls = {'primal': 0, 'tangent': 0}
    def primal(*args, **kwargs):
        calls['primal'] += 1
        return original_primal(*args, **kwargs)
    def tangent(*args, **kwargs):
        calls['tangent'] += 1
        return original_tangent(*args, **kwargs)
    monkeypatch.setattr(sensitivity, 'rollout_primal', primal)
    monkeypatch.setattr(sensitivity, 'rollout_tangents', tangent)
    for arm in sensitivity.ARMS:
        calls.update(primal=0, tangent=0)
        model, data, directions = fixture(horizon=4)
        sensitive = arm in learning.TANGENT_ARMS
        report = learning.train_step(model, learning.make_optimizer(model), *data, arm=arm,
            penalty_weight=0. if arm == 'unregularized' else .001,
            directions=directions if sensitive else None, horizons=(1, 2, 4))
        assert calls == {'primal': 0 if sensitive else 1, 'tangent': 1 if sensitive else 0}
        assert report['work'] == {'primal_window_steps': 8,
            'learned_tangent_direction_steps': 16 if sensitive else 0,
            'linear_tangent_direction_steps': 16 if sensitive else 0, 'backward_calls': 1, 'optimizer_steps': 1}


@pytest.mark.parametrize('failure', ['nan_entry', 'infinite_entry', 'finite_entries_norm_overflow'])
def test_nonfinite_gradient_or_native_norm_rejects_before_optimizer_mutation(failure, monkeypatch):
    model, data, _directions = fixture(horizon=4)
    optimizer = learning.make_optimizer(model)
    native_step(model, optimizer, data)
    parameters_before, optimizer_before = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    value = {'nan_entry': float('nan'), 'infinite_entry': float('inf'), 'finite_entries_norm_overflow': 1e308}[failure]
    handle = next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, value))
    monkeypatch.setattr(optimizer, 'step', lambda *a, **k: pytest.fail('invalid gradient must not reach Adam'))
    error = RuntimeError if failure == 'finite_entries_norm_overflow' else ValueError
    with pytest.raises(error, match='non-finite|nonfinite'):
        learning.train_step(model, optimizer, *data, arm='unregularized', penalty_weight=0., directions=None)
    handle.remove()
    if failure == 'finite_entries_norm_overflow':
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    exact_tree(model.state_dict(), parameters_before)
    exact_tree(optimizer.state_dict(), optimizer_before)


@pytest.mark.parametrize('failure', ['foreign', 'subset', 'duplicate'])
def test_mismatched_optimizer_ownership_is_rejected_before_clearing_gradients_or_forward(failure, monkeypatch):
    model, data, _directions = fixture(horizon=4)
    optimizer = learning.make_optimizer(copy.deepcopy(model) if failure == 'foreign' else model)
    if failure == 'subset':
        optimizer.param_groups[0]['params'].pop()
    elif failure == 'duplicate':
        optimizer.param_groups[0]['params'][1] = optimizer.param_groups[0]['params'][0]
    monkeypatch.setattr(optimizer, 'zero_grad', lambda *a, **k: pytest.fail('ownership must precede clearing'))
    monkeypatch.setattr(model, 'condition', lambda *a, **k: pytest.fail('ownership must precede inference'))
    with pytest.raises(ValueError, match='optimizer must own exactly'):
        learning.train_step(model, optimizer, *data, arm='unregularized', penalty_weight=0., directions=None)


def test_targets_affect_only_loss_and_update_and_never_mutate_caller_inputs(monkeypatch):
    original = sensitivity.rollout_primal
    captured = []
    def inference(model, future, state):
        result = original(model, future, state)
        captured.append((future.clone(), state.clone(), result['prediction'].detach().clone()))
        return result
    monkeypatch.setattr(sensitivity, 'rollout_primal', inference)
    first, data, _directions = fixture(horizon=4)
    second = copy.deepcopy(first)
    data = (*data[:3], torch.full_like(data[3], -1.))
    changed = (*data[:3], torch.full_like(data[3], 1.))
    before = tuple(value.clone() for value in (*data, changed[-1]))
    reports = []
    for model, batch in ((first, data), (second, changed)):
        reports.append(learning.train_step(model, learning.make_optimizer(model), *batch,
            arm='unregularized', penalty_weight=0., directions=None))
    exact_tree(captured[0], captured[1])
    exact_tree((*data, changed[-1]), before)
    assert reports[0]['mse'] != reports[1]['mse']
    assert any(not torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters(), strict=True))


def test_invalid_targets_and_unused_directions_reject_before_zero_grad(monkeypatch):
    model, data, directions = fixture(horizon=4)
    optimizer = learning.make_optimizer(model)
    monkeypatch.setattr(optimizer, 'zero_grad', lambda *a, **k: pytest.fail('validate caller domain first'))
    malformed = (*data[:3], data[3][:, :-1])
    with pytest.raises(ValueError, match='supervised target'):
        learning.train_step(model, optimizer, *malformed, arm='unregularized', penalty_weight=0., directions=None)
    with pytest.raises(ValueError, match='does not consume'):
        learning.train_step(model, optimizer, *data, arm='l2', penalty_weight=.001, directions=directions)
    with pytest.raises(ValueError, match='explicit detached'):
        learning.train_step(model, optimizer, *data, arm='relative_sensitivity', penalty_weight=.001, directions=None)
