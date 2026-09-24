"""Fabricated runner contracts; no sampled GP fields or scientific fit calls."""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import retention_study as study

from openjev.research import retention_memory as memory
from openjev.research.retention_policy import RetentionPolicy, leave_one_out_advantage


def fixture(batch=2, steps=12, queries=1):
    axis = np.arange(-2., 2.01, .25)
    grid = np.array([(x, y) for x in axis for y in axis], np.float64)
    indices = (37*np.arange(steps)[None]+11*np.arange(batch)[:, None]) % len(grid)
    x = grid[indices]
    y = np.sin(x[..., 0])+.25*np.cos(2*x[..., 1])+.01*np.sin(np.arange(steps))[None]
    paths = np.array([[[-1., -.5], [-.5, -.5], [0., -.5], [.5, -.5]],
                      [[-.5, -1.], [-.5, -.5], [-.5, 0.], [-.5, .5]],
                      [[-1., .5], [-.5, .5], [0., .5], [.5, .5]],
                      [[.5, -1.], [.5, -.5], [.5, 0.], [.5, .5]]])
    paths = np.broadcast_to(paths, (batch, queries, 4, 4, 2)).copy()
    return {'x': x, 'y': y, 'paths': paths, 'exposure': np.zeros((batch, queries, 4))}


def state_arrays(state):
    return {name: getattr(state, name) for name in ('Z', 'mean', 'cov', 'y') if hasattr(state, name)}


def test_path_prediction_uses_joint_covariance_not_marginal_only_average():
    state = memory.initial(1, 8)
    path = np.repeat(np.array([[.25, -.5]]), 4, axis=0)
    paths = np.broadcast_to(path, (1, 1, 4, 4, 2)).copy()
    prediction = study.path_prediction(state, paths)
    # Four copies of one latent variable have its variance, not one quarter.
    np.testing.assert_allclose(prediction['variance'], 1.00001, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(prediction['mean'], np.zeros((1, 1, 4)))
    np.testing.assert_allclose(prediction['risk'], ndtr(-.5/math.sqrt(1.00001)), atol=2e-15)
    assert not np.isclose(prediction['variance'][0, 0, 0], 1.00001/4)


def test_known_gaussian_metrics_and_defer_are_scored_against_public_risk():
    probability = .5*math.erfc(.5/math.sqrt(2))
    prediction = {'mean': np.zeros((1, 1, 4)), 'variance': np.ones((1, 1, 4)),
                  'risk': np.full((1, 1, 4), probability)}
    reference = {'risk': np.array([[[0., .3, .4, .5]]])}
    actual = study.metrics(prediction, reference, np.zeros((1, 1, 4)))
    expected = {'regret': .18, 'nll': .5*math.log(2*math.pi), 'brier': probability**2,
                'mse': 0., 'coverage90': 1., 'defer': 1., 'always_defer_regret': .18,
                'risk_mae': sum(abs(probability-value) for value in (0., .3, .4, .5))/4}
    assert set(actual) == set(study.METRICS)
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, rel=1e-14, abs=1e-14)
    costs = study.costs(np.array([[[.18, .18, .4, .5]]]))
    assert costs.shape == (1, 1, 5)
    assert costs.argmin(-1).item() == 0  # Registered lowest-index rule; defer is index four.


@pytest.mark.parametrize('method', ('learned', *study.CONTROLS))
def test_future_request_and_private_exposure_cannot_change_retention(method):
    torch.manual_seed(953201)
    policy = RetentionPolicy() if method == 'learned' else None
    inputs = fixture(batch=1)
    altered = {name: value.copy() for name, value in inputs.items()}
    altered['paths'] = altered['paths'][:, :, ::-1]+.25
    altered['exposure'].fill(1e100)
    _, first, trace, floors = study.evaluate(inputs, method, policy)
    _, second, other_trace, other_floors = study.evaluate(altered, method, policy)
    assert first.step == second.step == 12 and floors == other_floors
    for name, value in state_arrays(first).items():
        np.testing.assert_array_equal(value, state_arrays(second)[name])
    np.testing.assert_array_equal(trace, other_trace)
    assert tuple(inspect.signature(study.retain).parameters) == ('x', 'y', 'method', 'policy', 'stochastic')


def test_zero_policy_matches_kl8_actions_and_retained_distribution_exactly():
    torch.manual_seed(953201)
    data = fixture(batch=2, steps=20)
    policy = RetentionPolicy()
    learned = study.retain(data['x'], data['y'], 'learned', policy)
    analytic = study.retain(data['x'], data['y'], 'kl8')
    for name, value in state_arrays(learned[0]).items():
        np.testing.assert_array_equal(value, state_arrays(analytic[0])[name])
    np.testing.assert_array_equal(learned[1], analytic[1])
    assert np.all(learned[1][:, :8] == -1) and np.all(learned[1][:, 8:] >= 0)
    assert learned[2:4] == ([], []) and analytic[2:4] == ([], [])


@pytest.mark.parametrize(('method', 'expected'), [
    ('learned', 1008), ('kl8', 744), ('kl9', 904), ('fifo9', 904),
    ('fic9', 904), ('coverage41', 1024), ('recent41', 1024),
])
def test_complete_retained_state_plus_policy_obeys_frozen_byte_cap(method, expected):
    torch.manual_seed(953201)
    data = fixture(batch=2)
    policy = RetentionPolicy() if method == 'learned' else None
    state, _, _, _, _ = study.retain(data['x'], data['y'], method, policy)
    physical = sum(value.nbytes for value in state_arrays(state).values())//2+40
    assert physical == state.resident_bytes_per_context
    if policy is not None:
        physical += sum(value.numel()*value.element_size() for value in policy.parameters())
    assert physical == expected and physical <= study.CONFIG['state_budget_bytes'] == 1024


def test_three_fabricated_on_policy_updates_are_finite_and_resample_current_policy():
    torch.manual_seed(953201)
    torch.set_num_threads(1)
    public = fixture(batch=2, steps=12)
    repeats = np.repeat(np.arange(2), 4)
    policy = RetentionPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    initial = policy.arrays()
    observed_actions = []
    for _ in range(3):
        state, actions, logps, entropies, _ = study.retain(
            public['x'][repeats], public['y'][repeats], 'learned', policy, stochastic=True)
        prediction = study.path_prediction(state, public['paths'][repeats])
        # Fixed public-risk arithmetic fixture, no GP generator or true outcome.
        reference = {'risk': np.broadcast_to(np.array([.01, .3, .4, .5]), prediction['risk'].shape)}
        rewards = -study.regret(prediction, reference).mean(-1).reshape(2, 4)
        advantage = leave_one_out_advantage(torch.from_numpy(rewards)).reshape(-1)
        assert len(logps) == len(entropies) == 4
        loss = -(advantage*torch.stack(logps).mean(0)).mean()-.002*torch.stack(entropies).mean()
        optimizer.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in policy.parameters())
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
        observed_actions.append(actions)
    assert any(not np.array_equal(initial[name], value) for name, value in policy.arrays().items())
    assert any(not np.array_equal(observed_actions[0], value) for value in observed_actions[1:])


def test_inference_rejects_retained_parameter_gradients_before_writing(monkeypatch):
    policy = RetentionPolicy()
    next(policy.parameters()).grad = torch.ones_like(next(policy.parameters()))
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before retention')
    monkeypatch.setattr(study, 'retain', forbidden)
    with pytest.raises(ValueError, match='gradient tensors'):
        study.evaluate(fixture(batch=1), 'learned', policy)


def test_interrupted_training_preserves_partial_evidence_and_original_exception(tmp_path, monkeypatch):
    stop = RuntimeError('fabricated interrupted update')
    def interrupted(policy, optimizer, data, reference, seed, rows, actions, rewards, fields):
        with torch.no_grad():
            next(policy.parameters()).add_(.125)
        rows.append({'update': 1, 'loss': .25})
        actions.append(np.zeros((4, 12), np.int16))
        rewards.append(np.zeros((1, 4), np.float64))
        fields.append(np.array([0], np.int64))
        raise stop
    monkeypatch.setattr(study, '_updates', interrupted)
    with pytest.raises(RuntimeError) as caught:
        study.train(tmp_path, fixture(batch=1), {}, 953201)
    assert caught.value is stop
    names = {path.name for path in tmp_path.iterdir()}
    assert names == {'policy-953201-initial.npz', 'policy-953201-partial.npz',
                     'optimizer-953201-partial.pt', 'training-953201-partial.npz',
                     'training-953201-partial.json'}
    receipt = json.loads((tmp_path/'training-953201-partial.json').read_text())
    assert receipt['state'] == 'FAILED' and receipt['completed_update_records'] == 1
    assert receipt['error_type'] == 'RuntimeError' and receipt['error'] == str(stop)
    with (np.load(tmp_path/'policy-953201-initial.npz', allow_pickle=False) as initial,
          np.load(tmp_path/'policy-953201-partial.npz', allow_pickle=False) as partial):
        np.testing.assert_allclose(partial['hidden.weight']-initial['hidden.weight'], .125, atol=1e-16)
    with np.load(tmp_path/'training-953201-partial.npz', allow_pickle=False) as trace:
        assert trace['actions'].shape == (1, 4, 12) and trace['rewards'].shape == (1, 1, 4)


def test_successful_training_releases_gradients_and_freezes_inference_policy(tmp_path, monkeypatch):
    def completed(policy, optimizer, data, reference, seed, rows, actions, rewards, fields):
        sum(parameter.sum() for parameter in policy.parameters()).backward()
        rows.append({'update': 1, 'loss': .25})
        actions.append(np.zeros((4, 12), np.int16))
        rewards.append(np.zeros((1, 4), np.float64))
        fields.append(np.array([0], np.int64))
    monkeypatch.setattr(study, '_updates', completed)
    policy = study.train(tmp_path, fixture(batch=1), {}, 953201)
    assert not policy.training
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in policy.parameters())
    assert (tmp_path/'policy-953201-final.npz').is_file()
    assert not any('partial' in path.name for path in tmp_path.iterdir())


def rows_fixture():
    rows = []
    for phase in ('base', 'shift', 'long'):
        for cohort in range(3):
            for method in study.METHODS:
                regret = .08 if method.startswith('learned-') else .10
                rows.append({'phase': phase, 'cohort': cohort, 'method': method,
                             **dict.fromkeys(study.METRICS, .2), 'regret': regret,
                             'always_defer_regret': .3})
    return rows


def test_all_42_conditions_and_best_control_are_preserved():
    rows = rows_fixture()
    result = study.summarize(rows)
    assert len(rows) == 99 and len(result['checks']) == 42
    assert len({row['name'] for row in result['checks']}) == 42
    assert all(row['passed'] for row in result['checks']) and result['gate'] == 'ADVANCE_RETENTION'
    for row in rows:
        if row['phase'] == 'base' and row['method'] == 'recent41':
            row['regret'] = .07
    failed = study.summarize(rows)
    checks = {row['name']: row['passed'] for row in failed['checks']}
    assert not checks['base_regret_vs_recent41'] and not checks['base_cohort_0']
    assert checks['base_regret_vs_kl8'] and failed['gate'] == 'DO_NOT_ADVANCE_RETENTION'


@pytest.mark.parametrize('bad_axis', ['cohort', 'seed'])
def test_aggregate_success_does_not_rescue_bad_cohort_or_fit(bad_axis):
    rows = rows_fixture()
    for row in rows:
        if row['phase'] != 'base' or not row['method'].startswith('learned-'):
            continue
        bad = row['cohort'] == 0 if bad_axis == 'cohort' else row['method'] == 'learned-11'
        row['regret'] = .12 if bad else .06
    result = study.summarize(rows)
    checks = {row['name']: row['passed'] for row in result['checks']}
    assert checks['base_regret_vs_kl8']  # Mean .08 still clears .09.
    assert not checks['base_cohort_0' if bad_axis == 'cohort' else 'base_learned-11']
    assert result['gate'] == 'DO_NOT_ADVANCE_RETENTION'


def test_shift_5_percent_guard_and_nll_best_control_boundary():
    rows = rows_fixture()
    for row in rows:
        if row['phase'] == 'shift' and row['method'].startswith('learned-'):
            row['regret'] = .105002
        if row['phase'] == 'long' and row['method'] == 'coverage41':
            row['nll'] = .17
    checks = {row['name']: row['passed'] for row in study.summarize(rows)['checks']}
    assert not checks['shift_regret_vs_kl8']
    assert not checks['long_nll']  # Learned .20 exceeds best control .17 + .02.
