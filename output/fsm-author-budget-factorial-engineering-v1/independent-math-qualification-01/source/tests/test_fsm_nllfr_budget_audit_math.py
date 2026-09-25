# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated checks of independent budget replay; no producer/data imports."""
import copy
import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/fsm_nllfr_budget_audit_math.py'
SPEC = importlib.util.spec_from_file_location('budget_audit_math_test', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def model():
    return {'A': np.array([[.5]]), 'B_u': np.array([[2., 0., 0.]]),
        'C_y': np.array([[1.], [2.], [-1.]]), 'D_yu': np.diag([1., 3., 4.]),
        'B_w': np.array([[.02]]), 'C_z': np.array([[1.]]),
        'D_yw': np.array([[.1], [.2], [.05]]), 'D_zu': np.array([[.1, 0., 0.]]),
        'W0': np.ones((1, 1)), 'b0': np.array([.5]),
        'W1': np.ones((1, 1)), 'b1': np.array([.2]),
        'W2': np.ones((1, 1)), 'b2': np.array([-.1]),
        'u_mean': np.array([2., -3., .5]), 'u_std': np.array([.5, 2., 3.]),
        'y_mean': np.array([-4., 1., 7.]), 'y_std': np.array([3., .25, 2.]),
        'ts': np.array(1/6400., dtype=np.float64)}


def fixture():
    """Hand scalar recurrence: w=x+.63 for the constant input u0=.3."""
    m, x, rows = model(), 2., []
    u = np.zeros((1, 17, 3))
    u[:, :, 0] = .3
    for _ in range(17):
        w = x+.63
        rows.append([x+.3+.1*w, 2*x+.2*w, -x+.05*w])
        x = .5*x+.6+.02*w
    target = np.array([rows])
    return m, target, u, np.array([[1.]]), x


def exact(actual, expected):
    """Check bits, including signed zero, instead of a numerical tolerance."""
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            exact(a, e)
    elif isinstance(expected, (float, np.floating)):
        assert float(actual).hex() == float(expected).hex()
    else:
        assert actual == expected


@pytest.mark.parametrize('kind', ['nonlinear', 'linear', 'dead'])
def test_sixteen_matches_held_independent_solver_bitwise(kind):
    m, target, u, seed, _ = fixture()
    if kind == 'linear':
        m['B_w'].fill(0.)
        m['D_yw'].fill(0.)
    elif kind == 'dead':
        m['C_y'].fill(0.)
        m['D_yw'].fill(0.)
    old_policy = copy.deepcopy(audit.old.POLICY)
    exact(audit.solve_context(m, target, u, seed, iterations=16),
          audit.old.solve_context(m, target, u, seed))
    assert audit.old.POLICY == old_policy


def test_known_nonlinear_state_recovery_and_early_stop_are_budget_independent():
    m, target, u, seed, true_final = fixture()
    sixteen = audit.solve_context(m, target, u, seed, iterations=16)
    sixtyfour = audit.solve_context(m, target, u, seed, iterations=64)
    assert sixteen[2]['status'] == 'GRADIENT_TOL'
    exact(sixtyfour, sixteen)
    np.testing.assert_allclose(sixteen[1], [[2.]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(sixteen[0], [[true_final]], rtol=0, atol=1e-7)
    row = sixteen[2]
    assert row['accepted_steps'] > 0 and row['final_objective'] < row['initial_objective']
    assert row['jacobian_evaluations'] == row['directions_considered']+1
    assert row['trajectory_evaluations'] == row['jacobian_evaluations']+row['trial_attempts']


def test_physical_replay_parity_causal_index_and_policy_ownership():
    m = model()
    m['B_w'].fill(0.)
    m['D_yw'].fill(0.)
    # x1=2, output before update: x2=7 and x3=3.5. u0 is unavailable.
    y = np.array([[[999., -777., 555.], [5., 16., 18.], [7., 17., 1.]]])
    u = np.array([[[3., 4., 5.], [0., 1., 2.]]])
    future = np.array([[[4., 2., 1.], [0., 1., 0.]]])
    args = (y*m['y_std']+m['y_mean'], u*m['u_std']+m['u_mean'],
            future*m['u_std']+m['u_mean'])
    before = copy.deepcopy((m, args, audit.old.POLICY))
    expected = audit.old.replay_request(m, *args)
    result = audit.replay_request(m, *args, iterations=16)
    exact(result, expected)
    longer = audit.replay_request(m, *args, iterations=64)
    exact(longer[:3], result[:3])
    assert longer[3]['policy'] == {**result[3]['policy'], 'iterations': 64}
    exact(longer[3]['requests'], result[3]['requests'])
    np.testing.assert_allclose(result[2], [[3.5]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(result[0][:, :1],
        np.array([[[7.5, 13., .5]]])*m['y_std']+m['y_mean'], rtol=0, atol=1e-13)
    changed = [v.copy() for v in args]
    changed[0][:, 0] = 1e12
    changed[2][:, 1:] += 1e3
    again = audit.replay_request(m, *changed, iterations=64)
    exact(again[2], result[2])
    exact(again[3]['solved_context_start_states'], result[3]['solved_context_start_states'])
    exact(again[0][:, :1], result[0][:, :1])
    assert not np.array_equal(again[0][:, 1:], result[0][:, 1:])
    for iterations in (16, 64):
        empty = audit.replay_request(m, args[0], args[1], args[2][:, :0], iterations=iterations)
        assert empty[0].shape == (1, 0, 3)
        exact(empty[1], result[2])
        exact(empty[2], result[2])
        assert not np.shares_memory(empty[1], empty[2])
    exact((m, args, audit.old.POLICY), before)
    for value in result[:3]:
        assert not any(np.shares_memory(value, other) for other in args)
    result[3]['policy']['damping'] = 10.
    assert audit.policy()['damping'] == audit.old.POLICY['damping'] == .001
    with pytest.raises(TypeError):
        audit.replay_request(m, *args, target=np.zeros((1, 2, 3)))


def test_polynomial_gn_witness_extends_exact_prefix_without_restarting(monkeypatch):
    """Test the GN budget with independent f(x)=x² and analytic df/dx=2x.

    This is a solver control fixture, not an NL-LFR representation claim.
    For the normalized scalar problem each full step multiplies x by
    1-1/(2*1.001), so 16 steps cannot reach the declared gradient tolerance.
    """
    calls = []
    def polynomial(_m, inputs, state, *, jacobian=False):
        x = float(state[0, 0])
        calls.append((jacobian, x))
        shape = inputs.shape
        prediction = np.full(shape, x*x)
        derivative = np.full((*shape, 1), 2*x) if jacobian else None
        return prediction, state.copy(), derivative
    monkeypatch.setattr(audit, 'trajectory', polynomial)
    target, u, seed = np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.array([[1e8]])
    sixteen = audit.solve_context(model(), target, u, seed, iterations=16)
    calls.clear()
    longer = audit.solve_context(model(), target, u, seed, iterations=64)
    assert sixteen[2]['status'] == 'ITERATION_CAP'
    assert sixteen[2]['accepted_steps'] == sixteen[2]['directions_considered'] == 16
    assert longer[2]['directions_considered'] > 16
    exact(longer[2]['trace'][:16], sixteen[2]['trace'])
    np.testing.assert_allclose(sixteen[1], [[1e8*(1-1/(2*1.001))**16]], rtol=2e-14, atol=0)
    assert calls[0] == (True, 1e8)
    assert longer[2]['final_objective'] <= sixteen[2]['final_objective']
    assert longer[2]['status'] == 'GRADIENT_TOL'
    assert all(t['accepted'] and t['alpha'] == 1. for row in longer[2]['trace'] for t in row['trials'])
    np.testing.assert_array_equal(seed, [[1e8]])


def fake_trajectory(kind, calls):
    def run(_m, inputs, state, *, jacobian=False):
        calls.append((jacobian, state.copy()))
        x = float(state[0, 0])
        if kind == 'current':
            raise FloatingPointError('nonfinite current fixture')
        if kind == 'unknown':
            raise ValueError('unrelated structural fixture')
        if kind == 'factorization':
            raise np.linalg.LinAlgError('factorization fixture')
        if not jacobian and kind == 'trial' and x < -.75:
            raise FloatingPointError('nonfinite trial fixture')
        value = 1. if kind in ('stall', 'zero') else 1.+x
        prediction = np.full(inputs.shape, value)
        derivative = np.full((*inputs.shape, 1), 0. if kind == 'zero' else 1.) if jacobian else None
        return prediction, state.copy(), derivative
    return run


@pytest.mark.parametrize('kind,status', [('zero', 'ZERO_JACOBIAN'), ('stall', 'STALLED')])
def test_early_stop_not_retried_under_larger_budget(monkeypatch, kind, status):
    calls = []
    monkeypatch.setattr(audit, 'trajectory', fake_trajectory(kind, calls))
    args = (model(), np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.zeros((1, 1)))
    short = audit.solve_context(*args, iterations=16)
    calls.clear()
    long = audit.solve_context(*args, iterations=64)
    exact(long, short)
    row = long[2]
    assert row['status'] == status and row['accepted_steps'] == 0
    assert row['trial_attempts'] == (8 if kind == 'stall' else 0)
    assert row['trajectory_evaluations'] == len(calls) == 2+row['trial_attempts']


def test_only_nonfinite_trial_is_rejected_and_counted(monkeypatch):
    calls = []
    monkeypatch.setattr(audit, 'trajectory', fake_trajectory('trial', calls))
    _, state, row = audit.solve_context(model(), np.zeros((1, 2, 3)),
        np.zeros((1, 2, 3)), np.zeros((1, 1)), iterations=64)
    trial = row['trace'][0]['trials']
    assert len(trial) == 2 and trial[0]['nonfinite'] and not trial[0]['accepted']
    assert trial[1]['accepted'] and trial[1]['alpha'] == .5
    assert row['trajectory_evaluations'] == len(calls)
    assert row['trial_attempts'] == sum(len(t['trials']) for t in row['trace'])
    assert row['final_objective'] < row['initial_objective'] and np.isfinite(state).all()


@pytest.mark.parametrize('kind,error', [('current', FloatingPointError), ('unknown', ValueError),
                                       ('factorization', np.linalg.LinAlgError)])
def test_current_or_structural_failure_propagates_without_fallback(monkeypatch, kind, error):
    calls = []
    monkeypatch.setattr(audit, 'trajectory', fake_trajectory(kind, calls))
    with pytest.raises(error, match='fixture'):
        audit.solve_context(model(), np.zeros((1, 2, 3)), np.zeros((1, 2, 3)),
                            np.zeros((1, 1)), iterations=64)
    assert len(calls) == 1


@pytest.mark.parametrize('value', [0, 1, 15, 17, 63, 65, True, 16., np.int64(16), '64', None])
def test_only_exact_python_integer_budgets_accepted(value):
    with pytest.raises(ValueError, match='iterations must be exactly'):
        audit.policy(iterations=value)
    with pytest.raises(ValueError, match='iterations must be exactly'):
        audit.solve_context({}, None, None, None, iterations=value)
    with pytest.raises(ValueError, match='iterations must be exactly'):
        audit.replay_request({}, None, None, None, iterations=value)


@pytest.mark.parametrize('slot,change', [
    ('target', 'float32'), ('target', 'nan'), ('target', 'broadcast'),
    ('u', 'batch'), ('u', 'empty'), ('seed', 'nan'), ('seed', 'shape')])
def test_context_schema_cannot_broadcast_or_hide_invalid_values(slot, change):
    m, target, u, seed, _ = fixture()
    values = {'target': target, 'u': u, 'seed': seed}
    value = values[slot].copy()
    if change == 'float32':
        value = value.astype(np.float32)
    elif change == 'nan':
        value.flat[0] = np.nan
    elif change == 'broadcast':
        value = value[:, :1]
    elif change == 'batch':
        value = np.repeat(value, 2, axis=0)
    elif change == 'empty':
        value = value[:, :0]
    else:
        value = value.reshape(1)
    values[slot] = value
    with pytest.raises(ValueError):
        audit.solve_context(m, values['target'], values['u'], values['seed'], iterations=64)


def test_helper_hash_rejection_precedes_import(monkeypatch):
    original = Path.read_bytes
    def changed(path):
        return b'altered helper' if path == audit.ROOT/audit.HELPER else original(path)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='qualified independent helper changed'):
        runpy.run_path(str(SCRIPT))
