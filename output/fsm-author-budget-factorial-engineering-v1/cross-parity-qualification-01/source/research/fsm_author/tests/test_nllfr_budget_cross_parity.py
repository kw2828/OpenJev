# SPDX-License-Identifier: GPL-3.0-or-later
"""Fabricated integration only: production versus independent 16/64 replay.

Neither measured arrays nor trained checkpoints are loaded. The independent
implementation remains free of production imports; only this test imports both.
Tolerances are the original fixed replay/context tolerances, not fitted here.
"""
import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent/'src'))
sys.path.insert(0, str(HERE))
from test_nllfr_context import fixture  # noqa: E402

from openjev_fsm_author import nllfr_context_budget as production  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    'independent_context_budget_cross_test', ROOT/'scripts/fsm_nllfr_budget_audit_math.py')
independent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(independent)

ARRAY_TOLERANCE = {'atol': 1e-8, 'rtol': 1e-8}
CONTEXT_TOLERANCE = {'abs_tol': 1e-10, 'rel_tol': 1e-8}


def close_array(actual, expected, *, context=False):
    assert isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
    assert actual.dtype == expected.dtype == np.float64 and actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    tolerance = {'atol': CONTEXT_TOLERANCE['abs_tol'], 'rtol': CONTEXT_TOLERANCE['rel_tol']} if context else ARRAY_TOLERANCE
    np.testing.assert_allclose(actual, expected, **tolerance)


def close_diagnostics(actual, expected):
    """Numerical scalars tolerate rounding; branch identities and work do not."""
    if isinstance(expected, np.ndarray):
        close_array(actual, expected, context=True)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        for key in expected:
            close_diagnostics(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            close_diagnostics(left, right)
    elif isinstance(expected, (float, np.floating)):
        assert isinstance(actual, (float, np.floating))
        assert math.isfinite(actual) and math.isfinite(expected)
        assert math.isclose(actual, expected, **CONTEXT_TOLERANCE)
    else:
        assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize('iterations', [16, 64])
@pytest.mark.parametrize('nx,length,horizon', [(3, 7, 9), (3, 100, 128), (28, 100, 128)],
                         ids=['nonlinear-short', 'nonlinear-C100-H128', 'rank-deficient28-C100-H128'])
def test_same_legal_physical_request_matches_independent_replay(iterations, nx, length, horizon):
    arrays, observed, context_u, future_u, *_ = fixture(nx, 1, length, horizon)
    before = copy.deepcopy((arrays, observed, context_u, future_u))
    # Fixture has nonzero feedback, nonzero physical means, and unequal scales.
    assert np.count_nonzero(arrays['B_w']) and np.count_nonzero(arrays['D_yw'])
    assert np.count_nonzero(arrays['u_mean']) == np.count_nonzero(arrays['y_mean']) == 3
    assert len(set(arrays['u_std'])) == len(set(arrays['y_std'])) == 3

    actual_y, actual_end, actual_diag = production.predict(
        arrays, observed, context_u, future_u, iterations=iterations)
    actual_start, condition_diag = production.condition(
        arrays, observed, context_u, iterations=iterations)
    replay_y, replay_end, replay_start, replay_diag = independent.replay_request(
        arrays, observed, context_u, future_u, iterations=iterations)

    close_array(actual_y, replay_y)
    close_array(actual_end, replay_end)
    close_array(actual_start, replay_start)
    # Includes both seed/solved-state arrays, every direction/trial scalar,
    # line-search branch, rank, status and exact trajectory/Jacobian counts.
    close_diagnostics(actual_diag, replay_diag)
    close_diagnostics(condition_diag, replay_diag)
    assert actual_diag['policy'] == replay_diag['policy'] == independent.policy(iterations=iterations)
    own, replay = actual_diag['requests'][0], replay_diag['requests'][0]
    assert own['linear_rank'] == replay['linear_rank'] == 3
    assert own['accepted_steps'] > 0
    assert own['jacobian_evaluations'] == own['directions_considered']+1
    assert own['trajectory_evaluations'] == own['jacobian_evaluations']+own['trial_attempts']
    assert own['directions_considered'] <= iterations and own['trial_attempts'] <= iterations*8
    if nx == 28:
        # Unobservable extra coordinates remain the minimum-norm zero seed.
        np.testing.assert_array_equal(actual_diag['linear_seed_states'][:, 3:], 0.)
        np.testing.assert_array_equal(replay_diag['linear_seed_states'][:, 3:], 0.)

    saved_arrays, *saved_inputs = before
    for key, saved in saved_arrays.items():
        assert arrays[key].tobytes() == saved.tobytes()
    for actual, saved in zip((observed, context_u, future_u), saved_inputs, strict=True):
        assert actual.tobytes() == saved.tobytes()


@pytest.mark.parametrize('field,changed', [('status', 'STALLED'), ('directions_considered', 15),
                                        ('accepted', False), ('nonfinite', True)])
def test_comparison_never_tolerates_discrete_branch_or_work_disagreement(field, changed):
    expected = {'status': 'ITERATION_CAP', 'directions_considered': 16,
                'accepted': True, 'nonfinite': False}
    actual = {**expected, field: changed}
    with pytest.raises(AssertionError):
        close_diagnostics(actual, expected)


def test_diagnostic_arrays_use_stricter_context_tolerance():
    expected = {'solved_context_start_states': np.zeros((1, 3))}
    actual = {'solved_context_start_states': np.full((1, 3), 1e-9)}
    # This difference fits replay atol, but not the held context-array atol.
    close_array(actual['solved_context_start_states'], expected['solved_context_start_states'])
    with pytest.raises(AssertionError):
        close_diagnostics(actual, expected)
