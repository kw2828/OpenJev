"""Pure arithmetic fixtures, no model, optimizer, empirical data, or study seed."""
from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finite_gap_solver_fixtures as module


def oracle(row, p):
    return module.scalar_oracle(row['xb'], row['xo'], row['yb'], row['yo'], p)


def check(row, p, claimed, **options):
    return module.check_certificate(row['xb'], row['xo'], row['yb'], row['yo'], p, claimed, **options)


def test_exact_registered_roster_geometry_dtypes_and_no_rng_mutation():
    rng_before = pickle.dumps(np.random.get_state())
    first, second = module.fixtures(), module.fixtures()
    assert pickle.dumps(np.random.get_state()) == rng_before
    assert tuple(row['name'] for row in first) == module.NAMES and len(first) == 18
    assert len(set(module.NAMES)) == 18
    expected = {'name', 'xb', 'xo', 'yb', 'yo', 'initial', 'known_optimum', 'description'}
    for left, right in zip(first, second, strict=True):
        assert set(left) == expected
        for key, shape in (('xb', (16, 2, 8)), ('xo', (16, 2, 8)), ('yb', (16, 2, 4)), ('yo', (16, 2, 4)), ('initial', (4, 8))):
            assert left[key].shape == shape and left[key].dtype == np.float64
            assert np.isfinite(left[key]).all()
            np.testing.assert_array_equal(left[key], right[key])
            assert not np.shares_memory(left[key], right[key])
        np.testing.assert_allclose(left['yb'].sum(-1), 0., atol=1e-15, rtol=0)
        np.testing.assert_allclose(left['yo'].sum(-1), 0., atol=1e-15, rtol=0)
    first[0]['xb'][0, 0, 0] = .7
    assert first[1]['xb'][0, 0, 0] == second[0]['xb'][0, 0, 0] == 1.


def test_state_geometries_have_declared_mass_correlations_and_rank_deficiency():
    rows = {row['name']: row for row in module.fixtures()}
    for geometry in module.GEOMETRIES:
        row = rows[f'{geometry}__interior__exact']
        np.testing.assert_allclose(row['xb'][:, 0].sum(-1), 1., atol=1e-15, rtol=0)
        np.testing.assert_allclose(row['xb'][:, 1].sum(-1), ([0., .001, .5] * 6)[:16], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(row['xo'], row['xb'][(np.arange(16) + 5) % 16])
    for geometry, epsilon in (('correlated95', .05), ('correlated999', .001)):
        expected = np.eye(8, dtype=np.float64) * epsilon
        expected[:, 0] += 1. - epsilon
        np.testing.assert_array_equal(rows[f'{geometry}__interior__exact']['xb'][:8, 0], expected)
    rank = rows['rank_deficient__boundary__exact']['xb']
    for i in range(0, 8, 2):
        np.testing.assert_array_equal(rank[:, :, i], rank[:, :, i + 1])


@pytest.mark.parametrize('name', [name for name in module.NAMES if name.endswith('__exact')])
def test_known_exact_optima_have_zero_loss_and_feasible_global_certificates(name):
    row = next(row for row in module.fixtures() if row['name'] == name)
    p = row['known_optimum']
    assert p is not None
    np.testing.assert_allclose(p.sum(0), 1., atol=0, rtol=0)
    result = oracle(row, p)
    assert result['objective'] <= 1e-28
    assert result['passed']
    initial = oracle(row, row['initial'])
    assert initial['objective'] > 1e-8 and initial['fw_gap'] > 1e-8
    # Convexity gives a dual-gap upper bound on known suboptimality.
    assert initial['objective'] - result['objective'] <= initial['fw_gap'] + 1e-12


def test_misspecified_cases_do_not_claim_generating_head_is_optimum():
    rows = [row for row in module.fixtures() if row['name'].endswith('__misspecified')]
    assert len(rows) == 8
    for row in rows:
        assert row['known_optimum'] is None
        for states, targets in ((row['xb'], row['yb']), (row['xo'], row['yo'])):
            zero_rows = states.sum(-1) == 0
            assert zero_rows.sum() == 6
            assert np.all(np.sum(targets[zero_rows] ** 2, axis=-1) > 0)
        assert oracle(row, row['initial'])['objective'] > 0


def test_zero_support_known_constant_objective_and_no_identification():
    row = next(row for row in module.fixtures() if row['name'] == 'zero_support')
    uniform = oracle(row, row['initial'])
    vertex = np.zeros((4, 8), np.float64)
    vertex[2] = 1.
    other = oracle(row, vertex)
    assert uniform['objective'] == other['objective'] == 1 / 64
    assert uniform['fw_gap'] == other['fw_gap'] == 0.
    assert uniform['passed'] and other['passed']
    np.testing.assert_array_equal(uniform['gradient'], np.zeros((4, 8)))
    assert not oracle(row, np.zeros((4, 8), np.float64))['passed']


def test_small_mass_is_absolute_scale_coverage_not_coefficient_recovery():
    row = next(row for row in module.fixtures() if row['name'] == 'small_mass')
    initial = oracle(row, row['initial'])
    optimal = oracle(row, row['known_optimum'])
    assert initial['passed'] and initial['objective'] > optimal['objective']
    assert np.max(np.abs(row['initial'] - row['known_optimum'])) == .375
    assert 'without coefficient recovery' in row['description']


def hand_problem():
    states = np.zeros((2, 1, 8), np.float64)
    states[0, 0, 0] = 1.
    targets = np.zeros((2, 1, 4), np.float64)
    targets[0, 0] = [-.75, .25, .25, .25]
    initial = np.full((4, 8), .25, np.float64)
    return {'xb': states, 'xo': states.copy(), 'yb': targets, 'yo': targets.copy(), 'initial': initial}


def test_hand_objective_gradient_gap_and_full_zero_row_denominator():
    row = hand_problem()
    result = oracle(row, row['initial'])
    assert result['objective'] == 3 / 16
    assert result['fw_gap'] == 3 / 8
    expected = np.zeros((4, 8), np.float64)
    expected[:, 0] = [-3 / 8, 1 / 8, 1 / 8, 1 / 8]
    np.testing.assert_array_equal(result['gradient'], expected)
    assert not result['passed']
    # Removing the zero row doubles the objective, proving it remains in N.
    sliced = {name: row[name][:1] for name in ('xb', 'xo', 'yb', 'yo')}
    assert oracle(sliced, row['initial'])['objective'] == 3 / 8


def test_scalar_gradient_matches_independent_directional_difference():
    row = next(row for row in module.fixtures() if row['name'] == 'correlated95__interior__misspecified')
    direction = np.arange(32, dtype=np.float64).reshape(4, 8) / 31 - .5
    p = row['initial']
    step = 1e-5
    difference = (oracle(row, p + step * direction)['objective'] - oracle(row, p - step * direction)['objective']) / (2 * step)
    gradient_dot = float(np.sum(oracle(row, p)['gradient'] * direction))
    assert difference == pytest.approx(gradient_dot, abs=1e-11, rel=1e-9)


def test_checker_retains_honest_failure_and_can_require_success():
    row = hand_problem()
    certificate = oracle(row, row['initial'])
    result = check(row, row['initial'], certificate)
    assert result['passed'] is False
    with pytest.raises(ValueError, match='did not pass'):
        check(row, row['initial'], certificate, require_pass=True)


@pytest.mark.parametrize('field', ['objective', 'gradient', 'fw_gap', 'simplex_violation', 'minimum_probability', 'maximum_probability', 'passed'])
def test_checker_rejects_perturbed_or_false_claims(field):
    row = hand_problem()
    claim = copy.deepcopy(oracle(row, row['initial']))
    if field == 'gradient':
        claim[field][0, 0] += .001
    elif field == 'passed':
        claim[field] = True
    else:
        claim[field] += .001
    with pytest.raises(ValueError, match='disagreement'):
        check(row, row['initial'], claim)


def test_checker_rejects_nonfinite_gradient_or_schema_and_never_mutates_inputs():
    row = hand_problem()
    before = {key: value.copy() for key, value in row.items()}
    claim = oracle(row, row['initial'])
    wrong = copy.deepcopy(claim)
    wrong['gradient'][0, 0] = np.nan
    with pytest.raises(ValueError, match='gradient'):
        check(row, row['initial'], wrong)
    with pytest.raises(ValueError, match='fields'):
        check(row, row['initial'], {**claim, 'extra': 0})
    check(row, row['initial'], claim)
    for key, value in before.items():
        np.testing.assert_array_equal(row[key], value)


@pytest.mark.parametrize('kind', ['negative_state', 'mass_above_one', 'uncentered_target', 'float32', 'nan_probability', 'empty'])
def test_oracle_fails_on_malformed_inputs(kind):
    row = hand_problem()
    if kind == 'negative_state':
        row['xb'][0, 0, 0] = -.1
    elif kind == 'mass_above_one':
        row['xb'][0, 0, 0] = 1.1
    elif kind == 'uncentered_target':
        row['yb'][0, 0, 0] += .1
    elif kind == 'float32':
        row['xo'] = row['xo'].astype(np.float32)
    elif kind == 'nan_probability':
        row['initial'][0, 0] = np.nan
    else:
        row['xb'] = row['xb'][:0]
    with pytest.raises(ValueError):
        oracle(row, row['initial'])


def test_nonincrease_is_separate_fixed_comparison_not_certificate_promotion():
    assert module.check_nonincrease({'objective': 1.}, {'objective': .9})
    assert module.check_nonincrease({'objective': 0.}, {'objective': .5e-12})
    assert not module.check_nonincrease({'objective': 0.}, {'objective': 2e-12})
    with pytest.raises(ValueError, match='finite'):
        module.check_nonincrease({'objective': np.inf}, {'objective': 0.})
