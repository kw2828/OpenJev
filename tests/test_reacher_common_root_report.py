"""Actual report functions on synthetic scalars only; no artifact/backend import."""

import ast
import copy
import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

REPORT = Path(__file__).resolve().parents[1]/'output/reacher-common-root-diagnostic-v1/report_results.py'
ROLES = ('nominal', 'public_gain', 'true_state')
RULE = {'pooled_cost_reduction_fraction_min': .03,
        'pooled_cost_reduction_per_action_min': .001,
        'all_three_source_trajectory_means_nonworse': True,
        'nonworse_slot_means_min': 8, 'slot_count': 12}


@pytest.fixture
def useful():
    tree = ast.parse(REPORT.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'useful')
    namespace = {'np': np, 'math': math, 'Fraction': Fraction}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(REPORT), 'exec'), namespace)  # noqa: S102 - actual pure function, no backend imports.
    return namespace['useful']


def rows(baseline=1., candidate=.95):
    return [{'role': role, 'step': t,
             'contrasts': {name: {'baseline': baseline, 'candidate': candidate}
                           for name in ('search', 'horizon', 'gain')}}
            for role in ROLES for t in (0, 50, 100, 150)]


@pytest.mark.parametrize('contrast', ('search', 'horizon', 'gain'))
@pytest.mark.parametrize(('baseline', 'candidate'), ((1., .97), (.04, .0388), (.02, .019)))
def test_inclusive_relative_and_absolute_boundaries(useful, contrast, baseline, candidate):
    result = useful(rows(baseline, candidate), contrast, RULE)
    assert result['pass'] and all(result['checks'].values())
    assert result['nonworse_slots'] == 12
    assert result['baseline_mean'] == pytest.approx(baseline)
    assert result['candidate_mean'] == pytest.approx(candidate)
    assert result['improvement_percent'] == pytest.approx(100*(baseline-candidate)/baseline)


@pytest.mark.parametrize(('baseline', 'candidate', 'failed'), (
    (1., np.nextafter(.97, math.inf), 'relative'),
    (.02, np.nextafter(.019, math.inf), 'absolute'),
))
def test_one_float_step_below_margin_is_not_rounded_into_pass(useful, baseline, candidate, failed):
    result = useful(rows(baseline, float(candidate)), 'search', RULE)
    assert not result['pass'] and not result['checks'][failed]
    assert result['checks']['trajectory_means'] and result['checks']['nonworse_slots']


def test_each_trajectory_must_be_nonworse_even_when_pooled_margin_and_eight_slots_pass(useful):
    values = rows()
    for value in values:
        value['contrasts']['horizon']['candidate'] = 1.01 if value['role'] == 'nominal' else .9
    result = useful(values, 'horizon', RULE)
    assert result['checks'] == {'relative': True, 'absolute': True,
                                'trajectory_means': False, 'nonworse_slots': True}
    assert result['nonworse_slots'] == 8 and not result['pass']


@pytest.mark.parametrize(('better_per_role', 'passes'), (((3, 3, 2), True), ((2, 2, 3), False)))
def test_exact_eight_nonworse_slots_required_independent_of_pooled_improvement(useful, better_per_role, passes):
    values = rows()
    for i, value in enumerate(values):
        value['contrasts']['gain']['candidate'] = .7 if i % 4 < better_per_role[i//4] else 1.01
    result = useful(values, 'gain', RULE)
    assert result['checks']['relative'] and result['checks']['absolute'] and result['checks']['trajectory_means']
    assert result['nonworse_slots'] == sum(better_per_role)
    assert result['checks']['nonworse_slots'] is passes and result['pass'] is passes


def test_slot_equality_is_nonworse_and_inputs_are_untouched(useful):
    values = rows(candidate=1.)
    before = copy.deepcopy(values)
    result = useful(values, 'search', RULE)
    assert result['nonworse_slots'] == 12 and result['checks']['trajectory_means']
    assert not result['checks']['relative'] and not result['checks']['absolute']
    assert values == before


def test_contrasts_do_not_borrow_each_others_success(useful):
    values = rows(candidate=.9)
    for value in values: value['contrasts']['horizon']['candidate'] = 1.01
    assert useful(values, 'search', RULE)['pass']
    assert useful(values, 'gain', RULE)['pass']
    assert not useful(values, 'horizon', RULE)['pass']


@pytest.mark.parametrize('kind', ('missing', 'extra', 'uneven_roles', 'unknown_role',
                                 'baseline_nan', 'baseline_inf', 'baseline_zero', 'baseline_negative',
                                 'candidate_nan', 'candidate_inf', 'candidate_negative'))
def test_malformed_coverage_or_nonfinite_costs_rejected(useful, kind):
    values = rows()
    if kind == 'missing': values.pop()
    elif kind == 'extra': values.append(copy.deepcopy(values[0]))
    elif kind == 'uneven_roles': values[0]['role'] = 'public_gain'
    elif kind == 'unknown_role': values[0]['role'] = 'unknown'
    else:
        field, value = kind.split('_')
        values[0]['contrasts']['search'][field] = {
            'nan': math.nan, 'inf': math.inf, 'zero': 0., 'negative': -.01}[value]
    with pytest.raises((AssertionError, ValueError)):
        useful(values, 'search', RULE)


@pytest.mark.parametrize(('search', 'horizon', 'gain'), list(itertools.product((False, True), repeat=3)))
def test_actual_frozen_decision_tree_requires_gain_and_prefers_horizon(search, horizon, gain):
    tree = ast.parse(REPORT.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    decision = next(node for node in main.body if isinstance(node, ast.If)
                    and any(isinstance(child, ast.Name) and child.id == 'decision'
                            and isinstance(child.ctx, ast.Store) for child in ast.walk(node)))
    namespace = {'checks': {key: {'pass': value} for key, value in
                            (('search', search), ('horizon', horizon), ('gain', gain))}}
    exec(compile(ast.Module(body=[decision], type_ignores=[]), str(REPORT), 'exec'), namespace)  # noqa: S102 - actual scalar-only decision tree.
    result = namespace['decision']
    if gain and horizon:
        assert result.startswith('Prepare only a longer-horizon')
    elif gain and search:
        assert result.startswith('Prepare only the extra-short-search')
    else:
        assert result.startswith('Stop this task/controller recipe')
    assert 'launch' not in result or 'no fresh scientific launch' in result
