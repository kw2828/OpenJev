"""Fabricated independent scalar, probability, forward and closure checks."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_otto_cost_information as audit
import run_otto_cost_information as producer


def scalar_fixture():
    n, m = 4, audit.c.CONFIG['mc_draws']
    root = np.zeros((n, 2809), np.float64)
    root[:, 0] = 1.
    data = {'case_ids': np.array(['a', 'b', 'c', 'd']), 'regimes': np.array(['lambda3'] * 2 + ['lambda4'] * 2),
        'prefix_position': np.full((n, 2), 26, np.int64), 'root_grid': root,
        'actions': np.tile(np.array([0, 1] * 4, np.int64), (n, 1)),
        'exact_weights': np.zeros((n, 256), np.float64), 'exact_costs': np.zeros((n, 256, 4), np.float32),
        'mc_costs4': np.tile(np.array([1., 3., 5., 7.], np.float32), (n, 2, m, 1)),
        'mc_costs8': np.tile(np.array([1., 3., 5., 7.], np.float32), (n, 2, m, 1)),
        'mc_alive8': np.ones((n, 2, m), np.bool_)}
    data['exact_weights'][:, 0] = [1., .5, 1., .5]
    data['exact_costs'][:, 0] = [1., 3., 5., 7.]
    predictions = {}
    for family in audit.c.FAMILIES:
        for j, seed in enumerate(audit.c.FIT_SEEDS):
            value = np.ones((n, 8, 4), np.float32)
            value[:, :, j + 1] = 0.
            predictions[f'{family}__{seed}__cost'] = value
    return data, predictions


@pytest.fixture
def small_bootstrap(monkeypatch):
    monkeypatch.setitem(audit.c.CONFIG, 'bootstrap_replicates', 23)
    monkeypatch.setitem(audit.c.CONFIG, 'min_cases', 2)
    monkeypatch.setitem(audit.c.CONFIG, 'min_mc_survivors', 1)


@pytest.mark.parametrize('variant', ['positive', 'evaluation_reversal', 'zero_support', 'heterogeneous'])
def test_independent_bootstrap_and_full_scalar_report_parity(small_bootstrap, variant):
    data, predictions = scalar_fixture()
    if variant == 'evaluation_reversal':
        data['mc_costs8'][:, 1] = [8., 1., 1., 1.]
    elif variant == 'zero_support':
        data['mc_costs8'][[0, 2]] = 0.
        data['mc_alive8'][[0, 2]] = False
        data['exact_weights'][[0, 2]] = 0.
        data['exact_costs'][[0, 2]] = 0.
    elif variant == 'heterogeneous':
        data['mc_costs8'][0, 1, :64] = [4., 0., 8., 2.]
    before = {key: value.copy() for key, value in data.items()}
    actual = audit.analyze(data, predictions, np)
    expected = producer.analyze(data, predictions, np)
    audit.compare_report(expected, actual)
    assert len(actual['rows']) == 48 and len(actual['decompositions']) == 24
    assert len(actual['h4_sampling_validation']) == 8
    assert all(np.array_equal(value, before[key]) for key, value in data.items())
    if variant == 'positive':
        assert actual['gate']['passed'] is True
        assert all(row['control_gap'] == 4. and row['gain'] == 4. for row in actual['gate']['groups'])
    elif variant == 'evaluation_reversal':
        assert actual['gate']['passed'] is False
        assert all(row['gain'] == -7. for row in actual['gate']['groups'])
        assert all(row['reference_action'] == 0 for row in actual['gate']['cases'])
    elif variant == 'zero_support':
        assert all(row['control_gap'] == 2. and row['cases'] == 2 for row in actual['gate']['groups'])
        assert all(not row['conditions']['mc_support'] for row in actual['gate']['groups'])


def test_mass_multiplied_per_case_not_after_conditional_aggregation(small_bootstrap):
    data, predictions = scalar_fixture()
    data['exact_costs'][:, 0] = [[0, 4, 4, 4], [0, 8, 8, 8]] * 2
    result = audit.analyze(data, predictions, np)
    assert all(row['total_regret'] == 4. for row in result['rows'] if row['horizon'] == 4)
    assert all(row['information_advantage'] == 0. for row in result['rows'] if row['horizon'] == 4)


def test_exact_survival_removes_unique_visited_cells_once():
    data, _ = scalar_fixture()
    data['root_grid'][:] = 0.
    data['root_grid'][:, 25 * 53 + 26] = .25
    data['root_grid'][:, 26 * 53 + 26] = .5
    data['root_grid'][:, 0] = .25
    np.testing.assert_array_equal(audit.exact_survival(data, np, 8), np.full(4, .25))


@pytest.mark.parametrize('change', ['missing', 'dtype', 'nan', 'shape'])
def test_saved_prediction_contract(change):
    data, predictions = scalar_fixture()
    key = next(iter(predictions))
    if change == 'missing':
        predictions.pop(key)
    elif change == 'dtype':
        predictions[key] = predictions[key].astype(np.float64)
    elif change == 'nan':
        predictions[key][0, 0, 0] = np.nan
    else:
        predictions[key] = predictions[key][:, :4]
    with pytest.raises(ValueError):
        audit.primary_gate(data, predictions, np)


def test_scalar_deadline_callback_can_stop_bootstrap(small_bootstrap):
    data, predictions = scalar_fixture()
    count = 0

    def stop():
        nonlocal count
        count += 1
        if count == 9:
            raise TimeoutError('fabricated audit deadline')

    with pytest.raises(TimeoutError):
        audit.primary_gate(data, predictions, np, check=stop)
    assert count == 9


def physical_fixture():
    """Uniform odor law and dense public belief, no native/model computation."""
    from openjev.research import otto_conditional_cost_tree as tree
    from openjev.research import otto_predictive_belief as bayes

    sensor = np.full((105, 105, 4), .25, np.float64)
    sensor[52, 52] = 0.
    kernel = np.full((4, 107, 107), .25, np.float64)
    kernel[:, 53, 53] = 0.
    tables = {name: value.copy() for regime in ('lambda3', 'lambda4')
              for name, value in ((regime, sensor), (regime + '_raw', sensor), ('legacy_' + regime, kernel))}
    rows, roster = [], []
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    for i, regime in enumerate(('lambda3', 'lambda4')):
        identity = {'id': f'fabricated:{i}', 'regime': regime, 'seed': 710 + i, 'initial_hit': 1,
                    'select_seed': 720 + i, 'eval_seed': 730 + i}
        roster.append(identity)
        initial_legacy = np.full(2809, 1. / 2808)
        initial_legacy[26 * 53 + 26] = 0.
        initial_legacy *= .25
        initial_legacy /= initial_legacy.sum()
        initial = bayes.normalize_prior(bayes.cdf_law(initial_legacy)['probabilities'])['belief']
        strict, legacy = initial.copy(), initial_legacy.copy()
        prefix_actions = np.array([0, 1] * 4, np.int64)
        for position in ((25, 26), (26, 26)) * 4:
            index = position[0] * 53 + position[1]
            strict[index] = legacy[index] = 0.
            strict *= .25
            legacy *= .25
            strict /= strict.sum()
            legacy /= legacy.sum()
        grid = bayes.cdf_law(strict)['probabilities']
        actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([identity['seed'], 911]))).integers(0, 4, size=8, dtype=np.int64)
        positions = []
        point = [26, 26]
        for a in actions:
            point[a // 2] += 2 * (int(a) % 2) - 1
            positions.append(point.copy())
        indices = np.asarray([x * 53 + y for x, y in positions], np.int64)
        laws = np.stack([sensor[(coordinates - position + 52)[:, 0], (coordinates - position + 52)[:, 1]] for position in positions])

        def update(state, index, _odor):
            state[index] = 0.
            state *= .25
            mass = state.sum()
            if mass > 1e-10:
                state /= mass
            return state

        def score(_state, _position):
            return np.array([1., 2., 3., 4.], np.float32)

        exact = tree.exact_h4(grid, legacy, laws[:4], indices[:4], update, score)
        draws, streams = [], []
        for key in ('select_seed', 'eval_seed'):
            integers = np.random.PCG64(identity[key]).random_raw((128, 9)) >> np.uint64(11)
            draws.append(integers)
            streams.append(tree.sample_h8(grid, legacy, laws, indices, integers, exact, update, score))
        row = {'prefix': np.zeros((9, 31), np.float32), 'prefix_lengths': np.int64(9),
            'prefix_actions': prefix_actions, 'prefix_outcomes': np.zeros(8, np.int64),
            'prefix_position': np.array([26, 26], np.int64), 'actions': actions,
            'initial_belief': initial, 'root_strict': strict, 'root_grid': grid, 'legacy_root': legacy,
            'exact_weights': exact['weights'], 'exact_costs': exact['costs'], 'exact_alive': exact['supported'],
            'mc_draws': np.stack(draws)}
        row.update({'mc_' + key: np.stack([value[key] for value in streams])
                    for key in ('source_indices', 'outcomes', 'alive4', 'alive8', 'costs4', 'costs8')})
        rows.append(row)
    data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    data.update(case_ids=np.array([row['id'] for row in roster]), regimes=np.array([row['regime'] for row in roster]))
    return data, tables, roster


@pytest.fixture(scope='module')
def physical():
    return physical_fixture()


def test_independent_prefix_tree_integer_draws_and_exact_lookup(physical):
    data, tables, roster = physical
    before = {key: value.copy() for key, value in data.items()}
    report = audit.verify_probabilities(data, tables, roster, np)
    assert report['counts']['prefixes'] == 2
    assert report['counts']['exact_positive_leaves'] == 512
    assert report['counts']['source_draws'] == 512
    assert report['counts']['replayed_uniform_integers'] == 4608
    assert report['counts']['teacher_costs_recomputed'] is False
    assert all(np.array_equal(value, before[key]) for key, value in data.items())


@pytest.mark.parametrize('change', ['raw_sensor', 'grid', 'legacy', 'position', 'weight', 'draw', 'source', 'odor', 'lookup'])
def test_independent_probability_corruptions_fail_closed(physical, change):
    original, original_tables, roster = physical
    data, tables = {k: v.copy() for k, v in original.items()}, {k: v.copy() for k, v in original_tables.items()}
    if change == 'raw_sensor':
        tables['lambda3_raw'][0, 0] = [.5, .25, .125, .125]
    elif change == 'grid':
        data['root_grid'][0, 0] += 1e-5
    elif change == 'legacy':
        data['legacy_root'][0, 0] += .01
    elif change == 'position':
        data['prefix_position'][0, 0] += 1
    elif change == 'weight':
        data['exact_weights'][0, 0] *= .5
    elif change == 'draw':
        data['mc_draws'][0, 0, 0, 1] ^= np.uint64(1)
    elif change == 'source':
        data['mc_source_indices'][0, 0, 0] = (data['mc_source_indices'][0, 0, 0] + 1) % 2809
    elif change == 'odor':
        data['mc_outcomes'][0, 0, 0, 0] = (data['mc_outcomes'][0, 0, 0, 0] + 1) % 4
    else:
        sample = int(np.flatnonzero(data['mc_alive4'][0, 0])[0])
        data['mc_costs4'][0, 0, sample, 0] += 1
    with pytest.raises(ValueError):
        audit.verify_probabilities(data, tables, roster, np)


def test_cdf_boundary_and_zero_category_are_not_renormalized_twice():
    law = audit._cdf_law(np.array([.5, 0., .25, .25]), np)
    assert audit._integer_cdf(law) == [2**52, 2**52, 3 * 2**51, 2**53]
    with pytest.raises(ValueError):
        audit._integer_cdf([.1, .2, .3, .4])


def test_probability_verifier_honors_deadline_before_large_tree(physical):
    data, tables, roster = physical
    with pytest.raises(TimeoutError):
        audit.verify_probabilities(data, tables, roster, np,
            check=lambda: (_ for _ in ()).throw(TimeoutError('fabricated deadline')))


def forward_fixture():
    data = {'case_ids': np.array(['a']), 'exact_alive': np.zeros((1, 256), bool),
            'exact_costs': np.zeros((1, 256, 4), np.float32),
            'mc_alive8': np.zeros((1, 2, 128), bool), 'mc_costs8': np.zeros((1, 2, 128, 4), np.float32),
            'mc_outcomes': np.zeros((1, 2, 128, 8), np.int64)}
    data['exact_alive'][0, 1] = True
    data['exact_costs'][0, 1] = [2, 3, 4, 5]
    data['mc_alive8'][0, 1, 7] = True
    data['mc_costs8'][0, 1, 7] = [2, 3, 4, 5]
    events = []
    for ordinal, stream, mode, horizon, history, sample in ((1, 'exact', 'exact', 4, [0, 0, 0, 1], None),
                                                          (2, 'evaluate', 'mc', 8, [0] * 8, 7)):
        context = {'id': 'a', 'phase': 'counterfactual', 'stream': stream, 'mode': mode, 'horizon': horizon, 'history': history}
        if sample is not None:
            context['sample'] = sample
        events.append({'ordinal': ordinal, 'context': context, 'input_shape': [16, 105, 105], 'symmetry_average': True,
                       'branch_masses': [[.25] * 4] * 4, 'values': [1.] * 4 + [2.] * 4 + [3.] * 4 + [4.] * 4})
    return data, events


def test_forward_values_authenticate_cost_arithmetic_without_teacher():
    data, events = forward_fixture()
    result = audit.verify_forward_records(events, data, np)
    assert result['forward_q_reductions_checked'] == 2 and result['teacher_recomputed'] is False
    assert result['maximum_float32_reduction_difference'] == 0.


@pytest.mark.parametrize('change', ['missing', 'extra', 'context', 'cost', 'nonfinite'])
def test_forward_teacher_cost_join_fails_closed(change):
    data, events = forward_fixture()
    if change == 'missing':
        events.pop()
    elif change == 'extra':
        events.append(copy.deepcopy(events[-1]))
    elif change == 'context':
        events[0]['context']['history'][0] = 1
    elif change == 'cost':
        data['exact_costs'][0, 1, 0] += 1
    else:
        events[0]['values'][0] = math.nan
    with pytest.raises(ValueError):
        audit.verify_forward_records(events, data, np)


def test_physical_work_nested_forward_and_return_accounting():
    first = {'event': 'attempt', 'id': 1, 'channel': 'teacher_score', 'context': {'case': 'a'}}
    second = {'event': 'attempt', 'id': 2, 'channel': 'tensorflow_value', 'context': {'case': 'a'}}
    events = [first, second, {**second, 'event': 'return', 'seconds_including_nested_io': .1},
              {**first, 'event': 'return', 'seconds_including_nested_io': .2}]
    calls = {name: {'attempted': 1, 'returned': 1} for name in ('teacher_score', 'tensorflow_value')}
    assert audit.verify_work(events, calls) == {'work_events_checked': 4, 'work_operations_checked': 2}
    with pytest.raises(ValueError):
        audit.verify_work(events[:-1], calls)
    with pytest.raises(ValueError):
        audit.verify_work([second], calls)


def test_unclosed_predecessor_blocks_before_any_array_decode(monkeypatch, tmp_path):
    run = SimpleNamespace(args=SimpleNamespace(plan_sha256='fixed'), launch={'started_ns': 20})
    monkeypatch.setattr(audit.c, 'OUT', tmp_path)
    monkeypatch.setattr(audit.c, 'closed', lambda *_: (_ for _ in ()).throw(ValueError('unclosed original')))
    decoded = []
    monkeypatch.setattr(np, 'load', lambda *args, **kw: decoded.append(args))
    with pytest.raises(ValueError, match='unclosed original'):
        audit.run_audit(run)
    assert decoded == []


def test_subset_comparison_still_rejects_nonfinite_or_missing_scalars():
    with pytest.raises(ValueError):
        audit.compare_report({'value': math.nan}, {'value': 1.})
    with pytest.raises(ValueError):
        audit.compare_report({}, {'value': 1.})
    audit.compare_report({'value': 1., 'roundoff_records': []}, {'value': 1.})
