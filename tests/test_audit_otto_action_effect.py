"""Fabricated independent belief scores, oracle reconstruction and closure guards."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import otto_belief_distillation_metrics as producer

ROOT = Path(__file__).resolve().parents[1]
name = 'otto_action_effect_common'
if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/otto_action_effect_common.py')
    common = importlib.util.module_from_spec(spec)
    sys.modules[name] = common
    spec.loader.exec_module(common)
spec = importlib.util.spec_from_file_location('_test_action_effect_audit', ROOT / 'scripts/audit_otto_action_effect.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    """Two equiprobable static sources, with a spatially constant dyadic sensor."""
    n = 2
    sensor = np.broadcast_to([.5, .25, .125, .125], (105, 105, 4)).copy()
    sensor[52, 52] = 0
    tables = {name: sensor.copy() for name in ('lambda3', 'lambda4', 'lambda3_raw', 'lambda4_raw')}
    initial = np.zeros((n, 2809))
    initial[:, 26 * 53 + 28] = initial[:, 28 * 53 + 26] = .5
    outcomes = np.full((n, 8), 4, np.int64)
    outcomes[:, 0] = 0
    gap = np.zeros((n, 8, 5))
    found = np.array([0, .5, .5, .5, .5, 1, 1, 1])
    gap[:, :, :4] = (1 - found[None, :, None]) * [.5, .25, .125, .125]
    gap[:, :, 4] = found
    normal = gap.copy()
    normal[:, 2:] = [0, 0, 0, 0, 1]
    opposite = np.broadcast_to([.5, .25, .125, .125, 0], (n, 8, 5)).copy()
    data = {'initial_belief': initial, 'prefix_actions': np.tile([0, 1] * 4, (n, 1)).astype(np.int64),
        'prefix_outcomes': np.zeros((n, 8), np.int64), 'prefix_position': np.full((n, 2), 26, np.int64),
        'actions': np.tile([3, 3, 2, 2, 1, 1, 0, 0], (n, 1)).astype(np.int64),
        'outcomes': outcomes, 'gap_oracle': gap, 'normal_oracle': normal, 'opposite_oracle': opposite,
        'raw_costs': np.broadcast_to([2., 1., 3., 0.], (n, 8, 4)).copy(),
        'legal': np.broadcast_to((outcomes != 4)[..., None], (n, 8, 4)).copy(),
        'case_ids': np.array(['a', 'b']), 'regimes': np.array(['lambda3', 'lambda4'])}
    return data, tables


def predictions(data, condition):
    p = np.broadcast_to([.3, .2, .1, .1, .3], data['gap_oracle'].shape).copy()
    if condition == 'normal':
        for i in range(len(p)):
            first = np.flatnonzero(data['outcomes'][i] == 4)
            if len(first):
                p[i, first[0] + 1:] = [0, 0, 0, 0, 1]
    return p, np.zeros_like(data['raw_costs'])


@pytest.mark.parametrize('family', audit.FAMILIES)
@pytest.mark.parametrize('condition', audit.CONDITIONS)
def test_independent_full_scalar_oracle_report_matches_producer(family, condition):
    data, _ = fixture()
    p, costs = predictions(data, condition)
    before = {k: v.copy() for k, v in data.items()}
    callbacks = []
    expected = producer.score(data['outcomes'], p, costs, data['raw_costs'], data['legal'],
        oracle_probabilities=data[condition + '_oracle'], case_ids=data['case_ids'].tolist(),
        regimes=data['regimes'].tolist(), family=family, fit_seed=7, condition=condition, prediction_kind='probabilities')
    actual = audit.scalar_report(np, data, p, costs, family=family, fit_seed=7, condition=condition,
                                 check=lambda: callbacks.append(True))
    audit.close_equal(actual, expected)
    assert len(callbacks) >= 2 * 8
    assert actual['groups']['long']['overall']['case_weighted_decision_gap'] is None
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)


def test_exact_oracle_prediction_and_counted_roundoff_match():
    data, _ = fixture()
    p = data['normal_oracle'].copy()
    costs = np.zeros_like(data['raw_costs'])
    actual = audit.scalar_report(np, data, p, costs, family='effect_recurrent', fit_seed=1, condition='normal')
    expected = producer.score(data['outcomes'], p, costs, data['raw_costs'], data['legal'],
        oracle_probabilities=data['normal_oracle'], case_ids=data['case_ids'].tolist(), regimes=data['regimes'].tolist(),
        family='effect_recurrent', fit_seed=1, condition='normal', prediction_kind='probabilities')
    audit.close_equal(actual, expected)
    assert actual['oracle']['groups']['all']['all_rows']['overall']['case_weighted_brier_excess'] == 0


def test_independent_public_filter_checks_unique_visits_and_terminal_timing():
    data, tables = fixture()
    copies = {k: v.copy() for k, v in data.items()}
    result = audit.verify_oracles(np, data, tables)
    assert result['cases'] == 2 and result['prefix_updates'] == 16 and result['oracle_rows'] == 48
    assert result['cdf_rows'] == 22048 and result['native_calls'] == 0
    assert result['initial_prior_reconstructed'] is result['raw_sensor_functions_replayed'] is False
    assert data['gap_oracle'][0, 2, 4] == .5
    assert data['normal_oracle'][0, 1, 4] == .5 and data['normal_oracle'][0, 2, 4] == 1
    for key, value in copies.items():
        np.testing.assert_array_equal(data[key], value)


@pytest.mark.parametrize('change', ['cdf', 'zero_evidence', 'prefix_position', 'repeated_visit', 'terminal_early', 'opposite', 'nan'])
def test_oracle_corruption_fails_independent_reconstruction(change):
    data, tables = fixture()
    if change == 'cdf':
        tables['lambda3'][0, 0] = [.25, .5, .125, .125]
    elif change == 'zero_evidence':
        data['initial_belief'][:] = 0
        data['initial_belief'][:, 25 * 53 + 26] = 1
    elif change == 'prefix_position':
        data['prefix_position'][0, 0] += 1
    elif change == 'repeated_visit':
        data['gap_oracle'][0, 2] = [0, 0, 0, 0, 1]
    elif change == 'terminal_early':
        data['normal_oracle'][0, 1] = [0, 0, 0, 0, 1]
    elif change == 'opposite':
        data['opposite_oracle'][:] = data['gap_oracle']
    else:
        tables['lambda4_raw'][0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        audit.verify_oracles(np, data, tables)


def test_independent_signed_effect_all_cases_and_zero_signal_match_producer():
    data, _ = fixture()
    p, _ = predictions(data, 'gap')
    alt = p.copy()
    alt[0] = [.4, .2, .1, .1, .2]
    data['opposite_oracle'][1] = data['gap_oracle'][1]
    expected = producer.action_sensitivity(data['gap_oracle'], data['opposite_oracle'],
        case_ids=data['case_ids'].tolist(), regimes=data['regimes'].tolist(), actions=data['actions'],
        alternate_actions=data['actions'] ^ 1, predicted_original=p, predicted_alternate=alt)
    actual = audit.sensitivity_report(np, data, p, alt)
    audit.close_equal(actual, expected)
    assert actual['overall']['overall']['declared_cases'] == 2
    zero = actual['overall']['by_regime']['lambda4']
    assert zero['case_weighted_oracle_signal'] == zero['case_weighted_model_effect_error'] == 0


def test_invalid_probability_support_and_deadline_fail_closed():
    data, _ = fixture()
    p, cost = predictions(data, 'normal')
    p[0, 0] = [0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        audit.scalar_report(np, data, p, cost, family='effect_recurrent', fit_seed=1, condition='normal')

    def stop():
        raise TimeoutError('fabricated deadline')

    with pytest.raises(TimeoutError, match='fabricated deadline'):
        audit.verify_oracles(np, data, fixture()[1], check=stop)
    for value in (math.nan, math.inf, True, 1.000001):
        with pytest.raises(ValueError):
            audit.close_equal(value, 1.)


def test_unclosed_original_producer_rejects_before_array_decode(monkeypatch, tmp_path):
    run = SimpleNamespace(args=SimpleNamespace(plan_sha256='fabricated'), launch={'started_ns': 30})
    monkeypatch.setattr(audit.c, 'OUT', tmp_path)
    decoded = []
    monkeypatch.setattr(np, 'load', lambda *a, **kw: decoded.append(a))

    def unclosed(*args):
        raise ValueError('unclosed original producer')

    monkeypatch.setattr(audit.c, 'closed', unclosed)
    with pytest.raises(ValueError, match='unclosed original producer'):
        audit.run_audit(run)
    assert decoded == []


def training_fixture(tmp_path, monkeypatch):
    cfg = {**audit.c.CONFIG, 'min_train': 4, 'train_cases': 4}
    monkeypatch.setattr(audit.c, 'CONFIG', cfg)
    n, epochs = 4, cfg['epochs']
    normalization = {'cost_scale': float(np.float32(.001)), 'rms_squared_before_floor': 0.,
        'variance_floor': 1e-6, 'source': 'TRAIN only, all-four centered, equal cases and surviving rows',
        'teacher_units_divisor': 64., 'dev_decodes': 0, 'effect_variance': 1e-6,
        'effect_variance_before_floor': 0., 'effect_variance_floor': 1e-6,
        'effect_source': 'TRAIN only, mean squared signed oracle effect, equal cases and all four horizons'}
    (tmp_path / 'normalization.json').write_text(json.dumps(normalization))
    fits, orders, files = [], [], {}
    for i, seed in enumerate(cfg['fit_seeds']):
        for family in audit.FAMILIES[i:] + audit.FAMILIES[:i]:
            kind = audit.MODEL_KINDS[family]
            count = 8107 if kind == 'direct_horizon' else 8299
            checkpoint = {'sha256': 'a' * 64, 'bytes': 123}
            files[f'{family}-{seed}.npz'] = checkpoint
            fits.append({'family': family, 'seed': seed, 'epochs': epochs, 'updates': epochs,
                'cases': n, 'exposures': n * epochs, 'training_horizons': [1, 2, 3, 4],
                'evaluation_decodes_so_far': 0, 'model_kind': kind,
                'soft_targets': True, 'training_rollouts_per_batch': 3,
                'effect_weight': .1 if family == cfg['candidate'] else 0., 'blind_input_keys': ['prefix', 'prefix_lengths', 'actions'],
                'privileged_targets_only': ['raw_costs', 'gap_oracle', 'normal_oracle', 'opposite_oracle'], 'seconds': 1., 'last_batch_loss': .5, 'last_gradient_norm': .2,
                'checkpoint': checkpoint, 'changed_tensors': ['weight'], 'parameters': {'kind': kind,
                    'hidden_dim': 28, 'count': count, 'trainable_count': count,
                    'cost_scale': normalization['cost_scale'], 'parameters': {'weight': {'count': count}}}})
            rng = np.random.Generator(np.random.PCG64(seed))
            orders.extend({'family': family, 'seed': seed, 'epoch': epoch, 'indices': rng.permutation(n).tolist()}
                          for epoch in range(epochs))
    (tmp_path / 'fits.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in fits))
    (tmp_path / 'training-orders.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in orders))
    calls = {'optimizer_steps': 12 * epochs, 'fit_count': 12, 'dev_array_decodes': 1,
             'train_array_decodes': 1, 'training_rollouts': 36 * epochs}
    summary = {'fits': fits, 'calls': calls, 'data_cases': {'train': n}}
    return {'files': files, 'calls': calls}, {'cases': n, 'prior_dev_decodes': 0}, summary


def test_all_twelve_fit_records_and_960_epoch_permutations(tmp_path, monkeypatch):
    receipt, collection, summary = training_fixture(tmp_path, monkeypatch)
    result = audit.verify_training(np, tmp_path, receipt, collection, summary)
    assert result['fits_checked'] == 12 and result['training_epochs_checked'] == 960
    assert result['optimizer_steps_declared'] == 960 and result['training_case_exposures_declared'] == 3840
    changed = copy.deepcopy(summary)
    changed['fits'][0]['parameters']['kind'] = 'direct_horizon'
    with pytest.raises(ValueError):
        audit.verify_training(np, tmp_path, receipt, collection, changed)
    path = tmp_path / 'training-orders.jsonl'
    path.write_text(path.read_text() + '{}\n')
    with pytest.raises(ValueError, match='extra training orders'):
        audit.verify_training(np, tmp_path, receipt, collection, summary)


@pytest.mark.parametrize('change', ['phase', 'plan', 'test_calls', 'ordering', 'incomplete_fit', 'missing_sensor'])
def test_predecessor_phase_plan_and_roster_reject_before_decode(monkeypatch, tmp_path, change):
    fit_files = {'started.json', 'fits.jsonl', 'training-orders.jsonl', 'normalization.json',
                 'predictions.npz', 'reports.json', 'summary.json'} | {
                     f'{family}-{seed}.npz' for family in audit.FAMILIES for seed in audit.c.CONFIG['fit_seeds']}
    receipts = {phase: {'version': audit.c.VERSION, 'phase': phase, 'plan_sha256': 'fixed',
                       'old_test_decodes': 0, 'astra_calls': 0, 'files': {name: {} for name in files}}
                for phase, files in (('collect', {'dev.npz', 'sensor-laws.npz', 'summary.json', 'training-reference.json'}), ('fit', fit_files))}
    terminals = {'collection-native-01.terminal.json': {'started_ns': 0, 'finished_ns': 10},
                 'fit-native-01.terminal.json': {'started_ns': 11, 'finished_ns': 20}}
    if change == 'phase':
        receipts['fit']['phase'] = 'collect'
    elif change == 'plan':
        receipts['fit']['plan_sha256'] = 'other'
    elif change == 'test_calls':
        receipts['fit']['old_test_decodes'] = 1
    elif change == 'ordering':
        terminals['fit-native-01.terminal.json']['started_ns'] = 9
    elif change == 'missing_sensor':
        receipts['collect']['files'].pop('sensor-laws.npz')
    else:
        receipts['fit']['files'].pop('predictions.npz')
    monkeypatch.setattr(audit.c, 'OUT', tmp_path)
    monkeypatch.setattr(audit.c, 'closed', lambda path, terminal: receipts['collect' if path.name == 'collection-01' else 'fit'])
    monkeypatch.setattr(audit.c, 'read', lambda path: terminals[path.name])
    decoded = []
    monkeypatch.setattr(np, 'load', lambda *a, **kw: decoded.append(a))
    run = SimpleNamespace(args=SimpleNamespace(plan_sha256='fixed'), launch={'started_ns': 30})
    with pytest.raises(ValueError):
        audit.run_audit(run)
    assert decoded == []


def test_random_oracle_equal_prediction_retains_exact_roundoff_record():
    data, _ = fixture()
    data['outcomes'][:] = 0
    data['legal'][:] = True
    rng = np.random.Generator(np.random.PCG64(310))
    q = rng.uniform(.01, 1., size=(2, 8, 5))
    q /= q.sum(-1, keepdims=True)
    data['gap_oracle'] = q
    cost = np.zeros((2, 8, 4))
    actual = audit.scalar_report(np, data, q.copy(), cost, family='effect_recurrent', fit_seed=1, condition='gap')
    expected = producer.score(data['outcomes'], q.copy(), cost, data['raw_costs'], data['legal'],
        oracle_probabilities=q, case_ids=data['case_ids'].tolist(), regimes=data['regimes'].tolist(),
        family='effect_recurrent', fit_seed=1, condition='gap', prediction_kind='probabilities')
    audit.close_equal(actual, expected)


@pytest.mark.parametrize('change', [None, 'masked_feature_nan', 'outcome_float', 'missing', 'terminal_legal'])
def test_complete_dataset_contract_checks_unscored_arrays(change):
    data, _ = fixture()
    data['raw_costs'] = data['raw_costs'].astype(np.float32)
    data.update(prefix=np.zeros((2, 9, 31), np.float32), prefix_lengths=np.full(2, 9, np.int64),
                continuation=np.zeros((2, 8, 31), np.float32))
    if change == 'masked_feature_nan':
        data['continuation'][0, 7, 30] = np.nan
    elif change == 'outcome_float':
        data['outcomes'] = data['outcomes'].astype(np.float64)
    elif change == 'missing':
        data.pop('prefix')
    elif change == 'terminal_legal':
        data['legal'][0, 7, 0] = True
    if change is None:
        assert audit.validate_dataset(np, data) == 2
    else:
        with pytest.raises(ValueError):
            audit.validate_dataset(np, data)


def gate_fixture():
    """Reuse only fabricated inputs; the two implementations aggregate separately."""
    spec = importlib.util.spec_from_file_location('_fabricated_effect_inputs', ROOT / 'tests/test_otto_action_effect_metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports, sensitivity = module.fixture()
    return reports, sensitivity, module


@pytest.mark.parametrize('change', [None, 'seed_veto', 'zero_effect', 'support', 'normal_regression', 'undefined_gap'])
def test_independent_mechanism_gate_matches_registered_comparison(change):
    from openjev.research import otto_action_effect_metrics as effect

    reports, sensitivity, fixture_module = gate_fixture()
    options = copy.deepcopy(fixture_module.OPTIONS)
    if change == 'seed_veto':
        fixture_module.set_effect(sensitivity, 1., seed=3)
    elif change == 'zero_effect':
        for family in fixture_module.FAMILIES:
            fixture_module.set_effect(sensitivity, 0., family=family)
    elif change == 'support':
        options['thresholds']['minimum_supported_cases'] = 3
    elif change in ('normal_regression', 'undefined_gap'):
        for report in reports:
            if (change == 'undefined_gap' or report['family'] == fixture_module.FAMILIES[0]) and report['condition'] == ('normal' if change == 'normal_regression' else 'gap'):
                group = 'all' if change == 'normal_regression' else 'long'
                for leaf in report['groups'][group]['by_regime'].values():
                    leaf['case_weighted_decision_gap'] = 10. if change == 'normal_regression' else None
                    if change == 'undefined_gap':
                        leaf['supported_cases'] = leaf['decision_rows'] = 0
                        leaf['unsupported_case_ids'] = [row['case_id'] for row in leaf['cases']]
    before = copy.deepcopy((reports, sensitivity))
    actual = audit.mechanism_gate(reports, sensitivity, **options)
    expected = effect.evaluate_reports(reports, sensitivity, **options)
    audit.close_equal(actual, expected)
    assert actual['passed'] is (change is None)
    assert actual['total_cells'] == 18 and all(len(cell['conditions']) == 6 for cell in actual['cells'])
    assert (reports, sensitivity) == before


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'oracle', 'nan', 'rows'])
def test_independent_effect_gate_rejects_unpaired_reports(change):
    reports, sensitivity, fixture_module = gate_fixture()
    if change == 'missing':
        sensitivity.pop()
    elif change == 'duplicate':
        sensitivity[-1] = copy.deepcopy(sensitivity[0])
    else:
        leaf = sensitivity[0]['report']['per_horizon']['5']['by_regime']['lambda3']
        if change == 'oracle':
            leaf['cases'][0]['oracle_signal'] += .2
            leaf['case_weighted_oracle_signal'] += .1
        elif change == 'nan':
            leaf['cases'][0]['model_effect_error'] = math.nan
        else:
            leaf['cases'][0]['rows'] = 0
    with pytest.raises(ValueError):
        audit.mechanism_gate(reports, sensitivity, **fixture_module.OPTIONS)


@pytest.mark.parametrize('field,value', [('effect_variance', 0.), ('effect_variance', math.inf),
    ('effect_variance', .2), ('effect_variance_before_floor', -.1), ('effect_variance_floor', 0.),
    ('effect_source', 'DEV'), ('dev_decodes', 1)])
def test_effect_normalization_rejects_invalid_or_non_train_records(tmp_path, monkeypatch, field, value):
    receipt, reference, summary = training_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'normalization.json'
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit.verify_training(np, tmp_path, receipt, reference, summary)


@pytest.mark.parametrize('field,value', [('training_rollouts_per_batch', 2), ('effect_weight', 0.),
    ('soft_targets', False), ('privileged_targets_only', ['raw_costs', 'gap_oracle', 'normal_oracle'])])
def test_paired_training_record_preserves_registered_mechanism(tmp_path, monkeypatch, field, value):
    receipt, reference, summary = training_fixture(tmp_path, monkeypatch)
    summary['fits'][0][field] = value
    (tmp_path / 'fits.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in summary['fits']))
    with pytest.raises(ValueError):
        audit.verify_training(np, tmp_path, receipt, reference, summary)
