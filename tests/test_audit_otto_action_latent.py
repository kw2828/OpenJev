"""Independent fabricated saved-output arithmetic and metadata barrier tests."""
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

from openjev.research import otto_action_latent_metrics as producer

ROOT = Path(__file__).resolve().parents[1]
name = 'otto_action_latent_common'
if name not in sys.modules:
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/otto_action_latent_common.py')
    common = importlib.util.module_from_spec(spec)
    sys.modules[name] = common
    spec.loader.exec_module(common)
spec = importlib.util.spec_from_file_location('_test_action_latent_auditor', ROOT / 'scripts/audit_otto_action_latent.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fabricated():
    rng = np.random.Generator(np.random.PCG64(42))
    outcomes = rng.integers(0, 4, (6, 8), dtype=np.int64)
    outcomes[1, 5:] = 4
    outcomes[2] = 4
    outcomes[5, 2:] = 4
    legal = np.ones((6, 8, 4), np.bool_)
    legal[outcomes == 4] = False
    legal[0, :, 0] = False
    dataset = {'outcomes': outcomes, 'raw_costs': rng.normal(size=(6, 8, 4)).astype(np.float32),
               'legal': legal, 'case_ids': np.array(['a', 'a', 'b', 'c', 'd', 'e']),
               'regimes': np.array(['lambda3'] * 3 + ['lambda4'] * 3)}
    return rng, dataset


@pytest.mark.parametrize('family', audit.FAMILIES)
@pytest.mark.parametrize('condition', audit.CONDITIONS)
def test_independent_complete_scalar_report_matches_producer(family, condition):
    rng, data = fabricated()
    before = {k: v.copy() for k, v in data.items()}
    callbacks = []
    for seed in (323000001, 323000002, 323000003):
        logits = rng.normal(size=(6, 8, 5)).astype(np.float32)
        values = logits
        kind = 'logits'
        if family == 'ridge':
            values = np.exp(logits.astype(np.float64))
            values /= values.sum(axis=-1, keepdims=True)
            kind = 'probabilities'
        costs = rng.normal(size=(6, 8, 4)).astype(np.float32)
        expected = producer.score(data['outcomes'], values, costs, data['raw_costs'], data['legal'],
            case_ids=data['case_ids'].tolist(), regimes=data['regimes'].tolist(), family=family,
            fit_seed=seed, condition=condition, prediction_kind=kind)
        actual = audit.scalar_report(np, data, values, costs, family=family, fit_seed=seed,
                                    condition=condition, check=lambda: callbacks.append(True))
        audit.close_equal(expected, actual)
        assert len(actual['per_horizon']) == 8 and set(actual['groups']) == {'all', 'short', 'long'}
        assert actual['chosen_actions'][2] == [-1] * 8
    assert len(callbacks) >= 3 * 6 * 8
    for key, value in before.items():
        np.testing.assert_array_equal(value, data[key])


def test_known_scores_ties_absorption_and_extreme_logits():
    _, data = fabricated()
    data['outcomes'][:] = 0
    data['legal'][:] = True
    data['legal'][:, :, 0] = False
    data['raw_costs'][:] = [0, 2, 1, 3]
    costs = np.zeros((6, 8, 4))
    costs[:, :, 0] = -100
    values = np.zeros((6, 8, 5))
    result = audit.scalar_report(np, data, values, costs, family='action_recurrent', fit_seed=1, condition='gap')
    leaf = result['groups']['all']['overall']
    assert leaf['case_weighted_log_score'] == pytest.approx(math.log(5))
    assert leaf['case_weighted_brier'] == pytest.approx(.8)
    assert leaf['case_weighted_decision_gap'] == 1
    assert result['chosen_actions'] == [[1] * 8] * 6
    values[:] = [0, 1000, -1000, 0, 0]
    result = audit.scalar_report(np, data, values, costs, family='action_recurrent', fit_seed=1, condition='gap')
    assert result['groups']['all']['overall']['case_weighted_log_score'] == 1000
    data['outcomes'][:] = 4
    data['legal'][:] = False
    result = audit.scalar_report(np, data, values, costs, family='action_recurrent', fit_seed=1, condition='gap')
    leaf = result['groups']['long']['overall']
    assert leaf['case_weighted_decision_gap'] is leaf['full_case_denominator_gap'] is None
    assert leaf['decision_rows'] == leaf['supported_cases'] == 0
    assert leaf['terminal_rows'] == leaf['outcome_rows'] == 24


@pytest.mark.parametrize('change', ['absorbing', 'masked_nan', 'legal', 'logits', 'shape', 'zero_probability'])
def test_independent_validation_rejects_bad_predictions_or_targets(change):
    _, data = fabricated()
    costs, values = np.zeros((6, 8, 4)), np.zeros((6, 8, 5))
    family = 'action_recurrent'
    if change == 'absorbing':
        data['outcomes'][2, 7] = 0
    elif change == 'masked_nan':
        costs[2, 0, 0] = np.nan
    elif change == 'legal':
        data['legal'][2, 0, 0] = True
    elif change == 'logits':
        values[0, 0, 0] = np.inf
    elif change == 'shape':
        values = values[:, :7]
    else:
        family = 'ridge'
        values[:] = [.25, .25, .25, .25, 0]
    with pytest.raises(ValueError):
        audit.scalar_report(np, data, values, costs, family=family, fit_seed=1, condition='gap')


def test_deadline_callback_and_saved_scalar_tolerance_fail_closed():
    _, data = fabricated()

    def stop():
        raise TimeoutError('fabricated deadline')

    with pytest.raises(TimeoutError, match='fabricated deadline'):
        audit.scalar_report(np, data, np.zeros((6, 8, 5)), np.zeros((6, 8, 4)),
                            family='action_recurrent', fit_seed=1, condition='gap', check=stop)
    audit.close_equal({'x': 1. + 1e-10}, {'x': 1.})
    for value in (1. + 1e-8, math.nan, math.inf, True):
        with pytest.raises(ValueError):
            audit.close_equal({'x': value}, {'x': 1.})
    with pytest.raises(ValueError):
        audit.close_equal({'extra': 0}, {})


def test_unclosed_evidence_rejects_before_any_array_decode(monkeypatch, tmp_path):
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


@pytest.mark.parametrize('change', ['phase', 'plan', 'test_calls', 'ordering', 'incomplete_fit'])
def test_predecessor_phase_plan_and_roster_checked_before_decode(monkeypatch, tmp_path, change):
    cfg = audit.c.CONFIG
    fit_files = {'started.json', 'fits.jsonl', 'training-orders.jsonl', 'normalization.json', 'ridge.npz', 'ridge-fit.json',
                 'predictions.npz', 'reports.json', 'summary.json'} | {
                     f'{family}-{seed}.npz' for family in audit.FAMILIES[:3] for seed in cfg['fit_seeds']}
    receipts = {phase: {'version': audit.c.VERSION, 'phase': phase, 'plan_sha256': 'fixed',
                       'old_test_decodes': 0, 'astra_calls': 0, 'files': {name: {} for name in files}}
                for phase, files in (('collect', {'dev.npz', 'summary.json'}), ('fit', fit_files))}
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
    else:
        receipts['fit']['files'].pop('predictions.npz')
    monkeypatch.setattr(audit.c, 'OUT', tmp_path)
    monkeypatch.setattr(audit.c, 'closed', lambda path, terminal: receipts['collect' if path.name == 'collection-01' else 'fit'])
    monkeypatch.setattr(audit.c, 'read', lambda path: terminals[path.name])
    run = SimpleNamespace(args=SimpleNamespace(plan_sha256='fixed'), launch={'started_ns': 30})
    with pytest.raises(ValueError):
        audit.admit(run)


@pytest.fixture
def training_trace(tmp_path):
    cfg = audit.c.CONFIG
    n = cfg['min_train']
    normalization = {'cost_scale': float(np.float32(.01)), 'rms_squared_before_floor': .0001,
                     'variance_floor': 1e-6, 'source': 'TRAIN only, all-four centered, equal cases and surviving rows',
                     'teacher_units_divisor': 64., 'dev_decodes': 0}
    fits, orders, files = [], [], {}
    for i, seed in enumerate(cfg['fit_seeds']):
        families = audit.FAMILIES[:3]
        for family in families[i:] + families[:i]:
            rng = np.random.Generator(np.random.PCG64(seed))
            orders.extend({'family': family, 'seed': seed, 'epoch': epoch, 'indices': rng.permutation(n).tolist()}
                          for epoch in range(cfg['epochs']))
            filename = f'{family}-{seed}.npz'
            files[filename] = {'sha256': filename, 'bytes': 1}
            count = 8107 if family == 'direct_horizon' else 8299
            fits.append({'family': family, 'seed': seed, 'epochs': cfg['epochs'],
                'updates': math.ceil(n / cfg['batch']) * cfg['epochs'], 'cases': n,
                'exposures': n * cfg['epochs'], 'training_horizons': [1, 2, 3, 4],
                'evaluation_decodes_so_far': 0, 'seconds': 1., 'last_batch_loss': .5, 'last_gradient_norm': .1,
                'checkpoint': files[filename], 'parameters': {'kind': family, 'hidden_dim': 28,
                    'count': count, 'trainable_count': count, 'cost_scale': normalization['cost_scale'],
                    'parameters': {'fabricated': {'count': count}}},
                'changed_tensors': ['fabricated']})
    ridge = {'fits': 1, 'parameters': 3024, 'training_horizons': [1, 2, 3, 4],
             'evaluation_decodes_so_far': 0, 'normal_equation_rows': 8 * n, 'seconds': .1}
    calls = {'optimizer_steps': sum(r['updates'] for r in fits), 'fit_count': 9, 'dev_array_decodes': 1}
    summary = {'fits': copy.deepcopy(fits), 'ridge_fit': ridge, 'calls': calls, 'data_cases': {'train': n}}
    receipt = {'files': files, 'calls': calls}
    (tmp_path / 'fits.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in fits))
    (tmp_path / 'training-orders.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in orders))
    (tmp_path / 'ridge-fit.json').write_text(json.dumps(ridge))
    (tmp_path / 'normalization.json').write_text(json.dumps(normalization))
    return tmp_path, receipt, {'counts': {'train': {'lambda3': n, 'lambda4': 0}}}, summary, orders


def test_recorded_training_orders_and_counts_are_complete(training_trace):
    directory, receipt, collection, summary, _ = training_trace
    counts = audit.verify_training(np, directory, receipt, collection, summary)
    assert counts['fits_checked'] == 9 and counts['ridge_fits_checked'] == 1
    assert counts['training_epochs_checked'] == 720
    assert counts['optimizer_steps_declared'] == receipt['calls']['optimizer_steps']
    assert counts['normalization_records_checked'] == 1


@pytest.mark.parametrize('change', ['scale', 'negative_variance', 'nonfinite', 'floor', 'source',
                                  'dev', 'units', 'fit_scale', 'changed_buffer', 'old_ridge_size'])
def test_normalization_metadata_and_final_ridge_dimensions_are_checked(training_trace, change):
    directory, receipt, collection, summary, _ = training_trace
    normalization = json.loads((directory / 'normalization.json').read_text())
    if change == 'scale':
        normalization['cost_scale'] *= 2
    elif change == 'negative_variance':
        normalization['rms_squared_before_floor'] = -.01
    elif change == 'nonfinite':
        normalization['rms_squared_before_floor'] = math.inf
    elif change == 'floor':
        normalization['variance_floor'] = 1e-5
    elif change == 'source':
        normalization['source'] = 'TRAIN and DEV'
    elif change == 'dev':
        normalization['dev_decodes'] = 1
    elif change == 'units':
        normalization['teacher_units_divisor'] = 1.
    elif change in ('fit_scale', 'changed_buffer'):
        fits = summary['fits']
        if change == 'fit_scale':
            fits[0]['parameters']['cost_scale'] = .02
        else:
            fits[0]['changed_tensors'].append('cost_scale')
        (directory / 'fits.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in fits))
    else:
        summary['ridge_fit']['parameters'] = 513
        (directory / 'ridge-fit.json').write_text(json.dumps(summary['ridge_fit']))
    (directory / 'normalization.json').write_text(json.dumps(normalization))
    with pytest.raises(ValueError):
        audit.verify_training(np, directory, receipt, collection, summary)


def test_normalization_floor_is_consistent_without_recomputing_train(training_trace):
    directory, receipt, collection, summary, _ = training_trace
    normalization = json.loads((directory / 'normalization.json').read_text())
    normalization.update(rms_squared_before_floor=0., cost_scale=float(np.float32(.001)))
    (directory / 'normalization.json').write_text(json.dumps(normalization))
    for fit in summary['fits']:
        fit['parameters']['cost_scale'] = normalization['cost_scale']
    (directory / 'fits.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in summary['fits']))
    assert audit.verify_training(np, directory, receipt, collection, summary)['normalization_records_checked'] == 1


@pytest.mark.parametrize('change', ['omitted_epoch', 'duplicate_case', 'wrong_order', 'early_dev', 'missing_fit'])
def test_incomplete_or_unmatched_training_trace_is_rejected(training_trace, change):
    directory, receipt, collection, summary, orders = training_trace
    if change == 'omitted_epoch':
        orders.pop()
    elif change == 'duplicate_case':
        orders[0]['indices'][0] = orders[0]['indices'][1]
    elif change == 'wrong_order':
        orders[0]['indices'].reverse()
    else:
        fits = copy.deepcopy(summary['fits'])
        if change == 'early_dev':
            fits[0]['evaluation_decodes_so_far'] = 1
        else:
            fits.pop()
        summary['fits'] = fits
        (directory / 'fits.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in fits))
    (directory / 'training-orders.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in orders))
    with pytest.raises(ValueError):
        audit.verify_training(np, directory, receipt, collection, summary)
