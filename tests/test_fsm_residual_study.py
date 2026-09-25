"""Fabricated failure, causality and selection checks for the residual campaign."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from openjev.research import fsm_data, fsm_linear
from openjev.research import fsm_residual_study as study


def config():
    return {'context': 100, 'horizon': 128, 'order': 32, 'hidden_width': 24, 'batch': 2,
            'updates': 2, 'gradient_clip': 1., 'fit_timeout_seconds': 30,
            'evaluation_stride': 256, 'identity_atol': 1e-9, 'identity_rtol': 1e-9}


def fixture():
    n, c, r, p = np.indices((240, 3, 6, 2))
    data = fsm_data.records_from_fixture({
        f'{key}_{amp}_train': np.asarray(np.sin(n*.071+c*.13+r*.79+p*.33)
                                         +(.2 if key == 'y' else 0), dtype=np.float64)
        for amp in fsm_data.AMPLITUDES for key in ('u', 'y')})
    norm = fsm_data.fit_normalizer(data.partition('fit'))
    records = study.normalized_records(data.partition('fit'), norm)
    y, u = (torch.tensor(np.stack([getattr(r, k) for r in records]), dtype=torch.float64) for k in ('y', 'u'))
    coefficients = np.zeros((3, 196), dtype=np.float64)
    coefficients[:, 93:96] = np.eye(3)*.6
    coefficients[:, 96:99] = np.eye(3)*.3
    backbone = fsm_linear.VARXModel(coefficients, 32, 1e-6, 12*(240-32), fsm_linear.FIT_IDS)
    schedule = study.sampling_schedule(9201, updates=2, batch=2, records=12,
                                      samples=240, context=100, horizon=128)
    return data, norm, y, u, backbone, schedule


@pytest.mark.parametrize('arch', study.ARCHITECTURES)
def test_fit_checkpoint_causal_request_and_complete_evaluation(arch, tmp_path):
    data, norm, y, u, backbone, schedule = fixture(); cfg = config()
    model, receipt = study.fit(arch+'-lr0.0001', 9201, backbone, y, u, schedule, cfg, tmp_path/'fit')
    assert receipt['status'] == 'complete' and receipt['accepted_updates'] == 2
    assert receipt['initial_identity']['passed'] and receipt['backbone_unchanged']
    assert all(p.grad is None for p in model.parameters())
    saved = torch.load(tmp_path/'fit/final.pt', weights_only=True)
    assert saved['accepted_updates'] == 2
    assert all(int(v['step']) == 2 for v in saved['optimizer']['state'].values())
    record = data.partition('dev')[0]
    result = study.full_request(model, record, 0, cfg, norm)
    future_changed = record.y.copy(); future_changed[100:] = 1e10
    other = study.full_request(model, replace(record, y=future_changed), 0, cfg, norm)
    np.testing.assert_array_equal(result, other)
    evaluation = study.evaluate(model, receipt, data.partition('dev'), cfg, norm, tmp_path/'evaluation')
    assert len(evaluation['rows']) == 12 and all(r['status'] == 'complete' for r in evaluation['rows'])
    assert len(evaluation['request_ms']) == 24 and evaluation['timing_error'] is None
    assert evaluation['persistent_numeric_bytes'] == model.model_spec()['persistent_numeric_bytes_per_stream']+96
    raw_error = result[0]-record.y[100:228]
    np.testing.assert_allclose(evaluation['rows'][0]['native_output_per_channel_rmse'],
                               np.sqrt(np.mean(raw_error**2, axis=0)), rtol=1e-12, atol=1e-12)


def test_failed_nonfinite_fit_still_saves_full_receipt_and_partial_checkpoint(tmp_path, monkeypatch):
    _, _, y, u, backbone, schedule = fixture()
    def damage(optimizer, *args, **kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].flatten()[0] = float('nan')
    monkeypatch.setattr(torch.optim.Adam, 'step', damage)
    model, receipt = study.fit('tanh_feedback-lr0.0001', 9201, backbone, y, u, schedule,
                               config(), tmp_path/'failed')
    assert receipt['status'] == 'failed' and receipt['accepted_updates'] == 0
    assert receipt['error']['message'] == 'nonfinite residual after optimizer update'
    assert receipt['model_spec']['parameter_count'] == 4779
    assert (tmp_path/'failed/receipt.json').exists() and (tmp_path/'failed/final.pt').exists()
    assert all(p.grad is None for p in model.parameters())
    e = study.skipped_evaluation(receipt, tmp_path/'skipped')
    assert len(e['rows']) == 12 and all(r['status'] == 'not_run' for r in e['rows'])


def test_timeout_before_first_update_is_retained(tmp_path):
    _, _, y, u, backbone, schedule = fixture(); cfg = config(); cfg['fit_timeout_seconds'] = -1
    _, receipt = study.fit('affine_feedback-lr0.0001', 9201, backbone, y, u, schedule, cfg, tmp_path/'capped')
    assert receipt['status'] == 'failed' and receipt['error']['type'] == 'TimeoutError'
    assert receipt['accepted_updates'] == 0 and receipt['initial_identity']['passed']


def test_initial_identity_failure_stops_before_any_update(tmp_path, monkeypatch):
    _, _, y, u, backbone, schedule = fixture()
    monkeypatch.setattr(study, 'initial_identity', lambda *a, **k: {'passed': False})
    _, receipt = study.fit('affine_feedback-lr0.0001', 9201, backbone, y, u, schedule, config(), tmp_path/'identity')
    assert receipt['status'] == 'failed' and receipt['accepted_updates'] == 0
    assert receipt['error']['message'] == 'initial forecast differs from frozen backbone'


def results():
    fits, evaluations = [], []
    for family in study.RECIPES:
        arch, rate = study.RECIPE[family]
        for seed in study.SEEDS:
            fits.append({'family': family, 'seed': seed, 'status': 'complete', 'accepted_updates': 2,
                         'initial_identity': {'passed': True}, 'backbone_unchanged': True})
            error = .8 if arch == 'tanh_feedback' else 1.
            error += rate
            evaluations.append({'family': family, 'seed': seed,
                                'rows': [{'record_id': rid, 'status': 'complete', 'rmse': error} for rid in study.DEV_IDS],
                                'request_ms': [1.]*24, 'median_request_ms': 1., 'timing_error': None,
                                'persistent_numeric_bytes': 100 if arch.startswith('tanh') else 50})
    native = copy.deepcopy(evaluations[0]); native.update(family='native_varx', seed=None)
    for row in native['rows']:
        row['rmse'] = 1.
    evaluations.append(native)
    return fits, evaluations


def test_selection_uses_one_pooled_rate_not_individual_seed_winners():
    fits, evaluations = results()
    family = 'tanh_feedback-lr0.001'
    for e in evaluations:
        if e['family'] == family:
            for row in e['rows']:
                row['rmse'] = .1 if e['seed'] == 9201 else 2.
    summary = study.summarize(evaluations, fits, config())
    assert summary['candidate'] == 'tanh_feedback-lr0.0001'
    assert summary['status'] == 'DEVELOPMENT_PASS'
    assert summary['strongest_control'] == 'native_varx'


def test_rate_ties_choose_lowest_numeric_rate_and_affine_tie_uses_fixed_identity():
    fits, evaluations = results()
    for e in evaluations:
        for row in e['rows']:
            row['rmse'] = .8 if e['family'].startswith('tanh_feedback') else 1.
    summary = study.summarize(list(reversed(evaluations)), fits, config())
    assert all(v.endswith('lr0.0001') for v in summary['selected_by_architecture'].values())
    assert summary['strongest_affine'] == 'affine_feedback-lr0.0001'


@pytest.mark.parametrize('kind', ['missing_fit', 'duplicate_fit', 'unfinished', 'bad_identity', 'changed_backbone',
                                  'missing_eval', 'duplicate_eval', 'duplicate_record', 'nonfinite_error', 'timing_fail'])
def test_incomplete_or_inconsistent_evidence_cannot_pass(kind):
    fits, evaluations = results()
    if kind == 'missing_fit': fits.pop()
    elif kind == 'duplicate_fit': fits[0] = fits[1]
    elif kind == 'unfinished': fits[0]['accepted_updates'] = 1
    elif kind == 'bad_identity': fits[0]['initial_identity']['passed'] = False
    elif kind == 'changed_backbone': fits[0]['backbone_unchanged'] = False
    elif kind == 'missing_eval': evaluations.pop()
    elif kind == 'duplicate_eval': evaluations[0] = evaluations[1]
    elif kind == 'duplicate_record': evaluations[0]['rows'][0] = evaluations[0]['rows'][1]
    elif kind == 'nonfinite_error': evaluations[0]['rows'][0]['rmse'] = float('nan')
    elif kind == 'timing_fail': evaluations[0]['timing_error'] = 'broken'
    assert study.summarize(evaluations, fits, config())['status'] == 'DEVELOPMENT_FAIL'


def test_accuracy_selection_does_not_switch_to_cheaper_rate_after_cost_failure():
    fits, evaluations = results()
    for e in evaluations:
        if e['family'] == 'tanh_feedback-lr0.0001':
            e['request_ms'] = [10.]*24; e['median_request_ms'] = 10.
    summary = study.summarize(evaluations, fits, config())
    assert summary['candidate'] == 'tanh_feedback-lr0.0001'
    assert summary['conditions']['five_percent_below_strongest_control']
    assert not summary['conditions']['latency_within_ten_percent_tanh_output']
    assert summary['status'] == 'DEVELOPMENT_FAIL'


def test_failed_timing_does_not_replace_best_accuracy_recipe():
    fits, evaluations = results()
    for e in evaluations:
        if e['family'] == 'tanh_feedback-lr0.0001':
            e['timing_error'] = 'clock failed'; e['median_request_ms'] = None; e['request_ms'] = []
    summary = study.summarize(evaluations, fits, config())
    assert summary['candidate'] == 'tanh_feedback-lr0.0001'
    assert summary['conditions']['five_percent_below_strongest_control']
    assert not summary['conditions']['latency_within_ten_percent_tanh_output']
    assert summary['status'] == 'DEVELOPMENT_FAIL'


def test_only_global_strongest_control_defines_record_harm_guard():
    fits, evaluations = results()
    for e in evaluations:
        if e['family'].startswith('tanh_output_only'):
            e['rows'][0]['rmse'] = .01
            for row in e['rows'][1:]: row['rmse'] = 2.
    summary = study.summarize(evaluations, fits, config())
    assert summary['strongest_control'] == 'native_varx'
    assert summary['conditions']['no_record_over_two_percent_strongest_control']


def test_native_unit_overflow_is_retained_as_record_failure(tmp_path, monkeypatch):
    data, norm, _, _, backbone, _ = fixture(); cfg = config()
    norm = replace(norm, y_scale=np.full(3, 1e308))
    model = study.model_for('tanh_feedback-lr0.0001', 9201, backbone.coefficients, cfg)
    monkeypatch.setattr(study, 'predict', lambda model, inputs: np.full((len(inputs[0]), 128, 3), 1e10))
    with pytest.warns(RuntimeWarning, match='overflow'):
        result = study.evaluate(model, {'family': 'tanh_feedback-lr0.0001', 'seed': 9201},
                                data.partition('dev'), cfg, norm, tmp_path/'overflow')
    assert len(result['rows']) == 12 and all(r['status'] == 'failed' for r in result['rows'])
    assert all('nonfinite native-output error' in r['error'] for r in result['rows'])
    assert result['timing_error'] is not None
    assert (tmp_path/'overflow/receipt.json').exists()
