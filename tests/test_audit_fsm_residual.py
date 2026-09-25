"""Independent fabricated saved-evidence tests; no model or measurement calls."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('audit_fsm_residual', SCRIPTS/'audit_fsm_residual.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

CFG = {'updates': 2048, 'fit_timeout_seconds': 300, 'identity_atol': 1e-10, 'identity_rtol': 1e-10}


def evidence():
    fits, evaluations = [], []
    for family in (*audit.RECIPES, 'native_varx'):
        for seed in (None,) if family == 'native_varx' else audit.SEEDS:
            if seed:
                fits.append({'family': family, 'seed': seed, 'status': 'complete', 'accepted_updates': 2048,
                             'initial_identity': {'passed': True}, 'backbone_unchanged': True})
            evaluations.append({'family': family, 'seed': seed,
                                'rows': [{'record_id': r, 'status': 'complete', 'rmse': 8. if family.startswith('tanh_feedback') else 10.} for r in audit.DEV],
                                'timing_error': None, 'request_ms': [1.]*24,
                                'median_request_ms': 1., 'persistent_numeric_bytes': 100})
    return fits, evaluations


def change_error(evaluations, family, value, seed=None, record=None):
    for e in evaluations:
        if e['family'] == family and (seed is None or seed == e['seed']):
            for r in e['rows']:
                if record is None or r['record_id'] == record:
                    r['rmse'] = value


def test_full_geometry_nine_rules_and_numeric_rate_tie():
    fits, evaluations = evidence(); result = audit.decisions(evaluations, fits, CFG)
    assert len(fits) == 36 and len(evaluations) == 37
    assert sum(len(e['rows']) for e in evaluations) == 444
    assert result['passed'] == result['total'] == 9 and result['status'] == 'DEVELOPMENT_PASS'
    assert result['selected_by_architecture'] == {a: f'{a}-lr0.0001' for a in audit.ARCHITECTURES}
    assert result['strongest_affine'] == result['strongest_control'] == 'affine_feedback-lr0.0001'
    assert result['families'][result['candidate']]['mean_rmse'] == 8.


def test_rate_is_pooled_over_all_seeds_not_seed_specific():
    fits, evaluations = evidence()
    family = 'tanh_feedback-lr0.0003'
    for s, value in zip(audit.SEEDS, (1., 1., 25.), strict=True):
        change_error(evaluations, family, value, seed=s)
    result = audit.decisions(evaluations, fits, CFG)
    assert result['families'][family]['mean_rmse'] == 9.
    assert result['candidate'] == 'tanh_feedback-lr0.0001'
    change_error(evaluations, family, 15., seed=9203)
    assert audit.decisions(evaluations, fits, CFG)['candidate'] == family


def test_strongest_affine_identity_selected_globally():
    fits, evaluations = evidence()
    change_error(evaluations, 'affine_output_only-lr0.001', 7.)
    result = audit.decisions(evaluations, fits, CFG)
    assert result['strongest_affine'] == result['strongest_control'] == 'affine_output_only-lr0.001'
    assert result['conditions']['five_percent_below_strongest_control'] is False
    assert result['conditions']['every_seed_below_three_selected_controls'] is False


@pytest.mark.parametrize('kind,condition', [
    ('failed_fit', 'all_36_fits_complete'), ('missing_eval', 'all_37_evaluations_complete'),
    ('duplicate_seed', 'all_37_evaluations_complete'), ('identity', 'initial_identity_and_frozen_backbone'),
    ('frozen', 'initial_identity_and_frozen_backbone'), ('cost', 'latency_within_ten_percent_tanh_output'),
    ('storage', 'storage_no_more_than_tanh_output'), ('record', 'no_record_over_two_percent_strongest_control'),
    ('amplitude', 'both_amplitudes_below_strongest_control'), ('paired', 'every_seed_below_three_selected_controls'),
])
def test_each_failure_is_preserved(kind, condition):
    fits, evaluations = evidence()
    candidates = [e for e in evaluations if e['family'].startswith('tanh_feedback')]
    if kind == 'failed_fit':
        fits[0]['status'] = 'failed'
    elif kind == 'missing_eval':
        evaluations.pop()
    elif kind == 'duplicate_seed':
        evaluations[1]['seed'] = evaluations[0]['seed']
    elif kind == 'identity':
        fits[0]['initial_identity'] = None
    elif kind == 'frozen':
        fits[0]['backbone_unchanged'] = False
    elif kind in ('cost', 'storage'):
        for e in candidates:
            e['median_request_ms' if kind == 'cost' else 'persistent_numeric_bytes'] = 1.10001 if kind == 'cost' else 101
    elif kind == 'record':
        for e in candidates:
            e['rows'][0]['rmse'] = 10.20001
    elif kind == 'amplitude':
        for e in candidates:
            for r in e['rows']:
                r['rmse'] = 10. if r['record_id'].startswith('100mV') else 1.
    else:
        for e in candidates:
            if e['seed'] == 9201:
                for r in e['rows']:
                    r['rmse'] = 10.
    result = audit.decisions(evaluations, fits, CFG)
    assert result['conditions'][condition] is False
    assert result['status'] == 'DEVELOPMENT_FAIL'


def test_unselected_failure_cannot_vanish_and_not_run_has_no_fake_zero():
    fits, evaluations = evidence(); fit = fits[1]; fit['status'] = 'failed'; fit['accepted_updates'] = 0
    e = next(e for e in evaluations if (e['family'], e['seed']) == (fit['family'], fit['seed']))
    e.update(timing_error='fit_failed', request_ms=[], median_request_ms=None, persistent_numeric_bytes=None)
    e['rows'] = [{'record_id': r, 'status': 'not_run', 'error': 'fit_failed'} for r in audit.DEV]
    result = audit.decisions(evaluations, fits, CFG)
    assert result['families'][fit['family']] == {'eligible': False, 'score_eligible': False, 'latency_eligible': False, 'storage_eligible': False}
    assert result['selected_by_architecture']['affine_output_only'] == 'affine_output_only-lr0.0003'
    assert result['conditions']['all_36_fits_complete'] is False
    assert result['conditions']['all_37_evaluations_complete'] is False


def test_gate_boundaries_and_no_cost_based_rate_reselection():
    fits, evaluations = evidence()
    for e in evaluations:
        if e['family'].startswith('tanh_feedback'):
            e['median_request_ms'] = 1.1
            for r in e['rows']:
                r['rmse'] = 9.5
    assert audit.decisions(evaluations, fits, CFG)['passed'] == 9
    change_error(evaluations, 'tanh_feedback-lr0.001', 9.)
    for e in evaluations:
        if e['family'] == 'tanh_feedback-lr0.001':
            e['median_request_ms'] = 100.
    result = audit.decisions(evaluations, fits, CFG)
    assert result['candidate'] == 'tanh_feedback-lr0.001'
    assert result['conditions']['latency_within_ten_percent_tanh_output'] is False


def test_failed_timing_keeps_best_accuracy_rate_selected():
    fits, evaluations = evidence(); family = 'tanh_feedback-lr0.0003'
    change_error(evaluations, family, 7.)
    for e in evaluations:
        if e['family'] == family:
            e.update(timing_error='nonfinite timed forecast', request_ms=[1.], median_request_ms=None)
    result = audit.decisions(evaluations, fits, CFG)
    assert result['candidate'] == family
    assert result['conditions']['five_percent_below_strongest_control'] is True
    assert result['conditions']['latency_within_ten_percent_tanh_output'] is False
    assert result['conditions']['all_37_evaluations_complete'] is False


def checkpoint(kind='tanh'):
    family = f'{kind}_output_only-lr0.0001'; shapes = audit.parameter_shapes(kind)
    coefficients = np.zeros((3, 196)); initial = {'coefficients': torch.from_numpy(coefficients.copy())}
    initial.update({k: torch.zeros(s, dtype=torch.float64) for k, s in shapes.items()})
    slots = {i: {'step': torch.tensor(2048.), 'exp_avg': torch.zeros(s, dtype=torch.float64), 'exp_avg_sq': torch.zeros(s, dtype=torch.float64)} for i, s in enumerate(shapes.values())}
    final = {'model': copy.deepcopy(initial), 'accepted_updates': 2048,
             'optimizer': {'state': slots, 'param_groups': [{'params': list(slots), 'lr': .0001, 'weight_decay': 0., 'betas': (.9, .999), 'eps': 1e-8}]}}
    count = sum(int(np.prod(s)) for s in shapes.values())
    receipt = {'family': family, 'architecture': f'{kind}_output_only', 'learning_rate': .0001, 'seed': 9201,
               'accepted_updates': 2048, 'status': 'complete', 'error': None, 'fit_seconds': 20., 'backbone_unchanged': True,
               'initial_identity': {'passed': True, 'max_abs_difference': 0., 'atol': 1e-10, 'rtol': 1e-10},
               'model_spec': {'mode': 'output_only', 'residual_kind': kind, 'parameter_count': count,
                              'trainable_parameter_count': count, 'parameter_bytes': count*8,
                              'buffer_bytes': 4704, 'state_scalars': 192, 'numeric_metadata_bytes': 24 if kind == 'tanh' else 16}}
    return initial, final, receipt, coefficients


@pytest.mark.parametrize('kind', ['tanh', 'affine'])
def test_tensor_only_checkpoint_validation_and_exact_accounting(kind):
    i, f, r, coefficients = checkpoint(kind)
    expected = (42936, 24) if kind == 'tanh' else (9408, 16)
    assert audit.validate_checkpoint(i, f, r, coefficients, {}, CFG) == expected


@pytest.mark.parametrize('mutation', ['buffer', 'zero_head', 'step', 'slot_shape', 'extra_parameter', 'dtype', 'rate', 'count', 'missing_optimizer', 'nonfinite_complete'])
def test_checkpoint_tamper_rejected(mutation):
    i, f, r, coefficients = checkpoint()
    if mutation == 'buffer':
        f['model']['coefficients'][0, 0] = 1.
    elif mutation == 'zero_head':
        i['residual.2.bias'][0] = 1.
    elif mutation == 'step':
        f['optimizer']['state'][0]['step'] = torch.tensor(2047.)
    elif mutation == 'slot_shape':
        f['optimizer']['state'][0]['exp_avg'] = torch.zeros(1, dtype=torch.float64)
    elif mutation == 'extra_parameter':
        i['secret'] = torch.zeros(1)
    elif mutation == 'dtype':
        i['residual.0.weight'] = i['residual.0.weight'].float()
    elif mutation == 'rate':
        f['optimizer']['param_groups'][0]['lr'] = .1
    elif mutation == 'count':
        f['accepted_updates'] = 3
    elif mutation == 'missing_optimizer':
        f['optimizer']['state'].pop(0)
    else:
        f['model']['residual.0.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        audit.validate_checkpoint(i, f, r, coefficients, {}, CFG)


def test_initial_hidden_values_pair_across_modes_and_rates():
    i, f, r, coefficients = checkpoint(); paired = {}
    audit.validate_checkpoint(i, f, r, coefficients, paired, CFG)
    r.update(family='tanh_feedback-lr0.0003', architecture='tanh_feedback', learning_rate=.0003)
    r['model_spec']['mode'] = 'feedback'; f['optimizer']['param_groups'][0]['lr'] = .0003
    audit.validate_checkpoint(i, f, r, coefficients, paired, CFG)
    i['residual.0.weight'][0, 0] = 1.
    with pytest.raises(ValueError, match='paired'):
        audit.validate_checkpoint(i, f, r, coefficients, paired, CFG)


def test_failed_nonfinite_optimizer_attempt_is_preserved_not_repaired():
    i, f, r, coefficients = checkpoint()
    r.update(status='failed', error={'type': 'ValueError', 'message': 'nonfinite residual'}, accepted_updates=0)
    f['accepted_updates'] = 0
    for slots in f['optimizer']['state'].values():
        slots['step'] = torch.tensor(1.)
    f['model']['residual.0.weight'][0, 0] = float('nan')
    audit.validate_checkpoint(i, f, r, coefficients, {}, CFG)
    assert torch.isnan(f['model']['residual.0.weight'][0, 0])


def test_recomputed_metrics_detect_changed_or_nonfinite_saved_predictions():
    y = np.zeros((32, 128, 3)); p = np.ones_like(y)
    metric = audit.metrics(p, y); assert metric['rmse'] == metric['mse'] == 1.
    p[0, 0, 0] = 2.
    with pytest.raises(ValueError):
        audit.close(audit.metrics(p, y), metric)
    p[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.metrics(p, y)


def test_native_per_channel_units_are_scaled_separately():
    p = np.broadcast_to(np.array([1., 2., 3.]), (32, 128, 3)).copy()
    value = audit.record_metrics(p, np.zeros_like(p), np.array([2., 3., 4.]))
    assert value['per_channel_rmse'] == [1., 2., 3.]
    assert value['native_output_per_channel_rmse'] == [2., 6., 12.]
    assert value['mse'] == 14/3
    with pytest.raises(ValueError, match='native channel'):
        audit.record_metrics(p, np.zeros_like(p), np.full(3, np.finfo(float).max))


def test_original_failed_process_stops_before_registration_or_arrays(tmp_path, monkeypatch):
    process = tmp_path/'process.json'; process.write_text(json.dumps({'observed_exit_code': 1}))
    monkeypatch.setattr(audit, 'pin', lambda _: pytest.fail('must not inspect study or source'))
    with pytest.raises(ValueError, match='successful process'):
        audit.authenticate(tmp_path/'never-existed', process)


def authenticated_tree(tmp_path, monkeypatch, recorded_parent=None):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    study = tmp_path/'output/fsm-residual-study-v1'; study.mkdir(parents=True)
    parent = tmp_path/'output/fsm-correction-study-v1'; parent.mkdir(parents=True)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    source_pins = {}
    for name in audit.SOURCES:
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('opaque source '+name)
        copied = study/'source'/name; copied.parent.mkdir(parents=True, exist_ok=True); copied.write_bytes(p.read_bytes())
        source_pins[name] = audit.pin(p)['sha256']
    paths = {'coefficients': parent/'model.npz', 'fit_receipt': parent/'fit.json',
             'normalizer': parent/'norm.npz', 'registration': tmp_path/'parent-registration.json',
             'audit': tmp_path/'parent-audit.json', 'closure': parent/'closure.json',
             'summary': parent/'summary.json', 'process': tmp_path/'parent-process.json'}
    for key in ('coefficients', 'fit_receipt', 'normalizer', 'registration', 'summary'):
        save(paths[key], {'opaque': key})
    summary_sha = audit.pin(paths['summary'])['sha256']
    save(paths['process'], {'observed_exit_code': 0, 'summary_sha256': summary_sha})
    save(paths['closure'], {'status': 'completed', 'summary_sha256': summary_sha})
    save(paths['audit'], {'status': 'PASS', 'agreement': True,
                         'inputs': {'registration': audit.pin(paths['registration']),
                                    'process': audit.pin(paths['process']), 'study': str(parent) if recorded_parent is None else recorded_parent,
                                    'files': {str(paths[k].relative_to(parent)): audit.pin(paths[k]) for k in ('coefficients', 'fit_receipt', 'normalizer', 'summary', 'closure')}}})
    artifacts = {k: {'path': str(p.relative_to(tmp_path)), 'sha256': audit.pin(p)['sha256']} for k, p in paths.items()}
    for key, p in paths.items():
        dest = study/'parent'/(key+p.suffix); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    markdown = tmp_path/'research/fsm-residual-protocol.md'; markdown.parent.mkdir(parents=True); markdown.write_text('prospective fixture')
    plan = {'source_sha256': source_pins, 'parent_artifacts': artifacts,
            'protocol_markdown_path': str(markdown.relative_to(tmp_path)),
            'protocol_markdown_sha256': audit.pin(markdown)['sha256']}
    registration = save(tmp_path/audit.REG, plan); monkeypatch.setattr(audit, 'REGISTRATION_SHA256', audit.pin(registration)['sha256'])
    save(study/'protocol.json', plan); save(study/'summary.json', {'status': 'DEVELOPMENT_FAIL'})
    summary_sha = audit.pin(study/'summary.json')['sha256']
    save(study/'closure.json', {'status': 'completed', 'source_sha256': source_pins, 'summary_sha256': summary_sha})
    process = save(tmp_path/'process.json', {'observed_exit_code': 0, 'command': audit.COMMAND,
                                           'tool_session_id': 12, 'observation': 'original Codex exec/write_stdin completion',
                                           'summary_sha256': summary_sha})
    monkeypatch.setattr(audit, 'arrays', lambda *_: pytest.fail('metadata admission must not decode arrays'))
    return study, process, paths


def test_full_opaque_parent_and_original_closure_admission(tmp_path, monkeypatch):
    study, process, _ = authenticated_tree(tmp_path, monkeypatch)
    plan, paths, _, _ = audit.authenticate(study, process)
    assert len(plan['source_sha256']) == 7 and len(paths) == 8


@pytest.mark.parametrize('mutation', ['source', 'snapshot', 'parent', 'parent_copy', 'process', 'closure', 'registration', 'markdown'])
def test_metadata_drift_rejected_before_any_array_read(tmp_path, monkeypatch, mutation):
    study, process, paths = authenticated_tree(tmp_path, monkeypatch)
    source = min(audit.SOURCES)
    target = {'source': tmp_path/source, 'snapshot': study/'source'/source,
              'parent': paths['coefficients'], 'parent_copy': study/'parent/coefficients.npz',
              'process': process, 'closure': study/'closure.json', 'registration': tmp_path/audit.REG,
              'markdown': tmp_path/'research/fsm-residual-protocol.md'}[mutation]
    if mutation in ('process', 'closure'):
        data = json.loads(target.read_text()); data['summary_sha256'] = '0'*64; target.write_text(json.dumps(data))
    else:
        target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError):
        audit.authenticate(study, process)


@pytest.mark.parametrize('recorded_parent', [
    '/Users/kevinwu/Documents/research/WikiSkills-RL/OpenJev/output/fsm-correction-study-v1',
    '/different-host/clone/OpenJev/output/fsm-correction-study-v1',
])
def test_parent_join_portable_across_clone_roots(tmp_path, monkeypatch, recorded_parent):
    study, process, _ = authenticated_tree(tmp_path, monkeypatch, recorded_parent)
    plan, _, _, _ = audit.authenticate(study, process)
    relative = plan['parent_artifacts']['coefficients']['path']
    assert audit.parent_payload_key(relative, recorded_parent) == 'model.npz'
    assert audit.parent_payload_key(relative, str(tmp_path/audit.PARENT_ROOT)) == 'model.npz'


@pytest.mark.parametrize('relative,recorded', [
    ('output/fsm-correction-study-v1/model.npz', '/host/output/wrong-study'),
    ('output/fsm-correction-study-v1/model.npz', 'output/fsm-correction-study-v1'),
    ('output/fsm-correction-study-v1/model.npz', '/host/../output/fsm-correction-study-v1'),
    ('/host/output/fsm-correction-study-v1/model.npz', '/host/output/fsm-correction-study-v1'),
    ('output/fsm-correction-study-v1/../model.npz', '/host/output/fsm-correction-study-v1'),
    ('output/other-study/model.npz', '/host/output/fsm-correction-study-v1'),
    ('output/fsm-correction-study-v10/model.npz', '/host/output/fsm-correction-study-v1'),
    ('output/fsm-correction-study-v1', '/host/output/fsm-correction-study-v1'),
])
def test_parent_join_rejects_wrong_root_traversal_and_outside_payload(relative, recorded):
    with pytest.raises(ValueError):
        audit.parent_payload_key(relative, recorded)
