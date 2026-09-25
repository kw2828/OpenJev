"""Fabricated runner checks only; no measured arrays or optimizer updates."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_structured_study as study


def metric(value, horizon):
    return {'standardized_rmse': value, 'standardized_sse': value**2*120,
            'scalars': 120, 'physical_rmse_deg': value, 'per_joint_rmse_deg': [value]*6,
            'windows': 1, 'horizon': horizon}


def rows_fixture():
    cfg = study.config()
    rows = []
    for recording in cfg['partitions']['dev']:
        for arm in study.ALL_ARMS:
            for seed in cfg['seeds']:
                for rate in cfg['learning_rates']:
                    for h in cfg['horizons']:
                        value = .9 if arm == 'householder' else 1.
                        rows.append({'recording': recording, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': h, 'status': 'PASS',
                                     'metrics': metric(value, h), 'error': None})
        for arm in study.REFERENCES:
            for h in cfg['horizons']:
                rows.append({'recording': recording, 'arm': arm, 'seed': None,
                             'learning_rate': None, 'horizon': h, 'status': 'PASS',
                             'metrics': metric(1., h), 'error': None})
    return cfg, rows


def resources_fixture():
    rows = []
    for arm, count, latency in [('householder', 630, .008), ('dense_bounded', 806, .01)]:
        for seed in study.config()['seeds']:
            rows.append({'arm': arm, 'seed': seed, 'parameter_bytes': count*4,
                         'state_bytes': 48, 'normalizer_bytes': 192, 'buffer_bytes': 0,
                         'timing': {'median_seconds': latency}})
    return rows


def result(rows, cfg=None, resources=None):
    cfg = study.config() if cfg is None else cfg
    return study.evaluate_rule(rows, study.select(rows, cfg), resources_fixture() if resources is None else resources, cfg)


def condition_map(value):
    return {item['name']: item['passed'] for item in value['conditions']}


def test_complete_fixed_design_and_61_conditions():
    cfg, rows = rows_fixture()
    assert len(study.ARMS)*len(cfg['seeds'])*len(cfg['learning_rates']) == 30
    assert len(rows) == 160 and len(study.PAYLOADS) == 40
    assert cfg['updates'] == 4096 and cfg['train_horizon'] == cfg['dev_horizon'] == 128
    assert not cfg['legacy_refit'] and not cfg['causal_ridge_refit'] and cfg['batch_reuse']
    chosen = study.select(rows, cfg)
    assert set(chosen['selected_rates']) == set(study.ALL_ARMS)
    assert set(chosen['selected_rates'].values()) == {.001}
    assert chosen['selected_ridge'] == 'causal_ridge_1'
    value = result(rows, cfg)
    assert value['total'] == value['passed'] == 61
    assert len(condition_map(value)) == 61
    assert value['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'failed', 'nan', 'wrong_seed'])
def test_incomplete_or_failed_recipe_cannot_be_selected(damage):
    cfg, rows = rows_fixture()
    index = next(i for i, r in enumerate(rows) if r['arm'] == 'householder' and r['learning_rate'] == .001 and r['horizon'] == 128)
    if damage == 'missing':
        rows.pop(index)
    elif damage == 'duplicate':
        rows.append(copy.deepcopy(rows[index]))
    elif damage == 'failed':
        rows[index].update(status='FAILED', metrics=None)
    elif damage == 'nan':
        rows[index]['metrics']['standardized_rmse'] = float('nan')
    else:
        rows[index]['seed'] = 999
    selection = study.select(rows, cfg)
    assert selection['selected_rates']['householder'] == .003
    assert selection['options']['householder'][0]['eligible'] is False


def test_both_failed_rates_block_rule_and_cached_family_is_mandatory():
    cfg, rows = rows_fixture()
    for r in rows:
        if r['arm'] == 'legacy_instant':
            r.update(status='FAILED', metrics=None)
    value = result(rows, cfg)
    assert study.select(rows, cfg)['selected_rates']['legacy_instant'] is None
    assert not condition_map(value)['all_selected_families_and_causal_ridge_eligible']
    assert value['status'] == 'DO_NOT_ADVANCE_STRUCTURED_TRANSITION'


def test_selection_is_pooled_h128_sse_not_mean_rmse_or_h64():
    cfg, rows = rows_fixture()
    selected = [r for r in rows if r['arm'] == 'householder' and r['horizon'] == 128]
    for r in selected:
        if r['learning_rate'] == .001:
            r['metrics'] = metric(2., 128)
            r['metrics'].update(standardized_sse=4., scalars=1)
        else:
            r['metrics'] = metric(1., 128)
            r['metrics'].update(standardized_sse=1., scalars=1)
    selected[0]['metrics'].update(standardized_sse=0., scalars=10000, standardized_rmse=0.)
    for r in rows:
        if r['arm'] == 'householder' and r['horizon'] == 64 and r['learning_rate'] == .001:
            r.update(status='FAILED', metrics=None)
    assert study.select(rows, cfg)['selected_rates']['householder'] == .001


def test_ridge_requires_two_distinct_complete_dev_records():
    cfg, rows = rows_fixture()
    subset = [r for r in rows if r['arm'] == 'causal_ridge_1' and r['horizon'] == 128]
    subset[1]['recording'] = subset[0]['recording']
    assert study.select(rows, cfg)['selected_ridge'] == 'causal_ridge_100'


@pytest.mark.parametrize('change,key', [
    ('mean', '/mean_within_2pct/dense_bounded'),
    ('paired', '/seed8101_within_5pct/dense_bounded'),
    ('ridge', '/within_5pct_causal_ridge'),
    ('joint', '/joint5_no_10pct_harm'),
])
def test_quality_guard_failures_are_not_rescued_by_other_rows(change, key):
    cfg, rows = rows_fixture()
    recording = cfg['partitions']['dev'][0]
    for r in rows:
        if r['recording'] == recording and r['arm'] == 'householder' and r['horizon'] == 128:
            if change == 'mean':
                r['metrics'] = metric(1.021, 128)
            elif (change == 'paired' and r['seed'] == 8101) or change == 'ridge':
                r['metrics'] = metric(1.051, 128)
            elif change == 'joint':
                r['metrics']['per_joint_rmse_deg'][5] = 1.101
    assert condition_map(result(rows, cfg))[recording + key] is False


@pytest.mark.parametrize('damage,key', [('latency', 'at_most_80pct_dense_bounded_latency'),
                                       ('bytes', 'at_most_80pct_dense_bounded_numeric_storage'),
                                       ('duplicate', 'at_most_80pct_dense_bounded_latency')])
def test_cost_scope_and_median_complete_roster(damage, key):
    cfg, rows = rows_fixture()
    resources = resources_fixture()
    if damage == 'latency':
        for r in resources[:3]:
            r['timing']['median_seconds'] = .00801
    elif damage == 'bytes':
        resources[0]['buffer_bytes'] = 12
    else:
        resources[2]['seed'] = resources[1]['seed']
    assert condition_map(result(rows, cfg, resources))[key] is False


@pytest.mark.parametrize('arm,count,state', [('householder', 630, 12), ('dense_bounded', 806, 12),
                                            ('dense_unbounded', 806, 12), ('dense_mlp', 590, 12),
                                            ('gru32', 5916, 50), ('legacy_instant', 1014, 12)])
def test_model_resources_and_public_only_inference(arm, count, state):
    linear = np.zeros((6, 25), np.float64, order='F'); linear[:, :6] = np.eye(6)*.95
    model = study.model_for(arm, 97211, linear)
    resources = study.resource_model(model, arm)
    assert resources['parameters'] == count and resources['parameter_bytes'] == 4*count
    assert resources['state_scalars'] == state and resources['state_bytes'] == 4*state
    assert resources['normalizer_bytes'] == 192 and resources['buffer_bytes'] == 0
    assert resources['inactive_parameters'] == (192 if arm == 'legacy_instant' else 0)
    batch = {'q_context': np.sin(np.arange(2*4*6).reshape(2, 4, 6)/19),
             'u_context': np.cos(np.arange(2*4*6).reshape(2, 4, 6)/17),
             'future_u': np.sin(np.arange(2*6*6).reshape(2, 6, 6)/13), 'target': object()}
    first = study.old.infer(model, batch)
    batch['target'] = np.full((2, 6, 6), np.nan)
    assert torch.equal(first, study.old.infer(model, batch))
    batch['future_u'][:, 3:] += 10
    assert torch.equal(first[:, :3], study.old.infer(model, batch)[:, :3])
    assert first.shape == (2, 6, 6)


def test_gru32_scalar_torque_alignment_and_prefix_boundary():
    linear = np.zeros((6, 25), np.float64)
    linear[:, :6] = np.eye(6)*.5
    linear[:, 6:12] = np.eye(6)*.2
    linear[:, 12:18] = np.eye(6)*.3
    linear[:, 18:24] = np.eye(6)*.1
    model = study.GRU32(97212, linear)
    q = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)/10
    u = q/3
    state = model.condition(q, u)
    assert torch.equal(state.q, q[:, -1]) and torch.equal(state.prevq, q[:, -2])
    assert torch.equal(state.prevu, u[:, -2]) and state.hidden.shape == (1, 32)
    output, _ = model(u[:, -1:], state)
    expected = .5*q[:, -1]+.2*q[:, -2]+.3*u[:, -1]+.1*u[:, -2]
    torch.testing.assert_close(output[:, 0], expected)


@pytest.mark.parametrize('method', ['condition', 'forward'])
@pytest.mark.parametrize('message,translated', [('nonfinite structured transition output; no repair', True),
                                              ('bad caller shape', False)])
def test_only_declared_numeric_guard_translates(method, message, translated, monkeypatch):
    model = study.StructuredAdapter('householder', 1)
    error = ValueError(message)
    def raise_error(*args):
        raise error
    monkeypatch.setattr(model.cell, method, raise_error)
    with pytest.raises(study.old.FitFailure if translated else ValueError) as caught:
        getattr(model, method)(None, None)
    assert (caught.value.__cause__ is error) if translated else (caught.value is error)


def test_parent_array_pin_precedes_decode_and_preserves_layout(tmp_path, monkeypatch):
    path = tmp_path/'linear.npz'
    original = np.asfortranarray(np.arange(150.).reshape(6, 25))
    np.savez(path, coefficient=original)
    item = {'path': str(path), **study.old.descriptor(path)}
    plan = {'parent_payloads': {'linear.npz': item}}
    actual = study.load_parent_arrays(plan, 'linear.npz')['coefficient']
    assert actual.flags.f_contiguous and actual.flags.owndata and np.array_equal(actual, original)
    item['sha256'] = '0'*64
    def forbidden(*args, **kwargs):
        raise AssertionError('decoded before pin admission')
    monkeypatch.setattr(study.np, 'load', forbidden)
    with pytest.raises(ValueError, match='pinned file changed'):
        study.load_parent_arrays(plan, 'linear.npz')


def test_authentication_pins_source_parent_output_and_cached_payloads(tmp_path, monkeypatch):
    root, parent_folder = tmp_path/'repo', tmp_path/'parent'
    parent_folder.mkdir(); root.mkdir()
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'PARENT_FOLDER', parent_folder)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path': str(path), **study.old.descriptor(path)}
    sources = {}
    for name in study.SOURCES:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fabricated source')
        sources[name] = study.old.descriptor(path)
    payloads = {name: write(parent_folder/name, {'fabricated': name}) for name in study.PAYLOADS}
    closure = {}
    closure['manifest'] = write(parent_folder/'manifest.json', {'files': {n: {k: v[k] for k in ('sha256', 'bytes')} for n,v in payloads.items()}})
    closure['receipt'] = write(parent_folder/'receipt.json', {'status': 'PASS', 'registration_sha256': 'parent-sha'})
    closure['process'] = write(tmp_path/'process.json', {'returncode': 0, 'registration_sha256': 'parent-sha'})
    closure['audit'] = write(tmp_path/'audit.json', {'status': 'PASS', 'agreement': True, 'registration_sha256': 'parent-sha',
        'inputs': {'manifest': closure['manifest'], 'producer_receipt': closure['receipt'], 'run_receipt': closure['process'], 'parent_data': {}}})
    closure['audit_process'] = write(tmp_path/'audit-process.json', {'returncode': 0, 'audit_output': {k: closure['audit'][k] for k in ('sha256', 'bytes')}})
    monkeypatch.setattr(study.parent, 'authenticate', lambda path: ({'data': {}}, 'parent-sha'))
    plan = {'version': study.VERSION, 'config': study.config(), 'sources': sources,
            'parent_registration_sha256': 'parent-sha', 'data': {}, 'parent_closure': closure, 'parent_payloads': payloads}
    registration = tmp_path/'registration.json'; registration.write_text(json.dumps(plan))
    actual, sha = study.authenticate(registration)
    assert actual == plan and sha == hashlib.sha256(registration.read_bytes()).hexdigest()
    altered = copy.deepcopy(plan); altered['parent_payloads'].pop('fits.json')
    registration.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='payload roster'):
        study.authenticate(registration)
    registration.write_text(json.dumps(plan))
    (root/study.SOURCES[-1]).write_text('changed source')
    with pytest.raises(ValueError, match='source changed'):
        study.authenticate(registration)


def test_source_lifecycle_barrier_precedes_dev_and_only_fresh_arms_train():
    """Static lifecycle witness complements executed pure helpers, no fits."""
    source = inspect_source = Path(study.__file__).read_text()
    assert source.index("old.require(len(fits) == 30") < source.index("for name in cfg['partitions']['dev']:", source.index('def run('))
    assert source.index("'checkpoint-barrier.json'") < source.index("for name in cfg['partitions']['dev']:", source.index('def run('))
    loop = inspect_source[inspect_source.index('def run('):]
    assert loop.count('old.train_one(') == 1
    assert loop.index('for arm in ARMS:') < loop.index('old.train_one(') < loop.index("cached = json.loads")
    assert 'model.zero_grad(set_to_none=True)' in loop
