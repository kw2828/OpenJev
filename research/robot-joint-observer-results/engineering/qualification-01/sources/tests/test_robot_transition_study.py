"""Fabricated transition-runner contracts; never load measured recordings."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import robot_transition_study as study


def fixture_rows():
    cfg = study.config()
    rows = []
    for recording in cfg['partitions']['dev']:
        for arm in (*study.ARMS, *study.REFERENCES):
            neural = arm in study.ARMS
            for seed in cfg['seeds'] if neural else [None]:
                for rate in cfg['learning_rates'] if neural else [None]:
                    error = .8 if arm == 'lpv_recurrent' else 1.
                    if neural and rate == cfg['learning_rates'][1]:
                        error += .3
                    for horizon in cfg['horizons']:
                        scalars = 2 * horizon * 6
                        rows.append({'recording': recording, 'arm': arm, 'seed': seed,
                                     'learning_rate': rate, 'horizon': horizon, 'status': 'PASS',
                                     'metrics': {'standardized_rmse': error,
                                                 'standardized_sse': error**2 * scalars,
                                                 'scalars': scalars, 'windows': 2, 'horizon': horizon,
                                                 'physical_rmse_deg': error,
                                                 'per_joint_rmse_deg': [error] * 6}})
    return rows, cfg


def fixture_resources(cfg):
    return [{'arm': arm, 'seed': seed, 'learning_rate': cfg['learning_rates'][0],
             'parameter_bytes': 100, 'state_bytes': 20, 'buffer_bytes': 0,
             'normalizer_bytes': 192, 'timing': {'median_seconds': .01}}
            for arm in study.ARMS for seed in cfg['seeds']]


def get_row(rows, *, arm='lpv_recurrent', rate=.001, seed=8101, horizon=128, recording=None):
    return next(r for r in rows if r['arm'] == arm and r['learning_rate'] == rate
                and r['seed'] == seed and r['horizon'] == horizon
                and (recording is None or r['recording'] == recording))


def test_fixed_exposure_and_complete_selection():
    rows, cfg = fixture_rows()
    assert cfg['updates'] == 4096 and cfg['train_horizon'] == 128 and cfg['batch_size'] == 16
    assert cfg['learning_rates'] == [.001, .003]
    assert len(cfg['seeds']) * len(cfg['learning_rates']) * len(study.ARMS) == 24
    choice = study.select(rows, cfg)
    assert choice['selected_rates'] == dict.fromkeys(study.ARMS, .001)
    assert choice['selected_ridge'] == 'causal_ridge_1'
    assert all(o['eligible'] for values in choice['options'].values() for o in values)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'replace_seed', 'failed', 'nonfinite'])
def test_selection_rejects_incomplete_or_failed_rate_without_selecting_seeds(fault):
    rows, cfg = fixture_rows()
    row = get_row(rows)
    if fault == 'missing':
        rows.remove(row)
    elif fault == 'duplicate':
        rows.append(copy.deepcopy(row))
    elif fault == 'replace_seed':
        row['seed'] = cfg['seeds'][1]
    elif fault == 'failed':
        row.update(status='FAILED', metrics=None)
    else:
        row['metrics']['standardized_rmse'] = float('nan')
    choice = study.select(rows, cfg)
    assert not choice['options']['lpv_recurrent'][0]['eligible']
    assert choice['selected_rates']['lpv_recurrent'] == .003


def test_all_rates_failed_means_no_candidate_and_rule_cannot_pass():
    rows, cfg = fixture_rows()
    for row in rows:
        if row['arm'] == 'lpv_recurrent':
            row.update(status='FAILED', metrics=None)
    choice = study.select(rows, cfg)
    assert choice['selected_rates']['lpv_recurrent'] is None
    result = study.evaluate_rule(rows, choice, fixture_resources(cfg), cfg)
    assert result['status'] == 'DO_NOT_ADVANCE_TRANSITION'
    assert not result['conditions'][0]['passed']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'failed'])
def test_ridge_requires_both_distinct_dev_recordings(fault):
    rows, cfg = fixture_rows()
    row = get_row(rows, arm='causal_ridge_1', rate=None, seed=None)
    if fault == 'missing':
        rows.remove(row)
    elif fault == 'duplicate':
        rows.append(copy.deepcopy(row))
    else:
        row.update(status='FAILED', metrics=None)
    choice = study.select(rows, cfg)
    assert choice['selected_ridge'] == 'causal_ridge_100'
    assert not choice['ridge_options'][0]['eligible']


def test_selection_uses_pooled_sse_and_exact_rate_tie_break_not_h64():
    rows, cfg = fixture_rows()
    for row in rows:
        if row['arm'] == 'lpv_recurrent':
            m = row['metrics']
            error = .7 if row['learning_rate'] == .003 else .9
            m.update(standardized_rmse=error, standardized_sse=error**2 * m['scalars'])
            if row['horizon'] == 64:
                row.update(status='FAILED', metrics=None)
    assert study.select(rows, cfg)['selected_rates']['lpv_recurrent'] == .003
    for row in rows:
        if row['arm'] == 'lpv_recurrent' and row['horizon'] == 128:
            row['metrics']['standardized_sse'] = .7**2 * row['metrics']['scalars']
            row['metrics']['standardized_rmse'] = .7
    assert study.select(rows, cfg)['selected_rates']['lpv_recurrent'] == .001


def test_full_synthetic_rule_pass_has_exact_unique_45_conditions():
    rows, cfg = fixture_rows()
    result = study.evaluate_rule(rows, study.select(rows, cfg), fixture_resources(cfg), cfg)
    assert result['status'] == 'QUALIFIES_FOR_NEW_CONFIRMATION_PROTOCOL'
    assert result['passed'] == result['total'] == 45
    assert len({r['name'] for r in result['conditions']}) == 45
    assert all(r['passed'] for r in result['conditions'])


@pytest.mark.parametrize('fault', ['mean', 'paired', 'joint', 'ridge', 'latency', 'storage', 'cost_roster'])
def test_rule_fails_quality_or_cost_constraints_even_with_other_successes(fault):
    rows, cfg = fixture_rows()
    resources = fixture_resources(cfg)
    selection = study.select(rows, cfg)
    recording = cfg['partitions']['dev'][0]
    if fault in ('mean', 'paired', 'ridge'):
        for row in rows:
            if (row['recording'] == recording and row['arm'] == 'lpv_recurrent'
                    and row['learning_rate'] == .001 and row['horizon'] == 128
                    and (fault != 'paired' or row['seed'] == cfg['seeds'][0])):
                error = {'mean': .96, 'paired': .99, 'ridge': 1.06}[fault]
                row['metrics'].update(standardized_rmse=error,
                                      standardized_sse=error**2 * row['metrics']['scalars'])
        suffix = {'mean': '/mean_5pct/lpv_instant', 'paired': '/seed8101_2pct/lpv_instant',
                  'ridge': '/within_5pct_causal_ridge'}[fault]
        key = recording + suffix
    elif fault == 'joint':
        for row in rows:
            if row['recording'] == recording and row['arm'] == 'lpv_recurrent' and row['horizon'] == 128:
                row['metrics']['per_joint_rmse_deg'][5] = 1.11
        key = recording + '/joint5_no_10pct_harm'
    else:
        if fault == 'latency':
            for r in resources:
                if r['arm'] == 'lpv_recurrent':
                    r['timing']['median_seconds'] = .02001
            key = 'at_most_twice_gru_latency'
        elif fault == 'storage':
            resources[0]['parameter_bytes'] += 1
            key = 'at_most_gru_numeric_storage'
        else:
            resources.append(copy.deepcopy(resources[0]))
            key = 'at_most_twice_gru_latency'
    result = study.evaluate_rule(rows, selection, resources, cfg)
    assert result['status'] == 'DO_NOT_ADVANCE_TRANSITION'
    assert not next(c for c in result['conditions'] if c['name'] == key)['passed']


def test_ridge_batches_align_boundary_torque_and_do_not_cross_recordings():
    cfg = dict(study.config(), skip=2, direct_fit_stride=9)
    records = []
    for r, length in enumerate((173, 188)):
        t = np.arange(length)[:, None]
        j = np.arange(6)[None, :]
        records.append({'q': 10000 * r + 10 * t + j, 'u': -20000 * r - 20 * t - j})
    actual = study.ridge_batch(records, cfg)
    expected = [(r, s) for r, d in enumerate(records) for s in range(2, len(d['q']) - 159, 9)]
    assert actual['target'].shape == (len(expected), 128, 6)
    for b, (r, s) in enumerate(expected):
        np.testing.assert_array_equal(actual['q_context'][b], records[r]['q'][s:s + 32])
        np.testing.assert_array_equal(actual['u_context'][b], records[r]['u'][s:s + 32])
        np.testing.assert_array_equal(actual['future_u'][b], records[r]['u'][s + 31:s + 159])
        np.testing.assert_array_equal(actual['target'][b], records[r]['q'][s + 32:s + 160])


def test_saved_arrays_keep_fortran_layout_and_owned_bytes_with_pin_guard(tmp_path):
    path = tmp_path / 'fabricated.npz'
    original = np.asfortranarray(np.arange(150, dtype=np.float64).reshape(6, 25))
    np.savez_compressed(path, reference=original)
    plan = {'data': {'reference.npz': {'path': str(path), **study.old.descriptor(path)}}}
    first = study.load_arrays(plan, 'reference.npz')['reference']
    assert first.flags.f_contiguous and not first.flags.c_contiguous and first.flags.owndata
    np.testing.assert_array_equal(first, original)
    first[0, 0] = -1
    np.testing.assert_array_equal(study.load_arrays(plan, 'reference.npz')['reference'], original)
    with pytest.raises(ValueError, match='unregistered'):
        study.load_arrays(plan, 'other.npz')
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed before load'):
        study.load_arrays(plan, 'reference.npz')


@pytest.mark.parametrize('arm', study.ARMS)
def test_adapter_target_isolation_and_no_future_torque_leakage(arm):
    linear = np.zeros((6, 25))
    linear[:, :6] = .8 * np.eye(6)
    t, j = np.arange(32)[None, :, None], np.arange(6)[None, None, :]
    q = np.sin(.1 * t + .2 * j).astype(np.float32)
    batch = {'q_context': q, 'u_context': q * .2,
             'future_u': np.cos(.13 * np.arange(12)[None, :, None] + .2 * j).astype(np.float32),
             'target': object()}
    model = study.model_for(arm, 991101, linear)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        prediction = study.old.infer(model, batch)
        altered = dict(batch, target=np.full((1, 12, 6), np.nan))
        assert torch.equal(prediction, study.old.infer(model, altered))
        altered['future_u'] = batch['future_u'].copy()
        altered['future_u'][:, 5:] += 20
        changed = study.old.infer(model, altered)
    assert prediction.shape == (1, 12, 6) and torch.isfinite(prediction).all()
    assert torch.equal(prediction[:, :5], changed[:, :5])
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


@pytest.mark.parametrize('arm', study.ARMS)
def test_resource_rows_account_actual_parameters_buffers_and_state(arm):
    model = study.model_for(arm, 991101, np.zeros((6, 25)))
    resource = study.resource_model(model, arm)
    assert resource['parameters'] == sum(p.numel() for p in model.parameters())
    assert resource['parameter_bytes'] == sum(p.numel() * p.element_size() for p in model.parameters())
    assert resource['buffer_bytes'] == sum(b.numel() * b.element_size() for b in model.buffers())
    assert resource['state_scalars'] == (28 if arm == 'gru_residual' else model.cell.state_scalars)
    assert resource['state_bytes'] == resource['state_scalars'] * 4
    assert resource['inactive_parameters'] == (192 if arm == 'lpv_instant' else 0)


@pytest.mark.parametrize('entry,delegate', [('condition', 'condition'), ('forward', 'rollout')])
@pytest.mark.parametrize('numerical', [True, False])
def test_adapter_translates_only_declared_numerical_guard(monkeypatch, entry, delegate, numerical):
    adapter = study.LPVAdapter('recurrent', 991101)
    message = ('nonfinite LPV output; no rollout clipping or repair' if numerical
               else 'finite CPU tensor with exact shape/dtype: observed context positions')
    original = ValueError(message)

    def fail(*args):
        raise original

    monkeypatch.setattr(adapter.cell, delegate, fail)
    expected = study.old.FitFailure if numerical else ValueError
    with pytest.raises(expected, match=message) as caught:
        getattr(adapter, entry)(None, None)
    if numerical:
        assert caught.value.__cause__ is original
    else:
        assert caught.value is original
