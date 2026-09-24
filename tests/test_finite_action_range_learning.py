"""Fabricated runner boundary tests, without world generation or model calls.

Tiny identity labels below are metadata fixtures, not admitted scientific or
engineering RNG namespaces. The separately registered retained smoke uses
948001/948101 and is executed once by the qualification wrapper, not pytest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_finite_action_range_learning as runner
import run_finite_action_range_training as trainer
import run_finite_update_learning as frozen

FAKE = {'cohorts': [{'seed_namespace': 1, 'fit_seed': 11}, {'seed_namespace': 2, 'fit_seed': 12}],
        'train_attempts': 4, 'dev_attempts': 4, 'batch_size': 3,
        'prefix_updates': 2, 'joint_updates': 3, 'fit_cap_seconds': 30.}
SMOKE = {'cohorts': [{'seed_namespace': 948001, 'fit_seed': 948101}],
         'train_attempts': 64, 'dev_attempts': 32, 'batch_size': 16,
         'prefix_updates': 2, 'joint_updates': 3, 'fit_cap_seconds': 30.}


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fake_state(model):
    return {'fabricated': np.asarray([model.token], dtype=np.float64)}


def fit_record(folder, arm, seed, config, *, token=None):
    """Opaque checkpoint bytes plus allocation JSON, with no checkpoint decode."""
    transport = runner.TRANSPORT_ARMS[arm]
    token = seed * 10 + runner.ARMS.index(arm) if token is None else token
    final_hash = runner._state_hash(fake_state(SimpleNamespace(token=token)))
    checkpoints = []
    for label in ('initial', 'boundary', 'final'):
        model_path = folder / f'{label}-{arm}-{seed}.npz'
        optimizer_path = folder / f'{label}-optimizer-{arm}-{seed}.json'
        model_path.write_bytes(b'fabricated opaque checkpoint, intentionally not an NPZ decoder fixture')
        optimizer_path.write_text(json.dumps({'fabricated': True, 'label': label}))
        metadata = {
            'model': {'path': model_path.name, **runner._desc(model_path)},
            'optimizer': {'path': optimizer_path.name, **runner._desc(optimizer_path)},
            'model_state_sha256': final_hash if label == 'final' else digest((seed, transport, label, 'model')),
            'dynamics_state_sha256': digest((seed, arm if label == 'final' else transport, label, 'dynamics')),
            'head_state_sha256': digest((seed, arm if label == 'final' else 'paired', 'head')),
            'optimizer_state_sha256': digest((seed, transport, label, 'optimizer')),
        }
        checkpoints.append({'label': label, 'metadata': metadata})
    prefix, joint = config['prefix_updates'], config['joint_updates']
    trace = [{'kind': 'prefix', 'accepted': True, 'rolled_back': False} for _ in range(prefix)]
    trace += [{'kind': 'joint', 'accepted': True, 'rolled_back': False,
               'result': {'diagnostics': {'indices': [(seed + cursor) % config['train_attempts']]}}}
              for cursor in range(joint)]
    allocation = {'status': 'PASS', 'accepted_prefix_updates': prefix, 'accepted_joint_updates': joint,
        'joint_cursor': joint, 'attempted_updates': prefix + joint, 'accepted_updates': prefix + joint,
        'trace': trace, 'checkpoints': checkpoints}
    path = folder / f'allocation-{arm}-{seed}.json'
    path.write_text(json.dumps(allocation))
    fit = {'arm': arm, 'seed': seed, 'implementation': 'reuse', 'integration_version': trainer.VERSION,
        'allocation': {'path': path.name, **runner._desc(path)}, 'accepted_prefix_updates': prefix,
        'updates': joint, 'attempted_updates': prefix + joint,
        'initial_state_sha256': checkpoints[0]['metadata']['model_state_sha256'],
        'final_state_sha256': final_hash, 'checkpoint': checkpoints[-1]['metadata']['model']}
    for field in ('dynamics', 'head'):
        fit[field + '_boundary_sha256'] = {row['label']: row['metadata'][field + '_state_sha256']
                                         for row in checkpoints}
        fit[field + '_boundary_hash_evaluations'] = 3
    return fit


def patch_fabricated_flow(monkeypatch, folder, *, fail_at=None, tamper_at=None):
    events, fits_called = [], []
    expected = len(runner.ARMS) * len(FAKE['cohorts'])

    def reference(epsilon):
        if epsilon == .30:
            assert len(fits_called) == expected and (folder / 'checkpoint-barrier.json').is_file()
        events.append(('reference', epsilon))
        return SimpleNamespace(token=epsilon, parameters=list,
                               parameter_metadata=lambda: {'count': 0, 'epsilon': epsilon})

    def generate(split, attempts, horizon, *, seed_namespace):
        if split:
            assert len(fits_called) == expected
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            assert barrier['fit_count'] == expected and barrier['dev_generation_count'] == 0
            assert len(barrier['checkpoints']) == expected
            assert all((folder / row['path']).is_file() for row in barrier['checkpoints'])
        events.append(('generate', seed_namespace, split))
        n = attempts - 1  # One found prefix remains in all-attempt prefix supervision.
        prefixes = np.zeros((attempts, 9, 31), dtype=np.float64)
        lengths = np.full(attempts, 9, dtype=np.int64)
        lengths[-1] = 2
        prefix_data = {'prefix': prefixes, 'lengths': lengths,
            'event_mask': np.arange(9)[None, :] < lengths[:, None],
            'endpoint_eligible': np.arange(attempts) < n,
            'case_ids': np.asarray([f'fabricated-{seed_namespace}-{split}-{i}' for i in range(attempts)])}
        probabilities = np.zeros((n, horizon, 5), dtype=np.float64)
        probabilities[..., :4] = .25
        data = {'prefix': prefixes[:n].copy(), 'lengths': lengths[:n].copy(),
            'actions': np.zeros((n, horizon), dtype=np.int64),
            'observations': np.zeros((n, horizon), dtype=np.int64),
            'blind_costs': np.zeros((n, horizon, 4)), 'observed_costs': np.zeros((n, horizon, 4)),
            'blind_survival': np.ones((n, horizon)), 'observed_survival': np.ones((n, horizon)),
            'observed_probabilities': probabilities}
        return {'data': data, 'prefix_data': prefix_data,
                'oracle': {'prefix_state': np.full((n, 8), .125)},
                'counts': {'attempts': attempts, 'retained': n,
                           'valid_prefix_events': int(prefix_data['event_mask'].sum())}}

    def train(arm, seed, data, prefixes, config, output, counts, work, check, *, implementation):
        assert implementation == 'reuse'
        assert not any(event[0] == 'generate' and event[2] != 0 for event in events)
        assert len(prefixes['prefix']) == 4 and len(data['prefix']) == 3
        assert 'oracle_prefix' not in data and 'oracle_prefix' not in prefixes
        assert (output / 'oracle-train-check.json').is_file()
        if (seed, arm) == fail_at:
            marker = TimeoutError('fabricated original safety-cap stop')
            marker.action_range_loss_work = {'wrapper_calls': 1}
            runner._write(output / f'failed-action-range-loss-{arm}-{seed}.json', marker.action_range_loss_work)
            raise marker
        fit = fit_record(output, arm, seed, config)
        if (seed, arm) == tamper_at:
            fit['head_boundary_sha256']['boundary'] = 'f' * 64
        fits_called.append((seed, arm))
        events.append(('fit', seed, arm))
        for key in ('model_constructions', 'fit_count'):
            counts[key] += 1
        for key in ('checkpoint_writes', 'optimizer_checkpoint_writes'):
            counts[key] += 3
        counts['accepted_prefix_steps'] += config['prefix_updates']
        counts['accepted_joint_steps'] += config['joint_updates']
        for key in ('accepted_optimizer_steps', 'optimizer_attempts', 'optimizer_steps'):
            counts[key] += config['prefix_updates'] + config['joint_updates']
        work[arm]['training_prefix']['fabricated_calls'] = 1
        runner._append(output / 'fits.jsonl', fit)
        runner._append(output / 'training-orders.jsonl', {'arm': arm, 'seed': seed, 'fabricated': True})
        return SimpleNamespace(token=seed * 10 + runner.ARMS.index(arm)), fit

    def predict(model, arm, data, oracle, config, counts, check, *, shuffled, structural_work=None):
        if arm != 'exact_exact':
            assert len(fits_called) == expected and shuffled is True
            structural_work[arm]['evaluation_blind']['fabricated_calls'] = (
                structural_work[arm]['evaluation_blind'].get('fabricated_calls', 0) + 1)
        result = {name: data[name].copy() for name in runner.ORACLE_FIELDS}
        if shuffled:
            result['shuffled_blind_costs'] = data['blind_costs'].copy()
        return result

    def prefix(model, arm, seed, data, config, counts, work, check):
        assert len(data['prefix']) == 4 and not bool(data['endpoint_eligible'][-1])
        return ({'probabilities': np.zeros((4, 9, 5)), 'nll': np.zeros((4, 9))},
                {'arm': arm, 'seed': seed, 'attempts': 4,
                 'valid_events': int(data['event_mask'].sum()), 'mean_nll': 0.})

    monkeypatch.setattr(runner, 'make_reference', reference)
    monkeypatch.setattr(runner, 'generate_attempt_split', generate)
    monkeypatch.setattr(runner, 'train', train)
    monkeypatch.setattr(runner, 'predict', predict)
    monkeypatch.setattr(runner, 'predict_prefix', prefix)
    monkeypatch.setattr(runner, '_state', fake_state)
    monkeypatch.setattr(runner, '_baseline_rows', lambda data, regime, check: [
        {'regime': regime, 'horizon': h, 'cases': len(data['prefix']),
         'blind_cost_mse': 0., 'blind_regret': 0.} for h in (1, 2, 4, 8)])
    return events, fits_called


def test_explicit_trainer_and_default_cohort_roster_without_execution():
    assert runner.train is trainer.train
    assert runner.predict_prefix is frozen.predict_prefix
    assert runner._fit_config is frozen._config
    assert runner.ARMS == ('rounded_mse', 'rounded_double', 'rounded_range', 'free_mse', 'free_double', 'free_range')
    assert runner.DEFAULT_CONFIG['cohorts'] == [
        {'seed_namespace': 437260924 + i, 'fit_seed': 437261001 + i} for i in range(5)]
    checked = runner._config({**runner.DEFAULT_CONFIG, **FAKE})
    assert checked == {**runner.DEFAULT_CONFIG, **FAKE}
    assert checked['cohorts'] is not FAKE['cohorts']
    assert checked['cohorts'][0] is not FAKE['cohorts'][0]
    assert runner.cohort_config(checked, 1)['fit_seeds'] == [12]
    assert runner.cohort_config(checked, 1)['seed_namespace'] == 2
    smoke = runner._config({**runner.DEFAULT_CONFIG, **SMOKE})
    assert smoke == {**runner.DEFAULT_CONFIG, **SMOKE}
    assert runner.cohort_config(smoke, 0)['fit_seeds'] == [948101]


def test_all_cohorts_fit_before_any_dev_and_preserve_every_outcome(tmp_path, monkeypatch):
    folder = tmp_path / 'fabricated'
    events, called = patch_fabricated_flow(monkeypatch, folder)
    summary = runner.run(folder, FAKE, lambda: None)
    assert summary['version'] == 'finite-action-range-learning-v1'
    assert summary['implementation'] == 'reuse'
    assert called == [(11, arm) for arm in runner.ARMS] + [(12, arm) for arm in runner.ARMS[1:] + runner.ARMS[:1]]
    assert [event for event in events if event[0] == 'generate'] == [
        ('generate', 1, 0), ('generate', 2, 0), ('generate', 1, 1), ('generate', 1, 2),
        ('generate', 2, 1), ('generate', 2, 2)]
    assert [event for event in events if event[0] == 'reference'] == [
        ('reference', .12), ('reference', .12), ('reference', .30), ('reference', .30)]
    assert len(summary['fits']) == 12 and len(summary['rows']) == 96
    assert len(summary['prefix_rows']) == 24 and len(summary['baseline_rows']) == 16
    assert len(summary['prediction_times']) == len(summary['prefix_prediction_times']) == 24
    assert len(summary['files']) == 152  # Two cohorts * 75, plus root config/barrier.
    assert summary['counts']['fit_count'] == 12
    assert summary['counts']['checkpoint_writes'] == summary['counts']['optimizer_checkpoint_writes'] == 36
    assert summary['counts']['train_generation_count'] == 2 and summary['counts']['dev_generation_count'] == 4
    assert summary['counts']['accepted_prefix_steps'] == 24 and summary['counts']['accepted_joint_steps'] == 36
    assert summary['counts']['oracle_model_constructions'] == 4
    assert summary['counts'] == runner.aggregate_counts(summary['cohorts'])
    assert summary['structural_work'] == runner.aggregate_work(summary['cohorts'])
    assert all(summary['structural_work'][arm]['training_prefix']['fabricated_calls'] == 2 for arm in runner.ARMS)
    assert all(summary['structural_work'][arm]['evaluation_blind']['fabricated_calls'] == 4 for arm in runner.ARMS)
    assert not ({'advance', 'gates', 'selected_arm'} & set(summary))
    barrier = summary['checkpoint_barrier']
    assert barrier['fit_count'] == 12 and len(barrier['checkpoints']) == 12
    assert barrier['dev_generation_count'] == 0 and barrier['oracle_train_verified'] is True
    for cohort in summary['cohorts']:
        assert cohort['dataset_counts']['train']['attempts'] == 4
        assert cohort['dataset_counts']['train']['retained'] == 3
        assert cohort['oracle_metadata']['base']['epsilon'] == .12
        assert cohort['oracle_metadata']['shift']['epsilon'] == .30
        assert cohort['prefix_pair_checks']['all_initial_heads_paired'] is True
        assert cohort['prefix_pair_checks']['all_heads_unchanged_after_prefix'] is True
        assert all(group['same_initial_and_prefix_model'] and group['same_prefix_optimizer_states']
                   for group in cohort['prefix_pair_checks']['groups'])
    assert summary['cohorts'][0]['paired_batch_sha256'] != summary['cohorts'][1]['paired_batch_sha256']
    for row in summary['rows'] + summary['prefix_rows'] + summary['baseline_rows']:
        assert row['seed_namespace'] == summary['cohorts'][row['cohort_index']]['seed_namespace']
    hashes = {(row['cohort_index'], row['arm']): row['final_state_sha256'] for row in summary['fits']}
    assert all(row['model_state_before'] == row['model_state_after'] == hashes[row['cohort_index'], row['arm']]
               for row in summary['prediction_times'])
    for path, descriptor in summary['files'].items():
        assert runner._desc(folder / path) == descriptor
    with pytest.raises(FileExistsError):
        runner.run(folder, FAKE, lambda: None)


@pytest.mark.parametrize('key,value', [
    ('cohorts', []), ('cohorts', [{'seed_namespace': 1, 'fit_seed': 11}] * 2),
    ('cohorts', [{'seed_namespace': 1, 'fit_seed': True}]),
    ('cohorts', [{'seed_namespace': 1, 'fit_seed': 11, 'extra': 1}]),
    ('cohorts', [{'seed_namespace': 1, 'fit_seed': 11}, {'seed_namespace': 2, 'fit_seed': 11}]),
    ('cohorts', [{'seed_namespace': i, 'fit_seed': i} for i in range(6)]),
    ('prefix_updates', 0), ('joint_updates', True), ('fit_cap_seconds', 121.),
    ('fit_seeds', [11]), ('seed_namespace', 1), ('implementation', 'separate'),
])
def test_invalid_settings_fail_before_any_generation_or_output(tmp_path, monkeypatch, key, value):
    def forbidden(*args, **kwargs):
        raise AssertionError('no generation for invalid settings')
    monkeypatch.setattr(runner, 'generate_attempt_split', forbidden)
    with pytest.raises(ValueError):
        runner.run(tmp_path / 'absent', {**FAKE, key: value}, lambda: None)
    assert not (tmp_path / 'absent').exists()


def test_later_cohort_failure_retains_earlier_fits_without_any_dev(tmp_path, monkeypatch):
    folder = tmp_path / 'failed'
    events, called = patch_fabricated_flow(monkeypatch, folder, fail_at=(12, 'rounded_double'))
    with pytest.raises(TimeoutError, match='fabricated original safety-cap stop'):
        runner.run(folder, FAKE, lambda: None)
    assert called == [(11, arm) for arm in runner.ARMS]
    assert all(event[2] == 0 for event in events if event[0] == 'generate')
    failure = json.loads((folder / 'failure.json').read_text())
    assert failure['active_cohort'] == 1 and failure['completed_fits'] == 6
    assert failure['counts']['dev_generation_count'] == 0
    assert failure['failed_action_range_loss_work'] == {'wrapper_calls': 1}
    assert (folder / 'cohort-01' / 'failed-action-range-loss-rounded_double-12.json').is_file()
    assert (folder / 'cohort-00' / 'fits.jsonl').is_file()
    assert not (folder / 'checkpoint-barrier.json').exists()
    assert not (folder / 'summary.json').exists()
    assert not list(folder.rglob('base.npz')) and not list(folder.rglob('shift.npz'))


def test_corrupt_prefix_integrity_stops_before_global_barrier(tmp_path, monkeypatch):
    folder = tmp_path / 'corrupt'
    events, _ = patch_fabricated_flow(monkeypatch, folder, tamper_at=(12, 'free_range'))
    with pytest.raises(ValueError, match='fit head hashes'):
        runner.run(folder, FAKE, lambda: None)
    assert all(event[2] == 0 for event in events if event[0] == 'generate')
    assert not (folder / 'checkpoint-barrier.json').exists()
    assert json.loads((folder / 'failure.json').read_text())['completed_fits'] == 12


def pair_fixture(folder):
    config = runner.cohort_config(runner._config({**runner.DEFAULT_CONFIG, **FAKE}), 0)
    return [fit_record(folder, arm, 11, config) for arm in runner.ARMS], config


def mutate_allocation(folder, fit, mutate):
    path = folder / fit['allocation']['path']
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    fit['allocation'].update(runner._desc(path))


def test_pairing_reads_only_json_and_opaque_descriptors(tmp_path, monkeypatch):
    fits, config = pair_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('no checkpoint decode in producer pairing')
    monkeypatch.setattr(np, 'load', forbidden)
    assert len(runner.verify_paired_exposure(fits, tmp_path, config)) == 64
    paired = runner.verify_prefix_pairing(fits, tmp_path, config)
    assert paired['all_initial_heads_paired'] and paired['all_heads_unchanged_after_prefix']
    assert {group['transport_arm'] for group in paired['groups']} == {'rounded', 'matched_free'}


@pytest.mark.parametrize('arm', runner.ARMS)
def test_each_loss_arm_joint_schedule_is_checked(tmp_path, arm):
    fits, config = pair_fixture(tmp_path)
    fit = next(row for row in fits if row['arm'] == arm)
    mutate_allocation(tmp_path, fit, lambda value: value['trace'][-1]['result']['diagnostics'].update(indices=[99]))
    with pytest.raises(ValueError, match='identical ordered joint minibatches'):
        runner.verify_paired_exposure(fits, tmp_path, config)


@pytest.mark.parametrize('arm', ('rounded_double', 'rounded_range', 'free_double', 'free_range'))
@pytest.mark.parametrize('label,key', [('initial', 'model_state_sha256'), ('boundary', 'model_state_sha256'),
                                     ('initial', 'optimizer_state_sha256'), ('boundary', 'optimizer_state_sha256')])
def test_every_architecture_local_loss_pair_is_checked(tmp_path, arm, label, key):
    fits, config = pair_fixture(tmp_path)
    fit = next(row for row in fits if row['arm'] == arm)
    def change(value):
        next(row for row in value['checkpoints'] if row['label'] == label)['metadata'][key] = 'f' * 64
    mutate_allocation(tmp_path, fit, change)
    if label == 'initial' and key == 'model_state_sha256':
        fit['initial_state_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='same-architecture loss variants'):
        runner.verify_prefix_pairing(fits, tmp_path, config)


def test_common_initial_head_and_unchanged_prefix_head_are_required(tmp_path):
    fits, config = pair_fixture(tmp_path)
    fit = fits[-1]
    def change(value):
        for row in value['checkpoints'][:2]:
            row['metadata']['head_state_sha256'] = 'f' * 64
    mutate_allocation(tmp_path, fit, change)
    fit['head_boundary_sha256'].update(initial='f' * 64, boundary='f' * 64)
    with pytest.raises(ValueError, match='all six initial random heads'):
        runner.verify_prefix_pairing(fits, tmp_path, config)
    fits, config = pair_fixture(tmp_path)
    fit = fits[0]
    mutate_allocation(tmp_path, fit, lambda value: value['checkpoints'][1]['metadata'].update(head_state_sha256='f' * 64))
    fit['head_boundary_sha256']['boundary'] = 'f' * 64
    with pytest.raises(ValueError, match='every head remains unchanged'):
        runner.verify_prefix_pairing(fits, tmp_path, config)
