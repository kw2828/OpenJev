"""Engineering namespaces only; independent saved-output arithmetic and joins."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_observation_learning as a
import run_finite_observation_learning as r


def test_five_arm_engineering_run_preserves_predev_barrier_and_final_states(tmp_path, monkeypatch):
    folder = tmp_path / 'engineering-only'
    generate = r.generate_split
    calls = []

    def engineering_generate(split_id, attempts, horizon, **kwargs):
        assert kwargs == {'seed_namespace': 919001}
        assert attempts == 8
        calls.append(split_id)
        if split_id:
            barrier = json.loads((folder / 'checkpoint-barrier.json').read_text())
            fits = [json.loads(line) for line in (folder / 'fits.jsonl').read_text().splitlines()]
            assert barrier['fit_count'] == len(fits) == 5
            assert barrier['dev_generation_count'] == 0
            assert {(row['arm'], row['seed']) for row in fits} == {(arm, 919101) for arm in a.ARMS}
            for item in barrier['checkpoints']:
                raw = (folder / item['path']).read_bytes()
                assert len(raw) == item['bytes']
                assert hashlib.sha256(raw).hexdigest() == item['sha256']
        return generate(split_id, attempts, horizon, **kwargs)

    monkeypatch.setattr(r, 'generate_split', engineering_generate)
    config = {'seed_namespace': 919001, 'train_attempts': 8, 'dev_attempts': 8,
              'epochs': 1, 'batch_size': 8, 'fit_seeds': [919101]}
    result = r.run(folder, config, lambda: None)
    assert calls == [0, 1, 2]
    assert len(result['rows']) == 40 and len(result['baseline_rows']) == 8
    assert {(row['arm'], row['seed'], row['regime'], row['horizon']) for row in result['rows']} == {
        (arm, 919101, regime, horizon) for arm in a.ARMS for regime in ('base', 'shift') for horizon in a.HORIZONS}
    fits = {row['arm']: row for row in result['fits']}
    n = result['dataset_counts']['train']['retained']
    for fit in fits.values():
        assert fit['epochs'] == fit['updates'] == 1
        assert fit['training_cases'] == fit['training_case_exposures'] == n
    assert len({fit['case_order_sha256'] for fit in fits.values()}) == 1
    for init in ('dense', 'retentive'):
        assert fits['tied_' + init]['shared_initial_state_sha256'] == fits['untied_' + init]['shared_initial_state_sha256']
    for row in result['prediction_times']:
        assert row['model_state_before'] == row['model_state_after'] == fits[row['arm']]['final_state_sha256']
    counts = result['counts']
    for key in ('fit_count', 'model_constructions', 'checkpoint_writes', 'optimizer_attempts',
                'optimizer_steps', 'training_blind_rollouts', 'training_observed_rollouts'):
        assert counts[key] == 5
    assert counts['training_case_exposures'] == 5 * n
    for key in ('array_decodes', 'checkpoint_decodes', 'external_model_calls', 'native_calls', 'teacher_calls'):
        assert counts[key] == 0
    for regime in ('base', 'shift'):
        with np.load(folder / f'{regime}.npz', allow_pickle=False) as data:
            assert set(data.files) == a.DATA_KEYS
            assert data['prefix'].dtype == np.float32
            assert all(str(case).startswith(f'ns919001-split{1 if regime == "base" else 2}-') for case in data['case_ids'])
            cases = len(data['case_ids'])
        with np.load(folder / f'predictions-{regime}.npz', allow_pickle=False) as predictions:
            assert set(predictions.files) == {f'{arm}__919101__{field}' for arm in a.ARMS for field in a.FIELDS}
            for name in predictions.files:
                field = name.rsplit('__', 1)[1]
                expected = (cases, 8, 5) if field == 'observed_probabilities' else (cases, 8) if field.endswith('survival') else (cases, 8, 4)
                assert predictions[name].shape == expected
                assert predictions[name].dtype == np.float64
                assert np.isfinite(predictions[name]).all()
    with pytest.raises(FileExistsError):
        r.run(folder, config, lambda: None)
    assert calls == [0, 1, 2]


def arithmetic_fixture():
    truth = np.array([[-.3, -.1, .1, .3], [.3, -.3, -.1, .1]])
    predicted = np.array([[-.1, -.3, .1, .3], [.3, -.1, -.3, .1]])
    expanded = lambda x: np.repeat(x[:, None], 8, axis=1)
    data = {'case_ids': np.array(['fabricated-a', 'fabricated-b']),
            'blind_costs': expanded(truth), 'observed_costs': expanded(truth),
            'blind_survival': expanded(np.array([.8, .8])),
            'observed_survival': expanded(np.array([.9, .8])),
            'observed_probabilities': np.tile([.1, .2, .3, .35, .05], (2, 8, 1))}
    fields = {'blind_costs': expanded(predicted), 'observed_costs': expanded(predicted),
              'blind_survival': expanded(np.array([.7, .9])),
              'observed_survival': expanded(np.array([.8, .8])),
              'observed_probabilities': data['observed_probabilities'].copy(),
              'shuffled_blind_costs': np.tile([0., .1, .2, -.3], (2, 8, 1))}
    predictions = {f'{arm}__{seed}__{field}': value.copy()
                   for arm in a.ARMS for seed in a.SEEDS for field, value in fields.items()}
    uniform = expanded(np.array([[0., 5e-13, 1., 1.], [5e-13, 0., 1., 1.]]))
    return data, predictions, uniform


def test_independent_hand_arithmetic_and_reference_only_tie_rule():
    data, predictions, uniform = arithmetic_fixture()
    rows, baselines = a.rows_for(data, predictions, uniform, 'base', np)
    assert len(rows) == 60 and len(baselines) == 4
    for row in rows:
        assert row['blind_cost_mse'] == pytest.approx(.02)
        assert row['observed_cost_mse'] == pytest.approx(.02)
        assert row['blind_regret'] == pytest.approx(.2)
        assert row['shuffled_blind_regret'] == pytest.approx(.5)
        assert row['blind_survival_mae'] == pytest.approx(.1)
        assert row['observed_survival_mae'] == pytest.approx(.05)
        assert row['observed_kl'] == 0
    # Raw argmin would choose action1 in the second reference row and give0.
    assert all(row['blind_regret'] == pytest.approx(.3) for row in baselines)


@pytest.mark.parametrize('bad', ['float32', 'zero_support', 'missing_model'])
def test_saved_prediction_contract_rejects_invalid_support_and_roster(bad):
    data, predictions, _ = arithmetic_fixture()
    key = f'{a.ARMS[0]}__{a.SEEDS[0]}__observed_probabilities'
    if bad == 'float32':
        predictions[key] = predictions[key].astype(np.float32)
    elif bad == 'zero_support':
        predictions[key][..., 1] += predictions[key][..., 0]
        predictions[key][..., 0] = 0
    else:
        predictions.pop(key)
    with pytest.raises(ValueError):
        a.validate_predictions(data, predictions, np)


def test_h4_and_h8_gates_cannot_average_away_one_bad_seed_or_zero_reference():
    rows = [{'arm': arm, 'seed': seed, 'regime': regime, 'horizon': h,
             'blind_cost_mse': .01, 'blind_regret': .02, 'blind_survival_mae': .01, 'observed_kl': .01}
            for arm in a.ARMS for seed in a.SEEDS for regime in ('base', 'shift') for h in a.HORIZONS]
    bases = [{'regime': regime, 'horizon': h, 'blind_cost_mse': .1, 'blind_regret': .1}
             for regime in ('base', 'shift') for h in a.HORIZONS]
    counts = {'train': 300, 'base': 80, 'shift': 80}
    assert all(result['passed'] for result in a.gates(rows, bases, counts).values())
    failing = next(row for row in rows if (row['arm'], row['seed'], row['regime'], row['horizon'])
                   == ('tied_retentive', a.SEEDS[0], 'base', 4))
    failing['blind_regret'] = .051
    assert not a.gates(rows, bases, counts)['base']['passed']
    assert a.gates(rows, bases, counts)['shift']['passed']
    failing['blind_regret'] = .02
    next(row for row in bases if (row['regime'], row['horizon']) == ('base', 4))['blind_cost_mse'] = 0.
    assert not a.gates(rows, bases, counts)['base']['passed']


def metadata_fixture(folder):
    """Only fabricated JSON and opaque bytes, never model or dataset arrays."""
    def write(name, value, lines=False):
        (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))

    n = 2
    order_digest = hashlib.sha256(struct.pack('<qq', 0, 1) * 48).hexdigest()
    fits, orders, epochs = [], [], []
    for offset, seed in enumerate(a.SEEDS):
        for arm in a.ARMS[offset:] + a.ARMS[:offset]:
            filename = f'{arm}-{seed}.npz'
            (folder / filename).write_bytes(b'fabricated opaque checkpoint')
            raw = (folder / filename).read_bytes()
            checkpoint = {'path': filename, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            fits.append({'arm': arm, 'seed': seed, 'epochs': 48, 'updates': 48,
                         'training_cases': n, 'training_case_exposures': 96, 'seconds': 1.,
                         **{key: '1' * 64 for key in ('initial_state_sha256', 'shared_initial_state_sha256', 'final_state_sha256')},
                         'case_order_sha256': order_digest, 'checkpoint': checkpoint})
            for epoch in range(48):
                orders.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'indices': [0, 1], 'batch_size': 64})
                epochs.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'cases': n, 'updates': 1,
                               'mean_objective': .5, 'seconds': .1})
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'fit_count': 15, 'dev_generation_count': 0}
    times = [{'regime': regime, 'arm': arm, 'seed': seed, 'seconds': .1, 'cases': n, 'batch_size': 64,
              'model_state_before': '1' * 64, 'model_state_after': '1' * 64}
             for regime in ('base', 'shift') for arm in a.ARMS for seed in a.SEEDS]
    dataset_counts = {}
    for split_id, split in enumerate(('train', 'base', 'shift')):
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        excluded = attempts - n
        dataset_counts[split] = {'version': 'finite-observation-world-v1',
            'epsilon': .30 if split == 'shift' else .12,
            'seed_namespace': a.NAMESPACE, 'split_id': split_id, 'attempts': attempts,
            'horizon': horizon, 'retained': n, 'excluded_found': excluded,
            'prefix_found_by_step': [excluded] + [0] * 7, 'initial_odor_draws': attempts,
            'prefix_action_draws': 8 * attempts, 'prefix_event_draws': 8 * n + excluded,
            'forecast_action_draws': horizon * n}
    counts = {'train_generation_count': 1, 'dev_generation_count': 2, 'model_constructions': 15,
        'fit_count': 15, 'training_blind_rollouts': 720, 'training_observed_rollouts': 720,
        'optimizer_attempts': 720, 'optimizer_steps': 720, 'training_case_exposures': 1440,
        'checkpoint_writes': 15, 'evaluation_blind_rollouts': 30, 'evaluation_observed_rollouts': 30,
        'evaluation_shuffled_rollouts': 30, 'evaluation_case_views': 60, 'array_decodes': 0,
        'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0}
    for name in ('train.npz', 'base.npz', 'shift.npz', 'predictions-base.npz', 'predictions-shift.npz'):
        (folder / name).write_bytes(b'opaque fabricated data, not a NumPy archive')
    write('config.json', a.CONFIG)
    write('fits.jsonl', fits, True)
    write('training-orders.jsonl', orders, True)
    write('training-epochs.jsonl', epochs, True)
    write('checkpoint-barrier.json', barrier)
    write('prediction-times.jsonl', times, True)
    summary = {'version': 'finite-observation-learning-v1', 'config': copy.deepcopy(a.CONFIG),
        'fits': fits, 'checkpoint_barrier': barrier, 'prediction_times': times,
        'dataset_counts': dataset_counts, 'counts': counts,
        'files': {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                  for path in folder.iterdir()}}
    return summary


def test_metadata_audit_hashes_opaque_checkpoints_and_reconciles_complete_schedule(tmp_path):
    summary = metadata_fixture(tmp_path)
    result = a.metadata(tmp_path, summary)
    assert result['fits'] == result['checkpoint_files_hashed'] == 15
    assert result['epochs'] == result['optimizer_steps_attested'] == 720
    assert result['evaluation_views'] == 30
    assert result['historical_ordering_independently_replayed'] is False


@pytest.mark.parametrize('bad', ['checkpoint_bytes', 'missing_fit', 'barrier_dev', 'inference_state', 'optimizer_count', 'order'])
def test_metadata_audit_rejects_broken_saved_joins(tmp_path, bad):
    summary = metadata_fixture(tmp_path)
    if bad == 'checkpoint_bytes':
        (tmp_path / summary['fits'][0]['checkpoint']['path']).write_bytes(b'tampered')
    elif bad == 'missing_fit':
        summary['fits'].pop()
    elif bad == 'barrier_dev':
        summary['checkpoint_barrier']['dev_generation_count'] = 1
    elif bad == 'inference_state':
        summary['prediction_times'][0]['model_state_after'] = '2' * 64
    elif bad == 'optimizer_count':
        summary['counts']['optimizer_steps'] -= 1
    else:
        path = tmp_path / 'training-orders.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[48]['indices'] = [1, 0]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary['files'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    with pytest.raises(ValueError):
        a.metadata(tmp_path, summary)


def test_auditor_rejects_incomplete_metadata_before_array_decode(tmp_path, monkeypatch):
    summary = metadata_fixture(tmp_path)
    summary['counts']['optimizer_steps'] = math.inf
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    def forbidden_load(*args, **kwargs):
        pytest.fail('metadata failure must precede every array decode')
    monkeypatch.setattr(np, 'load', forbidden_load)
    with pytest.raises(ValueError):
        a.audit(tmp_path)
