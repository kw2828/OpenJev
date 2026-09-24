"""Hand-built finite-world data, independent arithmetic, and factor gate rules."""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_finite_factor_learning as a


def hand_case(split='base'):
    """Enumerate rational source/destination pairs, independent of audit helpers."""
    def advance(state, action):
        branches = [[F(0) for _ in range(8)] for _ in range(4)]
        found = F(0)
        for current in range(8):
            destination = [current ^ 1, (current + 1) % 8,
                           ((current << 1) | (current >> 2)) & 7, current ^ 4][action]
            for following in range(8):
                mass = state[current] * F(393 if following == destination else 1, 400)
                hazard = F(1 + ((following >> 2) ^ (action & 1)), 200)
                found += mass * hazard
                for odor in range(4):
                    branches[odor][following] += mass * (1 - hazard) * F(22 if odor == (following & 3) else 1, 25)
        return branches, [sum(branch) for branch in branches] + [found]

    def costs(state):
        return [sum(mass * (F(-3, 4) if decision == ((s ^ (s >> 1)) & 3) else F(1, 4))
                    for s, mass in enumerate(state)) for decision in range(4)]

    horizon = 2 if split == 'train' else 8
    prefix = np.zeros((1, 9, 31), np.float32)
    prefix[0, 0, 4] = prefix[0, 0, 9] = 1
    state = [F(22 if s % 4 == 0 else 1, 50) for s in range(8)]
    for step in range(8):
        action, odor = step % 4, (step + 1) % 4
        prefix[0, step + 1, action] = prefix[0, step + 1, 4 + odor] = 1
        branches, masses = advance(state, action)
        state = [v / masses[odor] for v in branches[odor]]
    prefix_state = np.array([[float(v) for v in state]], np.float64)
    actions = np.array([[step % 4 for step in range(horizon)]], np.int64)
    observations = np.array([[0] + [4] * (horizon - 1)], np.int64)
    arrays = {key: np.zeros((1, horizon, 5 if key == 'observed_probabilities' else 4), np.float64)
              if key.endswith('costs') or key == 'observed_probabilities'
              else np.zeros((1, horizon), np.float64) for key in a.TARGETS}
    blind, observed, absorbed = state.copy(), state.copy(), False
    for step, action in enumerate(actions[0]):
        branches, _ = advance(blind, int(action))
        blind = [sum(branch[s] for branch in branches) for s in range(8)]
        arrays['blind_costs'][0, step] = [float(v) for v in costs(blind)]
        arrays['blind_survival'][0, step] = float(sum(blind))
        if absorbed:
            arrays['observed_probabilities'][0, step, 4] = 1
            continue
        branches, masses = advance(observed, int(action))
        prior = [sum(branch[s] for branch in branches) for s in range(8)]
        arrays['observed_costs'][0, step] = [float(v) for v in costs(prior)]
        arrays['observed_survival'][0, step] = float(sum(prior))
        arrays['observed_probabilities'][0, step] = [float(v) for v in masses]
        odor = int(observations[0, step])
        if odor == 4:
            absorbed = True
        else:
            observed = [v / masses[odor] for v in branches[odor]]
    data = {'prefix': prefix, 'lengths': np.array([9], np.int64), 'actions': actions,
            'observations': observations,
            'case_ids': np.array([f'ns{a.NAMESPACE}-split{0 if split == "train" else 1}-case0000000000']),
            **arrays}
    return data, {'prefix_state': prefix_state}, {key: value.copy() for key, value in arrays.items()}


@pytest.mark.parametrize('split', ['train', 'base'])
def test_exact_control_and_prefix_match_independent_rational_pair_enumeration(split):
    data, prefix, oracle = hand_case(split)
    record, targets, uniform = a.reconstruct(data, prefix, oracle, split, np)
    assert record['prefix_maximum_absolute_error'] <= 1e-12
    assert max(record['target_maximum_absolute_errors'].values()) <= 1e-12
    assert max(record['oracle_maximum_absolute_errors'].values()) <= 1e-12
    assert uniform.shape == data['blind_costs'].shape
    # The found label at horizon2 cannot erase its pre-observation forecast.
    assert targets['observed_survival'][0, 1] > .9
    if split == 'base':
        assert np.all(targets['observed_survival'][0, 2:] == 0)
        assert np.all(targets['observed_costs'][0, 2:] == 0)
        assert np.all(targets['observed_probabilities'][0, 2:, 4] == 1)
        assert np.all(targets['blind_survival'][0, 2:] > .9)


@pytest.mark.parametrize('bad', ['oracle_input', 'oracle_output', 'target', 'namespace', 'oracle_dtype', 'extra_field'])
def test_reconstruction_rejects_incorrect_privileged_inputs_outputs_and_identity(bad):
    data, prefix, oracle = hand_case()
    if bad == 'oracle_input':
        prefix['prefix_state'][0, :2] += [1e-5, -1e-5]
    elif bad == 'oracle_output':
        oracle['blind_costs'][0, 3, 0] += 1e-5
    elif bad == 'target':
        data['observed_costs'][0, 1] = 0
    elif bad == 'namespace':
        data['case_ids'] = np.array(['ns420260924-split1-case0000000000'])
    elif bad == 'oracle_dtype':
        prefix['prefix_state'] = prefix['prefix_state'].astype(np.float32)
    else:
        oracle['shuffled_blind_costs'] = oracle['blind_costs'].copy()
    with pytest.raises(ValueError):
        a.reconstruct(data, prefix, oracle, 'base', np)


def arithmetic_fixture():
    expanded = lambda value: np.repeat(np.array(value, np.float64)[:, None], 8, axis=1)
    truth = expanded([[-.3, -.1, .1, .3], [.3, -.3, -.1, .1]])
    choice = expanded([[-.1, -.3, .1, .3], [.3, -.1, -.3, .1]])
    data = {'case_ids': np.array(['hand-a', 'hand-b']), 'blind_costs': truth, 'observed_costs': truth.copy(),
            'blind_survival': expanded([.8, .8]), 'observed_survival': expanded([.9, .8]),
            'observed_probabilities': np.tile([.1, .2, .3, .35, .05], (2, 8, 1))}
    fields = {'blind_costs': choice, 'observed_costs': choice.copy(),
              'blind_survival': expanded([.7, .9]), 'observed_survival': expanded([.8, .8]),
              'observed_probabilities': data['observed_probabilities'].copy(),
              'shuffled_blind_costs': np.tile([0., .1, .2, -.3], (2, 8, 1))}
    predictions = {f'{arm}__{seed}__{field}': value.copy()
                   for arm in a.ARMS for seed in a.SEEDS for field, value in fields.items()}
    reference = expanded([[0., 5e-13, 1., 1.], [5e-13, 0., 1., 1.]])
    return data, predictions, reference


def test_hand_metric_denominators_argmin_and_reference_only_ties():
    data, predictions, reference = arithmetic_fixture()
    rows, baselines = a.rows_for(data, predictions, reference, np)
    assert len(rows) == 48 and len(baselines) == 4
    for row in rows:
        for name, expected in {'blind_cost_mse': .02, 'observed_cost_mse': .02, 'blind_regret': .2,
                               'shuffled_blind_regret': .5, 'blind_survival_mae': .1,
                               'observed_survival_mae': .05, 'observed_kl': 0}.items():
            assert row[name] == pytest.approx(expected)
    assert all(row['blind_regret'] == pytest.approx(.3) for row in baselines)


@pytest.mark.parametrize('bad', ['float32', 'missing_model', 'zero_support'])
def test_prediction_schema_requires_twelve_models_float64_and_positive_oracle_support(bad):
    data, predictions, _ = arithmetic_fixture()
    key = f'{a.ARMS[0]}__{a.SEEDS[0]}__observed_probabilities'
    if bad == 'float32':
        predictions[key] = predictions[key].astype(np.float32)
    elif bad == 'missing_model':
        del predictions[key]
    else:
        predictions[key][..., 1] += predictions[key][..., 0]
        predictions[key][..., 0] = 0
    with pytest.raises(ValueError):
        a.validate_predictions(data, predictions, np)


def gate_fixture():
    rows = [{'arm': arm, 'seed': seed, 'regime': 'base', 'horizon': h,
             'blind_cost_mse': .01, 'blind_regret': .02, 'blind_survival_mae': .01, 'observed_kl': .01}
            for arm in a.ARMS for seed in a.SEEDS for h in a.HORIZONS]
    baselines = [{'regime': 'base', 'horizon': h, 'blind_cost_mse': .1, 'blind_regret': .1} for h in a.HORIZONS]
    return rows, baselines, {'train': 300, 'base': 80}


def test_observed_filtering_cannot_be_reported_as_blind_extrapolation():
    rows, baselines, counts = gate_fixture()
    arm = 'learned_learned'
    row = next(row for row in rows if (row['arm'], row['seed'], row['horizon']) == (arm, a.SEEDS[0], 4))
    row['observed_kl'] = .11
    result = a.gates(rows, baselines, counts)
    assert set(result) == set(a.FACTOR_ARMS)
    assert result[arm]['SHORT_HORIZON_LEARNING']['passed']
    assert result[arm]['BLIND_EXTRAPOLATION']['passed']
    assert not result[arm]['OBSERVED_FILTERING_EXTRAPOLATION']['passed']
    row['observed_kl'], row['blind_regret'] = .01, .051
    result = a.gates(rows, baselines, counts)
    assert result[arm]['SHORT_HORIZON_LEARNING']['passed']
    assert not result[arm]['BLIND_EXTRAPOLATION']['passed']
    assert result[arm]['OBSERVED_FILTERING_EXTRAPOLATION']['passed']


@pytest.mark.parametrize('bad', ['short_seed', 'short_kl', 'h8_survival', 'zero_reference', 'support', 'duplicate_row'])
def test_gates_require_each_seed_and_horizon_and_strictly_positive_reference(bad):
    rows, baselines, counts = gate_fixture()
    original = copy.deepcopy(a.gates(rows, baselines, counts))
    assert all(criterion['passed'] for result in original.values() for criterion in result.values())
    selected = lambda h: next(row for row in rows if (row['arm'], row['seed'], row['horizon'])
                             == ('exact_learned', a.SEEDS[1], h))
    if bad == 'short_seed':
        selected(2)['blind_cost_mse'] = .051
    elif bad == 'short_kl':
        selected(1)['observed_kl'] = .11
    elif bad == 'h8_survival':
        selected(8)['blind_survival_mae'] = .051
    elif bad == 'zero_reference':
        baselines[0]['blind_regret'] = 0
    elif bad == 'support':
        counts['base'] = 63
    else:
        rows[-1] = rows[0].copy()
        with pytest.raises(ValueError):
            a.gates(rows, baselines, counts)
        return
    result = a.gates(rows, baselines, counts)
    assert any(not criterion['passed'] for criterion in result['exact_learned'].values())


def fabricated_metadata(folder):
    """Full declared roster, but all checkpoint/data bytes are opaque fixtures."""
    def descriptor(path):
        return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

    def write(name, value, *, lines=False):
        (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))

    fits, orders, epochs = [], [], []
    order_hash = hashlib.sha256(struct.pack('<qq', 0, 1) * 480).hexdigest()
    for index, seed in enumerate(a.SEEDS):
        for arm in a.ARMS[index:] + a.ARMS[:index]:
            name = f'{arm}-{seed}.npz'
            (folder / name).write_bytes(b'opaque fabricated checkpoint, never decoded')
            fits.append({'arm': arm, 'seed': seed, 'epochs': 480, 'updates': 480, 'training_cases': 2,
                'training_case_exposures': 960, 'seconds': 1., 'case_order_sha256': order_hash,
                **{key: '1' * 64 for key in ('initial_state_sha256', 'final_state_sha256', 'fixed_buffers_sha256')},
                'prefix_initial_sha256': '2' * 64 if arm in ('learned_exact', 'learned_learned') else None,
                'operator_initial_sha256': '3' * 64 if arm in ('exact_learned', 'learned_learned') else None,
                'oracle_prefix_input': arm == 'exact_learned', 'checkpoint': {'path': name, **descriptor(folder / name)}})
            for epoch in range(480):
                orders.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'indices': [0, 1], 'batch_size': 64})
                epochs.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'cases': 2, 'updates': 1,
                               'mean_objective': .5, 'seconds': .1})
    times = [{'arm': arm, 'seed': seed, 'regime': 'base', 'seconds': .1, 'cases': 2, 'batch_size': 64,
              'oracle_prefix_input': arm == 'exact_learned', 'shuffle_offset': 1,
              'model_state_before': '1' * 64, 'model_state_after': '1' * 64}
             for arm in a.ARMS for seed in a.SEEDS]
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': 12, 'oracle_train_verified': True}
    dataset_counts, oracle_checks = {}, {}
    for split_id, split in enumerate(('train', 'base')):
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        dataset_counts[split] = {'version': 'finite-observation-world-v1', 'epsilon': .12,
            'seed_namespace': a.NAMESPACE, 'split_id': split_id, 'attempts': attempts, 'horizon': horizon,
            'retained': 2, 'excluded_found': attempts - 2, 'prefix_found_by_step': [attempts - 2] + [0] * 7,
            'initial_odor_draws': attempts, 'prefix_action_draws': 8 * attempts,
            'prefix_event_draws': 16 + attempts - 2, 'forecast_action_draws': 2 * horizon}
        oracle_checks[split] = dict.fromkeys(a.TARGETS, 0.)
        write('oracle-' + split + '-check.json', oracle_checks[split])
    counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': 12,
        'oracle_model_constructions': 1, 'fit_count': 12, 'checkpoint_writes': 12,
        'training_blind_rollouts': 5760, 'training_observed_rollouts': 5760,
        'optimizer_attempts': 5760, 'optimizer_steps': 5760, 'training_case_exposures': 11520,
        'evaluation_blind_rollouts': 12, 'evaluation_observed_rollouts': 12,
        'evaluation_shuffled_rollouts': 12, 'evaluation_case_views': 24,
        'oracle_blind_rollouts': 2, 'oracle_observed_rollouts': 2,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0}
    for name in ('train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
                 'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz'):
        (folder / name).write_bytes(b'opaque fake array, never decoded')
    write('config.json', a.CONFIG)
    write('fits.jsonl', fits, lines=True)
    write('training-orders.jsonl', orders, lines=True)
    write('training-epochs.jsonl', epochs, lines=True)
    write('checkpoint-barrier.json', barrier)
    write('prediction-times.jsonl', times, lines=True)
    return {'version': 'finite-factor-learning-v1', 'config': copy.deepcopy(a.CONFIG),
            'fits': fits, 'prediction_times': times, 'checkpoint_barrier': barrier,
            'dataset_counts': dataset_counts, 'counts': counts, 'oracle_checks': oracle_checks,
            'oracle_metadata': {'parameters': {}, 'parameter_count': 0, 'parameter_bytes': 0},
            'oracle_state_sha256': '4' * 64,
            'files': {path.name: descriptor(path) for path in folder.iterdir()}}


def test_metadata_joins_all_twelve_fits_and_5760_epochs_without_decoding(tmp_path, monkeypatch):
    summary = fabricated_metadata(tmp_path)
    def forbidden_load(*args, **kwargs):
        pytest.fail('opaque metadata checks cannot decode numerical arrays')
    monkeypatch.setattr(np, 'load', forbidden_load)
    result = a.validate_metadata(tmp_path, summary)
    assert result['fits'] == result['checkpoint_files_hashed'] == result['evaluation_views'] == 12
    assert result['epochs'] == result['optimizer_steps_attested'] == 5760
    assert result['training_case_exposures_attested'] == 11520
    assert result['historical_ordering_independently_replayed'] is False


@pytest.mark.parametrize('bad', ['checkpoint', 'oracle_train_barrier', 'missing_fit', 'group_pair', 'oracle_fit_count'])
def test_metadata_rejects_incomplete_or_unpaired_producer_before_any_decode(tmp_path, monkeypatch, bad):
    summary = fabricated_metadata(tmp_path)
    if bad == 'checkpoint':
        (tmp_path / summary['fits'][0]['checkpoint']['path']).write_bytes(b'tampered')
    elif bad == 'oracle_train_barrier':
        summary['checkpoint_barrier']['oracle_train_verified'] = False
    elif bad == 'missing_fit':
        summary['fits'].pop()
    elif bad == 'group_pair':
        summary['fits'][0]['prefix_initial_sha256'] = '9' * 64
        path = tmp_path / 'fits.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in summary['fits']))
        summary['files'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    else:
        summary['counts']['oracle_model_constructions'] = 3
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    def forbidden_load(*args, **kwargs):
        pytest.fail('invalid metadata must fail before numerical decoding')
    monkeypatch.setattr(np, 'load', forbidden_load)
    with pytest.raises(ValueError):
        a.audit(tmp_path)
