"""Hand-built finite-world data, independent arithmetic, and prefix-supervision gate rules."""
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
import audit_finite_prefix_learning as a


@pytest.fixture(autouse=True)
def engineering_namespace(monkeypatch):
    monkeypatch.setattr(a, 'NAMESPACE', 931001)
    monkeypatch.setattr(a, 'SEEDS', (931101, 931102, 931103))
    monkeypatch.setattr(a, 'CONFIG', {**a.CONFIG, 'seed_namespace': 931001, 'fit_seeds': [931101, 931102, 931103]})

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
    assert len(rows) == 24 and len(baselines) == 4
    for row in rows:
        for name, expected in {'blind_cost_mse': .02, 'observed_cost_mse': .02, 'blind_regret': .2,
                               'shuffled_blind_regret': .5, 'blind_survival_mae': .1,
                               'observed_survival_mae': .05, 'observed_kl': 0}.items():
            assert row[name] == pytest.approx(expected)
    assert all(row['blind_regret'] == pytest.approx(.3) for row in baselines)


@pytest.mark.parametrize('bad', ['float32', 'missing_model', 'zero_support'])
def test_prediction_schema_requires_six_models_float64_and_positive_oracle_support(bad):
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
    arm = 'endpoint_only'
    row = next(row for row in rows if (row['arm'], row['seed'], row['horizon']) == (arm, a.SEEDS[0], 4))
    row['observed_kl'] = .11
    result = a.gates(rows, baselines, counts)
    assert set(result) == set(a.ARMS)
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
                             == ('endpoint_plus_prefix', a.SEEDS[1], h))
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
    assert any(not criterion['passed'] for criterion in result['endpoint_plus_prefix'].values())


def prefix_fixture(split='base', *, two_survivors=False):
    n = 512 if split == 'train' else 128
    data, oracle, exact = hand_case(split)
    attempted = np.repeat(data['prefix'], n, axis=0)
    eligible = np.ones(n, bool)
    lengths = np.full(n, 9, np.int64)
    if two_survivors:
        found_indices = np.arange(2, n)
        attempted[found_indices, 1:] = 0
        attempted[found_indices, 1, 0] = attempted[found_indices, 1, 8] = 1
        eligible[found_indices], lengths[found_indices] = False, 2
    else:
        # Found at the first action and at the last action must both be excluded.
        attempted[0, 1:] = 0
        attempted[0, 1, 0] = attempted[0, 1, 8] = 1
        attempted[1, 8, 4:9] = 0
        attempted[1, 8, 8] = 1
        eligible[:2], lengths[0] = False, 2
    ids = np.array([f'ns{a.NAMESPACE}-split{int(split == "base")}-case{i:010d}' for i in range(n)], dtype='U64')
    mapping = np.full(n, -1, np.int64)
    mapping[eligible] = np.arange(int(eligible.sum()))
    prefixes = {'prefix': attempted, 'lengths': lengths, 'case_ids': ids,
        'event_mask': np.arange(9)[None] < lengths[:, None], 'endpoint_eligible': eligible, 'endpoint_rows': mapping}
    endpoints = {key: np.repeat(value, int(eligible.sum()), axis=0) for key, value in data.items()}
    endpoints['case_ids'] = ids[eligible].copy()
    return prefixes, endpoints, {key: np.repeat(value, int(eligible.sum()), axis=0) for key, value in oracle.items()}, exact


def test_all_attempts_include_first_found_and_exact_survivor_identity():
    prefixes, data, oracle, _ = prefix_fixture()
    record, posterior, probabilities = a.reconstruct_prefix(prefixes, data, 'base', np)
    assert record['attempts'] == 128 and record['retained'] == 126
    assert record['valid_events'] == 128 * 9 - 7
    assert record['prefix_found_by_step'] == [1, 0, 0, 0, 0, 0, 0, 1]
    np.testing.assert_allclose(posterior, oracle['prefix_state'], atol=1e-12, rtol=0)
    # Initial odor0 leaves equal mass in both bit2 halves; action xor1 does not
    # alter that balance. The next found probability is (.005+.010)/2.
    assert probabilities[0, 0, 4] == 0
    assert probabilities[0, 1, 4] == pytest.approx(3 / 400, abs=1e-15)
    assert np.all(probabilities[0, 2:] == 0)


@pytest.mark.parametrize('bad', ['drop_attempt', 'exclude_found_event', 'padding', 'found_twice',
                               'false_survivor', 'wrong_endpoint', 'wrong_map', 'length_dtype', 'id_order'])
def test_prefix_population_and_causal_geometry_guards(bad):
    p, data, _oracle, _ = prefix_fixture()
    if bad == 'drop_attempt':
        p = {key: value[:-1] for key, value in p.items()}
    elif bad == 'exclude_found_event':
        p['event_mask'][0, 1] = False
    elif bad == 'padding':
        p['prefix'][0, 2, 0] = 1
    elif bad == 'found_twice':
        p['prefix'][1, 2, 4:9] = [0, 0, 0, 0, 1]
    elif bad == 'false_survivor':
        p['endpoint_eligible'][1] = True
    elif bad == 'wrong_endpoint':
        data['prefix'][0, 0, 4:8] = [0, 1, 0, 0]
    elif bad == 'wrong_map':
        p['endpoint_rows'][2] = 1
    elif bad == 'length_dtype':
        p['lengths'] = p['lengths'].astype(np.int32)
    else:
        p['case_ids'][[0, 1]] = p['case_ids'][[1, 0]]
    with pytest.raises(ValueError):
        a.reconstruct_prefix(p, data, 'base', np)


def saved_prefix_fixture():
    p, _, _, _ = prefix_fixture()
    law = np.zeros((128, 9, 5), np.float64)
    law[:, 0, :4] = .25
    law[:, 1:] = [.1, .2, .3, .3, .1]
    law[~p['event_mask']] = 0
    labels = p['prefix'][..., 4:9].argmax(-1)
    nll = np.zeros((128, 9), np.float64)
    i, j = np.nonzero(p['event_mask'])
    nll[i, j] = -np.log(law[i, j, labels[i, j]])
    return p, {'probabilities': law, 'nll': nll}


def test_prefix_nll_is_global_event_mean_including_found_not_mean_of_case_means():
    p, prediction = saved_prefix_fixture()
    row = a.prefix_row(p, prediction, a.ARMS[0], a.SEEDS[0], np)
    expected = (128 * np.log(4) + prediction['nll'][:, 1:].sum()) / (128 * 9 - 7)
    assert row['mean_nll'] == pytest.approx(expected, abs=1e-14)
    assert row['valid_events'] == 128 * 9 - 7
    assert row['mean_nll'] != pytest.approx(np.mean(prediction['nll'].sum(-1) / p['lengths']), abs=1e-5)


@pytest.mark.parametrize('bad', ['zero_probability', 'nll', 'padding', 'reset_found', 'float32', 'unnormalized'])
def test_prefix_saved_probabilities_and_nll_fail_closed(bad):
    p, prediction = saved_prefix_fixture()
    if bad == 'zero_probability':
        prediction['probabilities'][0, 1] = [.2, .2, .3, .3, 0]
    elif bad == 'nll':
        prediction['nll'][0, 1] += .01
    elif bad == 'padding':
        prediction['nll'][0, 2] = 1
    elif bad == 'reset_found':
        prediction['probabilities'][0, 0] = [.2, .2, .2, .2, .2]
    elif bad == 'float32':
        prediction['probabilities'] = prediction['probabilities'].astype(np.float32)
    else:
        prediction['probabilities'][0, 1] *= .9
    with pytest.raises(ValueError):
        a.prefix_row(p, prediction, a.ARMS[0], a.SEEDS[0], np)


def test_exact_work_accounts_for_empty_endpoint_batches_and_first_found():
    prefixes, data, _, _ = prefix_fixture('train', two_survivors=True)
    values = a.prefix_work(prefixes, np.arange(64), np)
    assert values['prefix_probability_rows'] == values['prefix_nll_rows'] == 2 * 9 + 62 * 2
    assert values['prefix_filter_calls'] == 8 and values['prefix_filter_rows'] == 16
    assert values['reset_emission_rows'] == 64
    assert values['blind_transition_rows'] == values['cost_readout_rows'] == 0
    ordinary = a.endpoint_work(data['observations'], True, np)
    assert ordinary['observed_branch_calls'] == 1 and ordinary['observed_branch_rows'] == 2
    assert ordinary['event_probability_rows'] == 4 and ordinary['absorbed_rows'] == 0
    terminal = a.endpoint_work(np.array([[4, 4], [0, 4]], np.int64), True, np)
    assert terminal['observed_branch_calls'] == terminal['observed_branch_rows'] == terminal['absorbed_rows'] == 1
    blind = a.endpoint_work(np.array([[4, 4], [0, 4]], np.int64), False, np)
    assert blind['observed_branch_rows'] == blind['absorbed_rows'] == 0


def work_fixture():
    """Literal counts for one epoch/seed, two survivors and early-found others."""
    result = {}
    for arm in a.ARMS:
        result[arm] = {}
        for route in a.ROUTES:
            is_prefix, training = route.endswith('_prefix'), route.startswith('training_')
            if arm == 'endpoint_only' and route == 'training_prefix':
                result[arm][route] = {}
                continue
            values = dict.fromkeys(a.PREFIX_WORK_KEYS if is_prefix else a.ENDPOINT_WORK_KEYS, 0)
            if is_prefix:
                values.update({'reset_emission_softmax_calls': 24 if training else 6,
                    'reset_emission_rows': 1536 if training else 384,
                    'prefix_operator_softmax_calls': 24 if training else 6,
                    'prefix_filter_calls': 24, 'prefix_filter_rows': 48,
                    'prefix_probability_rows': 3114 if training else 810,
                    'prefix_nll_rows': 3114 if training else 810})
            else:
                values.update({'operator_observed_softmax_calls': 3, 'operator_marginal_sum_calls': 3,
                    'blind_transition_calls': 6 if training else 24,
                    'blind_transition_rows': 12 if training else 48,
                    'cost_readout_calls': 6 if training else 24, 'cost_readout_rows': 12 if training else 48,
                    'prefix_filter_calls': 24, 'prefix_filter_rows': 48,
                    'reset_emission_softmax_calls': 3, 'reset_emission_rows': 6, 'prefix_operator_softmax_calls': 3})
                if route.endswith('_observed'):
                    values.update({'event_probability_rows': 12 if training else 48,
                        'observed_branch_calls': 3, 'observed_branch_rows': 6, 'observed_conditioning_rows': 6,
                        'absorbed_rows': 0 if training else 36})
            result[arm][route] = values
    return result


@pytest.mark.parametrize('bad', [None, 'blind_absorbed', 'prefix_found_conditioned', 'prefix_padding', 'zero_batches'])
def test_full_work_reconstruction_uses_attempt_orders_and_both_loss_populations(bad):
    prefixes, datasets = {}, {}
    for split in ('train', 'base'):
        prefixes[split], datasets[split], _, _ = prefix_fixture(split, two_survivors=True)
    work = work_fixture()
    fits = [{'arm': arm, 'seed': seed, 'zero_endpoint_batches': 7} for arm in a.ARMS for seed in a.SEEDS]
    orders = [{'arm': arm, 'seed': seed, 'indices': list(range(512))} for arm in a.ARMS for seed in a.SEEDS]
    if bad == 'blind_absorbed':
        work[a.ARMS[0]]['training_blind']['absorbed_rows'] = 1
    elif bad == 'prefix_found_conditioned':
        work[a.ARMS[1]]['training_prefix']['prefix_filter_rows'] += 510 * 3
    elif bad == 'prefix_padding':
        work[a.ARMS[1]]['evaluation_prefix']['prefix_probability_rows'] = 128 * 9 * 3
    elif bad == 'zero_batches':
        fits[0]['zero_endpoint_batches'] = 0
    if bad:
        with pytest.raises(ValueError):
            a.validate_work(work, datasets, prefixes, orders, fits, np)
    else:
        result = a.validate_work(work, datasets, prefixes, orders, fits, np)
        assert result['zero_endpoint_batches'] == 42
        assert result['historical_forward_work_independently_replayed'] is False


def fabricated_metadata(folder):
    """Complete six-fit schedule and opaque payload bytes; no model or world."""
    def descriptor(path):
        return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

    def write(name, value, *, lines=False):
        (folder / name).write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))

    fits, orders, epochs = [], [], []
    indices = list(range(512))
    order_hash = hashlib.sha256(struct.pack('<' + 'q' * 512, *indices) * 480).hexdigest()
    for index, seed in enumerate(a.SEEDS):
        for arm in a.ARMS[index % 2:] + a.ARMS[:index % 2]:
            name = f'{arm}-{seed}.npz'
            (folder / name).write_bytes(b'opaque fabricated checkpoint, never decoded')
            weight = int(arm == a.ARMS[1])
            fits.append({'arm': arm, 'seed': seed, 'epochs': 480, 'updates': 3840,
                'training_cases': 2, 'training_case_exposures': 960, 'training_attempts': 512,
                'training_attempt_exposures': 245760, 'valid_prefix_events': 1038,
                'prefix_loss_weight': weight, 'training_prefix_event_exposures': weight * 498240,
                'zero_endpoint_batches': 3360, 'seconds': 1., 'case_order_sha256': order_hash,
                **{key: '1' * 64 for key in ('initial_state_sha256', 'final_state_sha256', 'fixed_buffers_sha256')},
                'prefix_initial_sha256': None, 'operator_initial_sha256': '3' * 64,
                'reset_initial_sha256': '4' * 64, 'prefix_operator_initial_sha256': '3' * 64,
                'oracle_prefix_input': False,
                'parameter_metadata': {'parameter_count': 1088, 'parameter_bytes': 8704, 'buffer_bytes': 256,
                    'parameters': {'reset_logits': {'count': 32, 'dtype': 'torch.float64', 'bytes': 256},
                        'observed_logits': {'count': 1056, 'dtype': 'torch.float64', 'bytes': 8448}}},
                'checkpoint': {'path': name, **descriptor(folder / name)}})
            for epoch in range(480):
                orders.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'indices': indices, 'batch_size': 64})
                epochs.append({'arm': arm, 'seed': seed, 'epoch': epoch, 'cases': 512, 'updates': 8,
                               'mean_objective': .5, 'seconds': .1})
    times = [{'arm': arm, 'seed': seed, 'regime': 'base', 'seconds': .1, 'cases': 2, 'batch_size': 64,
              'oracle_prefix_input': False, 'shuffle_offset': 1,
              'model_state_before': '1' * 64, 'model_state_after': '1' * 64}
             for arm in a.ARMS for seed in a.SEEDS]
    prefix_times = [{'arm': arm, 'seed': seed, 'seconds': .1, 'attempts': 128, 'valid_events': 270}
                    for arm in a.ARMS for seed in a.SEEDS]
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': 6, 'oracle_train_verified': True}
    dataset_counts, oracle_checks = {}, {}
    for split_id, split in enumerate(('train', 'base')):
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        dataset_counts[split] = {'version': 'finite-prefix-learning-data-v1', 'epsilon': .12,
            'seed_namespace': a.NAMESPACE, 'split_id': split_id, 'attempted': attempts, 'horizon': horizon,
            'retained': 2, 'discarded_found': attempts - 2, 'prefix_found_by_step': [attempts - 2] + [0] * 7,
            'initial_odor_draws': attempts, 'prefix_action_draws': 8 * attempts,
            'prefix_event_draws': attempts + 14, 'valid_prefix_events': 2 * attempts + 14,
            'forecast_action_draws': 2 * horizon, 'forecast_event_draws': 4, 'forecast_found_cases': 2}
        oracle_checks[split] = dict.fromkeys(a.TARGETS, 0.)
        write('oracle-' + split + '-check.json', oracle_checks[split])
    counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': 6,
        'oracle_model_constructions': 1, 'fit_count': 6, 'checkpoint_writes': 6,
        'training_blind_rollouts': 2880, 'training_observed_rollouts': 2880,
        'optimizer_attempts': 23040, 'optimizer_steps': 23040, 'training_case_exposures': 5760,
        'training_attempt_exposures': 1474560, 'training_prefix_rollouts': 11520,
        'training_prefix_event_exposures': 1494720, 'zero_endpoint_batches': 20160,
        'evaluation_blind_rollouts': 6, 'evaluation_observed_rollouts': 6,
        'evaluation_shuffled_rollouts': 6, 'evaluation_case_views': 12,
        'evaluation_prefix_rollouts': 12, 'evaluation_prefix_event_views': 1620,
        'oracle_blind_rollouts': 2, 'oracle_observed_rollouts': 2,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0}
    for name in ('train.npz', 'base.npz', 'train-prefix.npz', 'base-prefix.npz',
                 'train-oracle.npz', 'base-oracle.npz', 'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz'):
        (folder / name).write_bytes(b'opaque fake array, never decoded')
    for arm in a.ARMS:
        for seed in a.SEEDS:
            (folder / f'prefix-{arm}-{seed}-base.npz').write_bytes(b'opaque prefix predictions, never decoded')
    write('config.json', a.CONFIG)
    write('fits.jsonl', fits, lines=True)
    write('training-orders.jsonl', orders, lines=True)
    write('training-epochs.jsonl', epochs, lines=True)
    write('checkpoint-barrier.json', barrier)
    write('prediction-times.jsonl', times, lines=True)
    return {'version': 'finite-prefix-learning-v1', 'config': copy.deepcopy(a.CONFIG),
            'fits': fits, 'prediction_times': times, 'prefix_prediction_times': prefix_times, 'checkpoint_barrier': barrier,
            'structural_work': work_fixture(), 'dataset_counts': dataset_counts, 'counts': counts,
            'oracle_checks': oracle_checks, 'oracle_metadata': {'parameters': {}, 'parameter_count': 0, 'parameter_bytes': 0},
            'oracle_state_sha256': '4' * 64,
            'files': {path.name: descriptor(path) for path in folder.iterdir()}}


def test_metadata_authenticates_all_attempt_schedule_before_any_array_decode(tmp_path, monkeypatch):
    summary = fabricated_metadata(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('metadata admission must never decode arrays')
    monkeypatch.setattr(np, 'load', forbidden)
    result = a.validate_metadata(tmp_path, summary)
    assert result['fits'] == result['checkpoint_files_hashed'] == 6
    assert result['epochs'] == 2880 and result['optimizer_steps_attested'] == 23040
    assert result['training_attempt_exposures_attested'] == 1474560
    assert result['training_case_exposures_attested'] == 5760


@pytest.mark.parametrize('bad', ['checkpoint', 'missing_prefix_file', 'barrier', 'attempt_updates', 'loss_weight', 'initial_pair'])
def test_metadata_tamper_rejected_before_decode(tmp_path, monkeypatch, bad):
    summary = fabricated_metadata(tmp_path)
    if bad == 'checkpoint':
        (tmp_path / summary['fits'][0]['checkpoint']['path']).write_bytes(b'tampered')
    elif bad == 'missing_prefix_file':
        del summary['files']['base-prefix.npz']
    elif bad == 'barrier':
        summary['checkpoint_barrier']['dev_generation_count'] = 1
    else:
        row = summary['fits'][0]
        if bad == 'attempt_updates':
            row['updates'] = 480
        elif bad == 'loss_weight':
            row['prefix_loss_weight'] = 1
        else:
            row['initial_state_sha256'] = '9' * 64
        path = tmp_path / 'fits.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in summary['fits']))
        summary['files'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    def forbidden(*args, **kwargs):
        pytest.fail('invalid metadata must fail before array decode')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(ValueError):
        a.audit(tmp_path)
