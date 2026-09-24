"""Fabricated independent scalar, bank, schedule and saved-state audit oracles."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_otto_conditional_label as audit
import run_otto_conditional_label as producer


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setitem(audit.c.CONFIG, 'bootstrap_replicates', 13)
    monkeypatch.setitem(audit.c.CONFIG, 'min_dev_per_regime', 2)


def scalar_fixture():
    data = {'case_ids': np.array(['a', 'b', 'c', 'd']),
        'regimes': np.array(['lambda3', 'lambda3', 'lambda4', 'lambda4']),
        'costs': np.tile(np.array([1, 3, 5, 7], np.float32), (4, 128, 1)),
        'alive': np.ones((4, 128), np.bool_)}
    predictions = {}
    for family in audit.c.FAMILIES:
        for seed in audit.c.FIT_SEEDS:
            value = np.zeros((4, 8, 4), np.float32)
            value[..., 1 if family == 'sampled' else 0] = -1
            predictions[f'{family}__{seed}__cost'] = value
    return data, predictions


@pytest.mark.parametrize('variant', ['ordinary', 'found_half', 'all_found', 'heterogeneous', 'one_seed_tie'])
def test_independent_full_report_and_bootstrap(small, variant):
    data, predictions = scalar_fixture()
    if variant == 'found_half':
        data['costs'][:, :64] = 0
        data['alive'][:, :64] = False
    elif variant == 'all_found':
        data['costs'][:] = 0
        data['alive'][:] = False
    elif variant == 'heterogeneous':
        data['costs'][0, :64] = [3, 1, 8, 0]
    elif variant == 'one_seed_tie':
        predictions[f'mean32__{audit.c.FIT_SEEDS[0]}__cost'][:] = predictions[f'sampled__{audit.c.FIT_SEEDS[0]}__cost']
    before = {k: v.copy() for k, v in data.items()}
    actual = audit.analyze(data, predictions, np)
    audit.compare_report(producer.analyze(data, predictions, np), actual)
    assert len(actual['rows']) == 12 and len(actual['cases']) == 24
    assert all(np.array_equal(v, before[k]) for k, v in data.items())
    for row in actual['cases']:
        assert row['sampled_target_mse'] == pytest.approx(row['mean_target_mse'] + row['within_bank_variance'], abs=1e-12)
    if variant == 'ordinary':
        assert actual['gate']['passed'] is True
        assert all(g['sampled_gap'] == g['gain'] == 2 and g['mean32_gap'] == 0 for g in actual['gate']['groups'])
        assert all(g['gain_minus_required'] == 1.9 for g in actual['gate']['groups'])
    elif variant == 'found_half':
        assert all(g['sampled_gap'] == g['gain'] == 1 for g in actual['gate']['groups'])
    elif variant == 'all_found':
        assert not actual['gate']['passed']
        assert all(not g['conditions']['positive_sampled_gap'] for g in actual['gate']['groups'])
    elif variant == 'one_seed_tie':
        assert not actual['gate']['passed']
        assert all(not g['conditions']['all_paired_seeds_positive'] for g in actual['gate']['groups'])


def test_targets_preserve_found_zeros_and_center_every_draw():
    data, _ = scalar_fixture()
    data['costs'][:, :64] = 0
    result = audit.centered_bank(data, np)
    np.testing.assert_array_equal(result[:, :64], 0)
    np.testing.assert_array_equal(result[:, 64:], np.tile(np.array([-3., -1., 1., 3.]) / 64, (4, 64, 1)))
    assert np.square(result).mean() == 2.5 / 64**2


@pytest.mark.parametrize('change', ['missing', 'nan', 'dtype', 'shape'])
def test_prediction_contract_fails_closed(change):
    data, predictions = scalar_fixture()
    key = next(iter(predictions))
    if change == 'missing':
        del predictions[key]
    elif change == 'nan':
        predictions[key][0, 0, 0] = np.nan
    elif change == 'dtype':
        predictions[key] = predictions[key].astype(np.float64)
    else:
        predictions[key] = predictions[key][:, :4]
    with pytest.raises(ValueError):
        audit.analyze(data, predictions, np)


def physical_fixture():
    from openjev.research import otto_conditional_label_sampling as sampling
    from openjev.research import otto_predictive_belief as bayes
    from openjev.research.otto_query_gate import _features

    sensor = np.full((105, 105, 4), .25, np.float64)
    sensor[52, 52] = 0.
    kernel = np.full((4, 107, 107), .25, np.float64)
    kernel[:, 53, 53] = 0.
    tables = {name: array.copy() for regime in ('lambda3', 'lambda4')
              for name, array in ((regime, sensor), (regime + '_raw', sensor), ('legacy_' + regime, kernel))}
    identity = {'split': 'train', 'regime': 'lambda3', 'id': 'fabricated:train', 'seed': 710, 'initial_hit': 1, 'mc_seed': 720}
    prior = np.ones((53, 53), np.float64) / 2808
    prior[26, 26] = 0
    initial_legacy = prior * .25
    initial_legacy /= initial_legacy.sum()
    initial = bayes.normalize_prior(bayes.cdf_law(initial_legacy.reshape(-1))['probabilities'])['belief']
    strict, legacy = initial.copy(), initial_legacy.copy()
    past = np.array([0, 1] * 4, np.int64)
    prefix, previous = [], None
    for t, position in enumerate([(26, 26), *((25, 26), (26, 26)) * 4]):
        if t:
            j = position[0] * 53 + position[1]
            strict[j], legacy[position] = 0., 0.
            strict *= .25
            legacy *= .25
            strict /= strict.sum()
            legacy /= legacy.sum()
        public = {'position': position, 'hit': 1 if t == 0 else 0, 'step': t, 'done': False, 'valid_actions': (0, 1, 2, 3)}
        feature, previous = _features(SimpleNamespace(p_source=legacy.copy()), public, 3., np.zeros(4),
                                      None if t == 0 else int(past[t - 1]), t - t % 4, previous)
        prefix.append(feature)
    grid = bayes.cdf_law(strict)['probabilities']
    actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([710, 911]))).integers(0, 4, 8, dtype=np.int64)
    position, positions = [26, 26], []
    for action in actions:
        position[action // 2] += 2 * (int(action) % 2) - 1
        positions.append(position.copy())
    points = np.asarray(positions)
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    laws = np.stack([sensor[(coordinates - p + 52)[:, 0], (coordinates - p + 52)[:, 1]] for p in points])
    integers = np.random.PCG64(720).random_raw((32, 9)) >> np.uint64(11)

    def update(value, index, _odor):
        value[index] = 0
        value *= .25
        if value.sum() > 1e-10:
            value /= value.sum()
        return value

    sampled = sampling.sample_endpoint(grid, legacy.reshape(-1), laws, points[:, 0] * 53 + points[:, 1],
                                      integers, update, lambda _state, _index: np.array([1, 2, 3, 4], np.float32))
    row = {'prefix': np.stack(prefix), 'prefix_lengths': np.int64(9), 'prefix_actions': past,
        'prefix_outcomes': np.zeros(8, np.int64), 'prefix_position': np.array([26, 26], np.int64),
        'actions': actions, 'initial_belief': initial, 'root_strict': strict, 'root_grid': grid,
        'legacy_root': legacy.reshape(-1), 'mc_draws': integers, 'mc_source_indices': sampled['source_indices'],
        'mc_outcomes': sampled['outcomes'], 'alive': sampled['alive'], 'costs': sampled['costs']}
    data = {k: np.stack([v]) for k, v in row.items()}
    data.update(case_ids=np.array([identity['id']]), regimes=np.array(['lambda3']))
    return data, tables, [identity], sampled['work']


@pytest.fixture(scope='module')
def physical():
    return physical_fixture()


def test_complete_physical_bank_reconstruction(physical):
    data, tables, roster, work = physical
    audit.verify_tables(tables, np)
    result = audit.verify_probabilities(data, tables, roster, np, split='train')
    assert result['cases'][0]['work'] == work
    assert result['teacher_recomputed'] is False


@pytest.mark.parametrize('change', ['prefix_hit', 'prefix_action', 'legacy', 'root', 'draw', 'source', 'odor', 'terminal_cost'])
def test_saved_probability_corruptions(physical, change):
    original, tables, roster, _ = physical
    data = {k: v.copy() for k, v in original.items()}
    if change == 'prefix_hit':
        data['prefix'][0, 0, 7:11] = [1, 0, 0, 0]
    elif change == 'prefix_action':
        data['prefix'][0, 1, 11:15] = [0, 1, 0, 0]
    elif change == 'legacy':
        data['legacy_root'][0, 0] += .01
    elif change == 'root':
        data['root_grid'][0, 0] += .01
    elif change == 'draw':
        data['mc_draws'][0, 0, 2] ^= np.uint64(1)
    elif change == 'source':
        data['mc_source_indices'][0, 0] = (data['mc_source_indices'][0, 0] + 1) % 2809
    elif change == 'odor':
        data['mc_outcomes'][0, 0, 0] = (data['mc_outcomes'][0, 0, 0] + 1) % 4
    else:
        data['alive'][0, 0] = False
        data['mc_outcomes'][0, 0] = 4
        data['costs'][0, 0] = 1
    with pytest.raises(ValueError):
        audit.verify_probabilities(data, tables, roster, np, split='train')


def test_forward_links_endpoint_mass_context_and_q(physical):
    _data, tables, roster, _ = physical
    kernel = tables['legacy_lambda3']
    state = np.zeros(2809)
    state[0] = 1
    masses = np.full((4, 4), .25, np.float32)
    event = {'ordinal': 1, 'input_shape': [16, 105, 105], 'symmetry_average': True,
        'context': {**roster[0], 'phase': 'counterfactual', 'stream': 'train', 'mode': 'mc', 'horizon': 8, 'sample': 0, 'history': [0] * 8},
        'branch_masses': masses.tolist(), 'values': np.tile(np.array([0., 4., 8., 12.]), 4).tolist()}
    saved = np.full(4, 7, np.float32)
    assert audit.verify_forward(event, 1, roster[0], 0, [0] * 8, state, (26, 26), kernel, saved, np) == 0
    for change in ('mass', 'context', 'cost'):
        bad, q = copy.deepcopy(event), saved.copy()
        if change == 'mass':
            bad['branch_masses'][0][0] = .5
        elif change == 'context':
            bad['context']['sample'] = 1
        else:
            q[0] += 1
        with pytest.raises(ValueError):
            audit.verify_forward(bad, 1, roster[0], 0, [0] * 8, state, (26, 26), kernel, q, np)


def checkpoint_fixture():
    state = {k: np.zeros(shape, np.float32) for k, shape in audit.parameter_shapes().items()}
    state['cost_scale'] = np.array(.25, np.float32)
    unused = {k: v for k, v in state.items() if k.startswith(('outcome_head.', 'aux_head.'))}
    row = {'final_state_sha256': audit.state_digest(state), 'unused_heads_before_sha256': audit.state_digest(unused),
           'unused_heads_after_sha256': audit.state_digest(unused)}
    return state, row


@pytest.mark.parametrize('change', [None, 'extra', 'dtype', 'shape', 'nan', 'scale', 'unused'])
def test_saved_checkpoint_schema_and_hashes(change):
    state, row = checkpoint_fixture()
    if change == 'extra':
        state['unregistered'] = np.zeros(1)
    elif change == 'dtype':
        state['cost_head.weight'] = state['cost_head.weight'].astype(np.float64)
    elif change == 'shape':
        state['cost_head.weight'] = state['cost_head.weight'][:3]
    elif change == 'nan':
        state['cost_head.weight'][0, 0] = np.nan
    elif change == 'scale':
        state['cost_scale'] = np.array(.5, np.float32)
    elif change == 'unused':
        state['outcome_head.bias'][0] = 1
        row['final_state_sha256'] = audit.state_digest(state)
    if change is None:
        audit.verify_checkpoint(state, row, .25, np)
    else:
        with pytest.raises(ValueError):
            audit.verify_checkpoint(state, row, .25, np)


def training_fixture(tmp_path):
    data, _ = scalar_fixture()
    train = {'case_ids': data['case_ids'][:2], 'costs': data['costs'][:2, :32]}
    n, epochs = 2, audit.c.CONFIG['epochs']
    moment = 5 / 64**2
    scale = float(np.float32(math.sqrt(moment)))
    normalization = {'cost_scale': scale, 'second_moment_before_floor': moment, 'variance_floor': 1e-6,
        'teacher_units_divisor': 64., 'train_cases': n, 'draws_per_case': 32, 'dev_decodes': 0}
    draws, fits, orders, events = [], [], [], []
    for j, seed in enumerate(audit.c.FIT_SEEDS):
        permutations = producer.draw_permutations(seed, n, np)
        pin = hashlib.sha256(permutations.tobytes()).hexdigest()
        draws.append({'seed': seed, 'permutations': permutations.tolist(), 'sha256': pin, 'cycles': 3, 'index_seed_suffix': 717})
        for family in audit.c.FAMILIES[j % 2:] + audit.c.FAMILIES[:j % 2]:
            for epoch in range(epochs):
                indices, selected = producer.case_order(seed, epoch, n, np), permutations[:, epoch % 32]
                orders.append({'family': family, 'seed': seed, 'epoch': epoch, 'indices': indices.tolist(),
                    'draw_indices': selected.tolist(), 'index_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                    'draw_sha256': hashlib.sha256(selected.tobytes()).hexdigest()})
                context = {'family': family, 'seed': seed, 'epoch': epoch, 'batch_start': 0, 'cases': n, 'update': epoch}
                events.extend([{'event': 'attempt', **context}, {'event': 'return', **context, 'loss': 1., 'gradient_norm': 2., 'work': audit.expected_work(n)}])
            checkpoint = tmp_path / f'{family}-{seed}.npz'
            checkpoint.write_bytes(b'opaque fabricated checkpoint')
            shapes = audit.parameter_shapes()
            fits.append({'family': family, 'seed': seed, 'model_kind': 'action_recurrent', 'epochs': epochs,
                'cases': n, 'updates': epochs, 'exposures': n * epochs, 'target_draw_uses': n * epochs * (32 if family == 'mean32' else 1),
                'unique_teacher_labels': n * 32, 'draw_permutation_sha256': pin, 'sampled_cycles': 3 if family == 'sampled' else None,
                'initial_state_sha256': str(seed), 'training_horizons': [8], 'executed_horizons': list(range(1, 9)),
                'evaluation_decodes_so_far': 0, 'blind_input_keys': ['prefix', 'prefix_lengths', 'actions'],
                'privileged_targets_only': ['costs', 'alive'], 'effective_parameter_names': [k for k in shapes if not k.startswith(('outcome_head.', 'aux_head.'))],
                'parameters': {'kind': 'action_recurrent', 'hidden_dim': 28, 'cost_scale': scale, 'count': 8299, 'trainable_count': 8096,
                    'parameters': {k: {'shape': v, 'count': math.prod(v)} for k, v in shapes.items()}, 'zero_proposed_action_weight_parameters': 0, 'direct_descriptor': None},
                'work': audit.expected_work(n * epochs, epochs), 'checkpoint': audit.c.desc(checkpoint), 'seconds': 1., 'last_batch_loss': 1., 'last_gradient_norm': 2.})
    (tmp_path / 'fits.jsonl').write_text('\n'.join(map(json.dumps, fits)) + '\n')
    summary = {'normalization': normalization.copy(), 'fits': fits, 'train_data': {'sha256': 'fabricated'}}
    barrier = {'fits_completed': 6, 'checkpoint_files': {f'{f}-{s}.npz': audit.c.desc(tmp_path / f'{f}-{s}.npz') for f in audit.c.FAMILIES for s in audit.c.FIT_SEEDS},
        'fits_sha256': audit.c.desc(tmp_path / 'fits.jsonl')['sha256'], 'train_data': summary['train_data'], 'dev_array_decodes': 0, 'created_ns': 20}
    return train, summary, normalization, draws, orders, events, fits, barrier, tmp_path, {'started_ns': 10, 'finished_ns': 30}


@pytest.mark.parametrize('change', [None, 'mean_scale', 'draw_order', 'case_order', 'missing_return', 'mask', 'dev_barrier', 'paired_initial'])
def test_independent_training_bank_schedule_and_barrier(tmp_path, change):
    args = list(training_fixture(tmp_path))
    if change == 'mean_scale':
        args[2]['second_moment_before_floor'] *= .5
    elif change == 'draw_order':
        args[3][0]['permutations'][0][0] ^= 1
    elif change == 'case_order':
        args[4][0]['indices'].reverse()
    elif change == 'missing_return':
        del args[5][1]
    elif change == 'mask':
        args[6][0]['effective_parameter_names'].append('outcome_head.bias')
    elif change == 'dev_barrier':
        args[7]['dev_array_decodes'] = 1
    elif change == 'paired_initial':
        args[6][1]['initial_state_sha256'] = 'different'
    if change is None:
        result = audit.verify_training(*args, np)
        assert result['fits_checked'] == 6 and result['optimizer_updates_checked'] == 576
    else:
        with pytest.raises(ValueError):
            audit.verify_training(*args, np)


def test_original_closure_failure_precedes_any_array_decode(monkeypatch):
    monkeypatch.setattr(audit.c, 'closed', lambda *_args: (_ for _ in ()).throw(ValueError('unclosed original fit')))
    monkeypatch.setattr(np, 'load', lambda *_args, **_kwargs: pytest.fail('array decode before admission'))
    with pytest.raises(ValueError, match='unclosed original'):
        audit.run_audit(SimpleNamespace())
