"""Fabricated comparisons, matched schedules, and the pre-DEV fit barrier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_otto_conditional_label as r


def data(*, split='dev', cases=4):
    draws = r.c.CONFIG[split + '_draws']
    prefix_actions = np.tile([0, 1, 2, 3, 0, 1, 2, 3], (cases, 1)).astype(np.int64)
    root = np.zeros((cases, 2809), np.float64)
    root[:, 0] = 1.
    return {'prefix': np.zeros((cases, 9, 31), np.float32), 'prefix_lengths': np.full(cases, 9, np.int64),
        'prefix_actions': prefix_actions.copy(), 'prefix_outcomes': np.zeros((cases, 8), np.int64),
        'prefix_position': np.full((cases, 2), 26, np.int64), 'actions': prefix_actions.copy(),
        **{key: root.copy() for key in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root')},
        'mc_draws': np.zeros((cases, draws, 9), np.uint64), 'mc_source_indices': np.zeros((cases, draws), np.int64),
        'mc_outcomes': np.zeros((cases, draws, 8), np.int64), 'alive': np.ones((cases, draws), bool),
        'costs': np.tile(np.array([1., 3., 5., 7.], np.float32), (cases, draws, 1)),
        'case_ids': np.array([f'{split}:case{i}' for i in range(cases)]),
        'regimes': np.array(['lambda3' if split == 'train' or i < cases // 2 else 'lambda4' for i in range(cases)])}


def predictions(d, *, sampled_action=1, mean_action=0):
    result = {}
    for family, action in zip(r.FAMILIES, (sampled_action, mean_action), strict=True):
        for seed in r.c.CONFIG['fit_seeds']:
            value = np.zeros((len(d['case_ids']), 8, 4), np.float32)
            value[:, 7, action] = -1
            result[f'{family}__{seed}__cost'] = value
    return result


@pytest.fixture
def small_bootstrap(monkeypatch):
    config = {**r.c.CONFIG, 'bootstrap_replicates': 20, 'min_dev_per_regime': 2}
    monkeypatch.setattr(r.c, 'CONFIG', config)
    return config


def test_known_case_regret_mse_decomposition_and_paired_gate(small_bootstrap):
    d, p = data(), predictions(data())
    report = r.analyze(d, p, np)
    assert len(report['rows']) == 12 and len(report['cases']) == 24
    assert report['gate']['passed']
    for group in report['gate']['groups']:
        assert group['sampled_gap'] == 2. and group['mean32_gap'] == 0.
        assert group['gain_minus_required'] == pytest.approx(1.9)
        assert group['approximate_95_percent_interval'][3] == [1.9, 1.9]
        assert len(group['paired_seed_gains']) == 3
    for row in report['rows']:
        assert row['teacher_regret'] == (2 if row['family'] == 'sampled' else 0)
        assert row['sampled_target_mse'] == pytest.approx(row['mean_target_mse'] + row['within_bank_variance'])


def test_zero_found_prefix_kept_in_denominator_not_removed(small_bootstrap):
    d = data()
    for i in (0, 2):
        d['costs'][i] = 0
        d['alive'][i] = False
        d['mc_outcomes'][i] = 4
    r.validate_data(d, np, split='dev')
    report = r.analyze(d, predictions(d), np)
    assert all(group['sampled_gap'] == 1. and group['cases'] == 2 for group in report['gate']['groups'])
    assert sum(row['alive_draws'] == 0 for row in report['cases']) == 12
    # A nonzero forecast is penalized against zero even for an all-found bank.
    assert all(row['sampled_target_mse'] == .25 for row in report['cases'] if row['alive_draws'] == 0)


def test_each_seed_must_improve_even_when_pooled_gate_margin_is_positive(small_bootstrap):
    d, p = data(), predictions(data())
    bad = p[f'mean32__{r.c.CONFIG["fit_seeds"][0]}__cost']
    bad[:] = 0
    bad[:, 7, 2] = -1
    gate = r.primary_gate(d, p, np)
    assert not gate['passed']
    for group in gate['groups']:
        assert group['conditions']['resolved_gain']
        assert not group['conditions']['all_paired_seeds_positive']
        assert group['paired_seed_gains'][0]['gain'] == -2


def test_five_percent_boundary_requires_strictly_positive_lower_margin(small_bootstrap):
    d = data()
    d['costs'][:] = [0., 19., 20., 21.]
    gate = r.primary_gate(d, predictions(d, sampled_action=2, mean_action=1), np)
    assert not gate['passed']
    assert all(group['gain_minus_required'] == 0 and not group['conditions']['resolved_gain'] for group in gate['groups'])


def test_minimum_support_and_zero_baseline_are_not_silent_passes(small_bootstrap):
    d = data(cases=2)
    gate = r.primary_gate(d, predictions(d, sampled_action=0, mean_action=0), np)
    assert not gate['passed']
    assert all(not g['conditions']['minimum_cases'] and not g['conditions']['positive_sampled_gap'] for g in gate['groups'])


def test_bootstrap_shared_indices_match_independent_scalar_two_level_average(small_bootstrap):
    d = data(cases=6)
    for i in range(6):
        d['costs'][i, :, 1] += np.arange(128) % (i + 2)
    gate = r.primary_gate(d, predictions(d), np)
    for ri, regime in enumerate(('lambda3', 'lambda4')):
        ix = np.flatnonzero(d['regimes'] == regime)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([344000001, ri])))
        ci = rng.integers(0, 3, size=(20, 3), dtype=np.int64)
        di = rng.integers(0, 128, size=(20, 3, 128), dtype=np.int64)
        expected = []
        for rep in range(20):
            total = sum(float(d['costs'][ix[ci[rep, i]], di[rep, i, j], 1] - 1)
                        for i in range(3) for j in range(128)) / (3 * 128)
            expected.append([total, 0., total, .95 * total])
        np.testing.assert_allclose(gate['groups'][ri]['bootstrap_replicates'], expected, rtol=1e-14, atol=1e-14)


def test_balanced_label_schedule_and_case_order_are_reproducible_rng_local():
    before = np.random.get_state()
    order = r.draw_permutations(343000001, 5, np)
    assert all(sorted(row.tolist()) == list(range(32)) for row in order)
    counts = np.zeros((5, 32), np.int64)
    for epoch in range(96):
        counts[np.arange(5), order[:, epoch % 32]] += 1
        permutation = r.case_order(343000001, epoch, 5, np)
        assert sorted(permutation.tolist()) == list(range(5))
        np.testing.assert_array_equal(permutation, r.case_order(343000001, epoch, 5, np))
    assert (counts == 3).all()
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize('mutation', ['terminal_cost', 'nan', 'position', 'duplicate', 'nonabsorbing', 'draw_type'])
def test_dataset_contract_fails_on_malformed_banks(mutation):
    d = data()
    if mutation == 'terminal_cost':
        d['mc_outcomes'][0] = 4
        d['alive'][0] = False
    elif mutation == 'nan':
        d['costs'][0, 0, 0] = np.nan
    elif mutation == 'position':
        d['prefix_position'][0, 0] += 1
    elif mutation == 'duplicate':
        d['case_ids'][1] = d['case_ids'][0]
    elif mutation == 'nonabsorbing':
        d['mc_outcomes'][0, 0, 0] = 4
    else:
        d['mc_draws'] = d['mc_draws'].astype(np.int64)
    with pytest.raises(ValueError):
        r.validate_data(d, np, split='dev')


def test_prediction_roster_and_deadline_callback_fail_closed(small_bootstrap):
    d, p = data(), predictions(data())
    p.pop(next(iter(p)))
    with pytest.raises(ValueError, match='six complete'):
        r.analyze(d, p, np)
    def expired():
        raise TimeoutError('fabricated bound')
    with pytest.raises(TimeoutError, match='fabricated bound'):
        r.analyze(d, predictions(d), np, check=expired)


def test_unclosed_collection_rejected_before_any_array_decode(monkeypatch):
    pilot = r.Pilot.__new__(r.Pilot)
    def unclosed(*args):
        raise ValueError('original collection still open')
    monkeypatch.setattr(r.c, 'closed', unclosed)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('array decoded'))
    with pytest.raises(ValueError, match='still open'):
        pilot.body()


def test_all_six_final_checkpoints_exist_before_first_dev_decode(tmp_path, monkeypatch):
    config = {**r.c.CONFIG, 'min_train': 1, 'epochs': 1, 'batch_size': 1}
    monkeypatch.setattr(r.c, 'CONFIG', config)
    monkeypatch.setattr(r.c, 'OUT', tmp_path)
    monkeypatch.setattr(r.c, 'closed', lambda *args: {'plan_sha256': 'a' * 64})
    monkeypatch.setattr(torch, 'set_num_threads', lambda *args: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda *args: None)
    collection = tmp_path / 'collection-01'
    collection.mkdir()
    (collection / 'train.npz').write_bytes(b'fabricated opaque identity only')
    pilot = r.Pilot.__new__(r.Pilot)
    pilot.args = SimpleNamespace(plan_sha256='a' * 64)
    pilot.out = tmp_path / 'fit-01'
    pilot.out.mkdir()
    pilot.receipt = {}
    clock = iter(range(1_000, 100_000))
    pilot.clock = SimpleNamespace(now_ns=lambda: next(clock))
    pilot.check = lambda: None
    def loader(path, np_module, *, split):
        if split == 'train':
            return data(split='train', cases=1)
        barrier = json.loads((pilot.out / 'fit-barrier.json').read_text())
        fits = [json.loads(line) for line in (pilot.out / 'fits.jsonl').read_text().splitlines()]
        assert len(fits) == 6 and barrier['fits_completed'] == 6 and barrier['dev_array_decodes'] == 0
        assert len(barrier['checkpoint_files']) == 6
        for name, expected in barrier['checkpoint_files'].items():
            assert r.c.desc(pilot.out / name) == expected
        for seed in config['fit_seeds']:
            pair = [row for row in fits if row['seed'] == seed]
            assert pair[0]['initial_state_sha256'] == pair[1]['initial_state_sha256']
            assert all(row['unused_heads_before_sha256'] == row['unused_heads_after_sha256'] for row in pair)
        assert pilot.receipt['calls']['fit_count'] == 6
        assert pilot.receipt['calls']['mean_target_aggregation_calls'] == 1
        raise RuntimeError('intentional stop before DEV decode')
    monkeypatch.setattr(r, 'load_data', loader)
    with pytest.raises(RuntimeError, match='intentional stop'):
        pilot.body()
