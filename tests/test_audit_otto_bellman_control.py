"""Fabricated saved records only; no scientific dataset, simulator or training."""
from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_test_bellman_control_audit', ROOT/'scripts/audit_otto_bellman_control.py')
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def rows():
    result = []
    for regime, seed, hit, arm, block in M.evaluation_order():
        family = arm.split('@')[0]
        steps = {'backup': 9, 'mc': 10, 'reference': 10, 'analytic_inbounds': 9}[family]
        row = {'regime': regime, 'seed': seed, 'initial_hit': hit, 'arm': arm, 'block': block,
               'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0, 'final_update_assimilated': True,
               'init_seconds': .1, 'choose_seconds': .2, 'update_seconds': .1, 'setup_allocation_seconds': 0.,
               'controller_seconds': .4, 'environment_seconds': .3, 'state_bytes': 22472}
        result.append(row)
    return result


def mixtures():
    return {r: {'1': .5, '2': .25, '3': .25} for r in M.REGIMES}


def exported():
    return {'version': np.asarray('otto-return-value-v1'), 'kind': np.asarray('mlp8'),
            'input_dim': np.asarray(11028), 'c0': np.asarray(.5, np.float32),
            'first_weight': np.zeros((8, 11028), np.float32), 'final_weight': np.zeros(8, np.float32),
            'hidden_bias': np.zeros(8, np.float32), 'output_bias': np.asarray(0., np.float32)}


def test_complete_rotation_payloads_and_all42_conditions():
    sample = rows()
    result = M.aggregate(sample, mixtures())
    assert len(sample) == 1440 and len({(r['regime'], r['seed']) for r in sample}) == 144
    assert len(M.payload_names()) == 91
    assert len(result['competence_checks']) == 18 and len(result['improvement_checks']) == 24
    assert result['pilot_continuation']
    assert not result['learned_architecture_advantage_established']
    for regime in M.REGIMES:
        for arm in M.ARMS:
            assert result['regimes'][regime]['raw_counts'][arm] == {'found': 48, 'episodes': 48}


def test_paired_gain_over_weak_models_does_not_fake_analytic_competence():
    sample = rows()
    for row in sample:
        if row['arm'].startswith('backup'):
            row['steps'] = row['updates'] = 95
        elif row['arm'] != 'analytic_inbounds':
            row['steps'] = row['updates'] = 100
    result = M.aggregate(sample, mixtures())
    assert all(c['passes'] for c in result['improvement_checks'])
    assert not result['pilot_continuation']
    assert sum(c['passes'] for c in result['competence_checks']) == 9


def test_every_seed_competence_cannot_be_hidden_by_family_mean():
    sample = rows()
    for row in sample:
        if row['arm'] == 'backup@10101':
            row['steps'] = row['updates'] = 12
        elif row['arm'].startswith('backup'):
            row['steps'] = row['updates'] = 1
    result = M.aggregate(sample, mixtures())
    assert result['regimes']['lambda3']['family_means']['backup']['steps'] < 9
    assert not result['pilot_continuation']


def test_strict_block_sign_and_inclusive_five_percent_threshold():
    sample = rows()
    for row in sample:
        if row['arm'].startswith('backup'):
            row['steps'] = row['updates'] = 19
        elif row['arm'] != 'analytic_inbounds':
            row['steps'] = row['updates'] = 20
        else:
            row['steps'] = row['updates'] = 19
    result = M.aggregate(sample, mixtures())
    assert all(c['passes'] for c in result['improvement_checks'])
    for row in sample:
        if row['arm'].startswith('backup') and row['block'] >= 5:
            row['steps'] = row['updates'] = 20
    result = M.aggregate(sample, mixtures())
    checks = [c for c in result['improvement_checks'] if c['name'].endswith('positive_blocks')]
    assert all(c['value'] == 5 and not c['passes'] for c in checks)


def test_weighted_strata_and_equal_seed_means_not_pooled_outcomes():
    sample = rows()
    for row in sample:
        row['steps'] = row['updates'] = {1: 2, 2: 8, 3: 20}[row['initial_hit']]
    panel = M.aggregate(sample, mixtures())['regimes']['lambda3']
    assert panel['means']['backup@10101']['steps'] == 8
    assert panel['family_means']['backup']['steps'] == 8
    assert panel['strata']['3']['backup@10101']['steps'] == 20
    assert 8 != (2+8+20)/3


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order', 'nan', 'short_censor', 'partial_update', 'free_cost'])
def test_incomplete_or_malformed_trial_rejected(mutation):
    sample = rows()
    if mutation == 'missing':
        sample.pop()
    elif mutation == 'duplicate':
        sample[-1] = sample[-2].copy()
    elif mutation == 'order':
        sample[0], sample[1] = sample[1], sample[0]
    elif mutation == 'nan':
        sample[0]['controller_seconds'] = float('nan')
    elif mutation == 'short_censor':
        sample[0]['found'] = False
    elif mutation == 'partial_update':
        sample[0]['updates'] -= 1
    else:
        sample[0]['controller_seconds'] = 0.
    with pytest.raises(ValueError):
        M.aggregate(sample, mixtures())


def test_amortization_no_reference_retraining_or_double_deployment_charge():
    result = M.aggregate(rows(), mixtures())
    setup = {a: {'seconds': 9.} for a in M.ARMS[:-1]}
    fits = {a: 60. for a in setup if not a.startswith('reference')}
    for panel in result['regimes'].values():
        for arm, value in panel['means'].items():
            value['controller_seconds'] = 2.
            value['setup_allocation_seconds'] = 1. if arm != 'analytic_inbounds' else 0.
    table = M.amortization(result, setup, fits, 120., 18.)['lambda3']
    assert table['backup@10101']['seconds_per_search'] == pytest.approx(
        {'1': 92., '100': 1.91, '10000': 1.0091}, rel=0, abs=3e-16)
    assert table['reference@10101']['seconds_per_search']['1'] == 12.
    assert table['reference@10101']['preparation_share_seconds'] == 0.
    assert table['analytic_inbounds']['seconds_per_search'] == {'1': 2., '100': 2., '10000': 2.}


def journal():
    return [[1, 0, 'target_refresh', 0, None], [2, 0, 'target_readout', 1, 0],
            [2, 1, 'target_readout', .2, .3, .1], [3, 0, 'target_readout', 1, 1],
            [3, 1, 'target_readout', .1, .1, 0.], [1, 1, 'target_refresh', .5, .7, .2]]


def work(records):
    return M.Work(iter(records), [{'phase': 'target_refresh'}, {'phase': 'target_generation'}], M.B.Comparisons())


def test_nested_journal_context_counts_and_no_double_counted_parent_time():
    w = work(journal())
    handle = w.begin('target_refresh', {'phase': 'target_refresh'})
    for i in range(2):
        w.call('target_readout', {'phase': 'target_generation', 'step': i})
    result = w.end('target_refresh', handle)
    assert result['seconds'] == .5
    assert w.counts['target_readout']['returned'] == 2
    assert w.counts['target_refresh']['returned'] == 1 and not w.stack
    assert w.counts['target_readout']['seconds'] == pytest.approx(.3)


@pytest.mark.parametrize('mutation', ['return_id', 'context', 'infinite', 'missing'])
def test_journal_failure_does_not_claim_completion(mutation):
    records = journal()
    if mutation == 'return_id':
        records[2][0] = 9
    elif mutation == 'context':
        records[1][4] = 7
    elif mutation == 'infinite':
        records[2][3] = math.inf
    else:
        records = records[:2]
    w = work(records)
    w.begin('target_refresh', {'phase': 'target_refresh'})
    with pytest.raises((ValueError, StopIteration)):
        w.call('target_readout', {'phase': 'target_generation', 'step': 0})
    assert w.counts['target_refresh']['returned'] == 0
    assert w.counts.get('target_readout', {}).get('returned', 0) == 0


def test_checkpoint_identity_and_fixed_c0_are_byte_bound():
    original = exported()
    identity = M.content_identity(original)
    assert M.content_identity(copy.deepcopy(original)) == identity
    original['first_weight'][0, 0] = .125
    assert M.content_identity(original) != identity
    M.A.checkpoint(original, 'mlp8', .5, np)
    original['c0'][...] = .75
    with pytest.raises(ValueError, match='baseline'):
        M.A.checkpoint(original, 'mlp8', .5, np)


def test_target_cache_path_and_descriptor_are_required(tmp_path):
    audit = M.Audit(SimpleNamespace(output=tmp_path, run=tmp_path))
    audit.check = lambda: None
    file = tmp_path/'target-checkpoint-backup-10101-00.npz'
    file.write_bytes(b'fabricated checkpoint bytes')
    record = {'path': file.name, **M.B.digest(file)}
    audit.checkpoint_record(record, file.name)
    for replacement in ('other.npz', '../'+file.name):
        with pytest.raises(ValueError, match='path'):
            audit.checkpoint_record({**record, 'path': replacement}, file.name)
    with pytest.raises(ValueError, match='identity'):
        audit.checkpoint_record({**record, 'sha256': '0'*64}, file.name)


def test_analytic_immediate_found_and_public_boundary():
    belief = np.zeros((53, 53)); belief[25, 26] = 1
    kernel = np.full((4, 107, 107), .25); kernel[:, 53, 53] = 0
    scores = M.analytic_scores(belief, [26, 26], kernel, np)
    assert scores[0] == -1e-10
    assert M.choice(scores, [0, 1, 2, 3], False, np) == 0
    corner = M.analytic_scores(belief, [0, 0], kernel, np)
    assert corner[0] is None and corner[2] is None
    assert all(math.isfinite(corner[i]) for i in (1, 3))


def test_all_three_arm_names_use_same_actual_mlp8_readout(tmp_path):
    audit = M.Audit(SimpleNamespace(output=tmp_path))
    audit.np = np
    head = M.A.checkpoint(exported(), 'mlp8', .5, np)
    audit.heads = {f'{mode}@10101': head for mode in M.FAMILIES}
    x = np.zeros((16, 11028), np.float64); x[:, 0] = 1
    for arm in audit.heads:
        np.testing.assert_array_equal(audit.predicted(x, arm), np.full(16, .5))
    assert audit.receipt['saved_checkpoint_readout_calls'] == 3
    assert audit.receipt['saved_network_rows'] == 48
