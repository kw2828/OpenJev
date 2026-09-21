"""Fabricated conditioning-control records; no scientific arrays or simulator."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_conditioning_control_audit_test', ROOT/'scripts/audit_otto_conditioning_control.py')
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def rows():
    result = []
    for regime, seed, hit, arm, block in M.evaluation_order():
        steps = 10 if arm.startswith('gain1@') else 9
        result.append({'regime': regime, 'seed': seed, 'initial_hit': hit, 'arm': arm, 'block': block,
                       'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0,
                       'final_update_assimilated': True, 'init_seconds': .1, 'choose_seconds': .2,
                       'update_seconds': .1, 'setup_allocation_seconds': 0., 'controller_seconds': .4,
                       'environment_seconds': .3, 'state_bytes': 22472})
    return result


def mixtures():
    return {regime: {'1': .5, '2': .25, '3': .25} for regime in M.REGIMES}


def test_exact_payload_cohort_rotation_and_thirty_conditions():
    assert len(M.payload_names('qualify')) == 8
    assert len(M.payload_names('study')) == 10
    sample = rows()
    result = M.aggregate(sample, mixtures())
    assert len(sample) == 504 and len({(r['regime'], r['seed']) for r in sample}) == 72
    assert [r['arm'] for r in sample[7:14]] == [*M.ARMS[1:], M.ARMS[0]]
    assert sample[168]['seed'] == 15200001 and sample[168]['arm'] == M.ARMS[24 % 7]
    assert len(result['competence_checks']) == len(result['control_competence_checks']) == 18
    assert len(result['improvement_checks']) == 12 and result['pilot_continuation']
    assert not result['learned_architecture_advantage_established']
    for regime in M.REGIMES:
        for arm in M.ARMS:
            assert result['regimes'][regime]['raw_counts'][arm] == {'found': 24, 'episodes': 24}


def test_relative_gain_between_incompetent_families_does_not_pass():
    sample = rows()
    for row in sample:
        if row['arm'].startswith('gain53@'):
            row['steps'] = row['updates'] = 95
        elif row['arm'].startswith('gain1@'):
            row['steps'] = row['updates'] = 100
    result = M.aggregate(sample, mixtures())
    assert all(c['passes'] for c in result['improvement_checks'])
    assert sum(c['passes'] for c in result['competence_checks']) == 9
    assert not result['pilot_continuation']


def test_descriptive_control_competence_is_not_candidate_gate():
    result = M.aggregate(rows(), mixtures())
    assert not all(c['passes'] for c in result['control_competence_checks'])
    assert result['pilot_continuation']


def test_candidate_seed_failure_cannot_hide_in_family_average():
    sample = rows()
    for row in sample:
        if row['arm'] == 'gain53@10101':
            row['steps'] = row['updates'] = 12
        elif row['arm'].startswith('gain53@'):
            row['steps'] = row['updates'] = 1
    result = M.aggregate(sample, mixtures())
    assert result['regimes']['lambda3']['family_means']['gain53']['steps'] < 9
    assert not result['pilot_continuation']


def test_mixture_weighted_strata_and_equal_seed_average():
    sample = rows()
    for row in sample:
        row['steps'] = row['updates'] = {1: 2, 2: 8, 3: 20}[row['initial_hit']]
    panel = M.aggregate(sample, mixtures())['regimes']['lambda4']
    assert panel['means']['gain53@10101']['steps'] == 8
    assert panel['strata']['3']['gain53@10101']['steps'] == 20
    assert panel['blocks'][0]['gain53@10101'] == 8
    assert 8 != (2+8+20)/3


def test_strict_block_wins_and_no_numeric_gate_tolerance():
    sample = rows()
    for row in sample:
        if row['arm'].startswith('gain53@') and row['block'] >= 5:
            row['steps'] = row['updates'] = 10
    result = M.aggregate(sample, mixtures())
    assert all(c['value'] == 5 and not c['passes'] for c in result['improvement_checks'] if c['name'].endswith('positive_blocks'))
    sample = rows()
    for row in sample:
        if row['arm'].startswith('gain53@'):
            row['choose_seconds'] += 1e-12
            row['controller_seconds'] += 1e-12
    result = M.aggregate(sample, mixtures())
    assert all(not c['passes'] for c in result['improvement_checks'] if c['name'].endswith('.cost'))


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'order', 'nan', 'short_censor', 'missing_update', 'free_cost', 'blocked'])
def test_incomplete_or_malformed_cohort_rejected(defect):
    sample = rows()
    if defect == 'missing':
        sample.pop()
    elif defect == 'duplicate':
        sample[-1] = sample[-2].copy()
    elif defect == 'order':
        sample[0], sample[1] = sample[1], sample[0]
    elif defect == 'nan':
        sample[0]['controller_seconds'] = math.nan
    elif defect == 'short_censor':
        sample[0]['found'] = False
    elif defect == 'missing_update':
        sample[0]['updates'] -= 1
    elif defect == 'blocked':
        sample[0]['blocked_steps'] = 1
    else:
        sample[0]['controller_seconds'] = 0.
    with pytest.raises(ValueError):
        M.aggregate(sample, mixtures())


def test_amortization_removes_only_deployment_allocation():
    result = M.aggregate(rows(), mixtures())
    for panel in result['regimes'].values():
        for arm, metric in panel['means'].items():
            metric['controller_seconds'] = 2.
            metric['setup_allocation_seconds'] = 1. if arm != 'analytic_inbounds' else 0.
    setup = {arm: {'seconds': 6.} for arm in M.ARMS[:-1]}
    fits = dict.fromkeys(setup, 60.)
    values = M.amortization(result, setup, fits, 120., 12.)['lambda3']
    assert values['gain53@10101']['seconds_per_search'] == pytest.approx({'1': 89., '100': 1.88, '10000': 1.0088})
    assert values['gain1@10102']['preparation_share_seconds'] == 20.
    assert values['analytic_inbounds']['seconds_per_search'] == {'1': 2., '100': 2., '10000': 2.}


def parity():
    costs = np.asarray([-.5, -.5+5e-11, 2., 3.], np.float64)
    values = np.repeat(costs-1, 4)
    row = {'fit_id': 'gain53@10101', 'split': 'valid', 'row_index': 0,
           'values_numpy': values.tolist(), 'values_torch': values.tolist(),
           'costs_numpy': costs.tolist(), 'costs_torch': costs.tolist(),
           'allowed_actions': [1, 2, 3], 'action_numpy': 1, 'action_torch': 1, 'passed': True,
           'raw_masses': np.full((4, 4), .25).tolist(), 'weights': np.full((4, 4), .25).tolist(),
           'checkpoint_sha256': 'a'*64, 'posterior_sha256': 'b'*64,
           'position': [0, 26], 'sensing_length': 3., 'eligible_mask': [False, True, True, True]}
    return row, values, costs


def check_parity(row, values, costs, allowed=(1, 2, 3)):
    M.parity_record(row, 'gain53@10101', 'valid', 0, values, costs, list(allowed), np,
                    raw=np.full((4, 4), .25), weights=np.full((4, 4), .25),
                    checkpoint_sha256='a'*64, posterior_sha256='b'*64, position=[0, 26], sensing_length=3.)


def test_branch_only_parity_includes_signed_values_and_exact_allowed_selection():
    row, values, costs = parity()
    check_parity(row, values, costs)
    assert M.E.choice([1.+5e-11, 1., 2., 3.], [0, 1], True, np) == 0
    assert M.E.choice([1.+2e-10, 1., 2., 3.], [0, 1], True, np) == 1


@pytest.mark.parametrize('defect', ['missing_branch', 'nan', 'action', 'allowed', 'row', 'split', 'failed', 'torch_values',
                                  'dtype_bool_action', 'raw', 'weights', 'checkpoint', 'posterior', 'position', 'sensing', 'mask_type'])
def test_bad_qualification_witness_rejected(defect):
    row, values, costs = parity()
    if defect == 'missing_branch':
        row['values_numpy'].pop()
    elif defect == 'nan':
        row['values_numpy'][0] = math.nan
    elif defect == 'action':
        row['action_numpy'] = 0
    elif defect == 'allowed':
        row['allowed_actions'] = [0, 1, 2, 3]
    elif defect == 'row':
        row['row_index'] = 1
    elif defect == 'split':
        row['split'] = 'train'
    elif defect == 'failed':
        row['passed'] = False
    elif defect == 'torch_values':
        row['values_torch'][2] += 1e-6
    elif defect == 'dtype_bool_action':
        row['action_numpy'] = True
    elif defect in ('raw', 'weights'):
        row['raw_masses' if defect == 'raw' else 'weights'][0][0] += 1e-12
    elif defect in ('checkpoint', 'posterior'):
        row[defect+'_sha256'] = 'c'*64
    elif defect == 'position':
        row['position'] = [1, 26]
    elif defect == 'sensing':
        row['sensing_length'] = 4.
    else:
        row['eligible_mask'] = [0, 1, 1, 1]
    with pytest.raises(ValueError):
        check_parity(row, values, costs)


def test_close_costs_do_not_excuse_different_argmin():
    row, values, costs = parity()
    costs[:] = [1., 1., 2., 3.]
    values[:] = np.repeat(costs-1, 4)
    row['values_numpy'] = row['values_torch'] = values.tolist()
    row.update(allowed_actions=[0, 1], action_numpy=0, action_torch=1)
    row['eligible_mask'] = [True, True, False, False]
    row['costs_numpy'] = costs.tolist()
    row['costs_torch'] = [1.+1.5e-10, 1., 2., 3.]
    with pytest.raises(ValueError, match='exact eligible'):
        check_parity(row, values, costs, (0, 1))


def test_complete_synthetic_qualification_journal_and_schema(tmp_path, monkeypatch):
    """All96 rows exercise the complete orchestration, with a constant fake value."""
    run = tmp_path/'run'
    run.mkdir()
    args = SimpleNamespace(mode='qualify', output=tmp_path/'audit', run=run)
    audit = M.Audit(args)
    audit.check = lambda: None
    audit.np, audit.maximum_prediction_error = np, 0.
    probability = np.zeros((53, 53), np.float64)
    probability[10, 10] = 1.
    kernel = np.full((4, 107, 107), .25)
    kernel[:, 53, 53] = 0
    audit.kernels = {'lambda3': kernel}
    metadata = [{'public': {'valid_actions': [0, 1, 2, 3]}, 'regime': 'lambda3'} for _ in range(8)]
    audit.datasets = {split: {'beliefs': np.repeat(probability[None], 8, axis=0),
                             'positions': np.full((8, 2), 26, np.int64), 'sensing_length': np.full(8, 3.),
                             'rows': metadata} for split in ('train', 'valid')}

    def constant_prediction(features, _arm):
        audit.receipt['saved_checkpoint_readout_calls'] += 1
        audit.receipt['saved_network_rows'] += len(features)
        return np.full(len(features), -.5)

    audit.predicted = constant_prediction
    contexts, work, counts = [], [], {}

    def operation(channel, context):
        context = dict(context)
        step = context.pop('step', None)
        if context not in contexts:
            contexts.append(context)
        call_id = len(work)//2+1
        work.extend([[call_id, 0, channel, contexts.index(context), step],
                     [call_id, 1, channel, .001, .001, 0.]])
        item = counts.setdefault(channel, {'attempted': 0, 'returned': 0, 'seconds': 0.})
        item['attempted'] += 1
        item['returned'] += 1
        item['seconds'] += .001

    inputs, setup = {}, {}
    monkeypatch.setattr(M, 'ROOT', tmp_path)
    for arm in M.ARMS[:-1]:
        path = tmp_path/(arm+'.bin')
        path.write_bytes(b'fabricated checkpoint identity only')
        description = M.B.digest(path)
        inputs['head_'+arm.replace('@', '_')] = {'path': str(path), **description}
        setup[arm] = {'seconds': .002, 'instrumented_seconds': .002, 'excluded_io_seconds': 0.,
                      'checkpoint': path.name, 'sha256': description['sha256'], 'allocated_episodes': 0,
                      'storage': M.E.A.head_storage('mlp8')}
        operation('model_load', {'phase': 'inference_setup', 'arm': arm})
    records = []
    for arm in M.ARMS[:-1]:
        operation('parity_restore', {'phase': 'parity_restore', 'fit_id': arm})
        for split in ('train', 'valid'):
            for index in range(8):
                context = {'phase': 'parity', 'fit_id': arm, 'split': split, 'step': index}
                operation('parity_numpy_forward', context)
                operation('parity_torch_forward', context)
                records.append({'fit_id': arm, 'split': split, 'row_index': index,
                    'position': [26, 26], 'sensing_length': 3., 'eligible_mask': [True]*4,
                    'checkpoint_sha256': inputs['head_'+arm.replace('@', '_')]['sha256'],
                    'posterior_sha256': M.hashlib.sha256(probability.tobytes()).hexdigest(),
                    'raw_masses': [[.25]*4 for _ in range(4)], 'weights': [[.25]*4 for _ in range(4)],
                    'values_numpy': [-32.]*16, 'values_torch': [-32.]*16,
                    'costs_numpy': [-31.]*4, 'costs_torch': [-31.]*4, 'allowed_actions': [0, 1, 2, 3],
                    'action_numpy': 0, 'action_torch': 0, 'passed': True})
    prep = {'seconds': .1, 'states': [{'split': s, 'row_index': i} for s in ('train', 'valid') for i in range(8)],
            'scope': 'First eight TRAIN and VALID states; no targets, fitting or policy evaluation'}
    summary = {'version': 'otto-conditioning-control-v1', 'mode': 'qualify', 'parity_records': 96,
               'parity_passed': True, 'completed_episodes': 0, 'preparation_seconds': .1,
               'learned_architecture_advantage_established': False}
    for name, value in (('inference-setup.json', setup), ('preparation.json', prep), ('summary.json', summary)):
        (run/name).write_text(json.dumps(value))
    for name, values in (('work-contexts.jsonl', [{'id': i, 'context': c} for i, c in enumerate(contexts)]),
                         ('work.jsonl', work), ('parity.jsonl', records)):
        (run/name).write_text(''.join(json.dumps(value)+'\n' for value in values))
    audit.begin_work()
    result = audit.qualification({'inputs': inputs}, {'completed_episodes': 0, 'parity_passed': True,
                                  'calls': counts, 'wall_seconds': 2., 'artifact_io_seconds': 0.},
                                 {'shared_setup_seconds': .1, 'model_module_setup_seconds': .1})
    assert result['saved_checkpoint_readout_calls'] == 96 and result['saved_network_rows'] == 1536
    assert result['actual_work'] == {'model_load': 6, 'parity_restore': 6, 'parity_numpy_forward': 96, 'parity_torch_forward': 96}
    assert result['agreement'] and result['maximum_prediction_difference'] == 0.


def test_failure_before_authentication_preserved_without_readouts(tmp_path, monkeypatch):
    args = SimpleNamespace(mode='qualify', output=tmp_path/'exclusive')
    audit = M.Audit(args)
    monkeypatch.setattr(M.B, 'load', lambda *_: (_ for _ in ()).throw(RuntimeError('clock construction failure')))
    with pytest.raises(RuntimeError, match='clock construction'):
        audit.execute()
    failure = json.loads((args.output/'failed.json').read_text())
    assert failure['status'] == 'failed' and not failure['agreement']
    assert failure['saved_checkpoint_readout_calls'] == failure['saved_network_rows'] == 0
    assert not (args.output/'receipt.json').exists()
