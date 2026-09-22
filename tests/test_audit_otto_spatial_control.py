"""Fabricated saved records only; no checkpoint, Torch or simulator calls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_spatial_control_audit_test', ROOT/'scripts/audit_otto_spatial_control.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def rows():
    result = []
    for regime, seed, hit, arm, block in M.evaluation_order():
        steps = 9 if arm.startswith('spatial@') or arm == 'analytic_inbounds' else 10
        result.append({'regime': regime, 'seed': seed, 'initial_hit': hit, 'arm': arm, 'block': block,
                       'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0,
                       'final_update_assimilated': True, 'init_seconds': .1, 'choose_seconds': .2,
                       'update_seconds': .1, 'setup_allocation_seconds': 0., 'controller_seconds': .4,
                       'environment_seconds': .3, 'state_bytes': 22472})
    return result


def mixtures():
    return {regime: {'1': .5, '2': .25, '3': .25} for regime in M.REGIMES}


def test_exact_all_model_cohort_rotation_and_66_gate_denominators():
    values = rows()
    assert len(values) == 1152
    assert len({(r['regime'], r['seed']) for r in values}) == 72
    assert [r['arm'] for r in values[16:32]] == [*M.ARMS[1:], M.ARMS[0]]
    assert values[384]['seed'] == 16200001 and values[384]['arm'] == M.ARMS[24 % 16]
    summary = M.aggregate(values, mixtures())
    assert len(summary['competence_checks']) == 18
    assert len(summary['improvement_checks']) == 48
    assert len(summary['control_competence_checks']) == 72
    assert summary['pilot_continuation']
    assert not all(r['passes'] for r in summary['control_competence_checks'])
    assert not summary['learned_architecture_advantage_established']
    assert len(M.payload_names('qualify')) == len(M.payload_names('study')) == 11


def test_mixture_weights_and_seed_averaging_are_separate():
    values = rows()
    for row in values:
        row['steps'] = row['updates'] = {1: 2, 2: 8, 3: 20}[row['initial_hit']]
        if row['arm'] == 'spatial@10103':
            row['steps'] += 3
            row['updates'] += 3
    panel = M.aggregate(values, mixtures())['regimes']['lambda4']
    assert panel['means']['spatial@10101']['steps'] == 8
    assert panel['family_means']['spatial']['steps'] == 9
    assert panel['blocks'][0]['spatial@10103'] == 11
    assert panel['strata']['3']['spatial@10101']['steps'] == 20


def test_relative_improvement_cannot_replace_per_seed_competence():
    values = rows()
    for row in values:
        if row['arm'].startswith('spatial@'):
            row['steps'] = row['updates'] = 95
        elif row['arm'] != 'analytic_inbounds':
            row['steps'] = row['updates'] = 100
    summary = M.aggregate(values, mixtures())
    assert all(r['passes'] for r in summary['improvement_checks'])
    assert sum(r['passes'] for r in summary['competence_checks']) == 9
    assert not summary['pilot_continuation']


def test_one_candidate_seed_cannot_hide_and_every_control_family_matters():
    values = rows()
    for row in values:
        if row['arm'] == 'spatial@10101':
            row['steps'] = row['updates'] = 12
        elif row['arm'].startswith('spatial@'):
            row['steps'] = row['updates'] = 1
    summary = M.aggregate(values, mixtures())
    assert summary['regimes']['lambda3']['family_means']['spatial']['steps'] < 9
    assert not summary['pilot_continuation']
    values = rows()
    for row in values:
        if row['arm'].startswith('statistics@'):
            row['steps'] = row['updates'] = 8
    checks = M.aggregate(values, mixtures())['improvement_checks']
    assert all(r['passes'] for r in checks if '.statistics.' not in r['name'])
    assert any(not r['passes'] for r in checks if '.statistics.moves' in r['name'])


def test_strict_block_gain_and_cost_comparison_have_no_epsilon():
    values = rows()
    for row in values:
        if row['arm'].startswith('spatial@') and row['block'] >= 5:
            row['steps'] = row['updates'] = 10
    checks = M.aggregate(values, mixtures())['improvement_checks']
    assert all(r['value'] == 5 and not r['passes'] for r in checks if r['name'].endswith('positive_blocks'))
    values = rows()
    for row in values:
        if row['arm'].startswith('spatial@'):
            row['choose_seconds'] += 1e-12
            row['controller_seconds'] += 1e-12
    checks = M.aggregate(values, mixtures())['improvement_checks']
    assert all(not r['passes'] for r in checks if r['name'].endswith('.cost'))


@pytest.mark.parametrize('defect', ['missing', 'order', 'short_censor', 'missing_update', 'blocked', 'free_cost', 'nonfinite'])
def test_invalid_complete_cohort_fails(defect):
    values = rows()
    if defect == 'missing':
        values.pop()
    elif defect == 'order':
        values[0], values[1] = values[1], values[0]
    elif defect == 'short_censor':
        values[0]['found'] = False
    elif defect == 'missing_update':
        values[0]['updates'] -= 1
    elif defect == 'blocked':
        values[0]['blocked_steps'] = 1
    else:
        values[0]['controller_seconds'] = 0. if defect == 'free_cost' else float('nan')
    with pytest.raises(ValueError):
        M.aggregate(values, mixtures())


def test_amortization_does_not_double_charge_shared_setup():
    summary = M.aggregate(rows(), mixtures())
    for panel in summary['regimes'].values():
        for arm, row in panel['means'].items():
            row['controller_seconds'] = 2.
            row['setup_allocation_seconds'] = 1. if arm != 'analytic_inbounds' else 0.
    setup = {arm: {'seconds': 6.} for arm in M.ARMS[:-1]}
    result = M.amortization(summary, setup, dict.fromkeys(setup, 60.), 300., 30.)['lambda3']
    assert result['spatial@10101']['seconds_per_search'] == pytest.approx({'1': 89., '100': 1.88, '10000': 1.0088})
    assert result['spatial@10101']['preparation_share_seconds'] == 20.
    assert result['spatial@10101']['deployment_setup_seconds'] == 8.
    assert result['analytic_inbounds']['seconds_per_search'] == {'1': 2., '100': 2., '10000': 2.}


def test_all_52_tuple_ids_and_explicit_point_successors():
    identities = M.qualification_ids()
    assert len(identities) == len({r['tuple_id'] for r in identities}) == 52
    assert identities[0]['tuple_id'] == 'train:0' and identities[15]['tuple_id'] == 'valid:7'
    assert identities[16]['tuple_id'] == 'lambda3:center:asymmetric'
    assert identities[-1]['tuple_id'] == 'lambda5:upper:subfloor'
    for name, expected in [('center', (25, 26)), ('lower', (1, 0)), ('upper', (51, 52))]:
        p, q = M.synthetic_belief(name, 'point_successor', np)
        assert p.sum() == 1. and p[expected] == 1. and p[tuple(q)] == 0.
        small, _ = M.synthetic_belief(name, 'subfloor', np)
        assert small.sum() == pytest.approx(1e-12, rel=1e-15, abs=0.)
        zero, _ = M.synthetic_belief(name, 'zero', np)
        assert not zero.any()


def test_branch_geometry_floor_and_biased_empty_values_use_independent_sparse_oracle():
    p = np.zeros((53, 53), np.float64)
    p[1, 0], p[2, 1] = .3, .2
    kernel = np.empty((4, 107, 107), np.float64)
    for h in range(4):
        kernel[h].fill((h+1)/10)
    kernel[:, 53, 53] = 0.
    u, z, q, raw, weights = M.branches(p, [0, 0], kernel, np)
    assert np.array_equal(q[:4], np.zeros((4, 2), np.int64))
    assert np.array_equal(q[4:8], np.tile([1, 0], (4, 1)))
    assert np.allclose(raw[0], [.05, .1, .15, .2], rtol=0, atol=1e-16)
    assert np.allclose(raw[1], [.02, .04, .06, .08], rtol=0, atol=1e-16)
    assert u[4, 53, 53] == pytest.approx(.02)
    assert z[4, 53, 53] == 1.
    assert not z[:, 52, 52].any()
    _, empty, _, zero_raw, floored = M.branches(np.zeros_like(p), [0, 0], kernel, np)
    assert not empty.any() and not zero_raw.any()
    assert np.array_equal(floored, np.full((4, 4), 1e-10))
    assert np.array_equal(weights, np.maximum(raw, 1e-10))
    # A generic biased readout may return -2 on an empty branch. It is retained.
    costs = M.E.costs(np.full(16, -2., np.float64), floored, np)
    assert np.array_equal(costs, np.full(4, 1.-8e-10))


def test_exact_action_tie_is_separate_from_float64_parity():
    a = np.asarray([1., 1., 2., 3.], np.float64)
    b = np.asarray([1.+1.5e-10, 1., 2., 3.], np.float64)
    assert M.close_arrays(a, b, np, 'close physical costs') < 1e-8
    assert M.E.choice(a.tolist(), [0, 1], True, np) == 0
    assert M.E.choice(b.tolist(), [0, 1], True, np) == 1


@pytest.mark.parametrize('mode', ['qualify', 'study'])
def test_serialized_full_cap_replay_journal_fits_with_reserve(mode):
    projected = M.journal_projection(mode)
    n = M.MAX_CALLS[mode]
    actual = len(M.compact_bytes([n, 0, 14, 1151, 2188, 16]))+len(M.compact_bytes([n, 1, M.LIMITS[mode]['native_seconds']*10**9]))
    assert projected['maximum_journal_bytes'] == n*actual
    assert projected['maximum_journal_bytes']+projected['metadata_reserve_bytes'] < M.LIMITS[mode]['output_bytes']


def bare_audit(tmp_path, mode='qualify'):
    args = SimpleNamespace(mode=mode, output=tmp_path/'out')
    audit = M.Audit(args)
    audit.out.mkdir()
    audit.np, audit.check = np, lambda: None
    ticks = iter(range(0, 10000, 10))
    audit.clock = SimpleNamespace(now_ns=lambda: next(ticks))
    return audit


def test_actual_readout_failure_preserves_attempt_and_completed_history(tmp_path):
    audit = bare_audit(tmp_path)
    class FakeHead:
        calls = 0

        def normalized(self, *_):
            self.calls += 1
            if self.calls == 2:
                raise ArithmeticError('synthetic second readout')
            return np.arange(16, dtype=np.float64)
    head = FakeHead()
    audit.heads = {'spatial@10101': head}
    z, q = np.zeros((16, 105, 105)), np.zeros((16, 2), np.int64)
    audit.predicted(z, q, 3., 'spatial@10101')
    with pytest.raises(ArithmeticError, match='synthetic second'):
        audit.predicted(z, q, 3., 'spatial@10101')
    ledger = [json.loads(line) for line in (audit.out/'readouts.jsonl').read_text().splitlines()]
    assert [r[:2] for r in ledger] == [[1, 0], [1, 1], [2, 0]]
    assert audit.receipt['readout_attempts'] == 2 and audit.receipt['readout_returns'] == 1
    assert audit.receipt['saved_network_rows'] == 16 and audit.receipt['pending'][0] == 2


def test_external_pin_failure_precedes_decoding_or_model_import(tmp_path, monkeypatch):
    for key, value in M.THREADS.items():
        monkeypatch.setenv(key, value)
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    args = SimpleNamespace(mode='qualify', output=tmp_path/'out', plan=plan, run=tmp_path/'run',
                           terminal=tmp_path/'terminal.json', plan_sha256='0'*64,
                           receipt_sha256='0'*64, terminal_sha256='0'*64)
    audit = M.Audit(args)
    audit.check = lambda: None
    def forbidden(*_):
        raise AssertionError('decoding happened before external pin check')
    monkeypatch.setattr(M.B, 'read', forbidden)
    with pytest.raises(ValueError, match='external identity'):
        audit.authenticate()
    assert audit.model is None
    assert hashlib.sha256(plan.read_bytes()).hexdigest() != args.plan_sha256


def test_qualification_cache_never_decodes_targets_or_legacy_features():
    class PublicOnlyArchive:
        files = ('beliefs', 'positions', 'sensing_length', 'target', 'features')

        def __init__(self):
            self.requested = []
            self.values = {'beliefs': np.zeros((9, 53, 53), np.float64),
                           'positions': np.zeros((9, 2), np.int64), 'sensing_length': np.full(9, 3., np.float64)}

        def __getitem__(self, key):
            if key in ('target', 'features'):
                raise AssertionError('qualification accessed non-public cached column')
            self.requested.append(key)
            return self.values[key]
    archive = PublicOnlyArchive()
    selected = M.selected_public_cache(archive, 9, np)
    assert archive.requested == ['beliefs', 'positions', 'sensing_length']
    assert set(selected) == set(archive.values)
    assert all(len(v) == 8 for v in selected.values())
    selected['beliefs'][0, 0, 0] = 1.
    assert archive.values['beliefs'][0, 0, 0] == 0.


def test_late_publication_failure_demotes_completed_receipt(tmp_path, monkeypatch):
    args = SimpleNamespace(mode='qualify', output=tmp_path/'audit')
    audit = M.Audit(args)
    clock = SimpleNamespace(now_ns=lambda: 100, backend='synthetic')
    monkeypatch.setattr(M.B, 'load', lambda *_: SimpleNamespace(SuspendClock=lambda: clock))
    monkeypatch.setattr(audit, 'authenticate', lambda: ({}, {}))
    monkeypatch.setattr(audit, 'compute', lambda *_: {'agreement': True})
    def check():
        if (audit.out/'receipt.json').exists():
            raise RuntimeError('synthetic late cap failure')
    monkeypatch.setattr(audit, 'check', check)
    with pytest.raises(RuntimeError, match='synthetic late cap'):
        audit.execute()
    invalid = json.loads((audit.out/'invalid-completed-receipt.json').read_text())
    failed = json.loads((audit.out/'receipt.json').read_text())
    assert invalid['status'] == 'completed' and invalid['agreement']
    assert failed['status'] == 'failed' and not failed['agreement']
    assert (audit.out/'failed.json').exists()
