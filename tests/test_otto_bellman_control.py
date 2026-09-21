"""Synthetic orchestration checks; no saved scientific input is opened."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bellman_control_test', ROOT / 'scripts/study_otto_bellman_control.py')
S = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = S
SPEC.loader.exec_module(S)


def rows():
    result = []
    for regime, seed, block, hit, arm in S.evaluation_order(S.FIRST):
        family = arm.split('@')[0]
        steps = 8 if family == 'backup' else 10
        result.append({'regime': regime, 'seed': seed, 'block': block, 'initial_hit': hit, 'arm': arm,
            'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0, 'final_update_assimilated': True,
            'init_seconds': .1, 'choose_seconds': steps * .01, 'update_seconds': steps * .001,
            'setup_allocation_seconds': 0., 'controller_seconds': .1 + steps * .011,
            'environment_seconds': steps * .002, 'state_bytes': 100})
    return result


def test_balanced_fresh_case_order_and_rotation():
    cases = list(S.evaluation_order(S.FIRST))
    assert len(cases) == 1440
    for regime in S.REGIMES:
        values = [v for v in cases if v[0] == regime]
        assert len({v[1] for v in values}) == 48
        for arm in S.ARMS:
            for block in range(8):
                for hit in (1, 2, 3):
                    assert sum(v[2:] == (block, hit, arm) for v in values) == 2
    assert [v[-1] for v in cases[:10]] == list(S.ARMS)
    assert [v[-1] for v in cases[10:20]] == list(S.ARMS[1:] + S.ARMS[:1])


def test_all_gate_checks_and_absolute_competence():
    values = rows()
    mixtures = {r: {1: .2, 2: .3, 3: .5} for r in S.REGIMES}
    result = S.summarize(values, mixtures, S.FIRST)
    assert result['pilot_continuation']
    assert len(result['competence_checks']) == 18
    assert len(result['improvement_checks']) == 24
    for row in values:
        if row['arm'].startswith('backup'):
            row.update(steps=100, updates=100)
        elif row['arm'] != 'analytic_inbounds':
            row.update(steps=200, updates=200)
    result = S.summarize(values, mixtures, S.FIRST)
    assert not result['pilot_continuation']
    assert any(not c['passes'] for c in result['competence_checks'])


def test_missing_episode_or_unaccounted_controller_cost_rejected():
    values = rows()
    mix = {r: {1: .2, 2: .3, 3: .5} for r in S.REGIMES}
    with pytest.raises(ValueError, match='complete ordered'):
        S.summarize(values[:-1], mix, S.FIRST)
    values[0]['controller_seconds'] += 1
    with pytest.raises(ValueError, match='controller costs'):
        S.summarize(values, mix, S.FIRST)


def test_actual_synthetic_learning_refresh_export_audit_pipeline():
    import torch

    from openjev.research import otto_bellman_learning as learning
    from openjev.research import otto_bellman_targets as targets
    from openjev.research import otto_return_value as model
    from openjev.research import otto_value_branches as branches

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='bellman-synthetic-', dir=ROOT / 'output') as temporary:
        parent = Path(temporary)
        prior, out = parent / 'prior', parent / 'run'
        prior.mkdir()
        out.mkdir()
        initial = model.export_head(model.make_head('mlp8', 10101, .25))
        np.savez_compressed(prior / 'final-mlp8-10101.npz', **initial)
        beliefs = np.full((64, 53, 53), 1 / 53**2, dtype=np.float64)
        positions = np.full((64, 2), 26, dtype=np.int64)
        features = model.value_features(S.P.center(beliefs, positions, np), positions, 3., dtype='float32')
        data = {'beliefs': beliefs, 'positions': positions, 'sensing_length': np.full(64, 3., np.float64),
                'features': features, 'target': np.full(64, .5, np.float32), 'row_indices': np.arange(64, dtype=np.int64),
                'metadata': [{'regime': 'lambda3', 'public': {'valid_actions': [0, 1, 2, 3]}} for _ in range(64)]}
        kernel = np.full((4, 107, 107), .25, dtype=np.float64)
        kernel[:, 53, 53] = 0
        runner = S.Run(type('Args', (), {'output': out})())
        runner.plan = {'mode': 'qualify', 'limits': S.limits('qualify')}
        runner.np, runner.torch = np, torch
        runner.model, runner.learning, runner.targets, runner.branches = model, learning, targets, branches
        runner.prior_run, runner.full_training, runner.full_allowed = prior, data, [[0, 1, 2, 3]] * 64
        runner.kernels = {'lambda3': kernel}
        runner.preparation_seconds = 0.
        try:
            runner.fit(data)
            assert runner.receipt['completed_fits'] == 2
            assert runner.receipt['target_refreshes'] == 2
            assert runner.receipt['target_audit_passed'] and runner.receipt['parity_passed']
            assert runner.calls['optimizer_update']['returned'] == 24
            assert runner.calls['target_readout']['returned'] == 128
            assert not runner.pending
            audit = json.loads((out / 'target-audit.json').read_text())
            assert all(r['agreement'] and r['rows'] == 64 for r in audit['refreshes'])
            orders = [json.loads(line) for line in (out / 'epoch-orders.jsonl').read_text().splitlines()]
            assert [r['sha256'] for r in orders[:6]] == [r['sha256'] for r in orders[6:]]
            refresh = [json.loads(line) for line in (out / 'target-refreshes.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in refresh] == [1, 6]
            assert refresh[0]['checkpoint']['sha256'] != refresh[1]['checkpoint']['sha256']
        finally:
            for stream in runner.handles.values():
                stream.close()


def test_final_clock_failure_preserves_primary_and_pending_work(tmp_path):
    runner = S.Run(type('Args', (), {'output': tmp_path / 'run'})())
    runner.calls = {'target_readout': {'attempted': 1, 'returned': 0, 'seconds': 0.}}
    runner.pending = [{'channel': 'target_readout', 'id': 1}]
    class BrokenClock:
        def now_ns(self):
            raise RuntimeError('clock failed')
    def bind():
        runner.clock, runner.start = BrokenClock(), 0
    def setup():
        raise ValueError('primary failure')
    runner.bind, runner.setup = bind, setup
    with pytest.raises(ValueError, match='primary failure'):
        runner.execute()
    receipt = json.loads((runner.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['pending'] == runner.pending
    assert receipt['calls']['target_readout']['attempted'] == 1
    assert 'clock failed' in receipt['finalization_errors'][0]


def test_receipt_write_failure_does_not_replace_primary(tmp_path, monkeypatch):
    runner = S.Run(type('Args', (), {'output': tmp_path / 'run'})())
    runner.bind = lambda: None
    def setup():
        raise ValueError('primary failure')
    def failed_write(path, value):
        raise OSError('disk failed')
    runner.setup = setup
    monkeypatch.setattr(S, 'write', failed_write)
    with pytest.raises(ValueError, match='primary failure') as caught:
        runner.execute()
    assert any('disk failed' in note for note in caught.value.__notes__)
