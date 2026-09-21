"""Synthetic orchestration and checkpoint boundaries; no study inputs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_conditioning_runner', ROOT/'scripts/study_otto_conditioning.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_all_fixed_counts_and_independent_autonomous_stage():
    assert S.KINDS == ('gain1', 'gain53')
    assert S.configuration('qualify')['seeds'] == [10101]
    assert S.configuration('study')['seeds'] == [10101, 10102, 10103]
    assert S.limits('qualify')['optimizer_updates'] == 24
    assert S.limits('study')['optimizer_updates'] == 21120
    assert S.limits('study')['native_steps'] == 0
    assert len(S.payload_names('qualify')) == 18
    assert len(S.payload_names('study')) == 30
    assert 'regardless of scalar gate' in S.configuration('study')['policy_stage']


def test_gate_uses_all_seeds_and_direct_unrounded_values():
    values = {f'{kind}@{seed}': {'train': {'mse_normalized': .1}, 'valid': {'mse_normalized': .1}}
              for seed in S.SEEDS for kind in S.KINDS}
    for seed in S.SEEDS:
        values[f'gain53@{seed}']['train']['mse_normalized'] = .06
        values[f'gain53@{seed}']['valid']['mse_normalized'] = .08
    checks = S.gates(values, .02)
    assert len(checks) == 6 and all(c['passes'] for c in checks)
    values['gain53@10103']['valid']['mse_normalized'] = np.nextafter(.9*.1, np.inf)
    assert sum(c['passes'] for c in S.gates(values, .02)) == 5


def test_initial_checkpoints_are_never_overwritten(tmp_path):
    run = S.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.plan = {'configuration': S.configuration('qualify')}
    data = {'version': 'synthetic', 'weights': np.asarray([1., 2.], np.float32)}
    name = 'initial-gain1-10101.npz'
    pin = run.save(name, data)
    original = (tmp_path/name).read_bytes()
    assert run.save(name, data) == pin and (tmp_path/name).read_bytes() == original
    with pytest.raises(ValueError, match='identical initial checkpoint'):
        run.save(name, {**data, 'weights': np.asarray([1., 3.], np.float32)})
    assert (tmp_path/name).read_bytes() == original
    run.save('final-gain1-10101.npz', data)
    with pytest.raises(ValueError, match='only prepublished'):
        run.save('final-gain1-10101.npz', data)


@pytest.mark.parametrize('mismatch', [False, True])
def test_pairing_records_all_raw_rows_before_any_optimizer(tmp_path, mismatch):
    run = S.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.plan = {'configuration': S.configuration('qualify')}
    raw = np.zeros((256, 11028), np.float32)
    raw[:, 0] = np.arange(256, dtype=np.float32)/512
    run.data, run.c0 = {'train': (raw, np.zeros(256, np.float32))}, .4
    work = []

    class Frozen:
        def __init__(self, arrays):
            self.kind = arrays['kind']

        def normalized(self, x):
            result = x[:, 0].copy()
            return result+(1 if mismatch and self.kind == 'gain53' else 0)

    run.model = SimpleNamespace(make_head=lambda kind, seed, c0: kind,
                                export_head=lambda kind: {'kind': kind}, FrozenValue=Frozen)

    def call(channel, operation):
        assert 'optimizer' not in channel
        work.append((channel, dict(run.context)))
        return operation()

    run.call = call
    if mismatch:
        with pytest.raises(ValueError, match='initial represented-function'):
            run.initial_pairing()
        assert run.receipt['initial_pairing_passed'] is False
    else:
        run.initial_pairing()
        assert run.receipt['initial_pairing_passed'] is True
    record = json.loads((tmp_path/'initial-pairing.jsonl').read_text())
    assert record['rows'] == 258 and record['passed'] is not mismatch
    assert set(record['initial_sha256']) == set(S.KINDS)
    assert sum(c == 'initial_pair_forward' for c, _ in work) == 4
    assert record['predictions']['gain1'][-2:] == [0., 0.]


def test_bad_external_plan_stops_before_numeric_import(tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    with pytest.raises(ValueError, match='external plan pin'):
        S.authenticate(SimpleNamespace(plan=plan, plan_sha256='0'*64))


def test_failed_pairing_prevents_fitting_and_preserves_failure(monkeypatch, tmp_path):
    out = tmp_path/'new'
    run = S.Run(SimpleNamespace(output=out))
    monkeypatch.setattr(run, 'bind', lambda: None)
    monkeypatch.setattr(run, 'prepare', lambda: None)

    def fail():
        raise ValueError('pair failed')

    monkeypatch.setattr(run, 'initial_pairing', fail)
    monkeypatch.setattr(run, 'fit', lambda *_: pytest.fail('training after failed initial pairing'))
    with pytest.raises(ValueError, match='pair failed'):
        run.execute()
    assert json.loads((out/'failed.json').read_text())['status'] == 'failed'
    assert not (out/'receipt.json').exists()
