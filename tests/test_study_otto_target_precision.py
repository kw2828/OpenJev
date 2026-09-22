"""Fabricated integer-merge, target routing and failed-publication contracts."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research.otto_cost_regression import build_targets

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('precision_runner_fixture', ROOT / 'scripts/study_otto_target_precision.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def records(replicates, actions, steps):
    return [{'replicate_id': r, 'first_action': a, 'steps': steps, 'found': True}
            for r in replicates for a in actions]


def test_integer_merge_weights_16_and_48_and_refuses_missing_repeated_or_censored_short_records():
    old, new = records(range(16), (0, 3), 2), records(range(16, 64), (0, 3), 10)
    merged = runner.merge_record_panels(list(reversed(old)), list(reversed(new)), (0, 3))
    assert merged['means'] == [8., None, None, 8.]
    assert merged['integer_sums'] == {'0': 512, '3': 512}
    assert merged['reused_records'] == 32 and merged['new_records'] == 96 and merged['total_records'] == 128
    assert [(r['replicate_id'], r['first_action']) for r in merged['records']] == [
        (r, a) for r in range(64) for a in (0, 3)]
    for bad in (new[:-1], new + new[:1], [{**new[0], 'replicate_id': 0}, *new[1:]],
                [{**new[0], 'found': False}, *new[1:]], [{**new[0], 'steps': True}, *new[1:]]):
        with pytest.raises(ValueError):
            runner.merge_record_panels(old, bad, (0, 3))
    cap = [{**r, 'steps': 2188, 'found': False} for r in new]
    result = runner.merge_record_panels(old, cap, (0, 3))
    assert result['means'][0] == (16 * 2 + 48 * 2188) / 64
    assert sum(not r['found'] for r in result['records']) == 96


def test_runner_preserves_old_features_and_r16_arrays_and_does_not_scale_r64_by_its_own_rms(tmp_path):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    run.rows = [{'episode_id': f'fabricated:{i % 144}'} for i in range(558)]
    episodes = [r['episode_id'] for r in run.rows]
    run.features = np.zeros((558, 2836), dtype=np.float32)
    run.features[:, 0] = np.arange(558, dtype=np.float32)
    run.old_costs = np.tile(np.array([2., 4., np.inf, np.inf]), (558, 1))
    combined = np.tile(np.array([2., 8., np.inf, np.inf]), (558, 1))
    run.allowed = np.isfinite(run.old_costs)
    run.reference = build_targets(run.old_costs, run.allowed, episodes, kind='continuation')
    run.plan = {'inputs': {f'{runner.BASE}/run-01/training-data.npz': {'sha256': 'a' * 64, 'bytes': 1}}}
    features, bundle = run.training_data(combined)
    assert features is run.features
    assert bundle['r16']['scale'] == bundle['r64']['scale'] == run.reference['scale']
    for name, value in run.reference.items():
        if isinstance(value, np.ndarray):
            assert bundle['r16'][name].tobytes() == value.tobytes()
    assert np.array_equal(bundle['r64']['scaled_float32'][0], np.array([-3., 3., 0., 0.], dtype=np.float32))
    with np.load(tmp_path / 'training-data.npz', allow_pickle=False) as saved:
        assert saved['features'].tobytes() == run.features.tobytes()
        assert saved['r16_centered_float64'].tobytes() == run.reference['centered_float64'].tobytes()
        assert saved['r64_costs'].tobytes() == combined.tobytes()
    info = json.loads((tmp_path / 'training-data.json').read_text())
    assert info['r64_unfloored_rms_diagnostic_only'] == pytest.approx(3.)
    assert info['scales']['r64'] == pytest.approx(1.)


@pytest.mark.parametrize('failure', ['gzip_return', 'durable_panel_return'])
def test_new_panel_failure_keeps_original_records_pending_work_and_no_retry(tmp_path, monkeypatch, failure):
    run = runner.Run(SimpleNamespace(output=tmp_path))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    run.plan = {'bounds': {'sampler_events': 100, 'continuations': 48}}
    run.rows = [{'anchor_id': 0, 'regime': 'lambda3', 'public': {'valid_actions': [0]}}]
    run.beliefs, run.kernels = [np.zeros((53, 53))], {'lambda3': object()}
    original = records(range(16), (0,), 1)
    run.old_panels = [{'records': original, 'file': {'path': 'inherited', 'sha256': 'b' * 64, 'bytes': 1}}]
    run.record_class = SimpleNamespace
    calls = []

    def sample(_public, _belief, _kernel, **kwargs):
        calls.append(tuple(kwargs['replicate_ids']))
        event = {'event': 'attempt', 'operation_id': 1, 'operation': 'source_draw'}
        kwargs['emit'](event)
        kwargs['emit']({**event, 'event': 'return', 'selected_index': 7})
        return [SimpleNamespace(**r) for r in records(range(16, 64), (0,), 1)]

    run.sample = sample
    run.reduce = lambda rows, **_kwargs: {'actions': [{'action': 0, 'mean_cost': sum(r.steps for r in rows) / len(rows)}]}
    original_gzip = gzip.GzipFile
    if failure == 'gzip_return':
        class FailingGzip:
            def __init__(self, *args, **kwargs):
                self.inner = original_gzip(*args, **kwargs)

            def __enter__(self):
                self.inner.__enter__()
                return self

            def __exit__(self, *args):
                return self.inner.__exit__(*args)

            def write(self, data):
                if json.loads(data)['event'] == 'return':
                    raise OSError('fabricated gzip return failure')
                return self.inner.write(data)

        monkeypatch.setattr(runner.gzip, 'GzipFile', FailingGzip)
    else:
        emit = run.emit

        def failed_boundary(name, value, *, durable=False):
            if name == 'panels.jsonl' and value['event'] == 'return':
                assert durable
                raise OSError('fabricated durable panel failure')
            emit(name, value, durable=durable)

        run.emit = failed_boundary
    try:
        with pytest.raises(OSError, match='fabricated'):
            run.collect()
        assert calls == [tuple(range(16, 64))]
        assert original == records(range(16), (0,), 1)
        assert run.receipt['completed_panels'] == 0
        assert run.pending_panel == {'phase': 'sampling', 'anchor_id': 0}
        assert not (tmp_path / 'collection.json').exists()
        if failure == 'gzip_return':
            assert run.sampler_calls['source_draw'] == {'attempted': 1, 'returned': 0}
            assert set(run.sampler_pending) == {1}
            assert run.pending_emission['event'] == 'return'
        else:
            assert run.sampler_calls['source_draw'] == {'attempted': 1, 'returned': 1}
            assert run.sampler_pending == {} and run.pending_emission is None
    finally:
        for stream in run.streams.values():
            stream.close()
