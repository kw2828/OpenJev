"""Fabricated runner contracts only; no native, sampled or learned outcomes."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('teacher_learning_runner_fixture', ROOT / 'scripts/study_otto_teacher_learning.py')
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def fabricated_evaluation():
    rows = []
    for regime_index, (regime, first) in enumerate(runner.FIRST.items()):
        for case in range(24):
            shift = (regime_index * 24 + case) % len(runner.ARMS)
            arms = runner.ARMS[shift:] + runner.ARMS[:shift]
            for arm in arms:
                steps = 8 if arm.startswith('continuation@') else 10
                rows.append({'regime': regime, 'seed': first + case, 'case': case,
                             'initial_hit': 1 + case % 3, 'block': case // 3, 'arm': arm,
                             'steps': steps, 'found': True, 'updates': steps, 'blocked_steps': 0,
                             'init_seconds': 0.1, 'choose_seconds': 0.4, 'update_seconds': 0.4,
                             'setup_allocation_seconds': 0.1, 'controller_seconds': 1.0,
                             'environment_seconds': 0.5})
    return rows, {regime: [0.0, 0.2, 0.3, 0.5] for regime in runner.FIRST}


def test_all_33_gates_and_single_unqualified_fit_not_hidden_by_family_means():
    rows, mixtures = fabricated_evaluation()
    assert len(rows) == 504
    result = runner.aggregate(rows, mixtures)
    groups = ('positive_control_checks', 'competence_checks', 'relative_checks')
    assert [len(result[key]) for key in groups] == [3, 18, 12]
    checks = [check for key in groups for check in result[key]]
    assert len(checks) == 33 and all(check['passes'] for check in checks)
    assert result['pilot_continuation'] is True
    # All three continuation fits originally take8 moves. One becomes12,
    # making the family mean28/3 still <=9.5, yet its own12 exceeds10.5.
    changed = [{**row, 'steps': 12, 'updates': 12}
               if row['regime'] == 'lambda3' and row['arm'] == 'continuation@10101'
               else dict(row) for row in rows]
    degraded = runner.aggregate(changed, mixtures)
    assert all(check['passes'] for check in degraded['relative_checks'])
    assert sum(not check['passes'] for check in degraded['competence_checks']) == 1
    assert degraded['pilot_continuation'] is False


def test_checkpoint_hook_reloads_exact_bytes_before_accepting_descriptor(tmp_path):
    exported = {'version': 'fabricated', 'kind': 'dense_augmented', 'input_dim': 2836,
                'weight0': np.array([[0.0, -0.0, 0.5]], dtype=np.float32)}
    good = tmp_path / 'good'
    good.mkdir()
    run = runner.Run(SimpleNamespace(output=good))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    descriptor = run.checkpoint('analytic', 10101, 'final', exported)
    assert descriptor == {'path': 'analytic-10101-final.npz', **runner.digest(good / 'analytic-10101-final.npz')}
    with pytest.raises(FileExistsError):
        run.checkpoint('analytic', 10101, 'final', exported)
    bad = tmp_path / 'bad'
    bad.mkdir()
    run = runner.Run(SimpleNamespace(output=bad))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    original_writer = run.array_file

    def altered_file(name, arrays):
        changed = {**arrays, 'weight0': arrays['weight0'].copy()}
        changed['weight0'][0, 1] = 0.0
        assert np.array_equal(changed['weight0'], arrays['weight0'])
        return original_writer(name, changed)

    run.array_file = altered_file
    with pytest.raises(ValueError, match='exact export identity'):
        run.checkpoint('analytic', 10101, 'final', exported)
    assert (bad / 'analytic-10101-final.npz').is_file()
    assert np.signbit(exported['weight0'][0, 1])


def fake_collection(directory):
    run = runner.Run(SimpleNamespace(output=directory))
    run.np = np
    run.budget = SimpleNamespace(check=lambda **_kwargs: None)
    run.plan = {'bounds': {'sampler_events': 100, 'continuations': 1}}
    run.rows = [{'anchor_id': 0, 'regime': 'lambda3', 'public': {'valid_actions': [0]}}]
    run.beliefs = [np.zeros((53, 53), dtype=np.float64)]
    run.kernels = {'lambda3': object()}
    record = SimpleNamespace(replicate_id=0, first_action=0, steps=1, found=True)
    attempts = []

    def sample(_public, _belief, _kernel, **kwargs):
        attempts.append(True)
        event = {'event': 'attempt', 'operation_id': 0, 'operation': 'source_draw'}
        kwargs['emit'](event)
        kwargs['emit']({**event, 'event': 'return', 'selected_index': 7})
        return [record]

    run.sample = sample
    run.reduce = lambda _records, **_kwargs: {'actions': [{'action': 0, 'mean_cost': 1.0}]}
    return run, attempts


@pytest.mark.parametrize('failure', ['gzip_return', 'durable_panel_return'])
def test_interrupted_panel_retains_pending_work_and_never_counts_panel_complete(tmp_path, monkeypatch, failure):
    run, attempts = fake_collection(tmp_path)
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
                    raise OSError('fabricated compressed return write failure')
                return self.inner.write(data)

        monkeypatch.setattr(runner.gzip, 'GzipFile', FailingGzip)
    else:
        original_emit = run.emit

        def failing_emit(name, value, *, durable=False):
            if name == 'panels.jsonl' and value['event'] == 'return':
                assert durable is True
                raise OSError('fabricated durable panel return failure')
            return original_emit(name, value, durable=durable)

        run.emit = failing_emit
    try:
        with pytest.raises(OSError, match='fabricated'):
            run.collect()
        assert len(attempts) == 1
        assert run.receipt['completed_panels'] == 0
        assert run.pending_panel == {'phase': 'sampling', 'anchor_id': 0}
        assert not (tmp_path / 'collection.json').exists()
        assert json.loads((tmp_path / 'panels.jsonl').read_text())['event'] == 'attempt'
        with original_gzip(filename=str(tmp_path / 'panel-000.jsonl.gz'), mode='rb') as stream:
            saved_events = [json.loads(line) for line in stream]
        if failure == 'gzip_return':
            assert [event['event'] for event in saved_events] == ['attempt']
            assert run.sampler_calls['source_draw'] == {'attempted': 1, 'returned': 0}
            assert set(run.sampler_pending) == {0}
            assert run.pending_emission['event'] == 'return'
        else:
            assert [event['event'] for event in saved_events] == ['attempt', 'return']
            assert run.sampler_calls['source_draw'] == {'attempted': 1, 'returned': 1}
            assert run.sampler_pending == {} and run.pending_emission is None
    finally:
        for stream in run.streams.values():
            stream.close()
