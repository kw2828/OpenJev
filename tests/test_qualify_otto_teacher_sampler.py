"""Nested draw-journal regressions using only fabricated, nonnumerical objects.

No native environment, sampler, NumPy, model or empirical input is imported.
The real qualifier's call wrapper and failure receipt writer are exercised with
fake construction/step functions to preserve completed and pending draw work.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def qualifier():
    path = Path(__file__).resolve().parents[1] / 'scripts/qualify_otto_teacher_sampler.py'
    spec = importlib.util.spec_from_file_location('_fabricated_teacher_qualifier_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(qualifier, tmp_path):
    tmp_path = tmp_path.resolve()
    run = qualifier.Run(SimpleNamespace(
        output=tmp_path / 'run', plan=tmp_path / 'unused-plan.json',
        supervision=tmp_path / 'unused-launch.json'))
    # Isolate journal semantics from clocks, manifests and numerical execution.
    run.check = lambda: None
    run.bind = lambda: None
    return run


def fake_seeded_class(run):
    source_vector, hit_vector = [0.25, 0.75], [0., 1.]

    class OriginalNative:
        def __init__(self, *, fail_after_source=False):
            self._draw('source', source_vector)
            if fail_after_source:
                raise RuntimeError('constructor failed after source draw')

        def step(self, *, fail_after_hit=False):
            index = self._draw('hit', hit_vector)
            if fail_after_hit:
                raise RuntimeError('step failed after hit draw')
            return index

    recorded = run.recorded_native(OriginalNative)

    class FakeSeeded(recorded):
        def __init__(self, *, fail_after_source=False, fail_hit_draw=False):
            self._public_draw_counts = {'initial': 0, 'source': 0, 'hit': 0}
            self._draw_log = []
            self.original_calls = []
            self.fail_hit_draw = fail_hit_draw
            super().__init__(fail_after_source=fail_after_source)

        def _draw(self, channel, probabilities):
            # These identity checks detect copying/replacement of arguments.
            assert probabilities is (source_vector if channel == 'source' else hit_vector)
            self.original_calls.append(channel)
            if channel == 'hit' and self.fail_hit_draw:
                raise ArithmeticError('original hit draw failed')
            ordinal = self._public_draw_counts[channel]
            self._draw_log.append({
                'channel': channel, 'index': ordinal, 'probabilities': list(probabilities),
                'cdf_mass': 1., 'uniform': .5, 'selected_index': 1,
            })
            self._public_draw_counts[channel] += 1
            return 1

        @property
        def draw_log(self):
            return tuple(copy.deepcopy(self._draw_log))

    return FakeSeeded


def events(run):
    return [json.loads(line) for line in (run.out / 'work.jsonl').read_text().splitlines()]


def failure_receipt(run):
    receipt = json.loads((run.out / 'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['qualified'] is False
    assert receipt['requires_successful_original_supervisor'] is True
    assert not (run.out / 'summary.json').exists()
    return receipt


def test_returned_initialization_draw_survives_outer_constructor_failure(qualifier, tmp_path):
    run = make_run(qualifier, tmp_path)
    fake = fake_seeded_class(run)
    run.body = lambda: run.call('native_construction', {'fixture': 'constructor'},
                                fake, fail_after_source=True)
    with pytest.raises(RuntimeError, match='constructor failed after source draw'):
        run.execute()
    receipt = failure_receipt(run)
    assert receipt['calls'] == {
        'native_construction': {'attempted': 1, 'returned': 0},
        'native_source_draw': {'attempted': 1, 'returned': 1},
    }
    assert [frame['operation'] for frame in receipt['operation_stack']] == ['native_construction']
    assert receipt['pending']['operation'] == 'native_construction'
    saved, = [row for row in events(run)
              if row['event'] == 'return' and row['operation'] == 'native_source_draw']
    assert saved['draw'] == {
        'channel': 'source', 'index': 0, 'probabilities': [0.25, 0.75],
        'cdf_mass': 1., 'uniform': .5, 'selected_index': 1,
    }
    assert saved['parent'] == {'operation': 'native_construction', 'ordinal': 1}
    assert saved['native_draw_counts'] == {'initial': 0, 'source': 1, 'hit': 0}


def test_returned_hit_draw_survives_outer_step_failure(qualifier, tmp_path):
    run = make_run(qualifier, tmp_path)
    fake = fake_seeded_class(run)

    def body():
        env = run.call('native_construction', {}, fake)
        run.call('native_step', {'fixture': 'filter_failure'}, env.step, fail_after_hit=True)

    run.body = body
    with pytest.raises(RuntimeError, match='step failed after hit draw'):
        run.execute()
    receipt = failure_receipt(run)
    assert receipt['calls']['native_step'] == {'attempted': 1, 'returned': 0}
    assert receipt['calls']['native_hit_draw'] == {'attempted': 1, 'returned': 1}
    assert [frame['operation'] for frame in receipt['operation_stack']] == ['native_step']
    saved, = [row for row in events(run)
              if row['event'] == 'return' and row['operation'] == 'native_hit_draw']
    assert saved['draw']['probabilities'] == [0., 1.]
    assert saved['draw']['selected_index'] == 1
    assert saved['parent'] == {'operation': 'native_step', 'ordinal': 1}
    assert saved['native_draw_counts'] == {'initial': 0, 'source': 1, 'hit': 1}


def test_failed_nested_draw_retains_both_pending_frames_without_return(qualifier, tmp_path):
    run = make_run(qualifier, tmp_path)
    fake = fake_seeded_class(run)

    def body():
        env = run.call('native_construction', {}, fake, fail_hit_draw=True)
        run.call('native_step', {'fixture': 'draw_failure'}, env.step)

    run.body = body
    with pytest.raises(ArithmeticError, match='original hit draw failed'):
        run.execute()
    receipt = failure_receipt(run)
    assert receipt['calls']['native_step'] == {'attempted': 1, 'returned': 0}
    assert receipt['calls']['native_hit_draw'] == {'attempted': 1, 'returned': 0}
    assert [frame['operation'] for frame in receipt['operation_stack']] == [
        'native_step', 'native_hit_draw']
    assert receipt['pending']['operation'] == 'native_hit_draw'
    assert not any(row['event'] == 'return' and row['operation'] in ('native_step', 'native_hit_draw')
                   for row in events(run))


def test_successful_wrapper_delegates_once_and_closes_all_nested_operations(qualifier, tmp_path):
    run = make_run(qualifier, tmp_path)
    run.out.mkdir()
    fake = fake_seeded_class(run)
    env = run.call('native_construction', {'fixture': 'normal'}, fake)
    assert run.call('native_step', {'fixture': 'normal'}, env.step) == 1
    assert env.original_calls == ['source', 'hit']
    assert run.calls == {name: {'attempted': 1, 'returned': 1} for name in (
        'native_construction', 'native_source_draw', 'native_step', 'native_hit_draw')}
    assert run.operation_stack == [] and run.pending is None
    ledger = events(run)
    assert [(row['event'], row['operation']) for row in ledger] == [
        ('attempt', 'native_construction'), ('attempt', 'native_source_draw'),
        ('return', 'native_source_draw'), ('return', 'native_construction'),
        ('attempt', 'native_step'), ('attempt', 'native_hit_draw'),
        ('return', 'native_hit_draw'), ('return', 'native_step'),
    ]
    for row in ledger:
        if row['event'] == 'return' and 'draw' in row:
            channel = row['draw']['channel']
            original, = [entry for entry in env.draw_log if entry['channel'] == channel]
            assert row['draw'] == original


def test_resource_check_after_completed_draw_preserves_its_return_evidence(qualifier, tmp_path):
    run = make_run(qualifier, tmp_path)
    fake = fake_seeded_class(run)

    def check():
        if run.calls.get('native_hit_draw', {}).get('returned') == 1:
            raise TimeoutError('bound reached after returned draw')

    run.check = check

    def body():
        env = run.call('native_construction', {}, fake)
        run.call('native_step', {'fixture': 'post_draw_bound'}, env.step)

    run.body = body
    with pytest.raises(TimeoutError, match='bound reached after returned draw'):
        run.execute()
    receipt = failure_receipt(run)
    assert receipt['calls']['native_hit_draw'] == {'attempted': 1, 'returned': 1}
    assert receipt['calls']['native_step'] == {'attempted': 1, 'returned': 0}
    assert [frame['operation'] for frame in receipt['operation_stack']] == ['native_step']
    saved, = [row for row in events(run)
              if row['event'] == 'return' and row['operation'] == 'native_hit_draw']
    assert saved['draw']['selected_index'] == 1
