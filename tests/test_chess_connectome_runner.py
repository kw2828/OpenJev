"""No real chess labels: strict gate arithmetic and orchestration checks."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('connectome_runner', ROOT/'scripts/chess_connectome_study.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def passing_losses():
    return {runner.name(c): {'dev': .10 if c['variant'] == 'biological' else .2,
                            'shift': .20 if c['variant'] == 'biological' else .4}
            for c in runner.baselines()+runner.configurations()}


def test_exact_all_control_and_seed_gate():
    result = runner.continuation(passing_losses())
    assert result['continuation_passed']
    assert len(result['checks']) == 30
    assert result['means']['biological'] == pytest.approx({'dev': .1, 'shift': .2})


@pytest.mark.parametrize('comparator', ['rewire151', 'rewire163', 'rewire179'])
@pytest.mark.parametrize('split', ['dev', 'shift'])
def test_every_comparator_panel_and_seed_required(comparator, split):
    losses = passing_losses()
    losses[f'{comparator}-97'][split] = losses['biological-97'][split]
    result = runner.continuation(losses)
    assert not result['continuation_passed']
    assert any(c['criterion'] == 'strict_paired_improvement' and not c['passed'] for c in result['checks'])


def test_no_gain_against_zero_or_negative_reference_mean():
    for reference in (0., -.1):
        losses = passing_losses()
        for seed in (97, 109, 127):
            losses[f'rewire151-{seed}']['dev'] = reference
            losses[f'biological-{seed}']['dev'] = reference-.1
        assert not runner.continuation(losses)['continuation_passed']


def test_baseline_degradation_in_one_seed_blocks_even_with_better_mean():
    losses = passing_losses()
    losses['direct-97']['dev'] = .09
    result = runner.continuation(losses)
    assert not result['continuation_passed']
    assert result['means']['direct']['dev'] > result['means']['biological']['dev']


def test_relative_threshold_and_tolerance():
    losses = passing_losses()
    for seed in (97, 109, 127):
        losses[f'biological-{seed}']['dev'] = .18
        losses[f'biological-{seed}']['shift'] = .36
    assert runner.continuation(losses)['continuation_passed']
    losses['biological-97']['dev'] += 1e-8
    assert not runner.continuation(losses)['continuation_passed']


@pytest.mark.parametrize('fault', ['missing_model', 'missing_panel', 'nan', 'bool'])
def test_incomplete_or_nonfinite_cannot_pass(fault):
    losses = passing_losses()
    if fault == 'missing_model':
        losses.pop('dense-97')
    elif fault == 'missing_panel':
        losses['node_local-127'].pop('shift')
    else:
        losses['biological-97']['dev'] = float('nan') if fault == 'nan' else True
    with pytest.raises(ValueError):
        runner.continuation(losses)


def test_selection_fixed_distinct_and_split_boundaries():
    records = {s: [{'id': f'{s}-{i}'} for i in range(2048)] for s in ('dev', 'shift')}
    first, combined = runner.panels(records)
    assert runner.panels(records) == (first, combined)
    assert len(first['latency_indices']) == len(set(first['latency_indices'])) == 128
    assert sum(i < 2048 for i in first['latency_indices']) == 64
    assert len(combined) == 4096
    assert all(len(indices) == len(set(indices)) == 128 for indices in first['secondary_indices'].values())


def test_outputs_are_exclusive_regular_and_private(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    assert runner.private_output(tmp_path/'runs/study') == tmp_path/'runs/study'
    with pytest.raises(ValueError):
        runner.private_output(tmp_path/'models/weights')
    out = tmp_path/'completed'
    out.mkdir()
    runner.write(out/'data.json', {'ok': True})
    with pytest.raises(FileExistsError):
        runner.write(out/'data.json', {})
    runner.write(out/'completed.json', {'status': 'completed', 'files': runner.tree(out)})
    assert runner.completed(out)['status'] == 'completed'
    (out/'data.json').write_text('tampered')
    with pytest.raises(ValueError):
        runner.completed(out)
    (out/'symlink').symlink_to(out/'data.json')
    with pytest.raises(ValueError):
        runner.tree(out)


def test_prepare_and_verify_bind_sources_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    marker = tmp_path/'old-evidence.json'
    marker.write_text('{}')
    state = {'sources': {'synthetic': 'a'*64}, 'graph_hashes': {}}
    monkeypatch.setattr(runner, 'binding', lambda: copy.deepcopy(state))
    monkeypatch.setattr(runner, 'graphs', lambda _: {})
    monkeypatch.setattr(runner.data, 'collect_exclusions', lambda _: {
        'states': ['state-a', 'state-b'], 'files': {'old-evidence.json': runner.sha(marker)},
        'counts': {'total_natural_states': 2}, 'scope': 'synthetic-only'})
    plan = runner.prepare(tmp_path/'protocol')
    actual, excluded = runner.verify(plan)
    assert actual['configurations'] == runner.configurations()
    assert excluded == ['state-a', 'state-b']
    state['sources']['synthetic'] = 'b'*64
    with pytest.raises(ValueError, match='Frozen protocol'):
        runner.verify(plan)
    state['sources']['synthetic'] = 'a'*64
    marker.write_text('changed')
    with pytest.raises(ValueError, match='Exposure source'):
        runner.verify(plan)
    marker.write_text('{}')
    (plan.parent/'excluded-states.json.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='snapshot'):
        runner.verify(plan)


def test_edited_plan_fails_prepared_receipt_before_other_checks(tmp_path):
    plan = tmp_path/'plan.json'
    runner.write(plan, {'fixture': 'original'})
    runner.write(tmp_path/'prepared.json', {'status': 'prepared', 'plan_sha256': runner.sha(plan),
                                           'neural_evaluation_started': False})
    plan.write_text('{"fixture":"changed"}')
    with pytest.raises(ValueError, match='Prepared plan identity'):
        runner.verify(plan)


def stubbed_run(tmp_path, monkeypatch, fail_fit=None):
    """Test execution ordering with stand-ins, never model or engine calls."""
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = {'configurations': runner.configurations(), 'baselines': runner.baselines(),
            'binding': {'graph_hashes': {}, 'training_source_sha256': 'b'*64, 'engine_sha256': 'c'*64}}
    plan_path = tmp_path/'plan.json'
    runner.write(plan_path, plan)
    events = []
    records = {s: [{'id': s, 'fen': 'unused'}] for s in ('dev', 'shift')}
    monkeypatch.setattr(runner, 'verify', lambda _: (copy.deepcopy(plan), []))
    monkeypatch.setattr(runner, 'graphs', lambda _: {v: object() for v in runner.PROTOCOL['variants']})
    monkeypatch.setattr(runner.data, 'generate', lambda out, *_: out.mkdir())
    monkeypatch.setattr(runner.data, 'validate', lambda *_: records)
    monkeypatch.setattr(runner.data, 'training_rows', lambda _: records['dev'])
    monkeypatch.setattr(runner, 'panels', lambda _: ({'secondary_indices': {'dev': [0], 'shift': [0]},
                                                   'latency_indices': [0, 1]}, records['dev']+records['shift']))

    class FakeCache:
        def __init__(self, rows, **kwargs):
            self.rows = rows
            self.metadata = {'fixture': True}

    monkeypatch.setattr(runner, 'CachedPositions', FakeCache)
    monkeypatch.setattr(runner, 'load_backbone', lambda *_: object())
    monkeypatch.setattr(runner.study, 'state_sha256', lambda _: 'd'*64)
    monkeypatch.setattr(runner.study, 'load_checkpoint', lambda *args, **kwargs: object())
    monkeypatch.setattr(runner.torch.mps, 'empty_cache', lambda: None)

    def fit(config, backbone, graph, cache, out, **kwargs):
        events.append(('fit', runner.name(config)))
        out.mkdir()
        if fail_fit is not None and len(events) == fail_fit:
            raise RuntimeError('synthetic fixed-budget failure')
        runner.write(out/'training.json', {'fixture': True})
        (out/'weights.pt').write_bytes(b'synthetic-only')

    def evaluate(model, cache, out, **kwargs):
        events.append(('evaluate', runner.name(kwargs['configuration'])))
        out.write_text(''.join(json.dumps({'id': r['id'], 'choice': 'a2a3'})+'\n' for r in cache.rows))
        return {'metrics': {'fixture': True}}

    monkeypatch.setattr(runner.study, 'fit', fit)
    monkeypatch.setattr(runner.study, 'evaluate', evaluate)
    monkeypatch.setattr(runner.study, 'measure_latency', lambda *args, **kwargs: {'fixture': True})
    def score(plan, out, decisions, **kwargs):
        out.mkdir()
        receipt = {'cost': {'calls': 0, 'requested_nodes': 0}}
        runner.write(out/'completed.json', receipt)
        return receipt

    monkeypatch.setattr(runner.study, 'score_stronger', score)
    return plan_path, events


def test_all_eighteen_final_fits_precede_every_neural_evaluation(tmp_path, monkeypatch):
    plan, events = stubbed_run(tmp_path, monkeypatch)
    out = tmp_path/'runs/execution'
    runner.run(plan, out)
    assert [e[0] for e in events] == ['fit']*18+['evaluate']*42
    assert len({e[1] for e in events[:18]}) == 18
    assert len({e[1] for e in events[18:]}) == 21
    runner.completed(out, runner.sha(plan))
    runner.artifact_membership(out, runner.read(plan))
    with pytest.raises(FileExistsError):
        runner.run(plan, out)


@pytest.mark.parametrize('extra', ['fits/replacement-97', 'evaluation/replacement-97',
                                 'evaluation/direct-97/alternative.jsonl', 'extra.json'])
def test_rehashed_extra_attempt_is_rejected(tmp_path, monkeypatch, extra):
    plan, _ = stubbed_run(tmp_path, monkeypatch)
    out = tmp_path/'runs/execution'
    runner.run(plan, out)
    path = out/extra
    if path.suffix:
        path.write_text('{}')
    else:
        path.mkdir()
    receipt = runner.read(out/'completed.json')
    current = runner.tree(out)
    current.pop('completed.json')
    receipt['files'] = current
    (out/'completed.json').write_text(json.dumps(receipt))
    runner.completed(out, runner.sha(plan))
    with pytest.raises(ValueError, match='membership'):
        runner.artifact_membership(out, runner.read(plan))


def test_failed_fit_preserved_no_retry_or_evaluation(tmp_path, monkeypatch):
    plan, events = stubbed_run(tmp_path, monkeypatch, fail_fit=3)
    out = tmp_path/'runs/execution'
    with pytest.raises(RuntimeError, match='fixed-budget failure'):
        runner.run(plan, out)
    assert len(events) == 3 and all(e[0] == 'fit' for e in events)
    assert runner.read(out/'failed.json')['status'] == 'failed'
    assert not (out/'completed.json').exists()
    assert not (out/'all-fits-completed.json').exists()
