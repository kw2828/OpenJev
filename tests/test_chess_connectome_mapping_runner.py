"""Synthetic orchestration, identity and gate tests; no engines or trained fits."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_tested_connectome_mapping_runner',
                                             ROOT/'scripts/chess_connectome_mapping_study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def losses():
    values = {}
    for config in study.baselines()+study.configurations():
        if config['variant'] == 'direct':
            value = .1
        elif config['variant'] == 'biological':
            value = .08 if config['mapping_policy'] == 'learned' else .2
        else:
            value = .12 if config['mapping_policy'] == 'learned' else .15
        values[config['name']] = dict.fromkeys(('dev', 'shift'), value)
    return values


def test_full_configuration_and_proposal_schedule_matches_frozen_budget():
    configs = study.configurations()
    assert len(configs) == len({c['name'] for c in configs}) == 30
    assert configs == study.configurations()
    assert {(c['variant'], c['mapping_policy'], c['seed']) for c in configs} == {
        (v, p, s) for v in ('biological', 'rewire151', 'rewire163', 'rewire179', 'node_local')
        for p in ('fixed', 'learned') for s in (97, 109, 127)}
    assert len(study.baselines()) == 3
    settings = study.helper().TrainingSettings()
    assert settings.updates == 1536
    for seed in (97, 109, 127):
        proposals = study.helper().proposal_schedule(seed, settings)
        assert len(proposals) == 320
        assert [p['before_update'] for p in proposals] == list(range(257, 1537, 4))
    assert study.PROTOCOL['regret_call_ceiling'] == 34*128*2
    assert study.PROTOCOL['regret_node_ceiling'] == 34*128*2*20000
    assert study.PROTOCOL['primary_wall_seconds'] == 7200
    assert study.PROTOCOL['audit_wall_seconds'] == 1800
    assert 'Concurrent independent pin training is permitted' in study.PROTOCOL['timing']


def test_legacy_prediction_identity_never_replaces_unique_outer_identity():
    configs = {c['name']: c for c in study.configurations()}
    fixed, learned = configs['biological-fixed-97'], configs['biological-learned-97']
    assert fixed != learned
    assert study.legacy_config(fixed) == study.legacy_config(learned)
    assert study.legacy_config(fixed)['name'] == 'biological-97'
    direct = study.baselines()[0]
    assert study.legacy_config(direct) is direct


def test_gate_requires_all_controls_panels_and_seed_checks():
    result = study.continuation(losses())
    assert result['continuation_passed'] is True
    assert len(result['checks']) == 40
    assert len([c for c in result['checks'] if c['criterion'] == 'mapping_difference_in_differences']) == 8
    for split in ('dev', 'shift'):
        for comparator in ('biological-fixed', 'rewire151-learned', 'rewire163-learned', 'rewire179-learned'):
            values = losses()
            for seed in (97, 109, 127):
                values[f'{comparator}-{seed}'][split] = .085
            assert not study.continuation(values)['continuation_passed']


@pytest.mark.parametrize('comparator', ['rewire151', 'rewire163', 'rewire179', 'node_local'])
@pytest.mark.parametrize('split', ['dev', 'shift'])
def test_difference_in_differences_cannot_be_explained_by_other_interfaces(comparator, split):
    values = losses()
    for seed in (97, 109, 127):
        values[f'{comparator}-fixed-{seed}'][split] = .3
    result = study.continuation(values)
    matches = [c for c in result['checks'] if c['criterion'] == 'mapping_difference_in_differences'
               and c['comparator'] == comparator and c['split'] == split]
    assert len(matches) == 1 and not matches[0]['passed']
    assert not result['continuation_passed']


@pytest.mark.parametrize('split', ['dev', 'shift'])
@pytest.mark.parametrize('seed', [97, 109, 127])
def test_one_paired_seed_or_direct_degradation_blocks_gate(split, seed):
    values = losses()
    values[f'rewire151-learned-{seed}'][split] = .08
    assert not study.continuation(values)['continuation_passed']
    values = losses()
    values[f'direct-{seed}'][split] = .079
    assert not study.continuation(values)['continuation_passed']


@pytest.mark.parametrize('fault', ['missing', 'extra', 'missing_panel', 'extra_panel', 'nan', 'bool', 'bound'])
def test_gate_rejects_partial_or_invalid_values(fault):
    values = losses()
    first = next(iter(values))
    if fault == 'missing':
        values.pop(first)
    elif fault == 'extra':
        values['best-seed-only'] = {'dev': 0., 'shift': 0.}
    elif fault == 'missing_panel':
        values[first].pop('shift')
    elif fault == 'extra_panel':
        values[first]['test'] = .1
    else:
        values[first]['dev'] = {'nan': float('nan'), 'bool': True, 'bound': 2.1}[fault]
    with pytest.raises(ValueError):
        study.continuation(values)


def test_negative_signed_losses_retained_but_nonpositive_comparator_cannot_pass():
    values = losses()
    for seed in (97, 109, 127):
        values[f'biological-learned-{seed}']['dev'] = -.2
        values[f'biological-fixed-{seed}']['dev'] = -.1
    result = study.continuation(values)
    assert result['means']['biological-learned']['dev'] == pytest.approx(-.2)
    assert result['checks'][0]['reduction'] is None
    assert not result['continuation_passed']


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        study.parse(raw)


@pytest.mark.parametrize('path', [study.OLD_PLAN, f'{study.OLD_EXECUTION}/completed.json', study.OLD_SUMMARY])
def test_pinned_original_source_rejected_before_loading(path, monkeypatch):
    pins = {study.OLD_PLAN: study.OLD_PLAN_SHA, f'{study.OLD_EXECUTION}/completed.json': study.OLD_EXECUTION_SHA,
            study.OLD_SUMMARY: study.OLD_SUMMARY_SHA}
    monkeypatch.setattr(study, 'sha', lambda p: 'f'*64 if Path(p) == study.ROOT/path else pins[Path(p).relative_to(study.ROOT).as_posix()])
    monkeypatch.setattr(study, 'original', lambda: pytest.fail('Must reject before original verification'))
    with pytest.raises(ValueError, match='prerequisite hash'):
        study.prerequisites()


def frozen_stub(monkeypatch):
    signature = {'protocol': copy.deepcopy(study.PROTOCOL), 'sources': {'source.py': 'a'*64},
        'profile': {'sha256': 'b'*64}, 'configurations': study.configurations(), 'baselines': study.baselines(),
        'environment': {'mps_available': True, 'mps_fallback': '0'}, 'selection': {'secondary_indices': [1, 2]}}
    monkeypatch.setattr(study, 'signature', lambda: (copy.deepcopy(signature), {}))
    return signature


@pytest.mark.parametrize('fault', ['source', 'profile', 'protocol', 'configuration', 'selection', 'environment', 'extra'])
def test_frozen_identity_cannot_be_rehashed_to_change_inputs(tmp_path, monkeypatch, fault):
    frozen_stub(monkeypatch)
    directory = tmp_path/'protocol'
    study.prepare(directory)
    plan_path = directory/'plan.json'
    study.verify(plan_path)
    plan = study.read(plan_path)
    if fault == 'source':
        plan['sources']['source.py'] = 'c'*64
    elif fault == 'profile':
        plan['profile']['sha256'] = 'd'*64
    elif fault == 'protocol':
        plan['protocol']['primary_wall_seconds'] += 1
    elif fault == 'configuration':
        plan['configurations'].pop()
    elif fault == 'selection':
        plan['selection']['secondary_indices'].reverse()
    elif fault == 'environment':
        plan['environment']['mps_fallback'] = '1'
    else:
        plan['extra'] = True
    plan_path.write_text(json.dumps(plan))
    prepared = study.read(directory/'prepared.json')
    prepared['plan_sha256'] = study.sha(plan_path)
    (directory/'prepared.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='Frozen'):
        study.verify(plan_path)


def test_completed_manifest_rejects_extras_failed_and_symlink(tmp_path):
    directory = tmp_path/'done'
    directory.mkdir()
    (directory/'artifact.json').write_text('{}')
    study.write(directory/'completed.json', {'status': 'completed', 'plan_sha256': 'a'*64,
                                            'files': study.tree(directory)})
    study.completed(directory, 'a'*64)
    (directory/'extra.json').write_text('{}')
    with pytest.raises(ValueError):
        study.completed(directory, 'a'*64)
    (directory/'extra.json').unlink()
    (directory/'failed.json').write_text('{}')
    with pytest.raises(ValueError):
        study.completed(directory, 'a'*64)
    (directory/'failed.json').unlink()
    (directory/'extra.json').symlink_to(directory/'artifact.json')
    with pytest.raises(ValueError):
        study.tree(directory)


def test_rebuilt_cache_preserves_tensor_identity_with_new_measured_construction_costs():
    records = [{'id': 'synthetic-0', 'game_id': 'invented-game', 'fen': chess.STARTING_FEN,
                'target_uci': 'e2e4', 'target_value': 0.}]
    previous = study.CachedPositions(records, include_successors=False).metadata
    rebuilt = copy.deepcopy(previous)
    rebuilt['native_construction_wall_seconds'] += 1.
    rebuilt['fingerprinting_wall_seconds'] += 2.
    rebuilt['total_wall_seconds'] += 4.
    study.validate_cache_identity(rebuilt, previous, records)


@pytest.mark.parametrize('fault', ['tensor', 'row', 'unknown', 'unknown_both', 'missing',
                                 'negative_time', 'nan_time', 'bool_time', 'time_total'])
def test_cache_identity_rejects_schema_source_tensor_and_timing_corruption(fault):
    records = [{'id': 'synthetic-0', 'game_id': 'invented-game', 'fen': chess.STARTING_FEN,
                'target_uci': 'e2e4', 'target_value': 0.}]
    previous = study.CachedPositions(records, include_successors=False).metadata
    rebuilt = copy.deepcopy(previous)
    if fault == 'tensor':
        rebuilt['tensors']['observations']['sha256'] = 'f'*64
    elif fault == 'row':
        records[0]['id'] = 'different-source-row'
    elif fault in ('unknown', 'unknown_both'):
        rebuilt['unrecognized_timing'] = 0.
        if fault == 'unknown_both':
            previous['unrecognized_timing'] = 0.
    elif fault == 'missing':
        rebuilt.pop('construction_scope')
    elif fault == 'negative_time':
        rebuilt['fingerprinting_wall_seconds'] = -1.
    elif fault == 'nan_time':
        rebuilt['fingerprinting_wall_seconds'] = float('nan')
    elif fault == 'bool_time':
        rebuilt['fingerprinting_wall_seconds'] = True
    else:
        rebuilt['total_wall_seconds'] = 0.
    with pytest.raises(ValueError):
        study.validate_cache_identity(rebuilt, previous, records)


def grading_fixture():
    configs = study.baselines()+study.configurations()
    panels = {s: [{'id': f'{s}-{i}', 'fen': chess.STARTING_FEN} for i in range(128)] for s in ('dev', 'shift')}
    plan = {'panels': panels, 'secondary_indices': {s: list(range(128)) for s in panels},
            'configurations': [{'id': c['name']} for c in configs]}
    decisions = [{'configuration': c['name'], 'split': s, 'panel_index': i,
                  'id': row['id'], 'choice': 'e2e4'} for c in configs for s, panel in panels.items()
                 for i, row in enumerate(panel)]
    return plan, decisions


def test_generic_grading_retains_33_unique_names_and_caches_equal_moves():
    plan, decisions = grading_fixture()
    assert study.grading_inputs(plan, decisions) == 512
    for row in decisions:
        if row['configuration'].startswith('biological-learned-'):
            row['choice'] = 'd2d4'
    assert study.grading_inputs(plan, decisions) == 768


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'legacy_name', 'illegal_move', 'wrong_id', 'bad_index', 'extra_model'])
def test_grading_rejects_missing_aliasing_and_unbound_choices(fault):
    plan, decisions = grading_fixture()
    if fault == 'missing':
        decisions.pop()
    elif fault == 'duplicate':
        decisions[1] = decisions[0]
    elif fault == 'legacy_name':
        row = next(r for r in decisions if r['configuration'] == 'biological-learned-97')
        row['configuration'] = 'biological-97'
    elif fault == 'illegal_move':
        decisions[0]['choice'] = 'e2e5'
    elif fault == 'wrong_id':
        decisions[0]['id'] = 'wrong'
    elif fault == 'bad_index':
        decisions[0]['panel_index'] = True
    else:
        plan['configurations'].append({'id': 'best-seed'})
    with pytest.raises(ValueError):
        study.grading_inputs(plan, decisions)


def saved_generic_grading(tmp_path):
    """Exact generic score_secondary output schema, with invented finite scores."""
    plan, decisions = grading_fixture()
    engine = tmp_path/'never-executed-engine'
    engine.write_bytes(b'synthetic identity fixture, not an executable')
    plan['engine_path'] = str(engine)
    engine_hash, plan_hash = study.sha(engine), 'a'*64
    binding = study.grading_binding(plan, decisions, plan_hash, engine_hash)
    directory = tmp_path/'regret'
    directory.mkdir()
    study.write(directory/'started.json', {'status': 'started', **binding})
    study.write(directory/'engine.json', {'id': {'name': 'Stockfish 19 synthetic validation fixture'},
                                         'path': str(engine), 'sha256': engine_hash})
    analyses = []
    for split in ('dev', 'shift'):
        for index in range(128):
            for move, cp in ((None, 100), ('e2e4', 120)):
                analyses.append({'split': split, 'panel_index': index, 'id': f'{split}-{index}',
                    'root_move': move, 'pv_first': 'e2e4', 'score_cp': cp, 'mate': None,
                    'bounded_score': math.tanh(cp/600.), 'requested_nodes': 20000,
                    'reported_nodes': 20017, 'wall_seconds': .001})
    with (directory/'analyses.jsonl').open('x') as stream:
        stream.writelines(json.dumps(r)+'\n' for r in analyses)
    bounded = math.tanh(100/600.)-math.tanh(120/600.)
    with (directory/'regret.jsonl').open('x') as stream:
        stream.writelines(json.dumps({**r, 'bounded_regret': bounded, 'cp_loss': -20})+'\n' for r in decisions)
    cost = {'calls': len(analyses), **{k: sum(r[k] for r in analyses)
                                    for k in ('requested_nodes', 'reported_nodes', 'wall_seconds')}}
    study.write(directory/'completed.json', {'status': 'completed', **binding, 'cost': cost,
        'grading_wall_seconds': 1., 'files': study.tree(directory)})
    return plan, directory, decisions, plan_hash, engine_hash


def test_actual_generic_auditor_accepts_full_saved_schema_without_engine(tmp_path, monkeypatch):
    plan, directory, decisions, plan_hash, engine_hash = saved_generic_grading(tmp_path)
    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda *a, **k: pytest.fail('No engine calls permitted'))
    receipt = study.audit_grading(plan, directory, decisions, plan_hash=plan_hash, engine_hash=engine_hash)
    assert receipt['cost']['calls'] == 512
    records = study.rows(directory/'regret.jsonl')
    metrics = study.engine_metrics(records, study.baselines()+study.configurations())
    assert len(metrics) == 33
    for panels in metrics.values():
        for panel in panels.values():
            assert panel['positions'] == 128
            assert panel['mean_cp_loss'] == panel['p95_cp_loss'] == panel['max_cp_loss'] == -20
            assert panel['mean_signed_bounded_loss'] < 0


@pytest.mark.parametrize('fault', ['analysis_score', 'boolean_nodes', 'cost', 'missing_record', 'wrong_move',
                                 'decision_binding', 'source_binding', 'engine_identity'])
def test_actual_generic_auditor_rejects_rehashed_records_and_receipts(tmp_path, fault):
    plan, directory, decisions, plan_hash, engine_hash = saved_generic_grading(tmp_path)
    receipt_path = directory/'completed.json'
    receipt = study.read(receipt_path)
    if fault in ('analysis_score', 'boolean_nodes'):
        path = directory/'analyses.jsonl'
        records = study.rows(path)
        if fault == 'analysis_score':
            records[0]['score_cp'] += 1
        else:
            records[0]['reported_nodes'] = True
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    elif fault in ('missing_record', 'wrong_move'):
        path = directory/'regret.jsonl'
        records = study.rows(path)
        if fault == 'missing_record':
            records.pop()
        else:
            records[0]['choice'] = 'd2d4'
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    elif fault == 'cost':
        receipt['cost']['reported_nodes'] += 1
    elif fault == 'decision_binding':
        receipt['decisions_sha256'] = 'e'*64
    elif fault == 'source_binding':
        receipt['grader_source_sha256'] = 'f'*64
    elif fault == 'engine_identity':
        engine = study.read(directory/'engine.json')
        engine['id']['name'] = 'Different engine'
        (directory/'engine.json').write_text(json.dumps(engine))
    receipt['files'] = study.tree(directory)
    receipt['files'].pop('completed.json')
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        study.audit_grading(plan, directory, decisions, plan_hash=plan_hash, engine_hash=engine_hash)


def test_raw_centipawn_summary_does_not_hide_high_tail_or_negative_values():
    records = [{'configuration': 'fixture', 'split': split, 'cp_loss': i-100,
                'bounded_regret': (i-100)/1000} for split in ('dev', 'shift') for i in range(128)]
    metrics = study.engine_metrics(records, [{'name': 'fixture'}])
    assert metrics['fixture']['dev']['mean_cp_loss'] == -36.5
    assert metrics['fixture']['dev']['p95_cp_loss'] == pytest.approx(20.65)
    assert metrics['fixture']['dev']['max_cp_loss'] == 27
    assert metrics['fixture']['dev']['mean_signed_bounded_loss'] == pytest.approx(-.0365)


def lifecycle_fixture(tmp_path, monkeypatch, *, fail_fit=None):
    """Stub only heavy kernels, retaining real directories, receipts and orchestration."""
    actual_helper = study.helper()
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'deadline', lambda _: None)
    monkeypatch.setattr(study.signal, 'alarm', lambda _: None)
    monkeypatch.setattr(study.torch.mps, 'empty_cache', lambda: None)
    monkeypatch.setattr(study.gc, 'collect', lambda: None)
    configs = study.configurations()
    panels = {s: [{'id': f'{s}-{i}', 'fen': chess.STARTING_FEN, 'target_uci': 'e2e4', 'target_value': 0.}
                  for i in range(128)] for s in ('dev', 'shift')}
    training = [{'id': 'training-fixture'}]

    class Cache:
        def __init__(self, records, **kwargs):
            self.rows = records
            self.metadata = {k: None for k in study.CACHE_FIELDS}
            self.metadata.update({k: 0. for k in study.CACHE_TIMINGS})
            self.metadata['input_rows_sha256'] = study.digest([r['id'] for r in records])

    old_execution = tmp_path/study.OLD_EXECUTION
    (old_execution/'data').mkdir(parents=True)
    study.write(old_execution/'training-cache.json', Cache(training).metadata)
    study.write(old_execution/'evaluation-cache.json', {s: Cache(r).metadata for s, r in panels.items()})
    study.write(old_execution/'data/completed.json', {'status': 'synthetic'})
    engine = tmp_path/'runs/engine'
    engine.write_bytes(b'synthetic engine identity, never executed')
    plan = {'configurations': configs, 'baselines': study.baselines(), 'original_inputs': {'synthetic': True},
        'selection': {'secondary_indices': {s: list(range(128)) for s in panels},
                      'latency_indices': [0, 128]},
        'binding': {'graph_hashes': {}, 'training_source_sha256': 'a'*64, 'engine': 'runs/engine',
                    'engine_sha256': study.sha(engine), 'backbone_pretraining_costs': {}}}
    plan_path = tmp_path/'plan.json'
    study.write(plan_path, plan)
    events = []
    state = lambda name: study.digest({'synthetic_model': name})
    monkeypatch.setattr(study, 'verify', lambda _: (copy.deepcopy(plan), panels))
    monkeypatch.setattr(study, 'CachedPositions', Cache)
    monkeypatch.setattr(study, 'audit_cache_metadata', lambda metadata, records, **kwargs:
                        metadata == Cache(records).metadata or pytest.fail('Cache changed'))
    monkeypatch.setattr(study.data, 'training_rows', lambda _: training)
    monkeypatch.setattr(study, 'original', lambda: SimpleNamespace(
        graphs=lambda _: {v: None for v in study.PROTOCOL['variants']},
        load_backbone=lambda p, s: SimpleNamespace(name=f'direct-{s}')))
    monkeypatch.setattr(study.evaluation, 'state_sha256', lambda m: state(m.name))

    def fit(config, backbone, graph, cache, out, **kwargs):
        if len(events) == fail_fit:
            raise study.DeadlineExceeded('synthetic budget stop')
        events.append(('fit', config['name']))
        out.mkdir()
        for name in ('started.json', 'learning.jsonl', 'proposals.jsonl'):
            (out/name).write_text('{}\n')
        (out/'weights.pt').write_bytes(config['name'].encode())
        receipt = {'status': 'completed', 'configuration': config, 'plan_sha256': kwargs['plan_sha256'],
                   'state_sha256': state(config['name'])}
        study.write(out/'training.json', receipt)
        return receipt

    def load(path, backbone, graph, **kwargs):
        assert study.sha(path) == kwargs['expected_checkpoint_sha256']
        return SimpleNamespace(name=kwargs['expected_config']['name'])

    def audit_fit(directory, *args, **kwargs):
        events.append(('audit_fit', directory.name))
        return study.read(directory/'training.json')

    monkeypatch.setattr(study, 'helper', lambda: SimpleNamespace(
        configurations=actual_helper.configurations, legacy_config=actual_helper.legacy_config,
        TrainingSettings=actual_helper.TrainingSettings, fit=fit, load_checkpoint=load, audit_training=audit_fit))

    def evaluate(model, cache, path, **kwargs):
        assert len([e for e in events if e[0] == 'fit']) == 30
        assert kwargs['configuration']['name'] == ('-'.join(model.name.split('-')[:-2]+model.name.split('-')[-1:])
                                                   if '-fixed-' in model.name or '-learned-' in model.name else model.name)
        events.append(('evaluate', model.name))
        with path.open('x') as stream:
            for row in cache.rows:
                stream.write(json.dumps({'id': row['id'], 'choice': 'e2e4'})+'\n')
        return {'status': 'completed', 'metrics': {'examples': len(cache.rows)}}

    def audit_eval(receipt, path, records, **kwargs):
        events.append(('audit_evaluate', kwargs['model'].name))
        return receipt['metrics'], study.rows(path)

    monkeypatch.setattr(study.evaluation, 'evaluate', evaluate)
    monkeypatch.setattr(study.evaluation, 'audit_evaluation', audit_eval)
    monkeypatch.setattr(study.evaluation, 'measure_latency', lambda *args, **kwargs:
                        {'total_wall_ms': 2., 'warmup_wall_ms': 3., 'records': [1, 2]})
    monkeypatch.setattr(study.evaluation, 'audit_latency', lambda *args, **kwargs: None)
    cost = {'calls': 512, 'requested_nodes': 10240000, 'reported_nodes': 10240000, 'wall_seconds': 1.}

    def score(plan, directory, decisions, **kwargs):
        assert len([e for e in events if e[0] == 'evaluate']) == 66
        assert len(plan['configurations']) == 33
        directory.mkdir()
        with (directory/'regret.jsonl').open('x') as stream:
            for row in decisions:
                stream.write(json.dumps({**row, 'bounded_regret': .1, 'cp_loss': 60})+'\n')
        return {'cost': cost}

    monkeypatch.setattr(study, 'score_grading', score)
    monkeypatch.setattr(study, 'audit_grading', lambda *args, **kwargs: {'cost': cost})
    return plan_path, tmp_path/'runs/execution', tmp_path/'audit', events


def test_run_then_saved_output_audit_preserves_all30_fits_and33_evaluations(tmp_path, monkeypatch):
    plan_path, execution, audit, events = lifecycle_fixture(tmp_path, monkeypatch)
    assert study.run(plan_path, execution)['status'] == 'completed'
    assert [k for k, _ in events[:30]] == ['fit']*30
    assert len([e for e in events if e[0] == 'evaluate']) == 66
    original_eval_count = len(events)
    assert study.audit(plan_path, execution, audit)['status'] == 'completed'
    assert all(e[0] in ('audit_fit', 'audit_evaluate') for e in events[original_eval_count:])
    summary = study.read(audit/'summary.json')
    assert len(summary['training_costs']) == 30
    assert len(summary['engine_loss']) == len(summary['metrics']) == len(summary['latency']) == 33
    assert summary['mapping_comparison']['continuation_passed'] is False
    assert summary['report_new_model_or_engine_calls'] == 0
    assert study.read(audit/'receipt.json')['new_model_calls'] == 0
    with pytest.raises(FileExistsError):
        study.run(plan_path, execution)
    with pytest.raises(FileExistsError):
        study.audit(plan_path, execution, audit)


def test_budget_interrupt_keeps_failure_and_never_evaluates_partial_fits(tmp_path, monkeypatch):
    plan_path, execution, _, events = lifecycle_fixture(tmp_path, monkeypatch, fail_fit=2)
    with pytest.raises(study.DeadlineExceeded):
        study.run(plan_path, execution)
    assert len(events) == 2 and all(k == 'fit' for k, _ in events)
    assert study.read(execution/'failed.json')['error_type'] == 'DeadlineExceeded'
    assert not (execution/'all-fits-completed.json').exists()
    assert not (execution/'completed.json').exists()


@pytest.mark.parametrize('fault', ['extra_fit', 'extra_weights', 'outer_alias', 'state', 'fit_boundary', 'early_eval', 'over_budget'])
def test_audit_rejects_rehashed_coverage_identity_and_phase_corruption(tmp_path, monkeypatch, fault):
    plan_path, execution, audit, _ = lifecycle_fixture(tmp_path, monkeypatch)
    study.run(plan_path, execution)
    plan = study.read(plan_path)
    if fault == 'extra_fit':
        (execution/'fits/selected-best').mkdir()
    elif fault == 'extra_weights':
        (execution/'fits'/plan['configurations'][0]['name']/'extra.pt').write_bytes(b'extra')
    elif fault in ('outer_alias', 'state'):
        path = execution/'evaluation'/plan['configurations'][0]['name']/'completed.json'
        receipt = study.read(path)
        if fault == 'outer_alias':
            receipt['configuration'] = study.legacy_config(plan['configurations'][0])
        else:
            receipt['state_sha256'] = 'f'*64
        path.write_text(json.dumps(receipt))
    elif fault == 'fit_boundary':
        path = execution/'all-fits-completed.json'
        boundary = study.read(path)
        boundary['fits'].pop()
        path.write_text(json.dumps(boundary))
    elif fault == 'early_eval':
        path = execution/'evaluation-started.json'
        receipt = study.read(path)
        receipt['started_unix'] = 0.
        path.write_text(json.dumps(receipt))
    receipt = study.read(execution/'completed.json')
    receipt['files'] = study.tree(execution)
    receipt['files'].pop('completed.json')
    if fault == 'over_budget':
        receipt['wall_seconds'] = 7201.
    (execution/'completed.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        study.audit(plan_path, execution, audit)
    assert (audit/'failed.json').exists()
    assert not (audit/'receipt.json').exists()


def test_deadline_escapes_ordinary_exception_wrappers(monkeypatch):
    handler = {}
    monkeypatch.setattr(study.signal, 'signal', lambda sig, callback: handler.update(callback=callback))
    monkeypatch.setattr(study.signal, 'alarm', lambda _: None)
    study.deadline(1)
    with pytest.raises(study.DeadlineExceeded):
        try:
            handler['callback'](None, None)
        except Exception:  # noqa: BLE001 - deliberately simulate inherited broad policy handlers
            pytest.fail('A policy or engine Exception wrapper swallowed the deadline')


def test_supervisor_watchdog_stops_only_its_child_and_never_starts_audit(tmp_path, monkeypatch):
    launcher = tmp_path/'launcher'
    launcher.mkdir()
    plan = tmp_path/'plan.json'
    plan.write_text('{}')
    study.write(launcher/'request.json', {'plan': str(plan), 'plan_sha256': study.sha(plan),
        'execution': str(tmp_path/'execution'), 'audit': str(tmp_path/'audit')})
    waits, killed, launched = [], [], []

    class Child:
        pid = 991199

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise study.subprocess.TimeoutExpired('synthetic', timeout)
            return 0

    def popen(args, **kwargs):
        launched.append(args)
        assert kwargs['start_new_session'] is True
        return Child()

    monkeypatch.setattr(study.subprocess, 'Popen', popen)
    monkeypatch.setattr(study.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(TimeoutError):
        study.supervise(launcher)
    assert len(launched) == 1
    assert waits == [7210, 10]
    assert killed == [(991199, study.signal.SIGTERM)]
    assert study.read(launcher/'failed.json')['phase'] == 'run'


def test_private_graph_outputs_cannot_be_published(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='local runs'):
        study.private_output(tmp_path/'evidence/weights')
    assert study.private_output(tmp_path/'runs/fits') == tmp_path/'runs/fits'
