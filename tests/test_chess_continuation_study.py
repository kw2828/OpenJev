"""Synthetic study orchestration and evidence corruption tests, without inference."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    '_tested_continuation_study', Path(__file__).resolve().parents[1]/'scripts/chess_continuation_study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_predefined_nine_fit_schedule_and_primary_checkpoint():
    configs = study.configurations()
    assert {(r['objective'], r['seed']) for r in configs} == {
        (o, s) for o in ('policy', 'best_value', 'continuation') for s in (193, 211, 227)}
    assert len(configs) == len({r['name'] for r in configs}) == 9
    assert all(r['name'] == f"{r['objective']}-{r['seed']}" for r in configs)
    assert configs == study.configurations()
    p = study.PROTOCOL
    assert p['objectives'] == list(study.OBJECTIVES) and p['seeds'] == list(study.SEEDS)
    assert p['primary_checkpoint_epoch'] == p['epochs'] == 8
    assert p['diagnostic_epochs'] == list(range(1, 9))
    assert p['width'] == 32 and p['depth'] == 4 and p['batch_size'] == 128
    assert p['root_value_weight'] == .5 and p['candidate_value_weight'] == 1.
    assert p['relative_loss_reduction'] == .2 and p['arena_lower_points_threshold'] == .6
    assert p['regret_positions_per_split'] == 128 and p['regret_nodes'] == 20000
    assert p['primary_wall_seconds'] == 14400 and p['audit_wall_seconds'] == 2700


def test_real_count_schedule_has_paired_complete_epochs_and_tail():
    count = 88408
    reference = list(study.batches(count, study.SEEDS[0]))
    assert len(reference) == math.ceil(count/128) * 8 == 5528
    for epoch in range(1, 9):
        selected = [r for r in reference if r['epoch'] == epoch]
        assert [len(r['indices']) for r in selected] == [128] * 690 + [88]
        indices = [i for r in selected for i in r['indices']]
        assert sorted(indices) == list(range(count))
    assert [r['step'] for r in reference] == list(range(1, 5529))
    # Each objective receives exactly the same seed's order, including its tail.
    assert reference == list(study.batches(count, study.SEEDS[0]))
    assert reference[0]['indices'] != next(study.batches(count, study.SEEDS[1]))['indices']


@pytest.mark.parametrize('count', [0, -1, True, 3.0])
def test_schedule_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        list(study.batches(count, 193))


def regret_rows(treatment=.08, policy=.1, best_value=.125):
    values = {'continuation': treatment, 'policy': policy, 'best_value': best_value}
    return [{'configuration': f'{objective}-{seed}', 'split': split, 'panel_index': index,
             'id': f'{split}-{index:05d}', 'choice': 'e2e4', 'bounded_regret': values[objective], 'cp_loss': 10}
            for objective in study.OBJECTIVES for seed in study.SEEDS for split in ('dev', 'shift')
            for index in range(128)]


def arena_summary(*, wins=60, unfinished=0, failed=0, weak_control=None):
    games, seen = [], {'policy': 0, 'best_value': 0}
    for spec in study.arena.schedule():
        index = seen[spec['opponent']]
        seen[spec['opponent']] += 1
        count = wins if spec['opponent'] != weak_control else 40
        status = ('completed' if index < 96 - unfinished - failed else
                  'unfinished' if index < 96 - failed else 'failed')
        treatment_wins = index < count
        result = ('1-0' if (spec['white_objective'] == 'continuation') == treatment_wins else '0-1')
        game = {**spec, 'game_id': spec['id'], 'status': status,
                'result': result if status == 'completed' else '*',
                'termination': 'synthetic', 'mate_diagnostic': {'by_player': {}},
                'attempts': [], 'played_plies': 0, 'wall_seconds': 0.}
        games.append(game)
    assert seen == {'policy': 96, 'best_value': 96}
    return study.arena.summarize(games)


def test_gate_requires_both_controls_on_both_panels_and_positive_reference():
    arena = arena_summary()
    assert study.outcome_gate(regret_rows(), arena)['passed']
    for control in ('policy', 'best_value'):
        for split in ('dev', 'shift'):
            records = regret_rows()
            for row in records:
                if row['configuration'].startswith(control + '-') and row['split'] == split:
                    row['bounded_regret'] = .09
            assert not study.outcome_gate(records, arena)['passed']
    assert not study.outcome_gate(regret_rows(treatment=-.1, policy=0.), arena)['passed']
    assert not study.outcome_gate(regret_rows(treatment=-.1, best_value=-.01), arena)['passed']


def test_gate_uses_all_game_lower_bound_and_each_control():
    records = regret_rows()
    # Every completed game is a win, but 56 unresolved games per control make
    # completed-only win percentage unsuitable for the predefined 60% gate.
    summary = arena_summary(wins=40, unfinished=56)
    assert summary['by_opponent']['policy']['continuation']['score_lower_bound'] == 40/96
    assert not study.outcome_gate(records, summary)['passed']
    assert not study.outcome_gate(records, arena_summary(weak_control='best_value'))['passed']
    assert not study.outcome_gate(records, arena_summary(failed=1))['passed']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'extra', 'coverage', 'nan', 'infinity', 'bool', 'bad_index'])
def test_gate_rejects_missing_duplicate_nonfinite_and_misaligned_evidence(fault):
    records = regret_rows()
    if fault == 'missing':
        records.pop()
    elif fault == 'duplicate':
        records[1] = copy.deepcopy(records[0])
    elif fault == 'extra':
        records.append({**records[0], 'configuration': 'unknown'})
    elif fault == 'coverage':
        records[0]['panel_index'] = 1234
    elif fault == 'nan':
        records[0]['bounded_regret'] = float('nan')
    elif fault == 'infinity':
        records[0]['bounded_regret'] = float('inf')
    elif fault == 'bool':
        records[0]['bounded_regret'] = True
    else:
        records[0]['panel_index'] = False
    with pytest.raises(ValueError):
        study.outcome_gate(records, arena_summary())


def test_gate_cannot_be_overridden_by_tampered_arena_pass_boolean():
    summary = arena_summary(wins=40, unfinished=56)
    summary['continuation_gate']['passed'] = True
    summary['gate_passed'] = True
    try:
        result = study.outcome_gate(regret_rows(), summary)
    except ValueError:
        return
    assert result['passed'] is False


@pytest.mark.parametrize('fault', ['missing_control', 'nan_lower_bound', 'wrong_game_count'])
def test_gate_rejects_malformed_arena_accounting(fault):
    summary = arena_summary()
    if fault == 'missing_control':
        del summary['by_opponent']['best_value']
    elif fault == 'nan_lower_bound':
        summary['by_opponent']['policy']['continuation']['score_lower_bound'] = float('nan')
    else:
        summary['by_opponent']['policy']['games'] -= 1
    with pytest.raises(ValueError):
        study.outcome_gate(regret_rows(), summary)


def training_fixture(monkeypatch, objective='continuation'):
    monkeypatch.setitem(study.PROTOCOL, 'epochs', 2)
    monkeypatch.setitem(study.PROTOCOL, 'batch_size', 3)
    config = {'name': f'{objective}-193', 'objective': objective, 'seed': 193}
    training = [{'continuation_mask': i in (0, 1, 3, 4, 6),
                 'behavior_uci': 'e2e4' if i in (1, 4, 5) else 'd2d4', 'target_uci': 'e2e4'} for i in range(7)]
    plan = {'updates_per_fit': 6, 'examples_seen_per_fit': 14, 'initial_state_sha256': {'193': 'a'*64}}
    records = []
    for step in study.batches(7, 193):
        teacher = 0 if objective == 'policy' else len(step['indices'])
        continuation = (sum(training[i]['continuation_mask'] and training[i]['behavior_uci'] != training[i]['target_uci']
                            for i in step['indices']) if objective == 'continuation' else 0)
        candidate = 0. if objective == 'policy' else .3
        records.append({'step': step['step'], 'epoch': step['epoch'], 'batch_size': len(step['indices']),
                        'batch_indices_sha256': study.digest(step['indices']), 'policy_ce': .7, 'root_value_mse': .2,
                        'candidate_value_mse': candidate, 'total': .8 + candidate, 'preclip_gradient_norm': .25,
                        'elapsed_seconds': step['step'] * .1, 'teacher_targets_used': teacher,
                        'continuation_targets_used': continuation, 'supervised_targets_used': teacher + continuation})
    receipt = {'status': 'completed', **config, 'updates': 6, 'examples_seen': 14,
               'initial_state_sha256': 'a'*64, 'training_seconds': 1.}
    return config, records, receipt, training, plan


@pytest.mark.parametrize('objective', ['policy', 'best_value', 'continuation'])
def test_training_accounting_covers_tail_examples_and_distinct_actions(monkeypatch, objective):
    args = training_fixture(monkeypatch, objective)
    study.validate_training(*args)
    assert sum(r['batch_size'] for r in args[1]) == 14
    assert sum(r['teacher_targets_used'] for r in args[1]) == (0 if objective == 'policy' else 14)
    assert sum(r['continuation_targets_used'] for r in args[1]) == (6 if objective == 'continuation' else 0)


@pytest.mark.parametrize('fault', ['missing_update', 'batch_hash', 'batch_size', 'step', 'epoch', 'total',
                                 'supervised_count', 'teacher_count', 'continuation_count', 'example_count',
                                 'initialization', 'nan_metric', 'negative_metric', 'time_reversal',
                                 'inf_duration', 'nan_duration', 'bool_step', 'bool_count'])
def test_training_accounting_rejects_corrupted_evidence(monkeypatch, fault):
    config, records, receipt, training, plan = training_fixture(monkeypatch)
    if fault == 'missing_update':
        records.pop()
    elif fault == 'batch_hash':
        records[0]['batch_indices_sha256'] = 'f'*64
    elif fault in ('batch_size', 'step', 'epoch'):
        records[0][fault] += 1
    elif fault == 'total':
        records[0]['total'] += .1
    elif fault in ('supervised_count', 'teacher_count', 'continuation_count'):
        records[0][fault.replace('_count', '_targets_used')] += 1
    elif fault == 'example_count':
        receipt['examples_seen'] -= 1
    elif fault == 'initialization':
        receipt['initial_state_sha256'] = 'b'*64
    elif fault == 'nan_metric':
        records[0]['policy_ce'] = float('nan')
    elif fault == 'negative_metric':
        records[0]['preclip_gradient_norm'] = -.1
    elif fault == 'time_reversal':
        records[1]['elapsed_seconds'] = 0.
    elif fault == 'inf_duration':
        receipt['training_seconds'] = float('inf')
    elif fault == 'nan_duration':
        receipt['training_seconds'] = float('nan')
    elif fault == 'bool_step':
        records[0]['step'] = True
    else:
        tail = next(r for r in records if r['teacher_targets_used'] == 1)
        tail['teacher_targets_used'] = True
    with pytest.raises(ValueError):
        study.validate_training(config, records, receipt, training, plan)


def freeze_stub(monkeypatch):
    dataset = SimpleNamespace(membership={'partitions': {'train': {'roots': 7}}})
    signature = {'protocol': copy.deepcopy(study.PROTOCOL), 'sources': {'source.py': 'a'*64},
                 'dataset': {'membership': {'partitions': {'train': {'roots': 7, 'indices': [0, 1]}}}},
                 'configurations': study.configurations(), 'updates_per_fit': 8,
                 'environment': {'mps_available': True, 'mps_fallback': '0'}}
    monkeypatch.setattr(study.data, 'load_dataset', lambda _: dataset)
    monkeypatch.setattr(study, 'signature', lambda _: copy.deepcopy(signature))
    return dataset, signature


@pytest.mark.parametrize('fault', ['protocol', 'source', 'membership', 'schedule', 'extra'])
def test_freeze_rejects_any_changed_scientific_or_source_binding(tmp_path, monkeypatch, fault):
    freeze_stub(monkeypatch)
    study.prepare(tmp_path/'protocol')
    path = tmp_path/'protocol/plan.json'
    study.verify(path)
    plan = study.read(path)
    if fault == 'protocol':
        plan['protocol']['primary_wall_seconds'] += 1
    elif fault == 'source':
        plan['sources']['source.py'] = 'b'*64
    elif fault == 'membership':
        plan['dataset']['membership']['partitions']['train']['indices'].reverse()
    elif fault == 'schedule':
        plan['configurations'].pop()
    else:
        plan['extra'] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Frozen'):
        study.verify(path)


def test_prepare_and_writes_are_exclusive(tmp_path, monkeypatch):
    freeze_stub(monkeypatch)
    study.prepare(tmp_path/'protocol')
    before = (tmp_path/'protocol/plan.json').read_bytes()
    with pytest.raises(FileExistsError):
        study.prepare(tmp_path/'protocol')
    with pytest.raises(FileExistsError):
        study.write(tmp_path/'protocol/plan.json', {'changed': True})
    assert (tmp_path/'protocol/plan.json').read_bytes() == before


@pytest.mark.parametrize('method', ['read', 'rows'])
@pytest.mark.parametrize('content', ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_json_parsers_reject_nonfinite_and_duplicate_values(tmp_path, method, content):
    path = tmp_path/'input.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        getattr(study, method)(path)


def run_fixture(tmp_path, monkeypatch, *, fail_fit=None, fail_grader=False):
    events = []
    configs = study.configurations()
    dataset = SimpleNamespace(rows=[{'id': 'root-0'}, {'id': 'root-1'}], membership={
        'partitions': {'train': {'indices': [0, 1]}},
        'fixed_diagnostic_subsets': {'train': {'indices': [0]}, 'diagnostic': {'indices': [1]}}})
    plan = {'configurations': configs, 'panels': {'secondary_indices': {'dev': [0], 'shift': [0]},
            'latency_indices': [0, 1]}, 'arena': {'synthetic': True}}
    plan_path = tmp_path/'plan.json'
    study.write(plan_path, plan)
    out = tmp_path/'execution'
    monkeypatch.setattr(study, 'verify', lambda _: (copy.deepcopy(plan), dataset))
    monkeypatch.setattr(study, 'deadline', lambda _: None)
    monkeypatch.setattr(study.signal, 'alarm', lambda _: None)
    monkeypatch.setattr(study.torch, 'set_num_threads', lambda _: None)
    monkeypatch.setattr(study.torch, 'use_deterministic_algorithms', lambda _: None)
    monkeypatch.setattr(study.data, 'tensorize', lambda records: ({'targets': torch.zeros(len(records), dtype=torch.long)}, []))
    monkeypatch.setattr(study, 'tensorize', lambda records: ({}, []))
    def fit(config, tensors, directory, given_plan, plan_hash):
        assert not any(event[0] == 'evaluate' for event in events)
        assert len(tensors['targets']) == 2
        directory.mkdir()
        if len(events) == fail_fit:
            raise RuntimeError('synthetic fit failure')
        for epoch in range(1, 9):
            (directory/f'epoch-{epoch:02d}.pt').write_text('synthetic checkpoint')
        events.append(('fit', config['name']))
        return {'status': 'completed', **config}
    monkeypatch.setattr(study, 'fit', fit)
    panels = {s: [{'id': f'{s}-0'}] for s in ('dev', 'shift')}
    def panel_inputs():
        assert sum(e[0] == 'fit' for e in events) == 9
        return panels, {}, [*panels['dev'], *panels['shift']]
    monkeypatch.setattr(study, 'panel_inputs', panel_inputs)
    def load_model(config, path, _):
        assert path.is_file()
        return SimpleNamespace(name=config['name'], epoch=int(path.stem.split('-')[-1]))
    monkeypatch.setattr(study, 'load_model', load_model)
    def evaluate(model, records, tensors, menus, depth, path):
        assert sum(e[0] == 'fit' for e in events) == 9
        assert study.read(out/'training-completed.json')['evaluation_started'] is False
        assert study.read(out/'evaluation-started.json')['completed_fits'] == 9
        phase = 'primary' if 'predictions' in path.parts else 'diagnostic'
        if phase == 'primary':
            assert model.epoch == 8
        events.append(('evaluate', phase, model.name, model.epoch))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps({'id': r['id'], 'choice': 'e2e4'})+'\n' for r in records))
        # Earlier epochs look better; they must never replace the primary.
        return {'metrics': {'agreement': 1./model.epoch}, 'evaluation_wall_seconds': .001}
    monkeypatch.setattr(study, 'evaluate', evaluate)
    def probe(model, records, depth, indices):
        assert model.epoch == 8
        events.append(('latency', model.name))
        return {'synthetic': True}
    monkeypatch.setattr(study, 'probe', probe)
    def score_secondary(given, directory, decisions):
        assert len(decisions) == 18
        events.append(('grade',))
        if fail_grader:
            raise RuntimeError('synthetic grader failure')
        return {'calls': 0, 'requested_nodes': 0}
    monkeypatch.setattr(study, 'grader', lambda: SimpleNamespace(score_secondary=score_secondary))
    def arena_run(models, directory, plan_hash, arena_plan):
        assert len(models) == 9 and all(m.epoch == 8 for m in models.values())
        events.append(('arena',))
        directory.mkdir()
        study.write(directory/'completed.json', {'synthetic': True})
    monkeypatch.setattr(study.arena, 'run', arena_run)
    return plan_path, out, events


def test_all_nine_fits_finish_before_any_evaluation_and_final_epoch_only(tmp_path, monkeypatch):
    plan, out, events = run_fixture(tmp_path, monkeypatch)
    assert study.run(plan, out)['status'] == 'completed'
    assert all(e[0] == 'fit' for e in events[:9])
    assert sum(e[:2] == ('evaluate', 'diagnostic') for e in events) == 9 * 8 * 2
    assert sum(e[:2] == ('evaluate', 'primary') for e in events) == 18
    assert all(e[3] == 8 for e in events if e[:2] == ('evaluate', 'primary'))
    receipt = study.read(out/'completed.json')
    actual = study.files(out)
    actual.pop('completed.json')
    assert receipt['files'] == actual and not (out/'failed.json').exists()
    with pytest.raises(FileExistsError):
        study.run(plan, out)


def test_partial_fit_failure_preserves_failure_without_any_evaluation(tmp_path, monkeypatch):
    plan, out, events = run_fixture(tmp_path, monkeypatch, fail_fit=3)
    with pytest.raises(RuntimeError, match='fit failure'):
        study.run(plan, out)
    assert len(events) == 3 and all(e[0] == 'fit' for e in events)
    assert study.read(out/'failed.json')['error_type'] == 'RuntimeError'
    assert not (out/'training-completed.json').exists() and not (out/'evaluation-started.json').exists()
    assert not (out/'completed.json').exists()


def test_posttraining_grader_failure_keeps_all_recorded_cost_and_predictions(tmp_path, monkeypatch):
    plan, out, events = run_fixture(tmp_path, monkeypatch, fail_grader=True)
    with pytest.raises(RuntimeError, match='grader failure'):
        study.run(plan, out)
    assert sum(e[0] == 'fit' for e in events) == 9
    assert (out/'training-completed.json').exists() and (out/'latency.json').exists()
    assert len(study.read(out/'predictions.json')) == 18
    assert not any(e[0] == 'arena' for e in events)
    assert (out/'failed.json').exists() and not (out/'completed.json').exists()


def test_launch_spawn_failure_has_durable_failure_accounting(tmp_path, monkeypatch):
    plan = tmp_path/'plan.json'
    study.write(plan, {'synthetic': True})
    def failed(*args, **kwargs):
        raise OSError('synthetic spawn failure')
    monkeypatch.setattr(study.subprocess, 'Popen', failed)
    launcher = tmp_path/'launcher'
    with pytest.raises(OSError, match='spawn failure'):
        study.launch(plan, tmp_path/'execution', tmp_path/'audit', launcher)
    assert (launcher/'failed.json').exists()
    assert not (launcher/'launched.json').exists()


def test_supervisor_nonzero_run_stops_before_audit(tmp_path, monkeypatch):
    launcher = tmp_path/'launcher'
    launcher.mkdir()
    plan = tmp_path/'plan.json'
    study.write(plan, {'synthetic': True})
    study.write(launcher/'request.json', {'plan': str(plan), 'plan_sha256': study.sha(plan),
                'execution': str(tmp_path/'execution'), 'audit': str(tmp_path/'audit')})
    calls = []
    def spawn(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(pid=12345, wait=lambda **kw: 17)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    result = study.supervise(launcher)
    assert result['status'] == 'failed' and result['phase'] == 'run'
    assert len(calls) == 1 and 'run' in calls[0] and not (launcher/'audit.log').exists()
    assert study.read(launcher/'run-exit.json')['returncode'] == 17
    assert (launcher/'failed.json').exists() and not (launcher/'completed.json').exists()


def supervisor_fixture(tmp_path):
    launcher = tmp_path/'launcher'
    launcher.mkdir()
    plan = tmp_path/'plan.json'
    study.write(plan, {'synthetic': True})
    request = {'plan': str(plan), 'plan_sha256': study.sha(plan),
               'execution': str(tmp_path/'execution'), 'audit': str(tmp_path/'audit')}
    study.write(launcher/'request.json', request)
    return launcher, request


def test_supervisor_rejects_zero_exit_without_completion_receipt(tmp_path, monkeypatch):
    launcher, _ = supervisor_fixture(tmp_path)
    calls = []
    def spawn(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(pid=12345, wait=lambda **kw: 0)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.supervise(launcher)
    assert len(calls) == 1 and study.read(launcher/'failed.json')['phase'] == 'run'
    assert not (launcher/'completed.json').exists() and not (launcher/'audit.log').exists()


def test_supervisor_validates_both_terminal_receipts_before_completion(tmp_path, monkeypatch):
    launcher, request = supervisor_fixture(tmp_path)
    calls = []
    def spawn(args, **kwargs):
        calls.append(args)
        assert kwargs['start_new_session'] is True
        command = args[2]
        path = Path(request['execution' if command == 'run' else 'audit'])
        name = 'completed.json' if command == 'run' else 'receipt.json'
        duration = 'total_wall_seconds' if command == 'run' else 'audit_wall_seconds'
        study.write(path/name, {'status': 'completed', 'plan_sha256': request['plan_sha256'], duration: .1})
        return SimpleNamespace(pid=12345 + len(calls), wait=lambda **kw: 0)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    assert study.supervise(launcher)['status'] == 'completed'
    assert [args[2] for args in calls] == ['run', 'audit']
    assert (launcher/'completed.json').exists() and not (launcher/'failed.json').exists()


@pytest.mark.parametrize('needs_kill', [False, True])
def test_supervisor_watchdog_targets_only_owned_group_and_never_retries(tmp_path, monkeypatch, needs_kill):
    launcher, _ = supervisor_fixture(tmp_path)
    waits, signals, spawns = [], [], []
    def wait(timeout):
        waits.append(timeout)
        if len(waits) <= (2 if needs_kill else 1):
            raise study.subprocess.TimeoutExpired('synthetic', timeout)
        return -15
    def spawn(args, **kwargs):
        spawns.append(args)
        return SimpleNamespace(pid=12345, wait=wait)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    monkeypatch.setattr(study.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(TimeoutError, match='watchdog'):
        study.supervise(launcher)
    assert len(spawns) == 1
    assert waits[0] == study.PROTOCOL['primary_wall_seconds'] + 10
    assert signals[0] == (12345, study.signal.SIGTERM)
    assert signals == ([(12345, study.signal.SIGTERM), (12345, study.signal.SIGKILL)] if needs_kill
                       else [(12345, study.signal.SIGTERM)])
    assert study.read(launcher/'failed.json')['error_type'] == 'TimeoutError'
    assert not (launcher/'audit.log').exists()


def test_existing_destination_is_rejected_before_any_launcher_or_spawn(tmp_path, monkeypatch):
    plan = tmp_path/'plan.json'
    study.write(plan, {'synthetic': True})
    (tmp_path/'execution').mkdir()
    def forbidden(*args, **kwargs):
        pytest.fail('Must not spawn for an existing execution')
    monkeypatch.setattr(study.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='already exists'):
        study.launch(plan, tmp_path/'execution', tmp_path/'audit', tmp_path/'launcher')
    assert not (tmp_path/'launcher').exists()


def latency_fixture():
    board = chess.Board()
    board.push_uci('e2e4')
    combined = [{'id': 'root-0', 'fen': chess.STARTING_FEN}, {'id': 'root-1', 'fen': board.fen()}]
    plan = {'configurations': study.configurations(), 'panels': {'latency_indices': [1, 0]}}
    warmups = [{'id': 'starting-board', 'index': i, 'choice': 'e2e4', 'wall_ms': .25} for i in range(3)]
    records = [{'id': 'root-1', 'index': 1, 'choice': 'e7e5', 'wall_ms': 1.},
               {'id': 'root-0', 'index': 0, 'choice': 'e2e4', 'wall_ms': 2.}]
    saved = [{'configuration': c['name'], 'device': 'cpu', 'torch_threads': 2, 'depth': 4,
              'warmup_records': copy.deepcopy(warmups), 'records': copy.deepcopy(records),
              'total_wall_ms': 3., 'warmup_wall_ms': .75} for c in plan['configurations']]
    return saved, plan, combined


def test_latency_accounting_replays_native_identity_legality_and_time():
    args = latency_fixture()
    assert study.audit_latency(*args) == args[0]


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'illegal', 'index', 'identity', 'nan',
                                 'negative', 'sum', 'warmup', 'device'])
def test_latency_accounting_rejects_corruption(fault):
    saved, plan, combined = latency_fixture()
    if fault == 'missing':
        saved.pop()
    elif fault == 'duplicate':
        saved[1] = copy.deepcopy(saved[0])
    elif fault == 'illegal':
        saved[0]['records'][0]['choice'] = 'e2e4'
    elif fault == 'index':
        saved[0]['records'][0]['index'] = 0
    elif fault == 'identity':
        saved[0]['records'][0]['id'] = 'wrong-root'
    elif fault == 'nan':
        saved[0]['records'][0]['wall_ms'] = float('nan')
    elif fault == 'negative':
        saved[0]['records'][0]['wall_ms'] = -.1
    elif fault == 'sum':
        saved[0]['total_wall_ms'] += 1.
    elif fault == 'warmup':
        saved[0]['warmup_records'].pop()
    else:
        saved[0]['device'] = 'mps'
    with pytest.raises(ValueError):
        study.audit_latency(saved, plan, combined)


def test_deadline_bypasses_policy_exception_handlers_and_arms_exact_budget(monkeypatch):
    callbacks, alarms = {}, []
    monkeypatch.setattr(study.signal, 'signal', lambda sig, callback: callbacks.__setitem__(sig, callback))
    monkeypatch.setattr(study.signal, 'alarm', alarms.append)
    study.deadline(14400)
    assert alarms == [14400]
    assert issubclass(study.DeadlineExceeded, BaseException)
    assert not issubclass(study.DeadlineExceeded, Exception)
    with pytest.raises(study.DeadlineExceeded):
        try:
            callbacks[study.signal.SIGALRM](study.signal.SIGALRM, None)
        except Exception:  # noqa: BLE001 - Model the inherited policy exception boundary.
            pytest.fail('A policy Exception handler must not swallow the hard deadline')


def test_run_preserves_hard_deadline_failure_without_evaluation(tmp_path, monkeypatch):
    plan, out, events = run_fixture(tmp_path, monkeypatch)
    def expired(*args, **kwargs):
        raise study.DeadlineExceeded('synthetic hard budget stop')
    monkeypatch.setattr(study, 'fit', expired)
    with pytest.raises(study.DeadlineExceeded):
        study.run(plan, out)
    assert events == []
    assert study.read(out/'failed.json')['error_type'] == 'DeadlineExceeded'
    assert not (out/'evaluation-started.json').exists() and not (out/'completed.json').exists()


@pytest.mark.parametrize('phase', ['run', 'audit'])
@pytest.mark.parametrize('duration', ['overbudget', 'infinite', 'negative', 'boolean'])
def test_supervisor_rejects_bad_duration_despite_zero_exit(tmp_path, monkeypatch, phase, duration):
    launcher, request = supervisor_fixture(tmp_path)
    calls = []
    def spawn(args, **kwargs):
        calls.append(args)
        command = args[2]
        target = Path(request['execution' if command == 'run' else 'audit'])
        name = 'completed.json' if command == 'run' else 'receipt.json'
        budget = study.PROTOCOL['primary_wall_seconds' if command == 'run' else 'audit_wall_seconds']
        field = 'total_wall_seconds' if command == 'run' else 'audit_wall_seconds'
        value = .1
        if command == phase:
            value = {'overbudget': budget + 1., 'infinite': float('inf'), 'negative': -1., 'boolean': True}[duration]
        target.mkdir()
        # Deliberately malformed evidence is written directly, never by the
        # production writer, which already forbids nonfinite JSON.
        (target/name).write_text(json.dumps({'status': 'completed', 'plan_sha256': request['plan_sha256'], field: value}))
        return SimpleNamespace(pid=12345 + len(calls), wait=lambda **kw: 0)
    monkeypatch.setattr(study.subprocess, 'Popen', spawn)
    with pytest.raises(ValueError):
        study.supervise(launcher)
    assert len(calls) == (1 if phase == 'run' else 2)
    assert study.read(launcher/'failed.json')['phase'] == phase
    assert not (launcher/'completed.json').exists()
