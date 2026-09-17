"""Tiny CPU runner/report integration, with synthetic source/teacher/arena only.

Real caches, four candidate architectures, optimizer updates, evaluation,
checkpoints, receipt audits and frozen secondary-score arithmetic are retained.
No project data, engine process or real arena game is used.
"""

import copy
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import torch

from openjev.research.chess_candidate import CandidateChess

ROOT = Path(__file__).resolve().parents[1]


def load_study():
    spec = importlib.util.spec_from_file_location('_candidate_integration', ROOT/'scripts/chess_candidate_study.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in values))


def synthetic_rows(prefix, n=3):
    fens = [chess.STARTING_FEN, '7k/P7/8/8/8/8/8/7K w - - 0 1',
            '7k/6R1/8/5K2/8/8/8/8 b - - 0 1']
    return [{'id': f'{prefix}-{i}', 'game_id': i, 'fen': fen,
             'target_uci': min(m.uci() for m in chess.Board(fen).legal_moves),
             'target_value': .15*(-1)**i} for i, fen in enumerate(fens[:n])]


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    study = load_study()
    frozen_grader = study.grading_module()
    sandbox = tmp_path_factory.mktemp('candidate-integration')
    patch = pytest.MonkeyPatch()
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    protocol = copy.deepcopy(study.PROTOCOL)
    protocol.update(width=4, train_examples=3, epochs=1, batch_size=2, microbatch_size=1,
                    updates_per_fit=2, training_device='cpu', candidate_chunk_size=8,
                    regret_positions_per_split=1, latency_positions_per_split=1, bootstrap_replicates=16)
    patch.setattr(study, 'PROTOCOL', protocol)
    patch.setattr(study, 'ROOT', sandbox)
    training = synthetic_rows('train')
    evaluation = {s: synthetic_rows(s, 2) for s in ('dev', 'shift')}
    exclusions = {'states': [], 'files': {}, 'counts': {}}
    patch.setattr(study.data, 'collect_exclusions', lambda _: copy.deepcopy(exclusions))
    patch.setattr(study.data, 'training_rows', lambda _: copy.deepcopy(training))

    def generate(out, engine, states):
        assert states == []
        out.mkdir()
        for split, values in evaluation.items():
            jsonl(out/f'{split}.jsonl', values)
        write(out/'completed.json', {'status': 'completed', 'synthetic_external_data': True,
                                     'files': study.tree_files(out)})

    def validate(out, states):
        assert states == []
        receipt = study.read(out/'completed.json')
        assert receipt['status'] == 'completed'
        assert receipt['files'] == {f'{s}.jsonl': study.sha(out/f'{s}.jsonl') for s in evaluation}
        result = {s: study.rows(out/f'{s}.jsonl') for s in evaluation}
        assert result == evaluation
        return result

    patch.setattr(study.data, 'generate', generate)
    patch.setattr(study.data, 'validate', validate)
    engine_hash = 'e'*64
    frozen = {'protocol': protocol, 'configurations': study.configs(), 'engine_sha256': engine_hash,
              'scope': 'Synthetic integration fixture, not study evidence'}
    patch.setattr(study, 'signature', lambda exclusions=None: copy.deepcopy(frozen))

    def score_secondary(plan, directory, decisions):
        analyses, records, lookup = [], [], {}
        for split in ('dev', 'shift'):
            for index in plan['secondary_indices'][split]:
                row = plan['panels'][split][index]
                selected = [r for r in decisions if r['split'] == split and r['panel_index'] == index]
                legal = sorted(m.uci() for m in chess.Board(row['fen']).legal_moves)
                for move in [None, *sorted({r['choice'] for r in selected})]:
                    cp = 300 if move is None else 100+legal.index(move)
                    record = {'split': split, 'panel_index': index, 'id': row['id'], 'root_move': move,
                              'pv_first': legal[0] if move is None else move, 'score_cp': cp, 'mate': None,
                              'bounded_score': math.tanh(cp/600), 'requested_nodes': 20000,
                              'reported_nodes': 20001, 'wall_seconds': .01}
                    analyses.append(record)
                    lookup[split, index, move] = record
                for decision in selected:
                    best, choice = lookup[split, index, None], lookup[split, index, decision['choice']]
                    records.append({**decision, 'cp_loss': best['score_cp']-choice['score_cp'],
                                    'bounded_regret': best['bounded_score']-choice['bounded_score']})
        cost = {'calls': len(analyses), **{key: sum(r[key] for r in analyses)
                                         for key in ('requested_nodes', 'reported_nodes', 'wall_seconds')}}
        frozen_grader.validate_secondary(plan, decisions, analyses, records, cost)
        write(directory/'engine.json', {'id': {'name': 'Stockfish 19 synthetic fixture only'}, 'sha256': engine_hash})
        jsonl(directory/'analyses.jsonl', analyses)
        jsonl(directory/'regret.jsonl', records)
        return cost

    grader = SimpleNamespace(score_secondary=score_secondary, validate_secondary=frozen_grader.validate_secondary)
    patch.setattr(study, 'grading_module', lambda: grader)

    def arena_run(out, models, plan_hash):
        out.mkdir()
        assert set(models) == {(c['arm'], c['seed']) for c in study.configs()}
        bindings = {f'{arm}-{seed}': {'state_sha256': study.digest_state(model)}
                    for (arm, seed), model in models.items()}
        write(out/'started.json', {'status': 'started', 'plan_sha256': plan_hash, 'models': bindings})
        write(out/'summary.json', {'status': 'synthetic_arena_fixture', 'gate_passed': False, 'games': 0})
        write(out/'completed.json', {'status': 'completed', 'plan_sha256': plan_hash, 'files': study.tree_files(out)})

    def arena_report(saved, plan_hash):
        receipt = study.read(saved/'completed.json')
        assert receipt['plan_sha256'] == plan_hash
        assert receipt['files'] == {n: study.sha(saved/n) for n in ('started.json', 'summary.json')}
        assert study.read(saved/'started.json')['plan_sha256'] == plan_hash
        return study.read(saved/'summary.json')

    patch.setattr(study.arena, 'run', arena_run)
    patch.setattr(study.arena, 'report', arena_report)
    events = []
    original_fit, original_evaluate, original_cache = study.fit, study.evaluate, study.CachedPositions

    def tracked_cache(values):
        events.append(('cache', tuple(r['id'] for r in values)))
        return original_cache(values)

    def tracked_fit(config, cache, *args):
        assert tuple(r['id'] for r in cache.rows) == tuple(r['id'] for r in training)
        result = original_fit(config, cache, *args)
        events.append(('fit', config['name']))
        return result

    def tracked_evaluate(model, cache, outfile, **kwargs):
        assert len([e for e in events if e[0] == 'fit']) == len(study.configs())
        assert all((sandbox/'execution'/c['name']/'training.json').exists() for c in study.configs())
        assert all(not r['id'].startswith('train-') for r in cache.rows)
        events.append(('evaluate', model.arm, model.seed, kwargs.get('permuted', False)))
        return original_evaluate(model, cache, outfile, **kwargs)

    patch.setattr(study, 'CachedPositions', tracked_cache)
    patch.setattr(study, 'fit', tracked_fit)
    patch.setattr(study, 'evaluate', tracked_evaluate)
    plan = study.prepare(sandbox/'plan')
    execution = sandbox/'execution'
    study.run(plan, execution)
    report = study.report(plan, execution, sandbox/'report')
    try:
        yield {'study': study, 'plan': plan, 'execution': execution, 'report': report,
               'events': events, 'training': training, 'grader': grader}
    finally:
        patch.undo()
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


def reseal(study, execution, fit=None):
    if fit is not None:
        directory = execution/fit
        receipt = study.read(directory/'completed.json')
        receipt['files'] = {n: study.sha(directory/n) for n in receipt['files']}
        write(directory/'completed.json', receipt)
    complete = study.read(execution/'completed.json')
    complete['files'] = study.tree_files(execution)
    complete['files'].pop('completed.json')
    write(execution/'completed.json', complete)


def test_actual_tiny_twelve_fit_pipeline_audits_without_any_rerun(completed, tmp_path, monkeypatch):
    study, report = completed['study'], completed['report']
    names = {c['name'] for c in study.configs()}
    assert len(names) == 12 and set(report['metrics']) == names
    assert set(report['permutation_diagnostic']) == {f'delta-{s}' for s in study.PROTOCOL['seeds']}
    assert report['continuation_passed'] is False and report['elo_estimate'] is None
    assert all(row['updates'] == 2 and row['examples_seen'] == 3 for row in report['costs'].values())
    events = completed['events']
    assert len([e for e in events if e[0] == 'cache']) == 3
    first_evaluation = next(i for i, e in enumerate(events) if e[0] == 'evaluate')
    assert sum(e[0] == 'fit' for e in events[:first_evaluation]) == 12
    assert len([e for e in events if e[0] == 'evaluate']) == 30  # 24 primary + 6 diagnostic
    journals = {c['name']: study.rows(completed['execution']/c['name']/'learning.jsonl') for c in study.configs()}
    for seed in study.PROTOCOL['seeds']:
        assert len({tuple(r['indices_sha256'] for r in journals[f'{a}-{seed}']) for a in study.ARMS}) == 1
    def forbidden(*_, **__):
        raise AssertionError('Report must never train, infer, generate, or invoke the engine')
    monkeypatch.setattr(CandidateChess, 'forward', forbidden)
    monkeypatch.setattr(study, 'fit', forbidden)
    monkeypatch.setattr(study, 'CachedPositions', forbidden)
    monkeypatch.setattr(study.data, 'generate', forbidden)
    monkeypatch.setattr(completed['grader'], 'score_secondary', forbidden)
    assert study.report(completed['plan'], completed['execution'], tmp_path/'reproduced') == report


@pytest.mark.parametrize('fault', ['unbound_file', 'cache', 'learning_count', 'permutation',
                                  'checkpoint', 'arena_binding', 'fit_membership', 'grading'])
def test_rehashed_tampering_is_rejected_by_report(completed, tmp_path, fault):
    study = completed['study']
    execution = tmp_path/'execution'
    shutil.copytree(completed['execution'], execution)
    fit = 'delta-97'
    directory = execution/fit
    if fault == 'unbound_file':
        (directory/'dev.jsonl').write_text((directory/'dev.jsonl').read_text()+'\n')
    elif fault == 'cache':
        value = study.read(execution/'training-cache.json')
        value['input_rows_sha256'] = '0'*64
        write(execution/'training-cache.json', value)
        reseal(study, execution)
    elif fault == 'learning_count':
        records = study.rows(directory/'learning.jsonl')[:-1]
        jsonl(directory/'learning.jsonl', records)
        value = study.read(directory/'training.json')
        value['learning_sha256'] = study.sha(directory/'learning.jsonl')
        write(directory/'training.json', value)
        reseal(study, execution, fit)
    elif fault == 'permutation':
        records = study.rows(directory/'permuted-dev.jsonl')
        records[0]['successor_permutation_offset'] = 0
        jsonl(directory/'permuted-dev.jsonl', records)
        reseal(study, execution, fit)
    elif fault == 'checkpoint':
        value = torch.load(directory/'weights.pt', weights_only=True)
        value['state_dict']['encoder.0.weight'].flatten()[0] += .25
        torch.save(value, directory/'weights.pt')
        trained = study.read(directory/'training.json')
        trained['weights_sha256'] = study.sha(directory/'weights.pt')
        write(directory/'training.json', trained)
        reseal(study, execution, fit)
    elif fault == 'arena_binding':
        value = study.read(execution/'arena/started.json')
        value['models'][fit]['state_sha256'] = '0'*64
        write(execution/'arena/started.json', value)
        receipt = study.read(execution/'arena/completed.json')
        receipt['files']['started.json'] = study.sha(execution/'arena/started.json')
        write(execution/'arena/completed.json', receipt)
        reseal(study, execution)
    elif fault == 'fit_membership':
        receipt = study.read(directory/'completed.json')
        receipt['files'].pop('latency.json')
        write(directory/'completed.json', receipt)
        reseal(study, execution)
    else:
        records = study.rows(execution/'regret/regret.jsonl')
        records[0]['bounded_regret'] += .1
        jsonl(execution/'regret/regret.jsonl', records)
        receipt = study.read(execution/'regret/completed.json')
        receipt['files']['regret.jsonl'] = study.sha(execution/'regret/regret.jsonl')
        write(execution/'regret/completed.json', receipt)
        reseal(study, execution)
    with pytest.raises(ValueError):
        study.report(completed['plan'], execution, tmp_path/'must-not-publish')
    assert not (tmp_path/'must-not-publish').exists()
