"""Synthetic arena/provenance tests. No trained checkpoint or benchmark game."""

import copy
import importlib.util
import json
from pathlib import Path

import chess
import pytest
import torch

from openjev.research.chess_arena import play_game

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('chess_mate_arena_test', ROOT/'scripts/chess_mate_arena.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def environment(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    (root/'source.py').write_text('synthetic frozen source\n')
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'SOURCES', ('source.py',))
    spatial_plan = root/'evidence/chess-spatial-v1/plan.json'
    write(spatial_plan, {'synthetic_spatial_plan': True})
    for mode, seeds in (('predict', study.SEEDS), ('cnn', (17,))):
        for seed in seeds:
            path = root/f'models/chess-spatial-v1/{mode}-{seed}/weights.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(f'synthetic {mode} {seed}, not actual weights'.encode())
    training = root/'evidence/chess-mate-v1/plan.json'
    write(training.parent/'selection.json', {'synthetic_selection': True})
    names = ['source.py', 'evidence/chess-spatial-v1/plan.json']
    names += [f'models/chess-spatial-v1/predict-{seed}/weights.pt' for seed in study.SEEDS]
    write(training, {'protocol': {'seeds': list(study.SEEDS), 'arms': ['single', 'set'],
                                   'base_mode': 'predict', 'depth': 4, 'width': 32, 'updates_per_fit': 2},
                     'selection_sha256': study.sha(training.parent/'selection.json'),
                     'inputs': {name: study.sha(root/name) for name in names}})
    execution = root/'runs/mate-training'
    arena = study.prepare(root/'evidence/arena', training, execution)
    return root, training, execution, arena


def complete_training(training, execution):
    write(execution/'schedule.json', {'synthetic_schedule': True})
    receipts = {}
    for arm in ('frozen', 'single', 'set'):
        for seed in study.SEEDS:
            name = f'{arm}-{seed}'
            fit = execution/name
            fit.mkdir()
            filenames = ['mate_dev.jsonl', 'mate_confirm.jsonl', 'dev.jsonl', 'shift.jsonl']
            if arm != 'frozen':
                filenames += ['weights.pt', 'learning.jsonl']
            for filename in filenames:
                (fit/filename).write_bytes(f'synthetic {name} {filename}\n'.encode())
            write(fit/'completed.json', {'status': 'completed', 'arm': arm, 'seed': seed,
                                         'plan_sha256': study.sha(training),
                                         'updates': 0 if arm == 'frozen' else 2,
                                         'files': {filename: study.sha(fit/filename) for filename in filenames}})
            receipts[name] = study.sha(fit/'completed.json')
    write(execution/'completed.json', {'status': 'completed', 'plan_sha256': study.sha(training),
                                       'fits': receipts, 'schedule_sha256': study.sha(execution/'schedule.json')})


def fool_mate(board, depth=None):
    legal = sorted(move.uci() for move in board.legal_moves)
    return {'choice': ['f2f3', 'e7e5', 'g2g4', 'd8h4'][len(board.move_stack)],
            'probabilities': dict.fromkeys(legal, 1/len(legal)), 'value': 0.}


def protocol():
    return {'initial_fen': chess.STARTING_FEN, 'clock_seconds': 300., 'max_plies': 160,
            'claim_draw': False, 'games': study.schedule(), 'illustrative_game': 'game-01',
            'additional_illustration': 'game-09'}


def test_fixed_sixteen_game_schedule_balances_opponents_colors_and_illustrations():
    games = study.schedule()
    assert len(games) == len({game['id'] for game in games}) == 16
    assert [games[index]['white'] for index in (0, 4, 8, 12)] == list(study.ARMS)
    assert games[0] == {'id': 'game-01', 'white': 'frozen_predict17', 'black': 'greedy_material'}
    assert games[8] == {'id': 'game-09', 'white': 'set17', 'black': 'greedy_material'}
    assert games[-1] == {'id': 'game-16', 'white': 'cnn17', 'black': 'guarded_frozen_predict17'}
    for arm in study.ARMS:
        for opponent in ('greedy_material', 'cnn17'):
            assert sum(g['white'] == arm and g['black'] == opponent for g in games) == 1
            assert sum(g['black'] == arm and g['white'] == opponent for g in games) == 1


def test_prepare_precedes_training_completion_and_never_scores_or_creates_execution(tmp_path, monkeypatch):
    root, _, execution, arena = environment(tmp_path, monkeypatch)
    plan = study.verify_plan(arena)
    assert not execution.exists()
    assert plan['protocol']['warmups'] == plan['protocol']['retries'] == 0
    assert plan['protocol']['fallback'] is None and plan['protocol']['model_seed'] == 17
    assert plan['protocol']['clock_seconds'] == 300 and plan['protocol']['max_plies'] == 160
    assert plan['protocol']['additional_illustration'] == 'game-09'
    with pytest.raises(FileNotFoundError):
        study.verify_training(plan)
    with pytest.raises(FileExistsError):
        study.prepare(root/'evidence/arena', plan['training_plan']['path'], execution)


@pytest.mark.parametrize('kind', ['source', 'training', 'predict', 'cnn', 'schedule'])
def test_arena_plan_binds_sources_originals_and_schedule(tmp_path, monkeypatch, kind):
    root, training, _, arena = environment(tmp_path, monkeypatch)
    if kind == 'source':
        (root/'source.py').write_text('changed')
    elif kind == 'training':
        training.write_text('{}')
    elif kind in ('predict', 'cnn'):
        (root/f'models/chess-spatial-v1/{kind}-17/weights.pt').write_bytes(b'changed')
    else:
        value = json.loads(arena.read_text())
        value['protocol']['games'].reverse()
        write(arena, value)
    with pytest.raises(ValueError, match='binding changed'):
        study.verify_plan(arena)


def test_training_binding_retains_all_nine_receipts_and_six_weights(tmp_path, monkeypatch):
    _, training, execution, arena = environment(tmp_path, monkeypatch)
    complete_training(training, execution)
    binding = study.verify_training(study.verify_plan(arena))
    assert len(binding['fit_receipts']) == 9 and len(binding['checkpoints']) == 6
    assert len(binding['references']) == 3
    assert set(binding['selected_checkpoints']) == {'single-17', 'set-17'}
    (execution/'single-43/weights.pt').write_bytes(b'corrupted unselected weight')
    with pytest.raises(ValueError, match='checkpoint or evidence changed'):
        study.verify_training(study.verify_plan(arena))


@pytest.mark.parametrize('kind', ['partial', 'failed', 'receipt', 'provenance', 'reference', 'selection', 'schedule'])
def test_incomplete_or_corrupt_training_never_starts_an_arena(tmp_path, monkeypatch, kind):
    _, training, execution, arena = environment(tmp_path, monkeypatch)
    complete_training(training, execution)
    complete = study.read(execution/'completed.json')
    if kind == 'partial':
        complete['fits'].pop('set-43')
    elif kind == 'failed':
        write(execution/'failed.json', {'status': 'failed'})
    elif kind == 'receipt':
        complete['fits']['single-29'] = '0'*64
    elif kind == 'provenance':
        path = execution/'single-29/completed.json'
        receipt = study.read(path)
        receipt['plan_sha256'] = '0'*64
        write(path, receipt)
        complete['fits']['single-29'] = study.sha(path)
    elif kind == 'reference':
        (execution/'frozen-43/dev.jsonl').write_text('corrupted reference evidence')
    elif kind == 'selection':
        (training.parent/'selection.json').write_text('{}')
    else:
        (execution/'schedule.json').write_text('{}')
    write(execution/'completed.json', complete)
    with pytest.raises(ValueError):
        study.verify_training(study.verify_plan(arena))


class StubModel:
    def __init__(self, mode):
        self.mode, self.seed, self.width, self.depth = mode, 17, 32, 4

    def parameters(self):
        return iter([torch.empty(0)])

    choose = staticmethod(fool_mate)


def test_loading_uses_fixed_weights_and_guard_shares_reference_without_warmup(tmp_path, monkeypatch):
    _, training, execution, arena = environment(tmp_path, monkeypatch)
    complete_training(training, execution)
    plan = study.verify_plan(arena)
    binding = study.verify_training(plan)
    calls = []

    def load(path, *, expected_plan_sha256):
        calls.append((str(path), expected_plan_sha256))
        return StubModel('cnn' if 'cnn-17' in str(path) else 'predict')

    monkeypatch.setattr(study.SpatialChess, 'load', load)
    policies, loads = study.load_policies(plan, binding)
    assert len(calls) == len(loads) == 4
    assert set(policies) == {*study.ARMS, 'greedy_material', 'cnn17'}
    assert policies['guarded_frozen_predict17'].func is study.terminal_guard_decision
    assert policies['guarded_frozen_predict17'].args[0].mode == 'predict'
    assert policies['guarded_frozen_predict17'].keywords == {'depth': 4}
    assert {digest for _, digest in calls} == {study.sha(training), plan['originals']['cnn17']['training_plan_sha256']}


def test_synthetic_arena_keeps_every_game_failure_and_unfinished_state(tmp_path):
    spec = protocol()
    policies = {name: fool_mate for game in spec['games'] for name in (game['white'], game['black'])}
    policies['greedy_material'] = lambda _board: {'choice': '0000'}
    summary = study.execute_schedule({'protocol': spec}, policies, tmp_path)
    assert summary['status_counts'] == {'completed': 8, 'unfinished': 0, 'failed': 8}
    assert summary['policies']['greedy_material']['failed'] == 8
    assert summary['policies']['greedy_material']['losses'] == 0
    assert (tmp_path/'game-16.json').exists() and (tmp_path/'game-16.pgn').exists()
    limited = {**protocol(), 'max_plies': 1}
    first = limited['games'][0]
    result = play_game(fool_mate, fool_mate, white_name=first['white'], black_name=first['black'], max_plies=1)
    game = {'game_id': first['id'], **result.to_dict()}
    study.validate_game(game, first, limited)
    assert game['status'] == 'unfinished' and game['result'] == '*'
    bad = {**game, 'status': 'completed', 'result': '1/2-1/2'}
    with pytest.raises(ValueError):
        study.validate_game(bad, first, limited)


def synthetic_arena(tmp_path, monkeypatch):
    root, training, source, arena = environment(tmp_path, monkeypatch)
    complete_training(training, source)

    def load_policies(plan, binding):
        policies = {name: fool_mate for game in plan['protocol']['games'] for name in (game['white'], game['black'])}
        artifacts = dict(plan['originals'])
        artifacts.update({f'{arm}17': binding['selected_checkpoints'][f'{arm}-17'] for arm in ('single', 'set')})
        loads = [{'policy': name, 'path': a['path'], 'sha256': a['sha256'], 'wall_seconds': .0,
                  'shared_with': ['guarded_frozen_predict17'] if name == 'frozen_predict17' else []}
                 for name, a in artifacts.items()]
        return policies, loads

    monkeypatch.setattr(study, 'load_policies', load_policies)
    execution = root/'runs/arena'
    study.run(arena, execution)
    return arena, execution


def test_synthetic_run_and_report_replay_all_sixteen_games(tmp_path, monkeypatch):
    arena, execution = synthetic_arena(tmp_path, monkeypatch)
    summary = study.report(arena, execution, tmp_path/'report')
    assert summary['games'] == 16 and summary['status_counts']['completed'] == 16
    assert summary['additional_illustration'] == 'game-09'
    assert summary['elo_estimate'] is None and summary['novelty_established'] is False
    with pytest.raises(FileExistsError):
        study.run(arena, execution)


@pytest.mark.parametrize('kind', ['hash', 'clock', 'pgn', 'summary', 'loads', 'missing_game'])
def test_report_rejects_corrupted_complete_evidence(tmp_path, monkeypatch, kind):
    arena, execution = synthetic_arena(tmp_path, monkeypatch)
    complete = study.read(execution/'completed.json')
    if kind == 'hash':
        with (execution/'game-01.json').open('a') as stream:
            stream.write(' ')
    elif kind == 'clock':
        game = study.read(execution/'game-01.json')
        game['attempts'][0]['clocks']['white'] = 1.
        write(execution/'game-01.json', game)
        complete['files']['game-01.json'] = study.sha(execution/'game-01.json')
    elif kind == 'pgn':
        (execution/'game-01.pgn').write_text('[Result "1/2-1/2"]\n\n1/2-1/2\n')
        complete['files']['game-01.pgn'] = study.sha(execution/'game-01.pgn')
    elif kind == 'summary':
        value = study.read(execution/'summary.json')
        value['policies']['set17']['wins'] += 1
        write(execution/'summary.json', value)
        complete['files']['summary.json'] = study.sha(execution/'summary.json')
    elif kind == 'loads':
        value = study.read(execution/'loads.json')
        value[0]['shared_with'] = []
        write(execution/'loads.json', value)
        complete['files']['loads.json'] = study.sha(execution/'loads.json')
    else:
        complete['files'].pop('game-16.json')
    write(execution/'completed.json', complete)
    with pytest.raises(ValueError):
        study.report(arena, execution, tmp_path/'report')


def test_clock_corruption_cannot_change_recorded_timeout_to_a_scored_win():
    spec = protocol()
    game = spec['games'][0]
    ticks = iter((0., 301.))
    result = play_game(fool_mate, fool_mate, white_name=game['white'], black_name=game['black'],
                       max_plies=spec['max_plies'], timer=lambda: next(ticks))
    trace = {'game_id': game['id'], **result.to_dict()}
    study.validate_game(trace, game, spec)
    assert trace['termination'] == 'timeout' and trace['result'] == '0-1'
    changed = copy.deepcopy(trace)
    changed['result'] = '1-0'
    with pytest.raises(ValueError):
        study.validate_game(changed, game, spec)
