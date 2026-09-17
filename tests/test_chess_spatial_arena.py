import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('chess')
import chess

from openjev.research.chess_arena import play_game

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('chess_spatial_arena_study', ROOT/'scripts/chess_spatial_arena.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def environment(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    (root/'source.py').write_text('frozen source\n')
    monkeypatch.setattr(study, 'ROOT', root)
    monkeypatch.setattr(study, 'SOURCES', ('source.py',))
    write(root/'evidence/chess-student-v1/plan.json', {'old': True})
    old = root/'models/chess-student-v1/gru-17/weights.pt'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'old synthetic checkpoint')
    training = root/'evidence/chess-spatial-v1/plan.json'
    write(training, {'protocol': {'modes': list(study.MODES), 'seeds': list(study.SEEDS)},
                     'sources': {'source.py': study.sha(root/'source.py')}})
    execution = root/'runs/execution'
    arena_path = study.prepare(root/'evidence/arena', training, execution)
    return root, training, execution, arena_path


def complete_training(training, execution):
    write(execution/'data/completed.json', {'status': 'completed'})
    receipts = {}
    for mode in study.MODES:
        for seed in study.SEEDS:
            name = f'{mode}-{seed}'
            directory = execution/name
            directory.mkdir()
            for filename in ('weights.pt', 'learning.jsonl', 'dev.json', 'shift.json'):
                (directory/filename).write_bytes(f'synthetic {name} {filename}'.encode())
            write(directory/'completed.json', {
                'status': 'completed', 'mode': mode, 'seed': seed, 'plan_sha256': study.sha(training),
                'data_receipt_sha256': study.sha(execution/'data/completed.json'),
                'files': {filename: study.sha(directory/filename)
                          for filename in ('weights.pt', 'learning.jsonl', 'dev.json', 'shift.json')},
            })
            receipts[name] = study.sha(directory/'completed.json')
    write(execution/'completed.json', {'status': 'completed', 'fits': 12,
                                       'plan_sha256': study.sha(training), 'fit_receipts': receipts})


def fool_mate(board):
    return {'choice': ['f2f3', 'e7e5', 'g2g4', 'd8h4'][len(board.move_stack)]}


def protocol():
    return {'initial_fen': chess.STARTING_FEN, 'clock_seconds': 300., 'max_plies': 160,
            'claim_draw': False, 'games': study.schedule(), 'illustrative_game': 'game-01'}


def test_schedule_is_fixed_balanced_and_first_illustration_is_preselected():
    games = study.schedule()
    assert len(games) == 16 and len({game['id'] for game in games}) == 16
    assert games[0] == {'id': 'game-01', 'white': 'predict-17', 'black': 'greedy_material'}
    assert games[-1] == {'id': 'game-16', 'white': 'gru_v1', 'black': 'cnn-17'}
    assert [games[offset]['white'] for offset in (0, 4, 8, 12)] == [
        'predict-17', 'reconstruct-17', 'recurrent-17', 'cnn-17']
    for mode in study.MODES:
        for opponent in ('greedy_material', 'gru_v1'):
            assert sum(game['white'] == f'{mode}-17' and game['black'] == opponent for game in games) == 1
            assert sum(game['black'] == f'{mode}-17' and game['white'] == opponent for game in games) == 1


def test_prepare_binds_before_training_finishes_and_does_not_create_execution(tmp_path, monkeypatch):
    root, _, execution, arena_path = environment(tmp_path, monkeypatch)
    plan = study.verify_plan(arena_path)
    assert not execution.exists()
    assert plan['protocol']['illustrative_game'] == 'game-01'
    assert plan['protocol']['clock_seconds'] == 300 and plan['protocol']['max_plies'] == 160
    assert plan['protocol']['device'] == 'cpu' and plan['protocol']['torch_threads'] == 2
    assert plan['protocol']['retries'] == 0 and plan['protocol']['fallback'] is None
    with pytest.raises(FileNotFoundError):
        study.verify_training(plan)
    with pytest.raises(FileExistsError):
        study.prepare(root/'evidence/arena', plan['training_plan']['path'], execution)


@pytest.mark.parametrize('changed', ['source', 'training_plan', 'old_weights', 'schedule'])
def test_prepared_binding_rejects_corruption(tmp_path, monkeypatch, changed):
    root, training, _, arena_path = environment(tmp_path, monkeypatch)
    if changed == 'source':
        (root/'source.py').write_text('changed')
    elif changed == 'training_plan':
        training.write_text('{}')
    elif changed == 'old_weights':
        (root/'models/chess-student-v1/gru-17/weights.pt').write_bytes(b'changed')
    else:
        plan = json.loads(arena_path.read_text())
        plan['protocol']['games'].reverse()
        write(arena_path, plan)
    with pytest.raises(ValueError, match='binding changed'):
        study.verify_plan(arena_path)


def test_all_twelve_receipts_and_weights_bound_before_seed17_selection(tmp_path, monkeypatch):
    _, training, execution, arena_path = environment(tmp_path, monkeypatch)
    complete_training(training, execution)
    binding = study.verify_training(study.verify_plan(arena_path))
    assert binding['status'] == 'verified' and len(binding['checkpoints']) == len(binding['fit_receipts']) == 12
    assert set(binding['selected_checkpoints']) == {f'{mode}-17' for mode in study.MODES}
    # A non-selected seed is still part of the frozen required training execution.
    (execution/'cnn-43/weights.pt').write_bytes(b'corrupted unselected weight')
    with pytest.raises(ValueError, match='checkpoint or evidence'):
        study.verify_training(study.verify_plan(arena_path))


@pytest.mark.parametrize('changed', ['partial', 'failed', 'receipt', 'provenance', 'data'])
def test_incomplete_or_corrupt_training_is_rejected(tmp_path, monkeypatch, changed):
    _, training, execution, arena_path = environment(tmp_path, monkeypatch)
    complete_training(training, execution)
    complete = json.loads((execution/'completed.json').read_text())
    if changed == 'partial':
        complete['fit_receipts'].pop('predict-43')
    elif changed == 'failed':
        write(execution/'failed.json', {'status': 'failed'})
    elif changed == 'receipt':
        complete['fit_receipts']['predict-43'] = '0'*64
    elif changed == 'provenance':
        fit_path = execution/'predict-43/completed.json'
        receipt = json.loads(fit_path.read_text())
        receipt['plan_sha256'] = '0'*64
        write(fit_path, receipt)
        complete['fit_receipts']['predict-43'] = study.sha(fit_path)
    elif changed == 'data':
        write(execution/'data/failed.json', {'status': 'failed'})
    write(execution/'completed.json', complete)
    with pytest.raises(ValueError):
        study.verify_training(study.verify_plan(arena_path))


def test_complete_synthetic_schedule_keeps_every_game_and_pgn(tmp_path):
    specification = protocol()
    policies = {name: fool_mate for game in specification['games'] for name in (game['white'], game['black'])}
    summary = study.execute_schedule({'protocol': specification}, policies, tmp_path)
    assert summary['games'] == 16
    assert summary['status_counts'] == {'completed': 16, 'unfinished': 0, 'failed': 0}
    assert len(list(tmp_path.glob('game-*.json'))) == len(list(tmp_path.glob('game-*.pgn'))) == 16
    first = json.loads((tmp_path/'game-01.json').read_text())
    assert first['result'] == '0-1' and first['termination'] == 'checkmate'
    assert len(first['attempts']) == len(first['moves']) == 4


def test_policy_failure_is_unscored_and_does_not_skip_later_scheduled_games(tmp_path):
    specification = protocol()
    policies = {name: fool_mate for game in specification['games'] for name in (game['white'], game['black'])}
    policies['greedy_material'] = lambda board: {'choice': '0000'}
    summary = study.execute_schedule({'protocol': specification}, policies, tmp_path)
    assert summary['status_counts'] == {'completed': 8, 'unfinished': 0, 'failed': 8}
    assert summary['policies']['greedy_material']['failed'] == 8
    assert summary['policies']['greedy_material']['losses'] == 0
    assert (tmp_path/'game-16.pgn').exists()


@pytest.mark.parametrize('kind', ['unfinished', 'failed', 'timeout'])
def test_replay_status_clock_validation_and_non_scored_states(kind):
    specification = protocol()
    spec = specification['games'][0]
    options = {}
    white = fool_mate
    if kind == 'unfinished':
        specification['max_plies'] = 1
    elif kind == 'failed':
        def white(board):
            raise RuntimeError('synthetic policy error')
    else:
        ticks = iter((0., 301.))
        options['timer'] = lambda: next(ticks)
    result = play_game(white, fool_mate, white_name=spec['white'], black_name=spec['black'],
                       max_plies=specification['max_plies'], **options)
    game = {'game_id': spec['id'], **result.to_dict()}
    study.validate_game(game, spec, specification)
    if kind == 'timeout':
        assert game['status'] == 'completed' and game['result'] == '0-1' and not game['moves']
    else:
        assert game['status'] == kind and game['result'] == '*'
    corrupted = copy.deepcopy(game)
    corrupted['attempts'][0]['clocks']['white'] = 299.
    with pytest.raises(ValueError, match='clock debit'):
        study.validate_game(corrupted, spec, specification)


def test_ply_cap_is_never_promoted_to_a_draw():
    specification = {**protocol(), 'max_plies': 1}
    spec = specification['games'][0]
    result = play_game(fool_mate, fool_mate, white_name=spec['white'], black_name=spec['black'], max_plies=1)
    game = {'game_id': spec['id'], **result.to_dict(), 'status': 'completed', 'result': '1/2-1/2'}
    with pytest.raises(ValueError, match='legal outcome'):
        study.validate_game(game, spec, specification)
