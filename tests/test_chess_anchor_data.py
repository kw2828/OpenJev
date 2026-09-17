import copy
import hashlib
import json

import pytest

pytest.importorskip('chess')
import chess
import chess.engine

from openjev.research import chess_anchor_data as data
from openjev.research.chess_spatial_data import state_key


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def game():
    board = chess.Board()
    initial = board.fen()
    moves = []
    for ply, token in enumerate(('e2e4', 'e7e5'), 1):
        move = chess.Move.from_uci(token)
        row = {'played': True, 'ply': ply, 'side': 'white' if board.turn else 'black',
               'fen_before': board.fen(), 'move_uci': token, 'san': board.san(move)}
        board.push(move)
        row['fen_after'] = board.fen()
        moves.append(row)
    return {'game_id': 'game-01', 'status': 'unfinished', 'initial_fen': initial,
            'final_fen': board.fen(), 'moves': moves, 'attempts': copy.deepcopy(moves)}


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'DATA_SOURCES', tuple((path, split, 1) for path, split, _ in data.DATA_SOURCES))
    monkeypatch.setattr(data, 'MATE_COUNTS', {'train': 1, 'dev': 1, 'confirm': 1})
    monkeypatch.setattr(data, 'PUZZLE_COUNT', 2)
    monkeypatch.setattr(data, 'ARENA_SOURCES', tuple((path, prefix, 1) for path, prefix, _ in data.ARENA_SOURCES))
    for index, (path, split, _) in enumerate(data.DATA_SOURCES):
        write(tmp_path/path, {'id': f'row-{index}', 'split': split, 'fen': chess.STARTING_FEN,
                              'state_key': state_key(chess.Board()), 'target_uci': 'secret_teacher_label',
                              'score_cp': 123456})
    selection = {'splits': {split: [{'id': f'mate-{split}', 'fen': chess.STARTING_FEN,
                                    'target_uci': 'heldout_label_not_an_input'}]
                            for split in data.MATE_COUNTS}}
    write(tmp_path/data.MATE_SELECTION, selection)
    puzzles = []
    for index, setup in enumerate(('d2d4', 'g1f3')):
        board = chess.Board()
        board.push_uci(setup)
        puzzles.append({'puzzle_id': str(index), 'source_fen': chess.STARTING_FEN,
                        'setup_uci': setup, 'solver_fen': board.fen(), 'gold_first_uci': 'unused_label'})
    write(tmp_path/data.PUZZLE_PLAN, {'puzzles': {'positions': puzzles}})
    for path, prefix, _ in data.ARENA_SOURCES:
        write(tmp_path/path/(prefix+'01.json'), game())
        write(tmp_path/path/'completed.json', {'not_a_position': True})
        write(tmp_path/path/'summary.json', {'not_a_position': True})
        write(tmp_path/path/'report/game-99.json', {'not_a_root_game': True})
    return tmp_path


def test_collection_has_all_required_hashes_only_state_keys_and_exact_counts(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    result = data.collect_exclusions(root)
    assert result == data.collect_exclusions(root)
    assert set(result) == {'states', 'files', 'counts'}
    assert result['states'] == sorted(set(result['states']))
    assert result['counts']['dataset_rows'] == 5 and result['counts']['mate_selection_rows'] == 3
    assert result['counts']['old_puzzle_solver_positions'] == 2
    assert result['counts']['arena_games'] == 3 and result['counts']['arena_positions'] == 9
    assert result['counts']['observed_positions'] == 19 and result['counts']['source_files'] == 10
    assert len(result['files']) == len(result['counts']['by_file']) == 10
    for path, digest in result['files'].items():
        assert digest == hashlib.sha256((root/path).read_bytes()).hexdigest()
        assert path not in {'summary.json', 'completed.json'} and '/report/' not in path
    serialized = json.dumps(result)
    assert 'secret_teacher_label' not in serialized and 'heldout_label_not_an_input' not in serialized
    assert 'target_uci' not in serialized and 'score_cp' not in serialized
    assert all(len(key.split()) == 4 for key in result['states'])


def test_replay_preserves_literal_ep_in_actual_model_input_not_only_rendered_fen(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    result = data.collect_exclusions(root)
    board = chess.Board()
    board.push_uci('e2e4')
    assert board.fen().split()[3] == '-' and state_key(board).split()[3] == 'e3'
    assert state_key(board) in result['states']
    board.push_uci('e7e5')
    assert state_key(board) in result['states']
    board = chess.Board()
    board.push_uci('d2d4')
    assert state_key(board).split()[3] == 'd3' and state_key(board) in result['states']


def stored_transition(board, token):
    successor = board.copy(stack=False)
    successor.push_uci(token)
    return {'uci': token, 'next_fen': successor.fen(en_passant='fen')}


def test_all_prior_stored_behavior_and_auxiliary_targets_are_excluded_and_counted(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    for relative, _, _ in data.DATA_SOURCES[2:]:
        path = root/relative
        row = json.loads(path.read_text())
        board = chess.Board(row['fen'])
        behavior = stored_transition(board, 'e2e4')
        row.update(behavior_uci=behavior['uci'], next_fen=behavior['next_fen'],
                   aux_transitions=[stored_transition(board, token) for token in ('b1c3', 'a2a4')])
        write(path, row)
    original_path = root/data.DATA_SOURCES[0][0]
    original = json.loads(original_path.read_text())
    behavior = stored_transition(chess.Board(original['fen']), 'g2g4')
    original.update(behavior_uci=behavior['uci'], next_fen=behavior['next_fen'])
    write(original_path, original)
    result = data.collect_exclusions(root)
    assert result['counts']['dataset_input_positions'] == 5
    assert result['counts']['dataset_behavior_successor_positions'] == 4
    assert result['counts']['dataset_auxiliary_successor_positions'] == 6
    assert result['counts']['observed_positions'] == 29
    for token in ('e2e4', 'b1c3', 'a2a4', 'g2g4'):
        board = chess.Board()
        board.push_uci(token)
        assert state_key(board) in result['states']
    for relative, _, _ in data.DATA_SOURCES[2:]:
        counts = result['counts']['by_file'][relative]
        assert counts['positions'] == 4 and counts['input_positions'] == 1
        assert counts['behavior_successor_positions'] == 1 and counts['auxiliary_successor_positions'] == 2


@pytest.mark.parametrize('corruption', ['missing_behavior', 'illegal_auxiliary', 'wrong_successor',
                                     'missing_ep', 'duplicate_action', 'empty_auxiliary', 'null_action'])
def test_stored_successor_exclusions_require_exact_legal_action_and_board(tmp_path, monkeypatch, corruption):
    root = source_fixture(tmp_path, monkeypatch)
    path = root/data.DATA_SOURCES[2][0]
    row = json.loads(path.read_text())
    transition = stored_transition(chess.Board(row['fen']), 'a2a4')
    if corruption == 'missing_behavior':
        row['next_fen'] = transition['next_fen']
    else:
        row['aux_transitions'] = [transition]
        if corruption == 'illegal_auxiliary':
            transition['uci'] = 'a1a8'
        elif corruption == 'wrong_successor':
            transition['next_fen'] = chess.STARTING_FEN
        elif corruption == 'missing_ep':
            transition['next_fen'] = chess.Board(transition['next_fen']).fen()
        elif corruption == 'duplicate_action':
            row['aux_transitions'].append(transition.copy())
        elif corruption == 'empty_auxiliary':
            row['aux_transitions'] = []
        elif corruption == 'null_action':
            transition['uci'] = '0000'
    write(path, row)
    with pytest.raises(ValueError):
        data.collect_exclusions(root)


@pytest.mark.parametrize('corruption', ['missing_dataset', 'missing_game', 'extra_game', 'bad_fen',
                                     'wrong_split', 'bad_state_key', 'incomplete_jsonl', 'missing_mate_split',
                                     'duplicate_mate_id', 'bad_puzzle_setup', 'bad_solver_fen', 'illegal_move',
                                     'null_move', 'bad_successor', 'unplayed_in_moves', 'bad_attempts'])
def test_missing_or_malformed_inputs_never_silently_reduce_exclusions(tmp_path, monkeypatch, corruption):
    root = source_fixture(tmp_path, monkeypatch)
    row_path = root/data.DATA_SOURCES[0][0]
    game_path = root/data.ARENA_SOURCES[0][0]/(data.ARENA_SOURCES[0][1]+'01.json')
    if corruption == 'missing_dataset':
        row_path.unlink()
    elif corruption == 'missing_game':
        game_path.unlink()
    elif corruption == 'extra_game':
        write(game_path.parent/(data.ARENA_SOURCES[0][1]+'99.json'), game())
    elif corruption in {'bad_fen', 'wrong_split', 'bad_state_key', 'incomplete_jsonl'}:
        row = json.loads(row_path.read_text())
        if corruption == 'bad_fen':
            row['fen'] = '8/8/8/8/8/8/8/8 w - - 0 1'
        elif corruption == 'wrong_split':
            row['split'] = 'wrong'
        elif corruption == 'bad_state_key':
            row['state_key'] = 'wrong'
        write(row_path, row)
        if corruption == 'incomplete_jsonl':
            row_path.write_text(row_path.read_text().rstrip('\n'))
    elif corruption in {'missing_mate_split', 'duplicate_mate_id'}:
        path = root/data.MATE_SELECTION
        value = json.loads(path.read_text())
        if corruption == 'missing_mate_split':
            value['splits'].pop('confirm')
        else:
            value['splits']['confirm'][0]['id'] = value['splits']['dev'][0]['id']
        write(path, value)
    elif corruption in {'bad_puzzle_setup', 'bad_solver_fen'}:
        path = root/data.PUZZLE_PLAN
        value = json.loads(path.read_text())
        value['puzzles']['positions'][0]['setup_uci' if corruption == 'bad_puzzle_setup' else 'solver_fen'] = (
            '0000' if corruption == 'bad_puzzle_setup' else chess.STARTING_FEN)
        write(path, value)
    else:
        value = json.loads(game_path.read_text())
        if corruption in {'illegal_move', 'null_move'}:
            value['moves'][0]['move_uci'] = 'a1a8' if corruption == 'illegal_move' else '0000'
        elif corruption == 'bad_successor':
            value['moves'][0]['fen_after'] = chess.STARTING_FEN
        elif corruption == 'unplayed_in_moves':
            value['moves'][0]['played'] = False
        else:
            value['attempts'].pop()
        write(game_path, value)
    with pytest.raises(ValueError):
        data.collect_exclusions(root)


def test_nonregular_source_and_duplicate_json_field_rejected(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    path = root/data.DATA_SOURCES[0][0]
    original = path.read_bytes()
    path.write_text('{"id":"first","id":"second"}\n')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        data.collect_exclusions(root)
    path.unlink()
    replacement = root/'replacement.jsonl'
    replacement.write_bytes(original)
    path.symlink_to(replacement)
    with pytest.raises(ValueError, match='nonregular'):
        data.collect_exclusions(root)


def test_fresh_config_is_bounded_and_contains_no_training_split():
    assert data.FRESH_CONFIG['teacher_nodes'] == 2000
    assert data.FRESH_CONFIG['teacher_threads'] == 1 and data.FRESH_CONFIG['teacher_hash_mb'] == 16
    assert data.FRESH_CONFIG['aux_actions'] == 4
    assert [(row['name'], row['examples'], row['game_cap'], row['max_plies'],
             row['random_move_probability'], row['seed_base']) for row in data.FRESH_CONFIG['splits']] == [
        ('dev', 2048, 200, 64, .5, 102000000), ('shift', 2048, 200, 96, .1, 103000000)]


def test_wrapper_uses_original_generator_costs_exclusions_and_validator_with_stub_engine(tmp_path, monkeypatch):
    config = copy.deepcopy(data.FRESH_CONFIG)
    for split in config['splits']:
        split.update(examples=4, game_cap=4, max_plies=8, random_move_probability=1.)
    monkeypatch.setattr(data, 'FRESH_CONFIG', config)

    class Engine:
        def __init__(self):
            self.id = {'name': 'Stockfish 19 synthetic test'}
            self.calls = 0
            self.configured = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def configure(self, values):
            self.configured.append(values)

        def analyse(self, board, limit, info):
            assert not board.move_stack and limit.nodes == 2000
            assert info & chess.engine.INFO_SCORE
            self.calls += 1
            return {'pv': [min(board.legal_moves, key=lambda move: move.uci())],
                    'score': chess.engine.PovScore(chess.engine.Cp(100), board.turn), 'nodes': 2001}

    engine = Engine()
    monkeypatch.setattr(data.chess.engine.SimpleEngine, 'popen_uci', lambda _: engine)
    excluded = {state_key(chess.Board().mirror())}
    receipt = data.generate_fresh(tmp_path/'fresh', 'synthetic-engine', excluded)
    rows = data.validate_fresh(tmp_path/'fresh', excluded)
    assert receipt['counts'] == {'dev': 4, 'shift': 4} and receipt['unique_states'] == 8
    assert receipt['teacher_calls'] == engine.calls > 8
    assert receipt['requested_nodes'] == engine.calls*2000
    assert receipt['reported_nodes'] == engine.calls*2001 and receipt['unknown_node_calls'] == 0
    assert engine.configured[0] == {'Threads': 1, 'Hash': 16}
    assert engine.configured[1:] == [{'Clear Hash': None}]*engine.calls
    assert all(state_key(chess.Board(row['fen'])) != state_key(chess.Board())
               for split in rows.values() for row in split)
    with pytest.raises(FileExistsError):
        data.generate_fresh(tmp_path/'fresh', 'synthetic-engine', excluded)
