import copy
import json
import math

import pytest

pytest.importorskip('chess')
import chess
import chess.engine

from openjev.research.chess_spatial_data import (
    DEFAULT_CONFIG,
    auxiliary_transitions,
    generate_data,
    sha,
    state_key,
    stockfish_label,
    symmetry_key,
    validate_data,
)


def config():
    result = copy.deepcopy(DEFAULT_CONFIG)
    result['splits'] = [
        {'name': name, 'examples': examples, 'game_cap': 8, 'max_plies': 8,
         'random_move_probability': 1., 'seed_base': 1700+100*index}
        for index, (name, examples) in enumerate((('train', 12), ('dev', 6), ('shift', 4)))
    ]
    return result


def fake_label(board):
    return {'target_uci': min(move.uci() for move in board.legal_moves), 'score_cp': 100, 'mate': None,
            'target_value': math.tanh(100/600), 'requested_nodes': 2000, 'reported_nodes': 2001,
            'wall_seconds': .001}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disjoint_games_states_and_mirrored_exclusions_with_every_call_accounted(tmp_path):
    specification = config()
    excluded = {state_key(chess.Board().mirror())}
    calls = []

    def teacher(board):
        calls.append(board.fen(en_passant='fen'))
        return fake_label(board)

    receipt = generate_data(tmp_path/'data', teacher, specification, excluded)
    rows = validate_data(tmp_path/'data', specification, excluded)
    assert receipt['counts'] == {'train': 12, 'dev': 6, 'shift': 4}
    assert receipt['unique_states'] == 22
    assert receipt['excluded_states'] == receipt['excluded_symmetry_classes'] == 1
    assert receipt['reserved_state_keys'] == 46
    assert receipt['teacher_calls'] == receipt['successful_teacher_calls'] == len(calls)
    assert receipt['teacher_calls'] > receipt['unique_states']
    assert receipt['reported_nodes'] == receipt['teacher_calls']*2001
    assert receipt['requested_nodes'] == receipt['teacher_calls']*2000
    assert receipt['unknown_node_calls'] == 0
    assert receipt['teacher_wall_seconds'] == pytest.approx(len(calls)*.001)
    assert receipt['rollout_only_teacher_calls'] == len(calls)-22
    classes, game_sets = set(), []
    for name, split_rows in rows.items():
        game_sets.append({row['game_id'] for row in split_rows})
        for row in split_rows:
            board = chess.Board(row['fen'])
            key = symmetry_key(board)
            assert key not in classes and key != symmetry_key(chess.Board())
            classes.add(key)
            assert row['split'] == name
            assert not {'san', 'is_check', 'is_mate', 'reported_nodes', 'wall_seconds'} & row.keys()
            assert len(row['aux_transitions']) == min(4, board.legal_moves.count())
            assert len({entry['uci'] for entry in row['aux_transitions']}) == len(row['aux_transitions'])
            for entry in row['aux_transitions']+[
                    {'uci': row['behavior_uci'], 'next_fen': row['next_fen']}]:
                successor = board.copy(stack=False)
                successor.push_uci(entry['uci'])
                assert successor.fen(en_passant='fen') == entry['next_fen']
    assert all(left.isdisjoint(right) for i, left in enumerate(game_sets) for right in game_sets[i+1:])
    analyses = read_jsonl(tmp_path/'data/analyses.jsonl')
    assert [row['fen'] for row in analyses] == calls
    assert sum(row['labelled_position'] for row in analyses) == 22
    assert any(row['excluded_position'] for row in analyses)


def test_seed_reproduces_rows_and_games_but_elapsed_call_timing_is_not_claimed_deterministic(tmp_path):
    specification = config()
    first = generate_data(tmp_path/'one', fake_label, specification, set())
    second = generate_data(tmp_path/'two', fake_label, specification, set())
    for name in ('train.jsonl', 'dev.jsonl', 'shift.jsonl', 'games.jsonl', 'excluded-states.json'):
        assert first['files'][name] == second['files'][name]
    assert first['teacher_calls'] == second['teacher_calls']


def test_auxiliary_sampling_is_independent_of_behavior_rng_and_covers_special_moves():
    starting = chess.Board()
    assert auxiliary_transitions(starting, 123, 7) == auxiliary_transitions(starting, 123, 7)
    assert auxiliary_transitions(starting, 123, 7) != auxiliary_transitions(starting, 124, 7)
    assert starting.fen() == chess.STARTING_FEN
    for fen in ('7k/8/6K1/8/8/8/8/8 b - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2',
                '7k/P7/8/8/8/8/8/7K w - - 0 1'):
        board = chess.Board(fen)
        seen = set()
        for seed in range(100):
            entries = auxiliary_transitions(board, seed, 0)
            assert len(entries) == min(4, board.legal_moves.count())
            for entry in entries:
                seen.add(entry['uci'])
                successor = board.copy()
                successor.push_uci(entry['uci'])
                assert successor.fen(en_passant='fen') == entry['next_fen']
        assert seen == {move.uci() for move in board.legal_moves}


def test_game_cap_failure_preserves_all_calls_and_does_not_retry_or_extend(tmp_path):
    specification = config()
    specification['splits'] = [{**specification['splits'][0], 'examples': 3, 'game_cap': 2, 'max_plies': 1}]
    out = tmp_path/'insufficient'
    with pytest.raises(ValueError, match='fixed game cap'):
        generate_data(out, fake_label, specification, {state_key(chess.Board())})
    failure = json.loads((out/'failed.json').read_text())
    assert failure['counts'] == {'train': 0}
    assert failure['teacher_calls'] == failure['successful_teacher_calls'] == failure['generated_games'] == 2
    assert failure['requested_nodes'] == 4000 and failure['reported_nodes'] == 4002
    assert len(read_jsonl(out/'analyses.jsonl')) == len(read_jsonl(out/'games.jsonl')) == 2
    assert not (out/'completed.json').exists()
    with pytest.raises(FileExistsError):
        generate_data(out, fake_label, specification, set())


def test_teacher_exception_retains_failed_attempt_and_unknown_nodes_without_retry(tmp_path):
    calls = 0

    def teacher(board):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('test teacher failure')
        return fake_label(board)

    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='test teacher failure'):
        generate_data(out, teacher, config(), set())
    failure = json.loads((out/'failed.json').read_text())
    analyses = read_jsonl(out/'analyses.jsonl')
    assert calls == failure['teacher_calls'] == len(analyses) == 2
    assert failure['successful_teacher_calls'] == failure['known_node_calls'] == 1
    assert failure['unknown_node_calls'] == 1
    assert failure['requested_nodes'] == 4000 and failure['reported_nodes'] == 2001
    assert analyses[1]['status'] == 'failed' and analyses[1]['reported_nodes'] is None
    assert read_jsonl(out/'games.jsonl')[0]['stop'] == 'failed'
    assert failure['counts']['train'] == 1


def test_invalid_teacher_return_is_accounted_and_labeler_cannot_mutate_rollout(tmp_path):
    def invalid(board):
        return {**fake_label(board), 'target_uci': 'a1a8'}

    out = tmp_path/'invalid'
    with pytest.raises(ValueError, match='illegal'):
        generate_data(out, invalid, config(), set())
    failure = json.loads((out/'failed.json').read_text())
    assert failure['teacher_calls'] == failure['known_node_calls'] == 1
    assert failure['successful_teacher_calls'] == 0 and failure['reported_nodes'] == 2001

    def mutation(board):
        result = fake_label(board)
        board.push_uci(result['target_uci'])
        return result

    generate_data(tmp_path/'mutation', mutation, config(), set())
    validate_data(tmp_path/'mutation', config(), set())


def test_stockfish_label_clears_hash_uses_fen_only_and_preserves_mate_and_costs():
    class Engine:
        def __init__(self):
            self.cleared = False

        def configure(self, options):
            assert options == {'Clear Hash': None}
            self.cleared = True

        def analyse(self, board, limit, info):
            assert self.cleared and not board.move_stack
            assert limit.nodes == 2000 and board.turn == chess.BLACK
            assert info & chess.engine.INFO_SCORE
            return {'pv': [next(iter(board.legal_moves))],
                    'score': chess.engine.PovScore(chess.engine.Mate(-3), chess.BLACK), 'nodes': 2007}

    board = chess.Board()
    board.push_uci('e2e4')
    label = stockfish_label(Engine(), board, config())
    assert label['mate'] == -3 and label['score_cp'] == -9997
    assert label['target_value'] == pytest.approx(math.tanh(-9997/600))
    assert label['requested_nodes'] == 2000 and label['reported_nodes'] == 2007
    assert label['wall_seconds'] >= 0 and len(board.move_stack) == 1


def test_validator_rejects_changed_files_configuration_exclusions_and_successors(tmp_path):
    specification = config()
    out = tmp_path/'data'
    generate_data(out, fake_label, specification, set())
    with pytest.raises(ValueError, match='configuration'):
        validate_data(out, {**specification, 'teacher_nodes': 4000}, set())
    with pytest.raises(ValueError, match='exclusion'):
        validate_data(out, specification, {state_key(chess.Board())})
    rows = read_jsonl(out/'train.jsonl')
    rows[0]['next_fen'] = rows[0]['fen']
    (out/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='hash'):
        validate_data(out, specification, set())
    receipt = json.loads((out/'completed.json').read_text())
    receipt['files']['train.jsonl'] = sha(out/'train.jsonl')
    (out/'completed.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='successor'):
        validate_data(out, specification, set())


def test_symmetry_key_retains_ep_and_castling_but_ignores_counters():
    board = chess.Board('r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 2')
    assert symmetry_key(board) == symmetry_key(board.mirror())
    changed = board.copy()
    changed.halfmove_clock = 77
    changed.fullmove_number = 34
    assert symmetry_key(board) == symmetry_key(changed)
    changed.ep_square = None
    assert symmetry_key(board) != symmetry_key(changed)
    changed = board.copy()
    changed.castling_rights = chess.BB_EMPTY
    assert symmetry_key(board) != symmetry_key(changed)
