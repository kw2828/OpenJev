import io
import json

import pytest

pytest.importorskip('chess', reason='Install python-chess==1.999 for the chess arena')
import chess
import chess.pgn

from openjev.research.chess_arena import (
    MAX_LEGAL_CANDIDATES,
    chess_record,
    greedy_material_policy,
    play_game,
    random_policy,
)


def timer(*values):
    return iter(values).__next__


def scripted(*moves):
    sequence = iter(moves)
    return lambda board: {'choice': next(sequence)}


def forbidden(board):
    raise AssertionError('A terminal/capped position must not call a policy')


def test_record_roundtrips_all_twenty_start_moves_without_application_cap():
    board = chess.Board()
    before = board.fen()
    record = chess_record(board)
    assert set(record) == {'context', 'question', 'candidates'}
    ids = [candidate['id'] for candidate in record['candidates']]
    assert ids == sorted(move.uci() for move in board.legal_moves)
    assert len(ids) == 20 and MAX_LEGAL_CANDIDATES == 218
    for candidate in record['candidates']:
        move = board.parse_uci(candidate['id'])
        san, origin_target, uci = candidate['description'].split('; ')
        assert board.parse_san(san) == move
        assert origin_target == f'{chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}'
        assert uci == f'UCI {move.uci()}'
    assert before in record['context']
    assert '8 r n b q k b n r' in record['context']
    assert 'a b c d e f g h' in record['context']
    assert board.fen() == before and not board.move_stack


@pytest.mark.parametrize(('fen', 'moves'), [
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', {'e1g1', 'e1c1'}),
    ('7k/P7/8/8/8/8/8/7K w - - 0 1', {'a7a8q', 'a7a8r', 'a7a8b', 'a7a8n'}),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', {'e5d6'}),
])
def test_special_moves_are_complete_native_legal_candidates(fen, moves):
    board = chess.Board(fen)
    candidates = {candidate['id']: candidate['description'] for candidate in chess_record(board)['candidates']}
    assert moves <= candidates.keys()
    for uci in moves:
        assert board.parse_uci(uci) in board.legal_moves
        assert board.parse_san(candidates[uci].split('; ')[0]).uci() == uci


def test_pinned_piece_pseudo_legal_move_is_excluded():
    board = chess.Board('k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1')
    illegal = chess.Move.from_uci('e2d2')
    assert illegal in board.pseudo_legal_moves and illegal not in board.legal_moves
    assert illegal.uci() not in {candidate['id'] for candidate in chess_record(board)['candidates']}


def test_forced_single_move_and_checkmate_have_native_candidate_counts():
    forced = chess.Board('7k/7Q/5K2/8/8/8/8/8 b - - 0 1')
    assert [row['id'] for row in chess_record(forced)['candidates']] == ['h8h7']
    mate = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
    assert chess_record(mate)['candidates'] == []


def test_context_uses_only_public_chess_state_and_explicit_history_bound():
    board = chess.Board()
    for uci in ['e2e4', 'e7e5', 'g1f3', 'b8c6']:
        board.push_uci(uci)
    clean = chess_record(board, history_plies=2)
    board.engine_score = 'private-centipawn-score-982'
    board.target_move = 'private-tactic-label'
    assert chess_record(board, history_plies=2) == clean
    assert clean['context'].endswith('g1f3 b8c6')
    assert 'e2e4' not in clean['context']
    assert chess_record(board, history_plies=None)['context'].endswith('e2e4 e7e5 g1f3 b8c6')
    assert chess_record(board, history_plies=0)['context'].endswith('(none)')
    assert 'private-' not in json.dumps(clean)
    with pytest.raises(ValueError, match='history_plies'):
        chess_record(board, -1)


def test_checkmate_takes_precedence_over_ply_cap_and_pgn_replays_exactly():
    game = play_game(
        scripted('f2f3', 'g2g4'), scripted('e7e5', 'd8h4'), max_plies=4,
        white_name='Candidate model', black_name='Opponent',
        timer=timer(0, 1, 1, 3, 3, 6, 6, 10),
    )
    assert game.result == '0-1' and game.termination == 'checkmate' and game.status == 'completed'
    assert game.clocks == {'white': 296., 'black': 294.}
    assert len(game.attempts) == len(game.moves) == 4
    assert game.moves[-1]['san'] == 'Qh4#'
    assert game.moves[-1]['latency_ms'] == 4000
    loaded = chess.pgn.read_game(io.StringIO(game.to_pgn()))
    assert not loaded.errors and loaded.headers['Result'] == game.result
    assert loaded.headers['TimeControl'] == '300+0'
    assert loaded.headers['White'] == 'Candidate model'
    assert loaded.end().board().fen() == game.final_fen
    assert loaded.end().clock() == 294.
    assert loaded.end().emt() == 4.
    assert json.loads(json.dumps(game.to_dict()))['result'] == '0-1'


@pytest.mark.parametrize(('fen', 'termination'), [
    ('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', 'checkmate'),
    ('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', 'stalemate'),
    ('7k/8/6K1/8/8/8/8/8 w - - 0 1', 'insufficient_material'),
    ('7k/8/6K1/8/8/8/8/R7 w - - 150 100', 'seventyfive_moves'),
])
def test_native_terminal_states_make_no_policy_call(fen, termination):
    game = play_game(forbidden, forbidden, initial_fen=fen, max_plies=0)
    assert game.termination == termination and game.status == 'completed'
    assert game.attempts == game.moves == []
    assert game.final_fen == chess.Board(fen).fen()


def test_draw_claims_are_explicit_and_not_confused_with_automatic_draws():
    fen = '7k/8/6K1/8/8/8/8/R7 w - - 100 70'
    no_claim = play_game(forbidden, forbidden, initial_fen=fen, max_plies=0)
    claim = play_game(forbidden, forbidden, initial_fen=fen, max_plies=0, claim_draw=True)
    assert no_claim.result == '*' and no_claim.termination == 'max_plies'
    assert claim.result == '1/2-1/2' and claim.termination == 'fifty_moves'
    assert 'automatic claims enabled' in claim.to_pgn()


def test_fivefold_repetition_uses_real_history_and_native_automatic_draw():
    game = play_game(
        scripted(*(['g1f3', 'f3g1']*4)), scripted(*(['g8f6', 'f6g8']*4)),
        max_plies=16, timer=lambda: 0.,
    )
    assert game.result == '1/2-1/2' and game.termination == 'fivefold_repetition'
    assert len(game.moves) == 16


def test_clock_is_cumulative_per_side_and_timeout_discards_proposed_move():
    game = play_game(
        scripted('e2e4', 'g1f3'), scripted('e7e5'), clock_seconds=1.,
        timer=timer(0., .4, .4, .9, .9, 1.55),
    )
    assert game.result == '0-1' and game.termination == 'timeout'
    assert game.clocks['white'] == 0 and game.clocks['black'] == pytest.approx(.5)
    assert len(game.moves) == 2 and len(game.attempts) == 3
    failed = game.attempts[-1]
    assert failed['move_uci'] == 'g1f3' and not failed['played']
    assert failed['fen_before'] == failed['fen_after'] == game.final_fen
    assert failed['disposition'] == 'discarded_timeout'
    assert len(list(chess.pgn.read_game(io.StringIO(game.to_pgn())).mainline_moves())) == 2


def test_exact_clock_boundary_flags_and_opponent_insufficient_material_draws():
    game = play_game(
        scripted('a1a2'), forbidden, initial_fen='7k/8/6K1/8/8/8/8/R7 w - - 0 1',
        clock_seconds=1, timer=timer(0., 1.),
    )
    assert game.result == '1/2-1/2' and game.termination == 'timeout_insufficient_material'
    assert not game.moves


@pytest.mark.parametrize('response', [{'choice': 'e2e5'}, {'choice': '0000'}, {'choice': 'bogus'}])
def test_invalid_choice_is_a_visible_failed_game_with_no_fallback(response):
    game = play_game(lambda board: response, forbidden, timer=timer(0, .2))
    assert game.result == '*' and game.status == 'failed' and game.termination == 'invalid_choice'
    assert game.moves == [] and game.final_fen == chess.STARTING_FEN
    assert game.attempts[0]['policy'] == response
    assert game.clocks['white'] == pytest.approx(299.8)


@pytest.mark.parametrize('response', [
    'e2e4', {'choice': 1}, {'choice': 'e2e4', 'probabilities': {'e2e4': 1.}},
    {'choice': 'e2e4', 'latency_ms': float('nan')},
])
def test_invalid_response_is_unscored_and_retained_as_failure(response):
    game = play_game(lambda board: response, forbidden, timer=timer(0, .1))
    assert game.result == '*' and game.status == 'failed' and game.termination == 'invalid_response'
    assert 'error' in game.attempts[0] and not game.moves


def test_misnormalized_probabilities_do_not_get_silently_renormalized():
    probabilities = {move.uci(): .1 for move in chess.Board().legal_moves}
    response = {'choice': 'e2e4', 'probabilities': probabilities, 'model': 'example'}
    game = play_game(lambda board: response, forbidden, timer=timer(0, .1))
    assert game.termination == 'invalid_response' and game.status == 'failed'
    assert game.attempts[0]['policy'] == response
    assert game.attempts[0]['move_uci'] == 'e2e4' and not game.attempts[0]['played']


def test_policy_exception_and_timeout_precedence_are_recorded():
    def broken(board):
        raise RuntimeError('Test failure')

    failed = play_game(broken, forbidden, timer=timer(0, .1))
    assert failed.termination == 'policy_error' and failed.result == '*'
    assert failed.attempts[0]['error'] == {'type': 'RuntimeError', 'message': 'Test failure'}
    flagged = play_game(broken, forbidden, clock_seconds=1, timer=timer(0, 2))
    assert flagged.termination == 'timeout' and flagged.result == '0-1'
    assert flagged.attempts[0]['error']['type'] == 'RuntimeError'


def test_ply_cap_is_unfinished_and_policy_cannot_mutate_actual_board():
    def mutating(board):
        board.clear()
        return {'choice': 'e2e4', 'metadata': {'test': True}}

    game = play_game(mutating, forbidden, max_plies=1, timer=timer(0, .25))
    expected = chess.Board()
    expected.push_uci('e2e4')
    assert game.final_fen == expected.fen()
    assert game.result == '*' and game.status == 'unfinished' and game.termination == 'max_plies'
    assert game.moves[0]['policy']['metadata'] == {'test': True}


def test_pgn_preserves_nonstandard_start_and_special_move():
    fen = '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'
    game = play_game(scripted('e5d6'), forbidden, initial_fen=fen, max_plies=1, timer=timer(0, .1))
    loaded = chess.pgn.read_game(io.StringIO(game.to_pgn()))
    assert loaded.headers['SetUp'] == '1' and loaded.headers['FEN'] == fen
    assert loaded.end().board().fen() == game.final_fen
    assert loaded.end().board().piece_at(chess.D5) is None


def test_random_policy_is_seeded_and_declares_actual_uniform_sampling():
    a, b = random_policy(101), random_policy(101)
    board = chess.Board()
    choices = []
    for _ in range(10):
        first, second = a(board), b(board)
        assert first == second
        assert first['choice'] in first['probabilities']
        assert set(first['probabilities']) == {move.uci() for move in board.legal_moves}
        assert sum(first['probabilities'].values()) == pytest.approx(1.)
        choices.append(first['choice'])
    assert len(set(choices)) > 1


def test_material_heuristic_accounts_for_native_capture_and_promotion_without_mutation():
    policy = greedy_material_policy()
    capture = chess.Board('7k/8/6K1/8/8/8/q7/R7 w - - 0 1')
    before = capture.fen()
    assert policy(capture)['choice'] == 'a1a2'
    assert capture.fen() == before and not capture.move_stack
    promotion = chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
    selected = policy(promotion)
    assert selected['choice'] == 'a7a8q'
    assert selected['policy_kind'] == 'one_ply_material_heuristic'
    assert 'probabilities' not in selected


@pytest.mark.parametrize('clock_seconds', [0, -1, float('nan'), float('inf')])
def test_invalid_clock_rejected_before_play(clock_seconds):
    with pytest.raises(ValueError, match='clock_seconds'):
        play_game(forbidden, forbidden, clock_seconds=clock_seconds)


def test_invalid_or_variant_boards_are_rejected():
    with pytest.raises(ValueError, match='standard-chess'):
        chess_record(chess.Board(None))
    with pytest.raises(ValueError, match='standard-chess'):
        chess_record(chess.Board(chess960=True))
