import csv
import hashlib

import pytest

pytest.importorskip('chess')
import chess

from openjev.research.chess_mate_data import select_mates, source_game_id
from openjev.research.chess_spatial_data import state_key, symmetry_key

FIELDS = ('PuzzleId', 'FEN', 'Moves', 'Rating', 'Themes', 'GameUrl')


def puzzle(index, game=None, mirror=False):
    board = chess.Board.empty()
    board.turn = chess.BLACK
    board.set_piece_at(chess.square(index, 0), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.F3, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.B2, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    for square in (chess.F7, chess.G7, chess.H7):
        board.set_piece_at(square, chess.Piece(chess.PAWN, chess.BLACK))
    moves = 'g8h8 f3a8'
    if mirror:
        board = board.mirror()
        moves = 'g1h1 f6a1'
    assert board.is_valid()
    return {'PuzzleId': f'{index:05d}', 'FEN': board.fen(), 'Moves': moves, 'Rating': '1000',
            'Themes': 'mate mateIn1 short', 'GameUrl': f'https://lichess.org/game{index if game is None else game:04d}/black#31'}


def source(tmp_path, rows, suffix=b''):
    path = tmp_path/'puzzles.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if suffix:
        with path.open('ab') as stream:
            stream.write(suffix)
    return path


def test_solver_setup_and_all_mates_are_exact_and_selection_is_deterministic(tmp_path):
    path = source(tmp_path, [puzzle(index) for index in range(8)], b'incomplete,last,row')
    result = select_mates(path, set(), set(), sizes=(3, 2, 2), seed=7)
    assert result == select_mates(path, set(), set(), sizes=(3, 2, 2), seed=7)
    assert result['selection']['clean_candidates'] == 8
    assert result['selection']['discarded_incomplete_suffix_bytes'] == len(b'incomplete,last,row')
    assert result['selection']['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['selection']['unused_candidates'] == 1
    game_sets, state_sets = [], []
    for split, expected in (('train', 3), ('dev', 2), ('confirm', 2)):
        rows = result['splits'][split]
        assert len(rows) == expected
        game_sets.append({row['source_game_id'] for row in rows})
        state_sets.append({symmetry_key(chess.Board(row['fen'])) for row in rows})
        for row in rows:
            assert row['id'] == 'lichess-'+row['puzzle_id']
            board = chess.Board(row['fen'])
            assert board.turn == chess.WHITE and board.king(chess.BLACK) == chess.H8
            assert row['target_uci'] == 'f3a8'
            mates = []
            for move in board.legal_moves:
                child = board.copy()
                child.push(move)
                if child.is_checkmate():
                    mates.append(move.uci())
            assert row['mating_uci'] == sorted(mates)
            assert 'f3a8' in mates and 'b2b8' in mates
            assert row['source_line'] == int(row['puzzle_id'])+2
    for sets in (game_sets, state_sets):
        assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i+1:])
    assert result['selection']['selected_puzzle_ids'] != select_mates(
        path, set(), set(), sizes=(3, 2, 2), seed=8)['selection']['selected_puzzle_ids']


def test_whole_source_game_groups_stay_together_even_with_different_url_suffixes(tmp_path):
    rows = [puzzle(index, game=0 if index < 2 else index) for index in range(8)]
    rows[1]['GameUrl'] = 'https://lichess.org/game0000#42'
    result = select_mates(source(tmp_path, rows), set(), set(), sizes=(4, 2, 2), seed=17)
    members = [split for split, records in result['splits'].items()
               if any(row['source_game_id'] == 'game0000' for row in records)]
    assert len(members) == 1
    assert sum(row['source_game_id'] == 'game0000' for row in result['splits'][members[0]]) == 2
    assert result['selection']['candidate_source_games'] == 7


def test_exclusions_include_mirrored_solver_states_and_normalized_game_ids(tmp_path):
    rows = [puzzle(index) for index in range(8)]
    excluded = chess.Board(rows[1]['FEN'])
    excluded.push_uci('g8h8')
    result = select_mates(source(tmp_path, rows), {'https://lichess.org/game0000/white#3'},
                          {state_key(excluded.mirror())}, sizes=(2, 2, 2))
    selected = {row['puzzle_id'] for group in result['splits'].values() for row in group}
    assert selected == {f'{index:05d}' for index in range(2, 8)}
    assert result['selection']['rejection_counts'] == {
        'excluded_solver_state_or_mirror': 1, 'excluded_source_game': 1}


def test_mirror_duplicate_and_duplicate_puzzle_id_are_removed_before_splitting(tmp_path):
    rows = [puzzle(index) for index in range(6)]
    mirrored = {**puzzle(0, game=7, mirror=True), 'PuzzleId': '99998'}
    rows.extend((mirrored, rows[1].copy()))
    result = select_mates(source(tmp_path, rows), set(), set(), sizes=(2, 2, 2))
    assert result['selection']['rejection_counts'] == {
        'duplicate_puzzle_id': 1, 'duplicate_solver_state_or_mirror': 1}
    assert result['selection']['clean_candidates'] == 6


def test_excluded_game_blocks_duplicate_mirror_state_from_another_game(tmp_path):
    rows = [puzzle(index) for index in range(7)]
    rows.append({**puzzle(0, game=7, mirror=True), 'PuzzleId': '99998'})
    result = select_mates(source(tmp_path, rows), {'game0000'}, set(), sizes=(2, 2, 2))
    assert result['selection']['rejection_counts'] == {
        'excluded_solver_state_or_mirror': 1, 'excluded_source_game': 1}


def test_bad_labels_illegal_or_null_setup_and_nonmate_themes_are_rejected(tmp_path):
    rows = [puzzle(index) for index in range(3)]
    rows.extend([
        {**puzzle(3), 'Moves': 'g8h8 f3g3'},
        {**puzzle(4), 'Moves': 'g8a8 f3a8'},
        {**puzzle(5), 'Moves': '0000 f3a8'},
        {**puzzle(6), 'Themes': 'mateIn2'},
        {**puzzle(7), 'Moves': 'g8h8 f3a8 h8g8'},
    ])
    result = select_mates(source(tmp_path, rows), set(), set(), sizes=(1, 1, 1))
    assert result['selection']['rejection_counts'] == {
        'invalid_puzzle': 3, 'not_mateIn1': 1, 'recorded_answer_not_mate': 1}


def test_fixed_quota_shortage_and_whole_group_packing_failure_do_not_retry(tmp_path):
    rows = [puzzle(index, game=0) for index in range(4)]
    path = source(tmp_path, rows)
    with pytest.raises(ValueError, match='frozen total quota'):
        select_mates(path, set(), set(), sizes=(3, 2, 2))
    with pytest.raises(ValueError, match='whole source-game groups'):
        select_mates(path, set(), set(), sizes=(2, 1, 1))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['puzzles.csv']


def test_game_id_normalization_is_strict_and_ignores_color_and_ply():
    for value in ('game0001', 'https://lichess.org/game0001#42',
                  'https://lichess.org/game0001/black#31'):
        assert source_game_id(value) == 'game0001'
    for value in ('https://other.example/game0001', 'https://lichess.org/not-an-id', 'oops'):
        with pytest.raises(ValueError):
            source_game_id(value)
