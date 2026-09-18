import chess
import pytest

from openjev.research.chess_pin_factors import pin_witnesses, reference_witnesses, candidate_factors, changes


@pytest.mark.parametrize('fen,expected', [
    ('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1', ((0, chess.E1, chess.E2, chess.E8),)),
    ('k7/8/8/8/7b/8/5N2/4K3 w - - 0 1', ((0, chess.E1, chess.F2, chess.H4),)),
    ('k3b3/8/8/8/8/8/4N3/4K3 w - - 0 1', ()),
    ('k3r3/8/8/8/8/4B3/4N3/4K3 w - - 0 1', ()),
    ('4k3/4n3/8/8/8/8/8/K3R3 b - - 0 1', ((0, chess.E1, chess.E2, chess.E8),)),
])
def test_exact_pin_triples(fen, expected):
    board = chess.Board(fen)
    assert board.is_valid()
    assert pin_witnesses(board, board.turn) == reference_witnesses(board, board.turn) == expected


def test_pinned_piece_still_attacks_and_witness_is_not_legal_mask():
    board = chess.Board('k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1')
    assert chess.C3 in board.attacks(chess.E2)
    assert not any(m.from_square == chess.E2 for m in board.legal_moves)
    assert pin_witnesses(board, board.turn)


@pytest.mark.parametrize('fen', [
    chess.STARTING_FEN,
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    'k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1',
])
def test_native_candidates_mirrors_and_history_preservation(fen):
    board = chess.Board(fen)
    snapshot = board.fen(en_passant='fen'), list(board.move_stack)
    a = candidate_factors(board)
    assert a == candidate_factors(board, reference=True)
    assert snapshot == (board.fen(en_passant='fen'), list(board.move_stack))
    mirrored = candidate_factors(board.mirror())
    assert a['before'] == mirrored['before']
    lookup = {r['uci']: r for r in mirrored['candidates']}
    for row in a['candidates']:
        move = chess.Move.from_uci(row['uci'])
        inverse = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion).uci()
        assert row['after'] == lookup[inverse]['after']
        assert row['factors'] == lookup[inverse]['factors']


def test_retained_added_removed_and_duplicate_rejection():
    a, b, c = (0, 1, 2, 3), (1, 4, 5, 6), (0, 7, 8, 9)
    result = changes((a, b), (b, c))
    assert set(result) == {(*a, 2), (*b, 0), (*c, 1)}
    with pytest.raises(ValueError):
        changes((a, a), ())


def test_ep_double_vacancy_not_mislabeled_as_single_absolute_pin():
    board = chess.Board('k7/8/8/r4pPK/8/8/8/8 w - f6 0 1')
    assert board.is_valid()
    # g5xf6 would remove both blockers of the rook, exposing the king.
    assert not pin_witnesses(board, board.turn)
    assert not reference_witnesses(board, board.turn)
    assert chess.Move.from_uci('g5f6') not in board.legal_moves


def test_reject_invalid_variant_and_terminal_menu():
    with pytest.raises(ValueError):
        pin_witnesses(chess.Board(None), chess.WHITE)
    with pytest.raises(ValueError):
        pin_witnesses(chess.Board(chess960=True), chess.WHITE)
    with pytest.raises(ValueError):
        pin_witnesses(chess.Board(), 1)
    with pytest.raises(ValueError):
        candidate_factors(chess.Board('4k3/8/8/8/8/8/8/4K3 w - - 0 1'))


def test_nonempty_history_is_preserved():
    board = chess.Board()
    for uci in ('e2e4', 'e7e5', 'g1f3'):
        board.push_uci(uci)
    original = board.fen(en_passant='fen'), list(board.move_stack)
    assert candidate_factors(board) == candidate_factors(board, reference=True)
    assert original == (board.fen(en_passant='fen'), list(board.move_stack))
