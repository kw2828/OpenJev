import chess

from openjev.research.chess_lfm import board_tokens


def test_published_board_order_and_state_slots():
    tokens = board_tokens(chess.Board())
    assert len(tokens) == 80
    assert tokens[:5] == ['<|pos|>', '<c:r>', '<c:n>', '<c:b>', '<c:q>']
    assert tokens[65:71] == ['<stm:w>', '<cast:KQkq>', '<ep:->', '<hm:0>', '<rep:0>', '<|hist|>']
    assert tokens[71:79] == ['<m:0000>']*8
    assert tokens[-1] == '<|eval|>'


def test_history_and_repetition_survive_moves():
    b = chess.Board()
    for uci in ['g1f3','g8f6','f3g1','f6g8']:
        b.push_uci(uci)
    t = board_tokens(b)
    assert '<rep:1>' in t
    assert t[75:79] == ['<m:g1f3>','<m:g8f6>','<m:f3g1>','<m:f6g8>']


def test_en_passant_and_black_not_mirrored():
    b=chess.Board('rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3')
    assert '<ep:d>' in board_tokens(b)
    b.push_uci('e5d6')
    assert '<stm:b>' in board_tokens(b)
