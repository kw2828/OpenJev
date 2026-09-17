"""Inference mechanics on synthetic positions and untrained spatial models."""

import math

import chess
import numpy as np
import pytest
import torch

from openjev.research import chess_compute as compute
from openjev.research.chess_spatial import MODES, SpatialChess, encode_board, encode_candidates


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return SpatialChess('recurrent', 17, width=4)


def policy(board, order=None):
    names = sorted(move.uci() for move in board.legal_moves)
    order = names if order is None else list(order)+[name for name in names if name not in order]
    weights = {name: len(order)-index for index, name in enumerate(order)}
    total = sum(weights.values())
    return {'choice': order[0], 'probabilities': {name: weights[name]/total for name in names},
            'value': .125, 'depth': 4}


def nonterminal_successors(board):
    result = []
    for move in sorted(board.legal_moves, key=lambda move: move.uci()):
        after = board.copy(stack=True)
        after.push(move)
        if after.outcome(claim_draw=False) is None:
            result.append(after)
    return result


@pytest.mark.parametrize('mode', MODES)
def test_value_only_matches_full_forward_and_omits_policy_aux_and_candidate_modules(mode):
    model = SpatialChess(mode, 17, width=4)
    white = chess.Board()
    black = chess.Board()
    black.push_uci('e2e4')
    boards = [white, black]
    expected = []
    for board in boards:
        ids, candidates = encode_candidates(board)
        with torch.no_grad():
            _, values, _ = model(torch.from_numpy(encode_board(board))[None],
                                 torch.from_numpy(candidates)[None],
                                 torch.ones((1, len(ids)), dtype=torch.bool), depth=3)
            expected.append(float(values[0]))
    calls = []

    def forbidden(*_args):
        raise AssertionError('Value-only path must omit policy/candidate/auxiliary modules')

    hooks = [module.register_forward_pre_hook(forbidden) for module in
             (model.policy_head, model.promotion_embedding, model.dx_embedding,
              model.dy_embedding, model.aux_head)]
    hooks.append(model.encoder.register_forward_pre_hook(lambda _module, inputs: calls.append(inputs[0].clone())))
    try:
        actual = compute.value_boards(model, boards, depth=3)
    finally:
        for hook in hooks:
            hook.remove()
    assert actual == pytest.approx(expected, abs=1e-7)
    assert len(calls) == 1 and calls[0].shape == (2, 19, 8, 8)
    np.testing.assert_array_equal(calls[0][1].numpy(), encode_board(black))
    assert not np.array_equal(calls[0][1].numpy(), encode_board(black, chess.WHITE))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_empty_value_batch_makes_no_forward_and_terminal_boards_are_rejected(model):
    def forbidden(*_args):
        raise AssertionError('Encoder must not execute')
    handle = model.encoder.register_forward_pre_hook(forbidden)
    try:
        assert compute.value_boards(model, []) == []
        with pytest.raises(ValueError, match='nonterminal'):
            compute.value_boards(model, [chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')])
    finally:
        handle.remove()


def test_choose_depth_preserves_original_policy_and_counts(model):
    board = chess.Board()
    expected = model.choose(board, depth=2)
    result = compute.choose_depth(model, board, depth=2)
    for name in ('choice', 'probabilities', 'value', 'depth'):
        assert result[name] == expected[name]
    assert sum(result['probabilities'].values()) == pytest.approx(1.)
    assert result['compute']['policy_forward_calls'] == 1
    assert result['compute']['candidates_evaluated'] == 20
    assert result['compute']['boardvalue_evaluations'] == 0
    assert result['compute']['symbolic_transition_count'] == 0
    assert result['latency_ms'] == result['compute']['total_wall_ms'] >= 0


def test_exact_all_uses_negated_successor_values_and_no_root_policy(model, monkeypatch):
    board = chess.Board()
    observed = []

    def values(_model, boards, depth):
        observed.extend(boards)
        assert depth == 3
        return [-.8 if b.peek().uci() == 'e2e4' else .2 for b in boards]

    def forbidden(*_args, **_kwargs):
        raise AssertionError('All-successor control must not call the root policy')

    monkeypatch.setattr(compute, '_value_boards', values)
    monkeypatch.setattr(model, 'choose', forbidden)
    result = compute.exact_successor_decision(model, board, depth=3)
    assert result['choice'] == 'e2e4' and result['scores']['e2e4'] == .8
    assert result['scores']['a2a3'] == -.2 and 'probabilities' not in result
    assert list(result['scores']) == sorted(move.uci() for move in board.legal_moves)
    assert len(observed) == 20 and all(b.turn == chess.BLACK for b in observed)
    assert len({b.peek().uci() for b in observed}) == 20
    counts = result['compute']
    assert counts['policy_forward_calls'] == counts['terminal_evaluations'] == 0
    assert counts['boardvalue_evaluations'] == counts['candidates_evaluated'] == 20
    assert counts['symbolic_transition_count'] == counts['terminal_checks'] == 20
    assert counts['boardvalue_forward_calls'] == 1


MATE_IN_ONE = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'


def mates(board):
    result = []
    for move in board.legal_moves:
        after = board.copy()
        after.push(move)
        if after.is_checkmate():
            result.append(move.uci())
    return sorted(result)


def test_native_mate_wins_tie_with_rounded_tanh_one(model, monkeypatch):
    board = chess.Board(MATE_IN_ONE)
    expected = mates(board)
    assert expected and nonterminal_successors(board)
    monkeypatch.setattr(compute, '_value_boards', lambda _m, boards, _depth: [-1.]*len(boards))
    result = compute.exact_successor_decision(model, board)
    assert result['choice'] == expected[0]
    assert result['scores'][result['choice']] == 1.
    assert result['terminal_successors'][result['choice']]['termination'] == 'checkmate'
    assert any(value == 1. and move not in result['terminal_successors'] for move, value in result['scores'].items())


def test_top_k_does_not_push_or_inspect_excluded_terminal_moves(model, monkeypatch):
    board = chess.Board(MATE_IN_ONE)
    expected = mates(board)
    shortlist = [b.peek().uci() for b in nonterminal_successors(board)[:2]]
    assert len(shortlist) == 2 and not set(shortlist) & set(expected)
    response = policy(board, shortlist)
    monkeypatch.setattr(model, 'choose', lambda _board, depth: response)
    monkeypatch.setattr(compute, '_value_boards', lambda _model, boards, _depth: [0.]*len(boards))
    original_push = chess.Board.push
    pushed = []

    def push(self, move):
        pushed.append(move.uci())
        return original_push(self, move)

    monkeypatch.setattr(chess.Board, 'push', push)
    result = compute.exact_successor_decision(model, board, top_k=2)
    assert pushed == sorted(shortlist)
    assert set(result['scores']) == set(shortlist)
    assert result['choice'] == min(shortlist)
    assert result['terminal_successors'] == {}
    assert result['compute']['policy_forward_calls'] == 1
    assert result['compute']['boardvalue_evaluations'] == result['compute']['symbolic_transition_count'] == 2


def test_top_k_probability_ties_use_sorted_original_uci_and_k_clips_to_menu(model, monkeypatch):
    board = chess.Board()
    names = sorted(move.uci() for move in board.legal_moves)
    response = {'choice': names[0], 'probabilities': dict.fromkeys(reversed(names), 1/len(names)), 'value': 0.}
    monkeypatch.setattr(model, 'choose', lambda _board, depth: response)
    monkeypatch.setattr(compute, '_value_boards', lambda _model, boards, _depth: [0.]*len(boards))
    short = compute.exact_successor_decision(model, board, top_k=4)
    assert list(short['scores']) == names[:4]
    assert short['choice'] == names[0]
    full = compute.exact_successor_decision(model, board, top_k=100)
    assert len(full['scores']) == len(names) == full['compute']['candidates_evaluated']


def test_underpromotions_are_distinct_and_native_insufficient_material_gets_zero(model, monkeypatch):
    board = chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
    monkeypatch.setattr(compute, '_value_boards', lambda _model, boards, _depth: [.5]*len(boards))
    result = compute.exact_successor_decision(model, board)
    assert {'a7a8q', 'a7a8r', 'a7a8b', 'a7a8n'} <= result['scores'].keys()
    for move in ('a7a8b', 'a7a8n'):
        assert result['scores'][move] == 0.
        assert result['terminal_successors'][move]['termination'] == 'insufficient_material'
    assert result['scores']['a7a8q'] == result['scores']['a7a8r'] == -.5
    assert result['choice'] == 'a7a8b'


def test_en_passant_is_native_successor_with_both_pawns_updated(model, monkeypatch):
    board = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2')
    observed = {}

    def values(_model, boards, _depth):
        observed.update({b.peek().uci(): b for b in boards})
        return [-.7 if b.peek().uci() == 'e5d6' else 0. for b in boards]

    monkeypatch.setattr(compute, '_value_boards', values)
    result = compute.exact_successor_decision(model, board)
    assert result['choice'] == 'e5d6'
    after = observed['e5d6']
    assert after.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert after.piece_at(chess.D5) is None and after.piece_at(chess.E5) is None
    assert after.turn == chess.BLACK and after.ep_square is None


def test_real_history_preserves_fivefold_successor_and_does_not_mutate_caller(model, monkeypatch):
    board = chess.Board()
    for move in (['g1f3', 'g8f6', 'f3g1', 'f6g8']*4)[:-1]:
        board.push_uci(move)
    before, stack, root = board.fen(en_passant='fen'), list(board.move_stack), board.root().fen()
    assert board.outcome(claim_draw=False) is None
    monkeypatch.setattr(compute, '_value_boards', lambda _model, boards, _depth: [.25]*len(boards))
    result = compute.exact_successor_decision(model, board)
    assert result['scores']['f6g8'] == 0.
    assert result['terminal_successors']['f6g8']['termination'] == 'fivefold_repetition'
    assert board.fen(en_passant='fen') == before and board.move_stack == stack and board.root().fen() == root


def test_terminal_guard_finds_mate_without_policy_or_value_forward(model, monkeypatch):
    board = chess.Board(MATE_IN_ONE)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Known immediate mate must bypass network inference')

    monkeypatch.setattr(model, 'choose', forbidden)
    monkeypatch.setattr(compute, '_value_boards', forbidden)
    result = compute.terminal_guard_decision(model, board)
    assert result['guard_triggered'] and result['winning_mates'] == mates(board)
    assert result['choice'] == min(result['winning_mates'])
    assert 'probabilities' not in result
    assert result['compute']['policy_forward_calls'] == result['compute']['boardvalue_evaluations'] == 0
    assert result['compute']['symbolic_transition_count'] == board.legal_moves.count()


def test_terminal_guard_fallback_keeps_policy_exact_and_isolates_mutation(model, monkeypatch):
    board = chess.Board()
    board.push_uci('e2e4')
    before, stack = board.fen(), list(board.move_stack)
    response = policy(board)

    def mutating_policy(copy_board, depth):
        copy_board.push_uci(response['choice'])
        return response

    monkeypatch.setattr(model, 'choose', mutating_policy)
    result = compute.terminal_guard_decision(model, board)
    assert not result['guard_triggered']
    for name in ('choice', 'probabilities', 'value'):
        assert result[name] == response[name]
    assert board.fen() == before and board.move_stack == stack
    assert result['compute']['policy_forward_calls'] == 1


@pytest.mark.parametrize('top_k', [0, -1, 2.5, True, '2'])
def test_invalid_top_k_rejected(model, top_k):
    with pytest.raises(ValueError, match='top_k'):
        compute.exact_successor_decision(model, chess.Board(), top_k=top_k)


@pytest.mark.parametrize('depth', [0, -1, 1.5, True, None])
def test_invalid_depth_rejected_everywhere(model, depth):
    for function in (compute.choose_depth, compute.exact_successor_decision, compute.terminal_guard_decision):
        with pytest.raises(ValueError, match='depth'):
            function(model, chess.Board(), depth=depth)
    with pytest.raises(ValueError, match='depth'):
        compute.value_boards(model, [], depth=depth)


def test_cnn_cannot_exceed_frozen_block_depth():
    model = SpatialChess('cnn', 17, width=4)
    with pytest.raises(ValueError, match='CNN depth'):
        compute.choose_depth(model, chess.Board(), depth=5)
    with pytest.raises(ValueError, match='CNN depth'):
        compute.value_boards(model, [chess.Board()], depth=5)


@pytest.mark.parametrize('board', [None, chess.Board.empty(), chess.Board(chess960=True),
    chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'),
    chess.Board('7k/8/6K1/8/8/8/8/8 w - - 0 1')])
def test_invalid_or_terminal_inputs_are_rejected(model, board):
    for function in (compute.choose_depth, compute.exact_successor_decision, compute.terminal_guard_decision):
        with pytest.raises(ValueError):
            function(model, board, depth=4)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_timing_wraps_policy_preparation_and_device_synchronization(model, monkeypatch, device):
    board = chess.Board()
    response = policy(board)
    events = []
    times = iter([2., 2.25])

    def timer():
        events.append('timer')
        return next(times)

    def choose(_board, depth):
        events.append('policy')
        return response

    monkeypatch.setattr(compute, '_device', lambda _model: torch.device(device))
    monkeypatch.setattr(compute.time, 'perf_counter', timer)
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: events.append('sync'))
    monkeypatch.setattr(model, 'choose', choose)
    result = compute.choose_depth(model, board, depth=4)
    assert events == (['timer', 'sync', 'policy', 'sync', 'timer'] if device == 'mps'
                      else ['timer', 'policy', 'timer'])
    assert result['latency_ms'] == result['compute']['total_wall_ms'] == 250.
    assert result['compute']['device'] == device


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
def test_mps_untrained_model_value_only_and_decision_smoke():
    model = SpatialChess('predict', 17, width=4).to('mps')
    board = chess.Board()
    result = compute.choose_depth(model, board, depth=2)
    values = compute.value_boards(model, [board], depth=2)
    assert values[0] == pytest.approx(result['value'], abs=1e-6)
    assert result['compute']['device'] == 'mps:0'
    assert math.isfinite(result['latency_ms']) and result['latency_ms'] > 0


def test_rejects_malformed_policy_without_hidden_fallback(model, monkeypatch):
    result = policy(chess.Board())
    result['probabilities']['e2e4'] = math.nan
    monkeypatch.setattr(model, 'choose', lambda _board, depth: result)
    with pytest.raises(ValueError, match='Invalid spatial policy'):
        compute.exact_successor_decision(model, chess.Board(), top_k=2)
