"""Tiny independent checks; no public model or full connectome is loaded."""

import math
import struct

import chess
import numpy as np
import pytest
import torch

from openjev.research.chess_fly import (
    ACTION_INDEX,
    ACTIONS,
    FEATURES,
    ChessFlyNetwork,
    ChessFlyPolicy,
    encode_board,
    mirror_uci,
    parse_connectome,
)


def graph_bytes(rows, destinations, counts):
    return (b'CFLY'+struct.pack('<II', len(rows)-1, len(destinations))
            +np.asarray(rows, dtype='<u4').tobytes()
            +np.asarray(destinations, dtype='<u4').tobytes()
            +np.asarray(counts, dtype='<i2').tobytes())


def tiny_network():
    # Neuron 0 supplies input, 1/2 are read out, 3 is excluded optic tissue.
    raw = graph_bytes([0, 2, 4, 5, 5], [1, 1, 2, 3, 1], [99, 2, -7, 8, 3])
    graph = parse_connectome(raw, torch.log(torch.tensor([.4, .3, .2, .1, .5])))
    rng = torch.Generator().manual_seed(41)
    weights = {
        'encoder.weight': torch.randn((1, FEATURES), generator=rng)*.01,
        'encoder.bias': torch.tensor([.6]),
        'scale': torch.tensor([[1., .8, .7, 1.2], [.9, 1., 1.1, .5], [1.2, .9, .6, .8]]),
        'shift': torch.tensor([[.1, .2, .3, .1], [-.1, .2, .3, .1], [.1, .4, .2, .3]]),
        'decoder.weight': torch.tensor([[.2, .3], [-.7, .4]]),
        'decoder.bias': torch.tensor([-.1, .2]),
        'policy.weight': torch.randn((1968, 2), generator=rng)*.02,
        'policy.bias': torch.randn((1968,), generator=rng)*.02,
        'value.weight': torch.randn((64, 2), generator=rng)*.02,
        'value.bias': torch.randn((64,), generator=rng)*.02,
    }
    network = ChessFlyNetwork(graph, np.array([2, 0, 3, 1]), weights,
                              {'steps': '3', 'hidden': '2', 'alpha': '.5'})
    return network, weights


def test_action_space_contains_special_moves_and_has_exact_size():
    assert len(ACTIONS) == 1968
    assert len(set(ACTIONS)) == 1968
    assert tuple(sorted(ACTIONS)) == ACTIONS
    assert {'e1g1', 'e1c1', 'e8g8', 'e8c8', 'a7b8n', 'h2h1q'} <= set(ACTIONS)
    assert 'e2e4q' not in ACTION_INDEX
    assert 'a1b4' not in ACTION_INDEX


def test_vocabulary_covers_legal_moves_through_a_deterministic_game():
    board = chess.Board()
    rng = np.random.default_rng(192)
    for _ in range(150):
        legal = list(board.legal_moves)
        if not legal:
            break
        for move in legal:
            normalized = move.uci() if board.turn else mirror_uci(move.uci())
            assert normalized in ACTION_INDEX
        board.push(legal[int(rng.integers(len(legal)))])


def test_square_major_features_and_black_color_reversal():
    white = chess.Board()
    x = encode_board(white)
    assert x.shape == (780,) and x.dtype == torch.float32
    assert x.sum() == 36
    assert x[chess.A1*12+3] == 1  # white rook
    assert x[chess.B8*12+7] == 1  # black knight
    assert x[768:772].tolist() == [1, 1, 1, 1]
    assert torch.equal(x, encode_board(white.mirror()))
    white.push_uci('e2e4')
    # Exact FEN EP file is encoded even if the target has no legal capturer.
    assert encode_board(white)[772+4] == 1
    assert torch.equal(encode_board(white), encode_board(white.mirror()))
    assert mirror_uci('a2b1n') == 'a7b8n'
    assert mirror_uci(mirror_uci('e8c8')) == 'e8c8'


def test_sparse_transpose_preserves_learned_signs_duplicate_edges_and_order():
    raw = graph_bytes([0, 2, 3, 4], [2, 2, 0, 1], [200, -99, 0, -2])
    graph = parse_connectome(raw, torch.log(torch.tensor([2., 3., 4., 5.])))
    assert graph.crow_indices().tolist() == [0, 1, 2, 4]
    assert graph.col_indices().tolist() == [1, 2, 0, 0]
    assert graph.values().tolist() == pytest.approx([4., -5., 2., -3.])
    assert torch.sparse.mm(graph, torch.tensor([[2.], [3.], [4.]])).flatten().tolist() == pytest.approx([12, -20, -2])


@pytest.mark.parametrize('rows,destinations,counts', [
    ([1, 1, 1], [0], [1]),
    ([0, 2, 1], [0], [1]),
    ([0, 1, 1], [3], [1]),
])
def test_bad_connectome_indices_fail(rows, destinations, counts):
    with pytest.raises(ValueError):
        parse_connectome(graph_bytes(rows, destinations, counts), torch.zeros(len(destinations)))


def test_bad_connectome_bytes_and_gains_fail():
    raw = graph_bytes([0, 1, 1], [1], [1])
    for candidate in (b'', raw[:-1], b'XXXX'+raw[4:]):
        with pytest.raises(ValueError):
            parse_connectome(candidate, torch.zeros(1))
    for gains in (torch.zeros(2), torch.tensor([float('nan')]), torch.tensor([1000.])):
        with pytest.raises(ValueError):
            parse_connectome(raw, gains)


def test_settling_matches_independent_scalar_recurrence_and_resets_per_call():
    network, weights = tiny_network()
    features = torch.stack([encode_board(chess.Board()), encode_board(chess.Board().mirror())])
    logits, values = network.forward(features)
    w = {key: value.numpy().astype(np.float64) for key, value in weights.items()}
    # Independently declared edges, including both parallel 0->1 connections.
    edges = [(0, 1, .4), (0, 1, .3), (1, 2, -.2), (1, 3, .1), (2, 1, .5)]
    drive = np.zeros(4)
    drive[0] = sum(float(a)*float(b) for a, b in zip(w['encoder.weight'][0], features[0], strict=True))+.6
    state = np.zeros(4)
    for step in range(3):
        incoming = np.zeros(4)
        for source, target, gain in edges:
            incoming[target] += gain*state[source]
        state = np.array([
            .5*state[i]+.5*max(0, (incoming[i]+drive[i])*w['scale'][step, i]+w['shift'][step, i])
            for i in range(4)
        ])
    decoded = w['decoder.weight']@state[[1, 2]]+w['decoder.bias']
    hidden = np.array([.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3))) for x in decoded])
    np.testing.assert_allclose(logits[0], w['policy.weight']@hidden+w['policy.bias'], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(values[0], w['value.weight']@hidden+w['value.bias'], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(logits[0], logits[1])
    torch.testing.assert_close(network.forward(features[:1])[0][0], logits[0])
    torch.testing.assert_close(network.forward(features)[0], logits)


def test_legal_mask_and_black_mirror_keep_actual_move_ids():
    class FixedNetwork:
        steps = 5

        def forward(self, features):
            logits = torch.full((len(features), 1968), -1.)
            logits[:, ACTION_INDEX['e2e4']] = 9.
            logits[:, ACTION_INDEX['a1a8']] = 100.  # Illegal, must not enter softmax.
            return logits, torch.zeros((len(features), 64))

    policy = ChessFlyPolicy.__new__(ChessFlyPolicy)
    policy.network = FixedNetwork()
    white = chess.Board()
    black = white.mirror()
    results = policy.evaluate_boards([white, black])
    for board, result, best in zip([white, black], results, ['e2e4', 'e7e5'], strict=True):
        assert result['choice'] == best
        assert set(result['probabilities']) == {move.uci() for move in board.legal_moves}
        assert sum(result['probabilities'].values()) == pytest.approx(1)
        assert result['probabilities'][best] > .99
        assert result['win_probability'] == pytest.approx(.5)
        assert result['search_depth'] == 0
    with pytest.raises(ValueError):
        policy.evaluate_boards([])
    terminal = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
    with pytest.raises(ValueError, match='without legal moves'):
        policy(terminal)


def test_bad_model_dimensions_and_features_fail():
    network, weights = tiny_network()
    with pytest.raises(ValueError, match='weight tensor'):
        ChessFlyNetwork(network.graph, np.array([2, 0, 3, 1]),
                        {**weights, 'encoder.bias': torch.ones(2)},
                        {'steps': '3', 'hidden': '2', 'alpha': '.5'})
    for features in (torch.zeros(0, 780), torch.zeros(1, 779), torch.zeros(1, 780).double()):
        with pytest.raises(ValueError):
            network.forward(features)


def test_unpinned_assets_are_rejected_without_loading_model(tmp_path):
    (tmp_path/'meta.json').write_text('{}')
    with pytest.raises(ValueError, match='pinned ChessFly Space'):
        ChessFlyPolicy(tmp_path)
