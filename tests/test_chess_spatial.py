import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

pytest.importorskip('chess')
import chess

from openjev.research.chess_spatial import (
    DEPTH,
    ENCODING_VERSION,
    INPUT_CHANNELS,
    MODES,
    PIECE_CLASSES,
    WIDTH,
    SpatialChess,
    action_planes,
    encode_board,
    encode_candidates,
    piece_targets,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(boards):
    encoded = [encode_candidates(board) for board in boards]
    count = max(len(names) for names, _ in encoded)
    candidates = torch.zeros(len(boards), count, 5, dtype=torch.long)
    mask = torch.zeros(len(boards), count, dtype=torch.bool)
    for index, (names, features) in enumerate(encoded):
        candidates[index, :len(names)] = torch.from_numpy(features)
        mask[index, :len(names)] = True
    observations = torch.from_numpy(np.stack([encode_board(board) for board in boards]))
    return observations, candidates, mask


def test_initial_encoding_ownership_rights_counters_and_targets():
    board = chess.Board()
    observations = encode_board(board)
    assert observations.shape == (INPUT_CHANNELS, 8, 8) == (19, 8, 8)
    assert observations.dtype == np.float32 and observations[:12].sum() == 32
    assert observations[0, 1].sum() == 8 and observations[6, 6].sum() == 8
    assert observations[12:16].sum() == 4*64 and not observations[16:18].any()
    np.testing.assert_allclose(observations[18], math.log(2)/10)
    targets = piece_targets(board, chess.WHITE)
    assert targets.dtype == np.int64 and targets.shape == (8, 8)
    assert targets[0, 4] == 6 and targets[7, 4] == 12 and targets[3, 4] == 0
    board.castling_rights = chess.BB_A1 | chess.BB_H8
    np.testing.assert_array_equal(encode_board(board)[12:16, 0, 0], [0, 1, 1, 0])
    np.testing.assert_array_equal(encode_board(board, chess.BLACK)[12:16, 0, 0], [1, 0, 0, 1])


@pytest.mark.parametrize('fen', [chess.STARTING_FEN,
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 5 12',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2',
    '7k/P7/8/8/8/8/8/7K w - - 0 1'])
def test_color_rank_mirror_is_canonically_identical(fen):
    board = chess.Board(fen)
    mirrored = board.mirror()
    np.testing.assert_array_equal(encode_board(board), encode_board(mirrored))
    np.testing.assert_array_equal(piece_targets(board, board.turn), piece_targets(mirrored, mirrored.turn))
    names, features = encode_candidates(board)
    mirrored_names, mirrored_features = encode_candidates(mirrored)
    lookup = dict(zip(mirrored_names, mirrored_features, strict=True))
    for name, row in zip(names, features, strict=True):
        move = chess.Move.from_uci(name)
        opposite = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                              promotion=move.promotion)
        np.testing.assert_array_equal(row, lookup[opposite.uci()])


@pytest.mark.parametrize(('fen', 'uci', 'source', 'target', 'promotion', 'dx', 'dy'), [
    (chess.STARTING_FEN, 'e2e4', chess.E2, chess.E4, 0, 0, 2),
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1', chess.E1, chess.G1, 0, 2, 0),
    ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8c8', chess.E1, chess.C1, 0, -2, 0),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', 'e5d6', chess.E5, chess.D6, 0, -1, 1),
    ('4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2', 'e4d3', chess.E5, chess.D6, 0, -1, 1),
    ('7k/P7/8/8/8/8/8/7K w - - 0 1', 'a7a8n', chess.A7, chess.A8, 2, 0, 1),
    ('7k/8/8/8/8/8/p7/7K b - - 0 1', 'a2a1q', chess.A7, chess.A8, 5, 0, 1),
])
def test_exact_special_move_features_and_action_planes(fen, uci, source, target, promotion, dx, dy):
    board = chess.Board(fen)
    names, features = encode_candidates(board)
    assert names == tuple(sorted(move.uci() for move in board.legal_moves))
    assert features.dtype == np.int64 and features.shape == (len(names), 5)
    np.testing.assert_array_equal(features[names.index(uci)], [source, target, promotion, dx+7, dy+7])
    planes = action_planes(torch.from_numpy(features[names.index(uci)]).unsqueeze(0))
    assert planes.shape == (1, 3, 8, 8)
    assert planes[0, 0].flatten()[source] == 1 and planes[0, 0].sum() == 1
    assert planes[0, 1].flatten()[target] == 1 and planes[0, 1].sum() == 1
    assert planes[0, 2].flatten()[target] == pytest.approx(promotion/5)
    assert planes[0, 2].sum() == pytest.approx(promotion/5)


@pytest.mark.parametrize(('fen', 'uci', 'changed'), [
    (chess.STARTING_FEN, 'e2e4', {'e2': 0, 'e4': 1}),
    ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8c8', {'e1': 0, 'a1': 0, 'c1': 6, 'd1': 4}),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', 'e5d6', {'e5': 0, 'd5': 0, 'd6': 1}),
    ('7k/8/8/8/8/8/p7/7K b - - 0 1', 'a2a1q', {'a7': 0, 'a8': 5}),
])
def test_successor_labels_keep_pre_move_perspective_including_side_effects(fen, uci, changed):
    board = chess.Board(fen)
    perspective = board.turn
    board.push_uci(uci)
    targets = piece_targets(board, perspective)
    encoded = encode_board(board, perspective)
    for square, target in changed.items():
        index = chess.parse_square(square)
        assert targets.flat[index] == target
        if target:
            assert encoded[target-1].flat[index] == 1
        else:
            assert encoded[:12].reshape(12, -1)[:, index].sum() == 0
    assert not np.array_equal(encoded, encode_board(board))
    if uci == 'e2e4':
        assert encoded[16].flat[chess.E3] == 1


def test_initial_parameters_and_outputs_match_but_cnn_blocks_are_independent():
    torch.manual_seed(918)
    rng_before = torch.random.get_rng_state().clone()
    models = [SpatialChess(mode, 7) for mode in MODES]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    shared = ('encoder', 'promotion_embedding', 'dx_embedding', 'dy_embedding',
              'policy_head', 'value_head', 'aux_head')
    reference = models[1]
    for model in models:
        for name in shared:
            expected = getattr(reference, name).state_dict()
            for key, value in getattr(model, name).state_dict().items():
                assert torch.equal(expected[key], value)
        cores = model.blocks if model.mode == 'cnn' else [model.core]
        for core in cores:
            for key, value in core.state_dict().items():
                assert torch.equal(reference.core.state_dict()[key], value)
    pointers = [block.conv1.weight.data_ptr() for block in models[0].blocks]
    assert len(set(pointers)) == DEPTH
    assert len({model.parameter_count() for model in models[1:]}) == 1
    core_parameters = sum(parameter.numel() for parameter in reference.core.parameters())
    assert models[0].parameter_count()-reference.parameter_count() == (DEPTH-1)*core_parameters
    data = batch([chess.Board()])
    for model in models:
        for expected, actual in zip(reference(*data), model(*data), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_masked_variable_menus_and_candidate_permutation_equivariance(mode):
    model = SpatialChess(mode, 7)
    boards = [chess.Board(), chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')]
    observations, candidates, mask = batch(boards)
    logits, value, hidden = model(observations, candidates, mask)
    assert logits.shape == mask.shape and value.shape == (2,) and hidden.shape == (2, WIDTH, 8, 8)
    assert torch.isneginf(logits[~mask]).all() and torch.isfinite(logits[mask]).all()
    assert torch.equal(logits.softmax(-1)[~mask], torch.zeros_like(logits[~mask]))
    assert ((value >= -1) & (value <= 1)).all()
    permutation = torch.arange(candidates.shape[1]-1, -1, -1)
    permuted, next_value, next_hidden = model(observations, candidates[:, permutation], mask[:, permutation])
    torch.testing.assert_close(permuted, logits[:, permutation])
    torch.testing.assert_close(value, next_value, rtol=0, atol=0)
    torch.testing.assert_close(hidden, next_hidden, rtol=0, atol=0)
    single = model(*batch(boards[:1]))
    torch.testing.assert_close(logits[0], single[0][0])


@pytest.mark.parametrize('mode', MODES)
def test_policy_value_and_prediction_gradients_reach_every_parameter_without_updates(mode):
    model = SpatialChess(mode, 7)
    boards = [chess.Board(), chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')]
    observations, candidates, mask = batch(boards)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    logits, value, hidden = model(observations, candidates, mask)
    # Different legal actions on the same current hidden must be represented.
    actions = candidates[:, 0]
    prediction = model.predict_board(hidden, actions)
    assert prediction.shape == (2, PIECE_CLASSES, 8, 8)
    targets = []
    for board in boards:
        names, _ = encode_candidates(board)
        side = board.turn
        successor = board.copy()
        successor.push_uci(names[0])
        targets.append(piece_targets(successor, side))
    loss = (F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
            + F.mse_loss(value, torch.tensor([.5, -.5]))
            + F.cross_entropy(prediction, torch.from_numpy(np.stack(targets))))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.equal(before[name], parameter), name
    alternate = model.predict_board(hidden.detach(), candidates[:, 1])
    assert not torch.equal(prediction.detach(), alternate)


def test_reconstruction_and_prediction_share_architecture_but_different_explicit_labels():
    board = chess.Board()
    root = piece_targets(board, board.turn)
    side = board.turn
    board.push_uci('e2e4')
    future = piece_targets(board, side)
    assert np.count_nonzero(root != future) == 2
    good = torch.full((1, 13, 8, 8), -5.)
    target = torch.from_numpy(future).unsqueeze(0)
    good.scatter_(1, target.unsqueeze(1), 5.)
    assert F.cross_entropy(good, target) < F.cross_entropy(good, torch.from_numpy(root).unsqueeze(0))


def test_no_cross_move_state_history_or_teacher_labels_enter_decision():
    board = chess.Board()
    for move in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        board.push_uci(move)
    detached = chess.Board(board.fen(en_passant='fen'))
    detached.teacher_target = 'a2a4'
    detached.engine_score = 123456
    np.testing.assert_array_equal(encode_board(board), encode_board(detached))
    model = SpatialChess('predict', 7)
    initial = model.choose(board)
    model.choose(chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1'))
    final = model.choose(detached)
    for field in ('choice', 'probabilities', 'value', 'depth'):
        assert initial[field] == final[field]
    assert chess.Move.from_uci(final['choice']) in board.legal_moves
    assert set(final['probabilities']) == {move.uci() for move in board.legal_moves}
    assert sum(final['probabilities'].values()) == pytest.approx(1.)


@pytest.mark.parametrize('mode', MODES)
def test_checkpoint_roundtrip_and_identity_rejections(tmp_path, mode):
    model = SpatialChess(mode, 7, width=8, depth=2)
    path = tmp_path/'checkpoint.pt'
    model.save(path, plan_sha256='a'*64)
    restored = SpatialChess.load(path, expected_plan_sha256='a'*64)
    assert (restored.mode, restored.seed, restored.width, restored.depth) == (mode, 7, 8, 2)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with pytest.raises(ValueError, match='plan hash mismatch'):
        SpatialChess.load(path, expected_plan_sha256='b'*64)
    with pytest.raises(FileExistsError):
        model.save(path, plan_sha256='a'*64)
    payload = torch.load(path, weights_only=True)
    assert payload['encoding'] == ENCODING_VERSION
    payload['encoding'] = 'other-perspective'
    torch.save(payload, tmp_path/'wrong.pt')
    with pytest.raises(ValueError, match='encoding or format'):
        SpatialChess.load(tmp_path/'wrong.pt')


def test_invalid_inputs_fail_before_scoring():
    model = SpatialChess('cnn', 7)
    observations, candidates, mask = batch([chess.Board()])
    with pytest.raises(ValueError, match='nonempty boolean'):
        model(observations, candidates, torch.zeros_like(mask))
    with pytest.raises(ValueError, match='nonempty boolean'):
        model(observations, candidates, mask.float())
    with pytest.raises(ValueError, match='int64'):
        model(observations, candidates.float(), mask)
    broken = candidates.clone()
    broken[0, 0, 0] = 64
    with pytest.raises(ValueError, match='square indices'):
        model(observations, broken, mask)
    with pytest.raises(ValueError, match='untied blocks'):
        model(observations, candidates, mask, depth=DEPTH+1)
    with pytest.raises(ValueError, match='positive integer'):
        model(observations, candidates, mask, depth=0)
    with pytest.raises(ValueError, match='perspective'):
        encode_board(chess.Board(), perspective='white')
    with pytest.raises(ValueError, match='valid standard'):
        encode_board(chess.Board(None))
    terminal = chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
    assert terminal.is_checkmate()
    assert encode_candidates(terminal)[1].shape == (0, 5)
    with pytest.raises(ValueError, match='empty legal-move menu'):
        model.choose(terminal)


def test_recurrent_extra_computation_is_explicit_and_never_persistent():
    model = SpatialChess('recurrent', 7)
    data = batch([chess.Board()])
    shallow = model(*data, depth=1)
    deep = model(*data, depth=8)
    assert not torch.equal(shallow[2], deep[2])
    repeated = model(*data, depth=1)
    for expected, actual in zip(shallow, repeated, strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
