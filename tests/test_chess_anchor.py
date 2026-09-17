import numpy as np
import pytest
import torch
from torch.nn import functional as F

pytest.importorskip('chess')
import chess

from openjev.research.chess_anchor import CHECKPOINT_VERSION, RECURRENCES, AnchorChess
from openjev.research.chess_spatial import (
    ResidualBlock,
    SpatialChess,
    encode_board,
    encode_candidates,
    piece_targets,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(boards, device='cpu'):
    rows = [encode_candidates(board) for board in boards]
    size = max(len(ids) for ids, _ in rows)
    candidates = torch.zeros(len(boards), size, 5, dtype=torch.long)
    mask = torch.zeros(len(boards), size, dtype=torch.bool)
    for index, (ids, features) in enumerate(rows):
        candidates[index, :len(ids)] = torch.from_numpy(features)
        mask[index, :len(ids)] = True
    observations = torch.from_numpy(np.stack([encode_board(board) for board in boards]))
    return tuple(value.to(device) for value in (observations, candidates, mask))


def boards():
    return [chess.Board(), chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')]


def test_parameter_initialization_rng_and_modules_are_exactly_matched():
    torch.manual_seed(922)
    before = torch.random.get_rng_state().clone()
    reference = SpatialChess('recurrent', seed=17)
    for recurrence in RECURRENCES:
        model = AnchorChess(recurrence, seed=17)
        assert model.parameter_count() == reference.parameter_count() == 43726
        assert list(model.state_dict()) == list(reference.state_dict())
        for name, value in model.state_dict().items():
            assert torch.equal(value, reference.state_dict()[name]), name
        assert model.core.forward.__func__ is ResidualBlock.forward
        assert list(dict(model.named_modules())) == list(dict(reference.named_modules()))
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize('seed', [17, 29, 43])
def test_first_step_outputs_identical_for_both_recurrences(seed):
    data = batch(boards())
    original = SpatialChess('recurrent', seed=seed)
    expected = original(*data, depth=1)
    for recurrence in RECURRENCES:
        actual = AnchorChess(recurrence, seed=seed)(*data, depth=1)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('depth', [1, 2, 4, 8, 16])
def test_residual_matches_original_outputs_and_gradients_at_every_depth(depth):
    data = batch(boards())
    original = SpatialChess('recurrent', seed=17, width=8)
    residual = AnchorChess('residual', seed=17, width=8)
    outputs = [model(*data, depth=depth) for model in (original, residual)]
    for left, right in zip(*outputs, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for logits, values, hidden in outputs:
        (F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
         + values.square().mean() + hidden.square().mean()).backward()
    for (name, left), (_, right) in zip(original.named_parameters(), residual.named_parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)


def test_anchor_is_the_manual_fixed_differentiable_encoder_anchor():
    model = AnchorChess('anchor', seed=17, width=8)
    data = batch(boards())
    hidden = model(*data, depth=4)[2]
    anchor = model.encoder(data[0])
    expected = anchor
    detached_anchor = anchor
    for _ in range(4):
        expected = F.relu(anchor+model.core.conv2(F.relu(model.core.conv1(expected))))
        detached_anchor = F.relu(anchor.detach()+model.core.conv2(F.relu(model.core.conv1(detached_anchor))))
    torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(hidden.square().sum(), model.encoder[0].weight)[0]
    manual_grad = torch.autograd.grad(expected.square().sum(), model.encoder[0].weight, retain_graph=True)[0]
    detached_grad = torch.autograd.grad(detached_anchor.square().sum(), model.encoder[0].weight)[0]
    torch.testing.assert_close(actual_grad, manual_grad, rtol=0, atol=0)
    assert not torch.allclose(actual_grad, detached_grad)
    residual = AnchorChess('residual', seed=17, width=8)(*data, depth=4)[2]
    assert not torch.equal(hidden, residual)


@pytest.mark.parametrize('recurrence', RECURRENCES)
def test_candidate_permutation_padding_and_inputs_are_preserved(recurrence):
    model = AnchorChess(recurrence, seed=7)
    data = batch(boards())
    before = [value.clone() for value in data]
    logits, value, hidden = model(*data)
    observations, candidates, mask = data
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()
    assert torch.equal(logits.softmax(-1)[~mask], torch.zeros_like(logits[~mask]))
    permutation = torch.arange(candidates.shape[1]-1, -1, -1)
    changed = model(observations, candidates[:, permutation], mask[:, permutation])
    torch.testing.assert_close(changed[0], logits[:, permutation])
    torch.testing.assert_close(changed[1], value, rtol=0, atol=0)
    torch.testing.assert_close(changed[2], hidden, rtol=0, atol=0)
    for left, right in zip(data, before, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize('recurrence', RECURRENCES)
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_all_heads_receive_finite_gradients_without_parameter_updates(recurrence, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    model = AnchorChess(recurrence, seed=7).to(device)
    current = boards()
    data = batch(current, device)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    logits, value, hidden = model(*data)
    targets = []
    for board in current:
        ids, _ = encode_candidates(board)
        successor = board.copy()
        successor.push_uci(ids[0])
        targets.append(piece_targets(successor, board.turn))
    prediction = model.predict_board(hidden, data[1][:, 0])
    loss = (F.cross_entropy(logits, torch.zeros(2, device=device, dtype=torch.long))
            + F.mse_loss(value, torch.tensor([.5, -.5], device=device))
            + F.cross_entropy(prediction, torch.from_numpy(np.stack(targets)).to(device)))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.equal(before[name], parameter), name


@pytest.mark.parametrize('recurrence', RECURRENCES)
def test_choose_is_legal_stateless_and_does_not_mutate_history(recurrence):
    board = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
        board.push_uci(uci)
    original_fen, original_stack = board.fen(en_passant='fen'), list(board.move_stack)
    same_position = chess.Board(original_fen)
    same_position.teacher_target = 'e2e4'
    model = AnchorChess(recurrence)
    first = model.choose(board, depth=8)
    model.choose(boards()[1], depth=2)
    repeated = model.choose(same_position, depth=8)
    for field in ('choice', 'probabilities', 'value', 'depth', 'recurrence'):
        assert first[field] == repeated[field]
    assert first['recurrence'] == recurrence
    assert 'no search or cross-move state' in first['model_kind']
    assert first['choice'] in first['probabilities']
    assert set(first['probabilities']) == {move.uci() for move in board.legal_moves}
    assert sum(first['probabilities'].values()) == pytest.approx(1.)
    assert board.fen(en_passant='fen') == original_fen and board.move_stack == original_stack


@pytest.mark.parametrize('recurrence', RECURRENCES)
def test_checkpoint_roundtrip_plan_arm_seed_and_old_format_rejections(tmp_path, recurrence):
    model = AnchorChess(recurrence, seed=29, width=8, depth=2)
    path = tmp_path/'model.pt'
    model.save(path, plan_sha256='a'*64)
    restored = AnchorChess.load(path, expected_plan_sha256='a'*64,
                                expected_recurrence=recurrence, expected_seed=29)
    assert not restored.training
    assert (restored.recurrence, restored.seed, restored.width, restored.depth) == (recurrence, 29, 8, 2)
    for left, right in zip(model(*batch(boards()), depth=8), restored(*batch(boards()), depth=8), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        model.save(path, plan_sha256='a'*64)
    with pytest.raises(ValueError, match='plan mismatch'):
        AnchorChess.load(path, expected_plan_sha256='b'*64)
    with pytest.raises(ValueError, match='recurrence mismatch'):
        AnchorChess.load(path, expected_recurrence='anchor' if recurrence == 'residual' else 'residual')
    with pytest.raises(ValueError, match='seed mismatch'):
        AnchorChess.load(path, expected_seed=17)
    with pytest.raises(ValueError, match='encoding or format'):
        SpatialChess.load(path)
    old = tmp_path/'old.pt'
    SpatialChess('recurrent', seed=29, width=8, depth=2).save(old, plan_sha256='a'*64)
    with pytest.raises(ValueError, match='format or encoding'):
        AnchorChess.load(old)


@pytest.mark.parametrize(('key', 'value'), [
    ('format_version', 1), ('encoding', 'other'), ('recurrence', 'other'),
    ('seed', True), ('width', 0), ('depth', 0), ('plan_sha256', 'broken'),
])
def test_corrupt_checkpoint_metadata_rejected(tmp_path, key, value):
    path = tmp_path/'model.pt'
    AnchorChess().save(path, plan_sha256='a'*64)
    payload = torch.load(path, weights_only=True)
    assert payload['format_version'] == CHECKPOINT_VERSION
    payload[key] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        AnchorChess.load(path)


def test_checkpoint_extra_keys_and_missing_weights_rejected(tmp_path):
    path = tmp_path/'model.pt'
    AnchorChess().save(path, plan_sha256='a'*64)
    payload = torch.load(path, weights_only=True)
    payload['mode'] = 'predict'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='format or encoding'):
        AnchorChess.load(path)
    del payload['mode']
    del payload['state_dict']['core.conv1.weight']
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        AnchorChess.load(path)


@pytest.mark.parametrize('depth', [0, -1, True, 1.5])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError, match='positive integer'):
        AnchorChess()(*batch(boards()), depth=depth)


def test_invalid_recurrence_and_empty_menus_rejected():
    for recurrence in ('recurrent', '', None, True):
        with pytest.raises(ValueError, match='recurrence'):
            AnchorChess(recurrence)
    data = batch(boards())
    with pytest.raises(ValueError, match='nonempty boolean'):
        AnchorChess()(*data[:2], torch.zeros_like(data[2]))
    mate = chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
    assert mate.is_checkmate()
    with pytest.raises(ValueError, match='empty legal'):
        AnchorChess().choose(mate)
