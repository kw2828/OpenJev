import chess
import numpy as np
import pytest
import torch

from openjev.research.chess_graph_contrast import ARMS, GraphContrastHead, candidate_graphs
from openjev.research.chess_transport import TransportHead


def fixture():
    torch.manual_seed(239)
    nodes = torch.randn(2, 64, 32, dtype=torch.double)
    features = torch.randn(2, 3, 120, dtype=torch.double)
    candidates = torch.zeros(2, 3, 5, dtype=torch.long)
    candidates[..., :2] = torch.tensor([[[0, 2], [1, 3], [0, 0]], [[1, 2], [2, 1], [0, 0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    logits = torch.randn(2, 3, dtype=torch.double).masked_fill(~mask, -torch.inf)
    roots = torch.zeros(2, 2, 64, 64, dtype=torch.uint8)
    roots[:, 0, 0, 1] = 1; roots[:, 1, 2, 3] = 1; roots[:, 0, 1, 3] = 1
    children = roots[:, None].expand(-1, 3, -1, -1, -1).clone()
    children[:, 0, 0, 0, 1] = 0; children[:, 0, 0, 2, 1] = 1
    children[:, 1, 1, 2, 3] = 0; children[:, 1, 1, 1, 3] = 1
    return nodes, features, candidates, mask, logits, roots, children


def nonzero_head(arm):
    model = GraphContrastHead(arm, seed=241).double()
    with torch.no_grad(): model.output.weight.copy_(torch.linspace(-.2, .2, 32)[None])
    return model


@pytest.mark.parametrize('arm', ARMS)
def test_initial_policy_and_parameter_identity(arm):
    model = GraphContrastHead(arm).double()
    args = fixture(); permutation = torch.tensor([[1, 0, 2], [2, 0, 1]])
    assert sum(p.numel() for p in model.parameters()) == 16744
    assert torch.equal(model(*args, permutation=permutation), args[4])
    old = TransportHead(seed=1109).double()
    assert all(torch.equal(old.state_dict()[k], v) for k, v in model.state_dict().items())


def test_root_reference_matches_existing_transport_outputs_and_gradients():
    args = list(fixture()); args[0].requires_grad_(); args[1].requires_grad_()
    model = nonzero_head('root'); old = TransportHead().double()
    old.load_state_dict(model.state_dict())
    actual = model(*args); expected = old(*args[:6])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    chosen = args[3]
    actual_grad = torch.autograd.grad(actual[chosen].square().sum(), list(model.parameters())+args[:2], retain_graph=True)
    expected_grad = torch.autograd.grad(expected[chosen].square().sum(), list(old.parameters())+args[:2])
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_child_reference_matches_separate_per_candidate_transport():
    args = fixture(); head = nonzero_head('child'); old = TransportHead().double()
    old.load_state_dict(head.state_dict()); actual = head(*args)
    for b, m in args[3].nonzero().tolist():
        expected = old(args[0][b:b+1], args[1][b:b+1, m:m+1],
                       args[2][b:b+1, m:m+1], args[3][b:b+1, m:m+1],
                       args[4][b:b+1, m:m+1], args[6][b:b+1, m])
        torch.testing.assert_close(actual[b, m], expected.squeeze(), atol=1e-12, rtol=1e-12)


def test_contrast_is_difference_and_unchanged_graph_has_exact_zero_correction_gradient():
    args = fixture(); contrast = nonzero_head('contrast')
    child, root = nonzero_head('child'), nonzero_head('root')
    actual = contrast(*args); mask = args[3]
    expected = child(*args)[mask] - root(*args)[mask] + args[4][mask]
    torch.testing.assert_close(actual[mask], expected, atol=1e-12, rtol=1e-12)
    identical = list(args); identical[-1] = args[-2][:, None].expand_as(args[-1]).clone()
    result = contrast(*identical)
    assert torch.equal(result, args[4])
    result[mask].sum().backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in contrast.parameters())


def test_permuted_contrast_and_candidate_reordering_have_expected_semantics():
    args = list(fixture()); permutation = torch.tensor([[1, 0, 2], [2, 0, 1]])
    model = nonzero_head('permuted_contrast'); control = nonzero_head('contrast')
    actual = model(*args, permutation=permutation)
    changed = list(args); changed[-1] = args[-1][torch.arange(2)[:, None], permutation]
    torch.testing.assert_close(actual, control(*changed), atol=0, rtol=0)
    order = torch.tensor([2, 0, 1]); inverse = order.argsort()
    reordered = list(args)
    for i in (1, 2, 3, 4, 6): reordered[i] = args[i][:, order]
    new_permutation = inverse[permutation[:, order]]
    expected = model(*reordered, permutation=new_permutation)
    torch.testing.assert_close(actual[:, order], expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(actual[args[3]], control(*args)[args[3]])


@pytest.mark.parametrize('fen,move', [
    ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2', 'e5d6'),
    ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8n'),
])
@pytest.mark.parametrize('mirror', [False, True])
def test_native_graphs_include_special_moves_in_fixed_root_frame(fen, move, mirror):
    board = chess.Board(fen)
    if mirror:
        board = board.mirror(); original = chess.Move.from_uci(move)
        move = chess.Move(chess.square_mirror(original.from_square), chess.square_mirror(original.to_square),
                          promotion=original.promotion).uci()
    before = board.fen(en_passant='fen'), tuple(board.move_stack)
    result = candidate_graphs([board]); index = result['menus'][0].index(move)
    child = board.copy(stack=True); child.push_uci(move)
    expected = np.zeros((2, 64, 64), dtype=np.uint8)
    canonical = lambda square: square if board.turn else chess.square_mirror(square)
    for square, piece in child.piece_map().items():
        for target in child.attacks(square):
            expected[int(piece.color != board.turn), canonical(square), canonical(target)] = 1
    assert np.array_equal(result['children'][0, index].numpy(), expected)
    assert (board.fen(en_passant='fen'), tuple(board.move_stack)) == before
    n = len(result['menus'][0]); permutation = result['permutation'][0, :n]
    assert sorted(permutation.tolist()) == list(range(n))
    if n > 1: assert not (permutation == torch.arange(n)).any()
    assert torch.equal(candidate_graphs([board])['permutation'], result['permutation'])


def test_rejects_permutations_that_cross_padding_and_invalid_graphs():
    args = list(fixture()); model = nonzero_head('permuted_contrast')
    with pytest.raises(ValueError, match='legal menu'):
        model(*args, permutation=torch.tensor([[2, 0, 1], [0, 1, 2]]))
    args[-1] = args[-1].clone(); args[-1][0, 0, 0, 0, 1] = 2
    with pytest.raises(ValueError, match='binary'):
        nonzero_head('contrast')(*args)
