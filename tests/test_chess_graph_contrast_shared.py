import pytest
import torch

from openjev.research.chess_graph_contrast import ARMS
from openjev.research.chess_graph_contrast_shared import SharedRootGraphContrastHead
from test_chess_graph_contrast import fixture, nonzero_head


@pytest.mark.parametrize('arm', ARMS)
def test_shared_root_matches_dense_reference_output_and_gradients(arm):
    reference = nonzero_head(arm)
    model = SharedRootGraphContrastHead(arm).double(); model.load_state_dict(reference.state_dict())
    arguments = list(fixture()); permutation = torch.tensor([[1, 0, 2], [2, 0, 1]])
    arguments[0].requires_grad_(); arguments[1].requires_grad_()
    actual = model(*arguments, permutation=permutation)
    expected = reference(*arguments, permutation=permutation)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    mask = arguments[3]
    a = torch.autograd.grad(actual[mask].square().sum(), list(model.parameters())+arguments[:2], retain_graph=True)
    b = torch.autograd.grad(expected[mask].square().sum(), list(reference.parameters())+arguments[:2])
    for left, right in zip(a, b): torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    assert sum(p.numel() for p in model.parameters()) == 16744


@pytest.mark.parametrize('arm', ARMS)
def test_compact_children_match_padded_form_and_initial_policy(arm):
    args = list(fixture()); model = SharedRootGraphContrastHead(arm).double()
    permutation = torch.tensor([[1, 0, 2], [2, 0, 1]])
    assert torch.equal(model(*args, permutation=permutation), args[4])
    model.load_state_dict(nonzero_head(arm).state_dict())
    padded = model(*args, permutation=permutation)
    args[-1] = args[-1][args[3]]
    assert torch.equal(model(*args, permutation=permutation), padded)


def test_no_change_has_exact_forward_and_parameter_gradient_zero():
    args = list(fixture()); args[-1] = args[-2][:, None].expand_as(args[-1]).clone()
    model = SharedRootGraphContrastHead('contrast').double()
    model.load_state_dict(nonzero_head('contrast').state_dict())
    actual = model(*args); assert torch.equal(actual, args[4])
    actual[args[3]].sum().backward()
    assert all(p.grad is not None and not p.grad.count_nonzero() for p in model.parameters())


def test_root_does_not_require_child_construction():
    args = fixture(); model = SharedRootGraphContrastHead('root').double()
    assert torch.equal(model(*args[:6]), args[4])


def test_padding_cannot_become_a_permuted_child_and_native_self_edges_rejected():
    args = list(fixture()); model = SharedRootGraphContrastHead('permuted_contrast').double()
    with pytest.raises(ValueError, match='legal menu'):
        model(*args, permutation=torch.tensor([[2, 0, 1], [0, 1, 2]]))
    args[-1] = args[-1].clone(); args[-1][0, 0, 0, 3, 3] = 1
    with pytest.raises(ValueError, match='self edges'):
        model(*args, permutation=torch.tensor([[1, 0, 2], [2, 0, 1]]))
