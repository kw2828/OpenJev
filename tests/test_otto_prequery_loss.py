"""Hand-computed loss, poison isolation and calibration-gradient fixtures."""
import pytest
import torch

from openjev.research.otto_prequery_loss import nonquery_rows, prior_rows, weighted_loss


def test_nonquery_loss_preserves_original_separate_centering_and_blocked_gradients():
    pred = torch.tensor([[[64., 192., 1024., -256.], [5., 9., 1., 2.]]], requires_grad=True)
    target = torch.tensor([[[0., 64., -999., 999.], [3., 2., 1., 0.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False, False], [False, False, False, False]]])
    loss = nonquery_rows(pred, target, legal)
    torch.testing.assert_close(loss, torch.tensor([[.25, 0.]]), rtol=0, atol=0)
    allowed = legal.float()
    count = allowed.sum(-1, keepdim=True).clamp(min=1)
    p, q = pred / 64, target.detach() / 64
    difference = p - (p * allowed).sum(-1, keepdim=True) / count - q + (q * allowed).sum(-1, keepdim=True) / count
    oracle = (difference.square() * allowed).sum(-1) / count[:, :, 0]
    assert torch.equal(loss, oracle)
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[-1 / 128, 1 / 128, 0., 0.], [0., 0., 0., 0.]]]), rtol=0, atol=0)
    assert target.grad is None
    torch.testing.assert_close(nonquery_rows(pred.detach() + 256, target.detach() - 128, legal), loss, rtol=0, atol=0)


def test_prior_uses_all_four_coordinates_and_calibrates_gap_sign_not_mean():
    pred = torch.tensor([[[0., 0., 0., 128.]]], requires_grad=True)
    target = torch.zeros_like(pred, requires_grad=True)
    loss = prior_rows(pred, target)
    assert loss.item() == .75
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[-1 / 256, -1 / 256, -1 / 256, 3 / 256]]]), rtol=0, atol=0)
    assert target.grad is None
    # Action four contributes even if a separate action selector would block it.
    assert pred.grad[0, 0, 3] > 0
    assert prior_rows(pred.detach() + 128, target.detach() - 256).item() == .75
    assert prior_rows(target.detach() + 512, target.detach()).item() == 0


def tensors():
    prediction = torch.tensor([[[0., 0., 0., 0.], [0., 64., 0., 0.]],
                               [[0., 0., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    targets = torch.zeros_like(prediction, requires_grad=True)
    legal = torch.ones_like(prediction, dtype=torch.bool)
    legal[0, 0, 3] = False
    weights = torch.tensor([[0., 1 / 54], [0., 0.]], dtype=torch.float64, requires_grad=True)
    prior = torch.full_like(prediction, float("nan"))
    prior[0, 0] = torch.tensor([0., 0., 0., 128.])
    prior.requires_grad_()
    prior_target = torch.full_like(prior, float("nan"))
    prior_target[0, 0] = 0
    prior_target.requires_grad_()
    prior_weights = torch.tensor([[1 / 54, float("nan")], [float("nan"), float("nan")]],
                                 dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False], [False, False]])
    return prediction, prior, targets, legal, weights, prior_target, prior_weights, mask


def test_weighted_auxiliary_is_fixed_coefficient_one_and_keeps_zero_support_denominator():
    args = tensors()
    result = weighted_loss(*args, objective="query_aux")
    # First episode: NQ=.1875 and prior=.75. The second contributes zero, not removal.
    assert result["nonquery"].item() == pytest.approx(.1875 / 2)
    assert result["prior"].item() == pytest.approx(.75 / 2)
    assert result["total"].item() == pytest.approx(.9375 / 2)
    result["total"].backward()
    assert torch.isfinite(args[0].grad).all() and torch.isfinite(args[1].grad).all()
    assert torch.equal(args[1].grad[~args[-1]], torch.zeros(3, 4))
    assert args[1].grad[0, 0, 3] > 0
    for index in (2, 4, 5, 6):
        assert args[index].grad is None


def test_mse_never_reads_any_prior_argument_and_is_exact_original_weighted_loss():
    args = tensors()
    prediction, _, targets, legal, weights, *_ = args
    poison = object()
    result = weighted_loss(prediction, poison, targets, legal, weights, poison, poison, poison, objective="mse")
    expected = (nonquery_rows(prediction, targets, legal) * weights.detach().float()).sum() * 27
    assert torch.equal(result["total"], expected) and torch.equal(result["nonquery"], expected)
    assert result["prior"].item() == 0 and not result["prior"].requires_grad
    result["total"].backward()
    assert targets.grad is None and weights.grad is None


def test_empty_prior_mask_does_not_numerically_read_poison_or_create_prior_gradient():
    args = list(tensors())
    args[1] = torch.full_like(args[0], float("nan"), requires_grad=True)
    args[5] = torch.full_like(args[0], float("nan"), requires_grad=True)
    args[6] = torch.full_like(args[4], float("nan"), requires_grad=True)
    args[7] = torch.zeros_like(args[7])
    result = weighted_loss(*args, objective="query_aux")
    assert result["prior"].item() == 0
    result["total"].backward()
    assert args[1].grad is args[5].grad is args[6].grad is None


@pytest.mark.parametrize("fault", ["objective", "overlap", "active_poison", "active_weight", "shape"])
def test_invalid_active_contract_rejected(fault):
    args = list(tensors())
    objective = "query_aux"
    if fault == "objective":
        objective = "tuned"
    elif fault == "overlap":
        args[4] = args[4].detach().clone()
        args[4][0, 0] = 1 / 54
    elif fault == "active_poison":
        args[5] = args[5].detach().clone()
        args[5][0, 0, 0] = float("nan")
    elif fault == "active_weight":
        args[6] = args[6].detach().clone()
        args[6][0, 0] = 0
    else:
        args[7] = args[7][:1]
    with pytest.raises(ValueError):
        weighted_loss(*args, objective=objective)
