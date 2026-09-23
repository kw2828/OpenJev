"""Fabricated composition, baseline isolation, fixed weights and gradient checks."""
import pytest
import torch

from openjev.research import otto_action_focused_loss as loss
from openjev.research import otto_prequery_loss as base
from openjev.research import otto_spo_plus_loss as spo_base


def fixture():
    prediction = torch.tensor([[[0., 0., 0., 0.], [128., 0., -1024., 1024.]],
                               [[0., 0., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    targets = torch.tensor([[[0., 0., 0., 0.], [0., 64., 999., -999.]],
                            [[0., 0., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, True, False], [True, True, False, False]],
                          [[False, False, False, False], [False, False, False, False]]])
    weights = torch.tensor([[0., 2 / 54], [0., 0.]], dtype=torch.float64, requires_grad=True)
    prior = torch.full_like(prediction, float("nan"))
    prior[0, 0] = torch.tensor([0., 0., 0., 128.])
    prior.requires_grad_()
    prior_target = torch.full_like(prediction, float("nan"))
    prior_target[0, 0] = 0
    prior_target.requires_grad_()
    prior_weights = torch.full((2, 2), float("nan"), dtype=torch.float64)
    prior_weights[0, 0] = 1 / 54
    prior_weights.requires_grad_()
    mask = torch.tensor([[True, False], [False, False]])
    return prediction, prior, targets, legal, weights, prior_target, prior_weights, mask


def test_aux_exactly_preserves_old_query_aux_and_never_calls_spo(monkeypatch):
    args = fixture()
    expected = base.weighted_loss(*args, objective="query_aux")

    def forbidden(*args, **kwargs):
        raise AssertionError("AUX evaluated SPO+")

    monkeypatch.setattr(spo_base, "weighted_spo_plus_loss", forbidden)
    result = loss.weighted_loss(*args, objective="aux")
    assert set(result) == {"total", "nonquery", "prior", "spo"}
    for key in ("total", "nonquery", "prior"):
        assert torch.equal(result[key], expected[key])
    assert result["spo"].item() == 0 and not result["spo"].requires_grad
    actual_grads = torch.autograd.grad(result["total"], (args[0], args[1]), retain_graph=True)
    expected_grads = torch.autograd.grad(expected["total"], (args[0], args[1]))
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.equal(actual, reference)


def test_hand_components_and_sum_gradient_preserve_all_weights_and_blocked_prior_coordinate():
    args = fixture()
    result = loss.weighted_loss(*args, objective="spo")
    # The zero-support second episode stays in B=2. Effective NQ weight=1, prior=.5.
    expected = {"nonquery": 2.25, "prior": .375, "spo": 5., "total": 7.625}
    for key, value in expected.items():
        assert result[key].dtype == torch.float32 and result[key].ndim == 0
        assert result[key].item() == pytest.approx(value, rel=1e-7)
    result["total"].backward()
    torch.testing.assert_close(args[0].grad[0, 1], torch.tensor([7 / 128, -7 / 128, 0., 0.]),
                               rtol=1e-7, atol=0)
    torch.testing.assert_close(args[1].grad[0, 0], torch.tensor([-1 / 512, -1 / 512, -1 / 512, 3 / 512]),
                               rtol=1e-7, atol=0)
    assert not args[3][0, 0, 3] and args[1].grad[0, 0, 3] > 0
    assert torch.equal(args[0].grad[1], torch.zeros((2, 4)))
    assert torch.equal(args[1].grad[~args[-1]], torch.zeros((3, 4)))
    for index in (2, 4, 5, 6):
        assert args[index].grad is None


def test_spo_total_is_exact_component_addition_and_same_nonquery_weight_object(monkeypatch):
    args = fixture()
    reference = base.weighted_loss(*args, objective="query_aux")
    weighted = spo_base.weighted_spo_plus_loss
    calls = []

    def observed(prediction, targets, legal, weights):
        assert prediction is args[0] and targets is args[2] and legal is args[3] and weights is args[4]
        calls.append(1)
        return weighted(prediction, targets, legal, weights)

    monkeypatch.setattr(spo_base, "weighted_spo_plus_loss", observed)
    result = loss.weighted_loss(*args, objective="spo")
    assert calls == [1]
    assert torch.equal(result["nonquery"], reference["nonquery"])
    assert torch.equal(result["prior"], reference["prior"])
    assert torch.equal(result["total"], reference["total"] + result["spo"])


@pytest.mark.parametrize("objective", ["aux", "spo"])
def test_zero_target_chunk_is_connected_zero_and_never_reads_inactive_prior_poison(objective):
    prediction = torch.zeros((2, 3, 4), requires_grad=True)
    targets = torch.zeros_like(prediction, requires_grad=True)
    prior = torch.full_like(prediction, float("nan"), requires_grad=True)
    legal = torch.zeros_like(prediction, dtype=torch.bool)
    weights = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    prior_weights = torch.full((2, 3), float("nan"), dtype=torch.float64, requires_grad=True)
    prior_mask = torch.zeros((2, 3), dtype=torch.bool)
    result = loss.weighted_loss(prediction, prior, targets, legal, weights, prior, prior_weights,
                                prior_mask, objective=objective)
    assert all(value.item() == 0 for value in result.values())
    assert result["total"].requires_grad
    result["total"].backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert prior.grad is targets.grad is weights.grad is prior_weights.grad is None


def test_rescore_helpers_are_exact_qualified_functions():
    assert loss.nonquery_rows is base.nonquery_rows
    assert loss.prior_rows is base.prior_rows
    assert loss.spo_plus_rows is spo_base.spo_plus_rows
    assert loss.SCALE == 64 and loss.EPISODES == 54


@pytest.mark.parametrize("fault", ["objective", "prior_overlap", "active_prior_nan", "nonfinite_scores",
                                   "weighted_padding"])
def test_invalid_objective_or_inherited_support_is_rejected(fault):
    args = list(fixture())
    objective = "spo"
    if fault == "objective":
        objective = "tuned"
    elif fault == "prior_overlap":
        args[4] = args[4].detach().clone()
        args[4][0, 0] = 1 / 54
    elif fault == "active_prior_nan":
        args[1] = args[1].detach().clone()
        args[1][0, 0, 0] = float("nan")
    elif fault == "nonfinite_scores":
        args[0] = args[0].detach().clone()
        args[0][0, 1, 2] = float("inf")
    elif fault == "weighted_padding":
        args[3] = args[3].clone()
        args[3][0, 1] = False
    with pytest.raises(ValueError):
        loss.weighted_loss(*args, objective=objective)
