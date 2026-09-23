"""Fabricated SPO+ values, fixed weighting, subgradients and input contracts."""
from itertools import product

import pytest
import torch

from openjev.research.otto_spo_plus_loss import spo_plus_rows, weighted_spo_plus_loss


def test_hand_value_teacher_detachment_illegal_gradients_and_shift_invariance():
    prediction = torch.tensor([[[128., 0., -6400., 6400.]]], requires_grad=True)
    targets = torch.tensor([[[0., 64., -9999., 9999.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False, False]]])
    originals = [tensor.detach().clone() for tensor in (prediction, targets, legal)]
    # z=(2,0), c=(0,1), teacher reference=(1,0), max(c-2z)=1.
    rows = spo_plus_rows(prediction, targets, legal)
    assert rows.dtype == torch.float32 and rows.shape == (1, 1)
    assert rows.item() == 5
    rows.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1 / 32, -1 / 32, 0., 0.]]]),
                               rtol=0, atol=0)
    assert targets.grad is None
    shifted = spo_plus_rows(prediction.detach() + 256, targets.detach() - 128, legal)
    assert torch.equal(rows, shifted)
    for value, original in zip((prediction, targets, legal), originals, strict=True):
        assert torch.equal(value, original)


def test_uniform_exact_teacher_tie_and_first_maximizer_subgradient():
    prediction = torch.zeros((1, 1, 4), requires_grad=True)
    targets = torch.tensor([[[-0., 0., 64., 64.]]], requires_grad=True)
    legal = torch.ones_like(prediction, dtype=torch.bool)
    # Uniform teacher reference=(.5,.5,0,0), tied maximizers are actions2 and3.
    rows = spo_plus_rows(prediction, targets, legal)
    assert rows.item() == 1
    rows.sum().backward()
    expected = torch.tensor([[[1 / 64, 1 / 64, -1 / 32, 0.]]])
    torch.testing.assert_close(prediction.grad, expected, rtol=0, atol=0)
    assert targets.grad is None


def test_three_way_exact_teacher_tie_is_uniform_and_perfect_scores_zero():
    prediction = torch.tensor([[[32., 64., 96., -64.]]], requires_grad=True)
    targets = torch.tensor([[[16., 16., 16., 80.]]])
    legal = torch.ones_like(prediction, dtype=torch.bool)
    rows = spo_plus_rows(prediction, targets, legal)
    assert rows.item() == 5
    rows.sum().backward()
    torch.testing.assert_close(prediction.grad,
                               torch.tensor([[[1 / 96, 1 / 96, 1 / 96, -1 / 32]]]),
                               rtol=2e-7, atol=0)
    perfect = targets.detach().clone().requires_grad_()
    assert spo_plus_rows(perfect, targets, legal).item() == 0


def test_near_teacher_tie_is_not_an_exact_reference_tie():
    prediction = torch.tensor([[[64., 0., 999., -999.]]], requires_grad=True)
    targets = torch.tensor([[[0., 5e-11, -999., 999.]]])
    legal = torch.tensor([[[True, True, False, False]]])
    assert float(targets[0, 0, 1] - targets[0, 0, 0]) < 1e-10
    rows = spo_plus_rows(prediction, targets, legal)
    assert rows.item() == 2
    rows.sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1 / 32, -1 / 32, 0., 0.]]]),
                               rtol=0, atol=0)


def test_teacher_tie_membership_precedes_scale_underflow():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
    prediction = torch.tensor([[[64., 0., 0., 0.]]], requires_grad=True)
    targets = torch.tensor([[[0., 0., 0., 0.]]])
    targets[0, 0, 1] = tiny
    legal = torch.tensor([[[True, True, False, False]]])
    assert tiny > 0 and (tiny / 64).item() == 0
    spo_plus_rows(prediction, targets, legal).sum().backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1 / 32, -1 / 32, 0., 0.]]]),
                               rtol=0, atol=0)


def test_single_legal_action_and_padding_are_connected_exact_zero_even_at_finite_extremes():
    largest = torch.finfo(torch.float32).max
    prediction = torch.tensor([[[largest, -largest, largest, -largest],
                                [-largest, largest, -largest, largest]]], requires_grad=True)
    targets = -prediction.detach()
    legal = torch.tensor([[[False, True, False, False], [False, False, False, False]]])
    rows = spo_plus_rows(prediction, targets, legal)
    assert rows.requires_grad and torch.equal(rows, torch.zeros((1, 2)))
    rows.sum().backward()
    assert prediction.grad is not None and torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_weighting_preserves_fixed_denominator_and_detaches_weights():
    prediction = torch.tensor([[[128., 0., 0., 0.], [0., 0., 0., 0.]],
                               [[128., 0., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    targets = torch.tensor([[[0., 64., 0., 0.], [0., 0., 0., 0.]],
                            [[0., 64., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    legal = torch.tensor([[[True, True, False, False], [False, False, False, False]],
                          [[True, True, False, False], [False, False, False, False]]])
    weights = torch.tensor([[2 / 54, 0.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    total = weighted_spo_plus_loss(prediction, targets, legal, weights)
    # Only first episode has support, but B is still two; realized weight mass is not normalized.
    expected = (spo_plus_rows(prediction, targets, legal) * weights.detach().float()).sum() * 27
    assert torch.equal(total, expected)
    assert total.item() == pytest.approx(5)
    total.backward()
    assert targets.grad is None and weights.grad is None
    assert torch.equal(prediction.grad[1], torch.zeros((2, 4)))
    torch.testing.assert_close(prediction.grad[0, 0], torch.tensor([1 / 32, -1 / 32, 0., 0.]),
                               rtol=1e-7, atol=0)


def test_all_zero_weights_preserve_zero_gradient_connection():
    prediction = torch.tensor([[[128., 0., 0., 0.], [64., 128., 192., 256.]]], requires_grad=True)
    targets = torch.zeros_like(prediction, requires_grad=True)
    legal = torch.tensor([[[True, True, False, False], [False, False, False, False]]])
    total = weighted_spo_plus_loss(prediction, targets, legal, torch.zeros((1, 2)))
    assert total.item() == 0 and total.requires_grad
    total.backward()
    assert prediction.grad is not None and torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert targets.grad is None


def test_finite_difference_gradient_away_from_maximum_and_teacher_ties():
    prediction = torch.tensor([[[80., 16., 32., -1000.]]], requires_grad=True)
    targets = torch.tensor([[[0., 128., 64., -2000.]]])
    legal = torch.tensor([[[True, True, True, False]]])
    spo_plus_rows(prediction, targets, legal).sum().backward()
    step = .125
    differences = []
    for action in range(4):
        plus, minus = prediction.detach().clone(), prediction.detach().clone()
        plus[0, 0, action] += step
        minus[0, 0, action] -= step
        differences.append((spo_plus_rows(plus, targets, legal).item()
                            - spo_plus_rows(minus, targets, legal).item()) / (2 * step))
    torch.testing.assert_close(prediction.grad, torch.tensor([[differences]]), rtol=0, atol=1e-6)


def test_fabricated_grid_direct_formula_and_exact_oracle_regret_bound():
    # Exhaust all 3^4 teachers, 3^4 predictions and 15 nonempty masks. No model calls.
    grid = list(product((-64., 0., 64.), repeat=4))
    pairs = list(product(grid, repeat=2))
    targets = torch.tensor([pair[0] for pair in pairs]).unsqueeze(0)
    prediction = torch.tensor([pair[1] for pair in pairs]).unsqueeze(0)
    for bits in product((False, True), repeat=4):
        if not any(bits):
            continue
        legal = torch.tensor(bits).view(1, 1, 4).expand_as(prediction)
        rows = spo_plus_rows(prediction, targets, legal)
        # Float64 direct paper formula is independent of the anchored production route.
        c, z = targets.double() / 64, prediction.double() / 64
        cmin = c.masked_fill(~legal, float("inf")).min(-1, keepdim=True).values
        best = legal & (c == cmin)
        uniform = best.double() / best.sum(-1, keepdim=True)
        direct = ((c - 2 * z).masked_fill(~legal, float("-inf")).max(-1).values
                  + 2 * (z * uniform).sum(-1) - cmin[:, :, 0])
        torch.testing.assert_close(rows.double(), direct, rtol=2e-7, atol=3e-7)
        # Check the worst teacher cost among ALL exact predicted minima, including ties.
        pmin = prediction.masked_fill(~legal, float("inf")).min(-1, keepdim=True).values
        predicted_best = legal & (prediction == pmin)
        regret = c.masked_fill(~predicted_best, float("-inf")).max(-1).values - cmin[:, :, 0]
        assert bool((rows.double() + 3e-7 >= regret).all())
        assert bool((rows >= -3e-7).all())


@pytest.mark.parametrize("fault", ["prediction_dtype", "teacher_dtype", "teacher_shape", "legal_dtype",
                                    "legal_shape", "empty", "nan_prediction", "inf_teacher"])
def test_invalid_row_contract_is_rejected(fault):
    prediction = torch.zeros((1, 2, 4))
    targets = torch.ones_like(prediction)
    legal = torch.zeros_like(prediction, dtype=torch.bool)
    if fault == "prediction_dtype":
        prediction = prediction.double()
    elif fault == "teacher_dtype":
        targets = targets.double()
    elif fault == "teacher_shape":
        targets = targets[:, :1]
    elif fault == "legal_dtype":
        legal = legal.float()
    elif fault == "legal_shape":
        legal = legal[:, :1]
    elif fault == "empty":
        prediction, targets, legal = prediction[:, :0], targets[:, :0], legal[:, :0]
    elif fault == "nan_prediction":
        prediction[0, 0, 0] = float("nan")
    elif fault == "inf_teacher":
        targets[0, 0, 0] = float("inf")
    with pytest.raises(ValueError):
        spo_plus_rows(prediction, targets, legal)


@pytest.mark.parametrize("fault", ["weight_dtype", "weight_shape", "negative", "nan", "cast_overflow",
                                    "cast_underflow", "weighted_padding", "batch", "weighted_overflow"])
def test_invalid_weighted_contract_is_rejected(fault):
    prediction = torch.tensor([[[128., 0., 0., 0.]]])
    targets = torch.tensor([[[0., 64., 0., 0.]]])
    legal = torch.tensor([[[True, True, False, False]]])
    weights = torch.ones((1, 1), dtype=torch.float64)
    if fault == "weight_dtype":
        weights = weights.to(torch.int64)
    elif fault == "weight_shape":
        weights = weights[:, :, None]
    elif fault == "negative":
        weights[0, 0] = -1
    elif fault == "nan":
        weights[0, 0] = float("nan")
    elif fault == "cast_overflow":
        weights[0, 0] = 1e100
    elif fault == "cast_underflow":
        weights[0, 0] = 1e-100
    elif fault == "weighted_padding":
        legal[:] = False
    elif fault == "batch":
        prediction, targets, legal = (item.expand(7, -1, -1) for item in (prediction, targets, legal))
        weights = weights.expand(7, -1)
    elif fault == "weighted_overflow":
        weights[0, 0] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError):
        weighted_spo_plus_loss(prediction, targets, legal, weights)
