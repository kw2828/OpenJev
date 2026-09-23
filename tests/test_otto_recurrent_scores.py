"""Fabricated causal score-predictor engineering; no data or planner calls."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openjev.research import otto_recurrent_scores as scores


def inputs():
    return (torch.linspace(-.8, .9, 2 * 4 * 31).reshape(2, 4, 31),
            torch.tensor([[64., 80., 48., 96.], [1.25, 3.5, 2.75, 5.]], dtype=torch.float32),
            torch.tensor([4, 2], dtype=torch.int64))


def nonzero_head(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.2, .3, model.output.weight.numel()).reshape_as(model.output.weight))
        model.output.bias.copy_(torch.tensor([.03, -.02, .01, -.04]))


@pytest.mark.parametrize("kind,count", [("residual_gru", 5862), ("direct_gru", 5862),
                                       ("history_mlp", 5895), ("current_mlp", 5826)])
def test_zero_head_holds_exact_raw_query_and_masks_padding_without_mutation(kind, count):
    rng = torch.random.get_rng_state().clone()
    model = scores.make_head(kind, 123)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert scores.parameter_count(kind) == count == sum(p.numel() for p in model.parameters())
    assert max(scores.parameter_count(k) for k in scores.KINDS) / min(scores.parameter_count(k) for k in scores.KINDS) < 1.02
    x, anchor, lengths = inputs()
    originals = [value.clone() for value in (x, anchor, lengths)]
    result = model(x, anchor, lengths)
    assert result.dtype == torch.float32 and result.shape == (2, 4, 4)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(result[row, :length], anchor[row].expand(length, -1))
        assert torch.equal(result[row, length:], torch.zeros(4 - length, 4))
    assert torch.equal(scores.valid_mask(lengths), torch.tensor([[True, True, True, True], [True, True, False, False]]))
    assert all(torch.equal(actual, before) for actual, before in zip((x, anchor, lengths), originals, strict=True))


def test_gru_families_have_exact_paired_initial_parameters():
    left, right = (scores.make_head(kind, 321).state_dict() for kind in scores.KINDS[:2])
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    other = scores.make_head("residual_gru", 322).state_dict()
    assert not torch.equal(left["recurrent.weight_ih"], other["recurrent.weight_ih"])


@pytest.mark.parametrize("kind", scores.KINDS)
def test_nonzero_predictions_are_causal_and_padding_never_enters_network(kind):
    model = scores.make_head(kind, 123)
    nonzero_head(model)
    x, anchor, lengths = inputs()
    base = model(x, anchor, lengths)
    changed = x.clone()
    changed[0, 3] = torch.linspace(-9., 7., 31)
    changed[1, 2:] = float("nan")
    observed = model(changed, anchor, lengths)
    assert torch.equal(base[0, :3], observed[0, :3])
    assert torch.equal(base[1], observed[1])
    assert torch.equal(observed[:, 0], anchor)
    assert torch.isnan(changed[1, 2:]).all()
    assert not torch.equal(base[0, 1:], anchor[0].expand(3, -1))
    # The executed age2 readout cannot send gradients into age3 or padding.
    differentiable = x.clone().requires_grad_()
    prediction = model(differentiable, anchor, lengths)
    (prediction[0, 2, 0] - prediction[0, 2, 1]).backward()
    assert torch.equal(differentiable.grad[0, 3], torch.zeros(31))
    assert torch.equal(differentiable.grad[1], torch.zeros(4, 31))


@pytest.mark.parametrize("kind", scores.KINDS)
def test_each_call_is_a_fresh_query_window_and_batch_rows_do_not_share_state(kind):
    model = scores.make_head(kind, 999)
    nonzero_head(model)
    x, anchor, lengths = inputs()
    original = model(x, anchor, lengths).detach()
    model(-x, anchor * 3, lengths)
    repeated = model(x, anchor, lengths)
    assert torch.equal(original, repeated)
    for row in range(2):
        isolated = model(x[row:row + 1], anchor[row:row + 1], lengths[row:row + 1])
        torch.testing.assert_close(isolated[0], original[row], atol=3e-5, rtol=1e-6)


def test_recursive_residual_uses_its_own_previous_prediction_scalar_oracle():
    models = [scores.make_head(kind, 123) for kind in scores.KINDS[:2]]
    for model in models:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            # All reset/update gates are exactly1/2. Only the first candidate
            # coordinate depends on the first centered score, not features.
            model.recurrent.weight_ih[58, 31] = 1.
            model.output.weight[0, 0] = 1.
    anchor = torch.tensor([[64., 128., 192., 256.]])
    x, lengths = torch.zeros(1, 4, 31), torch.tensor([4])
    actual = [m(x, anchor, lengths)[0] for m in models]
    expected = []
    for recursive in (True, False):
        original = [1., 2., 3., 4.]
        previous = list(original)
        h = .5 * math.tanh(-1.5)
        result = [[v * 64 for v in previous]]
        for _ in range(3):
            incoming = previous if recursive else original
            centered_first = incoming[0] - sum(incoming) / 4
            h = .5 * h + .5 * math.tanh(centered_first)
            previous = [v + delta for v, delta in zip(incoming, (.75 * h, -.25 * h, -.25 * h, -.25 * h), strict=True)]
            result.append([v * 64 for v in previous])
        expected.append(torch.tensor(result, dtype=torch.float32))
    for answer, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(answer, oracle, atol=5e-5, rtol=1e-6)
    torch.testing.assert_close(actual[0][1], actual[1][1], atol=0, rtol=0)
    assert not torch.allclose(actual[0][2:], actual[1][2:], atol=.01, rtol=0)


@pytest.mark.parametrize("kind", scores.KINDS)
def test_transition_gradients_are_hidden_by_zero_head_then_reach_nonzero_head(kind):
    model = scores.make_head(kind, 77)
    x, anchor, lengths = inputs()
    desired = torch.tensor([3., -1., 2., -4.])
    def loss():
        prediction = model(x, anchor, lengths)[:, 1] / 64
        return ((prediction - desired) ** 2).sum()
    loss().backward()
    hidden = model.recurrent.weight_ih if "gru" in kind else model.hidden.weight
    assert torch.equal(hidden.grad, torch.zeros_like(hidden))
    assert model.output.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    nonzero_head(model)
    loss().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    if "gru" in kind:
        assert torch.isfinite(model.recurrent.weight_hh.grad).all()
        assert model.recurrent.weight_hh.grad.abs().sum() > 0


def test_strict_input_validation_and_explicit_subnormal_refusal():
    model = scores.make_head("history_mlp", 1)
    x, q, lengths = inputs()
    bad = [(x.double(), q, lengths), (x[:, :, :30], q, lengths), (x, q.double(), lengths),
           (x, q[:, :3], lengths), (x, q, lengths.float()), (x, q, torch.tensor([0, 2])),
           (x, q, torch.tensor([4, 5])), (x, q, torch.tensor([4])),
           (x * float("nan"), q, lengths), (x, q * float("inf"), lengths)]
    tiny = q.clone()
    tiny[0, 0] = torch.nextafter(torch.tensor(0., dtype=torch.float32), torch.tensor(1., dtype=torch.float32))
    bad.append((x, tiny, lengths))
    for args in bad:
        with pytest.raises(ValueError):
            model(*args)
    for kind, seed in (("unknown", 1), ("current_mlp", True), ("current_mlp", -1), ("current_mlp", 2**32)):
        with pytest.raises(ValueError):
            scores.make_head(kind, seed)
