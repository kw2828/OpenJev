"""Synthetic output-unit and strict-copy checks; no empirical data or fitting."""
from contextlib import nullcontext

import pytest
import torch

from openjev.research import otto_prequery_scores as original
from openjev.research import otto_protected_readout as raw
from openjev.research import otto_protected_training_model as M


def bits(value):
    return value.detach().contiguous().view(torch.uint8)


def inputs():
    lengths = torch.tensor([9, 6, 2])
    features = torch.linspace(-.2, .7, 3 * 9 * 31).reshape(3, 9, 31)
    scores = torch.full((3, 9, 4), float("nan"))
    mask = torch.zeros(3, 9, dtype=torch.bool)
    for row, length in enumerate(lengths.tolist()):
        for step in range(length):
            features[row, step, 15:18] = torch.tensor([step / 2188, (step % 4) / 2188, 1.])
            if step % 4 == 0:
                scores[row, step] = torch.tensor([12., 15., 11., 18.]) + row + step * .125
                mask[row, step] = True
        features[row, length:] = float("nan")
    return features, scores, lengths, mask


def nonzero_backbone(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.03, .05, 112).reshape(4, 28))
        model.output.bias.copy_(torch.tensor([.01, -.02, .03, -.01]))
    return model


def same_base(actual, expected):
    for name in ("prediction", "prior", "prior_mask"):
        assert torch.equal(bits(getattr(actual, name)), bits(getattr(expected, name)))
    assert actual.carry.kind == expected.carry.kind
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert torch.equal(bits(getattr(actual.carry, name)), bits(getattr(expected.carry, name)))


@pytest.mark.parametrize("mode", M.MODES)
def test_seeded_state_zero_residual_parameter_counts_and_rng_are_preserved(mode):
    before = torch.random.get_rng_state().clone()
    model, parent = M.make_head(mode, 61), raw.make_head(mode, 61)
    assert torch.equal(before, torch.random.get_rng_state())
    assert type(model.action_residual) is M.ScaledResidual
    assert set(model.state_dict()) == set(parent.state_dict())
    for name, value in parent.state_dict().items():
        assert torch.equal(bits(model.state_dict()[name]), bits(value))
    assert sum(p.numel() for p in model.parameters()) == M.parameter_count(mode) == 6112
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (
        M.parameter_count(mode, trainable_only=True)) == (116 if mode == "frozen" else 6112)
    nonzero_backbone(model); nonzero_backbone(parent)
    with torch.no_grad() if mode == "frozen" else nullcontext():
        expected = parent(*inputs(), episode_ends=torch.ones(3, dtype=torch.bool))
        actual = model(*inputs(), episode_ends=torch.ones(3, dtype=torch.bool))
    same_base(actual, expected)
    assert torch.equal(bits(actual.action_prediction), bits(expected.prediction))


def test_scaled_linear_values_and_gradients_are_exactly_raw_linear_times64():
    layer = M.ScaledResidual(28, 4, device="cpu", dtype=torch.float32)
    reference = torch.nn.Linear(28, 4, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-.03, .04, 112).reshape(4, 28))
        layer.bias.copy_(torch.tensor([.02, -.01, .03, -.04]))
    reference.load_state_dict(layer.state_dict(), strict=True)
    hidden = torch.linspace(-.4, .7, 2 * 3 * 28).reshape(2, 3, 28).requires_grad_(True)
    other_hidden = hidden.detach().clone().requires_grad_(True)
    actual, raw_value = layer(hidden), reference(other_hidden)
    assert torch.equal(bits(actual), bits(64 * raw_value))
    coefficients = torch.tensor([1., 2., 4., 8.])
    (actual * coefficients).sum().backward()
    (raw_value * coefficients).sum().backward()
    for scaled_gradient, raw_gradient in ((hidden.grad, other_hidden.grad),
                                          (layer.weight.grad, reference.weight.grad),
                                          (layer.bias.grad, reference.bias.grad)):
        assert scaled_gradient is not None and bool((scaled_gradient != 0).any())
        assert torch.equal(bits(scaled_gradient), bits(64 * raw_gradient))


@pytest.mark.parametrize("mode", M.MODES)
def test_strict_pretrained_copy_is_owned_zero_residual_and_matches_original(mode):
    reference = nonzero_backbone(original.make_head(raw.KIND, 71))
    reference.requires_grad_(mode == "joint")
    state = reference.state_dict()
    before = {name: value.clone() for name, value in state.items()}
    rng = torch.random.get_rng_state().clone()
    model = M.from_pretrained(mode, 72, state)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(state) == set(M.BACKBONE_SHAPES)
    assert sum(v.numel() for v in state.values()) == 5996
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        assert torch.equal(bits(parameters[name]), bits(value))
        assert parameters[name].data_ptr() != value.data_ptr()
        assert torch.equal(bits(value), bits(before[name]))
        assert parameters[name].requires_grad is (mode == "joint")
    assert bool((model.action_residual.weight == 0).all()) and bool((model.action_residual.bias == 0).all())
    assert model.action_residual.weight.requires_grad and model.action_residual.bias.requires_grad
    packet, ends = inputs(), torch.ones(3, dtype=torch.bool)
    with torch.no_grad() if mode == "frozen" else nullcontext():
        actual = model(*packet, episode_ends=ends)
        expected = reference(*packet, episode_ends=ends)
    same_base(actual, expected)
    assert torch.equal(bits(actual.action_prediction), bits(expected.prediction))
    with torch.no_grad():
        state["output.bias"].add_(100)
    assert torch.equal(bits(model.output.bias), bits(before["output.bias"]))


@pytest.mark.parametrize("defect", ["missing", "extra", "residual", "shape", "dtype", "nan", "inf", "not_tensor", "not_mapping"])
def test_invalid_backbone_state_rejected_before_model_construction(defect, monkeypatch):
    state = {name: torch.zeros(shape) for name, shape in M.BACKBONE_SHAPES.items()}
    if defect == "missing":
        state.pop("output.bias")
    elif defect == "extra":
        state["unknown"] = torch.zeros(1)
    elif defect == "residual":
        state["action_residual.weight"] = torch.zeros(4, 28)
    elif defect == "shape":
        state["output.bias"] = torch.zeros(1, 4)
    elif defect == "dtype":
        state["output.bias"] = torch.zeros(4, dtype=torch.float64)
    elif defect in ("nan", "inf"):
        state["output.bias"][0] = float(defect)
    elif defect == "not_tensor":
        state["output.bias"] = [0., 0., 0., 0.]
    else:
        state = list(state.items())
    monkeypatch.setattr(M, "make_head", lambda *_: pytest.fail("invalid state must not construct a model"))
    with pytest.raises(ValueError):
        M.from_pretrained("frozen", 1, state)


@pytest.mark.parametrize("mode", M.MODES)
def test_scaled_residual_has_no_feedback_and_retains_mode_gradient_locks(mode):
    reference = nonzero_backbone(original.make_head(raw.KIND, 81))
    model = M.from_pretrained(mode, 82, reference.state_dict())
    packet, ends = inputs(), torch.ones(3, dtype=torch.bool)
    before = model(*packet, episode_ends=ends)
    with torch.no_grad():
        model.action_residual.weight.copy_(torch.linspace(-.01, .02, 112).reshape(4, 28))
        model.action_residual.bias.copy_(torch.tensor([.01, -.02, .03, -.01]))
    after = model(*packet, episode_ends=ends)
    same_base(after, before)
    active = torch.arange(9)[None, :] < packet[2][:, None]
    nonquery = active & ~packet[3]
    assert bool((after.action_prediction[nonquery] != before.action_prediction[nonquery]).any())
    assert torch.equal(bits(after.action_prediction[~nonquery]), bits(after.prediction[~nonquery]))
    (after.action_prediction[nonquery] * torch.tensor([1., 2., 4., 8.])).square().mean().backward()
    for name, parameter in model.named_parameters():
        expected_trainable = mode == "joint" or name.startswith("action_residual.")
        assert parameter.requires_grad is expected_trainable
        if expected_trainable:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            assert bool((parameter.grad != 0).any())
        else:
            assert parameter.grad is None
    assert model._captured is None and model._action_hidden is None and not model._action_lock.locked()


@pytest.mark.parametrize("which", ["adapter", "prequery", "cross_query"])
def test_immutable_parent_pins_are_required(which, monkeypatch):
    if which == "adapter":
        monkeypatch.setattr(M, "PARENT_SOURCE_SHA256", "0" * 64)
    else:
        field = "PREQUERY_SOURCE_SHA256" if which == "prequery" else "CROSS_QUERY_SOURCE_SHA256"
        monkeypatch.setattr(raw, field, "0" * 64)
    with pytest.raises(ValueError, match="source pin"):
        M.make_head("frozen", 1)


def test_inherited_capture_forward_is_not_overridden():
    assert M.ProtectedTrainingModel.forward is raw.ProtectedReadout.forward
    assert M.ProtectedTrainingModel._prediction is raw.ProtectedReadout._prediction
    assert M.Forecast is raw.Forecast and M.detach_carry is raw.detach_carry
