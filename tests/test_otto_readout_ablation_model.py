"""Fabricated readout-mask and state-causality checks; no empirical inputs."""
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_model as original
from openjev.research import otto_readout_ablation_model as ablation
from openjev.research import otto_scheduled_predictor as scheduled

ARMS = ("action_residual_only", "both_readouts", "full_joint")
EXPECTED = {
    "action_residual_only": ({"slow.action_residual.weight", "slow.action_residual.bias"}, 116),
    "both_readouts": ({"slow.output.weight", "slow.output.bias", "slow.action_residual.weight",
                       "slow.action_residual.bias"}, 232),
    "full_joint": ({"slow." + name for name in scheduled.STATE_SHAPES}, 6112),
}
FIELDS = ("prediction", "slow_action_prediction", "prior", "shadow_prior", "action_prediction",
          "corrected_shadow_prior", "prewrite_correction", "prior_mask", "key_mask")


def same_bits(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().contiguous().reshape(-1).view(torch.uint8))


def weights(seed=43):
    model = scheduled.make_head("joint", seed, 4)
    with torch.no_grad():
        model.output.weight.copy_(torch.arange(112, dtype=torch.float32).sin().reshape(4, 28) * .02)
        model.output.bias.copy_(torch.tensor([.1, -.2, .3, -.1]))
        model.action_residual.weight.copy_(torch.arange(112, dtype=torch.float32).cos().reshape(4, 28) * .003)
        model.action_residual.bias.copy_(torch.tensor([.04, -.03, .02, -.01]))
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def packet(lengths=(65, 37, 1)):
    batch, span = len(lengths), max(lengths)
    features = torch.full((batch, span, 31), float("nan"))
    scores = torch.full((batch, span, 4), float("nan"))
    query = torch.zeros(batch, span, dtype=torch.bool)
    for lane, length in enumerate(lengths):
        steps = torch.arange(length)
        features[lane, :length] = torch.linspace(-.4, .6, length * 31).reshape(length, 31) + lane / 10
        features[lane, :length, 15] = steps / 2188
        features[lane, :length, 16] = steps.remainder(4) / 2188
        features[lane, :length, 17] = 1
        query[lane, :length] = steps.remainder(4) == 0
        scores[lane, :length][query[lane, :length]] = (
            torch.tensor([2., 7., 4., 6.])
            + steps[query[lane, :length]][:, None] * torch.tensor([.125, -.25, .5, .25]))
        scores[lane, 0, 0] = -.0
    return {"features": features, "query_scores": scores, "query_mask": query,
            "lengths": torch.tensor(lengths), "episode_ends": torch.ones(batch, dtype=torch.bool)}


def chunk(values, start, stop):
    return {**{name: values[name][:, start:stop].clone()
               for name in ("features", "query_scores", "query_mask")},
            "lengths": (values["lengths"] - start).clamp(0, stop - start),
            "episode_ends": values["episode_ends"] & (values["lengths"] <= stop)}


def compare_carry(left, right):
    assert type(left) is type(right) is original.Carry
    assert left.slow.query_period == right.slow.query_period == 4
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        same_bits(getattr(left.slow.base, name), getattr(right.slow.base, name))
    for name in ("matrix", "trace", "last_error", "has_query", "absolute_step", "ended"):
        same_bits(getattr(left.fast, name), getattr(right.fast, name))


@pytest.mark.parametrize("arm", ARMS)
def test_factories_preserve_rng_exact_eight_tensors_and_independent_state(arm):
    before = torch.random.get_rng_state().clone()
    fresh = ablation.make_model(arm, seed=43)
    same_bits(before, torch.random.get_rng_state())
    seeded = scheduled.make_head("joint", 43, 4)
    for name, value in seeded.state_dict().items():
        same_bits(value, fresh.slow.state_dict()[name])
    supplied = weights()
    snapshot = {name: value.clone() for name, value in supplied.items()}
    copied = ablation.from_state(arm, 43, supplied, query_period=4)
    same_bits(before, torch.random.get_rng_state())
    assert copied.seed == 43 and copied.query_period == 4
    assert set(copied.state_dict()) == {"slow." + name for name in scheduled.STATE_SHAPES}
    assert copied.projection is None and copied.config.mode == "none"
    for name, value in copied.slow.state_dict().items():
        same_bits(value, supplied[name])
        assert value.data_ptr() != supplied[name].data_ptr()
    with torch.no_grad():
        copied.slow.action_residual.bias.add_(.2)
    for name in supplied:
        same_bits(supplied[name], snapshot[name])


@pytest.mark.parametrize("arm", ARMS)
def test_effective_parameter_masks_are_exact_and_match_actual_flags(arm):
    model = ablation.make_model(arm, seed=11)
    names, count = EXPECTED[arm]
    named = dict(model.named_parameters())
    effective = dict(model.effective_named_parameters())
    assert set(effective) == names
    assert all(effective[name] is named[name] for name in names)
    assert {name for name, value in named.items() if value.requires_grad} == names
    assert sum(value.numel() for value in effective.values()) == count
    assert len(named) == 8 and sum(value.numel() for value in named.values()) == 6112
    metadata = model.parameter_metadata()
    assert set(metadata["names"]) == set(named)
    assert metadata["count"] == 6112
    assert set(metadata["requires_grad_names"]) == set(metadata["effective_names"]) == names
    assert metadata["requires_grad_count"] == metadata["effective_count"] == count
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 for value in named.values())


@pytest.mark.parametrize("grad_enabled", (False, True))
def test_full_joint_exactly_matches_qualified_original_across_detached_chunks(grad_enabled):
    state = weights()
    candidate = ablation.from_state("full_joint", 43, state)
    reference = original.from_states(memory.Config("none", 8), 43, 4, state, None, slow_mode="joint")
    values = packet()
    saved = {name: value.clone() for name, value in values.items()}
    before = torch.random.get_rng_state().clone()
    candidate_carry = reference_carry = None
    with torch.set_grad_enabled(grad_enabled):
        for start, stop in ((0, 32), (32, 64), (64, 65)):
            inputs = chunk(values, start, stop)
            actual = candidate(**inputs, carry=candidate_carry)
            expected = reference(**inputs, carry=reference_carry)
            assert type(actual) is type(expected) is original.Forecast
            for name in FIELDS:
                same_bits(getattr(actual, name), getattr(expected, name))
            compare_carry(actual.carry, expected.carry)
            assert actual.work_counts == expected.work_counts
            assert actual.memory_work_units == expected.memory_work_units
            assert actual.parameter_metadata == expected.parameter_metadata
            candidate_carry = original.detach_carry(actual.carry)
            reference_carry = original.detach_carry(expected.carry)
    same_bits(before, torch.random.get_rng_state())
    for name in values:
        same_bits(values[name], saved[name])


def test_residual_only_perturbation_changes_readout_without_changing_any_recurrent_call():
    state = weights()
    baseline = ablation.from_state("action_residual_only", 43, state)
    changed = ablation.from_state("action_residual_only", 43, state)
    with torch.no_grad():
        changed.slow.action_residual.weight.add_(torch.linspace(-.1, .1, 112).reshape(4, 28))
        changed.slow.action_residual.bias.add_(torch.tensor([.3, -.2, .1, -.1]))
    captured = [[], []]
    handles = []
    for index, model in enumerate((baseline, changed)):
        def capture(_module, _inputs, result, slot=index):
            captured[slot].append(tuple(value.detach().clone() for value in result))
        handles.append(model.slow.recurrent.register_forward_hook(capture))
    try:
        values = packet((17, 10, 1))
        first, second = baseline(**values), changed(**values)
    finally:
        for handle in handles:
            handle.remove()
    assert captured[0] and len(captured[0]) == len(captured[1])
    for left, right in zip(captured[0], captured[1], strict=True):
        for x, y in zip(left, right, strict=True):
            same_bits(x, y)
    for name in ("prediction", "prior", "prior_mask", "key_mask"):
        same_bits(getattr(first, name), getattr(second, name))
    compare_carry(first.carry, second.carry)
    assert not torch.equal(first.action_prediction, second.action_prediction)
    assert not torch.equal(first.shadow_prior, second.shadow_prior)
    assert first.work_counts == second.work_counts


def test_both_readouts_frozen_gru_can_change_hidden_after_later_query_innovation():
    state = weights()
    baseline = ablation.from_state("both_readouts", 43, state)
    changed = ablation.from_state("both_readouts", 43, state)
    with torch.no_grad():
        changed.slow.output.bias.add_(torch.tensor([.4, -.3, .2, -.1]))
    for name, parameter in changed.slow.recurrent.named_parameters():
        assert parameter.requires_grad is False
        same_bits(parameter, dict(baseline.slow.recurrent.named_parameters())[name])
    values = packet((8,))
    first = baseline(**chunk(values, 0, 4))
    altered_first = changed(**chunk(values, 0, 4))
    compare_carry(first.carry, altered_first.carry)
    second = baseline(**chunk(values, 4, 8), carry=original.detach_carry(first.carry))
    altered_second = changed(**chunk(values, 4, 8), carry=original.detach_carry(altered_first.carry))
    assert not torch.equal(second.carry.slow.base.hidden, altered_second.carry.slow.base.hidden)
    assert not torch.equal(second.prior, altered_second.prior)
    same_bits(second.carry.slow.base.raw_anchor, altered_second.carry.slow.base.raw_anchor)
    assert second.work_counts == altered_second.work_counts


@pytest.mark.parametrize("arm", ARMS)
def test_query_signed_zero_padding_and_no_write_are_preserved(arm):
    model = ablation.from_state(arm, 43, weights())
    values = packet((13, 5, 1))
    expected = model(**values)
    actual = model(**values, no_write=True)
    for name in FIELDS:
        same_bits(getattr(actual, name), getattr(expected, name))
    compare_carry(actual.carry, expected.carry)
    same_bits(actual.action_prediction[values["query_mask"]], values["query_scores"][values["query_mask"]])
    active = torch.arange(13)[None, :] < values["lengths"][:, None]
    for name in ("prediction", "action_prediction", "prior", "shadow_prior"):
        same_bits(getattr(actual, name)[~active], torch.zeros_like(getattr(actual, name)[~active]))


@pytest.mark.parametrize("kwargs", ({"arm": "unknown"}, {"arm": None}, {"seed": True},
                                   {"seed": -1}, {"seed": 2**32}, {"query_period": True},
                                   {"query_period": 8}, {"query_period": 0}))
def test_invalid_configuration_rejected_without_changing_rng(kwargs):
    inputs = {"arm": "full_joint", "seed": 3, "query_period": 4, **kwargs}
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        ablation.make_model(**inputs)
    same_bits(before, torch.random.get_rng_state())


@pytest.mark.parametrize("defect", ("missing", "extra", "shape", "dtype", "nan"))
def test_invalid_supplied_state_rejected_without_rng_or_source_mutation(defect):
    supplied = weights()
    key = "output.bias"
    if defect == "missing":
        del supplied[key]
    elif defect == "extra":
        supplied["unregistered"] = torch.zeros(1)
    elif defect == "shape":
        supplied[key] = torch.zeros(5)
    elif defect == "dtype":
        supplied[key] = supplied[key].double()
    else:
        supplied[key][0] = float("nan")
    snapshot = {name: value.clone() for name, value in supplied.items()}
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        ablation.from_state("full_joint", 3, supplied)
    same_bits(before, torch.random.get_rng_state())
    for name in supplied:
        same_bits(supplied[name], snapshot[name])
