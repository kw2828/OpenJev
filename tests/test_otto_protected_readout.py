"""Fabricated engineering tensors only; no empirical data or optimizer steps."""
from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import pytest
import torch

from openjev.research import otto_prequery_scores as original
from openjev.research import otto_protected_readout as M


def inputs(totals=(13, 5, 1)):
    batch, span = len(totals), max(totals)
    features = torch.linspace(-.3, .8, batch * span * 31, dtype=torch.float32).reshape(batch, span, 31)
    scores = torch.full((batch, span, 4), float("nan"), dtype=torch.float32)
    mask = torch.zeros(batch, span, dtype=torch.bool)
    for row, length in enumerate(totals):
        for step in range(length):
            features[row, step, 15] = step / 2188
            features[row, step, 16] = (step % 4) / 2188
            features[row, step, 17] = 1
            if step % 4 == 0:
                scores[row, step] = torch.tensor([12., 15., 11., 18.]) + row + step * .125
                mask[row, step] = True
        features[row, length:] = float("nan")
    return features, scores, torch.tensor(totals, dtype=torch.int64), mask


def bits(value):
    return value.contiguous().view(torch.uint8)


def same_carry(actual, expected):
    assert actual.kind == expected.kind
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert torch.equal(bits(getattr(actual, name)), bits(getattr(expected, name)))


def same_base(actual, expected):
    for name in ("prediction", "prior", "prior_mask"):
        assert torch.equal(bits(getattr(actual, name)), bits(getattr(expected, name)))
    same_carry(actual.carry, expected.carry)


def nonzero_backbone(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.04, .06, model.output.weight.numel()).reshape_as(model.output.weight))
        model.output.bias.copy_(torch.tensor([.015, -.02, .03, -.01]))
    return model


def nonzero_residual(model):
    with torch.no_grad():
        model.action_residual.weight.copy_(torch.linspace(-.7, .9, 112).reshape(4, 28))
        model.action_residual.bias.copy_(torch.tensor([.4, -.2, .8, -.3]))
    return model


@pytest.mark.parametrize("mode", M.MODES)
def test_exact_held_initialization_forecasts_parameter_counts_and_rng(mode):
    rng = torch.random.get_rng_state().clone()
    model, old = M.make_head(mode, 51), original.make_head(M.KIND, 51)
    old.requires_grad_(mode == "joint")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == M.parameter_count(mode) == 6112
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == M.parameter_count(mode, trainable_only=True)
    assert M.parameter_count(mode, trainable_only=True) == (116 if mode == "frozen" else 6112)
    assert set(model.state_dict()) == set(old.state_dict()) | {"action_residual.weight", "action_residual.bias"}
    for name, value in old.state_dict().items():
        assert torch.equal(bits(value), bits(model.state_dict()[name]))
    assert torch.equal(model.action_residual.weight, torch.zeros(4, 28))
    assert torch.equal(model.action_residual.bias, torch.zeros(4))
    nonzero_backbone(model); nonzero_backbone(old)
    packet, ends = inputs(), torch.ones(3, dtype=torch.bool)
    calls, old_calls = [], []

    def observe(log, label):
        return lambda _module, args, _output: log.append((label, tuple(args[0].shape)))

    hooks = [model.recurrent.register_forward_hook(observe(calls, "gru")),
             model.output.register_forward_hook(observe(calls, "readout")),
             old.recurrent.register_forward_hook(observe(old_calls, "gru")),
             old.output.register_forward_hook(observe(old_calls, "readout"))]
    try:
        actual = model(*packet, episode_ends=ends)
        with torch.no_grad() if mode == "frozen" else nullcontext():
            expected = old(*packet, episode_ends=ends)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == old_calls and len(calls) > 0
    same_base(actual, expected)
    assert torch.equal(bits(actual.action_prediction), bits(expected.prediction))
    assert actual.action_prediction.data_ptr() != actual.prediction.data_ptr()
    assert model._action_hidden is None and model._captured is None
    assert not model._action_lock.locked()


@pytest.mark.parametrize("mode", M.MODES)
def test_residual_matches_independent_linear_oracle_and_never_changes_base(mode):
    model = nonzero_residual(nonzero_backbone(M.make_head(mode, 73)))
    old = nonzero_backbone(original.make_head(M.KIND, 73))
    old.requires_grad_(mode == "joint")
    packet, ends = inputs((4, 4)), torch.ones(2, dtype=torch.bool)
    captured = []

    def hidden(_module, args, _output):
        if args[0].ndim == 3:
            captured.append(args[0].detach().clone())

    handle = old.output.register_forward_hook(hidden)
    try:
        with torch.no_grad() if mode == "frozen" else nullcontext():
            expected = old(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (2, 3, 28)
    actual = model(*packet, episode_ends=ends)
    same_base(actual, expected)
    raw = torch.einsum("bth,ah->bta", captured[0], model.action_residual.weight.detach())
    raw += model.action_residual.bias.detach()
    residual = raw - raw.mean(-1, keepdim=True)
    torch.testing.assert_close(actual.action_prediction[:, 1:], expected.prediction[:, 1:] + residual,
                               rtol=1e-6, atol=2e-6)
    assert bool((actual.action_prediction[:, 1:] != actual.prediction[:, 1:]).any())
    assert torch.equal(bits(actual.action_prediction[:, 0]), bits(packet[1][:, 0]))
    torch.testing.assert_close(residual.mean(-1), torch.zeros(2, 3), rtol=0, atol=2e-7)


@pytest.mark.parametrize("residual_is_nonzero", [False, True])
def test_each_mode_preserves_equivalent_original_and_its_cross_mode_difference(residual_is_nonzero):
    frozen = nonzero_backbone(M.make_head("frozen", 43))
    if residual_is_nonzero:
        nonzero_residual(frozen)
    joint = M.make_head("joint", 44)
    joint.load_state_dict(frozen.state_dict(), strict=True)
    for name, value in frozen.state_dict().items():
        assert torch.equal(bits(value), bits(joint.state_dict()[name]))
    original_frozen = original.make_head(M.KIND, 45)
    original_joint = original.make_head(M.KIND, 46)
    backbone_state = {name: value for name, value in frozen.state_dict().items()
                      if not name.startswith("action_residual.")}
    original_frozen.load_state_dict(backbone_state, strict=True)
    original_joint.load_state_dict(backbone_state, strict=True)
    original_frozen.requires_grad_(False)
    original_joint.requires_grad_(True)
    for adapter, reference in ((frozen, original_frozen), (joint, original_joint)):
        for name, parameter in reference.named_parameters():
            equivalent = dict(adapter.named_parameters())[name]
            assert torch.equal(bits(parameter), bits(equivalent))
            assert parameter.requires_grad is equivalent.requires_grad
    packet, ends = inputs((13, 10, 7, 2)), torch.ones(4, dtype=torch.bool)
    with torch.no_grad():
        frozen_result = frozen(*packet, episode_ends=ends)
        joint_result = joint(*packet, episode_ends=ends)
        frozen_reference = original_frozen(*packet, episode_ends=ends)
        joint_reference = original_joint(*packet, episode_ends=ends)
    same_base(frozen_result, frozen_reference)
    same_base(joint_result, joint_reference)
    # Same weights may differ across parameter flags even under common no_grad.
    # Preserve exactly the original difference instead of imposing a tolerance
    # or assuming that any particular runtime must produce a nonzero difference.
    for name in ("prediction", "prior"):
        adapter_delta = getattr(frozen_result, name) - getattr(joint_result, name)
        original_delta = getattr(frozen_reference, name) - getattr(joint_reference, name)
        assert torch.equal(bits(adapter_delta), bits(original_delta))
    for name in ("hidden", "raw_anchor", "absolute_step"):
        adapter_delta = getattr(frozen_result.carry, name) - getattr(joint_result.carry, name)
        original_delta = getattr(frozen_reference.carry, name) - getattr(joint_reference.carry, name)
        assert torch.equal(bits(adapter_delta), bits(original_delta))
    for owner, name in (("forecast", "prior_mask"), ("carry", "has_query"), ("carry", "ended")):
        values = [row if owner == "forecast" else row.carry
                  for row in (frozen_result, joint_result, frozen_reference, joint_reference)]
        assert torch.equal(getattr(values[0], name) ^ getattr(values[1], name),
                           getattr(values[2], name) ^ getattr(values[3], name))
    if not residual_is_nonzero:
        assert torch.equal(bits(frozen_result.action_prediction), bits(frozen_reference.prediction))
        assert torch.equal(bits(joint_result.action_prediction), bits(joint_reference.prediction))
    assert not frozen_result.action_prediction.requires_grad and not joint_result.action_prediction.requires_grad


@pytest.mark.parametrize("mode", M.MODES)
def test_large_residual_cannot_feedback_into_future_prior_carry_or_queries(mode):
    model = nonzero_backbone(M.make_head(mode, 91))
    packet, ends = inputs((13, 9, 1)), torch.ones(3, dtype=torch.bool)
    before = model(*packet, episode_ends=ends)
    nonzero_residual(model)
    with torch.no_grad():
        model.action_residual.weight.mul_(100)
        model.action_residual.bias.mul_(100)
    after = model(*packet, episode_ends=ends)
    same_base(after, before)
    lengths, query = packet[2:]
    active = torch.arange(13)[None, :] < lengths[:, None]
    nonquery = active & ~query
    assert bool((after.action_prediction[nonquery] != before.action_prediction[nonquery]).any())
    assert torch.equal(bits(after.action_prediction[query]), bits(before.prediction[query]))
    assert torch.equal(bits(after.action_prediction[~active]), bits(torch.zeros_like(after.action_prediction[~active])))
    assert torch.equal(bits(after.action_prediction[~nonquery]), bits(after.prediction[~nonquery]))


@pytest.mark.parametrize("mode", M.MODES)
def test_chunked_and_whole_forecasts_match_with_short_and_ended_lanes(mode):
    model = nonzero_residual(nonzero_backbone(M.make_head(mode, 17)))
    features, scores, totals, mask = inputs((13, 9, 1))
    carry, outputs = None, []
    with torch.no_grad():
        whole = model(features, scores, totals, mask, episode_ends=torch.ones(3, dtype=torch.bool))
        for start in range(0, 13, 4):
            stop = min(start + 4, 13)
            lengths = (totals - start).clamp(0, stop - start)
            result = model(features[:, start:stop], scores[:, start:stop], lengths, mask[:, start:stop],
                           carry=carry, episode_ends=totals <= stop)
            if carry is not None:
                assert torch.equal(result.carry.hidden[2], carry.hidden[2])
                assert torch.equal(result.carry.raw_anchor[2], carry.raw_anchor[2])
            outputs.append(result)
            carry = M.detach_carry(result.carry)
    for name in ("prediction", "action_prediction", "prior"):
        torch.testing.assert_close(torch.cat([getattr(row, name) for row in outputs], 1), getattr(whole, name),
                                   rtol=0, atol=3e-6)
    assert torch.equal(torch.cat([row.prior_mask for row in outputs], 1), whole.prior_mask)
    torch.testing.assert_close(carry.hidden, whole.carry.hidden, rtol=0, atol=2e-6)
    assert torch.equal(carry.absolute_step, totals)


@pytest.mark.parametrize("mode", M.MODES)
def test_mixed_active_length_groups_match_each_lane_run_alone(mode):
    model = nonzero_residual(nonzero_backbone(M.make_head(mode, 23)))
    features, scores, lengths, mask = inputs((13, 10, 7, 2))
    together = model(features, scores, lengths, mask, episode_ends=torch.ones(4, dtype=torch.bool))
    for row, length in enumerate(lengths.tolist()):
        alone = model(features[row:row+1, :length], scores[row:row+1, :length],
                      torch.tensor([length]), mask[row:row+1, :length], episode_ends=torch.tensor([True]))
        for name in ("prediction", "prior", "action_prediction"):
            torch.testing.assert_close(getattr(together, name)[row:row+1, :length], getattr(alone, name),
                                       rtol=2e-6, atol=8e-6)
        assert torch.equal(together.prior_mask[row:row+1, :length], alone.prior_mask)


def test_frozen_action_loss_reaches_only_residual_even_with_differentiable_inputs():
    model = nonzero_residual(nonzero_backbone(M.make_head("frozen", 29)))
    features, scores, lengths, mask = inputs((9,))
    features.requires_grad_(True); scores.requires_grad_(True)
    result = model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    weights = torch.tensor([1., 2., 4., 8.])
    (result.action_prediction[~mask] * weights).square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("action_residual."):
            assert parameter.requires_grad and parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all()) and bool((parameter.grad != 0).any())
        else:
            assert not parameter.requires_grad and parameter.grad is None
    assert features.grad is None and scores.grad is None
    assert not result.prediction.requires_grad and not result.prior.requires_grad
    assert not result.carry.hidden.requires_grad


def test_joint_action_loss_reaches_backbone_and_residual_but_prior_loss_skips_residual():
    model = nonzero_residual(nonzero_backbone(M.make_head("joint", 41)))
    packet = inputs((9,))
    result = model(*packet, episode_ends=torch.tensor([True]))
    (result.action_prediction[~packet[3]] * torch.tensor([1., 2., 4., 8.])).square().mean().backward()
    for name in ("recurrent.weight_ih_l0", "recurrent.weight_hh_l0", "output.weight",
                 "action_residual.weight", "action_residual.bias"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any())
    model.zero_grad(set_to_none=True)
    result = model(*packet, episode_ends=torch.tensor([True]))
    (result.prior[:, 8] * torch.tensor([1., 2., 4., 8.])).square().mean().backward()
    assert model.output.weight.grad is not None and bool((model.output.weight.grad != 0).any())
    assert model.action_residual.weight.grad is None and model.action_residual.bias.grad is None


@pytest.mark.parametrize("mode", M.MODES)
def test_future_query_changes_cannot_change_earlier_actions_and_poison_is_ignored(mode):
    model = nonzero_residual(nonzero_backbone(M.make_head(mode, 7)))
    features, scores, lengths, mask = inputs((13, 5))
    ends = torch.ones(2, dtype=torch.bool)
    saved = [value.clone() for value in (features, scores, lengths, mask)]
    before = model(features, scores, lengths, mask, episode_ends=ends)
    changed = scores.clone(); changed[0, 8] += torch.tensor([3., -1., 2., -2.])
    future = model(features, changed, lengths, mask, episode_ends=ends)
    assert torch.equal(bits(before.action_prediction[:, :8]), bits(future.action_prediction[:, :8]))
    assert torch.equal(bits(before.prior[:, :9]), bits(future.prior[:, :9]))
    poisoned_scores = scores.clone(); poisoned_scores[~mask] = float("inf")
    poisoned_features = features.clone(); poisoned_features[1, 5:] = float("inf")
    poisoned = model(poisoned_features, poisoned_scores, lengths, mask, episode_ends=ends)
    same_base(poisoned, before)
    assert torch.equal(bits(poisoned.action_prediction), bits(before.action_prediction))
    for value, original_value in zip((features, scores, lengths, mask), saved, strict=True):
        assert torch.equal(bits(value), bits(original_value))


@pytest.mark.parametrize("mode", M.MODES)
def test_current_input_validation_and_guard_cleanup(mode):
    model = M.make_head(mode, 11)
    features, scores, lengths, mask = inputs((8,))
    scores[0, 4, 0] = float("nan")
    with pytest.raises(ValueError, match="finite query scores"):
        model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    assert model._captured is None and model._action_hidden is None and not model._action_lock.locked()
    packet = inputs((8,))
    handle = model.action_residual.register_forward_hook(
        lambda *_: (_ for _ in ()).throw(RuntimeError("fabricated residual failure")))
    try:
        with pytest.raises(RuntimeError, match="residual failure"):
            model(*packet, episode_ends=torch.tensor([True]))
    finally:
        handle.remove()
    assert model._captured is None and model._action_hidden is None and not model._action_lock.locked()
    model(*packet, episode_ends=torch.tensor([True]))


@pytest.mark.parametrize("defect", ["mask", "dtype", "chunk_end", "attached_carry", "residual_dtype"])
def test_inherited_validation_is_not_relaxed_by_frozen_mode(defect):
    model = M.make_head("frozen", 31)
    features, scores, lengths, mask = inputs((8,))
    carry, ends = None, torch.tensor([True])
    if defect == "mask":
        mask[0, 1] = True
    elif defect == "dtype":
        features = features.double()
    elif defect == "chunk_end":
        lengths = torch.tensor([7]); mask[0, 7:] = False
        ends = torch.tensor([False])
    elif defect == "attached_carry":
        carry = model.initial_carry(1)
        carry.hidden.requires_grad_(True)
    else:
        model.action_residual.double()
    with pytest.raises(ValueError):
        model(features, scores, lengths, mask, carry=carry, episode_ends=ends)
    assert model._captured is None and model._action_hidden is None and not model._action_lock.locked()


def test_reentry_and_concurrent_lock_reject_without_disturbing_active_capture():
    model = M.make_head("joint", 17)
    packet, ends = inputs((8,)), torch.tensor([True])

    def reenter(*_):
        model(*packet, episode_ends=ends)

    handle = model.output.register_forward_hook(reenter)
    try:
        with pytest.raises(ValueError, match="cannot be reentered"):
            model(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert model._captured is None and model._action_hidden is None and not model._action_lock.locked()
    model._action_lock.acquire()
    sentinel = []
    model._action_hidden = sentinel
    try:
        with pytest.raises(ValueError, match="concurrently"):
            model(*packet, episode_ends=ends)
        assert model._action_hidden is sentinel and model._action_lock.locked()
    finally:
        model._action_hidden = None
        model._action_lock.release()
    model(*packet, episode_ends=ends)


@pytest.mark.parametrize("mode", M.MODES)
def test_readout_parameter_lock_changes_are_rejected(mode):
    model = M.make_head(mode, 13)
    model.output.weight.requires_grad_(mode == "frozen")
    with pytest.raises(ValueError, match="parameter locks"):
        model(*inputs((4,)), episode_ends=torch.tensor([True]))
    assert model._action_hidden is None and not model._action_lock.locked()


@pytest.mark.parametrize("pin", ["PREQUERY_SOURCE_SHA256", "CROSS_QUERY_SOURCE_SHA256"])
def test_changed_immutable_source_pin_fails_construction(pin, monkeypatch):
    monkeypatch.setattr(M, pin, "0" * 64)
    with pytest.raises(ValueError, match="base source pin"):
        M.make_head("frozen", 1)


@pytest.mark.parametrize("mode,seed", [("other", 1), ("frozen", -1), ("joint", 2**32), ("joint", True)])
def test_only_declared_mode_and_seed_admitted(mode, seed):
    with pytest.raises(ValueError, match="declared mode"):
        M.make_head(mode, seed)


def test_queries_only_do_not_call_residual_and_keep_owned_outputs():
    model = M.make_head("frozen", 19)
    packet = inputs((1,))
    packet[1][0, 0] = torch.tensor([-0., 0., 1., -1.])
    handle = model.action_residual.register_forward_hook(lambda *_: pytest.fail("no active nonquery readout"))
    try:
        result = model(*packet, episode_ends=torch.tensor([True]))
    finally:
        handle.remove()
    assert torch.equal(bits(result.action_prediction), bits(packet[1]))
    assert result.action_prediction.data_ptr() != result.prediction.data_ptr()
    with pytest.raises(FrozenInstanceError):
        result.action_prediction = torch.zeros_like(result.action_prediction)
