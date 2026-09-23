"""Fabricated tensors only; no files, teacher, environment or recorded data."""
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.nn import functional as F

from openjev.research import otto_cross_query_scores as original
from openjev.research import otto_prequery_scores as M


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


def nonzero(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.04, .06, model.output.weight.numel()).reshape_as(model.output.weight))
        model.output.bias.copy_(torch.tensor([.015, -.02, .03, -.01]))
        if model.kind == "innovation":
            model.correction.weight.copy_(torch.linspace(-.3, .4, model.correction.weight.numel()).reshape_as(model.correction.weight))
    return model


def bits(value):
    return value.contiguous().view(torch.uint8)


def same_carry(a, b):
    assert a.kind == b.kind
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert torch.equal(bits(getattr(a, name)), bits(getattr(b, name)))


def center(value):
    return value - value.mean(dim=-1, keepdim=True)


def scalar_step(cell, value, hidden):
    """Independent GRU equations for the first later-query prior witness."""
    ir, iz, inn = F.linear(value, cell.weight_ih_l0, cell.bias_ih_l0).chunk(3, -1)
    hr, hz, hn = F.linear(hidden, cell.weight_hh_l0, cell.bias_hh_l0).chunk(3, -1)
    reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    return (1 - update) * torch.tanh(inn + reset * hn) + update * hidden


@pytest.mark.parametrize("kind", M.KINDS)
def test_same_initialization_outputs_calls_and_nonquery_gradients(kind):
    rng = torch.random.get_rng_state().clone()
    model, old = M.make_head(kind, 51), original.make_head(kind, 51)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert sum(p.numel() for p in model.parameters()) == M.parameter_count(kind) == original.parameter_count(kind)
    assert set(model.state_dict()) == set(old.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(bits(value), bits(old.state_dict()[name]))
    nonzero(model); nonzero(old)
    packet = inputs()
    ends = torch.ones(3, dtype=torch.bool)
    calls, old_calls = [], []

    def observe(log, label):
        return lambda _module, args, _output: log.append((label, tuple(args[0].shape)))

    handles = [model.recurrent.register_forward_hook(observe(calls, "gru")),
               model.output.register_forward_hook(observe(calls, "readout")),
               old.recurrent.register_forward_hook(observe(old_calls, "gru")),
               old.output.register_forward_hook(observe(old_calls, "readout"))]
    try:
        new = model(*packet, episode_ends=ends)
        expected, old_carry = old(*packet, episode_ends=ends)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == old_calls and len(calls) > 0
    assert torch.equal(bits(new.prediction), bits(expected))
    same_carry(new.carry, old_carry)
    objective_weights = torch.arange(1, 5, dtype=torch.float32)
    ((new.prediction * objective_weights).square().sum() + new.carry.hidden.square().sum()).backward()
    ((expected * objective_weights).square().sum() + old_carry.hidden.square().sum()).backward()
    for a, b in zip(model.parameters(), old.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(bits(a.grad), bits(b.grad))
    assert model._captured is None


@pytest.mark.parametrize("kind", M.KINDS)
def test_mask_short_lanes_initial_hold_and_independent_prior_oracle(kind):
    model = M.make_head(kind, 17)
    features, scores, lengths, mask = inputs((9, 5, 1))
    with torch.no_grad():
        result = model(features, scores, lengths, mask, episode_ends=torch.ones(3, dtype=torch.bool))
    expected_mask = mask.clone(); expected_mask[:, 0] = False
    assert torch.equal(result.prior_mask, expected_mask)
    assert torch.equal(bits(result.prior[~expected_mask]), bits(torch.zeros_like(result.prior[~expected_mask])))
    for row, step in torch.nonzero(expected_mask).tolist():
        assert torch.equal(result.prior[row, step], scores[row, step - 4])
    assert torch.equal(bits(result.prediction[mask]), bits(scores[mask]))
    nonzero(model)
    with torch.no_grad():
        first = model(features[:1, :4], scores[:1, :4], torch.tensor([4]), mask[:1, :4],
                      episode_ends=torch.tensor([False]))
        value = [features[0, 4], center(first.carry.raw_anchor[0] / 64)]
        if kind == "innovation_gru":
            value += [torch.zeros(4), torch.zeros(1)]
        h = scalar_step(model.recurrent, torch.cat(value), first.carry.hidden[0])
        expected = (first.carry.raw_anchor[0] / 64 + center(F.linear(h, model.output.weight, model.output.bias))) * 64
        later = model(features[:1, 4:5], scores[:1, 4:5], torch.tensor([1]), mask[:1, 4:5],
                      carry=M.detach_carry(first.carry), episode_ends=torch.tensor([True]))
    torch.testing.assert_close(later.prior[0, 0], expected, rtol=2e-6, atol=2e-5)
    assert bool(later.prior_mask[0, 0])


@pytest.mark.parametrize("kind", M.KINDS)
def test_prior_is_causal_to_query_information_and_ignores_poison(kind):
    model = nonzero(M.make_head(kind, 41))
    features, scores, lengths, mask = inputs((13, 5))
    ends = torch.ones(2, dtype=torch.bool)
    originals = [value.clone() for value in (features, scores, lengths, mask)]
    with torch.no_grad():
        initial = model(features, scores, lengths, mask, episode_ends=ends)
        changed = scores.clone(); changed[0, 4] += torch.tensor([5., -3., 1., 2.])
        current = model(features, changed, lengths, mask, episode_ends=ends)
        assert torch.equal(initial.prior[:, :5], current.prior[:, :5])
        assert not torch.equal(initial.prediction[0, 4], current.prediction[0, 4])
        assert not torch.equal(initial.prior[0, 8], current.prior[0, 8])
        changed = scores.clone(); changed[0, 8] += torch.tensor([-2., 4., 1., -1.])
        future_features = features.clone(); future_features[0, 9:, :10] += 2
        future = model(future_features, changed, lengths, mask, episode_ends=ends)
        assert torch.equal(initial.prior[:, :9], future.prior[:, :9])
        changed = scores.clone(); changed[0, 0] += torch.tensor([3., -2., 4., 1.])
        earlier = model(features, changed, lengths, mask, episode_ends=ends)
        assert not torch.equal(initial.prior[0, 4], earlier.prior[0, 4])
        poison = scores.clone(); poison[~mask] = float("inf")
        poisoned_features = features.clone(); poisoned_features[1, 5:] = float("inf")
        ignored = model(poisoned_features, poison, lengths, mask, episode_ends=ends)
        assert torch.equal(initial.prediction, ignored.prediction)
        assert torch.equal(initial.prior, ignored.prior)
        same_carry(initial.carry, ignored.carry)
    for value, saved in zip((features, scores, lengths, mask), originals, strict=True):
        assert torch.equal(bits(value), bits(saved))


@pytest.mark.parametrize("kind", M.KINDS)
def test_chunk_mapping_matches_whole_history_and_owns_ended_state(kind):
    model = nonzero(M.make_head(kind, 73))
    features, scores, totals, mask = inputs((13, 9, 1))
    carry, predictions, priors, masks = None, [], [], []
    with torch.no_grad():
        whole = model(features, scores, totals, mask, episode_ends=torch.ones(3, dtype=torch.bool))
        for start in range(0, 13, 4):
            stop = min(start + 4, 13)
            lengths = (totals - start).clamp(0, stop - start)
            result = model(features[:, start:stop], scores[:, start:stop], lengths, mask[:, start:stop],
                           carry=carry, episode_ends=totals <= stop)
            predictions.append(result.prediction); priors.append(result.prior); masks.append(result.prior_mask)
            if carry is not None:
                assert torch.equal(result.carry.hidden[2], carry.hidden[2])
                assert torch.equal(result.carry.raw_anchor[2], carry.raw_anchor[2])
            carry = M.detach_carry(result.carry)
        torch.testing.assert_close(torch.cat(predictions, 1), whole.prediction, rtol=0, atol=2e-6)
        torch.testing.assert_close(torch.cat(priors, 1), whole.prior, rtol=0, atol=2e-6)
        torch.testing.assert_close(carry.hidden, whole.carry.hidden, rtol=0, atol=2e-6)
    assert torch.equal(torch.cat(masks, 1), whole.prior_mask)
    assert torch.equal(carry.absolute_step, totals)
    detached = M.detach_carry(whole.carry)
    assert detached.hidden.data_ptr() != whole.carry.hidden.data_ptr()
    assert detached.raw_anchor.data_ptr() != whole.carry.raw_anchor.data_ptr()
    detached.hidden.fill_(123); detached.raw_anchor.fill_(-99)
    assert not bool((whole.carry.hidden == 123).all())
    assert not bool((whole.carry.raw_anchor == -99).all())
    assert not bool((scores[mask] == -99).any())
    with pytest.raises(FrozenInstanceError):
        whole.prior = torch.zeros_like(whole.prior)


@pytest.mark.parametrize("kind", M.KINDS)
def test_prior_loss_reaches_existing_parameters_but_not_current_query(kind):
    model = nonzero(M.make_head(kind, 29))
    features, scores, lengths, mask = inputs((9,))
    scores.requires_grad_(True)
    forecast = model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    # Only t8 is scored: Q8 has not yet entered that prior; Q0/Q4 have.
    target = torch.tensor([[14., 12., 17., 11.]])
    (center(forecast.prior[:, 8] / 64) - center(target / 64)).square().mean().backward()
    parameters = [model.output.weight, model.recurrent.weight_ih_l0, model.recurrent.weight_hh_l0]
    if kind == "innovation":
        parameters.append(model.correction.weight)
    for parameter in parameters:
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0).any())
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())
    assert bool((scores.grad[:, :8] != 0).any())
    assert torch.equal(scores.grad[:, 8], torch.zeros_like(scores.grad[:, 8]))
    assert torch.equal(scores.grad[~mask], torch.zeros_like(scores.grad[~mask]))
    assert model._captured is None


@pytest.mark.parametrize("kind", M.KINDS)
def test_capture_failure_and_reentry_leave_no_hidden_state(kind):
    model = nonzero(M.make_head(kind, 91))
    packet = inputs((8,)); ends = torch.tensor([True])
    seen = []

    def fail_after_prior(_module, _inputs, _output):
        seen.append(1)
        if len(seen) == 3:
            assert model._captured is not None and len(model._captured) == 1
            raise RuntimeError("injected readout failure")

    handle = model.output.register_forward_hook(fail_after_prior)
    try:
        with pytest.raises(RuntimeError, match="injected readout"):
            model(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert model._captured is None

    def reenter(_module, _inputs, _output):
        model(*packet, episode_ends=ends)

    handle = model.output.register_forward_hook(reenter)
    try:
        with pytest.raises(ValueError, match="cannot be reentered"):
            model(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert model._captured is None
    with torch.no_grad():
        first = model(*packet, episode_ends=ends)
        second = model(*packet, episode_ends=ends)
    assert torch.equal(first.prediction, second.prediction) and torch.equal(first.prior, second.prior)
    same_carry(first.carry, second.carry)


@pytest.mark.parametrize("kind", ("persistent_direct", "reset_direct", "unknown"))
def test_only_declared_families_are_admitted(kind):
    with pytest.raises(ValueError, match="declared"):
        M.make_head(kind, 1)


@pytest.mark.parametrize("defect", ("future_query_nan", "wrong_mask", "wrong_dtype"))
def test_inherited_validation_and_capture_cleanup(defect):
    model = M.make_head("innovation", 7)
    features, scores, lengths, mask = inputs((8,))
    if defect == "future_query_nan":
        scores[0, 4, 0] = float("nan")
    elif defect == "wrong_mask":
        mask[0, 1] = True
    else:
        features = features.double()
    with pytest.raises(ValueError):
        model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    assert model._captured is None
