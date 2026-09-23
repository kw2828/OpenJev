"""Fabricated score histories only: no files, teacher, environment or saved fits."""
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.nn import functional as F

from openjev.research import otto_prequery_scores as shared
from openjev.research import otto_separate_prior_scores as M


def inputs(totals=(13, 5, 1)):
    batch, span = len(totals), max(totals)
    features = torch.linspace(-.3, .8, batch * span * 31, dtype=torch.float32).reshape(batch, span, 31)
    scores = torch.full((batch, span, 4), float("nan"), dtype=torch.float32)
    mask = torch.zeros(batch, span, dtype=torch.bool)
    for lane, length in enumerate(totals):
        for step in range(length):
            features[lane, step, 15] = step / 2188
            features[lane, step, 16] = (step % 4) / 2188
            features[lane, step, 17] = 1
            if step % 4 == 0:
                scores[lane, step] = torch.tensor([12., 15., 11., 18.]) + lane + step * .125
                mask[lane, step] = True
        features[lane, length:] = float("nan")
    return features, scores, torch.tensor(totals, dtype=torch.int64), mask


def bits(value):
    return value.contiguous().view(torch.uint8)


def center(value):
    return value - value.mean(dim=-1, keepdim=True)


def same_carry(left, right):
    assert left.kind == right.kind
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert torch.equal(bits(getattr(left, name)), bits(getattr(right, name)))


def nonzero(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.04, .06, model.output.weight.numel()).reshape_as(model.output.weight))
        model.output.bias.copy_(torch.tensor([.015, -.02, .03, -.01]))
        if hasattr(model, "prior_output"):
            model.prior_output.weight.copy_(torch.linspace(.03, -.02, model.prior_output.weight.numel()).reshape_as(model.prior_output.weight))
            model.prior_output.bias.copy_(torch.tensor([-.02, .01, -.01, .025]))
        if model.kind == "innovation":
            model.correction.weight.copy_(torch.linspace(-.3, .4, model.correction.weight.numel()).reshape_as(model.correction.weight))
    return model


def scalar_step(cell, value, hidden):
    """Independent scalar GRU equations for one held prequery transition."""
    ir, iz, inn = F.linear(value, cell.weight_ih_l0, cell.bias_ih_l0).chunk(3, -1)
    hr, hz, hn = F.linear(hidden, cell.weight_hh_l0, cell.bias_hh_l0).chunk(3, -1)
    reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    return (1 - update) * torch.tanh(inn + reset * hn) + update * hidden


@pytest.mark.parametrize("kind", M.KINDS)
def test_seeded_initialization_and_zero_head_exact_hold(kind):
    rng = torch.random.get_rng_state().clone()
    model, old = M.make_head(kind, 51), shared.make_head(kind, 51)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert M.parameter_count(kind) == sum(p.numel() for p in model.parameters()) == {
        "innovation": 6098, "innovation_gru": 6112}[kind]
    assert set(model.state_dict()) - set(old.state_dict()) == {"prior_output.weight", "prior_output.bias"}
    for name, value in old.state_dict().items():
        assert torch.equal(bits(value), bits(model.state_dict()[name]))
    assert torch.count_nonzero(model.prior_output.weight) == torch.count_nonzero(model.prior_output.bias) == 0
    packet = inputs(); ends = torch.ones(3, dtype=torch.bool)
    with torch.no_grad():
        actual, expected = model(*packet, episode_ends=ends), old(*packet, episode_ends=ends)
    assert isinstance(actual, shared.Forecast)
    assert torch.equal(bits(actual.prediction), bits(expected.prediction))
    assert torch.equal(bits(actual.prior), bits(expected.prior))
    assert torch.equal(actual.prior_mask, expected.prior_mask)
    same_carry(actual.carry, expected.carry)
    _features, scores, lengths, _mask = packet
    for lane, length in enumerate(lengths.tolist()):
        for step in range(length):
            assert torch.equal(actual.prediction[lane, step], scores[lane, step - step % 4])
    assert not actual.prior_mask[:, 0].any() and model._captured is None


@pytest.mark.parametrize("kind", M.KINDS)
def test_copying_head_recovers_shared_values_call_counts_and_gradients(kind):
    old, model = nonzero(shared.make_head(kind, 27)), M.make_head(kind, 27)
    with torch.no_grad():
        for name, value in old.state_dict().items():
            model.state_dict()[name].copy_(value)
        model.prior_output.load_state_dict(old.output.state_dict())
    calls, old_calls = [], []

    def hook(log, kind):
        return lambda _module, args, _output: log.append((kind, tuple(args[0].shape)))

    handles = [model.recurrent.register_forward_hook(hook(calls, "gru")),
               model.output.register_forward_hook(hook(calls, "readout")),
               model.prior_output.register_forward_hook(hook(calls, "readout")),
               old.recurrent.register_forward_hook(hook(old_calls, "gru")),
               old.output.register_forward_hook(hook(old_calls, "readout"))]
    try:
        packet, ends = inputs(), torch.ones(3, dtype=torch.bool)
        actual, expected = model(*packet, episode_ends=ends), old(*packet, episode_ends=ends)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == old_calls
    assert torch.equal(bits(actual.prediction), bits(expected.prediction))
    assert torch.equal(bits(actual.prior), bits(expected.prior))
    same_carry(actual.carry, expected.carry)
    for result in (actual, expected):
        (result.prediction.square().mean() + result.prior.square().mean()).backward()
    new_params = dict(model.named_parameters())
    for name, parameter in old.named_parameters():
        gradient = new_params[name].grad
        if name.startswith("output."):
            gradient = gradient + new_params["prior_" + name].grad
        assert gradient is not None and parameter.grad is not None
        torch.testing.assert_close(gradient, parameter.grad, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("kind", M.KINDS)
def test_independent_prior_readout_oracle_and_current_query_causality(kind):
    model = nonzero(M.make_head(kind, 41))
    features, scores, lengths, mask = inputs((13, 5))
    originals = [x.clone() for x in (features, scores, lengths, mask)]
    with torch.no_grad():
        actual = model(features, scores, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        first = model(features[:1, :4], scores[:1, :4], torch.tensor([4]), mask[:1, :4], episode_ends=torch.tensor([False]))
        pieces = [features[0, 4], center(first.carry.raw_anchor[0] / 64)]
        if kind == "innovation_gru":
            pieces += [torch.zeros(4), torch.zeros(1)]
        hidden = scalar_step(model.recurrent, torch.cat(pieces), first.carry.hidden[0])
        expected = (first.carry.raw_anchor[0] / 64
                    + center(F.linear(hidden, model.prior_output.weight, model.prior_output.bias))) * 64
        torch.testing.assert_close(actual.prior[0, 4], expected, rtol=2e-6, atol=2e-5)
        changed = scores.clone(); changed[0, 4] += torch.tensor([5., -3., 1., 2.])
        now = model(features, changed, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        assert torch.equal(actual.prior[:, :5], now.prior[:, :5])
        assert torch.equal(now.prediction[0, 4], changed[0, 4])
        assert not torch.equal(actual.prior[0, 8], now.prior[0, 8])
        changed = scores.clone(); changed[0, 8] += torch.tensor([-2., 4., 1., -1.])
        future = features.clone(); future[0, 9:, :10] += 2
        later = model(future, changed, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        assert torch.equal(actual.prior[:, :9], later.prior[:, :9])
        changed = scores.clone(); changed[0, 0] += torch.tensor([3., -2., 4., 1.])
        earlier = model(features, changed, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        assert not torch.equal(actual.prior[0, 4], earlier.prior[0, 4])
    for value, old in zip((features, scores, lengths, mask), originals, strict=True):
        assert torch.equal(bits(value), bits(old))


@pytest.mark.parametrize("kind", M.KINDS)
def test_direct_readout_gradients_are_separate_but_backbone_is_shared(kind):
    model = nonzero(M.make_head(kind, 29))
    packet = inputs((8,)); ends = torch.tensor([True]); target = torch.tensor([14., 12., 17., 11.])
    # First later-query prior has no earlier correction/readout-feedback path.
    result = model(*packet, episode_ends=ends)
    (center(result.prior[:, 4] / 64) - center(target / 64)).square().mean().backward()
    assert model.output.weight.grad is None and model.output.bias.grad is None
    for parameter in (model.prior_output.weight, model.recurrent.weight_ih_l0, model.recurrent.weight_hh_l0):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and (parameter.grad != 0).any()
    model.zero_grad(set_to_none=True)
    result = model(*packet, episode_ends=ends)
    (center(result.prediction[:, 1] / 64) - center(target / 64)).square().mean().backward()
    for parameter in model.prior_output.parameters():
        assert parameter.grad is None or not bool((parameter.grad != 0).any())
    assert model.output.weight.grad is not None and (model.output.weight.grad != 0).any()
    assert model.recurrent.weight_ih_l0.grad is not None and (model.recurrent.weight_ih_l0.grad != 0).any()
    # After t4, decision loss can still reach the prior head through correction.
    model.zero_grad(set_to_none=True)
    result = model(*packet, episode_ends=ends)
    (center(result.prediction[:, 5] / 64) - center(target / 64)).square().mean().backward()
    assert model.prior_output.weight.grad is not None and (model.prior_output.weight.grad != 0).any()
    with torch.no_grad():
        before = model(*packet, episode_ends=ends)
        model.prior_output.bias.add_(torch.tensor([.1, -.2, .05, .03]))
        after = model(*packet, episode_ends=ends)
    assert torch.equal(before.prediction[:, :4], after.prediction[:, :4])
    assert not torch.equal(before.prior[:, 4], after.prior[:, 4])
    assert not torch.equal(before.prediction[:, 5], after.prediction[:, 5])


@pytest.mark.parametrize("kind", M.KINDS)
def test_poison_masks_chunk_parity_ended_lanes_and_carry_ownership(kind):
    model = nonzero(M.make_head(kind, 73))
    features, scores, totals, mask = inputs((13, 9, 1))
    original_features, original_scores = features.clone(), scores.clone()
    carry, predictions, priors, masks = None, [], [], []
    with torch.no_grad():
        whole = model(features, scores, totals, mask, episode_ends=torch.ones(3, dtype=torch.bool))
        scores[~mask] = float("inf"); features[2, 1:] = float("-inf")
        ignored = model(features, scores, totals, mask, episode_ends=torch.ones(3, dtype=torch.bool))
        assert torch.equal(bits(whole.prediction), bits(ignored.prediction))
        assert torch.equal(bits(whole.prior), bits(ignored.prior))
        same_carry(whole.carry, ignored.carry)
        expected_mask = mask.clone(); expected_mask[:, 0] = False
        assert torch.equal(whole.prior_mask, expected_mask)
        assert torch.equal(bits(whole.prior[~expected_mask]), bits(torch.zeros_like(whole.prior[~expected_mask])))
        for start in range(0, 13, 4):
            stop = min(start + 4, 13)
            lengths = (totals - start).clamp(0, stop - start)
            saved_carry = None if carry is None else M.detach_carry(carry)
            result = model(features[:, start:stop], scores[:, start:stop], lengths, mask[:, start:stop],
                           carry=carry, episode_ends=totals <= stop)
            if carry is not None:
                same_carry(carry, saved_carry)
                assert torch.equal(result.carry.hidden[2], carry.hidden[2])
                assert torch.equal(result.carry.raw_anchor[2], carry.raw_anchor[2])
            predictions.append(result.prediction); priors.append(result.prior); masks.append(result.prior_mask)
            carry = M.detach_carry(result.carry)
        torch.testing.assert_close(torch.cat(predictions, 1), whole.prediction, rtol=0, atol=2e-6)
        torch.testing.assert_close(torch.cat(priors, 1), whole.prior, rtol=0, atol=2e-6)
        torch.testing.assert_close(carry.hidden, whole.carry.hidden, rtol=0, atol=2e-6)
    assert torch.equal(torch.cat(masks, 1), whole.prior_mask)
    assert torch.equal(carry.absolute_step, totals) and carry.ended.all()
    assert whole.carry.raw_anchor.data_ptr() != original_scores.data_ptr()
    detached = M.detach_carry(whole.carry)
    assert detached.hidden.data_ptr() != whole.carry.hidden.data_ptr()
    detached.hidden.fill_(123); detached.raw_anchor.fill_(-99)
    assert not (whole.carry.hidden == 123).all() and not (whole.carry.raw_anchor == -99).all()
    with pytest.raises(FrozenInstanceError):
        whole.prior = torch.zeros_like(whole.prior)
    # Reusing the model for a new episode cannot leak the previous carry.
    with torch.no_grad():
        reset = model(original_features, original_scores, totals, mask, episode_ends=torch.ones(3, dtype=torch.bool))
    assert torch.equal(reset.prediction, whole.prediction) and torch.equal(reset.prior, whole.prior)


@pytest.mark.parametrize("kind", M.KINDS)
def test_failure_cleanup_reentry_and_explicit_detach(kind):
    model = nonzero(M.make_head(kind, 91)); packet = inputs((8,)); ends = torch.tensor([True])

    def fail(_module, _inputs, _output):
        raise RuntimeError("injected prior readout failure")

    handle = model.prior_output.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected prior"):
            model(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert model._captured is None

    def reenter(_module, _inputs, _output):
        model(*packet, episode_ends=ends)

    handle = model.prior_output.register_forward_hook(reenter)
    try:
        with pytest.raises(ValueError, match="cannot be reentered"):
            model(*packet, episode_ends=ends)
    finally:
        handle.remove()
    assert model._captured is None
    features, scores, _lengths, mask = packet
    first = model(features[:, :4], scores[:, :4], torch.tensor([4]), mask[:, :4], episode_ends=torch.tensor([False]))
    with pytest.raises(ValueError, match="explicitly detached"):
        model(features[:, 4:], scores[:, 4:], torch.tensor([4]), mask[:, 4:], carry=first.carry, episode_ends=ends)
    assert model._captured is None
    with torch.no_grad():
        model(features[:, 4:], scores[:, 4:], torch.tensor([4]), mask[:, 4:],
              carry=M.detach_carry(first.carry), episode_ends=ends)
    with pytest.raises(ValueError, match="active inherited forward"):
        model._prediction(torch.zeros(1, model.width), torch.zeros(1, 4))


@pytest.mark.parametrize("defect", ("query_nan", "wrong_mask", "wrong_dtype", "unknown_family"))
def test_malformed_inputs_rejected_and_capture_cleared(defect):
    if defect == "unknown_family":
        with pytest.raises(ValueError, match="declared"):
            M.make_head("reset_direct", 1)
        return
    model = M.make_head("innovation", 17)
    features, scores, lengths, mask = inputs((8,))
    if defect == "query_nan":
        scores[0, 4, 0] = float("nan")
    elif defect == "wrong_mask":
        mask[0, 1] = True
    else:
        features = features.double()
    with pytest.raises(ValueError):
        model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    assert model._captured is None
