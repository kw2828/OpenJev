"""Fabricated chronological score sequences; no saved data or teacher calls."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from openjev.research import otto_cross_query_scores as M


def inputs(totals=(13, 9)):
    batch, span = len(totals), max(totals)
    features = torch.linspace(-.4, .6, batch * span * 31, dtype=torch.float32).reshape(batch, span, 31)
    scores = torch.full((batch, span, 4), float("nan"), dtype=torch.float32)
    mask = torch.zeros(batch, span, dtype=torch.bool)
    for row, length in enumerate(totals):
        for step in range(length):
            features[row, step, 15] = step / 2188
            features[row, step, 16] = (step % 4) / 2188
            features[row, step, 17] = 1
            if step % 4 == 0:
                scores[row, step] = torch.tensor([31., 33., 32., 34.]) + row + step * .25
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


def center(value):
    return value - value.mean(dim=-1, keepdim=True)


def scalar_gru(model, value, hidden):
    """Independent PyTorch GRU equations, without calling the recurrent module."""
    cell = model.recurrent
    ir, iz, inn = F.linear(value, cell.weight_ih_l0, cell.bias_ih_l0).chunk(3, -1)
    hr, hz, hn = F.linear(hidden, cell.weight_hh_l0, cell.bias_hh_l0).chunk(3, -1)
    reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    candidate = torch.tanh(inn + reset * hn)
    return (1 - update) * candidate + update * hidden


def oracle(model, features, scores, lengths):
    predicted = torch.zeros(*scores.shape, dtype=torch.float32)
    states, anchors = [], []
    for row, length in enumerate(lengths.tolist()):
        hidden = torch.zeros(model.width, dtype=torch.float32)
        anchor = torch.zeros(4, dtype=torch.float32)

        def network_input(x, raw, error=None, query=False, model=model):
            value = [x, center(raw / 64)]
            if model.kind == "innovation_gru":
                value += [torch.zeros(4) if error is None else error, torch.tensor([float(query)])]
            return torch.cat(value)

        def predict(h, raw, model=model):
            return (raw / 64 + center(F.linear(h, model.output.weight, model.output.bias))) * 64

        for step in range(length):
            x = features[row, step]
            if step % 4 == 0:
                observed = scores[row, step]
                if step == 0 or model.kind == "reset_direct":
                    hidden = scalar_gru(model, network_input(x, observed, query=True), torch.zeros_like(hidden))
                elif model.kind == "persistent_direct":
                    hidden = scalar_gru(model, network_input(x, observed), hidden)
                else:
                    prior = scalar_gru(model, network_input(x, anchor), hidden)
                    error = center((observed - predict(prior, anchor)) / 64)
                    if model.kind == "innovation":
                        hidden = prior + torch.tanh(F.linear(error, model.correction.weight))
                    else:
                        hidden = scalar_gru(model, network_input(x, observed, error, query=True), prior)
                anchor = observed
                predicted[row, step] = observed
            else:
                hidden = scalar_gru(model, network_input(x, anchor), hidden)
                predicted[row, step] = predict(hidden, anchor)
        states.append(hidden)
        anchors.append(anchor)
    return predicted, torch.stack(states), torch.stack(anchors)


@pytest.mark.parametrize("kind,count", [("innovation", 5978), ("innovation_gru", 5996),
                                       ("persistent_direct", 5862), ("reset_direct", 5862)])
def test_zero_head_holds_each_query_anchor_and_owns_state(kind, count):
    saved_rng = torch.random.get_rng_state().clone()
    model = M.make_head(kind, 47)
    assert torch.equal(saved_rng, torch.random.get_rng_state())
    assert M.parameter_count(kind) == sum(p.numel() for p in model.parameters()) == count
    features, scores, lengths, mask = inputs((9, 6, 1))
    originals = [x.clone() for x in (features, scores, lengths, mask)]
    initial = model.initial_carry(3)
    prediction, carry = model(features, scores, lengths, mask, carry=initial,
                              episode_ends=torch.ones(3, dtype=torch.bool))
    for row, length in enumerate(lengths.tolist()):
        for step in range(length):
            assert torch.equal(prediction[row, step], scores[row, step - step % 4])
        assert torch.equal(prediction[row, length:], torch.zeros_like(prediction[row, length:]))
        assert torch.equal(carry.raw_anchor[row], scores[row, (length - 1) // 4 * 4])
    assert torch.equal(carry.absolute_step, lengths) and bool(carry.ended.all())
    assert not bool(initial.has_query.any()) and not bool(initial.hidden.any())
    for value, old in zip((features, scores, lengths, mask), originals, strict=True):
        assert torch.equal(value.view(torch.uint8), old.view(torch.uint8))
    detached = M.detach_carry(carry)
    assert not detached.hidden.requires_grad and detached.hidden.grad_fn is None
    assert detached.hidden.data_ptr() != carry.hidden.data_ptr()
    assert detached.raw_anchor.data_ptr() != scores.data_ptr()
    detached.hidden.fill_(123)
    assert not bool((carry.hidden == 123).all())


def test_same_shape_models_have_identical_locally_seeded_initial_core():
    models = [M.make_head(kind, 321) for kind in ("innovation", "persistent_direct", "reset_direct")]
    for name, value in models[0].state_dict().items():
        if not name.startswith("correction."):
            assert all(torch.equal(value, m.state_dict()[name]) for m in models[1:])
    assert not bool(models[0].correction.weight.any())


@pytest.mark.parametrize("kind", M.KINDS)
def test_nonzero_predictions_match_independent_scalar_recurrence(kind):
    model = nonzero(M.make_head(kind, 23))
    features, scores, lengths, mask = inputs((11, 8))
    with torch.no_grad():
        actual, carry = model(features, scores, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        expected, hidden, anchor = oracle(model, features, scores, lengths)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(carry.hidden, hidden, rtol=2e-6, atol=2e-6)
    assert torch.equal(carry.raw_anchor, anchor)
    assert not torch.equal(actual[0, 1], scores[0, 0])
    assert torch.equal(actual[mask], scores[mask])


@pytest.mark.parametrize("kind", M.KINDS)
def test_chunks_equal_whole_episode_and_ended_lane_stops(kind):
    model = nonzero(M.make_head(kind, 91))
    features, scores, totals, mask = inputs((13, 9))
    with torch.no_grad():
        whole, final = model(features, scores, totals, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        pieces, carry = [], None
        ended_state = None
        for start in range(0, 13, 4):
            stop = min(start + 4, 13)
            lengths = (totals - start).clamp(0, stop - start)
            piece, carry = model(features[:, start:stop], scores[:, start:stop], lengths, mask[:, start:stop],
                                 carry=carry, episode_ends=totals <= stop)
            pieces.append(piece)
            carry = M.detach_carry(carry)
            if start == 8:
                ended_state = carry.hidden[1].clone(), carry.raw_anchor[1].clone()
        torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=0, atol=2e-6)
        torch.testing.assert_close(carry.hidden, final.hidden, rtol=0, atol=2e-6)
    assert torch.equal(carry.absolute_step, totals) and bool(carry.ended.all())
    assert torch.equal(carry.hidden[1], ended_state[0]) and torch.equal(carry.raw_anchor[1], ended_state[1])
    with pytest.raises(ValueError, match="ended lanes"):
        model(features[:, :4], scores[:, :4], torch.tensor([4, 0]), mask[:, :4], carry=carry,
              episode_ends=torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize("kind", M.KINDS)
def test_causality_and_skipped_score_poison(kind):
    model = nonzero(M.make_head(kind, 7))
    features, scores, lengths, mask = inputs((12, 5))
    with torch.no_grad():
        first, state = model(features, scores, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        poison = scores.clone()
        poison[~mask] = float("inf")
        second, other = model(features, poison, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
        assert torch.equal(first, second) and torch.equal(state.hidden, other.hidden)
        future = features.clone()
        future[0, 6:, :10] += 3
        future_scores = scores.clone()
        future_scores[0, 8] += torch.tensor([5., -2., 1., 3.])
        changed, _ = model(future, future_scores, lengths, mask, episode_ends=torch.ones(2, dtype=torch.bool))
    assert torch.equal(first[0, :6], changed[0, :6])
    assert torch.equal(first[1], changed[1])
    assert not torch.equal(first[0, 8], changed[0, 8])


def test_reset_direct_discards_previous_query_history_but_persistent_direct_retains_it():
    features, scores, lengths, mask = inputs((8,))
    altered = features.clone()
    altered[0, :4, 0] += 2
    for kind in ("reset_direct", "persistent_direct"):
        model = nonzero(M.make_head(kind, 29))
        with torch.no_grad():
            first, _ = model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
            second, _ = model(altered, scores, lengths, mask, episode_ends=torch.tensor([True]))
        if kind == "reset_direct":
            assert torch.equal(first[:, 4:], second[:, 4:])
        else:
            assert not torch.equal(first[:, 5:], second[:, 5:])


def test_zero_innovation_leaves_prior_hidden_exact_and_query_raw_bits_are_copied():
    model = nonzero(M.make_head("innovation", 18))
    features, scores, _, mask = inputs((8,))
    with torch.no_grad():
        _, carry = model(features[:, :4], scores[:, :4], torch.tensor([4]), mask[:, :4],
                         episode_ends=torch.tensor([False]))
        value = torch.cat((features[:, 4], center(carry.raw_anchor / 64)), dim=-1)
        _, expected = model.recurrent(value[:, None], carry.hidden[None])
        raw = (carry.raw_anchor / 64 + center(model.output(expected[0]))) * 64
        prediction, end = model(features[:, 4:5], raw[:, None], torch.tensor([1]), mask[:, 4:5],
                                carry=M.detach_carry(carry), episode_ends=torch.tensor([True]))
    assert torch.equal(end.hidden, expected[0])
    assert torch.equal(prediction[:, 0], raw)
    raw[0] = torch.tensor([-0., 0., 32., 64.])
    with torch.no_grad():
        prediction, _ = model(features[:, :1], raw[:, None], torch.tensor([1]), mask[:, :1],
                              episode_ends=torch.tensor([True]))
    assert torch.equal(prediction[:, 0].view(torch.uint8), raw.view(torch.uint8))


def test_later_correction_reaches_parameters_and_detachment_is_explicit():
    model = nonzero(M.make_head("innovation", 67))
    features, scores, lengths, mask = inputs((8,))
    prediction, carry = model(features, scores, lengths, mask, episode_ends=torch.tensor([False]))
    center(prediction[:, 5:]).square().sum().backward()
    for parameter in (model.output.weight, model.recurrent.weight_ih_l0,
                      model.recurrent.weight_hh_l0, model.correction.weight):
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad != 0).any())
    with pytest.raises(ValueError, match="explicitly detached"):
        model(features[:, :4], scores[:, :4], torch.tensor([4]), mask[:, :4], carry=carry,
              episode_ends=torch.tensor([True]))


@pytest.mark.parametrize("kind", ("innovation", "innovation_gru"))
def test_nonfinite_prequery_readout_cannot_be_hidden_by_query_override(kind):
    model = M.make_head(kind, 5)
    features, scores, _, mask = inputs((5,))
    with torch.no_grad():
        _, carry = model(features[:, :4], scores[:, :4], torch.tensor([4]), mask[:, :4],
                         episode_ends=torch.tensor([False]))
        model.output.bias.copy_(torch.tensor([torch.finfo(torch.float32).max, 0., 0., 0.]))
    with pytest.raises(ValueError, match="finite prequery"):
        model(features[:, 4:], scores[:, 4:], torch.tensor([1]), mask[:, 4:],
              carry=M.detach_carry(carry), episode_ends=torch.tensor([True]))


@pytest.mark.parametrize("defect", ["feature_dtype", "query_mask", "nonfinite_query", "nonterminal_tail",
                                   "feature_step", "unroundtrippable_query", "negative_length", "carry_family"])
def test_malformed_active_inputs_rejected(defect):
    model = M.make_head("innovation", 13)
    features, scores, lengths, mask = inputs((4,))
    ends, carry = torch.tensor([True]), None
    if defect == "feature_dtype":
        features = features.double()
    elif defect == "query_mask":
        mask[0, 1] = True
    elif defect == "nonfinite_query":
        scores[0, 0, 0] = float("nan")
    elif defect == "nonterminal_tail":
        lengths, ends = torch.tensor([3]), torch.tensor([False])
    elif defect == "feature_step":
        features[0, 2, 15] = 0
    elif defect == "unroundtrippable_query":
        scores[0, 0, 0] = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
    elif defect == "negative_length":
        lengths = torch.tensor([-1])
    elif defect == "carry_family":
        carry = replace(model.initial_carry(1), kind="reset_direct")
    with pytest.raises(ValueError):
        model(features, scores, lengths, mask, carry=carry, episode_ends=ends)
