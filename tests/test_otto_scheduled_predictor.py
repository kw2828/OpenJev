"""Fabricated tensors only: no empirical checkpoints, collection or fitting."""
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from openjev.research import otto_protected_training_model as original
from openjev.research import otto_scheduled_predictor as M

FIELDS = ("prediction", "action_prediction", "prior", "shadow_prior", "prior_mask", "keys_hidden", "key_mask")
STATE_FIELDS = ("hidden", "raw_anchor", "has_query", "absolute_step", "ended")


def bits(value):
    return value.detach().contiguous().view(torch.uint8)


def same_bits(actual, expected):
    assert torch.equal(bits(actual), bits(expected))


def packet(totals=(17, 9, 1), period=8, span=None):
    batch = len(totals)
    span = max(totals) if span is None else span
    assert span >= max(totals)
    features = torch.linspace(-.4, .6, batch * span * 31, dtype=torch.float32).reshape(batch, span, 31)
    scores = torch.full((batch, span, 4), float("nan"), dtype=torch.float32)
    mask = torch.zeros(batch, span, dtype=torch.bool)
    for lane, length in enumerate(totals):
        for step in range(length):
            features[lane, step, 15:18] = torch.tensor([step / 2188, (step % period) / 2188, 1.])
            if step % period == 0:
                scores[lane, step] = torch.tensor([12., 15., 11., 18.]) + lane + step * .125
                mask[lane, step] = True
        features[lane, length:] = float("nan")
    return features, scores, torch.tensor(totals, dtype=torch.int64), mask


def nonzero(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.linspace(-.03, .05, 112).reshape(4, 28))
        model.output.bias.copy_(torch.tensor([.01, -.02, .03, -.01]))
        model.action_residual.weight.copy_(torch.linspace(-.01, .02, 112).reshape(4, 28))
        model.action_residual.bias.copy_(torch.tensor([.01, -.02, .03, -.01]))
    return model


def compare_carry(actual, expected):
    assert actual.kind == expected.kind
    for name in STATE_FIELDS:
        same_bits(getattr(actual, name), getattr(expected, name))


def compare_forecast(actual, expected):
    for name in FIELDS:
        same_bits(getattr(actual, name), getattr(expected, name))
    assert actual.carry.query_period == expected.carry.query_period
    compare_carry(actual.carry.base, expected.carry.base)
    assert actual.work_counts == expected.work_counts


@pytest.mark.parametrize("mode", M.MODES)
@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_constructor_and_factory_preserve_rng_and_exact_original_state(mode, period):
    rng = torch.random.get_rng_state().clone()
    model = M.ScheduledPredictor(mode, 61, period)
    reference = original.make_head(mode, 61)
    same_bits(torch.random.get_rng_state(), rng)
    assert len(model.state_dict()) == 8 and set(model.state_dict()) == set(reference.state_dict())
    for name, value in reference.state_dict().items():
        same_bits(model.state_dict()[name], value)
    assert sum(p.numel() for p in model.parameters()) == M.parameter_count(mode) == 6112
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (
        M.parameter_count(mode, trainable_only=True)) == (116 if mode == "frozen" else 6112)
    same_bits(M.make_head(mode, 61, period).output.weight, model.output.weight)
    same_bits(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("mode", M.MODES)
@pytest.mark.parametrize("outer_no_grad", (False, True))
def test_period4_outputs_priors_carry_and_recurrent_calls_match_original_bitwise(mode, outer_no_grad):
    reference = nonzero(original.make_head(mode, 71))
    model = M.from_state(mode, 72, 4, reference.state_dict())
    inputs = packet((33, 9, 6, 1), 4, span=36)
    ends = torch.ones(4, dtype=torch.bool)
    actual_calls, expected_calls = [], []
    def capture(target):
        return lambda _module, args, _output: target.append(tuple(args[0].shape))
    left = model.recurrent.register_forward_hook(capture(actual_calls))
    right = reference.recurrent.register_forward_hook(capture(expected_calls))
    # Frozen inference is the new contract, including its action head. Match the
    # original's effective context as well as all eight parameter flags.
    with torch.no_grad() if mode == "frozen" or outer_no_grad else nullcontext():
        expected = reference(*inputs, episode_ends=ends)
    with torch.no_grad() if outer_no_grad else nullcontext():
        actual = model(*inputs, episode_ends=ends)
    left.remove()
    right.remove()
    for name in ("prediction", "action_prediction", "prior", "prior_mask"):
        same_bits(getattr(actual, name), getattr(expected, name))
    compare_carry(actual.carry.base, expected.carry)
    assert actual_calls == expected_calls
    assert actual.work_counts["recurrent_calls"] == len(actual_calls)
    assert actual.work_counts["recurrent_token_transitions"] == sum(b * t for b, t, _ in actual_calls)


def center(value):
    return value - value.mean(dim=-1, keepdim=True)


def scalar_gru(model, inputs, hidden):
    """Independent scalar GRU equations, never calls model.recurrent."""
    cell = model.recurrent
    ir, iz, inn = F.linear(inputs, cell.weight_ih_l0, cell.bias_ih_l0).chunk(3, -1)
    hr, hz, hn = F.linear(hidden, cell.weight_hh_l0, cell.bias_hh_l0).chunk(3, -1)
    reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
    candidate = torch.tanh(inn + reset * hn)
    return (1 - update) * candidate + update * hidden


def scalar_oracle(model, inputs):
    features, scores, lengths, _ = inputs
    period = model.query_period
    shape = scores.shape
    out = {name: torch.zeros(shape, dtype=torch.float32)
           for name in ("prediction", "action_prediction", "prior", "shadow_prior")}
    out["keys_hidden"] = torch.zeros(*shape[:2], 28)
    states, anchors = [], []
    def network_input(x, anchor, error=None, query=False):
        return torch.cat((x, center(anchor / 64), torch.zeros(4) if error is None else error,
                          torch.tensor([float(query)])))
    def predict(hidden, anchor):
        return (anchor / 64 + center(F.linear(hidden, model.output.weight, model.output.bias))) * 64
    def residual(hidden):
        return center(64 * F.linear(hidden, model.action_residual.weight, model.action_residual.bias))
    for lane, length in enumerate(lengths.tolist()):
        hidden, anchor = torch.zeros(28), torch.zeros(4)
        for step in range(length):
            x = features[lane, step]
            if step % period == 0:
                observed = scores[lane, step]
                if step == 0:
                    hidden = scalar_gru(model, network_input(x, observed, query=True), torch.zeros_like(hidden))
                else:
                    prior_hidden = scalar_gru(model, network_input(x, anchor), hidden)
                    prior = predict(prior_hidden, anchor)
                    out["prior"][lane, step] = prior
                    out["shadow_prior"][lane, step] = prior + residual(prior_hidden)
                    out["keys_hidden"][lane, step] = prior_hidden
                    error = center((observed - prior) / 64)
                    hidden = scalar_gru(model, network_input(x, observed, error, True), prior_hidden)
                anchor = observed
                out["prediction"][lane, step] = observed
                out["action_prediction"][lane, step] = observed
            else:
                hidden = scalar_gru(model, network_input(x, anchor), hidden)
                out["keys_hidden"][lane, step] = hidden
                out["prediction"][lane, step] = predict(hidden, anchor)
                out["action_prediction"][lane, step] = out["prediction"][lane, step] + residual(hidden)
        states.append(hidden)
        anchors.append(anchor)
    return out, torch.stack(states), torch.stack(anchors)


@pytest.mark.parametrize("mode", M.MODES)
def test_period8_matches_independent_scalar_recurrence_and_has_no_query_at4(mode):
    model = nonzero(M.make_head(mode, 81, 8))
    inputs = packet((19, 9, 1))
    with torch.no_grad():
        actual = model(*inputs, episode_ends=torch.ones(3, dtype=torch.bool))
        expected, hidden, anchor = scalar_oracle(model, inputs)
    # Same independently specified oracle bounds as the immutable original's
    # scalar-versus-batched GRU test; old/new period4 parity above is bitwise.
    for name, value in expected.items():
        torch.testing.assert_close(getattr(actual, name), value, rtol=2e-6,
                                   atol=2e-6 if name == "keys_hidden" else 2e-5)
    torch.testing.assert_close(actual.carry.base.hidden, hidden, rtol=2e-6, atol=2e-6)
    same_bits(actual.carry.base.raw_anchor, anchor)
    assert not actual.prior_mask[:, 4].any() and actual.key_mask[0, 4]
    assert torch.isnan(inputs[1][0, 4]).all()
    same_bits(actual.action_prediction[inputs[3]], inputs[1][inputs[3]])


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_current_query_and_future_inputs_cannot_change_prior_keys_or_earlier_outputs(period):
    model = nonzero(M.make_head("frozen", 91, period))
    inputs = packet((2 * period + 3,), period)
    ends = torch.tensor([True])
    initial = model(*inputs, episode_ends=ends)
    changed = inputs[1].clone()
    changed[0, period] += torch.tensor([4., -2., 3., -1.])
    current = model(inputs[0], changed, inputs[2], inputs[3], episode_ends=ends)
    for name in ("prior", "shadow_prior", "keys_hidden"):
        same_bits(getattr(initial, name)[:, :period + 1], getattr(current, name)[:, :period + 1])
    for name in ("prediction", "action_prediction"):
        same_bits(getattr(initial, name)[:, :period], getattr(current, name)[:, :period])
    same_bits(current.action_prediction[0, period], changed[0, period])
    assert not torch.equal(initial.keys_hidden[0, period + 1], current.keys_hidden[0, period + 1])
    future_features, future_scores = inputs[0].clone(), inputs[1].clone()
    future_features[:, period + 1:, :15] += 10
    future_scores[:, 2 * period] += 5
    future = model(future_features, future_scores, inputs[2], inputs[3], episode_ends=ends)
    for name in FIELDS:
        same_bits(getattr(initial, name)[:, :period + 1], getattr(future, name)[:, :period + 1])


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_skipped_scores_and_padding_poison_are_inert_and_inputs_are_unchanged(period):
    model = nonzero(M.make_head("frozen", 92, period))
    inputs = packet((period + 2, 1), period, span=2 * period)
    saved = [value.clone() for value in inputs]
    ends = torch.ones(2, dtype=torch.bool)
    before = model(*inputs, episode_ends=ends)
    features, scores = inputs[0].clone(), inputs[1].clone()
    active = torch.arange(features.shape[1])[None, :] < inputs[2][:, None]
    features[~active] = float("inf")
    scores[~inputs[3]] = float("-inf")
    after = model(features, scores, inputs[2], inputs[3], episode_ends=ends)
    compare_forecast(after, before)
    for value, snapshot in zip(inputs, saved, strict=True):
        same_bits(value, snapshot)
    for name in ("prediction", "action_prediction"):
        same_bits(getattr(after, name)[~active], torch.zeros_like(getattr(after, name)[~active]))
    for name in ("prior", "shadow_prior"):
        same_bits(getattr(after, name)[~after.prior_mask], torch.zeros_like(getattr(after, name)[~after.prior_mask]))
    same_bits(after.keys_hidden[~after.key_mask], torch.zeros_like(after.keys_hidden[~after.key_mask]))
    assert torch.equal(after.key_mask, active & (torch.arange(features.shape[1])[None, :] >= 1))


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_actual_queries_preserve_signed_zero_without_residual_addition(period):
    model = nonzero(M.make_head("frozen", 93, period))
    inputs = packet((period + 1,), period)
    inputs[1][inputs[3]] = torch.tensor([-0., 0., 1., -1.])
    result = model(*inputs, episode_ends=torch.tensor([True]))
    same_bits(result.prediction[inputs[3]], inputs[1][inputs[3]])
    same_bits(result.action_prediction[inputs[3]], inputs[1][inputs[3]])


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_empty_chunk_advances_nothing_and_performs_no_readout(period):
    model = M.make_head("frozen", 94, period)
    carry = model.initial_carry(2)
    features = torch.full((2, 32, 31), float("nan"))
    scores = torch.full((2, 32, 4), float("inf"))
    result = model(features, scores, torch.zeros(2, dtype=torch.int64),
                   torch.zeros(2, 32, dtype=torch.bool), carry=carry,
                   episode_ends=torch.zeros(2, dtype=torch.bool))
    for name in FIELDS:
        value = getattr(result, name)
        same_bits(value, torch.zeros_like(value))
    assert all(value == 0 for value in result.work_counts.values())
    compare_carry(result.carry.base, carry.base)
    for name in STATE_FIELDS:
        assert getattr(result.carry.base, name).data_ptr() != getattr(carry.base, name).data_ptr()


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_whole_history_and32_chunks_match_with_owned_detached_ended_lanes(period):
    model = nonzero(M.make_head("frozen", 101, period))
    inputs = packet((65, 33, 32, 1), period, span=96)
    whole = model(*inputs, episode_ends=torch.ones(4, dtype=torch.bool))
    pieces, carry, ended_snapshot = [], None, None
    for start in range(0, 96, 32):
        lengths = (inputs[2] - start).clamp(0, 32)
        ends = (lengths > 0) & (inputs[2] <= start + 32)
        chunk = model(inputs[0][:, start:start + 32], inputs[1][:, start:start + 32], lengths,
                      inputs[3][:, start:start + 32], carry=carry, episode_ends=ends)
        pieces.append(chunk)
        detached = M.detach_carry(chunk.carry)
        for name in STATE_FIELDS:
            value, source = getattr(detached.base, name), getattr(chunk.carry.base, name)
            same_bits(value, source)
            assert value.data_ptr() != source.data_ptr()
            assert not value.requires_grad and value.grad_fn is None
        carry = detached
        if start == 0:
            ended_snapshot = {n: getattr(carry.base, n)[2:].clone() for n in STATE_FIELDS}
        else:
            for name in STATE_FIELDS:
                same_bits(getattr(carry.base, name)[2:], ended_snapshot[name])
    for name in FIELDS:
        combined = torch.cat([getattr(piece, name) for piece in pieces], dim=1)
        if combined.dtype == torch.bool:
            same_bits(combined, getattr(whole, name))
        else:
            torch.testing.assert_close(combined, getattr(whole, name), rtol=0,
                                       atol=2e-6 if name == "keys_hidden" else 2e-5)
    torch.testing.assert_close(carry.base.hidden, whole.carry.base.hidden, rtol=0, atol=2e-6)
    for name in STATE_FIELDS[1:]:
        same_bits(getattr(carry.base, name), getattr(whole.carry.base, name))
    for name, count in whole.work_counts.items():
        assert sum(piece.work_counts[name] for piece in pieces) == count
    snapshot = whole.carry.base.hidden.clone()
    carry.base.hidden.add_(100)
    same_bits(whole.carry.base.hidden, snapshot)


@pytest.mark.parametrize("period", M.QUERY_PERIODS)
def test_full_horizon_tail_and_explicit_end(period):
    model = M.make_head("frozen", 102, period)
    # Only a final fabricated chunk is needed; the carry has a prior real query.
    carry = model.initial_carry(1)
    state = replace(carry.base, has_query=torch.tensor([True]), absolute_step=torch.tensor([2176]),
                    raw_anchor=torch.tensor([[12., 13., 14., 15.]]))
    carry = replace(carry, base=state)
    inputs = packet((12,), period, span=32)
    for step in range(12):
        inputs[0][0, step, 15:18] = torch.tensor([(2176 + step) / 2188, (step % period) / 2188, 1.])
    with pytest.raises(ValueError, match="boundaries|horizon"):
        model(*inputs, carry=carry, episode_ends=torch.tensor([False]))
    result = model(*inputs, carry=carry, episode_ends=torch.tensor([True]))
    assert result.carry.base.absolute_step.item() == 2188 and result.carry.base.ended.item()


@pytest.mark.parametrize("mode", M.MODES)
def test_gradient_contract_and_causal_shadow_prior(mode):
    model = nonzero(M.make_head(mode, 111, 8))
    features, scores, lengths, mask = packet((17,))
    features.requires_grad_(True)
    scores.requires_grad_(True)
    result = model(features, scores, lengths, mask, episode_ends=torch.tensor([True]))
    floating = [getattr(result, name) for name in FIELDS if getattr(result, name).dtype == torch.float32]
    floating += [result.carry.base.hidden, result.carry.base.raw_anchor]
    assert all(value.requires_grad is (mode == "joint") for value in floating)
    if mode == "frozen":
        assert all(p.grad is None for p in model.parameters())
        assert features.grad is scores.grad is None
    else:
        coefficients = torch.tensor([1., 2., 4., 8.])
        (result.shadow_prior[0, 8] * coefficients).square().mean().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert (parameter.grad != 0).any()
        assert scores.grad is not None and scores.grad[0, 0].abs().sum() > 0
        same_bits(scores.grad[0, 8:], torch.zeros_like(scores.grad[0, 8:]))
        same_bits(scores.grad[~mask], torch.zeros_like(scores.grad[~mask]))
        same_bits(features.grad[:, 9:], torch.zeros_like(features.grad[:, 9:]))


@pytest.mark.parametrize("mode", M.MODES)
def test_residual_cannot_feed_base_predictions_prior_keys_or_carry(mode):
    model = nonzero(M.make_head(mode, 112, 8))
    inputs = packet()
    with torch.no_grad():
        before = model(*inputs, episode_ends=torch.ones(3, dtype=torch.bool))
        model.action_residual.weight.mul_(3)
        model.action_residual.bias.add_(torch.tensor([1., -2., 3., -4.]))
        after = model(*inputs, episode_ends=torch.ones(3, dtype=torch.bool))
    for name in ("prediction", "prior", "prior_mask", "keys_hidden", "key_mask"):
        same_bits(getattr(after, name), getattr(before, name))
    compare_carry(after.carry.base, before.carry.base)
    assert not torch.equal(before.action_prediction, after.action_prediction)
    assert not torch.equal(before.shadow_prior, after.shadow_prior)


def test_work_counters_match_actual_hooks_and_hand_counted_period8_calls():
    model = nonzero(M.make_head("frozen", 113, 8))
    recurrent, readout, residual = [], [], []
    def capture(target):
        return lambda _module, args, _output: target.append(tuple(args[0].shape))
    handles = [model.recurrent.register_forward_hook(capture(recurrent)),
               model.output.register_forward_hook(capture(readout)),
               model.action_residual.register_forward_hook(capture(residual))]
    result = model(*packet(), episode_ends=torch.ones(3, dtype=torch.bool))
    for handle in handles:
        handle.remove()
    work = result.work_counts
    assert work == {"recurrent_calls": 7, "recurrent_token_transitions": 30,
                    "base_readout_calls": 4, "base_readout_rows": 24,
                    "action_readout_calls": 2, "action_readout_rows": 21,
                    "shadow_readout_calls": 2, "shadow_readout_rows": 3,
                    "active_rows": 27, "query_rows": 6, "later_query_rows": 3,
                    "nonquery_rows": 21, "key_rows": 24}
    assert len(recurrent) == work["recurrent_calls"]
    assert sum(b * t for b, t, _ in recurrent) == work["recurrent_token_transitions"]
    assert len(readout) == work["base_readout_calls"]
    assert sum(shape[0] * (shape[1] if len(shape) == 3 else 1) for shape in readout) == work["base_readout_rows"]
    assert len([s for s in residual if len(s) == 3]) == work["action_readout_calls"]
    assert len([s for s in residual if len(s) == 2]) == work["shadow_readout_calls"]


@pytest.mark.parametrize("mode", M.MODES)
def test_from_state_copies_eight_tensors_without_alias_or_rng_change(mode):
    source = nonzero(M.make_head("joint", 121, 4)).state_dict()
    saved = {name: value.clone() for name, value in source.items()}
    rng = torch.random.get_rng_state().clone()
    model = M.from_state(mode, 122, 8, source)
    same_bits(torch.random.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        same_bits(value, source[name])
        assert value.data_ptr() != source[name].data_ptr()
        same_bits(source[name], saved[name])
    source["output.bias"].add_(100)
    same_bits(model.output.bias, saved["output.bias"])


@pytest.mark.parametrize("defect", ("missing", "extra", "shape", "dtype", "nan", "inf", "meta", "not_tensor", "not_mapping"))
def test_invalid_state_is_rejected_before_model_construction(defect, monkeypatch):
    state = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in M.STATE_SHAPES.items()}
    if defect == "missing":
        state.pop("action_residual.bias")
    elif defect == "extra":
        state["extra"] = torch.zeros(1)
    elif defect == "shape":
        state["output.bias"] = torch.zeros(1, 4)
    elif defect == "dtype":
        state["output.bias"] = torch.zeros(4, dtype=torch.float64)
    elif defect in ("nan", "inf"):
        state["output.bias"][0] = float(defect)
    elif defect == "meta":
        state["output.bias"] = torch.empty(4, device="meta")
    elif defect == "not_tensor":
        state["output.bias"] = [0.] * 4
    else:
        state = list(state.items())
    monkeypatch.setattr(M, "make_head", lambda *_: pytest.fail("invalid state must not construct a model"))
    with pytest.raises(ValueError):
        M.from_state("frozen", 1, 8, state)


@pytest.mark.parametrize("args", (("bad", 1, 4), ("joint", True, 4), ("joint", -1, 4),
                                  ("joint", 2**32, 4), ("joint", 1, 4.), ("joint", 1, True), ("joint", 1, 3)))
def test_invalid_constructor_configuration(args):
    with pytest.raises(ValueError):
        M.make_head(*args)


def test_carry_rejects_period_switch_undetched_graph_and_resuming_ended_lane():
    model = nonzero(M.make_head("joint", 131, 8))
    inputs = packet((8,))
    result = model(*inputs, episode_ends=torch.tensor([False]))
    with pytest.raises(ValueError, match="explicitly detached"):
        model(*inputs, carry=result.carry, episode_ends=torch.tensor([False]))
    carry = M.detach_carry(result.carry)
    other = M.from_state("joint", 132, 4, model.state_dict())
    with pytest.raises(ValueError, match="switch query period"):
        other(*packet((4,), 4), carry=carry, episode_ends=torch.tensor([False]))
    ended = replace(carry, base=replace(carry.base, ended=torch.tensor([True])))
    with pytest.raises(ValueError, match="ended lanes"):
        model(*inputs, carry=ended, episode_ends=torch.tensor([True]))


@pytest.mark.parametrize("defect", ("mask", "age", "step", "flag", "dtype", "query_nan", "partial", "gradient_lock"))
def test_malformed_inputs_fail_and_release_forward_lock(defect):
    model = M.make_head("frozen", 141, 8)
    features, scores, lengths, mask = packet((8,))
    ends = torch.tensor([False])
    if defect == "mask":
        mask[0, 4] = True
    elif defect == "age":
        features[0, 4, 16] = 0
    elif defect == "step":
        features[0, 1, 15] = 0
    elif defect == "flag":
        features[0, 0, 17] = 0
    elif defect == "dtype":
        features = features.double()
    elif defect == "query_nan":
        scores[0, 0, 0] = float("nan")
    elif defect == "partial":
        lengths[0] = 7
    else:
        model.output.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        model(features, scores, lengths, mask, episode_ends=ends)
    assert not model._action_lock.locked()
    assert model._captured is model._action_hidden is None


def test_failure_and_reentry_do_not_leave_capture_or_lock(monkeypatch):
    model = nonzero(M.make_head("frozen", 151, 8))
    inputs, ends = packet(), torch.ones(3, dtype=torch.bool)
    real = model.output.forward
    def fail(_hidden):
        raise RuntimeError("fabricated readout failure")
    monkeypatch.setattr(model.output, "forward", fail)
    with pytest.raises(RuntimeError, match="fabricated"):
        model(*inputs, episode_ends=ends)
    assert not model._action_lock.locked() and model._captured is model._action_hidden is None
    monkeypatch.setattr(model.output, "forward", real)
    attempts = []
    def reenter(_module, _args, _output):
        with pytest.raises(ValueError, match="reentered"):
            model(*inputs, episode_ends=ends)
        attempts.append(True)
    handle = model.recurrent.register_forward_hook(reenter)
    first = model(*inputs, episode_ends=ends)
    handle.remove()
    assert attempts and not model._action_lock.locked()
    compare_forecast(model(*inputs, episode_ends=ends), first)


def test_original_source_pin_is_required(monkeypatch):
    monkeypatch.setattr(M, "ORIGINAL_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source pin"):
        M.make_head("frozen", 1, 4)
