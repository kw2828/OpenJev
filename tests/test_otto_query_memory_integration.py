"""Fabricated integration only; no empirical files, fitting or game calls."""
from dataclasses import replace

import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_scheduled_predictor as predictor


def packet(period, length=65):
    features = torch.linspace(-.3, .7, length * 31).reshape(1, length, 31)
    steps = torch.arange(length)
    features[0, :, 15] = steps / 2188
    features[0, :, 16] = (steps % period) / 2188
    features[0, :, 17] = 1
    mask = (steps % period == 0)[None]
    scores = torch.full((1, length, 4), float("nan"))
    scores[0, mask[0]] = torch.tensor([4., 7., 2., 6.]) + (
        steps[mask[0]][:, None] * torch.tensor([.125, -.25, .5, .25]))
    return features, scores, mask


def projection():
    return torch.arange(28 * 3, dtype=torch.float32).sin().reshape(28, 3)


def apply(kernel, slow, scores, mask, lengths, ends, weight, carry=None, **kwargs):
    # Fix projection call geometry across chunk sizes. A flattened whole-history
    # GEMM versus a one-step GEMV otherwise differs by float32 rounding at tails.
    keys = torch.stack([slow.keys_hidden[:, step] @ weight
                        for step in range(slow.keys_hidden.shape[1])], dim=1)
    return kernel(slow.action_prediction, slow.shadow_prior, keys,
                  scores, mask, slow.prior_mask, slow.key_mask, lengths, ends,
                  carry=carry, **kwargs)


@pytest.mark.parametrize("period", (4, 8))
def test_composed_full_episode_and_detached_chunks_match_and_leave_slow_state_unchanged(period):
    model = predictor.make_head("frozen", 73, period)
    kernel = memory.QueryMemory(memory.Config("trace_delta", key_dim=3))
    features, scores, mask = packet(period)
    ends, lengths = torch.tensor([True]), torch.tensor([65])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    slow = model(features, scores, lengths, mask, episode_ends=ends)
    full = apply(kernel, slow, scores, mask, lengths, ends, projection())
    chunks, corrections, slow_carry, fast_carry = [], [], None, None
    for start, stop in ((0, 32), (32, 64), (64, 65)):
        chunk_lengths, chunk_ends = torch.tensor([stop - start]), torch.tensor([stop == 65])
        part = model(features[:, start:stop], scores[:, start:stop], chunk_lengths,
                     mask[:, start:stop], episode_ends=chunk_ends, carry=slow_carry)
        result = apply(kernel, part, scores[:, start:stop], mask[:, start:stop],
                       chunk_lengths, chunk_ends, projection(), carry=fast_carry)
        chunks.append(result.action_prediction)
        corrections.append(result.prewrite_correction)
        slow_carry = predictor.detach_carry(part.carry)
        fast_carry = memory.detach_carry(result.carry)
    assert torch.equal(torch.cat(chunks, 1), full.action_prediction)
    assert torch.equal(torch.cat(corrections, 1), full.prewrite_correction)
    assert torch.equal(full.carry.matrix, fast_carry.matrix)
    assert torch.equal(full.carry.trace, fast_carry.trace)
    assert torch.equal(full.action_prediction[mask], scores[mask])
    assert torch.equal(slow_carry.base.hidden, slow.carry.base.hidden)
    assert torch.equal(slow_carry.base.raw_anchor, slow.carry.base.raw_anchor)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    repeated = model(features, scores, lengths, mask, episode_ends=ends)
    assert torch.equal(repeated.action_prediction, slow.action_prediction)
    assert torch.equal(repeated.shadow_prior, slow.shadow_prior)


@pytest.mark.parametrize("period", (4, 8))
def test_composed_prewrite_forecast_is_causal_but_later_memory_can_change(period):
    model = predictor.make_head("frozen", 74, period)
    kernel = memory.QueryMemory(memory.Config("trace_delta", key_dim=3))
    features, scores, mask = packet(period, length=2 * period + 1)
    lengths, ends = torch.tensor([features.shape[1]]), torch.tensor([True])
    slow = model(features, scores, lengths, mask, episode_ends=ends)
    initial = apply(kernel, slow, scores, mask, lengths, ends, projection())
    changed = scores.clone()
    changed[0, period] += torch.tensor([8., -4., 3., -7.])
    other_slow = model(features, changed, lengths, mask, episode_ends=ends)
    other = apply(kernel, other_slow, changed, mask, lengths, ends, projection())
    assert torch.equal(initial.prewrite_correction[:, :period + 1],
                       other.prewrite_correction[:, :period + 1])
    first_forecast = slow.shadow_prior + 64 * initial.prewrite_correction
    second_forecast = other_slow.shadow_prior + 64 * other.prewrite_correction
    assert torch.equal(first_forecast[0, period], second_forecast[0, period])
    assert not torch.equal(initial.prewrite_correction[0, period + 1],
                           other.prewrite_correction[0, period + 1])


@pytest.mark.parametrize("period", (4, 8))
def test_frozen_predictor_allows_key_projection_gradient_without_slow_weight_gradients(period):
    model = predictor.make_head("frozen", 75, period)
    kernel = memory.QueryMemory(memory.Config("trace_delta", key_dim=3))
    features, scores, mask = packet(period, length=2 * period + 2)
    lengths, ends = torch.tensor([features.shape[1]]), torch.tensor([True])
    slow = model(features, scores, lengths, mask, episode_ends=ends)
    weight = projection().requires_grad_()
    result = apply(kernel, slow, scores, mask, lengths, ends, weight)
    result.action_prediction[0, -1].square().sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not slow.keys_hidden.requires_grad and not slow.action_prediction.requires_grad


@pytest.mark.parametrize("period", (4, 8))
def test_query_write_targets_complete_forecast_including_nonzero_static_residual(period):
    model = predictor.make_head("frozen", 76, period)
    with torch.no_grad():
        model.action_residual.bias.copy_(torch.tensor([.3, -.2, .1, .4]))
    cfg = memory.Config("trace_delta", key_dim=3, trace_decay=0)
    kernel = memory.QueryMemory(cfg)
    features, scores, mask = packet(period, length=period + 1)
    lengths, ends = torch.tensor([period + 1]), torch.tensor([True])
    slow = model(features, scores, lengths, mask, episode_ends=ends)
    result = apply(kernel, slow, scores, mask, lengths, ends, projection())
    assert not torch.equal(slow.prior[0, period], slow.shadow_prior[0, period])
    assert result.work["matrix_writes"] == 1
    z = result.carry.trace[0]
    z = z / torch.linalg.vector_norm(z).clamp_min(cfg.epsilon)
    target = (scores[0, period] - slow.shadow_prior[0, period]) / 64
    target = target - target.mean()
    expected = cfg.step_size * target[:, None] * z[None, :] / (cfg.epsilon + z.square().sum())
    torch.testing.assert_close(result.carry.matrix[0], expected, rtol=2e-6, atol=1e-7)
    # Deliberate negative control: omitting the static residual is observably wrong.
    incomplete = apply(kernel, replace(slow, shadow_prior=slow.prior), scores, mask,
                       lengths, ends, projection())
    assert not torch.allclose(incomplete.carry.matrix, result.carry.matrix)
