"""Fabricated composition contracts; no files, fitting, games or empirical data."""
from dataclasses import replace

import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_model as composed
from openjev.research import otto_scheduled_predictor as predictor


def packet(period, lengths=(65, 37, 1)):
    batch, span = len(lengths), max(lengths)
    features = torch.full((batch, span, 31), float("nan"))
    scores = torch.full((batch, span, 4), float("nan"))
    mask = torch.zeros(batch, span, dtype=torch.bool)
    for row, length in enumerate(lengths):
        steps = torch.arange(length)
        features[row, :length] = torch.linspace(-.3, .7, length * 31).reshape(length, 31) + row / 10
        features[row, :length, 15] = steps / 2188
        features[row, :length, 16] = (steps % period) / 2188
        features[row, :length, 17] = 1
        mask[row, :length] = steps % period == 0
        scores[row, :length][mask[row, :length]] = torch.tensor([4., 7., 2., 6.]) + (
            steps[mask[row, :length]][:, None] * torch.tensor([.125, -.25, .5, .25]))
        scores[row, 0, 0] = -.0
    return {"features": features, "query_scores": scores, "query_mask": mask,
            "lengths": torch.tensor(lengths), "episode_ends": torch.ones(batch, dtype=torch.bool)}


def part(values, start, stop):
    result = {name: value[:, start:stop].clone() for name, value in values.items()
              if name not in ("lengths", "episode_ends")}
    result["lengths"] = (values["lengths"] - start).clamp(0, stop - start)
    result["episode_ends"] = values["episode_ends"] & (values["lengths"] <= stop)
    return result


def state(period=4, seed=11):
    slow = predictor.make_head("frozen", seed, period)
    with torch.no_grad():
        slow.output.weight.copy_(torch.arange(4 * 28).sin().reshape(4, 28) * .02)
        slow.action_residual.weight.copy_(torch.arange(4 * 28).cos().reshape(4, 28) * .003)
        slow.action_residual.bias.copy_(torch.tensor([.3, -.2, .1, .4]))
    return {name: value.detach().clone() for name, value in slow.state_dict().items()}


def weight(dimension=8):
    return torch.arange(dimension * 28, dtype=torch.float32).sin().reshape(dimension, 28) * .07


def model(mode, period=4, *, slow_mode="frozen", dimension=8, **kwargs):
    cfg = memory.Config(mode, dimension, **kwargs)
    projection = weight(dimension) if mode in memory.MATRIX_MODES else None
    return composed.from_states(cfg, 19, period, state(period), projection, slow_mode=slow_mode)


def bits(value):
    return value.detach().contiguous().numpy().tobytes()


@pytest.mark.parametrize("mode", memory.MODES)
def test_factories_preserve_rng_copy_exact_states_without_aliases_and_report_actual_parameters(mode):
    cfg = memory.Config(mode, key_dim=8)
    supplied = state()
    projection = weight() if mode in memory.MATRIX_MODES else None
    before = torch.random.get_rng_state().clone()
    result = composed.from_states(cfg, 23, 4, supplied, projection)
    assert torch.equal(before, torch.random.get_rng_state())
    for name, value in result.slow.state_dict().items():
        assert torch.equal(value, supplied[name]) and value.data_ptr() != supplied[name].data_ptr()
    metadata = result.parameter_metadata()
    if projection is None:
        assert result.projection is None and len(result.state_dict()) == 8
        assert metadata["count"] == 6112 and metadata["effective_count"] == 0
    else:
        assert result.projection.bias is None and len(result.state_dict()) == 9
        assert torch.equal(result.projection.weight, projection)
        assert result.projection.weight.data_ptr() != projection.data_ptr()
        assert metadata["count"] == 6336 and metadata["effective_count"] == 224
        assert metadata["effective_names"] == ["projection.weight"]
    assert metadata["requires_grad_count"] == 116 + (224 if projection is not None else 0)
    assert sum(p.numel() for _, p in result.effective_named_parameters()) == metadata["effective_count"]
    with torch.no_grad():
        next(iter(result.slow.parameters())).add_(1)
    assert not torch.equal(next(iter(result.slow.parameters())), supplied["recurrent.weight_ih_l0"])


@pytest.mark.parametrize("mode", ("none", "last_error"))
def test_supplied_slow_wrapper_never_constructs_unused_projection_or_mutates_slow(mode, monkeypatch):
    slow = predictor.make_head("frozen", 71, 4)
    before = {name: bits(value) for name, value in slow.state_dict().items()}
    flags = [p.requires_grad for p in slow.parameters()]
    rng = torch.random.get_rng_state().clone()

    def forbidden(*_args, **_kwargs):
        pytest.fail("unpadded baseline constructed a Linear")

    monkeypatch.setattr(torch.nn, "Linear", forbidden)
    result = composed.QueryMemoryModel(slow, memory.Config(mode, 8), 4)
    assert result.slow is slow and result.projection is None
    assert before == {name: bits(value) for name, value in slow.state_dict().items()}
    assert flags == [p.requires_grad for p in slow.parameters()]
    assert torch.equal(rng, torch.random.get_rng_state())


def test_default_key_width_and_seeded_projection_are_identical_across_matrix_modes():
    default = composed.make_model(seed=5)
    assert default.config.key_dim == 8 and default.projection is None
    rng = torch.random.get_rng_state().clone()
    candidates = [composed.make_model(memory.Config(mode, 8), seed=5) for mode in memory.MATRIX_MODES]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(item.projection.weight, candidates[0].projection.weight) for item in candidates)
    assert len({item.projection.weight.data_ptr() for item in candidates}) == 4


@pytest.mark.parametrize("period", (4, 8))
@pytest.mark.parametrize("mode", memory.MATRIX_MODES)
def test_only_projection_receives_gradients_while_frozen_predictor_and_targets_remain_unchanged(period, mode):
    item = model(mode, period)
    values = packet(period, (2 * period + 2,))
    values["features"].requires_grad_(True)
    values["query_scores"].requires_grad_(True)
    before = {name: bits(value) for name, value in item.slow.state_dict().items()}
    result = item(**values)
    loss = result.action_prediction[0, -1].square().sum() + result.corrected_shadow_prior[0, 2 * period].square().sum()
    loss.backward()
    gradient = item.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(p.grad is None for p in item.slow.parameters())
    assert values["features"].grad is values["query_scores"].grad is None
    assert not result.shadow_prior.requires_grad and not result.slow_action_prediction.requires_grad
    assert result.action_prediction.requires_grad and result.corrected_shadow_prior.requires_grad
    assert result.parameter_metadata["effective_names"] == ["projection.weight"]
    assert before == {name: bits(value) for name, value in item.slow.state_dict().items()}


@pytest.mark.parametrize("period", (4, 8))
def test_ordinary_none_joint_retains_original_action_and_prior_training_graphs(period):
    item = model("none", period, slow_mode="joint")
    values = packet(period, (2 * period + 2,))
    result = item(**values)
    assert result.action_prediction is result.slow_action_prediction
    assert result.prediction.requires_grad and result.prior.requires_grad and result.shadow_prior.requires_grad
    assert result.corrected_shadow_prior.requires_grad
    assert bits(result.corrected_shadow_prior) == bits(result.shadow_prior)
    loss = result.action_prediction[0, -1].square().sum() + result.prior[0, period].square().sum()
    loss = loss + result.corrected_shadow_prior[0, 2 * period].square().sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in item.slow.parameters())
    assert item.slow.recurrent.weight_ih_l0.grad.abs().sum() > 0
    assert item.slow.action_residual.weight.grad.abs().sum() > 0
    assert item.projection is None and result.parameter_metadata["effective_count"] == 6112
    assert len(item.effective_named_parameters()) == 8


@pytest.mark.parametrize("period", (4, 8))
@pytest.mark.parametrize("mode", memory.MODES)
def test_whole_and_detached_chunks_match_outputs_states_and_actual_work(period, mode):
    item = model(mode, period)
    values = packet(period)
    whole = item(**values)
    chunks, carry = [], None
    for start, stop in ((0, 32), (32, 64), (64, 65)):
        result = item(**part(values, start, stop), carry=carry)
        chunks.append(result)
        carry = composed.detach_carry(result.carry)
    for name in ("prediction", "slow_action_prediction", "prior", "shadow_prior", "action_prediction",
                 "corrected_shadow_prior", "prewrite_correction", "prior_mask", "key_mask"):
        assert bits(torch.cat([getattr(c, name) for c in chunks], dim=1)) == bits(getattr(whole, name))
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        assert bits(getattr(carry.slow.base, name)) == bits(getattr(whole.carry.slow.base, name))
    for name in ("matrix", "trace", "last_error", "has_query", "absolute_step", "ended"):
        assert bits(getattr(carry.fast, name)) == bits(getattr(whole.carry.fast, name))
    assert all(sum(c.work_counts[key] for c in chunks) == value for key, value in whole.work_counts.items())
    assert all(sum(c.memory_work_units[key] for c in chunks) == value for key, value in whole.memory_work_units.items())


@pytest.mark.parametrize("period", (4, 8))
def test_query_padding_bits_causal_prewrite_and_corrected_shadow_scope(period):
    item = model("trace_delta", period)
    values = packet(period, (2 * period + 2, period + 1))
    original = item(**values)
    changed = {name: value.clone() for name, value in values.items()}
    changed["query_scores"][0, period] += torch.tensor([8., -4., 3., -7.])
    other = item(**changed)
    assert bits(original.corrected_shadow_prior[:, :period + 1]) == bits(other.corrected_shadow_prior[:, :period + 1])
    assert bits(original.prewrite_correction[:, :period + 1]) == bits(other.prewrite_correction[:, :period + 1])
    assert not torch.equal(original.prewrite_correction[0, period + 1], other.prewrite_correction[0, period + 1])
    assert bits(original.action_prediction[values["query_mask"]]) == bits(values["query_scores"][values["query_mask"]])
    active = torch.arange(values["features"].shape[1])[None] < values["lengths"][:, None]
    assert not original.action_prediction[~active].any()
    assert bits(original.action_prediction[~active]) == bits(original.slow_action_prediction[~active])
    assert not original.corrected_shadow_prior[~original.prior_mask].any()
    assert bits(original.corrected_shadow_prior[~original.prior_mask]) == bits(original.shadow_prior[~original.prior_mask])


@pytest.mark.parametrize("period", (4, 8))
def test_first_write_uses_full_nonzero_static_residual_shadow_not_incomplete_base_prior(period):
    item = model("trace_delta", period, trace_decay=0)
    values = packet(period, (period + 1,))
    result = item(**values)
    assert not torch.equal(result.prior[0, period], result.shadow_prior[0, period])
    cue = result.carry.fast.trace[0]
    cue = cue / torch.linalg.vector_norm(cue).clamp_min(item.config.epsilon)
    target = (values["query_scores"][0, period] - result.shadow_prior[0, period]) / 64
    target = target - target.mean()
    expected = item.config.step_size * target[:, None] * cue[None] / (item.config.epsilon + cue.square().sum())
    torch.testing.assert_close(result.carry.fast.matrix[0], expected, rtol=2e-6, atol=1e-7)
    wrong_target = (values["query_scores"][0, period] - result.prior[0, period]) / 64
    wrong_target = wrong_target - wrong_target.mean()
    wrong = item.config.step_size * wrong_target[:, None] * cue[None] / (item.config.epsilon + cue.square().sum())
    assert not torch.allclose(result.carry.fast.matrix[0], wrong)


@pytest.mark.parametrize("mode", memory.MODES)
def test_no_write_uses_same_weights_preserves_baseline_bits_and_skips_only_write_work(mode):
    item = model(mode)
    values = packet(4, (10, 5))
    before = {name: bits(value) for name, value in item.state_dict().items()}
    regular = item(**values)
    intervention = item(**values, no_write=True)
    assert bits(intervention.action_prediction) == bits(intervention.slow_action_prediction)
    assert bits(intervention.corrected_shadow_prior) == bits(intervention.shadow_prior)
    assert bits(regular.slow_action_prediction) == bits(intervention.slow_action_prediction)
    assert bits(regular.shadow_prior) == bits(intervention.shadow_prior)
    assert before == {name: bits(value) for name, value in item.state_dict().items()}
    assert intervention.work_counts["memory_matrix_writes"] == intervention.work_counts["memory_last_error_writes"] == 0
    assert intervention.work_counts["memory_innovation_calculations"] == 0
    assert intervention.work_counts["projection_calls"] == regular.work_counts["projection_calls"]
    assert intervention.work_counts["memory_matrix_reads"] == regular.work_counts["memory_matrix_reads"]


def test_projection_geometry_and_executed_rows_include_partial_padding_without_flattening_time():
    item = model("trace_delta")
    calls = []
    original = item.projection.forward

    def record(value):
        calls.append(tuple(value.shape))
        return original(value)

    item.projection.forward = record
    result = item(**packet(4, (10, 5, 1)))
    assert calls == [(3, 28)] * 9
    assert result.work_counts["projection_calls"] == 9
    assert result.work_counts["projection_rows"] == 27
    assert result.work_counts["projection_key_rows"] == 9 + 4
    assert result.work_counts["projection_linear_terms"] == 27 * 28 * 8


@pytest.mark.parametrize("defect", ("step", "ended", "batch", "schedule", "fast_graph", "slow_graph"))
def test_carry_join_rejects_mismatch_or_attached_state_before_either_forward(defect, monkeypatch):
    item = model("trace_delta")
    carry = item.initial_carry(1)
    if defect == "step":
        carry = replace(carry, fast=replace(carry.fast, absolute_step=torch.tensor([4]), has_query=torch.tensor([True])))
    elif defect == "ended":
        slow_base = replace(carry.slow.base, absolute_step=torch.tensor([4]), has_query=torch.tensor([True]))
        fast = replace(carry.fast, absolute_step=torch.tensor([4]), has_query=torch.tensor([True]), ended=torch.tensor([True]))
        carry = composed.Carry(replace(carry.slow, base=slow_base), fast)
    elif defect == "batch":
        carry = item.initial_carry(2)
    elif defect == "schedule":
        carry = replace(carry, slow=replace(carry.slow, query_period=8))
    elif defect == "fast_graph":
        carry = replace(carry, fast=replace(carry.fast, matrix=carry.fast.matrix.clone().requires_grad_()))
    else:
        base = replace(carry.slow.base, hidden=carry.slow.base.hidden.clone().requires_grad_())
        carry = replace(carry, slow=replace(carry.slow, base=base))

    def forbidden(*_args, **_kwargs):
        pytest.fail("forward executed before carry admission")

    monkeypatch.setattr(item.slow, "forward", forbidden)
    monkeypatch.setattr(item.memory, "forward", forbidden)
    with pytest.raises(ValueError):
        item(**packet(4, (8,)), carry=carry)


def test_detach_owns_both_states_and_ended_zero_length_lane_remains_untouched():
    item = model("trace_delta")
    values = packet(4, (8, 1))
    values["episode_ends"][0] = False
    first = item(**values)
    detached = composed.detach_carry(first.carry)
    for name in ("hidden", "raw_anchor", "has_query", "absolute_step", "ended"):
        original, copied = getattr(first.carry.slow.base, name), getattr(detached.slow.base, name)
        assert original.data_ptr() != copied.data_ptr() and copied.grad_fn is None
    for name in ("matrix", "trace", "last_error", "has_query", "absolute_step", "ended"):
        original, copied = getattr(first.carry.fast, name), getattr(detached.fast, name)
        assert original.data_ptr() != copied.data_ptr() and copied.grad_fn is None
    next_values = part(packet(4, (12, 1)), 8, 12)
    next_result = item(**next_values, carry=detached)
    for name in ("matrix", "trace", "last_error", "has_query", "absolute_step", "ended"):
        assert bits(getattr(next_result.carry.fast, name)[1]) == bits(getattr(detached.fast, name)[1])
    assert not next_result.action_prediction[1].any()


@pytest.mark.parametrize("defect", ("missing", "extra", "slow_dtype", "slow_nan", "projection_shape", "projection_nan", "projection_dtype", "unused_projection", "missing_projection", "joint_memory", "seed"))
def test_from_states_rejects_invalid_inputs_before_constructing_a_predictor(defect, monkeypatch):
    supplied, projected = state(), weight()
    config, slow_mode, seed = memory.Config("trace_delta", 8), "frozen", 4
    if defect == "missing":
        supplied.pop("output.bias")
    elif defect == "extra":
        supplied["unknown"] = torch.zeros(1)
    elif defect == "slow_dtype":
        supplied["output.bias"] = supplied["output.bias"].double()
    elif defect == "slow_nan":
        supplied["output.bias"][0] = float("nan")
    elif defect == "projection_shape":
        projected = projected.T
    elif defect == "projection_nan":
        projected[0, 0] = float("nan")
    elif defect == "projection_dtype":
        projected = projected.double()
    elif defect == "unused_projection":
        config = memory.Config("none", 8)
    elif defect == "missing_projection":
        projected = None
    elif defect == "joint_memory":
        slow_mode = "joint"
    else:
        seed = True

    def forbidden(*_args, **_kwargs):
        pytest.fail("predictor constructed before full state validation")

    monkeypatch.setattr(predictor, "from_state", forbidden)
    with pytest.raises(ValueError):
        composed.from_states(config, seed, 4, supplied, projected, slow_mode=slow_mode)
