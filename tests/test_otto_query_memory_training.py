"""Fabricated complete-episode optimization contracts, without empirical IO."""
from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_data as data_module
from openjev.research import otto_query_memory_model as models
from openjev.research import otto_query_memory_training as training


def dataset(lengths=(65, 9, 1), *, stage="train", period=4):
    total = sum(lengths)
    features = np.linspace(-.5, .7, total * 31, dtype=np.float32).reshape(total, 31)
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    correction = np.zeros(total, np.bool_)
    target = np.zeros((total, 4), np.float32)
    for lane, (low, high) in enumerate(pairwise(offsets)):
        steps = np.arange(high - low)
        features[low:high, 15] = steps / 2188
        features[low:high, 16] = steps % 4 / 2188
        features[low:high, 17] = 1
        correction[low:high] = steps % 4 == 0
        target[low:high] = np.array([4, 7, 2, 6], np.float32) + lane / 8 + steps[:, None] * np.array(
            [.125, -.25, .5, .25], np.float32)
    legal = np.ones((total, 4), np.bool_)
    legal[1::3, 3] = False
    flat = {"features": features, "raw_q": target, "legal": legal,
            "actions": np.zeros(total, np.int64), "correction": correction, "episode_offsets": offsets}
    identities = [{"stage": stage, "episode_id": f"{stage}:fabricated:{i}", "episode_index": i,
                   "seed": 100 + i, "case": i, "regime": "lambda3", "arm": "analytic"}
                  for i in range(len(lengths))]
    return data_module.project_census(flat, identities, query_period=period, expected_stage=stage)


def model(mode="trace_delta"):
    item = models.make_model(memory.Config(mode, key_dim=8), seed=71,
                             slow_mode="joint" if mode == "none" else "frozen")
    with torch.no_grad():
        item.slow.output.weight.copy_(torch.arange(112).sin().reshape(4, 28) * .02)
        item.slow.action_residual.weight.copy_(torch.arange(112).cos().reshape(4, 28) * .003)
        item.slow.action_residual.bias.copy_(torch.tensor([.3, -.2, .1, .4]))
    return item


def clone(item):
    return models.from_states(item.config, item.seed, 4, item.slow.state_dict(),
        None if item.projection is None else item.projection.weight, slow_mode=item.slow.mode)


def bits(value):
    return value.detach().contiguous().numpy().tobytes()


def state_bits(item):
    return {name: bits(value) for name, value in item.state_dict().items()}


def manual_update(item, optimizer, data, indices):
    """Separate explicit chunk loop and one final Adam step, no kernel helpers."""
    optimizer.zero_grad(set_to_none=True)
    carry = item.initial_carry(len(indices))
    lengths = [int(data["episode_offsets"][i + 1] - data["episode_offsets"][i]) for i in indices]
    totals = {"total": 0., "nonquery": 0., "prior": 0.}
    backwards = 0
    for start in range(0, max(lengths), 32):
        packet = data_module.batch_chunk(data, indices, start)
        inputs = {name: torch.from_numpy(value.copy()) for name, value in packet["model_inputs"].items()}
        result = item(**inputs, carry=carry)
        terms = data_module.weighted_loss(result.action_prediction, result.corrected_shadow_prior,
            torch.from_numpy(packet["targets"].copy()), torch.from_numpy(packet["legal"].copy()),
            torch.from_numpy(packet["nonquery_weights"].copy()), torch.from_numpy(packet["prior_weights"].copy()),
            inputs["query_mask"], torch.from_numpy(packet["prior_mask"].copy()), episode_count=data["episode_count"])
        for name in totals:
            totals[name] += float(terms[name].detach())
        if terms["total"].requires_grad:
            terms["total"].backward()
            backwards += 1
        carry = models.detach_carry(result.carry)
    effective = [p for _, p in item.effective_named_parameters()]
    for parameter in effective:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    norm = torch.nn.utils.clip_grad_norm_(effective, 5., error_if_nonfinite=True)
    optimizer.step()
    return totals, backwards, float(norm)


@pytest.mark.parametrize("mode", ("none", *memory.MATRIX_MODES))
def test_optimizer_uses_only_effective_parameters_and_does_not_draw_rng(mode):
    item = model(mode)
    before = torch.random.get_rng_state().clone()
    optimizer = training.construct_optimizer(item)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not optimizer.state
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for _, p in item.effective_named_parameters()]
    assert optimizer.param_groups[0]["lr"] == .003
    assert optimizer.param_groups[0]["weight_decay"] == 0
    if mode != "none":
        assert item.slow.action_residual.weight.requires_grad
        assert all(id(p) != id(item.slow.action_residual.weight) for p in optimizer.param_groups[0]["params"])


@pytest.mark.parametrize("mode", ("none", *memory.MATRIX_MODES))
def test_full_episode_update_matches_independent_manual_accumulation_bitwise(mode):
    actual = model(mode)
    reference = clone(actual)
    data = dataset()
    optimizer = training.construct_optimizer(actual)
    other = torch.optim.Adam([p for _, p in reference.effective_named_parameters()], lr=.003, weight_decay=0.)
    recorded = training.batch_update(actual, optimizer, data, [0, 1, 2])
    expected, backwards, norm = manual_update(reference, other, data, [0, 1, 2])
    assert state_bits(actual) == state_bits(reference)
    assert recorded["loss"] == expected["total"]
    assert recorded["nonquery_loss"] == expected["nonquery"]
    assert recorded["prior_loss"] == expected["prior"]
    assert recorded["gradient_norm_before_clip"] == norm
    assert recorded["backward_chunks"] == backwards
    assert recorded["optimizer_updates"] == recorded["optimizer_step"] == 1
    for (_, left), (_, right) in zip(actual.effective_named_parameters(), reference.effective_named_parameters(), strict=True):
        assert {name: bits(value) for name, value in optimizer.state[left].items()} == {
            name: bits(value) for name, value in other.state[right].items()}


def test_weights_stay_fixed_across_chunks_and_carry_is_detached_before_reuse():
    item = model()
    optimizer = training.construct_optimizer(item)
    data = dataset()
    before = state_bits(item)
    clocks, ends, lengths = [], [], []

    def inspect(_module, _args, kwargs):
        assert state_bits(item) == before
        assert not optimizer.state
        carry = kwargs["carry"]
        assert carry.fast.matrix.grad_fn is carry.fast.trace.grad_fn is None
        assert carry.slow.base.hidden.grad_fn is None
        clocks.append(carry.fast.absolute_step.tolist())
        ends.append(carry.fast.ended.tolist())
        lengths.append(kwargs["lengths"].tolist())

    hook = item.register_forward_pre_hook(inspect, with_kwargs=True)
    result = training.batch_update(item, optimizer, data, [0, 1, 2])
    hook.remove()
    assert clocks == [[0, 0, 0], [32, 9, 1], [64, 9, 1]]
    assert ends == [[False, False, False], [False, True, True], [False, True, True]]
    assert lengths == [[32, 9, 1], [32, 0, 0], [1, 0, 0]]
    assert result["forward_rows"] == 75
    assert result["episode_exposures"] == 3
    assert result["forward_chunks"] == 3
    assert state_bits(item.slow) == {name.removeprefix("slow."): value for name, value in before.items()
                                   if name.startswith("slow.")}
    assert bits(item.projection.weight) != before["projection.weight"]
    assert all(p.grad is None and p not in optimizer.state for p in item.slow.parameters())


def test_chunk_tensors_own_storage_separate_from_numpy_packets(monkeypatch):
    item = model()
    optimizer = training.construct_optimizer(item)
    original = data_module.batch_chunk
    packet_holder = []

    def observed(*args, **kwargs):
        packet = original(*args, **kwargs)
        packet_holder.append(packet)
        return packet

    def inspect(_module, _args, kwargs):
        packet = packet_holder[-1]
        for name, value in packet["model_inputs"].items():
            assert kwargs[name].data_ptr() != value.__array_interface__["data"][0]

    monkeypatch.setattr(data_module, "batch_chunk", observed)
    hook = item.register_forward_pre_hook(inspect, with_kwargs=True)
    training.batch_update(item, optimizer, dataset((9, 1)), [0, 1])
    hook.remove()


@pytest.mark.parametrize("mode", ("none", "trace_delta"))
def test_query_only_batch_has_no_backward_but_zero_fills_effective_adam_schedule(mode):
    item = model(mode)
    optimizer = training.construct_optimizer(item)
    before = state_bits(item)
    result = training.batch_update(item, optimizer, dataset((1, 1)), [0, 1])
    assert result["loss"] == result["nonquery_loss"] == result["prior_loss"] == 0
    assert result["forward_chunks"] == result["no_gradient_chunks"] == 1
    assert result["backward_chunks"] == result["loss_chunks"] == 0
    assert result["zero_filled_gradient_names"] == [name for name, _ in item.effective_named_parameters()]
    assert state_bits(item) == before
    assert result["gradient_norm_before_clip"] == 0
    for _, parameter in item.effective_named_parameters():
        assert torch.count_nonzero(parameter.grad) == 0
        assert float(optimizer.state[parameter]["step"]) == 1
        assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"]) == 0
    assert result["frozen_optimizer_state_entries"] == 0


def test_query_only_later_batch_retains_adam_momentum_instead_of_skipping_update():
    item = model()
    optimizer = training.construct_optimizer(item)
    first = training.batch_update(item, optimizer, dataset((13,)), [0])
    weight = item.projection.weight
    assert first["gradient_norm_before_clip"] > 0
    before = bits(weight)
    result = training.batch_update(item, optimizer, dataset((1,)), [0])
    assert result["optimizer_step"] == 2 and result["optimizer_updates"] == 1
    assert result["backward_chunks"] == 0
    assert result["zero_filled_gradient_names"] == ["projection.weight"]
    assert bits(weight) != before


def test_projection_receives_gradient_after_informative_query_write():
    item = model()
    optimizer = training.construct_optimizer(item)
    original = bits(item.projection.weight)
    result = training.batch_update(item, optimizer, dataset((13,)), [0])
    assert result["gradient_norm_before_clip"] > 0
    assert torch.isfinite(item.projection.weight.grad).all()
    assert item.projection.weight.grad.abs().sum() > 0
    assert bits(item.projection.weight) != original
    assert result["work_counts"]["memory_matrix_writes"] == 3
    assert result["work_counts"]["projection_key_rows"] == 12
    assert result["work_counts"]["projection_calls"] == 12
    assert result["prior_rows"] == 3 and result["nonquery_rows"] == 9


def test_ordinary_joint_none_updates_recurrent_and_static_residual_parameters():
    item = model("none")
    before = state_bits(item)
    optimizer = training.construct_optimizer(item)
    result = training.batch_update(item, optimizer, dataset((13,)), [0])
    assert result["effective_parameter_count"] == 6112
    assert bits(item.slow.recurrent.weight_ih_l0) != before["slow.recurrent.weight_ih_l0"]
    assert bits(item.slow.action_residual.weight) != before["slow.action_residual.weight"]
    assert all(p.grad is not None for p in item.slow.parameters())
    assert result["work_counts"]["projection_calls"] == 0


def test_full_denominator_scaling_cancels_to_actual_batch_mean_even_for_54_paths():
    small = dataset((13, 1))
    large = dataset((13, *([1] * 53)))
    # Keep selected features identical; physical cohort creation otherwise changes
    # the linspace features. Projected data fields are immutable owned arrays.
    copied = dict(large)
    updated = large["features"].copy()
    updated[:14] = small["features"]
    copied["features"] = updated
    left = model()
    right = clone(left)
    a = training.batch_update(left, training.construct_optimizer(left), small, [0, 1])
    b = training.batch_update(right, training.construct_optimizer(right), copied, [0, 1])
    assert b["loss"] == pytest.approx(a["loss"], rel=2e-6, abs=1e-8)
    assert b["gradient_norm_before_clip"] == pytest.approx(a["gradient_norm_before_clip"], rel=2e-6, abs=1e-8)


def test_callbacks_and_return_values_report_actual_work_without_tensors():
    events, guards = [], []
    item = model()
    result = training.batch_update(item, training.construct_optimizer(item), dataset((9,)), [0],
        check=lambda: guards.append(True), stage=lambda name, start: events.append((name, start)))
    assert events[0] == ("zero_grad", None)
    assert events[-1] == ("optimizer_update", None)
    assert events.count(("chunk_forward", 0)) == events.count(("chunk_backward", 0)) == 1
    assert guards
    assert result["work_counts"]["slow_active_rows"] == 9
    assert all(type(v) is int and v >= 0 for v in result["memory_work_units"].values())
    assert all(math.isfinite(v) and v >= 0 for v in result["timing_seconds"].values())
    assert result["timing_seconds"]["wall"] >= result["timing_seconds"]["forward"]

    def check_plain(value):
        assert not isinstance(value, (torch.Tensor, np.ndarray))
        if isinstance(value, dict):
            for child in value.values():
                check_plain(child)
        elif isinstance(value, list):
            for child in value:
                check_plain(child)

    check_plain(result)


def test_outer_no_grad_context_does_not_silently_disable_training():
    item = model()
    optimizer = training.construct_optimizer(item)
    with torch.no_grad():
        result = training.batch_update(item, optimizer, dataset((13,)), [0])
        assert not torch.is_grad_enabled()
    assert result["backward_chunks"] == 1 and result["gradient_norm_before_clip"] > 0


@pytest.mark.parametrize("stage", ("dev", "test"))
def test_rejects_nontraining_split_before_forward(stage):
    item = model()
    optimizer = training.construct_optimizer(item)
    with pytest.raises(ValueError, match="TRAIN"):
        training.batch_update(item, optimizer, dataset(stage=stage), [0])
    assert not optimizer.state


@pytest.mark.parametrize("defect", ("extra", "missing", "order", "lr", "weight_decay", "foreign_state", "frozen_grad"))
def test_rejects_optimizer_ownership_or_recipe_errors_before_updates(defect):
    item = model("none" if defect in ("missing", "order") else "trace_delta")
    optimizer = training.construct_optimizer(item)
    group = optimizer.param_groups[0]
    if defect == "extra":
        group["params"].append(item.slow.action_residual.weight)
    elif defect == "missing":
        group["params"].pop()
    elif defect == "order":
        group["params"].reverse()
    elif defect in ("lr", "weight_decay"):
        group[defect] = 1
    elif defect == "foreign_state":
        optimizer.state[item.slow.output.weight] = {}
    else:
        item.slow.action_residual.weight.grad = torch.ones_like(item.slow.action_residual.weight)
    with pytest.raises(ValueError):
        training.batch_update(item, optimizer, dataset((9,)), [0])


@pytest.mark.parametrize("mode", ("none", "last_error"))
def test_frozen_nonfitting_views_cannot_construct_optimizer(mode):
    item = models.make_model(memory.Config(mode, key_dim=8), seed=11)
    with pytest.raises(ValueError, match="effective optimization"):
        training.construct_optimizer(item)


def test_dead_zero_projection_is_rejected_without_reinitialization():
    item = model()
    with torch.no_grad():
        item.projection.weight.zero_()
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="nonzero key projection"):
        training.construct_optimizer(item)
    assert torch.count_nonzero(item.projection.weight) == 0
    assert torch.equal(before, torch.random.get_rng_state())


def test_nonfinite_gradient_stops_before_adam_without_retry():
    item = model()
    optimizer = training.construct_optimizer(item)
    before = state_bits(item)
    hook = item.projection.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    with pytest.raises(ValueError, match="finite effective gradient"):
        training.batch_update(item, optimizer, dataset((13,)), [0])
    hook.remove()
    assert not optimizer.state and state_bits(item) == before


def test_guard_abort_before_optimizer_preserves_failure_without_step():
    item = model()
    optimizer = training.construct_optimizer(item)
    before = state_bits(item)

    def stage(name, _start):
        if name == "optimizer_update":
            raise RuntimeError("fabricated admission stop")

    with pytest.raises(RuntimeError, match="fabricated admission stop"):
        training.batch_update(item, optimizer, dataset((13,)), [0], stage=stage)
    assert not optimizer.state and state_bits(item) == before
