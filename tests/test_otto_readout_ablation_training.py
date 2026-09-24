"""Fabricated readout-mask optimization invariants; no empirical data or IO."""
from __future__ import annotations

import ast
import inspect
from itertools import pairwise

import numpy as np
import pytest
import torch

from openjev.research import otto_query_memory as memory
from openjev.research import otto_query_memory_data as data_module
from openjev.research import otto_query_memory_model as original_models
from openjev.research import otto_query_memory_training as original_training
from openjev.research import otto_readout_ablation_model as models
from openjev.research import otto_readout_ablation_training as training


def bits(value):
    return value.detach().contiguous().numpy().tobytes()


def state_bits(model):
    return {name: bits(value) for name, value in model.state_dict().items()}


def dataset(lengths=(65, 33, 9, 5, 2, 1), *, stage="train"):
    offsets = np.array([0, *np.cumsum(lengths)], dtype=np.int64)
    rows = int(offsets[-1])
    values = np.linspace(-.4, .6, rows * 31, dtype=np.float32).reshape(rows, 31)
    target = np.empty((rows, 4), dtype=np.float32)
    correction = np.zeros(rows, dtype=np.bool_)
    for lane, (low, high) in enumerate(pairwise(offsets)):
        step = np.arange(int(high - low))
        values[low:high, 15] = step / 2188
        values[low:high, 16] = step % 4 / 2188
        values[low:high, 17] = 1
        target[low:high] = np.array([2., 7., 4., 8.], dtype=np.float32) + lane * .125 + step[:, None] * np.array(
            [.25, -.125, .5, -.25], dtype=np.float32)
        correction[low:high] = step % 4 == 0
    legal = np.ones((rows, 4), dtype=np.bool_)
    legal[1::3, 3] = False
    flat = {"features": values, "raw_q": target, "legal": legal, "actions": np.zeros(rows, dtype=np.int64),
            "correction": correction, "episode_offsets": offsets}
    identities = [{"stage": stage, "episode_id": f"{stage}:fabricated-readout:{i}", "episode_index": i,
                   "seed": 700 + i, "case": i, "regime": "lambda3", "arm": "analytic"}
                  for i in range(len(lengths))]
    return data_module.project_census(flat, identities, query_period=4, expected_stage=stage)


def model(arm):
    result = models.make_model(arm, seed=271)
    with torch.no_grad():
        result.slow.output.weight.copy_(torch.arange(112).sin().reshape(4, 28) * .03)
        result.slow.output.bias.copy_(torch.tensor([.1, -.2, .05, .3]))
        result.slow.action_residual.weight.copy_(torch.arange(112).cos().reshape(4, 28) * .002)
        result.slow.action_residual.bias.copy_(torch.tensor([.03, -.02, .01, -.01]))
    return result


def manual_update(item, optimizer, data, indices):
    """Explicit independent control flow over the qualified loss and chunks."""
    optimizer.zero_grad(set_to_none=True)
    carry = item.initial_carry(len(indices))
    maximum = max(int(data["episode_offsets"][i + 1] - data["episode_offsets"][i]) for i in indices)
    totals = {key: 0. for key in ("total", "nonquery", "prior")}
    for start in range(0, maximum, 32):
        packet = data_module.batch_chunk(data, indices, start)
        inputs = {key: torch.from_numpy(value.copy()) for key, value in packet["model_inputs"].items()}
        forecast = item(**inputs, carry=carry)
        losses = data_module.weighted_loss(forecast.action_prediction, forecast.corrected_shadow_prior,
            torch.from_numpy(packet["targets"].copy()), torch.from_numpy(packet["legal"].copy()),
            torch.from_numpy(packet["nonquery_weights"].copy()), torch.from_numpy(packet["prior_weights"].copy()),
            inputs["query_mask"], torch.from_numpy(packet["prior_mask"].copy()), episode_count=data["episode_count"])
        for key in totals:
            totals[key] += float(losses[key].detach())
        if losses["total"].requires_grad:
            losses["total"].backward()
        carry = original_models.detach_carry(forecast.carry)
    parameters = [p for _, p in item.effective_named_parameters()]
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    optimizer.step()
    return totals, float(norm)


def test_training_fork_preserves_every_recipe_body_except_type_admission():
    def functions(module):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(inspect.getsource(module)).body if isinstance(node, ast.FunctionDef)}
    old, new = functions(original_training), functions(training)
    assert set(old) == set(new)
    assert {k: v for k, v in old.items() if k != "_effective"} == {k: v for k, v in new.items() if k != "_effective"}
    assert models.ReadoutPredictor._scheduled_forward is original_models.predictor.ScheduledPredictor._scheduled_forward
    assert models.ReadoutAblationModel.forward is original_models.QueryMemoryModel.forward


@pytest.mark.parametrize("arm", models.ARMS)
def test_exact_optimizer_membership_and_frozen_bytes_after_fabricated_update(arm):
    item = model(arm)
    before_rng = torch.random.get_rng_state().clone()
    optimizer = training.construct_optimizer(item)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    expected = item.effective_named_parameters()
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for _, p in expected]
    assert not optimizer.state
    before = state_bits(item)
    data = dataset()
    recorded = training.batch_update(item, optimizer, data, list(range(6)))
    after = state_bits(item)
    names = {name for name, _ in expected}
    assert recorded["effective_parameter_count"] == models.TRAINABLE_COUNTS[arm]
    assert set(recorded["effective_parameter_names"]) == names
    assert recorded["optimizer_updates"] == recorded["optimizer_step"] == 1
    assert recorded["forward_chunks"] == 3 and recorded["forward_rows"] == sum((65, 33, 9, 5, 2, 1))
    assert recorded["backward_chunks"] == recorded["loss_chunks"]
    assert any(before[name] != after[name] for name in names)
    for name, parameter in item.named_parameters():
        if name not in names:
            assert before[name] == after[name]
            assert parameter.grad is None and parameter not in optimizer.state
    assert {id(p) for p in optimizer.state} == {id(p) for _, p in expected}
    assert item.parameter_metadata()["requires_grad_count"] == item.parameter_metadata()["effective_count"]


@pytest.mark.parametrize("arm", models.ARMS)
def test_update_matches_separate_whole_episode_accumulation(arm):
    item = model(arm)
    other = models.from_state(arm, item.seed, item.slow.state_dict())
    data = dataset()
    optimizer = training.construct_optimizer(item)
    reference = torch.optim.Adam([p for _, p in other.effective_named_parameters()], lr=.003, weight_decay=0.)
    actual = training.batch_update(item, optimizer, data, [3, 0, 5])
    expected, norm = manual_update(other, reference, data, [3, 0, 5])
    assert state_bits(item) == state_bits(other)
    assert actual["loss"] == expected["total"] and actual["nonquery_loss"] == expected["nonquery"]
    assert actual["prior_loss"] == expected["prior"] and actual["gradient_norm_before_clip"] == norm
    for (_, left), (_, right) in zip(item.effective_named_parameters(), other.effective_named_parameters(), strict=True):
        assert {k: bits(v) for k, v in optimizer.state[left].items()} == {k: bits(v) for k, v in reference.state[right].items()}


def test_full_joint_one_update_is_bitwise_qualified_original_including_optimizer():
    item = model("full_joint")
    original = original_models.from_states(memory.Config("none", key_dim=8), item.seed, 4,
                                            item.slow.state_dict(), None, slow_mode="joint")
    data = dataset()
    optimizer, old_optimizer = training.construct_optimizer(item), original_training.construct_optimizer(original)
    actual = training.batch_update(item, optimizer, data, list(range(6)))
    expected = original_training.batch_update(original, old_optimizer, data, list(range(6)))
    assert state_bits(item) == state_bits(original)
    assert item.parameter_metadata() == original.parameter_metadata()
    assert {k: v for k, v in actual.items() if k not in {"version", "timing_seconds"}} == {
        k: v for k, v in expected.items() if k not in {"version", "timing_seconds"}}
    for (_, left), (_, right) in zip(item.named_parameters(), original.named_parameters(), strict=True):
        assert {k: bits(v) for k, v in optimizer.state[left].items()} == {k: bits(v) for k, v in old_optimizer.state[right].items()}


def test_both_readouts_gradient_flows_through_frozen_gru_feedback():
    item = model("both_readouts")
    packet = data_module.batch_chunk(dataset((9,)), [0], 0)
    inputs = {name: torch.from_numpy(value.copy()) for name, value in packet["model_inputs"].items()}
    forecast = item.slow(**inputs)
    # At step5, the hidden state has passed the later query at4. Its dependence
    # on output parameters is exclusively through the innovation-fed GRU.
    objective = forecast.keys_hidden[:, 5].square().sum()
    gradients = torch.autograd.grad(objective, (item.slow.output.weight, item.slow.output.bias))
    assert all(torch.isfinite(g).all() and (g != 0).any() for g in gradients)
    assert all(not p.requires_grad and p.grad is None for p in item.slow.recurrent.parameters())


def test_residual_only_actual_update_preserves_recurrent_trajectory():
    item = model("action_residual_only")
    data = dataset((9,))
    packet = data_module.batch_chunk(data, [0], 0)
    inputs = {name: torch.from_numpy(value.copy()) for name, value in packet["model_inputs"].items()}
    before = item.slow(**inputs)
    training.batch_update(item, training.construct_optimizer(item), data, [0])
    after = item.slow(**inputs)
    assert bits(before.keys_hidden) == bits(after.keys_hidden)
    assert bits(before.carry.base.hidden) == bits(after.carry.base.hidden)
    assert bits(before.prediction) == bits(after.prediction)
    assert bits(before.action_prediction) != bits(after.action_prediction)


@pytest.mark.parametrize("arm", models.ARMS)
def test_no_supported_rows_zero_complete_effective_gradients_only(arm):
    item = model(arm)
    before = state_bits(item)
    optimizer = training.construct_optimizer(item)
    result = training.batch_update(item, optimizer, dataset((1,)), [0])
    assert result["loss"] == 0 and result["backward_chunks"] == 0
    assert result["zero_filled_gradient_names"] == [name for name, _ in item.effective_named_parameters()]
    assert state_bits(item) == before and result["optimizer_step"] == 1


@pytest.mark.parametrize("defect", ("all_parameters", "reverse_order", "learning_rate", "gradient_lock", "frozen_gradient"))
def test_invalid_optimizer_or_mask_rejected_before_forward(defect):
    item = model("both_readouts")
    optimizer = training.construct_optimizer(item)
    if defect == "all_parameters":
        optimizer = torch.optim.Adam(item.parameters(), lr=.003)
    elif defect == "reverse_order":
        optimizer.param_groups[0]["params"].reverse()
    elif defect == "learning_rate":
        optimizer.param_groups[0]["lr"] = .004
    elif defect == "gradient_lock":
        item.slow.recurrent.weight_ih_l0.requires_grad_(True)
    else:
        item.slow.recurrent.weight_ih_l0.grad = torch.zeros_like(item.slow.recurrent.weight_ih_l0)
    before = state_bits(item)
    calls = []
    hook = item.register_forward_pre_hook(lambda *_a: calls.append(True))
    with pytest.raises(ValueError):
        training.batch_update(item, optimizer, dataset((9,)), [0])
    hook.remove()
    assert not calls and not optimizer.state and state_bits(item) == before


def test_parameters_fixed_and_carry_detached_until_single_final_update():
    item = model("both_readouts")
    optimizer = training.construct_optimizer(item)
    initial = state_bits(item)
    starts = []

    def inspect_chunk(_module, _args, kwargs):
        assert state_bits(item) == initial and not optimizer.state
        assert kwargs["carry"].slow.base.hidden.grad_fn is None
        assert kwargs["carry"].fast.matrix.grad_fn is None
        starts.append(kwargs["carry"].slow.base.absolute_step.tolist())

    hook = item.register_forward_pre_hook(inspect_chunk, with_kwargs=True)
    result = training.batch_update(item, optimizer, dataset((65, 9, 1)), [0, 1, 2])
    hook.remove()
    assert starts == [[0, 0, 0], [32, 9, 1], [64, 9, 1]]
    assert result["optimizer_updates"] == 1


def test_guard_failure_never_updates_or_retries():
    item = model("full_joint")
    optimizer = training.construct_optimizer(item)
    initial = state_bits(item)
    stages = []

    def abort(name, start):
        stages.append((name, start))
        if name == "chunk_materialize" and start == 32:
            raise TimeoutError("fabricated deadline")

    with pytest.raises(TimeoutError, match="fabricated deadline"):
        training.batch_update(item, optimizer, dataset((65,)), [0], stage=abort)
    assert state_bits(item) == initial and not optimizer.state
    assert ("optimizer_update", None) not in stages
    assert stages.count(("chunk_materialize", 32)) == 1


def test_dev_data_rejected_before_any_forward():
    item = model("full_joint")
    optimizer = training.construct_optimizer(item)
    with pytest.raises(ValueError, match="authenticated TRAIN"):
        training.batch_update(item, optimizer, dataset((9,), stage="dev"), [0])
    assert not optimizer.state
