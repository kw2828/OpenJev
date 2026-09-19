"""Tiny engineering410 loss/gradient/Adam comparisons, without speed tests."""

import pytest
import torch

from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_streaming_sequence_loss import (
    streaming_sequence_loss,
    streaming_sequence_work,
)
from openjev.research.reacher_two_observation_history import TwoObservationHistoryGRUWorldModel
from openjev.research.reacher_world_models import sequence_loss


@pytest.fixture(autouse=True)
def engineering_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            yield
    finally:
        torch.set_num_threads(threads)


def data(steps=14, missing=((), tuple(range(3, 9)), tuple(range(3, 13))), dtype=torch.float32):
    batch, points = len(missing), steps + 1
    angles = torch.linspace(-0.4, 0.8, batch * points * 2, dtype=dtype).reshape(batch, points, 2)
    packets = torch.zeros(batch, points, 8, dtype=dtype)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.tensor([0.1, -0.09], dtype=dtype)
    for case, absent in enumerate(missing):
        last = 0
        for point in range(points):
            if point in absent:
                packets[case, point, :4] = 0
                packets[case, point, 7] = (point - last) * 0.02
            else:
                packets[case, point, 6] = 1
                last = point
    actions = torch.linspace(-0.7, 0.8, batch * steps * 2, dtype=dtype).reshape(batch, steps, 2)
    rewards = -0.2 - actions.square().sum(-1)
    return packets, actions, rewards


def models(dtype=torch.float32, residual=True):
    parent = GRUResidualRewardWorldModel(hidden_size=4, residual_reward=residual).to(dtype=dtype)
    original = TwoObservationHistoryGRUWorldModel(hidden_size=4, residual_reward=residual).to(dtype=dtype)
    original.load_state_dict(parent.state_dict())
    return parent, original


def assert_metrics(left, right, *, atol=0, rtol=0):
    assert set(left) == set(right)
    for key in left:
        assert left[key] == pytest.approx(right[key], abs=atol, rel=rtol), key


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("kwargs", [{}, {"rollout_horizon": 1}, {"rollout_horizon": 20},
    {"rollout_weight": 0}, {"rollout_horizon": 3, "rollout_weight": 0.7,
                           "reward_scale": 2.3, "kl_weight": 0, "kl_balance": 0.2, "free_nats": 0}])
def test_original_loss_terms_reductions_and_windows_match(dtype, residual, kwargs):
    parent, original = models(dtype, residual)
    values = data(dtype=dtype)
    loss, metrics = streaming_sequence_loss(parent, *values, **kwargs)
    expected, old_metrics = sequence_loss(original, *values, **kwargs)
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    assert_metrics(metrics, old_metrics)


@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float32, 2e-5, 3e-6), (torch.float64, 1e-12, 1e-12)])
# The unchanged reference converts validity-count tensors directly to floats.
@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True to a scalar:UserWarning")
def test_named_parameter_and_public_input_gradients_match(dtype, rtol, atol):
    parent, original = models(dtype)
    left = [value.clone().requires_grad_() for value in data(dtype=dtype)]
    right = [value.detach().clone().requires_grad_() for value in left]
    streaming_sequence_loss(parent, *left)[0].backward()
    sequence_loss(original, *right)[0].backward()
    maximum = 0.
    for (name, actual), (other, expected) in zip(parent.named_parameters(), original.named_parameters(), strict=True):
        assert name == other and actual.grad is not None and expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)
        maximum = max(maximum, float((actual.grad - expected.grad).abs().max()))
    differences = []
    for actual, expected in zip(left, right, strict=True):
        assert actual.grad is not None and expected.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)
        differences.append(float((actual.grad - expected.grad).abs().max()))
    assert not left[0].grad[..., :4][left[0][..., 6] == 0].any()
    print(f"{dtype}: loss-gradient max parameter={maximum:.9g}; "
          f"packet={differences[0]:.9g}; command={differences[1]:.9g}; reward={differences[2]:.9g}")


@pytest.mark.parametrize("reward_scale", [0., 4.])
def test_three_paired_adam_steps_preserve_approximate_updates_and_moments(reward_scale):
    parent, original = models()
    # Match the original Adam recipe explicitly; no trainer/registry is used.
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0, amsgrad=False, foreach=False, maximize=False,
        capturable=False, differentiable=False, fused=False, decoupled_weight_decay=False)
        for model in (parent, original)]
    batches = [data(steps=5, missing=((), (2, 3))), data(steps=6, missing=((), (2, 3, 4))),
               data(steps=4, missing=((), (2,)))]
    max_parameter = max_first_moment = max_second_moment = 0.
    for update, values in enumerate(batches, 1):
        metrics, norms = [], []
        for model, optimizer, objective in zip((parent, original), optimizers,
                (streaming_sequence_loss, sequence_loss), strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss, row = objective(model, *values, rollout_horizon=3, reward_scale=reward_scale)
            loss.backward()
            assert all(value.grad is not None for value in model.parameters())
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True, foreach=False))
            optimizer.step()
            metrics.append(row)
        assert_metrics(metrics[0], metrics[1], atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(norms[0], norms[1], atol=3e-6, rtol=2e-5)
        for (name, actual), (other, expected) in zip(parent.named_parameters(), original.named_parameters(), strict=True):
            assert name == other
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            max_parameter = max(max_parameter, float((actual.detach() - expected.detach()).abs().max()))
            one, two = optimizers[0].state[actual], optimizers[1].state[expected]
            assert set(one) == set(two) == {"step", "exp_avg", "exp_avg_sq"}
            assert one["step"] == two["step"] == update
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(one[key], two[key], atol=3e-7, rtol=2e-5)
            max_first_moment = max(max_first_moment, float((one["exp_avg"] - two["exp_avg"]).abs().max()))
            max_second_moment = max(max_second_moment, float((one["exp_avg_sq"] - two["exp_avg_sq"]).abs().max()))
    print(f"three Adam steps reward_scale={reward_scale:g}: max parameter={max_parameter:.9g}; "
          f"first_moment={max_first_moment:.9g}; second_moment={max_second_moment:.9g}")


@pytest.mark.parametrize("steps,horizon,weight,residual", [(1, 1, .5, True), (4, 3, .5, True),
    (4, 3, 0., True), (4, 8, .5, True), (4, 3, .5, False)])
def test_hook_observed_work_includes_all_parent_heads(steps, horizon, weight, residual):
    parent, _ = models(residual=residual)
    samples = dict.fromkeys(("observation_update", "transition", "observation_head", "reward_head"), 0)
    calls = samples.copy()
    handles = []
    for name in samples:
        def count(_module, args, _output, name=name):
            calls[name] += 1
            samples[name] += len(args[0])
        handles.append(getattr(parent, name).register_forward_hook(count))
    try:
        streaming_sequence_loss(parent, *data(steps, missing=((), ())),
                                rollout_horizon=horizon, rollout_weight=weight)
    finally:
        for handle in handles:
            handle.remove()
    work = streaming_sequence_work(2, steps, rollout_horizon=horizon, rollout_weight=weight,
                                   residual_reward=residual)
    assert calls["observation_update"] == work["parent_assimilate_calls"]
    assert calls["transition"] == calls["observation_head"] == calls["reward_head"] == work["parent_advance_calls"]
    assert {name: work[name + "_samples"] for name in samples} == samples
    assert work["expected_action_cost_samples"] == (samples["transition"] if residual else 0)
    assert work["terminal_assimilations"] == 0


def test_full50_accounting_is_arithmetic_not_a_speed_measurement():
    work = streaming_sequence_work(32, 50)
    assert work["observation_update_samples"] == 4736
    assert work["transition_samples"] == work["observation_head_samples"] == work["reward_head_samples"] == 12096
    assert work["root_advance_calls"] == 98 and work["one_step_advance_calls"] == 50
    assert work["rollout_samples"] == 32 * 46 * 5


def test_missing_targets_are_zero_gradient_and_rewards_still_supervised():
    parent, original = models()
    packets, actions, rewards = data(steps=5, missing=(tuple(range(1, 6)),))
    packets.requires_grad_()
    loss, metrics = streaming_sequence_loss(parent, packets, actions, rewards, rollout_horizon=3)
    other, expected = sequence_loss(original, packets.detach(), actions, rewards, rollout_horizon=3)
    torch.testing.assert_close(loss, other, atol=0, rtol=0)
    assert_metrics(metrics, expected)
    assert metrics["observation_mse"] == metrics["rollout_observation_mse"] == 0
    assert metrics["reward_mse"] > 0 and metrics["rollout_reward_mse"] > 0
    loss.backward()
    assert not packets.grad[:, 1:, :4].any()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in parent.parameters())


def test_missing_root_rewards_are_excluded_only_from_rollout():
    parent, _ = models()
    values = list(data(steps=4, missing=((1, 2, 3, 4),)))
    _, first = streaming_sequence_loss(parent, *values, rollout_horizon=1)
    values[2] = values[2].clone()
    values[2][:, 1:] -= 2
    _, second = streaming_sequence_loss(parent, *values, rollout_horizon=1)
    assert first["rollout_reward_mse"] == second["rollout_reward_mse"]
    assert first["reward_mse"] != second["reward_mse"]


def test_terminal_target_is_not_assimilated_or_used_by_prediction():
    parent, original = models()
    packets, actions, rewards = data(steps=50, missing=((),))
    # An extra terminal assimilation would reject this changed goal/age. The
    # original loss only requires a finite valid target and never sees it as a root.
    changed = packets.clone()
    changed[:, -1, :4] *= -1
    changed[:, -1, 4:6] += 0.3
    changed[:, -1, 7] = 0.123
    _, baseline = streaming_sequence_loss(parent, packets, actions, rewards)
    loss, metrics = streaming_sequence_loss(parent, changed, actions, rewards)
    other, expected = sequence_loss(original, changed, actions, rewards)
    torch.testing.assert_close(loss, other, atol=0, rtol=0)
    assert_metrics(metrics, expected)
    assert metrics["reward_mse"] == baseline["reward_mse"]
    assert metrics["rollout_reward_mse"] == baseline["rollout_reward_mse"]
    assert metrics["observation_mse"] != baseline["observation_mse"]


def test_no_input_weight_rng_mutation_or_cross_call_cache():
    parent, _ = models()
    values = data(steps=4, missing=((),))
    inputs = [value.clone() for value in values]
    weights = {name: value.clone() for name, value in parent.state_dict().items()}
    rng = torch.get_rng_state().clone()
    one, metrics = streaming_sequence_loss(parent, *values)
    two, other = streaming_sequence_loss(parent, *values)
    torch.testing.assert_close(one, two, atol=0, rtol=0)
    assert metrics == other and torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(a, b) for a, b in zip(inputs, values, strict=True))
    assert all(torch.equal(value, parent.state_dict()[name]) for name, value in weights.items())


@pytest.mark.parametrize("defect", ["nan_reward", "masked_nan", "final_action_bound", "reward_shape",
    "terminal_overflow", "prefix_overflow", "actual_class", "bad_horizon", "negative_weight", "bad_kl"])
def test_invalid_inputs_fail_before_any_neural_work(defect):
    parent, other = models()
    values = list(data(steps=4, missing=((2,),)))
    kwargs = {}
    if defect == "nan_reward": values[2][0, 0] = float("nan")
    elif defect == "masked_nan": values[0][0, 2, 0] = float("nan")
    elif defect == "final_action_bound": values[1][0, -1, 0] = 1.01
    elif defect == "reward_shape": values[2] = values[2][:, :-1]
    elif defect == "terminal_overflow": values = data(steps=51, missing=((),))
    elif defect == "prefix_overflow": values = data(steps=14, missing=(tuple(range(1, 13)),))
    elif defect == "actual_class": parent = other
    elif defect == "bad_horizon": kwargs["rollout_horizon"] = 0
    elif defect == "negative_weight": kwargs["rollout_weight"] = -1
    else: kwargs["kl_balance"] = 2
    def forbidden(*_args):
        pytest.fail("Invalid input reached a neural layer")
    handle = parent.observation_update.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(ValueError):
            streaming_sequence_loss(parent, *values, **kwargs)
    finally:
        handle.remove()
