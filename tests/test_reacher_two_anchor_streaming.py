"""Tiny synthetic parity checks; no simulator, fitting or scientific inputs."""

import pytest
import torch

from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_two_anchor_streaming import stream_roots
from openjev.research.reacher_two_observation_history import TwoObservationHistoryGRUWorldModel
from openjev.research.reacher_world_models import repeat_index


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


def data(points=15, missing=((), tuple(range(3, 9)), tuple(range(3, 13))), dtype=torch.float32):
    batch = len(missing)
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
    actions = torch.linspace(-0.7, 0.8, batch * (points - 1) * 2, dtype=dtype).reshape(batch, points - 1, 2)
    return packets, actions


def models(dtype=torch.float32, residual=True):
    parent = GRUResidualRewardWorldModel(hidden_size=4, residual_reward=residual).to(dtype=dtype)
    rebuilt = TwoObservationHistoryGRUWorldModel(hidden_size=4, residual_reward=residual).to(dtype=dtype)
    rebuilt.load_state_dict(parent.state_dict())
    return parent, rebuilt


def rebuild(model, packets, actions):
    state = model.initial(len(packets))
    roots = []
    for step in range(packets.shape[1]):
        if step:
            state, _, _ = model.advance(state, actions[:, step - 1])
        state = model.assimilate(state, packets[:, step])
        roots.append(state)
    return {key: torch.stack([root[key] for root in roots], 1) for key in ("hidden", "packet")}


def exact(left, right):
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("residual", [False, True])
def test_every_root_and_three_private_predictions_match_reconstruction(dtype, residual):
    parent, reference = models(dtype, residual)
    packets, actions = data(dtype=dtype)
    result = stream_roots(parent, packets, actions)
    expected = rebuild(reference, packets, actions)
    exact(result.roots, expected)
    # Include missing roots and thirteen-slot scratch reacquisition in the same
    # fixed-size batch. No parent state is fed back into either real sequence.
    actual = {key: value[:, :-3].flatten(0, 1) for key, value in result.roots.items()}
    wanted = {key: value[:, :-3].flatten(0, 1) for key, value in expected.items()}
    for point in range(3):
        command = torch.full((len(actual["packet"]), 2), 0.13 * point, dtype=dtype)
        actual, angles, reward = parent.advance(actual, command)
        wanted, expected_angles, expected_reward = GRUResidualRewardWorldModel.advance(reference, wanted, command)
        exact(actual, wanted)
        torch.testing.assert_close(angles, expected_angles, atol=0, rtol=0)
        torch.testing.assert_close(reward, expected_reward, atol=0, rtol=0)


def prediction_loss(model, roots, actions):
    real = {key: value[:, :-1].flatten(0, 1) for key, value in roots.items()}
    _, angles, reward = GRUResidualRewardWorldModel.advance(model, real, actions.flatten(0, 1))
    return (roots["hidden"].square().sum() * 0.03 + roots["packet"].square().sum() * 0.001
            + angles.square().sum() * 0.07 + reward.square().sum() * 0.02)


@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float32, 2e-5, 3e-6), (torch.float64, 1e-12, 1e-12)])
def test_parameter_and_public_input_gradients_match_including_both_heads(dtype, rtol, atol):
    parent, reference = models(dtype)
    packets, actions = data(dtype=dtype)
    p1, a1 = packets.clone().requires_grad_(), actions.clone().requires_grad_()
    p2, a2 = packets.clone().requires_grad_(), actions.clone().requires_grad_()
    result = stream_roots(parent, p1, a1)
    expected = rebuild(reference, p2, a2)
    loss, other = prediction_loss(parent, result.roots, a1), prediction_loss(reference, expected, a2)
    torch.testing.assert_close(loss, other, rtol=0, atol=0)
    loss.backward()
    other.backward()
    maxima = []
    for (name, value), (other_name, other_value) in zip(parent.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other_name and value.grad is not None and other_value.grad is not None
        torch.testing.assert_close(value.grad, other_value.grad, rtol=rtol, atol=atol)
        maxima.append((value.grad - other_value.grad).abs().max().item())
    for actual, wanted in ((p1.grad, p2.grad), (a1.grad, a2.grad)):
        torch.testing.assert_close(actual, wanted, rtol=rtol, atol=atol)
        assert torch.isfinite(actual).all()
    assert not p1.grad[..., :4][packets[..., 6] == 0].any()
    print(f"{dtype}: max parameter-gradient difference={max(maxima):.9g}; "
          f"packet={float((p1.grad-p2.grad).abs().max()):.9g}; "
          f"command={float((a1.grad-a2.grad).abs().max()):.9g}")


def test_root_only_loss_none_and_explicit_zero_head_gradients_are_mathematically_zero():
    parent, reference = models()
    packets, actions = data()
    stream_roots(parent, packets, actions).roots["hidden"].sum().backward()
    rebuild(reference, packets, actions)["hidden"].sum().backward()
    for (name, actual), (_, wanted) in zip(parent.named_parameters(), reference.named_parameters(), strict=True):
        if "head" in name:
            # Padded reconstruction can retain zero-valued graph edges through
            # unused angle predictions. Do not claim None == zero to an Adam
            # optimizer with existing moments; compare derivatives only.
            assert actual.grad is None
            assert wanted.grad is None or not wanted.grad.any()
        else:
            torch.testing.assert_close(actual.grad, wanted.grad, atol=3e-6, rtol=2e-5)


def test_anchor_indices_at_start_gap_full_window_reacquisition_and_next_measurement():
    parent, _ = models()
    packets, actions = data()
    result = stream_roots(parent, packets, actions)
    assert result.older_anchor.dtype == result.newest_anchor.dtype == torch.int64
    for case in range(3):
        visible = []
        for point in range(15):
            if packets[case, point, 6] == 1:
                visible.append(point)
            assert result.newest_anchor[case, point] == visible[-1]
            assert result.older_anchor[case, point] == (visible[-2] if len(visible) > 1 else 0)
    assert result.older_anchor[2, 12:15].tolist() == [1, 2, 13]


def test_discarded_past_has_no_final_output_or_gradient_influence():
    parent, _ = models()
    packets, actions = data(points=15, missing=(tuple(range(9, 15)),))
    changed, altered = packets.clone(), actions.clone()
    # Last two measurements are 7 and 8; discard everything before packet7.
    changed[:, :7, :4] *= -1
    altered[:, :7] *= -1
    left, right = stream_roots(parent, packets, actions), stream_roots(parent, changed, altered)
    exact({k: v[:, -1] for k, v in left.roots.items()}, {k: v[:, -1] for k, v in right.roots.items()})
    packets.requires_grad_()
    actions.requires_grad_()
    stream_roots(parent, packets, actions).roots["hidden"][:, -1].sum().backward()
    assert not packets.grad[:, :7].any() and not actions.grad[:, :7].any()
    assert packets.grad[:, 7:9, :4].abs().sum() > 0
    assert actions.grad[:, 7:].abs().sum() > 0


def test_prefix_roots_do_not_depend_on_future_packets_or_commands():
    parent, _ = models()
    packets, actions = data()
    whole = stream_roots(parent, packets, actions)
    prefix = stream_roots(parent, packets[:, :8], actions[:, :7])
    exact(prefix.roots, {k: v[:, :8] for k, v in whole.roots.items()})
    changed, altered = packets.clone(), actions.clone()
    changed[:, 8:, :4] *= -1
    altered[:, 7:] *= -1
    other = stream_roots(parent, changed, altered)
    exact(prefix.roots, {k: v[:, :8] for k, v in other.roots.items()})
    packets.requires_grad_()
    actions.requires_grad_()
    stream_roots(parent, packets, actions).roots["hidden"][:, 7].sum().backward()
    assert not packets.grad[:, 8:].any() and not actions.grad[:, 7:].any()


def test_missing_nonfinite_placeholders_are_sanitized_without_mutation():
    parent, _ = models()
    packets, actions = data()
    poison = packets.clone()
    poison[..., :4][packets[..., 6] == 0] = torch.tensor([float("nan"), float("inf"), -1e9, 1e9])
    original = poison.clone()
    exact(stream_roots(parent, poison, actions).roots, stream_roots(parent, packets, actions).roots)
    torch.testing.assert_close(poison, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("residual", [False, True])
def test_fixed_completed_work_counts_include_unused_parent_readouts(residual):
    parent, _ = models(residual=residual)
    packets, actions = data(points=5, missing=((), (2, 3)))
    samples = dict.fromkeys(("observation_update", "transition", "observation_head", "reward_head"), 0)
    handles = []
    for name in samples:
        def count(_module, inputs, _output, name=name):
            samples[name] += len(inputs[0])
        handles.append(getattr(parent, name).register_forward_hook(count))
    try:
        result = stream_roots(parent, packets, actions)
    finally:
        for handle in handles:
            handle.remove()
    assert result.work["parent_assimilate_calls"] == 13
    assert result.work["parent_advance_calls"] == 8
    for name, count in samples.items():
        assert count == result.work[name + "_samples"]
    assert result.work["expected_action_cost_samples"] == (16 if residual else 0)


def test_calls_are_independent_rng_weights_and_private_candidate_roots():
    parent, _ = models()
    packets, actions = data()
    weights = {k: v.clone() for k, v in parent.state_dict().items()}
    rng = torch.get_rng_state().clone()
    pcopy, acopy = packets.clone(), actions.clone()
    result = stream_roots(parent, packets, actions)
    root = {k: v[:, 10].clone() for k, v in result.roots.items()}
    snapshot = {k: v.clone() for k, v in root.items()}
    candidate = repeat_index(root, torch.tensor([2, 0, 0, 1]))
    for _ in range(3):
        candidate, _, _ = parent.advance(candidate, torch.full((4, 2), 0.2))
    exact(root, snapshot)
    exact(parent.state_dict(), weights)
    exact(stream_roots(parent, packets, actions).roots, result.roots)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(packets, pcopy) and torch.equal(actions, acopy)
    with torch.no_grad():
        parent.observation_update.bias_ih.add_(0.1)
    changed = stream_roots(parent, packets, actions)
    assert not torch.equal(changed.roots["hidden"], result.roots["hidden"])


@pytest.mark.parametrize("points", [1, 51])
def test_startup_and_terminal_observation_are_supported(points):
    parent, reference = models()
    packets, actions = data(points=points, missing=((),))
    actual = stream_roots(parent, packets, actions)
    exact(actual.roots, rebuild(reference, packets, actions))
    assert actual.roots["hidden"].shape == (1, points, 4)
    assert actual.work["parent_assimilate_calls"] == 1 + 3 * (points - 1)
    assert actual.work["parent_advance_calls"] == 2 * (points - 1)


@pytest.mark.parametrize("missing", [tuple(range(1, 13)), tuple(range(2, 13))])
def test_historical_overflow_is_rejected_even_if_final_suffix_would_fit(missing):
    parent, _ = models()
    packets, actions = data(points=15, missing=(missing,))
    with pytest.raises(ValueError, match="exceeds twelve"):
        stream_roots(parent, packets, actions)


@pytest.mark.parametrize("defect", ["shape", "empty", "terminal", "dtype", "commands", "command_dtype",
                                  "command_nan", "command_bound", "known_nan", "visible_nan",
                                  "first_missing", "validity", "visible_age", "missing_age", "target",
                                  "model_class", "weight_nan", "dt", "noise", "residual"])
def test_invalid_inputs_fail_without_weight_or_rng_mutation(defect):
    parent, reference = models()
    packets, actions = data(points=5, missing=((2, 3),))
    if defect == "shape":
        packets = packets[..., :7]
    elif defect == "empty":
        packets, actions = packets[:0], actions[:0]
    elif defect == "terminal":
        packets, actions = data(points=52, missing=((),))
    elif defect == "dtype":
        packets = packets.double()
    elif defect == "commands":
        actions = actions[:, :-1]
    elif defect == "command_dtype":
        actions = actions.double()
    elif defect == "command_nan":
        actions[0, 0, 0] = float("nan")
    elif defect == "command_bound":
        actions[0, 0, 0] = 1.001
    elif defect in {"known_nan", "visible_nan"}:
        packets[0, 0, 4 if defect == "known_nan" else 0] = float("nan")
    elif defect == "first_missing":
        packets[0, 0, 6] = 0
    elif defect == "validity":
        packets[0, 2, 6] = 0.5
    elif defect == "visible_age":
        packets[0, 0, 7] = 1e-8
    elif defect == "missing_age":
        packets[0, 3, 7] += 0.002
    elif defect == "target":
        packets[0, 1, 4] += 0.1
    elif defect == "model_class":
        parent = reference
    elif defect == "weight_nan":
        with torch.no_grad():
            next(parent.parameters()).flatten()[0] = float("nan")
    else:
        setattr(parent, {"dt": "dt", "noise": "noise_std", "residual": "residual_reward"}[defect],
                {"dt": 0, "noise": -0.1, "residual": 1}[defect])
    weights, rng = {k: v.clone() for k, v in parent.state_dict().items()}, torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        stream_roots(parent, packets, actions)
    for key, value in parent.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(rng, torch.get_rng_state())


def test_age_tolerance_keeps_actual_rounded_public_age():
    parent, reference = models()
    packets, actions = data()
    packets[..., 7][packets[..., 6] == 0] += 1e-7
    actual = stream_roots(parent, packets, actions)
    exact(actual.roots, rebuild(reference, packets, actions))
    torch.testing.assert_close(actual.roots["packet"][..., 7], packets[..., 7], atol=0, rtol=0)
