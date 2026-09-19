"""Tiny engineering-only cell checks; no training, native environment or data."""

import pytest
import torch

from openjev.research.reacher_innovation_context import (
    STATE_KEYS,
    VARIANTS,
    InnovationContextWorldModel,
    capture_module_work,
)
from openjev.research.reacher_world_models import repeat_index


@pytest.fixture(autouse=True)
def isolated_engineering():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(410)
            yield
    finally:
        torch.set_num_threads(threads)


def model(variant="normalized", **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        return InnovationContextWorldModel(variant, hidden_size=4, context_size=4, **kwargs)


def packet(batch=2, *, visible=True, age=0.0):
    value = torch.zeros(batch, 8)
    angles = torch.linspace(-0.3, 0.7, batch * 2).reshape(batch, 2)
    value[:, :2], value[:, 2:4] = angles.cos(), angles.sin()
    value[:, 4:6] = torch.tensor([0.1, -0.09])
    value[:, 6] = int(visible)
    value[:, 7] = age
    if not visible:
        value[:, :4] = 0
    return value


def clone(state):
    return {name: value.clone() for name, value in state.items()}


def equal(left, right):
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


def prior(m, batch=2):
    root = m.assimilate(m.initial(batch), packet(batch))
    return m.advance(root, torch.full((batch, 2), 0.2))[0]


def fixed_gate(m):
    with torch.no_grad():
        m.gate.weight.copy_(torch.tensor([[0.7, 0.9]]))
        m.gate.bias.fill_(-0.3)


def test_same_parameter_names_shapes_initialization_and_state_for_all_variants():
    models = [model(variant) for variant in VARIANTS]
    baseline = models[0]
    for actual in models:
        equal(actual.state_dict(), baseline.state_dict())
        assert list(actual.state_dict()) == list(baseline.state_dict())
        assert sum(p.numel() for p in actual.parameters()) == 452
        assert not list(actual.named_buffers())
        equal(actual.initial(2), baseline.initial(2))
        assert set(actual.initial(2)) == STATE_KEYS
        config = actual.configuration()
        assert config["variant"] == actual.variant
        assert config["provisional_constructor_choices"]
        assert "uncalibrated" in config["variance_semantics"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_startup_has_no_previous_measurement_innovation_or_slow_correction(variant):
    m = model(variant)
    initial = m.initial(2)
    actual, diagnostic = m.assimilate_with_diagnostics(initial, packet())
    assert diagnostic["startup"].all() and not diagnostic["innovation_valid"].any()
    for key in ("innovation", "raw_squared_error", "normalized_squared_error", "elapsed_seconds", "effective_gate", "context_delta"):
        assert not diagnostic[key].any()
    assert not actual["context"].any()
    assert actual["hidden"].abs().sum() > 0
    assert (actual["real_index"] == 0).all() and (actual["last_valid_index"] == 0).all()
    equal(initial, m.initial(2))


def test_prior_is_predicted_before_returned_packet_and_never_recomputed_during_assimilation():
    m = model()
    with capture_module_work(m) as counts:
        prediction = prior(m)
        original = clone(prediction)
        first, second = packet(), packet()
        first[:, :4] = prediction["prior_mean"] + 0.1
        second[:, :4] = prediction["prior_mean"] + 0.8
        left, ld = m.assimilate_with_diagnostics(prediction, first)
        right, rd = m.assimilate_with_diagnostics(prediction, second)
    assert counts["observation_head"] == counts["variance_head"] == {"calls": 1, "samples": 2}
    equal(prediction, original)
    for key in ("prior_mean", "prior_variance"):
        torch.testing.assert_close(ld[key], original[key], rtol=0, atol=0)
        torch.testing.assert_close(rd[key], original[key], rtol=0, atol=0)
    assert not torch.equal(left["hidden"], right["hidden"])
    assert not torch.equal(ld["normalized_squared_error"], rd["normalized_squared_error"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_only_declared_gate_information_pathways_change(variant):
    m = model(variant)
    fixed_gate(m)
    prediction = prior(m)
    low, high = packet(), packet()
    low[:, :4] = prediction["prior_mean"] + 0.1
    high[:, :4] = prediction["prior_mean"] + 0.5
    _, small = m.assimilate_with_diagnostics(prediction, low)
    _, large = m.assimilate_with_diagnostics(prediction, high)
    higher_variance = clone(prediction)
    higher_variance["prior_variance"] = prediction["prior_variance"] * 1.2
    _, uncertain = m.assimilate_with_diagnostics(higher_variance, high)
    # The gate's error feature is masked, not the Ue correction itself.
    if variant in {"constant", "age"}:
        torch.testing.assert_close(small["gate"], large["gate"], rtol=0, atol=0)
    else:
        assert (large["gate"] > small["gate"]).all()
    if variant == "normalized":
        assert (uncertain["gate"] < large["gate"]).all()
    else:
        torch.testing.assert_close(uncertain["gate"], large["gate"], rtol=0, atol=0)
    raw = large["innovation"].square().sum(-1, keepdim=True)
    normalized = (large["innovation"].square() / prediction["prior_variance"]).sum(-1, keepdim=True)
    torch.testing.assert_close(large["raw_squared_error"], raw, rtol=0, atol=0)
    torch.testing.assert_close(large["normalized_squared_error"], normalized, rtol=0, atol=0)
    expected = torch.cat((large["elapsed_seconds"], torch.log1p(normalized if variant == "normalized" else raw)), -1)
    expected *= torch.tensor([variant != "constant", variant in {"raw", "normalized"}])
    torch.testing.assert_close(large["gate_features"], expected, rtol=0, atol=0)
    assert not large["gate_features"].requires_grad


@pytest.mark.parametrize("variant", VARIANTS)
def test_mixed_six_and_ten_gaps_hold_context_and_use_actual_elapsed_return_time(variant):
    m = model(variant)
    state = m.assimilate(m.initial(2), packet())
    last = [0, 0]
    for step in range(1, 14):
        context = state["context"].clone()
        advanced, _, _ = m.advance(state, torch.full((2, 2), 0.3))
        torch.testing.assert_close(advanced["context"], context, rtol=0, atol=0)
        assert not torch.equal(advanced["hidden"], state["hidden"])
        incoming = packet()
        missing = torch.tensor([3 <= step <= 8, 3 <= step <= 12])
        for case in range(2):
            if missing[case]:
                incoming[case, :4] = float("nan")
                incoming[case, 6] = 0
                incoming[case, 7] = (step - last[case]) * 0.02
        state, diagnostic = m.assimilate_with_diagnostics(advanced, incoming)
        torch.testing.assert_close(state["context"][missing], context[missing], rtol=0, atol=0)
        torch.testing.assert_close(state["hidden"][missing], advanced["hidden"][missing], rtol=0, atol=0)
        assert not state["packet"][missing, :4].any()
        assert not diagnostic["innovation_valid"][missing].any()
        for case in range(2):
            if not missing[case]:
                assert diagnostic["elapsed_seconds"][case, 0].item() == pytest.approx((step - last[case]) * 0.02)
                assert incoming[case, 7] == 0
                last[case] = step


def test_elapsed_gate_uses_clock_not_zero_age_of_reacquired_packet():
    m = model("age")
    fixed_gate(m)
    prediction = prior(m)
    longer = clone(prediction)
    longer["real_index"] += 4
    longer["packet"][:, 7] = 0.1
    _, short_gap = m.assimilate_with_diagnostics(prediction, packet())
    _, long_gap = m.assimilate_with_diagnostics(longer, packet())
    assert (long_gap["gate"] > short_gap["gate"]).all()
    assert long_gap["elapsed_seconds"][0, 0].item() == pytest.approx(0.1)


def test_zero_innovation_gives_exact_zero_context_correction_for_every_variant():
    for variant in VARIANTS:
        m = model(variant)
        prediction = prior(m)
        incoming = packet()
        incoming[:, :4] = prediction["prior_mean"]
        actual, diagnostic = m.assimilate_with_diagnostics(prediction, incoming)
        assert not diagnostic["context_delta"].any()
        torch.testing.assert_close(actual["context"], prediction["context"], rtol=0, atol=0)


def test_variance_supervision_changes_only_variance_head_gradients():
    m = model()
    advanced = prior(m)
    advanced["prior_variance"].sum().backward()
    for name, parameter in m.named_parameters():
        if name.startswith("variance_head."):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None


def test_slow_correction_gradient_cannot_game_variance_but_retains_residual_path():
    m = model()
    prediction = prior(m)
    incoming = packet().requires_grad_()
    corrected, diagnostic = m.assimilate_with_diagnostics(prediction, incoming)
    corrected["context"].sum().backward()
    assert not diagnostic["gate_features"].requires_grad
    assert m.variance_head.weight.grad is None and m.variance_head.bias.grad is None
    for parameter in (m.context_innovation.weight, m.observation_head.weight, m.transition.weight_ih, m.gate.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert incoming.grad[:, :4].abs().sum() > 0
    # Removing U's error projection removes every incoming-angle gradient to
    # this slow correction; no detached gate-statistic route can replace it.
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        m.context_innovation.weight.zero_()
    other = packet().requires_grad_()
    m.assimilate(prior(m), other)["context"].sum().backward()
    assert not other.grad.any()


def test_module_counts_match_across_variants_and_hooks_cleanup_after_error():
    totals = []
    for variant in VARIANTS:
        m = model(variant)
        with capture_module_work(m) as counts:
            state = m.assimilate(m.initial(2), packet())
            for step in range(1, 4):
                state, _, _ = m.advance(state, torch.zeros(2, 2))
                state = m.assimilate(state, packet(visible=step == 3, age=step * 0.02 if step < 3 else 0))
        totals.append(counts)
        assert counts["observation_update"] == counts["gate"] == counts["context_innovation"] == {"calls": 4, "samples": 8}
        for name in ("transition", "observation_head", "variance_head", "reward_head.0", "reward_head.2"):
            assert counts[name] == {"calls": 3, "samples": 6}
        assert not any(module._forward_hooks for module in m.modules())
    assert all(value == totals[0] for value in totals)
    with pytest.raises(RuntimeError, match="deliberate"), capture_module_work(m):
        raise RuntimeError("deliberate")
    assert not any(module._forward_hooks for module in m.modules())


def test_imagination_never_corrects_context_and_candidates_are_private():
    m = model()
    root = m.assimilate(prior(m), packet())
    assert root["context"].abs().sum() > 0
    snapshot, rng = clone(root), torch.get_rng_state().clone()
    branch = repeat_index(root, torch.tensor([1, 0, 0, 1]))
    initial_context = branch["context"].clone()
    for _ in range(5):
        branch, angles, reward = m.advance(branch, torch.full((4, 2), 0.2))
        torch.testing.assert_close(branch["context"], initial_context, rtol=0, atol=0)
        assert angles.shape == (4, 4) and reward.shape == (4,)
    equal(root, snapshot)
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="one issued-command"):
        m.assimilate(branch, packet(4))
    # A candidate mutation cannot corrupt the real root's retained context.
    branch["context"].zero_()
    equal(root, snapshot)


def test_prediction_is_causal_and_full_episode_terminal_is_enforced():
    m = model()
    state = m.assimilate(m.initial(1), packet(1))
    for step in range(1, 51):
        state, _, _ = m.advance(state, torch.zeros(1, 2))
        state = m.assimilate(state, packet(1))
        assert state["real_index"].item() == step
    with pytest.raises(ValueError, match="terminal50"):
        m.advance(state, torch.zeros(1, 2))


@pytest.mark.parametrize("defect", ["first_missing", "age", "target", "visible_nan", "known_nan", "validity",
                                  "action_nan", "action_bound", "no_start", "no_advance", "clock", "state_keys"])
def test_public_phase_and_input_contract_errors_leave_inputs_unchanged(defect):
    m = model()
    state, incoming, action = prior(m), packet(), torch.zeros(2, 2)
    operation = "assimilate"
    if defect == "first_missing":
        state, incoming = m.initial(2), packet(visible=False)
    elif defect == "age":
        incoming[:, 7] = 0.02
    elif defect == "target":
        incoming[:, 4] += 0.1
    elif defect in {"visible_nan", "known_nan"}:
        incoming[:, 0 if defect == "visible_nan" else 4] = float("nan")
    elif defect == "validity":
        incoming[:, 6] = 0.5
    elif defect in {"action_nan", "action_bound", "no_start"}:
        operation = "advance"
        if defect == "no_start":
            state = m.initial(2)
        else:
            action[:, 0] = float("nan") if defect == "action_nan" else 1.1
    elif defect == "no_advance":
        state = m.assimilate(m.initial(2), packet())
    elif defect == "clock":
        state["last_valid_index"] += 2
    else:
        state["private_simulator_state"] = torch.zeros(2, 4)
    snapshot = clone(state)
    with pytest.raises(ValueError):
        getattr(m, operation)(state, incoming if operation == "assimilate" else action)
    equal(state, snapshot)


@pytest.mark.parametrize("kwargs", [{"eta": 0}, {"variance_min": 0}, {"variance_max": 1e-5}, {"dt": float("nan")}])
def test_invalid_provisional_settings_fail_before_parameter_draws(kwargs):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        InnovationContextWorldModel(hidden_size=4, context_size=4, **kwargs)
    assert torch.equal(torch.get_rng_state(), rng)
