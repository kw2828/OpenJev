import pytest
import torch

from openjev.research.associative_policy import MODES, AssociativePolicy, delta_write
from openjev.research.predictive_memory import PredictiveMemory


def make_policy(mode):
    torch.manual_seed(714)
    return AssociativePolicy(mode, observation_dim=44, hidden=8, store_dim=4, adapter_hidden=5)


def sequence_inputs():
    torch.manual_seed(191)
    observations = torch.randn(4, 2, 44)
    previous = torch.randint(7, (4, 2))
    resets = torch.tensor([[True, False], [False, False], [False, True], [True, False]])
    return observations, previous, resets


def test_backbone_matches_frozen_model_exactly_for_every_mode():
    torch.manual_seed(83)
    original = PredictiveMemory()
    names = ('encoder.', 'memory.', 'actor.', 'value.')
    original_weights = {name: value for name, value in original.state_dict().items() if name.startswith(names)}
    for mode in MODES:
        torch.manual_seed(83)
        model = AssociativePolicy(mode)
        for name, value in original_weights.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        assert not any(name.startswith(('dynamics.', 'observation.', 'reward.', 'termination.'))
                       for name in model.state_dict())


def test_gru_control_matches_frozen_policy_outputs():
    torch.manual_seed(714)
    original = PredictiveMemory(observation_dim=44, hidden=8)
    model = make_policy('gru')
    observations, previous, resets = sequence_inputs()
    state = torch.randn(2, 8)
    for current_only in (False, True):
        expected = original.sequence(observations, previous, state, resets, current_only)
        actual = model.sequence(observations, previous, state, resets, current_only)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_fast_modes_have_identical_initial_outputs_and_documented_gate_difference():
    global_model, selective = make_policy('fast_global'), make_policy('fast_selective')
    for key, value in global_model.state_dict().items():
        torch.testing.assert_close(value, selective.state_dict()[key], rtol=0, atol=0)
    observations, previous, resets = sequence_inputs()
    a = global_model.sequence(observations, previous, global_model.initial_state(2), resets)
    b = selective.sequence(observations, previous, selective.initial_state(2), resets)
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not global_model.write_gate.weight.requires_grad
    assert selective.write_gate.weight.requires_grad
    assert global_model.write_gate.bias.requires_grad
    assert global_model.write_gate.bias.sigmoid().item() == pytest.approx(.1)
    assert not selective.write_gate.weight.count_nonzero()


def test_default_parameter_and_runtime_state_counts_are_explicit():
    expected = {'gru': (89864, 89864, 64), 'feedforward': (94185, 94185, 64),
                'fast_global': (94137, 94073, 320), 'fast_selective': (94137, 94137, 320)}
    for mode, (registered, trainable, state_size) in expected.items():
        model = AssociativePolicy(mode)
        assert sum(p.numel() for p in model.parameters()) == registered
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable
        assert model.state_size == state_size
        assert model.initial_state(3).shape == (3, state_size)
        assert not model.initial_state(3).requires_grad


@pytest.mark.parametrize('mode', MODES)
def test_episode_reset_prevents_state_and_previous_action_leakage(mode):
    model = make_policy(mode)
    observations, previous, _ = sequence_inputs()
    reset = torch.ones(2, dtype=torch.bool)
    a = model.observe(observations[0], previous[0], torch.randn_like(model.initial_state(2)), reset)
    b = model.observe(observations[0], (previous[0] + 3) % 7,
                      torch.randn_like(model.initial_state(2)) * 100, reset)
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_current_only_clears_all_state_without_silently_resetting_previous_action(mode):
    model = make_policy(mode)
    observations, previous, _ = sequence_inputs()
    reset = torch.zeros(2, dtype=torch.bool)
    actual = model.observe(observations[0], previous[0], torch.randn_like(model.initial_state(2)),
                           reset, current_only=True)
    expected = model.observe(observations[0], previous[0], model.initial_state(2), reset)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_sequence_matches_incremental_replay_and_future_cannot_change_prefix(mode):
    model = make_policy(mode)
    observations, previous, resets = sequence_inputs()
    initial = torch.randn_like(model.initial_state(2))
    store_resets = torch.tensor([[False, False], [True, False], [False, False], [False, True]])
    expected = model.sequence(observations, previous, initial, resets, reset_store=store_resets)
    state, logits, values, states = initial, [], [], []
    for t in range(len(observations)):
        logit, value, state = model.observe(observations[t], previous[t], state, resets[t],
                                           reset_store=store_resets[t])
        logits.append(logit)
        values.append(value)
        states.append(state)
    for first, second in zip(expected, (logits, values, states), strict=True):
        torch.testing.assert_close(first, torch.stack(second), rtol=0, atol=0)
    altered_obs, altered_previous = observations.clone(), previous.clone()
    altered_obs[2:] = torch.randn_like(altered_obs[2:]) * 100
    altered_previous[2:] = (altered_previous[2:] + 1) % 7
    actual = model.sequence(altered_obs, altered_previous, initial, resets, reset_store=store_resets)
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first[:2], second[:2], rtol=0, atol=0)


def test_delta_update_corrects_only_the_addressed_direction_without_in_place_mutation():
    store = torch.tensor([[[.2, 0.], [0., .3]]])
    before = store.clone()
    key = torch.tensor([[1., 0.]])
    content = torch.tensor([[.6, -.4]])
    actual = delta_write(store, key, content, torch.tensor([[.25]]))
    torch.testing.assert_close(actual, torch.tensor([[[.3, 0.], [-.1, .3]]]))
    torch.testing.assert_close(store, before, rtol=0, atol=0)
    full_write = delta_write(store, key, content, torch.ones(1, 1))
    torch.testing.assert_close(torch.bmm(full_write, key.unsqueeze(-1)).squeeze(-1), content)
    torch.testing.assert_close(full_write[:, :, 1], store[:, :, 1], rtol=0, atol=0)
    torch.testing.assert_close(delta_write(store, key, content, torch.zeros(1, 1)), store, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['fast_global', 'fast_selective'])
def test_store_reset_preserves_reactive_state_and_still_writes_current_observation(mode):
    model = make_policy(mode)
    observations, previous, _ = sequence_inputs()
    state = torch.randn_like(model.initial_state(2))
    reset = torch.zeros(2, dtype=torch.bool)
    intact = model.observe(observations[0], previous[0], state, reset)
    cleared = model.observe(observations[0], previous[0], state, reset, reset_store=True)
    zero_matrix = state.clone()
    zero_matrix[:, model.hidden:] = 0
    expected = model.observe(observations[0], previous[0], zero_matrix, reset)
    torch.testing.assert_close(intact[2][:, :model.hidden], cleared[2][:, :model.hidden], rtol=0, atol=0)
    for first, second in zip(cleared, expected, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert cleared[2][:, model.hidden:].count_nonzero() > 0
    assert not torch.allclose(intact[0], cleared[0])
    mixed = model.observe(observations[0], previous[0], state, reset,
                           reset_store=torch.tensor([True, False]))
    for actual, zeroed, original in zip(mixed, cleared, intact, strict=True):
        torch.testing.assert_close(actual[0], zeroed[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], original[1], rtol=0, atol=0)


@pytest.mark.parametrize('mode', MODES)
def test_finite_gradients_reach_trainable_paths_but_not_frozen_global_gate_weights(mode):
    model = make_policy(mode)
    observations, previous, resets = sequence_inputs()
    observations.requires_grad_()
    state = torch.randn_like(model.initial_state(2), requires_grad=True)
    logits, values, states = model.sequence(observations, previous, state, resets)
    loss = logits.square().mean() + values.square().mean() + .1 * states.square().mean()
    loss.backward()
    assert torch.isfinite(observations.grad).all()
    assert torch.isfinite(state.grad).all()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    if mode == 'fast_selective':
        assert model.write_gate.weight.grad.abs().sum() > 0
    if mode.startswith('fast_'):
        assert model.write_gate.bias.grad.abs().sum() > 0
        assert model.key.weight.grad.abs().sum() > 0
        assert model.query.weight.grad.abs().sum() > 0


def test_zero_projected_keys_and_queries_remain_finite():
    model = make_policy('fast_selective')
    with torch.no_grad():
        for layer in (model.key, model.query):
            layer.weight.zero_()
            layer.bias.zero_()
    observations, previous, resets = sequence_inputs()
    output = model.sequence(observations, previous, model.initial_state(2), resets)
    assert all(torch.isfinite(value).all() for value in output)
    assert output[2][:, :, model.hidden:].count_nonzero() == 0
    sum(value.square().mean() for value in output).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('field', ['observations', 'state'])
def test_nonfinite_inputs_are_rejected(field):
    model = make_policy('fast_selective')
    observations, previous, resets = sequence_inputs()
    state = model.initial_state(2)
    if field == 'observations':
        observations[0, 0, 0] = float('nan')
    else:
        state[0, -1] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        model.observe(observations[0], previous[0], state, resets[0])


def test_invalid_modes_and_state_shapes_are_rejected():
    with pytest.raises(ValueError, match='mode'):
        AssociativePolicy('unregistered')
    model = make_policy('fast_global')
    observations, previous, resets = sequence_inputs()
    with pytest.raises(ValueError, match='state shape'):
        model.observe(observations[0], previous[0], torch.zeros(2, 8), resets[0])
    with pytest.raises(ValueError, match='Store reset flags'):
        model.observe(observations[0], previous[0], model.initial_state(2), resets[0],
                       reset_store=torch.ones(2))
