"""Engineering fixtures only; no scored training or control experiments."""

import inspect

import pytest
import torch

from openjev.research.reacher_world_models import (
    GaussianRSSM,
    HistoryWorldModel,
    make_world_model,
    repeat_index,
    sequence_loss,
)


@pytest.fixture(autouse=True)
def seeded_single_thread():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(711)
    yield
    torch.set_num_threads(old_threads)


def packet(batch=3, valid=True):
    value = torch.zeros(batch, 8)
    value[:, :2] = 1
    value[:, 4:6] = torch.tensor([0.15, -0.07])
    value[:, 6] = float(valid)
    value[:, 7] = 0 if valid else 0.02
    return value


def sequence(batch=2, steps=6):
    packets = packet(batch * (steps + 1)).reshape(batch, steps + 1, 8)
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, 7] = torch.tensor([0.02, 0.04])[:packets[:, 2:4].shape[1]]
    commands = torch.linspace(-0.7, 0.7, batch * steps * 2).reshape(batch, steps, 2)
    rewards = -commands.square().sum(-1) - 0.2
    return packets, commands, rewards


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_public_shapes_action_gradients_and_root_immutability(kind):
    model = make_world_model(kind).eval()
    state = model.assimilate(model.initial(3), packet())
    before = {key: value.clone() for key, value in state.items()}
    action = torch.tensor([[0.3, -0.5], [0.5, 0.1], [-0.4, 0.2]], requires_grad=True)
    imagined, angles, reward = model.advance(state, action)
    assert angles.shape == (3, 4)
    assert reward.shape == (3,)
    assert not imagined["packet"][:, 6].any()
    torch.testing.assert_close(imagined["packet"][:, 4:6], packet()[:, 4:6])
    torch.testing.assert_close(imagined["packet"][:, 7], torch.full((3,), model.dt))
    (angles.square().sum() + reward.square().sum()).backward()
    assert torch.isfinite(action.grad).all()
    assert (action.grad.abs().sum(1) > 0).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    for key in state:
        torch.testing.assert_close(state[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_batch_independence_and_candidate_expansion(kind):
    model = make_world_model(kind).eval()
    public = packet()
    public[:, 0] = torch.tensor([0.3, 0.5, 0.7])
    state = model.assimilate(model.initial(3), public)
    indices = torch.tensor([2, 0, 2, 1])
    expanded = repeat_index(state, indices)
    action = torch.tensor([[0.1, 0.2], [0.2, 0.1], [-0.1, -0.2]])
    whole, obs, reward = model.advance(state, action)
    repeated, robs, rreward = model.advance(expanded, action.index_select(0, indices))
    torch.testing.assert_close(robs, obs.index_select(0, indices))
    torch.testing.assert_close(rreward, reward.index_select(0, indices))
    for key in repeated:
        torch.testing.assert_close(repeated[key], whole[key].index_select(0, indices))
    for index in range(3):
        single = model.assimilate(model.initial(1), public[index:index + 1])
        _, one_obs, one_reward = model.advance(single, action[index:index + 1])
        torch.testing.assert_close(one_obs, obs[index:index + 1])
        torch.testing.assert_close(one_reward, reward[index:index + 1])


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_missing_angles_never_enter_assimilation(kind):
    model = make_world_model(kind).eval()
    state = model.assimilate(model.initial(3), packet())
    state, _, _ = model.advance(state, torch.ones(3, 2) * 0.1)
    missing = packet(valid=False)
    poisoned = missing.clone()
    poisoned[:, :4] = torch.nan
    clean = model.assimilate(state, missing)
    dirty = model.assimilate(state, poisoned)
    for key in clean:
        torch.testing.assert_close(clean[key], dirty[key], rtol=0, atol=0)
    if "hidden" in state:
        torch.testing.assert_close(clean["hidden"], state["hidden"], rtol=0, atol=0)
    if kind == "rssm":
        torch.testing.assert_close(clean["stochastic"], state["stochastic"], rtol=0, atol=0)
        assert model.posterior_kl(clean).sum() == 0


def test_history_retains_autoregressive_estimates_without_marking_them_as_measurements():
    model = HistoryWorldModel(window=3, width=16).eval()
    original = model.assimilate(model.initial(1), packet(1))
    action = torch.tensor([[0.2, -0.1]])
    state, predicted, _ = model.advance(original, action)
    torch.testing.assert_close(state["packet"][:, :4], predicted, rtol=0, atol=0)
    assert state["packet"][:, 6].sum() == 0
    missing = packet(1, valid=False)
    missing[:, :4] = 777
    assimilated = model.assimilate(state, missing)
    torch.testing.assert_close(assimilated["packet"][:, :4], predicted, rtol=0, atol=0)
    assert assimilated["packet"][:, 6].sum() == 0
    tampered = {key: value.clone() for key, value in state.items()}
    tampered["packet"][:, :4] = 999
    tampered["history"][:, -1, :4] = -999
    _, clean_obs, clean_reward = model.advance(state, action)
    _, other_obs, other_reward = model.advance(tampered, action)
    assert not torch.equal(clean_obs, other_obs)
    assert not torch.equal(clean_reward, other_reward)
    for _ in range(model.window + 1):
        original, _, _ = model.advance(original, action)
        assert not original["packet"][:, 6].any()
    assert not original["history"][:, :, 6].any()
    clean_start = model.assimilate(model.initial(1), missing)
    assert not clean_start["packet"][:, :4].any()
    valid = packet(1)
    overwritten = model.assimilate(original, valid)
    torch.testing.assert_close(overwritten["packet"], valid, rtol=0, atol=0)


def test_rssm_samples_only_in_training_and_eval_does_not_consume_rng():
    model = GaussianRSSM(hidden_size=8, stochastic_size=3)
    initial = model.initial(2)
    first = model.assimilate(initial, packet(2))
    second = model.assimilate(initial, packet(2))
    assert not torch.equal(first["stochastic"], second["stochastic"])
    model.eval()
    rng = torch.random.get_rng_state().clone()
    first = model.assimilate(initial, packet(2))
    second = model.assimilate(initial, packet(2))
    next_state, _, _ = model.advance(first, torch.zeros(2, 2))
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(first["stochastic"], second["stochastic"], rtol=0, atol=0)
    torch.testing.assert_close(first["stochastic"], first["posterior_mean"], rtol=0, atol=0)
    torch.testing.assert_close(next_state["stochastic"], next_state["prior_mean"], rtol=0, atol=0)


@pytest.mark.parametrize("balance", [0.0, 0.8, 1.0])
def test_balanced_kl_matches_gaussian_formula_and_masks_absent_posterior(balance):
    model = GaussianRSSM(hidden_size=8, stochastic_size=3).eval()
    state = model.initial(2)
    state["posterior_mean"] = torch.full((2, 3), 0.7, requires_grad=True)
    state["prior_mean"] = torch.full((2, 3), -0.2, requires_grad=True)
    state["posterior_scale"] = torch.full((2, 3), 0.6, requires_grad=True)
    state["prior_scale"] = torch.full((2, 3), 1.1, requires_grad=True)
    state["posterior_valid"] = torch.tensor([[1.0], [0.0]])
    kl = model.posterior_kl(state, balance=balance, free_nats=0)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(state["posterior_mean"], state["posterior_scale"]),
        torch.distributions.Normal(state["prior_mean"], state["prior_scale"]),
    ).sum(-1)
    torch.testing.assert_close(kl[0], expected[0])
    assert kl[1] == 0
    kl.sum().backward()
    assert bool(state["prior_mean"].grad[0].abs().sum() > 0) == (balance > 0)
    assert bool(state["posterior_mean"].grad[0].abs().sum() > 0) == (balance < 1)
    for key in ("prior_mean", "posterior_mean", "prior_scale", "posterior_scale"):
        assert state[key].grad[1].abs().sum() == 0
    floored = model.posterior_kl(state, free_nats=50)
    torch.testing.assert_close(floored, torch.tensor([50.0, 0.0]))


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_sequence_loss_has_only_public_targets_and_all_missing_targets_are_safe(kind):
    model = make_world_model(kind)
    packets, commands, rewards = sequence()
    packets[..., 6] = 0
    packets[..., :4] = 99
    loss, metrics = sequence_loss(model, packets, commands, rewards)
    assert torch.isfinite(loss)
    assert metrics["observation_mse"] == 0
    assert metrics["rollout_observation_mse"] == 0
    assert metrics["rollout_reward_mse"] == 0
    assert metrics["kl_nats"] == 0
    assert metrics["reward_mse"] > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # The loss API cannot receive simulator-state or joint-velocity targets.
    assert set(inspect.signature(sequence_loss).parameters) == {
        "model", "packets", "commands", "rewards", "rollout_horizon", "rollout_weight",
        "reward_scale", "kl_weight", "kl_balance", "free_nats",
    }


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_rollout_from_one_prefix_cannot_assimilate_future_measurements(kind, monkeypatch):
    model = make_world_model(kind).eval()
    packets, commands, rewards = sequence(2, 5)
    packets[..., 6] = 1
    changed = packets.clone()
    changed[:, 1:, :4] = torch.rand_like(changed[:, 1:, :4])
    original = model.advance
    captures = []

    def capture(state, action):
        result = original(state, action)
        captures.append((result[1].detach().clone(), result[2].detach().clone()))
        return result

    monkeypatch.setattr(model, "advance", capture)
    sequence_loss(model, packets, commands, rewards, rollout_horizon=5)
    clean = captures.copy()
    captures.clear()
    sequence_loss(model, changed, commands, rewards, rollout_horizon=5)
    assert len(clean) == len(captures) == 10
    # First five calls are one-step fits; final five are a pure open-loop rollout.
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(clean[1:5], captures[1:5]))
    for a, b in zip(clean[5:], captures[5:]):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_full_sequence_backward_and_short_sequence_without_rollout(kind):
    model = make_world_model(kind)
    loss, metrics = sequence_loss(model, *sequence(), rollout_horizon=5)
    loss.backward()
    assert metrics["valid_rollout_starts"] == 4
    assert metrics["valid_observation_targets"] == 8
    assert metrics["rollout_reward_mse"] > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    _, short = sequence_loss(model, *sequence(2, 2), rollout_horizon=5)
    assert short["valid_rollout_starts"] == 0
    assert short["rollout_observation_mse"] == short["rollout_reward_mse"] == 0


@pytest.mark.parametrize("kind", ["history", "gru", "rssm"])
def test_missing_external_placeholders_do_not_change_training_objective(kind):
    model = make_world_model(kind).eval()
    packets, commands, rewards = sequence()
    changed = packets.clone()
    changed[..., :4][changed[..., 6] == 0] = 12345
    first, first_metrics = sequence_loss(model, packets, commands, rewards)
    second, second_metrics = sequence_loss(model, changed, commands, rewards)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first_metrics == second_metrics


@pytest.mark.parametrize("defect", ["packet_shape", "reward_shape", "nan", "valid", "age"])
def test_rejects_invalid_sequences(defect):
    model = make_world_model("gru")
    packets, commands, rewards = sequence()
    if defect == "packet_shape":
        packets = torch.cat((packets, torch.zeros_like(packets[..., :1])), -1)
    elif defect == "reward_shape":
        rewards = rewards[..., None]
    elif defect == "nan":
        rewards[0, 0] = torch.nan
    elif defect == "valid":
        packets[0, 0, 6] = 0.2
    else:
        packets[0, 0, 7] = -0.1
    with pytest.raises(ValueError):
        sequence_loss(model, packets, commands, rewards)


def test_constructor_and_loss_configuration_rejected():
    with pytest.raises(ValueError):
        make_world_model("dreamer")
    with pytest.raises(ValueError):
        make_world_model("gru", dt=0)
    with pytest.raises(ValueError):
        make_world_model("rssm", stochastic_size=0)
    with pytest.raises(ValueError):
        repeat_index({"x": torch.zeros(2, 3)}, torch.tensor([0.0]))
    model = make_world_model("history")
    for kwargs in ({"rollout_horizon": 0}, {"reward_scale": -1},
                   {"kl_balance": 1.2}, {"free_nats": -1}):
        with pytest.raises(ValueError):
            sequence_loss(model, *sequence(), **kwargs)
