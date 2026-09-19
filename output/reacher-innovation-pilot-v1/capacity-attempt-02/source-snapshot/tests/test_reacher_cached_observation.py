"""Synthetic public-cache invariants, no corpus, fits or scored streams."""

import io

import pytest
import torch

from openjev.research.reacher_cached_observation import (
    AGE_ATOL,
    AGE_RTOL,
    CachedObservationGRUWorldModel,
    CachedObservationMLPWorldModel,
    EncodedCurrentGRUWorldModel,
    configuration_for,
    operation_counts,
    parameter_accounting,
)
from openjev.research.reacher_observation_baseline import FeedForwardObservationWorldModel
from openjev.research.reacher_reward_residual import GRUResidualRewardWorldModel
from openjev.research.reacher_world_models import repeat_index, sequence_loss

CLASSES = (EncodedCurrentGRUWorldModel, CachedObservationGRUWorldModel, CachedObservationMLPWorldModel)
CACHED = (CachedObservationGRUWorldModel, CachedObservationMLPWorldModel)


@pytest.fixture(autouse=True)
def engineering_rng():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(previous)


def model_for(cls, **kwargs):
    return cls(**({"width": 7} if cls is CachedObservationMLPWorldModel else {"hidden_size": 5}), **kwargs)


def packets(batch=2, points=7, missing=()):
    angles = torch.linspace(-0.5, 0.8, batch * points * 2).reshape(batch, points, 2)
    public = torch.zeros(batch, points, 8)
    public[..., :2], public[..., 2:4] = angles.cos(), angles.sin()
    public[..., 4:6] = torch.tensor([0.1, -0.09])
    public[..., 6] = 1
    last = 0
    for step in range(points):
        if step in missing:
            public[:, step, :4] = 0
            public[:, step, 6] = 0
            public[:, step, 7] = (step - last) * 0.02
        else:
            last = step
    return public


def commands(batch=2, steps=6):
    return torch.linspace(-0.6, 0.7, batch * steps * 2).reshape(batch, steps, 2)


def real_rollout(model, public, issued):
    state = model.initial(len(public))
    for step in range(public.shape[1]):
        if step:
            state, _, _ = model.advance(state, issued[:, step - 1])
        state = model.assimilate(state, public[:, step])
    return state


def same(left, right):
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize("cls", CLASSES)
def test_constructor_rng_parameter_names_shapes_order_and_values_are_identical(cls):
    parent = (
        FeedForwardObservationWorldModel
        if cls is CachedObservationMLPWorldModel
        else GRUResidualRewardWorldModel
    )
    sizes = {"width": 7} if cls is CachedObservationMLPWorldModel else {"hidden_size": 5}
    torch.manual_seed(410)
    original = parent(**sizes)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(410)
    new = cls(**sizes)
    assert new.__init__.__func__ is parent.__init__
    assert torch.equal(rng, torch.get_rng_state())
    assert list(original.state_dict()) == list(new.state_dict())
    same(original.state_dict(), new.state_dict())
    assert not list(new.named_buffers())
    new.initial(2)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mlp", [False, True])
def test_fully_observed_matched_current_and_cached_predictions_equal_exactly(mlp):
    if mlp:
        current = FeedForwardObservationWorldModel(width=7)
        cache = CachedObservationMLPWorldModel(width=7)
    else:
        current = EncodedCurrentGRUWorldModel(hidden_size=5)
        cache = CachedObservationGRUWorldModel(hidden_size=5)
    cache.load_state_dict(current.state_dict())
    public, actions = packets(), commands(steps=7)
    one, two = current.initial(2), cache.initial(2)
    for step in range(7):
        one = current.assimilate(one, public[:, step])
        two = cache.assimilate(two, public[:, step])
        one, obs, reward = current.advance(one, actions[:, step])
        two, obs2, reward2 = cache.advance(two, actions[:, step])
        torch.testing.assert_close(obs, obs2, rtol=0, atol=0)
        torch.testing.assert_close(reward, reward2, rtol=0, atol=0)


@pytest.mark.parametrize("cls", CLASSES)
def test_encoder_uses_declared_features_without_falsifying_actual_packet(cls):
    model = model_for(cls)
    public = packets(points=2, missing=(1,))
    first = model.assimilate(model.initial(2), public[:, 0])
    previous, _, _ = model.advance(first, commands(steps=1)[:, 0])
    state = model.assimilate(previous, public[:, 1])
    expected_features = public[:, 1].clone()
    if cls in CACHED:
        expected_features[:, :4] = public[:, 0, :4]
        torch.testing.assert_close(state["cached_angles"], public[:, 0, :4], rtol=0, atol=0)
    torch.testing.assert_close(state["packet"], public[:, 1], rtol=0, atol=0)
    assert not state["packet"][:, :4].any() and not state["packet"][:, 6].any()
    if cls is CachedObservationMLPWorldModel:
        torch.testing.assert_close(state["encoder_features"], expected_features, rtol=0, atol=0)
        parent = FeedForwardObservationWorldModel(width=7)
        parent.load_state_dict(model.state_dict())
        _, expected_obs, expected_reward = parent.advance({"packet": expected_features}, torch.zeros(2, 2))
        advanced, obs, reward = model.advance(state, torch.zeros(2, 2))
        torch.testing.assert_close(obs, expected_obs, rtol=0, atol=0)
        torch.testing.assert_close(reward, expected_reward, rtol=0, atol=0)
        torch.testing.assert_close(advanced["encoder_features"], advanced["packet"], rtol=0, atol=0)
    else:
        expected_hidden = model.observation_update(expected_features, torch.zeros(2, 5))
        torch.testing.assert_close(state["hidden"], expected_hidden, rtol=0, atol=0)
        assert state["hidden"].abs().sum() > 0  # Missingness does not gate encoding.


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("gap", [6, 10])
def test_blackout_clock_and_cache_are_only_updated_by_actual_measurements(cls, gap):
    model = model_for(cls)
    public = packets(points=gap + 3, missing=range(1, gap + 1))
    actions = commands(steps=gap + 2)
    state = model.initial(2)
    for step in range(gap + 3):
        if step:
            state, _, _ = model.advance(state, actions[:, step - 1])
        state = model.assimilate(state, public[:, step])
        last = 0 if step <= gap else step
        assert (state["real_index"] == step).all()
        assert (state["last_visible_index"] == last).all()
        assert not state["imagined_depth"].any()
        if cls in CACHED:
            torch.testing.assert_close(state["cached_angles"], public[:, last, :4], rtol=0, atol=0)
    fresh = model.initial(2)
    assert (fresh["real_index"] == -1).all() and (fresh["last_visible_index"] == -1).all()
    if cls in CACHED:
        assert not fresh["cached_angles"].any()


@pytest.mark.parametrize("cls", CLASSES)
def test_same_declared_cache_and_current_packet_erase_older_actions_and_predictions(cls):
    model = model_for(cls)
    public = packets(points=7, missing=(5, 6))
    changed = public.clone()
    changed[:, :4, :4] *= -1
    actions = commands()
    left = real_rollout(model, public, actions)
    right = real_rollout(model, changed, -actions)
    same(left, right)
    for _ in range(3):
        left, obs, reward = model.advance(left, torch.zeros(2, 2))
        right, other, other_reward = model.advance(right, torch.zeros(2, 2))
        same(left, right)
        torch.testing.assert_close(obs, other, rtol=0, atol=0)
        torch.testing.assert_close(reward, other_reward, rtol=0, atol=0)


@pytest.mark.parametrize("cls", CLASSES)
def test_boundary_discards_old_learned_poison_and_missing_placeholders(cls):
    model = model_for(cls)
    public = packets(points=2, missing=(1,))
    root = model.assimilate(model.initial(2), public[:, 0])
    previous, _, _ = model.advance(root, torch.zeros(2, 2))
    clean = model.assimilate(previous, public[:, 1])
    poisoned = {key: value.clone() for key, value in previous.items()}
    for key in ("packet", "hidden", "encoder_features"):
        if key in poisoned:
            poisoned[key].fill_(float("nan"))
    incoming = public[:, 1].clone()
    incoming[:, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 321])
    same(clean, model.assimilate(poisoned, incoming))
    incoming[:, 4] = float("nan")
    with pytest.raises(ValueError, match="known public"):
        model.assimilate(poisoned, incoming)
    if cls in CACHED:
        poisoned["cached_angles"].fill_(float("nan"))
        with pytest.raises(ValueError, match="Nonfinite"):
            model.assimilate(poisoned, public[:, 1])


@pytest.mark.parametrize("cls", CLASSES)
def test_strict_real_phase_and_initial_observation(cls):
    model = model_for(cls)
    public = packets(points=2)
    initial = model.initial(2)
    with pytest.raises(ValueError, match="initial real packet"):
        model.advance(initial, torch.zeros(2, 2))
    missing = public[:, 0].clone()
    missing[:, 6] = 0
    with pytest.raises(ValueError, match="first real packet must be visible"):
        model.assimilate(initial, missing)
    root = model.assimilate(initial, public[:, 0])
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(root, public[:, 1])
    selected, _, _ = model.advance(root, torch.zeros(2, 2))
    terminal, _, _ = model.advance(selected, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="exactly one"):
        model.assimilate(terminal, public[:, 1])
    assert (model.assimilate(selected, public[:, 1])["real_index"] == 1).all()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("corruption", ["age", "visible_age", "target", "validity", "batch"])
def test_reject_inconsistent_real_packet_contract(cls, corruption):
    model = model_for(cls)
    public = packets(points=2, missing=(1,))
    root = model.assimilate(model.initial(2), public[:, 0])
    previous, _, _ = model.advance(root, torch.zeros(2, 2))
    incoming = public[:, 1].clone()
    if corruption == "age":
        incoming[:, 7] = 0.04
    elif corruption == "visible_age":
        incoming[:, 6] = 1
    elif corruption == "target":
        incoming[:, 4] += 0.001
    elif corruption == "validity":
        incoming[:, 6] = 0.5
    elif corruption == "batch":
        incoming = incoming[:1]
    with pytest.raises(ValueError):
        model.assimilate(previous, incoming)


def test_age_tolerance_is_explicit_and_supports_float32_roundoff():
    model = CachedObservationGRUWorldModel(hidden_size=5)
    public = packets(points=2, missing=(1,))
    root = model.assimilate(model.initial(2), public[:, 0])
    previous, _, _ = model.advance(root, torch.zeros(2, 2))
    public[:, 1, 7] += AGE_ATOL / 2
    model.assimilate(previous, public[:, 1])
    public[:, 1, 7] += 10 * AGE_ATOL
    with pytest.raises(ValueError, match="Public age"):
        model.assimilate(previous, public[:, 1])
    assert model.configuration()["age_validation"] == {"rtol": AGE_RTOL, "atol": AGE_ATOL, "visible_age": 0.0}


@pytest.mark.parametrize("cls", CLASSES)
def test_candidate_branches_and_returned_metadata_do_not_alias_real_root(cls):
    model = model_for(cls)
    public = packets(points=3, missing=(1, 2))
    root = real_rollout(model, public, commands(steps=2))
    saved = {key: value.clone() for key, value in root.items()}
    indices = torch.tensor([0, 0, 1])
    branch = repeat_index(root, indices)
    action = torch.tensor([[0.1, 0.5], [-0.2, 0.3], [0.3, -0.1]])
    for depth in range(1, 5):
        previous = branch
        branch, obs, reward = model.advance(branch, action)
        for index in range(3):
            isolated = {key: value[index : index + 1] for key, value in previous.items()}
            _, ref_obs, ref_reward = model.advance(isolated, action[index : index + 1])
            torch.testing.assert_close(obs[index : index + 1], ref_obs, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(reward[index : index + 1], ref_reward, rtol=1e-6, atol=1e-7)
        assert (branch["imagined_depth"] == depth).all()
        for key in ("real_index", "last_visible_index", "real_target", "cached_angles"):
            if key in branch:
                torch.testing.assert_close(branch[key], saved[key].index_select(0, indices), rtol=0, atol=0)
        assert not branch["packet"][:, 6].any()
    same(root, saved)
    for value in branch.values():
        value.detach().zero_()
    same(root, saved)
    # Assimilation results do not alias the original incoming public packet.
    public.zero_()
    same(root, saved)


@pytest.mark.parametrize("cls", CLASSES)
def test_gradient_cannot_cross_boundary_through_old_learned_state_or_actions(cls):
    model = model_for(cls)
    public = packets(points=3, missing=(2,)).requires_grad_()
    actions = commands(steps=2).requires_grad_()
    state = real_rollout(model, public, actions)
    _, obs, reward = model.advance(state, torch.full((2, 2), 0.2))
    obs.square().sum().add(reward.sum()).backward()
    assert public.grad is not None
    assert not public.grad[:, 0, :4].any()
    assert not public.grad[:, 2, :4].any()  # Missing placeholders never matter.
    if cls in CACHED:
        assert public.grad[:, 1, :4].abs().sum() > 0
    else:
        assert not public.grad[:, 1, :4].any()
    assert actions.grad is None or not actions.grad.any()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_unchanged_sequence_loss_masks_backprop_and_safe_weight_roundtrip(cls):
    model = model_for(cls)
    public, issued = packets(points=9, missing=(2, 3, 4, 6)), commands(steps=8)
    rewards = torch.linspace(-1.5, -0.1, 16).reshape(2, 8)
    loss, metrics = sequence_loss(model, public, issued, rewards, rollout_horizon=3)
    loss.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    assert torch.isfinite(loss)
    assert metrics["valid_observation_targets"] == float(public[:, 1:, 6].sum())
    assert metrics["valid_rollout_starts"] == float(public[:, :6, 6].sum())
    assert metrics["kl_nats"] == 0
    assert all(torch.isfinite(grad).all() for grad in gradients.values())
    altered = public.clone()
    altered[..., :4][altered[..., 6] == 0] = 9999
    model.zero_grad(set_to_none=True)
    other, other_metrics = sequence_loss(model, altered, issued, rewards, rollout_horizon=3)
    other.backward()
    assert metrics == other_metrics
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save({"configuration": model.configuration(), "weights": model.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored = model_for(cls)
    assert payload["configuration"] == restored.configuration()
    restored.load_state_dict(payload["weights"], strict=True)
    same(real_rollout(model, public, issued), real_rollout(restored, public, issued))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "field", ["extra", "dtype", "negative_index", "last_future", "depth", "target", "nonfinite", "action"]
)
def test_reject_corrupt_imagined_state_and_action(cls, field):
    model = model_for(cls)
    root = model.assimilate(model.initial(2), packets()[:, 0])
    action = torch.zeros(2, 2)
    if field == "extra":
        root["native_qvel"] = torch.zeros(2, 2)
    elif field == "dtype":
        root["real_index"] = root["real_index"].float()
    elif field == "negative_index":
        root["real_index"].fill_(-2)
    elif field == "last_future":
        root["last_visible_index"].fill_(1)
    elif field == "depth":
        root["imagined_depth"].fill_(-1)
    elif field == "target":
        root["packet"][:, 4] += 0.1
    elif field == "nonfinite":
        root["packet"][:, 0] = float("nan")
    elif field == "action":
        action.fill_(1.01)
    with pytest.raises(ValueError):
        model.advance(root, action)


def test_mixed_observation_batch_has_independent_cache_times():
    model = CachedObservationGRUWorldModel(hidden_size=5)
    public = packets(points=4)
    public[0, 1:3, :4] = 0
    public[0, 1:3, 6] = 0
    public[0, 1:3, 7] = torch.tensor([0.02, 0.04])
    public[1, 2:, :4] = 0
    public[1, 2:, 6] = 0
    public[1, 2:, 7] = torch.tensor([0.02, 0.04])
    state = real_rollout(model, public, commands(steps=3))
    assert state["last_visible_index"].flatten().tolist() == [3, 1]
    torch.testing.assert_close(
        state["cached_angles"], torch.stack((public[0, 3, :4], public[1, 1, :4])), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "cls,count",
    [
        (EncodedCurrentGRUWorldModel, 36805),
        (CachedObservationGRUWorldModel, 36805),
        (CachedObservationMLPWorldModel, 36599),
    ],
)
def test_parameter_cost_and_state_payload_accounting(cls, count):
    model = cls()
    report = parameter_accounting(model)
    assert (
        report["trainable_parameters"] == count == sum(parameter.numel() for parameter in model.parameters())
    )
    state = model.initial(1)
    assert report["state_payload_per_case"]["tensor_bytes"] == sum(
        value.numel() * value.element_size() for value in state.values()
    )
    calls = operation_counts(model, assimilate_samples=3, advance_samples=11)
    recurrent = cls is not CachedObservationMLPWorldModel
    assert calls["gru_cell_sample_calls"] == (14 if recurrent else 0)
    assert calls["linear_layer_sample_calls"] == (44 if recurrent else 66)
    assert calls["analytic_reward_sample_calls"] == 11
    assert (
        calls["dense_affine_macs"]
        == 3 * report["dense_affine_macs_per_sample"]["assimilate"]
        + 11 * report["dense_affine_macs_per_sample"]["advance"]
    )
    assert report["run_status_authority"] == "enclosing protocol and execution receipts"
    assert report["model_class"] == cls.__name__
    assert len(report["encoder_feature_order"]) == 8
    assert report["state_keys"] == sorted(state)
    model.residual_reward = False
    assert operation_counts(model, advance_samples=11)["analytic_reward_sample_calls"] == 0


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_operation_counts_refuse_invalid_counts(count):
    with pytest.raises(ValueError, match="Nonnegative integer"):
        operation_counts(EncodedCurrentGRUWorldModel(hidden_size=5), assimilate_samples=count)


def test_operation_counts_require_exact_declared_class():
    with pytest.raises(ValueError, match="exact public-cache"):
        operation_counts(GRUResidualRewardWorldModel(hidden_size=5))


@pytest.mark.parametrize("cls", CLASSES)
def test_configuration_helper_is_pure_and_matches_actual_class(cls, monkeypatch):
    model = model_for(cls)
    expected = model.configuration()
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Pure configuration helper must not construct or draw RNG")

    monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    actual = configuration_for(
        cls,
        width=7 if cls is CachedObservationMLPWorldModel else 5,
        dt=0.02,
        noise_std=0.05,
        residual_reward=True,
    )
    assert actual == expected
    assert torch.equal(rng, torch.get_rng_state())
    actual["encoder_feature_order"][0] = "mutated"
    assert expected["encoder_feature_order"][0] != "mutated"


@pytest.mark.parametrize(
    "key,bad",
    [
        ("width", True),
        ("width", 0),
        ("dt", float("nan")),
        ("dt", False),
        ("noise_std", -0.1),
        ("noise_std", float("inf")),
        ("residual_reward", 1),
    ],
)
def test_configuration_helper_rejects_invalid_scalars(key, bad):
    arguments = {"width": 5, "dt": 0.02, "noise_std": 0.05, "residual_reward": True}
    arguments[key] = bad
    with pytest.raises(ValueError):
        configuration_for(CachedObservationGRUWorldModel, **arguments)


def test_configuration_helper_rejects_wrong_class_and_subclass():
    class Undeclared(CachedObservationGRUWorldModel):
        pass

    for cls in (Undeclared, GRUResidualRewardWorldModel, "CachedObservationGRUWorldModel"):
        with pytest.raises(ValueError, match="Exact public-cache model class"):
            configuration_for(cls, width=5, dt=0.02, noise_std=0.05, residual_reward=True)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("field", ["_cached", "_recurrent", "_keys"])
@pytest.mark.parametrize("on_class", [False, True])
def test_configuration_rejects_behavior_flag_or_state_key_drift(cls, field, on_class, monkeypatch):
    model = model_for(cls)
    expected = model.configuration()
    target = cls if on_class else model
    if field == "_keys":
        wrong = set(model._keys()) | {"hidden_native_state"}
        replacement = (lambda self: wrong) if on_class else (lambda: wrong)
    else:
        replacement = not getattr(model, field)
    monkeypatch.setattr(target, field, replacement)
    # Pure expected metadata remains fixed even if runtime flags are changed.
    assert (
        configuration_for(
            cls,
            width=7 if cls is CachedObservationMLPWorldModel else 5,
            dt=0.02,
            noise_std=0.05,
            residual_reward=True,
        )
        == expected
    )
    with pytest.raises(ValueError, match="flags changed|state keys changed"):
        model.configuration()
