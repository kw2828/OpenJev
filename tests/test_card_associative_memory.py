"""Tiny CPU tensor tests only; no deck, environment, training or native calls."""

import json
import math

import pytest
import torch
from torch.nn import functional as F

from openjev.research.card_associative_memory import (
    INNOVATION_MODES,
    KALMAN_MODES,
    MODES,
    CardAssociativeMemory,
    clone_state,
    copy_shared_initialization,
    detach_state,
)


@pytest.fixture(autouse=True)
def isolated_tiny_cpu_randomness():
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        yield
    torch.set_num_threads(prior_threads)


def event(batch=1):
    return torch.arange(batch, dtype=torch.long), torch.arange(batch, dtype=torch.long), torch.ones(batch).bool()


def configured(mode):
    """A hand-specified nonuniform key and value, nonzero shared prior."""
    model = CardAssociativeMemory(mode).double()
    with torch.no_grad():
        model.key_embedding.weight[0].zero_()
        model.key_embedding.weight[0, :2] = torch.tensor([3., 4.], dtype=torch.float64)
        model.key_embedding.weight[1].zero_()
        model.key_embedding.weight[1, 0] = 1.
        model.value_embedding.weight[0] = torch.linspace(-.4, .8, 16, dtype=torch.float64)
        if mode != "gru":
            model.gate.weight.zero_()
            model.gate.bias.copy_(torch.tensor([math.log(.8 / .2), math.log(.3 / .7)], dtype=torch.float64))
        if mode in KALMAN_MODES:
            model.noise.weight.zero_()
            model.noise.bias.copy_(torch.tensor([math.log(math.expm1(.2 - 1e-4)),
                                                 math.log(math.expm1(.03 - 1e-5))], dtype=torch.float64))
        if mode in INNOVATION_MODES:
            model.innovation_scale.copy_(torch.tensor(math.log(math.expm1(.4)), dtype=torch.float64))
    state = model.init_state(1)
    state["S"] = torch.arange(512, dtype=torch.float64).reshape(1, 32, 16) / 5000
    if state["P"] is not None:
        state["P"] = torch.linspace(.5, 1.4, 32, dtype=torch.float64)[None]
    return model, state


@pytest.mark.parametrize("mode", MODES)
def test_initialization_shapes_priors_and_independent_episode_state(mode):
    model = CardAssociativeMemory(mode)
    state, second = model.init_state(2, "cpu"), model.init_state(2, "cpu")
    assert state["S"].shape == (2, 32, 16) and torch.count_nonzero(state["S"]) == 0
    state["S"][0, 0, 0] = 99
    assert second["S"][0, 0, 0] == 0 and state["S"][1, 0, 0] == 0
    if mode in KALMAN_MODES:
        torch.testing.assert_close(state["P"], torch.ones(2, 32))
        assert state["P"].data_ptr() != second["P"].data_ptr()
    else:
        assert state["P"] is second["P"] is None
    if mode != "gru":
        torch.testing.assert_close(model.gate.bias.sigmoid(), torch.tensor([.99, .5]))
    if mode in KALMAN_MODES:
        torch.testing.assert_close(F.softplus(model.noise.bias) + torch.tensor([1e-4, 1e-5]),
                                   torch.tensor([.1, .001]))
    if mode in INNOVATION_MODES:
        torch.testing.assert_close(F.softplus(model.innovation_scale), torch.tensor(.1))


@pytest.mark.parametrize("mode", MODES)
def test_shared_query_readout_matches_direct_matrix_math_without_writes(mode):
    model, state = configured(mode)
    queries = torch.tensor([[0, 1, 0]])
    before = clone_state(state)
    key = model.key_embedding(queries)
    key = key / torch.sqrt((key * key).sum(dim=-1, keepdim=True))
    expected_read = torch.stack([state["S"][0].T @ key[0, q] for q in range(3)])[None]
    expected = expected_read @ model.decoder.weight.T + model.decoder.bias
    actual = model.predict(state, queries)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(state["S"], before["S"], rtol=0, atol=0)
    assert actual.shape == (1, 3, 13)


@pytest.mark.parametrize("mode", MODES)
def test_read_before_write_has_no_future_rank_and_input_state_is_unchanged(mode):
    model = CardAssociativeMemory(mode)
    state = model.init_state(1)
    q = torch.tensor([[0, 1]])
    prior = model.predict(state, q).clone()
    next_a, _ = model.write(state, torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))
    next_b, _ = model.write(state, torch.tensor([0]), torch.tensor([1]), torch.tensor([True]))
    torch.testing.assert_close(model.predict(state, q), prior, atol=0, rtol=0)
    assert torch.count_nonzero(state["S"]) == 0
    assert not torch.equal(model.predict(next_a, q), model.predict(next_b, q))
    assert next_a["S"].data_ptr() != state["S"].data_ptr()


@pytest.mark.parametrize("mode", MODES)
def test_masked_padding_identity_and_valid_row_equals_unbatched_update(mode):
    model = CardAssociativeMemory(mode)
    state = model.init_state(2)
    state["S"][1].fill_(.125)
    if state["P"] is not None:
        state["P"][1].fill_(.7)
    old = clone_state(state)
    new, diag = model.write(state, torch.tensor([0, -999]), torch.tensor([1, 999]), torch.tensor([True, False]))
    single = {k: None if v is None else v[:1] for k, v in state.items()}
    expected, _ = model.write(single, torch.tensor([0]), torch.tensor([1]), torch.tensor([True]))
    for k in state:
        if state[k] is not None:
            torch.testing.assert_close(new[k][1], old[k][1], atol=0, rtol=0)
            torch.testing.assert_close(new[k][:1], expected[k], atol=2e-7, rtol=1e-6)
            torch.testing.assert_close(state[k], old[k], atol=0, rtol=0)
    for value in diag.values():
        assert value.shape == (2,) and value[1] == 0


@pytest.mark.parametrize("mode", MODES)
def test_all_padding_has_zero_parameter_gradients_and_preserves_state_gradient(mode):
    model = CardAssociativeMemory(mode)
    state = model.init_state(1)
    state["S"].requires_grad_()
    new, _ = model.write(state, torch.tensor([-1]), torch.tensor([-1]), torch.tensor([False]))
    new["S"].sum().backward()
    torch.testing.assert_close(state["S"].grad, torch.ones_like(state["S"]))
    for parameter in model.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("mode", ("delta", "gated_delta"))
def test_hand_computed_delta_updates_and_alpha_is_only_gated(mode):
    model, state = configured(mode)
    key = torch.zeros(32, dtype=torch.float64)
    key[:2] = torch.tensor([.6, .8], dtype=torch.float64)
    prior = state["S"][0] * (.8 if mode == "gated_delta" else 1.)
    residual = model.value_embedding.weight[0] - prior.T @ key
    expected = prior + .3 * torch.outer(key, residual)
    actual, diag = model.write(state, *event())
    torch.testing.assert_close(actual["S"][0], expected, atol=2e-16, rtol=1e-14)
    torch.testing.assert_close(diag["gain_along_key"], torch.tensor([.3], dtype=torch.float64))
    torch.testing.assert_close(diag["residual_power"], residual.square().mean()[None])
    assert diag["added_trace"].item() == 0 and actual["P"] is None
    assert diag["gain_available"].item()


@pytest.mark.parametrize("mode", KALMAN_MODES)
def test_hand_computed_kalman_diagonal_reverse_kl_and_inflation(mode):
    model, state = configured(mode)
    key = torch.zeros(32, dtype=torch.float64)
    key[:2] = torch.tensor([.6, .8], dtype=torch.float64)
    prior = .8 * state["S"][0]
    residual = model.value_embedding.weight[0] - prior.T @ key
    p = .8 ** 2 * state["P"][0] + .03
    addition = torch.zeros_like(p)
    if mode == "innovation_local":
        addition = .4 * residual.square().mean() * key.square()
    elif mode == "innovation_matched":
        addition.fill_(.4 * residual.square().mean().item() * key.pow(4).sum().item() / key.square().sum().item())
    p = p + addition
    gain = p * key / (.2 + (p * key.square()).sum())
    expected_matrix = prior + torch.outer(gain, residual)
    expected_variance = torch.stack([1 / (1 / pi + ki * ki / .2) for pi, ki in zip(p, key, strict=True)])
    new, diag = model.write(state, *event())
    torch.testing.assert_close(new["S"][0], expected_matrix, atol=2e-16, rtol=1e-14)
    torch.testing.assert_close(new["P"][0], expected_variance, atol=3e-16, rtol=1e-14)
    torch.testing.assert_close(diag["added_trace"], addition.sum()[None], atol=3e-15, rtol=1e-14)
    torch.testing.assert_close(diag["key_kurtosis"], key.pow(4).sum()[None])
    assert (new["P"] > 0).all()


def test_local_and_matched_have_equal_current_key_gain_but_different_p_and_cross_key_read():
    local, state = configured("innovation_local")
    matched = CardAssociativeMemory("innovation_matched").double()
    copy_shared_initialization(local, matched)
    a, da = local.write(state, *event())
    b, db = matched.write(state, *event())
    torch.testing.assert_close(da["gain_along_key"], db["gain_along_key"], atol=2e-16, rtol=1e-14)
    torch.testing.assert_close(local.predict(a, torch.tensor([[0]])), matched.predict(b, torch.tensor([[0]])),
                               atol=2e-16, rtol=1e-14)
    assert not torch.allclose(a["P"], b["P"])
    assert not torch.allclose(local.predict(a, torch.tensor([[1]])), matched.predict(b, torch.tensor([[1]])))
    assert db["added_trace"].item() > da["added_trace"].item()
    # Once their input states differ, equality is no longer promised on a later event.
    _, later_a = local.write(a, torch.tensor([1]), torch.tensor([0]), torch.tensor([True]))
    _, later_b = matched.write(b, torch.tensor([1]), torch.tensor([0]), torch.tensor([True]))
    assert not torch.allclose(later_a["gain_along_key"], later_b["gain_along_key"])


def test_uniform_keys_make_local_and_matched_update_coincide():
    local, state = configured("innovation_local")
    with torch.no_grad():
        local.key_embedding.weight[0].fill_(2.)
    matched = CardAssociativeMemory("innovation_matched").double()
    copy_shared_initialization(local, matched)
    a, da = local.write(state, *event())
    b, db = matched.write(state, *event())
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=3e-16, rtol=1e-14)
    torch.testing.assert_close(da["added_trace"], db["added_trace"], atol=3e-16, rtol=1e-14)
    torch.testing.assert_close(da["key_kurtosis"], torch.tensor([1 / 32], dtype=torch.float64))


def test_gru_uses_exact_cell_reshape_and_same_query_decoder():
    model, state = configured("gru")
    key = F.normalize(model.key_embedding(torch.tensor([0])), dim=-1)
    value = model.value_embedding(torch.tensor([0]))
    expected = model.gru(torch.cat((key, value), dim=-1), state["S"].reshape(1, 512)).reshape(1, 32, 16)
    actual, diag = model.write(state, *event())
    torch.testing.assert_close(actual["S"], expected, rtol=0, atol=0)
    assert actual["P"] is None and not diag["gain_available"].item()
    assert diag["gain_along_key"].item() == diag["added_trace"].item() == 0
    assert not hasattr(model, "gate") and not hasattr(model, "noise")


@pytest.mark.parametrize("mode", MODES)
def test_every_active_parameter_tensor_receives_finite_nonzero_gradient(mode):
    model = CardAssociativeMemory(mode)
    state = model.init_state(2)
    for t in range(3):
        state, _ = model.write(state, torch.tensor([t, t + 1]), torch.tensor([t + 2, t + 4]),
                               torch.tensor([True, True]))
    logits = model.predict(state, torch.tensor([[0, 1, 4], [1, 2, 5]]))
    F.cross_entropy(logits.reshape(-1, 13), torch.tensor([2, 3, 1, 4, 5, 2])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0, name
    if mode == "delta":
        assert torch.count_nonzero(model.gate.weight.grad[0]) == 0 and model.gate.bias.grad[0] == 0
    elif mode in KALMAN_MODES:
        assert torch.count_nonzero(model.gate.weight.grad[1]) == 0 and model.gate.bias.grad[1] == 0


@pytest.mark.parametrize("mode,registered,used,state_scalars", (
    ("delta", 2191, 2142, 512), ("gated_delta", 2191, 2191, 512),
    ("kalman", 2289, 2240, 544), ("innovation_local", 2290, 2241, 544),
    ("innovation_matched", 2290, 2241, 544), ("gru", 865325, 865325, 512),
))
def test_parameter_allocation_and_state_size_are_explicit_not_matched(mode, registered, used, state_scalars):
    model = CardAssociativeMemory(mode)
    assert model.parameter_counts() == {"registered_parameters": registered, "used_scalar_parameters": used,
                                        "state_scalars_per_case": state_scalars,
                                        "state_bytes_per_case": state_scalars * 4}
    assert sum(p.numel() for p in model.parameters()) == registered
    assert hasattr(model, "gru") == (mode == "gru")
    assert hasattr(model, "noise") == (mode in KALMAN_MODES)
    assert hasattr(model, "innovation_scale") == (mode in INNOVATION_MODES)


@pytest.mark.parametrize("target_mode", MODES)
def test_explicit_shared_copy_is_exact_nonaliasing_and_consumes_no_rng(target_mode):
    source = CardAssociativeMemory("innovation_local")
    target = CardAssociativeMemory(target_mode)
    rng = torch.get_rng_state().clone()
    copied = copy_shared_initialization(source, target)
    assert set(copied) == source.state_dict().keys() & target.state_dict().keys()
    assert "key_embedding.weight" in copied and "decoder.weight" in copied
    for name in copied:
        torch.testing.assert_close(source.state_dict()[name], target.state_dict()[name], rtol=0, atol=0)
        assert source.state_dict()[name].data_ptr() != target.state_dict()[name].data_ptr()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("delta", "kalman"))
def test_state_copy_and_detach_keep_storage_and_episode_graphs_separate(mode):
    model = CardAssociativeMemory(mode)
    state = model.init_state(1)
    state["S"].requires_grad_()
    cloned, detached = clone_state(state), detach_state(state)
    cloned["S"].sum().backward()
    torch.testing.assert_close(state["S"].grad, torch.ones_like(state["S"]))
    assert not detached["S"].requires_grad
    detached["S"][0, 0, 0] = 4
    assert state["S"][0, 0, 0] == cloned["S"][0, 0, 0] == 0
    if mode == "kalman":
        detached["P"].zero_()
        assert (state["P"] == 1).all()


@pytest.mark.parametrize("field,value", (("S", torch.zeros(1, 16, 32)), ("S", torch.full((1, 32, 16), float("nan"))),
                                       ("P", None), ("P", torch.zeros(1, 32)), ("P", torch.full((1, 32), -1.))))
def test_invalid_state_rejected(field, value):
    model = CardAssociativeMemory("kalman")
    state = model.init_state(1)
    state[field] = value
    with pytest.raises(ValueError):
        model.predict(state, torch.tensor([[0]]))


@pytest.mark.parametrize("pos,rank,valid", ((torch.tensor([52]), torch.tensor([0]), torch.tensor([True])),
                                         (torch.tensor([0]), torch.tensor([13]), torch.tensor([True])),
                                         (torch.tensor([0.]), torch.tensor([0]), torch.tensor([True])),
                                         (torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))))
def test_invalid_write_indices_or_mask_rejected(pos, rank, valid):
    model = CardAssociativeMemory("delta")
    state = model.init_state(1)
    with pytest.raises(ValueError):
        model.write(state, pos, rank, valid)
    assert torch.count_nonzero(state["S"]) == 0


def test_zero_key_and_foreign_query_rejected_instead_of_silent_degenerate_gain():
    model = CardAssociativeMemory("innovation_matched")
    state = model.init_state(1)
    with pytest.raises(ValueError):
        model.predict(state, torch.tensor([[52]]))
    with torch.no_grad():
        model.key_embedding.weight[0].zero_()
    with pytest.raises(ValueError, match="nonzero"):
        model.write(state, *event())
    assert torch.count_nonzero(state["S"]) == 0


@pytest.mark.parametrize("mode", MODES)
def test_configuration_binds_exact_mode_equations_dimensions_and_is_json_safe_copy(mode):
    model = CardAssociativeMemory(mode)
    config = model.configuration()
    assert config["mode"] == mode and config["model_class"] == "CardAssociativeMemory"
    assert (config["positions"], config["ranks"], config["key_dim"], config["value_dim"]) == (52, 13, 32, 16)
    assert (config["initial_P"] is None) == (mode not in KALMAN_MODES)
    assert (config["kalman_update"] is None) == (mode not in KALMAN_MODES)
    assert config["parameter_counts"] == model.parameter_counts()
    assert config["innovation_residual_gradient"] == ("fully_differentiable" if mode in INNOVATION_MODES else None)
    assert "calibration" in config["posterior_claim"]
    json.dumps(config, allow_nan=False)
    config["parameter_counts"]["state_scalars_per_case"] = -1
    config["mode"] = "tampered"
    assert model.configuration()["mode"] == mode
    assert model.configuration()["parameter_counts"]["state_scalars_per_case"] > 0


def test_identical_innovation_weight_schemas_have_distinct_configuration_binding():
    local = CardAssociativeMemory("innovation_local")
    matched = CardAssociativeMemory("innovation_matched")
    copy_shared_initialization(local, matched)
    assert local.state_dict().keys() == matched.state_dict().keys()
    for name in local.state_dict():
        torch.testing.assert_close(local.state_dict()[name], matched.state_dict()[name], rtol=0, atol=0)
    assert local.configuration()["update_rule"] != matched.configuration()["update_rule"]
    assert local.configuration()["innovation_residual_gradient"] == matched.configuration()["innovation_residual_gradient"]
