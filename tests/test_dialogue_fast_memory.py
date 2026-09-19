"""Tiny synthetic engineering tests; no encoder, corpus or scientific fits."""

import inspect

import numpy as np
import pytest
import torch

from openjev.research.dialogue_fast_memory import METHODS, DialogueMemory, state_bytes


@pytest.fixture(autouse=True)
def engineering_scope():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(410)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def model(method, **kwargs):
    return DialogueMemory(method, input_dim=6, width=3, gru_width=5, **kwargs).double()


def inputs():
    turns = torch.randn(2, 5, 6, dtype=torch.float64)
    valid = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])
    query = torch.randn(2, 3, 6, dtype=torch.float64)
    candidates = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    times = torch.tensor([[0, 3, 4], [0, 2, 3]])
    mask = torch.tensor([[[True, True, False, True]] * 3] * 2)
    return turns, valid, query, candidates, times, mask


@pytest.mark.parametrize("method", METHODS)
def test_forward_matches_explicit_stream_and_read_without_mutating_inputs(method):
    net = model(method)
    args = inputs()
    before = tuple(a.clone() for a in args)
    actual = net(*args)
    turns, valid, query, candidates, times, mask = args
    state = net.initial_state(2)
    expected = torch.zeros_like(actual)
    for t in range(5):
        saved = {k: v.clone() for k, v in state.items()}
        previous = state
        state, _ = net.update(turns[:, t], valid[:, t], state)
        for k in previous:
            torch.testing.assert_close(previous[k], saved[k], rtol=0, atol=0)
        read_before = {k: v.clone() for k, v in state.items()}
        logits = net.read(state, query, candidates, mask)
        for k in state:
            torch.testing.assert_close(state[k], read_before[k], rtol=0, atol=0)
        rows, slots = (times == t).nonzero(as_tuple=True)
        expected[rows, slots] = logits[rows, slots]
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    assert torch.isneginf(actual[~mask]).all()
    for a, b in zip(args, before, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_future_turns_and_finite_padding_cannot_change_prior_reads(method):
    net = model(method)
    turns, valid, query, candidates, times, mask = inputs()
    baseline = net(turns, valid, query, candidates, times, mask)
    padded = turns.clone()
    padded[~valid] = 123456.
    torch.testing.assert_close(net(padded, valid, query, candidates, times, mask), baseline, rtol=0, atol=0)
    future = turns.clone()
    future[:, 1:] += 9.
    changed = net(future, valid, query, candidates, times, mask)
    torch.testing.assert_close(changed[:, 0], baseline[:, 0], rtol=0, atol=0)
    prefix = net(turns[:, :1], valid[:, :1], query[:, :1], candidates[:, :1],
                 torch.zeros(2, 1, dtype=torch.int64), mask[:, :1])
    torch.testing.assert_close(prefix[:, 0], baseline[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_query_and_candidate_permutations_are_equivariant(method):
    net = model(method)
    turns, valid, query, candidates, times, mask = inputs()
    expected = net(turns, valid, query, candidates, times, mask)
    qp, cp = [2, 0, 1], [3, 1, 0, 2]
    actual = net(turns, valid, query[:, qp], candidates[:, qp][:, :, cp],
                 times[:, qp], mask[:, qp][:, :, cp])
    torch.testing.assert_close(actual, expected[:, qp][:, :, cp], rtol=1e-13, atol=1e-13)
    assert list(inspect.signature(net.update).parameters) == ["turn", "valid", "state"]


@pytest.mark.parametrize("method", METHODS)
def test_parameter_and_input_gradients_remain_finite_and_connected(method):
    net = model(method)
    turns, valid, query, candidates, times, mask = inputs()
    turns.requires_grad_()
    query.requires_grad_()
    candidates.requires_grad_()
    logits = net(turns, valid, query, candidates, times, mask)
    loss = -logits.log_softmax(-1)[..., 0].mean()
    loss.backward()
    for tensor in (turns, query, candidates):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert turns.grad[~valid].abs().sum() == 0
    for parameter in net.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    assert net.turn_features[0].weight.grad.abs().sum() > 0
    assert net.query.weight.grad.abs().sum() > 0
    if method in ("gated_delta", "kalman", "innovation_kalman"):
        assert net.gates.weight.grad.abs().sum() > 0


def fixed_projections(net, key, value, alpha=.8, beta=.4):
    with torch.no_grad():
        net.key.weight.zero_()
        net.key.bias.copy_(torch.tensor(key))
        net.value.weight.zero_()
        net.value.bias.copy_(torch.tensor(value))
        net.gates.weight.zero_()
        net.gates.bias.copy_(torch.logit(torch.tensor([alpha, beta], dtype=torch.float64)))


def numpy_kalman(S, P, k, v, alpha, beta, eta):
    k = k / np.linalg.norm(k)
    r = (1 - beta) / beta + .01
    sp = alpha * S
    pp = alpha**2 * P + .01 * np.eye(len(k))
    error = v - sp.T @ k
    denominator = r + k @ pp @ k
    inflation = eta * np.clip(np.mean(error**2) / denominator - 1, 0, 10)
    pp += inflation * np.outer(k, k)
    gain = pp @ k / (r + k @ pp @ k)
    A = np.eye(len(k)) - np.outer(gain, k)
    return sp + np.outer(gain, error), A @ pp @ A.T + r * np.outer(gain, gain), inflation


@pytest.mark.parametrize("method,eta", [("kalman", 0.), ("innovation_kalman", .1)])
@pytest.mark.parametrize("width", [1, 3])
def test_dense_joseph_matches_independent_numpy_including_inflation(method, eta, width):
    net = DialogueMemory(method, input_dim=6, width=width).double()
    key = np.arange(1., width + 1)
    value = np.arange(5., 5 + width)
    fixed_projections(net, key, value)
    state = net.initial_state(1)
    state["S"] = torch.arange(width**2, dtype=torch.float64).reshape(1, width, width) / 10
    state["P"] = (torch.eye(width, dtype=torch.float64) + .2).unsqueeze(0)
    expected_s, expected_p, amount = numpy_kalman(state["S"][0].numpy(), state["P"][0].numpy(),
                                                key, value, .8, .4, eta)
    actual, diagnostic = net.update(torch.zeros(1, 6, dtype=torch.float64), torch.tensor([True]), state)
    np.testing.assert_allclose(actual["S"].detach().numpy()[0], expected_s, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual["P"].detach().numpy()[0], expected_p, rtol=2e-14, atol=2e-14)
    assert diagnostic["inflation"].item() == pytest.approx(amount)
    assert torch.linalg.eigvalsh(actual["P"]).min() > 0


def test_innovation_eta_zero_is_exact_kalman_with_identical_parameters():
    base, extra = model("kalman"), model("innovation_kalman", innovation_eta=0.)
    extra.load_state_dict(base.state_dict(), strict=True)
    a, b = base.initial_state(2), extra.initial_state(2)
    turns, valid, *_ = inputs()
    for t in range(5):
        a, _ = base.update(turns[:, t], valid[:, t], a)
        b, _ = extra.update(turns[:, t], valid[:, t], b)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    args = inputs()
    torch.testing.assert_close(base(*args), extra(*args), rtol=0, atol=0)


def test_kalman_family_and_delta_share_identical_parameter_schema():
    nets = [model(name) for name in ("gated_delta", "kalman", "innovation_kalman")]
    for net in nets[1:]:
        net.load_state_dict(nets[0].state_dict(), strict=True)
        assert {k: tuple(v.shape) for k, v in net.state_dict().items()} == {
            k: tuple(v.shape) for k, v in nets[0].state_dict().items()}
    torch.testing.assert_close(nets[0].gates.bias, torch.tensor([3., 0.], dtype=torch.float64))


def test_delta_update_and_invalid_rows_are_hand_computable():
    net = model("gated_delta")
    fixed_projections(net, [1., 0., 0.], [2., 3., 4.], alpha=.8, beta=.4)
    state = net.initial_state(2)
    state["S"] += 1
    actual, _ = net.update(torch.zeros(2, 6, dtype=torch.float64), torch.tensor([True, False]), state)
    expected = state["S"][0] * .8
    expected[0] += .4 * (torch.tensor([2., 3., 4.], dtype=torch.float64) - .8)
    torch.testing.assert_close(actual["S"][0], expected)
    torch.testing.assert_close(actual["S"][1], state["S"][1], rtol=0, atol=0)


def test_current_forgets_prior_and_attention_retains_every_valid_value():
    current, attention = model("current"), model("attention")
    for net in (current, attention):
        state = net.initial_state(1)
        state, _ = net.update(torch.ones(1, 6, dtype=torch.float64), torch.tensor([True]), state)
        state, _ = net.update(torch.zeros(1, 6, dtype=torch.float64), torch.tensor([False]), state)
        state, _ = net.update(torch.full((1, 6), 2., dtype=torch.float64), torch.tensor([True]), state)
        if net.method == "current":
            direct, _ = net.update(torch.full((1, 6), 2., dtype=torch.float64), torch.tensor([True]),
                                   net.initial_state(1))
            torch.testing.assert_close(state["value"], direct["value"], rtol=0, atol=0)
        else:
            assert state["keys"].shape == (1, 3, 3)
            assert state["valid"].tolist() == [[True, False, True]]
            # Zero read keys leave only recency over the two valid values.
            with torch.no_grad():
                net.query.weight.zero_()
                net.query.bias.zero_()
            captured = []
            handle = net.readout[0].register_forward_pre_hook(
                lambda module, args, captured=captured: captured.append(args[0]))
            net.read(state, torch.zeros(1, 1, 6, dtype=torch.float64),
                     torch.zeros(1, 1, 2, 6, dtype=torch.float64), torch.ones(1, 1, 2, dtype=torch.bool))
            handle.remove()
            decay = torch.nn.functional.softplus(net.log_decay)
            weights = torch.stack((-decay, decay * 0)).softmax(0)
            expected = weights[0] * state["values"][:, 0] + weights[1] * state["values"][:, 2]
            torch.testing.assert_close(captured[0][..., 3:6], expected[:, None, None].expand(1, 1, 2, 3))


def test_inflation_feedback_has_value_gradient_and_does_not_detach_covariance():
    net = model("innovation_kalman")
    fixed_projections(net, [1., 0., 0.], [2.5, 2.5, 2.5])
    state, diagnostics = net.update(torch.zeros(1, 6, dtype=torch.float64), torch.tensor([True]),
                                    net.initial_state(1))
    assert 0 < diagnostics["inflation"].item() < 1.
    state["P"].sum().backward()
    assert net.value.bias.grad is not None and net.value.bias.grad.abs().sum() > 0


def test_attention_matches_direct_unnormalized_scaled_softmax():
    net = model("attention")
    turns, valid, query, candidates, _, mask = inputs()
    state = net.initial_state(2)
    for t in range(turns.shape[1]):
        state, _ = net.update(turns[:, t], valid[:, t], state)
    public_turns = torch.where(valid[..., None], turns, torch.zeros_like(turns))
    raw_keys = net.key(net.turn_features(public_turns))
    raw_values = net.value(net.turn_features(public_turns))
    torch.testing.assert_close(state["keys"], raw_keys, rtol=1e-14, atol=1e-14)
    expanded = query[:, :, None].expand_as(candidates)
    raw_queries = net.query(torch.cat((expanded, candidates), dim=-1))
    scores = (raw_queries.unsqueeze(-2) * raw_keys[:, None, None]).sum(-1) / np.sqrt(3)
    ages = torch.tensor([[3, 2, 2, 1, 0], [2, 2, 1, 0, 0]], dtype=torch.float64)
    scores -= torch.nn.functional.softplus(net.log_decay) * ages[:, None, None]
    weights = scores.masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    expected = (weights[..., None] * raw_values[:, None, None]).sum(-2)
    captured = []
    handle = net.readout[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        net.read(state, query, candidates, mask)
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0][..., 3:6], expected, rtol=1e-14, atol=1e-14)
    assert not torch.allclose(raw_keys.norm(dim=-1), torch.ones_like(valid, dtype=torch.float64))


def test_attention_recency_uses_real_turn_age_and_has_trainable_order_signal():
    net = model("attention")
    with torch.no_grad():
        net.query.weight.zero_()
        net.query.bias.zero_()
    a, b = torch.ones(1, 6, dtype=torch.float64), torch.full((1, 6), 2., dtype=torch.float64)
    states = []
    for sequence in ((a, b), (a, None, None, b, None), (b, a)):
        state = net.initial_state(1)
        for turn in sequence:
            state, _ = net.update(a if turn is None else turn, torch.tensor([turn is not None]), state)
        states.append(state)
    query, candidates = torch.zeros(1, 1, 6, dtype=torch.float64), torch.zeros(1, 1, 2, 6, dtype=torch.float64)
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    reads = [net.read(s, query, candidates, mask) for s in states]
    torch.testing.assert_close(reads[0], reads[1], rtol=1e-14, atol=1e-14)
    assert not torch.allclose(reads[0], reads[2], rtol=1e-10, atol=1e-10)
    assert torch.nn.functional.softplus(net.log_decay).item() == pytest.approx(.1)
    reads[0].sum().backward()
    assert net.log_decay.grad is not None and net.log_decay.grad.abs() > 0


def test_attention_empty_and_all_invalid_prefix_recall_zero():
    net = model("attention")
    _, _, query, candidates, _, mask = inputs()
    state = net.initial_state(2)
    empty = net.read(state, query, candidates, mask)
    state, _ = net.update(torch.ones(2, 6, dtype=torch.float64), torch.tensor([False, False]), state)
    torch.testing.assert_close(net.read(state, query, candidates, mask), empty, rtol=0, atol=0)


def test_default_state_payloads_are_not_claimed_equal():
    assert state_bytes(DialogueMemory("kalman").initial_state(1)) == 512 * 4
    assert state_bytes(DialogueMemory("gated_delta").initial_state(1)) == 256 * 4
    assert state_bytes(DialogueMemory("gru").initial_state(1)) == 128 * 4
    net = DialogueMemory("attention")
    state, _ = net.update(torch.zeros(1, 384), torch.tensor([True]), net.initial_state(1))
    assert state_bytes(state) == 32 * 4 + 1


@pytest.mark.parametrize("mutation", ["future_time", "padding_time", "float_time", "bad_mask", "empty_candidates",
                                      "nan_turn", "bad_dtype", "query_dim", "candidate_dim"])
def test_bad_forward_inputs_fail(mutation):
    net = model("kalman")
    args = list(inputs())
    if mutation == "future_time":
        args[4][0, 0] = 5
    elif mutation == "padding_time":
        args[4][0, 0] = 2
    elif mutation == "float_time":
        args[4] = args[4].double()
    elif mutation == "bad_mask":
        args[1] = args[1].long()
    elif mutation == "empty_candidates":
        args[5][0, 0] = False
    elif mutation == "nan_turn":
        args[0][0, 0, 0] = float("nan")
    elif mutation == "bad_dtype":
        args[0] = args[0].float()
    elif mutation == "query_dim":
        args[2] = args[2][..., :5]
    else:
        args[3] = args[3][..., :5]
    with pytest.raises(ValueError):
        net(*args)


def test_invalid_state_and_configuration_fail_without_mutation():
    with pytest.raises(ValueError):
        DialogueMemory("unknown")
    with pytest.raises(ValueError):
        DialogueMemory("kalman", width=0)
    with pytest.raises(ValueError):
        model("kalman", innovation_eta=-1)
    net = model("kalman")
    state = net.initial_state(1)
    state["S"] = state["S"][:, :2]
    before = state["P"].clone()
    with pytest.raises(ValueError):
        net.update(torch.zeros(1, 6, dtype=torch.float64), torch.tensor([True]), state)
    torch.testing.assert_close(state["P"], before, rtol=0, atol=0)
