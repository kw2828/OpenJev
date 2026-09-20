"""Tiny synthetic representation-adapter checks; no corpus/encoder/fit calls."""
import pytest
import torch

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2
from openjev.research.dialogue_joint_memory import METHODS, VERSION, DialogueJointMemory


@pytest.fixture(autouse=True)
def engineering_scope():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def model(method, dtype=torch.float32, cls=DialogueJointMemory):
    return cls(method, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3).to(dtype)


def inputs(steps=6, dtype=torch.float32, joint=True):
    shape = (2, steps, 2, 4, 6) if joint else (2, steps, 6)
    turns = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape).sin()
    valid = torch.ones(2, steps, dtype=torch.bool)
    valid[0, 2::3] = False
    valid[1, 0::4] = False
    query = torch.arange(24, dtype=dtype).reshape(2, 2, 6).cos()
    candidates = torch.arange(96, dtype=dtype).reshape(2, 2, 4, 6).sin()
    mask = torch.tensor([[[True, True, False, True], [True, True, True, True]]] * 2)
    lexical = (torch.arange(2 * steps * 2 * 4 * 10).reshape(2, steps, 2, 4, 10) % 3 == 0).to(dtype)
    return [turns, valid, query, candidates, mask, lexical]


def active_loss(output, valid):
    loss = -output[..., 1][valid].mean()
    assert torch.isfinite(loss)
    return loss


@pytest.mark.parametrize("method", METHODS)
def test_exact_parameter_schema_initial_tensors_and_rng_draws(method):
    rng = torch.random.get_rng_state()
    old = DialogueCopyMemoryV2(method)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    new = DialogueJointMemory(method)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert set(old.state_dict()) == set(new.state_dict())
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    assert sum(p.numel() for p in new.parameters()) == 99458
    config = new.configuration()
    assert config["implementation_version"] == VERSION
    assert config["normalized_parent_version"] == "dialogue-copy-memory-v2-normalized"
    assert config["parameters"] == old.configuration()["parameters"]
    config["supported_methods"].clear()
    assert new.configuration()["supported_methods"] == list(METHODS)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_path_exact_v2_output_input_and_parameter_gradients(method, dtype):
    old, new = model(method, dtype, DialogueCopyMemoryV2), model(method, dtype)
    new.load_state_dict(old.state_dict())
    a = inputs(dtype=dtype, joint=False)
    b = [x.clone() for x in a]
    for i in (0, 2, 3, 5):
        a[i].requires_grad_()
        b[i].requires_grad_()
    expected, actual = old(*a), new(*b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    active_loss(expected, a[1]).backward()
    active_loss(actual, b[1]).backward()
    for i in (0, 2, 3, 5):
        torch.testing.assert_close(a[i].grad, b[i].grad, rtol=0, atol=0)
    for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters(), strict=True):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_broadcast_joint_evidence_matches_v2_values_and_gradients(method, dtype):
    old, new = model(method, dtype, DialogueCopyMemoryV2), model(method, dtype)
    new.load_state_dict(old.state_dict())
    a = inputs(dtype=dtype, joint=False)
    b = [x.clone() for x in a]
    for i in (0, 2, 3, 5):
        a[i].requires_grad_()
        b[i].requires_grad_()
    common_turn = b[0]
    b[0] = common_turn[:, :, None, None].expand(-1, -1, 2, 4, -1)
    expected, actual = old(*a), new(*b)
    tol = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    active_loss(expected, a[1]).backward()
    active_loss(actual, b[1]).backward()
    torch.testing.assert_close(a[0].grad, common_turn.grad, rtol=tol, atol=tol)
    for i in (2, 3, 5):
        torch.testing.assert_close(a[i].grad, b[i].grad, rtol=tol, atol=tol)
    for p, q in zip(old.parameters(), new.parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("method", METHODS)
def test_joint_streaming_exact_forward_parity_and_state_input_ownership(method):
    net, args = model(method), inputs()
    before = [x.clone() for x in args]
    state = net.initial(args[4])
    rows = []
    for t in range(args[0].shape[1]):
        old = state
        snapshot = {key: value.clone() for key, value in state.items()}
        state, diagnostic = net.step(state, args[0][:, t], args[1][:, t], *args[2:5], args[5][:, t])
        assert all(torch.equal(old[k], snapshot[k]) for k in old)
        assert torch.equal(state["log_b"][~args[1][:, t]], old["log_b"][~args[1][:, t]])
        assert state["none_index"].data_ptr() != old["none_index"].data_ptr()
        assert set(diagnostic) == ({"write_log_probs", "departure_probabilities", "departure_mass", "valid"}
                                   if method == "scalar" else set())
        rows.append(state["log_b"])
    torch.testing.assert_close(net(*args), torch.stack(rows, 1), rtol=0, atol=0)
    assert all(torch.equal(x, y) for x, y in zip(args, before, strict=True))


@pytest.mark.parametrize("method", METHODS)
def test_joint_future_and_padding_causality(method):
    net, args = model(method), inputs()
    original = net(*args)
    changed = [x.clone() for x in args]
    changed[0][:, 3:] += 77
    changed[5][:, 3:] -= 17
    torch.testing.assert_close(net(*changed)[:, :3], original[:, :3], rtol=0, atol=0)
    changed = [x.clone() for x in args]
    changed[0][~args[1]] = 123
    changed[5][~args[1]] = -987
    changed[0].masked_fill_(~args[4][:, None, :, :, None], 666)
    torch.testing.assert_close(net(*changed), original, rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_candidate_query_permutations_and_unrelated_query_independence(method):
    net, args = model(method, torch.float64), inputs(dtype=torch.float64)
    original = net(*args)
    qp, cp = [1, 0], [3, 2, 0, 1]
    actual = net(args[0][:, :, qp][:, :, :, cp], args[1], args[2][:, qp], args[3][:, qp][:, :, cp],
                 args[4][:, qp][:, :, cp], args[5][:, :, qp][:, :, :, cp], torch.full((2, 2), 2, dtype=torch.int64))
    torch.testing.assert_close(actual, original[:, :, qp][:, :, :, cp], rtol=2e-12, atol=2e-12)
    alone = net(args[0][:, :, :1], args[1], args[2][:, :1], args[3][:, :1], args[4][:, :1], args[5][:, :, :1])
    torch.testing.assert_close(alone, original[:, :, :1], rtol=2e-12, atol=2e-12)
    changed = [x.clone() for x in args]
    changed[0][:, :, 1] += 100
    changed[2][:, 1] -= 99
    changed[3][:, 1] += 77
    changed[5][:, :, 1] -= 123
    torch.testing.assert_close(net(*changed)[:, :, :1], original[:, :, :1], rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_joint_finite_gradients_and_zero_masked_or_padded_input_gradients(method):
    net, args = model(method), inputs(steps=12)
    for i in (0, 2, 3, 5):
        args[i].requires_grad_()
    output = net(*args)
    active_loss(output, args[1]).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.turn_projection.weight.grad.abs().sum() > 0
    assert net.head.weight.grad.abs().sum() > 0
    for i in (0, 2, 3, 5):
        assert args[i].grad is not None and torch.isfinite(args[i].grad).all()
    assert args[0].grad[~args[1]].abs().sum() == 0
    assert args[5].grad[~args[1]].abs().sum() == 0
    assert args[0].grad.masked_select(~args[4][:, None, :, :, None]).abs().sum() == 0
    assert args[3].grad[~args[4]].abs().sum() == 0
    if method == "scalar":
        assert args[0].grad[0, 0].abs().sum() > 0


def test_long_float32_scalar_normalization_feature_priors_and_released_mass():
    net, args = model("scalar"), inputs(steps=256)
    features = []
    handle = net.feature.register_forward_pre_hook(lambda _, values: features.append(values[0][..., -2].detach().clone()))
    state = net.initial(args[4])
    state["log_b"][..., 0] += 2**-19
    initial = state["log_b"].clone()
    try:
        with torch.inference_mode():
            for t in range(args[0].shape[1]):
                previous = state["log_b"]
                state, diagnostic = net.step(state, args[0][:, t], args[1][:, t], *args[2:5], args[5][:, t])
                mass = diagnostic["departure_mass"][args[1][:, t]]
                assert torch.isfinite(mass).all() and (mass >= 0).all() and (mass <= 1 + 2e-6).all()
                actual = state["log_b"].exp()[args[1][:, t]].double().sum(-1)
                assert actual.numel() == 0 or (actual - 1).abs().max() <= 2e-6
                assert torch.equal(state["log_b"][~args[1][:, t]], previous[~args[1][:, t]])
                assert (features[-1].double().sum(-1) - 1).abs().max() <= 2e-6
    finally:
        handle.remove()
    assert initial[1, 0, 0] == 2**-19
    assert torch.isneginf(state["log_b"][~args[4]]).all()


@pytest.mark.parametrize("method", METHODS)
def test_single_valid_candidate_zero_log_probability_and_finite_backward(method):
    net, args = model(method), inputs()
    args[4].fill_(False)
    args[4][..., 0] = True
    result = net(*args)
    assert torch.equal(result[..., 0], torch.zeros_like(result[..., 0]))
    assert torch.isneginf(result[..., 1:]).all()
    (-result[..., 0].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("mutation", ["rank", "query_axis", "candidate_axis", "dtype", "nan_masked", "inf_padding", "bad_mask", "empty"])
def test_malformed_joint_inputs_rejected_without_mutation(mutation):
    net, args = model("scalar"), inputs()
    if mutation == "rank":
        args[0] = args[0][:, :, 0]
    elif mutation == "query_axis":
        args[0] = args[0][:, :, :1]
    elif mutation == "candidate_axis":
        args[0] = args[0][:, :, :, :3]
    elif mutation == "dtype":
        args[0] = args[0].double()
    elif mutation == "nan_masked":
        args[0][0, 0, 0, 2, 0] = float("nan")
    elif mutation == "inf_padding":
        args[0][0, 2, 0, 0, 0] = float("inf")
    elif mutation == "bad_mask":
        args[1] = args[1].float()
    else:
        args[0] = args[0][:, :0]
    with pytest.raises(ValueError):
        net(*args)


@pytest.mark.parametrize("method", ["selective", "candidate_gru", "missing"])
def test_only_declared_representation_control_methods(method):
    with pytest.raises(ValueError, match="readout or scalar"):
        model(method)


def test_joint_step_validates_state_and_turn_shape():
    net, args = model("scalar"), inputs()
    state = net.initial(args[4])
    with pytest.raises(ValueError, match="joint turn"):
        net.step(state, args[0][:, 0, :1], args[1][:, 0], *args[2:5], args[5][:, 0])
    state["log_b"][..., 0] = .1
    with pytest.raises(ValueError, match="normalized"):
        net.step(state, args[0][:, 0], args[1][:, 0], *args[2:5], args[5][:, 0])
