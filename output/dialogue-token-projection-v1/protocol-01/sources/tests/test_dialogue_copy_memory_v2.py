"""Synthetic float32 recurrence checks; no corpus, old-fit replay or optimizer."""
import copy

import numpy as np
import pytest
import torch

from openjev.research.dialogue_copy_memory import METHODS, DialogueCopyMemory, transition
from openjev.research.dialogue_copy_memory_v2 import (
    VERSION,
    DialogueCopyMemoryV2,
    normalize_log_b,
)


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


def model(method, dtype=torch.float32):
    return DialogueCopyMemoryV2(method, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3).to(dtype)


def inputs(steps=6, dtype=torch.float32):
    turns = torch.arange(2*steps*6, dtype=dtype).reshape(2, steps, 6).sin()
    valid = torch.ones(2, steps, dtype=torch.bool)
    valid[0, 2::3] = False
    valid[1, 0::4] = False
    query = torch.arange(24, dtype=dtype).reshape(2, 2, 6).cos()
    candidates = torch.arange(96, dtype=dtype).reshape(2, 2, 4, 6).sin()
    mask = torch.tensor([[[True, True, False, True], [True, True, True, True]]] * 2)
    lexical = (torch.arange(2*steps*2*4*10).reshape(2, steps, 2, 4, 10) % 3 == 0).to(dtype)
    return [turns, valid, query, candidates, mask, lexical]


def test_scalar_drift_has_predicted_mass_equation_and_projection_stops_amplification():
    # A deliberately tiny admissible float32 normalization error, not data.
    b = torch.tensor([[.7, .3]], dtype=torch.float32).log() + 2**-20
    corrected = b.clone()
    logits = torch.tensor([[.2, -.6]])
    departure = torch.tensor([[-3., -3.]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    for _ in range(18):
        s = b.double().exp().sum().item()
        old, _, _, mass = transition(b, logits, departure, mask, selective=False)
        expected_mass = s*(s-mass.item()) + mass.item()
        assert old.double().exp().sum().item() == pytest.approx(expected_mass, rel=4e-7, abs=4e-7)
        b = old
        corrected, *_ = transition(normalize_log_b(corrected), logits, departure, mask, selective=False)
        corrected = normalize_log_b(corrected)
        assert abs(corrected.exp().sum().item()-1) <= 3e-7
    assert abs(b.exp().sum().item()-1) > .01


@pytest.mark.parametrize("method", METHODS)
def test_float32_long_forward_keeps_every_real_state_normalized(method):
    net = model(method)
    args = inputs(steps=512)
    with torch.inference_mode():
        output = net(*args)
    assert output.dtype == torch.float32
    assert output.logsumexp(-1).abs().max().item() <= 4e-7
    assert (output.exp().sum(-1)-1).abs().max().item() <= 5e-7
    assert torch.isneginf(output.masked_select(~args[4][:, None].expand_as(output))).all()
    assert not torch.isnan(output).any() and not torch.isposinf(output).any()


@pytest.mark.parametrize("method", METHODS)
def test_step_forward_exact_parity_and_no_input_mutation(method):
    net = model(method)
    args = inputs()
    snapshots = [x.clone() for x in args]
    state = net.initial(args[4])
    states = []
    for t in range(args[0].shape[1]):
        before = {k: v.clone() for k, v in state.items()}
        old = state
        state, _ = net.step(state, args[0][:, t], args[1][:, t], *args[2:5], args[5][:, t])
        for k in old:
            assert torch.equal(old[k], before[k])
            if old[k].dtype.is_floating_point:
                assert torch.equal(state[k][~args[1][:, t]], old[k][~args[1][:, t]])
        states.append(state["log_b"])
    torch.testing.assert_close(net(*args), torch.stack(states, 1), rtol=0, atol=0)
    assert all(torch.equal(x, y) for x, y in zip(args, snapshots, strict=True))


@pytest.mark.parametrize("method", METHODS)
def test_feature_prior_normalized_and_padding_is_exact_even_for_small_external_drift(method):
    net = model(method)
    args = inputs()
    state = net.initial(args[4])
    state["log_b"][..., 0] += 2**-19  # Inside old public state-validation tolerance.
    prior = state["log_b"].clone()
    features = []
    handle = net.feature.register_forward_pre_hook(lambda _, arguments: features.append(arguments[0].detach().clone()))
    valid = torch.tensor([True, False])
    try:
        after, _ = net.step(state, args[0][:, 0], valid, *args[2:5], args[5][:, 0])
    finally:
        handle.remove()
    # The second-last feature is own previous probability; padding is computed
    # but never committed. Readout deliberately substitutes the normalized NONE.
    torch.testing.assert_close(features[0][..., -2].sum(-1), torch.ones(2, 2), rtol=0, atol=2e-7)
    assert torch.equal(after["log_b"][1], prior[1])
    assert torch.equal(state["log_b"], prior)
    assert after["log_b"][0].logsumexp(-1).abs().max() <= 2e-7


@pytest.mark.parametrize("method", METHODS)
def test_future_padding_and_unrelated_question_independence(method):
    net = model(method)
    args = inputs(dtype=torch.float64)
    net.double()
    original = net(*args)
    changed = [x.clone() for x in args]
    changed[0][:, 3:] += 99
    changed[5][:, 3:] -= 18
    torch.testing.assert_close(net(*changed)[:, :3], original[:, :3], rtol=0, atol=0)
    changed = [x.clone() for x in args]
    changed[0][~args[1]] = -100
    changed[5][~args[1]] = 17
    torch.testing.assert_close(net(*changed), original, rtol=0, atol=0)
    alone = net(args[0], args[1], args[2][:, :1], args[3][:, :1], args[4][:, :1], args[5][:, :, :1])
    torch.testing.assert_close(alone, original[:, :, :1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_candidate_query_permutation_and_masked_none_identity(method):
    net = model(method, torch.float64)
    args = inputs(dtype=torch.float64)
    expected = net(*args)
    qp, cp = [1, 0], [3, 2, 0, 1]
    actual = net(args[0], args[1], args[2][:, qp], args[3][:, qp][:, :, cp],
                 args[4][:, qp][:, :, cp], args[5][:, :, qp][:, :, :, cp],
                 torch.full((2, 2), 2, dtype=torch.int64))
    torch.testing.assert_close(actual, expected[:, :, qp][:, :, :, cp], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_float32_gradients_survive_normalization_feedback_and_masks(method):
    net = model(method)
    args = inputs(steps=24)
    for index in (0, 2, 3, 5):
        args[index].requires_grad_()
    result = net(*args)
    (-result[:, -1, :, 1].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.head.weight.grad is not None or method == "candidate_gru"
    for index in (0, 2, 3, 5):
        assert args[index].grad is not None and torch.isfinite(args[index].grad).all()
    assert args[0].grad[~args[1]].abs().sum() == 0
    assert args[5].grad[~args[1]].abs().sum() == 0
    if method != "readout":
        assert args[0].grad[0, 0].abs().sum() > 0


@pytest.mark.parametrize("method", ["scalar", "selective"])
@pytest.mark.parametrize("departure", [-1000., 1000.])
def test_copy_write_endpoints_and_gradients(method, departure):
    net = model(method)
    args = inputs()
    state = net.initial(args[4])
    b = torch.tensor([.6, .3, 0., .1])
    state["log_b"] = b.log().expand(2, 2, 4).clone()
    # q1 permits all4, but this zero entry remains legitimate support.
    with torch.no_grad():
        net.head.weight[1].zero_()
        net.head.bias[1].fill_(departure)
    result, diagnostic = net.step(state, args[0][:, 0], torch.ones(2, dtype=torch.bool), *args[2:5], args[5][:, 0])
    expected = normalize_log_b(state["log_b"]) if departure < 0 else diagnostic["write_log_probs"]
    torch.testing.assert_close(result["log_b"].exp(), expected.exp(), rtol=2e-6, atol=2e-7)
    (-result["log_b"][..., 1].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_v2_parameter_schema_pairs_and_configuration_cannot_impersonate_old():
    old = DialogueCopyMemory("scalar", input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3)
    corrected = model("scalar")
    corrected.load_state_dict(old.state_dict(), strict=True)
    assert all(torch.equal(v, corrected.state_dict()[k]) for k, v in old.state_dict().items())
    for method in ("selective", "selective_no_lexical"):
        paired = model(method)
        paired.load_state_dict(corrected.state_dict(), strict=True)
    conf = corrected.configuration()
    assert conf["class"] == "DialogueCopyMemoryV2"
    assert conf["implementation_version"] == VERSION
    assert conf != old.configuration()
    conf["lexical_fields"].clear()
    assert len(corrected.configuration()["lexical_fields"]) == 10


@pytest.mark.parametrize("method", METHODS)
def test_single_candidate_exact_mass_and_finite_gradient(method):
    net = model(method)
    args = inputs()
    args[4][:] = False
    args[4][..., 0] = True
    output = net(*args)
    assert torch.equal(output[..., 0], torch.zeros_like(output[..., 0]))
    (-output[..., 0].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_empty_nonfinite_support_and_bad_public_state_rejected():
    for values in ([float("-inf"), float("-inf")], [float("inf"), 0.], [float("nan"), 0.]):
        with pytest.raises(ValueError, match="normalization"):
            normalize_log_b(torch.tensor(values))
    net = model("scalar")
    args = inputs()
    state = net.initial(args[4])
    state["log_b"][..., 0] = .1
    previous = copy.deepcopy(state)
    with pytest.raises(ValueError, match="normalized"):
        net.step(state, args[0][:, 0], args[1][:, 0], *args[2:5], args[5][:, 0])
    assert torch.equal(state["log_b"], previous["log_b"])


def test_normalized_scalar_step_matches_float64_probability_reference():
    b = torch.tensor([[.7, .2, .1]], dtype=torch.float32)
    logits = torch.tensor([[.2, -.3, .8]])
    departure = torch.tensor([[-.7, .4, -1.2]])
    mask = torch.ones_like(b, dtype=torch.bool)
    output, *_ = transition(normalize_log_b(b.log()), logits, departure, mask, selective=False)
    actual = normalize_log_b(output).exp().numpy()[0]
    bn = b.double().numpy()[0]
    bn /= bn.sum()
    wn = np.exp(logits.double().numpy()[0])
    wn /= wn.sum()
    rn = 1/(1+np.exp(-departure.double().numpy()[0]))
    m = bn @ rn
    np.testing.assert_allclose(actual, (1-m)*bn+m*wn, atol=1e-7, rtol=2e-7)


@pytest.mark.parametrize("method", ["scalar", "selective"])
def test_long_near_one_hot_small_departure_captures_mass_bounds(method):
    class Capture(DialogueCopyMemoryV2):
        def _advance(self, *args):
            state, diagnostic = super()._advance(*args)
            self.masses.append(diagnostic["departure_mass"].detach())
            return state, diagnostic
    net = Capture(method, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3)
    net.masses = []
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias[0] = 0.
        net.head.bias[1] = -12.
    args = inputs(steps=256)
    with torch.inference_mode():
        output = net(*args)
    masses = torch.stack(net.masses)
    assert torch.isfinite(masses).all() and (masses >= 0).all() and (masses <= 1+3e-7).all()
    torch.testing.assert_close(masses, torch.full_like(masses, torch.sigmoid(torch.tensor(-12.)).item()),
                               rtol=2e-6, atol=1e-10)
    assert output.exp()[..., 0].min() > .998
    assert (output.exp().sum(-1)-1).abs().max() < 5e-7
