"""Synthetic equations, identity and gradient checks; no data or fitted models."""

import inspect

import numpy as np
import pytest
import torch

from openjev.research.dialogue_copy_memory import METHODS, DialogueCopyMemory, transition


@pytest.fixture(autouse=True)
def isolated_engineering():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def model(method):
    return DialogueCopyMemory(method, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3).double()


def inputs():
    turns = torch.randn(2, 4, 6, dtype=torch.float64)
    valid = torch.tensor([[True, True, False, True], [False, True, True, True]])
    query = torch.randn(2, 2, 6, dtype=torch.float64)
    candidates = torch.randn(2, 2, 4, 6, dtype=torch.float64)
    mask = torch.tensor([[[True, True, False, True], [True, True, True, True]]] * 2)
    lexical = torch.rand(2, 4, 2, 4, 10, dtype=torch.float64)
    return turns, valid, query, candidates, mask, lexical


@pytest.mark.parametrize("selective", [False, True])
def test_transition_matches_independent_numpy_and_preserves_mass(selective):
    belief = np.array([.25, .6, 0., .15])
    mask = torch.tensor([[True, True, False, True]])
    logits = np.array([.2, -.9, 10., .6])
    departures = np.array([-.3, 1.2, 90., -1.7])
    valid = mask.numpy()[0]
    w = np.zeros(4)
    w[valid] = np.exp(logits[valid] - max(logits[valid]))
    w /= w.sum()
    r = 1 / (1 + np.exp(-departures))
    mass = belief @ r
    expected = belief * (1 - r if selective else 1 - mass) + mass * w
    with np.errstate(divide="ignore"):
        log_b = torch.tensor(np.log(belief))[None]
    out, log_w, rates, m = transition(log_b, torch.tensor(logits)[None],
                                     torch.tensor(departures)[None], mask, selective=selective)
    np.testing.assert_allclose(out.exp().numpy()[0], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(log_w.exp().numpy()[0], w, rtol=1e-14, atol=1e-14)
    assert m.item() == pytest.approx(mass)
    assert rates.shape == (1, 4)
    assert out.exp().sum().item() == pytest.approx(1.)
    assert torch.isneginf(out[0, 2])


def test_selective_keeps_unrejected_mass_where_scalar_spreads_retention():
    b = torch.tensor([[.6, .3, .1]], dtype=torch.float64)
    r = torch.tensor([[.9, .1, .1]], dtype=torch.float64)
    w = torch.tensor([[.1, .2, .7]], dtype=torch.float64)
    args = b.log(), w.log(), torch.logit(r), torch.ones(1, 3, dtype=torch.bool)
    selected, *_ = transition(*args, selective=True)
    scalar, *_ = transition(*args, selective=False)
    m = (b * r).sum()
    torch.testing.assert_close(selected.exp() - scalar.exp(), b * (m - r), rtol=1e-14, atol=1e-14)
    assert selected.exp()[0, 1] > scalar.exp()[0, 1]
    assert selected.exp()[0, 0] < scalar.exp()[0, 0]


def test_one_hot_prior_and_uniform_departures_are_equivalent_controls():
    mask = torch.ones(1, 3, dtype=torch.bool)
    for b, r in ((torch.tensor([[0., 1., 0.]], dtype=torch.float64), torch.tensor([[-2., 2., .4]], dtype=torch.float64)),
                 (torch.tensor([[.2, .3, .5]], dtype=torch.float64), torch.full((1, 3), -.7, dtype=torch.float64))):
        a, *_ = transition(b.log(), torch.tensor([[.2, -.4, .5]], dtype=torch.float64), r, mask, selective=True)
        z, *_ = transition(b.log(), torch.tensor([[.2, -.4, .5]], dtype=torch.float64), r, mask, selective=False)
        torch.testing.assert_close(a, z, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("departure", [-1000., 1000.])
def test_extreme_departure_logits_approach_copy_or_write_without_nan_gradients(departure):
    b = torch.tensor([[.7, .3, 0.]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    write = torch.tensor([[.2, -.6, 100.]], dtype=torch.float64, requires_grad=True)
    r = torch.full_like(write, departure, requires_grad=True)
    actual, log_w, _, _ = transition(b.log(), write, r, mask, selective=True)
    expected = b if departure < 0 else log_w.exp()
    torch.testing.assert_close(actual.exp(), expected, rtol=1e-13, atol=1e-13)
    (-actual[0, 1]).backward()
    assert torch.isfinite(write.grad).all() and torch.isfinite(r.grad).all()
    assert write.grad[0, 2] == r.grad[0, 2] == 0


@pytest.mark.parametrize("method", METHODS)
def test_forward_streaming_equivalence_no_mutation_and_normalization(method):
    net = model(method)
    args = inputs()
    snapshots = [x.clone() for x in args]
    turns, valid, query, candidates, mask, lexical = args
    actual = net(*args)
    state, expected = net.initial(mask), []
    for t in range(turns.shape[1]):
        old = {k: v.clone() for k, v in state.items()}
        previous = state
        state, _ = net.step(state, turns[:, t], valid[:, t], query, candidates, mask, lexical[:, t])
        for key in previous:
            torch.testing.assert_close(previous[key], old[key], rtol=0, atol=0)
        expected.append(state["log_b"])
    torch.testing.assert_close(actual, torch.stack(expected, 1), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 4, 2, dtype=torch.float64),
                               rtol=0, atol=1e-13)
    assert torch.isneginf(actual.masked_select(~mask[:, None].expand_as(actual))).all()
    for a, b in zip(args, snapshots, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert list(inspect.signature(net.forward).parameters) == [
        "turns", "valid", "query", "candidates", "candidate_mask", "lexical", "none_index"]


@pytest.mark.parametrize("method", METHODS)
def test_prefix_causality_padding_and_future_lexical_separation(method):
    net = model(method)
    args = inputs()
    expected = net(*args)
    changed = [x.clone() for x in args]
    changed[0][:, 2:] += 99
    changed[5][:, 2:] += 33
    torch.testing.assert_close(net(*changed)[:, :2], expected[:, :2], rtol=0, atol=0)
    prefix = [args[0][:, :2], args[1][:, :2], *args[2:5], args[5][:, :2]]
    torch.testing.assert_close(net(*prefix), expected[:, :2], rtol=0, atol=0)
    changed = [x.clone() for x in args]
    changed[0][~args[1]] = -987
    changed[5][~args[1]] = 543
    torch.testing.assert_close(net(*changed), expected, rtol=0, atol=0)
    torch.testing.assert_close(expected[0, 2], expected[0, 1], rtol=0, atol=0)


@pytest.mark.parametrize("method", METHODS)
def test_candidate_and_query_permutation_including_none_identity(method):
    net = model(method)
    args = inputs()
    expected = net(*args)
    turns, valid, query, candidates, mask, lexical = args
    qp, cp = [1, 0], [3, 2, 0, 1]
    actual = net(turns, valid, query[:, qp], candidates[:, qp][:, :, cp],
                 mask[:, qp][:, :, cp], lexical[:, :, qp][:, :, :, cp],
                 torch.full((2, 2), 2, dtype=torch.int64))
    torch.testing.assert_close(actual, expected[:, :, qp][:, :, :, cp], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_finite_gradients_through_predicted_state_and_public_inputs(method):
    net = model(method)
    args = inputs()
    for i in (0, 2, 3, 5):
        args[i].requires_grad_()
    out = net(*args)
    loss = -out[:, -1, :, 1].mean()
    loss.backward()
    for i in (0, 2, 3, 5):
        assert args[i].grad is not None and torch.isfinite(args[i].grad).all()
        assert args[i].grad.abs().sum() > 0
    assert args[0].grad[~args[1]].abs().sum() == 0
    assert args[5].grad[~args[1]].abs().sum() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    if method != "readout":
        assert args[0].grad[0, 0].abs().sum() > 0
    if method == "selective_no_lexical":
        assert args[5].grad[..., :6].abs().sum() == 0
        assert args[5].grad[..., 8:].abs().sum() == 0
        assert args[5].grad[..., 6:8].abs().sum() > 0


@pytest.mark.parametrize("method", METHODS)
def test_unrelated_query_membership_does_not_affect_original_stream(method):
    net = model(method)
    turns, valid, query, candidates, mask, lexical = inputs()
    alone = net(turns, valid, query[:, :1], candidates[:, :1], mask[:, :1], lexical[:, :, :1])
    grouped = net(turns, valid, query, candidates, mask, lexical)
    torch.testing.assert_close(grouped[:, :, :1], alone, rtol=1e-12, atol=1e-12)
    # Even a later-requested unrelated query with entirely different public
    # inputs and ontology cannot influence an already requested query.
    query[:, 1] += 123
    candidates[:, 1] -= 321
    lexical[:, :, 1] += 9
    changed = net(turns, valid, query[:, [1, 0]], candidates[:, [1, 0]],
                  mask[:, [1, 0]], lexical[:, :, [1, 0]])
    torch.testing.assert_close(changed[:, :, 1:], alone, rtol=1e-12, atol=1e-12)


def test_transport_schema_initialization_pairing_and_ablation():
    nets = [model(m) for m in ("scalar", "selective", "selective_no_lexical")]
    for net in nets[1:]:
        net.load_state_dict(nets[0].state_dict(), strict=True)
        assert set(net.state_dict()) == set(nets[0].state_dict())
    args = inputs()
    torch.testing.assert_close(nets[0](*args), nets[1](*args), rtol=1e-13, atol=1e-13)
    modified = [x.clone() for x in args]
    modified[5][..., :6] += 88
    modified[5][..., 8:] -= 44
    torch.testing.assert_close(nets[2](*args), nets[2](*modified), rtol=0, atol=0)
    zeroed = [x.clone() for x in args]
    zeroed[5][..., :6] = 0
    zeroed[5][..., 8:] = 0
    torch.testing.assert_close(nets[1](*zeroed), nets[2](*args), rtol=0, atol=0)
    assert nets[0].configuration()["feature_dim"] == 36
    conf = nets[0].configuration()
    conf["lexical_fields"].clear()
    assert len(nets[0].configuration()["lexical_fields"]) == 10


def test_readout_ignores_earlier_learned_beliefs_but_keeps_public_literal_history():
    net = model("readout")
    args = inputs()
    baseline = net(*args)[:, -1]
    args[0][:, :-1] += 9
    args[5][:, :-1] -= 8
    torch.testing.assert_close(net(*args)[:, -1], baseline, rtol=0, atol=0)
    args[5][:, -1, :, :, 4] += 1
    assert not torch.allclose(net(*args)[:, -1], baseline)


@pytest.mark.parametrize("method", METHODS)
def test_single_valid_candidate_remains_exact_and_gradients_finite(method):
    net = model(method)
    args = list(inputs())
    args[4][:] = False
    args[4][..., 0] = True
    out = net(*args)
    torch.testing.assert_close(out[..., 0], torch.zeros_like(out[..., 0]), rtol=0, atol=1e-14)
    (-out[..., 0].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("kind", ["lexical_width", "nan", "bad_dtype", "bad_mask", "no_candidates", "bad_none"])
def test_invalid_inputs_fail_closed(kind):
    net = model("selective")
    args = list(inputs())
    extra = {}
    if kind == "lexical_width":
        args[5] = args[5][..., :8]
    elif kind == "nan":
        args[5][0, 0, 0, 0, 0] = torch.nan
    elif kind == "bad_dtype":
        args[0] = args[0].float()
    elif kind == "bad_mask":
        args[1] = args[1].long()
    elif kind == "no_candidates":
        args[4][0, 0] = False
    else:
        extra["none_index"] = torch.full((2, 2), 2, dtype=torch.int64)
    with pytest.raises(ValueError):
        net(*args, **extra)


def test_state_guard_preserves_bad_input_and_constructors_reject_unknown():
    with pytest.raises(ValueError):
        DialogueCopyMemory("unknown")
    with pytest.raises(ValueError):
        DialogueCopyMemory("scalar", projection_dim=0)
    net = model("selective")
    turns, valid, query, candidates, mask, lexical = inputs()
    state = net.initial(mask)
    state["log_b"][0, 0, 0] = .2
    before = state["log_b"].clone()
    with pytest.raises(ValueError):
        net.step(state, turns[:, 0], valid[:, 0], query, candidates, mask, lexical[:, 0])
    torch.testing.assert_close(state["log_b"], before, rtol=0, atol=0)
