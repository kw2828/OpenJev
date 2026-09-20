"""Tiny synthetic parity checks only; no corpus, encoder, fitting or cost run."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_batched import (
    PARENT_SHA256,
    PARENT_SOURCE,
    VERSION,
    DialogueTokenBatchedMemory,
)
from openjev.research.dialogue_token_memory import DialogueTokenMemory

ARMS = [(head, mode) for head in ("readout", "scalar") for mode in ("slot", "candidate")]
OUT = {"atol": 1e-5, "rtol": 1e-4}
LOSS = {"atol": 1e-6, "rtol": 1e-5}
GRAD = {"atol": 1e-5, "rtol": 1e-4}
FLOAT_INPUTS = (0, 2, 3, 5, 7)


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


def model(head="scalar", mode="candidate", cls=DialogueTokenBatchedMemory):
    return cls(head, attention_mode=mode, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3)


def actor():
    b, t, q, c, length, dim = 2, 6, 2, 4, 7, 6
    tokens = torch.randn(b, t, length, dim)
    valid = torch.tensor([[True, False, True, True, False, True], [False, True, True, False, True, True]])
    query, candidates = torch.randn(b, q, dim), torch.randn(b, q, c, dim)
    mask = torch.tensor([[[True, True, False, True], [True]*4]]*b)
    lexical = torch.randn(b, t, q, c, 10)
    token_mask = torch.ones(b, t, length, dtype=torch.bool)
    token_mask[0, :, -2:] = False
    token_mask[1, :, -1:] = False
    token_mask[~valid] = False
    prior = (torch.rand(b, t, length)+.1).masked_fill(~token_mask, 0.)
    total = prior.sum(-1, keepdim=True)
    prior /= total.masked_fill(total == 0, 1.)
    return [tokens, valid, query, candidates, mask, lexical, token_mask, prior]


def clone(a, gradients=False):
    return [v.detach().clone().requires_grad_(gradients and i in FLOAT_INPUTS) for i, v in enumerate(a)]


def loss(scores, valid):
    eligible = valid[:, :, None].expand(scores.shape[:3])
    labels = torch.ones(scores.shape[:3], dtype=torch.long)
    labels[:, 1::2] = 3
    ce = F.cross_entropy(scores[eligible], labels[eligible], reduction="none")
    result = (ce*torch.linspace(.7, 1.3, ce.numel())).mean()
    assert torch.isfinite(result)
    return result


def gradients_match(left, right, a, b, *, exact=False):
    tolerance = {"atol": 0., "rtol": 0.} if exact else GRAD
    lp, rp = dict(left.named_parameters()), dict(right.named_parameters())
    assert lp.keys() == rp.keys()
    for name, p in lp.items():
        q = rp[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, **tolerance)
    for i in FLOAT_INPUTS:
        assert a[i].grad is not None and b[i].grad is not None
        assert torch.isfinite(a[i].grad).all() and torch.isfinite(b[i].grad).all()
        torch.testing.assert_close(a[i].grad, b[i].grad, **tolerance)


@pytest.mark.parametrize("head,mode", ARMS)
def test_original_outputs_loss_and_all_input_parameter_gradients(head, mode):
    original, batched = model(head, mode, DialogueTokenMemory), model(head, mode)
    batched.load_state_dict(original.state_dict(), strict=True)
    a = clone(actor(), True)
    b = clone(a, True)
    expected, actual = original(*a), batched(*b)
    torch.testing.assert_close(actual, expected, **OUT)
    le, la = loss(expected, a[1]), loss(actual, b[1])
    torch.testing.assert_close(la, le, **LOSS)
    le.backward()
    la.backward()
    gradients_match(original, batched, a, b)
    for name in ("token_key.weight", "evidence_query.weight"):
        assert dict(batched.named_parameters())[name].grad.abs().sum() > 0
    assert b[0].grad[~b[6]].abs().sum() == b[7].grad[~b[6]].abs().sum() == 0
    assert b[3].grad[~b[4]].abs().sum() == 0


@pytest.mark.parametrize("head,mode", ARMS)
def test_causal_prefix_padding_and_inherited_steps(head, mode):
    m, a = model(head, mode), actor()
    snapshots = clone(a)
    full = m(*a)
    prefix = [x[:, :3] if i in (0, 1, 5, 6, 7) else x for i, x in enumerate(a)]
    torch.testing.assert_close(m(*prefix), full[:, :3], **OUT)
    changed = clone(a)
    changed[0][:, 3:] += 3
    changed[5][:, 3:] -= 5
    torch.testing.assert_close(m(*changed)[:, :3], full[:, :3], **OUT)
    changed = clone(a)
    changed[0][~a[6]] = 900
    changed[3][~a[4]] = -800
    changed[5][~a[1]] = 700
    torch.testing.assert_close(m(*changed), full, atol=0, rtol=0)
    state, outputs = m.initial(a[4]), []
    initial = state["log_b"].clone()
    for t in range(a[0].shape[1]):
        previous = {k: v.clone() for k, v in state.items()}
        state, _ = m.step(state, a[0][:, t], a[1][:, t], *a[2:5], a[5][:, t], a[6][:, t], a[7][:, t])
        assert torch.equal(state["log_b"][~a[1][:, t]], previous["log_b"][~a[1][:, t]])
        outputs.append(state["log_b"])
        old = full[:, t-1] if t else initial
        assert torch.equal(full[:, t][~a[1][:, t]], old[~a[1][:, t]])
    torch.testing.assert_close(full, torch.stack(outputs, 1), **OUT)
    assert (full.exp().double().sum(-1)-1).abs().max() <= 2e-6
    assert all(torch.equal(x, y) for x, y in zip(a, snapshots, strict=True))


@pytest.mark.parametrize("head,mode", ARMS)
def test_candidate_permutation_and_unrelated_query_independence(head, mode):
    m, a = model(head, mode), actor()
    full = m(*a)
    qp, cp = [1, 0], [3, 2, 0, 1]
    other = m(a[0], a[1], a[2][:, qp], a[3][:, qp][:, :, cp], a[4][:, qp][:, :, cp],
              a[5][:, :, qp][:, :, :, cp], a[6], a[7], torch.full((2, 2), 2, dtype=torch.long))
    torch.testing.assert_close(other, full[:, :, qp][:, :, :, cp], **OUT)
    single = m(a[0], a[1], a[2][:, :1], a[3][:, :1], a[4][:, :1], a[5][:, :, :1], a[6], a[7])
    torch.testing.assert_close(single, full[:, :, :1], **OUT)
    changed = clone(a)
    changed[2][:, 1] += 20
    changed[3][:, 1] -= 20
    changed[5][:, :, 1] += 10
    torch.testing.assert_close(m(*changed)[:, :, :1], full[:, :, :1], atol=0, rtol=0)


@pytest.mark.parametrize("head,mode", ARMS)
def test_actual_monitor_preserves_outputs_and_all_gradients_exactly(head, mode, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    spec = importlib.util.spec_from_file_location("batched_monitor", root/"scripts/study_dialogue_copy_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    class MonitoredTokenBatched(v2.MonitoredCopyMemoryV2, DialogueTokenBatchedMemory):
        pass
    plain, monitored = model(head, mode), model(head, mode, MonitoredTokenBatched)
    monitored.load_state_dict(plain.state_dict())
    a = clone(actor(), True)
    b = clone(a, True)
    monitored.begin_batch(b, [{"layout": {"shape": [6, 2, 4, 10]}}]*2)
    expected, actual = plain(*a), monitored(*b)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss(expected, a[1]).backward()
    loss(actual, b[1]).backward()
    gradients_match(plain, monitored, a, b, exact=True)
    n = int(a[1].sum())*2
    assert monitored.audit["incoming_checks"] == monitored.audit["feature_checks"] == monitored.audit["result_checks"] == n
    assert monitored.audit["advance_calls"] == monitored.audit["advance_returned"] == 6
    assert monitored.audit["mass_checks"] == (n if head == "scalar" else 0)
    assert monitored.audit["forward_calls"] == monitored.audit["forward_returned"] == 1


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_initialization_state_schema_count_and_parent_source_are_exact(head):
    start = torch.random.get_rng_state()
    original = DialogueTokenMemory(head)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    batched = DialogueTokenBatchedMemory(head)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert original.state_dict().keys() == batched.state_dict().keys()
    assert all(torch.equal(v, batched.state_dict()[k]) for k, v in original.state_dict().items())
    c = batched.configuration()
    assert c["implementation_version"] == VERSION and c["parameters"] == 173186
    assert c["parent_parameters"] == 99458 and c["adapter_parameters"] == 73728
    assert c["pooling_parent_source"] == {"path": PARENT_SOURCE, "sha256": PARENT_SHA256}
    assert hashlib.sha256((Path(__file__).resolve().parents[1]/PARENT_SOURCE).read_bytes()).hexdigest() == PARENT_SHA256


def test_one_batched_key_projection_and_unchanged_head_work():
    m, a = model(), actor()
    keys, queries, heads = [], [], []
    hooks = [m.token_key.register_forward_pre_hook(lambda _, x: keys.append(tuple(x[0].shape))),
             m.evidence_query.register_forward_pre_hook(lambda _, x: queries.append(tuple(x[0].shape))),
             m.feature.register_forward_pre_hook(lambda _, x: heads.append(tuple(x[0].shape)))]
    try:
        m(*a)
    finally:
        for h in hooks:
            h.remove()
    assert keys == [(2, 6, 7, 6)] and queries == [(2, 2, 4, 12)]
    assert len(heads) == 6 and all(shape[:3] == (2, 2, 4) for shape in heads)


@pytest.mark.parametrize("kind", ["token_nan", "prior_mass", "prior_mask", "empty_real", "lexical_shape", "candidate_nan"])
def test_original_validation_is_preserved(kind):
    a = actor()
    if kind == "token_nan":
        a[0][0, 0, 0, 0] = float("nan")
    elif kind == "prior_mass":
        a[7][0, 0, 0] += .01
    elif kind == "prior_mask":
        a[7][0, 0, -1] = .1
    elif kind == "empty_real":
        a[6][0, 0] = False
        a[7][0, 0] = 0
    elif kind == "lexical_shape":
        a[5] = a[5][..., :9]
    else:
        a[3][0, 0, 2, 0] = float("nan")
    for cls in (DialogueTokenMemory, DialogueTokenBatchedMemory):
        with pytest.raises(ValueError):
            model(cls=cls)(*a)
