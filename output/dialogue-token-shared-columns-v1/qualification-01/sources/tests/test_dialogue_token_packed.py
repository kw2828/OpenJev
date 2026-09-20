"""Small synthetic parity and public-mask work checks, never a cost/corpus run."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_memory import DialogueTokenMemory
from openjev.research.dialogue_token_packed import (
    PARENT_SHA256,
    PARENT_SOURCE,
    VERSION,
    DialogueTokenPackedMemory,
)

ARMS = [(h, m) for h in ("readout", "scalar") for m in ("slot", "candidate")]
OUT = {"atol": 1e-5, "rtol": 1e-4}
LOSS = {"atol": 1e-6, "rtol": 1e-5}
GRAD = {"atol": 1e-5, "rtol": 1e-4}
FLOATS = (0, 2, 3, 5, 7)


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


def model(head="scalar", mode="candidate", cls=DialogueTokenPackedMemory):
    return cls(head, attention_mode=mode, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3)


def actor(sparse=True):
    b, t, q, c, length, dim = 2, 5, 2, 4, 19, 6
    tokens = torch.randn(b, t, length, dim)
    valid = torch.ones(b, t, dtype=torch.bool)
    query, candidates = torch.randn(b, q, dim), torch.randn(b, q, c, dim)
    mask = torch.ones(b, q, c, dtype=torch.bool)
    lexical = torch.randn(b, t, q, c, 10)
    tm = torch.ones(b, t, length, dtype=torch.bool)
    if sparse:
        valid[0, 1] = valid[1, 0] = valid[:, 3] = False
        mask[0, 0, 2] = False
        mask[1, 1, 1:] = False  # A dummy query still has a supported NONE.
        tm[0, :, 1::2] = False  # Holes, not just a prefix length.
        tm[1, 1] = False
        tm[1, 1, [0, 7, 18]] = True
        tm[~valid] = False
    prior = (torch.rand(b, t, length)+.1).masked_fill(~tm, 0.)
    total = prior.sum(-1, keepdim=True)
    prior /= total.masked_fill(total == 0, 1.)
    return [tokens, valid, query, candidates, mask, lexical, tm, prior]


def clone(a, gradients=False):
    return [x.detach().clone().requires_grad_(gradients and i in FLOATS) for i, x in enumerate(a)]


def loss(output, valid):
    selected = valid[:, :, None].expand(output.shape[:3])
    # NONE is supported even by dummy questions; no unsupported -inf enters CE.
    ce = F.cross_entropy(output[selected], torch.zeros(int(selected.sum()), dtype=torch.long), reduction="none")
    result = (ce*torch.linspace(.7, 1.3, ce.numel())).mean()
    assert torch.isfinite(result)
    return result


def compare_gradients(left, right, a, b, *, exact=False):
    tolerance = {"atol": 0., "rtol": 0.} if exact else GRAD
    lp, rp = dict(left.named_parameters()), dict(right.named_parameters())
    assert lp.keys() == rp.keys()
    for name, p in lp.items():
        q = rp[name]
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, **tolerance)
    for i in FLOATS:
        assert (a[i].grad is None) == (b[i].grad is None)
        assert a[i].grad is not None, i
        assert torch.isfinite(a[i].grad).all() and torch.isfinite(b[i].grad).all()
        torch.testing.assert_close(a[i].grad, b[i].grad, **tolerance)


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("sparse", [False, True])
def test_dense_sparse_outputs_loss_all_input_and_parameter_gradients(head, mode, sparse):
    original, packed = model(head, mode, DialogueTokenMemory), model(head, mode)
    packed.load_state_dict(original.state_dict(), strict=True)
    a = clone(actor(sparse), True)
    b = clone(a, True)
    expected, actual = original(*a), packed(*b)
    torch.testing.assert_close(actual, expected, **OUT)
    le, la = loss(expected, a[1]), loss(actual, b[1])
    torch.testing.assert_close(la, le, **LOSS)
    le.backward()
    la.backward()
    compare_gradients(original, packed, a, b)
    for name in ("token_key.weight", "evidence_query.weight"):
        assert dict(packed.named_parameters())[name].grad.abs().sum() > 0
    assert b[0].grad[~b[6]].abs().sum() == b[7].grad[~b[6]].abs().sum() == 0
    assert b[3].grad[~b[4]].abs().sum() == 0
    assert packed.last_packing_work == packed.packing_work(b[1], b[4], b[6], 6)


@pytest.mark.parametrize("head,mode", ARMS)
def test_all_padded_preserves_zero_not_absent_gradient_paths(head, mode):
    original, packed = model(head, mode, DialogueTokenMemory), model(head, mode)
    packed.load_state_dict(original.state_dict())
    values = actor()
    values[1].fill_(False)
    values[6].fill_(False)
    values[7].zero_()
    a, b = clone(values, True), clone(values, True)
    expected, actual = original(*a), packed(*b)
    initial = packed.initial(b[4])["log_b"][:, None].expand_as(actual)
    assert torch.equal(actual, expected) and torch.equal(actual, initial)
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    compare_gradients(original, packed, a, b, exact=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in packed.parameters())
    assert packed.last_packing_work["pooling_groups"] == packed.last_packing_work["packed_token_key_positions"] == 0


@pytest.mark.parametrize("head,mode", ARMS)
def test_causal_prefix_inherited_steps_padding_and_ownership(head, mode):
    m, a = model(head, mode), actor()
    snapshots = clone(a)
    result = m(*a)
    prefix = [x[:, :3] if i in (0, 1, 5, 6, 7) else x for i, x in enumerate(a)]
    torch.testing.assert_close(m(*prefix), result[:, :3], **OUT)
    changed = clone(a)
    changed[0][:, 3:] += 20
    changed[5][:, 3:] -= 30
    torch.testing.assert_close(m(*changed)[:, :3], result[:, :3], **OUT)
    changed = clone(a)
    changed[0][~a[6]] = 900
    changed[3][~a[4]] = -700
    changed[5][~a[1]] = 500
    torch.testing.assert_close(m(*changed), result, atol=0, rtol=0)
    state, rows = m.initial(a[4]), []
    initial = state["log_b"].clone()
    for t in range(a[0].shape[1]):
        old = {k: v.clone() for k, v in state.items()}
        state, _ = m.step(state, a[0][:, t], a[1][:, t], *a[2:5], a[5][:, t], a[6][:, t], a[7][:, t])
        assert torch.equal(state["log_b"][~a[1][:, t]], old["log_b"][~a[1][:, t]])
        before = initial if t == 0 else result[:, t-1]
        assert torch.equal(result[:, t][~a[1][:, t]], before[~a[1][:, t]])
        rows.append(state["log_b"])
    torch.testing.assert_close(result, torch.stack(rows, 1), **OUT)
    assert (result.exp().double().sum(-1)-1).abs().max() <= 2e-6
    assert all(torch.equal(x, y) for x, y in zip(a, snapshots, strict=True))


@pytest.mark.parametrize("head,mode", ARMS)
def test_candidate_query_permutation_and_unrelated_query_independence(head, mode):
    m, a = model(head, mode), actor()
    old = m(*a)
    qp, cp = [1, 0], [3, 2, 0, 1]
    other = m(a[0], a[1], a[2][:, qp], a[3][:, qp][:, :, cp], a[4][:, qp][:, :, cp],
              a[5][:, :, qp][:, :, :, cp], a[6], a[7], torch.full((2, 2), 2, dtype=torch.long))
    torch.testing.assert_close(other, old[:, :, qp][:, :, :, cp], **OUT)
    alone = m(a[0], a[1], a[2][:, :1], a[3][:, :1], a[4][:, :1], a[5][:, :, :1], a[6], a[7])
    torch.testing.assert_close(alone, old[:, :, :1], **OUT)
    changed = clone(a)
    changed[2][:, 1] += 10
    changed[3][:, 1] -= 10
    changed[5][:, :, 1] += 10
    torch.testing.assert_close(m(*changed)[:, :, :1], old[:, :, :1], atol=0, rtol=0)


@pytest.mark.parametrize("head,mode", ARMS)
def test_monitored_outputs_all_gradients_and_real_dummy_coverage(head, mode, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    spec = importlib.util.spec_from_file_location("packed_monitor", root/"scripts/study_dialogue_copy_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    class MonitoredPacked(v2.MonitoredCopyMemoryV2, DialogueTokenPackedMemory):
        pass
    plain, watched = model(head, mode), model(head, mode, MonitoredPacked)
    watched.load_state_dict(plain.state_dict())
    a = clone(actor(), True)
    b = clone(a, True)
    watched.begin_batch(b, [{"layout": {"shape": [5, 2, 4, 10]}}, {"layout": {"shape": [5, 1, 4, 10]}}])
    expected, actual = plain(*a), watched(*b)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss(expected, a[1]).backward()
    loss(actual, b[1]).backward()
    compare_gradients(plain, watched, a, b, exact=True)
    executed = int(a[1].sum())*2
    assert watched.audit["incoming_checks"] == watched.audit["feature_checks"] == watched.audit["result_checks"] == executed
    assert watched.audit["real_question_updates"] == int(a[1][0].sum())*2+int(a[1][1].sum())
    assert watched.audit["advance_calls"] == watched.audit["advance_returned"] == 5
    assert watched.audit["mass_checks"] == (executed if head == "scalar" else 0)


def test_exact_work_holes_buckets_dummy_support_and_single_key_projection():
    valid = torch.tensor([[True, True, False], [True, False, False]])
    mask = torch.tensor([[[True]*3, [True, False, False]], [[True, True, False], [True, False, False]]])
    tm = torch.zeros(2, 3, 19, dtype=torch.bool)
    tm[0, 0, [0, 5, 18]] = True
    tm[0, 1, :17] = True
    tm[1, 0, [0, 1, 7, 8, 18]] = True
    expected = {"real_turns": 3, "supported_schema_pairs": 7, "packed_token_key_positions": 25,
        "dense_token_key_positions": 114, "packed_score_positions": 188, "dense_score_positions": 684,
        "packed_evidence_positions": 11, "dense_evidence_positions": 36, "pooling_groups": 3,
        "bucket_token_positions": 51, "packed_raw_token_scalars": 150, "grouped_raw_token_scalars": 306,
        "grouped_key_scalars": 3264, "grouped_schema_scalars": 704, "scatter_evidence_scalars": 66,
        "dense_evidence_scalars": 216}
    m = model()
    assert m.packing_work(valid, mask, tm, 6) == expected
    priors = tm.float()/tm.sum(-1, keepdim=True).clamp_min(1)
    shapes = []
    handle = m.token_key.register_forward_pre_hook(lambda _, x: shapes.append(tuple(x[0].shape)))
    try:
        m(torch.randn(2, 3, 19, 6), valid, torch.randn(2, 2, 6), torch.randn(2, 2, 3, 6), mask,
          torch.randn(2, 3, 2, 3, 10), tm, priors)
    finally:
        handle.remove()
    assert shapes == [(25, 6)] and m.last_packing_work == expected
    assert all(type(x) is int for x in m.last_packing_work.values())


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_parent_initialization_parameters_and_source_pin(head):
    start = torch.random.get_rng_state()
    original = DialogueTokenMemory(head)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    packed = DialogueTokenPackedMemory(head)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert original.state_dict().keys() == packed.state_dict().keys()
    assert all(torch.equal(x, packed.state_dict()[name]) for name, x in original.state_dict().items())
    c = packed.configuration()
    assert c["implementation_version"] == VERSION and c["parameters"] == 173186
    assert c["parent_parameters"] == 99458 and c["adapter_parameters"] == 73728
    assert c["pooling_parent_source"] == {"path": PARENT_SOURCE, "sha256": PARENT_SHA256}
    assert hashlib.sha256((Path(__file__).resolve().parents[1]/PARENT_SOURCE).read_bytes()).hexdigest() == PARENT_SHA256


@pytest.mark.parametrize("kind", ["nan_padding", "bad_prior", "empty_real", "schema_nan", "lexical_shape"])
def test_original_validation_and_failed_forward_clears_work(kind):
    m, a = model(), actor()
    m(*a)
    if kind == "nan_padding":
        a[0][0, 1, 0, 0] = float("nan")
    elif kind == "bad_prior":
        a[7][0, 0, 0] += .1
    elif kind == "empty_real":
        a[6][0, 0] = False
        a[7][0, 0] = 0
    elif kind == "schema_nan":
        a[3][0, 0, 2, 0] = float("nan")
    else:
        a[5] = a[5][..., :9]
    for cls in (DialogueTokenMemory, DialogueTokenPackedMemory):
        with pytest.raises(ValueError):
            model(cls=cls)(*a)
    with pytest.raises(ValueError):
        m(*a)
    assert m.last_packing_work is None
