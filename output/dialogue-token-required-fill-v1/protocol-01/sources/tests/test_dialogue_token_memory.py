"""Small synthetic token-adapter checks; no corpus, encoder, fits or checkpoints."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_copy_memory_v2 import DialogueCopyMemoryV2
from openjev.research.dialogue_joint_memory import DialogueJointMemory
from openjev.research.dialogue_token_memory import ADAPTER_PARAMETERS, VERSION, DialogueTokenMemory


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


def net(method="scalar", mode="candidate", dtype=torch.float32, cls=DialogueTokenMemory):
    return cls(method, attention_mode=mode, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3).to(dtype)


def args(steps=5, dtype=torch.float32):
    tokens = torch.arange(2*steps*7*6, dtype=dtype).reshape(2, steps, 7, 6).sin()
    valid = torch.ones((2, steps), dtype=torch.bool)
    valid[0, 1::3] = False
    valid[1, ::4] = False
    q = torch.arange(24, dtype=dtype).reshape(2, 2, 6).cos()
    c = torch.arange(96, dtype=dtype).reshape(2, 2, 4, 6).sin()
    mask = torch.tensor([[[True, True, False, True], [True, True, True, True]]]*2)
    lexical = (torch.arange(2*steps*2*4*10).reshape(2, steps, 2, 4, 10) % 3 == 0).to(dtype)
    tm = torch.ones((2, steps, 7), dtype=torch.bool)
    tm[0, :, -1] = False
    tm[~valid] = False
    prior = tm.to(dtype)/tm.sum(-1, keepdim=True).clamp_min(1)
    return [tokens, valid, q, c, mask, lexical, tm, prior]


def loss(output, valid):
    value = -output[..., 1][valid].mean()
    assert torch.isfinite(value)
    return value


@pytest.mark.parametrize("method", ["readout", "scalar"])
def test_parent_initialization_prefix_and_all_mode_tensors_are_exact(method):
    before = torch.random.get_rng_state()
    parent = DialogueCopyMemoryV2(method)
    torch.random.set_rng_state(before)
    slot = DialogueTokenMemory(method, attention_mode="slot")
    torch.random.set_rng_state(before)
    candidate = DialogueTokenMemory(method, attention_mode="candidate")
    assert set(slot.state_dict())-set(parent.state_dict()) == set(ADAPTER_PARAMETERS)
    assert all(torch.equal(v, slot.state_dict()[k]) for k, v in parent.state_dict().items())
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in slot.state_dict().items())
    config = slot.configuration()
    assert config["implementation_version"] == VERSION
    assert config["parameters"] == 173186 and config["parent_parameters"] == 99458
    assert config["adapter_parameters"] == 73728
    assert all("bias" not in n for n in ADAPTER_PARAMETERS)
    config["adapter_parameter_names"].clear()
    assert slot.configuration()["adapter_parameter_names"] == list(ADAPTER_PARAMETERS)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_zero_attention_logits_recover_original_chunk_weighted_mean(dtype, mode):
    model, a = net(mode=mode, dtype=dtype), args(dtype=dtype)
    with torch.no_grad():
        model.token_key.weight.zero_()
    # Two chunks with 1 and 2 content tokens: 3+4 tokens including CLS/SEP.
    # Each token has chunk-content/(total-content * chunk-token-count).
    a[6].fill_(True)
    a[7][:] = torch.tensor([1/9]*3+[1/6]*4, dtype=dtype)
    a[1].fill_(True)
    actual = model.pool(a[0][:, 0], a[1][:, 0], *a[2:5], a[6][:, 0], a[7][:, 0])
    expected = F.normalize(a[0][:, 0, :3].mean(1)/3 + a[0][:, 0, 3:].mean(1)*2/3, dim=-1)
    expected = expected[:, None, None].expand_as(actual).masked_fill(~a[4][..., None], 0)
    tol = 5e-7 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    parent = DialogueJointMemory("scalar", input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3).to(dtype)
    parent.load_state_dict({k: v for k, v in model.state_dict().items() if k not in ADAPTER_PARAMETERS})
    evidence = torch.stack([model.pool(a[0][:, t], a[1][:, t], *a[2:5], a[6][:, t], a[7][:, t])
                            for t in range(a[0].shape[1])], 1)
    torch.testing.assert_close(model(*a), parent(evidence, *a[1:6]), atol=0, rtol=0)


def test_nonuniform_attention_matches_independent_direct_softmax_reference():
    model, a = net(dtype=torch.float64), args(dtype=torch.float64)
    t = 2
    token = a[0][:, t]
    query = a[2][:, :, None].expand_as(a[3])
    pair = torch.cat([query, a[3]], -1).masked_fill(~a[4][..., None], 0)
    z = pair @ model.evidence_query.weight.T
    k = token @ model.token_key.weight.T
    expected = torch.zeros_like(a[3])
    for b in range(2):
        for q in range(2):
            for c in range(4):
                if a[4][b, q, c]:
                    supported = a[6][b, t]
                    logits = (k[b, supported] @ z[b, q, c])/8 + a[7][b, t, supported].log()
                    value = (logits.softmax(0)[:, None]*token[b, supported]).sum(0)
                    expected[b, q, c] = value/value.norm()
    actual = model.pool(token, a[1][:, t], *a[2:5], a[6][:, t], a[7][:, t])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["readout", "scalar"])
@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_step_forward_exact_parity_and_no_mutation(method, mode):
    model, a = net(method, mode), args()
    snapshot = [x.clone() for x in a]
    state, rows = model.initial(a[4]), []
    for t in range(a[0].shape[1]):
        old = state
        old_copy = {k: v.clone() for k, v in old.items()}
        state, diagnostic = model.step(state, a[0][:, t], a[1][:, t], *a[2:5], a[5][:, t], a[6][:, t], a[7][:, t])
        assert all(torch.equal(old[k], v) for k, v in old_copy.items())
        assert torch.equal(state["log_b"][~a[1][:, t]], old["log_b"][~a[1][:, t]])
        assert state["log_b"].data_ptr() != old["log_b"].data_ptr()
        assert (state["log_b"].exp().double().sum(-1)-1).abs().max() <= 2e-6
        assert torch.isneginf(state["log_b"][~a[4]]).all()
        if method == "scalar":
            mass = diagnostic["departure_mass"][a[1][:, t]]
            assert ((mass >= 0) & (mass <= 1+2e-6)).all()
        rows.append(state["log_b"])
    torch.testing.assert_close(model(*a), torch.stack(rows, 1), atol=0, rtol=0)
    assert all(torch.equal(x, y) for x, y in zip(a, snapshot, strict=True))


@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_future_padding_and_invalid_schema_cannot_change_earlier_predictions(mode):
    model, a = net(mode=mode), args()
    old = model(*a)
    changed = [x.clone() for x in a]
    changed[0][:, 3:] += 99
    changed[5][:, 3:] -= 100
    torch.testing.assert_close(model(*changed)[:, :3], old[:, :3], atol=0, rtol=0)
    changed = [x.clone() for x in a]
    changed[0][~a[6]] = 200
    changed[3][~a[4]] = -300
    changed[5][~a[1]] = 900
    torch.testing.assert_close(model(*changed), old, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_permutation_and_query_independence(mode):
    model, a = net(mode=mode, dtype=torch.float64), args(dtype=torch.float64)
    old = model(*a)
    qp, cp = [1, 0], [3, 2, 0, 1]
    actual = model(a[0], a[1], a[2][:, qp], a[3][:, qp][:, :, cp], a[4][:, qp][:, :, cp],
                   a[5][:, :, qp][:, :, :, cp], a[6], a[7], torch.full((2, 2), 2, dtype=torch.int64))
    torch.testing.assert_close(actual, old[:, :, qp][:, :, :, cp], atol=2e-12, rtol=2e-12)
    alone = model(a[0], a[1], a[2][:, :1], a[3][:, :1], a[4][:, :1], a[5][:, :, :1], a[6], a[7])
    torch.testing.assert_close(alone, old[:, :, :1], atol=2e-12, rtol=2e-12)
    changed = [x.clone() for x in a]
    changed[2][:, 1] += 80
    changed[3][:, 1] -= 80
    changed[5][:, :, 1] += 80
    torch.testing.assert_close(model(*changed)[:, :, :1], old[:, :, :1], atol=0, rtol=0)


def test_slot_pool_is_candidate_invariant_but_candidate_pool_can_respond():
    model, a = net(dtype=torch.float64), args(dtype=torch.float64)
    changed = a[3].clone()
    changed[:, 0, 1] += .75
    def pool():
        return model.pool(a[0][:, 2], a[1][:, 2], a[2], a[3], a[4], a[6][:, 2], a[7][:, 2])
    original = pool()
    other = model.pool(a[0][:, 2], a[1][:, 2], a[2], changed, a[4], a[6][:, 2], a[7][:, 2])
    assert (original[:, 0, 1]-other[:, 0, 1]).abs().max() > 1e-7
    model.attention_mode = "slot"
    original = pool()
    other = model.pool(a[0][:, 2], a[1][:, 2], a[2], changed, a[4], a[6][:, 2], a[7][:, 2])
    torch.testing.assert_close(original, other, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["readout", "scalar"])
@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_nonzero_finite_adapter_gradients_and_zero_padding_gradients(method, mode):
    model, a = net(method, mode), args()
    for i in (0, 2, 3, 5, 7):
        a[i].requires_grad_()
    loss(model(*a), a[1]).backward()
    for name, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
        if name in ADAPTER_PARAMETERS:
            assert p.grad is not None and p.grad.abs().sum() > 0
    for i in (0, 2, 3, 5, 7):
        assert a[i].grad is not None and torch.isfinite(a[i].grad).all()
    assert a[0].grad[~a[6]].abs().sum() == 0
    assert a[7].grad[~a[6]].abs().sum() == 0
    assert a[3].grad[~a[4]].abs().sum() == 0


@pytest.mark.parametrize("method", ["readout", "scalar"])
def test_actual_monitor_mro_preserves_outputs_and_parameter_gradients_bitwise(method, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("token_actual_monitor", root / "scripts/study_dialogue_copy_v2.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    class MonitoredToken(monitor.MonitoredCopyMemoryV2, DialogueTokenMemory):
        pass
    plain, watched = net(method), net(method, cls=MonitoredToken)
    watched.load_state_dict(plain.state_dict())
    a = args()
    watched.begin_batch(a, [{"layout": {"shape": [5, 2, 4, 10]}}]*2)
    expected, actual = plain(*a), watched(*a)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss(expected, a[1]).backward()
    loss(actual, a[1]).backward()
    for p, q in zip(plain.parameters(), watched.parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    n = int(a[1].sum())*2
    assert watched.audit["incoming_checks"] == watched.audit["feature_checks"] == watched.audit["result_checks"] == n
    assert watched.audit["mass_checks"] == (n if method == "scalar" else 0)
    assert watched.audit["forward_calls"] == watched.audit["forward_returned"] == 1


def test_projection_work_never_expands_raw_tokens_over_schema():
    model, a = net(), args()
    key_shapes, query_shapes = [], []
    handles = [model.token_key.register_forward_pre_hook(lambda _, x: key_shapes.append(tuple(x[0].shape))),
               model.evidence_query.register_forward_pre_hook(lambda _, x: query_shapes.append(tuple(x[0].shape)))]
    try:
        model(*a)
    finally:
        for h in handles:
            h.remove()
    assert key_shapes == [(2, 7, 6)]*5
    assert query_shapes == [(2, 2, 4, 12)]


def test_zero_token_vector_and_all_padded_context_are_finite_and_exact_noop():
    model, a = net(), args()
    a[0].zero_()
    result = model(*a)
    loss(result, a[1]).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    a[1].fill_(False)
    a[6].fill_(False)
    a[7].zero_()
    pooled = model.pool(a[0][:, 0], a[1][:, 0], *a[2:5], a[6][:, 0], a[7][:, 0])
    assert torch.equal(pooled, torch.zeros_like(pooled))
    initial = model.initial(a[4])["log_b"][:, None].expand(2, 5, 2, 4)
    assert torch.equal(model(*a), initial)


@pytest.mark.parametrize("kind", ["token_rank", "empty_tokens", "mask_type", "mask_shape", "prior_zero", "prior_negative",
    "prior_nonzero_padding", "prior_sum", "nan_token", "inf_prior", "dtype", "empty_real", "nan_candidate"])
def test_malformed_inputs_are_rejected(kind):
    model, a = net(), args()
    if kind == "token_rank":
        a[0] = a[0][:, :, 0]
    elif kind == "empty_tokens":
        a[0] = a[0][:, :, :0]
    elif kind == "mask_type":
        a[6] = a[6].float()
    elif kind == "mask_shape":
        a[6] = a[6][:, :, :6]
    elif kind == "prior_zero":
        a[7][0, 0, 0] = 0
    elif kind == "prior_negative":
        a[7][0, 0, 0] = -.1
    elif kind == "prior_nonzero_padding":
        a[7][0, 0, -1] = .1
    elif kind == "prior_sum":
        a[7][0, 0, 0] += .01
    elif kind == "nan_token":
        a[0][0, 1, 0, 0] = float("nan")
    elif kind == "inf_prior":
        a[7][0, 1, 0] = float("inf")
    elif kind == "dtype":
        a[7] = a[7].double()
    elif kind == "empty_real":
        a[6][0, 0] = False
        a[7][0, 0] = 0
    else:
        a[3][0, 0, 2, 0] = float("nan")
    with pytest.raises(ValueError):
        model(*a)
