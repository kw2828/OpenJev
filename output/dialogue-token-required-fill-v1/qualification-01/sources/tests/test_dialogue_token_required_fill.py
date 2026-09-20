"""Required-only schema fill correctness only, with tiny synthetic fixtures."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_consolidated import (
    FACTORED_SHA256,
    FACTORED_SOURCE,
)
from openjev.research.dialogue_token_factored import (
    COUNTER_SHA256,
    COUNTER_SOURCE,
    HEAD_SHA256,
    HEAD_SOURCE,
    LAYOUT_SHA256,
    LAYOUT_SOURCE,
    PARENT_SHA256,
    PARENT_SOURCE,
    REFERENCE_ONLY_FIELDS,
)
from openjev.research.dialogue_token_memory import DialogueTokenMemory
from openjev.research.dialogue_token_required_fill import (
    CONSOLIDATED_SHA256,
    CONSOLIDATED_SOURCE,
    FULL_FILL_REFERENCE_FIELDS,
    VERSION,
    DialogueTokenRequiredFillMemory,
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


def model(head="scalar", mode="candidate", cls=DialogueTokenRequiredFillMemory):
    return cls(head, attention_mode=mode, input_dim=6, projection_dim=4, hidden_dim=7, gru_width=3)


def actor(sparse=True, dtype=torch.float32):
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
    return [x.to(dtype) if x.is_floating_point() else x
            for x in (tokens, valid, query, candidates, mask, lexical, tm, prior)]



def fill_actor(kind, dtype=torch.float32):
    if kind == "full":
        return actor(True, dtype)
    a = actor(False, dtype)
    if kind == "partial":
        a[1][1, 2] = False
        a[4][0, 0, 2] = False
        a[4][1, 1, 1:] = False  # Dummy NONE remains supported.
        a[6][0, :, 1::2] = False
        a[6][1, 1] = False
        a[6][1, 1, [0, 7, 18]] = True
        a[6][~a[1]] = False
        a[7].masked_fill_(~a[6], 0.)
        a[7] /= a[7].sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    assert kind in ("none", "partial")
    return a

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
    tolerance = ({"atol": 0., "rtol": 0.} if exact else
                 {"atol": 1e-10, "rtol": 1e-8} if a[0].dtype == torch.float64 else GRAD)
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
@pytest.mark.parametrize("fill_kind", ["none", "partial", "full"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_fill_routes_outputs_loss_all_input_and_parameter_gradients(head, mode, fill_kind, dtype):
    original, packed = model(head, mode, DialogueTokenMemory), model(head, mode)
    packed.load_state_dict(original.state_dict(), strict=True)
    original, packed = original.to(dtype), packed.to(dtype)
    a = clone(fill_actor(fill_kind, dtype), True)
    b = clone(a, True)
    expected, actual = original(*a), packed(*b)
    torch.testing.assert_close(actual, expected, **({"atol": 1e-10, "rtol": 1e-8} if dtype == torch.float64 else OUT))
    le, la = loss(expected, a[1]), loss(actual, b[1])
    torch.testing.assert_close(la, le, **LOSS)
    le.backward()
    la.backward()
    compare_gradients(original, packed, a, b)
    for name in ("token_key.weight", "evidence_query.weight"):
        assert dict(packed.named_parameters())[name].grad.abs().sum() > 0
    assert b[0].grad[~b[6]].abs().sum() == b[7].grad[~b[6]].abs().sum() == 0
    assert b[3].grad[~b[4]].abs().sum() == 0
    assert packed.last_packing_work == packed.packing_work(b[1], b[4], b[6], 6, 4, 7)


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_padded_preserves_zero_not_absent_gradient_paths(head, mode, dtype, monkeypatch):
    original, packed = model(head, mode, DialogueTokenMemory), model(head, mode)
    packed.load_state_dict(original.state_dict())
    original, packed = original.to(dtype), packed.to(dtype)
    values = actor(dtype=dtype)
    values[1].fill_(False)
    values[6].fill_(False)
    values[7].zero_()
    # Finite padding whose sum overflows must still be masked before arithmetic.
    values[5].fill_(torch.finfo(dtype).max/4)
    a, b = clone(values, True), clone(values, True)
    projection_shapes = []
    exogenous_shapes = []
    exogenous = packed._exogenous

    def record_exogenous(*args):
        exogenous_shapes.append(tuple(args[0].shape))
        return exogenous(*args)

    monkeypatch.setattr(packed, "_exogenous", record_exogenous)
    handle = packed.turn_projection.register_forward_pre_hook(lambda _, x: projection_shapes.append(tuple(x[0].shape)))
    try:
        expected, actual = original(*a), packed(*b)
    finally:
        handle.remove()
    initial = packed.initial(b[4])["log_b"][:, None].expand_as(actual)
    assert torch.equal(actual, expected) and torch.equal(actual, initial)
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    compare_gradients(original, packed, a, b, exact=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in packed.parameters())
    assert packed.last_packing_work["pooling_groups"] == packed.last_packing_work["packed_token_key_positions"] == 0
    assert projection_shapes == [(0, 6)]
    assert packed.last_packing_work["empty_turn_projection_calls"] == packed.last_packing_work["turn_projection_calls"] == 1
    assert packed.last_packing_work["packed_turn_projection_positions"] == 0
    assert exogenous_shapes == [(2, 2, 4, 4)]  # Full schema fill, no supported-row call.
    assert packed.last_packing_work["group_feature_calls"] == 0
    assert packed.last_packing_work["pooled_concat_scalars"] == packed.last_packing_work["pooled_concat_bytes_float32"] == 0


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
@pytest.mark.parametrize("fill_kind", ["none", "partial", "full"])
def test_monitored_outputs_all_gradients_and_real_dummy_coverage(head, mode, fill_kind, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    spec = importlib.util.spec_from_file_location("packed_monitor", root/"scripts/study_dialogue_copy_v2.py")
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    class MonitoredPacked(v2.MonitoredCopyMemoryV2, DialogueTokenRequiredFillMemory):
        pass
    plain, watched = model(head, mode), model(head, mode, MonitoredPacked)
    watched.load_state_dict(plain.state_dict())
    a = clone(fill_actor(fill_kind), True)
    b = clone(a, True)
    layouts = [{"layout": {"shape": [5, 2, 4, 10]}}, {"layout": {"shape": [5, 1, 4, 10]}}]
    watched.begin_batch(b, layouts)
    seen = []
    handle = watched.feature.register_forward_pre_hook(
        lambda _, args: seen.append(tuple(x.detach().clone() for x in args)))
    expected, actual = plain(*a), watched(*b)
    assert len(seen) == 5 and all(len(args) == 2 and args[0].shape[-1] == 2 for args in seen)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    loss(expected, a[1]).backward()
    loss(actual, b[1]).backward()
    compare_gradients(plain, watched, a, b, exact=True)
    executed = int(a[1].sum())*2
    assert watched.audit["incoming_checks"] == watched.audit["feature_checks"] == watched.audit["result_checks"] == executed
    assert watched.audit["real_question_updates"] == int(a[1][0].sum())*2+int(a[1][1].sum())
    assert watched.audit["advance_calls"] == watched.audit["advance_returned"] == 5
    assert watched.audit["mass_checks"] == (executed if head == "scalar" else 0)
    split_features = seen[:]
    seen.clear()
    watched.begin_batch(b, layouts)
    state, rows = watched.initial(b[4]), []
    try:
        for t in range(b[0].shape[1]):
            state, _ = watched.step(state, b[0][:, t], b[1][:, t], *b[2:5], b[5][:, t], b[6][:, t], b[7][:, t])
            rows.append(state["log_b"])
    finally:
        handle.remove()
    assert len(seen) == 5 and all(len(args) == 1 and args[0].shape[-1] == 36 for args in seen)
    for split, ordinary in zip(split_features, seen, strict=True):
        torch.testing.assert_close(split[0], ordinary[0][..., -2:], **OUT)
    torch.testing.assert_close(actual, torch.stack(rows, 1), **OUT)
    assert watched.audit["incoming_checks"] == watched.audit["feature_checks"] == watched.audit["result_checks"] == executed
    assert watched.audit["advance_calls"] == watched.audit["advance_returned"] == 5
    assert watched.audit["mass_checks"] == (executed if head == "scalar" else 0)


def test_exact_work_holes_buckets_dummy_support_and_single_key_projection(monkeypatch):
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
        "dense_evidence_scalars": 216,
        "packed_turn_projection_positions": 11, "dense_turn_projection_positions": 36,
        "turn_projection_calls": 1, "empty_turn_projection_calls": 0,
        "projected_scatter_scalars": 44, "dense_projected_scalars": 144,
        "projected_bias_fill_positions": 36, "projected_skipped_positions": 25,
        "projected_scatter_bytes_float32": 176, "dense_projected_bytes_float32": 576,
        "exogenous_width": 34, "group_feature_positions": 11, "fill_feature_positions": 12,
        "state_feature_positions": 36, "group_feature_calls": 1, "fill_feature_calls": 1, "state_feature_calls": 3,
        "group_exogenous_input_scalars": 374, "fill_exogenous_input_scalars": 408, "state_feature_input_scalars": 72,
        "grouped_head_schema_scalars": 88, "grouped_lexical_scalars": 110,
        "scatter_hidden_scalars": 77, "dense_hidden_scalars": 252, "fill_hidden_scalars": 84,
        "group_feature_macs": 2618, "fill_feature_macs": 2856, "state_feature_macs": 504,
        "factored_feature_macs": 5978, "dense_feature_macs": 9072,
        "scatter_hidden_bytes_float32": 308, "dense_hidden_bytes_float32": 1008,
        "pooled_concat_scalars": 66, "pooled_concat_bytes_float32": 264,
        "full_fill_reference_positions": 12, "full_fill_reference_calls": 1,
        "full_fill_reference_exogenous_scalars": 408, "full_fill_reference_hidden_scalars": 84,
        "full_fill_reference_macs": 2856, "omitted_fill_positions": 0,
        "fill_mask_reduction_positions": 18, "fill_mask_boolean_scalars": 28, "fill_mask_boolean_bytes": 28,
        "fill_count_int64_scalars": 1, "fill_count_int64_bytes": 8,
        "fill_index_int64_scalars": 0, "fill_index_int64_bytes": 0,
        "fill_gathered_schema_scalars": 0, "fill_zero_lexical_scalars": 120,
        "fill_zero_storage_scalars": 0, "fill_schema_scatter_positions": 0, "fill_schema_scatter_scalars": 0,
        "fill_auxiliary_float32_bytes": 480, "fill_mask_predicate_calls": 1}
    m = model()
    assert m.packing_work(valid, mask, tm, 6, 4, 7) == expected
    priors = tm.float()/tm.sum(-1, keepdim=True).clamp_min(1)
    shapes, projection_shapes = [], []
    exogenous_shapes = []
    exogenous = m._exogenous

    def record_exogenous(*args):
        exogenous_shapes.append(tuple(args[0].shape))
        return exogenous(*args)

    monkeypatch.setattr(m, "_exogenous", record_exogenous)
    projection_handle = m.turn_projection.register_forward_pre_hook(lambda _, x: projection_shapes.append(tuple(x[0].shape)))
    handle = m.token_key.register_forward_pre_hook(lambda _, x: shapes.append(tuple(x[0].shape)))
    try:
        m(torch.randn(2, 3, 19, 6), valid, torch.randn(2, 2, 6), torch.randn(2, 2, 3, 6), mask,
          torch.randn(2, 3, 2, 3, 10), tm, priors)
    finally:
        handle.remove()
        projection_handle.remove()
    assert projection_shapes == [(11, 6)]
    assert exogenous_shapes == [(2, 2, 3, 4), (11, 4)]
    assert shapes == [(25, 6)] and m.last_packing_work == expected
    assert all(type(x) is int for x in m.last_packing_work.values())


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_parent_initialization_parameters_and_source_pin(head):
    start = torch.random.get_rng_state()
    original = DialogueTokenMemory(head)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    packed = DialogueTokenRequiredFillMemory(head)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert original.state_dict().keys() == packed.state_dict().keys()
    assert all(torch.equal(x, packed.state_dict()[name]) for name, x in original.state_dict().items())
    c = packed.configuration()
    assert c["packing_reference_only_fields"] == REFERENCE_ONLY_FIELDS+FULL_FILL_REFERENCE_FIELDS
    assert c["implementation_version"] == VERSION and c["parameters"] == 173186
    assert c["parent_parameters"] == 99458 and c["adapter_parameters"] == 73728
    assert c["pooling_parent_source"] == {"path": PARENT_SOURCE, "sha256": PARENT_SHA256}
    for name, path, digest in (("pooling_parent_source", PARENT_SOURCE, PARENT_SHA256),
                               ("packing_layout_source", LAYOUT_SOURCE, LAYOUT_SHA256),
                               ("preprojected_head_source", HEAD_SOURCE, HEAD_SHA256),
                               ("projected_counter_source", COUNTER_SOURCE, COUNTER_SHA256),
                               ("factored_parent_source", FACTORED_SOURCE, FACTORED_SHA256),
                               ("consolidated_parent_source", CONSOLIDATED_SOURCE, CONSOLIDATED_SHA256)):
        assert c[name] == {"path": path, "sha256": digest}
        assert hashlib.sha256((Path(__file__).resolve().parents[1]/path).read_bytes()).hexdigest() == digest


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
    for cls in (DialogueTokenMemory, DialogueTokenRequiredFillMemory):
        with pytest.raises(ValueError):
            model(cls=cls)(*a)
    with pytest.raises(ValueError):
        m(*a)
    assert m.last_packing_work is None


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_nonzero_bias_schema_fill_and_split_preactivation_match_original(head):
    original, factored = model(head, cls=DialogueTokenMemory).double(), model(head).double()
    with torch.no_grad():
        original.turn_projection.bias.copy_(torch.tensor([-.8, -.2, .4, .9], dtype=torch.float64))
        original.candidate_projection.bias.add_(.4)
        original.feature[0].bias.add_(.7)
    factored.load_state_dict(original.state_dict())
    a = fill_actor("partial", torch.float64)
    a[0].zero_()
    old_features, split_features = [], []
    ho = original.feature.register_forward_pre_hook(lambda _, x: old_features.append(x[0].detach().clone()))
    hn = factored.feature.register_forward_pre_hook(
        lambda _, x: split_features.append(tuple(v.detach().clone() for v in x)))
    try:
        expected, actual = original(*a), factored(*a)
    finally:
        ho.remove()
        hn.remove()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    assert len(old_features) == len(split_features) == 5
    for old, (state, exogenous) in zip(old_features, split_features, strict=True):
        assert state.shape[-1] == 2 and exogenous.shape[-1] == 7
        torch.testing.assert_close(state, old[..., -2:], atol=1e-10, rtol=1e-8)
        expected_exogenous = F.linear(old[..., :-2], original.feature[0].weight[:, :-2], original.feature[0].bias)
        torch.testing.assert_close(exogenous, expected_exogenous, atol=1e-10, rtol=1e-8)
        actual_pre = exogenous + F.linear(state, factored.feature[0].weight[:, -2:])
        torch.testing.assert_close(actual_pre, original.feature[0](old), atol=1e-10, rtol=1e-8)
        wanted = factored.turn_projection.bias.tanh().expand_as(old[..., :4])
        torch.testing.assert_close(old[..., :4], wanted, atol=0, rtol=0)
    # Skipped-schema preactivations are not a global zero or a bias-only fill.
    assert not torch.allclose(split_features[1][1][0, 0, 0], split_features[1][1][0, 0, 1])


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_equal_input_projection_width_does_not_misdispatch_inherited_step(head):
    original = DialogueTokenMemory(head, input_dim=6, projection_dim=6, hidden_dim=7).double()
    projected = DialogueTokenRequiredFillMemory(head, input_dim=6, projection_dim=6, hidden_dim=7).double()
    projected.load_state_dict(original.state_dict())
    a = actor(dtype=torch.float64)
    expected, actual = original(*a), projected(*a)
    state, rows = projected.initial(a[4]), []
    for t in range(a[0].shape[1]):
        pool_args = (a[0][:, t], a[1][:, t], *a[2:5], a[6][:, t], a[7][:, t])
        torch.testing.assert_close(projected.pool(*pool_args), original.pool(*pool_args), atol=0, rtol=0)
        state, _ = projected.step(state, a[0][:, t], a[1][:, t], *a[2:5], a[5][:, t], a[6][:, t], a[7][:, t])
        rows.append(state["log_b"])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(actual, torch.stack(rows, 1), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_projected_work_rejects_invalid_projection_width(width):
    a = actor()
    with pytest.raises(ValueError, match="Projection width"):
        DialogueTokenRequiredFillMemory.packing_work(a[1], a[4], a[6], 6, width)


def test_feature_adapter_retains_the_exact_constructed_children(monkeypatch):
    original_init = DialogueTokenMemory.__init__
    constructed = []

    def record(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructed.extend(self.feature.children())

    monkeypatch.setattr(DialogueTokenMemory, "__init__", record)
    m = model()
    assert len(constructed) == 2
    assert m.feature[0] is constructed[0] and m.feature[1] is constructed[1]
    assert list(m.feature.state_dict()) == ["0.weight", "0.bias"]


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_no_activation_cache_after_parameter_change_or_between_backward_calls(head):
    m = model(head)
    a = clone(actor(), True)
    before = m(*a)
    loss(before, a[1]).backward()
    with torch.no_grad():
        m.feature[0].weight[:, :4].add_(.2)
        m.turn_projection.weight.add_(.1)
    m.zero_grad(set_to_none=True)
    fresh = model(head)
    fresh.load_state_dict(m.state_dict())
    b, c = clone(a, True), clone(a, True)
    again, expected = m(*b), fresh(*c)
    assert not torch.equal(before, again)
    torch.testing.assert_close(again, expected, atol=0, rtol=0)
    loss(again, b[1]).backward()
    loss(expected, c[1]).backward()
    compare_gradients(m, fresh, b, c, exact=True)
    assert all(type(v) is int for v in m.last_packing_work.values())


def test_work_rejects_invalid_hidden_width():
    a = actor()
    with pytest.raises(ValueError, match="Hidden width"):
        DialogueTokenRequiredFillMemory.packing_work(a[1], a[4], a[6], 6, 4, 0)


@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_concatenated_normalized_rows_and_schema_lexical_gathers_share_order(mode, monkeypatch):
    m, a = model(mode=mode).double(), actor(dtype=torch.float64)
    original = model(mode=mode, cls=DialogueTokenMemory).double()
    original.load_state_dict(m.state_dict())
    rows = []
    for bi in range(a[0].shape[0]):
        for ti in range(a[0].shape[1]):
            if not a[1][bi, ti]:
                continue
            n = int(a[6][bi, ti].sum())
            group = (min(a[0].shape[2], 16*((n+15)//16)), int(a[4][bi].sum()))
            for qi, ci in a[4][bi].nonzero().tolist():
                rows.append((group, bi, ti, qi, ci))
    rows.sort()
    assert len({row[1:] for row in rows}) == len(rows)
    pools = [original.pool(a[0][:, t], a[1][:, t], *a[2:5], a[6][:, t], a[7][:, t])
             for t in range(a[0].shape[1])]
    expected_pool = torch.stack([pools[t][b, q, c] for _, b, t, q, c in rows])
    qp, cp = m._project_schema(a[2], a[3], a[4])
    expected_q = torch.stack([qp[b, q] for _, b, _, q, _ in rows])
    expected_c = torch.stack([cp[b, q, c] for _, b, _, q, c in rows])
    expected_lex = torch.stack([a[5][b, t, q, c] for _, b, t, q, c in rows])
    projected_inputs, gathered = [], []
    handle = m.turn_projection.register_forward_pre_hook(lambda _, args: projected_inputs.append(args[0].detach().clone()))
    exogenous = m._exogenous

    def record(*args):
        if args[0].ndim == 2:
            gathered.append(tuple(x.detach().clone() for x in args))
        return exogenous(*args)

    monkeypatch.setattr(m, "_exogenous", record)
    try:
        m(*a)
    finally:
        handle.remove()
    assert len(projected_inputs) == len(gathered) == 1
    torch.testing.assert_close(projected_inputs[0], expected_pool, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(projected_inputs[0].norm(dim=-1), torch.ones(len(rows), dtype=torch.float64),
                               atol=1e-10, rtol=1e-8)
    for actual, expected in zip(gathered[0][1:], (expected_q, expected_c, expected_lex), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("fill_kind,required", [("none", 0), ("partial", 9), ("full", 16)])
def test_actual_fill_calls_contents_and_every_omitted_destination_overwritten(fill_kind, required, monkeypatch):
    m, a = model().double(), fill_actor(fill_kind, torch.float64)
    with torch.no_grad():
        m.turn_projection.bias.add_(.3)
        m.candidate_projection.bias.add_(.4)
        m.feature[0].bias.add_(.5)
    qp, cp = m._project_schema(a[2], a[3], a[4])
    needs = (~a[1].all(1))[:, None, None] | (~a[4])
    assert int(needs.sum()) == required
    calls, consumed = [], []
    exogenous, advance = m._exogenous, m._advance

    def record_exogenous(*args):
        result = exogenous(*args)
        calls.append((tuple(x.detach().clone() for x in args), result.detach().clone()))
        return result

    def record_advance(state, turn, *args):
        consumed.append(turn.value.detach().clone())
        return advance(state, turn, *args)

    monkeypatch.setattr(m, "_exogenous", record_exogenous)
    monkeypatch.setattr(m, "_advance", record_advance)
    m(*a)
    assert len(calls) == int(required > 0)+1
    if required:
        args, value = calls[0]
        if required == 16:
            assert args[0].shape == (2, 2, 4, 4)
            fq, fc = qp[:, :, None].expand_as(cp), cp
        else:
            assert args[0].shape == (9, 4)
            ids = needs.flatten().nonzero().flatten()
            fq = qp.reshape(-1, 4)[ids//4]
            fc = cp.reshape(-1, 4)[ids]
        torch.testing.assert_close(args[0], m.turn_projection.bias.tanh().expand_as(args[0]), atol=0, rtol=0)
        torch.testing.assert_close(args[1], fq, atol=0, rtol=0)
        torch.testing.assert_close(args[2], fc, atol=0, rtol=0)
        assert args[3].count_nonzero() == 0
        expected_features = torch.cat((args[0], fq, fc, args[0]*fq, args[0]*fc, fq*fc, args[3]), -1)
        torch.testing.assert_close(value, F.linear(expected_features, m.feature[0].weight[:, :-2], m.feature[0].bias),
                                   atol=0, rtol=0)
    # Independently enumerate the unique supported rows in the public group order.
    rows = []
    for bi in range(2):
        for ti in range(5):
            if a[1][bi, ti]:
                length = int(a[6][bi, ti].sum())
                group = (min(19, 16*((length+15)//16)), int(a[4][bi].sum()))
                rows.extend((group, bi, ti, qi, ci) for qi, ci in a[4][bi].nonzero().tolist())
    rows.sort()
    support = [row[1:] for row in rows]
    assert len(set(support)) == len(support)
    dense = torch.stack(consumed, 1)
    for index, (bi, ti, qi, ci) in enumerate(support):
        torch.testing.assert_close(dense[bi, ti, qi, ci], calls[-1][1][index], atol=0, rtol=0)
    for bi, qi, ci in (~needs).nonzero().tolist():
        assert all((bi, ti, qi, ci) in support for ti in range(5))
    work = m.last_packing_work
    assert len(work) == 70 and all(type(x) is int for x in work.values())
    assert work["fill_feature_calls"] == int(required > 0)
    assert work["fill_feature_positions"] == required
    assert work["fill_feature_macs"] == required*34*7
    assert work["full_fill_reference_positions"] == 16
    assert work["full_fill_reference_macs"] == 16*34*7
    assert work["fill_mask_boolean_bytes"] == 36 and work["fill_mask_reduction_positions"] == 26
    assert work["fill_zero_storage_scalars"] == (112 if required < 16 else 0)
    if fill_kind == "partial":
        expected = {"fill_index_int64_scalars": 18, "fill_index_int64_bytes": 144,
                    "fill_gathered_schema_scalars": 72, "fill_zero_lexical_scalars": 90,
                    "fill_schema_scatter_positions": 9, "fill_schema_scatter_scalars": 63,
                    "fill_auxiliary_float32_bytes": 1096, "factored_feature_macs": 16352}
        assert {key: work[key] for key in expected} == expected
    else:
        assert work["fill_index_int64_scalars"] == work["fill_schema_scatter_positions"] == 0


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("changed_dialogues", [[1], [0, 1]])
def test_only_later_valid_masks_change_fill_route_without_changing_prefix_or_gradients(head, mode, changed_dialogues):
    left, right = model(head, mode).double(), model(head, mode).double()
    right.load_state_dict(left.state_dict())
    a, b = clone(fill_actor("none", torch.float64), True), clone(fill_actor("none", torch.float64), True)
    # Exactly identical values; only late public validity/token masks and priors differ.
    b = clone(a, True)
    with torch.no_grad():
        b[1][changed_dialogues, 4] = False
        b[6][changed_dialogues, 4] = False
        b[7][changed_dialogues, 4] = 0
    expected, actual = left(*a), right(*b)
    assert left.last_packing_work["fill_feature_positions"] == 0
    assert right.last_packing_work["fill_feature_positions"] == len(changed_dialogues)*8
    torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=1e-10, rtol=1e-8)
    loss(expected[:, :3], a[1][:, :3]).backward()
    loss(actual[:, :3], b[1][:, :3]).backward()
    compare_gradients(left, right, a, b)


def test_public_masks_are_recomputed_each_call_without_tensor_cache():
    m = model()
    for kind, required in [("none", 0), ("partial", 9), ("full", 16), ("none", 0)]:
        a = fill_actor(kind)
        fresh = model()
        fresh.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(*a), fresh(*a), atol=0, rtol=0)
        assert m.last_packing_work["fill_feature_positions"] == required
        assert all(type(x) is int for x in m.last_packing_work.values())
