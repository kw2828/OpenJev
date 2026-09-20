"""Tiny synthetic parity and local-view ownership checks, never cost timing."""
import hashlib
from pathlib import Path

import pytest
import torch
from study_dialogue_copy_v2 import MonitoredCopyMemoryV2
from test_dialogue_token_required_fill import (
    ARMS,
    LOSS,
    OUT,
    clone,
    compare_gradients,
    fill_actor,
    loss,
)
from torch.nn import functional as F

from openjev.research.dialogue_token_memory import DialogueTokenMemory
from openjev.research.dialogue_token_required_fill import DialogueTokenRequiredFillMemory
from openjev.research.dialogue_token_shared_columns import (
    REQUIRED_FILL_SHA256,
    REQUIRED_FILL_SOURCE,
    VERSION,
    DialogueTokenSharedColumnsMemory,
    _SharedHeadTerms,
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


def model(head="scalar", mode="candidate", cls=DialogueTokenSharedColumnsMemory, dtype=torch.float32, p=4):
    return cls(head, attention_mode=mode, input_dim=6, projection_dim=p, hidden_dim=7, gru_width=3).to(dtype)


def tolerance(dtype):
    return {"atol": 1e-10, "rtol": 1e-8} if dtype == torch.float64 else OUT


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fill_kind", ["none", "partial", "full"])
def test_original_outputs_loss_all_gradients_and_work(head, mode, dtype, fill_kind):
    original, shared = model(head, mode, DialogueTokenMemory, dtype), model(head, mode, dtype=dtype)
    shared.load_state_dict(original.state_dict())
    a = clone(fill_actor(fill_kind, dtype), True)
    b = clone(a, True)
    expected, actual = original(*a), shared(*b)
    torch.testing.assert_close(actual, expected, **tolerance(dtype))
    left, right = loss(expected, a[1]), loss(actual, b[1])
    torch.testing.assert_close(left, right, **LOSS)
    left.backward()
    right.backward()
    compare_gradients(original, shared, a, b)
    assert shared.last_packing_work == shared.packing_work(b[1], b[4], b[6], 6, 4, 7)
    assert len(shared.last_packing_work) == 74
    base = DialogueTokenRequiredFillMemory.packing_work(b[1], b[4], b[6], 6, 4, 7)
    assert {k: shared.last_packing_work[k] for k in base} == base
    assert shared.last_packing_work["state_column_view_expressions"] == 1
    assert shared.last_packing_work["state_column_reference_expressions"] == 5
    assert shared.last_packing_work["state_column_linear_calls"] == 5
    assert shared.last_packing_work["state_column_view_scalars"] == 14
    assert all(type(x) is int for x in shared.last_packing_work.values())


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_padding_preserves_all_zero_versus_none_gradients(head, mode, dtype):
    original, shared = model(head, mode, DialogueTokenMemory, dtype), model(head, mode, dtype=dtype)
    shared.load_state_dict(original.state_dict())
    a = fill_actor("full", dtype)
    a[1].fill_(False)
    a[6].fill_(False)
    a[7].zero_()
    a[5].fill_(torch.finfo(dtype).max/4)
    a, b = clone(a, True), clone(a, True)
    expected, actual = original(*a), shared(*b)
    assert torch.equal(expected, actual)
    assert torch.equal(actual, shared.initial(b[4])["log_b"][:, None].expand_as(actual))
    expected[..., 0].sum().backward()
    actual[..., 0].sum().backward()
    compare_gradients(original, shared, a, b, exact=True)
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in shared.parameters())
    assert shared.last_packing_work["state_column_view_expressions"] == 1
    assert shared.last_packing_work["state_column_linear_calls"] == 5


class MonitoredSharedColumns(MonitoredCopyMemoryV2, DialogueTokenSharedColumnsMemory):
    pass


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("fill_kind", ["none", "partial", "full"])
def test_actual_monitor_arg0_and_ordinary_step_coverage(head, mode, fill_kind):
    plain, watched = model(head, mode), model(head, mode, MonitoredSharedColumns)
    watched.load_state_dict(plain.state_dict())
    a = clone(fill_actor(fill_kind), True)
    b = clone(a, True)
    layouts = [{"layout": {"shape": [5, 2, 4, 10]}}, {"layout": {"shape": [5, 1, 4, 10]}}]
    watched.begin_batch(b, layouts)
    seen = []

    def record(_, args):
        seen.append((args[0].detach().clone(), args[1] if len(args) == 2 else None))

    handle = watched.feature.register_forward_pre_hook(record)
    try:
        expected, actual = plain(*a), watched(*b)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        loss(expected, a[1]).backward()
        loss(actual, b[1]).backward()
        compare_gradients(plain, watched, a, b, exact=True)
        assert len(seen) == 5
        assert all(x.shape[-1] == 2 and isinstance(terms, _SharedHeadTerms) for x, terms in seen)
        assert len({id(terms.state_columns) for _, terms in seen}) == 1
        executed = int(b[1].sum())*2
        assert watched.audit["feature_checks"] == watched.audit["incoming_checks"] == watched.audit["result_checks"] == executed
        assert watched.audit["advance_calls"] == watched.audit["advance_returned"] == 5
        assert watched.audit["mass_checks"] == (executed if head == "scalar" else 0)
        assert watched.audit["real_question_updates"] == int(b[1][0].sum())*2+int(b[1][1].sum())
        split = seen[:]
        seen.clear()
        watched.begin_batch(b, layouts)
        state, rows = watched.initial(b[4]), []
        for t in range(5):
            before = state["log_b"].clone()
            state, _ = watched.step(state, b[0][:, t], b[1][:, t], *b[2:5], b[5][:, t], b[6][:, t], b[7][:, t])
            assert torch.equal(state["log_b"][~b[1][:, t]], before[~b[1][:, t]])
            rows.append(state["log_b"])
        assert len(seen) == 5 and all(x.shape[-1] == 36 and terms is None for x, terms in seen)
        for (full, _), (live, _) in zip(seen, split, strict=True):
            torch.testing.assert_close(full[..., -2:], live, **OUT)
        torch.testing.assert_close(torch.stack(rows, 1), actual, **OUT)
        assert watched.audit["feature_checks"] == watched.audit["incoming_checks"] == watched.audit["result_checks"] == executed
        assert watched.audit["advance_calls"] == watched.audit["advance_returned"] == 5
    finally:
        handle.remove()


@pytest.mark.parametrize("head,mode", ARMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fresh_shared_view_two_forward_graphs_combined_backward_and_after_update(head, mode, dtype):
    original, shared = model(head, mode, DialogueTokenMemory, dtype), model(head, mode, dtype=dtype)
    shared.load_state_dict(original.state_dict())
    views = []
    hook = shared.feature.register_forward_pre_hook(lambda _, args: views.append(args[1].state_columns))
    old_batches = [clone(fill_actor(k, dtype), True) for k in ("none", "partial")]
    new_batches = [clone(a, True) for a in old_batches]
    try:
        expected = [original(*a) for a in old_batches]
        actual = [shared(*a) for a in new_batches]
        assert len(views) == 10 and all(v is views[0] for v in views[:5]) and all(v is views[5] for v in views[5:])
        assert views[0] is not views[5]
        for v in (views[0], views[5]):
            assert v.requires_grad and v.grad_fn is not None and not v.is_leaf
            assert v.untyped_storage().data_ptr() == shared.feature[0].weight.untyped_storage().data_ptr()
            assert v.shape == (7, 2) and v.stride() == shared.feature[0].weight[:, -2:].stride()
        callbacks = []
        for i in (0, 5):
            views[i].register_hook(lambda g, i=i: callbacks.append((i, g.detach().clone())))
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y, **tolerance(dtype))
        sum(loss(x, a[1]) for x, a in zip(expected, old_batches, strict=True)).backward()
        sum(loss(x, a[1]) for x, a in zip(actual, new_batches, strict=True)).backward()
        assert sorted(i for i, _ in callbacks) == [0, 5]
        assert all(torch.isfinite(g).all() for _, g in callbacks)
        for a, b in zip(old_batches, new_batches, strict=True):
            compare_gradients(original, shared, a, b)
        # No optimizer is needed to check freshness after changed parameter values.
        with torch.no_grad():
            shared.feature[0].weight[:, -2:].add_(.2)
        shared.zero_grad(set_to_none=True)
        fresh = model(head, mode, dtype=dtype)
        fresh.load_state_dict(shared.state_dict())
        a = clone(fill_actor("full", dtype), True)
        b = clone(a, True)
        current, wanted = shared(*a), fresh(*b)
        assert all(v is views[10] for v in views[10:]) and views[10] is not views[0] and views[10] is not views[5]
        torch.testing.assert_close(current, wanted, atol=0, rtol=0)
        loss(current, a[1]).backward()
        loss(wanted, b[1]).backward()
        compare_gradients(shared, fresh, a, b, exact=True)
        assert all(type(v) is int for v in shared.last_packing_work.values())
        assert not any(isinstance(v, (torch.Tensor, _SharedHeadTerms)) for v in vars(shared).values())
    finally:
        hook.remove()


@pytest.mark.parametrize("head,mode", ARMS)
def test_causality_permutation_unrelated_queries_and_input_ownership(head, mode):
    m = model(head, mode).double()
    a = fill_actor("none", torch.float64)
    before = clone(a)
    result = m(*a)
    changed = clone(a)
    changed[0][:, 3:] += 10
    changed[5][:, 3:] -= 20
    changed[1][1, 4] = changed[6][1, 4] = False
    changed[7][1, 4] = 0
    torch.testing.assert_close(m(*changed)[:, :3], result[:, :3], **tolerance(torch.float64))
    prefix = [x[:, :3] if i in (0, 1, 5, 6, 7) else x for i, x in enumerate(a)]
    torch.testing.assert_close(m(*prefix), result[:, :3], **tolerance(torch.float64))
    qp, cp = [1, 0], [3, 2, 0, 1]
    permuted = m(a[0], a[1], a[2][:, qp], a[3][:, qp][:, :, cp], a[4][:, qp][:, :, cp],
                 a[5][:, :, qp][:, :, :, cp], a[6], a[7], torch.full((2, 2), 2, dtype=torch.long))
    torch.testing.assert_close(permuted, result[:, :, qp][:, :, :, cp], **tolerance(torch.float64))
    alone = m(a[0], a[1], a[2][:, :1], a[3][:, :1], a[4][:, :1], a[5][:, :, :1], a[6], a[7])
    torch.testing.assert_close(alone, result[:, :, :1], **tolerance(torch.float64))
    assert all(torch.equal(x, y) for x, y in zip(a, before, strict=True))


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_initializer_exact_child_objects_names_no_rng_or_buffers_and_parent_pins(head, monkeypatch):
    start = torch.random.get_rng_state()
    original = DialogueTokenMemory(head)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    existing, captured = DialogueTokenRequiredFillMemory.__init__, []

    def capture(self, *args, **kwargs):
        existing(self, *args, **kwargs)
        captured.extend(self.feature.children())

    monkeypatch.setattr(DialogueTokenRequiredFillMemory, "__init__", capture)
    m = DialogueTokenSharedColumnsMemory(head)
    assert torch.equal(torch.random.get_rng_state(), after)
    assert m.feature[0] is captured[0] and m.feature[1] is captured[1]
    assert original.state_dict().keys() == m.state_dict().keys()
    assert all(torch.equal(x, m.state_dict()[k]) for k, x in original.state_dict().items())
    assert list(original.named_buffers()) == list(m.named_buffers())
    assert m._advance.__func__ is DialogueTokenRequiredFillMemory._advance
    assert m.step.__func__ is DialogueTokenRequiredFillMemory.step
    config = m.configuration()
    assert config["parameters"] == 173186 and config["implementation_version"] == VERSION
    assert config["required_fill_parent_source"] == {"path": REQUIRED_FILL_SOURCE, "sha256": REQUIRED_FILL_SHA256}
    assert len(config["packing_reference_only_fields"]) == 16
    assert config["packing_view_only_fields"] == ["state_column_view_scalars"]
    assert hashlib.sha256((Path(__file__).resolve().parents[1]/REQUIRED_FILL_SOURCE).read_bytes()).hexdigest() == REQUIRED_FILL_SHA256


@pytest.mark.parametrize("head", ["readout", "scalar"])
def test_equal_dimensions_ordinary_feature_and_step_fallback(head):
    original, shared = model(head, cls=DialogueTokenMemory, dtype=torch.float64, p=6), model(head, dtype=torch.float64, p=6)
    shared.load_state_dict(original.state_dict())
    a = fill_actor("partial", torch.float64)
    expected, actual = original(*a), shared(*a)
    state, rows = shared.initial(a[4]), []
    for t in range(5):
        state, _ = shared.step(state, a[0][:, t], a[1][:, t], *a[2:5], a[5][:, t], a[6][:, t], a[7][:, t])
        rows.append(state["log_b"])
    torch.testing.assert_close(actual, expected, **tolerance(torch.float64))
    torch.testing.assert_close(torch.stack(rows, 1), actual, **tolerance(torch.float64))
    full = torch.randn(3, 48, dtype=torch.float64)
    torch.testing.assert_close(shared.feature(full), original.feature(full), atol=0, rtol=0)
    live, exogenous = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 7, dtype=torch.float64)
    expected_feature = shared.feature[1](exogenous+F.linear(live, shared.feature[0].weight[:, -2:]))
    torch.testing.assert_close(shared.feature(live, exogenous), expected_feature, atol=0, rtol=0)


def test_failed_forward_retains_no_view_or_terms_and_next_call_uses_new_view():
    m, a = model(), fill_actor("none")
    views = []

    def interrupt(_, args):
        views.append(args[1].state_columns)
        if len(views) == 2:
            raise RuntimeError("synthetic interrupted head")

    h = m.feature.register_forward_pre_hook(interrupt)
    try:
        with pytest.raises(RuntimeError, match="synthetic interrupted head"):
            m(*a)
    finally:
        h.remove()
    assert len(views) == 2 and views[0] is views[1]
    assert not any(isinstance(v, (torch.Tensor, _SharedHeadTerms)) for v in vars(m).values())
    h = m.feature.register_forward_pre_hook(lambda _, args: views.append(args[1].state_columns))
    try:
        result = m(*a)
        assert views[2] is not views[0] and all(v is views[2] for v in views[2:])
        loss(result, a[1]).backward()
    finally:
        h.remove()
    assert len(m.last_packing_work) == 74


@pytest.mark.parametrize("field", [0, 2, 3, 5, 7])
def test_unmasked_finite_validation_and_work_reset(field):
    m, a = model(), fill_actor("full")
    m(*a)
    a[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        m(*a)
    assert m.last_packing_work is None
