"""Synthetic parity, causality and feedback checks; no corpus or checkpoints."""
import copy

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_belief_pooling import METHODS
from openjev.research.dialogue_warm_pooling import DialogueWarmPooling

torch.set_num_threads(1)


def fixture(method="belief_query", q=3, d=7, p=5, dtype=torch.float32):
    torch.manual_seed(1821)
    model = DialogueWarmPooling(method, input_dim=d, attention_dim=4,
                                projection_dim=p, hidden_dim=9).to(dtype)
    # A nonconstant learned departure tests feedback beyond the initialization.
    with torch.no_grad():
        model.memory.head.weight.normal_(0, .3)
    payload = {"token_states": [torch.randn(n, d, dtype=dtype) for n in (3, 6, 2, 4)],
        "pooled_turns": F.normalize(torch.randn(4, d, dtype=dtype), dim=-1) * 1.000001,
        "query": F.normalize(torch.randn(q, d, dtype=dtype), dim=-1),
        "candidates": F.normalize(torch.randn(q, 5, d, dtype=dtype), dim=-1),
        "candidate_mask": torch.ones(q, 5, dtype=torch.bool),
        "lexical": torch.randint(0, 2, (4, q, 5, 10)).to(dtype),
        "none_index": torch.arange(q) % 4}
    payload["candidate_mask"][0, -1] = False
    payload["lexical"][:, 0, -1] = 0
    return model, payload


def old_forward(model, p):
    return model.memory(p["pooled_turns"][None], torch.ones(1, 4, dtype=torch.bool),
        p["query"][None], p["candidates"][None], p["candidate_mask"][None],
        p["lexical"][None], p["none_index"][None])


def activate(model):
    with torch.no_grad():
        model.read_output.weight.normal_(0, .25)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("q,d,p", [(1, 7, 5), (3, 7, 5), (13, 384, 64)])
def test_zero_adapter_reproduces_original_float32_predictions_exactly(method, q, d, p):
    model, payload = fixture(method, q=q, d=d, p=p)
    observed, _ = model.observe(payload["token_states"][0], payload["pooled_turns"][0],
        payload["query"], payload["candidates"], payload["candidate_mask"],
        model.memory.initial(payload["candidate_mask"][None], payload["none_index"][None])["log_b"][0])
    assert torch.equal(observed, payload["pooled_turns"][0].expand(q, -1))
    expected = old_forward(model, payload)
    actual = model(**payload)
    assert torch.equal(actual, expected)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
    assert torch.isneginf(actual[0, :, 0, -1]).all()


@pytest.mark.parametrize("method", METHODS)
def test_causality_and_query_independence(method):
    model, p = fixture(method, dtype=torch.float64)
    activate(model)
    original = model(**p)
    changed = copy.deepcopy(p)
    changed["token_states"][-1] *= -3
    changed["pooled_turns"][-1] *= -1
    assert torch.equal(model(**changed)[:, :-1], original[:, :-1])
    changed = copy.deepcopy(p)
    changed["query"][1:] *= -1
    changed["candidates"][1:] *= -1
    assert torch.equal(model(**changed)[:, :, 0], original[:, :, 0])


@pytest.mark.parametrize("method", METHODS)
def test_candidate_and_query_permutation_and_padding(method):
    model, p = fixture(method, dtype=torch.float64)
    activate(model)
    original = model(**p)
    changed = copy.deepcopy(p)
    order = torch.tensor([4, 2, 0, 3, 1])
    changed["candidates"] = p["candidates"][:, order]
    changed["candidate_mask"] = p["candidate_mask"][:, order]
    changed["lexical"] = p["lexical"][:, :, order]
    changed["none_index"] = torch.argsort(order)[p["none_index"]]
    assert torch.allclose(model(**changed), original[..., order], atol=1e-13, rtol=1e-13)
    for key in ("query", "candidates", "candidate_mask", "none_index"):
        changed[key] = p[key].flip(0)
    changed["lexical"] = p["lexical"].flip(1)
    assert torch.allclose(model(**changed), original.flip(2), atol=1e-13, rtol=1e-13)
    p["candidates"][0, -1] = 1e6
    assert torch.equal(model(**p), original)


@pytest.mark.parametrize("method", METHODS)
def test_full_stream_gradients_and_audit(method):
    model, p = fixture(method, dtype=torch.float64)
    activate(model)
    p["pooled_turns"].requires_grad_()
    logs = model(**p)
    (-logs[0, -1, 0, 2] - logs[0, -1, 1, 0]).backward()
    assert torch.isfinite(p["pooled_turns"].grad).all()
    assert (p["pooled_turns"].grad.abs().sum(-1) > 0).all()
    for name, parameter in model.named_parameters():
        if method == "pooled" and name.startswith("read_"):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.last_audit == {"forward_attempts": 1, "forward_returns": 1,
        "observation_attempts": 4, "observation_returns": 4, "step_attempts": 4,
        "step_returns": 4, "state_checks": 4, "real_question_updates": 12,
        "attention_positions": 0 if method == "pooled" else 57}


@pytest.mark.parametrize("method", METHODS[1:])
def test_state_placement_changes_the_intended_attention_path(method):
    model, p = fixture(method, dtype=torch.float64)
    activate(model)
    a = torch.tensor([[.4, .3, .2, .1, 0.], [.3, .3, .2, .1, .1], [.3, .3, .2, .1, .1]], dtype=torch.float64)
    b = a.clone()
    b[:, 1], b[:, 2] = a[:, 2], a[:, 1]
    args = (p["token_states"][0], p["pooled_turns"][0], p["query"], p["candidates"], p["candidate_mask"])
    oa, wa = model.observe(*args, a.log())
    ob, wb = model.observe(*args, b.log())
    if method == "schema_attention":
        assert torch.equal(oa, ob) and torch.equal(wa, wb)
    else:
        assert not torch.allclose(oa, ob, atol=1e-10)
    if method == "belief_query":
        assert not torch.allclose(wa[:, 0] / wa[:, 1], wb[:, 0] / wb[:, 1], atol=1e-10)
    else:
        assert torch.allclose(wa[:, 0] / wa[:, 1], wb[:, 0] / wb[:, 1], atol=1e-13)


def test_nonzero_projection_agrees_with_direct_query_specific_scalar_steps():
    model, p = fixture(dtype=torch.float64)
    activate(model)
    actual = model(**p)
    state = model.memory.initial(p["candidate_mask"][:, None], p["none_index"][:, None])
    rows = []
    for time, tokens in enumerate(p["token_states"]):
        observed, _ = model.observe(tokens, p["pooled_turns"][time], p["query"],
            p["candidates"], p["candidate_mask"], state["log_b"][:, 0])
        state, _ = model.memory.step(state, observed, torch.ones(3, dtype=torch.bool),
            p["query"][:, None], p["candidates"][:, None], p["candidate_mask"][:, None], p["lexical"][time, :, None])
        rows.append(state["log_b"][:, 0])
    assert torch.allclose(actual, torch.stack(rows)[None], atol=1e-13, rtol=1e-13)


def test_first_optimizer_step_reaches_zero_initialized_attention_output():
    model, p = fixture()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    (-model(**p)[0, -1, 0, 2]).backward()
    assert model.read_output.weight.grad.abs().sum() > 0
    optimizer.step()
    assert model.read_output.weight.abs().sum() > 0
