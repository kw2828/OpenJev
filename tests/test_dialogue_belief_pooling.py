"""Synthetic causal, symmetry and gradient tests, without corpus/model assets."""
import copy

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_belief_pooling import METHODS, DialogueBeliefPooling

torch.set_num_threads(1)


def fixture(method="belief_query"):
    torch.manual_seed(725)
    model = DialogueBeliefPooling(method, input_dim=4, attention_dim=3, projection_dim=3, hidden_dim=5).double()
    payload = {"token_states": [torch.randn(n, 4, dtype=torch.float64) for n in (3, 5, 2)],
        "pooled_turns": F.normalize(torch.randn(3, 4, dtype=torch.float64), dim=-1),
        "query": F.normalize(torch.randn(2, 4, dtype=torch.float64), dim=-1),
        "candidates": F.normalize(torch.randn(2, 4, 4, dtype=torch.float64), dim=-1),
        "candidate_mask": torch.tensor([[True, True, True, False], [True, True, True, True]]),
        "lexical": torch.zeros(3, 2, 4, 10, dtype=torch.float64), "none_index": torch.tensor([1, 2])}
    payload["lexical"][:, 0, 1, 6] = 1
    payload["lexical"][:, 1, 2, 6] = 1
    return model, payload


def activate(model):
    with torch.no_grad():
        model.read_output.weight.copy_(torch.tensor([[.4, .1, -.2], [.2, -.1, .3],
            [.1, .2, -.1], [-.1, .4, .2]], dtype=model.read_output.weight.dtype))


@pytest.mark.parametrize("method", METHODS)
def test_normalized_full_stream_and_exact_scalar_initialization(method):
    model, p = fixture(method)
    actual = model(**p)
    expected = model.memory(p["pooled_turns"][None], torch.ones(1, 3, dtype=torch.bool),
        p["query"][None], p["candidates"][None], p["candidate_mask"][None],
        p["lexical"][None], none_index=p["none_index"][None])
    assert actual.shape == (1, 3, 2, 4)
    assert torch.isneginf(actual[0, :, 0, 3]).all()
    assert torch.allclose(actual.logsumexp(-1), torch.zeros(1, 3, 2, dtype=actual.dtype), atol=1e-14)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert model.configuration()["summary_is_lossless"] is False


def test_parameters_and_initializations_match_across_all_arms():
    models = [fixture(method)[0] for method in METHODS]
    for model in models[1:]:
        assert set(model.state_dict()) == set(models[0].state_dict())
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, models[0].state_dict()[name])
    assert models[0].configuration()["active_parameters"] < models[1].configuration()["active_parameters"]
    assert len({m.configuration()["active_parameters"] for m in models[1:]}) == 1


def test_float32_all_arms_start_bit_identically_and_count_executed_work():
    model, p = fixture("pooled")
    model.float()
    p = {k: ([x.float() for x in v] if k == "token_states" else
             v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
         for k, v in p.items()}
    # Stored float32 unit vectors are not guaranteed to have an exact norm1.
    p["pooled_turns"] *= 1.000001
    original = model(**p)
    for method in METHODS[1:]:
        other = fixture(method)[0].float()
        other.load_state_dict(model.state_dict())
        assert torch.equal(original, other(**p))
        assert other.last_audit == {"forward_attempts": 1, "forward_returns": 1,
            "observation_attempts": 3, "observation_returns": 3, "step_attempts": 3,
            "step_returns": 3, "state_checks": 3, "real_question_updates": 6,
            "attention_positions": 26}


def test_invalid_none_index_rejected_clearly():
    model, p = fixture()
    for none in ([1, 2], torch.tensor([True, False]), torch.tensor([1])):
        p["none_index"] = none
        with pytest.raises(ValueError, match="NONE"):
            model(**p)


@pytest.mark.parametrize("method", METHODS)
def test_candidate_and_query_permutations_preserve_meaning(method):
    model, p = fixture(method)
    activate(model)
    original = model(**p)
    changed = copy.deepcopy(p)
    order = torch.tensor([2, 3, 0, 1])
    changed["candidates"] = p["candidates"][:, order]
    changed["candidate_mask"] = p["candidate_mask"][:, order]
    changed["lexical"] = p["lexical"][:, :, order]
    changed["none_index"] = torch.argsort(order)[p["none_index"]]
    assert torch.allclose(model(**changed), original[..., order], atol=1e-13)
    for name in ("query", "candidates", "candidate_mask", "none_index"):
        changed[name] = p[name].flip(0)
    changed["lexical"] = p["lexical"].flip(1)
    assert torch.allclose(model(**changed), original.flip(2), atol=1e-13)


@pytest.mark.parametrize("method", METHODS)
def test_future_and_unrelated_query_cannot_affect_prior_predictions(method):
    model, p = fixture(method)
    activate(model)
    original = model(**p)
    changed = copy.deepcopy(p)
    changed["token_states"][-1] *= -2
    changed["pooled_turns"][-1] *= -1
    assert torch.equal(model(**changed)[:, :2], original[:, :2])
    changed = copy.deepcopy(p)
    changed["query"][1] *= -1
    changed["candidates"][1] *= -1
    assert torch.equal(model(**changed)[:, :, 0], original[:, :, 0])


@pytest.mark.parametrize("method", METHODS)
def test_padding_content_cannot_change_supported_outputs(method):
    model, p = fixture(method)
    activate(model)
    original = model(**p)
    p["candidates"][0, 3] = 1e6
    assert torch.equal(model(**p), original)


@pytest.mark.parametrize("method", ["schema_attention", "belief_query", "state_token"])
def test_feedback_placement_and_full_distribution_gradient(method):
    model, p = fixture(method)
    activate(model)
    # Swapping two masses preserves argmax and entropy but changes the summary.
    a = torch.tensor([[.5, .3, .2, 0.], [.5, .3, .1, .1]], dtype=torch.float64)
    b = torch.tensor([[.5, .2, .3, 0.], [.5, .1, .3, .1]], dtype=torch.float64)
    first, wa = model.observe(p["token_states"][0], p["pooled_turns"][0], p["query"],
                             p["candidates"], p["candidate_mask"], a.log())
    second, wb = model.observe(p["token_states"][0], p["pooled_turns"][0], p["query"],
                              p["candidates"], p["candidate_mask"], b.log())
    if method == "schema_attention":
        assert torch.equal(first, second) and torch.equal(wa, wb)
    else:
        assert not torch.allclose(first, second, atol=1e-10)
    # Only changing the appended KV token leaves ratios between original text
    # weights unchanged; changing the read query changes those ratios.
    ra, rb = wa[:, 0] / wa[:, 1], wb[:, 0] / wb[:, 1]
    if method == "belief_query":
        assert not torch.allclose(ra, rb, atol=1e-10)
    else:
        assert torch.allclose(ra, rb, atol=1e-14)
    log_b = a.log().detach().requires_grad_()
    out, _ = model.observe(p["token_states"][0], p["pooled_turns"][0], p["query"],
                          p["candidates"], p["candidate_mask"], log_b)
    grad = torch.autograd.grad(out[:, 0].sum(), log_b, allow_unused=True)[0]
    if method == "schema_attention":
        assert grad is None
    else:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("method", ["belief_query", "state_token"])
def test_summary_collision_is_a_documented_limit(method):
    model, p = fixture(method)
    activate(model)
    p["candidates"][:, 2] = p["candidates"][:, 1]
    a = torch.tensor([[.5, .25, .25, 0.], [.5, .125, .25, .125]], dtype=torch.float64)
    b = torch.tensor([[.5, .125, .375, 0.], [.5, .25, .125, .125]], dtype=torch.float64)
    inputs = (p["token_states"][0], p["pooled_turns"][0], p["query"], p["candidates"], p["candidate_mask"])
    assert torch.allclose(model.observe(*inputs, a.log())[0], model.observe(*inputs, b.log())[0], atol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_final_loss_backpropagates_through_complete_autonomous_stream(method):
    model, p = fixture(method)
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
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("damage", ["nan", "empty_turn", "bad_unit", "lexical_padding", "nonbinary"])
def test_invalid_actor_inputs_fail(damage):
    model, p = fixture()
    if damage == "nan":
        p["token_states"][0][0, 0] = float("nan")
    elif damage == "empty_turn":
        p["token_states"][0] = torch.empty(0, 4, dtype=torch.float64)
    elif damage == "bad_unit":
        p["pooled_turns"][0] *= 2
    elif damage == "lexical_padding":
        p["lexical"][0, 0, 3, 0] = 1
    else:
        p["lexical"][0, 0, 0, 0] = .5
    with pytest.raises(ValueError):
        model(**p)


def test_actor_does_not_accept_labels_or_gold_state():
    model, p = fixture()
    for name in ("labels", "previous_gold", "scored_mask"):
        with pytest.raises(TypeError):
            model(**p, **{name: torch.zeros(1)})
