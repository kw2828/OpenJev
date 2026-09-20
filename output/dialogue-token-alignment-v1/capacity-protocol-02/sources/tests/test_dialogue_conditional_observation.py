"""Synthetic conditional-decoder checks; no corpus, encoder or fitted weights."""
from __future__ import annotations

import inspect
import json

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_conditional_observation import (
    ATTENTION_TENSORS,
    COMMON_TENSORS,
    MODES,
    DialogueConditionalObservation,
    copy_initialization,
)
from openjev.research.dialogue_joint_memory import DialogueJointMemory


def sample(mode="candidate", dtype=torch.float64, *, gradients=False):
    torch.manual_seed(410)
    model = DialogueConditionalObservation(mode, input_dim=6, projection_dim=4, hidden_dim=5).to(dtype)
    mask = torch.tensor([[True, True, True, False], [True, False, True, True], [True, False, False, False]])
    actor = {
        "observation": torch.randn(3, 6, dtype=dtype) if mode == "mean" else torch.randn(3, 5, 6, dtype=dtype),
        "query": torch.randn(3, 6, dtype=dtype),
        "candidates": torch.randn(3, 4, 6, dtype=dtype),
        "candidate_mask": mask,
        "lexical": torch.rand(3, 4, 10, dtype=dtype),
        "previous_onehot": F.one_hot(torch.tensor([1, 2, 0]), 4).to(dtype),
    }
    if mode == "mean":
        actor["observation"] = F.normalize(actor["observation"], dim=-1)
    else:
        token_mask = torch.tensor([[True, False, True, True, False],
                                   [False, True, True, False, False],
                                   [True, True, True, True, True]])
        actor["token_mask"] = token_mask
        actor["token_prior"] = token_mask.to(dtype)/token_mask.sum(-1, keepdim=True)
    if gradients:
        for value in actor.values():
            if value.is_floating_point():
                value.requires_grad_()
    return model, actor


def pool(model, actor):
    return model.pool(**{k: v for k, v in actor.items() if k not in ("lexical", "previous_onehot")})


def clone(actor):
    return {k: v.detach().clone() for k, v in actor.items()}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalized_supported_predictions_preserve_input_ownership(mode, dtype):
    model, actor = sample(mode, dtype)
    original = clone(actor)
    scores = model(**actor)
    assert scores.shape == (3, 4)
    assert torch.isneginf(scores[~actor["candidate_mask"]]).all()
    assert torch.isfinite(scores[actor["candidate_mask"]]).all()
    torch.testing.assert_close(scores.exp().sum(-1), torch.ones(3, dtype=dtype))
    for key in actor:
        assert torch.equal(actor[key], original[key])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_writer_layout_matches_frozen_joint_writer_with_the_same_supplied_prior(mode, dtype):
    model, actor = sample(mode, dtype)
    old = DialogueJointMemory("scalar", input_dim=6, projection_dim=4, hidden_dim=5).to(dtype)
    with torch.no_grad():
        for name in ("turn_projection", "query_projection", "candidate_projection", "feature"):
            getattr(old, name).load_state_dict(getattr(model, name).state_dict())
        old.head.weight[0].copy_(model.head.weight[0])
        old.head.bias[0].copy_(model.head.bias[0])
    candidate_mask = actor["candidate_mask"][:, None]
    qp, cp = old._project_schema(actor["query"][:, None], actor["candidates"][:, None], candidate_mask)
    state = {"none_index": torch.zeros((3, 1), dtype=torch.int64),
             "log_b": actor["previous_onehot"].log()[:, None]}
    _, diagnostic = old._advance(state, pool(model, actor)[:, None], torch.ones(3, dtype=torch.bool),
                                 qp, cp, candidate_mask, actor["lexical"][:, None])
    # Compare only the writer, never the retained mixture or a reset readout.
    torch.testing.assert_close(model(**actor), diagnostic["write_log_probs"][:, 0])


@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_zero_attention_logits_recover_chunk_prior_mean_pooling_and_decoder(mode):
    model, actor = sample(mode)
    with torch.no_grad():
        model.token_key.weight.zero_()
    clean = torch.where(actor["token_mask"][..., None], actor["observation"], 0.)
    mean = F.normalize((actor["token_prior"][..., None]*clean).sum(1), dim=-1, eps=1e-12)
    expected = mean[:, None].expand(3, 4, 6).masked_fill(~actor["candidate_mask"][..., None], 0.)
    torch.testing.assert_close(pool(model, actor), expected)
    mean_model = DialogueConditionalObservation("mean", input_dim=6, projection_dim=4, hidden_dim=5).double()
    copy_initialization(model, mean_model)
    mean_actor = {k: v for k, v in actor.items() if k not in ("token_mask", "token_prior")}
    mean_actor["observation"] = mean
    torch.testing.assert_close(model(**actor), mean_model(**mean_actor))


@pytest.mark.parametrize("mode", ["slot", "candidate"])
def test_attention_matches_direct_supported_token_reference(mode):
    model, actor = sample(mode)
    actual = pool(model, actor)
    for b in range(3):
        indices = actor["token_mask"][b].nonzero().flatten()
        tokens = actor["observation"][b, indices]
        keys = F.linear(tokens, model.token_key.weight)
        for c in range(4):
            if not actor["candidate_mask"][b, c]:
                assert torch.equal(actual[b, c], torch.zeros(6, dtype=torch.float64))
                continue
            other = actor["query"][b] if mode == "slot" else actor["candidates"][b, c]
            q = F.linear(torch.cat((actor["query"][b], other)), model.evidence_query.weight)
            logits = (keys*q).sum(-1)/8+actor["token_prior"][b, indices].log()
            expected = F.normalize((logits.softmax(0)[:, None]*tokens).sum(0), dim=0, eps=1e-12)
            torch.testing.assert_close(actual[b, c], expected)


def test_supplied_previous_indicator_reaches_writer_without_none_reset():
    model, actor = sample("mean")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.feature[0].weight[0, -2] = 2.
        model.head.weight[0, 0] = 1.
    assert torch.equal(model(**actor).argmax(-1), torch.tensor([1, 2, 0]))
    actor["previous_onehot"] = F.one_hot(torch.tensor([2, 3, 0]), 4).double()
    assert torch.equal(model(**actor).argmax(-1), torch.tensor([2, 3, 0]))


def test_slot_ignores_candidate_in_attention_but_still_uses_it_in_scorer():
    model, actor = sample("slot")
    changed = clone(actor)
    changed["candidates"][0, 0, 0] += 2.
    assert torch.equal(pool(model, actor), pool(model, changed))
    with torch.no_grad():
        model.candidate_projection.weight.zero_()
        model.candidate_projection.weight[0, 0] = 1.
        model.candidate_projection.bias.zero_()
        model.feature[0].weight.zero_()
        model.feature[0].bias.zero_()
        model.feature[0].weight[0, 2*model.projection_dim] = 1.
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1.
    assert not torch.equal(model(**actor)[0], model(**changed)[0])


def test_candidate_attention_responds_to_candidate_without_changing_tokens():
    model, actor = sample("candidate")
    with torch.no_grad():
        model.token_key.weight.zero_()
        model.evidence_query.weight.zero_()
        model.token_key.weight[0, 0] = 1.
        model.evidence_query.weight[0, model.input_dim] = 8.
    changed = clone(actor)
    changed["candidates"][0, 0, 0] += 2.
    assert not torch.allclose(pool(model, actor)[0, 0], pool(model, changed)[0, 0])


@pytest.mark.parametrize("mode", MODES)
def test_candidate_permutation_and_independent_query_rows(mode):
    model, actor = sample(mode)
    expected = model(**actor)
    perm = torch.tensor([3, 1, 0, 2])
    permuted = clone(actor)
    for name in ("candidates", "candidate_mask", "lexical", "previous_onehot"):
        permuted[name] = actor[name][:, perm]
    torch.testing.assert_close(model(**permuted), expected[:, perm])
    row_order = torch.tensor([2, 0, 1, 0])
    together = {k: v[row_order] for k, v in actor.items()}
    torch.testing.assert_close(model(**together), expected[row_order])
    for row in range(3):
        torch.testing.assert_close(model(**{k: v[row:row+1] for k, v in actor.items()}), expected[row:row+1])


@pytest.mark.parametrize("mode", MODES)
def test_padding_values_have_no_effect_but_are_still_validated(mode):
    model, actor = sample(mode)
    expected = model(**actor)
    changed = clone(actor)
    changed["candidates"][~actor["candidate_mask"]] = 12345.
    changed["lexical"][~actor["candidate_mask"]] = -12345.
    if mode != "mean":
        changed["observation"][~actor["token_mask"]] = 12345.
    torch.testing.assert_close(model(**changed), expected, atol=0, rtol=0)
    changed["candidates"][0, 3, 0] = torch.nan
    with pytest.raises(ValueError, match="candidates"):
        model(**changed)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_input_and_parameter_gradients_and_zero_entropy_column(mode, dtype):
    model, actor = sample(mode, dtype, gradients=True)
    scores = model(**actor)
    loss = F.nll_loss(scores, torch.tensor([2, 3, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.equal(model.feature[0].weight.grad[:, -1], torch.zeros(model.hidden_dim, dtype=dtype))
    for name, value in actor.items():
        if value.is_floating_point():
            assert value.grad is not None, name
            assert torch.isfinite(value.grad).all(), name
    assert (actor["candidates"].grad[~actor["candidate_mask"]] == 0).all()
    assert (actor["lexical"].grad[~actor["candidate_mask"]] == 0).all()
    if mode != "mean":
        assert (actor["observation"].grad[~actor["token_mask"]] == 0).all()
        assert (actor["token_prior"].grad[~actor["token_mask"]] == 0).all()
        for name in ATTENTION_TENSORS:
            assert dict(model.named_parameters())[name].grad.abs().sum() > 0


def test_paired_copy_preserves_tensor_identity_independence_and_optional_adapter_scope():
    source, _ = sample("slot")
    target = DialogueConditionalObservation("candidate", 6, 4, 5).double()
    mean = DialogueConditionalObservation("mean", 6, 4, 5).double()
    assert copy_initialization(source, target, include_attention=True) == list(COMMON_TENSORS+ATTENTION_TENSORS)
    assert copy_initialization(source, mean) == list(COMMON_TENSORS)
    for name, parameter in source.named_parameters():
        copied = dict(target.named_parameters())[name]
        assert torch.equal(parameter, copied) and parameter.data_ptr() != copied.data_ptr()
        if name in COMMON_TENSORS:
            assert torch.equal(parameter, dict(mean.named_parameters())[name])
    before = {k: v.clone() for k, v in mean.state_dict().items()}
    with pytest.raises(ValueError, match="Incompatible"):
        copy_initialization(source, mean, include_attention=True)
    for name, value in before.items():
        assert torch.equal(value, mean.state_dict()[name])


@pytest.mark.parametrize("mode,parameters", [("mean", 99393), ("slot", 173121), ("candidate", 173121)])
def test_configuration_declares_only_writer_and_no_unused_mean_adapter(mode, parameters):
    model = DialogueConditionalObservation(mode)
    config = model.configuration()
    assert config["parameters"] == parameters
    assert config["zero_input_parameters"] == 64
    assert config["softmax_shift_parameters"] == 1
    assert "potentially_active_parameters" not in config
    assert model.head.weight.shape == (1, 64) and model.head.bias.shape == (1,)
    assert (hasattr(model, "token_key"), hasattr(model, "evidence_query")) == ((False, False) if mode == "mean" else (True, True))
    assert not hasattr(model, "_advance") and not hasattr(model, "initial")
    assert all("departure" not in name for name, _ in model.named_parameters())
    json.dumps(config, allow_nan=False)


def test_preserved_writer_bias_is_only_a_common_softmax_shift():
    model, actor = sample("mean")
    expected = model(**actor)
    with torch.no_grad():
        model.head.bias.add_(2.)
    torch.testing.assert_close(model(**actor), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_extreme_logits_that_overflow_log_probability_differences_are_rejected(dtype):
    model, actor = sample("mean", dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.feature[0].weight[0, -2] = 2.
        model.feature[0].bias[0] = -1.
        model.head.weight[0, 0] = torch.finfo(dtype).max
    captured = []
    hook = model.head.register_forward_hook(lambda _module, _args, output: captured.append(output.detach().clone()))
    try:
        with pytest.raises(ValueError, match="Invalid conditional output"):
            model(**actor)
    finally:
        hook.remove()
    assert len(captured) == 1 and torch.isfinite(captured[0]).all()


def test_forward_has_no_current_gold_future_or_sequence_input_and_stores_no_activations():
    names = tuple(inspect.signature(DialogueConditionalObservation.forward).parameters)
    assert names == ("self", "observation", "query", "candidates", "candidate_mask", "lexical",
                     "previous_onehot", "token_mask", "token_prior")
    model, actor = sample()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    expected = model(**actor)
    other = clone(actor)
    other["observation"] *= -3.
    model(**other)
    assert torch.equal(model(**actor), expected)
    assert not any(isinstance(value, torch.Tensor) for value in vars(model).values())
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])
    with pytest.raises(TypeError):
        model(**actor, current_gold=torch.tensor([1, 2, 0]))


@pytest.mark.parametrize("field", ["observation", "query", "candidates", "lexical", "previous_onehot", "token_prior"])
def test_nonfinite_float_inputs_are_rejected(field):
    model, actor = sample()
    actor[field].reshape(-1)[0] = torch.inf
    with pytest.raises(ValueError):
        model(**actor)


@pytest.mark.parametrize("kind", ["fractional", "zero", "two", "masked", "negative"])
def test_previous_value_must_be_exact_one_hot_on_current_support(kind):
    model, actor = sample()
    row = actor["previous_onehot"][0]
    row.zero_()
    if kind == "fractional":
        row[:2] = .5
    elif kind == "two":
        row[:2] = 1.
    elif kind == "masked":
        row[3] = 1.
    elif kind == "negative":
        row[0], row[1] = -1., 2.
    with pytest.raises(ValueError, match="one-hot"):
        model(**actor)


@pytest.mark.parametrize("kind", ["no_candidates", "no_tokens", "zero_prior", "padding_prior", "bad_mass", "wrong_mask_type", "wrong_dtype", "time_axis"])
def test_invalid_support_geometry_and_dtype_are_rejected(kind):
    model, actor = sample()
    if kind == "no_candidates":
        actor["candidate_mask"][0] = False
    elif kind == "no_tokens":
        actor["token_mask"][0] = False
        actor["token_prior"][0] = 0.
    elif kind == "zero_prior":
        actor["token_prior"][0, 0] = 0.
    elif kind == "padding_prior":
        actor["token_prior"][0, 1] = .1
    elif kind == "bad_mass":
        actor["token_prior"][0] *= 2.
    elif kind == "wrong_mask_type":
        actor["candidate_mask"] = actor["candidate_mask"].long()
    elif kind == "wrong_dtype":
        actor["query"] = actor["query"].float()
    else:
        actor["observation"] = actor["observation"][:, None]
    with pytest.raises(ValueError):
        model(**actor)


def test_mean_rejects_unused_token_inputs():
    model, actor = sample("mean")
    with pytest.raises(ValueError, match="unused token"):
        model(**actor, token_mask=torch.ones(3, 1, dtype=torch.bool))


@pytest.mark.parametrize("kwargs", [{"mode": "other"}, {"mode": "mean", "input_dim": True},
                                    {"mode": "slot", "projection_dim": 0},
                                    {"mode": "candidate", "hidden_dim": 2.5}])
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DialogueConditionalObservation(**kwargs)
