"""Synthetic math and input-boundary checks; no corpus or fitted weights."""
from __future__ import annotations

import inspect
import json
import math

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_conditional_observation import DialogueConditionalObservation
from openjev.research.dialogue_typed_observation import (
    CANDIDATE_TYPES,
    MODES,
    PARENT_SHA256,
    DialogueTypedObservation,
    copy_initialization,
)


def sample(mode="typed", dtype=torch.float64, gradients=False):
    torch.manual_seed(410)
    model = DialogueTypedObservation(mode, input_dim=6, projection_dim=4, hidden_dim=5).to(dtype)
    types = torch.tensor([[0, 1, 2, 3, 4, -1], [4, 1, -1, 0, 2, -1], [0, -1, 1, 4, -1, -1]])
    token_mask = torch.tensor([[True, False, True, True, False],
                               [False, True, True, False, False],
                               [True, True, True, True, True]])
    actor = {
        "observation": torch.randn(3, 5, 6, dtype=dtype),
        "query": torch.randn(3, 6, dtype=dtype),
        "candidates": torch.randn(3, 6, 6, dtype=dtype),
        "candidate_mask": types >= 0,
        "lexical": torch.rand(3, 6, 10, dtype=dtype),
        "previous_onehot": F.one_hot(torch.tensor([2, 3, 0]), 6).to(dtype),
        "candidate_types": types,
        "token_mask": token_mask,
        "token_prior": token_mask.to(dtype)/token_mask.sum(-1, keepdim=True),
    }
    if gradients:
        for value in actor.values():
            if value.is_floating_point():
                value.requires_grad_()
    return model, actor


def clone(actor):
    return {name: value.detach().clone() for name, value in actor.items()}


def factors(model, actor):
    return model._scores(**actor)


def pool(model, actor):
    return model.pool(**{k: v for k, v in actor.items()
                         if k not in ("lexical", "previous_onehot", "candidate_types")})


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_normalized_output_and_input_ownership(mode, dtype):
    model, actor = sample(mode, dtype)
    original = clone(actor)
    logp = model(**actor)
    assert logp.shape == (3, 6)
    assert torch.isfinite(logp[actor["candidate_mask"]]).all()
    assert torch.isneginf(logp[~actor["candidate_mask"]]).all()
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(3, dtype=dtype))
    for key in actor:
        assert torch.equal(original[key], actor[key])


@pytest.mark.parametrize("mode", MODES)
def test_full_forward_matches_independent_scalar_factorization_and_branch_mass(mode):
    model, actor = sample(mode)
    z, g, ids, membership = factors(model, actor)
    actual = model(**actor)
    # Python scalar reference, independent of the production log-softmax code.
    for b in range(3):
        supported = actor["candidate_mask"][b].nonzero().flatten().tolist()
        ez = [math.exp(float(z[b, c].detach())) for c in range(6)]
        eg = [math.exp(float(x)) for x in g[b].detach()]
        zsum = [math.fsum(ez[c] for c in supported if int(ids[b, c]) == t) for t in range(3)]
        denominator = math.fsum(eg[t]*zsum[t] for t in range(3))
        for c in supported:
            t = int(ids[b, c])
            expected = (ez[c]*eg[t]/denominator if mode == "flat"
                        else eg[t]/math.fsum(eg)*ez[c]/zsum[t])
            assert float(actual[b, c].exp().detach()) == pytest.approx(expected, abs=1e-13)
        masses = (actual[b].exp()[None]*membership[b]).sum(-1)
        expected_masses = (g[b]+torch.stack([z[b, membership[b, t]].logsumexp(0) for t in range(3)])).softmax(-1)
        if mode == "typed":
            expected_masses = g[b].softmax(-1)
        torch.testing.assert_close(masses, expected_masses)


def test_branch_means_and_feature_flags_are_exactly_masked_public_inputs():
    model, actor = sample()
    features, hidden, means = [], [], []
    handles = [model.feature.register_forward_pre_hook(lambda _m, args: features.append(args[0])),
               model.feature.register_forward_hook(lambda _m, _args, out: hidden.append(out)),
               model.gate.register_forward_pre_hook(lambda _m, args: means.append(args[0]))]
    try:
        model(**actor)
    finally:
        for handle in handles:
            handle.remove()
    assert len(features) == len(hidden) == len(means) == 1
    assert torch.equal(features[0][..., -2], actor["previous_onehot"])
    assert torch.equal(features[0][..., -1], torch.zeros_like(actor["previous_onehot"]))
    flags = F.one_hot(actor["candidate_types"].clamp(min=0), 5).double()
    flags = flags.masked_fill(~actor["candidate_mask"][..., None], 0.)
    assert torch.equal(features[0][..., -7:-2], flags)
    for b in range(3):
        for branch in range(3):
            keep = actor["candidate_mask"][b] & (actor["candidate_types"][b].clamp(max=2) == branch)
            torch.testing.assert_close(means[0][b, branch], hidden[0][b, keep].mean(0))


def test_modes_compute_identical_hidden_candidate_and_gate_scores():
    flat, actor = sample("flat")
    typed = DialogueTypedObservation("typed", 6, 4, 5).double()
    copy_initialization(flat, typed)
    for a, b in zip(factors(flat, actor), factors(typed, actor), strict=True):
        assert torch.equal(a, b)
    assert not torch.allclose(flat(**actor), typed(**actor))


@pytest.mark.parametrize("mode", MODES)
def test_large_logit_stability_and_exact_singleton_effect(mode):
    model, _ = sample(mode)
    mask = torch.ones(1, 4, dtype=torch.bool)
    ids, membership = model._types(torch.tensor([[0, 1, 2, 4]]), mask)
    z = torch.tensor([[10000., -10000., 9000., -9000.]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[-3000., 4000., 2000.]], dtype=torch.float64, requires_grad=True)
    actual = model._log_probabilities(z, g, ids, membership, mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(1, dtype=torch.float64))
    (-actual[0, 2]).backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(g.grad).all()
    if mode == "typed":
        assert torch.equal(z.grad[0, :2], torch.zeros(2, dtype=torch.float64))
        moved = z.detach().clone()
        moved[0, :2] += torch.tensor([1e6, -1e6], dtype=torch.float64)
        assert torch.equal(model._log_probabilities(moved, g.detach(), ids, membership, mask), actual)


@pytest.mark.parametrize("mode", MODES)
def test_common_score_biases_cancel_and_typed_singleton_z_has_zero_gradient(mode):
    model, actor = sample(mode)
    z, g, ids, membership = factors(model, actor)
    z.retain_grad()
    logp = model._log_probabilities(z, g, ids, membership, actor["candidate_mask"])
    F.nll_loss(logp, torch.tensor([3, 4, 3])).backward()
    if mode == "typed":
        assert torch.equal(z.grad[ids < 2], torch.zeros_like(z.grad[ids < 2]))
        assert torch.equal(z.grad[2], torch.zeros_like(z.grad[2]))
    original = model(**actor)
    with torch.no_grad():
        model.head.bias.add_(2.)
        model.gate.bias.sub_(3.)
    torch.testing.assert_close(model(**actor), original)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_all_input_parameter_gradients_and_disclosed_zero_entropy(mode, dtype):
    model, actor = sample(mode, dtype, gradients=True)
    loss = F.nll_loss(model(**actor), torch.tensor([3, 4, 3]))
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
    for name in ("candidates", "lexical"):
        assert (actor[name].grad[~actor["candidate_mask"]] == 0).all()
    for name in ("observation", "token_prior"):
        assert (actor[name].grad[~actor["token_mask"]] == 0).all()
    for name in ("token_key.weight", "evidence_query.weight", "gate.weight", "head.weight"):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0


@pytest.mark.parametrize("mode", MODES)
def test_supplied_previous_value_and_public_type_flags_reach_both_heads(mode):
    model, actor = sample(mode)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.feature[0].weight[0, -2] = 2.
        model.head.weight[0, 0] = 1.
        model.gate.weight[0, 0] = 1.
    first = model(**actor)
    changed = clone(actor)
    changed["previous_onehot"][0] = F.one_hot(torch.tensor(0), 6).double()
    assert model(**changed)[0, 0] > first[0, 0]
    with torch.no_grad():
        model.feature[0].weight.zero_()
        model.feature[0].weight[0, -5] = 2.  # TRUE flag among five public flags.
    base = model(**actor)
    changed = clone(actor)
    changed["candidate_types"][0, 2], changed["candidate_types"][0, 3] = 3, 2
    assert not torch.allclose(model(**changed)[0], base[0])


@pytest.mark.parametrize("mode", MODES)
def test_candidate_permutation_row_independence_and_padding(mode):
    model, actor = sample(mode)
    expected = model(**actor)
    order = torch.tensor([5, 3, 1, 4, 0, 2])
    permuted = clone(actor)
    for name in ("candidates", "candidate_mask", "lexical", "previous_onehot", "candidate_types"):
        permuted[name] = actor[name][:, order]
    torch.testing.assert_close(model(**permuted), expected[:, order])
    rows = torch.tensor([2, 0, 1, 0])
    torch.testing.assert_close(model(**{k: v[rows] for k, v in actor.items()}), expected[rows])
    for b in range(3):
        torch.testing.assert_close(model(**{k: v[b:b+1] for k, v in actor.items()}), expected[b:b+1])
    changed = clone(actor)
    changed["candidates"][~actor["candidate_mask"]] = 10000.
    changed["lexical"][~actor["candidate_mask"]] = -10000.
    changed["observation"][~actor["token_mask"]] = 10000.
    assert torch.equal(model(**changed), expected)


def test_frozen_candidate_attention_is_inherited_and_zero_logits_recover_prior_mean():
    model, actor = sample()
    old = DialogueConditionalObservation("candidate", 6, 4, 5).double()
    old.token_key.load_state_dict(model.token_key.state_dict())
    old.evidence_query.load_state_dict(model.evidence_query.state_dict())
    assert torch.equal(pool(old, actor), pool(model, actor))
    with torch.no_grad():
        model.token_key.weight.zero_()
    clean = torch.where(actor["token_mask"][..., None], actor["observation"], 0.)
    mean = F.normalize((clean*actor["token_prior"][..., None]).sum(1), dim=-1, eps=1e-12)
    expected = mean[:, None].expand(3, 6, 6).masked_fill(~actor["candidate_mask"][..., None], 0.)
    torch.testing.assert_close(pool(model, actor), expected)


def test_paired_initialization_same_draws_full_state_and_configuration():
    torch.manual_seed(410)
    flat = DialogueTypedObservation("flat")
    after_flat = torch.get_rng_state()
    torch.manual_seed(410)
    typed = DialogueTypedObservation("typed")
    assert torch.equal(after_flat, torch.get_rng_state())
    assert flat.state_dict().keys() == typed.state_dict().keys()
    for name, value in flat.state_dict().items():
        assert torch.equal(value, typed.state_dict()[name])
        assert value.data_ptr() != typed.state_dict()[name].data_ptr()
    for model in (flat, typed):
        config = model.configuration()
        assert config["parameters"] == 173506
        assert config["attention_parameters"] == 73728
        assert config["shared_scorer_parameters"] == 99778
        assert config["feature_dim"] == 401
        assert config["zero_input_parameters"] == 64
        assert config["softmax_shift_parameters"] == 2
        assert config["candidate_types"] == CANDIDATE_TYPES
        assert config["parent_source_sha256"] == PARENT_SHA256
        json.dumps(config, allow_nan=False)
    config["candidate_types"]["NONE"] = 999
    assert typed.configuration()["candidate_types"]["NONE"] == 0
    with torch.no_grad():
        typed.gate.weight.add_(1.)
    names = copy_initialization(flat, typed)
    assert names == list(dict(flat.named_parameters()))
    assert all(torch.equal(v, typed.state_dict()[k]) for k, v in flat.state_dict().items())


def test_copy_validation_is_atomic_and_state_is_not_retained():
    source, actor = sample()
    target = DialogueTypedObservation("flat", 6, 4, 6).double()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ValueError, match="Incompatible"):
        copy_initialization(source, target)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())
    original = source(**actor)
    changed = clone(actor)
    changed["observation"] *= -2.
    source(**changed)
    assert torch.equal(source(**actor), original)
    assert not any(isinstance(v, torch.Tensor) for v in vars(source).values())


@pytest.mark.parametrize("kind", ["dtype", "shape", "padded_type", "negative", "too_large", "missing_none", "duplicate_none", "missing_dontcare", "no_concrete"])
def test_invalid_types_and_missing_branches_are_rejected(kind):
    model, actor = sample()
    if kind == "dtype":
        actor["candidate_types"] = actor["candidate_types"].float()
    elif kind == "shape":
        actor["candidate_types"] = actor["candidate_types"][:, :-1]
    elif kind == "padded_type":
        actor["candidate_types"][0, 5] = 0
    elif kind == "negative":
        actor["candidate_types"][0, 2] = -1
    elif kind == "too_large":
        actor["candidate_types"][0, 2] = 5
    elif kind == "missing_none":
        actor["candidate_types"][0, 0] = 4
    elif kind == "duplicate_none":
        actor["candidate_types"][0, 2] = 0
    elif kind == "missing_dontcare":
        actor["candidate_types"][0, 1] = 4
    else:
        actor["candidate_mask"][2, 3] = False
        actor["candidate_types"][2, 3] = -1
    with pytest.raises(ValueError):
        model(**actor)


@pytest.mark.parametrize("field", ["observation", "query", "candidates", "lexical", "previous_onehot", "token_prior"])
def test_nonfinite_inputs_are_rejected_even_off_support(field):
    model, actor = sample()
    actor[field].reshape(-1)[-1] = torch.nan
    with pytest.raises(ValueError):
        model(**actor)


@pytest.mark.parametrize("kind", ["fractional_previous", "masked_previous", "bad_mask", "empty_tokens", "zero_prior", "padded_prior", "bad_prior_mass", "wrong_float_dtype"])
def test_inherited_prior_and_support_guards(kind):
    model, actor = sample()
    if kind == "fractional_previous":
        actor["previous_onehot"][0].zero_()
        actor["previous_onehot"][0, :2] = .5
    elif kind == "masked_previous":
        actor["previous_onehot"][0].zero_()
        actor["previous_onehot"][0, 5] = 1.
    elif kind == "bad_mask":
        actor["candidate_mask"] = actor["candidate_mask"].long()
    elif kind == "empty_tokens":
        actor["token_mask"][0] = False
        actor["token_prior"][0] = 0.
    elif kind == "zero_prior":
        actor["token_prior"][0, 0] = 0.
    elif kind == "padded_prior":
        actor["token_prior"][0, 1] = .1
    elif kind == "bad_prior_mass":
        actor["token_prior"][0] *= 2.
    else:
        actor["lexical"] = actor["lexical"].float()
    with pytest.raises(ValueError):
        model(**actor)


@pytest.mark.parametrize("mode", MODES)
def test_extreme_finite_score_differences_are_rejected_instead_of_repaired(mode):
    model, _ = sample(mode)
    mask = torch.ones(1, 4, dtype=torch.bool)
    ids, membership = model._types(torch.tensor([[0, 1, 2, 4]]), mask)
    maximum = torch.finfo(torch.float64).max
    z = torch.tensor([[0., 0., maximum, -maximum]], dtype=torch.float64)
    with pytest.raises(ValueError, match="Invalid typed output"):
        model._log_probabilities(z, torch.zeros(1, 3, dtype=torch.float64), ids, membership, mask)


def test_forward_has_no_current_gold_transition_or_sequence_input():
    names = tuple(inspect.signature(DialogueTypedObservation.forward).parameters)
    assert names == ("self", "observation", "query", "candidates", "candidate_mask", "lexical",
                     "previous_onehot", "candidate_types", "token_mask", "token_prior")
    model, actor = sample()
    assert not hasattr(model, "_advance")
    with pytest.raises(TypeError):
        model(**actor, current_gold=torch.tensor([3, 4, 3]))


@pytest.mark.parametrize("kwargs", [{"mode": "candidate"}, {"mode": "flat", "input_dim": True},
                                    {"mode": "typed", "projection_dim": 0}])
def test_invalid_constructors(kwargs):
    with pytest.raises(ValueError):
        DialogueTypedObservation(**kwargs)
