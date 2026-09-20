"""Small synthetic alignment checks; no corpus, encoder or fitted weights."""
from __future__ import annotations

import inspect
import json
import math

import pytest
import torch
from torch.nn import functional as F

from openjev.research.dialogue_token_alignment import (
    COMMON_TENSORS,
    MODES,
    DialogueTokenAlignment,
    copy_common_initialization,
    copy_initialization,
)
from openjev.research.dialogue_typed_observation import DialogueTypedObservation


def sample(mode='aligned', dtype=torch.float64, gradients=False):
    torch.manual_seed(517)
    model = DialogueTokenAlignment(mode, 6, 4, 5).to(dtype)
    types = torch.tensor([[0, 1, 2, 3, 4, -1], [4, 1, -1, 0, 2, -1]])
    tm = torch.tensor([[True, False, True, True], [False, True, True, False]])
    sm = torch.tensor([[[True, False, True], [True, True, True], [False, True, True],
                        [True, False, False], [True, True, False], [False, False, False]],
                       [[True, True, True], [False, True, False], [False, False, False],
                        [True, False, True], [True, True, False], [False, False, False]]])
    tp = tm.to(dtype)*torch.tensor([1., 2., 3., 4.], dtype=dtype)
    tp = tp/tp.sum(-1, keepdim=True)
    sp = sm.to(dtype)*torch.tensor([1., 2., 4.], dtype=dtype)
    sp = sp/sp.sum(-1, keepdim=True).clamp(min=1)
    actor = {
        'observation': torch.randn(2, 4, 6, dtype=dtype), 'query': torch.randn(2, 6, dtype=dtype),
        'candidates': torch.randn(2, 6, 6, dtype=dtype), 'candidate_mask': types >= 0,
        'lexical': torch.rand(2, 6, 10, dtype=dtype),
        'previous_onehot': F.one_hot(torch.tensor([2, 3]), 6).to(dtype), 'candidate_types': types,
        'token_mask': tm, 'token_prior': tp, 'schema_tokens': torch.randn(2, 6, 3, 6, dtype=dtype),
        'schema_mask': sm, 'schema_prior': sp,
    }
    if gradients:
        for value in actor.values():
            if value.is_floating_point():
                value.requires_grad_()
    return model, actor


def clone(actor):
    return {key: value.detach().clone() for key, value in actor.items()}


def pool(model, actor):
    return model.pool(**{k: v for k, v in actor.items() if k not in ('lexical', 'previous_onehot', 'candidate_types')})


def loss(model, actor):
    return F.nll_loss(model(**actor), torch.tensor([3, 4]))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_finite_masked_normalized_outputs_no_input_mutation(mode, dtype):
    model, actor = sample(mode, dtype)
    before = clone(actor)
    result = model(**actor)
    assert result.shape == (2, 6)
    assert torch.isfinite(result[actor['candidate_mask']]).all()
    assert torch.isneginf(result[~actor['candidate_mask']]).all()
    torch.testing.assert_close(result.exp().sum(-1), torch.ones(2, dtype=dtype))
    for key in actor:
        assert torch.equal(actor[key], before[key]), key
    assert pool(model, actor).shape == (2, 6, 4)


@pytest.mark.parametrize('mode', MODES)
def test_bidirectional_pool_matches_independent_token_loop_with_nonuniform_priors(mode):
    model, actor = sample(mode)
    expected = torch.zeros(2, 6, 4, dtype=torch.float64)
    # Reference operates only on supported tokens, without the production masks,
    # einsums, batched matrices or compare/combine module forwards.
    with torch.no_grad():
        for b in range(2):
            xs = actor['observation'][b, actor['token_mask'][b]]
            xp = actor['token_prior'][b, actor['token_mask'][b]]
            x = F.linear(xs, model.token_projection.weight, model.token_projection.bias)
            for c in range(6):
                if not actor['candidate_mask'][b, c]:
                    continue
                keep = actor['schema_mask'][b, c]
                y = F.linear(actor['schema_tokens'][b, c, keep], model.token_projection.weight, model.token_projection.bias)
                yp = actor['schema_prior'][b, c, keep]
                means = []
                for source, opposite, source_prior, opposite_prior in ((x, y, xp, yp), (y, x, yp, xp)):
                    compared = []
                    for token in source:
                        scores = [float(token.dot(other))/math.sqrt(4) if mode == 'aligned' else 0. for other in opposite]
                        maximum = max(scores)
                        weights = [math.exp(s-maximum)*float(p) for s, p in zip(scores, opposite_prior, strict=True)]
                        # Both priors in this fixture have exact double unit mass.
                        aligned = sum((w/math.fsum(weights))*other for w, other in zip(weights, opposite, strict=True))
                        features = torch.cat((token, aligned, token-aligned, token*aligned))
                        compared.append(F.linear(features, model.compare[0].weight, model.compare[0].bias).tanh())
                    means.append(sum(p*v for p, v in zip(source_prior, compared, strict=True)))
                expected[b, c] = F.linear(torch.cat(means), model.combine[0].weight, model.combine[0].bias).tanh()
    torch.testing.assert_close(pool(model, actor), expected, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('mode', MODES)
def test_shared_flat_scorer_and_branch_means_match_parent_formula(mode):
    model, actor = sample(mode)
    u = pool(model, actor)
    q = model.query_projection(actor['query']).tanh()[:, None].expand_as(u)
    c = model.candidate_projection(torch.where(actor['candidate_mask'][..., None], actor['candidates'], 0.)).tanh()
    lex = torch.where(actor['candidate_mask'][..., None], actor['lexical'], 0.)
    flags = F.one_hot(actor['candidate_types'].clamp(min=0), 5).double()
    flags = torch.where(actor['candidate_mask'][..., None], flags, 0.)
    features = torch.cat((u, q, c, u*q, u*c, q*c, lex, flags,
                          actor['previous_onehot'][..., None], torch.zeros(2, 6, 1, dtype=torch.float64)), -1)
    h = model.feature(features)
    z = model.head(h).squeeze(-1)
    expected = torch.full((2, 6), -torch.inf, dtype=torch.float64)
    for b in range(2):
        support = actor['candidate_mask'][b]
        ids = actor['candidate_types'][b].clamp(0, 2)
        gates = torch.stack([model.gate(h[b, support & (ids == k)].mean(0)).squeeze() for k in range(3)])
        expected[b, support] = (z[b, support]+gates[ids[support]]).log_softmax(-1)
    torch.testing.assert_close(model(**actor), expected)
    assert DialogueTokenAlignment._log_probabilities is DialogueTypedObservation._log_probabilities


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_single_token_sequences_make_modes_equal(dtype):
    mean, actor = sample('mean', dtype)
    aligned = DialogueTokenAlignment('aligned', 6, 4, 5).to(dtype)
    copy_initialization(mean, aligned)
    actor['observation'] = actor['observation'][:, :1]
    actor['token_mask'] = torch.ones(2, 1, dtype=torch.bool)
    actor['token_prior'] = torch.ones(2, 1, dtype=dtype)
    actor['schema_tokens'] = actor['schema_tokens'][:, :, :1]
    actor['schema_mask'] = actor['candidate_mask'][..., None]
    actor['schema_prior'] = actor['schema_mask'].to(dtype)
    torch.testing.assert_close(pool(mean, actor), pool(aligned, actor))
    torch.testing.assert_close(mean(**actor), aligned(**actor))


def test_zero_similarity_recovers_opposite_nonuniform_means_without_zero_comparison():
    mean, actor = sample('mean')
    with torch.no_grad():
        mean.token_projection.weight.zero_()
        mean.token_projection.weight[:, :4].copy_(torch.eye(4, dtype=torch.float64))
        mean.token_projection.bias.zero_()
    actor['observation'][..., 2:] = 0
    actor['schema_tokens'][..., :2] = 0
    actor['schema_tokens'][..., 4:] = 0
    aligned = DialogueTokenAlignment('aligned', 6, 4, 5).double()
    copy_initialization(mean, aligned)
    assert pool(mean, actor).abs().sum() > 0
    torch.testing.assert_close(pool(mean, actor), pool(aligned, actor), atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(mean(**actor), aligned(**actor), atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_all_input_parameter_gradients_finite_and_masked_gradients_zero(mode, dtype):
    model, actor = sample(mode, dtype, gradients=True)
    value = loss(model, actor)
    assert torch.isfinite(value)
    value.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name, value in actor.items():
        if value.is_floating_point():
            assert value.grad is not None, name
            assert torch.isfinite(value.grad).all(), name
    for name, mask in (('observation', 'token_mask'), ('token_prior', 'token_mask'),
                       ('schema_tokens', 'schema_mask'), ('schema_prior', 'schema_mask'),
                       ('candidates', 'candidate_mask'), ('lexical', 'candidate_mask')):
        assert (actor[name].grad[~actor[mask]] == 0).all(), name
    for name in ('token_projection.weight', 'compare.0.weight', 'combine.0.weight',
                 'query_projection.weight', 'candidate_projection.weight', 'feature.0.weight', 'head.weight', 'gate.weight'):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
    assert (model.feature[0].weight.grad[:, -1] == 0).all()


@pytest.mark.parametrize('mode', MODES)
def test_candidate_token_permutations_batch_independence_and_padding_poison(mode):
    model, actor = sample(mode)
    expected = model(**actor)
    changed = clone(actor)
    order = torch.tensor([5, 3, 1, 4, 0, 2])
    for key in ('candidates', 'candidate_mask', 'lexical', 'previous_onehot', 'candidate_types',
                'schema_tokens', 'schema_mask', 'schema_prior'):
        changed[key] = actor[key][:, order]
    torch.testing.assert_close(model(**changed), expected[:, order])
    changed = clone(actor)
    for key in ('observation', 'token_mask', 'token_prior'):
        changed[key] = actor[key][:, [3, 1, 0, 2]]
    for key in ('schema_tokens', 'schema_mask', 'schema_prior'):
        changed[key] = actor[key][:, :, [2, 0, 1]]
    torch.testing.assert_close(model(**changed), expected)
    for b in range(2):
        torch.testing.assert_close(model(**{k: v[b:b+1] for k, v in actor.items()}), expected[b:b+1])
    duplicate = torch.tensor([1, 0, 1])
    torch.testing.assert_close(model(**{k: v[duplicate] for k, v in actor.items()}), expected[duplicate])
    changed = clone(actor)
    for key, mask in (('observation', 'token_mask'), ('schema_tokens', 'schema_mask'),
                      ('candidates', 'candidate_mask'), ('lexical', 'candidate_mask')):
        changed[key][~actor[mask]] = 1e30
    assert torch.equal(model(**changed), expected)
    assert not any(isinstance(v, torch.Tensor) for v in vars(model).values())
    assert torch.equal(model(**actor), expected)


@pytest.mark.parametrize('mode', MODES)
def test_previous_value_is_used_directly_without_transition_or_current_target(mode):
    model, actor = sample(mode)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.feature[0].weight[0, -2] = 2
        model.head.weight[0, 0] = 1
    before = model(**actor)
    changed = clone(actor)
    changed['previous_onehot'][0] = F.one_hot(torch.tensor(0), 6).double()
    assert model(**changed)[0, 0] > before[0, 0]
    assert set(inspect.signature(model.forward).parameters) == {
        'observation', 'query', 'candidates', 'candidate_mask', 'lexical', 'previous_onehot',
        'candidate_types', 'token_mask', 'token_prior', 'schema_tokens', 'schema_mask', 'schema_prior'}
    assert not hasattr(model, '_advance')


def test_exact_paired_initialization_atomic_copy_and_compatible_parent_subset():
    torch.manual_seed(71)
    mean = DialogueTokenAlignment('mean', 6, 4, 5).double()
    after_mean = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    aligned = DialogueTokenAlignment('aligned', 6, 4, 5).double()
    assert torch.equal(after_mean, torch.random.get_rng_state())
    for key, value in mean.state_dict().items():
        assert torch.equal(value, aligned.state_dict()[key])
    with torch.no_grad():
        aligned.combine[0].bias.add_(1)
    copied = copy_initialization(mean, aligned)
    assert set(copied) == set(dict(mean.named_parameters()))
    for key, value in mean.state_dict().items():
        other = aligned.state_dict()[key]
        assert torch.equal(value, other) and value.data_ptr() != other.data_ptr()
    bad = DialogueTokenAlignment('aligned', 6, 4, 6).double()
    before = {k: v.clone() for k, v in bad.state_dict().items()}
    with pytest.raises(ValueError, match='Incompatible'):
        copy_initialization(mean, bad)
    assert all(torch.equal(v, bad.state_dict()[k]) for k, v in before.items())
    parent = DialogueTypedObservation('flat', 6, 4, 5).double()
    parent_pooling = parent.token_key.weight.detach().clone()
    assert copy_common_initialization(mean, parent) == list(COMMON_TENSORS)
    for key in COMMON_TENSORS:
        assert torch.equal(mean.state_dict()[key], parent.state_dict()[key])
    assert torch.equal(parent.token_key.weight, parent_pooling)


@pytest.mark.parametrize('mode', MODES)
def test_shape_work_matches_projection_comparison_call_shapes(mode):
    model, actor = sample(mode)
    projected, compared, combined = [], [], []
    hooks = [model.token_projection.register_forward_pre_hook(lambda _m, args: projected.append(tuple(args[0].shape))),
             model.compare.register_forward_pre_hook(lambda _m, args: compared.append(tuple(args[0].shape))),
             model.combine.register_forward_pre_hook(lambda _m, args: combined.append(tuple(args[0].shape)))]
    try:
        model(**actor)
    finally:
        for hook in hooks:
            hook.remove()
    work = model.work_counts(2, 6, 4, 3)
    assert projected == [(2, 4, 6)]+[(2, 3, 6)]*6
    assert sum(math.prod(s[:-1]) for s in projected) == work['context_projection_rows']+work['schema_projection_rows']
    assert sum(math.prod(s[:-1]) for s in compared) == work['comparison_rows']
    assert combined == [(2, 8)]*6
    assert work['pairwise_score_scalars'] == (144 if mode == 'aligned' else 0)
    assert work['largest_score_matrix_scalars'] == (24 if mode == 'aligned' else 0)
    assert all(len(s) < 4 for s in projected+compared+combined)


def test_configuration_has_no_unused_attention_or_turn_modules_and_registered_counts():
    model = DialogueTokenAlignment('mean')
    config = model.configuration()
    assert config['parameters'] == sum(p.numel() for p in model.parameters()) == 124482
    assert config['common_scorer_parameters'] == 75138
    assert config['token_comparison_parameters'] == 49344
    assert config['candidate_chunk_size'] == 1
    assert not {'turn_projection', 'token_key', 'evidence_query'} & set(dict(model.named_children()))
    assert 'Entropy' in config['parameter_count_scope']
    json.dumps(config, allow_nan=False)


@pytest.mark.parametrize('bad', ('schema_nan_padding', 'token_nan_padding', 'lexical_nan_padding', 'empty_context',
                                'empty_schema', 'schema_on_candidate_padding', 'schema_prior_padding',
                                'context_prior_padding', 'schema_prior_zero', 'schema_prior_mass',
                                'schema_prior_dtype', 'schema_shape', 'schema_mask_dtype',
                                'previous_fraction', 'previous_padding', 'type_padding', 'types_duplicate_none'))
def test_invalid_inputs_are_rejected_before_returning_probabilities(bad):
    model, actor = sample()
    if bad == 'schema_nan_padding': actor['schema_tokens'][0, 5, 0, 0] = torch.nan
    elif bad == 'token_nan_padding': actor['observation'][0, 1, 0] = torch.nan
    elif bad == 'lexical_nan_padding': actor['lexical'][0, 5, 0] = torch.nan
    elif bad == 'empty_context': actor['token_mask'][0] = False; actor['token_prior'][0] = 0
    elif bad == 'empty_schema': actor['schema_mask'][0, 0] = False; actor['schema_prior'][0, 0] = 0
    elif bad == 'schema_on_candidate_padding': actor['schema_mask'][0, 5, 0] = True; actor['schema_prior'][0, 5, 0] = 1
    elif bad == 'schema_prior_padding': actor['schema_prior'][0, 0, 1] = .1
    elif bad == 'context_prior_padding': actor['token_prior'][0, 1] = .1
    elif bad == 'schema_prior_zero': actor['schema_prior'][0, 0, 0] = 0
    elif bad == 'schema_prior_mass': actor['schema_prior'][0, 0] *= 1.1
    elif bad == 'schema_prior_dtype': actor['schema_prior'] = actor['schema_prior'].float()
    elif bad == 'schema_shape': actor['schema_tokens'] = actor['schema_tokens'][..., :-1]
    elif bad == 'schema_mask_dtype': actor['schema_mask'] = actor['schema_mask'].long()
    elif bad == 'previous_fraction': actor['previous_onehot'][0] = torch.tensor([.5, .5, 0, 0, 0, 0])
    elif bad == 'previous_padding': actor['previous_onehot'][0] = F.one_hot(torch.tensor(5), 6).double()
    elif bad == 'type_padding': actor['candidate_types'][0, 5] = 0
    elif bad == 'types_duplicate_none': actor['candidate_types'][0, 2] = 0
    with pytest.raises(ValueError):
        model(**actor)


@pytest.mark.parametrize('mode', MODES)
def test_large_projected_scores_reject_nonfinite_intermediates(mode):
    model, actor = sample(mode, torch.float32)
    with torch.no_grad():
        model.token_projection.weight.fill_(1e30)
    actor['observation'].fill_(1e30)
    with pytest.raises(ValueError, match='Nonfinite'):
        model(**actor)
