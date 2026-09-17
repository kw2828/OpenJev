import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from openjev.research.rule_memory import MODES, RuleMemoryHead, english_inputs

STUDY = runpy.run_path(str(Path(__file__).parents[1]/'scripts/rule_memory_study.py'))


def inputs():
    torch.manual_seed(932)
    return torch.randn(3, 8), torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool)


def test_english_features_cannot_read_gold_depth_or_formal_proofs():
    world = {'context': 'A is red. Red things are warm.', 'proof': 'private',
             'questions': [{'text': 'A is warm.', 'gold': 'true', 'depth': 2}]}
    before = english_inputs(world)
    world['questions'][0].update(gold='false', depth=9, proof='changed')
    world['proof'] = 'changed'
    assert english_inputs(world) == before
    assert before == (['A is red.', 'Red things are warm.'], ['A is warm.'])


@pytest.mark.parametrize('mode', MODES)
def test_memory_permutation_and_masked_padding_leave_prediction_unchanged(mode):
    model = RuleMemoryHead(mode, input_dim=8, latent_dim=4).eval()
    q, m, mask = inputs()
    reference = model(q, m, mask)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    torch.testing.assert_close(reference, model(q, m[:, permutation], mask[:, permutation]))
    padded = torch.cat((m, torch.randn(3, 2, 8)*100), 1)
    padded_mask = torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)), 1)
    torch.testing.assert_close(reference, model(q, padded, padded_mask))


def test_question_only_control_cannot_use_world_information():
    model = RuleMemoryHead('question_only', input_dim=8, latent_dim=4)
    q, m, mask = inputs()
    torch.testing.assert_close(model(q, m, mask), model(q, -m*7, ~mask))


def test_tied_heads_have_identical_initial_parameters_and_one_hop_behavior():
    models = [RuleMemoryHead(m, input_dim=8, latent_dim=4) for m in
              ('single_read', 'recurrent_3', 'recurrent_6', 'coverage_3')]
    reference = models[0].state_dict()
    for model in models[1:]:
        assert all(torch.equal(reference[k], v) for k, v in model.state_dict().items())
        model.hops = 1
        torch.testing.assert_close(models[0](*inputs()), model(*inputs()))


@pytest.mark.parametrize('mode', MODES)
def test_parameter_accounting_matches_gradient_graph(mode):
    model = RuleMemoryHead(mode, input_dim=8, latent_dim=4)
    model(*inputs()).square().sum().backward()
    active = sum(p.numel() for p in model.parameters() if p.grad is not None)
    assert active == model.cost(5)['active_parameters']
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_empty_world_is_rejected():
    q, m, mask = inputs()
    mask[0] = False
    with pytest.raises(ValueError, match='unmasked'):
        RuleMemoryHead('recurrent_3', input_dim=8, latent_dim=4)(q, m, mask)


def test_cluster_bootstrap_resamples_whole_worlds_not_queries():
    # Every query within a world has the same outcome. Duplicating each query
    # changes a naive query bootstrap, but not a correctly grouped bootstrap.
    delta = np.array([1., -1., .5])
    ids = np.array([0, 1, 2])
    a = STUDY['world_interval'](delta, ids)
    b = STUDY['world_interval'](np.repeat(delta, 10), np.repeat(ids, 10))
    assert a == b


def test_only_audited_development_parts_are_allowed():
    assert STUDY['PARTS'] == ('train', 'dev_in', 'dev_shift')
    assert set(STUDY['EXPECTED']) == set(STUDY['PARTS'])
