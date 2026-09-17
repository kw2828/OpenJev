import pytest
import torch

from openjev.research.bound_rules import BoundRuleNet, collate, ground


def score(context, questions, steps=32):
    return BoundRuleNet('fixed', steps)(collate([ground(context, questions)]))[0].tolist()


def test_entity_binding_and_multistep_reasoning():
    context = 'Alice is red. Bob is blue. Red people are warm. If someone is warm then they are nice.'
    q = ['Alice is nice.', 'Bob is nice.', 'Bob is not nice.']
    assert score(context, q) == [1., 0., 1.]
    assert score(context, q, steps=1) == [0., 0., 1.]


def test_negative_head_inhibits_and_negative_body_uses_closed_world():
    context = ('The cat is red. The cat is furry. If something is furry then it is not red. '
               'If something is not red then it is kind.')
    assert score(context, ['The cat is red.', 'The cat is kind.']) == [0., 1.]


def test_relations_preserve_argument_order():
    context = 'The cat visits the dog. If something visits the dog then the dog likes it.'
    assert score(context, ['The dog likes the cat.', 'The cat likes the dog.']) == [1., 0.]


def test_relevant_change_flips_and_irrelevant_entity_preserves_answer():
    context = 'Alice is red. Red people are warm.'
    assert score(context, ['Alice is warm.']) == [1.]
    assert score(context.replace('red.', 'blue.', 1), ['Alice is warm.']) == [0.]
    assert score(context+' Zoe is blue.', ['Alice is warm.']) == [1.]


def test_disjoint_batch_does_not_cross_contaminate_worlds():
    a = ground('A is red. Red things are warm.', ['A is warm.'])
    b = ground('A is blue. Red things are warm.', ['A is warm.'])
    assert BoundRuleNet('fixed')(collate([a, b]))[0].tolist() == [1., 0.]


def test_renaming_and_sentence_order_are_exact_invariants():
    text = 'The cat is red. Red things are blue. The dog is green.'
    assert score(text, ['The cat is blue.']) == score(
        'The fox is green. Red things are blue. The owl is red.', ['The owl is blue.'])


def test_missing_grammar_fails_explicitly():
    with pytest.raises(ValueError):
        ground('Alice probably wants a warm drink.', ['Alice is warm.'])


def test_learned_operator_has_finite_gradient_and_body_order_invariance():
    a = ground('A is red. A is blue. If someone is red and blue then they are warm.', ['A is warm.'])
    b = ground('A is red. A is blue. If someone is blue and red then they are warm.', ['A is warm.'])
    model = BoundRuleNet('learned', 6)
    pa, _ = model(collate([a]))
    pb, _ = model(collate([b]))
    torch.testing.assert_close(pa, pb)
    pa.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_operator_is_225_parameters():
    assert sum(p.numel() for p in BoundRuleNet().parameters()) == 225


def test_constructed_counterfactuals_are_balanced_and_require_binding():
    import runpy
    from pathlib import Path
    study = runpy.run_path(str(Path(__file__).parents[1]/'scripts/bound_rule_study.py'))
    worlds = study['challenge']()
    p, _ = study['predict'](BoundRuleNet('fixed', 32), study['graphs'](worlds))
    torch.testing.assert_close(p, study['targets'](worlds))
    for first, second in zip(worlds[::2], worlds[1::2], strict=True):
        assert sorted(first['context'].replace('.', ' .').split()) == sorted(
            second['context'].replace('.', ' .').split())
        assert first['questions'][0]['text'] == second['questions'][0]['text']
        assert first['questions'][0]['gold'] != second['questions'][0]['gold']
