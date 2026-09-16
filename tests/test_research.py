import math

import pytest

from openjev.research.conformal import LabeledDecision, SplitConformal, semantic_group_entropy


def fit(alpha=.25):
    return SplitConformal.fit([
        LabeledDecision(str(i), {'a': p, 'b': 1-p}, 'a')
        for i, p in enumerate([.9, .8, .7])
    ], alpha=alpha, scorer_signature='model+prompt-v1', task_signature='task-v1')


def predict(model, probabilities, unit='test', scorer='model+prompt-v1'):
    return model.predict(probabilities, unit_id=unit, scorer_signature=scorer, task_signature='task-v1')


def test_conformal_finite_sample_rank_ties_and_empty_sets():
    model = fit()
    assert model.threshold == pytest.approx(.3)
    assert predict(model, {'a': .7, 'b': .3}) == ('a',)
    assert predict(model, {'a': .5, 'b': .5}) == ()
    assert predict(fit(alpha=.01), {'a': .5, 'b': .5}) == ('a', 'b')


def test_conformal_rejects_overlap_changed_model_and_invalid_scores():
    model = fit()
    for kwargs in [{'unit': '0'}, {'scorer': 'changed'}]:
        with pytest.raises(ValueError):
            predict(model, {'a': .7, 'b': .3}, **kwargs)
    with pytest.raises(ValueError):
        predict(model, {'a': float('nan'), 'b': .3})
    with pytest.raises(ValueError):
        predict(model, {'a': .7, 'b': .7})
    with pytest.raises(ValueError, match='unique'):
        SplitConformal.fit([LabeledDecision('same-episode', {'a': 1}, 'a')]*2,
                           alpha=.1, scorer_signature='m', task_signature='t')


def test_semantic_entropy_requires_explicit_complete_groups():
    probabilities = {'left-a': .25, 'left-b': .25, 'right': .5}
    assert semantic_group_entropy(probabilities, {'left-a': 'left', 'left-b': 'left', 'right': 'right'}) == pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        semantic_group_entropy(probabilities, {'left-a': 'left'})


def test_architectures_candidate_equivariance_state_and_gradients():
    torch = pytest.importorskip('torch')
    from openjev.research.architectures import LatentTransformer, RecurrentWorldModel, SparseCircuit
    torch.manual_seed(7)
    observation = torch.randn(2, 11)
    candidates = torch.randn(2, 6, 8)
    action = torch.randn(2, 3)
    order = [4, 2, 0, 5, 1, 3]
    for model in [SparseCircuit(11, 8), LatentTransformer(11, 8), RecurrentWorldModel(11, 3, 8)]:
        def call(c, state=None, model=model):
            if isinstance(model, RecurrentWorldModel):
                return model(observation, action, c, state)
            return model(observation, c, state)
        scores, state = call(candidates)
        reordered, _ = call(candidates[:, order])
        assert scores.shape == (2, 6)
        assert torch.allclose(reordered, scores[:, order], atol=1e-5)
        assert not torch.allclose(call(candidates, state)[0], scores)
        assert torch.allclose(call(candidates)[0], scores)  # reset restores initial output
        loss = scores.square().mean()
        if isinstance(model, RecurrentWorldModel):
            future = model.imagine(state, action)
            loss = loss + future['observation'].square().mean() + future['reward'].square().mean()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        if isinstance(model, SparseCircuit):
            assert torch.all(model.recurrent.weight.grad[model.mask == 0] == 0)
