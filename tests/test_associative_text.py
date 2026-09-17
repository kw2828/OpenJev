import pytest

torch = pytest.importorskip('torch')

from openjev.research.associative_text import AssociativeRouter, conformal_threshold


def router(anchor=.5):
    return AssociativeRouter(torch.tensor([[1., 0.], [.8, .2], [0., 1.], [.2, .8]]),
                             torch.tensor([0, 0, 1, 1]), 2, top_k=2, anchor=anchor)


def test_full_and_empty_gate_match_controls_and_count_actual_passes():
    model = router()
    q = torch.tensor([[1., 0.], [.1, 1.]])
    simple = model.predict(q, 'prototype')
    gated = model.predict(q, 'gated_recurrent', -1.)
    assert torch.equal(simple[0], gated[0]) and gated[3] == 0
    full = model.predict(q, 'recurrent')
    gated = model.predict(q, 'gated_recurrent', 2.)
    assert torch.equal(full[0], gated[0]) and gated[3] == 4
    assert model.predict(q, 'nearest')[3] == 2


def test_anchored_update_cannot_inflate_original_support():
    q = torch.tensor([[.3, .7], [-1., -.2]])
    model = router()
    _, original = model.recall(q)
    _, after = model.recall(q, recurrent=True)
    assert (after <= original+1e-7).all()
    no_update = router(anchor=1.)
    a, _ = no_update.recall(q)
    b, _ = no_update.recall(q, recurrent=True)
    assert torch.allclose(a, b)


def test_memory_order_and_query_batch_do_not_change_scores():
    a = router()
    order = torch.tensor([3, 1, 0, 2])
    b = AssociativeRouter(a.memory[order], a.labels[order], 2, top_k=2)
    q = torch.tensor([[.31, .7], [.99, .01]])
    assert torch.allclose(a.recall(q, True)[0], b.recall(q, True)[0])
    assert torch.allclose(a.recall(q, True)[0], torch.cat([a.recall(r[None], True)[0] for r in q]))


def test_conformal_small_sample_quantile_and_invalid_inputs():
    p = torch.tensor([[.8, .2], [.3, .7]])
    y = torch.tensor([0, 1])
    assert conformal_threshold(p, y, .1) == 1.
    assert conformal_threshold(p, y, .5) == pytest.approx(.3)
    with pytest.raises(ValueError):
        conformal_threshold(p*2, y)
    with pytest.raises(ValueError):
        AssociativeRouter(torch.zeros(4, 2), torch.tensor([0, 0, 1, 1]), 2, top_k=2)
