import numpy as np
import pytest
import torch
from train_pose_adaptation import forecast, load_data, loss_function, make_model, predict

from openjev.research.pose_references import predict_reference
from openjev.research.rigid_motion import so3_exp


def sample():
    gen = torch.Generator().manual_seed(818)
    p = torch.randn(2, 57, 3, generator=gen) * .01
    r = so3_exp(torch.randn(2, 57, 3, generator=gen) * .04)
    a = torch.randn(2, 56, 40, generator=gen)
    return p, r, a


@pytest.mark.parametrize('variant', ['meta', 'meta_prior', 'static', 'static_adapt', 'public', 'gru'])
def test_prediction_never_reads_future_poses_or_later_actions(variant):
    p, r, a = sample()
    model = make_model(variant, torch.tensor([.01, .02, .005, .005]))
    old = predict(model, p, r, a, variant)
    altered_p, altered_r, altered_a = p.clone(), r.clone(), a.clone()
    altered_p[:, 32:] = float('nan')
    altered_r[:, 32:] = float('nan')
    altered_a[:, 41:] = 77
    new = predict(model, altered_p, altered_r, altered_a, variant)
    for x, y in zip(old, new, strict=True):
        torch.testing.assert_close(x[:, :10], y[:, :10], rtol=0, atol=0)


def test_initial_static_and_gru_match_valid_cv1():
    p, r, a = sample()
    expected = predict_reference('cv1', p, r, a)
    for variant in ['static', 'meta_prior', 'gru']:
        model = make_model(variant, torch.ones(4))
        actual = predict(model, p, r, a, variant)
        for x, y in zip(actual, expected, strict=True):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=3e-6)


def test_context_api_rejects_full_future_pose_input():
    p, r, a = sample()
    model = make_model('meta', torch.ones(4))
    with pytest.raises(ValueError, match='context'):
        forecast(model, p, r, a[:, :31], a[:, 31:], 'meta')


def test_static_meta_initial_weights_pair_exactly():
    models = []
    for v in ['meta', 'static']:
        torch.manual_seed(1101)
        models.append(make_model(v, torch.ones(4)))
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(tensor, models[1].state_dict()[name], rtol=0, atol=0)


def test_physical_loss_units_and_identity_gradient():
    p = torch.zeros(2, 25, 3, requires_grad=True)
    w = torch.zeros(2, 25, 3, requires_grad=True)
    r = so3_exp(w)
    loss = loss_function(p, r, torch.full_like(p, .1), r.detach())
    torch.testing.assert_close(loss, torch.tensor(3.))
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(w.grad).all()


def test_load_data_keeps_explicit_source_ids(tmp_path):
    x = np.zeros((2, 57, 9), dtype=np.float32)
    x[..., 6:] = 1
    np.savez(tmp_path / 'normalization.npz', obs_mean=np.zeros(9), obs_scale=np.ones(9))
    np.savez(tmp_path / 'train.npz', obs=x, actions=np.zeros((2, 56, 40), dtype=np.float32),
             source_ids=[1, 3], window_starts=[2, 4])
    p, r, a, ids = load_data(tmp_path, 'train')
    assert p.shape == (2, 57, 3) and r.shape == (2, 57, 3, 3) and a.shape == (2, 56, 40)
    np.testing.assert_array_equal(ids, [[1, 2], [3, 4]])
