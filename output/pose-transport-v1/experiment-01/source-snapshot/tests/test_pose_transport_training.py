import numpy as np
import torch
from train_pose_transport import load_data, loss_function, measure

from openjev.research.rigid_motion import so3_exp


def test_physical_loss_does_not_explode_on_nearconstant_cosine():
    p = torch.zeros(2, 25, 3)
    r = torch.eye(3).expand(2, 25, 3, 3).clone()
    predicted_p = p + torch.tensor([.1, 0., 0.])
    predicted_r = so3_exp(torch.tensor([.1, 0., 0.])).expand_as(r)
    torch.testing.assert_close(loss_function(predicted_p, predicted_r, p, r), torch.tensor(2.))
    m = measure(predicted_p, predicted_r, p, r)
    assert abs(m['position_rmse_m'] - .1) < 1e-6
    assert abs(m['rotation_rmse_rad'] - .1) < 1e-6


def test_physical_loss_has_finite_identity_gradient():
    p = torch.zeros(2, 25, 3, requires_grad=True)
    omega = torch.zeros(2, 25, 3, requires_grad=True)
    rotation = so3_exp(omega)
    loss_function(p, rotation, torch.zeros_like(p), torch.eye(3).expand_as(rotation)).backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(omega.grad).all()


def test_legacy_loader_restores_coordinates_and_preserves_ids(tmp_path):
    raw = np.zeros((2, 57, 9), dtype=np.float64)
    raw[:, :, :3] = [.2, .3, -.1]
    raw[:, :, 6:] = 1
    mean = np.arange(9, dtype=np.float64)
    scale = np.linspace(.1, .9, 9)
    np.savez(tmp_path / 'normalization.npz', obs_mean=mean, obs_scale=scale)
    np.savez(tmp_path / 'train.npz', obs=(raw-mean)/scale,
             actions=np.zeros((2, 56, 40), dtype=np.float32), source_ids=[4, 7], window_starts=[30, 80])
    p, r, a, ids = load_data(tmp_path, 'train')
    torch.testing.assert_close(p, torch.tensor(raw[..., :3], dtype=torch.float32))
    torch.testing.assert_close(r, torch.eye(3).expand(2, 57, 3, 3))
    assert p.dtype == r.dtype == a.dtype == torch.float32
    np.testing.assert_array_equal(ids, [[4, 30], [7, 80]])
