"""Small deterministic synthetic tensors, isolated seed410; no data/native calls."""
import inspect

import pytest
import torch

from openjev.research.residual_dynamics import ResidualDynamics, RLSAdapter


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def model(dtype=torch.float64):
    return ResidualDynamics(3, 2, 5, delta_scale=torch.tensor([.1, .2, .4], dtype=dtype), translation_dims=1)


def copied(state):
    return {key: value.detach().clone() for key, value in state.items()}


def same(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_initial_readout_is_exact_persistence_for_every_private_forecast(dtype):
    cell = model(dtype)
    obs = torch.tensor([[3., -.7, 2.]], dtype=dtype)
    state = cell.assimilate(cell.initial(1), obs)
    assert torch.equal(state['delta'], torch.zeros_like(obs))
    for step in range(10):
        state, prediction, features = cell.advance(state, torch.full((1, 2), step / 10., dtype=dtype))
        assert torch.equal(prediction, obs)
        assert torch.equal(state['obs'], obs)
        assert features['public'].shape == (1, 8)
        assert features['latent'].shape == (1, 6)
        assert features['bias'].tolist() == [[1.]]


def test_assimilation_uses_actual_previous_root_not_predicted_endpoint():
    cell = model()
    with torch.no_grad():
        cell.readout.bias.fill_(10.)
    initial_obs = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    root = cell.assimilate(cell.initial(1), initial_obs)
    prior, predicted, _ = cell.advance(root, torch.zeros(1, 2, dtype=torch.float64))
    assert not torch.equal(predicted, initial_obs)
    actual = torch.tensor([[1.2, 1.8, 3.4]], dtype=torch.float64)
    posterior = cell.assimilate(prior, actual)
    torch.testing.assert_close(posterior['delta'], torch.tensor([[2., -1., 1.]], dtype=torch.float64))
    assert torch.equal(posterior['previous_obs'], initial_obs)
    assert torch.equal(posterior['obs'], actual)
    assert not torch.allclose(posterior['delta'], (actual - predicted) / cell.delta_scale)
    later_prior, _, _ = cell.advance(posterior, torch.zeros(1, 2, dtype=torch.float64))
    assert torch.equal(later_prior['previous_obs'], actual)


def test_translation_invariance_includes_public_features_and_recurrent_states():
    cell = model()
    with torch.no_grad():
        cell.readout.weight.fill_(.07)
    left, right = cell.initial(1), cell.initial(1)
    translation = torch.tensor([[32., 0., 0.]], dtype=torch.float64)
    for step in range(3):
        obs = torch.tensor([[step * .25, .5, -.5]], dtype=torch.float64)
        left = cell.assimilate(left, obs)
        right = cell.assimilate(right, obs + translation)
        torch.testing.assert_close(left['hidden'], right['hidden'], rtol=1e-13, atol=1e-13)
        left, pred_left, f_left = cell.advance(left, torch.ones(1, 2, dtype=torch.float64))
        right, pred_right, f_right = cell.advance(right, torch.ones(1, 2, dtype=torch.float64))
        torch.testing.assert_close(pred_right - pred_left, translation, rtol=1e-13, atol=1e-13)
        for key in f_left:
            torch.testing.assert_close(f_left[key], f_right[key], rtol=1e-13, atol=1e-13)


def test_state_inputs_predictions_and_sibling_branches_have_independent_storage():
    cell = model()
    obs = torch.ones(2, 3, dtype=torch.float64)
    initial = cell.initial(2)
    saved_initial = copied(initial)
    root = cell.assimilate(initial, obs)
    saved_root = copied(root)
    obs.zero_()
    same(root, saved_root)
    same(initial, saved_initial)
    left, prediction, features = cell.advance(root, torch.zeros(2, 2, dtype=torch.float64))
    right, _, _ = cell.advance(root, torch.ones(2, 2, dtype=torch.float64))
    saved_right, saved_left = copied(right), copied(left)
    prediction.detach().fill_(99.)
    for value in features.values():
        value.detach().fill_(19.)
    same(left, saved_left)
    for value in left.values():
        value.detach().fill_(1)
    same(root, saved_root)
    same(right, saved_right)


def test_feature_formulas_are_root_public_and_prior_latent():
    cell = model()
    root = cell.assimilate(cell.initial(2), torch.tensor([[1., 2., 3.], [2., 0., 0.]], dtype=torch.float64))
    action = torch.tensor([[.2, .5], [-1., 1.]], dtype=torch.float64)
    prior, _, features = cell.advance(root, action)
    raw = torch.cat((root['obs'][:, 1:], root['delta'], action), -1)
    torch.testing.assert_close(features['public'][:, :-1], torch.nn.functional.normalize(raw, dim=-1))
    torch.testing.assert_close(features['latent'][:, :-1], torch.nn.functional.normalize(prior['hidden'], dim=-1))
    assert torch.equal(features['latent'][:, -1], torch.ones(2, dtype=torch.float64))
    assert tuple(inspect.signature(cell.advance).parameters) == ('state', 'action')


def test_future_observation_cannot_change_previously_computed_prediction():
    cell = model()
    root = cell.assimilate(cell.initial(1), torch.ones(1, 3, dtype=torch.float64))
    prior, prediction, features = cell.advance(root, torch.ones(1, 2, dtype=torch.float64))
    saved_prediction, saved_features = prediction.clone(), copied(features)
    cell.assimilate(prior, torch.full((1, 3), 100., dtype=torch.float64))
    cell.assimilate(prior, torch.full((1, 3), -100., dtype=torch.float64))
    assert torch.equal(prediction, saved_prediction)
    same(features, saved_features)


def test_initial_zero_head_then_nonzero_head_gradient_semantics():
    cell = model()
    obs = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    action = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    root = cell.assimilate(cell.initial(2), obs)
    _, prediction, _ = cell.advance(root, action)
    prediction.square().mean().backward()
    assert cell.readout.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None and not bool(parameter.grad.abs().sum())
               for parameter in cell.transition.parameters())
    cell.zero_grad(set_to_none=True)
    with torch.no_grad():
        cell.readout.weight.fill_(.1)
    obs.grad, action.grad = None, None
    root = cell.assimilate(cell.initial(2), obs)
    prior, prediction, _ = cell.advance(root, action)
    posterior = cell.assimilate(prior, obs + .2)
    _, prediction2, _ = cell.advance(posterior, action)
    (prediction.square().mean() + prediction2.square().mean()).backward()
    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert bool(parameter.grad.abs().sum()), name
    assert obs.grad.abs().sum() > 0 and action.grad.abs().sum() > 0


def test_action_changes_prior_hidden_and_nonzero_head_prediction():
    cell = model()
    with torch.no_grad():
        cell.readout.weight.fill_(.1)
    root = cell.assimilate(cell.initial(1), torch.ones(1, 3, dtype=torch.float64))
    left, p_left, _ = cell.advance(root, -torch.ones(1, 2, dtype=torch.float64))
    right, p_right, _ = cell.advance(root, torch.ones(1, 2, dtype=torch.float64))
    assert not torch.equal(left['hidden'], right['hidden'])
    assert not torch.equal(p_left, p_right)


def test_default_dimensions_exact_counts_and_scale_defensive_copy():
    scale = torch.ones(9)
    cell = ResidualDynamics(delta_scale=scale)
    scale.zero_()
    assert torch.equal(cell.delta_scale, torch.ones(9))
    counts = cell.parameter_and_state_counts()
    assert cell.observation_update.input_size == 15 and cell.transition.input_size == 55
    assert counts['feature_dims'] == {'public': 56, 'latent': 65, 'bias': 1}
    assert counts['active_parameters'] == 39369
    state = cell.initial(1)
    assert counts['state_bytes_per_case'] == sum(value.numel() * value.element_size() for value in state.values())
    cfg = cell.configuration()
    cfg['delta_scale'][0] = 99.
    assert cell.configuration()['delta_scale'][0] == 1.


@pytest.mark.parametrize('feature_dim', [1, 4, 8])
def test_rls_matches_batch_ridge_after_every_real_update(feature_dim):
    dtype = torch.float64
    adapter = RLSAdapter(feature_dim, 3, torch.tensor([.1, .2, .3], dtype=dtype))
    state = adapter.initial(2)
    xs = torch.sin(torch.arange(2 * 9 * feature_dim, dtype=dtype).reshape(2, 9, feature_dim) + .3)
    ys = torch.cos(torch.arange(2 * 9 * 3, dtype=dtype).reshape(2, 9, 3) + .7)
    for step in range(9):
        saved = copied(state)
        updated = adapter.observe(state, xs[:, step], ys[:, step])
        same(state, saved)
        state = updated
        x, y = xs[:, :step + 1], ys[:, :step + 1]
        normal = torch.eye(feature_dim, dtype=dtype).expand(2, -1, -1) + x.transpose(1, 2) @ x
        expected = torch.linalg.solve(normal, x.transpose(1, 2) @ y)
        torch.testing.assert_close(state['weights'], expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(state['covariance'], torch.linalg.inv(normal), rtol=1e-11, atol=1e-12)
        assert state['updates'].tolist() == [[step + 1], [step + 1]]
    correction = adapter.correct(state, xs[:, -1])
    expected_correction = (xs[:, -1].unsqueeze(1) @ state['weights']).squeeze(1) * adapter.delta_scale
    torch.testing.assert_close(correction, expected_correction)


def test_bias_rls_hand_formula_detach_and_correction_does_not_update():
    adapter = RLSAdapter(1, 2, torch.tensor([2., 3.], dtype=torch.float64))
    root = adapter.initial(1)
    phi = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[4., 6.]], dtype=torch.float64, requires_grad=True)
    assert adapter.correct(root, phi).tolist() == [[0., 0.]]
    left = adapter.observe(root, phi, target)
    right = adapter.observe(root, phi, -target)
    assert left['weights'].tolist() == [[[2., 3.]]]
    assert left['covariance'].tolist() == [[[.5]]]
    assert adapter.correct(left, phi).tolist() == [[4., 9.]]
    assert all(not value.requires_grad for value in left.values())
    saved = copied(left)
    for _ in range(3):
        correction = adapter.correct(left, phi)
        correction.zero_()
    same(left, saved)
    for value in left.values():
        value.fill_(1)
    assert right['weights'].tolist() == [[[-2., -3.]]]
    assert root['weights'].tolist() == [[[0., 0.]]]
    assert not list(adapter.parameters())
    assert adapter.parameter_and_state_counts()['state_bytes_per_case'] == 32


@pytest.mark.parametrize('scale', [torch.zeros(3), torch.tensor([1., -1., 1.]),
                                   torch.tensor([1., float('nan'), 1.]), torch.ones(2), torch.ones(3, dtype=torch.int64)])
def test_reject_invalid_scale(scale):
    with pytest.raises(ValueError):
        ResidualDynamics(3, 2, 5, delta_scale=scale)


@pytest.mark.parametrize('translation_dims', [-1, 4, True, 1.5])
def test_reject_invalid_translation_dims(translation_dims):
    with pytest.raises(ValueError):
        ResidualDynamics(3, 2, 5, delta_scale=torch.ones(3), translation_dims=translation_dims)


def test_reject_invalid_boundaries_without_mutation():
    cell = model()
    initial = cell.initial(1)
    with pytest.raises(ValueError, match='initial observation'):
        cell.advance(initial, torch.ones(1, 2, dtype=torch.float64))
    root = cell.assimilate(initial, torch.ones(1, 3, dtype=torch.float64))
    saved = copied(root)
    with pytest.raises(ValueError, match='action'):
        cell.advance(root, torch.full((1, 2), float('nan'), dtype=torch.float64))
    with pytest.raises(ValueError, match='observation'):
        cell.assimilate(root, torch.ones(1, 3))
    same(root, saved)
    adapter = RLSAdapter(2, 3, cell.delta_scale)
    state = adapter.initial(1)
    before = copied(state)
    with pytest.raises(ValueError, match='residual target'):
        adapter.observe(state, torch.ones(1, 2, dtype=torch.float64), torch.full((1, 3), float('inf'), dtype=torch.float64))
    same(state, before)
