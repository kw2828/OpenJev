"""Synthetic tensors only, isolated Torch seed410; no data/environment imports."""
import inspect
import math

import pytest
import torch

from openjev.research.action_filter_models import MODES, VARIANCE_FLOOR, ActionFilterModel


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def model(mode, *, readout_hidden=False, dtype=torch.float32):
    return ActionFilterModel(mode, 9, 4, readout_hidden=readout_hidden).to(dtype=dtype)


def snapshot(state):
    return {key: value.detach().clone() for key, value in state.items()}


def equal(actual, expected):
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in actual)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('hidden', [False, True])
def test_functional_shapes_and_no_storage_alias(mode, hidden):
    cell = model(mode, readout_hidden=hidden)
    initial = cell.initial(3)
    original = snapshot(initial)
    obs = torch.linspace(-1., 1., 27).reshape(3, 9)
    root = cell.assimilate(initial, obs)
    equal(initial, original)
    before = snapshot(root)
    advanced, prediction = cell.advance(root, torch.ones(3, 4))
    equal(root, before)
    assert prediction.shape == (3, 9) and torch.isfinite(prediction).all()
    for key in root:
        assert advanced[key].data_ptr() != root[key].data_ptr()
        assert root[key].data_ptr() != initial[key].data_ptr()
    obs.fill_(100.)
    equal(root, before)
    prediction.zero_()
    assert any(bool(value.abs().sum()) for value in advanced.values())


@pytest.mark.parametrize('mode', MODES)
def test_private_candidates_cannot_modify_root_or_other_branch(mode):
    cell = model(mode)
    root = cell.assimilate(cell.initial(2), torch.ones(2, 9))
    root_before = snapshot(root)
    left, pred = cell.advance(root, torch.full((2, 4), -.5))
    right, _ = cell.advance(root, torch.full((2, 4), .5))
    right_before = snapshot(right)
    for value in left.values():
        value.detach().fill_(99.)
    equal(root, root_before); equal(right, right_before)
    _, again = cell.advance(root, torch.full((2, 4), -.5))
    assert torch.equal(pred, again)


@pytest.mark.parametrize('mode', MODES)
def test_action_dependence_and_future_observation_absence(mode):
    cell = model(mode)
    root = cell.assimilate(cell.initial(2), torch.ones(2, 9))
    _, low = cell.advance(root, torch.full((2, 4), -1.))
    _, high = cell.advance(root, torch.full((2, 4), 1.))
    assert not torch.equal(low, high)
    assert tuple(inspect.signature(cell.advance).parameters) == ('state', 'action')
    # A future observation can change only its private posterior, not the
    # already-generated prior/prediction or another root's forecast.
    prior, forecast = cell.advance(root, torch.zeros(2, 4))
    saved = forecast.detach().clone()
    a = cell.assimilate(prior, torch.full((2, 9), 2.))
    b = cell.assimilate(prior, torch.full((2, 9), -2.))
    assert any(not torch.equal(a[key], b[key]) for key in a)
    assert torch.equal(forecast, saved)
    _, repeated = cell.advance(root, torch.zeros(2, 4))
    assert torch.equal(forecast, repeated)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_end_to_end_gradients_reach_real_observations_actions_and_modules(mode, dtype):
    cell = model(mode, dtype=dtype)
    obs = torch.linspace(-.9, .9, 2 * 4 * 9, dtype=dtype).reshape(2, 4, 9).requires_grad_()
    actions = torch.linspace(-.8, .8, 2 * 4 * 4, dtype=dtype).reshape(2, 4, 4).requires_grad_()
    state, loss = cell.initial(2), torch.zeros((), dtype=dtype)
    for t in range(4):
        state = cell.assimilate(state, obs[:, t])
        state, prediction = cell.advance(state, actions[:, t])
        loss = loss + (prediction - .3).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for value in (obs, actions):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_delta_hand_write_and_full_row_transport():
    cell = model('transported_delta', dtype=torch.float64)
    with torch.no_grad():
        for p in cell.parameters(): p.zero_()
        cell.key.bias[0] = 1.; cell.value.bias[0] = 1.
    root = cell.assimilate(cell.initial(1), torch.zeros(1, 9, dtype=torch.float64))
    expected = torch.zeros(1, 16, 4, dtype=torch.float64); expected[0, 0, 0] = .5
    assert torch.equal(root['memory'], expected)
    new, _ = cell.advance(root, torch.zeros(1, 4, dtype=torch.float64))
    # Three equally weighted operators: identity + uniform rows + uniform rows,
    # then retention=.5 and a zero action drive.
    transported = (.5 / 3) * (expected + 2 * expected.mean(1, keepdim=True))
    torch.testing.assert_close(new['memory'], transported, rtol=1e-14, atol=1e-15)


def test_convex_transport_and_drive_do_not_amplify_entrywise_bound():
    cell = model('transported_delta')
    state = {'memory': torch.linspace(-2., 2., 128).reshape(2, 16, 4)}
    for step in range(20):
        bound = max(1., float(state['memory'].abs().max().detach()))
        state, _ = cell.advance(state, torch.full((2, 4), (step - 10) / 2))
        assert float(state['memory'].abs().max().detach()) <= bound + 1e-6


def test_decay_omits_transport_work_and_preserves_shared_assimilation():
    full, decay = model('transported_delta'), model('decay_delta')
    shared = decay.state_dict()
    decay.load_state_dict({key: full.state_dict()[key].clone() for key in shared}, strict=True)
    assert not hasattr(decay, 'mixture') and not hasattr(decay, 'row_logits')
    assert full.parameter_and_state_counts()['active_parameters'] - decay.parameter_and_state_counts()['active_parameters'] == 512 + 15
    obs = torch.ones(2, 9)
    equal(full.assimilate(full.initial(2), obs), decay.assimilate(decay.initial(2), obs))
    state = {'memory': torch.arange(128, dtype=torch.float32).reshape(2, 16, 4) / 100}
    action = torch.zeros(2, 4)
    result, _ = decay.advance(state, action)
    expected = decay.retention(action).sigmoid()[:, :, None] * state['memory']
    torch.testing.assert_close(result['memory'], expected)


def test_diagonal_filter_hand_update_and_positive_uncertainty():
    cell = model('diagonal_filter', dtype=torch.float64)
    with torch.no_grad():
        cell.observation_mean.weight.zero_(); cell.observation_mean.bias.fill_(2.)
        cell.observation_noise.weight.zero_()
        cell.observation_noise.bias.fill_(math.log(math.expm1(.3 - VARIANCE_FLOOR)))
    state = {'mean': torch.zeros(2, 64, dtype=torch.float64),
             'variance': torch.full((2, 64), .7, dtype=torch.float64)}
    result = cell.assimilate(state, torch.zeros(2, 9, dtype=torch.float64))
    torch.testing.assert_close(result['mean'], torch.full((2, 64), 1.4, dtype=torch.float64))
    torch.testing.assert_close(result['variance'], torch.full((2, 64), .21, dtype=torch.float64))
    for step in range(20):
        result, _ = cell.advance(result, torch.full((2, 4), (step - 10.) * 10, dtype=torch.float64))
        result = cell.assimilate(result, torch.full((2, 9), step - 10., dtype=torch.float64))
        assert (result['variance'] > 0).all() and torch.isfinite(result['variance']).all()


@pytest.mark.parametrize('mode,parameters,state_size', [
    ('transported_delta', 1583, 64), ('decay_delta', 1056, 64),
    ('gru', 28425, 64), ('diagonal_filter', 2825, 128)])
def test_exact_counts_and_configuration(mode, parameters, state_size):
    cell = model(mode)
    counts = cell.parameter_and_state_counts()
    assert counts['registered_parameters'] == counts['active_parameters'] == parameters
    assert counts['state_scalars_per_case'] == state_size
    assert counts['state_bytes_per_case'] == state_size * 4
    assert sum(v.numel() for v in cell.initial(1).values()) == state_size
    config = cell.configuration(); config['mode'] = 'poison'
    assert cell.configuration()['mode'] == mode


@pytest.mark.parametrize('mode', MODES)
def test_rejects_shape_dtype_nonfinite_and_wrong_state(mode):
    cell = model(mode)
    state = cell.initial(2)
    with pytest.raises(ValueError): cell.assimilate(state, torch.zeros(2, 8))
    with pytest.raises(ValueError): cell.assimilate(state, torch.zeros(2, 9, dtype=torch.float64))
    with pytest.raises(ValueError): cell.assimilate(state, torch.full((2, 9), float('nan')))
    with pytest.raises(ValueError): cell.advance(state, torch.full((2, 4), float('inf')))
    with pytest.raises(ValueError): cell.advance({'other': torch.zeros(2, 64)}, torch.zeros(2, 4))
    with pytest.raises(ValueError): cell.initial(True)
    damaged = snapshot(state); damaged[next(iter(damaged))][0].fill_(float('nan'))
    with pytest.raises(ValueError): cell.advance(damaged, torch.zeros(2, 4))


@pytest.mark.parametrize('kwargs', [
    {'mode': 'unknown'}, {'obs_dim': 0}, {'action_dim': True},
    {'hidden_size': 32}, {'readout_hidden': 1}])
def test_rejects_invalid_configuration(kwargs):
    config = {'mode': 'gru', 'obs_dim': 9, 'action_dim': 4}
    config.update(kwargs)
    with pytest.raises(ValueError): ActionFilterModel(**config)
