"""Independent fabricated witnesses for causal observation-reliability filtering.

No world generator, empirical data, saved checkpoint or oracle state is used.
The scalar reference retains complete hidden paths instead of calling filter
helpers. Numerical fixtures use an engineering-only seed.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from openjev.research.finite_reliability_filter import ReliabilityFilter

ARMS = ('unchanged', 'global', 'static_bank', 'markov_bank', 'recurrent_bank', 'reset_bank')
TRAINABLE = ('global', 'markov_bank', 'recurrent_bank', 'reset_bank')
SEED = 949101
ATOL = 2e-12


def fields():
    """Strictly positive dyadic laws, unrelated to the scientific world law."""
    transition = torch.full((4, 8, 8), 1 / 16, dtype=torch.float64)
    emission = torch.full((4, 8), 1 / 16, dtype=torch.float64)
    hazard = torch.empty((4, 8), dtype=torch.float64)
    costs = torch.full((4, 8), 1 / 8, dtype=torch.float64)
    for state in range(8):
        emission[(3 * state + state // 4) % 4, state] += 3 / 4
        costs[(state + state // 4) % 4, state] -= 1 / 2
        for action in range(4):
            transition[action, (state + 2 * action + 1) % 8, state] += 1 / 2
            hazard[action, state] = 1 / 16 + ((action + 3 * state) % 4) / 32
    return {'transition': transition, 'emission': emission, 'hazard': hazard, 'costs': costs}


def sequence():
    return (torch.tensor([[0, 2, 1, 3], [3, 1, 2, 0]], dtype=torch.int64),
            torch.tensor([[1, 2, 0, 3, 1], [0, 3, 1, 2, 0]], dtype=torch.int64))


def scalar_paths(laws, actions, observations, *, q, prior=None, rate=0.):
    """Enumerate all mode/state histories for each observed event.

    In particular, do not replace the joint mode/state posterior by products of
    its marginals. At most two actions keep this independent oracle bounded.
    """
    t, o, h, c = (laws[name].tolist() for name in ('transition', 'emission', 'hazard', 'costs'))
    prior = [1 / len(q)] * len(q) if prior is None else prior
    paths = [(k, s, prior[k] / 8) for k in range(len(q)) for s in range(8)]
    output, states, costs = [], [], []
    for index, observation in enumerate(observations):
        if not paths:
            output.append([0., 0., 0., 0., 1.]); states.append([0.] * 8); costs.append([0.] * 4)
            continue
        if index == 0:
            branches = [(k, s, weight) for k, s, weight in paths]
            found = 0.
        else:
            action = actions[index - 1]
            branches, found_terms = [], []
            for old_mode, old_state, weight in paths:
                for mode, state in itertools.product(range(len(q)), range(8)):
                    mode_probability = (1 - rate) * (mode == old_mode) + rate * prior[mode]
                    joint = weight * mode_probability * t[action][state][old_state]
                    branches.append((mode, state, joint * (1 - h[action][state])))
                    found_terms.append(joint * h[action][state])
            found = math.fsum(found_terms)
        probabilities = [math.fsum(weight * ((1 - q[k]) * o[y][s] + q[k] / 4)
                                   for k, s, weight in branches) for y in range(4)] + [found]
        output.append(probabilities)
        evidence = probabilities[observation]
        assert evidence > 0
        if observation == 4:
            paths = []
        else:
            paths = [(k, s, weight * ((1 - q[k]) * o[observation][s] + q[k] / 4) / evidence)
                     for k, s, weight in branches]
        posterior = [math.fsum(weight for _, state, weight in paths if state == s) for s in range(8)]
        states.append(posterior)
        costs.append([math.fsum(c[d][s] * posterior[s] for s in range(8)) for d in range(4)])
    return {name: torch.tensor(value, dtype=torch.float64)[None]
            for name, value in zip(('probabilities', 'post_states', 'post_costs'), (output, states, costs), strict=True)}


def assert_outputs(left, right, *, atol=ATOL):
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=atol, msg=name)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('observations', ([1, 2, 0], [1, 2, 4]))
def test_complete_hidden_path_oracle(arm, observations):
    laws = fields()
    model = ReliabilityFilter(laws, arm, SEED)
    actions = [0, 2]
    q = [0.] if arm == 'unchanged' else [2 / 7] if arm == 'global' else [0., 2 / 7, 4 / 7]
    rate = 1 / (1 + math.exp(4)) if arm in ('markov_bank', 'recurrent_bank', 'reset_bank') else 0.
    expected = scalar_paths(laws, actions, observations, q=q, rate=rate)
    actual = model(torch.tensor([actions], dtype=torch.int64), torch.tensor([observations], dtype=torch.int64))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=ATOL, rtol=0, msg=name)


@pytest.mark.parametrize('arm', ARMS)
def test_full_probability_mass_readout_shapes_and_found_suffix(arm):
    model = ReliabilityFilter(fields(), arm, SEED)
    actions = torch.tensor([[1, 2, 0, 3], [0, 3, 1, 2]], dtype=torch.int64)
    observations = torch.tensor([[0, 2, 4, 4, 4], [1, 3, 0, 2, 1]], dtype=torch.int64)
    result = model(actions, observations)
    assert set(result) == {'probabilities', 'post_states', 'post_costs', 'reset_rates'}
    for name, shape in {'probabilities': (2, 5, 5), 'post_states': (2, 5, 8),
                        'post_costs': (2, 5, 4), 'reset_rates': (2, 4)}.items():
        assert result[name].shape == shape and result[name].dtype == torch.float64
        assert result[name].device.type == 'cpu' and torch.isfinite(result[name]).all()
    torch.testing.assert_close(result['probabilities'].sum(-1), torch.ones((2, 5), dtype=torch.float64), atol=ATOL, rtol=0)
    assert (result['probabilities'][:, 0, 4] == 0).all()
    assert (result['post_states'][0, 2:] == 0).all() and (result['post_costs'][0, 2:] == 0).all()
    assert torch.equal(result['probabilities'][0, 3:], torch.tensor([[0., 0., 0., 0., 1.]] * 2, dtype=torch.float64))
    torch.testing.assert_close(result['post_states'][1].sum(-1), torch.ones(5, dtype=torch.float64), atol=ATOL, rtol=0)
    torch.testing.assert_close(result['post_costs'], result['post_states'] @ fields()['costs'].T, atol=ATOL, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_current_prediction_and_gate_ignore_current_and_future_observations(arm):
    model = ReliabilityFilter(fields(), arm, SEED)
    actions, observations = sequence()
    if arm in ('recurrent_bank', 'reset_bank'):
        with torch.no_grad():
            model.reset_head.weight.copy_(torch.tensor([[.3, -.2, .1, .4]], dtype=torch.float64))
    original = model(actions, observations)
    changed = observations.clone(); changed[:, 2:] = torch.tensor([3, 0, 2])
    perturbed = model(actions, changed)
    # Observation index2 follows action index1. Its prediction and reset rate
    # must be published before that label is made available.
    torch.testing.assert_close(original['probabilities'][:, :3], perturbed['probabilities'][:, :3], atol=0, rtol=0)
    torch.testing.assert_close(original['reset_rates'][:, :2], perturbed['reset_rates'][:, :2], atol=0, rtol=0)
    torch.testing.assert_close(original['post_states'][:, :2], perturbed['post_states'][:, :2], atol=0, rtol=0)
    # Independent calls must start fresh, without persistent recurrent state.
    assert_outputs(model(actions, observations), original, atol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_prefix_truncation_and_blind_forks_cannot_see_future_labels(arm):
    model = ReliabilityFilter(fields(), arm, SEED)
    actions, observations = sequence()
    full = model(actions, observations)
    short = model(actions[:, :2], observations[:, :3])
    for name in ('probabilities', 'post_states', 'post_costs'):
        torch.testing.assert_close(full[name][:, :3], short[name], atol=0, rtol=0)
    torch.testing.assert_close(full['reset_rates'][:, :2], short['reset_rates'], atol=0, rtol=0)
    forks = torch.tensor([0, 1, 3, 2, 0, 2, 1, 3], dtype=torch.int64).expand(2, 5, 8).clone()
    result = model.blind_forks(full['post_states'], forks)
    truncated = model.blind_forks(short['post_states'], forks[:, :3])
    assert set(result) == {'costs', 'survival'}
    assert result['costs'].shape == (2, 5, 8, 4) and result['survival'].shape == (2, 5, 8)
    for name in result:
        torch.testing.assert_close(result[name][:, :3], truncated[name], atol=0, rtol=0)
    changed = observations.clone(); changed[:, 3:] = torch.tensor([0, 2])
    other = model(actions, changed)
    alternative = model.blind_forks(other['post_states'], forks)
    for name in result:
        torch.testing.assert_close(result[name][:, :3], alternative[name][:, :3], atol=0, rtol=0)


def test_blind_forks_match_independent_scalar_surviving_mass_and_absorb():
    laws = fields(); model = ReliabilityFilter(laws, 'static_bank', SEED)
    states = torch.tensor([[[.5, .25, .125, .125, 0, 0, 0, 0], [0.] * 8]], dtype=torch.float64)
    actions = torch.tensor([[[0, 3, 1, 2, 2, 1, 3, 0]] * 2], dtype=torch.int64)
    actual = model.blind_forks(states, actions)
    current = states[0, 0].tolist(); expected_costs, expected_mass = [], []
    t, h, c = (laws[name].tolist() for name in ('transition', 'hazard', 'costs'))
    for action in actions[0, 0].tolist():
        current = [(1 - h[action][j]) * math.fsum(t[action][j][i] * current[i] for i in range(8)) for j in range(8)]
        expected_mass.append(math.fsum(current))
        expected_costs.append([math.fsum(c[d][j] * current[j] for j in range(8)) for d in range(4)])
    torch.testing.assert_close(actual['costs'][0, 0], torch.tensor(expected_costs, dtype=torch.float64), atol=ATOL, rtol=0)
    torch.testing.assert_close(actual['survival'][0, 0], torch.tensor(expected_mass, dtype=torch.float64), atol=ATOL, rtol=0)
    assert (actual['costs'][0, 1] == 0).all() and (actual['survival'][0, 1] == 0).all()


def test_markov_and_both_recurrent_initial_functions_match():
    actions, observations = sequence()
    models = {arm: ReliabilityFilter(fields(), arm, SEED) for arm in ('markov_bank', 'recurrent_bank', 'reset_bank')}
    outputs = {arm: model(actions, observations) for arm, model in models.items()}
    for arm in ('recurrent_bank', 'reset_bank'):
        assert_outputs(outputs[arm], outputs['markov_bank'])
    for name, value in models['recurrent_bank'].state_dict().items():
        assert torch.equal(value, models['reset_bank'].state_dict()[name]), name
    torch.testing.assert_close(outputs['markov_bank']['reset_rates'], torch.full((2, 4), 1 / (1 + math.exp(4)), dtype=torch.float64), atol=4 * torch.finfo(torch.float64).eps, rtol=0)


def test_reset_hidden_control_changes_later_gates_once_readout_is_nonzero():
    recurrent = ReliabilityFilter(fields(), 'recurrent_bank', SEED)
    reset = ReliabilityFilter(fields(), 'reset_bank', SEED)
    with torch.no_grad():
        # Explicit parameters make recurrent memory, rather than a lucky random
        # initializer, responsible for the witness.
        for model in (recurrent, reset):
            model.gru.weight_ih.fill_(.07); model.gru.weight_hh.fill_(.11)
            model.gru.bias_ih.zero_(); model.gru.bias_hh.zero_()
            model.reset_head.weight.copy_(torch.tensor([[.3, -.2, .1, .4]], dtype=torch.float64))
    actions, observations = sequence()
    first, second = recurrent(actions, observations), reset(actions, observations)
    torch.testing.assert_close(first['reset_rates'][:, 0], second['reset_rates'][:, 0], atol=0, rtol=0)
    assert float((first['reset_rates'][:, 1:] - second['reset_rates'][:, 1:]).detach().abs().max()) > 1e-8
    assert float((first['probabilities'][:, 2:] - second['probabilities'][:, 2:]).detach().abs().max()) > 1e-10


@pytest.mark.parametrize('arm', TRAINABLE)
def test_finite_gradients_and_adam_update_without_field_updates(arm):
    model = ReliabilityFilter(fields(), arm, SEED)
    actions, observations = sequence()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = model(actions, observations)
        likelihood = result['probabilities'].gather(-1, observations[..., None]).squeeze(-1)
        objective = -likelihood.log().mean() + .03 * result['post_costs'].square().mean()
        objective.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert any(not torch.equal(value, initial[name]) for name, value in model.named_parameters())
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())


@pytest.mark.parametrize('arm', ('global', 'markov_bank', 'recurrent_bank', 'reset_bank'))
def test_predictive_likelihood_gradient_central_difference(arm):
    model = ReliabilityFilter(fields(), arm, SEED)
    if arm in ('recurrent_bank', 'reset_bank'):
        with torch.no_grad():
            model.reset_head.weight.copy_(torch.tensor([[.2, -.1, .3, -.2]], dtype=torch.float64))
    actions, observations = sequence()
    name = {'global': 'global_logit', 'markov_bank': 'reset_logit',
            'recurrent_bank': 'reset_head.weight', 'reset_bank': 'reset_head.weight'}[arm]
    parameter = dict(model.named_parameters())[name]
    direction = torch.full_like(parameter, .37)

    def objective():
        out = model(actions, observations)
        return -out['probabilities'].gather(-1, observations[..., None]).log().mean()

    objective().backward()
    analytical = float((parameter.grad * direction).sum())
    saved = parameter.detach().clone(); step = 1e-5
    with torch.no_grad():
        parameter.copy_(saved + step * direction); plus = float(objective())
        parameter.copy_(saved - step * direction); minus = float(objective())
        parameter.copy_(saved)
    # Central difference is O(h^2), with a conservative float64 subtraction
    # allowance proportional to epsilon/h, independently of observed errors.
    numerical = (plus - minus) / (2 * step)
    assert analytical == pytest.approx(numerical, rel=2e-6, abs=2e-8)


@pytest.mark.parametrize('arm', ARMS)
def test_batch_partition_permutation_and_latent_relabeling(arm):
    laws = fields(); model = ReliabilityFilter(laws, arm, SEED)
    actions, observations = sequence()
    output = model(actions, observations)
    split = [model(actions[i:i + 1], observations[i:i + 1]) for i in range(2)]
    assert_outputs(output, {key: torch.cat([item[key] for item in split]) for key in output})
    reordered = model(actions.flip(0), observations.flip(0))
    assert_outputs(output, {key: value.flip(0) for key, value in reordered.items()})
    permutation = torch.tensor([3, 7, 1, 6, 0, 5, 2, 4])
    other_fields = {'transition': laws['transition'][:, permutation][:, :, permutation],
                    'emission': laws['emission'][:, permutation], 'hazard': laws['hazard'][:, permutation],
                    'costs': laws['costs'][:, permutation]}
    other = ReliabilityFilter(other_fields, arm, SEED)(actions, observations)
    for key in ('probabilities', 'post_costs', 'reset_rates'):
        torch.testing.assert_close(output[key], other[key], atol=ATOL, rtol=0)
    torch.testing.assert_close(output['post_states'][..., permutation], other['post_states'], atol=ATOL, rtol=0)


@pytest.mark.parametrize('arm,count,state_count,mode_count', [
    ('unchanged', 0, 8, 1), ('global', 1, 8, 1), ('static_bank', 0, 24, 3),
    ('markov_bank', 4, 24, 3), ('recurrent_bank', 164, 28, 3), ('reset_bank', 164, 28, 3)])
def test_parameter_ownership_counts_and_local_rng(arm, count, state_count, mode_count):
    laws = fields()
    rng = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    model = ReliabilityFilter(laws, arm, SEED)
    assert torch.equal(torch.random.get_rng_state(), rng)
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert sum(p.numel() for p in model.parameters()) == model.parameter_count == count
    assert model.state_count == state_count and model.mode_count == mode_count
    assert all(p.dtype == torch.float64 and p.device.type == 'cpu' for p in model.parameters())
    actions, observations = sequence()
    expected = model(actions, observations)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for value in laws.values():
        value.zero_()
    assert_outputs(model(actions, observations), expected, atol=0)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_unclipped_event_likelihood_below_feature_floor_and_impossible_event():
    laws = fields()
    laws['emission'][0] = 1e-15
    laws['emission'][1:] = (1 - 1e-15) / 3
    model = ReliabilityFilter(laws, 'unchanged', SEED)
    out = model(torch.tensor([[0]], dtype=torch.int64), torch.tensor([[0, 1]], dtype=torch.int64))
    torch.testing.assert_close(out['probabilities'][0, 0, 0], torch.tensor(1e-15, dtype=torch.float64), atol=1e-29, rtol=0)
    laws['emission'][0].zero_(); laws['emission'][1:] = 1 / 3
    with pytest.raises(ValueError):
        ReliabilityFilter(laws, 'unchanged', SEED)(torch.tensor([[0]], dtype=torch.int64), torch.tensor([[0, 1]], dtype=torch.int64))


@pytest.mark.parametrize('bad', ('found_reset', 'found_then_odor', 'wrong_length', 'float_actions', 'bad_action', 'bad_observation'))
def test_invalid_public_sequences_fail(bad):
    model = ReliabilityFilter(fields(), 'unchanged', SEED)
    actions, observations = sequence()
    if bad == 'found_reset':
        observations[:, 0] = 4
    elif bad == 'found_then_odor':
        observations[0, 1] = 4
    elif bad == 'wrong_length':
        observations = observations[:, :-1]
    elif bad == 'float_actions':
        actions = actions.to(torch.float64)
    elif bad == 'bad_action':
        actions[0, 0] = 4
    else:
        observations[0, 0] = 5
    with pytest.raises(ValueError):
        model(actions, observations)


@pytest.mark.parametrize('bad', ('nan', 'negative', 'not_stochastic', 'wrong_dtype', 'hazard_bound', 'cost_shape'))
def test_invalid_frozen_fields_fail(bad):
    laws = fields()
    if bad == 'nan':
        laws['costs'][0, 0] = float('nan')
    elif bad == 'negative':
        laws['transition'][0, 0, 0] = -.1
    elif bad == 'not_stochastic':
        laws['emission'][0, 0] += .1
    elif bad == 'wrong_dtype':
        laws['emission'] = laws['emission'].float()
    elif bad == 'hazard_bound':
        laws['hazard'][0, 0] = 1.01
    else:
        laws['costs'] = laws['costs'][:, :-1]
    with pytest.raises(ValueError):
        ReliabilityFilter(laws, 'unchanged', SEED)


def test_bank_and_prior_mean_emission_match_initial_prediction_and_state():
    actions = torch.empty((2, 0), dtype=torch.int64)
    observations = torch.tensor([[0], [2]], dtype=torch.int64)
    bank = ReliabilityFilter(fields(), 'static_bank', SEED)(actions, observations)
    global_model = ReliabilityFilter(fields(), 'global', SEED)(actions, observations)
    assert_outputs(bank, global_model)


def test_recurrent_reset_features_are_conditional_likelihood_and_posterior():
    laws = fields()
    laws['emission'][:] = torch.tensor([.625, .125, .125, .125], dtype=torch.float64)[:, None]
    model = ReliabilityFilter(laws, 'recurrent_bank', SEED)
    prior = torch.tensor([1 / 7, 2 / 7, 4 / 7], dtype=torch.float64)
    with torch.no_grad():
        model.prior_logits.copy_(prior.log())
    captured = []
    hook = model.gru.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    try:
        model(torch.empty((1, 0), dtype=torch.int64), torch.tensor([[0]], dtype=torch.int64))
    finally:
        hook.remove()
    likelihood = torch.tensor([(1 - q) * .625 + q / 4 for q in (0., 2 / 7, 4 / 7)], dtype=torch.float64)
    evidence = (prior * likelihood).sum()
    expected = torch.cat((likelihood.log(), prior * likelihood / evidence, -evidence.log().reshape(1)))
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][0], expected, atol=ATOL, rtol=0)


def test_no_oracle_noise_or_regime_argument_and_public_inputs_unchanged():
    model = ReliabilityFilter(fields(), 'recurrent_bank', SEED)
    actions, observations = sequence()
    before_actions, before_observations = actions.clone(), observations.clone()
    for name in ('epsilon', 'regime', 'oracle_prefix', 'true_states'):
        with pytest.raises(TypeError, match=name):
            model(actions, observations, **{name: None})
    result = model(actions, observations)
    forks = torch.zeros((2, 5, 8), dtype=torch.int64)
    before_states = result['post_states'].detach().clone()
    before_forks = forks.clone()
    model.blind_forks(result['post_states'], forks)
    assert torch.equal(actions, before_actions) and torch.equal(observations, before_observations)
    assert torch.equal(result['post_states'], before_states) and torch.equal(forks, before_forks)


def test_actions_after_found_cannot_change_any_output():
    model = ReliabilityFilter(fields(), 'recurrent_bank', SEED)
    actions = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    observations = torch.tensor([[1, 4, 4, 4, 4]], dtype=torch.int64)
    original = model(actions, observations)
    changed = actions.clone(); changed[:, 1:] = torch.tensor([3, 0, 1])
    assert_outputs(original, model(changed, observations), atol=0)


@pytest.mark.parametrize('bad', ('subnormalized', 'negative', 'bad_action', 'wrong_horizon'))
def test_invalid_fork_inputs_fail(bad):
    model = ReliabilityFilter(fields(), 'unchanged', SEED)
    states = torch.full((1, 1, 8), 1 / 8, dtype=torch.float64)
    actions = torch.zeros((1, 1, 8), dtype=torch.int64)
    if bad == 'subnormalized':
        states *= .5
    elif bad == 'negative':
        states[0, 0, 0] = -.125; states[0, 0, 1] = .375
    elif bad == 'bad_action':
        actions[0, 0, 0] = 4
    else:
        actions = actions[..., :-1]
    with pytest.raises(ValueError):
        model.blind_forks(states, actions)
