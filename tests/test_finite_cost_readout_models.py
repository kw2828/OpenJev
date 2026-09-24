"""Fabricated engineering-only checks of readout trainability and shared filters."""
import pytest
import torch

from openjev.research.finite_cost_readout_models import ARMS, make_model, prefix_predictions
from openjev.research.finite_prefix_learning import prefix_predictions as original_prefix_predictions
from openjev.research.finite_shared_filter_models import SharedFilterModel
from openjev.research.otto_observation_operator_model import _work

SEED = 932101


def public_prefix(*, found_step=None):
    prefix = torch.zeros((1, 9, 31), dtype=torch.float32)
    prefix[0, 0, 6] = prefix[0, 0, 9] = 1
    length = 9 if found_step is None else found_step + 1
    for step in range(1, length):
        prefix[0, step, (step + 1) % 4] = 1
        prefix[0, step, 8 if step == found_step else 4 + step % 4] = 1
    return prefix, torch.tensor([length], dtype=torch.int64)


def expected_costs():
    # Literal preferred decisions in the declared eight-state basis.
    preferred = [0, 1, 3, 2, 2, 3, 1, 0]
    return torch.tensor([[-.75 if d == preferred[s] else .25 for s in range(8)]
                         for d in range(4)], dtype=torch.float64)


def rollouts(model, horizon=8):
    prefix, lengths = public_prefix()
    actions = torch.tensor([[0, 1, 3, 2, 2, 1, 0, 3]], dtype=torch.int64)[:, :horizon]
    observations = torch.tensor([[0, 2, 1, 3, 1, 0, 4, 4]], dtype=torch.int64)[:, :horizon]
    return model.blind_rollout(prefix, lengths, actions), model.observed_rollout(prefix, lengths, actions, observations)


def test_cost_sign_centering_and_softened_initial_equality():
    models = {arm: make_model(arm, SEED) for arm in ARMS}
    exact = expected_costs()
    assert torch.equal(models['fixed_exact'].readout_matrix(), exact)
    assert torch.equal(models['fixed_softened'].readout_matrix(), .9 * exact)
    learned = models['learned_readout'].readout_matrix()
    torch.testing.assert_close(learned, .9 * exact, rtol=0, atol=1e-12)
    assert bool(((learned > -.75) & (learned < .25)).all())
    torch.testing.assert_close(learned.sum(0), torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-15)
    assert torch.equal(learned.argmin(0), exact.argmin(0))
    # Columns have the cost sign, not the reversed preference score sign.
    assert float(learned[0, 0]) == pytest.approx(-.675, abs=1e-15)
    assert float(learned[1, 0]) == pytest.approx(.225, abs=1e-15)


def test_pair_initialization_matches_frozen_shared_filter_without_global_rng_effects():
    rng = torch.random.get_rng_state().clone()
    original = SharedFilterModel('shared_filter', SEED)
    models = [make_model(arm, SEED) for arm in ARMS]
    assert torch.equal(rng, torch.random.get_rng_state())
    for model in models:
        assert torch.equal(model.reset_logits, original.reset_logits)
        assert torch.equal(model.observed_logits, original.observed_logits)
    for new, old in zip(rollouts(models[0]), rollouts(original), strict=True):
        for key in ('prefix_state', 'prior_states', 'cost_contrasts', 'survival_mass', 'found_increments'):
            assert torch.equal(new[key], old[key])
        assert new['work'] == {**old['work'], 'cost_head_softmax_calls': 0, 'cost_head_probability_rows': 0}


def test_softened_and_learned_full_initial_forecasts_match():
    softened, learned = make_model('fixed_softened', SEED), make_model('learned_readout', SEED)
    for first, second in zip(rollouts(softened), rollouts(learned), strict=True):
        for key in ('prefix_state', 'prior_states', 'posterior_states', 'cost_contrasts',
                    'survival_mass', 'found_increments', 'evidence', 'probabilities'):
            if key not in first or first[key] is None:
                continue
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=1e-12)


def combined_objective(model):
    blind, observed = rollouts(model, horizon=2)
    target = torch.tensor([[[.2, -.3, .4, -.3], [-.1, .3, -.4, .2]]], dtype=torch.float64)
    prefix, lengths = public_prefix()
    prefix_loss = prefix_predictions(model, prefix, lengths)['nll'].mean()
    return ((blind['cost_contrasts'] - target).square().mean()
            + (observed['cost_contrasts'] - target.flip(-1)).square().mean()
            + blind['survival_mass'].square().mean()
            - observed['probabilities'][0, 1, 2].log() + prefix_loss)


def test_raw_filter_gradients_match_and_only_learned_readout_adds_head_gradient():
    softened, learned = make_model('fixed_softened', SEED), make_model('learned_readout', SEED)
    a, b = combined_objective(softened), combined_objective(learned)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    a.backward()
    b.backward()
    for name in ('reset_logits', 'observed_logits'):
        ga, gb = getattr(softened, name).grad, getattr(learned, name).grad
        assert ga is not None and gb is not None and bool((ga != 0).any())
        torch.testing.assert_close(ga, gb, rtol=0, atol=1e-12)
    assert learned.cost_logits.grad is not None and bool((learned.cost_logits.grad != 0).any())
    assert softened.costs.grad is None
    # This claim is deliberately before whole-model gradient clipping: the
    # additional head gradient can change the clipping factor.


def test_prefix_nll_is_identical_and_never_touches_cost_head():
    prefix, lengths = public_prefix(found_step=4)
    model = make_model('learned_readout', SEED)
    result = prefix_predictions(model, prefix, lengths)
    old = original_prefix_predictions(SharedFilterModel('shared_filter', SEED), prefix, lengths)
    assert torch.equal(result['probabilities'], old['probabilities'])
    assert torch.equal(result['nll'], old['nll'])
    assert result['work'] == {**old['work'], 'cost_head_softmax_calls': 0, 'cost_head_probability_rows': 0}
    result['nll'].sum().backward()
    assert model.cost_logits.grad is None
    assert bool((model.reset_logits.grad != 0).any()) and bool((model.observed_logits.grad != 0).any())
    with torch.no_grad():
        model.cost_logits.add_(torch.arange(4, dtype=torch.float64)[:, None] / 3)
    changed = prefix_predictions(model, prefix, lengths)
    assert torch.equal(changed['probabilities'], result['probabilities'])
    assert torch.equal(changed['nll'], result['nll'])


@pytest.mark.parametrize('arm', ARMS)
def test_linear_expectation_identity_and_absorbed_zero(arm):
    model = make_model(arm, SEED)
    first = torch.tensor([[.1, .2, .05, .15, .1, .2, .05, .15]], dtype=torch.float64)
    second = first.flip(-1)
    mixture = .3 * first + .7 * second
    value = model._read(mixture, _work())
    expectation = .3 * model._read(first, _work()) + .7 * model._read(second, _work())
    torch.testing.assert_close(value, expectation, rtol=0, atol=1e-15)
    torch.testing.assert_close(model._read(.4 * first, _work()), .4 * model._read(first, _work()), rtol=0, atol=1e-15)
    assert torch.equal(model._read(torch.zeros_like(first), _work()), torch.zeros((1, 4), dtype=torch.float64))
    _, observed = rollouts(model)
    assert torch.equal(observed['cost_contrasts'][0, 7], torch.zeros(4, dtype=torch.float64))
    assert float(observed['survival_mass'][0, 7]) == 0
    assert torch.equal(observed['probabilities'][0, 7], torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64))


@pytest.mark.parametrize('arm', ARMS)
def test_actual_work_counts_head_normalizations_each_horizon(arm):
    model = make_model(arm, SEED)
    for result in rollouts(model):
        work = result['work']
        assert work['cost_readout_calls'] == 8
        assert work['cost_head_softmax_calls'] == (8 if arm == 'learned_readout' else 0)
        assert work['cost_head_probability_rows'] == (64 if arm == 'learned_readout' else 0)
        assert work['prefix_operator_softmax_calls'] == work['operator_observed_softmax_calls'] == 1
        assert all(type(value) is int and value >= 0 for value in work.values())


@pytest.mark.parametrize('arm,parameters,buffers', [('fixed_exact', 1088, 256), ('fixed_softened', 1088, 256),
                                                 ('learned_readout', 1120, 0)])
def test_actual_inventory_and_no_dead_head_buffer(arm, parameters, buffers):
    model = make_model(arm, SEED)
    metadata = model.parameter_metadata()
    assert metadata['arm'] == arm
    assert metadata['count'] == metadata['trainable_count'] == parameters
    assert metadata['parameter_bytes'] == parameters * 8 and metadata['buffer_bytes'] == buffers
    assert metadata['head_logits'] == (32 if arm == 'learned_readout' else 0)
    assert metadata['head_identifiable_degrees'] == (24 if arm == 'learned_readout' else 0)
    assert set(dict(model.named_parameters())) == {'reset_logits', 'observed_logits'} | (
        {'cost_logits'} if arm == 'learned_readout' else set())
    assert set(dict(model.named_buffers())) == (set() if arm == 'learned_readout' else {'costs'})
    assert all(value.dtype == torch.float64 and value.device.type == 'cpu' for value in model.parameters())
    assert metadata['privileged_readout_initialization'] is True and metadata['privileged_prefix'] is False


@pytest.mark.parametrize('arm', ARMS)
def test_optimizer_updates_only_owned_parameters_and_fixed_buffers_survive(arm):
    model = make_model(arm, SEED)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    parameters = {name: value.clone() for name, value in model.named_parameters()}
    combined_objective(model).backward()
    torch.optim.Adam(model.parameters(), lr=.003).step()
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    assert all(not torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    costs = model.readout_matrix()
    assert torch.isfinite(costs).all()
    torch.testing.assert_close(costs.sum(0), torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-15)


def test_current_observation_and_future_actions_cannot_change_earlier_prediction():
    model = make_model('learned_readout', SEED)
    prefix, lengths = public_prefix()
    actions = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    first = model.observed_rollout(prefix, lengths, actions, torch.tensor([[0, 1, 2]]))
    second = model.observed_rollout(prefix, lengths, actions, torch.tensor([[0, 3, 2]]))
    assert torch.equal(first['cost_contrasts'][:, :2], second['cost_contrasts'][:, :2])
    assert torch.equal(first['probabilities'][:, :2], second['probabilities'][:, :2])
    one = model.blind_rollout(prefix, lengths, actions)
    two = model.blind_rollout(prefix, lengths, torch.tensor([[0, 1, 3]]))
    assert torch.equal(one['cost_contrasts'][:, :2], two['cost_contrasts'][:, :2])


def test_no_oracle_input_or_hidden_prefix_features():
    model = make_model('learned_readout', SEED)
    prefix, lengths = public_prefix()
    with pytest.raises(TypeError):
        model.encode_prefix(prefix, lengths, oracle_prefix=torch.full((1, 8), .125))
    with pytest.raises(TypeError):
        prefix_predictions(model, prefix, lengths, oracle_prefix=torch.full((1, 8), .125))
    prefix[0, 2, 20] = 1
    with pytest.raises(ValueError):
        prefix_predictions(model, prefix, lengths)


def test_readout_export_is_owned_and_fixed_corruption_is_rejected():
    for arm in ARMS:
        model = make_model(arm, SEED)
        expected = model.readout_matrix().detach().clone()
        model.readout_matrix().detach().zero_()
        assert torch.equal(model.readout_matrix(), expected)
    model = make_model('fixed_softened', SEED)
    with torch.no_grad():
        model.costs[0, 0] += .01
    with pytest.raises(ValueError, match='unchanged declared fixed'):
        model.readout_matrix()


def test_positive_probability_may_round_cost_to_boundary_without_clipping():
    model = make_model('learned_readout', SEED)
    with torch.no_grad():
        model.cost_logits[0, 0] = -100.
    probability = model.cost_logits.softmax(0)[0, 0]
    assert 0 < float(probability) < 1e-30
    assert float(model.readout_matrix()[0, 0]) == .25


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), 10000.])
def test_nonfinite_or_saturated_learned_head_fails_without_clipping(invalid):
    model = make_model('learned_readout', SEED)
    with torch.no_grad():
        model.cost_logits[0, 0] = invalid
    with pytest.raises(ValueError):
        model.readout_matrix()


@pytest.mark.parametrize('arm,seed', [('other', SEED), ('fixed_exact', True), ('learned_readout', -1)])
def test_invalid_factory(arm, seed):
    with pytest.raises(ValueError):
        make_model(arm, seed)
