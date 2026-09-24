"""Fabricated graph-sharing qualification against immutable separate routes.

No world generator, saved checkpoint, empirical array or timing assertion.
Absolute/relative tolerances are1e-10 for values and1e-9 for gradients,
clipped gradients and the short Adam trajectory. No bitwise-training claim.
"""

import copy

import pytest
import torch

from openjev.research import finite_joint_reuse as reuse
from openjev.research import finite_rounded_models as frozen
from openjev.research.finite_observation_models import objective

SEED = 943101
ARMS = ('original_free', 'matched_free', 'rounded')
VALUE_TOL = 1e-10
GRAD_TOL = 1e-9


def case(horizon=2, endings=(None, 1, None, 3, 8)):
    prefix = torch.zeros((len(endings), 9, 31), dtype=torch.float32)
    lengths = torch.tensor([9 if value is None else value + 1 for value in endings], dtype=torch.int64)
    for row, end in enumerate(endings):
        prefix[row, 0, 4 + row % 4] = prefix[row, 0, 9] = 1
        for step in range(1, int(lengths[row])):
            prefix[row, step, (row + step) % 4] = 1
            prefix[row, step, 8 if step == end else 4 + (2 * row + step) % 4] = 1
    positions = torch.tensor([i for i, end in enumerate(endings) if end is None], dtype=torch.int64)
    eligible = len(positions)
    actions = (torch.arange(eligible * horizon).reshape(eligible, horizon) + 1).remainder(4)
    observations = (actions + 1).remainder(4)
    if eligible:
        observations[0, min(1, horizon - 1):] = 4
    if eligible > 1 and horizon > 2:
        observations[1, -1] = 4
    target_cost = torch.tensor([.17, -.13, .04, -.08], dtype=torch.float64)
    blind_costs = target_cost.expand(eligible, horizon, 4).clone()
    observed_costs = target_cost.flip(0).expand(eligible, horizon, 4).clone()
    blind_survival = torch.linspace(.83, .31, horizon, dtype=torch.float64).expand(eligible, horizon).clone()
    observed_survival = torch.full((eligible, horizon), .91, dtype=torch.float64)
    probabilities = torch.tensor([.12, .23, .31, .27, .07], dtype=torch.float64).expand(eligible, horizon, 5).clone()
    for row in range(eligible):
        for step in range(1, horizon):
            if observations[row, step - 1] == 4:
                probabilities[row, step] = torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64)
                observed_costs[row, step] = 0
                observed_survival[row, step] = 0
    return {'prefix': prefix, 'lengths': lengths, 'endpoint_positions': positions,
            'actions': actions, 'observations': observations,
            'targets': {'blind_costs': blind_costs, 'observed_costs': observed_costs,
                       'blind_survival': blind_survival, 'observed_survival': observed_survival,
                       'observed_probabilities': probabilities},
            'total_attempts': 17 + len(endings), 'total_survivors': 7 + eligible,
            'total_events': 9 * (17 + len(endings))}


def model(arm, *, check=lambda: None):
    result = frozen.make_model(arm, SEED, check=check)
    # Deterministic asymmetric, nonuniform fixtures avoid testing only an
    # almost-uniform fixed point. No random input or oracle latent is used.
    with torch.no_grad():
        for index, parameter in enumerate(result.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
            parameter.add_(((values * (index + 3)).remainder(17) - 8) * .07)
    return result


def separate(model_value, values):
    positions = values['endpoint_positions']
    eligible = len(positions)
    loss = sum(parameter.sum() * 0 for parameter in model_value.parameters())
    blind = observed = None
    if eligible:
        prefix, lengths = values['prefix'][positions], values['lengths'][positions]
        blind = model_value.blind_rollout(prefix, lengths, values['actions'])
        observed = model_value.observed_rollout(prefix, lengths, values['actions'], values['observations'])
        loss = loss + objective(blind, observed, values['targets']) * eligible / values['total_survivors']
    prefix_result = frozen.prefix_predictions(model_value, values['prefix'], values['lengths'])
    loss = (loss + prefix_result['nll'].sum() / values['total_events']) * values['total_attempts'] / len(values['prefix'])
    return {'loss': loss, 'blind': blind, 'observed': observed, 'prefix': prefix_result}


def assert_values(first, second):
    for name in ('loss', 'blind', 'observed', 'prefix'):
        left, right = first[name], second[name]
        if isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                if key == 'work':
                    continue
                if left[key] is None:
                    assert right[key] is None
                else:
                    torch.testing.assert_close(left[key], right[key], atol=VALUE_TOL, rtol=VALUE_TOL)
        elif left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, atol=VALUE_TOL, rtol=VALUE_TOL)


def assert_gradients(first, second):
    for name, parameter in first.named_parameters():
        other = dict(second.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None
        assert parameter.grad.dtype == other.grad.dtype == torch.float64
        torch.testing.assert_close(parameter.grad, other.grad, atol=GRAD_TOL, rtol=GRAD_TOL)


def assert_optimizer(first, second):
    a, b = first.state_dict(), second.state_dict()
    assert a['param_groups'] == b['param_groups']
    assert a['state'].keys() == b['state'].keys()
    for key in a['state']:
        assert a['state'][key].keys() == b['state'][key].keys()
        for name, value in a['state'][key].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, b['state'][key][name], atol=GRAD_TOL, rtol=GRAD_TOL)
            else:
                assert value == b['state'][key][name]


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('horizon', [1, 2, 8])
def test_values_and_all_raw_gradients_match_frozen_routes(arm, horizon):
    old, new = model(arm), model(arm)
    values = case(horizon)
    expected, actual = separate(old, values), reuse.joint_objective(new, **values)
    assert_values(actual, expected)
    expected['loss'].backward()
    actual['loss'].backward()
    assert_gradients(old, new)
    for value in actual['prefix']['nll'][1, 2:]:
        assert value == 0
    assert actual['prefix']['nll'][1, 1] > 0  # Found event is scored, then padded.


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('endings', [(None,), (None, 1, None), tuple(None for _ in range(64)), (1, 4, 8)])
def test_partial_full_and_zero_endpoint_batches_keep_global_denominators(arm, endings):
    old, new = model(arm), model(arm)
    values = case(2, endings)
    expected, actual = separate(old, values), reuse.joint_objective(new, **values)
    assert_values(actual, expected)
    expected['loss'].backward()
    actual['loss'].backward()
    assert_gradients(old, new)
    if not len(values['endpoint_positions']):
        assert actual['blind'] is actual['observed'] is None
        assert torch.count_nonzero(new.cost_logits.grad) == 0
        assert not any(actual['work']['endpoint_prefix'].values())
        assert not any(actual['work']['blind'].values()) and not any(actual['work']['observed'].values())


@pytest.mark.parametrize('arm', ARMS)
def test_active_clipping_and_short_adam_trajectory_match(arm):
    old, new = model(arm), model(arm)
    optimizers = [torch.optim.Adam(value.parameters(), lr=.003) for value in (old, new)]
    for step in range(3):
        values = case(2, (None, None, 1) if step != 1 else (1, 3, 8))
        values.update(total_attempts=512, total_survivors=2, total_events=4608)
        values['targets']['blind_costs'] *= 100
        values['targets']['observed_costs'] *= 100
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected, actual = separate(old, values), reuse.joint_objective(new, **values)
        assert_values(actual, expected)
        expected['loss'].backward()
        actual['loss'].backward()
        assert_gradients(old, new)
        norms = [torch.nn.utils.clip_grad_norm_(value.parameters(), 5., error_if_nonfinite=True)
                 for value in (old, new)]
        if step == 0:
            assert norms[0] > 5 and norms[1] > 5
        assert_gradients(old, new)
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter, dict(new.named_parameters())[name], atol=GRAD_TOL, rtol=GRAD_TOL)
        assert_optimizer(*optimizers)


@pytest.mark.parametrize('arm', ARMS)
def test_shared_operations_are_counted_once_without_misattributing_routes(arm):
    old, new = model(arm), model(arm)
    values = case(8)
    expected, actual = separate(old, values), reuse.joint_objective(new, **values)
    blocks = actual['work']
    assert tuple(blocks) == reuse.WORK_ROUTES
    assert all(set(block) == set(frozen.WORK_KEYS) and
               all(type(value) is int and value >= 0 for value in block.values()) for block in blocks.values())
    shared = blocks['shared']
    assert shared['probability_field_calls'] == shared['factorization_calls'] == 1
    assert shared['operator_marginal_sum_calls'] == 1
    assert shared['rounded_transition_calls'] == int(arm == 'rounded')
    assert shared['transition_softmax_calls'] == int(arm != 'rounded')
    assert shared['emission_softmax_calls'] == shared['hazard_sigmoid_calls'] == 1
    assert sum(expected[route]['work']['probability_field_calls'] for route in ('blind', 'observed', 'prefix')) == 5
    assert sum(block['probability_field_calls'] for block in blocks.values()) == 1
    e, b, horizon = len(values['endpoint_positions']), len(values['prefix']), values['actions'].shape[1]
    assert blocks['endpoint_prefix']['reset_emission_rows'] == e
    assert blocks['endpoint_prefix']['prefix_filter_calls'] == 8
    assert blocks['endpoint_prefix']['prefix_filter_rows'] == 8 * e
    assert blocks['prefix_nll']['reset_emission_rows'] == b
    assert blocks['prefix_nll']['prefix_nll_rows'] == int(values['lengths'].sum())
    for route in ('blind', 'observed'):
        assert blocks[route]['cost_head_softmax_calls'] == horizon
        assert blocks[route]['cost_readout_rows'] == e * horizon
        assert blocks[route]['probability_field_calls'] == 0
        assert blocks[route]['prefix_filter_calls'] == 0
    assert new.last_work == {key: sum(block[key] for block in blocks.values()) for key in frozen.WORK_KEYS}
    saved = actual['prefix']['work']['prefix_nll_rows']
    blocks['prefix_nll']['prefix_nll_rows'] = -1
    assert actual['prefix']['work']['prefix_nll_rows'] == new.last_work['prefix_nll_rows'] == saved


def test_forecasts_precede_own_labels_and_blind_path_never_reads_future_observations():
    value = model('rounded')
    first = case(8, (None, None))
    first['observations'].zero_()
    second = copy.deepcopy(first)
    second['observations'][1, 2] = 3
    a, b = reuse.joint_objective(value, **first), reuse.joint_objective(value, **second)
    for key in ('prefix_state', 'prior_states', 'cost_contrasts', 'survival_mass', 'found_increments'):
        torch.testing.assert_close(a['blind'][key], b['blind'][key], atol=0, rtol=0)
    for key in ('prior_states', 'cost_contrasts', 'survival_mass', 'probabilities'):
        torch.testing.assert_close(a['observed'][key][:, :3], b['observed'][key][:, :3], atol=0, rtol=0)
    assert not torch.equal(a['observed']['posterior_states'][1, 2], b['observed']['posterior_states'][1, 2])
    found = case(8, (None,))
    found['observations'].fill_(4)
    found['targets']['observed_probabilities'][:, 1:] = torch.tensor([0., 0., 0., 0., 1.], dtype=torch.float64)
    found['targets']['observed_costs'][:, 1:] = 0
    found['targets']['observed_survival'][:, 1:] = 0
    out = reuse.joint_objective(value, **found)['observed']
    assert torch.count_nonzero(out['posterior_states']) == 0
    assert torch.count_nonzero(out['cost_contrasts'][:, 1:]) == 0
    assert torch.count_nonzero(out['survival_mass'][:, 1:]) == 0
    assert torch.equal(out['probabilities'][:, 1:, 4], torch.ones((1, 7), dtype=torch.float64))


def test_owned_outputs_inputs_parameters_flags_and_rng_without_persistent_cache():
    value = model('rounded')
    value.eval()
    values = case(2)
    before = copy.deepcopy(values)
    states = {key: tensor.clone() for key, tensor in value.state_dict().items()}
    flags = {name: parameter.requires_grad for name, parameter in value.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    actual = reuse.joint_objective(value, **values)
    assert torch.equal(torch.random.get_rng_state(), rng) and not value.training
    assert flags == {name: parameter.requires_grad for name, parameter in value.named_parameters()}
    for key, tensor in value.state_dict().items():
        assert torch.equal(tensor, states[key])
    for key in ('prefix', 'lengths', 'endpoint_positions', 'actions', 'observations'):
        assert torch.equal(values[key], before[key])
    for key in reuse.TARGET_FIELDS:
        assert torch.equal(values['targets'][key], before['targets'][key])
    assert actual['blind']['prefix_state'].data_ptr() != actual['observed']['prefix_state'].data_ptr()
    actual['loss'].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in value.parameters())
    with torch.no_grad():
        actual['blind']['prefix_state'].zero_()
        value.hazard_logits.add_(.3)
    next_result = reuse.joint_objective(value, **values)
    assert_values(next_result, separate(value, values))
    assert not torch.equal(actual['blind']['survival_mass'], next_result['blind']['survival_mass'])
    assert torch.count_nonzero(next_result['blind']['prefix_state']) > 0


@pytest.mark.parametrize('arm', ARMS)
def test_callbacks_and_owned_partial_failure_work(arm):
    calls, fail_at = [], [None]
    marker = TimeoutError('fabricated callback')

    def check():
        calls.append(len(calls) + 1)
        if len(calls) == fail_at[0]:
            raise marker

    value = model(arm, check=check)
    calls.clear()
    states = {key: tensor.clone() for key, tensor in value.state_dict().items()}
    fail_at[0] = 3 if arm == 'rounded' else 1
    with pytest.raises(TimeoutError) as caught:
        reuse.joint_objective(value, **case())
    assert caught.value is marker
    partial = copy.deepcopy(marker.joint_reuse_work)
    assert partial['shared']['probability_field_calls'] == 1
    assert not any(partial['blind'].values()) and not any(partial['observed'].values())
    assert marker.rounded_model_work == {key: sum(block[key] for block in partial.values()) for key in frozen.WORK_KEYS}
    calls.clear()
    fail_at[0] = None
    reuse.joint_objective(value, **case())
    assert len(calls) == (12 if arm == 'rounded' else 1)
    assert marker.joint_reuse_work == partial
    for key, tensor in value.state_dict().items():
        assert torch.equal(tensor, states[key])


@pytest.mark.parametrize('invalid', ['missing_endpoint', 'reordered', 'found_endpoint', 'bad_padding',
    'bad_found_suffix', 'nan_target', 'wrong_dtype', 'zero_denominator', 'bool_denominator',
    'small_population', 'unknown_target'])
def test_invalid_inputs_fail_before_probability_construction(invalid):
    calls = []
    value = model('rounded', check=lambda: calls.append(1))
    calls.clear()
    values = case(2)
    if invalid == 'missing_endpoint':
        values['endpoint_positions'] = values['endpoint_positions'][:1]
    elif invalid == 'reordered':
        values['endpoint_positions'] = values['endpoint_positions'].flip(0)
    elif invalid == 'found_endpoint':
        values['endpoint_positions'] = torch.tensor([0, 1], dtype=torch.int64)
    elif invalid == 'bad_padding':
        values['prefix'][1, 8, 4] = 1
    elif invalid == 'bad_found_suffix':
        values['observations'][0] = torch.tensor([4, 0])
    elif invalid == 'nan_target':
        values['targets']['blind_costs'][0, 0, 0] = float('nan')
    elif invalid == 'wrong_dtype':
        values['lengths'] = values['lengths'].to(torch.int32)
    elif invalid == 'zero_denominator':
        values['total_events'] = 0
    elif invalid == 'bool_denominator':
        values['total_attempts'] = True
    elif invalid == 'small_population':
        values['total_survivors'] = 1
    else:
        values['targets']['oracle_prefix'] = torch.zeros(8, dtype=torch.float64)
    with pytest.raises(ValueError) as caught:
        reuse.joint_objective(value, **values)
    assert not calls
    assert not any(caught.value.rounded_model_work.values())


def test_finite_probability_rejection_is_not_repaired_or_cached():
    value = model('rounded')
    with torch.no_grad():
        value.emission_logits[0].fill_(-10000)
    with pytest.raises(ValueError, match='strict factor probability bounds') as caught:
        reuse.joint_objective(value, **case())
    assert caught.value.joint_reuse_work['shared']['emission_softmax_calls'] == 1
    assert caught.value.joint_reuse_work['shared']['rounded_transition_calls'] == 1
    assert not any(caught.value.joint_reuse_work['prefix_nll'].values())
