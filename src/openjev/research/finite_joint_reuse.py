"""Per-call reuse for the frozen rounded-model joint objective.

This changes graph sharing, not the mathematical loss or model parameters.
One probability-field/factor construction serves every route. One ordinary
endpoint-prefix filter serves both continuations. The all-attempt NLL filter
stays separate: its reset sum/division order intentionally remains different
from the endpoint filter. Readouts still execute separately at every horizon.

All shared tensors retain autograd and exist only in this call. No persistent
cache, detach, parameter change or method replacement is used. Shared reverse
mode accumulation can differ in floating point from repeated graphs; numerical
parity, not bitwise gradients or long-training identity, is the contract.

Work has five disjoint blocks. ``shared`` owns probability construction and
the one marginal sum, ``endpoint_prefix`` owns its shared filtering, and the
other blocks own their actual NLL/continuation operations. Nested prediction
work is a copy of its corresponding block, not additional executed work.
Counters exclude backward, validation and unlisted scalar bookkeeping, as in
the frozen component. A failure exports owned partial blocks as
``joint_reuse_work`` and their sum as ``rounded_model_work``.
"""
from __future__ import annotations

import torch

from openjev.research.finite_cost_readout_models import _prefix_inputs
from openjev.research.finite_observation_models import objective
from openjev.research.finite_rounded_models import (
    WORK_KEYS,
    RoundedDynamicsModel,
    _validate_model,
    work_counts,
)
from openjev.research.otto_observation_operator_model import _finite, _tensor, require

VERSION = 'finite-joint-reuse-v1'
WORK_ROUTES = ('shared', 'endpoint_prefix', 'prefix_nll', 'blind', 'observed')
TARGET_FIELDS = ('blind_costs', 'observed_costs', 'blind_survival',
                 'observed_survival', 'observed_probabilities')


def _total(work):
    return {key: sum(block[key] for block in work.values()) for key in WORK_KEYS}


def _inputs(model, prefix, lengths, positions, actions, observations, targets,
            total_attempts, total_survivors, total_events):
    _validate_model(model)
    active = _prefix_inputs(prefix, lengths)
    batch = len(prefix)
    require(isinstance(positions, torch.Tensor) and positions.ndim == 1,
            'endpoint positions are a vector into this attempted minibatch')
    eligible = len(positions)
    _tensor(positions, torch.int64, (eligible,), 'endpoint positions')
    expected = torch.nonzero(~prefix[:, :, 8].bool().any(-1), as_tuple=False).flatten()
    require(torch.equal(positions, expected),
            'every surviving endpoint exactly once in original attempted order')
    require(isinstance(actions, torch.Tensor) and actions.ndim == 2,
            'endpoint actions [eligible,horizon]')
    horizon = actions.shape[1]
    require(1 <= horizon <= 8, 'endpoint horizon in1..8')
    _tensor(actions, torch.int64, (eligible, horizon), 'endpoint actions')
    _tensor(observations, torch.int64, (eligible, horizon), 'endpoint observations')
    require(bool(((actions >= 0) & (actions < 4)).all())
            and bool(((observations >= 0) & (observations <= 4)).all()), 'public endpoint labels')
    found = observations == 4
    require(torch.equal(found, found.to(torch.int64).cummax(-1).values.bool()), 'absorbing found suffix')
    require(type(targets) is dict and set(targets) == set(TARGET_FIELDS), 'exact endpoint target fields')
    for name in TARGET_FIELDS:
        shape = (eligible, horizon, 5) if name == 'observed_probabilities' else (
            (eligible, horizon, 4) if name.endswith('costs') else (eligible, horizon))
        _tensor(targets[name], torch.float64, shape, name)
        _finite(targets[name], name)
    require(bool((targets['observed_probabilities'] >= 0).all())
            and bool(((targets['observed_probabilities'].sum(-1) - 1).abs() <= 1e-6).all()),
            'nonnegative unit target probability rows')
    require(all(type(value) is int and value > 0
                for value in (total_attempts, total_survivors, total_events))
            and total_attempts >= batch and total_attempts >= total_survivors >= eligible
            and total_events >= total_attempts and total_events >= int(lengths.sum()),
            'fixed positive full-TRAIN denominators and current-batch support')
    if eligible:
        model._rollout_inputs(prefix[positions], lengths[positions], actions, observations)
    return active, horizon


def _endpoint_initial(model, prefix, lengths, emission, operators, work):
    """Frozen FactorizedDynamicsModel._initial arithmetic, supplied fields."""
    batch = model._prefix_inputs(prefix, lengths)
    labels = prefix[:, 0, 4:8].argmax(-1)
    unnormalized = emission[labels] / 8
    evidence = unnormalized.sum(-1)
    require(bool((unnormalized > 0).all()) and bool((evidence > 0).all()),
            'positive reset mass and evidence')
    state = unnormalized / evidence[:, None]
    model._state(state, normalized=True)
    work['reset_emission_rows'] += batch
    for step in range(1, prefix.shape[1]):
        lanes = torch.nonzero(lengths > step, as_tuple=False).flatten()
        if lanes.numel():
            actions = prefix[lanes, step, :4].argmax(-1)
            odors = prefix[lanes, step, 4:8].argmax(-1)
            branch = model._multiply(operators[actions, odors], state[lanes])
            evidence = branch.sum(-1)
            require(bool((evidence > 0).all()), 'positive prefix evidence')
            conditioned = branch / evidence[:, None]
            model._state(conditioned, normalized=True)
            state = state.index_copy(0, lanes, conditioned)
            work['prefix_filter_calls'] += 1
            work['prefix_filter_rows'] += len(lanes)
    return state


def _prefix_nll(model, prefix, active, emission, operators, event_law, work):
    """Frozen all-attempt reset/event NLL arithmetic, supplied fields."""
    batch = len(prefix)
    reset_law = emission.sum(-1) / 8
    require(bool((reset_law.sum() - 1).abs() <= model._tolerance()), 'normalized reset event law')
    probabilities = [torch.cat((reset_law, reset_law.new_zeros(1))).expand(batch, 5).clone()]
    labels = prefix[:, 0, 4:9].argmax(-1)
    evidence = reset_law[labels]
    require(bool((evidence > 0).all()), 'positive reset observed probability')
    nll = [-evidence.log()]
    initial_mass = emission[labels] / 8
    require(bool((initial_mass > 0).all()), 'reset mass product underflow rejected')
    state = initial_mass / evidence[:, None]
    model._state(state, normalized=True)
    work['reset_emission_rows'] += batch
    work['prefix_probability_rows'] += batch
    work['prefix_nll_rows'] += batch
    for step in range(1, 9):
        lanes = torch.nonzero(active[:, step], as_tuple=False).flatten()
        row_probabilities = operators.new_zeros((batch, 5))
        row_nll = operators.new_zeros(batch)
        if lanes.numel():
            actions = prefix[lanes, step, :4].argmax(-1)
            labels = prefix[lanes, step, 4:9].argmax(-1)
            predicted = model._multiply(event_law[actions], state[lanes])
            require(bool(((predicted.sum(-1) - 1).abs() <= model._tolerance()).all()),
                    'unit predictive event mass')
            observed = predicted.gather(1, labels[:, None]).squeeze(1)
            require(bool((observed > 0).all()), 'positive observed probability; no clipping')
            row_probabilities = row_probabilities.index_copy(0, lanes, predicted)
            row_nll = row_nll.index_copy(0, lanes, -observed.log())
            ordinary = torch.nonzero(labels < 4, as_tuple=False).flatten()
            updated = state.new_zeros((len(lanes), 8))
            if ordinary.numel():
                branch = model._multiply(operators[actions[ordinary], labels[ordinary]], state[lanes[ordinary]])
                branch_mass = branch.sum(-1)
                require(bool((branch_mass > 0).all()), 'positive conditioning evidence')
                posterior = branch / branch_mass[:, None]
                model._state(posterior, normalized=True)
                updated = updated.index_copy(0, ordinary, posterior)
                work['prefix_filter_calls'] += 1
                work['prefix_filter_rows'] += len(ordinary)
            state = state.index_copy(0, lanes, updated)
            work['prefix_probability_rows'] += len(lanes)
            work['prefix_nll_rows'] += len(lanes)
        probabilities.append(row_probabilities)
        nll.append(row_nll)
    probabilities, nll = torch.stack(probabilities, 1), torch.stack(nll, 1)
    _finite(probabilities, 'prefix predictions')
    _finite(nll, 'prefix negative log-likelihood')
    return {'probabilities': probabilities, 'nll': nll, 'work': dict(work)}


def _continuation(model, initial, actions, observations, operators, marginal, event_law, work):
    """Frozen FactorModel continuation, with shared fields and prefix state."""
    state, records, events = initial, [], []
    for step in range(actions.shape[1]):
        if observations is None:
            result = model._prior(state, actions[:, step], marginal, operators['found'], work)
            state = result['prior_state']
        else:
            probabilities = model._multiply(event_law[actions[:, step]], state)
            terminal = torch.zeros_like(probabilities)
            terminal[:, 4] = 1
            probabilities = torch.where((state.sum(-1) == 0)[:, None], terminal, probabilities)
            require(bool(((probabilities.sum(-1) - 1).abs() <= model._tolerance()).all()),
                    'full event probability mass')
            events.append(probabilities)
            work['event_probability_rows'] += len(state)
            result = model._observed(state, actions[:, step], observations[:, step], operators, marginal, work)
            state = result['posterior_state']
        records.append(result)
    result = {'prefix_state': initial.clone(),
        'prior_states': torch.stack([r['prior_state'] for r in records], 1),
        'posterior_states': None if observations is None else torch.stack([r['posterior_state'] for r in records], 1),
        'cost_contrasts': torch.stack([r['cost_contrasts'] for r in records], 1),
        'survival_mass': torch.stack([r['survival_mass'] for r in records], 1),
        'found_increments': torch.stack([r['found_increment'] for r in records], 1),
        'evidence': None if observations is None else torch.stack([r['evidence'] for r in records], 1),
        'work': dict(work)}
    if observations is not None:
        result['probabilities'] = torch.stack(events, 1)
    return result


def joint_objective(model, prefix, lengths, endpoint_positions, actions, observations, targets,
                    *, total_attempts, total_survivors, total_events):
    """Return loss, blind/observed predictions, prefix NLL and disjoint work.

    ``prefix`` is float32[B,9,31]; lengths and endpoint_positions are int64.
    Positions must list ALL non-found prefixes in their attempted batch order.
    Endpoint actions/observations are int64[E,H], H in1..8; targets are exactly
    TARGET_FIELDS as float64 endpoint arrays. E=0 is allowed with shaped empty
    arrays and returns blind=observed=None. Dataset denominators are explicit
    positive Python integers, never recomputed from the minibatch.

    The preserved arithmetic is n/b*(e/S*endpoint_objective+sum(prefix_NLL)/E),
    plus the frozen graph-zero term for every parameter, including the head.
    Here E in the formula denotes total_events, not endpoint row count.
    No prior is added to this JOINT objective; prefix-only learning is unchanged.
    """
    work = {route: work_counts() for route in WORK_ROUTES}
    try:
        active, _horizon = _inputs(model, prefix, lengths, endpoint_positions, actions, observations,
                                  targets, total_attempts, total_survivors, total_events)
        fields = model._fields(work['shared'])
        emission, branches, found = model._factorize_fields(fields, work['shared'])
        event_law = torch.cat((branches.sum(2), found[:, None]), 1)
        require(bool(((event_law.sum(1) - 1).abs() <= model._tolerance()).all()), 'stochastic event columns')
        eligible = len(endpoint_positions)
        loss = sum(parameter.sum() * 0 for parameter in model.parameters())
        blind = observed = None
        if eligible:
            initial = _endpoint_initial(model, prefix[endpoint_positions], lengths[endpoint_positions],
                                        emission, branches, work['endpoint_prefix'])
            marginal = branches.sum(1)
            work['shared']['operator_marginal_sum_calls'] += 1
            operators = {'observed': branches, 'found': found, 'blind': marginal, 'blind_found': found}
            blind = _continuation(model, initial, actions, None, operators, marginal, event_law, work['blind'])
            observed = _continuation(model, initial, actions, observations, operators, marginal,
                                     event_law, work['observed'])
            loss = loss + objective(blind, observed, targets) * eligible / total_survivors
        prefix_result = _prefix_nll(model, prefix, active, emission, branches, event_law, work['prefix_nll'])
        loss = (loss + prefix_result['nll'].sum() / total_events) * total_attempts / len(prefix)
        _finite(loss, 'finite shared joint objective')
        model.last_work = _total(work)
        return {'loss': loss, 'blind': blind, 'observed': observed, 'prefix': prefix_result,
                'work': {route: dict(block) for route, block in work.items()}}
    except BaseException as error:
        error.joint_reuse_work = {route: dict(block) for route, block in work.items()}
        error.rounded_model_work = _total(work)
        if type(model) is RoundedDynamicsModel:
            model.last_work = dict(error.rounded_model_work)
        raise
