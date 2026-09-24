"""All-attempt public prefixes and pre-assimilation likelihood for a fixed filter.

The collector preserves the frozen finite world's per-case PCG64 draw order.
It differs only by retaining first-found prefix rows instead of discarding the
whole prefix. Forecast targets exist only for prefixes surviving all eight
actions. Oracle boundary states are returned separately and never consumed by
the learned prefix-prediction adapter. No global RNG, files or fitted model
state is mutated. The two arm factories return the identical shared model.

Prefix NLL is returned per valid observed event, not reduced per sequence.
Callers own the declared dataset denominator and coefficient. Reset predicts
four odors before conditioning a uniform prior. Later actions predict all five
events before conditioning; the first found event is scored and terminates.
Post-found padding contributes no likelihood and has zero output arrays.
"""
from __future__ import annotations

import numpy as np
import torch

from openjev.research import finite_observation_world as fw
from openjev.research.finite_shared_filter_models import PREFIX_WORK_KEYS, SharedFilterModel
from openjev.research.otto_observation_operator_model import _finite, _tensor, _work, require

VERSION = 'finite-prefix-learning-data-v1'
SEED_NAMESPACE = 423260924
ARMS = ('endpoint_only', 'endpoint_plus_prefix')


def _attempt_prefix(operators, rng):
    belief = np.full(8, 1 / 8, dtype=np.float64)
    observation = fw._sample(rng, operators['emission'] @ belief)
    belief = operators['emission'][observation] * belief
    belief /= belief.sum(dtype=np.float64)
    fw._mass(belief, normalized=True, name='initial conditioned belief')
    rows = np.zeros((9, 31), dtype=np.float32)
    rows[0, 4 + observation] = rows[0, 9] = 1
    actions = rng.integers(0, 4, size=8, dtype=np.int64)
    for step, action in enumerate(actions):
        branches, probabilities = fw._event(operators, belief, int(action))
        observation = fw._sample(rng, probabilities)
        rows[step + 1, action] = rows[step + 1, 4 + observation] = 1
        if observation == 4:
            return rows, step + 2, None
        belief = fw._condition(branches, probabilities, observation)
    return rows, 9, belief


def generate_attempt_split(split, attempts, horizon, seed_namespace=SEED_NAMESPACE):
    """Return owned survivor targets, separate oracle and every attempted prefix.

    Public prefix arrays: prefix f32[N,9,31], lengths i64[N], case_ids U64[N],
    event_mask bool[N,9], endpoint_eligible bool[N], endpoint_rows i64[N]. The
    endpoint row is -1 for terminated cases and sequential for survivors.
    Empty splits retain exact ranks/dtypes. Survivor data has the original ten
    fields and original case IDs. Only oracle['prefix_state'] contains beliefs.
    """
    require(type(split) is int and split in fw.EPSILON, 'split in (0,1,2)')
    require(type(attempts) is int and 0 <= attempts < 2**32, 'uint32 attempts')
    require(type(horizon) is int and 1 <= horizon <= 8, 'horizon in1..8')
    require(type(seed_namespace) is int and 0 <= seed_namespace < 2**32, 'uint32 seed namespace')
    operators = fw.world(fw.EPSILON[split])
    specs = {'prefix': ((9, 31), np.float32), 'lengths': ((), np.int64),
             'actions': ((horizon,), np.int64), 'observations': ((horizon,), np.int64),
             'blind_costs': ((horizon, 4), np.float64), 'blind_survival': ((horizon,), np.float64),
             'observed_costs': ((horizon, 4), np.float64), 'observed_survival': ((horizon,), np.float64),
             'observed_probabilities': ((horizon, 5), np.float64), 'case_ids': ((), np.dtype('<U64'))}
    data_rows = {name: [] for name in specs}
    prefix = np.zeros((attempts, 9, 31), dtype=np.float32)
    lengths = np.zeros(attempts, dtype=np.int64)
    case_ids = np.empty(attempts, dtype='<U64')
    eligible = np.zeros(attempts, dtype=bool)
    endpoint_rows = np.full(attempts, -1, dtype=np.int64)
    boundary_states = []
    counts = {'version': VERSION, 'seed_namespace': seed_namespace, 'split_id': split,
              'epsilon': fw.EPSILON[split], 'horizon': horizon, 'attempted': attempts,
              'retained': 0, 'discarded_found': 0, 'valid_prefix_events': 0,
              'prefix_found_by_step': [0] * 8, 'initial_odor_draws': attempts,
              'prefix_event_draws': 0, 'prefix_action_draws': attempts * 8,
              'forecast_action_draws': 0, 'forecast_event_draws': 0, 'forecast_found_cases': 0}
    for index in range(attempts):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed_namespace, split, index])))
        rows, length, boundary = _attempt_prefix(operators, rng)
        case_id = f'ns{seed_namespace}-split{split}-case{index:010d}'
        prefix[index], lengths[index], case_ids[index] = rows, length, case_id
        counts['valid_prefix_events'] += length
        counts['prefix_event_draws'] += length - 1
        if boundary is None:
            counts['discarded_found'] += 1
            counts['prefix_found_by_step'][length - 2] += 1
            continue
        eligible[index] = True
        endpoint_rows[index] = counts['retained']
        actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
        targets, _, sampled_events = fw._forecast(operators, boundary, actions, rng)
        row = {'prefix': rows, 'lengths': 9, 'actions': actions, 'case_ids': case_id, **targets}
        for name, values in data_rows.items():
            values.append(row[name])
        boundary_states.append(boundary)
        counts['retained'] += 1
        counts['forecast_action_draws'] += horizon
        counts['forecast_event_draws'] += sampled_events
        counts['forecast_found_cases'] += int(np.any(targets['observations'] == 4))
    retained = counts['retained']
    data = {name: np.array(data_rows[name], dtype=dtype).reshape((retained, *shape))
            for name, (shape, dtype) in specs.items()}
    oracle = {'prefix_state': np.array(boundary_states, dtype=np.float64).reshape(retained, 8)}
    prefix_data = {'prefix': prefix, 'lengths': lengths, 'case_ids': case_ids,
                   'event_mask': np.arange(9)[None] < lengths[:, None],
                   'endpoint_eligible': eligible, 'endpoint_rows': endpoint_rows}
    require(retained + counts['discarded_found'] == attempts
            and sum(counts['prefix_found_by_step']) == counts['discarded_found']
            and counts['valid_prefix_events'] == attempts + counts['prefix_event_draws']
            and int(prefix_data['event_mask'].sum()) == counts['valid_prefix_events'],
            'complete all-attempt accounting')
    require(all(np.isfinite(value).all() for name, value in data.items() if name != 'case_ids')
            and np.isfinite(prefix).all() and np.isfinite(oracle['prefix_state']).all(), 'finite generated arrays')
    return {'data': data, 'oracle': oracle, 'prefix_data': prefix_data, 'counts': counts}


def make_model(arm, seed):
    require(type(arm) is str and arm in ARMS, 'declared prefix-learning arm')
    return SharedFilterModel('shared_filter', seed)


def _prefix_inputs(prefix, lengths):
    require(isinstance(prefix, torch.Tensor) and prefix.ndim == 3, 'prefix [N,9,31]')
    batch = len(prefix)
    require(batch > 0, 'nonempty prediction batch')
    _tensor(prefix, torch.float32, (batch, 9, 31), 'all-attempt prefix')
    _tensor(lengths, torch.int64, (batch,), 'all-attempt lengths')
    require(bool(((lengths >= 2) & (lengths <= 9)).all()), 'initial odor plus at least one action event')
    _finite(prefix, 'all prefix rows including padding')
    active = torch.arange(9, device='cpu')[None] < lengths[:, None]
    require(bool((prefix[~active] == 0).all()), 'zero padding')
    rows = prefix[active]
    require(bool(((rows == 0) | (rows == 1)).all()) and bool((rows[:, 10:] == 0).all()),
            'binary public tokens only')
    require(bool((rows[:, 4:9].sum(-1) == 1).all()), 'one event per active row')
    initial = prefix[:, 0]
    require(bool((initial[:, :4] == 0).all()) and bool((initial[:, 8] == 0).all())
            and bool((initial[:, 9] == 1).all()), 'ordinary reset odor without action or hazard')
    later = prefix[:, 1:][active[:, 1:]]
    require(bool((later[:, :4].sum(-1) == 1).all()) and bool((later[:, 9] == 0).all()),
            'one action per later row without reset')
    found = prefix[:, :, 8] == 1
    last = torch.arange(9, device='cpu')[None] == lengths[:, None] - 1
    require(not bool((found & ~last).any()) and bool((found.any(-1) | (lengths == 9)).all()),
            'only first-found termination can shorten prefix')
    return active


def prefix_predictions(model, prefix, lengths):
    """Return pre-label probabilities[N,9,5], event NLL[N,9] and work.

    Reset found probability and every padded output are exact zero. Valid NLL
    uses the actual recorded event; it never clips probabilities or averages
    by a case's realized length. Full five-event predictions precede each
    action's observation. Returned tensors retain gradients and own storage.
    """
    require(type(model) is SharedFilterModel and model.arm == 'shared_filter', 'same shared-filter model only')
    model._parameters_valid()
    active = _prefix_inputs(prefix, lengths)
    batch = len(prefix)
    work = {**_work(), **dict.fromkeys(PREFIX_WORK_KEYS, 0),
            'prefix_probability_rows': 0, 'prefix_nll_rows': 0}
    emission = model.reset_logits.softmax(0)
    work['reset_emission_softmax_calls'] += 1
    _finite(emission, 'reset emission')
    require(bool((emission > 0).all()), 'reset emission underflow rejected')
    reset_law = emission.sum(-1) / 8
    require(bool((reset_law.sum() - 1).abs() <= model._tolerance()), 'normalized reset event law')
    probabilities = [torch.cat((reset_law, reset_law.new_zeros(1))).expand(batch, 5).clone()]
    initial_labels = prefix[:, 0, 4:9].argmax(-1)
    evidence = reset_law[initial_labels]
    require(bool((evidence > 0).all()), 'positive reset observed probability')
    nll = [-evidence.log()]
    initial_mass = emission[initial_labels] / 8
    require(bool((initial_mass > 0).all()), 'reset mass product underflow rejected')
    state = initial_mass / evidence[:, None]
    model._state(state, normalized=True)
    work['reset_emission_rows'] += batch
    work['prefix_probability_rows'] += batch
    work['prefix_nll_rows'] += batch
    columns = model.observed_logits.softmax(1)
    work['prefix_operator_softmax_calls'] += 1
    _finite(columns, 'prefix operators')
    require(bool((columns > 0).all()), 'prefix operator underflow rejected')
    operators, found = columns[:, :-1].reshape(4, 4, 8, 8), columns[:, -1]
    event_law = torch.cat((operators.sum(2), found[:, None]), 1)
    require(bool(((event_law.sum(1) - 1).abs() <= model._tolerance()).all()), 'stochastic event columns')
    for step in range(1, 9):
        lanes = torch.nonzero(active[:, step], as_tuple=False).flatten()
        row_probabilities = columns.new_zeros((batch, 5))
        row_nll = columns.new_zeros(batch)
        if lanes.numel():
            actions = prefix[lanes, step, :4].argmax(-1)
            labels = prefix[lanes, step, 4:9].argmax(-1)
            predicted = model._multiply(event_law[actions], state[lanes])
            require(bool(((predicted.sum(-1) - 1).abs() <= model._tolerance()).all()), 'unit predictive event mass')
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
    return {'probabilities': probabilities, 'nll': nll, 'work': work}
