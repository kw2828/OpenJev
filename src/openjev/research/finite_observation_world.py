"""A finite synthetic POMDP with exact float64 predictive decision targets.

This is an eight-state reference world, not a trained model or a simulator
wrapper. B[a,o,next,current] includes transition, survival and odor emission;
found[a,current] completes each stochastic column. Blind A is exactly sum_o B.
Costs are centered across four decisions, with no bias or survival division.

The public prefix contains only action/observation one-hots and a reset bit.
Beliefs are target-generation state and appear only in the optional, separate
audit dictionary. All targets precede the current observation. Blind mass is
unconditional surviving mass; observed survival is a one-step conditional
probability following a normalized history. Do not sum observed found
probabilities across horizons. Found is absorbing with zero future cost.

Every attempted case has its own PCG64 SeedSequence. A found prefix is excluded
without replacement. Prefix actions are drawn before prefix events; the fixed
forecast action block is drawn before any forecast observation. Sampling
normalizes only a float64 probability total already checked to equal one within
roundoff. Targets are not sampled estimates, clipped, or silently repaired.
No file I/O, global RNG mutation, fitted parameters or external runtime is used.
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'finite-observation-world-v1'
SEED_NAMESPACE = 420260924
STATE_DIM, ACTION_DIM, ODOR_DIM, EVENT_DIM = 8, 4, 4, 5
FEATURE_DIM, PREFIX_STEPS, MAX_HORIZON = 31, 8, 8
EPSILON = {0: .12, 1: .12, 2: .30}
MASS_ATOL = 1e-12


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _mass(value, *, normalized, name):
    _require(value.dtype == np.float64 and np.isfinite(value).all()
             and (value >= 0).all(), 'finite nonnegative float64 ' + name)
    total = float(value.sum(dtype=np.float64))
    _require(math.isfinite(total) and total <= 1 + MASS_ATOL, 'bounded mass: ' + name)
    if normalized:
        _require(abs(total - 1) <= MASS_ATOL, 'unit mass: ' + name)
    return total


def world(epsilon):
    """Return owned float64 B, found, A, centered costs and emission arrays.

    Shapes are [4,4,8,8], [4,8], [4,8,8], [4,8] and [4,8]. Epsilon must
    be strictly between zero and one, keeping every odor likelihood positive.
    No caller array or shared cached world can be mutated by a returned array.
    """
    _require(type(epsilon) in (int, float) and math.isfinite(epsilon)
             and 0 < epsilon < 1, 'finite epsilon strictly between zero and one')
    states = np.arange(STATE_DIM, dtype=np.int64)
    destinations = np.stack((states ^ 1, (states + 1) % 8,
                             ((states << 1) & 7) | (states >> 2), states ^ 4))
    transition = np.full((ACTION_DIM, STATE_DIM, STATE_DIM), .02 / 8, dtype=np.float64)
    for action in range(ACTION_DIM):
        transition[action, destinations[action], states] += .98
    hazard = .005 + .005 * (((states[None, :] >> 2) & 1)
                             ^ (np.arange(ACTION_DIM, dtype=np.int64)[:, None] & 1))
    emission = np.full((ODOR_DIM, STATE_DIM), epsilon / 3, dtype=np.float64)
    emission[states & 3, states] = 1 - epsilon
    branches = (transition[:, None, :, :] * (1 - hazard[:, None, :, None])
                * emission[None, :, :, None])
    found = np.sum(transition * hazard[:, :, None], axis=1, dtype=np.float64)
    blind = branches.sum(axis=1, dtype=np.float64)
    costs = (np.arange(ACTION_DIM)[:, None] != ((states ^ (states >> 1)) & 3)).astype(np.float64)
    costs -= costs.mean(axis=0, keepdims=True)
    _require(np.isfinite(branches).all() and (branches > 0).all()
             and np.isfinite(found).all() and (found > 0).all()
             and np.all(np.abs(blind.sum(axis=1) + found - 1) <= MASS_ATOL)
             and np.all(np.abs(emission.sum(axis=0) - 1) <= MASS_ATOL)
             and np.isfinite(costs).all(), 'finite stochastic world')
    return {'B': branches, 'found': found, 'A': blind,
            'costs': costs, 'emission': emission}


def _sample(rng, probabilities):
    total = _mass(probabilities, normalized=True, name='event probabilities')
    return int(rng.choice(len(probabilities), p=probabilities / total))


def _event(operators, belief, action):
    """Return nonterminal branches and all five event masses, before evidence."""
    _mass(belief, normalized=True, name='observed belief')
    branches = operators['B'][action] @ belief
    probabilities = np.concatenate((branches.sum(axis=1, dtype=np.float64),
                                     np.array([operators['found'][action] @ belief], dtype=np.float64)))
    _mass(probabilities, normalized=True, name='predictive event distribution')
    return branches, probabilities


def _condition(branches, probabilities, observation):
    _require(0 <= observation < ODOR_DIM and probabilities[observation] > 0,
             'possible nonterminal observation')
    posterior = branches[observation] / probabilities[observation]
    _mass(posterior, normalized=True, name='conditioned belief')
    return posterior


def _prefix(operators, rng):
    belief = np.full(STATE_DIM, 1 / STATE_DIM, dtype=np.float64)
    observation = _sample(rng, operators['emission'] @ belief)
    belief = operators['emission'][observation] * belief
    belief /= belief.sum(dtype=np.float64)
    _mass(belief, normalized=True, name='initial conditioned belief')
    rows = np.zeros((PREFIX_STEPS + 1, FEATURE_DIM), dtype=np.float32)
    rows[0, 4 + observation] = 1
    rows[0, 9] = 1
    beliefs = np.zeros((PREFIX_STEPS + 1, STATE_DIM), dtype=np.float64)
    beliefs[0] = belief
    actions = rng.integers(0, ACTION_DIM, size=PREFIX_STEPS, dtype=np.int64)
    for step, action in enumerate(actions):
        branches, probabilities = _event(operators, belief, int(action))
        observation = _sample(rng, probabilities)
        if observation == ODOR_DIM:
            return None, None, step + 1
        belief = _condition(branches, probabilities, observation)
        rows[step + 1, action] = 1
        rows[step + 1, 4 + observation] = 1
        beliefs[step + 1] = belief
    return rows, beliefs, PREFIX_STEPS


def _forecast(operators, belief, actions, rng):
    horizon = len(actions)
    result = {
        'observations': np.full(horizon, ODOR_DIM, dtype=np.int64),
        'blind_costs': np.zeros((horizon, ACTION_DIM), dtype=np.float64),
        'blind_survival': np.zeros(horizon, dtype=np.float64),
        'observed_costs': np.zeros((horizon, ACTION_DIM), dtype=np.float64),
        'observed_survival': np.zeros(horizon, dtype=np.float64),
        'observed_probabilities': np.zeros((horizon, EVENT_DIM), dtype=np.float64),
    }
    audit = {name: np.zeros((horizon, STATE_DIM), dtype=np.float64)
             for name in ('blind_prior', 'observed_prior', 'observed_posterior')}
    blind = belief.copy()
    observed = belief.copy()
    absorbed = False
    sampled_events = 0
    for step, action in enumerate(actions):
        blind = operators['A'][action] @ blind
        result['blind_survival'][step] = _mass(blind, normalized=False, name='blind prior')
        result['blind_costs'][step] = operators['costs'] @ blind
        audit['blind_prior'][step] = blind
        if absorbed:
            result['observed_probabilities'][step, ODOR_DIM] = 1
            continue
        branches, probabilities = _event(operators, observed, int(action))
        prior = branches.sum(axis=0, dtype=np.float64)
        result['observed_survival'][step] = _mass(prior, normalized=False, name='observed prior')
        result['observed_costs'][step] = operators['costs'] @ prior
        result['observed_probabilities'][step] = probabilities
        audit['observed_prior'][step] = prior
        observation = _sample(rng, probabilities)
        sampled_events += 1
        result['observations'][step] = observation
        if observation == ODOR_DIM:
            absorbed = True
            observed = np.zeros(STATE_DIM, dtype=np.float64)
        else:
            observed = _condition(branches, probabilities, observation)
        audit['observed_posterior'][step] = observed
    _require(all(np.isfinite(value).all() for value in (*result.values(), *audit.values())),
             'finite exact forecast targets')
    return result, audit, sampled_events


def generate_split(split_id, attempts, horizon, *, include_oracle=False, seed_namespace=SEED_NAMESPACE):
    """Generate a declared split; each attempted seed is used at most once.

    Split IDs 0/1 use epsilon=.12; split 2 uses epsilon=.30. Horizons 1..8
    are supported; the intended study uses TRAIN H2 and DEV H8. Returns
    {'data': dict, 'counts': dict, 'oracle': dict|None}. Data has exactly the
    ten requested arrays: prefix float32[N,9,31], lengths int64[N], actions
    and observations int64[N,H], two costs float64[N,H,4], two survival
    arrays float64[N,H], observed_probabilities float64[N,H,5], case_ids
    Unicode[N]. The optional oracle contains prefix_beliefs[N,9,8] and
    blind_prior/observed_prior/observed_posterior[N,H,8], all float64.

    Empty retained splits keep these exact ranks and dtypes. Counts disclose
    exclusions, event draws and independently drawn action-block sizes. A
    found forecast is retained; its observed suffix is label4 with zero cost.
    Future labels never enter the blind propagation or public prefix.
    """
    _require(type(split_id) is int and split_id in EPSILON, 'split_id in (0,1,2)')
    _require(type(attempts) is int and 0 <= attempts < 2**32, 'uint32 attempt count')
    _require(type(horizon) is int and 1 <= horizon <= MAX_HORIZON, 'horizon in 1..8')
    _require(type(include_oracle) is bool, 'Boolean include_oracle')
    _require(type(seed_namespace) is int and 0 <= seed_namespace < 2**32, 'uint32 seed namespace')
    operators = world(EPSILON[split_id])
    specs = {
        'prefix': ((PREFIX_STEPS + 1, FEATURE_DIM), np.float32),
        'lengths': ((), np.int64),
        'actions': ((horizon,), np.int64),
        'observations': ((horizon,), np.int64),
        'blind_costs': ((horizon, ACTION_DIM), np.float64),
        'blind_survival': ((horizon,), np.float64),
        'observed_costs': ((horizon, ACTION_DIM), np.float64),
        'observed_survival': ((horizon,), np.float64),
        'observed_probabilities': ((horizon, EVENT_DIM), np.float64),
        'case_ids': ((), np.dtype('<U64')),
    }
    data_rows = {name: [] for name in specs}
    audit_rows = {name: [] for name in ('prefix_beliefs', 'blind_prior',
                                       'observed_prior', 'observed_posterior')}
    counts = {'version': VERSION, 'seed_namespace': seed_namespace, 'split_id': split_id,
              'epsilon': EPSILON[split_id], 'horizon': horizon, 'attempts': attempts,
              'retained': 0, 'excluded_found': 0, 'prefix_found_by_step': [0] * PREFIX_STEPS,
              'initial_odor_draws': attempts, 'prefix_event_draws': 0,
              'prefix_action_draws': attempts * PREFIX_STEPS,
              'forecast_action_draws': 0, 'forecast_event_draws': 0,
              'forecast_found_cases': 0}
    for index in range(attempts):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [seed_namespace, split_id, index])))
        prefix, prefix_beliefs, prefix_events = _prefix(operators, rng)
        counts['prefix_event_draws'] += prefix_events
        if prefix is None:
            counts['excluded_found'] += 1
            counts['prefix_found_by_step'][prefix_events - 1] += 1
            continue
        actions = rng.integers(0, ACTION_DIM, size=horizon, dtype=np.int64)
        targets, audit, sampled_events = _forecast(operators, prefix_beliefs[-1], actions, rng)
        row = {'prefix': prefix, 'lengths': PREFIX_STEPS + 1, 'actions': actions,
               'case_ids': f'ns{seed_namespace}-split{split_id}-case{index:010d}', **targets}
        for name, values in data_rows.items():
            values.append(row[name])
        if include_oracle:
            audit_rows['prefix_beliefs'].append(prefix_beliefs)
            for name, value in audit.items():
                audit_rows[name].append(value)
        counts['retained'] += 1
        counts['forecast_action_draws'] += horizon
        counts['forecast_event_draws'] += sampled_events
        counts['forecast_found_cases'] += int(np.any(targets['observations'] == ODOR_DIM))
    retained = counts['retained']
    data = {name: np.array(data_rows[name], dtype=dtype).reshape((retained, *shape))
            for name, (shape, dtype) in specs.items()}
    oracle = None
    if include_oracle:
        oracle = {name: np.array(values, dtype=np.float64).reshape(
            (retained, PREFIX_STEPS + 1 if name == 'prefix_beliefs' else horizon, STATE_DIM))
            for name, values in audit_rows.items()}
    _require(retained + counts['excluded_found'] == attempts
             and sum(counts['prefix_found_by_step']) == counts['excluded_found']
             and all(np.isfinite(value).all() for name, value in data.items() if name != 'case_ids'),
             'finite complete split accounting')
    return {'data': data, 'counts': counts, 'oracle': oracle}
