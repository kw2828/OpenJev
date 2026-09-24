"""Independent scalar audit of frozen-dynamics constrained cost-head fitting.

The caller authenticates original process, source, and payload closure before
``audit``. This module never imports a learned model, generator, or solver.
Saved latent states are execution attestations, not independently replayed states.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from audit_finite_observation_learning import compare, multiply, probabilities, require, scalar_world

VERSION = 'finite-convex-readout-audit-v1'
NAMESPACE = 427260924
TRAIN_NAMESPACE = 426260924
PARENTS = ('factorized', 'matched_free', 'dense_free')
HEADS = ('original', 'solved')
ARMS = tuple(parent + '_' + head for parent in PARENTS for head in HEADS)
SEEDS = (426261001, 426261002, 426261003)
HORIZONS = (1, 2, 4, 8)
TARGETS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')
FIELDS = (*TARGETS, 'shuffled_blind_costs')
DATA_KEYS = {'prefix', 'lengths', 'actions', 'observations', 'case_ids', *TARGETS}
PREFIX_KEYS = {'prefix', 'lengths', 'case_ids', 'event_mask', 'endpoint_eligible', 'endpoint_rows'}

def validate_data(data, split, np):
    require(split in ('train', 'base') and set(data) == DATA_KEYS, 'exact shared-filter study data schema')
    n, h = len(data['case_ids']), 2 if split == 'train' else 8
    shapes = {'prefix': (n, 9, 31), 'lengths': (n,), 'actions': (n, h), 'observations': (n, h),
              'case_ids': (n,), 'blind_costs': (n, h, 4), 'observed_costs': (n, h, 4),
              'blind_survival': (n, h), 'observed_survival': (n, h), 'observed_probabilities': (n, h, 5)}
    require(n > 0 and all(data[k].shape == shape for k, shape in shapes.items()), 'nonempty exact shared-filter study shapes')
    require(data['prefix'].dtype == np.float32 and all(data[k].dtype == np.int64
            for k in ('lengths', 'actions', 'observations')), 'exact public input dtypes')
    require(all(data[k].dtype == np.float64 for k in TARGETS), 'float64 saved reference targets')
    require(data['case_ids'].dtype.kind == 'U' and len(set(data['case_ids'])) == n, 'unique Unicode case identities')
    split_id, attempts = (0, 512) if split == 'train' else (1, 128)
    namespace = TRAIN_NAMESPACE if split == 'train' else NAMESPACE
    pattern = re.compile(rf'ns{namespace}-split{split_id}-case([0-9]{{10}})')
    matches = [pattern.fullmatch(str(value)) for value in data['case_ids']]
    require(all(matches), 'fresh registered namespace and split')
    indices = [int(match.group(1)) for match in matches]
    require(indices == sorted(set(indices)) and all(index < attempts for index in indices), 'fixed attempted case roster')
    require(all(np.isfinite(value).all() for key, value in data.items() if key != 'case_ids'), 'finite public inputs and targets')
    p = data['prefix']
    require(np.all(data['lengths'] == 9) and np.all((p == 0) | (p == 1)) and not p[:, :, 10:].any()
            and not p[:, 0, :4].any() and np.all(p[:, 1:, :4].sum(-1) == 1)
            and np.all(p[:, :, 4:8].sum(-1) == 1) and not p[:, :, 8].any()
            and np.all(p[:, 0, 9] == 1) and not p[:, 1:, 9].any(), 'public history features without oracle input')
    require(np.all((data['actions'] >= 0) & (data['actions'] < 4))
            and np.all((data['observations'] >= 0) & (data['observations'] <= 4)), 'legal public actions and labels')
    found = data['observations'] == 4
    require(np.array_equal(found, np.maximum.accumulate(found, axis=1)), 'absorbing found suffix')
    return n, h


def reconstruct(data, prefix_oracle, exact_predictions, split, np, *, check=lambda: None):
    """Derive all targets and privileged prefix states from public tokens alone."""
    n, horizon = validate_data(data, split, np)
    state_array = None
    if prefix_oracle is not None:
        require(set(prefix_oracle) == {'prefix_state'}, 'single separately stored oracle input')
        state_array = prefix_oracle['prefix_state']
        require(state_array.shape == (n, 8) and state_array.dtype == np.float64
                and np.isfinite(state_array).all() and np.all(state_array >= 0)
                and np.all(np.abs(state_array.sum(-1) - 1) <= 1e-12), 'normalized float64 privileged prefix state')
    if exact_predictions is not None:
        require(set(exact_predictions) == set(TARGETS), 'exact control has five fields')
        for name in TARGETS:
            value = exact_predictions[name]
            require(value.shape == data[name].shape and value.dtype == np.float64 and np.isfinite(value).all(),
                    'finite float64 exact-control output: ' + name)
    world = scalar_world(.12)
    targets = {key: np.zeros_like(data[key]) for key in TARGETS}
    prefix_states = np.zeros((n, 8), np.float64)
    uniform_costs = np.zeros_like(data['blind_costs'])
    for case in range(n):
        check()
        prefix = data['prefix'][case]
        odor = int(prefix[0, 4:8].argmax())
        state = [value / 8 for value in world['emission'][odor]]
        total = math.fsum(state)
        state = [value / total for value in state]
        for row in prefix[1:]:
            action, odor = int(row[:4].argmax()), int(row[4:8].argmax())
            branches, mass = probabilities(world, state, action)
            state = [value / mass[odor] for value in branches[odor]]
        prefix_states[case] = state
        blind, observed, uniform = state.copy(), state.copy(), [1 / 8] * 8
        absorbed = False
        for step, action in enumerate(data['actions'][case]):
            action = int(action)
            blind = multiply(world['A'][action], blind)
            uniform = multiply(world['A'][action], uniform)
            targets['blind_costs'][case, step] = multiply(world['costs'], blind)
            targets['blind_survival'][case, step] = math.fsum(blind)
            uniform_costs[case, step] = multiply(world['costs'], uniform)
            if absorbed:
                targets['observed_probabilities'][case, step, 4] = 1
                continue
            branches, mass = probabilities(world, observed, action)
            prior = [math.fsum(branch[j] for branch in branches) for j in range(8)]
            targets['observed_costs'][case, step] = multiply(world['costs'], prior)
            targets['observed_survival'][case, step] = math.fsum(prior)
            targets['observed_probabilities'][case, step] = mass
            odor = int(data['observations'][case, step])
            if odor == 4:
                absorbed, observed = True, [0.] * 8
            else:
                observed = [value / mass[odor] for value in branches[odor]]
    prefix_error = None if state_array is None else float(np.max(np.abs(prefix_states - state_array)))
    target_errors = {name: float(np.max(np.abs(targets[name] - data[name]))) for name in TARGETS}
    exact_errors = {} if exact_predictions is None else {name: float(np.max(np.abs(targets[name] - exact_predictions[name]))) for name in TARGETS}
    require((prefix_error is None or prefix_error <= 1e-12) and all(error <= 1e-12 for error in target_errors.values()),
            'independent public-history posterior and reference targets')
    require(all(error <= 1e-12 for error in exact_errors.values()), 'exact/exact oracle agreement at every saved horizon')
    return {'cases': n, 'horizon': horizon, 'prefix_maximum_absolute_error': prefix_error,
            'target_maximum_absolute_errors': target_errors, 'oracle_maximum_absolute_errors': exact_errors}, targets, uniform_costs


def validate_predictions(data, predictions, np):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in SEEDS for field in FIELDS}
    require(set(predictions) == expected, 'complete eighteen-view prediction fields')
    n = len(data['case_ids'])
    for name, value in predictions.items():
        field = name.rsplit('__', 1)[-1]
        shape = (n, 8, 5) if field == 'observed_probabilities' else (n, 8) if field.endswith('survival') else (n, 8, 4)
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), 'finite float64 prediction: ' + name)
        if field.endswith('survival') or field == 'observed_probabilities':
            require(np.all((value >= 0) & (value <= 1 + 1e-12)), 'bounded predicted probability: ' + name)
        if field == 'observed_probabilities':
            require(np.all(np.abs(value.sum(-1) - 1) <= 1e-6), 'normalized predicted event law')
            require(np.all(value[data['observed_probabilities'] > 0] > 0), 'positive oracle support requires positive predictions')


def rows_for(data, predictions, uniform_costs, np, *, check=lambda: None):
    validate_predictions(data, predictions, np)
    n = len(data['case_ids'])
    rows, baselines = [], []
    for horizon in HORIZONS:
        h = horizon - 1
        target, reference = data['blind_costs'][:, h], uniform_costs[:, h]
        chosen = (reference <= reference.min(-1, keepdims=True) + 1e-12).argmax(-1)
        baselines.append({'regime': 'base', 'horizon': horizon, 'cases': n,
            'blind_cost_mse': math.fsum(float(x) ** 2 for x in (reference - target).ravel()) / (n * 4),
            'blind_regret': math.fsum(float(target[i, chosen[i]] - min(target[i])) for i in range(n)) / n})
    for arm in ARMS:
        for seed in SEEDS:
            check()
            p = {field: predictions[f'{arm}__{seed}__{field}'] for field in FIELDS}
            for horizon in HORIZONS:
                h = horizon - 1
                blind = data['blind_costs'][:, h]
                selected, shuffled = p['blind_costs'][:, h].argmin(-1), p['shuffled_blind_costs'][:, h].argmin(-1)
                target_p = data['observed_probabilities'][:, h]
                kl = math.fsum(float(q) * (math.log(float(q)) - math.log(float(p['observed_probabilities'][i, h, o])))
                               for i in range(n) for o, q in enumerate(target_p[i]) if q > 0) / n
                row = {'arm': arm, 'seed': seed, 'regime': 'base', 'horizon': horizon, 'cases': n,
                    'blind_cost_mse': math.fsum(float(x) ** 2 for x in (p['blind_costs'][:, h] - blind).ravel()) / (n * 4),
                    'blind_regret': math.fsum(float(blind[i, selected[i]] - min(blind[i])) for i in range(n)) / n,
                    'blind_survival_mae': math.fsum(abs(float(x)) for x in p['blind_survival'][:, h] - data['blind_survival'][:, h]) / n,
                    'observed_cost_mse': math.fsum(float(x) ** 2 for x in (p['observed_costs'][:, h] - data['observed_costs'][:, h]).ravel()) / (n * 4),
                    'observed_survival_mae': math.fsum(abs(float(x)) for x in p['observed_survival'][:, h] - data['observed_survival'][:, h]) / n,
                    'observed_kl': kl,
                    'shuffled_blind_regret': math.fsum(float(blind[i, shuffled[i]] - min(blind[i])) for i in range(n)) / n}
                require(all(math.isfinite(value) for value in row.values() if type(value) is float), 'finite independently computed metrics')
                rows.append(row)
    return rows, baselines


def gates(rows, baselines, counts):
    """Three separate predeclared criteria for every learned arm."""
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    reference = {row['horizon']: row for row in baselines}
    require(len(rows) == len(keyed) == 72 and set(keyed)
            == {(arm, seed, h) for arm in ARMS for seed in SEEDS for h in HORIZONS}, 'complete metric gate roster')
    require(len(baselines) == len(reference) == 4 and set(reference) == set(HORIZONS), 'complete reference gate roster')
    results = {}
    for arm in ARMS:
        criteria = {}
        for kind, horizons in (('SHORT_HORIZON_LEARNING', (1, 2)), ('BLIND_EXTRAPOLATION', (4, 8)),
                               ('OBSERVED_FILTERING_EXTRAPOLATION', (4, 8))):
            conditions = {'minimum_train': counts['train'] >= 256, 'minimum_development': counts['base'] >= 64}
            for horizon in horizons:
                if kind != 'OBSERVED_FILTERING_EXTRAPOLATION':
                    conditions[f'h{horizon}_positive_reference_mse'] = reference[horizon]['blind_cost_mse'] > 0
                    conditions[f'h{horizon}_positive_reference_regret'] = reference[horizon]['blind_regret'] > 0
                for seed in SEEDS:
                    row = keyed[arm, seed, horizon]
                    if kind != 'OBSERVED_FILTERING_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_half_mse'] = row['blind_cost_mse'] <= .5 * reference[horizon]['blind_cost_mse']
                        conditions[f'{seed}_h{horizon}_half_regret'] = row['blind_regret'] <= .5 * reference[horizon]['blind_regret']
                    if kind != 'BLIND_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_observed_kl'] = row['observed_kl'] <= .1
            if kind == 'BLIND_EXTRAPOLATION':
                for seed in SEEDS:
                    conditions[f'{seed}_h8_survival'] = keyed[arm, seed, 8]['blind_survival_mae'] <= .05
            passed = all(conditions.values())
            criteria[kind] = {'passed': passed, 'status': kind + ('_PASS' if passed else '_FAIL'), 'conditions': conditions}
        results[arm] = criteria
    return results


def reconstruct_prefix(data, endpoints, split, np, *, check=lambda: None):
    """Validate the entire attempted population and condition only on public events."""
    require(split == 'base' and set(data) == PREFIX_KEYS, 'exact all-attempt prefix fields')
    n = 512 if split == 'train' else 128
    shapes = {'prefix': (n, 9, 31), 'lengths': (n,), 'case_ids': (n,), 'event_mask': (n, 9),
              'endpoint_eligible': (n,), 'endpoint_rows': (n,)}
    require(all(data[key].shape == shape for key, shape in shapes.items()), 'complete all-attempt prefix geometry')
    require(data['prefix'].dtype == np.float32 and data['lengths'].dtype == np.int64
            and data['endpoint_rows'].dtype == np.int64 and data['event_mask'].dtype == np.bool_
            and data['endpoint_eligible'].dtype == np.bool_ and data['case_ids'].dtype == np.dtype('U64'),
            'exact all-attempt prefix dtypes')
    expected_ids = np.array([f'ns{NAMESPACE}-split{0 if split == "train" else 1}-case{i:010d}'
                             for i in range(n)], dtype='U64')
    require(np.array_equal(data['case_ids'], expected_ids), 'every allocated attempt appears exactly once in order')
    p, lengths, mask = data['prefix'], data['lengths'], data['event_mask']
    require(np.isfinite(p).all() and np.all((p == 0) | (p == 1)), 'finite binary public prefix tokens')
    require(np.all((lengths >= 2) & (lengths <= 9))
            and np.array_equal(mask, np.arange(9)[None, :] < lengths[:, None]), 'first-found inclusive event mask')
    require(not p[~mask].any() and not p[:, :, 10:].any()
            and not p[:, 0, :4].any() and not p[:, 0, 8].any()
            and np.all(p[:, 0, 4:8].sum(-1) == 1) and np.all(p[:, 0, 9] == 1)
            and not p[:, 1:, 9].any(), 'reset event and zero unused/padding features')
    require(np.all(p[:, 1:, :4].sum(-1)[mask[:, 1:]] == 1)
            and np.all(p[:, :, 4:9].sum(-1)[mask] == 1), 'one public action and realized event per active transition')
    found = p[:, :, 8] == 1
    eligible = data['endpoint_eligible']
    require(np.array_equal(eligible, ~found.any(axis=1)) and np.all(lengths[eligible] == 9)
            and np.all(found.sum(-1) == (~eligible).astype(np.int64))
            and np.all(found[np.flatnonzero(~eligible), lengths[~eligible] - 1]), 'first found terminates exactly once')
    selected = np.flatnonzero(eligible)
    rows = data['endpoint_rows']
    require(np.all(rows[~eligible] == -1) and np.array_equal(rows[eligible], np.arange(len(selected))),
            'explicit sequential survivor endpoint mapping')
    require(len(endpoints['case_ids']) == len(selected)
            and np.array_equal(endpoints['case_ids'], data['case_ids'][selected])
            and np.array_equal(endpoints['prefix'], p[selected])
            and np.array_equal(endpoints['lengths'], lengths[selected]), 'endpoint inputs equal precisely the surviving public prefixes')
    world = scalar_world(.12)
    reference = np.zeros((n, 9, 5), np.float64)
    posterior = np.zeros((n, 8), np.float64)
    realized = p[:, :, 4:9].argmax(-1)
    for case in range(n):
        check()
        reset = [math.fsum(value / 8 for value in row) for row in world['emission']]
        reference[case, 0, :4] = reset
        odor = int(realized[case, 0])
        state = [value / (8 * reset[odor]) for value in world['emission'][odor]]
        for step in range(1, int(lengths[case])):
            branches, masses = probabilities(world, state, int(p[case, step, :4].argmax()))
            reference[case, step] = masses
            odor = int(realized[case, step])
            require(masses[odor] > 0, 'positive known-law probability of every public event')
            state = [0.] * 8 if odor == 4 else [value / masses[odor] for value in branches[odor]]
        posterior[case] = state
    event_nll = math.fsum(-math.log(float(reference[i, j, realized[i, j]]))
                         for i, j in zip(*np.nonzero(mask), strict=True))
    found_by_step = [int(found[:, step].sum()) for step in range(1, 9)]
    return {'attempts': n, 'retained': len(selected), 'discarded_found': n - len(selected),
            'valid_events': int(mask.sum()), 'prefix_found_by_step': found_by_step,
            'oracle_mean_nll': event_nll / int(mask.sum())}, posterior[selected], reference



def probability_matrix(value, np):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (4, 8) and np.isfinite(result).all(), 'finite four-by-eight probability matrix')
    return result


def simplex_violation(matrix):
    return max(max(0., -float(matrix.min())),
               max(abs(math.fsum(float(matrix[a, s]) for a in range(4)) - 1.) for s in range(8)))


def project_simplex(matrix, np):
    """Independent four-entry Euclidean projection, once per state column."""
    result = np.empty((4, 8), np.float64)
    for s in range(8):
        ordered = sorted((float(matrix[a, s]) for a in range(4)), reverse=True)
        active = [k for k in range(1, 5) if ordered[k - 1] > (math.fsum(ordered[:k]) - 1.) / k]
        require(bool(active), 'nonempty simplex projection active set')
        k = active[-1]
        threshold = (math.fsum(ordered[:k]) - 1.) / k
        for a in range(4):
            result[a, s] = max(float(matrix[a, s]) - threshold, 0.)
    return result


def validate_states(states, data, np, *, train):
    names = {'blind_states', 'observed_states'} if train else {
        head + '_' + field for head in HEADS for field in ('blind_states', 'observed_states', 'shuffled_states', 'prefix_states', 'shuffled_prefix_states')}
    require(set(states) == names, 'exact saved latent-state fields')
    n, h = len(data['case_ids']), 2 if train else 8
    for name, value in states.items():
        shape = (n, 8) if 'prefix_states' in name else (n, h, 8)
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all()
                and np.all(value >= 0) and np.all(value.sum(-1) <= 1 + 1e-12),
                'finite nonnegative subprobability prior states: ' + name)
        if 'observed' in name:
            previously_found = np.concatenate((np.zeros((n, 1), bool), data['observations'][:, :-1] == 4), axis=1)
            require(not value[previously_found].any(), 'absorbed observed prior is exactly zero')
    if not train:
        for field in ('blind_states', 'observed_states', 'shuffled_states', 'prefix_states', 'shuffled_prefix_states'):
            require(states['original_' + field].tobytes()
                    == states['solved_' + field].tobytes(), 'bitwise unchanged latent route: ' + field)
        for head in HEADS:
            for field in ('prefix_states', 'shuffled_prefix_states'):
                require(np.all(np.abs(states[head + '_' + field].sum(-1) - 1) <= 1e-12), 'normalized saved learned prefix state')


def direct_objective(states, data, matrix, np, *, check=lambda: None):
    """Original sum of two MSEs, evaluated from raw states without a Gram matrix."""
    validate_states(states, data, np, train=True)
    p = probability_matrix(matrix, np)
    n, h = states['blind_states'].shape[:2]
    denominator = 4 * n * h
    gradients = [[[] for _ in range(8)] for _ in range(4)]
    errors = []
    for route in ('blind', 'observed'):
        x, targets = states[route + '_states'], data[route + '_costs']
        require(targets.shape == (n, h, 4) and targets.dtype == np.float64 and np.isfinite(targets).all(),
                'finite float64 training cost target')
        for i in range(n):
            check()
            for t in range(h):
                for a in range(4):
                    error = math.fsum((.25 - float(p[a, s])) * float(x[i, t, s]) for s in range(8)) - float(targets[i, t, a])
                    errors.append(error * error)
                    for s in range(8):
                        gradients[a][s].append(error * float(x[i, t, s]))
    objective = math.fsum(errors) / denominator
    gradient = [[-2. * math.fsum(gradients[a][s]) / denominator for s in range(8)] for a in range(4)]
    gap = math.fsum(float(p[a, s]) * gradient[a][s] for a in range(4) for s in range(8)) \
        - math.fsum(min(gradient[a][s] for a in range(4)) for s in range(8))
    require(math.isfinite(objective) and all(math.isfinite(x) for row in gradient for x in row)
            and math.isfinite(gap), 'finite scalar optimization certificate')
    primal = simplex_violation(p)
    return {'objective': objective, 'gradient': gradient, 'fw_gap': gap,
            'simplex_violation': primal, 'minimum_probability': float(p.min()),
            'maximum_probability': float(p.max()), 'passed': primal <= 1e-12 and -1e-12 <= gap <= 1e-8}


def verify_solve(record, states, data, np, *, check=lambda: None):
    """Require optimizer success and a separately reconstructed certificate."""
    initial = probability_matrix(record['initial_probabilities'], np)
    raw = probability_matrix(record['raw_probabilities'], np)
    final = probability_matrix(record['probabilities'], np)
    require(simplex_violation(initial) <= 1e-12, 'feasible original softmax head')
    initial_check = direct_objective(states, data, initial, np, check=check)
    raw_check = direct_objective(states, data, raw, np, check=check)
    final_check = direct_objective(states, data, final, np, check=check)
    projected = project_simplex(raw, np)
    repair_max = float(np.max(np.abs(projected - raw)))
    repair_l2 = math.sqrt(math.fsum(float(value) ** 2 for value in (projected - raw).ravel()))
    require(raw_check['simplex_violation'] <= 1e-10 and repair_max <= 1e-10,
            'raw result admits only the registered tiny simplex repair')
    require(np.max(np.abs(final - projected)) <= 1e-14, 'saved final head equals one Euclidean simplex projection')
    compare(record['objective_initial'], initial_check['objective'], path='original TRAIN cost objective')
    compare(record['objective_raw'], raw_check['objective'], path='raw TRAIN cost objective')
    compare(record['raw_feasibility'], raw_check['simplex_violation'], path='raw simplex feasibility')
    compare(record['projection'], {'applied': True, 'max_abs': repair_max, 'l2': repair_l2}, path='single tiny projection')
    compare(record['certificate'], final_check, path='independent direct-residual certificate')
    require(final_check['simplex_violation'] <= 1e-12 and -1e-12 <= final_check['fw_gap'] <= 1e-8
            and final_check['objective'] <= initial_check['objective'] + 1e-12,
            'independently certified feasible globally bounded head with nonincreasing TRAIN objective')
    solver, work = record['solver'], record['work']
    require(solver['success'] is True and type(solver['status']) is int and solver['status'] == 0
            and type(solver['iterations']) is int and 0 <= solver['iterations'] <= 2000,
            'original SLSQP success within fixed iteration budget')
    require(set(work) == {'objective_calls', 'gradient_calls', 'iteration_callbacks',
                         'direct_objective_gradient_passes', 'simplex_projection_calls'}
            and all(type(value) is int and value >= 0 for value in work.values())
            and 0 < work['objective_calls'] <= 10000 and work['gradient_calls'] == work['objective_calls']
            and work['iteration_callbacks'] <= 2000 and work['direct_objective_gradient_passes'] == 3
            and work['simplex_projection_calls'] == 1, 'bounded declared solver and certificate work')
    require(all(type(solver[key]) is int and solver[key] >= 0 for key in ('nfev', 'njev')),
            'separately retained SciPy memoized evaluation counters')
    require(record['version'] == 'finite-convex-readout-v1' and record['status'] == 'SOLVE_PASS'
            and record['complete'] is True and record['failure_reasons'] == [], 'successful single solve without retry')
    costs = probability_matrix(record['cost_matrix'], np)
    require(np.max(np.abs(costs - (.25 - final))) <= 1e-15, 'cost matrix equals centered simplex probabilities')
    return {'objective_initial': initial_check['objective'], 'objective_final': final_check['objective'],
            'objective_change': final_check['objective'] - initial_check['objective'],
            'simplex_violation': final_check['simplex_violation'], 'fw_gap': final_check['fw_gap'],
            'raw_feasibility': raw_check['simplex_violation'], 'projection_max_abs': repair_max,
            'independently_certified': True}


def map_costs(states, matrix, np):
    result = np.empty((*states.shape[:2], 4), np.float64)
    for i in range(states.shape[0]):
        for h in range(states.shape[1]):
            for a in range(4):
                result[i, h, a] = math.fsum(float(matrix[a, s]) * float(states[i, h, s]) for s in range(8))
    return result


def verify_views(parent, seed, states, predictions, solve, data, np):
    validate_states(states, data, np, train=False)
    mapped_errors = {}
    for head in HEADS:
        matrix = .25 - probability_matrix(solve['initial_probabilities' if head == 'original' else 'probabilities'], np)
        prefix = f'{parent}_{head}__{seed}__'
        for route, field in (('blind', 'blind_costs'), ('observed', 'observed_costs'), ('shuffled', 'shuffled_blind_costs')):
            state = states[head + '_' + route + '_states']
            mapped = map_costs(state, matrix, np)
            error = float(np.max(np.abs(mapped - predictions[prefix + field])))
            require(error <= 1e-12, 'independent head-to-prior mapping: ' + prefix + field)
            mapped_errors[head + '_' + route] = error
            if route != 'shuffled':
                require(np.max(np.abs(state.sum(-1) - predictions[prefix + route + '_survival'])) <= 1e-12,
                        'saved prior mass equals route survival')
    for field in ('blind_survival', 'observed_survival', 'observed_probabilities'):
        require(predictions[f'{parent}_original__{seed}__{field}'].tobytes()
                == predictions[f'{parent}_solved__{seed}__{field}'].tobytes(), 'bitwise unchanged non-cost prediction: ' + field)
    return {'parent': parent, 'seed': seed, 'latent_states_bitwise_equal': True,
            'event_and_survival_bitwise_equal': True, 'head_mapping_maximum_absolute_errors': mapped_errors}

CONFIG = {'dev_seed_namespace': NAMESPACE, 'dev_attempts': 128, 'dev_horizon': 8,
          'batch_size': 64, 'parent_seeds': list(SEEDS), 'parent_arms': list(PARENTS),
          'train_namespace': TRAIN_NAMESPACE, 'train_horizon': 2,
          'solver_maxiter': 2000, 'solver_ftol': 1e-12, 'solver_max_objective_calls': 10000,
          'solver_gap_tolerance': 1e-8, 'solver_repair_tolerance': 1e-10, 'solver_nonincrease_tolerance': 1e-12}
ROUTES = ('train_blind', 'train_observed', *(head + '_' + route for head in HEADS for route in ('blind', 'observed', 'shuffled')))
COMMON_FILES = {'config.json', 'train.npz', 'base.npz', 'base-prefix.npz', 'base-oracle.npz',
                'oracle-base.npz', 'oracle-base-check.json', 'solves.jsonl', 'solve-barrier.json',
                'prediction-times.jsonl', 'predictions-base.npz'}


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular opaque input or payload')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def read_json(path, *, lines=False):
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


def validate_metadata(folder, summary, *, check=lambda: None):
    """Join original opaque checkpoint/TRAIN provenance before any array decoding."""
    require(summary['version'] == 'finite-convex-readout-v1' and summary['config'] == CONFIG, 'exact frozen diagnostic configuration')
    expected_files = COMMON_FILES | {f'{prefix}-{parent}-{seed}.{extension}'
        for parent in PARENTS for seed in SEEDS
        for prefix, extension in (('states-train', 'npz'), ('states-base', 'npz'), ('solve', 'json'))}
    require(set(summary['files']) == expected_files
            and {path.name for path in folder.iterdir()} == expected_files | {'summary.json'}, 'complete 38-payload successful solve run')
    for name in sorted(expected_files):
        check()
        compare(summary['files'][name], descriptor(folder / name), path='opaque saved payload ' + name)
    compare(read_json(folder / 'config.json'), CONFIG, path='saved frozen configuration')
    upstream = summary['upstream']
    prior_folder, prior_run = Path(upstream['folder']), Path(upstream['run'])
    require(prior_folder.is_absolute() and prior_folder.resolve() == prior_folder and not prior_folder.is_symlink()
            and prior_run.is_absolute() and prior_run.resolve() == prior_run and not prior_run.is_symlink()
            and prior_run.parent == prior_folder, 'original upstream directory identity')
    def parent_pin(path):
        name = str(path.relative_to(prior_folder))
        require(name in upstream['files'], 'parent input belongs to authenticated upstream inventory')
        compare(descriptor(path), upstream['files'][name], path='original input byte descriptor ' + name)
    parent_pin(prior_run / 'summary.json')
    parent_summary = read_json(prior_run / 'summary.json')
    parent_pin(prior_run / 'train.npz')
    compare(summary['original_train_descriptor'], descriptor(prior_run / 'train.npz'), path='original TRAIN bytes')
    compare(summary['files']['train.npz'], summary['original_train_descriptor'], path='byte-identical copied TRAIN')
    compare(summary['original_train_counts'], parent_summary['dataset_counts']['train'], path='upstream all-attempt TRAIN provenance')
    parent_rows = {(row['arm'], row['seed']): row for row in parent_summary['fits']}
    require(set(parent_rows) == {(parent, seed) for parent in PARENTS for seed in SEEDS}, 'all nine original trained checkpoints')
    expected_order = []
    for index, seed in enumerate(SEEDS):
        order = PARENTS[index % 3:] + PARENTS[:index % 3]
        expected_order.extend((parent, seed) for parent in order)
    solves = summary['solves']
    require([(row['parent'], row['seed']) for row in solves] == expected_order, 'all nine single solves in registered rotated order')
    compare(read_json(folder / 'solves.jsonl', lines=True), solves, path='complete durable solve journal')
    for row in solves:
        check()
        parent, seed = row['parent'], row['seed']
        source = prior_run / f'{parent}-{seed}.npz'
        parent_pin(source)
        compare(row['source_checkpoint'], {'path': str(source), **descriptor(source)}, path='original checkpoint descriptor')
        compare({key: row['source_checkpoint'][key] for key in ('sha256', 'bytes')},
                {key: parent_rows[parent, seed]['checkpoint'][key] for key in ('sha256', 'bytes')}, path='original final checkpoint')
        require(row['source_state_sha256'] == row['model_state_after'] == parent_rows[parent, seed]['final_state_sha256'],
                'frozen semantic state before and after extraction/solve')
        compare(row['original_cost_matrix'], parent_rows[parent, seed]['readout_final']['matrix'], path='original learned head from authenticated parent')
        require(row['frozen_weights'] is True and row['optimizer_weight_updates'] == 0 and row['horizon'] == 2
                and row['training_cases'] == summary['original_train_counts']['retained'], 'original TRAIN support and no dynamics updates')
        for key in ('load_seconds', 'extraction_seconds', 'solve_seconds', 'train_original_head_max_error'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'finite solve timing/error')
        require(row['train_original_head_max_error'] <= 1e-12, 'original TRAIN forward matches linear head')
        compare(read_json(folder / f'solve-{parent}-{seed}.json'), row, path='durable single solve record')
        state_name = f'states-train-{parent}-{seed}.npz'
        compare(row['train_state_file'], {'path': state_name, **summary['files'][state_name]}, path='TRAIN state artifact')
    barrier = {'solves': [{'parent': row['parent'], 'seed': row['seed'],
                    'file': f"solve-{row['parent']}-{row['seed']}.json",
                    **summary['files'][f"solve-{row['parent']}-{row['seed']}.json"]} for row in solves],
               'completed_solves': 9, 'dev_generation_count': 0}
    compare(summary['solve_barrier'], barrier, path='all successful TRAIN solves before fresh DEV attestation')
    compare(read_json(folder / 'solve-barrier.json'), barrier, path='durable TRAIN-only solve barrier')
    times = summary['prediction_times']
    require([(row['parent'], row['seed'], row['head']) for row in times]
            == [(parent, seed, head) for parent in PARENTS for seed in SEEDS for head in HEADS], 'all eighteen independent replay views')
    for row in times:
        require(row['arm'] == row['parent'] + '_' + row['head'] and row['horizon'] == 8
                and row['cases'] == summary['dataset_counts']['base']['retained']
                and row['model_state_before'] == row['model_state_after'] == parent_rows[row['parent'], row['seed']]['final_state_sha256'],
                'paired view state joins to exact original checkpoint')
        require(type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0
                and 0 <= row['original_head_max_error'] <= 1e-12, 'finite replay time and original head validation')
    compare(read_json(folder / 'prediction-times.jsonl', lines=True), times, path='durable replay timing/state journal')
    compare(read_json(folder / 'oracle-base-check.json'), summary['oracle_checks'], path='saved exact-control errors')
    return {'payloads': len(expected_files), 'parent_models': 9, 'head_solves': 9, 'views': 18,
            'upstream_train_reused': True, 'upstream_dev_decoded': False, 'checkpoint_decodes': 0,
            'original_closure_admission_external': True, 'historical_barrier_independently_replayed': False}


def validate_work(summary, train, base, np, *, check=lambda: None):
    from audit_finite_factorized_dynamics import ENDPOINT_WORK_KEYS, FACTORIZATION_WORK, endpoint_work

    work = summary['structural_work']
    require(set(work) == set(PARENTS) and all(set(work[parent]) == set(ROUTES) for parent in PARENTS), 'all 24 frozen-replay work routes')
    expected = {parent: {route: dict.fromkeys(ENDPOINT_WORK_KEYS, 0) for route in ROUTES} for parent in PARENTS}
    for parent in PARENTS:
        for route in ROUTES:
            require(set(work[parent][route]) == ENDPOINT_WORK_KEYS
                    and all(type(value) is int and value >= 0 for value in work[parent][route].values()), 'nonnegative complete structural counters')
            data = train if route.startswith('train_') else base
            for _seed in SEEDS:
                for start in range(0, len(data['case_ids']), 64):
                    check()
                    values = endpoint_work(data['observations'][start:start + 64], route.endswith('_observed'), np)
                    if parent == 'factorized':
                        values.update({key: 2 * value for key, value in FACTORIZATION_WORK.items()})
                        values.update(dict.fromkeys(('operator_observed_softmax_calls', 'prefix_operator_softmax_calls', 'reset_emission_softmax_calls'), 0))
                    values['cost_head_softmax_calls'] = values['cost_readout_calls']
                    values['cost_head_probability_rows'] = 8 * values['cost_readout_calls']
                    for key, value in values.items():
                        expected[parent][route][key] += value
    compare(work, expected, path='source-qualified frozen forwards reconstructed from public geometry')
    ntrain, nbase = len(train['case_ids']), len(base['case_ids'])
    bt, bd = math.ceil(ntrain / 64), math.ceil(nbase / 64)
    counts = {'train_array_decodes': 1, 'checkpoint_decodes': 9, 'model_constructions': 9,
        'oracle_model_constructions': 1, 'original_head_exports': 9, 'original_head_softmaxes': 9,
        'train_blind_calls': 9 * bt, 'train_observed_calls': 9 * bt,
        'train_readout_validation_products': 18 * bt, 'train_readout_validation_rows': 36 * ntrain,
        'solver_calls': 9, 'completed_solves': 9, 'dev_generation_count': 1,
        'oracle_blind_rollouts': bd, 'oracle_observed_rollouts': bd,
        'evaluation_blind_calls': 18 * bd, 'evaluation_observed_calls': 18 * bd, 'evaluation_shuffled_calls': 18 * bd,
        'head_matrix_products': 54 * bd, 'head_matrix_rows': 54 * nbase * 8,
        'evaluation_case_views': 18 * nbase, 'checkpoint_writes': 0, 'optimizer_steps': 0,
        'external_model_calls': 0, 'teacher_calls': 0, 'native_calls': 0}
    compare(summary['counts'], counts, path='complete executed-count roster')
    return {'routes': 24, 'counted_original_head_work_inside_both_views': True,
            'separate_posthoc_matrix_products': counts['head_matrix_products'],
            'forward_geometry_verified': True, 'historical_model_execution_replayed': False}


def paired_comparisons(rows):
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    fields = ('blind_cost_mse', 'blind_regret', 'observed_cost_mse', 'observed_kl',
              'blind_survival_mae', 'observed_survival_mae', 'shuffled_blind_regret')
    result = []
    for parent in PARENTS:
        for seed in SEEDS:
            for horizon in HORIZONS:
                old, new = (keyed[parent + '_' + head, seed, horizon] for head in HEADS)
                result.append({'parent': parent, 'seed': seed, 'horizon': horizon,
                    'metrics': {name: {'original': old[name], 'solved': new[name],
                        'difference': new[name] - old[name],
                        'relative_reduction': (old[name] - new[name]) / old[name] if old[name] > 0 else None} for name in fields}})
    return result


def audit(folder, *, check=lambda: None):
    """Audit only after the caller admits the original successful closed producer."""
    import numpy as np

    folder = Path(folder)
    require(folder.is_absolute() and folder.is_dir() and folder.resolve() == folder and not folder.is_symlink(), 'absolute original output directory')
    summary = read_json(folder / 'summary.json')
    metadata = validate_metadata(folder, summary, check=check)
    counters = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0}

    def load(name):
        check()
        with np.load(folder / name, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'unique saved array members')
            value = {key: archive[key] for key in archive.files}
        counters['array_decodes'] += 1
        return value

    def hashes(values):
        return {name: hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest() for name, value in values.items()}

    train, base = load('train.npz'), load('base.npz')
    prefix, oracle, exact = load('base-prefix.npz'), load('base-oracle.npz'), load('oracle-base.npz')
    train_record, train_targets, _ = reconstruct(train, None, None, 'train', np, check=check)
    base_record, base_targets, uniform = reconstruct(base, oracle, exact, 'base', np, check=check)
    prefix_record, prefix_states, _ = reconstruct_prefix(prefix, base, 'base', np, check=check)
    require(np.max(np.abs(prefix_states - oracle['prefix_state'])) <= 1e-12, 'public all-attempt posterior matches separately stored exact reference')
    declared = summary['dataset_counts']['base']
    require(declared['version'] == 'finite-prefix-learning-data-v1' and declared['seed_namespace'] == NAMESPACE
            and declared['split_id'] == 1 and declared['epsilon'] == .12 and declared['horizon'] == 8,
            'fresh fixed-world development source identity')
    for key, reported in (('attempts', 'attempted'), ('retained', 'retained'), ('discarded_found', 'discarded_found'),
                          ('valid_events', 'valid_prefix_events'), ('prefix_found_by_step', 'prefix_found_by_step')):
        compare(prefix_record[key], declared[reported], path='public prefix accounting ' + key)
    found = base['observations'] == 4
    found_cases = found.any(axis=1)
    event_draws = np.where(found_cases, found.argmax(axis=1) + 1, 8)
    require(declared['initial_odor_draws'] == 128 and declared['prefix_action_draws'] == 1024
            and declared['prefix_event_draws'] == prefix_record['valid_events'] - 128
            and declared['forecast_action_draws'] == 8 * len(base['case_ids'])
            and declared['forecast_found_cases'] == int(found_cases.sum())
            and declared['forecast_event_draws'] == int(event_draws.sum()), 'all fresh sampler draw counts reconstructed from public labels')
    require(summary['original_train_counts']['retained'] == len(train['case_ids'])
            and summary['original_train_counts']['seed_namespace'] == TRAIN_NAMESPACE
            and summary['original_train_counts']['attempted'] == 512, 'original TRAIN lineage support')
    require(not set(train['case_ids']) & set(base['case_ids']), 'fresh DEV identities distinct from reused TRAIN')
    producer_errors = {name: float(np.max(np.abs(exact[name] - base[name]))) for name in TARGETS}
    compare(summary['oracle_checks'], producer_errors, path='independent exact-control receipt errors')
    predictions = load('predictions-base.npz')
    validate_predictions(base, predictions, np)
    solve_rows = {(row['parent'], row['seed']): row for row in summary['solves']}
    times = {(row['parent'], row['seed'], row['head']): row for row in summary['prediction_times']}
    certificates, invariants = [], []
    for parent in PARENTS:
        for seed in SEEDS:
            check()
            row = solve_rows[parent, seed]
            states = load(f'states-train-{parent}-{seed}.npz')
            compare(row['train_state_array_sha256'], hashes(states), path='saved TRAIN latent state byte hashes')
            result = row['solver_result']
            compare(row['original_probabilities'], result['initial_probabilities'], path='original head initialization of single solve')
            p0 = probability_matrix(row['original_probabilities'], np)
            require(np.max(np.abs(probability_matrix(row['original_cost_matrix'], np) - (.25 - p0))) <= 1e-15,
                    'original cost matrix uses surviving-mass linear simplex head')
            certificates.append({'parent': parent, 'seed': seed,
                                 **verify_solve(result, states, {**train, **train_targets}, np, check=check)})
            base_states = load(f'states-base-{parent}-{seed}.npz')
            invariants.append(verify_views(parent, seed, base_states, predictions, result, base, np))
            for head in HEADS:
                head_states = {name[len(head) + 1:]: value for name, value in base_states.items() if name.startswith(head + '_')}
                view_predictions = {field: predictions[f'{parent}_{head}__{seed}__{field}'] for field in FIELDS}
                compare(times[parent, seed, head]['state_array_sha256'], hashes(head_states), path='independent replay state hashes')
                compare(times[parent, seed, head]['prediction_array_sha256'], hashes(view_predictions), path='independent replay prediction hashes')
                expected_prefix = head_states['prefix_states'][(np.arange(len(base['case_ids'])) + 1) % len(base['case_ids'])]
                require(np.max(np.abs(expected_prefix - head_states['shuffled_prefix_states'])) <= 1e-12,
                        'shuffled prefix follows fixed next-case rotation')
    invariant_fields = ('blind_survival', 'observed_survival', 'observed_probabilities',
                        'blind_states', 'observed_states', 'shuffled_states', 'prefix_states', 'shuffled_prefix_states')
    declared_invariants = [{'parent': parent, 'seed': seed, 'bitwise_equal': dict.fromkeys(invariant_fields, True)}
                           for parent in PARENTS for seed in SEEDS]
    compare(summary['frozen_invariants'], declared_invariants, path='all recorded frozen-output invariants')
    rows, baselines = rows_for({**base, **base_targets}, predictions, uniform, np, check=check)
    key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(summary['rows'], key=key), sorted(rows, key=key), path='all independently reconstructed paired endpoint metrics')
    compare(sorted(summary['baseline_rows'], key=lambda row: row['horizon']), baselines, path='independent uniform-state reference')
    metadata['structural_work_validation'] = validate_work(summary, train, base, np, check=check)
    require(counters['array_decodes'] == 24, 'exact fixed numerical decode roster')
    counts = {'train': len(train['case_ids']), 'base': len(base['case_ids'])}
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'counts': counters, 'metadata': metadata,
            'data_cases': counts, 'target_reconstruction': {'train': train_record, 'base': base_record},
            'prefix_reconstruction': prefix_record, 'exact_oracle_agreement': True,
            'certificates': certificates, 'frozen_invariants': invariants,
            'rows': rows, 'baseline_rows': baselines, 'gates': gates(rows, baselines, counts),
            'paired_comparisons': paired_comparisons(rows), 'structural_work': summary['structural_work'],
            'architecture_claim': False, 'latent_identification_claim': False,
            'limitations': [
                'Original source/process/payload admission is required externally before this saved-output audit.',
                'Scalar objective, gradient and Frank-Wolfe gap use saved TRAIN states and public targets; latent-state extraction is source-qualified rather than independently replayed.',
                'The numerical certificate uses float64 arithmetic, not interval arithmetic or a formal exact-arithmetic proof.',
                'Closed-simplex solutions can attain boundaries unavailable to finite softmax logits; gains cannot isolate optimizer choice.',
                'Original TRAIN is reused; DEV identities are fresh and no earlier DEV arrays are decoded.',
                'All original and solved views retain original model parameters; solved costs are explicit posthoc linear maps, not a new recurrent architecture.',
                'Both views pay original-head computation within model forwards plus separately charged posthoc products.',
                'Byte agreement and barrier metadata do not independently prove historical causal execution or temporal ordering.',
                'A certified TRAIN optimum or unchanged absolute gate does not identify latent state, establish lost information, or demonstrate native transfer.']}
