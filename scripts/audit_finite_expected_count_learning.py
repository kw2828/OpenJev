"""Independent saved-output audit for time-budgeted prefix learning before common joint training.

The caller authenticates original sources, process closure and payloads before
``audit``. Only frozen scalar rational-world arithmetic is reused. No model,
world generator, producer or optimizer is imported or executed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

from audit_finite_observation_learning import compare, multiply, probabilities, require, scalar_world

VERSION = 'finite-expected-count-learning-audit-v1'
NAMESPACE = 429260924
ARMS = ('joint_only', 'gradient_prefix', 'em_prefix')
SEEDS = (429261001, 429261002, 429261003)
HORIZONS = (1, 2, 4, 8)
TARGETS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')
FIELDS = (*TARGETS, 'shuffled_blind_costs')
DATA_KEYS = {'prefix', 'lengths', 'actions', 'observations', 'case_ids', *TARGETS}
CONFIG = {'seed_namespace': NAMESPACE, 'train_attempts': 512, 'dev_attempts': 128,
          'epochs': 480, 'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8, 'pretraining_seconds': 10.}
PREFIX_KEYS = {'prefix', 'lengths', 'case_ids', 'event_mask', 'endpoint_eligible', 'endpoint_rows'}
ROUTES = ('training_blind', 'training_observed', 'evaluation_blind', 'evaluation_observed',
          'evaluation_shuffled', 'training_prefix', 'evaluation_prefix')
CORE_WORK_KEYS = {'prefix_assimilation_calls', 'prefix_assimilation_rows', 'projection_calls', 'projection_rows',
    'prefix_softmax_calls', 'operator_observed_softmax_calls', 'operator_blind_softmax_calls',
    'operator_marginal_sum_calls', 'blind_transition_calls', 'blind_transition_rows',
    'observed_branch_calls', 'observed_branch_rows', 'observed_conditioning_rows',
    'cost_readout_calls', 'cost_readout_rows', 'absorbed_rows'}
FILTER_WORK_KEYS = {'prefix_filter_calls', 'prefix_filter_rows', 'reset_emission_softmax_calls',
                    'reset_emission_rows', 'prefix_operator_softmax_calls'}
HEAD_WORK_KEYS = {'cost_head_softmax_calls', 'cost_head_probability_rows'}
FACTORIZATION_WORK = {'factorization_calls': 1, 'transition_softmax_calls': 1,
    'transition_probability_columns': 32, 'emission_softmax_calls': 1, 'emission_probability_columns': 8,
    'hazard_sigmoid_calls': 1, 'hazard_probability_entries': 32,
    'factor_product_entries': 1536, 'factor_found_reduction_columns': 32}
FACTOR_WORK_KEYS = set(FACTORIZATION_WORK)
SNAPSHOT_WORK_KEYS = CORE_WORK_KEYS | FILTER_WORK_KEYS | HEAD_WORK_KEYS | FACTOR_WORK_KEYS
ENDPOINT_WORK_KEYS = SNAPSHOT_WORK_KEYS | {'privileged_prefix_rows', 'event_probability_rows'}
PREFIX_WORK_KEYS = SNAPSHOT_WORK_KEYS | {'prefix_probability_rows', 'prefix_nll_rows'}

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
    pattern = re.compile(rf'ns{NAMESPACE}-split{split_id}-case([0-9]{{10}})')
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
    require(set(prefix_oracle) == {'prefix_state'}, 'single separately stored oracle input')
    state_array = prefix_oracle['prefix_state']
    require(state_array.shape == (n, 8) and state_array.dtype == np.float64
            and np.isfinite(state_array).all() and np.all(state_array >= 0)
            and np.all(np.abs(state_array.sum(-1) - 1) <= 1e-12), 'normalized float64 privileged prefix state')
    require(set(exact_predictions) == set(TARGETS), 'exact/exact control has five fields and no fit seed')
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
    prefix_error = float(np.max(np.abs(prefix_states - state_array)))
    target_errors = {name: float(np.max(np.abs(targets[name] - data[name]))) for name in TARGETS}
    exact_errors = {name: float(np.max(np.abs(targets[name] - exact_predictions[name]))) for name in TARGETS}
    require(prefix_error <= 1e-12 and all(error <= 1e-12 for error in target_errors.values()),
            'independent public-history posterior and reference targets')
    require(all(error <= 1e-12 for error in exact_errors.values()), 'exact/exact oracle agreement at every saved horizon')
    return {'cases': n, 'horizon': horizon, 'prefix_maximum_absolute_error': prefix_error,
            'target_maximum_absolute_errors': target_errors, 'oracle_maximum_absolute_errors': exact_errors}, targets, uniform_costs


def validate_predictions(data, predictions, np):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in SEEDS for field in FIELDS}
    require(set(predictions) == expected, 'complete nine-model prediction fields')
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
    require(len(rows) == len(keyed) == 36 and set(keyed)
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
    require(split in ('train', 'base') and set(data) == PREFIX_KEYS, 'exact all-attempt prefix fields')
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


def prefix_row(data, prediction, arm, seed, np):
    """Independently score realized events; padding and reset-found are never targets."""
    require(arm in ARMS and seed in SEEDS and set(prediction) == {'probabilities', 'nll'}, 'exact saved prefix prediction fields')
    n = len(data['case_ids'])
    p, nll, mask = prediction['probabilities'], prediction['nll'], data['event_mask']
    require(p.shape == (n, 9, 5) and nll.shape == (n, 9)
            and p.dtype == nll.dtype == np.float64 and np.isfinite(p).all() and np.isfinite(nll).all(),
            'finite float64 prefix probability/NLL arrays')
    require(np.all((p >= 0) & (p <= 1 + 1e-12)) and np.all(nll >= -1e-12)
            and np.all(np.abs(p.sum(-1)[mask] - 1) <= 1e-10)
            and not p[~mask].any() and not nll[~mask].any() and not p[:, 0, 4].any(),
            'normalized active predictions, zero padding and no reset found event')
    labels = data['prefix'][:, :, 4:9].argmax(-1)
    terms = []
    for i, j in zip(*np.nonzero(mask), strict=True):
        assigned = float(p[i, j, labels[i, j]])
        require(assigned > 0, 'observed prefix event has strictly positive prediction; no clipping')
        expected = -math.log(assigned)
        require(math.isclose(float(nll[i, j]), expected, rel_tol=1e-12, abs_tol=1e-12), 'saved prefix NLL agrees with assigned event probability')
        terms.append(expected)
    return {'arm': arm, 'seed': seed, 'attempts': n,
            'valid_events': len(terms), 'mean_nll': math.fsum(terms) / len(terms)}


def validate_work_schema(work):
    require(type(work) is dict and set(work) == set(ARMS), 'all three-arm structural work records')
    for arm in ARMS:
        require(type(work[arm]) is dict and set(work[arm]) == set(ROUTES), 'all seven structural work routes')
        for route in ROUTES:
            expected = PREFIX_WORK_KEYS if route.endswith('_prefix') else ENDPOINT_WORK_KEYS
            value = work[arm][route]
            require(type(value) is dict and set(value) == expected
                    and all(type(v) is int and v >= 0 for v in value.values()), 'exact nonnegative structural work keys')


def endpoint_work(observations, observed, np):
    """Count executed forward geometry from public rows, without a model call."""
    n, horizon = observations.shape
    values = dict.fromkeys(ENDPOINT_WORK_KEYS, 0)
    if not n:
        return values
    values.update({'operator_observed_softmax_calls': 1, 'operator_marginal_sum_calls': 1,
        'blind_transition_calls': horizon, 'blind_transition_rows': horizon * n,
        'cost_readout_calls': horizon, 'cost_readout_rows': horizon * n,
        'prefix_filter_calls': 8, 'prefix_filter_rows': 8 * n,
        'reset_emission_softmax_calls': 1, 'reset_emission_rows': n, 'prefix_operator_softmax_calls': 1})
    if observed:
        ordinary = observations < 4
        already_found = np.concatenate((np.zeros((n, 1), dtype=bool), observations[:, :-1] == 4), axis=1)
        values.update({'event_probability_rows': horizon * n,
            'observed_branch_calls': int(ordinary.any(axis=0).sum()), 'observed_branch_rows': int(ordinary.sum()),
            'observed_conditioning_rows': int(ordinary.sum()), 'absorbed_rows': int(already_found.sum())})
    return values


def prefix_work(prefixes, indices, np):
    lengths = prefixes['lengths'][indices]
    found = prefixes['prefix'][indices, :, 8] == 1
    ordinary = (np.arange(9)[None, :] < lengths[:, None]) & ~found
    ordinary[:, 0] = False
    values = dict.fromkeys(PREFIX_WORK_KEYS, 0)
    values.update({'reset_emission_softmax_calls': 1, 'reset_emission_rows': len(indices),
        'prefix_operator_softmax_calls': 1, 'prefix_probability_rows': int(lengths.sum()),
        'prefix_nll_rows': int(lengths.sum()), 'prefix_filter_calls': int(ordinary.any(axis=0).sum()),
        'prefix_filter_rows': int(ordinary.sum())})
    return values


def validate_work(work, datasets, prefixes, orders, fits, np, *, check=lambda: None):
    """Reconstruct exact forward call/row counts and zero-endpoint batches."""
    validate_work_schema(work)
    expected = {arm: {route: dict.fromkeys(work[arm][route], 0) for route in ROUTES} for arm in ARMS}
    empty_batches = {(arm, seed): 0 for arm in ARMS for seed in SEEDS}

    def add(arm, route, values):
        if arm in ARMS:
            factors = 1 if route.endswith('_prefix') else 2
            values = {**values, **{key: amount * factors for key, amount in FACTORIZATION_WORK.items()},
                      'reset_emission_softmax_calls': 0, 'prefix_operator_softmax_calls': 0,
                      'operator_observed_softmax_calls': 0}
        if not route.endswith('_prefix'):
            values = {**values, 'cost_head_softmax_calls': values['cost_readout_calls'],
                      'cost_head_probability_rows': 8 * values['cost_readout_calls']}
        for key, value in values.items():
            expected[arm][route][key] += value

    for row in orders:
        check()
        arm, seed = row['arm'], row['seed']
        for start in range(0, 512, 64):
            indices = np.asarray(row['indices'][start:start + 64], dtype=np.int64)
            mapped = prefixes['train']['endpoint_rows'][indices]
            selected = mapped[mapped >= 0]
            if len(selected):
                observations = datasets['train']['observations'][selected]
                add(arm, 'training_blind', endpoint_work(observations, False, np))
                add(arm, 'training_observed', endpoint_work(observations, True, np))
            else:
                empty_batches[arm, seed] += 1
            add(arm, 'training_prefix', prefix_work(prefixes['train'], indices, np))
    for arm in ARMS:
        for _seed in SEEDS:
            for start in range(0, len(datasets['base']['case_ids']), 64):
                observations = datasets['base']['observations'][start:start + 64]
                for route in ('evaluation_blind', 'evaluation_observed', 'evaluation_shuffled'):
                    add(arm, route, endpoint_work(observations, route.endswith('_observed'), np))
            for start in range(0, 128, 64):
                add(arm, 'evaluation_prefix', prefix_work(prefixes['base'], np.arange(start, start + 64), np))
    compare(work, expected, path='structural forward work independently derived from public inputs and saved orders')
    for row in fits:
        require(row['zero_endpoint_batches'] == empty_batches[row['arm'], row['seed']], 'zero-endpoint batches reconstructed from complete attempted orders')
    return {'arms': 3, 'routes': 21, 'zero_endpoint_batches': sum(empty_batches.values()),
            'fixed_geometry_checked': True, 'historical_forward_work_independently_replayed': False,
            'scope': 'Forward call/row counts derived from public inputs and saved schedules; no model replay, FLOPs or backward-work measurement.'}


def readout_matrix(record):
    """Validate a JSON snapshot and its little-endian row-major float64 digest."""
    require(type(record) is dict and set(record) == {'matrix', 'sha256'}, 'exact readout snapshot fields')
    matrix = record['matrix']
    require(type(matrix) is list and len(matrix) == 4
            and all(type(row) is list and len(row) == 8 for row in matrix), 'four-by-eight saved readout matrix')
    flat = [value for row in matrix for value in row]
    require(all(type(value) is float and math.isfinite(value) for value in flat), 'finite float64-converted readout entries')
    require(record['sha256'] == hashlib.sha256(struct.pack('<32d', *flat)).hexdigest(), 'canonical readout snapshot byte digest')
    require(all(-.75 - 1e-12 <= value <= .25 + 1e-12 for value in flat)
            and all(abs(math.fsum(matrix[i][j] for i in range(4))) <= 1e-12 for j in range(8)),
            'bounded centered cost columns, allowing float64 boundary roundoff')
    return matrix


def validate_head_metadata(row):
    arm, pm = row['arm'], row['parameter_metadata']
    require(arm in ARMS, 'declared readout arm')
    sizes = {'transition_logits': 256, 'emission_logits': 32, 'hazard_logits': 32, 'cost_logits': 32}
    parameters = {name: {'count': count, 'dtype': 'torch.float64', 'bytes': count * 8} for name, count in sizes.items()}
    require(pm['parameter_count'] == 352
            and pm['parameter_bytes'] == 2816
            and pm['buffer_bytes'] == 0 and pm['parameters'] == parameters,
            'architecture-specific parameter storage and no unused buffers')
    initial = readout_matrix(row['readout_initial'])
    readout_matrix(row['readout_final'])
    scale = .9
    expected = [[scale * (-.75 if i == ((j ^ (j >> 1)) & 3) else .25) for j in range(8)] for i in range(4)]
    require(max(abs(initial[i][j] - expected[i][j]) for i in range(4) for j in range(8)) <= 1e-12,
            'prescribed world-aligned exact or softened initial readout')


DYNAMICS_SHAPES = {'reset_emission': (4, 8), 'observed': (4, 4, 8, 8), 'found': (4, 8), 'blind': (4, 8, 8)}


def tensor_snapshot(record, shape):
    require(type(record) is dict and set(record) == {'matrix', 'sha256'}, 'exact dynamics tensor snapshot fields')

    def flatten(value, dimensions):
        if not dimensions:
            require(type(value) is float and math.isfinite(value), 'finite float64-converted dynamics entry')
            return [value]
        require(type(value) is list and len(value) == dimensions[0], 'exact dynamics tensor shape')
        return [number for part in value for number in flatten(part, dimensions[1:])]

    flat = flatten(record['matrix'], shape)
    require(record['sha256'] == hashlib.sha256(struct.pack('<' + 'd' * len(flat), *flat)).hexdigest(),
            'raw little-endian float64 dynamics digest')
    require(all(0 < value <= 1 + 1e-12 for value in flat), 'positive bounded dynamics snapshot probabilities')
    return record['matrix'], flat


def dynamics_snapshot(record, arm):
    require(type(record) is dict and set(record) == {*DYNAMICS_SHAPES, 'work'}, 'complete dynamics snapshot')
    matrices = {name: tensor_snapshot(record[name], shape)[0] for name, shape in DYNAMICS_SHAPES.items()}
    emission, observed, found, blind = (matrices[name] for name in ('reset_emission', 'observed', 'found', 'blind'))
    reset_error = max(abs(math.fsum(emission[o][s] for o in range(4)) - 1) for s in range(8))
    marginal_error = max(abs(math.fsum(observed[a][o][n][s] for o in range(4)) - blind[a][n][s])
                         for a in range(4) for n in range(8) for s in range(8))
    mass_error = max(abs(math.fsum(blind[a][n][s] for n in range(8)) + found[a][s] - 1)
                     for a in range(4) for s in range(8))
    require(max(reset_error, marginal_error, mass_error) <= 1e-12, 'normalized reset, branch marginal and total outgoing mass')
    factor_error = None
    if arm in ARMS:
        factor_error = max(abs(observed[a][o][n][s] - emission[o][n] * blind[a][n][s])
                           for a in range(4) for o in range(4) for n in range(8) for s in range(8))
        require(factor_error <= 1e-12, 'reset emission equals conditional branch emission in the factorized arm')
    work = record['work']
    expected = dict.fromkeys(SNAPSHOT_WORK_KEYS, 0)
    expected['operator_marginal_sum_calls'] = 1
    if arm in ARMS:
        expected.update(FACTORIZATION_WORK)
    else:
        expected.update({'reset_emission_softmax_calls': 1, 'prefix_operator_softmax_calls': 1})
    require(type(work) is dict and all(type(value) is int and value >= 0 for value in work.values()),
            'nonnegative integer diagnostic work')
    compare(work, expected, path='one separately charged dynamics snapshot')
    return {'reset_normalization_error': reset_error, 'blind_marginal_error': marginal_error,
            'outgoing_mass_error': mass_error, 'shared_emission_factorization_error': factor_error}


def validate_dynamics_metadata(row):
    arm = row['arm']
    for name in ('dynamics_initial', 'dynamics_final'):
        dynamics_snapshot(row[name], arm)
    work = row['construction_work']
    expected = {**dict.fromkeys(FACTOR_WORK_KEYS, 0), 'matching_log_calls': 0, 'matching_log_entries': 0}
    require(type(work) is dict and all(type(value) is int and value >= 0 for value in work.values()),
            'nonnegative integer constructor work')
    compare(work, expected, path='separately charged constructor matching algebra')
    require(type(row['construction_seconds']) in (int, float) and math.isfinite(row['construction_seconds'])
            and row['construction_seconds'] >= 0, 'finite separately measured construction time')


def initial_function_checks(fits, pretraining):
    """Compare common starts before pretraining, not deliberately changed dynamics."""
    keyed = {(row['arm'], row['seed']): row for row in fits}
    stages = {(row['arm'], row['seed']): row for row in pretraining}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    require(len(fits) == len(keyed) == len(pretraining) == len(stages) == 9
            and set(keyed) == set(stages) == expected, 'complete common initial-state roster')
    results = []
    for seed in SEEDS:
        initial = stages[ARMS[0], seed]['initial_state_sha256']
        require(all(stages[arm, seed]['initial_state_sha256'] == initial for arm in ARMS),
                'all three arms have the identical full state before prefix pretraining')
        require(all(keyed[arm, seed]['initial_state_sha256'] == stages[arm, seed]['final_state_sha256'] for arm in ARMS),
                'joint training begins at each retained pretraining state')
        head = keyed[ARMS[0], seed]['readout_initial']['matrix']
        error = max(abs(head[i][j] - keyed[arm, seed]['readout_initial']['matrix'][i][j])
                    for arm in ARMS for i in range(4) for j in range(8))
        require(error <= 1e-12, 'all cost heads remain identical at joint-training start')
        results.append({'seed': seed, 'initial_state_sha256': initial,
                        'head_maximum_absolute_difference': error, 'passed': True})
    return results

def dynamics_comparisons(rows, fits, pretraining):
    """Descriptive pretraining-method contrasts; no relative result rescues a gate."""
    metrics = ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl',
               'blind_survival_mae', 'observed_survival_mae')
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    require(len(rows) == len(keyed) == 36
            and set(keyed) == {(arm, seed, h) for arm in ARMS for seed in SEEDS for h in HORIZONS},
            'complete paired dynamics contrast roster')
    contrasts = []
    for control in ('joint_only', 'gradient_prefix'):
        for seed in SEEDS:
            for horizon in (4, 8):
                learned, baseline = keyed['em_prefix', seed, horizon], keyed[control, seed, horizon]
                contrasts.append({'control': control, 'seed': seed, 'horizon': horizon,
                    'metrics': {name: {'em_prefix': learned[name], 'control': baseline[name],
                        'difference': learned[name] - baseline[name],
                        'relative_reduction': (baseline[name] - learned[name]) / baseline[name] if baseline[name] > 0 else None}
                        for name in metrics}})
    means = [{'arm': arm, 'horizon': h, 'fit_seeds': list(SEEDS),
              **{name: math.fsum(keyed[arm, seed, h][name] for seed in SEEDS) / len(SEEDS) for name in metrics}}
             for arm in ARMS for h in HORIZONS]
    fit_records = []
    for row in fits:
        initial, final = row['readout_initial']['matrix'], row['readout_final']['matrix']
        changes = [abs(final[i][j] - initial[i][j]) for i in range(4) for j in range(8)]
        fit_records.append({'arm': row['arm'], 'seed': row['seed'], 'initial': row['readout_initial'],
            'final': row['readout_final'], 'mean_absolute_change': math.fsum(changes) / 32,
            'maximum_absolute_change': max(changes), 'fit_seconds': row['seconds'],
            'construction_seconds': row['construction_seconds'], 'construction_work': row['construction_work'],
            'dynamics_snapshot_work': {name: row[name]['work'] for name in ('dynamics_initial', 'dynamics_final')},
            'parameter_count': row['parameter_metadata']['parameter_count'],
            'parameter_bytes': row['parameter_metadata']['parameter_bytes']})
    return {'paired': contrasts, 'means': means, 'fits': fit_records,
            'initial_function_checks': initial_function_checks(fits, pretraining),
            'interpretation': 'All arms use the same factorized architecture and initial state. Pretraining method/time and accepted updates differ; the common joint-training schedule is unchanged. Relative gains do not rescue failed absolute criteria.'}


def validate_metadata(folder, summary, *, check=lambda: None):
    """Join opaque files and execution attestations without replaying training."""
    def read(name, *, lines=False):
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular saved prefix-learning metadata')
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())

    require(summary['version'] == 'finite-expected-count-learning-v1', 'expected-count learning producer version')
    compare(summary['config'], CONFIG, path='registered prefix-pretraining configuration')
    fits = summary['fits']
    expected_pairs = [(arm, seed) for index, seed in enumerate(SEEDS) for arm in ARMS[index:] + ARMS[:index]]
    require([(row['arm'], row['seed']) for row in fits] == expected_pairs, 'nine completed common joint fits')
    expected_files = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
                      'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl',
                      'training-orders.jsonl', 'training-epochs.jsonl', 'checkpoint-barrier.json', 'prediction-times.jsonl',
                      'oracle-train-check.json', 'oracle-base-check.json'}
    expected_files |= {f'{arm}-{seed}.npz' for arm, seed in expected_pairs}
    expected_files |= {'train-prefix.npz', 'base-prefix.npz'}
    expected_files |= {f'prefix-{arm}-{seed}-base.npz' for arm, seed in expected_pairs}
    expected_files |= {'pretraining.jsonl'} | {f'pretraining-{stage}-{arm}-{seed}.npz'
        for stage in ('initial', 'final') for arm, seed in expected_pairs}
    require(set(summary['files']) == expected_files, 'complete expected-count learning producer payload roster')
    for name, descriptor in summary['files'].items():
        check()
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular original prefix-learning payload')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        compare(descriptor, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, path='payload ' + name)
    compare(read('config.json'), CONFIG, path='saved configuration')
    compare(read('fits.jsonl', lines=True), fits, path='durable fits')
    require(set(summary['dataset_counts']) == {'train', 'base'}, 'two exact dataset count records')
    retained = {}
    for split_id, split in enumerate(('train', 'base')):
        row = summary['dataset_counts'][split]
        attempts, horizon = (512, 2) if split == 'train' else (128, 8)
        require(row['version'] == 'finite-prefix-learning-data-v1' and row['epsilon'] == .12
                and row['seed_namespace'] == NAMESPACE and row['split_id'] == split_id
                and row['attempted'] == attempts and row['horizon'] == horizon, 'registered generation geometry')
        n, excluded, found = row['retained'], row['discarded_found'], row['prefix_found_by_step']
        require(type(n) is int and 0 < n <= attempts and type(excluded) is int and excluded >= 0
                and n + excluded == attempts and type(found) is list and len(found) == 8
                and all(type(v) is int and v >= 0 for v in found) and sum(found) == excluded,
                'complete fixed prefix allocation')
        require(row['initial_odor_draws'] == attempts and row['prefix_action_draws'] == 8 * attempts
                and row['prefix_event_draws'] == 8 * n + sum((i + 1) * v for i, v in enumerate(found))
                and row['forecast_action_draws'] == horizon * n
                and row['valid_prefix_events'] == attempts + row['prefix_event_draws'], 'fixed draw accounting')
        retained[split] = n
        errors = summary['oracle_checks'][split]
        require(set(errors) == set(TARGETS) and all(type(v) in (int, float) and math.isfinite(v)
                and 0 <= v <= 1e-12 for v in errors.values()), 'every exact-control validation passed')
        compare(read('oracle-' + split + '-check.json'), errors, path='durable exact-control checks')
    require(summary['oracle_metadata']['parameter_count'] == 0
            and summary['oracle_metadata']['parameter_bytes'] == 0
            and summary['oracle_metadata']['parameters'] == {}, 'exact control has no trainable parameters')
    require(type(summary['oracle_state_sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', summary['oracle_state_sha256']), 'exact control state digest')
    n, dev_n = retained['train'], retained['base']
    updates = 480 * 8
    events = summary['dataset_counts']['train']['valid_prefix_events']
    fit_map = {(row['arm'], row['seed']): row for row in fits}
    for row in fits:
        arm = row['arm']
        require(row['epochs'] == 480 and row['updates'] == updates and row['training_cases'] == n
                and row['training_case_exposures'] == 480 * n, 'complete fixed fit schedule')
        weight = 1
        require(row['training_attempts'] == 512 and row['training_attempt_exposures'] == 480 * 512
                and row['valid_prefix_events'] == events and row['prefix_loss_weight'] == weight
                and row['training_prefix_event_exposures'] == weight * 480 * events
                and type(row['zero_endpoint_batches']) is int and 0 <= row['zero_endpoint_batches'] <= updates,
                'fixed attempt, endpoint and valid-event denominators and exposures')
        validate_head_metadata(row)
        validate_dynamics_metadata(row)
        require(type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'finite completed fit duration')
        for key in ('initial_state_sha256', 'final_state_sha256', 'case_order_sha256', 'fixed_buffers_sha256'):
            require(type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key]), 'fit state/work digest')
        for key, present in (('prefix_initial_sha256', False),
                             ('operator_initial_sha256', False),
                             ('reset_initial_sha256', False),
                             ('prefix_operator_initial_sha256', False)):
            require((type(row[key]) is str and re.fullmatch('[0-9a-f]{64}', row[key])) if present else row[key] is None,
                    'present and absent parameter-group digests')
        require(row['oracle_prefix_input'] is False, 'no privileged prefix input in any learned fit')
        name = f"{arm}-{row['seed']}.npz"
        compare(row['checkpoint'], {'path': name, **summary['files'][name]}, path='final checkpoint')
    pretraining = validate_pretraining_metadata(summary, read)
    function_checks = initial_function_checks(fits, summary['pretraining'])
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': 9, 'oracle_train_verified': True}
    compare(summary['checkpoint_barrier'], barrier, path='summary barrier')
    compare(read('checkpoint-barrier.json'), barrier, path='durable barrier')
    orders, epochs = read('training-orders.jsonl', lines=True), read('training-epochs.jsonl', lines=True)
    expected_epochs = [(arm, seed, epoch) for arm, seed in expected_pairs for epoch in range(480)]
    for records in (orders, epochs):
        require([(r['arm'], r['seed'], r['epoch']) for r in records] == expected_epochs, 'complete 4320-epoch journal roster')
    order_hashes = {pair: hashlib.sha256() for pair in expected_pairs}
    paired_orders = {}
    for row in orders:
        indices = row['indices']
        require(set(row) == {'arm', 'seed', 'epoch', 'indices', 'batch_size'} and row['batch_size'] == 64
                and type(indices) is list and len(indices) == 512 and all(type(i) is int for i in indices)
                and sorted(indices) == list(range(512)), 'complete once-per-epoch case permutation')
        key = row['seed'], row['epoch']
        if key in paired_orders:
            require(indices == paired_orders[key], 'same-seed paired case order')
        paired_orders[key] = indices
        order_hashes[row['arm'], row['seed']].update(struct.pack('<' + 'q' * 512, *indices))
    for pair, digest in order_hashes.items():
        require(fit_map[pair]['case_order_sha256'] == digest.hexdigest(), 'saved case-order digest')
    for row in epochs:
        require(row['cases'] == 512 and row['updates'] == 8
                and all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                        for k in ('mean_objective', 'seconds')), 'finite completed epoch work')
    times = summary['prediction_times']
    require([(row['arm'], row['seed']) for row in times] == [(arm, seed) for arm in ARMS for seed in SEEDS],
            'nine complete evaluation views')
    compare(read('prediction-times.jsonl', lines=True), times, path='durable evaluation records')
    for row in times:
        require(row['regime'] == 'base' and row['cases'] == dev_n and row['batch_size'] == 64
                and row['oracle_prefix_input'] is False and row['shuffle_offset'] == 1
                and row['model_state_before'] == row['model_state_after'] == fit_map[row['arm'], row['seed']]['final_state_sha256']
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'unchanged final state and declared matched input routing')
    prefix_times = summary['prefix_prediction_times']
    require([(row['arm'], row['seed']) for row in prefix_times] == [(arm, seed) for arm in ARMS for seed in SEEDS],
            'nine complete prefix diagnostic views')
    for row in prefix_times:
        require(row['attempts'] == 128 and row['valid_events'] == summary['dataset_counts']['base']['valid_prefix_events']
                and type(row['seconds']) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'finite all-attempt prefix inference costs')
    zero = sum(row['zero_endpoint_batches'] for row in fits)
    eval_batches = 9 * math.ceil(dev_n / 64)
    oracle_batches = math.ceil(n / 64) + math.ceil(dev_n / 64)
    expected_counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': 9,
        'oracle_model_constructions': 1, 'fit_count': 9, 'checkpoint_writes': 9,
        'training_blind_rollouts': 9 * updates - zero, 'training_observed_rollouts': 9 * updates - zero,
        'optimizer_attempts': 9 * updates, 'optimizer_steps': 9 * updates, 'training_case_exposures': 9 * 480 * n,
        'training_attempt_exposures': 9 * 480 * 512, 'training_prefix_rollouts': 9 * updates,
        'training_prefix_event_exposures': 9 * 480 * events, 'zero_endpoint_batches': zero,
        'evaluation_blind_rollouts': eval_batches, 'evaluation_observed_rollouts': eval_batches,
        'evaluation_shuffled_rollouts': eval_batches, 'evaluation_case_views': 9 * dev_n,
        'evaluation_prefix_rollouts': 9 * 2,
        'evaluation_prefix_event_views': 9 * summary['dataset_counts']['base']['valid_prefix_events'],
        'oracle_blind_rollouts': oracle_batches, 'oracle_observed_rollouts': oracle_batches,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
        'readout_snapshot_evaluations': 18, 'readout_snapshot_softmax_evaluations': 18,
        'dynamics_snapshot_evaluations': 18, 'pretraining_checkpoint_writes': 18, 'pretraining_runs': 9}
    compare(summary['counts'], expected_counts, path='complete declared executed counters')
    validate_work_schema(summary['structural_work'])
    return {'fits': 9, 'checkpoint_files_hashed': 27, 'payloads': len(expected_files), 'pretraining': pretraining, 'epochs': 4320, 'evaluation_views': 9,
            'prefix_evaluation_views': 9, 'retained': retained, 'optimizer_steps_attested': 9 * updates,
            'training_case_exposures_attested': 9 * 480 * n, 'training_attempt_exposures_attested': 9 * 480 * 512,
            'initial_function_checks': function_checks,
            'historical_ordering_independently_replayed': False}


def audit(folder, *, check=lambda: None):
    """Read only an independently admitted and closed producer's saved outputs."""
    import numpy as np

    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'absolute original saved-output directory')
    summary_path = folder / 'summary.json'
    require(summary_path.is_file() and not summary_path.is_symlink(), 'regular producer summary')
    summary = json.loads(summary_path.read_text())
    metadata = validate_metadata(folder, summary, check=check)
    counters = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'optimizer_calls': 0,
                'world_or_generator_calls': 0, 'native_calls': 0}

    def load(name, *, checkpoint=False):
        check()
        with np.load(folder / name, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), 'unique saved array members')
            result = {key: archive[key] for key in archive.files}
        counters['array_decodes'] += 1
        counters['checkpoint_decodes'] += int(checkpoint)
        return result

    boundary_arrays = {(stage, row['arm'], row['seed']): load(row[stage + '_checkpoint']['path'], checkpoint=True)
        for row in summary['pretraining'] for stage in ('initial', 'final')}
    boundary_checks = verify_boundaries(summary['pretraining'], boundary_arrays, np)
    records, datasets, prefixes, targets, uniform, prefix_records = {}, {}, {}, {}, {}, {}
    for split in ('train', 'base'):
        data, prefix = load(split + '.npz'), load(split + '-prefix.npz')
        oracle, exact = load(split + '-oracle.npz'), load('oracle-' + split + '.npz')
        record, rebuilt, reference = reconstruct(data, oracle, exact, split, np, check=check)
        prefix_record, states, _reference = reconstruct_prefix(prefix, data, split, np, check=check)
        require(float(np.max(np.abs(states - oracle['prefix_state']))) <= 1e-12, 'all-attempt survivor posterior agrees with separate oracle')
        declared = summary['dataset_counts'][split]
        for key, reported_key in (('attempts', 'attempted'), ('retained', 'retained'),
                ('discarded_found', 'discarded_found'), ('valid_events', 'valid_prefix_events'),
                ('prefix_found_by_step', 'prefix_found_by_step')):
            compare(prefix_record[key], declared[reported_key], path='public-prefix count: ' + split + '/' + key)
        producer_errors = {name: float(np.max(np.abs(exact[name] - data[name]))) for name in TARGETS}
        compare(summary['oracle_checks'][split], producer_errors, path='independent exact-control error receipt')
        found = data['observations'] == 4
        found_cases = found.any(axis=1)
        event_draws = np.where(found_cases, found.argmax(axis=1) + 1, data['observations'].shape[1])
        require(declared['forecast_found_cases'] == int(found_cases.sum())
                and declared['forecast_event_draws'] == int(event_draws.sum()), 'forecast draw counts reconstructed from labels')
        records[split], datasets[split], prefixes[split] = record, data, prefix
        targets[split], uniform[split], prefix_records[split] = rebuilt, reference, prefix_record
    require(not set(prefixes['train']['case_ids']) & set(prefixes['base']['case_ids']), 'disjoint attempted TRAIN/DEV identities')
    predictions = load('predictions-base.npz')
    rows, baselines = rows_for({**datasets['base'], **targets['base']}, predictions, uniform['base'], np, check=check)
    row_key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(summary['rows'], key=row_key), sorted(rows, key=row_key), path='independent endpoint metrics')
    compare(sorted(summary['baseline_rows'], key=lambda row: row['horizon']), baselines, path='independent uniform-state reference')
    prefix_rows = [prefix_row(prefixes['base'], load(f'prefix-{arm}-{seed}-base.npz'), arm, seed, np)
                   for arm in ARMS for seed in SEEDS]
    compare(summary['prefix_rows'], prefix_rows, path='independent all-attempt event-weighted prefix NLL')
    counts = {split: len(data['case_ids']) for split, data in datasets.items()}
    compare(counts, metadata['retained'], path='endpoint support matches metadata')
    orders = [json.loads(line) for line in (folder / 'training-orders.jsonl').read_text().splitlines()]
    metadata['structural_work_validation'] = validate_work(summary['structural_work'], datasets, prefixes,
                                                           orders, summary['fits'], np, check=check)
    require(counters['array_decodes'] == 36 and counters['checkpoint_decodes'] == 18, 'exact saved-boundary and outcome decode roster')
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'counts': counters, 'metadata': metadata,
            'data_cases': counts, 'target_reconstruction': records, 'exact_oracle_agreement': True,
            'prefix_reconstruction': prefix_records, 'prefix_rows': prefix_rows,
            'rows': rows, 'baseline_rows': baselines, 'gates': gates(rows, baselines, counts),
            'dynamics_comparisons': dynamics_comparisons(rows, summary['fits'], summary['pretraining']),
            'pretraining': {'stages': metadata['pretraining'], 'boundary_checks': boundary_checks,
                            'budget_seconds': CONFIG['pretraining_seconds'], 'compute_matched': False,
                            'historical_updates_independently_replayed': False},
            'structural_work': summary['structural_work'], 'architecture_claim': False, 'latent_identification_claim': False,
            'limitations': [
                'Original process, source and payload admission belongs to the external caller.',
                'The exact control receives a privileged posterior and known operators; all three learned arms receive public inputs only.',
                'All arms share architecture, initial weights and common joint-training updates; timed prefix pretraining adds separately charged work and accepted updates.',
                'Prefix NLL includes every attempted case and its first found event, excludes padding and cannot rescue endpoint criteria.',
                'Likelihood arithmetic and public-label/work joins are independently checked; saved predictions do not independently prove causal model execution.',
                'Checkpoint and journal hashes do not replay training or establish historical ordering; fixed loss scaling remains source-qualified.',
                'Forward call and row counts exclude backward operations and are not a matched-compute or FLOP claim.',
                'The known initial cost basis privileges four decision signatures; learned readout gains do not identify the true eight-state posterior.',
                'Observed filtering has intervening observations and remains separate from blind extrapolation.',
                'All 18 pretraining boundary checkpoints are decoded only for parameter shapes, hashes and unchanged-head/retained-state joins. Joint-final checkpoints remain opaque; updates and likelihood traces are not replayed.',
                'The same 10-second eligibility window does not equalize total wall time, accepted updates or FLOPs; late work, rollback and outside-budget summaries are charged separately.',
                'Factorized hidden-state filtering is established; this favorable synthetic diagnostic cannot establish latent identification, native transfer or novelty.']}

PRETRAIN_METHODS = {'joint_only': 'none', 'gradient_prefix': 'gradient', 'em_prefix': 'em'}
TIMED_WORK_KEYS = {'full_gradient_passes', 'torch_diagnostic_passes', 'numpy_expectation_passes',
    'map_updates', 'adam_updates', 'probability_imports', 'probability_checks',
    'gradient_event_exposures', 'expectation_event_exposures', 'parameter_snapshot_tensors',
    'parameter_snapshot_bytes', 'parameter_restore_tensors', 'parameter_restore_bytes',
    'parameter_hash_calls', 'check_calls', 'clock_calls'}
TRACE_KEYS = {'update', 'start_elapsed', 'completed_elapsed', 'seconds', 'accepted', 'rolled_back',
    'penalized_log_likelihood_before', 'penalized_log_likelihood_after', 'roundtrip_max_abs',
    'dynamics_before_sha256', 'dynamics_attempted_sha256', 'dynamics_retained_sha256', 'work_delta'}
TIMED_KEYS = {'version', 'method', 'status', 'termination', 'budget_seconds', 'max_updates',
    'pseudocount', 'learning_rate', 'gradient_clip', 'accepted_updates', 'attempted_updates', 'trace',
    'valid_events', 'attempts', 'initial_penalized_log_likelihood', 'final_penalized_log_likelihood',
    'initial_dynamics_sha256', 'final_dynamics_sha256', 'initial_head_sha256', 'final_head_sha256',
    'head_unchanged', 'work', 'numpy_work', 'timed_seconds', 'overrun_seconds', 'final_summary_work',
    'compute_matched', 'hidden_state_input', 'optimizer_state_reused_across_steps',
    'late_optimizer_state_discarded', 'final_summary_seconds'}
DYNAMIC_NAMES = ('transition_logits', 'emission_logits', 'hazard_logits')
PARAMETER_SHAPES = {'transition_logits': (4, 8, 8), 'emission_logits': (4, 8),
                    'hazard_logits': (4, 8), 'cost_logits': (4, 8)}


def digest_value(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite_scalar(value):
    return type(value) in (int, float) and math.isfinite(value)


def expected_numpy_work(attempts, events, found):
    actions = events - attempts
    return {'sequences': attempts, 'reset_events': attempts, 'action_events': actions,
        'ordinary_events': actions - found, 'found_events': found,
        'forward_matvecs': actions, 'backward_matvecs': actions,
        'transition_posterior_matrices': actions, 'state_posterior_rows': events,
        'check_calls': 4 * actions + 3 * attempts}


def validate_timed_result(result, method, events, attempts, found):
    """Validate deadline/work/hash attestations, without replaying an update."""
    require(type(result) is dict and set(result) == TIMED_KEYS, 'exact timed pretraining result')
    require(result['version'] == 'finite-timed-prefix-v1' and result['method'] == method
            and result['status'] == 'PASS' and result['termination'] in ('budget_reached', 'late_update_rolled_back')
            and result['budget_seconds'] == CONFIG['pretraining_seconds'] and result['max_updates'] == 100000
            and result['pseudocount'] == .001 and result['learning_rate'] == .003 and result['gradient_clip'] == 5.,
            'registered timed method and update eligibility')
    require(result['attempts'] == attempts and result['valid_events'] == events
            and result['head_unchanged'] is True and result['compute_matched'] is False
            and result['hidden_state_input'] is False
            and result['optimizer_state_reused_across_steps'] is (method == 'gradient'), 'public-only prefix objective and honest compute scope')
    for key in ('initial_dynamics_sha256', 'final_dynamics_sha256', 'initial_head_sha256', 'final_head_sha256'):
        require(digest_value(result[key]), 'valid timed boundary state digest')
    require(result['initial_head_sha256'] == result['final_head_sha256'], 'unchanged cost head in timed stage')
    for key in ('timed_seconds', 'overrun_seconds', 'final_summary_seconds'):
        require(finite_scalar(result[key]) and result[key] >= 0, 'finite nonnegative timed-stage duration')
    require(finite_scalar(result['initial_penalized_log_likelihood']) and finite_scalar(result['final_penalized_log_likelihood']),
            'finite timed objective summaries')
    n, accepted, trace = result['attempted_updates'], result['accepted_updates'], result['trace']
    require(type(n) is int and type(accepted) is int and 1 <= accepted <= n <= 100000
            and type(trace) is list and len(trace) == n, 'nonempty completed pretraining within fixed update cap')
    numpy_one = expected_numpy_work(attempts, events, found)
    expected_total = dict.fromkeys(TIMED_WORK_KEYS, 0)
    expected_total.update(torch_diagnostic_passes=1, parameter_hash_calls=2)
    retained_hash = result['initial_dynamics_sha256']
    retained_likelihood = result['initial_penalized_log_likelihood']
    previous_completion, accepted_count, late_count = 0., 0, 0
    for i, row in enumerate(trace):
        require(type(row) is dict and set(row) == TRACE_KEYS and row['update'] == i + 1, 'complete ordered timed-update trace')
        require(all(finite_scalar(row[key]) for key in ('start_elapsed', 'completed_elapsed', 'seconds',
                    'penalized_log_likelihood_before', 'penalized_log_likelihood_after', 'roundtrip_max_abs')),
                'finite attempted-update values')
        began, completed = row['start_elapsed'], row['completed_elapsed']
        require(previous_completion <= began < CONFIG['pretraining_seconds'] and began <= completed,
                'updates begin before deadline with monotonic completion times')
        compare(row['seconds'], completed - began, path='attempt duration includes complete update')
        eligible = completed <= CONFIG['pretraining_seconds']
        require(row['accepted'] is eligible and row['rolled_back'] is (not eligible), 'completion time alone determines acceptance')
        require(eligible or i == n - 1, 'at most one late attempt and no continuation after rollback')
        require(all(digest_value(row[key]) for key in ('dynamics_before_sha256', 'dynamics_attempted_sha256', 'dynamics_retained_sha256'))
                and row['dynamics_before_sha256'] == retained_hash,
                'every attempt starts from the previous retained dynamic state')
        compare(row['penalized_log_likelihood_before'], retained_likelihood, path='retained objective to next attempt')
        require(row['dynamics_retained_sha256'] == (row['dynamics_attempted_sha256'] if eligible else retained_hash),
                'late update restores exact prior dynamic digest')
        require(0 <= row['roundtrip_max_abs'] <= 1e-12 and (method == 'em' or row['roundtrip_max_abs'] == 0),
                'only EM imports probabilities without clipping')
        if method == 'em':
            require(row['penalized_log_likelihood_after'] >= row['penalized_log_likelihood_before'] - 1e-9,
                    'penalized EM likelihood monotonicity within fixed roundoff')
        delta = dict.fromkeys(TIMED_WORK_KEYS, 0)
        delta.update(parameter_snapshot_tensors=3, parameter_snapshot_bytes=2560,
                     parameter_hash_calls=2 + int(not eligible), clock_calls=1,
                     parameter_restore_tensors=3 * int(not eligible), parameter_restore_bytes=2560 * int(not eligible))
        if method == 'em':
            delta.update(numpy_expectation_passes=2, map_updates=1, probability_imports=1,
                         probability_checks=3, expectation_event_exposures=2 * events,
                         check_calls=2 * numpy_one['check_calls'] + 3)
        else:
            delta.update(full_gradient_passes=1, torch_diagnostic_passes=1, adam_updates=1,
                         probability_checks=1, gradient_event_exposures=events, check_calls=1)
        require(type(row['work_delta']) is dict and all(type(v) is int and v >= 0 for v in row['work_delta'].values()),
                'integer executed attempted work including discarded work')
        compare(row['work_delta'], delta, path='one complete attempted prefix update including rollback')
        for key, value in delta.items():
            expected_total[key] += value
        accepted_count += int(eligible)
        late_count += int(not eligible)
        retained_hash = row['dynamics_retained_sha256']
        if eligible:
            retained_likelihood = row['penalized_log_likelihood_after']
        previous_completion = completed
    require(accepted == accepted_count and late_count <= 1 and result['final_dynamics_sha256'] == retained_hash,
            'final retained update/hash accounting')
    compare(result['final_penalized_log_likelihood'], retained_likelihood, path='final diagnostic on retained state')
    extra_loop = int(not late_count and previous_completion < CONFIG['pretraining_seconds'])
    expected_total['clock_calls'] += 2 + n + extra_loop
    expected_total['check_calls'] += 2 + n + extra_loop
    expected_termination = 'late_update_rolled_back' if late_count else 'budget_reached'
    require(result['termination'] == expected_termination and result['late_optimizer_state_discarded'] is (method == 'gradient' and bool(late_count)),
            'registered stop and discarded optimizer-state scope')
    require(result['timed_seconds'] >= previous_completion
            and (not extra_loop or result['timed_seconds'] >= CONFIG['pretraining_seconds']), 'all attempted and rollback time charged')
    compare(result['overrun_seconds'], max(0., result['timed_seconds'] - CONFIG['pretraining_seconds']), path='explicit charged eligibility overrun')
    require(type(result['work']) is dict and all(type(v) is int and v >= 0 for v in result['work'].values()), 'integer timed work')
    compare(result['work'], expected_total, path='all timed setup, attempt, boundary and rollback work')
    expected_numpy = {key: 2 * n * value for key, value in numpy_one.items()} if method == 'em' else {}
    compare(result['numpy_work'], expected_numpy, path='complete expected-count sweeps including discarded update')
    compare(result['final_summary_work'], {'torch_diagnostic_passes': 1, 'parameter_hash_calls': 2, 'check_calls': 1, 'clock_calls': 1},
            path='separately charged final diagnostic work')
    return {'method': method, 'accepted_updates': accepted, 'attempted_updates': n, 'late_updates': late_count,
            'timed_seconds': result['timed_seconds'], 'overrun_seconds': result['overrun_seconds'],
            'final_summary_seconds': result['final_summary_seconds'], 'historical_updates_independently_replayed': False}


def validate_pretraining_metadata(summary, read):
    records = summary['pretraining']
    expected = [(arm, seed) for index, seed in enumerate(SEEDS) for arm in ARMS[index:] + ARMS[:index]]
    require([(row['arm'], row['seed']) for row in records] == expected, 'all nine pretraining stages in fit order')
    compare(read('pretraining.jsonl', lines=True), records, path='durable prefix stage records')
    data = summary['dataset_counts']['train']
    checked = []
    for row in records:
        arm, seed = row['arm'], row['seed']
        method = PRETRAIN_METHODS[arm]
        require(row['method'] == method and row['head_unchanged'] is True and finite_scalar(row['seconds']) and row['seconds'] >= 0
                and digest_value(row['initial_state_sha256']) and digest_value(row['final_state_sha256']), 'valid prefix-stage identity and boundary witnesses')
        for stage in ('initial', 'final'):
            name = f'pretraining-{stage}-{arm}-{seed}.npz'
            compare(row[stage + '_checkpoint'], {'path': name, **summary['files'][name]}, path='pretraining boundary checkpoint')
        result = row['result']
        if method == 'none':
            compare(result, {'status': 'SKIPPED_CONTROL', 'termination': 'no_pretraining', 'attempted_updates': 0,
                'accepted_updates': 0, 'trace': [], 'timed_seconds': 0., 'overrun_seconds': 0.,
                'final_summary_seconds': 0., 'head_unchanged': True}, path='joint-only stage does no pretraining')
            require(row['initial_state_sha256'] == row['final_state_sha256'], 'joint-only boundary state unchanged')
            checked.append({'arm': arm, 'seed': seed, 'method': method, 'accepted_updates': 0, 'attempted_updates': 0})
        else:
            value = validate_timed_result(result, method, data['valid_prefix_events'], 512, data['discarded_found'])
            require(row['seconds'] + 1e-8 >= result['timed_seconds'] + result['final_summary_seconds'], 'wrapper charges eligibility and outside-budget summary time')
            checked.append({'arm': arm, 'seed': seed, **value})
    return checked


def boundary_hashes(arrays, np):
    require(set(arrays) == set(PARAMETER_SHAPES), 'exact four-parameter boundary checkpoint')
    for name, shape in PARAMETER_SHAPES.items():
        require(arrays[name].shape == shape and arrays[name].dtype == np.float64 and np.isfinite(arrays[name]).all(),
                'finite float64 boundary parameter: ' + name)
    full = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        header = json.dumps([name, value.dtype.str, list(value.shape)], separators=(',', ':')).encode()
        full.update(len(header).to_bytes(8, 'little'))
        full.update(header)
        full.update(value.tobytes(order='C'))
    dynamic = hashlib.sha256()
    for name in DYNAMIC_NAMES:
        dynamic.update(name.encode() + b'\0')
        dynamic.update(arrays[name].astype('<f8', copy=False).tobytes())
    head = hashlib.sha256(arrays['cost_logits'].astype('<f8', copy=False).tobytes()).hexdigest()
    return {'full': full.hexdigest(), 'dynamics': dynamic.hexdigest(), 'head': head}


def verify_boundaries(records, checkpoints, np):
    results, initial_by_seed = [], {}
    for row in records:
        arm, seed = row['arm'], row['seed']
        initial, final = (checkpoints[stage, arm, seed] for stage in ('initial', 'final'))
        first, last = boundary_hashes(initial, np), boundary_hashes(final, np)
        require(first['full'] == row['initial_state_sha256'] and last['full'] == row['final_state_sha256'], 'decoded full checkpoint state hashes')
        require(initial['cost_logits'].tobytes() == final['cost_logits'].tobytes(), 'decoded unchanged cost-head bytes')
        if seed in initial_by_seed:
            require(all(initial[name].tobytes() == initial_by_seed[seed][name].tobytes() for name in PARAMETER_SHAPES),
                    'all arms share actual initial parameter bytes')
        initial_by_seed[seed] = initial
        if row['method'] == 'none':
            require(all(initial[name].tobytes() == final[name].tobytes() for name in PARAMETER_SHAPES), 'decoded joint-only state unchanged')
        else:
            record = row['result']
            require(first['dynamics'] == record['initial_dynamics_sha256'] and last['dynamics'] == record['final_dynamics_sha256']
                    and first['head'] == record['initial_head_sha256'] == record['final_head_sha256']
                    and last['dynamics'] == record['trace'][-1]['dynamics_retained_sha256'],
                    'trace retention and rollback join independently decoded final checkpoint')
        results.append({'arm': arm, 'seed': seed, 'initial': first, 'final': last,
                        'head_bytes_unchanged': True, 'retained_state_joined': True})
    return results
