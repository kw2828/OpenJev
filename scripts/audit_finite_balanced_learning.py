"""Independent saved-output audit for balanced transport under one shared training schedule.

The caller authenticates original sources, process closure and payloads before
``audit``. Only frozen scalar rational-world arithmetic is reused. No model,
world generator, producer or optimizer is imported or executed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from audit_finite_observation_learning import compare, multiply, probabilities, require, scalar_world

VERSION = 'finite-balanced-learning-audit-v1'
NAMESPACE = 431260924
ARMS = ('original_free', 'matched_free', 'balanced')
SEEDS = (431261001, 431261002, 431261003)
HORIZONS = (1, 2, 4, 8)
TARGETS = ('blind_costs', 'blind_survival', 'observed_costs', 'observed_survival', 'observed_probabilities')
FIELDS = (*TARGETS, 'shuffled_blind_costs')
DATA_KEYS = {'prefix', 'lengths', 'actions', 'observations', 'case_ids', *TARGETS}
CONFIG = {'seed_namespace': NAMESPACE, 'train_attempts': 512, 'dev_attempts': 128,
          'batch_size': 64, 'learning_rate': .003,
          'fit_seeds': list(SEEDS), 'gradient_clip': 5., 'train_horizon': 2, 'dev_horizon': 8, 'stage1_seconds': 10., 'total_seconds': 40.}
@dataclass(frozen=True)
class StudyProfile:
    name: str
    namespace: int
    seeds: tuple[int, ...]
    train_attempts: int
    dev_attempts: int
    batch_size: int
    stage1_seconds: float
    total_seconds: float

    @property
    def config(self):
        return {'seed_namespace': self.namespace, 'fit_seeds': list(self.seeds),
                'train_attempts': self.train_attempts, 'dev_attempts': self.dev_attempts,
                'batch_size': self.batch_size, 'learning_rate': .003, 'gradient_clip': 5.,
                'train_horizon': 2, 'dev_horizon': 8, 'stage1_seconds': self.stage1_seconds,
                'total_seconds': self.total_seconds}


SCIENCE = StudyProfile('science', NAMESPACE, SEEDS, 512, 128, 64, 10., 40.)
PROFILES = MappingProxyType({'science': SCIENCE,
    'engineering-940001': StudyProfile('engineering-940001', 940001, (940101,), 8, 8, 3, 2., 4.)})


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
KERNEL_WORK = {
    'balanced_transition_calls': 1, 'row_logsumexp_calls': 64, 'column_logsumexp_calls': 64,
    'normalization_vectors': 4096, 'normalization_entries': 32768, 'completed_sweeps': 64,
    'exponential_calls': 1, 'exponential_entries': 256, 'residual_sum_calls': 2, 'check_calls': 66,
}
ADAPTER_WORK_KEYS = {
    'probability_field_calls', 'adapter_check_calls', 'prior_transition_log_softmax_calls',
    'prior_emission_log_softmax_calls', 'prior_hazard_logsigmoid_calls', 'prior_normalization_vectors',
    'prior_normalization_entries', 'prior_logsigmoid_entries', 'matching_log_calls', 'matching_log_entries',
    'matching_copy_calls', 'matching_copy_entries', 'matching_verification_softmax_calls',
    'matching_verification_probability_columns', 'privileged_prefix_rows', 'event_probability_rows',
    'prefix_probability_rows', 'prefix_nll_rows',
}
WORK_KEYS = CORE_WORK_KEYS | FILTER_WORK_KEYS | HEAD_WORK_KEYS | FACTOR_WORK_KEYS | set(KERNEL_WORK) | ADAPTER_WORK_KEYS
ENDPOINT_WORK_KEYS = PREFIX_WORK_KEYS = WORK_KEYS


def construction_work(arm):
    require(arm in ARMS, 'known construction arm')
    result = dict.fromkeys(WORK_KEYS, 0)
    result['adapter_check_calls'] = 1
    if arm == 'matched_free':
        result.update(KERNEL_WORK)
        result.update(matching_log_calls=1, matching_log_entries=256,
                      matching_copy_calls=1, matching_copy_entries=256,
                      matching_verification_softmax_calls=1, matching_verification_probability_columns=32)
    return result


def forward_work(arm, values, *, prefix, prior=False):
    """Source-derived field/kernel work; prior logs occur only in stage-one loss."""
    require(arm in ARMS and (not prior or prefix), 'known route and prior scope')
    factors = 1 if prefix else 2
    result = {**values, **{key: count * factors for key, count in FACTORIZATION_WORK.items()},
              'reset_emission_softmax_calls': 0, 'prefix_operator_softmax_calls': 0,
              'operator_observed_softmax_calls': 0,
              'probability_field_calls': factors, 'adapter_check_calls': factors}
    if arm == 'balanced':
        result.update({key: count * factors for key, count in KERNEL_WORK.items()})
        result.update(transition_softmax_calls=0, transition_probability_columns=0)
    if not prefix:
        result['cost_head_softmax_calls'] = result['cost_readout_calls']
        result['cost_head_probability_rows'] = 8 * result['cost_readout_calls']
    if prior:
        result.update(prior_emission_log_softmax_calls=1, prior_hazard_logsigmoid_calls=2,
                      prior_normalization_vectors=8, prior_normalization_entries=32, prior_logsigmoid_entries=64)
        if arm != 'balanced':
            result.update(prior_transition_log_softmax_calls=1,
                          prior_normalization_vectors=40, prior_normalization_entries=288)
    require(set(result) == WORK_KEYS, 'complete named structural-work roster')
    return result


def validate_data(data, split, np, *, spec=SCIENCE):
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
    split_id, attempts = (0, spec.train_attempts) if split == 'train' else (1, spec.dev_attempts)
    pattern = re.compile(rf'ns{spec.namespace}-split{split_id}-case([0-9]{{10}})')
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



def reconstruct(data, prefix_oracle, exact_predictions, split, np, *, check=lambda: None, spec=SCIENCE):
    """Derive all targets and privileged prefix states from public tokens alone."""
    n, horizon = validate_data(data, split, np, spec=spec)
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



def validate_predictions(data, predictions, np, *, spec=SCIENCE):
    expected = {f'{arm}__{seed}__{field}' for arm in ARMS for seed in spec.seeds for field in FIELDS}
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



def rows_for(data, predictions, uniform_costs, np, *, check=lambda: None, spec=SCIENCE):
    validate_predictions(data, predictions, np, spec=spec)
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
        for seed in spec.seeds:
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



def gates(rows, baselines, counts, *, spec=SCIENCE):
    """Three separate predeclared criteria for every learned arm."""
    keyed = {(row['arm'], row['seed'], row['horizon']): row for row in rows}
    reference = {row['horizon']: row for row in baselines}
    require(len(rows) == len(keyed) == len(ARMS) * len(spec.seeds) * len(HORIZONS) and set(keyed)
            == {(arm, seed, h) for arm in ARMS for seed in spec.seeds for h in HORIZONS}, 'complete metric gate roster')
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
                for seed in spec.seeds:
                    row = keyed[arm, seed, horizon]
                    if kind != 'OBSERVED_FILTERING_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_half_mse'] = row['blind_cost_mse'] <= .5 * reference[horizon]['blind_cost_mse']
                        conditions[f'{seed}_h{horizon}_half_regret'] = row['blind_regret'] <= .5 * reference[horizon]['blind_regret']
                    if kind != 'BLIND_EXTRAPOLATION':
                        conditions[f'{seed}_h{horizon}_observed_kl'] = row['observed_kl'] <= .1
            if kind == 'BLIND_EXTRAPOLATION':
                for seed in spec.seeds:
                    conditions[f'{seed}_h8_survival'] = keyed[arm, seed, 8]['blind_survival_mae'] <= .05
            passed = all(conditions.values())
            criteria[kind] = {'passed': passed, 'status': kind + ('_PASS' if passed else '_FAIL'), 'conditions': conditions}
        results[arm] = criteria
    return results



def reconstruct_prefix(data, endpoints, split, np, *, check=lambda: None, spec=SCIENCE):
    """Validate the entire attempted population and condition only on public events."""
    require(split in ('train', 'base') and set(data) == PREFIX_KEYS, 'exact all-attempt prefix fields')
    n = spec.train_attempts if split == 'train' else spec.dev_attempts
    shapes = {'prefix': (n, 9, 31), 'lengths': (n,), 'case_ids': (n,), 'event_mask': (n, 9),
              'endpoint_eligible': (n,), 'endpoint_rows': (n,)}
    require(all(data[key].shape == shape for key, shape in shapes.items()), 'complete all-attempt prefix geometry')
    require(data['prefix'].dtype == np.float32 and data['lengths'].dtype == np.int64
            and data['endpoint_rows'].dtype == np.int64 and data['event_mask'].dtype == np.bool_
            and data['endpoint_eligible'].dtype == np.bool_ and data['case_ids'].dtype == np.dtype('U64'),
            'exact all-attempt prefix dtypes')
    expected_ids = np.array([f'ns{spec.namespace}-split{0 if split == "train" else 1}-case{i:010d}'
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



def prefix_row(data, prediction, arm, seed, np, *, spec=SCIENCE):
    """Independently score realized events; padding and reset-found are never targets."""
    require(arm in ARMS and seed in spec.seeds and set(prediction) == {'probabilities', 'nll'}, 'exact saved prefix prediction fields')
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


DYNAMIC_NAMES = ('transition_logits', 'emission_logits', 'hazard_logits')
PARAMETER_SHAPES = {'transition_logits': (4, 8, 8), 'emission_logits': (4, 8),
                    'hazard_logits': (4, 8), 'cost_logits': (4, 8)}

def digest_value(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite_scalar(value):
    return type(value) in (int, float) and math.isfinite(value)


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




def allocation_comparisons(rows, fits, original_gates, *, spec=SCIENCE):
    """Apply registered descriptive contrasts without rescuing an absolute failure."""
    keyed = {(r['arm'], r['seed'], r['horizon']): r for r in rows}
    fitted = {(r['arm'], r['seed']): r for r in fits}
    require(len(rows) == len(keyed) == len(ARMS) * len(spec.seeds) * len(HORIZONS) and set(keyed) ==
            {(arm, seed, horizon) for arm in ARMS for seed in spec.seeds for horizon in HORIZONS},
            'complete allocation metric roster')
    require(len(fits) == len(fitted) == len(ARMS) * len(spec.seeds) and set(fitted) == {(arm, seed) for arm in ARMS for seed in spec.seeds},
            'complete allocation fit timing roster')
    metrics = ('blind_regret', 'blind_cost_mse', 'observed_cost_mse', 'observed_kl',
               'blind_survival_mae', 'observed_survival_mae')
    require(all(finite_scalar(row[name]) and row[name] >= 0 for row in rows for name in metrics)
            and all(finite_scalar(row['seconds']) and row['seconds'] > 0 for row in fits),
            'finite nonnegative metrics and positive complete fit times')
    candidate = 'balanced'
    criteria = ('SHORT_HORIZON_LEARNING', 'BLIND_EXTRAPOLATION', 'OBSERVED_FILTERING_EXTRAPOLATION')
    require(set(original_gates) == set(ARMS)
            and all(set(original_gates[arm]) == set(criteria) for arm in ARMS), 'unchanged original criteria roster')
    conditions = {name: original_gates[candidate][name]['passed'] is True for name in criteria}
    paired, means, times = [], [], []
    for control in ('original_free', 'matched_free'):
        for horizon in (4, 8):
            candidate_mean = math.fsum(keyed[candidate, seed, horizon]['blind_regret'] for seed in spec.seeds) / len(spec.seeds)
            control_mean = math.fsum(keyed[control, seed, horizon]['blind_regret'] for seed in spec.seeds) / len(spec.seeds)
            conditions[f'{control}_h{horizon}_ten_percent_mean_regret'] = (
                control_mean > 0 and candidate_mean <= .9 * control_mean)
            means.append({'control': control, 'horizon': horizon,
                'candidate_mean_regret': candidate_mean, 'control_mean_regret': control_mean,
                'relative_reduction': None if control_mean == 0 else 1 - candidate_mean / control_mean,
                'tied_zero': candidate_mean == control_mean == 0})
            for seed in spec.seeds:
                first, second = keyed[candidate, seed, horizon], keyed[control, seed, horizon]
                conditions[f'{control}_{seed}_h{horizon}_nonpositive_regret_difference'] = first['blind_regret'] <= second['blind_regret']
                paired.append({'candidate': candidate, 'control': control, 'seed': seed, 'horizon': horizon,
                    **{name: {'candidate': first[name], 'control': second[name],
                              'difference': first[name] - second[name]} for name in metrics}})
        candidate_time = math.fsum(fitted[candidate, seed]['seconds'] for seed in spec.seeds) / len(spec.seeds)
        control_time = math.fsum(fitted[control, seed]['seconds'] for seed in spec.seeds) / len(spec.seeds)
        conditions[control + '_mean_fit_time_within_five_percent'] = candidate_time <= 1.05 * control_time
        times.append({'control': control, 'candidate_mean_seconds': candidate_time,
                      'control_mean_seconds': control_time, 'ratio': candidate_time / control_time})
    require(len(conditions) == 3 + 2 * (2 * len(spec.seeds) + 3), 'complete registered allocation advance conditions')
    passed = all(conditions.values())
    return {'paired': paired, 'mean_regret': means, 'mean_fit_time': times,
            'advance': {'name': 'BALANCED_ADVANCE', 'passed': passed,
                        'status': 'BALANCED_ADVANCE_PASS' if passed else 'BALANCED_ADVANCE_FAIL',
                        'conditions': conditions},
            'interpretation': 'All arms use the same prefix-then-joint eligibility windows. Forward cost, accepted updates and full fit time can differ. Matched-free initialization controls initial transition probabilities, not parameter gradients. No mean rescues an absolute gate or a worse paired regret.'}



CONTROLLER_WORK = {'validation_calls', 'optimizer_constructions', 'model_snapshot_calls',
    'model_snapshot_tensors', 'model_snapshot_bytes', 'optimizer_snapshot_calls',
    'optimizer_snapshot_tensors', 'optimizer_snapshot_bytes', 'model_restore_calls',
    'model_restore_tensors', 'model_restore_bytes', 'optimizer_restore_calls',
    'optimizer_restore_tensors', 'optimizer_restore_bytes', 'model_hash_calls',
    'optimizer_hash_calls', 'joint_update_calls', 'prefix_update_calls',
    'checkpoint_calls', 'check_calls', 'clock_calls'}
ALLOCATION_KEYS = {'version', 'arm', 'status', 'termination', 'stage1_seconds', 'total_seconds',
    'max_attempts', 'metadata', 'initial_model_sha256', 'initial_optimizer_sha256',
    'final_model_sha256', 'final_optimizer_sha256', 'joint_cursor', 'attempted_updates',
    'accepted_updates', 'accepted_joint_updates', 'accepted_prefix_updates', 'trace', 'stages',
    'boundary', 'checkpoints', 'work', 'update_work', 'timed_seconds', 'overrun_seconds',
    'compute_matched', 'timing_scope', 'final_summary', 'final_summary_seconds', 'final_summary_work'}
ATTEMPT_KEYS = {'attempt', 'stage', 'kind', 'deadline_seconds', 'start_elapsed', 'completed_elapsed',
    'accepted', 'rolled_back', 'error', 'cursor_before', 'cursor_attempted', 'cursor_retained',
    'model_before_sha256', 'optimizer_before_sha256', 'model_attempted_sha256',
    'optimizer_attempted_sha256', 'model_retained_sha256', 'optimizer_retained_sha256', 'result', 'work_delta'}
STAGE_KEYS = {'stage', 'kind', 'deadline_seconds', 'start_elapsed', 'stopped_elapsed', 'status',
    'termination', 'trace_start', 'trace_stop', 'attempted_updates', 'accepted_updates',
    'joint_cursor_start', 'joint_cursor_end', 'model_start_sha256', 'optimizer_start_sha256',
    'model_end_sha256', 'optimizer_end_sha256', 'overrun_seconds'}
CHECKPOINT_KEYS = {'label', 'start_elapsed', 'completed_elapsed', 'seconds', 'joint_cursor',
                   'model_sha256', 'optimizer_sha256', 'metadata'}


def integer_work(value, keys=None):
    require(type(value) is dict and (keys is None or set(value) == keys)
            and all(type(key) is str and type(number) is int and number >= 0
                    for key, number in value.items()), 'nonnegative integer executed-work counters')


def validate_allocation(result, arm, *, spec=SCIENCE):
    """Check the complete controller state machine; intermediate updates are attestations."""
    require(type(result) is dict and set(result) == ALLOCATION_KEYS, 'exact allocation result schema')
    require(result['version'] == 'finite-training-allocation-v1' and result['arm'] == 'prefix_then_joint' and arm in ARMS
            and result['status'] == 'PASS' and result['termination'] == 'completed_stages'
            and result['stage1_seconds'] == spec.config['stage1_seconds'] and result['total_seconds'] == spec.config['total_seconds']
            and result['max_attempts'] == 100000 and result['compute_matched'] is False,
            'completed registered allocation under one global budget')
    for key in ('initial_model_sha256', 'initial_optimizer_sha256', 'final_model_sha256', 'final_optimizer_sha256'):
        require(digest_value(result[key]), 'canonical allocation state digest')
    require(all(finite_scalar(result[key]) and result[key] >= 0
                for key in ('timed_seconds', 'overrun_seconds', 'final_summary_seconds')), 'finite allocation durations')
    trace, stages, checkpoints = result['trace'], result['stages'], result['checkpoints']
    require(type(trace) is list and 2 <= len(trace) <= 100000 and len(trace) == result['attempted_updates']
            and type(stages) is list and len(stages) == 2 and type(checkpoints) is list and len(checkpoints) == 3,
            'both stages and all three boundaries retained')
    total_work = dict.fromkeys(CONTROLLER_WORK, 0)
    update_work, accepted_joint, accepted_prefix, trace_index, cursor, loops = {}, 0, 0, 0, 0, 0
    model_hash, optimizer_hash = result['initial_model_sha256'], result['initial_optimizer_sha256']
    stage_records = []
    for index, stage in enumerate(stages, 1):
        kind = 'prefix' if index == 1 else 'joint'
        deadline = spec.config['stage1_seconds'] if index == 1 else spec.config['total_seconds']
        require(type(stage) is dict and set(stage) == STAGE_KEYS and stage['stage'] == index
                and stage['kind'] == kind and stage['deadline_seconds'] == deadline and stage['status'] == 'PASS'
                and stage['termination'] in ('deadline_reached', 'late_update_rolled_back'), 'exact successful stage schema')
        require(all(finite_scalar(stage[key]) and stage[key] >= 0
                    for key in ('start_elapsed', 'stopped_elapsed', 'overrun_seconds')),
                'finite nonnegative stage timing')
        if index == 2:
            boundary = result['boundary']
            require(type(boundary) is dict and set(boundary) == {'joint_cursor', 'model_sha256', 'stage1_optimizer_kind',
                'optimizer_before_sha256', 'optimizer_after_sha256', 'stage2_optimizer_kind', 'optimizer_reset'},
                'exact optimizer stage-boundary record')
            require(boundary['joint_cursor'] == cursor and boundary['model_sha256'] == model_hash
                    and boundary['optimizer_before_sha256'] == optimizer_hash
                    and boundary['stage1_optimizer_kind'] == 'prefix'
                    and boundary['stage2_optimizer_kind'] == 'joint'
                    and boundary['optimizer_reset'] is True
                    and digest_value(boundary['optimizer_after_sha256']), 'optimizer transition keeps model and accepted cursor')
            optimizer_hash = boundary['optimizer_after_sha256']
        require(stage['trace_start'] == trace_index and stage['joint_cursor_start'] == cursor
                and stage['model_start_sha256'] == model_hash and stage['optimizer_start_sha256'] == optimizer_hash,
                'stage begins at the last retained state and accepted batch')
        require(type(stage['attempted_updates']) is int and type(stage['accepted_updates']) is int
                and 1 <= stage['accepted_updates'] <= stage['attempted_updates']
                and stage['trace_stop'] == trace_index + stage['attempted_updates'] <= len(trace),
                'nonempty successful stage and exact trace slice')
        naccepted, late, previous_time = 0, 0, stage['start_elapsed']
        optimizer_updates = 0
        for row in trace[trace_index:stage['trace_stop']]:
            require(type(row) is dict and set(row) == ATTEMPT_KEYS and row['attempt'] == trace_index + 1
                    and row['stage'] == index and row['kind'] == kind and row['deadline_seconds'] == deadline
                    and row['error'] is None, 'ordered complete successful attempted update')
            start, end = row['start_elapsed'], row['completed_elapsed']
            require(finite_scalar(start) and finite_scalar(end) and previous_time <= start < deadline and end >= start,
                    'attempt begins before absolute deadline and completion is monotonic')
            eligible = end <= deadline
            require(row['accepted'] is eligible and row['rolled_back'] is (not eligible), 'deadline alone determines retention')
            require(eligible or trace_index == stage['trace_stop'] - 1, 'late attempt stops stage immediately')
            require(row['cursor_before'] == cursor and row['cursor_attempted'] == cursor + int(kind == 'joint')
                    and row['cursor_retained'] == cursor + int(kind == 'joint' and eligible), 'accepted joint cursor only')
            require(all(digest_value(row[key]) for key in ('model_before_sha256', 'optimizer_before_sha256',
                'model_attempted_sha256', 'optimizer_attempted_sha256', 'model_retained_sha256', 'optimizer_retained_sha256'))
                    and row['model_before_sha256'] == model_hash and row['optimizer_before_sha256'] == optimizer_hash,
                    'attempt state chain begins at retained model and optimizer')
            for label, previous in (('model', model_hash), ('optimizer', optimizer_hash)):
                require(row[label + '_retained_sha256'] == (row[label + '_attempted_sha256'] if eligible else previous),
                        'late atomic update restores both parameter and optimizer hashes')
            output = row['result']
            require(type(output) is dict and set(output) == {'loss', 'diagnostics', 'work'}
                    and finite_scalar(output['loss']) and type(output['diagnostics']) is dict, 'finite update result')
            integer_work(output['work'])
            for key, value in output['work'].items():
                update_work[key] = update_work.get(key, 0) + value
            params, param_bytes = (3, 2560) if kind == 'prefix' else (4, 2816)
            opt_tensors = 3 * params if optimizer_updates else 0
            opt_bytes = 2 * param_bytes + 4 * params if optimizer_updates else 0
            expected = dict.fromkeys(CONTROLLER_WORK, 0)
            expected.update(model_snapshot_calls=1, model_snapshot_tensors=4, model_snapshot_bytes=2816,
                optimizer_snapshot_calls=1, optimizer_snapshot_tensors=opt_tensors, optimizer_snapshot_bytes=opt_bytes,
                model_hash_calls=2 + int(not eligible), optimizer_hash_calls=2 + int(not eligible),
                check_calls=1, clock_calls=1)
            expected[kind + '_update_calls'] = 1
            if not eligible:
                expected.update(model_restore_calls=1, model_restore_tensors=4, model_restore_bytes=2816,
                    optimizer_restore_calls=1, optimizer_restore_tensors=opt_tensors, optimizer_restore_bytes=opt_bytes)
            integer_work(row['work_delta'], CONTROLLER_WORK)
            compare(row['work_delta'], expected, path='all snapshot/hash/attempt/rollback controller work')
            for key, value in expected.items():
                total_work[key] += value
            optimizer_updates += int(eligible)
            naccepted += int(eligible)
            late += int(not eligible)
            cursor = row['cursor_retained']
            model_hash, optimizer_hash = row['model_retained_sha256'], row['optimizer_retained_sha256']
            previous_time = end
            trace_index += 1
        require(stage['accepted_updates'] == naccepted and stage['joint_cursor_end'] == cursor
                and stage['model_end_sha256'] == model_hash and stage['optimizer_end_sha256'] == optimizer_hash
                and stage['stopped_elapsed'] >= previous_time, 'stage retained state and totals')
        extra_loop = int(not late and previous_time < deadline)
        require(stage['termination'] == ('late_update_rolled_back' if late else 'deadline_reached')
                and (not extra_loop or stage['stopped_elapsed'] >= deadline), 'stage stops at the registered deadline')
        compare(stage['overrun_seconds'], max(0., stage['stopped_elapsed'] - deadline), path='stage overrun charged')
        loops += extra_loop
        accepted_joint += naccepted if kind == 'joint' else 0
        accepted_prefix += naccepted if kind == 'prefix' else 0
        stage_records.append({'stage': index, 'kind': kind, 'attempted_updates': stage['attempted_updates'],
            'accepted_updates': naccepted, 'late_updates': late, 'joint_cursor_start': stage['joint_cursor_start'],
            'joint_cursor_end': cursor, 'overrun_seconds': stage['overrun_seconds']})
    require(trace_index == len(trace) and result['joint_cursor'] == cursor == accepted_joint
            and result['accepted_joint_updates'] == accepted_joint and result['accepted_prefix_updates'] == accepted_prefix
            and result['accepted_updates'] == accepted_joint + accepted_prefix
            and result['final_model_sha256'] == model_hash and result['final_optimizer_sha256'] == optimizer_hash,
            'complete retained allocation totals')
    for index, label in enumerate(('initial', 'boundary', 'final')):
        row = checkpoints[index]
        require(type(row) is dict and set(row) == CHECKPOINT_KEYS and row['label'] == label
                and all(finite_scalar(row[key]) and row[key] >= 0 for key in ('start_elapsed', 'completed_elapsed', 'seconds'))
                and row['completed_elapsed'] >= row['start_elapsed'], 'complete immutable checkpoint callback record')
        compare(row['seconds'], row['completed_elapsed'] - row['start_elapsed'], path='checkpoint callback time')
        expected_model = result['initial_model_sha256'] if index == 0 else stages[index - 1]['model_end_sha256']
        expected_opt = result['initial_optimizer_sha256'] if index == 0 else stages[index - 1]['optimizer_end_sha256']
        require(row['model_sha256'] == expected_model and row['optimizer_sha256'] == expected_opt
                and row['joint_cursor'] == (0 if index == 0 else stages[index - 1]['joint_cursor_end']),
                'checkpoint is taken before restart from exact retained stage state')
    require(checkpoints[0]['completed_elapsed'] <= stages[0]['start_elapsed']
            <= stages[0]['stopped_elapsed'] <= checkpoints[1]['start_elapsed']
            and checkpoints[1]['completed_elapsed'] <= stages[1]['start_elapsed']
            <= stages[1]['stopped_elapsed'] <= checkpoints[2]['start_elapsed']
            and checkpoints[2]['completed_elapsed'] <= result['timed_seconds'], 'one global chronology includes boundary IO')
    total_work.update(validation_calls=1, optimizer_constructions=2, checkpoint_calls=3)
    for key in ('model_hash_calls', 'optimizer_hash_calls'):
        total_work[key] += 13
    total_work['check_calls'] += 10 + len(trace) + loops
    total_work['clock_calls'] += 12 + len(trace) + loops
    integer_work(result['work'], CONTROLLER_WORK)
    compare(result['work'], total_work, path='all controller work including rejected updates and boundaries')
    compare(result['update_work'], update_work, path='all numerical update work including discarded work')
    final_work = dict.fromkeys(CONTROLLER_WORK, 0)
    final_work.update(model_hash_calls=1, optimizer_hash_calls=1, check_calls=1, clock_calls=1)
    compare(result['final_summary_work'], final_work, path='separately charged read-only final summary controller work')
    compare(result['overrun_seconds'], max(0., result['timed_seconds'] - spec.config['total_seconds']), path='complete global overrun')
    return {'arm': arm, 'stages': stage_records, 'accepted_joint_updates': accepted_joint,
            'accepted_prefix_updates': accepted_prefix, 'attempted_updates': len(trace),
            'timed_seconds': result['timed_seconds'], 'overrun_seconds': result['overrun_seconds'],
            'final_summary_seconds': result['final_summary_seconds'], 'update_replay': False}



def optimizer_state(payload, kind, steps):
    """Validate lossless Adam JSON and its independently computed canonical digest."""
    require(kind in ('prefix', 'joint') and type(steps) is int and steps >= 0,
            'known optimizer kind and retained updates')
    require(type(payload) is dict and set(payload) == {'state', 'param_groups'}
            and type(payload['state']) is dict and type(payload['param_groups']) is list
            and len(payload['param_groups']) == 1, 'one exact Adam parameter group')
    group = payload['param_groups'][0]
    shapes = list(PARAMETER_SHAPES.values())[:3 if kind == 'prefix' else 4]
    require(type(group) is dict and group['params'] == list(range(len(shapes)))
            and group['lr'] == .003 and group['betas'] == [.9, .999] and group['eps'] == 1e-8
            and group['weight_decay'] == 0 and group['amsgrad'] is False
            and group['maximize'] is False and group['capturable'] is False and group['differentiable'] is False
            and group['foreach'] is None and group['fused'] is None
            and group.get('decoupled_weight_decay', False) is False, 'registered default Adam hyperparameters')
    state = payload['state']
    require(set(state) == ({str(i) for i in range(len(shapes))} if steps else set()),
            'Adam moments exist for exactly the parameters updated')
    total_tensors, total_bytes = 0, 0

    def tensor(item, shape, dtype, nonnegative=False):
        require(type(item) is dict and set(item) == {'kind', 'dtype', 'shape', 'values'}
                and item['kind'] == 'tensor' and item['dtype'] == dtype and item['shape'] == list(shape),
                'exact optimizer tensor shape/dtype')
        def values(value, dimensions):
            if not dimensions:
                require(finite_scalar(value) and (not nonnegative or value >= 0), 'finite optimizer state value')
                return [value]
            require(type(value) is list and len(value) == dimensions[0], 'optimizer JSON tensor geometry')
            return [entry for part in value for entry in values(part, dimensions[1:])]
        return values(item['values'], shape)

    for index, shape in enumerate(shapes):
        if not steps:
            break
        record = state[str(index)]
        require(type(record) is dict and set(record) == {'step', 'exp_avg', 'exp_avg_sq'}, 'exact ordinary Adam state')
        require(tensor(record['step'], (), 'torch.float32', True) == [steps], 'retained Adam step count')
        mean = tensor(record['exp_avg'], shape, 'torch.float64')
        tensor(record['exp_avg_sq'], shape, 'torch.float64', True)
        total_tensors += 3
        total_bytes += 4 + 16 * len(mean)
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise ValueError('finite lossless optimizer JSON') from error
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'tensors': total_tensors,
            'bytes': total_bytes, 'steps': steps, 'kind': kind}


def initial_transition_checks(checkpoints, np):
    """Rebuild fixed 64-sweep initial probabilities from saved logits only."""
    seeds = sorted({seed for label, _arm, seed in checkpoints if label == 'initial'})
    result = []
    for seed in seeds:
        initial = {arm: checkpoints['initial', arm, seed] for arm in ARMS}
        for arrays in initial.values():
            boundary_hashes(arrays, np)
        for name in ('emission_logits', 'hazard_logits', 'cost_logits'):
            require(len({initial[arm][name].tobytes(order='C') for arm in ARMS}) == 1,
                    'all initial emission, hazard and cost parameters agree exactly')
        require(initial['original_free']['transition_logits'].tobytes(order='C') ==
                initial['balanced']['transition_logits'].tobytes(order='C'),
                'original-free and balanced raw transition initialization agrees exactly')
        logs = initial['balanced']['transition_logits'].copy()
        for _ in range(64):
            for axis in (2, 1):
                maximum = np.max(logs, axis=axis, keepdims=True)
                normalizer = maximum + np.log(np.exp(logs - maximum).sum(axis=axis, keepdims=True))
                logs = logs - normalizer
        balanced = np.exp(logs)
        raw = initial['matched_free']['transition_logits']
        shifted = np.exp(raw - np.max(raw, axis=1, keepdims=True))
        matched = shifted / shifted.sum(axis=1, keepdims=True)
        row_error = float(np.max(np.abs(balanced.sum(axis=2) - 1)))
        column_error = float(np.max(np.abs(balanced.sum(axis=1) - 1)))
        match_error = float(np.max(np.abs(balanced - matched)))
        require(np.isfinite(balanced).all() and np.isfinite(matched).all()
                and ((balanced > 0) & (balanced < 1)).all()
                and ((matched > 0) & (matched < 1)).all()
                and row_error <= 1e-12 and column_error <= 1e-12 and match_error <= 1e-12,
                'independent 64-sweep balanced versus matched-free initial transition')
        result.append({'seed': seed, 'sweeps': 64, 'row_residual_max': row_error,
                       'column_residual_max': column_error, 'matched_max_abs': match_error,
                       'shared_initial_emission_hazard_head': True, 'shared_original_balanced_raw_transition': True})
    return result


def verify_boundaries(allocations, checkpoints, optimizer_payloads, np):
    """Join all three saved boundaries and validate the distinct initialization control."""
    initialization = {r['seed']: r for r in initial_transition_checks(checkpoints, np)}
    result = []
    for (arm, seed), allocation in allocations.items():
        initial = checkpoints['initial', arm, seed]
        initial_optimizer = optimizer_payloads['initial', arm, seed]
        optimizer_state(initial_optimizer, 'prefix', 0)
        empty_joint = {'state': {}, 'param_groups': [
            {**initial_optimizer['param_groups'][0], 'params': list(range(4))}]}
        empty_joint_hash = optimizer_state(empty_joint, 'joint', 0)['sha256']
        checked = []
        for index, label in enumerate(('initial', 'boundary', 'final')):
            arrays = checkpoints[label, arm, seed]
            hashes = boundary_hashes(arrays, np)
            steps = 0 if index == 0 else allocation['stages'][index - 1]['accepted_updates']
            kind = 'prefix' if index < 2 else 'joint'
            opt = optimizer_state(optimizer_payloads[label, arm, seed], kind, steps)
            record = allocation['checkpoints'][index]
            require(hashes['full'] == record['model_sha256'] and opt['sha256'] == record['optimizer_sha256'],
                    'independently decoded parameter and optimizer boundary digests')
            checked.append({'label': label, 'model': hashes, 'optimizer': opt})
        require(initial['cost_logits'].tobytes() == checkpoints['boundary', arm, seed]['cost_logits'].tobytes(),
                'every prefix-only stage preserves actual cost-head bytes')
        require(allocation['boundary']['optimizer_after_sha256'] == empty_joint_hash,
                'every joint stage uses fresh empty Adam state with the registered parameter group')
        result.append({'arm': arm, 'seed': seed, 'boundaries': checked,
                       'initial_transition_check': initialization[seed],
                       'optimizer_and_model_boundaries_joined': True, 'historical_updates_replayed': False})
    return result


UPDATE_WORK = {'joint_updates', 'prefix_updates', 'joint_attempt_exposures', 'joint_case_exposures',
    'joint_event_exposures', 'prefix_event_exposures', 'joint_blind_rollouts', 'joint_observed_rollouts',
    'joint_prefix_rollouts', 'prefix_full_rollouts', 'zero_endpoint_batches', 'backward_passes', 'adam_steps'}


def parameter_metadata(pm):
    expected = {name: {'count': math.prod(shape), 'dtype': 'torch.float64', 'bytes': 8 * math.prod(shape)}
                for name, shape in PARAMETER_SHAPES.items()}
    require(pm['parameter_count'] == 352 and pm['parameter_bytes'] == 2816 and pm['buffer_bytes'] == 0
            and pm['parameters'] == expected, 'unchanged 352-parameter factorized model')


def validate_metadata(folder, summary, *, check=lambda: None, spec=SCIENCE):
    """Authenticate the exact saved roster and all successful allocation records first."""
    def read(name, *, lines=False):
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular saved metadata')
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    require(summary['version'] == 'finite-balanced-learning-v1', 'registered allocation producer')
    compare(summary['config'], spec.config, path='registered allocation configuration')
    expected_pairs = [(arm, seed) for i, seed in enumerate(spec.seeds) for arm in ARMS[i:] + ARMS[:i]]
    fits = summary['fits']
    require([(r['arm'], r['seed']) for r in fits] == expected_pairs, 'nine complete rotated fits')
    expected_files = {'config.json', 'train.npz', 'base.npz', 'train-oracle.npz', 'base-oracle.npz',
        'oracle-train.npz', 'oracle-base.npz', 'predictions-base.npz', 'fits.jsonl', 'training-orders.jsonl',
        'checkpoint-barrier.json', 'prediction-times.jsonl', 'oracle-train-check.json', 'oracle-base-check.json',
        'train-prefix.npz', 'base-prefix.npz'}
    for arm, seed in expected_pairs:
        expected_files |= {f'prefix-{arm}-{seed}-base.npz', f'allocation-{arm}-{seed}.json'}
        for label in ('initial', 'boundary', 'final'):
            expected_files |= {f'{label}-{arm}-{seed}.npz', f'{label}-optimizer-{arm}-{seed}.json'}
    require(len(expected_files) == 16 + 8 * len(ARMS) * len(spec.seeds) and set(summary['files']) == expected_files, 'exact complete allocation payload roster')
    for name, descriptor in summary['files'].items():
        check()
        path = folder / name
        require(path.is_file() and not path.is_symlink(), 'regular original allocation payload')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        compare(descriptor, {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}, path='payload ' + name)
    compare(read('config.json'), spec.config, path='durable configuration')
    compare(read('fits.jsonl', lines=True), fits, path='durable fits')
    require(set(summary['dataset_counts']) == {'train', 'base'}, 'exact TRAIN and fresh DEV')
    retained = {}
    for split_id, split in enumerate(('train', 'base')):
        row = summary['dataset_counts'][split]
        attempts, horizon = (spec.train_attempts, 2) if split == 'train' else (spec.dev_attempts, 8)
        require(row['version'] == 'finite-prefix-learning-data-v1' and row['epsilon'] == .12
                and row['seed_namespace'] == spec.namespace and row['split_id'] == split_id
                and row['attempted'] == attempts and row['horizon'] == horizon, 'registered fixed population')
        n, excluded, found = row['retained'], row['discarded_found'], row['prefix_found_by_step']
        require(type(n) is int and 0 < n <= attempts and type(excluded) is int and excluded >= 0
                and n + excluded == attempts and type(found) is list and len(found) == 8
                and all(type(v) is int and v >= 0 for v in found) and sum(found) == excluded,
                'complete attempted prefix population')
        require(row['initial_odor_draws'] == attempts and row['prefix_action_draws'] == 8 * attempts
                and row['prefix_event_draws'] == 8 * n + sum((i + 1) * v for i, v in enumerate(found))
                and row['forecast_action_draws'] == horizon * n
                and row['valid_prefix_events'] == attempts + row['prefix_event_draws'], 'complete public event counts')
        retained[split] = n
        errors = summary['oracle_checks'][split]
        require(set(errors) == set(TARGETS) and all(finite_scalar(v) and 0 <= v <= 1e-12 for v in errors.values()),
                'every exact-control check passed')
        compare(read('oracle-' + split + '-check.json'), errors, path='durable exact-control check')
    require(summary['oracle_metadata']['parameter_count'] == 0
            and summary['oracle_metadata']['parameter_bytes'] == 0 and summary['oracle_metadata']['parameters'] == {}
            and digest_value(summary['oracle_state_sha256']), 'fixed privileged exact reference')
    allocations, allocation_checks = {}, []
    for row in fits:
        arm, seed = row['arm'], row['seed']
        parameter_metadata(row['parameter_metadata'])
        require(row['oracle_prefix_input'] is False and digest_value(row['initial_state_sha256'])
                and digest_value(row['final_state_sha256']) and finite_scalar(row['seconds']) and row['seconds'] > 0
                and finite_scalar(row['construction_seconds']) and row['construction_seconds'] >= 0,
                'public-only fit and complete finite timing')
        name = f'allocation-{arm}-{seed}.json'
        compare(row['allocation'], {'path': name, **summary['files'][name]}, path='complete allocation descriptor')
        allocation = read(name)
        allocations[arm, seed] = allocation
        allocation_checks.append({'seed': seed, **validate_allocation(allocation, arm, spec=spec)})
        compare(allocation['metadata'], {'parameter_metadata': row['parameter_metadata'], 'attempts': spec.train_attempts,
            'eligible': retained['train'], 'events': summary['dataset_counts']['train']['valid_prefix_events'],
            'batch_size': spec.batch_size, 'seed': seed, 'hidden_state_input': False}, path='fixed public training denominators')
        compare(allocation['final_summary'], {'parameter_metadata': row['parameter_metadata'],
            'joint_cursor': allocation['joint_cursor'], 'construction_seconds': row['construction_seconds']},
            path='read-only final metadata')
        require(row['initial_state_sha256'] == allocation['initial_model_sha256']
                and row['final_state_sha256'] == allocation['final_model_sha256']
                and row['updates'] == allocation['accepted_joint_updates']
                and row['accepted_prefix_updates'] == allocation['accepted_prefix_updates']
                and row['attempted_updates'] == allocation['attempted_updates'], 'fit joins retained controller state and work')
        compare(row['timed_seconds'], allocation['timed_seconds'], path='fit eligibility time')
        compare(row['final_summary_seconds'], allocation['final_summary_seconds'], path='fit outside-budget summary')
        require(row['seconds'] + 1e-8 >= allocation['timed_seconds'] + allocation['final_summary_seconds'],
                'complete fit time charges eligibility, overrun and final summary')
        integer_work(row['construction_work'], WORK_KEYS)
        compare(row['construction_work'], construction_work(arm), path='arm-specific constructor work')
        for point in allocation['checkpoints']:
            label, record = point['label'], point['metadata']
            require(type(record) is dict and set(record) == {'label', 'model', 'optimizer', 'model_state_sha256',
                'optimizer_state_sha256', 'joint_cursor'} and record['label'] == label
                and record['model_state_sha256'] == point['model_sha256']
                and record['optimizer_state_sha256'] == point['optimizer_sha256']
                and record['joint_cursor'] == point['joint_cursor'], 'complete boundary artifact joins')
            for kind, filename in (('model', f'{label}-{arm}-{seed}.npz'),
                                    ('optimizer', f'{label}-optimizer-{arm}-{seed}.json')):
                compare(record[kind], {'path': filename, **summary['files'][filename]}, path='boundary ' + kind)
        compare(row['checkpoint'], allocation['checkpoints'][-1]['metadata']['model'], path='final retained checkpoint')
    barrier = {'checkpoints': [{'arm': row['arm'], 'seed': row['seed'], **row['checkpoint']} for row in fits],
               'dev_generation_count': 0, 'fit_count': len(fits), 'oracle_train_verified': True}
    compare(summary['checkpoint_barrier'], barrier, path='all-nine-finals-before-DEV attestation')
    compare(read('checkpoint-barrier.json'), barrier, path='durable generation barrier')
    times = summary['prediction_times']
    require([(r['arm'], r['seed']) for r in times] == [(arm, seed) for arm in ARMS for seed in spec.seeds], 'all nine endpoint views')
    compare(read('prediction-times.jsonl', lines=True), times, path='durable prediction records')
    fit_map = {(r['arm'], r['seed']): r for r in fits}
    for row in times:
        require(row['regime'] == 'base' and row['cases'] == retained['base'] and row['batch_size'] == spec.batch_size
                and row['oracle_prefix_input'] is False and row['shuffle_offset'] == 1
                and row['model_state_before'] == row['model_state_after'] == fit_map[row['arm'], row['seed']]['final_state_sha256']
                and finite_scalar(row['seconds']) and row['seconds'] >= 0, 'evaluation leaves complete retained state unchanged')
    require([(r['arm'], r['seed']) for r in summary['prefix_prediction_times']]
            == [(arm, seed) for arm in ARMS for seed in spec.seeds], 'nine all-attempt prefix evaluations')
    for row in summary['prefix_prediction_times']:
        require(row['attempts'] == spec.dev_attempts and row['valid_events'] == summary['dataset_counts']['base']['valid_prefix_events']
                and finite_scalar(row['seconds']) and row['seconds'] >= 0, 'complete prefix diagnostic timing')
    # Numeric case/counter reconstruction happens only after this entire opaque manifest is admitted.
    return {'fits': len(fits), 'payloads': len(expected_files), 'checkpoint_files_hashed': 3 * len(fits), 'optimizer_files_hashed': 3 * len(fits),
            'allocation_checks': allocation_checks, 'retained': retained,
            'historical_ordering_independently_replayed': False}, allocations, read('training-orders.jsonl', lines=True)



def validate_execution(summary, allocations, datasets, prefixes, orders, np, *, check=lambda: None, spec=SCIENCE):
    """Reconstruct attempted batches and forward work, including discarded updates."""
    declared = summary['structural_work']
    require(set(declared) == set(ARMS) and all(set(declared[arm]) == set(ROUTES) for arm in ARMS),
            'all common structural-work routes')
    expected = {arm: {route: dict.fromkeys(PREFIX_WORK_KEYS if route.endswith('_prefix') else ENDPOINT_WORK_KEYS, 0)
                      for route in ROUTES} for arm in ARMS}
    order_map, order_sequence = {}, []
    for row in orders:
        check()
        require(type(row) is dict and set(row) == {'arm', 'seed', 'epoch', 'indices', 'batch_size'}
                and row['arm'] in ARMS and row['seed'] in spec.seeds and type(row['epoch']) is int and row['epoch'] >= 0
                and row['batch_size'] == spec.batch_size and type(row['indices']) is list and len(row['indices']) == spec.train_attempts
                and all(type(v) is int for v in row['indices']) and sorted(row['indices']) == list(range(spec.train_attempts)),
                'one complete deterministic epoch order')
        key = row['arm'], row['seed'], row['epoch']
        require(key not in order_map, 'each used epoch serialized once')
        rebuilt = np.random.Generator(np.random.PCG64(np.random.SeedSequence([row['seed'], row['epoch'], 818]))).permutation(spec.train_attempts)
        require(np.array_equal(rebuilt, np.asarray(row['indices'], np.int64)), 'registered pure cursor-to-permutation schedule')
        order_map[key] = rebuilt
        order_sequence.append(key)
    observed_orders = []
    seen_orders = set()
    aggregate = dict.fromkeys(UPDATE_WORK, 0)
    retained_work = dict.fromkeys(UPDATE_WORK, 0)
    rejected_work = dict.fromkeys(UPDATE_WORK, 0)

    def add(arm, route, values, *, prior=False):
        values = forward_work(arm, values, prefix=route.endswith('_prefix'), prior=prior)
        for key, value in values.items():
            expected[arm][route][key] += value

    for (arm, seed), allocation in allocations.items():
        for row in allocation['trace']:
            check()
            kind, diagnostics = row['kind'], row['result']['diagnostics']
            work = dict.fromkeys(UPDATE_WORK, 0)
            work.update(backward_passes=1, adam_steps=1)
            if kind == 'prefix':
                require(set(diagnostics) == {'kind', 'valid_events', 'attempts', 'pseudocount', 'cost_head_updated'}
                        and diagnostics['kind'] == 'prefix' and diagnostics['valid_events'] == int(prefixes['train']['event_mask'].sum())
                        and diagnostics['attempts'] == spec.train_attempts and diagnostics['pseudocount'] == .001
                        and diagnostics['cost_head_updated'] is False, 'registered full-public-prefix objective')
                work.update(prefix_updates=1, prefix_event_exposures=diagnostics['valid_events'], prefix_full_rollouts=1)
                add(arm, 'training_prefix', prefix_work(prefixes['train'], np.arange(spec.train_attempts), np), prior=True)
            else:
                cursor = row['cursor_before']
                epoch, offset = divmod(cursor, math.ceil(spec.train_attempts / spec.batch_size))
                key = arm, seed, epoch
                require(key in order_map, 'every attempted batch has its original epoch order')
                if key not in seen_orders:
                    seen_orders.add(key)
                    observed_orders.append(key)
                order = order_map[key]
                indices = order[offset * spec.batch_size:(offset + 1) * spec.batch_size]
                selected = prefixes['train']['endpoint_rows'][indices]
                selected = selected[selected >= 0]
                events = int(prefixes['train']['event_mask'][indices].sum())
                compare(diagnostics, {'kind': 'joint', 'cursor': cursor, 'epoch': epoch, 'batch_offset': offset,
                    'indices': indices.tolist(), 'eligible': len(selected), 'valid_events': events, 'batch_size': len(indices),
                    'order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()},
                    path='actual attempted joint batch and fixed global denominators')
                work.update(joint_updates=1, joint_attempt_exposures=len(indices), joint_case_exposures=len(selected),
                    joint_event_exposures=events, joint_prefix_rollouts=1, zero_endpoint_batches=int(not len(selected)))
                if len(selected):
                    observations = datasets['train']['observations'][selected]
                    add(arm, 'training_blind', endpoint_work(observations, False, np))
                    add(arm, 'training_observed', endpoint_work(observations, True, np))
                    work.update(joint_blind_rollouts=1, joint_observed_rollouts=1)
                add(arm, 'training_prefix', prefix_work(prefixes['train'], indices, np))
            integer_work(row['result']['work'], UPDATE_WORK)
            compare(row['result']['work'], work, path='independent attempted update and exposure counts')
            destination = retained_work if row['accepted'] else rejected_work
            for key, value in work.items():
                aggregate[key] += value
                destination[key] += value
    require(order_sequence == observed_orders, 'complete first-use epoch order journal without extra generated orders')
    for arm in ARMS:
        for _seed in spec.seeds:
            for start in range(0, len(datasets['base']['case_ids']), spec.batch_size):
                observations = datasets['base']['observations'][start:start + spec.batch_size]
                for route in ('evaluation_blind', 'evaluation_observed', 'evaluation_shuffled'):
                    add(arm, route, endpoint_work(observations, route == 'evaluation_observed', np))
            for start in range(0, spec.dev_attempts, spec.batch_size):
                add(arm, 'evaluation_prefix', prefix_work(prefixes['base'], np.arange(start, min(start + spec.batch_size, spec.dev_attempts)), np))
    for arm in ARMS:
        for route in ROUTES:
            value = declared[arm][route]
            integer_work(value)
            if value == {}:
                require(not any(expected[arm][route].values()), 'unused route alone may have no counters')
            else:
                compare(value, expected[arm][route], path='independently derived forward geometry: ' + arm + '/' + route)
    n, dev_n = len(datasets['train']['case_ids']), len(datasets['base']['case_ids'])
    attempted = sum(a['attempted_updates'] for a in allocations.values())
    fits = len(ARMS) * len(spec.seeds)
    eval_batches = fits * math.ceil(dev_n / spec.batch_size)
    oracle_batches = math.ceil(n / spec.batch_size) + math.ceil(dev_n / spec.batch_size)
    counts = {'train_generation_count': 1, 'dev_generation_count': 1, 'model_constructions': fits,
        'oracle_model_constructions': 1, 'fit_count': fits, 'checkpoint_writes': 3 * fits, 'optimizer_checkpoint_writes': 3 * fits,
        'training_blind_rollouts': aggregate['joint_blind_rollouts'], 'training_observed_rollouts': aggregate['joint_observed_rollouts'],
        'optimizer_attempts': attempted, 'optimizer_steps': attempted,
        'training_case_exposures': aggregate['joint_case_exposures'],
        'training_attempt_exposures': aggregate['joint_attempt_exposures'],
        'training_prefix_rollouts': aggregate['joint_prefix_rollouts'] + aggregate['prefix_full_rollouts'],
        'training_prefix_event_exposures': aggregate['joint_event_exposures'] + aggregate['prefix_event_exposures'],
        'zero_endpoint_batches': aggregate['zero_endpoint_batches'],
        'evaluation_blind_rollouts': eval_batches, 'evaluation_observed_rollouts': eval_batches,
        'evaluation_shuffled_rollouts': eval_batches, 'evaluation_case_views': fits * dev_n,
        'evaluation_prefix_rollouts': fits * math.ceil(spec.dev_attempts / spec.batch_size), 'evaluation_prefix_event_views': fits * int(prefixes['base']['event_mask'].sum()),
        'oracle_blind_rollouts': oracle_batches, 'oracle_observed_rollouts': oracle_batches,
        'array_decodes': 0, 'checkpoint_decodes': 0, 'external_model_calls': 0, 'native_calls': 0, 'teacher_calls': 0,
        'readout_snapshot_evaluations': 0, 'readout_snapshot_softmax_evaluations': 0, 'dynamics_snapshot_evaluations': 0,
        'epoch_order_generations': len(orders), 'accepted_optimizer_steps': retained_work['adam_steps'],
        'accepted_joint_steps': retained_work['joint_updates'], 'accepted_prefix_steps': retained_work['prefix_updates']}
    compare(summary['counts'], counts, path='all executed counts include rejected work; accepted counts remain separate')
    return {'arms': 3, 'routes': 21, 'epoch_permutations_reconstructed': len(orders),
            'attempted_update_work': aggregate, 'accepted_update_work': retained_work, 'discarded_update_work': rejected_work,
            'structural_forward_geometry_checked': True, 'model_updates_replayed': False}



def audit(folder, *, check=lambda: None, profile='science'):
    """Read only an independently admitted and closed producer's saved outputs."""
    import numpy as np

    require(type(profile) is str and profile in PROFILES, 'explicit immutable audit profile')
    spec = PROFILES[profile]
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve() == folder, 'absolute original saved-output directory')
    summary_path = folder / 'summary.json'
    require(summary_path.is_file() and not summary_path.is_symlink(), 'regular producer summary')
    summary = json.loads(summary_path.read_text())
    metadata, allocations, orders = validate_metadata(folder, summary, check=check, spec=spec)
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

    boundary_arrays = {(label, arm, seed): load(f'{label}-{arm}-{seed}.npz', checkpoint=True)
        for arm, seed in allocations for label in ('initial', 'boundary', 'final')}
    optimizer_payloads = {(label, arm, seed): json.loads((folder / f'{label}-optimizer-{arm}-{seed}.json').read_text())
        for arm, seed in allocations for label in ('initial', 'boundary', 'final')}
    counters['optimizer_json_decodes'] = len(optimizer_payloads)
    boundary_checks = verify_boundaries(allocations, boundary_arrays, optimizer_payloads, np)
    records, datasets, prefixes, targets, uniform, prefix_records = {}, {}, {}, {}, {}, {}
    for split in ('train', 'base'):
        data, prefix = load(split + '.npz'), load(split + '-prefix.npz')
        oracle, exact = load(split + '-oracle.npz'), load('oracle-' + split + '.npz')
        record, rebuilt, reference = reconstruct(data, oracle, exact, split, np, check=check, spec=spec)
        prefix_record, states, _reference = reconstruct_prefix(prefix, data, split, np, check=check, spec=spec)
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
    rows, baselines = rows_for({**datasets['base'], **targets['base']}, predictions, uniform['base'], np, check=check, spec=spec)
    row_key = lambda row: (row['arm'], row['seed'], row['horizon'])
    compare(sorted(summary['rows'], key=row_key), sorted(rows, key=row_key), path='independent endpoint metrics')
    compare(sorted(summary['baseline_rows'], key=lambda row: row['horizon']), baselines, path='independent uniform-state reference')
    prefix_rows = [prefix_row(prefixes['base'], load(f'prefix-{arm}-{seed}-base.npz'), arm, seed, np, spec=spec)
                   for arm in ARMS for seed in spec.seeds]
    compare(summary['prefix_rows'], prefix_rows, path='independent all-attempt event-weighted prefix NLL')
    counts = {split: len(data['case_ids']) for split, data in datasets.items()}
    compare(counts, metadata['retained'], path='endpoint support matches metadata')
    metadata['structural_work_validation'] = validate_execution(summary, allocations, datasets, prefixes, orders, np, check=check, spec=spec)
    require(counters['array_decodes'] == 9 + 4 * len(ARMS) * len(spec.seeds) and counters['checkpoint_decodes'] == 3 * len(ARMS) * len(spec.seeds)
            and counters['optimizer_json_decodes'] == 3 * len(ARMS) * len(spec.seeds), 'exact saved data/model/optimizer decode roster')
    original_gates = gates(rows, baselines, counts, spec=spec)
    contrasts = allocation_comparisons(rows, summary['fits'], original_gates, spec=spec)
    return {'version': VERSION, 'profile': profile, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'counts': counters, 'metadata': metadata,
            'data_cases': counts, 'target_reconstruction': records, 'exact_oracle_agreement': True,
            'prefix_reconstruction': prefix_records, 'prefix_rows': prefix_rows,
            'rows': rows, 'baseline_rows': baselines, 'gates': original_gates,
            'advance': contrasts['advance'], 'allocation_comparisons': contrasts,
            'allocation': {'stages': metadata['allocation_checks'], 'boundary_checks': boundary_checks,
                           'stage1_seconds': spec.stage1_seconds, 'total_seconds': spec.total_seconds,
                           'compute_matched': False, 'historical_updates_independently_replayed': False},
            'structural_work': summary['structural_work'], 'architecture_claim': False, 'latent_identification_claim': False,
            'limitations': [
                'Original process, source and payload admission belongs to the external caller.',
                'Exact targets and reference posterior use known dynamics; learned models receive only public histories and supervised labels.',
                'Every allocation has the same global eligibility window, not equal accepted updates, FLOPs or actual wall time.',
                'All arms use prefix-only then joint training, with independently timed updates. Matched initial probabilities do not imply matched gradients or accepted update counts.',
                'All attempted updates, including late discarded work, are counted. Shared generation, final summaries and complete fit time remain explicit.',
                'All model and optimizer boundary states are independently decoded and hashed; intermediate hashes, update losses, historical timing and source-qualified execution are not replayed.',
                'Initial balanced and matched-free transition probabilities are reconstructed with 64 NumPy log-domain sweeps; later normalization residual checks are not replayed.',
                'Equal stored parameter counts and initial functions do not establish equal degrees of freedom, gradients, accepted updates or compute.',
                'Batch schedules, public-label target arithmetic and structural forward counts are reconstructed without any model or optimizer call.',
                'Observed filtering receives intervening observations and remains separate from blind extrapolation.',
                'All three absolute criteria and every paired comparative requirement remain binding; no average rescues a failed condition.',
                'This synthetic balanced-transition diagnostic establishes neither latent identification, native transfer, statistical significance nor architectural novelty.']}
