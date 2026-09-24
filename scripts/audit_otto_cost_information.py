"""Independent saved-record cost-information audit, without model/native calls.

Rebuild probabilities, integer draws, fixed-horizon scalar decisions and paired
bootstrap arithmetic. Teacher costs remain authenticated saved outputs: no
teacher function or checkpoint is decoded or re-evaluated by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import otto_cost_information_common as c

VERSION = 'otto-cost-information-saved-audit-v1'
SCALARS = ('conditional_chosen_cost', 'conditional_blind_cost', 'conditional_oracle_cost',
           'information_advantage', 'approximation_regret', 'total_regret',
           'conditional_oracle_cost_variance', 'conditional_chosen_regret_variance')
VECTORS = ('conditional_mean_costs', 'conditional_cost_variances')
REGIMES = ('lambda3', 'lambda4')


def compare_report(saved, independent, path='report'):
    """Compare all independent fields; omit producer-specific roundoff metadata."""
    if isinstance(independent, dict):
        c.require(type(saved) is dict and set(independent) <= set(saved), path + ': fields')
        for key, value in independent.items():
            compare_report(saved[key], value, path + '.' + key)
    elif isinstance(independent, list):
        c.require(type(saved) is list and len(saved) == len(independent), path + ': complete list')
        for i, (left, right) in enumerate(zip(saved, independent, strict=True)):
            compare_report(left, right, f'{path}[{i}]')
    elif type(independent) is float:
        c.require(type(saved) in (int, float) and math.isfinite(saved) and math.isfinite(independent)
                  and abs(saved - independent) <= 1e-10 * (1 + abs(independent)), path + ': scalar agreement')
    else:
        c.require(type(saved) is type(independent) and saved == independent, path + ': exact identity')


def validate_predictions(data, predictions, np):
    wanted = {f'{family}__{seed}__cost' for family in c.FAMILIES for seed in c.FIT_SEEDS}
    c.require(type(predictions) is dict and set(predictions) == wanted, 'exact twelve frozen forecasts')
    for value in predictions.values():
        c.require(isinstance(value, np.ndarray) and value.dtype == np.float32
                  and value.shape == (len(data['case_ids']), 8, 4) and np.isfinite(value).all(), 'finite frozen forecasts')


def _mean(values):
    c.require(len(values) > 0, 'nonempty averaging denominator')
    result = math.fsum(float(v) / len(values) for v in values)
    c.require(math.isfinite(result), 'finite average')
    return result


def _path(position, actions):
    current, positions = list(map(int, position)), []
    for value in actions:
        action = int(value)
        c.require(0 <= action < 4, 'four movement actions')
        current[action // 2] += 2 * (action % 2) - 1
        c.require(all(0 < p < 52 for p in current), 'interior path supports all four decisions')
        positions.append(tuple(current))
    return positions


def exact_survival(data, np, horizon):
    result = []
    for i, root in enumerate(data['root_grid']):
        visited = {x * 53 + y for x, y in _path(data['prefix_position'][i], data['actions'][i, :horizon])}
        result.append(math.fsum(float(value) for j, value in enumerate(root) if j not in visited))
    return np.asarray(result, np.float64)


def _stats(weights, costs):
    mass = math.fsum(map(float, weights))
    c.require(math.isfinite(mass) and 0 <= mass <= 1 + 1e-12, 'fixed-horizon subprobability mass')
    rows = [list(map(float, row)) for row in costs]
    c.require(all(len(row) == 4 and all(math.isfinite(v) and v >= 0 for v in row) for row in rows), 'finite four teacher costs')
    c.require(all(math.isfinite(float(w)) and w >= 0 for w in weights), 'nonnegative branch mass')
    if not mass:
        return {'mass': 0., 'means': None, 'variances': None, 'oracle': None, 'oracle_variance': None,
                'regrets': None, 'regret_variances': None, 'blind': None}
    p = [float(w) / mass for w in weights]
    columns = list(zip(*rows, strict=True))
    means = [math.fsum(prob * value for prob, value in zip(p, column, strict=True)) for column in columns]
    minima = [min(row) for row in rows]
    oracle = math.fsum(prob * value for prob, value in zip(p, minima, strict=True))
    gaps = [[row[a] - best for row, best in zip(rows, minima, strict=True)] for a in range(4)]
    regrets = [math.fsum(prob * value for prob, value in zip(p, column, strict=True)) for column in gaps]

    def variance(column, average):
        value = math.fsum(prob * ((x - average) ** 2) for prob, x in zip(p, column, strict=True))
        c.require(math.isfinite(value) and value >= 0, 'finite branch population variance')
        return value

    return {'mass': mass, 'means': means,
            'variances': [variance(column, average) for column, average in zip(columns, means, strict=True)],
            'oracle': oracle, 'oracle_variance': variance(minima, oracle), 'regrets': regrets,
            'regret_variances': [variance(column, average) for column, average in zip(gaps, regrets, strict=True)],
            'blind': min(range(4), key=lambda a: (means[a], a))}


def decomposition(data, stats, actions, horizon):
    rows = []
    for i, state in enumerate(stats):
        action = int(actions[i])
        weights = data['exact_weights'][i] if horizon == 4 else data['mc_alive8'][i, 1].astype(float) / c.CONFIG['mc_draws']
        row = {'case_id': str(data['case_ids'][i]), 'horizon': horizon, 'chosen_action': action,
               'legal': [True] * 4, 'branches': len(weights), 'positive_weight_branches': sum(v > 0 for v in map(float, weights)),
               'support_mass': state['mass'], 'defined': state['mass'] > 0}
        if not row['defined']:
            row.update({k: None for k in (*SCALARS, *VECTORS)})
            row['blind_optimal_action'] = None
        else:
            information = min(state['regrets'])
            row.update(conditional_mean_costs=state['means'], conditional_cost_variances=state['variances'],
                conditional_chosen_cost=state['means'][action], conditional_blind_cost=state['means'][state['blind']],
                conditional_oracle_cost=state['oracle'], blind_optimal_action=state['blind'],
                information_advantage=information, approximation_regret=state['regrets'][action] - information,
                total_regret=state['regrets'][action], conditional_oracle_cost_variance=state['oracle_variance'],
                conditional_chosen_regret_variance=state['regret_variances'][action])
        rows.append(row)
    alive = [row for row in rows if row['defined']]
    aggregate = {'declared_cases': len(rows), 'supported_cases': len(alive),
        'unsupported_case_ids': [row['case_id'] for row in rows if not row['defined']],
        'support_mass_sum': math.fsum(row['support_mass'] for row in rows),
        'mean_support_mass': _mean([row['support_mass'] for row in rows])}
    for key in SCALARS:
        aggregate['case_weighted_' + key] = _mean([row[key] for row in alive]) if alive else None
    for key in VECTORS:
        aggregate['case_weighted_' + key] = [_mean([row[key][a] for row in alive]) for a in range(4)] if alive else None
    return {'version': 'otto-conditional-cost-metrics-v1', 'horizon': horizon, 'action_count': 4,
            'cases': rows, 'aggregate': aggregate, 'admits_execution': False}


def primary_gate(data, predictions, np, *, check=lambda: None):
    """Independent hierarchical bootstrap, conditioning on fixed selection draws."""
    validate_predictions(data, predictions, np)
    n, m, reps = len(data['case_ids']), c.CONFIG['mc_draws'], c.CONFIG['bootstrap_replicates']
    selected, control, case_rows = [], [], []
    survival = exact_survival(data, np, 8)
    for i in range(n):
        check()
        selection = data['mc_costs8'][i, 0]
        reference = min(range(4), key=lambda a: (math.fsum(map(float, selection[:, a])), a))
        actions = [min(range(4), key=lambda a, seed=seed: (float(predictions[f"{c.CONFIG['primary_family']}__{seed}__cost"][i, 7, a]), a))
                   for seed in c.FIT_SEEDS]
        evaluation = data['mc_costs8'][i, 1]
        sample_reference, sample_control = [], []
        for row in evaluation:
            best = min(map(float, row))
            sample_reference.append(float(row[reference]) - best)
            sample_control.append(_mean([float(row[a]) - best for a in actions]))
        selected.append(sample_reference)
        control.append(sample_control)
        case_rows.append({'case_id': str(data['case_ids'][i]), 'regime': str(data['regimes'][i]),
            'reference_action': reference, 'control_actions': actions,
            'control_gap': _mean(sample_control), 'reference_gap': _mean(sample_reference),
            'gain': _mean([left - right for left, right in zip(sample_control, sample_reference, strict=True)]),
            'exact_survival': float(survival[i]),
            'select_survivors': int(data['mc_alive8'][i, 0].sum()), 'eval_survivors': int(data['mc_alive8'][i, 1].sum())})
    control, selected = np.asarray(control, np.float64), np.asarray(selected, np.float64)
    gains, groups = control - selected, []
    for regime_index, regime in enumerate(REGIMES):
        check()
        indices = [i for i, label in enumerate(data['regimes']) if label == regime]
        count = len(indices)
        c.require(count > 0, 'both regime denominators retained')
        random = np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.CONFIG['bootstrap_seed'], regime_index])))
        case_draws = random.integers(0, count, size=(reps, count), dtype=np.int64)
        inner_draws = random.integers(0, m, size=(reps, count, m), dtype=np.int64)
        fraction = c.CONFIG['base_headroom_fraction'] if regime_index == 0 else c.CONFIG['shift_headroom_fraction']
        records = []
        local_control, local_gains = control[indices], gains[indices]
        for r in range(reps):
            check()
            chosen_cases, chosen_draws = case_draws[r, :, None], inner_draws[r]
            baseline = float(local_control[chosen_cases, chosen_draws].mean())
            gain = float(local_gains[chosen_cases, chosen_draws].mean())
            records.append([baseline, gain, gain - fraction * baseline])
        # Explicit linear order-statistic interpolation, not producer quantile.
        intervals = []
        for column in range(3):
            ordered = sorted(row[column] for row in records)
            endpoints = []
            for quantile in (c.CONFIG['quantile'], 1 - c.CONFIG['quantile']):
                rank = (reps - 1) * quantile
                lo, hi = math.floor(rank), math.ceil(rank)
                endpoints.append(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))
            intervals.append(endpoints)
        baseline, gain = _mean([row['control_gap'] for row in case_rows if row['regime'] == regime]), _mean([row['gain'] for row in case_rows if row['regime'] == regime])
        conditions = {'minimum_cases': count >= c.CONFIG['min_cases'],
            'mc_support': all(row['exact_survival'] == 0 or min(row['select_survivors'], row['eval_survivors']) >= c.CONFIG['min_mc_survivors']
                              for row in case_rows if row['regime'] == regime),
            'positive_control_gap': baseline > 0, 'resolved_gain': intervals[2][0] > 0}
        groups.append({'regime': regime, 'cases': count, 'control_gap': baseline,
            'reference_gap': _mean([row['reference_gap'] for row in case_rows if row['regime'] == regime]),
            'gain': gain, 'required_fraction': fraction, 'gain_minus_required': gain - fraction * baseline,
            'interval_columns': ['control_gap', 'gain', 'gain_minus_required'],
            'approximate_95_percent_interval': intervals, 'bootstrap_replicates': records,
            'bootstrap_index_sha256': hashlib.sha256(case_draws.tobytes() + inner_draws.tobytes()).hexdigest(),
            'conditions': conditions, 'passed': all(conditions.values())})
    passed = all(row['passed'] for row in groups)
    return {'version': 'otto-cost-information-headroom-v1', 'passed': passed,
        'status': 'HEADROOM_RESOLVED' if passed else 'HEADROOM_NOT_RESOLVED', 'groups': groups, 'cases': case_rows,
        'scope': 'Approximate hierarchical percentile intervals, conditional on realized selection draws and fixed models; not a rigorous population bound.',
        'selection': '128 separate draws choose a full-public-history reference action; 128 independent draws evaluate it.',
        'aggregation': 'Mean of three frozen policies as separate decisions; equal originating prefixes including found draws as zero.',
        'admits_model_training': False, 'architecture_claim': False}


def analyze(data, predictions, np, *, check=lambda: None):
    """Pure independent scalar report API; no probability provenance assumptions."""
    validate_predictions(data, predictions, np)
    rows, decompositions = [], []
    for horizon in (4, 8):
        check()
        weights = data['exact_weights'] if horizon == 4 else data['mc_alive8'][:, 1].astype(np.float64) / c.CONFIG['mc_draws']
        costs = data['exact_costs'] if horizon == 4 else data['mc_costs8'][:, 1]
        stats = []
        for w, q in zip(weights, costs, strict=True):
            check()
            stats.append(_stats(w, q))
        for family in c.FAMILIES:
            for seed in c.FIT_SEEDS:
                check()
                actions = predictions[f'{family}__{seed}__cost'][:, horizon - 1].argmin(-1)
                record = decomposition(data, stats, actions, horizon)
                decompositions.append({'family': family, 'fit_seed': seed, 'horizon': horizon, 'report': record})
                for regime in REGIMES:
                    subset = [row for i, row in enumerate(record['cases']) if data['regimes'][i] == regime]
                    rows.append({'family': family, 'fit_seed': seed, 'regime': regime, 'horizon': horizon, 'cases': len(subset),
                        'measure': 'exact_grid' if horizon == 4 else 'evaluation_MC_plugin_descriptive',
                        **{key: _mean([row['support_mass'] * row[key] if row['defined'] else 0. for row in subset])
                           for key in ('total_regret', 'information_advantage', 'approximation_regret')}})
    validation = []
    for i, identity in enumerate(data['case_ids']):
        check()
        exact = [math.fsum(float(weight) * (float(q[a]) - min(map(float, q)))
                           for weight, q in zip(data['exact_weights'][i], data['exact_costs'][i], strict=True)) for a in range(4)]
        for stream in range(2):
            sample = [[float(q[a]) - min(map(float, q)) for q in data['mc_costs4'][i, stream]] for a in range(4)]
            means = [_mean(column) for column in sample]
            size = c.CONFIG['mc_draws']
            se = [math.sqrt(math.fsum((v - average) ** 2 for v in column) / (size - 1) / size)
                  for column, average in zip(sample, means, strict=True)]
            validation.append({'case_id': str(identity), 'regime': str(data['regimes'][i]), 'stream': stream,
                'exact_unconditional_regrets': exact, 'sample_unconditional_regrets': means,
                'error': [left - right for left, right in zip(means, exact, strict=True)],
                'estimated_standard_error': se,
                'scope': 'Sampling diagnostic only; estimated standard errors are not bounded guarantees.'})
    return {'version': c.VERSION, 'rows': rows, 'decompositions': decompositions,
            'h4_sampling_validation': validation, 'gate': primary_gate(data, predictions, np, check=check),
            'original_action_effect_gate_unchanged': 'DEV_FAIL 6/18', 'new_training': False}


def validate_data(data, np):
    n, m = len(data['case_ids']), c.CONFIG['mc_draws']
    shapes = {'prefix': ((n, 9, 31), np.float32), 'prefix_lengths': ((n,), np.int64),
        'prefix_actions': ((n, 8), np.int64), 'prefix_outcomes': ((n, 8), np.int64),
        'prefix_position': ((n, 2), np.int64), 'actions': ((n, 8), np.int64),
        **{k: ((n, 2809), np.float64) for k in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root')},
        'exact_weights': ((n, 256), np.float64), 'exact_costs': ((n, 256, 4), np.float32),
        'exact_alive': ((n, 256), np.bool_), 'mc_draws': ((n, 2, m, 9), np.uint64),
        'mc_source_indices': ((n, 2, m), np.int64), 'mc_outcomes': ((n, 2, m, 8), np.int64),
        **{f'mc_alive{h}': ((n, 2, m), np.bool_) for h in (4, 8)},
        **{f'mc_costs{h}': ((n, 2, m, 4), np.float32) for h in (4, 8)}}
    c.require(type(data) is dict and set(data) == set(shapes) | {'case_ids', 'regimes'}, 'exact diagnostic array keys')
    c.require(n > 0, 'nonempty retained prefixes')
    for name, (shape, dtype) in shapes.items():
        value = data[name]
        c.require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                  and np.isfinite(value).all(), 'finite complete array ' + name)
    for key in ('case_ids', 'regimes'):
        c.require(isinstance(data[key], np.ndarray) and data[key].shape == (n,) and data[key].dtype.kind == 'U', 'Unicode case identities')
    c.require(len(set(data['case_ids'])) == n and all(str(v).strip() for v in data['case_ids'])
              and set(data['regimes']) <= set(REGIMES), 'unique nonempty case IDs and known regimes')
    c.require((data['prefix_lengths'] == 9).all(), 'complete nine-row prefix')
    for name in ('prefix_actions', 'prefix_outcomes', 'actions'):
        c.require(((data[name] >= 0) & (data[name] <= 3)).all(), 'surviving four-category public history')
    c.require(((data['prefix_position'] >= 18) & (data['prefix_position'] <= 34)).all(), 'eight moves from center geometry')
    for name in ('initial_belief', 'root_strict', 'root_grid', 'legacy_root'):
        c.require(((data[name] >= 0) & (data[name] <= 1)).all(), 'public belief interval')
        if name != 'legacy_root':
            c.require((np.abs(data[name].sum(-1) - 1) <= 1e-12).all(), 'unit public mass ' + name)
    c.require((data['exact_weights'] >= 0).all() and (data['exact_weights'].sum(-1) <= 1 + 1e-12).all(), 'exact subprobability')
    c.require(np.array_equal(data['exact_alive'], data['exact_weights'] > 0), 'positive exact leaf support')
    c.require((data['exact_costs'] >= 0).all() and (data['exact_costs'][~data['exact_alive']] == 0).all(), 'exact terminal placeholders')
    c.require((data['mc_draws'] < 2**53).all(), 'allocated 53-bit integer uniforms')
    c.require(((data['mc_source_indices'] >= 0) & (data['mc_source_indices'] < 2809)).all(), 'hypothetical source cells')
    c.require(((data['mc_outcomes'] >= 0) & (data['mc_outcomes'] <= 4)).all(), 'five MC outcomes')
    found = data['mc_outcomes'] == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, axis=-1)), 'absorbing MC found suffix')
    for horizon in (4, 8):
        alive, costs = data[f'mc_alive{horizon}'], data[f'mc_costs{horizon}']
        c.require(np.array_equal(alive, ~found[..., horizon - 1]) and (costs >= 0).all()
                  and (costs[~alive] == 0).all(), 'finite nonnegative costs with found zero')
    return n


def _cdf_law(raw, np):
    """Independent scalar sequential CDF, exactly the declared finite grid."""
    values, cumulative, running = list(map(float, raw)), [], 0.
    c.require(all(math.isfinite(v) and v >= 0 for v in values)
              and abs(math.fsum(values) - 1) <= 1e-10, 'normalized raw categorical law')
    for value in values:
        running += value
        cumulative.append(running)
    c.require(math.isfinite(running) and running > 0, 'positive finite cumulative mass')
    result, previous = [], 0
    for value in cumulative:
        boundary = math.ceil((value / running) * 2**53)
        result.append((boundary - previous) / 2**53)
        previous = boundary
    c.require(previous == 2**53 and all(v >= 0 for v in result), 'complete integer law')
    return np.asarray(result, np.float64)


def _integer_cdf(law):
    result, total = [], 0
    for value in law:
        scaled = float(value) * 2**53
        c.require(math.isfinite(scaled) and 0 <= scaled <= 2**53 and scaled == int(scaled), 'exact finite-grid probability bins')
        total += int(scaled)
        result.append(total)
    c.require(total == 2**53, 'integer categorical bins sum to 2**53')
    return result


def verify_probabilities(data, tables, roster, np, *, check=lambda: None):
    """Reconstruct strict/legacy prefix, exact masses and saved MC draw histories."""
    from bisect import bisect_right

    n = validate_data(data, np)
    c.require(set(tables) == {name for regime in REGIMES for name in (regime, regime + '_raw', 'legacy_' + regime)}, 'six saved sensor/kernel arrays')
    for regime in REGIMES:
        effective, raw, kernel = tables[regime], tables[regime + '_raw'], tables['legacy_' + regime]
        for value in (effective, raw):
            c.require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (105, 105, 4)
                      and np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), 'finite sensor table')
            c.require((value[52, 52] == 0).all(), 'zero sensor origin')
        c.require(isinstance(kernel, np.ndarray) and kernel.dtype == np.float64 and kernel.shape == (4, 107, 107)
                  and np.isfinite(kernel).all() and ((kernel >= 0) & (kernel <= 1)).all()
                  and (kernel[:, 53, 53] == 0).all(), 'unchanged PublicBeliefView legacy kernel contract')
        for x in range(105):
            check()
            for y in range(105):
                if x != 52 or y != 52:
                    c.require(np.array_equal(effective[x, y], _cdf_law(raw[x, y], np)), 'independent scalar sensor CDF conversion')
    by = {row['id']: row for row in roster}
    c.require(len(by) == len(roster), 'unique registered identities')
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    counts = {'prefixes': n, 'prefix_updates': 8 * n, 'exact_leaves': 256 * n,
              'exact_positive_leaves': 0, 'replayed_uniform_integers': 2 * c.CONFIG['mc_draws'] * 9 * n,
              'source_draws': 0, 'odor_draws': 0, 'h4_lookups': 0, 'teacher_costs_recomputed': False,
              'native_calls': 0, 'raw_sensor_functions_replayed': False}
    per_case = []
    for i, identity in enumerate(data['case_ids']):
        check()
        c.require(identity in by and by[identity]['regime'] == data['regimes'][i], 'registered retained case identity')
        entry, regime = by[identity], str(data['regimes'][i])
        kernel = tables['legacy_' + regime]

        def likelihood(position, regime=regime):
            offsets = coordinates - np.asarray(position) + 52
            return tables[regime][offsets[:, 0], offsets[:, 1]]

        def legacy_update(state, position, odor, kernel=kernel):
            value = state.reshape(53, 53).copy()
            value[tuple(position)] = 0.
            x, y = 53 - position[0], 53 - position[1]
            value *= kernel[odor, x:x + 53, y:y + 53]
            mass = float(value.sum(dtype=np.float64))
            if mass > 1e-10:
                value /= mass
            c.require(np.isfinite(value).all() and (value >= 0).all(), 'finite unmodified legacy filter')
            return value.reshape(-1)

        initial = np.full(2809, 1. / 2808, np.float64)
        initial[26 * 53 + 26] = 0.
        legacy = legacy_update(initial, (26, 26), entry['initial_hit'])
        initial = _cdf_law(legacy, np)
        initial /= initial.sum(dtype=np.float64)
        c.require(np.allclose(initial, data['initial_belief'][i], rtol=1e-12, atol=0.), 'reconstructed initial public prior')
        strict = initial.copy()
        prefix_positions = _path((26, 26), data['prefix_actions'][i])
        for position, odor in zip(prefix_positions, data['prefix_outcomes'][i], strict=True):
            check()
            index = position[0] * 53 + position[1]
            law = likelihood(position)[:, int(odor)]
            previous = strict.copy()
            previous[index] = 0.
            joint = previous * law
            c.require(not ((previous > 0) & (law > 0) & (joint == 0)).any(), 'no positive shadow product underflow')
            mass = float(joint.sum(dtype=np.float64))
            c.require(mass > 0, 'positive actual-prefix evidence')
            strict = joint / mass
            legacy = legacy_update(legacy, position, int(odor))
        c.require(tuple(data['prefix_position'][i]) == prefix_positions[-1], 'center-derived prefix position')
        for value, name in ((strict, 'root_strict'), (legacy, 'legacy_root')):
            c.require(np.allclose(value, data[name][i], rtol=1e-12, atol=0.), 'independent public prefix ' + name)
        grid = _cdf_law(strict, np)
        c.require(np.array_equal(grid, data['root_grid'][i]), 'exact declared root-grid law')
        tv = .5 * float(np.abs(strict - grid).sum(dtype=np.float64))
        c.require(tv <= c.CONFIG['root_grid_max_tv'], 'declared root-grid TV bound')
        actions = np.random.Generator(np.random.PCG64(np.random.SeedSequence([entry['seed'], 911]))).integers(0, 4, size=8, dtype=np.int64)
        c.require(np.array_equal(actions, data['actions'][i]), 'fixed public action seed replay')
        positions = _path(prefix_positions[-1], actions)
        indices = [x * 53 + y for x, y in positions]
        laws = [likelihood(position) for position in positions]
        joint = grid[None].copy()
        found_mass, internal, children, positive_found = [], 0, 0, 0
        for depth in range(4):
            check()
            internal += int(np.count_nonzero(joint.sum(-1) > 0))
            values = joint[:, indices[depth]]
            found_mass.extend(map(float, values))
            positive_found += int(np.count_nonzero(values > 0))
            joint[:, indices[depth]] = 0.
            product = joint[:, None, :] * laws[depth].T[None, :, :]
            c.require(not ((joint[:, None, :] > 0) & (laws[depth].T[None, :, :] > 0) & (product == 0)).any(), 'no exact-tree support underflow')
            joint = product.reshape(-1, 2809)
            children += int(np.count_nonzero(joint.sum(-1) > 0))
        weights = joint.sum(-1, dtype=np.float64)
        c.require(np.allclose(weights, data['exact_weights'][i], rtol=1e-12, atol=0.)
                  and np.array_equal(weights > 0, data['exact_alive'][i]), 'independent dense canonical exact tree')
        c.require(abs(math.fsum(found_mass) + math.fsum(map(float, weights)) - 1.) <= 1e-12, 'exact tree mass conservation')
        counts['exact_positive_leaves'] += int(np.count_nonzero(weights))
        root_cdf = _integer_cdf(grid)
        stream_work = []
        for stream, seed_name in enumerate(('select_seed', 'eval_seed')):
            check()
            expected_draws = (np.random.PCG64(entry[seed_name]).random_raw(c.CONFIG['mc_draws'] * 9) >> np.uint64(11)).reshape(-1, 9)
            c.require(np.array_equal(expected_draws, data['mc_draws'][i, stream]), 'complete independently replayed MC stream')
            odor_draws, lookups, alive8 = 0, 0, 0
            for sample, draws in enumerate(expected_draws):
                check()
                source = bisect_right(root_cdf, int(draws[0]))
                c.require(source == int(data['mc_source_indices'][i, stream, sample]), 'integer-bin source identity')
                outcomes, code = [4] * 8, 0
                for h, index in enumerate(indices):
                    if source == index:
                        break
                    odor = bisect_right(_integer_cdf(laws[h][source]), int(draws[h + 1]))
                    c.require(0 <= odor < 4, 'integer-bin odor')
                    outcomes[h] = odor
                    odor_draws += 1
                    if h < 4:
                        code = code * 4 + odor
                    if h == 3:
                        c.require(weights[code] > 0 and data['mc_costs4'][i, stream, sample].tobytes() == data['exact_costs'][i, code].tobytes(), 'bitwise MC horizon-four exact-cost lookup')
                        lookups += 1
                c.require(outcomes == data['mc_outcomes'][i, stream, sample].tolist(), 'complete absorbing MC history')
                alive8 += outcomes[7] != 4
            work = {'allocated_draw_integers': c.CONFIG['mc_draws'] * 9, 'source_draws': c.CONFIG['mc_draws'],
                    'odor_draws': odor_draws, 'unused_odor_draws': 8 * c.CONFIG['mc_draws'] - odor_draws,
                    'legacy_updates': odor_draws, 'h4_cost_lookups': lookups, 'teacher_calls': alive8}
            stream_work.append(work)
            counts['source_draws'] += c.CONFIG['mc_draws']
            counts['odor_draws'] += odor_draws
            counts['h4_lookups'] += lookups
        per_case.append({'case_id': str(identity), 'root_grid_total_variation': tv,
            'exact': {'survival_mass': math.fsum(map(float, weights)), 'found_mass': math.fsum(found_mass),
                      'work': {'internal_nodes': internal, 'positive_children': children, 'legacy_updates': children,
                               'teacher_calls': int(np.count_nonzero(weights)), 'found_positive_branches': positive_found}},
            'streams': stream_work})
    return {'counts': counts, 'cases': per_case,
            'scope': 'Initial/strict/legacy public prefixes, finite-grid laws, exact weights, saved PCG64 streams and lookup identities; no teacher re-evaluation.'}


def lines(path, check):
    import gzip
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            check()
            c.require(line.endswith('\n'), 'complete durable JSON line')
            yield json.loads(line)


def verify_forward_records(events, data, np, *, check=lambda: None):
    """Join recorded float32 forward values to every saved teacher Q vector."""
    wanted = []
    for i, identity in enumerate(data['case_ids']):
        for code in range(256):
            if data['exact_alive'][i, code]:
                history = [(code // (4**power)) % 4 for power in (3, 2, 1, 0)]
                wanted.append((str(identity), 'exact', 'exact', 4, history, None, data['exact_costs'][i, code]))
        for stream, label in enumerate(('select', 'evaluate')):
            for sample in range(c.CONFIG['mc_draws']):
                if data['mc_alive8'][i, stream, sample]:
                    wanted.append((str(identity), label, 'mc', 8, data['mc_outcomes'][i, stream, sample].tolist(),
                                   sample, data['mc_costs8'][i, stream, sample]))
    iterator = iter(events)
    maximum_error = 0.
    for ordinal, (identity, stream, mode, horizon, history, sample, saved) in enumerate(wanted, 1):
        check()
        event = next(iterator, None)
        c.require(event is not None and event['ordinal'] == ordinal and event['input_shape'] == [16, 105, 105]
                  and event['symmetry_average'] is True, 'complete ordered physical forward witness')
        context = event['context']
        c.require(context['id'] == identity and context['phase'] == 'counterfactual'
                  and context['stream'] == stream and context['mode'] == mode
                  and context['horizon'] == horizon and context['history'] == history
                  and (sample is None and 'sample' not in context or context.get('sample') == sample),
                  'exact leaf-to-forward context identity')
        masses = np.asarray(event['branch_masses'], np.float32)
        values = np.asarray(event['values'], np.float32)
        c.require(masses.shape == (4, 4) and values.shape == (16,) and np.isfinite(masses).all()
                  and (masses >= 0).all() and np.isfinite(values).all(), 'finite recorded original float32 forward')
        products = masses * values.reshape(4, 4)
        reconstructed = np.float32(1.) + products.sum(-1, dtype=np.float32)
        # TensorFlow and NumPy may associate four products differently. Bound
        # only that float32 reduction; no additional teacher/model evaluation.
        bound = 8 * np.finfo(np.float32).eps * (1 + np.abs(products.astype(np.float64)).sum(-1))
        error = np.abs(saved.astype(np.float64) - reconstructed.astype(np.float64))
        c.require(np.isfinite(reconstructed).all() and (error <= bound).all(), 'saved Q agrees with recorded forward reduction')
        maximum_error = max(maximum_error, float(error.max()))
    c.require(next(iterator, None) is None, 'no extra recorded teacher forwards')
    return {'forward_q_reductions_checked': len(wanted), 'maximum_float32_reduction_difference': maximum_error,
            'float32_reduction_error_factor': 8, 'teacher_recomputed': False}


def verify_work(events, calls, *, check=lambda: None):
    stack, sequence, counts = [], 0, {name: 0 for name in calls}
    teacher_forwards = {}
    total = 0
    for event in events:
        check()
        total += 1
        c.require(event['channel'] in calls, 'declared original work channel')
        if event['event'] == 'attempt':
            sequence += 1
            c.require(event['id'] == sequence, 'monotone physical operation starts')
            if event['channel'] == 'tensorflow_value':
                c.require(stack and stack[-1]['channel'] == 'teacher_score', 'physical forward belongs to teacher annotation')
                parent = stack[-1]['id']
                teacher_forwards[parent] = teacher_forwards.get(parent, 0) + 1
            stack.append(event)
        else:
            c.require(event['event'] == 'return' and stack, 'closed nested work event')
            previous = stack.pop()
            c.require(all(event[k] == previous[k] for k in ('id', 'channel', 'context')), 'matching physical operation return')
            elapsed = event['seconds_including_nested_io']
            c.require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed >= 0, 'finite recorded operation time')
            if event['channel'] == 'teacher_score':
                c.require(teacher_forwards.pop(event['id'], 0) == 1, 'one physical forward per teacher score')
            counts[event['channel']] += 1
    c.require(not stack and not teacher_forwards and total == 2 * sequence, 'all work operations returned')
    c.require(all(counts[k] == value['attempted'] == value['returned'] for k, value in calls.items()), 'work journal reconciles all call counts')
    return {'work_events_checked': total, 'work_operations_checked': sequence}


def verify_case_records(data, rows, summary, probability, np):
    roster = c.roster()
    c.require(len(rows) == len(roster) and [row['identity'] for row in rows] == roster, 'all attempted prefixes retained in original order')
    retained = [row for row in rows if row['excluded'] is None]
    c.require([row['identity']['id'] for row in retained] == data['case_ids'].tolist(), 'complete retained originating-case roster')
    fields = set(data) - {'case_ids', 'regimes'}
    exact_calls = mc_calls = tree_updates = mc_updates = 0
    for i, (row, prob) in enumerate(zip(retained, probability['cases'], strict=True)):
        c.require(row['native_steps'] == 8 and set(row['array_sha256']) == fields, 'all saved case arrays have identity hashes')
        for name in fields:
            c.require(hashlib.sha256(data[name][i].tobytes()).hexdigest() == row['array_sha256'][name], 'per-case array identity ' + name)
        compare_report(row['exact'], prob['exact'], 'exact tree work and mass')
        compare_report(row['root_grid']['total_variation'], prob['root_grid_total_variation'], 'root quantization distance')
        c.require(len(row['mc']) == 2, 'both independent streams retained')
        for stream, label in enumerate(('select', 'evaluate')):
            saved = row['mc'][stream]
            c.require(saved['stream'] == label and saved['seed'] == row['identity']['select_seed' if stream == 0 else 'eval_seed']
                      and saved['alive4'] == int(data['mc_alive4'][i, stream].sum())
                      and saved['alive8'] == int(data['mc_alive8'][i, stream].sum()), 'complete stream support')
            compare_report(saved['work'], prob['streams'][stream], 'independent MC work')
        exact_calls += prob['exact']['work']['teacher_calls']
        tree_updates += prob['exact']['work']['legacy_updates']
        mc_calls += sum(work['teacher_calls'] for work in prob['streams'])
        mc_updates += sum(work['legacy_updates'] for work in prob['streams'])
    excluded = [row for row in rows if row['excluded'] is not None]
    c.require(all(row['excluded'] == 'found_during_observed_prefix' and type(row['native_steps']) is int
                  and 1 <= row['native_steps'] <= 8 for row in excluded), 'prefix exclusions without replacement')
    counts = {regime: data['regimes'].tolist().count(regime) for regime in REGIMES}
    c.require(summary['counts'] == counts and all(value >= c.CONFIG['min_cases'] for value in counts.values())
              and summary['cases'] == len(rows) and summary['retained'] == len(retained)
              and summary['prefix_exclusions'] == len(excluded) and summary['replacement_cases'] == 0
              and summary['parent_data_array_decodes'] == summary['learner_calls'] == summary['native_continuation_steps'] == 0,
              'complete fresh cohort, no parent data or native continuation')
    native_steps = sum(row['native_steps'] for row in rows)
    expected = {'tensorflow_construction': 1, 'tensorflow_build': 1, 'tensorflow_load': 1,
        'tensorflow_value': exact_calls + mc_calls, 'teacher_score': exact_calls + mc_calls,
        'native_reset': len(rows), 'native_step': native_steps, 'actor_construction': len(rows), 'actor_update': native_steps,
        'analytic_score': native_steps + len(retained), 'feature_build': native_steps + len(retained),
        'backend_binding': len(retained), 'branch_view_construction': len(retained), 'sampler_table': 2,
        'sampler_lookup': native_steps + 8 * len(retained), 'initial_source_law': len(rows),
        'initial_normalization': len(rows), 'shadow_update': native_steps,
        'sampler_draw_validation': len(rows) + native_steps - len(excluded), 'committed_positions': len(retained),
        'root_grid_law': len(retained), 'exact_tree': len(retained), 'mc_draw_allocation': 2 * len(retained),
        'mc_rollout': 2 * len(retained), 'legacy_tree_update': tree_updates, 'legacy_mc_update': mc_updates}
    c.require(set(summary['calls']) == set(expected), 'exact original operation channels')
    for name, value in summary['calls'].items():
        c.require(type(value['attempted']) is int and type(value['returned']) is int
                  and value['attempted'] == value['returned'] == expected[name]
                  and type(value['seconds']) in (float, int) and math.isfinite(value['seconds']) and value['seconds'] >= 0,
                  'counted original operation ' + name)
    return {'attempted_prefixes_checked': len(rows), 'retained_prefixes_checked': len(retained),
            'exact_teacher_records': exact_calls, 'mc_teacher_records': mc_calls}


def admit(run):
    collection, prediction = c.OUT / 'collection-01', c.OUT / 'prediction-01'
    receipts, previous = {}, None
    for phase, directory, terminal_path in (
            ('collect', collection, c.OUT / 'collection-native-01.terminal.json'),
            ('predict', prediction, c.OUT / 'prediction-native-01.terminal.json')):
        receipt, terminal = c.closed(directory, terminal_path), c.read(terminal_path)
        c.require(receipt['version'] == c.VERSION and receipt['phase'] == phase
                  and receipt['plan_sha256'] == run.args.plan_sha256
                  and receipt['old_test_decodes'] == receipt['astra_calls'] == 0,
                  'same registered predecessor with no forbidden calls')
        c.require(previous is None or previous <= terminal['started_ns'], 'collection closes before prediction')
        previous = terminal['finished_ns']
        receipts[phase] = receipt
    c.require(previous <= run.launch['started_ns'], 'prediction closes before original audit')
    scientific = {'started.json', 'parent-reference.json', 'runtime.json', 'setup.json', 'sensor-laws.npz',
        'sensor-laws.json', 'cases.npz', 'cases.jsonl', 'commitments.jsonl', 'summary.json',
        'work.jsonl.gz', 'weights.jsonl.gz', 'forwards.jsonl.gz', 'native-draws.jsonl.gz',
        'shadow.jsonl.gz', 'source-laws.jsonl.gz', 'transitions.jsonl.gz'}
    c.require(set(receipts['collect']['files']) == scientific
              and set(receipts['predict']['files']) == {'started.json', 'predictions.npz', 'report.json', 'summary.json'},
              'exact closed scientific payload rosters')
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'mlx')), 'no model framework in independent audit')
    reference = c.parent_reference()
    c.require(c.read(collection / 'parent-reference.json') == reference
              and run.plan['parent_reference'] == reference, 'unchanged opaque parent-checkpoint lineage')
    return collection, prediction, receipts, reference


def run_audit(run):
    collection, prediction, receipts, reference = admit(run)
    import numpy as np

    counts = {'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0, 'native_calls': 0,
              'teacher_calls': 0, 'optimizer_calls': 0, 'parent_data_decodes': 0, 'old_test_decodes': 0}
    run.receipt['audit_counts'] = counts

    def arrays(directory, name, phase):
        run.check()
        c.require(c.desc(directory / name) == receipts[phase]['files'][name], 'unchanged authenticated array bytes')
        run.receipt['pending'] = {'decode': name}
        with np.load(directory / name, allow_pickle=False) as source:
            result = {key: source[key] for key in source.files}
        counts['array_decodes'] += 1
        run.receipt['pending'] = None
        return result

    data = arrays(collection, 'cases.npz', 'collect')
    predictions = arrays(prediction, 'predictions.npz', 'predict')
    tables = arrays(collection, 'sensor-laws.npz', 'collect')
    n = validate_data(data, np)
    validate_predictions(data, predictions, np)
    summary, saved = c.read(prediction / 'summary.json'), c.read(prediction / 'report.json')
    acquisition = c.read(collection / 'summary.json')
    expected_calls = {'data_array_decodes': 1, 'checkpoint_decodes': 12, 'model_constructions': 12,
                      'blind_rollouts': 12 * n, 'training_updates': 0,
                      'bootstrap_replicates': 2 * c.CONFIG['bootstrap_replicates']}
    c.require(summary['calls'] == receipts['predict']['calls'] == expected_calls
              and summary['parent_reference'] == acquisition['parent_reference'] == reference
              and summary['collection_receipt'] == c.desc(collection / 'receipt.json')
              and summary['data'] == receipts['collect']['files']['cases.npz'], 'complete frozen prediction lineage and calls')
    timing = summary['prediction_times']
    expected_order = [(f, s) for f in c.FAMILIES for s in c.FIT_SEEDS]
    c.require(len(timing) == 12 and [(r['family'], r['fit_seed']) for r in timing] == expected_order, 'all ordered frozen model costs')
    for row in timing:
        c.require(row['cases'] == n and type(row['seconds']) in (float, int) and math.isfinite(row['seconds']) and row['seconds'] >= 0
                  and type(row['work']) is dict and all(type(v) is int and v >= 0 for v in row['work'].values()), 'finite recorded frozen inference work')
    probability = verify_probabilities(data, tables, run.plan['roster'], np, check=run.check)
    counts.update(verify_case_records(data, list(lines(collection / 'cases.jsonl', run.check)), acquisition, probability, np))
    c.require(acquisition['calls'] == receipts['collect']['calls'] and summary['cases'] == acquisition['counts'], 'original collection call records')
    counts.update(verify_work(lines(collection / 'work.jsonl.gz', run.check), acquisition['calls'], check=run.check))
    counts.update(verify_forward_records(lines(collection / 'forwards.jsonl.gz', run.check), data, np, check=run.check))
    c.require(counts['forward_q_reductions_checked'] == counts['exact_teacher_records'] + counts['mc_teacher_records'], 'all saved teacher scores linked to original forwards')
    independent = analyze(data, predictions, np, check=run.check)
    c.require(set(saved) == set(independent) and len(saved['rows']) == 48 and len(saved['decompositions']) == 24
              and len(saved['h4_sampling_validation']) == 2 * n, 'complete saved scalar panels')
    compare_report(saved, independent)
    c.require(summary['status'] == independent['gate']['status'], 'unchanged prespecified diagnostic outcome')
    for phase, directory in (('collect', collection), ('predict', prediction)):
        for name, descriptor in receipts[phase]['files'].items():
            run.check()
            c.require(c.desc(directory / name) == descriptor, 'audit inputs unchanged after checks')
    counts.update(reports_checked=24, summary_rows_checked=48, bootstrap_replicates_checked=2 * c.CONFIG['bootstrap_replicates'])
    c.require(counts['array_decodes'] == 3, 'only cases, predictions and sensor/kernel arrays decoded')
    result = {'version': VERSION, 'agreement': True, 'technical_complete': False,
        'requires_original_supervisor_closure': True, 'counts': counts, 'gate': independent['gate'],
        'report': independent, 'probability_reconstruction': probability,
        'inputs': {'collection_receipt': c.desc(collection / 'receipt.json'), 'prediction_receipt': c.desc(prediction / 'receipt.json')},
        'limitations': ['Teacher Q is checked against recorded float32 masses/forward values; no model or teacher is replayed.',
            'Native sensor scalar functions and checkpoint contents are authenticated evidence, not recomputed.',
            'Meaningful conditional/unconditional scalars and variances are independently checked; producer-specific roundoff record formatting is not duplicated.',
            'MC plug-in decompositions are descriptive; selection-stream actions are evaluated only on independent evaluation draws.',
            'Bootstrap intervals condition on fixed selection streams and policies; no training or architecture claim is admitted.']}
    run.check()
    c.write(run.out / 'audit.json', result)
    run.receipt['agreement'] = True


class Audit(c.Run):
    def body(self):
        run_audit(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'qualified numerical audit interpreter')
    run = Audit(args, 'audit')
    try:
        run.body()
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
