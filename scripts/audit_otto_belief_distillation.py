"""Independently score saved action-gap predictions after original process closure.

Only DEV labels, sensor tables and saved predictions are decoded. Scalar proper scores, legal
action choices, case aggregation and target fingerprints are reconstructed here,
without calling the producer scorer, learner, model, simulator or any solver.
The fixed comparison function may consume these independently rebuilt reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import otto_belief_distillation_common as c

VERSION = 'otto-belief-distillation-saved-audit-v1'
FAMILIES = ('recurrent_soft', 'recurrent_sampled', 'action_blind_soft', 'direct_soft')
MODEL_KINDS = dict(zip(FAMILIES, ('action_recurrent', 'action_recurrent', 'action_blind', 'direct_horizon'), strict=True))
CONDITIONS = ('gap', 'normal')
HORIZONS = tuple(range(1, 9))
GROUPS = {'all': HORIZONS, 'short': HORIZONS[:4], 'long': HORIZONS[4:]}
DATA_FIELDS = {'prefix', 'prefix_lengths', 'actions', 'continuation', 'outcomes',
               'raw_costs', 'legal', 'case_ids', 'regimes', 'initial_belief', 'prefix_actions',
               'prefix_outcomes', 'prefix_position', 'gap_oracle', 'normal_oracle', 'opposite_oracle'}


def close_equal(actual, expected, name='saved report'):
    if isinstance(expected, dict):
        c.require(type(actual) is dict and set(actual) == set(expected), name + ': exact fields')
        for key in expected:
            close_equal(actual[key], expected[key], name + '.' + key)
    elif isinstance(expected, list):
        c.require(type(actual) is list and len(actual) == len(expected), name + ': complete list')
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close_equal(left, right, f'{name}[{i}]')
    elif type(expected) is float:
        c.require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                  and abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected), name + ': finite scalar agreement')
    else:
        c.require(type(actual) is type(expected) and actual == expected, name + ': exact identity')


def scalar_report(np, dataset, outcome_predictions, cost_predictions, *, family, fit_seed, condition, check=lambda: None):
    """Independent scalar arithmetic on supplied fabricated or admitted arrays."""
    target, raw, legal = (dataset[k] for k in ('outcomes', 'raw_costs', 'legal'))
    c.require(isinstance(target, np.ndarray) and target.ndim == 2 and target.shape[1] == 8
              and len(target) > 0 and target.dtype.kind in 'iu', 'integer eight-step outcomes')
    n = len(target)
    c.require(bool(((target >= 0) & (target <= 4)).all()), 'five known target classes')
    c.require(isinstance(legal, np.ndarray) and legal.shape == (n, 8, 4) and legal.dtype == np.bool_, 'legal Boolean mask')
    for array, shape, label in ((raw, (n, 8, 4), 'teacher costs'),
                                (cost_predictions, (n, 8, 4), 'cost predictions'),
                                (outcome_predictions, (n, 8, 5), 'outcome predictions')):
        c.require(isinstance(array, np.ndarray) and array.shape == shape and array.dtype.kind in 'fiu'
                  and bool(np.isfinite(array).all()), 'finite complete ' + label)
    ids, regimes = list(dataset['case_ids']), list(dataset['regimes'])
    c.require(len(ids) == len(regimes) == n and all(isinstance(v, str) and v.strip() for v in ids + regimes),
              'complete case and regime strings')
    ids, regimes = list(map(str, ids)), list(map(str, regimes))
    c.require(all(len({regimes[i] for i in range(n) if ids[i] == case}) == 1 for case in set(ids)), 'one regime per originating case')
    c.require(family in FAMILIES and type(fit_seed) is int and condition in CONDITIONS, 'declared view identity')
    logs, briers, gaps, actions = [], [], [], []
    for i in range(n):
        check()
        log_row, brier_row, gap_row, action_row = [], [], [], []
        found = False
        for t in range(8):
            check()
            label = int(target[i, t])
            c.require(not found or label == 4, 'absorbing target suffix')
            found = label == 4
            allowed = [a for a in range(4) if bool(legal[i, t, a])]
            c.require(bool(allowed) is (not found), 'nonterminal-only legal decisions')
            values = [float(v) for v in outcome_predictions[i, t]]
            c.require(all(0 <= v <= 1 for v in values), 'probability interval')
            total = math.fsum(values)
            c.require(abs(total - 1) <= 1e-12, 'unit probability row')
            probabilities = [v / total for v in values]
            c.require(probabilities[label] > 0, 'observed zero probability has infinite logarithmic loss')
            log_score = -math.log(probabilities[label])
            brier = math.fsum((p - int(a == label)) ** 2 for a, p in enumerate(probabilities))
            c.require(math.isfinite(log_score) and log_score >= 0 and math.isfinite(brier), 'finite proper scores')
            if allowed:
                # Tuple ordering makes exact legal ties choose the smallest ID.
                chosen = min(allowed, key=lambda a: (float(cost_predictions[i, t, a]), a))
                gap = float(raw[i, t, chosen]) - min(float(raw[i, t, a]) for a in allowed)
                c.require(math.isfinite(gap) and gap >= 0, 'finite independent decision gap')
            else:
                chosen, gap = -1, None
            log_row.append(log_score)
            brier_row.append(brier)
            gap_row.append(gap)
            action_row.append(chosen)
        logs.append(log_row)
        briers.append(brier_row)
        gaps.append(gap_row)
        actions.append(action_row)

    def mean(values):
        result = math.fsum(v / len(values) for v in values)
        c.require(math.isfinite(result), 'finite aggregate arithmetic')
        return result

    def summary(indices, horizons):
        records = []
        for case in sorted({ids[i] for i in indices}):
            check()
            members = [i for i in indices if ids[i] == case]
            pairs = [(i, h - 1) for i in members for h in horizons]
            supported = [(i, t) for i, t in pairs if gaps[i][t] is not None]
            records.append({'case_id': case, 'blocks': len(members), 'outcome_rows': len(pairs),
                'decision_rows': len(supported), 'terminal_rows': len(pairs) - len(supported),
                'log_score': mean([logs[i][t] for i, t in pairs]),
                'brier': mean([briers[i][t] for i, t in pairs]),
                'decision_gap': mean([gaps[i][t] for i, t in supported]) if supported else None})
        supported = [r['decision_gap'] for r in records if r['decision_gap'] is not None]
        return {'horizons': list(horizons), 'declared_cases': len(records), 'blocks': len(indices),
            'outcome_rows': sum(r['outcome_rows'] for r in records),
            'terminal_rows': sum(r['terminal_rows'] for r in records),
            'decision_rows': sum(r['decision_rows'] for r in records), 'supported_cases': len(supported),
            'unsupported_case_ids': [r['case_id'] for r in records if r['decision_gap'] is None],
            'case_weighted_log_score': mean([r['log_score'] for r in records]),
            'case_weighted_brier': mean([r['brier'] for r in records]),
            'case_weighted_decision_gap': mean(supported) if supported else None,
            'full_case_denominator_gap': mean([r['decision_gap'] if r['decision_gap'] is not None else 0. for r in records]) if supported else None,
            'cases': records}

    def groups(horizons):
        return {'overall': summary(list(range(n)), horizons),
                'by_regime': {r: summary([i for i in range(n) if regimes[i] == r], horizons) for r in sorted(set(regimes))}}

    digest = hashlib.sha256(b'otto-action-latent-targets-v1\0')
    digest.update(target.astype('<i8').tobytes())
    digest.update(legal.tobytes())
    digest.update(raw.astype('<f8').tobytes())
    report = {'version': 'otto-action-latent-metrics-v1', 'family': family, 'fit_seed': fit_seed,
        'condition': condition, 'prediction_kind': 'probabilities',
        'horizons': list(HORIZONS), 'target_sha256': digest.hexdigest(),
        'identity_manifest': [{'block_index': i, 'case_id': ids[i], 'regime': regimes[i]} for i in range(n)],
        'per_horizon': {str(h): groups((h,)) for h in HORIZONS},
        'groups': {name: groups(horizons) for name, horizons in GROUPS.items()}, 'chosen_actions': actions,
        'scope': 'proper five-outcome prediction and teacher-action imitation on fixed precommitted blocks'}
    report['oracle'] = oracle_report(np, dataset, outcome_predictions, condition=condition, check=check)
    return report


DIAGNOSTICS = ('kl', 'brier_excess', 'expected_log_score', 'oracle_entropy',
               'expected_brier', 'oracle_brier', 'sampled_log_score', 'sampled_brier')


def probabilities(np, value, shape, name):
    c.require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == np.float64
              and bool(np.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all()),
              'complete finite float64 probabilities: ' + name)
    # Canonical row normalization is part of the producer fingerprint contract.
    totals = value.sum(axis=-1, keepdims=True)
    c.require(bool((np.abs(totals - 1.) <= 1e-12).all()), 'unit probability mass: ' + name)
    return value / totals


def diagnostic_groups(rows, mask, ids, regimes, horizons, *, check=lambda: None):
    def mean(values):
        if not values:
            return None
        value = math.fsum(float(x) / len(values) for x in values)
        c.require(math.isfinite(value), 'finite independent diagnostic aggregate')
        return value

    def summary(indices):
        records = []
        for case in sorted({ids[i] for i in indices}):
            check()
            members = [i for i in indices if ids[i] == case]
            selected = [(i, h - 1) for i in members for h in horizons if mask[i][h - 1]]
            records.append({'case_id': case, 'blocks': len(members), 'rows': len(selected),
                **{name: mean([values[i][t] for i, t in selected]) for name, values in rows.items()}})
        supported = [row for row in records if row['rows']]
        return {'horizons': list(horizons), 'declared_cases': len(records), 'blocks': len(indices),
            'available_rows': len(indices) * len(horizons), 'scored_rows': sum(row['rows'] for row in records),
            'supported_cases': len(supported), 'unsupported_case_ids': [row['case_id'] for row in records if not row['rows']],
            **{'case_weighted_' + name: mean([row[name] for row in supported]) for name in rows}, 'cases': records}
    return {'overall': summary(list(range(len(ids)))),
            'by_regime': {r: summary([i for i, regime in enumerate(regimes) if regime == r]) for r in sorted(set(regimes))}}


def oracle_report(np, dataset, predictions, *, condition, check=lambda: None):
    labels = dataset['outcomes']
    n, horizon = labels.shape
    q = probabilities(np, dataset[condition + '_oracle'], (n, horizon, 5), 'oracle')
    p = probabilities(np, predictions, q.shape, 'predictions')
    ids, regimes = list(map(str, dataset['case_ids'])), list(map(str, dataset['regimes']))
    resolved = np.zeros((n, horizon), np.bool_)
    canonical_logs = np.zeros_like(p)
    np.log(p, out=canonical_logs, where=p > 0)
    rows = {name: [[] for _ in range(n)] for name in DIAGNOSTICS}
    roundoff = 0
    for i in range(n):
        check()
        observed_found = False
        for t in range(horizon):
            check()
            label = int(labels[i, t])
            c.require(q[i, t, label] > 0, 'observed label has positive oracle mass')
            resolved[i, t] = condition == 'normal' and observed_found
            if resolved[i, t]:
                c.require(list(q[i, t]) == list(p[i, t]) == [0., 0., 0., 0., 1.],
                          'normal known-terminal rows are exact caller-supplied one-hot found')
            observed_found |= label == 4
            positive = [a for a in range(5) if q[i, t, a] > 0]
            c.require(all(p[i, t, a] > 0 for a in positive), 'finite expected log score requires oracle support')
            logp = list(map(float, canonical_logs[i, t]))
            entropy = -math.fsum(float(q[i, t, a]) * math.log(float(q[i, t, a])) for a in positive)
            expected_log = -math.fsum(float(q[i, t, a]) * logp[a] for a in positive)
            kl = math.fsum(float(q[i, t, a]) * (math.log(float(q[i, t, a])) - logp[a]) for a in positive)
            c.require(math.isfinite(kl) and kl >= -1e-12, 'nonnegative finite independent KL')
            roundoff += int(kl < 0)
            excess = math.fsum((float(p[i, t, a]) - float(q[i, t, a])) ** 2 for a in range(5))
            oracle_brier = math.fsum(float(q[i, t, a]) * (1 - float(q[i, t, a])) for a in range(5))
            values = {'kl': max(kl, 0.), 'brier_excess': excess, 'expected_log_score': expected_log,
                'oracle_entropy': entropy, 'expected_brier': excess + oracle_brier, 'oracle_brier': oracle_brier,
                'sampled_log_score': -logp[label],
                'sampled_brier': math.fsum((float(p[i, t, a]) - int(a == label)) ** 2 for a in range(5))}
            c.require(all(math.isfinite(v) and v >= -1e-12 for v in values.values()), 'finite independent proper scores')
            for name, value in values.items():
                rows[name][i].append(value)
    all_rows = np.ones((n, horizon), np.bool_)

    def panel(horizons):
        return {'all_rows': diagnostic_groups(rows, all_rows, ids, regimes, horizons, check=check),
                'unresolved_rows': diagnostic_groups(rows, ~resolved, ids, regimes, horizons, check=check)}

    return {'version': 'otto-belief-distillation-metrics-v1',
        'target_sha256': hashlib.sha256(b'oracle-five-outcome-v1\0' + q.astype('<f8').tobytes()).hexdigest(),
        'unresolved_mask_sha256': hashlib.sha256((~resolved).tobytes()).hexdigest(),
        'resolved_rows': int(resolved.sum()), 'unresolved_rows': int((~resolved).sum()),
        'kl_roundoff_corrections': roundoff, 'kl_roundoff_tolerance': 1e-12,
        'per_horizon': {str(h): panel((h,)) for h in HORIZONS},
        'groups': {name: panel(horizons) for name, horizons in GROUPS.items()},
        'terminal_policy': 'gap has no found feedback; normal shortcut only after observed found',
        'supplementary_only': True}


def sensitivity_report(np, dataset, original_predictions, opposite_predictions, *, check=lambda: None):
    shape = dataset['gap_oracle'].shape
    p = probabilities(np, dataset['gap_oracle'], shape, 'gap oracle')
    alt_p = probabilities(np, dataset['opposite_oracle'], shape, 'opposite oracle')
    q = probabilities(np, original_predictions, shape, 'gap model')
    alt_q = probabilities(np, opposite_predictions, shape, 'opposite model')
    n, horizon, _ = shape
    ids, regimes = list(map(str, dataset['case_ids'])), list(map(str, dataset['regimes']))
    rows = {k: [[] for _ in range(n)] for k in ('jensen_shannon', 'total_variation', 'oracle_signal', 'model_effect_error')}
    for i in range(n):
        check()
        for t in range(horizon):
            check()
            terms = []
            for a in range(5):
                left, right = float(p[i, t, a]), float(alt_p[i, t, a])
                high, low = max(left, right), min(left, right)
                if high:
                    log_midpoint = math.log(high) + math.log1p(low / high) - math.log(2)
                    terms.extend(v * (math.log(v) - log_midpoint) for v in (left, right) if v > 0)
            signal = [float(p[i, t, a]) - float(alt_p[i, t, a]) for a in range(5)]
            error = [(float(q[i, t, a]) - float(alt_q[i, t, a])) - signal[a] for a in range(5)]
            values = {'jensen_shannon': max(0., .5 * math.fsum(terms)),
                'total_variation': .5 * math.fsum(abs(v) for v in signal),
                'oracle_signal': math.fsum(v * v for v in signal),
                'model_effect_error': math.fsum(v * v for v in error)}
            for name, value in values.items():
                rows[name][i].append(value)
    mask = [[True] * horizon for _ in range(n)]
    return {'version': 'otto-belief-action-sensitivity-v1', 'horizons': list(range(1, horizon + 1)),
        'overall': diagnostic_groups(rows, mask, ids, regimes, tuple(range(1, horizon + 1)), check=check),
        'per_horizon': {str(h): diagnostic_groups(rows, mask, ids, regimes, (h,), check=check) for h in range(1, horizon + 1)},
        'blocks': n, 'case_filtering': False, 'actions_verified_opposite': True,
        'condition': 'gap', 'action_mapping': 'a XOR1', 'model_effect_included': True,
        'scope': 'supplied paired oracle sensitivity; neither causal oracle validity nor model effectiveness is established'}


def validate_dataset(np, dataset):
    c.require(set(dataset) == DATA_FIELDS, 'exact complete DEV data fields')
    ids = dataset['case_ids']
    c.require(isinstance(ids, np.ndarray) and ids.ndim == 1 and len(ids) > 0
              and ids.dtype.kind in 'US', 'complete nonempty case identity vector')
    n = len(ids)
    specs = {'prefix': ((n, 9, 31), np.float32), 'prefix_lengths': ((n,), np.int64),
             'actions': ((n, 8), np.int64), 'continuation': ((n, 8, 31), np.float32),
             'outcomes': ((n, 8), np.int64), 'raw_costs': ((n, 8, 4), np.float32),
             'legal': ((n, 8, 4), np.bool_), 'initial_belief': ((n, 2809), np.float64),
             'prefix_actions': ((n, 8), np.int64), 'prefix_outcomes': ((n, 8), np.int64),
             'prefix_position': ((n, 2), np.int64),
             **{k: ((n, 8, 5), np.float64) for k in ('gap_oracle', 'normal_oracle', 'opposite_oracle')}}
    for name, (shape, dtype) in specs.items():
        value = dataset[name]
        c.require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype
                  and bool(np.isfinite(value).all()), 'complete finite dataset array ' + name)
    c.require(dataset['regimes'].shape == (n,) and dataset['regimes'].dtype.kind in 'US'
              and bool((dataset['prefix_lengths'] == 9).all()), 'complete regime and prefix geometry')
    target = dataset['outcomes']
    c.require(bool(((target >= 0) & (target <= 4)).all()), 'five target classes')
    found = target == 4
    c.require(np.array_equal(found, np.maximum.accumulate(found, axis=1))
              and np.array_equal(dataset['legal'].any(axis=-1), ~found), 'absorbing labels and decision support')
    return n


def verify_oracles(np, dataset, tables, *, check=lambda: None):
    """Independent public Bayes arithmetic, with no native or producer calls.

    The authenticated initial prior and raw scalar kernel are supplied evidence.
    This verifies their saved finite-grid law and the subsequent public filter;
    it does not rerun native scalar sensor code or reconstruct the initial prior.
    """
    c.require(set(tables) == {'lambda3', 'lambda4', 'lambda3_raw', 'lambda4_raw'}, 'exact saved sensor-law tables')
    for regime in ('lambda3', 'lambda4'):
        table, raw = tables[regime], tables[regime + '_raw']
        for value in (table, raw):
            c.require(isinstance(value, np.ndarray) and value.dtype == np.float64 and value.shape == (105, 105, 4)
                      and bool(np.isfinite(value).all()) and bool(((value >= 0) & (value <= 1)).all()),
                      'complete finite raw and effective sensor laws')
        c.require(bool((raw[52, 52] == 0).all()) and bool((table[52, 52] == 0).all()), 'zero origin has no odor draw')
        for x in range(105):
            check()
            for y in range(105):
                if x == y == 52:
                    continue
                values = list(map(float, raw[x, y]))
                c.require(abs(math.fsum(values) - 1.) <= 1e-10, 'normalized raw sensor law')
                cumulative, running = [], 0.
                for value in values:
                    running += value
                    cumulative.append(running)
                bins = [math.ceil((value / running) * 2**53) for value in cumulative]
                prior = 0
                for a, boundary in enumerate(bins):
                    c.require(table[x, y, a] == (boundary - prior) / 2**53, 'exact independently counted 53-bit CDF law')
                    prior = boundary
    labels, actions = dataset['outcomes'], dataset['actions']
    n = len(labels)
    for name, shape in (('actions', (n, 8)), ('prefix_actions', (n, 8)), ('prefix_outcomes', (n, 8)), ('prefix_position', (n, 2))):
        value = dataset[name]
        c.require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.shape == shape,
                  'complete int64 public history: ' + name)
        maximum = 52 if name == 'prefix_position' else 3
        c.require(bool(((value >= 0) & (value <= maximum)).all()), 'admissible surviving public history: ' + name)
    initial = dataset['initial_belief']
    c.require(isinstance(initial, np.ndarray) and initial.dtype == np.float64 and initial.shape == (n, 2809)
              and bool(np.isfinite(initial).all()) and bool(((initial >= 0) & (initial <= 1)).all())
              and bool((np.abs(initial.sum(axis=1) - 1.) <= 1e-12).all()), 'normalized saved initial public priors')
    for name in ('gap_oracle', 'normal_oracle', 'opposite_oracle'):
        probabilities(np, dataset[name], (n, 8, 5), name)
    coordinates = np.indices((53, 53)).reshape(2, -1).T
    rebuilt = {name: np.empty((n, 8, 5), np.float64) for name in ('gap_oracle', 'normal_oracle', 'opposite_oracle')}

    def move(position, action):
        result = list(position)
        result[int(action) // 2] += 1 if int(action) % 2 else -1
        c.require(all(0 <= v < 53 for v in result), 'entire committed path stays in bounds')
        return result

    def lookup(table, position):
        displacement = coordinates - np.asarray(position, np.int64)
        return table[displacement[:, 0] + 52, displacement[:, 1] + 52]

    def posterior(prior, likelihood, position, label):
        if label == 4:
            c.require(prior[position] > 0, 'positive observed found evidence')
            result = np.zeros_like(prior)
            result[position] = 1.
            return result
        mass = prior.copy()
        mass[position] = 0.
        joint = mass * likelihood[:, label]
        c.require(not bool(((mass > 0) & (likelihood[:, label] > 0) & (joint == 0)).any()), 'no positive evidence underflow')
        evidence = float(joint.sum())
        c.require(evidence > 0 and math.isfinite(evidence), 'positive public observation evidence')
        return joint / evidence

    for i in range(n):
        check()
        regime = str(dataset['regimes'][i])
        c.require(regime in ('lambda3', 'lambda4'), 'registered oracle regime')
        table, belief, position = tables[regime], initial[i].copy(), [26, 26]
        for action, label in zip(dataset['prefix_actions'][i], dataset['prefix_outcomes'][i], strict=True):
            check()
            position = move(position, action)
            likelihood = lookup(table, position)
            belief = posterior(belief, likelihood, position[0] * 53 + position[1], int(label))
        c.require(position == dataset['prefix_position'][i].tolist(), 'prefix motion reaches saved public position')
        prefix_position, prefix_belief = list(position), belief.copy()
        for name, block in (('gap_oracle', actions[i]), ('opposite_oracle', actions[i] ^ 1)):
            position, visited = list(prefix_position), set()
            for t, action in enumerate(block):
                check()
                position = move(position, action)
                visited.add(position[0] * 53 + position[1])
                mass = prefix_belief.copy()
                mass[list(visited)] = 0.
                likelihood = lookup(table, position)
                rebuilt[name][i, t, :4] = np.sum(mass[:, None] * likelihood, axis=0)
                rebuilt[name][i, t, 4] = math.fsum(float(prefix_belief[j]) for j in sorted(visited))
        position, belief, found = list(prefix_position), prefix_belief.copy(), False
        for t, action in enumerate(actions[i]):
            check()
            position = move(position, action)
            if found:
                c.require(int(labels[i, t]) == 4, 'absorbing public found suffix')
                rebuilt['normal_oracle'][i, t] = [0., 0., 0., 0., 1.]
                continue
            likelihood = lookup(table, position)
            index = position[0] * 53 + position[1]
            mass = belief.copy()
            mass[index] = 0.
            rebuilt['normal_oracle'][i, t, :4] = np.sum(mass[:, None] * likelihood, axis=0)
            rebuilt['normal_oracle'][i, t, 4] = belief[index]
            label = int(labels[i, t])
            c.require(0 <= label <= 4, 'five-class public observation')
            belief = posterior(belief, likelihood, index, label)
            found = label == 4
    for name, values in rebuilt.items():
        c.require(bool(np.isfinite(values).all()) and bool(np.allclose(values, dataset[name], rtol=1e-10, atol=1e-12)),
                  'independent public Bayes agrees with saved ' + name)
    return {'cases': n, 'prefix_updates': n * 8, 'oracle_rows': n * 8 * 3,
            'cdf_rows': 2 * (105 * 105 - 1), 'native_calls': 0,
            'initial_prior_reconstructed': False, 'raw_sensor_functions_replayed': False,
            'opposite_mapping': 'a XOR1', 'posterior_floor': None}


def thresholds():
    cfg = c.CONFIG
    return {'long_log_relative_gain': cfg['long_logscore_improvement'],
        'long_gap_relative_gain': cfg['long_gap_improvement'],
        'normal_log_relative_tolerance': cfg['normal_relative_tolerance'],
        'normal_gap_relative_tolerance': cfg['normal_relative_tolerance'],
        'minimum_supported_cases': cfg['minimum_supported_cases']}


def verify_training(np, directory, receipt, collection_summary, summary, *, check=lambda: None):
    """Check recorded fit counts and deterministic orders, without replaying fits."""
    normalization = c.read(directory / 'normalization.json')
    normalization_fields = {'cost_scale', 'rms_squared_before_floor', 'variance_floor', 'source',
                            'teacher_units_divisor', 'dev_decodes'}
    c.require(set(normalization) == normalization_fields
              and normalization['source'] == 'TRAIN only, all-four centered, equal cases and surviving rows'
              and normalization['variance_floor'] == 1e-6 and normalization['teacher_units_divisor'] == 64.
              and type(normalization['dev_decodes']) is int and normalization['dev_decodes'] == 0,
              'fixed TRAIN-only normalization metadata')
    variance, scale = normalization['rms_squared_before_floor'], normalization['cost_scale']
    c.require(type(variance) is float and math.isfinite(variance) and variance >= 0
              and type(scale) is float and math.isfinite(scale) and scale > 0,
              'finite recorded variance and positive float32 scale')
    with np.errstate(over='raise', invalid='raise'):
        try:
            expected_scale = float(np.float32(np.sqrt(max(variance, 1e-6))))
        except FloatingPointError as error:
            raise ValueError('finite float32 normalization arithmetic') from error
    c.require(scale == expected_scale, 'scale matches recorded variance and fixed floor; variance is not recomputed')
    n = collection_summary['counts']['train']['lambda3']
    cfg = c.CONFIG
    c.require(type(n) is int and n >= cfg['min_train'] and collection_summary['counts']['train']['lambda4'] == 0,
              'recorded surviving TRAIN cohort')
    epochs, batch = cfg['epochs'], cfg['batch']
    expected = []
    neural = FAMILIES
    for i, seed in enumerate(cfg['fit_seeds']):
        expected.extend((family, seed) for family in neural[i:] + neural[:i])
    with (directory / 'fits.jsonl').open() as stream:
        fits = [json.loads(line) for line in stream]
    c.require(len(fits) == 12 and [(r['family'], r['seed']) for r in fits] == expected
              and summary['fits'] == fits, 'all twelve ordered fit records')
    per_fit_updates = math.ceil(n / batch) * epochs
    for row in fits:
        check()
        c.require(row['epochs'] == epochs and row['updates'] == per_fit_updates and row['cases'] == n
                  and row['exposures'] == n * epochs and row['training_horizons'] == [1, 2, 3, 4]
                  and row['evaluation_decodes_so_far'] == 0, 'fixed complete TRAIN-only fit counts')
        c.require(row['model_kind'] == MODEL_KINDS[row['family']]
                  and row['soft_targets'] is (row['family'] != 'recurrent_sampled')
                  and row['blind_input_keys'] == ['prefix', 'prefix_lengths', 'actions']
                  and row['privileged_targets_only'] == ['raw_costs', 'gap_oracle', 'normal_oracle'],
                  'fixed architecture, target arm and privileged-input separation metadata')
        for field in ('seconds', 'last_batch_loss', 'last_gradient_norm'):
            c.require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] >= 0,
                      'finite recorded fit scalar')
        checkpoint = f"{row['family']}-{row['seed']}.npz"
        c.require(row['checkpoint'] == receipt['files'][checkpoint], 'recorded checkpoint identity')
        parameters = row['parameters']
        count = {'action_recurrent': 8299, 'action_blind': 8299, 'direct_horizon': 8107}[MODEL_KINDS[row['family']]]
        c.require(parameters['kind'] == MODEL_KINDS[row['family']] and parameters['hidden_dim'] == 28
                  and parameters['count'] == parameters['trainable_count'] == count
                  and parameters['cost_scale'] == scale and 'cost_scale' not in parameters['parameters']
                  and sum(v['count'] for v in parameters['parameters'].values()) == count,
                  'declared architecture parameter count and shared fixed normalization')
        c.require(isinstance(row['changed_tensors'], list) and row['changed_tensors']
                  and len(set(row['changed_tensors'])) == len(row['changed_tensors'])
                  and set(row['changed_tensors']) <= set(parameters['parameters']), 'changed tensor names retained')
    with (directory / 'training-orders.jsonl').open() as stream:
        for family, seed in expected:
            generator = np.random.Generator(np.random.PCG64(seed))
            for epoch in range(epochs):
                check()
                line = stream.readline()
                c.require(line.endswith('\n'), 'complete deterministic epoch record')
                row = json.loads(line)
                wanted = {'family': family, 'seed': seed, 'epoch': epoch, 'indices': generator.permutation(n).tolist()}
                c.require(row == wanted, 'same complete deterministic originating-case permutation')
        c.require(stream.read() == '', 'no extra training orders')
    calls = {'optimizer_steps': 12 * per_fit_updates, 'fit_count': 12, 'dev_array_decodes': 1}
    c.require(summary['calls'] == receipt['calls'] == calls and summary['data_cases']['train'] == n,
              'recorded complete fit and evaluation counts')
    return {'fits_checked': 12, 'normalization_records_checked': 1,
            'training_epochs_checked': 12 * epochs,
            'optimizer_steps_declared': calls['optimizer_steps'], 'training_case_exposures_declared': 12 * n * epochs}


def admit(run):
    """Authenticate both original closures before numerical imports or decoding."""
    collection, fitted = c.OUT / 'collection-01', c.OUT / 'fit-01'
    receipts = {}
    previous = None
    for phase, directory, terminal_name in (('collect', collection, 'collection-native-01.terminal.json'),
                                             ('fit', fitted, 'fit-native-01.terminal.json')):
        terminal_path = c.OUT / terminal_name
        receipt = c.closed(directory, terminal_path)
        terminal = c.read(terminal_path)
        c.require(receipt['version'] == c.VERSION and receipt['phase'] == phase
                  and receipt['plan_sha256'] == run.args.plan_sha256
                  and receipt['old_test_decodes'] == receipt['astra_calls'] == 0,
                  'same registered predecessor phase with no prohibited calls')
        c.require(previous is None or previous <= terminal['started_ns'], 'collection closes before fitting starts')
        previous = terminal['finished_ns']
        receipts[phase] = receipt
    c.require(previous <= run.launch['started_ns'], 'fit closes before independent audit starts')
    expected_fit = {'started.json', 'fits.jsonl', 'training-orders.jsonl', 'normalization.json',
                    'predictions.npz', 'reports.json', 'summary.json'} | {
                        f'{family}-{seed}.npz' for family in FAMILIES for seed in c.CONFIG['fit_seeds']}
    c.require({'dev.npz', 'sensor-laws.npz', 'summary.json'} <= set(receipts['collect']['files'])
              and set(receipts['fit']['files']) == expected_fit, 'exact closed fit evidence payloads')
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'mlx')), 'no model runtime in independent audit')
    return collection, fitted, receipts


def run_audit(run):
    collection, fitted, receipts = admit(run)
    import numpy as np

    from openjev.research import otto_belief_distillation_metrics as metrics

    counts = {'array_decodes': 0, 'views_checked': 0, 'model_calls': 0, 'optimizer_calls': 0,
              'solver_calls': 0, 'teacher_calls': 0, 'native_calls': 0, 'old_test_decodes': 0,
              'sensitivity_views_checked': 0}
    run.receipt['audit_counts'] = counts
    collection_summary = c.read(collection / 'summary.json')
    fit_summary = c.read(fitted / 'summary.json')
    counts.update(verify_training(np, fitted, receipts['fit'], collection_summary, fit_summary, check=run.check))
    c.require(fit_summary['collection_receipt'] == c.desc(collection / 'receipt.json')
              and fit_summary['train_data'] == receipts['collect']['files']['train.npz']
              and fit_summary['dev_data'] == receipts['collect']['files']['dev.npz'], 'fit binds original collection inputs')

    def arrays(directory, name, phase):
        run.check()
        c.require(c.desc(directory / name) == receipts[phase]['files'][name], 'unchanged predecode payload')
        run.receipt['pending'] = {'decode': name}
        with np.load(directory / name, allow_pickle=False) as archive:
            result = {k: archive[k] for k in archive.files}
        counts['array_decodes'] += 1
        run.receipt['pending'] = None
        return result

    dataset = arrays(collection, 'dev.npz', 'collect')
    predictions = arrays(fitted, 'predictions.npz', 'fit')
    tables = arrays(collection, 'sensor-laws.npz', 'collect')
    validate_dataset(np, dataset)
    seeds = c.CONFIG['fit_seeds']
    specs = [(family, seed, condition) for family in FAMILIES for seed in seeds for condition in CONDITIONS]
    expected_keys = {f'{a}__{s}__{condition}__{field}' for a, s, condition in specs for field in ('outcome', 'cost')}
    expected_keys |= {f'{a}__{s}__opposite__outcome' for a in FAMILIES for s in seeds}
    c.require(set(predictions) == expected_keys and len(expected_keys) == 60, 'exact 24-view saved prediction roster')
    identities = [str(v) for v in dataset['case_ids']]
    regimes = [str(v) for v in dataset['regimes']]
    roster = {r['id']: r for r in run.plan['roster'] if r['split'] == 'dev'}
    c.require(len(identities) == len(set(identities)) == len(regimes)
              and all(i in roster and roster[i]['regime'] == r for i, r in zip(identities, regimes, strict=True)),
              'only registered originating DEV cases')
    for regime in ('lambda3', 'lambda4'):
        actual = regimes.count(regime)
        c.require(actual == collection_summary['counts']['dev'][regime] >= c.CONFIG['min_dev_per_regime'], 'surviving-prefix count matches original collection')
    c.require(fit_summary['data_cases']['dev'] == len(identities), 'same complete evaluation cohort')
    timings = fit_summary['prediction_times']
    timed_specs = set(specs) | {(family, seed, 'opposite') for family in FAMILIES for seed in seeds}
    c.require(len(timings) == 36 and {(r['family'], r['seed'], r['condition']) for r in timings} == timed_specs,
              'complete saved inference cost roster')
    for row in timings:
        c.require(row['cases'] == len(identities) and type(row['seconds']) in (int, float)
                  and math.isfinite(row['seconds']) and row['seconds'] >= 0
                  and type(row['work']) is dict and all(type(v) is int and v >= 0 for v in row['work'].values()),
                  'finite complete recorded inference costs')
    saved = c.read(fitted / 'reports.json')
    c.require(set(saved) == {'reports', 'gate', 'sensitivity'} and len(saved['reports']) == 24, 'complete saved report and gate payload')
    saved_by = {(r['family'], r['fit_seed'], r['condition']): r for r in saved['reports']}
    c.require(len(saved_by) == 24 and set(saved_by) == set(specs), 'unique complete report roster')
    oracle_check = verify_oracles(np, dataset, tables, check=run.check)
    for i, identity in enumerate(identities):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([roster[identity]['seed'], 911])))
        c.require(np.array_equal(dataset['actions'][i], generator.integers(0, 4, size=8, dtype=np.int64)),
                  'fixed public action stream and registered case seed')
    reports = []
    for family, seed, condition in specs:
        run.check()
        run.receipt['pending'] = {'family': family, 'fit_seed': seed, 'condition': condition}
        prefix = f'{family}__{seed}__{condition}__'
        report = scalar_report(np, dataset, predictions[prefix + 'outcome'], predictions[prefix + 'cost'],
                               family=family, fit_seed=seed, condition=condition, check=run.check)
        close_equal(saved_by[family, seed, condition], report)
        reports.append(report)
        counts['views_checked'] += 1
        run.receipt['pending'] = None
    sensitivity = []
    c.require(type(saved['sensitivity']) is list and len(saved['sensitivity']) == 12, 'complete twelve paired effects')
    for index, (family, seed) in enumerate((f, s) for f in FAMILIES for s in seeds):
        run.check()
        run.receipt['pending'] = {'family': family, 'fit_seed': seed, 'condition': 'opposite'}
        report = sensitivity_report(np, dataset, predictions[f'{family}__{seed}__gap__outcome'],
                                    predictions[f'{family}__{seed}__opposite__outcome'], check=run.check)
        row = {'family': family, 'fit_seed': seed, 'report': report}
        close_equal(saved['sensitivity'][index], row, 'paired opposite-action effect')
        sensitivity.append(row)
        counts['sensitivity_views_checked'] += 1
        run.receipt['pending'] = None
    gate = metrics.evaluate_reports(reports, candidate=c.CONFIG['candidate'], controls=c.CONFIG['controls'],
                                    fit_seeds=seeds, regimes=('lambda3', 'lambda4'), thresholds=thresholds())
    close_equal(saved['gate'], gate, 'fixed gate')
    c.require(fit_summary['status'] == ('DEV_PASS' if gate['passed'] else 'DEV_FAIL'), 'original fixed candidate outcome')
    for phase, directory in (('collect', collection), ('fit', fitted)):
        for name in ('dev.npz', 'sensor-laws.npz', 'summary.json') if phase == 'collect' else (
                'predictions.npz', 'reports.json', 'fits.jsonl', 'training-orders.jsonl',
                'normalization.json', 'summary.json'):
            c.require(c.desc(directory / name) == receipts[phase]['files'][name], 'audit input bytes unchanged')
    c.require(counts['array_decodes'] == 3 and counts['views_checked'] == 24 and counts['sensitivity_views_checked'] == 12, 'complete independent audit accounting')
    result = {'version': VERSION, 'agreement': True, 'technical_complete': False,
        'requires_original_supervisor_closure': True, 'counts': counts, 'reports': reports, 'gate': gate,
        'sensitivity': sensitivity, 'oracle_reconstruction': oracle_check,
        'normalization': {'record': c.read(fitted / 'normalization.json'), 'train_variance_recomputed': False,
                          'scale_from_recorded_variance_checked': True, 'fit_metadata_scale_joins_checked': 12},
        'inputs': {'collection_receipt': c.desc(collection / 'receipt.json'),
                   'fit_receipt': c.desc(fitted / 'receipt.json')},
        'limitations': ['No model, fit or simulator is replayed; public Bayesian targets and sensor CDF laws are independently reconstructed.',
                        'Training orders and counts are checked as records; optimizer execution is not replayed.',
                        'TRAIN-derived variance is authenticated metadata, not independently recomputed from TRAIN arrays.',
                        'Initial public priors and raw scalar sensor values are authenticated inputs, not natively recomputed.',
                        'Repeated neural fits share cases and are not independent samples.',
                        'This audit cannot admit old TEST, confirmation or new execution.']}
    run.check()
    c.write(run.out / 'audit.json', result)
    run.receipt['agreement'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'supervision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    c.require(Path(sys.executable).absolute() == c.NUMERICAL, 'qualified numerical audit interpreter')
    run = c.Run(args, 'audit')
    try:
        run_audit(run)
        run.finish()
    except BaseException as error:
        run.finish(error)
        raise


if __name__ == '__main__':
    main()
