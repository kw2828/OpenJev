"""Descriptive decision-error arithmetic on already admitted cost arrays.

No files, models, generators, inference, gates or case selection are used here.
All three arms, five paired fit seeds and every supplied case are mandatory.
Actions use the first raw minimum, including exact ties. Shared cases across
fits are repeated policy evaluations, not independent observations or ensembles.
"""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np

VERSION = 'finite-decision-error-diagnostic-v1'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
CANDIDATE = 'rounded_random'
CONTROLS = ('matched_free_random', 'rounded_anchor')
HORIZONS = (1, 2, 4, 8)
LONG_HORIZONS = (4, 8)
METRICS = ('regret', 'mse', 'centered_mse', 'true_margin', 'true_chosen_gap', 'predicted_contrast')
COMPARISON_METRICS = ('candidate_regret', 'control_regret', 'delta_regret')
CATEGORIES = ('candidate_better', 'control_better', 'equal')
CUTPOINTS = (.001, .01, .1)
BIN_LABELS = ('[0,0.001)', '[0.001,0.01)', '[0.01,0.1)', '[0.1,inf)')
DEFINITIONS = {
    'action': 'First index attaining the raw minimum; exact ties are not perturbed.',
    'regret': 'true_costs[predicted_action] - true_costs[true_action]',
    'mse': 'Mean squared predicted-minus-true cost error over all four actions.',
    'centered_mse': 'Mean squared error after separately subtracting each four-action cost vector mean.',
    'true_margin': 'Second-smallest true cost, including ties, minus the smallest true cost.',
    'true_chosen_gap': 'Identical to regret, retained explicitly as chosen true cost minus optimal true cost.',
    'predicted_contrast': 'predicted_costs[predicted_action] - predicted_costs[true_action]; not confidence or calibration.',
    'delta_regret': 'Candidate regret minus control regret; negative means candidate better.',
    'category': 'Exact sign of delta_regret, with no tolerance: negative, positive, or zero.',
    'margin_bins': 'Fixed left-closed, right-open bins of true margin, including every case and all ties.',
    'population_contributions': 'Sum within category or bin divided by all cases in that comparison, not subgroup count.',
    'within_means': 'Mean within the category or bin; null when count is zero.',
    'fit_means': 'Equal-weight descriptive policy means on the same cases; no ensemble or independent-case interpretation.',
    'scope': 'Retrospective arithmetic only; does not rescue or replace the frozen failed decision rule.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _cost_array(value, shape=None):
    require(isinstance(value, np.ndarray) and value.dtype.kind == 'f', 'floating NumPy cost array required')
    require(value.ndim == 3 and value.shape[0] > 0 and value.shape[1] >= 8 and value.shape[2] == 4,
            'nonempty costs shaped [cases, horizons>=8, 4]')
    require(shape is None or value.shape == shape, 'all predicted costs match true cost shape')
    require(bool(np.isfinite(value).all()), 'finite input costs')
    copied = np.array(value, dtype=np.float64, copy=True, order='C')
    require(bool(np.isfinite(copied).all()), 'finite float64 cost representation')
    return copied


def _means(records, metrics):
    require(bool(records), 'nonempty mean population')
    try:
        result = {name: math.fsum(row[name] for row in records) / len(records) for name in metrics}
    except OverflowError as error:
        raise ValueError('finite aggregate arithmetic required') from error
    require(all(math.isfinite(value) for value in result.values()), 'finite aggregate means')
    return result


def _subgroup(records, total):
    n = len(records)
    means = _means(records, COMPARISON_METRICS) if n else None
    try:
        contributions = {name: math.fsum(row[name] for row in records) / total for name in COMPARISON_METRICS}
    except OverflowError as error:
        raise ValueError('finite subgroup arithmetic required') from error
    require(all(math.isfinite(value) for value in contributions.values()), 'finite subgroup contributions')
    return {'count': n, 'fraction': n / total, 'actions_differ': sum(row['actions_differ'] for row in records),
            'population_contributions': contributions, 'within_means': means}


def _comparison_summary(records):
    n = len(records)
    require(n > 0, 'nonempty comparison population')
    categories = [{'category': category, **_subgroup([r for r in records if r['category'] == category], n)}
                  for category in CATEGORIES]
    bins = [{'bin_index': index, 'lower': (0., *CUTPOINTS)[index],
             'upper': (*CUTPOINTS, None)[index], 'label': BIN_LABELS[index],
             **_subgroup([r for r in records if r['bin_index'] == index], n)} for index in range(4)]
    return {'actions_differ': sum(row['actions_differ'] for row in records),
            'means': _means(records, COMPARISON_METRICS), 'categories': categories, 'margin_bins': bins}


def analyze(true_costs, predicted_costs, case_ids, *, horizons=HORIZONS, check=lambda: None):
    """Return owned JSON-compatible ``records`` and ``summary`` without I/O.

    ``predicted_costs`` is a dict keyed by (arm, integer seed), containing the
    complete three-arm by five-seed product. ``case_ids`` is an ordered string
    sequence shared by every array. Horizons are one-based and fixed to 1/2/4/8.
    Inputs are never mutated. Material arithmetic overflow fails explicitly.
    """
    require(callable(check), 'callable external check')
    checks = 0
    def checked():
        nonlocal checks
        checks += 1
        check()
    checked()
    require(type(horizons) in (tuple, list) and all(type(h) is int for h in horizons)
            and tuple(horizons) == HORIZONS, 'all four fixed ordered horizons required')
    truth = _cost_array(true_costs)
    n = len(truth)
    require(type(case_ids) in (list, tuple) or isinstance(case_ids, np.ndarray), 'ordered case IDs required')
    require(not isinstance(case_ids, np.ndarray) or case_ids.ndim == 1, 'one-dimensional case IDs')
    require(len(case_ids) == n and all(isinstance(value, str) and bool(value) for value in case_ids),
            'one nonempty string ID for every case')
    ids = [str(value) for value in case_ids]
    require(len(set(ids)) == n, 'unique case IDs')
    require(type(predicted_costs) is dict and bool(predicted_costs), 'complete prediction dictionary required')
    require(all(type(key) is tuple and len(key) == 2 and type(key[0]) is str and key[0] in ARMS
                and type(key[1]) is int and 0 <= key[1] < 2**32 for key in predicted_costs),
            'declared arm and uint32 integer fit seed keys')
    seeds = sorted({seed for _arm, seed in predicted_costs})
    require(len(seeds) == 5 and set(predicted_costs) == {(arm, seed) for arm in ARMS for seed in seeds},
            'every arm and all five paired fits required')
    predictions = {key: _cost_array(value, truth.shape) for key, value in predicted_costs.items()}
    records, means, grouped = [], [], {}
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for arm in ARMS:
                for seed in seeds:
                    for horizon in HORIZONS:
                        checked()
                        group = []
                        for index, case_id in enumerate(ids):
                            target, predicted = truth[index, horizon - 1], predictions[arm, seed][index, horizon - 1]
                            true_action, predicted_action = int(np.argmin(target)), int(np.argmin(predicted))
                            regret = float(target[predicted_action] - target[true_action])
                            centered_error = (predicted - predicted.mean()) - (target - target.mean())
                            row = {'arm': arm, 'seed': seed, 'horizon': horizon, 'case_index': index, 'case_id': case_id,
                                   'true_costs': target.tolist(), 'predicted_costs': predicted.tolist(),
                                   'true_action': true_action, 'predicted_action': predicted_action,
                                   'regret': regret, 'mse': float(np.mean((predicted - target) ** 2)),
                                   'centered_mse': float(np.mean(centered_error ** 2)),
                                   'true_margin': float(np.sort(target)[1] - target[true_action]),
                                   'true_chosen_gap': regret,
                                   'predicted_contrast': float(predicted[predicted_action] - predicted[true_action])}
                            require(all(math.isfinite(row[name]) for name in METRICS), 'finite per-case arithmetic')
                            require(row['regret'] >= 0 and row['mse'] >= 0 and row['centered_mse'] >= 0
                                    and row['true_margin'] >= 0 and row['predicted_contrast'] <= 0,
                                    'decision-error signs without clipping')
                            group.append(row)
                        grouped[arm, seed, horizon] = group
                        records.extend(group)
                        means.append({'arm': arm, 'seed': seed, 'horizon': horizon,
                                      'cases': n, 'means': _means(group, METRICS)})
    except FloatingPointError as error:
        raise ValueError('finite per-case arithmetic required') from error
    arm_means = [{'arm': arm, 'horizon': horizon, 'fits': 5, 'cases_per_fit': n, 'case_views': 5 * n,
                  'means': _means([row for seed in seeds for row in grouped[arm, seed, horizon]], METRICS)}
                 for arm in ARMS for horizon in HORIZONS]
    contrasts, comparison_groups = [], {}
    for control in CONTROLS:
        for seed in seeds:
            for horizon in LONG_HORIZONS:
                checked()
                group = []
                for first, second in zip(grouped[CANDIDATE, seed, horizon], grouped[control, seed, horizon], strict=True):
                    delta = first['regret'] - second['regret']
                    require(math.isfinite(delta), 'finite paired regret difference')
                    group.append({'case_id': first['case_id'], 'case_index': first['case_index'],
                                  'true_margin': first['true_margin'], 'bin_index': bisect_right(CUTPOINTS, first['true_margin']),
                                  'candidate_action': first['predicted_action'], 'control_action': second['predicted_action'],
                                  'actions_differ': first['predicted_action'] != second['predicted_action'],
                                  'candidate_regret': first['regret'], 'control_regret': second['regret'],
                                  'delta_regret': delta,
                                  'category': 'candidate_better' if delta < 0 else 'control_better' if delta > 0 else 'equal'})
                comparison_groups[control, seed, horizon] = group
                contrasts.append({'candidate': CANDIDATE, 'control': control, 'seed': seed, 'horizon': horizon,
                                  'cases': n, 'records': group, **_comparison_summary(group)})
    contrast_means = [{'candidate': CANDIDATE, 'control': control, 'horizon': horizon,
                      'fits': 5, 'cases_per_fit': n, 'case_views': 5 * n,
                      **_comparison_summary([row for seed in seeds for row in comparison_groups[control, seed, horizon]])}
                     for control in CONTROLS for horizon in LONG_HORIZONS]
    checked()
    summary = {'version': VERSION,
               'roster': {'arms': list(ARMS), 'candidate': CANDIDATE, 'controls': list(CONTROLS),
                          'seeds': seeds, 'horizons': list(HORIZONS), 'long_horizons': list(LONG_HORIZONS),
                          'case_ids': ids, 'cases': n},
               'definitions': dict(DEFINITIONS), 'means': means, 'arm_means': arm_means,
               'contrasts': contrasts, 'contrast_means': contrast_means,
               'work': {'input_prediction_arrays': 15, 'case_records': 60 * n, 'mean_rows': 60,
                        'arm_mean_rows': 12, 'contrast_rows': 20, 'contrast_case_records': 20 * n,
                        'contrast_mean_rows': 4, 'check_calls': checks, 'model_calls': 0, 'array_file_decodes': 0},
               'descriptive_only': True, 'failed_rule_rescued': False}
    return {'records': records, 'summary': summary}
