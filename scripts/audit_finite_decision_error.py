"""Independent scalar reconciliation of a retrospective decision-error record.

The caller authenticates the original study and joins every supplied vector to
its saved prediction/target before entry. This module reads no files, imports no
numerical package, and invokes no producer, learned model or generator.
"""
from __future__ import annotations

import math

VERSION = 'finite-decision-error-audit-v1'
ARMS = ('rounded_anchor', 'rounded_random', 'matched_free_random')
CANDIDATE = 'rounded_random'
CONTROLS = ('matched_free_random', 'rounded_anchor')
HORIZONS = (1, 2, 4, 8)
LONG_HORIZONS = (4, 8)
METRICS = ('regret', 'mse', 'centered_mse', 'true_margin', 'true_chosen_gap', 'predicted_contrast')
CONTRAST_METRICS = ('candidate_regret', 'control_regret', 'delta_regret')
CATEGORIES = ('candidate_better', 'control_better', 'equal')
BIN_EDGES = (0., .001, .01, .1)
BIN_LABELS = ('[0,0.001)', '[0.001,0.01)', '[0.01,0.1)', '[0.1,inf)')
RECORD_KEYS = {'arm', 'seed', 'horizon', 'case_index', 'case_id', 'true_costs', 'predicted_costs',
               'true_action', 'predicted_action', *METRICS}
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


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def reconcile(actual, expected, path='record'):
    """Strict rosters and identities; ordinary scalar reduction-roundoff only."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), path + ': exact fields')
        for key, value in expected.items():
            reconcile(actual[key], value, path + '/' + str(key))
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), path + ': exact list')
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            reconcile(left, right, path + '/' + str(index))
    elif type(expected) is float:
        require(number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                path + ': independent scalar disagreement')
    else:
        require(type(actual) is type(expected) and actual == expected, path + ': exact identity')


def cost_metrics(target, predicted):
    """Raw first-index argmins; no near-tie or margin-based action correction."""
    require(type(target) is list and type(predicted) is list and len(target) == len(predicted) == 4
            and all(number(x) for x in target + predicted), 'finite four-cost vectors')
    true_action = min(range(4), key=lambda i: target[i])
    selected = min(range(4), key=lambda i: predicted[i])
    target_mean, predicted_mean = math.fsum(target) / 4, math.fsum(predicted) / 4
    errors = [predicted[i] - target[i] for i in range(4)]
    centered = [(predicted[i] - predicted_mean) - (target[i] - target_mean) for i in range(4)]
    ordered = sorted(target)
    regret = float(target[selected] - target[true_action])
    result = {'true_action': true_action, 'predicted_action': selected, 'regret': regret,
              'mse': math.fsum(x * x for x in errors) / 4,
              'centered_mse': math.fsum(x * x for x in centered) / 4,
              'true_margin': float(ordered[1] - ordered[0]), 'true_chosen_gap': regret,
              'predicted_contrast': float(predicted[selected] - predicted[true_action])}
    require(all(number(result[key]) for key in METRICS) and result['regret'] >= 0
            and result['predicted_contrast'] <= 0, 'finite argmin witnesses')
    return result


def margin_bin(value):
    require(number(value) and value >= 0, 'nonnegative true action margin')
    return sum(value >= edge for edge in BIN_EDGES[1:])


def metric_means(records):
    require(bool(records), 'positive full-population mean denominator')
    return {key: math.fsum(row[key] for row in records) / len(records) for key in METRICS}


def contribution(records, population):
    require(type(population) is int and population > 0, 'fixed positive population denominator')
    totals = {key: math.fsum(row[key] for row in records) for key in CONTRAST_METRICS}
    return {'count': len(records), 'fraction': len(records) / population,
            'actions_differ': sum(row['actions_differ'] for row in records),
            'population_contributions': {key: value / population for key, value in totals.items()},
            'within_means': None if not records else {key: value / len(records) for key, value in totals.items()}}


def partition(records):
    n = len(records)
    require(n > 0, 'positive complete contrast population')
    categories = [{'category': kind, **contribution([r for r in records if r['category'] == kind], n)}
                  for kind in CATEGORIES]
    bins = [{'bin_index': index, 'lower': lower,
             'upper': BIN_EDGES[index + 1] if index < 3 else None, 'label': BIN_LABELS[index],
             **contribution([r for r in records if r['bin_index'] == index], n)}
            for index, lower in enumerate(BIN_EDGES)]
    means = {key: math.fsum(row[key] for row in records) / n for key in CONTRAST_METRICS}
    for parts in (categories, bins):
        require(sum(row['count'] for row in parts) == n, 'partition covers every case exactly once')
        for key, expected in means.items():
            require(math.isclose(math.fsum(row['population_contributions'][key] for row in parts),
                                 expected, rel_tol=1e-10, abs_tol=1e-12), 'population contributions recover full mean')
    return {'actions_differ': sum(row['actions_differ'] for row in records), 'means': means,
            'categories': categories, 'margin_bins': bins}


def case_contrast(candidate, control):
    require(candidate['case_id'] == control['case_id'] and candidate['case_index'] == control['case_index'],
            'paired case identity')
    delta = candidate['regret'] - control['regret']
    return {'case_id': candidate['case_id'], 'case_index': candidate['case_index'],
            'true_margin': candidate['true_margin'], 'bin_index': margin_bin(candidate['true_margin']),
            'candidate_action': candidate['predicted_action'], 'control_action': control['predicted_action'],
            'actions_differ': candidate['predicted_action'] != control['predicted_action'],
            'candidate_regret': candidate['regret'], 'control_regret': control['regret'], 'delta_regret': delta,
            'category': 'candidate_better' if delta < 0 else 'control_better' if delta > 0 else 'equal'}


def audit(records, summary, parent_rows, *, expected_case_ids, check=lambda: None):
    """Verify complete saved JSON against caller-bound original IDs and means."""
    require(callable(check), 'callable external cap check')
    require(type(expected_case_ids) is list and bool(expected_case_ids)
            and all(type(x) is str and x for x in expected_case_ids)
            and len(set(expected_case_ids)) == len(expected_case_ids), 'original unique ordered case roster')
    require(type(summary) is dict and set(summary) == {'version', 'roster', 'definitions', 'means', 'arm_means',
            'contrasts', 'contrast_means', 'work', 'descriptive_only', 'failed_rule_rescued'}
            and summary['version'] == 'finite-decision-error-diagnostic-v1'
            and type(summary['roster']) is dict, 'exact declared diagnostic summary')
    reconcile(summary['definitions'], DEFINITIONS, 'fixed diagnostic definitions')
    roster = summary['roster']
    seeds = roster['seeds']
    require(type(seeds) is list and len(seeds) == 5 and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
            and seeds == sorted(set(seeds)), 'all five unique paired fit seeds')
    n = len(expected_case_ids)
    reconcile(roster, {'arms': list(ARMS), 'candidate': CANDIDATE, 'controls': list(CONTROLS),
              'seeds': seeds, 'horizons': list(HORIZONS),
              'long_horizons': list(LONG_HORIZONS), 'case_ids': expected_case_ids, 'cases': n}, 'roster')
    require(type(parent_rows) is list and len(parent_rows) == 60, 'all sixty published parent metric rows')
    parents = {(r['arm'], r['seed'], r['horizon']): r for r in parent_rows}
    expected = {(arm, seed, horizon) for arm in ARMS for seed in seeds for horizon in HORIZONS}
    require(len(parents) == 60 and set(parents) == expected
            and all(r['regime'] == 'base' and type(r['cases']) is int and r['cases'] == n for r in parent_rows),
            'parent binds full model, horizon and retained population roster')
    require(type(records) is list and len(records) == 60 * n, 'all sixty complete per-case populations')
    keyed, targets = {}, {}
    for row in records:
        check()
        require(type(row) is dict and set(row) == RECORD_KEYS, 'exact case-record fields')
        arm, seed, horizon, index = row['arm'], row['seed'], row['horizon'], row['case_index']
        require(type(seed) is int and type(horizon) is int and (arm, seed, horizon) in expected
                and type(index) is int and 0 <= index < n and row['case_id'] == expected_case_ids[index],
                'bound model and original case identity')
        key = arm, seed, horizon, index
        require(key not in keyed, 'no duplicate case records')
        computed = cost_metrics(row['true_costs'], row['predicted_costs'])
        reconcile({name: row[name] for name in computed}, computed, 'derived cost record')
        target_key = horizon, index
        if target_key not in targets:
            targets[target_key] = row['true_costs']
        require(row['true_costs'] == targets[target_key], 'one identical true target across every model')
        keyed[key] = {**row, **computed}
    require(len(keyed) == len(expected) * n, 'no omitted case at any model or horizon')
    means = []
    for arm in ARMS:
        for seed in seeds:
            for horizon in HORIZONS:
                check()
                result = metric_means([keyed[arm, seed, horizon, i] for i in range(n)])
                means.append({'arm': arm, 'seed': seed, 'horizon': horizon, 'cases': n, 'means': result})
                reconcile(parents[arm, seed, horizon]['blind_regret'], result['regret'], 'parent regret')
                reconcile(parents[arm, seed, horizon]['blind_cost_mse'], result['mse'], 'parent cost MSE')
    arm_means = [{'arm': arm, 'horizon': h, 'fits': 5, 'cases_per_fit': n, 'case_views': 5 * n,
                  'means': metric_means([keyed[arm, seed, h, i] for seed in seeds for i in range(n)])}
                 for arm in ARMS for h in HORIZONS]
    contrasts, contrast_means, paired = [], [], {}
    for control in CONTROLS:
        for seed in seeds:
            for horizon in LONG_HORIZONS:
                check()
                cases = [case_contrast(keyed[CANDIDATE, seed, horizon, i], keyed[control, seed, horizon, i]) for i in range(n)]
                paired[control, seed, horizon] = cases
                contrasts.append({'candidate': CANDIDATE, 'control': control, 'seed': seed,
                    'horizon': horizon, 'cases': n, 'records': cases, **partition(cases)})
        for horizon in LONG_HORIZONS:
            pooled = [row for seed in seeds for row in paired[control, seed, horizon]]
            contrast_means.append({'candidate': CANDIDATE, 'control': control, 'horizon': horizon,
                'fits': 5, 'cases_per_fit': n, 'case_views': 5 * n, **partition(pooled)})
    for field, rebuilt in (('means', means), ('arm_means', arm_means), ('contrasts', contrasts), ('contrast_means', contrast_means)):
        reconcile(summary[field], rebuilt, 'summary/' + field)
    require(summary['descriptive_only'] is True and summary['failed_rule_rescued'] is False,
            'retrospective decomposition cannot rescue the original failed rule')
    reconcile(summary['work'], {'input_prediction_arrays': 15, 'case_records': 60 * n, 'mean_rows': 60,
        'arm_mean_rows': 12, 'contrast_rows': 20, 'contrast_case_records': 20 * n,
        'contrast_mean_rows': 4, 'check_calls': 82, 'model_calls': 0, 'array_file_decodes': 0},
        'source-derived diagnostic work roster')
    return {'version': VERSION, 'agreement': True, 'technical_complete': False,
            'requires_original_supervisor_closure': True, 'means': means, 'arm_means': arm_means,
            'contrasts': [{key: value for key, value in row.items() if key != 'records'} for row in contrasts],
            'contrast_means': contrast_means,
            'counts': {'case_records_recomputed': 60 * n, 'model_horizon_means': 60,
                       'parent_mean_joins': 120, 'paired_case_contrasts': 20 * n, 'contrasts': 20,
                       'array_decodes': 0, 'checkpoint_decodes': 0, 'model_calls': 0,
                       'optimizer_calls': 0, 'world_or_generator_calls': 0},
            'original_case_ids_bound': True, 'all_population_contributions_reconciled': True,
            'descriptive_only': True, 'failed_rule_rescued': False,
            'limitations': ['The caller authenticates original files and joins every raw cost vector before this scalar audit.',
                'Intermediate latent states and learned updates are not decoded or replayed.',
                'Saved-case decompositions are retrospective associations, not causal identification or new generalization evidence.']}
