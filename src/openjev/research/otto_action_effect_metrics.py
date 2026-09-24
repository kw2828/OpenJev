"""Prospective action-effect gate, separate from the previous forecast gate.

Scoring and opposite-action diagnostics remain frozen belief-distillation APIs.
This module changes no previous result: it checks a new, explicitly supplied
hypothesis. Every originating case contributes to effect error, even at zero
oracle signal. A pure scalar gate cannot authorize execution or establish control.
"""
from __future__ import annotations

import math

from openjev.research import otto_belief_distillation_metrics as base

VERSION = 'otto-action-effect-gate-v1'
score = base.score
action_sensitivity = base.action_sensitivity
require = base.require
EFFECT_FIELDS = ('jensen_shannon', 'total_variation', 'oracle_signal', 'model_effect_error')
THRESHOLD_KEYS = {'long_effect_relative_gain', 'long_gap_relative_gain', 'long_log_relative_tolerance',
                  'normal_log_relative_tolerance', 'normal_gap_relative_tolerance', 'minimum_supported_cases'}


def _finite(value, name):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, 'finite nonnegative ' + name)
    return float(value)


def _mean(values):
    require(bool(values), 'nonempty originating-case average')
    value = math.fsum(float(v) / len(values) for v in values)
    require(math.isfinite(value), 'finite originating-case average')
    return value


def _agreement(left, right):
    return abs(left - right) <= 1e-12 + 1e-12 * abs(right)


def effect_cells(reports, sensitivity, *, families, fit_seeds, regimes):
    """Validate all12 paired effect views and reconstruct long equal-case E/S."""
    reference = reports[0]['identity_manifest']
    require(all(r['identity_manifest'] == reference and r['target_sha256'] == reports[0]['target_sha256']
                for r in reports), 'same factual cohort across gap and normal views')
    wanted = {(family, seed) for family in families for seed in fit_seeds}
    require(type(sensitivity) is list and len(sensitivity) == len(wanted)
            and all(type(row) is dict and set(row) == {'family', 'fit_seed', 'report'} for row in sensitivity),
            'complete exact paired sensitivity roster')
    indexed = {(row['family'], row['fit_seed']): row['report'] for row in sensitivity}
    require(set(indexed) == wanted, 'unique complete sensitivity identities')
    base.hard._finite_tree(sensitivity)
    canonical_oracles, result = {}, {}
    for family in families:
        for seed in fit_seeds:
            record = indexed[family, seed]
            require(record.get('version') == 'otto-belief-action-sensitivity-v1'
                    and record.get('horizons') == list(range(1, 9)) and record.get('blocks') == len(reference)
                    and record.get('case_filtering') is False and record.get('actions_verified_opposite') is True
                    and record.get('condition') == 'gap' and record.get('action_mapping') == 'a XOR1'
                    and record.get('model_effect_included') is True
                    and set(record.get('per_horizon', {})) == set(map(str, range(1, 9))),
                    'complete unfiltered opposite-action effect contract')
            for regime in regimes:
                counts = {}
                for row in reference:
                    if row['regime'] == regime:
                        counts[row['case_id']] = counts.get(row['case_id'], 0) + 1
                require(bool(counts), 'declared regime contains originating cases')
                by_case = {case: {'model_effect_error': [], 'oracle_signal': []} for case in counts}
                for horizon in range(1, 9):
                    group = record['per_horizon'][str(horizon)]
                    require(set(group['by_regime']) == set(regimes), 'complete sensitivity regimes')
                    leaf = group['by_regime'][regime]
                    blocks = sum(counts.values())
                    require(leaf['horizons'] == [horizon] and leaf['declared_cases'] == len(counts)
                            and leaf['blocks'] == leaf['available_rows'] == leaf['scored_rows'] == blocks
                            and leaf['supported_cases'] == len(counts) and leaf['unsupported_case_ids'] == []
                            and type(leaf['cases']) is list and len(leaf['cases']) == len(counts),
                            'all sensitivity cases and rows retained')
                    cases = {row['case_id']: row for row in leaf['cases']}
                    require(set(cases) == set(counts), 'exact sensitivity case roster')
                    for case, expected_count in counts.items():
                        row = cases[case]
                        require(row['blocks'] == row['rows'] == expected_count, 'complete repeated-block case support')
                        for name in EFFECT_FIELDS:
                            _finite(row[name], 'per-case ' + name)
                        oracle = tuple(row[name] for name in EFFECT_FIELDS[:3])
                        key = regime, horizon, case
                        if key not in canonical_oracles:
                            canonical_oracles[key] = oracle
                        require(oracle == canonical_oracles[key], 'same oracle action effects for every method and seed')
                        if horizon >= 5:
                            for name in by_case[case]:
                                by_case[case][name].append(row[name])
                    for name in EFFECT_FIELDS:
                        actual = _finite(leaf['case_weighted_' + name], 'saved sensitivity aggregate')
                        require(_agreement(actual, _mean([row[name] for row in cases.values()])),
                                'sensitivity aggregate equals complete case mean')
                cases = [{'case_id': case, 'horizons': 4,
                          **{name: _mean(values) for name, values in by_case[case].items()}}
                         for case in sorted(counts)]
                result[family, seed, regime] = {'declared_cases': len(counts), 'blocks': sum(counts.values()),
                    'rows': 4 * sum(counts.values()), 'horizons': [5, 6, 7, 8], 'cases': cases,
                    'case_weighted_model_effect_error': _mean([r['model_effect_error'] for r in cases]),
                    'case_weighted_oracle_signal': _mean([r['oracle_signal'] for r in cases])}
    return result


def evaluate_reports(reports, sensitivity, *, candidate, controls, fit_seeds, regimes, thresholds):
    """New six-condition gate for every seed, regime and control, with no defaults.

    Require10%/5% or other supplied relative gains in long E/decision gap;
    require supplied nonregression limits for long NLL and normal NLL/gap.
    Positive support applies to both decision panels. Equal zero losses cannot
    meet a strict improvement condition. The old gate is reported separately.
    """
    require(type(thresholds) is dict and set(thresholds) == THRESHOLD_KEYS, 'complete prospective effect thresholds')
    for key in THRESHOLD_KEYS - {'minimum_supported_cases'}:
        require(_finite(thresholds[key], key) <= 1, 'relative thresholds in [0,1]')
    require(type(thresholds['minimum_supported_cases']) is int and thresholds['minimum_supported_cases'] > 0,
            'positive supported-case minimum')
    # Reuse immutable paired metric validation, not its pass/fail decision.
    base.evaluate_reports(reports, candidate=candidate, controls=controls, fit_seeds=fit_seeds, regimes=regimes,
        thresholds={'long_log_relative_gain': 0., 'long_gap_relative_gain': 0.,
                    'normal_log_relative_tolerance': 0., 'normal_gap_relative_tolerance': 0.,
                    'minimum_supported_cases': thresholds['minimum_supported_cases']})
    families = (candidate, *controls)
    effects = effect_cells(reports, sensitivity, families=families, fit_seeds=fit_seeds, regimes=regimes)
    by = {(r['family'], r['fit_seed'], r['condition']): r for r in reports}
    cells = []
    for seed in fit_seeds:
        for regime in regimes:
            for control in controls:
                def leaf(family, condition, *, seed=seed, regime=regime):
                    name = 'long' if condition == 'gap' else 'all'
                    return by[family, seed, condition]['groups'][name]['by_regime'][regime]
                left, right = leaf(candidate, 'gap'), leaf(control, 'gap')
                normal_left, normal_right = leaf(candidate, 'normal'), leaf(control, 'normal')
                pairs = [
                    ('long_effect_gain', effects[candidate, seed, regime]['case_weighted_model_effect_error'],
                     effects[control, seed, regime]['case_weighted_model_effect_error'], 'long_effect_relative_gain', True),
                    ('long_gap_gain', left['case_weighted_decision_gap'], right['case_weighted_decision_gap'], 'long_gap_relative_gain', True),
                    ('long_log_nonregression', left['case_weighted_log_score'], right['case_weighted_log_score'], 'long_log_relative_tolerance', False),
                    ('normal_log_nonregression', normal_left['case_weighted_log_score'], normal_right['case_weighted_log_score'], 'normal_log_relative_tolerance', False),
                    ('normal_gap_nonregression', normal_left['case_weighted_decision_gap'], normal_right['case_weighted_decision_gap'], 'normal_gap_relative_tolerance', False)]
                conditions = []
                for name, a, b, key, gain in pairs:
                    margin = thresholds[key]
                    passed = a is not None and b is not None and (
                        a < b and a <= (1 - margin) * b if gain else a <= (1 + margin) * b)
                    conditions.append({'name': name, 'passed': passed, 'candidate': a, 'control': b,
                                       'relative_threshold': margin})
                support = {'long_gap': left['supported_cases'], 'normal': normal_left['supported_cases'],
                           'long_effect': effects[candidate, seed, regime]['declared_cases']}
                conditions.append({'name': 'paired_support', 'passed': min(support.values()) >= thresholds['minimum_supported_cases'],
                                   'actual': support, 'required': thresholds['minimum_supported_cases']})
                cells.append({'fit_seed': seed, 'regime': regime, 'control': control, 'conditions': conditions,
                              'passed': all(row['passed'] for row in conditions)})
    return {'version': VERSION, 'candidate': candidate, 'controls': list(controls), 'fit_seeds': list(fit_seeds),
        'regimes': list(regimes), 'thresholds': dict(thresholds), 'cells': cells,
        'passed_cells': sum(row['passed'] for row in cells), 'total_cells': len(cells),
        'passed': all(row['passed'] for row in cells), 'admits_execution': False,
        'hypothesis': 'prospective paired action-effect supervision; previous forecast gate is separately retained',
        'long_effects': [{'family': family, 'fit_seed': seed, 'regime': regime, **effects[family, seed, regime]}
                         for family in families for seed in fit_seeds for regime in regimes],
        'limitations': ['Every effect case is retained, including zero oracle signal; no E/S ratio selects cases.',
                        'Repeated fit seeds share cases and are not independent evaluation samples.',
                        'Teacher-action imitation is not autonomous-control utility.',
                        'This new gate does not change any previous study outcome or establish novelty.']}
