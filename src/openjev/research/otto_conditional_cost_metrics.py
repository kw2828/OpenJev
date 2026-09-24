"""Fixed-horizon, probability-weighted teacher-cost decomposition.

For each originating case, supplied branches contain *unconditional* nonterminal
probability mass at one declared horizon. Normalize that mass once, then compute
information advantage min(E Q) - E(min Q), chosen-action approximation regret
E Q_chosen - min(E Q), and total regret E(Q_chosen - min Q). These are teacher
costs, not environment return or an information floor for a compressed student.

The same nonempty legal mask must hold on every branch of a case, including
zero-weight branches. No branches are sampled, inferred or discarded here.
Empty/all-zero branches stay undefined. Aggregate cases equally conditional on
positive support, never in proportion to survival mass. This is not the old
random surviving-row denominator. Pure standard-library arithmetic, no I/O.
"""
from __future__ import annotations

import math

VERSION = 'otto-conditional-cost-metrics-v1'
ACTIONS = 4
ROUNDING_ULPS = 64
SCALARS = ('conditional_chosen_cost', 'conditional_blind_cost', 'conditional_oracle_cost',
           'information_advantage', 'approximation_regret', 'total_regret',
           'conditional_oracle_cost_variance', 'conditional_chosen_regret_variance')
VECTORS = ('conditional_mean_costs', 'conditional_cost_variances')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float), 'real scalar, not bool: ' + name)
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError('finite scalar: ' + name) from error
    _require(math.isfinite(value) and value >= 0., 'finite nonnegative scalar: ' + name)
    return value


def _sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError('finite arithmetic sum') from error
    _require(math.isfinite(result), 'finite arithmetic sum')
    return result


def _tolerance(*values):
    return ROUNDING_ULPS * math.ulp(max(abs(value) for value in values))


def _weighted(values, probabilities):
    return _sum(p * value for p, value in zip(probabilities, values, strict=True))


def _variance(values, probabilities, mean):
    # Weight before squaring to avoid unnecessary overflow for a tiny branch.
    try:
        return _sum((math.sqrt(p) * (value - mean)) ** 2
                    for p, value in zip(probabilities, values, strict=True))
    except OverflowError as error:
        raise ValueError('finite conditional variance') from error


def _difference(left, right, name, records):
    raw = left - right
    _require(math.isfinite(raw), 'finite difference: ' + name)
    if raw < 0.:
        tolerance = _tolerance(left, right)
        _require(-raw <= tolerance, 'nonnegative decomposition: ' + name)
        records.append({'quantity': name, 'raw': raw, 'stored': 0., 'tolerance': tolerance})
        return 0.
    return raw


def _identity(total, information, approximation, scale, records):
    residual = total - _sum((information, approximation))
    tolerance = _tolerance(total, information, approximation, scale)
    _require(math.isfinite(residual) and abs(residual) <= tolerance,
             'total equals information plus approximation within arithmetic roundoff')
    if residual != 0.:
        records.append({'quantity': 'decomposition_identity', 'raw': residual,
                        'stored': residual, 'tolerance': tolerance})
    return {'residual': residual, 'tolerance': tolerance, 'within_roundoff': True}


def decompose_cases(cases, chosen_actions, *, horizon):
    """Return complete per-case records and equal-supported-case aggregates.

    ``cases`` is a nonempty list/tuple of exact records ``{'case_id': str,
    'branches': [{'weight': number, 'costs': [four numbers], 'legal': [four bool]}]}``.
    ``chosen_actions`` maps each exact case ID to one integer action in [0,3],
    held fixed across its hypothetical branches. Costs and probability weights
    must be finite and nonnegative; total branch mass must be at most one apart
    from recorded arithmetic roundoff. ``horizon`` is an explicit positive int.

    A common legal mask is required within each case, not across cases. Even
    zero-weight branch values are validated; a supplied action must be legal
    when any branch establishes a mask. No mask can be inferred for an empty
    case. Undefined fields are None, with the case and its action retained.

    The fixed 64-ULP bound only handles floating arithmetic, not statistical
    uncertainty. Small negative subtraction artifacts are recorded before
    replacing them by zero. Other quantities are never clipped or renormalized
    across cases. Numerical failure raises ValueError without modifying inputs.
    """
    _require(type(horizon) is int and horizon > 0, 'explicit positive fixed horizon')
    _require(type(cases) in (list, tuple) and len(cases) > 0, 'nonempty originating-case list')
    _require(type(chosen_actions) is dict, 'explicit case-to-action mapping')
    prepared, ids = [], []
    for case in cases:
        _require(type(case) is dict and set(case) == {'case_id', 'branches'}, 'exact originating-case fields')
        identity, branches = case['case_id'], case['branches']
        _require(type(identity) is str and bool(identity.strip()) and identity not in ids, 'unique nonempty case ID')
        _require(type(branches) in (list, tuple), 'explicit branch list')
        ids.append(identity)
        rows, common_legal = [], None
        for branch in branches:
            _require(type(branch) is dict and set(branch) == {'weight', 'costs', 'legal'}, 'exact nonterminal branch fields')
            weight = _number(branch['weight'], 'branch weight')
            costs, legal = branch['costs'], branch['legal']
            _require(type(costs) in (list, tuple) and len(costs) == ACTIONS, 'four teacher costs')
            costs = tuple(_number(value, 'teacher cost') for value in costs)
            _require(type(legal) in (list, tuple) and len(legal) == ACTIONS
                     and all(type(value) is bool for value in legal) and any(legal), 'nonempty four-action Boolean mask')
            legal = tuple(legal)
            if common_legal is None:
                common_legal = legal
            _require(legal == common_legal, 'common legal action set across every branch')
            rows.append((weight, costs))
        prepared.append((identity, rows, common_legal))
    _require(set(chosen_actions) == set(ids), 'one chosen action for every originating case, no extras')
    for identity, _, legal in prepared:
        action = chosen_actions[identity]
        _require(type(action) is int and 0 <= action < ACTIONS, 'integer chosen action in [0,3]')
        _require(legal is None or legal[action], 'chosen action belongs to common legal set')

    results = []
    for identity, rows, legal in prepared:
        action, records = chosen_actions[identity], []
        mass = _sum(weight for weight, _ in rows)
        _require(mass <= 1. + _tolerance(1.), 'nonterminal probability mass at most one')
        if mass > 1.:
            records.append({'quantity': 'support_mass_above_one', 'raw': mass - 1.,
                            'stored': mass, 'tolerance': _tolerance(1.)})
        result = {'case_id': identity, 'horizon': horizon, 'chosen_action': action,
                  'legal': None if legal is None else list(legal), 'branches': len(rows),
                  'positive_weight_branches': sum(weight > 0. for weight, _ in rows),
                  'support_mass': mass, 'defined': mass > 0., 'roundoff_records': records}
        if mass == 0.:
            result.update({key: None for key in (*SCALARS, *VECTORS)})
            result.update(blind_optimal_action=None, identity=None)
            results.append(result)
            continue
        probabilities = [weight / mass for weight, _ in rows]
        normalization_residual = _sum(probabilities) - 1.
        _require(abs(normalization_residual) <= _tolerance(1.), 'finite conditional normalization')
        if normalization_residual != 0.:
            records.append({'quantity': 'conditional_probability_sum', 'raw': normalization_residual,
                            'stored': 1. + normalization_residual, 'tolerance': _tolerance(1.)})
        columns = [[costs[a] for _, costs in rows] for a in range(ACTIONS)]
        means = [_weighted(column, probabilities) for column in columns]
        allowed = [a for a in range(ACTIONS) if legal[a]]
        blind_action = min(allowed, key=lambda a: (means[a], a))
        optimal = [min(costs[a] for a in allowed) for _, costs in rows]
        oracle = _weighted(optimal, probabilities)
        chosen, blind = means[action], means[blind_action]
        information = _difference(blind, oracle, 'information_advantage', records)
        approximation = _difference(chosen, blind, 'approximation_regret', records)
        regrets = [costs[action] - value for (_, costs), value in zip(rows, optimal, strict=True)]
        total = _weighted(regrets, probabilities)
        result.update(conditional_mean_costs=means,
            conditional_cost_variances=[_variance(column, probabilities, mean)
                                        for column, mean in zip(columns, means, strict=True)],
            conditional_chosen_cost=chosen, conditional_blind_cost=blind,
            conditional_oracle_cost=oracle, blind_optimal_action=blind_action,
            information_advantage=information, approximation_regret=approximation, total_regret=total,
            conditional_oracle_cost_variance=_variance(optimal, probabilities, oracle),
            conditional_chosen_regret_variance=_variance(regrets, probabilities, total),
            identity=_identity(total, information, approximation, max(chosen, blind, oracle), records))
        results.append(result)

    supported = [row for row in results if row['defined']]
    aggregate = {'declared_cases': len(results), 'supported_cases': len(supported),
                 'unsupported_case_ids': [row['case_id'] for row in results if not row['defined']],
                 'support_mass_sum': _sum(row['support_mass'] for row in results),
                 'mean_support_mass': _sum(row['support_mass'] / len(results) for row in results),
                 'roundoff_records': []}
    for key in SCALARS:
        aggregate['case_weighted_' + key] = None if not supported else _sum(row[key] / len(supported) for row in supported)
    for key in VECTORS:
        aggregate['case_weighted_' + key] = None if not supported else [
            _sum(row[key][a] / len(supported) for row in supported) for a in range(ACTIONS)]
    aggregate['identity'] = None if not supported else _identity(
        aggregate['case_weighted_total_regret'], aggregate['case_weighted_information_advantage'],
        aggregate['case_weighted_approximation_regret'],
        max(aggregate['case_weighted_' + key] for key in
            ('conditional_chosen_cost', 'conditional_blind_cost', 'conditional_oracle_cost')),
        aggregate['roundoff_records'])
    return {'version': VERSION, 'horizon': horizon, 'action_count': ACTIONS, 'cases': results,
            'aggregate': aggregate, 'rounding_ulps': ROUNDING_ULPS, 'admits_execution': False,
            'conditioning': 'nonterminal at one fixed horizon, normalized separately for each originating case',
            'aggregation': 'equal originating-case weights among positive-support cases; zero-support cases retained',
            'scope': 'teacher-cost information advantage and decision approximation; no environment-return or student-information-floor claim'}
