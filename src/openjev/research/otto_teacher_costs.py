"""Summarize complete, paired finite-horizon teacher-continuation records.

This module only reduces supplied records. It neither generates rollouts nor
admits a training/evaluation study. A record's steps includes its forced first
action. The caller must establish one nonterminal public anchor, a fixed teacher,
one horizon, and independent, identically distributed replicate vectors from the
declared public-belief distribution. Shared randomness *within* a replicate is
allowed. IDs and numerical validation cannot establish those sampling facts.

Intervals are simultaneous over the declared action pairs at that single anchor,
conditional on those assumptions. They do not cover adaptive sampling, selected
anchors, multiple anchors, learned policies or architectural effectiveness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class ContinuationRecord:
    replicate_id: int
    first_action: int
    steps: int
    found: bool


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_costs(records, *, eligible_actions, replicate_ids, horizon, familywise_alpha=0.05):
    """Return capped costs and paired differences; lower cost is better.

    Censoring means a completed unsuccessful horizon, never a missing or failed
    rollout. Success on the final allowed action is distinct from censoring even
    though both have the same capped cost. Centering subtracts the unweighted
    mean across eligible actions. Pair differences are first minus second.

    Expected actions and replicates are supplied separately so missing records
    cannot silently reduce a denominator. A future collector must authenticate
    these declarations and preserve failures before using this reducer.
    """
    actions = tuple(eligible_actions)
    replicas = tuple(replicate_ids)
    _require(actions and all(type(a) is int and 0 <= a <= 3 for a in actions),
             'eligible_actions must be nonempty integers in [0, 3]')
    _require(len(set(actions)) == len(actions), 'eligible_actions must be distinct')
    _require(replicas and all(type(r) is int and r >= 0 for r in replicas),
             'replicate_ids must be nonempty nonnegative integers')
    _require(len(set(replicas)) == len(replicas), 'replicate_ids must be distinct')
    _require(type(horizon) is int and 1 <= horizon <= 2**53-1,
             'horizon must be a positive exactly representable integer')
    _require(type(familywise_alpha) in (int, float) and 0 < familywise_alpha < 1
             and math.isfinite(familywise_alpha),
             'familywise_alpha must be finite and strictly between zero and one')
    actions, replicas = tuple(sorted(actions)), tuple(sorted(replicas))
    action_set, replica_set = set(actions), set(replicas)
    panel = {}
    for record in records:
        _require(isinstance(record, ContinuationRecord), 'records must be ContinuationRecord instances')
        _require(type(record.replicate_id) is int and type(record.first_action) is int,
                 'record IDs must be integers, not booleans')
        _require(record.replicate_id in replica_set and record.first_action in action_set,
                 'record pair is outside the declared panel')
        _require(type(record.steps) is int and 1 <= record.steps <= horizon,
                 'steps must be an integer from one through the horizon')
        _require(type(record.found) is bool, 'found must be a boolean')
        _require(record.found or record.steps == horizon,
                 'censored records must complete the horizon')
        key = record.replicate_id, record.first_action
        _require(key not in panel, 'duplicate record pair')
        panel[key] = record
    count = len(replicas)
    _require(len(panel) == count*len(actions), 'missing records in declared panel')
    totals = {a: sum(panel[r, a].steps for r in replicas) for a in actions}
    grand_total = sum(totals.values())
    action_rows = []
    for action in actions:
        successes = sum(panel[r, action].found for r in replicas)
        action_rows.append({'action': action, 'mean_cost': totals[action]/count,
                            'centered_mean_cost': (len(actions)*totals[action]-grand_total)/(count*len(actions)),
                            'success_fraction': successes/count,
                            'censored_fraction': (count-successes)/count})

    pairs = []
    pair_count = len(actions)*(len(actions)-1)//2
    support = horizon-1
    # Hoeffding on [-support, support], union bound over all unordered pairs.
    radius = (2*support*math.sqrt((math.log(2*pair_count)-math.log(familywise_alpha))/(2*count))
              if pair_count else 0.)
    for first, second in combinations(actions, 2):
        differences = [panel[r, first].steps-panel[r, second].steps for r in replicas]
        total_difference = sum(differences)
        mean = total_difference/count
        # Integer sufficient statistics retain small paired variation even when
        # costs share a large offset. Subtracting a rounded float mean loses it.
        variance_numerator = count*sum(x*x for x in differences)-total_difference**2
        standard_error = (math.sqrt(variance_numerator/(count*count*(count-1))) if count > 1 else None)
        lower, upper = max(-support, mean-radius), min(support, mean+radius)
        direction = 'first_lower' if upper < 0 else 'second_lower' if lower > 0 else 'unresolved'
        pairs.append({'first_action': first, 'second_action': second,
                      'mean_difference': mean, 'paired_standard_error': standard_error,
                      'hoeffding_radius': radius, 'hoeffding_interval': [lower, upper],
                      'resolved_direction': direction})

    return {'version': 'otto-teacher-cost-summary-v1', 'horizon': horizon,
            'replicate_count': count, 'action_count': len(actions), 'pair_count': pair_count,
            'familywise_alpha': familywise_alpha, 'actions': action_rows, 'pairs': pairs,
            'scope': 'Supplied-record reduction only; conditional simultaneous pair intervals for one fixed anchor.',
            'assumptions': [
                'The anchor is nonterminal; steps includes the forced first action.',
                'Every declared action and replicate uses the same fixed teacher and horizon.',
                'Replicate vectors are independent and identically distributed from the declared public belief and observation model.',
                'Common randomness across actions within a replicate is permitted.',
                'Replicate IDs, eligible actions and sample allocation were fixed before observing outcomes.',
                'Provenance, public-only teacher inputs and sampling independence are not verified by this reducer.',
                'Intervals do not cover adaptive sampling, multiple or selected anchors, or policy effectiveness.',
                'Paired standard error is descriptive; zero empirical error does not establish a deterministic population.',
            ]}
