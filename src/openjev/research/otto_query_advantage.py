"""Pure reduction of complete saved, paired query-intervention costs.

Positive advantage means fewer capped moves after the neural first action.
Both branches must then use the same declared continuation policy. Horizon
includes the forced first move. Records cannot authenticate these assumptions,
public provenance, IID replicate draws or the within-replicate coupling.

No sampling, model, training, environment or numerical-array imports occur here.
Cohort statistics are descriptive, signed and unclipped; they are not confidence
intervals, signal admission or evidence of learned selectivity/effectiveness.
"""
from __future__ import annotations

import json
import math
from collections import Counter

from openjev.research.otto_teacher_costs import ContinuationRecord

VERSION = "otto-query-advantage-v1"
UINT32_MAX = 2**32 - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def summarize_advantage(records, *, analytic_action, neural_action, replicate_ids, horizon):
    """Reduce exactly the distinct endpoint actions, rejecting extras/missing rows.

    Replicate IDs are sorted before splitting into equal first/second halves.
    The component permits even counts >=4 and H=1..2188; a frozen caller owns
    the empirical allocation. Same-action branches use one physical record per
    replicate, explicitly yielding structural zero, rather than fabricated rows.
    Found at H and unsuccessful at H have equal capped costs but distinct flags.
    """
    _require(_integer(analytic_action, 0, 3) and _integer(neural_action, 0, 3),
             "endpoint actions must be integers in [0,3]")
    _require(_integer(horizon, 1, 2188), "horizon must be an integer in [1,2188]")
    replicas = tuple(replicate_ids)
    _require(len(replicas) >= 4 and len(replicas) % 2 == 0,
             "replicate IDs must have an even count of at least four")
    _require(all(_integer(rep, 0, UINT32_MAX) for rep in replicas)
             and len(set(replicas)) == len(replicas),
             "replicate IDs must be distinct uint32 integers")
    replicas = tuple(sorted(replicas))
    actions, replica_set = sorted({analytic_action, neural_action}), set(replicas)
    panel = {}
    for record in records:
        _require(isinstance(record, ContinuationRecord), "records must be ContinuationRecord instances")
        _require(_integer(record.replicate_id, 0, UINT32_MAX)
                 and _integer(record.first_action, 0, 3), "record IDs must be integers")
        _require(record.replicate_id in replica_set and record.first_action in actions,
                 "record is outside the declared endpoint panel")
        _require(_integer(record.steps, 1, horizon), "steps must include one through H moves")
        _require(type(record.found) is bool, "found must be a boolean")
        _require(record.found or record.steps == horizon, "an unsuccessful record must complete H moves")
        key = record.replicate_id, record.first_action
        _require(key not in panel, "duplicate record")
        panel[key] = record
    n = len(replicas)
    _require(len(panel) == n * len(actions), "incomplete endpoint panel")
    differences = [panel[rep, analytic_action].steps - panel[rep, neural_action].steps for rep in replicas]
    total, sum_squares = sum(differences), sum(d * d for d in differences)
    variance_numerator = n * sum_squares - total * total
    variance = variance_numerator / (n * (n - 1))
    mean_variance = variance_numerator / (n * n * (n - 1))
    half = n // 2
    half_sums = [sum(differences[:half]), sum(differences[half:])]
    signs = [_sign(value) for value in half_sums]
    branches = {}
    for name, action in (("analytic", analytic_action), ("neural", neural_action)):
        selected = [panel[rep, action] for rep in replicas]
        found = sum(record.found for record in selected)
        capped = sum(record.steps == horizon for record in selected)
        branches[name] = {"action": action, "mean_capped_moves": sum(record.steps for record in selected) / n,
                          "found_count": found, "censored_count": n - found, "at_cap_count": capped,
                          "found_fraction": found / n, "censored_fraction": (n - found) / n,
                          "at_cap_fraction": capped / n}
    both_censored = sum(not panel[rep, analytic_action].found and not panel[rep, neural_action].found
                        for rep in replicas)
    return {"version": VERSION, "horizon": horizon, "analytic_action": analytic_action,
            "neural_action": neural_action, "replicate_ids": list(replicas), "replicate_count": n,
            "physical_record_count": len(panel), "structural_zero": analytic_action == neural_action,
            "records": [{"replicate_id": rep, "first_action": action, "steps": panel[rep, action].steps,
                         "found": panel[rep, action].found} for rep in replicas for action in actions],
            "branches": branches, "differences": differences, "sum_difference": total,
            "sum_squared_difference": sum_squares, "variance_numerator": variance_numerator,
            "mean_advantage": total / n, "sample_variance": variance,
            "variance_of_mean": mean_variance, "paired_standard_error": math.sqrt(mean_variance),
            "positive_count": sum(d > 0 for d in differences), "negative_count": sum(d < 0 for d in differences),
            "tie_count": sum(d == 0 for d in differences), "both_censored_count": both_censored,
            "halves": [{"replicate_ids": list(ids), "sum_difference": value, "mean_advantage": value / half,
                        "sign": sign} for ids, value, sign in
                       zip((replicas[:half], replicas[half:]), half_sums, signs, strict=True)],
            "split_same_sign": signs[0] == signs[1],
            "split_same_nonzero_sign": signs[0] == signs[1] and signs[0] != 0,
            "scope": "Saved capped costs only; equal first actions imply structural zero under the declared coupling."}


def _validated_reduction(value):
    _require(type(value) is dict, "reduction must be a complete panel dictionary")
    required = {"analytic_action", "neural_action", "replicate_ids", "horizon", "records"}
    _require(required <= value.keys(), "missing reduction inputs")
    _require(type(value["records"]) is list, "canonical records must be a list")
    records = []
    for row in value["records"]:
        _require(type(row) is dict and set(row) == {"replicate_id", "first_action", "steps", "found"},
                 "canonical record fields mismatch")
        records.append(ContinuationRecord(**row))
    expected = summarize_advantage(records, analytic_action=value["analytic_action"],
                                   neural_action=value["neural_action"],
                                   replicate_ids=value["replicate_ids"], horizon=value["horizon"])
    _require(json.dumps(value, sort_keys=True, allow_nan=False) ==
             json.dumps(expected, sort_keys=True, allow_nan=False),
             "reduction disagrees with its canonical saved records")
    return expected


def _group(rows, weights):
    if not rows:
        return {"anchor_count": 0, "episode_count": 0, "original_weight_mass": 0.0, "statistics": None}
    mass = math.fsum(weights)
    weights = [weight / mass for weight in weights]
    reductions = [row["reduction"] for row in rows]

    def average(values):
        return math.fsum(weight * value for weight, value in zip(weights, values, strict=True))

    means = [row["mean_advantage"] for row in reductions]
    first = [row["halves"][0]["mean_advantage"] for row in reductions]
    second = [row["halves"][1]["mean_advantage"] for row in reductions]
    # Center relative to one saved value before weighted reduction. Constant
    # halves then have exact zero spread, including non-dyadic float means.
    first_offsets = [value - first[0] for value in first]
    second_offsets = [value - second[0] for value in second]
    first_offset_mean, second_offset_mean = average(first_offsets), average(second_offsets)
    first_mean, second_mean = first[0] + first_offset_mean, second[0] + second_offset_mean
    first_centered = [value - first_offset_mean for value in first_offsets]
    second_centered = [value - second_offset_mean for value in second_offsets]
    first_variance = average([value * value for value in first_centered])
    second_variance = average([value * value for value in second_centered])
    covariance = average([a * b for a, b in zip(first_centered, second_centered, strict=True)])
    pooled_variance = (first_variance + second_variance) / 2
    second_moment = average([value * value for value in means])
    noise = average([row["variance_of_mean"] for row in reductions])
    signal = second_moment - noise
    branch_rates = {name: {field: average([row["branches"][name][field] for row in reductions])
                           for field in ("censored_fraction", "at_cap_fraction", "found_fraction")}
                    for name in ("analytic", "neural")}
    return {"anchor_count": len(rows), "episode_count": len({row["episode_id"] for row in rows}),
            "original_weight_mass": mass,
            "statistics": {"mean_advantage": average(means), "second_moment": second_moment,
                           "estimated_mean_noise": noise, "untruncated_signal": signal,
                           "uncentered_reliability": signal / second_moment if second_moment > 0 else None,
                           "half_means": [first_mean, second_mean],
                           "half_variances": [first_variance, second_variance],
                           "cross_half_second_moment": average([a * b for a, b in zip(first, second, strict=True)]),
                           "cross_half_covariance": covariance, "pooled_half_variance": pooled_variance,
                           "repeatability": covariance / pooled_variance if pooled_variance > 0 else None,
                           "half_correlation": covariance / math.sqrt(first_variance * second_variance)
                           if first_variance > 0 and second_variance > 0 else None,
                           "structural_zero_fraction": average([int(row["structural_zero"]) for row in reductions]),
                           "zero_difference_fraction": average([row["tie_count"] / row["replicate_count"]
                                                             for row in reductions]),
                           "both_censored_fraction": average([row["both_censored_count"] / row["replicate_count"]
                                                               for row in reductions]),
                           "split_same_sign_fraction": average([int(row["split_same_sign"]) for row in reductions]),
                           "split_same_nonzero_sign_fraction": average([int(row["split_same_nonzero_sign"])
                                                                        for row in reductions]),
                           "branch_rates": branch_rates}}


def summarize_signal(rows, *, episode_ids):
    """Describe all anchors and the prespecified differing-action subset.

    Each input row has exactly episode_id, anchor_id and reduction. Every declared
    episode must contribute at least one anchor; IDs cannot be inferred from the
    surviving rows. Original weights are equal per episode, then per anchor in
    that episode. The differing-action subset retains and renormalizes those
    weights, not a new equal-episode weighting. Empty subsets are explicit.

    Centered C/V measures cross-half repeatable variation, not label calibration
    or predictive learnability. Uncentered signal/noise ratios are diagnostic
    only: a constant advantage can produce a large ratio with no selectivity.
    No statistic is clipped, and undefined denominators produce None.
    """
    episodes = tuple(episode_ids)
    _require(episodes and all(type(episode) is str and episode for episode in episodes)
             and len(set(episodes)) == len(episodes), "declared episode IDs must be distinct nonempty strings")
    episode_set, anchors, collected = set(episodes), set(), []
    for row in rows:
        _require(type(row) is dict and set(row) == {"episode_id", "anchor_id", "reduction"},
                 "cohort rows require exact episode_id/anchor_id/reduction fields")
        _require(type(row["episode_id"]) is str and row["episode_id"] in episode_set,
                 "row episode is not declared")
        _require(_integer(row["anchor_id"], 0, UINT32_MAX) and row["anchor_id"] not in anchors,
                 "anchor IDs must be unique uint32 integers")
        reduction = _validated_reduction(row["reduction"])
        collected.append({"episode_id": row["episode_id"], "anchor_id": row["anchor_id"], "reduction": reduction})
        anchors.add(row["anchor_id"])
    counts = Counter(row["episode_id"] for row in collected)
    _require(set(counts) == episode_set, "every declared episode requires at least one anchor")
    collected.sort(key=lambda row: (row["episode_id"], row["anchor_id"]))
    first = collected[0]["reduction"]
    _require(all(row["reduction"]["horizon"] == first["horizon"]
                 and row["reduction"]["replicate_ids"] == first["replicate_ids"] for row in collected),
             "cohort panels must share horizon and replicate declarations")
    weights = [1 / (len(episodes) * counts[row["episode_id"]]) for row in collected]
    differing = [(row, weight) for row, weight in zip(collected, weights, strict=True)
                 if not row["reduction"]["structural_zero"]]
    return {"version": VERSION, "episode_ids": sorted(episodes), "episode_count": len(episodes),
            "anchor_count": len(collected), "horizon": first["horizon"],
            "replicate_ids": first["replicate_ids"].copy(),
            "weights": [{"episode_id": row["episode_id"], "anchor_id": row["anchor_id"], "weight": weight}
                        for row, weight in zip(collected, weights, strict=True)],
            "all": _group(collected, weights),
            "different_action": _group([row for row, _ in differing], [weight for _, weight in differing]),
            "scope": "Descriptive supplied-label moments only; no confidence bound, admission or learned-policy claim.",
            "assumptions": ["H includes the forced first move; all unsuccessful branches complete H moves.",
                            "Branches share one fixed continuation policy after their first action.",
                            "Replicate vectors are IID conditional on each fixed public anchor; within-vector coupling is allowed.",
                            "Shared-action endpoints reuse one actual continuation per replicate.",
                            "Selection, episode declarations and sample allocation precede outcome inspection.",
                            "This reducer does not authenticate provenance, coupling or sampling independence.",
                            "Anchors within episodes can be dependent; these cohort moments are not confidence intervals."]}
