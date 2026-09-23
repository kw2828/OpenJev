"""Prespecified sparse-query calibration and complete-case reductions.

This module reads supplied scalar records only. It opens no data/model/runtime,
and it never selects a different primary arm after observing performance.
"""
from __future__ import annotations

import math
from fractions import Fraction

VERSION = "otto-sparse-metrics-v1"
ARMS = ("analytic", "neural", "period2", "random_pair", "entropy")
SPARSE = ARMS[2:]
REGIMES = ("lambda3", "lambda4", "lambda5")
HORIZON = 2188
METRICS = ("found", "steps", "queries", "init_seconds", "choose_seconds",
           "update_seconds", "setup_allocation_seconds", "controller_seconds",
           "paid_controller_seconds", "environment_seconds")


def require(value, message):
    if not value:
        raise ValueError(message)


def select_threshold(validation_decisions, validation_rows):
    """One lower weighted median, equal episodes and equal rows within episode."""
    require(len(validation_rows) == 24, "all twenty-four VALID episodes")
    lengths = {}
    for row in validation_rows:
        index, steps = row["episode_index"], row["steps"]
        require(row["stage"] == "valid" and type(index) is int and 0 <= index < 24
                and index not in lengths and type(steps) is int and 1 <= steps <= HORIZON,
                "unique complete VALID identity and length")
        lengths[index] = steps
    require(set(lengths) == set(range(24)), "every VALID episode")
    entries, seen = [], set()
    for row in validation_decisions:
        index, step, value = row["episode_index"], row["step"], row["entropy"]
        require(type(index) is int and index in lengths and type(step) is int
                and 0 <= step < lengths[index] and (index, step) not in seen,
                "all unique preaction VALID steps")
        require(type(value) is float and math.isfinite(value), "finite saved entropy")
        seen.add((index, step))
        entries.append((value, index, step, Fraction(1, 24 * lengths[index])))
    require(len(seen) == sum(lengths.values()), "no omitted VALID decisions")
    entries.sort(key=lambda x: x[:3])
    cumulative = Fraction(0)
    selected = None
    for value, _, _, weight in entries:
        cumulative += weight
        if cumulative >= Fraction(1, 2):
            selected = value
            break
    require(selected is not None, "defined weighted median")
    return {"value": selected, "threshold": selected,
            "method": "episode_balanced_lower_weighted_median", "episodes": 24,
            "decisions": len(entries), "weighting": "1/(24*episode_steps)",
            "direction": "entropy >= threshold", "total_weight": 1.0,
            "selection_cumulative_weight": float(cumulative),
            "selection_cumulative_weight_exact": [cumulative.numerator, cumulative.denominator],
            "source": "VALID period2 public float32 entropy only; no outcome optimization"}


def aggregate(rows, mixtures, calibration_paid_seconds):
    """All 360 EVAL rows; period2 is the only primary controller."""
    require(len(rows) == 360 and type(calibration_paid_seconds) in (int, float)
            and math.isfinite(calibration_paid_seconds) and calibration_paid_seconds >= 0,
            "complete evaluation and nonnegative calibration bill")
    copied, seen = [], set()
    for original in rows:
        row = dict(original)
        key = row["regime"], row["arm"], row["case"]
        require(row["stage"] == "eval" and key[0] in REGIMES and key[1] in ARMS
                and type(key[2]) is int and 0 <= key[2] < 24 and key not in seen,
                "unique EVAL cell")
        seen.add(key)
        require(row["initial_hit"] == 1 + row["case"] % 3 and row["block"] == row["case"] // 3,
                "fixed hit-balanced paired block")
        require(type(row["steps"]) is int and 1 <= row["steps"] <= HORIZON
                and type(row["found"]) is bool and (row["found"] or row["steps"] == HORIZON)
                and row["updates"] == row["steps"] and row["blocked_steps"] == 0
                and row["final_update_assimilated"] is True,
                "complete native tail and final update")
        require(type(row["queries"]) is int and 0 <= row["queries"] <= row["steps"],
                "integer deployed queries")
        require(row["arm"] != "analytic" or row["queries"] == 0, "analytic has zero queries")
        require(row["arm"] != "neural" or row["queries"] == row["steps"], "neural always queries")
        if row["arm"] in SPARSE:
            require(row["queries"] <= (row["steps"] + 1) // 2
                    and type(row["query_quota_valid"]) is bool, "saved quota witness")
        row["paid_controller_seconds"] = row["controller_seconds"] + (
            calibration_paid_seconds / 72 if row["arm"] == "entropy" else 0.0)
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in METRICS), "finite nonnegative outcomes/costs")
        require(math.isclose(row["controller_seconds"], math.fsum(row[k] for k in
                    ("init_seconds", "choose_seconds", "update_seconds", "setup_allocation_seconds")),
                    rel_tol=1e-12, abs_tol=1e-9), "complete online controller cost")
        copied.append(row)
    require(len(seen) == 360, "all three regimes, five arms, twenty-four cases")
    primary, diagnostic = [], []

    def condition(target, name, value, threshold, relation):
        target.append({"name": name, "value": value, "threshold": threshold,
                       "relation": relation,
                       "passes": bool(value >= threshold if relation == ">=" else value <= threshold)})

    condition(primary, "technical_complete", 1, 1, ">=")
    condition(primary, "causal_quotas", int(all(r["query_quota_valid"] for r in copied
                                               if r["arm"] in SPARSE)), 1, ">=")
    regimes = {}
    for regime in REGIMES:
        weights = {h: float(mixtures[regime].get(h, mixtures[regime].get(str(h)))) for h in (1, 2, 3)}
        require(all(math.isfinite(x) and x > 0 for x in weights.values())
                and abs(math.fsum(weights.values()) - 1) <= 1e-12, "qualified positive-hit mixture")
        local = [r for r in copied if r["regime"] == regime]

        def means(subset, weights=weights):
            return {m: math.fsum(weights[h] * math.fsum(float(r[m]) for r in subset if r["initial_hit"] == h)
                               / sum(r["initial_hit"] == h for r in subset) for h in (1, 2, 3)) for m in METRICS}

        arms = {arm: means([r for r in local if r["arm"] == arm]) for arm in ARMS}
        blocks = [{arm: means([r for r in local if r["arm"] == arm and r["block"] == block])
                   for arm in ARMS} for block in range(8)]
        if regime != "lambda5":
            for reference in ("analytic", "neural"):
                condition(primary, f"{regime}.{reference}.success", arms[reference]["found"], .95, ">=")
        for arm in SPARSE:
            own, neural, analytic = arms[arm], arms["neural"], arms["analytic"]
            comparisons = (
                ("success", own["found"], .95, ">="),
                ("success_vs_references", own["found"], max(neural["found"], analytic["found"]), ">="),
                ("moves_vs_neural", own["steps"], 1.05 * neural["steps"], "<="),
                ("moves_vs_analytic", own["steps"], .95 * analytic["steps"], "<="),
                ("paid_cost_vs_neural", own["paid_controller_seconds"], .60 * neural["paid_controller_seconds"], "<="),
            )
            for name, value, threshold, relation in comparisons:
                condition(diagnostic, f"{regime}.{arm}.{name}", value, threshold, relation)
                if regime != "lambda5" and arm == "period2":
                    condition(primary, f"{regime}.{arm}.{name}", value, threshold, relation)
            condition(diagnostic, f"{regime}.{arm}.old_half_cost_descriptive",
                      own["paid_controller_seconds"], .50 * neural["paid_controller_seconds"], "<=")
        regimes[regime] = {"weights": weights, "means": arms, "blocks": blocks,
            "raw_counts": {arm: {"episodes": 24, "found": sum(r["found"] for r in local if r["arm"] == arm),
                                    "steps": sum(r["steps"] for r in local if r["arm"] == arm),
                                    "queries": sum(r["queries"] for r in local if r["arm"] == arm)} for arm in ARMS}}
    require(len(primary) == 16 and len(diagnostic) == 54, "all prespecified criteria")
    return {"version": VERSION, "episodes": 360, "paired_cases": 72, "primary_arm": "period2",
            "regimes": regimes, "required": primary, "diagnostic": diagnostic,
            "required_conditions": 16, "required_passed": sum(c["passes"] for c in primary),
            "diagnostic_conditions": 54, "diagnostic_passed": sum(c["passes"] for c in diagnostic),
            "pilot_continuation": all(c["passes"] for c in primary),
            "calibration_paid_seconds": calibration_paid_seconds,
            "calibration_per_entropy_episode_seconds": calibration_paid_seconds / 72,
            "scope": "Period2 only primary; random/entropy and length5 diagnostic. Technical admission also requires "
                     "original process closure and independent saved audit. No recurrence or architecture efficacy claim."}
