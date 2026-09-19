"""Pure arithmetic on explicit saved rewards for the prospective42-row study.

No I/O, RNG, bootstrap, model, simulator or source authentication occurs here.
Rows contain panel, label, ordered unique string case_ids and native_rewards
shaped [cases,50]. The enclosing caller authenticates those values and identity
bindings. All42 rows and all case costs are returned; no fit selection occurs.
An arithmetic gate pass does not certify provenance, a freeze or effectiveness.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np

from openjev.research import reacher_two_observation_protocol as protocol

VERSION = "reacher-two-observation-results-v1"
ROW_KEYS = frozenset({"panel", "label", "case_ids", "native_rewards"})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cases(value, count):
    _require(isinstance(value, (list, tuple)) and len(value) == count
             and all(type(name) is str and name.strip() for name in value),
             "Explicit nonempty string case IDs in declared order")
    _require(len(set(value)) == count, "Unique case IDs required")
    return tuple(value)


def _costs(value, shape):
    _require(type(value) is np.ndarray and value.shape == shape and value.dtype.kind in "iuf",
             "Real numeric native_rewards array with exact [cases,50] shape")
    _require(bool(np.isfinite(value).all()), "Native rewards must be finite")
    # Use one declared arithmetic dtype regardless of saved scalar dtype. Inputs
    # remain untouched. Overflow is invalid, never an implicit +/-inf winner.
    with np.errstate(over="ignore", invalid="ignore"):
        rewards = value.astype(np.float64, copy=True)
        costs = -np.sum(rewards, axis=1, dtype=np.float64)
    _require(bool(np.isfinite(rewards).all()) and bool(np.isfinite(costs).all()),
             "Native reward/cost overflow in float64 arithmetic")
    _require(bool((costs >= 0).all()), "Native episode costs must be nonnegative")
    mean = float(np.mean(costs, dtype=np.float64))
    _require(math.isfinite(mean), "Native row mean overflow")
    return costs, mean


def evaluate_rows(plan, rows):
    """Return all native costs, family means and exactly25 inclusive checks.

    Row sequence order is immaterial; case order is strictly shared across all
    panels, fits and references. Rewards are summed in float64 per episode.
    Family means are arithmetic means of all three fit means; comparisons use
    the exact resulting float64 values with ``<=`` and no epsilon. A zero
    comparator is processed by the same literal inequality, without inventing
    a percentage estimate. No inferential or equivalence result is supplied.

    Engineering settings may shrink case counts as allowed by the protocol,
    but still require every fit, reference and criterion. Returned lists contain
    copies, not views into reward arrays or case-ID containers.
    """
    _require(type(plan) is dict, "Explicit prospective settings required")
    protocol.validate_settings(plan, engineering=plan.get("engineering") is True)
    expected = protocol.execution_order(plan)
    _require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
             and len(rows) == len(expected) == 42, "Exactly42 complete control rows required")
    expected_keys = {(row["panel"], row["label"]) for row in expected}
    supplied, case_ids = {}, None
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == ROW_KEYS, "Exact saved reward-row membership")
        _require(type(row["panel"]) is str and type(row["label"]) is str, "String panel and model/reference label")
        key = row["panel"], row["label"]
        _require(key in expected_keys and key not in supplied, "Unknown or duplicate panel/model/reference row")
        current_ids = _cases(row["case_ids"], plan["control_episodes"])
        if case_ids is None:
            case_ids = current_ids
        _require(current_ids == case_ids, "Identical case identity and order required across all rows/panels")
        costs, mean = _costs(row["native_rewards"], (plan["control_episodes"], plan["steps"]))
        supplied[key] = {"episode_costs": costs.tolist(), "mean_cost": mean,
                         "case_ids": list(current_ids), "reward_dtype": str(row["native_rewards"].dtype)}
    _require(set(supplied) == expected_keys, "Complete all-fit/all-reference coverage required")

    controls = {panel: {} for panel in protocol.PANELS}
    for row in expected:
        controls[row["panel"]][row["label"]] = supplied[row["panel"], row["label"]]
    families = {}
    for panel in protocol.PANELS:
        families[panel] = {}
        for arm in protocol.ARMS:
            names = [f"{arm}-{pair}" for pair in protocol.PAIRS]
            means = [controls[panel][name]["mean_cost"] for name in names]
            with np.errstate(over="ignore", invalid="ignore"):
                mean = float(np.mean(means, dtype=np.float64))
            _require(math.isfinite(mean), "Family mean overflow")
            families[panel][arm] = {"fit_names": names, "fit_mean_costs": means, "mean_cost": mean}

    checks = []
    for criterion in protocol.criterion_manifest(plan):
        panel = criterion["panel"]
        table = families[panel] if criterion["aggregation"] == "three_fit_mean" else controls[panel]
        left = table[criterion["treatment"]]["mean_cost"]
        control_mean = table[criterion["control"]]["mean_cost"]
        right = criterion["maximum_cost_multiplier"] * control_mean
        _require(math.isfinite(right), "Criterion bound overflow")
        checks.append({**copy.deepcopy(criterion), "left": left, "control_mean_cost": control_mean,
                       "right": right, "comparison": "le", "passed": bool(left <= right)})
    _require(len(checks) == len({row["name"] for row in checks}) == 25, "Exact25 unique checks")
    return {"version": VERSION, "study": protocol.STUDY, "status": "arithmetic_completed",
            "engineering": plan["engineering"], "case_ids": list(case_ids),
            "control_rows": 42, "learned_rows": 27, "reference_rows": 15,
            "cases_per_row": plan["control_episodes"], "steps_per_case": plan["steps"],
            "row_order": [row["path"] for row in expected], "controls": controls,
            "families": families, "criterion": copy.deepcopy(plan["criterion"]),
            "continuation_gate": {"passed": all(row["passed"] for row in checks),
                "checks_passed": sum(row["passed"] for row in checks), "total_checks": 25,
                "checks": checks, "secondary_cannot_rescue_primary": True},
            "arithmetic": "Float64 reward sums per episode, mean over all cases per fit, equal mean over all three fits; inclusive <= with no epsilon.",
            "scope": "Arithmetic only on caller-supplied saved rewards and case identities; no authenticity, freeze or effectiveness certification.",
            "excludes": ["source/weight/data/identity authentication", "native execution replay", "CEM and geometry verification",
                         "training verification", "statistical inference", "equivalence testing", "study authorization"],
            "new_model_calls": 0, "new_native_calls": 0, "new_rng_draws": 0}
