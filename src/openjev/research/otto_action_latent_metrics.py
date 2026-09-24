"""Pure scoring for prospective eight-step action/observation-gap predictions.

No IO, fitting, environment calls, hypothesis selection or split admission occurs
here. Arrays describe complete precommitted blocks; repeated case IDs are blocks
from the same originating case, not independent cases. Class 4 means found and
is absorbing in the target sequence. Outcome losses score every row, including
absorbed suffixes. Decision gaps score only nonterminal rows and first average
within each originating case, then across cases with decision support. Cases
without survivors are retained explicitly; no-survivor decision means are None.

Logarithmic score uses natural logarithms. Brier is the sum over all five classes,
without dividing by five. Exact lowest-index legal argmin defines the decision.
Only the ordering of predicted costs matters; a shared additive gauge is allowed.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

VERSION = "otto-action-latent-metrics-v1"
HORIZONS = tuple(range(1, 9))
GROUPS = {"all": HORIZONS, "short": HORIZONS[:4], "long": HORIZONS[4:]}
METRICS = ("log_score", "brier", "decision_gap")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "finite nonnegative " + label)
    return float(value)


def _array(value, shape, name):
    require(isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype.kind in "fiu" and value.dtype.kind != "b", "real array " + name)
    result = value.astype(np.float64, copy=True)
    require(bool(np.isfinite(result).all()), "finite values everywhere in " + name)
    return result


def _mean(values):
    try:
        # Divide first so an otherwise finite mean does not overflow its sum.
        value = math.fsum(float(v) / len(values) for v in values)
    except (OverflowError, ValueError) as error:
        raise ValueError("finite aggregate arithmetic") from error
    require(math.isfinite(value), "finite aggregate arithmetic")
    return value


def _finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), "no nonfinite value hidden in saved report")


def _outcome_scores(target, values, kind):
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            if kind == "logits":
                shifted = values - values.max(axis=2, keepdims=True)
                exponential = np.exp(shifted)
                denominator = exponential.sum(axis=2, keepdims=True)
                log_probabilities = shifted - np.log(denominator)
                probabilities = exponential / denominator
                nll = -np.take_along_axis(log_probabilities, target[..., None], axis=2)[..., 0]
            else:
                require(kind == "probabilities", "explicit logits or probabilities")
                require(bool((values >= 0).all()) and bool((values <= 1).all()), "probabilities in [0,1]")
                totals = values.sum(axis=2, keepdims=True)
                require(bool(np.isclose(totals, 1., rtol=0., atol=1e-12).all()), "unit probability rows")
                # Correct only floating-point summation drift within the declared
                # tolerance. Never clip zero probabilities or logits.
                probabilities = values / totals
                observed = np.take_along_axis(probabilities, target[..., None], axis=2)[..., 0]
                require(bool((observed > 0).all()), "observed zero probability has infinite logarithmic loss")
                nll = -np.log(observed)
            error = probabilities.copy()
            np.put_along_axis(error, target[..., None],
                              np.take_along_axis(error, target[..., None], axis=2) - 1, axis=2)
            brier = np.square(error).sum(axis=2)
        except FloatingPointError as error:
            raise ValueError("finite outcome-score arithmetic; no clipping") from error
    require(bool(np.isfinite(nll).all()) and bool(np.isfinite(brier).all()), "finite proper scores")
    return nll, brier


def score(outcomes, outcome_predictions, predicted_costs, raw_costs, legal, *, case_ids, regimes,
          family, fit_seed, condition, prediction_kind="logits"):
    """Score owned float64 computations on complete [blocks,8,...] inputs.

    outcomes is an integer array of labels 0..4; prediction shape is [B,8,5].
    Both cost arrays have shape [B,8,4] and must be finite even at terminal rows.
    legal is Boolean [B,8,4], all false exactly at terminal targets and nonempty
    otherwise. case_ids/regimes are nonempty strings for each block. A case ID
    can repeat only within its single regime. Family, seed and condition name
    are metadata, never model selection. Condition is exactly gap or normal.

    Per-case decision means condition on that case's surviving rows. The primary
    case_weighted_decision_gap conditions on supported cases; an additional
    full_case_denominator_gap discloses the same sum divided by all cases. Both
    are None if every case lacks decision support. No terminal action is scored.
    """
    require(isinstance(outcomes, np.ndarray) and outcomes.ndim == 2 and outcomes.shape[1] == 8
            and outcomes.shape[0] > 0 and outcomes.dtype.kind in "iu", "integer [blocks,8] outcomes")
    rows = len(outcomes)
    require(bool(((outcomes >= 0) & (outcomes <= 4)).all()), "five outcome classes")
    target = outcomes.astype(np.int64, copy=True)
    terminal = target == 4
    require(not bool((terminal[:, :-1] & ~terminal[:, 1:]).any()), "found targets remain absorbing")
    require(isinstance(legal, np.ndarray) and legal.dtype == np.bool_ and legal.shape == (rows, 8, 4),
            "Boolean legal-action mask")
    mask = legal.copy()
    require(bool((mask.any(axis=2) == ~terminal).all()), "legal actions exist exactly for nonterminal targets")
    require(isinstance(case_ids, (list, tuple)) and isinstance(regimes, (list, tuple))
            and len(case_ids) == len(regimes) == rows
            and all(type(v) is str and v.strip() for v in (*case_ids, *regimes)), "one explicit case and regime per block")
    require(all(len({r for c, r in zip(case_ids, regimes, strict=True) if c == case}) == 1 for case in set(case_ids)),
            "originating case belongs to one regime")
    require(type(family) is str and bool(family.strip()) and type(fit_seed) is int and fit_seed >= 0
            and condition in ("gap", "normal"), "explicit family, fit seed and evaluation condition")
    predicted = _array(predicted_costs, (rows, 8, 4), "predicted cost contrasts")
    teacher = _array(raw_costs, (rows, 8, 4), "raw teacher costs")
    values = _array(outcome_predictions, (rows, 8, 5), "outcome predictions")
    nll, brier = _outcome_scores(target, values, prediction_kind)
    gaps = np.zeros((rows, 8), np.float64)
    chosen = np.full((rows, 8), -1, np.int64)
    for block in range(rows):
        for index in range(8):
            if terminal[block, index]:
                continue
            allowed = np.flatnonzero(mask[block, index])
            action = int(allowed[np.argmin(predicted[block, index, allowed])])
            chosen[block, index] = action
            with np.errstate(over="raise", invalid="raise"):
                try:
                    gap = float(teacher[block, index, action] - teacher[block, index, allowed].min())
                except FloatingPointError as error:
                    raise ValueError("finite teacher-score gap arithmetic") from error
            gaps[block, index] = _finite(gap, "decision gap")

    def summary(indices, horizons):
        steps = [h - 1 for h in horizons]
        cases = sorted({case_ids[i] for i in indices})
        records = []
        for case in cases:
            blocks = [i for i in indices if case_ids[i] == case]
            pairs = [(i, t) for i in blocks for t in steps]
            decisions = [(i, t) for i, t in pairs if not terminal[i, t]]
            records.append({"case_id": case, "blocks": len(blocks), "outcome_rows": len(pairs),
                "decision_rows": len(decisions), "terminal_rows": len(pairs) - len(decisions),
                "log_score": _mean([nll[i, t] for i, t in pairs]),
                "brier": _mean([brier[i, t] for i, t in pairs]),
                "decision_gap": _mean([gaps[i, t] for i, t in decisions]) if decisions else None})
        supported = [r["decision_gap"] for r in records if r["decision_gap"] is not None]
        return {"horizons": list(horizons), "declared_cases": len(cases), "blocks": len(indices),
            "outcome_rows": sum(r["outcome_rows"] for r in records),
            "terminal_rows": sum(r["terminal_rows"] for r in records),
            "decision_rows": sum(r["decision_rows"] for r in records), "supported_cases": len(supported),
            "unsupported_case_ids": [r["case_id"] for r in records if r["decision_gap"] is None],
            "case_weighted_log_score": _mean([r["log_score"] for r in records]),
            "case_weighted_brier": _mean([r["brier"] for r in records]),
            "case_weighted_decision_gap": _mean(supported) if supported else None,
            "full_case_denominator_gap": _mean([r["decision_gap"] or 0. for r in records]) if supported else None,
            "cases": records}

    def groups(horizons):
        return {"overall": summary(list(range(rows)), horizons),
                "by_regime": {regime: summary([i for i, r in enumerate(regimes) if r == regime], horizons)
                              for regime in sorted(set(regimes))}}

    digest = hashlib.sha256(b"otto-action-latent-targets-v1\0")
    digest.update(target.astype("<i8").tobytes())
    digest.update(mask.tobytes())
    digest.update(teacher.astype("<f8").tobytes())
    return {"version": VERSION, "family": family, "fit_seed": fit_seed, "condition": condition,
        "prediction_kind": prediction_kind, "horizons": list(HORIZONS), "target_sha256": digest.hexdigest(),
        "identity_manifest": [{"block_index": i, "case_id": c, "regime": r}
                              for i, (c, r) in enumerate(zip(case_ids, regimes, strict=True))],
        "per_horizon": {str(h): groups((h,)) for h in HORIZONS},
        "groups": {name: groups(horizons) for name, horizons in GROUPS.items()},
        "chosen_actions": chosen.tolist(),
        "scope": "proper five-outcome prediction and teacher-action imitation on fixed precommitted blocks"}


def evaluate_reports(reports, *, candidate, controls, fit_seeds, regimes, thresholds):
    """Conjoin every seed/regime/control comparison; never select a candidate.

    Exactly three fit seeds and three fixed controls are required. Threshold keys
    must be long_log_relative_gain, long_gap_relative_gain,
    normal_log_relative_tolerance, normal_gap_relative_tolerance and
    minimum_supported_cases. No thresholds have empirical defaults. Long gains
    must also be strict, including when the requested margin is zero. Normal
    no-regression comparisons use the all-horizon group. A supported-case minimum
    applies to both long gaps and normal observations. Empty decision support
    always fails. Passing this pure arithmetic gate does not admit execution.
    """
    require(type(candidate) is str and candidate.strip() and isinstance(controls, (tuple, list))
            and len(controls) == len(set(controls)) == 3
            and all(type(c) is str and c.strip() and c != candidate for c in controls), "fixed candidate and three controls")
    require(isinstance(fit_seeds, (tuple, list)) and len(fit_seeds) == len(set(fit_seeds)) == 3
            and all(type(s) is int and s >= 0 for s in fit_seeds), "exactly three fixed fit seeds")
    require(isinstance(regimes, (tuple, list)) and len(regimes) == len(set(regimes)) > 0
            and all(type(r) is str and r.strip() for r in regimes), "explicit fixed regimes")
    keys = {"long_log_relative_gain", "long_gap_relative_gain", "normal_log_relative_tolerance",
            "normal_gap_relative_tolerance", "minimum_supported_cases"}
    require(type(thresholds) is dict and set(thresholds) == keys, "complete explicit prospective thresholds")
    for key in keys - {"minimum_supported_cases"}:
        value = _finite(thresholds[key], key)
        require(value <= 1, "relative threshold in [0,1]")
    require(type(thresholds["minimum_supported_cases"]) is int and thresholds["minimum_supported_cases"] > 0,
            "positive supported-case minimum")
    families = (candidate, *controls)
    wanted = {(c, a, s) for c in ("gap", "normal") for a in families for s in fit_seeds}
    require(isinstance(reports, (tuple, list)) and len(reports) == len(wanted)
            and all(type(r) is dict for r in reports), "complete paired report roster")
    _finite_tree(reports)
    by = {(r.get("condition"), r.get("family"), r.get("fit_seed")): r for r in reports}
    require(set(by) == wanted, "unique exact paired report roster")
    for condition in ("gap", "normal"):
        reference = by[condition, candidate, fit_seeds[0]]
        require(reference.get("version") == VERSION and reference.get("horizons") == list(HORIZONS), "qualified metric version and horizons")
        digest, manifest = reference.get("target_sha256"), reference.get("identity_manifest")
        require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
                "explicit complete target fingerprint")
        require(isinstance(manifest, list) and bool(manifest) and all(type(row) is dict
                and set(row) == {"block_index", "case_id", "regime"} and row["block_index"] == i
                and type(row["block_index"]) is int and type(row["case_id"]) is str and row["case_id"].strip()
                and row["regime"] in regimes for i, row in enumerate(manifest)), "complete ordered block identities")
        for family in families:
            for seed in fit_seeds:
                report = by[condition, family, seed]
                require(report.get("version") == VERSION and report.get("horizons") == list(HORIZONS)
                        and report.get("identity_manifest") == reference.get("identity_manifest")
                        and report.get("target_sha256") == reference.get("target_sha256"), "paired identical cases, labels, teacher costs and support")
                for name in GROUPS:
                    require(set(report["groups"][name]["by_regime"]) == set(regimes), "complete exact regime roster")

    def leaf(condition, family, seed, regime):
        name = "long" if condition == "gap" else "all"
        value = by[condition, family, seed]["groups"][name]["by_regime"][regime]
        require(type(value["declared_cases"]) is int and value["declared_cases"] > 0
                and type(value["supported_cases"]) is int and 0 <= value["supported_cases"] <= value["declared_cases"],
                "explicit decision support denominator")
        expected_cases = {r["case_id"] for r in by[condition, family, seed]["identity_manifest"] if r["regime"] == regime}
        require(value["declared_cases"] == len(expected_cases)
                and isinstance(value["unsupported_case_ids"], list)
                and len(set(value["unsupported_case_ids"])) == len(value["unsupported_case_ids"])
                and set(value["unsupported_case_ids"]) <= expected_cases, "support names match originating cases")
        require(type(value["outcome_rows"]) is int and value["outcome_rows"] > 0
                and type(value["decision_rows"]) is int and 0 <= value["decision_rows"] <= value["outcome_rows"]
                and (value["decision_rows"] > 0) == (value["supported_cases"] > 0), "explicit scored-row denominators")
        require(len(value["unsupported_case_ids"]) == value["declared_cases"] - value["supported_cases"], "retained unsupported cases")
        _finite(value["case_weighted_log_score"], "saved proper loss")
        _finite(value["case_weighted_brier"], "saved Brier loss")
        if value["supported_cases"]:
            _finite(value["case_weighted_decision_gap"], "saved supported decision gap")
        else:
            require(value["case_weighted_decision_gap"] is None, "undefined no-survivor decision gap")
        return value

    cells = []
    for seed in fit_seeds:
        for regime in regimes:
            for control in controls:
                conditions = []
                for condition in ("gap", "normal"):
                    left, right = leaf(condition, candidate, seed, regime), leaf(condition, control, seed, regime)
                    hkeys = ("declared_cases", "supported_cases", "unsupported_case_ids", "decision_rows", "outcome_rows")
                    require(all(left[k] == right[k] for k in hkeys), "identical paired scoring support")
                    support = left["supported_cases"] >= thresholds["minimum_supported_cases"]
                    conditions.append({"name": condition + "_support", "passed": support,
                                       "actual": left["supported_cases"], "required": thresholds["minimum_supported_cases"]})
                    for metric, short in (("log_score", "log"), ("decision_gap", "gap")):
                        a, b = left["case_weighted_" + metric], right["case_weighted_" + metric]
                        key = ("long_" + short + "_relative_gain") if condition == "gap" else ("normal_" + short + "_relative_tolerance")
                        margin = thresholds[key]
                        passed = (a is not None and b is not None and
                                  (a < b and a <= (1 - margin) * b if condition == "gap" else a <= (1 + margin) * b))
                        conditions.append({"name": condition + "_" + metric, "passed": passed,
                                           "candidate": a, "control": b, "relative_threshold": margin})
                cells.append({"fit_seed": seed, "regime": regime, "control": control,
                              "conditions": conditions, "passed": all(c["passed"] for c in conditions)})
    return {"version": "otto-action-latent-gate-v1", "candidate": candidate, "controls": list(controls),
        "fit_seeds": list(fit_seeds), "regimes": list(regimes), "thresholds": dict(thresholds),
        "cells": cells, "passed_cells": sum(c["passed"] for c in cells), "total_cells": len(cells),
        "passed": all(c["passed"] for c in cells), "admits_execution": False,
        "limitations": ["Repeated fits share originating cases and are not independent evaluation samples.",
                        "Repeated deterministic ridge views can represent one fit, not three independent fits.",
                        "A scalar comparison does not authenticate data, process closure or scientific novelty."]}
