"""Oracle-relative diagnostics for a prospective belief-distillation pilot.

The frozen hard-outcome scorer and its gate are reused without retuning. Oracle
KL and Brier excess are supplementary. No fitting, IO, model/environment calls,
case filtering or hypothesis selection occurs here. All logarithms are natural.

In gaps every row is unresolved: an unobserved found event cannot be fed back.
In normal observations only rows AFTER the first observed found event are
resolved. The caller must supply the exact analytical one-hot found distribution
there, for both prediction and oracle. This module validates, never repairs, it.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from openjev.research import otto_action_latent_metrics as hard

VERSION = "otto-belief-distillation-metrics-v1"
DIAGNOSTICS = ("kl", "brier_excess", "expected_log_score", "oracle_entropy",
               "expected_brier", "oracle_brier", "sampled_log_score", "sampled_brier")
require = hard.require


def _probabilities(value, shape, name):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in "fiu",
            "real probability array " + name)
    result = value.astype(np.float64, copy=True)
    require(bool(np.isfinite(result).all()) and bool(((result >= 0) & (result <= 1)).all()),
            "finite probability interval " + name)
    totals = result.sum(axis=-1, keepdims=True)
    require(bool(np.isclose(totals, 1., rtol=0., atol=1e-12).all()), "unit probability rows " + name)
    return result / totals


def _prediction(value, shape, kind):
    if kind == "probabilities":
        p = _probabilities(value, shape, "prediction")
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        return p, logp
    require(kind == "logits" and isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype.kind in "fiu", "explicit finite logits or probabilities")
    logits = value.astype(np.float64, copy=True)
    require(bool(np.isfinite(logits).all()), "finite logits")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exponential = np.exp(shifted)
            logp = shifted - np.log(exponential.sum(axis=-1, keepdims=True))
            p = np.exp(logp)
    except FloatingPointError as error:
        raise ValueError("finite stable prediction arithmetic") from error
    require(bool(np.isfinite(logp).all()) and bool(np.isfinite(p).all()), "finite predicted distribution")
    return p, logp


def _mean(values):
    if not values:
        return None
    try:
        result = math.fsum(float(v) / len(values) for v in values)
    except (ValueError, OverflowError) as error:
        raise ValueError("finite diagnostic aggregation") from error
    require(math.isfinite(result), "finite diagnostic aggregation")
    return result


def _groups(rows, mask, case_ids, regimes, horizons):
    """Equal originating-case means; retain unsupported cases as None."""
    def summary(indices):
        cases = []
        for case in sorted({case_ids[i] for i in indices}):
            members = [i for i in indices if case_ids[i] == case]
            pairs = [(i, h - 1) for i in members for h in horizons if mask[i, h - 1]]
            cases.append({"case_id": case, "blocks": len(members), "rows": len(pairs),
                          **{name: _mean([values[i, t] for i, t in pairs]) for name, values in rows.items()}})
        supported = [row for row in cases if row["rows"]]
        return {"horizons": list(horizons), "declared_cases": len(cases), "blocks": len(indices),
                "available_rows": len(indices) * len(horizons), "scored_rows": sum(r["rows"] for r in cases),
                "supported_cases": len(supported),
                "unsupported_case_ids": [r["case_id"] for r in cases if not r["rows"]],
                **{"case_weighted_" + name: _mean([r[name] for r in supported]) for name in rows}, "cases": cases}
    return {"overall": summary(list(range(len(case_ids)))),
            "by_regime": {regime: summary([i for i, r in enumerate(regimes) if r == regime])
                          for regime in sorted(set(regimes))}}


def score(outcomes, outcome_predictions, predicted_costs, raw_costs, legal, *, oracle_probabilities,
          case_ids, regimes, family, fit_seed, condition, prediction_kind="logits"):
    """Return the original hard report plus oracle diagnostics, for [N,8] rows.

    Zero oracle masses contribute zero to entropy/KL. Observed labels must have
    positive oracle mass. In probability mode, a zero prediction on any positive
    oracle mass is explicitly rejected as infinite expected log loss. No epsilon
    clipping is used. Stable logits retain finite log probabilities even when
    their materialized probabilities underflow. Tiny negative KL from floating
    summation within 1e-12 is floored at zero and counted, never silently hidden.
    """
    report = hard.score(outcomes, outcome_predictions, predicted_costs, raw_costs, legal,
                        case_ids=case_ids, regimes=regimes, family=family, fit_seed=fit_seed,
                        condition=condition, prediction_kind=prediction_kind)
    n, horizon = outcomes.shape
    q = _probabilities(oracle_probabilities, (n, horizon, 5), "oracle")
    p, logp = _prediction(outcome_predictions, q.shape, prediction_kind)
    require(bool((np.take_along_axis(q, outcomes.astype(np.int64)[..., None], axis=-1) > 0).all()),
            "observed label must have positive oracle probability")
    if prediction_kind == "probabilities":
        require(bool((p[q > 0] > 0).all()), "positive oracle mass with zero prediction has infinite expected log loss")
    resolved = np.zeros((n, horizon), np.bool_)
    if condition == "normal":
        resolved[:, 1:] = np.maximum.accumulate(outcomes == 4, axis=1)[:, :-1]
        expected = np.zeros((int(resolved.sum()), 5), np.float64)
        expected[:, 4] = 1
        require(np.array_equal(q[resolved], expected) and np.array_equal(p[resolved], expected),
                "normal known-terminal rows require exact caller-supplied one-hot found")
    all_rows = np.ones_like(resolved)
    rows = {name: np.zeros((n, horizon), np.float64) for name in DIAGNOSTICS}
    roundoff = 0
    for i in range(n):
        for t in range(horizon):
            positive = [a for a in range(5) if q[i, t, a] > 0]
            entropy = -math.fsum(float(q[i, t, a]) * math.log(float(q[i, t, a])) for a in positive)
            expected_log = -math.fsum(float(q[i, t, a]) * float(logp[i, t, a]) for a in positive)
            kl = math.fsum(float(q[i, t, a]) * (math.log(float(q[i, t, a])) - float(logp[i, t, a])) for a in positive)
            require(math.isfinite(kl) and kl >= -1e-12, "finite nonnegative oracle KL within summation tolerance")
            roundoff += int(kl < 0)
            excess = math.fsum((float(p[i, t, a]) - float(q[i, t, a])) ** 2 for a in range(5))
            oracle_brier = math.fsum(float(q[i, t, a]) * (1 - float(q[i, t, a])) for a in range(5))
            label = int(outcomes[i, t])
            values = {"kl": max(kl, 0.), "brier_excess": excess, "expected_log_score": expected_log,
                "oracle_entropy": entropy, "expected_brier": excess + oracle_brier, "oracle_brier": oracle_brier,
                "sampled_log_score": -float(logp[i, t, label]),
                "sampled_brier": math.fsum((float(p[i, t, a]) - int(a == label)) ** 2 for a in range(5))}
            require(all(math.isfinite(v) and v >= -1e-12 for v in values.values()), "finite oracle diagnostic row")
            for name, value in values.items():
                rows[name][i, t] = value

    def panel(horizons):
        return {"all_rows": _groups(rows, all_rows, case_ids, regimes, horizons),
                "unresolved_rows": _groups(rows, ~resolved, case_ids, regimes, horizons)}

    report["oracle"] = {"version": VERSION,
        "target_sha256": hashlib.sha256(b"oracle-five-outcome-v1\0" + q.astype("<f8").tobytes()).hexdigest(),
        "unresolved_mask_sha256": hashlib.sha256((~resolved).tobytes()).hexdigest(),
        "resolved_rows": int(resolved.sum()), "unresolved_rows": int((~resolved).sum()),
        "kl_roundoff_corrections": roundoff, "kl_roundoff_tolerance": 1e-12,
        "per_horizon": {str(h): panel((h,)) for h in hard.HORIZONS},
        "groups": {name: panel(horizons) for name, horizons in hard.GROUPS.items()},
        "terminal_policy": "gap has no found feedback; normal shortcut only after observed found",
        "supplementary_only": True}
    return report


def evaluate_reports(reports, **options):
    """Retain the hard NLL/gap gate; oracle diagnostics never select a method."""
    result = hard.evaluate_reports(reports, **options)
    for condition in ("gap", "normal"):
        selected = [r for r in reports if r["condition"] == condition]
        reference = selected[0].get("oracle")
        require(type(reference) is dict and reference.get("version") == VERSION, "complete oracle diagnostics")
        for key in ("target_sha256", "unresolved_mask_sha256"):
            digest = reference.get(key)
            require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
                    "complete oracle target and resolution fingerprints")
        for report in selected:
            oracle = report.get("oracle")
            require(type(oracle) is dict and oracle.get("version") == VERSION
                    and oracle.get("supplementary_only") is True
                    and set(oracle.get("per_horizon", {})) == set(map(str, hard.HORIZONS))
                    and set(oracle.get("groups", {})) == set(hard.GROUPS)
                    and all(type(oracle.get(k)) is int and oracle[k] >= 0 for k in ("resolved_rows", "unresolved_rows"))
                    and oracle["resolved_rows"] + oracle["unresolved_rows"] == len(report["identity_manifest"]) * 8
                    and all(oracle.get(k) == reference.get(k) for k in
                            ("target_sha256", "unresolved_mask_sha256", "resolved_rows", "unresolved_rows", "terminal_policy")),
                    "all methods share paired oracle targets and causal resolution")
    result["version"] = "otto-belief-distillation-gate-v1"
    result["oracle_diagnostics_used_for_selection"] = False
    result["limitations"] = [text for text in result["limitations"] if "ridge" not in text]
    return result


def model_effect_error(original_oracle, alternate_oracle, predicted_original, predicted_alternate):
    """Owned per-row oracle signal S and signed model-effect error E.

    S=sum((p(A)-p(pi(A)))²); E=sum(((q(A)-q(pi(A)))-(p(A)-p(pi(A))))²).
    Probabilities have shape [N,H,5]. No ratio, support filtering or selection is
    defined, including when S=0. The caller binds paired actions and prefixes.
    """
    require(isinstance(original_oracle, np.ndarray) and original_oracle.ndim == 3
            and original_oracle.shape[0] > 0 and 1 <= original_oracle.shape[1] <= 8
            and original_oracle.shape[2] == 5, "paired five-class effect arrays")
    shape = original_oracle.shape
    p = _probabilities(original_oracle, shape, "original oracle effect")
    alternate_p = _probabilities(alternate_oracle, shape, "alternate oracle effect")
    q = _probabilities(predicted_original, shape, "original model effect")
    alternate_q = _probabilities(predicted_alternate, shape, "alternate model effect")
    oracle_effect = p - alternate_p
    difference = (q - alternate_q) - oracle_effect
    return {"oracle_signal": np.square(oracle_effect).sum(axis=-1),
            "model_effect_error": np.square(difference).sum(axis=-1)}


def action_sensitivity(original_oracle, alternate_oracle, *, case_ids, regimes, actions, alternate_actions,
                       predicted_original=None, predicted_alternate=None):
    """All-case gap-only oracle JS/TV and optional model-effect error.

    This checks the action-array transformation, not when actions were committed
    or how the oracle was produced. No observed labels are assigned to the
    alternate branch. All cases and horizons, including zero effects, remain.
    pi(a)=a XOR1 for every action. Both blocks start at the same supplied prefix;
    the caller establishes that provenance. It is not a normal-observation
    counterfactual that holds subsequent observations fixed. Shapes are [N,H,5]
    probabilities and [N,H] integer actions, for 1<=H<=8.
    """
    require(isinstance(original_oracle, np.ndarray) and original_oracle.ndim == 3
            and original_oracle.shape[0] > 0 and 1 <= original_oracle.shape[1] <= 8
            and original_oracle.shape[2] == 5, "complete oracle block distribution")
    n, horizon, _ = original_oracle.shape
    p = _probabilities(original_oracle, original_oracle.shape, "original action oracle")
    q = _probabilities(alternate_oracle, original_oracle.shape, "opposite action oracle")
    require(isinstance(actions, np.ndarray) and isinstance(alternate_actions, np.ndarray)
            and actions.shape == alternate_actions.shape == (n, horizon)
            and actions.dtype.kind in "iu" and alternate_actions.dtype.kind in "iu"
            and bool(((actions >= 0) & (actions < 4)).all())
            and np.array_equal(alternate_actions, np.bitwise_xor(actions, 1)), "fixed opposite-action mapping a XOR1")
    require((predicted_original is None) == (predicted_alternate is None), "both paired model predictions or neither")
    require(isinstance(case_ids, (list, tuple)) and isinstance(regimes, (list, tuple))
            and len(case_ids) == len(regimes) == n
            and all(type(v) is str and v.strip() for v in (*case_ids, *regimes)), "explicit all-case identities")
    require(all(len({r for c, r in zip(case_ids, regimes, strict=True) if c == case}) == 1 for case in set(case_ids)),
            "one regime per originating case")
    js, tv = np.zeros((n, horizon)), np.zeros((n, horizon))
    for i in range(n):
        for t in range(horizon):
            terms = []
            for a in range(5):
                left, right = float(p[i, t, a]), float(q[i, t, a])
                high, low = max(left, right), min(left, right)
                if high == 0:
                    continue
                # Log-space midpoint keeps subnormal probabilities valid.
                log_midpoint = math.log(high) + math.log1p(low / high) - math.log(2)
                terms.extend(v * (math.log(v) - log_midpoint) for v in (left, right) if v > 0)
            js[i, t] = max(0., .5 * math.fsum(terms))
            tv[i, t] = .5 * math.fsum(abs(float(p[i, t, a]) - float(q[i, t, a])) for a in range(5))
    mask = np.ones((n, horizon), np.bool_)
    rows = {"jensen_shannon": js, "total_variation": tv, "oracle_signal": np.square(p - q).sum(axis=-1)}
    if predicted_original is not None:
        rows.update(model_effect_error(p, q, predicted_original, predicted_alternate))
    return {"version": "otto-belief-action-sensitivity-v1", "horizons": list(range(1, horizon + 1)),
        "overall": _groups(rows, mask, case_ids, regimes, tuple(range(1, horizon + 1))),
        "per_horizon": {str(h): _groups(rows, mask, case_ids, regimes, (h,)) for h in range(1, horizon + 1)},
        "blocks": n, "case_filtering": False, "actions_verified_opposite": True,
        "condition": "gap", "action_mapping": "a XOR1", "model_effect_included": predicted_original is not None,
        "scope": "supplied paired oracle sensitivity; neither causal oracle validity nor model effectiveness is established"}
