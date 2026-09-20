"""Pure metrics for the prospectively fixed observation-learning factorial.

No files, models, training, corpus loading or probability repair. The outer
reporter must authenticate complete execution and the prepared canonical ledger
before calling these functions. Rows are in prepared DEV dialogue/source order,
with contiguous row_index and the required fields checked in _layout below.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

VERSION = "dialogue-observation-metrics-v1"
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
SEEDS = (6901, 6902, 6903)
PANELS = ("all", "seen", "unseen")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
BINS = (*STRATA[:2], "first_assignment", "revision", "clear")
TYPES = ("none", "dontcare", "true", "false", "other")
NONE, DONTCARE = "reserved:NOT_MENTIONED", "reserved:DONTCARE"
TOLERANCE = 2e-6
MAX_CANDIDATES = 12
RATE_KEYS = {"accuracy", "error", "nll", "brier", "false_positive_rate", "type_recall"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value):
    return type(value) is int and value >= 0


def _layout(rows):
    _require(type(rows) is list and rows, "Nonempty canonical DEV row ledger")
    n = len(rows)
    mask = np.zeros((n, MAX_CANDIDATES), bool)
    kinds = np.full((n, MAX_CANDIDATES), -1, np.int8)
    labels, bins, unseen, services = [], [], [], []
    source_keys, endpoint_keys, service_panels = set(), set(), {}
    schemas = {}
    for i, row in enumerate(rows):
        _require(type(row) is dict and row["row_index"] == i and _integer(row["row_index"])
                 and row["split"] == "dev", "Canonical contiguous DEV row order")
        did, service, slot = row["dialogue_id"], row["service"], row["slot"]
        _require(all(type(v) is str and v for v in (did, service, slot)), "Public row identity")
        _require(all(_integer(row[k]) for k in ("source_row_index", "time", "query_index")),
                 "Integer source-row/public-time/query indices")
        source_key = (row["split"], did, row["source_row_index"])
        endpoint_key = (row["split"], did, row["time"], row["query_index"])
        _require(source_key not in source_keys and endpoint_key not in endpoint_keys, "Duplicate endpoint identity")
        source_keys.add(source_key)
        endpoint_keys.add(endpoint_key)
        ids, values = row["candidate_ids"], row["candidate_values"]
        _require(type(ids) is list and type(values) is list and 3 <= len(ids) <= MAX_CANDIDATES
                 and len(values) == len(ids) and all(type(v) is str for v in ids)
                 and len(set(ids)) == len(ids) and ids.count(NONE) == ids.count(DONTCARE) == 1,
                 "Complete distinct canonical candidate support")
        for cid, value in zip(ids, values, strict=True):
            _require(value is None if cid in (NONE, DONTCARE) else
                     type(value) is str and value.strip() and cid == "value:" + value,
                     "Candidate identity/value binding")
        schema = (service, slot, tuple(ids), tuple(values))
        qi = row["query_index"]
        _require(schemas.setdefault(qi, schema) == schema, "Inconsistent query schema")
        ordinary = {v.strip().casefold() for v in values if v is not None}
        boolean = ordinary == {"true", "false"}
        for j, (cid, value) in enumerate(zip(ids, values, strict=True)):
            kind = ("none" if cid == NONE else "dontcare" if cid == DONTCARE else
                    value.strip().casefold() if boolean else "other")
            kinds[i, j] = TYPES.index(kind)
        mask[i, :len(ids)] = True
        label = row["label_index"]
        _require(_integer(label) and label < len(ids) and row["label_id"] == ids[label], "Canonical target binding")
        bin_name = row["bin"]
        _require(bin_name in BINS and type(row["unseen"]) is bool, "Transition/panel metadata")
        if "stratum" in row:
            _require(row["stratum"] == (bin_name if bin_name in STRATA[:2] else "changed"), "Stratum differs")
        if "dontcare" in row:
            _require(type(row["dontcare"]) is bool and row["dontcare"] == (ids[label] == DONTCARE),
                     "DONTCARE metadata differs")
        _require(service_panels.setdefault(service, row["unseen"]) == row["unseen"], "Inconsistent service panel")
        labels.append(label)
        bins.append(bin_name)
        unseen.append(row["unseen"])
        services.append(service)
    return {"mask": mask, "kinds": kinds, "labels": np.asarray(labels, np.int64),
            "bins": np.asarray(bins), "unseen": np.asarray(unseen, bool),
            "services": np.asarray(services), "service_panels": service_panels}


def _validate(logs, indices, layout):
    n, mask = len(layout["labels"]), layout["mask"]
    _require(isinstance(indices, np.ndarray) and indices.dtype == np.int64
             and np.array_equal(indices, np.arange(n, dtype=np.int64)), "Saved row indices differ from canonical order")
    _require(isinstance(logs, np.ndarray) and logs.dtype == np.float32
             and logs.shape == (n, MAX_CANDIDATES), "Saved float32 log-probability shape")
    _require(np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(),
             "Supported nonfinite log probability or incorrect padded support: technical failure")
    _require(float(logs[mask].max()) <= math.log1p(TOLERANCE), "Probability exceeds allowed total mass")
    probabilities = np.exp(logs.astype(np.float64))
    errors = np.abs(probabilities.sum(-1) - 1.)
    _require(bool(np.all(errors <= TOLERANCE)), "Saved distribution mass: technical failure")
    ties = (logs == logs.max(-1, keepdims=True)).sum(-1) > 1
    return {"rows": n, "supported_candidate_positions": int(mask.sum()),
            "maximum_mass_error": float(errors.max()), "tolerance": TOLERANCE,
            "exact_top1_tie_rows": int(ties.sum()),
            "float64_exponent_underflow_positions": int((probabilities[mask] == 0).sum()),
            "policy": "Finite supported saved logs; direct log NLL; no clipping/floor/renormalization"}, probabilities


def validate_log_predictions(log_probs, row_indices, canonical_rows):
    """Validate raw saved arrays; reject supported -inf rather than drop rows."""
    return _validate(log_probs, row_indices, _layout(canonical_rows))[0]


def _mean(values):
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def _grouped(layout, choices, losses=None):
    labels, bins = layout["labels"], layout["bins"]
    correct = choices == labels
    target_types = layout["kinds"][np.arange(len(labels)), labels]
    predicted_types = layout["kinds"][np.arange(len(labels)), choices]

    def cell(selected):
        count, hits = int(selected.sum()), int((selected & correct).sum())
        result = {"count": count, "correct": hits, "incorrect": count - hits,
                  "accuracy": hits / count if count else None, "error": (count - hits) / count if count else None}
        if losses is not None:
            result.update({k: math.fsum(map(float, v[selected])) / count if count else None
                           for k, v in losses.items()})
        return result

    def panel(selected):
        by_bin = {name: cell(selected & (bins == name)) for name in BINS}
        strata = {name: by_bin[name] for name in STRATA[:2]}
        strata["changed"] = cell(selected & np.isin(bins, BINS[2:]))
        types = {}
        for code, name in enumerate(TYPES):
            target = target_types == code
            predicted = predicted_types == code
            supported = (layout["kinds"] == code).any(-1)
            target_count = int((selected & target).sum())
            negative_count = int((selected & supported & ~target).sum())
            tp = int((selected & target & predicted).sum())
            fp = int((selected & ~target & predicted).sum())
            types[name] = {**cell(selected & target), "supported_rows": int((selected & supported).sum()),
                           "predicted_support": int((selected & predicted).sum()), "true_positives": tp,
                           "false_positives": fp, "false_positive_denominator": negative_count,
                           "false_positive_rate": fp / negative_count if negative_count else None,
                           "type_recall": tp / target_count if target_count else None,
                           "bins": {b: cell(selected & target & (bins == b)) for b in BINS},
                           "strata": {s: cell(selected & target & ((bins == s) if s in STRATA[:2]
                                                                  else np.isin(bins, BINS[2:]))) for s in STRATA}}
        names = ("accuracy", "nll", "brier") if losses is not None else ("accuracy",)
        return {"micro": cell(selected), "strata": strata, "bins": by_bin,
                "macro_three": {k: _mean([strata[s][k] for s in STRATA]) for k in names}, "types": types}

    panels = {name: panel(np.ones(len(labels), bool) if name == "all" else
                          layout["unseen"] if name == "unseen" else ~layout["unseen"]) for name in PANELS}
    services = {name: {"unseen": layout["service_panels"][name],
                       "metrics": panel(layout["services"] == name)} for name in sorted(layout["service_panels"])}
    return {"panels": panels, "services": services}


def score_panels(log_probs, canonical_rows, *, row_indices=None):
    """All panel/bin/service/type metrics on validated raw saved log scores.

    Pass saved row_indices explicitly at the reporting boundary. Omitting them
    is only suitable for arrays already joined to the supplied canonical rows.
    Type recall is binary type detection; target accuracy remains exact-ID
    correctness even if distinct catalog candidates share a Boolean spelling.
    """
    layout = _layout(canonical_rows)
    if row_indices is None:
        row_indices = np.arange(len(canonical_rows), dtype=np.int64)
    validation, probabilities = _validate(log_probs, row_indices, layout)
    labels = layout["labels"]
    choices = log_probs.argmax(-1)
    residual = probabilities.copy()
    residual[np.arange(len(labels)), labels] -= 1.
    losses = {"nll": -log_probs[np.arange(len(labels)), labels].astype(np.float64),
              "brier": np.square(residual).sum(-1)}
    return {**_grouped(layout, choices, losses), "validation": validation}


def literal_metrics(choices, row_indices, canonical_rows):
    """Deterministic original/number literal registers: categorical metrics only."""
    layout = _layout(canonical_rows)
    n = len(canonical_rows)
    _require(isinstance(row_indices, np.ndarray) and row_indices.dtype == np.int64
             and np.array_equal(row_indices, np.arange(n, dtype=np.int64)), "Literal row alignment")
    _require(isinstance(choices, np.ndarray) and choices.dtype == np.int64 and choices.shape == (n,)
             and ((choices >= 0) & (choices < layout["mask"].sum(-1))).all(), "Literal candidate support")
    _require(not (layout["kinds"][np.arange(n), choices] == TYPES.index("dontcare")).any(),
             "Literal register cannot select reserved DONTCARE")
    return {**_grouped(layout, choices), "scope": "Deterministic literal carry, no probabilistic forecast"}


def _fit_names():
    return {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}


def _check_fits(fits):
    _require(type(fits) is dict and set(fits) == _fit_names(), "Exactly all four arms by three seeds required")
    support = None
    for fit in fits.values():
        counts = []
        for panel in PANELS:
            for group in (fit["panels"][panel]["micro"],
                          *(fit["panels"][panel]["strata"][s] for s in STRATA)):
                _require(_integer(group["count"]) and _integer(group["correct"])
                         and group["correct"] <= group["count"], "Integer metric counts")
                counts.append(group["count"])
        _require(fit["validation"]["rows"] == fit["panels"]["all"]["micro"]["count"], "Fit row coverage")
        _require(support is None or support == counts, "All twelve fits require identical panel/stratum support")
        support = counts


def _metric_tree(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if key in RATE_KEYS:
            result[key] = item
        elif isinstance(item, dict):
            child = _metric_tree(item)
            if child:
                result[key] = child
    return result


def _combine(trees, operation):
    first = trees[0]
    if isinstance(first, dict):
        _require(all(isinstance(t, dict) and set(t) == set(first) for t in trees), "Metric tree mismatch")
        return {k: _combine([t[k] for t in trees], operation) for k in first}
    return None if any(t is None for t in trees) else operation(trees)


def factorial_summary(fits):
    """Descriptive equal-seed means and paired difference-of-differences.

    Integer supports remain in per-fit results, not averaged into fractional
    counts. Positive accuracy is favorable; positive loss/error is unfavorable.
    Undefined subgroup metrics remain undefined throughout the contrasts.
    """
    _check_fits(fits)
    metrics = {name: _metric_tree(fit) for name, fit in fits.items()}
    families = {arm: {"fit_names": [f"{arm}-{seed}" for seed in SEEDS],
                      "mean": _combine([metrics[f"{arm}-{seed}"] for seed in SEEDS], _mean)} for arm in ARMS}
    pairs = {"encoder_numbers": ("trainable_numbers", "frozen_numbers"),
             "encoder_original": ("trainable_original", "frozen_original"),
             "numbers_frozen": ("frozen_numbers", "frozen_original"),
             "numbers_trainable": ("trainable_numbers", "trainable_original")}
    contrasts = {}
    for name, (new, old) in pairs.items():
        seeds = {str(seed): _combine([metrics[f"{new}-{seed}"], metrics[f"{old}-{seed}"]],
                                    lambda v: v[0] - v[1]) for seed in SEEDS}
        contrasts[name] = {"definition": f"{new} minus {old}", "seeds": seeds,
                           "mean": _combine(list(seeds.values()), _mean)}
    interaction = {str(seed): _combine([contrasts["encoder_numbers"]["seeds"][str(seed)],
                                       contrasts["encoder_original"]["seeds"][str(seed)]],
                                      lambda v: v[0] - v[1]) for seed in SEEDS}
    return {"families": families, "contrasts": contrasts,
            "interaction": {"definition": "(trainable_numbers-frozen_numbers)-(trainable_original-frozen_original)",
                            "seeds": interaction, "mean": _combine(list(interaction.values()), _mean)},
            "scope": "Descriptive factorial effects; not another continuation rule or significance test"}


def _macro(panel):
    strata = panel["strata"]
    if any(strata[name]["count"] == 0 for name in STRATA):
        return None
    return sum((Fraction(strata[name]["correct"], strata[name]["count"]) for name in STRATA), Fraction()) / 3


def criteria(fits):
    """Seven fixed scientific checks; complete-fit membership is separate.

    This pure function cannot authenticate execution, weights or update
    witnesses. The outer reporter must establish that technical prerequisite.
    """
    _check_fits(fits)
    checks = []

    def add(name, values, threshold, comparison, arithmetic):
        present = values is not None and all(v is not None for v in values)
        mean = None if not present else (sum(values, Fraction()) / 3 if arithmetic == "exact" else _mean(values))
        passed = present and (mean >= threshold if comparison == "ge" else mean <= threshold)
        checks.append({"name": name, "passed": bool(passed),
                       "paired_seed_values": None if values is None else [None if v is None else float(v) for v in values],
                       "mean": None if mean is None else float(mean), "threshold": float(threshold),
                       "comparison": comparison, "arithmetic": arithmetic})

    def deltas(panel, getter):
        result = []
        for seed in SEEDS:
            a = getter(fits[f"trainable_numbers-{seed}"]["panels"][panel])
            b = getter(fits[f"frozen_numbers-{seed}"]["panels"][panel])
            result.append(None if a is None or b is None else a - b)
        return result

    unseen = deltas("unseen", _macro)
    add("unseen_macro_gain_1pp", unseen, Fraction(1, 100), "ge", "exact")
    wins = None if any(v is None for v in unseen) else sum(v > 0 for v in unseen)
    checks.append({"name": "unseen_macro_strict_paired_wins", "passed": wins is not None and wins >= 2,
                   "wins": wins, "required": 2, "arithmetic": "exact"})
    for score in ("nll", "brier"):
        primary = [fits[f"trainable_numbers-{s}"]["panels"]["unseen"]["micro"][score] for s in SEEDS]
        control = [fits[f"frozen_numbers-{s}"]["panels"]["unseen"]["micro"][score] for s in SEEDS]
        present = all(type(v) in (int, float) and math.isfinite(v) for v in primary + control)
        a, b = (_mean(primary), _mean(control)) if present else (None, None)
        checks.append({"name": f"unseen_micro_{score}_nonworse", "passed": present and a <= b,
                       "paired_seed_values": [x - y for x, y in zip(primary, control, strict=True)] if present else None,
                       "mean": a - b if present else None, "primary_mean": a, "control_mean": b,
                       "threshold": 0., "comparison": "le", "arithmetic": "float64 arm means, no epsilon"})
    add("seen_macro_deficit_at_most_1pp", deltas("seen", _macro), -Fraction(1, 100), "ge", "exact")

    def error(panel):
        item = panel["strata"]["assigned_retention"]
        return Fraction(item["count"] - item["correct"], item["count"]) if item["count"] else None

    for panel in ("seen", "unseen"):
        add(f"{panel}_assigned_retention_error_increase_at_most_half_pp", deltas(panel, error),
            Fraction(1, 200), "le", "exact")
    return {"passed": all(check["passed"] for check in checks), "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": 7, "checks": checks, "complete_fit_membership": True,
            "technical_validity_scope": "Membership only here; outer reporter must authenticate complete valid execution"}
