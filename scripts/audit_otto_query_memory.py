"""Independent DEV saved-output audit, with no model or optimizer execution.

Only standard-library metadata is read until all source, original process,
runtime, payload, checkpoint and pre-DEV barrier identities are authenticated.
Independent scalar arithmetic then reconstructs observed schedules, metrics,
support and the fixed gate. Saved slow/fork/no-write equality is checked bitwise.
Source-authenticated causality is not an independent numerical model replay.
An audit worker's result is conditional on its own original supervisor closure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import struct
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_query_memory.py"
TEST = "tests/test_audit_otto_query_memory.py"
NUMERICAL_TEST = "tests/test_otto_query_memory_audit_numerics.py"
NEW_COMPONENTS = {SELF, TEST, NUMERICAL_TEST}
PRODUCER = "scripts/train_otto_query_memory.py"
VERSION = "otto-query-memory-dev-saved-audit-v1"
PRODUCER_VERSION = "otto-query-memory-training-v1"
METRIC_VERSION = "otto-query-memory-metrics-v1"
GATE_VERSION = "otto-query-memory-gate-v1"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
LIMITS = {"seconds": 600, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
SEEDS = (309000001, 309000002, 309000003)
FITS = ("pretrained", "joint_aux", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
VIEWS = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled", "trace_no_write")
CONTROLS = ("pretrained", "last_error", "instant_delta", "joint_aux")
ID_FIELDS = ("stage", "episode_id", "episode_index", "seed", "case", "regime", "arm")
METRIC_NAMES = ("agreement", "raw_gap", "first_argmin_match", "centered_mse")
SLOW_SHAPES = {"recurrent.weight_ih_l0": (84, 40), "recurrent.weight_hh_l0": (84, 28),
               "recurrent.bias_ih_l0": (84,), "recurrent.bias_hh_l0": (84,),
               "output.weight": (4, 28), "output.bias": (4,),
               "action_residual.weight": (4, 28), "action_residual.bias": (4,)}
PREDICTION_FIELDS = {"action_prediction", "slow_action_prediction", "base_prediction", "shadow_prior",
                     "corrected_shadow_prior", "prior_mask", "prewrite_correction", "episode_offsets"}
LIMITATIONS = [
    "No model, optimizer, teacher or simulator is called; checkpoint inference is not numerically replayed.",
    "Causal input schedules and source/qualification lineage are authenticated; hidden-state and write causality remain source-tested execution claims.",
    "Scalar metrics, complete-path support, gate arithmetic and saved frozen/fork/no-write equality are independently reconstructed.",
    "Teacher/native numerical truth, actual gradients and physical timings remain authenticated producer evidence.",
    "Decision metrics and scheduled work are recomputed; training optimization and canonical float32 AUX losses are not numerically replayed.",
    "A passed worker gate cannot admit TEST until this audit's original supervisor closes successfully.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained evidence")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def centered_mse(predicted, target, allowed):
    left = [float(predicted[a]) for a in allowed]
    right = [float(target[a]) for a in allowed]
    lm, rm = math.fsum(left) / len(left), math.fsum(right) / len(right)
    return math.fsum(((p - lm) - (q - rm)) ** 2 for p, q in zip(left, right, strict=True)) / len(left)


def scalar_row(target, legal, scores):
    """Independent Python/struct implementation of the deployment f32 tie rule."""
    allowed = [action for action in range(4) if bool(legal[action])]
    require(bool(allowed), "legal teacher-score support")
    pmin = min(float(scores[a]) for a in allowed)
    qmin = min(float(target[a]) for a in allowed)
    chosen = next(a for a in allowed if f32(float(scores[a]) - pmin) < f32(1e-10))
    oracle = [a for a in allowed if f32(float(target[a]) - qmin) < f32(1e-10)]
    return (float(chosen in oracle), float(target[chosen]) - qmin,
            float(chosen == oracle[0]), centered_mse(scores, target, allowed))


def scalar_report(np, identities, target, legal, predictions, prior, offsets, family, seed, *, check=lambda: None):
    """Rebuild the entire P4 TRAIN/DEV report without importing producer metrics.

    Arrays must already be authenticated and validated by the caller. All raw
    costs are converted to Python float; fsum preserves declared aggregation
    order. Case means retain each collector and each unsupported path.
    """
    require(identities and identities[0]["stage"] in ("train", "dev"), "only TRAIN/DEV scalar audit")
    order = sorted(range(len(identities)), key=lambda i: identities[i]["episode_index"])
    records = []
    for index in order:
        check()
        identity = {key: identities[index][key] for key in ID_FIELDS}
        low, high = map(int, offsets[index:index + 2])
        row_values, prior_values = [], []
        for absolute in range(low, high):
            step = absolute - low
            if step % 4:
                row_values.append((step, scalar_row(target[absolute], legal[absolute], predictions[absolute])))
            elif step:
                prior_values.append(centered_mse(prior[absolute], target[absolute], (0, 1, 2, 3)))
        bins = {}
        for scope in ("full", "initial", "later"):
            selected = [(step, values) for step, values in row_values
                        if scope == "full" or (step < 4 if scope == "initial" else step >= 5)]
            for age in (None, 1, 2, 3):
                values = [row for step, row in selected if age is None or step % 4 == age]
                bins[scope, age] = {"rows": len(values), "sums": {
                    name: math.fsum(row[column] for row in values) for column, name in enumerate(METRIC_NAMES)}}
        bins["prior", None] = {"rows": len(prior_values), "sums": {"centered_mse": math.fsum(prior_values)}}
        digest = hashlib.sha256(b"otto-query-memory-target-v1\0")
        digest.update((high - low).to_bytes(8, "little"))
        digest.update(target[low:high].astype("<f4", copy=False).tobytes(order="C"))
        digest.update(legal[low:high].tobytes(order="C"))
        records.append({"identity": identity, "length": high - low, "target_sha256": digest.hexdigest(), "bins": bins})

    def summary(indices, scope, age=None):
        leaves = [records[i]["bins"][scope, age] for i in indices]
        ids = [records[i]["identity"] for i in indices]
        cases = sorted({(row["regime"], row["case"]) for row in ids})
        supported = [i for i, leaf in enumerate(leaves) if leaf["rows"]]
        supported_cases = {(ids[i]["regime"], ids[i]["case"]) for i in supported}
        names = ("centered_mse",) if scope == "prior" else METRIC_NAMES
        sums = {name: math.fsum(leaf["sums"][name] for leaf in leaves) for name in names}
        rows = sum(leaf["rows"] for leaf in leaves)
        value = {"episodes": len(indices), "supported_episodes": len(supported),
            "zero_support_episode_ids": [row["episode_id"] for row, leaf in zip(ids, leaves, strict=True) if not leaf["rows"]],
            "rows": rows, "weight_mass": len(supported) / len(indices), "declared_case_count": len(cases),
            "supported_case_count": len(supported_cases), "raw_sums": sums,
            "supported_cases": [{"regime": r, "case": c} for r, c in sorted(supported_cases)],
            "zero_support_cases": [{"regime": r, "case": c} for r, c in sorted(set(cases) - supported_cases)]}
        for name in names:
            means = [leaf["sums"][name] / leaf["rows"] if leaf["rows"] else 0. for leaf in leaves]
            total = math.fsum(means)
            value["episode_weighted_" + name] = total / len(indices)
            value["supported_episode_" + name] = total / len(supported) if supported else None
            value["row_weighted_" + name] = sums[name] / rows if rows else None
            by_case = []
            for case in cases:
                members = [i for i, row in enumerate(ids) if (row["regime"], row["case"]) == case]
                by_case.append(math.fsum(means[i] for i in members) / len(members))
            value["case_weighted_" + name] = math.fsum(by_case) / len(cases)
        if scope != "prior" and age is None:
            value["by_age"] = {str(a): summary(indices, scope, a) for a in (1, 2, 3)}
        return value

    def groups(scope):
        indices = list(range(len(records)))
        ids = [r["identity"] for r in records]
        return {"overall": summary(indices, scope),
            "by_regime": {regime: summary([i for i in indices if ids[i]["regime"] == regime], scope)
                          for regime in sorted({r["regime"] for r in ids})},
            "by_collector": {arm: summary([i for i in indices if ids[i]["arm"] == arm], scope)
                             for arm in sorted({r["arm"] for r in ids})},
            "by_case": [{"regime": regime, "case": case, **summary(
                [i for i in indices if (ids[i]["regime"], ids[i]["case"]) == (regime, case)], scope)}
                for regime, case in sorted({(r["regime"], r["case"]) for r in ids})]}

    return {"version": METRIC_VERSION, "scope": "teacher-score imitation on fixed collector paths; no autonomous efficacy",
        "family": family, "seed": seed, "query_period": 4, "stage": identities[0]["stage"], "episodes": len(records),
        "identity_manifest": [{**r["identity"], "length": r["length"], "target_sha256": r["target_sha256"]} for r in records],
        "scopes": {scope: groups(scope) for scope in ("full", "initial", "later")}, "prequery": groups("prior")}


def independent_gate(reports, *, technical_complete=False):
    require(type(technical_complete) is bool, "strict technical completion Boolean")
    require(len(reports) == 24 and {(r["family"], r["seed"]) for r in reports}
            == {(view, seed) for view in VIEWS for seed in SEEDS}, "all independent DEV model reports")
    require(all(r["stage"] == "dev" and r["query_period"] == 4 and r["episodes"] == 18 for r in reports), "complete DEV P4")
    by = {(r["family"], r["seed"]): r for r in reports}

    def leaf(family, seed, scope, regime):
        value = by[family, seed]["scopes"][scope]["by_regime"][regime]
        require(value["episodes"] == 9 and value["declared_case_count"] == 3, "complete DEV cases")
        gap = value["case_weighted_raw_gap"]
        require(type(gap) in (int, float) and math.isfinite(gap) and gap >= 0, "finite independent legal gap")
        return value

    def mean(family, scope, regime):
        return math.fsum(leaf(family, seed, scope, regime)["case_weighted_raw_gap"] for seed in SEEDS) / 3

    result = [{"name": "technical_completion", "passed": technical_complete}]
    for regime in ("lambda3", "lambda4"):
        prefix = regime + ":P4"
        supports = {leaf(family, seed, "later", regime)["supported_case_count"] for family in VIEWS for seed in SEEDS}
        require(len(supports) == 1, "identical independently reconstructed case support")
        support = supports.pop()
        require(type(support) is int and 0 <= support <= 3, "bounded independent case support")
        result.append({"name": prefix + ":supported_cases", "passed": support >= 2, "actual": support, "required": 2})
        candidate = mean("trace_delta", "later", regime)
        controls = {family: mean(family, "later", regime) for family in CONTROLS}
        best = min(controls.values())
        result.append({"name": prefix + ":later_gap_10pct", "passed": candidate <= .9 * best and candidate < best,
                       "candidate": candidate, "controls": controls, "best_control": best})
        candidate = mean("trace_delta", "full", regime)
        controls = {family: mean(family, "full", regime) for family in CONTROLS}
        result.append({"name": prefix + ":full_gap_nonregression", "passed": candidate <= min(controls.values()),
                       "candidate": candidate, "controls": controls})
        for seed in SEEDS:
            candidate = leaf("trace_delta", seed, "later", regime)["case_weighted_raw_gap"]
            controls = {family: leaf(family, seed, "later", regime)["case_weighted_raw_gap"] for family in CONTROLS}
            result.append({"name": prefix + f":seed_{seed}_nonregression", "passed": candidate <= min(controls.values()),
                           "candidate": candidate, "controls": controls})
    require(len(result) == 13 and len({r["name"] for r in result}) == 13, "exact independent DEV conjunction")
    return {"version": GATE_VERSION, "stage": "dev", "candidate": "trace_delta", "technical_complete": technical_complete,
            "conditions": result, "passed_conditions": sum(row["passed"] for row in result), "total_conditions": 13,
            "passed": all(row["passed"] for row in result),
            "scope": "prospective mechanism screen; no autonomous, total-compute or novelty claim"}


def tensor_witness(arrays, *, slow_only=False):
    selected = {name.removeprefix("slow.") if slow_only else name: value for name, value in arrays.items()
                if not slow_only or name.startswith("slow.")}
    digest, result = hashlib.sha256(), {}
    for name in sorted(selected):
        value = selected[name]
        raw = value.tobytes()
        result[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "shape": list(value.shape),
                        "dtype": "torch.float32", "bytes": len(raw)}
        digest.update(name.encode() + b"\0" + raw)
    return {"sha256": digest.hexdigest(), "tensors": result}


def expected_history(np, flat, identities):
    """Independent complete P4 observations and fixed-denominator weights."""
    count = len(identities)
    offsets = flat["episode_offsets"]
    require(offsets.dtype == np.int64 and offsets.shape == (count + 1,) and offsets[0] == 0, "complete census offsets")
    lengths = np.diff(offsets)
    require(bool(((lengths >= 1) & (lengths <= 2188)).all()), "bounded full episode lengths")
    total = int(offsets[-1])
    expected = {"features": (np.float32, (total, 31)), "raw_q": (np.float32, (total, 4)),
                "legal": (np.bool_, (total, 4)), "actions": (np.int64, (total,)), "correction": (np.bool_, (total,))}
    require(set(flat) == set(expected) | {"episode_offsets"}, "exact six census arrays")
    for name, (dtype, shape) in expected.items():
        require(flat[name].dtype == dtype and flat[name].shape == shape and bool(np.isfinite(flat[name]).all()),
                "finite typed census " + name)
    require(bool(flat["legal"].any(-1).all()) and bool(((flat["actions"] >= 0) & (flat["actions"] < 4)).all())
            and bool(flat["legal"][np.arange(total), flat["actions"]].all()), "legal collector actions")
    query = np.zeros(total, np.bool_)
    prior = np.zeros(total, np.bool_)
    weight = np.zeros(total, np.float64)
    prior_weight = np.zeros(total, np.float64)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        local_query = steps % 4 == 0
        local_prior = local_query & (steps > 0)
        require(np.array_equal(flat["features"][low:high, 15], (steps / 2188).astype(np.float32))
                and np.array_equal(flat["features"][low:high, 16], (steps % 4 / 2188).astype(np.float32))
                and bool((flat["features"][low:high, 17] == 1).all()), "public observation clocks")
        query[low:high], prior[low:high] = local_query, local_prior
        nonquery_count, prior_count = int((~local_query).sum()), int(local_prior.sum())
        if nonquery_count:
            weight[low:high][~local_query] = 1 / (count * nonquery_count)
        if prior_count:
            prior_weight[low:high][local_prior] = 1 / (count * prior_count)
    require(np.array_equal(flat["correction"], query), "actual collector P4 query schedule")
    visible = np.full((total, 4), np.nan, np.float32)
    visible[query] = flat["raw_q"][query]
    require(np.array_equal(visible[query] / np.float32(64) * np.float32(64), visible[query]), "exact answer scale roundtrip")
    return {"features": flat["features"], "query_scores": visible, "targets": flat["raw_q"], "legal": flat["legal"],
            "actions": flat["actions"], "episode_offsets": offsets, "query_mask": query, "prior_mask": prior,
            "nonquery_weights": weight, "prior_weights": prior_weight}


def verify_prediction(np, saved, history, view, reference=None):
    """Saved-output algebra/bit checks; no reconstruction of hidden inference."""
    total = len(history["targets"])
    require(set(saved) == PREDICTION_FIELDS, "exact eight saved prediction fields")
    for name, value in saved.items():
        dtype, shape = (np.int64, history["episode_offsets"].shape) if name == "episode_offsets" else (
            (np.bool_, (total,)) if name == "prior_mask" else (np.float32, (total, 4)))
        require(value.dtype == dtype and value.shape == shape and bool(np.isfinite(value).all()), "finite saved field " + name)
    require(saved["episode_offsets"].tobytes() == history["episode_offsets"].tobytes()
            and np.array_equal(saved["prior_mask"], history["prior_mask"]), "complete prediction path/prior support")
    query, prior = history["query_mask"], history["prior_mask"]
    for name in ("action_prediction", "slow_action_prediction", "base_prediction"):
        require(saved[name][query].tobytes() == history["query_scores"][query].tobytes(), "query outputs copy only actual answers")
    for name in ("shadow_prior", "corrected_shadow_prior"):
        require(saved[name][~prior].tobytes() == np.zeros_like(saved[name][~prior]).tobytes(), "positive-zero inactive priors")
    first = history["episode_offsets"][:-1]
    require(saved["prewrite_correction"][first].tobytes() == np.zeros((len(first), 4), np.float32).tobytes(),
            "first query performs no correction")
    if view in ("pretrained", "joint_aux", "trace_no_write"):
        require(saved["prewrite_correction"].tobytes() == np.zeros_like(saved["prewrite_correction"]).tobytes(),
                "no-memory/no-write correction is zero")
        require(saved["action_prediction"].tobytes() == saved["slow_action_prediction"].tobytes()
                and saved["corrected_shadow_prior"].tobytes() == saved["shadow_prior"].tobytes(), "exact baseline/no-write action and prior")
    else:
        expected_action = saved["slow_action_prediction"][~query] + np.float32(64) * saved["prewrite_correction"][~query]
        expected_prior = saved["shadow_prior"][prior] + np.float32(64) * saved["prewrite_correction"][prior]
        require(saved["action_prediction"][~query].tobytes() == expected_action.tobytes()
                and saved["corrected_shadow_prior"][prior].tobytes() == expected_prior.tobytes(), "same prewrite correction applied to saved action/prior")
    if reference is not None and view != "joint_aux":
        fields = ("base_prediction", "slow_action_prediction", "shadow_prior", "prior_mask", "episode_offsets")
        if view == "trace_no_write":
            fields += ("action_prediction", "corrected_shadow_prior", "prewrite_correction")
        require(all(saved[name].tobytes() == reference[name].tobytes() for name in fields), "all frozen/no-write fields equal canonical pretraining")


def close_equal(actual, expected, label):
    """Exact structural/scalar comparison for independently rebuilt metrics."""
    require(type(actual) is type(expected), label + ": exact value type")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), label + ": exact keys")
        for key in expected:
            close_equal(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), label + ": list length")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            close_equal(left, right, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(math.isfinite(actual) and actual == expected, label + ": exact finite scalar")
    else:
        require(actual == expected, label + ": exact value")


def add_counts(destination, values):
    require(isinstance(values, dict) and all(type(value) is int and value >= 0 for value in values.values()),
            "finite nonnegative integer work")
    for key, value in values.items():
        destination[key] = destination.get(key, 0) + value


def scheduled_work(lengths, start, view):
    """Exact qualified P4/32 operation geometry, without recurrent execution."""
    require(start >= 0 and start % 32 == 0 and all(type(n) is int and 0 <= n <= 32 for n in lengths),
            "fixed chronological chunk geometry")
    active = sum(lengths)
    queries = sum((n + 3) // 4 for n in lengths)
    first = sum(n > 0 for n in lengths) if start == 0 else 0
    later, keys, nonquery = queries - first, active - first, active - queries
    query_groups = sum(any(n > offset for n in lengths) for offset in range(0, 32, 4))
    first_groups = int(start == 0 and any(lengths))
    later_groups = query_groups - first_groups
    ordinary_groups = sum(len({min(3, n - offset - 1) for n in lengths if n > offset + 1})
                          for offset in range(0, 32, 4))
    slow = {"recurrent_calls": first_groups + 2 * later_groups + ordinary_groups,
        "recurrent_token_transitions": active + later, "base_readout_calls": later_groups + ordinary_groups,
        "base_readout_rows": keys, "action_readout_calls": ordinary_groups, "action_readout_rows": nonquery,
        "shadow_readout_calls": later_groups, "shadow_readout_rows": later, "active_rows": active,
        "query_rows": queries, "later_query_rows": later, "nonquery_rows": nonquery, "key_rows": keys}
    matrix = view in ("instant_delta", "trace_delta", "trace_additive", "trace_scrambled", "trace_no_write")
    last = view == "last_error"
    projection_calls = max(0, max(lengths, default=0) - int(start == 0)) if matrix else 0
    projection_rows = len(lengths) * projection_calls
    counters = {"active_steps": active, "query_steps": queries, "key_steps": keys, "eligible_write_steps": later,
        "key_normalizations": keys if matrix else 0, "cue_normalizations": keys if matrix and view != "instant_delta" else 0,
        "trace_updates": keys if matrix else 0, "past_trace_rotations": keys if view == "trace_scrambled" else 0,
        "matrix_decays": keys if matrix else 0, "matrix_reads": keys if matrix else 0,
        "matrix_writes": later if matrix and view != "trace_no_write" else 0,
        "innovation_calculations": later if last or (matrix and view != "trace_no_write") else 0,
        "last_error_decays": keys if last else 0, "last_error_reads": keys if last else 0,
        "last_error_writes": later if last else 0, "action_corrections": nonquery if matrix or last else 0}
    units = {"normalized_coordinates": 8 * (counters["key_normalizations"] + counters["cue_normalizations"]),
        "trace_mixed_coordinates": 8 * counters["trace_updates"],
        "past_trace_permuted_coordinates": 8 * counters["past_trace_rotations"],
        "matrix_decayed_coordinates": 32 * counters["matrix_decays"], "matrix_read_terms": 32 * counters["matrix_reads"],
        "matrix_write_terms": 32 * counters["matrix_writes"], "innovation_coordinates": 4 * counters["innovation_calculations"],
        "last_error_decayed_coordinates": 4 * counters["last_error_decays"],
        "last_error_read_coordinates": 4 * counters["last_error_reads"],
        "last_error_written_coordinates": 4 * counters["last_error_writes"],
        "action_corrected_coordinates": 4 * counters["action_corrections"]}
    work = {"slow_" + name: value for name, value in slow.items()}
    work.update({"memory_" + name: value for name, value in counters.items()})
    work.update(projection_calls=projection_calls, projection_rows=projection_rows,
                projection_key_rows=keys if matrix else 0, projection_linear_terms=projection_rows * 28 * 8)
    return work, units


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.plan = None
        self.authenticated = False
        self.receipt = {"version": VERSION, "status": "started", "complete": False, "pending": None,
            "model_calls": 0, "optimizer_calls": 0, "teacher_calls": 0, "simulator_calls": 0,
            "test_array_decodes": 0, "array_decodes": 0, "checks": 0, "limitations": LIMITATIONS,
            "limits": LIMITS, "requires_successful_original_supervisor": True}

    def check(self):
        self.receipt["checks"] += 1
        if self.clock is not None and self.launch is not None:
            require(self.clock.now_ns() < self.launch["deadline_ns"], "original 600-second audit deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"], "audit RSS cap")
        if self.out.exists():
            require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) < LIMITS["output_bytes"] - 1024**2,
                    "audit output cap with failure reserve")

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN and descriptor(SUPERVISOR)["sha256"] == SUPERVISOR_PIN,
                "qualified original clock/supervisor bytes")
        self.clock = load(ROOT / CLOCK, "_query_memory_audit_clock").SuspendClock()
        self.started = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.started < 5 * 10**9, "original audit launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["parent_pid"] == os.getppid() and self.launch["pgid"] == os.getpgrp()
                and self.launch["cwd"] == str(ROOT) == str(Path.cwd())
                and self.launch["cap_seconds"] == 600 and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                and self.launch["clock_source_sha256"] == CLOCK_PIN and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.started < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 600 * 10**9, "original actual bounded audit process")
        self.check()
        self.inputs = {}
        for name in ("plan", "worker", "terminal"):
            path = regular(getattr(self.args, name))
            pin = descriptor(path)
            require(pin["sha256"] == getattr(self.args, name + "_sha256"), "external audit input hash: " + name)
            self.inputs[name] = {"path": str(path), **pin}
        self.plan = read(self.args.plan)
        require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_fitting"
                and NEW_COMPONENTS | {PRODUCER, CLOCK, SUPERVISOR} <= set(self.plan["sources"]), "auditor frozen before fitting")
        for name, expected in self.plan["sources"].items():
            self.check()
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and descriptor(name)["sha256"] == expected, "complete current frozen source identity")
        # The producer module exposes standard-library metadata authentication
        # only at import. No producer model, loss, metric or gate function runs.
        self.producer = load(ROOT / PRODUCER, "_query_memory_audit_metadata")
        p = self.producer
        require(self.plan["configuration"] == p.CONFIG and self.plan["limits"] == p.LIMITS
                and set(self.plan["payloads"]) == p.PAYLOADS and self.plan["runtime"] == p.runtime_record(),
                "fixed producer recipe, runtime and payload allocation")
        require(all(os.environ.get(name) == "1" for name in p.THREADS), "one numerical thread for audit")
        self.collection_plan, self.collection_receipt, self.collection, sources, capacity = p.authenticate_inputs(self.plan["inputs"])
        require(sources == self.plan["sources"] and capacity["projected_seconds"] == self.plan["capacity_projected_seconds"],
                "same authenticated collection/capacity/engineering source closure")
        engineering = read(self.plan["inputs"]["engineering"]["path"])
        require(NEW_COMPONENTS <= set(engineering["source_after"]), "every frozen auditor component was qualified before fitting")
        self.worker = read(self.args.worker)
        require(self.worker["version"] == PRODUCER_VERSION and self.worker["status"] == "completed"
                and self.worker["complete"] is True and self.worker["plan_sha256"] == self.args.plan_sha256
                and self.worker["sources"] == self.plan["sources"] and self.worker["inputs"] == self.plan["inputs"]
                and self.worker["limits"] == p.LIMITS and self.worker["fits_completed"] == 18
                and self.worker["optimizer_steps"] == 7560 and self.worker["episode_exposures"] == 45360
                and self.worker["teacher_calls"] == self.worker["native_calls"] == self.worker["test_array_decodes"] == 0
                and self.worker["technical_complete"] is False and self.worker["pending"] is self.worker["pending_emission"] is None
                and self.worker["requires_successful_original_supervisor"] is True
                and self.worker["requires_independent_saved_audit"] is True
                and self.worker["peak_rss_bytes"] <= p.LIMITS["rss_bytes"], "complete original TRAIN/DEV worker")
        process_inputs = {"training_plan": self.inputs["plan"], "training_receipt": self.inputs["worker"],
                          "training_terminal": self.inputs["terminal"]}
        self.parent = p.successful_process(process_inputs, "training", self.worker, PRODUCER, ".venv/bin/python", 21600)
        self.run = p.closed_files(self.args.worker, self.worker, p.PAYLOADS)
        require(sum(value["bytes"] for value in self.worker["files"].values()) <= p.LIMITS["output_bytes"], "producer output cap")
        self.receipt.update(inputs=self.inputs, source_plan_sha256=self.args.plan_sha256, sources=self.plan["sources"],
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        self.authenticate_metadata()
        self.authenticated = True
        write(self.out / "started.json", {"version": VERSION, "started_ns": self.started, "launch": self.launch,
                                         "inputs": self.inputs, "authenticated_before_numerical_reads": True})

    def authenticate_metadata(self):
        """All checkpoint and barrier payload identities precede NumPy import."""
        self.fits = read(self.run / "fits.json")["fits"]
        self.forks = read(self.run / "forks.json")["forks"]
        self.views = {}
        for stage in ("train", "dev"):
            value = read(self.run / (stage + "-views.json"))
            require(value["stage"] == stage and value["technical_complete"] is False, "producer views do not self-admit")
            self.views[stage] = value["views"]
            require([(r["view"], r["seed"]) for r in value["views"]] == [(v, s) for s in SEEDS for v in VIEWS], "all ordered canonical views")
            for row in value["views"]:
                expected = f"{stage}-prediction-{row['view']}-{row['seed']}.npz"
                require(row["prediction_path"] == expected and row["prediction"] == self.worker["files"][expected], "view payload identity")
        require([(r["family"], r["seed"]) for r in self.fits] == [(f, s) for s in SEEDS for f in FITS], "all ordered final fits")
        require([r["seed"] for r in self.forks] == list(SEEDS), "all same-seed forks")
        for row in self.fits:
            expected = f"checkpoint-{row['family']}-{row['seed']}.npz"
            require(row["checkpoint_path"] == expected and row["checkpoint"] == self.worker["files"][expected], "final checkpoint identity")
        barrier = read(self.run / "dev-barrier.json")
        expected = {row["checkpoint_path"] for row in self.fits} | {row["prediction_path"] for row in self.views["train"]}
        expected |= {"fits.json", "forks.json", "train-views.json", "train-history.json", "train-history.npz"}
        require(barrier["event"] == "all18_checkpoints_and24_TRAIN_views_closed_before_DEV"
                and barrier["fits"] == 18 and barrier["train_views"] == 24 and barrier["optimizer_steps"] == 7560
                and barrier["episode_exposures"] == 45360 and type(barrier["work_sequence"]) is int
                and barrier["work_sequence"] > 7560 and set(barrier["files"]) == expected and len(expected) == 47,
                "complete pre-DEV durable barrier")
        require(all(pin == self.worker["files"][name] for name, pin in barrier["files"].items()), "all pre-DEV closed payload hashes")
        self.barrier = barrier
        progress = self.rows(self.run / "progress.jsonl")
        completed, barrier_seen, decode_seen = [], False, False
        for row in progress:
            self.check()
            if row["event"] == "fit_complete":
                require(not barrier_seen, "all fits before DEV barrier")
                completed.append({key: value for key, value in row.items() if key != "event"})
            elif row["event"] == barrier["event"]:
                require(not barrier_seen and completed == self.fits, "one barrier after all final fits")
                require({key: value for key, value in row.items() if key != "barrier"} == barrier
                        and row["barrier"] == self.worker["files"]["dev-barrier.json"], "same progress barrier")
                barrier_seen = True
            elif row["event"] == "dev_decode_after_barrier":
                require(barrier_seen and not decode_seen and row["work_sequence"] == barrier["work_sequence"]
                        and row["barrier"] == self.worker["files"]["dev-barrier.json"], "source-authenticated DEV decode follows durable barrier")
                decode_seen = True
            else:
                require(row["event"] == "epoch_complete" and not barrier_seen, "only declared progress events")
        require(completed == self.fits and barrier_seen and decode_seen, "complete chronological pre-DEV progress")
        self.summary = read(self.run / "summary.json")
        require(self.summary["version"] == PRODUCER_VERSION and self.summary["configuration"] == self.plan["configuration"]
                and self.summary["fits"] == self.fits and self.summary["train_views"] == self.views["train"]
                and self.summary["dev_views"] == self.views["dev"] and self.summary["technical_complete"] is False
                and self.summary["test_array_decodes"] == 0 and self.summary["test_evaluation_admitted"] is False,
                "summary copies exact unadmitted producer records")
        runtime = read(self.run / "runtime.json")
        require(all(runtime[key] == value for key, value in self.plan["runtime"].items())
                and runtime["torch_threads"] == runtime["interop_threads"] == 1 and runtime["deterministic"] is True
                and runtime["cuda_used"] is runtime["mps_used"] is False, "actual recorded CPU runtime")

    def rows(self, path):
        with regular(path).open() as stream:
            for line in stream:
                require(line.endswith("\n"), "complete durable journal line")
                value = json.loads(line)
                require(isinstance(value, dict), "journal object record")
                yield value

    def arrays(self, path, pin):
        self.check()
        require(self.authenticated, "all metadata/closure/barrier checks precede numerical reads")
        require(path.name != "test.npz" and not path.name.startswith("test-"), "DEV auditor cannot decode TEST")
        require(descriptor(path) == pin, "bound payload immediately before numerical decode")
        self.receipt["pending"] = {"decode": path.name}
        with self.np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "unique NPZ entries")
            result = {name: archive[name] for name in archive.files}
        self.receipt["array_decodes"] += 1
        self.receipt["pending"] = None
        self.check()
        return result

    def histories(self):
        self.histories_by_stage, self.ids = {}, {}
        for stage, count in (("train", 54), ("dev", 18)):
            identities = [row for row in self.collection_plan["cohort"] if row["stage"] == stage]
            require(len(identities) == count, "full declared stage roster")
            flat = self.arrays(self.collection / (stage + ".npz"), self.collection_receipt["files"][stage + ".npz"])
            expected = expected_history(self.np, flat, identities)
            saved = self.arrays(self.run / (stage + "-history.npz"), self.worker["files"][stage + "-history.npz"])
            require(set(saved) == set(expected), "exact projected history fields")
            for name, array in expected.items():
                require(saved[name].dtype == array.dtype and saved[name].shape == array.shape
                        and saved[name].tobytes() == array.tobytes(), "independent complete observation history: " + name)
            metadata = read(self.run / (stage + "-history.json"))
            wanted = {"version": "otto-query-memory-data-v1", "query_period": 4, "stage": stage,
                "episode_ids": [row["episode_id"] for row in identities],
                "identities": [[[name, row[name]] for name in sorted(ID_FIELDS)] for row in identities], "episode_count": count}
            close_equal(metadata, wanted, "independent history metadata")
            self.histories_by_stage[stage], self.ids[stage] = saved, identities

    def checkpoints(self):
        np = self.np
        by = {(row["family"], row["seed"]): row for row in self.fits}
        for seed in SEEDS:
            parent = None
            projection = None
            branch_orders = None
            for family in FITS:
                self.check()
                row = by[family, seed]
                arrays = self.arrays(self.run / row["checkpoint_path"], row["checkpoint"])
                shapes = {"slow." + name: shape for name, shape in SLOW_SHAPES.items()}
                matrix = family not in ("pretrained", "joint_aux")
                if matrix:
                    shapes["projection.weight"] = (8, 28)
                require(set(arrays) == set(shapes), "exact final checkpoint tensors")
                for name, shape in shapes.items():
                    require(arrays[name].dtype == np.float32 and arrays[name].shape == shape
                            and bool(np.isfinite(arrays[name]).all()), "finite checkpoint tensor: " + name)
                close_equal(row["final"], tensor_witness(arrays), "actual final tensor witness")
                close_equal(row["final_slow"], tensor_witness(arrays, slow_only=True), "actual final slow witness")
                epochs = 80 if family == "pretrained" else 40
                require(row["epochs"] == epochs and row["steps"] == epochs * 9
                        and row["episode_exposures"] == epochs * 54 and row["optimizer_initial_state_entries"] == 0,
                        "fixed final epoch and fresh Adam counters")
                order_rng = np.random.Generator(np.random.PCG64(seed))
                orders = [order_rng.permutation(54).astype(np.int64) for _ in range(epochs)]
                require(row["episode_orders"] == [value.tolist() for value in orders]
                        and row["permutation_sha256"] == hashlib.sha256(b"".join(value.tobytes() for value in orders)).hexdigest(),
                        "independent fixed PCG64 episode orders")
                names = list(shapes)
                effective = ["projection.weight"] if matrix else names
                flags = [name for name in names if name.startswith("slow.action_residual.") or name == "projection.weight"] if matrix else names
                parameters = {"names": names, "count": 6336 if matrix else 6112, "requires_grad_names": flags,
                              "requires_grad_count": 340 if matrix else 6112, "effective_names": effective,
                              "effective_count": 224 if matrix else 6112}
                close_equal(row["parameters"], parameters, "recorded effective parameter contract")
                require(row["optimizer_parameter_names"] == effective
                        and row["optimizer_final_steps"] == {name: epochs * 9 for name in effective}
                        and row["slow_mode"] == ("frozen" if matrix else "joint")
                        and row["mode"] == (family if matrix else "none")
                        and row["stage"] == ("pretrain" if family == "pretrained" else "branch"), "fixed fit mode/optimizer identity")
                if family == "pretrained":
                    require(row["pretrained_checkpoint"] is None and row["frozen_slow_unchanged"] is None, "fresh pretraining")
                    parent = {name: value.copy() for name, value in arrays.items()}
                else:
                    pretrained = by["pretrained", seed]
                    close_equal(row["initial_slow"], pretrained["final_slow"], "all branches fork exact final same-seed slow state")
                    require(row["pretrained_checkpoint"] == {"path": pretrained["checkpoint_path"], **pretrained["checkpoint"]},
                            "parent checkpoint identity")
                    require(branch_orders is None or row["episode_orders"] == branch_orders, "matched adaptation episode order")
                    branch_orders = row["episode_orders"]
                    for name in SLOW_SHAPES:
                        require(row["initial"]["tensors"]["slow." + name] == pretrained["final"]["tensors"]["slow." + name],
                                "full initial branch witness has parent slow tensors")
                    if matrix:
                        require(row["frozen_slow_unchanged"] is True and all(arrays[name].tobytes() == value.tobytes()
                                for name, value in parent.items()), "all eight frozen slow tensors equal saved pretrained checkpoint")
                        current = row["initial"]["tensors"]["projection.weight"]
                        require(current["shape"] == [8, 28] and current["dtype"] == "torch.float32" and current["bytes"] == 896
                                and current["sha256"] != hashlib.sha256(np.zeros((8, 28), np.float32).tobytes()).hexdigest()
                                and (projection is None or current == projection), "identical nonzero initialized memory projections")
                        projection = current
                    else:
                        require(row["frozen_slow_unchanged"] is None, "joint branch makes no frozen equality claim")
            fork = next(row for row in self.forks if row["seed"] == seed)
            pretrained = by["pretrained", seed]
            expected_fork = {"seed": seed, "pretrained_checkpoint": pretrained["checkpoint"],
                "pretrained_checkpoint_path": pretrained["checkpoint_path"], "memory_initial_projection": projection,
                "branches": [{"family": family, **{name: by[family, seed][name] for name in
                    ("initial", "initial_slow", "checkpoint_path", "checkpoint")}} for family in FITS[1:]]}
            close_equal(fork, expected_fork, "complete same-seed fork metadata")

    def journal(self):
        rows = iter(self.rows(self.run / "work.jsonl"))
        sequence = 0
        marker = object()

        def pair(expected):
            nonlocal sequence
            self.check()
            sequence += 1
            attempt, returned = next(rows, marker), next(rows, marker)
            require(attempt is not marker and returned is not marker, "complete original work attempt/return pair")
            require(attempt == {"event": "attempt", "call_id": sequence, **expected}, "chronological exact work attempt")
            require(returned.get("event") == "return" and returned.get("call_id") == sequence
                    and all(returned.get(k) == value for k, value in expected.items()), "same-call successful work return")
            return returned

        train_lengths = self.np.diff(self.histories_by_stage["train"]["episode_offsets"])
        total_updates = total_exposures = 0
        for fit in self.fits:
            totals, work, units, timings = {}, {}, {}, {}
            for epoch, order in enumerate(fit["episode_orders"], 1):
                for batch in range(9):
                    indices = order[batch * 6:(batch + 1) * 6]
                    returned = pair({"family": fit["family"], "seed": fit["seed"], "epoch": epoch,
                                     "batch": batch, "episode_indices": indices})
                    value = returned["result"]
                    lengths = [int(train_lengths[index]) for index in indices]
                    chunks = (max(lengths) + 31) // 32
                    prior = sum((length - 1) // 4 for length in lengths)
                    nonquery = sum(length - (length + 3) // 4 for length in lengths)
                    require(value["version"] == "otto-query-memory-training-v1" and value["objective"] == "full_forecast_aux"
                            and value["episode_indices"] == indices and value["episode_exposures"] == 6
                            and value["optimizer_updates"] == 1 and value["optimizer_step"] == (epoch - 1) * 9 + batch + 1
                            and value["forward_rows"] == sum(lengths) and value["forward_chunks"] == chunks
                            and value["nonquery_rows"] == nonquery and value["prior_rows"] == prior,
                            "complete fixed training batch exposure")
                    backward = value["backward_chunks"]
                    require(type(backward) is int and 0 <= backward <= chunks
                            and value["loss_chunks"] == (chunks if max(lengths) > 1 else 0)
                            and backward == value["loss_chunks"]
                            and value["differentiable_chunks"] == backward
                            and value["no_gradient_chunks"] == value["skipped_backward_chunks"] == chunks - backward
                            and value["effective_parameter_names"] == fit["optimizer_parameter_names"]
                            and value["effective_parameter_count"] == fit["parameters"]["effective_count"]
                            and value["frozen_optimizer_state_entries"] == 0
                            and set(value["zero_filled_gradient_names"]) <= set(fit["optimizer_parameter_names"]),
                            "effective gradient work and optimizer ownership")
                    for name in ("loss", "nonquery_loss", "prior_loss", "gradient_norm_before_clip"):
                        require(type(value[name]) in (int, float) and math.isfinite(value[name]) and value[name] >= 0,
                                "finite preserved batch scalar")
                    expected_work, expected_units = {}, {}
                    for start in range(0, max(lengths), 32):
                        part = [max(0, min(32, length - start)) for length in lengths]
                        part_work, part_units = scheduled_work(part, start, fit["family"])
                        add_counts(expected_work, part_work)
                        add_counts(expected_units, part_units)
                    close_equal(value["work_counts"], expected_work, "independent scheduled training operations")
                    close_equal(value["memory_work_units"], expected_units, "independent scheduled training units")
                    for name in ("forward_rows", "nonquery_rows", "prior_rows", "forward_chunks", "loss_chunks",
                                 "backward_chunks", "differentiable_chunks", "no_gradient_chunks", "skipped_backward_chunks"):
                        require(type(value[name]) is int and value[name] >= 0, "integer batch work")
                        totals[name] = totals.get(name, 0) + value[name]
                    add_counts(work, value["work_counts"])
                    add_counts(units, value["memory_work_units"])
                    require(set(value["timing_seconds"]) == {"wall", "setup", "materialize", "forward", "loss",
                            "backward", "carry_detach", "gradient_completion", "gradient_clip", "optimizer", "overhead"},
                            "complete fixed batch timing fields")
                    for name, seconds in value["timing_seconds"].items():
                        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "preserved finite measured timing")
                        timings[name] = timings.get(name, 0.) + seconds
                    total_updates += 1
                    total_exposures += 6
            for name, expected in (("totals", totals), ("work_counts", work), ("memory_work_units", units), ("batch_timing_seconds", timings)):
                close_equal(fit[name], expected, "independent summed fit " + name)
        require(total_updates == 7560 and total_exposures == 45360, "all fixed optimization work")
        for stage in ("train", "dev"):
            if stage == "dev":
                require(sequence == self.barrier["work_sequence"], "every TRAIN/update call precedes DEV barrier")
            history = self.histories_by_stage[stage]
            lengths = self.np.diff(history["episode_offsets"])
            for view in self.views[stage]:
                work, units, chunks, count = {}, {}, 0, 0
                for first in range(0, len(lengths), 6):
                    indices = list(range(first, min(first + 6, len(lengths))))
                    for start in range(0, max(int(lengths[i]) for i in indices), 32):
                        returned = pair({"phase": stage, "view": view["view"], "seed": view["seed"],
                                         "episode_indices": indices, "start": start})
                        actual = sum(max(0, min(32, int(lengths[i]) - start)) for i in indices)
                        require(returned["rows"] == actual, "all canonical inference rows")
                        expected_work, expected_units = scheduled_work(
                            [max(0, min(32, int(lengths[i]) - start)) for i in indices], start, view["view"])
                        close_equal(returned["work_counts"], expected_work, "independent scheduled inference operations")
                        close_equal(returned["memory_work_units"], expected_units, "independent scheduled inference units")
                        add_counts(work, returned["work_counts"])
                        add_counts(units, returned["memory_work_units"])
                        count += actual
                        chunks += 1
                require(view["rows"] == count == int(history["episode_offsets"][-1]) and view["chunks"] == chunks,
                        "all rows and chunks in canonical view")
                close_equal(view["work_counts"], work, "all canonical operation counts")
                close_equal(view["memory_work_units"], units, "all canonical operation units")
        require(next(rows, marker) is marker, "no unregistered extra work call")
        self.receipt.update(training_updates_checked=total_updates, episode_exposures_checked=total_exposures,
                            paired_work_calls_checked=sequence)

    def predictions(self):
        reports = []
        for stage in ("train", "dev"):
            history, identities = self.histories_by_stage[stage], self.ids[stage]
            reference, reference_seed = None, None
            for record in self.views[stage]:
                self.check()
                view, seed = record["view"], record["seed"]
                parent = "pretrained" if view == "last_error" else "trace_delta" if view == "trace_no_write" else view
                require(record["parent_fit"] == parent and record["stage"] == stage, "canonical view parent/stage")
                expected_evaluation = {"slow_mode": "frozen", "grad_enabled": False, "residual_requires_grad": True,
                                       "projection_requires_grad": view not in ("pretrained", "joint_aux", "last_error")}
                close_equal(record["evaluation"], expected_evaluation, "recorded canonical flag contract")
                saved = self.arrays(self.run / record["prediction_path"], record["prediction"])
                if view == "pretrained":
                    reference, reference_seed = saved, seed
                require(reference_seed == seed, "same-seed canonical frozen reference")
                verify_prediction(self.np, saved, history, view, reference if view != "pretrained" else None)
                if view not in ("pretrained", "joint_aux"):
                    require(record["frozen_predictions_equal_pretrained"] is True, "recorded frozen parity agrees")
                independent = scalar_report(self.np, identities, history["targets"], history["legal"], saved["action_prediction"],
                    saved["corrected_shadow_prior"], history["episode_offsets"], view, seed, check=self.check)
                close_equal(record["metrics"], independent, "independent complete " + stage + " metrics")
                require(set(record["loss"]) == {"total", "nonquery", "prior"}
                        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in record["loss"].values()),
                        "finite recorded float32 canonical AUX components")
                if stage == "dev":
                    reports.append(independent)
        require(len(reports) == 24, "all independently reconstructed DEV views")
        producer_gate = independent_gate(reports, technical_complete=False)
        close_equal(self.summary["gate"], producer_gate, "independent unadmitted producer gate")
        return reports, independent_gate(reports, technical_complete=True)

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(path.is_symlink() for path in self.out.parents), "contained exclusive audit output")
        self.out.mkdir(exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("original audit supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            import numpy as np

            self.np = np
            self.histories()
            self.checkpoints()
            self.journal()
            reports, gate = self.predictions()
            self.check()
            result = {"version": VERSION, "agreement": True, "stage": "dev", "metrics": reports, "gate": gate,
                "producer_gate": self.summary["gate"], "limitations": LIMITATIONS,
                "conditional_test_admission_requires_successful_original_audit_supervisor": True,
                "test_evaluation_admitted": False, "inputs": self.inputs,
                "zero_call_counts": {name: self.receipt[name] for name in
                    ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes")}}
            write(self.out / "audit.json", result)
            for name, expected in self.plan["sources"].items():
                self.check()
                require(descriptor(name)["sha256"] == expected, "unchanged final source")
            for record in self.inputs.values():
                require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")}, "unchanged external audit inputs")
            for directory, receipt in ((self.run, self.worker), (self.collection, self.collection_receipt)):
                for name, expected in receipt["files"].items():
                    self.check()
                    require(descriptor(directory / name) == expected, "unchanged final bound payload")
            require(descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged own original launch")
            self.receipt.update(status="completed", complete=True, agreement=True, pending=None,
                started_ns=self.started, finished_ns=self.clock.now_ns(),
                files={name: descriptor(self.out / name) for name in ("started.json", "audit.json")})
            self.receipt["wall_seconds"] = (self.receipt["finished_ns"] - self.started) / 1e9
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {path.name: descriptor(path) for path in self.out.iterdir() if path.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain the original failure
                error.add_note("Failure receipt publication also failed: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "worker", "terminal", "supervision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan", "worker", "terminal"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    require(all(getattr(args, name).is_absolute() for name in ("plan", "worker", "terminal", "supervision", "output")),
            "absolute registered audit inputs/output")
    Audit(args).execute()


if __name__ == "__main__":
    main()
