"""Independent conditional TEST saved-output audit, with zero scientific calls.

The evaluator's original successful process closure, its DEV admission chain,
all immutable inputs, sources and checkpoint identities precede array decoding.
Only scalar decision arithmetic and schedule geometry are reconstructed here.
Source-tested model causality is not an independent numerical inference replay.
The saved result remains conditional on this audit's original supervisor exit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
import traceback
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_AUDITOR = "scripts/audit_otto_query_memory.py"
SPEC = importlib.util.spec_from_file_location("_qualified_query_memory_dev_audit", ROOT / DEV_AUDITOR)
dev = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dev
SPEC.loader.exec_module(dev)

SELF = "scripts/audit_otto_query_memory_test.py"
TEST = "tests/test_audit_otto_query_memory_test.py"
NEW_COMPONENTS = {SELF, TEST}
PRODUCER = "scripts/evaluate_otto_query_memory.py"
VERSION = "otto-query-memory-test-saved-audit-v1"
PRODUCER_VERSION = "otto-query-memory-test-v1"
PERIODS = (4, 8)
SCOPES = ("full", "initial", "later", "common_full", "common_initial", "common_later")
SEEDS, VIEWS, FITS, CONTROLS = dev.SEEDS, dev.VIEWS, dev.FITS, dev.CONTROLS
LIMITS = {"seconds": 600, "rss_bytes": 2 * 1024**3, "output_bytes": 256 * 1024**2}
require, descriptor, read, write, load = dev.require, dev.descriptor, dev.read, dev.write, dev.load
regular, close_equal, add_counts = dev.regular, dev.close_equal, dev.add_counts
scalar_row, centered_mse = dev.scalar_row, dev.centered_mse
ID_FIELDS, METRIC_NAMES = dev.ID_FIELDS, dev.METRIC_NAMES
LIMITATIONS = [
    "No model, optimizer, teacher or simulator is called; checkpoint inference is not numerically replayed.",
    "Causal input schedules and source/qualification lineage are authenticated; hidden-state and write causality remain source-tested execution claims.",
    "TEST scalar metrics, complete-path support, the 25-condition gate and saved frozen/fork/no-write equality are independently reconstructed.",
    "Teacher/native numerical truth, actual inference and physical timings remain authenticated producer evidence.",
    "P4/P8 share fixed collector paths; census costs remain paid and these results do not establish autonomous efficacy or physical teacher savings.",
    "A passed worker gate is conditional on this audit's genuine successful original supervisor closure.",
]


def scalar_report(np, identities, target, legal, predictions, prior, offsets, family, seed,
                  query_period, *, check=lambda: None):
    """Independent full-path TEST scope construction and equal-case aggregation."""
    require(query_period in PERIODS and identities and all(r["stage"] == "test" for r in identities),
            "only TEST P4/P8 scalar audit")
    order = sorted(range(len(identities)), key=lambda i: identities[i]["episode_index"])
    records = []
    for index in order:
        check()
        identity = {key: identities[index][key] for key in ID_FIELDS}
        low, high = map(int, offsets[index:index + 2])
        rows, priors = [], []
        for absolute in range(low, high):
            step = absolute - low
            if step % query_period:
                rows.append((step, scalar_row(target[absolute], legal[absolute], predictions[absolute])))
            elif step:
                priors.append(centered_mse(prior[absolute], target[absolute], (0, 1, 2, 3)))
        bins = {}
        for scope in SCOPES:
            selected = []
            for step, values in rows:
                include = (scope == "full" or scope == "initial" and step < query_period
                           or scope == "later" and step >= query_period + 1
                           or scope == "common_full" and step % 4 != 0
                           or scope == "common_initial" and step % 4 != 0 and step < 8
                           or scope == "common_later" and step % 4 != 0 and step >= 9)
                if include:
                    selected.append((step, values))
            ages = (None,) if scope.startswith("common_") else (None, *range(1, query_period))
            for age in ages:
                values = [row for step, row in selected if age is None or step % query_period == age]
                bins[scope, age] = {"rows": len(values), "sums": {
                    name: math.fsum(row[column] for row in values) for column, name in enumerate(METRIC_NAMES)}}
        bins["prior", None] = {"rows": len(priors), "sums": {"centered_mse": math.fsum(priors)}}
        digest = hashlib.sha256(b"otto-query-memory-target-v1\0")
        digest.update((high - low).to_bytes(8, "little"))
        digest.update(target[low:high].astype("<f4", copy=False).tobytes(order="C"))
        digest.update(legal[low:high].tobytes(order="C"))
        records.append({"identity": identity, "length": high - low,
                        "target_sha256": digest.hexdigest(), "bins": bins})

    def summary(indices, scope, age=None):
        leaves = [records[i]["bins"][scope, age] for i in indices]
        ids = [records[i]["identity"] for i in indices]
        cases = sorted({(r["regime"], r["case"]) for r in ids})
        supported = [i for i, leaf in enumerate(leaves) if leaf["rows"]]
        supported_cases = {(ids[i]["regime"], ids[i]["case"]) for i in supported}
        names = ("centered_mse",) if scope == "prior" else METRIC_NAMES
        sums = {name: math.fsum(leaf["sums"][name] for leaf in leaves) for name in names}
        rows = sum(leaf["rows"] for leaf in leaves)
        value = {"episodes": len(indices), "supported_episodes": len(supported),
            "zero_support_episode_ids": [r["episode_id"] for r, leaf in zip(ids, leaves, strict=True) if not leaf["rows"]],
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
            case_means = []
            for case in cases:
                members = [i for i, r in enumerate(ids) if (r["regime"], r["case"]) == case]
                case_means.append(math.fsum(means[i] for i in members) / len(members))
            value["case_weighted_" + name] = math.fsum(case_means) / len(cases)
        if scope in ("full", "initial", "later") and age is None:
            value["by_age"] = {str(a): summary(indices, scope, a) for a in range(1, query_period)}
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

    return {"version": dev.METRIC_VERSION,
        "scope": "teacher-score imitation on fixed collector paths; no autonomous efficacy",
        "family": family, "seed": seed, "query_period": query_period, "stage": "test", "episodes": len(records),
        "identity_manifest": [{**r["identity"], "length": r["length"], "target_sha256": r["target_sha256"]} for r in records],
        "scopes": {scope: groups(scope) for scope in SCOPES}, "prequery": groups("prior")}


def independent_gate(reports, *, technical_complete=False):
    require(type(technical_complete) is bool, "strict technical completion Boolean")
    require(len(reports) == 48 and {(r["family"], r["seed"], r["query_period"]) for r in reports}
            == {(v, s, p) for p in PERIODS for s in SEEDS for v in VIEWS}, "all48 independent TEST reports")
    require(all(r["stage"] == "test" and r["episodes"] == 36 for r in reports), "complete TEST roster")
    by = {(r["family"], r["seed"], r["query_period"]): r for r in reports}

    def leaf(family, seed, period, scope, regime):
        result = by[family, seed, period]["scopes"][scope]["by_regime"][regime]
        require(result["episodes"] == 18 and result["declared_case_count"] == 6, "complete six TEST cases")
        gap = result["case_weighted_raw_gap"]
        require(type(gap) in (int, float) and math.isfinite(gap) and gap >= 0, "finite independent legal gap")
        return result

    def mean(family, period, scope, regime):
        return math.fsum(leaf(family, s, period, scope, regime)["case_weighted_raw_gap"] for s in SEEDS) / 3

    conditions = [{"name": "technical_completion", "passed": technical_complete}]
    for period in PERIODS:
        for regime in ("lambda3", "lambda4"):
            prefix = f"{regime}:P{period}"
            supports = {leaf(v, s, period, "later", regime)["supported_case_count"] for v in VIEWS for s in SEEDS}
            require(len(supports) == 1, "identical independently reconstructed case support")
            support = supports.pop()
            require(type(support) is int and 0 <= support <= 6, "bounded independent case support")
            conditions.append({"name": prefix + ":supported_cases", "passed": support >= 4, "actual": support, "required": 4})
            candidate = mean("trace_delta", period, "later", regime)
            controls = {v: mean(v, period, "later", regime) for v in CONTROLS}
            best = min(controls.values())
            conditions.append({"name": prefix + ":later_gap_10pct", "passed": candidate <= .9 * best and candidate < best,
                               "candidate": candidate, "controls": controls, "best_control": best})
            candidate = mean("trace_delta", period, "full", regime)
            controls = {v: mean(v, period, "full", regime) for v in CONTROLS}
            conditions.append({"name": prefix + ":full_gap_nonregression", "passed": candidate <= min(controls.values()),
                               "candidate": candidate, "controls": controls})
            for seed in SEEDS:
                candidate = leaf("trace_delta", seed, period, "later", regime)["case_weighted_raw_gap"]
                controls = {v: leaf(v, seed, period, "later", regime)["case_weighted_raw_gap"] for v in CONTROLS}
                conditions.append({"name": prefix + f":seed_{seed}_nonregression", "passed": candidate <= min(controls.values()),
                                   "candidate": candidate, "controls": controls})
    require(len(conditions) == len({r["name"] for r in conditions}) == 25, "exact25 independent TEST conditions")
    return {"version": dev.GATE_VERSION, "stage": "test", "candidate": "trace_delta", "technical_complete": technical_complete,
            "conditions": conditions, "passed_conditions": sum(r["passed"] for r in conditions), "total_conditions": 25,
            "passed": all(r["passed"] for r in conditions),
            "scope": "prospective mechanism screen; no autonomous, total-compute or novelty claim"}


def expected_history(np, flat, identities, period):
    """Check original P4 census, then change exactly the public age column."""
    require(period in PERIODS and len(identities) == 36 and all(r["stage"] == "test" for r in identities),
            "complete TEST P4/P8 history")
    original = dev.expected_history(np, flat, identities)
    projected = {name: values.copy() for name, values in original.items()}
    if period == 4:
        return projected
    for low, high in pairwise(projected["episode_offsets"]):
        steps = np.arange(high - low)
        query = steps % period == 0
        prior = query & (steps > 0)
        projected["features"][low:high, 16] = (steps % period / 2188).astype(np.float32)
        projected["query_mask"][low:high], projected["prior_mask"][low:high] = query, prior
        projected["nonquery_weights"][low:high] = 0
        projected["prior_weights"][low:high] = 0
        if int((~query).sum()):
            projected["nonquery_weights"][low:high][~query] = 1 / (36 * int((~query).sum()))
        if int(prior.sum()):
            projected["prior_weights"][low:high][prior] = 1 / (36 * int(prior.sum()))
    projected["query_scores"][:] = np.nan
    projected["query_scores"][projected["query_mask"]] = flat["raw_q"][projected["query_mask"]]
    return projected


def scheduled_work(lengths, start, view, period):
    """Independent P4/P8 operation geometry; no recurrent/projection execution."""
    require(period in PERIODS and start >= 0 and start % 32 == 0
            and all(type(n) is int and 0 <= n <= 32 for n in lengths), "fixed TEST chronological chunk geometry")
    active = sum(lengths)
    queries = sum((n + period - 1) // period for n in lengths)
    first = sum(n > 0 for n in lengths) if start == 0 else 0
    later, keys, nonquery = queries - first, active - first, active - queries
    query_groups = sum(any(n > offset for n in lengths) for offset in range(0, 32, period))
    first_groups = int(start == 0 and any(lengths))
    later_groups = query_groups - first_groups
    ordinary_groups = sum(len({min(period - 1, n - offset - 1) for n in lengths if n > offset + 1})
                          for offset in range(0, 32, period))
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


class Audit(dev.Audit):
    def __init__(self, args):
        super().__init__(args)
        self.allowed_arrays = {}
        self.receipt.update(version=VERSION, limitations=LIMITATIONS, checkpoint_decodes=0,
                            prediction_decodes=0, projected_history_decodes=0,
                            train_array_decodes=0, dev_array_decodes=0,
                            technical_complete=False, continuation_admitted=False)

    def admit(self):
        require(descriptor(dev.CLOCK)["sha256"] == dev.CLOCK_PIN
                and descriptor(dev.SUPERVISOR)["sha256"] == dev.SUPERVISOR_PIN, "qualified original clock/supervisor bytes")
        self.clock = load(ROOT / dev.CLOCK, "_query_memory_test_audit_clock").SuspendClock()
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
                and self.launch["cwd"] == str(ROOT) == str(Path.cwd()) and self.launch["cap_seconds"] == 600
                and self.launch["watchdog_sha256"] == dev.SUPERVISOR_PIN
                and self.launch["clock_source_sha256"] == dev.CLOCK_PIN and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.started < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + 600 * 10**9, "original actual bounded TEST audit process")
        self.check()
        self.inputs = {}
        for name in ("plan", "worker", "terminal"):
            path = regular(getattr(self.args, name))
            pin = descriptor(path)
            require(pin["sha256"] == getattr(self.args, name + "_sha256"), "external audit input hash: " + name)
            self.inputs[name] = {"path": str(path), **pin}
        self.plan = read(self.args.plan)
        require(self.plan["version"] == PRODUCER_VERSION and self.plan["status"] == "frozen_before_TEST"
                and NEW_COMPONENTS | {PRODUCER, DEV_AUDITOR, dev.CLOCK, dev.SUPERVISOR} <= set(self.plan["sources"]),
                "TEST auditor frozen before TEST decoding")
        for name, expected in self.plan["sources"].items():
            self.check()
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and descriptor(name)["sha256"] == expected, "complete current frozen TEST source identity")
        # Only metadata admission runs in this module. Its model, metric and
        # continuation functions are never used to reconstruct TEST results.
        self.evaluator = load(ROOT / PRODUCER, "_query_memory_test_audit_metadata")
        p = self.evaluator
        require(self.plan["configuration"] == p.CONFIG and self.plan["limits"] == p.LIMITS
                and set(self.plan["payloads"]) == p.PAYLOADS, "fixed TEST recipe and payload allocation")
        self.context = p.authenticate_inputs(self.plan["inputs"])
        self.producer = self.context["producer"]
        require(self.context["sources"] == self.plan["sources"]
                and self.plan["runtime"] == self.producer.runtime_record(), "complete original TEST input/source/runtime closure")
        require(all(os.environ.get(name) == "1" for name in self.producer.THREADS), "one numerical audit thread")
        engineering = read(self.plan["inputs"]["engineering"]["path"])
        require(NEW_COMPONENTS <= set(engineering["source_after"]), "TEST auditor was qualified before TEST decoding")
        self.worker = read(self.args.worker)
        require(self.worker["version"] == PRODUCER_VERSION and self.worker["status"] == "completed"
                and self.worker["complete"] is True and self.worker["plan_sha256"] == self.args.plan_sha256
                and self.worker["sources"] == self.plan["sources"] and self.worker["inputs"] == self.plan["inputs"]
                and self.worker["limits"] == p.LIMITS and self.worker["views_completed"] == 48
                and self.worker["checkpoint_decodes"] == 18 and self.worker["test_array_decodes"] == 1
                and all(self.worker[name] == 0 for name in p.ZERO_COUNTS)
                and self.worker["technical_complete"] is False and self.worker["pending"] is self.worker["pending_emission"] is None
                and self.worker["requires_successful_original_supervisor"] is True
                and self.worker["requires_independent_saved_audit"] is True
                and self.worker["peak_rss_bytes"] <= p.LIMITS["rss_bytes"], "complete original conditional TEST worker")
        process_inputs = {"test_plan": self.inputs["plan"], "test_receipt": self.inputs["worker"],
                          "test_terminal": self.inputs["terminal"]}
        self.parent = self.producer.successful_process(process_inputs, "test", self.worker, PRODUCER, ".venv/bin/python", 1800)
        require(type(self.worker["model_calls"]) is int and self.worker["model_calls"] > 0
                and self.worker["wall_seconds"] == (self.worker["finished_ns"] - self.worker["started_ns"]) / 1e9,
                "exact charged TEST worker calls and elapsed time")
        self.run = self.producer.closed_files(self.args.worker, self.worker, p.PAYLOADS)
        require(sum(pin["bytes"] for pin in self.worker["files"].values()) <= p.LIMITS["output_bytes"], "TEST output cap")
        self.receipt.update(inputs=self.inputs, source_plan_sha256=self.args.plan_sha256, sources=self.plan["sources"],
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        self.authenticate_metadata()
        self.authenticated = True
        write(self.out / "started.json", {"version": VERSION, "started_ns": self.started, "launch": self.launch,
            "inputs": self.inputs, "authenticated_before_numerical_reads": True})

    def authenticate_metadata(self):
        training, collection = self.context["training"], self.context["collection"]
        self.fits = training["fits"]
        self.forks = read(training["run"] / "forks.json")["forks"]
        require([(r["family"], r["seed"]) for r in self.fits] == [(f, s) for s in SEEDS for f in FITS],
                "all18 unchanged trained checkpoints")
        manifest = read(self.run / "checkpoint-manifest.json")
        expected = [{k: r[k] for k in ("family", "seed", "checkpoint_path", "checkpoint", "final", "final_slow")}
                    for r in self.fits]
        close_equal(manifest, {"training_receipt": self.plan["inputs"]["training_receipt"], "checkpoints": expected},
                    "all18 TEST source checkpoint identities")
        self.allowed_arrays = {str(training["run"] / r["checkpoint_path"]): (r["checkpoint"], "checkpoint_decodes") for r in self.fits}
        self.allowed_arrays[str(collection["run"] / "test.npz")] = (collection["receipt"]["files"]["test.npz"], "test_array_decodes")
        value = read(self.run / "test-views.json")
        require(value["stage"] == "test" and value["technical_complete"] is False, "TEST producer views cannot self-admit")
        self.views = value["views"]
        require([(r["query_period"], r["seed"], r["view"]) for r in self.views]
                == [(p, s, v) for p in PERIODS for s in SEEDS for v in VIEWS], "all48 ordered canonical TEST views")
        for row in self.views:
            name = f"test-P{row['query_period']}-prediction-{row['view']}-{row['seed']}.npz"
            require(row["prediction_path"] == name and row["prediction"] == self.worker["files"][name], "TEST view payload identity")
            parent = "pretrained" if row["view"] == "last_error" else "trace_delta" if row["view"] == "trace_no_write" else row["view"]
            fit = next(f for f in self.fits if (f["family"], f["seed"]) == (parent, row["seed"]))
            require(row["parent_fit"] == parent and row["stage"] == "test" and row["checkpoint_path"] == fit["checkpoint_path"]
                    and row["checkpoint"] == fit["checkpoint"] and row["model_before"] == row["model_after"] == fit["final"],
                    "same final unmodified checkpoint in both TEST periods")
            self.allowed_arrays[str(self.run / name)] = (row["prediction"], "prediction_decodes")
            for key in ("clone_seconds", "inference_and_loss_seconds", "metric_seconds", "serialization_seconds"):
                require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, "finite charged TEST view timing")
        for period in PERIODS:
            name = f"test-P{period}-history.npz"
            self.allowed_arrays[str(self.run / name)] = (self.worker["files"][name], "projected_history_decodes")
        require(len(self.allowed_arrays) == 69, "exact69 independently decoded TEST audit payloads")
        self.summary = read(self.run / "summary.json")
        require(self.summary["version"] == PRODUCER_VERSION and self.summary["configuration"] == self.plan["configuration"]
                and self.summary["views"] == self.views and self.summary["technical_complete"] is False
                and all(self.summary[name] == self.worker[name] for name in
                        (*self.evaluator.ZERO_COUNTS, "views_completed", "checkpoint_decodes", "test_array_decodes", "model_calls")),
                "TEST summary copies unchanged unadmitted worker records")
        require(type(self.summary["setup_seconds"]) in (int, float)
                and math.isfinite(self.summary["setup_seconds"]) and self.summary["setup_seconds"] >= 0, "finite charged TEST setup")
        runtime = read(self.run / "runtime.json")
        require(all(runtime[key] == value for key, value in self.plan["runtime"].items())
                and runtime["torch_threads"] == runtime["interop_threads"] == 1 and runtime["deterministic"] is True
                and runtime["cuda_used"] is runtime["mps_used"] is False, "actual recorded TEST CPU runtime")

    def arrays(self, path, pin):
        self.check()
        require(self.authenticated, "all TEST metadata and original closure checks precede numerical reads")
        require(str(path) in self.allowed_arrays and self.allowed_arrays[str(path)][0] == pin,
                "only the69 admitted TEST audit array payloads")
        require(descriptor(path) == pin, "bound payload immediately before numerical decode")
        self.receipt["pending"] = {"decode": path.name}
        with self.np.load(path, allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "unique NPZ entries")
            result = {name: archive[name] for name in archive.files}
        self.receipt["array_decodes"] += 1
        self.receipt[self.allowed_arrays[str(path)][1]] += 1
        self.receipt["pending"] = None
        self.check()
        return result

    def histories(self):
        source = self.context["collection"]
        self.ids = [r for r in source["plan"]["cohort"] if r["stage"] == "test"]
        wanted = {(regime, case, first + case, arm) for regime, first in (("lambda3", 307000001), ("lambda4", 308000001))
                  for case in range(6) for arm in ("analytic", "neural", "period4_hold")}
        require(len(self.ids) == 36 and {(r["regime"], r["case"], r["seed"], r["arm"]) for r in self.ids} == wanted
                and len({r["episode_id"] for r in self.ids}) == len({r["episode_index"] for r in self.ids}) == 36,
                "complete exact prospective TEST cohort")
        flat = self.arrays(source["run"] / "test.npz", source["receipt"]["files"]["test.npz"])
        self.histories_by_period = {}
        for period in PERIODS:
            expected = expected_history(self.np, flat, self.ids, period)
            name = f"test-P{period}-history.npz"
            saved = self.arrays(self.run / name, self.worker["files"][name])
            require(set(saved) == set(expected), "exact TEST projected history fields")
            for key, array in expected.items():
                require(saved[key].dtype == array.dtype and saved[key].shape == array.shape and saved[key].tobytes() == array.tobytes(),
                        "independent P-scheduled history: " + key)
            metadata = read(self.run / f"test-P{period}-history.json")
            close_equal(metadata, {"version": "otto-query-memory-data-v1", "query_period": period, "stage": "test",
                "episode_ids": [r["episode_id"] for r in self.ids], "episode_count": 36,
                "identities": [[[name, row[name]] for name in sorted(ID_FIELDS)] for row in self.ids]},
                "independent complete TEST history identity")
            self.histories_by_period[period] = saved

    def checkpoints(self):
        # Qualified tensor and fork checks only. This helper runs no model and
        # reads no TRAIN or DEV examples. Our explicit allowlist permits exactly
        # the eighteen already-frozen final checkpoints, once each.
        test_run = self.run
        try:
            self.run = self.context["training"]["run"]
            dev.Audit.checkpoints(self)
        finally:
            self.run = test_run

    def journal(self):
        rows = iter(self.rows(self.run / "work.jsonl"))
        marker, sequence = object(), 0
        total_work, total_units = {}, {}
        for view in self.views:
            period = view["query_period"]
            history = self.histories_by_period[period]
            lengths = self.np.diff(history["episode_offsets"])
            work, units, chunks, count = {}, {}, 0, 0
            for first in range(0, len(lengths), 6):
                indices = list(range(first, min(first + 6, len(lengths))))
                for start in range(0, max(int(lengths[i]) for i in indices), 32):
                    self.check()
                    sequence += 1
                    identity = {"phase": "test", "view": view["view"], "seed": view["seed"], "query_period": period,
                                "episode_indices": indices, "start": start}
                    attempt, returned = next(rows, marker), next(rows, marker)
                    require(attempt is not marker and returned is not marker, "complete original TEST work pair")
                    require(attempt == {"event": "attempt", "call_id": sequence, **identity}, "chronological exact TEST work attempt")
                    local = [max(0, min(32, int(lengths[i]) - start)) for i in indices]
                    expected_work, expected_units = scheduled_work(local, start, view["view"], period)
                    close_equal(returned, {"event": "return", "call_id": sequence, **identity, "rows": sum(local),
                        "work_counts": expected_work, "memory_work_units": expected_units}, "independent scheduled TEST work return")
                    add_counts(work, expected_work)
                    add_counts(units, expected_units)
                    count += sum(local)
                    chunks += 1
            require(view["rows"] == count == int(history["episode_offsets"][-1]) and view["chunks"] == chunks,
                    "all TEST view rows and chunks")
            close_equal(view["work_counts"], work, "all independently reconstructed TEST operations")
            close_equal(view["memory_work_units"], units, "all independently reconstructed TEST units")
            add_counts(total_work, work)
            add_counts(total_units, units)
        require(next(rows, marker) is marker and sequence == self.worker["model_calls"], "no unregistered TEST work call")
        self.receipt.update(paired_work_calls_checked=sequence, inference_work_counts=total_work, memory_work_units=total_units)

    def predictions(self):
        reports, reference, reference_key = [], None, None
        for row in self.views:
            self.check()
            view, seed, period = row["view"], row["seed"], row["query_period"]
            history = self.histories_by_period[period]
            close_equal(row["evaluation"], {"slow_mode": "frozen", "grad_enabled": False, "residual_requires_grad": True,
                "projection_requires_grad": view not in ("pretrained", "joint_aux", "last_error")}, "canonical TEST flag contract")
            saved = self.arrays(self.run / row["prediction_path"], row["prediction"])
            if view == "pretrained":
                reference, reference_key = saved, (seed, period)
            require(reference_key == (seed, period), "same-seed same-period canonical frozen reference")
            dev.verify_prediction(self.np, saved, history, view, reference if view != "pretrained" else None)
            if view not in ("pretrained", "joint_aux"):
                require(row["frozen_predictions_equal_pretrained"] is True, "recorded frozen TEST parity agrees")
            independent = scalar_report(self.np, self.ids, history["targets"], history["legal"], saved["action_prediction"],
                saved["corrected_shadow_prior"], history["episode_offsets"], view, seed, period, check=self.check)
            close_equal(row["metrics"], independent, "independent complete TEST metrics")
            require(set(row["loss"]) == {"total", "nonquery", "prior"}
                    and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in row["loss"].values()),
                    "finite recorded TEST float32 AUX components")
            reports.append(independent)
        close_equal(self.summary["gate"], independent_gate(reports, technical_complete=False), "independent unadmitted TEST gate")
        return reports, independent_gate(reports, technical_complete=True)

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(path.is_symlink() for path in self.out.parents), "contained exclusive TEST audit output")
        self.out.mkdir(exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("original TEST audit supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            import numpy as np

            self.np = np
            self.histories()
            self.checkpoints()
            self.journal()
            reports, gate = self.predictions()
            require(self.receipt["array_decodes"] == 69 and self.receipt["checkpoint_decodes"] == 18
                    and self.receipt["prediction_decodes"] == 48 and self.receipt["projected_history_decodes"] == 2
                    and self.receipt["test_array_decodes"] == 1, "complete69 bounded TEST audit decodes")
            costs = {"test_evaluation_work": self.receipt["inference_work_counts"], "test_memory_units": self.receipt["memory_work_units"],
                "view_seconds": {name: math.fsum(r[name] for r in self.views) for name in
                    ("clone_seconds", "inference_and_loss_seconds", "metric_seconds", "serialization_seconds")},
                "setup_seconds": self.summary["setup_seconds"], "worker_wall_seconds": self.worker["wall_seconds"],
                "worker_peak_rss_bytes": self.worker["peak_rss_bytes"],
                "inherited_training_receipt": self.plan["inputs"]["training_receipt"],
                "inherited_full_census_receipt": self.context["training"]["plan"]["inputs"]["collection_receipt"],
                "scope": "all48 TEST views charged separately; shared training and full census remain paid inherited costs"}
            result = {"version": VERSION, "agreement": True, "stage": "test", "metrics": reports, "gate": gate,
                "producer_gate": self.summary["gate"], "limitations": LIMITATIONS, "costs": costs,
                "requires_successful_original_audit_supervisor": True, "continuation_admitted": False,
                "technical_complete": False, "inputs": self.inputs,
                "zero_call_counts": {name: self.receipt[name] for name in
                    ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "train_array_decodes", "dev_array_decodes")}}
            write(self.out / "audit.json", result)
            for name, pin in self.plan["sources"].items():
                self.check()
                require(descriptor(name)["sha256"] == pin, "unchanged final TEST audit source")
            for record in (*self.inputs.values(), *self.plan["inputs"].values()):
                require(descriptor(record["path"]) == {key: record[key] for key in ("sha256", "bytes")}, "unchanged external audit input")
            for source in ({"run": self.run, "receipt": self.worker}, self.context["training"],
                           self.context["collection"], self.context["dev_audit"]):
                for name, pin in source["receipt"]["files"].items():
                    self.check()
                    require(descriptor(source["run"] / name) == pin, "unchanged final bound TEST audit payload")
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
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
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
            "absolute registered TEST audit input/output")
    Audit(args).execute()


if __name__ == "__main__":
    main()
