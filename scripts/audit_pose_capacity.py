"""Saved-output verification of target-aware fixed-expert capacity bounds.

This is not a forecast evaluation: targets deliberately choose the oracle
coefficients. No optimization, model, optimizer, simulator or RNG is invoked.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPAIR_SHA256 = "78a4382787e00298dc3f4486451f6ae23bcd8b6035ab3a03c916aff8662e6aa6"
REPAIR_TEST_SHA256 = "c920113556327e77c8bf6d9f9ba97c96c71fa545cfbbd96fcee102d973e1bfe8"
PARENT_PROTOCOL = "b9e0d4c9a43527cd14d1931242cd130fd224eb948cf6f7d8662e43cb9d2b5143"
PARENT_COMPLETED = "3f57c8727e574a01e078ffea2e3cce5179619a72c8753e1b928d4f3115759be2"
PARENT_SUMMARY = "b1d8b3ab1bc379919102b7116784a974787697f6df1abbe52a66803897f4288b"
PARENT_RECEIPT = "1bf635372997ec0349857edb038ffaaf4f961989b4e94868b3efece78c644a24"


def load_pinned(path, digest):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("pinned audit source identity")
    module = ModuleType("_pose_capacity_pinned_audit")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)  # noqa: S102 - exactly hash-pinned local source
    return module


_repair = load_pinned(ROOT / "scripts/audit_pose_crossfit_repair.py", REPAIR_SHA256)
_audit = _repair.load_original()
_audit.close_number = _repair.close_number
require, sha, read_json, write_json = _audit.require, _audit.sha, _audit.read_json, _audit.write_json
members, finite, error_arrays = _audit.members, _audit.finite, _audit.error_arrays
PANELS, SEEDS, ENDPOINTS = _audit.PANELS, _audit.SEEDS, _audit.ENDPOINTS
CONTROLS = (*_audit.VARIANTS, *_audit.REFERENCES)
SOURCES = _audit.SOURCES | {"scripts/audit_pose_crossfit_repair.py", "tests/test_audit_pose_crossfit_repair.py",
    "scripts/run_pose_capacity.py", "scripts/audit_pose_capacity.py", "tests/test_audit_pose_capacity.py",
    "research/pose-capacity-protocol.md"}


def hard_errors(fp, fr, sp, sr, tp, tr):
    """Independent endpoint choices; hard_step additionally changes each step."""
    fast, slow = error_arrays(fp, fr, tp, tr), error_arrays(sp, sr, tp, tr)
    window = np.stack([np.minimum(fast[e].mean(1), slow[e].mean(1)) for e in ENDPOINTS], axis=-1)
    step = np.stack([np.minimum(fast[e], slow[e]).mean(1) for e in ENDPOINTS], axis=-1)
    require(np.all(step <= window + 1e-14), "relaxed step bound exceeds window bound")
    return window, step


def verify_records(records, fp, fr, sp, sr, tp, tr, *, tolerance=1e-7, max_evaluations=4095):
    require(type(records) is list and len(records) == len(fp), "one certificate per window")
    position, lower, upper, alpha, verified = [], [], [], [], []
    for index, record in enumerate(records):
        result = _audit.verify_constant(record, *(x[index:index + 1] for x in (fp, fr, sp, sr, tp, tr)),
                                        tolerance=tolerance, max_evaluations=max_evaluations)
        position.append(record["position"]["objective_m2"])
        lower.append(record["rotation"]["lower_bound"])
        upper.append(record["rotation"]["upper_bound"])
        alpha.append(record["alpha"])
        verified.append(result)
    window, step = hard_errors(fp, fr, sp, sr, tp, tr)
    arrays = {"hard_window": window, "hard_step": step,
              "continuous_position_mse": np.asarray(position, np.float64),
              "continuous_rotation_lower_mse": np.asarray(lower, np.float64),
              "continuous_rotation_upper_mse": np.asarray(upper, np.float64),
              "alpha": np.asarray(alpha, np.float64),
              "certified": np.asarray([r["certified"] for r in verified], bool),
              "calls": np.asarray([r["call_count"] for r in verified], np.int64)}
    require(all(np.isfinite(x).all() for x in arrays.values()), "nonfinite oracle arithmetic")
    return arrays, verified


def verify_arrays(saved, expected):
    require(set(saved) == set(expected), "capacity array fields")
    for key, value in expected.items():
        actual = saved[key]
        require(actual.dtype == value.dtype and actual.shape == value.shape and np.isfinite(actual).all(),
                "capacity array shape/dtype " + key)
        equal = (np.array_equal(actual, value) if value.dtype != np.float64
                 else np.allclose(actual, value, rtol=1e-10, atol=1e-11))
        require(equal, "capacity array arithmetic " + key)


def capacity_comparison(lower_mse, upper_mse, control_rmse, *, projected=False, control_mse=None):
    """An optimistic bound can rule out the threshold, never prove a usable method."""
    for value in (lower_mse, upper_mse, control_rmse):
        finite(value, "capacity comparison")
    require(lower_mse <= upper_mse, "reversed capacity interval")
    control_mse = control_rmse ** 2 if control_mse is None else finite(control_mse, "control MSE")
    low, high = math.sqrt(lower_mse), math.sqrt(upper_mse)
    threshold = Decimal(".81") * Decimal(str(control_mse))
    excluded = Decimal(str(lower_mse)) > threshold
    reaches = control_mse > 0 and Decimal(str(upper_mse)) <= threshold
    result = {"lower_rmse": low, "upper_rmse": high, "control_rmse": control_rmse,
              "control_mse": control_mse, "threshold_mse": float(threshold), "threshold_rmse": math.sqrt(float(threshold)),
              "optimistic_improvement_percent": None if control_rmse == 0 else 100 * (1 - low / control_rmse)}
    if projected:
        result.update(projected_lower_below_reported_threshold=not excluded,
                      projected_upper_below_reported_threshold=reaches,
                      scope="SVD-projected interpolation versus reported threshold only; no original-float32 exclusion or usable forecast claim")
    else:
        result.update(excluded_even_optimistically=excluded, observed_oracle_reaches_threshold=reaches,
                      interval_status="excluded" if excluded else "oracle_reaches" if reaches else "unresolved")
    return result


def interval_metrics(lower, upper, ids):
    require(lower.shape == upper.shape == (len(ids),) and np.isfinite(lower).all()
            and np.isfinite(upper).all() and (lower >= 0).all() and (lower <= upper).all(), "window interval")
    parents = sorted(np.unique(ids[:, 0]).tolist())
    return {"lower_mse": float(lower.mean()), "upper_mse": float(upper.mean()),
            "lower_rmse": math.sqrt(float(lower.mean())), "upper_rmse": math.sqrt(float(upper.mean())),
            "parent_ids": parents,
            "parent_lower_mse": [float(lower[ids[:, 0] == parent].mean()) for parent in parents],
            "parent_upper_mse": [float(upper[ids[:, 0] == parent].mean()) for parent in parents]}


def aggregate(panel_arrays, identifiers, parent_summary):
    families, comparisons = {}, {}
    for panel in PANELS:
        arrays = {key: np.concatenate([panel_arrays[panel][seed][key] for seed in SEEDS])
                  for key in panel_arrays[panel][SEEDS[0]]}
        ids = np.concatenate([identifiers[panel] for _ in SEEDS])
        family = {kind: {endpoint: interval_metrics(arrays[kind][:, index], arrays[kind][:, index], ids)
                         for index, endpoint in enumerate(ENDPOINTS)} for kind in ("hard_window", "hard_step")}
        family["continuous"] = {
            "position": interval_metrics(arrays["continuous_position_mse"], arrays["continuous_position_mse"], ids),
            "rotation": interval_metrics(arrays["continuous_rotation_lower_mse"], arrays["continuous_rotation_upper_mse"], ids)}
        families[panel], comparisons[panel] = family, {}
        for kind, endpoints in family.items():
            comparisons[panel][kind] = {}
            for endpoint, metric in endpoints.items():
                against = {}
                for control in CONTROLS:
                    group = "families" if control in _audit.VARIANTS else "references"
                    baseline = parent_summary[group][panel][control][endpoint]
                    against[control] = capacity_comparison(metric["lower_mse"], metric["upper_mse"], baseline["rmse"],
                        control_mse=baseline["mse"], projected=kind == "continuous" and endpoint == "rotation")
                comparisons[panel][kind][endpoint] = {
                    "controls": against, "control_count": len(against)}
    return families, comparisons


def verify_trace(payload, panel, seed, ids, data):
    require(set(payload) == {"panel", "seed", "records"} and payload["panel"] == panel
            and payload["seed"] == seed and type(payload["records"]) is list
            and len(payload["records"]) == len(ids), "trace panel/seed/coverage")
    records, seconds = [], []
    for index, (record, identifier) in enumerate(zip(payload["records"], ids, strict=True)):
        require(set(record) == {"window", "source_id", "window_start", "optimization", "seconds"}
                and type(record["window"]) is int and record["window"] == index
                and type(record["source_id"]) is int and type(record["window_start"]) is int
                and [record["source_id"], record["window_start"]] == identifier.tolist(), "trace window provenance")
        records.append(record["optimization"])
        seconds.append(finite(record["seconds"], "constant search seconds"))
    arrays, verified = verify_records(records, *data)
    arrays["ids"] = ids
    return arrays, verified, sum(seconds)


def expected_members():
    names = {"started.json", "completed.json"}
    names.update(f"{panel}-{seed}-{suffix}" for panel in PANELS for seed in SEEDS
                 for suffix in ("traces.json.gz", "bounds.npz", "row.json"))
    require(len(names) == 20, "internal completed membership")
    return names


def authenticate_parent(experiment, report):
    context = _audit.validate_inputs(experiment, PARENT_PROTOCOL)
    run = experiment / "run-01"
    before = members(run)
    require(set(before) == _audit.expected_members() and before["completed.json"]["sha256"] == PARENT_COMPLETED,
            "parent execution membership/completion")
    done = read_json(run / "completed.json")
    require(done["status"] == "completed" and done["protocol_sha256"] == PARENT_PROTOCOL
            and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "parent execution seal")
    report_members = members(report)
    require(set(report_members) == {"summary.json", "window-errors.npz", "receipt.json"}
            and report_members["summary.json"]["sha256"] == PARENT_SUMMARY
            and report_members["receipt.json"]["sha256"] == PARENT_RECEIPT, "parent audited report identity")
    summary, receipt = read_json(report / "summary.json"), read_json(report / "receipt.json")
    require(summary["status"] == receipt["status"] == "completed"
            and summary["protocol_sha256"] == receipt["protocol_sha256"] == PARENT_PROTOCOL
            and summary["execution_completed_sha256"] == receipt["execution_completed_sha256"] == PARENT_COMPLETED
            and receipt["execution_members"] == before
            and receipt["files"] == {k: v for k, v in report_members.items() if k != "receipt.json"}, "parent audited seal")
    require(summary["audit_repair"] == receipt["audit_repair"]
            and receipt["audit_repair"]["repair_source_sha256"] == REPAIR_SHA256
            and receipt["audit_repair"]["repair_test_sha256"] == REPAIR_TEST_SHA256, "parent signed-scalar repair binding")
    for panel in PANELS:
        require(set(summary["families"][panel]) == set(_audit.VARIANTS)
                and set(summary["references"][panel]) == set(_audit.REFERENCES), "all33 retained controls")
    return {"run": run, "members": before, "summary": summary, "report": report,
            "report_members": report_members, "context": context}


def load_inputs(parent, panel, seed):
    run = parent["run"]
    with np.load(run / f"{panel}-targets.npz", allow_pickle=False) as value:
        require(set(value.files) == {"p", "R", "ids"}, "parent target fields")
        tp, tr, ids = value["p"], value["R"], value["ids"]
    require(tp.dtype == tr.dtype == np.float32 and tp.shape == (160, 25, 3)
            and tr.shape == (160, 25, 3, 3) and np.isfinite(tp).all() and np.isfinite(tr).all()
            and ids.dtype == np.int64 and ids.shape == (160, 2)
            and len(np.unique(ids, axis=0)) == 160
            and np.array_equal(np.unique(ids[:, 0], return_counts=True)[0], np.arange(10))
            and np.array_equal(np.unique(ids[:, 0], return_counts=True)[1], np.full(10, 16)), "parent target coverage")
    fp, fr, fa = _audit._previous.load_prediction(run / f"{panel}-fast-{seed}-predictions.npz", alpha=True)
    sp, sr, sa = _audit._previous.load_prediction(run / f"{panel}-slow-{seed}-predictions.npz", alpha=True)
    require(all(x.dtype == np.float32 for x in (fp, fr, sp, sr))
            and np.array_equal(fa, np.ones((160, 2), np.float32))
            and np.array_equal(sa, np.zeros((160, 2), np.float32)), "fixed expert forecast identity")
    return ids, (fp, fr, sp, sr, tp, tr)


def validate_inputs(experiment, protocol_sha256):
    require(sha(experiment / "protocol.json") == protocol_sha256, "external protocol digest")
    protocol = read_json(experiment / "protocol.json")
    required = {
        "study": "pose-capacity-v1", "scope": "post-outcome hindsight capacity diagnosis; evaluation targets inform oracle choices",
        "parent_protocol_sha256": PARENT_PROTOCOL, "parent_completed_sha256": PARENT_COMPLETED,
        "parent_summary_sha256": PARENT_SUMMARY, "parent_receipt_sha256": PARENT_RECEIPT,
        "seeds": list(SEEDS), "panels": list(PANELS), "windows_per_row": 160, "horizon": 25, "rows": 6,
        "constant_searches": 960, "constant_tolerance_rad_squared": 1e-7, "constant_max_evaluations": 4095,
        "position_class": "One scalar alpha in[0,1] per window, fixed across25 steps; exact clipped least squares.",
        "rotation_class": "One scalar alpha in[0,1] per window, fixed across25 steps; geodesic interpolation of float64 SO3-projected experts and projected targets.",
        "hard_window": "For each endpoint and window, min of the two expert mean squared errors over25 steps.",
        "hard_step": "For each endpoint and window, mean of the per-step minimum squared errors; separate relaxed hard-routing diagnostic, not a bound for continuous mixing.",
        "comparison": "Pool MSE across all480 seed-window pairs before square root. Reference threshold is0.81 times each reported control MSE. Equality meets the necessary10percent margin.",
        "controls": sorted(CONTROLS),
        "rotation_qualification": "Projected bounds do not uniformly bound original float32 production interpolation. Report projected lower/upper comparisons to reported thresholds explicitly; no original-production exclusion claim.",
        "no_success_claim": "Oracle labels use future targets; favorable capacity is not a causal selector, trained architecture, latency result, fresh confirmation or full continuation pass.",
        "no_selection": "All960 cases, complete traces, fixed tolerance and cap. No retries, replacements, exclusions or budget extensions; capped searches stay visible.",
        "execution_files": 20, "model_training_calls": 0, "model_inference_calls": 0}
    require(set(protocol) == set(required) | {"parent_experiment", "parent_report", "sources", "runtime"}
            and all(protocol[k] == value for k, value in required.items()), "frozen capacity settings")
    require(len(SOURCES) == 30 and set(protocol["sources"]) == SOURCES
            and protocol["sources"] == {k: sha(ROOT / k) for k in SOURCES}
            == {k: v["sha256"] for k, v in members(experiment / "source-snapshot").items()}, "all30 source bindings")
    require(protocol["sources"]["scripts/audit_pose_crossfit_repair.py"] == REPAIR_SHA256
            and protocol["sources"]["tests/test_audit_pose_crossfit_repair.py"] == REPAIR_TEST_SHA256,
            "exact signed scalar repair")
    require(protocol["runtime"] == {"python": platform.python_version(), "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"), "platform": platform.platform()}, "runtime identity")
    parent = authenticate_parent(Path(protocol["parent_experiment"]), Path(protocol["parent_report"]))
    return protocol, parent


def audit(experiment, out, *, protocol_sha256, completed_sha256):
    start = time.perf_counter()
    require(not out.resolve().is_relative_to(experiment.resolve()), "audit output inside experiment")
    out.mkdir(parents=True, exist_ok=False)
    try:
        protocol, parent = validate_inputs(experiment, protocol_sha256)
        require(not out.resolve().is_relative_to(parent["run"]) and not out.resolve().is_relative_to(parent["report"]),
                "audit output inside parent evidence")
        run = experiment / "run-01"
        before = members(run)
        require(set(before) == expected_members() and before["completed.json"]["sha256"] == completed_sha256,
                "external execution digest and exact20 members")
        done = read_json(run / "completed.json")
        require(set(done) == {"status", "protocol_sha256", "rows", "constant_searches", "objective_evaluations",
                "certified_searches", "wall_seconds", "files"}
                and done["status"] == "completed" and done["protocol_sha256"] == protocol_sha256
                and done["constant_searches"] == 960 and len(done["rows"]) == 6
                and done["files"] == {k: v for k, v in before.items() if k != "completed.json"}, "producer completion seal")
        require(read_json(run / "started.json") == {"protocol_sha256": protocol_sha256}, "producer start binding")
        all_arrays, all_ids, rows, numeric_arrays = {}, {}, {}, {}
        row_index, total_calls, certified, search_seconds, row_seconds = 0, 0, 0, 0., 0.
        largest_gap, statuses = 0., {}
        for panel in PANELS:
            all_arrays[panel], rows[panel] = {}, {}
            for seed in SEEDS:
                prefix = f"{panel}-{seed}"
                ids, inputs = load_inputs(parent, panel, seed)
                if panel in all_ids:
                    require(np.array_equal(all_ids[panel], ids), "paired target ordering")
                all_ids[panel] = ids
                with gzip.open(run / f"{prefix}-traces.json.gz", "rt") as stream:
                    trace = json.load(stream)
                arrays, verified, seconds = verify_trace(trace, panel, seed, ids, inputs)
                with np.load(run / f"{prefix}-bounds.npz", allow_pickle=False) as saved:
                    verify_arrays(saved, arrays)
                row = read_json(run / f"{prefix}-row.json")
                calls, count = int(arrays["calls"].sum()), int(arrays["certified"].sum())
                require(set(row) == {"panel", "seed", "trace_sha256", "bounds_sha256", "constant_searches",
                    "objective_evaluations", "certified_searches", "search_seconds", "wall_seconds"}
                    and row == done["rows"][row_index] and row["panel"] == panel and row["seed"] == seed
                    and row["trace_sha256"] == before[f"{prefix}-traces.json.gz"]["sha256"]
                    and row["bounds_sha256"] == before[f"{prefix}-bounds.npz"]["sha256"]
                    and row["constant_searches"] == 160 and row["objective_evaluations"] == calls
                    and row["certified_searches"] == count, "row identity/counters")
                _repair.close_number(row["search_seconds"], seconds, "row search times")
                wall = finite(row["wall_seconds"], "row wall", positive=True)
                require(seconds <= wall, "row nested search timing")
                for item in verified:
                    statuses[item["status"]] = statuses.get(item["status"], 0) + 1
                    largest_gap = max(largest_gap, item["gap"])
                all_arrays[panel][seed] = arrays
                methods = {name: {endpoint: interval_metrics(arrays[name][:, i], arrays[name][:, i], ids)
                           for i, endpoint in enumerate(ENDPOINTS)} for name in ("hard_window", "hard_step")}
                methods["continuous"] = {
                    "position": interval_metrics(arrays["continuous_position_mse"], arrays["continuous_position_mse"], ids),
                    "rotation": interval_metrics(arrays["continuous_rotation_lower_mse"], arrays["continuous_rotation_upper_mse"], ids)}
                rows[panel][str(seed)] = {"metrics": methods, "work": row, "certificates": verified,
                    "alpha_mean": arrays["alpha"].mean(0).tolist(), "alpha_min": arrays["alpha"].min(0).tolist(),
                    "alpha_max": arrays["alpha"].max(0).tolist()}
                for key, value in arrays.items():
                    numeric_arrays[prefix + "__" + key] = value
                row_index += 1
                total_calls += calls
                certified += count
                search_seconds += seconds
                row_seconds += wall
        wall = finite(done["wall_seconds"], "execution wall", positive=True)
        require(row_index == 6 and done["objective_evaluations"] == total_calls
                and done["certified_searches"] == certified and row_seconds <= wall + 1e-6, "complete work/timing accounting")
        families, comparisons = aggregate(all_arrays, all_ids, parent["summary"])
        summary = {"status": "completed", "study": "pose-capacity-v1", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "parent_summary_sha256": PARENT_SUMMARY,
            "rows": rows, "families": families, "comparisons": comparisons,
            "controls": {"families": parent["summary"]["families"], "references": parent["summary"]["references"]},
            "certificate_summary": {"searches": 960, "certified": certified, "uncertified": 960 - certified,
                "all_certified": certified == 960, "precise_projected_continuous_claims_available": certified == 960,
                "statuses": statuses, "maximum_gap_rad_squared": largest_gap, "objective_evaluations": total_calls},
            "counts": {"rows": 6, "windows_per_row": 160, "horizon": 25, "controls": 33, "execution_files": 20,
                "verified_searches": 960, "new_model_calls": 0, "new_optimizer_calls": 0, "new_native_calls": 0,
                "new_random_draws": 0, "new_constant_optimizations_by_audit": 0},
            "costs": {"execution_wall_seconds": wall, "row_wall_seconds": row_seconds, "search_seconds": search_seconds,
                "scope": "Nested numeric oracle-search cost, not forecast latency. Original model and data costs remain inherited."},
            "no_qualification_gate": True,
            "limits": ["Post-outcome target-aware capacity diagnosis on exposed development windows; not an implementable forecast or new architecture.",
                "Hard-window and hard-step independently select each endpoint. Stepwise hard routing relaxes windowwise hard routing only, not continuous mixtures.",
                "Continuous position is the scalar clipped least-squares optimum for each complete25-step window.",
                "Continuous rotation bounds cover guarded float64 SVD-projected interpolation, not uniform original-float32 production error or formal interval arithmetic.",
                "Every search including any capped trace remains retained. Intervals stay informative without certification; precise continuous statements require all960 certificates.",
                "All33 reported control metrics are inherited from the externally pinned completed audit, whose full payload hashes are rechecked. No control forecasts are rerun.",
                "The mean-margin comparison alone says nothing about paired/parent/leave-one-out requirements, latency, fresh generalization or full continuation.",
                "No model, optimizer, simulator, random draw or constant reoptimization was invoked by this audit."]}
        write_json(out / "summary.json", summary)
        with (out / "window-errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **numeric_arrays)
        require(members(run) == before and members(parent["run"]) == parent["members"]
                and members(parent["report"]) == parent["report_members"], "evidence changed during audit")
        validate_inputs(experiment, protocol_sha256)
        receipt = {"status": "completed", "protocol_sha256": protocol_sha256,
            "execution_completed_sha256": completed_sha256, "parent_summary_sha256": PARENT_SUMMARY,
            "parent_receipt_sha256": PARENT_RECEIPT, "execution_members": before, "sources": protocol["sources"],
            "parent_execution_members": parent["members"], "parent_report_members": parent["report_members"],
            "auditor_sha256": sha(__file__), "inherited_auditor_sha256": _repair.ORIGINAL_SHA256,
            "signed_scalar_repair_sha256": REPAIR_SHA256, "files": members(out), "all_certified": certified == 960,
            "certified_searches": certified, "verified_searches": 960, "objective_evaluations": total_calls,
            "no_qualification_gate": True, "new_model_calls": 0, "new_optimizer_calls": 0,
            "new_native_calls": 0, "new_random_draws": 0, "new_constant_optimizations": 0,
            "wall_seconds": time.perf_counter() - start}
        write_json(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write_json(out / "failed.json", {"status": "failed", "error": repr(error),
                "wall_seconds": time.perf_counter() - start, "auditor_sha256": sha(__file__)})
        except BaseException as secondary:  # noqa: BLE001 - preserve original audit error
            error.add_note(f"Could not retain failure receipt: {secondary!r}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--completed-sha256", required=True)
    args = parser.parse_args()
    result = audit(args.experiment, args.out, protocol_sha256=args.protocol_sha256, completed_sha256=args.completed_sha256)
    print(json.dumps({k: result[k] for k in ("status", "all_certified", "verified_searches", "objective_evaluations", "wall_seconds")}))
