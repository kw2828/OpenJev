"""Pure synthetic qualification of raw scoring, fixed gates and completion boundaries."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import dialogue_observation_metrics as original

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts"))
SPEC = importlib.util.spec_from_file_location("warm_pooling_report_tested", REPO/"scripts/report_dialogue_warm_pooling.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def rows_fixture():
    rows = []
    for did, service, unseen, query in (("d0", "seen-service", False, 0), ("d1", "unseen-service", True, 1)):
        for time, (bin_name, label) in enumerate((("unmentioned_retention", 0), ("first_assignment", 2), ("assigned_retention", 2))):
            ids = [report.NONE, report.DONTCARE, "value:red"]
            stratum = bin_name if bin_name in report.STRATA[:2] else "changed"
            rows.append({"row_index": len(rows), "split": "dev", "dialogue_id": did,
                         "source_row_index": time, "time": time*2, "turn_index": time*4,
                         "query_index": query, "query_position": 0, "query_id": json.dumps([service, "slot"]),
                         "service": service, "slot": "slot", "candidate_ids": ids, "candidate_values": [None, None, "red"],
                         "label_index": label, "label_id": ids[label], "bin": bin_name, "stratum": stratum,
                         "stratum_index": report.STRATA.index(stratum), "unseen": unseen, "dontcare": False})
    return rows


def logs_fixture():
    result = np.full((6, 12), -np.inf, np.float32)
    result[:, :3] = np.log([[.6, .1, .3], [.1, .1, .8], [.1, .1, .8]]*2).astype(np.float32)
    return result


def test_original_full_metric_tree_preserved_and_closed_form():
    rows, logs, indices = rows_fixture(), logs_fixture(), np.arange(6, dtype=np.int64)
    got = report.score(logs, indices, rows)
    assert {k: v for k, v in got.items() if k != "recovery"} == original.score_panels(logs, rows, row_indices=indices)
    assert set(got["panels"]) == {"all", "seen", "unseen"}
    assert set(got["services"]) == {"seen-service", "unseen-service"}
    assert got["panels"]["all"]["bins"]["clear"]["accuracy"] is None
    nll = math.fsum(-float(logs[i, row["label_index"]]) for i, row in enumerate(rows))/6
    assert got["panels"]["all"]["micro"]["nll"] == nll
    residual = np.exp(logs.astype(np.float64))
    residual[np.arange(6), [r["label_index"] for r in rows]] -= 1
    assert got["panels"]["all"]["micro"]["brier"] == math.fsum(np.square(residual).sum(1))/6
    json.dumps(got, allow_nan=False)


def test_extreme_finite_log_target_and_canonical_tie():
    rows, logs = rows_fixture(), logs_fixture()
    logs[0, :3] = [-1000., -1000., 0.]
    logs[3, :3] = np.log([.5, .5, 1e-10]).astype(np.float32)
    got = report.score(logs, np.arange(6, dtype=np.int64), rows)
    assert got["validation"]["float64_exponent_underflow_positions"] == 2
    assert got["validation"]["exact_top1_tie_rows"] == 1
    assert got["panels"]["seen"]["strata"]["unmentioned_retention"]["nll"] == 1000.
    assert got["panels"]["unseen"]["strata"]["unmentioned_retention"]["correct"] == 1


@pytest.mark.parametrize("defect", ["padding", "supported_inf", "nan", "mass", "dtype", "indices", "candidate", "source", "query", "stratum"])
def test_raw_packet_and_identity_rejection(defect):
    rows, logs, indices = rows_fixture(), logs_fixture(), np.arange(6, dtype=np.int64)
    if defect == "padding":
        logs[0, 3] = -1000.
    elif defect == "supported_inf":
        logs[0, 0] = -np.inf
    elif defect == "nan":
        logs[0, 0] = np.nan
    elif defect == "mass":
        logs[0, :3] = 0.
    elif defect == "dtype":
        logs = logs.astype(np.float64)
    elif defect == "indices":
        indices = indices[::-1]
    elif defect == "candidate":
        rows[0]["candidate_ids"][2] = report.NONE
    elif defect == "source":
        rows[1]["source_row_index"] = 2
    elif defect == "query":
        rows[0]["query_id"] = '["other","slot"]'
    else:
        rows[0]["stratum_index"] = 2
    with pytest.raises(ValueError):
        report.score(logs, indices, rows)


def gate_fixture(count=1000):
    fits = {}
    for name in report.SCORE_ORDER:
        new = name.startswith("belief_query-")
        panels = {}
        for panel in ("seen", "unseen"):
            hits = count//2 + (count//50 if new and panel == "unseen" else 0)
            panels[panel] = {"strata": {s: {"count": count, "correct": hits} for s in report.STRATA},
                             "micro": {"count": 3*count, "nll": .5, "brier": .5}}
        panels["all"] = {"strata": {s: {"count": 2*count, "correct": panels["seen"]["strata"][s]["correct"]
                                          + panels["unseen"]["strata"][s]["correct"]} for s in report.STRATA},
                         "micro": {"count": 6*count, "nll": .5, "brier": .5}}
        fits[name] = {"panels": panels}
    return fits


def set_correct(fit, panel, stratum, value):
    fit["panels"][panel]["strata"][stratum]["correct"] = value
    fit["panels"]["all"]["strata"][stratum]["correct"] = sum(fit["panels"][p]["strata"][stratum]["correct"] for p in ("seen", "unseen"))


def checks(fits, name):
    return [c for c in report.criteria(fits)["checks"] if c["name"] == name]


def test_all_thirty_two_conditions_and_untouched_control_are_mandatory():
    fits = gate_fixture()
    gate = report.criteria(fits)
    assert gate["passed"] and gate["checks_passed"] == gate["checks_total"] == 32
    assert {c["comparator"] for c in gate["checks"]} == set(report.COMPARATORS)
    for seed in report.SEEDS:
        for s in report.STRATA:
            set_correct(fits[f"untouched-{seed}"], "unseen", s, 600)
    gate = report.criteria(fits)
    assert not gate["passed"]
    assert all(c["passed"] for c in gate["checks"] if c["comparator"] != "untouched")


@pytest.mark.parametrize("name, group", [("unseen_macro_gain_at_least_1pp", None), ("unseen_changed_gain_at_least_1pp", "changed")])
def test_exact_one_percent_boundary_cannot_round_to_pass(name, group):
    count = 10**18
    fits = gate_fixture(count)
    for seed in report.SEEDS:
        for s in report.STRATA:
            set_correct(fits[f"belief_query-{seed}"], "unseen", s, count//2+count//100)
    assert all(c["passed"] for c in checks(fits, name))
    set_correct(fits["belief_query-6901"], "unseen", group or "unmentioned_retention", count//2+count//100-1)
    values = checks(fits, name)
    assert all(not c["passed"] and c["mean_paired_change"] == .01 for c in values)
    assert Fraction(**values[0]["mean_paired_change_exact"]) < Fraction(1, 100)


def test_two_positive_seeds_required_but_third_stays_visible():
    fits = gate_fixture()
    for seed, hits in ((6901, 530), (6902, 530), (6903, 490)):
        for s in report.STRATA:
            set_correct(fits[f"belief_query-{seed}"], "unseen", s, hits)
    assert all(c["passed"] and c["positive_seeds"] == 2 for c in checks(fits, "unseen_macro_positive_at_least_two_seeds"))
    for s in report.STRATA:
        set_correct(fits["belief_query-6902"], "unseen", s, 500)
    assert all(not c["passed"] and len(c["paired_seed_changes"]) == 3 for c in checks(fits, "unseen_macro_positive_at_least_two_seeds"))


@pytest.mark.parametrize("metric", ["nll", "brier"])
def test_proper_scores_strict_float64_arm_means_without_epsilon(metric):
    fits = gate_fixture()
    assert all(c["passed"] for c in checks(fits, "unseen_"+metric+"_nonworse"))
    after = float(np.nextafter(.5, math.inf))
    assert math.fsum([after]*3)/3 > math.fsum([.5]*3)/3
    for seed in report.SEEDS:
        fits[f"belief_query-{seed}"]["panels"]["unseen"]["micro"][metric] = after
    assert all(not c["passed"] for c in checks(fits, "unseen_"+metric+"_nonworse"))


@pytest.mark.parametrize("panel", ["seen", "unseen"])
def test_half_point_retention_boundary_is_exact(panel):
    fits = gate_fixture(10**18)
    count = 10**18
    for seed in report.SEEDS:
        set_correct(fits[f"belief_query-{seed}"], panel, "assigned_retention", count//2-count//200)
    name = panel+"_assigned_retention_error_increase_at_most_half_pp"
    assert all(c["passed"] for c in checks(fits, name))
    set_correct(fits["belief_query-6901"], panel, "assigned_retention", count//2-count//200-1)
    assert all(not c["passed"] and c["mean_paired_change"] == .005 for c in checks(fits, name))


def test_seen_macro_one_point_decline_boundary_is_exact():
    fits = gate_fixture(10**18)
    count = 10**18
    for seed in report.SEEDS:
        for s in report.STRATA:
            set_correct(fits[f"belief_query-{seed}"], "seen", s, count//2-count//100)
    name = "seen_macro_decline_at_most_1pp"
    assert all(c["passed"] for c in checks(fits, name))
    set_correct(fits["belief_query-6901"], "seen", "changed", count//2-count//100-1)
    assert all(not c["passed"] and c["mean_paired_change"] == -.01 for c in checks(fits, name))


@pytest.mark.parametrize("defect", ["missing_fit", "extra_fit", "missing_stratum", "support", "partition", "nonfinite"])
def test_incomplete_gate_never_passes(defect):
    fits = gate_fixture()
    first = fits[report.SCORE_ORDER[0]]
    if defect == "missing_fit":
        del fits[report.SCORE_ORDER[-1]]
    elif defect == "extra_fit":
        fits["other"] = copy.deepcopy(first)
    elif defect == "missing_stratum":
        first["panels"]["unseen"]["strata"]["changed"]["count"] = 0
    elif defect == "support":
        first["panels"]["unseen"]["micro"]["count"] += 1
    elif defect == "partition":
        first["panels"]["all"]["strata"]["changed"]["correct"] += 1
    else:
        first["panels"]["unseen"]["micro"]["nll"] = math.nan
    with pytest.raises(ValueError):
        report.criteria(fits)


class FakeBudget:
    def __init__(self, out):
        self.out = out
        self.clock = SimpleNamespace(backend="CLOCK_BOOTTIME")

    def check(self):
        pass

    def storage(self):
        pass

    def elapsed(self):
        return .5


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    report.write(path, value)


def overwrite(path, value):
    path.write_text(json.dumps(value))


def save_predictions(path):
    with path.open("xb") as stream:
        np.savez(stream, log_probs=logs_fixture(), row_indices=np.arange(6, dtype=np.int64))


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "TRAIN_DIALOGUES", 2)
    monkeypatch.setattr(report, "DEV_DIALOGUES", 2)
    monkeypatch.setattr(report, "DEV_ROWS", 6)
    monkeypatch.setattr(report, "runtime", lambda: {"artificial": True})
    monkeypatch.setattr(report, "Budget", FakeBudget)
    monkeypatch.setattr(report.signal, "signal", lambda *_: 1)
    monkeypatch.setattr(report.signal, "setitimer", lambda *_: None)
    sources = {}
    for name in report.SOURCES | {report.RUNNER, report.CLOCK, report.SUPERVISOR}:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source " + name)
        sources[name] = report.sha(path)
    study_path, run, original_run = tmp_path/"study-freeze/plan.json", tmp_path/"run", tmp_path/"original"
    run.mkdir()
    original_run.mkdir()
    limits = {"wall_seconds": 7200, "rss_bytes": 16*1024**3, "mps_driver_bytes": 8*1024**3, "output_bytes": 32*1024**3}
    selected = {"train": ["t0", "t1"], "dev": ["d0", "d1"]}
    study = {"version": report.STUDY_VERSION, "fit_order": report.FIT_ORDER, "limits": {"run": limits},
             "source_sha256": {k: sources[k] for k in (report.RUNNER, report.CLOCK, report.SUPERVISOR)},
             "selected": selected, "loss_counts": {s: {str(i): 2 for i in range(3)} for s in ("train", "dev")},
             "checkpoints": {"run": str(original_run), "fits": {str(s): {"encoder_sha256": "e"*64} for s in report.SEEDS}}}
    put(study_path, study)
    (run/"plan.json").write_bytes(study_path.read_bytes())
    study_pin = report.sha(study_path)
    launch_path, terminal_path = tmp_path/"launch.json", tmp_path/"terminal.json"
    request = {"command": "run", "plan": str(study_path), "plan_sha256": study_pin,
               "supervision": str(launch_path), "out": str(run)}
    command = [sys.executable, str(tmp_path/report.RUNNER), "run", "--plan", str(study_path),
               "--plan-sha256", study_pin, "--supervision", str(launch_path), "--out", str(run)]
    launch = {"version": "dialogue-observation-supervision-v2", "command": command, "cwd": str(tmp_path),
              "parent_pid": 10, "pid": 20, "pgid": 20, "cap_seconds": 7200, "clock_backend": "CLOCK_BOOTTIME",
              "started_ns": 100, "deadline_ns": 7200_000_000_100,
              "watchdog_sha256": sources[report.SUPERVISOR], "clock_source_sha256": sources[report.CLOCK]}
    put(launch_path, launch)
    terminal = {**launch, "status": "completed", "returncode": 0, "group_absent": True, "timed_out": False,
                "error": None, "clock_error": None, "timing_available": True, "finished_ns": 400,
                "elapsed_ns": 300, "wall_seconds": 300/1e9, "cleanup": {"group_absent": True, "reaped": True, "errors": []}}
    put(terminal_path, terminal)
    put(run/"started.json", {"request": request, "limits": limits, "clock_backend": "CLOCK_BOOTTIME",
                            "worker_started_ns": 150, "parent_started_ns": 100, "deadline_ns": launch["deadline_ns"],
                            "supervision_sha256": report.sha(launch_path)})
    for path in (run/"rows-train.jsonl", run/"rows-dev.jsonl", original_run/"evaluation-rows.jsonl"):
        path.write_text("".join(json.dumps(row)+"\n" for row in rows_fixture()))
    caches = []
    for seed in report.SEEDS:
        directory = run/f"cache-{seed}"
        directory.mkdir()
        for name in ("tokens.npy", "pooled.npy", "offsets.npy", "index.json"):
            (directory/name).write_bytes(b"synthetic cache; never decoded")
        cache = {"status": "completed", "seed": seed, "encoder_sha256": "e"*64,
                 "wall_seconds": .01, "encoder_work": {"encoder_calls": 2}, "files": report.members(directory)}
        put(directory/"completed.json", cache)
        caches.append(cache)
    qualifications = []
    for name in report.QUALIFICATION_ORDER:
        directory = run/"qualification"/name
        directory.mkdir(parents=True)
        save_predictions(directory/"predictions.npz")
        record = {"status": "completed", "fit_id": name, "initial_state_sha256": "a"*64,
                  "parity": {"passed": True, "endpoints": 6, "maximum_log_difference": 0., "maximum_probability_difference": 0.,
                             "selected_choice_changes": 0, "top_tie_mask_changes": 0},
                  "counts": {"evaluation_forwards": 2}, "work": {}, "rows": 6, "wall_seconds": .01,
                  "files": report.members(directory)}
        put(directory/"completed.json", record)
        qualifications.append(record)
    put(run/"qualification.json", {"status": "completed", "paths": qualifications,
                                   "training_started": False, "optimizer_created": False, "temperature_applied": False})
    fits = []
    for name in report.FIT_ORDER:
        arm, seed = name.rsplit("-", 1)
        directory = run/name
        directory.mkdir()
        (directory/"weights.pt").write_bytes(b"opaque fake checkpoint; never deserialize")
        (directory/"updates.jsonl").write_text("opaque fake updates; never decode")
        save_predictions(directory/"predictions.npz")
        fit = {"status": "completed", "fit_id": name, "method": arm, "seed": int(seed),
               "initial_state_sha256": "a"*64, "final_state_sha256": "f"*64,
               "training_counts": {"optimizer_updates": 5, "optimizer_attempts": 5, "forward_attempts": 10,
                                   "forward_returns": 10, "training_forwards": 10, "backward_attempts": 10,
                                   "backward_returns": 10, "training_endpoints": 30}, "training_work": {},
               "evaluation": {"rows": 6, "counts": {"evaluation_forwards": 2, "forward_attempts": 2, "forward_returns": 2},
                              "work": {}, "wall_seconds": .02},
               "files": report.members(directory), "training_seconds": .1, "wall_seconds": .13,
               "registered_parameters": 10, "gradient_present_parameters": 10, "nonzero_gradient_parameters": 10,
               "configuration": {"method": arm}}
        put(directory/"completed.json", fit)
        fits.append({**fit, "completed_sha256": report.sha(directory/"completed.json")})
    done = {"status": "completed", "phase": "run", "version": report.STUDY_VERSION, "fit_order": report.FIT_ORDER,
            "fits": fits, "plan_sha256": study_pin, "source_sha256": study["source_sha256"],
            "quality_metrics_computed": False, "official_test_opened": False, "temperature_applied": False,
            "limits": limits, "peak_rss_bytes": 1000, "sampled_mps_driver_max_bytes": 0,
            "files": report.members(run), "request": request, "clock_backend": "CLOCK_BOOTTIME",
            "worker_started_ns": 150, "started_ns": 100, "deadline_ns": launch["deadline_ns"], "supervision_sha256": report.sha(launch_path),
            "timing_available": True, "finished_ns": 300, "elapsed_ns": 200, "wall_seconds": 200/1e9,
            "caches": caches, "qualification_paths": qualifications, "row_counts": {"train": 6, "dev": 6}}
    put(run/"completed.json", done)
    source_pins = {k: sources[k] for k in report.SOURCES}
    qual = tmp_path/"synthetic-qualification.json"
    put(qual, {"status": "completed", "actual_exit_code": 0, "source_sha256": source_pins, "model_calls": 0, "real_task_inputs_read": False})
    plan = {"version": report.VERSION, "limits": report.LIMITS, "sources": source_pins, "runtime": {"artificial": True},
            "synthetic_qualification": {"path": qual.name, "sha256": report.sha(qual)},
            "study_plan": {"path": str(study_path.relative_to(tmp_path)), "sha256": study_pin},
            "run": {"path": "run", "completed_sha256": report.sha(run/"completed.json"),
                    "terminal": {"path": terminal_path.name, "sha256": report.sha(terminal_path)}}, "out": report.BASE+"/report-01"}
    plan_path = tmp_path/report.BASE/"report-plan-01.json"
    put(plan_path, plan)
    args = SimpleNamespace(plan=plan_path, plan_sha256=report.sha(plan_path), out=tmp_path/plan["out"])
    calls = []
    def authenticate(path, digest, out):
        calls.append("upstream-read-only")
        assert path == study_path and digest == study_pin and out == run
        return copy.deepcopy(study), {}
    monkeypatch.setattr(report, "upstream_authenticate", authenticate)
    return SimpleNamespace(args=args, plan=plan, study=study, done=done, terminal=terminal, launch=launch,
                           run=run, root=tmp_path, terminal_path=terminal_path, calls=calls)


def repin(f):
    f.done["files"] = report.members(f.run)
    overwrite(f.run/"completed.json", f.done)
    f.plan["run"]["completed_sha256"] = report.sha(f.run/"completed.json")
    f.plan["run"]["terminal"]["sha256"] = report.sha(f.terminal_path)
    overwrite(f.args.plan, f.plan)
    f.args.plan_sha256 = report.sha(f.args.plan)


def test_complete_synthetic_report_keeps_fifteen_raw_fits_and_all_conditions(artifacts):
    f = artifacts
    receipt = report.execute(f.args)
    summary = report.read(f.args.out/"summary.json")
    assert receipt["status"] == "completed" and receipt["complete_adapted_fits"] == 12
    assert receipt["complete_untouched_references"] == 3 and receipt["scored_fits"] == 15
    assert not receipt["temperature_fitted"] and receipt["model_calls"] == 0
    assert summary["continuation"]["checks_total"] == 32 and not summary["continuation"]["passed"]
    assert set(summary["fits"]) == set(report.SCORE_ORDER)
    assert set(summary["families"]) == {*report.ARMS, "untouched"}
    assert set(summary["contrasts"]) == set(report.COMPARATORS)
    assert all(set(c["paired_seeds"]) == set(map(str, report.SEEDS)) for c in summary["contrasts"].values())
    assert f.calls == ["upstream-read-only", "upstream-read-only"]
    assert "All three seeds" in (f.args.out/"report.md").read_text()
    report.manifest(f.args.out, receipt["files"], "receipt.json")


@pytest.mark.parametrize("defect", ["source", "plan", "missing_prediction", "extra_payload", "symlink", "failed_terminal",
                                  "late_terminal", "command_mismatch", "incomplete_fit", "unqualified", "cache_encoder",
                                  "missing_qualification", "parity_failed", "parity_tolerance", "tie_change", "bad_counts", "initialization"])
def test_failures_precede_prediction_decode(artifacts, monkeypatch, defect):
    f = artifacts
    if defect == "source":
        (f.root/next(iter(report.SOURCES))).write_text("changed source")
    elif defect == "plan":
        f.args.plan_sha256 = "0"*64
    elif defect == "missing_prediction":
        (f.run/report.FIT_ORDER[0]/"predictions.npz").unlink()
    elif defect == "extra_payload":
        (f.run/"extra.txt").write_text("unexpected")
        repin(f)
    elif defect == "symlink":
        (f.run/"extra.txt").symlink_to(f.args.plan)
    elif defect in ("failed_terminal", "late_terminal", "command_mismatch"):
        terminal = copy.deepcopy(f.terminal)
        if defect == "failed_terminal":
            terminal.update(status="failed", returncode=1)
        elif defect == "late_terminal":
            terminal.update(finished_ns=terminal["deadline_ns"], elapsed_ns=7200_000_000_000, wall_seconds=7200.)
        else:
            terminal["command"][-1] = "wrong-output"
        overwrite(f.terminal_path, terminal)
        repin(f)
    elif defect == "incomplete_fit":
        f.done["fits"] = f.done["fits"][:-1]
        repin(f)
    elif defect == "unqualified":
        (f.root/"synthetic-qualification.json").write_text("changed qualification")
    elif defect == "cache_encoder":
        f.done["caches"][0]["encoder_sha256"] = "b"*64
        overwrite(f.run/f"cache-{report.SEEDS[0]}"/"completed.json", f.done["caches"][0])
        repin(f)
    elif defect in ("missing_qualification", "parity_failed", "parity_tolerance", "tie_change"):
        if defect == "missing_qualification":
            f.done["qualification_paths"] = f.done["qualification_paths"][:-1]
        else:
            item = f.done["qualification_paths"][0]
            if defect == "parity_failed":
                item["parity"]["passed"] = False
            elif defect == "parity_tolerance":
                item["parity"]["maximum_log_difference"] = 1.00001e-5
            else:
                item["parity"]["top_tie_mask_changes"] = 1
            overwrite(f.run/"qualification"/item["fit_id"]/"completed.json", item)
        q = report.read(f.run/"qualification.json")
        q["paths"] = f.done["qualification_paths"]
        overwrite(f.run/"qualification.json", q)
        repin(f)
    else:
        item = f.done["fits"][0]
        if defect == "bad_counts":
            item["training_counts"]["optimizer_updates"] -= 1
        else:
            item["initial_state_sha256"] = "b"*64
        directory = f.run/item["fit_id"]
        overwrite(directory/"completed.json", {k: v for k, v in item.items() if k != "completed_sha256"})
        item["completed_sha256"] = report.sha(directory/"completed.json")
        repin(f)
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Decoded predictions before authentication"))
    with pytest.raises((ValueError, FileNotFoundError)):
        report.execute(f.args)
    assert not (f.args.out/"summary.json").exists()
    assert not report.read(f.args.out/"failed.json")["scientific_result_qualified"]


def test_canonical_ledger_mismatch_rejected_before_prediction_decode(artifacts, monkeypatch):
    f = artifacts
    rows = rows_fixture()
    rows[0]["label_id"], rows[0]["label_index"] = "value:red", 2
    (f.run/"rows-dev.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    repin(f)
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Decoded predictions before original row equality"))
    with pytest.raises(ValueError, match="unchanged original DEV"):
        report.execute(f.args)


def test_existing_output_never_modified(artifacts):
    f = artifacts
    f.args.out.mkdir()
    existing = f.args.out/"keep.txt"
    existing.write_text("original")
    with pytest.raises(FileExistsError):
        report.execute(f.args)
    assert existing.read_text() == "original" and list(f.args.out.iterdir()) == [existing]


def test_late_publication_demotes_success(artifacts, monkeypatch):
    f = artifacts
    def storage(self):
        if (self.out/"receipt.json").exists():
            raise TimeoutError("synthetic final deadline")
    monkeypatch.setattr(FakeBudget, "storage", storage)
    with pytest.raises(TimeoutError):
        report.execute(f.args)
    assert (f.args.out/"invalid-receipt.json").is_file() and not (f.args.out/"receipt.json").exists()


def test_failure_receipt_error_preserves_primary(artifacts, monkeypatch):
    f = artifacts
    f.args.plan_sha256 = "0"*64
    original_write = report.write
    def broken(path, value):
        if path.name == "failed.json":
            raise OSError("synthetic disk failure")
        return original_write(path, value)
    monkeypatch.setattr(report, "write", broken)
    with pytest.raises(ValueError, match="External report plan pin") as captured:
        report.execute(f.args)
    assert any("synthetic disk failure" in note for note in captured.value.__notes__)


def test_source_contract_exact_payload_count_and_no_inference_overrides():
    assert len(report.expected_members()) == 98
    assert set(vars(report.parse_args(["--plan", "p", "--plan-sha256", "a"*64, "--out", "o"]))) == {"plan", "plan_sha256", "out"}


def test_manifest_traversal_rejected_before_read(tmp_path):
    with pytest.raises(ValueError, match="Canonical"):
        report.manifest(tmp_path, {"../escape": {"sha256": "a"*64, "bytes": 1}})


def test_recovery_is_model_specific_descriptive_and_preserves_public_spacing():
    rows, logs = rows_fixture(), logs_fixture()
    logs[1, :3] = np.log([.7, .1, .2]).astype(np.float32)
    scored = report.score(logs, np.arange(6, dtype=np.int64), rows)
    recovery = scored["recovery"]
    assert recovery["eligible_pairs"] == 4 and recovery["first_scored_endpoints"] == 2
    assert recovery["panels"]["all"]["after_previous_scored_error"]["count"] == 1
    assert recovery["panels"]["all"]["after_previous_scored_error"]["accuracy"] == 1.
    assert recovery["panels"]["unseen"]["after_previous_scored_error"]["accuracy"] is None
    assert recovery["after_error_nonadjacent_public_turns"] == 1
    assert recovery["after_error_mean_public_turn_gap"] == recovery["after_error_max_public_turn_gap"] == 2
    assert "unscored public turns" in recovery["scope"]
