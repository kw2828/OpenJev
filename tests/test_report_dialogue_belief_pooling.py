"""Synthetic numerical and saved-only boundary qualification; no task inputs."""
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"scripts"))
SPEC = importlib.util.spec_from_file_location("belief_pooling_report_tested", REPO/"scripts/report_dialogue_belief_pooling.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def rows_fixture():
    rows = []
    for did in ("d0", "d1"):
        for time, (bin_name, label) in enumerate((("unmentioned_retention", 0), ("first_assignment", 2), ("assigned_retention", 2))):
            ids = [report.NONE, report.DONTCARE, "value:red"]
            stratum = bin_name if bin_name in report.STRATA[:2] else "changed"
            rows.append({"row_index": len(rows), "split": "dev", "dialogue_id": did,
                         "source_row_index": time, "time": time*2, "turn_index": time*4,
                         "query_index": 0, "query_position": 0, "query_id": '["service","slot"]',
                         "service": "service", "slot": "slot", "candidate_ids": ids,
                         "candidate_values": [None, None, "red"], "label_index": label,
                         "label_id": ids[label], "bin": bin_name, "stratum": stratum,
                         "stratum_index": report.STRATA.index(stratum), "unseen": False, "dontcare": False})
    return rows


def probabilities(values):
    logs = np.full((len(values), report.MAX_CANDIDATES), -np.inf, dtype=np.float32)
    logs[:, :len(values[0])] = np.log(np.asarray(values, dtype=np.float64)).astype(np.float32)
    return logs


def evaluate(logs, rows=None):
    rows = rows_fixture() if rows is None else rows
    return report.score(logs, np.arange(len(rows), dtype=np.int64), rows)


def test_closed_form_proper_scores_and_missing_descriptive_support():
    logs = probabilities([[.25, .25, .5]]*6)
    scored = evaluate(logs)
    expected_nll = math.fsum(-float(logs[i, r["label_index"]]) for i, r in enumerate(rows_fixture()))/6
    p = np.exp(logs.astype(np.float64))
    target = np.zeros_like(p)
    target[np.arange(6), [r["label_index"] for r in rows_fixture()]] = 1.
    expected_brier = math.fsum(np.square(p-target).sum(1))/6
    assert scored["panels"]["all"]["micro"]["nll"] == expected_nll
    assert scored["panels"]["all"]["micro"]["brier"] == expected_brier
    assert scored["panels"]["all"]["macro_accuracy_exact"] == {"numerator": 2, "denominator": 3}
    assert scored["panels"]["unseen"]["micro"]["count"] == 0
    assert all(scored["panels"]["unseen"]["micro"][k] is None for k in ("accuracy", "error", "nll", "brier"))
    assert scored["panels"]["unseen"]["macro_accuracy"] is None
    json.dumps(scored, allow_nan=False)


def test_direct_logs_preserve_tiny_target_without_probability_floor():
    logs = probabilities([[.25, .25, .5]]*6)
    logs[0, :3] = [-1000., -1000., 0.]
    scored = evaluate(logs)
    assert scored["validation"]["float64_underflow_positions"] == 2
    expected = math.fsum(-float(logs[i, row["label_index"]]) for i, row in enumerate(rows_fixture()))/6
    assert scored["panels"]["all"]["micro"]["nll"] == expected
    assert scored["panels"]["all"]["micro"]["nll"] > 166


def test_canonical_first_argmax_tie_and_recovery_spacing():
    logs = probabilities([[.5, .5, 1e-9], [.6, .1, .3], [.1, .1, .8]]*2)
    scored = evaluate(logs)
    assert scored["validation"]["top_tie_rows"] == 2
    assert scored["panels"]["all"]["strata"]["unmentioned_retention"]["correct"] == 2
    recovery = scored["recovery"]
    assert recovery["eligible_pairs"] == 4 and recovery["first_scored_endpoints"] == 2
    assert recovery["after_previous_scored_error"]["count"] == 2
    assert recovery["after_previous_scored_error"]["accuracy"] == 1.
    assert recovery["after_error_nonadjacent_public_turns"] == 2
    assert recovery["after_error_mean_public_turn_gap"] == recovery["after_error_max_public_turn_gap"] == 2


@pytest.mark.parametrize("defect", ["padding", "supported_inf", "nan", "mass", "dtype", "indices", "shape"])
def test_invalid_raw_packet_rejected_without_repair(defect):
    logs = probabilities([[.2, .2, .6]]*6)
    indices = np.arange(6, dtype=np.int64)
    if defect == "padding":
        logs[0, 3] = -1000
    elif defect == "supported_inf":
        logs[0, 0] = -np.inf
    elif defect == "nan":
        logs[0, 0] = np.nan
    elif defect == "mass":
        logs[0, :3] = 0
    elif defect == "dtype":
        logs = logs.astype(np.float64)
    elif defect == "indices":
        indices = indices[::-1]
    else:
        logs = logs[:, :3]
    with pytest.raises(ValueError):
        report.score(logs, indices, rows_fixture())


@pytest.mark.parametrize("defect", ["candidate_order", "duplicate_candidate", "missing_reserved", "target", "source_gap",
                                  "query_identity", "query_position", "duplicate_endpoint", "exposure", "stratum", "missing_primary_stratum"])
def test_canonical_candidate_and_endpoint_validation(defect):
    rows = rows_fixture()
    if defect == "candidate_order":
        rows[1]["candidate_ids"] = [report.DONTCARE, report.NONE, "value:red"]
    elif defect == "duplicate_candidate":
        rows[0]["candidate_ids"][2] = report.NONE
    elif defect == "missing_reserved":
        rows[0]["candidate_ids"][1] = "value:blue"
    elif defect == "target":
        rows[0]["label_id"] = "value:red"
    elif defect == "source_gap":
        rows[1]["source_row_index"] = 2
    elif defect == "query_identity":
        rows[0]["query_id"] = '["different","slot"]'
    elif defect == "query_position":
        rows[1]["query_position"] = 1
    elif defect == "duplicate_endpoint":
        rows[1]["time"] = 0
    elif defect == "exposure":
        rows[1]["unseen"] = True
    elif defect == "stratum":
        rows[0]["stratum_index"] = 2
    else:
        for row in rows:
            if row["stratum"] == "assigned_retention":
                row.update(bin="first_assignment", stratum="changed", stratum_index=2)
    with pytest.raises(ValueError):
        evaluate(probabilities([[.2, .2, .6]]*6), rows)


def gate_fixture():
    fits = {}
    for name in report.FIT_ORDER:
        is_new = name.startswith("belief_query-")
        strata = {s: {"count": 100, "correct": 52 if is_new else 50} for s in report.STRATA}
        fits[name] = {"panels": {"all": {"strata": strata, "micro": {"count": 300, "nll": .5, "brier": .25}}}}
    return fits


def test_gate_all_eighteen_and_cheap_pooled_comparator_required():
    fits = gate_fixture()
    gate = report.criteria(fits)
    assert gate["passed"] and gate["checks_passed"] == gate["checks_total"] == 18
    assert {c["comparator"] for c in gate["checks"]} == set(report.COMPARATORS)
    for seed in report.SEEDS:
        for cell in fits[f"pooled-{seed}"]["panels"]["all"]["strata"].values():
            cell["correct"] = 60
    gate = report.criteria(fits)
    assert not gate["passed"]
    assert all(c["passed"] for c in gate["checks"] if c["comparator"] != "pooled")


def test_exact_macro_threshold_does_not_round_to_pass():
    fits = gate_fixture()
    count = 10**18
    for name, fit in fits.items():
        new = name.startswith("belief_query-")
        for cell in fit["panels"]["all"]["strata"].values():
            cell.update(count=count, correct=count//2 + (count//100 if new else 0))
        fit["panels"]["all"]["micro"]["count"] = 3*count
    assert report.criteria(fits)["passed"]
    fits["belief_query-7101"]["panels"]["all"]["strata"]["changed"]["correct"] -= 1
    gate = report.criteria(fits)
    gains = [c for c in gate["checks"] if c["name"] == "macro_gain_at_least_1pp"]
    assert all(c["mean_paired_change"] == .01 and not c["passed"] for c in gains)
    assert Fraction(**gains[0]["mean_paired_change_exact"]) < Fraction(1, 100)


def test_every_seed_must_improve_even_when_mean_passes():
    fits = gate_fixture()
    for seed, correct in ((7101, 50), (7102, 54)):
        for cell in fits[f"belief_query-{seed}"]["panels"]["all"]["strata"].values():
            cell["correct"] = correct
    gate = report.criteria(fits)
    assert not gate["passed"]
    assert all(not c["passed"] for c in gate["checks"] if c["name"] == "macro_strictly_positive_each_seed")
    assert all(c["passed"] for c in gate["checks"] if c["name"] == "macro_gain_at_least_1pp")


@pytest.mark.parametrize("metric", ["nll", "brier"])
def test_proper_score_means_have_no_epsilon(metric):
    fits = gate_fixture()
    for fit in fits.values():
        fit["panels"]["all"]["micro"][metric] = .5
    for seed in report.SEEDS:
        fits[f"belief_query-{seed}"]["panels"]["all"]["micro"][metric] = float(np.nextafter(.5, math.inf))
    checks = [c for c in report.criteria(fits)["checks"] if c["name"] == metric+"_nonworse"]
    assert all(not c["passed"] for c in checks)


def test_retention_uses_exact_counts_and_missing_fit_fails():
    fits = gate_fixture()
    for seed in report.SEEDS:
        fits[f"belief_query-{seed}"]["panels"]["all"]["strata"]["assigned_retention"]["correct"] = 49
    checks = [c for c in report.criteria(fits)["checks"] if c["name"] == "assigned_retention_error_nonworse"]
    assert all(not c["passed"] and c["mean_paired_change_exact"] == {"numerator": 1, "denominator": 100} for c in checks)
    del fits[report.FIT_ORDER[-1]]
    with pytest.raises(ValueError, match="eight fixed"):
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


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "DIALOGUES", 2)
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
    pilot_path = tmp_path/"pilot-freeze/plan.json"
    run = tmp_path/"run"
    run.mkdir()
    limits = {"wall_seconds": 1800, "rss_bytes": 8*1024**3, "mps_driver_bytes": 8*1024**3, "output_bytes": 2*1024**3}
    selected = {"train": ["t0", "t1"], "dev": ["d0", "d1"]}
    pilot = {"version": report.PILOT_VERSION, "fit_order": report.FIT_ORDER, "limits": {"run": limits},
             "source_sha256": {k: sources[k] for k in (report.RUNNER, report.CLOCK, report.SUPERVISOR)},
             "selected": selected}
    put(pilot_path, pilot)
    (run/"plan.json").write_bytes(pilot_path.read_bytes())
    pilot_pin = report.sha(pilot_path)
    launch_path, terminal_path = tmp_path/"launch.json", tmp_path/"terminal.json"
    request = {"command": "run", "plan": str(pilot_path), "plan_sha256": pilot_pin,
               "supervision": str(launch_path), "out": str(run)}
    command = [sys.executable, str(tmp_path/report.RUNNER), "run", "--plan", str(pilot_path),
               "--plan-sha256", pilot_pin, "--supervision", str(launch_path), "--out", str(run)]
    launch = {"version": "dialogue-observation-supervision-v2", "command": command, "cwd": str(tmp_path),
              "parent_pid": 10, "pid": 20, "pgid": 20, "cap_seconds": 1800, "clock_backend": "CLOCK_BOOTTIME",
              "started_ns": 100, "deadline_ns": 1800_000_000_100,
              "watchdog_sha256": sources[report.SUPERVISOR], "clock_source_sha256": sources[report.CLOCK]}
    put(launch_path, launch)
    terminal = {**launch, "status": "completed", "returncode": 0, "group_absent": True, "timed_out": False,
                "error": None, "clock_error": None, "timing_available": True, "finished_ns": 400,
                "elapsed_ns": 300, "wall_seconds": 300/1e9, "cleanup": {"group_absent": True, "reaped": True, "errors": []}}
    put(terminal_path, terminal)
    put(run/"started.json", {"request": request, "limits": limits, "clock_backend": "CLOCK_BOOTTIME",
                            "worker_started_ns": 150, "parent_started_ns": 100, "deadline_ns": launch["deadline_ns"],
                            "supervision_sha256": report.sha(launch_path)})
    rows = rows_fixture()
    for split in ("train", "dev"):
        (run/f"rows-{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    cache_dir = run/"cache"
    cache_dir.mkdir()
    for name in ("tokens.npy", "pooled.npy", "offsets.npy", "index.json"):
        (cache_dir/name).write_bytes(b"artificial cache payload; never decoded")
    cache = {"maximum_pooled_error": 0., "files": report.members(cache_dir), "wall_seconds": .01}
    put(cache_dir/"completed.json", {"status": "completed", **cache})
    fits = []
    for name in report.FIT_ORDER:
        arm, seed = name.rsplit("-", 1)
        path = run/name
        path.mkdir()
        (path/"weights.pt").write_bytes(b"opaque fake checkpoint; must never load")
        (path/"updates.jsonl").write_text("opaque fake journal; must never decode")
        with (path/"predictions.npz").open("xb") as stream:
            np.savez(stream, log_probs=probabilities([[.6, .1, .3], [.1, .1, .8], [.1, .1, .8]]*2), row_indices=np.arange(6, dtype=np.int64))
        def work(n, arm=arm):
            return {"forward_attempts": n, "forward_returns": n, "observation_attempts": n*3,
                    "observation_returns": n*3, "step_attempts": n*3, "step_returns": n*3,
                    "state_checks": n*3, "real_question_updates": n*3, "attention_positions": 0 if arm == "pooled" else n*12}
        fit = {"status": "completed", "fit_id": name, "method": arm, "seed": int(seed), "evaluation_rows": 6,
               "initial_state_sha256": "a"*64, "counts": {"optimizer_updates": 3, "optimizer_attempts": 3,
                "training_forwards": 6, "backward_attempts": 6, "backward_returns": 6, "evaluation_forwards": 2,
                "forward_attempts": 8, "forward_returns": 8, "training_endpoints": 18, "public_turns": 24, "question_updates": 24},
               "training_work": work(6), "evaluation_work": work(2), "files": report.members(path),
               "training_seconds": .1, "evaluation_seconds": .02, "wall_seconds": .13, "registered_parameters": 10,
               "gradient_present_parameters": 10, "nonzero_gradient_parameters": 10, "configuration": {"method": arm}}
        put(path/"completed.json", fit)
        fits.append({**fit, "completed_sha256": report.sha(path/"completed.json")})
    done = {"status": "completed", "phase": "run", "version": report.PILOT_VERSION, "fit_order": report.FIT_ORDER,
            "fits": fits, "plan_sha256": pilot_pin, "source_sha256": pilot["source_sha256"], "selected": selected,
            "quality_metrics_computed": False, "official_test_opened": False, "limits": limits, "peak_rss_bytes": 1000,
            "sampled_mps_driver_max_bytes": 0, "files": report.members(run), "request": request, "clock_backend": "CLOCK_BOOTTIME",
            "worker_started_ns": 150, "started_ns": 100, "deadline_ns": launch["deadline_ns"], "supervision_sha256": report.sha(launch_path),
            "timing_available": True, "finished_ns": 300, "elapsed_ns": 200, "wall_seconds": 200/1e9,
            "cache": cache, "row_counts": {"train": 6, "dev": 6}}
    put(run/"completed.json", done)
    source_pins = {k: sources[k] for k in report.SOURCES}
    qual = tmp_path/"qualification.json"
    put(qual, {"status": "completed", "actual_exit_code": 0, "source_sha256": source_pins, "model_calls": 0, "real_task_inputs_read": False})
    plan = {"version": report.VERSION, "limits": report.LIMITS, "sources": source_pins, "runtime": {"artificial": True},
            "synthetic_qualification": {"path": qual.name, "sha256": report.sha(qual)},
            "pilot_plan": {"path": str(pilot_path.relative_to(tmp_path)), "sha256": pilot_pin},
            "run": {"path": "run", "completed_sha256": report.sha(run/"completed.json"),
                    "terminal": {"path": terminal_path.name, "sha256": report.sha(terminal_path)}}, "out": report.BASE+"/report-01"}
    plan_path = tmp_path/report.BASE/"report-plan-01.json"
    put(plan_path, plan)
    args = SimpleNamespace(plan=plan_path, plan_sha256=report.sha(plan_path), out=tmp_path/plan["out"])
    calls = []
    def authenticate(path, digest, out):
        calls.append("upstream-read-only")
        assert path == pilot_path and digest == pilot_pin and out == run
        return copy.deepcopy(pilot), {}
    monkeypatch.setattr(report, "upstream_authenticate", authenticate)
    return SimpleNamespace(args=args, plan=plan, pilot=pilot, done=done, terminal=terminal, launch=launch,
                           run=run, root=tmp_path, terminal_path=terminal_path, calls=calls)


def test_complete_synthetic_report_all_fits_raw_scope_and_eighteen_gate(artifacts):
    f = artifacts
    receipt = report.execute(f.args)
    summary = report.read(f.args.out/"summary.json")
    assert receipt["status"] == "completed" and receipt["complete_fits"] == 8
    assert not receipt["temperature_fitted"] and receipt["model_calls"] == 0
    assert summary["continuation"]["checks_total"] == 18 and not summary["continuation"]["passed"]
    assert set(summary["fits"]) == set(report.FIT_ORDER)
    assert set(summary["families"]) == set(report.ARMS)
    assert f.calls == ["upstream-read-only", "upstream-read-only"]
    assert "unscored turns may intervene" in (f.args.out/"report.md").read_text()
    report.manifest(f.args.out, receipt["files"], "receipt.json")


@pytest.mark.parametrize("defect", ["source", "plan", "missing_prediction", "extra_payload", "symlink", "failed_terminal",
                                  "late_terminal", "command_mismatch", "incomplete_fit", "unqualified"])
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
    elif defect == "symlink":
        (f.run/"extra.txt").symlink_to(f.args.plan)
    elif defect in ("failed_terminal", "late_terminal", "command_mismatch"):
        terminal = copy.deepcopy(f.terminal)
        if defect == "failed_terminal":
            terminal.update(status="failed", returncode=1)
        elif defect == "late_terminal":
            terminal.update(finished_ns=terminal["deadline_ns"], elapsed_ns=1800_000_000_000, wall_seconds=1800.)
        else:
            terminal["command"][-1] = "wrong-output"
        f.terminal_path.write_text(json.dumps(terminal))
        f.plan["run"]["terminal"]["sha256"] = report.sha(f.terminal_path)
        f.args.plan.write_text(json.dumps(f.plan))
        f.args.plan_sha256 = report.sha(f.args.plan)
    elif defect == "incomplete_fit":
        done = copy.deepcopy(f.done)
        done["fits"] = done["fits"][:-1]
        (f.run/"completed.json").write_text(json.dumps(done))
        f.plan["run"]["completed_sha256"] = report.sha(f.run/"completed.json")
        f.args.plan.write_text(json.dumps(f.plan))
        f.args.plan_sha256 = report.sha(f.args.plan)
    else:
        (f.root/"qualification.json").write_text("changed qualification")
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Decoded predictions before authentication"))
    with pytest.raises((ValueError, FileNotFoundError)):
        report.execute(f.args)
    assert not (f.args.out/"summary.json").exists()
    assert not report.read(f.args.out/"failed.json")["scientific_result_qualified"]


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
    assert (f.args.out/"invalid-receipt.json").is_file()
    assert not (f.args.out/"receipt.json").exists()


def test_failure_receipt_error_does_not_mask_primary(artifacts, monkeypatch):
    f = artifacts
    f.args.plan_sha256 = "0"*64
    original = report.write
    def broken(path, value):
        if path.name == "failed.json":
            raise OSError("synthetic disk failure")
        return original(path, value)
    monkeypatch.setattr(report, "write", broken)
    with pytest.raises(ValueError, match="External report plan pin") as captured:
        report.execute(f.args)
    assert any("synthetic disk failure" in note for note in captured.value.__notes__)


def test_manifest_traversal_rejected_before_member_read(tmp_path):
    with pytest.raises(ValueError, match="Canonical"):
        report.manifest(tmp_path, {"../escape": {"sha256": "a"*64, "bytes": 1}})


def test_cli_has_no_model_or_training_overrides():
    args = report.parse_args(["--plan", "p", "--plan-sha256", "a"*64, "--out", "o"])
    assert set(vars(args)) == {"plan", "plan_sha256", "out"}
