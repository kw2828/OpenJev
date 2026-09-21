"""Synthetic saved-only report integration; no checkpoints, models or task data.

The already qualified common reader is replaced at its component boundary.
Actual per-fit/phase byte manifests and the new reader's joins remain active.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openjev.research import dialogue_observation_metrics as old
from openjev.research.suspend_clock import Deadline

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("calibration_report_tested", REPO / "scripts/report_dialogue_calibration.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Clock:
    backend = "CLOCK_BOOTTIME"

    def __init__(self):
        self.value = 0

    def now_ns(self):
        self.value += 1_000_000
        return self.value

    def deadline_after(self, seconds):
        start = self.now_ns()
        return Deadline(self, start, start + seconds * 1_000_000_000)


def row(index, *, calibration=False):
    category = index % 3
    label = 0 if category == 0 else 2
    bin_name = ("unmentioned_retention", "assigned_retention", "first_assignment")[category]
    unseen = index >= 3 and not calibration
    service = "unseen" if unseen else "seen"
    ids = [old.NONE, old.DONTCARE, "value:red"]
    return {"row_index": index, "split": "train" if calibration else "dev",
            "source_split": "train" if calibration else "dev",
            "analysis_role": "calibration" if calibration else "evaluation",
            "dialogue_id": ("cal" if calibration else "dev") + str(index), "source_row_index": 0,
            "time": 0, "turn_index": 0, "query_index": int(unseen), "query_position": 0,
            "query_id": json.dumps([service, "slot"]), "service": service, "slot": "slot",
            "candidate_ids": ids, "candidate_values": [None, None, "red"], "label_index": label,
            "label_id": ids[label], "bin": bin_name,
            "stratum": bin_name if category < 2 else "changed", "dontcare": False, "unseen": unseen}


def saved(probabilities):
    logs = np.full((len(probabilities), 12), -np.inf, np.float32)
    logs[:, :3] = np.log(np.asarray(probabilities, np.float64)).astype(np.float32)
    return logs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    report.common.write(path, value)


def packet(path, logs):
    with path.open("xb") as stream:
        np.savez(stream, log_probs=logs, row_indices=np.arange(len(logs), dtype=np.int64))


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "DEV_ROWS", 6)
    monkeypatch.setattr(report, "CALIBRATION_ROWS", 4)
    monkeypatch.setattr(report, "CALIBRATION_DIALOGUES", 4)
    monkeypatch.setattr(report.common, "environment", lambda: {"runtime": "artificial"})
    monkeypatch.setattr(report.common, "peak_rss", lambda: 1024)
    monkeypatch.setattr(report, "SuspendClock", Clock)
    monkeypatch.setattr(report.signal, "signal", lambda *_: 1)
    monkeypatch.setattr(report.signal, "setitimer", lambda *_: None)
    sources = {}
    for name in report.SOURCES | {"scripts/dialogue_calibration_common.py", "src/openjev/research/suspend_clock.py"}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source descriptor: " + name)
        sources[name] = report.common.sha(path)
    new_sources = {name: sources[name] for name in report.SOURCES}
    control = {"sources": {"scripts/dialogue_calibration_common.py": sources["scripts/dialogue_calibration_common.py"]},
               "inputs": {"src/openjev/research/suspend_clock.py": sources["src/openjev/research/suspend_clock.py"]},
               "outputs": {phase: f"synthetic/{phase}" for phase in report.PHASES}}
    control_path = tmp_path / "control-plan.json"
    write_json(control_path, control)
    write_json(tmp_path / "synthetic-qualified.json", {"status": "completed", "actual_exit_code": 0,
               "source_sha256": new_sources, "model_calls": 0, "real_task_inputs_read": False})
    dev = [row(i) for i in range(6)]
    calibration = [row(i, calibration=True) for i in range(4)]
    for index, item in enumerate(calibration):
        item["label_index"] = 0 if index < 3 else 2
        item["label_id"] = item["candidate_ids"][item["label_index"]]
        item["bin"] = item["stratum"] = "unmentioned_retention" if index < 3 else "changed"
        if index == 3:
            item["bin"] = "first_assignment"
    base = saved([[.9, .05, .05], [.05, .05, .9], [.05, .05, .9],
                  [.9, .05, .05], [.4, .2, .4], [.9, .05, .05]])
    treatment = saved([[.9, .05, .05], [.05, .05, .9], [.05, .05, .9],
                       [.9, .05, .05], [.999, .0005, .0005], [.05, .05, .9]])
    cal_logs = saved([[.8, .1, .1]] * 4)
    oldrun, oldreport = tmp_path / "old-run", tmp_path / "old-report"
    oldrun.mkdir()
    oldreport.mkdir()
    report.common.write_lines(oldrun / "evaluation-rows.jsonl", dev)
    fits = {}
    for fit_id in report.FIT_ORDER:
        arm, seed = fit_id.rsplit("-", 1)
        directory = oldrun / fit_id
        directory.mkdir()
        logs = treatment if arm == "trainable_numbers" else base
        packet(directory / "predictions.npz", logs)
        write_json(directory / "completed.json", {"final_encoder_sha256": "e"*64, "checkpoint": {"sha256": "a"*64}})
        fits[fit_id] = old.score_panels(logs, dev)
    published = {"technical_validity_passed": True, "technical_complete_fits": 12, "fits": fits,
                 "continuation": old.criteria(fits), "costs": {"scope": "Synthetic historical training only"}}
    assert published["continuation"]["passed"] is False and published["continuation"]["checks_passed"] == 6
    write_json(oldreport / "summary.json", published)
    phases, receipts, terminal_paths = {}, {}, {}
    ctx = {"plan_path": control_path.resolve(), "plan": control,
           "science": {"fit_order": report.FIT_ORDER},
           "prior": SimpleNamespace(run=oldrun, report=oldreport,
                                     report_summary_sha256=report.common.sha(oldreport / "summary.json"))}
    for phase in report.PHASES:
        directory = tmp_path / control["outputs"][phase]
        directory.mkdir(parents=True)
        write_json(directory / "started.json", {"synthetic": True})
        done = {"status": "completed", "wall_seconds": 2., "peak_rss_bytes": 1024,
                "sampled_mps_driver_max_bytes": 0}
        if phase == "prepare":
            report.common.write_lines(directory / "evaluation-rows.jsonl", calibration)
            write_json(directory / "selection.json", {"selected_ids": [r["dialogue_id"] for r in calibration]})
            done.update(counts={"dialogues": 4, "scored_endpoints": 4}, source_split="train", analysis_role="calibration",
                        model_calls=0, neural_calls=0)
        else:
            done.update(prepared_sha256=phases["prepare"]["completed_sha256"],
                        prepared_terminal_sha256=phases["prepare"]["terminal"]["sha256"],
                        fit_order=report.FIT_ORDER, fits=[], model_weight_updates=0, optimizer_created=False,
                        temperature_applied=False, official_test_opened=False, task_metrics_computed=False)
            count = 2 if phase == "qualify" else 4
            for fit_id in report.FIT_ORDER:
                arm, seed = fit_id.rsplit("-", 1)
                target = directory / fit_id
                target.mkdir()
                counts = {op + "_" + suffix: count for op in ("public_forward", "encoding", "memory_forward")
                          for suffix in ("attempts", "returns")}
                state = {"encoder": "e"*64, "memory": "f"*64}
                fit = {"status": "completed", "fit_id": fit_id, "arm": arm, "seed": int(seed),
                       "witness": {"counts": counts, "synthetic_injection": False, "optimizer_created": False,
                                   "temperature_applied": False, "restored_sha256": state, "final_sha256": state,
                                   "checkpoint_sha256": "a"*64}}
                if phase == "infer":
                    packet(target / "predictions.npz", cal_logs)
                    fit.update(dialogues=4, endpoint_rows=4)
                else:
                    fit["cases"] = [{"parity": {"canonical_first_argmax_equal": True,
                                               "maximum_supported_log_error": 0., "maximum_probability_error": 0.}}]*2
                fit["files"] = report.common.members(target)
                write_json(target / "completed.json", fit)
                done["fits"].append({**fit, "completed_sha256": report.common.sha(target / "completed.json")})
            if phase == "qualify":
                done.update(parity_passed=True, completed_forwards=24,
                            projection={"admitted": True, "threshold_seconds": 1800, "total_seconds": 60., "preparation_seconds": 3.})
                write_json(directory / "projection.json", done["projection"])
            else:
                done.update(qualification_sha256=phases["qualify"]["completed_sha256"],
                            qualification_terminal_sha256=phases["qualify"]["terminal"]["sha256"],
                            completed_forwards=48, endpoint_rows_per_fit=4, total_endpoint_rows=48)
        done["files"] = report.common.members(directory)
        write_json(directory / "completed.json", done)
        terminal_path = tmp_path / f"{phase}-terminal.json"
        write_json(terminal_path, {"status": "completed", "returncode": 0, "wall_seconds": 3.})
        phases[phase] = {"path": control["outputs"][phase], "completed_sha256": report.common.sha(directory / "completed.json"),
                         "terminal": {"path": terminal_path.name, "sha256": report.common.sha(terminal_path)}}
        receipts[phase] = done
        terminal_paths[phase] = terminal_path
    plan = {"version": report.VERSION, "limits": report.LIMITS, "sources": new_sources,
            "control_plan": {"path": "control-plan.json", "sha256": report.common.sha(control_path)},
            "phases": phases, "environment": {"runtime": "artificial"},
            "synthetic_qualification": {"path": "synthetic-qualified.json", "sha256": report.common.sha(tmp_path / "synthetic-qualified.json")},
            "out": report.OUTPUT_PATH}
    plan_path = tmp_path / report.PLAN_PATH
    write_json(plan_path, plan)
    args = SimpleNamespace(plan=plan_path, plan_sha256=report.common.sha(plan_path), out=tmp_path / report.OUTPUT_PATH)
    events = []

    def common_auth(request, budget):
        events.append("upstream-read-only")
        assert request.command == "infer" and request.out == tmp_path / control["outputs"]["infer"]
        assert request.plan.resolve() == control_path.resolve()
        assert request.plan_sha256 == plan["control_plan"]["sha256"]
        budget.check()
        return ctx

    def common_output(path, digest, context, phase):
        events.append("payloads-" + phase)
        assert context is ctx
        report.require(report.common.sha(path / "completed.json") == digest, "Phase completion changed")
        done = report.common.read(path / "completed.json")
        report.require(done["status"] == "completed", "Complete phase required")
        report.common.manifest(path, done["files"])
        return done

    def common_terminal(path, digest, receipt):
        events.append("terminal")
        report.require(report.common.sha(path) == digest, "External parent terminal changed")
        value = report.common.read(path)
        report.require(value["status"] == "completed" and value["returncode"] == 0, "Successful parent required")
        return value

    monkeypatch.setattr(report.common, "authenticate", common_auth)
    monkeypatch.setattr(report.common, "authenticate_output", common_output)
    monkeypatch.setattr(report.common, "authenticate_terminal", common_terminal)
    monkeypatch.setattr(report.common, "execute", lambda *_: pytest.fail("Neural lifecycle must never execute"))
    return SimpleNamespace(root=tmp_path, args=args, plan=plan, ctx=ctx, receipts=receipts,
                           events=events, published=published, terminals=terminal_paths)


def run_auth(f):
    f.args.out.mkdir(parents=True)
    return report.authenticate(f.args, report.Budget(f.args.out, Clock()))


def test_complete_artificial_report_preserves_raw_failure_and_full_fit_files(artifacts):
    f = artifacts
    receipt = report.execute(f.args)
    summary = report.common.read(f.args.out / "summary.json")
    assert receipt["status"] == "completed" and receipt["model_calls"] == 0
    assert receipt["original_raw_failure_preserved"] is True and receipt["checks_total"] == 11
    assert receipt["wall_seconds"] < 300 and receipt["timing_available"] is True
    assert receipt["new_control_total_seconds"] == 9 + receipt["wall_seconds"]
    assert len(receipt["files"]) == 15 and set(summary["fits"]) == set(report.FIT_ORDER)
    assert "panels" not in summary["fits"][report.FIT_ORDER[0]]
    for name, entry in summary["fits"].items():
        fit_path = f.args.out / entry["path"]
        assert report.descriptor(fit_path) == {k: entry[k] for k in ("bytes", "sha256")}
        value = report.common.read(fit_path)
        assert set(value) == {"raw", "normalized", "calibrated", "temperature", "calibration_validation"}
        assert value["raw"] == f.published["fits"][name]
        assert value["temperature"]["selection_data"] == "calibration_only"
        assert value["calibration_validation"]["rows"] == 4
        assert value["calibrated"]["validation"]["top_tie_masks_unchanged"]
    assert summary["continuation"]["original_conditions"]["raw"] == f.published["continuation"]
    assert "FAIL (6/7)" in (f.args.out / "report.md").read_text()
    assert f.events.count("upstream-read-only") == 2 and f.events.count("terminal") == 6
    report.common.manifest(f.args.out, receipt["files"], "receipt.json")


@pytest.mark.parametrize("defect", ["missing_infer", "failed_parent", "mutated_prediction", "source_change", "unqualified_source"])
def test_auth_failure_never_decodes_rows_or_predictions(artifacts, monkeypatch, defect):
    f = artifacts
    infer = f.root / f.plan["phases"]["infer"]["path"]
    if defect == "missing_infer":
        (infer / "completed.json").unlink()
    elif defect == "failed_parent":
        f.terminals["infer"].write_text(json.dumps({"status": "failed", "returncode": 1}))
    elif defect == "mutated_prediction":
        with (infer / report.FIT_ORDER[-1] / "predictions.npz").open("ab") as stream:
            stream.write(b"corrupt")
    elif defect == "source_change":
        (f.root / "scripts/report_dialogue_calibration.py").write_text("changed")
    else:
        path = f.root / "synthetic-qualified.json"
        value = report.common.read(path)
        value["actual_exit_code"] = 1
        path.write_text(json.dumps(value))
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Prediction decoded before authentication"))
    monkeypatch.setattr(report, "read_lines", lambda *_: pytest.fail("Evaluator decoded before authentication"))
    with pytest.raises((ValueError, FileNotFoundError)):
        report.execute(f.args)
    failed = report.common.read(f.args.out / "failed.json")
    assert failed["status"] == "failed" and failed["model_calls"] == 0 and failed["scientific_result_qualified"] is False
    assert failed["wall_seconds"] is None and not (f.args.out / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        report.execute(f.args)


def test_cost_denial_cannot_be_used_to_fit_temperatures(artifacts, monkeypatch):
    f = artifacts
    original = report.common.authenticate_output

    def denied(path, digest, ctx, phase):
        value = copy.deepcopy(original(path, digest, ctx, phase))
        if phase == "qualify":
            value["projection"].update(admitted=False, total_seconds=5081.543)
        return value

    monkeypatch.setattr(report.common, "authenticate_output", denied)
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Denied qualification decoded predictions"))
    monkeypatch.setattr(report, "read_lines", lambda *_: pytest.fail("Denied qualification decoded evaluator rows"))
    with pytest.raises(ValueError, match="cost admission"):
        report.execute(f.args)
    assert report.common.read(f.args.out / "failed.json")["progress"]["completed_fits"] == []


def test_incomplete_or_misbound_fit_is_rejected_after_byte_auth(artifacts, monkeypatch):
    f = artifacts
    original = report.common.authenticate_output

    def incomplete(path, digest, ctx, phase):
        value = copy.deepcopy(original(path, digest, ctx, phase))
        if phase == "infer":
            value["fits"] = value["fits"][:-1]
        return value

    monkeypatch.setattr(report.common, "authenticate_output", incomplete)
    monkeypatch.setattr(report.np, "load", lambda *_a, **_k: pytest.fail("Partial fit metrics decoded"))
    with pytest.raises(ValueError, match="all twelve"):
        run_auth(f)


def test_raw_published_disagreement_preserves_partial_evidence(artifacts, monkeypatch):
    f = artifacts
    authenticate = report.authenticate

    def wrong(args, budget):
        plan, ctx, receipts, terminals, published = authenticate(args, budget)
        published = copy.deepcopy(published)
        published["fits"][report.FIT_ORDER[1]]["panels"]["all"]["micro"]["nll"] += .001
        return plan, ctx, receipts, terminals, published

    monkeypatch.setattr(report, "authenticate", wrong)
    with pytest.raises(ValueError, match="Recomputed raw fit differs"):
        report.execute(f.args)
    assert (f.args.out / "fits" / (report.FIT_ORDER[0] + ".json")).exists()
    assert not (f.args.out / "fits" / (report.FIT_ORDER[1] + ".json")).exists()
    assert not (f.args.out / "summary.json").exists()
    assert report.common.read(f.args.out / "failed.json")["progress"]["completed_fits"] == [report.FIT_ORDER[0]]


@pytest.mark.parametrize("defect", ["float64", "row_order", "missing_member"])
def test_saved_packet_requires_raw_float32_and_exact_atomic_rows(tmp_path, defect):
    path = tmp_path / "predictions.npz"
    logs = saved([[.2, .3, .5]] * 2)
    ids = np.arange(2, dtype=np.int64)
    if defect == "float64":
        logs = logs.astype(np.float64)
    elif defect == "row_order":
        ids = ids[::-1]
    with path.open("xb") as stream:
        np.savez(stream, log_probs=logs, **({} if defect == "missing_member" else {"row_indices": ids}))
    with pytest.raises(ValueError):
        report.load_packet(path, 2)


def test_clock_initialization_failure_is_preserved_without_retry(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("Synthetic unavailable native timer")

    monkeypatch.setattr(report, "SuspendClock", broken)
    args = SimpleNamespace(out=tmp_path / "attempt", plan=tmp_path / "plan", plan_sha256="a"*64)
    with pytest.raises(RuntimeError, match="unavailable native"):
        report.execute(args)
    failure = report.common.read(args.out / "failed.json")
    assert failure["progress"] is None and failure["last_successful_elapsed_ns"] is None
    assert failure["timing_available"] is False and failure["wall_seconds"] is None


def test_success_published_at_expiry_is_demoted(artifacts, monkeypatch):
    f = artifacts
    write = report.common.write
    clock = Clock()
    monkeypatch.setattr(report, "SuspendClock", lambda: clock)

    def late(path, value):
        write(path, value)
        if path.name == "receipt.json":
            clock.value = value["deadline_ns"]

    monkeypatch.setattr(report.common, "write", late)
    with pytest.raises(TimeoutError):
        report.execute(f.args)
    assert not (f.args.out / "receipt.json").exists()
    assert (f.args.out / "invalid-receipt.json").exists()
    assert report.common.read(f.args.out / "failed.json")["timing_available"] is False


def test_reader_cli_contains_no_execution_or_model_override_flags():
    args = report.parse_args(["--plan", "plan", "--plan-sha256", "a"*64, "--out", "out"])
    assert set(vars(args)) == {"plan", "plan_sha256", "out"}
