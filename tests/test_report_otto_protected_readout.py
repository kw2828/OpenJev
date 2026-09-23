"""Fabricated provenance, complete reporting and rendering, without science calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_protected_readout_report", ROOT / "scripts/report_otto_protected_readout.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def rules():
    rows = [{"name": "common.technical_complete", "value": False, "relation": "==", "threshold": True, "passes": False}]
    for regime in report.REGIMES:
        for scope in ("initial", "age1", "age2", "age3"):
            rows.append({"name": f"common.{regime}.{scope}.case_support", "value": 6,
                         "relation": ">=", "threshold": 4, "passes": True})
        rows.append({"name": f"common.{regime}.hold.postcorrection.positive_gap", "value": 3.,
                     "relation": ">", "threshold": 0., "passes": True})
    for regime in report.REGIMES:
        for control in ("pretrained", "frozen_aux", "joint_aux", "joint_spo"):
            rows.append({"name": f"candidate.{regime}.postcorrection.gap_vs_{control}", "value": 2.,
                         "relation": "<= and <", "threshold": 1.8, "strict_upper_bound": 2., "passes": False})
        for scope in ("full", "initial"):
            rows.append({"name": f"candidate.{regime}.{scope}.agreement", "value": .5,
                         "relation": ">=", "threshold": .6, "passes": False})
        for seed in (301000001, 301000002, 301000003):
            rows.append({"name": f"candidate.{regime}.{seed}.postcorrection.gap_vs_frozen_aux", "value": 2.,
                         "relation": "<=", "threshold": 1., "passes": False})
    return rows


def metrics(family, seed):
    delta = 0. if seed is None else (seed - 301000001) * .25
    gap = {"pretrained": 3., "frozen_aux": 2., "frozen_spo": 1., "joint_aux": .5, "joint_spo": 1.5, "hold": 4.}[family]
    result = {"primary_mask": "nonquery and absolute_step >= 5"}
    for scope in report.SCOPES:
        result[scope] = {"by_regime": {regime: {"episodes": 18, "supported_episodes": 15,
            "declared_case_count": 6, "supported_case_count": 5, "weight_mass": 15., "nonquery_rows": 60,
            report.GAP: gap + delta + (regime == "lambda4"), report.AGREEMENT: .5,
            "episode_weighted_centered_mse": 2., "episode_weighted_first_argmin_match": .5}
            for regime in report.REGIMES}}
    return result


def aggregates():
    models, fits = [], []
    for seed in report.SEEDS:
        for family in report.FAMILIES:
            pretrained = family == "pretrained"
            models.append({"family": family, "seed": seed, "metrics": metrics(family, seed)})
            fits.append({"family": family, "seed": seed, "training_mode": "pretrain" if pretrained else family.split("_")[0],
                "stage": "pretrain" if pretrained else "adaptation", "readout": "shared" if pretrained else "protected",
                "objective": "spo" if family.endswith("spo") else "aux", "parameter_count": 5996 if pretrained else 6112,
                "trainable_parameter_count": 5996 if pretrained else 116 if family.startswith("frozen") else 6112,
                "epochs": 80 if pretrained else 40, "steps": 720 if pretrained else 360,
                "wall_seconds": 12. if pretrained else 7., "fit_seconds": 10. if pretrained else 5.,
                "evaluation_clone_seconds": .25, "train_rescore_seconds": .5, "checkpoint_seconds": .25,
                "forward_chunks": 12, "backward_chunks": 8, "no_grad_chunks": 4,
                "differentiable_chunks": 8, "skipped_backward_chunks": 0})
    required = rules()
    producer = {"version": report.TRAIN_VERSION, "models": models, "hold": metrics("hold", None),
        "required": required, "required_conditions": 29, "required_passed": 10, "gates": report.gate_records(required),
        "technical_complete_pending_saved_audit": True, "scientific_conditions": 28, "scientific_passed": 10,
        "train_counts": {"episodes": 54}, "validation_counts": {"episodes": 36},
        "setup_seconds": 1., "fitting_seconds": 125., "validation_seconds": 3.,
        "scope": "Fabricated fixed-path aggregates only.", "capacity_evidence_scope": "Fabricated backbone evidence only."}
    final = copy.deepcopy(producer)
    final["required"][0].update(value=True, passes=True)
    final.update(required_passed=11, gates=report.gate_records(final["required"]), technical_complete_pending_saved_audit=False)
    audit = {"version": report.AUDIT_VERSION, "agreement": True, "producer_summary": copy.deepcopy(producer),
        "summary": final, "fits": fits, "technical_condition_requires_successful_original_audit_supervisor": True,
        "counts": {"fits": 15, "optimizer_steps": 6480, "training_events": 67, "work_events": 12960,
            "prediction_files": 31, "training_payloads": 58, "collection_episodes": 90, "collection_payloads": 18,
            "required_conditions": 29, "required_passed": 11}, "limitations": ["Fabricated saved aggregates only."]}
    return producer, audit


def test_all_seeds_controls_scopes_and_disjoint_costs_are_preserved():
    summary, audit = aggregates()
    before = copy.deepcopy((summary, audit))
    data = report.tables(summary, audit)
    assert {key: len(value) for key, value in data.items()} == {"regime_metrics": 96, "family_means": 30,
        "paired_changes": 72, "paired_means": 24, "fit_costs": 15}
    primary = [r for r in data["paired_changes"] if r["scope"] == "postcorrection"]
    assert len(primary) == 24
    assert {(r["control"], r["seed"], r["regime"]) for r in primary} == {
        (family, seed, regime) for family in report.CONTROLS for seed in report.SEEDS for regime in report.REGIMES}
    for row in primary:
        assert row["delta_" + report.GAP] == {"pretrained": -2., "frozen_aux": -1., "joint_aux": .5, "joint_spo": -.5}[row["control"]]
    costs = report.cost_summary(summary, data)
    assert costs["shared_pretraining_seconds"] == 36.
    assert costs["adaptation_seconds"] == 84.
    assert costs["fit_wall_seconds"] == 120.
    assert costs["fitting_stage_other_seconds"] == 5.
    assert costs["per_fit_validation_seconds"] is None
    assert (summary, audit) == before


@pytest.mark.parametrize("defect", ("missing_model", "duplicate_fit", "nonfinite", "support", "cost", "epochs"))
def test_incomplete_or_incomparable_display_is_rejected(defect):
    summary, audit = aggregates()
    if defect == "missing_model":
        summary["models"].pop()
    elif defect == "duplicate_fit":
        audit["fits"][-1] = audit["fits"][0]
    elif defect == "nonfinite":
        summary["models"][0]["metrics"]["full"]["by_regime"]["lambda3"][report.GAP] = float("nan")
    elif defect == "support":
        summary["models"][0]["metrics"]["full"]["by_regime"]["lambda3"]["weight_mass"] = 1.
    elif defect == "cost":
        audit["fits"][0]["wall_seconds"] = 1.
    else:
        audit["fits"][1]["epochs"] = 80
    with pytest.raises(ValueError):
        report.tables(summary, audit)


def test_failed_gate_is_retained_and_all_conditions_are_required():
    rows = rules()
    gate = report.gate_records(rows)["protected_readout"]
    assert gate["total"] == 29 and gate["passed"] == 10 and gate["passes"] is False
    for row in rows:
        row["passes"] = True
    rows[-1]["passes"] = False
    assert report.gate_records(rows)["protected_readout"]["passes"] is False
    rows[-1]["name"] = rows[-2]["name"]
    with pytest.raises(ValueError, match="ordered 29"):
        report.gate_records(rows)


def descriptor(path):
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return descriptor(path)


def phase(root, directory, script, cap, expected, *, mode=None):
    launch_path = root / (directory.name + ".launch.json")
    command = [str(root / ".venv/bin/python"), "-u", str(root / script)] + ([] if mode is None else [mode])
    command += [item for pair in {**expected, "--supervision": str(launch_path)}.items() for item in pair]
    launch = {"command": command, "cwd": str(root), "cap_seconds": cap, "started_ns": 10**9,
        "deadline_ns": (cap + 1) * 10**9, "clock_source_sha256": report.base.base.CLOCK_PIN,
        "watchdog_sha256": report.base.base.SUPERVISOR_PIN, "pid": 10, "pgid": 10, "parent_pid": 9}
    launch_pin = save(launch_path, launch)
    finish = 132 if mode else 3
    worker = {"started_ns": 2 * 10**9, "finished_ns": finish * 10**9, "wall_seconds": float(finish - 2),
              "supervision_sha256": launch_pin["sha256"]}
    terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False, "error": None,
        "clock_error": None, "group_absent": True, "cleanup": {"reaped": True, "errors": []},
        "finished_ns": (finish + 1) * 10**9, "elapsed_ns": finish * 10**9, "wall_seconds": float(finish)}
    return worker, terminal, launch


def closed_fixture(tmp_path, monkeypatch, defect=None):
    for module in (report, report.base, report.base.base):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    producer, result = aggregates()
    train, audit_dir, output = (tmp_path / name for name in ("train", "audit", "report"))
    for directory in (train, audit_dir, output):
        directory.mkdir()
    sources = {}
    for name in report.SCIENCE_PINS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fabricated pinned source\n")
        sources[name] = descriptor(path)["sha256"]
    monkeypatch.setattr(report, "SCIENCE_PINS", sources.copy())
    inputs = {role: save(tmp_path / (role + ".json"), {"role": role}) for role in
              ("collection_plan", "collection_receipt", "collection_terminal", "engineering",
               "capacity_plan", "capacity_receipt", "capacity_terminal")}
    plan_path = tmp_path / "plan.json"
    plan_pin = save(plan_path, {"version": report.TRAIN_VERSION, "status": "frozen_before_fitting",
                               "sources": sources, "inputs": inputs})
    worker, terminal, launch = phase(tmp_path, train, report.TRAINER, 14400,
        {"--plan": str(plan_path), "--plan-sha256": plan_pin["sha256"], "--output": str(train)}, mode="run")
    if defect == "training_timeout":
        terminal["timed_out"] = True
    elif defect == "training_wrong_command":
        terminal["command"] = terminal["command"][:-2] + ["--wrong", "value"]
    for name in report.TRAIN_PAYLOADS:
        (train / name).write_bytes(b"Opaque fabricated bytes, deliberately invalid NPZ and JSON.\n")
    save(train / "summary.json", producer)
    save(train / "fits.json", {"fits": result["fits"]})
    save(train / "started.json", {"started_ns": worker["started_ns"], "launch": launch})
    worker.update(version=report.TRAIN_VERSION, status="completed", complete=True, fits_completed=15,
        optimizer_steps=6480, pending=None, pending_emission=None, requires_successful_original_supervisor=True,
        plan_sha256=plan_pin["sha256"], sources=sources, inputs=inputs,
        files={name: descriptor(train / name) for name in report.TRAIN_PAYLOADS})
    roles = {"plan": plan_pin, "worker": save(train / "receipt.json", worker),
             "terminal": save(tmp_path / "train.terminal.json", terminal)}
    expected = {"--output": str(audit_dir)}
    for role, pin in roles.items():
        expected.update({"--" + role: pin["path"], "--" + role + "-sha256": pin["sha256"]})
    receipt, terminal, launch = phase(tmp_path, audit_dir, report.AUDITOR, 300, expected)
    if defect == "audit_unreaped":
        terminal["cleanup"]["reaped"] = False
    elif defect == "audit_wrong_cap":
        terminal["cap_seconds"] = 240
    elif defect == "audit_nonzero":
        terminal["returncode"] = 1
    elif defect == "audit_launch":
        launch = {**launch, "parent_pid": -1}
    save(audit_dir / "started.json", {"started_ns": receipt["started_ns"], "launch": launch, "producer_inputs": roles})
    if defect == "changed_gate":
        result["summary"]["gates"]["protected_readout"]["passes"] = True
    save(audit_dir / "audit.json", result)
    receipt.update(version=report.AUDIT_VERSION, status="completed", agreement=True, failures=[],
        requires_successful_original_supervisor=True, native_calls=0, model_calls=0, optimizer_calls=0, teacher_calls=0,
        producer_inputs=roles, plan_sha256=plan_pin["sha256"], sources={report.AUDITOR: sources[report.AUDITOR]},
        files={name: descriptor(audit_dir / name) for name in ("started.json", "audit.json")})
    receipt_pin = save(audit_dir / "receipt.json", receipt)
    terminal_path = tmp_path / "audit.terminal.json"
    terminal_pin = save(terminal_path, terminal)
    if defect == "source":
        (tmp_path / report.TRAINER).write_bytes(b"changed source")
    elif defect == "payload":
        (train / "prediction-frozen_spo-301000001.npz").write_bytes(b"changed opaque payload")
    elif defect == "extra_file":
        (train / "unexpected.json").write_text("{}")
    elif defect == "input":
        Path(inputs["collection_terminal"]["path"]).write_text("{}")
    args = SimpleNamespace(output=output, audit_directory=audit_dir,
        audit_receipt_sha256=receipt_pin["sha256"], audit_terminal=terminal_path,
        audit_terminal_sha256=terminal_pin["sha256"])
    return report.Report(args), result


def test_both_original_parents_and_every_opaque_payload_close_before_display(tmp_path, monkeypatch):
    runner, result = closed_fixture(tmp_path, monkeypatch)
    summary, audit = runner.authenticate()
    assert summary == result["summary"] and audit == result
    assert summary["gates"]["protected_readout"]["passes"] is False
    assert sum(path.endswith(".npz") for path in runner.bound) == 49
    assert runner.receipt["array_decodes"] == runner.receipt["model_calls"] == 0
    assert runner.receipt["original_seconds"] == {"training_worker": 130., "training_parent": 132.,
                                                 "audit_worker": 1., "audit_parent": 3.}


@pytest.mark.parametrize("defect", ("training_timeout", "training_wrong_command", "audit_unreaped", "audit_wrong_cap",
    "audit_nonzero", "audit_launch", "source", "payload", "extra_file", "input"))
def test_invalid_original_closure_is_rejected_before_outcome_decode(tmp_path, monkeypatch, defect):
    runner, _ = closed_fixture(tmp_path, monkeypatch, defect)
    original_read = runner.read

    def metadata_only(path):
        assert path.name not in ("summary.json", "audit.json", "fits.json"), "Outcome decoded before complete original closure"
        return original_read(path)

    monkeypatch.setattr(runner, "read", metadata_only)
    with pytest.raises(ValueError):
        runner.authenticate()


def test_report_cannot_promote_failed_scientific_gate(tmp_path, monkeypatch):
    runner, _ = closed_fixture(tmp_path, monkeypatch, "changed_gate")
    with pytest.raises(ValueError):
        runner.authenticate()


def test_three_figures_render_all_seeds_and_clear_cost_scope(tmp_path, monkeypatch):
    _, audit = aggregates()
    summary = audit["summary"]
    runner = report.Report(SimpleNamespace(output=tmp_path))
    captured = {}

    def saved(fig, stem):
        text = [item.get_text() for item in fig.texts]
        text += [ax.yaxis.label.get_text() for ax in fig.axes]
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        captured[stem] = (text, buffer.getvalue())

    monkeypatch.setattr(runner, "save_figure", saved)
    runner.figures(summary, report.tables(summary, audit))
    assert set(captured) == set(report.FIGURES)
    assert all(content.startswith(b"<?xml") for _, content in captured.values())
    assert any("FAIL (11/29)" in text for text in captured[report.FIGURES[0]][0])
    assert any("counted once per seed" in text for text in captured[report.FIGURES[2]][0])
    assert any("No per-fit VALID timer exists" in text for text in captured[report.FIGURES[2]][0])
    assert not any("regret" in text.lower() for texts, _ in captured.values() for text in texts)


def test_render_failure_leaves_failed_receipt_and_no_success_claim(tmp_path, monkeypatch):
    runner, audit = closed_fixture(tmp_path, monkeypatch)
    runner.out.rmdir()
    monkeypatch.setattr(runner, "bind", lambda *_: None)
    monkeypatch.setattr(runner, "authenticate", lambda: (audit["summary"], audit))
    runner.receipt["original_seconds"] = {"training_worker": 130.}

    def fail(*_):
        raise RuntimeError("fabricated rendering failure")

    monkeypatch.setattr(runner, "figures", fail)
    # The report source descriptors here are fabricated; never inspect real run files.
    monkeypatch.setattr(runner, "descriptor", lambda path: {"path": str(path), "sha256": "0" * 64, "bytes": 0})
    with pytest.raises(RuntimeError, match="fabricated rendering failure"):
        runner.execute()
    receipt = json.loads((runner.out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and "fabricated rendering failure" in receipt["error"]
    assert all(not (runner.out / f"{stem}.png").exists() for stem in report.FIGURES)
