"""Fabricated aggregate coverage, authentic parent joins and publication lifecycle."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_action_focused_plot_fixture", ROOT / "scripts/plot_otto_action_focused.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def rules():
    names = ["common.technical_complete"]
    for regime in ("lambda3", "lambda4"):
        names.append(f"common.{regime}.initial.case_support")
        names.extend(f"common.{regime}.age{age}.case_support" for age in (1, 2, 3))
        names.append(f"common.{regime}.hold.postcorrection.positive_gap")
    for architecture in ("innovation", "gru"):
        for regime in ("lambda3", "lambda4"):
            names.extend(f"objective.{architecture}.{regime}.{suffix}"
                         for suffix in ("postcorrection.gap", "full.agreement", "initial.agreement"))
            names.extend(f"objective.{architecture}.{regime}.{seed}.postcorrection.gap_nonregression"
                         for seed in (295000001, 295000002, 295000003))
    for regime in ("lambda3", "lambda4"):
        names.extend(f"architecture.{regime}.{suffix}"
                     for suffix in ("postcorrection.gap", "full.agreement", "initial.agreement"))
    result = [{"name": name, "value": 0., "relation": ">=", "threshold": 1., "passes": False} for name in names]
    result[0] = {"name": names[0], "value": False, "relation": "==", "threshold": True, "passes": False}
    return result


def group(episodes, cases, agreement, gap, *, support=None):
    support = episodes if support is None else support
    mass = support / episodes
    values = {"episodes": episodes, "supported_episodes": support,
        "zero_support_episode_ids": [] if support else ["fabricated-zero"],
        "nonquery_rows": support * 3, "weight_mass": mass,
        "episode_weighted_agreement": agreement if support else 0.,
        "episode_weighted_raw_gap": gap if support else 0.,
        "episode_weighted_centered_mse": gap * 2 if support else 0.,
        "episode_weighted_first_argmin_match": agreement / 2 if support else 0.,
        "supported_episode_agreement": agreement / mass if support else None,
        "supported_episode_raw_gap": gap / mass if support else None,
        "supported_episode_centered_mse": gap * 2 / mass if support else None,
        "declared_case_count": cases, "supported_case_count": cases if support else 0,
        "supported_cases": list(range(cases)) if support else [],
        "zero_support_cases": [] if support else list(range(cases))}
    return {**values, "by_age": {str(age): dict(values) for age in (1, 2, 3)}}


def metric_report(kind, seed):
    seed_index = 0 if seed is None else seed - 295000001
    is_spo = kind.endswith("_spo")
    agreement_delta = (.04 if kind.startswith("innovation") else -.02) if is_spo else 0.
    gap_delta = (-.3 if kind.startswith("innovation") else .4) if is_spo else 0.

    def variant(per_case):
        sections = {}
        for scope_index, scope in enumerate(("initial", "full", "postcorrection")):
            agreement = .5 + .02 * seed_index - .03 * scope_index + agreement_delta
            gap = 2. + .2 * seed_index + .3 * scope_index + gap_delta
            sections[scope] = {"overall": group(12 * per_case, 12, agreement, gap),
                "by_regime": {regime: group(6 * per_case, 6, agreement + .01 * i, gap + .1 * i)
                              for i, regime in enumerate(("lambda3", "lambda4"))},
                "by_case": [{"regime": regime, "case": case,
                             **group(per_case, 1, agreement, gap, support=0 if case == 0 else per_case)}
                            for regime in ("lambda3", "lambda4") for case in range(6)]}
        return {"primary_mask": "nonquery and absolute_step >= 5", **sections}

    return {**variant(3), "by_collector": {arm: variant(1) for arm in ("analytic", "neural", "period4_hold")}}


def aggregates():
    models, fits = [], []
    for seed_index, seed in enumerate((295000001, 295000002, 295000003)):
        for kind in ("innovation_aux", "innovation_spo", "gru_aux", "gru_spo"):
            architecture = "innovation" if kind.startswith("innovation") else "innovation_gru"
            objective = kind.rsplit("_", 1)[1]
            factor = 1.5 if objective == "spo" else 1.
            models.append({"family": kind, "seed": seed, "metrics": metric_report(kind, seed)})
            fits.append({"family": kind, "seed": seed, "architecture": architecture, "objective": objective,
                "readout": "shared", "parameter_count": 5978 if architecture == "innovation" else 5996,
                "epochs": 80, "steps": 720, "fit_seconds": (10. + seed_index) * factor,
                "wall_seconds": (12. + seed_index) * factor, "train_rescore_seconds": factor,
                "checkpoint_seconds": factor / 2, "forward_chunks": 1600, "backward_chunks": 800,
                "no_grad_chunks": 800, "spo_loss_calls": 800 if objective == "spo" else 0,
                "spo_weighted_rows": 100 if objective == "spo" else 0, "final_nonquery_loss": .2,
                "final_prior_loss": .3, "final_spo_loss": .4,
                "final_objective_spo_loss": .4 if objective == "spo" else 0.,
                "final_train_loss": .9 if objective == "spo" else .5})
    required = rules()
    summary = {"version": plot.TRAIN_VERSION, "models": models, "hold": metric_report("hold", None),
        "required": required, "required_passed": 0, "required_conditions": 41,
        "gates": plot.gate_records(required), "technical_complete_pending_saved_audit": True,
        "train_counts": {"episodes": 54}, "validation_counts": {"episodes": 36},
        "scope": "Fabricated forced paths only.", "setup_seconds": 1., "fitting_seconds": 200., "validation_seconds": 3.}
    final = copy.deepcopy(summary)
    final["required"][0].update(value=True, passes=True)
    final.update(required_passed=1, gates=plot.gate_records(final["required"]), technical_complete_pending_saved_audit=False)
    audit = {"version": plot.AUDIT_VERSION, "agreement": True, "producer_summary": summary, "summary": final,
        "fits": fits, "counts": {"fits": 12, "prediction_files": 25, "training_payloads": 49,
            "collection_episodes": 90, "required_conditions": 41, "required_passed": 1},
        "technical_condition_requires_successful_original_audit_supervisor": True,
        "limitations": ["Fabricated saved aggregates only."]}
    return summary, audit


def test_every_seed_scope_zero_cell_and_signed_paired_cost_is_retained():
    summary, audit = aggregates()
    before = copy.deepcopy((summary, audit))
    data = plot.tables(summary, audit)
    assert {k: len(v) for k, v in data.items()} == {"forecast_metrics": 9360, "regime_metrics": 78,
        "family_means": 24, "objective_contrasts": 36, "objective_contrast_means": 12, "costs": 12}
    zeros = [r for r in data["forecast_metrics"] if r["partition"] == "case" and r["case"] == 0]
    assert len(zeros) == 1248
    assert all(r["episodes"] > 0 and r["weight_mass"] == 0 and r["supported_episode_agreement"] is None for r in zeros)
    assert {r["seed"] for r in data["regime_metrics"]} == {None, 295000001, 295000002, 295000003}
    for row in data["objective_contrasts"]:
        assert row["delta_episode_weighted_agreement"] == pytest.approx(.04 if row["architecture"] == "innovation" else -.02)
        assert row["delta_episode_weighted_raw_gap"] == pytest.approx(-.3 if row["architecture"] == "innovation" else .4)
    mean = next(r for r in data["family_means"] if (r["family"], r["scope"], r["regime"]) == ("innovation_aux", "full", "lambda3"))
    assert mean["episode_weighted_agreement"] == pytest.approx(.49)
    assert mean["episode_weighted_raw_gap"] == pytest.approx(2.5)
    for row in data["costs"]:
        assert row["wall_seconds_ratio_to_paired_aux"] == pytest.approx(1.5 if row["objective"] == "spo" else 1.)
        assert row["wall_seconds_delta_to_paired_aux"] == pytest.approx((12 + row["seed"] - 295000001) * (.5 if row["objective"] == "spo" else 0.))
    assert (summary, audit) == before


@pytest.mark.parametrize("defect", ("missing", "nonfinite", "support"))
def test_incomplete_or_incomparable_display_rejected(defect):
    summary, audit = aggregates()
    if defect == "missing":
        audit["fits"].pop()
    elif defect == "nonfinite":
        summary["models"][0]["metrics"]["full"]["by_regime"]["lambda3"][plot.GAP] = float("nan")
    else:
        summary["models"][1]["metrics"]["full"]["by_regime"]["lambda3"]["nonquery_rows"] += 1
    with pytest.raises(ValueError):
        plot.tables(summary, audit)


def test_three_named_negative_gates_and_exact_memberships():
    rows = rules()
    gates = plot.gate_records(rows)
    assert [g["total"] for g in gates.values()] == [23, 23, 29]
    assert all(not g["passes"] and g["passed"] == 0 for g in gates.values())
    for row in rows:
        row["passes"] = True
    rows[-1]["passes"] = False
    gates = plot.gate_records(rows)
    assert gates["innovation_objective"]["passes"] and gates["gru_objective"]["passes"]
    assert not gates["architecture"]["passes"] and gates["architecture"]["passed"] == 28
    rows[-1]["name"] = rows[-2]["name"]
    with pytest.raises(ValueError, match="ordered"):
        plot.gate_records(rows)


def descriptor(path):
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return descriptor(path)


def phase(root, directory, script, cap, expected, *, mode=None):
    launch_path = root / (directory.name + "-launch.json")
    command = [str(root / ".venv/bin/python"), "-u", str(root / script)] + ([] if mode is None else [mode])
    options = {**expected, "--supervision": str(launch_path)}
    command += [item for pair in options.items() for item in pair]
    launch = {"command": command, "cwd": str(root), "cap_seconds": cap, "started_ns": 10**9,
        "deadline_ns": (cap + 1) * 10**9, "clock_source_sha256": plot.base.CLOCK_PIN,
        "watchdog_sha256": plot.base.SUPERVISOR_PIN, "pid": 10, "pgid": 10, "parent_pid": 9}
    save(launch_path, launch)
    worker = {"started_ns": 2 * 10**9, "finished_ns": 3 * 10**9, "wall_seconds": 1.,
              "supervision_sha256": descriptor(launch_path)["sha256"]}
    terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False, "error": None,
        "clock_error": None, "group_absent": True, "cleanup": {"reaped": True, "errors": []},
        "finished_ns": 4 * 10**9, "elapsed_ns": 3 * 10**9, "wall_seconds": 3.}
    return worker, terminal, launch


def closed_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "ROOT", tmp_path)
    monkeypatch.setattr(plot.base, "ROOT", tmp_path)
    summary, audit = aggregates()
    train, audit_dir, output = (tmp_path / name for name in ("train", "audit", "plots"))
    for directory in (train, audit_dir, output):
        directory.mkdir()
    source_names = {plot.AUDITOR, plot.TRAINER, "src/openjev/research/otto_action_focused_loss.py",
        "src/openjev/research/otto_action_focused_metrics.py", "src/openjev/research/otto_spo_plus_loss.py"}
    sources = {}
    for name in sorted(source_names):
        path = tmp_path / name; path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b"opaque fabricated source\n"); sources[name] = descriptor(path)["sha256"]
    input_path = tmp_path / "engineering.json"
    input_pin = save(input_path, {"status": "passed"})
    plan_path = tmp_path / "plan.json"
    plan_pin = save(plan_path, {"version": plot.TRAIN_VERSION, "status": "frozen_before_fitting",
        "sources": sources, "inputs": {"engineering": input_pin}})
    worker, terminal, launch = phase(tmp_path, train, plot.TRAINER, 14400,
        {"--plan": str(plan_path), "--plan-sha256": plan_pin["sha256"], "--output": str(train)}, mode="run")
    for name in plot.TRAIN_PAYLOADS:
        (train / name).write_bytes(b"opaque bytes, deliberately not NPZ or JSON\n")
    save(train / "summary.json", summary)
    save(train / "started.json", {"launch": launch, "started_ns": worker["started_ns"]})
    worker.update(version=plot.TRAIN_VERSION, status="completed", complete=True, fits_completed=12,
        optimizer_steps=8640, pending=None, pending_emission=None, requires_successful_original_supervisor=True,
        plan_sha256=plan_pin["sha256"], sources=sources, inputs={"engineering": input_pin},
        files={name: {k: v for k, v in descriptor(train / name).items() if k != "path"} for name in plot.TRAIN_PAYLOADS})
    roles = {"plan": plan_pin, "worker": save(train / "receipt.json", worker),
             "terminal": save(tmp_path / "training-terminal.json", terminal)}
    expected = {"--output": str(audit_dir)}
    for role, pin in roles.items():
        expected.update({"--" + role: pin["path"], "--" + role + "-sha256": pin["sha256"]})
    receipt, audit_terminal, launch = phase(tmp_path, audit_dir, plot.AUDITOR, 240, expected)
    save(audit_dir / "started.json", {"launch": launch, "started_ns": receipt["started_ns"], "producer_inputs": roles})
    save(audit_dir / "audit.json", audit)
    receipt.update(version=plot.AUDIT_VERSION, status="completed", agreement=True, failures=[],
        requires_successful_original_supervisor=True, producer_inputs=roles, plan_sha256=plan_pin["sha256"],
        sources={plot.AUDITOR: sources[plot.AUDITOR]},
        files={name: {k: v for k, v in descriptor(audit_dir / name).items() if k != "path"}
               for name in ("started.json", "audit.json")})
    receipt_pin = save(audit_dir / "receipt.json", receipt)
    terminal_path = tmp_path / "audit-terminal.json"
    terminal_pin = save(terminal_path, audit_terminal)
    args = SimpleNamespace(audit_directory=audit_dir, audit_receipt_sha256=receipt_pin["sha256"],
        audit_terminal=terminal_path, audit_terminal_sha256=terminal_pin["sha256"], output=output)
    return plot.Plot(args), summary, audit


def test_original_parent_and_all_payload_bytes_close_before_display(tmp_path, monkeypatch):
    runner, _, audit = closed_fixture(tmp_path, monkeypatch)
    summary, actual = runner.authenticate()
    assert actual == audit and summary == audit["summary"]
    assert len(plot.TRAIN_PAYLOADS) == 49
    assert sum(path.endswith(".npz") for path in runner.bound) == 40
    assert runner.receipt["original_seconds"] == {"training_worker": 1., "training_parent": 3., "audit_worker": 1., "audit_parent": 3.}
    assert all(not gate["passes"] for gate in summary["gates"].values())


@pytest.mark.parametrize("defect", ("timeout", "script", "source", "payload", "extra"))
def test_bad_original_closure_rejected_before_any_aggregate_decode(tmp_path, monkeypatch, defect):
    runner, _, _ = closed_fixture(tmp_path, monkeypatch)
    if defect in ("timeout", "script"):
        terminal = json.loads(runner.args.audit_terminal.read_text())
        if defect == "timeout":
            terminal["timed_out"] = True
        else:
            terminal["command"][2] = str(tmp_path / "wrong-script.py")
        runner.args.audit_terminal_sha256 = save(runner.args.audit_terminal, terminal)["sha256"]
    elif defect == "source":
        (tmp_path / plot.TRAINER).write_bytes(b"changed")
    elif defect == "payload":
        (tmp_path / "train/prediction-hold.npz").write_bytes(b"changed")
    else:
        (tmp_path / "train/unlisted.json").write_text("{}")
    seen, original = [], runner.read

    def read(path):
        seen.append(path.name)
        return original(path)

    monkeypatch.setattr(runner, "read", read)
    with pytest.raises(ValueError):
        runner.authenticate()
    assert "summary.json" not in seen and "audit.json" not in seen


def test_late_receipt_failure_is_demoted_and_original_error_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "ROOT", tmp_path)
    _, audit = aggregates()
    args = SimpleNamespace(output=tmp_path / "new-presentation")
    runner = plot.Plot(args)
    monkeypatch.setattr(runner, "bind", lambda *_: None)
    monkeypatch.setattr(runner, "descriptor", lambda path: {"path": str(path), "bytes": path.stat().st_size if path.exists() else 0, "sha256": "a" * 64})
    runner.receipt["original_seconds"] = {}
    monkeypatch.setattr(runner, "authenticate", lambda: (audit["summary"], audit))

    def figures(*_):
        for stem in plot.FIGURES:
            for extension in ("png", "svg"):
                (runner.out / f"{stem}.{extension}").write_bytes(b"fabricated figure")
        return "fabricated-only"

    def check():
        if (runner.out / "receipt.json").exists():
            raise RuntimeError("fabricated final publication failure")

    monkeypatch.setattr(runner, "figures", figures)
    monkeypatch.setattr(runner, "check", check)
    with pytest.raises(RuntimeError, match="final publication failure"):
        runner.execute()
    assert json.loads((runner.out / "receipt.invalid.json").read_text())["status"] == "completed"
    failed = json.loads((runner.out / "receipt.json").read_text())
    assert failed["status"] == "failed" and "final publication failure" in failed["error"]
    assert set(failed["files"]) == plot.PAYLOADS | {"receipt.invalid.json"}
    with pytest.raises(FileExistsError):
        runner.execute()
