"""Artificial portable-subset tests; no corpus, models, or actual fit outputs."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"scripts/recheck_dialogue_alignment.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")


def record(module, path):
    return {"sha256": module.digest(path), "bytes": path.stat().st_size}


def synthetic_rows():
    rows = []
    ids = ["reserved:NOT_MENTIONED", "reserved:DONTCARE", "value:True", "value:False"]
    for i, (target, old) in enumerate(((2, 0), (1, 0), (0, 0), (2, 2))):
        rows.append({"row_index": i, "split": "train", "admission": "admitted", "heldout_service": True,
            "candidate_count": 4, "candidate_types": [0, 1, 2, 3], "current_label_index": target,
            "previous_current_index": old, "current_candidate_id": ids[target], "previous_candidate_id": ids[old],
            "current_value_group": ("none", "dontcare", "true", "false")[target],
            "derived_bin": "first_assignment" if target != old else
                "unmentioned_retention" if target == 0 else "assigned_retention",
            "service": "A" if i % 2 else "B", "dialogue_id": str(i)})
    return rows


def seal(fixture):
    """Re-seal modified synthetic artifacts so semantic guards are exercised."""
    m, run, report = fixture.m, fixture.args.run, fixture.args.report
    save(run/"plan.json", fixture.plan)
    for name in fixture.selected:
        fixture.done["files"][name] = record(m, run/name)
    fixture.done["plan_sha256"] = m.digest(run/"plan.json")
    save(run/"completed.json", fixture.done)
    fixture.args.completed_sha256 = m.digest(run/"completed.json")
    fixture.summary.update(plan_sha256=fixture.done["plan_sha256"],
                           execution_completed_sha256=fixture.args.completed_sha256)
    save(report/"summary.json", fixture.summary)
    fixture.receipt.update(plan_sha256=fixture.done["plan_sha256"],
        execution_completed_sha256=fixture.args.completed_sha256,
        execution_members=copy.deepcopy(fixture.done["files"]),
        files={"summary.json": record(m, report/"summary.json")})
    save(report/"receipt.json", fixture.receipt)
    fixture.args.report_receipt_sha256 = m.digest(report/"receipt.json")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Copy the checker and its exact helper to a different checkout location.
    initial = load(SOURCE, "portable_template")
    checkout = tmp_path/"old checkout"
    (checkout/"scripts").mkdir(parents=True)
    shutil.copyfile(SOURCE, checkout/"scripts"/SOURCE.name)
    helper = checkout/initial.HELPER
    helper.parent.mkdir(parents=True)
    shutil.copyfile(ROOT/initial.HELPER, helper)
    run, report = checkout/"subset", checkout/"report"
    run.mkdir(); report.mkdir()
    rows = synthetic_rows()
    (run/"evaluation-rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    logs = np.full((4, 12), -np.inf, np.float32)
    logs[:, :4] = np.log(np.array([[.35, .05, .3, .3], [.1, .7, .1, .1],
                                 [.8, .1, .05, .05], [.1, .1, .7, .1]]))
    for name in initial.ORDER:
        dest = run/"fits"/name
        dest.mkdir(parents=True)
        np.savez(dest/"predictions.npz", row_indices=np.arange(4, dtype=np.int64), log_probs=logs)
    # Physically move the entire fixture before importing or using its checker.
    moved = tmp_path/"relocated"/"checkout"
    moved.parent.mkdir()
    shutil.move(str(checkout), moved)
    m = load(moved/"scripts"/SOURCE.name, "relocated_portable_checker")
    assert m.ROOT == moved
    # Scale only row-count constants for this artificial four-row envelope.
    monkeypatch.setattr(m, "SUPPORT", {"evaluation_rows": 4, "heldout_rows": 4,
        "changed": 2, "retained": 2, "heldout_services": 2, "changed_types": [0, 1, 1, 0]})
    run, report = moved/"subset", moved/"report"
    arithmetic = m.load_arithmetic()
    meta = arithmetic.row_metadata(rows)
    scores, choices = arithmetic.score(rows, meta, {"row_indices": np.arange(4, dtype=np.int64), "log_probs": logs})
    fits = {name: copy.deepcopy(scores) for name in m.ORDER}
    pairs = {control: {str(seed): arithmetic.pair_counts(meta, choices, choices) for seed in m.SEEDS}
             for control in m.METHODS[:2]}
    rule = arithmetic.continuation(fits)
    checks = [{"control": key.split("/", 1)[0], "name": key.split("/", 1)[1], **value}
              for key, value in rule["checks"].items()]
    sources = {"scripts/report_dialogue_alignment.py": "a"*64}
    plan = {"version": m.STUDY_VERSION, "expected_fits": m.ORDER,
        "config": {"methods": list(m.METHODS), "seeds": list(m.SEEDS)}, "source_sha256": sources,
        "evaluation_row_indices": [0, 1, 2, 3], "split": {"heldout_services": ["A", "B"]},
        "prepared_path": "/absent/original/cache", "schema_cache_path": "/absent/old/schema",
        "capacity_plan_path": "/absent/old/capacity/plan.json"}
    manifest = {name: {"sha256": "0"*64, "bytes": 99} for name in m.expected_manifest()}
    done = {"version": m.STUDY_VERSION, "status": "completed", "phase": "train",
        "completed_fits": m.ORDER, "expected_fits": m.ORDER, "files": manifest,
        "source_sha256": sources, "progress": {"completed_fits": m.ORDER, "active_fit": None}}
    summary = {"status": "completed", "version": m.STUDY_VERSION, "technical_validity_passed": True,
        "source_sha256": sources, "fits": fits, "pairs": pairs, "continuation": {**rule, "checks": checks}}
    receipt = {"status": "completed", "version": m.STUDY_VERSION, "technical_validity_passed": True,
        "execution_files": 44, "source_sha256": sources["scripts/report_dialogue_alignment.py"]}
    args = argparse.Namespace(run=run, completed_sha256="", report=report,
                              report_receipt_sha256="", out=tmp_path/"result")
    fixture = argparse.Namespace(m=m, args=args, plan=plan, done=done, rows=rows,
        summary=summary, receipt=receipt, selected={"plan.json", "evaluation-rows.jsonl"} |
            {f"fits/{name}/predictions.npz" for name in m.ORDER})
    seal(fixture)
    return fixture


def test_moved_minimal_bundle_recomputes_all_primary_metrics(evidence):
    f = evidence
    result = f.m.execute(f.args)
    assert result["metric_agreement"] and not result["continuation"]["passed"]
    assert result["continuation"]["total_checks"] == 22
    assert len(result["fits"]) == 9 and len(result["selected_run_members"]) == 11
    assert "technical_validity_passed" not in result
    cell = result["fits"][f.m.ORDER[0]]["cells"]["heldout_service/changed"]
    assert cell["nll"]["row"] == pytest.approx(-.5*(np.log(.3)+np.log(.7)))
    assert cell["wrong_selected_branch"]["row"] == .5  # Largest branch mass would pick concrete.
    assert not (f.args.run/"started.json").exists() and not (f.args.run/"references.npz").exists()
    assert all(not (f.args.run/"fits"/name/"weights.pt").exists() for name in f.m.ORDER)
    receipt = json.loads((f.args.out/"receipt.json").read_text())
    assert receipt["arithmetic_source_sha256"] == f.m.HELPER_SHA256
    assert receipt["files"]["summary.json"] == record(f.m, f.args.out/"summary.json")
    assert receipt["model_calls"] == receipt["checkpoint_deserializations"] == 0


@pytest.mark.parametrize("kind", ["incomplete", "active", "order", "manifest", "report", "check", "rows", "panel"])
def test_resealed_identity_or_metric_corruption_rejected(evidence, kind):
    f = evidence
    if kind == "incomplete": f.done["completed_fits"] = f.m.ORDER[:-1]
    elif kind == "active": f.done["progress"]["active_fit"] = {"name": f.m.ORDER[-1]}
    elif kind == "order": f.plan["expected_fits"] = list(reversed(f.m.ORDER))
    elif kind == "manifest": del f.done["files"]["references.npz"]
    elif kind == "report": f.summary["fits"][f.m.ORDER[0]]["cells"]["heldout_service/changed"]["nll"]["row"] += .1
    elif kind == "check": f.summary["continuation"]["checks"][0]["passed"] = True
    elif kind == "rows":
        (f.args.run/"evaluation-rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in reversed(f.rows)))
    else:
        f.rows[0]["heldout_service"] = False
        (f.args.run/"evaluation-rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in f.rows))
    seal(f)
    with pytest.raises(ValueError):
        f.m.execute(f.args)
    assert (f.args.out/"failed.json").exists() and not (f.args.out/"receipt.json").exists()


@pytest.mark.parametrize("kind", ["row_order", "mass", "padding", "nan", "dtype", "member"])
def test_resealed_prediction_corruption_rejected(evidence, kind):
    f = evidence
    path = f.args.run/"fits"/f.m.ORDER[-1]/"predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        packet = {key: archive[key] for key in archive.files}
    if kind == "row_order": packet = {k: v[::-1] for k, v in packet.items()}
    elif kind == "mass": packet["log_probs"][0, 0] = 0
    elif kind == "padding": packet["log_probs"][0, 5] = -10
    elif kind == "nan": packet["log_probs"][0, 0] = np.nan
    elif kind == "dtype": packet["log_probs"] = packet["log_probs"].astype(np.float64)
    else: packet["foreign"] = np.zeros(1)
    np.savez(path, **packet)
    seal(f)
    with pytest.raises(ValueError):
        f.m.execute(f.args)


def test_hash_failure_precedes_all_prediction_decoding(evidence, monkeypatch):
    f = evidence
    path = f.args.run/"fits"/f.m.ORDER[-1]/"predictions.npz"
    path.write_bytes(path.read_bytes()+b"corrupt")
    def forbidden(*_args, **_kwargs):
        pytest.fail("NPZ loading before full selected-payload authentication")
    monkeypatch.setattr(f.m.np, "load", forbidden)
    with pytest.raises(ValueError, match="Selected payload binding"):
        f.m.execute(f.args)


def test_missing_prediction_is_not_skipped(evidence):
    f = evidence
    (f.args.run/"fits"/f.m.ORDER[0]/"predictions.npz").unlink()
    with pytest.raises(ValueError, match="Selected payload binding"):
        f.m.execute(f.args)


def test_helper_pin_and_exclusive_output(evidence):
    f = evidence
    with (f.m.ROOT/f.m.HELPER).open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="Arithmetic helper source pin"):
        f.m.execute(f.args)
    before = (f.args.out/"failed.json").read_bytes()
    with pytest.raises(FileExistsError):
        f.m.execute(f.args)
    assert (f.args.out/"failed.json").read_bytes() == before


def test_external_completion_pin_is_required(evidence):
    evidence.args.completed_sha256 = "f"*64
    with pytest.raises(ValueError, match="External completion pin"):
        evidence.m.execute(evidence.args)


def test_output_must_not_mutate_input_tree(evidence):
    evidence.args.out = evidence.args.run/"new-result"
    with pytest.raises(ValueError, match="Separate exclusive output"):
        evidence.m.execute(evidence.args)
    assert not evidence.args.out.exists()


def test_fixed_real_support_contract():
    module = load(SOURCE, "portable_support_contract")
    assert module.SUPPORT == {"evaluation_rows": 13599, "heldout_rows": 7819, "changed": 578,
        "retained": 7241, "heldout_services": 6, "changed_types": [0, 5, 29, 0]}
    assert len(module.expected_manifest()) == 43
