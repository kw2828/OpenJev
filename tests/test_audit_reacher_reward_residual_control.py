"""Synthetic evidence corruption tests, with no real model or policy calls."""

from __future__ import annotations

import copy
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_reacher_reward_residual_control as audit

spec = importlib.util.spec_from_file_location(
    "old_residual_evidence_fixture", ROOT / "tests/test_audit_reacher_reward_residual_study.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
base, DIGEST = audit.base, "a" * 64


def rebind(execution):
    value = base.read(execution / "completed.json")
    value["files"] = {
        str(path.relative_to(execution)): base.sha(path)
        for path in execution.rglob("*")
        if path.is_file() and path.name != "completed.json"
    }
    # Nested completed receipts are members too.
    value["files"].update(
        {
            str(path.relative_to(execution)): base.sha(path)
            for path in (execution / "fits").rglob("completed.json")
        }
    )
    base.write(execution / "completed.json", value)


def bind_manifest(plan, execution):
    registry = audit.streams.concrete_streams(plan)
    plan["random_stream_contract"] = {
        "namespace": audit.VERSION,
        "registry": registry,
        "generators": audit.streams.generator_manifest(registry),
        "priors": [
            {"plan_path": source["plan_path"], "plan_sha256": source["plan_sha256"], "include_train": train}
            for source, train in ((plan["training_source"], True), (plan["fit_source"], False))
        ],
        "intentional_reuse": ["Paired control arms reuse corresponding named streams."],
        "draws_for_manifest": 0,
    }
    base.write(execution / "random-streams.json", plan["random_stream_contract"])
    path = execution / "evaluation-started.json"
    if path.exists():
        evaluation = base.read(path)
        if "random_streams_sha256" in evaluation:
            evaluation["random_streams_sha256"] = base.sha(execution / "random-streams.json")
            base.write(path, evaluation)


def build_fixture(tmp_path):
    parent = fixture.plan_fixture() | {"study": "reacher-reward-residual-v1", "version": 1}
    old = fixture.build_fixture(tmp_path, parent)
    original = tmp_path / "stopped"
    old.rename(original)
    (original / "completed.json").unlink()
    parent_path = tmp_path / "inherited-plan.json"
    base.write(parent_path, parent)
    parent_sha = base.sha(parent_path)
    boundary_path = original / "all-fits-completed.json"
    boundary = base.read(boundary_path)
    boundary["plan_sha256"] = parent_sha
    base.write(boundary_path, boundary)
    base.write(original / "failed.json", {"error": "KeyboardInterrupt()", "wall_seconds": 0.8})
    members = {
        f"fits/{name}/{file}": base.sha(original / "fits" / name / file)
        for name in parent["fit_order"]
        for file in audit.FIT_FILES
    }
    invalid = {
        "status": "invalid_protocol_stopped",
        "plan_sha256": parent_sha,
        "process_exit_code": 130,
        "resumable": False,
        "wall_seconds": 0.8,
        "failure_receipt_sha256": base.sha(original / "failed.json"),
        "all_six_fits_completed_sha256": base.sha(boundary_path),
        "completed_fit_members": members,
    }
    invalid_path = tmp_path / "invalid.json"
    base.write(invalid_path, invalid)
    train = base.load_records(original / "train", parent["train_episodes"])
    fits = {name: audit.prior.audit_fit(parent, original, name, train)[0] for name in parent["fit_order"]}
    inheritance = {
        "status": "all_six_fits_authenticated_for_unselected_inheritance",
        "source_plan_sha256": parent_sha,
        "source_invalid_receipt_sha256": base.sha(invalid_path),
        "fit_receipts_sha256": {name: members[f"fits/{name}/completed.json"] for name in parent["fit_order"]},
        "fits": {
            name: {key: value for key, value in fit.items() if key != "epochs"} for name, fit in fits.items()
        },
        "inherited_fit_wall_seconds": sum(fit["wall_seconds"] for fit in fits.values()),
        "paired_initial_tensors_identical": True,
        "training_roles_have_no_seed_collisions": True,
        "new_fits": 0,
        "new_model_calls": 0,
        "efficacy_metrics_read": False,
    }
    inheritance_path = tmp_path / "inheritance.json"
    base.write(inheritance_path, inheritance)
    plan = copy.deepcopy(parent)
    plan.update(version=2, study=audit.VERSION, **audit.streams.domain_bases(audit.VERSION))
    plan["fit_source"] = {
        "execution_path": str(original),
        "plan_path": str(parent_path),
        "plan_sha256": parent_sha,
        "invalid_receipt_path": str(invalid_path),
        "invalid_receipt_sha256": base.sha(invalid_path),
        "inheritance_receipt_path": str(inheritance_path),
        "inheritance_receipt_sha256": base.sha(inheritance_path),
        "all_fits_boundary_sha256": base.sha(boundary_path),
        "failure_receipt_sha256": base.sha(original / "failed.json"),
        "members": members,
        "inherited_fit_wall_seconds": inheritance["inherited_fit_wall_seconds"],
        "prior_invalid_attempt_wall_seconds": 0.8,
    }
    execution = tmp_path / "execution"
    shutil.copytree(original, execution)
    (execution / "all-fits-completed.json").rename(execution / "source-all-fits-completed.json")
    (execution / "failed.json").rename(execution / "source-failed.json")
    for member, (path_key, _) in audit.COPIES.items():
        shutil.copyfile(plan["fit_source"][path_key], execution / member)
    base.write(execution / "fit-provenance.json", plan["fit_source"])
    base.write(
        execution / "inherited-fits-ready.json",
        {
            "plan_sha256": DIGEST,
            "fit_order": plan["fit_order"],
            "files": boundary["files"],
            "unix_time": 1.001,
            "elapsed_seconds": 0.001,
            "inherited_fit_wall_seconds": inheritance["inherited_fit_wall_seconds"],
            "new_fits": 0,
        },
    )
    bind_manifest(plan, execution)
    base.write(
        execution / "evaluation-started.json",
        {
            "plan_sha256": DIGEST,
            "inherited_fits_ready_sha256": base.sha(execution / "inherited-fits-ready.json"),
            "random_streams_sha256": base.sha(execution / "random-streams.json"),
            "unix_time": 1.002,
            "elapsed_seconds": 0.002,
        },
    )
    fixture.save_records(execution / "prediction", [fixture.fake_record(plan, "prediction")])
    for panel in base.PANELS:
        for arm in base.arms(plan, panel):
            record = fixture.fake_record(plan, "control", panel=panel)
            if arm == "uniform":
                record["policy"]["commands"] = (
                    np.random.default_rng(plan["candidate_seed"] + 200000)
                    .uniform(-1, 1, (50, 1, 2))[:, 0]
                    .astype(np.float32)
                )
            fixture.save_records(execution / "control" / panel / arm, [record])
    base.write(
        execution / "completed.json",
        {
            "status": "completed",
            "plan_sha256": DIGEST,
            "fits": 6,
            "new_fits": 0,
            "astra_calls": 0,
            "wall_seconds": 1.0,
            "evaluation_started_elapsed_seconds": 0.002,
            "inherited_fit_wall_seconds": inheritance["inherited_fit_wall_seconds"],
            "prior_invalid_attempt_wall_seconds": 0.8,
            "files": {},
        },
    )
    rebind(execution)
    return plan, execution


@pytest.fixture
def saved(tmp_path, monkeypatch):
    plan, execution = build_fixture(tmp_path)
    checked = []

    def replay(record):
        checked.append(record)
        return {"transitions": 50, "max_abs_error": 0.0, "saved_output_only": True, "new_policy_calls": 0}

    monkeypatch.setattr(base, "native_replay", replay)
    return plan, execution, tmp_path / "audit", checked


def test_all_inherited_fits_fresh_cohorts_and_costs_are_separate(saved):
    plan, execution, out, checked = saved
    summary = audit.audit_saved(plan, DIGEST, execution, out)
    assert len(checked) == 45 and summary["native_transitions_checked"] == 2250
    assert len(summary["continuation_gate"]["checks"]) == 17
    assert len(summary["fits"]) == 6 and summary["new_fits"] == 0
    assert (
        summary["phase_boundary"]["inherited_fits_ready_elapsed_seconds"]
        < summary["costs"]["inherited_fit_wall_seconds"]
    )
    assert summary["costs"]["new_evaluation_wall_seconds"] == 1.0
    assert summary["costs"]["cumulative_attempt_wall_seconds"] == 1.8
    assert summary["random_streams"]["priors_checked"] == 2
    receipt = base.read(out / "receipt.json")
    assert receipt["random_stream_contract"] == plan["random_stream_contract"]
    assert receipt["files"]["summary.json"] == base.sha(out / "summary.json")


@pytest.mark.parametrize(
    "key",
    ["epochs", "fit_order", "fit_seeds", "candidates", "ordinary_gap", "cap_seconds", "training_source"],
)
def test_parent_settings_are_immutable(saved, key):
    plan, execution, out, _ = saved
    if key in ("fit_order", "fit_seeds"):
        plan[key] = list(reversed(plan[key]))
    elif key == "training_source":
        plan[key] = plan[key] | {"extra": True}
    else:
        plan[key] += 1
    with pytest.raises(ValueError, match="Inherited plan field changed"):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra_weights", "member set"),
        ("missing_initial", "member set"),
        ("changed_weight", "Inherited fit bytes"),
        ("old_complete", "remain stopped"),
        ("copied_plan", "Inherited source hash"),
        ("copied_failure", "Inherited terminal hash"),
        ("source_manifest", "Inherited fit member set"),
        ("fit_provenance", "Copied fit provenance"),
        ("new_fits", "completion identity"),
        ("new_fits_bool", "completion identity"),
        ("doublecount", "timing binding"),
        ("cap", "cap exceeded"),
        ("legacy_boundary", "member set"),
        ("boundary_members", "boundary fit membership"),
        ("boundary_new_fit", "boundary identity"),
        ("boundary_order", "boundary identity"),
        ("boundary_hash", "dependency hash"),
        ("boundary_timing", "chronology"),
        ("evaluation_time", "timing binding"),
        ("negative_timing", "Execution timing"),
    ],
)
def test_inheritance_and_cost_corruption_cannot_be_rehashed_away(saved, mutation, match):
    plan, execution, out, _ = saved
    if mutation == "extra_weights":
        torch.save({}, execution / "fits/free-271/best.pt")
    elif mutation == "missing_initial":
        (execution / "fits/free-271/initial-weights.pt").unlink()
    elif mutation == "changed_weight":
        path = execution / "fits/free-271/weights.pt"
        value = torch.load(path, weights_only=True)
        value[next(iter(value))] += 1
        torch.save(value, path)
    elif mutation == "old_complete":
        base.write(Path(plan["fit_source"]["execution_path"]) / "completed.json", {})
    elif mutation in ("copied_plan", "copied_failure", "fit_provenance"):
        filename = {
            "copied_plan": "inherited-plan.json",
            "copied_failure": "source-failed.json",
            "fit_provenance": "fit-provenance.json",
        }[mutation]
        value = base.read(execution / filename) | {"tampered": True}
        base.write(execution / filename, value)
    elif mutation == "source_manifest":
        plan["fit_source"]["members"].pop("fits/free-271/weights.pt")
        base.write(execution / "fit-provenance.json", plan["fit_source"])
    elif mutation == "legacy_boundary":
        shutil.copyfile(execution / "inherited-fits-ready.json", execution / "all-fits-completed.json")
    elif mutation.startswith("boundary"):
        if mutation == "boundary_hash":
            path = execution / "evaluation-started.json"
            value = base.read(path) | {"inherited_fits_ready_sha256": "f" * 64}
        else:
            path = execution / "inherited-fits-ready.json"
            value = base.read(path)
            if mutation == "boundary_members":
                value["files"].pop(next(iter(value["files"])))
            if mutation == "boundary_new_fit":
                value["new_fits"] = 1
            if mutation == "boundary_order":
                value["fit_order"].reverse()
            if mutation == "boundary_timing":
                value["elapsed_seconds"] = 0.003
        base.write(path, value)
        if mutation != "boundary_hash":
            evaluation = base.read(execution / "evaluation-started.json")
            evaluation["inherited_fits_ready_sha256"] = base.sha(path)
            base.write(execution / "evaluation-started.json", evaluation)
    else:
        path = execution / "completed.json"
        value = base.read(path)
        key, new = {
            "new_fits": ("new_fits", 1),
            "new_fits_bool": ("new_fits", False),
            "doublecount": ("inherited_fit_wall_seconds", 1.0),
            "cap": ("wall_seconds", 101.0),
            "evaluation_time": ("evaluation_started_elapsed_seconds", 0.004),
            "negative_timing": ("wall_seconds", -1),
        }[mutation]
        value[key] = new
        base.write(path, value)
    rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("manifest_seed", "registry mismatch"),
        ("manifest_state", "state mismatch"),
        ("spawn_child", "state mismatch"),
        ("missing_prior", "Both prior"),
        ("wrong_prior_scope", "Prior stream identity"),
        ("draws", "declaration"),
        ("copy", "Copied random stream"),
        ("namespace", "namespace"),
    ],
)
def test_stream_manifest_is_reexpanded_and_checks_every_prior(saved, mutation, match):
    plan, execution, out, _ = saved
    contract = plan["random_stream_contract"]
    if mutation == "manifest_seed":
        contract["registry"]["control/actuator_noise/0"] += 1
    if mutation == "manifest_state":
        contract["generators"]["planner/candidate_bank/0"]["initial_state_sha256"] = "b" * 64
    if mutation == "spawn_child":
        contract["generators"].pop("planner/particle_filter/0/resample")
    if mutation == "missing_prior":
        contract["priors"].pop()
    if mutation == "wrong_prior_scope":
        contract["priors"][0]["include_train"] = False
    if mutation == "draws":
        contract["draws_for_manifest"] = 1
    if mutation == "namespace":
        contract["namespace"] = "other"
    base.write(execution / "random-streams.json", {} if mutation == "copy" else contract)
    rebind(execution)
    with pytest.raises(ValueError, match=match):
        audit.audit_saved(plan, DIGEST, execution, out)


def test_shift_threshold_failure_still_fails_and_reset_is_descriptive():
    plan = fixture.plan_fixture()
    predictions, control = fixture.passing_metrics(plan)
    gate, _ = audit.prior.qualify(plan, predictions, control)
    assert gate["passed"] and len(gate["checks"]) == 17
    control["shift"]["residual-293"] = {"mean_cost": 90.000001, "episode_costs": [90.000001]}
    gate, _ = audit.prior.qualify(plan, predictions, control)
    assert not gate["passed"] and gate["reset_is_descriptive_only"]


@pytest.mark.parametrize(
    "source_name,field,new_role,offset",
    [
        ("training_source", "train_seed", "planner/candidate_bank/0", 0),
        ("training_source", "noise_seed", "analysis/bootstrap/0", 0),
        ("training_source", "schedule_seed", "floor/uniform/0", 0),
        ("training_source", "exploration_seed", "planner/particle_filter/0", 0),
        ("fit_source", "bootstrap_seed", "control/actuator_noise/0", 0),
        ("fit_source", "candidate_seed", "prediction/reset/0", 200000),
        ("fit_source", "noise_seed", "prediction/exploration/0", 200000),
        ("fit_source", "filter_seed", "control/sensor_schedule/0", 0),
    ],
)
def test_cross_role_overlap_with_either_prior_is_rejected(saved, source_name, field, new_role, offset):
    plan, execution, _, _ = saved
    source = plan[source_name]
    path = Path(source["plan_path"])
    parent = base.read(path)
    parent[field] = plan["random_stream_contract"]["registry"][new_role] - offset
    base.write(path, parent)
    source["plan_sha256"] = base.sha(path)
    bind_manifest(plan, execution)
    with pytest.raises(ValueError, match="overlap a prior random-stream registry"):
        audit.validate_streams(plan, execution)


def test_control_timing_cannot_exceed_new_evaluation_interval(saved):
    plan, execution, out, _ = saved
    path = execution / "control/ordinary/known_state-planning.npz"
    values = base.load_npz(path)
    values["batch_decision_seconds"][:] = 1.0
    np.savez_compressed(path, **values)
    rebind(execution)
    with pytest.raises(ValueError, match="Control timings exceed new evaluation interval"):
        audit.audit_saved(plan, DIGEST, execution, out)


def test_source_paths_are_repository_relative_and_manifest_draws_do_not_touch_global_rng(
    saved, monkeypatch, tmp_path
):
    plan, execution, out, _ = saved
    before = np.random.get_state()
    audit.validate_streams(plan, execution)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    # New fit-source paths resolve relative to the repository, not process cwd.
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    source = plan["fit_source"]
    for key in ("execution_path", "plan_path", "invalid_receipt_path", "inheritance_receipt_path"):
        source[key] = str(Path(source[key]).relative_to(tmp_path))
    base.write(execution / "fit-provenance.json", source)
    bind_manifest(plan, execution)
    rebind(execution)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert audit.audit_saved(plan, DIGEST, execution, out)["status"] == "completed"


def test_native_mismatch_does_not_create_success_receipt(saved, monkeypatch):
    plan, execution, out, _ = saved

    def fail(_record):
        raise ValueError("Native replay mismatch")

    monkeypatch.setattr(base, "native_replay", fail)
    with pytest.raises(ValueError, match="Native replay mismatch"):
        audit.audit_saved(plan, DIGEST, execution, out)
    assert not out.exists()
