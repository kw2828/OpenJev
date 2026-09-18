"""Small engineering cohorts, disjoint from the scored residual experiment."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("reacher_residual_fixture", SCRIPTS / "reacher_reward_residual_study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture_source(tmp_path):
    """Create an explicitly synthetic parent receipt for four native episodes."""
    s.base.prepare(tmp_path / "parent-protocol")
    path = tmp_path / "parent-protocol/plan.json"
    parent = json.loads(path.read_text())
    parent.update(train_episodes=4, train_seed=8800001, schedule_seed=8810001,
                  noise_seed=8820001, exploration_seed=8830001, engineering_fixture=True)
    s.write(path, parent)
    records = [s.collect_episode(parent["train_seed"] + i, s.base.schedule(parent, i),
        noise_std=parent["noise_std"], noise_seed=parent["noise_seed"] + i,
        action_seed=parent["exploration_seed"] + i, policy="mixed") for i in range(4)]
    execution = tmp_path / "parent-execution"
    execution.mkdir()
    s.base.save_records(execution / "train", records)
    members = {name: s.sha(execution / name) for name in ("train.npz", "train.json")}
    s.write(execution / "completed.json", {"status": "completed", "plan_sha256": s.sha(path),
                                           "files": members, "synthetic_fixture": True})
    audit_path = tmp_path / "synthetic-parent-receipt.json"
    s.write(audit_path, {"status": "completed", "plan_sha256": s.sha(path),
                         "execution_members": members, "synthetic_fixture": True,
                         "execution_completed_sha256": s.sha(execution / "completed.json")})
    return {"execution_path": str(execution), "plan_path": str(path), "plan_sha256": s.sha(path),
            "audit_receipt_path": str(audit_path), "audit_receipt_sha256": s.sha(audit_path),
            "members": members, "cohort_plan": {key: parent[key] for key in s.COHORT_KEYS}}


def tiny_plan(tmp_path):
    source = fixture_source(tmp_path)
    s.prepare(tmp_path / "protocol", training_source=source)
    path = tmp_path / "protocol/plan.json"
    plan = json.loads(path.read_text())
    plan.update(prediction_episodes=2, control_episodes=2, fit_seeds=[919],
                fit_order=["residual-919", "free-919"], epochs=1, batch_size=2,
                candidates=8, planning_horizon=3, particles=4, bootstrap_samples=32,
                hidden_size=8, prediction_seed=8900001, control_seed=8910001,
                schedule_seed=8920001, noise_seed=8930001, exploration_seed=8940001,
                candidate_seed=8950001, filter_seed=8960001, bootstrap_seed=8970001,
                cap_seconds=120, engineering_fixture=True)
    s.write(path, plan)
    return path, plan


def test_parent_identity_cannot_be_relabelled(tmp_path):
    source = fixture_source(tmp_path)
    source["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        s.authenticate_training(source)


def test_training_authentication_rejects_mutated_member(tmp_path):
    source = fixture_source(tmp_path)
    s.authenticate_training(source)
    path = Path(source["execution_path"]) / "train.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="member changed"):
        s.authenticate_training(source)


def test_plan_rejects_duplicate_order_and_changed_parent_task(tmp_path):
    path, plan = tiny_plan(tmp_path)
    plan["fit_order"] = ["free-919", "free-919"]
    s.write(path, plan)
    with pytest.raises(ValueError, match="membership/order"):
        s.validate(path, s.sha(path))
    plan["fit_order"] = ["free-919", "residual-919"]
    plan["noise_std"] = 0.2
    s.write(path, plan)
    with pytest.raises(ValueError, match="Training task"):
        s.validate(path, s.sha(path))


def test_complete_tiny_matched_run_and_saved_audit(tmp_path):
    path, plan = tiny_plan(tmp_path)
    digest = s.sha(path)
    execution, audit = tmp_path / "execution", tmp_path / "audit"
    s.run(path, digest, execution)
    assert s.sha(execution / "train.npz") == plan["training_source"]["members"]["train.npz"]
    initials = [torch.load(execution / "fits" / f"{kind}-919" / "initial-weights.pt", weights_only=True)
                for kind in plan["kinds"]]
    assert initials[0].keys() == initials[1].keys()
    assert all(torch.equal(initials[0][key], initials[1][key]) for key in initials[0])
    expected_order = hashlib.sha256(torch.randperm(4, generator=torch.Generator().manual_seed(919 + 4100000)).numpy().tobytes()).hexdigest()
    for kind in plan["kinds"]:
        receipt = json.loads((execution / "fits" / f"{kind}-919" / "completed.json").read_text())
        assert receipt["minibatch_order_sha256"] == expected_order
        assert receipt["residual_reward"] == (kind == "residual")
    summary = s.audit(path, digest, execution, audit)
    assert summary["status"] == "completed"
    assert summary["new_model_calls"] == summary["new_policy_calls"] == summary["new_mpc_calls"] == 0
    assert summary["native_transitions_checked"] == (4 + 2 + 2 * (6 + 8 + 8)) * 50
    assert len(summary["fits"]) == 2
    with pytest.raises(FileExistsError):
        s.run(path, digest, execution)


def test_expired_run_preserves_failure_without_fit(tmp_path):
    path, plan = tiny_plan(tmp_path)
    plan["cap_seconds"] = 0.0
    s.write(path, plan)
    execution = tmp_path / "execution"
    with pytest.raises(TimeoutError):
        s.run(path, s.sha(path), execution)
    receipt = json.loads((execution / "failed.json").read_text())
    assert receipt["progress"]["phase"] == "copy-training"
    assert not (execution / "completed.json").exists()
    assert not (execution / "fits").exists()
