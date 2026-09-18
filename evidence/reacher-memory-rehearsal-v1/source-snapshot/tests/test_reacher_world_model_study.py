"""Engineering fixtures do not use the prospective scored cohort."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/reacher_world_model_study.py"
SPEC = importlib.util.spec_from_file_location("reacher_study", PATH)
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


def miniature():
    return {
        "steps": 50, "schedule_seed": 123, "ordinary_gap": 6, "shift_gap": 10,
        "candidate_seed": 332, "planning_horizon": 3, "candidates": 8, "action_block": 2,
        "control_episodes": 2, "control_seed": 870001, "noise_seed": 880001,
        "noise_std": .05, "filter_seed": 890001, "particles": 4,
        "filter_bandwidth": .02,
    }


def test_schedule_information_and_shift_are_explicit():
    p = miniature()
    ordinary, shifted = s.schedule(p, 0), s.schedule(p, 0, "shift")
    assert ordinary.shape == (51,) and ordinary[0]
    assert (~ordinary).sum() == 12 and (~shifted).sum() == 20
    assert not np.any(shifted & ~ordinary)
    assert s.schedule(p, 0, "full").all()


def test_bank_is_identical_across_models_and_truncates():
    p = miniature()
    np.testing.assert_array_equal(s.candidate_bank(p, 0, 2), s.candidate_bank(p, 0, 2))
    assert s.candidate_bank(p, 49, 2).shape == (2, 8, 1, 2)
    assert np.all(s.candidate_bank(p, 0, 2)[:, 0] == 0)


def test_learning_allowlist_excludes_privileged_fields():
    record = {"policy": {"packets": np.ones((51, 8)), "commands": np.zeros((50, 2))},
              "audit": {"rewards": np.ones(50), "qvel": "DO NOT READ", "integration_state": object()}}
    p, a, r = s.learning_tensors([record])
    assert p.shape == (1, 51, 8) and a.shape == (1, 50, 2) and r.shape == (1, 50)


def test_saved_nested_episode_roundtrip(tmp_path):
    record = {"policy": {"packets": np.ones((51, 8)), "commands": np.zeros((50, 2))},
              "audit": {"rewards": np.ones(50)}, "metadata": {"seed": 41}}
    s.save_records(tmp_path / "cohort", [record])
    loaded = s.load_records(tmp_path / "cohort")[0]
    for category in ("policy", "audit"):
        for key in record[category]:
            np.testing.assert_array_equal(record[category][key], loaded[category][key])
    assert loaded["metadata"] == record["metadata"]


def test_mpc_matches_independent_scalar_candidate_rollout():
    torch.manual_seed(731)
    model = s.make_world_model("gru", hidden_size=8).eval()
    state = model.initial(2)
    packet = torch.tensor([[1., 1., 0., 0., .1, .1, 1., 0.]]).repeat(2, 1)
    state = model.assimilate(state, packet)
    bank = s.candidate_bank(miniature(), 0, 2)
    actual, scores = s.learned_mpc(model, state, bank)
    with torch.no_grad():
        for i in range(2):
            expected = []
            for candidate in bank[i]:
                trial = {k: v[i:i + 1].clone() for k, v in state.items()}
                total = 0.
                for action in candidate:
                    trial, _, reward = model.advance(trial, torch.tensor(action)[None])
                    total += float(reward.clamp(-2.5, 0))
                expected.append(total)
            np.testing.assert_allclose(scores[i], expected, atol=1e-6)
            np.testing.assert_array_equal(actual[i], bank[i, np.argmax(expected), 0])


def test_cap_is_terminal():
    with pytest.raises(TimeoutError):
        s.check_cap(time.monotonic() - 1)


def test_tiny_learned_rollout_retains_every_case_without_hidden_input():
    p = miniature()
    torch.manual_seed(911)
    model = s.make_world_model("gru", hidden_size=8).eval()
    records, extra = s.learned_control(p, model, "ordinary", True, time.monotonic() + 30)
    assert len(records) == 2
    assert all(r["policy"]["packets"].shape == (51, 8) for r in records)
    assert extra["candidate_scores"].shape == (2, 50, 8)
    assert extra["planner_used"]
    assert np.isfinite(extra["candidate_scores"]).all()


def test_complete_tiny_pipeline_and_independent_saved_audit(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    s.prepare(tmp_path / "protocol")
    path = tmp_path / "protocol/plan.json"
    plan = json.loads(path.read_text())
    plan.update(train_episodes=4, prediction_episodes=2, control_episodes=2,
                fit_seeds=[211], epochs=1, batch_size=2, candidates=8,
                planning_horizon=3, particles=4, bootstrap_samples=32,
                train_seed=8720001, prediction_seed=8730001, control_seed=8740001,
                schedule_seed=8750001, noise_seed=8760001,
                exploration_seed=8770001, candidate_seed=8780001,
                filter_seed=8790001, bootstrap_seed=8800001,
                engineering_fixture=True)
    s.write(path, plan)
    digest = s.sha(path)
    execution, audit = tmp_path / "execution", tmp_path / "audit"
    s.run(path, digest, execution)
    s.audit(path, digest, execution, audit)
    summary = json.loads((audit / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["new_policy_calls"] == summary["new_mpc_calls"] == 0
    assert summary["native_transitions_checked"] == (4 + 2 + 2 * (7 + 9 + 9)) * 50
    with pytest.raises(FileExistsError):
        s.run(path, digest, execution)
    with pytest.raises(ValueError, match="identity"):
        s.run(path, "0" * 64, tmp_path / "never-started")
