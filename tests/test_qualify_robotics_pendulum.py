"""Qualification tests use tiny engineering fixtures, never scored study data."""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
from gymnasium.envs.classic_control.pendulum import PendulumEnv

PATH = Path(__file__).resolve().parents[1] / "scripts/qualify_robotics_pendulum.py"
SPEC = importlib.util.spec_from_file_location("pendulum_qualification", PATH)
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


@pytest.fixture
def plan(tmp_path):
    q.prepare(tmp_path / "protocol")
    path = tmp_path / "protocol/plan.json"
    value = json.loads(path.read_text())
    value.update(cases=2, steps=5, horizon=3, candidates=5, switch_step=3, bootstrap_samples=32)
    q.write(path, value)
    return path, value


def test_mpc_choice_matches_independent_scalar_gymnasium():
    states = np.array([[2., .4], [-.5, -.3]])
    gains = np.array([.65, 1.35])
    actions = np.random.default_rng(37).uniform(-2, 2, (2, 7, 4))
    expected = []
    native = PendulumEnv()
    for i in range(2):
        costs = []
        for candidate in actions[i]:
            native.state = states[i].copy()
            cost = 0.
            for command in candidate:
                _, reward, _, _, _ = native.step(np.array([np.clip(command * gains[i], -2, 2)]))
                cost -= reward
            costs.append(cost)
        expected.append(actions[i, np.argmin(costs), 0])
    np.testing.assert_array_equal(q.mpc(states, gains, actions, q.Config()), expected)
    native.close()


def test_candidates_paired_and_truncated(plan):
    _, p = plan
    np.testing.assert_array_equal(q.candidates(p, 0, 2), q.candidates(p, 0, 2))
    assert q.candidates(p, 4, 2).shape == (2, 5, 1)
    np.testing.assert_array_equal(q.candidates(p, 0, 2)[:, 0], 0.)


def test_probe_and_hidden_gain_schedule(plan):
    _, p = plan
    a, _ = q.rollout(p, "switch", "current_nominal", deadline=time.monotonic() + 10)
    np.testing.assert_array_equal(a["command"][:, :2], np.tile([.8, -.8], (2, 1)))
    np.testing.assert_array_equal(a["gain"][:, 3], 2 - a["gain"][:, 0])
    np.testing.assert_array_equal(a["estimated"][:, 2:, 2], 1.)
    np.testing.assert_array_equal(a["estimated"][:, 2:, 1], 0.)


def test_deadline_does_not_silently_continue(plan):
    _, p = plan
    with pytest.raises(TimeoutError):
        q.rollout(p, "stationary", "current_nominal", deadline=0.)


def test_native_replay_rejects_corrupted_transition(plan):
    _, p = plan
    a, _ = q.rollout(p, "stationary", "uniform", deadline=time.monotonic() + 10)
    a["state"][0, 2, 1] += .1
    with pytest.raises(ValueError, match="replay mismatch"):
        q.native_replay(a["state"], a["command"], a["gain"], p["config"])


def test_end_to_end_saved_output_audit_and_no_overwrite(plan, tmp_path):
    path, p = plan
    execution, audit = tmp_path / "execution", tmp_path / "audit"
    q.run(path, execution, expected_plan_sha256=q.sha(path))
    result = q.audit(path, execution, audit, expected_plan_sha256=q.sha(path))
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["independent_replayed_transitions"] == 2 * 6 * 2 * 5 + 4 * 256 * 48
    assert summary["max_state_error"] < 1e-10
    receipt = json.loads((audit / "receipt.json").read_text())
    assert receipt["new_policy_calls"] == 0
    for panel in p["panels"]:
        assert set(summary["panels"][panel]) == set(q.ARMS)
        assert summary["panels"][panel]["known_state_physics"]["paired_difference_interval"] == [0., 0.]
    with pytest.raises(FileExistsError):
        q.run(path, execution, expected_plan_sha256=q.sha(path))
    (execution / "timing.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Changed execution"):
        q.audit(path, execution, tmp_path / "bad-audit", expected_plan_sha256=q.sha(path))


def test_changed_runtime_rejected(plan):
    _, p = plan
    p["runtime"]["numpy"] = "wrong"
    with pytest.raises(ValueError, match="runtime"):
        q.validate(p)


def test_changed_frozen_source_rejected(plan):
    _, p = plan
    p["sources"]["scripts/qualify_robotics_pendulum.py"] = "wrong"
    with pytest.raises(ValueError, match="Frozen source"):
        q.validate(p)


def test_empty_source_membership_rejected(plan):
    _, p = plan
    p["sources"] = {}
    with pytest.raises(ValueError, match="membership"):
        q.validate(p)


def test_wrong_external_plan_digest_rejected(plan, tmp_path):
    path, _ = plan
    with pytest.raises(ValueError, match="external frozen"):
        q.run(path, tmp_path / "never-created", expected_plan_sha256="wrong")
    assert not (tmp_path / "never-created").exists()


def test_nonfinite_replay_rejected(plan):
    _, p = plan
    a, _ = q.rollout(p, "stationary", "uniform", deadline=time.monotonic() + 10)
    a["command"][0, 2] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        q.native_replay(a["state"], a["command"], a["gain"], p["config"])
