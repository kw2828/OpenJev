"""Controller information and action/observation ordering, engineering only."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium.envs.mujoco.reacher_v5")

from openjev.research import reacher_tracking_policy as policy
from openjev.research.reacher_tracking_dynamics import nominal_model
from openjev.research.robotics_reacher import packet


class FakeBank:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def plan(self, qpos, qvel, target, gain, inputs, *, step, deadline):
        self.calls.append({"qpos": qpos.copy(), "qvel": qvel.copy(), "target": target.copy(),
                           "gain": gain, "step": step})
        return SimpleNamespace(selected_actions=np.array([[.01, -.02]], np.float32)), {}, {}

    def snapshot(self, **kwargs):
        return {"calls": len(self.calls)}


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(policy, "GainPlanningBank", FakeBank)
    model = nominal_model()
    def constructor(arm="nominal", **kwargs):
        settings = {"steps": 4, "gain_grid": np.array([.5, 1., 1.5]), "window": 2,
                    "freeze_after": 2, "noise_std": .05, "planning_horizon": 2, "action_block": 1}
        settings.update(kwargs)
        return policy.TrackingPolicy(model, packet([.1, -.1], [.12, -.04]), arm=arm, **settings)
    return constructor


@pytest.mark.parametrize("arm", ["nominal", "adaptive", "frozen", "zero"])
def test_public_roles_reject_privileged_inputs_before_planning(make, arm):
    controller = make(arm)
    with pytest.raises(ValueError, match="current_gain"):
        controller.decide(None, current_gain=1.)
    with pytest.raises(ValueError, match="privileged state"):
        controller.decide(None, true_state=(np.zeros(4), np.zeros(4)))
    assert controller.snapshot()["costs"]["decisions_completed"] == 0


def test_current_parameter_reference_keeps_public_state(make):
    controller = make("public_gain")
    with pytest.raises(ValueError, match="current_gain"):
        controller.decide(None)
    _, trace = controller.decide(None, current_gain=1.5)
    np.testing.assert_array_equal(trace["root_qpos"], trace["public_qpos"])
    np.testing.assert_array_equal(trace["root_qvel"], trace["public_qvel"])
    assert trace["planning_gain"] == 1.5


def test_true_state_role_preserves_public_target_precision(make):
    controller = make("true_state")
    value = (np.array([.2, .3, .12, -.04]), np.array([.5, -.2, 0., 0.]))
    _, trace = controller.decide(None, current_gain=.5, true_state=value)
    np.testing.assert_array_equal(trace["root_qpos"][:2], value[0][:2])
    np.testing.assert_array_equal(trace["root_qvel"], value[1])
    np.testing.assert_array_equal(trace["root_qpos"][2:], trace["public_packet"][4:6])
    np.testing.assert_array_equal(trace["public_qvel"], np.zeros(4))


def test_true_state_cannot_change_goal_or_supply_target_velocity(make):
    controller = make("true_state")
    with pytest.raises(ValueError, match="public target"):
        controller.decide(None, current_gain=1., true_state=(np.zeros(4), np.zeros(4)))
    with pytest.raises(ValueError, match="public target"):
        controller.decide(None, current_gain=1.,
            true_state=(np.array([.1, -.1, .12, -.04]), np.ones(4)))


def test_exactly_one_real_observation_per_selected_action(make):
    controller = make()
    with pytest.raises(RuntimeError, match="pending"):
        controller.observe([0, 0], packet([.1, -.1], [.12, -.04]))
    action, _ = controller.decide(None)
    with pytest.raises(RuntimeError, match="awaiting"):
        controller.decide(None)
    with pytest.raises(ValueError, match="differs"):
        controller.observe([0, 0], packet([.1, -.1], [.12, -.04]))
    assert controller.snapshot()["step"] == 0
    controller.observe(action, packet([.11, -.12], [.12, -.04]))
    assert controller.snapshot()["step"] == 1
    with pytest.raises(RuntimeError, match="pending"):
        controller.observe(action, packet([.11, -.12], [.12, -.04]))


def test_frozen_identifier_stops_but_public_observer_continues(make):
    controller = make("frozen")
    frozen_gain = None
    for step in range(4):
        action, trace = controller.decide(None)
        observation = controller.observe(action, packet([.1+(step+1)*.01, -.1], [.12, -.04]))
        assert observation["identifier_updated"] == (step < 2)
        if step == 1:
            frozen_gain = observation["identified_gain"]
        elif step > 1:
            assert observation["identified_gain"] == frozen_gain
            assert trace["planning_gain"] == frozen_gain
    snapshot = controller.snapshot()
    assert snapshot["observer"]["step_index"] == 4
    assert snapshot["identifier"]["step_index"] == 2
    assert snapshot["costs"]["identifier_updates_completed"] == 2
    with pytest.raises(RuntimeError, match="terminal"):
        controller.decide(None)


def test_adaptive_identifier_and_common_observer_remain_identical(make):
    controller = make("adaptive")
    for step in range(4):
        action, _ = controller.decide(None)
        controller.observe(action, packet([.1+step*.01, -.1-step*.01], [-.12, .03]))
    snapshot = controller.snapshot()
    np.testing.assert_array_equal(snapshot["observer"]["qpos_estimate"],
                                  snapshot["identifier"]["observer"]["qpos_estimate"])
    np.testing.assert_array_equal(snapshot["observer"]["qvel_estimate"],
                                  snapshot["identifier"]["observer"]["qvel_estimate"])
    assert snapshot["costs"]["identifier_updates_completed"] == 4


def test_planning_does_not_advance_or_mutate_public_observation(make):
    controller = make("adaptive")
    before = controller.snapshot()
    action, trace = controller.decide(None)
    trace["public_packet"][:] = 0
    action[:] = 0
    after = controller.snapshot()
    np.testing.assert_array_equal(before["observer"]["qpos_estimate"], after["observer"]["qpos_estimate"])
    assert after["identifier"]["step_index"] == 0
    np.testing.assert_array_equal(after["pending_command"], np.array([.01, -.02], np.float32))


def test_missing_observations_rejected_before_any_update(make):
    controller = make("adaptive")
    action, _ = controller.decide(None)
    with pytest.raises(ValueError, match="full visible"):
        controller.observe(action, packet([0, 0], [.12, -.04], valid=False, age_seconds=.02))
    assert controller.snapshot()["observer"]["step_index"] == 0
    assert controller.snapshot()["identifier"]["step_index"] == 0


def test_planning_failure_is_terminal_across_later_calls(make, monkeypatch):
    controller = make()
    def fail(*args, **kwargs):
        raise TimeoutError("engineering failure")
    monkeypatch.setattr(controller._bank, "plan", fail)
    with pytest.raises(TimeoutError):
        controller.decide(None)
    assert controller.snapshot()["failed"]
    with pytest.raises(RuntimeError):
        controller.decide(None)


def test_zero_role_does_not_construct_or_call_a_planner(make):
    controller = make("zero")
    action, trace = controller.decide(None)
    np.testing.assert_array_equal(action, np.zeros(2))
    assert trace["planning_gain"] == 1.
    assert trace["search"] is trace["journal"] is trace["bank_snapshot"] is None
    assert controller.snapshot()["planner"] is None
