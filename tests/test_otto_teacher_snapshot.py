"""Fabricated snapshot/filter cases only; no native environment or saved inputs."""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

from openjev.research.otto_reference_control import SpaceAwareActor
from openjev.research.otto_released_policy import PublicBeliefView, ReleasedPolicyActor
from openjev.research.otto_teacher_snapshot import NORMALIZATION_ATOL, TeacherSnapshot


def packet(position=(10, 10), *, step=119, hit=3, done=False):
    return {"position": list(position), "step": step, "hit": hit, "done": done,
            "valid_actions": [] if done else [a for a in range(4)
                if 0 <= position[a//2]+2*(a % 2)-1 < 53]}


def kernel():
    x, y = np.indices((107, 107), dtype=np.float64)
    result = np.stack(((x+1)/432, (y+1)/432, np.full_like(x, .25), .75-(x+y+2)/432))
    result[:, 53, 53] = 0
    return result


def belief():
    result = np.zeros((53, 53), dtype=np.float64)
    result[9, 10], result[8, 10], result[12, 11] = .2, .3, .5
    return result


def move(position, action):
    result = list(position)
    result[action//2] = max(0, min(52, result[action//2]+2*(action % 2)-1))
    return tuple(result)


def test_anchor_copied_without_reset_or_last_hit_reassimilation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("constructor must not reset or assimilate")

    monkeypatch.setattr(PublicBeliefView, "_reset", forbidden)
    monkeypatch.setattr(PublicBeliefView, "_observe", forbidden)
    original, public = belief(), packet()
    actor = TeacherSnapshot(public, original, kernel())
    np.testing.assert_array_equal(actor.belief, original)
    assert actor.public == {**public, "position": (10, 10), "valid_actions": (0, 1, 2, 3)}
    assert actor.public["step"] == 119 and actor.public["hit"] == 3
    assert TeacherSnapshot.choose is SpaceAwareActor.choose
    assert TeacherSnapshot.update is ReleasedPolicyActor.update
    assert TeacherSnapshot.reset is ReleasedPolicyActor.reset


def test_roundoff_tolerance_preserves_accepted_mass_without_rescaling():
    original = np.zeros((53, 53), dtype=np.float64)
    original[9, 10] = 1+NORMALIZATION_ATOL/4
    actor = TeacherSnapshot(packet(), original, kernel())
    assert actor.belief.tobytes() == original.tobytes()
    assert float(actor.belief.sum()) > 1


def test_zero_hit_and_arbitrary_step_zero_do_not_invoke_initial_prior_rules():
    actor = TeacherSnapshot(packet((3, 4), step=0, hit=0), belief(), kernel())
    assert actor.public["position"] == (3, 4) and actor.public["step"] == 0
    np.testing.assert_array_equal(actor.belief, belief())


def test_named_public_packet_uses_the_same_strict_interface():
    observation = namedtuple("Observation", "position step hit done valid_actions")
    actor = TeacherSnapshot(observation(**packet(step=250)), belief(), kernel())
    assert actor.public["step"] == 250
    np.testing.assert_array_equal(actor.belief, belief())


def test_caller_arrays_and_packet_cannot_mutate_the_owned_snapshot():
    original, public, k = belief(), packet(), kernel()
    actor = TeacherSnapshot(public, original, k)
    expected = actor.belief
    expected_kernel = k.copy()
    original[:] = 0
    k[:] = 0
    public["position"][0] = 0
    public["valid_actions"].clear()
    actor.belief[:] = 7
    exposed = actor.public
    exposed["position"], exposed["step"] = (0, 0), 0
    np.testing.assert_array_equal(actor.belief, expected)
    np.testing.assert_array_equal(actor._view.p_Poisson, expected_kernel)
    assert actor.public["position"] == (10, 10) and actor.public["step"] == 119
    assert actor.allowed_actions == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        actor._view.p_Poisson.flags.writeable = True
    assert not hasattr(actor, "__dict__")
    assert all(not hasattr(actor, name) for name in ("source", "seed", "stream", "history", "model", "env"))
    assert actor.storage_bytes()["mutable_array_bytes"] == 53*53*8
    assert actor.storage_bytes()["immutable_array_bytes"] == 5*107*107*8


@pytest.mark.parametrize("position", [(0, 0), (52, 52), (0, 26), (52, 26)])
def test_boundary_choices_use_only_inbounds_actions(position):
    public = packet(position, step=41, hit=0)
    expected_action = public["valid_actions"][0]
    original = np.zeros((53, 53), dtype=np.float64)
    original[move(position, expected_action)] = 1
    actor = TeacherSnapshot(public, original, kernel())
    action, costs = actor.choose()
    assert action == expected_action and action in actor.allowed_actions
    assert costs.dtype == np.float64 and costs.shape == (4,)
    assert costs[action] == -1e-10
    for direction in range(4):
        assert np.isfinite(costs[direction]) if direction in actor.allowed_actions else np.isposinf(costs[direction])


def test_forced_first_action_updates_the_already_assimilated_belief_once():
    actor = TeacherSnapshot(packet(step=80), belief(), kernel())
    actor.update(0, packet((9, 10), step=81, hit=1))
    expected = np.zeros((53, 53), dtype=np.float64)
    # The visited .2 disappears; hit1 has y-likelihoods54/432 and55/432.
    expected[8, 10], expected[12, 11] = 162/437, 275/437
    np.testing.assert_allclose(actor.belief, expected, rtol=1e-15, atol=0)
    assert actor.public["step"] == 81 and actor.public["position"] == (9, 10)
    assert actor.choose()[0] in actor.allowed_actions


def test_pending_choice_and_invalid_update_preserve_prior_state():
    actor = TeacherSnapshot(packet(), belief(), kernel())
    action, _ = actor.choose()
    before, public = actor.belief, actor.public
    with pytest.raises(RuntimeError, match="awaiting"):
        actor.choose()
    wrong = (action+1) % 4
    with pytest.raises(ValueError, match="pending"):
        actor.update(wrong, packet(move(public["position"], wrong), step=120, hit=0))
    np.testing.assert_array_equal(actor.belief, before)
    assert actor.public == public
    with pytest.raises(ValueError, match="expected step"):
        actor.update(action, packet(move(public["position"], action), step=121, hit=0))
    np.testing.assert_array_equal(actor.belief, before)
    assert actor.public == public


def test_found_sentinel_produces_point_mass_then_blocks_further_work():
    actor = TeacherSnapshot(packet(step=217), belief(), kernel())
    actor.update(0, packet((9, 10), step=218, hit=-2, done=True))
    expected = np.zeros((53, 53), dtype=np.float64)
    expected[9, 10] = 1
    np.testing.assert_array_equal(actor.belief, expected)
    assert actor.public["done"] is True and actor.allowed_actions == ()
    with pytest.raises(RuntimeError, match="found"):
        actor.choose()
    with pytest.raises(RuntimeError, match="found"):
        actor.update(0, packet((8, 10), step=219, hit=0))


def test_final_unsuccessful_horizon_packet_is_assimilated_without_fabricated_terminal():
    actor = TeacherSnapshot(packet(step=2187), belief(), kernel())
    actor.update(0, packet((9, 10), step=2188, hit=1))
    assert actor.public["step"] == 2188 and actor.public["done"] is False
    assert actor.belief[9, 10] == 0
    assert actor.belief[8, 10] == pytest.approx(162/437)
    # The caller, not this inherited actor, enforces its continuation horizon.
    assert actor.choose()[0] in actor.allowed_actions


def test_inherited_reset_starts_new_center_prior_instead_of_restoring_anchor():
    k = kernel()
    actor = TeacherSnapshot(packet(step=71), belief(), k)
    initial = packet((26, 26), step=0, hit=2)
    actor.reset(initial)
    reference = SpaceAwareActor(initial, k, allow_stay=False)
    assert actor.public == reference.public
    np.testing.assert_array_equal(actor.belief, reference.belief)
    assert actor.allowed_actions == reference.allowed_actions
    with pytest.raises(ValueError):
        actor.reset(packet(step=71))
    np.testing.assert_array_equal(actor.belief, reference.belief)


@pytest.mark.parametrize("defect", ["float32", "shape", "nan", "infinity", "negative", "zero", "half", "subfloor", "current_cell"])
def test_unsupported_beliefs_are_rejected_without_repairs(defect):
    original = belief()
    if defect == "float32":
        original = original.astype(np.float32)
    elif defect == "shape":
        original = original[:-1]
    elif defect in {"nan", "infinity", "negative"}:
        original[9, 10] = {"nan": np.nan, "infinity": np.inf, "negative": -.2}[defect]
    elif defect in {"zero", "half", "subfloor"}:
        original *= {"zero": 0, "half": .5, "subfloor": 1e-12}[defect]
    else:
        original[10, 10] = 1e-300
    with pytest.raises(ValueError):
        TeacherSnapshot(packet(), original, kernel())


@pytest.mark.parametrize("defect", ["terminal", "negative_step", "boolean_step", "outside", "wrong_actions", "sentinel", "hidden_field"])
def test_invalid_public_anchor_packets_are_rejected(defect):
    public = packet()
    if defect == "terminal":
        public = packet(done=True, hit=-2)
    elif defect == "negative_step":
        public["step"] = -1
    elif defect == "boolean_step":
        public["step"] = True
    elif defect == "outside":
        public["position"] = [53, 0]
    elif defect == "wrong_actions":
        public["valid_actions"] = [3, 2, 1, 0]
    elif defect == "sentinel":
        public["hit"] = -2
    else:
        public["source"] = [9, 10]
    with pytest.raises((ValueError, TypeError)):
        TeacherSnapshot(public, belief(), kernel())


def test_kernel_validation_and_input_surface_remain_restricted():
    wrong_origin = kernel()
    wrong_origin[0, 53, 53] = .1
    for invalid in (kernel().astype(np.float32), wrong_origin):
        with pytest.raises(ValueError, match="kernel"):
            TeacherSnapshot(packet(), belief(), invalid)
    with pytest.raises(TypeError):
        TeacherSnapshot(packet(), belief(), kernel(), source=(9, 10))
