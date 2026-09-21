"""Synthetic costs and public packets only; no TF, native OTTO or checkpoints."""
from __future__ import annotations

import numpy as np
import pytest

from openjev.research.otto_released_policy import ReleasedPolicyActor
from openjev.research.otto_restricted_policy import RestrictedPolicyActor, select_inbounds_action


def kernel():
    result = np.full((4, 107, 107), .25, dtype=np.float64)
    result[:, 53, 53] = 0
    return result


def packet(step=0, position=(26, 26), hit=1, done=False):
    return {"position": list(position), "hit": hit, "done": done, "step": step,
            "valid_actions": [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


class FakePolicy:
    def __init__(self, *, env, model, sym_avg):
        assert not any(hasattr(env, n) for n in ("source", "seed", "draw_log", "env", "native"))
        self.env, self.model, self.sym_avg = env, model, sym_avg
        self.calls = 0

    def _value_policy(self):
        self.calls += 1
        scores = self.model["scores"]
        if "action" in self.model:
            action = self.model["action"]
        else:
            action = int(np.flatnonzero(np.abs(scores - scores.min()) < 1e-10)[0])
        return action, scores


def actor(scores=(0., 1., 2., 3.), cls=RestrictedPolicyActor, **kwargs):
    model = {"scores": np.asarray(scores, dtype=np.float32)}
    return cls(packet(), kernel(), model, FakePolicy, **kwargs)


def advance(state, action, *, hit=0, done=False):
    old = state.public
    position = list(old["position"])
    axis = action // 2
    position[axis] = max(0, min(52, position[axis] + 2 * (action % 2) - 1))
    state.update(action, packet(old["step"] + 1, position, -2 if done else hit, done))


def reach(state, position):
    for axis in range(2):
        while state.public["position"][axis] != position[axis]:
            action = 2 * axis + int(position[axis] > state.public["position"][axis])
            advance(state, action)


@pytest.mark.parametrize("scores", [(1., 1., 1., 1.), (4., 3., 2., 1.),
                                  (5e-11, 0., 2e-10, 3e-10), (-4., -3., -2., -1.)])
def test_interior_is_bitwise_original_choice_and_all_four_raw_scores(scores):
    original, restricted = actor(scores, ReleasedPolicyActor), actor(scores)
    a, x = original.choose()
    b, y = restricted.choose()
    assert a == b and x.dtype == y.dtype == np.float32 and x.tobytes() == y.tobytes()
    assert restricted.allowed_actions == (0, 1, 2, 3)
    assert restricted.selection_mask == (True, True, True, True)
    assert original._policy.calls == restricted._policy.calls == 1


@pytest.mark.parametrize("position", [(0, 26), (52, 26), (26, 0), (26, 52),
                                    (0, 0), (0, 52), (52, 0), (52, 52)])
def test_every_boundary_restricts_selection_only_and_keeps_raw_scores(position):
    state = actor()
    reach(state, position)
    allowed = state.public["valid_actions"]
    blocked = next(a for a in range(4) if a not in allowed)
    raw = np.asarray([4., 3., 2., 1.], dtype=np.float32)
    raw[blocked] = -10.
    state._policy.model["scores"] = raw
    expected = min(allowed, key=lambda a: (float(raw[a]), a))
    old = state.belief
    chosen, returned = state.choose()
    assert chosen == expected and chosen != blocked
    assert state.allowed_actions == allowed
    assert state.selection_mask == tuple(a in allowed for a in range(4))
    assert returned.tobytes() == raw.tobytes() and np.isfinite(returned).all()
    assert state._policy.calls == 1 and state._pending_action == chosen
    np.testing.assert_array_equal(state.belief, old)
    with pytest.raises(ValueError, match="pending"):
        advance(state, blocked)
    assert state._pending_action == chosen and state._policy.calls == 1
    advance(state, chosen)
    assert state._pending_action is None and state.public["position"] != position


def test_restricted_tie_uses_first_action_within_float32_epsilon_not_exact_argmin():
    scores = np.asarray([-1., 5e-11, -2., 0.], dtype=np.float32)
    assert select_inbounds_action(scores, (1, 3)) == 1
    scores[1] = np.float32(1e-10)
    assert select_inbounds_action(scores, (1, 3)) == 3  # Strict epsilon threshold.
    scores[1] = np.nextafter(np.float32(1e-10), np.float32(0))
    assert select_inbounds_action(scores, (1, 3)) == 1


def test_same_public_history_keeps_original_belief_exact_including_prescribed_stay():
    original, restricted = actor(cls=ReleasedPolicyActor), actor()
    for state in (original, restricted):
        reach(state, (0, 0))
    for action, hit in ((0, 3), (2, 2), (1, 0), (3, 1)):
        advance(original, action, hit=hit)
        advance(restricted, action, hit=hit)
        assert original.public == restricted.public
        assert original.belief.tobytes() == restricted.belief.tobytes()
    assert restricted._policy.calls == 0
    assert restricted.storage_bytes() == original.storage_bytes()


def test_choose_pending_reset_found_and_censored_update_lifecycle():
    state = actor()
    chosen, _ = state.choose()
    with pytest.raises(RuntimeError, match="awaiting"):
        state.choose()
    assert state._policy.calls == 1
    previous = state.belief
    advance(state, chosen)  # Caller may stop after this final censored update.
    assert not state.public["done"] and not np.array_equal(previous, state.belief)
    state.choose()
    state.reset(packet(hit=3))  # Reset clears pending and retains fixed ownership.
    assert state.public["step"] == 0 and state._pending_action is None
    chosen, _ = state.choose()
    advance(state, chosen, done=True)
    assert state.allowed_actions == () and state.selection_mask == (False,) * 4
    assert np.count_nonzero(state.belief) == 1
    calls = state._policy.calls
    with pytest.raises(RuntimeError, match="found"):
        state.choose()
    with pytest.raises(RuntimeError, match="found"):
        advance(state, chosen)
    assert state._policy.calls == calls


def test_returned_costs_and_masks_cannot_change_actor_or_injected_scores():
    state = actor()
    raw = state._policy.model["scores"].copy()
    chosen, returned = state.choose()
    returned[:] = 100.
    np.testing.assert_array_equal(state._policy.model["scores"], raw)
    assert state._pending_action == chosen
    mask, allowed = state.selection_mask, state.allowed_actions
    assert isinstance(mask, tuple) and isinstance(allowed, tuple)
    with pytest.raises(AttributeError):
        state.allowed_actions = (3,)
    assert not hasattr(state, "__dict__") and not hasattr(state, "history")


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "inf", "list", "action_tie", "bool_action"])
def test_invalid_official_result_rejected_without_committing_action(bad):
    state = actor((0., 0., 2., 3.))
    model = state._policy.model
    model["action"] = 0
    if bad == "dtype":
        model["scores"] = model["scores"].astype(np.float64)
    elif bad == "shape":
        model["scores"] = model["scores"][:3]
    elif bad in ("nan", "inf"):
        model["scores"][3] = np.nan if bad == "nan" else np.inf
    elif bad == "list":
        model["scores"] = model["scores"].tolist()
    elif bad == "action_tie":
        model["action"] = 1
    else:
        model["action"] = True
    before = state.belief
    with pytest.raises((ValueError, TypeError)):
        state.choose()
    assert state._pending_action is None and state._policy.calls == 1
    np.testing.assert_array_equal(state.belief, before)


@pytest.mark.parametrize("allowed", [(), (1, 1), (3, 1), (-1, 2), (0, 4), (True, 2), {1, 3}])
def test_pure_saved_cost_selector_rejects_bad_selection_contract(allowed):
    with pytest.raises((ValueError, TypeError)):
        select_inbounds_action(np.arange(4, dtype=np.float32), allowed)


def test_original_policy_failure_is_not_retried_or_replaced():
    class FailingPolicy(FakePolicy):
        def _value_policy(self):
            self.calls += 1
            raise RuntimeError("recorded injected failure")

    state = RestrictedPolicyActor(packet(), kernel(), None, FailingPolicy)
    with pytest.raises(RuntimeError, match="recorded injected failure"):
        state.choose()
    assert state._policy.calls == 1 and state._pending_action is None


def test_symmetry_argument_and_exact_public_packet_validation_are_inherited():
    state = actor(sym_avg=False)
    assert state._policy.sym_avg is False
    with pytest.raises(TypeError):
        actor(sym_avg=1)
    hidden = packet()
    hidden["source"] = [1, 2]
    with pytest.raises(ValueError, match="exact public"):
        RestrictedPolicyActor(hidden, kernel(), None, FakePolicy)
    assert RestrictedPolicyActor.update is ReleasedPolicyActor.update
    assert RestrictedPolicyActor.reset is ReleasedPolicyActor.reset
    assert RestrictedPolicyActor.storage_bytes is ReleasedPolicyActor.storage_bytes
