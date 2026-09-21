"""Synthetic public histories and injected fake policy only; no OTTO or model."""
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np
import pytest

from openjev.research import otto_released_policy as M


def kernel():
    x, y = np.indices((107, 107), dtype=np.float64)
    result = np.stack(((x + 1) / 432, (y + 1) / 432, np.full_like(x, .25),
                       .75 - (x + y + 2) / 432))
    result[:, 53, 53] = 0
    return result


def packet(step=0, position=(26, 26), hit=1, done=False):
    return {"position": list(position), "hit": hit, "done": done, "step": step,
            "valid_actions": [] if done else [a for a in range(4)
                                               if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


class FakePolicy:
    def __init__(self, *, env, model, sym_avg):
        assert not any(hasattr(env, name) for name in ("source", "seed", "draw_log", "env", "hit_map"))
        self.env, self.model, self.sym_avg = env, model, sym_avg
        self.calls = 0

    def _value_policy(self):
        self.calls += 1
        scores = np.ones(4, dtype=np.float32)
        scores[self.model] = 0
        return self.model, scores


def actor(hit=1, action=0, *, k=None):
    return M.ReleasedPolicyActor(packet(hit=hit), kernel() if k is None else k, action, FakePolicy)


def advance(state, action, *, hit=0, done=False):
    public = state.public
    pos = list(public["position"])
    axis, delta = action // 2, 2 * (action % 2) - 1
    pos[axis] = max(0, min(52, pos[axis] + delta))
    state.update(action, packet(public["step"] + 1, pos, -2 if done else hit, done))


def oracle_update(p, k, pos, hit):
    # Scalar public likelihood product and fsum, independent of the view's crop.
    result = np.zeros((53, 53), dtype=np.float64)
    for i in range(53):
        for j in range(53):
            if (i, j) != pos:
                result[i, j] = p[i, j] * k[hit, 53 + i - pos[0], 53 + j - pos[1]]
    total = math.fsum(result.flat)
    if total > 1e-10:
        result /= total
    return result


@pytest.mark.parametrize("hit", [1, 2, 3])
def test_initial_hit_prior_and_sequential_public_history_match_scalar_oracle(hit):
    k = kernel()
    state = actor(hit, k=k)
    p = np.full((53, 53), 1 / 2808)
    p[26, 26] = 0
    p = oracle_update(p, k, (26, 26), hit)
    np.testing.assert_allclose(state.belief, p, rtol=2e-14, atol=1e-18)
    for action, h in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        advance(state, action, hit=h)
        p = oracle_update(p, k, state.public["position"], h)
        np.testing.assert_allclose(state.belief, p, rtol=2e-14, atol=1e-18)
    assert state.public["step"] == 4
    assert state.belief[26, 26] == 0  # Returning never restores excluded support.


@pytest.mark.parametrize("mass", [0., .5e-10, 1e-10, 2e-10])
def test_exact_upstream_normalization_threshold_including_equality_and_zero(mass):
    k = np.zeros((4, 107, 107), dtype=np.float64)
    k[1, 30, 32] = 1  # Reset posterior is certain at public-grid cell3,5.
    k[2, 31, 32] = mass  # At position25,26 after action0.
    state = actor(k=k)
    assert state.belief[3, 5] == 1
    advance(state, 0, hit=2)
    assert state.belief[3, 5] == (1 if mass > 1e-10 else mass)
    assert np.count_nonzero(state.belief) == int(mass != 0)


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_every_blocked_direction_is_a_valid_policy_action_and_new_observation(action):
    state = actor(action=action)
    for _ in range(26):
        advance(state, action, hit=0)
    assert action not in state.public["valid_actions"]
    before, oldpos = state.belief, state.public["position"]
    chosen, scores = state.choose()
    assert chosen == action and scores.shape == (4,)
    advance(state, action, hit=3)
    assert state.public["position"] == oldpos and state.public["step"] == 27
    assert not np.array_equal(before, state.belief)
    assert state.belief[oldpos] == 0


def test_found_sentinel_becomes_point_mass_without_likelihood_indexing_and_requires_reset():
    state = actor(action=1)
    assert state.choose()[0] == 1
    advance(state, 1, done=True)
    assert state.public["hit"] == -2 and state.public["done"] is True
    assert state.belief[27, 26] == 1 and np.count_nonzero(state.belief) == 1
    with pytest.raises(RuntimeError, match="found"):
        state.choose()
    with pytest.raises(RuntimeError, match="found"):
        state.update(1, packet(2, (28, 26), 0))
    state.reset(packet(hit=3))
    assert state.public == {**packet(hit=3), "position": (26, 26), "valid_actions": (0, 1, 2, 3)}
    assert np.count_nonzero(state.belief) > 1
    assert state.choose()[0] == 1


def test_censoring_final_nonterminal_update_is_incorporated_without_fabricated_done():
    state = actor()
    before = state.belief
    chosen, _ = state.choose()
    advance(state, chosen, hit=0)  # Caller stops at its declared horizon1.
    assert state.public["step"] == 1 and state.public["done"] is False
    assert not np.array_equal(before, state.belief)


def test_pending_choice_prevents_duplicate_inference_and_wrong_action_without_mutation():
    state = actor(action=2)
    state.choose()
    before = state.belief
    with pytest.raises(RuntimeError, match="awaiting"):
        state.choose()
    with pytest.raises(ValueError, match="pending"):
        state.update(3, packet(1, (26, 27), 0))
    np.testing.assert_array_equal(state.belief, before)
    advance(state, 2, hit=0)
    assert state.choose()[0] == 2


def test_input_and_output_ownership_and_fixed_storage():
    k, initial = kernel(), packet()
    state = M.ReleasedPolicyActor(initial, k, 0, FakePolicy)
    expected = state.belief
    initial["position"][0] = 0
    initial["valid_actions"].clear()
    k[:] = 0
    snapshot = state.belief
    snapshot[:] = 100
    public = state.public
    public["hit"] = 3
    np.testing.assert_array_equal(state.belief, expected)
    assert state.public["position"] == (26, 26) and state.public["hit"] == 1
    with pytest.raises(ValueError):
        state._view.p_Poisson.flags.writeable = True
    state._view.agent[0] = 0
    state._view.p_source[:] = 0
    np.testing.assert_array_equal(state.belief, expected)
    storage = state.storage_bytes()
    assert storage["mutable_array_bytes"] == 53 * 53 * 8
    assert storage["immutable_array_bytes"] == 4 * 107 * 107 * 8
    for _ in range(40):
        advance(state, 0, hit=0)
    assert state.storage_bytes() == storage
    assert not hasattr(state, "history") and not hasattr(state._view, "history")


def test_public_namedtuple_is_supported_without_other_fields():
    Public = namedtuple("Public", "position hit done step valid_actions")
    state = M.ReleasedPolicyActor(Public(**packet()), kernel(), 0, FakePolicy)
    np.testing.assert_array_equal(state.belief, actor().belief)


@pytest.mark.parametrize("bad", ["extra", "missing", "step", "position", "zero_hit", "terminal"])
def test_reset_contract_rejects_hidden_or_noninitial_packet_without_mutation(bad):
    state = actor()
    before = state.belief
    p = packet()
    if bad == "extra":
        p["source"] = (5, 5)
    elif bad == "missing":
        p.pop("hit")
    elif bad == "step":
        p["step"] = 1
    elif bad == "position":
        p["position"] = [25, 26]
    elif bad == "zero_hit":
        p["hit"] = 0
    else:
        p = packet(0, hit=-2, done=True)
    with pytest.raises(ValueError):
        state.reset(p)
    np.testing.assert_array_equal(state.belief, before)


@pytest.mark.parametrize("bad", ["skip", "repeat", "diagonal", "wrong_direction", "moves", "source",
                                 "negative_hit", "false_done", "wrong_terminal", "bool_action", "big_action"])
def test_invalid_transition_is_atomic(bad):
    state = actor()
    before, public = state.belief, state.public
    p, action = packet(1, (25, 26), 0), 0
    if bad == "skip":
        p["step"] = 2
    elif bad == "repeat":
        p["step"] = 0
    elif bad == "diagonal":
        p["position"] = [25, 25]
    elif bad == "wrong_direction":
        p["position"] = [27, 26]
    elif bad == "moves":
        p["valid_actions"] = [0, 1, 2]
    elif bad == "source":
        p["source"] = (25, 26)
    elif bad == "negative_hit":
        p["hit"] = -1
    elif bad == "false_done":
        p["hit"] = -2
    elif bad == "wrong_terminal":
        p["done"], p["valid_actions"] = True, []
    elif bad == "bool_action":
        action = True
    else:
        action = 4
    with pytest.raises((ValueError, TypeError)):
        state.update(action, p)
    np.testing.assert_array_equal(state.belief, before)
    assert state.public == public


@pytest.mark.parametrize("field,value", [("step", True), ("hit", True), ("done", 1),
                                         ("position", [True, 26]), ("valid_actions", [False, 1, 2, 3])])
def test_public_boolean_integer_ambiguities_rejected(field, value):
    p = packet()
    p[field] = value
    with pytest.raises((TypeError, ValueError)):
        M.ReleasedPolicyActor(p, kernel(), 0, FakePolicy)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "negative", "above_one", "origin"])
def test_invalid_kernel_fails_before_policy_constructor(bad):
    k = kernel()
    if bad == "shape":
        k = k[:, :-1]
    elif bad == "dtype":
        k = k.astype(np.float32)
    else:
        k[0, 53 if bad == "origin" else 0, 53 if bad == "origin" else 0] = {
            "nan": np.nan, "negative": -.01, "above_one": 1.01, "origin": .1}[bad]

    def forbidden(**kwargs):
        raise AssertionError("invalid input must fail before policy construction")

    with pytest.raises(ValueError):
        M.ReleasedPolicyActor(packet(), k, None, forbidden)


@pytest.mark.parametrize("position", [(0, 0), (52, 52), (0, 26), (26, 26)])
def test_view_crop_and_centering_match_independent_index_formula(position):
    view = M.PublicBeliefView(kernel())
    for side, center in [(107, 53), (105, 52)]:
        x = np.arange(2 * side * side, dtype=np.float64).reshape(2, side, side)
        got = view._extract_N_from_2N(x, position)
        for i, j in [(0, 0), (52, 52), (13, 29)]:
            np.testing.assert_array_equal(got[:, i, j], x[:, center + i - position[0], center + j - position[1]])
    grid = np.zeros((53, 53), dtype=np.float64)
    grid[5, 13] = .5e-10
    centered = view._centeragent(grid, position)
    assert centered.dtype == np.float64
    assert centered[52 + 5 - position[0], 52 + 13 - position[1]] == .5e-10
    assert centered.sum() == .5e-10
    assert view._centeragent(np.zeros_like(grid), position).sum() == 0


@pytest.mark.parametrize("bad", ["not_first_tie", "nan", "shape", "dtype"])
def test_policy_result_contract_and_no_silent_mask_or_repair(bad):
    class InvalidPolicy(FakePolicy):
        def _value_policy(self):
            scores = np.zeros(4, dtype=np.float32)
            action = 1 if bad == "not_first_tie" else 0
            if bad == "nan":
                scores[0] = np.nan
            elif bad == "shape":
                scores = scores[:3]
            elif bad == "dtype":
                scores = scores.astype(np.float64)
            return action, scores

    state = M.ReleasedPolicyActor(packet(), kernel(), None, InvalidPolicy)
    with pytest.raises(ValueError):
        state.choose()
    assert state._pending_action is None and state.public["step"] == 0


def test_symmetry_flag_reaches_injected_policy_and_is_boolean():
    state = M.ReleasedPolicyActor(packet(), kernel(), 0, FakePolicy, sym_avg=False)
    assert state._policy.sym_avg is False
    with pytest.raises(TypeError):
        M.ReleasedPolicyActor(packet(), kernel(), 0, FakePolicy, sym_avg=1)


def test_valid_actions_rejects_unordered_set():
    p = packet()
    p["valid_actions"] = {0, 1, 2, 3}
    with pytest.raises(TypeError, match="ordered public sequence"):
        M.ReleasedPolicyActor(p, kernel(), 0, FakePolicy)
