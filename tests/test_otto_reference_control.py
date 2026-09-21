"""Pure synthetic math/public-history checks; no OTTO or trained-model calls."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openjev.research import otto_reference_control as M
from openjev.research import otto_released_policy as R


def packet(position=(26, 26), hit=1, step=0, done=False):
    return {"position": list(position), "hit": hit, "done": done, "step": step,
            "valid_actions": [] if done else [a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53]}


def moved(position, action):
    result = list(position)
    axis = action // 2
    result[axis] = max(0, min(52, result[axis] + 2 * (action % 2) - 1))
    return tuple(result)


def kernel():
    x, y = np.indices((107, 107), dtype=np.float64)
    k = np.stack(((x + 1) / 432, (y + 1) / 432,
                  np.full_like(x, .25), .75 - (x + y + 2) / 432))
    k[:, 53, 53] = 0
    return k


def advance(actor, action, hit=0, done=False):
    p = actor.public
    actor.update(action, packet(moved(p["position"], action), -2 if done else hit, p["step"] + 1, done))


def scalar_oracle(probability, position, k, allow_stay):
    """Enumerate support and branches using scalar math, not vectorized crops."""
    scores = []
    support = [(i, j, float(probability[i, j])) for i, j in zip(*np.nonzero(probability), strict=True)]
    for action in range(4):
        target = moved(position, action)
        if target == position and not allow_stay:
            scores.append(math.inf)
            continue
        end = float(probability[target])
        if end > 1 - 1e-10:
            scores.append(-1e-10)
            continue
        mass = math.fsum(p for i, j, p in support if (i, j) != target)
        surviving = [(i, j, p / mass if mass > 1e-10 else p)
                     for i, j, p in support if (i, j) != target]
        terms = []
        for hit in range(4):
            branch = [(i, j, p * float(k[hit, 53 + i - target[0], 53 + j - target[1]]))
                      for i, j, p in surviving]
            weight = math.fsum(p for _, _, p in branch)
            posterior = [(i, j, p / weight if weight > 1e-10 else p) for i, j, p in branch]
            distance = math.fsum(p * (abs(i - target[0]) + abs(j - target[1])) for i, j, p in posterior)
            entropy = math.fsum(-p * math.log2(p) for _, _, p in posterior if p > 1e-10)
            value = distance + 2 ** (entropy - 1) - .5
            terms.append((1 - end) * weight * (math.log2(value) if value > 0 else value))
        scores.append(math.fsum(terms))
    return np.asarray(scores, dtype=np.float64)


def synthetic_view(probability, position, k):
    # Evaluator-only fixtures, never an actor constructor or public input route.
    view = R.PublicBeliefView(k)
    view._probability = probability.copy()
    view._agent = tuple(position)
    return view


@pytest.mark.parametrize("position", [(26, 26), (0, 0), (52, 26), (52, 52)])
@pytest.mark.parametrize("allow_stay", [True, False])
def test_scalar_branch_oracle_matches_all_costs_including_stays(position, allow_stay):
    k = kernel()
    p = np.zeros((53, 53), dtype=np.float64)
    for point, value in [((1, 4), .2), ((29, 26), .3), ((51, 48), .5)]:
        p[point] = value
    view = synthetic_view(p, position, k)
    policy = M._SpaceAwarePolicy(env=view, model=None, sym_avg=False, allow_stay=allow_stay)
    before = view.p_source
    action, scores = policy._value_policy()
    expected = scalar_oracle(p, position, k, allow_stay)
    np.testing.assert_allclose(scores, expected, rtol=2e-14, atol=2e-14)
    assert action == int(np.flatnonzero(np.abs(expected - expected.min()) < 1e-10)[0])
    np.testing.assert_array_equal(view.p_source, before)
    assert scores.dtype == np.float64


def test_certain_adjacent_source_closed_form_and_first_tie():
    k = np.zeros((4, 107, 107), dtype=np.float64)
    k[0] = 1
    k[0, 52, 53], k[1, 52, 53] = 0, 1
    k[:, 53, 53] = 0
    actor = M.SpaceAwareActor(packet(), k)
    assert actor.belief[25, 26] == 1
    action, scores = actor.choose()
    np.testing.assert_array_equal(scores, [-1e-10, 1., 1., 1.])
    assert action == 0
    # An empty subnormalized native belief is deliberately not repaired.
    view = synthetic_view(np.zeros((53, 53)), (0, 0), k)
    for allow, expected in [(True, 0), (False, 1)]:
        policy = M._SpaceAwarePolicy(env=view, model=None, sym_avg=False, allow_stay=allow)
        action, scores = policy._value_policy()
        assert action == expected
        np.testing.assert_array_equal(scores[np.isfinite(scores)], 0)


@pytest.mark.parametrize("mass", [0., .5e-10, 1e-10, 2e-10])
def test_tiny_branch_thresholds_are_not_floored_or_unconditionally_normalized(mass):
    k = np.full((4, 107, 107), mass, dtype=np.float64)
    k[:, 53, 53] = 0
    p = np.zeros((53, 53), dtype=np.float64)
    p[40, 40] = .25
    p[41, 40] = .75
    view = synthetic_view(p, (26, 26), k)
    _, scores = M._SpaceAwarePolicy(env=view, model=None, sym_avg=False, allow_stay=True)._value_policy()
    np.testing.assert_allclose(scores, scalar_oracle(p, (26, 26), k, True), rtol=2e-14, atol=1e-24)
    if mass == 0:
        np.testing.assert_array_equal(scores, 0)


def test_entropy_uses_upstream_strict_probability_threshold():
    p = np.asarray([0., .5e-10, 1e-10, 2e-10, .5], dtype=np.float64)
    assert M._entropy(p) == np.sum(np.asarray([-(2e-10) * np.log2(2e-10), .5]))


@pytest.mark.parametrize("hit", [1, 2, 3])
def test_both_action_variants_inherit_exact_public_updates(hit):
    k = kernel()
    four = M.SpaceAwareActor(packet(hit=hit), k)
    inside = M.SpaceAwareActor(packet(hit=hit), k, allow_stay=False)
    qualified = R.PublicBeliefView(k)
    qualified._reset(hit)
    for action, observed in [(0, 0), (2, 1), (1, 2), (3, 3)]:
        np.testing.assert_array_equal(four.belief, qualified.p_source)
        np.testing.assert_array_equal(inside.belief, qualified.p_source)
        advance(four, action, observed)
        advance(inside, action, observed)
        qualified._observe(four.public)
    np.testing.assert_array_equal(four.belief, qualified.p_source)
    np.testing.assert_array_equal(inside.belief, qualified.p_source)
    assert four.public["done"] is False and four.public["step"] == 4
    assert M.SpaceAwareActor.update is R.ReleasedPolicyActor.update
    assert M.SpaceAwareActor.reset is R.ReleasedPolicyActor.reset


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_boundary_mask_is_only_readout_difference(direction):
    four = M.SpaceAwareActor(packet(), kernel())
    inside = M.SpaceAwareActor(packet(), kernel(), allow_stay=False)
    for _ in range(26):
        advance(four, direction)
        advance(inside, direction)
    _, all_scores = four.choose()
    inside_action, in_scores = inside.choose()
    valid = inside.allowed_actions
    assert four.allowed_actions == (0, 1, 2, 3) and direction not in valid
    assert np.isfinite(all_scores).all() and np.isposinf(in_scores[direction])
    np.testing.assert_array_equal(all_scores[list(valid)], in_scores[list(valid)])
    assert inside_action in valid
    np.testing.assert_array_equal(four.belief, inside.belief)


def test_pending_terminal_and_reset_contract_without_simulator():
    actor = M.SpaceAwareActor(packet(), kernel())
    chosen, scores = actor.choose()
    original = actor.belief
    scores[:] = 0
    with pytest.raises(RuntimeError, match="awaiting"):
        actor.choose()
    with pytest.raises(ValueError, match="pending"):
        advance(actor, (chosen + 1) % 4)
    np.testing.assert_array_equal(actor.belief, original)
    advance(actor, chosen, done=True)
    assert actor.public["done"] and actor.allowed_actions == ()
    assert actor.belief[actor.public["position"]] == 1
    with pytest.raises(RuntimeError, match="found"):
        actor.choose()
    with pytest.raises(RuntimeError, match="found"):
        advance(actor, chosen)
    actor.reset(packet(hit=3))
    assert actor.allowed_actions == (0, 1, 2, 3)
    assert actor.choose()[0] in actor.allowed_actions


def test_accounted_cached_geometry_input_ownership_and_no_hidden_actor_fields():
    initial, k = packet(), kernel()
    actor = M.SpaceAwareActor(initial, k)
    before, storage = actor.belief, actor.storage_bytes()
    initial["position"][0] = 0
    k[:] = 0
    actor.belief[:] = 999
    np.testing.assert_array_equal(actor.belief, before)
    assert not hasattr(actor, "__dict__") and not hasattr(actor._policy, "__dict__")
    assert not any(hasattr(actor, name) for name in ("source", "seed", "history", "model", "env"))
    assert storage["mutable_array_bytes"] == 53 * 53 * 8
    assert storage["immutable_array_bytes"] == 5 * 107 * 107 * 8
    assert storage["immutable_arrays"]["manhattan_distance_table"] == 107 * 107 * 8
    cached = actor._policy.distance_array
    with pytest.raises(ValueError):
        cached.flags.writeable = True
    x, y = np.indices((107, 107))
    np.testing.assert_array_equal(cached, np.abs(x - 53) + np.abs(y - 53))
    action, _ = actor.choose()
    advance(actor, action)
    assert actor.storage_bytes() == storage
    assert actor._policy.distance_array is cached
    actor.choose()
    assert actor._policy.distance_array is cached


@pytest.mark.parametrize("allow", [1, 0, None, "yes", np.bool_(True)])
def test_action_contract_requires_explicit_python_bool(allow):
    with pytest.raises(TypeError, match="boolean"):
        M.SpaceAwareActor(packet(), kernel(), allow_stay=allow)


def test_no_score_repair_for_invalid_analytic_state():
    p = np.full((53, 53), np.nan)
    view = synthetic_view(p, (26, 26), kernel())
    with pytest.raises(FloatingPointError, match="no score repair"):
        M._SpaceAwarePolicy(env=view, model=None, sym_avg=False, allow_stay=True)._value_policy()


def test_hidden_fields_rejected_before_actor_state_construction():
    initial = packet()
    initial["source"] = (1, 1)
    with pytest.raises(ValueError, match="fields"):
        M.SpaceAwareActor(initial, kernel())
