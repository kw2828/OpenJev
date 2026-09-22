"""Fabricated collection contracts, never native environments or actual models."""
from __future__ import annotations

import importlib.util
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_query_collection", ROOT / "scripts/collect_otto_query_gate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_fixed_cohort_rotations_and_complete_schedule_tails():
    rows = C.cohort()
    assert len(rows) == len({r["episode_id"] for r in rows}) == 60
    assert [r["episode_index"] for r in rows] == list(range(60))
    assert Counter((r["regime"], r["initial_hit"], r["schedule"]) for r in rows) == Counter(
        {(regime, hit, schedule): 2 for regime in ("base", "shift") for hit in (1, 2, 3) for schedule in C.SCHEDULES})
    for index in range(12):
        group = rows[index * 5:(index + 1) * 5]
        expected = ("always", "never", "period2", "period8", "initial_only")
        offset = index % 5
        assert tuple(r["schedule"] for r in group) == expected[offset:] + expected[:offset]
        assert len({r["seed"] for r in group}) == 1
    counts = {name: sum(C.scheduled_query(name, t) for t in range(2188)) for name in C.SCHEDULES}
    assert counts == {"always": 2188, "never": 0, "period2": 1094, "period8": 274, "initial_only": 1}
    assert C.LIMITS["native_steps"] == C.LIMITS["tensorflow_value_calls"] == 60 * 2188
    with pytest.raises(ValueError, match="preaction"):
        C.scheduled_query("never", 2188)


def test_annotation_uses_eligible_near_set_not_selected_action_inequality():
    # Action 1 is a near-tie despite action 0 being the selected first minimum.
    # The lower blocked action 3 cannot change the eligible minimum or label.
    costs = np.array([0, 2**-34, 2**-33, -9], np.float32)
    original = costs.tobytes()
    label, gap = C.annotation_label(costs, (0, 1, 2), 1, np)
    assert label is False and gap.dtype == np.float32 and gap == np.float32(2**-34)
    assert C.annotation_label(costs, (0, 1, 2), 2, np)[0] is True
    assert C.annotation_label(np.zeros(4, np.float32), (0, 1, 2, 3), 3, np) == (False, np.float32(0))
    assert costs.tobytes() == original
    with pytest.raises(ValueError, match="ordered"):
        C.annotation_label(costs, (1, 0), 1, np)
    with pytest.raises(ValueError, match="float32"):
        C.annotation_label(costs.astype(np.float64), (0, 1), 0, np)


def test_query_reuses_annotation_and_skipped_annotation_cannot_mutate_history():
    run = object.__new__(C.Run)
    actor = types.SimpleNamespace(public={"step": 4}, state=np.array([5], np.float32),
                                  last_choice={"queried": False, "action": 1}, progress={"pending_action": 1},
                                  belief=np.array([.2, .8], np.float64))
    calls = []

    def call(channel, function):
        calls.append(channel)
        return function()

    run.ledger = types.SimpleNamespace(call=call)
    scores = np.array([3, 2, 1, 0], np.float32)
    policy = types.SimpleNamespace(_value_policy=lambda: (3, scores))
    actual = run.annotate(actor, {"neural": policy}, {"neural": scores}, True)
    assert calls == [] and actual.tobytes() == scores.tobytes() and actual is not scores
    actual = run.annotate(actor, {"neural": policy}, {}, False)
    assert calls == ["annotation_score"] and actual.tobytes() == scores.tobytes()
    assert actor.state.tolist() == [5] and actor.last_choice == {"queried": False, "action": 1}

    def invalid_backend():
        actor.state[0] = 0
        return 3, scores

    policy._value_policy = invalid_backend
    with pytest.raises(ValueError, match="cannot change"):
        run.annotate(actor, {"neural": policy}, {}, False)


def test_failed_return_publication_retains_attempt_and_pending(tmp_path):
    ledger = C.Ledger(types.SimpleNamespace(check=lambda: None, out=tmp_path))
    emitted = []

    def emit(_name, value):
        emitted.append(value)
        if value["event"] == "return":
            raise OSError("fabricated return fsync failure")

    ledger.emit = emit
    with pytest.raises(OSError, match="fsync"):
        ledger.call("actor_choose", lambda: 2)
    assert [x["event"] for x in emitted] == ["attempt", "return"]
    assert ledger.calls["actor_choose"] == {"attempted": 1, "returned": 0, "seconds": 0.}
    assert len(ledger.pending) == 1 and ledger.pending[0]["call_id"] == 1
