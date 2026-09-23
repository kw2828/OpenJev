"""Small fabricated public cases, no native simulator or learned model."""
from __future__ import annotations

import numpy as np
import pytest

from openjev.research import otto_query_pair_rollouts as pair
from openjev.research import otto_teacher_rollouts as old


def fixture(position=(26, 26)):
    p = np.zeros((53, 53), np.float64)
    p[24, 26], p[26, 29] = .25, .75
    k = np.zeros((4, 107, 107), np.float64)
    k[0], k[1], k[2], k[3] = .125, .25, .125, .5
    k[:, 53, 53] = 0
    public = {"position": position, "step": 9, "hit": 1, "done": False,
              "valid_actions": tuple(a for a in range(4)
                if 0 <= position[a // 2] + 2 * (a % 2) - 1 < 53)}
    return public, p, k


def options():
    return {"seed": 19, "anchor_id": 4, "replicate_ids": (2, 0, 1), "horizon": 3}


def test_subset_matches_all_actions_and_preserves_streams_and_inputs():
    public, p, k = fixture()
    before = p.tobytes(), k.tobytes(), dict(public)
    all_events, events = [], []
    full = old.sample_teacher_panel(public, p, k, **options(), emit=all_events.append)
    actual = pair.sample_query_pair(public, p, k, analytic_action=3, neural_action=0,
                                    **options(), emit=events.append)
    assert actual == tuple(r for r in full if r.first_action in (0, 3))
    assert [(r.replicate_id, r.first_action) for r in actual] == [(r, a) for r in range(3) for a in (0, 3)]
    def draws(rows):
        return {(e.get("replicate_id"), e.get("first_action"), e.get("local_step"), e["operation"]):
                (e["uniform"], e["selected_index"], e["cdf_mass"])
                for e in rows if e["event"] == "return" and e["operation"] in ("hit_draw", "source_draw")
                and e.get("first_action", 0) in (0, 3)}
    assert draws(events) == draws(all_events)
    assert (p.tobytes(), k.tobytes(), public) == before


def test_identical_endpoints_sample_once_and_found_skips_odor():
    public, p, k = fixture()
    p[:] = 0
    p[25, 26] = 1
    events = []
    result = pair.sample_query_pair(public, p, k, analytic_action=0, neural_action=0,
                                    **options(), emit=events.append)
    assert len(result) == 3 and all(r.found and r.steps == 1 for r in result)
    returns = [e for e in events if e["event"] == "return"]
    assert sum(e["operation"] == "source_draw" for e in returns) == 3
    assert sum(e["operation"] == "teacher_snapshot" for e in returns) == 3
    assert not any(e["operation"] == "hit_draw" for e in returns)
    updates = [e for e in returns if e["operation"] == "teacher_update"]
    assert len(updates) == 3
    assert all(e["public"]["step"] == 10 and e["public"]["hit"] == -2 for e in updates)


def test_censored_final_update_and_failure_never_becomes_a_record(monkeypatch):
    public, p, k = fixture()
    events = []
    result = pair.sample_query_pair(public, p, k, analytic_action=1, neural_action=2,
             **{**options(), "horizon": 1}, emit=events.append)
    assert all(r.steps == 1 and not r.found for r in result)
    assert sum(e["event"] == "return" and e["operation"] == "teacher_update" for e in events) == 6
    def fail(*_args, **_kwargs):
        raise RuntimeError("failed update")
    monkeypatch.setattr(old, "_update", fail)
    events.clear()
    with pytest.raises(RuntimeError, match="failed update"):
        pair.sample_query_pair(public, p, k, analytic_action=1, neural_action=2, **options(), emit=events.append)
    assert events[-1]["event"] == "attempt" and events[-1]["operation"] == "teacher_update"
    assert not any(e["event"] in ("record", "panel_complete") for e in events)


@pytest.mark.parametrize("changes", [{"analytic_action": 0}, {"replicate_ids": (1, 1)},
                                    {"horizon": 0}, {"seed": True}])
def test_invalid_declarations(changes):
    args = {**options(), "analytic_action": 1, "neural_action": 3, **changes}
    with pytest.raises((TypeError, ValueError)):
        pair.sample_query_pair(*fixture((0, 0)), **args)
