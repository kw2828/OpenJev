"""Fabricated runner contracts only; no native environment or neural model."""
from __future__ import annotations

import importlib.util
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_query_advantage_runner", ROOT / "scripts/study_otto_query_advantage.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def test_fixed_72_paths_anchor_ids_rotation_and_real_call_bounds():
    rows = S.cohort()
    assert len(rows) == len({r["episode_id"] for r in rows}) == 72
    assert [r["episode_index"] for r in rows] == list(range(72))
    assert Counter((r["regime"], r["initial_hit"], r["schedule"]) for r in rows) == Counter(
        {(regime, hit, schedule): 4 for regime in ("base", "shift") for hit in (1, 2, 3) for schedule in S.SCHEDULES})
    ids = []
    queries = annotations = 0
    for index in range(24):
        group = rows[3*index:3*index+3]; offset = index % 3
        assert tuple(r["schedule"] for r in group) == S.SCHEDULES[offset:] + S.SCHEDULES[:offset]
    for row in rows:
        queries += sum(S.scheduled_query(row["schedule"], t) for t in range(2188))
        annotations += sum(not S.scheduled_query(row["schedule"], t) for t in (0, 4, 8, 16, 32))
        ids.extend(row["episode_index"]*5+i for i in range(5))
    assert ids == list(range(360))
    assert queries + annotations == S.CALL_CAPS["tensorflow_value"] == 78888
    assert annotations == S.CALL_CAPS["annotation_score"] == 120
    assert S.LIMITS["native_steps"] == 72*2188 and S.LIMITS["paired_steps"] == 360*16*2*32


def test_anchor_is_preaction_public_owned_and_unsupported_fails_before_labels():
    run = S.Run(types.SimpleNamespace(output=Path("unused")))
    run.runtime = types.SimpleNamespace(np=np, kernels={"base": np.zeros((4, 107, 107), np.float64)})
    p = np.zeros((53, 53), np.float64); p[1, 1] = 1
    public = {"position": (26, 26), "step": 4, "hit": 0, "done": False, "valid_actions": (0, 1, 2, 3)}
    actor = types.SimpleNamespace(public=public, belief=p)
    features = np.arange(31, dtype=np.float32)
    seen = {"features": features, "analytic_action": 1, "analytic": np.array([3, 0, 2, 4], np.float64)}
    costs = np.array([1, 2, 3, 0], np.float32)
    events, calls = [], []
    run.ledger = types.SimpleNamespace(call=lambda _name, fn: fn(), emit=lambda name, r: events.append((name, r)))
    run.select_neural = lambda scores, allowed: 3
    def snapshot(packet, belief, kernel):
        calls.append((dict(packet), belief.tobytes(), kernel.shape))
    run.snapshot = snapshot
    run.capture_anchor(S.cohort()[0], 4, actor, seen, costs)
    assert len(calls) == len(run.anchor_arrays) == len(run.anchors) == 1 and run.panels == []
    row = run.anchors[0]
    assert row["anchor_id"] == 1 and row["preaction_step"] == row["public"]["step"] == 4
    assert not {"source", "source_evaluation_only", "draws"} & row.keys()
    p[:] = 0; features[:] = -1
    assert run.anchor_arrays[0][0][1, 1] == 1 and run.anchor_arrays[0][1][0] == 0
    assert not run.anchor_arrays[0][0].flags.writeable and not run.anchor_arrays[0][1].flags.writeable
    def unsupported(*_args):
        raise ValueError("anchor belief must already be normalized; no repair")
    run.snapshot = unsupported
    with pytest.raises(ValueError, match="no repair"):
        run.capture_anchor(S.cohort()[1], 4, actor, seen, costs)
    assert events[-1][1]["status"] == "unsupported" and len(run.anchor_arrays) == 1 and run.panels == []
    def broken_publication(*_args):
        raise OSError("unsupported evidence write failed")
    run.ledger.emit = broken_publication
    with pytest.raises(ValueError, match="no repair") as failed:
        run.capture_anchor(S.cohort()[1], 4, actor, seen, costs)
    assert "evidence write failed" in failed.value.__notes__[0]


def test_labels_require_all_trajectories_and_saved_anchor_archive(tmp_path):
    run = S.Run(types.SimpleNamespace(output=tmp_path))
    called = []
    run.sample_pair = lambda *_a, **_k: called.append(1)
    with pytest.raises(ValueError, match="all physical paths"):
        run.label_anchors()
    run.rows = [None]*72; run.anchors = [None]*360
    with pytest.raises(ValueError, match="all physical paths"):
        run.label_anchors()
    assert called == []


def test_sampler_return_write_failure_retains_attempt_and_uncertainty(tmp_path):
    run = S.Run(types.SimpleNamespace(output=tmp_path))
    emitted = []
    def emit(_name, event):
        if event["event"] == "return":
            raise OSError("durable return failure")
        emitted.append(event)
    run.ledger = types.SimpleNamespace(emit=emit)
    base = {"anchor_id": 0, "seed": 19300001, "operation_id": 1, "operation": "source_draw"}
    run.sampler_event(0, {**base, "event": "attempt"})
    with pytest.raises(OSError, match="durable"):
        run.sampler_event(0, {**base, "event": "return", "source": [0, 0]})
    assert run.sampler_calls["source_draw"] == {"attempted": 1, "returned": 0}
    assert run.sampler_pending == [{"anchor_id": 0, "operation_id": 1, "operation": "source_draw"}]
    assert run.sampler_events == 1 and run.sampler_records == 0 and len(emitted) == 1
