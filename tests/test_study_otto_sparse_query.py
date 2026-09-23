"""Fabricated scheduling, accounting and durability contracts, no native calls."""
from __future__ import annotations

import importlib.util
import json
import math
import types
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_sparse_runner_test", ROOT / "scripts/study_otto_sparse_query.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def test_fixed_validation_then_rotated_five_arm_evaluation():
    rows = S.cohort()
    assert len(rows) == len({r["episode_id"] for r in rows}) == 384
    assert [r["episode_index"] for r in rows] == list(range(384))
    assert rows[:24] == S.cohort("valid")
    assert all(r["arm"] == "period2" for r in rows[:24])
    assert Counter((r["regime"], r["initial_hit"]) for r in rows[:24]) == Counter(
        {(regime, hit): 4 for regime in S.VALID_FIRST for hit in (1, 2, 3)})
    for case_index in range(72):
        group = rows[24 + 5 * case_index:29 + 5 * case_index]
        offset = case_index % 5
        assert tuple(r["arm"] for r in group) == S.ARMS[offset:] + S.ARMS[:offset]
        assert len({r["seed"] for r in group}) == 1
        assert all(r["stage"] == "eval" and r["block"] == r["case"] // 3 for r in group)
    assert Counter((r["regime"], r["arm"], r["initial_hit"]) for r in rows[24:]) == Counter(
        {(regime, arm, hit): 8 for regime in S.FIRST for arm in S.ARMS for hit in (1, 2, 3)})
    assert {r["seed"] for r in rows[:24]}.isdisjoint(r["seed"] for r in rows[24:])
    assert S.CALL_CAPS["tensorflow_value"] == 420096


def test_disjoint_setup_shares_conserve_physical_cost_once():
    setup = {"common_import_seconds": 1., "shared_evaluator_setup_seconds": 2., "model_setup_seconds": 13.}
    allocations = [S.setup_allocation(setup, r["arm"], 7., 3.) for r in S.cohort()]
    assert math.fsum(a["common_setup_allocation_seconds"] for a in allocations) == pytest.approx(6.)
    assert math.fsum(a["tf_setup_allocation_seconds"] for a in allocations) == pytest.approx(13.)
    assert math.fsum(a["gate_setup_allocation_seconds"] for a in allocations) == pytest.approx(7.)
    assert math.fsum(a["setup_allocation_seconds"] for a in allocations) == pytest.approx(26.)
    valid_share = math.fsum(a["setup_allocation_seconds"] for a in allocations[:24])
    assert valid_share == pytest.approx(24 * (6 / 384 + 13 / 312 + 7 / 240))
    assert all(a["gate_setup_allocation_seconds"] == 0 for a, r in zip(allocations, S.cohort(), strict=True)
               if r["arm"] in ("analytic", "neural"))


def test_pairing_checks_union_of_later_arm_draw_prefixes():
    paired = {}
    def row(values):
        return {"regime": "lambda3", "case": 0, "source_evaluation_only": [7, 9],
            "draws_evaluation_only": [{"channel": "hit", "index": i, "uniform": u}
                                     for i, u in enumerate(values)]}
    S.paired_identity(row([.1]), paired)
    S.paired_identity(row([.1, .2, .3]), paired)
    with pytest.raises(ValueError, match="paired original"):
        S.paired_identity(row([.1, .2, .4]), paired)


@pytest.mark.parametrize("failure", ["flush", "return"])
def test_episode_acknowledgement_waits_for_durable_full_boundary(tmp_path, failure):
    run = S.Run(types.SimpleNamespace(output=tmp_path))
    identity = S.cohort("valid")[0]
    row = {**identity, "steps": 3}
    marker = OSError("original durable write failure")
    events = []
    def episode(_):
        run.pending_episode = dict(identity)
        return row
    def flush():
        events.append("flush")
        if failure == "flush":
            raise marker
    def append(name, record):
        events.append(name)
        if record.get("event") == "return":
            raise marker
    run.episode, run.ledger.flush, run.append_durable = episode, flush, append
    run.check = lambda: None
    with pytest.raises(OSError) as caught:
        run.complete_episode(identity, {})
    assert caught.value is marker
    assert run.pending_episode == identity and run.rows == []
    assert events == (["flush"] if failure == "flush" else ["flush", "episodes.jsonl", "episode-boundaries.jsonl"])


def test_all_validation_paths_close_before_single_threshold_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    run = S.Run(types.SimpleNamespace(output=tmp_path))
    events = []
    run.check = lambda: None
    run.ledger.flush = lambda: events.append("flushed")
    def complete(identity, _paired):
        assert identity["stage"] == "valid" and run.threshold is None
        run.rows.append({**identity, "steps": 1, "setup_allocation_seconds": .25})
        run.validation_decisions.append({"episode_index": identity["episode_index"], "step": 0, "entropy": .5})
        events.append(identity["episode_index"])
    def select(decisions, rows):
        assert len(rows) == len(decisions) == 24 and events == list(range(24))
        assert not (tmp_path / "threshold.json").exists()
        events.append("selected")
        return {"value": .5, "episodes": 24, "decisions": 24}
    run.complete_episode = complete
    run.metrics = types.SimpleNamespace(select_threshold=select)
    run.ledger.call = lambda channel, function: function()
    run.calibrate()
    assert events == [*range(24), "selected", "flushed"]
    assert json.loads((tmp_path / "threshold.json").read_text()) == run.threshold
    assert run.threshold_descriptor == S.descriptor(tmp_path / "threshold.json")
    boundary = json.loads((tmp_path / "episode-boundaries.jsonl").read_text())
    assert boundary == {"event": "threshold_published", "threshold": run.threshold_descriptor,
                        "completed_validation_episodes": 24, "completed_evaluation_episodes": 0}
    assert run.validation_setup_seconds == 6.
    assert run.calibration_paid_seconds == run.validation_wall_seconds + run.threshold_wall_seconds + 6.


def test_missing_validation_completion_blocks_threshold(tmp_path):
    run = S.Run(types.SimpleNamespace(output=tmp_path))
    called = []
    def incomplete(identity, _paired):
        if identity["episode_index"] != 23:
            run.rows.append(identity)
    run.complete_episode = incomplete
    run.metrics = types.SimpleNamespace(select_threshold=lambda *args: called.append(args))
    with pytest.raises(ValueError, match="all VALID closed"):
        run.calibrate()
    assert called == [] and not (tmp_path / "threshold.json").exists()
