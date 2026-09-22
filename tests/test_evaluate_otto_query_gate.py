"""Fabricated metric/accounting contracts, no native or learned-model calls."""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_query_evaluation", ROOT / "scripts/evaluate_otto_query_gate.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


def fabricated():
    rows = []
    for identity in E.cohort():
        arm = identity["arm"]
        steps = 90 if arm.startswith("gru32") else 200 if arm.startswith("mlp190") else 100
        cost = 40. if arm.startswith("gru32") else 80. if arm.startswith("mlp190") else 100.
        rows.append({**identity, "steps": steps, "found": True, "updates": steps, "blocked_steps": 0,
                     "queries": 0 if arm == "analytic" else steps,
                     "init_seconds": 1., "choose_seconds": cost - 3., "update_seconds": 1.,
                     "setup_allocation_seconds": 1., "controller_seconds": cost, "environment_seconds": 2.,
                     "query_age_histogram": {0: steps}})
    return rows, {regime: {1: .6, 2: .3, 3: .1} for regime in E.FIRST}


def test_losing_mlp_does_not_veto_gru_and_transfer_remains_visible():
    rows, mixtures = fabricated()
    assert len(rows) == len({r["episode_id"] for r in rows}) == 576
    for r in rows:
        if r["regime"] == "lambda5":
            r.update(found=False, steps=2188, updates=2188)
    result = E.aggregate(rows, mixtures)
    assert result["required_conditions"] == result["required_passed"] == 38
    assert result["reported_conditions"] == 62
    assert result["pilot_continuation"] is True
    assert sum(x["passes"] for x in result["criteria"]["mlp_absolute"]) == 6
    assert result["reported_passed"] == 44
    assert result["regimes"]["lambda5"]["family_means"]["gru32"]["found"] == 0
    assert len(result["regimes"]["lambda5"]["blocks"]) == 8
    with pytest.raises(ValueError, match="complete 576"):
        E.aggregate(rows[:-1], mixtures)
    rows[0]["found"] = False
    with pytest.raises(ValueError, match="complete finite"):
        E.aggregate(rows, mixtures)


def test_hit_mixture_and_every_paired_block_use_declared_weights():
    rows, mixtures = fabricated()
    for r in rows:
        r["steps"] = r["updates"] = {1: 10, 2: 20, 3: 80}[r["initial_hit"]]
    result = E.aggregate(rows, mixtures)
    for regime in E.FIRST:
        for arm in E.ARMS:
            assert result["regimes"][regime]["means"][arm]["steps"] == 20
            assert [b[arm]["steps"] for b in result["regimes"][regime]["blocks"]] == [20] * 8
    assert [r["arm"] for r in rows[8:16]] == list(E.ARMS[1:] + E.ARMS[:1])


def test_cold_allocations_exclude_tf_from_analytic_and_conserve_physical_cost():
    setup = {"common_import_seconds": 2., "shared_evaluator_setup_seconds": 4., "model_setup_seconds": 14.}
    loads = dict.fromkeys(E.LEARNED, 3.)
    by_arm = {a: E.setup_allocation(setup, a, loads, 12., 6.) for a in E.ARMS}
    assert by_arm["analytic"]["tf_setup_allocation_seconds"] == 0
    assert by_arm["analytic"]["gate_setup_allocation_seconds"] == 0
    assert sum(row["tf_setup_allocation_seconds"] * 72 for row in by_arm.values()) == pytest.approx(14)
    assert sum(row["common_setup_allocation_seconds"] * 72 for row in by_arm.values()) == pytest.approx(12)
    assert sum(row["gate_setup_allocation_seconds"] * 72 for row in by_arm.values()) == pytest.approx(30)
    assert sum(row["setup_allocation_seconds"] * 72 for row in by_arm.values()) == pytest.approx(56)


def test_uncertain_return_and_cleanup_failure_cannot_publish_completion(tmp_path, monkeypatch):
    run = E.Run(types.SimpleNamespace(output=tmp_path / "attempt"))
    run.check = lambda: None
    run.bind = lambda: None
    original = OSError("fabricated return publication failure")

    def emit(_name, event):
        if event["event"] == "return":
            raise original

    run.ledger.emit = emit
    run.body = lambda: run.ledger.call("actor_choose", lambda: 2)

    def close(**_kwargs):
        raise OSError("fabricated stream-close failure")

    run.ledger.close = close
    monkeypatch.setattr(E.signal, "signal", lambda *_args: None)
    with pytest.raises(OSError) as caught:
        run.execute()
    assert caught.value is original
    receipt = json.loads((run.out / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["completed_episodes"] == 0
    assert receipt["calls"]["actor_choose"]["attempted"] == 1
    assert receipt["calls"]["actor_choose"]["returned"] == 0
    assert receipt["pending"][0]["call_id"] == 1
    assert "stream-close" in receipt["cleanup_errors"][0]["error"]
