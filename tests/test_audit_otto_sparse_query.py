"""Independent-auditor regressions using fabricated scalar records only."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_otto_sparse_query.py"
    spec = importlib.util.spec_from_file_location("_synthetic_sparse_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(module):
    result = []
    for identity in module.cohort():
        arm = identity["arm"]
        steps = {"analytic": 100, "neural": 80, "period2": 80, "random_pair": 200, "entropy": 200}[arm]
        cost = {"analytic": .1, "neural": 10., "period2": 5., "random_pair": 100., "entropy": 100.}[arm]
        queries = 0 if arm == "analytic" else steps if arm == "neural" else steps // 2
        result.append({**identity, "found": True, "steps": steps, "queries": queries, "updates": steps,
            "blocked_steps": 0, "final_update_assimilated": True, "query_quota_valid": True,
            "init_seconds": 0., "choose_seconds": cost, "update_seconds": 0.,
            "setup_allocation_seconds": 0., "controller_seconds": cost, "environment_seconds": 0.})
    return result


def test_episode_balanced_exact_lower_median_does_not_follow_long_episode_rows(audit_module):
    rows = [{"episode_index": index, "steps": 1 if index < 12 else 3} for index in range(24)]
    points = [{"episode_index": r["episode_index"], "step": step, "entropy": .25 if r["episode_index"] < 12 else .75}
              for r in rows for step in range(r["steps"])]
    before = copy.deepcopy(points)
    answer = audit_module.median_record(list(reversed(points)), list(reversed(rows)))
    assert answer["value"] == answer["threshold"] == .25
    assert answer["selection_cumulative_weight_exact"] == [1, 2]
    assert answer["decisions"] == 48 and points == before
    with pytest.raises(ValueError, match="unique"):
        audit_module.median_record(points[:-1] + [points[0]], rows)
    with pytest.raises(ValueError, match="all VALID"):
        audit_module.median_record(points[:-1], rows)


def test_causal_credit_entropy_tie_and_random_pair_complete_exposure(audit_module):
    used, choices = 0, []
    for step, entropy in enumerate((.25, .5, 1.25, .5, -.25, .5)):
        query, offset, digest = audit_module.expected_query("entropy", 12, step, used, entropy, .5)
        assert offset is digest is None
        choices.append(query); used += int(query)
    assert choices == [False, True, True, False, False, True]
    used, choices = 0, []
    for step in range(32):
        query, offset, digest = audit_module.expected_query("random_pair", 123, step, used, .5, None)
        assert query == (step % 2 == offset) and len(digest) == 64
        choices.append(query); used += int(query)
        assert used <= (step + 2) // 2
    assert used == 16 and all(sum(choices[s:s+2]) == 1 for s in range(0, 32, 2))
    with pytest.raises(ValueError, match="prior count"):
        audit_module.expected_query("period2", 1, 3, 1, .5, None)
    with pytest.raises(ValueError, match="invalid chronological"):
        audit_module.expected_query("entropy", 1, 2, 2, .5, .5)


def test_original_endpoint_precision_strict_tie_and_blocked_actions(audit_module):
    f32 = audit_module.f32
    epsilon = f32(1e-10)
    assert audit_module.selected_action([None, epsilon, 0., None], [1, 2], True) == 2
    assert audit_module.selected_action([None, f32(5e-11), 0., None], [1, 2], True) == 1
    assert audit_module.selected_action([None, 1e-10, 0., None], [1, 2], False) == 2
    assert audit_module.selected_action([None, 9e-11, 0., None], [1, 2], False) == 1
    with pytest.raises(ValueError, match="float32"):
        audit_module.selected_action([None, 1e-10, 0., None], [1, 2], True)
    with pytest.raises(ValueError, match="eligible"):
        audit_module.selected_action([0., 1., 2., 3.], [2, 1], False)


def test_primary_periodic_gate_has_no_control_veto_or_replacement_winner(audit_module):
    rows = [r for r in records(audit_module) if r["stage"] == "eval"]
    for row in rows:
        if row["regime"] == "lambda5" and row["arm"] == "period2":
            row["steps"] = row["updates"] = 500
            row["controller_seconds"] = row["choose_seconds"] = 100.
    mixtures = {regime: {str(h): 1/3 for h in (1, 2, 3)} for regime in audit_module.FIRST}
    original = copy.deepcopy(rows)
    answer = audit_module.reduce_rows(rows, mixtures, 72.)
    assert answer["pilot_continuation"] and answer["required_passed"] == 16
    assert len(answer["required"]) == 16 and len(answer["diagnostic"]) == 54
    assert answer["diagnostic_passed"] < 54
    assert answer["regimes"]["lambda3"]["means"]["entropy"]["paid_controller_seconds"] == 101.
    assert rows == original
    for row in rows:
        if row["regime"] == "lambda3" and row["arm"] == "period2":
            row["controller_seconds"] = row["choose_seconds"] = 6.01
        if row["arm"] in ("random_pair", "entropy"):
            row["steps"] = row["updates"] = 80
            row["controller_seconds"] = row["choose_seconds"] = 4.
    failed = audit_module.reduce_rows(rows, mixtures, 0.)
    assert not failed["pilot_continuation"] and failed["required_passed"] == 15
    assert any(c["name"] == "lambda3.entropy.paid_cost_vs_neural" and c["passes"] for c in failed["diagnostic"])


def test_explicit_threshold_boundary_must_precede_first_eval_attempt(audit_module, tmp_path):
    identities = audit_module.cohort()
    rows, boundaries = [], []
    pin = {"sha256": "a" * 64, "bytes": 100}
    for identity in identities:
        if identity["episode_index"] == 24:
            boundaries.append({"event": "threshold_published", "threshold": pin,
                "completed_validation_episodes": 24, "completed_evaluation_episodes": 0})
        row = {**identity, "steps": 1, "found": True, "queries": int(identity["arm"] != "analytic"),
            "updates": 1, "blocked_steps": 0, "final_update_assimilated": True, "query_quota_valid": True,
            "final_public": {"step": 1, "done": True}, "source_evaluation_only": [26, 25],
            "draws_evaluation_only": [{"channel": "source", "index": 0, "uniform": .3, "selected_index": 1403, "cdf_mass": 1.}],
            **{k: 0. for k in audit_module.METRICS if k.endswith("seconds")}}
        rows.append(row)
        boundaries.extend(({"event": "attempt", **identity}, {"event": "return", **identity, "steps": 1}))
    auditor = audit_module.Audit(SimpleNamespace(output=tmp_path / "unused"))
    auditor.run, auditor.worker = tmp_path, {"threshold": pin}
    auditor.rows = lambda path: iter(copy.deepcopy(rows if path.name == "episodes.jsonl" else boundaries))
    assert len(auditor.episodes()) == 384
    boundaries[48], boundaries[49] = boundaries[49], boundaries[48]
    with pytest.raises(ValueError, match="threshold barrier"):
        auditor.episodes()


def test_late_failure_demotes_completed_receipt_and_preserves_original_error(audit_module, tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "ROOT", tmp_path)
    monkeypatch.setattr(audit_module.signal, "signal", lambda *_: None)
    output = tmp_path / "audit"
    auditor = audit_module.Audit(SimpleNamespace(output=output))
    auditor.clock, auditor.start = SimpleNamespace(now_ns=lambda: 10), 0
    def body():
        audit_module.write(output / "started.json", {"fabricated": True})
        audit_module.write(output / "audit.json", {"fabricated": True})
    auditor.body = body
    original = TimeoutError("fabricated late deadline")
    def check():
        if (output / "receipt.json").exists():
            raise original
    auditor.check = check
    with pytest.raises(TimeoutError) as caught:
        auditor.execute()
    assert caught.value is original
    failed = json.loads((output / "receipt.json").read_text())
    old = json.loads((output / "receipt.invalid.json").read_text())
    assert failed["status"] == "failed" and failed["agreement"] is False
    assert old["status"] == "completed" and "fabricated late deadline" in failed["failures"][0]["error"]
    assert (output / "audit.json").exists()


def test_authentication_rejects_relative_producer_command_and_failed_parent(audit_module, tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "ROOT", tmp_path)
    args = SimpleNamespace(output=tmp_path / "unused", worker=tmp_path / "worker.json",
                           terminal=tmp_path / "terminal.json", plan=tmp_path / "plan.json", plan_sha256="plan-pin")
    worker = {"status": "completed", "complete": True, "version": audit_module.PRODUCER_VERSION,
        "pending": [], "pending_emission": None, "pending_episode": None, "completed_episodes": 384,
        "validation_episodes": 24, "evaluation_episodes": 360, "training_updates": 0, "annotations": 0,
        "plan_sha256": "plan-pin", "supervision_sha256": "launch-pin"}
    command = [str(tmp_path / ".venv-otto-released-native/bin/python"), str(tmp_path / audit_module.PRODUCER), "run",
               "--supervision", str(tmp_path / "launch.json")]
    parent = {"command": command, "status": "completed", "returncode": 1, "timed_out": False,
        "error": None, "clock_error": None, "group_absent": True, "cleanup": {"reaped": True, "errors": []}}
    auditor = audit_module.Audit(args)
    auditor.plan = {}
    auditor.read = lambda path: copy.deepcopy(worker if path == args.worker else parent if path == args.terminal else {})
    auditor.path = Path
    auditor.digest = lambda _path: {"sha256": "launch-pin", "bytes": 1}
    with pytest.raises(ValueError, match="successful original parent"):
        auditor.authenticate()
    parent["returncode"] = 0
    parent["command"][1] = audit_module.PRODUCER
    with pytest.raises(ValueError, match="absolute original command"):
        auditor.authenticate()
