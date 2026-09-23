"""Fabricated saved-record arithmetic only; no model, teacher or simulator calls."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_otto_score_forecasts.py"
SPEC = importlib.util.spec_from_file_location("_test_saved_score_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def metrics(gap=2., agreement=.5):
    result = {}
    for regime in audit.REGIMES:
        result[regime] = {"episode_weighted_raw_gap": gap, "episode_weighted_agreement": agreement,
                         "by_age": {str(age): {"episode_weighted_raw_gap": gap} for age in (1, 2, 3)}}
    return {"by_regime": result}


def complete_models():
    return [{"family": family, "seed": seed,
             "metrics": metrics(1. if family == "residual_gru" else 2., .75)}
            for family in audit.FAMILIES for seed in audit.SEEDS]


def test_scalar_metrics_hand_weights_ties_current_mask_centering_and_zero_support():
    # Episode0 has 2 forecasts, episode1 one, episode2 is query-only.
    q = [[[0., 2., 4., -99.]] * 4 for _ in range(3)]
    p = copy.deepcopy(q)
    p[0][1] = [2., 0., 4., -999.]
    p[0][2] = [100., 102., 104., -999.]
    p[1][1] = [0., 2., 4., -999.]
    masks = [[[True, True, False, False]] * 4 for _ in range(3)]
    value = audit.scalar_metrics(p, q, masks, [3, 2, 1], [0, 1, 2],
                                 ["a", "b", "empty"], ["lambda3", "lambda3", "lambda4"])
    overall, base = value["overall"], value["by_regime"]["lambda3"]
    assert overall["episode_weighted_agreement"] == .5
    assert overall["episode_weighted_raw_gap"] == pytest.approx(1 / 3)
    assert overall["episode_weighted_centered_mse"] == pytest.approx(2 / 3)
    assert overall["zero_support_episode_ids"] == ["empty"]
    assert overall["weight_mass"] == pytest.approx(2 / 3)
    assert base["episode_weighted_agreement"] == .75
    assert base["by_age"]["2"]["episode_weighted_agreement"] == .5
    assert base["by_age"]["2"]["supported_episode_agreement"] == 1
    assert value["by_regime"]["lambda4"]["supported_episode_raw_gap"] is None


def test_float32_strict_near_ties_and_blocked_minimum():
    eps = audit.f32(1e-10)
    inside = audit.f32(eps * (1 - 2**-23))
    assert audit.near_set([0., inside, eps, -999.], [True, True, True, False])[0] == [0, 1]
    assert audit.near_set([inside, 0., eps, -999.], [True, True, True, False])[0][0] == 0
    with pytest.raises(ValueError):
        audit.near_set([0., 1., 2., 3.], [False] * 4)


def test_all_45_conditions_are_required_and_zero_baseline_has_no_epsilon_escape():
    models = complete_models()
    support = {regime: {str(age): 6 for age in (1, 2, 3)} for regime in audit.REGIMES}
    rules = audit.continuation_rules(models, metrics(2., .5), support)
    assert len(rules) == 45 and all(row["passes"] for row in rules)
    assert len({row["name"] for row in rules}) == 45
    support["lambda4"]["3"] = 3
    rules = audit.continuation_rules(models, metrics(2., .5), support)
    assert [row["name"] for row in rules if not row["passes"]] == ["lambda4.age3.case_support"]
    support["lambda4"]["3"] = 6
    for row in models:
        row["metrics"] = metrics(0., .75)
    assert all(row["passes"] for row in audit.continuation_rules(models, metrics(0., .5), support))
    models[0]["metrics"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = 1e-30
    rules = audit.continuation_rules(models, metrics(0., .5), support)
    failed = {row["name"] for row in rules if not row["passes"]}
    assert "lambda3.225001.gap_vs_hold" in failed
    assert "lambda3.mean.episode_weighted_raw_gap_vs_history_mlp" in failed


def test_every_candidate_seed_and_age_must_pass_no_winner_selection():
    models = complete_models()
    support = {regime: {str(age): 6 for age in (1, 2, 3)} for regime in audit.REGIMES}
    target = next(row for row in models if row["family"] == "residual_gru" and row["seed"] == 225003)
    target["metrics"]["by_regime"]["lambda4"]["by_age"]["3"]["episode_weighted_raw_gap"] = 2.1
    failures = [row["name"] for row in audit.continuation_rules(models, metrics(), support) if not row["passes"]]
    assert failures == ["lambda4.225003.age3.gap_vs_hold"]
    with pytest.raises(ValueError, match="twelve"):
        audit.continuation_rules(models[:-1], metrics(), support)


@pytest.mark.parametrize("defect", ["timeout", "group_survived", "worker_after_deadline"])
def test_failed_original_parent_cannot_be_promoted_from_completed_worker(defect):
    launch = {"cwd": str(audit.ROOT), "cap_seconds": 600, "clock_source_sha256": audit.CLOCK_PIN,
              "watchdog_sha256": audit.SUPERVISOR_PIN, "started_ns": 100,
              "deadline_ns": 600 * 10**9 + 100}
    terminal = {**launch, "status": "completed", "returncode": 0, "timed_out": False,
                "error": None, "clock_error": None, "group_absent": True,
                "cleanup": {"reaped": True, "errors": []}, "finished_ns": 400,
                "elapsed_ns": 300, "wall_seconds": 300 / 1e9}
    worker = {"status": "completed", "complete": True, "started_ns": 200,
              "finished_ns": 300, "wall_seconds": 100 / 1e9}
    audit.closed_parent(worker, terminal, launch, 600)
    if defect == "timeout":
        terminal["timed_out"] = True
    elif defect == "group_survived":
        terminal["group_absent"] = False
    else:
        worker["finished_ns"] = launch["deadline_ns"] + 1
    with pytest.raises(ValueError):
        audit.closed_parent(worker, terminal, launch, 600)


def test_late_budget_failure_demotes_completed_receipt_and_preserves_original_error(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    args = SimpleNamespace(output=tmp_path / "audit", supervision=tmp_path / "launch.json")

    class FabricatedAudit(audit.Audit):
        def admit(self):
            self.clock = SimpleNamespace(now_ns=lambda: 200)
            self.start, self.launch = 100, {"deadline_ns": 1000}
            self.plan, self.inputs = {"sources": {}}, {}
            self.receipt["supervision_sha256"] = "fake-launch"
            audit.write(self.out / "started.json", {"fixture": True})

        def body(self):
            self.result = {"fixture": True}

        def sha(self, path):
            return "fake-launch" if path == args.supervision else super().sha(path)

        def check(self):
            if (self.out / "receipt.json").exists():
                raise TimeoutError("late fabricated bound")

    with pytest.raises(TimeoutError, match="late fabricated bound"):
        FabricatedAudit(args).execute()
    saved = json.loads((args.output / "receipt.json").read_text())
    invalid = json.loads((args.output / "receipt.invalid.json").read_text())
    assert saved["status"] == "failed" and saved["agreement"] is False
    assert saved["error"] == "TimeoutError('late fabricated bound')"
    assert invalid["status"] == "completed" and invalid["agreement"] is True
