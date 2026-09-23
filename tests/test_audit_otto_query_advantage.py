"""Tiny fabricated saved-record arithmetic and lifecycle tests; no policies."""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_advantage_audit_test", ROOT / "scripts/audit_otto_query_advantage.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def records(differences, *, same=False):
    result = []
    for rep, value in enumerate(differences):
        result.append({"replicate_id": rep, "first_action": 0, "steps": 16 + value, "found": True})
        if not same:
            result.append({"replicate_id": rep, "first_action": 1, "steps": 16, "found": True})
    return result


def panel(first, second=None, *, same=False):
    second = first if second is None else second
    return audit.panel_statistics(records([first] * 8 + [second] * 8, same=same), 0, 0 if same else 1)


def row(episode, anchor, first, second=None, *, same=False):
    return {"episode_id": episode, "anchor_id": anchor, "reduction": panel(first, second, same=same)}


def test_exact_paired_integer_variance_and_sign():
    saved = records([-2, 0, 2, 4])
    reduced = audit.panel_statistics(saved[::-1], 0, 1, replicate_count=4)
    assert reduced["mean_advantage"] == 1
    assert reduced["sample_variance"] == pytest.approx(20 / 3)
    assert reduced["variance_of_mean"] == pytest.approx(5 / 3)
    assert reduced["paired_standard_error"] == pytest.approx(math.sqrt(5 / 3))
    assert [h["mean_advantage"] for h in reduced["halves"]] == [-1, 3]
    assert reduced["split_same_sign"] is False
    assert saved == records([-2, 0, 2, 4])


def test_same_action_is_one_physical_branch_and_exact_zero():
    reduced = panel(0, same=True)
    assert reduced["physical_record_count"] == 16
    assert reduced["structural_zero"] is True
    assert reduced["differences"] == [0] * 16
    assert reduced["sample_variance"] == reduced["mean_advantage"] == 0
    assert reduced["branches"]["analytic"] == reduced["branches"]["neural"]


def test_cap_success_and_censor_are_distinct():
    saved = records([0] * 16)
    saved[0].update(steps=32, found=True)
    saved[1].update(steps=32, found=False)
    reduced = audit.panel_statistics(saved, 0, 1)
    assert reduced["differences"][0] == 0
    assert reduced["branches"]["analytic"]["at_cap_count"] == 1
    assert reduced["branches"]["analytic"]["censored_count"] == 0
    assert reduced["branches"]["neural"]["censored_count"] == 1
    assert reduced["both_censored_count"] == 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "bool_id", "short_censor"])
def test_incomplete_or_malformed_physical_panels_are_refused(defect):
    saved = records([0] * 16)
    if defect == "missing":
        saved.pop()
    elif defect == "duplicate":
        saved.append(saved[0].copy())
    elif defect == "bool_id":
        saved[0]["replicate_id"] = False
    else:
        saved[0]["found"] = False
    with pytest.raises(ValueError):
        audit.panel_statistics(saved, 0, 1)


def test_equal_episode_weights_and_inherited_subset_weights():
    rows = [row("a", 0, 0, same=True), row("a", 1, 2), row("b", 2, 4)]
    result = audit.signal_statistics(rows[::-1], ["b", "a"])
    assert [r["weight"] for r in result["weights"]] == [.25, .25, .5]
    assert result["all"]["statistics"]["cross_half_covariance"] == 2.75
    assert result["all"]["statistics"]["repeatability"] == 1
    different = result["different_action"]
    assert different["original_weight_mass"] == .75
    assert different["statistics"]["half_means"][0] == pytest.approx(10 / 3)
    assert different["statistics"]["cross_half_covariance"] == pytest.approx(8 / 9)


def test_constant_nonzero_halves_have_no_centered_signal():
    result = audit.weighted_statistics([row("a", 0, 3), row("b", 1, 3), row("c", 2, 3)], [1 / 7, 2 / 7, 3 / 7])
    stats = result["statistics"]
    assert stats["half_variances"] == [0., 0.]
    assert stats["cross_half_covariance"] == 0.
    assert stats["repeatability"] is None
    assert stats["second_moment"] > 0


def test_anticorrelated_halves_remain_signed_and_empty_is_explicit():
    stats = audit.weighted_statistics([row("a", 0, -1, 1), row("b", 1, 1, -1)], [.5, .5])["statistics"]
    assert stats["cross_half_covariance"] == -1
    assert stats["repeatability"] == -1
    assert audit.weighted_statistics([], []) == {"anchor_count": 0, "episode_count": 0, "original_weight_mass": 0., "statistics": None}
    with pytest.raises(ValueError, match="all episodes"):
        audit.signal_statistics([row("a", 0, 1)], ["a", "b"])


def cohort_rows(*, sparse=False):
    rows = []
    for identity in audit.cohort():
        value = identity["case"] - 6
        same = sparse and identity["case"] != 0
        rows.append({**identity, **row(identity["episode_id"], identity["episode_index"] * 5,
                                      0 if same else value, same=same)})
    return rows


def test_fixed_cluster_bootstrap_order_covariance_rank_and_eleven_conditions():
    draws = []
    result = audit.bootstrap_admission(cohort_rows(), emit=draws.append)
    rng = random.Random(19400001)
    for i, draw in enumerate(draws):
        assert draw["draw"] == i
        for regime in ("base", "shift"):
            chosen = [rng.randrange(12) for _ in range(12)]
            assert draw["sampled_cases"][regime] == chosen
            values = [case - 6 for case in chosen]
            mean = math.fsum(values) / 12
            variance = math.fsum((x - mean)**2 for x in values) / 12
            assert draw["covariances"][regime]["all"] == pytest.approx(variance)
            assert draw["covariances"][regime]["different_action"] == pytest.approx(variance)
    assert len(draws) == 2000
    for group in ("all", "different_action"):
        assert result["lower_bounds"][group] == sorted(r["pooled_covariances"][group] for r in draws)[199]
    assert result["conditions_total"] == result["conditions_passed"] == 11
    assert result["signal_admitted"] is True


def test_empty_different_resamples_are_kept_at_zero_and_fail_support():
    draws = []
    result = audit.bootstrap_admission(cohort_rows(sparse=True), emit=draws.append)
    assert len(draws) == 2000 and result["signal_admitted"] is False
    for regime in ("base", "shift"):
        empty = [r for r in draws if 0 not in r["sampled_cases"][regime]]
        assert empty and all(r["covariances"][regime]["different_action"] == 0 for r in empty)
        assert result["degenerate_resamples"][regime]["empty_different"] == len(empty)
        assert result["support"][regime] == {"different_anchors": 3, "different_cases": 1}
    assert result["lower_bounds"]["different_action"] == 0


def test_wrong_external_bytes_fail_before_any_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    value = tmp_path / "source.json"
    value.write_text("not JSON")
    runner = audit.Audit(SimpleNamespace(output=tmp_path / "out", run=tmp_path))
    with pytest.raises(ValueError, match="external or frozen"):
        runner.bind(value, "0" * 64)


def test_existing_completion_demoted_on_late_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    out = tmp_path / "audit"
    runner = audit.Audit(SimpleNamespace(output=out, run=tmp_path))

    def fail_after_receipt():
        audit.write(out / "receipt.json", {"status": "completed", "agreement": True})
        raise OSError("fabricated late publication")

    monkeypatch.setattr(runner, "admit", fail_after_receipt)
    with pytest.raises(OSError, match="fabricated late"):
        runner.execute()
    assert json.loads((out / "receipt.invalid.json").read_text())["status"] == "completed"
    failed = json.loads((out / "receipt.json").read_text())
    assert failed["status"] == "failed" and failed["agreement"] is False
    assert "fabricated late publication" in failed["error"]
