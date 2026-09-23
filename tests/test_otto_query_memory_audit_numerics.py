"""Independent fabricated numerical audit and timing contracts; no empirical IO."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from openjev.research import otto_query_memory_gate as producer_gate
from openjev.research import otto_query_memory_metrics as producer_metrics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_independent_query_memory_audit_numerics", ROOT / "scripts/audit_otto_query_memory.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def ids(stage="dev", count=None):
    result = []
    cases = 3 if stage == "dev" else 9
    starts = (305000001, 306000001) if stage == "dev" else (303000001, 304000001)
    for regime, start in zip(("lambda3", "lambda4"), starts, strict=True):
        for case in range(cases):
            for arm in ("analytic", "neural", "period4_hold"):
                result.append({"stage": stage, "episode_id": f"{stage}:{regime}:{start + case}:{arm}",
                    "episode_index": len(result), "seed": start + case, "case": case, "regime": regime, "arm": arm})
    return result if count is None else result[:count]


def fixture(lengths=(17, 9, 1), *, stage="train"):
    identities = ids(stage, len(lengths))
    offsets = np.array([0, *np.cumsum(lengths)], np.int64)
    total = int(offsets[-1])
    q = np.tile(np.array([1, 0, 2, 3], np.float32), (total, 1))
    legal = np.ones((total, 4), np.bool_)
    legal[1::4, 3] = False
    features = np.zeros((total, 31), np.float32)
    correction = np.zeros(total, np.bool_)
    for low, high in pairwise(offsets):
        steps = np.arange(high - low)
        features[low:high, 15] = steps / 2188
        features[low:high, 16] = steps % 4 / 2188
        features[low:high, 17] = 1
        correction[low:high] = steps % 4 == 0
    flat = {"features": features, "raw_q": q, "legal": legal, "actions": np.zeros(total, np.int64),
            "correction": correction, "episode_offsets": offsets}
    predicted = np.zeros_like(q)
    prior = np.full_like(q, 2)
    return identities, flat, predicted, prior


def producer_report(identities, flat, predicted, prior, family="trace_delta", seed=309000001):
    episodes = []
    for identity, low, high in zip(identities, flat["episode_offsets"][:-1], flat["episode_offsets"][1:], strict=True):
        episodes.append(producer_metrics.episode_metrics(identity, 4, flat["raw_q"][low:high],
            flat["legal"][low:high], predicted[low:high], prequery_forecast=prior[low:high]))
    return producer_metrics.aggregate_episodes(episodes, family=family, fit_seed=seed, expected_identities=identities)


@pytest.mark.parametrize("stage", ("train", "dev"))
def test_independent_scalar_report_matches_every_field_without_producer_imports_in_auditor(stage):
    identities, flat, predicted, prior = fixture(stage=stage)
    expected = producer_report(identities, flat, predicted, prior)
    actual = audit.scalar_report(np, identities, flat["raw_q"], flat["legal"], predicted, prior,
                                flat["episode_offsets"], "trace_delta", 309000001)
    audit.close_equal(actual, expected, "entire fabricated independent report")
    assert actual["scopes"]["full"]["overall"]["episodes"] == 3
    assert actual["scopes"]["full"]["overall"]["supported_episodes"] == 2
    assert actual["scopes"]["later"]["overall"]["zero_support_episode_ids"] == [identities[2]["episode_id"]]
    source = (ROOT / "scripts/audit_otto_query_memory.py").read_text()
    assert "import torch" not in source
    assert "import otto_query_memory_metrics" not in source
    assert "import otto_query_memory_gate" not in source


def test_independent_legal_tie_threshold_and_large_float32_subtraction():
    q = np.array([-100, 5e-11, -100, 0], np.float32)
    legal = np.array([False, True, False, True])
    predicted = q.copy()
    first = audit.scalar_row(q, legal, predicted)
    assert first[:3] == (1., float(np.float32(5e-11)), 1.)
    predicted[1] = np.float32(1e-10)
    assert audit.scalar_row(q, legal, predicted)[:3] == (1., 0., 0.)
    q = np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max, 0, 0], np.float32)
    result = audit.scalar_row(q, np.ones(4, np.bool_), np.zeros(4, np.float32))
    assert result[1] == 2 * float(np.finfo(np.float32).max) and math.isfinite(result[3])


def test_independent_scope_weights_and_raw_sums_are_hand_calculated():
    identities, flat, predicted, prior = fixture((6, 1))
    flat["legal"][:] = True
    result = audit.scalar_report(np, identities, flat["raw_q"], flat["legal"], predicted, prior,
                                flat["episode_offsets"], "trace_delta", 309000001)
    value = result["scopes"]["full"]["overall"]
    assert value["raw_sums"]["raw_gap"] == 4
    assert value["episode_weighted_raw_gap"] == .5
    assert value["supported_episode_raw_gap"] == value["row_weighted_raw_gap"] == 1
    assert value["episode_weighted_centered_mse"] == .625
    assert result["scopes"]["later"]["overall"]["by_age"]["2"]["supported_episode_raw_gap"] is None


def gate_reports():
    identities, flat, predicted, prior = fixture((9,) * 18, stage="dev")
    reports = []
    for view in audit.VIEWS:
        for seed in audit.SEEDS:
            predicted[:] = 1
            predicted[:, 1 if view == "trace_delta" else 0] = 0
            reports.append(audit.scalar_report(np, identities, flat["raw_q"], flat["legal"], predicted, prior,
                flat["episode_offsets"], view, seed))
    return reports, identities


def test_independent_gate_exactly_matches_producer_and_keeps_technical_false_default():
    reports, identities = gate_reports()
    for technical in (False, True):
        actual = audit.independent_gate(reports, technical_complete=technical)
        expected = producer_gate.continuation_gate(reports, identities, technical_complete=technical)
        audit.close_equal(actual, expected, "independent gate")
        assert actual["passed"] is technical
        assert actual["total_conditions"] == 13


def test_independent_gate_uses_minimum_control_means_not_mean_of_seed_minima():
    reports, identities = gate_reports()
    for report in reports:
        seed_index = audit.SEEDS.index(report["seed"])
        for regime in ("lambda3", "lambda4"):
            value = report["scopes"]["later"]["by_regime"][regime]
            if report["family"] == "trace_delta":
                gap = .85
            elif report["family"] == "pretrained":
                gap = (.8, 2., 2.)[seed_index]
            elif report["family"] == "joint_aux":
                gap = (2., .8, 2.)[seed_index]
            elif report["family"] == "last_error":
                gap = (2., 2., .8)[seed_index]
            else:
                gap = 2.
            value["case_weighted_raw_gap"] = gap
    actual = audit.independent_gate(reports, technical_complete=True)
    expected = producer_gate.continuation_gate(reports, identities, technical_complete=True)
    audit.close_equal(actual, expected, "crossing control means")
    assert all(row["passed"] for row in actual["conditions"] if row["name"].endswith("later_gap_10pct"))
    assert sum(not row["passed"] for row in actual["conditions"]) == 6


@pytest.mark.parametrize("candidate,passes", ((9., True), (math.nextafter(9., math.inf), False)))
def test_independent_ninety_percent_boundary_has_no_epsilon_rescue(candidate, passes):
    reports, _ = gate_reports()
    for report in reports:
        for regime in ("lambda3", "lambda4"):
            report["scopes"]["later"]["by_regime"][regime]["case_weighted_raw_gap"] = (
                candidate if report["family"] == "trace_delta" else 10.)
    result = audit.independent_gate(reports, technical_complete=True)
    assert all(row["passed"] is passes for row in result["conditions"] if row["name"].endswith("later_gap_10pct"))


def test_zero_best_control_and_one_case_support_each_block_continuation():
    reports, _ = gate_reports()
    for report in reports:
        for regime in ("lambda3", "lambda4"):
            value = report["scopes"]["later"]["by_regime"][regime]
            value["supported_case_count"] = 1
            if report["family"] == "pretrained":
                value["case_weighted_raw_gap"] = 0.
    result = audit.independent_gate(reports, technical_complete=True)
    assert result["passed"] is False
    relevant = [row for row in result["conditions"] if row["name"].endswith(("later_gap_10pct", "supported_cases"))]
    assert len(relevant) == 4 and not any(row["passed"] for row in relevant)


@pytest.mark.parametrize("defect", ("missing", "duplicate", "wrong_stage", "wrong_period", "nonfinite", "support"))
def test_independent_gate_rejects_incomplete_or_inconsistent_reports(defect):
    reports, _ = gate_reports()
    if defect == "missing":
        reports.pop()
    elif defect == "duplicate":
        reports[-1] = reports[0]
    elif defect == "wrong_stage":
        reports[0]["stage"] = "test"
    elif defect == "wrong_period":
        reports[0]["query_period"] = 8
    elif defect == "nonfinite":
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = math.nan
    else:
        reports[0]["scopes"]["later"]["by_regime"]["lambda3"]["supported_case_count"] -= 1
    with pytest.raises(ValueError):
        audit.independent_gate(reports, technical_complete=True)


def test_independent_history_matches_observation_masks_and_fixed_zero_support_weights():
    identities, flat, _, _ = fixture()
    result = audit.expected_history(np, flat, identities)
    assert result["query_mask"].sum() == 9
    assert result["prior_mask"].sum() == 6
    assert np.isnan(result["query_scores"][~result["query_mask"]]).all()
    assert result["nonquery_weights"].sum() == pytest.approx(2 / 3)
    assert result["prior_weights"].sum() == pytest.approx(2 / 3)
    assert result["nonquery_weights"][-1] == result["prior_weights"][-1] == 0
    broken = {name: value.copy() for name, value in flat.items()}
    broken["features"][4, 16] = 4 / 2188
    with pytest.raises(ValueError, match="clocks"):
        audit.expected_history(np, broken, identities)


def saved_fixture(view="pretrained"):
    identities, flat, _, _ = fixture((9, 1))
    history = audit.expected_history(np, flat, identities)
    saved = {name: np.zeros_like(flat["raw_q"]) for name in audit.PREDICTION_FIELDS
             if name not in ("episode_offsets", "prior_mask")}
    for name in ("action_prediction", "slow_action_prediction", "base_prediction"):
        saved[name][:] = flat["raw_q"]
    for name in ("shadow_prior", "corrected_shadow_prior"):
        saved[name][history["prior_mask"]] = flat["raw_q"][history["prior_mask"]]
    saved["episode_offsets"] = flat["episode_offsets"].copy()
    saved["prior_mask"] = history["prior_mask"].copy()
    if view not in ("pretrained", "joint_aux", "trace_no_write"):
        saved["prewrite_correction"][1:9] = np.array([.25, -.25, .5, -.5], np.float32)
        nonquery = ~history["query_mask"]
        prior = history["prior_mask"]
        saved["action_prediction"][nonquery] += np.float32(64) * saved["prewrite_correction"][nonquery]
        saved["corrected_shadow_prior"][prior] += np.float32(64) * saved["prewrite_correction"][prior]
    return history, saved


@pytest.mark.parametrize("view", audit.VIEWS)
def test_saved_prediction_algebra_and_frozen_reference_bytes(view):
    history, saved = saved_fixture(view)
    _, reference = saved_fixture()
    audit.verify_prediction(np, saved, history, view, reference if view != "pretrained" else None)


@pytest.mark.parametrize("field", ("base_prediction", "slow_action_prediction", "shadow_prior"))
def test_frozen_reference_drift_cannot_pass_even_if_correction_algebra_is_preserved(field):
    history, saved = saved_fixture("trace_delta")
    _, reference = saved_fixture()
    step = 4 if field == "shadow_prior" else 1
    saved[field][step, 0] += 1
    if field == "slow_action_prediction":
        saved["action_prediction"][step, 0] += 1
    elif field == "shadow_prior":
        saved["corrected_shadow_prior"][step, 0] += 1
    with pytest.raises(ValueError, match="frozen/no-write"):
        audit.verify_prediction(np, saved, history, "trace_delta", reference)


def test_signed_zero_is_preserved_only_for_no_write_copy_not_generic_addition():
    history, saved = saved_fixture("trace_no_write")
    saved["slow_action_prediction"][1, 0] = np.float32(-0.)
    saved["action_prediction"][1, 0] = np.float32(-0.)
    audit.verify_prediction(np, saved, history, "trace_no_write")
    saved["action_prediction"][1, 0] = np.float32(0.)
    with pytest.raises(ValueError, match="exact baseline"):
        audit.verify_prediction(np, saved, history, "trace_no_write")
    audit.verify_prediction(np, saved, history, "trace_delta")


@pytest.mark.parametrize("defect", ("query", "prior_mask", "inactive_prior", "first_write", "correction", "offset", "nan"))
def test_saved_prediction_defects_fail_closed(defect):
    history, saved = saved_fixture("trace_delta")
    if defect == "query":
        saved["action_prediction"][0, 0] += 1
    elif defect == "prior_mask":
        saved["prior_mask"][0] = True
    elif defect == "inactive_prior":
        saved["shadow_prior"][1, 0] = 1
    elif defect == "first_write":
        saved["prewrite_correction"][0, 0] = 1
    elif defect == "correction":
        saved["action_prediction"][1, 0] += 1
    elif defect == "offset":
        saved["episode_offsets"][1] -= 1
    else:
        saved["shadow_prior"][4, 0] = np.nan
    with pytest.raises(ValueError):
        audit.verify_prediction(np, saved, history, "trace_delta")


def test_exact_scalar_comparison_never_loosens_gap_or_accepts_nonfinite_values():
    with pytest.raises(ValueError):
        audit.close_equal({"gap": .9}, {"gap": math.nextafter(.9, math.inf)}, "exact gap")
    with pytest.raises(ValueError):
        audit.close_equal({"gap": math.nan}, {"gap": math.nan}, "finite gap")
    with pytest.raises(ValueError):
        audit.close_equal({"count": True}, {"count": 1}, "strict type")


def test_scalar_capacity_bound_is_measured_on_synthetic_long_episodes(tmp_path):
    """One timed fabricated 6x2188 view, without empirical input or timing retry."""
    identities, flat, predicted, prior = fixture((2188,) * 6)
    before = time.perf_counter()
    result = audit.scalar_report(np, identities, flat["raw_q"], flat["legal"], predicted, prior,
                                flat["episode_offsets"], "trace_delta", 309000001)
    elapsed = time.perf_counter() - before
    rows, worst_rows = 6 * 2188, 3780864
    projected = 2 * elapsed * worst_rows / rows
    metadata = {"kind": "fabricated_scalar_audit_capacity", "empirical_reads": 0, "model_calls": 0,
                "measured_rows": rows, "measured_seconds": elapsed, "worst_case_rows": worst_rows,
                "multiplier": 2, "reserve_seconds": 120, "projected_seconds": projected,
                "threshold_seconds": 480, "audit_cap_seconds": 600,
                "projected_total_seconds": projected + 120, "admitted": projected <= 480}
    with (tmp_path / "audit-capacity.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    assert result["episodes"] == 6 and result["scopes"]["full"]["overall"]["rows"] == 6 * 1641
    assert elapsed > 0 and metadata["admitted"], metadata
