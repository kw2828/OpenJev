"""Synthetic probability arithmetic and pre-decode lifecycle checks only.

No scientific prediction arrays, evaluator labels, models or corpus are read.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dialogue_probability_diagnostic", ROOT / "scripts/diagnose_dialogue_probabilities.py"
)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def info_for(logs, target=None):
    rows = len(logs)
    return {
        "rows": rows,
        "target": np.zeros(rows, dtype=np.int64) if target is None else np.asarray(target, dtype=np.int64),
        "mask": np.isfinite(logs),
        "unseen": np.arange(rows) % 2 == 1,
        "strata": np.asarray([
            ("unmentioned_retention", "assigned_retention", "changed")[i % 3] for i in range(rows)
        ]),
    }


def minimal_stats(confidence):
    confidence = np.asarray(confidence, dtype=np.float64)
    return {
        "confidence": confidence,
        "correct": np.arange(len(confidence)) % 2 == 0,
        "nll": np.arange(1, len(confidence) + 1, dtype=np.float64),
    }


def logs_with_target_probability(probabilities):
    return np.log(np.asarray([[p, 0.6 * (1 - p), 0.4 * (1 - p)] for p in probabilities]))


def test_correctness_partition_reconstructs_sum_and_full_group_contributions():
    logs = np.log(np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]))
    stats = d.row_stats(logs, np.asarray([0, 0, 0]))
    selected = np.ones(3, dtype=bool)
    result = d.partition(stats, selected, {
        "correct": stats["correct"], "incorrect": ~stats["correct"],
    })
    expected = -math.log(0.8) - math.log(0.1) - math.log(0.5)
    assert [result[k]["count"] for k in ("correct", "incorrect")] == [2, 1]
    assert math.fsum(v["nll_sum"] for v in result.values()) == pytest.approx(expected)
    assert math.fsum(v["nll_contribution"] for v in result.values()) == pytest.approx(expected / 3)
    assert result["incorrect"]["nll_mean"] == pytest.approx(-math.log(0.1))
    assert result["incorrect"]["nll_contribution"] == pytest.approx(-math.log(0.1) / 3)


@pytest.mark.parametrize("defect", ["missing", "overlap", "outside_selected", "no_partitions"])
def test_partition_rejects_missing_overlapping_or_outside_rows(defect):
    stats = minimal_stats([0.6, 0.7, 0.8])
    selected = np.asarray([True, True, False])
    masks = {"first": np.asarray([True, False, False]), "second": np.asarray([False, True, False])}
    if defect == "missing":
        masks.pop("second")
    elif defect == "overlap":
        masks["second"][0] = True
    elif defect == "outside_selected":
        masks["second"][2] = True
    else:
        masks = {}
    with pytest.raises(ValueError, match="Exhaustive disjoint partition"):
        d.partition(stats, selected, masks)


def test_confidence_boundaries_include_lower_exclude_upper_and_preserve_above_one():
    confidence = np.asarray([0, np.nextafter(0.5, 0), 0.5, 0.7, 0.9, 0.95, 0.99, 1, 1 + 1e-7])
    selected = np.ones(len(confidence), dtype=bool)
    masks = d.confidence_masks(confidence, selected)
    expected = [0, 0, 1, 2, 3, 4, 5, 5, 5]
    for row, bin_index in enumerate(expected):
        assert [bool(mask[row]) for mask in masks.values()] == [i == bin_index for i in range(6)]
    result = d.partition(minimal_stats(confidence), selected, masks)
    assert [v["count"] for v in result.values()] == [2, 1, 1, 1, 1, 3]
    assert result["[0.99,inf)"]["mean_confidence"] == pytest.approx((0.99 + 1 + 1 + 1e-7) / 3)
    assert math.fsum(v["nll_sum"] for v in result.values()) == 45


def test_empty_bins_keep_undefined_means_and_empty_groups_keep_undefined_contributions():
    stats = minimal_stats([0.4, 0.995])
    selected = np.ones(2, dtype=bool)
    result = d.partition(stats, selected, d.confidence_masks(stats["confidence"], selected))
    empty = result["[0.7,0.9)"]
    assert empty["count"] == 0 and empty["nll_sum"] == 0 and empty["nll_contribution"] == 0
    for key in ("nll_mean", "accuracy", "mean_confidence", "confidence_minus_accuracy"):
        assert empty[key] is None
    selected[:] = False
    result = d.partition(stats, selected, d.confidence_masks(stats["confidence"], selected))
    assert all(v["count"] == 0 and v["nll_contribution"] is None for v in result.values())


def test_fit_diagnostic_counts_permitted_raw_confidence_roundoff_without_clipping():
    logs = np.asarray([[math.log1p(1e-7), -1000, -1000]], dtype=np.float64)
    original = logs.copy()
    result, _ = d.fit_diagnostics(logs, info_for(logs))
    group = result["groups"]["all/micro"]
    assert group["confidence_above_one"] == 1
    assert group["confidence_bins"]["[0.99,inf)"]["count"] == 1
    assert group["raw"]["mean_confidence"] > 1
    assert group["raw"]["nll_mean"] == -logs[0, 0]
    assert result["temperature_grid"]["1.0"]["all/micro"]["nll"] == -logs[0, 0]
    np.testing.assert_array_equal(logs, original)


@pytest.mark.parametrize("threshold", d.TAILS)
def test_target_probability_threshold_is_strict_below_in_log_space(threshold):
    boundary = math.log(threshold)
    targets = np.asarray([np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)])
    logs = np.asarray([[x, math.log1p(-math.exp(x)) - math.log(2),
                        math.log1p(-math.exp(x)) - math.log(2)] for x in targets])
    result, _ = d.fit_diagnostics(logs, info_for(logs))
    tail = result["groups"]["all/micro"]["target_probability_tails"][str(threshold)]
    assert tail["count"] == 1
    assert tail["nll_sum"] == -targets[0]
    assert tail["nll_contribution"] == -targets[0] / 3


def test_finite_log_underflow_preserves_nll_and_all_tail_memberships():
    logs = np.asarray([[-1000, math.log(0.75), math.log(0.25), -np.inf]])
    result, stats = d.fit_diagnostics(logs, info_for(logs))
    assert math.exp(logs[0, 0]) == 0
    assert stats["nll"][0] == 1000 and np.isfinite(stats["nll"]).all()
    assert stats["brier"][0] == pytest.approx(1.625)
    group = result["groups"]["all/micro"]
    assert group["raw"]["nll_mean"] == 1000
    assert all(v["count"] == 1 and v["nll_sum"] == 1000
               for v in group["target_probability_tails"].values())
    assert result["temperature_selected"] is None
    assert set(result["temperature_grid"]) == {str(t) for t in d.TEMPERATURES}


@pytest.mark.parametrize("temperature", d.TEMPERATURES)
def test_temperature_grid_preserves_variable_support_and_canonical_first_ties(temperature):
    logs = np.full((4, 4), -np.inf, dtype=np.float32)
    logs[0, :3] = [-math.log(2), -math.log(2), -1000]
    logs[1] = np.log([1e-300, 0.1, 0.2, 0.7])
    logs[2, :3] = np.log([0.8, 0.1, 0.1])
    logs[3] = np.log([0.25, 0.25, 0.25, 0.25])
    original = logs.copy()
    supported = np.isfinite(logs)
    transformed = d.temperature_logs(logs, temperature)
    assert np.isfinite(transformed[supported]).all()
    assert np.isneginf(transformed[~supported]).all()
    np.testing.assert_array_equal(np.argmax(transformed, axis=1), [0, 3, 0, 0])
    tolerance = 2e-6 if temperature == 1 else 1e-12
    np.testing.assert_allclose(np.exp(transformed.astype(np.float64)).sum(axis=1), 1, rtol=0, atol=tolerance)
    if temperature == 1:
        assert transformed is not logs and transformed.dtype == logs.dtype
        np.testing.assert_array_equal(transformed, logs)
        target = np.asarray([2, 3, 1, 0])
        np.testing.assert_array_equal(d.row_stats(transformed, target)["nll"], d.row_stats(logs, target)["nll"])
    np.testing.assert_array_equal(logs, original)


@pytest.mark.parametrize("temperature", [0, -1, np.inf, -np.inf, np.nan])
def test_temperature_rejects_nonpositive_and_nonfinite_values(temperature):
    with pytest.raises(ValueError, match="Positive finite temperature"):
        d.temperature_logs(np.log([[0.5, 0.3, 0.2]]), temperature)


def test_all_four_correctness_transitions_reconstruct_a_negative_paired_gap():
    control_logs = np.log([[0.1, 0.7, 0.2], [0.1, 0.6, 0.3], [0.8, 0.15, 0.05], [0.6, 0.25, 0.15]])
    treatment_logs = np.log([[0.3, 0.5, 0.2], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.9, 0.05, 0.05]])
    info = info_for(control_logs)
    control, treatment = (d.row_stats(logs, info["target"]) for logs in (control_logs, treatment_logs))
    result = d.paired_diagnostics(control, treatment, info)
    transitions = result["all/micro"]["correctness_transitions"]
    assert set(transitions) == {"False_to_False", "False_to_True", "True_to_False", "True_to_True"}
    assert all(v["count"] == 1 for v in transitions.values())
    assert transitions["True_to_False"]["signed_gap_contribution"] > 0
    expected = math.fsum(math.log(a / b) for a, b in zip([0.1, 0.1, 0.8, 0.6], [0.3, 0.7, 0.2, 0.9])) / 4
    assert result["all/micro"]["raw"]["signed_gap_contribution"] == pytest.approx(expected)
    assert expected < 0
    assert set(result) == set(d.group_masks(info)) and len(result) == 12
    for group in result.values():
        raw = group["raw"]
        parts = group["correctness_transitions"].values()
        assert sum(v["count"] for v in parts) == raw["count"]
        assert math.fsum(v["signed_gap_sum"] for v in parts) == pytest.approx(raw["signed_gap_sum"])
        if raw["count"]:
            assert math.fsum(v["signed_gap_contribution"] for v in parts) == pytest.approx(
                raw["signed_gap_contribution"]
            )


def test_paired_tail_union_uses_identical_rows_and_complement_reconstructs_gap():
    control_logs = logs_with_target_probability([0.001, 0.8, 0.9])
    treatment_logs = logs_with_target_probability([0.9, 0.002, 0.8])
    info = info_for(control_logs)
    control, treatment = (d.row_stats(logs, info["target"]) for logs in (control_logs, treatment_logs))
    result = d.paired_diagnostics(control, treatment, info)["all/micro"]
    tail = result["common_target_probability_tails"]["0.01"]["either_below"]
    assert tail["count"] == 2
    assert tail["control"]["nll_sum"] == pytest.approx(-math.log(0.001) - math.log(0.8))
    assert tail["treatment"]["nll_sum"] == pytest.approx(-math.log(0.9) - math.log(0.002))
    for parts in result["common_target_probability_tails"].values():
        assert sum(v["count"] for v in parts.values()) == 3
        assert math.fsum(v["signed_gap_contribution"] for v in parts.values()) == pytest.approx(
            result["raw"]["signed_gap_contribution"]
        )


def test_bad_plan_pin_fails_before_json_authentication_or_prediction_decode(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text("Not JSON: pin rejection must happen before reading this.\n")
    out = tmp_path / "diagnostic"
    clock_type = d.SuspendClock
    ticks = itertools.count(0, 1_000_000)
    monkeypatch.setattr(d, "SuspendClock", lambda: clock_type(lambda: next(ticks)))
    monkeypatch.setattr(d.signal, "signal", lambda *_: 0)
    monkeypatch.setattr(d.signal, "setitimer", lambda *_: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unauthenticated input was accessed")

    monkeypatch.setattr(d, "read", forbidden)
    monkeypatch.setattr(d, "authenticate", forbidden)
    monkeypatch.setattr(d.np, "load", forbidden)
    args = SimpleNamespace(plan=plan, plan_sha256="0" * 64, out=out)
    with pytest.raises(ValueError, match="Diagnostic plan pin"):
        d.execute(args)
    assert {p.name for p in out.iterdir()} == {"failed.json"}
    failed = json.loads((out / "failed.json").read_text())
    assert failed["status"] == "failed" and failed["model_calls"] == 0
    assert "Diagnostic plan pin" in failed["error"]
    assert plan.read_text() == "Not JSON: pin rejection must happen before reading this.\n"


def test_existing_output_is_untouched_and_rejected_before_execution(tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "receipt.json").write_bytes(b"Preserve existing receipt exactly.\n")
    (out / "nested").mkdir()
    (out / "nested" / "evidence.bin").write_bytes(b"\x00\xffexisting")
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Existing output reached execution")

    monkeypatch.setattr(d, "SuspendClock", forbidden)
    monkeypatch.setattr(d, "authenticate", forbidden)
    monkeypatch.setattr(d.np, "load", forbidden)
    with pytest.raises(FileExistsError):
        d.execute(SimpleNamespace(plan=tmp_path / "absent.json", plan_sha256="0" * 64, out=out))
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert after == before


INPUT_ROLES = (
    "completion", "scientific_plan", "launch", "terminal", "producer_receipt",
    "producer_summary", "audit_receipt", "audit_summary", "exit_review",
)


@pytest.fixture
def synthetic_authentication_plan(tmp_path, monkeypatch):
    """Real hashes of artificial bytes; imports and decoded reads are barriers."""
    root = tmp_path / "synthetic-repo"
    root.mkdir()
    monkeypatch.setattr(d, "ROOT", root)

    def pinned_file(relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Artificial metadata bytes only: " + relative + "\n")
        return d.sha(path)

    names = {
        "completion": "synthetic/run/completed.json",
        "scientific_plan": "synthetic/freeze/plan.json",
        "launch": "synthetic/process.launch.json",
        "terminal": "synthetic/process.terminal.json",
        "producer_receipt": "synthetic/report/receipt.json",
        "producer_summary": "synthetic/report/summary.json",
        "audit_receipt": "synthetic/audit/receipt.json",
        "audit_summary": "synthetic/audit/summary.json",
        "exit_review": "synthetic/actual-exit-review.json",
    }
    inputs = {relative: pinned_file(relative) for relative in names.values()}
    sources = {relative: pinned_file(relative) for relative in sorted(d.DIAGNOSTIC_SOURCES)}
    monkeypatch.setattr(d, "AUDITOR_PIN", pinned_file(d.AUDITOR))
    request = {
        "run": "synthetic/run", "completed_sha256": inputs[names["completion"]],
        "plan": names["scientific_plan"], "plan_sha256": inputs[names["scientific_plan"]],
        "launch": names["launch"], "launch_sha256": inputs[names["launch"]],
        "terminal": names["terminal"], "terminal_sha256": inputs[names["terminal"]],
        "report": "synthetic/report", "report_receipt_sha256": inputs[names["producer_receipt"]],
        "report_summary_sha256": inputs[names["producer_summary"]],
    }
    plan = {
        "sources": sources, "inputs": inputs, "audit_request": request,
        "independent_audit": "synthetic/audit",
        "independent_audit_receipt_sha256": inputs[names["audit_receipt"]],
        "independent_audit_summary_sha256": inputs[names["audit_summary"]],
        "actual_exit_review": names["exit_review"],
        "actual_exit_review_sha256": inputs[names["exit_review"]],
        "environment": {"python": d.platform.python_version(), "numpy": np.__version__},
    }

    def import_barrier(*_args, **_kwargs):
        raise AssertionError("Synthetic qualified-auditor import boundary")

    def decode_barrier(*_args, **_kwargs):
        raise AssertionError("Unauthenticated scientific input was decoded")

    monkeypatch.setattr(d, "importlib", SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=import_barrier, module_from_spec=import_barrier,
    )))
    monkeypatch.setattr(d, "read", decode_barrier)
    monkeypatch.setattr(d.np, "load", decode_barrier)
    return plan, names


def test_complete_synthetic_authentication_passes_pin_checks_before_import_barrier(synthetic_authentication_plan):
    plan, _ = synthetic_authentication_plan
    assert len(plan["inputs"]) == 9
    with pytest.raises(AssertionError, match="Synthetic qualified-auditor import boundary"):
        d.authenticate(plan)


@pytest.mark.parametrize("role", INPUT_ROLES)
def test_authentication_rejects_each_missing_external_pin_before_decode(synthetic_authentication_plan, role):
    plan, names = synthetic_authentication_plan
    del plan["inputs"][names[role]]
    with pytest.raises(ValueError, match="Complete exact external input bindings"):
        d.authenticate(plan)


def test_authentication_rejects_extraneous_external_pin_before_decode(synthetic_authentication_plan):
    plan, _ = synthetic_authentication_plan
    plan["inputs"]["synthetic/unrequested.json"] = "0" * 64
    with pytest.raises(ValueError, match="Complete exact external input bindings"):
        d.authenticate(plan)


@pytest.mark.parametrize("source", sorted(d.DIAGNOSTIC_SOURCES))
def test_authentication_rejects_each_missing_source_before_decode(synthetic_authentication_plan, source):
    plan, _ = synthetic_authentication_plan
    del plan["sources"][source]
    with pytest.raises(ValueError, match="Complete diagnostic source closure"):
        d.authenticate(plan)


def test_authentication_rejects_extraneous_source_before_decode(synthetic_authentication_plan):
    plan, _ = synthetic_authentication_plan
    plan["sources"]["synthetic/unrequested.py"] = "0" * 64
    with pytest.raises(ValueError, match="Complete diagnostic source closure"):
        d.authenticate(plan)


@pytest.mark.parametrize("source", sorted(d.DIAGNOSTIC_SOURCES))
def test_authentication_rejects_each_wrong_source_hash_before_decode(synthetic_authentication_plan, source):
    plan, _ = synthetic_authentication_plan
    plan["sources"][source] = "0" * 64
    with pytest.raises(ValueError, match="Diagnostic source changed"):
        d.authenticate(plan)
