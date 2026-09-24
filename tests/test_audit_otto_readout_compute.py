"""Independent fabricated audit boundaries; no empirical data or model calls."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readout_compute_audit_tests", ROOT / "scripts/audit_otto_readout_compute.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def reports(joint=19., control=20., support=6):
    return [{"family": family, "seed": seed, "stage": "dev", "query_period": 4, "episodes": 36,
             "scopes": {scope: {"by_regime": {regime: {
                 "episodes": 18, "declared_case_count": 6, "supported_case_count": support,
                 "case_weighted_raw_gap": (joint if family == "full_joint" else control) if scope == "later" else 10.,
             } for regime in audit.REGIMES}} for scope in ("full", "later")}}
            for seed in audit.SEEDS for family in audit.VIEWS]


def fit_records(ratio=1.):
    return [{"family": family, "seed": seed, "seconds": ratio if family == "action_residual_only" else 1.}
            for seed in audit.SEEDS for family in audit.ARMS]


def gap(rows, family, *, scope="later", seed=audit.SEEDS[0], regime="lambda3"):
    row = next(row for row in rows if row["family"] == family and row["seed"] == seed)
    return row["scopes"][scope]["by_regime"][regime]


def test_exact_five_percent_boundary_all_six_cells_and_order_independence():
    rows = reports()
    result = audit.comparisons(rows, fit_records())
    assert result == audit.comparisons(list(reversed(rows)), fit_records())
    assert result["candidate"] == "full_joint" and result["margin"] == .05
    assert result["passed_cells"] == result["total_cells"] == 6
    assert result["overall_passed"] is True
    assert [(row["seed"], row["regime"]) for row in result["cells"]] == [
        (seed, regime) for seed in audit.SEEDS for regime in audit.REGIMES]
    assert all(all(row["later_gain_at_least_5pct"].values()) for row in result["cells"])
    assert result["held_out_evidence"] is result["statistical_equivalence"] is False
    assert result["test_admitted"] is result["confirmation_admitted"] is False
    gap(rows, "full_joint")["case_weighted_raw_gap"] = float(np.nextafter(19., np.inf))
    failed = audit.comparisons(rows, fit_records())
    assert failed["passed_cells"] == 5 and failed["overall_passed"] is False


@pytest.mark.parametrize("family", ("pretrained", "action_residual_only"))
@pytest.mark.parametrize("scope", ("later", "full"))
def test_each_control_and_each_gap_condition_is_required(family, scope):
    rows = reports()
    gap(rows, family, scope=scope)["case_weighted_raw_gap"] = 18. if scope == "later" else 9.
    result = audit.comparisons(rows, fit_records())
    assert result["passed_cells"] == 5 and result["overall_passed"] is False
    cell = result["cells"][0]
    key = "later_gain_at_least_5pct" if scope == "later" else "full_gap_nonregression"
    assert cell[key][family] is False
    assert all(value for name, value in cell[key].items() if name != family)


def test_zero_support_retained_and_zero_against_zero_never_counts_as_gain():
    result = audit.comparisons(reports(joint=0., control=0., support=0), fit_records())
    assert len(result["cells"]) == 6 and result["passed_cells"] == 0
    assert result["overall_passed"] is False
    for row in result["cells"]:
        assert row["supported_later_cases"] == 0
        assert not any(row["later_gain_at_least_5pct"].values())
        for proximity in row["readout_proximity_descriptive_only"].values():
            assert proximity["relative_excess"] is None and proximity["zero_joint_gap"] is True


def test_readout_proximity_is_descriptive_without_equivalence_or_continuation():
    result = audit.comparisons(reports(joint=100., control=104.), fit_records())
    assert result["passed_cells"] == 0 and result["statistical_equivalence"] is False
    for row in result["cells"]:
        for proximity in row["readout_proximity_descriptive_only"].values():
            assert proximity["no_more_than_5pct_worse"] is True
            assert proximity["relative_excess"] == pytest.approx(.04)


@pytest.mark.parametrize("ratio", (.9, 1., 1.1))
def test_paired_time_bounds_are_inclusive_and_keep_each_seed(ratio):
    fits = fit_records(ratio)
    result = audit.compute_comparability(fits)
    assert result == audit.compute_comparability(list(reversed(fits)))
    assert result["bounds"] == [.9, 1.1] and result["passed"] is True
    assert [row["seed"] for row in result["pairs"]] == list(audit.SEEDS)
    assert all(row["ratio"] == ratio and row["passed"] for row in result["pairs"])


@pytest.mark.parametrize("ratio", (float(np.nextafter(.9, 0.)), float(np.nextafter(1.1, np.inf))))
def test_one_time_pair_fails_without_hiding_efficacy(ratio):
    fits = fit_records()
    fits[0]["seconds"] = ratio
    result = audit.comparisons(reports(), fits)
    assert result["efficacy_passed"] is True and result["passed_cells"] == 6
    assert result["compute_comparability"]["passed"] is False and result["overall_passed"] is False
    assert [row["passed"] for row in result["compute_comparability"]["pairs"]] == [False, True, True]


def test_comparability_does_not_rescue_failed_efficacy_and_freshness_is_explicit():
    rows = reports()
    gap(rows, "pretrained")["case_weighted_raw_gap"] = 18.
    result = audit.comparisons(rows, fit_records())
    assert result["compute_comparability"]["passed"] is True
    assert result["efficacy_passed"] is result["overall_passed"] is False
    assert result["held_out_from_training"] is result["fresh_dev"] is result["development_only"] is True
    assert result["held_out_evidence"] is result["confirmation_admitted"] is result["test_admitted"] is False


@pytest.mark.parametrize("value", (0., -.1, float("nan"), float("inf"), True))
def test_nonpositive_or_invalid_fit_time_is_a_technical_failure(value):
    fits = fit_records()
    fits[1]["seconds"] = value
    with pytest.raises(ValueError, match="positive continuation time"):
        audit.compute_comparability(fits)


@pytest.mark.parametrize("defect", ("missing", "duplicate", "seed", "family"))
def test_complete_six_fit_timing_roster_required(defect):
    fits = fit_records()
    if defect == "missing":
        fits.pop()
    elif defect == "duplicate":
        fits[-1] = fits[0].copy()
    elif defect == "seed":
        fits[0]["seed"] = 123
    else:
        fits[0]["family"] = "both_readouts"
    with pytest.raises(ValueError, match="all six timed continuation fits"):
        audit.compute_comparability(fits)


def test_literal_calibrated_budget_and_closed_payload_roster():
    assert audit.EPOCHS == {"action_residual_only": 74, "full_joint": 40}
    assert sum(9 * audit.EPOCHS[family] for _ in audit.SEEDS for family in audit.ARMS) == 3078
    assert sum(54 * audit.EPOCHS[family] for _ in audit.SEEDS for family in audit.ARMS) == 18468
    assert len(audit.PAYLOADS) == 27
    assert sum(name.startswith("checkpoint-") for name in audit.PAYLOADS) == 6
    assert sum(name.startswith("prediction-") for name in audit.PAYLOADS) == 9
    assert not any("test" in name or "confirm" in name for name in audit.PAYLOADS)


@pytest.mark.parametrize("defect", ("missing", "duplicate", "unknown_family", "unknown_seed",
                                   "stage", "period", "episodes", "case_count", "support"))
def test_incomplete_or_malformed_comparison_roster_rejected(defect):
    rows = reports()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif defect in ("unknown_family", "unknown_seed"):
        rows[0]["family" if defect == "unknown_family" else "seed"] = "unregistered"
    elif defect in ("stage", "period", "episodes"):
        key, value = {"stage": ("stage", "test"), "period": ("query_period", 8), "episodes": ("episodes", 17)}[defect]
        rows[0][key] = value
    elif defect == "case_count":
        gap(rows, "pretrained")["declared_case_count"] = 2
    else:
        gap(rows, "pretrained")["supported_case_count"] = 2
    with pytest.raises(ValueError):
        audit.comparisons(rows, fit_records())


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -.1, True))
def test_diagnostic_gap_must_be_finite_nonnegative_number(value):
    rows = reports()
    gap(rows, "full_joint")["case_weighted_raw_gap"] = value
    with pytest.raises(ValueError, match="finite diagnostic gap"):
        audit.comparisons(rows, fit_records())


def prediction():
    query = np.array([True, False, False, False, True, True, False, False, False])
    prior = np.array([False, False, False, False, True, False, False, False, False])
    scores = np.full((9, 4), np.nan, dtype=np.float32)
    scores[query] = np.array([[-.0, 1., 2., 3.], [4., 3., 2., 1.], [1., 4., 2., 3.]], dtype=np.float32)
    action = np.full((9, 4), .25, dtype=np.float32)
    action[query] = scores[query]
    shadow = np.zeros((9, 4), dtype=np.float32)
    shadow[prior] = np.array([.1, -.2, .3, -.4], dtype=np.float32)
    offsets = np.array([0, 5, 9], dtype=np.int64)
    saved = {"action_prediction": action, "corrected_shadow_prior": shadow,
             "prior_mask": prior.copy(), "episode_offsets": offsets.copy()}
    history = {"targets": np.zeros((9, 4), dtype=np.float32), "query_scores": scores,
               "query_mask": query, "prior_mask": prior, "episode_offsets": offsets}
    return saved, history


def test_prediction_query_bytes_and_inactive_positive_zero_without_mutation():
    saved, history = prediction()
    before = {name: value.tobytes() for name, value in saved.items()}
    audit.verify_prediction(np, saved, history)
    assert {name: value.tobytes() for name, value in saved.items()} == before
    assert np.signbit(saved["action_prediction"][0, 0])


@pytest.mark.parametrize("defect", ("query_signed_zero", "inactive_signed_zero", "mask", "offsets",
                                   "dtype", "shape", "nonfinite", "extra"))
def test_prediction_invariants_fail_closed(defect):
    saved, history = prediction()
    if defect == "query_signed_zero":
        saved["action_prediction"][0, 0] = 0.
    elif defect == "inactive_signed_zero":
        saved["corrected_shadow_prior"][0, 0] = -.0
    elif defect == "mask":
        saved["prior_mask"][0] = True
    elif defect == "offsets":
        saved["episode_offsets"][1] = 4
    elif defect == "dtype":
        saved["action_prediction"] = saved["action_prediction"].astype(np.float64)
    elif defect == "shape":
        saved["action_prediction"] = saved["action_prediction"][:-1]
    elif defect == "nonfinite":
        saved["action_prediction"][1, 0] = np.nan
    else:
        saved["unexpected"] = np.zeros(1)
    with pytest.raises(ValueError):
        audit.verify_prediction(np, saved, history)


@pytest.mark.parametrize("family,count,prefixes", (
    ("action_residual_only", 116, ("slow.action_residual.",)),
    ("full_joint", 6112, ("slow.",)),
))
def test_independent_parameter_masks_and_checkpoint_geometry(family, count, prefixes):
    metadata = audit.parameter_metadata(family)
    expected = [name for name in audit.SHAPES if name.startswith(prefixes)]
    assert len(metadata["names"]) == 8 and metadata["count"] == 6112
    assert metadata["effective_names"] == metadata["requires_grad_names"] == expected
    assert metadata["effective_count"] == metadata["requires_grad_count"] == count
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in audit.SHAPES.items()}
    assert sum(value.size for value in arrays.values()) == 6112
    audit.validate_checkpoint(np, arrays)


@pytest.mark.parametrize("defect", ("missing", "extra", "shape", "dtype", "nonfinite"))
def test_malformed_checkpoint_rejected(defect):
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in audit.SHAPES.items()}
    name = "slow.output.bias"
    if defect == "missing":
        arrays.pop(name)
    elif defect == "extra":
        arrays["unregistered"] = np.zeros(1, dtype=np.float32)
    elif defect == "shape":
        arrays[name] = np.zeros(5, dtype=np.float32)
    elif defect == "dtype":
        arrays[name] = arrays[name].astype(np.float64)
    else:
        arrays[name][0] = np.inf
    with pytest.raises(ValueError):
        audit.validate_checkpoint(np, arrays)


def test_incomplete_producer_roster_rejected_before_any_array_decode(tmp_path):
    class NoDecode:
        calls = 0

        def load(self, *_args, **_kwargs):
            self.calls += 1
            pytest.fail("incomplete payload roster reached numerical decode")

    fake = NoDecode()
    receipt = {"version": audit.PRODUCER_VERSION, "phase": "train", "status": "completed",
               "pending": None, "error": None, "fits_completed": 6, "optimizer_steps": 3078,
               "episode_exposures": 18468, "views_completed": 9, "array_decodes": 5,
               "checkpoint_decodes": 3, "teacher_calls": 0, "native_calls": 0,
               "test_array_decodes": 0, "files": {}}
    checks = []
    with pytest.raises(ValueError, match="complete exact producer payload set"):
        audit.audit(fake, {}, receipt, tmp_path, lambda: checks.append(True))
    assert checks and fake.calls == 0


@pytest.fixture
def complete_journal(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("readout_compute_journal_reference_tests", ROOT / audit.REFERENCE)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "work.jsonl"
    fits = [{"family": family, "seed": seed, "parameters": audit.parameter_metadata(family),
             "orders": [list(range(54)) for _ in range(audit.EPOCHS[family])]}
            for seed in audit.SEEDS for family in audit.ARMS]
    geometry = audit.batch_geometry(ref, [6] * 6)
    call_id = 0
    with path.open("w") as stream:
        for fit in fits:
            for epoch in range(1, audit.EPOCHS[fit["family"]] + 1):
                for batch in range(9):
                    call_id += 1
                    indices = list(range(6 * batch, 6 * (batch + 1)))
                    identity = {"call_id": call_id, "family": fit["family"], "seed": fit["seed"],
                                "epoch": epoch, "batch": batch, "episode_indices": indices}
                    result = {"version": "otto-readout-ablation-training-v1", "objective": "full_forecast_aux",
                              "episode_indices": indices, "episode_exposures": 6, "optimizer_updates": 1,
                              "optimizer_step": (epoch - 1) * 9 + batch + 1,
                              "effective_parameter_names": fit["parameters"]["effective_names"],
                              "effective_parameter_count": fit["parameters"]["effective_count"],
                              "frozen_optimizer_state_entries": 0, **geometry,
                              "loss": 0., "nonquery_loss": 0., "prior_loss": 0.,
                              "gradient_norm_before_clip": 0., "zero_filled_gradient_names": [],
                              "timing_seconds": dict.fromkeys(audit.TIMINGS, 0.)}
                    stream.write(json.dumps({"event": "attempt", **identity}) + "\n")
                    stream.write(json.dumps({"event": "return", **identity, "result": result}) + "\n")
    return path, fits, ref


def test_complete_fabricated_3078_update_journal_is_counted(complete_journal):
    path, fits, ref = complete_journal
    checks = []
    result = audit.verify_journal(ref, audit.descriptor(path), fits, [6] * 54, lambda: checks.append(True))
    assert result["updates"] == 3078 and result["episode_exposures"] == 18468
    assert result["scheduled_training"]["forward_rows"] == 3078 * 36
    assert result["scheduled_training"]["forward_chunks"] == 3078
    assert len(checks) == 3078


@pytest.mark.parametrize("defect", ("missing_return", "trailing_work"))
def test_journal_incomplete_or_extra_work_rejected_with_recomputed_pin(complete_journal, defect):
    path, fits, ref = complete_journal
    if defect == "missing_return":
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]))
    else:
        with path.open("a") as stream:
            stream.write('{"event":"unregistered"}\n')
    with pytest.raises(ValueError, match="durable journal|extra journal"):
        audit.verify_journal(ref, audit.descriptor(path), fits, [6] * 54, lambda: None)
