"""Fabricated independent objective, export, parity and diagnostic-gate oracles."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_saved_audit_tests", ROOT / "scripts/audit_otto_direct_readout.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_legal_scalar_objective_has_fixed_episode_weights_and_no_illegal_leak():
    z = np.zeros((3, 29))
    z[:, -1] = 1
    errors = np.array([[1., 3., 999., -999.], [1., 2., 3., 4.], [100., 200., 300., 400.]])
    legal = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool)
    weights = np.array([.25, .125, 0.])
    loss, normal = audit.projected_objective(np, z, errors, legal, weights, np.zeros((4, 29)))
    # Row one contributes .25 * (1 + 1); row two .125 * (2.25+.25+.25+2.25).
    assert loss == 1.125
    assert normal.shape == (87,)
    assert np.all(normal.reshape(3, 29)[:, :-1] == 0)
    assert normal.reshape(3, 29)[:, -1] == pytest.approx([.25, .375, .25])
    errors[:, 2:] += 1e6 * (~legal[:, 2:])
    assert audit.projected_objective(np, z, errors, legal, weights, np.zeros((4, 29)))[0] == loss


def test_objective_common_action_gauge_and_all_zero_support():
    z = np.ones((2, 29))
    error = np.arange(8, dtype=float).reshape(2, 4)
    legal = np.ones((2, 4), dtype=bool)
    zero = np.zeros((4, 29))
    left = audit.projected_objective(np, z, error, legal, np.ones(2), zero)
    right = audit.projected_objective(np, z, error + 80, legal, np.ones(2), zero + 3)
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    loss, normal = audit.projected_objective(np, z, error, legal & False, np.zeros(2), zero)
    assert loss == 0 and not normal.any()


def solved_fixture(family):
    shapes = {"slow.recurrent.weight_ih_l0": (84, 40), "slow.recurrent.weight_hh_l0": (84, 28),
              "slow.recurrent.bias_ih_l0": (84,), "slow.recurrent.bias_hh_l0": (84,),
              "slow.output.weight": (4, 28), "slow.output.bias": (4,),
              "slow.action_residual.weight": (4, 28), "slow.action_residual.bias": (4,)}
    parent = {k: np.zeros(shape, np.float32) for k, shape in shapes.items()}
    adam = {k: v.copy() for k, v in parent.items()}
    solved = {k: v.copy() for k, v in parent.items()}
    penalty = .0001 if family == "ridge" else 0.
    coefficient = 2 / (1 + penalty)
    B = np.zeros((3, 29))
    B[0, -1] = coefficient
    # Independent explicit contrast direction [1,1,-1,-1]/2.
    increment = np.zeros((4, 29))
    increment[:, -1] = coefficient * np.array([.5, .5, -.5, -.5])
    solved["slow.action_residual.bias"] = increment[:, -1].astype(np.float32)
    adam["slow.action_residual.bias"][:] = [.5, .5, -.5, -.5]
    z = np.zeros((2, 29))
    z[:, -1] = 1
    cache = {"z": z, "design_error": np.array([[1., 1., -1., -1.]] * 2),
             "design_legal": np.ones((2, 4), dtype=bool), "design_weights": np.full(2, .5)}
    spectrum = [math.sqrt(1 + penalty)] * 3 + ([math.sqrt(penalty)] * 84 if penalty else [0.] * 5)
    rank = 87 if penalty else 3
    condition = spectrum[0] / spectrum[rank - 1]
    loss = (coefficient - 2) ** 2
    regularization = penalty * coefficient**2
    diagnostics = {"mode": family, "rcond": 1e-10, "ridge": penalty,
                   "spectrum_scope": "augmented" if penalty else "data", "data_rows": 8,
                   "feature_dimension": 29, "coordinates": 87, "rows": 95 if penalty else 8,
                   "rank": rank, "full_column_rank": bool(penalty), "singular_cutoff": 1e-10 * spectrum[0],
                   "largest_singular_value": spectrum[0], "smallest_retained_singular_value": spectrum[rank - 1],
                   "retained_condition_number": condition, "condition_number": condition if penalty else None,
                   "objective_before": 4., "data_loss": loss, "regularization": regularization,
                   "total_objective": loss + regularization,
                   "normal_residual": abs(coefficient - 2 + penalty * coefficient)}
    exported_coefficient = float(solved["slow.action_residual.bias"][0]) * 2
    row = {"family": family, "seed": audit.SEEDS[0], "solver": {"coefficients": B.reshape(-1).tolist(),
           "B": B.tolist(), "increment": increment.tolist(), "singular_values": spectrum, "diagnostics": diagnostics},
           "frozen_names": [k.removeprefix("slow.") for k in list(shapes)[:6]],
           "cached_control_losses": {"pretrained": 4., "action_residual_only": 1.},
           "exported_cached_loss": (exported_coefficient - 2) ** 2,
           "cache_metadata": {"seconds": 1.}, "cache_design_seconds": 2., "solve_seconds": 3.,
           "export_seconds": 4., "train_validation_seconds": 5., "standalone_adaptation_seconds": 15.}
    return row, cache, parent, adam, solved


@pytest.mark.parametrize("family", audit.ARMS)
def test_closed_form_bias_only_ols_and_ridge_fit_oracle(family):
    row, cache, parent, adam, solved = solved_fixture(family)
    result = audit.check_fit(np, row, cache, parent, adam, solved)
    assert result["parent_loss"] == 4. and result["adam74_loss"] == 1.
    assert result["svd_recomputed"] is False and result["frozen_tensors_checked"] == 6
    assert result["total_objective"] == pytest.approx(4 * .0001 / 1.0001 if family == "ridge" else 0.)


@pytest.mark.parametrize("family", audit.ARMS)
def test_real_fabricated_svd_result_agrees_with_scalar_saved_audit(family):
    from openjev.research import otto_direct_readout as solver

    row, cache, parent, adam, solved = solved_fixture(family)
    rng = np.random.default_rng(993)
    z = rng.uniform(-1, 1, (64, 29))
    z[:, -1] = 1
    z[:, 7:14] = z[:, :7]  # Deliberate rank deficiency before ridge augmentation.
    error = rng.normal(size=(64, 4))
    legal = rng.uniform(size=(64, 4)) > .3
    legal[:, 0] = True
    weights = rng.uniform(0, .01, 64)
    weights[::5] = 0
    cache.update(z=z, design_error=error, design_legal=legal, design_weights=weights)
    A, b = solver.build_design(z, error, legal, weights)
    result = solver.solve_design(A, b, mode=family)
    row["solver"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    head = result["increment"].astype(np.float32)
    solved["slow.action_residual.weight"] = head[:, :28].copy()
    solved["slow.action_residual.bias"] = head[:, 28].copy()
    U = solver.contrast_basis()
    adam_coefficient = (U.T @ audit.theta(np, adam)).reshape(-1)
    exported_coefficient = (U.T @ audit.theta(np, solved)).reshape(-1)
    row["cached_control_losses"] = {"pretrained": float(b @ b),
                                    "action_residual_only": float(np.square(A @ adam_coefficient - b).sum())}
    row["exported_cached_loss"] = float(np.square(A @ exported_coefficient - b).sum())
    checked = audit.check_fit(np, row, cache, parent, adam, solved)
    assert checked["normal_residual"] < 1e-10
    assert checked["svd_recomputed"] is False


@pytest.mark.parametrize("defect", ("frozen", "export", "coefficient", "increment", "loss", "penalty",
                                   "normal", "spectrum", "rank", "control", "export_loss", "cost"))
def test_independent_fit_witness_rejects_corruptions(defect):
    row, cache, parent, adam, solved = solved_fixture("ridge")
    if defect == "frozen":
        solved["slow.output.bias"][0] = 1
    elif defect == "export":
        solved["slow.action_residual.bias"][0] += .1
    elif defect == "coefficient":
        row["solver"]["coefficients"][0] = 1.
    elif defect == "increment":
        row["solver"]["increment"][0][0] = 1.
    elif defect in ("loss", "penalty", "normal", "rank"):
        key = {"loss": "data_loss", "penalty": "regularization", "normal": "normal_residual", "rank": "rank"}[defect]
        row["solver"]["diagnostics"][key] += 1
    elif defect == "spectrum":
        row["solver"]["singular_values"][1] = 100.
    elif defect == "control":
        row["cached_control_losses"]["action_residual_only"] = 0.
    elif defect == "export_loss":
        row["exported_cached_loss"] += .1
    else:
        row["standalone_adaptation_seconds"] = 14.
    with pytest.raises(ValueError):
        audit.check_fit(np, row, cache, parent, adam, solved)


def test_saved_train_parity_uses_prequery_prior_and_rejects_false_success():
    cache = {"base": np.zeros((3, 4), np.float32), "z": np.zeros((3, 29)),
             "support_mask": np.array([False, True, True]), "prior_mask": np.array([False, False, True])}
    cache["z"][:, -1] = 1
    head = np.zeros((4, 29))
    head[:, -1] = [.5, -.5, .25, -.25]
    score = np.array([32., -32., 16., -16.], dtype=np.float32)
    predicted = {"action_prediction": np.array([[100.] * 4, score, [999.] * 4], dtype=np.float32),
                 "corrected_shadow_prior": np.array([[0.] * 4, [0.] * 4, score], dtype=np.float32)}
    expected = audit.parity(np, cache, head, predicted)
    assert expected == {"passed": True, "max_abs": 0., "max_tolerance_ratio": 0., "supported_rows": 2}
    audit.verify_parity(expected, expected)
    predicted["corrected_shadow_prior"][2, 0] += 1
    changed = audit.parity(np, cache, head, predicted)
    assert changed["passed"] is False
    with pytest.raises(ValueError, match="passing ordinary TRAIN parity"):
        audit.verify_parity(expected, changed)


def reports(candidate=19., control=20., support=6):
    return [{"family": family, "seed": seed, "stage": "dev", "query_period": 4, "episodes": 36,
             "scopes": {scope: {"by_regime": {regime: {"episodes": 18, "declared_case_count": 6,
                "supported_case_count": support, "case_weighted_raw_gap":
                (candidate if family == "ridge" else control) if scope == "later" else 10.}
                for regime in audit.REGIMES}} for scope in ("full", "later")}}
            for seed in audit.SEEDS for family in audit.VIEWS]


def test_ridge_fixed_candidate_exact_boundary_and_reused_scope():
    rows = reports()
    result = audit.comparisons(rows)
    assert result == audit.comparisons(list(reversed(rows)))
    assert result["passed_cells"] == 6 and result["overall_passed"] is True
    assert result["candidate"] == "ridge" and result["dev_reused"] is True
    assert result["fresh_dev"] is result["held_out_evidence"] is result["test_admitted"] is False
    row = next(r for r in rows if r["family"] == "ridge")
    row["scopes"]["later"]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = float(np.nextafter(19., np.inf))
    assert audit.comparisons(rows)["passed_cells"] == 5


@pytest.mark.parametrize("family", audit.CONTROLS)
@pytest.mark.parametrize("scope", ("full", "later"))
def test_each_parent_adam_condition_is_required(family, scope):
    rows = reports()
    row = next(r for r in rows if r["family"] == family)
    row["scopes"][scope]["by_regime"]["lambda3"]["case_weighted_raw_gap"] = 9. if scope == "full" else 18.
    assert audit.comparisons(rows)["passed_cells"] == 5


def test_zero_comparator_and_better_ols_cannot_rescue_ridge():
    rows = reports(candidate=0., control=0., support=0)
    assert audit.comparisons(rows)["passed_cells"] == 0
    rows = reports(candidate=20., control=20.)
    for row in rows:
        if row["family"] in ("ols", "full_joint"):
            for scope in ("full", "later"):
                for leaf in row["scopes"][scope]["by_regime"].values():
                    leaf["case_weighted_raw_gap"] = 0.
    assert audit.comparisons(rows)["overall_passed"] is False


@pytest.mark.parametrize("defect", ("missing", "duplicate", "stage", "support", "negative", "nonfinite"))
def test_invalid_view_rosters_and_metrics_fail_closed(defect):
    rows = reports()
    leaf = rows[0]["scopes"]["later"]["by_regime"]["lambda3"]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif defect == "stage":
        rows[0]["stage"] = "test"
    elif defect == "support":
        leaf["supported_case_count"] = 5
    else:
        leaf["case_weighted_raw_gap"] = -.1 if defect == "negative" else float("nan")
    with pytest.raises(ValueError):
        audit.comparisons(rows)


def test_incomplete_receipt_rejected_before_any_array_decode(tmp_path):
    class NoArrays:
        def load(self, *_args, **_kwargs):
            pytest.fail("incomplete closure reached numerical decode")

    receipt = {"version": "otto-direct-readout-v1", "phase": "train", "status": "completed", "pending": None,
               "error": None, "files": {}, "fits_completed": 6, "solves_completed": 6,
               "train_views_completed": 12, "views_completed": 15, "optimizer_steps": 0,
               "episode_exposures": 0, "checkpoint_decodes": 9, "array_decodes": 11,
               "teacher_calls": 0, "native_calls": 0, "test_array_decodes": 0}
    with pytest.raises(ValueError, match="complete producer payloads"):
        audit.audit(NoArrays(), {}, receipt, tmp_path, lambda: None)
