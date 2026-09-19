"""Independent arithmetic/corruption tests; no trained model is evaluated."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_reacher_geometry_study as audit


def bank_fixture(mode="geometry"):
    commands = np.zeros((2, 3, 4, 2), np.float32)
    commands[..., 0], commands[..., 1] = .2, -.3
    target = np.zeros((2, 2), np.float32)
    shape = commands.shape[:-1]
    values = {"commands": commands.copy(), "root_target": target.copy(),
              "predicted_angles": np.broadcast_to(np.array([1, 1, 0, 0], np.float32), (*shape, 4)).copy(),
              "learned_rewards": np.full(shape, -.4, np.float32)}
    if mode == "geometry":
        values.update(geometry_joint_angles=np.zeros((*shape, 2), np.float32),
                      geometry_pair_norms=np.ones((*shape, 2), np.float32),
                      geometry_fingertip=np.broadcast_to(np.array([.21, 0], np.float32), (*shape, 2)).copy(),
                      geometry_distance=np.full(shape, .21, np.float32),
                      geometry_action_cost=np.full(shape, .13, np.float32))
        values["selected_rewards"] = -values["geometry_distance"] - values["geometry_action_cost"]
    else:
        values["selected_rewards"] = values["learned_rewards"].copy()
    return {"noise_std": 0.0}, values, commands, target


@pytest.mark.parametrize("sigma", [0., .05, .8, 10.])
def test_independent_expected_cost_matches_quadrature(sigma):
    commands = np.array([[0., .2], [-.7, 1.]], np.float64)
    result = audit.expected_action_cost(commands, sigma)
    if sigma == 0:
        expected = (commands**2).sum(-1)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(160)
        expected = []
        for row in commands:
            total = 0.
            for u in row:
                density = np.exp(-.5 * ((nodes - u) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))
                outside = 1 - .5 * (math.erf((1 - u) / (sigma * math.sqrt(2)))
                                     - math.erf((-1 - u) / (sigma * math.sqrt(2))))
                total += float(weights @ (nodes**2 * density)) + outside
            expected.append(total)
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("angles,tip", [([0, 0], [.21, 0]), ([math.pi/2, 0], [0, .21]),
                                       ([0, math.pi/2], [.10, .11]), ([math.pi/2, -math.pi/2], [.11, .10])])
def test_geometry_matches_analytic_postures_without_scorer_import(dtype, angles, tip):
    q = np.asarray(angles, dtype=dtype)
    prediction = np.concatenate((np.cos(q), np.sin(q)))
    target, action = np.array([.03, -.04], dtype=dtype), np.array([.3, -.4], dtype=dtype)
    result = audit.geometry_components(prediction, target, action, 0.)
    tol = 1e-7 if dtype == np.float32 else 1e-15
    np.testing.assert_allclose(result["fingertip"], tip, rtol=0, atol=tol)
    assert result["reward"] == pytest.approx(-math.hypot(tip[0]-.03, tip[1]+.04)-.25, abs=tol)
    assert result["reward"].dtype == dtype


@pytest.mark.parametrize("mode", audit.MODES)
def test_bank_arithmetic_and_no_input_mutation(mode):
    plan, values, commands, target = bank_fixture(mode)
    before = copy.deepcopy(values)
    result = audit.audit_bank_arrays(plan, mode, values, commands, target)
    assert result["candidate_evaluations"] == 6 and result["imagined_transitions"] == 24
    assert result["geometry_samples"] == (24 if mode == "geometry" else 0)
    for key in values:
        np.testing.assert_array_equal(values[key], before[key])


@pytest.mark.parametrize("mutation", ["order", "target", "twice_cost", "distance", "norm", "angle", "tip",
                                      "rewards", "missing", "extra", "broadcast", "dtype", "nan", "degenerate"])
def test_rehashed_semantic_scoring_corruption_rejected(mutation):
    plan, values, commands, target = bank_fixture()
    if mutation == "order":
        values["predicted_angles"][..., [1, 2]] = values["predicted_angles"][..., [2, 1]]
    elif mutation == "target": values["root_target"] += .1
    elif mutation == "twice_cost": values["selected_rewards"] -= values["geometry_action_cost"]
    elif mutation in ("distance", "norm", "angle", "tip"):
        key = {"distance": "distance", "norm": "pair_norms", "angle": "joint_angles", "tip": "fingertip"}[mutation]
        values["geometry_"+key].flat[0] += .01
    elif mutation == "rewards": values["selected_rewards"].flat[0] += .01
    elif mutation == "missing": values.pop("geometry_pair_norms")
    elif mutation == "extra": values["hidden_native_label"] = np.zeros(1)
    elif mutation == "broadcast": values["geometry_distance"] = values["geometry_distance"][:1]
    elif mutation == "dtype": values["predicted_angles"] = values["predicted_angles"].astype(np.float64)
    elif mutation == "nan": values["learned_rewards"].flat[0] = np.nan
    else: values["predicted_angles"][..., 0] = 0
    with pytest.raises(ValueError): audit.audit_bank_arrays(plan, "geometry", values, commands, target)


def test_learned_baseline_does_not_evaluate_geometry(monkeypatch):
    plan, values, commands, target = bank_fixture("learned")
    values["predicted_angles"].fill(0)  # Degenerate geometry does not invalidate a learned-only scorer.
    monkeypatch.setattr(audit, "geometry_components", lambda *a: pytest.fail("geometry called in baseline"))
    audit.audit_bank_arrays(plan, "learned", values, commands, target)
    values["selected_rewards"].flat[0] += .001
    with pytest.raises(ValueError, match="baseline reward changed"):
        audit.audit_bank_arrays(plan, "learned", values, commands, target)


def test_byte_exact_dedup_retains_every_identity_and_first_occurrence():
    slots = np.zeros((5, 3, 2), np.float32)
    slots[1, :, 0] = .2
    slots[2] = slots[1]
    slots[4, 0, 1] = .1
    unique, mapping = audit.first_occurrence_union(slots)
    np.testing.assert_array_equal(mapping, [0, 1, 1, 0, 2])
    np.testing.assert_array_equal(unique[mapping], slots)
    unique[0].fill(1)
    assert np.all(slots[0] == 0)


def controls_fixture():
    result = {}
    for panel in audit.PANELS:
        result[panel] = {}
        for family in audit.FAMILIES:
            for pair in audit.PAIRS:
                for mode in audit.MODES:
                    cost = 8.0 if mode == "geometry" else 9.0
                    result[panel][f"{family}-{pair}--{mode}"] = {"episode_costs": [cost, cost], "mean_cost": cost}
        for name, cost in (("known_state", 7.), ("particle", 7.5), ("zero", 12.), ("uniform", 20.), ("public_kinematic", 7.)):
            result[panel][name] = {"episode_costs": [cost, cost], "mean_cost": cost}
    return result


def test_all25_checks_and_descriptive_mlp_cannot_rescue_failed_gru():
    controls = controls_fixture()
    gate, differences = audit.qualification({"control_episodes": 2}, controls)
    assert gate["passed"] and len(gate["checks"]) == 25
    assert differences["ordinary"]["residual_gru"] == [-1, -1]
    controls["shift"]["residual_gru-pair2--geometry"] = {"episode_costs": [9.5, 9.5], "mean_cost": 9.5}
    gate, _ = audit.qualification({"control_episodes": 2}, controls)
    assert not gate["passed"]
    assert not next(x for x in gate["checks"] if x["name"] == "shift/pair2: GRU geometry nonworse than learned")["passed"]


@pytest.mark.parametrize("mutation", ["omit", "extra", "mean", "cases", "nan"])
def test_incomplete_or_false_control_summary_rejected(mutation):
    controls = controls_fixture()
    row = controls["ordinary"]["cached_mlp-pair1--learned"]
    if mutation == "omit": controls["full"].pop("cached_mlp-pair0--geometry")
    elif mutation == "extra": controls["shift"]["winner_only"] = copy.deepcopy(row)
    elif mutation == "mean": row["mean_cost"] += .1
    elif mutation == "cases": row["episode_costs"].pop()
    else: row["episode_costs"][0] = float("nan")
    with pytest.raises(ValueError): audit.qualification({"control_episodes": 2}, controls)


def test_runtime_boundary_checks_real_runtime_and_rejects_drift():
    import importlib.metadata
    import platform

    import gymnasium.envs.mujoco.reacher_v5 as native
    import mujoco

    runtime = {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "gymnasium", "mujoco")},
        "native_source_sha256": audit.sha(native.__file__),
        "native_xml_sha256": audit.sha(Path(native.__file__).parent / "assets/reacher.xml"),
        "mujoco_init_sha256": audit.sha(mujoco.__file__)}
    source = "scripts/audit_reacher_geometry_study.py"
    plan = {"runtime": runtime, "sources": {source: audit.sha(ROOT / source)}}
    audit.validate_runtime_sources(plan)
    plan["runtime"]["packages"]["numpy"] = "invalid-drift"
    with pytest.raises(ValueError, match="runtime"):
        audit.validate_runtime_sources(plan)


def bank_metadata_fixture():
    plan, values, commands, target = bank_fixture()
    root = {"packet": np.concatenate((np.ones((2, 2), np.float32), np.zeros((2, 6), np.float32)), axis=1)}
    root["packet"][:, 4:6] = target
    root_digest = audit.np_state_hash(root)
    meta = {"score_mode": "geometry", "root_sha256": root_digest, "bank_shape": list(commands.shape),
        "bank_dtype": "float32", "bank_sha256": audit.hashlib.sha256(commands.tobytes()).hexdigest(),
        "completed_model_offsets": 4, "completed_scored_offsets": 4, "model_advance_attempted_samples": 24,
        "model_advance_samples": 24, "geometry_attempted_samples": 24, "geometry_samples": 24,
        "model_advance_seconds": .01, "geometry_seconds": .01, "bank_wall_seconds": .03,
        "maximum_candidate_state_tensor_bytes": root["packet"].nbytes * 3, "minimum_pair_norm": 1.,
        "joint_limit_violation_samples": 0, "terminal_state_sha256": "b" * 64,
        "prefix_state_sha256": ["b" * 64] * 4, "geometry_diagnostics_omitted_in_learned_mode": False}
    return plan, values, root, meta


def test_metadata_accounts_all_geometry_and_model_calls():
    plan, values, root, meta = bank_metadata_fixture()
    result = audit.audit_bank_metadata(plan, "geometry", meta, values, root)
    assert result["model_advance_samples"] == result["geometry_samples"] == 24


@pytest.mark.parametrize("field", ["root_sha256", "bank_sha256", "completed_scored_offsets", "model_advance_attempted_samples",
    "geometry_attempted_samples", "geometry_samples", "maximum_candidate_state_tensor_bytes", "minimum_pair_norm",
    "joint_limit_violation_samples", "prefix_state_sha256", "terminal_state_sha256", "bank_wall_seconds", "extra"])
def test_metadata_rejects_rehashed_scope_and_cost_corruption(field):
    plan, values, root, meta = bank_metadata_fixture()
    if field in ("root_sha256", "bank_sha256", "terminal_state_sha256"):
        meta[field] = "c" * 64
    elif field == "prefix_state_sha256":
        meta[field] = meta[field][:-1]
    elif field == "bank_wall_seconds":
        meta[field] = .001
    elif field == "extra":
        meta[field] = True
    else:
        meta[field] += 1
    with pytest.raises(ValueError):
        audit.audit_bank_metadata(plan, "geometry", meta, values, root)


def test_failed_audit_preserves_failure_without_completed_receipt(tmp_path):
    plan = audit.protocol.settings(engineering=True)
    plan.update(engineering=True, cap_seconds=10, audit_cap_seconds=10, runtime={}, sources={})
    output = tmp_path / "failed-audit"
    with pytest.raises(ValueError):
        audit.audit_saved(plan, "a" * 64, tmp_path / "absent", output, engineering=True)
    failure = audit.read(output / "failed.json")
    assert failure["status"] == "failed" and failure["saved_output_only"] is True
    assert not (output / "receipt.json").exists()
    with pytest.raises(ValueError, match="Exclusive"):
        audit.audit_saved(plan, "a" * 64, tmp_path / "absent", output, engineering=True)


RETAINED = ROOT / "output/reacher-geometry-rehearsal-v1/attempt-01"


@pytest.mark.skipif(not (RETAINED / "execution/completed.json").exists(), reason="Optional retained engineering repair fixture")
def test_retained_engineering_lower_level_full_schema_without_new_inference(monkeypatch):
    """Repair regression only: it cannot issue a replacement audit receipt.

    The source-authenticated attempt-01 audit remains failed. These lower-level
    checks validate its saved engineering arrays with current development code;
    a new whole-tree execution is required to bind the final current sources.
    """
    import torch

    def forbid(*args, **kwargs):
        raise AssertionError("Saved-output auditor must not run neural inference or construct Adam")

    monkeypatch.setattr(torch.nn.Linear, "forward", forbid)
    monkeypatch.setattr(torch.nn.GRUCell, "forward", forbid)
    monkeypatch.setattr(torch.optim.Adam, "__init__", forbid)
    plan, execution = audit.read(RETAINED / "plan.json"), RETAINED / "execution"
    assert plan["engineering"] is True and plan["control_episodes"] == 1
    assert plan["rng_namespace"] == "reacher-geometry-engineering-whole-tree-v1"
    digest = audit.sha(RETAINED / "plan.json")
    completed = audit.validate_members(plan, digest, execution)
    parent, _ = audit.validate_lineage(plan, execution)
    audit.validate_streams(plan, execution)
    fits, restored = audit.audit_restoration(plan, parent, execution)
    assert len(fits) == 6 and all(row["observed_before_after_equal"] for row in fits.values())
    histories = {panel: audit.base.load_records(execution / "inherited/control" / panel / "residual_gru-pair0/episodes", parent["control_episodes"])
                 for panel in audit.PANELS}
    diagnostics = [audit.audit_diagnostic_root(plan, execution, row, histories[row["panel"]][row["case_index"]], fits)
                   for row in audit.protocol.diagnostic_manifest(plan)]
    assert len(diagnostics) == 3
    inputs = [audit.search_audit.audit_innovations(plan, execution / f"innovations/control/{step:03d}",
              f"planner/control/{step}", 1) for step in range(50)]
    controls = {panel: {} for panel in audit.PANELS}
    for row in audit.protocol.execution_order(plan):
        folder = execution / row["path"]
        records = audit.base.load_records(folder / "episodes", 1)
        assert audit.audit_cohort(plan, records, row["panel"])["transitions"] == 50
        if "fit" in row:
            result = audit.audit_learned_control(plan, folder, row, records, inputs, fits[row["fit"]])
        elif row["reference"] == "public_kinematic":
            result = audit.inherited.audit_kinematic_control(plan, folder, records, inputs)
        else:
            result = audit.search_audit.audit_control({**plan, "planners": ["cem256"]}, folder, row, records, inputs)
        controls[row["panel"]][row["label"]] = result
    audit.audit_boundaries(plan, digest, execution, completed, restored, diagnostics)
    costs = audit.audit_costs(plan, execution, completed, restored, diagnostics, controls)
    assert costs["control_native_transitions"] == 2550 and costs["new_fits"] == 0
    gate, differences = audit.qualification(plan, controls)
    assert len(gate["checks"]) == 25
    comparisons = audit.paired_comparisons(plan, controls, differences)
    assert set(comparisons) == set(audit.PANELS)
    assert (RETAINED / "audit/failed.json").exists() and not (RETAINED / "audit/receipt.json").exists()
