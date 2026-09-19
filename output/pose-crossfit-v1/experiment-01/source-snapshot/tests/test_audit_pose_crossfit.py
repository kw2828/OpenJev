"""Synthetic saved-array checks, with no model construction or random draws."""
import copy
import math

import audit_pose_crossfit as audit
import numpy as np
import pytest


def errors(value, parents=None):
    parents = [value] * 10 if parents is None else parents
    return {e: {"mse": value, "rmse": math.sqrt(value), "horizon_mse": [value] * 25,
                "horizon_rmse": [math.sqrt(value)] * 25, "parent_ids": list(range(10)),
                "parent_mse": parents, "parent_rmse": np.sqrt(parents).tolist()} for e in audit.ENDPOINTS}


def rows():
    return {p: {f"{v}-{s}" if s else v: {"errors": errors(.81 if v == audit.PRIMARY else 1.),
                "latency_ms": [1.5 if v == audit.PRIMARY else 1.] * 20}
                for v in (*audit.VARIANTS, *audit.REFERENCES)
                for s in (audit.SEEDS if v in audit.VARIANTS else (None,))} for p in audit.PANELS}


def test_exact_1921_gate_and_seventeen_group_partition():
    families, refs, latency, contrasts, gate = audit.aggregate(rows())
    assert gate["passed"] and gate["checks_passed"] == gate["total_checks"] == 1921
    assert gate["requirements_passed"] == gate["total_requirements"] == 17
    names = [n for group in gate["requirements"] for n in group["comparison_names"]]
    assert len(names) == len(set(names)) == 1921
    assert len(families["test_sin"]) == 27 and len(refs["test_sin"]) == 6
    assert len(contrasts["test_sin"]) == 32
    assert latency["oof_recurrent"]["samples"] == 120 and latency["cv1"]["samples"] == 40


@pytest.mark.parametrize("violation", ["family", "paired", "parents", "loo", "latency"])
def test_each_failed_condition_prevents_continuation(violation):
    value = rows()
    for seed in audit.SEEDS:
        if violation == "family":
            value["test_sin"][f"oof_recurrent-{seed}"]["errors"] = errors(.811)
        elif violation == "paired":
            value["test_sin"][f"oof_recurrent-{seed}"]["errors"] = errors(1.01 if seed == 1101 else .01)
        elif violation == "parents":
            value["test_sin"][f"oof_recurrent-{seed}"]["errors"] = errors(.8, [.5] * 7 + [1.5] * 3)
        elif violation == "loo":
            value["test_sin"][f"oof_recurrent-{seed}"]["errors"] = errors(.5)
            value["test_sin"][f"is_constant-{seed}"]["errors"] = errors(.9, [4.5] + [.5] * 9)
        else:
            for panel in audit.PANELS:
                value[panel][f"oof_recurrent-{seed}"]["latency_ms"] = [1.5001] * 20
    assert not audit.aggregate(value)[-1]["passed"]


def identifiers():
    return np.column_stack((np.repeat(np.arange(30), 24), np.tile(np.arange(24), 30))).astype(np.int64)


@pytest.mark.parametrize("fold", range(3))
def test_whole_parent_exclusion(fold):
    ids = identifiers()
    idx = audit.eligible_indices(ids, fold)
    assert len(idx) == 480 and np.all(ids[idx, 0] % 3 != fold)
    assert len(np.unique(ids[idx, 0])) == 20


@pytest.mark.parametrize("damage", ["duplicate", "parent", "dtype", "fold"])
def test_invalid_fold_membership_rejected(damage):
    ids = identifiers(); fold = 0
    if damage == "duplicate":
        ids[1] = ids[0]
    elif damage == "parent":
        ids[0, 0] = 30
    elif damage == "dtype":
        ids = ids.astype(np.float64)
    else:
        fold = 3
    with pytest.raises(ValueError):
        audit.eligible_indices(ids, fold)


def test_excluded_rows_cannot_change_preprocessing():
    ids = identifiers(); idx = audit.eligible_indices(ids, 0)
    p = np.zeros((720, 57, 3), np.float32)
    p[:, :, 0] = np.arange(57, dtype=np.float32) * .25
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (720, 57, 3, 3)).copy()
    a = np.broadcast_to(np.arange(720, dtype=np.float32)[:, None, None], (720, 56, 40)).copy()
    before = audit.fold_statistics(p, r, a, idx)
    excluded = ids[:, 0] % 3 == 0
    p[excluded] *= 100
    a[excluded] += 100000
    after = audit.fold_statistics(p, r, a, idx)
    assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))
    assert before[0][0] == np.float32(np.arange(720)[idx].mean())
    assert np.isclose(before[2][0], .25 / math.sqrt(3))
    assert np.array_equal(before[2][1:], np.full(3, 1e-5, np.float32))


@pytest.mark.parametrize("stage", audit.STAGES)
def test_exact_cache_assignment_and_corruption(stage):
    ids = identifiers()
    chosen = (ids[:, 0] % 3 + (stage == "is")) % 3
    folds = {k: {"fp": np.full((720, 25, 3), k, np.float32),
                 "sp": np.full((720, 25, 3), k + 3, np.float32),
                 "fR": np.broadcast_to(np.eye(3, dtype=np.float32), (720, 25, 3, 3)).copy(),
                 "sR": np.broadcast_to(np.eye(3, dtype=np.float32), (720, 25, 3, 3)).copy()} for k in range(3)}
    combined = {key: np.stack([folds[int(f)][key][i] for i, f in enumerate(chosen)]) for key in folds[0]}
    result = audit.verify_assignment(stage, ids, chosen, combined, folds)
    assert result["per_expert_windows"] == [240] * 3
    assert result["experts_excluded_queried_parent"] == (stage == "oof")
    combined["fp"][0, 0, 0] += .1
    with pytest.raises(ValueError, match="forecast selection"):
        audit.verify_assignment(stage, ids, chosen, combined, folds)
    with pytest.raises(ValueError, match="fold assignment"):
        audit.verify_assignment(stage, ids, (chosen + 1) % 3, combined, folds)


def test_full_permutation_and_repeated_index_rejection():
    orders = np.broadcast_to(np.arange(480), (46, 480)).copy()
    audit.validate_orders(orders, 46, 480)
    orders[4, 0] = 1
    with pytest.raises(ValueError, match="permutations"):
        audit.validate_orders(orders, 46, 480)


def constant_fixture(*, capped=False):
    """Two exact quarter-turn examples with analytic distances a*pi/2 and (1-a)*pi/2."""
    eye = np.eye(3, dtype=np.float32)
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    fp = np.full((2, 1, 3), 2., np.float32); sp = np.zeros_like(fp); tp = np.ones_like(fp)
    fr = np.broadcast_to(turn, (2, 1, 3, 3)).copy()
    sr = np.broadcast_to(eye, fr.shape).copy(); tr = np.stack((eye, turn))[:, None]
    records = []
    for i, (alpha, radius) in enumerate([(0., 0.), (1., 0.), (.5, .5)] + ([] if capped else [(.25, .25), (.75, .25)])):
        distance = np.array([alpha, 1 - alpha]) * (math.pi / 2)
        lower = max(0., float(np.square(np.maximum(0., distance - 1e-12 - (math.pi / 2 + 1e-12) * radius)).mean()) - 1e-12)
        records.append({"id": i, "alpha": alpha, "radius": radius, "role": "endpoint" if i < 2 else "interval_center",
                        "objective": float(np.square(distance).mean()), "lower_bound": lower})
    intervals = [{"id": 0, "parent": None, "left": 0., "right": 1., "center_evaluation": 2,
                  "lower_bound": records[2]["lower_bound"], "children": None if capped else [1, 2]}]
    if not capped:
        intervals.extend({"id": i, "parent": 0, "left": left, "right": right, "center_evaluation": i + 2,
                          "lower_bound": records[i + 2]["lower_bound"], "children": None}
                         for i, left, right in [(1, 0., .5), (2, .5, 1.)])
    leaves = [0] if capped else [1, 2]
    objective = (math.pi / 4) ** 2; lower = min(intervals[i]["lower_bound"] for i in leaves)
    projection = {"matrices": 2, "max_frobenius_change": 0., "min_input_singular_value": 1.,
                  "reflections_corrected": 0, "max_orthogonality_error": 0., "max_determinant_error": 0.}
    tolerance, cap = (1e-7, 3) if capped else (.4, 5)
    record = {"version": "pose-constant-v1", "alpha": [.5, .5], "input_shape": [2, 1, 3], "examples": 2,
        "position": {"numerator": 12., "denominator": 24., "unclipped_alpha": .5, "degenerate": False, "objective_m2": 0.},
        "rotation": {
            "objective": "mean squared geodesic radians on float64 SVD-projected SO(3) inputs",
            "interpolation": "Exp(alpha*Log(Rfast*Rslow.T))*Rslow; principal log, largest-axis positive at exact pi",
            "certificate_scope": "numerical guarded bound for projected interpolation only; not float32 production or formal interval arithmetic",
            "projection_method": "float64 SVD U diag(1,1,det(UVt)) Vt; no input mutation",
            "projection": {name: dict(projection) for name in ("fast", "slow", "target")},
            "tolerance": tolerance, "max_evaluations": cap, "call_count": len(records),
            "status": "evaluation_cap" if capped else "tolerance_reached", "certified": not capped,
            "alpha": .5, "objective_at_alpha": objective, "upper_bound": objective + 1e-12,
            "lower_bound": lower, "gap": objective + 1e-12 - lower,
            "angle_guard": 1e-12, "objective_guard": 1e-12, "speed_min": math.pi / 2, "speed_max": math.pi / 2,
            "tie_rule": "smallest evaluated alpha among exact equal float64 objectives",
            "queue_rule": "smallest interval lower bound, then left endpoint, then interval id",
            "evaluations": records, "intervals": intervals, "leaves": leaves}}
    return record, (fp, fr, sp, sr, tp, tr), {"tolerance": tolerance, "max_evaluations": cap}


@pytest.mark.parametrize("capped", [False, True])
def test_hand_computed_interval_certificate_and_honest_cap(capped):
    record, data, settings = constant_fixture(capped=capped)
    result = audit.verify_constant(record, *data, **settings)
    assert result["certified"] is (not capped)
    assert result["call_count"] == (3 if capped else 5)
    assert result["float32_blend_diagnostic"]["position_mse_m2"] == 0.
    assert math.isclose(result["float32_blend_diagnostic"]["rotation_mse_rad2"], (math.pi / 4) ** 2, abs_tol=1e-7)


@pytest.mark.parametrize("damage", ["objective", "bound", "projection", "position", "nonbest", "partition", "leaves", "count", "cap_claim", "continued"])
def test_constant_certificate_tampering_rejected(damage):
    record, data, settings = constant_fixture(capped=damage == "cap_claim")
    record = copy.deepcopy(record); r = record["rotation"]
    if damage == "objective":
        r["evaluations"][3]["objective"] += .01
    elif damage == "bound":
        r["evaluations"][3]["lower_bound"] += .01
    elif damage == "projection":
        r["projection"]["fast"]["max_frobenius_change"] = .1
    elif damage == "position":
        record["position"]["numerator"] = 13.
    elif damage == "nonbest":
        record["alpha"][1] = r["alpha"] = .25
    elif damage == "partition":
        r["intervals"][2]["left"] = .51
    elif damage == "leaves":
        r["leaves"] = [1]
    elif damage == "count":
        r["call_count"] += 1
    elif damage == "cap_claim":
        r["status"] = "tolerance_reached"; r["certified"] = True
    else:
        settings["tolerance"] = r["tolerance"] = 2.
    with pytest.raises(ValueError):
        audit.verify_constant(record, *data, **settings)


def test_uncertified_constant_forces_failure_even_when_all_numeric_groups_pass():
    gate = audit.aggregate(rows())[-1]
    constants = {str(i): {"certified": True} for i in range(9)}
    assert audit.apply_certification_gate(gate, constants)["passed"]
    constants['7']["certified"] = False
    failed = audit.apply_certification_gate(gate, constants)
    assert failed["numerical_comparisons_passed"] and failed["checks_passed"] == 1921
    assert not failed["constant_certification_pass"] and not failed["passed"]
    assert failed["requirements_passed"] == 17


def test_exact_payload_contract_and_exclusive_failure(tmp_path):
    names = audit.expected_members()
    assert len(names) == 288 and len(audit.SOURCES) == 24
    assert sum(n.endswith('-predictions.npz') for n in names) == 54
    assert sum(n.endswith('-initial.pt') for n in names) == 30
    assert sum(n.endswith('-normalization.npz') for n in names) == 9
    out = tmp_path / 'audit'
    with pytest.raises(FileNotFoundError):
        audit.audit(tmp_path / 'absent', out, protocol_sha256='a', completed_sha256='b')
    payload = (out / 'failed.json').read_bytes()
    assert not (out / 'receipt.json').exists()
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path / 'absent', out, protocol_sha256='a', completed_sha256='b')
    assert (out / 'failed.json').read_bytes() == payload


@pytest.mark.parametrize("variant, canonical", [("fast", "decay_huber3"), ("slow", "gru")])
def test_alias_replay_uses_pure_pose_parent_schema(tmp_path, variant, canonical):
    support = tmp_path / 'support'; support.mkdir()
    coordination = tmp_path / 'coordination'; coordination.mkdir()
    p = np.zeros((2, 25, 3), np.float32)
    r = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 25, 3, 3)).copy()
    file = support / f'test_sin-{canonical}-1101-predictions.npz'
    np.savez(file, p=p, R=r)
    np.savez(coordination / f'test_sin-{variant}-1101-predictions.npz', p=p, R=r, alpha=np.ones((2, 2), np.float32))
    context = {'prior_run': coordination, 'parent': {'prior_run': support}}
    result = audit.replay_expert(p, r, context, 'test_sin', variant, 1101)
    assert result['prior_predictions_sha256'] == audit.sha(file)
    assert result['p']['exact_array_equal'] and result['R']['exact_array_equal']
