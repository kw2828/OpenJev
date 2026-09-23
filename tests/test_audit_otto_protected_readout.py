"""Fabricated independent saved-output checks; no models or empirical inputs."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_test_protected_saved_audit", ROOT / "scripts/audit_otto_protected_readout.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def instance(tmp_path):
    obj = audit.Audit(SimpleNamespace(output=tmp_path))
    obj.np, obj.run = np, tmp_path
    obj.check = lambda: None
    return obj


def gate_fixture():
    def report(gap):
        return {scope: {"by_regime": {regime: {"episode_weighted_raw_gap": gap,
            "episode_weighted_agreement": .5, "supported_case_count": 6,
            "by_age": {str(age): {"supported_case_count": 6} for age in (1, 2, 3)}}
            for regime in audit.REGIMES}} for scope in ("initial", "full", "postcorrection")}
    models = [{"family": kind, "seed": seed, "evaluation": audit.evaluation_metadata(kind),
               "metrics": report(9. if kind == "frozen_spo" else 10.)}
              for seed in audit.SEEDS for kind in audit.FAMILIES]
    return models, report(20.)


def test_hand_calculated_gate_uses_every_control_and_original_technical_closure():
    models, hold = gate_fixture()
    rows = audit.continuation_rules(models, hold, technical_complete=False)
    gate = audit.gate_decisions(rows)
    assert len(rows) == 29 and [r["name"] for r in rows] == audit.condition_names()
    assert gate["protected_readout"]["passed"] == 28 and gate["protected_readout"]["passes"] is False
    assert all(r["threshold"] == 9. and r["strict_upper_bound"] == 10. for r in rows if "strict_upper_bound" in r)
    assert audit.gate_decisions(audit.continuation_rules(models, hold, technical_complete=True))["protected_readout"]["passes"]
    row = next(r for r in models if r["family"] == "joint_spo" and r["seed"] == audit.SEEDS[0])
    row["metrics"]["postcorrection"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = 0.
    rules = audit.continuation_rules(models, hold, technical_complete=True)
    assert next(r for r in rules if r["name"] == "candidate.lambda3.postcorrection.gap_vs_joint_spo")["passes"] is False


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reordered", "canonical_flags", "nonbool_technical"])
def test_gate_never_accepts_partial_or_misrepresented_comparison(defect):
    models, hold = gate_fixture()
    technical = True
    if defect == "missing":
        models.pop()
    elif defect == "duplicate":
        models[-1] = copy.deepcopy(models[-2])
    elif defect == "reordered":
        models[0], models[1] = models[1], models[0]
    elif defect == "canonical_flags":
        models[0]["evaluation"]["backbone_requires_grad"] = True
    else:
        technical = 1
    with pytest.raises(ValueError):
        audit.continuation_rules(models, hold, technical_complete=technical)


def test_zero_ties_and_one_seed_regression_are_failures():
    models, hold = gate_fixture()
    for row in models:
        row["metrics"]["postcorrection"]["by_regime"]["lambda3"]["episode_weighted_raw_gap"] = 0.
    rows = audit.continuation_rules(models, hold, technical_complete=True)
    assert all(not r["passes"] for r in rows if r["name"].startswith("candidate.lambda3.postcorrection.gap_vs_"))
    models, hold = gate_fixture()
    for row in models:
        if row["family"] == "frozen_spo":
            row["metrics"]["postcorrection"]["by_regime"]["lambda4"]["episode_weighted_raw_gap"] = (
                11. if row["seed"] == audit.SEEDS[0] else 1.)
    rows = audit.continuation_rules(models, hold, technical_complete=True)
    assert not next(r for r in rows if r["name"] == f"candidate.lambda4.{audit.SEEDS[0]}.postcorrection.gap_vs_frozen_aux")["passes"]


def history_fixture():
    lengths = [5] * 54
    offsets = np.arange(55, dtype=np.int64) * 5
    count = int(offsets[-1])
    query_mask = np.zeros(count, np.bool_)
    prior_mask = np.zeros(count, np.bool_)
    weights, prior_weights = np.zeros(count, np.float64), np.zeros(count, np.float64)
    for low in offsets[:-1]:
        query_mask[low:low+5:4] = True
        prior_mask[low+4] = True
        weights[low+1:low+4] = 1 / (54*3)
        prior_weights[low+4] = 1 / 54
    ids = audit.cohort()[:54]
    meta = {"episode_ids": [row["episode_id"] for row in ids],
            "episode_regimes": [row["regime"] for row in ids],
            "counts": {"episodes": 54, "rows": count, "prior_rows": 54}}
    train = {"episode_offsets": offsets, "query_scores": np.zeros((count, 4), np.float32),
             "targets": np.zeros((count, 4), np.float32), "prior_targets": np.zeros((count, 4), np.float32),
             "query_mask": query_mask, "prior_mask": prior_mask, "weights": weights,
             "prior_weights": prior_weights, "legal": np.ones((count, 4), np.bool_)}
    return train, meta, ids, lengths


def test_frozen_prior_only_chunks_are_paid_but_not_backpropagated(tmp_path):
    obj = instance(tmp_path)
    train, _, _, _ = history_fixture()
    train["weights"][:] = 0
    order = [list(range(54))]
    frozen = obj.expected_batches(train, order, "frozen_spo")
    joint = obj.expected_batches(train, order, "joint_spo")
    for a, b in zip(frozen, joint, strict=True):
        assert a["forward_chunks"] == a["loss_chunks"] == a["spo_loss_calls"] == 1
        assert a["differentiable_chunks"] == a["backward_chunks"] == a["spo_weighted_rows"] == 0
        assert a["skipped_backward_chunks"] == a["no_grad_chunks"] == 1
        assert b["differentiable_chunks"] == b["backward_chunks"] == 1
        assert b["skipped_backward_chunks"] == b["no_grad_chunks"] == 0
        assert a["trainable_parameter_count"] == 116 and b["trainable_parameter_count"] == 6112


def test_frozen_equality_is_bitwise_and_covers_base_prior_and_masks(tmp_path):
    obj = instance(tmp_path)
    reference = {"predictions": np.zeros((2, 4), np.float32), "prior": np.zeros((2, 4), np.float32),
                 "prior_mask": np.zeros(2, np.bool_)}
    saved = {"base_predictions": reference["predictions"].copy(), "prior": reference["prior"].copy(),
             "prior_mask": reference["prior_mask"].copy()}
    obj.same_frozen(saved, reference, "fabricated")
    for key in saved:
        changed = copy.deepcopy(saved)
        changed[key].flat[0] = True if changed[key].dtype == np.bool_ else -0.
        with pytest.raises(ValueError, match="equals pretrained"):
            obj.same_frozen(changed, reference, "fabricated")


def full_training_fixture(tmp_path):
    obj = instance(tmp_path)
    train, meta, ids, _ = history_fixture()
    obj.episodes, obj.train_reports = ids, []
    shapes = {"recurrent.weight_ih_l0": (84, 40), "recurrent.weight_hh_l0": (84, 28),
              "recurrent.bias_ih_l0": (84,), "recurrent.bias_hh_l0": (84,),
              "output.weight": (4, 28), "output.bias": (4,)}
    arrays, fits, progress, work, pins = {}, [], [], [], {}
    serial = 0
    for seed in audit.SEEDS:
        parent = None
        for kind in audit.FAMILIES:
            pretrained, frozen = kind == "pretrained", kind.startswith("frozen_")
            checkpoint = {k: np.zeros(shape, np.float32) for k, shape in shapes.items()}
            if not pretrained:
                checkpoint.update({"action_residual.weight": np.zeros((4, 28), np.float32),
                                   "action_residual.bias": np.zeros(4, np.float32)})
            name = f"{kind}-{seed}.npz"
            arrays[name] = checkpoint
            pins[name] = {"sha256": "a" * 64, "bytes": 100}
            full_hash, initial = obj.witness(checkpoint)
            backbone_hash, _ = obj.witness(checkpoint, backbone_only=True)
            rng, permutation = np.random.default_rng(seed), hashlib.sha256()
            orders = [rng.permutation(54).astype(np.int64) for _ in range(audit.EPOCHS[kind])]
            for order in orders:
                permutation.update(order.tobytes())
            sums = {key: 0 for key in audit.WORK_KEYS}
            progress.append({"event": "fit_start", "family": kind, "seed": seed})
            for expected in obj.expected_batches(train, [order.tolist() for order in orders], kind):
                serial += 1
                identity = {"call_id": serial, "family": kind, "seed": seed,
                    "epoch": expected["epoch"]-1, "batch": expected["batch"], "episode_indices": expected["episode_indices"]}
                value = {k: v for k, v in expected.items() if k not in ("epoch", "batch")}
                value.update(loss=0., nonquery_loss=0., prior_loss=0., spo_loss=0., gradient_norm_before_clip=0.,
                             all_parameter_gradients_finite=True)
                work += [{"event": "attempt", **identity}, {"event": "return", **identity, "seconds": 0., "result": value}]
                for key in sums:
                    sums[key] += value[key]
            for epoch in range(20, audit.EPOCHS[kind]+1, 20):
                progress.append({"event": "epoch", "family": kind, "seed": seed, "epoch": epoch,
                                 "optimizer_steps": epoch*9, "episode_exposures": epoch*54})
            keys = list(checkpoint)
            fit = {"family": kind, "seed": seed, "architecture": "innovation_gru",
                "readout": "shared" if pretrained else "protected", "objective": audit.CELLS[kind][2],
                "training_mode": "pretrain" if pretrained else kind.split("_")[0],
                "stage": "pretrain" if pretrained else "adaptation", "parameter_count": 5996 if pretrained else 6112,
                "trainable_parameter_count": 116 if frozen else 5996 if pretrained else 6112,
                "optimizer_parameter_names": keys[-2:] if frozen else keys, "optimizer_initial_state_entries": 0,
                "residual_output_scale": None if pretrained else 64., "checkpoint_path": name, "checkpoint": pins[name],
                "initial_tensors": initial, "initial_sha256": full_hash, "initial_backbone_sha256": backbone_hash,
                "initial_residual_zero": None if pretrained else True, "final_backbone_sha256": backbone_hash,
                "frozen_backbone_unchanged": True if frozen else None,
                "pretrained_checkpoint_path": None if pretrained else parent["checkpoint_path"],
                "pretrained_checkpoint": None if pretrained else parent["checkpoint"],
                "train_base_equals_pretrained": True if frozen else None, "evaluation": audit.evaluation_metadata(kind),
                "epochs": audit.EPOCHS[kind], "steps": audit.EPOCHS[kind]*9,
                "episode_exposures": audit.EPOCHS[kind]*54, "episodes_per_epoch": 54,
                "episode_orders": [order.tolist() for order in orders], "permutation_sha256": permutation.hexdigest(),
                **sums, "final_train_loss": 0., "final_nonquery_loss": 0., "final_prior_loss": 0.,
                "final_objective_prior_loss": 0.,
                "final_prior_loss_scope": "constant diagnostic in frozen mode; trained in pretrain and joint modes",
                "final_spo_loss": 0., "final_objective_spo_loss": 0.,
                "final_spo_loss_scope": "post-fit diagnostic for every cell; trained only by spo cells",
                "final_rescore_spo_loss_calls": 1, "train_rescore": {"forward_chunks": 9, "forward_rows": 270, "prior_rows": 54},
                "fit_seconds": 0., "evaluation_clone_seconds": 0., "train_rescore_seconds": 0.,
                "checkpoint_seconds": 0., "wall_seconds": 0.}
            progress.append({"event": "fit_complete", **fit})
            fits.append(fit)
            if pretrained:
                parent = fit
            arrays["training-prediction-" + name] = {"predictions": np.zeros((270, 4), np.float32),
                "base_predictions": np.zeros((270, 4), np.float32), "prior": np.zeros((270, 4), np.float32),
                "prior_mask": train["prior_mask"].copy()}
    progress.append({"event": "all_checkpoints_closed_before_VALID", "fits_completed": 15,
                     "checkpoints": {row["checkpoint_path"]: row["checkpoint"] for row in fits}})
    obj.worker = {"files": pins, "optimizer_steps": 6480, "training_rescore_chunks": 15*9, "training_rescore_rows": 15*270,
                  **{"training_" + key: sum(fit[key] for fit in fits) for key in audit.WORK_KEYS if key != "no_grad_chunks"}}
    obj.read = lambda path: {"fits": fits, "train_counts": meta["counts"]}
    obj.arrays = lambda path: arrays[path.name]
    obj.rows = lambda path: iter(progress if path.name == "progress.jsonl" else work)
    return obj, train, meta, fits, arrays, progress, work


def test_complete_fabricated_fifteen_fit_barrier_and_independent_losses(tmp_path):
    obj, train, meta, fits, _, progress, work = full_training_fixture(tmp_path)
    assert obj.training(train, meta) == fits
    assert len(progress) == 67 and len(work) == 12960
    assert obj.counts["fits"] == 15 and obj.counts["optimizer_steps"] == 6480
    assert obj.counts["training_prediction_files"] == len(obj.train_reports) == 15
    assert all(row["loss_check"]["spo_float64"] == 0 for row in obj.train_reports)


@pytest.mark.parametrize("defect", ["barrier", "extra_work", "fork_hash", "frozen_checkpoint", "train_base", "train_prior",
                                  "optimizer_membership", "residual_scale", "fit_extra", "false_loss", "canonical_flags"])
def test_saved_defects_fail_instead_of_relaxing_invariants(tmp_path, defect):
    obj, train, meta, fits, arrays, progress, work = full_training_fixture(tmp_path)
    frozen = fits[1]
    if defect == "barrier":
        progress.pop()
    elif defect == "extra_work":
        work.append(None)
    elif defect == "fork_hash":
        frozen["initial_sha256"] = "b" * 64
    elif defect == "frozen_checkpoint":
        arrays[frozen["checkpoint_path"]]["output.bias"][0] = .1
    elif defect in ("train_base", "train_prior"):
        saved = arrays["training-prediction-" + frozen["checkpoint_path"]]
        saved["base_predictions" if defect == "train_base" else "prior"][1 if defect == "train_base" else 4, 0] = .1
    elif defect == "optimizer_membership":
        frozen["optimizer_parameter_names"].append("output.weight")
    elif defect == "residual_scale":
        frozen["residual_output_scale"] = 1.
    elif defect == "fit_extra":
        frozen["selected_best"] = True
    elif defect == "false_loss":
        frozen["final_nonquery_loss"] = 1.
    else:
        frozen["evaluation"]["grad_enabled"] = True
    # Keep journal metadata synchronized to isolate the intended independent check.
    for i, row in enumerate(progress):
        if row.get("event") == "fit_complete" and row.get("family") == frozen["family"] and row.get("seed") == frozen["seed"]:
            progress[i] = {"event": "fit_complete", **frozen}
    with pytest.raises((ValueError, StopIteration)):
        obj.training(train, meta)


def test_arithmetic_is_inherited_and_no_producer_metric_or_loss_is_imported():
    assert audit.scalar_training_loss is audit.old.scalar_training_loss
    assert audit.scalar_spo_loss is audit.old.scalar_spo_loss
    assert audit.scalar_metrics is audit.old.scalar_metrics
    assert audit.Audit.sampled_training is audit.old.Audit.sampled_training
    assert audit.Audit.capacity is audit.old.Audit.capacity
    assert hashlib.sha256((ROOT / audit.HELPER).read_bytes()).hexdigest() == audit.HELPER_PIN
    tree = ast.parse((ROOT / audit.SELF).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(name.startswith(("torch", "tensorflow", "mlx", "openjev")) for name in names)


def test_fresh_cohort_and_prospective_limits():
    cohort = audit.cohort()
    assert len(cohort) == 90 and sum(row["stage"] == "train" for row in cohort) == 54
    assert {row["seed"] for row in cohort if row["stage"] == "valid"} == set(range(299000001, 299000007)) | set(range(300000001, 300000007))
    assert audit.LIMITS == {"seconds": 300, "rss_bytes": 2*1024**3, "output_bytes": 256*1024**2}
    assert sum(audit.EPOCHS[kind]*9 for kind in audit.FAMILIES for _ in audit.SEEDS) == 6480


def test_metadata_only_trainer_config_and_saved_schema_match_independent_auditor():
    # Import only top-level metadata. No Run instance, tensor import, model or
    # producer loss/metric routine is executed or used as an arithmetic oracle.
    path = ROOT / "scripts/train_otto_protected_readout.py"
    spec = importlib.util.spec_from_file_location("_protected_trainer_metadata_only", path)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    assert producer.CONFIG == audit.CONFIG
    assert producer.VERSION == audit.PRODUCER_VERSION
    assert producer.KINDS == audit.FAMILIES and producer.SEEDS == audit.SEEDS
    assert producer.EPOCHS == audit.EPOCHS and producer.CELLS == audit.CELLS
    assert producer.CAPACITY_SCOPE == audit.CAPACITY_SCOPE
    assert len(producer.PAYLOADS) == 58
    tree = ast.parse(path.read_text())
    fit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fit")
    saved = next(node.value for node in ast.walk(fit) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "fit" for target in node.targets))
    literal_keys = {key.value for key in saved.keys if isinstance(key, ast.Constant)}
    assert literal_keys | set(audit.WORK_KEYS) == audit.FIT_KEYS
