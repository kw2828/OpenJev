"""Constructed saved-array audits only; no model, optimizer, corpus or simulator."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("innovation_saved_audit", Path(__file__).with_name("audit_pilot.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture_predictions():
    p = np.zeros((2, 19, 8), dtype=np.float32)
    p[..., :2], p[..., 4], p[..., 6] = 1, .1, 1
    for start in (2, 9):
        p[:, start:start + 2, :4] = 0
        p[:, start:start + 2, 6] = 0
    for case in range(2):
        last = 0
        for step in range(19):
            if p[case, step, 6]: last = step
            p[case, step, 7] = (step - last) * .02
    data = {"packets": torch.from_numpy(p), "commands": torch.zeros(2, 18, 2), "rewards": torch.full((2, 18), -.4)}
    # Fixed arithmetic values, not predictions from any model.
    pred = {"one_step_mean": torch.full((2, 18, 4), .25), "one_step_variance": torch.full((2, 18, 4), .5),
        "one_step_reward": torch.full((2, 18), -.2), "open_loop_mean": torch.full((2, 14, 5, 4), .25),
        "open_loop_variance": torch.full((2, 14, 5, 4), .5), "open_loop_reward": torch.full((2, 14, 5), -.2)}
    pred["one_step_target_valid"] = data["packets"][:, 1:, 6] == 1
    pred["reacquisition_prior_mask"] = pred["one_step_target_valid"] & (data["packets"][:, :-1, 6] == 0)
    roots = torch.zeros(2, 14, dtype=torch.bool)
    roots[:, [4, 11]] = True
    pred["post_reacquisition_root_mask"] = roots
    targets = torch.stack([data["packets"][:, 1 + i:15 + i] for i in range(5)], 2)
    pred["post_reacquisition_target_valid"] = roots[..., None] & (targets[:, :, [0, 2], 6] == 1)
    pred["post_reacquisition_squared_error"] = (pred["open_loop_mean"][:, :, [0, 2]] - targets[:, :, [0, 2], :4]).square().mean(-1)
    return pred, data


def test_prediction_metrics_independent_known_arithmetic_and_masks():
    pred, data = fixture_predictions()
    rows = audit.prediction_metrics(pred, data, audit.BASE_CONFIG)
    assert len(rows) == 2
    for row in rows:
        assert row["one_step_angle_mse"] == row["post_reacquisition_mean_mse"] == .3125
        assert row["post_reacquisition_h1_mse"] == row["post_reacquisition_h3_mse"] == .3125
        assert row["one_step_reward_mse"] == pytest.approx(.04)
        assert row["residual_moment_score"] == pytest.approx(.5 * (4 * np.log(.5) + 2.5))
        assert row["counts"]["reacquisition_targets"] == row["counts"]["post_reacquisition_complete_roots"] == 2
    audit.compare_rows(rows, rows)


@pytest.mark.parametrize("bad", ["mean_nan", "negative_variance", "missing_array", "mask", "errors", "partial_recovery", "dtype"])
def test_bad_predictions_or_masks_rejected(bad):
    pred, data = fixture_predictions()
    if bad == "mean_nan": pred["one_step_mean"][0, 0, 0] = float("nan")
    elif bad == "negative_variance": pred["open_loop_variance"][0, 0, 0, 0] = -.1
    elif bad == "missing_array": del pred["one_step_reward"]
    elif bad == "mask": pred["one_step_target_valid"][0, 0] = False
    elif bad == "errors": pred["post_reacquisition_squared_error"][0, 4, 0] += .1
    elif bad == "dtype": pred["open_loop_mean"] = pred["open_loop_mean"].double()
    else:
        data["packets"][:, 11, 6] = 0
    with pytest.raises(ValueError): audit.prediction_metrics(pred, data, audit.BASE_CONFIG)


def test_bound_fraction_uses_torch_float32_threshold_not_python_double():
    pred, data = fixture_predictions()
    threshold = np.float32(1.01 * audit.BASE_CONFIG["variance_min"])
    pred["one_step_variance"].fill_(float(threshold))
    rows = audit.prediction_metrics(pred, data, audit.BASE_CONFIG)
    assert all(row["variance_lower_bound_fraction"] == 1. for row in rows)


def all_evaluations():
    return {(variant, pair, stage, panel): [{"episode": i,
        "post_reacquisition_mean_mse": .8 if variant == "normalized" and stage == "final" else 1.,
        "one_step_angle_mse": 1., "one_step_reward_mse": 1.} for i in range(2)]
        for variant in audit.VARIANTS for pair in range(3) for stage in audit.STAGES for panel in audit.PANELS}


def test_all29criteria_inclusive_and_preserve_every_family_pair_stage_panel():
    values = all_evaluations()
    result = audit.criteria(values)
    assert result["continuation"]["passed"] and result["continuation"]["checks_passed"] == 29
    assert len(result["fit_means"]) == 48 and len(result["family_means"]) == 16
    for key, rows in values.items():
        if key[0] == "normalized" and key[2] == "initial":
            for row in rows: row["post_reacquisition_mean_mse"] = .8 / .9
    assert audit.criteria(values)["continuation"]["passed"]


def test_good_family_mean_cannot_rescue_one_failed_pair():
    values = all_evaluations()
    for row in values["normalized", 0, "final", "dev10"]:
        row["post_reacquisition_mean_mse"] = 1.01
    gate = audit.criteria(values)["continuation"]
    assert gate["checks_passed"] == 27 and not gate["passed"]
    failed = [row["name"] for row in gate["checks"] if not row["passed"]]
    assert failed == ["dev10/normalized/age/pair0_recovery", "dev10/normalized/raw/pair0_recovery"]


@pytest.mark.parametrize("metric,ratio", [("one_step_angle_mse", 1.02), ("one_step_reward_mse", 1.05)])
def test_prediction_guard_is_not_replaced_by_primary_improvement(metric, ratio):
    values = all_evaluations()
    for pair in range(3):
        for row in values["normalized", pair, "final", "dev6"]: row[metric] = ratio + .01
    result = audit.criteria(values)["continuation"]
    assert result["checks_passed"] == 27 and not result["passed"]


def test_missing_evaluation_is_not_complete_evidence():
    values = all_evaluations()
    del values["constant", 2, "initial", "dev10"]
    with pytest.raises(ValueError, match="All48"): audit.criteria(values)


def test_real_shaped_outer_members_and_nested_files_are_distinct(tmp_path):
    (tmp_path / "data.pt").write_bytes(b"synthetic payload")
    members = audit.inventory(tmp_path, float("inf"))
    audit.write(tmp_path / "completed.json", {"members": members})
    receipt = audit.read(tmp_path / "completed.json")
    assert len(audit.bound_tree(tmp_path, receipt, {"data.pt"}, float("inf"), member_key="members")) == 2
    with pytest.raises(KeyError): audit.bound_tree(tmp_path, receipt, {"data.pt"}, float("inf"))
    (tmp_path / "data.pt").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="hashes"): audit.bound_tree(tmp_path, receipt, {"data.pt"}, float("inf"), member_key="members")


def test_extra_tree_members_rejected_and_no_path_escape(tmp_path):
    audit.write(tmp_path / "completed.json", {"files": {}})
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="membership"): audit.bound_tree(tmp_path, {"files": {}}, set(), float("inf"))
    with pytest.raises(ValueError, match="Safe"): audit.child(tmp_path, "../elsewhere")


def test_hash_primitives_match_saved_tensor_contract():
    # Import only the historical hashing primitive, never any model constructor.
    from openjev.research.reacher_objective_training import canonical_state_hash, canonical_tensor_hash
    value = {"x": torch.arange(6, dtype=torch.float32).reshape(2, 3), "z": torch.tensor([True])}
    assert audit.tensor_hash(value) == canonical_tensor_hash(value)
    nested = {"tensors": value, "optimizer": {0: {"step": torch.tensor(2.)}}, "tuple": (.9, .999)}
    assert audit.state_hash(nested) == canonical_state_hash(nested)


def fake_fit(folder):
    folder.mkdir()
    cfg = {**audit.BASE_CONFIG, "variant": "normalized", "hidden_size": 4, "context_size": 4, "epochs": 1, "batch_size": 2}
    weights = {key: torch.zeros(shape) for key, shape in audit.parameter_shapes(cfg).items()}
    orders = torch.tensor([[0, 1]])
    train = {"packets": torch.zeros(2, 7, 8), "commands": torch.zeros(2, 6, 2), "rewards": torch.zeros(2, 6)}
    train["packets"][..., 6] = 1
    pair = {"initial_tensor_sha256": audit.tensor_hash(weights), "orders_tensor_sha256": audit.tensor_hash({"orders": orders})}
    plan = {"configurations": {"normalized": cfg}, "source_sha256": {"fixture": "a" * 64}, "runtime": {"synthetic": True},
        "data_tensor_sha256": {"train": audit.tensor_hash(train)}, "cap_seconds": 100}
    binding = {"version": "reacher-innovation-pilot-v1", "config": cfg, "recipe": audit.RECIPE, "adam": audit.ADAM,
        "source_sha256": plan["source_sha256"], "runtime": plan["runtime"], "constructor_seed": 410,
        "constructor_tensors_overwritten": True, "initial_sha256": pair["initial_tensor_sha256"],
        "orders_sha256": pair["orders_tensor_sha256"], "data_sha256": plan["data_tensor_sha256"]["train"]}
    counts = {"attempted_updates": 1, "optimizer_steps": 1, "flushed_updates": 1}
    opt = {"state": {i: {"step": torch.tensor(1.), "exp_avg": torch.zeros_like(v), "exp_avg_sq": torch.zeros_like(v)}
                     for i, v in enumerate(weights.values())},
           "param_groups": [{**audit.ADAM, "params": list(range(len(weights))), "lr": .001}]}
    ckpt = {"version": binding["version"], "binding": binding, "model_configuration": {**cfg, "model_class": "InnovationContextWorldModel"},
        "weights": weights, "optimizer": opt, "parameter_names": list(weights), "counts": counts, "resume_authorized": False}
    audit.write(folder / "started.json", binding)
    for name, value in (("initial-weights.pt", weights), ("weights.pt", weights), ("epoch-orders.pt", orders), ("checkpoint.pt", ckpt)):
        torch.save(value, folder / name)
    modules = {name: {"calls": 6, "samples": 12} for name in ("observation_update", "context_innovation", "gate")}
    modules.update({name: {"calls": 11, "samples": 32} for name in ("transition", "observation_head", "variance_head", "reward_head.0", "reward_head.2")})
    log = {"epoch": 0, "batch": 0, "update": 1, "indices": [0, 1], "indices_sha256": audit.tensor_hash({"indices": orders[0]}),
        "metrics": {"observation_mse": 1., "reward_mse": 1., "rollout_observation_mse": 1., "rollout_reward_mse": 1.,
            "prediction_objective": 7.5, "residual_moment_score": -1., "loss": 7.4,
            "valid_observation_targets": 12., "valid_variance_targets": 12., "valid_rollout_starts": 4.},
        "gradient_clip": audit.RECIPE["gradient_clip"], "gradient_norm_before_clip": {"backbone": 1., "variance_head": 11.},
        "gradient_clipped": {"backbone": False, "variance_head": True}, "fit_elapsed_seconds_before_log": 1., "update_seconds_before_log": .5,
        "work": {"modules": modules, "expected_action_cost_samples": 32, "completed_forward_only": True, "excludes": "synthetic"}}
    (folder / "training.jsonl").write_text(json.dumps(log) + "\n")
    done = {"status": "completed", **binding, "counts": counts, "episodes": 2, "steps": 6,
        "evaluation_performed": False, "automatic_retry": False, "resume_authorized": False, "wall_seconds": 2.,
        "final_weights_sha256": audit.tensor_hash(weights), "checkpoint_sha256": audit.state_hash(ckpt),
        "files": audit.inventory(folder, float("inf"))}
    audit.write(folder / "completed.json", done)
    return plan, pair, train, weights, orders


def test_handbuilt_checkpoint_log_and_adam_accounting_without_optimizer(tmp_path):
    args = fake_fit(tmp_path / "fit")
    row = audit.audit_fit(tmp_path / "fit", args[0], args[1], "normalized", *args[2:], float("inf"))
    assert row["counts"]["optimizer_steps"] == 1


@pytest.mark.parametrize("bad", ["order", "clipping", "loss", "work", "adam_step", "class", "checkpoint_input", "checkpoint_version"])
def test_fit_corruption_rejected_even_with_resealed_file_hashes(tmp_path, bad):
    folder = tmp_path / "fit"
    args = fake_fit(folder)
    done = audit.read(folder / "completed.json")
    if bad in {"adam_step", "class", "checkpoint_input", "checkpoint_version"}:
        payload = audit.load(folder / "checkpoint.pt")
        if bad == "adam_step": payload["optimizer"]["state"][0]["step"].fill_(2)
        elif bad == "class": payload["model_configuration"]["model_class"] = "AnotherClass"
        elif bad == "checkpoint_input": payload["binding"]["data_sha256"] = "b" * 64
        else: payload["version"] = "wrong"
        torch.save(payload, folder / "checkpoint.pt")
        done["checkpoint_sha256"] = audit.state_hash(payload)
    else:
        row = json.loads((folder / "training.jsonl").read_text())
        if bad == "order": row["indices"] = [1, 0]
        elif bad == "clipping": row["gradient_clipped"]["variance_head"] = False
        elif bad == "loss": row["metrics"]["loss"] += .1
        else: row["work"]["expected_action_cost_samples"] -= 1
        (folder / "training.jsonl").write_text(json.dumps(row) + "\n")
    done["files"] = {name: value for name, value in audit.inventory(folder, float("inf")).items() if name != "completed.json"}
    (folder / "completed.json").write_text(json.dumps(done))
    with pytest.raises(ValueError): audit.audit_fit(folder, args[0], args[1], "normalized", *args[2:], float("inf"))


def test_bad_external_digest_preserves_failure_and_cannot_retry(tmp_path):
    protocol, execution, out = tmp_path / "protocol", tmp_path / "execution", tmp_path / "audit"
    protocol.mkdir(); execution.mkdir()
    (protocol / "plan.json").write_text("{}")
    (execution / "completed.json").write_text("{}")
    with pytest.raises(ValueError, match="External"):
        audit.audit(protocol / "plan.json", "f" * 64, execution, "a" * 64, out, root=tmp_path, audit_cap_seconds=10)
    assert audit.read(out / "failed.json")["complete_evidence_passed"] is False
    with pytest.raises(FileExistsError):
        audit.audit(protocol / "plan.json", "f" * 64, execution, "a" * 64, out, root=tmp_path, audit_cap_seconds=10)


def test_audit_cap_and_input_output_separation(tmp_path):
    with pytest.raises(TimeoutError): audit.check(time.monotonic() - 1)
    with pytest.raises(ValueError, match="cannot change"):
        audit.audit(tmp_path / "protocol/plan.json", "a" * 64, tmp_path / "execution", "b" * 64,
                    tmp_path / "execution/audit", root=tmp_path, audit_cap_seconds=10)


def test_registry_binds_every_pair_to_six_declared_roles():
    plan = {"random_registry": {f"pair{p}/{kind}": 2 * p + j for p in range(3)
                              for j, kind in enumerate(("initialization", "orders"))},
            "pairs": [{"pair": p, "initialization_seed": 2 * p, "orders_seed": 2 * p + 1,
                       "torch_initial_rng_sha256": "a" * 64, "torch_final_rng_sha256": "b" * 64}
                      for p in range(3)]}
    audit.validate_registry(plan)
    plan["pairs"][1]["orders_seed"] = 9
    with pytest.raises(ValueError, match="bound to registry"): audit.validate_registry(plan)


def outer_fixture(tmp_path, monkeypatch, *, bad_index=False, excess_cost=False):
    """Wrapper-only 280-payload fixture; numerical/source reducers are explicit stubs."""
    protocol, execution = tmp_path / "protocol", tmp_path / "execution"
    protocol.mkdir(); execution.mkdir()
    public = {}
    for name in ("train", "dev6", "dev10"):
        n = 640 if name == "train" else 128
        public[name] = {"packets": torch.zeros(n, 51, 8), "commands": torch.zeros(n, 50, 2), "rewards": torch.zeros(n, 50)}
        torch.save(public[name], protocol / f"{name}.pt")
    audit.write(protocol / "split.json", {"synthetic": True})
    initial = {"synthetic": torch.tensor([0.])}
    initial_hash = audit.tensor_hash(initial)
    # Literal engineering410 only, unrelated to prospective scored role allocation.
    generator = np.random.default_rng(410)
    before = generator.bit_generator.state
    orders = torch.from_numpy(np.stack([generator.permutation(640) for _ in range(8)]))
    after = generator.bit_generator.state
    pairs = []
    for p in range(3):
        torch.save(initial, protocol / f"initial-pair{p}.pt")
        torch.save(orders, protocol / f"orders-pair{p}.pt")
        pairs.append({"pair": p, "initial_tensor_sha256": initial_hash,
            "orders_tensor_sha256": audit.tensor_hash({"orders": orders}), "orders_seed": 410,
            "numpy_initial_rng": before, "numpy_final_rng": after})
    plan = {"pairs": pairs, "configurations": {v: {**audit.BASE_CONFIG, "variant": v} for v in audit.VARIANTS},
        "source_sha256": {}, "runtime": {"synthetic": True}, "cap_seconds": 1800,
        "data_tensor_sha256": {k: audit.tensor_hash(v) for k, v in public.items()}, "preparation_wall_seconds": 1.}
    audit.write(protocol / "plan.json", plan)
    plan_sha = audit.sha(protocol / "plan.json")
    audit.write(execution / "started.json", {"utc": "2026-01-01T00:00:00+00:00", "plan_sha256": plan_sha, "no_retry": True})
    rows = [{"episode": i, "post_reacquisition_mean_mse": 1., "one_step_angle_mse": 1., "one_step_reward_mse": 1.} for i in range(128)]
    modules = {name: {"calls": 50, "samples": 1600} for name in ("observation_update", "context_innovation", "gate")}
    modules.update({name: {"calls": 55, "samples": 8960} for name in ("transition", "observation_head", "variance_head", "reward_head.0", "reward_head.2")})
    work = {"modules": modules, "expected_action_cost_samples": 8960, "completed_forward_only": True, "excludes": "synthetic"}
    fit_boundaries, index = {}, []
    for p in range(3):
        for variant in audit.VARIANTS:
            name = f"pair{p}-{variant}"
            folder = execution / "fits" / name
            folder.mkdir(parents=True)
            for member in audit.FIT_FILES: (folder / member).write_bytes(b"synthetic fit payload")
            done = {"wall_seconds": 1., "final_weights_sha256": initial_hash, "files": audit.inventory(folder, float("inf"))}
            audit.write(folder / "completed.json", done)
            fit_boundaries[name] = {"receipt": done, "completed_sha256": audit.sha(folder / "completed.json")}
            binding = {"version": "reacher-innovation-pilot-v1", "config": plan["configurations"][variant],
                "recipe": audit.RECIPE, "adam": audit.ADAM, "source_sha256": {}, "runtime": plan["runtime"],
                "constructor_seed": 410, "constructor_tensors_overwritten": True, "weights_sha256": initial_hash}
            for stage in audit.STAGES:
                for panel in audit.PANELS:
                    folder = execution / "evaluation" / name / stage / panel
                    folder.mkdir(parents=True)
                    row_binding = {**binding, "data_sha256": plan["data_tensor_sha256"][panel]}
                    audit.write(folder / "started.json", row_binding)
                    torch.save({}, folder / "predictions.pt")
                    summary = {"episodes": 128, "steps": 50, "horizon": 5, "counts": {"optimizer_steps": 0, "completed_batches": 4},
                        "per_episode": rows, "work_by_batch": [work] * 4}
                    audit.write(folder / "summary.json", summary)
                    audit.write(folder / "completed.json", {"status": "completed", **row_binding,
                        "split": "development", "new_optimizer_steps": 0, "counts": summary["counts"],
                        "wall_seconds": 4. if excess_cost else 1., "files": audit.inventory(folder, float("inf"))})
                    index.append({"pair": p, "variant": variant, "stage": stage, "panel": panel, "summary": summary})
    audit.write(execution / "all-fits-completed.json", {"utc": "2026-01-01T00:01:00+00:00", "fits": fit_boundaries})
    audit.write(execution / "evaluation-started.json", {"utc": "2026-01-01T00:01:01+00:00",
        "all_fits_completed_sha256": audit.sha(execution / "all-fits-completed.json")})
    if bad_index: index[0], index[1] = index[1], index[0]
    audit.write(execution / "evaluation-index.json", index)
    members = audit.inventory(execution, float("inf"))
    assert len(members) == 280
    audit.write(execution / "completed.json", {"utc": "2026-01-01T00:03:00+00:00", "plan_sha256": plan_sha,
        "fits": 12, "evaluations": 48, "development_only": True, "native_control_measured": False,
        "wall_seconds": 100., "members": members})
    monkeypatch.setattr(audit, "validate_plan", lambda plan, folder, root, deadline: audit.inventory(folder, deadline))
    monkeypatch.setattr(audit, "public_splits", lambda *args: None)
    monkeypatch.setattr(audit, "weights_schema", lambda *args: initial_hash)
    monkeypatch.setattr(audit, "prediction_metrics", lambda *args: rows)
    def stub_fit(folder, *args):
        done = audit.read(folder / "completed.json")
        return {**done, "receipt": done, "completed_sha256": audit.sha(folder / "completed.json")}
    monkeypatch.setattr(audit, "audit_fit", stub_fit)
    return protocol / "plan.json", plan_sha, execution, audit.sha(execution / "completed.json")


@pytest.mark.parametrize("failure", [None, "index", "cost", "terminal_cap"])
def test_outer280payload_wrapper_success_and_failure_receipts(tmp_path, monkeypatch, failure):
    args = outer_fixture(tmp_path, monkeypatch, bad_index=failure == "index", excess_cost=failure == "cost")
    out = tmp_path / "audit"
    if failure == "terminal_cap":
        original = audit.write
        def expire_after_terminal(path, value):
            original(path, value)
            if path.name == "receipt.json":
                monkeypatch.setattr(audit, "check", lambda deadline: (_ for _ in ()).throw(TimeoutError("synthetic terminal cap")))
        monkeypatch.setattr(audit, "write", expire_after_terminal)
    if failure:
        with pytest.raises((ValueError, TimeoutError)):
            audit.audit(*args, out, root=tmp_path, audit_cap_seconds=60)
        assert audit.read(out / "failed.json")["complete_evidence_passed"] is False
        assert not (out / "receipt.json").exists()
        if failure == "terminal_cap": assert (out / "invalid-receipt.json").exists()
    else:
        summary = audit.audit(*args, out, root=tmp_path, audit_cap_seconds=60)
        receipt = audit.read(out / "receipt.json")
        assert summary["status"] == "completed" and summary["complete_evidence_passed"] is True
        assert receipt["qualification_passed"] is False and receipt["checks_passed"] == 22
        assert len(receipt["execution_members"]) == 281
        assert summary["costs"]["fit_wall_seconds_nested"] == 12
        assert summary["costs"]["evaluation_wall_seconds_nested"] == 48
        assert audit.sha(out / "summary.json") == receipt["files"]["summary.json"]["sha256"]
