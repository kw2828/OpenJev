"""Independent saved-artifact corruption tests, synthetic seed410 only."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_reacher_cache_study as audit

from openjev.research import reacher_cache_control as control
from openjev.research import reacher_cache_protocol as protocol
from openjev.research import reacher_cache_training as training


@pytest.fixture(autouse=True)
def isolated_engineering_rng():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(410)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def plan():
    value = protocol.settings(engineering=True)
    value.update(hidden_size=4, mlp_width=7, train_episodes=2, batch_size=1,
                 epochs=1, steps=6, control_episodes=2, planning_horizon=2,
                 sources={"synthetic": "a" * 64}, runtime={"engineering": True},
                 bootstrap_samples=64)
    return value


def public_data(cfg):
    count, steps = cfg["train_episodes"], cfg["steps"]
    angles = torch.linspace(-.3, .4, count * (steps + 1) * 2).reshape(count, steps + 1, 2)
    packets = torch.zeros(count, steps + 1, 8)
    packets[..., :2], packets[..., 2:4] = angles.cos(), angles.sin()
    packets[..., 4:6] = torch.tensor([.1, -.05])
    packets[..., 6] = 1
    packets[:, 2:4, :4] = 0
    packets[:, 2:4, 6] = 0
    packets[:, 2:4, 7] = torch.tensor([.02, .04])
    commands = torch.linspace(-.2, .2, count * steps * 2).reshape(count, steps, 2)
    return {"packets": packets, "commands": commands,
            "rewards": -commands.square().sum(-1) - .1}


def fit(arm):
    cfg = plan()
    settings = training.CacheTrainingSettings(**{key: cfg[key] for key in audit.TRAINING_FIELDS})
    initial = training.make_initialization(410, 410, settings)
    order = training.make_orders(410, settings)
    data = public_data(cfg)
    trainer = training.make_trainer(arm, initial, order, data,
                                    source_sha256=cfg["sources"], runtime=cfg["runtime"])
    rows = [trainer.train_next() for _ in range(2)]
    return cfg, trainer, initial, order, rows, data


def reseal(payload):
    return training._seal({key: value for key, value in payload.items() if key != "integrity_sha256"})


def check_checkpoint(cfg, trainer, initial, order, payload=None, arm=None):
    return audit.audit_checkpoint(cfg, payload or trainer.export_checkpoint(), initial, order,
                                  arm or trainer.kind, trainer.data_sha256)


def write_logs(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def rechain(rows, checkpoint):
    chain = training.canonical_state_hash([])
    for row in rows:
        row["previous_log_sha256"] = chain
        row.pop("log_sha256", None)
        chain = training.canonical_state_hash(row)
        row["log_sha256"] = chain
    checkpoint["log_chain_sha256"] = chain


@pytest.mark.parametrize("arm", protocol.ARMS)
def test_tensor_only_initialization_reconstruction_matches_actual_constructor(arm):
    cfg = plan()
    settings = training.CacheTrainingSettings(**{key: cfg[key] for key in audit.TRAINING_FIELDS})
    initial = training.make_initialization(410, 410, settings)
    expected = initial["states"]["mlp" if arm in ("packet_mlp", "cached_mlp") else "gru"]
    before = torch.get_rng_state().clone()
    actual = audit.seeded_tensors(cfg, arm, 410)
    assert list(actual) == list(expected)
    assert audit.canonical_tensor_hash(actual) == training.canonical_tensor_hash(expected)
    assert torch.equal(before, torch.get_rng_state())
    assert audit.training_configuration(cfg) == settings.configuration()
    assert audit.model_configuration(cfg, arm) == training.model_configuration(arm, settings)


@pytest.mark.parametrize("arm", protocol.ARMS)
def test_actual_two_update_checkpoint_and_logs_pass_independent_arithmetic(arm, tmp_path):
    cfg, trainer, initial, order, rows, data = fit(arm)
    body = check_checkpoint(cfg, trainer, initial, order)
    path = tmp_path / "training.jsonl"
    write_logs(path, rows)
    summary = audit.audit_logs(cfg, path, arm, order, body, packets=data["packets"].numpy())
    assert summary["updates"] == 2
    assert body["model_configuration"]["model_class"] == type(trainer.student).__name__
    assert body["model_configuration"]["run_status_authority"] == "enclosing protocol and execution receipts"


@pytest.mark.parametrize("original,wrong", [
    ("residual_gru", "encoded_current_gru"), ("encoded_current_gru", "cached_gru"), ("cached_gru", "residual_gru")])
def test_identical_gru_tensor_schema_cannot_hide_wrong_semantic_class(original, wrong):
    cfg, trainer, initial, order, _, _ = fit(original)
    assert audit.shapes(cfg, original) == audit.shapes(cfg, wrong)
    with pytest.raises(ValueError, match="Actual class"):
        check_checkpoint(cfg, trainer, initial, order, arm=wrong)
    payload = trainer.export_checkpoint()
    payload["model_configuration"]["model_class"] = protocol.MODEL_CLASSES[wrong]
    with pytest.raises(ValueError, match="Actual class"):
        check_checkpoint(cfg, trainer, initial, order, reseal(payload))


@pytest.mark.parametrize("mutation", ["step", "negative_moment", "missing_slot", "parameter_order", "cursor", "stale_status", "source", "nonfinite_weight"])
def test_rehashed_checkpoint_semantic_corruption_is_rejected(mutation):
    cfg, trainer, initial, order, _, _ = fit("residual_gru")
    payload = trainer.export_checkpoint()
    if mutation == "step":
        payload["optimizer_state"]["state"][0]["step"].sub_(1)
    elif mutation == "negative_moment":
        payload["optimizer_state"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif mutation == "missing_slot":
        payload["optimizer_state"]["state"].pop(0)
    elif mutation == "parameter_order":
        payload["optimizer_group_names"][0].reverse()
    elif mutation == "cursor":
        payload["cursor"] = {"epoch": 0, "batch": 0}
    elif mutation == "stale_status":
        payload["settings"]["engineering_only"] = True
    elif mutation == "source":
        payload["source_sha256"] = {"synthetic": "b" * 64}
    else:
        next(iter(payload["student_state"].values())).flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        check_checkpoint(cfg, trainer, initial, order, reseal(payload))


@pytest.mark.parametrize("mutation", ["order", "update", "loss", "target_count", "rollout_count", "work", "clip", "timing", "chain", "missing_row"])
def test_training_log_arithmetic_and_public_masks_are_independently_checked(mutation, tmp_path):
    cfg, trainer, initial, order, rows, data = fit("cached_gru")
    body = check_checkpoint(cfg, trainer, initial, order)
    rows = copy.deepcopy(rows)
    if mutation == "order":
        rows[0]["indices"] = [1 - rows[0]["indices"][0]]
        rows[0]["indices_sha256"] = training.canonical_tensor_hash({"indices": torch.tensor(rows[0]["indices"])})
    elif mutation == "update":
        rows[0]["update"] = 2
    elif mutation == "loss":
        rows[0]["loss"] += 1
    elif mutation == "target_count":
        rows[0]["anchor"]["valid_observation_targets"] += 1
    elif mutation == "rollout_count":
        rows[0]["anchor"]["valid_rollout_starts"] += 1
    elif mutation == "work":
        rows[0]["work"]["operations"]["cache_update_opportunity_samples"] += 1
    elif mutation == "clip":
        rows[0]["gradient_clipped"] = not rows[0]["gradient_clipped"]
    elif mutation == "timing":
        rows[0]["costs"]["optimizer_seconds"] = -1
    elif mutation == "missing_row":
        rows.pop()
    if mutation != "chain":
        rechain(rows, body)
    else:
        rows[0]["previous_log_sha256"] = "0" * 64
    path = tmp_path / "training.jsonl"
    write_logs(path, rows)
    with pytest.raises(ValueError):
        audit.audit_logs(cfg, path, "cached_gru", order, body, packets=data["packets"].numpy())


def state_fixture(tmp_path, arm):
    cfg = plan()
    cfg["steps"] = 50
    data = public_data(cfg)
    width = {"width": cfg["mlp_width"]} if arm in ("packet_mlp", "cached_mlp") else {"hidden_size": cfg["hidden_size"]}
    model = training.REGISTRY[arm](**width, noise_std=cfg["noise_std"]).eval()
    n, steps = cfg["control_episodes"], cfg["steps"]
    state = model.initial(n)
    arrays, angles, rewards, rows = {}, [], [], []
    with torch.no_grad():
        for step in range(steps):
            root = model.assimilate(state, data["packets"][:, step])
            state, predicted, reward = model.advance(root, data["commands"][:, step])
            for phase, values in (("root", root), ("carried", state)):
                for name, value in values.items():
                    arrays.setdefault(f"{phase}__{name}", []).append(value.clone().numpy())
            angles.append(predicted.numpy())
            rewards.append(reward.numpy())
            payload_bytes = sum(value.numel() * value.element_size() for value in root.values())
            imagined = n * 256 * min(cfg["planning_horizon"], steps-step)
            rows.append({"step": step, "root_sha256": training.canonical_tensor_hash(root),
                         "carried_sha256": training.canonical_tensor_hash(state),
                         "root_tensor_bytes": payload_bytes, "carried_tensor_bytes": payload_bytes,
                         "search": {"candidate_evaluations": n*256, "imagined_transitions": imagined,
                                    "root_tensor_bytes": payload_bytes,
                                    "max_single_candidate_state_tensor_bytes": payload_bytes*64,
                                    "tensor_bytes_are_not_peak_process_memory": True,
                                    "model_work": control.work_accounting(model, assimilate_samples=0, advance_samples=imagined)},
                         "model_work": control.work_accounting(model, assimilate_samples=n, advance_samples=n+imagined)})
    arrays = {key: np.stack(values, axis=1) for key, values in arrays.items()}
    fit_info = {"student_tensor_sha256": training.canonical_tensor_hash(model.state_dict()),
                "parameters": sum(p.numel() for p in model.parameters())}
    metadata = {"model": control.model_identity(model, cfg), "steps": rows,
                "state_arrays_bytes": sum(value.nbytes for value in arrays.values()),
                "max_single_candidate_state_tensor_bytes": rows[0]["search"]["max_single_candidate_state_tensor_bytes"],
                "tensor_bytes_are_not_peak_process_memory": True,
                "aggregate_model_work": control.work_accounting(model, assimilate_samples=n*steps, advance_samples=n*steps+sum(r["search"]["imagined_transitions"] for r in rows)),
                "auxiliary_teacher_calls": 0, "reset_calls": 0}
    np.savez_compressed(tmp_path / "states.npz", **arrays)
    np.savez_compressed(tmp_path / "executed_predictions.npz", angles=np.stack(angles, axis=1), rewards=np.stack(rewards, axis=1))
    (tmp_path / "state-work.json").write_text(json.dumps(metadata))
    records = [{"policy": {key: data[key][i].numpy() for key in ("packets", "commands")}} for i in range(n)]
    return cfg, arrays, metadata, fit_info, records


@pytest.mark.parametrize("arm", protocol.ARMS)
def test_saved_real_state_buffers_of_each_actual_class_pass_without_model_rerun(arm, tmp_path):
    cfg, _, _, info, records = state_fixture(tmp_path, arm)
    result = audit.audit_states(cfg, tmp_path, records, arm, info)
    assert result["public_state_buffers_checked"] is True
    assert result["actual_class"] == protocol.MODEL_CLASSES[arm]


@pytest.mark.parametrize("mutation", ["stale_packet", "stale_cache", "predicted_cache", "wrong_real_index", "wrong_last_index", "target", "imagined_root", "candidate_carry", "encoder_features", "carried_features", "missing_persistence", "dtype", "state_hash", "work"])
def test_cache_public_history_phase_and_cost_corruption_rejected(mutation, tmp_path):
    arm = "residual_gru" if mutation == "missing_persistence" else "cached_mlp"
    cfg, arrays, meta, info, records = state_fixture(tmp_path, arm)
    if mutation == "stale_packet":
        arrays["root__packet"][:, 4, 0] += 1
    elif mutation == "stale_cache":
        arrays["root__cached_angles"][:, 2] = records[0]["policy"]["packets"][0, :4]
    elif mutation == "predicted_cache":
        arrays["carried__cached_angles"][:, 2] = arrays["carried__packet"][:, 2, :4]
    elif mutation == "wrong_real_index":
        arrays["root__real_index"][:, 2] += 1
    elif mutation == "wrong_last_index":
        arrays["root__last_visible_index"][:, 3] = 3
    elif mutation == "target":
        arrays["root__real_target"][:, 2] += 1
    elif mutation == "imagined_root":
        arrays["root__imagined_depth"][:, 2] = 1
    elif mutation == "candidate_carry":
        arrays["carried__imagined_depth"][:, 4] = 2
    elif mutation == "encoder_features":
        arrays["root__encoder_features"][:, 2, :4] = 0
    elif mutation == "carried_features":
        arrays["carried__encoder_features"][:, 2] = arrays["root__encoder_features"][:, 2]
    elif mutation == "missing_persistence":
        arrays["root__hidden"][:, 2] += 1
    elif mutation == "dtype":
        arrays["root__real_index"] = arrays["root__real_index"].astype(np.float32)
    elif mutation == "state_hash":
        meta["steps"][0]["root_sha256"] = "0" * 64
    else:
        meta["aggregate_model_work"]["advance_samples"] += 1
    # Reseal all state hashes: this tests independent semantics, not just byte integrity.
    if mutation != "state_hash":
        for step, row in enumerate(meta["steps"]):
            for phase in ("root", "carried"):
                values = {key.removeprefix(phase+"__"): torch.from_numpy(value[:, step].copy())
                          for key, value in arrays.items() if key.startswith(phase+"__")}
                row[phase+"_sha256"] = training.canonical_tensor_hash(values)
    np.savez_compressed(tmp_path / "states.npz", **arrays)
    (tmp_path / "state-work.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        audit.audit_states(cfg, tmp_path, records, arm, info)

def costs():
    rows = {}
    means = {"residual_gru": 8., "encoded_current_gru": 10., "cached_gru": 9., "packet_mlp": 11., "cached_mlp": 9.5,
             "known_state": 5., "particle": 7., "zero": 12., "uniform": 15., "public_kinematic": 20.}
    for panel in protocol.PANELS:
        rows[panel] = {}
        for arm, mean in means.items():
            for name in ([f"{arm}-{pair}" for pair in protocol.PAIRS] if arm in protocol.ARMS else [arm]):
                rows[panel][name] = {"episode_costs": [mean-.25, mean+.25], "mean_cost": mean}
    return rows


def set_cost(controls, panel, name, mean):
    controls[panel][name] = {"episode_costs": [mean-.25, mean+.25], "mean_cost": mean}


def test_all_twenty_eight_checks_and_descriptive_kinematic_reference_retained():
    cfg, controls = plan(), costs()
    gate, differences = audit.control_qualification(cfg, controls)
    assert gate["passed"] is True
    assert len(gate["checks"]) == 28
    assert gate["public_kinematic"] == "descriptive_only"
    assert set(differences) == set(protocol.PANELS)
    assert all(len(values) == 2 for rows in differences.values() for values in rows.values())
    assert differences["shift"]["persistent_minus_public_kinematic"] == [-12., -12.]


@pytest.mark.parametrize("mutation", ["missing_fit", "extra_fit", "missing_panel", "missing_reference", "nonfinite", "wrong_mean", "short_cases"])
def test_partial_or_malformed_control_coverage_rejected(mutation):
    controls, cfg = costs(), plan()
    row = controls["shift"]["residual_gru-pair2"]
    if mutation == "missing_fit":
        controls["shift"].pop("residual_gru-pair2")
    elif mutation == "extra_fit":
        controls["full"]["winner"] = copy.deepcopy(row)
    elif mutation == "missing_panel":
        controls.pop("shift")
    elif mutation == "missing_reference":
        controls["ordinary"].pop("particle")
    elif mutation == "nonfinite":
        row["episode_costs"][0] = float("nan")
    elif mutation == "wrong_mean":
        row["mean_cost"] += 1
    else:
        row["episode_costs"].pop()
    with pytest.raises(ValueError):
        audit.control_qualification(cfg, controls)


@pytest.mark.parametrize("reference", ["known_state", "particle"])
@pytest.mark.parametrize("panel", ["ordinary", "shift"])
def test_failed_physics_reference_fails_whole_gate_despite_neural_improvement(reference, panel):
    controls = costs()
    set_cost(controls, panel, reference, 12.)
    gate, _ = audit.control_qualification(plan(), controls)
    assert gate["passed"] is False
    failed = [row for row in gate["checks"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["name"] == f"{panel}/{reference}: physics beats zero by 10%"


def test_one_losing_seed_cannot_be_hidden_by_successful_family_mean():
    controls = costs()
    for pair in protocol.PAIRS[:2]:
        set_cost(controls, "ordinary", f"residual_gru-{pair}", 1.)
    set_cost(controls, "ordinary", "residual_gru-pair2", 10.)
    gate, _ = audit.control_qualification(plan(), controls)
    checks = {row["name"]: row for row in gate["checks"]}
    assert checks["ordinary: persistent mean improves cached_gru by 3%"]["passed"] is True
    assert checks["ordinary/pair2: persistent nonworse than cached_gru"]["passed"] is False
    assert gate["passed"] is False


def test_losing_to_cached_mlp_cannot_be_hidden_by_gru_or_secondary_improvement():
    controls = costs()
    set_cost(controls, "shift", "cached_mlp-pair2", 7.)
    gate, differences = audit.control_qualification(plan(), controls)
    checks = {row["name"]: row for row in gate["checks"]}
    assert checks["shift: persistent mean improves cached_gru by 3%"]["passed"]
    assert checks["shift: persistent mean improves cached_mlp by 3%"]["passed"]
    assert not checks["shift/pair2: persistent nonworse than cached_mlp"]["passed"]
    assert all(value < 0 for value in differences["shift"]["mlp_cache_information"])
    assert not gate["passed"]
    assert "not proof of equivalence" in gate["fail_interpretation"]


def test_external_plan_hash_failure_preserves_terminal_receipt_without_audit(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    out = tmp_path / "audit"

    def forbidden(*args, **kwargs):
        raise AssertionError("No evidence/native audit after failed external authentication")

    monkeypatch.setattr(audit, "audit_saved", forbidden)
    with pytest.raises(ValueError):
        audit.audit_plan(path, "0" * 64, tmp_path / "execution", out)
    failed = json.loads((out / "failed.json").read_text())
    assert failed["stage"] == "plan_authentication"
    assert failed["status"] == "failed" and failed["saved_output_only"]
    assert failed["plan_sha256"] == "0" * 64
    assert not (out / "receipt.json").exists()
    with pytest.raises(ValueError, match="Exclusive"):
        audit.audit_plan(path, "0" * 64, tmp_path / "execution", out)


def test_paired_bootstrap_consumes_only_manifested_analysis_role(monkeypatch):
    called = []
    monkeypatch.setattr(audit.protocol, "seed", lambda cfg, role: called.append(role) or 410)
    difference = {"shift": {"persistent_minus_current_gru": [-2., 1.]}}
    result = audit.paired_comparisons(plan(), difference)["shift"]["persistent_minus_current_gru"]
    rng = np.random.default_rng(410)
    draws = rng.integers(0, 2, (64, 2))
    expected = np.quantile(np.asarray([-2., 1.])[draws].mean(1), [.025, .975])
    np.testing.assert_array_equal(result["episode_paired_percentile_95"], expected)
    assert called == ["analysis/bootstrap/0"]
    assert result["conditional_on_all_three_saved_fit_pairs"] is True
    assert result["cases"] == 2


@pytest.mark.parametrize("endpoint", ["cached_gru_three_percent", "cached_mlp_three_percent", "full_two_percent"])
def test_inclusive_mean_thresholds_and_immediately_worse_values(endpoint):
    controls = costs()
    if endpoint == "cached_gru_three_percent":
        panel, limit, name = "ordinary", .97 * 9., "ordinary: persistent mean improves cached_gru by 3%"
    elif endpoint == "cached_mlp_three_percent":
        panel, limit, name = "shift", .97 * 9.5, "shift: persistent mean improves cached_mlp by 3%"
    else:
        panel, limit, name = "full", 1.02 * 9., "full: persistent mean degrades cached_gru at most 2%"
    for pair in protocol.PAIRS:
        controls[panel][f"residual_gru-{pair}"] = {"episode_costs": [limit, limit], "mean_cost": limit}
    gate, _ = audit.control_qualification(plan(), controls)
    checks = {row["name"]: row for row in gate["checks"]}
    assert checks[name]["passed"] is True
    # Enough representable spacing that averaging three identical values cannot erase it.
    worse = float(limit + 1e-12)
    for pair in protocol.PAIRS:
        controls[panel][f"residual_gru-{pair}"] = {"episode_costs": [worse, worse], "mean_cost": worse}
    gate, _ = audit.control_qualification(plan(), controls)
    assert next(row for row in gate["checks"] if row["name"] == name)["passed"] is False

def test_auditor_source_does_not_import_or_construct_learned_components():
    import ast

    source = ast.parse(Path(audit.__file__).read_text())
    imported = []
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
    forbidden = ("cache_training", "cache_control", "world_models", "observation_baseline", "cached_observation")
    assert not any(word in name for name in imported for word in forbidden)
    constructors = set(protocol.MODEL_CLASSES.values())
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in constructors for node in ast.walk(source))


def test_public_learning_tensors_match_fit_allowlist_without_privileged_reads_or_aliases():
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Privileged fixture state must not enter learning tensors")

    records = []
    for index in range(2):
        records.append({
            "policy": {"packets": np.arange(24, dtype=np.float64).reshape(3, 8) / (index+3),
                       "commands": np.full((2, 2), index/10, dtype=np.float64)},
            "audit": {"rewards": np.asarray([-.125, -.3333333333-index], dtype=np.float64),
                      "qpos": Forbidden(), "qvel": Forbidden(), "raw_obs": Forbidden(),
                      "actuator_noise": Forbidden(), "integration_state": Forbidden()},
            "metadata": Forbidden(),
        })
    # This reproduces the independently read fit runner's three allowlisted
    # tensor constructors, without importing a runner into the auditor.
    expected = {
        name: torch.tensor(np.stack([row[group][name] for row in records]), dtype=torch.float32)
        for group, name in (("policy", "packets"), ("policy", "commands"), ("audit", "rewards"))
    }
    actual = audit.public_learning_tensors(records)
    assert set(actual) == {"packets", "commands", "rewards"}
    for name, value in expected.items():
        assert actual[name].dtype == torch.float32 and actual[name].device.type == "cpu"
        assert torch.equal(actual[name], value)
    assert audit.canonical_tensor_hash(actual) == training.canonical_tensor_hash(expected)
    records[0]["policy"]["packets"][:] = -100
    records[0]["audit"]["rewards"][:] = -100
    assert torch.equal(actual["packets"], expected["packets"])
    assert torch.equal(actual["rewards"], expected["rewards"])
    actual["commands"].fill_(99)
    assert np.all(records[0]["policy"]["commands"] == 0)
