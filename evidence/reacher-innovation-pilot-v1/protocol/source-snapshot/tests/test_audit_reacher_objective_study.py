"""Synthetic numerical corruption tests; no learned scoring or real study reads."""

import copy
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_reacher_objective_study as audit


def records(batch=2, steps=8):
    rng = np.random.default_rng(716)
    output = []
    for case in range(batch):
        packets = rng.normal(size=(steps + 1, 8)).astype(np.float32)
        valid = np.ones(steps + 1, dtype=bool)
        valid[[2, 3, 6]] = False
        packets[:, 6] = valid
        packets[:, 7] = 0
        native = packets.copy()
        packets[~valid, :4] = 0
        output.append({"policy": {"packets": packets},
                       "audit": {"raw_obs": native, "rewards": np.arange(steps, dtype=float)},
                       "metadata": {"sensor_schedule": valid.tolist()}})
    return output


def saved_predictions(cohort):
    packets = np.stack([record["policy"]["packets"] for record in cohort])
    steps = packets.shape[1] - 1
    values = {"one_step_angles": packets[:, 1:, :4].copy(),
              "one_step_rewards": np.stack([record["audit"]["rewards"] for record in cohort]).astype(np.float32)}
    for h in (1, 3, 7):
        values[f"h{h}_angles"] = packets[:, h:, :4].copy()
        values[f"h{h}_valid"] = (packets[:, :steps + 1 - h, 6] > .5) & (packets[:, h:, 6] > .5)
    return values


def controls():
    result = {}
    for panel in ("full", "ordinary", "shift"):
        rows = {}
        for pair in ("pair0", "pair1", "pair2"):
            for arm, mean in (("anchor", 10.), ("raw", 9.), ("latent", 8.)):
                rows[f"{arm}-{pair}"] = {"episode_costs": [mean - .2, mean + .2], "mean_cost": mean}
                if panel != "full":
                    rows[f"{arm}-{pair}-reset"] = {"episode_costs": [mean + .8, mean + 1.2],
                                                    "mean_cost": mean + 1}
        for name, mean in (("known_state", 7.), ("particle", 8.), ("zero", 12.), ("uniform", 40.)):
            rows[name] = {"episode_costs": [mean, mean], "mean_cost": mean}
        result[panel] = rows
    return result


def set_cost(controls, panel, name, value):
    controls[panel][name] = {"episode_costs": [value, value], "mean_cost": value}


def test_gate_recomputes_all_31_checks_without_summary_booleans():
    rows = controls()
    gate, paired = audit.control_qualification({"control_episodes": 2}, rows)
    assert gate["passed"] and len(gate["checks"]) == 31
    assert all(check["passed"] for check in gate["checks"])
    np.testing.assert_allclose(paired["ordinary"]["latent_minus_raw"], [-1, -1])
    np.testing.assert_allclose(paired["shift"]["latent_reset_minus_intact"], [1, 1])


@pytest.mark.parametrize("panel", ["ordinary", "shift"])
def test_family_improvement_cannot_hide_one_worse_fit(panel):
    rows = controls()
    set_cost(rows, panel, "latent-pair0", 9.1)
    set_cost(rows, panel, "latent-pair1", 7)
    set_cost(rows, panel, "latent-pair2", 7)
    gate, _ = audit.control_qualification({"control_episodes": 2}, rows)
    found = {row["name"]: row["passed"] for row in gate["checks"]}
    assert found[f"latent/{panel}/mean_vs_raw"]
    assert not found[f"latent-pair0/{panel}/vs_raw"] and not gate["passed"]


def test_reset_effect_requires_every_pair_even_when_family_mean_passes():
    rows = controls()
    set_cost(rows, "ordinary", "latent-pair0-reset", 8)
    gate, _ = audit.control_qualification({"control_episodes": 2}, rows)
    found = {row["name"]: row["passed"] for row in gate["checks"]}
    assert found["latent/ordinary/reset_effect"]
    assert not found["latent-pair0/ordinary/reset_effect"]


@pytest.mark.parametrize("corruption", ["omitted_fit", "missing_reset", "mean", "nan", "case_count"])
def test_gate_rejects_incomplete_or_fabricated_control_arrays(corruption):
    rows = controls()
    if corruption == "omitted_fit":
        del rows["full"]["anchor-pair2"]
    elif corruption == "missing_reset":
        del rows["ordinary"]["raw-pair1-reset"]
    elif corruption == "mean":
        rows["ordinary"]["latent-pair0"]["mean_cost"] = 1
    elif corruption == "nan":
        rows["ordinary"]["latent-pair0"]["episode_costs"][0] = float("nan")
    else:
        rows["ordinary"]["latent-pair0"]["episode_costs"].append(8)
    with pytest.raises(ValueError):
        audit.control_qualification({"control_episodes": 2}, rows)


def test_prediction_metrics_derive_targets_masks_and_component_counts():
    cohort = records()
    values = saved_predictions(cohort)
    result = audit.prediction_metrics({"steps": 8}, cohort, values)
    assert result["one_step_reward_mse"] == 0
    assert result["one_step_angles"]["mse"] == 0
    for h in (1, 3, 7):
        wanted = int(values[f"h{h}_valid"].sum())
        assert result["endpoints"][str(h)] == {"mse": 0., "targets": wanted, "components": wanted * 4}
    values["one_step_rewards"] += 2
    assert audit.prediction_metrics({"steps": 8}, cohort, values)["one_step_reward_mse"] == 4


@pytest.mark.parametrize("corruption", ["mask", "h1_disagreement", "nan", "shape", "extra_target"])
def test_prediction_metrics_reject_saved_masks_or_inconsistent_predictions(corruption):
    cohort = records()
    values = saved_predictions(cohort)
    if corruption == "mask":
        values["h3_valid"][0, 0] = ~values["h3_valid"][0, 0]
    elif corruption == "h1_disagreement":
        values["h1_angles"][0, 0, 0] += 1
    elif corruption == "nan":
        values["h7_angles"][0, 0, 0] = np.nan
    elif corruption == "shape":
        values["h7_angles"] = values["h7_angles"][:, :-1]
    else:
        values["privileged_target"] = np.zeros(1)
    with pytest.raises(ValueError):
        audit.prediction_metrics({"steps": 8}, cohort, values)


def test_reset_events_preserve_last_visible_public_packet_exactly(tmp_path):
    cohort = records()
    public = np.stack([record["policy"]["packets"] for record in cohort])[:, :-1]
    mask = audit.reset_mask(np.asarray([r["metadata"]["sensor_schedule"] for r in cohort]), 8)
    path = tmp_path / "reset-events.npz"
    np.savez(path, mask=mask, packets=np.where(mask[..., None], public, 0))
    result = audit.audit_reset_events(path, cohort, enabled=True)
    assert result["events"] == 4 and result["steps_per_case"] == [[1, 5], [1, 5]]


@pytest.mark.parametrize("corruption", ["first_missing_step", "erased_measurement", "off_mask", "missing_event"])
def test_reset_audit_rejects_wrong_time_or_lost_measurements(tmp_path, corruption):
    cohort = records()
    public = np.stack([record["policy"]["packets"] for record in cohort])[:, :-1]
    mask = audit.reset_mask(np.asarray([r["metadata"]["sensor_schedule"] for r in cohort]), 8)
    events = np.where(mask[..., None], public, 0)
    if corruption == "first_missing_step":
        mask = np.roll(mask, 1, axis=1)
    elif corruption == "erased_measurement":
        events[0, 1, :4] = 0
    elif corruption == "off_mask":
        events[0, 0, 0] = 1
    else:
        mask[0, 1] = False
    path = tmp_path / "reset-events.npz"
    np.savez(path, mask=mask, packets=events)
    with pytest.raises(ValueError):
        audit.audit_reset_events(path, cohort, enabled=True)


def ema_witness():
    before = {"weight": torch.tensor([1., 3.]), "bias": torch.tensor([2.])}
    student = {key: value + 4 for key, value in before.items()}
    after = {key: value.clone().mul_(.99).add_(student[key], alpha=.01) for key, value in before.items()}
    return {"teacher_before": before, "student_after_optimizer": student, "teacher_after_ema": after,
            "buffers": {"student": {"count": torch.tensor(7)}, "teacher": {"count": torch.tensor(7)}}}


def test_ema_witness_recomputes_parameter_arithmetic_and_exact_buffer_copy():
    witness = ema_witness()
    result = audit.audit_ema_witness(witness, .99, {"weight": (2,), "bias": (1,)})
    assert result == {"parameter_tensors": 2, "buffer_tensors": 1}


@pytest.mark.parametrize("corruption", ["student_as_teacher", "wrong_momentum", "averaged_buffer", "nan"])
def test_ema_witness_cannot_be_replaced_by_claimed_success(corruption):
    witness = copy.deepcopy(ema_witness())
    if corruption == "student_as_teacher":
        witness["teacher_after_ema"] = witness["student_after_optimizer"]
    elif corruption == "wrong_momentum":
        witness["teacher_after_ema"]["weight"] += .01
    elif corruption == "averaged_buffer":
        witness["buffers"]["teacher"]["count"] = torch.tensor(6)
    else:
        witness["teacher_before"]["weight"][0] = float("nan")
    with pytest.raises(ValueError):
        audit.audit_ema_witness(witness, .99, {"weight": (2,), "bias": (1,)})


def test_complete_engineering_tree_saved_only_audit_and_deep_corruptions(tmp_path, monkeypatch):
    """Exercise the actual nine tiny fits/57 rows, then prohibit neural execution.

    Four native-valid records repeated to 128 are engineering training inputs.
    Production seeds, inherited weights and historical efficacy are not used.
    Only synthetic inheritance identity hooks differ from the production audit.
    """
    import reacher_objective_fixture as fixture

    # Optional exclusive destination preserves a public engineering preflight.
    destination = Path(os.environ.get("OPENJEV_OBJECTIVE_ENGINEERING_OUTPUT", tmp_path / "whole-tree"))
    case = fixture.prepare_fixture(destination)
    fixture.run_fixture(case)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-output audit attempted neural execution")

    with monkeypatch.context() as no_neural:
        no_neural.setattr(torch.nn.Module, "__call__", forbidden)
        no_neural.setattr(torch.optim.Adam, "step", forbidden)
        summary = fixture.audit_fixture(case)
    assert summary["status"] == "completed" and summary["engineering"]
    assert summary["new_model_calls"] == summary["new_policy_calls"] == summary["new_fits"] == 0
    assert summary["coverage"] == {"fits": 9, "control_rows": 57,
                                   "control_episodes_per_row": 1, "prediction_episodes": 2}
    assert summary["native_transitions_checked"] == (128 + 2 + 57) * 50
    assert summary["native_max_abs_error"] == 0
    assert len(summary["continuation_gate"]["checks"]) == 31
    assert all(row["updates"] == 4 for row in summary["fits"].values())

    stream_path = case.execution / "random-streams.json"
    stream_bytes = stream_path.read_bytes()
    try:
        changed = audit.base.read(stream_path)
        changed["torch_registry"]["fit/student/pair0"] = 1121
        audit.base.write(stream_path, changed)
        with pytest.raises(ValueError, match="Named stream manifest"):
            audit.validate_streams(case.plan, case.baseline, case.execution)
    finally:
        stream_path.write_bytes(stream_bytes)

    initializations = audit.audit_initializations(case.plan, case.execution)
    orders = audit.audit_orders(case.plan, case.execution)
    calibration = audit.audit_calibration(case.plan, case.execution, initializations)
    checkpoint = audit.read_tensor_file(case.execution / "fits/latent-pair0/checkpoint.pt")
    for corruption in ("optimizer_step", "optimizer_moment", "teacher_in_optimizer", "ema_count",
                       "generator_state", "initial_pair", "multiplier"):
        changed = copy.deepcopy(checkpoint)
        if corruption == "optimizer_step":
            next(iter(changed["optimizer_state"]["state"].values()))["step"] -= 1
        elif corruption == "optimizer_moment":
            next(iter(changed["optimizer_state"]["state"].values()))["exp_avg_sq"].fill_(-1)
        elif corruption == "teacher_in_optimizer":
            changed["optimizer_group_names"][0][0] = "teacher.observation_update.weight_ih"
        elif corruption == "ema_count":
            changed["ema_updates"] -= 1
        elif corruption == "generator_state":
            changed["minibatch_rng_state"] = torch.Generator().manual_seed(72).get_state()
        elif corruption == "initial_pair":
            changed["initialization"] = initializations["pair1"]
        else:
            changed["loss_multiplier"] *= 2
        changed["integrity_sha256"] = audit.canonical_state_hash(
            {key: value for key, value in changed.items() if key != "integrity_sha256"})
        with pytest.raises(ValueError):
            audit.audit_checkpoint(case.plan, changed, initializations["pair0"], "latent", "pair0",
                                   calibration["multipliers"]["latent"], orders["pair0"])

    calibration_copy = tmp_path / "calibration-corruptions"
    calibration_copy.mkdir()
    shutil.copyfile(case.execution / "calibration-gradients.pt", calibration_copy / "calibration-gradients.pt")
    original = audit.base.read(case.execution / "calibration.json")
    for corruption in ("first_training_batch", "norm", "lambda", "decoder_cost", "teacher_cost"):
        changed = copy.deepcopy(original)
        if corruption == "first_training_batch":
            changed["rows"][0]["indices"][0] = 127
        elif corruption == "norm":
            changed["rows"][0]["losses"]["latent"]["backbone_norm"] *= 2
        elif corruption == "lambda":
            changed["multipliers"]["latent"]["value"] *= 2
        elif corruption == "decoder_cost":
            changed["rows"][0]["work"]["raw"]["batch_forward_calls"]["student_endpoint_decoder"] = 0
        else:
            changed["rows"][0]["work"]["latent"]["batch_forward_calls"]["teacher_assimilate"] = 0
        audit.base.write(calibration_copy / "calibration.json", changed)
        with pytest.raises(ValueError):
            audit.audit_calibration(case.plan, calibration_copy, initializations)

    failed = tmp_path / "refused-production"
    with pytest.raises(ValueError):
        audit.audit_saved(case.plan, case.digest, case.execution, failed, engineering=False)
    assert audit.base.read(failed / "failed.json")["status"] == "failed"
    assert not (failed / "receipt.json").exists()

    def expired():
        raise TimeoutError("Synthetic audit deadline")

    cap_failure = tmp_path / "audit-cap-failure"
    with monkeypatch.context() as deadline:
        deadline.setattr(audit, "check_budget", expired)
        with fixture.fixture_audit_boundary(case), pytest.raises(TimeoutError, match="deadline"):
            audit.audit_saved(case.plan, case.digest, case.execution, cap_failure, engineering=True)
    assert audit.base.read(cap_failure / "failed.json")["status"] == "failed"
    assert not (cap_failure / "receipt.json").exists()


def test_plan_authentication_failure_is_preserved_before_execution_reads(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    out = tmp_path / "audit"
    with pytest.raises(ValueError, match="identity"):
        audit.audit_plan(path, "0" * 64, tmp_path / "does-not-exist", out)
    assert audit.base.read(out / "failed.json")["stage"] == "plan_authentication"
    with pytest.raises(ValueError, match="Exclusive"):
        audit.audit_plan(path, audit.base.sha(path), tmp_path / "does-not-exist", out)
