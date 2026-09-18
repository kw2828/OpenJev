"""Synthetic saved-score evidence, with no learned scoring or scored study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_reacher_search_study as audit

base = audit.base


def plan_fixture():
    return {
        "steps": 50,
        "planning_horizon": 12,
        "action_block": 3,
        "planners": ["rs64", "rs256", "cem256"],
        "kinds": ["free", "residual"],
        "fit_seeds": [271, 283, 293],
        "control_episodes": 2,
        "bootstrap_seed": 901,
        "bootstrap_samples": 32,
        "random_stream_contract": {"registry": {"analysis/bootstrap/0": 80019}},
    }


def innovations(n=2):
    rng = np.random.default_rng(13579)
    return audit.SearchInputs(
        rng.normal(size=(n, 64, 4, 2)),
        rng.normal(size=(n, 192, 4, 2)),
        tuple(rng.normal(size=(n, size, 4, 2)) for size in (64, 64, 63)),
    )


def save_trace(stem, inputs, method="cem256", step=0):
    raw_parts = []

    def scorer(bank):
        raw = (np.float32(2.6) * bank[..., 0] - np.float32(1.0)).astype(np.float32)
        raw_parts.append(raw.copy())
        score = np.zeros(raw.shape[:2], dtype=np.float32)
        for t in range(raw.shape[2]):
            score = score + np.maximum(np.minimum(raw[:, :, t], 0), -2.5)
        return score

    result = audit.search(method, inputs, scorer, step=step)
    data = {
        "chunks": result.sequences[:, :, ::3].copy(),
        "scores": result.scores.copy(),
        "raw_rewards": np.concatenate(raw_parts, axis=1),
        "selected_ids": result.selected_ids.copy(),
    }
    stages = []
    for index, stage in enumerate(result.stages):
        fields = {}
        for field in audit.STAGE_ARRAYS:
            value = getattr(stage, field)
            if value is not None:
                key = f"stage_{index}_{field}"
                fields[field] = key
                data[key] = value.copy()
        stages.append(
            {field: getattr(stage, field) for field in audit.STAGE_KEYS - {"array_fields"}}
            | {"array_fields": fields}
        )
    meta = {
        key: getattr(result, key)
        for key in (
            "method",
            "horizon",
            "action_block",
            "candidate_evaluations_per_case",
            "candidate_evaluations",
            "imagined_transitions_per_case",
            "imagined_transitions",
        )
    }
    meta.update(
        candidate_ids=list(result.candidate_ids),
        input_identities=dict(result.input_identities),
        callback_sizes=[part.shape[1] for part in raw_parts],
        search_seconds=0.001,
        stages=stages,
    )
    np.savez_compressed(stem.with_suffix(".npz"), **data)
    base.write(stem.with_suffix(".json"), meta)
    return result


@pytest.mark.parametrize("method", ["rs64", "rs256", "cem256"])
@pytest.mark.parametrize("step", [0, 39, 47, 49])
def test_reconstructs_every_candidate_and_paid_budget_without_inference(tmp_path, method, step):
    draws, stem, plan = innovations(), tmp_path / "trace", plan_fixture()
    result = save_trace(stem, draws, method, step)
    checked = audit.audit_trace(plan, draws, stem, step=step)
    np.testing.assert_array_equal(checked["result"].sequences, result.sequences)
    assert checked["result"].imagined_transitions == 2 * (64 if method == "rs64" else 256) * min(
        12, 50 - step
    )
    assert checked["clipped_predictions"] > 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("extra", "array membership"),
        ("chunks", "proposals differ"),
        ("raw", "reward arithmetic"),
        ("selected", "Global best"),
        ("mean", "reconstruction"),
        ("std", "reconstruction"),
        ("elites", "reconstruction"),
        ("budget", "Search budget"),
        ("calls", "call schedule"),
        ("unpaid_mean", "stage field"),
        ("hash", "innovation identities"),
        ("skip_stage", "stage coverage"),
        ("negative_time", "timing"),
        ("nonfinite", "Nonfinite"),
    ],
)
def test_trace_corruption_is_not_a_planning_result(tmp_path, mutation, match):
    draws, stem = innovations(), tmp_path / "trace"
    save_trace(stem, draws)
    data, meta = base.load_npz(stem.with_suffix(".npz")), base.read(stem.with_suffix(".json"))
    if mutation == "extra":
        data["unlogged_score"] = np.zeros(1)
    if mutation == "chunks":
        data["chunks"][0, 64, 0, 0] *= -1
    if mutation == "raw":
        data["raw_rewards"][0, 1, 0] = -100
    if mutation == "selected":
        data["selected_ids"][0] = (data["selected_ids"][0] + 1) % 256
    if mutation in ("mean", "std", "elites"):
        key = {
            "mean": "stage_1_proposal_mean",
            "std": "stage_1_proposal_std",
            "elites": "stage_1_source_elite_ids",
        }[mutation]
        data[key].flat[0] += 1
    if mutation == "budget":
        meta["imagined_transitions"] += 1
    if mutation == "calls":
        meta["callback_sizes"] = [256]
    if mutation == "unpaid_mean":
        meta["stages"][-1]["mean_candidate_id"] = None
    if mutation == "hash":
        meta["input_identities"]["cem/3"] = "f" * 64
    if mutation == "skip_stage":
        meta["stages"].pop()
    if mutation == "negative_time":
        meta["search_seconds"] = -1
    if mutation == "nonfinite":
        data["raw_rewards"][0, 0, 0] = np.nan
    np.savez_compressed(stem.with_suffix(".npz"), **data)
    base.write(stem.with_suffix(".json"), meta)
    with pytest.raises(ValueError, match=match):
        audit.audit_trace(plan_fixture(), draws, stem, step=0)


def test_average_ties_and_undefined_rank_are_not_fabricated_zero():
    np.testing.assert_array_equal(audit.average_ranks([2, 2, 4, 1]), [1.5, 1.5, 3, 0])
    assert audit.rank_agreement([1, 2, 3], [6, 5, 4]) == pytest.approx(-1)
    assert audit.rank_agreement([1, 1, 1], [1, 2, 3]) is None


def test_native_union_regret_and_bias_use_native_outcomes_not_search_scores(tmp_path):
    draws, stem = innovations(n=1), tmp_path / "trace"
    save_trace(stem, draws)
    checked = audit.audit_trace(plan_fixture(), draws, stem, step=0)
    outcomes = np.full((66, 4), -9.0)
    outcomes[64] = [-8, -7, -6, -7]
    outcomes[65] = [-4, -5, -4, -3]
    result = audit.diagnostic_metrics(checked, outcomes, 64)
    assert result["native_selected_return"] == -7
    assert result["finite_set_regret"] == 3
    assert result["common_bank_spearman"] is None
    assert result["selected_raw_prediction_bias"] == pytest.approx(checked["selected_raw_return"][0] + 7)


def passing_controls(plan):
    result = {}
    for panel in base.PANELS:
        result[panel] = {}
        for arm in base.REFERENCES:
            cost = 150.0 if arm == "zero" else 100.0
            result[panel][arm] = {"mean_cost": cost, "episode_costs": [cost] * plan["control_episodes"]}
        for kind in plan["kinds"]:
            for seed in plan["fit_seeds"]:
                for method in plan["planners"]:
                    cost = (
                        120.0 if kind == "free" else {"rs64": 110.0, "rs256": 100.0, "cem256": 95.0}[method]
                    )
                    result[panel][f"{kind}-{seed}-{method}"] = {
                        "mean_cost": cost,
                        "episode_costs": [cost] * plan["control_episodes"],
                    }
    return result


def test_primary_threshold_and_control_competence_are_separate():
    plan = plan_fixture()
    controls = passing_controls(plan)
    gate, competence, _ = audit.qualify_search(plan, controls)
    assert gate["passed"] and len(gate["checks"]) == 8
    assert all(item["passed"] and len(item["checks"]) == 15 for item in competence.values())
    for panel in base.PANELS:
        controls[panel]["zero"] = {"mean_cost": 1.0, "episode_costs": [1.0, 1.0]}
    gate, competence, _ = audit.qualify_search(plan, controls)
    assert gate["passed"] and not any(item["passed"] for item in competence.values())


def test_bootstrap_consumes_manifested_role_not_inherited_seed(monkeypatch):
    plan = plan_fixture()
    used = []
    original = np.random.default_rng

    def tracked(seed):
        used.append(seed)
        return original(seed)

    monkeypatch.setattr(np.random, "default_rng", tracked)
    audit.qualify_search(plan, passing_controls(plan))
    assert used == [80019] * 6
    assert plan["bootstrap_seed"] == 901


@pytest.mark.parametrize("mutation", ["one_seed_tie", "shift_mean", "missing", "nonfinite"])
def test_all_seeds_both_panels_and_complete_finite_outcomes_required(mutation):
    plan = plan_fixture()
    controls = passing_controls(plan)
    if mutation == "one_seed_tie":
        controls["shift"]["residual-293-cem256"] = {"mean_cost": 100.0, "episode_costs": [100.0, 100.0]}
    if mutation == "shift_mean":
        for seed in plan["fit_seeds"]:
            controls["shift"][f"residual-{seed}-cem256"] = {"mean_cost": 96.0, "episode_costs": [96.0, 96.0]}
    if mutation == "missing":
        controls["full"].pop("free-271-rs64")
    if mutation == "nonfinite":
        controls["ordinary"]["residual-271-rs64"]["mean_cost"] = float("nan")
    if mutation in ("missing", "nonfinite"):
        with pytest.raises(ValueError):
            audit.qualify_search(plan, controls)
    else:
        assert not audit.qualify_search(plan, controls)[0]["passed"]


@pytest.fixture
def native_branches():
    """Small saved-native fixture, no learned model or controller."""
    native = audit.native
    record = native.collect_episode(983_413, np.ones(51, dtype=bool), 0.05, 983_417, 983_419, policy="mixed")
    step, horizon = 47, 3
    commands = np.array([[[0.0, 0.0]] * horizon, [[1.0, -1.0]] * horizon], dtype=np.float32)
    noise = np.random.default_rng(983_423).normal(0.0, 0.05, (2, horizon, 2))
    rows = []
    env = native.make_env()
    try:
        env.reset(seed=983_413)
        for sequence in commands:
            for branch in noise:
                native.restore_native(
                    env, {"step": step, "integration_state": record["audit"]["integration_state"][step]}
                )
                values = {
                    key: []
                    for key in (
                        "qpos",
                        "qvel",
                        "raw_obs",
                        "integration_state",
                        "time",
                        "commands",
                        "applied_actions",
                        "actuator_noise",
                        "rewards",
                        "reward_dist",
                        "reward_ctrl",
                        "terminated",
                        "truncated",
                    )
                }
                raw = env.unwrapped._get_obs()
                for t in range(horizon + 1):
                    for key, value in {
                        "qpos": env.unwrapped.data.qpos,
                        "qvel": env.unwrapped.data.qvel,
                        "raw_obs": raw,
                        "integration_state": native._integration_state(env),
                        "time": env.unwrapped.data.time,
                    }.items():
                        values[key].append(np.array(value, copy=True))
                    if t == horizon:
                        break
                    applied = np.clip(sequence[t].astype(float) + branch[t], -1, 1)
                    raw, reward, terminated, truncated, info = env.step(applied)
                    for key, value in {
                        "commands": sequence[t],
                        "applied_actions": applied,
                        "actuator_noise": branch[t],
                        "rewards": reward,
                        "reward_dist": info["reward_dist"],
                        "reward_ctrl": info["reward_ctrl"],
                        "terminated": terminated,
                        "truncated": truncated,
                    }.items():
                        values[key].append(np.array(value, copy=True))
                rows.append({key: np.asarray(value) for key, value in values.items()})
    finally:
        env.close()
    saved = {key: np.stack([row[key] for row in rows]).reshape(2, 2, *rows[0][key].shape) for key in rows[0]}
    return record, step, commands, noise, saved


def test_native_replay_checks_all_branches_and_absolute_terminal_boundary(native_branches):
    record, step, commands, noise, saved = native_branches
    assert np.all(saved["truncated"][:, :, -1])
    assert not np.any(saved["truncated"][:, :, :-1])
    result = audit.replay_branches(record, step, commands, noise, saved)
    assert result == {
        "transitions": 12,
        "max_abs_error": 0.0,
        "saved_output_only": True,
        "new_policy_calls": 0,
    }


@pytest.mark.parametrize(
    "mutation",
    ["root", "reward", "qvel", "noise", "command", "applied", "terminal", "missing", "extra", "nan", "dtype"],
)
def test_native_replay_refuses_corrupt_or_partial_branch_evidence(native_branches, mutation):
    record, step, commands, noise, saved = native_branches
    if mutation == "root":
        saved["integration_state"][0, 0, 0, 0] += 0.1
    elif mutation in ("reward", "qvel"):
        saved["rewards" if mutation == "reward" else "qvel"][0, 0, 1] += 0.1
    elif mutation == "noise":
        saved["actuator_noise"][0, 0, 0, 0] += 0.1
    elif mutation == "command":
        saved["commands"][0, 0, 0, 0] += 0.1
    elif mutation == "applied":
        saved["applied_actions"][0, 0, 0, 0] += 0.1
    elif mutation == "terminal":
        saved["truncated"][0, 0, 0] = True
    elif mutation == "missing":
        saved["qpos"] = saved["qpos"][:1]
    elif mutation == "extra":
        saved["secret_state"] = np.zeros(1)
    elif mutation == "nan":
        saved["qpos"][0, 0, 0, 0] = np.nan
    else:
        saved["rewards"] = saved["rewards"].astype(np.float32)
    with pytest.raises(ValueError):
        audit.replay_branches(record, step, commands, noise, saved)


def structural_plan():
    plan = plan_fixture()
    plan.update(
        fit_order=[f"{kind}-{seed}" for kind in plan["kinds"] for seed in plan["fit_seeds"]],
        panels=list(audit.protocol.PANELS),
        references=list(audit.protocol.REFERENCES),
        diagnostic_episodes=1,
        diagnostic_branches=2,
        steps=2,
        cap_seconds=100.0,
        study=audit.VERSION,
    )
    return plan


@pytest.fixture
def completed_tree(tmp_path):
    plan = structural_plan()
    folder = tmp_path / "complete"
    folder.mkdir()
    members = audit.expected_members(plan)
    for name in members:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic membership fixture\n")
    value = {
        "status": "completed",
        "version": audit.VERSION,
        "plan_sha256": "a" * 64,
        "files": {name: audit.protocol.sha(folder / name) for name in members},
        "fits": len(plan["fit_order"]),
        "new_fits": 0,
        "astra_calls": 0,
        "wall_seconds": 5.0,
        "evaluation_started_elapsed_seconds": 1.0,
        "control_rows": len(audit.control_order(plan)),
        "diagnostic_roots": 4,
        "inherited_fit_wall_seconds": 2.0,
        "prior_cumulative_attempt_wall_seconds": 10.0,
        "cumulative_attempt_wall_seconds": 15.0,
    }
    base.write(folder / "completed.json", value)
    return plan, folder, value


def test_exact_members_cover_all_rows_roots_and_no_new_predictions(completed_tree):
    plan, folder, value = completed_tree
    assert audit.validate_members(plan, "a" * 64, folder) == value
    assert not any(name.startswith("predictions/") for name in value["files"])
    assert "diagnostic/roots/000-03/search/shift/residual-293/cem256.npz" in value["files"]
    assert "control/shift/residual-293-cem256/decisions/001.json" in value["files"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missingroot",
        "missingtrace",
        "failed",
        "partial",
        "extraweight",
        "rehashmissing",
        "changed",
        "symlink",
        "overcap",
        "nan",
        "count",
        "plan",
    ],
)
def test_complete_membership_rejects_gaps_partial_terminal_and_cap(completed_tree, mutation):
    plan, folder, value = completed_tree
    key = "diagnostic/roots/000-03/native.npz"
    if mutation == "missingroot":
        (folder / key).unlink()
    elif mutation == "missingtrace":
        (folder / "control/shift/residual-293-cem256/decisions/001.json").unlink()
    elif mutation in ("failed", "partial", "extraweight"):
        (
            folder
            / {"failed": "failed.json", "partial": "native-partial.npz", "extraweight": "extra-weights.pt"}[
                mutation
            ]
        ).write_text("extra")
    elif mutation == "rehashmissing":
        (folder / key).unlink()
        del value["files"][key]
    elif mutation == "changed":
        (folder / key).write_text("changed")
    elif mutation == "symlink":
        (folder / key).unlink()
        (folder / key).symlink_to(folder / "train.npz")
    elif mutation == "overcap":
        value["wall_seconds"] = plan["cap_seconds"] + 0.01
    elif mutation == "nan":
        value["wall_seconds"] = "nan"
    elif mutation == "count":
        value["fits"] -= 1
    else:
        value["plan_sha256"] = "b" * 64
    base.write(folder / "completed.json", value)
    with pytest.raises(ValueError):
        audit.validate_members(plan, "a" * 64, folder)


def test_timing_requires_every_decision_and_charges_setup_native_and_search(tmp_path):
    plan = plan_fixture()
    row = {
        "setup_seconds": 1.0,
        "decision_seconds": [0.1] * 50,
        "native_step_seconds": [0.01] * 50,
        "row_wall_seconds": 7.0,
        "observation_assimilations": 100,
        "executed_action_advances": 100,
    }
    path = tmp_path / "timings.json"
    base.write(path, row)
    assert audit.timing_row(plan, path, True) == row
    for key, value in (
        ("decision_seconds", [0.1] * 49),
        ("row_wall_seconds", 6.0),
        ("setup_seconds", "nan"),
        ("executed_action_advances", 99),
    ):
        base.write(path, row | {key: value})
        with pytest.raises(ValueError):
            audit.timing_row(plan, path, True)


def test_complete_synthetic_tree_through_audit_saved(tmp_path, monkeypatch):
    """Full saved-output path; only historical lineage/fitting is fixture-injected.

    Every fresh stream, member, phase, cost, score trace, executed action,
    diagnostic history/union/native branch and bootstrap remains audited.
    """
    import test_reacher_search_study as engineering
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture = engineering.build_complete_synthetic_execution(tmp_path, monkeypatch)
        monkeypatch.setattr(audit, "ROOT", fixture.root)
        historical = {"continuation_gate": {"passed": False, "checks": [], "synthetic_fixture": True}}
        monkeypatch.setattr(audit, "validate_parent", lambda *_: (fixture.parent, historical))
        fitmeta = {"wall_seconds": 3.0, "epochs": [], "minibatch_order_sha256": "synthetic"}
        inheritance = {
            "fits": {
                name: {key: value for key, value in fitmeta.items() if key != "epochs"}
                for name in fixture.plan["fit_order"]
            }
        }
        monkeypatch.setattr(audit.inherited, "validate_inheritance", lambda *_: (fixture.parent, inheritance))
        monkeypatch.setattr(audit.inherited.prior, "validate_training_source", lambda *_: fixture.parent)

        def fit(_, execution, name, train):
            assert len(train) == 1
            return dict(fitmeta), torch.load(
                execution / "fits" / name / "initial-weights.pt", weights_only=True
            )

        monkeypatch.setattr(audit.inherited.prior, "audit_fit", fit)

        def inherited_training(_, records, split):
            assert split == "train"
            values = [audit.native.native_replay(record) for record in records]
            return {
                "episodes": len(records),
                "transitions": sum(row["transitions"] for row in values),
                "max_abs_error": max(row["max_abs_error"] for row in values),
            }

        monkeypatch.setattr(audit.base, "audit_cohort", inherited_training)

        # Inference is prohibited during the entire saved-output audit.
        def forbidden(*args, **kwargs):
            raise AssertionError("The saved-output auditor called a learned model or planner")

        monkeypatch.setattr(engineering.GRUWorldModel, "advance", forbidden)
        monkeypatch.setattr(engineering.GRUWorldModel, "assimilate", forbidden)
        monkeypatch.setattr(engineering.study.PhysicsMPC, "plan", forbidden)
        result = audit.audit_saved(fixture.plan, fixture.digest, fixture.execution, tmp_path / "audit")
        assert result["native_max_abs_error"] == 0
        assert result["native_transitions_checked"] == 4798
        assert result["native_transition_counts"] == {
            "control": 1500,
            "diagnostic_cohort": 50,
            "diagnostic_branches": 3198,
            "inherited_training": 50,
            "fresh_evaluation": 4748,
        }
        assert result["learned_control_imagined_transitions"] == 1_845_504
        assert len(result["control"]) == 3
        assert all(len(panel) == 10 for panel in result["control"].values())
        assert len(result["diagnostics"]["roots"]) == 4
        assert all(root["identity_slots"] == 82 for root in result["diagnostics"]["roots"])
        assert result["historical_parent_qualification"]["rerun"] is False
        assert result["new_model_calls"] == result["new_mpc_calls"] == 0
        receipt = base.read(tmp_path / "audit" / "receipt.json")
        assert set(receipt["execution_members"]) == audit.expected_members(fixture.plan)
        assert receipt["files"]["summary.json"] == audit.protocol.sha(tmp_path / "audit" / "summary.json")
    finally:
        torch.set_num_threads(threads)


def test_phase_boundaries_require_all_weights_and_monotone_finite_chronology(tmp_path):
    plan = structural_plan()
    hashes = {f"fits/{name}/weights.pt": "f" * 64 for name in plan["fit_order"]}
    plan["parent_source"] = {"members": hashes}
    base.write(tmp_path / "random-streams.json", {"synthetic": True})
    rows = {
        "started.json": {"plan_sha256": "a" * 64, "unix_time": 10.0},
        "inherited-fits-ready.json": {
            "plan_sha256": "a" * 64,
            "fit_order": plan["fit_order"],
            "new_fits": 0,
            "files": hashes,
            "elapsed_seconds": 0.5,
            "unix_time": 10.5,
        },
        "control-completed.json": {
            "plan_sha256": "a" * 64,
            "rows": len(audit.control_order(plan)),
            "elapsed_seconds": 3.0,
        },
        "diagnostic-completed.json": {
            "plan_sha256": "a" * 64,
            "roots": 4,
            "native_transitions": 100,
            "elapsed_seconds": 4.0,
        },
    }
    for name, row in rows.items():
        base.write(tmp_path / name, row)
    rows["evaluation-started.json"] = {
        "plan_sha256": "a" * 64,
        "elapsed_seconds": 1.0,
        "unix_time": 11.0,
        "inherited_fits_ready_sha256": audit.protocol.sha(tmp_path / "inherited-fits-ready.json"),
        "random_streams_sha256": audit.protocol.sha(tmp_path / "random-streams.json"),
    }
    base.write(tmp_path / "evaluation-started.json", rows["evaluation-started.json"])
    completion = {"wall_seconds": 5.0, "evaluation_started_elapsed_seconds": 1.0}
    assert audit.audit_boundaries(plan, "a" * 64, tmp_path, completion)["logged_all_fits_before_evaluation"]
    for file, change in [
        ("evaluation-started.json", {"elapsed_seconds": 0.1}),
        ("evaluation-started.json", {"unix_time": 9.0}),
        ("evaluation-started.json", {"random_streams_sha256": "0" * 64}),
        ("inherited-fits-ready.json", {"files": {}}),
        ("control-completed.json", {"rows": 1}),
        ("diagnostic-completed.json", {"elapsed_seconds": 6.0}),
        ("diagnostic-completed.json", {"roots": 3}),
        ("started.json", {"unix_time": "nan"}),
    ]:
        base.write(tmp_path / file, rows[file] | change)
        with pytest.raises(ValueError):
            audit.audit_boundaries(plan, "a" * 64, tmp_path, completion)
        base.write(tmp_path / file, rows[file])


def test_costs_are_reconstructed_and_inherited_fits_not_added_twice(tmp_path):
    prior = {"inherited_fit_wall_seconds": 10.0, "cumulative_attempt_wall_seconds": 40.0}
    plan = {"parent_source": {"prior_costs": prior}}
    completed = {
        "inherited_fit_wall_seconds": 10.0,
        "prior_cumulative_attempt_wall_seconds": 40.0,
        "cumulative_attempt_wall_seconds": 60.0,
        "wall_seconds": 20.0,
    }
    boundary = {
        "evaluation_started_elapsed_seconds": 1.0,
        "control_completed_elapsed_seconds": 8.0,
        "diagnostic_completed_elapsed_seconds": 16.0,
    }
    controls = {
        "ordinary": {
            "synthetic": {
                "row_wall_seconds": 5.0,
                "setup_seconds": 1.0,
                "decision_wall_seconds": 2.0,
                "native_step_seconds": 1.0,
            }
        }
    }
    roots = [{"timing": {"row_wall_seconds": 4.0}}]
    costs = {
        "inherited_setup_seconds": 1.0,
        "innovation_generation_and_storage_seconds": 1.0,
        "control_row_wall_seconds": 5.0,
        "control_setup_seconds": 1.0,
        "control_decision_seconds": 2.0,
        "control_native_step_seconds": 1.0,
        "diagnostic_collection_seconds": 2.0,
        "diagnostic_root_wall_seconds": 4.0,
        "prior_costs": prior,
        "new_fits": 0,
        "accounting": "synthetic",
    }
    base.write(tmp_path / "costs.json", costs)
    actual = audit.audit_costs(plan, tmp_path, completed, boundary, controls, roots)
    assert actual["cumulative_attempt_wall_seconds"] == 60.0
    assert actual["fresh_evaluation_plus_inherited_fits_seconds"] == 30.0
    for key, value in [
        ("control_decision_seconds", 1.5),
        ("diagnostic_root_wall_seconds", 3.0),
        ("innovation_generation_and_storage_seconds", 10.0),
        ("new_fits", 1),
        ("diagnostic_collection_seconds", "nan"),
    ]:
        base.write(tmp_path / "costs.json", costs | {key: value})
        with pytest.raises(ValueError):
            audit.audit_costs(plan, tmp_path, completed, boundary, controls, roots)
    base.write(tmp_path / "costs.json", costs)
    with pytest.raises(ValueError, match="double counting"):
        audit.audit_costs(
            plan, tmp_path, completed | {"cumulative_attempt_wall_seconds": 70.0}, boundary, controls, roots
        )


def test_scored_manifest_excludes_all_engineering_role_seeds_and_spawned_states():
    plan = structural_plan()
    plan.update(
        rng_namespace=audit.protocol.SCORED_NAMESPACE,
        engineering_rng_namespaces=list(audit.protocol.ENGINEERING_NAMESPACES),
    )
    # Manifest construction hashes initial states only, never consumes samples.
    contract = audit.protocol.stream_contract(plan)
    assert contract["namespace"] == audit.protocol.SCORED_NAMESPACE
    assert len(contract["engineering_exclusions"]) == 4
    seeds = set(contract["registry"].values())
    states = {row["initial_state_sha256"] for row in contract["generators"].values()}
    for prior in contract["engineering_exclusions"]:
        assert prior["coverage"] == {
            "control_episodes": 64,
            "diagnostic_episodes": 16,
            "steps": 50,
            "diagnostic_branches": 4,
        }
        assert not seeds.intersection(prior["registry"].values())
        assert not states.intersection(row["initial_state_sha256"] for row in prior["generators"].values())
        assert all(row["draws_for_manifest"] == 0 for row in prior["generators"].values())
    with pytest.raises(ValueError, match="Engineering root seed collision"):
        audit.protocol.stream_contract(plan | {"rng_namespace": audit.protocol.ENGINEERING_NAMESPACES[0]})
