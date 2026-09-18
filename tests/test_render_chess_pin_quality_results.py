# SPDX-License-Identifier: GPL-3.0-only
"""Constructed receipt/manifest fixtures only; no chess result files are read."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_chess_pin_quality_results as figures


def build_fixture(folder):
    """Fake numerical/artifact fixture, visibly marked and never study evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    execution, audit = folder / "execution", folder / "audit"
    execution.mkdir()
    audit.mkdir()
    arms = [*figures.CORE, "union:edits"]
    methods = ["base", *arms]
    p = {"version": figures.VERSION, "arms": arms, "methods": methods, "seeds": list(figures.SEEDS),
         "fits": 24, "training_updates": 36864, "updates_per_fit": 1536, "training_roots": 32768,
         "batch_size": 128, "evaluation_roots_per_panel": 2048, "prediction_records": 110592,
         "candidate_scores": 3226149, "native_root_backbone_reconstructions": 12288, "quality_checks": 16,
         "score_tolerance": 1e-5, "time_cap_seconds": 43200, "audit_time_cap_seconds": 7200,
         "parameters": {arm: 16658 for arm in arms}, "limits": "Constructed test fixture, not chess results.",
         "prior_knowledge": "Synthetic values used solely to exercise the renderer."}
    plan = {"protocol": p, "publication_fixture": True,
            "recovery": {"prior_update_seconds_retained": 80., "prior_fit_seconds_overlapping": 50.,
                         "prior_termination_cause_known": False}}
    path = folder / "plan.json"
    figures.write(path, plan)
    plan_hash = figures.sha(path)
    for name in figures.expected_members(p):
        target = execution / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Synthetic fixture placeholder; not a model or prediction.\n")
    metrics = {}
    for index, arm in enumerate(methods):
        for seed_index, seed in enumerate(figures.SEEDS):
            metrics[f"{arm}-{seed}"] = {split: {"examples": 2048,
                "agreement": (700 + 20 * index + 4 * seed_index + 12 * (split == "shift")) / 2048,
                "target_nll": 3 - .02 * index + .01 * seed_index + .1 * (split == "shift")}
                for split in figures.SPLITS}
    gate = figures.quality_checks(metrics, arms)
    intervals = [{"split": row["split"], "comparator": row["comparator"], "games": 512, "roots": 2048,
                  "draws": 2000, "seed": 996101 + (row["split"] == "shift"), "point_gain": row["mean_gain"],
                  "percentile95": [row["mean_gain"] - .003, row["mean_gain"] + .003]} for row in gate]
    native = {arm: {"max_score_error": 1e-7, "failed_cases": 0, "choice_changes": 0} for arm in methods}
    native["joint"] = {"max_score_error": .0002, "failed_cases": 4, "choice_changes": 2}
    numerical = {"native_root_backbone_reconstructions": 12288, "native_method_comparisons": 110592,
                 "candidate_score_comparisons": 3226149, "max_native_cached_score_error": .0002,
                 "failed_native_method_cases": 4, "native_choice_changes": 2,
                 "numerical_gate_passed": False, "native_by_method": native}
    summary = {"status": "completed", "plan_sha256": plan_hash, "fits": 24, "training_updates": 36864,
               "prediction_records": 110592, "new_engine_calls": 0, "external_model_calls": 0,
               "copied_reference_predictions": 0, "original_failed_criteria_unchanged": True,
               "limits": p["limits"], "metrics": metrics, "gate_checks": gate,
               "quality_gate_passed": all(row["passed"] for row in gate),
               "conditional_game_bootstrap": intervals, **numerical, "continuation_passed": False,
               "wall_seconds": 200.}
    for index, arm in enumerate(arms):
        for seed_index, seed in enumerate(figures.SEEDS):
            target = execution / f"fits/{figures.fit_name(arm, seed)}"
            fit = {"status": "completed", "arm": arm, "seed": seed, "plan_sha256": plan_hash,
                   "updates": 1536, "examples_seen": 1536 * 128, "parameters": p["parameters"][arm],
                   "seconds": 2 + .3 * index + .1 * seed_index,
                   "initial_sha256": figures.sha(target / "initial.pt"),
                   "weights_sha256": figures.sha(target / "weights.pt"),
                   "learning_sha256": figures.sha(target / "learning.jsonl")}
            figures.write(target / "training.json", fit)
    figures.write(execution / "summary.json", summary)
    members = {name: figures.sha(execution / name) for name in figures.expected_members(p)}
    figures.write(execution / "completed.json", {"status": "completed", "plan_sha256": plan_hash, "files": members})
    replay_members = {}
    for seed in figures.SEEDS:
        for split in figures.SPLITS:
            name = f"native/{seed}-{split}.jsonl"
            target = audit / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((execution / name).read_bytes())
            replay_members[name] = figures.sha(target)
    receipt = {"status": "completed", "plan_sha256": plan_hash, "summary_sha256": figures.sha(execution / "summary.json"),
               "execution_receipt_sha256": figures.sha(execution / "completed.json"), "auditor_sha256": "c" * 64,
               "training_updates_checked": 36864, "fresh_initial_states_exact": 24, "final_checkpoint_identities_checked": 24,
               "prediction_records_replayed": 110592, "candidate_scores_replayed": 3226149,
               "cached_scores_and_nll_exact": True, "native_vectors_exact": True,
               "native_root_backbone_reconstructions": 12288, "gate_recomputed": gate,
               "numerical_gate_recomputed": numerical, "bootstrap_intervals_recomputed": 16,
               "replay_native_files_sha256": replay_members, "continuation_passed": False,
               "wall_seconds": 30., "scope": "Constructed saved-artifact fixture only, no neural audit was performed."}
    figures.write(audit / "receipt.json", receipt)
    return path, audit, execution, figures.sha(audit / "receipt.json")


def test_all_fits_baselines_gates_and_finite_parity_failures_are_retained(tmp_path):
    inputs = build_fixture(tmp_path / "fixture")
    data = figures.figure_data(*figures.authenticate(*inputs), inputs[-1])
    assert len(data["fits"]) == 24 and len(data["metrics"]) == 27
    assert len(data["gate_checks"]) == len(data["conditional_game_bootstrap"]) == 16
    assert not data["quality_gate_passed"] and not data["continuation_passed"]
    assert not data["numerical"]["numerical_gate_passed"]
    assert data["numerical"]["failed_native_method_cases"] == 4
    assert data["numerical"]["native_choice_changes"] == 2
    assert data["costs"]["v2_logged_update_seconds_lower_bound"] == 80
    assert data["costs"]["v2_completed_fit_seconds_overlapping"] == 50
    assert "total" not in data["costs"]
    assert data["gate_checks"][0]["required_mean_gain"] == 0
    assert data["gate_checks"][1]["required_mean_gain"] == .01


@pytest.mark.parametrize("corruption", ["receipt", "unfinished", "fit_cost", "missing_fit", "extra_member", "native", "symlink"])
def test_authentication_refuses_unbound_or_partial_members_before_rendering(tmp_path, corruption):
    plan, audit, execution, digest = build_fixture(tmp_path / "fixture")
    if corruption == "receipt":
        digest = "0" * 64
    elif corruption == "unfinished":
        (audit / "failed.json").write_text('{}')
    elif corruption == "fit_cost":
        path = execution / "fits/joint-97/training.json"
        value = figures.read(path)
        value["seconds"] = .00001
        figures.write(path, value)
    elif corruption == "missing_fit":
        (execution / "fits/joint-127/weights.pt").unlink()
    elif corruption == "extra_member":
        (execution / "partial.json").write_text('{}')
    elif corruption == "native":
        (audit / "native/109-shift.jsonl").write_text('Changed native bytes')
    else:
        path = execution / "fits/joint-97/weights.pt"
        path.unlink()
        path.symlink_to(execution / "fits/joint-109/weights.pt")
    out = tmp_path / "figures"
    with pytest.raises(ValueError):
        figures.render(plan, audit, execution, digest, out)
    assert not out.exists()


@pytest.mark.parametrize("corruption", ["baseline", "gate", "native_gate", "interval", "fit_receipt", "negative_cost"])
def test_numerical_scope_and_gate_corruptions_are_not_hidden(tmp_path, corruption):
    inputs = build_fixture(tmp_path / "fixture")
    plan, receipt, summary, fits, completed = copy.deepcopy(figures.authenticate(*inputs))
    if corruption == "baseline":
        del summary["metrics"]["base-127"]
    elif corruption == "gate":
        summary["quality_gate_passed"] = True
    elif corruption == "native_gate":
        summary["numerical_gate_passed"] = receipt["numerical_gate_recomputed"]["numerical_gate_passed"] = True
    elif corruption == "interval":
        summary["conditional_game_bootstrap"].pop()
    elif corruption == "fit_receipt":
        fits["joint-97"]["initial_sha256"] = "0" * 64
    else:
        fits["joint-97"]["seconds"] = -1
    with pytest.raises(ValueError):
        figures.figure_data(plan, receipt, summary, fits, completed, inputs[-1])


def test_constructed_fixture_renders_all_outputs_and_provenance(tmp_path):
    inputs = build_fixture(tmp_path / "fixture")
    out = tmp_path / "figures"
    receipt = figures.render(*inputs, out)
    expected = {f"{name}.{suffix}" for name in figures.FIGURES for suffix in ("png", "svg", "pdf")}
    assert set(receipt["files"]) == expected | {"README.md", "figure-data.json"}
    assert receipt["new_model_calls"] == receipt["new_engine_calls"] == 0
    for name, digest in receipt["files"].items():
        assert figures.sha(out / name) == digest
    for name in figures.FIGURES:
        assert "SYNTHETIC ENGINEERING FIXTURE: NOT RESEARCH RESULTS" in (out / f"{name}.svg").read_text()
    text = (out / "README.md").read_text()
    assert "Continuation: **FAIL**" in text and "No Elo or gameplay claim" in text
    assert "Frozen backbone" in text and "Union edits" in text
    with pytest.raises(ValueError, match="Exclusive"):
        figures.render(*inputs, out)
