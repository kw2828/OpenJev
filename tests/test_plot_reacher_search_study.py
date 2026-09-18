"""Publication validation using fabricated complete summaries, never study output."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_reacher_search_study as plot


@pytest.fixture
def complete_summary():
    """Every number is fabricated solely to exercise validation and layout."""
    controls = {panel: {} for panel in plot.PANELS}
    for panel in plot.PANELS:
        for name in sorted(plot.LEARNED | plot.REFERENCES):
            learned = name in plot.LEARNED
            planned = learned or name in ("known_state", "particle")
            count = (64 if name.endswith("rs64") or name in plot.REFERENCES else 256) if planned else 0
            cost = 80.0 if name.startswith("residual") else 100.0
            if name.endswith("cem256"):
                cost *= 0.9
            controls[panel][name] = {
                "episode_costs": [cost] * 64,
                "mean_cost": cost,
                "setup_seconds": 0.1,
                "decision_seconds": [0.01] * 50,
                "decision_wall_seconds": 0.5,
                "native_step_seconds": 0.1,
                "row_wall_seconds": 1.0,
                "batch_latency_seconds": {key: 0.01 for key in ("mean", "p50", "p95", "max")},
                "per_case_amortized_seconds": 0.01 / 64,
                "search_seconds": [0.005] * 50 if learned else [],
                "clipping_fraction": 0.2 if learned else None,
                "planner_used": planned,
                "candidate_evaluations": 64 * 50 * count,
                "imagined_transitions": 64 * count * 534,
            }

    def checks(expected):
        values = [
            {
                "name": name,
                "actual": actual,
                "threshold": threshold,
                "direction": "lt" if strict else "le",
                "passed": bool(actual < threshold if strict else actual <= threshold),
            }
            for name, (actual, threshold, strict) in expected.items()
        ]
        return {
            "checks": values,
            "passed": all(row["passed"] for row in values),
            "scope": "synthetic fixture",
        }

    gate, competence = {}, {}
    for panel in plot.PANELS[1:]:
        gate[f"residual/{panel}/cem_mean_vs_rs256"] = (72.0, 76.0, False)
        for seed in plot.SEEDS:
            gate[f"residual-{seed}/{panel}/cem_vs_rs256"] = (72.0, 80.0, True)
    for method in plot.PLANNERS:
        rows = {"physics_ordinary_vs_zero": (100.0, 90.0, False)}
        residual, free = (72.0, 90.0) if method == "cem256" else (80.0, 100.0)
        for panel in plot.PANELS[1:]:
            for seed in plot.SEEDS:
                rows[f"residual-{seed}/{panel}/vs_zero"] = (residual, 90.0, False)
                rows[f"residual-{seed}/{panel}/vs_paired_free"] = (residual, free, True)
            rows[f"residual/{panel}/mean_vs_free"] = (residual, 0.95 * free, False)
        competence[method] = checks(rows)
    order = [
        {"panel": panel, "fit": fit, "planner": method, "union_index": 64 + i}
        for i, (panel, fit, method) in enumerate(
            (p, f, m) for p in plot.PANELS for f in plot.FIT_ORDER for m in plot.PLANNERS
        )
    ]
    ids = [f"common/{i}" for i in range(64)] + [
        f"selected/{r['panel']}/{r['fit']}/{r['planner']}" for r in order
    ]
    roots = []
    for episode in range(16):
        for ordinal, step in enumerate((6, 10, 14, 47)):
            horizon = min(12, 50 - step)
            metrics = {panel: {} for panel in plot.PANELS}
            for panel in plot.PANELS:
                for name in plot.LEARNED:
                    metrics[panel][name] = {
                        "common_bank_spearman": 0.5 if ordinal % 2 else None,
                        "native_selected_return": -float(horizon),
                        "native_selected_branch_returns": [-float(horizon)] * 4,
                        "finite_set_regret": 1.0,
                        "evaluated_union_slots": 118,
                        "selected_raw_prediction_bias": 2.0,
                        "selected_clipped_prediction_bias": 1.0,
                        "common_bank_raw_return_mse": 4.0,
                        "common_bank_clipped_return_mse": 1.0,
                        "reward_prediction_clipping_fraction": 0.2,
                    }
            roots.append(
                {
                    "episode_index": episode,
                    "root_ordinal": ordinal,
                    "phase": 0,
                    "step": step,
                    "horizon": horizon,
                    "identity_slots": 118,
                    "unique_sequence_count": 100,
                    "duplicates_retained": True,
                    "union_ids": ids,
                    "search_order": order,
                    "history_members": {f"history-{panel}.npz": "d" * 64 for panel in plot.PANELS},
                    "metrics": metrics,
                    "native_replay": {
                        "transitions": 118 * 4 * horizon,
                        "max_abs_error": 0.0,
                        "saved_output_only": True,
                        "new_policy_calls": 0,
                    },
                    "timing": {
                        "identity_slots": 118,
                        "native_transitions": 118 * 4 * horizon,
                        "observation_assimilations": 18 * (step + 1),
                        "history_action_advances": 18 * step,
                        "belief_seconds": 0.1,
                        "search_seconds": 0.1,
                        "native_and_storage_seconds": 0.1,
                        "row_wall_seconds": 0.4,
                    },
                }
            )
    aggregate = {panel: {} for panel in plot.PANELS}
    for panel in plot.PANELS:
        for name in plot.LEARNED:
            aggregate[panel][name] = {
                "common_bank_spearman_mean": 0.5,
                "rank_defined_roots": 32,
                "rank_undefined_roots": 32,
                **{
                    key + "_mean": float(np.mean([r["metrics"][panel][name][key] for r in roots]))
                    for key in plot.METRIC_FIELDS
                },
            }
    prior = {"inherited_fit_wall_seconds": 6.0, "cumulative_attempt_wall_seconds": 100.0}
    costs = {
        "inherited_setup_seconds": 1.0,
        "innovation_generation_and_storage_seconds": 1.0,
        "control_row_wall_seconds": 66.0,
        "control_setup_seconds": 6.6,
        "control_decision_seconds": 33.0,
        "control_native_step_seconds": 6.6,
        "diagnostic_collection_seconds": 2.0,
        "diagnostic_root_wall_seconds": 25.6,
        "new_evaluation_wall_seconds": 200.0,
        "inherited_fit_wall_seconds": 6.0,
        "prior_cumulative_attempt_wall_seconds": 100.0,
        "audit_validation_wall_seconds": 1.0,
        "cumulative_attempt_wall_seconds": 300.0,
        "fresh_evaluation_plus_inherited_fits_seconds": 206.0,
        "prior_costs": prior,
    }
    return {
        "status": "completed",
        "version": "reacher-search-v1",
        "saved_output_only": True,
        "plan_sha256": "a" * 64,
        "execution_completed_sha256": "b" * 64,
        "new_model_calls": 0,
        "new_policy_calls": 0,
        "new_mpc_calls": 0,
        "new_fits": 0,
        "fits": {name: {"wall_seconds": 1.0} for name in plot.FIT_ORDER},
        "native_transition_counts": {
            "control": 211200,
            "diagnostic_cohort": 800,
            "diagnostic_branches": 294528,
            "inherited_training": 38400,
            "fresh_evaluation": 506528,
        },
        "native_transitions_checked": 544928,
        "learned_control_imagined_transitions": 354336768,
        "native_max_abs_error": 0.0,
        "wall_seconds": 200.0,
        "costs": costs,
        "source_lineage": {"plan_sha256": "c" * 64, "audit_receipt_sha256": "d" * 64, "prior_costs": prior},
        "historical_parent_qualification": {
            "rerun": False,
            "plan_sha256": "c" * 64,
            "audit_receipt_sha256": "d" * 64,
        },
        "control": controls,
        "continuation_gate": checks(gate),
        "control_competence": competence,
        "diagnostics": {"roots": roots, "aggregate": aggregate},
    }


def save_audit(folder, summary):
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(summary, allow_nan=False))
    (folder / "README.md").write_text("Fabricated engineering data only.\n")
    receipt = {
        "status": "completed",
        "version": "reacher-search-v1",
        "saved_output_only": True,
        "plan_sha256": summary["plan_sha256"],
        "execution_completed_sha256": summary["execution_completed_sha256"],
        "plan_canonical_sha256": "e" * 64,
        "source_sha256": {"synthetic.py": "f" * 64},
        "runtime": {"synthetic_fixture": True},
        "execution_members": {"synthetic.npz": "0" * 64},
        "parent_source": copy.deepcopy(summary["source_lineage"]),
        "costs": copy.deepcopy(summary["costs"]),
        "files": {name: plot.sha(folder / name) for name in ("summary.json", "README.md")},
    }
    (folder / "receipt.json").write_text(json.dumps(receipt, allow_nan=False))
    return plot.sha(folder / "receipt.json")


def test_complete_synthetic_summary_authenticates_and_renders(tmp_path, complete_summary):
    folder = tmp_path / "audit"
    expected = save_audit(folder, complete_summary)
    assert plot.authenticate(folder, expected) == complete_summary
    result = plot.render(folder, expected, tmp_path / "figures")
    assert set(result["files"]) == {
        "all-control-rows.csv",
        "all-diagnostic-rows.csv",
        "search-control.png",
        "search-compute.png",
        "search-model-error.png",
    }
    assert (tmp_path / "figures" / "search-control.png").stat().st_size > 1000


@pytest.mark.parametrize(
    "mutation",
    [
        "fit",
        "control",
        "extra_control",
        "episodes",
        "decision_count",
        "mean",
        "decision_sum",
        "amortized",
        "latency",
        "nested_time",
        "candidate_count",
        "imagined_count",
        "search_time",
        "clipping",
        "inference",
        "root",
        "duplicate_root",
        "slots",
        "unique_slots",
        "root_phase",
        "history",
        "arm",
        "branch",
        "branch_mean",
        "aggregate",
        "aggregate_mean",
        "undefined_rank",
        "gate",
        "gate_count",
        "competence",
        "prior",
        "cumulative",
        "cost_sum",
        "wall",
        "historical",
        "transition_total",
    ],
)
def test_authenticated_but_partial_or_inconsistent_summary_is_rejected(tmp_path, complete_summary, mutation):
    summary = complete_summary
    name = "residual-271-cem256"
    row = summary["control"]["ordinary"][name]
    root = summary["diagnostics"]["roots"][0]
    metric = root["metrics"]["ordinary"][name]
    aggregate = summary["diagnostics"]["aggregate"]["ordinary"][name]
    if mutation == "fit":
        del summary["fits"]["free-271"]
    elif mutation == "control":
        del summary["control"]["shift"][name]
    elif mutation == "extra_control":
        summary["control"]["shift"]["selected-best"] = row
    elif mutation == "episodes":
        row["episode_costs"].pop()
    elif mutation == "decision_count":
        row["decision_seconds"].pop()
    elif mutation == "mean":
        row["mean_cost"] += 1
    elif mutation == "decision_sum":
        row["decision_wall_seconds"] += 1
    elif mutation == "amortized":
        row["per_case_amortized_seconds"] *= 64
    elif mutation == "latency":
        row["batch_latency_seconds"]["p95"] += 1
    elif mutation == "nested_time":
        row["row_wall_seconds"] = 0.5
    elif mutation == "candidate_count":
        row["candidate_evaluations"] -= 1
    elif mutation == "imagined_count":
        row["imagined_transitions"] -= 1
    elif mutation == "search_time":
        row["search_seconds"][0] = 0.5
    elif mutation == "clipping":
        row["clipping_fraction"] = 1.1
    elif mutation == "inference":
        summary["new_model_calls"] = 1
    elif mutation == "root":
        summary["diagnostics"]["roots"].pop()
    elif mutation == "duplicate_root":
        summary["diagnostics"]["roots"][-1] = root
    elif mutation == "slots":
        root["identity_slots"] = 117
    elif mutation == "unique_slots":
        root["unique_sequence_count"] = 119
    elif mutation == "root_phase":
        root["step"] += 1
    elif mutation == "history":
        root["history_members"].pop("history-shift.npz")
    elif mutation == "arm":
        del root["metrics"]["shift"][name]
    elif mutation == "branch":
        metric["native_selected_branch_returns"].pop()
    elif mutation == "branch_mean":
        metric["native_selected_return"] += 1
    elif mutation == "aggregate":
        del summary["diagnostics"]["aggregate"]["shift"][name]
    elif mutation == "aggregate_mean":
        aggregate["finite_set_regret_mean"] += 1
    elif mutation == "undefined_rank":
        aggregate["rank_undefined_roots"] -= 1
    elif mutation == "gate":
        summary["continuation_gate"]["passed"] = not summary["continuation_gate"]["passed"]
    elif mutation == "gate_count":
        summary["continuation_gate"]["checks"].pop()
    elif mutation == "competence":
        del summary["control_competence"]["rs64"]
    elif mutation == "prior":
        summary["costs"]["prior_cumulative_attempt_wall_seconds"] += 1
    elif mutation == "cumulative":
        summary["costs"]["cumulative_attempt_wall_seconds"] += 6
    elif mutation == "cost_sum":
        summary["costs"]["control_row_wall_seconds"] += 1
    elif mutation == "wall":
        summary["wall_seconds"] = 3601
    elif mutation == "historical":
        summary["historical_parent_qualification"]["rerun"] = True
    else:
        summary["native_transitions_checked"] -= 1
    expected = save_audit(tmp_path / "audit", summary)
    with pytest.raises(ValueError):
        plot.authenticate(tmp_path / "audit", expected)


@pytest.mark.parametrize("key", ["plan_sha256", "execution_completed_sha256", "parent_source", "costs"])
def test_receipt_summary_provenance_mismatch_rejected(tmp_path, complete_summary, key):
    folder = tmp_path / "audit"
    save_audit(folder, complete_summary)
    receipt = plot.read(folder / "receipt.json")
    receipt[key] = "0" * 64 if key.endswith("sha256") else {}
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="provenance|lineage|cost"):
        plot.authenticate(folder, plot.sha(folder / "receipt.json"))


@pytest.mark.parametrize(
    "text", ['{"n":NaN}', '{"n":Infinity}', '{"n":-Infinity}', '{"n":1e999}', '{"n":0,"n":1}']
)
def test_strict_json_rejects_nonfinite_and_duplicate_keys(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        plot.read(path)


def test_fabricated_minimal_layout_fixture_is_rejected(tmp_path):
    # Recreate the former permissive layout receipt without relying on local tmp artifacts.
    folder = tmp_path / "fabricated-layout"
    folder.mkdir()
    minimal = {"status": "completed", "version": "reacher-search-v1", "saved_output_only": True}
    (folder / "summary.json").write_text(json.dumps(minimal))
    receipt = minimal | {"files": {"summary.json": plot.sha(folder / "summary.json")}}
    (folder / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        plot.authenticate(folder, plot.sha(folder / "receipt.json"))


def test_external_identity_member_tamper_and_symlink_rejected(tmp_path, complete_summary):
    folder = tmp_path / "audit"
    expected = save_audit(folder, complete_summary)
    with pytest.raises(ValueError, match="External"):
        plot.authenticate(folder, "0" * 64)
    (folder / "README.md").write_text("Changed after receipt")
    with pytest.raises(ValueError, match="member"):
        plot.authenticate(folder, expected)
    expected = save_audit(folder, complete_summary)
    moved = tmp_path / "moved-summary.json"
    (folder / "summary.json").rename(moved)
    (folder / "summary.json").symlink_to(moved)
    with pytest.raises(ValueError, match="member"):
        plot.authenticate(folder, expected)
