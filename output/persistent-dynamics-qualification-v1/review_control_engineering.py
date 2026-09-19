"""Terminal saved-output reviewer; no simulation, search, learning or RNG.

Only the completed six-role engineering410 case is accepted. The caller supplies
its expected outer completion hash. Receipt integrity is not external process
attestation. All payloads authenticate before numerical result aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("nominal", "adaptive", "frozen", "public_gain", "true_state", "zero")
WINDOWS = {"TOTAL": (0, 200), "PRE": (0, 80), "POST": (80, 200), "MOVE": (100, 150)}
DRIVER = "output/persistent-dynamics-qualification-v1/run_control_engineering.py"
DRIVER_SHA = "aeb94b726292103a97d62bdd1c5c06d7f945380a1b644a17623ea182cc534062"
PROTECTED = (
    ("evidence/reacher-two-observation-study-v1/protocol/plan.json", "sources", 116,
     "e3196e1e3ed6b513ad65918a1f7780def1f708867c6ec7aaa61ccba668624038"),
    ("evidence/reacher-innovation-pilot-v1/protocol/plan.json", "source_sha256", 136,
     "561eb5a73f30ce81453941a6ade73cf15f0332af26cfb6ef49a7a17a35eff3de"),
)
SOURCE_NAMES = ["src/openjev/research/"+name+".py" for name in (
    "reacher_tracking_dynamics", "reacher_tracking_identification", "reacher_tracking_control",
    "reacher_tracking_policy", "reacher_tracking_rollout", "reacher_tracking_audit",
    "reacher_geometry_physics", "reacher_geometry_reward", "reacher_reward_residual",
    "reacher_physics_control", "reacher_adaptive_search", "reacher_search_protocol", "robotics_reacher")]+[DRIVER]
LIMITS = [
    "One reused seed410 engineering case, one gain-switch direction and fixed goals; no independent cohort or qualification gate.",
    "Recorded native costs describe this case, not a learned-model advantage or a confirmed adaptation benefit.",
    "Root estimates use only completed past transitions; update estimates are separately compared with the gain of the transition just observed.",
    "MOVE is fixed [100,150), the first complete goal interval after the switch; it is not a selected favorable interval.",
    "Known-physics planning and identification are charged; equal candidate budgets are not equal total compute.",
    "Shared-host wall times include instrumentation; nested times must not be added to enclosing times.",
    "This reviewer authenticates saved audit claims and recomputes arithmetic; it performs no native replay or inference and does not attest subprocess exits.",
]


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)
    return float(value)


def checked(path, expected):
    require(type(expected) is str and len(expected) == 64 and not Path(path).is_symlink()
            and sha(path) == expected, "authenticated file " + str(path))


def window_stats(values):
    values = np.asarray(values, np.float64)
    require(values.shape == (200,) and np.isfinite(values).all(), "complete 200 finite values")
    return {name: {"start": first, "stop": last, "actions": last-first,
                   "sum": float(values[first:last].sum()), "mean_per_action": float(values[first:last].mean())}
            for name, (first, last) in WINDOWS.items()}


def authenticate(execution, expected_completed):
    """Finish all six terminal/hash checks before parsing any episode metrics."""
    execution = Path(execution).resolve()
    checked(execution/"completed.json", expected_completed)
    complete = read(execution/"completed.json")
    require(complete["status"] == "completed" and complete["engineering"] is True
            and complete["scientific_gate"] is None and complete["scientific_data_allocated"] is False,
            "terminal engineering-only classification")
    require([row["arm"] for row in complete["rows"]] == list(ARMS)
            and [row["arm"] for row in complete["audits"]] == list(ARMS), "all six rows and completed audits")
    checked(execution/"execution-completed.json", complete["execution_completed_sha256"])
    executed = read(execution/"execution-completed.json")
    require(executed["status"] == "completed" and executed["engineering"] is True
            and executed["rows"] == complete["rows"], "execution terminal six-row boundary")
    started = read(execution/"started.json")
    expected_settings = {"engineering": True, "status": "started", "seed": 410, "case_count": 1,
        "arms": list(ARMS), "steps": 200, "gain_change_step": 80, "gain_before": .7, "gain_after": 1.3,
        "noise_std": .05, "window": 20, "freeze_after": 40, "planning_horizon": 12, "action_block": 3,
        "execution_cap_seconds": 600, "audit_cap_seconds": 600, "all_rows_before_audits": True,
        "scientific_gate": None, "automatic_retry": False,
        "target_events": {"0": [.12, -.04], "50": [-.12, .08], "100": [.08, .13], "150": [-.09, -.12]}}
    require(all(started.get(k) == v for k, v in expected_settings.items()), "exact engineered recipe")
    sources = started["source_sha256"]
    require(set(sources) == set(SOURCE_NAMES) and complete["source_sha256"] == sources
            and sources[DRIVER] == DRIVER_SHA, "exact fourteen-source driver contract")
    expected_files = {"started.json", "completed.json", "execution-completed.json", "case.npz"}
    for name, digest in sources.items():
        checked(ROOT/name, digest); checked(execution/"source-snapshot"/name, digest)
        expected_files.add("source-snapshot/"+name)
    protected = []
    for name, key, count, digest in PROTECTED:
        checked(ROOT/name, digest); mapping = read(ROOT/name)[key]
        require(len(mapping) == count, "protected source count")
        for path, original in mapping.items():
            checked(ROOT/path, original)
        protected.append({"plan": name, "plan_sha256": digest, "unchanged_sources": count})
    require(started["protected_sources"] == [{"plan": p["plan"], "unchanged_sources": p["unchanged_sources"]}
                                               for p in protected], "declared protected sources")
    receipts, audits, input_manifest = {}, {}, None
    for row, audit_ref in zip(complete["rows"], complete["audits"], strict=True):
        arm = row["arm"]; folder = execution/"rows"/arm
        checked(folder/"completed.json", row["completed_sha256"])
        receipt = read(folder/"completed.json")
        require(receipt["status"] == "completed" and receipt["engineering"] is True and receipt["arm"] == arm
                and receipt["steps"] == receipt["native_decisions"] == receipt["policy_decisions"]
                == receipt["observation_updates"] == 200 and receipt["wall_seconds"] == row["wall_seconds"],
                "complete engineering row identity")
        expected_row = {"started.json", "initial-controller.json", "controller.json", "episode.json"}
        expected_row |= {f"decisions/{t:03d}.{ext}" for t in range(200) for ext in ("npz", "json")}
        expected_row |= {f"observations/{t+1:03d}.json" for t in range(200)}
        require(set(receipt["files"]) == expected_row, "exact row payload coverage")
        for path, digest in receipt["files"].items():
            checked(folder/path, digest); expected_files.add(f"rows/{arm}/"+path)
        expected_files.add(f"rows/{arm}/completed.json")
        row_start = read(folder/"started.json")
        require(row_start["arm"] == arm and row_start["steps"] == 200 and row_start["reset_seed"] == 410
                and row_start["engineering"] is True and row_start["automatic_retry"] is False,
                "engineering row source case")
        for key in ("window", "freeze_after", "noise_std", "gain_grid", "planning_horizon", "action_block"):
            require(row_start[key] == started[key], "same controller settings " + key)
        require(row_start["frame_skip"] == 2 and len(row_start["inputs_by_step"]) == 200, "input/frame coverage")
        if input_manifest is None:
            input_manifest = row_start["inputs_by_step"]
        else:
            require(row_start["inputs_by_step"] == input_manifest, "same complete innovations across all six rows")
        audit_name = "audit-"+arm+".json"; expected_files.add(audit_name)
        checked(execution/audit_name, audit_ref["sha256"]); audit = read(execution/audit_name)
        require(audit["status"] == "completed" and audit["arm"] == arm
                and audit["row_completed_sha256"] == row["completed_sha256"]
                and audit["new_model_calls"] == audit["new_planner_calls"] == audit["new_rng_draws"] == 0,
                "completed saved-only audit identity")
        require(audit["native_control"]["transitions"] == 200 and audit["native_control"]["substeps"] == 400,
                "all native real transitions audited")
        require(audit["candidate_transitions_checked"] == (0 if arm == "zero" else 597504)
                and audit["selected_transitions_checked"] == (0 if arm == "zero" else 200)
                and audit["identifier_transitions_checked"] == {"adaptive": 4200, "frozen": 840}.get(arm, 0),
                "all candidate/selected/identifier replay coverage")
        for value in (audit["native_control"]["max_abs_error"], audit["max_candidate_state_abs_error"],
                      audit["max_identifier_abs_error"]):
            require(finite(value, "finite native maximum") <= 1e-10, "native replay tolerance")
        finite(audit["max_geometry_abs_error"], "geometry replay maximum")
        receipts[arm], audits[arm] = receipt, audit
    for t, binding in enumerate(input_manifest):
        require(set(binding) == {"stem", "npz_sha256", "json_sha256"}, "shared input binding fields")
        require(Path(binding["stem"]).resolve() == execution/"inputs"/f"{t:03d}", "exact shared input stem")
        for ext in ("npz", "json"):
            name = f"inputs/{t:03d}.{ext}"; checked(execution/name, binding[ext+"_sha256"]); expected_files.add(name)
    actual = {p.relative_to(execution).as_posix() for p in execution.rglob("*") if p.is_file()}
    require(actual == expected_files and not any(p.is_symlink() for p in execution.rglob("*")),
            "complete exact root membership without failures or symlinks")
    require(executed["native_decisions"] == 1200 and executed["candidate_native_transitions"] == 2987520
            and executed["selected_native_transitions"] == 1000 and executed["identifier_native_transitions"] == 5040,
            "outer execution work")
    require(finite(executed["wall_seconds"], "execution time") <= 600
            and finite(complete["audit_wall_seconds"], "audit time") <= 600,
            "separate engineering caps")
    require(sum(r["wall_seconds"] for r in receipts.values()) <= executed["wall_seconds"]+1e-8
            and sum(a["audit_wall_seconds"] for a in audits.values()) <= complete["audit_wall_seconds"]+1e-8,
            "all sequential row/audit wall totals")
    require(finite(complete["total_wall_seconds"], "whole time") >=
            executed["wall_seconds"]+complete["audit_wall_seconds"]-1e-8, "nonoverlapping execution/audit phases")
    return {"started": started, "completed": complete, "execution": executed,
            "receipts": receipts, "audits": audits, "protected_sources": protected,
            "all_payload_sha256": {name: sha(execution/name) for name in sorted(expected_files)},
            "members": len(expected_files)}


def calculate(execution, bound):
    """Arithmetic only, called after terminal and all-payload authentication."""
    execution = Path(execution)
    with np.load(execution/"case.npz", allow_pickle=False) as saved:
        require(set(saved.files) == {"targets", "gains", "noise"}, "case array fields")
        targets, gains, noise = (saved[k] for k in ("targets", "gains", "noise"))
    for value, shape in ((targets, (201, 2)), (gains, (200,)), (noise, (200, 2))):
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), "case tensor schema")
    expected_targets = np.empty((201, 2))
    for first, stop, point in ((0, 50, [.12, -.04]), (50, 100, [-.12, .08]),
                               (100, 150, [.08, .13]), (150, 201, [-.09, -.12])):
        expected_targets[first:stop] = point
    require(np.array_equal(targets, expected_targets) and np.array_equal(gains, np.r_[np.full(80, .7), np.full(120, 1.3)]),
            "exact fixed target/gain recipe")
    grid = np.asarray(bound["started"]["gain_grid"], np.float64)
    require(np.array_equal(grid, np.linspace(.5, 1.5, 21)), "fixed global gain grid")
    rows = {}; initial = None
    for arm in ARMS:
        folder = execution/"rows"/arm; record = read(folder/"episode.json"); meta = record["metadata"]
        require(meta["seed"] == 410 and meta["horizon"] == meta["completed_decisions"] == 200
                and meta["status"] == "completed", "all complete native cases")
        for key, expected in (("target_path", targets), ("gear_multiplier", gains), ("noise", noise),
                              ("sensor_schedule", np.ones(201, bool))):
            require(np.array_equal(meta[key], expected), "exact paired case " + key)
        first = record["audit"]["decision_states"][0]
        if initial is None:
            initial = first
        else:
            require(first == initial, "same native initial state across all six roles")
        transitions = record["audit"]["transitions"]
        require(len(transitions) == 200 and len(record["policy"]["packets"]) == 201,
                "all200 commands/201 packets")
        rewards = np.asarray([r["reward"] for r in transitions], np.float64)
        distance = -np.asarray([r["reward_dist"] for r in transitions], np.float64)
        action = -np.asarray([r["reward_ctrl"] for r in transitions], np.float64)
        require(np.array_equal(rewards, bound["audits"][arm]["native_control"]["native_rewards"])
                and float(-rewards.sum()) == bound["receipts"][arm]["native_cost"]
                == bound["audits"][arm]["native_control"]["mean_cost"], "recomputed episode sum cost")
        require(np.allclose(-rewards, distance+action, rtol=0, atol=1e-14), "native reward decomposition")
        root_gains, update_gains = [], []
        for t in range(200):
            with np.load(folder/f"decisions/{t:03d}.npz", allow_pickle=False) as decision:
                root_gains.append(float(decision["planning_gain"]))
            observed = read(folder/f"observations/{t+1:03d}.json")
            update_gains.append(observed["identified_gain"])
        roots = np.asarray(root_gains)
        require(np.isfinite(roots).all() and np.isin(roots, grid).all(), "recorded finite gain choices")
        identification = None
        if arm in ("adaptive", "frozen"):
            after = np.asarray(update_gains, np.float64)
            require(np.isfinite(after).all() and np.isin(after, grid).all()
                    and roots[0] == 1. and np.array_equal(roots[1:], after[:-1]), "estimate/transition alignment")
            if arm == "frozen":
                require(np.all(after[39:] == after[39]), "estimate freezes after transitions0..39")
            identification = {
                "root_current_gain_absolute_error": window_stats(np.abs(roots-gains)),
                "after_observation_transition_gain_absolute_error": window_stats(np.abs(after-gains)),
                "root_definition": "At decision t, estimate based on observations through packet t versus current gain g_t.",
                "update_definition": "After transition t, updated/retained estimate versus g_t which generated that transition; not future g_(t+1).",
                "root_estimates": roots.tolist(), "after_transition_estimates": after.tolist()}
        else:
            require(all(x is None for x in update_gains), "nonidentifying row has no latent estimate")
        audit = bound["audits"][arm]; receipt = bound["receipts"][arm]
        rows[arm] = {"native_cost": window_stats(-rewards), "distance_cost": window_stats(distance),
            "applied_action_cost": window_stats(action), "identification": identification,
            "planning_gain_root_mae": window_stats(np.abs(roots-gains)),
            "planning_gain_scope": "May be fixed nominal or privileged current gain; only adaptive/frozen are identifier estimates.",
            "work": {"native_control_transitions": 200, "candidate_transitions": audit["candidate_transitions_checked"],
                     "selected_transitions": audit["selected_transitions_checked"],
                     "identifier_transitions": audit["identifier_transitions_checked"]},
            "wall_seconds": {key: receipt[key] for key in ("setup_seconds", "decision_seconds", "native_seconds",
                             "observation_seconds", "decision_serialization_seconds", "wall_seconds")},
            "nested_controller_costs": audit["controller_costs"], "audit_wall_seconds": audit["audit_wall_seconds"]}
    return rows


def review(execution, expected_completed, out):
    execution, out = Path(execution).resolve(), Path(out).resolve()
    require(not out.is_relative_to(execution) and not execution.is_relative_to(out), "review output separate from saved tree")
    out.mkdir(parents=True, exist_ok=False)
    tick = time.perf_counter()
    try:
        bound = authenticate(execution, expected_completed)
        rows = calculate(execution, bound)
        for path, digest in bound["all_payload_sha256"].items():
            checked(execution/path, digest)
        for name, digest in bound["started"]["source_sha256"].items():
            checked(ROOT/name, digest)
        for name, key, _, digest in PROTECTED:
            checked(ROOT/name, digest)
            for path, original in read(ROOT/name)[key].items():
                checked(ROOT/path, original)
        checked(execution/"completed.json", expected_completed)
        result = {"status": "completed", "scope": "engineering-only saved-output arithmetic review", "case_count": 1,
            "seed": 410, "scientific_gate": None, "scientific_qualification": "not assessed", "rows": rows,
            "sources_verified": 14, "protected_sources": bound["protected_sources"],
            "execution_completed_sha256": bound["completed"]["execution_completed_sha256"],
            "outer_completed_sha256": expected_completed,
            "row_receipts": bound["completed"]["rows"], "audit_receipts": bound["completed"]["audits"],
            "all_payload_sha256": bound["all_payload_sha256"], "members_verified": bound["members"],
            "work_totals": {key: sum(row["work"][key] for row in rows.values()) for key in rows["zero"]["work"]},
            "execution_wall_seconds": bound["execution"]["wall_seconds"],
            "audit_wall_seconds": bound["completed"]["audit_wall_seconds"],
            "whole_case_wall_seconds": bound["completed"]["total_wall_seconds"],
            "reviewer_sha256": sha(__file__), "review_wall_seconds": time.perf_counter()-tick,
            "new_native_calls": 0, "new_model_calls": 0, "new_rng_draws": 0, "limits": LIMITS}
        (out/"control-results-review.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
        lines = ["# One engineering case: saved-output review", "", "No scientific qualification gate was evaluated.", "",
                 "Native cost per issued action; lower is better. PRE [0,80), POST [80,200), MOVE [100,150).", "",
                 "| Role | Total | PRE | POST | MOVE |", "|---|---:|---:|---:|---:|"]
        for arm, row in rows.items():
            lines.append("| "+arm+" | "+" | ".join(f"{row['native_cost'][w]['mean_per_action']:.8f}" for w in WINDOWS)+" |")
        lines += ["", "Identifier mean absolute error uses the estimate available at the decision root, before its action.", "",
                  "| Role | Total | PRE | POST | MOVE |", "|---|---:|---:|---:|---:|"]
        for arm in ("adaptive", "frozen"):
            values = rows[arm]["identification"]["root_current_gain_absolute_error"]
            lines.append("| "+arm+" | "+" | ".join(f"{values[w]['mean_per_action']:.8f}" for w in WINDOWS)+" |")
        lines += ["", "Post-observation errors against the just-completed transition's gain are reported separately in JSON.",
                  "", *["- "+limit for limit in LIMITS], "", "Outer completion SHA: `"+expected_completed+"`."]
        (out/"results_review.md").write_text("\n".join(lines)+"\n")
        receipt = {"status": "completed", "engineering": True, "scientific_gate": None,
                   "outer_completed_sha256": expected_completed, "reviewer_sha256": sha(__file__),
                   "files": {name: sha(out/name) for name in ("control-results-review.json", "results_review.md")}}
        (out/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        return receipt
    except BaseException as error:
        try:
            (out/"failed.json").write_text(json.dumps({"status": "failed", "engineering": True,
                "error": repr(error), "expected_completed_sha256": expected_completed,
                "reviewer_sha256": sha(__file__), "automatic_retry": False}, indent=2)+"\n")
        except BaseException as writing_error:  # noqa: BLE001 - preserve original validation failure.
            error.add_note("Failure receipt could not be written: "+repr(writing_error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--expected-completed-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.execution, args.expected_completed_sha256, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
