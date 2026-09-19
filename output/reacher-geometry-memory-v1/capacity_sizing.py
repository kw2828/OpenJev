"""Read-only capacity arithmetic, with no project, NumPy, Torch or native imports.

This is not a capacity benchmark or a scientific readiness check. Optional
input is a hash-authenticated COMPLETED engineering rehearsal, from which only
timings, file sizes and structural coverage are read. The companion design
specifies the future full-batch measurements needed before choosing run caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("residual_gru", "encoded_current_gru", "cached_gru", "cached_mlp")
PANELS = ("full", "ordinary", "shift")
REFERENCES = ("known_state", "particle", "public_kinematic", "zero", "uniform")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, relative, members):
    path = root / relative
    require(relative in members and path.is_file() and not path.is_symlink(), "Bound regular member required")
    require(sha(path) == members[relative], "Member hash mismatch: " + relative)
    return path


def exact_work():
    n, decisions, budget = 64, 50, 256
    horizon_histogram = {h: 1 if h < 12 else 39 for h in range(1, 13)}
    horizon_sum = sum(h * count for h, count in horizon_histogram.items())
    learned_rows, physics_rows, floor_rows = 36, 9, 6
    learned_sequences = learned_rows * n * decisions * budget
    physics_sequences = physics_rows * n * decisions * budget
    learned_transitions = learned_rows * n * budget * horizon_sum
    physics_transitions = physics_rows * n * budget * horizon_sum
    learned_selected, physics_selected = learned_rows * n * decisions, physics_rows * n * decisions
    # Physics: command8 + angle16 + qpos/qvel64 + completion booleans2
    # + substep int64 8 + nine geometry float32 values36 =134 per step.
    # Per sequence: initial qpos/qvel64 + score4 + initialized bool1 =69.
    # Learned candidate scoring: command8 + angle16 + learned4 + score4
    # + eight geometry float32 values32 =64 per step.
    payload = {
        "learned_candidate_scoring_arrays": 64 * learned_transitions,
        "physics_candidate_bank_arrays": 134 * physics_transitions + 69 * physics_sequences,
        "physics_selected_bank_arrays": 203 * physics_selected,
        "learned_root_and_selected_scoring_arrays": 80 * learned_selected,
        "physics_explicit_root_arrays": 72 * physics_selected,
    }
    return {
        "cases": n, "decisions_per_case": decisions,
        "horizon_histogram_per_row": horizon_histogram, "horizon_sum": horizon_sum,
        "learned_rows": learned_rows, "physics_rows": physics_rows, "floor_rows": floor_rows,
        "learned_candidate_sequences": learned_sequences,
        "physics_candidate_sequences": physics_sequences,
        "learned_candidate_transitions": learned_transitions,
        "physics_candidate_transitions": physics_transitions,
        "learned_selected_advances": learned_selected,
        "physics_selected_advances": physics_selected,
        "physics_candidate_native_substeps": 2 * physics_transitions,
        "physics_selected_native_substeps": 2 * physics_selected,
        "executed_native_control_transitions": (learned_rows + physics_rows + floor_rows) * n * decisions,
        "kinematic_nominal_observer_transitions": 3 * n * 49,
        "particle_nominal_observer_transitions": 3 * n * 49 * 32,
        "independent_nominal_replay_transitions": physics_transitions + physics_selected,
        "raw_array_payload_bytes": payload,
        "raw_array_payload_subtotal_bytes": sum(payload.values()),
        "raw_array_payload_subtotal_gib": sum(payload.values()) / 2**30,
        "payload_limit": "Uncompressed numeric payload subtotal only. Excludes CEM trace arrays, native episodes, observers, model snapshots, JSON/ZIP metadata, inherited copies, innovations, source snapshots and packaging; not a disk or peak-RAM upper bound.",
    }


def rehearsal_context(folder, expected_receipt):
    folder = Path(folder).resolve()
    receipt_path = folder / "audit" / "receipt.json"
    require(sha(receipt_path) == expected_receipt, "Externally supplied engineering audit SHA required")
    receipt = read(receipt_path)
    require(receipt["status"] == "completed" and receipt["engineering"] is True, "Completed engineering audit only")
    execution = folder / "execution"
    completed_path = execution / "completed.json"
    require(sha(completed_path) == receipt["execution_completed_sha256"], "Completion authentication")
    completed = read(completed_path)
    require(completed["status"] == "completed" and completed["control_rows"] == 51
            and completed["restored_models"] == 12, "Complete 51-row,12-model engineering rehearsal")
    plan_path = folder / "plan.json"
    require(sha(plan_path) == receipt["plan_sha256"] == completed["plan_sha256"], "Plan authentication")
    plan = read(plan_path)
    require(plan["engineering"] is True and plan["control_episodes"] == 1 and plan["steps"] == 50
            and plan["planning_horizon"] == 12 and plan["action_block"] == 3,
            "Specific full-horizon single-case rehearsal context")
    members = completed["files"]
    require(members == receipt["execution_members"], "Audit/completion member-set equality")
    expected = {f"control/{panel}/{arm}-pair{pair}/timings.json"
                for panel in PANELS for arm in ARMS for pair in range(3)}
    expected |= {f"control/{panel}/{name}/timings.json" for panel in PANELS for name in REFERENCES}
    require({name for name in members if name.endswith("/timings.json")} == expected,
            "Every declared row timing must be present without extra rows")
    rows = []
    for name in sorted(expected):
        timing = read(checked(execution, name, members))
        for key in ("row_wall_seconds", "setup_seconds"):
            require(isinstance(timing[key], (int, float)) and math.isfinite(timing[key]) and timing[key] >= 0,
                    "Finite engineering time")
        for key in ("decision_seconds", "native_step_seconds"):
            require(len(timing[key]) == 50 and all(math.isfinite(x) and x >= 0 for x in timing[key]),
                    "Complete nonnegative step times")
        prefix = name.removesuffix("timings.json")
        # These are size metadata only. Byte identities were already checked by
        # the supplied completed audit; this calculator reauthenticates timings.
        size = sum((execution / member).stat().st_size for member in members if member.startswith(prefix))
        rows.append({"path": prefix, "row_wall_seconds": timing["row_wall_seconds"],
                     "decision_seconds": sum(timing["decision_seconds"]),
                     "native_step_seconds": sum(timing["native_step_seconds"]),
                     "recorded_file_bytes_current_stat_only": size})
    audit_seconds = receipt["costs"]["audit_validation_wall_seconds"]
    return {"receipt_sha256": expected_receipt, "plan_sha256": receipt["plan_sha256"],
            "completion_sha256": receipt["execution_completed_sha256"],
            "execution_seconds": completed["wall_seconds"], "audit_seconds": audit_seconds,
            "naive_64x_audit_sensitivity_seconds": 64 * audit_seconds,
            "rows": rows, "new_model_calls": 0, "new_native_calls": 0,
            "interpretation": "Engineering timing context only. Tiny widths and N1 do not identify full-width N64 learned throughput. Multiplying audit by64 overcounts fixed overhead; it is a sensitivity check, not a chosen cap. No utility, gate or prediction-quality fields are read."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    require(bool(args.rehearsal) == bool(args.receipt_sha256), "Supply both rehearsal and external receipt SHA")
    result = {"status": "capacity_design_only", "exact_work": exact_work(),
              "capacity_profile_executed": False, "scientific_plan_created": False,
              "recommended_scientific_cap_seconds": None,
              "cap_reason": "Full-batch engineering execution and independently timed native replay are still required."}
    if args.rehearsal:
        result["authenticated_engineering_context"] = rehearsal_context(args.rehearsal, args.receipt_sha256)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
