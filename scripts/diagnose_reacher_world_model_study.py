"""Descriptive arithmetic over authenticated saved outputs; no new fits or calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(audit, expected, execution, out):
    receipt_path = audit / "receipt.json"
    if sha(receipt_path) != expected:
        raise ValueError("Unexpected audit receipt")
    receipt = json.loads(receipt_path.read_text())
    for name, digest in receipt["files"].items():
        if sha(audit / name) != digest:
            raise ValueError(f"Audit member changed: {name}")
    summary = json.loads((audit / "summary.json").read_text())
    inputs = {}

    def arrays(name):
        path = execution / name
        digest = sha(path)
        if digest != receipt["execution_members"][name]:
            raise ValueError(f"Execution member changed: {name}")
        inputs[name] = digest
        with np.load(path, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}

    physics = arrays("control/ordinary/known_state-planning.npz")["candidate_scores"][:, 0]
    rows = {}
    for name, fit in summary["fits"].items():
        epochs = fit["epochs"]
        records = arrays(f"control/ordinary/{name}.npz")
        planning = arrays(f"control/ordinary/{name}-planning.npz")
        choice = planning["candidate_scores"][:, 0].argmax(axis=-1)
        physics_selected = physics[np.arange(len(choice)), choice]
        rows[name] = {
            "epoch_9_to_12_relative_loss_decrease": 1 - epochs[-1]["loss"] / epochs[8]["loss"],
            "ordinary_zero_command_fraction": float(np.all(records["policy__commands"] == 0, axis=-1).mean()),
            "ordinary_positive_selected_reward_predictions": int((planning["selected_predicted_reward"] > 0).sum()),
            "initial_candidate_zero_selection_fraction": float((choice == 0).mean()),
            "initial_regret_under_saved_known_state_physics_scores": float((physics.max(axis=-1) - physics_selected).mean()),
        }
    result = {
        "audit_receipt_sha256": expected,
        "saved_output_only": True,
        "new_model_calls": 0,
        "new_physics_calls": 0,
        "new_fits": 0,
        "source_sha256": sha(Path(__file__)),
        "inputs": inputs,
        "fits": rows,
        "initial_physics_nonzero_choice_count": int((physics.argmax(axis=-1) != 0).sum()),
        "initial_physics_cases": len(physics),
        "initial_physics_mean_advantage_over_zero": float((physics.max(axis=-1) - physics[:, 0]).mean()),
        "limits": [
            "Post-hoc descriptive diagnosis; it does not change the frozen continuation gate.",
            "Training-loss decrease does not establish that more training improves control.",
            "Initial candidate banks and physical states are paired. Later trajectories differ, so no cross-policy score comparison is made there.",
            "Known-state physics sees the initial hidden velocities and has supplied dynamics. This is not an equal-information learned comparator.",
            "Candidate-score regret is relative to the saved finite action bank and nominal physics horizon, not optimal control or actual episode return.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.audit, args.expected_audit_receipt_sha256, args.execution, args.out)
