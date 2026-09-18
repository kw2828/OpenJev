"""Render a descriptive figure from an authenticated completed audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "current_nominal": "Current angle, nominal gain",
    "velocity_nominal": "Two-angle velocity, nominal gain",
    "history_identified": "Full-history gain estimate",
    "recent3_identified": "Three-observation estimate",
    "known_state_physics": "Known-state reference",
    "uniform": "Uniform actions",
}
COLORS = {"stationary": "#126174", "switch": "#AE5C29"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(audit, expected_receipt, out):
    receipt_path = audit / "receipt.json"
    if digest(receipt_path) != expected_receipt:
        raise ValueError("Audit receipt differs from external identity")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "completed" or not receipt["saved_output_only"]:
        raise ValueError("A completed saved-output audit is required")
    for name, expected in receipt["files"].items():
        if digest(audit / name) != expected:
            raise ValueError(f"Audit artifact changed: {name}")
    summary = json.loads((audit / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    max_cost = max(max(a["costs"]) for panel in summary["panels"].values() for a in panel.values())

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#b6bec5",
        "axes.labelcolor": "#33404a", "text.color": "#1d2933",
        "xtick.color": "#52616c", "ytick.color": "#33404a",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4), gridspec_kw={"width_ratios": [1.2, 1]})
    for row, panel in enumerate(("stationary", "switch")):
        data, color = summary["panels"][panel], COLORS[panel]
        ax = axes[row, 0]
        for y, (arm, label) in enumerate(LABELS.items()):
            values = np.array(data[arm]["costs"])
            # Deterministic vertical spacing is purely graphical, not resampling.
            offsets = np.linspace(-.18, .18, len(values))
            ax.scatter(values, y + offsets, s=10, color=color, alpha=.2, linewidths=0)
            ax.scatter(data[arm]["mean_cost"], y, s=55, marker="D", color=color, zorder=3)
            ax.annotate(f"{data[arm]['mean_cost']:.1f}", (data[arm]["mean_cost"], y),
                        xytext=(7, -3), textcoords="offset points", fontsize=9, color=color)
        ax.set_yticks(range(len(LABELS)), list(LABELS.values()))
        ax.invert_yaxis()
        ax.set_xlim(-.015 * max_cost, 1.08 * max_cost)
        ax.set_xlabel("Native episode cost (lower is better)")
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        title = "Stationary hidden gain" if panel == "stationary" else "Unannounced gain switch at step 100"
        ax.set_title(title, loc="left", fontweight="bold", pad=14)

        ax = axes[row, 1]
        arms = ("velocity_nominal", "history_identified", "recent3_identified")
        for y, arm in enumerate(arms):
            mean = data[arm]["paired_mean_difference"]
            low, high = data[arm]["paired_difference_interval"]
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]],
                        color=color, fmt="D", markersize=6, capsize=4, linewidth=2)
        ax.axvline(0, color="#7d8a92", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
        ax.set_ylim(2.6, -.6)
        ax.set_xlim(-35, 35)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Paired cost difference\nfrom known-state reference")
        ax.grid(axis="x", alpha=.15)
        ax.set_title("History controls: 95% episode intervals", loc="left", fontsize=10, pad=14)

    fig.suptitle("Short history approaches reference mean control cost", x=.025, ha="left", y=.98,
                 fontsize=19, fontweight="bold")
    fig.text(.025, .925, "Supplied physics, no learned models. 32 paired cases per panel; identical MPC candidates and two initial probes.", fontsize=11)
    fig.text(.025, .025, "Small points: every episode. Diamonds: means. Intervals describe episodes, not training-seed uncertainty.\n"
             "Three-observation cost: +3.3% overall, but +19.1% in one gain cell. The mid-episode switch is a weak challenge here.\n"
             "All six aggregate qualification checks passed. No learned-model or biological-wiring advantage is established.", fontsize=10, linespacing=1.7)
    fig.subplots_adjust(left=.245, right=.98, bottom=.205, top=.86, wspace=.96, hspace=.6)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"pendulum-qualification.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    (out / "receipt.json").write_text(json.dumps({
        "audit_receipt_sha256": expected_receipt,
        "source_sha256": digest(Path(__file__)),
        "saved_output_only": True, "new_model_calls": 0,
        "files": {p.name: digest(p) for p in sorted(out.iterdir())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.audit, args.expected_audit_receipt_sha256, args.out)
