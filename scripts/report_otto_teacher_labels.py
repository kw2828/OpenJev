"""Plot every paired cost from a completed independent saved-output audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def descriptor(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.audit / "receipt.json"
    if descriptor(receipt_path)["sha256"] != args.audit_sha256:
        raise ValueError("external saved-audit receipt identity")
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "completed" or receipt["agreement"] is not True or receipt["failures"]:
        raise ValueError("completed independent agreement required")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or descriptor(args.audit / name) != expected:
            raise ValueError("closed saved-audit payload")
    result = json.loads((args.audit / "audit.json").read_text())
    plan_desc = receipt["primary_files"]["plan"]
    plan_path = Path(plan_desc["path"])
    if descriptor(plan_path) != {k: plan_desc[k] for k in ("sha256", "bytes")}:
        raise ValueError("frozen plan identity")
    selected = json.loads(plan_path.read_text())["selected_anchors"]
    panels = result["panels"]
    args.output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows, centered, maximum = [], [], 0.
    for anchor, panel in zip(selected, panels, strict=True):
        actions = anchor["public"]["valid_actions"]
        costs = {(r["replicate_id"], r["first_action"]): r["steps"] for r in panel["records"]}
        values = []
        for replica in range(16):
            average = sum(costs[replica, action] for action in actions) / len(actions)
            values.append([costs[replica, action] - average for action in actions])
        maximum = max(maximum, max(abs(x) for row in values for x in row))
        centered.append(values)
        means = panel["reduction"]["actions"]
        ordered = sorted(means, key=lambda x: (x["mean_cost"], x["action"]))
        rows.append({"anchor_id": anchor["anchor_id"], "regime": anchor["regime"],
                     "prefix_index": anchor["prefix_index"], "actions": means,
                     "lowest_mean_action_ids": [x["action"] for x in ordered
                                                if x["mean_cost"] == ordered[0]["mean_cost"]],
                     "lowest_to_next_mean_gap": ordered[1]["mean_cost"] - ordered[0]["mean_cost"],
                     "maximum_continuation_moves": max(r["steps"] for r in panel["records"]),
                     "sampler_seconds_including_io": panel["sampler_seconds_including_io"]})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    limit = math.ceil(maximum * 1.08)
    for ax, anchor, panel, values in zip(axes.flat, selected, panels, centered, strict=True):
        actions = anchor["public"]["valid_actions"]
        for value in values:
            ax.plot(actions, value, color="#8495b6", linewidth=.8, alpha=.45,
                    marker="o", markersize=2.3)
        means = [r["centered_mean_cost"] for r in panel["reduction"]["actions"]]
        ax.plot(actions, means, color="#c34c18", linewidth=2.5, marker="o", markersize=5, zorder=4)
        ax.axhline(0, color="#444d5c", linestyle=":", linewidth=.8)
        ax.set_title(f"State {anchor['anchor_id']} | {anchor['regime']} | step {anchor['prefix_index']}",
                     loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks(actions, [f"Action {a}" for a in actions])
        ax.set_xlim(-.25, 3.25)
        ax.set_ylim(-limit, limit)
        ax.grid(axis="y", color="#e7eaf0", linewidth=.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Moves minus this replicate's\naverage across legal actions")
    fig.suptitle("Teacher continuation costs vary across paired source draws", x=.075,
                 ha="left", fontsize=16, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color="#8495b6", alpha=.7, label="Each of 16 paired replicates"),
                        Line2D([0], [0], color="#c34c18", linewidth=2.5, marker="o", label="Mean centered cost")],
               loc="upper left", bbox_to_anchor=(.066, .936), ncol=2, frameon=False)
    fig.text(.075, .022, f"Lower is better. {result['found']}/{result['continuations']} continuations found the source; horizon = 2,188 moves.\n"
             "Six fixed training states only. These are teacher targets, not learned-policy results or confidence intervals.",
             fontsize=9, color="#394151")
    fig.subplots_adjust(left=.075, right=.985, top=.83, bottom=.13, hspace=.34, wspace=.2)
    fig.savefig(args.output / "paired-costs.png", dpi=180, facecolor="white")
    fig.savefig(args.output / "paired-costs.svg", facecolor="white")
    plt.close(fig)
    summary = {"scope": "Descriptive plot of every audited paired cost, no new sampling or fitting",
               "source_sha256": descriptor(Path(__file__))["sha256"],
               "audit_receipt_sha256": args.audit_sha256, "states": rows,
               "maximum_continuation_moves": max(r["maximum_continuation_moves"] for r in rows),
               "sampler_seconds_total_including_io": sum(r["sampler_seconds_including_io"] for r in rows),
               "new_sampler_calls": 0, "new_model_calls": 0,
               "figure_files": {name: descriptor(args.output / name) for name in ("paired-costs.png", "paired-costs.svg")}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scope": summary["scope"]}))


if __name__ == "__main__":
    main()
