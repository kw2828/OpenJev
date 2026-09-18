"""Saved-output figures and a schematic replay of the first scheduled case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

FAMILIES = (
    ("history", "History model", "#52697f"),
    ("gru", "GRU", "#167385"),
    ("gru-reset", "GRU, reset at gap onset", "#75acb6"),
    ("rssm", "RSSM", "#754aa0"),
    ("rssm-reset", "RSSM, reset at gap onset", "#ae94ca"),
    ("known_state", "Known-state physics", "#c47729"),
    ("particle", "Particle-filter physics", "#d9aa6b"),
    ("zero", "Zero command", "#71787d"),
    ("uniform", "Uniform commands", "#a4aaae"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def members(family, controls):
    if family in controls:
        return [family]
    kind = family.split("-")[0]
    reset = family.endswith("-reset")
    return [name for name in controls if name.startswith(kind + "-") and name.endswith("-reset") == reset]


def render(audit, expected_receipt, execution, out):
    if sha(audit / "receipt.json") != expected_receipt:
        raise ValueError("External audit identity mismatch")
    receipt = json.loads((audit / "receipt.json").read_text())
    if receipt["status"] != "completed" or not receipt["saved_output_only"]:
        raise ValueError("Completed saved-output audit required")
    for name, digest in receipt["files"].items():
        if sha(audit / name) != digest:
            raise ValueError("Audit artifact changed")
    summary = json.loads((audit / "summary.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "text.color": "#20303b"})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6), sharey=True)
    for ax, panel, label in zip(axes, ("full", "ordinary", "shift"),
                               ("Full angle sensing", "Six-step sensor gaps", "Ten-step sensor gaps"), strict=True):
        controls = summary["control"][panel]
        for row, (family, _, color) in enumerate(FAMILIES):
            names = members(family, controls)
            if not names:
                continue
            costs = [controls[name]["mean_cost"] for name in names]
            ax.plot([min(costs), max(costs)], [row, row], color=color, linewidth=2)
            ax.scatter(costs, row + np.linspace(-.11, .11, len(costs)) if len(costs) > 1 else [row],
                       color=color, s=36, zorder=3)
            ax.annotate(f"{np.mean(costs):.2f}", (max(costs), row),
                        xytext=(6, -3), textcoords="offset points", fontsize=9, color=color)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=16)
        largest = max(item["mean_cost"] for item in controls.values())
        ax.set_xlim(0, largest * 1.22)
        ax.set_xlabel("Mean episode cost (lower is better)")
        ax.set_yticks(range(len(FAMILIES)), [label for _, label, _ in FAMILIES])
        ax.grid(axis="x", alpha=.16)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    statuses = summary["continuation_gate"]["kinds"]
    gate_text = ", ".join(f"{kind.upper()}: {'PASS' if value['passed'] else 'FAIL'}" for kind, value in statuses.items())
    fig.suptitle("Learned world models under missing observations", x=.025, ha="left", y=.98,
                 fontsize=19, fontweight="bold")
    fig.text(.025, .91, "Every final fit is shown. Frozen continuation rules: " + gate_text, fontsize=11)
    fig.text(.025, .055, "Each learned dot: one fit averaged over the same 32 episodes. Lines span fits; they are not confidence intervals.\n"
             "Physics references have supplied dynamics. Different parameter counts and compute; no biological topology tested.", fontsize=10, linespacing=1.65)
    fig.subplots_adjust(left=.22, right=.97, top=.81, bottom=.20, wspace=.28)
    fig.savefig(out / "reacher-world-models.png", dpi=180, facecolor="white")
    plt.close(fig)

    # Authenticate only the exact raw files used for this scheduled replay.
    seeds = sorted(int(name.split("-")[1]) for name in summary["control"]["ordinary"]
                   if name.startswith("history-"))
    first_seed = seeds[0]
    traces = []
    for kind in ("history", "gru", "rssm"):
        relative = f"control/ordinary/{kind}-{first_seed}.npz"
        if sha(execution / relative) != receipt["execution_members"][relative]:
            raise ValueError("Replay trajectory changed")
        with np.load(execution / relative, allow_pickle=False) as data:
            traces.append({"qpos": data["audit__qpos"][0], "packets": data["policy__packets"][0],
                           "rewards": data["audit__rewards"][0]})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6))
    artists = []
    for ax, kind, trace in zip(axes, ("History", "GRU", "RSSM"), traces, strict=True):
        ax.set_aspect("equal")
        ax.set_xlim(-.24, .24)
        ax.set_ylim(-.24, .24)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(*trace["qpos"][0, 2:4], color="#c54737", s=70, marker="x", linewidth=2)
        ax.set_title(f"{kind}, fit {first_seed}", fontsize=11, fontweight="bold")
        arm, = ax.plot([], [], "o-", linewidth=4, markersize=7, color="#167385")
        status = ax.text(.02, .03, "", transform=ax.transAxes, fontsize=9)
        artists.append((arm, status))
    heading = fig.suptitle("First scheduled ordinary case", fontsize=13)
    fig.text(.025, .025, "Recorded true positions; policies had masked sensors. Red cross: target. Playback is 2x slower than simulation.", fontsize=8)
    fig.subplots_adjust(left=.025, right=.985, bottom=.12, top=.84, wspace=.08)

    def update(t):
        heading.set_text(f"First scheduled ordinary case | {t * .02:.2f} simulation seconds")
        for ax, trace, (arm, status) in zip(axes, traces, artists, strict=True):
            q0, q1 = trace["qpos"][t, :2]
            x = [0, .1 * np.cos(q0), .1 * np.cos(q0) + .11 * np.cos(q0 + q1)]
            y = [0, .1 * np.sin(q0), .1 * np.sin(q0) + .11 * np.sin(q0 + q1)]
            arm.set_data(x, y)
            visible = bool(trace["packets"][t, 6])
            ax.set_facecolor("white" if visible else "#e6e9eb")
            status.set_text(f"{'Sensor visible' if visible else 'Sensor missing'}\nCost so far: {-sum(trace['rewards'][:t]):.2f}")
        return [heading, *[item for pair in artists for item in pair]]

    animation = FuncAnimation(fig, update, frames=len(traces[0]["qpos"]), interval=40, blit=False)
    animation.save(out / "reacher-first-case.gif", writer=PillowWriter(fps=25), dpi=110)
    update(12)
    fig.savefig(out / "reacher-first-case.png", dpi=150, facecolor="white")
    plt.close(fig)
    (out / "receipt.json").write_text(json.dumps({
        "audit_receipt_sha256": expected_receipt, "source_sha256": sha(__file__),
        "saved_output_only": True, "new_policy_calls": 0, "new_model_calls": 0,
        "replay_selection": {"panel": "ordinary", "fit_seed": first_seed, "case_index": 0},
        "files": {p.name: sha(p) for p in sorted(out.iterdir())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.audit, args.expected_audit_receipt_sha256, args.execution, args.out)
