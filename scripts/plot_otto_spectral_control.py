"""Render all autonomous outcomes and the preselected first-case replays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ARMS = ("full_bayes", "exact_log", "recent32", "recent32_hard", "dct16_neutral", "dct16_nearest")
LABELS = ("Full Bayes", "Exact additive log", "Recent 32", "Recent 32 + hard support", "DCT 16 / neutral", "DCT 16 / nearest")
COLORS = ("#3f7066", "#6b958e", "#7f7a87", "#aaa4af", "#407dab", "#d49332")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def render(args):
    run = args.run.resolve()
    if sha(run / "receipt.json") != args.receipt_sha256 or sha(args.terminal) != args.terminal_sha256:
        raise ValueError("External receipt/terminal pin mismatch")
    receipt, terminal = read(run / "receipt.json"), read(args.terminal)
    if receipt["status"] != "completed" or terminal["status"] != "completed" or terminal["returncode"] != 0:
        raise ValueError("Only a completed comparison can be illustrated")
    for name, witness in receipt["files"].items():
        path = run / name
        if sha(path) != witness["sha256"] or path.stat().st_size != witness["bytes"]:
            raise ValueError(f"Changed saved file: {name}")
    summary = read(run / "summary.json")
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    if len(episodes) != 1152 or summary["episodes"] != 1152:
        raise ValueError("Incomplete comparison")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=.17, right=.97, top=.8, bottom=.2, wspace=.5, hspace=.45)
    fig.text(.05, .95, "Does compact memory work when it chooses its own path?", fontsize=20, weight="bold")
    fig.text(.05, .905, "1,152 autonomous searches | Six fixed controllers | 96 paired cases in each sensing regime", fontsize=12)
    fig.text(.05, .865, f"Compact-control rule: {'PASS' if summary['compact_control_viable'] else 'FAIL'}"
             f"     Utility/compute advantage: {'PASS' if summary['utility_compute_advantage'] else 'FAIL'}"
             "     No trained architecture or biological-wiring claim", fontsize=10)
    plotted = {}
    for col, (cohort, title) in enumerate((("base", "Baseline sensing, lambda = 3"), ("shift", "Changed sensing, lambda = 4"))):
        means = summary["cohorts"][cohort]["means"]
        moves = [means[a]["capped_time"] for a in ARMS]
        cost = [means[a]["controller_seconds"] * 1000 for a in ARMS]
        found = [sum(e["found"] for e in episodes if e["cohort"] == cohort and e["arm"] == a) for a in ARMS]
        plotted[cohort] = {"mean_capped_moves": moves, "mean_controller_ms": cost, "successful_episodes": found}
        for row, values in enumerate((moves, cost)):
            ax = axes[row, col]
            y = np.arange(6)
            ax.barh(y, values, color=COLORS, height=.62)
            ax.set_yticks(y, LABELS)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.set_xlim(0, max(values) * 1.43)
            ax.set_title(title if row == 0 else "Complete controller computation", loc="left", weight="bold", pad=13)
            ax.set_xlabel("Mixture-weighted capped moves, lower is better" if row == 0 else "Mixture-weighted ms per whole episode")
            for i, value in enumerate(values):
                extra = f"  ({found[i]}/96 found)" if row == 0 else ""
                ax.text(value + max(values) * .018, i, f"{value:.2f}{extra}", va="center", fontsize=8)
    notes = [
        "Each environment uses its own initial-hit mixture. Success counts in parentheses are unweighted. Every failed search costs the full 2,188-move horizon.",
        "Controller cost includes actor setup, all updates, decoding, planning and allocated shared-model construction. Environment execution is measured separately.",
        "DCT state includes 256 coefficients and an exact support mask. Shared tables, dense workspaces and object overhead are additional to evolving-array counts.",
        "Both finite fills must pass in both environments. These are exploratory engineering rules, not statistical noninferiority or an architectural novelty result.",
    ]
    for i, line in enumerate(notes):
        fig.text(.05, .135 - .029 * i, line, fontsize=9, color="#475569")
    fig.savefig(output / "spectral-control.png", dpi=180)
    fig.savefig(output / "spectral-control.pdf")
    plt.close(fig)

    # Fixed before execution in the protocol: case zero under each regime.
    selected = {("base", 630001), ("shift", 640001)}
    paths = {}
    with (run / "transitions.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if (r["cohort"], r["seed"]) not in selected:
                continue
            key = (r["cohort"], r["arm"])
            if r["kind"] == "reset":
                paths[key] = {"positions": [r["public"]["position"]], "source": r["source_evaluation_only"]}
            else:
                paths[key]["positions"].append(r["public"]["position"])
    if len(paths) != 12:
        raise ValueError("Missing first-case replay arm")
    for e in episodes:
        if (e["cohort"], e["seed"]) in selected:
            p = paths[e["cohort"], e["arm"]]
            if len(p["positions"]) != e["steps"] + 1:
                raise ValueError("First-case trajectory count mismatch")
            p.update(found=e["found"], steps=e["steps"])
    max_moves = max(p["steps"] for p in paths.values())
    stride = max(1, math.ceil(max_moves / 150))
    frame_moves = list(range(0, max_moves + 1, stride))
    if frame_moves[-1] != max_moves:
        frame_moves.append(max_moves)
    fig, axes = plt.subplots(2, 6, figsize=(15, 6.2), dpi=100)
    fig.subplots_adjust(left=.025, right=.99, bottom=.09, top=.83, hspace=.43, wspace=.17)
    fig.suptitle("First preselected case in each regime: all six controllers", fontsize=15, weight="bold", y=.97)
    subtitle = fig.text(.5, .91, "", ha="center", fontsize=10)
    fig.text(.5, .025, "Saved trajectories; source stars are shown to viewers only. Animation timing is illustrative.", ha="center", fontsize=9)
    artists = {}
    for row, cohort in enumerate(("base", "shift")):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            p = paths[cohort, arm]
            ax.set(xlim=(-1, 53), ylim=(53, -1), aspect="equal", xticks=[], yticks=[])
            ax.set_facecolor("#f1f5f9")
            source = p["source"]
            ax.scatter(source[1], source[0], marker="*", s=130, color="#a33c31", zorder=4)
            trail, = ax.plot([], [], color=COLORS[col], linewidth=1.3, alpha=.7)
            dot, = ax.plot([], [], "o", color=COLORS[col], markersize=6)
            title = ax.set_title("", fontsize=8)
            artists[cohort, arm] = trail, dot, title
    images = []
    for move in frame_moves:
        subtitle.set_text(f"Common move index {move} | Frames every {stride} moves | Baseline above, changed sensing below")
        for (cohort, arm), (trail, dot, title) in artists.items():
            p = paths[cohort, arm]
            upto = min(move, p["steps"])
            positions = np.asarray(p["positions"][:upto + 1])
            trail.set_data(positions[:, 1], positions[:, 0])
            dot.set_data([positions[-1, 1]], [positions[-1, 0]])
            status = "found" if move >= p["steps"] and p["found"] else "capped" if move >= p["steps"] else "searching"
            title.set_text(f"{LABELS[ARMS.index(arm)]}\nMove {upto}: {status}")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    images[0].save(output / "first-cases.gif", save_all=True, append_images=images[1:],
                   duration=[170] * (len(images) - 1) + [1500], loop=0, optimize=True)
    images[-1].save(output / "first-cases-final.png")
    (output / "plotted-values.json").write_text(json.dumps({"arms": ARMS, **plotted}, indent=2) + "\n")
    (output / "replay-receipt.json").write_text(json.dumps({"scope": "Preselected saved replay only, zero simulator calls",
        "cases": sorted(selected), "frame_move_indices": frame_moves, "stride": stride,
        "receipt_sha256": args.receipt_sha256, "terminal_sha256": args.terminal_sha256,
        "transitions_sha256": sha(run / "transitions.jsonl"), "source_markers": "viewer-only",
        "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in output.iterdir()}}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--terminal", type=Path, required=True)
    p.add_argument("--terminal-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    render(p.parse_args())
