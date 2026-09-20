"""Plot every saved packed-pooling timing cell without model imports or calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    completed = args.run/"completed.json"
    assert sha(completed) == args.completion_sha256
    receipt = json.loads(completed.read_text())
    assert receipt["status"] == "completed" and receipt["version"] == "dialogue-token-packing-qualification-v1"
    for name, item in receipt["files"].items():
        p = args.run/name
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], name
    summary = json.loads((args.run/"summary.json").read_text())
    plan = json.loads((args.run/"plan.json").read_text())
    cells = summary["cells"]
    assert [c["case"] for c in cells] == [c["name"] for c in plan["recipe"]["cases"]]
    assert len(cells) == 16 and summary["optimizer_updates"] == 160
    assert summary["all_parity_passed"] and summary["parity_forward_backward_passes"] == 32
    passed = sum(c["speed_passed"] for c in cells)
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, 8.5), facecolor="white")
    ax.set_facecolor("white")
    ax.axvline(1., color="#738294", ls="--", lw=1)
    all_ratios = [x for c in cells for x in c["paired_ratios"]]
    lower, upper = max(0., min(all_ratios)-.12), max(all_ratios)+.3
    labels = []
    for i, cell in enumerate(cells):
        ratios = cell["paired_ratios"]
        assert len(ratios) == 4 and statistics.median(ratios) == cell["median_speed_ratio"]
        assert cell["speed_passed"] == (cell["median_speed_ratio"] >= cell["minimum_ratio"])
        color = "#137c88" if cell["speed_passed"] else "#b85332"
        if i in (4, 8, 12):
            ax.axhline(i-.5, color="#dce2e8", lw=1)
        ax.plot([min(ratios), max(ratios)], [i, i], color=color, alpha=.45, lw=1.8)
        ax.scatter(ratios, [i-.09, i-.03, i+.03, i+.09], color=color, alpha=.45, s=20)
        ax.scatter([cell["median_speed_ratio"]], [i], color=color, s=55, zorder=4)
        ax.scatter([cell["minimum_ratio"]], [i], color="#344050", marker="|", s=105, zorder=3)
        ax.text(upper-.025, i, f"{cell['median_speed_ratio']:.2f}x", va="center", ha="right", color=color, weight="bold")
        size, mode, head = cell["case"].split("-")
        labels.append(f"{'Real masks' if size == 'geometry' else size.capitalize()}  |  {mode}/{head}")
    ax.set_yticks(range(16), labels)
    ax.invert_yaxis()
    ax.set_xlim(min(lower, .85), upper)
    ax.set_xlabel("Original time / packed time: above 1 means faster")
    ax.xaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    outcome = "passed" if summary["engineering_admission"] else "failed"
    fig.suptitle(f"Mask-packed pooling: fixed speed qualification {outcome}", x=.03, ha="left", fontsize=17, weight="bold")
    fig.text(.03, .925, f"16 numerical-agreement checks pass  |  {passed}/16 speed cells pass  |  Artificial values and labels throughout", fontsize=11)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#137c88", markeredgecolor="#137c88", label="Median passes"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#b85332", markeredgecolor="#b85332", label="Median fails"),
               Line2D([0], [0], marker="|", color="none", markeredgecolor="#344050", markersize=10, label="Required ratio")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.64, .052), ncol=3, frameon=False)
    fig.text(.03, .024, "Small dots: all four pairs; lines: observed range, not confidence intervals. Real masks come from one preselected batch.", fontsize=9, color="#536477")
    fig.subplots_adjust(left=.30, right=.97, top=.88, bottom=.14)
    for suffix in ("png", "svg"):
        fig.savefig(args.out/f"timing.{suffix}", dpi=160)
    plt.close(fig)
    saved = {"status": "completed", "scope": "Saved timing visualization; no model or dataset calls",
             "sources": {str(Path(__file__)): sha(__file__), str(completed): args.completion_sha256},
             "outputs": {p.name: sha(p) for p in sorted(args.out.iterdir())}}
    (args.out/"receipt.json").write_text(json.dumps(saved, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"out": str(args.out), "engineering_admission": summary["engineering_admission"]}))


if __name__ == "__main__":
    main()
