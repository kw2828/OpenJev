"""Plot only preserved execution counts, never model predictions or task metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    failure_path = args.execution / "failed.json"
    plan_path = args.execution / "plan.json"
    failure = json.loads(failure_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert failure["status"] == "failed" and failure["error_type"] == "TimeoutError"
    assert failure["resume_authorized"] is False
    arms = ("slot_readout", "slot_scalar", "candidate_readout", "candidate_scalar")
    names = [f"{a}-{s}" for s in (4101, 4102, 4103) for a in arms]
    complete = {f"{x['method']}-{x['seed']}": x["updates"] for x in failure["completed_fits"]}
    partial = failure["active_progress"]
    target = plan["optimizer_updates_per_fit"]
    assert list(complete) == names[:7] and partial["fit"] == names[7]
    assert target == 1280 and all(x == target for x in complete.values())
    counts = [complete.get(n, partial["completed_optimizer_steps"] if n == partial["fit"] else 0)
              for n in names]
    args.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")
    ax.set_facecolor("white")
    colors = ["#147d92" if n in complete else "#d8872a" if n == partial["fit"] else "#dce2e8" for n in names]
    ax.barh(range(12), [target]*12, color="#edf0f4", height=.67)
    ax.barh(range(12), counts, color=colors, height=.67)
    ax.set_yticks(range(12), [n.replace("_", "/").replace("-", "  |  ") for n in names])
    ax.invert_yaxis()
    for i, value in enumerate(counts):
        label = "Complete: 1,280" if value == target else f"Partial: {value}" if value else "Not started"
        ax.text(value-16 if value == target else value+16, i, label,
                va="center", ha="right" if value == target else "left",
                color="white" if value == target else "#263648", fontsize=9)
    ax.set_xlim(0, target+160)
    ax.set_xlabel("Completed optimizer updates per fit (planned: 1,280)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, alpha=.14)
    ax.set_axisbelow(True)
    fig.suptitle("Shared-token dialogue study stopped at its fixed time limit", x=.035, ha="left", fontsize=16, weight="bold")
    fig.text(.035, .91, f"7 of 12 fits complete  |  {sum(counts):,} of {target*12:,} updates  |  {failure['wall_seconds']:,.2f} seconds", fontsize=11)
    ax.legend(handles=[Patch(color="#147d92", label="Completed fit"), Patch(color="#d8872a", label="Preserved partial fit")],
              loc="lower right", frameon=False, fontsize=9)
    fig.text(.035, .025, "Execution evidence only. No accuracy comparison. Partial work retained; no retry or resume.", fontsize=9, color="#536477")
    fig.subplots_adjust(left=.28, right=.98, top=.86, bottom=.115)
    for suffix in ("png", "svg"):
        fig.savefig(args.out/f"execution.{suffix}", dpi=160)
    plt.close(fig)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"status": "completed", "scope": "Saved execution counts only; no predictions read or model calls",
               "sources": {str(p): digest(p) for p in (Path(__file__), failure_path, plan_path)},
               "outputs": {p.name: digest(p) for p in sorted(args.out.iterdir())}, "counts": dict(zip(names, counts, strict=True))}
    (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"out": str(args.out), "completed_updates": sum(counts)}))


if __name__ == "__main__":
    main()
