"""Render all saved target-precision fits; no models or environments are called."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.run / "receipt.json")["sha256"] != args.worker_sha256 or digest(args.audit)["sha256"] != args.audit_sha256:
        raise ValueError("Externally pinned worker and audit required")
    receipt, audit = read(args.run / "receipt.json"), read(args.audit)
    if receipt["status"] != "completed" or audit["status"] != "completed" or not audit["agreement"]:
        raise ValueError("Complete study and agreeing saved-record audit required")
    if digest(args.run / "summary.json") != receipt["files"]["summary.json"]:
        raise ValueError("Summary changed since worker closure")
    producer = audit["producer_inputs"]
    if (producer["worker"]["sha256"] != args.worker_sha256
            or producer["plan"]["sha256"] != receipt["plan_sha256"]):
        raise ValueError("Audit must agree with this worker and frozen plan")
    summary = read(args.run / "summary.json")
    if summary["version"] != "otto-target-precision-v1":
        raise ValueError("Expected frozen precision experiment")
    args.output.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    fig.subplots_adjust(left=.055, right=.985, bottom=.27, top=.77, wspace=.28)
    regimes = ("lambda3", "lambda4", "lambda5")
    metrics = (("found", "Source found (%)", 100),
               ("steps", "Capped moves (lower is better)", 1),
               ("controller_seconds", "Controller seconds per search", 1))
    colors = {"r16": "#2878A8", "r64": "#CA641C"}
    points = []
    for ax, (metric, label, multiplier) in zip(axes, metrics, strict=True):
        for family, color in colors.items():
            for seed, marker in zip((20101, 20102, 20103), ("o", "s", "^"), strict=True):
                arm = f"{family}@{seed}"
                ys = [summary["regimes"][r]["means"][arm][metric] * multiplier for r in regimes]
                ax.plot(range(3), ys, color=color, alpha=.48, linewidth=.8, marker=marker,
                        markersize=4, markerfacecolor="white")
                points.extend({"regime": r, "arm": arm, "metric": metric, "display_value": y}
                              for r, y in zip(regimes, ys, strict=True))
            means = [summary["regimes"][r]["family_means"][family][metric] * multiplier for r in regimes]
            ax.plot(range(3), means, color=color, linewidth=2.4, marker="D", markersize=5)
            points.extend({"regime": r, "arm": f"{family}_family_mean", "metric": metric, "display_value": y}
                          for r, y in zip(regimes, means, strict=True))
        reference = [summary["regimes"][r]["means"]["analytic_inbounds"][metric] * multiplier for r in regimes]
        ax.plot(range(3), reference, color="#222222", linewidth=1.8, linestyle="--", marker="x", markersize=6)
        points.extend({"regime": r, "arm": "analytic_inbounds", "metric": metric, "display_value": y}
                      for r, y in zip(regimes, reference, strict=True))
        ax.set_xticks(range(3), ["Length 3", "Length 4", "Length 5\n(unseen in training)"])
        ax.set_ylabel(label)
        ax.set_xlim(-.15, 2.15)
        ax.set_ylim(bottom=0)
        if metric == "found":
            ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.18)
    checks = [item for group in ("positive_control_checks", "competence_checks", "relative_checks")
              for item in summary[group]]
    count = sum(c["passes"] for c in checks)
    fig.suptitle("Does more precise supervision improve search control?", fontsize=17, y=.98)
    fig.text(.5, .885, f"558 training states | 6 matched fits | 504 fresh searches | "
             f"Continuation rule: {'PASS' if summary['pilot_continuation'] else 'FAIL'} ({count}/33)",
             ha="center", fontsize=11)
    handles = [Line2D([], [], color=colors[k], linewidth=2.4, marker="D", label=label)
               for k, label in (("r16", "16-sample targets, family mean"),
                                ("r64", "64-sample targets, family mean"))]
    handles.append(Line2D([], [], color="#222222", linestyle="--", marker="x", label="Analytic controller"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .11), ncol=3, frameon=False)
    fig.text(.5, .077, "Thin lines show every individual fit. All outcomes use the same initial-hit mixture weights.",
             ha="center", fontsize=9)
    fig.text(.5, .035, "Controller time includes features, filtering and allocated checkpoint loading. "
             "Training and label generation are reported separately. No architecture advantage is established.",
             ha="center", fontsize=8.7)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"target-precision.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    payload = {"source": digest(Path(__file__)), "summary": digest(args.run / "summary.json"),
               "worker": digest(args.run / "receipt.json"), "audit": digest(args.audit),
               "matplotlib_version": matplotlib.__version__, "points": points,
               "scope": "All six fits plus analytic reference; saved aggregates only; no new inference or fitting",
               "files": {p.name: digest(p) for p in args.output.iterdir()}}
    (args.output / "receipt.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "rendered", "points": len(points), "checks_passed": count}))


if __name__ == "__main__":
    main()
