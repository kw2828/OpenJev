"""Plot completed, audited 80/320-epoch evidence without model or native calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-training-budget-render-v1"
SEEDS, KINDS = (30101, 30102, 30103), ("short", "long")
ARMS = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS) + ("analytic_inbounds",)
REGIMES = ("lambda3", "lambda4", "lambda5")
COLORS = {"short": "#2878A8", "long": "#CA641C"}
LABELS = {"short": "80 epochs", "long": "320 epochs"}
GROUPS = ("positive_control_checks", "competence_checks", "relative_checks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "regular nonsymlink file")
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def same(left, right):
    """Use the frozen saved-audit rounding tolerance, never for pass flags."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def authenticate(args, bound):
    def bind(path, expected=None, sha=None):
        actual = digest(path)
        require(expected is None or actual == expected, f"payload descriptor: {path.name}")
        require(sha is None or actual["sha256"] == sha, f"external pin: {path.name}")
        bound[str(path)] = actual
        return actual

    bind(args.plan, sha=args.plan_sha256)
    bind(args.run / "receipt.json", sha=args.worker_sha256)
    bind(args.terminal, sha=args.terminal_sha256)
    bind(args.audit, sha=args.audit_sha256)
    plan, worker, terminal, audit = (read(p) for p in (args.plan, args.run / "receipt.json", args.terminal, args.audit))
    require(plan["version"] == worker["version"] == "otto-training-budget-v1"
            and plan["status"] == "frozen_before_training" and worker["status"] == "completed"
            and worker["plan_sha256"] == args.plan_sha256, "completed fixed-budget producer")
    require(audit["version"] == "otto-training-budget-saved-audit-v1"
            and audit["status"] == "completed" and audit["agreement"] is True, "completed agreeing independent audit")
    require(audit["producer_inputs"] == {
        "plan": {"path": str(args.plan), "sha256": args.plan_sha256},
        "worker": {"path": str(args.run / "receipt.json"), "sha256": args.worker_sha256},
        "terminal": {"path": str(args.terminal), "sha256": args.terminal_sha256}}, "audit/worker/parent identity joins")
    require(worker["completed_fits"] == worker["completed_diagnostics"] == 6
            and worker["completed_episodes"] == 504 and worker["pending"] == []
            and worker["pending_episode"] is None and worker["sampler_calls"] == worker["new_label_calls"] == 0,
            "all prescribed phases complete")
    roots = {"started.json", "training-data.npz", "training-data.json", "training.jsonl", "fits.jsonl",
             "training-summary.json", "diagnostics.jsonl", "diagnostic-summary.json", "deployment.json",
             "evaluation.jsonl", "eval-transitions.jsonl.gz", "work.jsonl", "summary.json"}
    names = roots | {f"{kind}-{seed}-{phase}.npz" for kind in KINDS for seed in SEEDS
                     for phase in ("initial", "final", "diagnostics")} | {f"long-{seed}-prefix.npz" for seed in SEEDS}
    require(set(worker["files"]) == names and {p.name for p in args.run.iterdir()} == names | {"receipt.json"},
            "exact 34-payload worker inventory")
    for name in ("summary.json", "diagnostic-summary.json", "started.json"):
        bind(args.run / name, expected=worker["files"][name])
    require(set(audit["files"]) == {"started.json", "audit.json"}
            and {p.name for p in args.audit.parent.iterdir()} == {"started.json", "audit.json", "receipt.json"},
            "exact saved-audit inventory")
    for name, desc in audit["files"].items():
        bind(args.audit.parent / name, expected=desc)
    started = read(args.run / "started.json")
    launch_path = Path(started["request"]["supervision"])
    bind(launch_path, sha=worker["supervision_sha256"])
    launch = read(launch_path)
    require(started["launch"] == launch and all(terminal[k] == v for k, v in launch.items())
            and terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["timed_out"] is False and terminal["error"] is None and terminal["clock_error"] is None
            and terminal["group_absent"] is True and terminal["cleanup"]["errors"] == []
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and launch["started_ns"] <= worker["started_ns"] <= worker["finished_ns"]
            <= terminal["finished_ns"] < launch["deadline_ns"], "successful original supervised worker")
    require(sys.executable == plan["python_executable"] and importlib.metadata.version("matplotlib")
            == plan["all_distributions"]["matplotlib"], "qualified rendering runtime")
    summary, diagnostic, checked = (read(p) for p in
        (args.run / "summary.json", args.run / "diagnostic-summary.json", args.audit.parent / "audit.json"))
    require(summary["version"] == checked["producer_version"] == "otto-training-budget-v1"
            and checked["version"] == audit["version"], "audited summary versions")
    for key in ("regimes", *GROUPS, "pilot_continuation", "costs"):
        require(same(summary[key], checked[key]), f"displayed outcome agrees with independent audit: {key}")
    require(same(diagnostic, checked["train_diagnostics"]), "all final TRAIN diagnostics independently checked")
    require(summary["architecture_advantage_established"] is False and summary["episodes"] == 504
            and summary["paired_cases"] == 72 and set(summary["regimes"]) == set(REGIMES), "scope and coverage")
    for regime in REGIMES:
        require(set(summary["regimes"][regime]["means"]) == set(ARMS)
                and set(summary["regimes"][regime]["family_means"]) == set(KINDS), "every evaluation arm")
    fits = diagnostic["fits"]
    require([f["fit_id"] for f in fits] == list(ARMS[:-1]) and diagnostic["forwards"] == 240
            and diagnostic["view_rows"] == 26784 and diagnostic["checkpoint_restores"] == 6, "all fixed diagnostic fits")
    checks = [item for group in GROUPS for item in summary[group]]
    require([len(summary[g]) for g in GROUPS] == [3, 18, 12] and len({c["name"] for c in checks}) == 33
            and all(type(c["passes"]) is bool for c in checks), "all 33 distinct decisions")
    passed = sum(c["passes"] for c in checks)
    require(passed == audit["criteria_passed"] == checked["criteria_passed"] and audit["criteria_total"] == 33
            and summary["pilot_continuation"] == audit["pilot_continuation"] == (passed == 33), "unchanged overall rule")
    return summary, diagnostic, checks, passed


def save_figure(fig, output, stem):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.18)


def figures(summary, diagnostic, count, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    fig.subplots_adjust(left=.06, right=.985, bottom=.29, top=.77, wspace=.3)
    metrics = (("found", "Source found (%)", 100), ("steps", "Capped moves (lower is better)", 1),
               ("controller_seconds", "Controller seconds per search", 1))
    points = []
    for ax, (metric, label, multiplier) in zip(axes, metrics, strict=True):
        for family, color in COLORS.items():
            for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True):
                arm = f"{family}@{seed}"
                ys = [summary["regimes"][r]["means"][arm][metric] * multiplier for r in REGIMES]
                ax.plot(range(3), ys, color=color, alpha=.48, linewidth=.8, marker=marker,
                        markersize=4, markerfacecolor="white")
                points.extend({"regime": r, "arm": arm, "metric": metric, "display_value": y}
                              for r, y in zip(REGIMES, ys, strict=True))
            means = [summary["regimes"][r]["family_means"][family][metric] * multiplier for r in REGIMES]
            ax.plot(range(3), means, color=color, linewidth=2.4, marker="D", markersize=5)
            points.extend({"regime": r, "arm": f"{family}_family_mean", "metric": metric, "display_value": y}
                          for r, y in zip(REGIMES, means, strict=True))
        reference = [summary["regimes"][r]["means"]["analytic_inbounds"][metric] * multiplier for r in REGIMES]
        ax.plot(range(3), reference, color="#222222", linewidth=1.8, linestyle="--", marker="x", markersize=6)
        points.extend({"regime": r, "arm": "analytic_inbounds", "metric": metric, "display_value": y}
                      for r, y in zip(REGIMES, reference, strict=True))
        ax.set_xticks(range(3), ["Length 3", "Length 4", "Length 5\n(unseen in training)"])
        ax.set_ylabel(label)
        ax.set_xlim(-.15, 2.15)
        ax.set_ylim(bottom=0)
        if metric == "found":
            ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("Does longer fitting improve autonomous search?", fontsize=17, y=.98)
    fig.text(.5, .885, f"Identical R64 targets | 6 paired fits | 504 searches | "
             f"Frozen rule: {'PASS' if summary['pilot_continuation'] else 'FAIL'} ({count}/33)", ha="center", fontsize=11)
    handles = [Line2D([], [], color=COLORS[k], linewidth=2.4, marker="D", label=f"{LABELS[k]}, family mean") for k in KINDS]
    handles.append(Line2D([], [], color="#222222", linestyle="--", marker="x", label="Analytic controller"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .13), ncol=3, frameon=False)
    fig.text(.5, .095, "Thin lines retain all three fit seeds. Outcomes use the same initial-hit mixture weights; 72 paired cases.",
             ha="center", fontsize=9)
    fig.text(.5, .055, "Controller time includes features, filtering and allocated deployment loading. "
             "Fitting, TRAIN diagnostics and inherited labels are separate costs.", ha="center", fontsize=8.8)
    fig.text(.5, .015, "One CPU timing pass. No architecture advantage is established.", ha="center", fontsize=8.8)
    save_figure(fig, output, "training-budget")
    plt.close(fig)
    require(len(points) == 81, "all 81 evaluation points retained")

    rows = []
    for fit in diagnostic["fits"]:
        arm = fit["fit_id"]
        rows.append({"fit_id": arm, "epochs": 80 if arm.startswith("short@") else 320, **fit["metrics"],
                     "fit_seconds_with_pair_setup": summary["fit_costs"][arm]["total_training_seconds"],
                     "diagnostic_seconds": fit["seconds"], "diagnostic_restore_seconds": fit["restore_seconds"],
                     "diagnostic_prediction_seconds": fit["prediction_seconds"]})
    by_arm = {row["fit_id"]: row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    fig.subplots_adjust(left=.06, right=.985, bottom=.27, top=.76, wspace=.32)
    for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True):
        for kind in KINDS:
            row = by_arm[f"{kind}@{seed}"]
            axes[0].plot((0, 1), (row["single_view_weighted_mse"], row["eight_view_weighted_mse"]),
                         color=COLORS[kind], marker=marker, markersize=5, linewidth=.9, alpha=.75)
        for ax, key in zip(axes[1:], ("label_regret", "fit_seconds_with_pair_setup"), strict=True):
            ys = [by_arm[f"{kind}@{seed}"][key] for kind in KINDS]
            ax.plot((0, 1), ys, color="#999999", linewidth=.8, alpha=.6)
            for x, kind in enumerate(KINDS):
                ax.plot(x, ys[x], color=COLORS[kind], marker=marker, markersize=6)
    axes[0].set_xticks((0, 1), ("Deployed view", "Eight-view objective"))
    axes[0].set_ylabel("Final TRAIN centered MSE")
    axes[1].set_ylabel("Mean gap to best saved label (moves)")
    axes[2].set_ylabel("Fit seconds, including paired setup")
    for ax in axes[1:]:
        ax.set_xticks((0, 1), ("80 epochs", "320 epochs"))
    for ax in axes:
        ax.set_xlim(-.18, 1.18)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("Fixed-final TRAIN diagnostics and paid fitting cost", fontsize=17, y=.98)
    fig.text(.5, .88, "Every final checkpoint, every one of 558 TRAIN rows. No checkpoint selection.", ha="center", fontsize=11)
    legend = [Line2D([], [], color=COLORS[k], marker="o", label=LABELS[k]) for k in KINDS]
    legend += [Line2D([], [], color="#555555", marker=m, linestyle="none", label=f"Seed {seed}")
               for seed, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .13), ncol=5, frameon=False)
    fig.text(.5, .085, "MSE uses rounded training targets/weights with float64 reduction. "
             "Label gaps use original episode weights and sampled R64 costs.", ha="center", fontsize=8.8)
    fig.text(.5, .035, "These are in-sample fit diagnostics, not held-out accuracy or true environment regret. "
             "Diagnostic inference is charged separately from fitting and deployment.", ha="center", fontsize=8.8)
    save_figure(fig, output, "training-budget-train")
    plt.close(fig)
    with (output / "train-diagnostics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return points, rows, matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan-sha256", "worker-sha256", "terminal-sha256", "audit-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    require(all(getattr(args, name).is_absolute() for name in ("plan", "run", "terminal", "audit", "output")),
            "absolute evidence/output paths required")
    args.output.mkdir(exist_ok=False)
    bound, tick = {}, time.perf_counter()
    source = digest(Path(__file__).resolve())
    receipt = {"version": VERSION, "status": "started", "source": source,
               "request": {key: str(value) for key, value in vars(args).items()},
               "model_calls": 0, "training_calls": 0, "simulator_calls": 0}
    try:
        summary, diagnostic, checks, count = authenticate(args, bound)
        points, rows, mpl = figures(summary, diagnostic, count, args.output)
        write(args.output / "plotted-values.json", {"evaluation_points": points, "train_rows": rows, "conditions": checks,
              "costs": summary["costs"], "diagnostic_phase_seconds": diagnostic["seconds"],
              "fit_costs": summary["fit_costs"], "overall_passed": summary["pilot_continuation"],
              "scope": "All six final fits and analytic control; saved aggregates only. No model or trajectory replay."})
        require(all(digest(Path(name)) == desc for name, desc in bound.items())
                and digest(Path(__file__).resolve()) == source, "unchanged figure inputs and renderer")
        receipt.update(status="completed", inputs=bound, matplotlib_version=mpl, evaluation_points=81,
                       train_fits=6, criteria_passed=count, criteria_total=33,
                       files={p.name: digest(p) for p in args.output.iterdir()},
                       wall_seconds=time.perf_counter() - tick,
                       verification_scope="Externally pinned worker/parent/audit; displayed JSON payload hashes and agreement. "
                       "Full historical and numerical payload closure verification is inherited from the completed independent audit.")
        write(args.output / "receipt.json", receipt)
        print(json.dumps({"status": "completed", "receipt": digest(args.output / "receipt.json"), "evaluation_points": 81}))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error), inputs=bound, wall_seconds=time.perf_counter() - tick)
        try:
            target = args.output / "receipt.json"
            if target.exists():
                target.rename(args.output / "receipt.invalid.json")
            receipt["partial_files"] = {p.name: {"bytes": p.stat().st_size} for p in args.output.iterdir() if p.is_file()}
            write(target, receipt)
        except BaseException as secondary:  # noqa: BLE001 - keep original failure and partial artifacts
            error.add_note(f"Failed render receipt publication: {secondary!r}")
        raise


if __name__ == "__main__":
    main()
