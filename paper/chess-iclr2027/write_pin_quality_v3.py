"""Generate manuscript tables from the authenticated, completed v3 figure data.

This separate publication helper preserves the original paper generator. It
does not import a study runner, neural library, checkpoint loader or engine.
The externally pinned audit and figure receipts bind all numeric inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/chess-pin-quality-v3"
AUDIT_SHA = "c86562f02e450ad67ed6af6225cf3ba0b13a958c2ae63a8746441dcbf9e9c52e"
FIGURE_SHA = "3e4f54de6afe9e5f043509b0bec8f2fa320d6ab55b843345a9a693552b4a248c"
COST_SHA = "80415131be41da01af1e8befa1e2263be29db4be229438d3e048c08a03cdc53b"
LABELS = {
    "base": "Frozen backbone", "wldn": "WLDN", "joint": "Joint pin",
    "separable": "Separable", "pairwise": "Pairwise", "root_only": "Root only",
    "counts": "Counts", "graph_mlp": "Graph MLP", "union:edits": "Union edits",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if Path(path).is_symlink() or sha(path) != expected:
        raise ValueError(f"Artifact identity mismatch: {path}")
    return json.loads(Path(path).read_text())


def authenticate():
    audit = checked(EVIDENCE / "audit/receipt.json", AUDIT_SHA)
    figures = checked(EVIDENCE / "figures/receipt.json", FIGURE_SHA)
    if audit["status"] != "completed" or figures["status"] != "completed":
        raise ValueError("Completed evidence required")
    if figures["audit_receipt_sha256"] != AUDIT_SHA or figures["engineering"]:
        raise ValueError("Actual v3 audit/figure binding required")
    for name, digest in figures["files"].items():
        if Path(name).name != name or sha(EVIDENCE / "figures" / name) != digest:
            raise ValueError(f"Figure member mismatch: {name}")
    plan = checked(EVIDENCE / "protocol/plan.json", audit["plan_sha256"])
    data = checked(EVIDENCE / "figures/figure-data.json", figures["files"]["figure-data.json"])
    if data["provenance"]["audit_receipt_sha256"] != AUDIT_SHA:
        raise ValueError("Numeric-data audit binding")
    if data["methods"] != list(LABELS) or data["seeds"] != [97, 109, 127]:
        raise ValueError("All nine methods and three paired seeds required")
    if len(data["fits"]) != 24 or len(data["metrics"]) != 27:
        raise ValueError("Full fit and frozen-backbone coverage required")
    if data["gate_checks"] != audit["gate_recomputed"]:
        raise ValueError("Exact audited gate checks required")
    if data["numerical"] != audit["numerical_gate_recomputed"]:
        raise ValueError("Exact audited numerical checks required")
    for panel in ("dev", "shift"):
        for arm in LABELS:
            for metric in ("agreement", "target_nll"):
                values = [data["metrics"][f"{arm}-{seed}"][panel][metric]
                          for seed in data["seeds"]]
                mean = statistics.mean(values)
                if not math.isfinite(mean) or not math.isclose(
                    mean, data["family_means"][panel][arm][metric], abs_tol=1e-14
                ):
                    raise ValueError("Family mean differs from retained fit metrics")
    return data, plan, figures


def authenticate_cost():
    folder = ROOT / "evidence/chess-pin-trained-cost-v2"
    execution = ROOT / "runs/chess-pin-trained-cost-v2/execution"
    receipt = checked(folder / "audit/receipt.json", COST_SHA)
    completed = checked(execution / "completed.json", receipt["execution_receipt_sha256"])
    summary = checked(execution / "summary.json", receipt["summary_sha256"])
    checked(folder / "protocol/plan.json", receipt["plan_sha256"])
    if receipt["status"] != "completed" or completed["status"] != "completed":
        raise ValueError("Completed cost study required")
    if completed["plan_sha256"] != receipt["plan_sha256"]:
        raise ValueError("Cost execution/plan identity")
    for name, digest in completed["files"].items():
        if Path(name).name != name or sha(execution / name) != digest:
            raise ValueError("Cost execution member identity")
    if receipt["quality_evidence"]["hashes"]["evidence/chess-pin-quality-v3/audit/receipt.json"] != AUDIT_SHA:
        raise ValueError("Cost study must use this quality study")
    if receipt["timing_records_checked"] != 31104 or receipt["warmup_records_checked"] != 54:
        raise ValueError("Complete timing and warmup coverage")
    if receipt["native_audit"]["records"] != 3456:
        raise ValueError("Complete native timing audit")
    if sha(folder / "audit/native-replay.jsonl") != receipt["native_replay_sha256"]:
        raise ValueError("Native timing replay identity")
    for check in (receipt["native_audit"], summary["timed_checks"], summary["warmup_checks"]):
        if check["failed_records"] or check["choice_changes"] or check["max_score_error"]:
            raise ValueError("The reported exact timing parity must hold")
    return receipt, summary


def paired_figure(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.4), sharex=True)
    markers = ("o", "s", "^")
    intervals = {(r["split"], r["comparator"]): r["percentile95"]
                 for r in data["conditional_game_bootstrap"]}
    for ax, panel, title in zip(axes, ("dev", "shift"), ("Ordinary development panel", "Shifted development panel"), strict=True):
        checks = [r for r in data["gate_checks"] if r["split"] == panel]
        for y, row in enumerate(checks):
            color = "#26765c" if row["passed"] else "#a7333c"
            interval = [100*v for v in intervals[(panel, row["comparator"])]]
            ax.plot(interval, (y, y), color=color, lw=2, alpha=.5)
            for offset, marker, value in zip((-.13, 0, .13), markers, row["paired_seed_gains"], strict=True):
                ax.scatter(100*value, y+offset, marker=marker, s=16, color=color, zorder=3)
            ax.scatter(100*row["mean_gain"], y, marker="D", s=17, color="black", zorder=4)
            ax.plot([100*row["required_mean_gain"]]*2, [y-.3,y+.3], color="#88939e", lw=1)
        ax.set_yticks(range(len(checks)), [LABELS[r["comparator"]] for r in checks])
        ax.set_ylim(7.6, -.6)
        ax.set_title(title, loc="left", pad=6)
        ax.axvline(0, color="#9ba5b0", lw=.8)
        ax.axvline(-.5, color="#9ba5b0", ls=":", lw=.8)
        ax.grid(axis="x", color="#e4e8ed", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlim(-2.5, 6.5)
    axes[-1].set_xlabel("Joint minus comparator agreement (percentage points)")
    handles = [Line2D([], [], marker=m, linestyle="none", color="#52606d", markersize=4, label=f"Seed {s}")
               for m,s in zip(markers,data["seeds"],strict=True)]
    handles.append(Line2D([], [], marker="D", linestyle="none", color="black", markersize=4, label="Mean"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.6, .998), ncol=4, frameon=False)
    fig.subplots_adjust(left=.22, right=.98, top=.93, bottom=.12, hspace=.26)
    fig.text(.22,.027,"Exposed development panels; all 16 checks retained.\nBars: saved conditional game-bootstrap intervals, not the gate.",fontsize=7)
    paths = []
    for suffix in ("pdf", "png"):
        path = PAPER / f"pin-quality-v3-comparisons.{suffix}"
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=180, metadata=metadata)
        paths.append(path.name)
    plt.close(fig)
    return paths


def generate():
    data, plan, figures = authenticate()
    cost_receipt, cost_summary = authenticate_cost()
    rows = [r"\begin{table}[ht]", r"\centering\small",
            r"\caption{Matched pin-quality v3. Means over all three seeds; agreement is",
            r"in percent and target NLL in nats. Each exposed development panel has",
            r"2,048 roots. All eight trained arms and all three frozen backbones are",
            r"retained; the designated joint arm fails its quality criterion.}",
            r"\label{tab:pin-quality-v3}", r"\begin{tabular}{lrrrr}", r"\toprule",
            r"& \multicolumn{2}{c}{Agreement (\%)} & \multicolumn{2}{c}{Target NLL} \\",
            r"Method & Ordinary & Shift & Ordinary & Shift \\", r"\midrule"]
    for arm, label in LABELS.items():
        ordinary, shift = (data["family_means"][p][arm] for p in ("dev", "shift"))
        rows.append(f"{label} & {100*ordinary['agreement']:.3f} & "
                    f"{100*shift['agreement']:.3f} & {ordinary['target_nll']:.6f} & "
                    f"{shift['target_nll']:.6f} " + r"\\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    checks = [r"\begin{table}[ht]", r"\centering\small",
              r"\caption{All sixteen original joint-minus-comparator quality checks.",
              r"Mean gain and worst paired-seed gain are percentage points. Every",
              r"check requires a paired-seed floor of $-0.5$ points; mean gain must",
              r"be at least zero against the backbone and $+1.0$ against trained controls.}",
              r"\label{tab:pin-quality-v3-checks}", r"\begin{tabular}{llrrc}",
              r"\toprule", r"Panel & Comparator & Mean gain & Worst seed & Check \\",
              r"\midrule"]
    for row in data["gate_checks"]:
        panel = "Ordinary" if row["split"] == "dev" else "Shift"
        checks.append(f"{panel} & {LABELS[row['comparator']]} & "
                      f"{100*row['mean_gain']:+.3f} & "
                      f"{100*row['minimum_paired_seed_gain']:+.3f} & "
                      f"{'Pass' if row['passed'] else 'Fail'} " + r"\\")
    checks.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    passed = sum(row["passed"] for row in data["gate_checks"])
    costs = data["costs"]
    macros = {
        "PinQualityChecksPassed": str(passed), "PinQualityChecksTotal": str(len(data["gate_checks"])),
        "PinQualityExecutionSeconds": f"{costs['execution_wall_seconds']:,.2f}",
        "PinQualityAuditSeconds": f"{costs['audit_wall_seconds']:,.2f}",
        "PinQualityFitSeconds": f"{costs['fit_wall_seconds_nested']:,.2f}",
        "PinQualityPriorLoggedSeconds": f"{costs['v2_logged_update_seconds_lower_bound']:,.2f}",
        "PinQualityPriorFitSeconds": f"{costs['v2_completed_fit_seconds_overlapping']:,.2f}",
        "PinCostJointMilliseconds": f"{cost_summary['timings']['median_complete_ms']['joint']:.3f}",
        "PinCostWLDNMilliseconds": f"{cost_summary['timings']['median_complete_ms']['wldn']:.3f}",
        "PinCostJointRatio": f"{cost_summary['timings']['median_paired_ratio_to_wldn']['joint']:.6f}",
        "PinCostJointPercent": f"{100*(cost_summary['timings']['median_paired_ratio_to_wldn']['joint']-1):.1f}",
    }
    outputs = {
        "pin-quality-v3-table.tex": "\n".join(rows) + "\n",
        "pin-quality-v3-checks.tex": "\n".join(checks) + "\n",
        "pin-quality-v3-results.tex": "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
        ) + "\n",
    }
    for name, value in outputs.items():
        (PAPER / name).write_text("% Generated by write_pin_quality_v3.py from hash-bound saved evidence.\n" + value)
    figure_paths = paired_figure(data)
    receipt = {
        "status": "completed", "audit_receipt_sha256": AUDIT_SHA, "figure_receipt_sha256": FIGURE_SHA,
        "plan_sha256": sha(EVIDENCE / "protocol/plan.json"),
        "figure_data_sha256": figures["files"]["figure-data.json"],
        "generator_sha256": sha(__file__), "methods": list(LABELS), "trained_fits": len(data["fits"]),
        "frozen_backbone_seeds": data["seeds"], "checks_passed": passed, "checks_total": 16,
        "continuation_passed": data["continuation_passed"],
        "saved_artifact_only": True, "new_model_calls": 0, "new_engine_calls": 0,
        "protocol_limits": plan["protocol"]["limits"],
        "cost_audit_receipt_sha256": COST_SHA,
        "cost_summary_sha256": cost_receipt["summary_sha256"],
        "cost_execution_receipt_sha256": cost_receipt["execution_receipt_sha256"],
        "cost_plan_sha256": cost_receipt["plan_sha256"],
        "outputs_sha256": {name: sha(PAPER / name) for name in (*outputs, *figure_paths)},
    }
    (PAPER / "pin-quality-v3-table-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"tables": 2, "methods": len(LABELS), "trained_fits": len(data["fits"]),
                      "checks_passed": passed, "checks_total": 16, "new_model_calls": 0}))


if __name__ == "__main__":
    generate()
