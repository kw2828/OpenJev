"""Plot a completed, independently audited alignment report; no fit-array reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
FITS = {f"{m}-{s}" for m in METHODS for s in SEEDS}
LABELS = {"flat_stratum": "Flat baseline", "token_mean": "Token mean", "token_aligned": "Token aligned"}
MARKERS, COLORS = ("o", "s", "^"), ("#3269a8", "#c47232", "#498364")
METRICS = ("accuracy", "wrong_selected_branch", "retained_error", "true_false_positive_rate", "dontcare_false_positive_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def bad(value): raise ValueError("Nonfinite JSON: "+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)


def manifest(root, receipt, terminal):
    require(set(receipt["files"]) | {terminal} == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()},
            "Exact completed report/audit closure")
    for name, item in receipt["files"].items():
        path = root/name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink()
                and path.resolve().is_relative_to(root.resolve()) and path.stat().st_size == item["bytes"]
                and sha(path) == item["sha256"], "Hash-bound report/audit payload")


def count(summary, method, seed, metric):
    c = summary["fits"][f"{method}-{seed}"]["decisions"][metric]
    n, d = c["numerator"], c["denominator"]
    require(type(n) is int and type(d) is int and 0 <= n <= d and d > 0, "Supported integer count")
    return n, d


def rate(summary, method, seed, metric):
    n, d = count(summary, method, seed, metric)
    return 100*n/d


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Finite nonnegative cost")
    return value


def authenticate(args):
    report, audit = Path(args.report).resolve(), Path(args.audit).resolve()
    require(sha(report/"summary.json") == args.summary_sha256 and sha(report/"receipt.json") == args.receipt_sha256,
            "External main-report pins")
    require(sha(audit/"receipt.json") == args.audit_receipt_sha256, "External independent-audit pin")
    rr, ar = read(report/"receipt.json"), read(audit/"receipt.json")
    require(rr["status"] == ar["status"] == "completed" and rr["technical_validity_passed"] is True
            and ar["agreement"] is True, "Completed report and independent agreement required")
    manifest(report, rr, "receipt.json"); manifest(audit, ar, "receipt.json")
    summary, checked = read(report/"summary.json"), read(audit/"summary.json")
    require(summary["status"] == checked["status"] == "completed"
            and summary["version"] == "dialogue-token-alignment-scientific-v1"
            and summary["technical_validity_passed"] is checked["technical_validity_passed"] is True
            and checked["agreement"] is True and summary["original_capacity_admitted"] is False,
            "Complete separate scientific campaign")
    request = ar["request"]
    require(request["report_receipt_sha256"] == args.receipt_sha256
            and request["completed_sha256"] == summary["execution_completed_sha256"] == rr["execution_completed_sha256"]
            and request["plan_sha256"] == summary["plan_sha256"] == rr["plan_sha256"], "Audit/report same execution")
    require(set(summary["fits"]) == set(checked["fits"]) == set(summary["costs"]["per_fit"]) == FITS, "All nine fits")
    rule = summary["continuation"]
    require(rule["total_checks"] == len(rule["checks"]) == 22 and all(type(c["passed"]) is bool for c in rule["checks"])
            and rule["checks_passed"] == sum(c["passed"] for c in rule["checks"])
            and rule["passed"] == all(c["passed"] for c in rule["checks"])
            and summary["continuation_allowed"] is rr["continuation_allowed"] is ar["continuation_passed"] is rule["passed"],
            "Frozen behavioral status")
    require(all(checked["continuation"][k] == rule[k] for k in ("passed", "checks_passed", "total_checks")), "Independent behavioral status")
    for c in rule["checks"]:
        other = checked["continuation"]["checks"][c["control"]+"/"+c["name"]]
        require(all(c[k] == v for k, v in other.items()), "Audited behavioral check")
    for method in METHODS:
        for seed in SEEDS:
            name = f"{method}-{seed}"
            require(summary["fits"][name]["decisions"] == checked["fits"][name]["decisions"], "Audited decision counts")
            require(summary["fits"][name]["cells"]["heldout_service/changed"]["rows"] == 578, "Fixed primary population")
            for metric in METRICS+("true_changed_recall", "dontcare_changed_recall"):
                count(summary, method, seed, metric)
    expected = {"accuracy": 578, "wrong_selected_branch": 578, "retained_error": 7241,
                "true_changed_recall": 29, "dontcare_changed_recall": 5}
    for metric in METRICS+("true_changed_recall", "dontcare_changed_recall"):
        denominators = {count(summary, m, s, metric)[1] for m in METHODS for s in SEEDS}
        require(len(denominators) == 1 and (metric not in expected or denominators == {expected[metric]}), "Common fixed support")
    whole, fit_walls = finite(summary["costs"]["whole_wall_seconds"]), []
    for cost in summary["costs"]["per_fit"].values():
        train, evaluation, wall = (finite(cost[k]) for k in ("training_wall_seconds", "evaluation_wall_seconds", "wall_seconds"))
        require(train > 0 and evaluation > 0 and train+evaluation <= wall <= whole, "Nested cost scope")
        fit_walls.append(wall)
    require(math.fsum(fit_walls) <= whole <= 7200, "Complete fixed-allocation cost")
    require(0 < finite(summary["costs"]["process_lifetime_peak_rss_bytes"]) <= 6*1024**3, "Recorded RSS")
    return summary


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#dce2e8", lw=.7)
    ax.set_axisbelow(True); ax.tick_params(labelsize=9)


def arm_panel(ax, summary, metric, title):
    labels = []
    for i, method in enumerate(METHODS):
        values = [rate(summary, method, seed, metric) for seed in SEEDS]
        for j, value in enumerate(values):
            ax.scatter(i+(j-1)*.14, value, marker=MARKERS[j], color=COLORS[j], s=52, zorder=3)
        ax.hlines(math.fsum(values)/3, i-.27, i+.27, color="#253645", lw=2)
        ns = [count(summary, method, seed, metric)[0] for seed in SEEDS]
        labels.append(LABELS[method]+"\n"+" | ".join(map(str, ns))+" / 578")
    ax.set_xticks(range(3), labels, fontsize=9); ax.set_xlim(-.5, 2.5); ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of 578 changed rows"); ax.set_title(title, loc="left")
    style(ax)


def difference_panel(ax, summary, harm=False):
    metrics = (("retained_error", "Retained error"), ("true_false_positive_rate", "TRUE false positive"),
               ("dontcare_false_positive_rate", "DONTCARE false positive")) if harm else (
               ("accuracy", "Accuracy gain"), ("wrong_selected_branch", "Wrong-branch reduction"))
    labels, all_values = [], []
    row = 0
    for control in METHODS[:2]:
        for metric, label in metrics:
            sign = -1 if metric == "wrong_selected_branch" else 1
            values = [sign*(rate(summary, "token_aligned", seed, metric)-rate(summary, control, seed, metric)) for seed in SEEDS]
            for j, value in enumerate(values):
                ax.scatter(value, row+(j-1)*.12, marker=MARKERS[j], color=COLORS[j], s=40, zorder=3)
            mean = math.fsum(values)/3
            ax.scatter(mean, row, marker="D", facecolors="white", edgecolors="#253645", s=54, linewidths=1.5, zorder=4)
            all_values.extend(values); all_values.append(mean)
            support = count(summary, control, SEEDS[0], metric)[1]
            labels.append(f"{label}\nvs {LABELS[control]}"+(f"; n={support:,}" if harm else ""))
            row += 1
        if control == METHODS[0]: ax.axhline(row-.5, color="#c6ced7", lw=1)
    threshold = .5 if harm else 2.
    ax.axvline(0, color="#798694", lw=.8)
    ax.axvline(threshold, color="#a45139", lw=1.5, ls="--")
    extent = [*all_values, 0., threshold]
    pad = max(.35, (max(extent)-min(extent))*.12)
    ax.set_xlim(min(extent)-pad, max(extent)+pad)
    ax.set_yticks(range(row), labels, fontsize=8); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Aligned minus control (percentage points)" if harm else "Decision improvement (percentage points)")
    ax.set_title("D  Harm limits: mean increase ≤0.5 pp" if harm else "C  Gains: mean ≥2 pp; every seed ≥0", loc="left")
    style(ax, "x")


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 13), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=(1., 1.3, .95), hspace=.65, wspace=.58)
    arm_panel(fig.add_subplot(grid[0, 0]), summary, "accuracy", "A  Changed-state accuracy: higher is better")
    arm_panel(fig.add_subplot(grid[0, 1]), summary, "wrong_selected_branch", "B  Wrong selected branch: lower is better")
    difference_panel(fig.add_subplot(grid[1, 0]), summary)
    difference_panel(fig.add_subplot(grid[1, 1]), summary, harm=True)
    ax = fig.add_subplot(grid[2, 0]); costs = summary["costs"]
    palette = ("#667f99", "#c28b53", "#bcc6cf")
    totals = []
    for i, method in enumerate(METHODS):
        arm = [costs["per_fit"][f"{method}-{s}"] for s in SEEDS]
        values = [math.fsum(c[k] for c in arm) for k in ("training_wall_seconds", "evaluation_wall_seconds", "wall_seconds")]
        parts = [values[0], values[1], values[2]-values[0]-values[1]]
        left = 0.
        for value, color in zip(parts, palette, strict=True):
            ax.barh(i, value/60, left=left/60, height=.57, color=color); left += value
        ax.text(left/60, i, f"  {left/60:.1f} min", va="center", fontsize=9)
        totals.append(values[2])
    shared = costs["whole_wall_seconds"]-math.fsum(totals)
    ax.barh(3, shared/60, height=.57, color="#d9dfe5")
    ax.text(shared/60, 3, f"  {shared:.1f} s", va="center", fontsize=9)
    ax.set_yticks(range(4), [LABELS[m]+" (3 fits)" for m in METHODS]+["Shared run work"])
    ax.invert_yaxis(); ax.set_xlim(0, max(*totals, shared)/60*1.27); ax.set_xlabel("Recorded minutes; nested scopes counted once")
    ax.set_title("E  Actual campaign compute", loc="left"); style(ax, "x")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(palette, ("Train + checkpoint", "Evaluation + save", "Other fit work"), strict=True)],
              loc="upper left", bbox_to_anchor=(0, -.26), ncol=2, frameon=False, fontsize=8)
    ax = fig.add_subplot(grid[2, 1]); ax.axis("off")
    rule = summary["continuation"]
    per_control = {c: sum(check["passed"] for check in rule["checks"] if check["control"] == c) for c in METHODS[:2]}
    lines = [f"Both controls required: {rule['checks_passed']}/22 checks passed",
             f"Against flat baseline: {per_control['flat_stratum']}/11",
             f"Against token mean: {per_control['token_mean']}/11", "",
             "Primary panel: 6 held-out services, 7,819 rows",
             "Changed: 578  |  Retained: 7,241",
             "TRUE changes: 29  |  DONTCARE changes: 5",
             "No FALSE changes or clears; sparse recalls descriptive.", "",
             f"Whole run: {costs['whole_wall_seconds']/60:.2f} min / 120 min cap",
             f"Peak RSS: {costs['process_lifetime_peak_rss_bytes']/1024**3:.2f} GiB (process lifetime)",
             f"Schema preparation: {costs['schema_preparation']['wall_seconds']:.2f} s, separate"]
    ax.text(0, 1, "F  Scope and fixed rule", fontsize=12, weight="bold", va="top")
    ax.text(0, .86, "\n".join(lines), fontsize=10, va="top", linespacing=1.35)
    status = "PASS" if rule["passed"] else "FAIL"
    fig.suptitle(f"Token alignment: behavioral continuation {status}", x=.065, y=.98, ha="left", fontsize=20, weight="bold")
    fig.text(.065, .947, "Historically exposed TRAIN development split; correct previous state supplied. All nine final fits and both controls shown.", fontsize=11)
    legend = [Line2D([0], [0], marker=MARKERS[j], color=COLORS[j], linestyle="none", label=str(seed)) for j, seed in enumerate(SEEDS)]
    legend += [Line2D([0], [0], color="#253645", lw=2, label="Three-seed mean in A/B"),
               Line2D([0], [0], marker="D", markerfacecolor="white", color="#253645", linestyle="none", label="Three-seed mean in C/D")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.06, .932), ncol=5, frameon=False, fontsize=9)
    fig.text(.065, .052, "A/B count labels follow seed order 6201 | 6202 | 6203. Branch is the branch of the selected candidate, not the largest summed branch mass.", fontsize=9, color="#526174")
    fig.text(.065, .033, "Three seeds repeat optimization, not independent service sampling. No novelty, calibration, deployment or fresh-confirmation claim.", fontsize=9, color="#526174")
    fig.text(.065, .014, "The earlier 48-minute capacity admission failed and remains closed; this campaign used a separately declared 120-minute allocation.", fontsize=9, color="#526174")
    fig.subplots_adjust(left=.13, right=.96, bottom=.15, top=.867)
    for suffix in ("png", "pdf"): fig.savefig(out/f"alignment.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def execute(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started, source_pin = time.monotonic(), sha(__file__)
    try:
        summary = authenticate(args)
        render(summary, out)
        require(sha(Path(args.report)/"summary.json") == args.summary_sha256
                and sha(Path(args.report)/"receipt.json") == args.receipt_sha256
                and sha(Path(args.audit)/"receipt.json") == args.audit_receipt_sha256
                and sha(__file__) == source_pin, "End input/source identity")
        record = {"status": "completed", "source_sha256": source_pin, "summary_sha256": args.summary_sha256,
                  "report_receipt_sha256": args.receipt_sha256, "audit_receipt_sha256": args.audit_receipt_sha256,
                  "execution_completed_sha256": summary["execution_completed_sha256"], "plan_sha256": summary["plan_sha256"],
                  "continuation_passed": summary["continuation_allowed"], "fits": 9, "checks": 22,
                  "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()},
                  "wall_seconds": time.monotonic()-started,
                  "scope": "Visualization of externally pinned completed report and agreeing independent audit. Reads no predictions, weights, feature arrays or corpus; no model/encoder calls. Actual cost witnesses are not a new timing benchmark."}
        with (out/"receipt.json").open("x") as stream: json.dump(record, stream, sort_keys=True, indent=2); stream.write("\n")
        return sha(out/"receipt.json")
    except BaseException as error:
        try:
            with (out/"failed.json").open("x") as stream:
                json.dump({"status": "failed", "error": repr(error), "source_sha256": source_pin,
                           "summary_sha256": args.summary_sha256, "model_calls": 0}, stream, indent=2)
        except BaseException as secondary:  # noqa: BLE001 - preserve the original error
            if callable(getattr(error, "add_note", None)): error.add_note("Failure receipt: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps({"receipt_sha256": execute(parser.parse_args())}))
