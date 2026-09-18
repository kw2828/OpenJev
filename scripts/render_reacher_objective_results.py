"""Render authenticated saved Reacher objective audits without models or engines.

The caller supplies the externally authenticated audit receipt SHA-256. This
script reads only that receipt and its audit members, never execution traces,
checkpoints or training data. Plotted numbers are exported before rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

VERSION = "reacher-objective-ablation-v1"
ARMS = ("anchor", "raw", "latent")
PAIRS = ("pair0", "pair1", "pair2")
PANELS = ("full", "ordinary", "shift")
NAMES = {"anchor": "Anchor", "raw": "Raw endpoint", "latent": "EMA latent"}
PANEL_NAMES = {"full": "Full sensing", "ordinary": "Ordinary blackout", "shift": "Shifted blackout"}
COLORS = {"anchor": "#466784", "raw": "#CD7D33", "latent": "#278878"}
MARKERS = ("o", "s", "^")
FIGURES = ("native-costs", "objective-changes", "reset-penalties", "training-deployment-cost")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def finite(value, label, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value) and (not positive or value > 0), label)
    return float(value)


def load_audit(folder, expected_receipt_sha256):
    """Authenticate all audit members before trusting any plotted number."""
    folder = Path(folder)
    receipt_path = folder / "receipt.json"
    require(valid_hash(expected_receipt_sha256), "External audit receipt SHA-256 required")
    require(folder.is_dir() and not folder.is_symlink() and receipt_path.is_file()
            and not receipt_path.is_symlink(), "Regular audit directory and receipt required")
    require(sha(receipt_path) == expected_receipt_sha256, "Audit receipt SHA-256 mismatch")
    receipt = read(receipt_path)
    require(receipt.get("status") == "completed" and receipt.get("version") == VERSION
            and receipt.get("saved_output_only") is True and type(receipt.get("engineering")) is bool,
            "Completed saved-output objective audit required")
    members = receipt.get("files")
    require(isinstance(members, dict) and "summary.json" in members, "Authenticated summary member required")
    for name, digest in members.items():
        require(isinstance(name, str), "Audit member path must be a string")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and name == relative.as_posix()
                and all(part not in (".", "..") for part in relative.parts)
                and "\\" not in name and name != "receipt.json", "Unsafe audit member path")
        path = folder / name
        require(valid_hash(digest) and path.is_file() and not path.is_symlink()
                and all(not parent.is_symlink() for parent in path.parents if parent != folder.parent)
                and path.resolve().is_relative_to(folder.resolve()), "Regular contained audit member required")
        require(sha(path) == digest, f"Audit member SHA-256 mismatch: {name}")
    summary = read(folder / "summary.json")
    require(summary.get("status") == "completed" and summary.get("version") == VERSION
            and summary.get("engineering") is receipt["engineering"]
            and summary.get("saved_output_only") is True, "Summary and receipt state disagreement")
    for key in ("plan_sha256", "execution_completed_sha256"):
        require(valid_hash(receipt.get(key)) and summary.get(key) == receipt[key], f"Audit identity: {key}")
    require(summary.get("costs") == receipt.get("costs"), "Cost receipt binding")
    require(all(type(summary.get(key)) is int and summary[key] == 0
                for key in ("new_model_calls", "new_policy_calls", "new_fits")), "Audit must not score models")
    return summary, receipt


def figure_data(summary, receipt, receipt_sha256):
    """Recompute means and paired percentage changes from saved episode costs."""
    coverage = summary["coverage"]
    require(coverage["fits"] == 9 and coverage["control_rows"] == 57, "All nine fits and 57 rows required")
    count = coverage["control_episodes_per_row"]
    require(type(count) is int and count > 0, "Positive complete control cohort")
    names = {f"{arm}-{pair}" for arm in ARMS for pair in PAIRS}
    require(set(summary["fits"]) == names and set(summary["control"]) == set(PANELS), "Complete family/panel scope")
    controls, costs = {}, {}
    for panel in PANELS:
        rows = summary["control"][panel]
        expected = names | {"known_state", "particle", "zero", "uniform"}
        if panel != "full":
            expected |= {name + "-reset" for name in names}
        require(set(rows) == expected, f"Complete control rows: {panel}")
        controls[panel] = {}
        for name, row in rows.items():
            values = np.asarray(row["episode_costs"], dtype=float)
            require(values.shape == (count,) and np.isfinite(values).all() and (values > 0).all(),
                    "Finite positive native episode costs")
            mean = float(values.mean())
            require(math.isclose(mean, row["mean_cost"], rel_tol=1e-12, abs_tol=1e-12), "Native mean arithmetic")
            item = {"episode_costs": values.tolist(), "mean_cost": mean}
            if name in names:
                decision = finite(row["decision_wall_seconds"], "Decision wall", positive=True)
                steps = len(row["decision_seconds"])
                require(steps == 50 and math.isclose(sum(row["decision_seconds"]), decision,
                        rel_tol=1e-12, abs_tol=1e-7), "Deployment decision timing coverage/arithmetic")
                amortized = decision / (count * steps)
                require(math.isclose(amortized, row["per_case_amortized_seconds"],
                        rel_tol=1e-12, abs_tol=1e-12), "Amortized timing arithmetic")
                item.update(per_case_amortized_ms=1000 * amortized,
                            candidate_evaluations=row["candidate_evaluations"],
                            imagined_transitions=row["imagined_transitions"])
            controls[panel][name] = item
    for name in sorted(names):
        fit = summary["fits"][name]
        arm, pair = name.split("-")
        require(fit["name"] == name and fit["arm"] == arm and fit["pair"] == pair, "Paired fit identity")
        wall = finite(fit["wall_seconds"], "Whole fit wall", positive=True)
        training = finite(fit["training_seconds"], "Training wall", positive=True)
        setup = finite(fit["setup_seconds"], "Setup wall")
        require(setup >= 0 and setup + training <= wall + 1e-6, "Nested fit timing")
        costs[name] = {"whole_fit_seconds": wall, "training_seconds": training, "setup_seconds": setup,
                       "updates": fit["updates"], "ema_updates": fit["ema_updates"],
                       "trainable_parameters": fit["trainable_parameters"],
                       "student_parameters": fit["parameters"]}
    require(math.isclose(sum(row["whole_fit_seconds"] for row in costs.values()),
                         summary["costs"]["fit_wall_seconds"], rel_tol=1e-12, abs_tol=1e-7),
            "Whole training cost reconstruction")
    family, changes, latent_raw, resets = {}, {}, {}, {}
    for panel in PANELS:
        family[panel] = {arm: float(np.mean([controls[panel][f"{arm}-{pair}"]["mean_cost"]
                                            for pair in PAIRS])) for arm in ARMS}
        changes[panel] = {}
        for arm in ("raw", "latent"):
            changes[panel][arm] = {
                "per_pair_percent": {pair: 100 * (controls[panel][f"{arm}-{pair}"]["mean_cost"] /
                         controls[panel][f"anchor-{pair}"]["mean_cost"] - 1) for pair in PAIRS},
                "family_mean_percent": 100 * (family[panel][arm] / family[panel]["anchor"] - 1)}
        latent_raw[panel] = {
            "per_pair_percent": {pair: 100 * (controls[panel][f"latent-{pair}"]["mean_cost"] /
                                controls[panel][f"raw-{pair}"]["mean_cost"] - 1) for pair in PAIRS},
            "family_mean_percent": 100 * (family[panel]["latent"] / family[panel]["raw"] - 1)}
        if panel != "full":
            resets[panel] = {}
            for arm in ARMS:
                reset_mean = float(np.mean([controls[panel][f"{arm}-{pair}-reset"]["mean_cost"] for pair in PAIRS]))
                resets[panel][arm] = {
                    "per_pair_percent": {pair: 100 * (controls[panel][f"{arm}-{pair}-reset"]["mean_cost"] /
                                      controls[panel][f"{arm}-{pair}"]["mean_cost"] - 1) for pair in PAIRS},
                    "family_mean_percent": 100 * (reset_mean / family[panel][arm] - 1)}
    gate = summary["continuation_gate"]
    require(len(gate["checks"]) == 31 and type(gate["passed"]) is bool
            and gate["passed"] is all(row["passed"] for row in gate["checks"]), "Complete audited continuation gate")
    return {
        "version": VERSION, "engineering": summary["engineering"],
        "provenance": {"audit_receipt_sha256": receipt_sha256, "audit_members": receipt["files"],
                       "plan_sha256": receipt["plan_sha256"],
                       "execution_completed_sha256": receipt["execution_completed_sha256"],
                       "source_sha256": receipt["source_sha256"], "runtime": receipt["runtime"]},
        "coverage": coverage, "native_cost_definition": "Negative sum of 50 native step rewards; lower is better.",
        "control": controls, "family_mean_cost": family, "versus_anchor": changes,
        "latent_versus_raw": latent_raw,
        "reset_penalty": resets, "fit_costs": costs, "phase_costs": summary["costs"],
        "audited_paired_intervals": summary["paired_descriptive_comparisons"], "continuation_gate": gate,
        "limits": summary["limits"],
        "plot_notes": [
            "Shapes identify paired initializations, not independent environment replications.",
            "Black diamonds are ratios of family means for percentage plots; no error bars are fabricated.",
            "Audited confidence intervals are conditional on the three saved fits and shared training corpus.",
            "Reset differences measure intervention sensitivity, not useful memory versus a trained observation-only model.",
            "Deployment timing is amortized across the recorded batch and includes stored-input loading, assimilation, CEM, selected advance and trace saving. It excludes row setup and native stepping.",
            "Wall times were measured on a shared host and can include concurrent workloads; they are not isolated latency measurements.",
            "Whole-fit wall includes setup, learning and serialization. Shared calibration is shown separately and is not allocated to individual arms.",
            "Equal CEM candidate budget does not establish equal FLOPs, training work or latency."]}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "figure.titlesize": 15, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#BBC3CA", "axes.labelcolor": "#233340",
        "text.color": "#233340", "xtick.color": "#425360", "ytick.color": "#425360",
        "grid.color": "#DDE3E8", "grid.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "openjev-reacher-objective-v1", "savefig.facecolor": "white"})


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.75)
    ax.tick_params(axis="both", length=0, pad=7)


def finish(fig, data, title, note, out, name, *, arm_legend=False):
    fig.suptitle(title, x=.06, y=.98, ha="left", fontweight="bold")
    handles = [Line2D([], [], marker=marker, color="#596975", linestyle="none", markersize=6,
                      label=f"Paired fit {index + 1}") for index, marker in enumerate(MARKERS)]
    handles.append(Line2D([], [], marker="D", color="#1D2933", linestyle="none", markersize=5,
                          label="Family mean"))
    if arm_legend:
        handles += [Line2D([], [], marker="o", color=COLORS[arm], linestyle="none", label=NAMES[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .905), frameon=False,
               ncol=len(handles), fontsize=8.5, handletextpad=.45, columnspacing=1.25)
    fig.text(.06, .11, note, fontsize=8.5, va="top", linespacing=1.45)
    footer = ("ENGINEERING FIXTURE: NOT RESEARCH RESULTS" if data["engineering"] else
              "OpenJev | Three paired fits, one shared training corpus | Saved-output audit")
    fig.text(.06, .014, footer, fontsize=9, weight="bold" if data["engineering"] else "normal",
             color="#A44238" if data["engineering"] else "#697780")
    fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.23, wspace=.26)
    for extension in ("png", "svg", "pdf"):
        metadata = {"Creator": "OpenJev authenticated saved-audit renderer"}
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(out / f"{name}.{extension}", dpi=220, metadata=metadata)
    plt.close(fig)


def paired_points(ax, arms, values, means, *, names=None, colors=None):
    names, colors = NAMES if names is None else names, COLORS if colors is None else colors
    x = np.arange(len(arms))
    for index, pair in enumerate(PAIRS):
        jitter = (index - 1) * .06
        ax.plot(x + jitter, [values[arm][pair] for arm in arms], color="#B6C0C8", linewidth=.7, zorder=1)
        for position, arm in zip(x, arms, strict=True):
            ax.scatter(position + jitter, values[arm][pair], marker=MARKERS[index], s=47,
                       color=colors[arm], edgecolor="white", linewidth=.5, zorder=3)
    ax.scatter(x, [means[arm] for arm in arms], marker="D", s=26, color="#1D2933", zorder=4)
    ax.set_xticks(x, [names[arm] for arm in arms])
    ax.set_xlim(-.4, len(arms) - .6)
    axes_style(ax)


def render_native(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.3), sharey=True)
    for panel, ax in zip(PANELS, axes, strict=True):
        rows = data["control"][panel]
        values = {arm: {pair: rows[f"{arm}-{pair}"]["mean_cost"] for pair in PAIRS} for arm in ARMS}
        paired_points(ax, ARMS, values, data["family_mean_cost"][panel])
        ax.axhline(rows["known_state"]["mean_cost"], color="#727E88", linestyle="--", linewidth=.9)
        ax.axhline(rows["zero"]["mean_cost"], color="#727E88", linestyle=":", linewidth=1.1)
        ax.set_title(PANEL_NAMES[panel], pad=14)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean native episode cost (lower is better)")
    finish(fig, data, "All nine fits under the same CEM256 controller",
           f"Matched cases per fit: {data['coverage']['control_episodes_per_row']}; 50 steps per episode. "
           "Lines connect paired initializations.\nDashed: known-state physics reference. Dotted: zero-action reference. "
           "All reference values remain in figure-data.json.", out, "native-costs")


def render_change(data, out, *, reset=False):
    panels = ("ordinary", "shift") if reset else PANELS
    arms = ARMS if reset else ("raw", "latent", "latent_raw")
    fig, axes = plt.subplots(1, len(panels), figsize=(12.6, 5.3), sharey=True)
    key = "reset_penalty" if reset else "versus_anchor"
    for panel, ax in zip(panels, axes, strict=True):
        rows = dict(data[key][panel])
        names, colors = NAMES, COLORS
        if not reset:
            rows["latent_raw"] = data["latent_versus_raw"][panel]
            names = {"raw": "Raw vs\nanchor", "latent": "Latent vs\nanchor", "latent_raw": "Latent vs\nraw"}
            colors = {**COLORS, "latent_raw": COLORS["latent"]}
        paired_points(ax, arms, {arm: rows[arm]["per_pair_percent"] for arm in arms},
                      {arm: rows[arm]["family_mean_percent"] for arm in arms}, names=names, colors=colors)
        ax.axhline(0, color="#647681", linewidth=.9)
        if panel != "full":
            ax.axhline(5 if reset else -5, color="#929BA3", linestyle=":", linewidth=1)
        ax.set_title(PANEL_NAMES[panel], pad=14)
    axes[0].set_ylabel("Native cost change (%)" + (" after memory reset" if reset else " versus named comparator"))
    if reset:
        title = "Reset sensitivity while preserving the last visible measurement"
        note = ("Positive values mean reset increased cost. Dotted: the latent family's +5% continuation threshold.\n"
                "Sensitivity alone does not establish useful memory against a trained observation-only model.")
    else:
        title = "Objective changes against both prespecified comparators"
        note = ("Negative values mean lower cost. Diamonds use the ratio of family means; shapes use paired-fit ratios.\n"
                "Dotted: the latent family's -5% threshold on blackout panels. Full sensing and raw changes are descriptive.")
    finish(fig, data, title, note, out, "reset-penalties" if reset else "objective-changes")


def render_cost(data, out):
    fig, (train, deploy) = plt.subplots(1, 2, figsize=(12.6, 5.5))
    values = {arm: {pair: data["fit_costs"][f"{arm}-{pair}"]["whole_fit_seconds"] for pair in PAIRS} for arm in ARMS}
    paired_points(train, ARMS, values, {arm: float(np.mean(list(values[arm].values()))) for arm in ARMS})
    train.set_title("Whole training run per fit", pad=14)
    train.set_ylabel("Wall time (seconds)")
    train.set_ylim(bottom=0)
    for position, panel in enumerate(PANELS):
        for offset, arm in enumerate(ARMS):
            x = position + (offset - 1) * .22
            times = []
            for index, pair in enumerate(PAIRS):
                value = data["control"][panel][f"{arm}-{pair}"]["per_case_amortized_ms"]
                times.append(value)
                deploy.scatter(x + (index - 1) * .035, value, marker=MARKERS[index], s=35,
                               color=COLORS[arm], edgecolor="white", linewidth=.4)
            deploy.scatter(x, np.mean(times), color="#1D2933", marker="D", s=19)
    deploy.set_xticks(np.arange(3), ["Full", "Ordinary", "Shift"])
    deploy.set_title("Intact controller decision time", pad=14)
    deploy.set_ylabel("Amortized milliseconds per case per decision")
    deploy.set_xlim(-.45, 2.45)
    deploy.set_ylim(bottom=0)
    axes_style(deploy)
    calibration = data["phase_costs"]["calibration_seconds"]
    finish(fig, data, "Training work and measured deployment timing",
           f"Shared calibration: {calibration:.3f} s, charged separately. "
           "Decision timing includes trace storage; excludes row setup and native stepping.\n"
           "Wall times come from a shared host. Batch amortization and equal candidate budgets do not imply isolated latency or equal compute.",
           out, "training-deployment-cost", arm_legend=True)


def report(data):
    engineering = data["engineering"]
    lines = ["# Reacher objective ablation", "",
             ("**Engineering fixture only. These numbers test reporting and cannot support research claims.**" if engineering
              else "Results below come from the authenticated completed saved-output audit."), "",
             "All nine fits are retained. Every learned controller uses the same CEM256 configuration.", "",
             "| Panel | Anchor | Raw endpoint | EMA latent | Physics reference | Zero action | Particle | Uniform |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for panel in PANELS:
        numbers = [data["family_mean_cost"][panel][arm] for arm in ARMS]
        numbers += [data["control"][panel][name]["mean_cost"] for name in ("known_state", "zero", "particle", "uniform")]
        lines.append(f"| {PANEL_NAMES[panel]} | " + " | ".join(f"{value:.6f}" for value in numbers) + " |")
    lines += ["", "Native cost is negative total native reward across 50 steps; lower is better.", ""]
    lines += ["| Panel and contrast | Family mean change | Paired fit 1 | Paired fit 2 | Paired fit 3 |",
              "|---|---:|---:|---:|---:|"]
    for panel in PANELS:
        contrasts = [("Raw vs anchor", data["versus_anchor"][panel]["raw"]),
                     ("Latent vs anchor", data["versus_anchor"][panel]["latent"]),
                     ("Latent vs raw", data["latent_versus_raw"][panel])]
        for label, row in contrasts:
            values = [row["family_mean_percent"], *[row["per_pair_percent"][pair] for pair in PAIRS]]
            lines.append(f"| {PANEL_NAMES[panel]}: {label} | " + " | ".join(f"{value:+.3f}%" for value in values) + " |")
    lines += ["", "Negative changes favor the first objective. The latent continuation rule requires improvement against both anchor and raw on ordinary and shifted blackouts; raw gains alone do not establish a latent-objective gain.", ""]
    for name, alt in zip(FIGURES, ("All nine fit costs", "Paired objective changes", "Memory reset penalties",
                                 "Training and deployment costs"), strict=True):
        lines += [f"![{alt}]({name}.png)", ""]
    gate = data["continuation_gate"]
    passed = sum(row["passed"] for row in gate["checks"])
    lines += [f"Recorded continuation checks: {passed}/31. " +
              ("This engineering gate is a software check, not an efficacy result." if engineering else
               f"Frozen gate: {'PASS' if gate['passed'] else 'FAIL'}."), "",
              "| Audited paired difference | Mean native cost change | Conditional 95% interval |", "|---|---:|---|" ]
    for panel, rows in data["audited_paired_intervals"].items():
        for name, row in rows.items():
            low, high = row["episode_paired_percentile_95"]
            lines.append(f"| {panel}: {name.replace('_', ' ')} | {row['mean_cost_difference']:.6f} | [{low:.6f}, {high:.6f}] |")
    lines += ["", "Intervals condition on these three paired fits and the shared corpus; they do not estimate uncertainty over retraining.",
              "", "All plotted values, episode costs, costs and provenance are in [figure-data.json](figure-data.json).", ""]
    lines += [f"- {note}" for note in data["plot_notes"]]
    lines += ["", "Audit limitations:", ""] + [f"- {limit}" for limit in data["limits"]]
    lines += ["", f"Audit receipt SHA-256: `{data['provenance']['audit_receipt_sha256']}`.", ""]
    return "\n".join(lines)


def render(folder, expected_receipt_sha256, out):
    summary, receipt = load_audit(folder, expected_receipt_sha256)
    data = figure_data(summary, receipt, expected_receipt_sha256)
    out = Path(out)
    require(not out.exists(), "Exclusive figure output directory required")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write(out / "figure-data.json", data)
        style()
        render_native(data, out)
        render_change(data, out)
        render_change(data, out, reset=True)
        render_cost(data, out)
        (out / "README.md").write_text(report(data))
        again, checked_receipt = load_audit(folder, expected_receipt_sha256)
        require(again == summary and checked_receipt == receipt, "Audit changed during rendering")
        members = {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}
        output_receipt = {"status": "completed", "version": VERSION, "engineering": data["engineering"],
                          "audit_directory": str(Path(folder).resolve()),
                          "audit_receipt_sha256": expected_receipt_sha256,
                          "audit_members": receipt["files"], "plan_sha256": receipt["plan_sha256"],
                          "renderer_sha256": sha(Path(__file__)), "saved_audit_only": True,
                          "new_model_calls": 0, "new_engine_calls": 0,
                          "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                                      "numpy": np.__version__}, "files": members}
        write(out / "receipt.json", output_receipt)
        return output_receipt
    except BaseException as error:
        plt.close("all")
        write(out / "failed.json", {"status": "failed", "audit_receipt_sha256": expected_receipt_sha256,
                                    "error": repr(error), "saved_audit_only": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.audit, args.expected_audit_receipt_sha256, args.out)
    print(json.dumps({"status": result["status"], "engineering": result["engineering"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
