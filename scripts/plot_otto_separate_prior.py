"""Display every separate-prior fit from closed, independently audited JSON.

No scientific producer, model, teacher, NPZ or trajectory is loaded. The pinned
older plotter supplies only saved-file guards, JSON comparison and CSV writing.
The completed audit supplies scientific verification. This is presentation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-separate-prior-plot-v1"
BASE_PATH = "scripts/plot_otto_prequery_calibration.py"
BASE_PIN = "69c3542f693e8bdceb474da0beb8880b81a8614360107119fcc97173f9653403"
AUDITOR = "scripts/audit_otto_separate_prior.py"
AUDIT_VERSION = "otto-separate-prior-saved-audit-v1"
TRAIN_VERSION = "otto-separate-prior-training-v1"
KINDS = ("innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse", "innovation_separate_aux",
         "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux")
SEEDS = (285000001, 285000002, 285000003)
REGIMES = ("lambda3", "lambda4")
METRICS = (("initial", "episode_weighted_agreement", "Initial agreement (%) | steps 1-3", 100.),
           ("full", "episode_weighted_agreement", "Full nonquery agreement (%)", 100.),
           ("postcorrection", "episode_weighted_raw_gap", "Primary raw teacher gap | steps >= 5", 1.),
           ("prior", "episode_weighted_centered_mse", "Raw prior-score MSE | queries >= 4", 1.))
GATES = {"innovation_mechanism": ("Explicit mechanism", 19),
         "gru_mechanism": ("GRU mechanism", 19), "architecture": ("Architecture", 29)}
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2,
          "metadata_json_bytes": 16 * 1024**2, "result_json_bytes": 256 * 1024**2}
COLORS = ("#8ba8d2", "#426dac", "#61a0dc", "#125e9b", "#99bda9", "#547d66", "#55ac93", "#08795f")
LABELS = tuple(("Explicit" if k.startswith("innovation") else "GRU") + "\n"
               + k.split("_")[-2] + "/" + k.split("_")[-1].upper() for k in KINDS)


def _base():
    path = ROOT / BASE_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_PIN:
        raise ValueError("unchanged saved-only plotting helpers")
    spec = importlib.util.spec_from_file_location("_separate_prior_plot_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VERSION, module.LIMITS = VERSION, LIMITS
    return module


base = _base()
require, same, write, write_csv = base.require, base.same, base.write, base.write_csv


def gates(rows):
    """Validate and display named memberships; do not recompute scientific tests."""
    names = {"common.technical_complete"}
    for regime in REGIMES:
        names.add(f"common.{regime}.initial.case_support")
        names.update(f"common.{regime}.age{age}.case_support" for age in (1, 2, 3))
        for architecture in ("innovation", "gru"):
            names.update(f"mechanism.{architecture}.{regime}.{suffix}" for suffix in (
                "full.agreement_recovery", "initial.agreement_recovery", "full.gap", "postcorrection.gap", "prior.mse"))
        names.update(f"architecture.{regime}.{suffix}" for suffix in (
            "postcorrection.agreement", "postcorrection.gap", "initial.agreement", "full.agreement", "full.gap"))
    require(len(rows) == len(names) == 39 and {r["name"] for r in rows} == names
            and all(type(r["passes"]) is bool for r in rows), "all 39 named condition records")
    common = [r for r in rows if r["name"].startswith("common.")]
    mechanism = {a: [r for r in rows if r["name"].startswith(f"mechanism.{a}.")] for a in ("innovation", "gru")}
    memberships = {"innovation_mechanism": common + mechanism["innovation"],
                   "gru_mechanism": common + mechanism["gru"],
                   "architecture": common + mechanism["innovation"] + [r for r in rows if r["name"].startswith("architecture.")]}
    result = {name: {"conditions": [r["name"] for r in selected], "passed": sum(r["passes"] for r in selected),
                     "total": len(selected), "passes": all(r["passes"] for r in selected)}
              for name, selected in memberships.items()}
    require(all(result[k]["total"] == count for k, (_, count) in GATES.items()), "19/19/29 named decisions")
    return result


def tables(summary, audit):
    expected = [(kind, seed) for seed in SEEDS for kind in KINDS]
    require([(r["family"], r["seed"]) for r in summary["models"]] == expected
            and [(r["family"], r["seed"]) for r in audit["fits"]] == expected, "all 24 final fits in fixed order")
    points, means = [], []
    for record in [*summary["models"], {"family": "hold", "seed": None, "metrics": summary["hold"]}]:
        require(record["metrics"]["primary_mask"] == "nonquery and absolute_step >= 5", "fixed primary domain")
        for regime in REGIMES:
            for scope, metric, _, scale in METRICS:
                if scope == "prior" and record["family"] == "hold":
                    continue
                group = (record["prior_metrics"] if scope == "prior" else record["metrics"][scope])["by_regime"][regime]
                value = group[metric]
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite displayed value")
                require(metric != "episode_weighted_agreement" or value <= 1, "bounded agreement")
                require(group["episodes"] == 18 and group["declared_case_count"] == 6, "all VALID cases and collectors")
                points.append({"family": record["family"], "seed": record["seed"], "regime": regime,
                    "scope": scope, "metric": metric, "raw_value": value, "display_value": value * scale,
                    "episodes": group["episodes"], "supported_episodes": group["supported_episodes"],
                    "supported_cases": group["supported_case_count"], "weight_mass": group["weight_mass"]})
    for kind in KINDS:
        for regime in REGIMES:
            for scope, metric, _, scale in METRICS:
                selected = [p for p in points if (p["family"], p["regime"], p["scope"]) == (kind, regime, scope)]
                require(len(selected) == 3 and {p["seed"] for p in selected} == set(SEEDS), "no seed selection")
                value = math.fsum(p["raw_value"] for p in selected) / 3
                means.append({"family": kind, "regime": regime, "scope": scope, "metric": metric,
                              "seed_count": 3, "raw_value": value, "display_value": value * scale})
    require(len(points) == 198 and len(means) == 64, "complete all-fit and hold display coverage")
    by = {(m["family"], m["seed"]): m for m in summary["models"]}
    costs = []
    for fit in audit["fits"]:
        item = {k: fit[k] for k in ("family", "seed", "architecture", "readout", "objective", "parameter_count",
            "epochs", "steps", "fit_seconds", "wall_seconds", "train_rescore_seconds", "checkpoint_seconds",
            "forward_chunks", "backward_chunks", "no_grad_chunks")}
        require(item["epochs"] == 80 and item["steps"] == 720, "equal update exposure")
        for key in ("fit_seconds", "wall_seconds", "train_rescore_seconds", "checkpoint_seconds"):
            require(type(item[key]) in (int, float) and math.isfinite(item[key]) and item[key] >= 0, "finite saved cost")
        work = by[fit["family"], fit["seed"]]["prediction_work"]
        require(set(work) == {"forward_chunks", "forward_rows", "prior_rows"}
                and all(type(v) is int and v >= 0 for v in work.values())
                and work["forward_chunks"] > 0 and work["forward_rows"] > 0,
                "saved VALID work counts, no per-fit timer")
        item.update({"valid_" + k: v for k, v in work.items()})
        costs.append(item)
    return points, means, costs


class Plot(base.Plot):
    def read(self, path):
        limit = LIMITS["result_json_bytes"] if path in (self.args.summary, self.args.audit / "audit.json") else LIMITS["metadata_json_bytes"]
        require(str(path) in self.bound and self.bound[str(path)]["bytes"] <= limit, "authenticated bounded JSON")
        return json.loads(Path(path).read_text())

    def authenticate(self):
        args = self.args
        audit_path = args.audit / "receipt.json"
        self.bind(args.summary, args.summary_sha256)
        self.bind(audit_path, args.audit_sha256)
        self.bind(args.audit_terminal, args.audit_terminal_sha256)
        receipt, terminal = self.read(audit_path), self.read(args.audit_terminal)
        require(receipt["version"] == AUDIT_VERSION and receipt["status"] == "completed"
                and receipt["agreement"] is True and not receipt["failures"]
                and receipt["requires_successful_original_supervisor"] is True, "completed agreeing independent audit")
        require(set(receipt["files"]) == {"started.json", "audit.json"}
                and {p.name for p in args.audit.iterdir()} == {"started.json", "audit.json", "receipt.json"}, "closed audit inventory")
        for name, pin in receipt["files"].items():
            self.bind(args.audit / name, pin)
        for name, pin in receipt["sources"].items():
            self.bind(ROOT / name, pin)
        require(AUDITOR in receipt["sources"] and set(receipt["producer_inputs"]) == {"plan", "worker", "terminal"},
                "original auditor and complete producer joins")
        for pin in receipt["producer_inputs"].values():
            self.bind(Path(pin["path"]), pin)
        require(receipt["plan_sha256"] == receipt["producer_inputs"]["plan"]["sha256"], "frozen training plan")
        worker_path = Path(receipt["producer_inputs"]["worker"]["path"])
        worker = self.read(worker_path)
        require(worker["version"] == TRAIN_VERSION and worker["status"] == "completed" and worker["complete"] is True
                and worker["fits_completed"] == 24 and args.summary == worker_path.parent / "summary.json",
                "same completely fitted producer summary")
        self.bind(args.summary, worker["files"]["summary.json"])
        require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
                and terminal["error"] is terminal["clock_error"] is None
                and terminal["group_absent"] is terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == []
                and terminal["cap_seconds"] == 240 and terminal["clock_source_sha256"] == base.CLOCK_PIN
                and terminal["watchdog_sha256"] == base.SUPERVISOR_PIN and terminal["cwd"] == str(ROOT)
                and terminal["deadline_ns"] == terminal["started_ns"] + 240 * 10**9
                and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
                <= terminal["finished_ns"] <= terminal["deadline_ns"], "successful original audit process")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command[:2] == [str(ROOT / ".venv/bin/python"), str(ROOT / AUDITOR)]
                and len(command[2:]) == 16 and len(set(command[2::2])) == 8, "actual original audit command")
        options = dict(zip(command[2::2], command[3::2], strict=True))
        expected = {"--output": str(args.audit)}
        for role, pin in receipt["producer_inputs"].items():
            expected[f"--{role}"], expected[f"--{role}-sha256"] = pin["path"], pin["sha256"]
        require(set(options) == set(expected) | {"--supervision"}
                and all(options[k] == v for k, v in expected.items()), "audit command joins")
        launch_path = Path(options["--supervision"])
        self.bind(launch_path, receipt["supervision_sha256"])
        launch, started = self.read(launch_path), self.read(args.audit / "started.json")
        require(started["launch"] == launch and all(terminal[k] == v for k, v in launch.items()), "original launch retained")
        require(terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
                and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
                and receipt["wall_seconds"] == (receipt["finished_ns"] - receipt["started_ns"]) / 1e9, "audit elapsed accounting")
        summary, audit = self.read(args.summary), self.read(args.audit / "audit.json")
        require(summary["version"] == TRAIN_VERSION and audit["version"] == AUDIT_VERSION and audit["agreement"] is True,
                "audited experiment versions")
        same(summary, audit["producer_summary"])
        require(audit["technical_condition_requires_successful_original_audit_supervisor"] is True
                and summary["technical_complete_pending_saved_audit"] is True, "original producer remains provisional")
        same(summary["required"][0], {"name": "common.technical_complete", "value": False,
             "relation": "==", "threshold": True, "passes": False})
        same(summary["gates"], gates(summary["required"]))
        final_rules = [{**summary["required"][0], "value": True, "passes": True}, *summary["required"][1:]]
        final = {**summary, "required": final_rules, "required_passed": summary["required_passed"] + 1,
                 "gates": gates(final_rules), "technical_complete_pending_saved_audit": False}
        same(audit["summary"], final)
        count = sum(r["passes"] for r in final_rules)
        require(count == final["required_passed"] == audit["counts"]["required_passed"]
                and final["required_conditions"] == audit["counts"]["required_conditions"] == 39
                and audit["counts"]["fits"] == 24 and audit["counts"]["prediction_files"] == 49
                and audit["counts"]["training_payloads"] == len(worker["files"]) == 85
                and audit["counts"]["collection_episodes"] == 90 and final["train_counts"]["episodes"] == 54
                and final["validation_counts"]["episodes"] == 36, "complete audited coverage")
        return final, audit

    def save_figure(self, fig, name):
        for suffix in ("png", "svg"):
            with (self.out / f"{name}.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())

    @staticmethod
    def dots(ax, values):
        for x, (kind, color) in enumerate(zip(KINDS, COLORS, strict=True)):
            points = []
            for seed, offset, marker in zip(SEEDS, (-.16, 0., .16), ("o", "s", "^"), strict=True):
                value = values[kind, seed]
                points.append(value)
                ax.plot(x + offset, value, marker=marker, markerfacecolor="white", markeredgecolor=color,
                        linestyle="none", markersize=5.8, zorder=3)
            mean = math.fsum(points) / 3
            ax.plot((x - .27, x + .27), (mean, mean), color=color, linewidth=2.4, zorder=4)
        ax.set_xticks(range(8), LABELS, fontsize=8)
        ax.set_xlim(-.6, 7.6); ax.axvline(3.5, color="#d1d5db", linewidth=.8)
        ax.grid(axis="y", alpha=.18)

    @staticmethod
    def gate_label(summary):
        return "   |   ".join(f"{label}: {'PASS' if summary['gates'][key]['passes'] else 'FAIL'} "
                            f"({summary['gates'][key]['passed']}/{count})" for key, (label, count) in GATES.items())

    def figures(self, summary, points, costs):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        handles = [Line2D([], [], color="#4b5563", marker=m, markerfacecolor="white", linestyle="none", label=f"Seed {s}")
                   for s, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
        handles += [Line2D([], [], color="#4b5563", linewidth=2.4, label="Mean of all 3 seeds"),
                    Line2D([], [], color="#6b7280", linestyle="--", label="Hold-Q reference (nonquery metrics)")]
        fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharey="row")
        fig.subplots_adjust(left=.07, right=.985, top=.885, bottom=.145, hspace=.46, wspace=.12)
        for row, (scope, metric, label, _) in enumerate(METRICS):
            metric_values = [p["display_value"] for p in points if p["scope"] == scope]
            maximum = max(metric_values)
            if scope == "prior":
                require(all(v > 0 for v in metric_values), "all prior-MSE points positive for logarithmic display")
                prior_limits = (min(metric_values) / 1.35, maximum * 1.35)
            for col, regime in enumerate(REGIMES):
                ax = axes[row, col]
                selected = [p for p in points if p["scope"] == scope and p["regime"] == regime]
                self.dots(ax, {(p["family"], p["seed"]): p["display_value"] for p in selected if p["family"] != "hold"})
                if scope != "prior":
                    held = next(p["display_value"] for p in selected if p["family"] == "hold")
                    ax.axhline(held, linestyle="--", linewidth=1., color="#6b7280", alpha=.8)
                if scope == "prior":
                    ax.set_yscale("log")
                    ax.set_ylim(*prior_limits)
                else:
                    ax.set_ylim(0, 100 if metric.endswith("agreement") else maximum * 1.12 if maximum else 1)
                if col == 0:
                    ax.set_ylabel(label + ("\nLog scale (lower is better)" if scope == "prior" else ""), fontsize=11)
                if row == 0:
                    ax.set_title(f"Sensing length {regime[-1]} | 6 paired VALID cases", fontsize=12, pad=12)
        fig.suptitle("OpenJev: shared versus separate prior readouts", fontsize=20, y=.975)
        fig.text(.5, .942, self.gate_label(summary), ha="center", fontsize=12, weight="bold")
        fig.text(.5, .915, "24 final fits | MSE: nonquery loss | AUX: nonquery + prior loss | No seed selection", ha="center", fontsize=11)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .096), ncol=5, frameon=False, fontsize=10)
        fig.text(.5, .07, "Equal episode weights; zero-support paths stay in the denominator. Shared scales within each metric. "
                 "Prior MSE centers all four raw scores; logarithmic axis.", ha="center", fontsize=10)
        fig.text(.5, .044, "Three distinct gates (19 / 19 / 29); no omnibus 39-condition promotion decision. "
                 "Hold has no defined prior metric here.", ha="center", fontsize=10)
        fig.text(.5, .019, "Forced-path forecasts, not autonomous control. Raw gaps measure teacher imitation, not true action regret.",
                 ha="center", fontsize=10)
        self.save_figure(fig, "otto-separate-prior-forecast"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.31, wspace=.2)
        for ax, key, label in zip(axes, ("wall_seconds", "valid_forward_chunks"),
            ("Fit + TRAIN rescore + checkpoint (seconds)", "VALID forward chunks (work count, not latency)"), strict=True):
            self.dots(ax, {(r["family"], r["seed"]): r[key] for r in costs})
            ax.set_ylim(bottom=0); ax.set_ylabel(label)
        fig.suptitle("Every fit's measured training time and VALID work", fontsize=19, y=.96)
        fig.text(.5, .865, "Same 80 epochs and 720 updates per fit. Equal updates do not mean equal compute.", ha="center", fontsize=11)
        fig.legend(handles=handles[:4], loc="lower center", bbox_to_anchor=(.5, .205), ncol=4, frameon=False)
        fig.text(.5, .155, f"Whole VALID stage: {summary['validation_seconds']:.3f} seconds, including loading, all forecasts, metrics and I/O.",
                 ha="center", fontsize=10)
        fig.text(.5, .111, "Per-fit VALID forward seconds were not recorded. Counts cannot establish inference speed.", ha="center", fontsize=10)
        fig.text(.5, .069, "Fit times exclude collection, shared setup and VALID. All 24 dots are retained; actual backward counts are in costs.csv.",
                 ha="center", fontsize=10)
        fig.text(.5, .027, "Forced-path forecasts, not autonomous control. Timings are inherited from the authenticated producer.", ha="center", fontsize=10)
        self.save_figure(fig, "otto-separate-prior-costs"); plt.close(fig)
        return matplotlib.__version__

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(p.is_symlink() for p in self.out.parents), "exclusive contained output")
        self.out.mkdir(exist_ok=False)
        try:
            self.bind(Path(__file__).absolute(), self.descriptor(Path(__file__).absolute()))
            self.bind(ROOT / BASE_PATH, BASE_PIN)
            summary, audit = self.authenticate()
            points, means, costs = tables(summary, audit)
            write_csv(self.out / "individual-points.csv", points)
            write_csv(self.out / "family-means.csv", means)
            write_csv(self.out / "costs.csv", costs)
            write_csv(self.out / "conditions.csv", [{**r, "strict_upper_bound": r.get("strict_upper_bound")} for r in summary["required"]])
            write_csv(self.out / "gates.csv", [{"gate": k, **v} for k, v in summary["gates"].items()])
            write(self.out / "plotted-values.json", {"version": VERSION, "individual_points": points, "family_means": means,
                "fit_costs": costs, "gates": summary["gates"], "conditions": summary["required"],
                "interactions": summary["interactions"], "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
                "physical_stage_seconds": {k: summary[k] for k in ("setup_seconds", "fitting_seconds", "validation_seconds")},
                "scope": summary["scope"], "audit_limitations": audit["limitations"],
                "cost_limitation": "No per-fit VALID timing exists. Forward chunks are counts; whole VALID seconds include metrics and I/O."})
            self.receipt["matplotlib_version"] = self.figures(summary, points, costs)
            self.check()
            for path, descriptor in self.bound.items():
                require(self.descriptor(Path(path)) == descriptor, "unchanged evidence after plotting")
            payloads = {"otto-separate-prior-forecast.png", "otto-separate-prior-forecast.svg", "otto-separate-prior-costs.png",
                "otto-separate-prior-costs.svg", "individual-points.csv", "family-means.csv", "costs.csv", "conditions.csv", "gates.csv", "plotted-values.json"}
            require({p.name for p in self.out.iterdir()} == payloads, "exact ten presentation payloads")
            self.receipt.update(status="completed", inputs=self.bound, gates=summary["gates"], individual_points=len(points),
                family_mean_points=len(means), fit_cost_rows=len(costs), condition_rows=39,
                files={p.name: self.descriptor(p) for p in sorted(self.out.iterdir())}, wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            write(self.out / "receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "receipt": self.descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.failed = True
            self.receipt.update(status="failed", inputs=self.bound, error=repr(error), traceback=traceback.format_exc(),
                                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as publication:  # noqa: BLE001 - preserve original presentation failure
                error.add_note(f"Failure receipt publication: {publication!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "audit", "audit-terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    Plot(parser.parse_args()).execute()


if __name__ == "__main__":
    main()
