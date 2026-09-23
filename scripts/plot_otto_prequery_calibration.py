"""Plot every prequery calibration fit from closed, independently audited JSON.

Only saved summaries, audit/provenance JSON and source bytes are consumed. No
checkpoint, NPZ, trajectory journal, model or simulator is opened. The original
saved audit supplies scientific verification; this is a display-only reduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-prequery-calibration-plot-v1"
AUDITOR = "scripts/audit_otto_prequery_calibration.py"
AUDIT_VERSION = "otto-prequery-calibration-saved-audit-v1"
TRAIN_VERSION = "otto-prequery-calibration-training-v1"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
KINDS = ("innovation_aux", "innovation_mse", "innovation_gru_mse", "innovation_gru_aux")
SEEDS = (275000001, 275000002, 275000003)
REGIMES = ("lambda3", "lambda4")
METRICS = (("prior_centered_mse", "Raw prior-score MSE (all four actions)", 1.),
           ("episode_weighted_raw_gap", "Raw teacher-score gap (lower is better)", 1.))
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2,
          "metadata_json_bytes": 16 * 1024**2, "audit_json_bytes": 128 * 1024**2}

COLORS = {"hold": "#6b7280", "innovation_aux": "#2563eb", "innovation_mse": "#16846b",
          "innovation_gru_mse": "#8b5cf6", "innovation_gru_aux": "#d97706"}
LABELS = {"hold": "Held scores", "innovation_aux": "Innovation\n+ prior loss", "innovation_mse": "Innovation\nMSE only",
          "innovation_gru_mse": "Error GRU\nMSE only", "innovation_gru_aux": "Error GRU\n+ prior loss"}


def write_csv(path, rows):
    require(bool(rows), "nonempty saved table")
    columns = list(rows[0])
    require(all(set(row) == set(columns) for row in rows), "uniform saved table fields")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
        stream.flush(); os.fsync(stream.fileno())


def tables(summary, audit):
    """Retain every scope and fit; plotted points use only the frozen primary domain."""
    rows, points, means = [], [], []
    records = [{"family": "hold", "seed": None, "metrics": summary["hold"]}, *summary["models"]]
    require([(r["family"], r["seed"]) for r in records] == [("hold", None),
            *[(kind, seed) for seed in SEEDS for kind in KINDS]], "all model identities in fixed order")
    for record in records:
        report = record["metrics"]
        require(report["primary_mask"] == "nonquery and absolute_step >= 5", "declared primary domain")
        variants = [("all", report), *report["by_collector"].items()]
        require({name for name, _ in variants} == {"all", "analytic", "neural", "period4_hold"}, "all collectors")
        for collector, variant in variants:
            for scope in ("full", "postcorrection"):
                group = variant[scope]
                groups = [("overall", None, None, group["overall"])]
                groups += [("regime", regime, None, values) for regime, values in group["by_regime"].items()]
                groups += [("case", value["regime"], value["case"], value) for value in group["by_case"]]
                for partition, regime, case, values in groups:
                    for age, item in [("all", values), *values["by_age"].items()]:
                        scalar = {k: v for k, v in item.items() if k not in ("by_age", "regime", "case")}
                        rows.append({"family": record["family"], "seed": record["seed"], "scope": scope,
                            "collector": collector, "partition": partition, "regime": regime, "case": case,
                            "age": age, **scalar})
        for regime in REGIMES:
            for metric, _, factor in METRICS:
                if metric == "prior_centered_mse":
                    if record["family"] == "hold":
                        continue
                    value = record["prior_metrics"]["by_regime"][regime]["episode_weighted_centered_mse"]
                    scope = "later queries absolute_step >= 4; all four actions"
                else:
                    value = report["postcorrection"]["by_regime"][regime][metric]
                    scope = "postcorrection nonquery absolute_step >= 5"
                require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "finite plotted value")
                points.append({"family": record["family"], "seed": record["seed"], "regime": regime,
                    "metric": metric, "raw_value": value, "display_value": factor * value,
                    "scope": scope})
    for kind in KINDS:
        for regime in REGIMES:
            for metric, _, factor in METRICS:
                selected = [p for p in points if (p["family"], p["regime"], p["metric"]) == (kind, regime, metric)]
                require({p["seed"] for p in selected} == set(SEEDS) and len(selected) == 3, "all seeds in each mean")
                value = math.fsum(p["raw_value"] for p in selected) / 3
                means.append({"family": kind, "regime": regime, "metric": metric, "seed_count": 3,
                              "raw_value": value, "display_value": factor * value})
    prior_rows = []
    for split, reports, episodes, cases in (("train", audit["training_forecast_checks"], 54, 18),
                                           ("valid", summary["models"], 36, 12)):
        require([(r["family"], r["seed"]) for r in reports] == [(k, s) for s in SEEDS for k in KINDS],
                "every final TRAIN and VALID prior report")
        for record in reports:
            prior = record["prior_metrics"]
            require(prior["prior_mask"] == "query and absolute_step >= 4"
                    and prior["overall"]["episodes"] == episodes and set(prior["by_regime"]) == set(REGIMES)
                    and len(prior["by_case"]) == cases and len(prior["by_episode"]) == episodes,
                    "complete fixed prior domains")
            groups = [("overall", None, None, None, None, prior["overall"])]
            groups += [("regime", r, None, None, None, g) for r, g in prior["by_regime"].items()]
            groups += [("case", g["regime"], g["case"], None, None, g) for g in prior["by_case"]]
            groups += [("episode", g["regime"], g["case"], g["episode_id"], g["arm"], g) for g in prior["by_episode"]]
            for partition, regime, case, episode_id, arm, group in groups:
                prior_rows.append({"family": record["family"], "seed": record["seed"], "split": split,
                    "partition": partition, "regime": regime, "case": case, "episode_id": episode_id, "arm": arm,
                    **{k: v for k, v in group.items() if k not in ("regime", "case", "episode_id", "arm")}})
    require(len(points) == 50 and len(means) == 16 and len(rows) == 6240 and len(prior_rows) == 1512,
            "complete point and scope coverage")
    return rows, points, means, prior_rows


def cost_rows(fits):
    """Every measured fit plus within-seed, within-architecture MSE comparison."""
    by = {(r["family"], r["seed"]): r for r in fits}
    require(set(by) == {(kind, seed) for seed in SEEDS for kind in KINDS} and len(fits) == 12,
            "all final fit costs without selection")
    rows = []
    for fit in fits:
        baseline_kind = "innovation_gru_mse" if fit["architecture"] == "innovation_gru" else "innovation_mse"
        baseline = by[baseline_kind, fit["seed"]]
        require(all(type(fit[k]) in (int, float) and math.isfinite(fit[k]) and fit[k] > 0
                    for k in ("fit_seconds", "wall_seconds")), "positive measured fit costs")
        rows.append({k: fit[k] for k in ("family", "architecture", "objective", "seed", "fit_seconds", "wall_seconds",
            "train_rescore_seconds", "checkpoint_seconds", "forward_chunks", "backward_chunks", "no_grad_chunks")}
            | {"paired_mse_family": baseline_kind,
               "fit_seconds_ratio_to_paired_mse": fit["fit_seconds"] / baseline["fit_seconds"],
               "wall_seconds_ratio_to_paired_mse": fit["wall_seconds"] / baseline["wall_seconds"]})
    return rows


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(actual, expected):
    """Use exactly the original saved-audit comparison tolerance, never a new one."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "exact audited schema")
        for key, value in expected.items():
            same(actual[key], value)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), "exact audited list coverage")
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), "audited saved numeric value")
    else:
        require(type(actual) is type(expected) and actual == expected, "exact audited nonnumeric value")


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


class Plot:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.start, self.bound, self.failed = time.monotonic_ns(), {}, False
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
            "model_calls": 0, "native_calls": 0, "optimizer_calls": 0, "array_decodes": 0,
            "scope": "Saved JSON display only. Scientific arithmetic and original producer closure are inherited from the completed audit."}

    def check(self):
        require(time.monotonic_ns() - self.start < LIMITS["seconds"] * 10**9, "plot deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "plot RSS/output caps with failure reserve")

    def descriptor(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained evidence")
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                if not self.failed:
                    self.check()
                digest.update(block); size += len(block)
        return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}

    def bind(self, path, expected):
        descriptor = self.descriptor(path)
        if isinstance(expected, str):
            require(descriptor["sha256"] == expected, "externally pinned SHA256")
        else:
            require({k: descriptor[k] for k in expected} == expected, "exact saved evidence descriptor")
        self.bound[str(path)] = descriptor
        return descriptor

    def read(self, path):
        # The audit retains both complete summaries plus all-fit provenance.
        limit = LIMITS["audit_json_bytes"] if path == self.args.audit / "audit.json" else LIMITS["metadata_json_bytes"]
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
                and receipt["requires_successful_original_supervisor"] is True,
                "completed agreeing independent saved audit")
        require(set(receipt["files"]) == {"started.json", "audit.json"}
                and {p.name for p in args.audit.iterdir()} == {"started.json", "audit.json", "receipt.json"},
                "closed two-payload audit inventory")
        for name, descriptor in receipt["files"].items():
            self.bind(args.audit / name, descriptor)
        for name, pin in receipt["sources"].items():
            self.bind(ROOT / name, pin)
        require(AUDITOR in receipt["sources"], "original prospective auditor identity")
        require(set(receipt["producer_inputs"]) == {"plan", "worker", "terminal"}, "all original producer joins")
        for descriptor in receipt["producer_inputs"].values():
            self.bind(Path(descriptor["path"]), descriptor)
        require(receipt["plan_sha256"] == receipt["producer_inputs"]["plan"]["sha256"], "same frozen training plan")
        worker_path = Path(receipt["producer_inputs"]["worker"]["path"])
        worker = self.read(worker_path)
        require(worker["version"] == TRAIN_VERSION and worker["status"] == "completed" and worker["complete"] is True
                and worker["fits_completed"] == 12 and args.summary == worker_path.parent / "summary.json",
                "same completely fitted producer summary")
        self.bind(args.summary, worker["files"]["summary.json"])
        require(terminal["status"] == "completed" and terminal["returncode"] == 0
                and terminal["timed_out"] is False and terminal["error"] is terminal["clock_error"] is None
                and terminal["group_absent"] is terminal["cleanup"]["reaped"] is True
                and terminal["cleanup"]["errors"] == [] and terminal["cap_seconds"] == 120
                and terminal["clock_source_sha256"] == CLOCK_PIN and terminal["watchdog_sha256"] == SUPERVISOR_PIN
                and terminal["cwd"] == str(ROOT) and terminal["deadline_ns"] == terminal["started_ns"] + 120 * 10**9
                and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
                <= terminal["finished_ns"] <= terminal["deadline_ns"], "successful original audit process closure")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command[:2] == [str(ROOT / ".venv/bin/python"), str(ROOT / AUDITOR)]
                and len(command[2:]) == 16 and len(set(command[2::2])) == 8, "actual absolute original audit command")
        options = dict(zip(command[2::2], command[3::2], strict=True))
        expected = {"--output": str(args.audit)}
        for role, descriptor in receipt["producer_inputs"].items():
            expected[f"--{role}"] = descriptor["path"]
            expected[f"--{role}-sha256"] = descriptor["sha256"]
        require(set(options) == set(expected) | {"--supervision"}
                and all(options[k] == value for k, value in expected.items()), "original audit command input/output joins")
        launch_path = Path(options["--supervision"])
        self.bind(launch_path, receipt["supervision_sha256"])
        launch, started = self.read(launch_path), self.read(args.audit / "started.json")
        require(started["launch"] == launch and all(terminal[k] == value for k, value in launch.items()),
                "original audit launch retained through terminal")
        require(terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
                and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
                and receipt["wall_seconds"] == (receipt["finished_ns"] - receipt["started_ns"]) / 1e9,
                "original audit elapsed-time accounting")
        summary, audit = self.read(args.summary), self.read(args.audit / "audit.json")
        require(summary["version"] == TRAIN_VERSION and audit["version"] == AUDIT_VERSION and audit["agreement"] is True,
                "correct audited experiment versions")
        same(summary, audit["producer_summary"])
        require(audit["technical_condition_requires_successful_original_audit_supervisor"] is True,
                "technical gate requires the successful parent verified above")
        require(summary["technical_complete_pending_saved_audit"] is True
                and summary["forecast_continuation"] is False, "producer remains provisional")
        same(summary["required"][0], {"name": "technical_complete", "value": False,
             "relation": "==", "threshold": True, "passes": False})
        final_rules = [{**summary["required"][0], "value": True, "passes": True}, *summary["required"][1:]]
        final = {**summary, "required": final_rules, "required_passed": summary["required_passed"] + 1,
            "forecast_continuation": all(row["passes"] for row in final_rules),
            "technical_complete_pending_saved_audit": False}
        same(audit["summary"], final)
        summary = audit["summary"]
        expected = [(family, seed) for seed in SEEDS for family in KINDS]
        require([(r["family"], r["seed"]) for r in summary["models"]] == expected
                and [(r["family"], r["seed"]) for r in audit["fits"]] == expected,
                "all twelve final fits without selection")
        rules = summary["required"]
        require(len(rules) == len({r["name"] for r in rules}) == summary["required_conditions"] == 55
                and all(type(r["passes"]) is bool for r in rules), "all fixed continuation conditions")
        count = sum(r["passes"] for r in rules)
        require(count == summary["required_passed"] == audit["counts"]["required_passed"]
                and summary["forecast_continuation"] is (count == 55)
                and audit["counts"]["required_conditions"] == 55
                and audit["counts"]["fits"] == 12 and audit["counts"]["prediction_files"] == 25
                and audit["counts"]["collection_episodes"] == 90
                and summary["train_counts"]["episodes"] == 54 and summary["validation_counts"]["episodes"] == 36,
                "complete audited coverage and unchanged continuation outcome")
        return summary, audit

    def figure(self, summary, points, means):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey="row")
        fig.subplots_adjust(left=.085, right=.98, top=.81, bottom=.235, wspace=.15, hspace=.38)
        for row, (metric, label, _) in enumerate(METRICS):
            families = KINDS if metric == "prior_centered_mse" else ("hold", *KINDS)
            ceiling = max(p["display_value"] for p in points if p["metric"] == metric)
            for column, regime in enumerate(REGIMES):
                ax = axes[row, column]
                selected = [p for p in points if p["regime"] == regime and p["metric"] == metric]
                if metric != "prior_centered_mse":
                    hold = next(p for p in selected if p["family"] == "hold")
                    ax.axhline(hold["display_value"], color=COLORS["hold"], linestyle="--", linewidth=1., alpha=.65)
                    ax.plot(0, hold["display_value"], marker="x", color=COLORS["hold"], markersize=8, linestyle="none")
                for x, family in enumerate(KINDS, start=int(metric != "prior_centered_mse")):
                    for seed, shift, marker in zip(SEEDS, (-.14, 0., .14), ("o", "s", "^"), strict=True):
                        value = next(p["display_value"] for p in selected if p["family"] == family and p["seed"] == seed)
                        ax.plot(x + shift, value, marker=marker, markersize=6, markerfacecolor="white",
                                markeredgecolor=COLORS[family], linestyle="none", zorder=3)
                    mean = next(m["display_value"] for m in means
                                if (m["family"], m["regime"], m["metric"]) == (family, regime, metric))
                    ax.plot((x - .23, x + .23), (mean, mean), color=COLORS[family], linewidth=2.5, zorder=4)
                ax.set_xticks(range(len(families)), [LABELS[f] for f in families])
                ax.set_xlim(-.45, len(families) - .55)
                ax.set_ylim(0, ceiling * 1.12 if ceiling else 1)
                ax.grid(axis="y", alpha=.18)
                if column == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"Sensing length {regime[-1]} | 6 paired VALID cases", fontsize=12, pad=12)
        passed, count = summary["forecast_continuation"], summary["required_passed"]
        fig.suptitle("OpenJev: calibrating the forecast before a planner query", fontsize=18, y=.98)
        fig.text(.5, .925, f"FIXED 55-CONDITION SCREEN: {'PASS' if passed else 'FAIL'} ({count}/55 passed)",
                 ha="center", fontsize=13, weight="bold", color="#216b43" if passed else "#a62a2a")
        fig.text(.5, .879, "Calibration: later queries >= 4 | PRIMARY forecast gap: nonquery steps >= 5 | Every final seed retained",
                 ha="center", fontsize=11)
        handles = [Line2D([], [], color="#444444", marker=marker, markerfacecolor="white", linestyle="none", label=f"Seed {seed}")
                   for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True)]
        handles += [Line2D([], [], color="#444444", linewidth=2.5, label="Mean of all 3 seeds"),
                    Line2D([], [], color=COLORS["hold"], linestyle="--", marker="x", label="Hold-Q baseline")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .143), ncol=5, frameon=False)
        fig.text(.5, .109, "Complete fixed VALID paths; equal episode weights. "
                 "Zero-support episodes remain in the denominator.", ha="center", fontsize=9)
        fig.text(.5, .071, "Prior MSE centers all four raw scores before the query arrives. Hold has no defined prior forecast here. "
                 "Shared scales within each metric.", ha="center", fontsize=9)
        fig.text(.5, .033, "Fixed-path forecasts, not autonomous control. Raw gap measures teacher imitation, not true action regret.",
                 ha="center", fontsize=9)
        for suffix in ("png", "svg"):
            with (self.out / f"prequery-forecast.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())
        plt.close(fig)
        return matplotlib.__version__

    def condition_figure(self, summary):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(17, 14))
        fig.subplots_adjust(left=.035, right=.99, top=.895, bottom=.075, wspace=.035)
        rules = summary["required"]
        for column, ax in enumerate(axes):
            ax.axis("off")
            selected = list(enumerate(rules, start=1))[column*28:(column+1)*28]
            for row, (number, rule) in enumerate(selected):
                y = 1 - (row + .5)/28
                passed = rule["passes"]
                color = "#216b43" if passed else "#a62a2a"
                ax.text(.0, y, f"{number:02d} {'PASS' if passed else 'FAIL'}", color=color,
                        fontsize=8.5, weight="bold", va="center", transform=ax.transAxes)
                ax.text(.15, y, rule["name"], fontsize=8.1, va="center", transform=ax.transAxes)
        fig.suptitle("OpenJev prequery calibrations: every continuation condition", fontsize=18, y=.98)
        fig.text(.5, .94, f"{'PASS' if summary['forecast_continuation'] else 'FAIL'}: "
                 f"{summary['required_passed']}/55 passed | No seed or setting selected", ha="center", fontsize=13)
        fig.text(.5, .915, "Technical closure + 54 scientific conditions | Numerical operands are retained in conditions.csv",
                 ha="center", fontsize=10)
        fig.text(.5, .042, "Fixed-path forecasts, not autonomous control. Primary: nonquery absolute step >= 5.",
                 ha="center", fontsize=10)
        for suffix in ("png", "svg"):
            with (self.out / f"prequery-conditions.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())
        plt.close(fig)

    def cost_figure(self, rows):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.80, wspace=.22)
        for ax, metric, label in zip(axes, ("fit_seconds", "wall_seconds"),
                                    ("Fit interval (seconds)", "Fit + final TRAIN diagnostics + checkpoint (seconds)"), strict=True):
            for x, kind in enumerate(KINDS):
                values = []
                for seed, shift, marker in zip(SEEDS, (-.14, 0., .14), ("o", "s", "^"), strict=True):
                    value = next(r[metric] for r in rows if (r["family"], r["seed"]) == (kind, seed))
                    values.append(value)
                    ax.plot(x + shift, value, marker=marker, color=COLORS[kind], markerfacecolor="white", linestyle="none")
                mean = math.fsum(values) / 3
                ax.plot((x - .23, x + .23), (mean, mean), color=COLORS[kind], linewidth=2.5)
            ax.set_xticks(range(4), [LABELS[k] for k in KINDS])
            ax.set_ylim(bottom=0); ax.set_ylabel(label); ax.grid(axis="y", alpha=.18)
        fig.suptitle("OpenJev: measured fitting cost for every final model", fontsize=17, y=.96)
        fig.text(.5, .865, "Same 80 epochs and 720 optimizer updates per fit; auxiliary backward work is fully counted",
                 ha="center", fontsize=10)
        fig.text(.5, .15, "Dots retain all three seeds; horizontal bars average those same seeds. "
                 "costs.csv includes paired MSE ratios and backward counts.", ha="center", fontsize=9)
        fig.text(.5, .08, "Fit intervals exclude collection, shared setup and VALID evaluation. "
                 "Setup/VALID totals are retained in plotted-values.json; collection remains in linked provenance.", ha="center", fontsize=9)
        fig.text(.5, .025, "Fixed-path forecasts, not autonomous control. Timing witnesses are inherited from the audited producer.",
                 ha="center", fontsize=9)
        for suffix in ("png", "svg"):
            with (self.out / f"prequery-costs.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())
        plt.close(fig)

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(p.is_symlink() for p in self.out.parents), "exclusive contained plot output")
        self.out.mkdir(exist_ok=False)
        try:
            self.bind(Path(__file__).absolute(), self.descriptor(Path(__file__).absolute()))
            summary, audit = self.authenticate()
            rows, points, means, priors = tables(summary, audit)
            costs = cost_rows(audit["fits"])
            write_csv(self.out / "forecast-metrics.csv", rows)
            write_csv(self.out / "prior-metrics.csv", priors)
            write_csv(self.out / "family-means.csv", means)
            write_csv(self.out / "conditions.csv", [{"condition_number": i + 1, **row,
                       "strict_upper_bound": row.get("strict_upper_bound")}
                       for i, row in enumerate(summary["required"])])
            write_csv(self.out / "fits.csv", audit["fits"])
            write_csv(self.out / "costs.csv", costs)
            write(self.out / "plotted-values.json", {"version": VERSION, "individual_points": points,
                "family_means": means, "required": summary["required"], "scope": summary["scope"],
                "primary_mask": "nonquery and absolute_step >= 5", "train_counts": summary["train_counts"],
                "prior_mask": "query and absolute_step >= 4", "fit_costs": costs,
                "physical_stage_seconds": {k: summary[k] for k in ("setup_seconds", "fitting_seconds", "validation_seconds")},
                "physical_stage_scope": "training phase only; collection and other original phases remain in linked provenance",
                "validation_counts": summary["validation_counts"], "audit_limitations": audit["limitations"]})
            self.receipt["matplotlib_version"] = self.figure(summary, points, means)
            self.condition_figure(summary)
            self.cost_figure(costs)
            self.check()
            for path, descriptor in self.bound.items():
                require(self.descriptor(Path(path)) == descriptor, "unchanged saved evidence/source after plotting")
            payloads = {"prequery-forecast.png", "prequery-forecast.svg", "prequery-conditions.png",
                        "prequery-conditions.svg", "forecast-metrics.csv", "family-means.csv",
                        "conditions.csv", "fits.csv", "plotted-values.json", "prior-metrics.csv", "costs.csv",
                        "prequery-costs.png", "prequery-costs.svg"}
            require({p.name for p in self.out.iterdir()} == payloads, "exact thirteen plot payloads")
            self.receipt.update(status="completed", inputs=self.bound, required_passed=summary["required_passed"],
                required_conditions=55, forecast_continuation=summary["forecast_continuation"], metric_rows=len(rows),
                prior_metric_rows=len(priors), fit_cost_rows=len(costs),
                individual_points=len(points), family_mean_points=len(means),
                files={p.name: self.descriptor(p) for p in sorted(self.out.iterdir()) if p.is_file()},
                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
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
            except BaseException as publication:  # noqa: BLE001 - preserve original plotting failure
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
