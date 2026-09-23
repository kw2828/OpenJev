"""Plot every sampled-TRAIN forecast fit from closed, independently audited JSON.

Only saved summaries, audit/provenance JSON and source bytes are consumed. No
checkpoint, NPZ, trajectory journal, model or simulator is opened. The original
saved audit supplies scientific verification; this is a display-only reduction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-sampled-forecast-plot-v1"
AUDITOR = "scripts/audit_otto_sampled_forecasts.py"
AUDIT_VERSION = "otto-sampled-forecast-saved-audit-v1"
TRAIN_VERSION = "otto-sampled-forecast-training-v1"
TABLES = "scripts/render_otto_score_forecast_report.py"
TABLES_PIN = "cea4883aad35787f44ca2297eab9424da22aaf02488a31cef625c79310af60a9"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
KINDS = ("residual_gru", "direct_gru", "history_mlp", "current_mlp")
SEEDS = (235001, 235002, 235003)
REGIMES = ("lambda3", "lambda4")
METRICS = (("episode_weighted_agreement", "Teacher near-minimum agreement (%)", 100.),
           ("episode_weighted_raw_gap", "Raw teacher-score gap (lower is better)", 1.))
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}


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
        require(str(path) in self.bound and self.bound[str(path)]["bytes"] <= 16 * 1024**2, "authenticated bounded JSON")
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
        same(summary, audit["summary"])
        expected = [(family, seed) for seed in SEEDS for family in KINDS]
        require([(r["family"], r["seed"]) for r in summary["models"]] == expected
                and [(r["family"], r["seed"]) for r in audit["fits"]] == expected,
                "all twelve final fits without selection")
        rules = summary["required"]
        require(len(rules) == len({r["name"] for r in rules}) == summary["required_conditions"] == 45
                and all(type(r["passes"]) is bool for r in rules), "all fixed continuation conditions")
        count = sum(r["passes"] for r in rules)
        require(count == summary["required_passed"] == audit["counts"]["required_passed"]
                and summary["forecast_continuation"] is (count == 45)
                and audit["counts"]["required_conditions"] == 45
                and audit["counts"]["fits"] == 12 and audit["counts"]["prediction_files"] == 13
                and audit["counts"]["collection_episodes"] == 90
                and summary["train_counts"]["episodes"] == 54 and summary["validation_counts"]["episodes"] == 36,
                "complete audited coverage and unchanged continuation outcome")
        return summary, audit

    def figure(self, summary, points, means, tables):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row")
        fig.subplots_adjust(left=.085, right=.98, top=.81, bottom=.235, wspace=.15, hspace=.38)
        families = ("hold", *KINDS)
        for row, (metric, label, _) in enumerate(METRICS):
            ceiling = max(p["display_value"] for p in points if p["metric"] == metric)
            for column, regime in enumerate(REGIMES):
                ax = axes[row, column]
                selected = [p for p in points if p["regime"] == regime and p["metric"] == metric]
                hold = next(p for p in selected if p["family"] == "hold")
                ax.axhline(hold["display_value"], color=tables.COLORS["hold"], linestyle="--", linewidth=1., alpha=.65)
                ax.plot(0, hold["display_value"], marker="x", color=tables.COLORS["hold"], markersize=8, linestyle="none")
                for x, family in enumerate(KINDS, start=1):
                    for seed, shift, marker in zip(SEEDS, (-.14, 0., .14), ("o", "s", "^"), strict=True):
                        value = next(p["display_value"] for p in selected if p["family"] == family and p["seed"] == seed)
                        ax.plot(x + shift, value, marker=marker, markersize=6, markerfacecolor="white",
                                markeredgecolor=tables.COLORS[family], linestyle="none", zorder=3)
                    mean = next(m["display_value"] for m in means
                                if (m["family"], m["regime"], m["metric"]) == (family, regime, metric))
                    ax.plot((x - .23, x + .23), (mean, mean), color=tables.COLORS[family], linewidth=2.5, zorder=4)
                ax.set_xticks(range(5), [tables.LABELS[f].replace(" ", "\n", 1) for f in families])
                ax.set_xlim(-.45, 4.45)
                ax.set_ylim(0, 105 if metric.endswith("agreement") else (ceiling * 1.12 if ceiling else 1))
                ax.grid(axis="y", alpha=.18)
                if column == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"Sensing length {regime[-1]} | 6 paired VALID cases", fontsize=12, pad=12)
        passed, count = summary["forecast_continuation"], summary["required_passed"]
        fig.suptitle("OpenJev: forecast teacher scores between queries", fontsize=18, y=.98)
        fig.text(.5, .925, f"FIXED 45-CONDITION SCREEN: {'PASS' if passed else 'FAIL'} ({count}/45 passed)",
                 ha="center", fontsize=13, weight="bold", color="#216b43" if passed else "#a62a2a")
        fig.text(.5, .879, "FORCED-PATH VALID ONLY | Sampled TRAIN windows | Every final model seed retained",
                 ha="center", fontsize=11)
        handles = [Line2D([], [], color="#444444", marker=marker, markerfacecolor="white", linestyle="none", label=f"Seed {seed}")
                   for seed, marker in zip(SEEDS, ("o", "s", "^"), strict=True)]
        handles += [Line2D([], [], color="#444444", linewidth=2.5, label="Mean of all 3 seeds"),
                    Line2D([], [], color=tables.COLORS["hold"], linestyle="--", marker="x", label="Hold-Q baseline")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .143), ncol=5, frameon=False)
        fig.text(.5, .109, "Complete VALID paths; equal episode weights; query rows excluded. "
                 "Zero-support episodes remain in the denominator.", ha="center", fontsize=9)
        fig.text(.5, .071, "Gap is in raw teacher-score units; agreement uses the strict float32 near-minimum set. "
                 "Shared vertical scales within each metric.", ha="center", fontsize=9)
        fig.text(.5, .033, "Planner-imitation forecasts only: no autonomous-control, transfer, true-action-regret or novelty claim.",
                 ha="center", fontsize=9)
        for suffix in ("png", "svg"):
            with (self.out / f"sampled-forecast.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())
        plt.close(fig)
        return matplotlib.__version__

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(p.is_symlink() for p in self.out.parents), "exclusive contained plot output")
        self.out.mkdir(exist_ok=False)
        try:
            self.bind(Path(__file__).absolute(), self.descriptor(Path(__file__).absolute()))
            self.bind(ROOT / TABLES, TABLES_PIN)
            summary, audit = self.authenticate()
            spec = importlib.util.spec_from_file_location("_sampled_forecast_plot_tables", ROOT / TABLES)
            tables = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tables)
            tables.SEEDS = SEEDS
            records = [{"family": "hold", "seed": None, "metrics": summary["hold"]}, *summary["models"]]
            rows, points, means = tables.tables(records)
            tables.write_csv(self.out / "forecast-metrics.csv", rows)
            tables.write_csv(self.out / "family-means.csv", means)
            tables.write_csv(self.out / "conditions.csv", summary["required"])
            tables.write_csv(self.out / "fits.csv", audit["fits"])
            write(self.out / "plotted-values.json", {"version": VERSION, "individual_points": points,
                "family_means": means, "required": summary["required"], "scope": summary["scope"],
                "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
                "support": summary["support"], "audit_limitations": audit["limitations"]})
            self.receipt["matplotlib_version"] = self.figure(summary, points, means, tables)
            self.check()
            for path, descriptor in self.bound.items():
                require(self.descriptor(Path(path)) == descriptor, "unchanged saved evidence/source after plotting")
            self.receipt.update(status="completed", inputs=self.bound, required_passed=summary["required_passed"],
                required_conditions=45, forecast_continuation=summary["forecast_continuation"], metric_rows=len(rows),
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
