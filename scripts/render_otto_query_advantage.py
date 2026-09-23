"""Render closed, independently audited query-intervention evidence from JSON.

Every source and payload is authenticated before outcome decoding. No policy,
sampler, model or scientific implementation is imported; array files are hashed
as opaque bytes. Matplotlib is imported only for the final saved-data figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-advantage-render-v1"
PLAN_PIN = "d3ccfc6990297491f79253640cecee41cab2f8d38c1a8bfbb226f67721139ac6"
AUDITOR = "scripts/audit_otto_query_advantage.py"
AUDITOR_PIN = "b394f85d570155b103a74ee991e1bec7b88680d75095a75ffaf80d1a582b8529"
PRODUCER = "scripts/study_otto_query_advantage.py"
PRODUCER_PIN = "ab67e579a7db5147e8cc61cc6e467a85da65edc8c1fc9d121c0180393a1ca6ff"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}
PAYLOADS = {name + ".jsonl" for name in ("work", "weights", "forwards", "gate-operations",
    "public-transitions", "native-truth", "episodes", "anchors", "sampler-events", "panels")} | {
    "started.json", "runtime.json", "setup.json", "anchors.npz", "costs.json", "summary.json"}
AUDIT_PAYLOADS = {"started.json", "anchors.jsonl", "bootstrap.jsonl", "audit.json"}
REGIMES, GROUPS = ("base", "shift"), ("all", "different_action")
COLORS = {"all": "#666666", "different_action": "#2166ac"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def csv_write(path, records):
    require(records, "nonempty complete CSV")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())


def same(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), "audited field coverage")
        for key, value in expected.items():
            same(actual[key], value)
    elif isinstance(expected, list):
        require(type(actual) is list and len(actual) == len(expected), "audited list coverage")
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), "saved audited number")
    else:
        require(type(actual) is type(expected) and actual == expected, "saved audited value")


class Render:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.start, self.bound = time.monotonic_ns(), {}
        self.failed = False
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
            "model_calls": 0, "native_calls": 0, "sampler_calls": 0, "optimizer_calls": 0,
            "array_decodes": 0, "scope": "Saved JSON presentation; independent scientific audit is inherited."}

    def check(self):
        require(time.monotonic_ns() - self.start < LIMITS["seconds"] * 10**9, "render deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir())
                <= LIMITS["output_bytes"], "render RSS/output cap")

    def digest(self, path):
        require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
        result, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                if not self.failed:
                    self.check()
                result.update(block)
                size += len(block)
        return {"sha256": result.hexdigest(), "bytes": size}

    def bind(self, value, expected=None):
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        if str(path) not in self.bound:
            self.bound[str(path)] = self.digest(path)
        actual = self.bound[str(path)]
        if isinstance(expected, dict):
            require(actual == expected, "exact frozen descriptor")
        elif expected is not None:
            require(actual["sha256"] == expected, "external/frozen SHA256")
        return path

    def closed(self, directory, receipt, expected):
        require(set(receipt["files"]) == expected and {p.name for p in directory.iterdir()}
                == expected | {"receipt.json"}, "exact closed payload inventory")
        for name, desc in receipt["files"].items():
            self.bind(directory / name, desc)

    def parent(self, receipt, terminal, directory, script, cap, options):
        require(terminal["status"] == "completed" and terminal["returncode"] == 0
            and terminal["timed_out"] is False and terminal["error"] is terminal["clock_error"] is None
            and terminal["group_absent"] is terminal["cleanup"]["group_absent"] is terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["errors"] == [] and terminal["cap_seconds"] == cap
            and terminal["clock_source_sha256"] == CLOCK_PIN and terminal["watchdog_sha256"] == SUPERVISOR_PIN
            and terminal["deadline_ns"] == terminal["started_ns"] + cap * 10**9
            and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= terminal["finished_ns"] <= terminal["deadline_ns"] and Path(terminal["cwd"]) == ROOT,
            "successful original parent and bounded worker")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command[0] == str(ROOT / (".venv-otto-released-native/bin/python" if script == PRODUCER
                                        else ".venv/bin/python")), "original phase interpreter")
        named = Path(command[1])
        require((named if named.is_absolute() else ROOT / named) == ROOT / script, "original phase source")
        offset = 3 if script == PRODUCER else 2
        require(script != PRODUCER or command[2] == "run", "original collection mode")
        require(len(command[offset:]) == 2 * (len(options) + 1), "exact original argument coverage")
        actual = dict(zip(command[offset::2], command[offset+1::2], strict=True))
        require(set(actual) == set(options) | {"--supervision"}
                and all(actual[k] == str(v) for k, v in options.items()), "original input/output invocation")
        launch_path = self.bind(actual["--supervision"], receipt["supervision_sha256"])
        launch, started = read(launch_path), read(directory / "started.json")
        require(launch == started["launch"] and all(terminal[k] == v for k, v in launch.items())
                and launch["pid"] == launch["pgid"] != launch["parent_pid"], "actual original launch joins")
        same(receipt["wall_seconds"], (receipt["finished_ns"] - receipt["started_ns"]) / 1e9)
        same(terminal["wall_seconds"], (terminal["finished_ns"] - terminal["started_ns"]) / 1e9)

    def authenticate(self):
        a = self.args
        require(a.plan_sha256 == PLAN_PIN, "exact prospective pilot plan")
        for path, pin in ((a.plan, a.plan_sha256), (a.run / "receipt.json", a.worker_sha256),
                         (a.terminal, a.terminal_sha256), (a.audit / "receipt.json", a.audit_sha256),
                         (a.audit_terminal, a.audit_terminal_sha256)):
            self.bind(path, pin)
        plan, worker, terminal = read(a.plan), read(a.run / "receipt.json"), read(a.terminal)
        audit, audit_terminal = read(a.audit / "receipt.json"), read(a.audit_terminal)
        require(plan["version"] == worker["version"] == "otto-query-advantage-study-v1"
            and plan["status"] == "frozen_before_collection" and worker["status"] == "completed"
            and worker["complete"] is True and worker["completed_episodes"] == 72
            and worker["pending"] == worker["pending_publications"] == worker["sampler_pending"] == []
            and worker["training_updates"] == worker["evaluation_episodes"] == 0
            and worker["plan_sha256"] == a.plan_sha256 and worker["requires_successful_original_supervisor"] is True,
            "completed exact TRAIN-only pilot")
        require(audit["version"] == "otto-query-advantage-audit-v1" and audit["status"] == "completed"
            and audit["agreement"] is True and audit["requires_successful_original_supervisor"] is True
            and audit["plan_sha256"] == a.plan_sha256 and audit["worker_sha256"] == a.worker_sha256
            and audit["terminal_sha256"] == a.terminal_sha256
            and audit["sources"] == worker["sources"] == plan["sources"]
            and plan["sources"][AUDITOR] == AUDITOR_PIN and plan["sources"][PRODUCER] == PRODUCER_PIN,
            "independent agreement and exact producer identities")
        for name, pin in plan["sources"].items():
            self.bind(name, pin)
        for key in ("inputs", "native_inputs"):
            require(worker[key] == plan[key], "complete input lineage")
            for desc in plan[key].values():
                self.bind(desc["path"], {k: desc[k] for k in ("sha256", "bytes")})
        require(set(plan["payloads"]) == PAYLOADS, "frozen payload set")
        self.closed(a.run, worker, PAYLOADS)
        self.closed(a.audit, audit, AUDIT_PAYLOADS)
        self.parent(worker, terminal, a.run, PRODUCER, 900,
            {"--plan": a.plan, "--plan-sha256": a.plan_sha256, "--output": a.run})
        self.parent(audit, audit_terminal, a.audit, AUDITOR, 120,
            {"--plan": a.plan, "--plan-sha256": a.plan_sha256, "--run": a.run,
             "--receipt-sha256": a.worker_sha256, "--terminal": a.terminal,
             "--terminal-sha256": a.terminal_sha256, "--output": a.audit})
        self.worker, self.audit_receipt = worker, audit
        self.times = {"pilot_worker_seconds": worker["wall_seconds"], "pilot_parent_seconds": terminal["wall_seconds"],
                      "audit_worker_seconds": audit["wall_seconds"], "audit_parent_seconds": audit_terminal["wall_seconds"]}

    def data(self):
        a = self.args
        original, result = read(a.run / "summary.json"), read(a.audit / "audit.json")
        require(result["agreement"] is True and result["version"] == "otto-query-advantage-audit-v1", "closed audit result")
        summary = result["summary"]
        same(original["costs"], summary["costs"])
        require(original["complete"] is True and original["anchor_slots"] == 360
            and len(original["episodes"]) == 72 and summary["counts"]["episodes"] == 72, "all paths/anchor slots")
        anchors, bootstrap = rows(a.audit / "anchors.jsonl"), rows(a.audit / "bootstrap.jsonl")
        slots = rows(a.run / "anchors.jsonl")
        available = [r for r in slots if r["status"] == "available"]
        require(len(slots) == 360 and [r["anchor_id"] for r in slots] == list(range(360))
            and [r["anchor_id"] for r in anchors] == [r["anchor_id"] for r in available]
            and len(anchors) == original["available_anchors"] == summary["counts"]["anchors"]
            and len(bootstrap) == summary["bootstrap_draws"] == 2000
            and [r["draw"] for r in bootstrap] == list(range(2000)), "every anchor and bootstrap draw")
        conditions = summary["conditions"]
        names = {"technical_complete", "pooled_lower.all", "pooled_lower.different_action"} | {
            f"{regime}.{suffix}" for regime in REGIMES for suffix in
            ("different_anchors", "different_cases", "covariance.all", "covariance.different_action")}
        require(len(conditions) == 11 and {r["name"] for r in conditions} == names, "all eleven conditions")
        for row in conditions:
            require(row["relation"] in (">", ">=") and math.isfinite(row["value"])
                and math.isfinite(row["threshold"]) and type(row["passes"]) is bool
                and row["passes"] == (row["value"] > row["threshold"] if row["relation"] == ">"
                                      else row["value"] >= row["threshold"]), "saved admission Boolean")
        require(summary["conditions_passed"] == sum(r["passes"] for r in conditions)
            and summary["conditions_total"] == 11 and summary["signal_admitted"] == all(r["passes"] for r in conditions)
            and all(self.audit_receipt[k] == summary[k] for k in
                    ("conditions_passed", "conditions_total", "signal_admitted")), "audit receipt/admission joins")
        statistics, weights = [], {}
        for regime in REGIMES:
            local = summary["regimes"][regime]
            same(original["signals"][regime], local)
            weights.update({r["anchor_id"]: r["weight"] for r in local["weights"]})
            for group in GROUPS:
                values, stats = local[group], local[group]["statistics"]
                record = {"regime": regime, "group": group, "anchors": values["anchor_count"],
                    "episodes": values["episode_count"], "original_weight_mass": values["original_weight_mass"]}
                for key in ("mean_advantage", "cross_half_covariance", "pooled_half_variance", "repeatability",
                            "zero_difference_fraction", "both_censored_fraction"):
                    record[key] = None if stats is None else stats[key]
                for role in ("analytic", "neural"):
                    for rate in ("found_fraction", "censored_fraction", "at_cap_fraction"):
                        record[f"{role}_{rate}"] = None if stats is None else stats["branch_rates"][role][rate]
                statistics.append(record)
        plotted = []
        for row, source in zip(anchors, available, strict=True):
            for key in ("anchor_id", "episode_id", "regime", "case", "schedule", "preaction_step"):
                require(row[key] == source[key], "displayed anchor identity")
            reduction = row["reduction"]
            first, second = (half["mean_advantage"] for half in reduction["halves"])
            require(all(math.isfinite(x) and -31 <= x <= 31 for x in (first, second)), "uncropped capped half means")
            record = {k: row[k] for k in ("anchor_id", "episode_id", "regime", "case", "schedule", "preaction_step")}
            record.update(weight=weights[row["anchor_id"]], same_action=reduction["structural_zero"],
                half_a=first, half_b=second, mean_advantage=reduction["mean_advantage"],
                paired_standard_error=reduction["paired_standard_error"], physical_records=reduction["physical_record_count"],
                tied_replicates=reduction["tie_count"], both_censored_replicates=reduction["both_censored_count"])
            for role in ("analytic", "neural"):
                record[role + "_censored_fraction"] = reduction["branches"][role]["censored_fraction"]
            plotted.append(record)
        draws = []
        for row in bootstrap:
            record = {"draw": row["draw"]}
            for group in GROUPS:
                value = row["pooled_covariances"][group]
                require(math.isfinite(value), "finite unfiltered bootstrap value")
                record[group] = value
                for regime in REGIMES:
                    record[regime + "_" + group] = row["covariances"][regime][group]
            for regime in REGIMES:
                record[regime + "_sampled_cases"] = json.dumps(row["sampled_cases"][regime], separators=(",", ":"))
            draws.append(record)
        for group in GROUPS:
            require(sorted(r[group] for r in draws)[199] == summary["lower_bounds"][group], "frozen lower percentile")
        return summary, plotted, draws, statistics, result["limitations"]

    def figure(self, summary, anchors, draws):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.subplots_adjust(left=.055, right=.985, bottom=.26, top=.77, wspace=.3)
        extent = max(1., max(abs(r[k]) for r in anchors for k in ("half_a", "half_b"))) * 1.08
        for regime, ax in zip(REGIMES, axes[:2], strict=True):
            local = [r for r in anchors if r["regime"] == regime]
            for same_action, color, mark, label in ((True, "#8c8c8c", "x", "Same action"),
                    (False, COLORS["different_action"], "o", "Different actions")):
                subset = [r for r in local if r["same_action"] is same_action]
                ax.scatter([r["half_a"] for r in subset], [r["half_b"] for r in subset],
                    c=color, marker=mark, s=30, alpha=.65, linewidths=.8,
                    label=f"{label}: {len(subset)}", zorder=3)
            ax.plot([-extent, extent], [-extent, extent], color="#aaaaaa", linestyle=":", linewidth=1)
            ax.axhline(0, color="#666666", linewidth=.7)
            ax.axvline(0, color="#666666", linewidth=.7)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="Replicates 0-7: mean advantage (moves)",
                ylabel="Replicates 8-15: mean advantage (moves)",
                title=f"{'Base: sensing 3' if regime == 'base' else 'Shift: sensing 4'}\n{len(local)} available anchors")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper left", fontsize=8)
        ax = axes[2]
        values = [r[g] for r in draws for g in GROUPS] + [0.]
        low, high = min(values), max(values)
        margin = max((high-low) * .07, .01)
        edges = [low-margin + (high-low+2*margin) * i / 40 for i in range(41)]
        for group, label in (("all", "All anchors"), ("different_action", "Different actions")):
            bound = summary["lower_bounds"][group]
            ax.hist([r[group] for r in draws], bins=edges, histtype="step", linewidth=1.8,
                    color=COLORS[group], label=f"{label}: lower {bound:.4g}")
            ax.axvline(bound, color=COLORS[group], linestyle="--", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(xlim=(edges[0], edges[-1]), ylim=(0, None), xlabel="Pooled centered covariance (moves squared)",
               ylabel="Bootstrap draws / bin", title="Case-cluster resamples\nAll 2,000 draws in each distribution")
        ax.legend(fontsize=8)
        status = "PASS" if summary["signal_admitted"] else "FAIL"
        fig.suptitle(f"Value-of-query signal pilot: {status}, {summary['conditions_passed']}/11 conditions", fontsize=16, y=.94)
        fig.text(.5, .865, f"72 full TRAIN paths; {len(anchors)}/360 anchors available; 16 paired replicas; horizon 32 including first action",
                 ha="center", fontsize=10)
        fig.text(.5, .12, "Advantage = analytic moves minus neural moves, then the same analytic continuation.\n"
            "Every anchor is plotted without jitter; overlaps remain overlaps. Equal scatter scales; CSV retains all points and episode weights.\n"
            "Dashed bootstrap lines: prespecified 200th sorted draw (descriptive 90% lower value). No efficacy or confidence-coverage claim.",
            ha="center", fontsize=9)
        fig.text(.5, .035, f"Physical pilot {self.times['pilot_parent_seconds']:.2f}s; saved audit {self.times['audit_parent_seconds']:.2f}s. "
                 "Separate phase costs, censoring, all conditions and every resample are supplied in the data files.", ha="center", fontsize=9)
        self.check()
        for extension in ("png", "svg"):
            path = self.out / ("query-advantage." + extension)
            require(not path.exists(), "exclusive figure")
            fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
            self.check()
        plt.close(fig)

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)

        def interrupt(_signum, _frame):
            raise InterruptedError("renderer time limit")

        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
        try:
            source = self.bind(Path(__file__).resolve())
            self.receipt["source"] = {"path": str(source), **self.bound[str(source)]}
            self.authenticate()
            summary, anchors, draws, statistics, limitations = self.data()
            self.figure(summary, anchors, draws)
            for name, records in (("anchors.csv", anchors), ("bootstrap.csv", draws),
                                  ("conditions.csv", summary["conditions"]), ("statistics.csv", statistics)):
                csv_write(self.out / name, records)
            costs = [{"metric": k, "seconds": v, "scope": "Full physical phase; do not add worker and parent."}
                     for k, v in self.times.items()]
            costs += [{"metric": k, "seconds": v, "scope": "Saved phase or overlapping I/O; see producer scope."}
                      for k, v in summary["costs"].items() if k.endswith("seconds") and type(v) in (int, float)]
            costs += [{"metric": "operation." + k, "seconds": v["seconds"],
                       "scope": "Nested instrumented work excluding measured I/O; not additive to phase wall."}
                      for k, v in summary["costs"]["operation_seconds"].items()]
            csv_write(self.out / "costs.csv", costs)
            counts = [{"metric": "audit." + k, "count": v} for k, v in summary["counts"].items()]
            counts += [{"metric": "producer." + k, "count": self.worker[k]} for k in
                       ("completed_episodes", "trajectory_rows", "available_anchors", "panels", "queries",
                        "external_annotations", "sampler_records", "sampler_events")]
            csv_write(self.out / "counts.csv", counts)
            write(self.out / "plotted-values.json", {"anchors": anchors, "bootstrap": draws, "statistics": statistics,
                "conditions": summary["conditions"], "lower_bounds": summary["lower_bounds"],
                "counts": summary["counts"], "case_contributions": summary["case_contributions"],
                "degenerate_resamples": summary["degenerate_resamples"], "phase_costs": summary["costs"],
                "physical_times": self.times, "limitations": limitations,
                "scatter_weighting": "One unjittered marker per anchor; admission statistics use saved episode-balanced weights."})
            for path, desc in self.bound.items():
                require(self.digest(Path(path)) == desc, "unchanged evidence and renderer")
            files = {p.name: self.digest(p) for p in self.out.iterdir()}
            self.check()
            finished = time.monotonic_ns()
            self.receipt.update(status="completed", files=files, inputs=self.bound, conditions_passed=summary["conditions_passed"],
                conditions_total=11, signal_admitted=summary["signal_admitted"], plotted_anchors=len(anchors),
                bootstrap_draws=len(draws), started_ns=self.start, finished_ns=finished,
                wall_seconds=(finished-self.start)/1e9)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.failed = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            self.receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.digest(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure and partial artifacts
                error.add_note(f"Failure receipt: {secondary!r}")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "audit", "audit-terminal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan", "worker", "terminal", "audit", "audit-terminal"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    require(all(getattr(args, k).is_absolute() for k in ("plan", "run", "terminal", "audit", "audit_terminal", "output")),
            "absolute paths required")
    Render(args).execute()


if __name__ == "__main__":
    main()
