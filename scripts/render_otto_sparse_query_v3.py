"""Render all prespecified sparse-query outcomes from closed, audited JSON.

Every source and payload is authenticated before outcome decoding. The original
failed reader and its closed process remain bound through a separate repair plan. No policy,
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
VERSION = "otto-sparse-query-render-v3"
PRODUCER = "scripts/study_otto_sparse_query.py"
ORIGINAL_AUDITOR = "scripts/audit_otto_sparse_query.py"
AUDITOR = "scripts/audit_otto_sparse_query_v2.py"
PRODUCER_VERSION = "otto-sparse-query-study-v1"
AUDIT_VERSION = "otto-sparse-query-saved-audit-v2"
REPAIR_VERSION = "otto-sparse-query-audit-repair-v2"
REPAIR_SOURCES = {AUDITOR, "tests/test_audit_otto_sparse_query_v2.py", ORIGINAL_AUDITOR,
                  "src/openjev/research/suspend_clock.py", "scripts/supervise_dialogue_observation_v2.py",
                  "research/otto-sparse-query-audit-repair.md"}
CORRECTION = ("Original saved audit failed on a cost-field type check. A separately frozen V2 reader corrected "
              "that check; the original failure is retained. No trajectory, model or scientific criterion was rerun or changed.")
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}
PAYLOADS = {name + ".jsonl.gz" for name in ("work", "weights", "forwards", "gate-operations",
    "gate-decisions", "transitions", "validation-decisions")} | {
    "started.json", "runtime.json", "setup.json", "deployment.json", "episode-boundaries.jsonl",
    "episodes.jsonl", "threshold.json", "costs.json", "summary.json"}
AUDIT_PAYLOADS = {"started.json", "audit.json"}
REGIMES = ("lambda3", "lambda4", "lambda5")
ARMS = ("analytic", "neural", "period2", "random_pair", "entropy")
LABELS = {"analytic": "Analytic", "neural": "Neural", "period2": "Period2 (primary)",
          "random_pair": "Random pair", "entropy": "Entropy"}
COLORS = {"analytic": "#444444", "neural": "#7a3db8", "period2": "#0072b2",
          "random_pair": "#d55e00", "entropy": "#009e73"}
METRICS = ("found", "steps", "queries", "init_seconds", "choose_seconds", "update_seconds",
           "setup_allocation_seconds", "controller_seconds", "paid_controller_seconds", "environment_seconds")


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
        require(script != PRODUCER or command[1] == str(ROOT / PRODUCER), "canonical absolute sparse-query producer")
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
        return command

    def repair(self, audit, plan):
        a = self.args
        repair = read(a.repair_plan)
        require(repair["version"] == REPAIR_VERSION and repair["status"] == "frozen_before_saved_arithmetic"
            and repair["limits"] == {"seconds": 120, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2}
            and set(repair["sources"]) == REPAIR_SOURCES
            and set(repair["inputs"]) == {"plan", "worker", "terminal", "failed_audit", "failed_audit_terminal"},
            "separately frozen bounded audit repair")
        require(audit["repair_plan"] == {"path": str(a.repair_plan), **self.bound[str(a.repair_plan)]}
            and audit["sources"] == repair["sources"]
            and repair["sources"][ORIGINAL_AUDITOR] == plan["sources"][ORIGINAL_AUDITOR]
            and repair["sources"][CLOCK] == CLOCK_PIN and repair["sources"][SUPERVISOR] == SUPERVISOR_PIN,
            "new reader and unchanged original reader identities")
        for name, pin in repair["sources"].items():
            self.bind(name, pin)
        for role, desc in repair["inputs"].items():
            require(set(desc) == {"path", "sha256", "bytes"} and Path(desc["path"]).is_absolute(), "repair input descriptor")
            self.bind(desc["path"], {k: desc[k] for k in ("sha256", "bytes")})
            if role in audit["producer_inputs"]:
                require(desc == audit["producer_inputs"][role], "repair uses same original scientific run")
        failure = {key: repair["inputs"][key] for key in ("failed_audit", "failed_audit_terminal")}
        require(audit["preserved_failure"] == failure, "closed original failure retained by corrected audit")
        old_path = Path(failure["failed_audit"]["path"])
        old = read(old_path)
        terminal = read(Path(failure["failed_audit_terminal"]["path"]))
        require(old["version"] == "otto-sparse-query-saved-audit-v1" and old["status"] == "failed"
            and old["agreement"] is False and old["failures"]
            and old["plan_sha256"] == a.plan_sha256 and old["producer_inputs"] == audit["producer_inputs"]
            and old["sources"] == {key: plan["sources"][key] for key in (ORIGINAL_AUDITOR, CLOCK, SUPERVISOR)},
            "original saved-reader failure and original producer binding")
        self.closed(old_path.parent, old, {"started.json"})
        require(terminal["status"] == "failed" and terminal["returncode"] == 1
            and terminal["timed_out"] is False and terminal["error"] is terminal["clock_error"] is None
            and terminal["group_absent"] is terminal["cleanup"]["group_absent"] is terminal["cleanup"]["reaped"] is True
            and terminal["cleanup"]["errors"] == [] and terminal["cap_seconds"] == 120
            and terminal["deadline_ns"] == terminal["started_ns"] + 120 * 10**9
            and terminal["started_ns"] < terminal["finished_ns"] <= terminal["deadline_ns"]
            and terminal["clock_source_sha256"] == CLOCK_PIN and terminal["watchdog_sha256"] == SUPERVISOR_PIN
            and Path(terminal["cwd"]) == ROOT, "original failed audit process closed without timeout")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        named = Path(command[1])
        require(command[0] == str(ROOT / ".venv/bin/python")
            and (named if named.is_absolute() else ROOT / named) == ROOT / ORIGINAL_AUDITOR,
            "original failed reader command")
        expected = {"--plan": str(a.plan), "--plan-sha256": a.plan_sha256,
            "--worker": str(a.run / "receipt.json"), "--worker-sha256": a.worker_sha256,
            "--terminal": str(a.terminal), "--terminal-sha256": a.terminal_sha256,
            "--output": str(old_path.parent)}
        require(len(command[2:]) == 2 * (len(expected) + 1), "original failed reader complete arguments")
        options = dict(zip(command[2::2], command[3::2], strict=True))
        require(set(options) == set(expected) | {"--supervision"} and all(options[k] == v for k, v in expected.items()),
            "original failed reader exact scientific inputs")
        launch = read(self.bind(options["--supervision"], old["supervision_sha256"]))
        started = read(old_path.parent / "started.json")
        require(launch == started["launch"] and all(terminal[k] == v for k, v in launch.items())
            and launch["pid"] == launch["pgid"] != launch["parent_pid"]
            and terminal["started_ns"] <= started["started_ns"] <= terminal["finished_ns"],
            "original failed reader launch and preserved start")
        same(terminal["wall_seconds"], (terminal["finished_ns"] - terminal["started_ns"]) / 1e9)
        self.correction = {"note": CORRECTION, "repair_plan": audit["repair_plan"], "preserved_failure": failure,
                           "original_audit_parent_seconds": terminal["wall_seconds"]}

    def authenticate(self):
        a = self.args
        for path, pin in ((a.plan, a.plan_sha256), (a.run / "receipt.json", a.worker_sha256),
                         (a.terminal, a.terminal_sha256), (a.audit / "receipt.json", a.audit_sha256),
                         (a.audit_terminal, a.audit_terminal_sha256), (a.repair_plan, a.repair_plan_sha256)):
            self.bind(path, pin)
        plan, worker, terminal = read(a.plan), read(a.run / "receipt.json"), read(a.terminal)
        audit, audit_terminal = read(a.audit / "receipt.json"), read(a.audit_terminal)
        require(plan["version"] == worker["version"] == PRODUCER_VERSION
            and plan["status"] == "frozen_before_validation" and worker["status"] == "completed"
            and worker["complete"] is True and worker["completed_episodes"] == 384
            and worker["validation_episodes"] == 24 and worker["evaluation_episodes"] == 360
            and worker["pending"] == [] and worker["pending_episode"] is worker["pending_emission"] is None
            and not worker.get("cleanup_errors") and worker["training_updates"] == worker["annotations"] == 0
            and worker["plan_sha256"] == a.plan_sha256 and worker["requires_successful_original_supervisor"] is True,
            "completed full sparse-query worker")
        same(plan["configuration"], {"arms": list(ARMS), "primary_arm": "period2", "required_criteria": 16,
            "validation_episodes": 24, "evaluation_episodes": 360, "horizon": 2188, "training_updates": 0, "annotations": 0})
        require(audit["version"] == AUDIT_VERSION and audit["status"] == "completed"
            and audit["agreement"] is True and audit["plan_sha256"] == a.plan_sha256
            and plan["sources"][CLOCK] == CLOCK_PIN and plan["sources"][SUPERVISOR] == SUPERVISOR_PIN
            and PRODUCER in plan["sources"] and worker["sources"] == plan["sources"],
            "frozen independent auditor and original producer identities")
        expected_inputs = {"plan": a.plan, "worker": a.run / "receipt.json", "terminal": a.terminal}
        require(set(audit["producer_inputs"]) == set(expected_inputs), "all audit producer identities")
        for role, path in expected_inputs.items():
            require(audit["producer_inputs"][role] == {"path": str(path), **self.bound[str(path)]},
                    "exact audit external input join")
        self.repair(audit, plan)
        for name, pin in plan["sources"].items():
            self.bind(name, pin)
        for key in ("inputs", "native_inputs"):
            require(worker[key] == plan[key], "complete frozen input lineage")
            for desc in plan[key].values():
                self.bind(desc["path"], {k: desc[k] for k in ("sha256", "bytes")})
        require(set(plan["payloads"]) == PAYLOADS, "exact frozen payload set")
        self.closed(a.run, worker, PAYLOADS)
        self.closed(a.audit, audit, AUDIT_PAYLOADS)
        self.parent(worker, terminal, a.run, PRODUCER, 1800,
            {"--plan": a.plan, "--plan-sha256": a.plan_sha256, "--output": a.run})
        self.parent(audit, audit_terminal, a.audit, AUDITOR, 120,
            {"--plan": a.plan, "--plan-sha256": a.plan_sha256,
             "--worker": a.run / "receipt.json", "--worker-sha256": a.worker_sha256,
             "--terminal": a.terminal, "--terminal-sha256": a.terminal_sha256, "--output": a.audit,
             "--repair-plan": a.repair_plan, "--repair-plan-sha256": a.repair_plan_sha256})
        require(audit["counts"]["episodes"] == 384 and audit["counts"]["paired_cases"] == 72
            and audit["counts"]["payloads"] == len(PAYLOADS)
            and audit["counts"]["source_files"] == len(plan["sources"])
            and audit["counts"]["forwards"] == worker["calls"]["tensorflow_value"]["returned"],
            "complete independent counts")
        self.worker, self.audit_receipt = worker, audit
        self.times = {"study_worker_seconds": worker["wall_seconds"], "study_parent_seconds": terminal["wall_seconds"],
                      "audit_worker_seconds": audit["wall_seconds"], "audit_parent_seconds": audit_terminal["wall_seconds"],
                      "original_failed_audit_parent_seconds": self.correction["original_audit_parent_seconds"]}

    def data(self):
        original = read(self.args.run / "summary.json")
        audited = read(self.args.audit / "audit.json")
        require(audited["version"] == AUDIT_VERSION and audited["agreement"] is True, "saved audit agreement")
        # The independently joined entire producer summary supplies every displayed value.
        require(set(original) == set(audited["summary"]), "complete audited summary field coverage")
        same(original, audited["summary"])
        summary = original
        require(summary["version"] == PRODUCER_VERSION and summary["primary_arm"] == "period2"
            and summary["episodes"] == 360 and summary["paired_cases"] == 72
            and summary["validation_episodes"] == 24 and summary["evaluation_episodes"] == 360
            and set(summary["regimes"]) == set(REGIMES), "complete prespecified evaluation")
        same(summary["threshold"], read(self.args.run / "threshold.json"))
        same(summary["costs"], read(self.args.run / "costs.json"))
        same(summary["deployment"], read(self.args.run / "deployment.json"))
        require(self.worker["threshold"] == self.bound[str(self.args.run / "threshold.json")], "fixed threshold identity")
        conditions = []
        for group, count in (("required", 16), ("diagnostic", 54)):
            local = summary[group]
            require(len(local) == len({c["name"] for c in local}) == count, "every declared condition")
            for c in local:
                require(c["relation"] in (">=", "<=") and math.isfinite(c["value"])
                    and math.isfinite(c["threshold"]) and type(c["passes"]) is bool
                    and c["passes"] == (c["value"] >= c["threshold"] if c["relation"] == ">="
                                        else c["value"] <= c["threshold"]), "saved condition arithmetic")
                conditions.append({"group": group, **c})
            require(summary[group + "_conditions"] == count
                and summary[group + "_passed"] == sum(c["passes"] for c in local), "complete condition totals")
        require(summary["pilot_continuation"] is all(c["passes"] for c in summary["required"]),
                "fixed primary continuation rule")
        metrics, blocks = [], []
        for regime in REGIMES:
            local = summary["regimes"][regime]
            require(set(local["means"]) == set(local["raw_counts"]) == set(ARMS)
                and len(local["blocks"]) == 8, "all five arms and eight blocks")
            for block, values in [(None, local["means"]), *enumerate(local["blocks"])]:
                require(set(values) == set(ARMS), "complete block arm coverage")
                for arm in ARMS:
                    value = values[arm]
                    require(set(value) == set(METRICS) and all(type(value[k]) in (int, float)
                        and math.isfinite(value[k]) and value[k] >= 0 for k in METRICS)
                        and 0 <= value["found"] <= 1 + 1e-12 and 1 <= value["steps"] <= 2188
                        and value["queries"] <= value["steps"], "finite complete unfiltered metric cell")
                    row = {"regime": regime, "arm": arm, **value,
                           "success_percent": 100 * value["found"],
                           "query_fraction": value["queries"] / value["steps"]}
                    if block is None:
                        require(local["raw_counts"][arm]["episodes"] == 24, "all twenty-four paired cases per cell")
                        row.update({"raw_" + k: v for k, v in local["raw_counts"][arm].items()})
                        metrics.append(row)
                    else:
                        blocks.append({"block": block, **row})
        require(len(metrics) == 15 and len(blocks) == 120, "all plotting and block records")
        return summary, metrics, blocks, conditions, audited["limitations"]

    def figure(self, summary, metrics):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.985, bottom=.345, top=.735, wspace=.13)
        x_values = [r["paid_controller_seconds"] for r in metrics]
        x_values += [r["controller_seconds"] for r in metrics if r["arm"] == "entropy"]
        log_x = min(x_values) > 0
        low, high = (min(x_values) / 1.8, max(x_values) * 2.4) if log_x else (0., max(1., max(x_values)) * 1.3)
        ymax = max(r["steps"] for r in metrics) * 1.22
        for regime, ax in zip(REGIMES, axes, strict=True):
            for row in (r for r in metrics if r["regime"] == regime):
                arm, y = row["arm"], row["steps"]
                if arm == "entropy":
                    ax.plot([row["controller_seconds"], row["paid_controller_seconds"]], [y, y],
                            color=COLORS[arm], linewidth=1., alpha=.7)
                    ax.scatter([row["controller_seconds"]], [y], marker="o", s=48,
                               facecolors="none", edgecolors=COLORS[arm], linewidths=1.2)
                ax.scatter([row["paid_controller_seconds"]], [y], color=COLORS[arm],
                           marker="*" if arm == "period2" else "o", s=145 if arm == "period2" else 45, zorder=3)
            if log_x:
                ax.set_xscale("log")
            ax.set(xlim=(low, high), ylim=(0, ymax),
                   xlabel="Fully paid controller seconds / episode" + (" (log scale)" if log_x else ""),
                   title=f"Sensing length {regime[-1]}: {'primary setting' if regime != 'lambda5' else 'transfer diagnostic'}")
            ax.grid(axis="both", alpha=.17)
        axes[0].set_ylabel("Hit-mixture-weighted capped moves / episode")
        status = "PASS" if summary["pilot_continuation"] else "FAIL"
        fig.suptitle(f"Sparse query control: {status}, {summary['required_passed']}/16 primary conditions", fontsize=16, y=.97)
        fig.text(.5, .91, "360 fresh EVAL paths; five arms in every setting; 24 paired cases per arm/setting; full horizon 2,188", ha="center")
        handles = [Line2D([], [], linestyle="none", marker="*" if arm == "period2" else "o",
                          markersize=12 if arm == "period2" else 7, color=COLORS[arm], label=LABELS[arm])
                   for arm in ARMS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .865), ncol=5,
                   frameon=False, columnspacing=2.5, handletextpad=.6)
        fig.text(.5, .17, "Lower and left is better. Every arm retained; common axes. Star: sole primary candidate, period2.\n"
                 "Entropy: open marker is online cost; filled marker adds its full VALID calibration bill, allocated over all 72 entropy paths.\n"
                 "All 15 cells, 120 block rows, 16 primary and 54 diagnostic conditions are exported. Diagnostics cannot replace a failed primary.",
                 ha="center", fontsize=9)
        fig.text(.5, .085, f"Physical study {self.times['study_parent_seconds']:.2f}s; saved audit {self.times['audit_parent_seconds']:.2f}s. "
                 "Fixed schedules test a control opportunity, not learned recurrence or architecture efficacy.", ha="center", fontsize=9)
        fig.text(.5, .025, "Audit correction: original reader failed on a cost-field type check; separately frozen V2 reader passed.\n"
                 "Original failure retained. Scientific trajectories and criteria unchanged; no model or native rerun.",
                 ha="center", fontsize=8)
        self.check()
        for suffix in ("png", "svg"):
            path = self.out / ("sparse-query." + suffix)
            require(not path.exists(), "exclusive figure")
            fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
            self.check()
        plt.close(fig)
        return {"x_scale": "log" if log_x else "linear", "common_xlim": [low, high], "common_ylim": [0, ymax],
                "all_paid_cells": 15, "additional_entropy_online_markers": 3}

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
            summary, metrics, blocks, conditions, limitations = self.data()
            axes = self.figure(summary, metrics)
            for name, records in (("metrics.csv", metrics), ("blocks.csv", blocks), ("conditions.csv", conditions)):
                csv_write(self.out / name, records)
            costs = [{"metric": k, "seconds": v, "scope": "Full physical phase; do not add worker and parent."}
                     for k, v in self.times.items()]
            costs += [{"metric": k, "seconds": v, "scope": "Saved phase, allocation or overlapping I/O; see original cost scope."}
                      for k, v in summary["costs"].items() if k.endswith("seconds") and type(v) in (int, float)]
            costs += [{"metric": "operation." + k, "seconds": v["seconds"],
                       "scope": "Nested instrumented work excluding measured I/O; not additive to physical phases."}
                      for k, v in summary["costs"]["operation_seconds"].items()]
            csv_write(self.out / "costs.csv", costs)
            counts = [{"metric": "audit." + k, "count": v} for k, v in self.audit_receipt["counts"].items()]
            counts += [{"metric": "producer." + k + "." + stage, "count": value[stage]}
                       for k, value in self.worker["calls"].items() for stage in ("attempted", "returned")]
            csv_write(self.out / "counts.csv", counts)
            write(self.out / "plotted-values.json", {"metrics": metrics, "blocks": blocks, "conditions": conditions,
                "threshold": summary["threshold"], "costs": summary["costs"], "deployment": summary["deployment"],
                "physical_times": self.times, "counts": self.audit_receipt["counts"], "axes": axes,
                "mixture_weights": {r: summary["regimes"][r]["weights"] for r in REGIMES},
                "primary_arm": "period2", "required_passed": summary["required_passed"],
                "required_conditions": 16, "diagnostic_passed": summary["diagnostic_passed"],
                "diagnostic_conditions": 54, "pilot_continuation": summary["pilot_continuation"],
                "query_fraction_scope": "Ratio of weighted mean queries to weighted mean moves; not mean per-path query fraction.",
                "audit_correction": self.correction, "limitations": limitations})
            for path, desc in self.bound.items():
                require(self.digest(Path(path)) == desc, "unchanged evidence and renderer")
            files = {p.name: self.digest(p) for p in self.out.iterdir()}
            self.check()
            finished = time.monotonic_ns()
            self.receipt.update(status="completed", files=files, inputs=self.bound, audit_correction=self.correction, plotted_cells=15, block_rows=120,
                required_conditions=16, required_passed=summary["required_passed"], diagnostic_conditions=54,
                diagnostic_passed=summary["diagnostic_passed"], pilot_continuation=summary["pilot_continuation"],
                started_ns=self.start, finished_ns=finished, wall_seconds=(finished-self.start)/1e9)
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
            except BaseException as secondary:  # noqa: BLE001 - retain original failure and partial artifacts
                error.add_note(f"Failure receipt: {secondary!r}")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "audit", "audit-terminal", "repair-plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan", "worker", "terminal", "audit", "audit-terminal", "repair-plan"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    require(all(getattr(args, k).is_absolute() for k in ("plan", "run", "terminal", "audit", "audit_terminal", "repair_plan", "output")),
            "absolute paths required")
    Render(args).execute()


if __name__ == "__main__":
    main()
