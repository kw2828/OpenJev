"""Present authenticated completed query-gate results, without scientific replay.

Only saved JSON is decoded. Checkpoints, arrays and journals are opaque hashed
payloads; no producer, model, optimizer or simulator module is imported.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-query-gate-render-v1"
SEEDS, KINDS = (40101, 40102, 40103), ("gru32", "mlp190")
ARMS = tuple(f"{kind}@{seed}" for seed in SEEDS for kind in KINDS) + ("neural", "analytic")
REGIMES = ("lambda3", "lambda4", "lambda5")
GROUPS = {"gru_absolute": 24, "mlp_absolute": 24, "usefulness": 4, "recurrence": 6, "reference": 4}
LABELS = {"gru32": "GRU32", "mlp190": "MLP190", "neural": "Always neural", "analytic": "Analytic"}
COLORS = {"gru32": "#2166ac", "mlp190": "#d97706", "neural": "#7a3db8", "analytic": "#222222"}
BASE = {"started.json", "runtime.json", "setup.json", "weights.jsonl", "forwards.jsonl", "work.jsonl",
        "gate-operations.jsonl", "episodes.jsonl", "summary.json"}
PAYLOADS = {
    "native": BASE | {"qualification-costs.json", "features.jsonl", "checks.jsonl"},
    "collection": BASE | {"public-transitions.jsonl", "training-data.npz", "collection-costs.json"},
    "training": {"started.json", "runtime.json", "data.json", "work.jsonl", "orders.jsonl", "epochs.jsonl",
                 "parity.jsonl", "fits.jsonl", "summary.json"}
                | {f"{kind}-{seed}{suffix}" for seed in SEEDS for kind in KINDS
                   for suffix in ("-initial.npz", ".npz", "-parity.npz")},
    "evaluation": {"started.json", "runtime.json", "setup.json", "deployment.json", "episodes.jsonl", "summary.json"}
                  | {name + ".gz" for name in ("work.jsonl", "weights.jsonl", "forwards.jsonl",
                     "gate-operations.jsonl", "gate-decisions.jsonl", "transitions.jsonl")},
}
SCRIPTS = {"native": "qualify_otto_query_gate_native.py", "collection": "collect_otto_query_gate.py",
           "training": "train_otto_query_gate.py", "evaluation": "evaluate_otto_query_gate.py"}
VERSIONS = {"native": "otto-query-gate-native-qualification-v1", "collection": "otto-query-gate-collection-v1",
            "training": "otto-query-gate-training-v1", "evaluation": "otto-query-gate-evaluation-v1"}
CAPS = {"native": 180, "collection": 600, "training": 300, "evaluation": 3600}
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 64 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Render:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.start = time.monotonic_ns()
        self.bound, self.stages = {}, {}
        self.receipt = {"version": VERSION, "status": "started", "limits": LIMITS,
                        "model_calls": 0, "optimizer_calls": 0, "simulator_calls": 0,
                        "array_decodes": 0, "scope": "Saved JSON presentation and byte authentication only; no independent numerical replay."}

    def check(self):
        require(time.monotonic_ns() - self.start < LIMITS["seconds"] * 10**9, "render time cap")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir()) <= LIMITS["output_bytes"],
                "render RSS/output bounds")

    def digest(self, path):
        require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
        result, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                self.check()
                result.update(block)
                size += len(block)
        return {"sha256": result.hexdigest(), "bytes": size}

    def bind(self, path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        value = self.bound.get(str(path))
        if value is None:
            value = self.digest(path)
            self.bound[str(path)] = value
        if isinstance(expected, str):
            require(value["sha256"] == expected, "external SHA256 pin")
        elif expected is not None:
            require(value == expected, "exact input descriptor")
        return path

    def record(self, descriptor):
        require(set(descriptor) == {"path", "sha256", "bytes"}, "source-bound role descriptor")
        return self.bind(descriptor["path"], {k: descriptor[k] for k in ("sha256", "bytes")})

    def stage(self, stage, plan_path, receipt_path, terminal_path):
        plan, worker, terminal = read(plan_path), read(receipt_path), read(terminal_path)
        require(plan["version"] == worker["version"] == VERSIONS[stage]
                and worker["status"] == terminal["status"] == "completed" and worker["pending"] == []
                and terminal["returncode"] == 0 and terminal["timed_out"] is False
                and terminal["group_absent"] is terminal["cleanup"]["group_absent"] is terminal["cleanup"]["reaped"] is True
                and terminal["error"] is terminal["clock_error"] is None and terminal["cleanup"]["errors"] == []
                and terminal["cap_seconds"] == CAPS[stage], f"completed original {stage} phase")
        require(worker["plan_sha256"] == self.bound[str(plan_path)]["sha256"]
                and worker["sources"] == plan["sources"], f"{stage} plan/source identity")
        roles_key = "roles" if stage == "training" else "inputs"
        require(worker[roles_key] == plan[roles_key], f"{stage} input identity")
        require(set(worker["files"]) == PAYLOADS[stage] == set(plan["payloads"])
                and {p.name for p in receipt_path.parent.iterdir()} == PAYLOADS[stage] | {"receipt.json"},
                f"{stage} complete exact payload closure")
        for name, value in worker["files"].items():
            self.bind(receipt_path.parent / name, value)
        for name, pin in plan["sources"].items():
            self.bind(ROOT / name, pin)
        if stage == "training":
            for name, value in plan["inputs"].items():
                self.bind(ROOT / name, value)
        else:
            for value in plan["inputs"].values():
                self.record(value)
        for key in ("native_inputs", "checkpoints"):
            for value in plan.get(key, {}).values():
                self.record(value)
        started = read(receipt_path.parent / "started.json")
        launch_path = self.bind(started["request"]["supervision"], worker["supervision_sha256"])
        launch = read(launch_path)
        require(launch == started["launch"] and all(terminal[key] == value for key, value in launch.items()),
                f"{stage} actual launch/terminal join")
        command = list(terminal["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        interpreter = ".venv/bin/python" if stage == "training" else ".venv-otto-released-native/bin/python"
        require(len(command) == 11 and command[0] == str(ROOT / interpreter)
                and command[1:3] == ["scripts/" + SCRIPTS[stage], "run"], f"{stage} exact producer invocation")
        options = dict(zip(command[3::2], command[4::2], strict=True))
        require(options == {"--plan": str(plan_path), "--plan-sha256": self.bound[str(plan_path)]["sha256"],
                            "--supervision": str(launch_path), "--output": str(receipt_path.parent)},
                f"{stage} actual input/output command")
        require(terminal["started_ns"] <= worker["started_ns"] < worker["finished_ns"] <= terminal["finished_ns"]
                <= terminal["deadline_ns"] and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9,
                f"{stage} bounded physical time")
        self.stages[stage] = {"plan": plan, "worker": worker, "terminal": terminal,
                              "directory": receipt_path.parent, "plan_path": plan_path, "receipt_path": receipt_path}
        return plan

    def authenticate(self):
        a = self.args
        self.bind(a.plan, a.plan_sha256)
        self.bind(a.run / "receipt.json", a.worker_sha256)
        self.bind(a.terminal, a.terminal_sha256)
        plan = self.stage("evaluation", a.plan, a.run / "receipt.json", a.terminal)
        for stage in ("native", "collection", "training"):
            paths = {part: self.record(plan["inputs"][f"{stage}_{part}"]) for part in ("plan", "receipt", "terminal")}
            if stage != "native":
                self.bind(paths["receipt"], getattr(a, stage + "_worker_sha256"))
                self.bind(paths["terminal"], getattr(a, stage + "_terminal_sha256"))
            self.stage(stage, paths["plan"], paths["receipt"], paths["terminal"])
        require(self.stages["evaluation"]["worker"]["completed_episodes"] == 576
                and self.stages["evaluation"]["worker"]["pending_episode"] is None
                and self.stages["collection"]["worker"]["complete"] is True
                and self.stages["collection"]["worker"]["completed_episodes"] == 60
                and self.stages["native"]["worker"]["qualified"] is True
                and self.stages["native"]["worker"]["completed_episodes"] == 18
                and self.stages["training"]["worker"]["qualified"] is True
                and self.stages["training"]["worker"]["completed_fits"] == 6
                and self.stages["training"]["worker"]["all_train_parity_passed"] is True,
                "complete physical collection, six qualified fits and 576 evaluation paths")
        training = self.stages["training"]
        require(training["plan"]["roles"]["collection_receipt"]["sha256"] == a.collection_worker_sha256,
                "all fits consume this collection")
        self.authenticate_audit()
        # Outcome JSON decoding begins only after every phase/input/source closure.
        summary = read(a.run / "summary.json")
        audited = read(a.audit / "audit.json")
        require(audited["agreement"] is True and set(audited["summary"]) == {
            "regimes", "criteria", "required_conditions", "required_passed", "reported_conditions",
            "reported_passed", "pilot_continuation", "deployment"}
            and all(summary[key] == value for key, value in audited["summary"].items()),
            "every displayed evaluation field agrees with the independent saved audit")
        self.receipt["audit_limitations"] = audited["limitations"]
        train_summary = read(training["directory"] / "summary.json")
        require(summary["version"] == VERSIONS["evaluation"] and summary["episodes"] == 576
                and summary["paired_cases"] == 72 and set(summary["regimes"]) == set(REGIMES), "complete saved evaluation")
        require([r["fit_id"] for r in train_summary["fits"]] == list(ARMS[:-2]), "every final fit included")
        conditions = []
        require(set(summary["criteria"]) == set(GROUPS), "exact prespecified criterion groups")
        for group, count in GROUPS.items():
            rows = summary["criteria"][group]
            require(len(rows) == count, "every prespecified condition")
            for row in rows:
                require(math.isfinite(row["value"]) and math.isfinite(row["threshold"])
                        and row["relation"] in (">=", "<="), "finite declared comparison")
                passed = row["value"] >= row["threshold"] if row["relation"] == ">=" else row["value"] <= row["threshold"]
                require(type(row["passes"]) is bool and row["passes"] == passed, "saved Boolean arithmetic")
                conditions.append({"group": group, "required": group != "mlp_absolute", **row})
        required = [r for r in conditions if r["required"]]
        require(len({r["name"] for r in conditions}) == 62 and len(required) == 38
                and summary["reported_conditions"] == 62 and summary["required_conditions"] == 38
                and summary["reported_passed"] == sum(r["passes"] for r in conditions)
                and summary["required_passed"] == sum(r["passes"] for r in required)
                and summary["pilot_continuation"] == all(r["passes"] for r in required), "all62 and required38 summaries")
        rows = []
        for regime in REGIMES:
            local = summary["regimes"][regime]
            require(set(local["means"]) == set(ARMS) and set(local["family_means"]) == set(KINDS), "all model/control cells")
            for arm, values in [*local["means"].items(), *[(k + "_mean", v) for k, v in local["family_means"].items()]]:
                require(all(math.isfinite(values[m]) and values[m] >= 0 for m in ("found", "steps", "queries", "controller_seconds"))
                        and values["found"] <= 1 + 1e-12 and values["steps"] > 0
                        and values["queries"] <= values["steps"], "finite plot cell without clipping")
                rows.append({"regime": regime, "arm": arm, "level": "family_mean" if arm.endswith("_mean") else "individual",
                             "success_percent": 100 * values["found"], "capped_moves": values["steps"],
                             "controller_seconds": values["controller_seconds"], "weighted_queries": values["queries"],
                             "query_fraction": values["queries"] / values["steps"]})
        return summary, train_summary, rows, conditions

    def authenticate_audit(self):
        a = self.args
        self.bind(a.audit / "receipt.json", a.audit_sha256)
        self.bind(a.audit_terminal, a.audit_terminal_sha256)
        receipt, terminal = read(a.audit / "receipt.json"), read(a.audit_terminal)
        require(receipt["version"] == "otto-query-gate-saved-audit-v1"
                and receipt["status"] == "completed" and receipt["agreement"] is True
                and terminal["status"] == "completed" and terminal["returncode"] == 0
                and terminal["timed_out"] is False and terminal["group_absent"] is True
                and terminal["cleanup"]["group_absent"] is terminal["cleanup"]["reaped"] is True
                and terminal["cleanup"]["errors"] == [] and terminal["error"] is terminal["clock_error"] is None
                and terminal["cap_seconds"] == 120, "completed original independent audit")
        expected = {"plan": a.plan, "worker": a.run / "receipt.json", "terminal": a.terminal}
        require(set(receipt["producer_inputs"]) == set(expected), "exact audited producer inputs")
        for key, path in expected.items():
            require(self.record(receipt["producer_inputs"][key]) == path, "audit identity matches rendered run")
        require(set(receipt["files"]) == {"started.json", "audit.json"}
                and {p.name for p in a.audit.iterdir()} == {"started.json", "audit.json", "receipt.json"}, "complete audit closure")
        for name, value in receipt["files"].items():
            self.bind(a.audit / name, value)
        command = list(terminal["command"])
        require("--plan" in command and "--plan-sha256" in command and "--supervision" in command
                and "--output" in command, "audited original command")
        self.bind(Path(command[command.index("--plan") + 1]), receipt["plan_sha256"])
        require(command[command.index("--plan-sha256") + 1] == receipt["plan_sha256"]
                and Path(command[command.index("--output") + 1]) == a.audit, "audit plan/output joins")
        launch_path = self.bind(Path(command[command.index("--supervision") + 1]), receipt["supervision_sha256"])
        launch = read(launch_path)
        require(all(terminal[key] == value for key, value in launch.items())
                and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"] <= terminal["finished_ns"]
                <= terminal["deadline_ns"], "original audit parent/worker chronology")
        self.receipt["audit_receipt_sha256"] = a.audit_sha256

    def figures(self, summary, rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        by = {(r["regime"], r["arm"]): r for r in rows}
        marks = dict(zip(SEEDS, ("o", "s", "^"), strict=True))
        legend = [Line2D([], [], color=COLORS[k], marker="o", linestyle="none", label=LABELS[k]) for k in KINDS]
        legend += [Line2D([], [], color=COLORS[k], marker=m, linestyle="none", label=LABELS[k])
                   for k, m in (("neural", "X"), ("analytic", "P"))]
        legend += [Line2D([], [], color="#777777", marker=marks[s], linestyle="none", label=str(s)) for s in SEEDS]
        legend.append(Line2D([], [], color="#777777", marker="D", markerfacecolor="white", linestyle="none", label="Family mean"))
        status = f"{'PASS' if summary['pilot_continuation'] else 'FAIL'}: {summary['required_passed']}/38 required; {summary['reported_passed']}/62 reported"
        maximum_seconds = max(r["controller_seconds"] for r in rows)
        maximum_moves = max(r["capped_moves"] for r in rows)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharex=True, sharey=True)
        fig.subplots_adjust(left=.075, right=.985, bottom=.28, top=.78, wspace=.15)
        for regime, ax in zip(REGIMES, axes, strict=True):
            for arm in ARMS:
                row = by[regime, arm]
                kind = arm.split("@")[0]
                marker = marks[int(arm.split("@")[1])] if "@" in arm else "X" if arm == "neural" else "P"
                ax.scatter(row["controller_seconds"], row["capped_moves"], color=COLORS[kind], marker=marker,
                           s=58 if "@" in arm else 105, alpha=.85, edgecolors="white", linewidths=.6, zorder=3)
            for kind in KINDS:
                row = by[regime, kind + "_mean"]
                ax.scatter(row["controller_seconds"], row["capped_moves"], edgecolors=COLORS[kind], facecolors="none",
                           marker="D", s=145, linewidths=1.8, zorder=4)
            ax.set_title(f"Sensing length {regime[-1]}\n{'Transfer, descriptive' if regime == 'lambda5' else 'Primary'}", fontsize=11)
            ax.set_xlabel("Complete controller seconds / search")
            ax.set_xlim(0, max(1e-9, maximum_seconds * 1.08))
            ax.set_ylim(0, maximum_moves * 1.08)
            ax.grid(alpha=.18)
            if regime == "lambda5":
                ax.set_facecolor("#f3f3f3")
        axes[0].set_ylabel("Capped moves / search (lower is better)")
        fig.suptitle("Query allocation: search quality versus computation", fontsize=17, y=.98)
        fig.text(.5, .87, f"All six fixed fits and both controls | 576 searches | {status}", ha="center", fontsize=10.5)
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .13), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, .075, "Same axis scales across settings. Each point is a saved hit-mixture-weighted mean; no selected frontier or seed.", ha="center", fontsize=9)
        fig.text(.5, .025, "Controller time includes filters, features, gate, requested planner and fresh setup allocation. TRAIN costs are separate; one timing pass.", ha="center", fontsize=8.5)
        self.save_figure(fig, "query-gate-tradeoff")
        plt.close(fig)
        metrics = (("success_percent", "Source found (%)"), ("capped_moves", "Capped moves"),
                   ("controller_seconds", "Controller seconds"), ("query_fraction", "Queries / moves"))
        fig, axes = plt.subplots(4, 3, figsize=(14, 11.7), sharex=True, sharey="row")
        fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.9, hspace=.32, wspace=.15)
        labels = ("G1", "M1", "G2", "M2", "G3", "M3", "Neural", "Analytic")
        for col, regime in enumerate(REGIMES):
            for row_index, (metric, label) in enumerate(metrics):
                ax = axes[row_index, col]
                for index, arm in enumerate(ARMS):
                    kind = arm.split("@")[0]
                    marker = marks[int(arm.split("@")[1])] if "@" in arm else "X" if arm == "neural" else "P"
                    ax.scatter(index, by[regime, arm][metric], color=COLORS[kind], marker=marker, s=45)
                ax.set_ylim(0, max(1e-9, max(r[metric] for r in rows) * 1.08))
                if metric in ("success_percent", "query_fraction"):
                    ax.set_ylim(0, 105 if metric == "success_percent" else 1.05)
                ax.grid(axis="y", alpha=.18)
                ax.set_xticks(range(8), labels, rotation=40, ha="right")
                if col == 0:
                    ax.set_ylabel(label)
                if row_index == 0:
                    ax.set_title(f"Length {regime[-1]}: {'transfer' if col == 2 else 'primary'}")
                if col == 2:
                    ax.set_facecolor("#f3f3f3")
        fig.suptitle("All fixed controllers, every setting", fontsize=17, y=.98)
        fig.text(.5, .94, status, ha="center", fontsize=11)
        fig.text(.5, .07, "G = GRU32; M = MLP190; 1/2/3 = seeds 40101/40102/40103. Per-row axis scales are shared across settings.", ha="center", fontsize=9)
        fig.text(.5, .035, "Queries / moves is a ratio of weighted means, not the mean episode query fraction. Transfer is outside the 38-condition rule.", ha="center", fontsize=9)
        self.save_figure(fig, "query-gate-metrics")
        plt.close(fig)
        return matplotlib.__version__

    def save_figure(self, figure, name):
        self.check()
        for suffix in ("png", "svg"):
            target = self.out / f"{name}.{suffix}"
            with target.open("xb") as stream:
                figure.savefig(stream, format=suffix, dpi=180, bbox_inches="tight")
                stream.flush()
                os.fsync(stream.fileno())
            self.check()

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        try:
            source = self.digest(Path(__file__).resolve())
            self.receipt["source"] = source
            summary, training, rows, conditions = self.authenticate()
            mpl = self.figures(summary, rows)
            for name, records in (("controller-values.csv", rows), ("criteria.csv", conditions)):
                with (self.out / name).open("x", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            phase_costs = {name: {"worker_seconds": stage["worker"]["wall_seconds"],
                                 "parent_seconds": stage["terminal"]["wall_seconds"]} for name, stage in self.stages.items()}
            write(self.out / "plotted-values.json", {"rows": rows, "criteria": conditions, "fits": training["fits"],
                  "phase_costs": phase_costs, "deployment": summary["deployment"], "pilot_continuation": summary["pilot_continuation"],
                  "scope": "All24 arm/setting cells, six equal-seed family means and all62 conditions. No model, array or trajectory replay.",
                  "query_fraction": "queries / steps using each saved weighted mean; family points use family means"})
            for name, value in self.bound.items():
                require(self.digest(Path(name)) == value, "unchanged presentation input")
            require(self.digest(Path(__file__).resolve()) == source, "unchanged renderer source")
            files = {p.name: self.digest(p) for p in self.out.iterdir()}
            self.check()
            self.receipt.update(status="completed", inputs=self.bound, files=files, matplotlib_version=mpl,
                                individual_cells=24, family_cells=6, metric_points=96, tradeoff_points=30,
                                reported_conditions=62, required_conditions=38,
                                required_passed=summary["required_passed"], reported_passed=summary["reported_passed"],
                                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": self.digest(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", error=repr(error), inputs=self.bound,
                                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            try:
                receipt = self.out / "receipt.json"
                if receipt.exists():
                    receipt.rename(self.out / "receipt.invalid.json")
                write(receipt, self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve the original failure if publication also fails
                error.add_note(f"Failed receipt: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "run", "terminal", "audit", "audit-terminal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("plan", "worker", "terminal", "collection-worker", "collection-terminal", "training-worker", "training-terminal",
                 "audit", "audit-terminal"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    require(all(getattr(args, name).is_absolute() for name in ("plan", "run", "terminal", "audit", "audit_terminal", "output")),
            "absolute evidence/output paths")
    Render(args).execute()


if __name__ == "__main__":
    main()
