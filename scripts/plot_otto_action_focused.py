"""Present all fixed action-focused fits from closed independently audited JSON.

Only JSON metadata and byte hashes are read. No scientific module, array,
checkpoint decoder, trajectory reader, model or optimizer is invoked. The old
hash-pinned plot helper supplies file guards, CSV output and numeric comparison.
All scientific reconstruction remains the responsibility of the closed audit.
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
SELF = "scripts/plot_otto_action_focused.py"
VERSION = "otto-action-focused-plot-v1"
HELPER = "scripts/plot_otto_prequery_calibration.py"
HELPER_PIN = "69c3542f693e8bdceb474da0beb8880b81a8614360107119fcc97173f9653403"
AUDITOR = "scripts/audit_otto_action_focused.py"
TRAINER = "scripts/train_otto_action_focused.py"
AUDIT_VERSION = "otto-action-focused-saved-audit-v1"
TRAIN_VERSION = "otto-action-focused-training-v1"
KINDS = ("innovation_aux", "innovation_spo", "gru_aux", "gru_spo")
SEEDS = (295000001, 295000002, 295000003)
REGIMES = ("lambda3", "lambda4")
SCOPES = ("initial", "full", "postcorrection")
ARMS = ("analytic", "neural", "period4_hold")
AGREEMENT = "episode_weighted_agreement"
GAP = "episode_weighted_raw_gap"
METRICS = (AGREEMENT, GAP, "episode_weighted_centered_mse", "episode_weighted_first_argmin_match")
PANELS = (("postcorrection", GAP, "Primary teacher-score gap | steps >= 5", 1.),
          ("full", AGREEMENT, "Full nonquery agreement (%)", 100.),
          ("initial", AGREEMENT, "Initial agreement (%) | steps 1-3", 100.))
GATES = {"innovation_objective": ("Explicit objective", 23),
         "gru_objective": ("GRU objective", 23), "architecture": ("Architecture", 29)}
LABELS = ("Explicit\nAUX", "Explicit\nAUX + SPO+", "GRU\nAUX", "GRU\nAUX + SPO+")
COLORS = ("#729ac7", "#195c99", "#74a991", "#087957")
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2,
          "metadata_json_bytes": 16 * 1024**2, "audit_json_bytes": 256 * 1024**2}
TRAIN_PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "summary.json",
    "training-history.npz", "training-history.json", "validation-history.npz", "validation-history.json",
    "validation-windows.npz", "validation-windows.json", "prediction-hold.npz"} | {
    f"{prefix}{kind}-{seed}.npz" for prefix in ("", "prediction-", "training-prediction-")
    for kind in KINDS for seed in SEEDS}
FIGURES = ("otto-action-focused-forecast", "otto-action-focused-changes", "otto-action-focused-costs")
PAYLOADS = {f"{stem}.{suffix}" for stem in FIGURES for suffix in ("png", "svg")} | {
    "forecast-metrics.csv", "regime-metrics.csv", "family-means.csv", "objective-contrasts.csv",
    "objective-contrast-means.csv", "costs.csv", "conditions.csv", "gates.csv", "plotted-values.json"}


def _helper():
    path = ROOT / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_PIN:
        raise ValueError("unchanged saved-only plot helpers")
    spec = importlib.util.spec_from_file_location("_action_focused_plot_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VERSION, module.LIMITS = VERSION, LIMITS
    return module


base = _helper()
require, same, write, write_csv = base.require, base.same, base.write, base.write_csv


def gate_records(rows):
    """Check named membership, without re-evaluating scientific comparisons."""
    names = ["common.technical_complete"]
    for regime in REGIMES:
        names.append(f"common.{regime}.initial.case_support")
        names.extend(f"common.{regime}.age{age}.case_support" for age in (1, 2, 3))
        names.append(f"common.{regime}.hold.postcorrection.positive_gap")
    for architecture in ("innovation", "gru"):
        for regime in REGIMES:
            prefix = f"objective.{architecture}.{regime}"
            names.extend(f"{prefix}.{suffix}" for suffix in ("postcorrection.gap", "full.agreement", "initial.agreement"))
            names.extend(f"{prefix}.{seed}.postcorrection.gap_nonregression" for seed in SEEDS)
    for regime in REGIMES:
        names.extend(f"architecture.{regime}.{suffix}" for suffix in ("postcorrection.gap", "full.agreement", "initial.agreement"))
    require(len(rows) == 41 and [r["name"] for r in rows] == names
            and all(type(r["passes"]) is bool for r in rows), "exact ordered 41 Boolean condition records")
    common = [r for r in rows if r["name"].startswith("common.")]
    objective = {a: [r for r in rows if r["name"].startswith(f"objective.{a}.")] for a in ("innovation", "gru")}
    members = {"innovation_objective": common + objective["innovation"],
               "gru_objective": common + objective["gru"],
               "architecture": common + objective["innovation"] + [r for r in rows if r["name"].startswith("architecture.")]}
    result = {name: {"conditions": [r["name"] for r in selected], "passed": sum(r["passes"] for r in selected),
                     "total": len(selected), "passes": all(r["passes"] for r in selected)} for name, selected in members.items()}
    require(all(result[k]["total"] == size for k, (_, size) in GATES.items()), "three distinct 23/23/29 decisions")
    return result


def parent_options(receipt, terminal, *, script, cap, expected, mode=None):
    """Authenticate the actual successful original command before joining launch."""
    require(terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["timed_out"] is False
        and terminal["error"] is terminal["clock_error"] is None
        and terminal["group_absent"] is terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["errors"] == []
        and terminal["cap_seconds"] == cap and terminal["clock_source_sha256"] == base.CLOCK_PIN
        and terminal["watchdog_sha256"] == base.SUPERVISOR_PIN and terminal["cwd"] == str(ROOT)
        and terminal["deadline_ns"] == terminal["started_ns"] + cap * 10**9
        and terminal["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
        <= terminal["finished_ns"] <= terminal["deadline_ns"], "successful original process closure")
    require(terminal["elapsed_ns"] == terminal["finished_ns"] - terminal["started_ns"]
        and terminal["wall_seconds"] == terminal["elapsed_ns"] / 1e9
        and receipt["wall_seconds"] == (receipt["finished_ns"] - receipt["started_ns"]) / 1e9,
        "original elapsed accounting")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    prefix = [str(ROOT / ".venv/bin/python"), str(ROOT / script)] + ([] if mode is None else [mode])
    require(command[:len(prefix)] == prefix and len(command) == len(prefix) + 2 * (len(expected) + 1),
            "original absolute command and exact argument count")
    options = command[len(prefix):]
    require(len(set(options[::2])) == len(expected) + 1, "unique original options")
    parsed = dict(zip(options[::2], options[1::2], strict=True))
    require(set(parsed) == set(expected) | {"--supervision"} and all(parsed[k] == v for k, v in expected.items()),
            "original command input/output joins")
    return parsed


def tables(summary, audit):
    """Export all saved scopes, every fit and all within-seed SPO-minus-AUX changes."""
    expected = [(family, seed) for seed in SEEDS for family in KINDS]
    require([(r["family"], r["seed"]) for r in summary["models"]] == expected
            and [(r["family"], r["seed"]) for r in audit["fits"]] == expected, "all twelve final fits in fixed order")
    records = [*summary["models"], {"family": "hold", "seed": None, "metrics": summary["hold"]}]
    full_rows, rows, means = [], [], []
    for record in records:
        report = record["metrics"]
        require(report["primary_mask"] == "nonquery and absolute_step >= 5"
                and set(report["by_collector"]) == set(ARMS), "complete fixed forecast domain")
        for collector, variant in [("all", report), *[(a, report["by_collector"][a]) for a in ARMS]]:
            for scope in SCOPES:
                section = variant[scope]
                require(set(section["by_regime"]) == set(REGIMES) and len(section["by_case"]) == 12,
                        "both settings and all originating cases")
                groups = [("overall", None, None, section["overall"])]
                groups += [("regime", r, None, section["by_regime"][r]) for r in REGIMES]
                groups += [("case", g["regime"], g["case"], g) for g in section["by_case"]]
                for partition, regime, case, group in groups:
                    require(set(group["by_age"]) == {"1", "2", "3"}, "all three ages retained")
                    for age, item in [("all", group), *[(a, group["by_age"][a]) for a in ("1", "2", "3")]]:
                        full_rows.append({"family": record["family"], "seed": record["seed"], "scope": scope,
                            "collector": collector, "partition": partition, "regime": regime, "case": case, "age": age,
                            **{k: v for k, v in item.items() if k not in ("by_age", "regime", "case")}})
        for regime in REGIMES:
            for scope in SCOPES:
                group = report[scope]["by_regime"][regime]
                require(group["episodes"] == 18 and group["declared_case_count"] == 6, "fixed episode/case denominator")
                for metric in METRICS:
                    value = group[metric]
                    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite displayed metric")
                    require(metric not in (AGREEMENT, "episode_weighted_first_argmin_match") or value <= 1,
                            "bounded action agreement")
                rows.append({"family": record["family"], "seed": record["seed"], "regime": regime, "scope": scope,
                    **{k: group[k] for k in ("episodes", "supported_episodes", "declared_case_count", "supported_case_count",
                                            "weight_mass", "nonquery_rows", *METRICS)}})
    by = {(r["family"], r["seed"], r["regime"], r["scope"]): r for r in rows}
    for family in KINDS:
        for regime in REGIMES:
            for scope in SCOPES:
                selected = [by[family, seed, regime, scope] for seed in SEEDS]
                require(all({k: r[k] for k in ("episodes", "supported_episodes", "supported_case_count", "weight_mass")}
                            == {k: selected[0][k] for k in ("episodes", "supported_episodes", "supported_case_count", "weight_mass")}
                            for r in selected), "identical seed metric support")
                means.append({"family": family, "regime": regime, "scope": scope, "seed_count": 3,
                              **{k: math.fsum(r[k] for r in selected) / 3 for k in METRICS}})
    changes, change_means = [], []
    for architecture in ("innovation", "gru"):
        for regime in REGIMES:
            for scope in SCOPES:
                selected = []
                for seed in SEEDS:
                    aux, spo = (by[f"{architecture}_{objective}", seed, regime, scope] for objective in ("aux", "spo"))
                    require(all(aux[k] == spo[k] for k in ("episodes", "supported_episodes", "declared_case_count",
                                "supported_case_count", "weight_mass", "nonquery_rows")), "identical paired objective support")
                    row = {"architecture": architecture, "seed": seed, "regime": regime, "scope": scope,
                           "direction": "SPO minus AUX", "episodes": aux["episodes"], "weight_mass": aux["weight_mass"]}
                    for key in METRICS:
                        row.update({"aux_" + key: aux[key], "spo_" + key: spo[key], "delta_" + key: spo[key] - aux[key]})
                    changes.append(row); selected.append(row)
                change_means.append({"architecture": architecture, "regime": regime, "scope": scope,
                    "direction": "SPO minus AUX", "seed_count": 3,
                    **{prefix + key: math.fsum(r[prefix + key] for r in selected) / 3
                       for prefix in ("aux_", "spo_", "delta_") for key in METRICS}})
    costs, fits_by = [], {(r["family"], r["seed"]): r for r in audit["fits"]}
    for fit in audit["fits"]:
        family, seed = fit["family"], fit["seed"]
        architecture = "innovation" if family.startswith("innovation") else "gru"
        control = fits_by[f"{architecture}_aux", seed]
        require(fit["epochs"] == 80 and fit["steps"] == 720 and fit["readout"] == "shared", "fixed fit recipe")
        keys = ("architecture", "readout", "objective", "parameter_count", "epochs", "steps", "fit_seconds", "wall_seconds",
                "train_rescore_seconds", "checkpoint_seconds", "forward_chunks", "backward_chunks", "no_grad_chunks",
                "spo_loss_calls", "spo_weighted_rows", "final_nonquery_loss", "final_prior_loss", "final_spo_loss",
                "final_objective_spo_loss", "final_train_loss")
        row = {"family": family, "seed": seed, **{k: fit[k] for k in keys}}
        for key in ("fit_seconds", "wall_seconds", "train_rescore_seconds", "checkpoint_seconds"):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, "finite measured fit cost")
        for key in ("fit_seconds", "wall_seconds"):
            require(control[key] > 0 and row[key] > 0, "positive actual fit interval")
            row[key + "_delta_to_paired_aux"] = row[key] - control[key]
            row[key + "_ratio_to_paired_aux"] = row[key] / control[key]
        row["paired_aux_family"] = f"{architecture}_aux"
        costs.append(row)
    require((len(full_rows), len(rows), len(means), len(changes), len(change_means), len(costs))
            == (9360, 78, 24, 36, 12, 12), "complete metrics, contrasts and measured cost coverage")
    return {"forecast_metrics": full_rows, "regime_metrics": rows, "family_means": means,
            "objective_contrasts": changes, "objective_contrast_means": change_means, "costs": costs}


class Plot(base.Plot):
    def read(self, path):
        limit = LIMITS["audit_json_bytes"] if path == self.args.audit_directory / "audit.json" else LIMITS["metadata_json_bytes"]
        require(str(path) in self.bound and self.bound[str(path)]["bytes"] <= limit, "authenticated bounded JSON")
        return json.loads(Path(path).read_text())

    def closed(self, directory, receipt, expected):
        require(set(receipt["files"]) == expected and {p.name for p in directory.iterdir()} == expected | {"receipt.json"},
                "exact closed phase inventory")
        for name in sorted(expected):
            self.bind(directory / name, receipt["files"][name])

    def joined_parent(self, receipt, terminal, started, *, script, cap, expected, mode=None):
        options = parent_options(receipt, terminal, script=script, cap=cap, expected=expected, mode=mode)
        launch_path = Path(options["--supervision"])
        self.bind(launch_path, receipt["supervision_sha256"])
        launch = self.read(launch_path)
        require(started["launch"] == launch and all(terminal[k] == v for k, v in launch.items()), "original launch retained")

    def authenticate(self):
        args = self.args
        self.bind(args.audit_directory / "receipt.json", args.audit_receipt_sha256)
        self.bind(args.audit_terminal, args.audit_terminal_sha256)
        receipt, terminal = self.read(args.audit_directory / "receipt.json"), self.read(args.audit_terminal)
        require(receipt["version"] == AUDIT_VERSION and receipt["status"] == "completed" and receipt["agreement"] is True
            and receipt["failures"] == [] and receipt["requires_successful_original_supervisor"] is True,
            "completed agreeing independent audit")
        self.closed(args.audit_directory, receipt, {"started.json", "audit.json"})
        require(set(receipt["producer_inputs"]) == {"plan", "worker", "terminal"}, "three producer role descriptors")
        for descriptor in receipt["producer_inputs"].values():
            self.bind(Path(descriptor["path"]), descriptor)
        expected = {"--output": str(args.audit_directory)}
        for role, descriptor in receipt["producer_inputs"].items():
            expected[f"--{role}"], expected[f"--{role}-sha256"] = descriptor["path"], descriptor["sha256"]
        audit_started = self.read(args.audit_directory / "started.json")
        require(audit_started["producer_inputs"] == receipt["producer_inputs"]
                and audit_started["started_ns"] == receipt["started_ns"], "original audit input and start witness")
        self.joined_parent(receipt, terminal, audit_started,
                           script=AUDITOR, cap=240, expected=expected)
        roles = receipt["producer_inputs"]
        plan, worker, train_terminal = (self.read(Path(roles[k]["path"])) for k in ("plan", "worker", "terminal"))
        require(plan["version"] == worker["version"] == TRAIN_VERSION and plan["status"] == "frozen_before_fitting"
            and worker["status"] == "completed" and worker["complete"] is True
            and worker["fits_completed"] == 12 and worker["optimizer_steps"] == 8640
            and worker["pending"] is worker["pending_emission"] is None
            and worker["requires_successful_original_supervisor"] is True
            and worker["plan_sha256"] == receipt["plan_sha256"] == roles["plan"]["sha256"]
            and worker["sources"] == plan["sources"] and worker["inputs"] == plan["inputs"], "complete same training allocation")
        require({AUDITOR, TRAINER, "src/openjev/research/otto_action_focused_loss.py",
                 "src/openjev/research/otto_action_focused_metrics.py", "src/openjev/research/otto_spo_plus_loss.py"}
                <= set(plan["sources"]), "scientific components frozen in source closure")
        for name, pin in plan["sources"].items():
            self.bind(ROOT / name, pin)
        require(AUDITOR in receipt["sources"] and all(plan["sources"].get(k) == v for k, v in receipt["sources"].items()),
                "same prospective auditor sources")
        for descriptor in plan["inputs"].values():
            self.bind(Path(descriptor["path"]), descriptor)
        run = Path(roles["worker"]["path"]).parent
        self.closed(run, worker, TRAIN_PAYLOADS)
        train_started = self.read(run / "started.json")
        require(train_started["started_ns"] == worker["started_ns"], "original training start witness")
        self.joined_parent(worker, train_terminal, train_started, script=TRAINER, cap=14400, mode="run",
            expected={"--plan": roles["plan"]["path"], "--plan-sha256": roles["plan"]["sha256"], "--output": str(run)})
        # Decode scientific aggregates only after both successful parent and full byte closures.
        summary, audit = self.read(run / "summary.json"), self.read(args.audit_directory / "audit.json")
        require(summary["version"] == TRAIN_VERSION and audit["version"] == AUDIT_VERSION and audit["agreement"] is True
            and audit["technical_condition_requires_successful_original_audit_supervisor"] is True
            and summary["technical_complete_pending_saved_audit"] is True, "audited provisional/final distinction")
        same(summary, audit["producer_summary"])
        same(summary["required"][0], {"name": "common.technical_complete", "value": False,
                                      "relation": "==", "threshold": True, "passes": False})
        same(summary["gates"], gate_records(summary["required"]))
        rules = [{**summary["required"][0], "value": True, "passes": True}, *summary["required"][1:]]
        final = {**summary, "required": rules, "required_passed": summary["required_passed"] + 1,
                 "gates": gate_records(rules), "technical_complete_pending_saved_audit": False}
        same(audit["summary"], final)
        require(audit["counts"]["fits"] == 12 and audit["counts"]["prediction_files"] == 25
            and audit["counts"]["training_payloads"] == len(worker["files"]) == 49
            and audit["counts"]["collection_episodes"] == 90
            and audit["counts"]["required_conditions"] == final["required_conditions"] == 41
            and audit["counts"]["required_passed"] == final["required_passed"] == sum(r["passes"] for r in rules)
            and final["train_counts"]["episodes"] == 54 and final["validation_counts"]["episodes"] == 36,
            "all audited fits, predictions, conditions and paths")
        self.receipt["original_seconds"] = {"training_worker": worker["wall_seconds"], "training_parent": train_terminal["wall_seconds"],
                                             "audit_worker": receipt["wall_seconds"], "audit_parent": terminal["wall_seconds"]}
        return audit["summary"], audit

    def save_figure(self, fig, stem):
        for suffix in ("png", "svg"):
            with (self.out / f"{stem}.{suffix}").open("xb") as stream:
                fig.savefig(stream, format=suffix, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=.2)
                stream.flush(); os.fsync(stream.fileno())

    @staticmethod
    def dots(ax, values, families=KINDS):
        for x, family in enumerate(families):
            color = COLORS[KINDS.index(family)]
            selected = []
            for seed, shift, marker in zip(SEEDS, (-.13, 0., .13), ("o", "s", "^"), strict=True):
                value = values[family, seed]; selected.append(value)
                ax.plot(x + shift, value, marker=marker, color=color, markerfacecolor="white", linestyle="none", markersize=6)
            mean = math.fsum(selected) / 3
            ax.plot((x - .24, x + .24), (mean, mean), color=color, linewidth=2.5)
        ax.set_xticks(range(len(families)), [LABELS[KINDS.index(f)] for f in families])
        ax.set_xlim(-.5, len(families) - .5); ax.grid(axis="y", alpha=.18)

    def figures(self, summary, data):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        handles = [Line2D([], [], marker=m, color="#4b5563", markerfacecolor="white", linestyle="none", label=f"Seed {s}")
                   for s, m in zip(SEEDS, ("o", "s", "^"), strict=True)]
        handles += [Line2D([], [], color="#4b5563", linewidth=2.5, label="Mean of all three seeds")]
        gates = "  |  ".join(f"{label}: {'PASS' if summary['gates'][key]['passes'] else 'FAIL'} "
                            f"({summary['gates'][key]['passed']}/{count})" for key, (label, count) in GATES.items())
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey="row")
        fig.subplots_adjust(top=.85, bottom=.15, hspace=.48, wspace=.14)
        for row, (scope, metric, label, scale) in enumerate(PANELS):
            chosen = [r for r in data["regime_metrics"] if r["scope"] == scope]
            maximum = max(r[metric] * scale for r in chosen)
            for col, regime in enumerate(REGIMES):
                ax = axes[row, col]
                selected = [r for r in chosen if r["regime"] == regime]
                self.dots(ax, {(r["family"], r["seed"]): r[metric] * scale for r in selected if r["family"] != "hold"})
                ax.axhline(next(r[metric] * scale for r in selected if r["family"] == "hold"),
                           color="#6b7280", linestyle="--", linewidth=1)
                ax.set_ylim(0, 100 if metric == AGREEMENT else maximum * 1.12 if maximum else 1)
                if col == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"Sensing length {regime[-1]} | 6 paired VALID cases", pad=12)
        fig.suptitle("Action-focused loss on fixed paths", fontsize=19, y=.98)
        fig.text(.5, .942, gates, ha="center", fontsize=10, weight="bold")
        fig.text(.5, .908, "AUX = nonquery MSE + prior MSE; treatment adds unit-weight SPO+. All 12 final fits retained.", ha="center")
        fig.legend(handles=handles + [Line2D([], [], color="#6b7280", linestyle="--", label="Hold-Q reference")],
                   loc="lower center", bbox_to_anchor=(.5, .085), ncol=5, frameon=False, fontsize=9)
        fig.text(.5, .065, "Equal declared episode weights, including zero-support paths. Shared axes within each metric.", ha="center", fontsize=9)
        fig.text(.5, .038, "Three separate gates, no all-41 promotion rule. Fresh development screen; no autonomous or unseen-shift result.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[0]); plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey="row")
        fig.subplots_adjust(top=.86, bottom=.15, hspace=.43, wspace=.15)
        for row, (scope, metric, _, scale) in enumerate(PANELS):
            chosen = [r for r in data["objective_contrasts"] if r["scope"] == scope]
            extent = max(abs(r["delta_" + metric] * scale) for r in chosen)
            extent = extent * 1.2 if extent else 1.
            for col, regime in enumerate(REGIMES):
                ax = axes[row, col]
                values = {(r["architecture"] + "_spo", r["seed"]): r["delta_" + metric] * scale
                          for r in chosen if r["regime"] == regime}
                self.dots(ax, values, ("innovation_spo", "gru_spo"))
                ax.set_xticks((0, 1), ("Explicit: SPO minus AUX", "GRU: SPO minus AUX"), fontsize=9)
                ax.axhline(0, color="#555", linewidth=1); ax.set_ylim(-extent, extent)
                if col == 0:
                    ax.set_ylabel("Primary gap change\nNegative is better" if metric == GAP else
                                  ("Full" if scope == "full" else "Initial") + " agreement change (pp)\nPositive is better")
                if row == 0:
                    ax.set_title(f"Sensing length {regime[-1]}", pad=12)
        fig.suptitle("Paired objective changes, with every seed", fontsize=18, y=.975)
        fig.text(.5, .923, "Each point compares the same architecture, seed, exposure and shared initialization.", ha="center")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .082), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, .058, "Changes are signed metric differences, not percentage changes. Three seeds are not independent environment cohorts.", ha="center", fontsize=9)
        fig.text(.5, .03, "Forced-path teacher imitation only; neither lower loss nor an objective gain establishes an architecture advantage.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[1]); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(top=.79, bottom=.3, wspace=.3)
        self.dots(axes[0], {(r["family"], r["seed"]): r["wall_seconds"] for r in data["costs"]})
        axes[0].set_ylabel("Measured fit wall seconds\nIncludes final TRAIN rescore and checkpoint")
        axes[0].set_ylim(bottom=0)
        self.dots(axes[1], {(r["family"], r["seed"]): r["wall_seconds_ratio_to_paired_aux"] for r in data["costs"]
                           if r["family"].endswith("_spo")}, ("innovation_spo", "gru_spo"))
        axes[1].set_xticks((0, 1), ("Explicit SPO / paired AUX", "GRU SPO / paired AUX"), fontsize=9)
        axes[1].set_ylabel("Within-seed measured fit wall-time ratio")
        axes[1].axhline(1, color="#6b7280", linestyle="--"); axes[1].set_ylim(bottom=0)
        fig.suptitle("Measured fitting cost, not deployment latency", fontsize=18, y=.965)
        fig.text(.5, .862, "Same 80 epochs and 720 updates per fit; actual objective arithmetic and rescore costs are included.", ha="center", fontsize=10)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .17), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, .115, "costs.csv includes every fit interval, total wall interval, final loss component, and paired AUX cost ratio/delta.", ha="center", fontsize=9)
        fig.text(.5, .065, "Collection, shared setup and VALID are excluded from per-fit intervals. Whole training-stage seconds remain in plotted-values.json.", ha="center", fontsize=9)
        fig.text(.5, .022, "Equal exposure is not equal compute. Timings are authenticated saved measurements; no new timing experiment is performed.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[2]); plt.close(fig)
        return matplotlib.__version__

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(p.is_symlink() for p in self.out.parents), "exclusive contained presentation output")
        self.out.mkdir(exist_ok=False)
        try:
            self.bind(ROOT / SELF, self.descriptor(ROOT / SELF))
            self.bind(ROOT / HELPER, HELPER_PIN)
            summary, audit = self.authenticate()
            data = tables(summary, audit)
            for key, rows in data.items():
                write_csv(self.out / (key.replace("_", "-") + ".csv"), rows)
            write_csv(self.out / "conditions.csv", [{**r, "strict_upper_bound": r.get("strict_upper_bound")}
                                                  for r in summary["required"]])
            write_csv(self.out / "gates.csv", [{"gate": k, **v} for k, v in summary["gates"].items()])
            write(self.out / "plotted-values.json", {"version": VERSION, **{k: v for k, v in data.items() if k != "forecast_metrics"},
                "conditions": summary["required"], "gates": summary["gates"], "scope": summary["scope"],
                "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
                "physical_stage_seconds": {k: summary[k] for k in ("setup_seconds", "fitting_seconds", "validation_seconds")},
                "original_seconds": self.receipt["original_seconds"], "audit_limitations": audit["limitations"],
                "cost_scope": "Measured fit wall intervals, not deployment latency. No per-fit VALID timer exists; collection and shared setup are excluded."})
            self.receipt["matplotlib_version"] = self.figures(summary, data)
            for path, descriptor in self.bound.items():
                require(self.descriptor(Path(path)) == descriptor, "unchanged evidence and sources after plotting")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact fifteen presentation payloads")
            self.check()
            self.receipt.update(status="completed", inputs=self.bound, gates=summary["gates"],
                counts={k: len(v) for k, v in data.items()},
                files={p.name: self.descriptor(p) for p in sorted(self.out.iterdir())},
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
            except BaseException as secondary:  # noqa: BLE001 - preserve the original publication failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--audit-terminal", type=Path, required=True)
    parser.add_argument("--audit-terminal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    Plot(parser.parse_args()).execute()


if __name__ == "__main__":
    main()
