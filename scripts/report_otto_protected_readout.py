"""Render a closed protected-readout study from authenticated saved JSON only.

Both original supervisors must have exited successfully, reaped their children
and retained the original launches. The complete training byte inventory and
prospectively pinned scientific sources are authenticated before outcome JSON
is decoded. The independent saved audit supplies numerical verification; this
report neither reruns scientific arithmetic nor opens arrays/checkpoints with
a decoder. Hashing those payloads is necessary to authenticate their identity.

All five families and three seeds remain visible, including a failed scientific
gate. The one 29-condition gate is preserved without selecting another gate.
Teacher-score gap is a forced-path imitation metric, not autonomous utility.
Shared pretraining is counted once per seed; branch times exclude that common
cost. No per-fit VALID timer exists, so validation is shown only in aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/report_otto_protected_readout.py"
VERSION = "otto-protected-readout-report-v1"
HELPER = "scripts/plot_otto_action_focused.py"
HELPER_PIN = "851065d57c65c1f5f9babfacf55cc260bb002d48093fca34b10330f86fecb080"
AUDITOR = "scripts/audit_otto_protected_readout.py"
TRAINER = "scripts/train_otto_protected_readout.py"
AUDIT_VERSION = "otto-protected-readout-saved-audit-v1"
TRAIN_VERSION = "otto-protected-readout-training-v1"
SCIENCE_PINS = {
    AUDITOR: "5a92e4d4c989fe6adf9dd242c3f62a466c7d6f73c4a801662497349ca4d9dbb9",
    TRAINER: "6efd2199b44f069a9c29ce597e8592948d35d211de56fdadf8dab06b7a7bb507",
    "src/openjev/research/otto_protected_readout_metrics.py":
        "c74858fa74a0c1eeabc8e8f23a7a2b9ca491c7e5bcb350db60a887bc51cb8b5c",
    "src/openjev/research/otto_protected_training_model.py":
        "2168dfa34096e154445599cba93964e6c60cfd6ccd3b77ac84624039e6db8bd0",
    "src/openjev/research/otto_protected_readout.py":
        "3cfa99320472abad632cde112664be80212a7e283c44f4275e64d6c25c9a59e3",
}
FAMILIES = ("pretrained", "frozen_aux", "frozen_spo", "joint_aux", "joint_spo")
CONTROLS = ("pretrained", "frozen_aux", "joint_aux", "joint_spo")
CANDIDATE = "frozen_spo"
SEEDS = (301000001, 301000002, 301000003)
REGIMES = ("lambda3", "lambda4")
SCOPES = ("initial", "full", "postcorrection")
GAP = "episode_weighted_raw_gap"
AGREEMENT = "episode_weighted_agreement"
METRICS = (GAP, AGREEMENT, "episode_weighted_centered_mse", "episode_weighted_first_argmin_match")
SUPPORT = ("episodes", "supported_episodes", "declared_case_count", "supported_case_count", "weight_mass", "nonquery_rows")
LABELS = {"pretrained": "Shared\npretrain", "frozen_aux": "Frozen\nAUX", "frozen_spo": "Frozen\nAUX + SPO+",
          "joint_aux": "Joint\nAUX", "joint_spo": "Joint\nAUX + SPO+"}
COLORS = {"pretrained": "#64748b", "frozen_aux": "#77a8cf", "frozen_spo": "#175b96",
          "joint_aux": "#7dad91", "joint_spo": "#127957"}
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2,
          "metadata_json_bytes": 16 * 1024**2, "audit_json_bytes": 256 * 1024**2}
TRAIN_PAYLOADS = {"started.json", "runtime.json", "progress.jsonl", "work.jsonl", "fits.json", "summary.json",
    "training-history.npz", "training-history.json", "validation-history.npz", "validation-history.json",
    "validation-windows.npz", "validation-windows.json", "prediction-hold.npz"} | {
    f"{prefix}{family}-{seed}.npz" for prefix in ("", "prediction-", "training-prediction-")
    for family in FAMILIES for seed in SEEDS}
FIGURES = ("otto-protected-readout-gaps", "otto-protected-readout-changes", "otto-protected-readout-costs")
TABLES = ("regime_metrics", "family_means", "paired_changes", "paired_means", "fit_costs")
PAYLOADS = {f"{stem}.{suffix}" for stem in FIGURES for suffix in ("png", "svg")} | {
    name.replace("_", "-") + ".csv" for name in TABLES} | {"conditions.csv", "summary.json"}
COST_SCOPE = ("Actual saved wall intervals, not inference latency. Shared pretraining is counted once per seed; "
    "branch times exclude shared pretraining. Per-fit wall includes fitting, evaluation-clone construction, "
    "final TRAIN rescore, checkpoint publication and remaining overhead. There is no per-fit VALID timer. "
    "Whole VALID timing includes all models and common evaluation work. Collection and audit are separate phases.")


def _helper():
    path = ROOT / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_PIN:
        raise ValueError("immutable saved-only presentation helper")
    spec = importlib.util.spec_from_file_location("_protected_readout_report_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.VERSION, module.LIMITS = VERSION, LIMITS
    module.base.VERSION, module.base.LIMITS = VERSION, LIMITS
    return module


base = _helper()
require, same, write, write_csv = base.require, base.same, base.write, base.write_csv


def gate_records(rows):
    """Retain exact prospective gate membership, without rejudging comparisons."""
    names = ["common.technical_complete"]
    for regime in REGIMES:
        names.append(f"common.{regime}.initial.case_support")
        names.extend(f"common.{regime}.age{age}.case_support" for age in (1, 2, 3))
        names.append(f"common.{regime}.hold.postcorrection.positive_gap")
    for regime in REGIMES:
        names.extend(f"candidate.{regime}.postcorrection.gap_vs_{control}" for control in CONTROLS)
        names.extend(f"candidate.{regime}.{scope}.agreement" for scope in ("full", "initial"))
        names.extend(f"candidate.{regime}.{seed}.postcorrection.gap_vs_frozen_aux" for seed in SEEDS)
    require(len(rows) == 29 and [r["name"] for r in rows] == names
            and all(type(r["passes"]) is bool for r in rows), "exact ordered 29 Boolean condition records")
    return {"protected_readout": {"conditions": names, "passed": sum(r["passes"] for r in rows),
                                 "total": 29, "passes": all(r["passes"] for r in rows)}}


def numeric(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0), label)
    return value


def tables(summary, audit):
    """Keep every saved cell, three metric scopes and all four paired controls."""
    expected = [(family, seed) for seed in SEEDS for family in FAMILIES]
    require([(r["family"], r["seed"]) for r in summary["models"]] == expected
            and [(r["family"], r["seed"]) for r in audit["fits"]] == expected, "all fifteen fits in fixed order")
    rows = []
    for record in [*summary["models"], {"family": "hold", "seed": None, "metrics": summary["hold"]}]:
        report = record["metrics"]
        require(report["primary_mask"] == "nonquery and absolute_step >= 5", "fixed primary domain")
        for regime in REGIMES:
            for scope in SCOPES:
                require(set(report[scope]["by_regime"]) == set(REGIMES), "both saved regimes")
                group = report[scope]["by_regime"][regime]
                require(group["episodes"] == 18 and group["declared_case_count"] == 6, "fixed VALID denominators")
                for metric in METRICS:
                    numeric(group[metric], "finite displayed metric")
                    require(metric not in (AGREEMENT, "episode_weighted_first_argmin_match") or group[metric] <= 1,
                            "bounded agreement")
                rows.append({"family": record["family"], "seed": record["seed"], "regime": regime, "scope": scope,
                             **{key: group[key] for key in (*SUPPORT, *METRICS)}})
    by = {(r["family"], r["seed"], r["regime"], r["scope"]): r for r in rows}
    means, changes, paired_means = [], [], []
    for regime in REGIMES:
        for scope in SCOPES:
            reference = by[CANDIDATE, SEEDS[0], regime, scope]
            for family in FAMILIES:
                selected = [by[family, seed, regime, scope] for seed in SEEDS]
                require(all(all(row[k] == reference[k] for k in SUPPORT) for row in selected), "identical paired support")
                means.append({"family": family, "regime": regime, "scope": scope, "seed_count": 3,
                              **{metric: math.fsum(row[metric] for row in selected) / 3 for metric in METRICS}})
            for control in CONTROLS:
                selected = []
                for seed in SEEDS:
                    candidate, baseline = by[CANDIDATE, seed, regime, scope], by[control, seed, regime, scope]
                    row = {"candidate": CANDIDATE, "control": control, "seed": seed, "regime": regime,
                           "scope": scope, "direction": "frozen_spo minus control"}
                    for metric in METRICS:
                        row.update({"candidate_" + metric: candidate[metric], "control_" + metric: baseline[metric],
                                    "delta_" + metric: candidate[metric] - baseline[metric]})
                    changes.append(row); selected.append(row)
                paired_means.append({"candidate": CANDIDATE, "control": control, "regime": regime, "scope": scope,
                    "direction": "frozen_spo minus control", "seed_count": 3,
                    **{prefix + metric: math.fsum(row[prefix + metric] for row in selected) / 3
                       for prefix in ("candidate_", "control_", "delta_") for metric in METRICS}})
    costs = []
    components = ("fit_seconds", "evaluation_clone_seconds", "train_rescore_seconds", "checkpoint_seconds")
    for fit in audit["fits"]:
        pretrained = fit["family"] == "pretrained"
        require(fit["epochs"] == (80 if pretrained else 40) and fit["steps"] == (720 if pretrained else 360)
            and fit["parameter_count"] == (5996 if pretrained else 6112)
            and fit["readout"] == ("shared" if pretrained else "protected"), "fixed staged fit recipe")
        keys = ("family", "seed", "training_mode", "stage", "readout", "objective", "parameter_count",
            "trainable_parameter_count", "epochs", "steps", "wall_seconds", *components,
            "forward_chunks", "backward_chunks", "no_grad_chunks", "differentiable_chunks", "skipped_backward_chunks")
        row = {key: fit[key] for key in keys}
        for key in ("wall_seconds", *components):
            numeric(row[key], "finite actual fit interval", positive=key in ("wall_seconds", "fit_seconds"))
        measured = math.fsum(row[key] for key in components)
        require(measured <= row["wall_seconds"] + 1e-9, "fit components inside wall interval")
        # At most the already-audited 1ns conversion roundoff is floored for display.
        row["other_seconds"] = max(0., row["wall_seconds"] - measured)
        row["allocation"] = "shared_pretraining_once_per_seed" if pretrained else "branch_excluding_shared_pretraining"
        costs.append(row)
    result = dict(zip(TABLES, (rows, means, changes, paired_means, costs), strict=True))
    require([len(result[k]) for k in TABLES] == [96, 30, 72, 24, 15], "complete saved display coverage")
    return result


def cost_summary(summary, data):
    """Disjoint actual fit allocations and separately measured phase intervals."""
    rows = data["fit_costs"]
    totals = {family: math.fsum(r["wall_seconds"] for r in rows if r["family"] == family) for family in FAMILIES}
    phases = {key: numeric(summary[key], "finite physical stage")
              for key in ("setup_seconds", "fitting_seconds", "validation_seconds")}
    fit_total = math.fsum(totals.values())
    require(fit_total <= phases["fitting_seconds"] + 1e-9, "all fits inside fitting stage")
    return {"family_wall_seconds": totals, "shared_pretraining_seconds": totals["pretrained"],
        "adaptation_seconds": math.fsum(totals[f] for f in FAMILIES if f != "pretrained"),
        "fit_wall_seconds": fit_total, "fitting_stage_other_seconds": max(0., phases["fitting_seconds"] - fit_total),
        "physical_stage_seconds": phases, "shared_pretrain_fits": 3, "adaptation_fits": 12,
        "per_fit_validation_seconds": None, "scope": COST_SCOPE}


class Report(base.Plot):
    def authenticate(self):
        args = self.args
        self.bind(args.audit_directory / "receipt.json", args.audit_receipt_sha256)
        self.bind(args.audit_terminal, args.audit_terminal_sha256)
        receipt, terminal = self.read(args.audit_directory / "receipt.json"), self.read(args.audit_terminal)
        require(receipt["version"] == AUDIT_VERSION and receipt["status"] == "completed" and receipt["agreement"] is True
            and receipt["failures"] == [] and receipt["requires_successful_original_supervisor"] is True,
            "completed agreeing independent audit")
        require(all(receipt[key] == 0 for key in ("native_calls", "model_calls", "optimizer_calls", "teacher_calls")),
                "saved-only original audit")
        self.closed(args.audit_directory, receipt, {"started.json", "audit.json"})
        require(set(receipt["producer_inputs"]) == {"plan", "worker", "terminal"}, "three original producer role descriptors")
        roles = receipt["producer_inputs"]
        expected = {"--output": str(args.audit_directory)}
        for role, descriptor in roles.items():
            self.bind(Path(descriptor["path"]), descriptor)
            expected[f"--{role}"], expected[f"--{role}-sha256"] = descriptor["path"], descriptor["sha256"]
        audit_started = self.read(args.audit_directory / "started.json")
        require(audit_started["producer_inputs"] == roles and audit_started["started_ns"] == receipt["started_ns"],
                "original audit input and start witness")
        self.joined_parent(receipt, terminal, audit_started, script=AUDITOR, cap=300, expected=expected)
        plan, worker, train_terminal = (self.read(Path(roles[k]["path"])) for k in ("plan", "worker", "terminal"))
        require(plan["version"] == worker["version"] == TRAIN_VERSION and plan["status"] == "frozen_before_fitting"
            and worker["status"] == "completed" and worker["complete"] is True
            and worker["fits_completed"] == 15 and worker["optimizer_steps"] == 6480
            and worker["pending"] is worker["pending_emission"] is None
            and worker["requires_successful_original_supervisor"] is True
            and worker["plan_sha256"] == receipt["plan_sha256"] == roles["plan"]["sha256"]
            and worker["sources"] == plan["sources"] and worker["inputs"] == plan["inputs"], "same complete training allocation")
        require(all(plan["sources"].get(name) == pin for name, pin in SCIENCE_PINS.items()),
                "qualified scientific sources frozen prospectively")
        for name, pin in plan["sources"].items():
            self.bind(ROOT / name, pin)
        require(AUDITOR in receipt["sources"] and all(plan["sources"].get(k) == v for k, v in receipt["sources"].items()),
                "same prospective independent audit sources")
        require(set(plan["inputs"]) == {"collection_plan", "collection_receipt", "collection_terminal", "engineering",
                    "capacity_plan", "capacity_receipt", "capacity_terminal"}, "exact original input roles")
        for descriptor in plan["inputs"].values():
            self.bind(Path(descriptor["path"]), descriptor)
        run = Path(roles["worker"]["path"]).parent
        self.closed(run, worker, TRAIN_PAYLOADS)
        train_started = self.read(run / "started.json")
        require(train_started["started_ns"] == worker["started_ns"], "original training start witness")
        self.joined_parent(worker, train_terminal, train_started, script=TRAINER, cap=14400, mode="run",
            expected={"--plan": roles["plan"]["path"], "--plan-sha256": roles["plan"]["sha256"], "--output": str(run)})
        # No outcome JSON decoding before both original parents and byte closures.
        summary, audit = self.read(run / "summary.json"), self.read(args.audit_directory / "audit.json")
        require(summary["version"] == TRAIN_VERSION and audit["version"] == AUDIT_VERSION and audit["agreement"] is True
            and audit["technical_condition_requires_successful_original_audit_supervisor"] is True
            and summary["technical_complete_pending_saved_audit"] is True, "original provisional and audited distinction")
        same(summary, audit["producer_summary"])
        same(summary["required"][0], {"name": "common.technical_complete", "value": False,
                                      "relation": "==", "threshold": True, "passes": False})
        same(summary["gates"], gate_records(summary["required"]))
        rules = [{**summary["required"][0], "value": True, "passes": True}, *summary["required"][1:]]
        final = {**summary, "required": rules, "required_passed": summary["required_passed"] + 1,
                 "gates": gate_records(rules), "technical_complete_pending_saved_audit": False}
        same(audit["summary"], final)
        same(audit["fits"], self.read(run / "fits.json")["fits"])
        counts = audit["counts"]
        require(counts["fits"] == 15 and counts["optimizer_steps"] == 6480 and counts["training_events"] == 67
            and counts["work_events"] == 12960 and counts["prediction_files"] == 31
            and counts["training_payloads"] == len(worker["files"]) == 58
            and counts["collection_episodes"] == 90 and counts["collection_payloads"] == 18
            and counts["required_conditions"] == final["required_conditions"] == 29
            and counts["required_passed"] == final["required_passed"] == sum(r["passes"] for r in rules)
            and final["scientific_conditions"] == 28 and final["scientific_passed"] == sum(r["passes"] for r in rules[1:])
            and final["train_counts"]["episodes"] == 54 and final["validation_counts"]["episodes"] == 36,
            "complete audited fits, conditions and paths")
        self.receipt["original_seconds"] = {"training_worker": worker["wall_seconds"], "training_parent": train_terminal["wall_seconds"],
                                             "audit_worker": receipt["wall_seconds"], "audit_parent": terminal["wall_seconds"]}
        return audit["summary"], audit

    @staticmethod
    def dots(ax, values, families=FAMILIES):
        for x, family in enumerate(families):
            selected = []
            for seed, shift, marker in zip(SEEDS, (-.13, 0., .13), ("o", "s", "^"), strict=True):
                value = values[family, seed]; selected.append(value)
                ax.plot(x + shift, value, marker=marker, color=COLORS[family], markerfacecolor="white",
                        linestyle="none", markersize=6)
            mean = math.fsum(selected) / 3
            ax.plot((x - .24, x + .24), (mean, mean), color=COLORS[family], linewidth=2.5)
        ax.set_xticks(range(len(families)), [LABELS[f] for f in families], fontsize=9)
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
        handles.append(Line2D([], [], color="#4b5563", linewidth=2.5, label="Mean of all three seeds"))
        gate = summary["gates"]["protected_readout"]
        gate_text = f"Prospective protected-readout gate: {'PASS' if gate['passes'] else 'FAIL'} ({gate['passed']}/29)"
        chosen = [r for r in data["regime_metrics"] if r["scope"] == "postcorrection"]
        maximum = max(r[GAP] for r in chosen)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
        fig.subplots_adjust(top=.75, bottom=.28, wspace=.12)
        for ax, regime in zip(axes, REGIMES, strict=True):
            rows = [r for r in chosen if r["regime"] == regime]
            self.dots(ax, {(r["family"], r["seed"]): r[GAP] for r in rows if r["family"] != "hold"})
            ax.axhline(next(r[GAP] for r in rows if r["family"] == "hold"), color="#9ca3af", linestyle="--", linewidth=1)
            ax.set_title(f"Sensing length {regime[-1]} | 6 VALID cases", pad=12)
            ax.set_ylim(0, maximum * 1.12 if maximum else 1)
        axes[0].set_ylabel("Later teacher-score gap\nLower is better")
        fig.suptitle("Protected action readout on fixed paths", fontsize=18, y=.975)
        fig.text(.5, .902, gate_text, ha="center", weight="bold")
        fig.text(.5, .847, "Primary domain: nonquery steps >= 5. All 15 final fits are retained.", ha="center")
        fig.legend(handles=handles + [Line2D([], [], color="#9ca3af", linestyle="--", label="Hold-Q reference")],
                   loc="lower center", bbox_to_anchor=(.5, .13), ncol=5, frameon=False, fontsize=8)
        fig.text(.5, .091, "Episode-weighted gaps, then means over fit seeds. Seed dots are not independent environment cohorts.", ha="center", fontsize=9)
        fig.text(.5, .035, "Forced-path teacher imitation; no autonomous utility, scenario shift or architecture superiority claim.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[0]); plt.close(fig)

        changes = [r for r in data["paired_changes"] if r["scope"] == "postcorrection"]
        extent = max(abs(r["delta_" + GAP]) for r in changes)
        extent = extent * 1.2 if extent else 1.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
        fig.subplots_adjust(top=.78, bottom=.28, wspace=.12)
        for ax, regime in zip(axes, REGIMES, strict=True):
            self.dots(ax, {(r["control"], r["seed"]): r["delta_" + GAP] for r in changes if r["regime"] == regime}, CONTROLS)
            ax.axhline(0, color="#64748b", linewidth=1); ax.set_ylim(-extent, extent)
            ax.set_title(f"Sensing length {regime[-1]}", pad=12)
            ax.set_xlabel("Paired control")
        axes[0].set_ylabel("Frozen SPO+ minus control gap\nNegative is better")
        fig.suptitle("Candidate against every paired control", fontsize=18, y=.975)
        fig.text(.5, .88, "Each point uses the same seed and VALID paths. All four controls remain visible.", ha="center")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .115), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, .078, "Signed differences in raw teacher-score units. The 29-condition gate also checks agreement and support.", ha="center", fontsize=9)
        fig.text(.5, .029, "Pretrained is the common 80-epoch parent; each adaptation branch adds 40 epochs with a fresh optimizer.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[1]); plt.close(fig)

        costs = cost_summary(summary, data)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(top=.78, bottom=.30, wspace=.27)
        self.dots(axes[0], {(r["family"], r["seed"]): r["wall_seconds"] for r in data["fit_costs"]})
        axes[0].set_ylabel("Actual per-fit wall seconds")
        axes[0].set_ylim(bottom=0)
        axes[0].set_title("3 shared pretrains + 12 separate adaptations", pad=12)
        phases = costs["physical_stage_seconds"]
        values = [costs["shared_pretraining_seconds"], costs["adaptation_seconds"], costs["fitting_stage_other_seconds"],
                  phases["setup_seconds"], phases["validation_seconds"]]
        axes[1].bar(range(5), values, color=("#64748b", "#175b96", "#cbd5e1", "#a78bfa", "#16846b"), width=.65)
        axes[1].set_xticks(range(5), ("Shared\npretrain", "All 12\nbranches", "Other\nfitting", "Shared\nsetup", "All-model\nVALID"), fontsize=9)
        axes[1].set_ylabel("Actual aggregate seconds")
        axes[1].set_title("Disjoint allocation of measured stage time", pad=12)
        axes[1].grid(axis="y", alpha=.18); axes[1].set_axisbelow(True)
        fig.suptitle("Measured training and evaluation cost", fontsize=18, y=.975)
        fig.text(.5, .885, "Shared pretraining is counted once per seed. Each branch excludes that shared cost.", ha="center")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .16), ncol=4, frameon=False, fontsize=9)
        fig.text(.5, .115, "Fit wall includes fitting, clone construction, final TRAIN rescore, checkpoint publication and overhead.", ha="center", fontsize=9)
        fig.text(.5, .070, "No per-fit VALID timer exists. Collection and audit are separate; this is not deployment latency.", ha="center", fontsize=9)
        fig.text(.5, .022, "All timing components and original supervisor intervals are retained in summary.json and fit-costs.csv.", ha="center", fontsize=9)
        self.save_figure(fig, FIGURES[2]); plt.close(fig)
        return matplotlib.__version__

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
                and not any(p.is_symlink() for p in self.out.parents), "exclusive contained report output")
        self.out.mkdir(exist_ok=False)
        try:
            self.bind(ROOT / SELF, self.descriptor(ROOT / SELF))
            self.bind(ROOT / HELPER, HELPER_PIN)
            self.bind(ROOT / base.HELPER, base.HELPER_PIN)
            summary, audit = self.authenticate()
            data = tables(summary, audit)
            costs = cost_summary(summary, data)
            for key, rows in data.items():
                write_csv(self.out / (key.replace("_", "-") + ".csv"), rows)
            write_csv(self.out / "conditions.csv", [{**r, "strict_upper_bound": r.get("strict_upper_bound")}
                                                  for r in summary["required"]])
            write(self.out / "summary.json", {"version": VERSION, **data, "costs": costs,
                "conditions": summary["required"], "gates": summary["gates"], "required_conditions": 29,
                "required_passed": summary["required_passed"], "scope": summary["scope"],
                "technical_complete": True, "scientific_gate_passes": summary["gates"]["protected_readout"]["passes"],
                "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
                "original_seconds": self.receipt["original_seconds"], "audit_limitations": audit["limitations"],
                "capacity_evidence_scope": summary["capacity_evidence_scope"]})
            self.receipt["matplotlib_version"] = self.figures(summary, data)
            for path, descriptor in self.bound.items():
                require(self.descriptor(Path(path)) == descriptor, "unchanged evidence and sources after reporting")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact thirteen report payloads")
            self.check()
            self.receipt.update(status="completed", inputs=self.bound, gates=summary["gates"],
                counts={key: len(rows) for key, rows in data.items()},
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
            except BaseException as secondary:  # noqa: BLE001 - preserve the original presentation failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--audit-terminal", type=Path, required=True)
    parser.add_argument("--audit-terminal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    Report(parser.parse_args()).execute()


if __name__ == "__main__":
    main()
