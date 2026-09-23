"""Descriptive figures from the closed interruption diagnostic, never a trial pass.

The immutable presentation helper supplies complete tables and figures. This
adapter authenticates the new diagnostic's real supervisor and retains the
original training closure failure in every figure, CSV and output receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/plot_otto_action_focused_interruption.py"
TEST = "tests/test_plot_otto_action_focused_interruption.py"
VERSION = "otto-action-focused-interruption-plot-v1"
HELPER = "scripts/plot_otto_action_focused.py"
HELPER_PIN = "851065d57c65c1f5f9babfacf55cc260bb002d48093fca34b10330f86fecb080"
DIAGNOSTIC = "scripts/diagnose_otto_action_focused_interruption.py"
DIAGNOSTIC_PIN = "0d4de3ff576542323cfe96002512c0f8290126138788454111c3c645ad3de82c"
DIAGNOSTIC_VERSION = "otto-action-focused-interruption-diagnostic-v1"
BANNER = "DESCRIPTIVE ONLY / ORIGINAL TRAINING PARENT UNVERIFIED"
EVIDENCE_SCOPE = "Descriptive saved-output arithmetic only; original training parent unverified"
TIMING_SCOPE = "Worker-reported time; original training parent bounds, exit and reaping unverified"
LIMITS = {"seconds": 180, "rss_bytes": 2 * 1024**3, "output_bytes": 128 * 1024**2,
          "metadata_json_bytes": 16 * 1024**2, "audit_json_bytes": 256 * 1024**2}


def load_helper():
    path = ROOT / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_PIN:
        raise ValueError("immutable qualified figure and table helper")
    spec = importlib.util.spec_from_file_location("_interruption_plot_helper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


original = load_helper()
require, same, write, write_csv = original.require, original.same, original.write, original.write_csv


def checked_counts(body, receipt):
    """Only the verification counter may advance during post-body closure checks."""
    require(isinstance(body, dict) and isinstance(receipt, dict)
        and "checks" in body and "checks" in receipt,
        "body and closed receipt require diagnostic check counters")
    require(type(body["checks"]) is int and type(receipt["checks"]) is int
        and 0 <= body["checks"] <= receipt["checks"],
        "nonnegative integer diagnostic check counter cannot decrease")
    require(set(body) == set(receipt) and all(
        type(value) is type(receipt[name]) and value == receipt[name]
        for name, value in body.items() if name != "checks"),
        "exact non-check diagnostic counts")


def checked_summary(result):
    """Require the false technical condition without altering any saved criterion."""
    require(result["version"] == DIAGNOSTIC_VERSION and result["arithmetic_agreement"] is True
        and result["original_training_process_closure"] is False
        and result["original_process_closure"] is False and result["technical_complete"] is False
        and result["eligible_for_continuation"] is False and result["original_continuation_eligible"] is False
        and result["original_audit"] == "unavailable_not_replaced", "descriptive diagnostic never qualifies original training")
    same(result["scientific_calls"], {"model": 0, "optimizer": 0, "teacher": 0, "native": 0})
    summary = result["producer_summary"]
    same(summary["required"][0], {"name": "common.technical_complete", "value": False,
                                 "relation": "==", "threshold": True, "passes": False})
    gates = original.gate_records(summary["required"])
    same(summary["gates"], gates)
    require(set(result["gates"]) == set(gates), "all three original gates retained")
    for name, gate in result["gates"].items():
        require(set(gate) == {*gates[name], "eligible", "reason"} and gate["eligible"] is False
            and gate["passes"] is False and isinstance(gate["reason"], str) and bool(gate["reason"]),
            "each original gate remains explicitly ineligible")
        same({k: gate[k] for k in gates[name]}, gates[name])
    same(result["summary"], {**summary, "gates": result["gates"], "technical_complete": False,
        "eligible_for_continuation": False, "original_training_process_closure": False})
    require(summary["technical_complete_pending_saved_audit"] is True
        and summary["required_conditions"] == result["counts"]["required_conditions"] == 41
        and summary["required_passed"] == result["counts"]["required_passed"] == sum(r["passes"] for r in summary["required"])
        and result["counts"]["fits"] == 12 and result["counts"]["prediction_files"] == 25
        and result["counts"]["training_payloads"] == 49 and result["counts"]["collection_episodes"] == 90
        and summary["train_counts"]["episodes"] == 54 and summary["validation_counts"]["episodes"] == 36,
        "complete descriptive twelve-fit coverage")
    return summary


def tables(summary, result):
    values = original.tables(summary, result)
    for key, rows in values.items():
        values[key] = [{**row, "evidence_scope": EVIDENCE_SCOPE, "eligible_for_continuation": False,
                       **({"timing_scope": TIMING_SCOPE} if key == "costs" else {})} for row in rows]
    return values


def label_figure(fig, stem):
    """Apply explicit provenance at the final save boundary on every figure."""
    require(stem in original.FIGURES, "only the three complete original figures")
    for item in fig.texts:
        if item.get_text() == "Three separate gates, no all-41 promotion rule. Fresh development screen; no autonomous or unseen-shift result.":
            item.set_text("All three original continuation gates are ineligible. No autonomous or unseen-shift result.")
    if stem == original.FIGURES[2]:
        replacements = {
            "Measured fitting cost, not deployment latency": "Worker-reported fitting time, not deployment latency",
            "Measured fit wall seconds\nIncludes final TRAIN rescore and checkpoint":
                "Worker-reported fit wall seconds\nIncludes final TRAIN rescore and checkpoint",
            "Within-seed measured fit wall-time ratio": "Within-seed worker-reported time ratio",
            "Equal exposure is not equal compute. Timings are authenticated saved measurements; no new timing experiment is performed.":
                "Worker-reported times only. Original training parent bounds, exit and reaping remain unverified.",
            "Collection, shared setup and VALID are excluded from per-fit intervals. Whole training-stage seconds remain in plotted-values.json.":
                "Per-fit intervals exclude collection, setup and VALID. All original training times are worker-reported.",
        }
        replaced = set()
        for item in [*fig.texts, *(ax.yaxis.label for ax in fig.axes)]:
            text = item.get_text()
            if text in replacements:
                item.set_text(replacements[text])
                replaced.add(text)
        require(replaced == set(replacements), "all cost labels explicitly worker-reported")
    fig.text(.5, 1.035, BANNER, ha="center", va="bottom", fontsize=12, weight="bold", color="#8b1e17",
             bbox={"boxstyle": "round,pad=.5", "facecolor": "#fff0ec", "edgecolor": "#8b1e17"})


class Plot(original.Plot):
    def __init__(self, args):
        super().__init__(args)
        self.receipt.update(version=VERSION, limits=LIMITS, scope=EVIDENCE_SCOPE,
            original_training_process_closure=False, technical_complete=False, eligible_for_continuation=False,
            teacher_calls=0, timing_scope=TIMING_SCOPE)

    def read(self, path):
        path = Path(path)
        limit = LIMITS["audit_json_bytes"] if path == self.args.diagnostic_directory / "diagnostic.json" else LIMITS["metadata_json_bytes"]
        require(str(path) in self.bound and self.bound[str(path)]["bytes"] <= limit, "authenticated bounded diagnostic JSON")
        return json.loads(path.read_text())

    def authenticate(self):
        args = self.args
        self.bind(args.diagnostic_plan, args.diagnostic_plan_sha256)
        self.bind(args.diagnostic_directory / "receipt.json", args.diagnostic_receipt_sha256)
        self.bind(args.diagnostic_terminal, args.diagnostic_terminal_sha256)
        plan = self.read(args.diagnostic_plan)
        receipt = self.read(args.diagnostic_directory / "receipt.json")
        terminal = self.read(args.diagnostic_terminal)
        require(plan["version"] == receipt["version"] == DIAGNOSTIC_VERSION
            and plan["status"] == "frozen_before_saved_array_decode"
            and receipt["status"] == "completed" and receipt["arithmetic_agreement"] is True
            and receipt["failures"] == [] and receipt["requires_successful_original_diagnostic_supervisor"] is True
            and plan["original_process_closure"] is receipt["original_process_closure"] is False
            and plan["technical_complete"] is receipt["technical_complete"] is False
            and plan["original_audit"] == receipt["original_audit"] == "unavailable_not_replaced"
            and receipt["plan_sha256"] == args.diagnostic_plan_sha256
            and receipt["sources"] == plan["sources"] and receipt["inputs"] == plan["inputs"],
            "same completed descriptive diagnostic, not original trial qualification")
        require(plan["limits"] == receipt["limits"] == {"seconds": 240, "rss_bytes": 2 * 1024**3,
                                                       "output_bytes": 256 * 1024**2}
            and all(type(receipt[k]) is int and receipt[k] == 0 for k in
                    ("model_calls", "native_calls", "optimizer_calls", "teacher_calls")),
            "bounded saved-only diagnostic scope")
        require(plan["sources"].get(DIAGNOSTIC) == DIAGNOSTIC_PIN, "qualified diagnostic source pin")
        for name, pin in plan["sources"].items():
            self.bind(ROOT / name, pin)
        require(set(plan["inputs"]) == {"training_plan", "worker", "launch", "interruption",
                                      "engineering", "launcher_engineering"}, "six frozen diagnostic input roles")
        for item in plan["inputs"].values():
            self.bind(Path(item["path"]), item)
        self.closed(args.diagnostic_directory, receipt, {"started.json", "diagnostic.json"})
        started = self.read(args.diagnostic_directory / "started.json")
        require(started["inputs"] == receipt["inputs"] and started["started_ns"] == receipt["started_ns"]
            and started["original_process_closure"] is started["technical_complete"] is False,
            "same original diagnostic start and inputs")
        self.joined_parent(receipt, terminal, started, script=DIAGNOSTIC, cap=240, mode="run",
            expected={"--plan": str(args.diagnostic_plan), "--plan-sha256": args.diagnostic_plan_sha256,
                      "--output": str(args.diagnostic_directory)})
        # This absence preserves scope; it is never treated as process success.
        terminal_path = Path(plan["original_terminal_path"])
        require(terminal_path.is_absolute() and terminal_path.is_relative_to(ROOT)
            and ".." not in terminal_path.parts and not terminal_path.exists(), "original training terminal still unavailable")
        result = self.read(args.diagnostic_directory / "diagnostic.json")
        summary = checked_summary(result)
        checked_counts(result["counts"], receipt["counts"])
        self.receipt["diagnostic_seconds"] = {"worker": receipt["wall_seconds"], "closed_parent": terminal["wall_seconds"]}
        self.receipt["worker_reported_training_seconds"] = result["worker_reported_wall_seconds"]
        return summary, result

    def save_figure(self, fig, stem):
        label_figure(fig, stem)
        super().save_figure(fig, stem)

    def execute(self):
        require(self.out.is_absolute() and self.out.is_relative_to(ROOT) and ".." not in self.out.parts
            and not any(p.is_symlink() for p in self.out.parents), "exclusive contained descriptive presentation output")
        self.out.mkdir(exist_ok=False)
        try:
            for name in (SELF, TEST):
                self.bind(ROOT / name, self.descriptor(ROOT / name))
            self.bind(ROOT / HELPER, HELPER_PIN)
            self.bind(ROOT / original.HELPER, original.HELPER_PIN)
            summary, result = self.authenticate()
            data = tables(summary, result)
            for key, rows in data.items():
                write_csv(self.out / (key.replace("_", "-") + ".csv"), rows)
            conditions = [{**r, "strict_upper_bound": r.get("strict_upper_bound"),
                "evidence_scope": EVIDENCE_SCOPE, "eligible_for_continuation": False} for r in summary["required"]]
            gates = [{"gate": k, **v, "evidence_scope": EVIDENCE_SCOPE} for k, v in result["gates"].items()]
            write_csv(self.out / "conditions.csv", conditions)
            write_csv(self.out / "gates.csv", gates)
            write(self.out / "plotted-values.json", {"version": VERSION,
                **{k: v for k, v in data.items() if k != "forecast_metrics"},
                "conditions": conditions, "gates": result["gates"], "scope": EVIDENCE_SCOPE,
                "arithmetic_agreement": True, "original_training_process_closure": False,
                "technical_complete": False, "eligible_for_continuation": False,
                "train_counts": summary["train_counts"], "validation_counts": summary["validation_counts"],
                "worker_reported_stage_seconds": {k: summary[k] for k in ("setup_seconds", "fitting_seconds", "validation_seconds")},
                "worker_reported_training_seconds": self.receipt["worker_reported_training_seconds"],
                "diagnostic_seconds": self.receipt["diagnostic_seconds"], "diagnostic_limitations": result["limitations"],
                "cost_scope": TIMING_SCOPE, "banner": BANNER})
            self.receipt["matplotlib_version"] = self.figures(summary, data)
            for path, item in self.bound.items():
                require(self.descriptor(Path(path)) == item, "unchanged evidence and sources after plotting")
            require({p.name for p in self.out.iterdir()} == original.PAYLOADS, "all fifteen descriptive presentation payloads")
            self.check()
            self.receipt.update(status="completed", inputs=self.bound, gates=result["gates"],
                arithmetic_agreement=True, counts={k: len(v) for k, v in data.items()},
                files={p.name: self.descriptor(p) for p in sorted(self.out.iterdir())},
                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "eligible_for_continuation": False,
                              "receipt": self.descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.failed = True
            self.receipt.update(status="failed", inputs=self.bound, error=repr(error), traceback=traceback.format_exc(),
                                wall_seconds=(time.monotonic_ns() - self.start) / 1e9)
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: self.descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure
                error.add_note(f"Failure receipt publication: {secondary!r}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic-plan", "diagnostic-terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--diagnostic-directory", type=Path, required=True)
    parser.add_argument("--diagnostic-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    Plot(parser.parse_args()).execute()


if __name__ == "__main__":
    main()
