"""Complete saved-output temperature control with a separate analysis freeze.

The frozen common.authenticate is called only as a read-only provenance reader;
common.execute, neural replay and checkpoint deserialization are never invoked.
All three successful phase terminals precede evaluator or NPZ decoding here.
Full fit diagnostics are saved once; summary.json indexes their authenticated
files and carries complete factorial, paired-change and eleven-condition trees.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
from pathlib import Path
from types import SimpleNamespace

import dialogue_calibration_common as common
import numpy as np

from openjev.research.suspend_clock import SuspendClock

ROOT, require = common.ROOT, common.require
VERSION = "dialogue-calibration-report-v1"
LIMITS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
PLAN_PATH = "output/dialogue-calibration-control-v1/report-plan-01.json"
OUTPUT_PATH = "output/dialogue-calibration-control-v1/report-01"
SOURCES = {"scripts/report_dialogue_calibration.py", "tests/test_report_dialogue_calibration.py",
           "src/openjev/research/dialogue_calibration_metrics.py", "tests/test_dialogue_calibration_metrics.py"}
FIT_ORDER = [f"{arm}-{seed}" for seed in common.CONFIG["seeds"] for arm in common.CONFIG["arms"]]
DEV_ROWS, CALIBRATION_ROWS, CALIBRATION_DIALOGUES = 62329, 13333, 512
PHASES = ("prepare", "qualify", "infer")
SCOPE = ("Saved-only development control, not untouched confirmation. Temperatures use calibration TRAIN endpoints only. "
         "No model/encoder/tokenizer calls, checkpoint deserialization, optimizer or official TEST access. "
         "Upstream configuration, cohort selection and internal execution witnesses are authenticated through the frozen reader; "
         "the numerical scorer reuses the qualified original group definitions. Original raw FAIL6/7 remains unchanged.")


def absolute(path):
    return (ROOT / path).resolve()


def descriptor(path):
    return {"sha256": common.sha(path), "bytes": Path(path).stat().st_size}


def pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Budget:
    def __init__(self, out, clock):
        self.out, self.clock = out, clock
        self.deadline = clock.deadline_after(LIMITS["wall_seconds"])
        self.progress = {"operation": "authentication", "completed_fits": []}
        self.last_elapsed_ns = None

    def check(self):
        now = self.clock.now_ns()
        require(now >= self.deadline.started_ns, "Nonregressing analysis clock")
        if now >= self.deadline.expires_ns:
            raise TimeoutError("Suspend-inclusive calibration report deadline expired")
        require(common.peak_rss() <= LIMITS["rss_bytes"], "Analysis RSS cap")
        self.last_elapsed_ns = now - self.deadline.started_ns

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"],
                "Analysis output cap")

    def elapsed(self):
        self.check()
        return self.last_elapsed_ns / 1e9


def validate_analysis_plan(args, budget):
    require(args.plan.resolve() == absolute(PLAN_PATH) and args.out.resolve() == absolute(OUTPUT_PATH),
            "Fixed separate analysis plan and output")
    require(pin(args.plan_sha256) and common.sha(args.plan) == args.plan_sha256, "External analysis plan pin")
    plan = common.read(args.plan)
    require(set(plan) == {"version", "limits", "sources", "control_plan", "phases", "environment",
                          "synthetic_qualification", "out"}
            and plan["version"] == VERSION and plan["limits"] == LIMITS
            and absolute(plan["out"]) == args.out.resolve(), "Exact prospective analysis contract")
    require(set(plan["sources"]) == SOURCES and plan["environment"] == common.environment(),
            "Complete new source set and fixed runtime")
    for name, digest in plan["sources"].items():
        budget.check()
        require(pin(digest) and common.sha(ROOT / name) == digest, "Analysis source pin: " + name)
    require(set(plan["control_plan"]) == {"path", "sha256"}
            and pin(plan["control_plan"]["sha256"]), "Control plan descriptor")
    control_path = absolute(plan["control_plan"]["path"])
    require(common.sha(control_path) == plan["control_plan"]["sha256"], "Pinned control plan before helper use")
    control = common.read(control_path)
    # The upstream reader validates its complete transitive closure. These two
    # directly executed helpers are additionally bound before that reader call.
    for name in ("scripts/dialogue_calibration_common.py", "src/openjev/research/suspend_clock.py"):
        digest = control["sources"].get(name, control["inputs"].get(name))
        require(pin(digest) and common.sha(ROOT / name) == digest, "Direct upstream helper source pin")
    require(set(plan["phases"]) == set(PHASES), "All three phase bindings required")
    for phase, binding in plan["phases"].items():
        require(set(binding) == {"path", "completed_sha256", "terminal"}
                and set(binding["terminal"]) == {"path", "sha256"}
                and pin(binding["completed_sha256"]) and pin(binding["terminal"]["sha256"])
                and absolute(binding["path"]) == absolute(control["outputs"][phase]), "Exact phase binding")
    qualification = plan["synthetic_qualification"]
    require(set(qualification) == {"path", "sha256"} and pin(qualification["sha256"])
            and common.sha(absolute(qualification["path"])) == qualification["sha256"], "Synthetic analysis qualification pin")
    qualified = common.read(absolute(qualification["path"]))
    require(qualified["status"] == "completed" and qualified["actual_exit_code"] == 0
            and qualified["source_sha256"] == plan["sources"] and qualified["model_calls"] == 0
            and qualified["real_task_inputs_read"] is False, "Matching completed artificial qualification")
    return plan, control_path


def check_phase_links(plan, receipts, terminals):
    prepared, qualified, inferred = (receipts[p] for p in PHASES)
    require(prepared["counts"]["dialogues"] == CALIBRATION_DIALOGUES
            and prepared["counts"]["scored_endpoints"] == CALIBRATION_ROWS
            and prepared["source_split"] == "train" and prepared["analysis_role"] == "calibration"
            and prepared["model_calls"] == prepared["neural_calls"] == 0, "Complete calibration preparation")
    prep_pin = plan["phases"]["prepare"]["completed_sha256"]
    prep_terminal_pin = plan["phases"]["prepare"]["terminal"]["sha256"]
    for receipt in (qualified, inferred):
        require(receipt["prepared_sha256"] == prep_pin and receipt["prepared_terminal_sha256"] == prep_terminal_pin
                and receipt["fit_order"] == FIT_ORDER and [f["fit_id"] for f in receipt["fits"]] == FIT_ORDER
                and receipt["model_weight_updates"] == 0 and receipt["optimizer_created"] is False
                and receipt["temperature_applied"] is False and receipt["official_test_opened"] is False
                and receipt["task_metrics_computed"] is False, "Same preparation and all twelve inference-only checkpoints")
    require(qualified["parity_passed"] is True and qualified["completed_forwards"] == 24
            and qualified["projection"]["admitted"] is True
            and qualified["projection"]["threshold_seconds"] == 1800
            and math.isfinite(qualified["projection"]["total_seconds"])
            and 0 < qualified["projection"]["total_seconds"] <= 1800
            and qualified["projection"]["preparation_seconds"] == terminals["prepare"]["wall_seconds"],
            "Completed fixed parity and cost admission")
    require(inferred["qualification_sha256"] == plan["phases"]["qualify"]["completed_sha256"]
            and inferred["qualification_terminal_sha256"] == plan["phases"]["qualify"]["terminal"]["sha256"]
            and inferred["completed_forwards"] == 12 * CALIBRATION_DIALOGUES
            and inferred["endpoint_rows_per_fit"] == CALIBRATION_ROWS
            and inferred["total_endpoint_rows"] == 12 * CALIBRATION_ROWS, "Complete admitted calibration inference")


def authenticate(args, budget):
    plan, control_path = validate_analysis_plan(args, budget)
    phases = plan["phases"]
    # Fixed historical infer request, used solely by the read-only validator.
    # No execute/handshake/model-loading function is called.
    historical = SimpleNamespace(command="infer", plan=control_path,
                                 plan_sha256=plan["control_plan"]["sha256"], out=absolute(phases["infer"]["path"]))
    ctx = common.authenticate(historical, budget)
    require(ctx["plan_path"] == control_path, "Exact authenticated control plan path")
    receipts, terminals = {}, {}
    for phase in PHASES:
        binding = phases[phase]
        budget.check()
        receipts[phase] = common.authenticate_output(absolute(binding["path"]), binding["completed_sha256"], ctx, phase)
        terminals[phase] = common.authenticate_terminal(absolute(binding["terminal"]["path"]),
                                                       binding["terminal"]["sha256"], receipts[phase])
    check_phase_links(plan, receipts, terminals)
    require(ctx["science"]["fit_order"] == FIT_ORDER, "Exact original seed-outer ordering")
    for phase in ("qualify", "infer"):
        directory = absolute(phases[phase]["path"])
        for record in receipts[phase]["fits"]:
            budget.check()
            name = record["fit_id"]
            path = directory / name / "completed.json"
            require(common.sha(path) == record["completed_sha256"], "Nested complete fit pin")
            fit = common.read(path)
            require(fit == {k: v for k, v in record.items() if k != "completed_sha256"}
                    and fit["status"] == "completed", "Root and per-fit complete metadata agree")
            common.manifest(path.parent, fit["files"])
            arm, seed = name.rsplit("-", 1)
            require(fit["arm"] == arm and fit["seed"] == int(seed), "Exact arm and seed")
            witness = fit["witness"]
            count = 2 if phase == "qualify" else CALIBRATION_DIALOGUES
            for operation in ("public_forward", "encoding", "memory_forward"):
                require(witness["counts"][operation + "_attempts"] == witness["counts"][operation + "_returns"] == count,
                        "Every complete public dialogue returned")
            require(witness["synthetic_injection"] is False and witness["optimizer_created"] is False
                    and witness["temperature_applied"] is False
                    and witness["restored_sha256"] == witness["final_sha256"], "Unchanged production final checkpoint")
            original = common.read(Path(ctx["prior"].run) / name / "completed.json")
            require(witness["restored_sha256"]["encoder"] == original["final_encoder_sha256"]
                    and witness["checkpoint_sha256"] == original["checkpoint"]["sha256"], "Original authenticated final state identity")
            if phase == "infer":
                require(fit["dialogues"] == CALIBRATION_DIALOGUES and fit["endpoint_rows"] == CALIBRATION_ROWS,
                        "Complete per-checkpoint calibration cohort")
            else:
                require(len(fit["cases"]) == 2 and all(c["parity"]["canonical_first_argmax_equal"] is True
                        and c["parity"]["maximum_supported_log_error"] <= 1e-5
                        and c["parity"]["maximum_probability_error"] <= 1e-6 for c in fit["cases"]),
                        "Both predeclared full-dialogue parity witnesses")
    require(common.read(absolute(phases["qualify"]["path"]) / "projection.json") == receipts["qualify"]["projection"],
            "Saved qualification projection agrees")
    published = common.read(Path(ctx["prior"].report) / "summary.json")
    require(published["technical_validity_passed"] is True and published["technical_complete_fits"] == 12
            and set(published["fits"]) == set(FIT_ORDER)
            and published["continuation"]["passed"] is False and published["continuation"]["checks_passed"] == 6,
            "Authenticated original raw FAIL6/7 retained")
    return plan, ctx, receipts, terminals, published


def read_lines(path, budget):
    result = []
    with path.open() as stream:
        for line in stream:
            budget.check()
            require(bool(line.strip()), "Nonempty canonical row records")
            result.append(json.loads(line))
    return result


def load_packet(path, expected_rows):
    with np.load(path, allow_pickle=False) as packet:
        require(set(packet.files) == {"log_probs", "row_indices"}, "Exact raw saved prediction members")
        logs, indices = packet["log_probs"], packet["row_indices"]
    require(logs.dtype == np.float32 and logs.shape == (expected_rows, 12)
            and indices.dtype == np.int64 and np.array_equal(indices, np.arange(expected_rows, dtype=np.int64)),
            "Complete float32 raw predictions and exact canonical row identity")
    return logs, indices


def compute(args, budget, plan, ctx, receipts, terminals, published):
    # This import occurs only after the complete source and phase chain passes.
    from openjev.research import dialogue_calibration_metrics as metrics

    prepared = absolute(plan["phases"]["prepare"]["path"])
    inferred = absolute(plan["phases"]["infer"]["path"])
    dev_rows = read_lines(Path(ctx["prior"].run) / "evaluation-rows.jsonl", budget)
    cal_rows = read_lines(prepared / "evaluation-rows.jsonl", budget)
    selection = common.read(prepared / "selection.json")
    require(len(dev_rows) == DEV_ROWS and len(cal_rows) == CALIBRATION_ROWS, "Complete original DEV and calibration rows")
    selected = selection["selected_ids"]
    require(len(selected) == len(set(selected)) == CALIBRATION_DIALOGUES
            and list(dict.fromkeys(r["dialogue_id"] for r in cal_rows)) == selected,
            "Exact prospective 512-dialogue calibration identity and order")
    fits, file_index, fit_panels, temperatures = {}, {}, {}, {}
    (args.out / "fits").mkdir(exist_ok=False)
    for name in FIT_ORDER:
        budget.progress.update(operation="fit-temperature-and-report", fit_id=name)
        budget.check()
        raw, dev_indices = load_packet(Path(ctx["prior"].run) / name / "predictions.npz", DEV_ROWS)
        calibration, calibration_indices = load_packet(inferred / name / "predictions.npz", CALIBRATION_ROWS)
        value = metrics.evaluate_fit(raw, dev_indices, dev_rows, calibration, calibration_indices, cal_rows)
        require(value["raw"] == published["fits"][name], "Recomputed raw fit differs from original published result: " + name)
        require(value["calibration_validation"]["dialogues"] == CALIBRATION_DIALOGUES
                and value["calibration_validation"]["rows"] == CALIBRATION_ROWS, "Every calibration endpoint fitted")
        path = args.out / "fits" / (name + ".json")
        common.write(path, value)
        fits[name] = value
        file_index[name] = {"fit_id": name, "path": path.relative_to(args.out).as_posix(), **descriptor(path)}
        fit_panels[name] = {route: {panel: {key: value[route]["panels"][panel][key] for key in ("micro", "strata", "macro_three")}
                                  for panel in ("all", "seen", "unseen")} for route in metrics.ROUTES}
        temperatures[name] = value["temperature"]
        budget.progress["completed_fits"].append(name)
        del raw, dev_indices, calibration, calibration_indices
        budget.storage()
    summary = metrics.summarize(fits)
    require(summary["continuation"]["original_conditions"]["raw"] == published["continuation"],
            "Exact original seven-check raw outcome remains unchanged")
    require(summary["continuation"]["checks_total"] == 11
            and summary["continuation"]["raw_result_revised"] is False
            and summary["continuation"]["decision_conditions_unchanged"] is True, "Fixed complete calibration-control conditions")
    separate_costs = {phase: {"parent_wall_seconds": terminals[phase]["wall_seconds"],
                              "receipt_wall_seconds": receipts[phase]["wall_seconds"],
                              "peak_rss_bytes": receipts[phase]["peak_rss_bytes"],
                              "sampled_mps_driver_max_bytes": receipts[phase]["sampled_mps_driver_max_bytes"]}
                      for phase in PHASES}
    summary.update(version=VERSION, status="completed", technical_validity_passed=True, complete_fits=12,
                   fits=file_index, fit_panels=fit_panels, temperatures=temperatures,
                   fit_storage="Complete three-route service/type/bin panels and validations are in the 12 indexed fit JSON files.",
                   plan_sha256=args.plan_sha256, control_plan_sha256=plan["control_plan"]["sha256"],
                   phases=plan["phases"], model_calls=0, official_test_opened=False,
                   original_raw_result={"passed": False, "checks_passed": 6, "checks_total": 7,
                                        "all_fit_metrics_exactly_reproduced": True,
                                        "summary_sha256": ctx["prior"].report_summary_sha256,
                                        "summary_path": str(Path(ctx["prior"].report) / "summary.json")},
                   costs={"separate_phases": separate_costs,
                          "nonnested_prior_phase_seconds": math.fsum(t["wall_seconds"] for t in terminals.values()),
                          "original_training": published["costs"],
                          "scope": "Parent prepare/qualify/infer intervals are separate and summed once. Original training is separate, not added. "
                                   "This report's final interval is in receipt.json; audit and figure costs are not included. RSS/MPS values are not additive."},
                   scope=SCOPE)
    return summary


def report_text(summary):
    gate = summary["continuation"]
    lines = ["# Complete output-temperature control", "",
             (f"Technical coverage: 12/12. Calibration control: {'PASS' if gate['passed'] else 'FAIL'} "
              f"({gate['checks_passed']}/11). Original raw result remains FAIL (6/7)."), "", SCOPE, "",
             "| Fit | Beta | Fit location | Raw unseen NLL | Normalized unseen NLL | Calibrated unseen NLL | Calibrated unseen Brier |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for name in FIT_ORDER:
        t, panels = summary["temperatures"][name], summary["fit_panels"][name]
        a, b, c = (panels[route]["unseen"]["micro"] for route in ("raw", "normalized", "calibrated"))
        lines.append(f"| {name} | {t['beta']:.8g} | {t['location']} | {a['nll']:.8g} | {b['nll']:.8g} | {c['nll']:.8g} | {c['brier']:.8g} |")
    lines += ["", "| Condition | Result |", "|---|---|"]
    lines.extend(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} |" for check in gate["checks"])
    lines += ["", ("Every seed, paired change, factorial mean and normalization-only drift is retained in [summary.json](summary.json). "
              "Its fits index links the twelve complete diagnostics, including seen/unseen and per-service/type/bin panels. "
              "Temperature fitting used calibration endpoints only; unchanged predictions cannot establish a new memory or architecture benefit."), ""]
    return "\n".join(lines)


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget = clock = handler = None
    sources = {}
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        clock = SuspendClock()
        require(clock.backend in common.NATIVE, "Native suspend-inclusive analysis clock required")
        budget = Budget(args.out, clock)

        def timeout(*_):
            raise TimeoutError("Supplementary calibration analysis alarm expired")

        handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        common.write(args.out / "started.json", {"version": VERSION, "request": request, "limits": LIMITS,
                     "clock_backend": clock.backend, "started_ns": budget.deadline.started_ns,
                     "deadline_ns": budget.deadline.expires_ns, "model_calls": 0})
        plan, ctx, receipts, terminals, published = authenticate(args, budget)
        sources = plan["sources"]
        summary = compute(args, budget, plan, ctx, receipts, terminals, published)
        common.write(args.out / "summary.json", summary)
        with (args.out / "report.md").open("x") as stream:
            stream.write(report_text(summary))
        # Recheck all opaque inputs and external terminals after numerical work.
        # This remains saved-only authentication, never neural inference.
        authenticate(args, budget)
        budget.storage()
        files = common.members(args.out)
        expected = {"started.json", "summary.json", "report.md"} | {f"fits/{name}.json" for name in FIT_ORDER}
        require(set(files) == expected, "Exact successful report payload closure")
        budget.check()
        elapsed = budget.last_elapsed_ns
        receipt = {"version": VERSION, "status": "completed", "technical_validity_passed": True,
                   "complete_fits": 12, "request": request, "plan_sha256": args.plan_sha256,
                   "control_plan_sha256": plan["control_plan"]["sha256"], "source_sha256": sources,
                   "phases": plan["phases"], "files": files, "limits": LIMITS, "model_calls": 0,
                   "official_test_opened": False, "original_raw_failure_preserved": True,
                   "original_raw_checks_passed": 6, "original_raw_checks_total": 7,
                   "continuation_passed": summary["continuation"]["passed"],
                   "checks_passed": summary["continuation"]["checks_passed"], "checks_total": 11,
                   "clock_backend": clock.backend, "started_ns": budget.deadline.started_ns,
                   "finished_ns": budget.deadline.started_ns + elapsed, "deadline_ns": budget.deadline.expires_ns,
                   "elapsed_ns": elapsed, "wall_seconds": elapsed / 1e9, "timing_available": True,
                   "peak_rss_bytes": common.peak_rss(), "scope": SCOPE,
                   "new_control_total_seconds": summary["costs"]["nonnested_prior_phase_seconds"] + elapsed / 1e9,
                   "timing_scope": "Analysis start through final pre-publication check; publication rechecked before successful return. "
                                   "Prior parent phase intervals are nonnested. Future audit/figure costs are excluded."}
        common.write(args.out / "receipt.json", receipt)
        budget.storage()
        return receipt
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            common.write(args.out / "failed.json", {"version": VERSION, "status": "failed", "request": request,
                         "source_sha256": sources, "error_type": type(error).__name__, "error": str(error),
                         "model_calls": 0, "scientific_result_qualified": False, "timing_available": False,
                         "wall_seconds": None, "elapsed_ns": None,
                         "last_successful_elapsed_ns": None if budget is None else budget.last_elapsed_ns,
                         "progress": None if budget is None else budget.progress,
                         "scope": "Original failure and partial outputs retained; no automatic retry or partial passing claim."})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original failure
            error.add_note("Failure receipt error: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    result = execute(arguments)
    print(json.dumps({"status": result["status"], "continuation_passed": result["continuation_passed"],
                      "checks_passed": result["checks_passed"], "receipt_sha256": common.sha(arguments.out / "receipt.json")}))
