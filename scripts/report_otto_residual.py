"""Render a closed DEV reanalysis from saved audited JSON, never numerical files.

This is a publication step, not another evaluation or admission. Original
numerical-payload verification is inherited from the pinned completed closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-residual-reanalysis-v1"
RUNNER = "scripts/run_otto_residual_reanalysis.py"
SEEDS = (309000001, 309000002, 309000003)
TAUS = (.01, .1, 1., 10.)
BASELINES = ("pretrained", "joint_aux", "last_error", "trace_delta")
RLS = ("rls_full", "rls_diagonal", "rls_shrink_025", "rls_shrink_050", "rls_shrink_075")
METHODS = BASELINES + RLS
PANELS = (("lambda3", "later"), ("lambda3", "full"), ("lambda4", "later"), ("lambda4", "full"))
LABELS = ("Pretrained GRU", "Joint auxiliary GRU", "Last error", "Trace delta",
          "Full RLS (candidate)", "Diagonal RLS", "Full RLS, read x0.25", "Full RLS, read x0.50", "Full RLS, read x0.75")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
    return path


def digest(path):
    path = regular(path)
    require(path.suffix not in {".npz", ".npy", ".h5", ".pt", ".pth", ".safetensors"}, "no numerical file access")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    path = regular(path)
    require(path.suffix == ".json", "JSON metadata only")
    return json.loads(path.read_text())


def pinned(record):
    require(set(record) == {"path", "sha256", "bytes"}
            and digest(record["path"]) == {k: record[k] for k in ("sha256", "bytes")}, "evidence descriptor pin")
    return read(record["path"])


def phase(inputs, prefix, phase_name):
    plan, receipt, terminal = (pinned(inputs[prefix + suffix]) for suffix in ("_plan", "_receipt", "_terminal"))
    require(plan["version"] == receipt["version"] == VERSION and plan["phase"] == receipt["phase"] == phase_name
            and plan["status"] == "frozen_before_execution" and receipt["status"] == "completed"
            and receipt["complete"] is True and receipt["requires_successful_original_supervisor"] is True
            and receipt["plan_sha256"] == inputs[prefix + "_plan"]["sha256"]
            and receipt["inputs"] == plan["inputs"] and receipt["sources"] == plan["sources"]
            and receipt["views_completed"] == 72 and receipt["caches_completed"] == 3
            and all(receipt[k] is None for k in ("pending", "pending_io", "pending_emission", "pending_log")),
            "completed original phase")
    expected_counts = {"checkpoint_decodes": 9 if phase_name == "evaluate" else 0,
        "array_decodes": 85 if phase_name == "evaluate" else 76,
        "model_constructions": 6 if phase_name == "evaluate" else 0,
        "model_construction_attempts": 6 if phase_name == "evaluate" else 0,
        "cache_builds": 3 if phase_name == "evaluate" else 0, "replays": 72,
        "teacher_calls": 0, "native_calls": 0, "optimizer_steps": 0, "confirm_decodes": 0}
    require(receipt["counts"] == expected_counts and all(type(v) is int for v in receipt["counts"].values()),
            "exact completed work counts")
    directory = regular(inputs[prefix + "_receipt"]["path"]).parent
    require(set(receipt["files"]) == set(plan["payloads"]), "original payload manifest roster")
    started_path = directory / "started.json"
    require(digest(started_path) == receipt["files"]["started.json"], "original worker metadata pin")
    started = read(started_path)
    launch = started["launch"]
    require(all(terminal.get(k) == v for k, v in launch.items())
            and terminal["status"] == "completed" and type(terminal["returncode"]) is int
            and terminal["returncode"] == 0 and terminal["timed_out"] is False
            and terminal["group_absent"] is True and terminal["timing_available"] is True
            and terminal["cleanup"]["reaped"] is True and terminal["cleanup"]["group_absent"] is True
            and terminal["cleanup"]["errors"] == [] and terminal["error"] is terminal["clock_error"] is None
            and terminal["cap_seconds"] == 900 and terminal["cwd"] == str(ROOT)
            and terminal["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
            and terminal["pid"] == terminal["pgid"] != terminal["parent_pid"]
            and terminal["deadline_ns"] == terminal["started_ns"] + 900 * 10**9
            and terminal["started_ns"] <= receipt["started_ns"] == started["started_ns"]
            < receipt["finished_ns"] <= terminal["finished_ns"] < terminal["deadline_ns"], "original successful supervisor")
    command = list(terminal["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:3] == [str(ROOT / ".venv/bin/python"), str(ROOT / RUNNER), "run"]
            and len(command) == 11 and len(set(command[3::2])) == 4, "original phase command")
    options = dict(zip(command[3::2], command[4::2], strict=True))
    require(set(options) == {"--plan", "--plan-sha256", "--supervision", "--output"}
            and regular(options["--plan"]) == regular(inputs[prefix + "_plan"]["path"])
            and options["--plan-sha256"] == inputs[prefix + "_plan"]["sha256"]
            and options["--output"] == str(directory)
            and digest(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and read(options["--supervision"]) == launch, "original argument and launch joins")
    for name, sha in plan["sources"].items():
        require(not Path(name).is_absolute() and digest(name)["sha256"] == sha, "unchanged bound source")
    for key in ("clock_source_sha256", "watchdog_sha256"):
        source = "src/openjev/research/suspend_clock.py" if key.startswith("clock") else "scripts/supervise_dialogue_observation_v2.py"
        require(terminal[key] == plan["sources"][source], "original supervisor source identity")
    for evidence in plan["inputs"].values():
        pinned(evidence)
    return plan, receipt, terminal, directory


def family(method, tau):
    return method if method in BASELINES else f"{method}@tau={tau:g}"


def number(value):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0, "finite nonnegative saved gap")
    return value


def authenticate(path, sha):
    require(path.is_absolute() and digest(path)["sha256"] == sha, "external closure pin")
    closure = read(path)
    require(closure["version"] == VERSION and closure["technical_complete"] is True
            and closure["confirmation_execution_admitted"] is False and closure["old_test_access"] is False
            and closure["prior_screen_remains_closed"] is True and closure["prior_confirmation_admitted"] is False,
            "closed prior-development reanalysis only")
    audit_plan, audit_receipt, audit_parent, audit_dir = phase(closure["inputs"], "audit", "audit")
    producer_plan, producer_receipt, producer_parent, _ = phase(audit_plan["inputs"], "producer", "evaluate")
    require(producer_parent["finished_ns"] <= audit_parent["started_ns"]
            and producer_plan["sources"] == audit_plan["sources"]
            and producer_plan["lineage"] == audit_plan["lineage"]
            and closure["original_audit_seconds"] == (audit_parent["finished_ns"] - audit_parent["started_ns"]) / 1e9,
            "paired producer and later audit")
    audit = pinned(closure["audit_output"])
    require(regular(closure["audit_output"]["path"]) == audit_dir / "audit.json"
            and digest(audit_dir / "audit.json") == audit_receipt["files"]["audit.json"]
            and digest(audit_dir / "summary.json") == audit_receipt["files"]["summary.json"], "audited report pins")
    decision = audit["decision"]
    summary = read(audit_dir / "summary.json")
    require(audit["version"] == summary["version"] == VERSION and summary["phase"] == "audit"
            and decision == closure["decision"] == summary["decision"]
            and summary["counts"] == audit_receipt["counts"]
            and audit["requires_successful_original_supervisor"] is True
            and decision["technical_complete"] is True and decision["stage"] == "dev"
            and decision["candidate"] == "rls_full" and decision["selected_tau"] in TAUS
            and decision["reports"] == len(audit["reports"]) == len(audit["verified_replays"]) == 72,
            "same complete audited decision")
    conditions = decision["conditions"]
    names = {"technical_completion"} | {regime + ":P4:" + suffix for regime in ("lambda3", "lambda4")
        for suffix in ("supported_cases", "later_gap_10pct", "full_gap_nonregression",
                       *(f"seed_{seed}_nonregression" for seed in SEEDS))}
    require(len(conditions) == decision["total_conditions"] == 13 and {row["name"] for row in conditions} == names
            and all(type(row["passed"]) is bool for row in conditions)
            and decision["passed_conditions"] == sum(row["passed"] for row in conditions)
            and type(decision["passed"]) is bool and decision["passed"] == all(row["passed"] for row in conditions)
            and closure["status"] == ("DEV_PASS" if decision["passed"] else "DEV_FAIL")
            and closure["confirmation_eligible_for_separate_registration"] is decision["passed"], "preserved complete gate")
    reports = {(r["family"], r["seed"]): r for r in audit["reports"]}
    expected = {(family(method, tau), seed) for method in METHODS
                for tau in ((None,) if method in BASELINES else TAUS) for seed in SEEDS}
    require(len(reports) == 72 and set(reports) == expected, "all methods, ratios and fit seeds")
    manifest = audit["reports"][0]["identity_manifest"]
    require(len(manifest) == 18 and sum(row["length"] for row in manifest) == 4816, "original 18 paths and 4816 rows")
    for report in reports.values():
        require(report["identity_manifest"] == manifest and report["stage"] == "dev"
                and report["query_period"] == 4 and report["episodes"] == 18, "matched fixed-path reports")
        for regime, scope in PANELS:
            number(report["scopes"][scope]["by_regime"][regime]["case_weighted_raw_gap"])
    require([row["tau"] for row in decision["selection"]["grid"]] == list(TAUS)
            and decision["selection"]["uses_confirmation_for_selection"] is False, "entire DEV selection grid")
    contrasts = decision["mechanistic_contrasts"]
    require(len(contrasts) == 16 and {(r["tau"], r["control"]) for r in contrasts}
            == {(t, family(method, t)) for t in TAUS for method in RLS[1:]}, "all mechanism contrasts")
    for row in contrasts:
        require([p["regime"] for p in row["panels"]] == ["lambda3", "lambda4"], "both contrast settings")
        for panel in row["panels"]:
            require([p["seed"] for p in panel["paired"]] == list(SEEDS), "all paired contrast seeds")
            for key in ("candidate_later", "control_later", "candidate_full", "control_full"):
                number(panel[key])
    return closure, decision, reports, (producer_receipt, producer_parent), (audit_receipt, audit_parent)


def gap(reports, method, tau, seed, regime, scope):
    return reports[family(method, tau), seed]["scopes"][scope]["by_regime"][regime]["case_weighted_raw_gap"]


def mean(reports, method, tau, regime, scope):
    return math.fsum(gap(reports, method, tau, seed, regime, scope) for seed in SEEDS) / 3


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in
                     (headers, ["---"] * len(headers), *rows))


def fmt(value):
    return f"{value:.9g}"


def link(path, output):
    return "[" + Path(path).name + "](<" + os.path.relpath(path, output) + ">)"


def render(path, sha, output):
    require(output.is_absolute() and output.is_relative_to(ROOT) and ".." not in output.parts
            and output.parent.is_dir() and not output.exists() and not output.is_symlink()
            and not any(p.is_symlink() for p in output.parents), "fresh contained report directory")
    closure, decision, reports, producer, audit = authenticate(path, sha)
    tau = decision["selected_tau"]
    output.mkdir()
    # Plotting is the only numerical-library use; no experiment module or data decoder is imported.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    colors, markers = ("#1764a0", "#ca6b16", "#38843c"), ("o", "s", "^")
    for ax, (regime, scope) in zip(axes.flat, PANELS, strict=True):
        ax.axhspan(3.55, 4.45, color="#e8f0fa", zorder=0)
        for i, (seed, color, marker) in enumerate(zip(SEEDS, colors, markers, strict=True)):
            ax.scatter([gap(reports, method, tau, seed, regime, scope) for method in METHODS],
                       [j + (i - 1) * .17 for j in range(9)], c=color, marker=marker, s=37,
                       label=f"Fit {seed}", zorder=3)
        ax.scatter([mean(reports, method, tau, regime, scope) for method in METHODS], range(9),
                   marker="D", s=27, color="#151b24", label="Mean of 3 fits", zorder=4)
        ax.set_yticks(range(9), LABELS, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="#dce1e6", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_title(f"{regime.replace('lambda', 'Lambda ')} | {scope} non-query steps", fontsize=12)
        ax.set_xlabel("Case-weighted raw score gap (lower is better)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"OpenJev residual estimators | {closure['status']} ({decision['passed_conditions']}/13) | common tau={tau:g}\n"
                 "18 prior DEV paths; fixed recorded actions; fit variation shown without confidence intervals", fontsize=14)
    fig.savefig(output / "methods.png", dpi=180, facecolor="white")
    plt.close(fig)
    rows = [[method, str(seed), *(fmt(gap(reports, method, tau, seed, regime, scope)) for regime, scope in PANELS)]
            for method in METHODS for seed in SEEDS]
    heads = ["Method", "Fit seed", "Lambda 3 later", "Lambda 3 full", "Lambda 4 later", "Lambda 4 full"]
    text = ["# OpenJev residual-estimator DEV reanalysis", "",
        (f"**{closure['status']}: {decision['passed_conditions']}/13 conditions passed.** "
        f"The sole candidate is full RLS, selected at tau={tau:g} by the registered worst-setting mean later-gap rule."), "",
        ("This separately registered reanalysis uses 18 previously collected development paths (4,816 retained rows), "
        "two settings and three fit seeds. The first screen remains closed after its technical failure. "
        "These are fixed-path score-imitation results, not independent held-out evidence or autonomous gameplay. "
        "No novelty, world-model, biological-learning, calibration or speedup claim follows from this screen. "
        "Neither prior confirmation nor the earlier study's TEST is opened. Confirmation execution is not admitted."), "",
        "![Every method and fit seed at the same selected tau](methods.png)", "",
        ("Lower raw score gap is better. Full covers all non-query steps; later covers non-query steps at step 5 or later "
        "(zero-based). Query period is four. Each point is one fit; diamonds average all three fits. "
        "These fits share the same paths and are not three independent environment samples. No confidence intervals are estimated."), "",
        "## Every method and fit seed", "", (f"All RLS methods use the same selected tau={tau:g}; ordinary controls have no tau. "
        "No control or best seed replaces the registered candidate."), "", table(heads, rows), "",
        "## Full-RLS selection grid", "", table(["Tau", "Lambda 3 mean later", "Lambda 4 mean later", "Worst setting", "Selected"],
        [[fmt(r["tau"]), fmt(r["later_means"]["lambda3"]), fmt(r["later_means"]["lambda4"]), fmt(r["objective"]),
          "yes" if r["tau"] == tau else ""] for r in decision["selection"]["grid"]]), "",
        table(["Tau", *heads[1:]], [[fmt(t), str(seed), *(fmt(gap(reports, "rls_full", t, seed, regime, scope))
              for regime, scope in PANELS)] for t in TAUS for seed in SEEDS]), "", "## Ordinary usefulness controls", "",
        "All four registered controls are shown below. Bold values mark the lowest mean within each column; ties remain ties.", ""]
    best = {panel: min(mean(reports, method, tau, *panel) for method in BASELINES) for panel in PANELS}
    text += [table(["Control", *heads[2:]], [[method, *[("**" + fmt(value) + "**" if value == best[panel] else fmt(value))
              for panel in PANELS for value in (mean(reports, method, tau, *panel),)]] for method in BASELINES]), "",
             "## All 13 continuation conditions", ""]
    conditions = []
    for row in decision["conditions"]:
        detail = "Original producer and audit completed successfully."
        if "actual" in row:
            detail = f"Supported cases {row['actual']}; required {row['required']}."
        elif "candidate" in row:
            detail = f"Candidate {fmt(row['candidate'])}; controls " + ", ".join(f"{k}={fmt(v)}" for k, v in row["controls"].items())
        conditions.append([row["name"], "PASS" if row["passed"] else "FAIL", detail])
    text += [table(["Condition", "Saved result", "Evidence"], conditions), "",
        ("Later improvement requires at least 10% below the best ordinary control and strict improvement. "
        "Full-gap and each paired-fit later-gap checks require no regression against the best ordinary control. "
        "All 13 conditions must pass; displayed rounding does not change the saved decision."), "",
        "## Separate mechanism comparisons", "",
        ("The diagonal covariance and weaker-read controls do not select tau or replace the full-RLS candidate. "
        "Weaker reads share the full posterior and writes; only the applied correction is scaled. "
        "The table uses the selected common tau and saved audited mean comparisons."), ""]
    contrasts = decision["mechanistic_contrasts"]
    text += [table(["Control", "Setting", "Full RLS later", "Control later", "Full RLS full", "Control full"],
        [[r["control"], p["regime"], *(fmt(p[key]) for key in ("candidate_later", "control_later", "candidate_full", "control_full"))]
         for r in contrasts if r["tau"] == tau for p in r["panels"]]), "",
        table(["Tau", "Full-vs-diagonal condition conjunction", "Confirmed with usefulness"],
        [[fmt(r["tau"]), "PASS" if r["contrast_passed"] else "FAIL", str(r["confirmed_with_usefulness"]).lower()]
         for r in contrasts if r["control"].startswith("rls_diagonal@")]), "",
        "A DEV covariance contrast is a separate diagnostic, not a confirmation or an overall usefulness pass.", "",
        "## Recorded execution cost", ""]
    text += [table(["Phase", "Original supervisor seconds", "Worker seconds", "Peak RSS MiB", "Array decodes", "Models", "Views"],
        [[name, fmt((parent["finished_ns"] - parent["started_ns"]) / 1e9), fmt(receipt["wall_seconds"]),
          fmt(receipt["peak_rss_bytes"] / 1024**2), receipt["counts"]["array_decodes"],
          receipt["counts"]["model_constructions"], receipt["views_completed"]]
         for name, (receipt, parent) in (("Producer", producer), ("Independent saved-output audit", audit))]), "",
        ("These are total instrumented process costs, including IO and checks, not matched algorithm latency benchmarks. "
        "The collector scans resource usage at every callback. Evaluation and audit check the deadline on each callback "
        "and poll RSS/output usage every 250 ms, with forced IO-boundary checks."), "", "## Evidence", "",
        "Closure: " + link(path, output) + ". Saved audited reports: [audit.json in complete evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-residual-reanalysis-v1).",
        "Audit evidence: " + ", ".join(link(r["path"], output) for r in closure["inputs"].values()) + ".",
        "Producer evidence: " + ", ".join(link(r["path"], output) for r in
             read(closure["inputs"]["audit_plan"]["path"])["inputs"].values()) + ".",
        "Protocol: " + link(ROOT / "research/otto-residual-reanalysis-protocol.md", output) + ".",
        ("The renderer verifies the closure, saved report and metadata hashes, source pins, and original process joins. "
        "Numerical payload validation is inherited from the completed audit; this publication step performs no array/checkpoint "
        "decode, model call, native simulation, teacher call or gate re-evaluation."), ""]
    (output / "README.md").write_text("\n".join(text))
    receipt = {"version": "otto-residual-report-v1", "status": "completed", "closure": {"path": str(path), **digest(path)},
        "audit_output": closure["audit_output"], "renderer": {"path": str(Path(__file__).resolve()), **digest(Path(__file__).resolve())},
        "decision_status": closure["status"], "passed_conditions": decision["passed_conditions"],
        "scientific_array_decodes": 0, "checkpoint_decodes": 0, "model_calls": 0, "native_calls": 0,
        "files": {name: digest(output / name) for name in ("README.md", "methods.png")}}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.closure, args.closure_sha256, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
