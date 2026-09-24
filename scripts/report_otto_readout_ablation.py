"""Publish saved, independently audited readout-ablation JSON after closure.

The registered runner authenticates original process closures and opaque payload
hashes. This publication step never decodes a numerical payload or runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/run_otto_readout_ablation.py"
RUNNER_SHA256 = "0eec16a53e0453956c79a4efa9c44b2afd9ccd77927d72896e37f7e1fa08db7b"
VERSION = "otto-readout-state-ablation-report-v1"
SEEDS = (309000001, 309000002, 309000003)
ARMS = ("pretrained", "action_residual_only", "both_readouts", "full_joint")
LABELS = ("Pretrained parent", "Action residual only", "Both readouts", "Full joint")
REGIMES = ("lambda3", "lambda4")
CONTROLS = ARMS[:3]
PARAMETERS = {"pretrained": 0, "action_residual_only": 116, "both_readouts": 232, "full_joint": 6112}
ROLES = ("plan", "producer_receipt", "producer_terminal", "audit_receipt", "audit_terminal", "audit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_relative_to(ROOT) and ".." not in path.parts and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), "contained regular evidence")
    return path


def descriptor(value):
    path = regular(value)
    require(path.suffix in {".json", ".py", ".md", ".png"}, "publication metadata or source only")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    path = regular(value)
    require(path.suffix == ".json", "saved JSON only")
    return json.loads(path.read_text())


def pinned(record):
    require(type(record) is dict and set(record) == {"path", "sha256", "bytes"}
            and descriptor(record["path"]) == record, "unchanged complete descriptor")
    return read(record["path"])


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "finite nonnegative saved metric or cost")
    return value


def gap(reports, arm, seed, regime, scope):
    return reports[arm, seed]["scopes"][scope]["by_regime"][regime]["case_weighted_raw_gap"]


def authenticate(path, sha):
    require(path.is_absolute() and descriptor(path)["sha256"] == sha, "external closure pin")
    closure = read(path)
    require(closure["version"] == "otto-readout-state-ablation-closure-v1"
            and closure["status"] in ("MECHANISM_DEV_PASS", "MECHANISM_DEV_FAIL"), "closed diagnostic study")
    records = {key: pinned(closure[key]) for key in ROLES}
    plan = records["plan"]
    require(plan["sources"][RUNNER] == RUNNER_SHA256 == descriptor(RUNNER)["sha256"],
            "original registered closure authenticator")
    spec = importlib.util.spec_from_file_location("_readout_report_original_runner", ROOT / RUNNER)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    for prefix, expected_phase in (("producer", "train"), ("audit", "audit")):
        checked_plan, checked_receipt, directory = runner.process_closure(
            closure["plan"]["path"], closure[prefix + "_receipt"]["path"], closure[prefix + "_terminal"]["path"])
        require(checked_plan == plan and checked_receipt == records[prefix + "_receipt"]
                and checked_receipt["phase"] == expected_phase, "same original bound phase")
        if prefix == "audit":
            require(regular(closure["audit"]["path"]) == directory / "audit.json"
                    and checked_receipt["files"]["audit.json"] == closure["audit"], "closed audit output")
    producer, auditor = records["producer_receipt"], records["audit_receipt"]
    require(auditor["producer_receipt"] == closure["producer_receipt"]
            and auditor["producer_terminal"] == closure["producer_terminal"]
            and records["producer_terminal"]["finished_ns"] <= records["audit_terminal"]["started_ns"],
            "audit follows and authenticates this original producer")
    audit = records["audit"]
    require(audit["version"] == "otto-readout-ablation-saved-audit-v1" and audit["agreement"] is True
            and audit["status"] == "completed_pending_original_audit_closure"
            and audit["technical_complete"] is False and audit["requires_original_supervisor_closure"] is True
            and audit["test_admitted"] is False and audit["confirmation_admitted"] is False,
            "saved audit is promoted only by the authenticated original closure")
    counts = {"array_decodes": 28, "checkpoint_decodes": 12, "views_completed": 12, "fits_checked": 9,
              "model_calls": 0, "optimizer_calls": 0, "teacher_calls": 0, "native_calls": 0,
              "test_array_decodes": 0, "journal_updates_checked": 3240, "episode_exposures_checked": 19440}
    require(audit["counts"] == auditor["audit_counts"] == counts, "complete independent audit accounting")
    reports = {(r["family"], r["seed"]): r for r in audit["reports"]}
    require(len(audit["reports"]) == len(reports) == 12
            and set(reports) == {(arm, seed) for arm in ARMS for seed in SEEDS}, "all twelve views")
    manifest = audit["reports"][0]["identity_manifest"]
    require(len(manifest) == 18 and all(type(row["length"]) is int and row["length"] > 0 for row in manifest),
            "complete fixed-path manifest")
    for report in reports.values():
        require(report["identity_manifest"] == manifest and report["stage"] == "dev"
                and report["query_period"] == 4 and report["episodes"] == 18, "matched reused DEV views")
        for scope in ("full", "initial", "later"):
            for regime in REGIMES:
                leaf = report["scopes"][scope]["by_regime"][regime]
                require(leaf["episodes"] == 9 and leaf["declared_case_count"] == 3, "matched case denominators")
                for metric in ("case_weighted_raw_gap", "case_weighted_agreement", "case_weighted_centered_mse"):
                    finite(leaf[metric])
        for regime in REGIMES:
            finite(report["prequery"]["by_regime"][regime]["case_weighted_centered_mse"])
    comparison = audit["comparisons"]
    cells = comparison["cells"]
    require(comparison["candidate"] == "full_joint" and comparison["margin"] == .05
            and comparison["held_out_evidence"] is False and comparison["statistical_equivalence"] is False
            and comparison["test_admitted"] is False and comparison["confirmation_admitted"] is False
            and len(cells) == comparison["total_cells"] == 6
            and {(row["seed"], row["regime"]) for row in cells} == {(s, r) for s in SEEDS for r in REGIMES},
            "all fixed paired comparison cells")
    for cell in cells:
        for scope in ("full", "later"):
            require(cell["gaps"][scope] == {arm: gap(reports, arm, cell["seed"], cell["regime"], scope)
                                          for arm in ARMS}, "saved comparison and report agreement")
        flags = []
        for key in ("later_gain_at_least_5pct", "full_gap_nonregression"):
            require(set(cell[key]) == set(CONTROLS) and all(type(v) is bool for v in cell[key].values()),
                    "all saved control checks")
            flags.extend(cell[key].values())
        require(type(cell["passed"]) is bool and cell["passed"] == all(flags), "saved complete cell outcome")
    passed = all(row["passed"] for row in cells)
    require(comparison["passed_cells"] == sum(row["passed"] for row in cells)
            and comparison["recurrent_weight_hypothesis_continues"] is passed
            and closure["status"] == ("MECHANISM_DEV_PASS" if passed else "MECHANISM_DEV_FAIL"),
            "preserved original diagnostic decision")
    costs = {}
    for key, arms in (("measured_fit_seconds", ARMS[1:]), ("measured_view_seconds", ARMS)):
        rows = audit[key]
        costs[key] = {(row["family"], row["seed"]): finite(row["seconds"]) for row in rows}
        require(len(rows) == len(costs[key]) == 3 * len(arms)
                and set(costs[key]) == {(arm, seed) for arm in arms for seed in SEEDS}, "all measured costs")
    for receipt in (producer, auditor):
        finite(receipt["wall_seconds"])
    require(not any(name in sys.modules for name in ("numpy", "torch", "tensorflow", "jax", "mlx")),
            "authentication imported no numerical runtime")
    return closure, records, reports, costs


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in
                     (headers, ["---"] * len(headers), *rows))


def fmt(value):
    return f"{value:.9g}"


def mean(values):
    return math.fsum(values) / len(values)


def link(path, output):
    if Path(path).name == "audit.json":
        return "[audit.json in the evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-readout-state-ablation-v1)"
    return "[" + Path(path).name + "](<" + os.path.relpath(path, output) + ">)"


def render(path, sha, output):
    require(output.is_absolute() and output.is_relative_to(ROOT) and ".." not in output.parts
            and output.parent.is_dir() and not output.exists() and not output.is_symlink()
            and not any(p.is_symlink() for p in output.parents), "fresh contained output directory")
    closure, records, reports, costs = authenticate(path, sha)
    renderer = descriptor(__file__)
    output.mkdir()
    # Only plotting imports numerical libraries; all evidence above is JSON or opaque bytes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    colors, markers, offsets = ("#1764a0", "#bf6613", "#358342"), ("o", "s", "^"), (-.14, 0., .14)

    def dots(ax, arms, labels, values):
        for index, seed in enumerate(SEEDS):
            ax.scatter([values(arm, seed) for arm in arms], [i + offsets[index] for i in range(len(arms))],
                       c=colors[index], marker=markers[index], s=45, label=str(seed), zorder=3)
        ax.scatter([mean([values(arm, seed) for seed in SEEDS]) for arm in arms], list(range(len(arms))),
                   c="#202020", marker="D", s=48, label="Equal fit mean", zorder=4)
        ax.set_yticks(range(len(arms)), labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="x", color="#dddddd", zorder=0)
        ax.spines[["right", "top"]].set_visible(False)

    for ax, regime in zip(axes[0], REGIMES, strict=True):
        dots(ax, ARMS, LABELS, lambda arm, seed, regime=regime: gap(reports, arm, seed, regime, "later"))
        ax.set_title(f"{regime}: later nonquery actions (step >= 5)", loc="left", weight="bold")
        ax.set_xlabel("Case-weighted teacher-score gap (lower is better)")
    axes[0, 0].legend(loc="best", fontsize=8, title="Fit seed; same 18 DEV paths")
    dots(axes[1, 0], ARMS[1:], LABELS[1:], lambda arm, seed: costs["measured_fit_seconds"][arm, seed])
    axes[1, 0].set_title("Measured continuation training cost", loc="left", weight="bold")
    axes[1, 0].set_xlabel("Seconds per fit: 40 epochs, 360 updates\nParent reused; original pretraining cost not measured here")
    comparison = records["audit"]["comparisons"]
    cells = sorted(comparison["cells"], key=lambda c: (c["seed"], c["regime"]))
    ax = axes[1, 1]
    flags = [[int(cell[key][arm]) for arm in CONTROLS
              for key in ("later_gain_at_least_5pct", "full_gap_nonregression")] for cell in cells]
    from matplotlib.colors import ListedColormap
    ax.imshow(flags, cmap=ListedColormap(("#f5d6d6", "#d6e9d8")), vmin=0, vmax=1, aspect="auto")
    for row, values in enumerate(flags):
        for column, value in enumerate(values):
            ax.text(column, row, "PASS" if value else "FAIL", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(6), ["Parent\nLater", "Parent\nFull", "Residual\nLater", "Residual\nFull", "Both\nLater", "Both\nFull"])
    ax.set_yticks(range(6), [f"{cell['seed']} / {cell['regime']}" for cell in cells])
    ax.tick_params(length=0)
    ax.set_title("Full joint vs every control: all saved checks", loc="left", weight="bold")
    ax.set_xlabel("Later: >=5% gap gain and strict improvement; full: no gap regression")
    fig.suptitle(f"Readout/state ablation: {closure['status']} ({comparison['passed_cells']}/6 paired cells)\n"
                 "Reused DEV paths; dots are three fits, not independent evaluation cohorts", fontsize=15, weight="bold")
    fig.savefig(output / "methods.png", dpi=160)
    plt.close(fig)

    manifest = next(iter(reports.values()))["identity_manifest"]
    text = ["# OTTO readout/state ablation", "",
            f"**{closure['status']}: {comparison['passed_cells']}/6 paired seed/setting cells passed.**", "",
            ("This compares continuing training of the action residual head (116 parameters), both readouts "
             "(232), or all GRU/readout parameters (6,112), against the same pretrained parents. All nine "
             "fits used 40 epochs, 360 updates, matched orders, and final checkpoints only."), "",
            (f"The 12 views use the same 18 previously exposed DEV paths ({sum(row['length'] for row in manifest):,} "
             "rows), P4 teacher queries, and three fit seeds. Lower teacher-score gap is better. "
             "These are OTTO fixed-path teacher-score imitation results, not autonomous performance, independent held-out "
             "evidence, statistical equivalence, or a novelty claim."), "",
            "![All arms, fit seeds, training costs, and comparison checks](methods.png)", "",
            ("Dots show each fit; black diamonds are equal means across the three fits. No confidence "
             "interval is inferred from three fits on shared paths. The pretrained parent has no "
             "continuation training cost; its original pretraining cost is excluded."), "",
            "## All twelve views", "",
            ("Values below are the saved case-weighted raw gaps. Full covers all nonquery actions; "
             "later covers nonquery steps >=5, after the first post-start teacher query."), "",
            table(["Arm", "Fit seed", "lambda3 full", "lambda3 later", "lambda4 full", "lambda4 later", "View seconds"],
                  [[arm, seed, *(fmt(gap(reports, arm, seed, regime, scope)) for regime in REGIMES
                                 for scope in ("full", "later")), fmt(costs["measured_view_seconds"][arm, seed])]
                   for arm in ARMS for seed in SEEDS]), "",
            "## Complete paired decision", "",
            ("The fixed full-joint candidate must improve later gap by at least 5% against each control "
             "with strict improvement, and must not regress full gap. Every seed/setting cell must pass. "
             "A zero control gap cannot pass the strict-improvement check. No failed cell is excluded."), "",
            table(["Fit seed", "Setting", "Later supported cases", "vs parent later/full", "vs residual later/full",
                   "vs both later/full", "Cell"],
                  [[cell["seed"], cell["regime"], f"{cell['supported_later_cases']}/3",
                    *(" / ".join("PASS" if cell[key][arm] else "FAIL"
                                 for key in ("later_gain_at_least_5pct", "full_gap_nonregression")) for arm in CONTROLS),
                    "PASS" if cell["passed"] else "FAIL"] for cell in cells]), "",
            "## Readout proximity, descriptive only", "",
            ("These saved comparisons ask whether a readout-only later gap is at most 5% above full joint. "
             "They do not establish equivalence or select a replacement candidate. Relative excess is "
             "undefined when the full-joint gap is zero."), "",
            table(["Fit seed", "Setting", "Readout", "Relative excess", "No more than 5% worse"],
                  [[cell["seed"], cell["regime"], arm,
                    "undefined (zero full-joint gap)" if value["relative_excess"] is None else fmt(value["relative_excess"]),
                    "yes" if value["no_more_than_5pct_worse"] else "no"]
                   for cell in cells for arm, value in cell["readout_proximity_descriptive_only"].items()]), "",
            "## Measured cost", "",
            table(["Arm", "Trainable parameters", "Fit seed", "Continuation seconds"],
                  [[arm, PARAMETERS[arm], seed, fmt(costs["measured_fit_seconds"][arm, seed])]
                   for arm in ARMS[1:] for seed in SEEDS]), "",
            (f"Saved worker wall times: producer {fmt(records['producer_receipt']['wall_seconds'])} s; "
             f"independent audit {fmt(records['audit_receipt']['wall_seconds'])} s. "
             "Per-fit and per-view timers cover their recorded worker regions; these are not a matched "
             "inference-speed benchmark or end-to-end compute estimates."), "",
            "## Evidence and scope", "",
            "Original closure: " + link(path, output) + ". External SHA-256: `" + sha + "`.", "",
            *["- " + role.replace("_", " ") + ": " + link(closure[role]["path"], output)
              + " (`" + closure[role]["sha256"] + "`)" for role in ROLES], "",
            ("The original producer and independent auditor both completed under their original supervisors. "
             "The audit checked all nine fits, 3,240 updates, 19,440 episode exposures, and twelve views. "
             "Publication reads saved JSON and verifies opaque payload hashes; it performs no array/checkpoint "
             "decoding, model calls, training, teacher calls, or simulator calls. TEST and confirmation remain unadmitted."), ""]
    (output / "README.md").write_text("\n".join(text))
    for role in ROLES:
        require(descriptor(closure[role]["path"]) == closure[role], "publication input unchanged")
    require(descriptor(path)["sha256"] == sha and descriptor(__file__) == renderer, "publication closure and source unchanged")
    receipt = {"version": VERSION, "status": "completed", "scientific_status": closure["status"],
               "closure": descriptor(path), "inputs": {role: closure[role] for role in ROLES}, "renderer": renderer,
               "original_closures_verified": ["train", "audit"], "views_displayed": 12, "fit_costs_displayed": 9,
               "paired_cells_displayed": 6, "individual_checks_displayed": 36,
               "publication_array_decodes": 0, "publication_checkpoint_decodes": 0, "publication_model_calls": 0,
               "publication_teacher_calls": 0, "publication_simulator_calls": 0, "publication_training_updates": 0,
               "files": {name: descriptor(output / name) for name in ("README.md", "methods.png")}}
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return descriptor(output / "receipt.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.closure, args.closure_sha256, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
