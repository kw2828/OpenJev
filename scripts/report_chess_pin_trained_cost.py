"""Render authenticated saved chess timing artifacts without neural imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = "evidence/chess-pin-trained-cost-v2/protocol/plan.json"
PLAN_SHA = "46bd0965241aa8fa8e0a54c5f19a925dfd6f5102a93b97eefc8abe4832abe464"
QUALITY_PLAN_SHA = "b0b3da433d1f507d012208624c4231ee777febe7d6843685f9dacd169c691582"
EXECUTION = "runs/chess-pin-trained-cost-v2/execution"
AUDIT = "evidence/chess-pin-trained-cost-v2/audit"
LAUNCHER = "runs/chess-pin-trained-cost-v2/launcher"
METHODS = ("base", "wldn", "joint", "separable", "pairwise", "root_only", "counts", "graph_mlp", "union:edits")
SEEDS = (97, 109, 127)
LABELS = ("Base", "WLDN", "Joint", "Separable", "Pairwise", "Root only", "Counts", "Graph MLP", "Union: edits")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def read(path):
    return decode(Path(path).read_bytes())


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def artifact(root, name, expected, inputs):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact path")
    path = root / relative
    require(path.is_file() and path.resolve() == path, "Missing or nonregular artifact: " + name)
    digest = sha(path)
    require(expected is None or digest == expected, "Artifact hash mismatch: " + name)
    inputs[name] = digest
    return path


def manifest(root, folder, expected_plan, inputs):
    require(not (root / folder / "failed.json").exists(), "Conflicting failed execution")
    completed = artifact(root, folder + "/completed.json", None, inputs)
    value = read(completed)
    require(value["status"] == "completed" and value["plan_sha256"] == expected_plan, "Completion identity")
    actual = set()
    for path in (root / folder).rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Nonregular execution member")
        if path.is_file() and path != completed:
            actual.add(path.relative_to(root / folder).as_posix())
    require(actual == set(value["files"]), "Exact execution member set differs")
    for name, digest in value["files"].items():
        artifact(root, folder + "/" + name, digest, inputs)
    return value


def source_bindings(value):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith(".py") and isinstance(item, str) and re.fullmatch("[0-9a-f]{64}", item):
                result[key] = item
            for name, digest in source_bindings(item).items():
                require(name not in result or result[name] == digest, "Conflicting source binding")
                result[name] = digest
    elif isinstance(value, list):
        for item in value:
            for name, digest in source_bindings(item).items():
                require(name not in result or result[name] == digest, "Conflicting source binding")
                result[name] = digest
    return result


def keys(kind):
    for si, seed in enumerate(SEEDS):
        if kind == "warmups":
            yield from ((seed, 0, repeat, method) for method in METHODS for repeat in range(2))
        elif kind == "native":
            yield from ((seed, index, 0, method) for index in range(128) for method in METHODS)
        else:
            for index in range(128):
                for repeat in range(9):
                    rotation = (si + index + repeat) % 9
                    for method in METHODS[rotation:] + METHODS[:rotation]:
                        yield seed, index, repeat, method


def records(path, kind, panel):
    rows = [decode(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    require([(r["seed"], r["root_index"], r["repeat"], r["method"]) for r in rows] == list(keys(kind)),
            "Complete ordered " + kind + " coverage required")
    for row in rows:
        require(row["id"] == panel[row["root_index"]]["id"], "Root identity differs")
        comparison = row["comparison"]
        require(finite(comparison["max_score_error"])
                and type(comparison["score_tolerance_passed"]) is bool
                and comparison["score_tolerance_passed"] == (comparison["max_score_error"] <= 1e-5)
                and type(comparison["choice_matches"]) is bool, "Invalid saved comparison")
        prediction = row["prediction"]
        menu, scores = prediction["menus"], prediction["scores"]
        require(menu and menu == sorted(set(menu)) and len(menu) == len(scores)
                and all(type(x) in (int, float) and math.isfinite(x) for x in scores), "Invalid score vector")
        require(prediction["choice"] == menu[max(range(len(scores)), key=scores.__getitem__)], "Argmax differs")
        if kind == "timings":
            require(finite(row["milliseconds"], positive=True), "Positive finite timing required")
    return rows


def comparisons(rows):
    return {"records": len(rows),
            "failed_records": sum(not (r["comparison"]["score_tolerance_passed"] and r["comparison"]["choice_matches"])
                                  for r in rows),
            "choice_changes": sum(not r["comparison"]["choice_matches"] for r in rows),
            "max_score_error": max(r["comparison"]["max_score_error"] for r in rows)}


def aggregate(rows):
    if not rows:
        return None
    timing = {(r["seed"], r["root_index"], r["repeat"], r["method"]): r["milliseconds"] for r in rows}
    pairs = sorted({key[:3] for key in timing})
    present_seeds = [seed for seed in SEEDS if any(row["seed"] == seed for row in rows)]
    return {"timed_records": len(rows),
            "median_complete_ms": {m: statistics.median(r["milliseconds"] for r in rows if r["method"] == m) for m in METHODS},
            "per_seed_median_ms": {str(s): {m: statistics.median(r["milliseconds"] for r in rows
                                                              if r["method"] == m and r["seed"] == s)
                                           for m in METHODS} for s in present_seeds},
            "median_paired_ratio_to_wldn": {m: statistics.median(timing[s, i, r, m] / timing[s, i, r, "wldn"]
                                                                for s, i, r in pairs) for m in METHODS}}


def authenticate(root, expected_audit_sha256):
    """External audit identity is checked before any result payload is opened."""
    root = Path(root).resolve()
    require(isinstance(expected_audit_sha256, str) and re.fullmatch("[0-9a-f]{64}", expected_audit_sha256),
            "External completed-audit SHA256 required")
    inputs = {}
    receipt_path = artifact(root, AUDIT + "/receipt.json", expected_audit_sha256, inputs)
    audit = read(receipt_path)
    require(audit["status"] == "completed" and audit["plan_sha256"] == PLAN_SHA, "Completed cost audit required")
    require(not (root / AUDIT / "failed.json").exists(), "Conflicting failed audit")
    terminal = read(artifact(root, LAUNCHER + "/terminal.json", None, inputs))
    require(terminal["status"] == "completed" and terminal["retry"] is False
            and terminal["plan_sha256"] == PLAN_SHA and terminal["audit_receipt_sha256"] == expected_audit_sha256,
            "Completed cost lifecycle required")
    for phase in ("run", "audit"):
        stamp = read(artifact(root, LAUNCHER + f"/{phase}-terminal.json", None, inputs))
        require(stamp["status"] == "exited" and stamp["returncode"] == 0 and stamp["error"] is None
                and finite(stamp["wall_seconds"], positive=True) and stamp["wall_seconds"] <= 1800,
                "Completed bounded cost phase required")
    plan = read(artifact(root, PLAN, PLAN_SHA, inputs))
    p = plan["protocol"]
    require(p["version"] == "trained-pin-complete-native-cost-v2" and p["methods"] == list(METHODS)
            and p["seeds"] == list(SEEDS) and p["roots"] == 128 and p["repeats"] == 9
            and p["timed_records"] == 31104 and p["native_audit_decisions"] == 3456
            and p["warmups_per_method_seed"] == 2 and p["score_tolerance"] == 1e-5
            and p["time_cap_seconds"] == p["audit_time_cap_seconds"] == 1800, "Frozen cost scope differs")
    for name, digest in source_bindings(plan).items():
        artifact(root, name, digest, inputs)
    require(audit["auditor_sha256"] == inputs["scripts/chess_pin_trained_cost.py"], "Cost auditor source differs")
    completion = manifest(root, EXECUTION, PLAN_SHA, inputs)
    require(set(completion["files"]) == {"started.json", "timings.jsonl", "warmups.jsonl", "summary.json"},
            "Cost primary membership differs")
    require(inputs[EXECUTION + "/completed.json"] == audit["execution_receipt_sha256"], "Cost completion binding")
    summary = read(artifact(root, EXECUTION + "/summary.json", audit["summary_sha256"], inputs))
    artifact(root, AUDIT + "/native-replay.jsonl", audit["native_replay_sha256"], inputs)
    require(summary["status"] == "completed" and summary["plan_sha256"] == PLAN_SHA, "Cost summary identity")
    require(all(finite(x["wall_seconds"], positive=True) and x["wall_seconds"] <= 1800 for x in (summary, audit)),
            "Cost phase budget exceeded")
    require(all(x[k] == 0 for x in (summary, audit) for k in
                ("new_training_updates", "new_engine_calls", "external_model_calls")), "Cost scope changed")
    require(summary["quality_evidence"] == audit["quality_evidence"] and summary["limits"] == audit["limits"] == p["limits"],
            "Quality evidence or limits differ")
    require(summary["pin_input_coverage"] == audit["pin_input_coverage"] == plan["pin_input_coverage"],
            "Pin-input coverage differs")
    binding = plan["quality_bindings"]
    quality_plan = read(artifact(root, binding["plan"], QUALITY_PLAN_SHA, inputs))
    require(plan["quality_plan_sha256"] == QUALITY_PLAN_SHA and plan["quality_signature"]
            == {k: v for k, v in quality_plan.items() if k != "prepared_unix"}, "Quality plan lineage differs")
    evidence = summary["quality_evidence"]
    require(set(evidence["hashes"]) == {binding["audit"], binding["execution"] + "/completed.json",
                                       binding["execution"] + "/summary.json"}, "Quality binding membership")
    for name, digest in evidence["hashes"].items():
        artifact(root, name, digest, inputs)
    manifest(root, binding["execution"], QUALITY_PLAN_SHA, inputs)
    quality_audit, quality = read(root / binding["audit"]), read(root / binding["execution"] / "summary.json")
    require(quality_audit["status"] == quality["status"] == "completed"
            and quality_audit["plan_sha256"] == quality["plan_sha256"] == QUALITY_PLAN_SHA
            and quality_audit["execution_receipt_sha256"] == inputs[binding["execution"] + "/completed.json"]
            and quality_audit["summary_sha256"] == inputs[binding["execution"] + "/summary.json"], "Quality audit identity")
    require(quality_audit["gate_recomputed"] == quality["gate_checks"] and len(quality["gate_checks"]) == 16
            and all(type(row["passed"]) is bool for row in quality["gate_checks"]), "Quality gate binding")
    for field in ("quality_gate_passed", "numerical_gate_passed", "continuation_passed"):
        evidence_field = "quality_" + field if field != "quality_gate_passed" else field
        require(type(quality[field]) is bool and evidence[evidence_field] == quality[field], "Quality gate changed")
    require(quality["quality_gate_passed"] == all(row["passed"] for row in quality["gate_checks"]), "Quality gate arithmetic")
    rows = records(root / EXECUTION / "timings.jsonl", "timings", plan["panel"])
    warm = records(root / EXECUTION / "warmups.jsonl", "warmups", plan["panel"])
    replay = records(root / AUDIT / "native-replay.jsonl", "native", plan["panel"])
    metrics = aggregate(rows)
    require(metrics == summary["timings"], "Saved timing aggregate differs")
    for name, recorded, expected in (("timed", summary["timed_checks"], comparisons(rows)),
                                     ("warmup", summary["warmup_checks"], comparisons(warm)),
                                     ("native", audit["native_audit"], comparisons(replay))):
        require(recorded == expected, name + " comparison counts differ")
    require(audit["timing_records_checked"] == 31104 and audit["warmup_records_checked"] == 54,
            "Audit coverage differs")
    strata = {}
    for field in ("root_pin_present", "pin_change_present"):
        for present in (False, True):
            indices = {row["index"] for row in plan["pin_input_coverage"]["roots"] if row[field] == present}
            strata[f"{field}={str(present).lower()}"] = aggregate([row for row in rows if row["root_index"] in indices])
    require(strata == summary["strata"], "Saved timing strata differ")
    require(summary["cost_path_numerical_gate_passed"] == (comparisons(rows)["failed_records"] == comparisons(warm)["failed_records"] == 0)
            and audit["cost_path_numerical_gate_passed"] == (summary["cost_path_numerical_gate_passed"]
                                                            and comparisons(replay)["failed_records"] == 0), "Cost gate differs")
    per_seed_ratios = {str(seed): aggregate([r for r in rows if r["seed"] == seed])["median_paired_ratio_to_wldn"]["joint"]
                       for seed in SEEDS}
    return {"timings": metrics, "strata": strata, "joint_wldn_per_seed_paired_median_ratio": per_seed_ratios,
            "quality_checks_passed": sum(row["passed"] for row in quality["gate_checks"]), "quality_checks": 16,
            "quality_gate_passed": quality["quality_gate_passed"], "quality_numerical_gate_passed": quality["numerical_gate_passed"],
            "quality_continuation_passed": quality["continuation_passed"],
            "cost_path_numerical_gate_passed": audit["cost_path_numerical_gate_passed"],
            "coverage": {"methods": 9, "seeds": 3, "roots": 128, "repeats": 9, "timings": 31104, "warmups": 54, "native_audit": 3456},
            "timed_checks": summary["timed_checks"], "warmup_checks": summary["warmup_checks"], "native_audit": audit["native_audit"],
            "primary_wall_seconds": summary["wall_seconds"], "audit_wall_seconds": audit["wall_seconds"],
            "timing_definition": p["timing"], "limits": p["limits"], "inputs_sha256": inputs,
            "model_calls": 0, "engine_calls": 0, "scope": "Saved-byte authentication and arithmetic, not fresh inference or independent neural replay."}


def figure(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "pin-cost-v2"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1.6, 1]})
    median = result["timings"]["median_complete_ms"]
    axes[0].barh(range(9), [median[m] for m in METHODS], color=["#137d76" if m == "joint" else "#cbd5e1" for m in METHODS])
    colors = ("#2b4c7e", "#be6b21", "#733b8f")
    for seed, color in zip(SEEDS, colors, strict=True):
        axes[0].scatter([result["timings"]["per_seed_median_ms"][str(seed)][m] for m in METHODS], range(9),
                        color=color, s=24, zorder=3, label=f"Seed {seed}")
    axes[0].set_yticks(range(9), LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median complete decision time (ms)")
    axes[0].set_title("All nine policies; dots show per-seed medians", loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="upper right")
    ratios = [result["joint_wldn_per_seed_paired_median_ratio"][str(seed)] for seed in SEEDS]
    axes[1].scatter(range(3), ratios, color=colors, s=55, zorder=3)
    axes[1].axhline(1, color="#64748b", linestyle="--", label="Equal time")
    overall = result["timings"]["median_paired_ratio_to_wldn"]["joint"]
    axes[1].axhline(overall, color="#137d76", label=f"Pooled median: {overall:.3f}x")
    axes[1].set_xticks(range(3), [str(seed) for seed in SEEDS])
    axes[1].set_xlim(-.5, 2.5)
    axes[1].set_xlabel("Paired backbone seed")
    axes[1].set_ylabel("Joint / WLDN decision time")
    axes[1].set_title("Ratio of matched decisions, then median", loc="left", fontsize=11)
    axes[1].legend(frameon=False, loc="best")
    for axis in axes:
        axis.grid(axis="x" if axis is axes[0] else "y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("Trained chess policies: native decision cost", x=.08, ha="left", fontweight="bold", fontsize=16)
    gate = "passed" if result["quality_gate_passed"] else "failed"
    fig.text(.08, .07, f"Quality criterion remains {gate}: {result['quality_checks_passed']}/16 checks. "
             "Seed spread is descriptive, not a confidence interval.", fontsize=10)
    fig.text(.08, .035, "31,104 timings | 128 reused roots | 3 fitted seeds | 9 repeats | CPU, 2 threads, shared host. No gameplay or novelty claim.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.10, right=.97, bottom=.22, top=.86, wspace=.40)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / ("native-cost." + extension), dpi=180)
    plt.close(fig)


def render(root, expected_audit_sha256, out):
    result = authenticate(root, expected_audit_sha256)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        figure(result, out)
        lines = ["# Trained pin-policy decision cost", "", "Descriptive timing on the reused ordinary panel.", "",
                 (f"The quality criterion remains {'passed' if result['quality_gate_passed'] else 'failed'} "
                  f"({result['quality_checks_passed']}/16 checks). Timing does not repair this criterion."), "",
                 "| Method | Median ms | Seed 97 | Seed 109 | Seed 127 | Paired ratio to WLDN |",
                 "|---|---:|---:|---:|---:|---:|"]
        for method, label in zip(METHODS, LABELS, strict=True):
            values = [result["timings"]["median_complete_ms"][method]] + [result["timings"]["per_seed_median_ms"][str(s)][method] for s in SEEDS]
            lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values)
                         + f" | {result['timings']['median_paired_ratio_to_wldn'][method]:.4f} |")
        lines += ["", "Ratios are medians of matched seed/root/repeat timing ratios, not ratios of aggregate medians.",
                  "Per-seed spread is descriptive and is not a confidence interval.", "", result["timing_definition"], "",
                  "Coverage: all 9 methods, 3 seeds, 128 roots, 9 repeats; 31,104 timings, 54 warmups and 3,456 audited native decisions.", "",
                  (f"Cost-path numerical criterion: {'passed' if result['cost_path_numerical_gate_passed'] else 'failed'}. "
                   f"Quality numerical criterion: {'passed' if result['quality_numerical_gate_passed'] else 'failed'}. "
                   f"Quality continuation: {'passed' if result['quality_continuation_passed'] else 'failed'}."), "",
                  "No Elo, gameplay strength or architecture novelty is established. These are scalar shared-host timings, not throughput or a hardware-general speed result.", "",
                  result["scope"], "", "![Complete native decision timings](native-cost.png)", ""]
        (out / "report.md").write_text("\n".join(lines))
        (out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        require(all(sha(Path(root).resolve() / name) == digest for name, digest in result["inputs_sha256"].items()),
                "Bound input changed during rendering")
        receipt = {"status": "completed", "external_audit_sha256": expected_audit_sha256,
                   "script_sha256": sha(__file__), "inputs_sha256": result["inputs_sha256"],
                   "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                   "model_calls": 0, "engine_calls": 0, "quality_gate_changed": False}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except BaseException as error:
        (out / "failed.json").write_text(json.dumps({"status": "failed", "error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.expected_audit_sha256, args.out)
