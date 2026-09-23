"""Render an explicitly supplied, already authenticated saved-audit dictionary.

This presentation helper does not authenticate files or original supervisors.
The caller must establish those closures before supplying an empirical audit.
No evidence file, model, checkpoint or environment is opened here. Synthetic
qualification artifacts are visibly labelled and cannot count as results.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

FAMILIES = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta",
            "trace_additive", "trace_scrambled", "trace_no_write")
SEEDS = (309000001, 309000002, 309000003)
REGIMES = ("lambda3", "lambda4")
CONTROLS = ("pretrained", "last_error", "instant_delta", "joint_aux")
CANDIDATE = "trace_delta"
SCOPES = ("later", "full")
LIMITATION = ("Teacher-score gaps on fixed collector paths, with equal originating-case and fit-seed weighting. "
              "Lower is better. This does not establish autonomous performance, calibration, total-compute savings or architecture novelty.")
SYNTHETIC_LABEL = "SYNTHETIC FIXTURE - NOT EXPERIMENTAL RESULTS"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    """Reject NaN/infinity anywhere in a supplied JSON-like result."""
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        require(math.isfinite(value), "finite saved-audit numbers")
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, dict):
        require(all(type(key) is str for key in value), "JSON string keys")
        for item in value.values():
            finite_tree(item)
    else:
        raise ValueError("saved audit must contain only JSON values")


def validate_report(audit, *, stage, synthetic=False):
    """Validate presentation coverage, without claiming evidence authentication."""
    require(stage in ("dev", "test") and type(synthetic) is bool, "explicit DEV/TEST and synthetic status")
    require(isinstance(audit, dict), "explicit saved-audit dictionary")
    finite_tree(audit)
    require(audit["stage"] == stage and audit["agreement"] is True, "matching completed independent audit")
    periods, cases = ((4,), 3) if stage == "dev" else ((4, 8), 6)
    wanted = [(period, regime, scope) for period in periods for regime in REGIMES for scope in SCOPES]
    reports = audit["metrics"]
    require(isinstance(reports, list) and len(reports) == len(FAMILIES) * len(SEEDS) * len(periods),
            "complete24/48 saved metric roster")
    by = {}
    for report in reports:
        require(report["stage"] == stage and report["episodes"] == 6 * cases, "complete reported split")
        key = report["family"], report["seed"], report["query_period"]
        require(report["family"] in FAMILIES and type(report["seed"]) is int and report["seed"] in SEEDS
                and type(report["query_period"]) is int and report["query_period"] in periods
                and key not in by, "distinct declared family/seed/period")
        by[key] = report
    require(set(by) == {(family, seed, period) for family in FAMILIES for seed in SEEDS for period in periods},
            "no missing family/seed/period")
    panels = []
    for period, regime, scope in wanted:
        rows, supports = [], set()
        for family in FAMILIES:
            values = []
            for seed in SEEDS:
                leaf = by[family, seed, period]["scopes"][scope]["by_regime"][regime]
                value = leaf["case_weighted_raw_gap"]
                support = leaf["supported_case_count"]
                require(leaf["episodes"] == 3 * cases and leaf["declared_case_count"] == cases
                        and type(support) is int and 0 <= support <= cases, "complete originating-case denominators")
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        "finite nonnegative raw teacher-score gap")
                values.append(float(value))
                supports.add(support)
            rows.append({"family": family, "seed_gaps": values, "mean_gap": math.fsum(values) / len(SEEDS)})
        require(len(supports) == 1, "matched support across all methods and fit seeds")
        panels.append({"query_period": period, "regime": regime, "scope": scope,
                       "supported_cases": supports.pop(), "declared_cases": cases, "rows": rows})
    gate = audit["gate"]
    names = ["technical_completion"]
    for period in periods:
        for regime in REGIMES:
            prefix = f"{regime}:P{period}:"
            names.extend([prefix + "supported_cases", prefix + "later_gap_10pct", prefix + "full_gap_nonregression"])
            names.extend(prefix + f"seed_{seed}_nonregression" for seed in SEEDS)
    conditions = gate["conditions"]
    require(gate["stage"] == stage and gate["candidate"] == CANDIDATE
            and type(gate["technical_complete"]) is bool and type(gate["passed"]) is bool
            and isinstance(conditions, list) and [row["name"] for row in conditions] == names
            and all(type(row["passed"]) is bool for row in conditions)
            and conditions[0]["passed"] == gate["technical_complete"]
            and type(gate["total_conditions"]) is int and gate["total_conditions"] == len(names)
            and type(gate["passed_conditions"]) is int and gate["passed_conditions"] == sum(row["passed"] for row in conditions)
            and gate["passed"] == all(row["passed"] for row in conditions), "exact coherent13/25-condition saved gate")
    # This is display validation. The original independent audit remains the
    # authority for condition arithmetic and underlying numerical evidence.
    return {"stage": stage, "synthetic": synthetic, "periods": list(periods), "seeds": list(SEEDS),
            "families": list(FAMILIES), "panels": panels, "gate": json.loads(json.dumps(gate)),
            "record_count": len(reports), "scope": LIMITATION}


def number(value):
    return format(value, ".17g")


def markdown_report(data):
    stage, gate = data["stage"].upper(), data["gate"]
    status = "PASS" if gate["passed"] else "FAIL"
    title = f"OpenJev query memory: {stage} {status}"
    lines = ["# " + title, ""]
    if data["synthetic"]:
        lines.extend(["**" + SYNTHETIC_LABEL + "**", ""])
    lines.extend([f"Candidate: `{CANDIDATE}`. **{gate['passed_conditions']}/{gate['total_conditions']} conditions passed.**",
        "", LIMITATION, "", "All eight methods and all three fit seeds are shown. The candidate is fixed; an alternative method is not promoted after seeing results.",
        "", f"![Every method and fit seed on all {stage} panels](query-memory-{data['stage']}.png)", ""])
    for panel in data["panels"]:
        lines.extend([f"## {panel['regime']} / P{panel['query_period']} / {panel['scope']} nonqueries", "",
            f"Supported originating cases: {panel['supported_cases']}/{panel['declared_cases']}. Every collector path remains in the denominator.", "",
            "| Method | Mean gap | " + " | ".join(str(seed) for seed in SEEDS) + " |",
            "| --- | ---: | ---: | ---: | ---: |"])
        for row in panel["rows"]:
            label = f"`{row['family']}`" + (" (candidate)" if row["family"] == CANDIDATE else "")
            lines.append("| " + label + " | " + " | ".join(number(v) for v in [row["mean_gap"], *row["seed_gaps"]]) + " |")
        lines.append("")
    lines.extend(["## Every continuation condition", "", "| Condition | Result | Saved details |", "| --- | --- | --- |"])
    for condition in gate["conditions"]:
        details = {key: value for key, value in condition.items() if key not in ("name", "passed")}
        text = json.dumps(details, sort_keys=True, separators=(", ", ": "), allow_nan=False).replace("|", "\\|")
        lines.append(f"| `{condition['name']}` | {'PASS' if condition['passed'] else '**FAIL**'} | `{text}` |")
    failed = [row["name"] for row in gate["conditions"] if not row["passed"]]
    lines.extend(["", "Failed conditions: " + (", ".join(f"`{name}`" for name in failed) if failed else "none") + ".", "",
        "The caller authenticates the source audit and original process closures separately. This renderer makes no new admission decision.", ""])
    return "\n".join(lines)


def draw_figure(path, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    row_count = len(data["panels"]) // 2
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False}):
        fig, axes = plt.subplots(row_count, 2, figsize=(15, 4.1 * row_count + 1.5), squeeze=False)
        for panel_index, panel in enumerate(data["panels"]):
            axis = axes[panel_index // 2][panel_index % 2]
            colors = ["#ba522b" if family == CANDIDATE else "#467dab" if family in CONTROLS else "#abb3bd" for family in FAMILIES]
            axis.barh(range(8), [row["mean_gap"] for row in panel["rows"]], height=.65, color=colors, alpha=.85)
            maximum = max(value for row in panel["rows"] for value in row["seed_gaps"])
            for seed_index, marker in enumerate(("o", "s", "^")):
                axis.scatter([row["seed_gaps"][seed_index] for row in panel["rows"]],
                    [index + (seed_index - 1) * .14 for index in range(8)], marker=marker,
                    s=23, facecolors="white", edgecolors="#172331", linewidths=.8, zorder=3)
            axis.set_yticks(range(8), [family + (" *" if family == CANDIDATE else "") for family in FAMILIES])
            axis.invert_yaxis()
            axis.set_xlim(0, maximum * 1.13 if maximum else 1)
            axis.set_xlabel("Raw teacher-score gap (lower is better)")
            axis.set_title(f"{panel['regime']} / P{panel['query_period']} / {panel['scope']} nonqueries\n"
                           f"{panel['supported_cases']}/{panel['declared_cases']} originating cases supported", loc="left")
            axis.set_axisbelow(True)
            axis.grid(axis="x", color="#e8eaed", linewidth=.6)
        gate = data["gate"]
        status = "PASS" if gate["passed"] else "FAIL"
        title = f"OpenJev query memory: {data['stage'].upper()} {status} ({gate['passed_conditions']}/{gate['total_conditions']} conditions)"
        if data["synthetic"]:
            title = SYNTHETIC_LABEL + "\n" + title
        fig.suptitle(title, fontsize=15, fontweight="bold", y=.99)
        handles = [Line2D([0], [0], color="#ba522b", lw=7, label="trace_delta: fixed candidate")]
        handles += [Line2D([0], [0], marker=marker, color="none", markeredgecolor="#172331",
                          markerfacecolor="white", label=str(seed)) for marker, seed in zip(("o", "s", "^"), SEEDS, strict=True)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
        fig.text(.5, .014, "Bars: mean of three fit seeds. Fixed paths only; no autonomous, calibration or novelty claim.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, .92))
        try:
            with path.open("xb") as stream:
                fig.savefig(stream, format="png", dpi=160, facecolor="white",
                            metadata={"Software": "OpenJev saved-audit renderer", "Description": SYNTHETIC_LABEL if data["synthetic"] else LIMITATION})
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            plt.close(fig)
    return matplotlib.__version__


def render(audit, *, stage, output, synthetic=False):
    """Write an exclusive Markdown/PNG artifact pair from caller-owned evidence."""
    data = validate_report(audit, stage=stage, synthetic=synthetic)
    output = Path(output)
    require(not output.exists() and not any(p.is_symlink() for p in output.parents), "exclusive nonsymlink rendering output")
    output.mkdir(parents=False, exist_ok=False)
    markdown = output / f"query-memory-{stage}.md"
    png = output / f"query-memory-{stage}.png"
    with markdown.open("x") as stream:
        stream.write(markdown_report(data))
        stream.flush()
        os.fsync(stream.fileno())
    version = draw_figure(png, data)
    return {"stage": stage, "synthetic": synthetic, "record_count": data["record_count"],
        "panels": [{key: panel[key] for key in ("query_period", "regime", "scope")} for panel in data["panels"]],
        "matplotlib_version": version, "files": {path.name: {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (markdown, png)}}
