"""Saved-only dialogue copy report using explicitly reused, frozen metric helpers.

No encoder/model/checkpoint deserialization or neural replay. Lexical extraction,
reference generation and learned computations remain source/test bound.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import platform
import re
import time
import types
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = "scripts/report_dialogue_memory.py"
BASE_SHA256 = "bcaf33e75c87705b6bc11866e1ccf6ae558a624f00d586b5e86e3808f7a736c7"
_base_bytes = (ROOT / BASE_PATH).read_bytes()
if hashlib.sha256(_base_bytes).hexdigest() != BASE_SHA256:
    raise ValueError("Frozen metric helper source changed")
base = types.ModuleType("dialogue_copy_frozen_metrics")
base.__file__ = str(ROOT / BASE_PATH)
exec(compile(_base_bytes, base.__file__, "exec"), base.__dict__)  # noqa: S102 - execute only exact hash-pinned local bytes

METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
SEEDS = (4101, 4102, 4103)
PANELS = base.PANELS
STRATA = base.STRATA
PRIMARY = "selective"
REQUIRED_SOURCES = {
    "scripts/study_dialogue_copy.py", "scripts/prepare_dialogue_copy.py", "scripts/report_dialogue_copy.py",
    "src/openjev/research/dialogue_copy_features.py", "src/openjev/research/dialogue_copy_memory.py",
    "tests/test_dialogue_copy_features.py", "tests/test_dialogue_copy_memory.py", "tests/test_study_dialogue_copy.py",
    "tests/test_report_dialogue_copy.py", "research/dialogue-copy-protocol.md", "scripts/study_dialogue_memory.py",
    BASE_PATH, "src/openjev/research/dialogue_carry.py",
}
PRACTICAL_CHECKS = {
    "panels": ["seen", "unseen"], "primary": PRIMARY,
    "macro_gain_over_literal": .03,
    "macro_gain_over_scalar": .005, "micro_nll_no_worse_than": "scalar",
    "strict_paired_macro_wins_over_scalar": 2,
    "conventional_controls": ["readout", "candidate_gru"],
    "maximum_revision_deficit_to_literal": .01,
    "required_complete_fits": 15, "total_panel_checks": 12,
}


def family_metrics(rows):
    families = {}
    for method in METHODS:
        family = {}
        for panel in PANELS:
            paired = [rows[f"{method}-{seed}"]["metrics"][panel] for seed in SEEDS]

            def pool(items):
                base.require(len({item["count"] for item in items}) == 1, "Unequal fit support")
                return {"count_per_fit": items[0]["count"], "fit_count": len(SEEDS),
                        **{key: base.mean_defined([item[key] for item in items])
                           for key in ("accuracy", "nll", "brier")},
                        "zero_target_probabilities_all_fits": sum(item["zero_target_probabilities"] for item in items)}

            family[panel] = {"micro": pool([item["micro"] for item in paired]),
                             "macro_three": {key: base.mean_defined([item["macro_three"][key] for item in paired])
                                             for key in ("accuracy", "nll", "brier")},
                             "bins": {name: pool([item["bins"][name] for item in paired]) for name in base.BINS},
                             "revision": pool([item["revision"] for item in paired])}
        families[method] = family
    return families


def criteria(rows, references):
    """Integer-ratio accuracy comparisons, inclusive margins, no epsilon."""
    identities = {f"{method}-{seed}" for method in METHODS for seed in SEEDS}
    checks = [{"name": "all_15_fits_completed", "passed": set(rows) == identities}]
    for panel in ("seen", "unseen"):
        metric = {method: [rows.get(f"{method}-{seed}", {}).get("metrics", {}).get(panel)
                           for seed in SEEDS] for method in METHODS}
        values = {method: [None if item is None else base.macro_exact(item) for item in items]
                  for method, items in metric.items()}
        means = {method: None if None in vals else sum(vals, Fraction()) / len(SEEDS)
                 for method, vals in values.items()}
        literal = references.get("literal", {}).get(panel)
        literal_macro = None if literal is None else base.macro_exact(literal)

        def add(name, left, right, comparison="ge", panel=panel, **extra):
            present = left is not None and right is not None
            passed = present and (left >= right if comparison == "ge" else left <= right)
            checks.append({"name": f"{panel}/{name}", "passed": bool(passed),
                           "left": None if left is None else float(left),
                           "right": None if right is None else float(right), "comparison": comparison,
                           "arithmetic": "direct float64; no epsilon" if comparison == "le"
                           else "exact integer count ratios; no epsilon", **extra})

        add("macro_gain_over_literal_3pp", means[PRIMARY],
            None if literal_macro is None else literal_macro + Fraction(3, 100))
        add("macro_gain_over_scalar_half_pp", means[PRIMARY],
            None if means["scalar"] is None else means["scalar"] + Fraction(1, 200))
        nll = {}
        for method in (PRIMARY, "scalar"):
            vals = [None if item is None else item["micro"]["nll"] for item in metric[method]]
            nll[method] = (base.mean_defined(vals) if all(value is None or
                           (type(value) in (int, float) and math.isfinite(value)) for value in vals) else None)
        add("micro_nll_nonworse_scalar", nll[PRIMARY], nll["scalar"], "le")
        wins = None if None in values[PRIMARY] + values["scalar"] else sum(
            a > b for a, b in zip(values[PRIMARY], values["scalar"], strict=True))
        add("strict_macro_win_at_least_two_pairs", wins, 2)
        controls = ("readout", "candidate_gru")
        strongest = None if any(means[m] is None for m in controls) else max(controls, key=lambda m: means[m])
        add("macro_nonworse_best_conventional", means[PRIMARY],
            None if strongest is None else means[strongest], strongest=strongest)
        revisions = [None if item is None or item["revision"]["count"] == 0 else
                     Fraction(item["revision"]["correct"], item["revision"]["count"])
                     for item in metric[PRIMARY]]
        revision_mean = None if None in revisions else sum(revisions, Fraction()) / len(SEEDS)
        literal_revision = None if literal is None or literal["revision"]["count"] == 0 else Fraction(
            literal["revision"]["correct"], literal["revision"]["count"])
        add("revision_within_1pp_literal", revision_mean,
            None if literal_revision is None else literal_revision - Fraction(1, 100))
    return {"passed": all(check["passed"] for check in checks),
            "checks_passed": sum(check["passed"] for check in checks), "checks_total": len(checks),
            "panel_checks_total": 12, "checks": checks,
            "scope": "Fixed development screen; official test untouched; no full-DST or novelty certification"}


def verify_execution_members(run):
    expected = {"plan.json", "completed.json", "references.npz"} | {
        f"fits/{method}-{seed}/{name}" for method in METHODS for seed in SEEDS
        for name in ("completed.json", "weights.pt", "dev-predictions.npz")}
    paths = list(run.rglob("*"))
    base.require(not any(path.is_symlink() for path in paths), "Execution contains a symlink")
    base.require({path.relative_to(run).as_posix() for path in paths if path.is_file()} == expected,
                 "Exact 48 execution files required")


def authenticate(run, packet_path, lexical_path, plan_sha256, completed_sha256, root=ROOT):
    for directory in (run, packet_path, lexical_path):
        base.clean(directory)
    base.bind(run / "plan.json", plan_sha256)
    base.bind(run / "completed.json", completed_sha256)
    verify_execution_members(run)
    plan, completed = base.read(run / "plan.json"), base.read(run / "completed.json")
    base.require(plan["study"] == "dialogue-copy-v1" and tuple(plan["config"]["methods"]) == METHODS
                 and tuple(plan["config"]["seeds"]) == SEEDS, "Study/method/seed mismatch")
    base.require(plan["practical_checks"] == PRACTICAL_CHECKS, "Practical criteria changed")
    source_map = plan["source_sha256"]
    base.require(type(source_map) is dict and set(source_map) == REQUIRED_SOURCES, "Exact 13-source closure required")
    base.require(source_map[BASE_PATH] == BASE_SHA256
                 and source_map["scripts/report_dialogue_copy.py"] == base.sha(__file__), "Reporter identity mismatch")
    for name, digest in source_map.items():
        base.bind(base.safe(root, name), digest)
    base.require(type(plan["runtime"]) is dict and plan["runtime"], "Missing execution runtime")
    base.require(completed["status"] == "completed" and completed["fit_count"] == 15
                 and completed["plan_sha256"] == plan_sha256
                 and type(completed["external_model_api_calls"]) is int
                 and completed["external_model_api_calls"] == 0, "Incomplete/foreign training execution")
    base.positive(completed["wall_seconds"], "execution wall time")
    base.bind(packet_path / "completed.json", plan["packet_completed_sha256"])
    packet_receipt = base.read(packet_path / "completed.json")
    base.require(packet_receipt["status"] == "completed" and set(packet_receipt["files"]) == {
        "encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}, "Feature packet incomplete")
    for name, digest in packet_receipt["files"].items():
        base.bind(base.safe(packet_path, name), digest)
    packet = base.read(packet_path / "packet.json")
    expected, candidate_counts = base.expected_records(packet)
    base.require(packet_receipt["cohorts"]["dev"]["queries"] == len(candidate_counts)
                 and packet_receipt["cohorts"]["dev"]["dialogues"] == len(packet["cohorts"]["dev"]),
                 "Feature cohort count mismatch")
    base.bind(lexical_path / "completed.json", plan["lexical_completed_sha256"])
    lexical = base.read(lexical_path / "completed.json")
    base.require(lexical["status"] == "completed" and lexical["packet_completed_sha256"] ==
                 plan["packet_completed_sha256"] and type(lexical["files"]) is dict and lexical["files"],
                 "Lexical receipt incomplete or uses another packet")
    for name, digest in lexical["files"].items():
        base.bind(base.safe(lexical_path, name), digest)
    base.require(set(lexical["files"]) == {"started.json", "index.json", "lexical.npy"},
                 "Lexical payload membership mismatch")
    base.require(type(lexical["source_sha256"]) is dict and lexical["source_sha256"], "Lexical source map absent")
    base.require(set(lexical["source_sha256"]) <= set(source_map), "Lexical source missing from scientific closure")
    for name, digest in lexical["source_sha256"].items():
        base.bind(base.safe(root, name), digest)
        base.require(name not in source_map or source_map[name] == digest, "Lexical/plan source identity differs")
    ids = [f"{method}-{seed}" for seed in SEEDS for method in METHODS]
    base.require([f"{row['method']}-{row['seed']}" for row in completed["fits"]] == ids, "Fit closure/order mismatch")
    base.require({p.name for p in (run / "fits").iterdir()} == set(ids), "Missing/foreign fit directory")
    config, train = plan["config"], packet["cohorts"]["train"]
    for key in ("epochs", "batch_size"):
        base.positive(config[key], key, integer=True)
    updates = math.ceil(len(train) / config["batch_size"]) * config["epochs"]
    train_queries = sum(len(d["queries"]) for d in train) * config["epochs"]
    real_questions = sum(len(d["turns"]) * len({q["query"] for q in d["queries"]}) for d in train) * config["epochs"]
    real_candidates = sum(len(d["turns"]) * sum(len(packet["queries"][qi]["candidate_ids"])
                          for qi in {q["query"] for q in d["queries"]}) for d in train) * config["epochs"]
    rows, files, paired_initial = {}, {"plan.json": plan_sha256, "completed.json": completed_sha256}, {}
    for record, name in zip(completed["fits"], ids, strict=True):
        folder = run / "fits" / name
        base.clean(folder)
        base.require({p.name for p in folder.iterdir()} == {"weights.pt", "dev-predictions.npz", "completed.json"},
                     "Fit member set mismatch")
        own = base.read(folder / "completed.json")
        base.require(own == record and own["status"] == "completed", "Fit receipt differs from completion")
        base.require(own["epochs"] == config["epochs"] and len(own["training_losses"]) == config["epochs"]
                     and own["updates"] == updates and own["training_queries"] == train_queries,
                     "Incomplete training counts")
        base.require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in own["training_losses"]),
                     "Invalid training losses")
        for key in ("parameters", "parameters_with_final_gradient"):
            base.positive(own[key], key, integer=True)
        base.require(own["parameters_with_final_gradient"] <= own["parameters"], "Invalid active parameter count")
        configuration = own.get("configuration")
        base.require(type(configuration) is dict, "Fit configuration required")
        base.require(configuration["class"] == "DialogueCopyMemory" and configuration["method"] == own["method"]
                     and configuration["parameters"] == own["parameters"], "Configuration identity mismatch")
        for key in ("projection_dim", "hidden_dim", "gru_width"):
            base.require(configuration[key] == config[key], "Configuration dimensions differ from plan")
        initial = own["initial_tensors_sha256"]
        base.require(type(initial) is str and re.fullmatch(r"[0-9a-f]{64}", initial), "Missing initialization identity")
        if own["method"] in ("scalar", "selective", "selective_no_lexical"):
            base.require(paired_initial.setdefault(own["seed"], initial) == initial, "Unpaired copy initialization")
        shapes = own["training_actor_shapes"]
        shape_fields = {"real_turns", "padded_turn_positions", "padded_query_positions", "padded_candidate_positions"}
        base.require(shape_fields <= set(shapes) <= shape_fields | {"scored_queries", "real_question_steps", "real_candidate_steps"}
                     and all(type(v) is int and v > 0 for v in shapes.values()), "Invalid actor shape counts")
        for key, expected_count in (("scored_queries", train_queries), ("real_question_steps", real_questions),
                                    ("real_candidate_steps", real_candidates)):
            base.require(key not in shapes or shapes[key] == expected_count, "Actor work inconsistent: " + key)
        base.require(shapes["real_turns"] == sum(len(d["turns"]) for d in train) * config["epochs"]
                     and shapes["padded_turn_positions"] >= shapes["real_turns"]
                     and shapes["padded_query_positions"] >= max(train_queries, real_questions)
                     and shapes["padded_candidate_positions"] >= max(shapes["padded_query_positions"], real_candidates),
                     "Inconsistent actor work")
        base.positive(own["train_wall_seconds"], "training wall time")
        base.positive(own["evaluation"]["wall_seconds"], "evaluation wall time")
        base.require(own["evaluation"]["queries"] == len(candidate_counts), "Incomplete evaluation")
        for member, digest in (("weights.pt", own["weights_sha256"]), ("dev-predictions.npz", own["predictions_sha256"])):
            base.bind(folder / member, digest)
            files[f"fits/{name}/{member}"] = digest
        files[f"fits/{name}/completed.json"] = base.sha(folder / "completed.json")
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        base.validate_predictions(arrays, expected, candidate_counts)
        rows[name] = {key: own[key] for key in ("method", "seed", "parameters", "parameters_with_final_gradient",
            "updates", "training_queries", "training_actor_shapes", "initial_tensors_sha256", "train_wall_seconds",
            "predictions_sha256", "weights_sha256")}
        rows[name].update(metrics=base.metrics(arrays), eval_wall_seconds=own["evaluation"]["wall_seconds"],
                          configuration_source_bound=configuration,
                          evaluation_metadata_source_bound={key: value for key, value in own["evaluation"].items()
                                                            if key not in ("wall_seconds", "queries")})
    reference = completed["references"]
    base.require(reference["file"] == "references.npz" and reference["queries"] == len(candidate_counts),
                 "Reference identity/count mismatch")
    base.positive(reference["wall_seconds"], "reference wall time")
    base.bind(run / "references.npz", reference["sha256"])
    with np.load(run / "references.npz", allow_pickle=False) as saved:
        references = base.reference_metrics({key: saved[key] for key in saved.files}, expected, candidate_counts)
    files["references.npz"] = reference["sha256"]
    return plan, completed, packet_receipt, lexical, rows, files, references


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharey=True)
    labels = ("Readout", "Scalar\ncopy", "Selective\ncopy", "Selective\nno lexical", "Candidate\nGRU")
    gate = summary["continuation_gate"]
    fig.suptitle("Dialogue copy memory: development comparison", fontsize=16, y=.98)
    fig.text(.5, .935, f"Fixed selective-copy screen: {'PASS' if gate['passed'] else 'FAIL'} "
             f"({gate['checks_passed']}/13 checks passed) | All 15 final fits", ha="center")
    for col, panel in enumerate(("seen", "unseen")):
        for line, metric in enumerate(("macro_three", "revision")):
            ax = axes[line, col]
            values = np.asarray([[summary["rows"][f"{method}-{seed}"]["metrics"][panel][metric]["accuracy"]
                                   for seed in SEEDS] for method in METHODS], dtype=float) * 100
            means = [summary["families"][method][panel][metric]["accuracy"] for method in METHODS]
            means = np.asarray(means, dtype=float) * 100
            ax.bar(np.arange(5), means, color=["#4682a3" if m == PRIMARY else "#c6cfd9" for m in METHODS], zorder=2)
            for index, marker in enumerate(("o", "s", "^")):
                ax.scatter(np.arange(5) + (index - 1) * .12, values[:, index], marker=marker,
                           s=22, color="#223c4b", edgecolors="white", linewidths=.4, zorder=4)
            for reference, color, style in (("literal", "#a13c26", "--"), ("none", "#606060", ":")):
                value = summary["references"][reference][panel][metric]["accuracy"]
                if value is not None:
                    ax.axhline(100 * value, color=color, linestyle=style, linewidth=1.7, zorder=3)
            support = summary["references"]["literal"][panel]["revision"]["count"]
            ax.set_title(f"{panel.title()} services" + (f": {support} revision queries" if line else ""))
            ax.set_xticks(np.arange(5), labels, fontsize=9)
            ax.set_ylim(-1, 101)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Three-stratum macro accuracy (%)" if line == 0 else "Revision accuracy (%)")
    handles = [Line2D([], [], color="#a13c26", linestyle="--", label="Literal mention + carry"),
               Line2D([], [], color="#606060", linestyle=":", label="Always NOT_MENTIONED"),
               Line2D([], [], color="#223c4b", marker="o", linestyle="none", label="Three paired fit points")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .045, "Macro = equal mean of unmentioned retention, assigned retention, and changed state.", ha="center", fontsize=9)
    fig.text(.5, .02, "The same development queries are reused across seeds; points are not independent datasets.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, top=.825, bottom=.12, hspace=.37, wspace=.14)
    fig.savefig(out / "comparison.png", dpi=170, facecolor="white")
    plt.close(fig)


def report(run, packet, lexical, plan_sha256, completed_sha256, out, *, root=ROOT):
    run, packet, lexical, out = map(Path, (run, packet, lexical, out))
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        base.write(out / "started.json", {"plan_sha256": plan_sha256, "completed_sha256": completed_sha256,
                                          "reporter_sha256": base.sha(__file__), "reused_metric_source_sha256": BASE_SHA256})
        plan, completed, packet_receipt, lexical_receipt, rows, files, references = authenticate(
            run, packet, lexical, plan_sha256, completed_sha256, root)
        summary = {"study": "dialogue-copy-v1", "status": "completed", "development_only": True,
                   "method_order": list(METHODS), "seeds": list(SEEDS), "rows": rows,
                   "families": family_metrics(rows), "references": references,
                   "continuation_gate": criteria(rows, references), "source_sha256": plan["source_sha256"],
                   "execution_runtime": plan["runtime"], "execution_wall_seconds": completed["wall_seconds"],
                   "reference_wall_seconds": completed["references"]["wall_seconds"],
                   "encoder": {key: packet_receipt[key] for key in ("wall_seconds", "unique_texts",
                               "encoder_sequences", "encoder_tokens_with_special")},
                   "lexical_preparation_source_bound": {key: lexical_receipt.get(key) for key in
                                                        ("wall_seconds", "counts", "float32_scalars", "bytes")},
                   "scope": "Saved-prediction cohort authentication and arithmetic; frozen prior metric helpers reused; no neural replay",
                   "zero_probability_policy": "No floor. NLL null with positive zero-target count means infinite NLL.",
                   "timing_scope": "Evaluation includes batch assembly and prediction storage; excludes encoder and lexical preparation. "
                       "Each unique schema is advanced through public turns; dense padded work is charged. "
                       "These batch timings are not incremental serving latency.",
                   "lexical_completed_sha256": plan["lexical_completed_sha256"]}
        summary["parameter_scope"] = ("Registered parameters and parameters belonging to tensors with final gradients; "
                                      "gradient counts do not imply every element had a nonzero gradient.")
        base.write(out / "summary.json", summary)
        gate = summary["continuation_gate"]
        lines = ["# Dialogue copy development results", "", summary["scope"] + ".", "",
                 f"Fixed selective-copy continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['checks_passed']}/13 checks).",
                 "Official development only, with supplied service/slot/candidates. Official test remains untouched.", "",
                 "| Method | Parameters per fit | Training seconds, 3 fits | Evaluation seconds, 3 fits |",
                 "|---|---:|---:|---:|"]
        for method in METHODS:
            selected = [rows[f"{method}-{seed}"] for seed in SEEDS]
            lines.append(f"| {method} | {', '.join(map(str, sorted({r['parameters'] for r in selected})))} | "
                         f"{math.fsum(r['train_wall_seconds'] for r in selected):.3f} | "
                         f"{math.fsum(r['eval_wall_seconds'] for r in selected):.3f} |")

        def display(value, scale=1):
            return "undefined" if value is None else f"{value * scale:.4f}"

        for panel in ("seen", "unseen"):
            lines.extend(["", f"## {panel.title()} services", "",
                          "| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |",
                          "|---|---:|---:|---:|---:|---:|"])
            for method in METHODS:
                row = summary["families"][method][panel]
                lines.append(f"| {method} | {display(row['macro_three']['accuracy'], 100)} | "
                             f"{display(row['micro']['nll'])} | {display(row['micro']['brier'])} | "
                             f"{display(row['revision']['accuracy'], 100)} | {row['revision']['count_per_fit']} |")
            for method, panels in references.items():
                row = panels[panel]
                lines.append(f"| Reference: {method} | {display(row['macro_three']['accuracy'], 100)} | not probabilistic | "
                             f"not probabilistic | {display(row['revision']['accuracy'], 100)} | {row['revision']['count']} |")
        lines.extend(["", summary["timing_scope"], "", ("Macro equally weights the three state strata, then the three fits. "
                      "No-lexical is an ablation, not a selectable primary. Missing required strata or revision support cannot pass their checks."),
                      "", ("Lexical extraction, literal decisions, initialization digests and neural computation are source/test bound. "
                      "This reporter reuses frozen cohort/metric functions and rechecks saved references, labels, probabilities and criteria; "
                      "it is not an independent implementation of every pipeline stage. Raw dialogue/predictions remain local.")])
        (out / "report.md").write_text("\n".join(lines) + "\n")
        render(summary, out)
        for name, digest in plan["source_sha256"].items():
            base.bind(base.safe(root, name), digest)
        base.bind(run / "plan.json", plan_sha256)
        base.bind(run / "completed.json", completed_sha256)
        for directory, receipt, digest in ((packet, packet_receipt, plan["packet_completed_sha256"]),
                                            (lexical, lexical_receipt, plan["lexical_completed_sha256"])):
            base.bind(directory / "completed.json", digest)
            for name, item in receipt["files"].items():
                base.bind(base.safe(directory, name), item)
        for name, digest in files.items():
            base.bind(base.safe(run, name), digest)
        verify_execution_members(run)
        payloads = ("started.json", "summary.json", "report.md", "comparison.png")
        receipt = {"status": "completed", "study": "dialogue-copy-v1", "plan_sha256": plan_sha256,
                   "execution_completed_sha256": completed_sha256,
                   "feature_packet_completed_sha256": plan["packet_completed_sha256"],
                   "lexical_completed_sha256": plan["lexical_completed_sha256"],
                   "reporter_source_sha256": base.sha(__file__), "reused_metric_source_sha256": BASE_SHA256,
                   "source_sha256": plan["source_sha256"], "execution_members": files,
                   "fit_count": 15, "neural_calls": 0, "encoder_calls": 0,
                   "continuation_passed": gate["passed"], "checks_passed": gate["checks_passed"], "checks_total": 13,
                   "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                   "wall_seconds": time.perf_counter() - start,
                   "files": {name: {"sha256": base.sha(out / name), "bytes": (out / name).stat().st_size} for name in payloads}}
        base.write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            base.write(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                            "error": str(error), "wall_seconds": time.perf_counter() - start})
        except BaseException as preservation_error:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failed-receipt preservation error: " + repr(preservation_error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "packet", "lexical", "plan-sha256", "completed-sha256", "out"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    result = report(args.run, args.packet, args.lexical, args.plan_sha256, args.completed_sha256, args.out)
    print(base.json.dumps({"status": result["status"], "continuation_passed": result["continuation_passed"]}))
