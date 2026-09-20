"""Authenticate saved dialogue predictions and recompute development metrics.

No models, encoders, training modules, random generators or test data are read.
The feature packet is the authenticated label authority. Its extraction and the
neural forward computation are source/test bound, not independently reexecuted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("current", "gru", "attention", "gated_delta", "kalman", "innovation_kalman", "carry")
SEEDS = (1729, 2718, 3141)
PANELS = ("all", "seen", "unseen")
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
PRACTICAL_CHECKS = {"panels": ["seen", "unseen"], "primary": "innovation_kalman",
                    "macro_definition": "equal three-stratum mean", "macro_gain_over_kalman": .01,
                    "micro_nll_no_worse_than": "kalman", "strict_paired_macro_wins_over_kalman": 2,
                    "macro_no_worse_than": "gated_delta", "conventional_controls": ["gru", "attention", "carry"],
                    "maximum_macro_deficit_to_best_conventional": .01,
                    "required_complete_fits": 21, "total_panel_checks": 10}
FIELDS = {"probabilities", "labels", "choice", "bin", "unseen", "dialogue", "time", "query"}
SOURCES = {"scripts/study_dialogue_memory.py", "scripts/prepare_sgd_state.py",
           "src/openjev/research/dialogue_state_data.py", "src/openjev/research/dialogue_fast_memory.py",
           "src/openjev/research/dialogue_carry.py", "tests/test_dialogue_state_data.py",
           "tests/test_dialogue_fast_memory.py", "tests/test_dialogue_carry.py",
           "tests/test_study_dialogue_memory.py", "research/dialogue-memory-protocol.md",
           "scripts/report_dialogue_memory.py", "tests/test_report_dialogue_memory.py"}
FAILURES = ("failed.json", "late-completion.json", "cleanup-error.json", "completion-before-cleanup-error.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bind(path, expected):
    require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid SHA-256")
    require(Path(path).is_file() and sha(path) == expected, "Changed or missing file: " + str(path))


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Unsafe member path")
    result = root / name
    require(result.resolve().is_relative_to(root.resolve()), "Member escapes root")
    return result


def clean(directory):
    require(not any((directory / name).exists() for name in FAILURES), "Retained failure marker present")


def positive(value, name, integer=False):
    require((type(value) is int if integer else type(value) in (int, float))
            and math.isfinite(value) and value > 0, "Invalid " + name)


def expected_records(packet):
    """Independently flatten the public packet's exact development cohort."""
    queries, dialogs = packet["queries"], packet["cohorts"]["dev"]
    require(type(queries) is list and type(dialogs) is list and dialogs, "Missing packet cohort")
    fields = {k: [] for k in FIELDS - {"probabilities", "choice"}}
    counts, seen, dialogue_ids = [], set(), set()
    for dialog in dialogs:
        identity = dialog["id"]
        require(type(identity) is str and identity and identity not in dialogue_ids, "Duplicate dialogue")
        dialogue_ids.add(identity)
        require(type(dialog["turns"]) is list and dialog["turns"], "Empty dialogue prefix")
        require(type(dialog["queries"]) is list and dialog["queries"], "Empty scored dialogue")
        previous = {}
        for row in dialog["queries"]:
            qi, ti, label = row["query"], row["time"], row["label"]
            require(type(qi) is int and 0 <= qi < len(queries) and type(ti) is int
                    and 0 <= ti < len(dialog["turns"]), "Invalid query/time")
            query = queries[qi]
            ids = query["candidate_ids"]
            require(query["split"] == "dev" and 3 <= len(ids) <= 12 and len(set(ids)) == len(ids)
                    and ids[:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"]
                    and all(type(x) is str and x.startswith("value:") for x in ids[2:])
                    and len(query["candidates"]) == len(ids), "Invalid supplied candidate catalog")
            require(type(label) is int and 0 <= label < len(ids), "Invalid packet label")
            require(type(row["unseen"]) is bool and type(row["dontcare"]) is bool
                    and row["dontcare"] == (label == 1), "Invalid unseen/DONTCARE label metadata")
            require(row["bin"] in BINS, "Unknown transition bin")
            old_time, old_label = previous.get(qi, (-1, 0))
            require(ti > old_time, "Noncausal query record ordering")
            transition = ("unmentioned_retention" if label == old_label == 0 else
                          "assigned_retention" if label == old_label else
                          "first_assignment" if old_label == 0 else "clear" if label == 0 else "revision")
            require(row["bin"] == transition, "Transition bin/current label inconsistent")
            previous[qi] = (ti, label)
            key = (identity, ti, qi)
            require(key not in seen, "Duplicate query identity")
            seen.add(key)
            for name, value in (("labels", label), ("bin", row["bin"]), ("unseen", row["unseen"]),
                                ("dialogue", identity), ("time", ti), ("query", qi)):
                fields[name].append(value)
            counts.append(len(ids))
    return {name: np.asarray(values) for name, values in fields.items()}, np.asarray(counts)


def validate_predictions(arrays, expected, candidate_counts):
    require(set(arrays) == FIELDS, "Prediction member set mismatch")
    n = len(candidate_counts)
    for field, values in expected.items():
        actual = arrays[field]
        kind = "iu" if field in ("labels", "time", "query") else "b" if field == "unseen" else "U"
        require(actual.shape == (n,) and actual.dtype.kind in kind and np.array_equal(actual, values),
                "Prediction cohort/label mismatch: " + field)
    p, choice = arrays["probabilities"], arrays["choice"]
    require(p.shape == (n, 12) and p.dtype.kind == "f" and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all(), "Invalid saved probabilities")
    mask = np.arange(12)[None, :] < candidate_counts[:, None]
    require(np.count_nonzero(p[~mask]) == 0, "Nonzero candidate padding")
    require(np.allclose(p.astype(np.float64).sum(-1), 1., atol=1e-5, rtol=0), "Unnormalized probabilities")
    require(choice.shape == (n,) and choice.dtype.kind in "iu"
            and np.array_equal(choice, p.argmax(-1)), "Choice disagrees with probabilities")


def score(p, y, selection):
    n = int(selection.sum())
    if not n:
        return {"count": 0, "correct": 0, "accuracy": None, "nll": None, "brier": None,
                "zero_target_probabilities": 0}
    x, labels = p[selection].astype(np.float64), y[selection]
    targets = x[np.arange(n), labels]
    zeros = int((targets == 0).sum())
    nll = None if zeros else math.fsum(-math.log(float(z)) for z in targets) / n
    # Multiclass Brier is the SUM over candidates, then mean over query records.
    residual = x.copy()
    residual[np.arange(n), labels] -= 1
    values = np.square(residual).sum(-1)
    correct = int((x.argmax(-1) == labels).sum())
    return {"count": n, "correct": correct, "accuracy": correct / n, "nll": nll,
            "brier": math.fsum(map(float, values)) / n, "zero_target_probabilities": zeros}


def mean_defined(values):
    return math.fsum(values) / len(values) if values and all(x is not None for x in values) else None


def metrics(arrays):
    output = {}
    for panel in PANELS:
        selected = (np.ones(len(arrays["labels"]), dtype=bool) if panel == "all"
                    else arrays["unseen"] if panel == "unseen" else ~arrays["unseen"])
        bins = {name: score(arrays["probabilities"], arrays["labels"], selected & (arrays["bin"] == name))
                for name in BINS}
        strata = {name: bins[name] for name in STRATA[:2]}
        strata["changed"] = score(arrays["probabilities"], arrays["labels"],
                                   selected & np.isin(arrays["bin"], BINS[2:]))
        output[panel] = {"micro": score(arrays["probabilities"], arrays["labels"], selected),
                         "bins": bins, "strata": strata,
                         "macro_three": {metric: mean_defined([strata[s][metric] for s in STRATA])
                                         for metric in ("accuracy", "nll", "brier")},
                         "revision": bins["revision"],
                         "not_mentioned_count": int((selected & (arrays["labels"] == 0)).sum()),
                         "dontcare_count": int((selected & (arrays["labels"] == 1)).sum())}
    return output


def reference_metrics(arrays, expected, candidate_counts):
    """Categorical accuracy only: deterministic rules have no forecast probabilities."""
    require(set(arrays) == set(expected) | {"pred_none", "pred_literal"}, "Reference field set mismatch")
    for name, values in expected.items():
        require(arrays[name].dtype == values.dtype and np.array_equal(arrays[name], values),
                "Reference cohort mismatch: " + name)
    n, results = len(candidate_counts), {}
    for name in ("pred_none", "pred_literal"):
        predictions = arrays[name]
        require(predictions.shape == (n,) and predictions.dtype.kind in "iu"
                and ((predictions >= 0) & (predictions < candidate_counts)).all(), "Invalid reference choice")
        require((predictions == 0).all() if name == "pred_none" else (predictions != 1).all(),
                "Reserved value inconsistent with reference contract")
        def count(selection, predictions=predictions):
            size = int(selection.sum())
            correct = int(((predictions == expected["labels"]) & selection).sum())
            return {"count": size, "correct": correct, "accuracy": correct / size if size else None}
        result = {}
        for panel in PANELS:
            select = (np.ones(n, bool) if panel == "all" else expected["unseen"] if panel == "unseen"
                      else ~expected["unseen"])
            bins = {label: count(select & (expected["bin"] == label)) for label in BINS}
            strata = {label: bins[label] for label in STRATA[:2]}
            strata["changed"] = count(select & np.isin(expected["bin"], BINS[2:]))
            result[panel] = {"micro": count(select), "bins": bins, "strata": strata,
                             "macro_three": {"accuracy": mean_defined([strata[s]["accuracy"] for s in STRATA])},
                             "revision": bins["revision"]}
        results[name.removeprefix("pred_")] = result
    return results


def family_metrics(rows):
    families = {}
    for method in METHODS:
        family = {}
        for panel in PANELS:
            paired = [rows[f"{method}-{seed}"]["metrics"][panel] for seed in SEEDS]
            def pooled(items):
                require(len({x["count"] for x in items}) == 1, "Unequal fit membership")
                return {"count_per_fit": items[0]["count"], "fit_count": len(SEEDS),
                        **{k: mean_defined([x[k] for x in items]) for k in ("accuracy", "nll", "brier")},
                        "zero_target_probabilities_all_fits": sum(x["zero_target_probabilities"] for x in items)}
            family[panel] = {"micro": pooled([x["micro"] for x in paired]),
                             "macro_three": {k: mean_defined([x["macro_three"][k] for x in paired])
                                             for k in ("accuracy", "nll", "brier")},
                             "bins": {name: pooled([x["bins"][name] for x in paired]) for name in BINS},
                             "revision": pooled([x["revision"] for x in paired])}
        families[method] = family
    return families


def macro_exact(row):
    parts = row["strata"]
    if any(parts[name]["count"] == 0 for name in STRATA):
        return None
    return sum((Fraction(parts[name]["correct"], parts[name]["count"]) for name in STRATA), Fraction()) / 3


def criteria(rows):
    """Exact rational accuracy thresholds; direct inclusive floating NLL comparison."""
    checks = [{"name": "all_21_fits_completed", "passed": set(rows) == {
        f"{method}-{seed}" for method in METHODS for seed in SEEDS}}]
    for panel in ("seen", "unseen"):
        values = {method: [macro_exact(rows[f"{method}-{seed}"]["metrics"][panel]) for seed in SEEDS]
                  for method in METHODS}
        means = {method: None if None in v else sum(v, Fraction()) / 3 for method, v in values.items()}
        primary, kalman = means["innovation_kalman"], means["kalman"]
        def add(name, left, right, operation, panel=panel, **extras):
            ok = left is not None and right is not None
            passed = ok and (left >= right if operation == "ge" else left <= right)
            checks.append({"name": panel + "/" + name, "passed": bool(passed), "comparison": operation,
                           "left": None if left is None else float(left),
                           "right": None if right is None else float(right),
                           "arithmetic": ("exact integer-ratio accuracy; no epsilon" if isinstance(left, Fraction)
                                          else "integer paired-win count" if name.startswith("strict_macro")
                                          else "direct float64 mean NLL; no epsilon"), **extras})
        add("macro_gain_over_kalman_1pp", primary, None if kalman is None else kalman + Fraction(1, 100), "ge")
        nll = {method: mean_defined([rows[f"{method}-{seed}"]["metrics"][panel]["micro"]["nll"]
                                    for seed in SEEDS]) for method in ("innovation_kalman", "kalman")}
        add("micro_nll_nonworse_kalman", nll["innovation_kalman"], nll["kalman"], "le")
        wins = (None if None in values["innovation_kalman"] + values["kalman"] else
                sum(a > b for a, b in zip(values["innovation_kalman"], values["kalman"], strict=True)))
        add("strict_macro_win_at_least_two_pairs", wins, 2, "ge")
        add("macro_nonworse_delta", primary, means["gated_delta"], "ge")
        conventional = ("gru", "attention", "carry")
        strongest = (None if any(means[m] is None for m in conventional)
                     else max(conventional, key=lambda m: means[m]))
        target = None if strongest is None else means[strongest] - Fraction(1, 100)
        add("within_1pp_strongest_conventional", primary, target, "ge", strongest=strongest)
    return {"passed": all(c["passed"] for c in checks), "checks_passed": sum(c["passed"] for c in checks),
            "checks_total": len(checks), "panel_checks_total": 10, "checks": checks,
            "scope": "Fixed development screen; not full DST, significance or architecture novelty"}


def authenticate(run, packet_path, plan_sha256, completed_sha256, root=ROOT):
    clean(run)
    clean(packet_path)
    bind(run / "plan.json", plan_sha256)
    bind(run / "completed.json", completed_sha256)
    plan, completed = read(run / "plan.json"), read(run / "completed.json")
    require(plan["study"] == "dialogue-memory-v1" and tuple(plan["config"]["methods"]) == METHODS
            and tuple(plan["config"]["seeds"]) == SEEDS, "Study/method/seed mismatch")
    require(plan["practical_checks"] == PRACTICAL_CHECKS, "Practical criteria changed")
    require(set(plan["source_sha256"]) == SOURCES, "Scientific source closure mismatch")
    for name, digest in plan["source_sha256"].items():
        bind(safe(root, name), digest)
    require(completed["status"] == "completed" and completed["fit_count"] == 21
            and completed["plan_sha256"] == plan_sha256, "Incomplete training execution")
    bind(packet_path / "completed.json", plan["packet_completed_sha256"])
    packet_receipt = read(packet_path / "completed.json")
    require(packet_receipt["status"] == "completed" and set(packet_receipt["files"]) == {
        "encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}, "Feature packet incomplete")
    for name, digest in packet_receipt["files"].items():
        bind(safe(packet_path, name), digest)
    packet = read(packet_path / "packet.json")
    expected, candidate_counts = expected_records(packet)
    require(packet_receipt["cohorts"]["dev"]["queries"] == len(candidate_counts)
            and packet_receipt["cohorts"]["dev"]["dialogues"] == len(packet["cohorts"]["dev"]),
            "Feature cohort count mismatch")
    ids = [f"{method}-{seed}" for seed in SEEDS for method in METHODS]
    require([f"{r['method']}-{r['seed']}" for r in completed["fits"]] == ids, "Fit closure/order mismatch")
    require({p.name for p in (run / "fits").iterdir()} == set(ids), "Foreign or missing fit directory")
    rows, files, initial_by_seed = {}, {}, {}
    config = plan["config"]
    train = packet["cohorts"]["train"]
    expected_updates = math.ceil(len(train) / config["batch_size"]) * config["epochs"]
    train_queries = sum(len(d["queries"]) for d in train) * config["epochs"]
    for record, name in zip(completed["fits"], ids, strict=True):
        folder = run / "fits" / name
        clean(folder)
        require({p.name for p in folder.iterdir()} == {"weights.pt", "dev-predictions.npz", "completed.json"},
                "Fit member set mismatch")
        own = read(folder / "completed.json")
        require(own == record and own["status"] == "completed", "Fit receipt differs from completion")
        require(own["epochs"] == config["epochs"] and len(own["training_losses"]) == config["epochs"]
                and own["updates"] == expected_updates and own["training_queries"] == train_queries,
                "Incomplete training counts")
        require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in own["training_losses"]),
                "Invalid training losses")
        positive(own["parameters"], "parameters", integer=True)
        positive(own["parameters_with_final_gradient"], "gradient-active parameters", integer=True)
        require(own["parameters_with_final_gradient"] <= own["parameters"], "Active parameters exceed total")
        initial = own["initial_tensors_sha256"]
        require(type(initial) is str and re.fullmatch(r"[0-9a-f]{64}", initial), "Missing initial tensor binding")
        if own["method"] in ("gated_delta", "kalman", "innovation_kalman"):
            require(initial_by_seed.setdefault(own["seed"], initial) == initial, "Unpaired matrix initialization")
        shapes = own["training_actor_shapes"]
        require(set(shapes) == {"real_turns", "padded_turn_positions", "padded_query_positions",
                               "padded_candidate_positions"}
                and all(type(v) is int and v > 0 for v in shapes.values()), "Invalid actor shape counts")
        require(shapes["real_turns"] == sum(len(d["turns"]) for d in train) * config["epochs"]
                and shapes["padded_turn_positions"] >= shapes["real_turns"]
                and shapes["padded_query_positions"] >= train_queries
                and shapes["padded_candidate_positions"] >= shapes["padded_query_positions"],
                "Inconsistent actor shape counts")
        positive(own["train_wall_seconds"], "training time")
        positive(own["evaluation"]["wall_seconds"], "evaluation time")
        require(own["evaluation"]["queries"] == len(candidate_counts), "Incomplete evaluation")
        for member, digest in (("weights.pt", own["weights_sha256"]),
                               ("dev-predictions.npz", own["predictions_sha256"])):
            bind(folder / member, digest)
            files[f"fits/{name}/{member}"] = digest
        files[f"fits/{name}/completed.json"] = sha(folder / "completed.json")
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        validate_predictions(arrays, expected, candidate_counts)
        rows[name] = {"method": own["method"], "seed": own["seed"], "metrics": metrics(arrays),
                      "parameters": own["parameters"], "updates": own["updates"],
                      "parameters_with_final_gradient": own["parameters_with_final_gradient"],
                      "training_actor_shapes": shapes, "initial_tensors_sha256": initial,
                      "training_queries": own["training_queries"],
                      "train_wall_seconds": own["train_wall_seconds"],
                      "eval_wall_seconds": own["evaluation"]["wall_seconds"],
                      "innovation_diagnostics_source_bound": own["evaluation"].get("innovation_diagnostics"),
                      "predictions_sha256": own["predictions_sha256"], "weights_sha256": own["weights_sha256"]}
    reference = completed["references"]
    require(reference["file"] == "references.npz" and reference["queries"] == len(candidate_counts),
            "Reference file/count mismatch")
    positive(reference["wall_seconds"], "reference wall time")
    bind(run / "references.npz", reference["sha256"])
    with np.load(run / "references.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    references = reference_metrics(arrays, expected, candidate_counts)
    files["references.npz"] = reference["sha256"]
    return plan, completed, packet_receipt, rows, files, references


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("Current", "GRU", "Attention", "Delta", "Kalman", "Innovation", "Carry")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    colors = ["#8596a6"] * 7
    colors[5] = "#c75c38"
    for col, panel in enumerate(PANELS):
        for r, metric in enumerate(("macro_three", "revision")):
            ax = axes[r, col]
            vals = [summary["families"][m][panel][metric]["accuracy"] for m in METHODS]
            for i, (method, mean) in enumerate(zip(METHODS, vals, strict=True)):
                if mean is not None:
                    ax.bar(i, mean * 100, color=colors[i], alpha=.6, width=.7)
                for j, seed in enumerate(SEEDS):
                    value = summary["rows"][f"{method}-{seed}"]["metrics"][panel][metric]["accuracy"]
                    if value is not None:
                        ax.scatter(i + (j - 1) * .16, value * 100, s=19, marker=("o", "s", "^")[j],
                                   color="#243544", zorder=3)
            ax.set_title(panel.capitalize() + (" services" if panel != "all" else " development"))
            ax.set_xticks(range(7), labels, rotation=35, ha="right", fontsize=9)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(("Three-stratum macro" if r == 0 else "Revision-only") + " accuracy (%)")
            if r == 1:
                n = summary["families"]["current"][panel]["revision"]["count_per_fit"]
                ax.text(.02, .95, f"{n:,} revision queries per fit", transform=ax.transAxes, va="top", fontsize=8)
    gate = summary["continuation_gate"]
    fig.suptitle("Supplied-schema dialogue memory | Development only\n"
                 f"Fixed innovation screen: {'PASS' if gate['passed'] else 'FAIL'} "
                 f"{gate['checks_passed']}/{gate['checks_total']} checks including fit completion", fontsize=14)
    fig.text(.5, .01, "Bars: equal means of all three fits. Points: seeds 1729 (circle), 2718 (square), 3141 (triangle). "
             "Same development queries reused across fits; not full DST.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .91))
    fig.savefig(out / "accuracy.png", dpi=180)
    plt.close(fig)


def report(run, packet, plan_sha256, completed_sha256, out, *, root=ROOT):
    run, packet, out = Path(run), Path(packet), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        write(out / "started.json", {"plan_sha256": plan_sha256, "completed_sha256": completed_sha256,
                                     "reporter_sha256": sha(__file__)})
        plan, completed, packet_receipt, rows, files, references = authenticate(
            run, packet, plan_sha256, completed_sha256, root)
        families = family_metrics(rows)
        summary = {"study": "dialogue-memory-v1", "status": "completed", "development_only": True,
                   "rows": rows, "families": families, "references": references,
                   "reference_wall_seconds": completed["references"]["wall_seconds"],
                   "continuation_gate": criteria(rows),
                   "method_order": list(METHODS), "seeds": list(SEEDS),
                   "source_sha256": plan["source_sha256"], "execution_runtime": plan["runtime"],
                   "scope": "Saved-prediction arithmetic and cohort authentication; no neural replay",
                   "zero_probability_policy": "No floor. NLL null with positive zero-target count means infinite NLL.",
                   "encoder": {k: packet_receipt[k] for k in
                               ("wall_seconds", "unique_texts", "encoder_sequences", "encoder_tokens_with_special")},
                   "execution_wall_seconds": completed["wall_seconds"],
                   "logical_state_payload": {"current": "16 float scalars per dialogue (64 float32 bytes)",
                       "gru": "128 float scalars per dialogue (512 float32 bytes)",
                       "gated_delta": "256 float scalars per dialogue (1024 float32 bytes)",
                       "kalman": "512 float scalars per dialogue (2048 float32 bytes)",
                       "innovation_kalman": "512 float scalars per dialogue (2048 float32 bytes)",
                       "attention": "32*T float scalars plus T booleans (129*T float32/bool bytes)",
                       "carry": "C float log probabilities per independently replayed question; batched storage B*Q*C",
                       "scope": "Source-derived retained tensor payload only; excludes parameters, autograd, "
                       "temporary read/update copies, Python objects and encoder/cache. Not measured peak memory."},
                   "timing_scope": "Evaluation includes batch assembly and prediction storage, excludes encoder. "
                   "Carry replays a prefix for each question; no incremental-dictionary speed claim."}
        write(out / "summary.json", summary)
        lines = ["# Dialogue memory development results", "", summary["scope"] + ".",
                 "Official development only; supplied service/slot routing. No official-test or full-DST claim.", "",
                 (f"Fixed innovation continuation: **{'PASS' if summary['continuation_gate']['passed'] else 'FAIL'}** "
                  f"({summary['continuation_gate']['checks_passed']}/11, including all 21 fits)."), "",
                 "| Method | Parameters per fit | Training seconds (three fits) | Evaluation seconds (three fits) |",
                 "|---|---:|---:|---:|"]
        for method in METHODS:
            rr = [rows[f"{method}-{seed}"] for seed in SEEDS]
            params = sorted({r["parameters"] for r in rr})
            lines.append(f"| {method} | {', '.join(map(str, params))} | "
                         f"{math.fsum(r['train_wall_seconds'] for r in rr):.3f} | "
                         f"{math.fsum(r['eval_wall_seconds'] for r in rr):.3f} |")
        def display(value, scale=1):
            return "undefined" if value is None else f"{value * scale:.4f}"
        for panel in ("seen", "unseen"):
            lines.extend(["", f"## {panel.capitalize()} services", "",
                          "| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |",
                          "|---|---:|---:|---:|---:|---:|"])
            for method in METHODS:
                row = families[method][panel]
                lines.append(f"| {method} | {display(row['macro_three']['accuracy'], 100)} | "
                             f"{display(row['micro']['nll'])} | {display(row['micro']['brier'])} | "
                             f"{display(row['revision']['accuracy'], 100)} | {row['revision']['count_per_fit']} |")
            for method, reference in references.items():
                row = reference[panel]
                lines.append(f"| Reference: {method} | {display(row['macro_three']['accuracy'], 100)} | "
                             f"not probabilistic | not probabilistic | {display(row['revision']['accuracy'], 100)} | "
                             f"{row['revision']['count']} |")
        lines.extend(["", summary["timing_scope"], "",
                      ("All fit points and query/bin counts are retained in summary.json. "
                      "Accuracy thresholds use exact count ratios. Micro NLL uses saved probabilities without flooring. "
                      "Missing bins cannot satisfy their criteria. Raw text, weights and individual predictions stay local."), "",
                      ("The reporter verifies labels against the hash-bound feature packet. "
                       "Parser semantics, frozen encoder and neural forecasts are source/test bound, not independently rerun."),
                      ("Literal-reference decisions and innovation diagnostics are source-bound; their generation is not replayed. "
                      "Reference accuracies are independently recomputed from saved choices. Logical state payloads in "
                      "summary.json exclude autograd and temporary copies; no peak-memory advantage is established.")])
        (out / "report.md").write_text("\n".join(lines) + "\n")
        render(summary, out)
        # Detect source/receipt/payload changes during reporting without any inference.
        for name, digest in plan["source_sha256"].items():
            bind(safe(root, name), digest)
        bind(run / "completed.json", completed_sha256)
        bind(run / "plan.json", plan_sha256)
        bind(packet / "completed.json", plan["packet_completed_sha256"])
        for name, digest in packet_receipt["files"].items():
            bind(safe(packet, name), digest)
        for name, digest in files.items():
            bind(safe(run, name), digest)
        payloads = ("started.json", "summary.json", "report.md", "accuracy.png")
        receipt = {"status": "completed", "plan_sha256": plan_sha256,
                   "execution_completed_sha256": completed_sha256,
                   "feature_packet_completed_sha256": plan["packet_completed_sha256"],
                   "reporter_source_sha256": sha(__file__), "source_sha256": plan["source_sha256"],
                   "execution_members": files, "fit_count": 21, "neural_calls": 0, "encoder_calls": 0,
                   "continuation_passed": summary["continuation_gate"]["passed"],
                   "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                   "wall_seconds": time.perf_counter() - started,
                   "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                             for name in payloads}}
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                        "error": str(error), "wall_seconds": time.perf_counter() - started})
        except BaseException as preservation_error:  # noqa: BLE001 - keep the original failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failed-receipt preservation error: " + repr(preservation_error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "packet", "plan-sha256", "completed-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    receipt = report(args.run, args.packet, args.plan_sha256, args.completed_sha256, args.out)
    print(json.dumps({"status": receipt["status"], "continuation_passed": receipt["continuation_passed"]}))


if __name__ == "__main__":
    main()
