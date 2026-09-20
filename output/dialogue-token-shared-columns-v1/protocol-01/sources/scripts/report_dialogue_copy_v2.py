"""Saved-only report for fresh normalized-copy fits; no model or checkpoint loads.

The original metric/gate/plot helpers are explicitly reused by exact source hash.
Internal-state arithmetic is authenticated instrumentation, not neural replay.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = "scripts/report_dialogue_copy.py"
OLD_SHA = "e5601033aeebdd73cff71fed41e285f9db148db2eb1d70dcba202bbf34571202"
_source = (ROOT / OLD_PATH).read_bytes()
if hashlib.sha256(_source).hexdigest() != OLD_SHA:
    raise ValueError("Frozen reporting helper changed")
old = types.ModuleType("dialogue_copy_v2_frozen_reporting")
old.__file__ = str(ROOT / OLD_PATH)
exec(compile(_source, old.__file__, "exec"), old.__dict__)  # noqa: S102 - exact pinned local bytes only
base, np = old.base, old.np
METHODS, SEEDS, PANELS = old.METHODS, old.SEEDS, old.PANELS
PRACTICAL_CHECKS = old.PRACTICAL_CHECKS
SOURCES = old.REQUIRED_SOURCES | {
    "src/openjev/research/dialogue_copy_memory_v2.py", "tests/test_dialogue_copy_memory_v2.py",
    "scripts/study_dialogue_copy_v2.py", "tests/test_study_dialogue_copy_v2.py",
    "scripts/report_dialogue_copy_v2.py", "tests/test_report_dialogue_copy_v2.py",
    "research/dialogue-copy-v2-protocol.md",
}
CAP = 3600.
TOLERANCE = 2e-6
VERSION = "dialogue-copy-memory-v2-normalized"
OLD_PLAN = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
OLD_COMPLETED = "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300"
COUNT_KEYS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
              "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks",
              "result_checks", "mass_checks", "mass_above_one_count")
MAX_KEYS = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")
CONFIG = {"methods": list(METHODS), "seeds": list(SEEDS), "epochs": 20, "batch_size": 32,
          "learning_rate": .001, "weight_decay": .0001, "gradient_clip": 1., "threads": 4,
          "projection_dim": 64, "hidden_dim": 64, "gru_width": 16,
          "loss": "Equal training weight across three state-transition strata",
          "selection": "All final fits; no best epoch, seed or variant; development only"}


def execution_members(run):
    expected = {"plan.json", "started.json", "completed.json", "references.npz"} | {
        f"fits/{method}-{seed}/{name}" for method in METHODS for seed in SEEDS
        for name in ("completed.json", "weights.pt", "dev-predictions.npz", "batches.jsonl")}
    paths = list(run.rglob("*"))
    base.require(not any(p.is_symlink() for p in paths), "Symlink in execution")
    base.require({p.relative_to(run).as_posix() for p in paths if p.is_file()} == expected,
                 "Exact 64 execution files required")
    return expected


def actor_counts(dialogs):
    """Public layout work, independent of gold availability or scored labels."""
    t = [len(d["turns"]) for d in dialogs]
    q = [len({r["query"] for r in d["queries"]}) for d in dialogs]
    return {"real_question_steps": sum(a * b for a, b in zip(t, q, strict=True)),
            "executed_question_steps": sum(t) * max(q), "advance_calls": max(t), "forward_calls": 1}


def authenticate_inputs(run, packet, lexical, plan_sha, completed_sha, root):
    execution_members(run)
    base.bind(run / "plan.json", plan_sha)
    base.bind(run / "completed.json", completed_sha)
    plan, done = base.read(run / "plan.json"), base.read(run / "completed.json")
    base.require(plan["study"] == "dialogue-copy-v2" and done["status"] == "completed", "Wrong/incomplete V2 study")
    base.require(plan["config"] == CONFIG, "Fixed replication recipe differs")
    base.require(plan["implementation_version"] == VERSION and plan["wall_cap_seconds"] == CAP
                 and plan["normalization_tolerance"] == TOLERANCE, "Version/cap/tolerance differs")
    base.require(plan["practical_checks"] == PRACTICAL_CHECKS, "Original thirteen scientific checks changed")
    base.require(set(plan["source_sha256"]) == SOURCES and plan["source_sha256"][OLD_PATH] == OLD_SHA
                 and plan["source_sha256"][old.BASE_PATH] == old.BASE_SHA256
                 and plan["source_sha256"]["scripts/report_dialogue_copy_v2.py"] == base.sha(__file__), "Source closure/identity")
    for name, value in plan["source_sha256"].items():
        base.bind(base.safe(root, name), value)
    base.require(set(plan["runtime"]) == {"python", "torch", "numpy", "platform"}
                 and all(type(v) is str and v for v in plan["runtime"].values()), "Runtime identity absent")
    base.require(done["plan_sha256"] == plan_sha and done["fit_count"] == 15
                 and done["study"] == "dialogue-copy-v2" and done["external_model_api_calls"] == 0
                 and done["normalization_tolerance"] == TOLERANCE and done["resume_authorized"] is False,
                 "Execution completion")
    base.positive(done["wall_seconds"], "whole execution wall time")
    base.require(done["wall_seconds"] <= CAP, "Execution exceeded fixed cap")
    base.require(base.read(run / "started.json") == {"status": "started", "plan_sha256": plan_sha,
                 "runtime": plan["runtime"], "wall_cap_seconds": CAP}, "Started receipt differs")
    base.require(set(plan["paths"]) == {"packet", "lexical", "old_study"}
                 and base.safe(root, plan["paths"]["packet"]).resolve() == packet.resolve()
                 and base.safe(root, plan["paths"]["lexical"]).resolve() == lexical.resolve(), "Input paths differ")
    parents = {}
    for directory, key, payloads in ((packet, "packet_completed_sha256", {
        "encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}),
        (lexical, "lexical_completed_sha256", {"started.json", "index.json", "lexical.npy"})):
        base.clean(directory)
        base.bind(directory / "completed.json", plan[key])
        receipt = base.read(directory / "completed.json")
        base.require(receipt["status"] == "completed" and set(receipt["files"]) == payloads, "Input receipt/membership")
        for name, value in receipt["files"].items():
            base.bind(base.safe(directory, name), value)
        parents[key] = receipt
    lex = parents["lexical_completed_sha256"]
    base.require(lex["packet_completed_sha256"] == plan["packet_completed_sha256"], "Lexical packet differs")
    base.require(lex["source_sha256"] and all(plan["source_sha256"].get(k) == v for k, v in lex["source_sha256"].items()),
                 "Lexical source identity")
    data = base.read(packet / "packet.json")
    expected, counts = base.expected_records(data)
    cohort = parents["packet_completed_sha256"]["cohorts"]["dev"]
    base.require(cohort["queries"] == len(counts) and cohort["dialogues"] == len(data["cohorts"]["dev"]),
                 "Packet cohort counts")
    return plan, done, parents, data, expected, counts


def original_records(plan, root):
    """Bind published original metadata, without reading old weights/predictions."""
    base.require(plan["old_plan_sha256"] == OLD_PLAN and plan["old_completed_sha256"] == OLD_COMPLETED,
                 "Original lineage pins differ")
    path = base.safe(root, plan["paths"]["old_study"])
    base.bind(path / "plan.json", OLD_PLAN)
    base.bind(path / "completed.json", OLD_COMPLETED)
    prior, done = base.read(path / "plan.json"), base.read(path / "completed.json")
    base.require(prior["study"] == "dialogue-copy-v1" and done["status"] == "completed"
                 and done["plan_sha256"] == OLD_PLAN and done["fit_count"] == 15, "Original incomplete")
    for key in ("config", "practical_checks", "runtime", "packet_completed_sha256", "lexical_completed_sha256",
                "loss_counts", "loss_weights", "optimizer_updates_per_fit"):
        base.require(plan[key] == prior[key], "Original recipe/input differs: " + key)
    base.require(set(prior["source_sha256"]) == old.REQUIRED_SOURCES
                 and all(plan["source_sha256"][k] == v for k, v in prior["source_sha256"].items()), "Original sources differ")
    names = [f"{m}-{s}" for s in SEEDS for m in METHODS]
    base.require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "Original fit order")
    result = {}
    for name, record in zip(names, done["fits"], strict=True):
        member = path / "fits" / name / "completed.json"
        base.require(not member.is_symlink() and base.read(member) == record and record["status"] == "completed",
                     "Original fit receipt differs")
        result[name] = record
    return result


def work_counts(dialogs, queries):
    count = actor_counts(dialogs)
    b, t = len(dialogs), count["advance_calls"]
    question_ids = [{r["query"] for r in d["queries"]} for d in dialogs]
    q = max(map(len, question_ids))
    c = max(len(queries[i]["candidate_ids"]) for ids in question_ids for i in ids)
    return {"real_turns": sum(len(d["turns"]) for d in dialogs), "padded_turn_positions": b*t,
            "padded_query_positions": b*t*q, "padded_candidate_positions": b*t*q*c,
            "real_question_steps": count["real_question_steps"]}


def empty_invariants():
    return {**dict.fromkeys(COUNT_KEYS, 0), **dict.fromkeys(MAX_KEYS, 0.),
            "mass_min": None, "mass_max": None, "tolerance": TOLERANCE}


def merge_invariants(total, batch):
    for key in COUNT_KEYS:
        total[key] += batch[key]
    for key in MAX_KEYS:
        total[key] = max(total[key], batch[key])
    if batch["mass_checks"]:
        total["mass_min"] = batch["mass_min"] if total["mass_min"] is None else min(total["mass_min"], batch["mass_min"])
        total["mass_max"] = batch["mass_max"] if total["mass_max"] is None else max(total["mass_max"], batch["mass_max"])


def validate_invariants(record, dialogs, method):
    expected = actor_counts(dialogs)
    n = expected["executed_question_steps"]
    transport = method in ("scalar", "selective", "selective_no_lexical")
    required = {"forward_calls": 1, "forward_returned": 1, "advance_calls": expected["advance_calls"],
        "advance_returned": expected["advance_calls"], "valid_turns": sum(len(d["turns"]) for d in dialogs),
        "executed_valid_question_slots": n, "real_question_updates": expected["real_question_steps"],
        "incoming_checks": n, "feature_checks": n, "result_checks": n, "mass_checks": n if transport else 0}
    base.require(set(record) == set(empty_invariants()) and record["tolerance"] == TOLERANCE, "Invariant schema/tolerance")
    base.require(all(type(record[k]) is int and record[k] == v for k, v in required.items()), "Invariant public-update coverage")
    base.require(type(record["mass_above_one_count"]) is int
                 and 0 <= record["mass_above_one_count"] <= record["mass_checks"], "Mass overshoot count")
    base.require(all(type(record[k]) in (float, int) and math.isfinite(record[k])
                     and 0 <= record[k] <= TOLERANCE for k in MAX_KEYS), "Invariant tolerance violated")
    if transport:
        lo, hi = record["mass_min"], record["mass_max"]
        base.require(all(type(v) in (float, int) and math.isfinite(v) for v in (lo, hi))
                     and 0 <= lo <= hi <= 1 + TOLERANCE, "Released mass range")
        base.require(record["mass_max_overshoot"] == max(0., hi-1)
                     and (record["mass_above_one_count"] > 0) == (hi > 1), "Mass overshoot inconsistency")
    else:
        base.require(record["mass_min"] is None and record["mass_max"] is None
                     and record["mass_max_overshoot"] == record["mass_above_one_count"] == 0, "Nontransport mass checks")


def audit_batches(ledger, dialogs, queries, method, *, training):
    """Validate complete permutations or ordered development batches, no RNG replay."""
    steps = math.ceil(len(dialogs)/CONFIG["batch_size"])
    epochs = CONFIG["epochs"] if training else 1
    base.require(len(ledger) == epochs*steps, "Batch ledger length")
    aggregate, shapes, orders, losses = empty_invariants(), Counter(), [], []
    for epoch in range(epochs):
        order, weighted, scored = [], 0., 0
        for j, record in enumerate(ledger[epoch*steps:(epoch+1)*steps]):
            ids = record["indices"]
            size = min(CONFIG["batch_size"], len(dialogs)-j*CONFIG["batch_size"])
            base.require(type(ids) is list and len(ids) == size and all(type(i) is int and 0 <= i < len(dialogs) for i in ids),
                         "Batch indices")
            if not training:
                base.require(ids == list(range(j*CONFIG["batch_size"], j*CONFIG["batch_size"]+size)), "Evaluation batch order")
            subset = [dialogs[i] for i in ids]
            work = work_counts(subset, queries)
            base.require(record["actor_shapes"] == work, "Actor work differs from public layout")
            validate_invariants(record["invariants"], subset, method)
            merge_invariants(aggregate, record["invariants"])
            shapes.update(work)
            order += ids
            if training:
                n = sum(len(d["queries"]) for d in subset)
                base.require(record["epoch"] == epoch and record["update"] == epoch*steps+j+1
                             and record["supervised_queries"] == n, "Training cursor/label count")
                loss = record["loss"]
                base.require(type(loss) in (float, int) and math.isfinite(loss) and loss >= 0, "Invalid training loss")
                weighted += loss*n
                scored += n
        base.require(sorted(order) == list(range(len(dialogs))), "Epoch is not a complete permutation")
        orders.append(order)
        if training:
            losses.append(weighted/scored)
    return aggregate, dict(shapes), orders, losses


def audit_saved(run, packet, lexical, plan_sha, completed_sha, root=ROOT):
    plan, done, parents, data, expected, counts = authenticate_inputs(run, packet, lexical, plan_sha, completed_sha, root)
    originals = original_records(plan, root)
    payloads = execution_members(run) - {"completed.json"}
    base.require(set(done["files"]) == payloads, "Completion manifest membership")
    files = {"completed.json": completed_sha}
    for name, entry in done["files"].items():
        path = base.safe(run, name)
        base.require(set(entry) == {"sha256", "bytes"} and type(entry["bytes"]) is int
                     and path.stat().st_size == entry["bytes"], "Manifest size/schema")
        base.bind(path, entry["sha256"])
        files[name] = entry["sha256"]
    train, dev = data["cohorts"]["train"], data["cohorts"]["dev"]
    loss_counts = Counter("0" if r["bin"] == "unmentioned_retention" else "1" if r["bin"] == "assigned_retention" else "2"
                          for d in train for r in d["queries"])
    base.require(set(loss_counts) == {"0", "1", "2"} and plan["loss_counts"] == dict(loss_counts), "Training stratum counts")
    weight = [sum(loss_counts.values())/(3*loss_counts[str(i)]) for i in range(3)]
    updates = math.ceil(len(train)/CONFIG["batch_size"])*CONFIG["epochs"]
    base.require(plan["loss_weights"] == weight and plan["optimizer_updates_per_fit"] == updates, "Loss weights/update plan")
    names = [f"{m}-{s}" for s in SEEDS for m in METHODS]
    base.require([f"{r['method']}-{r['seed']}" for r in done["fits"]] == names, "Fit closure/order")
    rows, paired_initial, paired_common, paired_orders = {}, {}, {}, {}
    train_total, eval_total = empty_invariants(), empty_invariants()
    for name, record in zip(names, done["fits"], strict=True):
        folder = run / "fits" / name
        base.require(base.read(folder / "completed.json") == record and record["status"] == "completed", "Fit receipt differs")
        method, seed = record["method"], record["seed"]
        base.require(record["epochs"] == CONFIG["epochs"] and record["updates"] == updates
                     and record["training_queries"] == sum(loss_counts.values())*CONFIG["epochs"], "Training completion counts")
        for file, key in (("weights.pt", "weights_sha256"), ("dev-predictions.npz", "predictions_sha256"), ("batches.jsonl", "batches_sha256")):
            base.require(files[f"fits/{name}/{file}"] == record[key], "Fit member binding")
        original = originals[name]
        for key in ("initial_tensors_sha256", "original_initial_tensors_sha256", "common_initial_tensors_sha256"):
            base.require(type(record[key]) is str and re.fullmatch(r"[0-9a-f]{64}", record[key]), "Invalid initialization digest")
        base.require(record["initial_tensors_sha256"] == record["original_initial_tensors_sha256"] == original["initial_tensors_sha256"],
                     "Original initialization differs")
        base.require(paired_common.setdefault(seed, record["common_initial_tensors_sha256"]) == record["common_initial_tensors_sha256"],
                     "Common initialization unpaired")
        if method != "candidate_gru":
            base.require(record["common_initial_tensors_sha256"] == record["initial_tensors_sha256"], "Non-GRU common tensors differ")
        if method in ("scalar", "selective", "selective_no_lexical"):
            base.require(paired_initial.setdefault(seed, record["initial_tensors_sha256"]) == record["initial_tensors_sha256"], "Copy initialization unpaired")
        configuration = {**original["configuration"], "class": "MonitoredCopyMemoryV2", "implementation_version": VERSION,
            "normalization": "log_b -= logsumexp(log_b) before features/transition and after each real update",
            "padding": "exact original state values; no normalization committed on padded turns",
            "parameter_schema": "same as DialogueCopyMemory for the same method and dimensions",
            "old_fit_scope": "parameter loading possible, but not corrected training or empirical efficacy evidence"}
        base.require(record["configuration"] == configuration and record["parameters"] == original["parameters"]
                     and record["parameters"] == (103411 if method == "candidate_gru" else 99458), "Configuration/parameter identity")
        base.positive(record["parameters_with_final_gradient"], "gradient parameter count", integer=True)
        base.require(record["parameters_with_final_gradient"] <= record["parameters"], "Invalid gradient parameter count")
        ledger = [base.json.loads(line) for line in (folder / "batches.jsonl").read_text().splitlines()]
        inv, shapes, orders, losses = audit_batches(ledger, train, data["queries"], method, training=True)
        base.require(record["training_invariants"] == inv and record["training_actor_shapes"] == shapes
                     and record["training_losses"] == losses, "Training aggregate differs from ledger")
        base.require(paired_orders.setdefault(seed, orders) == orders, "Epoch orders unpaired")
        evaluation = record["evaluation"]
        einv, eshapes, _, _ = audit_batches(evaluation["batches"], dev, data["queries"], method, training=False)
        base.require(evaluation["queries"] == len(counts) and evaluation["invariants"] == einv
                     and evaluation["actor_shapes"] == eshapes, "Evaluation aggregate/coverage")
        merge_invariants(train_total, inv)
        merge_invariants(eval_total, einv)
        for value in (record["train_wall_seconds"], record["fit_wall_seconds_before_receipt"], evaluation["wall_seconds"]):
            base.positive(value, "fit timing")
        base.require(record["train_wall_seconds"]+evaluation["wall_seconds"] <= record["fit_wall_seconds_before_receipt"], "Fit timing scopes overlap")
        with np.load(folder / "dev-predictions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        base.validate_predictions(arrays, expected, counts)
        rows[name] = {**record, "metrics": base.metrics(arrays), "eval_wall_seconds": evaluation["wall_seconds"]}
        # Keep aggregates and all seed points, not the large per-batch copied ledger.
        rows[name]["evaluation"] = {k: v for k, v in evaluation.items() if k != "batches"}
    reference = done["references"]
    base.require(reference["file"] == "references.npz" and reference["queries"] == len(counts)
                 and reference["sha256"] == files["references.npz"], "Reference binding/coverage")
    base.positive(reference["wall_seconds"], "reference timing")
    base.require(math.fsum(r["fit_wall_seconds_before_receipt"] for r in rows.values()) + reference["wall_seconds"]
                 <= done["wall_seconds"], "Whole execution timing smaller than disjoint work")
    with np.load(run / "references.npz", allow_pickle=False) as saved:
        references = base.reference_metrics({k: saved[k] for k in saved.files}, expected, counts)
    technical = {"passed": True, "exact_execution_files": 64, "training": train_total, "evaluation": eval_total,
        "scope": "Technical validity authenticates every saved batch's raw incoming, feature-prior, result and released-mass "
                 "witness and full public-update coverage. State arithmetic, initialization tensor digests and optimizer "
                 "execution remain source/test-bound; no checkpoint deserialization or neural replay."}
    summary = {"study": "dialogue-copy-v2", "status": "completed", "development_only": True,
        "method_order": list(METHODS), "seeds": list(SEEDS), "rows": rows, "families": old.family_metrics(rows),
        "references": references, "continuation_gate": old.criteria(rows, references), "technical_validity": technical,
        "source_sha256": plan["source_sha256"], "execution_runtime": plan["runtime"],
        "execution_wall_seconds": done["wall_seconds"], "reference_wall_seconds": reference["wall_seconds"],
        "optimizer_updates": sum(r["updates"] for r in rows.values()),
        "supervised_presentations": sum(r["training_queries"] for r in rows.values()),
        "encoder": {k: parents["packet_completed_sha256"][k] for k in ("wall_seconds", "unique_texts", "encoder_sequences", "encoder_tokens_with_special")},
        "lexical_preparation_source_bound": {k: parents["lexical_completed_sha256"].get(k) for k in ("wall_seconds", "counts", "float32_scalars", "bytes")},
        "scope": "Corrected fresh replication on exposed development. Official test untouched according to authenticated "
                 "preparation lineage. No new architecture, full-DST or untouched-confirmation claim.",
        "zero_probability_policy": "No floor; null NLL with positive zero-target count represents infinite NLL.",
        "timing_scope": "Whole execution includes authentication, monitored training/evaluation, hashes and I/O up to "
                        "the terminal receipt. Fit times include checkpoint I/O; evaluation includes assembly and prediction "
                        "storage. Shared encoder/lexical preparation is separate. These are instrumented batch costs, "
                        "not incremental serving latency."}
    return summary, files, plan, done, parents


def historical_deltas(families, historical):
    base.require(historical["study"] == "dialogue-copy-v1" and historical["status"] == "completed",
                 "Historical summary is not completed V1")
    deltas = {}
    for method in METHODS:
        deltas[method] = {}
        for panel in PANELS:
            a, b = families[method][panel], historical["families"][method][panel]
            values = {}
            for label, path in (("macro_accuracy", ("macro_three", "accuracy")), ("micro_nll", ("micro", "nll")),
                                ("brier", ("micro", "brier")), ("revision_accuracy", ("revision", "accuracy"))):
                new, prior = a[path[0]][path[1]], b[path[0]][path[1]]
                values[label] = {"v1": prior, "v2": new,
                                 "v2_minus_v1": None if prior is None or new is None else new - prior}
            deltas[method][panel] = values
    return {"scope": "Descriptive fresh-fit replication deltas, outside the continuation rule. V1 scalar normalization failed; "
                     "these are not evidence for a novel architecture or an isolated causal effect of normalization.",
            "methods": deltas}


def report(run, packet, lexical, plan_sha256, completed_sha256, out, *, root=ROOT,
           historical_summary=None, historical_summary_sha256=None):
    run, packet, lexical, out = map(Path, (run, packet, lexical, out))
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        base.write(out / "started.json", {"study": "dialogue-copy-v2", "plan_sha256": plan_sha256,
            "completed_sha256": completed_sha256, "reporter_sha256": base.sha(__file__),
            "reused_reporting_source_sha256": OLD_SHA})
        summary, files, plan, done, parents = audit_saved(run, packet, lexical, plan_sha256, completed_sha256, root)
        if historical_summary is not None or historical_summary_sha256 is not None:
            base.require(historical_summary is not None and historical_summary_sha256 is not None, "Both historical path and pin required")
            base.bind(Path(historical_summary), historical_summary_sha256)
            summary["historical_v1_descriptive"] = {"sha256": historical_summary_sha256,
                **historical_deltas(summary["families"], base.read(historical_summary))}
        base.write(out / "summary.json", summary)
        (out / "report.md").write_text(report_text(summary))
        old.render(summary, out)
        for name, value in plan["source_sha256"].items():
            base.bind(base.safe(root, name), value)
        for name, value in files.items():
            base.bind(base.safe(run, name), value)
        execution_members(run)
        for directory, key in ((packet, "packet_completed_sha256"), (lexical, "lexical_completed_sha256")):
            base.bind(directory / "completed.json", plan[key])
            for name, value in parents[key]["files"].items():
                base.bind(base.safe(directory, name), value)
        if historical_summary is not None:
            base.bind(Path(historical_summary), historical_summary_sha256)
        receipt = {"study": "dialogue-copy-v2", "status": "completed", "plan_sha256": plan_sha256,
            "execution_completed_sha256": completed_sha256, "execution_members": files,
            "source_sha256": plan["source_sha256"], "reporter_sha256": base.sha(__file__),
            "reused_reporting_source_sha256": OLD_SHA, "reused_metric_source_sha256": old.BASE_SHA256,
            "feature_packet_completed_sha256": plan["packet_completed_sha256"],
            "lexical_completed_sha256": plan["lexical_completed_sha256"],
            "technical_validity_passed": summary["technical_validity"]["passed"],
            "continuation_passed": summary["continuation_gate"]["passed"],
            "checks_passed": summary["continuation_gate"]["checks_passed"], "checks_total": 13,
            "fit_count": 15, "neural_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0,
            "execution_wall_seconds": done["wall_seconds"], "wall_seconds": time.perf_counter() - start,
            "historical_summary_sha256": historical_summary_sha256,
            "files": {n: {"sha256": base.sha(out / n), "bytes": (out / n).stat().st_size}
                      for n in ("started.json", "summary.json", "report.md", "comparison.png")}}
        base.write(out / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        try:
            base.write(out / "failed.json", {"status": "failed", "error": str(error), "error_type": type(error).__name__,
                                             "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - preserve the actual audit exception
            error.add_note("Failure receipt: " + repr(secondary))
        raise


def report_text(summary):
    gate = summary["continuation_gate"]
    lines = ["# Normalized dialogue-copy replication", "", ("Technical validity: **PASS**. "
             f"Original scientific continuation: **{'PASS' if gate['passed'] else 'FAIL'}** ({gate['checks_passed']}/13)."), "",
             ("Fresh final fits after an explicit state-normalization correction. Exposed development only; "
             "official test untouched. This numerical correction is not architectural novelty."), "",
             "| Method | Parameters | Training seconds, 3 fits | Evaluation seconds, 3 fits |",
             "|---|---:|---:|---:|"]
    for method in METHODS:
        rows = [summary["rows"][f"{method}-{seed}"] for seed in SEEDS]
        lines.append(f"| {method} | {rows[0]['parameters']} | {math.fsum(r['train_wall_seconds'] for r in rows):.3f} | "
                     f"{math.fsum(r['eval_wall_seconds'] for r in rows):.3f} |")
    def display(value, scale=1):
        return "undefined/infinite" if value is None else f"{value * scale:.4f}"
    for panel in ("seen", "unseen"):
        lines += ["", f"## {panel.title()} services", "",
            "| Method | Three-stratum macro (%) | Micro NLL | Brier | Revision (%) | Revision queries per fit |",
            "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            r = summary["families"][method][panel]
            lines.append(f"| {method} | {display(r['macro_three']['accuracy'], 100)} | {display(r['micro']['nll'])} | "
                f"{display(r['micro']['brier'])} | {display(r['revision']['accuracy'], 100)} | {r['revision']['count_per_fit']} |")
        for method, panels in summary["references"].items():
            r = panels[panel]
            lines.append(f"| Reference: {method} | {display(r['macro_three']['accuracy'], 100)} | not probabilistic | "
                f"not probabilistic | {display(r['revision']['accuracy'], 100)} | {r['revision']['count']} |")
    lines += ["", summary["technical_validity"]["scope"], "", summary["timing_scope"], "", summary["scope"]]
    if "historical_v1_descriptive" in summary:
        lines += ["", summary["historical_v1_descriptive"]["scope"], "Descriptive deltas are retained in summary.json."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "packet", "lexical", "plan-sha256", "completed-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--historical-summary")
    parser.add_argument("--historical-summary-sha256")
    args = parser.parse_args()
    result = report(args.run, args.packet, args.lexical, args.plan_sha256, args.completed_sha256, args.out,
                    historical_summary=args.historical_summary, historical_summary_sha256=args.historical_summary_sha256)
    print(base.json.dumps({"status": result["status"], "technical_validity_passed": result["technical_validity_passed"],
                          "continuation_passed": result["continuation_passed"]}))
