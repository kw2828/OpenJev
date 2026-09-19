"""Independent saved-only numerical cross-check of the normalized replication.

No production reporter, training module, model, encoder, RNG or checkpoint loader
is imported. Neural state witnesses remain recorded, source-bound observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PLAN_SHA = "9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905"
OLD_PLAN = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
OLD_COMPLETED = "8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300"
METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
SEEDS = (4101, 4102, 4103)
PANELS = ("all", "seen", "unseen")
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
FIELDS = {"labels", "bin", "unseen", "dialogue", "time", "query"}
SOURCES = {
    "scripts/study_dialogue_copy.py", "scripts/prepare_dialogue_copy.py", "scripts/report_dialogue_copy.py",
    "src/openjev/research/dialogue_copy_features.py", "src/openjev/research/dialogue_copy_memory.py",
    "tests/test_dialogue_copy_features.py", "tests/test_dialogue_copy_memory.py", "tests/test_study_dialogue_copy.py",
    "tests/test_report_dialogue_copy.py", "research/dialogue-copy-protocol.md", "scripts/study_dialogue_memory.py",
    "scripts/report_dialogue_memory.py", "src/openjev/research/dialogue_carry.py",
    "src/openjev/research/dialogue_copy_memory_v2.py", "tests/test_dialogue_copy_memory_v2.py",
    "scripts/study_dialogue_copy_v2.py", "tests/test_study_dialogue_copy_v2.py",
    "scripts/report_dialogue_copy_v2.py", "tests/test_report_dialogue_copy_v2.py", "research/dialogue-copy-v2-protocol.md",
}
COUNTERS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
    "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks", "result_checks",
    "mass_checks", "mass_above_one_count")
MAXIMA = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")
TOL = 2e-6
SCORE_COMPARISON_ATOL = 2e-12  # Independent NumPy log/reductions versus producer scalar math.


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "Duplicate JSON key")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe(root, name):
    require(type(name) is str and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe relative path")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "Path escapes root")
    return path


def bind(path, digest, inventory):
    require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid digest")
    require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Hash mismatch: " + str(path))
    inventory[str(path.resolve())] = digest


def match(actual, expected, path="value"):
    """Exact identities/counts; tiny numerical comparison allowance, never gate slack."""
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), path + " keys differ")
        for key, value in expected.items():
            match(actual[key], value, path + "/" + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), path + " list differs")
        for i, value in enumerate(expected):
            match(actual[i], value, path + "/" + str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=SCORE_COMPARISON_ATOL, abs_tol=SCORE_COMPARISON_ATOL),
                path + " number differs")
    else:
        require(type(actual) is type(expected) and actual == expected, path + " differs")


def cohort(packet):
    fields = {k: [] for k in FIELDS}
    counts, identities, dialogues = [], set(), set()
    for d in packet["cohorts"]["dev"]:
        require(d["id"] not in dialogues and d["turns"] and d["queries"], "Duplicate/empty dialogue")
        dialogues.add(d["id"])
        previous = {}
        for row in d["queries"]:
            qi, ti, label = row["query"], row["time"], row["label"]
            require(type(qi) is int and 0 <= qi < len(packet["queries"])
                    and type(ti) is int and 0 <= ti < len(d["turns"]), "Invalid query clock")
            q = packet["queries"][qi]
            ids = q["candidate_ids"]
            require(q["split"] == "dev" and 3 <= len(ids) <= 12 and len(ids) == len(set(ids))
                    and ids[:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"]
                    and all(type(x) is str and x.startswith("value:") for x in ids[2:])
                    and len(q["candidates"]) == len(ids), "Candidate schema")
            require(type(label) is int and 0 <= label < len(ids) and type(row["unseen"]) is bool
                    and type(row["dontcare"]) is bool and row["dontcare"] == (label == 1), "Label schema")
            old_t, old_y = previous.get(qi, (-1, 0))
            kind = ("unmentioned_retention" if old_y == label == 0 else "assigned_retention" if old_y == label
                    else "first_assignment" if old_y == 0 else "clear" if label == 0 else "revision")
            require(ti > old_t and row["bin"] == kind, "Transition label or causal order")
            previous[qi] = (ti, label)
            identity = (d["id"], ti, qi)
            require(identity not in identities, "Repeated query")
            identities.add(identity)
            for k, v in {"labels": label, "bin": kind, "unseen": row["unseen"],
                         "dialogue": d["id"], "time": ti, "query": qi}.items():
                fields[k].append(v)
            counts.append(len(ids))
    return {k: np.asarray(v) for k, v in fields.items()}, np.asarray(counts)


def load_predictions(path, expected, counts, *, reference=False):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    extras = {"pred_none", "pred_literal"} if reference else {"probabilities", "choice"}
    require(set(arrays) == FIELDS | extras, "Array field closure")
    n = len(counts)
    for k, v in expected.items():
        kind = "b" if k == "unseen" else "iu" if k in ("labels", "time", "query") else "U"
        require(arrays[k].shape == (n,) and arrays[k].dtype.kind in kind and np.array_equal(arrays[k], v),
                "Saved cohort differs: " + k)
    if reference:
        for k in extras:
            require(arrays[k].shape == (n,) and arrays[k].dtype.kind in "iu"
                    and ((arrays[k] >= 0) & (arrays[k] < counts)).all(), "Invalid reference")
        require((arrays["pred_none"] == 0).all() and (arrays["pred_literal"] != 1).all(), "Reference reserved values")
    else:
        p = arrays["probabilities"]
        require(p.shape == (n, 12) and p.dtype.kind == "f" and np.isfinite(p).all()
                and ((p >= 0) & (p <= 1)).all(), "Invalid probability")
        require((p[np.arange(12)[None] >= counts[:, None]] == 0).all(), "Nonzero padding")
        require(np.all(np.abs(p.astype(np.float64).sum(1) - 1) <= 1e-5), "Probability mass")
        require(arrays["choice"].shape == (n,) and arrays["choice"].dtype.kind in "iu"
                and np.array_equal(arrays["choice"], p.argmax(1)), "Choice mismatch")
    return arrays


def average(values):
    return None if not values or any(v is None for v in values) else math.fsum(values) / len(values)


def metrics(arrays, choice=None):
    y = arrays["labels"]
    p = None if choice is not None else arrays["probabilities"].astype(np.float64)
    chosen = choice if choice is not None else p.argmax(1)
    def score(mask):
        n = int(mask.sum())
        correct = int(np.count_nonzero((chosen == y) & mask))
        result = {"count": n, "correct": correct, "accuracy": correct / n if n else None}
        if p is not None:
            target = p[np.flatnonzero(mask), y[mask]]
            zeros = int(np.count_nonzero(target == 0))
            # Independent NumPy log and one-hot vector difference, no floor.
            residual = p[mask] - np.eye(12, dtype=np.float64)[y[mask]]
            result.update(nll=float(np.log(target).sum(dtype=np.float64) / -n) if n and not zeros else None,
                          brier=float(np.square(residual).sum(dtype=np.float64) / n) if n else None,
                          zero_target_probabilities=zeros)
        return result
    output = {}
    for panel in PANELS:
        selected = np.ones(len(y), bool) if panel == "all" else arrays["unseen"] if panel == "unseen" else ~arrays["unseen"]
        bins = {b: score(selected & (arrays["bin"] == b)) for b in BINS}
        strata = {k: bins[k] for k in STRATA[:2]}
        strata["changed"] = score(selected & np.isin(arrays["bin"], BINS[2:]))
        output[panel] = {"micro": score(selected), "bins": bins, "strata": strata,
            "macro_three": {k: average([strata[s][k] for s in STRATA])
                            for k in (("accuracy",) if p is None else ("accuracy", "nll", "brier"))},
            "revision": bins["revision"]}
        if p is not None:
            output[panel].update(not_mentioned_count=int(np.count_nonzero(selected & (y == 0))),
                                 dontcare_count=int(np.count_nonzero(selected & (y == 1))))
    return output


def families(rows):
    def pool(items):
        require(len({x["count"] for x in items}) == 1, "Unpaired cohort")
        return {"count_per_fit": items[0]["count"], "fit_count": 3,
                **{k: average([x[k] for x in items]) for k in ("accuracy", "nll", "brier")},
                "zero_target_probabilities_all_fits": sum(x["zero_target_probabilities"] for x in items)}
    output = {}
    for method in METHODS:
        output[method] = {}
        for panel in PANELS:
            m = [rows[f"{method}-{s}"][panel] for s in SEEDS]
            output[method][panel] = {"micro": pool([x["micro"] for x in m]),
                "macro_three": {k: average([x["macro_three"][k] for x in m]) for k in ("accuracy", "nll", "brier")},
                "bins": {b: pool([x["bins"][b] for x in m]) for b in BINS},
                "revision": pool([x["revision"] for x in m])}
    return output


def exact_macro(m):
    if any(m["strata"][s]["count"] == 0 for s in STRATA):
        return None
    return sum((Fraction(m["strata"][s]["correct"], m["strata"][s]["count"]) for s in STRATA), Fraction()) / 3


def gate(rows, refs):
    checks = [{"name": "all_15_fits_completed", "passed": set(rows) == {f"{m}-{s}" for m in METHODS for s in SEEDS}}]
    for panel in ("seen", "unseen"):
        paired = {m: [exact_macro(rows[f"{m}-{s}"][panel]) for s in SEEDS] for m in METHODS}
        means = {m: None if any(x is None for x in xs) else sum(xs, Fraction()) / 3 for m, xs in paired.items()}
        literal = refs["literal"][panel]
        lm = exact_macro(literal)
        def add(name, a, b, *, nll=False, panel=panel, **extra):
            checks.append({"name": panel + "/" + name,
                "passed": a is not None and b is not None and bool(a <= b if nll else a >= b),
                "left": None if a is None else float(a), "right": None if b is None else float(b), **extra})
        add("macro_gain_over_literal_3pp", means["selective"], None if lm is None else lm + Fraction(3, 100))
        add("macro_gain_over_scalar_half_pp", means["selective"], None if means["scalar"] is None else means["scalar"] + Fraction(1, 200))
        add("micro_nll_nonworse_scalar", average([rows[f"selective-{s}"][panel]["micro"]["nll"] for s in SEEDS]),
            average([rows[f"scalar-{s}"][panel]["micro"]["nll"] for s in SEEDS]), nll=True)
        wins = None if any(v is None for v in paired["selective"] + paired["scalar"]) else sum(
            a > b for a, b in zip(paired["selective"], paired["scalar"], strict=True))
        add("strict_macro_win_at_least_two_pairs", wins, 2)
        strongest = None if means["readout"] is None or means["candidate_gru"] is None else max(
            ("readout", "candidate_gru"), key=lambda m: means[m])
        add("macro_nonworse_best_conventional", means["selective"], None if strongest is None else means[strongest], strongest=strongest)
        rev = [rows[f"selective-{s}"][panel]["revision"] for s in SEEDS]
        rm = None if any(r["count"] == 0 for r in rev) else sum((Fraction(r["correct"], r["count"]) for r in rev), Fraction()) / 3
        lr = literal["revision"]
        add("revision_within_1pp_literal", rm, None if lr["count"] == 0 else Fraction(lr["correct"], lr["count"]) - Fraction(1, 100))
    return {"passed": all(x["passed"] for x in checks), "checks_passed": sum(x["passed"] for x in checks),
            "checks_total": 13, "checks": checks}


def empty_witness():
    return {**dict.fromkeys(COUNTERS, 0), **dict.fromkeys(MAXIMA, 0.),
            "mass_min": None, "mass_max": None, "tolerance": TOL}


def merge(total, item):
    for k in COUNTERS:
        total[k] += item[k]
    for k in MAXIMA:
        total[k] = max(total[k], item[k])
    if item["mass_checks"]:
        total["mass_min"] = item["mass_min"] if total["mass_min"] is None else min(total["mass_min"], item["mass_min"])
        total["mass_max"] = item["mass_max"] if total["mass_max"] is None else max(total["mass_max"], item["mass_max"])


def audit_ledger(ledger, dialogs, queries, method, training):
    epochs, per_epoch = (20 if training else 1), math.ceil(len(dialogs) / 32)
    require(len(ledger) == epochs * per_epoch, "Incomplete batch ledger")
    aggregate, shapes, losses, orders = empty_witness(), Counter(), [], []
    for epoch in range(epochs):
        order, numerator, denominator = [], 0., 0
        for j in range(per_epoch):
            record = ledger[epoch * per_epoch + j]
            ids = record["indices"]
            require(type(ids) is list and len(ids) == min(32, len(dialogs) - j * 32)
                    and all(type(i) is int and 0 <= i < len(dialogs) for i in ids), "Invalid batch indices")
            if not training:
                require(ids == list(range(j * 32, min((j + 1) * 32, len(dialogs)))), "Development order")
            ds = [dialogs[i] for i in ids]
            ts = [len(d["turns"]) for d in ds]
            qs = [{r["query"] for r in d["queries"]} for d in ds]
            b, t, q = len(ds), max(ts), max(map(len, qs))
            c = max(len(queries[qi]["candidate_ids"]) for group in qs for qi in group)
            real = sum(n * len(group) for n, group in zip(ts, qs, strict=True))
            n = sum(ts) * q
            work = {"real_turns": sum(ts), "padded_turn_positions": b*t, "padded_query_positions": b*t*q,
                    "padded_candidate_positions": b*t*q*c, "real_question_steps": real}
            require(record["actor_shapes"] == work, "Public layout work mismatch")
            transport = method in ("scalar", "selective", "selective_no_lexical")
            witness = record["invariants"]
            expected = {"forward_calls": 1, "forward_returned": 1, "advance_calls": t, "advance_returned": t,
                "valid_turns": sum(ts), "executed_valid_question_slots": n, "real_question_updates": real,
                "incoming_checks": n, "feature_checks": n, "result_checks": n, "mass_checks": n if transport else 0}
            require(set(witness) == set(empty_witness()) and witness["tolerance"] == TOL, "Witness schema")
            require(all(type(witness[k]) is int and witness[k] == v for k, v in expected.items()), "Witness coverage")
            require(all(type(witness[k]) in (int, float) and math.isfinite(witness[k]) and 0 <= witness[k] <= TOL
                        for k in MAXIMA), "Normalization tolerance")
            count = witness["mass_above_one_count"]
            require(type(count) is int and 0 <= count <= witness["mass_checks"], "Mass overshoot count")
            if transport:
                lo, hi = witness["mass_min"], witness["mass_max"]
                require(type(lo) in (int, float) and type(hi) in (int, float)
                        and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= 1 + TOL, "Mass range")
                require(witness["mass_max_overshoot"] == max(0., hi - 1) and (count > 0) == (hi > 1), "Mass consistency")
            else:
                require(witness["mass_min"] is witness["mass_max"] is None
                        and count == witness["mass_max_overshoot"] == 0, "Unexpected mass checks")
            merge(aggregate, witness)
            shapes.update(work)
            order.extend(ids)
            if training:
                scored = sum(len(d["queries"]) for d in ds)
                require(record["epoch"] == epoch and record["update"] == epoch * per_epoch + j + 1
                        and record["supervised_queries"] == scored, "Update/label coverage")
                require(type(record["loss"]) in (float, int) and math.isfinite(record["loss"]) and record["loss"] >= 0,
                        "Invalid loss")
                numerator += record["loss"] * scored
                denominator += scored
        require(sorted(order) == list(range(len(dialogs))), "Incomplete/duplicated epoch")
        orders.append(order)
        if training:
            losses.append(numerator / denominator)
    return aggregate, dict(shapes), losses, orders


def audit(args):
    out, run, report = Path(args.out), Path(args.run), Path(args.report)
    out.mkdir(parents=True, exist_ok=False)
    start, inventory, progress = time.perf_counter(), {}, {"stage": "authentication", "fits_checked": []}
    try:
        write(out / "started.json", {"status": "started", "source_sha256": sha(__file__),
            "plan_sha256": args.plan_sha256, "execution_completed_sha256": args.completed_sha256,
            "report_receipt_sha256": args.report_receipt_sha256, "scope": "Saved-only independent arithmetic"})
        require(args.plan_sha256 == PLAN_SHA, "Different frozen study")
        bind(run / "plan.json", args.plan_sha256, inventory)
        bind(run / "completed.json", args.completed_sha256, inventory)
        bind(report / "receipt.json", args.report_receipt_sha256, inventory)
        plan, done, receipt = read(run / "plan.json"), read(run / "completed.json"), read(report / "receipt.json")
        require(plan["study"] == done["study"] == receipt["study"] == "dialogue-copy-v2"
                and done["status"] == receipt["status"] == "completed", "Incomplete study/report")
        require(done["plan_sha256"] == receipt["plan_sha256"] == args.plan_sha256
                and receipt["execution_completed_sha256"] == args.completed_sha256, "Completion/report binding")
        require(set(plan["source_sha256"]) == SOURCES and receipt["source_sha256"] == plan["source_sha256"], "Source closure")
        for name, digest in plan["source_sha256"].items():
            bind(safe(ROOT, name), digest, inventory)
        require(done["fit_count"] == 15 and done["external_model_api_calls"] == 0 and done["resume_authorized"] is False
                and done["normalization_tolerance"] == TOL and 0 < done["wall_seconds"] <= 3600, "Execution bounds")
        names = [f"{m}-{s}" for s in SEEDS for m in METHODS]
        members = {"plan.json", "started.json", "completed.json", "references.npz"} | {
            f"fits/{name}/{file}" for name in names for file in ("weights.pt", "completed.json", "dev-predictions.npz", "batches.jsonl")}
        paths = list(run.rglob("*"))
        require(not any(p.is_symlink() for p in paths) and {p.relative_to(run).as_posix() for p in paths if p.is_file()} == members,
                "Exact 64 execution members")
        require(set(done["files"]) == members - {"completed.json"}, "Completion manifest closure")
        for name, entry in done["files"].items():
            path = safe(run, name)
            require(set(entry) == {"bytes", "sha256"} and type(entry["bytes"]) is int and path.stat().st_size == entry["bytes"], "Payload size")
            bind(path, entry["sha256"], inventory)
        require(read(run / "started.json") == {"status": "started", "plan_sha256": args.plan_sha256,
            "runtime": plan["runtime"], "wall_cap_seconds": 3600.}, "Starting execution identity")
        require(set(receipt["files"]) == {"started.json", "summary.json", "report.md", "comparison.png"}, "Report payload closure")
        for name, entry in receipt["files"].items():
            path = safe(report, name)
            require(path.stat().st_size == entry["bytes"], "Report payload size")
            bind(path, entry["sha256"], inventory)
        require(not any((report / p).exists() for p in ("failed.json", "late-completion.json", "cleanup-error.json")), "Failed report marker")
        summary = read(report / "summary.json")
        require(summary["source_sha256"] == plan["source_sha256"] and summary["execution_runtime"] == plan["runtime"], "Summary identity")
        for label in ("packet", "lexical"):
            directory = safe(ROOT, plan["paths"][label])
            bind(directory / "completed.json", plan[label + "_completed_sha256"], inventory)
            source = read(directory / "completed.json")
            require(source["status"] == "completed", "Input completion")
            for name, digest in source["files"].items():
                bind(safe(directory, name), digest, inventory)
        packet = read(safe(ROOT, plan["paths"]["packet"]) / "packet.json")
        expected, counts = cohort(packet)
        original = safe(ROOT, plan["paths"]["old_study"])
        require(plan["old_plan_sha256"] == OLD_PLAN and plan["old_completed_sha256"] == OLD_COMPLETED, "Original lineage pins")
        bind(original / "plan.json", OLD_PLAN, inventory)
        bind(original / "completed.json", OLD_COMPLETED, inventory)
        prior_plan, prior_done = read(original / "plan.json"), read(original / "completed.json")
        for key in ("config", "practical_checks", "runtime", "loss_weights", "loss_counts", "optimizer_updates_per_fit",
                    "packet_completed_sha256", "lexical_completed_sha256"):
            require(plan[key] == prior_plan[key], "Original recipe mismatch: " + key)
        require([f"{x['method']}-{x['seed']}" for x in done["fits"]] == names
                and [f"{x['method']}-{x['seed']}" for x in prior_done["fits"]] == names, "Fit identities/order")
        rows, initials, orders, common, initial_receipts = {}, {}, {}, {}, {}
        totals = {"training": empty_witness(), "evaluation": empty_witness()}
        progress["stage"] = "per_fit_predictions_and_witnesses"
        for name, fit, prior in zip(names, done["fits"], prior_done["fits"], strict=True):
            folder, method, seed = run / "fits" / name, fit["method"], fit["seed"]
            require(read(folder / "completed.json") == fit and fit["status"] == "completed", "Per-fit completion")
            prior_path = original / "fits" / name / "completed.json"
            require(read(prior_path) == prior, "Original initializer receipt differs")
            initial_receipts[name] = {"receipt_sha256": sha(prior_path), "initial_tensors_sha256": prior["initial_tensors_sha256"]}
            bind(prior_path, initial_receipts[name]["receipt_sha256"], inventory)
            require(fit["initial_tensors_sha256"] == fit["original_initial_tensors_sha256"] == prior["initial_tensors_sha256"], "Initial tensors differ")
            require(common.setdefault(seed, fit["common_initial_tensors_sha256"]) == fit["common_initial_tensors_sha256"], "Common init mismatch")
            if method in ("scalar", "selective", "selective_no_lexical"):
                require(initials.setdefault(seed, fit["initial_tensors_sha256"]) == fit["initial_tensors_sha256"], "Unpaired transport initialization")
            for file, key in (("weights.pt", "weights_sha256"), ("dev-predictions.npz", "predictions_sha256"), ("batches.jsonl", "batches_sha256")):
                require(done["files"][f"fits/{name}/{file}"]["sha256"] == fit[key], "Fit file binding")
            ledger = [json.loads(line) for line in (folder / "batches.jsonl").read_text().splitlines()]
            inv, shapes, losses, order = audit_ledger(ledger, packet["cohorts"]["train"], packet["queries"], method, True)
            require(inv == fit["training_invariants"] and shapes == fit["training_actor_shapes"] and losses == fit["training_losses"], "Training aggregate mismatch")
            require(orders.setdefault(seed, order) == order, "Paired training order mismatch")
            updates = math.ceil(len(packet["cohorts"]["train"]) / 32) * 20
            presentations = sum(len(d["queries"]) for d in packet["cohorts"]["train"]) * 20
            require(fit["epochs"] == 20 and fit["updates"] == updates and fit["training_queries"] == presentations, "Fit work budget")
            merge(totals["training"], inv)
            evaluation = fit["evaluation"]
            inv, shapes, _, _ = audit_ledger(evaluation["batches"], packet["cohorts"]["dev"], packet["queries"], method, False)
            require(evaluation["queries"] == len(counts) and inv == evaluation["invariants"] and shapes == evaluation["actor_shapes"], "Evaluation aggregate mismatch")
            merge(totals["evaluation"], inv)
            arrays = load_predictions(folder / "dev-predictions.npz", expected, counts)
            rows[name] = metrics(arrays)
            match(summary["rows"][name]["metrics"], rows[name], name)
            progress["fits_checked"].append(name)
        refs_saved = load_predictions(run / "references.npz", expected, counts, reference=True)
        require(done["references"]["file"] == "references.npz" and done["references"]["queries"] == len(counts)
                and done["references"]["sha256"] == done["files"]["references.npz"]["sha256"], "Reference receipt binding")
        refs = {name: metrics(refs_saved, refs_saved["pred_" + name]) for name in ("none", "literal")}
        match(summary["references"], refs, "references")
        family = families(rows)
        match(summary["families"], family, "families")
        calculated = gate(rows, refs)
        reported = summary["continuation_gate"]
        for key in ("passed", "checks_passed", "checks_total"):
            match(reported[key], calculated[key], "gate/" + key)
        require(len(reported["checks"]) == 13, "Gate check count")
        for actual, wanted in zip(reported["checks"], calculated["checks"], strict=True):
            match({k: actual[k] for k in wanted}, wanted, "gate/" + wanted["name"])
        technical = summary["technical_validity"]
        require(technical["passed"] is True and technical["exact_execution_files"] == 64
                and technical["training"] == totals["training"] and technical["evaluation"] == totals["evaluation"], "Reported witness aggregate mismatch")
        require(receipt["technical_validity_passed"] is True and receipt["continuation_passed"] == calculated["passed"]
                and receipt["checks_passed"] == calculated["checks_passed"] and receipt["checks_total"] == 13, "Report receipt gate mismatch")
        progress["stage"] = "exit_reauthentication"
        for path, digest in tuple(inventory.items()):
            bind(Path(path), digest, inventory)
        result = {"status": "passed", "study": "dialogue-copy-v2", "rows": rows, "families": family,
            "references": refs, "continuation_gate": calculated, "recorded_invariant_totals": totals,
            "original_initialization_receipts": initial_receipts, "plan_sha256": args.plan_sha256,
            "execution_completed_sha256": args.completed_sha256, "report_receipt_sha256": args.report_receipt_sha256,
            "authenticated_files": inventory, "auditor_source_sha256": sha(__file__), "wall_seconds": time.perf_counter() - start,
            "runtime": {"python": platform.python_version(), "numpy": str(np.__version__)},
            "metric_comparison_absolute_relative_tolerance": SCORE_COMPARISON_ATOL,
            "gate_margin_tolerance": 0, "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0,
            "scope": "Independently implemented NumPy saved-probability metrics and Fraction accuracy criteria. "
                     "Neural execution, initialization tensor generation, lexical reference generation and internal "
                     "state witnesses remain authenticated source-bound records, not independent neural replay. "
                     "Audit pass means agreement and completeness, not scientific continuation success."}
        write(out / "summary.json", result)
        write(out / "completed.json", {"status": "passed", "summary_sha256": sha(out / "summary.json"),
            "started_sha256": sha(out / "started.json"), "auditor_source_sha256": sha(__file__),
            "fit_count": 15, "execution_files": 64, "source_files": 20, "checks_passed": calculated["checks_passed"],
            "continuation_passed": calculated["passed"], "wall_seconds": time.perf_counter() - start})
        return result
    except BaseException as error:
        try:
            write(out / "failed.json", {"status": "failed", "error": str(error), "type": type(error).__name__,
                "traceback": traceback.format_exc(), "progress": progress, "authenticated_files": inventory,
                "wall_seconds": time.perf_counter() - start})
        except BaseException as secondary:  # noqa: BLE001 - original mismatch remains primary
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation also failed: " + repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "report", "completed-sha256", "report-receipt-sha256", "out"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--plan-sha256", default=PLAN_SHA)
    result = audit(parser.parse_args())
    print(json.dumps({"status": result["status"], "gate": result["continuation_gate"]["checks_passed"]}))
