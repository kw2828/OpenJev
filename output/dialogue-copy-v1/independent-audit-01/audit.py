"""Independent saved-array arithmetic; never imports a study or model module."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "83f0002004f15299e5ed0715bcd91f29b65869fa"
PLAN = "609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b"
METHODS = ("readout", "scalar", "selective", "selective_no_lexical", "candidate_gru")
SEEDS = (4101, 4102, 4103)
PANELS = ("all", "seen", "unseen")
BINS = ("unmentioned_retention", "assigned_retention", "first_assignment", "revision", "clear")
STRATA = ("unmentioned_retention", "assigned_retention", "changed")
SOURCES = {
    "scripts/study_dialogue_copy.py", "scripts/prepare_dialogue_copy.py", "scripts/report_dialogue_copy.py",
    "src/openjev/research/dialogue_copy_features.py", "src/openjev/research/dialogue_copy_memory.py",
    "tests/test_dialogue_copy_features.py", "tests/test_dialogue_copy_memory.py", "tests/test_study_dialogue_copy.py",
    "tests/test_report_dialogue_copy.py", "research/dialogue-copy-protocol.md", "scripts/study_dialogue_memory.py",
    "scripts/report_dialogue_memory.py", "src/openjev/research/dialogue_carry.py",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "Duplicate JSON key")
            obj[key] = value
        return obj

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, allow_nan=False, sort_keys=True, indent=2)
        f.write("\n")


def bind(path, expected):
    require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected), "Invalid digest")
    require(not Path(path).is_symlink() and digest(path) == expected, "Hash mismatch: " + str(path))


def member(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Unsafe member")
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "Escaping member")
    return path


def closure(root, names):
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Symlink in artifact tree")
    require({p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(names),
            "Artifact membership differs: " + str(root))


def number(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (not positive or value > 0), "Invalid nonnegative quantity")
    return value


def average(values):
    return None if not values or any(v is None for v in values) else float(np.mean(values, dtype=np.float64))


def compare(expected, actual, path="root"):
    """Compare recomputed fields; numeric tolerance never changes gate decisions."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) <= set(actual), path + " keys")
        return max((compare(v, actual[k], path + "/" + k) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + " list")
        return max((compare(a, b, path + f"/{i}") for i, (a, b) in enumerate(zip(expected, actual, strict=True))),
                   default=0.)
    if expected is None or type(expected) in (str, bool, int):
        require(type(actual) is type(expected) and actual == expected, path + " exact value")
        return 0.
    require(type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(float(expected), actual, rel_tol=1e-12, abs_tol=1e-12), path + " arithmetic")
    return abs(float(expected) - actual)


def packet_rows(packet):
    fields = {k: [] for k in ("labels", "bin", "unseen", "dialogue", "time", "query")}
    counts, identities, dialogues = [], set(), set()
    for d in packet["cohorts"]["dev"]:
        require(d["id"] not in dialogues, "Duplicate dialogue")
        dialogues.add(d["id"])
        prior = {}
        for row in d["queries"]:
            q, t, y = (row[k] for k in ("query", "time", "label"))
            require(type(q) is type(t) is type(y) is int and 0 <= q < len(packet["queries"])
                    and 0 <= t < len(d["turns"]), "Invalid packet index")
            catalog = packet["queries"][q]
            ids = catalog["candidate_ids"]
            require(catalog["split"] == "dev" and 3 <= len(ids) <= 12 and len(set(ids)) == len(ids)
                    and ids[:2] == ["reserved:NOT_MENTIONED", "reserved:DONTCARE"]
                    and all(s.startswith("value:") for s in ids[2:])
                    and len(ids) == len(catalog["candidates"]) and 0 <= y < len(ids), "Candidate authority")
            last_t, last_y = prior.get(q, (-1, 0))
            require(t > last_t and (d["id"], t, q) not in identities, "Duplicate/noncausal query")
            if last_y == y:
                label_bin = "unmentioned_retention" if y == 0 else "assigned_retention"
            else:
                label_bin = "first_assignment" if last_y == 0 else "clear" if y == 0 else "revision"
            require(row["bin"] == label_bin and type(row["unseen"]) is bool
                    and type(row["dontcare"]) is bool and row["dontcare"] == (y == 1), "Annotation mismatch")
            prior[q] = t, y
            identities.add((d["id"], t, q))
            values = (y, label_bin, row["unseen"], d["id"], t, q)
            for key, value in zip(fields, values, strict=True):
                fields[key].append(value)
            counts.append(len(ids))
    return {k: np.asarray(v) for k, v in fields.items()}, np.asarray(counts, dtype=np.int64)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_arrays(arrays, truth, sizes, *, reference=False):
    extra = {"pred_none", "pred_literal"} if reference else {"probabilities", "choice"}
    require(set(arrays) == set(truth) | extra, "Prediction field set")
    for key, expected in truth.items():
        a = arrays[key]
        kind = "iu" if key in ("labels", "time", "query") else "b" if key == "unseen" else "U"
        require(a.dtype.kind in kind and a.shape == expected.shape and np.array_equal(a, expected),
                "Wrong saved record: " + key)
    if reference:
        for key in extra:
            a = arrays[key]
            require(a.shape == sizes.shape and a.dtype.kind in "iu" and ((a >= 0) & (a < sizes)).all(),
                    "Invalid reference")
        require((arrays["pred_none"] == 0).all() and (arrays["pred_literal"] != 1).all(), "Reference sentinel")
        return
    p = arrays["probabilities"]
    require(p.dtype == np.float32 and p.shape == (len(sizes), 12) and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities")
    padding = np.arange(12)[None, :] >= sizes[:, None]
    require((p[padding] == 0).all() and (np.abs(p.astype(np.float64).sum(1) - 1) <= 1e-5).all(),
            "Candidate mask or row mass")
    require(arrays["choice"].dtype.kind in "iu" and arrays["choice"].shape == sizes.shape
            and np.array_equal(arrays["choice"], np.argmax(p, axis=1)), "Wrong argmax")


def metrics(truth, choices, probabilities=None):
    """Vectorized proper scores, direct stored probabilities without renormalizing."""
    labels = truth["labels"]
    correct = choices == labels
    if probabilities is not None:
        p = probabilities.astype(np.float64)
        target = p[np.arange(len(labels)), labels]
        zero = target == 0
        logloss = np.zeros(len(labels), dtype=np.float64)
        np.log(target, out=logloss, where=~zero)
        logloss *= -1
        # Algebraically sum_c (p_c - one_hot(y)_c)^2; no one-hot helper.
        brier = (p * p).sum(1) - 2 * target + 1

    def reduce(mask):
        n = int(mask.sum())
        hits = int(correct[mask].sum())
        result = {"count": n, "correct": hits, "accuracy": hits / n if n else None}
        if probabilities is not None:
            z = int(zero[mask].sum())
            result.update(nll=float(logloss[mask].mean()) if n and not z else None,
                          brier=float(brier[mask].mean()) if n else None, zero_target_probabilities=z)
        return result

    result = {}
    for panel in PANELS:
        mask = np.ones(len(labels), bool) if panel == "all" else truth["unseen"] == (panel == "unseen")
        bins = {b: reduce(mask & (truth["bin"] == b)) for b in BINS}
        strata = {b: bins[b] for b in STRATA[:2]}
        strata["changed"] = reduce(mask & np.isin(truth["bin"], BINS[2:]))
        keys = ("accuracy", "nll", "brier") if probabilities is not None else ("accuracy",)
        value = {"micro": reduce(mask), "bins": bins, "strata": strata,
                 "macro_three": {k: average([strata[s][k] for s in STRATA]) for k in keys},
                 "revision": bins["revision"]}
        if probabilities is not None:
            value.update(not_mentioned_count=int((mask & (labels == 0)).sum()),
                         dontcare_count=int((mask & (labels == 1)).sum()))
        result[panel] = value
    return result


def families(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for panel in PANELS:
            fits = [rows[f"{method}-{s}"][panel] for s in SEEDS]

            def pool(items):
                require(len({v["count"] for v in items}) == 1, "Unequal paired support")
                return {"count_per_fit": items[0]["count"], "fit_count": 3,
                        **{k: average([v[k] for v in items]) for k in ("accuracy", "nll", "brier")},
                        "zero_target_probabilities_all_fits": sum(v["zero_target_probabilities"] for v in items)}

            result[method][panel] = {"micro": pool([v["micro"] for v in fits]),
                "macro_three": {k: average([v["macro_three"][k] for v in fits]) for k in ("accuracy", "nll", "brier")},
                "revision": pool([v["revision"] for v in fits]),
                "bins": {b: pool([v["bins"][b] for v in fits]) for b in BINS}}
    return result


def checks(rows, references):
    def macro(value):
        parts = [value["strata"][s] for s in STRATA]
        return None if any(p["count"] == 0 for p in parts) else sum(
            (Fraction(p["correct"], p["count"]) for p in parts), Fraction()) / 3

    def rational_mean(values):
        return None if any(v is None for v in values) else sum(values, Fraction()) / len(values)

    output = [{"name": "all_15_fits_completed", "passed": set(rows) == {
        f"{m}-{s}" for m in METHODS for s in SEEDS}}]
    for panel in ("seen", "unseen"):
        values = {m: [macro(rows[f"{m}-{s}"][panel]) for s in SEEDS] for m in METHODS}
        means = {m: rational_mean(v) for m, v in values.items()}
        literal = references["literal"][panel]
        literal_macro = macro(literal)
        ps, ss = values["selective"], values["scalar"]
        nll = {m: average([rows[f"{m}-{s}"][panel]["micro"]["nll"] for s in SEEDS])
               for m in ("selective", "scalar")}
        win = None if None in ps + ss else sum(a > b for a, b in zip(ps, ss, strict=True))
        conventional = ("readout", "candidate_gru")
        best = None if any(means[m] is None for m in conventional) else max(conventional, key=lambda m: means[m])
        revisions = [rows[f"selective-{s}"][panel]["revision"] for s in SEEDS]
        revision = rational_mean([Fraction(v["correct"], v["count"]) if v["count"] else None for v in revisions])
        lr = literal["revision"]
        lr = Fraction(lr["correct"], lr["count"]) if lr["count"] else None
        tests = [
            ("macro_gain_over_literal_3pp", means["selective"], None if literal_macro is None else literal_macro + Fraction(3, 100), "ge"),
            ("macro_gain_over_scalar_half_pp", means["selective"], None if means["scalar"] is None else means["scalar"] + Fraction(1, 200), "ge"),
            ("micro_nll_nonworse_scalar", nll["selective"], nll["scalar"], "le"),
            ("strict_macro_win_at_least_two_pairs", win, 2, "ge"),
            ("macro_nonworse_best_conventional", means["selective"], None if best is None else means[best], "ge"),
            ("revision_within_1pp_literal", revision, None if lr is None else lr - Fraction(1, 100), "ge"),
        ]
        for name, left, right, operation in tests:
            ok = left is not None and right is not None and (left >= right if operation == "ge" else left <= right)
            output.append({"name": panel + "/" + name, "passed": bool(ok), "comparison": operation,
                           "left": None if left is None else float(left), "right": None if right is None else float(right)})
    require(len(output) == 13, "Gate cardinality")
    return {"passed": all(v["passed"] for v in output), "checks_passed": sum(v["passed"] for v in output),
            "checks_total": 13, "panel_checks_total": 12, "checks": output}


def run(args):
    started = time.perf_counter()
    run_dir, packet_dir, lexical_dir, report_dir = map(Path, (args.run, args.packet, args.lexical, args.report))
    out = Path(__file__).parent
    require(not any((out / n).exists() for n in ("result.json", "receipt.json", "failed.json")), "Prior audit output exists")
    pins = {run_dir / "plan.json": PLAN, run_dir / "completed.json": args.completed_sha256,
            report_dir / "summary.json": args.summary_sha256, report_dir / "receipt.json": args.receipt_sha256}
    for path, value in pins.items():
        bind(path, value)
    plan, completed, summary, receipt = [read(p) for p in pins]
    require(plan["study"] == summary["study"] == receipt["study"] == "dialogue-copy-v1", "Foreign study")
    require(completed["status"] == summary["status"] == receipt["status"] == "completed", "Incomplete output")
    require(completed["plan_sha256"] == receipt["plan_sha256"] == PLAN
            and receipt["execution_completed_sha256"] == args.completed_sha256, "Report lineage")
    require(set(plan["source_sha256"]) == SOURCES, "Exact source closure")
    for name, value in plan["source_sha256"].items():
        path = member(ROOT, name)
        bind(path, value)
        blob = subprocess.run(["git", "show", COMMIT + ":" + name], cwd=ROOT, check=True, capture_output=True).stdout
        require(hashlib.sha256(blob).hexdigest() == value, "Prospective Git source differs: " + name)
        pins[path] = value
    for key in ("source_sha256", "execution_runtime"):
        require(summary[key] == (plan["source_sha256"] if key == "source_sha256" else plan["runtime"]), "Source/runtime attribution")
    report_files = {"started.json", "summary.json", "report.md", "comparison.png"}
    require(set(receipt["files"]) == report_files, "Report payload closure")
    closure(report_dir, report_files | {"receipt.json"})
    for name, item in receipt["files"].items():
        path = member(report_dir, name)
        bind(path, item["sha256"])
        require(path.stat().st_size == item["bytes"], "Report byte count")
        pins[path] = item["sha256"]
    for directory, key, expected_files in (
        (packet_dir, "packet_completed_sha256", {"encoder-plan.json", "encoder-source.py", "features.npy", "packet.json"}),
        (lexical_dir, "lexical_completed_sha256", {"started.json", "index.json", "lexical.npy"}),
    ):
        bind(directory / "completed.json", plan[key])
        pins[directory / "completed.json"] = plan[key]
        parent = read(directory / "completed.json")
        require(parent["status"] == "completed" and set(parent["files"]) == expected_files, "Input receipt")
        closure(directory, expected_files | {"completed.json"})
        for name, value in parent["files"].items():
            bind(member(directory, name), value)
            pins[member(directory, name)] = value
        if directory == lexical_dir:
            require(parent["packet_completed_sha256"] == plan["packet_completed_sha256"], "Lexical packet identity")
            require(all(plan["source_sha256"].get(k) == v for k, v in parent["source_sha256"].items()), "Lexical sources")
    packet = read(packet_dir / "packet.json")
    truth, sizes = packet_rows(packet)
    train = packet["cohorts"]["train"]
    require((len(train), sum(len(d["queries"]) for d in train), len(packet["cohorts"]["dev"]), len(sizes))
            == (2017, 51741, 2363, 62329), "Frozen cohort counts")
    config = plan["config"]
    require(config["methods"] == list(METHODS) and config["seeds"] == list(SEEDS)
            and config["epochs"] == 20 and config["batch_size"] == 32, "Frozen fit schedule")
    names = [f"{m}-{s}" for s in SEEDS for m in METHODS]
    members = {"plan.json": PLAN, "completed.json": args.completed_sha256}
    expected_members = set(members) | {"references.npz"} | {
        f"fits/{name}/{file}" for name in names for file in ("completed.json", "weights.pt", "dev-predictions.npz")}
    closure(run_dir, expected_members)
    require(completed["fit_count"] == 15 and completed["external_model_api_calls"] == 0
            and [f"{v['method']}-{v['seed']}" for v in completed["fits"]] == names, "Fit coverage/order")
    actual, initial, times, max_error = {}, {}, [], 0.
    require(set(summary["rows"]) == set(names), "Reported fit membership")
    for name, record in zip(names, completed["fits"], strict=True):
        folder = run_dir / "fits" / name
        own = read(folder / "completed.json")
        require(own == record and own["status"] == "completed", "Different fit receipt")
        require((own["epochs"], own["updates"], own["training_queries"], len(own["training_losses"]))
                == (20, 1280, 1034820, 20), "Incomplete training")
        for loss in own["training_losses"]:
            number(loss)
        cfg = own["configuration"]
        require(cfg["class"] == "DialogueCopyMemory" and cfg["method"] == own["method"]
                and cfg["parameters"] == own["parameters"], "Model identity")
        require(all(cfg[k] == config[k] for k in ("projection_dim", "hidden_dim", "gru_width")), "Model dimensions")
        require(0 < own["parameters_with_final_gradient"] <= own["parameters"], "Active parameter count")
        init = own["initial_tensors_sha256"]
        require(type(init) is str and re.fullmatch("[0-9a-f]{64}", init), "Initialization digest")
        if own["method"] in ("scalar", "selective", "selective_no_lexical"):
            require(initial.setdefault(own["seed"], init) == init, "Unpaired initial digest")
        shapes = own["training_actor_shapes"]
        real_turns = sum(len(d["turns"]) for d in train) * 20
        real_questions = sum(len(d["turns"]) * len({q["query"] for q in d["queries"]}) for d in train) * 20
        require(shapes["real_turns"] == real_turns and shapes["real_question_steps"] == real_questions
                and shapes["padded_turn_positions"] >= real_turns and shapes["padded_query_positions"] >= real_questions
                and shapes["padded_candidate_positions"] >= shapes["padded_query_positions"], "Work counts")
        require(own["evaluation"]["queries"] == len(sizes), "Evaluation count")
        times.append(number(own["train_wall_seconds"], positive=True) + number(own["evaluation"]["wall_seconds"], positive=True))
        members[f"fits/{name}/completed.json"] = digest(folder / "completed.json")
        for file, key in (("weights.pt", "weights_sha256"), ("dev-predictions.npz", "predictions_sha256")):
            bind(folder / file, own[key])
            members[f"fits/{name}/{file}"] = own[key]
        a = load_arrays(folder / "dev-predictions.npz")
        validate_arrays(a, truth, sizes)
        actual[name] = metrics(truth, a["choice"], a["probabilities"])
        max_error = max(max_error, compare(actual[name], summary["rows"][name]["metrics"], name))
        for key in ("method", "seed", "parameters", "parameters_with_final_gradient", "updates", "training_queries",
                    "training_actor_shapes", "initial_tensors_sha256", "train_wall_seconds", "predictions_sha256", "weights_sha256"):
            require(summary["rows"][name][key] == own[key], "Reported fit metadata differs")
    ref = completed["references"]
    require(ref["file"] == "references.npz" and ref["queries"] == len(sizes), "Reference coverage")
    members["references.npz"] = ref["sha256"]
    bind(run_dir / "references.npz", ref["sha256"])
    a = load_arrays(run_dir / "references.npz")
    validate_arrays(a, truth, sizes, reference=True)
    references = {name: metrics(truth, a["pred_" + name]) for name in ("none", "literal")}
    pooled, gate = families(actual), checks(actual, references)
    for value, reported, label in ((references, summary["references"], "references"),
                                    (pooled, summary["families"], "families"),
                                    (gate, summary["continuation_gate"], "gate")):
        max_error = max(max_error, compare(value, reported, label))
    require(receipt["execution_members"] == members and len(members) == 48, "Report execution seal")
    require(receipt["continuation_passed"] == gate["passed"] and receipt["checks_passed"] == gate["checks_passed"]
            and receipt["checks_total"] == 13 and receipt["neural_calls"] == receipt["encoder_calls"] == 0, "Report outcome")
    paid_seconds = math.fsum(times) + number(ref["wall_seconds"], positive=True)
    require(paid_seconds <= number(completed["wall_seconds"], positive=True) + 1e-6, "Nested cost exceeds total")
    for name, value in members.items():
        pins[run_dir / name] = value
    for path, value in pins.items():
        bind(path, value)
    closure(run_dir, expected_members)
    result = {"status": "completed", "study": "dialogue-copy-v1", "independent_arithmetic": True,
              "gate": gate, "families": pooled, "references": references, "fits": actual,
              "counts": {"fits": 15, "prediction_rows": len(sizes) * 15, "queries_per_fit": len(sizes),
                         "updates": 19200, "supervised_presentations": 15522300, "execution_files": 48,
                         "sources": 13, "paired_initialization_digests": initial},
              "maximum_absolute_report_arithmetic_difference": max_error,
              "costs": {"execution_seconds": completed["wall_seconds"], "fit_plus_eval_plus_reference_seconds": paid_seconds},
              "limits": ["Development data already exposed; no untouched-test or architectural novelty claim.",
                         "No model, encoder, optimizer, random generator or environment calls; no reporter metric imports.",
                         "Weights remain opaque bytes; initial digest pairing is recorded provenance, not tensor replay.",
                         "Public extraction, literal decisions and learned computations remain pinned-source/test bound.",
                         "No label or prediction arrays copied into this aggregate audit output."]}
    write(out / "result.json", result)
    write(out / "receipt.json", {"status": "completed", "source_sha256": digest(__file__),
        "prospective_commit": COMMIT, "plan_sha256": PLAN, "execution_completed_sha256": args.completed_sha256,
        "summary_sha256": args.summary_sha256, "report_receipt_sha256": args.receipt_sha256,
        "input_sha256": {str(p): h for p, h in pins.items()}, "result_sha256": digest(out / "result.json"),
        "wall_seconds": time.perf_counter() - started, "gate_passed": gate["passed"],
        "checks_passed": gate["checks_passed"], "checks_total": 13, "new_neural_calls": 0})
    return {"status": "completed", "gate": gate, "maximum_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "packet", "lexical", "report", "completed-sha256", "summary-sha256", "receipt-sha256"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args)))
    except BaseException as error:
        failure = Path(__file__).with_name("failed.json")
        if not failure.exists() and not Path(__file__).with_name("receipt.json").exists():
            try:
                write(failure, {"status": "failed", "type": type(error).__name__, "error": str(error)})
            except BaseException as preservation_error:  # noqa: BLE001 - preserve original audit failure
                error.add_note("Failure receipt could not be written: " + repr(preservation_error))
        raise
