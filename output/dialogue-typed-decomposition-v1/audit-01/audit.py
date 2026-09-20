"""Independent saved branch/value arithmetic; no project or model imports.

This source is prepared before the diagnostic outputs are authorized for reading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("flat_stratum", "flat_balanced", "typed_stratum", "typed_balanced")
SEEDS = (6101, 6102, 6103)
FITS = {f"{arm}-{seed}" for arm in ARMS for seed in SEEDS}
RUN_SHA = "65bd57084dbc7f7b99b782e140bae2e18f25b9e237e89a0132cead5fbfe2a095"
ORIGINAL_SUMMARY_SHA = "3969bcb7803d3f625a90a9e165a775bb9108f8301b9f225d198eed2e9e84a81c"
PRIOR_AUDIT_SUMMARY_SHA = "dded7a075ff56ba5d4b16cdab74d6a559dd5c6e9b49c1c1cde8410e467d2bd2a"
PAIRS = {"typing_balanced": ("flat_balanced", "typed_balanced"),
         "typing_stratum": ("flat_stratum", "typed_stratum"),
         "weighting_flat": ("flat_stratum", "flat_balanced"),
         "weighting_typed": ("typed_stratum", "typed_balanced")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(text):
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, "Duplicate JSON key")
            result[k] = v
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON number: "+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text())


def close(actual, expected, name):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), "Fields: "+name)
        for k, v in expected.items(): close(actual[k], v, name+"/"+k)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "List: "+name)
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)): close(a, b, name+"/"+str(i))
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Arithmetic: "+name)
    else:
        require(type(actual) is type(expected) and actual == expected, "Identity: "+name)


def components(rows, log_probs, row_indices):
    require(row_indices.dtype == np.int64 and row_indices.tolist() == [r["row_index"] for r in rows], "Exact saved row order")
    require(log_probs.dtype == np.float32 and log_probs.shape == (len(rows), 12), "Float32 prediction shape")
    logs = log_probs.astype(np.float64)
    choice = logs.argmax(1)
    total, branch, value, selected_branch, target_branch = [], [], [], [], []
    for i, row in enumerate(rows):
        count = row["candidate_count"]
        kinds = row["candidate_types"]
        target = row["current_label_index"]
        require(type(target) is int and 0 <= target < count and len(kinds) == count and 3 <= count <= 12, "Canonical supported target")
        require(kinds.count(0) == kinds.count(1) == 1 and all(type(t) is int and 0 <= t <= 4 for t in kinds), "Three-branch public support")
        supported = logs[i, :count]
        require(np.isfinite(supported).all() and np.all(supported <= 0) and np.isneginf(logs[i, count:]).all(), "Supported/masked saved logs")
        require(abs(math.fsum(float(x) for x in np.exp(supported))-1) <= 2e-6, "Unrepaired raw probability mass")
        target_kind = min(kinds[target], 2)
        same = [j for j, kind in enumerate(kinds) if min(kind, 2) == target_kind]
        maximum = max(float(supported[j]) for j in same)
        log_branch = maximum+math.log(math.fsum(math.exp(float(supported[j])-maximum) for j in same))
        log_target = float(supported[target])
        total.append(-log_target)
        branch.append(-log_branch)
        value.append(log_branch-log_target)
        selected_branch.append(min(kinds[int(choice[i])], 2))
        target_branch.append(target_kind)
    total, branch, value = (np.asarray(v, dtype=np.float64) for v in (total, branch, value))
    residual = np.abs(total-branch-value)
    require(np.all(residual <= 1e-10), "Per-row NLL decomposition identity")
    target = np.asarray([r["current_label_index"] for r in rows])
    correct = choice == target
    wrong_branch = np.asarray(selected_branch) != np.asarray(target_branch)
    within_wrong = ~correct & ~wrong_branch
    require(np.all(correct.astype(int)+wrong_branch.astype(int)+within_wrong.astype(int) == 1), "Selected-candidate error partition")
    return {"total": total, "branch": branch, "value": value, "correct": correct,
            "wrong_branch": wrong_branch, "within_branch_wrong": within_wrong,
            "selected_branch": np.asarray(selected_branch), "target_branch": np.asarray(target_branch),
            "maximum_identity_error": float(residual.max())}


def means(rows, indices, vector):
    if not indices:
        return {"row": None, "equal_service": None}
    services = {}
    for i in indices: services.setdefault(rows[i]["service"], []).append(i)
    return {"row": math.fsum(float(vector[i]) for i in indices)/len(indices),
            "equal_service": math.fsum(math.fsum(float(vector[i]) for i in group)/len(group)
                                       for group in services.values())/len(services)}


def partition(indices, quantities):
    result = {"rows": len(indices), "correct": sum(bool(quantities["correct"][i]) for i in indices),
              "wrong_branch": sum(bool(quantities["wrong_branch"][i]) for i in indices),
              "within_branch_wrong": sum(bool(quantities["within_branch_wrong"][i]) for i in indices)}
    require(result["correct"]+result["wrong_branch"]+result["within_branch_wrong"] == result["rows"], "Group partition")
    return result


def decision_changes(indices, left, right):
    counts = {"both_correct": 0, "correct_to_wrong": 0, "wrong_to_correct": 0, "both_wrong": 0}
    for i in indices:
        a, b = bool(left["correct"][i]), bool(right["correct"][i])
        key = "both_correct" if a and b else "correct_to_wrong" if a else "wrong_to_correct" if b else "both_wrong"
        counts[key] += 1
    delta = sum(int(right["correct"][i])-int(left["correct"][i]) for i in indices)
    require(sum(counts.values()) == len(indices) and counts["wrong_to_correct"]-counts["correct_to_wrong"] == delta,
            "Paired accuracy cancellation")
    return {"rows": len(indices), **counts, "net_correct_change": delta}


def authenticate(args):
    """Inherit the completed study audit; rehash the exact saved study payloads."""
    run = Path(args.run).resolve()
    bindings = {}
    def bind(path, pin, size=None):
        p = Path(path).resolve()
        require(p.is_relative_to(ROOT) and p.is_file(), "Input path boundary")
        require(sha(p) == pin and (size is None or p.stat().st_size == size), "Input digest/size: "+str(p))
        bindings[p.relative_to(ROOT).as_posix()] = {"sha256": pin, "bytes": p.stat().st_size}
    require(args.run_sha256 == RUN_SHA, "Fixed completed study pin")
    bind(run/"completed.json", RUN_SHA)
    done = read(run/"completed.json")
    require(done["status"] == "completed" and set(done["completed_fits"]) == FITS and len(done["completed_fits"]) == 12,
            "All twelve final fits")
    expected = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{seed}.npy" for seed in SEEDS}
    expected |= {f"fits/{name}/{part}" for name in FITS for part in ("completed.json", "weights.pt", "updates.jsonl", "predictions.npz")}
    require(set(done["files"]) == expected and {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()} == expected | {"completed.json"}, "Exact 56-file study closure")
    for name, entry in done["files"].items():
        require(not (run/name).is_symlink(), "No symlinked study payload")
        bind(run/name, entry["sha256"], entry["bytes"])
    require(sha(run/"plan.json") == done["plan_sha256"], "Frozen plan pin")
    original_path = ROOT/"output/dialogue-typed-v1/analysis-02/summary.json"
    prior_path = ROOT/"output/dialogue-typed-v1/independent-audit-01/result-01/summary.json"
    bind(original_path, ORIGINAL_SUMMARY_SHA)
    bind(prior_path, PRIOR_AUDIT_SUMMARY_SHA)
    original, prior = read(original_path), read(prior_path)
    require(original["status"] == prior["status"] == "completed" and original["technical_validity_passed"] is True
            and original["execution_completed_sha256"] == prior["run_completed_sha256"] == RUN_SHA
            and prior["all_scoped_quantities_match"] is True
            and original["continuation_allowed"] is False and prior["continuation"]["passed"] is False, "Inherited failed study and completed independent audit")
    analysis_path = Path(args.analysis).resolve()
    bind(analysis_path, args.analysis_sha256)
    analysis = read(analysis_path)
    require(analysis["status"] == "completed" and analysis["version"] == "dialogue-typed-decomposition-v1"
            and analysis["execution_completed_sha256"] == RUN_SHA
            and analysis["published_summary_sha256"] == ORIGINAL_SUMMARY_SHA
            and analysis["original_continuation_allowed"] is False, "Completed diagnostic and closed study binding")
    require(set(analysis["fits"]) == FITS and set(analysis["pairs"]) == set(PAIRS), "Fixed diagnostic fits/pairs")
    rows = [decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    plan = read(run/"plan.json")
    require([r["row_index"] for r in rows] == plan["evaluation_row_indices"] == sorted(set(plan["evaluation_row_indices"])), "Canonical row identities")
    require(all(r["split"] == "train" and r["admission"] == "admitted" and type(r["heldout_service"]) is bool for r in rows), "Original TRAIN conditional cohort")
    return run, rows, analysis, original, bindings


def analyze(run, rows, published, original):
    indices = [i for i, row in enumerate(rows) if row["heldout_service"] and row["current_label_index"] != row["previous_current_index"]]
    require(indices, "Nonempty fixed held-out changed panel")
    denominator = {"rows": len(indices), "services": len({rows[i]["service"] for i in indices}),
                   "dialogues": len({rows[i]["dialogue_id"] for i in indices}),
                   "schema_queries": len({rows[i]["query_id"] for i in indices})}
    quantities, fitted = {}, {}
    group = "heldout_service/changed"
    for name in sorted(FITS):
        with np.load(run/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"row_indices", "log_probs"}, "Exact prediction fields")
            q = components(rows, archive["log_probs"], archive["row_indices"])
        quantities[name] = q
        loss = {metric: means(rows, indices, q[metric]) for metric in ("total", "branch", "value")}
        counts = partition(indices, q)
        decisions = {"correct": counts["correct"], "wrong_selected_branch": counts["wrong_branch"], "wrong_value": counts["within_branch_wrong"]}
        confusion = [[0]*3 for _ in range(3)]
        for i in indices: confusion[int(q["target_branch"][i])][int(q["selected_branch"][i])] += 1
        cell = published["fits"][name]["groups"][group]
        for field, value in denominator.items(): close(cell[field], value, name+"/"+field)
        for metric, weighted in loss.items():
            for weight, value in weighted.items(): close(cell["loss"][metric][weight], value, name+"/"+metric+"/"+weight)
        for field, value in decisions.items(): close(cell["decisions"][field], value, name+"/"+field)
        close(cell["selected_branch_confusion"], confusion, name+"/confusion")
        for weight in ("row", "equal_service"):
            close(loss["total"][weight], original["fits"][name]["cells"][group]["nll"][weight], name+"/original total/"+weight)
            require(abs(loss["total"][weight]-loss["branch"][weight]-loss["value"][weight]) <= 1e-10,
                    "Aggregate NLL decomposition")
        fitted[name] = {**denominator, "loss": loss, "decisions": decisions, "selected_branch_confusion": confusion,
                        "maximum_all_row_identity_error": q["maximum_identity_error"]}
    paired = {}
    for label, (left_arm, right_arm) in PAIRS.items():
        require(set(published["pairs"][label]) == {str(seed) for seed in SEEDS}, "All paired seeds")
        paired[label] = {}
        for seed in SEEDS:
            left, right = (quantities[f"{arm}-{seed}"] for arm in (left_arm, right_arm))
            changes = decision_changes(indices, left, right)
            correct_counts = {"CC": changes["both_correct"], "CW": changes["correct_to_wrong"],
                              "WC": changes["wrong_to_correct"], "WW": changes["both_wrong"]}
            transition = [[0]*3 for _ in range(3)]
            for i in indices:
                a = 0 if left["correct"][i] else 1 if left["wrong_branch"][i] else 2
                b = 0 if right["correct"][i] else 1 if right["wrong_branch"][i] else 2
                transition[a][b] += 1
            losses = {metric: means(rows, indices, right[metric]-left[metric]) for metric in ("total", "branch", "value")}
            accuracy = changes["net_correct_change"]/len(indices)
            pair = published["pairs"][label][str(seed)]
            require(pair["base"] == f"{left_arm}-{seed}" and pair["candidate"] == f"{right_arm}-{seed}", "Paired contrast orientation")
            cell = pair["groups"][group]
            close(cell["rows"], len(indices), label+"/"+str(seed)+"/rows")
            close(cell["correctness_counts"], correct_counts, label+"/"+str(seed)+"/counts")
            close(cell["error_transitions"], transition, label+"/"+str(seed)+"/transitions")
            close(cell["accuracy_difference"], accuracy, label+"/"+str(seed)+"/accuracy")
            for metric, weighted in losses.items():
                for weight, value in weighted.items(): close(cell["loss_difference"][metric][weight], value, label+"/"+str(seed)+"/"+metric+"/"+weight)
            paired[label][str(seed)] = {"correctness_counts": correct_counts, "error_transitions": transition,
                                       "loss_difference": losses, "accuracy_difference": accuracy}
    return {"fits": fitted, "pairs": paired, "denominators": denominator,
            "all_row_decomposition_identities_checked": len(rows)*len(FITS), "fit_metric_means_checked": 72,
            "paired_metric_means_checked": 72, "fit_partitions_checked": 12, "paired_transition_tables_checked": 12}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); source = sha(__file__)
    try:
        run, rows, published, original, inputs = authenticate(args)
        result = analyze(run, rows, published, original)
        for name, item in inputs.items():
            path = ROOT/name
            require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], "End input stability")
        result.update(status="completed", all_scoped_quantities_match=True, original_continuation_allowed=False,
                      diagnostic_summary_sha256=args.analysis_sha256, execution_completed_sha256=RUN_SHA,
                      scope=["Independent stdlib/NumPy only: twelve fits and four paired comparisons at all three seeds, held-out-service changed panel. Raw saved float32 logs promoted to float64, no repair, no reoptimization.",
                             "Per-row total/branch/within-value identity checked across all evaluation rows; primary row/equal-service means, selected-candidate branch partitions/confusions and paired accuracy cancellation checked. First-index candidate argmax retained.",
                             "Equal-dialogue means, other detailed panels, mass-branch diagnostics, conditional-value correctness and tie counts are outside this narrow independent check.",
                             "Full prior study validity and semantic row admission inherit the authenticated original report and prior independent audit. Current exact 56-file study payloads, plan pin and saved row order rehashed; weights never deserialized.",
                             "Posthoc descriptive evidence on historically exposed TRAIN with privileged previous values; no altered continuation gate, inference, training, RNG, calibration or causal claim."])
        write(out/"summary.json", result)
        write(out/"receipt.json", {"status": "completed", "source_sha256": source,
              "source_path": str(Path(__file__).resolve().relative_to(ROOT)), "execution_completed_sha256": RUN_SHA,
              "diagnostic_summary_sha256": args.analysis_sha256, "authenticated_inputs": inputs,
              "files": {"summary.json": {"sha256": sha(out/"summary.json"), "bytes": (out/"summary.json").stat().st_size}},
              "wall_seconds": time.monotonic()-started, "model_calls": 0, "training_calls": 0, "rng_calls": 0})
        print(json.dumps({"status": "completed", "summary_sha256": sha(out/"summary.json"), "receipt_sha256": sha(out/"receipt.json"), "source_sha256": source}))
    except BaseException as error:
        try:
            write(out/"failed.json", {"status": "failed", "error": repr(error), "source_sha256": source,
                                      "diagnostic_summary_sha256": args.analysis_sha256, "execution_completed_sha256": RUN_SHA,
                                      "wall_seconds": time.monotonic()-started, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve the original audit failure
            if callable(getattr(error, "add_note", None)): error.add_note("Failure preservation error: "+repr(secondary))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--analysis-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
