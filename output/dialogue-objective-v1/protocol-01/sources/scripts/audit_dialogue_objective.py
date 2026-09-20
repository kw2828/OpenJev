"""Independent saved-prediction primary audit for the six-fit objective study.

No producer, runner, model, cache, checkpoint, journal or order imports/loads.
Only arithmetic from the SHA-pinned independent commitment audit is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import signal
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-objective-independent-primary-audit-v1"
HELPER = "scripts/audit_dialogue_commitment.py"
HELPER_PIN = "862368fadf6d2f143b70e3a8a2a48b02418d849defd1a184f9251f881b8184ab"
SEEDS = (6201, 6202, 6203)
METHODS = ("stratum", "uniform")
FIT_ORDER = ("stratum-6201", "uniform-6201", "uniform-6202", "stratum-6202", "stratum-6203", "uniform-6203")
READOUTS = ("stratum-original", "stratum-corrected", "uniform-original", "uniform-reweighted")
COUNTS = (17666, 9246, 2299)
LIMITS = {"wall_seconds": 60, "rss_bytes": 1024**3, "output_bytes": 64*1024**2}
TRAIN_LIMITS = {"wall_seconds": 6000., "rss_bytes": 6*1024**3, "output_bytes": 512*1024**2}
ROWS_PIN = "56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0"
HISTORICAL = "output/dialogue-weight-prior-v1"
HISTORICAL_PINS = {
    "diagnostic-01/summary.json": "97e6388f89020628854b3274b8c6d05f8524ea69e7a31931738840ff1bbbd961",
    "diagnostic-01/receipt.json": "a0b5209b81a81f33bcebb833b3badc8e7fcf4144ceda5eb2f64bd03adeb67975",
    "audit-01/receipt.json": "b1f802b477fddfc3c11bf304588753780536d580ec36037acba04290bc96d771",
}
SCOPE = ("Independent normalized primary metrics for twelve readouts and five held-out-service strata, "
         "both paired readout comparisons, and all thirteen objective/practical decisions. "
         "Scoring reuses the pinned independent commitment arithmetic, while weight transformations and "
         "decision rules are independently implemented. Only selected saved metric payloads and fit "
         "receipts are checked against the complete structural run manifest. No replay of model "
         "configuration, training, caches, checkpoints, journals or orders; those technical validations "
         "are inherited from the authenticated main report. Secondary panels, service tables, descriptive "
         "means and cost accounting are not independently reproduced except service counts needed for gates. "
         "Historical references remain exposed development comparisons; prior FAIL18/22 is unchanged.")


def require(ok, message):
    if not ok: raise ValueError(message)


def load_helper():
    path = ROOT/HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_PIN, "Independent arithmetic source pin")
    spec = importlib.util.spec_from_file_location("_objective_independent_arithmetic", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def shifted(logs, previous, candidate_types, mask, direction):
    """Only public previous/type enters; +1 inverse weighting, -1 reweighting."""
    require(direction in (-1, 1), "Signed correction")
    n = len(logs); rows = np.arange(n)
    require(logs.dtype == np.float64 and logs.shape == mask.shape == candidate_types.shape,
            "Correction support shapes")
    require(previous.shape == (n,) and previous.dtype == np.int64 and np.all(previous >= 0)
            and np.all(previous < logs.shape[1]) and mask[rows, previous].all(), "Supported previous")
    require(np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(), "Correction finite support")
    # Dividing all logits by the same change weight cancels. Only keep odds shift.
    retention = np.where(candidate_types[rows, previous] == 0, COUNTS[0], COUNTS[1])
    delta = direction*np.log(retention.astype(np.float64)/COUNTS[2])
    result = logs.copy(); result[rows, previous] += delta
    result -= np.logaddexp.reduce(result, axis=1)[:, None]
    require(float(np.abs(np.expm1(np.logaddexp.reduce(result, axis=1))).max()) <= 1e-12,
            "Normalized signed correction")
    return result


def mean_tree(items):
    first = items[0]
    if isinstance(first, dict): return {k: mean_tree([v[k] for v in items]) for k in first}
    if first is None:
        require(all(v is None for v in items), "Common undefined support")
        return None
    return math.fsum(items)/len(items)


def exact_rates(rows, meta, choice, mask):
    indices = np.flatnonzero(mask)
    if not len(indices): return {"row": None, "equal_service": None}
    correct = choice == meta["labels"]
    by_service = {}
    for i in indices: by_service.setdefault(rows[i]["service"], []).append(int(correct[i]))
    return {"row": Fraction(int(correct[indices].sum()), len(indices)),
            "equal_service": sum((Fraction(sum(v), len(v)) for v in by_service.values()), Fraction())/len(by_service)}


def historical_rates(fit, services):
    c = fit["cells"]["heldout_service/all"]
    for rec in [c]+[fit["services"][s]["all"] for s in services]:
        require(type(rec["rows"]) is int and rec["rows"] > 0
                and type(rec["counts"]["correct"]) is int
                and 0 <= rec["counts"]["correct"] <= rec["rows"], "Historical exact support")
    return {"row": Fraction(c["counts"]["correct"], c["rows"]),
            "equal_service": sum((Fraction(fit["services"][s]["all"]["counts"]["correct"],
                                            fit["services"][s]["all"]["rows"]) for s in services), Fraction())/len(services)}


def decisions(fits, exact, historical, services):
    """Exact rational accuracy/FP rules, floating NLL comparisons without slack."""
    def frac_mean(values):
        return None if any(v is None for v in values) else sum(values, Fraction())/len(values)
    def acc(category, stratum, weighting):
        return frac_mean([exact[f"{category}-{seed}"][stratum][weighting] for seed in SEEDS])
    def nll_difference(left, right, weighting):
        pairs = [(a["cells"]["heldout_service/all"]["metrics"]["nll"][weighting],
                  b["cells"]["heldout_service/all"]["metrics"]["nll"][weighting])
                 for a, b in zip(left, right, strict=True)]
        if any(a is None or b is None or not math.isfinite(a) or not math.isfinite(b) for a, b in pairs):
            return None
        return math.fsum(a-b for a, b in pairs)/len(pairs)
    def ge(a, b, gap=Fraction()): return a is not None and b is not None and a-b >= gap
    def le(a, b): return a is not None and b is not None and a <= b
    primary, control = "uniform-original", "stratum-corrected"
    objective = {}
    for weight in ("row", "equal_service"):
        objective["changed_accuracy_"+weight] = ge(acc(primary, "changed", weight), acc(control, "changed", weight), Fraction(2, 100))
        objective["retained_error_"+weight] = ge(acc(primary, "retained", weight), acc(control, "retained", weight))
        objective["overall_nll_"+weight] = le(nll_difference([fits[f"{primary}-{s}"] for s in SEEDS],
                                                                  [fits[f"{control}-{s}"] for s in SEEDS], weight), 0.)
    paired = []
    for seed in SEEDS:
        a, b = exact[f"{primary}-{seed}"], exact[f"{control}-{seed}"]
        if all(a["changed"][w] is not None and b["changed"][w] is not None
               and a["changed"][w] > b["changed"][w]
               and ge(a["retained"][w], b["retained"][w]) for w in ("row", "equal_service")):
            paired.append(seed)
    objective["common_paired_seeds"] = len(paired) >= 2
    refs = {seed: historical["fits"][f"flat_stratum-corrected-{seed}"] for seed in SEEDS}
    hacc = {seed: historical_rates(refs[seed], services) for seed in SEEDS}
    practical = {}
    for weight in ("row", "equal_service"):
        practical["overall_accuracy_"+weight] = ge(acc(primary, "all", weight), frac_mean([hacc[s][weight] for s in SEEDS]), Fraction(1, 400))
        practical["overall_nll_"+weight] = le(nll_difference([fits[f"{primary}-{s}"] for s in SEEDS],
                                                                  [refs[s] for s in SEEDS], weight), 0.)
    def fp(fit):
        v = fit["cells"]["heldout_service/all"]["rare"]["true_false_positive"]
        return Fraction(v["numerator"], v["denominator"]) if v["denominator"] else None
    afp = frac_mean([fp(fits[f"{primary}-{s}"]) for s in SEEDS]); bfp = frac_mean([fp(refs[s]) for s in SEEDS])
    practical["true_false_positive"] = afp is not None and bfp is not None and afp-bfp <= Fraction(1, 200)
    practical_seeds = []
    for seed in SEEDS:
        cell = fits[f"{primary}-{seed}"]["cells"]["heldout_service/all"]
        ref = refs[seed]["cells"]["heldout_service/all"]
        if all(exact[f"{primary}-{seed}"]["all"][w] is not None
               and exact[f"{primary}-{seed}"]["all"][w] > hacc[seed][w]
               and le(cell["metrics"]["nll"][w], ref["metrics"]["nll"][w]) for w in ("row", "equal_service")):
            practical_seeds.append(seed)
    practical["common_paired_seeds"] = len(practical_seeds) >= 2
    names = {
        "changed_accuracy_row": "changed_accuracy_row", "changed_accuracy_equal_service": "changed_accuracy_equal_service",
        "retained_error_row": "retained_error_row", "retained_error_equal_service": "retained_error_equal_service",
        "overall_nll_row": "nll_row", "overall_nll_equal_service": "nll_equal_service", "common_paired_seeds": "joint_seeds"}
    checks = {"objective_"+names[k]: {"passed": bool(v)} for k, v in objective.items()}
    pnames = {"overall_accuracy_row": "accuracy_row", "overall_accuracy_equal_service": "accuracy_equal_service",
              "overall_nll_row": "nll_row", "overall_nll_equal_service": "nll_equal_service",
              "true_false_positive": "true_false_positive", "common_paired_seeds": "joint_seeds"}
    checks.update({"practical_"+pnames[k]: {"passed": bool(v)} for k, v in practical.items()})
    return {"objective_passed": all(objective.values()), "practical_passed": all(practical.values()),
            "passed": all(objective.values()) and all(practical.values()),
            "checks_passed": sum(objective.values())+sum(practical.values()), "total_checks": 13, "checks": checks}



def reconstruct(rows, packets, historical, helper):
    require(set(packets) == set(FIT_ORDER), "All six raw fits")
    meta = helper.metadata(rows); masks = helper.primary_masks(meta)
    fits, exact, choices, vectors, normalizations = {}, {}, {}, {}, {}
    for name in FIT_ORDER:
        packet = packets[name]
        require(set(packet) == {"row_indices", "log_probs"} and packet["row_indices"].dtype == np.int64
                and np.array_equal(packet["row_indices"], meta["ids"]), "Prediction exact rows/schema")
        original, z, norm = helper.normalize(packet["log_probs"], meta["mask"])
        norm["nll_normalized_minus_original_raw"] = {"heldout_service/"+key: helper.means(z, rows, mask) for key, mask in masks.items()}
        normalizations[name] = {"raw_maximum_mass_error": norm["maximum_raw_mass_error"],
                                "max_abs_log_z": norm["max_abs_log_z"], "mean_log_z": norm["mean_log_z"],
                                "nll_normalized_minus_original_raw": norm["nll_normalized_minus_original_raw"]}
        method, seed = name.split("-")
        output = {"original": original,
                  "corrected" if method == "stratum" else "reweighted": shifted(
                      original, meta["previous"], meta["types"], meta["mask"], 1 if method == "stratum" else -1)}
        for readout, logs in output.items():
            key = f"{method}-{readout}-{seed}"
            choice, value, ties = helper.score(logs, meta)
            choices[key], vectors[key] = choice, value
            fits[key] = {"cells": {"heldout_service/"+kind: helper.cell(rows, meta, choice, value, ties, mask)
                                   for kind, mask in masks.items()}}
            exact[key] = {kind: exact_rates(rows, meta, choice, mask) for kind, mask in masks.items()}
    comparisons = {}
    for label, a, b in (("primary", "uniform-original", "stratum-corrected"),
                        ("secondary", "uniform-reweighted", "stratum-original")):
        pairs = {}
        for seed in SEEDS:
            ak, bk = f"{a}-{seed}", f"{b}-{seed}"
            cells = {}
            for kind, mask in masks.items():
                key = "heldout_service/"+kind
                rec = helper.paired(rows, meta["labels"], choices[ak], choices[bk], vectors[ak], vectors[bk], mask)
                ca, cb = fits[ak]["cells"][key], fits[bk]["cells"][key]
                rec.update(rows=int(mask.sum()), count_differences={k: ca["counts"][k]-cb["counts"][k] for k in ca["counts"]},
                    rare_rate_differences={k: None if ca["rare"][k]["rate"] is None else ca["rare"][k]["rate"]-cb["rare"][k]["rate"] for k in ca["rare"]})
                cells[key] = rec
            pairs[str(seed)] = {"cells": cells}
        comparisons[label] = {"seeds": pairs, "mean": mean_tree(list(pairs.values()))}
    services = sorted({row["service"] for row in rows if row["heldout_service"]})
    return {"fits": fits, "pairs": comparisons, "source_normalization": normalizations,
            "continuation": decisions(fits, exact, historical, services),
            "counts": {"raw_fits": 6, "readouts": 12, "primary_cells": 60, "paired_cells": 30,
                       "rows_per_fit": len(rows), "primary_support": {k: int(v.sum()) for k, v in masks.items()}}}


EXPECTED_SUPPORT = {"all": 7819, "changed": 578, "retained": 7241,
                    "unmentioned_retention": 4032, "assigned_retention": 3209}
EXPECTED_ROWS = 13599
REPORT_VERSION = "dialogue-objective-v1"
REPORT_HELPERS = {
    "scripts/report_dialogue_alignment.py": "68674dfd7ef456a6353a0567a9b9e54e2dc1cdbc2d9d59fc25ca92ee2d45539e",
    "scripts/report_dialogue_typed_v2.py": "6c97b4839352083885a3df5240cb1d3564a9652ec90ed30948a5bbada8b54365",
}
REQUIRED_SOURCES = {"scripts/study_dialogue_objective.py", "tests/test_study_dialogue_objective.py",
                    "scripts/report_dialogue_objective.py", "tests/test_report_dialogue_objective.py",
                    "scripts/audit_dialogue_objective.py", "tests/test_audit_dialogue_objective.py",
                    "research/dialogue-objective-protocol.md", HELPER, *REPORT_HELPERS}


def authenticate(args, h):
    """Authenticate selected bytes first, never open excluded training payloads."""
    run, report = Path(args.run).resolve(), Path(args.report).resolve()
    require(h.sha(run/"completed.json") == args.completed_sha256, "External execution completion")
    require(h.sha(report/"receipt.json") == args.report_receipt_sha256, "External report receipt")
    done, receipt = h.read(run/"completed.json"), h.read(report/"receipt.json")
    require(done["status"] == "completed" and done["phase"] == "train" and done["version"] == "dialogue-objective-v1"
            and done["completed_fits"] == done["expected_fits"] == list(FIT_ORDER)
            and done["limits"] == TRAIN_LIMITS and done["no_retry"] is True
            and done["quality_metrics_computed"] is False and done["encoder_calls"] == 0
            and done["official_dev_inference"] is False and done["test_contents_accessed"] is False,
            "Complete bounded six-fit study")
    require(0 < done["wall_seconds"] <= TRAIN_LIMITS["wall_seconds"]
            and 0 < done["process_lifetime_peak_rss_bytes"] <= TRAIN_LIMITS["rss_bytes"], "Execution resource bounds")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"}
    names |= {f"orders-{seed}.npy" for seed in SEEDS}
    names |= {f"fits/{fit}/{name}" for fit in FIT_ORDER
              for name in ("weights.pt", "updates.jsonl", "predictions.npz", "completed.json")}
    require(set(done["files"]) == names and len(names) == 31, "Exact structural 32-file manifest")
    require({p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()} == names | {"completed.json"},
            "Exact execution membership")
    for name, item in done["files"].items():
        h.safe(run, name)
        require(set(item) == {"sha256", "bytes"} and type(item["bytes"]) is int and item["bytes"] >= 0
                and type(item["sha256"]) is str and len(item["sha256"]) == 64
                and set(item["sha256"]) <= set("0123456789abcdef"), "Execution manifest record")
    require(sum(v["bytes"] for v in done["files"].values())+(run/"completed.json").stat().st_size
            <= TRAIN_LIMITS["output_bytes"], "Execution declared storage")
    selected = {"plan.json", "evaluation-rows.jsonl"}
    selected |= {f"fits/{fit}/{name}" for fit in FIT_ORDER for name in ("predictions.npz", "completed.json")}
    bindings = {run/name: done["files"][name] for name in selected}
    bindings[run/"completed.json"] = h.item(run/"completed.json")
    for path, item in bindings.items(): require(h.item(path) == item, "Selected metric payload: "+str(path))
    plan = h.read(run/"plan.json")
    require(h.sha(run/"plan.json") == done["plan_sha256"]
            and h.sha(run/"evaluation-rows.jsonl") == ROWS_PIN
            and plan["version"] == "dialogue-objective-v1" and plan["expected_fits"] == list(FIT_ORDER)
            and plan["limits"] == TRAIN_LIMITS and plan["model_method"] == "token_aligned"
            and plan["objective"]["stratum_counts"] == list(COUNTS)
            and plan["source_sha256"] == done["source_sha256"] and plan["runtime"] == done["runtime"], "Plan/cohort identity")
    require(REQUIRED_SOURCES <= set(plan["source_sha256"]), "Required scientific source closure")
    for name, pin in plan["source_sha256"].items():
        path = h.safe(ROOT, name)
        require(h.sha(path) == pin, "Study source identity: "+name)
        bindings[path] = h.item(path)
    for name in FIT_ORDER:
        record = h.read(run/"fits"/name/"completed.json"); method, seed = name.split("-")
        require(record["status"] == "completed" and record["version"] == "dialogue-objective-v1"
                and record["method"] == record["objective_weighting"] == method and record["seed"] == int(seed)
                and record["model_method"] == "token_aligned" and record["plan_sha256"] == done["plan_sha256"], "Fit receipt identity")
        require(set(record["files"]) == {"weights.pt", "updates.jsonl", "predictions.npz"}
                and all(record["files"][file] == done["files"][f"fits/{name}/{file}"] for file in record["files"]),
                "Nested fit manifest")
    require(receipt["status"] == "completed" and receipt["version"] == REPORT_VERSION
            and receipt["execution_completed_sha256"] == args.completed_sha256
            and receipt["plan_sha256"] == done["plan_sha256"]
            and receipt["technical_validity_passed"] is True
            and receipt["execution_members"] == done["files"]
            and receipt["source_sha256"] == {"scripts/report_dialogue_objective.py": plan["source_sha256"]["scripts/report_dialogue_objective.py"], **REPORT_HELPERS}
            and all(plan["source_sha256"][name] == pin for name, pin in REPORT_HELPERS.items())
            and receipt["model_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0
            and receipt["no_retry"] is True and receipt["limits"] == LIMITS
            and 0 < receipt["wall_seconds"] <= LIMITS["wall_seconds"]
            and 0 < receipt["process_lifetime_peak_rss_bytes"] <= LIMITS["rss_bytes"], "Report technical/execution binding")
    require(set(receipt["files"]) == {"started.json", "summary.json", "report.md"}, "Report payload closure")
    h.manifest(report, receipt["files"], terminal="receipt.json")
    bindings.update({report/name: item for name, item in receipt["files"].items()})
    bindings[report/"receipt.json"] = h.item(report/"receipt.json")
    producer = h.read(report/"summary.json")
    require(producer["status"] == "completed" and producer["version"] == REPORT_VERSION
            and producer["technical_validity_passed"] is True
            and producer["execution_completed_sha256"] == args.completed_sha256
            and producer["plan_sha256"] == done["plan_sha256"]
            and producer["source_sha256"] == plan["source_sha256"]
            and receipt["continuation_passed"] is producer["continuation"]["passed"], "Report completed technical summary")
    require(sum(p.stat().st_size for p in report.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Report storage bound")
    for name, pin in HISTORICAL_PINS.items():
        path = ROOT/HISTORICAL/name
        require(h.sha(path) == pin, "Pinned historical corrected reference")
        bindings[path] = h.item(path)
    historical = h.read(ROOT/HISTORICAL/"diagnostic-01/summary.json")
    hist_receipt = h.read(ROOT/HISTORICAL/"diagnostic-01/receipt.json")
    hist_audit = h.read(ROOT/HISTORICAL/"audit-01/receipt.json")
    require(historical["status"] == hist_receipt["status"] == hist_audit["status"] == "completed"
            and hist_audit["agreement"] is True
            and hist_receipt["files"]["summary.json"] == bindings[ROOT/HISTORICAL/"diagnostic-01/summary.json"]
            and hist_audit["producer_summary_sha256"] == HISTORICAL_PINS["diagnostic-01/summary.json"]
            and hist_audit["producer_receipt_sha256"] == HISTORICAL_PINS["diagnostic-01/receipt.json"],
            "Historical reference/audit joins")
    require(hist_receipt["input_sha256"]["rows"] == ROWS_PIN
            and producer["historical"]["fits"] == historical["fits"], "Historical reference cohort and reported values")
    # No arrays or row data are decoded before every selected payload is bound.
    rows = [h.decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    meta = h.metadata(rows)
    require(len(rows) == EXPECTED_ROWS and meta["ids"].tolist() == plan["evaluation_row_indices"]
            and {k: int(v.sum()) for k, v in h.primary_masks(meta).items()} == EXPECTED_SUPPORT, "Fixed evaluation support")
    return rows, producer, historical, bindings


def verify(result, producer, helper):
    require(set(producer["fits"]) == {f"{r}-{s}" for r in READOUTS for s in SEEDS}, "All twelve report readouts")
    checked = 0
    for key in ("fits", "pairs", "continuation"):
        checked += helper.compare(producer[key], result[key], key)
    # The original normalized distributions are explicit, not raw-log NLL.
    for key, expected in result["source_normalization"].items():
        checked += helper.compare(producer["source_normalization"][key], expected, "normalization/"+key)
    return checked


def execute(args):
    out = Path(args.out).resolve()
    for source in (args.run, args.report):
        parent = Path(source).resolve()
        require(not out.is_relative_to(parent) and not parent.is_relative_to(out), "Separate audit output")
    out.mkdir(parents=True, exist_ok=False)
    start, prior, helper = time.monotonic(), None, None
    sources, request = {}, {k: str(v) for k, v in vars(args).items()}
    def check():
        require(time.monotonic()-start <= LIMITS["wall_seconds"], "Audit wall limit")
        require(helper.peak_rss() <= LIMITS["rss_bytes"], "Audit process-lifetime RSS")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
    def timeout(_sig, _frame): raise TimeoutError("Objective audit wall limit")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "Existing process timer")
        prior = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        helper = load_helper()
        sources = {name: helper.sha(ROOT/name) for name in
                   ("scripts/audit_dialogue_objective.py", "tests/test_audit_dialogue_objective.py", HELPER)}
        helper.write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
                                         "limits": LIMITS, "scope": SCOPE})
        rows, producer, historical, bindings = authenticate(args, helper)
        check(); packets = {}
        for name in FIT_ORDER:
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "Prediction NPZ fields")
                packets[name] = {k: archive[k] for k in archive.files}
            check()
        result = reconstruct(rows, packets, historical, helper)
        result["scalar_comparisons"] = verify(result, producer, helper)
        check()
        result.update(status="completed", agreement=True, version=VERSION, scope=SCOPE,
                      execution_completed_sha256=args.completed_sha256,
                      producer_receipt_sha256=args.report_receipt_sha256,
                      producer_summary_sha256=helper.sha(Path(args.report)/"summary.json"))
        helper.write(out/"summary.json", result)
        for path, item in bindings.items(): require(helper.item(path) == item, "End input stability: "+str(path))
        for name, pin in sources.items(): require(helper.sha(ROOT/name) == pin, "End source stability")
        check()
        helper.write(out/"receipt.json", {"status": "completed", "version": VERSION, "agreement": True,
            "execution_completed_sha256": args.completed_sha256, "producer_receipt_sha256": args.report_receipt_sha256,
            "producer_summary_sha256": result["producer_summary_sha256"], "source_sha256": sources,
            "request": request, "files": {n: helper.item(out/n) for n in ("started.json", "summary.json")},
            "input_members": {str(p): v for p, v in bindings.items()}, "scope": SCOPE, "limits": LIMITS,
            "continuation_passed": result["continuation"]["passed"], "scalar_comparisons": result["scalar_comparisons"],
            "wall_seconds": time.monotonic()-start, "process_lifetime_peak_rss_bytes": helper.peak_rss(),
            "wall_scope": "Through input stability and payload hashing; terminal write/return cap-checked",
            "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0})
        check()
        return result
    except BaseException as error:
        if prior is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            import json
            if (out/"receipt.json").exists(): (out/"receipt.json").rename(out/"receipt-before-error.json")
            with (out/"failed.json").open("x") as stream:
                json.dump({"status": "failed", "version": VERSION, "request": request, "source_sha256": sources,
                           "error": repr(error), "scope": SCOPE, "wall_seconds": time.monotonic()-start,
                           "no_retry": True}, stream, indent=2, sort_keys=True, allow_nan=False)
        except BaseException as secondary:  # noqa: BLE001 - preserve original failure
            if callable(getattr(error, "add_note", None)): error.add_note("Audit receipt: "+repr(secondary))
        raise
    finally:
        if prior is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "completed-sha256", "report", "report-receipt-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
