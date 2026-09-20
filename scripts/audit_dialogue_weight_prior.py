"""Independent primary arithmetic for the fixed FIT-weight prior diagnostic.

Reuses hash-pinned independent authentication/scoring helpers, never producer
arithmetic. The new correction uses a direct analytic stay-log-odds shift.
No models, feature caches, checkpoints, journals or orders are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import signal
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/audit_dialogue_commitment.py"
HELPER_PIN = "862368fadf6d2f143b70e3a8a2a48b02418d849defd1a184f9251f881b8184ab"
SPEC = "research/dialogue-weight-prior-diagnostic.md"
SPEC_PIN = "e4ca9363e00ac4bddc3e6fedb7a6ef8e6200d3c08ad4169e8045bbbac97b3ac9"
VERSION = "dialogue-weight-prior-independent-primary-audit-v1"
COUNTS = {"unmentioned_retention": 17666, "assigned_retention": 9246, "changed": 2299}
TOTAL = 29211
LIMITS = {"wall_seconds": 60, "rss_bytes": 1024**3, "output_bytes": 64*1024**2}
SCOPE = (
    "Independent FIT-count correction using stay-log-odds shifts and logaddexp normalization. "
    "Authentication and primary scoring reuse frozen independent helper "+HELPER_PIN+". "
    "All eighteen fits and five held-out-service strata, rare supports, normalized NLL/Brier "
    "and paired repairs/harms are recomputed. Other panels, service tables and training are "
    "not independently replayed. Only selected plan/rows/nine predictions are byte-checked "
    "against the structural training manifest. No checkpoint, cache, journal or order access. "
    "Posthoc descriptive audit only: the original campaign remains FAIL 18/22."
)


def load_helpers():
    path = ROOT/HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_PIN:
        raise ValueError("Frozen independent helper identity")
    spec = importlib.util.spec_from_file_location("_weight_prior_independent_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correct(raw, mask, previous, candidate_types, helper):
    """Inputs are public prior/support and model logs; no target argument."""
    q, z, normalization = helper.normalize(raw, mask)
    row = np.arange(len(q))
    helper.require(previous.dtype == np.int64 and previous.shape == (len(q),)
                   and np.all((previous >= 0) & (previous < mask.sum(axis=1))), "Supported prior index")
    helper.require(candidate_types.shape == mask.shape and candidate_types.dtype.kind in "iu"
                   and np.all((candidate_types[mask] >= 0) & (candidate_types[mask] <= 4))
                   and np.all(candidate_types[~mask] == -1), "Public type layout")
    retention_count = np.where(candidate_types[row, previous] == 0,
                               COUNTS["unmentioned_retention"], COUNTS["assigned_retention"])
    shift = np.asarray([math.log(int(n)/COUNTS["changed"]) for n in retention_count])
    adjusted = q.copy()
    adjusted[row, previous] += shift
    corrected = adjusted-np.logaddexp.reduce(adjusted, axis=1)[:, None]
    helper.require(np.isfinite(corrected[mask]).all() and np.isneginf(corrected[~mask]).all(), "Corrected finite support")
    mass_error = float(np.abs(np.expm1(np.logaddexp.reduce(corrected, axis=1))).max())
    helper.require(mass_error <= 1e-12, "Corrected unit mass")
    alternatives = mask.copy(); alternatives[row, previous] = False
    helper.require(alternatives.any(axis=1).all(), "At least one alternative")
    before_mass = np.logaddexp.reduce(np.where(alternatives, q, -np.inf), axis=1)
    after_mass = np.logaddexp.reduce(np.where(alternatives, corrected, -np.inf), axis=1)
    before = q-before_mass[:, None]; after = corrected-after_mass[:, None]
    helper.require(np.allclose(before[alternatives], after[alternatives], atol=1e-10, rtol=1e-12), "Alternative conditional identity")
    odds_shift = (corrected[row, previous]-after_mass)-(q[row, previous]-before_mass)
    helper.require(np.allclose(odds_shift, shift, atol=1e-10, rtol=1e-12), "Analytic stay-odds shift")
    witness = {"maximum_mass_error": mass_error,
               "alternative_conditional_max_abs_error": float(np.abs(before[alternatives]-after[alternatives]).max()),
               "keep_log_odds_shift_max_abs_error": float(np.abs(odds_shift-shift).max())}
    return q, corrected, normalization, z, witness


def analytic_weights():
    return {key: {"count": n, "analytic_float64": TOTAL/(3*n),
                  "training_float32": float(np.float32(TOTAL/(3*n)))} for key, n in COUNTS.items()}


def aggregate(rows, packets, producer, helper):
    helper.require(set(packets) == set(helper.PREDICTION_PINS), "All nine original distributions")
    expected = {f"{m}-{r}-{s}" for m in helper.METHODS for r in ("original", "corrected") for s in helper.SEEDS}
    helper.require(set(producer["fits"]) == expected, "All eighteen diagnostic cells")
    meta = helper.metadata(rows); masks = helper.primary_masks(meta)
    fits, normalizations, comparisons, witnesses = {}, {}, {}, {}
    weights = analytic_weights()
    helper.require(producer["correction_weights"]["fit_counts"] == list(COUNTS.values())
                   and producer["correction_weights"]["strata"] == list(COUNTS), "Fixed correction-count metadata")
    checked = helper.compare(producer["correction_weights"], {
        "fit_rows": TOTAL, "analytic_float64": {k: v["analytic_float64"] for k, v in weights.items()},
        "training_float32_representation": {k: v["training_float32"] for k, v in weights.items()}}, "correction_weights")
    for seed in helper.SEEDS:
        choices, vectors, cells = {}, {}, {}
        for method in helper.METHODS:
            name = f"{method}-{seed}"; packet = packets[name]
            helper.require(set(packet) == {"row_indices", "log_probs"}
                           and packet["row_indices"].dtype == np.int64
                           and np.array_equal(packet["row_indices"], meta["ids"]), "Prediction atomic row join")
            original, corrected, norm, z, witness = correct(packet["log_probs"], meta["mask"],
                meta["previous"], meta["types"], helper)
            norm["nll_normalized_minus_original_raw"] = {"heldout_service/"+key: helper.means(z, rows, mask)
                                                          for key, mask in masks.items()}
            normalizations[name] = norm
            checked += helper.compare(producer["source_normalization"][name], norm, "normalization/"+name)
            witnesses[name] = witness
            checked += helper.compare(producer["validations"][name], {
                "corrected_maximum_mass_error": witness["maximum_mass_error"],
                "alternative_conditional_max_abs_error": witness["alternative_conditional_max_abs_error"],
                "keep_odds_identity_max_abs_error": witness["keep_log_odds_shift_max_abs_error"]}, "correction_validation/"+name)
            for readout, logs in (("original", original), ("corrected", corrected)):
                key = method+"-"+readout; fit = key+"-"+str(seed)
                choice, value, ties = helper.score(logs, meta)
                choices[key], vectors[key] = choice, value
                cells[key] = {"heldout_service/"+g: helper.cell(rows, meta, choice, value, ties, mask)
                              for g, mask in masks.items()}
                fits[fit] = {"cells": cells[key]}
                checked += helper.compare(producer["fits"][fit], fits[fit], "fits/"+fit)
                checked += helper.compare(producer["fits"][fit]["validation"], {
                    "maximum_mass_error": float(np.abs(np.expm1(np.logaddexp.reduce(logs, axis=1))).max()),
                    "exact_tie_rows": int(ties.sum())}, "fit_validation/"+fit)
        for name in cells:
            for method in helper.METHODS:
                control = method+"-original"
                pairs = {}
                for group, mask in masks.items():
                    key = "heldout_service/"+group
                    record = helper.paired(rows, meta["labels"], choices[name], choices[control],
                                           vectors[name], vectors[control], mask)
                    a, b = cells[name][key], cells[control][key]
                    record.update(rows=int(mask.sum()),
                        count_differences={k: a["counts"][k]-b["counts"][k] for k in a["counts"]},
                        rare_rate_differences={k: None if a["rare"][k]["rate"] is None else
                            a["rare"][k]["rate"]-b["rare"][k]["rate"] for k in a["rare"]})
                    pairs[key] = record
                label = control
                comparisons.setdefault(name, {}).setdefault(label, {"seeds": {}})["seeds"][str(seed)] = {"cells": pairs}
                checked += helper.compare(producer["comparisons"][name][label]["seeds"][str(seed)],
                    {"cells": pairs}, f"comparisons/{name}/{label}/{seed}")
    return {"fits": fits, "comparisons": comparisons, "source_normalization": normalizations,
            "correction_validation": witnesses, "analytic_weights": analytic_weights(), "scalar_comparisons": checked,
            "counts": {"original_fits": 9, "constructions": 18, "primary_cells": 90, "paired_cells": 270,
                       "rows_per_fit": len(rows), "primary_support": {k: int(v.sum()) for k, v in masks.items()}}}


def authenticate(args, h):
    run, diagnostic = Path(args.run).resolve(), Path(args.diagnostic).resolve()
    h.require(args.run_sha256 == h.RUN_PIN and h.sha(run/"completed.json") == h.RUN_PIN, "Fixed completed run")
    h.require(h.sha(diagnostic/"summary.json") == args.summary_sha256
              and h.sha(diagnostic/"receipt.json") == args.receipt_sha256, "External diagnostic pins")
    h.require(h.sha(ROOT/SPEC) == SPEC_PIN, "Published weight-prior specification")
    done, plan = h.read(run/"completed.json"), h.read(run/"plan.json")
    h.require(h.sha(run/"plan.json") == h.PLAN_PIN and done["plan_sha256"] == h.PLAN_PIN
              and h.sha(run/"evaluation-rows.jsonl") == h.ROWS_PIN, "Frozen plan/row identity")
    h.require(done["status"] == "completed" and len(done["completed_fits"]) == 9
              and set(done["completed_fits"]) == set(h.PREDICTION_PINS), "Nine complete fits")
    names = {"started.json", "plan.json", "evaluation-rows.jsonl", "references.npz"} | {f"orders-{s}.npy" for s in h.SEEDS}
    names |= {f"fits/{f}/{n}" for f in h.PREDICTION_PINS for n in ("completed.json", "updates.jsonl", "weights.pt", "predictions.npz")}
    h.require(set(done["files"]) == names and len(names) == 43
              and plan["source_sha256"] == done["source_sha256"], "Structural manifest and source identity")
    selected = {"plan.json", "evaluation-rows.jsonl"} | {f"fits/{f}/predictions.npz" for f in h.PREDICTION_PINS}
    bindings = {run/n: done["files"][n] for n in selected}
    for path, record in bindings.items():
        h.require(h.item(path) == record, "Selected metric payload: "+str(path))
    for fit, pin in h.PREDICTION_PINS.items():
        h.require(done["files"][f"fits/{fit}/predictions.npz"]["sha256"] == pin, "Fixed prediction identity")
    bindings[run/"completed.json"] = h.item(run/"completed.json")
    parent = ROOT/"output/dialogue-token-alignment-scientific-v1"
    for name, pin in h.PARENT_PINS.items():
        h.require(h.sha(parent/name) == pin, "Inherited report/audit pin")
        bindings[parent/name] = h.item(parent/name)
    objective = plan["objective"]
    h.require(objective["rows"] == TOTAL and objective["stratum_counts"] == list(COUNTS.values()), "Fixed FIT-only counts")
    for i, n in enumerate(COUNTS.values()):
        h.require(sum(objective["counts"][i]) == n, "FIT target-count reconciliation")
        for count, weight in zip(objective["counts"][i], objective["weights"]["stratum"][i], strict=True):
            h.require(weight == TOTAL/(3*n) if count else weight is None, "Declared analytic training objective")
    receipt, producer = h.read(diagnostic/"receipt.json"), h.read(diagnostic/"summary.json")
    h.require(receipt["status"] == producer["status"] == "completed"
              and receipt["version"] == producer["version"] == "dialogue-weight-prior-diagnostic-v1"
              and receipt["diagnostic_has_gate"] is producer["diagnostic_has_gate"] is False, "Completed descriptive diagnostic")
    h.require(producer["original_campaign"] == {"passed": False, "checks_passed": 18, "total_checks": 22}, "Original failure retained")
    h.require(set(receipt["files"]) == {"started.json", "summary.json"}, "Diagnostic closure")
    h.manifest(diagnostic, receipt["files"], terminal="receipt.json")
    bindings.update({diagnostic/n: v for n, v in receipt["files"].items()})
    bindings[diagnostic/"receipt.json"] = h.item(diagnostic/"receipt.json")
    h.require(receipt["source_sha256"] == producer["source_sha256"]
              == h.read(diagnostic/"started.json")["source_sha256"], "Diagnostic source maps")
    for name, pin in receipt["source_sha256"].items():
        path = h.safe(ROOT, name)
        h.require(h.sha(path) == pin, "Diagnostic source identity")
        bindings[path] = h.item(path)
    h.require(receipt["prediction_sha256"] == producer["prediction_sha256"] == h.PREDICTION_PINS,
              "Diagnostic prediction pins")
    expected_pins = {"completed": h.RUN_PIN, "plan": h.PLAN_PIN, "rows": h.ROWS_PIN,
                     "report_receipt": h.PARENT_PINS["report-01/receipt.json"],
                     "report_summary": h.PARENT_PINS["report-01/summary.json"],
                     "audit_receipt": h.PARENT_PINS["audit-01/result-01/receipt.json"]}
    h.require(receipt["input_sha256"] == producer["input_sha256"] == expected_pins, "Diagnostic input bindings")
    h.require(receipt["model_calls"] == receipt["encoder_calls"] == receipt["checkpoint_deserializations"] == 0, "Saved-only scope")
    rows = [h.decode(line) for line in (run/"evaluation-rows.jsonl").read_text().splitlines()]
    meta = h.metadata(rows)
    h.require(len(rows) == 13599 and meta["ids"].tolist() == plan["evaluation_row_indices"], "Canonical row coverage")
    h.require({k: int(v.sum()) for k, v in h.primary_masks(meta).items()} ==
              {"all": 7819, "changed": 578, "retained": 7241, "unmentioned_retention": 4032, "assigned_retention": 3209}, "Primary supports")
    return rows, producer, bindings


def execute(args):
    import json
    out = Path(args.out).resolve()
    for value in (args.run, args.diagnostic):
        path = Path(value).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError("Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); old_handler = None; sources = {}
    request = {k: str(v) for k, v in vars(args).items()}

    def write(path, value):
        with Path(path).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    def timeout(_signal, _frame):
        raise TimeoutError("Independent weight-prior audit 60-second limit")
    try:
        h = load_helpers()
        sources = {HELPER: HELPER_PIN, SPEC: SPEC_PIN, "scripts/audit_dialogue_weight_prior.py": h.sha(__file__),
                   "tests/test_audit_dialogue_weight_prior.py": h.sha(ROOT/"tests/test_audit_dialogue_weight_prior.py")}
        h.require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "No overlapping timer")
        old_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"]-(time.monotonic()-started))
        def check():
            h.require(time.monotonic()-started <= LIMITS["wall_seconds"] and h.peak_rss() <= LIMITS["rss_bytes"], "Audit time/RSS cap")
            h.require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")
        write(out/"started.json", {"request": request, "source_sha256": sources, "scope": SCOPE, "limits": LIMITS})
        check(); rows, producer, bindings = authenticate(args, h); check()
        packets = {}
        for name in h.PREDICTION_PINS:
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                h.require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "Prediction NPZ schema")
                packets[name] = {key: archive[key] for key in archive.files}
            check()
        result = aggregate(rows, packets, producer, h); check()
        result.update(status="completed", agreement=True, version=VERSION, scope=SCOPE,
            execution_completed_sha256=h.RUN_PIN, producer_summary_sha256=args.summary_sha256,
            producer_receipt_sha256=args.receipt_sha256, original_campaign_passed=False)
        write(out/"summary.json", result)
        for path, record in bindings.items():
            h.require(h.item(path) == record, "End input stability")
        for name, pin in sources.items():
            h.require(h.sha(ROOT/name) == pin, "End source stability")
        check()
        write(out/"receipt.json", {"status": "completed", "agreement": True, "version": VERSION,
            "producer_receipt_sha256": args.receipt_sha256, "producer_summary_sha256": args.summary_sha256,
            "execution_completed_sha256": h.RUN_PIN, "source_sha256": sources,
            "files": {name: h.item(out/name) for name in ("started.json", "summary.json")},
            "request": request, "scope": SCOPE, "limits": LIMITS,
            "input_members": {str(p): v for p, v in bindings.items()}, "scalar_comparisons": result["scalar_comparisons"],
            "wall_seconds": time.monotonic()-started, "process_lifetime_peak_rss_bytes": h.peak_rss(),
            "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0})
        check()
        return result
    except BaseException as error:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "error": repr(error), "request": request,
                "source_sha256": sources, "wall_seconds": time.monotonic()-started, "scope": SCOPE,
                "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original exception
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation: "+repr(secondary))
        raise
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "run-sha256", "diagnostic", "summary-sha256", "receipt-sha256", "out"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
