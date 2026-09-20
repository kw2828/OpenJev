"""Fixed analytic inverse-loss-weight diagnostic on saved distributions only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-weight-prior-diagnostic-v1"
PARENT = "scripts/diagnose_dialogue_commitment.py"
PARENT_PIN = "1501eca952896dbd9b9e90f939091f73e775e503c12db9aa3834c9c8e764227c"
SPEC = "research/dialogue-weight-prior-diagnostic.md"
SPEC_PIN = "e4ca9363e00ac4bddc3e6fedb7a6ef8e6200d3c08ad4169e8045bbbac97b3ac9"
METHODS = ("flat_stratum", "token_mean", "token_aligned")
SEEDS = (6201, 6202, 6203)
READOUTS = ("original", "corrected")
CATEGORIES = tuple(f"{m}-{r}" for m in METHODS for r in READOUTS)
COUNTS = (17666, 9246, 2299)
LIMITS = {"wall_seconds": 60, "peak_rss_bytes": 1024**3, "output_bytes": 64*1024**2}
SCOPE = (
    "Posthoc fixed inverse-training-weight correction on exposed TRAIN development rows, using a "
    "privileged previous candidate and public candidate types only in construction. Targets and "
    "transition labels are used only for descriptive scoring. No threshold, strength, seed, hybrid "
    "or model is selected. Analytic float64 weights invert the declared objective, not the exact "
    "finite-precision optimizer. No calibration guarantee, new architecture, autonomous recurrence "
    "claim or new pass/fail gate. Historical campaign remains FAIL 18/22. Original prediction "
    "costs remain paid separately; this receipt measures only saved-output analysis. Full training, "
    "source execution, corpus, cache and checkpoint verification is not repeated. Positive keep-odds "
    "shifts preserve the winning alternative or move the decision to the previous candidate in exact "
    "arithmetic; changed-row accuracy gains are not expected."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_parent():
    path = ROOT/PARENT
    require(hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_PIN, "Frozen commitment helper pin")
    spec = importlib.util.spec_from_file_location("_weight_prior_saved_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights():
    counts = np.asarray(COUNTS, np.float64)
    values = counts.sum()/(3.*counts)
    names = ("unmentioned_retention", "assigned_retention", "changed")
    return values, {"fit_rows": sum(COUNTS), "strata": list(names), "fit_counts": list(COUNTS),
                    "analytic_float64": dict(zip(names, values.tolist(), strict=True)),
                    "training_float32_representation": dict(zip(names, values.astype(np.float32).astype(np.float64).tolist(), strict=True)),
                    "scope": "Weights are fixed from the pinned FIT counts. Float32 casts are reported, never used for correction."}


def correct(raw, sizes, previous, candidate_types, parent):
    """No current label, transition, held-out frequency or tunable strength."""
    original, source_witness, z = parent.normalize(raw, sizes, previous)
    support = np.arange(12)[None, :] < sizes[:, None]
    require(candidate_types.shape == raw.shape and candidate_types.dtype.kind in "iu"
            and ((candidate_types[support] >= 0) & (candidate_types[support] <= 4)).all()
            and (candidate_types[~support] == -1).all(), "Public candidate type support")
    row = np.arange(len(raw)); prior_types = candidate_types[row, previous]
    retention = np.where(prior_types == 0, 0, 1)
    analytic, _ = weights()
    adjusted = raw.astype(np.float64)-np.log(analytic[2])
    adjusted[row, previous] = raw[row, previous].astype(np.float64)-np.log(analytic[retention])
    corrected = adjusted-parent.logsumexp(adjusted)[:, None]
    require(np.isfinite(corrected[support]).all() and np.isneginf(corrected[~support]).all(), "Corrected support")
    mass = float(np.max(np.abs(np.exp(corrected).sum(axis=1)-1)))
    original_mass = float(np.max(np.abs(np.exp(original).sum(axis=1)-1)))
    require(max(mass, original_mass) <= 1e-12, "Float64 diagnostic normalization")
    old_alternatives = original.copy(); old_alternatives[row, previous] = -np.inf
    new_alternatives = corrected.copy(); new_alternatives[row, previous] = -np.inf
    old_change = parent.logsumexp(old_alternatives); new_change = parent.logsumexp(new_alternatives)
    old_conditional = old_alternatives-old_change[:, None]
    new_conditional = new_alternatives-new_change[:, None]
    alternatives = support.copy(); alternatives[row, previous] = False
    conditional_error = float(np.max(np.abs(old_conditional[alternatives]-new_conditional[alternatives])))
    require(np.allclose(old_conditional[alternatives], new_conditional[alternatives], atol=1e-10, rtol=1e-12),
            "Unchanged conditional alternative distribution")
    old_odds = original[row, previous]-old_change
    new_odds = corrected[row, previous]-new_change
    expected_shift = np.log(np.asarray(COUNTS, np.float64)[retention]/COUNTS[2])
    require(np.allclose(new_odds, old_odds+expected_shift, atol=1e-10, rtol=1e-12), "Analytic keep-odds identity")
    shift_witness = {}
    for key, mask in (("none_previous", prior_types == 0), ("assigned_previous", prior_types != 0)):
        observed = (new_odds-old_odds)[mask]
        shift_witness[key] = {"rows": int(mask.sum()),
            "expected": float(np.log(COUNTS[0 if key == "none_previous" else 1]/COUNTS[2])),
            "observed_min": float(observed.min()) if len(observed) else None,
            "observed_max": float(observed.max()) if len(observed) else None}
    old_choice, new_choice = np.argmax(original, axis=1), np.argmax(corrected, axis=1)
    validation = {"original_maximum_mass_error": original_mass, "corrected_maximum_mass_error": mass,
                  "alternative_conditional_max_abs_error": conditional_error,
                  "keep_odds_identity_max_abs_error": float(np.max(np.abs(new_odds-(old_odds+expected_shift)))),
                  "keep_odds_shift": shift_witness,
                  "hard_choice_witness": {"switches_to_previous": int(((old_choice != new_choice) & (new_choice == previous)).sum()),
                      "other_changes": int(((old_choice != new_choice) & (new_choice != previous)).sum()),
                      "previous_choice_reversals": int(((old_choice == previous) & (new_choice != previous)).sum())},
                  "tie_rule": "First exact maximum in original candidate order; no tolerance enlargement"}
    return {"original": original, "corrected": corrected}, validation, source_witness, z


def analyze(rows, meta, packets, parent, arithmetic, check=lambda: None):
    cells, services = parent.layouts(rows, meta)
    previous = np.asarray([r["previous_current_index"] for r in rows], np.int64)
    fits, choices, validations, source_norm, raw_primary, raw_choices = {}, {}, {}, {}, {}, {}
    for method in METHODS:
        for seed in SEEDS:
            check(); source = f"{method}-{seed}"
            raw_primary[source], raw_choices[source] = arithmetic.score(rows, meta, packets[source])
            outputs, validation, witness, z = correct(packets[source]["log_probs"], meta["sizes"], previous, meta["types"], parent)
            validations[source] = validation
            source_norm[source] = {**witness, "nll_normalized_minus_original_raw": {key: parent.means(z, group) for key, group in cells.items()}}
            for readout, logs in outputs.items():
                check(); key = f"{method}-{readout}-{seed}"
                values, choice, types, ties = parent.vectors(logs, meta)
                if readout == "original":
                    require(np.array_equal(choice, raw_choices[source]), "Normalized original argmax identity")
                choices[key] = choice
                fits[key] = {"source_fit": source, "method": method, "readout": readout, "seed": seed,
                    "validation": {"maximum_mass_error": validation[readout+"_maximum_mass_error"], "exact_tie_rows": int(ties.sum())},
                    "cells": {name: parent.describe(values, choice, types, ties, meta, group) for name, group in cells.items()},
                    "services": {service: {name: parent.describe(values, choice, types, ties, meta, group) for name, group in strata.items()}
                                 for service, strata in services.items()}}
    comparisons = {}
    for category in CATEGORIES:
        comparisons[category] = {}
        for method in METHODS:
            control = method+"-original"; seeds = {}
            for seed in SEEDS:
                check(); akey, bkey = f"{category}-{seed}", f"{control}-{seed}"
                a, b = fits[akey], fits[bkey]
                seeds[str(seed)] = {
                    "cells": {key: parent.paired_cell(a["cells"][key], b["cells"][key], choices[akey], choices[bkey], meta, group)
                              for key, group in cells.items()},
                    "services": {svc: {key: parent.paired_cell(a["services"][svc][key], b["services"][svc][key], choices[akey], choices[bkey], meta, group)
                                        for key, group in strata.items()} for svc, strata in services.items()}}
            comparisons[category][control] = {"seeds": seeds, "mean": parent.average(list(seeds.values()))}
    tradeoffs = {}
    for method in METHODS:
        paired = {}
        for seed in SEEDS:
            original, corrected = (fits[f"{method}-{r}-{seed}"]["cells"] for r in READOUTS)
            paired[str(seed)] = {}
            for panel in ("all", "seen_service", "heldout_service"):
                retained, changed = panel+"/retained", panel+"/changed"
                paired[str(seed)][panel] = {
                    "retained_errors_recovered": original[retained]["counts"]["error"]-corrected[retained]["counts"]["error"],
                    "changed_correct_delta": corrected[changed]["counts"]["correct"]-original[changed]["counts"]["correct"],
                    "retained_rows": original[retained]["rows"], "changed_rows": original[changed]["rows"]}
        tradeoffs[method] = {"seeds": paired, "mean": parent.average(list(paired.values()))}
    raw_pairs = {control: {str(seed): arithmetic.pair_counts(meta, raw_choices[f"token_aligned-{seed}"], raw_choices[f"{control}-{seed}"])
                          for seed in SEEDS} for control in METHODS[:2]}
    return {"fits": fits, "validations": validations, "source_normalization": source_norm,
            "correction_weights": weights()[1], "original_raw_primary": raw_primary, "original_raw_pairs": raw_pairs,
            "means": {category: {part: parent.average([fits[f"{category}-{seed}"][part] for seed in SEEDS])
                                  for part in ("cells", "services")} for category in CATEGORIES},
            "comparisons": comparisons, "tradeoffs": tradeoffs,
            "mean_scope": "Equal mean of three optimizer seeds on identical rows; mean counts are averages, not additional independent examples."}


def execute(args):
    out = Path(args.out).resolve()
    require(all(not out.is_relative_to(Path(getattr(args, name)).resolve()) and
                not Path(getattr(args, name)).resolve().is_relative_to(out) for name in ("run", "report", "audit")), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); prior_handler = None
    request = {k: str(v) for k, v in vars(args).items()}
    sources = {PARENT: PARENT_PIN, SPEC: SPEC_PIN}
    def check():
        require(time.monotonic()-started <= LIMITS["wall_seconds"], "Diagnostic wall cap")
        require(parent.peak_rss() <= LIMITS["peak_rss_bytes"], "Diagnostic RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Diagnostic output cap")
    def timeout(_sig, _frame):
        raise TimeoutError("Diagnostic 60-second wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "No overlapping timer")
        prior_handler = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        parent = load_parent(); portable = parent.load_portable()
        sources.update({parent.PORTABLE: parent.PORTABLE_PIN, portable.HELPER: portable.HELPER_SHA256,
            "scripts/diagnose_dialogue_weight_prior.py": portable.digest(__file__, check),
            "tests/test_dialogue_weight_prior.py": portable.digest(ROOT/"tests/test_dialogue_weight_prior.py", check)})
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
            "input_sha256": parent.PINS, "prediction_sha256": parent.PREDICTIONS, "limits": LIMITS, "scope": SCOPE, "no_retry": True})
        require(portable.digest(ROOT/SPEC, check) == SPEC_PIN, "Published diagnostic specification pin")
        arithmetic = portable.load_arithmetic(check)
        rows, meta, original, selected, summary_member, paths, audit_receipt = parent.authenticate(args, portable, arithmetic, check)
        plan = arithmetic.read(Path(args.run)/"plan.json")
        require(plan["objective"]["rows"] == sum(COUNTS) and plan["objective"]["stratum_counts"] == list(COUNTS), "Pinned FIT-count objective")
        packets = {}
        for name in arithmetic.ORDER:
            check()
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "Prediction NPZ schema")
                packets[name] = {key: archive[key] for key in archive.files}
        result = analyze(rows, meta, packets, parent, arithmetic, check)
        historical = arithmetic.continuation(result["original_raw_primary"])
        arithmetic.compare_report(result["original_raw_primary"], result["original_raw_pairs"], historical, original)
        require(historical["passed"] is False and historical["checks_passed"] == 18, "Historical failed result unchanged")
        result.update(status="completed", version=VERSION, scope=SCOPE, diagnostic_has_gate=False,
            original_campaign={"passed": False, "checks_passed": 18, "total_checks": 22},
            input_sha256=parent.PINS, prediction_sha256=parent.PREDICTIONS, source_sha256=sources,
            counts={"rows_per_seed": len(rows), "seeds": 3, "original_fits": 9, "readouts_per_fit": 2,
                    "saved_distribution_cells": 18, "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0})
        check(); write(out/"summary.json", result); check()
        for name, item in selected.items(): portable.bind(Path(args.run)/name, item, check)
        portable.bind(Path(args.report)/"summary.json", summary_member, check)
        for name, item in audit_receipt["files"].items(): portable.bind(arithmetic.safe(Path(args.audit), name), item, check)
        for name, path in paths.items(): require(portable.digest(path, check) == parent.PINS[name], "End input stability: "+name)
        for name, pin in sources.items(): require(portable.digest(ROOT/name, check) == pin, "End source stability: "+name)
        files = {p.name: {"sha256": portable.digest(p, check), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        write(out/"receipt.json", {"status": "completed", "version": VERSION, "request": request,
            "source_sha256": sources, "input_sha256": parent.PINS, "prediction_sha256": parent.PREDICTIONS,
            "files": files, "limits": LIMITS, "scope": SCOPE, "diagnostic_has_gate": False,
            "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True,
            "wall_seconds": time.monotonic()-started, "wall_scope": "Saved-only analysis through payload hashing; completion write/return cap-checked",
            "process_lifetime_peak_rss_bytes": parent.peak_rss()})
        check()
        return result
    except BaseException as error:
        if prior_handler is not None: signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists(): (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                "source_sha256": sources, "error": repr(error), "wall_seconds": time.monotonic()-started,
                "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original diagnostic failure
            if callable(getattr(error, "add_note", None)): error.add_note("Failure preservation: "+repr(secondary))
        raise
    finally:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "report", "audit", "out"): parser.add_argument("--"+key, required=True)
    execute(parser.parse_args())
