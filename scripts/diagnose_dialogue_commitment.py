"""Fixed saved-distribution counterfactuals, with no new scientific gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dialogue-commitment-diagnostic-v1"
PORTABLE = "scripts/recheck_dialogue_alignment.py"
PORTABLE_PIN = "77b432cd0dab796b671fe64107a022f493190667981d4fea3c25ddb6b249e5de"
SPEC = "research/dialogue-alignment-commitment-diagnostic.md"
SEEDS = (6201, 6202, 6203)
CONSTRUCTIONS = ("MM", "AA", "MA", "AM", "flat")
CONTROLS = ("flat", "MM", "AA")
STRATA = ("all", "changed", "retained", "unmentioned_retention", "assigned_retention")
PINS = {
    "completed": "e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c",
    "plan": "e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1",
    "rows": "56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0",
    "report_receipt": "7591182a5c07c9cd4ee4035128b6c0900d7d9177ba89efccfe92bfd528c8a6c5",
    "report_summary": "b4b8c1d3f6e9d1e98096c2684a5bca84b011496f89044e3f3eb31260f2fbbdc3",
    "audit_receipt": "8773cf48d0b07b10f6bb920549d3628a4716beb814c98185df021081046b19bd",
}
PREDICTIONS = {
    "flat_stratum-6201": "dbccd26acbaa4622de3b2ecd66a752f7a06d67cc9598c880dc933ee1c8ba27ec",
    "flat_stratum-6202": "1ff1ae1b6a8f746fd086a63e7ecfa355751e2c1bcb5a6f7cdeaa8cba42444537",
    "flat_stratum-6203": "773ee404db257cfa81c5de0fe75bc6e495de10759c2094c9ac06857e256caf40",
    "token_mean-6201": "1979aa6c082415d208d4ea5ab8dfff738a76c0f09efa569199a083f573e16dc3",
    "token_mean-6202": "ec1534af8240f36585c5c6512414cf2eba9c13d27ffac7083c0e7411738ea72d",
    "token_mean-6203": "2b20d456f367bf0fd2cff57ee0684faba454f0aeea54042cef956620b5098c9b",
    "token_aligned-6201": "bae0149549750430358c52ef0ecff2b2e237704ff442fc232d6ea2a912d4a170",
    "token_aligned-6202": "0067cbf5e9fb38a24a221873d27addea3f1cd3a5e9bda8125806afbb9bbe42d6",
    "token_aligned-6203": "066b953cd9073fc4a927feeb77e0a6c44225f255957a4a57f038fd3745bcfb0a",
}
LIMITS = {"wall_seconds": 60, "peak_rss_bytes": 1024**3, "output_bytes": 64*1024**2}
SCOPE = (
    "Posthoc algebraic counterfactuals on previously exposed TRAIN development rows with privileged "
    "previous candidates. No new gold, model, encoder, cache or checkpoint access. Conditional "
    "alternative distributions include concentration as well as ordering; donating previous-candidate "
    "mass does not donate an argmax keep/change gate. These are factors of final distributions, not "
    "internal mechanisms. No calibration, new trained model, winner selection or continuation gate "
    "is established. The original campaign remains FAIL 18/22. Both source prediction costs remain "
    "part of the original campaign, not free. Source/input authentication is reused from the pinned "
    "portable checker; normalized diagnostic scoring and counterfactuals are new saved-only arithmetic."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_portable():
    path = ROOT/PORTABLE
    require(hashlib.sha256(path.read_bytes()).hexdigest() == PORTABLE_PIN, "Portable source pin")
    spec = importlib.util.spec_from_file_location("_commitment_portable", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def logsumexp(logs):
    maximum = np.max(logs, axis=1)
    require(np.isfinite(maximum).all(), "Nonempty finite log support")
    return maximum + np.log(np.exp(logs-maximum[:, None]).sum(axis=1))


def normalize(raw, sizes, previous):
    """No labels or outcome-based routing. Keep arbitrarily small finite logs."""
    require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 12, "Saved float32 log schema")
    require(sizes.shape == previous.shape == (len(raw),) and sizes.dtype.kind in "iu"
            and previous.dtype.kind in "iu" and (sizes >= 2).all() and (sizes <= 12).all()
            and (previous >= 0).all() and (previous < sizes).all(), "Supported prior and alternative")
    support = np.arange(12)[None, :] < sizes[:, None]
    require(np.isfinite(raw[support]).all() and (raw[support] <= 0).all()
            and np.isneginf(raw[~support]).all(), "Raw supported logs / padding")
    logs = raw.astype(np.float64)
    raw_mass_error = float(np.max(np.abs(np.exp(logs).sum(axis=1)-1)))
    require(raw_mass_error <= 2e-6, "Original probability mass")
    z = logsumexp(logs)
    q = logs-z[:, None]
    require(np.array_equal(np.argmax(q, axis=1), np.argmax(raw, axis=1)), "Normalization argmax identity")
    return q, {"maximum_raw_mass_error": raw_mass_error, "max_abs_log_z": float(np.max(np.abs(z))),
               "mean_log_z": float(math.fsum(z)/len(z)), "min_log_z": float(z.min()), "max_log_z": float(z.max())}, z


def construct(mean, aligned, sizes, previous):
    """Return canonical self cells and both fixed cross combinations."""
    sources, witnesses, zs = {}, {}, {}
    for key, raw in (("M", mean), ("A", aligned)):
        sources[key], witnesses[key], zs[key] = normalize(raw, sizes, previous)
    row = np.arange(len(mean)); support = np.arange(12)[None, :] < sizes[:, None]
    pieces = {}
    for name, q in sources.items():
        alternatives = q.copy(); alternatives[row, previous] = -np.inf
        change = logsumexp(alternatives)
        pieces[name] = (q[row, previous], change, alternatives-change[:, None])
    outputs, validations = {}, {}
    for name in CONSTRUCTIONS[:4]:
        x, y = name
        stay, change, _ = pieces[x]
        hybrid = change[:, None]+pieces[y][2]
        hybrid[row, previous] = stay
        max_difference, exact_choice = None, None
        if x == y:
            max_difference = float(np.max(np.abs(hybrid[support]-sources[x][support])))
            require(np.allclose(hybrid[support], sources[x][support], atol=1e-10, rtol=1e-12), "Self reconstruction identity")
            hybrid = sources[x].copy()
            exact_choice = np.array_equal(np.argmax(hybrid, axis=1), np.argmax(mean if x == "M" else aligned, axis=1))
            require(exact_choice, "Self canonical argmax")
        require(np.isfinite(hybrid[support]).all() and np.isneginf(hybrid[~support]).all(), "Constructed support")
        mass_error = float(np.max(np.abs(np.exp(hybrid).sum(axis=1)-1)))
        require(mass_error <= 1e-12, "Constructed float64 normalization")
        outputs[name] = hybrid
        validations[name] = {"maximum_mass_error": mass_error, "self_reconstruction_max_abs": max_difference,
                              "self_argmax_exact": None if exact_choice is None else bool(exact_choice)}
    return outputs, validations, witnesses, zs


def layouts(rows, meta):
    n = len(rows)
    strata = {"all": np.ones(n, bool), "changed": meta["changed"], "retained": ~meta["changed"],
              "unmentioned_retention": ~meta["changed"] & (meta["target_types"] == 0),
              "assigned_retention": ~meta["changed"] & (meta["target_types"] != 0)}
    panels = {"all": np.ones(n, bool), "seen_service": ~meta["held"], "heldout_service": meta["held"]}
    service = np.asarray([r["service"] for r in rows]); dialogue = np.asarray([r["dialogue_id"] for r in rows])
    def layout(mask):
        selected = np.flatnonzero(mask)
        result = {"indices": selected}
        for name, values in (("equal_service", service), ("equal_dialogue", dialogue)):
            _, inverse, counts = np.unique(values[selected], return_inverse=True, return_counts=True)
            result[name] = (inverse, counts)
        return result
    cells = {f"{p}/{s}": layout(pmask & smask) for p, pmask in panels.items() for s, smask in strata.items()}
    services = {str(name): {s: layout((service == name) & mask) for s, mask in strata.items()} for name in np.unique(service)}
    return cells, services


def means(vector, group):
    selected = vector[group["indices"]]
    if not len(selected):
        return dict.fromkeys(("row", "equal_service", "equal_dialogue"))
    result = {"row": float(np.mean(selected, dtype=np.float64))}
    for name in ("equal_service", "equal_dialogue"):
        inverse, counts = group[name]
        result[name] = float(np.mean(np.bincount(inverse, weights=selected)/counts))
    return result


def vectors(logs, meta):
    choices = np.argmax(logs, axis=1)
    correct = choices == meta["labels"]
    selected_types = meta["types"][np.arange(len(logs)), choices]
    branch = np.minimum(selected_types, 2) != np.minimum(meta["target_types"], 2)
    wrong_value = ~correct & ~branch
    probabilities = np.exp(logs)
    probabilities[np.arange(len(logs)), meta["labels"]] -= 1
    tie = (logs == logs.max(axis=1, keepdims=True)).sum(axis=1) > 1
    return {"accuracy": correct.astype(float), "error": (~correct).astype(float),
            "wrong_selected_branch": branch.astype(float), "wrong_value": wrong_value.astype(float),
            "nll": -logs[np.arange(len(logs)), meta["labels"]],
            "brier": np.square(probabilities).sum(axis=1)}, choices, selected_types, tie


def describe(values, choices, selected_types, ties, meta, group):
    indices = group["indices"]
    correct = choices == meta["labels"]
    rare = {}
    for name, code in (("true", 2), ("dontcare", 1)):
        for kind in ("recall", "false_positive"):
            eligible = meta["target_types"] == code if kind == "recall" else (
                (meta["target_types"] != code) & np.any(meta["types"] == code, axis=1))
            hit = correct if kind == "recall" else selected_types == code
            denominator = int(eligible[indices].sum())
            numerator = int((hit & eligible)[indices].sum())
            rare[f"{name}_{kind}"] = {"numerator": numerator, "denominator": denominator,
                                       "rate": numerator/denominator if denominator else None}
    return {"rows": len(indices), "counts": {key: int(values[value][indices].sum()) for key, value in
            (("correct", "accuracy"), ("error", "error"), ("wrong_selected_branch", "wrong_selected_branch"), ("wrong_value", "wrong_value"))},
            "rare": rare, "metrics": {name: means(v, group) for name, v in values.items()},
            "exact_tie_rows": int(ties[indices].sum())}


def average(tree):
    """Equal-seed descriptive means, never a pooled count of independent rows."""
    first = tree[0]
    if isinstance(first, dict):
        return {key: average([t[key] for t in tree]) for key in first}
    if first is None:
        require(all(v is None for v in tree), "Paired undefined support")
        return None
    require(all(v is not None for v in tree), "Equal-seed defined support")
    return math.fsum(tree)/len(tree)


def differences(a, b):
    if isinstance(a, dict):
        return {k: differences(a[k], b[k]) for k in a}
    require((a is None) == (b is None), "Paired metric support")
    return None if a is None else a-b


def paired_cell(a, b, choices_a, choices_b, meta, group):
    indices = group["indices"]
    ac = choices_a[indices] == meta["labels"][indices]
    bc = choices_b[indices] == meta["labels"][indices]
    return {"rows": len(indices), "count_differences": differences(a["counts"], b["counts"]),
            "metric_differences": differences(a["metrics"], b["metrics"]),
            "rare_rate_differences": {key: differences(a["rare"][key]["rate"], b["rare"][key]["rate"]) for key in a["rare"]},
            "paired": {"rows": len(indices), "wrong_to_correct": int((ac & ~bc).sum()),
                       "correct_to_wrong": int((~ac & bc).sum()), "both_correct": int((ac & bc).sum()),
                       "both_wrong": int((~ac & ~bc).sum())}}


def analyze(rows, meta, packets, arithmetic, check=lambda: None):
    cells, services = layouts(rows, meta)
    previous = np.asarray([r["previous_current_index"] for r in rows], np.int64)
    fits, choices, source_norm, raw_primary, raw_choices = {}, {}, {}, {}, {}
    for name, packet in packets.items():
        check()
        raw_primary[name], raw_choices[name] = arithmetic.score(rows, meta, packet)
    for seed in SEEDS:
        check()
        m, a, f = (packets[f"{method}-{seed}"]["log_probs"] for method in ("token_mean", "token_aligned", "flat_stratum"))
        outputs, validations, witnesses, zs = construct(m, a, meta["sizes"], previous)
        outputs["flat"], witnesses["F"], zs["F"] = normalize(f, meta["sizes"], previous)
        validations["flat"] = {"maximum_mass_error": float(np.max(np.abs(np.exp(outputs["flat"]).sum(axis=1)-1))),
                               "self_reconstruction_max_abs": None, "self_argmax_exact": True}
        require(validations["flat"]["maximum_mass_error"] <= 1e-12, "Flat float64 normalization")
        for key, method in (("M", "token_mean"), ("A", "token_aligned"), ("F", "flat_stratum")):
            source_norm[f"{method}-{seed}"] = {**witnesses[key],
                "nll_normalized_minus_original_raw": {name: means(zs[key], group) for name, group in cells.items()}}
        for name, logs in outputs.items():
            check()
            values, choice, types, ties = vectors(logs, meta)
            fit = f"{name}-{seed}"
            choices[fit] = choice
            fits[fit] = {"validation": {**validations[name], "exact_tie_rows": int(ties.sum())},
                "cells": {key: describe(values, choice, types, ties, meta, group) for key, group in cells.items()},
                "services": {service: {key: describe(values, choice, types, ties, meta, group) for key, group in strata.items()}
                             for service, strata in services.items()}}
    comparisons = {}
    for name in CONSTRUCTIONS:
        comparisons[name] = {}
        for control in CONTROLS:
            paired = {}
            for seed in SEEDS:
                check(); af, bf = f"{name}-{seed}", f"{control}-{seed}"
                a, b = fits[af], fits[bf]
                paired[str(seed)] = {
                    "cells": {key: paired_cell(a["cells"][key], b["cells"][key], choices[af], choices[bf], meta, group) for key, group in cells.items()},
                    "services": {svc: {key: paired_cell(a["services"][svc][key], b["services"][svc][key], choices[af], choices[bf], meta, group)
                                        for key, group in strata.items()} for svc, strata in services.items()}}
            comparisons[name][control] = {"seeds": paired, "mean": average(list(paired.values()))}
    tradeoffs = {}
    for seed in SEEDS:
        tradeoffs[str(seed)] = {}
        for name in CONSTRUCTIONS:
            tradeoffs[str(seed)][name] = {}
            for panel in ("all", "seen_service", "heldout_service"):
                current, aa, mm = (fits[f"{n}-{seed}"]["cells"] for n in (name, "AA", "MM"))
                retained, changed = panel+"/retained", panel+"/changed"
                tradeoffs[str(seed)][name][panel] = {
                    "retained_errors_recovered_vs_AA": aa[retained]["counts"]["error"]-current[retained]["counts"]["error"],
                    "changed_correct_gain_vs_MM": current[changed]["counts"]["correct"]-mm[changed]["counts"]["correct"],
                    "retained_rows": current[retained]["rows"], "changed_rows": current[changed]["rows"]}
    original_pairs = {control: {str(seed): arithmetic.pair_counts(meta, raw_choices[f"token_aligned-{seed}"], raw_choices[f"{control}-{seed}"])
                                for seed in SEEDS} for control in ("flat_stratum", "token_mean")}
    return {"fits": fits, "source_normalization": source_norm, "original_raw_primary": raw_primary,
            "original_raw_pairs": original_pairs,
            "means": {name: {section: average([fits[f"{name}-{s}"][section] for s in SEEDS]) for section in ("cells", "services")} for name in CONSTRUCTIONS},
            "comparisons": comparisons, "tradeoffs": {"seeds": tradeoffs, "mean": average(list(tradeoffs.values()))},
            "mean_scope": "Equal mean of three paired seeds on the same rows. Mean count fields are average counts, not independently pooled examples."}


def authenticate(args, portable, arithmetic, check):
    run, report, audit = (Path(getattr(args, name)).resolve() for name in ("run", "report", "audit"))
    paths = {"completed": run/"completed.json", "plan": run/"plan.json", "rows": run/"evaluation-rows.jsonl",
             "report_receipt": report/"receipt.json", "report_summary": report/"summary.json", "audit_receipt": audit/"receipt.json"}
    for name, path in paths.items():
        require(portable.digest(path, check) == PINS[name], "Fixed input pin: "+name)
    for name, pin in PREDICTIONS.items():
        path = run/"fits"/name/"predictions.npz"
        require(portable.digest(path, check) == pin, "Fixed prediction pin: "+name)
    audit_receipt = arithmetic.read(audit/"receipt.json")
    require(audit_receipt["status"] == "completed" and audit_receipt["agreement"] is True
            and audit_receipt["continuation_passed"] is False
            and audit_receipt["request"]["completed_sha256"] == PINS["completed"]
            and audit_receipt["request"]["report_receipt_sha256"] == PINS["report_receipt"]
            and audit_receipt["request"]["plan_sha256"] == PINS["plan"], "Matching independent audit")
    for name, item in audit_receipt["files"].items():
        portable.bind(arithmetic.safe(audit, name), item, check)
    request = argparse.Namespace(run=run, report=report, completed_sha256=PINS["completed"], report_receipt_sha256=PINS["report_receipt"])
    rows, meta, main_report, selected, summary_member, plan_pin = portable.authenticate(request, arithmetic, check)
    require(plan_pin == PINS["plan"] and selected["evaluation-rows.jsonl"]["sha256"] == PINS["rows"], "Canonical metadata pins")
    unmentioned = meta["held"] & ~meta["changed"] & (meta["target_types"] == 0)
    assigned = meta["held"] & ~meta["changed"] & (meta["target_types"] != 0)
    require(int(unmentioned.sum()) == 4032 and int(assigned.sum()) == 3209, "Declared retained support")
    return rows, meta, main_report, selected, summary_member, paths, audit_receipt


def peak_rss():
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if sys.platform == "darwin" else amount*1024)


def execute(args):
    out = Path(args.out).resolve()
    require(all(not out.is_relative_to(Path(getattr(args, name)).resolve()) and
                not Path(getattr(args, name)).resolve().is_relative_to(out) for name in ("run", "report", "audit")), "Separate output tree")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); prior_handler = None
    request = {k: str(v) for k, v in vars(args).items()}
    sources = {PORTABLE: PORTABLE_PIN, SPEC: args.spec_sha256}
    def check():
        require(time.monotonic()-started <= LIMITS["wall_seconds"], "Diagnostic wall cap")
        require(peak_rss() <= LIMITS["peak_rss_bytes"], "Diagnostic process-lifetime RSS cap")
        require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "Diagnostic output cap")
    def timeout(_sig, _frame):
        raise TimeoutError("Diagnostic 60-second wall cap")
    try:
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "No overlapping timer")
        prior_handler = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMITS["wall_seconds"])
        portable = load_portable()
        sources.update({portable.HELPER: portable.HELPER_SHA256,
                        "scripts/diagnose_dialogue_commitment.py": portable.digest(__file__, check),
                        "tests/test_diagnose_dialogue_commitment.py": portable.digest(ROOT/"tests/test_diagnose_dialogue_commitment.py", check)})
        write(out/"started.json", {"version": VERSION, "request": request, "source_sha256": sources,
            "fixed_input_sha256": PINS, "prediction_sha256": PREDICTIONS, "limits": LIMITS, "scope": SCOPE, "no_retry": True})
        require(portable.pin(args.spec_sha256) and portable.digest(ROOT/SPEC, check) == args.spec_sha256, "Prospective specification pin")
        arithmetic = portable.load_arithmetic(check)
        rows, meta, original, selected, summary_member, paths, audit_receipt = authenticate(args, portable, arithmetic, check)
        packets = {}
        for name in arithmetic.ORDER:
            check()
            with np.load(Path(args.run)/"fits"/name/"predictions.npz", allow_pickle=False) as archive:
                require(len(archive.files) == 2 and set(archive.files) == {"row_indices", "log_probs"}, "Prediction NPZ schema")
                packets[name] = {key: archive[key] for key in archive.files}
        result = analyze(rows, meta, packets, arithmetic, check)
        historical = arithmetic.continuation(result["original_raw_primary"])
        arithmetic.compare_report(result["original_raw_primary"], result["original_raw_pairs"], historical, original)
        require(historical["passed"] is False and historical["checks_passed"] == 18, "Historical failed result unchanged")
        result.update(status="completed", version=VERSION, scope=SCOPE, diagnostic_has_gate=False,
            original_campaign={"passed": False, "checks_passed": 18, "total_checks": 22},
            input_sha256=PINS, prediction_sha256=PREDICTIONS, source_sha256=sources,
            counts={"rows_per_seed": len(rows), "seeds": 3, "original_fits": 9, "constructions_per_seed": 5,
                    "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0})
        check(); portable.write(out/"summary.json", result); check()
        for name, item in selected.items():
            portable.bind(Path(args.run)/name, item, check)
        portable.bind(Path(args.report)/"summary.json", summary_member, check)
        for name, item in audit_receipt["files"].items():
            portable.bind(arithmetic.safe(Path(args.audit), name), item, check)
        for name, path in paths.items():
            require(portable.digest(path, check) == PINS[name], "End input stability: "+name)
        for name, pin in sources.items():
            require(portable.digest(ROOT/name, check) == pin, "End source stability: "+name)
        files = {p.name: {"sha256": portable.digest(p, check), "bytes": p.stat().st_size} for p in out.iterdir() if p.is_file()}
        portable.write(out/"receipt.json", {"status": "completed", "version": VERSION, "request": request,
            "source_sha256": sources, "input_sha256": PINS, "prediction_sha256": PREDICTIONS,
            "files": files, "limits": LIMITS, "scope": SCOPE, "diagnostic_has_gate": False,
            "model_calls": 0, "encoder_calls": 0, "checkpoint_deserializations": 0, "no_retry": True,
            "wall_seconds": time.monotonic()-started, "wall_scope": "Through payload hashing; completion write/return remain cap-checked",
            "process_lifetime_peak_rss_bytes": peak_rss()})
        check()
        return result
    except BaseException as error:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (out/"receipt.json").exists():
                (out/"receipt.json").rename(out/"receipt-before-error.json")
            write(out/"failed.json", {"status": "failed", "version": VERSION, "request": request,
                "source_sha256": sources, "input_sha256": PINS, "error": repr(error),
                "wall_seconds": time.monotonic()-started, "model_calls": 0, "no_retry": True})
        except BaseException as secondary:  # noqa: BLE001 - preserve original diagnostic failure
            if callable(getattr(error, "add_note", None)):
                error.add_note("Failure preservation: "+repr(secondary))
        raise
    finally:
        if prior_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "report", "audit", "spec-sha256", "out"):
        parser.add_argument("--"+key, required=True)
    execute(parser.parse_args())
