"""Independent dense-covariance audit of saved residual-memory predictions.

No producer, residual-model, generator, Torch or checkpoint-deserialization
imports. Shared frozen metric utilities are pure saved-array computations.
Prediction reconstruction conditions dense covariance matrices, independently
of the producer's rank-16 weighted sufficient statistics. Timing and historical
generation order remain attestations, not numerical replays.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from audit_query_feature_study import (
    ATOL,
    DATA_KEYS,
    METRICS,
    NOISE,
    PREDICTION_KEYS,
    RTOL,
    close,
    compare,
    descriptor,
    load_arrays,
    read,
    require,
    score,
    validate_data,
    validate_prediction,
)
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp, ndtr

ROOT = Path(__file__).resolve().parents[1]
VERSION = "residual-memory-audit-v1"
MODES = ("sor", "query_only", "fic", "full")
SEEDS = (11, 23, 37)
JITTER = 1e-6
RESIDUAL_TOLERANCE = 1e-12
CONFIG = {
    "version": "residual-memory-v1", "namespace": 442260924,
    "fit_seeds": list(SEEDS), "modes": list(MODES), "cohorts": 3,
    "contexts": 128, "queries": 4, "fewshots": 4,
    "noise_variance": NOISE, "defer_cost": .2,
    "populations": {
        "base": {"blocks": 4, "basis_points": 16, "extent": 2., "offset": 100},
        "shift": {"blocks": 8, "basis_points": 16, "extent": 3., "offset": 200},
        "long": {"blocks": 4, "basis_points": 128, "extent": 2., "offset": 300}},
    "latency_warmups": 3, "latency_repeats": 20, "wall_cap_seconds": 1200,
    "training_updates": 0, "checkpoint_selection": "all three parent static Nyström final fits",
}
SOURCES = {
    "src/openjev/research/query_feature_data.py", "src/openjev/research/residual_memory.py",
    "scripts/residual_memory_study.py", "scripts/audit_residual_memory_study.py",
    "scripts/audit_query_feature_study.py", "tests/test_residual_memory.py",
    "tests/test_residual_memory_study.py", "tests/test_audit_residual_memory_study.py",
    "research/residual-memory-protocol.md",
}
PARENT = ROOT/"output/query-feature-v1"


def rbf(left, right, length, amplitude):
    """Literal dense kernel; amplitude is standard deviation."""
    return amplitude**2*np.exp(-.5*np.sum(
        ((left[:, None, :]-right[None, :, :])/length)**2, axis=-1))


def residual_diagonal(kernel_diagonal, approximation_diagonal):
    residual = kernel_diagonal-approximation_diagonal
    require(np.isfinite(residual).all() and (residual >= -RESIDUAL_TOLERANCE).all(),
            "nonnegative residual within declared roundoff guard")
    # The producer declares this roundoff-only floor and counts it. This is not
    # a fitted correction, added jitter, or a repair of a materially negative R.
    return np.maximum(residual, 0.)


def reconstruct(data, length, amplitude, mode, *, check=lambda: None):
    """Target-free public prediction under a directly constructed covariance.

    Cache each dense archive Cholesky across its requests. FIC uses independent
    residuals per observation event, even when distinct events share inputs.
    query_only changes only the final query variance, after SoR conditioning.
    """
    require(mode in MODES and type(length) in (float, int)
            and type(amplitude) in (float, int) and math.isfinite(length)
            and math.isfinite(amplitude) and length > 0 and amplitude > 0,
            "declared mode and positive kernel parameters")
    bx, by, fx, fy, qx = (data[name] for name in ("bx", "by", "fx", "fy", "qx"))
    contexts, blocks, basis = bx.shape[:3]
    queries, fewshots = fx.shape[1:3]
    shape = (contexts, queries, blocks)
    means, variances, evidence = (np.empty(shape, np.float64) for _ in range(3))
    grid = np.array([(x, y) for x in np.linspace(-2., 2., 4)
                    for y in np.linspace(-2., 2., 4)], np.float64)
    grid_factor = None
    if mode != "full":
        grid_factor = cho_factor(rbf(grid, grid, length, amplitude)+JITTER*np.eye(16), lower=True)

    for context in range(contexts):
        check()
        for block in range(blocks):
            archive_x, archive_y = bx[context, block], by[context, block]
            if mode == "full":
                archive_cov = rbf(archive_x, archive_x, length, amplitude)
                archive_grid = None
            else:
                archive_grid = rbf(archive_x, grid, length, amplitude)
                archive_cov = archive_grid@cho_solve(grid_factor, archive_grid.T)
                if mode == "fic":
                    archive_cov = archive_cov + np.diag(residual_diagonal(
                        amplitude**2, np.diag(archive_cov)))
            archive_factor = cho_factor(archive_cov+NOISE*np.eye(basis), lower=True)
            alpha = cho_solve(archive_factor, archive_y)
            for request in range(queries):
                # The last member of U is the noisy query. All likelihoods use
                # only the leading F labels, never its target or private ID.
                request_x = np.concatenate((fx[context, request], qx[context, request][None]), axis=0)
                if mode == "full":
                    cross = rbf(request_x, archive_x, length, amplitude)
                    covariance = rbf(request_x, request_x, length, amplitude)
                    query_residual = 0.
                else:
                    request_grid = rbf(request_x, grid, length, amplitude)
                    projected = cho_solve(grid_factor, request_grid.T)
                    cross = request_grid@cho_solve(grid_factor, archive_grid.T)
                    covariance = request_grid@projected
                    diagonal = residual_diagonal(amplitude**2, np.diag(covariance))
                    query_residual = diagonal[-1]
                    if mode == "fic":
                        covariance = covariance+np.diag(diagonal)
                conditional_mean = cross@alpha
                conditional_cov = covariance+NOISE*np.eye(fewshots+1)
                conditional_cov -= cross@cho_solve(archive_factor, cross.T)
                request_factor = cho_factor(conditional_cov[:fewshots, :fewshots], lower=True)
                delta = fy[context, request]-conditional_mean[:fewshots]
                solved = cho_solve(request_factor, delta)
                evidence[context, request, block] = -.5*(fewshots*math.log(2*math.pi)
                    +2*np.log(np.diag(request_factor[0])).sum()+delta@solved)
                query_cross = conditional_cov[-1, :fewshots]
                means[context, request, block] = conditional_mean[-1]+query_cross@solved
                variance = conditional_cov[-1, -1]-query_cross@cho_solve(request_factor, query_cross)
                if mode == "query_only":
                    variance += query_residual
                variances[context, request, block] = variance
    require(np.isfinite(means).all() and np.isfinite(variances).all()
            and (variances >= NOISE-ATOL).all() and np.isfinite(evidence).all(),
            "finite dense predictive law including observation noise")
    weights = evidence-logsumexp(evidence, axis=-1, keepdims=True)
    positive = np.sum(np.exp(weights)*ndtr(means/np.sqrt(variances)), axis=-1)
    return {"component_mean": means, "component_variance": variances,
            "log_weights": weights, "prob_positive": positive}


def density(prediction, target):
    return logsumexp(prediction["log_weights"]-.5*(math.log(2*math.pi)
        +np.log(prediction["component_variance"])
        +(target[..., None]-prediction["component_mean"])**2/prediction["component_variance"]), axis=-1)


def compare_prediction(saved, reconstructed, target):
    validate_prediction(saved, target, saved["component_mean"].shape[-1])
    for name, value in reconstructed.items():
        close(saved[name], value, "independent dense "+name)
    close(saved["log_prob"], density(reconstructed, target), "independent dense mixture density")


def paired_diagnostics(prediction, reference):
    """Unselected all-request disagreement with the supplied-law mixture."""
    weight_error = np.exp(prediction["log_weights"])-np.exp(reference["log_weights"])
    return {"mean_weight_l1": float(np.abs(weight_error).sum(-1).mean()),
            "mean_sign_probability_absolute_error": float(np.abs(
                prediction["prob_positive"]-reference["prob_positive"]).mean())}


def identities():
    return [(phase, cohort, mode, seed)
            for phase in CONFIG["populations"] for cohort in range(3)
            for mode, seed in [("true_gp", None), *((m, s) for s in SEEDS for m in MODES)]]


def classification(rows):
    require(len(rows) == 117 and {(r["phase"], r["cohort"], r["mode"], r["fit_seed"])
            for r in rows} == set(identities()), "complete 117-row classification roster")
    means = {}
    for phase in CONFIG["populations"]:
        means[phase] = {}
        for mode in (*MODES, "true_gp"):
            group = [r for r in rows if r["phase"] == phase and r["mode"] == mode]
            means[phase][mode] = {name: float(np.mean([r[name] for r in group])) for name in METRICS}
    checks = []
    for phase in CONFIG["populations"]:
        candidate, baseline = means[phase]["fic"], means[phase]["sor"]
        checks.append({"name": phase+"_nll_noninferior", "passed": candidate["nll"] <= baseline["nll"]+.02})
        if phase != "shift":
            checks.append({"name": phase+"_regret_preserved",
                           "passed": candidate["regret"] <= 1.05*baseline["regret"]+1e-6})
    candidate, baseline = means["shift"]["fic"], means["shift"]["sor"]
    checks.extend([
        {"name": "shift_regret_25_percent_lower", "passed": candidate["regret"] <= .75*baseline["regret"]},
        {"name": "shift_nll_0.1_nat_lower", "passed": candidate["nll"] <= baseline["nll"]-.1},
        {"name": "shift_beats_always_defer", "passed": candidate["regret"] < candidate["always_defer_regret"]}])
    for cohort in range(3):
        regrets = {mode: float(np.mean([r["regret"] for r in rows if r["phase"] == "shift"
                    and r["cohort"] == cohort and r["mode"] == mode])) for mode in ("fic", "sor")}
        checks.append({"name": f"shift_cohort_{cohort}_regret_win", "passed": regrets["fic"] < regrets["sor"]})
    return {"means": means, "checks": checks,
            "gate": "RESIDUAL_CONTROL_QUALIFIED" if all(r["passed"] for r in checks)
                    else "RESIDUAL_CONTROL_NOT_QUALIFIED",
            "scope": "Established covariance approximation diagnostic; no novel architecture or parent rescue."}


def authenticate(folder):
    registration = read(folder/"registration.json")
    require(registration["config"] == CONFIG and set(registration["sources"]) == SOURCES,
            "registered configuration and exact source roster")
    for name, pin in registration["sources"].items():
        require(descriptor(ROOT/name) == pin == descriptor(folder/"source"/name), "source/snapshot identity")
    require({str(p.relative_to(folder/"source")) for p in (folder/"source").rglob("*") if p.is_file()}
            == SOURCES, "exact source snapshot roster")
    environment = registration["environment"]
    require(environment["python"] == sys.version and environment["numpy"] == np.__version__
            and environment["scipy"] == scipy.__version__ and environment["platform"] == platform.platform()
            and environment["threads"] == 1, "registered audit numeric runtime")
    lineage = registration["parent"]
    require(set(lineage) == {"manifest", "registration", "checkpoints"}
            and lineage["manifest"] == descriptor(PARENT/"manifest.json")
            and lineage["registration"] == descriptor(PARENT/"registration.json"), "parent registration/manifest pins")
    parent_manifest = read(PARENT/"manifest.json")
    for name, pin in parent_manifest.items():
        require(descriptor(PARENT/name) == pin, "opaque original parent payload")
    parent_receipt = read(PARENT/"run-receipt.json")
    require(parent_receipt["state"] == "EXITED" and parent_receipt["exit_code"] == 0,
            "original parent receipt closed")
    checkpoint_names = {f"nystrom16-{seed}.pt" for seed in SEEDS}
    require(set(lineage["checkpoints"]) == checkpoint_names, "all three inherited checkpoints")
    for name, pin in lineage["checkpoints"].items():
        require(pin == parent_manifest[name] == descriptor(PARENT/name) == descriptor(folder/name),
                "opaque inherited checkpoint identity")
    manifest = read(folder/"manifest.json")
    actual = {str(p.relative_to(folder)): descriptor(p) for p in folder.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    require(manifest == actual, "complete producer file inventory")
    raw = {"registration.json", "started.json", "kernels.json", "metrics.json", "resources.json",
           "diagnostics.json", "pairwise.json", "summary.json", "run-receipt.json", *checkpoint_names}
    for phase in CONFIG["populations"]:
        for cohort in range(3):
            raw.add(f"{phase}-{cohort}-data.npz")
    for phase, cohort, mode, seed in identities():
        raw.add(f"{phase}-{cohort}-{mode}-{seed}.npz")
    require(set(manifest) == raw | {"source/"+name for name in SOURCES}, "exact 138 raw-file roster")
    receipt = read(folder/"run-receipt.json")
    require(receipt["state"] == "EXITED" and receipt["exit_code"] == 0
            and receipt["training_updates"] == 0 and type(receipt["elapsed_seconds"]) in (int, float)
            and math.isfinite(receipt["elapsed_seconds"]) and 0 < receipt["elapsed_seconds"] <= 1200,
            "closed successful producer receipt and unchanged zero-training scope")
    return {"registration": descriptor(folder/"registration.json"), "manifest": descriptor(folder/"manifest.json"),
            "receipt": descriptor(folder/"run-receipt.json"), "sources": registration["sources"],
            "parent": lineage, "raw_files": len(raw), "producer_seconds": receipt["elapsed_seconds"]}


def kernel_records(folder):
    records = read(folder/"kernels.json")
    require(isinstance(records, list) and len(records) == 3
            and [r["fit_seed"] for r in records] == list(SEEDS), "all inherited kernel records")
    for row in records:
        require(set(row) == {"fit_seed", "length", "amplitude", "checkpoint", "checkpoint_descriptor"},
                "exact extracted kernel schema")
        require(row["checkpoint"] == f"nystrom16-{row['fit_seed']}.pt"
                and row["checkpoint_descriptor"] == descriptor(folder/row["checkpoint"]),
                "extraction record bound to original checkpoint bytes")
        require(all(type(row[name]) in (int, float) and math.isfinite(row[name]) and row[name] > 0
                    for name in ("length", "amplitude")), "positive finite extracted kernel")
    return {r["fit_seed"]: r for r in records}


def cache_bytes(mode, blocks, basis):
    return 8*blocks*(basis*basis+3*basis) if mode in ("full", "true_gp") else 8*(32+256+blocks*(256+16))


def duration(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name)


def validate_resources(rows):
    expected = {(phase, mode, seed) for phase, cohort, mode, seed in identities() if cohort == 0}
    require(len(rows) == 39 and {(r["phase"], r["mode"], r["fit_seed"]) for r in rows} == expected,
            "all 39 resource groups")
    for row in rows:
        require(set(row) == {"phase", "mode", "fit_seed", "timings", "median_context_seconds",
                "cache_array_bytes", "input_archive_bytes", "queries_per_archive"}, "resource row schema")
        population = CONFIG["populations"][row["phase"]]
        k, n = population["blocks"], population["basis_points"]
        require(row["queries_per_archive"] == 4 and row["cache_array_bytes"] == cache_bytes(row["mode"], k, n)
                and row["input_archive_bytes"] == 24*k*n, "all retained cache arrays and separate public archive bytes")
        require(len(row["timings"]) == 20, "all registered timing repetitions")
        for timing in row["timings"]:
            require(set(timing) == {"archive_seconds", "query_seconds", "total_seconds"}, "timing schema")
            for name, value in timing.items():
                duration(value, name)
            close(timing["total_seconds"], timing["archive_seconds"]+timing["query_seconds"], "complete timing scope")
        close(row["median_context_seconds"], np.median([r["total_seconds"] for r in row["timings"]]), "timing median")


def validate_diagnostics(rows):
    require(len(rows) == 117 and {(r["phase"], r["cohort"], r["mode"], r["fit_seed"])
            for r in rows} == set(identities()), "all 117 diagnostic groups")
    for row in rows:
        require(set(row) == {"phase", "cohort", "mode", "fit_seed", "archive_seconds", "query_seconds",
                            "context_diagnostics"}, "diagnostic group schema")
        duration(row["archive_seconds"], "archive duration")
        duration(row["query_seconds"], "query duration")
        population = CONFIG["populations"][row["phase"]]
        blocks, basis, mode = population["blocks"], population["basis_points"], row["mode"]
        require(len(row["context_diagnostics"]) == 128, "all context diagnostics")
        for context in row["context_diagnostics"]:
            require(set(context) == {"archive", "queries", "cache_array_bytes"}
                    and context["cache_array_bytes"] == cache_bytes(mode, blocks, basis)
                    and len(context["queries"]) == 4, "all request and cache diagnostics")
            expected_counts = [blocks*basis if mode == "fic" else 0]+[5 if mode == "fic" else 1
                                if mode == "query_only" else 0]*4
            for record, count in zip([context["archive"], *context["queries"]], expected_counts, strict=True):
                require(set(record) == {"residual_points", "minimum_raw_residual", "residual_floor_count"}
                        and type(record["residual_points"]) is int and record["residual_points"] == count
                        and type(record["residual_floor_count"]) is int
                        and 0 <= record["residual_floor_count"] <= count, "residual computation scope/counts")
                minimum = record["minimum_raw_residual"]
                if count == 0:
                    require(minimum is None and record["residual_floor_count"] == 0, "no unperformed residual computation")
                else:
                    require(type(minimum) in (float, int) and math.isfinite(minimum)
                            and minimum >= -RESIDUAL_TOLERANCE
                            and ((minimum < 0) == (record["residual_floor_count"] > 0)), "declared residual floor bound")


def audit(folder, output=None, *, check=lambda: None):
    started = time.perf_counter()
    folder = Path(folder).resolve()
    if output is not None:
        output = Path(output).resolve()
        require(not output.exists() and not output.is_relative_to(folder), "exclusive audit outside producer folder")
    check()
    admission = authenticate(folder)
    kernels = kernel_records(folder)
    saved_rows = read(folder/"metrics.json")
    require(len(saved_rows) == 117 and [(r["phase"], r["cohort"], r["mode"], r["fit_seed"])
            for r in saved_rows] == identities(), "exact complete saved metric order")
    rows, evidence, pairwise = [], [], []
    for phase, population in CONFIG["populations"].items():
        for cohort in range(3):
            check()
            stem = f"{phase}-{cohort}"
            data = load_arrays(folder/f"{stem}-data.npz", DATA_KEYS)
            validate_data(data, 128, 4, population["blocks"], population["extent"],
                          basis=population["basis_points"])
            group_predictions = {}
            reference = reconstruct(data, 1., 1., "full", check=check)
            reference_positive = reference["prob_positive"]
            group_roster = [("true_gp", None), *((mode, seed) for seed in SEEDS for mode in MODES)]
            for mode, seed in group_roster:
                pred = load_arrays(folder/f"{stem}-{mode}-{seed}.npz", PREDICTION_KEYS)
                exact = reference if mode == "true_gp" else reconstruct(data, kernels[seed]["length"],
                    kernels[seed]["amplitude"], mode, check=check)
                compare_prediction(pred, exact, data["target"])
                values, context = score(pred, data["target"], reference_positive)
                identity = {"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed}
                row = identity | values
                compare(saved_rows[len(rows)], row, "independent saved metrics")
                rows.append(row)
                evidence.append(identity | context)
                group_predictions[mode, seed] = pred
            for seed in SEEDS:
                for mode in MODES[:-1]:
                    pairwise.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed,
                        **paired_diagnostics(group_predictions[mode, seed], group_predictions["full", seed])})
    compare(read(folder/"pairwise.json"), pairwise, "all same-fitted-kernel comparisons")
    result = classification(rows)
    compare(read(folder/"summary.json"), result, "unchanged registered eleven-condition rule")
    require(read(folder/"run-receipt.json")["gate"] == result["gate"], "producer gate receipt")
    resources, diagnostics = read(folder/"resources.json"), read(folder/"diagnostics.json")
    validate_resources(resources)
    validate_diagnostics(diagnostics)
    check()
    require(authenticate(folder) == admission, "all input identities unchanged after independent audit")
    report = {"version": VERSION, "agreement": True, "independent_predictions_agree": True,
        "result": result, "rows": rows, "context_evidence": evidence, "pairwise": pairwise,
        "resources": resources, "diagnostics": diagnostics, "kernels": list(kernels.values()),
        "admission": admission, "tolerance": {"rtol": RTOL, "atol": ATOL},
        "counts": {"npz_decodes": 126, "array_loads": 648, "checkpoint_decodes": 0,
            "evaluation_contexts": 1152, "public_requests": 4608, "prediction_groups": 117,
            "scored_query_distributions": 59904, "metric_rows": 117, "paired_rows": 81,
            "resource_rows": 39, "model_calls": 0, "generator_calls": 0, "optimizer_calls": 0},
        "limitations": [
            "Kernel scalar extraction is source/producer evidence bound to opaque original checkpoint bytes; no Torch deserialization.",
            "Timings, warmups, generation order and roundoff-floor operation counts are producer attestations; no historical execution replay.",
            "Residual diagnostic domains and operation scope are checked; dense predictions are independently reconstructed.",
            "External original process closure remains required for publication; the audit's own output cannot prove its process closure.",
            "Established fixed-kernel approximation diagnostic only; no new architecture or rescue of the parent candidate."],
        "seconds": time.perf_counter()-started}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder, args.output)
    print(json.dumps({"agreement": result["agreement"], "gate": result["result"]["gate"],
                      "counts": result["counts"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
