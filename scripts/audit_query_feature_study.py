"""Independent saved-distribution audit for the nonlinear feature pilot.

No producer, model, generator, Torch or optimizer imports. Neural checkpoint
bytes and source attestations are authenticated, not executed. Known-law GP
predictions are independently reconstructed by direct augmented conditioning;
block weights use joint-density differences rather than the producer's Schur
complement few-shot calculation. Generation and elapsed times are not replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp, ndtr

ROOT = Path(__file__).resolve().parents[1]
VERSION = "query-feature-audit-v1"
ARMS = ("static16", "centered16", "static32", "nystrom16", "centered_nystrom16")
CONTROLS = ("static16", "static32", "nystrom16", "centered_nystrom16")
SEEDS = (11, 23, 37)
NOISE = .0225
RTOL, ATOL = 1e-10, 1e-12
CONFIG = {
    "version": "query-feature-v1", "namespace": 441260924,
    "train_contexts": 1024, "train_queries": 2, "epochs": 16,
    "batch_contexts": 32, "learning_rate": .003, "gradient_clip": 10.,
    "fit_seeds": list(SEEDS), "arms": list(ARMS), "cohorts": 3,
    "eval_contexts": 128, "eval_queries": 4,
    "basis_points": 16, "fewshots": 4, "noise_variance": NOISE,
    "base": {"blocks": 4, "extent": 2.},
    "shift": {"blocks": 8, "extent": 3.},
    "defer_cost": .2, "wall_cap_seconds": 3600,
    "latency_repeats": 20, "latency_warmups": 3,
}
DATA_KEYS = {"bx", "by", "fx", "fy", "qx", "target", "private_selected_block"}
PREDICTION_KEYS = {"component_mean", "component_variance", "log_weights", "prob_positive", "log_prob"}
METRICS = ("regret", "nll", "brier", "mse", "defer", "negative", "positive", "always_defer_regret")
SOURCES = {
    "src/openjev/research/query_feature_models.py", "src/openjev/research/query_feature_data.py",
    "src/openjev/research/replay_evidence.py", "tests/test_query_feature_models.py",
    "tests/test_query_feature_data.py", "tests/test_replay_evidence.py", "tests/test_query_feature_study.py",
    "scripts/query_feature_study.py", "research/query-feature-protocol.md",
    "scripts/audit_query_feature_study.py", "tests/test_audit_query_feature_study.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary evidence file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def close(left, right, name):
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            and np.allclose(left, right, rtol=RTOL, atol=ATOL), name)


def compare(left, right, name="saved scalars"):
    if isinstance(right, dict):
        require(isinstance(left, dict) and left.keys() == right.keys(), name+" keys")
        for key in right:
            compare(left[key], right[key], name+"/"+key)
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), name+" list")
        for index, value in enumerate(right):
            compare(left[index], value, name+"/"+str(index))
    elif type(right) is float:
        require(type(left) in (int, float) and math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL), name)
    else:
        require(type(left) is type(right) and left == right, name)


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)) and set(saved.files) == set(keys), "exact array roster")
        return {name: saved[name].copy() for name in saved.files}


def validate_data(data, contexts, queries, blocks, extent, *, basis=16, fewshots=4):
    shapes = {"bx": (contexts, blocks, basis, 2), "by": (contexts, blocks, basis),
              "fx": (contexts, queries, fewshots, 2), "fy": (contexts, queries, fewshots),
              "qx": (contexts, queries, 2), "target": (contexts, queries),
              "private_selected_block": (contexts, queries)}
    require(set(data) == DATA_KEYS, "exact public/private data keys")
    for name, shape in shapes.items():
        value = data[name]
        dtype = "int64" if name == "private_selected_block" else "float64"
        require(isinstance(value, np.ndarray) and value.shape == shape
                and value.dtype == np.dtype(dtype) and np.isfinite(value).all(), "data shape/dtype/finite: "+name)
    for name in ("bx", "fx", "qx"):
        require((np.abs(data[name]) <= extent).all(), "registered coordinate support")
    identity = data["private_selected_block"]
    require(((identity >= 0) & (identity < blocks)).all(), "private index domain only")


def kernel(left, right):
    """Independent literal unit-amplitude, unit-length RBF law."""
    return np.exp(-.5*np.sum((left[..., :, None, :]-right[..., None, :, :])**2, axis=-1))


def gaussian_density(cholesky, values):
    standardized = np.linalg.solve(cholesky, values[..., None])[..., 0]
    dimension = values.shape[-1]
    return -.5*(dimension*math.log(2*math.pi)
                + 2*np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(-1)
                + np.sum(standardized**2, axis=-1))


def exact_reference(data, *, check=lambda: None):
    """Directly condition on archive+few-shots. Private ID and target unread.

    Vectorizes queries and blocks within a context. This retains independent
    arithmetic from the sequential archive-then-request producer reference.
    """
    bx, by, fx, fy, qx = (data[name] for name in ("bx", "by", "fx", "fy", "qx"))
    contexts, queries, fewshots = fx.shape[:3]
    blocks, basis = bx.shape[1:3]
    means = np.empty((contexts, queries, blocks), np.float64)
    variances, evidence = np.empty_like(means), np.empty_like(means)
    for context in range(contexts):
        check()
        archive_x = np.broadcast_to(bx[context], (queries, blocks, basis, 2))
        archive_y = np.broadcast_to(by[context], (queries, blocks, basis))
        request_x = np.broadcast_to(fx[context, :, None], (queries, blocks, fewshots, 2))
        request_y = np.broadcast_to(fy[context, :, None], (queries, blocks, fewshots))
        augmented_x = np.concatenate((archive_x, request_x), axis=-2)
        augmented_y = np.concatenate((archive_y, request_y), axis=-1)
        covariance = kernel(augmented_x, augmented_x) + NOISE*np.eye(basis+fewshots)
        factor = np.linalg.cholesky(covariance)
        archive_factor = np.linalg.cholesky(kernel(archive_x, archive_x)+NOISE*np.eye(basis))
        evidence[context] = (gaussian_density(factor, augmented_y)
                             - gaussian_density(archive_factor, archive_y))
        query = np.broadcast_to(qx[context, :, None, None], (queries, blocks, 1, 2))
        cross = kernel(augmented_x, query)
        whitened_cross = np.linalg.solve(factor, cross)[..., 0]
        whitened_values = np.linalg.solve(factor, augmented_y[..., None])[..., 0]
        means[context] = np.sum(whitened_cross*whitened_values, axis=-1)
        variances[context] = 1.+NOISE-np.sum(whitened_cross**2, axis=-1)
    require(np.isfinite(means).all() and np.isfinite(variances).all()
            and (variances >= NOISE-ATOL).all() and np.isfinite(evidence).all(), "finite independent GP moments")
    weights = evidence-logsumexp(evidence, axis=-1, keepdims=True)
    positive = np.sum(np.exp(weights)*ndtr(means/np.sqrt(variances)), axis=-1)
    return {"component_mean": means, "component_variance": variances,
            "log_weights": weights, "prob_positive": positive}


def validate_prediction(prediction, target, blocks):
    require(set(prediction) == PREDICTION_KEYS, "exact prediction keys")
    for name in PREDICTION_KEYS:
        value = prediction[name]
        shape = target.shape+((blocks,) if name in ("component_mean", "component_variance", "log_weights") else ())
        require(isinstance(value, np.ndarray) and value.dtype == np.float64
                and value.shape == shape and np.isfinite(value).all(), "finite prediction schema: "+name)
    means, variance, weights = (prediction[name] for name in ("component_mean", "component_variance", "log_weights"))
    require((variance >= NOISE-ATOL).all(), "predictive variance includes new observation noise")
    close(logsumexp(weights, axis=-1), np.zeros_like(target), "normalized log mixture weights")
    positive = np.sum(np.exp(weights)*ndtr(means/np.sqrt(variance)), axis=-1)
    require(((prediction["prob_positive"] >= 0) & (prediction["prob_positive"] <= 1+ATOL)).all(), "sign probability range")
    close(prediction["prob_positive"], positive, "independent mixture sign probability")
    density = logsumexp(weights-.5*(math.log(2*math.pi)+np.log(variance)
                                    +(target[..., None]-means)**2/variance), axis=-1)
    close(prediction["log_prob"], density, "independent noisy-target log density")
    return density


def score(prediction, target, oracle_positive):
    density = validate_prediction(prediction, target, prediction["component_mean"].shape[-1])
    positive = prediction["prob_positive"]
    costs = np.stack((positive, np.full_like(positive, .2), 1-positive), axis=-1)
    action = np.argmin(costs, axis=-1)
    oracle = np.stack((oracle_positive, np.full_like(positive, .2), 1-oracle_positive), axis=-1)
    regret = np.take_along_axis(oracle, action[..., None], axis=-1)[..., 0]-np.min(oracle, axis=-1)
    mean = np.sum(np.exp(prediction["log_weights"])*prediction["component_mean"], axis=-1)
    values = {"nll": -density, "regret": regret, "brier": (positive-(target > 0))**2,
              "mse": (mean-target)**2, "defer": (action == 1).astype(np.float64),
              "negative": (action == 0).astype(np.float64), "positive": (action == 2).astype(np.float64),
              "always_defer_regret": .2-np.min(oracle, axis=-1)}
    require(all(np.isfinite(value).all() for value in values.values()), "finite independently scored values")
    return {name: float(np.mean(value)) for name, value in values.items()}, {
        "context_metrics": {name: value.mean(axis=1).tolist() for name, value in values.items()},
        "action_counts": {str(a): int(np.count_nonzero(action == a)) for a in range(3)},
        "always_defer_regret": float(np.mean(.2-np.min(oracle, axis=-1))),
    }


def classification(rows):
    """Equal cohort/fit means; repeated seeds are not independent datasets."""
    means = {}
    for phase in ("base", "shift"):
        means[phase] = {}
        for arm in (*ARMS, "full_gp"):
            subset = [row for row in rows if row["phase"] == phase and row["arm"] == arm]
            require(len(subset) == (3 if arm == "full_gp" else 9), "complete mean support")
            means[phase][arm] = {metric: float(np.mean([r[metric] for r in subset])) for metric in METRICS}
    checks = []
    for phase in ("base", "shift"):
        candidate = means[phase]["centered16"]
        for control in CONTROLS:
            baseline = means[phase][control]
            wins = []
            for cohort in range(3):
                regrets = {arm: float(np.mean([r["regret"] for r in rows if r["phase"] == phase
                    and r["cohort"] == cohort and r["arm"] == arm])) for arm in ("centered16", control)}
                wins.append(regrets["centered16"] < regrets[control])
            checks.append({"phase": phase, "control": control,
                "ten_percent_regret_gain": bool(candidate["regret"] <= .9*baseline["regret"]
                                                and candidate["regret"] < baseline["regret"]),
                "nll_noninferiority": bool(candidate["nll"] <= baseline["nll"]+.02),
                "all_cohort_mean_wins": all(wins), "cohort_wins": wins})
    passed = all(r[name] for r in checks for name in
                 ("ten_percent_regret_gain", "nll_noninferiority", "all_cohort_mean_wins"))
    return {"means": means, "checks": checks,
            "gate": "CONTINUE_TO_SELECTIVE_REFINEMENT" if passed else "DO_NOT_ADVANCE_THIS_CANDIDATE",
            "fit_seeds_are_not_independent_datasets": True}


def parameter_count(arm):
    return 2 if "nystrom" in arm else 96+33*(32 if arm == "static32" else 16)


def validate_resources(rows):
    roster = {(phase, arm, seed) for phase in ("base", "shift") for seed in SEEDS for arm in ARMS}
    roster.update((phase, "full_gp", None) for phase in ("base", "shift"))
    require(len(rows) == 32 and {(r["phase"], r["arm"], r["fit_seed"]) for r in rows} == roster, "all32 latency records")
    for row in rows:
        arm, blocks = row["arm"], CONFIG[row["phase"]]["blocks"]
        if arm == "full_gp":
            require(row["cached_static"] is False and row["requests_per_archive"] == 4
                    and row["archive_array_bytes"] == blocks*16*3*8
                    and row["note"] == "Exact-law NumPy reference recomputes archive factorization for each request.",
                    "separate GP resource scope")
            require(len(row["timings"]) == 20 and all(type(v) in (int, float)
                    and math.isfinite(v) and v >= 0 for v in row["timings"]), "all GP latency repetitions")
            close(row["median_context_seconds"], np.median(row["timings"]), "recorded GP median")
            continue
        centered = arm in ("centered16", "centered_nystrom16")
        rank = 32 if arm == "static32" else 16
        require(row["requests_per_archive"] == 4 and row["cached_static"] is (not centered)
                and row["archive_array_bytes"] == blocks*16*3*8
                and row["parameter_bytes"] == parameter_count(arm)*8
                and row["buffer_bytes"] == (256 if "nystrom" in arm else 0), "resource schema")
        expected_cache = 0 if centered else 8*blocks*(16*rank+2*rank*rank+2*rank)
        require(row["cache_tensor_bytes"] == expected_cache, "all five cache tensor bytes counted")
        times = row["timings"]
        require(len(times) == 20, "all fixed latency repetitions retained")
        for item in times:
            require(set(item) == {"archive_seconds", "queries_seconds", "total_seconds"}
                    and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in item.values()), "finite nonnegative elapsed scopes")
            close(item["total_seconds"], item["archive_seconds"]+item["queries_seconds"], "additive latency scope")
        close(row["median_context_seconds"], np.median([t["total_seconds"] for t in times]), "recorded median")


def authenticate(folder):
    """All source and opaque payload joins precede any scientific array read."""
    folder = Path(folder).resolve()
    plan = read(folder/"registration.json")
    require(plan["config"] == CONFIG, "fixed scientific configuration")
    require(set(plan["sources"]) == SOURCES, "exact11 registered source files")
    for name, pin in plan["sources"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "relative source path")
        require(descriptor(ROOT/name)["sha256"] == descriptor(folder/"source"/name)["sha256"] == pin, "frozen source/snapshot identity")
    require({str(p.relative_to(folder/"source")) for p in (folder/"source").rglob("*") if p.is_file()}
            == set(plan["sources"]), "complete snapshot")
    environment = plan["environment"]
    require(environment["python"] == sys.version and environment["numpy"] == np.__version__
            and environment["scipy"] == scipy.__version__ and environment["platform"] == platform.platform()
            and environment["machine"] == platform.machine() and environment["threads"] == 1,
            "registered audit runtime")
    manifest = read(folder/"manifest.json")
    actual = {str(p.relative_to(folder)): descriptor(p) for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
    require(actual == manifest, "complete immutable producer inventory")
    receipt = read(folder/"run-receipt.json")
    require(receipt["state"] == "EXITED" and receipt["exit_code"] == 0
            and math.isfinite(receipt["elapsed_seconds"]) and 0 < receipt["elapsed_seconds"] <= 3600,
            "original successful producer receipt")
    stems = [f"{arm}-{seed}" for seed in SEEDS for arm in ARMS]
    expected = {"registration.json", "started.json", "train.npz", "fits.json", "checkpoint-barrier.json",
                "metrics.json", "resources.json", "summary.json", "run-receipt.json"}
    expected.update(name+suffix for name in stems for suffix in (".pt", "-fit.json"))
    for phase in ("base", "shift"):
        for cohort in range(3):
            stem = f"{phase}-{cohort}"
            expected.update((stem+"-data.npz", stem+"-full_gp.npz"))
            expected.update(stem+"-"+name+".npz" for name in stems)
    require({name for name in manifest if not name.startswith("source/")} == expected, "complete fixed raw payload roster")
    return {"registration": descriptor(folder/"registration.json"), "manifest": descriptor(folder/"manifest.json"),
            "receipt": descriptor(folder/"run-receipt.json"), "sources": plan["sources"], "environment": environment,
            "payload_files": len(expected), "producer_seconds": receipt["elapsed_seconds"]}


def fit_metadata(folder):
    fits = read(folder/"fits.json")
    require(len(fits) == 15 and [(r["arm"], r["fit_seed"]) for r in fits]
            == [(arm, seed) for seed in SEEDS for arm in ARMS], "all15final fits in frozen order")
    checkpoints = {}
    for row in fits:
        arm, seed = row["arm"], row["fit_seed"]
        require(read(folder/f"{arm}-{seed}-fit.json") == row, "individual/combined fit journal join")
        require(row["parameters"] == parameter_count(arm) and row["updates"] == 512
                and row["request_exposures"] == 32768
                and math.isfinite(row["train_seconds"]) and row["train_seconds"] >= 0, "fixed fit counts and duration")
        require(len(row["curve"]) == 16, "all16epochs retained")
        for epoch, point in enumerate(row["curve"], 1):
            require(point["epoch"] == epoch and math.isfinite(point["train_nll"])
                    and math.isfinite(point["mean_gradient_norm_before_clip"])
                    and point["mean_gradient_norm_before_clip"] >= 0, "finite epoch journal")
        checkpoints[f"{arm}-{seed}.pt"] = descriptor(folder/f"{arm}-{seed}.pt")
    barrier = read(folder/"checkpoint-barrier.json")
    require(set(barrier) == {"checkpoints", "recorded_unix", "evaluation_generation_started"}
            and barrier["checkpoints"] == {name: item["sha256"] for name, item in checkpoints.items()}
            and type(barrier["recorded_unix"]) in (int, float) and math.isfinite(barrier["recorded_unix"])
            and barrier["recorded_unix"] >= read(folder/"started.json")["started_unix"]
            and barrier["evaluation_generation_started"] is False, "saved all-final pre-evaluation barrier")
    return {"fits": fits, "checkpoints": checkpoints,
            "updates_per_fit": 512, "request_exposures_per_fit": 32768,
            "training_scope": "Counts, curves and ordering authenticate producer/source attestations; no optimizer or checkpoint inference replay."}


def audit(folder, output, *, check=lambda: None):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(folder), "exclusive audit output outside immutable producer folder")
    admission = authenticate(folder)
    fit_checks = fit_metadata(folder)
    train = load_arrays(folder/"train.npz", DATA_KEYS)
    validate_data(train, 1024, 2, 4, 2.)
    del train
    saved_rows = read(folder/"metrics.json")
    roster = [(phase, cohort, arm, seed) for phase in ("base", "shift") for cohort in range(3)
              for arm, seed in [("full_gp", None), *((a, s) for s in SEEDS for a in ARMS)]]
    require(len(saved_rows) == 96 and [(r["phase"], r["cohort"], r["arm"], r["fit_seed"]) for r in saved_rows]
            == roster, "all96 metric groups in frozen order")
    rows, evidence, cursor = [], [], 0
    counts = {"npz_decodes": 1, "data_files": 1, "prediction_files": 0, "checkpoint_files_hashed": 15,
              "checkpoint_decodes": 0, "independent_gp_contexts": 0, "independent_gp_queries": 0,
              "scored_query_predictions": 0, "model_calls": 0, "optimizer_calls": 0,
              "generator_calls": 0, "rng_replays": 0}
    for phase in ("base", "shift"):
        for cohort in range(3):
            check()
            stem = f"{phase}-{cohort}"
            blocks = CONFIG[phase]["blocks"]
            data = load_arrays(folder/(stem+"-data.npz"), DATA_KEYS)
            validate_data(data, 128, 4, blocks, CONFIG[phase]["extent"])
            counts["npz_decodes"] += 1; counts["data_files"] += 1
            reference = load_arrays(folder/(stem+"-full_gp.npz"), PREDICTION_KEYS)
            independent = exact_reference(data, check=check)
            for name, value in independent.items():
                close(reference[name], value, "independent augmented-GP reference: "+name)
            counts["independent_gp_contexts"] += 128; counts["independent_gp_queries"] += 512
            for arm, seed in [("full_gp", None), *((a, s) for s in SEEDS for a in ARMS)]:
                prediction = reference if arm == "full_gp" else load_arrays(folder/f"{stem}-{arm}-{seed}.npz", PREDICTION_KEYS)
                counts["npz_decodes"] += 1; counts["prediction_files"] += 1
                values, detail = score(prediction, data["target"], reference["prob_positive"])
                saved = saved_rows[cursor]; cursor += 1
                require(type(saved["evaluation_seconds"]) in (int, float)
                        and math.isfinite(saved["evaluation_seconds"]) and saved["evaluation_seconds"] >= 0, "finite inference duration")
                row = {"phase": phase, "cohort": cohort, "arm": arm, "fit_seed": seed,
                       "evaluation_seconds": saved["evaluation_seconds"], **values}
                compare(saved, row, "independent metric row")
                rows.append(row)
                evidence.append({"phase": phase, "cohort": cohort, "arm": arm, "fit_seed": seed, **detail})
                counts["scored_query_predictions"] += 512
    result = classification(rows)
    compare(read(folder/"summary.json"), result, "independent continuation rule")
    require(read(folder/"run-receipt.json")["gate"] == result["gate"], "producer receipt outcome join")
    resources = read(folder/"resources.json")
    validate_resources(resources)
    require(counts["npz_decodes"] == 103 and counts["prediction_files"] == 96
            and counts["independent_gp_queries"] == 3072 and counts["scored_query_predictions"] == 49152,
            "complete fixed scoring and reconstruction work")
    require(authenticate(folder) == admission, "all original evidence unchanged after audit")
    record = {"version": VERSION, "agreement": True, "config": CONFIG, "result": result,
              "rows": rows, "context_evidence": evidence, "fit_checks": fit_checks,
              "resources": resources, "counts": counts, "admission": admission,
              "tolerance": {"relative": RTOL, "absolute": ATOL},
              "scope": "Independent saved Gaussian-mixture metrics, direct full-GP conditioning and fixed-rule reconstruction. Neural predictions are not independently executed. Raw data support is checked, not stochastic generation. Checkpoints are opaque hash checks only. Training order/counts, historical pre-DEV ordering and wall times remain source/producer attestations. Original process closure must be checked separately before publication."}
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.out, args.output)
    print(json.dumps({"agreement": result["agreement"], "gate": result["result"]["gate"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
