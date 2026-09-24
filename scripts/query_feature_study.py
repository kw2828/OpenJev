#!/usr/bin/env python3
"""Registered finite-budget nonlinear feature pilot. No hyperparameter search."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from openjev.research.query_feature_data import exact_gp_mixture, generate_contexts, reference_predict
from openjev.research.query_feature_models import NystromFeatureModel, QueryFeatureModel

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("static16", "centered16", "static32", "nystrom16", "centered_nystrom16")
SEEDS = (11, 23, 37)
CONFIG = {
    "version": "query-feature-v1", "namespace": 441260924,
    "train_contexts": 1024, "train_queries": 2, "epochs": 16,
    "batch_contexts": 32, "learning_rate": .003, "gradient_clip": 10.,
    "fit_seeds": SEEDS, "arms": ARMS, "cohorts": 3,
    "eval_contexts": 128, "eval_queries": 4,
    "basis_points": 16, "fewshots": 4, "noise_variance": .0225,
    "base": {"blocks": 4, "extent": 2.},
    "shift": {"blocks": 8, "extent": 3.},
    "defer_cost": .2, "wall_cap_seconds": 3600,
    "latency_repeats": 20, "latency_warmups": 3,
}
SOURCES = (
    "src/openjev/research/query_feature_models.py",
    "src/openjev/research/query_feature_data.py",
    "src/openjev/research/replay_evidence.py",
    "tests/test_query_feature_models.py", "tests/test_query_feature_data.py",
    "tests/test_replay_evidence.py", "tests/test_query_feature_study.py",
    "scripts/query_feature_study.py", "research/query-feature-protocol.md",
    "scripts/audit_query_feature_study.py", "tests/test_audit_query_feature_study.py",
)
PUBLIC = ("bx", "by", "fx", "fy", "qx")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def make_model(arm, seed):
    if arm not in ARMS:
        raise ValueError("unknown frozen arm")
    if "nystrom" in arm:
        return NystromFeatureModel(centered=arm == "centered_nystrom16")
    return QueryFeatureModel(rank=32 if arm == "static32" else 16,
                             centered=arm == "centered16", seed=seed)


def public_batch(data, indices):
    """Flatten requests, repeating only their own public archive."""
    c = len(indices)
    q = data["qx"].shape[1]
    arrays = {}
    for name in PUBLIC:
        selected = data[name][indices]
        if name in ("bx", "by"):
            selected = np.repeat(selected[:, None], q, axis=1)
        arrays[name] = torch.from_numpy(np.ascontiguousarray(selected.reshape(c*q, *selected.shape[2:])))
    return arrays


def metrics(prediction, target, reference_positive, defer_cost=.2):
    """Score stored predictions; reference integrates unknown block identity."""
    p = np.asarray(prediction["prob_positive"])
    rp = np.asarray(reference_positive)
    if p.shape != target.shape or rp.shape != target.shape:
        raise ValueError("matched query shapes required")
    costs = np.stack((p, np.full_like(p, defer_cost), 1-p), axis=-1)
    action = costs.argmin(axis=-1)
    true_costs = np.stack((rp, np.full_like(rp, defer_cost), 1-rp), axis=-1)
    regret = np.take_along_axis(true_costs, action[..., None], axis=-1)[..., 0] - true_costs.min(-1)
    weights = np.exp(prediction["log_weights"])
    mean = (weights * prediction["component_mean"]).sum(-1)
    return {"nll": -np.asarray(prediction["log_prob"]), "regret": regret,
            "brier": (p-(target > 0))**2, "mse": (mean-target)**2,
            "defer": (action == 1).astype(np.float64),
            "negative": (action == 0).astype(np.float64),
            "positive": (action == 2).astype(np.float64),
            "always_defer_regret": defer_cost - true_costs.min(-1)}


def register(out):
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source"
    hashes = {}
    for relative in SOURCES:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        hashes[relative] = digest(source)
    record = {"config": CONFIG, "sources": hashes,
              "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                              "torch": torch.__version__, "platform": platform.platform(),
                              "machine": platform.machine(), "threads": 1}}
    write_json(out / "registration.json", record)
    print(json.dumps({"registered": str(out), "sha256": digest(out / "registration.json")}), flush=True)


def verify_sources(out):
    reg = json.loads((out / "registration.json").read_text())
    if reg["config"] != json.loads(json.dumps(CONFIG)):
        raise ValueError("configuration differs from registration")
    if set(reg["sources"]) != set(SOURCES):
        raise ValueError("registration source roster differs")
    for relative, expected in reg["sources"].items():
        if digest(ROOT / relative) != expected or digest(out / "source" / relative) != expected:
            raise ValueError("source changed: " + relative)


def train(out, data, arm, seed):
    model = make_model(arm, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    rng = np.random.default_rng(CONFIG["namespace"] + 10000 + seed)
    curve = []
    started = time.perf_counter()
    model.train()
    updates = 0
    try:
        for epoch in range(CONFIG["epochs"]):
            indices = rng.permutation(CONFIG["train_contexts"])
            loss_sum, norm_sum, examples = 0., 0., 0
            for start in range(0, len(indices), CONFIG["batch_contexts"]):
                ix = indices[start:start+CONFIG["batch_contexts"]]
                batch = public_batch(data, ix)
                target = torch.from_numpy(np.ascontiguousarray(data["target"][ix].reshape(-1)))
                optimizer.zero_grad(set_to_none=True)
                loss = -model(**batch).log_prob(target).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"],
                                                       error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(target)
                norm_sum += float(norm) * len(target)
                examples += len(target)
            curve.append({"epoch": epoch+1, "train_nll": loss_sum/examples,
                          "mean_gradient_norm_before_clip": norm_sum/examples})
            print(json.dumps({"arm": arm, "fit_seed": seed, **curve[-1]}), flush=True)
    except BaseException as exc:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                   out / f"{arm}-{seed}-partial.pt")
        write_json(out / f"{arm}-{seed}-partial.json", {
            "arm": arm, "fit_seed": seed, "completed_updates": updates,
            "completed_epochs": curve, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    elapsed = time.perf_counter() - started
    torch.save(model.state_dict(), out / f"{arm}-{seed}.pt")
    record = {"arm": arm, "fit_seed": seed, "parameters": model.parameter_count,
              "updates": CONFIG["epochs"] * (CONFIG["train_contexts"]//CONFIG["batch_contexts"]),
              "request_exposures": CONFIG["epochs"]*CONFIG["train_contexts"]*CONFIG["train_queries"],
              "train_seconds": elapsed, "curve": curve}
    write_json(out / f"{arm}-{seed}-fit.json", record)
    return model, record


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    c, q = data["target"].shape
    parts = {}
    elapsed = 0.
    for start in range(0, c, CONFIG["batch_contexts"]):
        ix = np.arange(start, min(start+CONFIG["batch_contexts"], c))
        batch = public_batch(data, ix)
        t0 = time.perf_counter()
        pred = model(**batch)
        elapsed += time.perf_counter() - t0
        target = torch.from_numpy(np.ascontiguousarray(data["target"][ix].reshape(-1)))
        pred["log_prob"] = pred.log_prob(target)
        for key, value in pred.items():
            parts.setdefault(key, []).append(value.numpy())
    return {key: np.concatenate(values).reshape(c, q, *values[0].shape[1:])
            for key, values in parts.items()}, elapsed


@torch.no_grad()
def latency(model, data):
    """Batch-one deployment path, same first archive and four requests for all arms.

    Static models may cache archive statistics; centered models rebuild for each
    query. This is measured whole-context cost, not a matched-compute comparison.
    """
    model.eval()
    bx, by = (torch.from_numpy(data[name][:1]) for name in ("bx", "by"))
    requests = [{name: torch.from_numpy(data[name][:1, j]) for name in ("fx", "fy", "qx")}
                for j in range(data["qx"].shape[1])]
    timings = []
    for repetition in range(CONFIG["latency_warmups"]+CONFIG["latency_repeats"]):
        t0 = time.perf_counter()
        cache = model.encode_archive(bx, by) if not model.centered else None
        t1 = time.perf_counter()
        for request in requests:
            if cache is None:
                model(bx, by, **request)
            else:
                model.predict_cached(cache, **request)
        t2 = time.perf_counter()
        if repetition >= CONFIG["latency_warmups"]:
            timings.append({"archive_seconds": t1-t0, "queries_seconds": t2-t1,
                            "total_seconds": t2-t0})
    return {"requests_per_archive": len(requests), "cached_static": not model.centered,
            "timings": timings,
            "median_context_seconds": float(np.median([row["total_seconds"] for row in timings])),
            "archive_array_bytes": data["bx"][0].nbytes + data["by"][0].nbytes,
            "parameter_bytes": sum(x.numel()*x.element_size() for x in model.parameters()),
            "buffer_bytes": sum(x.numel()*x.element_size() for x in model.buffers()),
            "cache_tensor_bytes": 0 if cache is None else cache.tensor_bytes}


def gp_latency(data):
    timings = []
    for repetition in range(CONFIG["latency_warmups"]+CONFIG["latency_repeats"]):
        t0 = time.perf_counter()
        for j in range(data["qx"].shape[1]):
            exact_gp_mixture(data["bx"][:1], data["by"][:1], data["fx"][:1, j],
                             data["fy"][:1, j], data["qx"][:1, j])
        if repetition >= CONFIG["latency_warmups"]:
            timings.append(time.perf_counter()-t0)
    return {"arm": "full_gp", "fit_seed": None, "cached_static": False,
            "requests_per_archive": data["qx"].shape[1], "timings": timings,
            "median_context_seconds": float(np.median(timings)),
            "archive_array_bytes": data["bx"][0].nbytes + data["by"][0].nbytes,
            "note": "Exact-law NumPy reference recomputes archive factorization for each request."}


def summarize(rows):
    result = {}
    for phase in ("base", "shift"):
        result[phase] = {}
        for arm in (*ARMS, "full_gp"):
            subset = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
            result[phase][arm] = {metric: float(np.mean([r[metric] for r in subset]))
                                   for metric in ("regret", "nll", "brier", "mse", "defer",
                                                  "negative", "positive", "always_defer_regret")}
    checks = []
    for phase in ("base", "shift"):
        candidate = result[phase]["centered16"]
        for control in ("static16", "static32", "nystrom16", "centered_nystrom16"):
            baseline = result[phase][control]
            cohort_wins = []
            for cohort in range(CONFIG["cohorts"]):
                values = {a: np.mean([r["regret"] for r in rows if r["phase"] == phase
                                      and r["cohort"] == cohort and r["arm"] == a])
                          for a in ("centered16", control)}
                cohort_wins.append(bool(values["centered16"] < values[control]))
            checks.append({"phase": phase, "control": control,
                           "ten_percent_regret_gain": bool(candidate["regret"] <= .9*baseline["regret"]
                                                           and candidate["regret"] < baseline["regret"]),
                           "nll_noninferiority": bool(candidate["nll"] <= baseline["nll"] + .02),
                           "all_cohort_mean_wins": all(cohort_wins), "cohort_wins": cohort_wins})
    passed = all(row[key] for row in checks for key in
                 ("ten_percent_regret_gain", "nll_noninferiority", "all_cohort_mean_wins"))
    return {"means": result, "checks": checks,
            "gate": "CONTINUE_TO_SELECTIVE_REFINEMENT" if passed else "DO_NOT_ADVANCE_THIS_CANDIDATE",
            "fit_seeds_are_not_independent_datasets": True}


def run(out):
    verify_sources(out)
    with (out / "started.json").open("x") as stream:
        json.dump({"started_unix": time.time()}, stream)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("frozen wall cap")))
    signal.alarm(CONFIG["wall_cap_seconds"])
    receipt = {"state": "RUNNING"}
    try:
        dataset = generate_contexts(seed=CONFIG["namespace"], contexts=CONFIG["train_contexts"],
                                    queries=CONFIG["train_queries"])
        np.savez_compressed(out / "train.npz", **dataset)
        fits, models = [], {}
        for seed in SEEDS:
            for arm in ARMS:
                model, fit = train(out, dataset, arm, seed)
                fits.append(fit)
                models[(arm, seed)] = model
        write_json(out / "fits.json", fits)
        write_json(out / "checkpoint-barrier.json", {
            "recorded_unix": time.time(), "evaluation_generation_started": False,
            "checkpoints": {f"{arm}-{seed}.pt": digest(out / f"{arm}-{seed}.pt")
                            for seed in SEEDS for arm in ARMS}})
        # Evaluation data is first generated after all checkpoints are final.
        rows, resources = [], []
        for phase, offset in (("base", 100), ("shift", 200)):
            for cohort in range(CONFIG["cohorts"]):
                dataset = generate_contexts(seed=CONFIG["namespace"]+offset+cohort,
                                            contexts=CONFIG["eval_contexts"], queries=CONFIG["eval_queries"],
                                            **CONFIG[phase])
                stem = f"{phase}-{cohort}"
                np.savez_compressed(out / f"{stem}-data.npz", **dataset)
                t0 = time.perf_counter()
                reference = reference_predict(dataset)
                reference_seconds = time.perf_counter()-t0
                np.savez_compressed(out / f"{stem}-full_gp.npz", **reference)
                if cohort == 0:
                    resources.append({"phase": phase, **gp_latency(dataset)})
                m = metrics(reference, dataset["target"], reference["prob_positive"])
                rows.append({"phase": phase, "cohort": cohort, "arm": "full_gp", "fit_seed": None,
                             "evaluation_seconds": reference_seconds,
                             **{key: float(value.mean()) for key, value in m.items()}})
                for seed in SEEDS:
                    for arm in ARMS:
                        pred, seconds = evaluate(models[(arm, seed)], dataset)
                        np.savez_compressed(out / f"{stem}-{arm}-{seed}.npz", **pred)
                        m = metrics(pred, dataset["target"], reference["prob_positive"])
                        rows.append({"phase": phase, "cohort": cohort, "arm": arm, "fit_seed": seed,
                                     "evaluation_seconds": seconds,
                                     **{key: float(value.mean()) for key, value in m.items()}})
                        if cohort == 0:
                            resources.append({"phase": phase, "arm": arm, "fit_seed": seed,
                                              **latency(models[(arm, seed)], dataset)})
                print(json.dumps({"evaluated": stem}), flush=True)
        write_json(out / "metrics.json", rows)
        write_json(out / "resources.json", resources)
        summary = summarize(rows)
        write_json(out / "summary.json", summary)
        verify_sources(out)
        receipt.update(state="EXITED", exit_code=0, gate=summary["gate"])
    except BaseException as exc:
        receipt.update(state="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.perf_counter()-started
        write_json(out / "run-receipt.json", receipt)
        manifest = {str(p.relative_to(out)): {"sha256": digest(p), "bytes": p.stat().st_size}
                    for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == "run" and args.registration_sha256 != digest(args.out / "registration.json"):
        raise ValueError("run requires the previously recorded registration SHA256")
    (register if args.mode == "register" else run)(args.out.resolve())


if __name__ == "__main__":
    main()
