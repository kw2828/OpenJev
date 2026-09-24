"""Separate fixed-checkpoint residual-covariance mechanism experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.special import logsumexp

from openjev.research.query_feature_data import generate_contexts
from openjev.research.residual_memory import build_archive, predict

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "output/query-feature-v1"
MODES = ("sor", "query_only", "fic", "full")
SEEDS = (11, 23, 37)
METRICS = ("regret", "nll", "brier", "mse", "defer", "negative", "positive", "always_defer_regret")
CONFIG = {
    "version": "residual-memory-v1", "namespace": 442260924,
    "fit_seeds": list(SEEDS), "modes": list(MODES), "cohorts": 3,
    "contexts": 128, "queries": 4, "fewshots": 4,
    "noise_variance": .0225, "defer_cost": .2,
    "populations": {
        "base": {"blocks": 4, "basis_points": 16, "extent": 2., "offset": 100},
        "shift": {"blocks": 8, "basis_points": 16, "extent": 3., "offset": 200},
        "long": {"blocks": 4, "basis_points": 128, "extent": 2., "offset": 300}},
    "latency_warmups": 3, "latency_repeats": 20, "wall_cap_seconds": 1200,
    "training_updates": 0, "checkpoint_selection": "all three parent static Nyström final fits",
}
SOURCES = (
    "src/openjev/research/query_feature_data.py", "src/openjev/research/residual_memory.py",
    "scripts/residual_memory_study.py", "scripts/audit_residual_memory_study.py",
    "tests/test_residual_memory.py", "tests/test_residual_memory_study.py",
    "tests/test_audit_residual_memory_study.py", "research/residual-memory-protocol.md",
    "scripts/audit_query_feature_study.py",
)


def descriptor(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def parent_pins():
    manifest = json.loads((PARENT / "manifest.json").read_text())
    for name, pin in manifest.items():
        if descriptor(PARENT / name) != pin:
            raise ValueError("parent evidence changed: " + name)
    return {"manifest": descriptor(PARENT / "manifest.json"),
            "registration": descriptor(PARENT / "registration.json"),
            "checkpoints": {f"nystrom16-{seed}.pt": descriptor(PARENT / f"nystrom16-{seed}.pt")
                            for seed in SEEDS}}


def register(out):
    pins = parent_pins()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in SOURCES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        sources[name] = descriptor(target)
    write(out / "registration.json", {"config": CONFIG, "sources": sources, "parent": pins,
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "torch": torch.__version__, "platform": platform.platform(), "threads": 1}})
    print(json.dumps({"registered": str(out), **descriptor(out / "registration.json")}), flush=True)


def verify(out):
    registration = json.loads((out / "registration.json").read_text())
    if registration["config"] != CONFIG or set(registration["sources"]) != set(SOURCES):
        raise ValueError("frozen configuration/source roster")
    for name, pin in registration["sources"].items():
        if descriptor(ROOT / name) != pin or descriptor(out / "source" / name) != pin:
            raise ValueError("frozen source changed: " + name)
    if parent_pins() != registration["parent"]:
        raise ValueError("parent checkpoint lineage changed")


def score(pred, target, reference_positive):
    p, rp = pred["prob_positive"], reference_positive
    action = np.stack((p, np.full_like(p, .2), 1-p), -1).argmin(-1)
    costs = np.stack((rp, np.full_like(rp, .2), 1-rp), -1)
    regret = np.take_along_axis(costs, action[..., None], -1)[..., 0]-costs.min(-1)
    mean = (np.exp(pred["log_weights"])*pred["component_mean"]).sum(-1)
    values = {"regret": regret, "nll": -pred["log_prob"], "brier": (p-(target > 0))**2,
              "mse": (mean-target)**2, "defer": (action == 1).astype(float),
              "negative": (action == 0).astype(float), "positive": (action == 2).astype(float),
              "always_defer_regret": .2-costs.min(-1)}
    return {name: float(value.mean()) for name, value in values.items()}


def evaluate(data, mode, length, amplitude):
    c, q = data["target"].shape
    k = data["bx"].shape[1]
    output = {name: np.empty((c, q, k)) for name in ("component_mean", "component_variance", "log_weights")}
    output.update(prob_positive=np.empty((c, q)), log_prob=np.empty((c, q)))
    archive_seconds, query_seconds = 0., 0.
    records = []
    for context in range(c):
        start = time.perf_counter()
        cache = build_archive(data["bx"][context], data["by"][context], length=length,
                              amplitude=amplitude, mode=mode)
        archive_seconds += time.perf_counter()-start
        per_request = []
        for query in range(q):
            diagnostics = {}
            start = time.perf_counter()
            pred = predict(cache, data["fx"][context, query], data["fy"][context, query],
                           data["qx"][context, query], diagnostics=diagnostics)
            query_seconds += time.perf_counter()-start
            for name, value in pred.items():
                output[name][context, query] = value
            variance = pred["component_variance"]
            residual = data["target"][context, query]-pred["component_mean"]
            output["log_prob"][context, query] = logsumexp(pred["log_weights"]
                -.5*(math.log(2*math.pi)+np.log(variance)+residual**2/variance))
            per_request.append(diagnostics)
        records.append({"archive": dict(cache.diagnostics), "queries": per_request,
                        "cache_array_bytes": cache.array_bytes})
    if not all(np.isfinite(x).all() for x in output.values()):
        raise ValueError("nonfinite prediction")
    return output, {"archive_seconds": archive_seconds, "query_seconds": query_seconds,
                    "context_diagnostics": records}


def latency(data, mode, length, amplitude):
    rows = []
    for repetition in range(CONFIG["latency_warmups"]+CONFIG["latency_repeats"]):
        t0 = time.perf_counter()
        cache = build_archive(data["bx"][0], data["by"][0], length=length, amplitude=amplitude, mode=mode)
        t1 = time.perf_counter()
        for query in range(CONFIG["queries"]):
            predict(cache, data["fx"][0, query], data["fy"][0, query], data["qx"][0, query])
        t2 = time.perf_counter()
        if repetition >= CONFIG["latency_warmups"]:
            rows.append({"archive_seconds": t1-t0, "query_seconds": t2-t1, "total_seconds": t2-t0})
    return {"timings": rows, "median_context_seconds": float(np.median([r["total_seconds"] for r in rows])),
            "cache_array_bytes": cache.array_bytes, "input_archive_bytes": data["bx"][0].nbytes+data["by"][0].nbytes,
            "queries_per_archive": CONFIG["queries"]}


def summarize(rows):
    means = {}
    for phase in CONFIG["populations"]:
        means[phase] = {}
        for mode in (*MODES, "true_gp"):
            selected = [r for r in rows if r["phase"] == phase and r["mode"] == mode]
            means[phase][mode] = {name: float(np.mean([r[name] for r in selected])) for name in METRICS}
    checks = []
    for phase in CONFIG["populations"]:
        f, s = means[phase]["fic"], means[phase]["sor"]
        checks.append({"name": phase+"_nll_noninferior", "passed": f["nll"] <= s["nll"]+.02})
        if phase != "shift":
            checks.append({"name": phase+"_regret_preserved", "passed": f["regret"] <= 1.05*s["regret"]+1e-6})
    f, s = means["shift"]["fic"], means["shift"]["sor"]
    checks.extend([
        {"name": "shift_regret_25_percent_lower", "passed": f["regret"] <= .75*s["regret"]},
        {"name": "shift_nll_0.1_nat_lower", "passed": f["nll"] <= s["nll"]-.1},
        {"name": "shift_beats_always_defer", "passed": f["regret"] < f["always_defer_regret"]}])
    for cohort in range(3):
        values = {m: np.mean([r["regret"] for r in rows if r["phase"] == "shift"
                             and r["cohort"] == cohort and r["mode"] == m]) for m in ("fic", "sor")}
        checks.append({"name": f"shift_cohort_{cohort}_regret_win", "passed": bool(values["fic"] < values["sor"])})
    return {"means": means, "checks": checks,
            "gate": "RESIDUAL_CONTROL_QUALIFIED" if all(r["passed"] for r in checks) else "RESIDUAL_CONTROL_NOT_QUALIFIED",
            "scope": "Established covariance approximation diagnostic; no novel architecture or parent rescue."}


def run(out):
    verify(out)
    with (out / "started.json").open("x") as stream:
        json.dump({"unix_time": time.time()}, stream)
    start = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("frozen total wall cap")))
    signal.alarm(CONFIG["wall_cap_seconds"])
    receipt = {"state": "RUNNING", "training_updates": 0}
    try:
        kernels = []
        for seed in SEEDS:
            name = f"nystrom16-{seed}.pt"
            checkpoint = torch.load(PARENT / name, map_location="cpu", weights_only=True)
            if set(checkpoint) != {"inducing_grid", "log_length", "log_amplitude"}:
                raise ValueError("parent static Nyström state schema")
            expected = torch.cartesian_prod(torch.linspace(-2., 2., 4, dtype=torch.float64),
                                           torch.linspace(-2., 2., 4, dtype=torch.float64))
            if not torch.equal(checkpoint["inducing_grid"], expected):
                raise ValueError("parent fixed inducing grid changed")
            kernel = {"fit_seed": seed, "length": math.exp(float(checkpoint["log_length"])),
                      "amplitude": math.exp(float(checkpoint["log_amplitude"])),
                      "checkpoint": name, "checkpoint_descriptor": descriptor(PARENT / name)}
            if not all(math.isfinite(kernel[x]) and kernel[x] > 0 for x in ("length", "amplitude")):
                raise ValueError("finite positive inherited parameters")
            kernels.append(kernel)
            (out / name).write_bytes((PARENT / name).read_bytes())
        write(out / "kernels.json", kernels)
        rows, resources, diagnostics, pairwise = [], [], [], []
        for phase, population in CONFIG["populations"].items():
            for cohort in range(CONFIG["cohorts"]):
                parameters = {k: v for k, v in population.items() if k != "offset"}
                data = generate_contexts(seed=CONFIG["namespace"]+population["offset"]+cohort,
                                         contexts=CONFIG["contexts"], queries=CONFIG["queries"], **parameters)
                stem = f"{phase}-{cohort}"
                np.savez_compressed(out / f"{stem}-data.npz", **data)
                schedules = [("true_gp", None, "full", 1., 1.)]
                schedules.extend((mode, kernel["fit_seed"], mode, kernel["length"], kernel["amplitude"])
                                 for kernel in kernels for mode in MODES)
                reference = None
                predictions = {}
                for mode, seed, inference_mode, length, amplitude in schedules:
                    pred, cost = evaluate(data, inference_mode, length, amplitude)
                    predictions[(mode, seed)] = pred
                    if mode == "true_gp":
                        reference = pred["prob_positive"]
                    np.savez_compressed(out / f"{stem}-{mode}-{seed}.npz", **pred)
                    rows.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed,
                                 **score(pred, data["target"], reference)})
                    diagnostics.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed, **cost})
                    if cohort == 0:
                        resources.append({"phase": phase, "mode": mode, "fit_seed": seed,
                                          **latency(data, inference_mode, length, amplitude)})
                for seed in SEEDS:
                    full = predictions[("full", seed)]
                    for mode in ("sor", "query_only", "fic"):
                        pred = predictions[(mode, seed)]
                        pairwise.append({"phase": phase, "cohort": cohort, "mode": mode, "fit_seed": seed,
                            "mean_weight_l1": float(np.abs(np.exp(pred["log_weights"])
                                -np.exp(full["log_weights"])).sum(-1).mean()),
                            "mean_sign_probability_absolute_error": float(np.abs(
                                pred["prob_positive"]-full["prob_positive"]).mean())})
                print(json.dumps({"evaluated": stem, "prediction_groups": len(rows)}), flush=True)
        write(out / "metrics.json", rows)
        write(out / "resources.json", resources)
        write(out / "diagnostics.json", diagnostics)
        write(out / "pairwise.json", pairwise)
        result = summarize(rows)
        write(out / "summary.json", result)
        verify(out)
        receipt.update(state="EXITED", exit_code=0, gate=result["gate"])
    except BaseException as exc:
        receipt.update(state="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.perf_counter()-start
        write(out / "run-receipt.json", receipt)
        write(out / "manifest.json", {str(p.relative_to(out)): descriptor(p)
              for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "run" and args.registration_sha256 != descriptor(args.out / "registration.json")["sha256"]:
        raise ValueError("expected registration SHA256 required")
    (register if args.mode == "register" else run)(args.out.resolve())


if __name__ == "__main__":
    main()
