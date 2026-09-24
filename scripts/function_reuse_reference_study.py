"""Prospective strong-control diagnostic for noiseless linear function reuse.

This is an independent task implementation, not the paper's neural experiment.
Only generate() and reporting see private matrices, indices and query targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from openjev.research.function_reuse_reference import fewshot_only, fit_blocks, predict

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"namespace": 440260924, "cohorts": 5, "blocks": [1, 3, 8, 16], "cases": 64,
              "dimensions": 8, "basis_examples": 16, "fewshot_examples": 4, "queries": 8, "choices": 6}
SOURCES = ["scripts/function_reuse_reference_study.py",
           "scripts/audit_function_reuse_reference.py",
           "src/openjev/research/function_reuse_reference.py",
           "tests/test_function_reuse_reference.py",
           "tests/test_function_reuse_reference_study.py",
           "tests/test_function_reuse_reference_audit.py",
           "research/function-reuse-reference-protocol.md"]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def register(folder):
    if folder.exists() or any(os.environ.get(key) != "1" for key in THREADS):
        raise ValueError("exclusive folder and single-thread numerical environment required")
    sources = {name: pin(ROOT / name) for name in SOURCES}
    folder.mkdir(parents=True)
    for name in SOURCES:
        target = folder / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write(folder / "registration.json", {
        "config": CONFIG, "sources": sources, "created_unix": time.time(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                     "executable": sys.executable, "threads": {k: os.environ[k] for k in THREADS}},
        "seconds_cap": 120, "run_attempts": 1, "data_selection": "none", "training_updates": 0})


def check_sources(folder):
    registration = json.loads((folder / "registration.json").read_text())
    if registration["config"] != CONFIG:
        raise ValueError("registered configuration changed")
    for name, expected in registration["sources"].items():
        if pin(ROOT / name) != expected or pin(folder / "sources" / name) != expected:
            raise ValueError("frozen source changed: " + name)
    if any(os.environ.get(key) != "1" for key in THREADS):
        raise ValueError("single-thread environment changed")
    return registration


def generate(cohort, blocks, case):
    """Fresh independent cases; hidden state is returned for scoring only."""
    children = np.random.SeedSequence([CONFIG["namespace"], cohort, blocks, case]).spawn(6)
    matrix_rng, basis_rng, index_rng, fewshot_rng, query_rng, cost_rng = map(
        np.random.default_rng, children)
    d, b, f, q, c = (CONFIG[k] for k in
                     ("dimensions", "basis_examples", "fewshot_examples", "queries", "choices"))
    matrices = matrix_rng.normal(size=(blocks, d, d)) / np.sqrt(d)
    basis_x = basis_rng.normal(size=(blocks, b, d))
    basis_y = np.einsum("kbd,kdo->kbo", basis_x, matrices)
    true_index = index_rng.integers(0, blocks, size=q, dtype=np.int64)
    fewshot_x = fewshot_rng.normal(size=(q, f, d))
    fewshot_y = np.einsum("qfd,qdo->qfo", fewshot_x, matrices[true_index])
    query_x = query_rng.normal(size=(q, d))
    target = np.einsum("qd,qdo->qo", query_x, matrices[true_index])
    cost_vectors = cost_rng.normal(size=(q, c, d))
    cost_vectors /= np.linalg.norm(cost_vectors, axis=-1, keepdims=True)
    return {"matrices": matrices, "basis_x": basis_x, "basis_y": basis_y,
                "true_index": true_index, "fewshot_x": fewshot_x, "fewshot_y": fewshot_y,
                "query_x": query_x, "target": target, "cost_vectors": cost_vectors}


def evaluate(public):
    """Only declared public arrays may cross the predictor boundary."""
    keys = {"basis_x", "basis_y", "fewshot_x", "fewshot_y", "query_x", "cost_vectors"}
    if set(public) != keys:
        raise ValueError("exact public predictor boundary")
    started = time.perf_counter()
    fitted = fit_blocks(public["basis_x"], public["basis_y"])
    fit_seconds = time.perf_counter() - started
    lookup, selected, residuals, choices = [], [], [], []
    started = time.perf_counter()
    for x, y, query, costs in zip(public["fewshot_x"], public["fewshot_y"],
                                  public["query_x"], public["cost_vectors"], strict=True):
        answer = predict(fitted, x, y, query)
        lookup.append(answer["prediction"])
        selected.append(answer["selected_block"])
        residuals.append(answer["block_residuals"])
        choices.append(int(np.argmin(costs @ answer["prediction"])))
    lookup_seconds = time.perf_counter() - started
    fewshot, ranks, singular, fewshot_choices = [], [], [], []
    started = time.perf_counter()
    for x, y, query, costs in zip(public["fewshot_x"], public["fewshot_y"],
                                  public["query_x"], public["cost_vectors"], strict=True):
        answer = fewshot_only(x, y, query)
        fewshot.append(answer["prediction"])
        ranks.append(answer["rank"])
        singular.append(answer["singular_values"])
        fewshot_choices.append(int(np.argmin(costs @ answer["prediction"])))
    fewshot_seconds = time.perf_counter() - started
    return {"maps": fitted["maps"], "singular_values": fitted["singular_values"], "ranks": fitted["ranks"],
                "lookup_prediction": np.asarray(lookup), "selected_index": np.asarray(selected, dtype=np.int64),
                "block_residuals": np.asarray(residuals), "lookup_choice": np.asarray(choices, dtype=np.int64),
                "fewshot_prediction": np.asarray(fewshot), "fewshot_ranks": np.asarray(ranks, dtype=np.int64),
                "fewshot_singular_values": np.asarray(singular),
                "fewshot_choice": np.asarray(fewshot_choices, dtype=np.int64),
                "lookup_fit_seconds": np.asarray(fit_seconds), "lookup_query_seconds": np.asarray(lookup_seconds),
                "fewshot_query_seconds": np.asarray(fewshot_seconds)}


def summarize(arrays, cohort, blocks):
    true_costs = np.einsum("eqcd,eqd->eqc", arrays["cost_vectors"], arrays["target"])
    optimum = true_costs.min(axis=-1)
    row = {"cohort": cohort, "blocks": blocks, "cases": len(arrays["target"]),
               "queries": int(np.prod(arrays["target"].shape[:2]))}
    for arm in ("lookup", "fewshot"):
        chosen = arrays[arm + "_choice"]
        costs = np.take_along_axis(true_costs, chosen[..., None], axis=-1)[..., 0]
        row[arm + "_mse"] = float(np.mean((arrays[arm + "_prediction"] - arrays["target"])**2))
        row[arm + "_regret"] = float(np.mean(costs - optimum))
        row[arm + "_action_accuracy"] = float(np.mean(chosen == true_costs.argmin(axis=-1)))
    row["retrieval_accuracy"] = float(np.mean(arrays["selected_index"] == arrays["true_index"]))
    row["full_rank"] = bool(np.all(arrays["ranks"] == CONFIG["dimensions"]))
    row["min_relative_singular_value"] = float(np.min(
        arrays["singular_values"][..., -1] / arrays["singular_values"][..., 0]))
    for key in ("lookup_fit_seconds", "lookup_query_seconds", "fewshot_query_seconds"):
        row[key] = float(arrays[key].sum())
    row["map_array_bytes_per_context"] = int(arrays["maps"][0].nbytes)
    row["diagnostic_array_bytes_per_context"] = int(
        arrays["singular_values"][0].nbytes + arrays["ranks"][0].nbytes + blocks)
    row["raw_basis_array_bytes_per_context"] = int(arrays["basis_x"][0].nbytes + arrays["basis_y"][0].nbytes)
    row["conditions"] = {"full_rank": row["full_rank"], "numerical_mse": row["lookup_mse"] <= 1e-12,
                              "retrieval_exact": row["retrieval_accuracy"] == 1.0,
                              "numerical_regret": row["lookup_regret"] <= 1e-10}
    return row


def run(folder):
    registration = check_sources(folder)
    run_dir = folder / "run"
    run_dir.mkdir()
    started = time.perf_counter()
    write(run_dir / "started.json", {"unix": time.time(), "registration": pin(folder / "registration.json")})
    rows = []
    try:
        for cohort in range(CONFIG["cohorts"]):
            for blocks in CONFIG["blocks"]:
                cases = []
                for case in range(CONFIG["cases"]):
                    if time.perf_counter() - started > registration["seconds_cap"]:
                        raise TimeoutError("fixed diagnostic cap")
                    data = generate(cohort, blocks, case)
                    public = {k: data[k] for k in ("basis_x", "basis_y", "fewshot_x", "fewshot_y",
                                                 "query_x", "cost_vectors")}
                    cases.append({**data, **evaluate(public)})
                arrays = {key: np.stack([case[key] for case in cases]) for key in cases[0]}
                with (run_dir / f"cohort-{cohort:02d}-k-{blocks:02d}.npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                rows.append(summarize(arrays, cohort, blocks))
                if time.perf_counter() - started > registration["seconds_cap"]:
                    raise TimeoutError("fixed diagnostic cap after group persistence")
        check_sources(folder)
        summary = {"config": CONFIG, "rows": rows, "training_updates": 0, "external_model_calls": 0,
                       "status": ("LINEAR_TASK_SOLVED_BY_CLASSICAL_REFERENCE"
                               if all(all(r["conditions"].values()) for r in rows) else "REFERENCE_NOT_SOLVED")}
        write(run_dir / "summary.json", summary)
        files = {str(p.relative_to(folder)): pin(p) for p in sorted(run_dir.glob("*")) if p.is_file()}
        if time.perf_counter() - started > registration["seconds_cap"]:
            raise TimeoutError("fixed diagnostic cap before completion")
        write(run_dir / "receipt.json", {"status": "COMPLETE", "files": files,
              "elapsed_seconds": time.perf_counter() - started, "registration": pin(folder / "registration.json")})
        print(json.dumps({"status": summary["status"], "groups": len(rows)}))
    except BaseException as exc:
        write(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "elapsed_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "run"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    (register if args.phase == "register" else run)(args.folder.resolve())


if __name__ == "__main__":
    main()
