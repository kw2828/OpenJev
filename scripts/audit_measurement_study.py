"""Independent saved-evidence audit of fixed-budget Gaussian measurements.

No producer, generator, memory, controls, or Torch imports. Numeric references
condition the original Gaussian law on independently reconstructed observations.
Field generation and measured runtimes remain source/receipt attestations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
VERSION = "measurement-audit-v1"
RTOL = ATOL = 1e-8
KINDS = ("spectral", "dct", "bins")
METHODS = tuple(kind + "118" for kind in KINDS) + ("coverage118", "recent98", "coverage98", "full")
METRICS = ("regret", "nll", "brier", "mse", "coverage90", "defer", "always_defer_regret", "risk_mae")
POPULATIONS = {
    "base": {"observations": 64, "geometry": "axial", "offset": 100},
    "shift": {"observations": 64, "geometry": "diagonal", "offset": 200},
    "long": {"observations": 192, "geometry": "axial", "offset": 300},
    "long_shift": {"observations": 192, "geometry": "diagonal", "offset": 400},
}
RAW = ("coverage118", "recent98", "coverage98")
CONFIG = {
    "version": "measurement-v1", "namespace": 553260924, "cohorts": 3,
    "evaluation_contexts": 128, "queries": 4, "state_budget_bytes": 1024,
    "populations": POPULATIONS, "methods": list(METHODS), "warmups": 3,
    "timing_repeats": 10, "bootstrap_repeats": 1000,
    "bootstrap_offsets": {"long": 900, "long_shift": 901}, "wall_cap_seconds": 1200,
}
SOURCES = {
    "src/openjev/research/measurement_memory.py", "src/openjev/research/retention_data.py",
    "scripts/measurement_study.py",
    "scripts/audit_measurement_study.py", "tests/test_measurement_memory.py",
    "tests/test_measurement_study.py", "tests/test_audit_measurement_study.py",
    "research/measurement-protocol.md",
}


def require(condition, message):
    if not condition:
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


def close(actual, expected, name):
    left, right = np.asarray(actual), np.asarray(expected)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            and np.allclose(left, right, rtol=RTOL, atol=ATOL), "numerical agreement: " + name)


def compare(actual, expected, name="saved JSON"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), name + " schema")
        for key in expected:
            compare(actual[key], expected[key], name + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + " list")
        for a, b in zip(actual, expected, strict=True):
            compare(a, b, name)
    elif type(expected) is float:
        require(type(actual) in (int, float), name + " scalar type")
        close(actual, expected, name)
    else:
        require(type(actual) is type(expected) and actual == expected, name + " identity")


def array(value, shape, name, dtype=np.float64):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
            and value.shape == shape and np.isfinite(value).all(), "finite declared array: " + name)


def grid():
    axis = np.arange(17, dtype=np.float64) / 4 - 2
    return np.array([(a, b) for a in axis for b in axis])


def grid_ids(coordinates):
    require(coordinates.dtype == np.float64 and coordinates.shape[-1] == 2
            and np.isfinite(coordinates).all() and np.all(np.abs(coordinates) <= 2)
            and np.array_equal(coordinates * 4, np.rint(coordinates * 4)), "fixed grid coordinates")
    indices = np.rint((coordinates + 2) * 4).astype(np.int64)
    return indices[..., 0] * 17 + indices[..., 1]


def basis(kind):
    """Fixed rows, independently formed without calling a production basis."""
    require(kind in KINDS, "known sketch kind")
    if kind == "bins":
        result = np.zeros((118, 289))
        result[(np.arange(289) * 118) // 289, np.arange(289)] = 1.
        return result
    if kind == "dct":
        axis = np.arange(17)
        vectors = np.cos(np.pi * np.arange(17)[:, None] * (axis[None] + .5) / 17)
        vectors[0] *= math.sqrt(1 / 17)
        vectors[1:] *= math.sqrt(2 / 17)
        order = sorted(((i, j) for i in range(17) for j in range(17)), key=lambda ij: (ij[0]**2 + ij[1]**2, *ij))
    else:
        axis = np.arange(17) / 4
        values, columns = np.linalg.eigh(np.exp(-.5 * (axis[:, None] - axis[None])**2))
        order = np.argsort(-values, kind="stable")
        values, vectors = values[order], columns[:, order].T
        for row in vectors:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        order = sorted(((i, j) for i in range(17) for j in range(17)),
                       key=lambda ij: (-float(values[ij[0]] * values[ij[1]]), *ij))
    return np.stack([np.outer(vectors[i], vectors[j]).ravel() for i, j in order[:118]])


def kernel(x, z):
    difference = x[..., :, None, :] - z[..., None, :, :]
    return np.exp(-.5 * np.sum(difference**2, axis=-1)) + 1e-5 * np.all(difference == 0, axis=-1)


def validate_data(data, contexts, steps, geometry):
    require(set(data) == {"x", "y", "paths", "exposure"}, "public/private data schema")
    for key, shape in (("x", (contexts, steps, 2)), ("y", (contexts, steps)),
                       ("paths", (contexts, 4, 4, 4, 2)), ("exposure", (contexts, 4, 4))):
        array(data[key], shape, key)
    identities = grid_ids(data["x"])
    grid_ids(data["paths"])
    require(all(len(set(row)) == steps for row in identities), "unique stream coordinates")
    changes = np.diff(data["paths"], axis=3)
    require(np.array_equal(changes, np.repeat(changes[:, :, :, :1], 3, axis=3)), "straight equally spaced paths")
    if geometry == "axial":
        require(np.all((changes == 0) | (changes == .5)) and np.all(changes.sum(-1) == .5), "canonical axial paths")
    else:
        require(geometry == "diagonal" and np.all(changes[..., 0] == .5)
                and np.all(np.abs(changes[..., 1]) == .5), "canonical diagonal paths")
    for context in data["paths"]:
        for request in context:
            require(len({tuple(path.flat) for path in request}) == 4, "four distinct requested actions")


def packed_mask(identities):
    bits = np.zeros((len(identities), 296), dtype=np.uint8)
    for row, selected in zip(bits, identities, strict=True):
        row[selected] = 1
    return np.packbits(bits, axis=1, bitorder="little")


def retained_indices(coordinates, method):
    """Indices into the original stream; new packed coverage ties use grid ID."""
    require(method in ("coverage118", "coverage98", "recent98"), "known raw rule")
    count = len(coordinates)
    if method == "recent98":
        return np.arange(max(0, count - 98), count)
    capacity = 118 if method == "coverage118" else 98
    ids = grid_ids(coordinates)
    selected = []
    for index in range(count):
        selected.append(index)
        if len(selected) <= capacity:
            continue
        points = coordinates[selected]
        squared = np.sum((points[:, None] - points[None])**2, axis=-1)
        np.fill_diagonal(squared, np.inf)
        nearest = np.min(squared, axis=1)
        tied = np.flatnonzero(nearest == np.min(nearest))
        drop = min(tied, key=lambda i: ids[selected[i]]) if method == "coverage118" else tied[0]
        selected.pop(int(drop))
    if method == "coverage118":
        selected.sort(key=lambda i: ids[i])
    return np.asarray(selected, dtype=np.int64)


def reconstructed_state(x, y, method, matrices=None):
    batch, steps = y.shape
    ids = grid_ids(x)
    if method in tuple(kind + "118" for kind in KINDS):
        kind = method[:-3]
        values = np.zeros((batch, 118))
        if steps <= 118:
            for i in range(batch):
                values[i, :steps] = y[i, np.argsort(ids[i])]
        else:
            projection = basis(kind) if matrices is None else matrices[kind]
            for i in range(batch):
                values[i] = projection[:, ids[i]] @ y[i]
        return {"mask": packed_mask(ids), "values": values}
    selected = [retained_indices(coordinates, method) for coordinates in x]
    if method == "coverage118":
        values = np.zeros((batch, 118))
        for i, keep in enumerate(selected):
            values[i, :len(keep)] = y[i, keep]
        return {"mask": packed_mask([ids[i, keep] for i, keep in enumerate(selected)]), "values": values}
    if method in ("recent98", "coverage98"):
        identities = np.zeros((batch, 98), dtype=np.uint16)
        values = np.zeros((batch, 98))
        for i, keep in enumerate(selected):
            identities[i, :len(keep)] = ids[i, keep]
            values[i, :len(keep)] = y[i, keep]
        return {"ids": identities, "values": values}
    raise ValueError("bounded state method")


def conditional(coordinates, labels, points, projection=None):
    """Condition directly on row-space observations under the original GP law."""
    covariance = kernel(coordinates, coordinates) + .09 * np.eye(len(labels))
    cross = kernel(points, coordinates)
    rank, cutoff = len(labels), 0.
    if projection is not None:
        # A basis of the measurement row space avoids dividing saved sums by
        # singular values. This reference uses public y only to reconstruct z.
        _, singular, right = np.linalg.svd(projection, full_matrices=False)
        cutoff = float(np.finfo(np.float64).eps * max(projection.shape) * singular[0])
        rank = int(np.count_nonzero(singular > cutoff))
        require(rank > 0, "nonempty measurement row space")
        rows = right[:rank]
        labels = rows @ labels
        covariance = rows @ covariance @ rows.T
        cross = cross @ rows.T
    factor = cho_factor(covariance, lower=True)
    mean = cross @ cho_solve(factor, labels)
    posterior = kernel(points, points) - cross @ cho_solve(factor, cross.T)
    return mean, posterior, rank, cutoff


def moments(point_mean, covariance, queries):
    mean = point_mean.reshape(queries, 4, 4).mean(-1)
    blocks = covariance.reshape(queries * 4, 4, queries * 4, 4)
    variance = np.array([blocks[i, :, i, :].sum() / 16 for i in range(queries * 4)]).reshape(queries, 4)
    require(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0), "positive latent path variance")
    return {"mean": mean, "variance": variance, "risk": ndtr((mean - .5) / np.sqrt(variance))}


def predict(data, method, matrices=None, check=lambda: None):
    require(method in METHODS, "known prediction method")
    predictions, ranks, cutoffs = [], [], []
    hybrid = method[:-3] in KINDS and method.endswith("118")
    for x, y, paths in zip(data["x"], data["y"], data["paths"], strict=True):
        check()
        projection = None
        if hybrid and len(y) > 118:
            projection = (basis(method[:-3]) if matrices is None else matrices[method[:-3]])[:, grid_ids(x)]
        elif method not in ("full", "spectral118", "dct118", "bins118"):
            keep = retained_indices(x, method)
            x, y = x[keep], y[keep]
        mean, covariance, rank, cutoff = conditional(x, y, paths.reshape(-1, 2), projection)
        predictions.append(moments(mean, covariance, paths.shape[0]))
        ranks.append(rank)
        cutoffs.append(cutoff)
    result = {key: np.stack([row[key] for row in predictions]) for key in ("mean", "variance", "risk")}
    if method != "full":
        result.update(rank=np.asarray(ranks, dtype=np.int64), rank_cutoff=np.asarray(cutoffs))
    return result


def metrics(prediction, reference, exposure):
    shape = exposure.shape
    require(set(prediction) == {"mean", "variance", "risk"}, "prediction metric schema")
    for name in prediction:
        array(prediction[name], shape, name)
    require(np.all(prediction["variance"] > 0) and np.all((prediction["risk"] >= 0) & (prediction["risk"] <= 1)),
            "positive Gaussian variance and bounded risk")
    close(prediction["risk"], ndtr((prediction["mean"] - .5) / np.sqrt(prediction["variance"])), "Gaussian tail risk")
    costs = np.concatenate((.02 + prediction["risk"], np.full((*shape[:-1], 1), .2)), axis=-1)
    true_costs = np.concatenate((.02 + reference["risk"], np.full((*shape[:-1], 1), .2)), axis=-1)
    chosen = costs.argmin(axis=-1)
    optimum = true_costs.min(axis=-1)
    error = exposure - prediction["mean"]
    return {"regret": float(np.mean(np.take_along_axis(true_costs, chosen[..., None], axis=-1)[..., 0] - optimum)),
            "nll": float(np.mean(.5 * (math.log(2 * math.pi) + np.log(prediction["variance"]) + error**2 / prediction["variance"]))),
            "brier": float(np.mean((prediction["risk"] - (exposure > .5))**2)), "mse": float(np.mean(error**2)),
            "coverage90": float(np.mean(np.abs(error) <= 1.6448536269514722 * np.sqrt(prediction["variance"]))),
            "defer": float(np.mean(chosen == 4)), "always_defer_regret": float(np.mean(.2 - optimum)),
            "risk_mae": float(np.mean(np.abs(prediction["risk"] - reference["risk"])))}


def identities():
    return [(population, cohort, method) for population in POPULATIONS for cohort in range(3) for method in METHODS]


def expected_files():
    names = {"registration.json", "started.json", "run-status.json", "summary.json",
             "metrics.json", "resources.json", "bootstrap.json"}
    for population, cohort, method in identities():
        stem = f"{population}-{cohort}"
        names.add(f"data/{stem}.npz")
        names.add(f"pred/{stem}-{method}.npz")
        if method != "full":
            names.add(f"state/{stem}-{method}.npz")
    return names | {"source/" + name for name in SOURCES}


def absolute_argument(value):
    require(type(value) is str, "command string arguments")
    path = Path(value)
    # Do not resolve the virtualenv executable through its interpreter symlink.
    return str(path.absolute() if path.is_absolute() else (ROOT / path).absolute())


def authenticate(folder, process):
    folder, process = Path(folder).resolve(), Path(process).resolve()
    plan = read(folder / "registration.json")
    require(set(plan) == {"config", "sources", "parent_result", "environment"}, "registration schema")
    compare(plan["config"], CONFIG, "fixed configuration")
    require(set(plan["sources"]) == SOURCES, "exact eight source closure")
    for name, pin in plan["sources"].items():
        require(descriptor(ROOT / name) == pin == descriptor(folder / "source" / name), "current source and snapshot pins")
    require(descriptor(ROOT / "research/retention-results/summary.json") == plan["parent_result"], "closed failed parent pin")
    compare(plan["environment"], {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                                  "platform": platform.platform(), "threads": 1}, "registered runtime")
    manifest = read(folder / "manifest.json")
    require(set(manifest) == {"files"} and set(manifest["files"]) == expected_files(), "complete producer roster")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "no symlink evidence")
    observed = {p.relative_to(folder).as_posix(): descriptor(p) for p in folder.rglob("*")
                if p.is_file() and p != folder / "manifest.json"}
    require(observed == manifest["files"], "complete original opaque inventory before decoding")
    registration = descriptor(folder / "registration.json")
    compare(read(folder / "started.json"), {"version": CONFIG["version"], "registration": registration}, "original start")
    status = read(folder / "run-status.json")
    require(set(status) == {"state", "elapsed_seconds", "metric_groups", "resources"}
            and status["state"] == "COMPLETE" and type(status["metric_groups"]) is int
            and status["metric_groups"] == 84 and type(status["resources"]) is int and status["resources"] == 28,
            "complete original producer")
    seconds = status["elapsed_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 1200, "producer elapsed cap")
    terminal = read(process)
    require(terminal["state"] == "EXITED" and type(terminal["returncode"]) is int and terminal["returncode"] == 0,
            "original successful process closure")
    elapsed = terminal["elapsed_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and seconds <= elapsed <= 1200,
            "original process elapsed cap and nesting")
    command = terminal["command"]
    require(isinstance(command, list) and len(command) == 7 and all(type(v) is str for v in command), "exact original run argv")
    require(absolute_argument(command[0]) == str(ROOT / ".venv/bin/python")
            and absolute_argument(command[1]) == str(ROOT / "scripts/measurement_study.py")
            and command[2:4] == ["run", "--out"] and absolute_argument(command[4]) == str(folder)
            and command[5:] == ["--registration-sha256", registration["sha256"]], "exact original run command and registration")
    return {"registration": registration, "manifest": descriptor(folder / "manifest.json"),
            "process": descriptor(process), "run_status": descriptor(folder / "run-status.json"),
            "sources": plan["sources"], "parent_result": plan["parent_result"],
            "manifest_members": len(observed), "producer_seconds": seconds, "process_seconds": elapsed}


def load_arrays(path, names, counts):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), "exact NPZ schema: " + path.name)
        result = {name: archive[name].copy() for name in names}
    counts["npz_decodes"] += 1
    counts["array_loads"] += len(result)
    return result


def resident_bytes(method, steps):
    if method[:-3] in KINDS and method.endswith("118"):
        return 1022
    return {"coverage118": 1021, "recent98": 1020, "coverage98": 1020, "full": 40 + 24 * steps}[method]


def validate_state(saved, data, method, matrices):
    expected = reconstructed_state(data["x"], data["y"], method, matrices)
    hybrid = method[:-3] in KINDS and method.endswith("118")
    require(set(saved) == set(expected) | {"step"} | ({"kind"} if hybrid else set()), "complete state and no hidden arrays")
    array(saved["step"], (1,), "event count", np.int64)
    require(saved["step"][0] == data["y"].shape[1], "all actual events processed")
    if hybrid:
        array(saved["kind"], (1,), "serialized kind code", np.uint8)
        require(saved["kind"][0] == KINDS.index(method[:-3]), "immutable sketch kind")
    for name, value in expected.items():
        array(saved[name], value.shape, name, value.dtype)
        if name in ("mask", "ids", "Z") or not hybrid or data["y"].shape[1] <= 118:
            require(saved[name].tobytes() == value.tobytes(), "exact retained state: " + name)
        else:
            close(saved[name], value, "all-input sketch sums")
    actual = sum(saved[key].nbytes for key in expected) // len(data["y"]) + 40 + int(hybrid)
    require(actual == resident_bytes(method, data["y"].shape[1]) <= 1024, "full persistent logical state")


def field_regrets(prediction, reference):
    risk, truth = prediction["risk"], reference["risk"]
    selected = np.concatenate((risk + .02, np.full((*risk.shape[:-1], 1), .20)), -1).argmin(-1)
    costs = np.concatenate((truth + .02, np.full((*truth.shape[:-1], 1), .20)), -1)
    return (np.take_along_axis(costs, selected[..., None], -1)[..., 0] - costs.min(-1)).mean(-1)


def bootstrap(differences, seed):
    values = np.asarray(differences, dtype=np.float64)
    array(values, (384,), "all independent field differences")
    require(type(seed) is int and seed in (553261824, 553261825), "fixed bootstrap seed")
    # Recompute only the prespecified statistical resampling, never GP fields.
    random = np.random.Generator(np.random.PCG64(seed))
    resampled = np.empty(1000)
    for i in range(1000):
        resampled[i] = values[random.integers(384, size=384)].mean()
    lower, upper = np.quantile(resampled, [.025, .975], method="linear")
    return {"difference_mean": float(values.mean()), "ci_low": float(lower), "ci_high": float(upper),
            "context_differences": values.tolist(), "seed": seed, "repetitions": 1000}


def conditions(rows, intervals):
    require(len(rows) == 84 and {(r["population"], r["cohort"], r["method"]) for r in rows} == set(identities()),
            "all 84 unique metric rows")
    keyed = {}
    for row in rows:
        require(set(row) == {"population", "cohort", "method", "metrics"}
                and type(row["cohort"]) is int and set(row["metrics"]) == set(METRICS), "metric row schema")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in row["metrics"].values()), "finite metrics")
        keyed[row["population"], row["cohort"], row["method"]] = row["metrics"]
    require(set(intervals) == {"long", "long_shift"}, "two complete bootstrap intervals")
    output = []
    def mean(population, method, metric):
        return float(np.mean([keyed[population, c, method][metric] for c in range(3)]))
    def add(name, actual, limit, strict=False):
        require(math.isfinite(actual) and math.isfinite(limit), "finite rule inputs")
        output.append({"name": name, "actual": actual, "limit": limit,
                       "comparison": "<" if strict else "<=", "pass": bool(actual < limit if strict else actual <= limit)})
    for population in ("base", "shift"):
        for cohort in range(3):
            add(f"{population}/{cohort}/exact-prefix", keyed[population, cohort, "spectral118"]["regret"], 1e-8)
    for population in ("long", "long_shift"):
        for method in RAW:
            add(f"{population}/mean-versus-{method}", mean(population, "spectral118", "regret"),
                .7 * mean(population, method, "regret") + 1e-6)
    for population in POPULATIONS:
        add(f"{population}/nll", mean(population, "spectral118", "nll"),
            min(mean(population, method, "nll") for method in RAW) + .02)
    for population in ("long", "long_shift"):
        for cohort in range(3):
            add(f"{population}/{cohort}/best-raw", keyed[population, cohort, "spectral118"]["regret"],
                .9 * min(keyed[population, cohort, method]["regret"] for method in RAW) + 1e-6)
        add(f"{population}/defer", mean(population, "spectral118", "regret"),
            mean(population, "spectral118", "always_defer_regret"), True)
        add(f"{population}/paired-upper", intervals[population]["ci_high"], 0., True)
    return output


def validate_resources(resources):
    require(len(resources) == 28 and {(r["population"], r["method"]) for r in resources}
            == {(p, m) for p in POPULATIONS for m in METHODS}, "complete 28 resource rows")
    for row in resources:
        require(set(row) == {"population", "method", "logical_bytes", "median_ms", "all_ms", "peak_python_bytes"},
                "resource row schema")
        require(type(row["logical_bytes"]) is int and row["logical_bytes"]
                == resident_bytes(row["method"], POPULATIONS[row["population"]]["observations"]), "logical bytes")
        require(type(row["peak_python_bytes"]) is int and row["peak_python_bytes"] >= 0, "traced allocation")
        times = row["all_ms"]
        require(isinstance(times, list) and len(times) == 10
                and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times), "all ten positive timing samples")
        close(row["median_ms"], np.median(times), "saved timing median")


def audit(study, *, process, check=lambda: None):
    started = time.perf_counter()
    folder = Path(study).resolve()
    check()
    admission = authenticate(folder, process)
    summary = read(folder / "summary.json")
    require(set(summary) == {"version", "rows", "resources", "conditions", "passed", "total", "gate"}
            and summary["version"] == CONFIG["version"], "summary schema")
    counts = {"npz_decodes": 0, "array_loads": 0, "model_calls": 0, "generator_calls": 0,
              "optimizer_calls": 0, "basis_reconstructions": 3, "state_reconstructions": 0,
              "prediction_groups": 0, "bootstrap_reconstructions": 2}
    matrices = {kind: basis(kind) for kind in KINDS}
    rows, differences = [], {"long": [], "long_shift": []}
    for population, spec in POPULATIONS.items():
        for cohort in range(3):
            check()
            stem = f"{population}-{cohort}"
            data = load_arrays(folder / "data" / (stem + ".npz"), ("x", "y", "paths", "exposure"), counts)
            validate_data(data, 128, spec["observations"], spec["geometry"])
            reference = predict(data, "full", check=check)
            cohort_fields = {}
            for method in METHODS:
                check()
                pred_names = {"mean", "variance", "risk"}
                if method != "full":
                    pred_names |= {"rank", "rank_cutoff"}
                saved = load_arrays(folder / "pred" / f"{stem}-{method}.npz", pred_names, counts)
                reconstructed = reference if method == "full" else predict(data, method, matrices, check)
                for name, expected in reconstructed.items():
                    array(saved[name], expected.shape, name, expected.dtype)
                    if name == "rank":
                        require(np.array_equal(saved[name], expected), "reported numerical rank")
                    elif name == "rank_cutoff":
                        require(np.all(saved[name] >= 0) and np.allclose(saved[name], expected, rtol=RTOL, atol=0.),
                                "relative numerical rank cutoff (raw cutoff exactly zero)")
                    else:
                        close(saved[name], expected, "independent prediction " + name)
                if method != "full":
                    state_names = {"ids", "values", "step"} if method in ("recent98", "coverage98") else {"mask", "values", "step"}
                    if method[:-3] in KINDS and method.endswith("118"):
                        state_names.add("kind")
                    saved_state = load_arrays(folder / "state" / f"{stem}-{method}.npz", state_names, counts)
                    validate_state(saved_state, data, method, matrices)
                    counts["state_reconstructions"] += 1
                prediction = {name: saved[name] for name in ("mean", "variance", "risk")}
                # Metrics use authenticated, independently agreed predictions,
                # preserving their exact action ordering at numerical near ties.
                rows.append({"population": population, "cohort": cohort, "method": method,
                             "metrics": metrics(prediction, reference, data["exposure"])})
                cohort_fields[method] = field_regrets(prediction, reference)
                counts["prediction_groups"] += 1
            if population in differences:
                differences[population].extend((cohort_fields["spectral118"] - cohort_fields["coverage98"]).tolist())
    compare(summary["rows"], rows, "summary metric reconciliation")
    compare(read(folder / "metrics.json"), rows, "raw metric reconciliation")
    intervals = {p: bootstrap(values, CONFIG["namespace"] + CONFIG["bootstrap_offsets"][p]) for p, values in differences.items()}
    compare(read(folder / "bootstrap.json"), intervals, "independent paired field bootstrap")
    tests = conditions(rows, intervals)
    compare(summary["conditions"], tests, "all 26 unchanged conditions")
    compare(summary["passed"], sum(c["pass"] for c in tests), "passed count")
    compare(summary["total"], 26, "condition count")
    gate = "ADVANCE_CONSOLIDATION_BASELINE" if all(c["pass"] for c in tests) else "DO_NOT_ADVANCE_CONSOLIDATION"
    compare(summary["gate"], gate, "unrescued primary result")
    validate_resources(summary["resources"])
    compare(read(folder / "resources.json"), summary["resources"], "resource reconciliation")
    require(counts["npz_decodes"] == 168 and counts["array_loads"] == 696
            and counts["state_reconstructions"] == 72 and counts["prediction_groups"] == 84, "complete decode and reconstruction accounting")
    check()
    require(authenticate(folder, process) == admission, "original evidence and sources unchanged after audit")
    return {"version": VERSION, "agreement": True, "admission": admission, "counts": counts,
            "rows": rows, "resources": summary["resources"], "conditions": tests, "bootstrap": intervals,
            "passed": summary["passed"], "total": 26, "gate": gate, "seconds": time.perf_counter() - started,
            "tolerance": {"rtol": RTOL, "atol": ATOL, "rank_cutoff_atol": 0.},
            "limitations": ["No field regeneration; private exposures are scored as saved.",
                            "No model, optimizer, producer or Gaussian memory implementation is called.",
                            "Timing and historical execution are original source/receipt attestations.",
                            "Logical retained state excludes temporary workspace, Python overhead and native RSS.",
                            "Predictions condition the original law on retained numerical row-space observations; no full-history recovery claim after compression."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--process", required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists() and not args.out.resolve().is_relative_to(args.study.resolve()), "new audit outside original study")
    result = audit(args.study, process=args.process)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"agreement": result["agreement"], "gate": result["gate"], "counts": result["counts"]}), flush=True)


if __name__ == "__main__":
    main()
