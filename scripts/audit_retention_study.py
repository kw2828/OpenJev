"""Independent saved-array audit of bounded GP retention.

Imports no producer, memory, controls, data generator, policy or Torch. Public
evaluation streams and saved drops are replayed with dense Gaussian formulas;
training execution, optimizer history and measured time remain attestations.
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
VERSION = "retention-audit-v1"
RTOL = ATOL = 1e-8
SEEDS = (11, 23, 37)
CONTROLS = ("kl8", "kl9", "fifo9", "fic9", "coverage41", "recent41")
METHODS = tuple(f"learned-{seed}" for seed in SEEDS) + CONTROLS + ("full", "diagonal")
METRICS = ("regret", "nll", "brier", "mse", "coverage90", "defer", "always_defer_regret", "risk_mae")
CONFIG = {
    "version": "retention-v1", "namespace": 443260924, "fit_seeds": list(SEEDS),
    "train_contexts": 256, "train_observations": 64, "queries": 4,
    "epochs": 16, "batch_contexts": 8, "group_size": 4, "learning_rate": .01,
    "entropy_coefficient": .002, "gradient_clip": 1., "cohorts": 3, "evaluation_contexts": 128,
    "state_budget_bytes": 1024,
    "populations": {"base": {"observations": 64, "geometry": "axial", "offset": 100},
                    "shift": {"observations": 64, "geometry": "diagonal", "offset": 200},
                    "long": {"observations": 192, "geometry": "axial", "offset": 300}},
    "methods": list(METHODS), "warmups": 3, "timing_repeats": 10, "wall_cap_seconds": 1800,
}
SOURCES = {
    "src/openjev/research/retention_memory.py", "src/openjev/research/retention_data.py",
    "src/openjev/research/retention_controls.py", "src/openjev/research/retention_policy.py",
    "scripts/retention_study.py", "scripts/audit_retention_study.py",
    "tests/test_retention_memory.py", "tests/test_retention_data.py", "tests/test_retention_controls.py",
    "tests/test_retention_policy.py", "tests/test_retention_study.py", "tests/test_audit_retention_study.py",
    "research/retention-protocol.md",
}
POLICY_SHAPES = {"hidden.weight": (4, 6), "hidden.bias": (4,), "output.weight": (1, 4), "output.bias": (1,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def descriptor(path):
    require(path.is_file() and not path.is_symlink(), "ordinary evidence file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


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


def identities():
    return [(phase, cohort, method) for phase in CONFIG["populations"] for cohort in range(3) for method in METHODS]


def expected_files():
    names = {"registration.json", "started.json", "run-receipt.json", "metrics.json", "resources.json",
             "diagnostics.json", "summary.json", "train-data.npz", "train-reference.npz"}
    for seed in SEEDS:
        names.update({f"policy-{seed}-initial.npz", f"policy-{seed}-final.npz", f"optimizer-{seed}-final.pt",
                      f"training-{seed}.npz", f"training-{seed}.json"})
    for phase, cohort, method in identities():
        stem = f"{phase}-{cohort}"
        names.update({stem + "-data.npz", f"{stem}-{method}-prediction.npz"})
        if method not in ("full", "diagonal"):
            names.add(f"{stem}-{method}-state.npz")
    return names | {"source/" + name for name in SOURCES}


def authenticate(folder):
    plan = read(folder / "registration.json")
    require(plan["config"] == CONFIG and set(plan["sources"]) == SOURCES, "fixed config and thirteen sources")
    for name, expected in plan["sources"].items():
        require(descriptor(ROOT / name) == expected == descriptor(folder / "source" / name), "source and snapshot pins")
    require(plan["parent_result"] == descriptor(ROOT / "research/residual-memory-results/summary.json"), "parent result pin")
    environment = plan["environment"]
    require(environment["python"] == sys.version and environment["numpy"] == np.__version__
            and environment["scipy"] == scipy.__version__ and environment["platform"] == platform.platform()
            and environment["threads"] == 1, "registered numerical runtime")
    manifest = read(folder / "manifest.json")
    require(set(manifest) == expected_files(), "complete registered producer roster")
    actual = {p.relative_to(folder).as_posix(): descriptor(p) for p in folder.rglob("*")
              if p.is_file() and p != folder / "manifest.json"}
    require(actual == manifest, "complete opaque manifest before array decoding")
    receipt = read(folder / "run-receipt.json")
    require(receipt["state"] == "EXITED" and type(receipt["exit_code"]) is int and receipt["exit_code"] == 0
            and receipt["training_updates"] == 1536 and type(receipt["elapsed_seconds"]) in (int, float)
            and math.isfinite(receipt["elapsed_seconds"]) and 0 < receipt["elapsed_seconds"] <= 1800,
            "closed successful original producer")
    return {"registration": descriptor(folder / "registration.json"), "manifest": descriptor(folder / "manifest.json"),
            "receipt": descriptor(folder / "run-receipt.json"), "sources": plan["sources"],
            "parent_result": plan["parent_result"], "producer_seconds": receipt["elapsed_seconds"],
            "manifest_members": len(manifest)}


def load_arrays(path, keys, counts):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(keys), "exact NPZ keys: " + path.name)
        result = {key: archive[key].copy() for key in keys}
    counts["npz_decodes"] += 1
    counts["array_loads"] += len(result)
    return result


def array(value, shape, name, dtype=np.float64):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype) and value.shape == shape
            and np.isfinite(value).all(), "finite declared array: " + name)


def validate_data(data, contexts, steps, geometry):
    require(set(data) == {"x", "y", "paths", "exposure"}, "public/private data schema")
    for key, shape in (("x", (contexts, steps, 2)), ("y", (contexts, steps)),
                       ("paths", (contexts, 4, 4, 4, 2)), ("exposure", (contexts, 4, 4))):
        array(data[key], shape, key)
    for name in ("x", "paths"):
        value = data[name]
        require((np.abs(value) <= 2).all() and np.array_equal(value * 4, np.rint(value * 4)), "fixed grid coordinates")
    for coordinates in data["x"]:
        require(len({tuple(point) for point in coordinates}) == steps, "unique stream coordinates")
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


def kernel(x, z):
    difference = x[..., :, None, :] - z[..., None, :, :]
    return np.exp(-.5 * np.sum(difference**2, axis=-1)) + 1e-5*np.all(difference == 0, axis=-1)


def path_moments(mean, covariance, queries):
    batch = mean.shape[0]
    path_mean = mean.reshape(batch, queries, 4, 4).mean(-1)
    path_cov = covariance.reshape(batch, queries*4, 4, queries*4, 4)
    variance = np.stack([np.sum(path_cov[:, action, :, action, :], axis=(1, 2))/16
                         for action in range(queries*4)], -1).reshape(batch, queries, 4)
    require(np.isfinite(path_mean).all() and np.isfinite(variance).all() and (variance > 0).all(), "positive path variance")
    return {"mean": path_mean, "variance": variance, "risk": ndtr((path_mean-.5)/np.sqrt(variance))}


def exact_reference(x, y, paths):
    means, covariances = [], []
    for coordinates, labels, requests in zip(x, y, paths, strict=True):
        points = requests.reshape(-1, 2)
        factor = cho_factor(kernel(coordinates, coordinates)+.09*np.eye(len(labels)), lower=True)
        cross = kernel(points, coordinates)
        means.append(cross @ cho_solve(factor, labels))
        covariances.append(kernel(points, points)-cross @ cho_solve(factor, cross.T))
    point_mean, covariance = np.stack(means), np.stack(covariances)
    result = path_moments(point_mean, covariance, paths.shape[1])
    diagonal = np.diagonal(covariance, axis1=1, axis2=2).reshape(paths.shape[:4]).sum(-1)/16
    result.update(diag_variance=diagonal, diag_risk=ndtr((result["mean"]-.5)/np.sqrt(diagonal)))
    return result


def state_prediction(state, paths, method):
    if method in ("coverage41", "recent41"):
        result = exact_reference(state["Z"], state["y"], paths)
        return {name: result[name] for name in ("mean", "variance", "risk")}
    points = paths.reshape(paths.shape[0], -1, 2)
    prior = kernel(state["Z"], state["Z"])
    cross = kernel(points, state["Z"])
    coefficient = np.linalg.solve(prior, cross.swapaxes(-1, -2)).swapaxes(-1, -2)
    means = np.einsum("bni,bi->bn", coefficient, state["mean"])
    covariance = coefficient @ state["cov"] @ coefficient.swapaxes(-1, -2)
    if method == "fic9":
        residual = 1.00001 - np.sum(coefficient*cross, axis=-1)
        require((residual >= 0).all(), "nonnegative FIC event residual")
        covariance += np.eye(points.shape[1])[None]*residual[:, :, None]
    else:
        covariance += kernel(points, points)-coefficient @ cross.swapaxes(-1, -2)
    return path_moments(means, covariance, paths.shape[1])


def deletion_priorities(z, mean, covariance, policy=None):
    prior = kernel(z, z)
    inverse_prior, inverse_cov = np.linalg.inv(prior), np.linalg.inv(covariance)
    pdiag = np.diagonal(inverse_prior, axis1=1, axis2=2)
    qdiag = np.diagonal(inverse_cov, axis1=1, axis2=2)
    r, v = 1/pdiag, 1/qdiag
    normalized = inverse_prior/pdiag[:, :, None]
    residual_mean = np.einsum("bij,bj->bi", normalized, mean)
    residual_variance = np.einsum("bij,bjk,bik->bi", normalized, covariance, normalized)
    raw = .5*(np.log(r/v)+(residual_variance+residual_mean**2)/r-1)
    require(np.isfinite(raw).all() and (raw >= -1e-8).all(), "independent KL roundoff")
    kl = np.maximum(raw, 0.)
    if policy is None:
        return kl
    features = np.stack((z[:, :, 0]/2, z[:, :, 1]/2, residual_mean/np.sqrt(r), np.log(v/r),
                         np.log1p(kl), np.diagonal(covariance, axis1=1, axis2=2)/1.00001), axis=-1)
    hidden = np.tanh(features @ policy["hidden.weight"].T + policy["hidden.bias"])
    correction = 2*np.tanh(hidden @ policy["output.weight"].T + policy["output.bias"])[..., 0]
    return np.log(kl+1e-12)+correction


def replay_projected(x, y, traces, method, policy=None, *, check=lambda: None):
    batch, steps = y.shape
    capacity = 8 if method.startswith("learned-") or method == "kl8" else 9
    z, mean, covariance = np.empty((batch, 0, 2)), np.empty((batch, 0)), np.empty((batch, 0, 0))
    near_ties, priority_checks = 0, 0
    for t in range(steps):
        check()
        size = mean.shape[1]
        if size:
            coefficient = np.linalg.solve(kernel(z, z), kernel(z, x[:, t:t+1]))
            extra_mean = np.einsum("bi,bi->b", coefficient[..., 0], mean)
            posterior_cross = covariance @ coefficient
            extra_variance = 1.00001 - np.sum(kernel(z, x[:, t:t+1])*coefficient, axis=(1, 2))
            extra_variance += np.sum(coefficient*posterior_cross, axis=(1, 2))
        else:
            extra_mean, extra_variance = np.zeros(batch), np.full(batch, 1.00001)
            posterior_cross = np.empty((batch, 0, 1))
        extended = np.empty((batch, size+1, size+1))
        extended[:, :size, :size] = covariance
        extended[:, :size, size:] = posterior_cross
        extended[:, size:, :size] = posterior_cross.swapaxes(-1, -2)
        extended[:, size, size] = extra_variance
        mean = np.concatenate((mean, extra_mean[:, None]), axis=1)
        z = np.concatenate((z, x[:, t:t+1]), axis=1)
        gain_numerator = extended[:, :, -1].copy()
        denominator = extra_variance+.09
        mean += gain_numerator*(y[:, t]-extra_mean)[:, None]/denominator[:, None]
        covariance = extended-gain_numerator[:, :, None]*gain_numerator[:, None, :]/denominator[:, None, None]
        covariance = (covariance+covariance.swapaxes(-1, -2))/2
        if size+1 <= capacity:
            require(np.all(traces[:, t] == -1), "no deletion before capacity")
            continue
        drop = traces[:, t].astype(np.int64)
        require(np.all((drop >= 0) & (drop <= capacity)), "legal saved deletion")
        if method == "fifo9":
            require(np.all(drop == 0), "FIFO deletes oldest")
        else:
            priorities = deletion_priorities(z, mean, covariance, policy)
            minimum = priorities.min(-1)
            tolerance = ATOL+RTOL*np.abs(minimum)
            chosen = priorities[np.arange(batch), drop]
            require(np.all(chosen <= minimum+tolerance), "saved deletion minimizes independent priority")
            near_ties += int(np.sum(np.sum(priorities <= minimum[:, None]+tolerance[:, None], axis=-1) > 1))
            priority_checks += batch
        keep = (np.arange(capacity+1)[None] != drop[:, None])
        indices = np.broadcast_to(np.arange(capacity+1), keep.shape)[keep].reshape(batch, capacity)
        z = z[np.arange(batch)[:, None], indices]
        mean = mean[np.arange(batch)[:, None], indices]
        covariance = covariance[np.arange(batch)[:, None, None], indices[:, :, None], indices[:, None, :]]
    return {"Z": z, "mean": mean, "cov": covariance}, {"priority_checks": priority_checks, "near_tie_priority_checks": near_ties}


def replay_raw(x, y, method):
    z, labels = x[:, :41].copy(), y[:, :41].copy()
    for t in range(41, y.shape[1]):
        augmented = np.concatenate((z, x[:, t:t+1]), axis=1)
        values = np.concatenate((labels, y[:, t:t+1]), axis=1)
        if method == "recent41":
            drop = np.zeros(y.shape[0], dtype=int)
        else:
            differences = augmented[:, :, None]-augmented[:, None, :]
            distances = np.sum(differences**2, axis=-1)
            distances[:, np.arange(42), np.arange(42)] = np.inf
            drop = np.argmin(distances.min(-1), axis=-1)
        keep = np.arange(42)[None] != drop[:, None]
        z, labels = augmented[keep].reshape(-1, 41, 2), values[keep].reshape(-1, 41)
    return {"Z": z, "y": labels}


def batch_fic(x, y):
    anchors = np.array([(a, b) for a in (-1.8, .1, 1.8) for b in (-1.8, .1, 1.8)])
    prior = kernel(anchors, anchors)
    coefficient = np.linalg.solve(prior, kernel(anchors[None], x)).swapaxes(-1, -2)
    residual = 1.00001-np.sum(coefficient*kernel(x, anchors[None]), axis=-1)
    require((residual >= 0).all(), "nonnegative independent FIC residual")
    variance = residual+.09
    precision = np.linalg.inv(prior)[None]+np.einsum("bni,bnj,bn->bij", coefficient, coefficient, 1/variance)
    eta = np.einsum("bni,bn->bi", coefficient, y/variance)
    covariance = np.linalg.inv(precision)
    return {"Z": np.broadcast_to(anchors, (x.shape[0], 9, 2)).copy(),
            "mean": np.einsum("bij,bj->bi", covariance, eta), "cov": covariance}


def resident_bytes(method, steps):
    if method in ("full", "diagonal"):
        return 24*steps+40
    if method in ("coverage41", "recent41"):
        return 1024
    if method.startswith("learned-"):
        return 1008
    return 744 if method == "kl8" else 904


def validate_state(state, data, method, policy, counts, check):
    batch, steps = data["y"].shape
    raw = method in ("coverage41", "recent41")
    size = 41 if raw else 8 if method.startswith("learned-") or method == "kl8" else 9
    keys = {"Z", "y"} if raw else {"Z", "mean", "cov"}
    require(set(state) == keys | {"step", "traces", "resident_bytes"}, "state has no hidden arrays")
    array(state["Z"], (batch, size, 2), "state coordinates")
    if raw:
        array(state["y"], (batch, size), "retained labels")
    else:
        array(state["mean"], (batch, size), "posterior mean")
        array(state["cov"], (batch, size, size), "posterior covariance")
        close(state["cov"], state["cov"].swapaxes(-1, -2), "symmetric posterior")
        np.linalg.cholesky(state["cov"])
    array(state["step"], (), "step", np.int64)
    array(state["resident_bytes"], (), "resident bytes", np.int64)
    array(state["traces"], (batch, steps), "drop trace", np.int16)
    require(int(state["step"]) == steps, "all observations processed")
    physical = sum(state[key].nbytes for key in keys)//batch+40
    expected = resident_bytes(method, steps)-(264 if method.startswith("learned-") else 0)
    require(int(state["resident_bytes"]) == expected == physical, "actual retained state byte accounting")
    require(resident_bytes(method, steps) <= 1024, "bounded state including policy")
    if raw:
        require(np.all(state["traces"] == -1), "raw control has no projected drop trace")
        replayed = replay_raw(data["x"], data["y"], method)
        for name in keys:
            require(np.array_equal(state[name], replayed[name]), "exact raw retention and label pairing")
        counts["raw_state_reconstructions"] += 1
    elif method == "fic9":
        require(np.all(state["traces"] == -1), "fixed FIC has no projected drop trace")
        replayed = batch_fic(data["x"], data["y"])
        for name in keys:
            close(state[name], replayed[name], "batch FIC state " + name)
        counts["fic_state_reconstructions"] += 1
    else:
        replayed, work = replay_projected(data["x"], data["y"], state["traces"], method, policy, check=check)
        require(np.array_equal(state["Z"], replayed["Z"]), "saved projected dictionary replay")
        for name in ("mean", "cov"):
            close(state[name], replayed[name], "one-assimilation projected replay " + name)
        counts["projected_state_replays"] += 1
        for name, value in work.items():
            counts[name] += value


def validate_policy(values, initial=False):
    require(set(values) == set(POLICY_SHAPES), "exact four policy tensors")
    for name, shape in POLICY_SHAPES.items():
        array(values[name], shape, name)
    require(sum(value.nbytes for value in values.values()) == 264, "33 float64 policy parameters")
    if initial:
        require(np.all(values["output.weight"] == 0) and np.all(values["output.bias"] == 0), "zero initial residual policy")


def validate_training(record, arrays, seed):
    require(set(record) == {"fit_seed", "updates", "trace", "seconds", "parameter_bytes", "group_trajectories"}
            and record["fit_seed"] == seed and record["updates"] == 512 and record["parameter_bytes"] == 264
            and record["group_trajectories"] == 16384, "complete fixed training record")
    require(type(record["seconds"]) in (float, int) and math.isfinite(record["seconds"]) and record["seconds"] > 0,
            "positive training time")
    array(arrays["actions"], (512, 32, 64), "training actions", np.int16)
    array(arrays["rewards"], (512, 8, 4), "training rewards")
    array(arrays["fields"], (512, 8), "training fields", np.int64)
    require(np.all(arrays["actions"][:, :, :8] == -1)
            and np.all((arrays["actions"][:, :, 8:] >= 0) & (arrays["actions"][:, :, 8:] <= 8)), "training action domains")
    require(np.all((arrays["rewards"] >= -1-ATOL) & (arrays["rewards"] <= ATOL)), "bounded conditional-regret rewards")
    for epoch in range(16):
        require(np.array_equal(np.sort(arrays["fields"][32*epoch:32*(epoch+1)].ravel()), np.arange(256)),
                "every training field once per epoch")
    require(len(record["trace"]) == 512, "all training update rows")
    for i, row in enumerate(record["trace"]):
        require(set(row) == {"epoch", "update", "loss", "mean_reward", "group_reward_std", "entropy",
                             "gradient_norm", "kl_roundoff_floors"}
                and row["epoch"] == i//32 and row["update"] == i+1, "training trace schema and order")
        for key in ("loss", "mean_reward", "group_reward_std", "entropy", "gradient_norm"):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]), "finite training diagnostic")
        require(0 <= row["entropy"] <= math.log(9)+ATOL and row["gradient_norm"] >= 0
                and type(row["kl_roundoff_floors"]) is int and 0 <= row["kl_roundoff_floors"] <= 32*56*9,
                "training entropy, gradient and floor-count bounds")
        close(row["mean_reward"], arrays["rewards"][i].mean(), "recorded mean reward")
        close(row["group_reward_std"], arrays["rewards"][i].std(axis=1).mean(), "group reward spread")


def metrics(prediction, reference, exposure):
    shape = exposure.shape
    require(set(prediction) == {"mean", "variance", "risk"}, "prediction schema")
    for name in prediction:
        array(prediction[name], shape, name)
    require(np.all(prediction["variance"] > 0) and np.all((prediction["risk"] >= 0) & (prediction["risk"] <= 1)),
            "positive Gaussian variance and bounded risk")
    close(prediction["risk"], ndtr((prediction["mean"]-.5)/np.sqrt(prediction["variance"])), "Gaussian tail risk")
    costs = np.concatenate((.02+prediction["risk"], np.full((*shape[:-1], 1), .2)), axis=-1)
    reference_costs = np.concatenate((.02+reference["risk"], np.full((*shape[:-1], 1), .2)), axis=-1)
    chosen = costs.argmin(axis=-1)
    optimum = reference_costs.min(axis=-1)
    errors = exposure-prediction["mean"]
    return {"regret": float(np.mean(np.take_along_axis(reference_costs, chosen[..., None], axis=-1)[..., 0]-optimum)),
            "nll": float(np.mean(.5*(math.log(2*math.pi)+np.log(prediction["variance"])+errors**2/prediction["variance"]))),
            "brier": float(np.mean((prediction["risk"]-(exposure > .5))**2)), "mse": float(np.mean(errors**2)),
            "coverage90": float(np.mean(np.abs(errors) <= 1.6448536269514722*np.sqrt(prediction["variance"]))),
            "defer": float(np.mean(chosen == 4)), "always_defer_regret": float(np.mean(.2-optimum)),
            "risk_mae": float(np.mean(np.abs(prediction["risk"]-reference["risk"])))}


def classification(rows):
    require(len(rows) == 99 and {(r["phase"], r["cohort"], r["method"]) for r in rows} == set(identities()),
            "all 99 unselected metric groups")
    means, checks = {}, []
    learned = [f"learned-{seed}" for seed in SEEDS]
    for phase in CONFIG["populations"]:
        means[phase] = {method: {metric: float(np.mean([r[metric] for r in rows
                         if r["phase"] == phase and r["method"] == method])) for metric in METRICS} for method in METHODS}
        candidate = {metric: float(np.mean([means[phase][method][metric] for method in learned])) for metric in METRICS}
        means[phase]["learned_mean"] = candidate
        factor = 1.05 if phase == "shift" else .90
        for control in CONTROLS:
            checks.append({"name": f"{phase}_regret_vs_{control}",
                           "passed": candidate["regret"] <= factor*means[phase][control]["regret"]+1e-6})
        checks.extend([{"name": f"{phase}_nll", "passed": candidate["nll"] <= min(means[phase][c]["nll"] for c in CONTROLS)+.02},
                       {"name": f"{phase}_beats_defer", "passed": candidate["regret"] < candidate["always_defer_regret"]}])
        for cohort in range(3):
            group = {r["method"]: r for r in rows if r["phase"] == phase and r["cohort"] == cohort}
            bound = min(group[control]["regret"] for control in CONTROLS)
            value = np.mean([group[method]["regret"] for method in learned])
            checks.append({"name": f"{phase}_cohort_{cohort}",
                           "passed": bool(value <= (1.05 if phase == "shift" else 1.)*bound+1e-6)})
        bound = min(means[phase][control]["regret"] for control in CONTROLS)
        for method in learned:
            checks.append({"name": f"{phase}_{method}",
                           "passed": means[phase][method]["regret"] <= (1.05 if phase == "shift" else 1.)*bound+1e-6})
    return {"means": means, "checks": checks,
            "gate": "ADVANCE_RETENTION" if all(r["passed"] for r in checks) else "DO_NOT_ADVANCE_RETENTION",
            "scope": "Learned deletion pilot on supplied known-law GP fields; no general architecture claim."}


def validate_resources(rows):
    require(len(rows) == 33 and {(r["phase"], r["method"]) for r in rows}
            == {(phase, method) for phase in CONFIG["populations"] for method in METHODS}, "33 resource rows")
    for row in rows:
        require(set(row) == {"phase", "method", "timings_seconds", "median_seconds", "resident_bytes",
                             "python_numpy_traced_peak_bytes", "traced_current_bytes", "native_workspace_measured",
                             "retained_gradient_tensors", "scope"},
                "resource row schema")
        require(len(row["timings_seconds"]) == 10 and all(type(v) in (float, int) and math.isfinite(v) and v > 0
                                                        for v in row["timings_seconds"]), "ten actual timing repeats")
        close(row["median_seconds"], float(np.median(row["timings_seconds"])), "timing median")
        require(row["resident_bytes"] == resident_bytes(row["method"], CONFIG["populations"][row["phase"]]["observations"]),
                "resource retained bytes include all policy weights")
        require(type(row["python_numpy_traced_peak_bytes"]) is int and type(row["traced_current_bytes"]) is int
                and row["python_numpy_traced_peak_bytes"] >= row["traced_current_bytes"] >= 0
                and row["native_workspace_measured"] is False
                and type(row["retained_gradient_tensors"]) is int and row["retained_gradient_tensors"] == 0,
                "limited allocation tracer scope and no retained gradients")
        require(row["scope"] == "One context, complete stream then four requests; native allocations and peak RSS unmeasured.",
                "resource scope unchanged")


def audit(folder, output=None, *, check=lambda: None):
    started = time.perf_counter()
    folder = Path(folder).resolve()
    if output is not None:
        output = Path(output).resolve()
        require(not output.exists() and not output.is_relative_to(folder), "exclusive audit outside producer folder")
    admission = authenticate(folder)
    counts = dict.fromkeys(("npz_decodes", "array_loads", "projected_state_replays", "raw_state_reconstructions",
                           "fic_state_reconstructions", "priority_checks", "near_tie_priority_checks"), 0)
    policies, training = {}, []
    data = load_arrays(folder / "train-data.npz", ("x", "y", "paths", "exposure"), counts)
    validate_data(data, 256, 64, "axial")
    saved = load_arrays(folder / "train-reference.npz", ("mean", "variance", "risk", "diag_variance", "diag_risk"), counts)
    reference = exact_reference(data["x"], data["y"], data["paths"])
    for name in reference:
        array(saved[name], (256, 4, 4), "training reference " + name)
        close(saved[name], reference[name], "training full-history reference")
    for seed in SEEDS:
        check()
        initial = load_arrays(folder / f"policy-{seed}-initial.npz", POLICY_SHAPES, counts)
        final = load_arrays(folder / f"policy-{seed}-final.npz", POLICY_SHAPES, counts)
        validate_policy(initial, initial=True)
        validate_policy(final)
        policies[seed] = final
        record = read(folder / f"training-{seed}.json")
        arrays = load_arrays(folder / f"training-{seed}.npz", ("actions", "rewards", "fields"), counts)
        validate_training(record, arrays, seed)
        training.append({key: value for key, value in record.items() if key != "trace"})
    saved_rows = read(folder / "metrics.json")
    require(len(saved_rows) == 99 and [(r["phase"], r["cohort"], r["method"]) for r in saved_rows] == identities(),
            "complete ordered metric rows")
    rows = []
    for phase, population in CONFIG["populations"].items():
        for cohort in range(3):
            check()
            stem = f"{phase}-{cohort}"
            data = load_arrays(folder / (stem + "-data.npz"), ("x", "y", "paths", "exposure"), counts)
            validate_data(data, 128, population["observations"], population["geometry"])
            reference = exact_reference(data["x"], data["y"], data["paths"])
            for method in METHODS:
                check()
                saved = load_arrays(folder / f"{stem}-{method}-prediction.npz", ("mean", "variance", "risk"), counts)
                if method in ("full", "diagonal"):
                    predicted = {name: reference[name] for name in ("mean", "variance", "risk")}
                    if method == "diagonal":
                        predicted.update(variance=reference["diag_variance"], risk=reference["diag_risk"])
                else:
                    keys = ("Z", "y") if method in ("coverage41", "recent41") else ("Z", "mean", "cov")
                    state = load_arrays(folder / f"{stem}-{method}-state.npz", (*keys, "step", "traces", "resident_bytes"), counts)
                    policy = policies[int(method.split("-")[1])] if method.startswith("learned-") else None
                    validate_state(state, data, method, policy, counts, check)
                    predicted = state_prediction(state, data["paths"], method)
                for name in predicted:
                    close(saved[name], predicted[name], "independent state-to-path " + name)
                row = {"phase": phase, "cohort": cohort, "method": method, **metrics(saved, reference, data["exposure"])}
                compare(saved_rows[len(rows)], row, "all saved scores")
                rows.append(row)
    result = classification(rows)
    compare(read(folder / "summary.json"), result, "42 registered requirements")
    require(read(folder / "run-receipt.json")["gate"] == result["gate"], "original gate binding")
    resources = read(folder / "resources.json")
    validate_resources(resources)
    diagnostics = read(folder / "diagnostics.json")
    require(len(diagnostics) == 99 and [(r["phase"], r["cohort"], r["method"]) for r in diagnostics] == identities(),
            "all diagnostic rows")
    for row in diagnostics:
        require(set(row) == {"phase", "cohort", "method", "kl_roundoff_floors"}
                and type(row["kl_roundoff_floors"]) is int and row["kl_roundoff_floors"] >= 0, "diagnostic floor counters")
        capacity = 8 if row["method"].startswith("learned-") or row["method"] == "kl8" else 9
        bound = 128*(CONFIG["populations"][row["phase"]]["observations"]-capacity)*(capacity+1)
        if row["method"] not in ("kl8", "kl9") and not row["method"].startswith("learned-"):
            bound = 0
        require(row["kl_roundoff_floors"] <= bound, "floor counter operation domain")
    require(counts["npz_decodes"] == 200 and counts["array_loads"] == 843
            and counts["projected_state_replays"] == 54 and counts["raw_state_reconstructions"] == 18
            and counts["fic_state_reconstructions"] == 9, "complete independent audit coverage")
    require(authenticate(folder) == admission, "all inputs unchanged after audit")
    report = {"version": VERSION, "agreement": True, "result": result, "rows": rows, "resources": resources,
              "training": training, "diagnostics": diagnostics, "counts": counts | {"prediction_groups": 99,
              "metric_conditions": 42, "state_archives": 81, "optimizer_decodes": 0, "model_calls": 0,
              "generator_calls": 0, "training_updates_replayed": 0}, "admission": admission,
              "tolerance": {"rtol": RTOL, "atol": ATOL}, "seconds": time.perf_counter()-started,
              "limitations": ["Evaluation writes are independently replayed; training optimizer execution is not replayed.",
                  "Near-tie priority checks use declared tolerance. Exact historical first-index ties are source-attested.",
                  "Timing, RNG provenance, warmups, temporary native memory and roundoff-floor counts remain producer attestations.",
                  "Private exposures are scored as saved; no latent-field generation replay is performed.",
                  "Publication still requires the independent original audit process to close successfully."]}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.folder, args.output)
    print(json.dumps({"agreement": report["agreement"], "gate": report["result"]["gate"], "counts": report["counts"]}))


if __name__ == "__main__":
    main()
