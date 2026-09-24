"""Independent saved-output arithmetic for the direct readout diagnostic.

The caller authenticates original process closure before entry. This auditor
reconstructs histories, scalar prediction metrics, projected objectives, export
relationships and TRAIN parity from saved arrays. It neither fits a solver nor
executes a model. Cache extraction, SVD optimality and inference causality remain
qualified producer evidence, not an independently replayed computation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_direct_readout.py"
REFERENCE = "scripts/audit_otto_query_memory.py"
REFERENCE_SHA = "22c83c3f02db95afa1495ca8c37fb108da1f442b0bb0d9fe393f9fed4c7bf4ff"
SHAPE_REFERENCE = "scripts/audit_otto_readout_compute.py"
SHAPE_SHA = "918069d2757a08a9f04c579adb005a98e43753777e1e78e59faf112fcfeb9cd5"
VERSION = "otto-direct-readout-saved-audit-v1"
SEEDS = (309000001, 309000002, 309000003)
ARMS = ("ols", "ridge")
TRAIN_VIEWS = ("pretrained", "action_residual_only", *ARMS)
VIEWS = (*TRAIN_VIEWS, "full_joint")
CONTROLS = TRAIN_VIEWS[:2]
REGIMES = ("lambda3", "lambda4")
PAYLOADS = {
    "started.json", "runtime.json", "progress.jsonl", "fits.json", "views.json",
    "training-barrier.json", "summary.json", "train-history.npz", "train-history.json",
    "dev-history.npz", "dev-history.json", *(f"cache-{seed}.npz" for seed in SEEDS),
    *(f"checkpoint-{arm}-{seed}.npz" for seed in SEEDS for arm in ARMS),
    *(f"train-prediction-{arm}-{seed}.npz" for seed in SEEDS for arm in TRAIN_VIEWS),
    *(f"dev-prediction-{arm}-{seed}.npz" for seed in SEEDS for arm in VIEWS),
}
CACHE_ARRAYS = {"z", "base", "parent_prediction", "query_mask", "prior_mask", "nonquery_mask",
                "support_mask", "episode_offsets", "design_error", "design_legal", "design_weights",
                "theta_parent"}
LIMITATIONS = [
    "Previously exposed DEV paths; no fresh evidence, independent confirmation or autonomous efficacy claim.",
    "The batch-one float64 quadratic is a declared surrogate, not the historical batch-six float32 Adam program.",
    "Saved losses, exported heads, frozen tensors, scalar metrics and parity arithmetic are independently checked.",
    "No model, optimizer, solver, teacher or simulator is called; SVD optimality and cache extraction are not replayed.",
    "Ordinary TRAIN parity is checked against saved model outputs, not a fresh inference execution.",
    "Successful original audit supervisor closure is still required; this result never admits TEST or confirmation.",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)


def near(actual, expected, label, *, atol=1e-11, rtol=1e-8):
    finite(actual, label)
    finite(expected, label)
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), label)


def regular(value):
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "regular contained evidence")
    return path


def descriptor(value):
    path = regular(value)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def verify(pin):
    require(type(pin) is dict and set(pin) == {"path", "sha256", "bytes"}
            and descriptor(pin["path"]) == pin, "unchanged exact evidence descriptor")


def read(pin):
    verify(pin)
    return json.loads(regular(pin["path"]).read_text())


def references(plan):
    result = []
    for path, sha in ((REFERENCE, REFERENCE_SHA), (SHAPE_REFERENCE, SHAPE_SHA)):
        require(plan["sources"][path] == sha and descriptor(ROOT / path)["sha256"] == sha,
                "qualified immutable audit helper")
        spec = importlib.util.spec_from_file_location("_direct_audit_" + Path(path).stem, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result.append(module)
    return result


def theta(np, state):
    return np.column_stack((state["slow.action_residual.weight"],
                            state["slow.action_residual.bias"])).astype(np.float64)


def basis(np):
    # Independent fixed contrast coordinates, not imported from the solver.
    return np.array([[.5, .5, .5], [.5, -.5, -.5], [-.5, .5, -.5], [-.5, -.5, .5]])


def affine(np, cache, head):
    """Reconstruct normalized scores without producer cache or solver imports."""
    result = np.zeros_like(cache["base"], dtype=np.float64)
    support = cache["support_mask"]
    residual = np.einsum("ij,kj->ik", cache["z"][support], head)
    residual -= residual.mean(axis=1, keepdims=True)
    result[support] = cache["base"][support].astype(np.float64) / 64 + residual
    require(bool(np.isfinite(result).all()), "finite reconstructed affine prediction")
    return result


def projected_objective(np, z, error, legal, weights, increment, *, check=lambda: None):
    """Scalar legal centering with fixed weights, including zero-support rows.

    Returns the data loss and its half-gradient in the 87 contrast coordinates.
    This does not construct the producer design matrix or solve normal equations.
    """
    require(z.ndim == 2 and z.shape[1] == 29 and error.shape == legal.shape == (len(z), 4)
            and weights.shape == (len(z),) and increment.shape == (4, 29), "objective geometry")
    require(legal.dtype == np.bool_ and bool(np.isfinite(z).all()) and bool(np.isfinite(error).all())
            and bool(np.isfinite(weights).all()) and bool((weights >= 0).all())
            and bool(np.isfinite(increment).all()), "finite objective inputs")
    losses = []
    normal = np.zeros((3, 29), np.float64)
    U = basis(np)
    for row in range(len(z)):
        if row % 128 == 0:
            check()
        weight = float(weights[row])
        if weight == 0:
            continue
        choices = [a for a in range(4) if legal[row, a]]
        require(bool(choices), "positive-weight legal support")
        residual = {a: math.fsum(float(increment[a, j]) * float(z[row, j]) for j in range(29))
                    - float(error[row, a]) for a in choices}
        mean = math.fsum(residual.values()) / len(choices)
        centered = {a: value - mean for a, value in residual.items()}
        losses.append(weight * math.fsum(value * value for value in centered.values()))
        for contrast in range(3):
            coefficient = weight * math.fsum(float(U[a, contrast]) * centered[a] for a in choices)
            normal[contrast] += coefficient * z[row]
    loss = math.fsum(losses)
    finite(loss, "finite reconstructed projected loss")
    require(bool(np.isfinite(normal).all()), "finite reconstructed normal residual")
    return loss, normal.reshape(-1)


def validate_cache(np, cache, history, parent):
    require(set(cache) == CACHE_ARRAYS, "exact cache arrays")
    total = len(history["targets"])
    masks = {"query_mask": history["query_mask"], "prior_mask": history["prior_mask"],
             "nonquery_mask": ~history["query_mask"],
             "support_mask": ~history["query_mask"] | history["prior_mask"]}
    for name, expected in masks.items():
        require(cache[name].dtype == np.bool_ and cache[name].shape == (total,)
                and cache[name].tobytes() == expected.tobytes(), "exact cache mask " + name)
    require(cache["episode_offsets"].dtype == np.int64
            and cache["episode_offsets"].tobytes() == history["episode_offsets"].tobytes(), "cache complete offsets")
    for name, dtype, shape in (("z", np.float64, (total, 29)), ("base", np.float32, (total, 4)),
                               ("parent_prediction", np.float32, (total, 4)),
                               ("design_error", np.float64, (total, 4)),
                               ("design_weights", np.float64, (total,)),
                               ("theta_parent", np.float64, (4, 29))):
        value = cache[name]
        require(value.dtype == dtype and value.shape == shape and bool(np.isfinite(value).all()),
                "finite typed cache " + name)
    require(bool((cache["z"][:, -1] == 1).all())
            and np.array_equal(cache["z"][:, :28].astype(np.float32).astype(np.float64), cache["z"][:, :28]),
            "unit bias and exactly promoted float32 hidden features")
    unsupported = ~masks["support_mask"]
    for name, value in (("z", cache["z"][:, :28]), ("base", cache["base"]),
                        ("parent_prediction", cache["parent_prediction"])):
        require(value[unsupported].tobytes() == np.zeros_like(value[unsupported]).tobytes(),
                "positive-zero unsupported cache " + name)
    require(cache["theta_parent"].tobytes() == theta(np, parent).tobytes(), "actual parent head in design")
    legal = history["legal"].copy()
    legal[history["prior_mask"]] = True
    require(cache["design_legal"].dtype == np.bool_ and cache["design_legal"].shape == legal.shape
            and cache["design_legal"].tobytes() == legal.tobytes(), "independent legal projectors")
    weights = history["nonquery_weights"] / legal.sum(1) + history["prior_weights"] / 4
    require(cache["design_weights"].tobytes() == weights.tobytes(), "independent fixed episode/action weights")
    error = history["targets"].astype(np.float64) / 64 - affine(np, cache, theta(np, parent))
    require(bool(np.allclose(cache["design_error"], error, rtol=1e-12, atol=1e-12)),
            "teacher minus exact promoted parent affine error")


def parity(np, cache, head, prediction):
    expected = affine(np, cache, head)
    actual = prediction["action_prediction"].astype(np.float64) / 64
    prior = cache["prior_mask"]
    actual[prior] = prediction["corrected_shadow_prior"][prior].astype(np.float64) / 64
    support = cache["support_mask"]
    difference = np.abs(actual[support] - expected[support])
    tolerance = 1e-5 + 1e-5 * np.abs(expected[support])
    return {"passed": bool((difference <= tolerance).all()), "max_abs": float(difference.max(initial=0)),
            "max_tolerance_ratio": float((difference / tolerance).max(initial=0)),
            "supported_rows": int(support.sum())}


def verify_parity(record, expected):
    require(set(record) == set(expected) and record["passed"] is expected["passed"] is True
            and type(record["supported_rows"]) is int and record["supported_rows"] == expected["supported_rows"],
            "complete passing ordinary TRAIN parity")
    near(record["max_abs"], expected["max_abs"], "independent maximum TRAIN discrepancy")
    near(record["max_tolerance_ratio"], expected["max_tolerance_ratio"], "independent TRAIN tolerance ratio",
         atol=1e-8, rtol=1e-7)


def check_fit(np, row, cache, parent, adam, solved, *, check=lambda: None):
    """Independently check objective, contrast coordinates and exported weights."""
    family = row["family"]
    require(family in ARMS, "registered direct solver")
    frozen = [name for name in parent if not name.startswith("slow.action_residual.")]
    require(len(frozen) == 6 and row["frozen_names"] == [name.removeprefix("slow.") for name in frozen]
            and all(solved[name].tobytes() == parent[name].tobytes() == adam[name].tobytes() for name in frozen),
            "six frozen tensors equal original parent")
    result = row["solver"]
    require(set(result) == {"coefficients", "B", "increment", "singular_values", "diagnostics"}, "exact solver result")
    values = {name: np.asarray(result[name], dtype=np.float64) for name in ("coefficients", "B", "increment", "singular_values")}
    for name, shape in (("coefficients", (87,)), ("B", (3, 29)), ("increment", (4, 29))):
        require(values[name].shape == shape and bool(np.isfinite(values[name]).all()), "finite solver " + name)
    coefficient, increment = values["coefficients"], values["increment"]
    require(np.array_equal(coefficient.reshape(3, 29), values["B"])
            and np.array_equal(basis(np) @ values["B"], increment), "fixed contrast head coordinates")
    exported = (theta(np, parent) + increment).astype(np.float32)
    require(solved["slow.action_residual.weight"].tobytes() == exported[:, :28].tobytes()
            and solved["slow.action_residual.bias"].tobytes() == exported[:, 28].tobytes(), "exact float32 solved-head export")
    objective_args = (np, cache["z"], cache["design_error"], cache["design_legal"], cache["design_weights"])
    data_loss, normal = projected_objective(*objective_args, increment, check=check)
    zero = np.zeros((4, 29), np.float64)
    before, _ = projected_objective(*objective_args, zero, check=check)
    adam_loss, _ = projected_objective(*objective_args, theta(np, adam) - theta(np, parent), check=check)
    exported_loss, _ = projected_objective(*objective_args, theta(np, solved) - theta(np, parent), check=check)
    penalty = .0001 if family == "ridge" else 0.
    regularization = penalty * math.fsum(float(v) ** 2 for v in coefficient)
    normal += penalty * coefficient
    normal_norm = math.sqrt(math.fsum(float(v) ** 2 for v in normal))
    diagnostics = result["diagnostics"]
    fields = {"mode": family, "rcond": 1e-10, "ridge": penalty,
              "spectrum_scope": "augmented" if penalty else "data", "data_rows": 4 * len(cache["z"]),
              "feature_dimension": 29, "coordinates": 87,
              "rows": 4 * len(cache["z"]) + (87 if penalty else 0)}
    require(all(type(diagnostics[k]) is type(v) and diagnostics[k] == v for k, v in fields.items()),
            "fixed declared solver diagnostics")
    for key, expected in (("objective_before", before), ("data_loss", data_loss),
                          ("regularization", regularization), ("total_objective", data_loss + regularization),
                          ("normal_residual", normal_norm)):
        near(diagnostics[key], expected, "independent solver " + key)
    spectrum, rank = values["singular_values"], diagnostics["rank"]
    require(spectrum.shape == (min(diagnostics["rows"], 87),) and bool(np.isfinite(spectrum).all())
            and bool((spectrum >= 0).all()) and bool((np.diff(spectrum) <= 0).all())
            and type(rank) is int and 0 <= rank <= len(spectrum), "finite ordered spectrum and rank")
    largest = float(spectrum[0]) if len(spectrum) else 0.
    require(rank == int((spectrum > 1e-10 * largest).sum()) and (not penalty or rank == 87),
            "declared cutoff and nonsingular ridge")
    retained = float(spectrum[rank - 1]) if rank else None
    condition = largest / retained if retained is not None else None
    require(diagnostics["full_column_rank"] is (rank == 87), "full-column rank flag")
    for name, expected in (("largest_singular_value", largest), ("singular_cutoff", 1e-10 * largest),
                           ("smallest_retained_singular_value", retained), ("retained_condition_number", condition),
                           ("condition_number", condition if rank == 87 else None)):
        if expected is None:
            require(diagnostics[name] is None, "rank-deficient optional condition " + name)
        else:
            near(diagnostics[name], expected, "self-consistent spectrum " + name)
    require(set(row["cached_control_losses"]) == set(CONTROLS), "both cache controls retained")
    near(row["cached_control_losses"]["pretrained"], before, "independent parent control loss")
    near(row["cached_control_losses"]["action_residual_only"], adam_loss, "independent Adam control loss")
    near(row["exported_cached_loss"], exported_loss, "independent exported-head loss")
    for name in ("cache_design_seconds", "solve_seconds", "export_seconds", "train_validation_seconds",
                 "standalone_adaptation_seconds"):
        finite(row[name], "finite measured " + name)
    near(row["standalone_adaptation_seconds"], row["cache_metadata"]["seconds"] + sum(row[k] for k in
         ("cache_design_seconds", "solve_seconds", "export_seconds", "train_validation_seconds")), "fully charged adaptation cost")
    return {"family": family, "seed": row["seed"], "data_loss": data_loss, "regularization": regularization,
            "total_objective": data_loss + regularization, "normal_residual": normal_norm,
            "parent_loss": before, "adam74_loss": adam_loss, "exported_cached_loss": exported_loss,
            "rank": rank, "svd_recomputed": False, "frozen_tensors_checked": 6}


def comparisons(reports):
    require(len(reports) == 15 and {(r["family"], r["seed"]) for r in reports}
            == {(family, seed) for seed in SEEDS for family in VIEWS}, "complete fifteen DEV views")
    require(all(r["stage"] == "dev" and r["query_period"] == 4 and r["episodes"] == 36 for r in reports),
            "complete reused DEV P4 reports")
    by = {(r["family"], r["seed"]): r for r in reports}
    cells = []
    for seed in SEEDS:
        for regime in REGIMES:
            gaps, support = {}, []
            for scope in ("full", "later"):
                gaps[scope] = {}
                for family in VIEWS:
                    leaf = by[family, seed]["scopes"][scope]["by_regime"][regime]
                    require(leaf["episodes"] == 18 and leaf["declared_case_count"] == 6,
                            "all equal-case denominators retained")
                    finite(leaf["case_weighted_raw_gap"], "finite diagnostic gap")
                    gaps[scope][family] = leaf["case_weighted_raw_gap"]
                    if scope == "later":
                        support.append(leaf["supported_case_count"])
            require(all(type(n) is int and 0 <= n <= 6 for n in support) and len(set(support)) == 1,
                    "same later-case support")
            gain = {family: gaps["later"]["ridge"] <= .95 * gaps["later"][family]
                    and gaps["later"]["ridge"] < gaps["later"][family] for family in CONTROLS}
            nonreg = {family: gaps["full"]["ridge"] <= gaps["full"][family] for family in CONTROLS}
            cells.append({"seed": seed, "regime": regime, "supported_later_cases": support[0], "gaps": gaps,
                          "later_gain_at_least_5pct": gain, "full_gap_nonregression": nonreg,
                          "passed": all(gain.values()) and all(nonreg.values())})
    return {"candidate": "ridge", "controls": list(CONTROLS), "margin": .05, "cells": cells,
            "passed_cells": sum(c["passed"] for c in cells), "total_cells": 6,
            "overall_passed": all(c["passed"] for c in cells), "dev_reused": True, "fresh_dev": False,
            "held_out_from_training": True, "held_out_evidence": False, "development_only": True,
            "test_admitted": False, "confirmation_admitted": False}


def audit(np, plan, receipt, directory, check):
    directory = Path(directory)
    check()
    require(receipt["version"] == "otto-direct-readout-v1" and receipt["phase"] == "train"
            and receipt["status"] == "completed" and receipt["pending"] is None and receipt["error"] is None,
            "completed direct-readout producer")
    expected_counts = {"fits_completed": 6, "solves_completed": 6, "train_views_completed": 12,
                       "views_completed": 15, "optimizer_steps": 0, "episode_exposures": 0,
                       "checkpoint_decodes": 9, "array_decodes": 11,
                       "teacher_calls": 0, "native_calls": 0, "test_array_decodes": 0}
    require(all(type(receipt[k]) is int and receipt[k] == v for k, v in expected_counts.items()),
            "complete declared producer counts")
    require(set(receipt["files"]) == PAYLOADS and {p.name for p in directory.iterdir()} == PAYLOADS | {"receipt.json"},
            "exact complete producer payloads")
    require(receipt["sources"] == plan["sources"] and SELF in plan["sources"], "same bound audit sources")
    for path, sha in plan["sources"].items():
        check()
        require(descriptor(ROOT / path)["sha256"] == sha, "unchanged registered source")
    files = receipt["files"]
    for name, pin in files.items():
        require(Path(pin["path"]) == directory / name, "exact contained producer payload")
        verify(pin)
    fits, views = read(files["fits.json"]), read(files["views.json"])
    prior_views = read(plan["prior_views"])
    require(len(prior_views) == 9 and {(r["family"], r["seed"]) for r in prior_views}
            == {(f, s) for s in SEEDS for f in ("pretrained", "action_residual_only", "full_joint")},
            "all nine published historical baseline reports")
    prior_reports = {(r["family"], r["seed"]): r["metrics"] for r in prior_views}
    require([(r["family"], r["seed"]) for r in fits] == [(a, s) for s in SEEDS for a in ARMS], "six ordered fits")
    wanted_views = [(stage, family, seed) for stage, families in (("train", TRAIN_VIEWS), ("dev", VIEWS))
                    for seed in SEEDS for family in families]
    require([(r["stage"], r["family"], r["seed"]) for r in views] == wanted_views, "all ordered TRAIN and DEV views")
    barrier = read(files["training-barrier.json"])
    require(set(barrier) == {"solves", "train_views", "dev_decodes", "fits", "created_ns", "checkpoints"}
            and barrier["solves"] == 6 and barrier["train_views"] == 12 and barrier["dev_decodes"] == 0
            and barrier["fits"] == files["fits.json"] and barrier["checkpoints"] == [r["checkpoint"] for r in fits]
            and type(barrier["created_ns"]) is int
            and receipt["started_ns"] <= barrier["created_ns"] <= receipt["finished_ns"], "complete pre-DEV solve/export barrier")
    require(read(files["runtime.json"]) == plan["runtime"], "same qualified runtime")
    require(read(files["summary.json"]) == {"status": "completed_pending_independent_audit", "solves": 6,
            "train_views": 12, "dev_views": 15, "dev_reused": True, "fresh_dev": False}, "unadmitted producer summary")
    with regular(files["progress.jsonl"]["path"]).open() as stream:
        for view in views:
            # The producer writes view completion before attaching its later parity calculation.
            expected = {"event": "view_complete", **{k: v for k, v in view.items() if k != "cache_parity"}}
            line = stream.readline()
            require(line.endswith("\n") and json.loads(line) == expected, "durable ordered view completion")
        require(stream.read() == "", "no extra progress events")
    ref, shapes = references(plan)
    counts = {"array_decodes": 0, "checkpoint_decodes": 0, "cache_decodes": 0, "views_completed": 0,
              "train_views_completed": 0, "dev_views_completed": 0, "fits_checked": 0,
              "model_calls": 0, "optimizer_calls": 0, "solver_calls": 0, "teacher_calls": 0,
              "native_calls": 0, "test_array_decodes": 0}

    def arrays(pin, *, checkpoint=False):
        check()
        verify(pin)
        require(not Path(pin["path"]).name.startswith(("test", "confirm")), "no TEST or confirmation decode")
        with np.load(pin["path"], allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "unique NPZ fields")
            result = {name: archive[name] for name in archive.files}
        counts["array_decodes"] += 1
        counts["checkpoint_decodes"] += int(checkpoint)
        if checkpoint:
            shapes.validate_checkpoint(np, result)
        check()
        return result

    histories = {}
    for stage, count in (("train", 54), ("dev", 36)):
        ids = plan[stage]["identities"]
        require(len(ids) == count and all(r["stage"] == stage for r in ids)
                and len({r["episode_id"] for r in ids}) == count, "complete admitted census identities")
        expected = ref.expected_history(np, arrays(plan[stage]["descriptor"]), ids)
        saved = arrays(files[stage + "-history.npz"])
        require(set(saved) == set(expected), "exact projected history fields")
        for name, value in expected.items():
            require(saved[name].dtype == value.dtype and saved[name].shape == value.shape
                    and saved[name].tobytes() == value.tobytes(), "independent complete history " + name)
        metadata = {"version": "otto-query-memory-data-v1", "query_period": 4, "stage": stage,
                    "episode_ids": [r["episode_id"] for r in ids], "episode_count": count,
                    "identities": [[[name, r[name]] for name in sorted(ref.ID_FIELDS)] for r in ids]}
        ref.close_equal(read(files[stage + "-history.json"]), metadata, "history metadata")
        histories[stage] = saved
    states, caches, fit_checks = {}, {}, []
    fit_by = {(r["family"], r["seed"]): r for r in fits}
    for seed in SEEDS:
        parent = arrays(plan["lineage"]["checkpoints"]["pretrained"][str(seed)], checkpoint=True)
        states["pretrained", seed] = parent
        for family in ("action_residual_only", "full_joint"):
            states[family, seed] = arrays(plan["comparators"][family][str(seed)], checkpoint=True)
        cache = arrays(files[f"cache-{seed}.npz"])
        counts["cache_decodes"] += 1
        validate_cache(np, cache, histories["train"], parent)
        caches[seed] = cache
        for family in ARMS:
            row = fit_by[family, seed]
            require(row["cache"] == files[f"cache-{seed}.npz"]
                    and row["checkpoint"] == files[f"checkpoint-{family}-{seed}.npz"], "fit/cache payload joins")
            metadata = row["cache_metadata"]
            require(set(metadata) == {"version", "work", "seconds", "helper_perf_counter_seconds"}
                    and metadata["version"] == "otto-direct-readout-cache-v1"
                    and type(metadata["work"]) is dict and bool(metadata["work"])
                    and all(type(n) is int and n >= 0 for n in metadata["work"].values()), "complete cache producer metadata")
            finite(metadata["seconds"], "finite cache extraction time")
            finite(metadata["helper_perf_counter_seconds"], "finite inner cache timer")
            require(metadata == fit_by["ols", seed]["cache_metadata"], "shared cache metadata")
            states[family, seed] = arrays(row["checkpoint"], checkpoint=True)
            fit_checks.append(check_fit(np, row, cache, parent, states["action_residual_only", seed],
                                        states[family, seed], check=check))
            counts["fits_checked"] += 1
    reports = []
    for view in views:
        stage, family, seed = view["stage"], view["family"], view["seed"]
        require(view["prediction"] == files[f"{stage}-prediction-{family}-{seed}.npz"], "view payload join")
        finite(view["seconds"], "finite ordinary inference cost")
        history = histories[stage]
        prediction = arrays(view["prediction"])
        shapes.verify_prediction(np, prediction, history)
        report = ref.scalar_report(np, plan[stage]["identities"], history["targets"], history["legal"],
            prediction["action_prediction"], prediction["corrected_shadow_prior"], history["episode_offsets"],
            family, seed, check=check)
        ref.close_equal(view["metrics"], report, "independent complete " + stage + " scalar report")
        if stage == "dev" and family not in ARMS:
            ref.close_equal(report, prior_reports[family, seed], "published baseline reproduced")
        if stage == "train":
            expected = parity(np, caches[seed], theta(np, states[family, seed]), prediction)
            verify_parity(view["cache_parity"], expected)
            if family == "pretrained":
                raw = prediction["action_prediction"].copy()
                mask = history["prior_mask"]
                raw[mask] = prediction["corrected_shadow_prior"][mask]
                raw[~caches[seed]["support_mask"]] = 0
                require(raw.tobytes() == caches[seed]["parent_prediction"].tobytes(), "cache parent equals ordinary predictions")
            if family in ARMS:
                fit = fit_by[family, seed]
                verify_parity(fit["parity"], expected)
                require(fit["train_validation_seconds"] == view["seconds"], "ordinary TRAIN validation cost join")
        reports.append(report)
        counts["views_completed"] += 1
        counts[stage + "_views_completed"] += 1
    require(counts["array_decodes"] == 49 and counts["checkpoint_decodes"] == 15
            and counts["cache_decodes"] == 3 and counts["views_completed"] == 27, "complete saved-array audit accounting")
    check()
    return {"version": VERSION, "status": "completed_pending_original_audit_closure", "agreement": True,
            "technical_complete": False, "requires_original_supervisor_closure": True,
            "counts": counts, "comparisons": comparisons([r for r in reports if r["stage"] == "dev"]),
            "reports": reports, "fit_checks": fit_checks, "limitations": LIMITATIONS}
