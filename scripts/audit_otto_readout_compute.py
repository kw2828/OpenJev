"""Independent saved-output audit of the calibrated readout-compute comparison.

The supervised caller authenticates the original producer process and source
closure before calling ``audit``. This module verifies complete payloads before
decoding, then uses the qualified independent scalar auditor, never producer
metric or gate arithmetic. Optimization, inference causality and wall time are
source-tested producer evidence, not numerically replayed here. Its result still
requires successful closure of its own original audit supervisor.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_otto_readout_compute.py"
REFERENCE = "scripts/audit_otto_query_memory.py"
REFERENCE_SHA256 = "22c83c3f02db95afa1495ca8c37fb108da1f442b0bb0d9fe393f9fed4c7bf4ff"
VERSION = "otto-readout-compute-saved-audit-v1"
PRODUCER_VERSION = "otto-readout-compute-v1"
SEEDS = (309000001, 309000002, 309000003)
ARMS = ("action_residual_only", "full_joint")
EPOCHS = {"action_residual_only": 74, "full_joint": 40}
VIEWS = ("pretrained", *ARMS)
READOUTS = ARMS[:1]
REGIMES = ("lambda3", "lambda4")
SHAPES = {
    "slow.recurrent.weight_ih_l0": (84, 40), "slow.recurrent.weight_hh_l0": (84, 28),
    "slow.recurrent.bias_ih_l0": (84,), "slow.recurrent.bias_hh_l0": (84,),
    "slow.output.weight": (4, 28), "slow.output.bias": (4,),
    "slow.action_residual.weight": (4, 28), "slow.action_residual.bias": (4,),
}
ID_FIELDS = ("stage", "episode_id", "episode_index", "seed", "case", "regime", "arm")
PREDICTION_FIELDS = {"action_prediction", "corrected_shadow_prior", "prior_mask", "episode_offsets"}
PAYLOADS = {
    "started.json", "runtime.json", "fits.json", "views.json", "training-barrier.json",
    "train-history.npz", "train-history.json", "dev-history.npz", "dev-history.json",
    "work.jsonl", "progress.jsonl", "summary.json",
    *(f"checkpoint-{arm}-{seed}.npz" for seed in SEEDS for arm in ARMS),
    *(f"prediction-{view}-{seed}.npz" for seed in SEEDS for view in VIEWS),
}
TIMINGS = {"wall", "setup", "materialize", "forward", "loss", "backward", "carry_detach",
           "gradient_completion", "gradient_clip", "optimizer", "overhead"}
LIMITATIONS = [
    "Fresh registered DEV paths; no independent confirmation, autonomous efficacy or novelty claim.",
    "74 versus 40 epochs are fixed from historical timing calibration, not exact matched FLOPs or live time quotas.",
    "Compute comparability requires each paired elapsed continuation-time ratio in [0.9,1.1]; fits are never extended.",
    "Saved scalar metrics, query bytes, frozen tensors, orders and scheduled work are independently checked.",
    "No model, optimizer, teacher or simulator is called; gradients and inference are not numerically replayed.",
    "Hidden-state causality, optimization, runtime and pre-DEV ordering remain source-tested producer evidence.",
    "Readout proximity is descriptive, not statistical equivalence or noninferiority.",
    "This result requires its own original successful audit supervisor closure and never admits TEST or confirmation.",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)


def regular(value):
    path = Path(value)
    require(path.is_absolute() and path.is_relative_to(ROOT) and ".." not in path.parts
            and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "regular contained audit evidence")
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
            and descriptor(pin["path"]) == pin, "unchanged complete evidence descriptor")


def read(pin):
    verify(pin)
    return json.loads(regular(pin["path"]).read_text())


def reference(plan):
    require(plan["sources"][REFERENCE] == REFERENCE_SHA256
            and descriptor(ROOT / REFERENCE)["sha256"] == REFERENCE_SHA256,
            "qualified independent scalar source")
    spec = importlib.util.spec_from_file_location("_readout_scalar_reference", ROOT / REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parameter_metadata(family):
    require(family in ARMS, "registered continuation arm")
    names = list(SHAPES)
    effective = names[-2:] if family == ARMS[0] else names
    count = sum(math.prod(SHAPES[name]) for name in effective)
    return {"names": names, "count": 6112, "requires_grad_names": effective,
            "requires_grad_count": count, "effective_names": effective, "effective_count": count}


def validate_checkpoint(np, arrays):
    require(set(arrays) == set(SHAPES), "exact eight slow checkpoint tensors")
    for name, shape in SHAPES.items():
        value = arrays[name]
        require(value.dtype == np.float32 and value.shape == shape and bool(np.isfinite(value).all()),
                "finite float32 checkpoint tensor: " + name)


def verify_prediction(np, saved, history):
    require(set(saved) == PREDICTION_FIELDS, "exact four prediction fields")
    total = len(history["targets"])
    for name, value in saved.items():
        dtype, shape = (np.int64, history["episode_offsets"].shape) if name == "episode_offsets" else (
            (np.bool_, (total,)) if name == "prior_mask" else (np.float32, (total, 4)))
        require(value.dtype == dtype and value.shape == shape and bool(np.isfinite(value).all()),
                "finite typed prediction: " + name)
    require(saved["episode_offsets"].tobytes() == history["episode_offsets"].tobytes()
            and saved["prior_mask"].tobytes() == history["prior_mask"].tobytes(),
            "complete prediction episode and later-query support")
    query, prior = history["query_mask"], history["prior_mask"]
    require(saved["action_prediction"][query].tobytes() == history["query_scores"][query].tobytes(),
            "query action bytes copy actual observations")
    inactive = saved["corrected_shadow_prior"][~prior]
    require(inactive.tobytes() == np.zeros_like(inactive).tobytes(), "positive-zero inactive prior")


def compute_comparability(fits):
    """Elapsed continuation time includes its harness; shared pretraining is excluded."""
    require(len(fits) == 6 and {(r["family"], r["seed"]) for r in fits}
            == {(family, seed) for seed in SEEDS for family in ARMS}, "all six timed continuation fits")
    by = {(r["family"], r["seed"]): r for r in fits}
    pairs = []
    for seed in SEEDS:
        residual = by["action_residual_only", seed]["seconds"]
        joint = by["full_joint", seed]["seconds"]
        for seconds in (residual, joint):
            finite_nonnegative(seconds, "finite positive continuation time")
            require(seconds > 0, "finite positive continuation time")
        ratio = residual / joint
        require(math.isfinite(ratio), "finite continuation-time ratio")
        pairs.append({"seed": seed, "residual_seconds": residual, "full_joint_seconds": joint,
                      "ratio": ratio, "passed": .9 <= ratio <= 1.1})
    return {"bounds": [.9, 1.1], "pairs": pairs, "passed": all(row["passed"] for row in pairs),
            "measure": "elapsed continuation fit seconds including harness; shared pretraining excluded"}


def comparisons(reports, fits):
    """Fixed full-joint candidate, all paired cells, no post-result selection."""
    require(len(reports) == 9 and {(r["family"], r["seed"]) for r in reports}
            == {(family, seed) for seed in SEEDS for family in VIEWS}, "all nine diagnostic reports")
    require(all(r["stage"] == "dev" and r["query_period"] == 4 and r["episodes"] == 36 for r in reports),
            "complete P4 fresh DEV reports")
    by = {(r["family"], r["seed"]): r for r in reports}

    def leaf(family, seed, scope, regime):
        value = by[family, seed]["scopes"][scope]["by_regime"][regime]
        require(value["episodes"] == 18 and value["declared_case_count"] == 6,
                "complete equal-case diagnostic denominators")
        finite_nonnegative(value["case_weighted_raw_gap"], "finite diagnostic gap")
        return value

    cells = []
    for seed in SEEDS:
        for regime in REGIMES:
            values = {scope: {family: leaf(family, seed, scope, regime)["case_weighted_raw_gap"]
                              for family in VIEWS} for scope in ("full", "later")}
            support = [leaf(family, seed, "later", regime)["supported_case_count"] for family in VIEWS]
            require(len(set(support)) == 1 and type(support[0]) is int and 0 <= support[0] <= 6,
                    "identical complete-path support across methods")
            later, full = values["later"], values["full"]
            controls = ("pretrained", *READOUTS)
            gain = {family: later["full_joint"] <= .95 * later[family]
                    and later["full_joint"] < later[family] for family in controls}
            nonregression = {family: full["full_joint"] <= full[family] for family in controls}
            proximity = {family: {
                "no_more_than_5pct_worse": later[family] <= 1.05 * later["full_joint"],
                "relative_excess": (later[family] / later["full_joint"] - 1) if later["full_joint"] else None,
                "zero_joint_gap": later["full_joint"] == 0,
            } for family in READOUTS}
            cells.append({"seed": seed, "regime": regime, "supported_later_cases": support[0],
                "gaps": values, "later_gain_at_least_5pct": gain, "full_gap_nonregression": nonregression,
                "readout_proximity_descriptive_only": proximity,
                "passed": all(gain.values()) and all(nonregression.values())})
    efficacy = all(row["passed"] for row in cells)
    comparable = compute_comparability(fits)
    return {"candidate": "full_joint", "margin": .05, "cells": cells,
            "passed_cells": sum(row["passed"] for row in cells), "total_cells": 6,
            "efficacy_passed": efficacy, "compute_comparability": comparable,
            "overall_passed": efficacy and comparable["passed"],
            "held_out_evidence": False, "held_out_from_training": True, "fresh_dev": True,
            "development_only": True, "statistical_equivalence": False,
            "test_admitted": False, "confirmation_admitted": False}


def batch_geometry(ref, lengths):
    """Batch-six no-memory scheduled operations; no numerical model calls."""
    work, units = {}, {}
    for start in range(0, max(lengths), 32):
        chunk = [max(0, min(32, length - start)) for length in lengths]
        part_work, part_units = ref.scheduled_work(chunk, start, "joint_aux")
        ref.add_counts(work, part_work)
        ref.add_counts(units, part_units)
    chunks = (max(lengths) + 31) // 32
    backward = chunks if max(lengths) > 1 else 0
    return {"forward_rows": sum(lengths), "nonquery_rows": sum(n - (n + 3) // 4 for n in lengths),
            "prior_rows": sum((n - 1) // 4 for n in lengths), "forward_chunks": chunks,
            "loss_chunks": backward, "backward_chunks": backward, "differentiable_chunks": backward,
            "no_gradient_chunks": chunks - backward, "skipped_backward_chunks": chunks - backward,
            "work_counts": work, "memory_work_units": units}


def verify_journal(ref, path, fits, lengths, check):
    updates = 0
    totals, aggregate_work = {}, {}
    verify(path)
    with regular(path["path"]).open() as stream:
        def row():
            line = stream.readline()
            require(line.endswith("\n"), "complete durable journal row")
            value = json.loads(line)
            require(type(value) is dict, "journal object")
            return value

        for fit in fits:
            names = fit["parameters"]["effective_names"]
            for epoch, order in enumerate(fit["orders"], 1):
                for batch in range(9):
                    check()
                    updates += 1
                    indices = order[6 * batch:6 * (batch + 1)]
                    expected = {"call_id": updates, "family": fit["family"], "seed": fit["seed"],
                                "epoch": epoch, "batch": batch, "episode_indices": indices}
                    ref.close_equal(row(), {"event": "attempt", **expected}, "journal attempt")
                    returned = row()
                    require(set(returned) == set(expected) | {"event", "result"}, "exact journal return fields")
                    ref.close_equal({k: v for k, v in returned.items() if k != "result"},
                                    {"event": "return", **expected}, "journal return identity")
                    value = returned["result"]
                    geometry = batch_geometry(ref, [int(lengths[i]) for i in indices])
                    fixed = {"version": "otto-readout-ablation-training-v1", "objective": "full_forecast_aux",
                             "episode_indices": indices, "episode_exposures": 6, "optimizer_updates": 1,
                             "optimizer_step": (epoch - 1) * 9 + batch + 1, "effective_parameter_names": names,
                             "effective_parameter_count": fit["parameters"]["effective_count"],
                             "frozen_optimizer_state_entries": 0, **geometry}
                    scalar_names = {"loss", "nonquery_loss", "prior_loss", "gradient_norm_before_clip"}
                    require(set(value) == set(fixed) | scalar_names | {"zero_filled_gradient_names", "timing_seconds"},
                            "exact training result fields")
                    for name, expected_value in fixed.items():
                        ref.close_equal(value[name], expected_value, "independent batch " + name)
                    missing = value["zero_filled_gradient_names"]
                    require(type(missing) is list and len(set(missing)) == len(missing)
                            and all(name in names for name in missing), "only effective zero-filled gradients")
                    for name in scalar_names:
                        finite_nonnegative(value[name], "finite preserved training scalar")
                    require(set(value["timing_seconds"]) == TIMINGS, "complete measured timing fields")
                    for seconds in value["timing_seconds"].values():
                        finite_nonnegative(seconds, "finite preserved timing")
                    ref.add_counts(aggregate_work, geometry["work_counts"])
                    ref.add_counts(totals, {k: v for k, v in geometry.items() if type(v) is int})
        require(stream.read() == "", "no extra journal work")
    require(updates == 3078, "all six calibrated continuation update counts")
    return {"updates": updates, "episode_exposures": updates * 6,
            "scheduled_training": totals, "scheduled_training_work": aggregate_work}


def audit(np, plan, receipt, directory, check):
    """Caller must have authenticated original source/process closure first."""
    directory = Path(directory)
    require(callable(check), "bounded audit guard")
    check()
    require(receipt["version"] == PRODUCER_VERSION and receipt["phase"] == "train"
            and receipt["status"] == "completed" and receipt["pending"] is None and receipt["error"] is None,
            "completed original training producer")
    expected_counts = {"fits_completed": 6, "optimizer_steps": 3078, "episode_exposures": 18468,
                       "views_completed": 9, "array_decodes": 5, "checkpoint_decodes": 3,
                       "teacher_calls": 0, "native_calls": 0, "test_array_decodes": 0}
    for key, value in expected_counts.items():
        require(type(receipt[key]) is int and receipt[key] == value, "exact producer count: " + key)
    require(set(receipt["files"]) == PAYLOADS
            and {p.name for p in directory.iterdir()} == PAYLOADS | {"receipt.json"}, "complete exact producer payload set")
    require(receipt["sources"] == plan["sources"], "same registered source closure")
    for name, pin in plan["sources"].items():
        check()
        require(descriptor(ROOT / name)["sha256"] == pin, "unchanged registered source: " + name)
    require(SELF in plan["sources"], "independent auditor included in registered closure")
    files = receipt["files"]
    for name, pin in files.items():
        require(Path(pin["path"]) == directory / name, "same contained declared payload")
        verify(pin)
    fits, views = read(files["fits.json"]), read(files["views.json"])
    require(type(fits) is list and [(r["family"], r["seed"]) for r in fits]
            == [(arm, seed) for seed in SEEDS for arm in ARMS], "all six ordered final fits")
    require(type(views) is list and [(r["family"], r["seed"]) for r in views]
            == [(family, seed) for seed in SEEDS for family in VIEWS], "all nine ordered canonical views")
    for row in fits:
        require(row["checkpoint"] == files[f"checkpoint-{row['family']}-{row['seed']}.npz"], "fit checkpoint payload join")
    for row in views:
        require(row["prediction"] == files[f"prediction-{row['family']}-{row['seed']}.npz"], "view prediction payload join")
        finite_nonnegative(row["seconds"], "finite measured inference time")
    barrier = read(files["training-barrier.json"])
    wanted_barrier = {"fits_completed": 6, "optimizer_steps": 3078, "episode_exposures": 18468,
                      "fits": files["fits.json"], "checkpoints": [r["checkpoint"] for r in fits], "dev_decodes": 0}
    require(set(barrier) == set(wanted_barrier) | {"created_ns"}
            and type(barrier["created_ns"]) is int and barrier["created_ns"] > 0,
            "complete pre-DEV barrier fields")
    require(all(barrier[k] == value for k, value in wanted_barrier.items()), "all six checkpoints close before DEV")
    require(type(receipt["started_ns"]) is int and type(receipt["finished_ns"]) is int
            and receipt["started_ns"] <= barrier["created_ns"] <= receipt["finished_ns"],
            "training barrier within original producer interval")
    with regular(files["progress.jsonl"]["path"]).open() as stream:
        expected_progress = [{"event": "fit_complete", **row} for row in fits]
        expected_progress += [{"event": "view_complete", **row} for row in views]
        for row in expected_progress:
            line = stream.readline()
            require(line.endswith("\n") and json.loads(line) == row, "complete ordered durable fit and view progress")
        require(stream.read() == "", "no extra progress events")
    require(read(files["runtime.json"]) == plan["runtime"], "same recorded qualified runtime")
    require(read(files["summary.json"]) == {"status": "completed_pending_independent_audit", "fits": 6, "views": 9,
            "scope": "fresh development paths; no autonomous or confirmation claim"}, "unadmitted complete producer summary")
    for stage, count in (("train", 54), ("dev", 36)):
        record = plan[stage]
        identities = record["identities"]
        require(len(identities) == count and all(row["stage"] == stage for row in identities)
                and len({row["episode_id"] for row in identities}) == count, "complete admitted identity roster")
        verify(record["descriptor"])
    for seed in SEEDS:
        verify(plan["lineage"]["checkpoints"]["pretrained"][str(seed)])
    ref = reference(plan)
    counts = {"array_decodes": 0, "checkpoint_decodes": 0, "views_completed": 0, "fits_checked": 0,
              "model_calls": 0, "optimizer_calls": 0, "teacher_calls": 0, "native_calls": 0,
              "test_array_decodes": 0}

    def arrays(pin, *, checkpoint=False):
        check()
        verify(pin)
        name = Path(pin["path"]).name
        require(name != "test.npz" and not name.startswith(("test-", "confirm")), "no TEST or confirmation decode")
        with np.load(pin["path"], allow_pickle=False) as archive:
            require(len(archive.files) == len(set(archive.files)), "unique NPZ fields")
            result = {name: archive[name] for name in archive.files}
        counts["array_decodes"] += 1
        counts["checkpoint_decodes"] += int(checkpoint)
        check()
        return result

    histories = {}
    for stage, count in (("train", 54), ("dev", 36)):
        ids = plan[stage]["identities"]
        expected = ref.expected_history(np, arrays(plan[stage]["descriptor"]), ids)
        saved = arrays(files[stage + "-history.npz"])
        require(set(saved) == set(expected), "exact projected history arrays")
        for name, value in expected.items():
            require(saved[name].dtype == value.dtype and saved[name].shape == value.shape
                    and saved[name].tobytes() == value.tobytes(), "independent history bytes: " + name)
        metadata = {"version": "otto-query-memory-data-v1", "query_period": 4, "stage": stage,
                    "episode_ids": [row["episode_id"] for row in ids],
                    "identities": [[[name, row[name]] for name in sorted(ID_FIELDS)] for row in ids],
                    "episode_count": count}
        ref.close_equal(read(files[stage + "-history.json"]), metadata, "projected history metadata")
        histories[stage] = saved

    fit_by = {(row["family"], row["seed"]): row for row in fits}
    for seed in SEEDS:
        parent = arrays(plan["lineage"]["checkpoints"]["pretrained"][str(seed)], checkpoint=True)
        validate_checkpoint(np, parent)
        initial = ref.tensor_witness(parent)
        rng = np.random.Generator(np.random.PCG64(seed))
        orders = [rng.permutation(54).astype(np.int64) for _ in range(max(EPOCHS.values()))]
        for family in ARMS:
            selected_orders = orders[:EPOCHS[family]]
            permutation = hashlib.sha256(b"".join(order.tobytes() for order in selected_orders)).hexdigest()
            row = fit_by[family, seed]
            final = arrays(row["checkpoint"], checkpoint=True)
            validate_checkpoint(np, final)
            ref.close_equal(row["initial"], initial, "exact same-seed parent tensor witness")
            ref.close_equal(row["final"], ref.tensor_witness(final), "saved final tensor witness")
            parameters = parameter_metadata(family)
            ref.close_equal(row["parameters"], parameters, "actual optimizer and gradient masks")
            frozen = [name for name in SHAPES if name not in parameters["effective_names"]]
            require(row["frozen_names"] == frozen and row["frozen_unchanged"] is True
                    and all(final[name].tobytes() == parent[name].tobytes() for name in frozen),
                    "all frozen checkpoint bytes equal corresponding parent")
            fixed = {"steps": 9 * EPOCHS[family], "epochs": EPOCHS[family], "exposures": 54 * EPOCHS[family],
                     "optimizer_steps": dict.fromkeys(parameters["effective_names"], 9 * EPOCHS[family]),
                     "orders": [order.tolist() for order in selected_orders], "permutation_sha256": permutation}
            for name, value in fixed.items():
                ref.close_equal(row[name], value, "fixed matched continuation " + name)
            finite_nonnegative(row["seconds"], "finite measured fit time")
            counts["fits_checked"] += 1

    journal = verify_journal(ref, files["work.jsonl"], fits, np.diff(histories["train"]["episode_offsets"]), check)
    counts.update(journal_updates_checked=journal["updates"], episode_exposures_checked=journal["episode_exposures"])
    history = histories["dev"]
    expected_work = {}
    for length in np.diff(history["episode_offsets"]):
        ref.add_counts(expected_work, batch_geometry(ref, [int(length)])["work_counts"])
    reports = []
    for view in views:
        check()
        prediction = arrays(view["prediction"])
        verify_prediction(np, prediction, history)
        ref.close_equal(view["work"], expected_work, "physical batch-one canonical view work")
        report = ref.scalar_report(np, plan["dev"]["identities"], history["targets"], history["legal"],
            prediction["action_prediction"], prediction["corrected_shadow_prior"], history["episode_offsets"],
            view["family"], view["seed"], check=check)
        ref.close_equal(view["metrics"], report, "independent complete DEV scalar report")
        reports.append(report)
        counts["views_completed"] += 1
    require(counts["array_decodes"] == 22 and counts["checkpoint_decodes"] == 9, "complete saved-array audit accounting")
    check()
    return {"version": VERSION, "status": "completed_pending_original_audit_closure", "agreement": True,
            "technical_complete": False, "requires_original_supervisor_closure": True,
            "counts": counts, "reports": reports, "comparisons": comparisons(reports, fits), "journal": journal,
            "measured_fit_seconds": [{"family": r["family"], "seed": r["seed"], "seconds": r["seconds"]} for r in fits],
            "measured_view_seconds": [{"family": r["family"], "seed": r["seed"], "seconds": r["seconds"]} for r in views],
            "test_admitted": False, "confirmation_admitted": False, "limitations": LIMITATIONS}
