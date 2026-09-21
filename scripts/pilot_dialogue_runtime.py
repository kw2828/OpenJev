"""Separate fixed-sample runtime pilot, with estimator forecasts frozen in order.

Every model call replays complete DEV public streams under the unchanged final
checkpoint inference wrapper. TRAIN calibration contributes only authenticated
work totals, never actor inputs or targets. No task score or temperature is fit.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path

import dialogue_runtime_common as common
import numpy as np
import replay_dialogue_calibration as replay
from dialogue_runtime_geometry import report_geometry

from openjev.research import dialogue_runtime_projection as projection

require = common.require
FIT_ORDER = list(projection.FIT_ORDER)
DEV_ENDPOINTS = replay.DEV_ENDPOINTS
PUBLIC_ENDPOINT_KEYS = ("row_index", "source_row_index", "time", "turn_index", "query_position",
                        "query_index", "query_id", "service", "slot", "candidate_ids", "candidate_values")


def public_inputs(ctx, budget):
    """Select on complete public work metadata before reading any prediction."""
    workloads = common.read(ctx["original_prepared"] / "workloads.json")
    profiles = [p for p in workloads["profiles"] if p["split"] == "dev"]
    legacy = common.read(ctx["prepared"] / "replay-cases.json")["cases"]
    sample = projection.select_timing_sample(profiles, [c["dialogue_id"] for c in legacy])
    require(sample["fit_order"] == FIT_ORDER == ctx["science"]["fit_order"], "Unchanged twelve checkpoint order")
    order = sample["warmup_ids"] + sample["estimator"]["dialogue_ids"] + sample["verification"]["dialogue_ids"]
    require(len(order) == len(set(order)) == 130, "Two legacy warmups and 128 distinct timing dialogues")
    selected = set(order)
    actors = {}
    for actor in replay.read_lines(ctx["original_prepared"] / "actors-dev.jsonl", budget):
        if actor["dialogue_id"] in selected:
            require(actor["dialogue_id"] not in actors and actor["split"] == "dev", "Distinct original selected DEV actors")
            actors[actor["dialogue_id"]] = actor
    require(set(actors) == selected, "All complete sampled public actors")
    indexed = {p["dialogue_id"]: p["work"] for p in profiles}
    endpoints = {did: [] for did in order}
    count = 0
    with (Path(ctx["prior"].run) / "evaluation-rows.jsonl").open() as stream:
        for line in stream:
            budget.check()
            row = json.loads(line)
            require(type(row["row_index"]) is int and row["row_index"] == count and row["split"] == "dev",
                    "Canonical original DEV endpoint order")
            count += 1
            did = row["dialogue_id"]
            if did in selected:
                # Whole JSON decoding is unavoidable, but targets, transitions,
                # and correctness are neither used nor retained in this pilot.
                endpoints[did].append({key: row[key] for key in PUBLIC_ENDPOINT_KEYS})
    require(count == DEV_ENDPOINTS and all(endpoints.values()), "Complete original endpoint coverage for every selected dialogue")
    for case in legacy:
        did = case["dialogue_id"]
        require(case["work"] == indexed[did] and case["endpoints"] == endpoints[did]
                and case["row_indices"] == [r["row_index"] for r in endpoints[did]], "Exactly the two legacy replay cases")
    calibration = common.read(ctx["prepared"] / "workloads.json")
    records = calibration["profiles"]
    require(len(records) == len({p["dialogue_id"] for p in records}) == 512
            and all(p["split"] == "train" for p in records)
            and ctx["preparation_receipt"]["counts"]["dialogues"] == 512, "Complete existing calibration work metadata")
    calibration_work = {"dialogue_count": len(records),
                        **{key: sum(p["work"][key] for p in records) for key in projection.WORK_MEASURES[1:]}}
    require(all(calibration_work[key] == calibration["all"]["totals"][key]
                for key in projection.WORK_MEASURES[1:]), "Calibration work sums without neural inputs")
    sample["geometry"] = report_geometry(profiles, records, sample)
    return sample, actors, indexed, endpoints, calibration_work


def raw_arrays(directory, count):
    logs = np.lib.format.open_memmap(directory / "partial-log-probs.npy", mode="w+", dtype=np.float32, shape=(count, 12))
    ids = np.lib.format.open_memmap(directory / "partial-row-indices.npy", mode="w+", dtype=np.int64, shape=(count,))
    logs[:] = -np.inf
    ids[:] = -1
    return logs, ids


def flush_preserving_error(*arrays):
    primary = sys.exception()
    for array in arrays:
        try:
            array.flush()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("Partial pilot array flush failed: " + repr(error))


def block(api, route, name, dialogue_ids, actors, profiles, endpoints, lexical, arm,
          baseline, logs, row_ids, cursor, journal, budget, progress):
    """Disjoint whole-block timing includes journal and raw-array writes."""
    budget.sync(api.torch)
    start = budget.elapsed()
    work, maxima, endpoint_count = Counter(), Counter(), 0
    for position, did in enumerate(dialogue_ids):
        progress.update(operation="runtime-public-forward", block=name, dialogue_id=did,
                        block_position=position, saved_endpoint_rows=cursor)
        if name == "verification":
            progress["verification_forwards_started"] = position + 1
        case_start = budget.elapsed()
        result = route.forward_public(replay.payload(actors[did], lexical, arm, "qualify"))
        replay.validate_work(result, profiles[did])
        values, indices = replay.gather_public(result, endpoints[did])
        require(np.all(indices < DEV_ENDPOINTS) and cursor + len(indices) <= len(row_ids), "Complete selected endpoint bounds")
        logs[cursor:cursor + len(indices)] = values
        row_ids[cursor:cursor + len(indices)] = indices
        logs.flush()
        row_ids.flush()
        # Obtainable replay outputs survive a subsequent parity failure.
        parity = replay.parity(values, baseline[indices], endpoints[did])
        cursor += len(indices)
        endpoint_count += len(indices)
        work.update(result.encoder_work)
        for key in ("maximum_supported_log_error", "maximum_probability_error", "maximum_raw_mass_error"):
            maxima[key] = max(maxima[key], parity[key])
        budget.sync(api.torch)
        replay.journal_write(journal, {"block": name, "block_position": position, "dialogue_id": did,
                                      "row_indices": indices.tolist(), "payload_seconds": budget.elapsed() - case_start,
                                      "encoder_work": result.encoder_work, "invariants": result.invariants,
                                      "parity": parity})
        progress["completed_forwards"] += 1
        progress["saved_endpoint_rows"] = cursor
        budget.storage()
    budget.sync(api.torch)
    end = budget.elapsed()
    geometry = {"dialogue_count": len(dialogue_ids),
                **{key: sum(profiles[did][key] for did in dialogue_ids) for key in projection.WORK_MEASURES[1:]}}
    return {"block": name, "dialogue_ids": list(dialogue_ids), "work": geometry,
            "started_elapsed_seconds": start, "finished_elapsed_seconds": end, "seconds": end - start,
            "endpoint_rows": endpoint_count, "encoder_work": dict(work), "parity_maxima": dict(maxima)}, cursor


def pilot_fit(api, route, directory, fit_id, sample, actors, profiles, endpoints, lexical,
              baseline, sample_sha256, budget, progress):
    order = sample["warmup_ids"] + sample["estimator"]["dialogue_ids"] + sample["verification"]["dialogue_ids"]
    expected_rows = [r["row_index"] for did in order for r in endpoints[did]]
    logs, row_ids = raw_arrays(directory, len(expected_rows))
    cursor, observed_work = 0, Counter()
    arm = fit_id.rsplit("-", 1)[0]
    try:
        with (directory / "dialogues.jsonl").open("x") as journal:
            warmup, cursor = block(api, route, "warmup", sample["warmup_ids"], actors, profiles, endpoints,
                                   lexical, arm, baseline, logs, row_ids, cursor, journal, budget, progress)
            observed_work.update(warmup["encoder_work"])
            estimator, cursor = block(api, route, "estimator", sample["estimator"]["dialogue_ids"], actors,
                                      profiles, endpoints, lexical, arm, baseline, logs, row_ids, cursor, journal, budget, progress)
            observed_work.update(estimator["encoder_work"])
            require(estimator["work"] == sample["estimator"]["work"], "Whole estimator geometry, no omitted timing outliers")
            prediction = projection.predict_verification(fit_id, estimator["seconds"], estimator["work"],
                                                         sample["verification"]["work"])
            forecast = {"fit_id": fit_id, "sample_sha256": sample_sha256, "prediction": prediction,
                        "created_elapsed_seconds": budget.elapsed(), "verification_forwards_started": 0}
            common.write(directory / "forecast.json", forecast)
            forecast_sha = common.sha(directory / "forecast.json")
            budget.storage()
            published_at = budget.elapsed()
            progress.update(operation="verification-forecast-published", forecast_sha256=forecast_sha,
                            verification_forwards_started=0)
            verification, cursor = block(api, route, "verification", sample["verification"]["dialogue_ids"], actors,
                                         profiles, endpoints, lexical, arm, baseline, logs, row_ids, cursor, journal, budget, progress)
            observed_work.update(verification["encoder_work"])
            require(verification["work"] == sample["verification"]["work"]
                    and estimator["finished_elapsed_seconds"] <= forecast["created_elapsed_seconds"]
                    <= published_at <= verification["started_elapsed_seconds"]
                    and common.sha(directory / "forecast.json") == forecast_sha, "Unchanged forecast published before verification")
            validation = projection.validate_verification(prediction, verification["seconds"])
        require(cursor == len(expected_rows) and row_ids.tolist() == expected_rows, "All 130 public dialogues and exact endpoint order")
        start = budget.elapsed()
        replay.save_packet(directory / "predictions.npz", logs, row_ids)
        budget.storage()
        finalization_seconds = budget.elapsed() - start
    finally:
        flush_preserving_error(logs, row_ids)
        del logs, row_ids
    witness = replay.finish_fit(route, len(order), observed_work)
    (directory / "partial-log-probs.npy").unlink()
    (directory / "partial-row-indices.npy").unlink()
    return {"warmup": warmup, "estimator": estimator, "verification": verification,
            "forecast_sha256": forecast_sha, "forecast_published_elapsed_seconds": published_at,
            "verification_result": validation, "witness": witness, "endpoint_rows": cursor,
            "prediction_finalization_seconds": finalization_seconds}


def finalize(args, ctx, budget, metadata):
    """Called by the common lifecycle after final upstream authentication.

    Projection publication and terminal publication necessarily follow this
    measurement; the actual supervisor terminal must expose that residual cost.
    No estimator or verification record is changed in this final bookkeeping.
    """
    snapshot_seconds = budget.elapsed()
    records = {fit["fit_id"]: fit["verification_result"] for fit in metadata["fits"]}
    loads = {fit["fit_id"]: fit["loading_seconds"] for fit in metadata["fits"]}
    warmups = {fit["fit_id"]: fit["warmup"]["seconds"] for fit in metadata["fits"]}
    disjoint = math.fsum(fit["loading_seconds"] + math.fsum(fit[b]["seconds"] for b in ("warmup", "estimator", "verification"))
                         for fit in metadata["fits"])
    overhead = snapshot_seconds - disjoint
    require(math.isfinite(overhead) and overhead >= 0, "All paid non-block overhead without double counting")
    result = projection.project_calibration(records, metadata["calibration_work"], loading_seconds_by_fit=loads,
                                           warmup_seconds_by_fit=warmups,
                                           preparation_parent_wall_seconds=ctx["preparation_terminal"]["wall_seconds"],
                                           pilot_nonblock_overhead_seconds=overhead)
    common.write(args.out / "projection.json", result)
    metadata.update(projection=result, projection_elapsed_seconds=snapshot_seconds,
                    disjoint_measured_seconds=disjoint, pilot_nonblock_overhead_seconds=overhead,
                    projection_publication_scope="All cost through projection computation, including final input reauthentication. "
                    "Projection/terminal publication tail is separately exposed by the actual parent terminal; no unseen residual is claimed measured.")
    return metadata


def body(args, ctx, budget):
    require(ctx["old_qualification"]["projection"]["admitted"] is False
            and ctx["old_qualification"]["parity_passed"] is True, "Preserve the prior completed parity and failed V1 admission")
    sample, actors, profiles, endpoints, calibration_work = public_inputs(ctx, budget)
    common.write(args.out / "sample.json", sample)
    sample_sha = common.sha(args.out / "sample.json")
    lexical = replay.load_lexical(ctx["original_prepared"])
    api = replay.backend()
    progress = ctx["progress"]
    progress.update(completed_fits=[], completed_forwards=0)
    fits = []
    for fit_id in FIT_ORDER:
        directory = args.out / fit_id
        directory.mkdir(exist_ok=False)
        progress.update(block=None, block_position=None, saved_endpoint_rows=0,
                        forecast_sha256=None, verification_forwards_started=0)
        route = None
        try:
            fit_started = budget.elapsed()
            route, loading_seconds = replay.load_route(api, ctx, fit_id, budget)
            with np.load(Path(ctx["prior"].run) / fit_id / "predictions.npz", allow_pickle=False) as saved:
                require(set(saved.files) == {"row_indices", "log_probs"}
                        and saved["row_indices"].dtype == np.int64
                        and np.array_equal(saved["row_indices"], np.arange(DEV_ENDPOINTS)), "Complete original saved row identity")
                baseline = saved["log_probs"]
                require(baseline.dtype == np.float32 and baseline.shape == (DEV_ENDPOINTS, 12), "Original raw DEV output geometry")
            result = pilot_fit(api, route, directory, fit_id, sample, actors, profiles, endpoints, lexical,
                               baseline, sample_sha, budget, progress)
            del baseline
            arm, seed = fit_id.rsplit("-", 1)
            receipt = {"status": "completed", "fit_id": fit_id, "arm": arm, "seed": int(seed),
                       "loading_seconds": loading_seconds, "wall_seconds": budget.elapsed() - fit_started,
                       **result, "files": common.members(directory)}
            common.write(directory / "completed.json", receipt)
            fits.append({**receipt, "completed_sha256": common.sha(directory / "completed.json")})
            progress["completed_fits"].append(fit_id)
        finally:
            primary = sys.exception()
            try:
                if route is not None:
                    progress["route"] = route.snapshot()
            except BaseException as snapshot_error:
                if primary is None:
                    raise
                primary.add_note("Pilot route failure witness unavailable: " + repr(snapshot_error))
            del route
            gc.collect()
        api.torch.mps.empty_cache()
        budget.sync(api.torch)
        budget.storage()
        print(json.dumps({"phase": "runtime-pilot", "completed_fits": len(fits), "fit_id": fit_id}), flush=True)
    require([fit["fit_id"] for fit in fits] == FIT_ORDER and progress["completed_forwards"] == 12 * 130,
            "All twelve fits and every paid warmup/estimator/verification forward completed")
    return {"fit_order": FIT_ORDER, "fits": fits, "sample_sha256": sample_sha,
            "calibration_work": calibration_work, "completed_forwards": progress["completed_forwards"],
            "timed_forwards": 12 * 128, "warmup_forwards": 24, "parity_passed": True,
            "legacy_v1_admission_revised": False, "scientific_conditions_changed": False,
            "calibration_forward_calls": 0, "model_weight_updates": 0, "temperature_fits": 0,
            "optimizer_created": False, "temperature_applied": False,
            "official_test_opened": False, "task_metrics_computed": False,
            "scope": "New fixed runtime estimator qualification only. Complete DEV streams and endpoint numerical parity, "
                     "no task efficacy scores or calibration TRAIN model calls; internal old trajectories were not saved for comparison."}


body.finalize = finalize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "supervision", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.set_defaults(command="pilot")
    return parser.parse_args(argv)


if __name__ == "__main__":
    common.execute(parse_args(), body)
