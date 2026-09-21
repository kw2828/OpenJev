"""Final-checkpoint replay qualification and complete TRAIN calibration inference.

The common lifecycle authenticates inputs before this body is entered and owns
the native parent deadline. No training, temperature fitting, or task scoring is
performed. Qualification gathers only public DEV addresses, never target labels.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import dialogue_calibration_common as common
import numpy as np

require = common.require
FIT_ORDER = [f"{arm}-{seed}" for seed in common.CONFIG["seeds"] for arm in common.CONFIG["arms"]]
SAMPLE_SIZE, DEV_ENDPOINTS = 512, 62329
WORK_MEASURES = ("encoder_calls", "padded_attention_positions", "real_question_updates")
ENCODER_KEYS = ("input_texts", "content_tokens", "encoder_sequences", "encoder_calls",
                "special_token_positions", "valid_token_positions", "padded_token_positions",
                "padded_attention_positions", "padding_token_positions", "overlength_texts_chunked",
                "truncated_tokens")


def read_lines(path, budget):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            budget.check()
            require(bool(line.strip()), "Nonempty JSONL records")
            result.append(json.loads(line))
    return result


def backend():
    """Production has no factory/device override flags or optimizer imports."""
    import torch

    from openjev.research.dialogue_calibration_inference import (
        collect_endpoints,
        load_final_checkpoint,
    )

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(torch.backends.mps.is_available(), "Qualified float32 MPS encoder is required")
    return SimpleNamespace(torch=torch, load=load_final_checkpoint, collect=collect_endpoints)


def numpy_cpu(value):
    return value.detach().cpu().numpy()


def validate_logs(logs, rows):
    require(logs.dtype == np.float32 and logs.shape == (len(rows), 12), "Raw float32 endpoint geometry")
    mask = np.arange(12)[None, :] < np.array([len(row["candidate_ids"]) for row in rows])[:, None]
    require(len(rows) > 0 and np.isfinite(logs[mask]).all() and np.isneginf(logs[~mask]).all(),
            "Finite supported logs and exact negative-infinity padding")
    error = float(np.max(np.abs(np.exp(logs.astype(np.float64)).sum(1) - 1)))
    require(error <= common.CONFIG["mass_tolerance"], "Raw endpoint mass tolerance")
    return mask, error


def gather_public(result, endpoints):
    """DEV replay addresses have no labels; do not invent evaluator targets."""
    require(result.split == "dev" and result.analysis_role == "qualification" and endpoints,
            "Separate DEV numerical replay role")
    full = numpy_cpu(result.log_probs)
    output = np.full((len(endpoints), 12), -np.inf, dtype=np.float32)
    seen, rows = set(), []
    for i, row in enumerate(endpoints):
        require(not ({"label_id", "label_index", "bin", "stratum", "target"} & set(row)),
                "Qualification references contain public addresses only")
        require(all(type(row[k]) is int and row[k] >= 0 for k in
                    ("row_index", "source_row_index", "time", "turn_index", "query_position", "query_index")),
                "Integral public replay addresses")
        t, q = row["time"], row["query_position"]
        require(t < full.shape[1] and q < len(result.query_ids)
                and result.query_ids[q] == row["query_index"]
                and result.user_turn_indices[t] == row["turn_index"], "Exact public time/query join")
        ids = result.candidate_ids[q]
        require(3 <= len(ids) <= 12 and list(ids) == row["candidate_ids"], "Exact replay candidate order")
        require((t, q) not in seen and row["row_index"] not in rows, "Distinct complete replay endpoints")
        seen.add((t, q))
        rows.append(row["row_index"])
        output[i, :len(ids)] = full[0, t, q, :len(ids)]
    validate_logs(output, endpoints)
    return output, np.asarray(rows, dtype=np.int64)


def parity(replayed, baseline, endpoints):
    mask, raw_error = validate_logs(replayed, endpoints)
    _, old_error = validate_logs(baseline, endpoints)
    a, b = replayed.astype(np.float64), baseline.astype(np.float64)
    log_error = float(np.max(np.abs(a[mask] - b[mask])))
    probability_error = float(np.max(np.abs(np.exp(a) - np.exp(b))))
    choices_same = bool(np.array_equal(np.argmax(a, axis=1), np.argmax(b, axis=1)))
    require(log_error <= common.CONFIG["supported_log_tolerance"]
            and probability_error <= common.CONFIG["probability_tolerance"] and choices_same,
            "Final-checkpoint replay parity failed")
    return {"rows": len(endpoints), "maximum_supported_log_error": log_error,
            "maximum_probability_error": probability_error, "canonical_first_argmax_equal": choices_same,
            "maximum_raw_mass_error": raw_error, "baseline_maximum_raw_mass_error": old_error}


def projection(cases, loading_seconds, totals, preparation_seconds):
    require(len(cases) == 2 * len(FIT_ORDER) and len(loading_seconds) == len(FIT_ORDER),
            "All 24 replay cases and 12 loads enter projection")
    require(all(math.isfinite(v) and v > 0 for v in [preparation_seconds, *loading_seconds]),
            "Positive finite complete preparation and loading costs")
    estimates = {}
    for key in WORK_MEASURES:
        require(type(totals[key]) is int and totals[key] > 0, "Positive complete calibration work")
        rates = []
        for case in cases:
            seconds, work = case["case_seconds"], case["work"][key]
            require(math.isfinite(seconds) and seconds > 0 and type(work) is int and work > 0,
                    "Positive synchronized replay work and duration")
            rates.append(seconds / work)
        total = len(FIT_ORDER) * totals[key]
        estimates[key] = {"maximum_seconds_per_work_unit": max(rates), "all_fit_work": total,
                          "projected_seconds": max(rates) * total}
    variable = max(item["projected_seconds"] for item in estimates.values())
    loads = math.fsum(loading_seconds)
    total = variable + loads + preparation_seconds
    require(math.isfinite(total), "Finite full projection")
    return {"measures": estimates, "variable_seconds": variable, "loading_seconds": loads,
            "preparation_seconds": preparation_seconds, "total_seconds": total,
            "threshold_seconds": common.CONFIG["projection_admission_seconds"],
            "admitted": total <= common.CONFIG["projection_admission_seconds"],
            "scope": "Heuristic screening of 12 x 512 forwards, not a runtime guarantee; fixed hard cap remains."}


def validate_work(result, work):
    require(result.encoder_work == {k: work[k] for k in ENCODER_KEYS}, "Every encoder token/chunk counted")
    inv = result.invariants
    expected = {"forward_calls": 1, "forward_returned": 1,
                **dict.fromkeys(("advance_calls", "advance_returned", "valid_turns"), work["public_user_turns"]),
                **dict.fromkeys(("executed_valid_question_slots", "real_question_updates", "incoming_checks",
                                 "feature_checks", "result_checks", "mass_checks"), work["real_question_updates"])}
    require(all(inv[k] == v for k, v in expected.items()), "Every full public recurrent update monitored")
    require(inv["tolerance"] == 2e-6 and 0 <= inv["mass_above_one_count"] <= work["real_question_updates"]
            and inv["mass_min"] is not None and 0 <= inv["mass_min"] <= inv["mass_max"] <= 1 + 2e-6,
            "Departure mass witnesses")
    for key in ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot"):
        require(math.isfinite(inv[key]) and 0 <= inv[key] <= 2e-6, "Finite monitored normalization tolerance")


def payload(actor, lexical, arm, phase):
    offset, shape = actor["lexical_offset"], actor["lexical_shape"]
    require(type(offset) is int and offset >= 0 and len(shape) == 4
            and all(type(v) is int and v > 0 for v in shape), "Public lexical slice geometry")
    flat = lexical["numbers" if arm.endswith("numbers") else "original"]
    size = math.prod(shape)
    require(offset + size <= flat.size, "Lexical slice lies within authenticated payload")
    role, split = ("qualification", "dev") if phase == "qualify" else ("calibration", "train")
    require(actor["split"] == split, "Original actor split")
    return {**actor, "source_split": split, "analysis_role": role,
            "lexical": np.asarray(flat[offset:offset + size]).reshape(shape).copy()}


def load_lexical(directory):
    arrays = {name: np.load(directory / f"lexical-{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("original", "numbers")}
    require(all(a.dtype == np.float32 and a.ndim == 1 for a in arrays.values()), "Flat raw lexical arrays")
    return arrays


def load_route(api, ctx, fit_id, budget):
    ctx["progress"].update(operation="load-final-checkpoint", fit_id=fit_id, dialogue_id=None,
                           case_index=None, endpoint_rows=0, load={}, route=None)
    budget.sync(api.torch)
    start = budget.elapsed()
    directory = Path(ctx["prior"].run) / fit_id
    receipt = common.read(directory / "completed.json")
    arm, seed = fit_id.rsplit("-", 1)
    require(receipt["status"] == "completed" and receipt["fit_id"] == fit_id
            and receipt["arm"] == arm and receipt["seed"] == int(seed), "Exact final checkpoint fit identity")
    pin = receipt["checkpoint"]["sha256"]
    require(receipt["files"]["weights.pt"]["sha256"] == pin, "Final checkpoint manifest binding")
    parent = ctx["parent"]
    route = api.load(directory / "weights.pt", arm=arm, checkpoint_sha256=pin,
                     snapshot=parent["snapshot"], model_files=parent["model_files"],
                     tokenizer_ids=parent["tokenizer_ids"], expected_encoder_sha256=receipt["final_encoder_sha256"],
                     check=budget.check, progress=ctx["progress"]["load"])
    witness = route.snapshot()
    ctx["progress"]["route"] = witness
    require(witness["synthetic_injection"] is False and witness["encoder_device"].split(":")[0] == "mps"
            and witness["memory_device"] == "cpu" and witness["dtype"] == "float32"
            and witness["optimizer_created"] is False and witness["temperature_applied"] is False,
            "Production inference uses only the original device/mode path")
    budget.sync(api.torch)
    return route, budget.elapsed() - start


def save_packet(path, logs, indices):
    with path.open("xb") as stream:
        np.savez(stream, log_probs=logs, row_indices=indices)


def journal_write(stream, record):
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def finish_fit(route, expected_dialogues, expected_work):
    final = route.verify_parameters_unchanged()
    witness = route.snapshot()
    require(witness["encoder_work"] == dict(expected_work), "Complete fit encoder work sum")
    for prefix in ("checkpoint_load", "encoder_load"):
        require(witness["counts"][prefix + "_attempts"] == witness["counts"][prefix + "_returns"] == 1,
                "Exactly one final checkpoint and encoder load")
    for prefix in ("public_forward", "encoding", "memory_forward"):
        require(witness["counts"][prefix + "_attempts"] == witness["counts"][prefix + "_returns"] == expected_dialogues,
                "Every required complete public forward returned")
    require(witness["counts"]["encoder_forward_attempts"] == witness["counts"]["encoder_forward_returns"]
            == expected_work["encoder_calls"], "Actual encoder attempt/return coverage")
    return {**witness, "final_sha256": final}


def qualify_fit(api, route, directory, cases, actors, lexical, arm, baseline, budget, progress):
    records, work = [], Counter()
    with (directory / "cases.jsonl").open("x") as journal:
        for index, case in enumerate(cases):
            progress.update(operation="qualification-forward", dialogue_id=case["dialogue_id"], case_index=index)
            budget.sync(api.torch)
            start = budget.elapsed()
            result = route.forward_public(payload(actors[case["dialogue_id"]], lexical, arm, "qualify"))
            validate_work(result, case["work"])
            logs, indices = gather_public(result, case["endpoints"])
            require(indices.tolist() == case["row_indices"], "Complete predeclared DEV endpoint order")
            require(np.all(indices < DEV_ENDPOINTS), "Saved DEV endpoint address bounds")
            # Retain obtainable raw replay before rejecting parity, without task scoring.
            save_packet(directory / f"case-{index:02}.npz", logs, indices)
            equality = parity(logs, baseline[indices], case["endpoints"])
            budget.sync(api.torch)
            payload_seconds = budget.elapsed() - start
            record = {"dialogue_id": case["dialogue_id"], "case_index": index,
                      "row_indices": indices.tolist(), "work": case["work"], "payload_seconds": payload_seconds,
                      "parity": equality, "encoder_work": result.encoder_work, "invariants": result.invariants}
            journal_write(journal, record)
            work.update(result.encoder_work)
            progress["completed_cases"] += 1
            budget.storage()
            budget.sync(api.torch)
            records.append({**record, "case_seconds": budget.elapsed() - start})
    return {"cases": records, "endpoint_rows": sum(r["parity"]["rows"] for r in records),
            "witness": finish_fit(route, len(cases), work)}


def infer_fit(api, route, directory, actors, rows_by_dialogue, lexical, arm, profiles, row_count, budget, progress):
    logs = np.lib.format.open_memmap(directory / "partial-log-probs.npy", mode="w+", dtype=np.float32, shape=(row_count, 12))
    indices = np.lib.format.open_memmap(directory / "partial-row-indices.npy", mode="w+", dtype=np.int64, shape=(row_count,))
    logs[:] = -np.inf
    indices[:] = -1
    cursor, work, seconds_total = 0, Counter(), 0.0
    try:
        with (directory / "dialogues.jsonl").open("x") as journal:
            for actor in actors:
                did = actor["dialogue_id"]
                progress.update(operation="calibration-forward", dialogue_id=did, endpoint_rows=cursor)
                budget.sync(api.torch)
                start = budget.elapsed()
                result = route.forward_public(payload(actor, lexical, arm, "infer"))
                validate_work(result, profiles[did])
                # This is the only label-aware call. It runs after the full public trajectory.
                packet = api.collect(result, rows_by_dialogue[did])
                block, ids = numpy_cpu(packet["log_probs"]), numpy_cpu(packet["row_indices"])
                validate_logs(block, rows_by_dialogue[did])
                require(ids.dtype == np.int64 and np.array_equal(ids, np.arange(cursor, cursor + len(ids))),
                        "Every canonical calibration endpoint exactly once, in saved order")
                logs[cursor:cursor + len(ids)], indices[cursor:cursor + len(ids)] = block, ids
                cursor += len(ids)
                logs.flush()
                indices.flush()
                budget.sync(api.torch)
                payload_seconds = budget.elapsed() - start
                journal_write(journal, {"dialogue_id": did, "rows": len(ids), "row_start": cursor - len(ids),
                                       "payload_seconds": payload_seconds, "encoder_work": result.encoder_work,
                                       "invariants": result.invariants})
                work.update(result.encoder_work)
                progress["completed_forwards"] += 1
                progress["endpoint_rows"] = cursor
                budget.storage()
                budget.sync(api.torch)
                seconds_total += budget.elapsed() - start
        require(cursor == row_count, "All calibration rows retained")
        start = budget.elapsed()
        save_packet(directory / "predictions.npz", logs, indices)
        budget.storage()
        finalization = budget.elapsed() - start
    finally:
        primary = sys.exception()
        for array in (logs, indices):
            try:
                array.flush()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note("Partial array flush failed: " + repr(cleanup_error))
        del logs, indices
    witness = finish_fit(route, len(actors), work)
    (directory / "partial-log-probs.npy").unlink()
    (directory / "partial-row-indices.npy").unlink()
    return {"dialogues": len(actors), "endpoint_rows": cursor, "dialogue_seconds": seconds_total,
            "prediction_finalization_seconds": finalization, "witness": witness}


def canonical_calibration(actors, rows, selected):
    require(len(actors) == SAMPLE_SIZE and [a["dialogue_id"] for a in actors] == selected
            and len(set(selected)) == SAMPLE_SIZE, "Exact 512 selected calibration actor order")
    grouped = {did: [] for did in selected}
    for i, row in enumerate(rows):
        require(type(row["row_index"]) is int and row["row_index"] == i
                and row["dialogue_id"] in grouped and row["split"] == row["source_split"] == "train"
                and row["analysis_role"] == "calibration", "Canonical TRAIN calibration evaluator identity")
        grouped[row["dialogue_id"]].append(row)
    require(all(grouped.values()) and [r["row_index"] for did in selected for r in grouped[did]] == list(range(len(rows))),
            "Complete nonempty actor/evaluator joins in selected dialogue order")
    return grouped


def body(args, ctx, budget):
    # Successful parent exits are required before decoding either prepared arrays
    # or loading any neural backend, even when all payload bytes are present.
    prepared = common.authenticate_output(args.prepared, args.prepared_sha256, ctx, "prepare")
    prep_terminal = common.authenticate_terminal(args.prepared_terminal, args.prepared_terminal_sha256, prepared)
    require(ctx["science"]["fit_order"] == FIT_ORDER, "Exact original seed-outer four-arm fit order")
    qualification = None
    if args.command == "infer":
        qualification = common.authenticate_output(args.qualification, args.qualification_sha256, ctx, "qualify")
        common.authenticate_terminal(args.qualification_terminal, args.qualification_terminal_sha256, qualification)
        require(qualification["prepared_sha256"] == args.prepared_sha256
                and qualification["prepared_terminal_sha256"] == args.prepared_terminal_sha256
                and qualification["fit_order"] == FIT_ORDER and qualification["parity_passed"] is True
                and qualification["projection"]["admitted"] is True, "Complete admitted qualification for these inputs")
    selected = common.read(args.prepared / "selection.json")["selected_ids"]
    workloads = common.read(args.prepared / "workloads.json")
    profiles = {p["dialogue_id"]: p["work"] for p in workloads["profiles"]}
    require(len(selected) == SAMPLE_SIZE and len(set(selected)) == SAMPLE_SIZE
            and len(workloads["profiles"]) == SAMPLE_SIZE and set(profiles) == set(selected)
            and all(p["split"] == "train" for p in workloads["profiles"]), "All 512 selected public work profiles")
    totals = {key: sum(profiles[did][key] for did in selected) for key in WORK_MEASURES}
    require(all(totals[k] == workloads["all"]["totals"][k] for k in WORK_MEASURES), "Complete geometry sums")
    if qualification is not None:
        replay_records = [case for fit in qualification["fits"] for case in fit["cases"]]
        checked = projection(replay_records, [fit["loading_seconds"] for fit in qualification["fits"]],
                             totals, prep_terminal["wall_seconds"])
        require(checked == qualification["projection"] and checked["admitted"], "Recomputed fixed admission, no override")
    replay = common.read(args.prepared / "replay-cases.json")
    cases = replay["cases"]
    require(len(cases) == 2 and len({c["dialogue_id"] for c in cases}) == 2, "Two fixed complete DEV cases")
    if args.command == "qualify":
        original = read_lines(ctx["original_prepared"] / "actors-dev.jsonl", budget)
        actors = {a["dialogue_id"]: a for a in original if a["dialogue_id"] in {c["dialogue_id"] for c in cases}}
        require(len(actors) == 2, "Both original complete DEV actors")
        lexical = load_lexical(ctx["original_prepared"])
        rows, grouped = None, None
    else:
        actors = read_lines(args.prepared / "actors.jsonl", budget)
        rows = read_lines(args.prepared / "evaluation-rows.jsonl", budget)
        grouped = canonical_calibration(actors, rows, selected)
        require(len(rows) == prepared["counts"]["scored_endpoints"], "All prepared calibration endpoints")
        lexical = load_lexical(args.prepared)
    api = backend()
    progress = ctx["progress"]
    progress.update(completed_fits=[], completed_cases=0, completed_forwards=0)
    fits, all_cases, loads = [], [], []
    for fit_id in FIT_ORDER:
        directory = args.out / fit_id
        directory.mkdir(exist_ok=False)
        route = None
        try:
            fit_start = budget.elapsed()
            route, loading = load_route(api, ctx, fit_id, budget)
            arm, seed = fit_id.rsplit("-", 1)
            if args.command == "qualify":
                with np.load(Path(ctx["prior"].run) / fit_id / "predictions.npz", allow_pickle=False) as packet:
                    require(set(packet.files) == {"log_probs", "row_indices"}
                            and packet["row_indices"].dtype == np.int64
                            and np.array_equal(packet["row_indices"], np.arange(DEV_ENDPOINTS)), "Original saved row identity")
                    baseline = packet["log_probs"]
                    require(baseline.dtype == np.float32 and baseline.shape == (DEV_ENDPOINTS, 12), "Original raw DEV logs")
                result = qualify_fit(api, route, directory, cases, actors, lexical, arm, baseline, budget, progress)
                del baseline
                all_cases.extend(result["cases"])
            else:
                result = infer_fit(api, route, directory, actors, grouped, lexical, arm, profiles, len(rows), budget, progress)
            receipt = {"status": "completed", "fit_id": fit_id, "arm": arm, "seed": int(seed),
                       "loading_seconds": loading, "wall_seconds": budget.elapsed() - fit_start, **result,
                       "files": common.members(directory)}
            common.write(directory / "completed.json", receipt)
            fits.append({**receipt, "completed_sha256": common.sha(directory / "completed.json")})
            loads.append(loading)
            progress["completed_fits"].append(fit_id)
        finally:
            if route is not None:
                progress["route"] = route.snapshot()
            del route
            gc.collect()
        api.torch.mps.empty_cache()
        budget.sync(api.torch)
        budget.storage()
        print(json.dumps({"phase": args.command, "completed_fits": len(fits), "fit_id": fit_id}), flush=True)
    require([fit["fit_id"] for fit in fits] == FIT_ORDER, "All twelve final checkpoints completed")
    metadata = {"fit_order": FIT_ORDER, "fits": fits, "prepared_sha256": args.prepared_sha256,
                "prepared_terminal_sha256": args.prepared_terminal_sha256,
                "model_weight_updates": 0, "optimizer_created": False, "temperature_applied": False,
                "official_test_opened": False, "task_metrics_computed": False,
                "cost_scope": "Whole phase includes authentication, loading, synchronization, all writes and cleanup. "
                              "Per-dialogue costs include actor materialization, full forward, endpoint gather, raw-array and journal writes, "
                              "storage checks and final synchronization. Journal payload_seconds ends before its own write; "
                              "returned case_seconds includes it. Fit receipts and final digest checks are additional whole-phase costs."}
    if args.command == "qualify":
        screening = projection(all_cases, loads, totals, prep_terminal["wall_seconds"])
        common.write(args.out / "projection.json", screening)
        metadata.update(parity_passed=True, completed_forwards=len(all_cases), projection=screening,
                        parity_scope="Raw saved endpoints and newly monitored updates; no assertion about unsaved old hidden trajectories.")
    else:
        require(progress["completed_forwards"] == SAMPLE_SIZE * len(FIT_ORDER), "All 6144 complete forwards")
        metadata.update(completed_forwards=progress["completed_forwards"], endpoint_rows_per_fit=len(rows),
                        total_endpoint_rows=len(rows) * len(FIT_ORDER), qualification_sha256=args.qualification_sha256,
                        qualification_terminal_sha256=args.qualification_terminal_sha256)
    common.authenticate_output(args.prepared, args.prepared_sha256, ctx, "prepare")
    common.authenticate_terminal(args.prepared_terminal, args.prepared_terminal_sha256, prepared)
    if qualification is not None:
        common.authenticate_output(args.qualification, args.qualification_sha256, ctx, "qualify")
        common.authenticate_terminal(args.qualification_terminal, args.qualification_terminal_sha256, qualification)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("qualify", "infer"):
        part = commands.add_parser(command)
        for name in ("plan", "supervision", "out", "prepared", "prepared-terminal"):
            part.add_argument("--" + name, type=Path, required=True)
        for name in ("plan-sha256", "prepared-sha256", "prepared-terminal-sha256"):
            part.add_argument("--" + name, required=True)
        if command == "infer":
            for name in ("qualification", "qualification-terminal"):
                part.add_argument("--" + name, type=Path, required=True)
                part.add_argument("--" + name + "-sha256", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    common.execute(parse_args(), body)
