"""Complete fixed calibration inference after admitted runtime V2 qualification.

The new common lifecycle authenticates the complete pilot and actual parent
exit before body is entered. Model restoration, full-stream execution and
endpoint collection reuse the unchanged calibration inference implementation.
There is no temperature fitting, task scoring, training or new DEV inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import dialogue_runtime_common as common
import replay_dialogue_calibration as replay

require = common.require
FIT_ORDER = list(replay.FIT_ORDER)
SAMPLE_SIZE = replay.SAMPLE_SIZE


def calibration_inputs(ctx, budget):
    """Validate the already authenticated 512-dialogue preparation before loading."""
    budget.check()
    prepared = ctx["prepared"]
    selection = common.read(prepared / "selection.json")
    selected = selection["selected_ids"]
    require(type(selected) is list and len(selected) == len(set(selected)) == SAMPLE_SIZE
            and all(type(did) is str and did for did in selected)
            and selection["sample_size"] == SAMPLE_SIZE and selection["source_split"] == "train"
            and selection["analysis_role"] == "calibration", "Unchanged exact 512-dialogue TRAIN calibration selection")
    workloads = common.read(prepared / "workloads.json")
    records = workloads["profiles"]
    profiles = {record["dialogue_id"]: record["work"] for record in records}
    require(len(records) == SAMPLE_SIZE and set(profiles) == set(selected)
            and all(record["split"] == "train" for record in records)
            and workloads["all"]["dialogues"] == SAMPLE_SIZE
            and ctx["preparation_receipt"]["counts"]["dialogues"] == SAMPLE_SIZE,
            "All original selected TRAIN profiles, without replacement or omission")
    measured = (*replay.ENCODER_KEYS, "public_user_turns", "real_question_updates")
    for did in selected:
        budget.check()
        require(all(type(profiles[did][key]) is int and profiles[did][key] >= 0 for key in measured)
                and all(profiles[did][key] > 0 for key in replay.WORK_MEASURES), "Complete integer calibration work")
    totals = {key: sum(profiles[did][key] for did in selected) for key in measured}
    require(all(totals[key] == workloads["all"]["totals"][key] for key in measured), "All prepared encoder and recurrence sums")
    calibration_work = {"dialogue_count": SAMPLE_SIZE, **{key: totals[key] for key in replay.WORK_MEASURES}}
    require(calibration_work == ctx["pilot_receipt"]["calibration_work"], "Admitted pilot projected this exact calibration work")
    actors = replay.read_lines(prepared / "actors.jsonl", budget)
    require(all(actor["split"] == actor["source_split"] == "train" and actor["analysis_role"] == "calibration"
                and not ({"label", "label_id", "label_index", "bin", "stratum", "target", "rows"} & set(actor))
                for actor in actors), "Public TRAIN actor payloads remain separate from evaluator targets")
    rows = replay.read_lines(prepared / "evaluation-rows.jsonl", budget)
    grouped = replay.canonical_calibration(actors, rows, selected)
    require(len(rows) == ctx["preparation_receipt"]["counts"]["scored_endpoints"], "Every prepared calibration endpoint")
    budget.check()
    return actors, rows, grouped, profiles, calibration_work


def body(args, ctx, budget):
    budget.check()
    require(args.command == "infer" and ctx["science"]["fit_order"] == FIT_ORDER
            == ctx["runtime_plan"]["config"]["fit_order"], "Unchanged twelve final checkpoints in frozen order")
    pilot = ctx["pilot_receipt"]
    require(pilot["fit_order"] == FIT_ORDER and pilot["projection"]["admitted"] is True
            and pilot["projection"]["all_verifications_passed"] is True
            and pilot["projection"]["verifications_passed"] == pilot["projection"]["verifications_total"] == 12
            and pilot["parity_passed"] is True and pilot["completed_forwards"] == 12 * 130,
            "Authenticated complete admitted runtime pilot before calibration inference")
    require(ctx["pilot_terminal"]["status"] == ctx["preparation_terminal"]["status"] == "completed"
            and ctx["pilot_terminal"]["returncode"] == ctx["preparation_terminal"]["returncode"] == 0,
            "Both prerequisite parent processes completed successfully")
    actors, rows, grouped, profiles, calibration_work = calibration_inputs(ctx, budget)
    lexical = replay.load_lexical(ctx["prepared"])
    api = replay.backend()
    progress = ctx["progress"]
    progress.update(completed_fits=[], completed_forwards=0, endpoint_rows=0)
    fits = []
    for fit_id in FIT_ORDER:
        directory = args.out / fit_id
        directory.mkdir(exist_ok=False)
        route = None
        try:
            started = budget.elapsed()
            route, loading = replay.load_route(api, ctx, fit_id, budget)
            arm, seed = fit_id.rsplit("-", 1)
            result = replay.infer_fit(api, route, directory, actors, grouped, lexical, arm, profiles,
                                      len(rows), budget, progress)
            require(result["dialogues"] == SAMPLE_SIZE and result["endpoint_rows"] == len(rows),
                    "Every full calibration dialogue and endpoint returned for this fit")
            files = common.members(directory)
            require(set(files) == {"predictions.npz", "dialogues.jsonl"}, "Exact completed inference fit payloads")
            receipt = {"status": "completed", "fit_id": fit_id, "arm": arm, "seed": int(seed),
                       "loading_seconds": loading, "wall_seconds": budget.elapsed() - started,
                       **result, "files": files}
            common.write(directory / "completed.json", receipt)
            fits.append({**receipt, "completed_sha256": common.sha(directory / "completed.json")})
            progress["completed_fits"].append(fit_id)
        finally:
            primary = sys.exception()
            if route is not None:
                try:
                    progress["route"] = route.snapshot()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("Route snapshot during inference cleanup: " + repr(error))
            del route
            gc.collect()
        api.torch.mps.empty_cache()
        budget.sync(api.torch)
        budget.storage()
        print(json.dumps({"phase": "runtime-inference", "completed_fits": len(fits), "fit_id": fit_id}), flush=True)
    require([fit["fit_id"] for fit in fits] == FIT_ORDER and progress["completed_fits"] == FIT_ORDER
            and progress["completed_forwards"] == SAMPLE_SIZE * len(FIT_ORDER), "All 6144 complete calibration forwards")
    return {
        "fit_order": FIT_ORDER, "fits": fits, "source_split": "train", "analysis_role": "calibration",
        "prepared_sha256": common.PINS[common.PREPARED + "/completed.json"],
        "prepared_terminal_sha256": common.PINS[common.PREP_TERMINAL],
        "pilot_sha256": args.pilot_sha256, "pilot_terminal": str(args.pilot_terminal.resolve()),
        "pilot_terminal_sha256": args.pilot_terminal_sha256,
        "calibration_work": calibration_work, "completed_forwards": progress["completed_forwards"],
        "endpoint_rows_per_fit": len(rows), "total_endpoint_rows": len(rows) * len(FIT_ORDER),
        "model_weight_updates": 0, "optimizer_created": False, "temperature_applied": False, "temperature_fits": 0,
        "official_test_opened": False, "official_dev_inference": False, "task_metrics_computed": False,
        "legacy_v1_admission_revised": False, "scientific_conditions_changed": False,
        "scope": "Same 512 prepared TRAIN dialogues and twelve final checkpoints; complete raw inference only. "
                 "No task scoring or temperature fitting. Original inference code and scientific criteria remain unchanged.",
        "cost_scope": "Parent deadline pays authentication, checkpoint loading, all complete forwards, synchronization, "
                      "raw array and journal writes, parameter verification, final receipts and cleanup.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "supervision", "out", "pilot-terminal"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "pilot-sha256", "pilot-terminal-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.set_defaults(command="infer")
    return parser.parse_args(argv)


def main(argv=None):
    return common.execute(parse_args(argv), body)


if __name__ == "__main__":
    main()
