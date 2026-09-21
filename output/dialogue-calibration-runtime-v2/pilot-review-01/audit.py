"""Independent saved-only runtime arithmetic and endpoint replay audit.

Only frozen common authentication is reused. Sampling, timing projection,
probability/parity arithmetic and work totals are independently reconstructed.
No producer projection/selection/forward functions or model loaders are called.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import resource
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from openjev.research.suspend_clock import SuspendClock

VERSION = "dialogue-runtime-independent-pilot-audit-v1"
LIMITS = {"wall_seconds": 300, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
ARMS = ("frozen_original", "frozen_numbers", "trainable_original", "trainable_numbers")
FITS = [f"{arm}-{seed}" for seed in (6901, 6902, 6903) for arm in ARMS]
WORK = ("dialogue_count", "encoder_calls", "padded_attention_positions", "real_question_updates")
ENCODER = ("input_texts", "content_tokens", "encoder_sequences", "encoder_calls", "special_token_positions",
           "valid_token_positions", "padded_token_positions", "padded_attention_positions",
           "padding_token_positions", "overlength_texts_chunked", "truncated_tokens")
COUNTS = ("forward_calls", "forward_returned", "advance_calls", "advance_returned", "valid_turns",
          "executed_valid_question_slots", "real_question_updates", "incoming_checks", "feature_checks",
          "result_checks", "mass_checks", "mass_above_one_count")
MAXIMA = ("incoming_max_sum_error", "feature_max_sum_error", "result_max_sum_error", "mass_max_overshoot")
SCOPE = ("Independent fixed sample, all1560 saved replay coverage/parity, public work/count arithmetic, forecast chronology, "
         "all12 verification comparisons and complete fixed2x projection. Frozen common reader authenticates source/input "
         "manifests and actual parent timing. Raw saved outputs and internal monitor witnesses do not replay inference "
         "or establish actual clock/model truth. Geometry range text is authenticated, not independently recomputed. "
         "No task quality metrics, calibration predictions, weights deserialization, TEST or model calls.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class Budget:
    def __init__(self, out, clock):
        self.out, self.clock = out, clock
        self.deadline = clock.deadline_after(300)
        self.progress, self.last = {}, None

    def check(self):
        now = self.clock.now_ns()
        if now >= self.deadline.expires_ns:
            raise TimeoutError("Independent runtime audit deadline expired")
        require(rss() <= LIMITS["rss_bytes"], "Audit RSS cap")
        self.last = now - self.deadline.started_ns

    def storage(self):
        self.check()
        require(sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()) <= LIMITS["output_bytes"], "Audit output cap")


class Checks:
    def __init__(self):
        self.count = 0

    def equal(self, actual, expected, label):
        self.count += 1
        require(actual == expected, "Mismatch: " + label)

    def close(self, actual, expected, label):
        self.count += 1
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Numerical mismatch: " + label)


def authenticate(args, budget):
    require(sha(args.plan) == args.plan_sha256, "External runtime plan pin")
    plan = read(args.plan)
    path = "scripts/dialogue_runtime_common.py"
    require(sha(ROOT / path) == plan["sources"][path], "Qualified common reader source before import")
    common = importlib.import_module("dialogue_runtime_common")
    directory = ROOT / plan["outputs"]["pilot"]
    request = SimpleNamespace(command="pilot", plan=args.plan, plan_sha256=args.plan_sha256, out=directory)
    ctx = common.authenticate(request, budget)
    done = common.authenticate_output(directory, args.pilot_sha256, ctx, "pilot")
    terminal = common.authenticate_terminal(args.terminal, args.terminal_sha256, done)
    require(done["fit_order"] == FITS and [f["fit_id"] for f in done["fits"]] == FITS
            and done["completed_forwards"] == 1560 and done["timed_forwards"] == 1536
            and done["warmup_forwards"] == 24 and done["parity_passed"] is True,
            "Every required fit and public forward completed")
    require(done["calibration_forward_calls"] == 0 and done["model_weight_updates"] == 0
            and done["temperature_fits"] == 0 and done["optimizer_created"] is False
            and done["temperature_applied"] is False and done["task_metrics_computed"] is False
            and done["legacy_v1_admission_revised"] is False and done["official_test_opened"] is False
            and done["scientific_conditions_changed"] is False, "Resource-only outcome without altering prior failure")
    for fit in done["fits"]:
        budget.check()
        target = directory / fit["fit_id"]
        require(sha(target / "completed.json") == fit["completed_sha256"], "Nested fit completion pin")
        require(read(target / "completed.json") == {k: v for k, v in fit.items() if k != "completed_sha256"}, "Nested complete fit identity")
        common.manifest(target, fit["files"])
    return common, ctx, directory, done, terminal


def lines(path, budget):
    result = []
    with Path(path).open() as stream:
        for line in stream:
            budget.check()
            result.append(json.loads(line))
    return result


def geometry(profiles, ids):
    return {"dialogue_count": len(ids), **{k: sum(profiles[did][k] for did in ids) for k in WORK[1:]}}


def reconstruct_sample(ctx, directory, done, checks, budget):
    source = [p for p in read(ctx["original_prepared"] / "workloads.json")["profiles"] if p["split"] == "dev"]
    require(len(source) == 2363 and len({p["dialogue_id"] for p in source}) == 2363, "Complete distinct original DEV profiles")
    profiles = {p["dialogue_id"]: p["work"] for p in source}
    legacy = read(ctx["prepared"] / "replay-cases.json")["cases"]
    warmup = [c["dialogue_id"] for c in legacy]
    require(len(warmup) == len(set(warmup)) == 2 and set(warmup) <= set(profiles), "Both fixed legacy warmups")

    def rank(did):
        return hashlib.sha256(("openjev-calibration-runtime-v2:" + did).encode()).hexdigest(), did

    order = sorted(set(profiles) - set(warmup), key=rank)[:128]
    expected = {"warmup": warmup, "estimator": order[:64], "verification": order[64:]}
    sample = read(directory / "sample.json")
    checks.equal(sha(directory / "sample.json"), done["sample_sha256"], "sample pin")
    checks.equal(sample["warmup_ids"], warmup, "legacy warmup order")
    checks.equal(sample["fit_order"], FITS, "sample all fits")
    checks.equal(sample["salt"], "openjev-calibration-runtime-v2:", "sampling salt")
    checks.equal(sample["counts"], {"evaluated_dev_dialogues": 2363, "excluded_legacy_replay_dialogues": 2,
                 "eligible_dialogues": 2361, "selected_dialogues": 128, "estimator_dialogues": 64,
                 "verification_dialogues": 64, "fits": 12, "timed_forwards": 1536, "paid_parity_checked_warmups": 24}, "sample counts")
    for block in ("estimator", "verification"):
        ids = expected[block]
        checks.equal(sample[block]["dialogue_ids"], ids, "hash-selected " + block)
        checks.equal(sample[block]["work"], geometry(profiles, ids), "sample block work")
        checks.equal(sample[block]["profiles"], [{"dialogue_id": did, "salted_id_sha256": rank(did)[0],
                      "work": geometry(profiles, [did])} for did in ids], "sample per-dialogue public work")
    calibration = read(ctx["prepared"] / "workloads.json")["profiles"]
    require(len(calibration) == len({p["dialogue_id"] for p in calibration}) == 512
            and all(p["split"] == "train" for p in calibration), "Complete calibration geometry only")
    calprofiles = {p["dialogue_id"]: p["work"] for p in calibration}
    calwork = geometry(calprofiles, list(calprofiles))
    checks.equal(calwork, done["calibration_work"], "all512 calibration work")
    selected = set(warmup + order)
    endpoints = {did: [] for did in selected}
    with (Path(ctx["prior"].run) / "evaluation-rows.jsonl").open() as stream:
        count = 0
        for line in stream:
            budget.check()
            row = json.loads(line)
            require(type(row["row_index"]) is int and row["row_index"] == count and row["split"] == "dev", "Original canonical public addresses")
            count += 1
            if row["dialogue_id"] in selected:
                ids = row["candidate_ids"]
                require(3 <= len(ids) <= 12 and len(set(ids)) == len(ids), "Canonical candidate support")
                endpoints[row["dialogue_id"]].append((row["row_index"], len(ids)))
    require(count == 62329 and all(endpoints.values()), "All original rows and every selected dialogue")
    return sample, profiles, expected, calwork, endpoints


def raw_packet(path, row_count):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"row_indices", "log_probs"}, "Exact raw packet fields")
        logs, indices = data["log_probs"], data["row_indices"]
    require(logs.dtype == np.float32 and logs.shape == (row_count, 12)
            and indices.dtype == np.int64 and indices.shape == (row_count,), "Raw packet dtype and shape")
    return logs, indices


def parity_values(new, original, counts):
    support = np.arange(12)[None, :] < np.asarray(counts)[:, None]
    for values in (new, original):
        require(np.isfinite(values[support]).all() and np.isneginf(values[~support]).all(), "Finite supported raw logs and exact padding")
    a, b = new.astype(np.float64), original.astype(np.float64)
    pa, pb = np.exp(a), np.exp(b)
    mass_a = max(abs(math.fsum(map(float, row)) - 1.) for row in pa)
    mass_b = max(abs(math.fsum(map(float, row)) - 1.) for row in pb)
    log_difference = float(np.max(np.abs(a[support] - b[support])))
    probability_difference = float(np.max(np.abs(pa - pb)))
    same = bool(np.array_equal(np.argmax(a, axis=1), np.argmax(b, axis=1)))
    require(mass_a <= 2e-6 and mass_b <= 2e-6 and log_difference <= 1e-5
            and probability_difference <= 1e-6 and same, "Every saved replay endpoint satisfies fixed numerical parity")
    return {"rows": len(new), "maximum_supported_log_error": log_difference,
            "maximum_probability_error": probability_difference, "canonical_first_argmax_equal": same,
            "maximum_raw_mass_error": mass_a, "baseline_maximum_raw_mass_error": mass_b}


def check_invariants(inv, profile, checks):
    t, q = profile["public_user_turns"], profile["real_question_updates"]
    expected = {"forward_calls": 1, "forward_returned": 1,
                **dict.fromkeys(("advance_calls", "advance_returned", "valid_turns"), t),
                **dict.fromkeys(("executed_valid_question_slots", "real_question_updates", "incoming_checks",
                                 "feature_checks", "result_checks", "mass_checks"), q)}
    for name, count in expected.items():
        checks.equal(inv[name], count, "monitored public update coverage")
    require(inv["tolerance"] == 2e-6 and 0 <= inv["mass_above_one_count"] <= q
            and 0 <= inv["mass_min"] <= inv["mass_max"] <= 1 + 2e-6
            and all(math.isfinite(inv[k]) and 0 <= inv[k] <= 2e-6 for k in MAXIMA), "Finite internal invariant witnesses")


def fit_arithmetic(directory, fit, sample, profiles, blocks, endpoints, ctx, checks, budget, previous_end):
    name = fit["fit_id"]
    order = [did for b in ("warmup", "estimator", "verification") for did in blocks[b]]
    arm, seed = name.rsplit("-", 1)
    checks.equal((fit["status"], fit["arm"], fit["seed"]), ("completed", arm, int(seed)), "fit identity")
    require(math.isfinite(fit["loading_seconds"]) and fit["loading_seconds"] > 0
            and math.isfinite(fit["wall_seconds"]) and fit["wall_seconds"] > fit["loading_seconds"], "Positive paid fit/load durations")
    refs = [row for did in order for row in endpoints[did]]
    new, indices = raw_packet(directory / name / "predictions.npz", len(refs))
    original, original_indices = raw_packet(Path(ctx["prior"].run) / name / "predictions.npz", 62329)
    require(np.array_equal(original_indices, np.arange(62329)) and indices.tolist() == [r[0] for r in refs], "Exact complete saved endpoint order")
    checks.equal(fit["endpoint_rows"], len(refs), "fit endpoint count")
    rows = lines(directory / name / "dialogues.jsonl", budget)
    require(len(rows) == 130, "Complete per-dialogue journal")
    cursor, audit = 0, []
    for record, did in zip(rows, order, strict=True):
        budget.check()
        block_name = "warmup" if len(audit) < 2 else "estimator" if len(audit) < 66 else "verification"
        checks.equal((record["block"], record["dialogue_id"], record["block_position"]),
                     (block_name, did, blocks[block_name].index(did)), "journal chronology")
        identities, counts = zip(*endpoints[did], strict=True)
        checks.equal(record["row_indices"], list(identities), "all dialogue endpoints")
        value = parity_values(new[cursor:cursor+len(identities)], original[list(identities)], counts)
        for key, expected in value.items():
            if type(expected) is float:
                checks.close(record["parity"][key], expected, "independent endpoint parity " + key)
            else:
                checks.equal(record["parity"][key], expected, "independent endpoint parity " + key)
        checks.equal(record["encoder_work"], {k: profiles[did][k] for k in ENCODER}, "complete actual encoder work")
        check_invariants(record["invariants"], profiles[did], checks)
        audit.append(record)
        cursor += len(identities)
    witness = fit["witness"]
    require(witness["synthetic_injection"] is False and witness["encoder_device"].split(":")[0] == "mps"
            and witness["memory_device"] == "cpu" and witness["dtype"] == "float32"
            and witness["chunk_tokens"] == 254 and witness["encoder_batch"] == 32
            and witness["optimizer_created"] is False and witness["temperature_applied"] is False,
            "Original production inference path witness")
    checks.equal(witness["encoder_work"], {k: sum(profiles[did][k] for did in order) for k in ENCODER}, "fit encoder work")
    dispatched = ("encoder_calls", "encoder_sequences", "valid_token_positions", "padded_token_positions", "padded_attention_positions")
    for kind in ("encoder_attempted_work", "encoder_returned_work"):
        checks.equal(witness[kind], {k: sum(profiles[did][k] for did in order) for k in dispatched}, "Actual encoder dispatch work")
    for key in COUNTS:
        checks.equal(witness["invariants"][key], sum(r["invariants"][key] for r in audit), "fit invariant count")
    for key in MAXIMA:
        checks.close(witness["invariants"][key], max(r["invariants"][key] for r in audit), "fit invariant maximum")
    checks.close(witness["invariants"]["mass_min"], min(r["invariants"]["mass_min"] for r in audit), "fit departure minimum")
    checks.close(witness["invariants"]["mass_max"], max(r["invariants"]["mass_max"] for r in audit), "fit departure maximum")
    for operation in ("checkpoint_load", "encoder_load", "public_forward", "encoding", "memory_forward", "encoder_forward"):
        expected = 1 if operation in ("checkpoint_load", "encoder_load") else sum(profiles[d]["encoder_calls"] for d in order) if operation == "encoder_forward" else 130
        for suffix in ("attempts", "returns"):
            checks.equal(witness["counts"][operation + "_" + suffix], expected, "actual complete dispatches")
    checks.equal(witness["final_sha256"], witness["restored_sha256"], "unchanged final weights witness")
    oldfit = read(Path(ctx["prior"].run) / name / "completed.json")
    checks.equal(witness["restored_sha256"]["encoder"], oldfit["final_encoder_sha256"], "final encoder identity")
    checks.equal(witness["checkpoint_sha256"], oldfit["checkpoint"]["sha256"], "original final checkpoint identity")
    forecast_path = directory / name / "forecast.json"
    forecast = read(forecast_path)
    checks.equal(sha(forecast_path), fit["forecast_sha256"], "forecast pin")
    checks.equal((forecast["fit_id"], forecast["sample_sha256"], forecast["verification_forwards_started"]),
                 (name, sha(directory / "sample.json"), 0), "forecast before verification identity")
    require(fit["estimator"]["finished_elapsed_seconds"] <= forecast["created_elapsed_seconds"]
            <= fit["forecast_published_elapsed_seconds"] <= fit["verification"]["started_elapsed_seconds"], "Prediction publication chronology")
    for block_name in ("warmup", "estimator", "verification"):
        block = fit[block_name]
        checks.equal(block["dialogue_ids"], blocks[block_name], "fixed block order")
        checks.equal(block["work"], geometry(profiles, blocks[block_name]), "independent block work")
        checks.close(block["seconds"], block["finished_elapsed_seconds"] - block["started_elapsed_seconds"], "complete block elapsed")
        require(previous_end <= block["started_elapsed_seconds"] < block["finished_elapsed_seconds"], "Disjoint ordered native intervals")
        previous_end = block["finished_elapsed_seconds"]
        members = [r for r in audit if r["block"] == block_name]
        require(all(math.isfinite(r["payload_seconds"]) and r["payload_seconds"] > 0 for r in members)
                and math.fsum(r["payload_seconds"] for r in members) <= block["seconds"], "Block charges every case payload")
        checks.equal(block["encoder_work"], {k: sum(r["encoder_work"][k] for r in members) for k in ENCODER}, "block encoder sums")
        checks.equal(block["endpoint_rows"], sum(len(r["row_indices"]) for r in members), "block endpoint sum")
        for key in ("maximum_supported_log_error", "maximum_probability_error", "maximum_raw_mass_error"):
            checks.close(block["parity_maxima"][key], max(r["parity"][key] for r in members), "block parity maximum")
    ew, vw = (geometry(profiles, blocks[b]) for b in ("estimator", "verification"))
    ratios = {k: vw[k] / ew[k] for k in WORK}
    predicted = 2.0 * fit["estimator"]["seconds"] * max(ratios.values())
    prediction = forecast["prediction"]
    checks.equal(prediction["estimator_work"], ew, "forecast estimator work")
    checks.equal(prediction["verification_work"], vw, "forecast verification work")
    checks.equal(prediction["safety_factor"], 2.0, "fixed factor of two")
    checks.equal(prediction["fit_id"], name, "forecast fit identity")
    checks.close(prediction["estimator_seconds"], fit["estimator"]["seconds"], "complete estimator duration")
    for key in WORK:
        checks.close(prediction["work_ratios"][key], ratios[key], "verification work ratio")
    checks.close(prediction["maximum_work_ratio"], max(ratios.values()), "maximum verification geometry ratio")
    checks.close(prediction["predicted_seconds"], predicted, "independent forecast")
    actual = fit["verification"]["seconds"]
    checks.equal(fit["verification_result"]["prediction"], prediction, "verification does not refit")
    checks.close(fit["verification_result"]["actual_seconds"], actual, "actual verification block cost")
    checks.close(fit["verification_result"]["actual_over_prediction"], actual / predicted, "verification ratio")
    checks.equal(fit["verification_result"]["passed"], actual <= predicted, "inclusive independent verification")
    return {"rows": len(refs), "estimator_seconds": fit["estimator"]["seconds"], "estimator_work": ew,
            "verification_seconds": actual, "verification_predicted_seconds": predicted,
            "verification_passed": actual <= predicted, "loading_seconds": fit["loading_seconds"],
            "warmup_seconds": fit["warmup"]["seconds"]}, previous_end


def recompute(ctx, directory, done, terminal, budget):
    checks = Checks()
    sample, profiles, blocks, calwork, endpoints = reconstruct_sample(ctx, directory, done, checks, budget)
    fits, previous_end = {}, 0.
    for fit in done["fits"]:
        budget.progress["fit_id"] = fit["fit_id"]
        value, previous_end = fit_arithmetic(directory, fit, sample, profiles, blocks, endpoints, ctx, checks, budget, previous_end)
        fits[fit["fit_id"]] = value
    result = read(directory / "projection.json")
    checks.equal(result, done["projection"], "root and saved projection identity")
    checks.equal(result["fit_order"], FITS, "projection complete fit order")
    for key in ("fits", "verification", "loading_seconds_by_fit", "warmup_seconds_by_fit"):
        checks.equal(set(result[key]), set(FITS), "projection complete mapping " + key)
    for fit in done["fits"]:
        name = fit["fit_id"]
        checks.equal(result["verification"][name], fit["verification_result"], "projection uses recorded verification")
        checks.close(result["loading_seconds_by_fit"][name], fit["loading_seconds"], "projection uses paid load")
        checks.close(result["warmup_seconds_by_fit"][name], fit["warmup"]["seconds"], "projection uses paid warmups")
    require(previous_end <= done["projection_elapsed_seconds"] <= done["wall_seconds"] <= terminal["wall_seconds"], "Final timing snapshot inside parent")
    disjoint = math.fsum(f["loading_seconds"] + math.fsum(f[b]["seconds"] for b in ("warmup", "estimator", "verification")) for f in done["fits"])
    overhead = done["projection_elapsed_seconds"] - disjoint
    require(overhead >= 0, "Nonnegative paid nonblock cost")
    checks.close(done["disjoint_measured_seconds"], disjoint, "disjoint measured cost")
    checks.close(done["pilot_nonblock_overhead_seconds"], overhead, "complete nonblock overhead")
    projected = {}
    for name, value in fits.items():
        ratios = {key: calwork[key] / value["estimator_work"][key] for key in WORK}
        projected[name] = 2 * value["estimator_seconds"] * max(ratios.values())
        recorded = result["fits"][name]
        checks.equal(recorded["estimator_work"], value["estimator_work"], "calibration forecast source geometry")
        checks.equal(recorded["calibration_work"], calwork, "calibration forecast target geometry")
        checks.equal(recorded["safety_factor"], 2.0, "calibration forecast fixed factor")
        checks.close(recorded["estimator_seconds"], value["estimator_seconds"], "no verification refit")
        for key in WORK:
            checks.close(recorded["work_ratios"][key], ratios[key], "calibration work ratio")
        checks.close(recorded["maximum_work_ratio"], max(ratios.values()), "calibration maximum ratio")
        checks.close(recorded["projected_seconds"], projected[name], "all512 fit forecast")
    components = {"projected_calibration_forward_seconds": math.fsum(projected.values()),
                  "loading_seconds": math.fsum(f["loading_seconds"] for f in fits.values()),
                  "warmup_seconds": math.fsum(f["warmup_seconds"] for f in fits.values()),
                  "preparation_parent_wall_seconds": ctx["preparation_terminal"]["wall_seconds"],
                  "pilot_nonblock_overhead_seconds": overhead}
    checks.equal(set(result["components"]), set(components), "all nonnested projection components")
    for key, value in components.items():
        checks.close(result["components"][key], value, "nonduplicated total component")
    total = math.fsum(components.values())
    passed = sum(f["verification_passed"] for f in fits.values())
    checks.close(result["total_seconds"], total, "complete estimator-only projection")
    for key, value in {"threshold_seconds": 1800, "all_verifications_passed": passed == 12,
                       "verifications_passed": passed, "verifications_total": 12,
                       "projection_within_threshold": total <= 1800, "admitted": passed == 12 and total <= 1800,
                       "projected_dialogue_forwards": 6144, "legacy_v1_admission_revised": False,
                       "scientific_conditions_changed": False}.items():
        checks.equal(result[key], value, "unchanged fixed admission " + key)
    return {"status": "completed", "agreement": True, "fit_order": FITS, "fits": fits,
            "verification_passed": passed, "verification_total": 12, "total_projected_seconds": total,
            "admitted": passed == 12 and total <= 1800, "threshold_seconds": 1800,
            "components": components, "completed_forwards": 1560,
            "saved_replay_endpoints_checked": sum(f["rows"] for f in fits.values()),
            "checks": checks.count, "whole_pilot_parent_seconds": terminal["wall_seconds"],
            "projection_to_actual_parent_exit_seconds": terminal["wall_seconds"] - done["projection_elapsed_seconds"],
            "actual_pilot_cost_scope": "All pilot forwards are paid sunk work, separate from the future512 forecast; publication/cleanup tail is shown, not hidden.",
            "scope": SCOPE}


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    clock = budget = handler = None
    request = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source_pin = None
    try:
        source_pin = sha(__file__)
        clock = SuspendClock()
        require(clock.backend in {"mach_continuous_time", "CLOCK_BOOTTIME"}, "Native audit clock")
        budget = Budget(args.out, clock)

        def expired(*_):
            raise TimeoutError("Supplementary independent audit alarm expired")

        handler = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, 300)
        write(args.out / "started.json", {"version": VERSION, "request": request, "source_sha256": source_pin,
              "limits": LIMITS, "model_calls": 0})
        common, ctx, directory, done, terminal = authenticate(args, budget)
        summary = recompute(ctx, directory, done, terminal, budget)
        write(args.out / "summary.json", summary)
        common.manifest(directory, done["files"])
        authenticate(args, budget)
        require(sha(__file__) == source_pin, "Stable independent audit source")
        budget.storage()
        elapsed = budget.last
        receipt = {"version": VERSION, "status": "completed", "agreement": True, "request": request,
                   "plan_sha256": args.plan_sha256, "pilot_completed_sha256": args.pilot_sha256,
                   "terminal_sha256": args.terminal_sha256, "source_sha256": source_pin,
                   "files": {name: {"sha256": sha(args.out / name), "bytes": (args.out / name).stat().st_size}
                             for name in ("started.json", "summary.json")}, "limits": LIMITS,
                   "clock_backend": clock.backend, "started_ns": budget.deadline.started_ns,
                   "finished_ns": budget.deadline.started_ns + elapsed, "deadline_ns": budget.deadline.expires_ns,
                   "elapsed_ns": elapsed, "wall_seconds": elapsed / 1e9, "timing_available": True,
                   "peak_rss_bytes": rss(), "model_calls": 0, "admitted": summary["admitted"],
                   "checks": summary["checks"], "scope": SCOPE}
        write(args.out / "receipt.json", receipt)
        budget.storage()
        return receipt
    except BaseException as error:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if (args.out / "receipt.json").exists():
                (args.out / "receipt.json").rename(args.out / "invalid-receipt.json")
            write(args.out / "failed.json", {"version": VERSION, "status": "failed", "agreement": False,
                  "request": request, "source_sha256": source_pin, "error_type": type(error).__name__, "error": str(error),
                  "timing_available": False, "wall_seconds": None, "elapsed_ns": None,
                  "last_successful_elapsed_ns": None if budget is None else budget.last,
                  "progress": None if budget is None else budget.progress, "model_calls": 0})
        except BaseException as secondary:  # noqa: BLE001 - preserve primary audit failure
            error.add_note("Failure receipt: " + repr(secondary))
        raise
    finally:
        if handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "terminal", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "pilot-sha256", "terminal-sha256"):
        parser.add_argument("--" + name, required=True)
    outcome = execute(parser.parse_args())
    print(json.dumps({"status": outcome["status"], "agreement": outcome["agreement"], "admitted": outcome["admitted"]}))
