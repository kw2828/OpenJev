"""Prospective saved-spatial-study audit; imports no model or arrays at import."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "otto-spatial-study-saved-audit-v1"
RUNNER = "scripts/study_otto_spatial.py"
MODEL = "src/openjev/research/otto_spatial_value.py"
MODEL_PIN = "1b57a7e46edd81ad3d8459af4d90bd2b8170f19a121ab7ef8a2b768ea706f857"
HELPER = "scripts/audit_otto_capacity.py"
HELPER_PIN = "68b6c39837c8bf58ab519bf0437a7e7ffb39c0cd66fcf2c26274786fbf3073b0"
CLOCK = "src/openjev/research/suspend_clock.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
KINDS = ("spatial", "neighbor_free", "cnn", "dense128", "statistics")
SEEDS = (10101, 10102, 10103)
ROWS = {"train": 5589, "valid": 1109}
COUNTS = {"model_initialization": 15, "checkpoint_export": 30, "optimizer_initialization": 15,
          "training_forward": 52800, "backward": 52800, "optimizer_update": 52800,
          "deployment_construction": 15, "parity_restore": 15,
          "numpy_prediction": 6300, "torch_prediction": 6300}
LIMITS = {"seconds": 1800, "rss_bytes": 4*1024**3, "output_bytes": 128*1024**2}
THREADS = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
SCOPE = ("All saved final checkpoints replayed through the pinned, separately qualified NumPy implementation; "
         "that algebra is shared with producer NumPy scoring, not a third independent network implementation. "
         "The replay is compared with both saved NumPy and independent Torch64 values. Cache metadata, public "
         "posterior witnesses, legacy features, targets, centered inputs, epoch permutations, update allocations, "
         "loss denominators and scalar summaries are checked independently of the study runner. "
         "Original cached posterior correctness, Torch random draws, training/gradient/optimizer execution and "
         "timing truth remain authenticated historical/producer evidence. No optimization, environment, new "
         "data collection, action scoring, or efficacy gate executes. Initial zero-final-layer functions are "
         "checked algebraically; no extra initialization readouts are hidden in the replay allocation.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_threads():
    require(all(os.environ.get(key) == value for key, value in THREADS.items()), "audit CPU1 environment before NumPy import")


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload_names():
    names = {"started.json", "runtime.json", "preparation.json", "summary.json", "work.jsonl",
             "initializations.jsonl", "epoch-orders.jsonl", "updates.jsonl", "fit-curves.jsonl",
             "fits.jsonl", "parity.jsonl"}
    names.update(f"{phase}-{kind}-{seed}.npz" for seed in SEEDS for kind in KINDS
                 for phase in ("initial", "final", "predictions"))
    return names


def center_batch(beliefs, positions, np):
    """Independent lossless public-board placement, preserving original f64."""
    n = len(beliefs)
    require(beliefs.shape == (n, 53, 53) and beliefs.dtype == np.float64 and 0 < n <= 128
            and np.isfinite(beliefs).all() and (beliefs >= 0).all(), "finite public float64 belief batch")
    require(positions.shape == (n, 2) and positions.dtype == np.int64
            and (positions >= 0).all() and (positions <= 52).all(), "integer public positions")
    result = np.zeros((n, 105, 105), np.float64)
    for i, q in enumerate(positions):
        x, y = map(int, q)
        result[i, 52-x:105-x, 52-y:105-y] = beliefs[i]
    return result


def shapes_for(kind):
    require(kind in KINDS, "declared spatial family")
    result = {}
    if kind == "cnn":
        for i, channels in enumerate((1, 4, 4)):
            result[f"conv_weight_{i}"], result[f"conv_bias_{i}"] = (4, channels, 3, 3), (4,)
    if kind in ("spatial", "neighbor_free", "cnn"):
        widths = (20, 8, 8) if kind == "cnn" else (12, 24, 8)
        for i in (0, 1):
            result[f"cell_weight_{i}"], result[f"cell_bias_{i}"] = (widths[i+1], widths[i]), (widths[i+1],)
    widths = (11028, 128, 1) if kind == "dense128" else (12, 16, 1)
    for i in (0, 1):
        result[f"readout_weight_{i}"], result[f"readout_bias_{i}"] = (widths[i+1], widths[i]), (widths[i+1],)
    return result


def checkpoint(saved, kind, c0, np, *, initial=False):
    shapes = shapes_for(kind)
    require(set(saved) == {*shapes, "version", "kind", "input_dim", "c0"}, "exact checkpoint member set")
    for key in ("version", "kind", "input_dim", "c0"):
        require(saved[key].shape == (), "scalar checkpoint metadata")
    require(saved["version"].dtype.kind in "US" and saved["version"].item() == "otto-spatial-value-v1"
            and saved["kind"].dtype.kind in "US" and saved["kind"].item() == kind
            and saved["input_dim"].dtype.kind in "iu" and saved["input_dim"].item() == 11028,
            "checkpoint architecture identity")
    for key, shape in {**shapes, "c0": ()}.items():
        require(saved[key].shape == shape and saved[key].dtype == np.float32
                and np.isfinite(saved[key]).all(), "finite exact f32 checkpoint tensor: " + key)
    require(float(saved["c0"]) == c0, "unchanged TRAIN-only baseline")
    if initial:
        for key in shapes:
            if "bias" in key or key == "readout_weight_1":
                require(np.count_nonzero(saved[key]) == 0, "zero final readout and initial biases")
            else:
                require(np.count_nonzero(saved[key]) > 0, "nonzero hidden initialization")
    return sum(math.prod(shape) for shape in shapes.values())


def compare_values(actual, reference, np, *, atol=1e-8, rtol=1e-10):
    require(actual.shape == reference.shape and actual.ndim == 1 and len(actual) > 0
            and actual.dtype == reference.dtype == np.float64
            and np.isfinite(actual).all() and np.isfinite(reference).all(), "finite complete prediction vectors")
    error = np.abs(actual-reference)
    require(np.all(error <= atol+rtol*np.abs(reference)), "unchanged deployed parity predicate")
    return float(error.max(initial=0))


def epoch_records(fit_id, epoch, order, order_row, update_rows, curve, work, np, helper):
    order_sha = hashlib.sha256(order.tobytes(order="C")).hexdigest()
    helper.close(order_row, {"fit_id": fit_id, "epoch": epoch, "rows": len(order),
                             "order": order.tolist(), "sha256": order_sha}, "epoch order")
    losses, seconds = [], 0.0
    for batch, offset in enumerate(range(0, len(order), 128)):
        indices = order[offset:offset+128]
        context = {"fit_id": fit_id, "epoch": epoch, "batch": batch}
        row = next(update_rows)
        require(set(row) == {*context, "rows", "batch_indices_sha256", "loss", "gradient_norm", "update_index"}
                and all(row[k] == value for k, value in context.items())
                and type(row["rows"]) is int and row["rows"] == len(indices)
                and row["batch_indices_sha256"] == hashlib.sha256(indices.tobytes(order="C")).hexdigest()
                and row["update_index"] == (epoch-1)*44+batch+1, "complete batch identities and short tail")
        require(all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ("loss", "gradient_norm")), "finite recorded online loss/preclip gradient norm")
        losses.append(float(row["loss"])*len(indices))
        for channel in ("training_forward", "backward", "optimizer_update"):
            seconds += work.take(channel, context)
    helper.close(curve, {"fit_id": fit_id, "epoch": epoch, "rows": len(order), "updates": len(losses),
                        "training_mse_normalized": math.fsum(losses)/len(order), "order_sha256": order_sha,
                        "scope": "row-weighted mean of pre-update minibatch losses; not final-checkpoint MSE"}, "online epoch reduction")
    return seconds


def aggregate(fits):
    require([row["fit_id"] for row in fits] == [f"{kind}@{seed}" for seed in SEEDS for kind in KINDS], "all fifteen fixed fits")
    means = {}
    for kind in KINDS:
        group = [row for row in fits if row["kind"] == kind]
        means[kind] = {split: {key: math.fsum(r["metrics"][split][key] for r in group)/3
                              for key in ("mse_normalized", "mae_physical", "negative_predictions",
                                          "minimum_normalized", "maximum_normalized")}
                       for split in ROWS}
    return {"version": "otto-spatial-study-v1", "metrics": {row["fit_id"]: row["metrics"] for row in fits},
            "family_means": means,
            "training_costs": {row["fit_id"]: row["fit_seconds"] for row in fits},
            "diagnostic_costs": {row["fit_id"]: row["diagnostic_seconds"] for row in fits},
            "rows": dict(ROWS), "fits": [row["fit_id"] for row in fits], "scalar_admission_gate": None, "alias_floor": None,
            "learned_architecture_advantage_established": False,
            "scope": "All final original-float64 scalar predictions; exposed VALID descriptive only. No autonomous control, architecture novelty or superiority admission."}


class Audit:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = None
        self.start = None
        self.rss = 0
        self.readout_attempts = self.readout_returns = self.readout_rows = 0
        self.readout_seconds = 0.0
        self.pending = None
        self.maximum_numpy_difference = self.maximum_torch_difference = 0.0

    def check(self):
        require(self.clock.now_ns() < self.deadline, "saved audit suspend-inclusive deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)
        self.rss = max(self.rss, rss)
        require(self.rss <= LIMITS["rss_bytes"], "saved audit peak RSS")
        require(sum(p.stat().st_size for p in self.out.iterdir() if p.is_file()) <= LIMITS["output_bytes"], "saved audit disk cap")

    def arrays(self, path):
        self.check()
        path = Path(path)
        path = path if path.is_absolute() else ROOT/path
        with self.np.load(path, allow_pickle=False) as archive:
            result = {name: archive[name] for name in archive.files}
        self.check()
        return result

    def predict(self, frozen, centered, positions, sensing, context):
        self.check()
        require(self.readout_attempts < 6300 and self.pending is None, "fixed replay allocation")
        self.readout_attempts += 1
        self.pending = {"id": self.readout_attempts, **context, "rows": len(centered)}
        self.emit_readout({**self.pending, "event": "attempt"})
        start = self.clock.now_ns()
        result = frozen.normalized(centered, positions, sensing)
        elapsed = (self.clock.now_ns()-start)/1e9
        self.readout_returns += 1
        self.readout_rows += len(result)
        self.readout_seconds += elapsed
        self.emit_readout({**self.pending, "event": "return", "seconds": elapsed})
        self.pending = None
        self.check()
        return result

    def emit_readout(self, row):
        with (self.out/"readouts.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            stream.flush()

    def check_terminal(self, plan, worker, started, terminal):
        h, args = self.helper, self.args
        launch, request = started["launch"], started["request"]
        require(terminal["status"] == "completed" and terminal["returncode"] == 0
                and terminal["timed_out"] is False and terminal["group_absent"] is True
                and terminal["error"] is None and terminal["clock_error"] is None
                and terminal["cleanup"]["group_absent"] is True and terminal["cleanup"]["reaped"] is True
                and terminal["cleanup"]["errors"] == [], "successful original supervisor")
        for key in ("pid", "pgid", "parent_pid", "command", "cwd", "started_ns", "deadline_ns", "clock_backend",
                    "cap_seconds", "watchdog_sha256", "clock_source_sha256"):
            require(launch[key] == terminal[key], "original parent launch/terminal identity")
        require(request == {"plan": str(args.plan), "plan_sha256": args.plan_sha256,
                            "output": str(args.run), "supervision": request["supervision"]}, "original absolute request")
        require(h.digest(Path(request["supervision"]), self.check)["sha256"] == worker["supervision_sha256"]
                and h.read(request["supervision"]) == launch, "parent launch bytes")
        command = list(launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [plan["python_executable"], str(ROOT/RUNNER), "--plan", str(args.plan),
                            "--plan-sha256", args.plan_sha256, "--output", str(args.run),
                            "--supervision", request["supervision"]], "exact producer command")
        require(launch["cwd"] == str(ROOT) and launch["pid"] == launch["pgid"] != launch["parent_pid"]
                and all(type(launch[k]) is int and launch[k] > 0 for k in ("pid", "pgid", "parent_pid")), "process group identity")
        require(worker["clock_backend"] == launch["clock_backend"] in ("mach_continuous_time", "CLOCK_BOOTTIME")
                and launch["clock_source_sha256"] == CLOCK_PIN
                and launch["watchdog_sha256"] == plan["sources"]["scripts/supervise_dialogue_observation_v2.py"]
                and launch["cap_seconds"] == plan["limits"]["native_seconds"] == 7200
                and launch["deadline_ns"] == launch["started_ns"]+7200*10**9
                and launch["started_ns"] <= worker["started_ns"] <= worker["finished_ns"]
                <= terminal["finished_ns"] < launch["deadline_ns"]
                and started["started_ns"] == worker["started_ns"], "native timing enclosure")
        require(worker["wall_seconds"] == (worker["finished_ns"]-worker["started_ns"])/1e9
                and terminal["elapsed_ns"] == terminal["finished_ns"]-launch["started_ns"]
                and terminal["wall_seconds"] == terminal["elapsed_ns"]/1e9, "recorded native elapsed arithmetic")

    def authenticate(self):
        require_threads()
        h, args = self.helper, self.args
        require(h.digest(args.plan, self.check)["sha256"] == args.plan_sha256, "external prospective plan pin")
        plan = h.read(args.plan)
        require(plan["version"] == "otto-spatial-study-v1" and plan["status"] == "frozen_before_execution"
                and plan["independent_audit_limits"] == LIMITS, "fixed spatial study and audit allocation")
        own = str(Path(__file__).resolve().relative_to(ROOT))
        require(plan["sources"][own] == h.digest(Path(__file__), self.check)["sha256"]
                and "tests/test_audit_otto_spatial_study.py" in plan["sources"]
                and plan["sources"][MODEL] == MODEL_PIN and plan["sources"][HELPER] == HELPER_PIN
                and plan["sources"][CLOCK] == CLOCK_PIN, "qualified inference and independent prospective audit pins")
        for name, pin in plan["sources"].items():
            require(h.digest(ROOT/name, self.check)["sha256"] == pin, "complete source closure")
        require(h.digest(args.terminal, self.check)["sha256"] == args.terminal_sha256, "external original parent pin")
        worker = h.manifest(args.run, args.receipt_sha256, payload_names(), self.check)
        require(worker["version"] == plan["version"] and worker["sources"] == plan["sources"]
                and worker["inputs"] == plan["inputs"] and worker["limits"] == plan["limits"]
                and worker["plan_sha256"] == args.plan_sha256 and worker["completed_fits"] == 15
                and worker["pending"] == [] and worker["parity_passed"] is True
                and worker["initial_pairing_passed"] is True and worker["mode"] == plan["mode"] == "study"
                and worker["qualification"] == plan["qualification"] and plan["expected_calls"] == COUNTS
                and all(worker[k] == 0 for k in ("external_model_calls", "native_steps", "native_resets")),
                "complete fifteen-fit worker before scientific arrays")
        require(type(worker["peak_rss_bytes"]) is int and 0 < worker["peak_rss_bytes"] <= plan["limits"]["rss_bytes"]
                and sum(p.stat().st_size for p in args.run.iterdir()) <= plan["limits"]["output_bytes"], "worker resource bounds")
        self.check_terminal(plan, worker, h.read(args.run/"started.json"), h.read(args.terminal))
        # This is the only producer reuse: metadata/source/runtime lineage,
        # never its arrays, training, prediction, metrics or aggregation.
        producer = load(ROOT/RUNNER, "_spatial_saved_audit_lineage")
        require(producer.authenticate(argparse.Namespace(plan=args.plan, plan_sha256=args.plan_sha256)) == plan,
                "prior capacity/cache and synthetic qualification closures")
        runtime = h.read(args.run/"runtime.json")
        h.close(runtime, {"python": sys.version, "executable": sys.executable,
                          "threads": {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                          "torch_threads": 1, "torch_interop_threads": 1, "torch_deterministic": True,
                          "model_source_sha256": MODEL_PIN}, "recorded single-CPU runtime")
        self.plan, self.worker = plan, worker

    def compute(self):
        import numpy as np

        self.np = np
        h, run = self.helper, self.args.run
        model = load(ROOT/MODEL, "_spatial_saved_audit_qualified_numpy")
        data, evidence = {}, {}
        for split in ROWS:
            full = self.arrays(self.plan["inputs"][split+"_data"]["path"])
            rows_path = Path(self.plan["inputs"][split+"_rows"]["path"])
            rows = list(h.lines(rows_path if rows_path.is_absolute() else ROOT/rows_path, self.check))
            h.validate_cache(full, rows, split, np, self.check)
            episodes = [[r["episode_id"], r["regime"], r["seed"], r["initial_hit"], r["total_steps"]]
                        for r in rows if r["prefix_index"] == 0]
            evidence[split] = {"rows": len(rows), "episodes": episodes,
                               "array_sha256": {key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                                                for key, value in full.items()},
                               "metadata_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            data[split] = {key: value for key, value in full.items() if key != "features"}
            del full, rows
        c0 = float(np.float32(np.mean(data["train"]["target"], dtype=np.float64)))
        preparation = h.read(run/"preparation.json")
        seconds = preparation["preparation_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, "paid preparation time")
        h.close(preparation, {"rows": dict(ROWS), "episodes": {"train": 192, "valid": 48}, "c0_float32": c0,
                              "datasets": evidence, "preparation_seconds": seconds,
                              "training_view": "lossless centered float64 then float32 cast",
                              "diagnostic_view": "original centered float64; no cached-float32 upcast", "alias_floor": None},
                "independent complete original data preparation")
        streams = {name: iter(h.lines(run/name, self.check)) for name in ("initializations.jsonl", "epoch-orders.jsonl",
                   "updates.jsonl", "fit-curves.jsonl", "fits.jsonl", "parity.jsonl")}
        work, fits, paired = h.Work(h.lines(run/"work.jsonl", self.check)), [], {}
        for seed in SEEDS:
            for kind in KINDS:
                self.check()
                fit_id, context = f"{kind}@{seed}", {"fit_id": f"{kind}@{seed}"}
                fit = next(streams["fits.jsonl"])
                require((fit["fit_id"], fit["kind"], fit["seed"], fit["epochs"], fit["training_rows"],
                         fit["validation_rows"], fit["updates"]) == (fit_id, kind, seed, 80, 5589, 1109, 3520), "fixed final fit identity")
                initial_path, final_path = run/f"initial-{kind}-{seed}.npz", run/f"final-{kind}-{seed}.npz"
                initial = self.arrays(initial_path)
                count = checkpoint(initial, kind, c0, np, initial=True)
                tensor_sha = {name: hashlib.sha256(initial[name].tobytes(order="C")).hexdigest()
                              for name in (*shapes_for(kind), "c0")}
                if kind == "spatial":
                    paired[seed] = tensor_sha
                elif kind == "neighbor_free":
                    require(tensor_sha == paired[seed], "exact spatial/neighbor-free initial tensors")
                h.close(next(streams["initializations.jsonl"]), {"fit_id": fit_id, "kind": kind, "seed": seed,
                        "initial_sha256": self.worker["files"][initial_path.name]["sha256"], "tensor_sha256": tensor_sha,
                        "c0_float32": c0, "zero_final_and_biases": True,
                        "paired_spatial_tensors": True if kind == "neighbor_free" else None,
                        "parameter_count": count}, "initial tensors and common function")
                fit_call_seconds = sum(work.take(channel, context) for channel in
                                       ("model_initialization", "checkpoint_export", "optimizer_initialization"))
                training_call_seconds = 0.0
                rng = np.random.default_rng(seed+20000)
                for epoch in range(1, 81):
                    order = rng.permutation(5589).astype(np.int64, copy=False)
                    training_call_seconds += epoch_records(fit_id, epoch, order, next(streams["epoch-orders.jsonl"]),
                                                           streams["updates.jsonl"], next(streams["fit-curves.jsonl"]), work, np, h)
                fit_call_seconds += training_call_seconds + work.take("checkpoint_export", context)
                final = self.arrays(final_path)
                require(checkpoint(final, kind, c0, np) == count, "unchanged final model shape")
                require(set(fit) == {"fit_id", "kind", "seed", "epochs", "training_rows", "validation_rows", "updates",
                        "optimizer_steps_before", "optimizer_steps", "parameter_count", "c0_float32", "initial_sha256",
                        "checkpoint_sha256", "prediction_sha256", "training_array_sha256", "row_order_sha256", "metrics",
                        "training_seconds", "fit_seconds", "diagnostic_seconds", "parity_rows", "maximum_parity_error",
                        "parity_passed", "storage", "deployment_setup_seconds", "prediction_seconds"}, "complete fit record fields")
                require(fit["optimizer_steps_before"] == {} and fit["optimizer_steps"] == dict.fromkeys(shapes_for(kind), 3520)
                        and fit["parameter_count"] == count and fit["c0_float32"] == c0
                        and fit["initial_sha256"] == self.worker["files"][initial_path.name]["sha256"]
                        and fit["checkpoint_sha256"] == self.worker["files"][final_path.name]["sha256"]
                        and fit["training_array_sha256"] == evidence["train"]["array_sha256"]
                        and fit["row_order_sha256"] == evidence["train"]["metadata_sha256"], "checkpoint/data/optimizer joins")
                for key in ("fit_seconds", "training_seconds", "diagnostic_seconds", "deployment_setup_seconds", "prediction_seconds"):
                    require(type(fit[key]) in (int, float) and math.isfinite(fit[key]) and fit[key] >= 0, "finite paid fit costs")
                require(training_call_seconds <= fit["training_seconds"] <= fit["fit_seconds"]
                        and fit_call_seconds <= fit["fit_seconds"], "nested recorded training cost scopes")
                diagnostic_calls = work.take("deployment_construction", context)+work.take("parity_restore", context)
                deployment_calls = diagnostic_calls
                require(deployment_calls <= fit["deployment_setup_seconds"], "complete deployment construction costs")
                h.close(fit["diagnostic_seconds"], fit["deployment_setup_seconds"]+fit["prediction_seconds"], "disjoint diagnostic partition")
                frozen = model.FrozenValue(final)
                storage = {"parameter_array_bytes": 8*count, "baseline_array_bytes": 8, "mutable_array_bytes": 0,
                           "scope": "Owned immutable float64 parameters and baseline; excludes scalar metadata, inputs, and temporary geometry/feature/convolution workspace."}
                h.close(fit["storage"], storage, "independent immutable deployment storage")
                saved = self.arrays(run/f"predictions-{kind}-{seed}.npz")
                require(set(saved) == {split+"_"+backend for split in ROWS for backend in ("numpy", "torch64")}
                        and fit["prediction_sha256"] == self.worker["files"][f"predictions-{kind}-{seed}.npz"]["sha256"],
                        "both full saved backends")
                metrics, maximum, seen = {}, 0.0, 0
                for split, n in ROWS.items():
                    actual, reference = saved[split+"_numpy"], saved[split+"_torch64"]
                    require(actual.shape == reference.shape == (n,), "all final prediction rows")
                    compare_values(actual, reference, np)
                    current = data[split]
                    for offset in range(0, n, 16):
                        stop = min(offset+16, n)
                        batch_context = {**context, "split": split, "offset": offset}
                        diagnostic_calls += work.take("numpy_prediction", batch_context)
                        diagnostic_calls += work.take("torch_prediction", batch_context)
                        centered = center_batch(current["beliefs"][offset:stop], current["positions"][offset:stop], np)
                        q, sensing = current["positions"][offset:stop], current["sensing_length"][offset:stop]
                        expected_inputs = {key: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                                           for key, value in (("centered", centered), ("positions", q), ("sensing_length", sensing))}
                        delta = float(np.max(np.abs(actual[offset:stop]-reference[offset:stop])))
                        h.close(next(streams["parity.jsonl"]), {**batch_context, "rows": stop-offset,
                                "input_sha256": expected_inputs, "maximum_absolute_difference": delta, "passed": True},
                                "every original-float64 batch parity witness", atol=0.0, rtol=0.0)
                        replay = self.predict(frozen, centered, q, sensing, batch_context)
                        self.maximum_numpy_difference = max(self.maximum_numpy_difference, compare_values(actual[offset:stop], replay, np))
                        self.maximum_torch_difference = max(self.maximum_torch_difference, compare_values(replay, reference[offset:stop], np))
                        maximum, seen = max(maximum, delta), seen+stop-offset
                    metrics[split] = h.scalar_metrics(actual, current["target"], np)
                require(fit["parity_rows"] == seen == 6698 and fit["parity_passed"] is True
                        and fit["maximum_parity_error"] == maximum
                        and diagnostic_calls <= fit["diagnostic_seconds"]
                        and diagnostic_calls-deployment_calls <= fit["prediction_seconds"], "all parity rows and complete diagnostic cost")
                h.close(fit["metrics"], metrics, "independent final scalar statistics")
                fits.append({**fit, "metrics": metrics})
                del initial, final, frozen, saved
        for stream in streams.values():
            require(next(stream, None) is None, "no omitted or extra scientific rows")
        require(next(work.rows, None) is None and work.sequence == 171090
                and {k: v["attempted"] for k, v in work.counts.items()} == COUNTS
                and {k: v["returned"] for k, v in work.counts.items()} == COUNTS, "all 171090 returned producer operations")
        h.close(self.worker["calls"], work.counts, "all attempted/returned work counts and costs")
        require(seconds+math.fsum(f["fit_seconds"]+f["diagnostic_seconds"] for f in fits) <= self.worker["wall_seconds"],
                "disjoint preparation/fit/diagnostic time within worker")
        expected = {**aggregate(fits), "preparation_seconds": seconds, "calls": work.counts, "training_updates": 52800}
        h.close(h.read(run/"summary.json"), expected, "all fifteen fits/five equal-seed means and no scalar gate")
        require(self.readout_attempts == self.readout_returns == 6300 and self.readout_rows == 100470
                and self.pending is None, "exact independent replay work")
        return {**expected, "audit_version": VERSION, "agreement": True, "scope": SCOPE,
                "reconstructed_cache": evidence, "independent_readout_attempts": self.readout_attempts,
                "independent_readout_returns": self.readout_returns, "independent_readout_rows": self.readout_rows,
                "independent_readout_seconds": self.readout_seconds, "maximum_numpy_difference": self.maximum_numpy_difference,
                "maximum_torch_difference": self.maximum_torch_difference}

    def execute(self):
        require(self.out.is_absolute() and not self.out.exists()
                and not any(p.is_symlink() for p in self.out.parents), "exclusive absolute audit output")
        self.out.mkdir(parents=True, exist_ok=False)
        try:
            require(file_sha(ROOT/CLOCK) == CLOCK_PIN and file_sha(ROOT/HELPER) == HELPER_PIN, "pinned pure audit helpers")
            self.clock = load(ROOT/CLOCK, "_spatial_study_audit_clock").SuspendClock()
            self.start = self.clock.now_ns()
            self.deadline = self.start+LIMITS["seconds"]*10**9
            signal.signal(signal.SIGALRM, alarm)
            signal.setitimer(signal.ITIMER_REAL, LIMITS["seconds"])
            self.helper = load(ROOT/HELPER, "_spatial_study_independent_cache_audit")
            h = self.helper
            h.write(self.out/"started.json", {"version": VERSION, "request": {k: str(v) for k, v in vars(self.args).items()},
                    "source": h.digest(Path(__file__), self.check), "limits": LIMITS, "clock_backend": self.clock.backend,
                    "started_ns": self.start, "deadline_ns": self.deadline, "scope": SCOPE,
                    "requested_threads": THREADS})
            self.authenticate()
            summary = self.compute()
            self.authenticate()
            h.write(self.out/"summary.json", summary)
            self.check()
            finish = self.clock.now_ns()
            require(finish < self.deadline, "strict final audit deadline")
            receipt = {"version": VERSION, "status": "completed", "agreement": True,
                       "source": h.digest(Path(__file__), self.check), "plan_sha256": self.args.plan_sha256,
                       "worker_sha256": self.args.receipt_sha256, "terminal_sha256": self.args.terminal_sha256,
                       "producer_source_sha256": self.plan["sources"][RUNNER], "qualified_numpy_source_sha256": MODEL_PIN,
                       "independent_helper_sha256": HELPER_PIN, "limits": LIMITS, "threads": THREADS, "clock_backend": self.clock.backend,
                       "started_ns": self.start, "finished_ns": finish, "wall_seconds": (finish-self.start)/1e9,
                       "peak_rss_bytes": self.rss, "independent_readout_attempts": self.readout_attempts,
                       "independent_readout_returns": self.readout_returns, "independent_readout_rows": self.readout_rows,
                       "independent_readout_seconds": self.readout_seconds, "pending": self.pending,
                       "optimizer_calls": 0, "native_steps": 0, "external_model_calls": 0, "scope": SCOPE,
                       "files": {name: h.digest(self.out/name, self.check) for name in ("started.json", "readouts.jsonl", "summary.json")}}
            h.write(self.out/"receipt.json", receipt)
            self.check()
            signal.setitimer(signal.ITIMER_REAL, 0)
            return receipt
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            failure = {"version": VERSION, "status": "failed", "agreement": False, "error": repr(error), "scope": SCOPE,
                       "independent_readout_attempts": self.readout_attempts, "independent_readout_returns": self.readout_returns,
                       "independent_readout_rows": self.readout_rows, "independent_readout_seconds": self.readout_seconds,
                       "pending": self.pending, "started_ns": self.start, "finished_ns": None, "wall_seconds": None}
            try:
                if (self.out/"receipt.json").exists():
                    (self.out/"receipt.json").rename(self.out/"invalid-completed-receipt.json")
                with (self.out/"receipt.json").open("x") as stream:
                    json.dump(failure, stream, indent=2, sort_keys=True, allow_nan=False)
                    stream.write("\n")
            except BaseException as secondary:  # noqa: BLE001 - preserve original failure
                error.add_note(f"Failure publication: {secondary!r}")
            raise


def alarm(*_):
    raise TimeoutError("saved spatial study audit deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "run", "terminal", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    for key in ("plan-sha256", "receipt-sha256", "terminal-sha256"):
        parser.add_argument("--"+key, required=True)
    receipt = Audit(parser.parse_args()).execute()
    print(json.dumps({"status": receipt["status"], "agreement": receipt["agreement"],
                      "independent_readout_returns": receipt["independent_readout_returns"]}), flush=True)


if __name__ == "__main__":
    main()
