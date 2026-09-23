"""Fixed synthetic full-history capacity screen; no empirical or teacher inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/qualify_otto_separate_prior_capacity.py"
TEST = "tests/test_qualify_otto_separate_prior_capacity.py"
TRAINER = "scripts/train_otto_separate_prior.py"
TRAINER_TEST = "tests/test_train_otto_separate_prior.py"
MODEL = "src/openjev/research/otto_prequery_scores.py"
MODEL_TEST = "tests/test_otto_prequery_scores.py"
SEPARATE_MODEL = "src/openjev/research/otto_separate_prior_scores.py"
SEPARATE_TEST = "tests/test_otto_separate_prior_scores.py"
SEPARATE_PIN = "086348b0db778308b152fd3abf29b645d258d14e12abc12a6b879371081b7575"
SEPARATE_TEST_PIN = "c8dd82079a16b91a496b03ac47bdb8203c75aa9c0ccb11489a6a922b99e55748"
SAMPLING = "src/openjev/research/otto_sampled_forecast_data.py"
DATA = "src/openjev/research/otto_score_forecast_data.py"
CHRONOLOGY = "src/openjev/research/otto_prequery_data.py"
LOSS = "src/openjev/research/otto_prequery_loss.py"
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
PROTOCOL = "research/otto-separate-prior-protocol.md"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
MODEL_PIN = "6dfff08284004a134abf47b4ab4dc425f5d9c96f63c58475007295a666b24f4e"
MODEL_TEST_PIN = "fbc18300688c75e7b872afa2f8fd1fd1fb7f87f69be5cda47ad7cde1b29af63a"
COMPONENTS = {SELF, TEST, TRAINER, TRAINER_TEST, MODEL, MODEL_TEST, SEPARATE_MODEL, SEPARATE_TEST, CHRONOLOGY, LOSS,
              "tests/test_otto_prequery_data.py", "tests/test_otto_prequery_loss.py"}
SOURCES = COMPONENTS | {SAMPLING, DATA, CLOCK, SUPERVISOR, PROTOCOL,
           "scripts/train_otto_cross_query_forecasts.py", "src/openjev/research/otto_separate_prior_metrics.py",
           "src/openjev/research/otto_prequery_metrics.py",
           "scripts/train_otto_prequery_calibration.py", "scripts/qualify_otto_prequery_capacity.py",
           "src/openjev/research/otto_cross_query_metrics.py",
           "src/openjev/research/otto_cross_query_scores.py", "src/openjev/research/otto_cross_query_data.py",
           "src/openjev/__init__.py", "src/openjev/research/__init__.py"}
VERSION = "otto-separate-prior-capacity-v1"
KINDS = ("innovation_shared_mse", "innovation_shared_aux", "innovation_separate_mse", "innovation_separate_aux",
         "gru_shared_mse", "gru_shared_aux", "gru_separate_mse", "gru_separate_aux")
CELLS = {"innovation_shared_mse": ("innovation", "shared", "mse"),
         "innovation_shared_aux": ("innovation", "shared", "query_aux"),
         "innovation_separate_mse": ("innovation", "separate", "mse"),
         "innovation_separate_aux": ("innovation", "separate", "query_aux"),
         "gru_shared_mse": ("innovation_gru", "shared", "mse"),
         "gru_shared_aux": ("innovation_gru", "shared", "query_aux"),
         "gru_separate_mse": ("innovation_gru", "separate", "mse"),
         "gru_separate_aux": ("innovation_gru", "separate", "query_aux")}
LIMITS = {"seconds": 180, "rss_bytes": 4 * 1024**3, "output_bytes": 128 * 1024**2}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
CONFIG = {"families": list(KINDS), "batch": 6, "length": 2188, "chunk": 32, "period": 4,
          "selected_windows_per_episode": 8, "episode_denominator": 54,
          "feature_seed": 0x53514331, "model_seed": 0x53514D31,
          "selection_seeds": [0x53515700 + i for i in range(6)],
          "learning_rate": .003, "clip_norm": 5., "weight_decay": 0., "scale": 64.,
          "fit_seeds": 3, "updates_per_fit": 720, "projection_multiplier": 1.5,
          "fixed_overhead_seconds": 180., "admission_seconds": 16200., "training_cap_seconds": 21600,
          "zero_target_chunks": "forward without autograd, still charged and counted",
          "cells": {k: {"architecture": v[0], "readout": v[1], "objective": v[2]} for k, v in CELLS.items()},
          "prior_rows_per_episode": 546, "prior_weight": "1/(54*546)", "prior_coefficient": 1.,
          "scope": "synthetic throughput planning only; no performance or scientific result"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained input")
    return path


def descriptor(path):
    path = regular(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def read(path):
    return json.loads(regular(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime():
    return {"python": sys.version, "executable": sys.executable,
            "torch": importlib.metadata.version("torch"), "numpy": importlib.metadata.version("numpy")}


def engineering(value):
    require(descriptor(value["path"]) == {k: value[k] for k in ("sha256", "bytes")}, "engineering pin")
    receipt = read(value["path"])
    expected = {name: descriptor(name)["sha256"] for name in COMPONENTS}
    require(receipt["status"] == "passed" and receipt["sources_before"] == receipt["sources_after"]
            and all(receipt["sources_after"].get(k) == v for k, v in expected.items())
            and receipt["results"] and all(type(r["exit_code"]) is int and r["exit_code"] == 0 for r in receipt["results"]),
            "reviewed and qualified current capacity/training components")
    parent = regular(value["path"]).parent
    require({p.name for p in parent.iterdir()} == set(receipt["files"]) | {"receipt.json"}, "engineering closure")
    for name, pin in receipt["files"].items():
        require(Path(name).name == name and descriptor(parent / name) == pin, "engineering payload")


def source_pins():
    pins = {name: descriptor(name)["sha256"] for name in SOURCES}
    require(pins[MODEL] == MODEL_PIN and pins[MODEL_TEST] == MODEL_TEST_PIN
            and pins[SEPARATE_MODEL] == SEPARATE_PIN and pins[SEPARATE_TEST] == SEPARATE_TEST_PIN
            and pins["src/openjev/research/otto_cross_query_scores.py"] ==
                "799979c0c60460352df076750d15b2cb9a5db6b6976707605dcca7c90fa76e08"
            and pins[CLOCK] == CLOCK_PIN and pins[SUPERVISOR] == SUPERVISOR_PIN, "fixed qualified sources")
    return pins


def freeze(args):
    item = {"path": str(regular(args.engineering)), **descriptor(args.engineering)}
    require(item["sha256"] == args.engineering_sha256, "external engineering receipt")
    engineering(item)
    value = {"version": VERSION, "status": "frozen_before_synthetic_work", "configuration": CONFIG,
             "limits": LIMITS, "sources": source_pins(), "engineering": item, "runtime": runtime()}
    write(args.output, value)
    print(json.dumps({"plan": str(args.output), **descriptor(args.output)}), flush=True)


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.start = self.plan = None
        self.sequence = 0
        self.receipt = {"version": VERSION, "status": "started", "pending": None, "pending_emission": None,
                        "completed_families": [], "optimizer_updates": 0, "new_teacher_calls": 0,
                        "new_native_calls": 0, "empirical_payloads_read": 0}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original synthetic deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "synthetic RSS/output cap")

    def event(self, record):
        self.check()
        self.receipt["pending_emission"] = record
        with (self.out / "work.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def call(self, kind, operation, function, chunk=None):
        self.check(); self.sequence += 1
        record = {"call_id": self.sequence, "family": kind, "operation": operation, "chunk": chunk}
        self.receipt["pending"] = record
        self.event({"event": "attempt", **record})
        tick = self.clock.now_ns()
        result = function()
        elapsed = (self.clock.now_ns() - tick) / 1e9
        self.event({"event": "return", **record, "seconds": elapsed})
        self.receipt["pending"] = None
        if operation == "optimizer_update":
            self.receipt["optimizer_updates"] += 1
        self.check()
        return result

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN, "clock before import")
        self.clock = load(CLOCK, "_separate_prior_capacity_clock").SuspendClock()
        self.start = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.start < 5 * 10**9, "supervisor launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid()
                and self.launch["cap_seconds"] == LIMITS["seconds"]
                and self.launch["watchdog_sha256"] == SUPERVISOR_PIN
                and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
                "original bounded synthetic process")
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external capacity plan")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_synthetic_work"
                and self.plan["configuration"] == CONFIG and self.plan["limits"] == LIMITS
                and self.plan["sources"] == source_pins() and self.plan["runtime"] == runtime(),
                "fixed synthetic plan and runtime")
        engineering(self.plan["engineering"])
        require(all(os.environ.get(name) == "1" for name in THREADS), "CPU1 numerical environment")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"],
                            engineering=self.plan["engineering"], limits=LIMITS,
                            supervision_sha256=descriptor(self.args.supervision)["sha256"])
        self.check(); write(self.out / "started.json", {"launch": self.launch, "started_ns": self.start})

    def synthetic(self, torch, sampling):
        """Six maximum-length synthetic histories with fixed local generators."""
        np = self.np
        generator = torch.Generator(device="cpu").manual_seed(CONFIG["feature_seed"])
        batch, total = CONFIG["batch"], CONFIG["length"]
        features = torch.rand(batch, total, 31, generator=generator, dtype=torch.float32) * .5
        step = torch.arange(total, dtype=torch.int64)
        features[:, :, 15] = (step.to(torch.float64) / 2188).to(torch.float32)
        features[:, :, 16] = ((step % 4).to(torch.float64) / 2188).to(torch.float32)
        features[:, :, 17] = 1
        legal = torch.ones(batch, total, 4, dtype=torch.bool)
        legal[:, ::7, 0] = False
        features[:, :, 2:6] = legal.to(torch.float32)
        raw = (32 + torch.rand(batch, total, 4, generator=generator, dtype=torch.float32) * 4).numpy()
        query_mask = np.zeros((batch, total), np.bool_); query_mask[:, ::4] = True
        query = np.zeros_like(raw); query[query_mask] = raw[query_mask]
        targets = np.zeros_like(raw)
        weights = np.zeros((batch, total), np.float64)
        prior_mask = query_mask.copy(); prior_mask[:, 0] = False
        prior_targets = np.zeros_like(raw); prior_targets[prior_mask] = raw[prior_mask]
        prior_weights = np.zeros_like(weights); prior_weights[prior_mask] = 1 / (54 * 546)
        selections = []
        for row, seed in enumerate(CONFIG["selection_seeds"]):
            selection = sampling.select_windows(total, seed)
            require(selection["selected_windows"] == 8, "eight synthetic loss windows")
            selections.append(selection)
            weight = (selection["population_windows"] / 8) / (54 * (total - selection["population_windows"]))
            for offset in selection["start_offsets"]:
                targets[row, offset + 1:offset + 4] = raw[row, offset + 1:offset + 4]
                weights[row, offset + 1:offset + 4] = weight
        arrays = {"features": features.numpy().reshape(-1, 31), "query_scores": query.reshape(-1, 4),
            "query_mask": query_mask.reshape(-1), "legal": legal.numpy().reshape(-1, 4),
            "targets": targets.reshape(-1, 4), "weights": weights.reshape(-1),
            "prior_targets": prior_targets.reshape(-1, 4), "prior_weights": prior_weights.reshape(-1),
            "prior_mask": prior_mask.reshape(-1), "episode_offsets": np.arange(batch + 1, dtype=np.int64) * total,
            "actions": np.ones(batch * total, dtype=np.int64)}
        metadata = {name: {"shape": list(v.shape), "dtype": str(v.dtype),
                           "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for name, v in arrays.items()}
        self.check(); write(self.out / "synthetic.json", {"configuration": CONFIG, "selections": selections,
            "arrays": metadata, "selected_nonquery_rows": int((weights > 0).sum()),
            "prior_rows": int(prior_mask.sum()),
            "formula": "local fixed generators; complete query scores; sampled nonquery and complete later-query targets"})
        data = {"version": self.data.VERSION, **arrays,
            "episode_ids": tuple(f"synthetic-{i}" for i in range(batch)),
            "episode_regimes": ("synthetic",) * batch, "episode_splits": ("synthetic",) * batch}
        return data, metadata

    def one_family(self, kind, data):
        torch = self.torch
        architecture, readout, objective = CELLS[kind]
        module = self.trainer.model_provider(kind, self.models, self.separate_models)
        model = self.call(kind, "initialization", lambda: module.make_head(architecture, CONFIG["model_seed"]))
        initial = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = self.call(kind, "optimizer_initialization", lambda: torch.optim.Adam(
            model.parameters(), lr=.003, weight_decay=0.))
        tick = self.clock.now_ns()

        def stage(name, offset):
            self.check()
            self.receipt["pending"]["stage"] = name
            self.receipt["pending"]["chunk_start"] = offset
            self.event({"event": "stage", "family": kind, "operation": name, "chunk": offset})

        result = self.call(kind, "batch_update", lambda: self.trainer.batch_update(
            torch, module, self.data, self.losses, model, optimizer, data, list(range(6)),
            objective=objective, check=self.check, stage=stage))
        self.receipt["optimizer_updates"] += 1
        require(result["forward_chunks"] == 69 and result["forward_rows"] == 6 * 2188
                and result["prior_rows"] == 6 * 546 and result["optimizer_step"] == 1,
                "complete maximum-history synthetic batch")
        if objective == "query_aux":
            require(result["backward_chunks"] == 69, "all auxiliary query-bearing chunks train")
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                "finite materialized gradient for every parameter")
        require(all(bool(torch.isfinite(p).all()) and float(optimizer.state[p]["step"]) == 1.
                    for p in model.parameters()), "one finite Adam step per parameter")
        changed = [name for name, parameter in model.named_parameters() if not torch.equal(parameter, initial[name])]
        require(any(name.startswith("output.") for name in changed), "synthetic optimizer changed the output head")
        gradient_norms = {name: float(torch.linalg.vector_norm(p.grad)) for name, p in model.named_parameters()}
        seconds = (self.clock.now_ns() - tick) / 1e9
        row = {"family": kind, "architecture": architecture, "readout": readout, "objective": objective,
            "parameter_count": module.parameter_count(architecture), "batch_seconds": seconds,
            **result, "chunks": result["forward_chunks"], "optimizer_updates": 1,
            "changed_parameters": changed, "parameter_gradient_norms_after_clip": gradient_norms,
            "timing_scope": "exact production batch_update plus guards, durable stage/return logging and final checks; initialization excluded"}
        self.event({"event": "family_complete", **row})
        self.receipt["completed_families"].append(kind)
        return row

    def body(self):
        import numpy as np
        import torch

        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        self.torch, self.np = torch, np
        sys.path.insert(0, str(ROOT / "src"))
        self.models = load(MODEL, "_separate_prior_capacity_models")
        self.separate_models = load(SEPARATE_MODEL, "_separate_prior_capacity_separate_models")
        self.data = load(CHRONOLOGY, "_separate_prior_capacity_data")
        self.losses = load(LOSS, "_separate_prior_capacity_loss")
        self.trainer = load(TRAINER, "_separate_prior_capacity_trainer")
        sampling = load(SAMPLING, "_separate_prior_capacity_sampling")
        require(self.models.KINDS == self.separate_models.KINDS == ("innovation", "innovation_gru") and self.trainer.KINDS == KINDS
                and self.trainer.CELLS == CELLS and self.trainer.LIMITS["seconds"] == 21600
                and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
                "fixed cells, production batch and CPU1")
        self.check(); write(self.out / "runtime.json", {**runtime(), "torch_import": torch.__version__,
            "numpy_import": np.__version__, "threads": 1, "interop_threads": 1, "deterministic": True,
            "cuda_used": False, "mps_used": False})
        data, metadata = self.call(None, "synthetic_inputs", lambda: self.synthetic(torch, sampling))
        rows = [self.one_family(kind, data) for kind in KINDS]
        for name, descriptor in metadata.items():
            require(hashlib.sha256(data[name].tobytes()).hexdigest() == descriptor["sha256"], "synthetic inputs unchanged")
        total = math.fsum(row["batch_seconds"] for row in rows)
        projected = 1.5 * 3 * 720 * total + 180
        summary = {"version": VERSION, "technical_complete": True, "families": rows,
            "sum_batch_seconds": total, "projected_seconds": projected, "threshold_seconds": 16200.,
            "admitted": projected <= 16200., "formula": "1.5 * 3 * 720 * sum(batch_seconds) + 180 <= 16200",
            "scope": "synthetic throughput planning; auxiliary costs measured independently; no efficacy or timing guarantee",
            "all_eight_cells_required": True, "new_teacher_calls": 0, "new_native_calls": 0,
            "empirical_payloads_read": 0, "optimizer_updates": 8}
        self.check(); write(self.out / "summary.json", summary)
        self.receipt["admitted"] = summary["admitted"]

    def execute(self):
        self.out.mkdir(exist_ok=False)

        def interrupted(_signum, _frame):
            raise InterruptedError("original capacity supervisor stopped worker")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.admit(); self.body()
            self.check(); require(source_pins() == self.plan["sources"], "unchanged capacity sources")
            engineering(self.plan["engineering"])
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"],
                    "unchanged admitted plan and original launch")
            require(self.receipt["completed_families"] == list(KINDS) and self.receipt["optimizer_updates"] == 8
                    and self.receipt["pending"] is self.receipt["pending_emission"] is None, "complete synthetic work")
            require({p.name for p in self.out.iterdir()} == {
                "started.json", "runtime.json", "synthetic.json", "work.jsonl", "summary.json"},
                "exact five synthetic payloads")
            files = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
            self.check(); finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, started_ns=self.start,
                finished_ns=finished, wall_seconds=(finished - self.start) / 1e9,
                requires_successful_original_supervisor=True)
            write(self.out / "receipt.json", self.receipt); self.check()
            print(json.dumps({"status": "completed", "admitted": self.receipt["admitted"],
                              "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            try:
                self.receipt.update(status="failed", complete=False, admitted=False, error=repr(error),
                                    traceback=traceback.format_exc())
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - preserve original work or publication failure
                error.add_note("Failure receipt publication also failed: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--engineering", type=Path, required=True)
    plan.add_argument("--engineering-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path.cwd() == ROOT and Path(sys.executable).absolute() == ROOT / ".venv/bin/python",
            "original general interpreter and checkout")
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), "exclusive contained absolute output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute execution inputs")
        Run(args).execute()


if __name__ == "__main__":
    main()
