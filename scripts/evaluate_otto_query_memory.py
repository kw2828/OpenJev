"""Conditional frozen-checkpoint TEST evaluation at P4/P8, with no fitting.

Standard-library admission authenticates the original closed TRAIN/DEV producer,
its independent DEV audit and that audit's original successful supervisor before
any TEST or checkpoint array is decoded. A producer gate is always provisional.
"""
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
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/evaluate_otto_query_memory.py"
TEST = "tests/test_evaluate_otto_query_memory.py"
PRODUCER = "scripts/train_otto_query_memory.py"
DEV_AUDITOR = "scripts/audit_otto_query_memory.py"
NEW_COMPONENTS = {SELF, TEST, "scripts/audit_otto_query_memory_test.py", "tests/test_audit_otto_query_memory_test.py"}
CLOCK = "src/openjev/research/suspend_clock.py"
SUPERVISOR = "scripts/supervise_dialogue_observation_v2.py"
CLOCK_PIN = "cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124"
SUPERVISOR_PIN = "610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144"
VERSION = "otto-query-memory-test-v1"
SEEDS = (309000001, 309000002, 309000003)
FITS = ("pretrained", "joint_aux", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled")
VIEWS = ("pretrained", "joint_aux", "last_error", "instant_delta", "trace_delta", "trace_additive", "trace_scrambled", "trace_no_write")
PERIODS = (4, 8)
LIMITS = {"seconds": 1800, "rss_bytes": 4 * 1024**3, "output_bytes": 2 * 1024**3}
CONFIG = {"fit_seeds": list(SEEDS), "views": list(VIEWS), "query_periods": list(PERIODS), "test_episodes": 36,
          "evaluation_views": 48, "checkpoint_decodes": 18, "test_array_decodes": 1,
          "batch_episodes": 6, "chunk": 32, "optimizer_calls": 0, "teacher_calls": 0, "native_calls": 0,
          "train_array_decodes": 0, "dev_array_decodes": 0, "device": "cpu", "dtype": "float32",
          "checkpoint": "all fixed final fits already independently audited at DEV",
          "evaluation": "frozen slow forward, required residual/projection flags True, outer no_grad",
          "scope": "fixed collector paths with actual P4/P8 observation schedules; no autonomous result"}
ROLES = ("training_plan", "training_receipt", "training_terminal", "dev_audit_receipt", "dev_audit_terminal", "engineering")
PREDICTION_FIELDS = ("action_prediction", "slow_action_prediction", "base_prediction", "shadow_prior",
                     "corrected_shadow_prior", "prior_mask", "prewrite_correction", "episode_offsets")
PAYLOADS = {"started.json", "runtime.json", "work.jsonl", "test-views.json", "summary.json", "checkpoint-manifest.json"}
PAYLOADS |= {f"test-P{p}-history.{extension}" for p in PERIODS for extension in ("json", "npz")}
PAYLOADS |= {f"test-P{p}-prediction-{view}-{seed}.npz" for p in PERIODS for seed in SEEDS for view in VIEWS}
ZERO_COUNTS = ("optimizer_calls", "teacher_calls", "native_calls", "train_array_decodes", "dev_array_decodes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and path.is_relative_to(ROOT) and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)), "regular contained evidence")
    return path


def descriptor(value):
    path, digest = regular(value), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def read(value):
    return json.loads(regular(value).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticate_engineering(path, sources):
    engineering = read(path)
    require(engineering["status"] == "passed" and engineering["source_before"] == engineering["source_after"]
            and NEW_COMPONENTS <= set(engineering["source_after"])
            and isinstance(engineering["commands"], list) and engineering["commands"]
            and all(type(c["returncode"]) is int and c["returncode"] == 0 and c["timed_out"] is False
                    and c["reaped"] is True for c in engineering["commands"]), "qualified new TEST producer and auditor")
    for name, expected in engineering["source_after"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and set(expected) == {"sha256", "bytes"} and descriptor(name) == expected,
                "qualified TEST source descriptor")
        require(name not in sources or sources[name] == expected["sha256"], "no change to inherited frozen sources")
        sources[name] = expected["sha256"]
    directory, names = regular(path).parent, set()
    for entry in directory.iterdir():
        if entry.name == "pytest-temp":
            require(entry.is_dir() and not entry.is_symlink(), "genuine excluded pytest fixture directory")
        else:
            require(entry.is_file() and not entry.is_symlink(), "regular qualification file")
            names.add(entry.name)
    require(names == set(engineering["files"]) | {"receipt.json"}, "exact TEST qualification closure")
    for name, pin in engineering["files"].items():
        require(Path(name).name == name and descriptor(directory / name) == pin, "TEST qualification log pin")
    for command in engineering["commands"]:
        require(command["log"] in engineering["files"] and {k: command[k] for k in ("sha256", "bytes")}
                == engineering["files"][command["log"]], "qualified TEST command log identity")


def authenticate_dev_audit(inputs, training, sources, producer, auditor):
    receipt = read(inputs["dev_audit_receipt"]["path"])
    expected_inputs = {name: inputs[role] for name, role in
                       (("plan", "training_plan"), ("worker", "training_receipt"), ("terminal", "training_terminal"))}
    require(receipt["version"] == auditor.VERSION and receipt["status"] == "completed"
            and receipt["complete"] is receipt["agreement"] is True and receipt["pending"] is None
            and receipt["sources"] == sources and receipt["inputs"] == expected_inputs
            and receipt["source_plan_sha256"] == inputs["training_plan"]["sha256"]
            and receipt["limits"] == auditor.LIMITS and receipt["peak_rss_bytes"] <= auditor.LIMITS["rss_bytes"]
            and receipt["requires_successful_original_supervisor"] is True
            and all(receipt[name] == 0 for name in
                    ("model_calls", "optimizer_calls", "teacher_calls", "simulator_calls", "test_array_decodes")),
            "complete independent DEV audit with no TEST decode")
    parent = read(inputs["dev_audit_terminal"]["path"])
    require(parent["status"] == "completed" and type(parent["returncode"]) is int and parent["returncode"] == 0
            and parent["timed_out"] is False and parent["group_absent"] is True
            and parent["cleanup"]["reaped"] is True and parent["cleanup"]["group_absent"] is True
            and parent["cleanup"]["errors"] == [] and parent["timing_available"] is True
            and parent["error"] is parent["clock_error"] is None and parent["cap_seconds"] == 600
            and Path(parent["cwd"]) == ROOT and parent["pid"] == parent["pgid"] != parent["parent_pid"]
            and parent["clock_source_sha256"] == CLOCK_PIN and parent["watchdog_sha256"] == SUPERVISOR_PIN
            and parent["deadline_ns"] == parent["started_ns"] + 600 * 10**9
            and parent["started_ns"] <= receipt["started_ns"] < receipt["finished_ns"]
            <= parent["finished_ns"] <= parent["deadline_ns"], "original successful DEV audit supervisor")
    command = list(parent["command"])
    if command[1:2] == ["-u"]:
        command.pop(1)
    require(command[:2] == [str(ROOT / ".venv/bin/python"), str(ROOT / DEV_AUDITOR)]
            and len(command) == 18 and len(set(command[2::2])) == 8, "original DEV audit command")
    options = dict(zip(command[2::2], command[3::2], strict=True))
    require(set(options) == {"--plan", "--plan-sha256", "--worker", "--worker-sha256", "--terminal",
                              "--terminal-sha256", "--supervision", "--output"}, "exact DEV audit options")
    for name, pin in expected_inputs.items():
        require(options["--" + name] == pin["path"] and options["--" + name + "-sha256"] == pin["sha256"],
                "DEV audit uses same closed training evidence")
    directory = producer.closed_files(inputs["dev_audit_receipt"]["path"], receipt, {"started.json", "audit.json"})
    require(options["--output"] == str(directory), "original DEV audit output")
    launch = read(options["--supervision"])
    require(descriptor(options["--supervision"])["sha256"] == receipt["supervision_sha256"]
            and all(parent[key] == value for key, value in launch.items()), "original DEV audit launch identity")
    started = read(directory / "started.json")
    require(started["version"] == auditor.VERSION and started["launch"] == launch
            and started["started_ns"] == receipt["started_ns"] and started["inputs"] == expected_inputs
            and started["authenticated_before_numerical_reads"] is True, "original DEV audit started identity")
    require(sum(pin["bytes"] for pin in receipt["files"].values()) <= auditor.LIMITS["output_bytes"], "DEV audit output bound")
    result = read(directory / "audit.json")
    require(result["version"] == auditor.VERSION and result["agreement"] is True and result["stage"] == "dev"
            and result["inputs"] == expected_inputs and result["test_evaluation_admitted"] is False
            and result["conditional_test_admission_requires_successful_original_audit_supervisor"] is True
            and all(value == 0 for value in result["zero_call_counts"].values()), "independent saved DEV audit result")
    expected_gate = auditor.independent_gate(result["metrics"], technical_complete=True)
    require(result["gate"] == expected_gate and expected_gate["passed"] is True
            and len(expected_gate["conditions"]) == 13 and all(row["passed"] is True for row in expected_gate["conditions"]),
            "all13 independently audited DEV conditions must pass")
    require(result["producer_gate"] == training["summary"]["gate"]
            == auditor.independent_gate(result["metrics"], technical_complete=False), "audited DEV gate joins producer")
    return {"receipt": receipt, "result": result, "run": directory, "terminal": parent}


def authenticate_inputs(inputs):
    """Metadata-only complete training, collection and DEV-audit admission."""
    require(set(inputs) == set(ROLES), "exact conditional TEST input roles")
    for record in inputs.values():
        require(set(record) == {"path", "sha256", "bytes"} and descriptor(record["path"])
                == {key: record[key] for key in ("sha256", "bytes")}, "external TEST input pin")
    plan = read(inputs["training_plan"]["path"])
    require(plan["version"] == "otto-query-memory-training-v1" and plan["status"] == "frozen_before_fitting"
            and {PRODUCER, DEV_AUDITOR, CLOCK, SUPERVISOR} <= set(plan["sources"]), "frozen original TRAIN/DEV sources")
    for name, pin in plan["sources"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and descriptor(name)["sha256"] == pin, "immutable original training source")
    producer = load(ROOT / PRODUCER, "_test_query_memory_training_metadata")
    auditor = load(ROOT / DEV_AUDITOR, "_test_query_memory_dev_audit_metadata")
    require(plan["configuration"] == producer.CONFIG and plan["limits"] == producer.LIMITS
            and set(plan["payloads"]) == producer.PAYLOADS and plan["runtime"] == producer.runtime_record(),
            "same frozen training recipe and runtime")
    collection_plan, collection_receipt, collection_run, sources, capacity = producer.authenticate_inputs(plan["inputs"])
    require(sources == plan["sources"] and capacity["projected_seconds"] == plan["capacity_projected_seconds"],
            "complete inherited collection and capacity closure")
    worker = read(inputs["training_receipt"]["path"])
    require(worker["version"] == producer.VERSION and worker["status"] == "completed" and worker["complete"] is True
            and worker["plan_sha256"] == inputs["training_plan"]["sha256"] and worker["sources"] == sources
            and worker["inputs"] == plan["inputs"] and worker["limits"] == producer.LIMITS
            and worker["fits_completed"] == 18 and worker["optimizer_steps"] == 7560 and worker["episode_exposures"] == 45360
            and worker["teacher_calls"] == worker["native_calls"] == worker["test_array_decodes"] == 0
            and worker["technical_complete"] is False and worker["pending"] is worker["pending_emission"] is None
            and worker["requires_successful_original_supervisor"] is worker["requires_independent_saved_audit"] is True
            and worker["peak_rss_bytes"] <= producer.LIMITS["rss_bytes"], "closed complete TRAIN/DEV producer")
    producer.successful_process(inputs, "training", worker, PRODUCER, ".venv/bin/python", 21600)
    directory = producer.closed_files(inputs["training_receipt"]["path"], worker, producer.PAYLOADS)
    require(sum(pin["bytes"] for pin in worker["files"].values()) <= producer.LIMITS["output_bytes"], "training output bound")
    # Qualified metadata checks include every final checkpoint and the durable
    # all-model/TRAIN barrier. They do not load any numerical arrays.
    metadata = auditor.Audit(argparse.Namespace(output=directory))
    metadata.check = lambda: None
    metadata.run, metadata.worker, metadata.plan = directory, worker, plan
    metadata.authenticate_metadata()
    training = {"plan": plan, "receipt": worker, "run": directory, "fits": metadata.fits, "summary": metadata.summary}
    dev = authenticate_dev_audit(inputs, training, sources, producer, auditor)
    authenticate_engineering(inputs["engineering"]["path"], sources)
    return {"training": training, "collection": {"plan": collection_plan, "receipt": collection_receipt, "run": collection_run},
            "dev_audit": dev, "sources": sources, "producer": producer, "dev_auditor": auditor}


def freeze(args):
    inputs = {}
    for role in ROLES:
        path = regular(getattr(args, role))
        pin = descriptor(path)
        require(pin["sha256"] == getattr(args, role + "_sha256"), "external planning input pin")
        inputs[role] = {"path": str(path), **pin}
    context = authenticate_inputs(inputs)
    write(args.output, {"version": VERSION, "status": "frozen_before_TEST", "configuration": CONFIG,
        "limits": LIMITS, "sources": context["sources"], "inputs": inputs,
        "runtime": context["producer"].runtime_record(), "payloads": sorted(PAYLOADS)})
    print(json.dumps({"status": "frozen_before_TEST", "plan": descriptor(args.output)}), flush=True)


def load_view(np, torch, models, producer, arrays, view, seed, period):
    require(view in VIEWS and seed in SEEDS and period in PERIODS, "declared final TEST view")
    slow = {name.removeprefix("slow."): torch.tensor(value.copy()) for name, value in arrays.items() if name.startswith("slow.")}
    projection = None if "projection.weight" not in arrays else torch.tensor(arrays["projection.weight"].copy())
    config = models.memory.Config(mode=producer.mode_for(view), key_dim=8)
    model = models.from_states(config, seed, period, slow, projection, slow_mode="frozen")
    model.eval()
    require(model.slow.mode == "frozen" and all(p.requires_grad == name.startswith("action_residual.")
            for name, p in model.slow.named_parameters()) and (model.projection is None or model.projection.weight.requires_grad),
            "canonical TEST residual/projection flags")
    for name, value in model.state_dict().items():
        require(value.detach().numpy().tobytes() == arrays[name].tobytes(), "exact saved checkpoint bytes in TEST model")
    return model


class Run:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.clock = self.launch = self.plan = None
        self.authenticated, self.sequence = False, 0
        self.receipt = {"version": VERSION, "status": "started", "complete": False, "technical_complete": False,
            "requires_independent_saved_audit": True, "requires_successful_original_supervisor": True,
            "pending": None, "pending_emission": None, "views_completed": 0, "model_calls": 0,
            "checkpoint_decodes": 0, "test_array_decodes": 0, **dict.fromkeys(ZERO_COUNTS, 0)}

    def check(self):
        require(self.clock.now_ns() < self.launch["deadline_ns"], "original1800-second TEST deadline")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.receipt["peak_rss_bytes"] = rss
        require(rss <= LIMITS["rss_bytes"] and sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())
                < LIMITS["output_bytes"] - 1024**2, "TEST RSS/output bound with failure reserve")

    def publish(self, name, value):
        require(name in PAYLOADS, "declared TEST JSON payload")
        self.check()
        self.receipt["pending_emission"] = {"file": name}
        write(self.out / name, value)
        self.receipt["pending_emission"] = None

    def event(self, value):
        self.check()
        self.receipt["pending_emission"] = {"file": "work.jsonl", "record": value}
        with (self.out / "work.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.receipt["pending_emission"] = None

    def npz(self, name, arrays):
        require(name in PAYLOADS, "declared TEST numerical payload")
        self.check()
        self.receipt["pending_emission"] = {"file": name}
        with (self.out / name).open("xb") as stream:
            self.np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with self.np.load(self.out / name, allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays) and all(saved[k].dtype == arrays[k].dtype
                    and saved[k].shape == arrays[k].shape and saved[k].tobytes() == arrays[k].tobytes()
                    for k in arrays), "TEST numerical byte roundtrip")
        self.receipt["pending_emission"] = None
        return descriptor(self.out / name)

    def admit(self):
        require(descriptor(CLOCK)["sha256"] == CLOCK_PIN and descriptor(SUPERVISOR)["sha256"] == SUPERVISOR_PIN,
                "qualified original TEST clock/supervisor")
        self.clock = load(ROOT / CLOCK, "_query_memory_test_clock").SuspendClock()
        self.started = self.clock.now_ns()
        while not self.args.supervision.exists():
            require(self.clock.now_ns() - self.started < 5 * 10**9, "original TEST launch available")
            time.sleep(.01)
        self.launch = read(self.args.supervision)
        command = list(self.launch["command"])
        if command[1:2] == ["-u"]:
            command.pop(1)
        require(command == [sys.executable, *sys.argv] and self.launch["pid"] == os.getpid()
                and self.launch["pgid"] == os.getpgrp() and self.launch["parent_pid"] == os.getppid() != os.getpid()
                and Path(self.launch["cwd"]) == Path.cwd() == ROOT and self.launch["cap_seconds"] == LIMITS["seconds"]
                and self.launch["watchdog_sha256"] == SUPERVISOR_PIN and self.launch["clock_source_sha256"] == CLOCK_PIN
                and self.launch["clock_backend"] == self.clock.backend
                and self.launch["started_ns"] <= self.started < self.launch["deadline_ns"]
                and self.launch["deadline_ns"] == self.launch["started_ns"] + LIMITS["seconds"] * 10**9,
                "original actual bounded TEST process")
        self.check()
        require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256, "external TEST plan pin")
        self.plan = read(self.args.plan)
        require(self.plan["version"] == VERSION and self.plan["status"] == "frozen_before_TEST"
                and self.plan["configuration"] == CONFIG and self.plan["limits"] == LIMITS
                and set(self.plan["payloads"]) == PAYLOADS, "fixed conditional TEST plan")
        self.context = authenticate_inputs(self.plan["inputs"])
        self.producer = self.context["producer"]
        require(self.context["sources"] == self.plan["sources"] and self.plan["runtime"] == self.producer.runtime_record(),
                "unchanged complete TEST source and runtime closure")
        require(all(os.environ.get(name) == "1" for name in self.producer.THREADS), "one TEST numerical CPU thread")
        self.receipt.update(plan_sha256=self.args.plan_sha256, sources=self.plan["sources"], inputs=self.plan["inputs"],
                            limits=LIMITS, supervision_sha256=descriptor(self.args.supervision)["sha256"])
        self.publish("started.json", {"launch": self.launch, "started_ns": self.started,
            "request": {key: str(value) for key, value in vars(self.args).items()},
            "authenticated_before_numerical_reads": True})
        self.authenticated = True

    def decode_checkpoints(self):
        require(self.authenticated, "closed independent DEV admission before checkpoint decode")
        result, manifest = {}, []
        source = self.context["training"]
        for fit in source["fits"]:
            self.check()
            path = source["run"] / fit["checkpoint_path"]
            require(descriptor(path) == fit["checkpoint"] == source["receipt"]["files"][path.name], "final checkpoint pin before decode")
            self.receipt["pending"] = {"operation": "checkpoint_decode", "path": path.name}
            with self.np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            self.receipt["checkpoint_decodes"] += 1
            expected = {"slow." + k: v for k, v in self.context["dev_auditor"].SLOW_SHAPES.items()}
            if fit["family"] not in ("pretrained", "joint_aux"):
                expected["projection.weight"] = (8, 28)
            require(set(arrays) == set(expected) and all(a.dtype == self.np.float32 and a.shape == expected[name]
                    and bool(self.np.isfinite(a).all()) for name, a in arrays.items()), "all finite exact final checkpoint tensors")
            witness = self.context["dev_auditor"].tensor_witness(arrays)
            require(witness == fit["final"], "TEST checkpoint equals audited final fit witness")
            result[fit["seed"], fit["family"]] = arrays
            manifest.append({"family": fit["family"], "seed": fit["seed"], "checkpoint_path": fit["checkpoint_path"],
                "checkpoint": fit["checkpoint"], "final": witness, "final_slow": fit["final_slow"]})
            self.receipt["pending"] = None
        require(len(result) == self.receipt["checkpoint_decodes"] == 18, "all18 final fits retained")
        self.publish("checkpoint-manifest.json", {"training_receipt": self.plan["inputs"]["training_receipt"], "checkpoints": manifest})
        return result

    def decode_test(self):
        require(self.authenticated and self.receipt["checkpoint_decodes"] == 18
                and self.receipt["test_array_decodes"] == 0, "one TEST decode only after admitted complete checkpoint set")
        source = self.context["collection"]
        path = source["run"] / "test.npz"
        require(descriptor(path) == source["receipt"]["files"][path.name], "exact TEST census pin before decode")
        self.receipt["pending"] = {"operation": "test_decode"}
        with self.np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == self.producer.ARRAY_KEYS, "exact six TEST census arrays")
            flat = {name: archive[name] for name in archive.files}
        self.receipt["test_array_decodes"] += 1
        identities = [row for row in source["plan"]["cohort"] if row["stage"] == "test"]
        require(len(identities) == 36, "all36 untouched TEST paths")
        self.receipt["pending"] = None
        return flat, identities

    def predict(self, model, view, seed, period, history, identities):
        np, torch = self.np, self.torch
        total = int(history["episode_offsets"][-1])
        saved = {name: np.zeros((total, 4), np.float32) for name in PREDICTION_FIELDS if name not in ("prior_mask", "episode_offsets")}
        saved["prior_mask"] = np.zeros(total, np.bool_)
        saved["episode_offsets"] = history["episode_offsets"].copy()
        work, units, chunks, rows = {}, {}, 0, 0
        terms = {name: [] for name in ("total", "nonquery", "prior")}
        tick = self.clock.now_ns()
        with torch.no_grad():
            for first in range(0, history["episode_count"], 6):
                indices = list(range(first, min(first + 6, history["episode_count"])))
                lengths = [int(history["episode_offsets"][i + 1] - history["episode_offsets"][i]) for i in indices]
                carry = model.initial_carry(len(indices))
                for start in range(0, max(lengths), 32):
                    self.check()
                    packet = self.data.batch_chunk(history, indices, start)
                    inputs = {name: torch.from_numpy(value) for name, value in packet["model_inputs"].items()}
                    before = {"phase": "test", "view": view, "seed": seed, "query_period": period, "episode_indices": indices, "start": start}
                    self.sequence += 1
                    call_id = self.sequence
                    self.receipt["pending"] = {"call_id": call_id, **before}
                    self.event({"event": "attempt", "call_id": call_id, **before})
                    forecast = model(**inputs, carry=carry, no_write=view == "trace_no_write")
                    self.receipt["model_calls"] += 1
                    require(torch.equal(forecast.prior_mask, torch.from_numpy(packet["prior_mask"])), "canonical P-scheduled prior support")
                    loss = self.data.weighted_loss(forecast.action_prediction, forecast.corrected_shadow_prior,
                        torch.from_numpy(packet["targets"]), torch.from_numpy(packet["legal"]),
                        torch.from_numpy(packet["nonquery_weights"]), torch.from_numpy(packet["prior_weights"]),
                        inputs["query_mask"], forecast.prior_mask, episode_count=history["episode_count"])
                    for name, values in terms.items():
                        values.append(float(loss[name]) * len(indices) / history["episode_count"])
                    fields = {name: getattr(forecast, "prediction" if name == "base_prediction" else name)
                              for name in PREDICTION_FIELDS if name != "episode_offsets"}
                    for lane, index in enumerate(indices):
                        length = int(inputs["lengths"][lane])
                        low = int(history["episode_offsets"][index]) + start
                        for name, value in fields.items():
                            tail = value[lane, length:].detach().numpy()
                            require(tail.tobytes() == np.zeros_like(tail).tobytes(), "canonical positive-zero padding")
                            saved[name][low:low + length] = value[lane, :length].detach().numpy()
                    carry = self.models.detach_carry(forecast.carry)
                    self.producer.add_counts(work, forecast.work_counts)
                    self.producer.add_counts(units, forecast.memory_work_units)
                    chunk_rows = int(inputs["lengths"].sum())
                    rows += chunk_rows
                    chunks += 1
                    self.event({"event": "return", "call_id": call_id, **before, "rows": chunk_rows,
                                "work_counts": forecast.work_counts, "memory_work_units": forecast.memory_work_units})
                    self.receipt["pending"] = None
                require(carry.slow.base.ended.tolist() == [True] * len(indices)
                        and carry.slow.base.absolute_step.tolist() == lengths and carry.fast.absolute_step.tolist() == lengths,
                        "complete P-scheduled episode tails")
        require(rows == total and all(bool(np.isfinite(value).all()) for value in saved.values()), "finite complete TEST predictions")
        for name in ("action_prediction", "slow_action_prediction", "base_prediction"):
            require(saved[name][history["query_mask"]].tobytes() == history["query_scores"][history["query_mask"]].tobytes(),
                    "exact actual P-scheduled query outputs")
        require(np.array_equal(saved["prior_mask"], history["prior_mask"]), "complete saved TEST prior support")
        inference_seconds = (self.clock.now_ns() - tick) / 1e9
        tick = self.clock.now_ns()
        episodes = []
        for index, identity in enumerate(identities):
            low, high = map(int, history["episode_offsets"][index:index + 2])
            episodes.append(self.metrics.episode_metrics(identity, period, history["targets"][low:high],
                history["legal"][low:high], saved["action_prediction"][low:high],
                prequery_forecast=saved["corrected_shadow_prior"][low:high]))
        report = self.metrics.aggregate_episodes(episodes, family=view, fit_seed=seed, expected_identities=identities)
        return saved, {"view": view, "seed": seed, "stage": "test", "query_period": period,
            "parent_fit": self.producer.view_parent(view), "work_counts": work, "memory_work_units": units,
            "chunks": chunks, "rows": rows, "loss": {name: math.fsum(values) for name, values in terms.items()},
            "metrics": report, "inference_and_loss_seconds": inference_seconds,
            "metric_seconds": (self.clock.now_ns() - tick) / 1e9,
            "evaluation": {"slow_mode": "frozen", "grad_enabled": False, "residual_requires_grad": True,
                           "projection_requires_grad": model.projection is not None}}

    def evaluate(self, checkpoints, flat, identities):
        records = []
        fits = {(row["seed"], row["family"]): row for row in self.context["training"]["fits"]}
        for period in PERIODS:
            history = self.data.project_census(flat, identities, query_period=period, expected_stage="test")
            self.npz(f"test-P{period}-history.npz", {k: v for k, v in history.items() if isinstance(v, self.np.ndarray)})
            self.publish(f"test-P{period}-history.json", {k: v for k, v in history.items() if not isinstance(v, self.np.ndarray)})
            for seed in SEEDS:
                reference = None
                for view in VIEWS:
                    parent = self.producer.view_parent(view)
                    arrays, fit = checkpoints[seed, parent], fits[seed, parent]
                    tick = self.clock.now_ns()
                    model = load_view(self.np, self.torch, self.models, self.producer, arrays, view, seed, period)
                    clone_seconds = (self.clock.now_ns() - tick) / 1e9
                    before = self.producer.state_witness(model)
                    require(before == fit["final"], "canonical TEST view retains same final weights")
                    saved, row = self.predict(model, view, seed, period, history, identities)
                    after = self.producer.state_witness(model)
                    require(after == before, "TEST inference cannot modify any final parameter")
                    if view == "pretrained":
                        reference = saved
                    elif view != "joint_aux":
                        row["frozen_predictions_equal_pretrained"] = self.producer.same_frozen_predictions(
                            saved, reference, no_write=view == "trace_no_write")
                    name = f"test-P{period}-prediction-{view}-{seed}.npz"
                    tick = self.clock.now_ns()
                    pin = self.npz(name, saved)
                    records.append({**row, "clone_seconds": clone_seconds, "prediction_path": name, "prediction": pin,
                        "serialization_seconds": (self.clock.now_ns() - tick) / 1e9, "model_before": before, "model_after": after,
                        "checkpoint_path": fit["checkpoint_path"], "checkpoint": fit["checkpoint"]})
                    self.receipt["views_completed"] += 1
        self.metrics.validate_matched_reports([r["metrics"] for r in records], families=VIEWS,
            fit_seeds=SEEDS, query_periods=PERIODS, expected_identities=identities)
        self.publish("test-views.json", {"stage": "test", "views": records, "technical_complete": False})
        return records

    def body(self):
        require(self.authenticated, "full conditional admission before numerical imports")
        tick = self.clock.now_ns()
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(ROOT / "src"))
        from openjev.research import otto_query_memory_data as data
        from openjev.research import otto_query_memory_gate as gate
        from openjev.research import otto_query_memory_metrics as metrics
        from openjev.research import otto_query_memory_model as models

        self.np, self.torch, self.data, self.metrics, self.models = np, torch, data, metrics, models
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "fixed TEST numerical thread counts")
        self.publish("runtime.json", {**self.producer.runtime_record(), "numpy": np.__version__, "torch": torch.__version__,
            "torch_threads": 1, "interop_threads": 1, "deterministic": True, "cuda_used": False, "mps_used": False})
        checkpoints = self.decode_checkpoints()
        flat, identities = self.decode_test()
        setup_seconds = (self.clock.now_ns() - tick) / 1e9
        views = self.evaluate(checkpoints, flat, identities)
        prospective_gate = gate.continuation_gate([r["metrics"] for r in views], identities, technical_complete=False)
        self.publish("summary.json", {"version": VERSION, "configuration": CONFIG, "views": views,
            "technical_complete": False, "gate": prospective_gate, "setup_seconds": setup_seconds,
            "gate_status": "pending independent TEST saved-output audit and original producer closure",
            "test_array_decodes": self.receipt["test_array_decodes"], "checkpoint_decodes": self.receipt["checkpoint_decodes"],
            "views_completed": self.receipt["views_completed"], "model_calls": self.receipt["model_calls"],
            **{name: self.receipt[name] for name in ZERO_COUNTS},
            "cost_scope": "all48 canonical TEST views plus admission, serialization and gate; prior training and full census costs remain separate",
            "scope": CONFIG["scope"]})

    def execute(self):
        self.out.mkdir(parents=False, exist_ok=False)
        def interrupt(_signum, _frame):
            raise InterruptedError("original TEST supervisor stopped worker")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            self.admit()
            self.body()
            for name, pin in self.plan["sources"].items():
                self.check()
                require(descriptor(name)["sha256"] == pin, "unchanged final TEST source")
            for pin in self.plan["inputs"].values():
                require(descriptor(pin["path"]) == {k: pin[k] for k in ("sha256", "bytes")}, "unchanged final TEST input")
            for source in (self.context["training"], self.context["collection"], self.context["dev_audit"]):
                for name, pin in source["receipt"]["files"].items():
                    self.check()
                    require(descriptor(source["run"] / name) == pin, "unchanged inherited payload")
            require(descriptor(self.args.plan)["sha256"] == self.args.plan_sha256
                    and descriptor(self.args.supervision)["sha256"] == self.receipt["supervision_sha256"], "unchanged TEST plan/launch")
            require(self.receipt["views_completed"] == 48 and self.receipt["checkpoint_decodes"] == 18
                    and self.receipt["test_array_decodes"] == 1 and all(self.receipt[name] == 0 for name in ZERO_COUNTS)
                    and self.receipt["pending"] is self.receipt["pending_emission"] is None, "complete frozen TEST producer")
            require({p.name for p in self.out.iterdir()} == PAYLOADS, "exact58 TEST payloads")
            files = {name: descriptor(self.out / name) for name in sorted(PAYLOADS)}
            self.check()
            finished = self.clock.now_ns()
            self.receipt.update(status="completed", complete=True, files=files, started_ns=self.started, finished_ns=finished,
                                wall_seconds=(finished - self.started) / 1e9)
            write(self.out / "receipt.json", self.receipt)
            self.check()
            print(json.dumps({"status": "completed", "receipt": descriptor(self.out / "receipt.json")}), flush=True)
        except BaseException as error:
            self.receipt.update(status="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
            try:
                if (self.out / "receipt.json").exists():
                    (self.out / "receipt.json").rename(self.out / "receipt.invalid.json")
                self.receipt["files"] = {p.name: descriptor(p) for p in self.out.iterdir() if p.is_file()}
                write(self.out / "receipt.json", self.receipt)
            except BaseException as secondary:  # noqa: BLE001 - retain original failure
                error.add_note("Failure receipt also failed: " + repr(secondary))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for role in ROLES:
        plan.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
        plan.add_argument("--" + role.replace("_", "-") + "-sha256", required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "supervision", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    require(Path(sys.executable).absolute() == ROOT / ".venv/bin/python" and Path.cwd() == ROOT, "original CPU interpreter/checkout")
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT) and ".." not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), "contained exclusive absolute output")
    if args.mode == "plan":
        freeze(args)
    else:
        require(args.plan.is_absolute() and args.supervision.is_absolute(), "absolute TEST execution inputs")
        Run(args).execute()


if __name__ == "__main__":
    main()
